using System.Text.Json;

using Cove.Core.Auth;
using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Review;

// Write side: record verdicts, delete applied false positives from Cove (with undo), and hand marks
// to the exporter.
internal sealed class AiReviewMarkService(
    IServiceScopeFactory scopeFactory,
    AiReviewSettingsProvider settingsProvider,
    AiReviewExportService exportService)
{
    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;
    private readonly AiReviewSettingsProvider _settingsProvider = settingsProvider;
    private readonly AiReviewExportService _exportService = exportService;

    public async Task<AiReviewQueueItem?> SetVerdictAsync(AiReviewSetVerdictRequest request, CovePrincipal? principal, CancellationToken ct)
    {
        if (!AiReviewVerdicts.IsValid(request.Verdict))
        {
            throw new ArgumentException("Verdict must be 'fp' or 'tp'.");
        }

        var settings = await _settingsProvider.GetAsync(ct);
        int markId;
        int videoId;
        bool deleteNow;

        using (var scope = _scopeFactory.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<DbContext>();
            var segment = await db.Set<Segment>().AsNoTracking().FirstOrDefaultAsync(s => s.Id == request.SegmentId, ct);
            if (segment is null || segment.HostType != SegmentHostType.Video || segment.TagId is null)
            {
                return null;
            }

            var mark = await db.Set<ExtAiReviewMarkEntity>().FirstOrDefaultAsync(m => m.SegmentId == request.SegmentId, ct);
            var now = DateTime.UtcNow;
            var verdictChanged = mark is null || mark.Verdict != request.Verdict;
            if (mark is null)
            {
                var tagName = await db.Set<Tag>().AsNoTracking().Where(t => t.Id == segment.TagId).Select(t => t.Name).FirstOrDefaultAsync(ct)
                              ?? segment.Title ?? $"Tag {segment.TagId}";
                var observations = AiReviewFramePlanner.ReadObservationCount(segment.Payload);
                mark = new ExtAiReviewMarkEntity
                {
                    VideoId = segment.HostId,
                    SegmentId = segment.Id,
                    TagId = segment.TagId.Value,
                    TagName = tagName,
                    ModelKey = AiReviewFramePlanner.ReadModelKey(segment.Payload),
                    SourceKey = segment.SourceKey,
                    SourceRunId = segment.SourceRunId,
                    PayloadJson = segment.Payload?.RootElement.GetRawText(),
                    StartSec = segment.StartSec,
                    EndSec = segment.EndSec ?? segment.StartSec,
                    Confidence = segment.Confidence,
                    FrameTimesJson = JsonSerializer.Serialize(AiReviewFramePlanner.Plan(segment.StartSec, segment.EndSec, observations, settings.MaxFramesPerSegment)),
                    CreatedAt = now,
                };
                db.Set<ExtAiReviewMarkEntity>().Add(mark);
            }

            mark.Verdict = request.Verdict;
            mark.Note = string.IsNullOrWhiteSpace(request.Note) ? mark.Note : request.Note.Trim();
            mark.ReviewerUserId = principal?.UserId;
            mark.ReviewerName = principal?.Username;
            mark.UpdatedAt = now;
            if (verdictChanged)
            {
                // Previously exported frames (if any) sit in the other verdict's folder; the exporter moves them.
                mark.ExportStatus = AiReviewExportStatus.Pending;
                mark.ExportError = null;
            }

            await db.SaveChangesAsync(ct);
            markId = mark.Id;
            videoId = mark.VideoId;
            deleteNow = settings.DeleteOnMark && request.Verdict == AiReviewVerdicts.FalsePositive;
        }

        if (deleteNow)
        {
            await DeleteMarkedSegmentsAsync([markId], settings, ct);
        }

        _exportService.Enqueue(markId);
        return await LoadItemAsync(markId, settings, ct);
    }

    public async Task<bool> ClearVerdictAsync(int markId, CancellationToken ct)
    {
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();
        var mark = await db.Set<ExtAiReviewMarkEntity>().FirstOrDefaultAsync(m => m.Id == markId, ct);
        if (mark is null)
        {
            return false;
        }

        if (mark.SegmentDeletedAt is not null)
        {
            throw new InvalidOperationException("The segment was already deleted from Cove. Restore it first, then clear the verdict.");
        }

        AiReviewExportService.DeleteExportedFiles(AiReviewQueryService.ReadStringList(mark.ExportedFilesJson));
        db.Set<ExtAiReviewMarkEntity>().Remove(mark);
        await db.SaveChangesAsync(ct);
        _exportService.RequestManifestRewrite();
        return true;
    }

    // Deletes the Cove segments of every pending false positive in scope. Returns what was removed.
    public async Task<AiReviewApplyResult> ApplyAsync(AiReviewApplyRequest request, CancellationToken ct)
    {
        var settings = await _settingsProvider.GetAsync(ct);
        List<int> markIds;
        using (var scope = _scopeFactory.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<DbContext>();
            var query = db.Set<ExtAiReviewMarkEntity>().AsNoTracking()
                .Where(m => m.Verdict == AiReviewVerdicts.FalsePositive && m.SegmentDeletedAt == null && m.SegmentId != null);
            if (request.VideoId is { } vid)
            {
                query = query.Where(m => m.VideoId == vid);
            }

            if (request.TagId is { } tagId)
            {
                query = query.Where(m => m.TagId == tagId);
            }

            markIds = await query.Select(m => m.Id).ToListAsync(ct);
        }

        return await DeleteMarkedSegmentsAsync(markIds, settings, ct);
    }

    public async Task<AiReviewQueueItem?> RestoreAsync(int markId, CancellationToken ct)
    {
        var settings = await _settingsProvider.GetAsync(ct);
        using (var scope = _scopeFactory.CreateScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<DbContext>();
            var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();
            var invalidator = scope.ServiceProvider.GetService<ISegmentSpanCacheInvalidator>();

            var mark = await db.Set<ExtAiReviewMarkEntity>().FirstOrDefaultAsync(m => m.Id == markId, ct);
            if (mark is null)
            {
                return null;
            }

            if (mark.SegmentDeletedAt is null)
            {
                return await LoadItemAsync(markId, settings, ct);
            }

            var segment = new Segment
            {
                HostType = SegmentHostType.Video,
                HostId = mark.VideoId,
                StartSec = mark.StartSec,
                EndSec = mark.EndSec,
                TagId = mark.TagId,
                Kind = "tag",
                Payload = string.IsNullOrWhiteSpace(mark.PayloadJson) ? null : JsonDocument.Parse(mark.PayloadJson),
                SourceKey = mark.SourceKey,
                SourceRunId = mark.SourceRunId,
                Confidence = mark.Confidence is { } c ? (float)c : null,
                Title = mark.TagName,
            };
            segmentRepo.Add(segment);
            await segmentRepo.SaveChangesAsync(ct);

            mark.SegmentId = segment.Id;
            mark.SegmentDeletedAt = null;
            mark.UpdatedAt = DateTime.UtcNow;
            await db.SaveChangesAsync(ct);
            invalidator?.InvalidateVideo(mark.VideoId);
        }

        return await LoadItemAsync(markId, settings, ct);
    }

    private async Task<AiReviewApplyResult> DeleteMarkedSegmentsAsync(IReadOnlyList<int> markIds, AiReviewSettings settings, CancellationToken ct)
    {
        if (markIds.Count == 0)
        {
            return new AiReviewApplyResult(0, 0, 0);
        }

        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();
        var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();
        var tagAppRepo = scope.ServiceProvider.GetService<ITagApplicationRepository>();
        var invalidator = scope.ServiceProvider.GetService<ISegmentSpanCacheInvalidator>();

        var marks = await db.Set<ExtAiReviewMarkEntity>()
            .Where(m => markIds.Contains(m.Id) && m.SegmentDeletedAt == null && m.SegmentId != null)
            .ToListAsync(ct);
        if (marks.Count == 0)
        {
            return new AiReviewApplyResult(0, 0, 0);
        }

        var segmentIds = marks.Select(m => m.SegmentId!.Value).ToList();
        var segments = await db.Set<Segment>().Where(s => segmentIds.Contains(s.Id)).ToListAsync(ct);
        if (segments.Count > 0)
        {
            segmentRepo.RemoveRange(segments);
        }

        var now = DateTime.UtcNow;
        foreach (var mark in marks)
        {
            mark.SegmentDeletedAt = now;
            mark.UpdatedAt = now;
        }

        await db.SaveChangesAsync(ct);

        var affectedVideos = marks.Select(m => m.VideoId).Distinct().ToList();
        var removedApplications = 0;
        if (settings.RemoveTagFromVideoWhenEmpty && tagAppRepo is not null)
        {
            foreach (var videoId in affectedVideos)
            {
                var tagIds = marks.Where(m => m.VideoId == videoId).Select(m => m.TagId).Distinct().ToList();
                var remaining = await db.Set<Segment>().AsNoTracking()
                    .Where(s => s.HostType == SegmentHostType.Video && s.HostId == videoId && s.SourceKey == settings.SourceKey && s.TagId != null && tagIds.Contains(s.TagId.Value))
                    .Select(s => s.TagId!.Value)
                    .Distinct()
                    .ToListAsync(ct);
                var emptyTags = tagIds.Except(remaining).ToList();
                if (emptyTags.Count == 0)
                {
                    continue;
                }

                var applications = await tagAppRepo.FindAsync(new TagApplicationFilter
                {
                    HostType = AffinityHostType.Video,
                    HostId = videoId,
                    SourceKey = settings.SourceKey,
                }, ct);
                var toRemove = applications.Where(a => emptyTags.Contains(a.TagId)).ToList();
                if (toRemove.Count == 0)
                {
                    continue;
                }

                tagAppRepo.RemoveRange(toRemove);
                await tagAppRepo.RemoveOrphanedTagLinksAsync(AffinityHostType.Video, [videoId], settings.SourceKey, ct);
                removedApplications += toRemove.Count;
            }

            if (removedApplications > 0)
            {
                await tagAppRepo.SaveChangesAsync(ct);
            }
        }

        if (invalidator is not null)
        {
            foreach (var videoId in affectedVideos)
            {
                invalidator.InvalidateVideo(videoId);
            }
        }

        return new AiReviewApplyResult(segments.Count, affectedVideos.Count, removedApplications);
    }

    private async Task<AiReviewQueueItem?> LoadItemAsync(int markId, AiReviewSettings settings, CancellationToken ct)
    {
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();
        var mark = await db.Set<ExtAiReviewMarkEntity>().AsNoTracking().FirstOrDefaultAsync(m => m.Id == markId, ct);
        if (mark is null)
        {
            return null;
        }

        var title = await db.Set<Video>().AsNoTracking().Where(v => v.Id == mark.VideoId).Select(v => v.Title).FirstOrDefaultAsync(ct) ?? string.Empty;
        if (mark.SegmentId is { } segmentId)
        {
            var segment = await db.Set<Segment>().AsNoTracking().FirstOrDefaultAsync(s => s.Id == segmentId, ct);
            if (segment is not null)
            {
                return AiReviewQueryService.ToItem(segment, mark, mark.TagName, title);
            }
        }

        return AiReviewQueryService.DeletedItem(mark, title, settings.SourceKey);
    }
}
