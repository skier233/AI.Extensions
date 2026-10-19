using System.Text.Json;

using Cove.Core.DTOs;
using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Review;

// Read side: which tags have AI segments (and how many are still unreviewed), and the review queue for
// one tag, either across the library or inside one video. The queue is Cove's own segment library
// query (ISegmentRepository.FindAsync) scoped to the tag, so every filter, sort and the random seed
// behave exactly as on the Segments page; the extension only overlays its verdict marks.
internal sealed class AiReviewQueryService(IServiceScopeFactory scopeFactory, AiReviewSettingsProvider settingsProvider)
{
    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;
    private readonly AiReviewSettingsProvider _settingsProvider = settingsProvider;

    public async Task<IReadOnlyList<AiReviewTagSummary>> GetTagsAsync(int? videoId, CancellationToken ct)
    {
        var settings = await _settingsProvider.GetAsync(ct);
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();

        // The segments table is large (millions of AI rows in a big library), so it is only touched with
        // a plain GROUP BY; the (small) marks table is aggregated separately and merged in memory.
        var segments = BaseSegments(db, settings.SourceKey, videoId);
        var segmentStats = await segments
            .GroupBy(s => s.TagId!.Value)
            .Select(g => new { TagId = g.Key, Total = g.Count() })
            .ToListAsync(ct);

        var marksQuery = db.Set<ExtAiReviewMarkEntity>().AsNoTracking().Where(m => m.SourceKey == settings.SourceKey);
        if (videoId is { } vid)
        {
            marksQuery = marksQuery.Where(m => m.VideoId == vid);
        }

        var markStats = await marksQuery
            .GroupBy(m => new { m.TagId, m.Verdict, Deleted = m.SegmentDeletedAt != null })
            .Select(g => new { g.Key.TagId, g.Key.Verdict, g.Key.Deleted, Count = g.Count() })
            .ToListAsync(ct);

        var tagIds = segmentStats.Select(s => s.TagId).Concat(markStats.Select(d => d.TagId)).Distinct().ToList();
        if (tagIds.Count == 0)
        {
            return [];
        }

        var tagNames = await db.Set<Tag>().AsNoTracking()
            .Where(t => tagIds.Contains(t.Id))
            .Select(t => new { t.Id, t.Name })
            .ToDictionaryAsync(t => t.Id, t => t.Name, ct);

        var totals = segmentStats.ToDictionary(s => s.TagId, s => s.Total);
        var result = new List<AiReviewTagSummary>();
        foreach (var tagId in tagIds)
        {
            var total = totals.GetValueOrDefault(tagId);
            var marksForTag = markStats.Where(m => m.TagId == tagId).ToList();
            var fpLive = marksForTag.Where(m => m.Verdict == AiReviewVerdicts.FalsePositive && !m.Deleted).Sum(m => m.Count);
            var fpDeleted = marksForTag.Where(m => m.Verdict == AiReviewVerdicts.FalsePositive && m.Deleted).Sum(m => m.Count);
            var tpLive = marksForTag.Where(m => m.Verdict == AiReviewVerdicts.TruePositive && !m.Deleted).Sum(m => m.Count);
            result.Add(new AiReviewTagSummary(
                tagId,
                tagNames.GetValueOrDefault(tagId, $"Tag {tagId}"),
                total,
                Math.Max(0, total - fpLive - tpLive),
                fpLive + fpDeleted,
                tpLive,
                fpLive));
        }

        return result.OrderBy(t => t.TagName, StringComparer.OrdinalIgnoreCase).ToList();
    }

    public async Task<int> CountUnreviewedAsync(int videoId, CancellationToken ct)
    {
        var settings = await _settingsProvider.GetAsync(ct);
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();
        var segments = BaseSegments(db, settings.SourceKey, videoId);
        var markedSegmentIds = db.Set<ExtAiReviewMarkEntity>().AsNoTracking().Where(m => m.SegmentId != null).Select(m => m.SegmentId!.Value);
        return await segments.CountAsync(s => !markedSegmentIds.Contains(s.Id), ct);
    }

    public async Task<AiReviewQueuePage> GetQueueAsync(AiReviewQueueRequest request, CancellationToken ct)
    {
        var settings = await _settingsProvider.GetAsync(ct);
        var findFilter = request.FindFilter ?? new FindFilter();
        findFilter.Page = Math.Max(1, findFilter.Page);
        findFilter.PerPage = Math.Clamp(findFilter.PerPage <= 0 ? 48 : findFilter.PerPage, 1, 250);
        var verdict = (request.Filter ?? AiReviewQueueFilters.Unreviewed).Trim().ToLowerInvariant();

        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();
        var segments = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();

        if (verdict == AiReviewQueueFilters.Deleted)
        {
            return await QueryDeletedMarksAsync(db, request, settings, findFilter, ct);
        }

        // The marks of this tag are the only thing the extension adds to the segment library query: the
        // unreviewed view leaves them out, the verdict views keep only them, and the "all" view decorates.
        var marks = db.Set<ExtAiReviewMarkEntity>().AsNoTracking()
            .Where(m => m.TagId == request.TagId && m.SourceKey == settings.SourceKey && m.SegmentId != null && m.SegmentDeletedAt == null);
        if (request.VideoId is { } scopedVideoId)
        {
            marks = marks.Where(m => m.VideoId == scopedVideoId);
        }
        if (NormalizeExportStatus(request.ExportStatus) is { } exportStatus)
        {
            marks = marks.Where(m => m.ExportStatus == exportStatus);
        }
        if (verdict is AiReviewQueueFilters.FalsePositive or AiReviewQueueFilters.TruePositive)
        {
            marks = marks.Where(m => m.Verdict == verdict);
        }

        var markedSegmentIds = await marks.Select(m => m.SegmentId!.Value).Distinct().ToListAsync(ct);

        var filter = ScopeToQueue(request.ObjectFilter ?? new SegmentFilter(), request, settings);

        switch (verdict)
        {
            case AiReviewQueueFilters.Unreviewed:
                filter.ExcludeIds = markedSegmentIds.Count > 0 ? markedSegmentIds : null;
                break;
            case AiReviewQueueFilters.FalsePositive:
            case AiReviewQueueFilters.TruePositive:
                if (markedSegmentIds.Count == 0)
                {
                    return new AiReviewQueuePage([], 0, findFilter.Page, findFilter.PerPage);
                }

                filter.Ids = filter.Ids is { Count: > 0 } requestedIds ? requestedIds.Intersect(markedSegmentIds).ToList() : markedSegmentIds;
                if (filter.Ids.Count == 0)
                {
                    return new AiReviewQueuePage([], 0, findFilter.Page, findFilter.PerPage);
                }

                break;
            default:
                if (request.ExportStatus is not null)
                {
                    // An export-status filter only has meaning for reviewed segments.
                    filter.Ids = markedSegmentIds.Count > 0 ? markedSegmentIds : [-1];
                }

                break;
        }

        var (rows, total) = await segments.FindAsync(filter, findFilter, ct);

        var pageSegmentIds = rows.Select(row => row.Segment.Id).ToList();
        var pageMarks = pageSegmentIds.Count == 0
            ? new Dictionary<int, ExtAiReviewMarkEntity>()
            : await db.Set<ExtAiReviewMarkEntity>().AsNoTracking()
                .Where(m => m.SegmentId != null && pageSegmentIds.Contains(m.SegmentId.Value) && m.SegmentDeletedAt == null)
                .ToDictionaryAsync(m => m.SegmentId!.Value, ct);

        var items = rows
            .Select(row => ToItem(row, pageMarks.GetValueOrDefault(row.Segment.Id)))
            .ToList();
        return new AiReviewQueuePage(items, total, findFilter.Page, findFilter.PerPage);
    }

    /// <summary>
    /// False positives whose segment Cove has already deleted. They no longer exist in the segment
    /// library, so this is the one view that pages over the marks table itself: newest verdict first,
    /// optionally narrowed to one video, so a deletion can still be found and undone.
    /// </summary>
    private static async Task<AiReviewQueuePage> QueryDeletedMarksAsync(DbContext db, AiReviewQueueRequest request, AiReviewSettings settings, FindFilter findFilter, CancellationToken ct)
    {
        var marks = db.Set<ExtAiReviewMarkEntity>().AsNoTracking()
            .Where(m => m.TagId == request.TagId && m.SourceKey == settings.SourceKey && m.SegmentDeletedAt != null);
        if (request.VideoId is { } vid)
        {
            marks = marks.Where(m => m.VideoId == vid);
        }
        if (NormalizeExportStatus(request.ExportStatus) is { } exportStatus)
        {
            marks = marks.Where(m => m.ExportStatus == exportStatus);
        }

        var titleQuery = (findFilter.Q ?? string.Empty).Trim().ToLowerInvariant();
        if (titleQuery.Length > 0)
        {
            var matchingVideos = db.Set<Video>().Where(v => v.Title != null && v.Title.ToLower().Contains(titleQuery)).Select(v => v.Id);
            marks = marks.Where(m => matchingVideos.Contains(m.VideoId));
        }

        var total = await marks.CountAsync(ct);
        var rows = await marks
            .OrderByDescending(m => m.UpdatedAt).ThenByDescending(m => m.Id)
            .Skip((findFilter.Page - 1) * findFilter.PerPage)
            .Take(findFilter.PerPage)
            .ToListAsync(ct);

        var videoIds = rows.Select(r => r.VideoId).Distinct().ToList();
        var titles = await db.Set<Video>().AsNoTracking()
            .Where(v => videoIds.Contains(v.Id))
            .Select(v => new { v.Id, v.Title })
            .ToDictionaryAsync(v => v.Id, v => v.Title ?? string.Empty, ct);

        var items = rows.Select(m => DeletedItem(m, titles.GetValueOrDefault(m.VideoId, string.Empty), settings.SourceKey)).ToList();
        return new AiReviewQueuePage(items, total, findFilter.Page, findFilter.PerPage);
    }

    /// <summary>
    /// The filter dialog criteria plus the queue scope: whatever else was asked for, the queue is one
    /// tag, this source, and AI tag segments only. The scoping fields are init-only, so this copies the
    /// filter rather than editing it.
    /// </summary>
    private static SegmentFilter ScopeToQueue(SegmentFilter source, AiReviewQueueRequest request, AiReviewSettings settings)
        => new()
        {
            HostType = SegmentHostType.Video,
            HostId = request.VideoId,
            SourceKey = settings.SourceKey,
            RefIds = source.RefIds,
            Ids = source.Ids,
            ExcludeIds = source.ExcludeIds,
            VideoIds = source.VideoIds,
            ExcludeVideoIds = source.ExcludeVideoIds,
            TagIds = [request.TagId],
            TagDepth = null,
            VideoTagIds = source.VideoTagIds,
            VideoTagDepth = source.VideoTagDepth,
            PerformerIds = source.PerformerIds,
            SourceCategory = source.SourceCategory,
            Tagged = true,
            MinConfidence = source.MinConfidence,
            MinDurationSec = source.MinDurationSec,
            TitleCriterion = source.TitleCriterion,
            KindCriterion = new StringCriterion { Value = "tag", Modifier = CriterionModifier.Equals },
            SourceKeyCriterion = null,
            SourceRunIdCriterion = source.SourceRunIdCriterion,
            ColorHintCriterion = source.ColorHintCriterion,
            VideoTitleCriterion = source.VideoTitleCriterion,
            HasImageCriterion = source.HasImageCriterion,
            HasPayloadCriterion = source.HasPayloadCriterion,
            StartSecCriterion = source.StartSecCriterion,
            EndSecCriterion = source.EndSecCriterion,
            DurationCriterion = source.DurationCriterion,
            ConfidenceCriterion = source.ConfidenceCriterion,
            CreatedAtCriterion = source.CreatedAtCriterion,
            UpdatedAtCriterion = source.UpdatedAtCriterion,
            TagsCriterion = source.TagsCriterion,
            VideosCriterion = source.VideosCriterion,
            VideoTagsCriterion = source.VideoTagsCriterion,
            PerformersCriterion = source.PerformersCriterion,
            FacesCriterion = source.FacesCriterion,
        };

    private static string? NormalizeExportStatus(string? raw)
    {
        var status = raw?.Trim().ToLowerInvariant();
        return status is AiReviewExportStatus.Pending or AiReviewExportStatus.Done or AiReviewExportStatus.Failed or AiReviewExportStatus.Skipped ? status : null;
    }

    internal static IQueryable<Segment> BaseSegments(DbContext db, string sourceKey, int? videoId)
    {
        var query = db.Set<Segment>().AsNoTracking()
            .Where(s => s.HostType == SegmentHostType.Video && s.SourceKey == sourceKey && s.Kind == "tag" && s.TagId != null);
        if (videoId is { } vid)
        {
            query = query.Where(s => s.HostId == vid);
        }

        return query;
    }

    internal static AiReviewQueueItem ToItem(SegmentLibraryItem row, ExtAiReviewMarkEntity? mark)
        => new(SegmentRecordDto.FromLibraryItem(row), mark is null ? null : ToMarkDto(mark));

    internal static AiReviewQueueItem ToItem(Segment segment, ExtAiReviewMarkEntity? mark, string tagName, string videoTitle)
        => ToItem(new SegmentLibraryItem { Segment = segment, VideoTitle = videoTitle, TagName = tagName }, mark);

    /// <summary>A mark whose segment is gone: the record is rebuilt from what the mark remembered.</summary>
    internal static AiReviewQueueItem DeletedItem(ExtAiReviewMarkEntity mark, string videoTitle, string sourceKey)
        => new(
            new SegmentRecordDto(
                -mark.Id,
                SegmentHostType.Video,
                mark.VideoId,
                videoTitle,
                mark.StartSec,
                mark.EndSec,
                mark.TagId,
                mark.TagName,
                "tag",
                null,
                null,
                null,
                null,
                null,
                sourceKey,
                mark.SourceRunId,
                (float?)mark.Confidence,
                null,
                null,
                mark.CreatedAt.ToString("o"),
                mark.UpdatedAt.ToString("o")),
            ToMarkDto(mark));

    internal static AiReviewMarkDto ToMarkDto(ExtAiReviewMarkEntity mark)
        => new(
            mark.Id,
            mark.SegmentId,
            mark.Verdict,
            mark.ExportStatus,
            ReadStringList(mark.ExportedFilesJson).Count,
            mark.ExportError,
            mark.SegmentDeletedAt,
            mark.ReviewerName,
            mark.UpdatedAt);

    internal static IReadOnlyList<double> ReadFrameTimes(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return [];
        }

        try
        {
            return JsonSerializer.Deserialize<double[]>(json) ?? [];
        }
        catch (JsonException)
        {
            return [];
        }
    }

    internal static IReadOnlyList<string> ReadStringList(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return [];
        }

        try
        {
            return JsonSerializer.Deserialize<string[]>(json) ?? [];
        }
        catch (JsonException)
        {
            return [];
        }
    }
}
