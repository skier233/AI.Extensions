using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Cove.Data;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Tagging;

internal sealed class AiTaggingPersistenceService(IServiceScopeFactory scopeFactory)
{
    private const string TagNameUniqueConstraint = "IX_tags_Name";
    private const int MaxTagResolutionAttempts = 3;

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;

    public async Task<IReadOnlyList<string>> PersistAsync(AiDispatchRequest request, AiPreparedArtifactBatch batch, CancellationToken ct = default)
    {
        if (request.Context.HostEntityId is null || string.IsNullOrWhiteSpace(request.Context.HostEntityType))
        {
            return ["AI.Tagging prepared artifacts but skipped persistence because no Cove host entity identity was supplied."];
        }

        var hostEntityType = request.Context.HostEntityType.Trim().ToLowerInvariant();
        if (hostEntityType is not ("scene" or "image"))
        {
            return [$"AI.Tagging persistence does not support host entity type '{request.Context.HostEntityType}'."];
        }

        await using var scope = _scopeFactory.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var tagProvenanceService = scope.ServiceProvider.GetRequiredService<ITagProvenanceService>();
        var hostEntityId = request.Context.HostEntityId.Value;
        var hostDurationSeconds = request.Result.DurationSeconds ?? request.Context.DurationSeconds;
        var tagsByName = await ResolveTagsAsync(db, CollectTagNames(batch).ToArray(), ct);

        var notes = new List<string>();
        var persistedTagEvidenceCount = hostEntityType switch
        {
            "scene" => await PersistSceneTagsAsync(db, hostEntityId, tagsByName, batch, tagProvenanceService, request.Context.RunId, hostDurationSeconds, ct),
            "image" => await PersistImageTagsAsync(db, hostEntityId, tagsByName, batch, tagProvenanceService, request.Context.RunId, hostDurationSeconds, ct),
            _ => 0,
        };

        var persistedSegmentCount = hostEntityType == "scene"
            ? await PersistSceneSegmentsAsync(db, hostEntityId, tagsByName, batch, request, ct)
            : 0;

        await db.SaveChangesAsync(ct);

        if (persistedTagEvidenceCount > 0)
        {
            notes.Add($"Persisted {persistedTagEvidenceCount} AI-generated tag evidence record(s) onto the {hostEntityType}.");
        }

        if (persistedSegmentCount > 0)
        {
            notes.Add($"Persisted {persistedSegmentCount} AI-generated tagging segment(s) onto the scene timeline.");
        }

        if (notes.Count == 0)
        {
            notes.Add("AI.Tagging found no new tag or segment rows to persist for this host entity.");
        }

        return notes;
    }

    private static IEnumerable<string> CollectTagNames(AiPreparedArtifactBatch batch)
    {
        foreach (var tagLink in batch.TagLinks)
        {
            if (!string.IsNullOrWhiteSpace(tagLink.TagName))
            {
                yield return tagLink.TagName.Trim();
            }
        }

        foreach (var segment in batch.Segments)
        {
            if (!string.IsNullOrWhiteSpace(segment.TagName))
            {
                yield return segment.TagName.Trim();
            }
        }
    }

    private static async Task<Dictionary<string, Tag>> ResolveTagsAsync(CoveContext db, IReadOnlyList<string> tagNames, CancellationToken ct)
    {
        var normalizedNames = tagNames
            .Where(static tagName => !string.IsNullOrWhiteSpace(tagName))
            .Select(static tagName => tagName.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (normalizedNames.Length == 0)
        {
            return new Dictionary<string, Tag>(StringComparer.OrdinalIgnoreCase);
        }

        var loweredNames = normalizedNames
            .Select(static tagName => tagName.ToLower())
            .ToArray();

        for (var attempt = 0; attempt < MaxTagResolutionAttempts; attempt++)
        {
            var existingTags = await db.Tags
                .Where(tag => loweredNames.Contains(tag.Name.ToLower()))
                .ToListAsync(ct);

            var tagsByName = existingTags.ToDictionary(static tag => tag.Name, StringComparer.OrdinalIgnoreCase);
            var createdTags = new List<Tag>();
            foreach (var tagName in normalizedNames)
            {
                if (tagsByName.ContainsKey(tagName))
                {
                    continue;
                }

                var tag = new Tag
                {
                    Name = tagName,
                    SortName = tagName,
                };

                db.Tags.Add(tag);
                tagsByName[tagName] = tag;
                createdTags.Add(tag);
            }

            if (createdTags.Count == 0)
            {
                return tagsByName;
            }

            try
            {
                await db.SaveChangesAsync(ct);
                return tagsByName;
            }
            catch (DbUpdateException exception) when (attempt < MaxTagResolutionAttempts - 1 && IsTagNameUniqueViolation(exception))
            {
                foreach (var tag in createdTags)
                {
                    db.Entry(tag).State = EntityState.Detached;
                }
            }
        }

        throw new InvalidOperationException("AI.Tagging could not resolve tags after a duplicate tag-name retry.");
    }

    private static async Task<int> PersistSceneTagsAsync(CoveContext db, int sceneId, IReadOnlyDictionary<string, Tag> tagsByName, AiPreparedArtifactBatch batch, ITagProvenanceService tagProvenanceService, string runId, double? hostDurationSec, CancellationToken ct)
    {
        var provenanceRecords = BuildTagProvenanceRecords(tagsByName, batch, includeSegments: true);
        foreach (var provenance in provenanceRecords)
        {
            await tagProvenanceService.RecordAsync(
                AffinityHostType.Scene,
                sceneId,
                provenance.TagId,
                provenance.SourceKey,
                runId,
                provenance.ModelKey,
                provenance.Confidence,
                totalDurationSec: provenance.TotalDurationSec,
                hostDurationSec: hostDurationSec,
                cancellationToken: ct);
        }

        return provenanceRecords.Count;
    }

    private static async Task<int> PersistImageTagsAsync(CoveContext db, int imageId, IReadOnlyDictionary<string, Tag> tagsByName, AiPreparedArtifactBatch batch, ITagProvenanceService tagProvenanceService, string runId, double? hostDurationSec, CancellationToken ct)
    {
        var existingTagIds = await db.Set<ImageTag>()
            .Where(imageTag => imageTag.ImageId == imageId)
            .Select(imageTag => imageTag.TagId)
            .ToListAsync(ct);
        var existing = new HashSet<int>(existingTagIds);

        foreach (var tagLink in batch.TagLinks)
        {
            var tagId = ResolveTagId(tagsByName, tagLink.TagName);
            if (tagId <= 0 || !existing.Add(tagId))
            {
                continue;
            }

            db.Add(new ImageTag
            {
                ImageId = imageId,
                TagId = tagId,
            });
        }

        var provenanceRecords = BuildTagProvenanceRecords(tagsByName, batch, includeSegments: false);
        foreach (var provenance in provenanceRecords)
        {
            await tagProvenanceService.RecordAsync(
                AffinityHostType.Image,
                imageId,
                provenance.TagId,
                provenance.SourceKey,
                runId,
                provenance.ModelKey,
                provenance.Confidence,
                totalDurationSec: provenance.TotalDurationSec,
                hostDurationSec: hostDurationSec,
                cancellationToken: ct);
        }

        return provenanceRecords.Count;
    }

    private static bool IsTagNameUniqueViolation(DbUpdateException exception)
    {
        var inner = exception.InnerException;
        if (inner is null)
        {
            return false;
        }

        var sqlState = inner.GetType().GetProperty("SqlState")?.GetValue(inner) as string;
        var constraintName = inner.GetType().GetProperty("ConstraintName")?.GetValue(inner) as string;
        return string.Equals(sqlState, "23505", StringComparison.Ordinal)
            && string.Equals(constraintName, TagNameUniqueConstraint, StringComparison.Ordinal);
    }

    private static async Task<int> PersistSceneSegmentsAsync(CoveContext db, int sceneId, IReadOnlyDictionary<string, Tag> tagsByName, AiPreparedArtifactBatch batch, AiDispatchRequest request, CancellationToken ct)
    {
        var modelKeys = batch.Segments
            .Select(static segment => segment.Metadata is not null && segment.Metadata.TryGetValue("modelKey", out var modelKey) ? modelKey : null)
            .Where(static modelKey => !string.IsNullOrWhiteSpace(modelKey))
            .Select(static modelKey => modelKey!.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var existingSegments = await db.Segments
            .Where(segment => segment.HostType == SegmentHostType.Scene && segment.HostId == sceneId && segment.SourceKey == "ext:ai.tagging")
            .ToListAsync(ct);
        var segmentsToReplace = modelKeys.Length == 0
            ? []
            : existingSegments.Where(segment => MatchesModelKey(segment.Payload, modelKeys)).ToArray();

        if (segmentsToReplace.Length > 0)
        {
            db.Segments.RemoveRange(segmentsToReplace);
        }

        var inserted = 0;
        foreach (var segment in batch.Segments)
        {
            db.Segments.Add(new Segment
            {
                HostType = SegmentHostType.Scene,
                HostId = sceneId,
                StartSec = segment.StartSeconds,
                EndSec = segment.EndSeconds,
                TagId = ResolveNullableTagId(tagsByName, segment.TagName),
                Kind = segment.Kind,
                Payload = SerializePayload(segment.Metadata),
                SourceKey = segment.SourceKey,
                SourceRunId = request.Context.RunId,
                Confidence = segment.Confidence is null ? null : (float)segment.Confidence.Value,
                Title = segment.Title,
            });
            inserted++;
        }

        return inserted;
    }

    private static bool MatchesModelKey(JsonDocument? document, IReadOnlyCollection<string> modelKeys)
    {
        if (document is null || document.RootElement.ValueKind != JsonValueKind.Object || !document.RootElement.TryGetProperty("modelKey", out var element))
        {
            return false;
        }

        var raw = element.GetString();
        return !string.IsNullOrWhiteSpace(raw) && modelKeys.Contains(raw.Trim(), StringComparer.OrdinalIgnoreCase);
    }

    private static int ResolveTagId(IReadOnlyDictionary<string, Tag> tagsByName, string? tagName)
    {
        if (string.IsNullOrWhiteSpace(tagName))
        {
            return 0;
        }

        return tagsByName.TryGetValue(tagName.Trim(), out var tag) ? tag.Id : 0;
    }

    private static int? ResolveNullableTagId(IReadOnlyDictionary<string, Tag> tagsByName, string? tagName)
    {
        var tagId = ResolveTagId(tagsByName, tagName);
        return tagId > 0 ? tagId : null;
    }

    private static JsonDocument? SerializePayload(IReadOnlyDictionary<string, string>? metadata)
    {
        if (metadata is null || metadata.Count == 0)
        {
            return null;
        }

        return JsonDocument.Parse(JsonSerializer.Serialize(metadata));
    }

    private static IReadOnlyList<TagProvenanceRecord> BuildTagProvenanceRecords(IReadOnlyDictionary<string, Tag> tagsByName, AiPreparedArtifactBatch batch, bool includeSegments)
    {
        var records = new List<TagProvenanceRecord>();

        foreach (var tagLink in batch.TagLinks)
        {
            var tagId = ResolveTagId(tagsByName, tagLink.TagName);
            if (tagId <= 0)
            {
                continue;
            }

            records.Add(new TagProvenanceRecord(
                tagId,
                tagLink.SourceKey,
                tagLink.ModelKey,
                tagLink.Confidence is null ? null : (float)tagLink.Confidence.Value,
                TotalDurationSec: null));
        }

        if (includeSegments)
        {
            foreach (var segment in batch.Segments)
            {
                var tagId = ResolveTagId(tagsByName, segment.TagName);
                if (tagId <= 0)
                {
                    continue;
                }

                string? modelKey = null;
                segment.Metadata?.TryGetValue("modelKey", out modelKey);
                records.Add(new TagProvenanceRecord(
                    tagId,
                    segment.SourceKey,
                    modelKey,
                    segment.Confidence is null ? null : (float)segment.Confidence.Value,
                    TotalDurationSec: GetSegmentDurationSeconds(segment)));
            }
        }

        return records
            .GroupBy(record => new { record.TagId, record.SourceKey, ModelKey = record.ModelKey ?? string.Empty })
            .Select(group => new TagProvenanceRecord(
                group.Key.TagId,
                group.Key.SourceKey,
                string.IsNullOrWhiteSpace(group.Key.ModelKey) ? null : group.Key.ModelKey,
                group.Max(record => record.Confidence),
                group.Where(record => record.TotalDurationSec.HasValue).Sum(record => record.TotalDurationSec) is var summedDuration && summedDuration > 0d
                    ? summedDuration
                    : null))
            .ToArray();
    }

    private static double? GetSegmentDurationSeconds(AiPreparedSegment segment)
    {
        if (!segment.EndSeconds.HasValue)
        {
            return null;
        }

        var duration = segment.EndSeconds.Value - segment.StartSeconds;
        return duration > 0d ? duration : null;
    }

    private sealed record TagProvenanceRecord(int TagId, string SourceKey, string? ModelKey, float? Confidence, double? TotalDurationSec);
}