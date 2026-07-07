using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

namespace AI.Tagging;

internal sealed class AiTaggingPersistenceService(IServiceScopeFactory scopeFactory)
{
    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;

    public async Task<IReadOnlyList<string>> PersistAsync(AiDispatchRequest request, AiPreparedArtifactBatch batch, CancellationToken ct = default)
    {
        if (request.Context.HostEntityId is null || string.IsNullOrWhiteSpace(request.Context.HostEntityType))
        {
            return ["AI.Tagging prepared artifacts but skipped persistence because no Cove host entity identity was supplied."];
        }

        var hostEntityType = request.Context.HostEntityType.Trim().ToLowerInvariant();
        if (hostEntityType is not ("video" or "image"))
        {
            return [$"AI.Tagging persistence does not support host entity type '{request.Context.HostEntityType}'."];
        }

        await using var scope = _scopeFactory.CreateAsyncScope();
        var tagRepo = scope.ServiceProvider.GetRequiredService<ITagRepository>();
        var imageRepo = scope.ServiceProvider.GetRequiredService<IImageRepository>();
        var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();
        var tagProvenanceService = scope.ServiceProvider.GetRequiredService<ITagProvenanceService>();
        var hostEntityId = request.Context.HostEntityId.Value;
        var hostDurationSeconds = request.Result.DurationSeconds ?? request.Context.DurationSeconds;
        var settings = await AiTaggingSettingsStore.LoadAsync(scope.ServiceProvider, ct);
        var effectiveBatch = ApplyTagNameOverrides(batch, settings.ToOverrideMap());
        var tagsByName = await tagRepo.FindOrCreateByNamesAsync(CollectTagNames(effectiveBatch).ToArray(), ct);

        var notes = new List<string>();
        var persistedTagEvidenceCount = hostEntityType switch
        {
            "video" => await PersistVideoTagsAsync(hostEntityId, tagsByName, effectiveBatch, tagProvenanceService, request.Context.RunId, hostDurationSeconds, ct),
            "image" => await PersistImageTagsAsync(imageRepo, hostEntityId, tagsByName, effectiveBatch, tagProvenanceService, request.Context.RunId, hostDurationSeconds, ct),
            _ => 0,
        };

        var persistedSegmentCount = hostEntityType == "video"
            ? await PersistVideoSegmentsAsync(segmentRepo, hostEntityId, tagsByName, effectiveBatch, request, ct)
            : 0;

        await tagRepo.FindOrCreateByNamesAsync([], ct); // no-op save handled inside; call SaveChanges via segmentRepo
        await segmentRepo.SaveChangesAsync(ct);

        if (persistedTagEvidenceCount > 0)
        {
            notes.Add($"Persisted {persistedTagEvidenceCount} AI-generated tag evidence record(s) onto the {hostEntityType}.");
        }

        if (persistedSegmentCount > 0)
        {
            notes.Add($"Persisted {persistedSegmentCount} AI-generated tagging segment(s) onto the video timeline.");
        }

        if (notes.Count == 0)
        {
            notes.Add("AI.Tagging found no new tag or segment rows to persist for this host entity.");
        }

        return notes;
    }

    // Persists a whole batch of host entities in one scope and one final SaveChanges. The host's
    // SaveChangesAsync recomputes denormalized counts + id arrays on every call, so collapsing ~one save per
    // image into one save per batch turns that (heavy) recompute from O(images) into O(1) per batch — the main
    // reason large tagging runs ground Cove to a halt. Returns per-item notes aligned with the input order.
    public async Task<IReadOnlyList<IReadOnlyList<string>>> PersistBatchAsync(
        IReadOnlyList<(AiDispatchRequest Request, AiPreparedArtifactBatch Batch)> items,
        CancellationToken ct = default)
    {
        var notes = new List<string>[items.Count];
        for (var i = 0; i < items.Count; i++)
        {
            notes[i] = [];
        }

        var valid = new List<(int Index, string HostEntityType, int HostEntityId, double? HostDurationSeconds, string RunId, AiPreparedArtifactBatch EffectiveBatch)>();

        await using var scope = _scopeFactory.CreateAsyncScope();
        var tagRepo = scope.ServiceProvider.GetRequiredService<ITagRepository>();
        var imageRepo = scope.ServiceProvider.GetRequiredService<IImageRepository>();
        var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();
        var tagProvenanceService = scope.ServiceProvider.GetRequiredService<ITagProvenanceService>();
        var settings = await AiTaggingSettingsStore.LoadAsync(scope.ServiceProvider, ct);
        var overrideMap = settings.ToOverrideMap();

        for (var i = 0; i < items.Count; i++)
        {
            var request = items[i].Request;
            if (request.Context.HostEntityId is null || string.IsNullOrWhiteSpace(request.Context.HostEntityType))
            {
                notes[i].Add("AI.Tagging prepared artifacts but skipped persistence because no Cove host entity identity was supplied.");
                continue;
            }

            var hostEntityType = request.Context.HostEntityType.Trim().ToLowerInvariant();
            if (hostEntityType is not ("video" or "image"))
            {
                notes[i].Add($"AI.Tagging persistence does not support host entity type '{request.Context.HostEntityType}'.");
                continue;
            }

            valid.Add((
                i,
                hostEntityType,
                request.Context.HostEntityId.Value,
                request.Result.DurationSeconds ?? request.Context.DurationSeconds,
                request.Context.RunId,
                ApplyTagNameOverrides(items[i].Batch, overrideMap)));
        }

        if (valid.Count == 0)
        {
            return notes.Select(static n => (IReadOnlyList<string>)n).ToArray();
        }

        // Resolve every tag name across the whole batch in one round-trip (creates new tags once).
        var allTagNames = valid.SelectMany(item => CollectTagNames(item.EffectiveBatch)).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
        var tagsByName = await tagRepo.FindOrCreateByNamesAsync(allTagNames, ct);

        foreach (var item in valid)
        {
            var persistedTagEvidenceCount = item.HostEntityType switch
            {
                "video" => await PersistVideoTagsAsync(item.HostEntityId, tagsByName, item.EffectiveBatch, tagProvenanceService, item.RunId, item.HostDurationSeconds, ct),
                "image" => await PersistImageTagsAsync(imageRepo, item.HostEntityId, tagsByName, item.EffectiveBatch, tagProvenanceService, item.RunId, item.HostDurationSeconds, ct),
                _ => 0,
            };

            var persistedSegmentCount = item.HostEntityType == "video"
                ? await PersistVideoSegmentsAsync(segmentRepo, item.HostEntityId, tagsByName, item.EffectiveBatch, items[item.Index].Request, ct)
                : 0;

            if (persistedTagEvidenceCount > 0)
            {
                notes[item.Index].Add($"Persisted {persistedTagEvidenceCount} AI-generated tag evidence record(s) onto the {item.HostEntityType}.");
            }

            if (persistedSegmentCount > 0)
            {
                notes[item.Index].Add($"Persisted {persistedSegmentCount} AI-generated tagging segment(s) onto the video timeline.");
            }

            if (notes[item.Index].Count == 0)
            {
                notes[item.Index].Add("AI.Tagging found no new tag or segment rows to persist for this host entity.");
            }
        }

        // One save for the whole batch → one denormalized-count/id-array recompute instead of one per image.
        await segmentRepo.SaveChangesAsync(ct);

        return notes.Select(static n => (IReadOnlyList<string>)n).ToArray();
    }

    private static AiPreparedArtifactBatch ApplyTagNameOverrides(AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, string> tagNameOverrides)
    {
        if (tagNameOverrides.Count == 0)
        {
            return batch;
        }

        var effective = new AiPreparedArtifactBatch();
        effective.FaceAppearances.AddRange(batch.FaceAppearances);
        effective.Detections.AddRange(batch.Detections);
        effective.Embeddings.AddRange(batch.Embeddings);
        effective.Faces.AddRange(batch.Faces);
        effective.DeferredWorkItems.AddRange(batch.DeferredWorkItems);
        effective.Notes.AddRange(batch.Notes);

        effective.TagLinks.AddRange(batch.TagLinks.Select(tagLink =>
        {
            var resolvedName = ResolveOverride(tagLink.TagName, tagNameOverrides) ?? tagLink.TagName;
            return string.Equals(resolvedName, tagLink.TagName, StringComparison.Ordinal)
                ? tagLink
                : tagLink with { TagName = resolvedName };
        }));

        effective.Segments.AddRange(batch.Segments.Select(segment =>
        {
            var resolvedName = ResolveOverride(segment.TagName, tagNameOverrides);
            var resolvedTitle = string.IsNullOrWhiteSpace(segment.Title) || string.Equals(segment.Title.Trim(), segment.TagName?.Trim(), StringComparison.OrdinalIgnoreCase)
                ? resolvedName
                : segment.Title;
            return string.Equals(resolvedName, segment.TagName, StringComparison.Ordinal) && string.Equals(resolvedTitle, segment.Title, StringComparison.Ordinal)
                ? segment
                : segment with { TagName = resolvedName, Title = resolvedTitle };
        }));

        return effective;
    }

    private static string? ResolveOverride(string? tagName, IReadOnlyDictionary<string, string> tagNameOverrides)
    {
        if (string.IsNullOrWhiteSpace(tagName))
        {
            return tagName;
        }

        var trimmed = tagName.Trim();
        return tagNameOverrides.TryGetValue(trimmed, out var overrideName) && !string.IsNullOrWhiteSpace(overrideName)
            ? overrideName.Trim()
            : trimmed;
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

    private static async Task<int> PersistVideoTagsAsync(int videoId, IReadOnlyDictionary<string, Tag> tagsByName, AiPreparedArtifactBatch batch, ITagProvenanceService tagProvenanceService, string runId, double? hostDurationSec, CancellationToken ct)
    {
        var provenanceRecords = BuildTagProvenanceRecords(tagsByName, batch, includeSegments: true);
        foreach (var provenance in provenanceRecords)
        {
            await tagProvenanceService.RecordAsync(
                AffinityHostType.Video,
                videoId,
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

    private static async Task<int> PersistImageTagsAsync(IImageRepository imageRepo, int imageId, IReadOnlyDictionary<string, Tag> tagsByName, AiPreparedArtifactBatch batch, ITagProvenanceService tagProvenanceService, string runId, double? hostDurationSec, CancellationToken ct)
    {
        var existingTagIds = await imageRepo.GetTagIdsAsync(imageId, ct);
        var existing = new HashSet<int>(existingTagIds);

        foreach (var tagLink in batch.TagLinks)
        {
            var tagId = ResolveTagId(tagsByName, tagLink.TagName);
            if (tagId <= 0 || !existing.Add(tagId))
            {
                continue;
            }

            imageRepo.AddTagLink(imageId, tagId);
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

    private static async Task<int> PersistVideoSegmentsAsync(ISegmentRepository segmentRepo, int videoId, IReadOnlyDictionary<string, Tag> tagsByName, AiPreparedArtifactBatch batch, AiDispatchRequest request, CancellationToken ct)
    {
        var modelKeys = batch.Segments
            .Select(static segment => segment.Metadata is not null && segment.Metadata.TryGetValue("modelKey", out var modelKey) ? modelKey : null)
            .Where(static modelKey => !string.IsNullOrWhiteSpace(modelKey))
            .Select(static modelKey => modelKey!.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var existingSegments = await segmentRepo.FindAsync(new SegmentFilter
        {
            HostType = SegmentHostType.Video,
            HostId = videoId,
            SourceKey = "ext:ai.tagging",
        }, ct);

        var segmentsToReplace = modelKeys.Length == 0
            ? []
            : existingSegments.Where(segment => MatchesModelKey(segment.Payload, modelKeys)).ToArray();

        if (segmentsToReplace.Length > 0)
        {
            segmentRepo.RemoveRange(segmentsToReplace);
        }

        var inserted = 0;
        foreach (var segment in batch.Segments)
        {
            segmentRepo.Add(new Segment
            {
                HostType = SegmentHostType.Video,
                HostId = videoId,
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

    private static bool MatchesModelKey(System.Text.Json.JsonDocument? document, IReadOnlyCollection<string> modelKeys)
    {
        if (document is null || document.RootElement.ValueKind != System.Text.Json.JsonValueKind.Object || !document.RootElement.TryGetProperty("modelKey", out var element))
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
