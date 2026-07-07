using System.Text.Json;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Core;

public interface IAiArtifactReplaceService
{
    Task ReplaceAsync(string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunExecutionPlan> plans, CancellationToken ct = default);

    /// <summary>Batch form of <see cref="ReplaceAsync"/>: queues the stale-artifact removals for every host in
    /// the batch and flushes them in a single <c>SaveChanges</c> (plus one orphaned-tag-link sweep), instead
    /// of a transaction per host. Used by batch image runs.</summary>
    Task ReplaceBatchAsync(IReadOnlyList<AiArtifactReplaceTarget> targets, CancellationToken ct = default);
}

public sealed record AiArtifactReplaceTarget(string? HostEntityType, int? HostEntityId, IReadOnlyList<AiRunExecutionPlan> Plans);

internal sealed class AiArtifactReplaceService(
    IEmbeddingRepository embeddingRepo,
    IDetectionRepository detectionRepo,
    IFaceRepository faceRepo,
    ITagApplicationRepository tagAppRepo,
    ISegmentRepository segmentRepo) : IAiArtifactReplaceService
{
    public async Task ReplaceAsync(string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunExecutionPlan> plans, CancellationToken ct = default)
    {
        var affectedTagEntityIds = new List<(AffinityHostType HostType, int HostId)>();
        await QueueRemovalsAsync(hostEntityType, hostEntityId, plans, affectedTagEntityIds, preloadedEmbeddings: null, ct);

        await embeddingRepo.SaveChangesAsync(ct);
        await SweepOrphanedTagLinksAsync(affectedTagEntityIds, ct);
    }

    public async Task ReplaceBatchAsync(IReadOnlyList<AiArtifactReplaceTarget> targets, CancellationToken ct = default)
    {
        if (targets.Count == 0)
        {
            return;
        }

        // The existing embeddings to remove are the heavy read (they carry vectors). Bulk-load them for every
        // host in the batch in one query per (host-type, source) instead of one per host, then queue each
        // host's removals and flush once. All repos here share one scoped CoveContext, so a single SaveChanges
        // commits the whole batch's deletes.
        var preloadedEmbeddings = await PreloadEmbeddingsAsync(targets, ct);

        var affectedTagEntityIds = new List<(AffinityHostType HostType, int HostId)>();
        foreach (var target in targets)
        {
            await QueueRemovalsAsync(target.HostEntityType, target.HostEntityId, target.Plans, affectedTagEntityIds, preloadedEmbeddings, ct);
        }

        await embeddingRepo.SaveChangesAsync(ct);
        await SweepOrphanedTagLinksAsync(affectedTagEntityIds, ct);
    }

    // Bulk-loads, for the whole batch, the existing embeddings that might need removal — one query per
    // (embedding host-type, source key) using a HostIds IN-list — keyed by (host-type, source, host id).
    private async Task<IReadOnlyDictionary<(EmbeddingHostType HostType, string SourceKey, int HostId), List<Embedding>>> PreloadEmbeddingsAsync(
        IReadOnlyList<AiArtifactReplaceTarget> targets,
        CancellationToken ct)
    {
        var hostIdsByGroup = new Dictionary<(EmbeddingHostType HostType, string SourceKey), HashSet<int>>();
        foreach (var target in targets)
        {
            if (!target.HostEntityId.HasValue || string.IsNullOrWhiteSpace(target.HostEntityType))
            {
                continue;
            }

            var normalizedHostType = target.HostEntityType.Trim().ToLowerInvariant();
            if (!TryResolveEmbeddingHostType(normalizedHostType, out var embeddingHostType))
            {
                continue;
            }

            foreach (var sourceKey in BuildModelKeysBySource(target.Plans).Keys)
            {
                var key = (embeddingHostType, sourceKey);
                if (!hostIdsByGroup.TryGetValue(key, out var hostIds))
                {
                    hostIds = [];
                    hostIdsByGroup[key] = hostIds;
                }

                hostIds.Add(target.HostEntityId.Value);
            }
        }

        var lookup = new Dictionary<(EmbeddingHostType, string, int), List<Embedding>>();
        foreach (var ((embeddingHostType, sourceKey), hostIds) in hostIdsByGroup)
        {
            var embeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
            {
                SourceKey = sourceKey,
                HostType = embeddingHostType,
                HostIds = hostIds.ToArray(),
            }, ct);

            foreach (var embedding in embeddings)
            {
                var key = (embeddingHostType, sourceKey, embedding.HostId);
                if (!lookup.TryGetValue(key, out var list))
                {
                    list = [];
                    lookup[key] = list;
                }

                list.Add(embedding);
            }
        }

        return lookup;
    }

    private static Dictionary<string, string[]> BuildModelKeysBySource(IReadOnlyList<AiRunExecutionPlan> plans)
        => plans
            .Where(static plan => plan.Decision == AiRunPlanDecision.Rerun && plan.ReplacementArtifactKeys.Count > 0)
            .GroupBy(static plan => ResolveArtifactSourceKey(plan.ExtensionId), StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                static group => group.Key,
                static group => group.SelectMany(plan => plan.ReplacementArtifactKeys).Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
                StringComparer.OrdinalIgnoreCase);

    private async Task SweepOrphanedTagLinksAsync(List<(AffinityHostType HostType, int HostId)> affectedTagEntityIds, CancellationToken ct)
    {
        foreach (var group in affectedTagEntityIds.Distinct().GroupBy(static item => item.HostType))
        {
            await tagAppRepo.RemoveOrphanedTagLinksAsync(group.Key, group.Select(static item => item.HostId).ToArray(), string.Empty, ct);
        }
    }

    // Queues (does not save) the removal of stale artifacts for one host that this run is about to re-create.
    // When <paramref name="preloadedEmbeddings"/> is supplied (batch path), existing embeddings are read from
    // it instead of issuing a per-host query.
    private async Task QueueRemovalsAsync(
        string? hostEntityType,
        int? hostEntityId,
        IReadOnlyList<AiRunExecutionPlan> plans,
        List<(AffinityHostType HostType, int HostId)> affectedTagEntityIds,
        IReadOnlyDictionary<(EmbeddingHostType HostType, string SourceKey, int HostId), List<Embedding>>? preloadedEmbeddings,
        CancellationToken ct)
    {
        if (plans.Count == 0 || !hostEntityId.HasValue || string.IsNullOrWhiteSpace(hostEntityType))
        {
            return;
        }

        var normalizedHostType = hostEntityType.Trim().ToLowerInvariant();
        var modelKeysBySource = BuildModelKeysBySource(plans);

        if (modelKeysBySource.Count == 0)
        {
            return;
        }

        foreach (var (sourceKey, modelKeys) in modelKeysBySource)
        {
            if (TryResolveEmbeddingHostType(normalizedHostType, out var embeddingHostType))
            {
                IReadOnlyList<Embedding> embeddings;
                if (preloadedEmbeddings is not null)
                {
                    embeddings = preloadedEmbeddings.TryGetValue((embeddingHostType, sourceKey, hostEntityId.Value), out var preloaded)
                        ? preloaded
                        : [];
                }
                else
                {
                    embeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
                    {
                        SourceKey = sourceKey,
                        HostType = embeddingHostType,
                        HostId = hostEntityId.Value,
                    }, ct);
                }

                var toRemove = embeddings.Where(e => MatchesModelKey(e.Meta, modelKeys)).ToArray();
                if (toRemove.Length > 0)
                {
                    embeddingRepo.RemoveRange(toRemove);
                }
            }

            if (TryResolveDetectionHostType(normalizedHostType, out var detectionHostType))
            {
                var detections = await detectionRepo.FindAsync(new DetectionFilter
                {
                    SourceKey = sourceKey,
                    HostType = detectionHostType,
                    HostId = hostEntityId.Value,
                }, ct);
                var toRemove = detections.Where(d => MatchesModelKey(d.Extra, modelKeys)).ToArray();
                if (toRemove.Length > 0)
                {
                    detectionRepo.RemoveRange(toRemove);
                }
            }

            if (normalizedHostType == "video")
            {
                var segments = await segmentRepo.FindAsync(new SegmentFilter
                {
                    SourceKey = sourceKey,
                    HostType = SegmentHostType.Video,
                    HostId = hostEntityId.Value,
                }, ct);
                var toRemove = segments.Where(s => MatchesModelKey(s.Payload, modelKeys)).ToArray();
                if (toRemove.Length > 0)
                {
                    segmentRepo.RemoveRange(toRemove);
                }
            }

            if (normalizedHostType is "video" or "image")
            {
                var appearanceHostType = normalizedHostType == "video" ? FaceAppearanceHostType.Video : FaceAppearanceHostType.Image;
                var appearances = await faceRepo.FindAppearancesAsync(new FaceAppearanceFilter
                {
                    SourceKey = sourceKey,
                    HostType = appearanceHostType,
                    HostId = hostEntityId.Value,
                }, ct);
                var toRemove = appearances.Where(a => MatchesModelKey(a.Payload, modelKeys)).ToArray();
                if (toRemove.Length > 0)
                {
                    faceRepo.RemoveAppearances(toRemove);
                }

                var affinityHostType = normalizedHostType == "video" ? AffinityHostType.Video : AffinityHostType.Image;
                var applications = await tagAppRepo.FindAsync(new TagApplicationFilter
                {
                    SourceKey = sourceKey,
                    HostType = affinityHostType,
                    HostId = hostEntityId.Value,
                    ModelKeys = modelKeys,
                }, ct);
                if (applications.Count > 0)
                {
                    tagAppRepo.RemoveRange(applications);
                    affectedTagEntityIds.Add((affinityHostType, hostEntityId.Value));
                }
            }
        }
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

    private static string ResolveArtifactSourceKey(string extensionId)
    {
        var normalized = (extensionId ?? string.Empty).Trim();
        if (normalized.StartsWith("ext:", StringComparison.OrdinalIgnoreCase))
        {
            return normalized;
        }

        return normalized.StartsWith("cove.community.ai.", StringComparison.OrdinalIgnoreCase)
            ? $"ext:ai.{normalized["cove.community.ai.".Length..]}"
            : normalized;
    }

    private static bool TryResolveEmbeddingHostType(string hostEntityType, out EmbeddingHostType hostType)
    {
        switch (hostEntityType)
        {
            case "video":
                hostType = EmbeddingHostType.Video;
                return true;
            case "image":
                hostType = EmbeddingHostType.Image;
                return true;
            case "face":
                hostType = EmbeddingHostType.Face;
                return true;
            default:
                hostType = default;
                return false;
        }
    }

    private static bool TryResolveDetectionHostType(string hostEntityType, out DetectionHostType hostType)
    {
        switch (hostEntityType)
        {
            case "video":
                hostType = DetectionHostType.Video;
                return true;
            case "image":
                hostType = DetectionHostType.Image;
                return true;
            default:
                hostType = default;
                return false;
        }
    }
}
