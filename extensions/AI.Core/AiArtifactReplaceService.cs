using System.Text.Json;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Core;

public interface IAiArtifactReplaceService
{
    Task ReplaceAsync(string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunExecutionPlan> plans, CancellationToken ct = default);
}

internal sealed class AiArtifactReplaceService(
    IEmbeddingRepository embeddingRepo,
    IDetectionRepository detectionRepo,
    IFaceRepository faceRepo,
    ITagApplicationRepository tagAppRepo,
    ISegmentRepository segmentRepo) : IAiArtifactReplaceService
{
    public async Task ReplaceAsync(string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunExecutionPlan> plans, CancellationToken ct = default)
    {
        if (plans.Count == 0 || !hostEntityId.HasValue || string.IsNullOrWhiteSpace(hostEntityType))
        {
            return;
        }

        var normalizedHostType = hostEntityType.Trim().ToLowerInvariant();
        var modelKeysBySource = plans
            .Where(static plan => plan.Decision == AiRunPlanDecision.Rerun && plan.ReplacementArtifactKeys.Count > 0)
            .GroupBy(static plan => ResolveArtifactSourceKey(plan.ExtensionId), StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                static group => group.Key,
                static group => group.SelectMany(plan => plan.ReplacementArtifactKeys).Distinct(StringComparer.OrdinalIgnoreCase).ToArray(),
                StringComparer.OrdinalIgnoreCase);

        if (modelKeysBySource.Count == 0)
        {
            return;
        }

        var affectedTagEntityIds = new List<(AffinityHostType HostType, int HostId)>();

        foreach (var (sourceKey, modelKeys) in modelKeysBySource)
        {
            if (TryResolveEmbeddingHostType(normalizedHostType, out var embeddingHostType))
            {
                var embeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
                {
                    SourceKey = sourceKey,
                    HostType = embeddingHostType,
                    HostId = hostEntityId.Value,
                }, ct);
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

        await embeddingRepo.SaveChangesAsync(ct);

        foreach (var (hostType, entityId) in affectedTagEntityIds.Distinct())
        {
            await tagAppRepo.RemoveOrphanedTagLinksAsync(hostType, [entityId], string.Empty, ct);
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
