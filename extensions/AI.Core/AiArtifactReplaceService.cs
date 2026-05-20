using System.Text.Json;

using Cove.Core.Entities;
using Cove.Data;

using Microsoft.EntityFrameworkCore;

namespace AI.Core;

public interface IAiArtifactReplaceService
{
    Task ReplaceAsync(string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunExecutionPlan> plans, CancellationToken ct = default);
}

internal sealed class AiArtifactReplaceService(CoveContext dbContext) : IAiArtifactReplaceService
{
    private readonly CoveContext _dbContext = dbContext;

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

        var affectedTagPairs = new List<TagHostPair>();
        foreach (var (sourceKey, modelKeys) in modelKeysBySource)
        {
            if (TryResolveEmbeddingHostType(normalizedHostType, out var embeddingHostType))
            {
                var embeddings = await _dbContext.Embeddings
                    .Where(embedding => embedding.SourceKey == sourceKey && embedding.HostType == embeddingHostType && embedding.HostId == hostEntityId.Value)
                    .ToListAsync(ct);
                var toRemove = embeddings.Where(embedding => MatchesModelKey(embedding.Meta, modelKeys)).ToArray();
                if (toRemove.Length > 0)
                {
                    _dbContext.Embeddings.RemoveRange(toRemove);
                }
            }

            if (TryResolveDetectionHostType(normalizedHostType, out var detectionHostType))
            {
                var detections = await _dbContext.Set<Detection>()
                    .Where(detection => detection.SourceKey == sourceKey && detection.HostType == detectionHostType && detection.HostId == hostEntityId.Value)
                    .ToListAsync(ct);
                var toRemove = detections.Where(detection => MatchesModelKey(detection.Extra, modelKeys)).ToArray();
                if (toRemove.Length > 0)
                {
                    _dbContext.Set<Detection>().RemoveRange(toRemove);
                }
            }

            if (normalizedHostType == "scene")
            {
                var segments = await _dbContext.Segments
                    .Where(segment => segment.SourceKey == sourceKey && segment.HostType == SegmentHostType.Scene && segment.HostId == hostEntityId.Value)
                    .ToListAsync(ct);
                var toRemove = segments.Where(segment => MatchesModelKey(segment.Payload, modelKeys)).ToArray();
                if (toRemove.Length > 0)
                {
                    _dbContext.Segments.RemoveRange(toRemove);
                }
            }

            if (normalizedHostType is "scene" or "image")
            {
                var appearanceHostType = normalizedHostType == "scene" ? FaceAppearanceHostType.Scene : FaceAppearanceHostType.Image;
                var appearances = await _dbContext.FaceAppearances
                    .Where(appearance => appearance.SourceKey == sourceKey && appearance.HostType == appearanceHostType && appearance.HostId == hostEntityId.Value)
                    .ToListAsync(ct);
                var appearancesToRemove = appearances.Where(appearance => MatchesModelKey(appearance.Payload, modelKeys)).ToArray();
                if (appearancesToRemove.Length > 0)
                {
                    _dbContext.FaceAppearances.RemoveRange(appearancesToRemove);
                }

                var affinityHostType = normalizedHostType == "scene" ? AffinityHostType.Scene : AffinityHostType.Image;
                var applications = await _dbContext.TagApplications
                    .Where(application => application.SourceKey == sourceKey && application.HostType == affinityHostType && application.HostId == hostEntityId.Value && modelKeys.Contains(application.ModelKey))
                    .ToListAsync(ct);
                if (applications.Count > 0)
                {
                    _dbContext.TagApplications.RemoveRange(applications);
                    affectedTagPairs.AddRange(applications.Select(application => new TagHostPair(application.HostType, application.HostId, application.TagId)));
                }
            }
        }

        if (_dbContext.ChangeTracker.HasChanges())
        {
            await _dbContext.SaveChangesAsync(ct);
        }

        if (affectedTagPairs.Count > 0)
        {
            await RemoveOrphanedTagLinksAsync(affectedTagPairs.Distinct().ToArray(), ct);
        }
    }

    private async Task RemoveOrphanedTagLinksAsync(IReadOnlyCollection<TagHostPair> affectedPairs, CancellationToken ct)
    {
        var scenePairs = affectedPairs.Where(pair => pair.HostType == AffinityHostType.Scene).ToArray();
        if (scenePairs.Length > 0)
        {
            var sceneIds = scenePairs.Select(static pair => pair.HostId).Distinct().ToArray();
            var sceneTags = await _dbContext.Set<SceneTag>().Where(sceneTag => sceneIds.Contains(sceneTag.SceneId)).ToListAsync(ct);
            var remainingPairs = await _dbContext.TagApplications
                .Where(application => application.HostType == AffinityHostType.Scene && sceneIds.Contains(application.HostId))
                .Select(application => new TagHostPair(application.HostType, application.HostId, application.TagId))
                .Distinct()
                .ToListAsync(ct);
            var remainingSet = remainingPairs.ToHashSet();
            var orphaned = scenePairs.Where(pair => !remainingSet.Contains(pair)).ToHashSet();
            var toRemove = sceneTags.Where(sceneTag => orphaned.Contains(new TagHostPair(AffinityHostType.Scene, sceneTag.SceneId, sceneTag.TagId))).ToArray();
            if (toRemove.Length > 0)
            {
                _dbContext.Set<SceneTag>().RemoveRange(toRemove);
                await _dbContext.SaveChangesAsync(ct);
            }
        }

        var imagePairs = affectedPairs.Where(pair => pair.HostType == AffinityHostType.Image).ToArray();
        if (imagePairs.Length > 0)
        {
            var imageIds = imagePairs.Select(static pair => pair.HostId).Distinct().ToArray();
            var imageTags = await _dbContext.Set<ImageTag>().Where(imageTag => imageIds.Contains(imageTag.ImageId)).ToListAsync(ct);
            var remainingPairs = await _dbContext.TagApplications
                .Where(application => application.HostType == AffinityHostType.Image && imageIds.Contains(application.HostId))
                .Select(application => new TagHostPair(application.HostType, application.HostId, application.TagId))
                .Distinct()
                .ToListAsync(ct);
            var remainingSet = remainingPairs.ToHashSet();
            var orphaned = imagePairs.Where(pair => !remainingSet.Contains(pair)).ToHashSet();
            var toRemove = imageTags.Where(imageTag => orphaned.Contains(new TagHostPair(AffinityHostType.Image, imageTag.ImageId, imageTag.TagId))).ToArray();
            if (toRemove.Length > 0)
            {
                _dbContext.Set<ImageTag>().RemoveRange(toRemove);
                await _dbContext.SaveChangesAsync(ct);
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

        return normalized.StartsWith("cove.ai.", StringComparison.OrdinalIgnoreCase)
            ? $"ext:ai.{normalized["cove.ai.".Length..]}"
            : normalized;
    }

    private static bool TryResolveEmbeddingHostType(string hostEntityType, out EmbeddingHostType hostType)
    {
        switch (hostEntityType)
        {
            case "scene":
                hostType = EmbeddingHostType.Scene;
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
            case "scene":
                hostType = DetectionHostType.Scene;
                return true;
            case "image":
                hostType = DetectionHostType.Image;
                return true;
            default:
                hostType = default;
                return false;
        }
    }

    private sealed record TagHostPair(AffinityHostType HostType, int HostId, int TagId);
}