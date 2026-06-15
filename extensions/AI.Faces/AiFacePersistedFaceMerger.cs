using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Faces;

internal sealed record AiFacePersistedFaceMergeResult(
    int MergedPersistedFaceCount,
    IReadOnlyList<int> RelinkedTargetFaceIds,
    IReadOnlyList<int> AffectedFaceIds)
{
    public static readonly AiFacePersistedFaceMergeResult Empty = new(0, [], []);
}

/// <summary>
/// Applies extension-state identity merges to already-persisted Cove face rows: re-points
/// appearances, detections, segments, and embeddings from each duplicate face to its merge target and
/// stamps the duplicate with <c>MergedIntoFaceId</c>. Shared by the reference backfill (pack import)
/// and the per-asset persistence path, so merges decided during normal runs no longer leave orphaned
/// duplicate face rows behind. All updates go through tracked entities; the caller owns
/// <c>SaveChangesAsync</c>.
/// </summary>
internal static class AiFacePersistedFaceMerger
{
    public static async Task<AiFacePersistedFaceMergeResult> ApplyAsync(
        IFaceRepository faceRepo,
        IEmbeddingRepository embeddingRepo,
        IDetectionRepository detectionRepo,
        ISegmentRepository segmentRepo,
        string faceSourceKey,
        IReadOnlyDictionary<string, string> mergedFaceKeyMap,
        CancellationToken ct)
    {
        if (mergedFaceKeyMap.Count == 0)
        {
            return AiFacePersistedFaceMergeResult.Empty;
        }

        var faceKeys = mergedFaceKeyMap.Keys
            .Concat(mergedFaceKeyMap.Values)
            .Where(static key => !string.IsNullOrWhiteSpace(key))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (faceKeys.Length == 0)
        {
            return AiFacePersistedFaceMergeResult.Empty;
        }

        var faces = await faceRepo.FindFacesAsync(new FaceFilter { PrimarySourceKeys = faceKeys }, tracking: true, ct);
        var facesByKey = faces
            .Where(static face => !string.IsNullOrWhiteSpace(face.PrimarySourceKey))
            .GroupBy(static face => face.PrimarySourceKey!, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(static group => group.Key, static group => group.First(), StringComparer.OrdinalIgnoreCase);

        var mergeOperations = mergedFaceKeyMap
            .Select(entry => new
            {
                Duplicate = facesByKey.GetValueOrDefault(entry.Key),
                Target = facesByKey.GetValueOrDefault(entry.Value),
            })
            .Where(static entry => entry.Duplicate is not null
                && entry.Target is not null
                && entry.Duplicate.Id != entry.Target.Id)
            .ToArray();
        if (mergeOperations.Length == 0)
        {
            return AiFacePersistedFaceMergeResult.Empty;
        }

        var duplicateFaceIds = mergeOperations.Select(static entry => entry.Duplicate!.Id).Distinct().ToArray();
        var targetFaceIdByDuplicateFaceId = mergeOperations
            .ToDictionary(static entry => entry.Duplicate!.Id, static entry => entry.Target!.Id);

        // Re-point every evidence row from the duplicates to their targets, grouped so each distinct
        // target gets one tracked bulk update per row kind.
        foreach (var group in targetFaceIdByDuplicateFaceId.GroupBy(static pair => pair.Value))
        {
            var sourceFaceIds = group.Select(static pair => pair.Key).ToArray();
            var sourceRefIds = sourceFaceIds.Select(static id => (long)id).ToArray();
            await faceRepo.UpdateAppearanceFaceIdAsync(faceSourceKey, sourceFaceIds, group.Key, ct);
            await embeddingRepo.UpdateHostIdAsync(EmbeddingHostType.Face, faceSourceKey, sourceFaceIds, group.Key, ct);
            await detectionRepo.UpdateRefIdAsync(faceSourceKey, "face", sourceRefIds, group.Key, ct);
            await segmentRepo.UpdateRefIdAsync(faceSourceKey, sourceRefIds, group.Key, ct);
        }

        var relinkedTargetFaceIds = new List<int>();
        var mergedPersistedFaces = 0;
        foreach (var operation in mergeOperations)
        {
            var duplicate = operation.Duplicate!;
            var target = operation.Target!;

            if (duplicate.PerformerId.HasValue && !target.PerformerId.HasValue)
            {
                target.PerformerId = duplicate.PerformerId;
                relinkedTargetFaceIds.Add(target.Id);
            }

            if (ShouldRefreshTargetLabel(target, duplicate))
            {
                target.Label = duplicate.Label;
            }

            if (duplicate.MergedIntoFaceId != target.Id)
            {
                duplicate.MergedIntoFaceId = target.Id;
                mergedPersistedFaces++;
            }
        }

        var affectedFaceIds = duplicateFaceIds
            .Concat(targetFaceIdByDuplicateFaceId.Values)
            .Distinct()
            .ToArray();
        return new AiFacePersistedFaceMergeResult(mergedPersistedFaces, relinkedTargetFaceIds, affectedFaceIds);
    }

    private static bool ShouldRefreshTargetLabel(Face target, Face duplicate)
        => !string.IsNullOrWhiteSpace(duplicate.Label)
           && (string.IsNullOrWhiteSpace(target.Label)
               || string.Equals(target.Label, target.PrimarySourceKey, StringComparison.OrdinalIgnoreCase));
}
