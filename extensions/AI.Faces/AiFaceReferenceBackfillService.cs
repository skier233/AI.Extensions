using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

namespace AI.Faces;

internal sealed record AiFaceReferenceBackfillResult(
    int MergedIdentityCount,
    int ReferencePromotedIdentityCount,
    int EvidencePromotedIdentityCount,
    int MergedPersistedFaceCount,
    int RelabeledPersistedFaceCount);

internal sealed class AiFaceReferenceBackfillService(IServiceScopeFactory scopeFactory)
{
    private const string FaceSourceKey = "ext:ai.faces";

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;

    public async Task<AiFaceReferenceBackfillResult> BackfillAsync(SaieReferencePack referencePack, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(referencePack);

        await using var scope = _scopeFactory.CreateAsyncScope();
        var stateStore = scope.ServiceProvider.GetRequiredService<IFaceIdentityStateStore>();
        var settingsStore = scope.ServiceProvider.GetRequiredService<IAiFacesSettingsStore>();
        var reconciler = scope.ServiceProvider.GetRequiredService<AiFaceIdentityReconciler>();
        var faceRepo = scope.ServiceProvider.GetRequiredService<IFaceRepository>();
        var embeddingRepo = scope.ServiceProvider.GetRequiredService<IEmbeddingRepository>();
        var detectionRepo = scope.ServiceProvider.GetRequiredService<IDetectionRepository>();
        var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();

        var snapshot = await stateStore.LoadAsync(ct);
        if (snapshot.Identities.Count == 0)
        {
            return new AiFaceReferenceBackfillResult(0, 0, 0, 0, 0);
        }

        var settings = await settingsStore.LoadAsync(ct);
        var report = reconciler.Reconcile(snapshot, referencePack, settings);
        ApplyReferenceDisplayLabels(snapshot);
        await stateStore.SaveAsync(snapshot, ct);

        var (mergedPersistedFaces, relabeledPersistedFaces) = await ApplyPersistedFaceUpdatesAsync(
            faceRepo, embeddingRepo, detectionRepo, segmentRepo, snapshot, report.MergedFaceKeyMap, ct);

        return new AiFaceReferenceBackfillResult(
            report.MergedIdentityCount,
            report.ReferencePromotedIdentityCount,
            report.EvidencePromotedIdentityCount,
            mergedPersistedFaces,
            relabeledPersistedFaces);
    }

    private static async Task<(int MergedPersistedFaces, int RelabeledPersistedFaces)> ApplyPersistedFaceUpdatesAsync(
        IFaceRepository faceRepo,
        IEmbeddingRepository embeddingRepo,
        IDetectionRepository detectionRepo,
        ISegmentRepository segmentRepo,
        FaceIdentitySnapshot snapshot,
        IReadOnlyDictionary<string, string> mergedFaceKeyMap,
        CancellationToken ct)
    {
        var faceKeys = snapshot.Identities
            .Select(static identity => identity.FaceKey)
            .Concat(mergedFaceKeyMap.Keys)
            .Concat(mergedFaceKeyMap.Values)
            .Where(static key => !string.IsNullOrWhiteSpace(key))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (faceKeys.Length == 0)
        {
            return (0, 0);
        }

        var faces = await faceRepo.FindFacesAsync(
            new FaceFilter { PrimarySourceKeys = faceKeys }, tracking: true, ct);
        var facesByKey = faces
            .Where(static face => !string.IsNullOrWhiteSpace(face.PrimarySourceKey))
            .GroupBy(static face => face.PrimarySourceKey!, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(static group => group.Key, static group => group.First(), StringComparer.OrdinalIgnoreCase);

        var relabeledPersistedFaces = 0;
        foreach (var identity in snapshot.Identities)
        {
            if (!facesByKey.TryGetValue(identity.FaceKey, out var face))
                continue;
            if (!ShouldRefreshFaceLabel(face, identity))
                continue;

            face.Label = identity.Label;
            relabeledPersistedFaces++;
        }

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
            await faceRepo.SaveChangesAsync(ct);
            return (0, relabeledPersistedFaces);
        }

        var duplicateFaceIds = mergeOperations.Select(static entry => entry.Duplicate!.Id).Distinct().ToArray();
        var targetFaceIdByDuplicateFaceId = mergeOperations
            .ToDictionary(static entry => entry.Duplicate!.Id, static entry => entry.Target!.Id);

        await faceRepo.UpdateAppearanceFaceIdAsync(FaceSourceKey, duplicateFaceIds, -1 /* temp */, ct);
        // Re-point each duplicate face's appearances to their target
        var appearances = await faceRepo.FindAppearancesAsync(
            new FaceAppearanceFilter { SourceKey = FaceSourceKey, FaceIds = duplicateFaceIds }, ct);
        foreach (var appearance in appearances)
        {
            appearance.FaceId = targetFaceIdByDuplicateFaceId[appearance.FaceId];
        }

        var detections = await detectionRepo.FindAsync(new DetectionFilter
        {
            SourceKey = FaceSourceKey,
            RefKind = "face",
            RefIds = duplicateFaceIds.Select(static id => (long)id).ToArray(),
        }, ct);
        foreach (var detection in detections)
        {
            detection.RefId = targetFaceIdByDuplicateFaceId[(int)detection.RefId!.Value];
        }

        var segments = await segmentRepo.FindAsync(new SegmentFilter
        {
            SourceKey = FaceSourceKey,
            RefIds = duplicateFaceIds.Select(static id => (long)id).ToArray(),
        }, ct);
        foreach (var segment in segments)
        {
            segment.RefId = targetFaceIdByDuplicateFaceId[(int)segment.RefId!.Value];
        }

        await embeddingRepo.UpdateHostIdAsync(
            EmbeddingHostType.Face, FaceSourceKey, duplicateFaceIds,
            newHostId: -1 /* placeholder; per-face handled below */, ct);
        var embeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
        {
            SourceKey = FaceSourceKey,
            HostType = EmbeddingHostType.Face,
            HostIds = duplicateFaceIds,
        }, ct);
        foreach (var embedding in embeddings)
        {
            embedding.HostId = targetFaceIdByDuplicateFaceId[embedding.HostId];
        }

        var mergedPersistedFaces = 0;
        foreach (var operation in mergeOperations)
        {
            var duplicate = operation.Duplicate!;
            var target = operation.Target!;

            if (duplicate.PerformerId.HasValue && !target.PerformerId.HasValue)
                target.PerformerId = duplicate.PerformerId;

            if (ShouldRefreshTargetLabel(target, duplicate))
                target.Label = duplicate.Label;

            if (duplicate.MergedIntoFaceId != target.Id)
            {
                duplicate.MergedIntoFaceId = target.Id;
                mergedPersistedFaces++;
            }
        }

        await faceRepo.SaveChangesAsync(ct);
        return (mergedPersistedFaces, relabeledPersistedFaces);
    }

    private static bool ShouldRefreshFaceLabel(Face face, StoredFaceIdentity identity)
        => !string.IsNullOrWhiteSpace(ResolveDisplayLabel(identity))
           && (string.IsNullOrWhiteSpace(face.Label)
               || string.Equals(face.Label, face.PrimarySourceKey, StringComparison.OrdinalIgnoreCase));

    private static void ApplyReferenceDisplayLabels(FaceIdentitySnapshot snapshot)
    {
        foreach (var identity in snapshot.Identities)
        {
            if (string.IsNullOrWhiteSpace(identity.ReferenceDisplayName))
                continue;

            if (string.IsNullOrWhiteSpace(identity.Label)
                || string.Equals(identity.Label, identity.FaceKey, StringComparison.OrdinalIgnoreCase))
            {
                identity.Label = identity.ReferenceDisplayName;
            }
        }
    }

    private static string? ResolveDisplayLabel(StoredFaceIdentity identity)
        => !string.IsNullOrWhiteSpace(identity.ReferenceDisplayName)
            ? identity.ReferenceDisplayName
            : identity.Label;

    private static bool ShouldRefreshTargetLabel(Face target, Face duplicate)
        => !string.IsNullOrWhiteSpace(duplicate.Label)
           && (string.IsNullOrWhiteSpace(target.Label)
               || string.Equals(target.Label, target.PrimarySourceKey, StringComparison.OrdinalIgnoreCase));
}
