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
        var store = scope.ServiceProvider.GetRequiredService<IFaceIdentityStore>();
        var settingsStore = scope.ServiceProvider.GetRequiredService<IAiFacesSettingsStore>();
        var reconciler = scope.ServiceProvider.GetRequiredService<AiFaceIdentityReconciler>();
        var faceRepo = scope.ServiceProvider.GetRequiredService<IFaceRepository>();
        var embeddingRepo = scope.ServiceProvider.GetRequiredService<IEmbeddingRepository>();
        var detectionRepo = scope.ServiceProvider.GetRequiredService<IDetectionRepository>();
        var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();

        // A pack import is a rare, whole-graph operation: load every identity, re-match against the new
        // pack, and re-consolidate. BeginFullAsync takes the same reconcile gate as the per-asset path.
        await using var transaction = await store.BeginFullAsync(ct);
        var snapshot = transaction.Snapshot;
        if (snapshot.Identities.Count == 0)
        {
            return new AiFaceReferenceBackfillResult(0, 0, 0, 0, 0);
        }

        var settings = await settingsStore.LoadAsync(ct);
        var report = reconciler.Reconcile(snapshot, referencePack, settings);
        ApplyReferenceDisplayLabels(snapshot);
        await transaction.CommitAsync(ct);

        var relinkedTargetFaceIds = new List<int>();
        var (mergedPersistedFaces, relabeledPersistedFaces) = await ApplyPersistedFaceUpdatesAsync(
            faceRepo, embeddingRepo, detectionRepo, segmentRepo, snapshot, report.MergedFaceKeyMap, relinkedTargetFaceIds, ct);

        // A merge that transferred a performer onto the target face must propagate that performer to the
        // target's hosts (videos/images) — including the merged-in face's former hosts, now re-pointed
        // to the target — otherwise the auto-linked performer never reaches those entities.
        if (relinkedTargetFaceIds.Count > 0)
        {
            var propagation = scope.ServiceProvider.GetService<IFacePerformerPropagationService>();
            if (propagation is not null)
            {
                var affectedHosts = await faceRepo.FindAppearancesAsync(
                    new FaceAppearanceFilter { FaceIds = relinkedTargetFaceIds.Distinct().ToArray() }, ct);
                foreach (var hostGroup in affectedHosts.GroupBy(appearance => (appearance.HostType, appearance.HostId)))
                    await propagation.ReconcileHostAsync(hostGroup.Key.HostType, hostGroup.Key.HostId, ct);
                await faceRepo.SaveChangesAsync(ct);
            }
        }

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
        ICollection<int> relinkedTargetFaceIds,
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

        var mergeResult = await AiFacePersistedFaceMerger.ApplyAsync(
            faceRepo, embeddingRepo, detectionRepo, segmentRepo, FaceSourceKey, mergedFaceKeyMap, ct);
        foreach (var targetFaceId in mergeResult.RelinkedTargetFaceIds)
        {
            relinkedTargetFaceIds.Add(targetFaceId);
        }

        await faceRepo.SaveChangesAsync(ct);
        return (mergeResult.MergedPersistedFaceCount, relabeledPersistedFaces);
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
}
