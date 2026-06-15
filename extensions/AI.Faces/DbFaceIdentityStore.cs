using System.Text.Json;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

using Pgvector;
using Pgvector.EntityFrameworkCore;

namespace AI.Faces;

// DB-backed replacement for the serialized `face-identity-snapshot` blob. Instead of loading, globally
// re-merging, and re-saving the entire identity graph on every asset (O(N^2+) per asset, unbounded N),
// reconcile now loads only the similarity candidates relevant to the current asset and persists deltas.
//
// Unit-of-work: BeginIncrementalAsync/BeginFullAsync return a FaceIdentityTransaction holding a DI scope,
// the host DbContext, the loaded baseline, and a working FaceIdentitySnapshot (the in-memory model the
// reconciler mutates). The caller mutates Snapshot then CommitAsync persists the diff. A process-wide gate
// serializes the reconcile critical section (also fixing the old blob's last-writer-wins race); other AI
// work (tagging/visual/server calls) still runs in parallel.
internal interface IFaceIdentityStore
{
    Task<FaceIdentityTransaction> BeginIncrementalAsync(
        IReadOnlyList<IReadOnlyList<float>> queryVectors,
        IReadOnlyCollection<string> referenceExternalIds,
        int candidateK,
        CancellationToken ct = default);

    Task<FaceIdentityTransaction> BeginFullAsync(CancellationToken ct = default);

    Task DeleteByFaceKeyAsync(string faceKey, CancellationToken ct = default);

    Task ClearAllAsync(CancellationToken ct = default);
}

internal sealed class DbFaceIdentityStore(
    IServiceScopeFactory scopeFactory,
    IFaceIdentityStateStore legacyBlobStore) : IFaceIdentityStore
{
    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;
    private readonly IFaceIdentityStateStore _legacyBlobStore = legacyBlobStore;

    // Serializes the reconcile critical section across assets so concurrent dispatches (MaxInFlight > 1)
    // cannot clobber each other's identity-graph edits or race on ordinal allocation.
    private readonly SemaphoreSlim _gate = new(1, 1);

    private bool _importChecked;

    public Task<FaceIdentityTransaction> BeginIncrementalAsync(
        IReadOnlyList<IReadOnlyList<float>> queryVectors,
        IReadOnlyCollection<string> referenceExternalIds,
        int candidateK,
        CancellationToken ct = default)
        => BeginAsync(loadAll: false, queryVectors, referenceExternalIds, candidateK, ct);

    public Task<FaceIdentityTransaction> BeginFullAsync(CancellationToken ct = default)
        => BeginAsync(loadAll: true, [], [], 0, ct);

    private async Task<FaceIdentityTransaction> BeginAsync(
        bool loadAll,
        IReadOnlyList<IReadOnlyList<float>> queryVectors,
        IReadOnlyCollection<string> referenceExternalIds,
        int candidateK,
        CancellationToken ct)
    {
        await _gate.WaitAsync(ct);
        IServiceScope? scope = null;
        try
        {
            scope = _scopeFactory.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<DbContext>();

            await EnsureImportedAsync(db, ct);

            var identities = db.Set<ExtAiFacesIdentityEntity>();
            List<ExtAiFacesIdentityEntity> baseline;
            if (loadAll)
            {
                baseline = await identities.Include(item => item.Anchors).ToListAsync(ct);
            }
            else
            {
                var candidateIds = await ResolveCandidateIdentityIdsAsync(db, queryVectors, candidateK, ct);
                var referenceKeys = referenceExternalIds
                    .Where(static value => !string.IsNullOrWhiteSpace(value))
                    .Select(static value => value.Trim())
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToArray();

                baseline = candidateIds.Count == 0 && referenceKeys.Length == 0
                    ? []
                    : await identities
                        .Include(item => item.Anchors)
                        .Where(item => candidateIds.Contains(item.Id)
                            || (item.ReferenceExternalId != null && referenceKeys.Contains(item.ReferenceExternalId)))
                        .ToListAsync(ct);
            }

            // NextIdentityOrdinal must stay globally unique even though we only loaded a subset.
            var maxOrdinal = await identities.Select(item => (int?)item.Ordinal).MaxAsync(ct) ?? 0;

            var snapshot = new FaceIdentitySnapshot
            {
                NextIdentityOrdinal = maxOrdinal + 1,
                Identities = baseline.Select(ToStored).ToList(),
            };

            var transaction = new DbFaceIdentityTransaction(scope, db, _gate, baseline, snapshot);
            scope = null; // ownership transferred to the transaction
            return transaction;
        }
        catch
        {
            scope?.Dispose();
            _gate.Release();
            throw;
        }
    }

    public async Task DeleteByFaceKeyAsync(string faceKey, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(faceKey))
        {
            return;
        }

        await _gate.WaitAsync(ct);
        try
        {
            using var scope = _scopeFactory.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<DbContext>();
            // Anchors cascade via the FK.
            await db.Set<ExtAiFacesIdentityEntity>()
                .Where(item => item.FaceKey == faceKey)
                .ExecuteDeleteAsync(ct);
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task ClearAllAsync(CancellationToken ct = default)
    {
        await _gate.WaitAsync(ct);
        try
        {
            using var scope = _scopeFactory.CreateScope();
            var db = scope.ServiceProvider.GetRequiredService<DbContext>();
            // Delete anchors first (no cascade guarantee on a bulk identity delete across providers).
            await db.Set<ExtAiFacesIdentityAnchorEntity>().ExecuteDeleteAsync(ct);
            await db.Set<ExtAiFacesIdentityEntity>().ExecuteDeleteAsync(ct);
        }
        finally
        {
            _gate.Release();
        }
    }

    private static async Task<HashSet<int>> ResolveCandidateIdentityIdsAsync(
        DbContext db,
        IReadOnlyList<IReadOnlyList<float>> queryVectors,
        int candidateK,
        CancellationToken ct)
    {
        var ids = new HashSet<int>();
        var vectors = queryVectors
            .Where(static vector => vector is { Count: > 0 })
            .Select(static vector => vector.ToArray())
            .ToArray();
        if (vectors.Length == 0 || candidateK <= 0)
        {
            return ids;
        }

        var anchors = db.Set<ExtAiFacesIdentityAnchorEntity>();
        var usePgvector = db.Database.ProviderName?.Contains("Npgsql", StringComparison.Ordinal) == true;

        if (usePgvector)
        {
            // pgvector cosine-distance KNN (the `<=>` operator), same translation the host's
            // EmbeddingService uses. AsNoTracking keeps these throwaway rows out of the change tracker so
            // they don't collide with the tracked baseline loaded afterwards.
            foreach (var values in vectors)
            {
                var query = new Vector(values);
                var matches = await anchors
                    .AsNoTracking()
                    .OrderBy(anchor => anchor.Vector.CosineDistance(query))
                    .Take(candidateK)
                    .Select(anchor => anchor.IdentityId)
                    .ToListAsync(ct);
                ids.UnionWith(matches);
            }

            return ids;
        }

        // Provider fallback (e.g. tests): pull the (small) anchor table and rank in memory.
        var allAnchors = await anchors
            .Select(anchor => new { anchor.IdentityId, anchor.Vector })
            .ToListAsync(ct);
        foreach (var values in vectors)
        {
            var ranked = allAnchors
                .Select(anchor => new { anchor.IdentityId, Distance = CosineDistance(values, anchor.Vector.ToArray()) })
                .OrderBy(static item => item.Distance)
                .Take(candidateK)
                .Select(static item => item.IdentityId);
            ids.UnionWith(ranked);
        }

        return ids;
    }

    private static double CosineDistance(float[] left, float[] right)
    {
        var length = Math.Min(left.Length, right.Length);
        double dot = 0, leftNorm = 0, rightNorm = 0;
        for (var index = 0; index < length; index++)
        {
            dot += left[index] * right[index];
            leftNorm += left[index] * left[index];
            rightNorm += right[index] * right[index];
        }

        if (leftNorm <= 0 || rightNorm <= 0)
        {
            return 1.0;
        }

        return 1.0 - (dot / (Math.Sqrt(leftNorm) * Math.Sqrt(rightNorm)));
    }

    // One-time migration of the legacy `face-identity-snapshot` blob into the tables. Runs under the gate
    // on first reconcile; guarded by an empty identity table so it never resurrects data after a purge
    // (the blob is cleared once imported).
    private async Task EnsureImportedAsync(DbContext db, CancellationToken ct)
    {
        if (_importChecked)
        {
            return;
        }

        _importChecked = true;

        if (await db.Set<ExtAiFacesIdentityEntity>().AnyAsync(ct))
        {
            return;
        }

        var blob = await _legacyBlobStore.LoadAsync(ct);
        if (blob.Identities.Count == 0)
        {
            await _legacyBlobStore.ClearAsync(ct);
            return;
        }

        var now = DateTime.UtcNow;
        db.Set<ExtAiFacesIdentityEntity>().AddRange(blob.Identities.Select(stored => NewEntity(stored, now)));
        await db.SaveChangesAsync(ct);
        await _legacyBlobStore.ClearAsync(ct);
    }

    internal static ExtAiFacesIdentityEntity NewEntity(StoredFaceIdentity stored, DateTime now)
    {
        var entity = new ExtAiFacesIdentityEntity { CreatedAt = now, UpdatedAt = now };
        ApplyFields(entity, stored);
        entity.Anchors = stored.Anchors.Select(MapAnchor).ToList();
        return entity;
    }

    internal static void ApplyFields(ExtAiFacesIdentityEntity entity, StoredFaceIdentity stored)
    {
        entity.FaceKey = stored.FaceKey;
        entity.Ordinal = ParseOrdinal(stored.FaceKey);
        entity.Label = stored.Label;
        entity.LifecycleStatus = stored.LifecycleStatus;
        entity.PromotionReason = stored.PromotionReason;
        entity.ReferenceExternalId = stored.ReferenceExternalId;
        entity.ReferenceDisplayName = stored.ReferenceDisplayName;
        entity.ReferencePackId = stored.ReferencePackId;
        entity.ReferenceSuggestionId = stored.ReferenceSuggestionId;
        entity.QualityScore = stored.QualityScore;
        entity.CoverAssetId = stored.CoverAssetId;
        entity.CoverX1 = stored.CoverBoundingBox?.X1;
        entity.CoverY1 = stored.CoverBoundingBox?.Y1;
        entity.CoverX2 = stored.CoverBoundingBox?.X2;
        entity.CoverY2 = stored.CoverBoundingBox?.Y2;
        entity.CoverQualityScore = stored.CoverQualityScore;
        entity.ObservationCount = stored.ObservationCount;
        entity.AssetIdsJson = SerializeAssetIds(stored.AssetIds);
    }

    internal static ExtAiFacesIdentityAnchorEntity MapAnchor(StoredFaceAnchor anchor)
        => new()
        {
            ModelKey = anchor.ModelKey,
            QualityScore = anchor.QualityScore,
            Vector = new Vector(anchor.Vector.ToArray()),
        };

    // FaceKey is "face-{ordinal:0000}"; recover the ordinal for the global allocation column.
    private static int ParseOrdinal(string faceKey)
    {
        var dash = faceKey.LastIndexOf('-');
        return dash >= 0 && int.TryParse(faceKey.AsSpan(dash + 1), out var ordinal) ? ordinal : 0;
    }

    internal static StoredFaceIdentity ToStored(ExtAiFacesIdentityEntity entity)
        => new()
        {
            FaceKey = entity.FaceKey,
            Label = entity.Label,
            LifecycleStatus = entity.LifecycleStatus,
            PromotionReason = entity.PromotionReason,
            ReferenceExternalId = entity.ReferenceExternalId,
            ReferenceDisplayName = entity.ReferenceDisplayName,
            ReferencePackId = entity.ReferencePackId,
            ReferenceSuggestionId = entity.ReferenceSuggestionId,
            QualityScore = entity.QualityScore,
            CoverAssetId = entity.CoverAssetId,
            CoverBoundingBox = entity is { CoverX1: { } x1, CoverY1: { } y1, CoverX2: { } x2, CoverY2: { } y2 }
                ? new StoredBoundingBox(x1, y1, x2, y2)
                : null,
            CoverQualityScore = entity.CoverQualityScore,
            ObservationCount = entity.ObservationCount,
            AssetIds = DeserializeAssetIds(entity.AssetIdsJson),
            Anchors = entity.Anchors
                .Select(static anchor => new StoredFaceAnchor
                {
                    ModelKey = anchor.ModelKey,
                    QualityScore = anchor.QualityScore,
                    Vector = anchor.Vector.ToArray().ToList(),
                })
                .ToList(),
        };

    internal static List<string> DeserializeAssetIds(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return [];
        }

        try
        {
            return JsonSerializer.Deserialize<List<string>>(json) ?? [];
        }
        catch (JsonException)
        {
            return [];
        }
    }

    internal static string SerializeAssetIds(IReadOnlyList<string> assetIds)
        => JsonSerializer.Serialize(assetIds);
}
