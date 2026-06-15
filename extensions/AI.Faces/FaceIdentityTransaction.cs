using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Faces;

// Unit-of-work returned by IFaceIdentityStore. The caller mutates Snapshot (the in-memory model the
// reconciler operates on) and calls CommitAsync to persist the diff; disposing releases any resources
// (the reconcile gate, DI scope) held by the implementation. Abstract so tests can supply an in-memory
// transaction without a DbContext.
internal abstract class FaceIdentityTransaction : IAsyncDisposable
{
    public abstract FaceIdentitySnapshot Snapshot { get; }

    public abstract Task CommitAsync(CancellationToken ct = default);

    public abstract ValueTask DisposeAsync();
}

// DbContext-backed transaction: holds the loaded baseline (tracked entities) and the working snapshot;
// CommitAsync persists only the diff. Disposing releases the reconcile gate and the DI scope.
internal sealed class DbFaceIdentityTransaction(
    IServiceScope scope,
    DbContext db,
    SemaphoreSlim gate,
    IReadOnlyList<ExtAiFacesIdentityEntity> baseline,
    FaceIdentitySnapshot snapshot) : FaceIdentityTransaction
{
    private readonly IServiceScope _scope = scope;
    private readonly DbContext _db = db;
    private readonly SemaphoreSlim _gate = gate;
    private readonly IReadOnlyList<ExtAiFacesIdentityEntity> _baseline = baseline;
    private bool _committed;
    private bool _disposed;

    public override FaceIdentitySnapshot Snapshot { get; } = snapshot;

    public override async Task CommitAsync(CancellationToken ct = default)
    {
        if (_committed)
        {
            return;
        }

        var now = DateTime.UtcNow;
        var identities = _db.Set<ExtAiFacesIdentityEntity>();
        var baselineByKey = _baseline.ToDictionary(static item => item.FaceKey, StringComparer.Ordinal);
        var snapshotKeys = new HashSet<string>(StringComparer.Ordinal);

        foreach (var stored in Snapshot.Identities)
        {
            snapshotKeys.Add(stored.FaceKey);

            if (baselineByKey.TryGetValue(stored.FaceKey, out var entity))
            {
                DbFaceIdentityStore.ApplyFields(entity, stored);
                entity.UpdatedAt = now;
                // Replace anchors on the tracked collection: cleared rows are deleted as orphans (the FK
                // is required + cascade), and the re-added rows are inserted (≤12 per identity).
                entity.Anchors.Clear();
                foreach (var anchor in stored.Anchors)
                {
                    entity.Anchors.Add(DbFaceIdentityStore.MapAnchor(anchor));
                }
            }
            else
            {
                identities.Add(DbFaceIdentityStore.NewEntity(stored, now));
            }
        }

        foreach (var entity in _baseline)
        {
            if (!snapshotKeys.Contains(entity.FaceKey))
            {
                // Merged away during reconcile; cascade removes its anchors.
                identities.Remove(entity);
            }
        }

        await _db.SaveChangesAsync(ct);
        _committed = true;
    }

    public override async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _scope.Dispose();
        _gate.Release();
        await ValueTask.CompletedTask;
    }
}
