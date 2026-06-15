using System.Text.Json;

using Cove.Plugins;

namespace AI.Faces;

internal interface IFaceIdentityStateStore
{
    Task<FaceIdentitySnapshot> LoadAsync(CancellationToken ct = default);

    Task SaveAsync(FaceIdentitySnapshot snapshot, CancellationToken ct = default);

    Task DeleteAsync(string faceKey, CancellationToken ct = default);

    Task ClearAsync(CancellationToken ct = default);
}

internal sealed class StoreBackedFaceIdentityStateStore : IFaceIdentityStateStore
{
    private const string StoreKey = "face-identity-snapshot";

    private static readonly JsonSerializerOptions SerializerOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false,
    };

    private IExtensionStore? _store;

    public void Attach(IExtensionStore store)
    {
        _store = store;
    }

    public async Task<FaceIdentitySnapshot> LoadAsync(CancellationToken ct = default)
    {
        if (_store is null)
        {
            return new FaceIdentitySnapshot();
        }

        var payload = await _store.GetAsync(StoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
        {
            return new FaceIdentitySnapshot();
        }

        return JsonSerializer.Deserialize<FaceIdentitySnapshot>(payload, SerializerOptions) ?? new FaceIdentitySnapshot();
    }

    public Task SaveAsync(FaceIdentitySnapshot snapshot, CancellationToken ct = default)
    {
        if (_store is null)
        {
            return Task.CompletedTask;
        }

        var payload = JsonSerializer.Serialize(snapshot, SerializerOptions);
        return _store.SetAsync(StoreKey, payload, ct);
    }

    public async Task DeleteAsync(string faceKey, CancellationToken ct = default)
    {
        if (_store is null || string.IsNullOrWhiteSpace(faceKey))
        {
            return;
        }

        var snapshot = await LoadAsync(ct);
        var removed = snapshot.Identities.RemoveAll(identity => string.Equals(identity.FaceKey, faceKey, StringComparison.Ordinal));
        if (removed == 0)
        {
            return;
        }

        await SaveAsync(snapshot, ct);
    }

    // Drops the legacy blob once its contents have been migrated into the relational store.
    public Task ClearAsync(CancellationToken ct = default)
        => _store is null ? Task.CompletedTask : _store.DeleteAsync(StoreKey, ct);
}
