using System.Text.Json;

using Cove.Plugins;

namespace AI.Faces;

/// <summary>
/// Durable record of identity pairs the user has explicitly pulled apart (see
/// <see cref="AiFaceSplitService"/>). Kept in its own extension-store key, independent of the identity
/// graph, so the decision survives re-analysis, identity merges, and pack imports — exactly like the
/// not-present suppressions in <see cref="AiFacePresenceSuppressionStore"/>.
///
/// Only user decisions live here. Automatic co-occurrence evidence is recomputed from the asset's own
/// frames on every run and never persisted, so this stays small (bounded by explicit corrections).
/// </summary>
internal sealed class AiFaceIdentityExclusionStore
{
    private const string StoreKey = "identity-exclusions";

    private static readonly JsonSerializerOptions SerializerOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false,
    };

    private IExtensionStore? _store;

    public void Attach(IExtensionStore store) => _store = store;

    public async Task<AiFaceIdentityExclusions> LoadAsync(CancellationToken ct = default)
    {
        if (_store is null)
        {
            return AiFaceIdentityExclusions.Empty;
        }

        var payload = await _store.GetAsync(StoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
        {
            return AiFaceIdentityExclusions.Empty;
        }

        var entries = JsonSerializer.Deserialize<List<StoredExclusion>>(payload, SerializerOptions) ?? [];
        return AiFaceIdentityExclusions.FromPairs(entries
            .Select(static entry => AiFaceIdentityExclusionPair.TryCreate(entry.LeftFaceKey, entry.RightFaceKey))
            .Where(static pair => pair.HasValue)
            .Select(static pair => pair!.Value));
    }

    public async Task AddAsync(IEnumerable<AiFaceIdentityExclusionPair> pairs, CancellationToken ct = default)
    {
        if (_store is null)
        {
            return;
        }

        var incoming = pairs
            .Select(static pair => AiFaceIdentityExclusionPair.TryCreate(pair.LeftFaceKey, pair.RightFaceKey))
            .Where(static pair => pair.HasValue)
            .Select(static pair => pair!.Value)
            .ToArray();
        if (incoming.Length == 0)
        {
            return;
        }

        var existing = await LoadAsync(ct);
        var combined = existing.With(incoming);
        if (combined.Pairs.Count == existing.Pairs.Count)
        {
            return;
        }

        await SaveAsync(combined, ct);
    }

    /// <summary>
    /// Rewrites stored pairs through an identity merge map. Called after reconciliation so an exclusion
    /// recorded against an identity that has since been folded into another keeps protecting the survivor.
    /// </summary>
    public async Task RemapAsync(IReadOnlyDictionary<string, string> mergedFaceKeyMap, CancellationToken ct = default)
    {
        if (_store is null || mergedFaceKeyMap.Count == 0)
        {
            return;
        }

        var existing = await LoadAsync(ct);
        if (existing.IsEmpty)
        {
            return;
        }

        var remapped = existing.Remap(mergedFaceKeyMap);
        if (remapped.Pairs.SequenceEqual(existing.Pairs))
        {
            return;
        }

        await SaveAsync(remapped, ct);
    }

    /// <summary>Drops every pair naming a face key, for when that face is deleted.</summary>
    public async Task RemoveAsync(string faceKey, CancellationToken ct = default)
    {
        if (_store is null || string.IsNullOrWhiteSpace(faceKey))
        {
            return;
        }

        var existing = await LoadAsync(ct);
        var retained = existing.Pairs
            .Where(pair => !string.Equals(pair.LeftFaceKey, faceKey, StringComparison.OrdinalIgnoreCase)
                && !string.Equals(pair.RightFaceKey, faceKey, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        if (retained.Length == existing.Pairs.Count)
        {
            return;
        }

        await SaveAsync(AiFaceIdentityExclusions.FromPairs(retained), ct);
    }

    /// <summary>
    /// Forgets every split decision. Only for a full purge of this extension's face data: identity
    /// ordinals restart from 1 afterwards, so stale pairs would otherwise apply to unrelated new faces
    /// that happen to reuse the same keys.
    /// </summary>
    public Task ClearAsync(CancellationToken ct = default)
        => _store is null ? Task.CompletedTask : _store.DeleteAsync(StoreKey, ct);

    private Task SaveAsync(AiFaceIdentityExclusions exclusions, CancellationToken ct)
    {
        var payload = JsonSerializer.Serialize(
            exclusions.Pairs.Select(static pair => new StoredExclusion(pair.LeftFaceKey, pair.RightFaceKey)).ToArray(),
            SerializerOptions);
        return _store!.SetAsync(StoreKey, payload, ct);
    }

    private sealed record StoredExclusion(string LeftFaceKey, string RightFaceKey);
}
