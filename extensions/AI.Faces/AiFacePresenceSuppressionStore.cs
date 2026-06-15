using System.Text.Json;

using Cove.Plugins;

namespace AI.Faces;

internal sealed record AiFacePresenceSuppression(string FaceKey, string HostType, int HostId);

/// <summary>
/// Durable record of "this face is not present on this host" decisions made by the user via the
/// not-present action. Persistence consults it so a future AI re-run of the host does not re-attach a
/// face the user has explicitly removed from it. Stored in its own extension-store key, independent of
/// the face-identity state, so the negative signal survives even as that state changes.
/// </summary>
internal sealed class AiFacePresenceSuppressionStore
{
    private const string StoreKey = "presence-suppressions";

    private static readonly JsonSerializerOptions SerializerOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false,
    };

    private IExtensionStore? _store;

    public void Attach(IExtensionStore store) => _store = store;

    public async Task<IReadOnlyList<AiFacePresenceSuppression>> LoadAsync(CancellationToken ct = default)
    {
        if (_store is null)
        {
            return [];
        }

        var payload = await _store.GetAsync(StoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
        {
            return [];
        }

        return JsonSerializer.Deserialize<List<AiFacePresenceSuppression>>(payload, SerializerOptions) ?? [];
    }

    /// <summary>Face keys the user has marked not-present on the given host. Empty when none.</summary>
    public async Task<IReadOnlySet<string>> GetSuppressedFaceKeysAsync(string hostType, int hostId, CancellationToken ct = default)
    {
        var normalizedHostType = NormalizeHostType(hostType);
        var all = await LoadAsync(ct);
        return all
            .Where(entry => entry.HostId == hostId && string.Equals(NormalizeHostType(entry.HostType), normalizedHostType, StringComparison.OrdinalIgnoreCase))
            .Select(entry => entry.FaceKey)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
    }

    public async Task AddAsync(IEnumerable<AiFacePresenceSuppression> suppressions, CancellationToken ct = default)
    {
        if (_store is null)
        {
            return;
        }

        var incoming = suppressions
            .Where(entry => !string.IsNullOrWhiteSpace(entry.FaceKey))
            .Select(entry => entry with { HostType = NormalizeHostType(entry.HostType) })
            .ToArray();
        if (incoming.Length == 0)
        {
            return;
        }

        var existing = (await LoadAsync(ct)).ToList();
        var seen = existing
            .Select(Key)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var changed = false;
        foreach (var entry in incoming)
        {
            if (seen.Add(Key(entry)))
            {
                existing.Add(entry);
                changed = true;
            }
        }

        if (!changed)
        {
            return;
        }

        await _store.SetAsync(StoreKey, JsonSerializer.Serialize(existing, SerializerOptions), ct);
    }

    private static string Key(AiFacePresenceSuppression entry)
        => $"{entry.FaceKey}{NormalizeHostType(entry.HostType)}{entry.HostId}";

    private static string NormalizeHostType(string hostType)
        => hostType.Trim().ToLowerInvariant() switch
        {
            "video" or "videos" => "video",
            "image" or "images" => "image",
            var other => other,
        };
}
