using System.Text.Json;

using Cove.Plugins;

namespace AI.Faces;

internal sealed class AiFaceReferenceSuggestionDecisionStore
{
    private const string StateStoreKey = "reference-suggestion-rejections";

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    private readonly SemaphoreSlim _gate = new(1, 1);

    private IExtensionStore? _store;

    public void Attach(IExtensionStore store)
    {
        _store = store;
    }

    public async Task<HashSet<string>> GetRejectedAsync(int faceId, CancellationToken ct = default)
    {
        await _gate.WaitAsync(ct);
        try
        {
            var state = await LoadStateAsync(ct);
            return state.RejectionsByFaceId.TryGetValue(faceId.ToString(), out var values)
                ? values.ToHashSet(StringComparer.OrdinalIgnoreCase)
                : [];
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task<IReadOnlyDictionary<int, HashSet<string>>> GetRejectedAsync(IReadOnlyCollection<int> faceIds, CancellationToken ct = default)
    {
        if (faceIds.Count == 0)
        {
            return new Dictionary<int, HashSet<string>>();
        }

        await _gate.WaitAsync(ct);
        try
        {
            var state = await LoadStateAsync(ct);
            var rejectedByFaceId = new Dictionary<int, HashSet<string>>();
            foreach (var faceId in faceIds.Where(static id => id > 0).Distinct())
            {
                rejectedByFaceId[faceId] = state.RejectionsByFaceId.TryGetValue(faceId.ToString(), out var values)
                    ? values.ToHashSet(StringComparer.OrdinalIgnoreCase)
                    : [];
            }

            return rejectedByFaceId;
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task RejectAsync(int faceId, string referenceIdentityId, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(referenceIdentityId);

        await _gate.WaitAsync(ct);
        try
        {
            var state = await LoadStateAsync(ct);
            var key = faceId.ToString();
            if (!state.RejectionsByFaceId.TryGetValue(key, out var values))
            {
                values = [];
                state.RejectionsByFaceId[key] = values;
            }

            if (!values.Contains(referenceIdentityId, StringComparer.OrdinalIgnoreCase))
                values.Add(referenceIdentityId);

            await SaveStateAsync(state, ct);
        }
        finally
        {
            _gate.Release();
        }
    }

    public async Task ClearAsync(int faceId, string referenceIdentityId, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(referenceIdentityId);

        await _gate.WaitAsync(ct);
        try
        {
            var state = await LoadStateAsync(ct);
            var key = faceId.ToString();
            if (!state.RejectionsByFaceId.TryGetValue(key, out var values))
                return;

            values.RemoveAll(value => string.Equals(value, referenceIdentityId, StringComparison.OrdinalIgnoreCase));
            if (values.Count == 0)
                state.RejectionsByFaceId.Remove(key);

            await SaveStateAsync(state, ct);
        }
        finally
        {
            _gate.Release();
        }
    }

    private async Task<AiFaceReferenceSuggestionDecisionState> LoadStateAsync(CancellationToken ct)
    {
        if (_store is null)
            return new AiFaceReferenceSuggestionDecisionState(new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase));

        var payload = await _store.GetAsync(StateStoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
            return new AiFaceReferenceSuggestionDecisionState(new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase));

        return JsonSerializer.Deserialize<AiFaceReferenceSuggestionDecisionState>(payload, JsonOptions)
               ?? new AiFaceReferenceSuggestionDecisionState(new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase));
    }

    private Task SaveStateAsync(AiFaceReferenceSuggestionDecisionState state, CancellationToken ct)
    {
        if (_store is null)
            return Task.CompletedTask;

        return _store.SetAsync(StateStoreKey, JsonSerializer.Serialize(state, JsonOptions), ct);
    }

    private sealed record AiFaceReferenceSuggestionDecisionState(Dictionary<string, List<string>> RejectionsByFaceId);
}