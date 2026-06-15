using System.Collections.Concurrent;

namespace AI.Core;

/// <summary>
/// Identifies which model listing endpoint a cached result belongs to.
/// </summary>
public enum AiModelCatalogCacheKind
{
    Catalog,
    Loaded,
}

/// <summary>
/// Caches the results of the (read-only) <c>/v4/models/catalog</c> and <c>/v4/models/loaded</c>
/// endpoints for a short window so that batch runs don't re-fetch them for every item.
/// </summary>
public interface IAiModelCatalogCache
{
    /// <summary>
    /// Returns a cached value if one is present and unexpired, otherwise invokes <paramref name="factory"/>
    /// and caches the result for <paramref name="ttl"/>. Concurrent callers for the same key share a single
    /// in-flight request. A non-positive <paramref name="ttl"/> bypasses the cache entirely.
    /// </summary>
    Task<IReadOnlyList<AiModelCatalogEntry>> GetOrCreateAsync(
        string serverBaseUrl,
        AiModelCatalogCacheKind kind,
        TimeSpan ttl,
        Func<CancellationToken, Task<IReadOnlyList<AiModelCatalogEntry>>> factory,
        CancellationToken ct);

    /// <summary>
    /// Drops every cached entry for a server. Call this after mutating the loaded model set
    /// (load/unload) so the next read reflects the new state immediately.
    /// </summary>
    void Invalidate(string serverBaseUrl);
}

public sealed class AiModelCatalogCache(TimeProvider timeProvider) : IAiModelCatalogCache
{
    private readonly TimeProvider _timeProvider = timeProvider;
    private readonly ConcurrentDictionary<string, CacheEntry> _entries = new(StringComparer.OrdinalIgnoreCase);

    public async Task<IReadOnlyList<AiModelCatalogEntry>> GetOrCreateAsync(
        string serverBaseUrl,
        AiModelCatalogCacheKind kind,
        TimeSpan ttl,
        Func<CancellationToken, Task<IReadOnlyList<AiModelCatalogEntry>>> factory,
        CancellationToken ct)
    {
        if (ttl <= TimeSpan.Zero)
        {
            return await factory(ct);
        }

        var key = BuildKey(serverBaseUrl, kind);

        while (true)
        {
            var now = _timeProvider.GetUtcNow();
            var entry = _entries.GetOrAdd(
                key,
                static (_, state) => new CacheEntry(
                    // Use a detached token so one caller cancelling doesn't poison the shared task.
                    new Lazy<Task<IReadOnlyList<AiModelCatalogEntry>>>(() => state.Factory(CancellationToken.None)),
                    state.Now + state.Ttl),
                (Factory: factory, Now: now, Ttl: ttl));

            if (entry.ExpiresAt <= now)
            {
                Remove(key, entry);
                continue;
            }

            try
            {
                return await entry.Value.Value.WaitAsync(ct);
            }
            catch
            {
                // Don't cache a failed fetch — evict so the next caller retries.
                Remove(key, entry);
                throw;
            }
        }
    }

    public void Invalidate(string serverBaseUrl)
    {
        foreach (var kind in Enum.GetValues<AiModelCatalogCacheKind>())
        {
            _entries.TryRemove(BuildKey(serverBaseUrl, kind), out _);
        }
    }

    private void Remove(string key, CacheEntry entry)
        => ((ICollection<KeyValuePair<string, CacheEntry>>)_entries).Remove(new KeyValuePair<string, CacheEntry>(key, entry));

    private static string BuildKey(string serverBaseUrl, AiModelCatalogCacheKind kind)
        => $"{serverBaseUrl}{kind}";

    private sealed record CacheEntry(Lazy<Task<IReadOnlyList<AiModelCatalogEntry>>> Value, DateTimeOffset ExpiresAt);
}
