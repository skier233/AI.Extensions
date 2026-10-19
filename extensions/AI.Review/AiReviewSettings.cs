using System.Text.Json;

using Cove.Plugins;

namespace AI.Review;

public sealed record AiReviewSettings
{
    public const string DefaultSourceKey = "ext:ai.tagging";

    // Where exported frames and manifest.csv go. Empty = <extensions data dir>/<extension id>/export.
    public string ExportRootPath { get; init; } = string.Empty;

    // Width of the thumbnails shown in the review grid (exported frames are full resolution unless
    // ExportFullResolution is off, in which case this width is used for them as well).
    public int ThumbnailWidth { get; init; } = 480;

    public bool ExportFullResolution { get; init; } = true;

    // Upper bound on frames per segment. The planner picks 1 (short), 2 or 3 (long) frames up to this.
    public int MaxFramesPerSegment { get; init; } = 3;

    // Also export frames for true positives (not only false positives).
    public bool ExportTruePositives { get; init; } = true;

    // Delete a false-positive segment from Cove the moment it is marked, instead of waiting for "Apply".
    public bool DeleteOnMark { get; init; }

    // When the last AI segment of a tag on a video is deleted, also remove the AI tag application so the
    // tag stops showing on the video.
    public bool RemoveTagFromVideoWhenEmpty { get; init; } = true;

    // Which segment source is reviewed. AI.Tagging writes "ext:ai.tagging".
    public string SourceKey { get; init; } = DefaultSourceKey;

    public AiReviewSettings Normalize()
        => this with
        {
            ExportRootPath = (ExportRootPath ?? string.Empty).Trim(),
            ThumbnailWidth = Math.Clamp(ThumbnailWidth <= 0 ? 480 : ThumbnailWidth, 160, 1920),
            MaxFramesPerSegment = Math.Clamp(MaxFramesPerSegment <= 0 ? 3 : MaxFramesPerSegment, 1, 3),
            SourceKey = string.IsNullOrWhiteSpace(SourceKey) ? DefaultSourceKey : SourceKey.Trim(),
        };
}

internal static class AiReviewSettingsStore
{
    public const string SettingsStoreKey = "settings";

    private static readonly JsonSerializerOptions StateJson = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    public static async Task<AiReviewSettings> LoadAsync(IExtensionStore store, CancellationToken ct = default)
    {
        var payload = await store.GetAsync(SettingsStoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
        {
            return new AiReviewSettings().Normalize();
        }

        try
        {
            var settings = JsonSerializer.Deserialize<AiReviewSettings>(payload, StateJson) ?? new AiReviewSettings();
            return settings.Normalize();
        }
        catch (JsonException)
        {
            return new AiReviewSettings().Normalize();
        }
    }

    public static Task SaveAsync(IExtensionStore store, AiReviewSettings settings, CancellationToken ct = default)
        => store.SetAsync(SettingsStoreKey, JsonSerializer.Serialize(settings.Normalize(), StateJson), ct);
}

// Settings access for singleton services. The extension's IExtensionStore is only reachable from the
// extension instance, so it hands the services a delegate at startup.
internal sealed class AiReviewSettingsProvider
{
    private Func<CancellationToken, Task<AiReviewSettings>>? _loader;
    private AiReviewSettings _cached = new AiReviewSettings().Normalize();
    private DateTime _cachedAt = DateTime.MinValue;
    private static readonly TimeSpan CacheTtl = TimeSpan.FromSeconds(5);

    public void Attach(Func<CancellationToken, Task<AiReviewSettings>> loader) => _loader = loader;

    public void Invalidate() => _cachedAt = DateTime.MinValue;

    public async Task<AiReviewSettings> GetAsync(CancellationToken ct = default)
    {
        if (_loader is null)
        {
            return _cached;
        }

        if (DateTime.UtcNow - _cachedAt < CacheTtl)
        {
            return _cached;
        }

        _cached = await _loader(ct);
        _cachedAt = DateTime.UtcNow;
        return _cached;
    }
}
