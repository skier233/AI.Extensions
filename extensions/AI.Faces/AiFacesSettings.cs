using System.Text.Json;

using Cove.Plugins;

namespace AI.Faces;

internal sealed class AiFacesSettings
{
    public const double DefaultDetectionKeyframeIoUThreshold = 0.86;

    public const int DefaultMaxDetectionKeyframesPerTrack = 18;

    public const double DefaultDetectionKeyframeMaxGapSeconds = 2.5;

    // Cosine floor for attaching a track to an existing identity. Wrong-person absorption is far
    // harder to undo than a duplicate identity (duplicates auto-merge later; a polluted identity
    // attracts more of the wrong person), so this errs toward splitting.
    public double IdentityMatchThreshold { get; set; } = 0.50;

    public double IdentityAmbiguityMargin { get; set; } = 0.05;

    public double AssetClusterSimilarityThreshold { get; set; } = 0.48;

    public double AssetClusterAmbiguityMargin { get; set; } = 0.02;

    public double ReferenceMatchThreshold { get; set; } = 0.58;

    public double ReferenceAmbiguityMargin { get; set; } = 0.03;

    // A face must accumulate at least this many seconds of screen time in a video before it is marked
    // present (listed under "Faces in this video", given a timeline segment, etc.). Brief appearances
    // are usually a single mis-attributed detection lumped onto the wrong identity, or an incidental
    // cameo in an intro/outro clip — neither is worth surfacing as "this person is in this video".
    // For very short videos the effective floor is capped at half the duration so a legitimately short
    // clip still surfaces its main face. Images are a single frame and are never gated. Set to 0 to
    // disable.
    public double MinimumVideoFacePresenceSeconds { get; set; } = 8.0;

    public int PromotionMinimumVideoSamples { get; set; } = 24;

    public double PromotionMinimumVideoEvidenceSeconds { get; set; } = 24.0;

    public int PromotionMinimumSparseVideoSamples { get; set; } = 2;

    public double SparseVideoPromotionFrameIntervalSeconds { get; set; } = 10.0;

    public double PromotionMinimumSparseVideoSampleCoverageRatio { get; set; } = 0.10;

    public double ConsolidationSimilarityThreshold { get; set; } = 0.54;

    public double ConsolidationSameAssetSimilarityThreshold { get; set; } = 0.72;

    // Two identities that are both already promoted (user-visible Cove faces) and share no asset can
    // still be the same person split across different videos. They merge only above this stricter
    // floor, since merging visible faces is a heavier action than merging provisional state.
    public double ConsolidationPromotedSimilarityThreshold { get; set; } = 0.70;

    // When the user marks a face "not present" on a video/image, the face's other occurrences are
    // partitioned so the wrong-person ones can be split off. An occurrence joins the split-off bucket
    // only when its similarity to the marked exemplar clears this floor and it is clearly closer to the
    // exemplar than to the face's retained identity. Higher = more conservative (less likely to yank a
    // borderline-but-correct occurrence off the face).
    public double NotPresentSplitSimilarityThreshold { get; set; } = 0.50;

    public double ConsolidationAmbiguityMargin { get; set; } = 0.03;

    public double DetectionKeyframeIoUThreshold { get; set; } = DefaultDetectionKeyframeIoUThreshold;

    public int MaxDetectionKeyframesPerTrack { get; set; } = DefaultMaxDetectionKeyframesPerTrack;

    public double DetectionKeyframeMaxGapSeconds { get; set; } = DefaultDetectionKeyframeMaxGapSeconds;

    // When accepting a reference (metadata-server) face match that resolves to an existing local
    // performer, also scrape/pull that performer's metadata from the originating server to keep it up to
    // date. The originating site's remote id is recorded on the performer regardless of this setting.
    public bool UpdateExistingPerformersFromMetadataServers { get; set; } = true;

    public AiFacesSettings Normalize()
    {
        IdentityMatchThreshold = Math.Clamp(IdentityMatchThreshold, 0.0, 1.0);
        IdentityAmbiguityMargin = Math.Clamp(IdentityAmbiguityMargin, 0.0, 1.0);
        AssetClusterSimilarityThreshold = Math.Clamp(AssetClusterSimilarityThreshold, 0.0, 1.0);
        AssetClusterAmbiguityMargin = Math.Clamp(AssetClusterAmbiguityMargin, 0.0, 1.0);
        ReferenceMatchThreshold = Math.Clamp(ReferenceMatchThreshold, 0.0, 1.0);
        ReferenceAmbiguityMargin = Math.Clamp(ReferenceAmbiguityMargin, 0.0, 1.0);
        MinimumVideoFacePresenceSeconds = Math.Clamp(MinimumVideoFacePresenceSeconds, 0.0, 3600.0);
        PromotionMinimumVideoSamples = Math.Clamp(PromotionMinimumVideoSamples, 1, 1000);
        PromotionMinimumVideoEvidenceSeconds = Math.Clamp(PromotionMinimumVideoEvidenceSeconds, 0.0, 3600.0);
        PromotionMinimumSparseVideoSamples = Math.Clamp(PromotionMinimumSparseVideoSamples, 1, 1000);
        SparseVideoPromotionFrameIntervalSeconds = Math.Clamp(SparseVideoPromotionFrameIntervalSeconds, 1.0, 3600.0);
        PromotionMinimumSparseVideoSampleCoverageRatio = Math.Clamp(PromotionMinimumSparseVideoSampleCoverageRatio, 0.0, 1.0);
        ConsolidationSimilarityThreshold = Math.Clamp(ConsolidationSimilarityThreshold, 0.0, 1.0);
        ConsolidationSameAssetSimilarityThreshold = Math.Clamp(ConsolidationSameAssetSimilarityThreshold, 0.0, 1.0);
        ConsolidationPromotedSimilarityThreshold = Math.Clamp(ConsolidationPromotedSimilarityThreshold, 0.0, 1.0);
        NotPresentSplitSimilarityThreshold = Math.Clamp(NotPresentSplitSimilarityThreshold, 0.0, 1.0);
        ConsolidationAmbiguityMargin = Math.Clamp(ConsolidationAmbiguityMargin, 0.0, 1.0);
        DetectionKeyframeIoUThreshold = Math.Clamp(DetectionKeyframeIoUThreshold, 0.0, 1.0);
        MaxDetectionKeyframesPerTrack = Math.Clamp(MaxDetectionKeyframesPerTrack, 1, 60);
        DetectionKeyframeMaxGapSeconds = Math.Clamp(DetectionKeyframeMaxGapSeconds, 0.0, 60.0);
        return this;
    }
}

internal interface IAiFacesSettingsStore
{
    Task<AiFacesSettings> LoadAsync(CancellationToken ct = default);

    Task SaveAsync(AiFacesSettings settings, CancellationToken ct = default);
}

internal static class AiFacesSettingsRuntime
{
    private static IAiFacesSettingsStore _store = new NullAiFacesSettingsStore();

    public static void Attach(IAiFacesSettingsStore store)
    {
        _store = store ?? new NullAiFacesSettingsStore();
    }

    public static Task<AiFacesSettings> LoadAsync(CancellationToken ct = default)
        => _store.LoadAsync(ct);

    private sealed class NullAiFacesSettingsStore : IAiFacesSettingsStore
    {
        public Task<AiFacesSettings> LoadAsync(CancellationToken ct = default)
            => Task.FromResult(new AiFacesSettings().Normalize());

        public Task SaveAsync(AiFacesSettings settings, CancellationToken ct = default)
            => Task.CompletedTask;
    }
}

internal sealed class StoreBackedAiFacesSettingsStore : IAiFacesSettingsStore
{
    private const string StoreKey = "settings";

    private static readonly JsonSerializerOptions SerializerOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = false,
    };

    private IExtensionStore? _store;

    public void Attach(IExtensionStore store)
    {
        _store = store;
    }

    public async Task<AiFacesSettings> LoadAsync(CancellationToken ct = default)
    {
        if (_store is null)
        {
            return new AiFacesSettings().Normalize();
        }

        var payload = await _store.GetAsync(StoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
        {
            return new AiFacesSettings().Normalize();
        }

        var settings = JsonSerializer.Deserialize<AiFacesSettings>(payload, SerializerOptions) ?? new AiFacesSettings();
        return settings.Normalize();
    }

    public Task SaveAsync(AiFacesSettings settings, CancellationToken ct = default)
    {
        if (_store is null)
        {
            return Task.CompletedTask;
        }

        var normalized = settings.Normalize();
        var payload = JsonSerializer.Serialize(normalized, SerializerOptions);
        return _store.SetAsync(StoreKey, payload, ct);
    }
}