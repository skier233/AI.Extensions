using System.Text.Json;

using Cove.Plugins;

namespace AI.Faces;

internal sealed class AiFacesSettings
{
    public const double DefaultDetectionKeyframeIoUThreshold = 0.78;

    public const int DefaultMaxDetectionKeyframesPerTrack = 5;

    public double MinimumPoseQuality { get; set; } = 0.78;

    public double MinimumImageQuality { get; set; } = 0.22;

    public double IdentityMatchThreshold { get; set; } = 0.42;

    public double IdentityAmbiguityMargin { get; set; } = 0.02;

    public double AssetClusterSimilarityThreshold { get; set; } = 0.48;

    public double AssetClusterAmbiguityMargin { get; set; } = 0.02;

    public double ReferenceMatchThreshold { get; set; } = 0.58;

    public double ReferenceAmbiguityMargin { get; set; } = 0.03;

    public int PromotionMinimumVideoSamples { get; set; } = 24;

    public double PromotionMinimumVideoEvidenceSeconds { get; set; } = 24.0;

    public int PromotionMinimumSparseVideoSamples { get; set; } = 2;

    public double SparseVideoPromotionFrameIntervalSeconds { get; set; } = 10.0;

    public double PromotionMinimumSparseVideoSampleCoverageRatio { get; set; } = 0.10;

    public double ConsolidationSimilarityThreshold { get; set; } = 0.54;

    public double ConsolidationSameAssetSimilarityThreshold { get; set; } = 0.72;

    public double ConsolidationAmbiguityMargin { get; set; } = 0.03;

    public double DetectionKeyframeIoUThreshold { get; set; } = DefaultDetectionKeyframeIoUThreshold;

    public int MaxDetectionKeyframesPerTrack { get; set; } = DefaultMaxDetectionKeyframesPerTrack;

    public AiFacesSettings Normalize()
    {
        MinimumPoseQuality = Math.Clamp(MinimumPoseQuality, 0.0, 1.0);
        MinimumImageQuality = Math.Clamp(MinimumImageQuality, 0.0, 1.0);
        IdentityMatchThreshold = Math.Clamp(IdentityMatchThreshold, 0.0, 1.0);
        IdentityAmbiguityMargin = Math.Clamp(IdentityAmbiguityMargin, 0.0, 1.0);
        AssetClusterSimilarityThreshold = Math.Clamp(AssetClusterSimilarityThreshold, 0.0, 1.0);
        AssetClusterAmbiguityMargin = Math.Clamp(AssetClusterAmbiguityMargin, 0.0, 1.0);
        ReferenceMatchThreshold = Math.Clamp(ReferenceMatchThreshold, 0.0, 1.0);
        ReferenceAmbiguityMargin = Math.Clamp(ReferenceAmbiguityMargin, 0.0, 1.0);
        PromotionMinimumVideoSamples = Math.Clamp(PromotionMinimumVideoSamples, 1, 1000);
        PromotionMinimumVideoEvidenceSeconds = Math.Clamp(PromotionMinimumVideoEvidenceSeconds, 0.0, 3600.0);
        PromotionMinimumSparseVideoSamples = Math.Clamp(PromotionMinimumSparseVideoSamples, 1, 1000);
        SparseVideoPromotionFrameIntervalSeconds = Math.Clamp(SparseVideoPromotionFrameIntervalSeconds, 1.0, 3600.0);
        PromotionMinimumSparseVideoSampleCoverageRatio = Math.Clamp(PromotionMinimumSparseVideoSampleCoverageRatio, 0.0, 1.0);
        ConsolidationSimilarityThreshold = Math.Clamp(ConsolidationSimilarityThreshold, 0.0, 1.0);
        ConsolidationSameAssetSimilarityThreshold = Math.Clamp(ConsolidationSameAssetSimilarityThreshold, 0.0, 1.0);
        ConsolidationAmbiguityMargin = Math.Clamp(ConsolidationAmbiguityMargin, 0.0, 1.0);
        DetectionKeyframeIoUThreshold = Math.Clamp(DetectionKeyframeIoUThreshold, 0.0, 1.0);
        MaxDetectionKeyframesPerTrack = Math.Clamp(MaxDetectionKeyframesPerTrack, 1, 25);
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