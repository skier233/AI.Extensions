namespace AI.Extensions.Abstractions;

public sealed class AiPreparedArtifactBatch
{
    public List<AiPreparedTagLink> TagLinks { get; } = [];

    public List<AiPreparedSegment> Segments { get; } = [];

    public List<AiPreparedFaceAppearance> FaceAppearances { get; } = [];

    public List<AiPreparedDetection> Detections { get; } = [];

    public List<AiPreparedEmbedding> Embeddings { get; } = [];

    public List<AiPreparedFaceIdentity> Faces { get; } = [];

    public List<string> DeferredWorkItems { get; } = [];

    public List<string> Notes { get; } = [];
}

public sealed record AiPreparedTagLink(
    string AssetId,
    string SourceKey,
    string TagName,
    double? Confidence = null,
    string? ModelKey = null,
    string? MediaKind = null,
    IReadOnlyDictionary<string, string>? Metadata = null
);

public sealed record AiPreparedSegment(
    string AssetId,
    string SourceKey,
    string Kind,
    double StartSeconds,
    double? EndSeconds = null,
    string? TagName = null,
    string? Title = null,
    double? Confidence = null,
    string? RefKind = null,
    string? RefKey = null,
    IReadOnlyDictionary<string, string>? Metadata = null
);

public sealed record AiPreparedFaceAppearance(
    string AssetId,
    string SourceKey,
    int SampleCount,
    int RetainedSpatialSampleCount,
    int SegmentCount = 0,
    double? FirstSeenSeconds = null,
    double? LastSeenSeconds = null,
    double? TopConfidence = null,
    double? RepresentativeFrameSeconds = null,
    string? RefKind = null,
    string? RefKey = null,
    string? GroupKey = null,
    IReadOnlyDictionary<string, string>? Metadata = null
);

public sealed record AiPreparedDetection(
    string AssetId,
    string SourceKey,
    string Class,
    double? ObservedAtSeconds,
    double Score,
    AiBoundingBox BoundingBox,
    string? ModelKey = null,
    string? RefKind = null,
    string? RefKey = null,
    string? GroupKey = null,
    IReadOnlyDictionary<string, string>? Metadata = null
);

public sealed record AiPreparedEmbedding(
    string AssetId,
    string SourceKey,
    string Kind,
    string KindFamily,
    string Modality,
    bool IsSemantic,
    IReadOnlyList<float> Vector,
    double? Norm = null,
    string? HostRefKind = null,
    string? HostRefKey = null,
    int SectionIndex = 0,
    double? StartSeconds = null,
    double? EndSeconds = null,
    string? ModelKey = null,
    IReadOnlyDictionary<string, string>? Metadata = null
);

public sealed record AiPreparedFaceIdentity(
    string FaceKey,
    string SourceKey,
    string? Label = null,
    bool IsProvisional = true,
    double? QualityScore = null,
    string? CoverAssetId = null,
    AiBoundingBox? CoverBoundingBox = null,
    IReadOnlyDictionary<string, string>? Metadata = null
);
