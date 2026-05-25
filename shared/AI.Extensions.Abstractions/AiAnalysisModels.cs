namespace AI.Extensions.Abstractions;

public sealed record AiRunContext(
    string RunId,
    string MediaKind,
    string AssetId,
    string Subject,
    string? HostEntityType = null,
    int? HostEntityId = null,
    double? DurationSeconds = null,
    double? FrameIntervalSeconds = null,
    IReadOnlyDictionary<string, string>? Metadata = null
);

public sealed class AiAnalyzeResult
{
    public string MediaKind { get; init; } = string.Empty;

    public string AssetId { get; init; } = string.Empty;

    public double? DurationSeconds { get; init; }

    public double? FrameIntervalSeconds { get; init; }

    public IReadOnlyList<AiModelDescriptor> Models { get; init; } = [];

    public IReadOnlyList<string> RequestedModelNames { get; init; } = [];

    public AiAnalysisNode? AssetAnalysis { get; init; }

    public IReadOnlyList<AiTemporalSlice> Frames { get; init; } = [];

    public IReadOnlyList<AiTemporalSlice> Windows { get; init; } = [];

    public IReadOnlyDictionary<string, double> Metrics { get; init; } = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
}

public sealed record AiModelDescriptor(
    string ConfigName,
    string Name,
    string? Type = null,
    IReadOnlyList<string>? Capabilities = null,
    IReadOnlyList<string>? SupportedScopes = null,
    IReadOnlyList<string>? Categories = null,
    string? Version = null,
    bool Active = false,
    bool Loaded = false
);

public sealed record AiTemporalSlice(
    string SliceKind,
    int? Index,
    double? TimeSeconds,
    double? StartSeconds,
    double? EndSeconds,
    AiAnalysisNode Analysis
);

public sealed class AiAnalysisNode
{
    public IReadOnlyList<AiTagPrediction> Tags { get; init; } = [];

    public IReadOnlyList<AiClassificationPrediction> Classifications { get; init; } = [];

    public IReadOnlyList<AiDetectionObservation> Detections { get; init; } = [];

    public IReadOnlyList<AiEmbeddingObservation> Embeddings { get; init; } = [];

    public IReadOnlyList<AiRegionBranch> RegionBranches { get; init; } = [];

    public IReadOnlyDictionary<string, string> Other { get; init; } = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
}

public sealed record AiTagPrediction(
    string ModelKey,
    string Tag,
    double? Confidence
);

public sealed record AiClassificationPrediction(
    string ModelKey,
    string Label,
    double? Confidence
);

public readonly record struct AiBoundingBox(
    double X1,
    double Y1,
    double X2,
    double Y2
)
{
    public double Width => Math.Max(0.0, X2 - X1);

    public double Height => Math.Max(0.0, Y2 - Y1);

    public double Area => Width * Height;

    public double CenterX => X1 + (Width / 2.0);

    public double CenterY => Y1 + (Height / 2.0);
}

public sealed record AiDetectionObservation(
    string ModelKey,
    int DetectionIndex,
    string Label,
    double Score,
    AiBoundingBox BoundingBox,
    IReadOnlyDictionary<string, string>? Metadata = null
);

public sealed record AiEmbeddingObservation(
    string ModelKey,
    string Scope,
    IReadOnlyList<float> Vector,
    double? Norm = null,
    int? DetectionIndex = null,
    string? SourceBranchKey = null,
    IReadOnlyDictionary<string, string>? Metadata = null
);

public sealed record AiRegionBranch(
    string BranchKey,
    int? DetectionIndex,
    AiAnalysisNode Analysis
);
