namespace AI.Faces;

internal static class StoredFaceIdentityLifecycle
{
    public const string Provisional = "provisional";

    public const string Promoted = "promoted";
}

internal sealed class FaceIdentitySnapshot
{
    public int NextIdentityOrdinal { get; set; } = 1;

    public List<StoredFaceIdentity> Identities { get; set; } = [];
}

internal sealed class StoredFaceIdentity
{
    public string FaceKey { get; set; } = string.Empty;

    public string? Label { get; set; }

    public string LifecycleStatus { get; set; } = StoredFaceIdentityLifecycle.Provisional;

    public string? PromotionReason { get; set; }

    public string? ReferenceExternalId { get; set; }

    public string? ReferenceDisplayName { get; set; }

    public string? ReferencePackId { get; set; }

    public int? ReferenceSuggestionId { get; set; }

    public double QualityScore { get; set; }

    public string? CoverAssetId { get; set; }

    public StoredBoundingBox? CoverBoundingBox { get; set; }

    public double CoverQualityScore { get; set; }

    public int ObservationCount { get; set; }

    public List<string> AssetIds { get; set; } = [];

    public List<StoredFaceAnchor> Anchors { get; set; } = [];
}

internal sealed class StoredFaceAnchor
{
    public string ModelKey { get; set; } = string.Empty;

    public double QualityScore { get; set; }

    public List<float> Vector { get; set; } = [];
}

internal sealed record StoredBoundingBox(double X1, double Y1, double X2, double Y2);
