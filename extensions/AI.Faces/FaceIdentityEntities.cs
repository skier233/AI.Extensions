using Pgvector;

namespace AI.Faces;

// Persistence-only entities for the AI.Faces provisional/promoted identity graph. These replace the
// single serialized `face-identity-snapshot` blob with relational, pgvector-backed storage owned by the
// extension (created via IDataExtension.ConfigureModel + a raw ExtensionMigration). The reconcile/match
// logic continues to operate on the in-memory StoredFaceIdentity/StoredFaceAnchor model
// (see AiFaceIdentityState.cs); DbFaceIdentityStore maps between the two and persists only deltas.
internal sealed class ExtAiFacesIdentityEntity
{
    public int Id { get; set; }

    public string FaceKey { get; set; } = string.Empty;

    public int Ordinal { get; set; }

    public string? Label { get; set; }

    public string LifecycleStatus { get; set; } = StoredFaceIdentityLifecycle.Provisional;

    public string? PromotionReason { get; set; }

    public string? ReferenceExternalId { get; set; }

    public string? ReferenceDisplayName { get; set; }

    public string? ReferencePackId { get; set; }

    public int? ReferenceSuggestionId { get; set; }

    public double QualityScore { get; set; }

    public string? CoverAssetId { get; set; }

    // Cover bounding box stored as four nullable components (null when no cover is recorded).
    public double? CoverX1 { get; set; }

    public double? CoverY1 { get; set; }

    public double? CoverX2 { get; set; }

    public double? CoverY2 { get; set; }

    public double CoverQualityScore { get; set; }

    public int ObservationCount { get; set; }

    // JSON array of asset ids the identity has been observed in (bounded list, mirrors StoredFaceIdentity).
    public string AssetIdsJson { get; set; } = "[]";

    public DateTime CreatedAt { get; set; }

    public DateTime UpdatedAt { get; set; }

    public List<ExtAiFacesIdentityAnchorEntity> Anchors { get; set; } = [];
}

internal sealed class ExtAiFacesIdentityAnchorEntity
{
    public int Id { get; set; }

    public int IdentityId { get; set; }

    public string ModelKey { get; set; } = string.Empty;

    public double QualityScore { get; set; }

    // Face embedding anchor. Column type is `vector` (pgvector) via CoveContext.ConfigureVectorStorage;
    // similarity candidate lookup runs over this column.
    public Vector Vector { get; set; } = new(Array.Empty<float>());
}
