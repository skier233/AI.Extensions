namespace AI.Faces;

/// <summary>
/// A pair of face identities that are known to be different people. Stored canonically (lexicographically
/// ordered, case-insensitive) so a pair is recorded once regardless of which side it was observed from.
/// </summary>
internal readonly record struct AiFaceIdentityExclusionPair(string LeftFaceKey, string RightFaceKey)
{
    public static AiFaceIdentityExclusionPair? TryCreate(string? leftFaceKey, string? rightFaceKey)
    {
        if (string.IsNullOrWhiteSpace(leftFaceKey)
            || string.IsNullOrWhiteSpace(rightFaceKey)
            || string.Equals(leftFaceKey, rightFaceKey, StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return string.Compare(leftFaceKey, rightFaceKey, StringComparison.OrdinalIgnoreCase) <= 0
            ? new AiFaceIdentityExclusionPair(leftFaceKey, rightFaceKey)
            : new AiFaceIdentityExclusionPair(rightFaceKey, leftFaceKey);
    }
}

/// <summary>
/// Symmetric "these identities cannot be the same person" lookup.
///
/// Two sources feed it, both far stronger evidence than any embedding-similarity threshold:
/// <list type="bullet">
/// <item>asset-local co-occurrence — two face clusters detected in the same frame are, by construction,
/// two people (the asset clusterer already refuses to merge them for this reason; before this the signal
/// was thrown away as soon as clusters were matched against the stored identity graph);</item>
/// <item>a user's explicit split decision, which must survive re-analysis.</item>
/// </list>
///
/// The set only ever <em>blocks</em> a merge or a match, so an empty instance reproduces the previous
/// behaviour exactly — which is what every caller without co-occurrence context (reference-pack backfill,
/// tests) passes.
/// </summary>
internal sealed class AiFaceIdentityExclusions
{
    public static readonly AiFaceIdentityExclusions Empty = new([]);

    private readonly Dictionary<string, HashSet<string>> _byFaceKey;

    private AiFaceIdentityExclusions(IEnumerable<AiFaceIdentityExclusionPair> pairs)
    {
        _byFaceKey = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
        foreach (var pair in pairs)
        {
            Link(pair.LeftFaceKey, pair.RightFaceKey);
            Link(pair.RightFaceKey, pair.LeftFaceKey);
        }
    }

    public bool IsEmpty => _byFaceKey.Count == 0;

    public IReadOnlyCollection<AiFaceIdentityExclusionPair> Pairs => _byFaceKey
        .SelectMany(entry => entry.Value.Select(other => AiFaceIdentityExclusionPair.TryCreate(entry.Key, other)))
        .Where(static pair => pair.HasValue)
        .Select(static pair => pair!.Value)
        .Distinct()
        .ToArray();

    public static AiFaceIdentityExclusions FromPairs(IEnumerable<AiFaceIdentityExclusionPair> pairs)
    {
        var materialized = pairs
            .Select(static pair => AiFaceIdentityExclusionPair.TryCreate(pair.LeftFaceKey, pair.RightFaceKey))
            .Where(static pair => pair.HasValue)
            .Select(static pair => pair!.Value)
            .ToArray();
        return materialized.Length == 0 ? Empty : new AiFaceIdentityExclusions(materialized);
    }

    /// <summary>True when the two identities are known to be different people.</summary>
    public bool Excludes(string? leftFaceKey, string? rightFaceKey)
        => !string.IsNullOrWhiteSpace(leftFaceKey)
           && !string.IsNullOrWhiteSpace(rightFaceKey)
           && _byFaceKey.TryGetValue(leftFaceKey, out var excluded)
           && excluded.Contains(rightFaceKey);

    public AiFaceIdentityExclusions With(IEnumerable<AiFaceIdentityExclusionPair> pairs)
        => FromPairs(Pairs.Concat(pairs));

    /// <summary>
    /// Rewrites face keys through an identity merge map so exclusions recorded against a merged-away
    /// identity keep applying to the identity it was folded into. Pairs that collapse onto themselves are
    /// dropped.
    /// </summary>
    public AiFaceIdentityExclusions Remap(IReadOnlyDictionary<string, string> mergedFaceKeyMap)
    {
        if (mergedFaceKeyMap.Count == 0 || IsEmpty)
        {
            return this;
        }

        return FromPairs(Pairs.Select(pair => new AiFaceIdentityExclusionPair(
            Resolve(pair.LeftFaceKey, mergedFaceKeyMap),
            Resolve(pair.RightFaceKey, mergedFaceKeyMap))));
    }

    private static string Resolve(string faceKey, IReadOnlyDictionary<string, string> mergedFaceKeyMap)
        => mergedFaceKeyMap.TryGetValue(faceKey, out var target) ? target : faceKey;

    private void Link(string faceKey, string otherFaceKey)
    {
        if (!_byFaceKey.TryGetValue(faceKey, out var excluded))
        {
            excluded = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            _byFaceKey[faceKey] = excluded;
        }

        excluded.Add(otherFaceKey);
    }
}

/// <summary>
/// Optional context for a reconciliation pass. Both members narrow behaviour that was previously
/// unconditional, and both default to "unknown", which reproduces the previous behaviour:
/// <list type="bullet">
/// <item><see cref="Exclusions"/> vetoes merges between identities proven to be different people;</item>
/// <item><see cref="CurrentAssetId"/> scopes the relaxed same-asset consolidation floors to the asset
/// actually being processed. Those relaxations exist to reunite one performer fragmented across several
/// clusters of the video being analysed; leaving them switched on for every later run meant two
/// co-performers who happened to share one old video could still be merged at the relaxed floor by an
/// unrelated asset's run, long after the co-occurrence evidence was gone.</item>
/// </list>
/// </summary>
internal sealed record AiFaceReconciliationContext(
    AiFaceIdentityExclusions Exclusions,
    string? CurrentAssetId)
{
    public static readonly AiFaceReconciliationContext Unscoped = new(AiFaceIdentityExclusions.Empty, null);

    public bool Excludes(StoredFaceIdentity left, StoredFaceIdentity right)
        => Exclusions.Excludes(left.FaceKey, right.FaceKey);

    /// <summary>
    /// Whether the relaxed same-asset consolidation floor may be used for this pair. Allowed when the
    /// caller did not scope the pass to an asset (whole-graph backfill, tests) or when both identities
    /// were observed in the asset being processed right now.
    /// </summary>
    public bool AllowsSameAssetRelaxation(StoredFaceIdentity left, StoredFaceIdentity right)
        => string.IsNullOrWhiteSpace(CurrentAssetId)
           || (Contains(left, CurrentAssetId) && Contains(right, CurrentAssetId));

    private static bool Contains(StoredFaceIdentity identity, string assetId)
        => identity.AssetIds.Any(value => string.Equals(value, assetId, StringComparison.OrdinalIgnoreCase));
}
