namespace AI.Faces;

internal sealed class AiFaceIdentityReconciler
{
    private const double DuplicateAnchorSimilarity = 0.975;
    private const int MaxAnchorsPerIdentity = 12;
    private const int MaxAssetIdsPerIdentity = 32;

    // applyReferenceMatches re-scores every (still-unreferenced) identity's anchors against every
    // performer in the reference pack — O(identities × anchors × packPerformers × dims), the dominant
    // per-image cost when a large pack is loaded. In the incremental per-asset path it is pure waste (the
    // loaded candidates were already reference-matched when created, and this image's own faces are
    // reference-matched in the assignment loop), so the caller disables it there. Bulk re-matching
    // (e.g. on pack import) still passes true via the backfill path.
    public AiFaceIdentityReconciliationReport Reconcile(
        FaceIdentitySnapshot snapshot,
        SaieReferencePack? referencePack,
        AiFacesSettings settings,
        bool applyReferenceMatches = true,
        AiFaceReconciliationContext? context = null)
    {
        var resolvedContext = context ?? AiFaceReconciliationContext.Unscoped;
        var referencePromotions = applyReferenceMatches ? ApplyReferenceMatches(snapshot, referencePack, settings) : 0;
        var mergedFaceKeyMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var merges = MergeSimilarIdentities(snapshot, settings, resolvedContext, mergedFaceKeyMap);
        var evidencePromotions = PromoteEvidenceBackedIdentities(snapshot);

        return new AiFaceIdentityReconciliationReport(
            merges,
            referencePromotions,
            evidencePromotions,
            mergedFaceKeyMap);
    }

    private static int ApplyReferenceMatches(FaceIdentitySnapshot snapshot, SaieReferencePack? referencePack, AiFacesSettings settings)
    {
        if (referencePack is null || referencePack.Identities.Count == 0)
        {
            return 0;
        }

        var promoted = 0;
        foreach (var identity in snapshot.Identities)
        {
            if (!string.IsNullOrWhiteSpace(identity.ReferenceExternalId) || identity.Anchors.Count == 0)
            {
                continue;
            }

            var referenceMatch = TryMatchReference(identity, referencePack, settings);
            if (referenceMatch is null)
            {
                continue;
            }

            identity.ReferenceExternalId = referenceMatch.Identity.ExternalId;
            identity.ReferenceDisplayName = referenceMatch.Identity.DisplayName;
            identity.ReferencePackId = referenceMatch.PackId;
            identity.ReferenceSuggestionId = referenceMatch.SuggestionId;
            if (string.IsNullOrWhiteSpace(identity.Label))
            {
                identity.Label = referenceMatch.Identity.DisplayName;
            }

            if (!IsPromoted(identity))
            {
                identity.LifecycleStatus = StoredFaceIdentityLifecycle.Promoted;
                identity.PromotionReason = "reference";
                promoted++;
            }
        }

        return promoted;
    }

    // Semantically this is "scan the identities in order, apply the first merge found, start over" until
    // a full scan finds nothing. Starting over literally re-scored every pair after every merge, which made
    // a run cost (merges + 1) full O(n^2) passes. A merge only changes the surviving identity, so a source
    // that was scanned and found nothing to merge stays settled until the merge touches something its
    // verdict depended on (see SettledSource); only unsettled sources are re-scored on the next scan.
    private static int MergeSimilarIdentities(
        FaceIdentitySnapshot snapshot,
        AiFacesSettings settings,
        AiFaceReconciliationContext context,
        Dictionary<string, string> mergedFaceKeyMap)
    {
        var mergeCount = 0;
        var margin = settings.ConsolidationAmbiguityMargin;
        var vectors = new FaceAnchorVectorCache();
        var settled = new Dictionary<StoredFaceIdentity, SettledSource>(ReferenceEqualityComparer.Instance);
        var scored = new List<(StoredFaceIdentity Identity, double Score)>();
        var merged = true;
        while (merged)
        {
            merged = false;
            foreach (var source in snapshot.Identities.ToArray())
            {
                if (source.Anchors.Count == 0 || settled.ContainsKey(source))
                {
                    continue;
                }

                // Best candidate is the first highest-scoring one in snapshot order; the runner-up is the
                // highest score among the rest (equal to the best on a tie).
                var sourceVectors = vectors.Get(source);
                StoredFaceIdentity? best = null;
                var bestScore = 0.0;
                var secondBestScore = 0.0;
                scored.Clear();
                foreach (var candidate in snapshot.Identities)
                {
                    if (ReferenceEquals(candidate, source) || !CanConsiderMerge(source, candidate, context))
                    {
                        continue;
                    }

                    var score = ScoreAnchors(sourceVectors, vectors.Get(candidate));
                    if (score <= 0.0)
                    {
                        continue;
                    }

                    scored.Add((candidate, score));
                    if (best is null || score > bestScore)
                    {
                        secondBestScore = best is null ? 0.0 : bestScore;
                        best = candidate;
                        bestScore = score;
                    }
                    else if (score > secondBestScore)
                    {
                        secondBestScore = score;
                    }
                }

                var mergeable = best is not null
                    && bestScore >= ResolveMergeThreshold(source, best, settings, context);
                if (mergeable && (bestScore - secondBestScore) < margin)
                {
                    // With three or more identities of the same person, every candidate scores within
                    // the margin of every other — a plain margin check deadlocks all merges exactly
                    // when duplicates are most numerous. Ambiguity only blocks when an in-margin rival
                    // is plausibly a *different* person from the best candidate, i.e. not itself
                    // mergeable with it.
                    var bestVectors = vectors.Get(best!);
                    foreach (var rival in scored)
                    {
                        if (ReferenceEquals(rival.Identity, best) || (bestScore - rival.Score) >= margin)
                        {
                            continue;
                        }

                        if (!CanConsiderMerge(best!, rival.Identity, context)
                            || ScoreAnchors(bestVectors, vectors.Get(rival.Identity)) < ResolveMergeThreshold(best!, rival.Identity, settings, context))
                        {
                            mergeable = false;
                            break;
                        }
                    }
                }

                if (!mergeable)
                {
                    var contenders = new List<StoredFaceIdentity>();
                    foreach (var candidate in scored)
                    {
                        if ((bestScore - candidate.Score) <= margin)
                        {
                            contenders.Add(candidate.Identity);
                        }
                    }

                    settled[source] = new SettledSource(bestScore, contenders);
                    continue;
                }

                var target = ChooseMergeTarget(source, best!);
                var duplicate = ReferenceEquals(target, source) ? best! : source;
                MergeInto(target, duplicate);
                snapshot.Identities.Remove(duplicate);
                mergedFaceKeyMap[duplicate.FaceKey] = target.FaceKey;
                foreach (var key in mergedFaceKeyMap.Keys.ToArray())
                {
                    if (string.Equals(mergedFaceKeyMap[key], duplicate.FaceKey, StringComparison.OrdinalIgnoreCase))
                    {
                        mergedFaceKeyMap[key] = target.FaceKey;
                    }
                }

                // The merge changed the target and removed the duplicate; nothing else moved. Unsettle
                // exactly the sources whose verdict could read differently now: the pair itself, anyone
                // whose contenders included either of them, and anyone the changed target now scores
                // high enough against to become a contender.
                vectors.Forget(target);
                vectors.Forget(duplicate);
                settled.Remove(target);
                settled.Remove(duplicate);
                var targetVectors = vectors.Get(target);
                foreach (var (identity, verdict) in settled.ToArray())
                {
                    if (verdict.Contenders.Contains(target)
                        || verdict.Contenders.Contains(duplicate)
                        || (CanConsiderMerge(identity, target, context)
                            && ScoreAnchors(vectors.Get(identity), targetVectors) is > 0.0 and var score
                            && (verdict.BestScore - score) <= margin))
                    {
                        settled.Remove(identity);
                    }
                }

                mergeCount++;
                merged = true;
                break;
            }
        }

        return mergeCount;
    }

    private static int PromoteEvidenceBackedIdentities(FaceIdentitySnapshot snapshot)
    {
        var promoted = 0;
        foreach (var identity in snapshot.Identities)
        {
            if (IsPromoted(identity))
            {
                continue;
            }

            if (!string.IsNullOrWhiteSpace(identity.ReferenceExternalId))
            {
                identity.LifecycleStatus = StoredFaceIdentityLifecycle.Promoted;
                identity.PromotionReason = "reference";
                promoted++;
                continue;
            }

            if (identity.AssetIds.Distinct(StringComparer.OrdinalIgnoreCase).Skip(1).Any())
            {
                identity.LifecycleStatus = StoredFaceIdentityLifecycle.Promoted;
                identity.PromotionReason = "multi-asset";
                promoted++;
            }
        }

        return promoted;
    }

    // Two hard vetoes: conflicting reference identities, and identities proven to be different people
    // (detected in the same frame, or split apart by the user). Everything else is left to the
    // similarity floors — promoted+promoted pairs without a shared asset are considered too, since the
    // same performer split across disjoint videos is exactly that shape, but ResolveMergeThreshold holds
    // them to the stricter promoted floor.
    private static bool CanConsiderMerge(StoredFaceIdentity left, StoredFaceIdentity right, AiFaceReconciliationContext context)
        => !HasConflictingReference(left, right) && !context.Excludes(left, right);

    private static StoredFaceIdentity ChooseMergeTarget(StoredFaceIdentity left, StoredFaceIdentity right)
    {
        if (IsPromoted(left) != IsPromoted(right))
        {
            return IsPromoted(left) ? left : right;
        }

        if (!string.IsNullOrWhiteSpace(left.ReferenceExternalId) != !string.IsNullOrWhiteSpace(right.ReferenceExternalId))
        {
            return !string.IsNullOrWhiteSpace(left.ReferenceExternalId) ? left : right;
        }

        if (left.ObservationCount != right.ObservationCount)
        {
            return left.ObservationCount > right.ObservationCount ? left : right;
        }

        return left.QualityScore >= right.QualityScore ? left : right;
    }

    private static void MergeInto(StoredFaceIdentity target, StoredFaceIdentity duplicate)
    {
        if (string.IsNullOrWhiteSpace(target.Label))
        {
            target.Label = duplicate.Label;
        }

        if (string.IsNullOrWhiteSpace(target.ReferenceExternalId))
        {
            target.ReferenceExternalId = duplicate.ReferenceExternalId;
            target.ReferenceDisplayName = duplicate.ReferenceDisplayName;
            target.ReferencePackId = duplicate.ReferencePackId;
            target.ReferenceSuggestionId = duplicate.ReferenceSuggestionId;
        }

        if (IsPromoted(duplicate) && !IsPromoted(target))
        {
            target.LifecycleStatus = duplicate.LifecycleStatus;
            target.PromotionReason = duplicate.PromotionReason;
        }

        target.ObservationCount += duplicate.ObservationCount;
        foreach (var assetId in duplicate.AssetIds.AsEnumerable().Reverse())
        {
            RememberAsset(target, assetId);
        }

        if (ShouldUseDuplicateCover(target, duplicate))
        {
            target.CoverAssetId = duplicate.CoverAssetId;
            target.CoverBoundingBox = duplicate.CoverBoundingBox;
            target.CoverQualityScore = duplicate.CoverQualityScore;
        }

        target.QualityScore = Math.Max(target.QualityScore, duplicate.QualityScore);
        foreach (var anchor in duplicate.Anchors.OrderByDescending(static anchor => anchor.QualityScore))
        {
            if (target.Anchors.Any(existing => CosineSimilarity(existing.Vector, anchor.Vector) >= DuplicateAnchorSimilarity))
            {
                continue;
            }

            target.Anchors.Add(new StoredFaceAnchor
            {
                ModelKey = anchor.ModelKey,
                QualityScore = anchor.QualityScore,
                Vector = anchor.Vector.ToList(),
            });
        }

        if (target.Anchors.Count > MaxAnchorsPerIdentity)
        {
            target.Anchors = target.Anchors
                .OrderByDescending(static anchor => anchor.QualityScore)
                .Take(MaxAnchorsPerIdentity)
                .ToList();
        }
    }

    private static bool ShouldUseDuplicateCover(StoredFaceIdentity target, StoredFaceIdentity duplicate)
        => duplicate.CoverBoundingBox is not null
           && (target.CoverBoundingBox is null || duplicate.CoverQualityScore > target.CoverQualityScore);

    private static void RememberAsset(StoredFaceIdentity identity, string assetId)
    {
        if (string.IsNullOrWhiteSpace(assetId))
        {
            return;
        }

        identity.AssetIds.RemoveAll(value => string.Equals(value, assetId, StringComparison.OrdinalIgnoreCase));
        identity.AssetIds.Insert(0, assetId);
        if (identity.AssetIds.Count > MaxAssetIdsPerIdentity)
        {
            identity.AssetIds = identity.AssetIds.Take(MaxAssetIdsPerIdentity).ToList();
        }
    }

    private static bool HasSameReference(StoredFaceIdentity left, StoredFaceIdentity right)
        => !string.IsNullOrWhiteSpace(left.ReferenceExternalId)
           && string.Equals(left.ReferenceExternalId, right.ReferenceExternalId, StringComparison.OrdinalIgnoreCase);

    private static double ResolveMergeThreshold(
        StoredFaceIdentity left,
        StoredFaceIdentity right,
        AiFacesSettings settings,
        AiFaceReconciliationContext context)
    {
        var sharedAsset = HaveSharedAsset(left, right);
        var threshold = sharedAsset
            ? settings.ConsolidationSameAssetSimilarityThreshold
            : settings.ConsolidationSimilarityThreshold;

        if (!sharedAsset)
        {
            if (IsPromoted(left) && IsPromoted(right) && !HasSameReference(left, right))
            {
                return Math.Max(threshold, settings.ConsolidationPromotedSimilarityThreshold);
            }

            return threshold;
        }

        // The relaxed same-asset floors below exist to reunite one performer who was fragmented into
        // several clusters of the video being analysed. Two co-performers in that same video match the
        // same description, so the relaxation only applies while that video is the one being processed —
        // where asset-local co-occurrence is available to veto the co-performer case outright. On every
        // later run the pair falls back to the strict same-asset floor.
        if (!context.AllowsSameAssetRelaxation(left, right))
        {
            return threshold;
        }

        if (!string.IsNullOrWhiteSpace(left.ReferenceExternalId) || !string.IsNullOrWhiteSpace(right.ReferenceExternalId))
        {
            return Math.Min(threshold, settings.IdentityMatchThreshold);
        }

        if (ShouldRelaxSameAssetAnonymousThreshold(left, right))
        {
            return Math.Min(threshold, settings.IdentityMatchThreshold);
        }

        return threshold;
    }

    private static bool ShouldRelaxSameAssetAnonymousThreshold(StoredFaceIdentity left, StoredFaceIdentity right)
        => IsAnonymousVideoEvidenceIdentity(left) && IsAnonymousVideoEvidenceIdentity(right);

    private static bool IsAnonymousVideoEvidenceIdentity(StoredFaceIdentity identity)
        => IsPromoted(identity)
           && string.IsNullOrWhiteSpace(identity.ReferenceExternalId)
           && string.Equals(identity.PromotionReason, "video-evidence", StringComparison.OrdinalIgnoreCase);

    private static bool HasConflictingReference(StoredFaceIdentity left, StoredFaceIdentity right)
        => !string.IsNullOrWhiteSpace(left.ReferenceExternalId)
           && !string.IsNullOrWhiteSpace(right.ReferenceExternalId)
           && !string.Equals(left.ReferenceExternalId, right.ReferenceExternalId, StringComparison.OrdinalIgnoreCase);

    private static bool HaveSharedAsset(StoredFaceIdentity left, StoredFaceIdentity right)
        => left.AssetIds.Any(leftAsset => right.AssetIds.Any(rightAsset => string.Equals(leftAsset, rightAsset, StringComparison.OrdinalIgnoreCase)));

    private static bool IsPromoted(StoredFaceIdentity identity)
        => string.Equals(identity.LifecycleStatus, StoredFaceIdentityLifecycle.Promoted, StringComparison.OrdinalIgnoreCase);

    // Identity-vs-identity similarity, shared with the preparation service's duplicate-aware
    // ambiguity check so "are these two stored identities the same person" means one thing.
    internal static double ScoreIdentityPair(StoredFaceIdentity left, StoredFaceIdentity right)
        => ScoreAnchors(FaceAnchorVectorCache.Normalize(left), FaceAnchorVectorCache.Normalize(right));

    // For each left anchor, its best cosine against the right anchors; the score is the mean of the two
    // strongest of those (or the single one). Inputs are unit vectors, so cosine is a plain dot product.
    private static double ScoreAnchors(float[]?[] left, float[]?[] right)
    {
        if (left.Length == 0 || right.Length == 0)
        {
            return 0.0;
        }

        var best = 0.0;
        var second = 0.0;
        foreach (var leftVector in left)
        {
            var bestSimilarity = 0.0;
            if (leftVector is not null)
            {
                foreach (var rightVector in right)
                {
                    if (rightVector is null || rightVector.Length != leftVector.Length)
                    {
                        continue;
                    }

                    var similarity = Math.Clamp((double)SaieReferencePack.Dot(leftVector, rightVector), 0.0, 1.0);
                    if (similarity > bestSimilarity)
                    {
                        bestSimilarity = similarity;
                    }
                }
            }

            if (bestSimilarity > best)
            {
                second = best;
                best = bestSimilarity;
            }
            else if (bestSimilarity > second)
            {
                second = bestSimilarity;
            }
        }

        return left.Length >= 2 ? (best + second) / 2.0 : best;
    }

    private static FaceReferenceMatch? TryMatchReference(StoredFaceIdentity identity, SaieReferencePack referencePack, AiFacesSettings settings)
    {
        if (identity.Anchors.Count == 0 || referencePack.Identities.Count == 0)
        {
            return null;
        }

        var match = SaieReferenceMatcher.FindBest(referencePack, identity.Anchors.Select(static anchor => (IReadOnlyList<float>)anchor.Vector).ToArray());
        if (match is not { } best
            || best.Score < settings.ReferenceMatchThreshold
            || (best.Score - best.SecondScore) < settings.ReferenceAmbiguityMargin)
        {
            return null;
        }

        var referenceIdentity = referencePack.Identities[best.Ordinal];
        return new FaceReferenceMatch(referenceIdentity, referencePack.Manifest.PackId, AiFaceReferenceSuggestionIds.FromOrdinal(referenceIdentity.Ordinal), best.Score);
    }

    private static double CosineSimilarity(IReadOnlyList<float> left, IReadOnlyList<float> right)
    {
        if (left.Count == 0 || right.Count == 0 || left.Count != right.Count)
        {
            return 0.0;
        }

        double dot = 0.0;
        double leftNorm = 0.0;
        double rightNorm = 0.0;
        for (var index = 0; index < left.Count; index++)
        {
            dot += left[index] * right[index];
            leftNorm += left[index] * left[index];
            rightNorm += right[index] * right[index];
        }

        if (leftNorm <= 0.0 || rightNorm <= 0.0)
        {
            return 0.0;
        }

        return Math.Clamp(dot / (Math.Sqrt(leftNorm) * Math.Sqrt(rightNorm)), 0.0, 1.0);
    }

    private sealed record FaceReferenceMatch(
        SaieReferenceIdentity Identity,
        string PackId,
        int SuggestionId,
        double Score
    );

    // A source that was scanned and found nothing to merge. The verdict read only the source, its best
    // score, and the contenders (candidates within the ambiguity margin of the best, the best included),
    // so it stands until one of those changes or a changed identity scores its way into that band.
    private sealed record SettledSource(double BestScore, List<StoredFaceIdentity> Contenders);

    // Unit-length copies of each identity's anchor vectors, built once per identity instead of
    // re-deriving both norms on every pair. A null entry is an anchor with no usable vector (empty or
    // zero-length), which scores 0 against everything. Callers Forget an identity whose anchors changed.
    private sealed class FaceAnchorVectorCache
    {
        private readonly Dictionary<StoredFaceIdentity, float[]?[]> _byIdentity = new(ReferenceEqualityComparer.Instance);

        public float[]?[] Get(StoredFaceIdentity identity)
        {
            if (!_byIdentity.TryGetValue(identity, out var vectors))
            {
                vectors = Normalize(identity);
                _byIdentity[identity] = vectors;
            }

            return vectors;
        }

        public void Forget(StoredFaceIdentity identity) => _byIdentity.Remove(identity);

        public static float[]?[] Normalize(StoredFaceIdentity identity)
        {
            var vectors = new float[]?[identity.Anchors.Count];
            for (var index = 0; index < vectors.Length; index++)
            {
                vectors[index] = Normalize(identity.Anchors[index].Vector);
            }

            return vectors;
        }

        private static float[]? Normalize(List<float> vector)
        {
            if (vector.Count == 0)
            {
                return null;
            }

            var normSquared = 0.0;
            foreach (var value in vector)
            {
                normSquared += (double)value * value;
            }

            if (normSquared <= 0.0)
            {
                return null;
            }

            var norm = Math.Sqrt(normSquared);
            var unit = new float[vector.Count];
            for (var index = 0; index < unit.Length; index++)
            {
                unit[index] = (float)(vector[index] / norm);
            }

            return unit;
        }
    }
}

internal sealed record AiFaceIdentityReconciliationReport(
    int MergedIdentityCount,
    int ReferencePromotedIdentityCount,
    int EvidencePromotedIdentityCount,
    IReadOnlyDictionary<string, string> MergedFaceKeyMap
);