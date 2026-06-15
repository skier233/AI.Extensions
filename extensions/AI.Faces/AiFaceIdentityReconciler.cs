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
    public AiFaceIdentityReconciliationReport Reconcile(FaceIdentitySnapshot snapshot, SaieReferencePack? referencePack, AiFacesSettings settings, bool applyReferenceMatches = true)
    {
        var referencePromotions = applyReferenceMatches ? ApplyReferenceMatches(snapshot, referencePack, settings) : 0;
        var mergedFaceKeyMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        var merges = MergeSimilarIdentities(snapshot, settings, mergedFaceKeyMap);
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

    private static int MergeSimilarIdentities(FaceIdentitySnapshot snapshot, AiFacesSettings settings, Dictionary<string, string> mergedFaceKeyMap)
    {
        var mergeCount = 0;
        var merged = true;
        while (merged)
        {
            merged = false;
            foreach (var source in snapshot.Identities.ToArray())
            {
                if (!snapshot.Identities.Contains(source) || source.Anchors.Count == 0)
                {
                    continue;
                }

                var ranked = snapshot.Identities
                    .Where(candidate => !ReferenceEquals(candidate, source) && CanConsiderMerge(source, candidate))
                    .Select(candidate => new
                    {
                        Identity = candidate,
                        Score = ScoreIdentityPair(source, candidate),
                        Threshold = ResolveMergeThreshold(source, candidate, settings),
                    })
                    .Where(static candidate => candidate.Score > 0.0)
                    .OrderByDescending(static candidate => candidate.Score)
                    .ToArray();
                if (ranked.Length == 0)
                {
                    continue;
                }

                var best = ranked[0];
                var secondBestScore = ranked.Length > 1 ? ranked[1].Score : 0.0;
                if (best.Score < best.Threshold)
                {
                    continue;
                }

                if ((best.Score - secondBestScore) < settings.ConsolidationAmbiguityMargin)
                {
                    // With three or more identities of the same person, every candidate scores within
                    // the margin of every other — a plain margin check deadlocks all merges exactly
                    // when duplicates are most numerous. Ambiguity only blocks when an in-margin rival
                    // is plausibly a *different* person from the best candidate, i.e. not itself
                    // mergeable with it.
                    var rivalsAreDuplicatesOfBest = ranked
                        .Skip(1)
                        .TakeWhile(candidate => (best.Score - candidate.Score) < settings.ConsolidationAmbiguityMargin)
                        .All(candidate => !HasConflictingReference(best.Identity, candidate.Identity)
                            && ScoreIdentityPair(best.Identity, candidate.Identity) >= ResolveMergeThreshold(best.Identity, candidate.Identity, settings));
                    if (!rivalsAreDuplicatesOfBest)
                    {
                        continue;
                    }
                }

                var target = ChooseMergeTarget(source, best.Identity);
                var duplicate = ReferenceEquals(target, source) ? best.Identity : source;
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

    // Conflicting references are the only hard veto. Promoted+promoted pairs without a shared asset
    // are considered too — the same performer split across disjoint videos is exactly that shape —
    // but ResolveMergeThreshold holds them to the stricter promoted floor.
    private static bool CanConsiderMerge(StoredFaceIdentity left, StoredFaceIdentity right)
        => !HasConflictingReference(left, right);

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

    private static double ResolveMergeThreshold(StoredFaceIdentity left, StoredFaceIdentity right, AiFacesSettings settings)
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
    {
        if (left.Anchors.Count == 0 || right.Anchors.Count == 0)
        {
            return 0.0;
        }

        var scores = new List<double>();
        foreach (var leftAnchor in left.Anchors)
        {
            var bestSimilarity = right.Anchors
                .Select(rightAnchor => CosineSimilarity(leftAnchor.Vector, rightAnchor.Vector))
                .DefaultIfEmpty(0.0)
                .Max();
            scores.Add(bestSimilarity);
        }

        return scores.OrderByDescending(static score => score).Take(Math.Min(2, scores.Count)).Average();
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
}

internal sealed record AiFaceIdentityReconciliationReport(
    int MergedIdentityCount,
    int ReferencePromotedIdentityCount,
    int EvidencePromotedIdentityCount,
    IReadOnlyDictionary<string, string> MergedFaceKeyMap
);