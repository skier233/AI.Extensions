namespace AI.Faces;

internal sealed class AiFaceIdentityReconciler
{
    private const double DuplicateAnchorSimilarity = 0.975;
    private const int MaxAnchorsPerIdentity = 12;
    private const int MaxAssetIdsPerIdentity = 32;

    public AiFaceIdentityReconciliationReport Reconcile(FaceIdentitySnapshot snapshot, SaieReferencePack? referencePack, AiFacesSettings settings)
    {
        var referencePromotions = ApplyReferenceMatches(snapshot, referencePack, settings);
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
                        Score = ScoreIdentity(source, candidate),
                        SharedAsset = HaveSharedAsset(source, candidate),
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
                if (best.Score < best.Threshold || (best.Score - secondBestScore) < settings.ConsolidationAmbiguityMargin)
                {
                    continue;
                }

                if (!CanAutoMerge(source, best.Identity, best.SharedAsset))
                {
                    continue;
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

    private static bool CanConsiderMerge(StoredFaceIdentity left, StoredFaceIdentity right)
    {
        if (HasConflictingReference(left, right))
        {
            return false;
        }

        if (HasSameReference(left, right))
        {
            return true;
        }

        return !IsPromoted(left) || !IsPromoted(right) || HaveSharedAsset(left, right);
    }

    private static bool CanAutoMerge(StoredFaceIdentity left, StoredFaceIdentity right, bool sharedAsset)
    {
        if (HasConflictingReference(left, right))
        {
            return false;
        }

        if (HasSameReference(left, right))
        {
            return true;
        }

        return !IsPromoted(left) || !IsPromoted(right) || sharedAsset;
    }

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

    private static double ScoreIdentity(StoredFaceIdentity left, StoredFaceIdentity right)
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
        var ranked = referencePack.Identities
            .Select(referenceIdentity => new FaceReferenceMatch(
                referenceIdentity,
                referencePack.Manifest.PackId,
                AiFaceReferenceSuggestionIds.FromOrdinal(referenceIdentity.Ordinal),
                ScoreReference(identity, referencePack, referenceIdentity.Ordinal)))
            .Where(static match => match.Score > 0.0)
            .OrderByDescending(static match => match.Score)
            .ToArray();
        if (ranked.Length == 0)
        {
            return null;
        }

        var best = ranked[0];
        var secondBestScore = ranked.Length > 1 ? ranked[1].Score : 0.0;
        if (best.Score < settings.ReferenceMatchThreshold || (best.Score - secondBestScore) < settings.ReferenceAmbiguityMargin)
        {
            return null;
        }

        return best;
    }

    private static double ScoreReference(StoredFaceIdentity identity, SaieReferencePack referencePack, int ordinal)
    {
        if (ordinal < 0 || ordinal >= referencePack.Identities.Count)
        {
            return 0.0;
        }

        var centroid = referencePack.GetCentroid(ordinal);
        var centroidNorm = referencePack.GetCentroidNorm(ordinal);
        var scores = new List<double>();
        foreach (var anchor in identity.Anchors)
        {
            if (anchor.Vector.Count != referencePack.Manifest.EmbeddingDim)
            {
                continue;
            }

            var score = CosineSimilarity(anchor.Vector, centroid, centroidNorm);
            if (score > 0.0)
            {
                scores.Add(score);
            }
        }

        return scores.Count == 0
            ? 0.0
            : scores.OrderByDescending(static score => score).Take(Math.Min(2, scores.Count)).Average();
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

    private static double CosineSimilarity(IReadOnlyList<float> left, ReadOnlySpan<float> right, float rightNorm)
    {
        if (left.Count == 0 || left.Count != right.Length || rightNorm <= 0f)
        {
            return 0.0;
        }

        double dot = 0.0;
        double leftNorm = 0.0;
        for (var index = 0; index < left.Count; index++)
        {
            dot += left[index] * right[index];
            leftNorm += left[index] * left[index];
        }

        if (leftNorm <= 0.0)
        {
            return 0.0;
        }

        return Math.Clamp(dot / (Math.Sqrt(leftNorm) * rightNorm), 0.0, 1.0);
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