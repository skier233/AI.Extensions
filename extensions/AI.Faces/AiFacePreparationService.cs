using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

namespace AI.Faces;

internal sealed class AiFacePreparationService
{
    private const string SourceKey = "ext:ai.faces";
    private const double DefaultFrameSpanSeconds = 0.5;
    private const int MaxGapFrames = 3;
    private const int MaxEmbeddingGapFrames = 12;
    private const double IoUMatchThreshold = 0.5;
    private const double EmbeddingMatchThreshold = 0.68;
    private const double LongGapEmbeddingMatchThreshold = 0.82;
    private const double TrackerIoUWeight = 0.35;
    private const double TrackerEmbeddingWeight = 0.65;
    private const double HardMinimumDetectionScore = 0.3;
    private const double HardMinimumEmbeddingNorm = 10.0;
    private const double AnchorDetectionScore = 0.65;
    private const double AnchorEmbeddingNorm = 18.0;
    private const double ProvisionalMinimumPoseQuality = 0.6;
    private const double MinimumNormalizedAnchorArea = 0.01;
    private const double MinimumPixelAnchorArea = 4096.0;
    private const double RepresentativeDedupSimilarity = 0.96;
    private const double IdentityAnchorDuplicateSimilarity = 0.975;
    private const double SameAssetIdentityMatchRelaxation = 0.07;
    private const int MaxRepresentativeEmbeddings = 6;
    private const int MaxAnchorsPerIdentity = 12;
    private const int MaxAssetIdsPerIdentity = 32;

    private readonly IFaceIdentityStateStore _stateStore;
    private readonly AiAssetFaceClusterer _assetClusterer;
    private readonly AiFaceIdentityReconciler _identityReconciler;
    private readonly AiFaceReferencePackStore? _referencePackStore;

    public AiFacePreparationService(IFaceIdentityStateStore stateStore)
        : this(stateStore, new AiAssetFaceClusterer(), new AiFaceIdentityReconciler(), null)
    {
    }

    public AiFacePreparationService(
        IFaceIdentityStateStore stateStore,
        AiAssetFaceClusterer assetClusterer,
        AiFaceIdentityReconciler? identityReconciler = null,
        AiFaceReferencePackStore? referencePackStore = null)
    {
        _stateStore = stateStore;
        _assetClusterer = assetClusterer;
        _identityReconciler = identityReconciler ?? new AiFaceIdentityReconciler();
        _referencePackStore = referencePackStore;
    }

    public async Task<AiPreparedArtifactBatch> PrepareAsync(AiDispatchRequest request, CancellationToken ct = default)
    {
        var batch = new AiPreparedArtifactBatch();
        if (request.Result.MediaKind is not (AiMediaKinds.Image or AiMediaKinds.Video))
        {
            batch.Notes.Add($"AI.Faces does not consume media kind '{request.Result.MediaKind}'.");
            return batch;
        }

        var snapshot = await _stateStore.LoadAsync(ct);
        var settings = await AiFacesSettingsRuntime.LoadAsync(ct);
        var rawTracks = request.Result.MediaKind == AiMediaKinds.Image
            ? BuildImageTracks(request)
            : BuildVideoTracks(request);
        var clusterResult = _assetClusterer.ClusterWithDiagnostics(rawTracks, settings);
        var tracks = clusterResult.Tracks;

        if (tracks.Count == 0)
        {
            batch.Notes.Add("No face detections were available to prepare.");
            return batch;
        }

        var referencePack = _referencePackStore is null ? null : await _referencePackStore.GetActivePackAsync(ct);
        var initialReconciliation = _identityReconciler.Reconcile(snapshot, referencePack, settings);

        var emittedFaces = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var unresolvedTracks = 0;
        var provisionalClusters = 0;
        var newIdentityCount = 0;
        var promotedIdentityCount = 0;
        var seededMatchCount = 0;
        var conflictingReferenceCount = 0;
        var preparedTracks = new List<PreparedFaceAssignment>();

        foreach (var track in tracks.OrderByDescending(static track => track.TrackQuality))
        {
            var representativeEmbeddings = SelectRepresentativeEmbeddings(track, settings);
            var hardPassEmbeddings = representativeEmbeddings.Where(static embedding => embedding.PassesHardFloor).ToArray();
            var identityPassEmbeddings = representativeEmbeddings.Where(static embedding => embedding.PassesIdentityFloor).ToArray();

            if (identityPassEmbeddings.Length == 0)
            {
                preparedTracks.Add(new PreparedFaceAssignment(track, null, [], false));
                unresolvedTracks++;
                continue;
            }

            var anchorEmbeddings = hardPassEmbeddings.Where(static embedding => embedding.IsAnchor).ToArray();
            var identityEmbeddings = anchorEmbeddings.Length > 0 ? anchorEmbeddings : identityPassEmbeddings;
            var referenceMatch = TryMatchReference(identityEmbeddings, referencePack, settings);
            if (referenceMatch is not null)
            {
                seededMatchCount++;
            }

            var referenceApplied = false;
            var wasCreated = false;
            var identity = TryFindReferenceIdentity(snapshot, referenceMatch)
                ?? TryMatchIdentity(identityEmbeddings, snapshot.Identities, settings, request.Context.AssetId);

            if (identity is not null && HasConflictingReference(identity, referenceMatch))
            {
                identity = null;
                conflictingReferenceCount++;
            }

            if (identity is null)
            {
                identity = CreateIdentity(snapshot, request.Context.AssetId, track, referenceMatch);
                wasCreated = true;
                newIdentityCount++;
            }

            referenceApplied = TryApplyReferenceMatch(identity, referenceMatch);
            UpdateIdentity(identity, request.Context.AssetId, track, identityEmbeddings);
            if (TryPromoteIdentity(request, identity, track, settings, referenceApplied))
            {
                promotedIdentityCount++;
            }

            preparedTracks.Add(new PreparedFaceAssignment(track, identity, identityEmbeddings, wasCreated));
        }

        var finalReconciliation = _identityReconciler.Reconcile(snapshot, referencePack, settings);

        foreach (var preparedTrack in preparedTracks)
        {
            var identity = ResolveFinalIdentity(preparedTrack.Identity, snapshot, finalReconciliation);
            if (identity is null || !ShouldEmitPromotedIdentity(request, identity, preparedTrack.Track, settings))
            {
                EmitDetections(batch, request, preparedTrack.Track, faceKey: null, settings);
                if (identity is not null)
                {
                    provisionalClusters++;
                }

                continue;
            }

            var wasCreated = preparedTrack.WasCreated
                && preparedTrack.Identity is not null
                && string.Equals(preparedTrack.Identity.FaceKey, identity.FaceKey, StringComparison.OrdinalIgnoreCase);
            EmitFace(batch, request, identity, preparedTrack.Track, emittedFaces, wasCreated);
            var trackWindows = BuildFaceTrackWindows(preparedTrack.Track);
            var retainedSamples = EmitDetections(batch, request, preparedTrack.Track, identity.FaceKey, trackWindows, settings);
            var segmentCount = EmitSegments(batch, request, preparedTrack.Track, identity.FaceKey, trackWindows, retainedSamples, settings);
            EmitFaceAppearance(batch, request, preparedTrack.Track, identity.FaceKey, retainedSamples, segmentCount);
            EmitTrackEmbedding(batch, request, preparedTrack.Track, identity.FaceKey, preparedTrack.IdentityEmbeddings);
        }

        await _stateStore.SaveAsync(snapshot, ct);

        if (newIdentityCount > 0)
        {
            batch.Notes.Add($"Prepared {newIdentityCount} new face identity state(s) after asset-local clustering.");
        }

        if (promotedIdentityCount > 0)
        {
            batch.Notes.Add($"Promoted {promotedIdentityCount} face identity candidate(s) for Cove face persistence.");
        }

        if (unresolvedTracks > 0)
        {
            batch.Notes.Add($"Left {unresolvedTracks} track(s) unresolved rather than forcing them into muddy clusters.");
        }

        if (provisionalClusters > 0)
        {
            batch.Notes.Add($"Kept {provisionalClusters} unknown face cluster(s) provisional instead of creating visible Cove face rows.");
        }

        AddTelemetryNotes(
            batch,
            settings,
            clusterResult.Diagnostics,
            initialReconciliation,
            finalReconciliation,
            newIdentityCount,
            promotedIdentityCount,
            seededMatchCount,
            conflictingReferenceCount,
            unresolvedTracks,
            provisionalClusters);

        return batch;
    }

    private static IReadOnlyList<PreparedFaceTrack> BuildImageTracks(AiDispatchRequest request)
    {
        if (request.Result.AssetAnalysis is null)
        {
            return [];
        }

        var detections = request.Result.AssetAnalysis.FindDetections("face").ToArray();
        if (detections.Length == 0)
        {
            return [];
        }

        var embeddingsByDetection = request.Result.AssetAnalysis
            .EnumerateAllEmbeddings()
            .Where(static embedding => embedding.DetectionIndex.HasValue)
            .GroupBy(static embedding => embedding.DetectionIndex!.Value)
            .ToDictionary(static group => group.Key, static group => (IReadOnlyList<AiEmbeddingObservation>)group.ToArray());

        var tracks = new List<PreparedFaceTrack>();
        var counter = 1;
        foreach (var detection in detections)
        {
            embeddingsByDetection.TryGetValue(detection.DetectionIndex, out var embeddings);
            var sample = new FaceFrameSample(
                detection,
                embeddings ?? [],
                null,
                0);
            tracks.Add(CreatePreparedTrack($"image-face-{counter}", [sample], null, null, null));
            counter++;
        }

        return tracks;
    }

    private static PreparedFaceTrack CreatePreparedTrack(
        string trackKey,
        IReadOnlyList<FaceFrameSample> samples,
        double? startSeconds,
        double? endSeconds,
        double? frameIntervalSeconds)
        => new(
            trackKey,
            samples,
            startSeconds,
            endSeconds,
            frameIntervalSeconds,
            ScoreTrack(samples),
            AiFaceQualityScorer.SelectBestCoverSample(samples));

    private static IReadOnlyList<PreparedFaceTrack> BuildVideoTracks(AiDispatchRequest request)
    {
        var sliceSpan = request.Result.FrameIntervalSeconds ?? request.Context.FrameIntervalSeconds ?? DefaultFrameSpanSeconds;
        var orderedFrames = request.Result.Frames
            .OrderBy(static frame => frame.TimeSeconds ?? double.MinValue)
            .ThenBy(static frame => frame.Index ?? int.MinValue)
            .ToArray();
        var openTracks = new List<OpenTrack>();
        var completed = new List<PreparedFaceTrack>();
        var nextTrackId = 1;

        foreach (var frame in orderedFrames)
        {
            var frameOrder = frame.Index ?? nextTrackId;
            var timeSeconds = frame.TimeSeconds ?? (frameOrder * sliceSpan);
            var detections = frame.Analysis.FindDetections("face").ToArray();
            var embeddingsByDetection = frame.Analysis
                .EnumerateAllEmbeddings()
                .Where(static embedding => embedding.DetectionIndex.HasValue)
                .GroupBy(static embedding => embedding.DetectionIndex!.Value)
                .ToDictionary(static group => group.Key, static group => (IReadOnlyList<AiEmbeddingObservation>)group.ToArray());
            var frameSamples = detections
                .Select(detection => new FaceFrameSample(
                    detection,
                    embeddingsByDetection.TryGetValue(detection.DetectionIndex, out var embeddings) ? embeddings : [],
                    timeSeconds,
                    frameOrder))
                .ToArray();

            var matchedTracks = new HashSet<int>();
            var matchedSamples = new HashSet<int>();
            var candidatePairs = new List<(double Score, int TrackIndex, int SampleIndex)>();
            for (var trackIndex = 0; trackIndex < openTracks.Count; trackIndex++)
            {
                var track = openTracks[trackIndex];
                var frameGap = frameOrder - track.LastFrameOrder;
                if (frameGap <= 0 || frameGap > MaxEmbeddingGapFrames)
                {
                    continue;
                }

                for (var sampleIndex = 0; sampleIndex < frameSamples.Length; sampleIndex++)
                {
                    if (TryScoreTrackContinuation(track, frameSamples[sampleIndex], frameGap, out var score))
                    {
                        candidatePairs.Add((score, trackIndex, sampleIndex));
                    }
                }
            }

            foreach (var (_, trackIndex, sampleIndex) in candidatePairs.OrderByDescending(static pair => pair.Score))
            {
                if (!matchedTracks.Add(trackIndex) || !matchedSamples.Add(sampleIndex))
                {
                    continue;
                }

                openTracks[trackIndex].Append(frameSamples[sampleIndex]);
            }

            for (var trackIndex = openTracks.Count - 1; trackIndex >= 0; trackIndex--)
            {
                if (matchedTracks.Contains(trackIndex))
                {
                    continue;
                }

                if ((frameOrder - openTracks[trackIndex].LastFrameOrder) <= MaxGapFrames)
                {
                    continue;
                }

                completed.Add(openTracks[trackIndex].Close(sliceSpan));
                openTracks.RemoveAt(trackIndex);
            }

            for (var sampleIndex = 0; sampleIndex < frameSamples.Length; sampleIndex++)
            {
                if (matchedSamples.Contains(sampleIndex))
                {
                    continue;
                }

                openTracks.Add(new OpenTrack($"video-face-{nextTrackId}", frameSamples[sampleIndex]));
                nextTrackId++;
            }
        }

        completed.AddRange(openTracks.Select(track => track.Close(sliceSpan)));
        return completed;
    }

    private static bool TryScoreTrackContinuation(OpenTrack track, FaceFrameSample sample, int frameGap, out double score)
    {
        score = 0.0;
        var iou = ComputeIoU(track.LastBoundingBox, sample.Detection.BoundingBox);
        var embeddingSimilarity = ComputeBestEmbeddingSimilarity(track.LastSample, sample);
        var requiredEmbeddingSimilarity = frameGap <= MaxGapFrames
            ? EmbeddingMatchThreshold
            : LongGapEmbeddingMatchThreshold;
        var hasIoUMatch = frameGap <= MaxGapFrames && iou >= IoUMatchThreshold;
        var hasEmbeddingMatch = embeddingSimilarity >= requiredEmbeddingSimilarity;

        if (!hasIoUMatch && !hasEmbeddingMatch)
        {
            return false;
        }

        var gapPenalty = Math.Min(0.25, Math.Max(0, frameGap - 1) * 0.025);
        score = (iou * TrackerIoUWeight) + (embeddingSimilarity * TrackerEmbeddingWeight) - gapPenalty;
        return score > 0.0;
    }

    private static double ComputeBestEmbeddingSimilarity(FaceFrameSample left, FaceFrameSample right)
    {
        var best = 0.0;
        foreach (var leftEmbedding in left.Embeddings)
        {
            foreach (var rightEmbedding in right.Embeddings)
            {
                best = Math.Max(best, CosineSimilarity(leftEmbedding.Vector, rightEmbedding.Vector));
            }
        }

        return best;
    }

    private static IReadOnlyList<RepresentativeFaceEmbedding> SelectRepresentativeEmbeddings(PreparedFaceTrack track, AiFacesSettings settings)
    {
        var candidates = new List<RepresentativeFaceEmbedding>();
        foreach (var sample in track.Samples)
        {
            foreach (var embedding in sample.Embeddings)
            {
                var poseQuality = GetMetadataQuality(embedding, AiFaceQualityScorer.PoseQualityMetadataKey);
                var imageQuality = GetMetadataQuality(embedding, AiFaceQualityScorer.ImageQualityMetadataKey);
                var qualityScore = (embedding.Norm ?? 0.0) * sample.Detection.Score * poseQuality * imageQuality;
                var hardPass = sample.Detection.Score >= HardMinimumDetectionScore
                    && (embedding.Norm ?? 0.0) >= HardMinimumEmbeddingNorm
                    && poseQuality >= settings.MinimumPoseQuality
                    && imageQuality >= settings.MinimumImageQuality;
                var identityPass = sample.Detection.Score >= AnchorDetectionScore
                    && (embedding.Norm ?? 0.0) >= AnchorEmbeddingNorm
                    && poseQuality >= ProvisionalMinimumPoseQuality
                    && imageQuality >= settings.MinimumImageQuality;
                var area = sample.Detection.BoundingBox.Area;
                var minimumArea = area <= 1.0 ? MinimumNormalizedAnchorArea : MinimumPixelAnchorArea;
                var isAnchor = hardPass
                    && sample.Detection.Score >= AnchorDetectionScore
                    && (embedding.Norm ?? 0.0) >= AnchorEmbeddingNorm
                    && area >= minimumArea;

                candidates.Add(new RepresentativeFaceEmbedding(
                    embedding.ModelKey,
                    embedding.Vector,
                    embedding.Norm ?? 0.0,
                    sample.Detection.Score,
                    sample.Detection.BoundingBox,
                    sample.TimeSeconds,
                    poseQuality,
                    imageQuality,
                    qualityScore,
                    hardPass,
                    identityPass,
                    isAnchor));
            }
        }

        var selected = new List<RepresentativeFaceEmbedding>();
        foreach (var candidate in candidates.OrderByDescending(static candidate => candidate.QualityScore))
        {
            if (!candidate.PassesHardFloor && !candidate.PassesIdentityFloor)
            {
                continue;
            }

            if (selected.Any(existing => CosineSimilarity(existing.Vector, candidate.Vector) >= RepresentativeDedupSimilarity))
            {
                continue;
            }

            selected.Add(candidate);
            if (selected.Count >= MaxRepresentativeEmbeddings)
            {
                break;
            }
        }

        return selected;
    }

    private static StoredFaceIdentity? TryMatchIdentity(IReadOnlyList<RepresentativeFaceEmbedding> anchors, IReadOnlyList<StoredFaceIdentity> identities, AiFacesSettings settings, string? assetId)
    {
        var ranked = identities
            .Select(identity => new
            {
                Identity = identity,
                Score = ScoreIdentity(anchors, identity),
            })
            .Where(static candidate => candidate.Score > 0)
            .OrderByDescending(static candidate => candidate.Score)
            .ToArray();

        if (ranked.Length == 0)
        {
            return null;
        }

        var best = ranked[0];
        var secondBest = ranked.Length > 1 ? ranked[1].Score : 0.0;
        var matchThreshold = IsObservedInAsset(best.Identity, assetId)
            ? Math.Max(0.0, settings.IdentityMatchThreshold - SameAssetIdentityMatchRelaxation)
            : settings.IdentityMatchThreshold;
        if (best.Score < matchThreshold || (best.Score - secondBest) < settings.IdentityAmbiguityMargin)
        {
            return null;
        }

        return best.Identity;
    }

    private static bool IsObservedInAsset(StoredFaceIdentity identity, string? assetId)
        => !string.IsNullOrWhiteSpace(assetId)
           && identity.AssetIds.Any(value => string.Equals(value, assetId, StringComparison.OrdinalIgnoreCase));

    private static double ScoreIdentity(IReadOnlyList<RepresentativeFaceEmbedding> anchors, StoredFaceIdentity identity)
    {
        if (anchors.Count == 0 || identity.Anchors.Count == 0)
        {
            return 0.0;
        }

        var scores = new List<double>();
        foreach (var anchor in anchors)
        {
            var bestSimilarity = identity.Anchors
                .Select(storedAnchor => CosineSimilarity(anchor.Vector, storedAnchor.Vector))
                .DefaultIfEmpty(0.0)
                .Max();
            scores.Add(bestSimilarity);
        }

        return scores.OrderByDescending(static score => score).Take(Math.Min(2, scores.Count)).Average();
    }

    private static FaceReferenceMatch? TryMatchReference(IReadOnlyList<RepresentativeFaceEmbedding> anchors, SaieReferencePack? referencePack, AiFacesSettings settings)
    {
        if (referencePack is null || anchors.Count == 0 || referencePack.Identities.Count == 0)
        {
            return null;
        }

        var ranked = referencePack.Identities
            .Select(identity => new FaceReferenceMatch(
                identity,
                referencePack.Manifest.PackId,
                AiFaceReferenceSuggestionIds.FromOrdinal(identity.Ordinal),
                ScoreReferenceIdentity(anchors, referencePack, identity.Ordinal)))
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

    private static double ScoreReferenceIdentity(IReadOnlyList<RepresentativeFaceEmbedding> anchors, SaieReferencePack referencePack, int ordinal)
    {
        if (ordinal < 0 || ordinal >= referencePack.Identities.Count)
        {
            return 0.0;
        }

        var centroid = referencePack.GetCentroid(ordinal);
        var centroidNorm = referencePack.GetCentroidNorm(ordinal);
        var scores = new List<double>();
        foreach (var anchor in anchors)
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

    private static StoredFaceIdentity? TryFindReferenceIdentity(FaceIdentitySnapshot snapshot, FaceReferenceMatch? referenceMatch)
    {
        if (referenceMatch is null)
        {
            return null;
        }

        return snapshot.Identities.FirstOrDefault(identity => string.Equals(
            identity.ReferenceExternalId,
            referenceMatch.Identity.ExternalId,
            StringComparison.OrdinalIgnoreCase));
    }

    private static bool HasConflictingReference(StoredFaceIdentity identity, FaceReferenceMatch? referenceMatch)
        => referenceMatch is not null
           && !string.IsNullOrWhiteSpace(identity.ReferenceExternalId)
           && !string.Equals(identity.ReferenceExternalId, referenceMatch.Identity.ExternalId, StringComparison.OrdinalIgnoreCase);

    private static bool TryApplyReferenceMatch(StoredFaceIdentity identity, FaceReferenceMatch? referenceMatch)
    {
        if (referenceMatch is null || HasConflictingReference(identity, referenceMatch))
        {
            return false;
        }

        identity.ReferenceExternalId = referenceMatch.Identity.ExternalId;
        identity.ReferenceDisplayName = referenceMatch.Identity.DisplayName;
        identity.ReferencePackId = referenceMatch.PackId;
        identity.ReferenceSuggestionId = referenceMatch.SuggestionId;
        if (string.IsNullOrWhiteSpace(identity.Label))
        {
            identity.Label = referenceMatch.Identity.DisplayName;
        }

        return true;
    }

    private static StoredFaceIdentity CreateIdentity(FaceIdentitySnapshot snapshot, string assetId, PreparedFaceTrack track, FaceReferenceMatch? referenceMatch)
    {
        var identity = new StoredFaceIdentity
        {
            FaceKey = $"face-{snapshot.NextIdentityOrdinal:0000}",
            QualityScore = track.TrackQuality,
            ObservationCount = 0,
            CoverAssetId = assetId,
            CoverBoundingBox = new StoredBoundingBox(
                track.BestSample.Detection.BoundingBox.X1,
                track.BestSample.Detection.BoundingBox.Y1,
                track.BestSample.Detection.BoundingBox.X2,
                track.BestSample.Detection.BoundingBox.Y2),
        };
        TryApplyReferenceMatch(identity, referenceMatch);

        snapshot.NextIdentityOrdinal++;
        snapshot.Identities.Add(identity);
        return identity;
    }

    private static bool TryPromoteIdentity(AiDispatchRequest request, StoredFaceIdentity identity, PreparedFaceTrack track, AiFacesSettings settings, bool referenceApplied)
    {
        if (IsPromoted(identity))
        {
            return false;
        }

        var promotionReason = ResolvePromotionReason(request, identity, track, settings, referenceApplied);
        if (promotionReason is null)
        {
            return false;
        }

        identity.LifecycleStatus = StoredFaceIdentityLifecycle.Promoted;
        identity.PromotionReason = promotionReason;
        return true;
    }

    private static string? ResolvePromotionReason(AiDispatchRequest request, StoredFaceIdentity identity, PreparedFaceTrack track, AiFacesSettings settings, bool referenceApplied)
    {
        if (referenceApplied || !string.IsNullOrWhiteSpace(identity.ReferenceExternalId))
        {
            return "reference";
        }

        if (request.Result.MediaKind == AiMediaKinds.Image)
        {
            return "image";
        }

        if (identity.AssetIds.Count > 1)
        {
            return "multi-asset";
        }

        if (request.Result.MediaKind == AiMediaKinds.Video && HasSufficientVideoPromotionEvidence(request, track, settings))
        {
            return "video-evidence";
        }

        return null;
    }

    private static bool HasSufficientVideoPromotionEvidence(AiDispatchRequest request, PreparedFaceTrack track, AiFacesSettings settings)
    {
        if (track.Samples.Count >= settings.PromotionMinimumVideoSamples)
        {
            return true;
        }

        var frameIntervalSeconds = ResolveFrameIntervalSeconds(request, track);
        if (frameIntervalSeconds < settings.SparseVideoPromotionFrameIntervalSeconds)
        {
            return false;
        }

        if (IsSingleSampleShortSparseVideo(request, track, frameIntervalSeconds))
        {
            return true;
        }

        var requiredSamples = ResolveSparseVideoPromotionMinimumSamples(request, frameIntervalSeconds, settings);
        return track.Samples.Count >= requiredSamples
            && EstimateRepresentedVideoEvidenceSeconds(request, track, frameIntervalSeconds) >= settings.PromotionMinimumVideoEvidenceSeconds;
    }

    private static bool ShouldEmitPromotedIdentity(AiDispatchRequest request, StoredFaceIdentity identity, PreparedFaceTrack track, AiFacesSettings settings)
    {
        if (!IsPromoted(identity))
        {
            return false;
        }

        if (request.Result.MediaKind != AiMediaKinds.Video)
        {
            return true;
        }

        if (!string.Equals(identity.PromotionReason, "video-evidence", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (!string.IsNullOrWhiteSpace(identity.ReferenceExternalId))
        {
            return true;
        }

        var currentAssetId = request.Context.AssetId;
        if (!string.IsNullOrWhiteSpace(currentAssetId)
            && identity.AssetIds.Any(assetId => !string.Equals(assetId, currentAssetId, StringComparison.OrdinalIgnoreCase)))
        {
            return true;
        }

        return HasSufficientVideoPromotionEvidence(request, track, settings);
    }

    private static int ResolveSparseVideoPromotionMinimumSamples(AiDispatchRequest request, double frameIntervalSeconds, AiFacesSettings settings)
    {
        var requiredSamples = settings.PromotionMinimumSparseVideoSamples;
        var durationSeconds = ResolveDurationSeconds(request);
        if (durationSeconds is > 0.0 && settings.PromotionMinimumSparseVideoSampleCoverageRatio > 0.0)
        {
            var sampledFrameEstimate = Math.Max(1.0, Math.Ceiling(durationSeconds.Value / frameIntervalSeconds));
            var coverageSamples = (int)Math.Ceiling(sampledFrameEstimate * settings.PromotionMinimumSparseVideoSampleCoverageRatio);
            requiredSamples = Math.Max(requiredSamples, coverageSamples);
        }

        return Math.Min(settings.PromotionMinimumVideoSamples, requiredSamples);
    }

    private static double ResolveFrameIntervalSeconds(AiDispatchRequest request, PreparedFaceTrack track)
        => Math.Max(1.0,
            track.FrameIntervalSeconds
            ?? request.Result.FrameIntervalSeconds
            ?? request.Context.FrameIntervalSeconds
            ?? DefaultFrameSpanSeconds);

    private static bool IsSingleSampleShortSparseVideo(AiDispatchRequest request, PreparedFaceTrack track, double frameIntervalSeconds)
    {
        if (track.Samples.Count != 1)
        {
            return false;
        }

        var durationSeconds = ResolveDurationSeconds(request);
        return durationSeconds is > 0.0 && durationSeconds <= frameIntervalSeconds;
    }

    private static double EstimateRepresentedVideoEvidenceSeconds(AiDispatchRequest request, PreparedFaceTrack track, double frameIntervalSeconds)
    {
        var durationSeconds = ResolveDurationSeconds(request);
        var sampleCoverageSeconds = track.Samples.Count * frameIntervalSeconds;
        if (durationSeconds is > 0.0)
        {
            sampleCoverageSeconds = Math.Min(sampleCoverageSeconds, durationSeconds.Value);
        }

        var spanCoverageSeconds = 0.0;
        if (track.StartSeconds.HasValue && track.EndSeconds.HasValue && track.EndSeconds.Value >= track.StartSeconds.Value)
        {
            spanCoverageSeconds = (track.EndSeconds.Value - track.StartSeconds.Value) + frameIntervalSeconds;
            if (durationSeconds is > 0.0)
            {
                spanCoverageSeconds = Math.Min(spanCoverageSeconds, durationSeconds.Value);
            }
        }

        return Math.Max(sampleCoverageSeconds, spanCoverageSeconds);
    }

    private static double? ResolveDurationSeconds(AiDispatchRequest request)
        => request.Result.DurationSeconds ?? request.Context.DurationSeconds;

    private static bool IsPromoted(StoredFaceIdentity identity)
        => string.Equals(identity.LifecycleStatus, StoredFaceIdentityLifecycle.Promoted, StringComparison.OrdinalIgnoreCase);

    private static StoredFaceIdentity? ResolveFinalIdentity(
        StoredFaceIdentity? identity,
        FaceIdentitySnapshot snapshot,
        AiFaceIdentityReconciliationReport reconciliation)
    {
        if (identity is null)
        {
            return null;
        }

        if (snapshot.Identities.Contains(identity))
        {
            return identity;
        }

        if (reconciliation.MergedFaceKeyMap.TryGetValue(identity.FaceKey, out var targetFaceKey))
        {
            return snapshot.Identities.FirstOrDefault(candidate => string.Equals(candidate.FaceKey, targetFaceKey, StringComparison.OrdinalIgnoreCase));
        }

        return snapshot.Identities.FirstOrDefault(candidate => string.Equals(candidate.FaceKey, identity.FaceKey, StringComparison.OrdinalIgnoreCase));
    }

    private static void UpdateIdentity(StoredFaceIdentity identity, string assetId, PreparedFaceTrack track, IReadOnlyList<RepresentativeFaceEmbedding> anchors)
    {
        identity.ObservationCount += track.Samples.Count;
        RememberIdentityAsset(identity, assetId);
        var coverQualityScore = AiFaceQualityScorer.ScoreCoverQuality(track.Samples, track.BestSample);
        var shouldRefreshCover = identity.CoverBoundingBox is null
            || string.IsNullOrWhiteSpace(identity.CoverAssetId)
            || !HasUsableCover(identity.CoverBoundingBox)
            || coverQualityScore > identity.CoverQualityScore;
        identity.QualityScore = Math.Max(identity.QualityScore, track.TrackQuality);
        if (shouldRefreshCover)
        {
            identity.CoverAssetId = assetId;
            identity.CoverBoundingBox = new StoredBoundingBox(
                track.BestSample.Detection.BoundingBox.X1,
                track.BestSample.Detection.BoundingBox.Y1,
                track.BestSample.Detection.BoundingBox.X2,
                track.BestSample.Detection.BoundingBox.Y2);
            identity.CoverQualityScore = coverQualityScore;
        }

        foreach (var anchor in anchors.OrderByDescending(static anchor => anchor.QualityScore))
        {
            if (identity.Anchors.Any(existing => CosineSimilarity(existing.Vector, anchor.Vector) >= IdentityAnchorDuplicateSimilarity))
            {
                continue;
            }

            identity.Anchors.Add(new StoredFaceAnchor
            {
                ModelKey = anchor.ModelKey,
                QualityScore = anchor.QualityScore,
                Vector = anchor.Vector.ToList(),
            });
        }

        if (identity.Anchors.Count > MaxAnchorsPerIdentity)
        {
            identity.Anchors = identity.Anchors
                .OrderByDescending(static anchor => anchor.QualityScore)
                .Take(MaxAnchorsPerIdentity)
                .ToList();
        }
    }

    private static void RememberIdentityAsset(StoredFaceIdentity identity, string assetId)
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

    private static void EmitFace(AiPreparedArtifactBatch batch, AiDispatchRequest request, StoredFaceIdentity identity, PreparedFaceTrack track, HashSet<string> emittedFaces, bool wasCreated)
    {
        if (!emittedFaces.Add(identity.FaceKey))
        {
            return;
        }

        var coverAssetId = string.IsNullOrWhiteSpace(identity.CoverAssetId)
            ? request.Context.AssetId
            : identity.CoverAssetId;
        var coverBoundingBox = identity.CoverBoundingBox is null
            ? track.BestSample.Detection.BoundingBox
            : new AiBoundingBox(
                identity.CoverBoundingBox.X1,
                identity.CoverBoundingBox.Y1,
                identity.CoverBoundingBox.X2,
                identity.CoverBoundingBox.Y2);
        var metadata = new Dictionary<string, string>
        {
            ["runId"] = request.Context.RunId,
            ["status"] = wasCreated ? "created" : "matched",
            ["lifecycle"] = identity.LifecycleStatus,
            ["promotionReason"] = identity.PromotionReason ?? string.Empty,
            ["observationCount"] = identity.ObservationCount.ToString(CultureInfo.InvariantCulture),
            ["coverQualityScore"] = identity.CoverQualityScore.ToString("R", CultureInfo.InvariantCulture),
        };
        if (!string.IsNullOrWhiteSpace(identity.ReferenceExternalId))
        {
            metadata["referenceExternalId"] = identity.ReferenceExternalId;
        }

        if (!string.IsNullOrWhiteSpace(identity.ReferencePackId))
        {
            metadata["referencePackId"] = identity.ReferencePackId;
        }

        if (identity.ReferenceSuggestionId.HasValue)
        {
            metadata["referenceSuggestionId"] = identity.ReferenceSuggestionId.Value.ToString(CultureInfo.InvariantCulture);
        }

        batch.Faces.Add(new AiPreparedFaceIdentity(
            identity.FaceKey,
            SourceKey,
            identity.Label,
            IsProvisional: !IsPromoted(identity),
            QualityScore: identity.QualityScore,
            CoverAssetId: coverAssetId,
            CoverBoundingBox: coverBoundingBox,
            Metadata: metadata));
    }

    private static bool HasUsableCover(StoredBoundingBox? coverBoundingBox)
    {
        if (coverBoundingBox is null)
        {
            return false;
        }

        return IsFinite(coverBoundingBox.X1)
            && IsFinite(coverBoundingBox.Y1)
            && IsFinite(coverBoundingBox.X2)
            && IsFinite(coverBoundingBox.Y2)
            && coverBoundingBox.X2 > coverBoundingBox.X1
            && coverBoundingBox.Y2 > coverBoundingBox.Y1;
    }

    private static bool IsFinite(double value)
        => !double.IsNaN(value) && !double.IsInfinity(value);

    private static IReadOnlyList<FaceFrameSample> EmitDetections(AiPreparedArtifactBatch batch, AiDispatchRequest request, PreparedFaceTrack track, string? faceKey, AiFacesSettings settings)
        => EmitDetections(batch, request, track, faceKey, BuildFaceTrackWindows(track), settings);

    private static IReadOnlyList<FaceFrameSample> EmitDetections(
        AiPreparedArtifactBatch batch,
        AiDispatchRequest request,
        PreparedFaceTrack track,
        string? faceKey,
        IReadOnlyList<FaceTrackWindow> windows,
        AiFacesSettings settings)
    {
        var retainedSamples = SelectDetectionKeyframes(track, settings);
        foreach (var sample in retainedSamples)
        {
            var window = windows.FirstOrDefault(candidate => candidate.Contains(sample));
            var groupKey = window?.GroupKey ?? track.TrackKey;
            batch.Detections.Add(new AiPreparedDetection(
                request.Context.AssetId,
                SourceKey,
                Class: "face",
                ObservedAtSeconds: sample.TimeSeconds,
                Score: sample.Detection.Score,
                BoundingBox: sample.Detection.BoundingBox,
                ModelKey: sample.Detection.ModelKey,
                RefKind: faceKey is null ? null : "face",
                RefKey: faceKey,
                GroupKey: groupKey,
                Metadata: new Dictionary<string, string>
                {
                    ["runId"] = request.Context.RunId,
                    ["trackKey"] = groupKey,
                    ["clusterTrackKey"] = track.TrackKey,
                    ["sampleCount"] = (window?.Samples.Count ?? track.Samples.Count).ToString(CultureInfo.InvariantCulture),
                    ["role"] = ResolveKeyframeRole(window?.Samples ?? track.Samples, window?.BestSample ?? track.BestSample, sample),
                }));
        }

        return retainedSamples;
    }

    private static IReadOnlyList<FaceFrameSample> SelectDetectionKeyframes(PreparedFaceTrack track, AiFacesSettings settings)
        => SelectDetectionKeyframes(track.Samples, track.BestSample, settings);

    private static IReadOnlyList<FaceFrameSample> SelectDetectionKeyframes(
        IReadOnlyList<FaceFrameSample> samples,
        FaceFrameSample bestSample,
        AiFacesSettings settings)
    {
        if (samples.Count == 0)
        {
            return [];
        }

        var orderedSamples = samples
            .OrderBy(static sample => sample.TimeSeconds ?? double.MinValue)
            .ThenBy(static sample => sample.FrameOrder)
            .ToArray();

        var selected = new List<FaceFrameSample> { orderedSamples[0] };
        var lastMaterialSample = orderedSamples[0];

        foreach (var sample in orderedSamples.Skip(1))
        {
            if (!HasMaterialDetectionChange(lastMaterialSample, sample, settings.DetectionKeyframeIoUThreshold))
            {
                continue;
            }

            selected.Add(sample);
            lastMaterialSample = sample;
        }

        AddIfMateriallyDistinct(selected, bestSample, settings.DetectionKeyframeIoUThreshold);
        AddIfMateriallyDistinct(selected, orderedSamples[^1], settings.DetectionKeyframeIoUThreshold);

        return CapDetectionKeyframes(selected, bestSample, settings.MaxDetectionKeyframesPerTrack);
    }

    private static void AddIfMateriallyDistinct(List<FaceFrameSample> selected, FaceFrameSample sample, double iouThreshold)
    {
        if (selected.Any(existing => IsSameSample(existing, sample)))
        {
            return;
        }

        if (selected.Any(existing => !HasMaterialDetectionChange(existing, sample, iouThreshold)))
        {
            return;
        }

        selected.Add(sample);
    }

    private static IReadOnlyList<FaceFrameSample> CapDetectionKeyframes(List<FaceFrameSample> selected, FaceFrameSample bestSample, int maxCount)
    {
        var distinct = selected
            .DistinctBy(GetSampleIdentity)
            .OrderBy(static sample => sample.TimeSeconds ?? double.MinValue)
            .ThenBy(static sample => sample.FrameOrder)
            .ToArray();
        if (distinct.Length <= maxCount)
        {
            return distinct;
        }

        var required = new List<FaceFrameSample>();
        AddRequired(required, distinct[0]);
    AddRequired(required, bestSample);
        AddRequired(required, distinct[^1]);

        var remainingSlots = Math.Max(0, maxCount - required.Count);
        var optional = distinct
            .Where(sample => !required.Any(requiredSample => IsSameSample(requiredSample, sample)))
            .ToArray();

        if (remainingSlots > 0 && optional.Length > 0)
        {
            if (remainingSlots == 1)
            {
                AddRequired(required, optional[optional.Length / 2]);
            }
            else
            {
                for (var index = 0; index < remainingSlots; index++)
                {
                    var optionalIndex = (int)Math.Round(index * (optional.Length - 1) / (double)(remainingSlots - 1), MidpointRounding.AwayFromZero);
                    AddRequired(required, optional[optionalIndex]);
                }
            }
        }

        return required
            .DistinctBy(GetSampleIdentity)
            .OrderBy(static sample => sample.TimeSeconds ?? double.MinValue)
            .ThenBy(static sample => sample.FrameOrder)
            .ToArray();

        static void AddRequired(List<FaceFrameSample> required, FaceFrameSample sample)
        {
            if (!required.Any(existing => IsSameSample(existing, sample)))
            {
                required.Add(sample);
            }
        }
    }

    private static string GetSampleIdentity(FaceFrameSample sample)
        => $"{sample.FrameOrder}\u001F{sample.TimeSeconds?.ToString("R", CultureInfo.InvariantCulture)}\u001F{FormatBoundingBox(sample.Detection.BoundingBox)}";

    private static bool HasMaterialDetectionChange(FaceFrameSample previous, FaceFrameSample current, double iouThreshold)
    {
        var previousBox = previous.Detection.BoundingBox;
        var currentBox = current.Detection.BoundingBox;
        if (previousBox.Area <= 0.0 || currentBox.Area <= 0.0)
        {
            return true;
        }

        return ComputeIoU(previousBox, currentBox) < iouThreshold;
    }

    private static bool IsSameSample(FaceFrameSample left, FaceFrameSample right)
        => left.FrameOrder == right.FrameOrder
           && Nullable.Equals(left.TimeSeconds, right.TimeSeconds)
           && BoundingBoxesMatch(left.Detection.BoundingBox, right.Detection.BoundingBox);

    private static bool BoundingBoxesMatch(AiBoundingBox left, AiBoundingBox right)
        => Math.Abs(left.X1 - right.X1) <= 0.0001
           && Math.Abs(left.Y1 - right.Y1) <= 0.0001
           && Math.Abs(left.X2 - right.X2) <= 0.0001
           && Math.Abs(left.Y2 - right.Y2) <= 0.0001;

    private static string ResolveKeyframeRole(IReadOnlyList<FaceFrameSample> samples, FaceFrameSample bestSample, FaceFrameSample sample)
    {
        if (samples.Count == 0)
        {
            return "best";
        }

        if (IsSameSample(sample, samples[0]))
        {
            return "first";
        }

        if (IsSameSample(sample, bestSample))
        {
            return "best";
        }

        if (IsSameSample(sample, samples[^1]))
        {
            return "last";
        }

        return "motion-keyframe";
    }

    private static void EmitFaceAppearance(
        AiPreparedArtifactBatch batch,
        AiDispatchRequest request,
        PreparedFaceTrack track,
        string faceKey,
        IReadOnlyList<FaceFrameSample> retainedSamples,
        int segmentCount)
    {
        batch.FaceAppearances.Add(new AiPreparedFaceAppearance(
            request.Context.AssetId,
            SourceKey,
            SampleCount: track.Samples.Count,
            RetainedSpatialSampleCount: retainedSamples.Count,
            SegmentCount: segmentCount,
            FirstSeenSeconds: track.StartSeconds,
            LastSeenSeconds: track.EndSeconds,
            TopConfidence: track.Samples.Select(static sample => sample.Detection.Score).DefaultIfEmpty(0.0).Max(),
            RepresentativeFrameSeconds: track.BestSample.TimeSeconds,
            RefKind: "face",
            RefKey: faceKey,
            GroupKey: track.TrackKey,
            Metadata: new Dictionary<string, string>
            {
                ["runId"] = request.Context.RunId,
                ["trackKey"] = track.TrackKey,
                ["modelKey"] = track.BestSample.Detection.ModelKey,
                ["bestScore"] = track.BestSample.Detection.Score.ToString("R", CultureInfo.InvariantCulture),
                ["bestBbox"] = FormatBoundingBox(track.BestSample.Detection.BoundingBox),
            }));
    }

    private static void EmitTrackEmbedding(AiPreparedArtifactBatch batch, AiDispatchRequest request, PreparedFaceTrack track, string faceKey, IReadOnlyList<RepresentativeFaceEmbedding> embeddings)
    {
        var centroid = BuildCentroid(embeddings);
        if (centroid is null)
        {
            return;
        }

        batch.Embeddings.Add(new AiPreparedEmbedding(
            request.Context.AssetId,
            SourceKey,
            "face.embed.v1",
            "face.v1",
            "Face",
            false,
            centroid.Value.Vector,
            centroid.Value.Norm,
            HostRefKind: "face",
            HostRefKey: faceKey,
            StartSeconds: track.StartSeconds,
            EndSeconds: track.EndSeconds,
            Metadata: new Dictionary<string, string>
            {
                ["runId"] = request.Context.RunId,
                ["trackKey"] = track.TrackKey,
                ["embeddingCount"] = embeddings.Count.ToString(System.Globalization.CultureInfo.InvariantCulture),
            }));
    }

    private static int EmitSegments(
        AiPreparedArtifactBatch batch,
        AiDispatchRequest request,
        PreparedFaceTrack track,
        string faceKey,
        IReadOnlyList<FaceTrackWindow> windows,
        IReadOnlyList<FaceFrameSample> retainedSamples,
        AiFacesSettings settings)
    {
        var emitted = 0;
        foreach (var window in windows)
        {
            if (!window.StartSeconds.HasValue)
            {
                continue;
            }

            var windowKeyframes = retainedSamples.Where(window.Contains).ToArray();
            if (windowKeyframes.Length == 0)
            {
                windowKeyframes = SelectDetectionKeyframes(window.Samples, window.BestSample, settings).ToArray();
            }

            batch.Segments.Add(new AiPreparedSegment(
                request.Context.AssetId,
                SourceKey,
                Kind: "face",
                StartSeconds: window.StartSeconds.Value,
                EndSeconds: window.EndSeconds,
                Title: faceKey,
                Confidence: window.BestSample.Detection.Score,
                RefKind: "face",
                RefKey: faceKey,
                Metadata: new Dictionary<string, string>
                {
                    ["runId"] = request.Context.RunId,
                    ["trackKey"] = window.GroupKey,
                    ["clusterTrackKey"] = track.TrackKey,
                    ["modelKey"] = window.BestSample.Detection.ModelKey,
                    ["sampleCount"] = window.Samples.Count.ToString(CultureInfo.InvariantCulture),
                    ["retainedSpatialSampleCount"] = windowKeyframes.Length.ToString(CultureInfo.InvariantCulture),
                    ["frameIntervalSec"] = track.FrameIntervalSeconds?.ToString("R", CultureInfo.InvariantCulture) ?? string.Empty,
                    ["bestTimeSec"] = window.BestSample.TimeSeconds?.ToString("R", CultureInfo.InvariantCulture) ?? string.Empty,
                    ["bestScore"] = window.BestSample.Detection.Score.ToString("R", CultureInfo.InvariantCulture),
                    ["bestBbox"] = FormatBoundingBox(window.BestSample.Detection.BoundingBox),
                    ["keyframes"] = FormatKeyframes(windowKeyframes),
                }));
            emitted++;
        }

        return emitted;
    }

    private static IReadOnlyList<FaceTrackWindow> BuildFaceTrackWindows(PreparedFaceTrack track)
    {
        if (track.Samples.Count == 0)
        {
            return [];
        }

        var orderedSamples = track.Samples
            .OrderBy(static sample => sample.TimeSeconds ?? double.MinValue)
            .ThenBy(static sample => sample.FrameOrder)
            .ToArray();
        var breakThresholdSeconds = ResolveTrackSegmentBreakThresholdSeconds(track);
        var windows = new List<FaceTrackWindow>();
        var current = new List<FaceFrameSample> { orderedSamples[0] };
        var previousMoment = ResolveSampleMoment(orderedSamples[0], track.FrameIntervalSeconds);

        foreach (var sample in orderedSamples.Skip(1))
        {
            var sampleMoment = ResolveSampleMoment(sample, track.FrameIntervalSeconds);
            if (previousMoment.HasValue
                && sampleMoment.HasValue
                && (sampleMoment.Value - previousMoment.Value) > breakThresholdSeconds)
            {
                windows.Add(CreateFaceTrackWindow(track, current.ToArray()));
                current.Clear();
            }

            current.Add(sample);
            if (sampleMoment.HasValue)
            {
                previousMoment = sampleMoment;
            }
        }

        if (current.Count > 0)
        {
            windows.Add(CreateFaceTrackWindow(track, current.ToArray()));
        }

        if (windows.Count <= 1)
        {
            return windows;
        }

        return windows
            .Select((window, index) => window with { GroupKey = $"{track.TrackKey}:span-{index + 1}" })
            .ToArray();
    }

    private static FaceTrackWindow CreateFaceTrackWindow(PreparedFaceTrack track, IReadOnlyList<FaceFrameSample> samples)
    {
        var bestSample = samples.FirstOrDefault(sample => IsSameSample(sample, track.BestSample))
            ?? samples.OrderByDescending(static sample => sample.Detection.Score)
                .ThenBy(static sample => sample.TimeSeconds ?? double.MinValue)
                .First();

        return new FaceTrackWindow(
            track.TrackKey,
            samples,
            bestSample,
            ResolveSampleMoment(samples[0], track.FrameIntervalSeconds),
            ResolveSampleWindowEnd(samples[^1], track.FrameIntervalSeconds),
            samples.Select(GetSampleIdentity).ToHashSet(StringComparer.Ordinal));
    }

    private static double ResolveTrackSegmentBreakThresholdSeconds(PreparedFaceTrack track)
    {
        var frameIntervalSeconds = Math.Max(0.25, track.FrameIntervalSeconds ?? DefaultFrameSpanSeconds);
        return Math.Max(8.0, frameIntervalSeconds * 4.0);
    }

    private static double? ResolveSampleMoment(FaceFrameSample sample, double? frameIntervalSeconds)
    {
        if (sample.TimeSeconds.HasValue)
        {
            return sample.TimeSeconds.Value;
        }

        if (frameIntervalSeconds is > 0.0 && sample.FrameOrder >= 0)
        {
            return sample.FrameOrder * frameIntervalSeconds.Value;
        }

        return null;
    }

    private static double? ResolveSampleWindowEnd(FaceFrameSample sample, double? frameIntervalSeconds)
    {
        var moment = ResolveSampleMoment(sample, frameIntervalSeconds);
        if (!moment.HasValue)
        {
            return null;
        }

        var span = Math.Max(0.25, frameIntervalSeconds ?? DefaultFrameSpanSeconds);
        return moment.Value + span;
    }

    private static string FormatBoundingBox(AiBoundingBox boundingBox)
        => JsonSerializer.Serialize(new[]
        {
            boundingBox.X1,
            boundingBox.Y1,
            boundingBox.X2,
            boundingBox.Y2,
        });

    private static string FormatKeyframes(IReadOnlyList<FaceFrameSample> samples)
        => JsonSerializer.Serialize(samples.Select(static sample => new
        {
            t = sample.TimeSeconds,
            bbox = new[]
            {
                sample.Detection.BoundingBox.X1,
                sample.Detection.BoundingBox.Y1,
                sample.Detection.BoundingBox.X2,
                sample.Detection.BoundingBox.Y2,
            },
            score = sample.Detection.Score,
        }));

    private static (IReadOnlyList<float> Vector, double Norm)? BuildCentroid(IEnumerable<RepresentativeFaceEmbedding> embeddings)
    {
        var materialized = embeddings.Where(static embedding => embedding.Vector.Count > 0).ToArray();
        if (materialized.Length == 0)
        {
            return null;
        }

        var length = materialized[0].Vector.Count;
        var buffer = new double[length];
        var totalWeight = 0.0;
        foreach (var embedding in materialized)
        {
            var weight = Math.Max(embedding.QualityScore, 1e-6);
            totalWeight += weight;
            for (var index = 0; index < length; index++)
            {
                buffer[index] += embedding.Vector[index] * weight;
            }
        }

        var divisor = totalWeight > 0.0 ? totalWeight : materialized.Length;
        var averaged = buffer.Select(value => (float)(value / divisor)).ToArray();
        var norm = Math.Sqrt(averaged.Sum(static value => value * value));
        if (norm > 0)
        {
            for (var index = 0; index < averaged.Length; index++)
            {
                averaged[index] = (float)(averaged[index] / norm);
            }
        }

        return (averaged, norm);
    }

    private static double ScoreTrack(IReadOnlyList<FaceFrameSample> samples)
    {
        return AiFaceQualityScorer.ScoreIdentityEvidence(samples);
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

        if (leftNorm <= 0 || rightNorm <= 0)
        {
            return 0.0;
        }

        return dot / (Math.Sqrt(leftNorm) * Math.Sqrt(rightNorm));
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

    private static double ComputeIoU(AiBoundingBox left, AiBoundingBox right)
    {
        var x1 = Math.Max(left.X1, right.X1);
        var y1 = Math.Max(left.Y1, right.Y1);
        var x2 = Math.Min(left.X2, right.X2);
        var y2 = Math.Min(left.Y2, right.Y2);

        var intersection = Math.Max(0.0, x2 - x1) * Math.Max(0.0, y2 - y1);
        if (intersection <= 0)
        {
            return 0.0;
        }

        var union = left.Area + right.Area - intersection;
        return union <= 0 ? 0.0 : intersection / union;
    }

    private static double GetMetadataQuality(AiEmbeddingObservation embedding, string key)
        => AiFaceQualityScorer.GetMetadataQuality(embedding, key);

    private static void AddTelemetryNotes(
        AiPreparedArtifactBatch batch,
        AiFacesSettings settings,
        AiAssetFaceClusterDiagnostics clusterDiagnostics,
        AiFaceIdentityReconciliationReport initialReconciliation,
        AiFaceIdentityReconciliationReport finalReconciliation,
        int newIdentityCount,
        int promotedIdentityCount,
        int seededMatchCount,
        int conflictingReferenceCount,
        int unresolvedTracks,
        int provisionalClusters)
    {
        batch.Notes.Add(string.Create(
            CultureInfo.InvariantCulture,
            $"AI.Faces telemetry: rawTracks={clusterDiagnostics.InputTrackCount}; assetClusters={clusterDiagnostics.ClusterCount}; clusterMerges={clusterDiagnostics.MergedTrackCount}; clusterRejectedConcurrency={clusterDiagnostics.RejectedByConcurrencyCount}; clusterRejectedThreshold={clusterDiagnostics.RejectedByThresholdCount}; clusterRejectedAmbiguous={clusterDiagnostics.RejectedByAmbiguityCount}; createdIdentities={newIdentityCount}; promotedThisRun={promotedIdentityCount}; provisionalClusters={provisionalClusters}; seededMatches={seededMatchCount}; conflictingReferences={conflictingReferenceCount}; unresolvedTracks={unresolvedTracks}; faces={batch.Faces.Count}; detections={batch.Detections.Count}; reconciliationMerges={initialReconciliation.MergedIdentityCount + finalReconciliation.MergedIdentityCount}; reconciliationReferencePromotions={initialReconciliation.ReferencePromotedIdentityCount + finalReconciliation.ReferencePromotedIdentityCount}; reconciliationEvidencePromotions={initialReconciliation.EvidencePromotedIdentityCount + finalReconciliation.EvidencePromotedIdentityCount}"));
        batch.Notes.Add(string.Create(
            CultureInfo.InvariantCulture,
                $"AI.Faces thresholds: identity={settings.IdentityMatchThreshold:R}/{settings.IdentityAmbiguityMargin:R}; assetCluster={settings.AssetClusterSimilarityThreshold:R}/{settings.AssetClusterAmbiguityMargin:R}; reference={settings.ReferenceMatchThreshold:R}/{settings.ReferenceAmbiguityMargin:R}; consolidation={settings.ConsolidationSimilarityThreshold:R}/{settings.ConsolidationAmbiguityMargin:R}; sameAssetConsolidation={settings.ConsolidationSameAssetSimilarityThreshold:R}; videoPromotionSamples={settings.PromotionMinimumVideoSamples}; videoPromotionEvidenceSeconds={settings.PromotionMinimumVideoEvidenceSeconds:R}; sparseVideoPromotionSamples={settings.PromotionMinimumSparseVideoSamples}; sparseVideoPromotionFrameInterval={settings.SparseVideoPromotionFrameIntervalSeconds:R}; sparseVideoPromotionCoverageRatio={settings.PromotionMinimumSparseVideoSampleCoverageRatio:R}"));
    }

    private sealed class OpenTrack
    {
        private readonly List<FaceFrameSample> _samples;

        public OpenTrack(string trackKey, FaceFrameSample sample)
        {
            TrackKey = trackKey;
            _samples = [sample];
            LastBoundingBox = sample.Detection.BoundingBox;
            LastSample = sample;
            LastFrameOrder = sample.FrameOrder;
        }

        public string TrackKey { get; }

        public AiBoundingBox LastBoundingBox { get; private set; }

        public FaceFrameSample LastSample { get; private set; }

        public int LastFrameOrder { get; private set; }

        public void Append(FaceFrameSample sample)
        {
            _samples.Add(sample);
            LastBoundingBox = sample.Detection.BoundingBox;
            LastSample = sample;
            LastFrameOrder = sample.FrameOrder;
        }

        public PreparedFaceTrack Close(double frameIntervalSeconds)
        {
            return CreatePreparedTrack(
                TrackKey,
                _samples.ToArray(),
                _samples.First().TimeSeconds,
                _samples.Last().TimeSeconds,
                frameIntervalSeconds);
        }
    }

    private sealed record PreparedFaceAssignment(
        PreparedFaceTrack Track,
        StoredFaceIdentity? Identity,
        IReadOnlyList<RepresentativeFaceEmbedding> IdentityEmbeddings,
        bool WasCreated);

    private sealed record FaceTrackWindow(
        string GroupKey,
        IReadOnlyList<FaceFrameSample> Samples,
        FaceFrameSample BestSample,
        double? StartSeconds,
        double? EndSeconds,
        HashSet<string> SampleKeys)
    {
        public bool Contains(FaceFrameSample sample)
            => SampleKeys.Contains(GetSampleIdentity(sample));
    }

    private sealed record FaceReferenceMatch(
        SaieReferenceIdentity Identity,
        string PackId,
        int SuggestionId,
        double Score
    );
}