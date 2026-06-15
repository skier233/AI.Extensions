using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Media;

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
    private const double MinimumNormalizedAnchorArea = 0.01;
    private const double MinimumPixelAnchorArea = 4096.0;
    private const double RepresentativeDedupSimilarity = 0.96;
    private const double IdentityAnchorDuplicateSimilarity = 0.975;
    private const double SameAssetIdentityMatchRelaxation = 0.07;
    private const int MaxRepresentativeEmbeddings = 6;
    private const int MaxAnchorsPerIdentity = 12;
    private const int MaxAssetIdsPerIdentity = 32;

    // Candidate identities loaded per query embedding for the incremental reconcile. Generous enough to
    // cover both the best identity match and the local merge neighborhood, while keeping the working set
    // bounded regardless of total corpus size.
    private const int CandidateK = 20;

    private readonly IFaceIdentityStore _store;
    private readonly AiAssetFaceClusterer _assetClusterer;
    private readonly AiFaceIdentityReconciler _identityReconciler;
    private readonly AiFaceReferencePackStore? _referencePackStore;

    public AiFacePreparationService(IFaceIdentityStore store)
        : this(store, new AiAssetFaceClusterer(), new AiFaceIdentityReconciler(), null)
    {
    }

    public AiFacePreparationService(
        IFaceIdentityStore store,
        AiAssetFaceClusterer assetClusterer,
        AiFaceIdentityReconciler? identityReconciler = null,
        AiFaceReferencePackStore? referencePackStore = null)
    {
        _store = store;
        _assetClusterer = assetClusterer;
        _identityReconciler = identityReconciler ?? new AiFaceIdentityReconciler();
        _referencePackStore = referencePackStore;
    }

    public async Task<AiPreparedArtifactBatch> PrepareAsync(AiDispatchRequest request, CancellationToken ct = default)
        => (await PrepareWithReportAsync(request, ct)).Batch;

    public async Task<AiFacePreparationOutcome> PrepareWithReportAsync(AiDispatchRequest request, CancellationToken ct = default)
    {
        var batch = new AiPreparedArtifactBatch();
        if (request.Result.MediaKind is not (AiMediaKinds.Image or AiMediaKinds.Video))
        {
            batch.Notes.Add($"AI.Faces does not consume media kind '{request.Result.MediaKind}'.");
            return new AiFacePreparationOutcome(batch, EmptyMergedFaceKeyMap);
        }

        var settings = await AiFacesSettingsRuntime.LoadAsync(ct);
        var rawTracks = request.Result.MediaKind == AiMediaKinds.Image
            ? BuildImageTracks(request)
            : BuildVideoTracks(request);
        var clusterResult = _assetClusterer.ClusterWithDiagnostics(rawTracks, settings);
        var tracks = clusterResult.Tracks;

        if (tracks.Count == 0)
        {
            batch.Notes.Add("No face detections were available to prepare.");
            return new AiFacePreparationOutcome(batch, EmptyMergedFaceKeyMap);
        }

        // Seeding/clustering currently reconciles against a single pack; with several packs active we
        // seed from the first (deterministically ordered by pack id). Cross-pack suggestion matching is
        // handled later by the suggester, which scores against every active pack.
        var referencePacks = _referencePackStore is null
            ? (IReadOnlyList<SaieReferencePack>)Array.Empty<SaieReferencePack>()
            : await _referencePackStore.GetActivePacksAsync(ct);
        var referencePack = referencePacks.Count > 0 ? referencePacks[0] : null;

        // Load only the identity-graph candidates relevant to this asset (similarity neighbors of the
        // track embeddings, plus any reference-linked identities), instead of the whole graph. The
        // reconcile/match logic below operates unchanged on this bounded working snapshot; the
        // transaction persists only the resulting deltas.
        var (candidateVectors, candidateReferenceIds) = CollectCandidateKeys(tracks, referencePack, settings);
        await using var transaction = await _store.BeginIncrementalAsync(candidateVectors, candidateReferenceIds, CandidateK, ct);
        var snapshot = transaction.Snapshot;
        // Per-asset: skip whole-snapshot reference re-matching (the dominant cost with a large pack). The
        // assignment loop below reference-matches this image's own faces; loaded candidates are already
        // settled. Pack-import backfill still does the full re-match.
        var initialReconciliation = _identityReconciler.Reconcile(snapshot, referencePack, settings, applyReferenceMatches: false);

        var emittedFaces = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var unresolvedTracks = 0;
        var lowQualityCreationSkips = 0;
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
                // Only mint a new identity from anchor-grade evidence (or a confident reference-pack
                // match). Tracks that clear the identity floor but never produce an anchor are mostly
                // blurred, tiny, occluded, or non-face crops; matching them onto an existing identity
                // is fine, but creating a new visible face from them just makes junk the user deletes.
                if (anchorEmbeddings.Length == 0 && referenceMatch is null)
                {
                    preparedTracks.Add(new PreparedFaceAssignment(track, null, [], false));
                    lowQualityCreationSkips++;
                    continue;
                }

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

        var finalReconciliation = _identityReconciler.Reconcile(snapshot, referencePack, settings, applyReferenceMatches: false);

        var briefPresenceFaceKeys = ResolveBriefPresenceFaceKeys(request, preparedTracks, snapshot, finalReconciliation, settings);
        var briefPresenceSuppressions = 0;

        foreach (var preparedTrack in preparedTracks)
        {
            var identity = ResolveFinalIdentity(preparedTrack.Identity, snapshot, finalReconciliation);
            var suppressedForBriefPresence = identity is not null && briefPresenceFaceKeys.Contains(identity.FaceKey);
            if (identity is null || suppressedForBriefPresence || !ShouldEmitPromotedIdentity(request, identity, preparedTrack.Track, settings))
            {
                EmitDetections(batch, request, preparedTrack.Track, faceKey: null, settings);
                if (identity is not null)
                {
                    if (suppressedForBriefPresence)
                    {
                        briefPresenceSuppressions++;
                    }
                    else
                    {
                        provisionalClusters++;
                    }
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

        await transaction.CommitAsync(ct);

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

        if (lowQualityCreationSkips > 0)
        {
            batch.Notes.Add($"Skipped creating {lowQualityCreationSkips} face(s) from tracks without anchor-grade evidence.");
        }

        if (provisionalClusters > 0)
        {
            batch.Notes.Add($"Kept {provisionalClusters} unknown face cluster(s) provisional instead of creating visible Cove face rows.");
        }

        if (briefPresenceSuppressions > 0)
        {
            batch.Notes.Add($"Did not mark {briefPresenceSuppressions} face(s) present on this video for falling below the {settings.MinimumVideoFacePresenceSeconds:R}s minimum screen time.");
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
            lowQualityCreationSkips,
            provisionalClusters);

        return new AiFacePreparationOutcome(
            batch,
            CombineMergedFaceKeyMaps(initialReconciliation.MergedFaceKeyMap, finalReconciliation.MergedFaceKeyMap));
    }

    // Mirrors the per-track computation at the head of the assignment loop to gather the keys the
    // incremental working set must load: every identity-grade embedding (similarity candidates) plus any
    // reference-pack external id a track matches (so an already-linked identity is in the working set and
    // is reused rather than duplicated). The duplicated representative-embedding/reference work is cheap
    // SIMD relative to the cost it avoids (loading and globally re-merging the whole graph).
    private (IReadOnlyList<IReadOnlyList<float>> Vectors, IReadOnlyCollection<string> ReferenceExternalIds) CollectCandidateKeys(
        IReadOnlyList<PreparedFaceTrack> tracks,
        SaieReferencePack? referencePack,
        AiFacesSettings settings)
    {
        var vectors = new List<IReadOnlyList<float>>();
        var referenceIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var track in tracks)
        {
            var representativeEmbeddings = SelectRepresentativeEmbeddings(track, settings);
            var identityPassEmbeddings = representativeEmbeddings.Where(static embedding => embedding.PassesIdentityFloor).ToArray();
            if (identityPassEmbeddings.Length == 0)
            {
                continue;
            }

            var anchorEmbeddings = representativeEmbeddings
                .Where(static embedding => embedding.PassesHardFloor && embedding.IsAnchor)
                .ToArray();
            var identityEmbeddings = anchorEmbeddings.Length > 0 ? anchorEmbeddings : identityPassEmbeddings;

            foreach (var embedding in identityEmbeddings)
            {
                vectors.Add(embedding.Vector);
            }

            var referenceMatch = TryMatchReference(identityEmbeddings, referencePack, settings);
            if (referenceMatch is not null && !string.IsNullOrWhiteSpace(referenceMatch.Identity.ExternalId))
            {
                referenceIds.Add(referenceMatch.Identity.ExternalId);
            }
        }

        return (vectors, referenceIds);
    }

    private static readonly IReadOnlyDictionary<string, string> EmptyMergedFaceKeyMap =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

    // Chains the two reconciliation passes' duplicate→target maps: a key merged away in the first
    // pass whose target was itself merged in the second pass resolves to the final target.
    private static IReadOnlyDictionary<string, string> CombineMergedFaceKeyMaps(
        IReadOnlyDictionary<string, string> first,
        IReadOnlyDictionary<string, string> second)
    {
        if (first.Count == 0)
        {
            return second;
        }

        var combined = new Dictionary<string, string>(second, StringComparer.OrdinalIgnoreCase);
        foreach (var (duplicate, target) in first)
        {
            combined[duplicate] = second.TryGetValue(target, out var finalTarget) ? finalTarget : target;
        }

        return combined;
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
        var orderedFrames = request.Result.Frames
            .OrderBy(static frame => frame.TimeSeconds ?? double.MinValue)
            .ThenBy(static frame => frame.Index ?? int.MinValue)
            .ToArray();
        var sliceSpan = ResolveVideoSliceSpan(request, orderedFrames);
        var openTracks = new List<OpenTrack>();
        var completed = new List<PreparedFaceTrack>();
        var nextTrackId = 1;
        var frameOrdinal = 0;

        foreach (var frame in orderedFrames)
        {
            frameOrdinal++;
            var frameOrder = frameOrdinal;
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

    private static double ResolveVideoSliceSpan(AiDispatchRequest request, IReadOnlyList<AiTemporalSlice> orderedFrames)
    {
        var explicitInterval = request.Result.FrameIntervalSeconds ?? request.Context.FrameIntervalSeconds;
        if (explicitInterval is > 0.0)
        {
            return explicitInterval.Value;
        }

        var timestamps = orderedFrames
            .Select(static frame => frame.TimeSeconds)
            .Where(static time => time.HasValue)
            .Select(static time => time!.Value)
            .Distinct()
            .Order()
            .ToArray();
        var deltas = new List<double>();
        for (var index = 1; index < timestamps.Length; index++)
        {
            var delta = timestamps[index] - timestamps[index - 1];
            if (delta > 0.0)
            {
                deltas.Add(delta);
            }
        }

        var orderedDeltas = deltas
            .Order()
            .ToArray();

        return orderedDeltas.Length == 0
            ? DefaultFrameSpanSeconds
            : Math.Max(0.25, orderedDeltas[orderedDeltas.Length / 2]);
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
                // Pose and image quality rank *which* instances win representative/cover selection (see
                // qualityScore below and AiFaceQualityScorer) — they no longer gate whether a face exists.
                // A face's existence keys off the robust ArcFace embedding norm and the detector score, so a
                // strong, clearly-recognizable face (high norm) is never discarded just because a single
                // still wasn't perfectly frontal. Pose is a noisy per-frame heuristic; norm is the reliable
                // face-quality signal, and over a growing set of detections the best-pose instance still
                // wins the thumbnail without ever vetoing the identity.
                var qualityScore = (embedding.Norm ?? 0.0) * sample.Detection.Score * poseQuality * imageQuality;
                var hardPass = sample.Detection.Score >= HardMinimumDetectionScore
                    && (embedding.Norm ?? 0.0) >= HardMinimumEmbeddingNorm;
                var identityPass = sample.Detection.Score >= AnchorDetectionScore
                    && (embedding.Norm ?? 0.0) >= AnchorEmbeddingNorm;
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
        if (best.Score < matchThreshold)
        {
            return null;
        }

        if ((best.Score - secondBest) < settings.IdentityAmbiguityMargin)
        {
            // An ambiguous best match usually means the runner-ups are duplicates of the best
            // identity (the same person already split across faces). Refusing the match would mint
            // yet another duplicate — the snowball that splits one performer across many faces — so
            // ambiguity only blocks when an in-margin rival looks like a genuinely different person.
            var rivalsAreDuplicatesOfBest = ranked
                .Skip(1)
                .TakeWhile(candidate => (best.Score - candidate.Score) < settings.IdentityAmbiguityMargin)
                .All(candidate => AreLikelyDuplicateIdentities(best.Identity, candidate.Identity, settings));
            if (!rivalsAreDuplicatesOfBest)
            {
                return null;
            }
        }

        return best.Identity;
    }

    private static bool AreLikelyDuplicateIdentities(StoredFaceIdentity left, StoredFaceIdentity right, AiFacesSettings settings)
    {
        var conflictingReferences = !string.IsNullOrWhiteSpace(left.ReferenceExternalId)
            && !string.IsNullOrWhiteSpace(right.ReferenceExternalId)
            && !string.Equals(left.ReferenceExternalId, right.ReferenceExternalId, StringComparison.OrdinalIgnoreCase);
        return !conflictingReferences
            && AiFaceIdentityReconciler.ScoreIdentityPair(left, right) >= settings.ConsolidationSimilarityThreshold;
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

        var match = SaieReferenceMatcher.FindBest(referencePack, anchors.Select(static anchor => anchor.Vector).ToArray());
        if (match is not { } best
            || best.Score < settings.ReferenceMatchThreshold
            || (best.Score - best.SecondScore) < settings.ReferenceAmbiguityMargin)
        {
            return null;
        }

        var identity = referencePack.Identities[best.Ordinal];
        return new FaceReferenceMatch(identity, referencePack.Manifest.PackId, AiFaceReferenceSuggestionIds.FromOrdinal(identity.Ordinal), best.Score);
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

    // Faces whose total screen time in this video falls below the configured presence floor. Their
    // tracks still emit detections, but the face is not marked present (no appearance/segment), which
    // suppresses mis-attributed single detections and incidental intro/outro cameos. Aggregated per
    // resolved identity so a face split across several short tracks is judged on its combined time, and
    // the floor is capped at half the video's duration so a legitimately short clip keeps its main face.
    private static HashSet<string> ResolveBriefPresenceFaceKeys(
        AiDispatchRequest request,
        IReadOnlyList<PreparedFaceAssignment> preparedTracks,
        FaceIdentitySnapshot snapshot,
        AiFaceIdentityReconciliationReport reconciliation,
        AiFacesSettings settings)
    {
        var suppressed = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (request.Result.MediaKind != AiMediaKinds.Video || settings.MinimumVideoFacePresenceSeconds <= 0.0)
        {
            return suppressed;
        }

        // Cap the floor at half the video's length so a legitimately short clip still surfaces its main
        // face. Prefer the real duration; when it is absent fall back to the analyzed sample span so the
        // cap still scales to the content (and stays a no-op for single-moment clips).
        var durationSeconds = ResolveDurationSeconds(request) ?? ResolveAnalyzedSpanSeconds(preparedTracks);
        var threshold = settings.MinimumVideoFacePresenceSeconds;
        if (durationSeconds is > 0.0)
        {
            threshold = Math.Min(threshold, durationSeconds.Value * 0.5);
        }

        if (threshold <= 0.0)
        {
            return suppressed;
        }

        var secondsByFaceKey = new Dictionary<string, double>(StringComparer.OrdinalIgnoreCase);
        foreach (var preparedTrack in preparedTracks)
        {
            var identity = ResolveFinalIdentity(preparedTrack.Identity, snapshot, reconciliation);
            if (identity is null)
            {
                continue;
            }

            var frameIntervalSeconds = ResolveFrameIntervalSeconds(request, preparedTrack.Track);
            var trackSeconds = EstimateRepresentedVideoEvidenceSeconds(request, preparedTrack.Track, frameIntervalSeconds);
            secondsByFaceKey[identity.FaceKey] = secondsByFaceKey.GetValueOrDefault(identity.FaceKey) + trackSeconds;
        }

        foreach (var (faceKey, seconds) in secondsByFaceKey)
        {
            if (seconds < threshold)
            {
                suppressed.Add(faceKey);
            }
        }

        return suppressed;
    }

    private static double? ResolveAnalyzedSpanSeconds(IReadOnlyList<PreparedFaceAssignment> preparedTracks)
    {
        var starts = preparedTracks
            .Select(static preparedTrack => preparedTrack.Track.StartSeconds)
            .Where(static value => value.HasValue)
            .Select(static value => value!.Value)
            .ToArray();
        var ends = preparedTracks
            .Select(static preparedTrack => preparedTrack.Track.EndSeconds)
            .Where(static value => value.HasValue)
            .Select(static value => value!.Value)
            .ToArray();
        if (starts.Length == 0 || ends.Length == 0)
        {
            return null;
        }

        var span = ends.Max() - starts.Min();
        return span > 0.0 ? span : null;
    }

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
                    ["coverQualityScore"] = AiFaceQualityScorer.ScoreCoverQuality(window?.Samples ?? track.Samples, sample).ToString("R", CultureInfo.InvariantCulture),
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
        var options = new BoundingBoxKeyframeSelectionOptions
        {
            IoUThreshold = settings.DetectionKeyframeIoUThreshold,
            MaxKeyframes = settings.MaxDetectionKeyframesPerTrack,
            MaxGapSeconds = settings.DetectionKeyframeMaxGapSeconds,
        };

        return BoundingBoxKeyframeSelector.Select(samples, bestSample, ToBoundingBoxKeyframe, options);
    }

    private static BoundingBoxKeyframe ToBoundingBoxKeyframe(FaceFrameSample sample)
        => new(
            sample.Detection.BoundingBox.X1,
            sample.Detection.BoundingBox.Y1,
            sample.Detection.BoundingBox.X2,
            sample.Detection.BoundingBox.Y2,
            sample.TimeSeconds,
            sample.FrameOrder,
            GetSampleIdentity(sample));

    private static string GetSampleIdentity(FaceFrameSample sample)
        => $"{sample.FrameOrder}\u001F{sample.TimeSeconds?.ToString("R", CultureInfo.InvariantCulture)}\u001F{FormatBoundingBox(sample.Detection.BoundingBox)}";

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
                    ["detectionKeyframeIoUThreshold"] = settings.DetectionKeyframeIoUThreshold.ToString("R", CultureInfo.InvariantCulture),
                    ["detectionKeyframeMaxGapSec"] = settings.DetectionKeyframeMaxGapSeconds.ToString("R", CultureInfo.InvariantCulture),
                    ["maxDetectionKeyframes"] = settings.MaxDetectionKeyframesPerTrack.ToString(CultureInfo.InvariantCulture),
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
        int lowQualityCreationSkips,
        int provisionalClusters)
    {
        batch.Notes.Add(string.Create(
            CultureInfo.InvariantCulture,
            $"AI.Faces telemetry: rawTracks={clusterDiagnostics.InputTrackCount}; assetClusters={clusterDiagnostics.ClusterCount}; clusterMerges={clusterDiagnostics.MergedTrackCount}; clusterRejectedConcurrency={clusterDiagnostics.RejectedByConcurrencyCount}; clusterRejectedThreshold={clusterDiagnostics.RejectedByThresholdCount}; clusterRejectedAmbiguous={clusterDiagnostics.RejectedByAmbiguityCount}; createdIdentities={newIdentityCount}; promotedThisRun={promotedIdentityCount}; provisionalClusters={provisionalClusters}; seededMatches={seededMatchCount}; conflictingReferences={conflictingReferenceCount}; unresolvedTracks={unresolvedTracks}; lowQualityCreationSkips={lowQualityCreationSkips}; faces={batch.Faces.Count}; detections={batch.Detections.Count}; reconciliationMerges={initialReconciliation.MergedIdentityCount + finalReconciliation.MergedIdentityCount}; reconciliationReferencePromotions={initialReconciliation.ReferencePromotedIdentityCount + finalReconciliation.ReferencePromotedIdentityCount}; reconciliationEvidencePromotions={initialReconciliation.EvidencePromotedIdentityCount + finalReconciliation.EvidencePromotedIdentityCount}"));
        batch.Notes.Add(string.Create(
            CultureInfo.InvariantCulture,
                $"AI.Faces thresholds: identity={settings.IdentityMatchThreshold:R}/{settings.IdentityAmbiguityMargin:R}; assetCluster={settings.AssetClusterSimilarityThreshold:R}/{settings.AssetClusterAmbiguityMargin:R}; reference={settings.ReferenceMatchThreshold:R}/{settings.ReferenceAmbiguityMargin:R}; consolidation={settings.ConsolidationSimilarityThreshold:R}/{settings.ConsolidationAmbiguityMargin:R}; sameAssetConsolidation={settings.ConsolidationSameAssetSimilarityThreshold:R}; videoPromotionSamples={settings.PromotionMinimumVideoSamples}; videoPromotionEvidenceSeconds={settings.PromotionMinimumVideoEvidenceSeconds:R}; sparseVideoPromotionSamples={settings.PromotionMinimumSparseVideoSamples}; sparseVideoPromotionFrameInterval={settings.SparseVideoPromotionFrameIntervalSeconds:R}; sparseVideoPromotionCoverageRatio={settings.PromotionMinimumSparseVideoSampleCoverageRatio:R}; detectionKeyframeIoU={settings.DetectionKeyframeIoUThreshold:R}; detectionKeyframeMaxGap={settings.DetectionKeyframeMaxGapSeconds:R}; maxDetectionKeyframes={settings.MaxDetectionKeyframesPerTrack}"));
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

internal sealed record AiFacePreparationOutcome(
    AiPreparedArtifactBatch Batch,
    IReadOnlyDictionary<string, string> MergedFaceKeyMap);