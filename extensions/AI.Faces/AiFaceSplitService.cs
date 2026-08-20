using System.Text.Json;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

using Pgvector;

namespace AI.Faces;

/// <summary>
/// One detected face-track of a face on one host — the unit a split moves.
/// <see cref="RepresentativeDetectionId"/> is the track's best detection, which callers can render
/// through the host's existing <c>/api/stream/detection/{id}/crop</c> endpoint to show the actual face.
/// </summary>
internal sealed record AiFaceTrackSummary(
    string GroupKey,
    double? FirstSeenSeconds,
    double? LastSeenSeconds,
    int SampleCount,
    int DetectionCount,
    double? RepresentativeFrameSeconds,
    double? TopConfidence,
    int? RepresentativeDetectionId,
    IReadOnlyList<float>? RepresentativeBoundingBox,
    double? RepresentativeDetectionSeconds,
    // Which person this track is believed to be, among the people sharing this face on this host.
    // 0 is the most-represented one. Grouping the appearances is the whole job the caller would
    // otherwise be doing by eye, and the embeddings needed to do it properly live here.
    int SuggestedGroup);

internal sealed record AiFaceSplitResult(
    bool FaceFound,
    bool HostHadFace,
    bool GroupKeysMatched,
    bool WouldEmptyFace,
    int MovedAppearanceCount,
    int MovedDetectionCount,
    int MovedSegmentCount,
    int MovedEmbeddingCount,
    int TargetFaceId,
    string? TargetFaceKey,
    bool CreatedNewFace,
    bool MergedIntoExistingFace,
    bool RecordedIdentitySplit)
{
    public static readonly AiFaceSplitResult NotFound = new(false, false, false, false, 0, 0, 0, 0, 0, null, false, false, false);
    public static readonly AiFaceSplitResult NoFaceOnHost = new(true, false, false, false, 0, 0, 0, 0, 0, null, false, false, false);
    public static readonly AiFaceSplitResult NoMatchingGroups = new(true, true, false, false, 0, 0, 0, 0, 0, null, false, false, false);
    public static readonly AiFaceSplitResult WouldEmpty = new(true, true, true, true, 0, 0, 0, 0, 0, null, false, false, false);
}

/// <summary>
/// Pulls part of a face apart <em>within</em> a single video or image.
///
/// The existing not-present action can only reject a face from a whole host, because it re-homes rows by
/// run id and one run is one host — so when two performers ended up on one face inside the same video
/// there was no way to untangle them. This works one level finer: the caller names the tracks (appearance
/// group keys, one per detected face-track) that belong to the other person, and every row backing those
/// tracks — appearance, detections, timeline segments, embeddings — moves to a face that matches them, or
/// to a new one.
///
/// Two things make the correction stick rather than being undone by the next analysis run:
/// the extension identity graph is updated (wrong-person anchors leave the source identity and seed the
/// target's, so the source stops attracting that person in <em>other</em> videos), and the pair is written
/// to <see cref="AiFaceIdentityExclusionStore"/>, which vetoes any future match or merge between them.
/// </summary>
internal sealed class AiFaceSplitService(
    IServiceScopeFactory scopeFactory,
    AiFaceIdentityExclusionStore exclusionStore)
{
    private const string FaceSourceKey = "ext:ai.faces";
    private const string TrackKeyMetadataKey = "trackKey";
    private const int TargetCandidateK = 20;

    public async Task<AiFaceSplitResult> SplitAsync(
        int faceId,
        string hostType,
        int hostId,
        IReadOnlyList<string> groupKeys,
        CancellationToken ct = default)
    {
        var normalizedHostType = NormalizeHostType(hostType);
        if (normalizedHostType is not ("video" or "image") || faceId <= 0 || hostId <= 0 || groupKeys.Count == 0)
        {
            return AiFaceSplitResult.NotFound;
        }

        var selectedGroupKeys = groupKeys
            .Where(static key => !string.IsNullOrWhiteSpace(key))
            .Select(static key => key.Trim())
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        if (selectedGroupKeys.Count == 0)
        {
            return AiFaceSplitResult.NotFound;
        }

        await using var scope = scopeFactory.CreateAsyncScope();
        var faceRepo = scope.ServiceProvider.GetRequiredService<IFaceRepository>();
        var embeddingRepo = scope.ServiceProvider.GetRequiredService<IEmbeddingRepository>();
        var detectionRepo = scope.ServiceProvider.GetRequiredService<IDetectionRepository>();
        var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();
        var embeddingService = scope.ServiceProvider.GetRequiredService<IEmbeddingService>();
        var identityStore = scope.ServiceProvider.GetRequiredService<IFaceIdentityStore>();
        var settings = await AiFacesSettingsRuntime.LoadAsync(ct);

        var face = await faceRepo.GetFaceAsync(faceId, tracking: true, ct);
        if (face is null)
        {
            return AiFaceSplitResult.NotFound;
        }

        var appearanceHostType = normalizedHostType == "video" ? FaceAppearanceHostType.Video : FaceAppearanceHostType.Image;
        var allAppearances = await faceRepo.FindAppearancesAsync(
            new FaceAppearanceFilter { FaceIds = [faceId], SourceKey = FaceSourceKey }, ct);
        var hostAppearances = allAppearances
            .Where(appearance => appearance.HostType == appearanceHostType && appearance.HostId == hostId)
            .ToArray();
        if (hostAppearances.Length == 0)
        {
            return AiFaceSplitResult.NoFaceOnHost;
        }

        var movingAppearances = hostAppearances
            .Where(appearance => !string.IsNullOrWhiteSpace(appearance.GroupKey)
                && selectedGroupKeys.Contains(appearance.GroupKey!))
            .ToArray();
        if (movingAppearances.Length == 0)
        {
            return AiFaceSplitResult.NoMatchingGroups;
        }

        // Splitting away everything the face has is a deletion, not a split — the not-present action is
        // the right tool for that, and it records the negative signal this one does not.
        if (movingAppearances.Length >= allAppearances.Count)
        {
            return AiFaceSplitResult.WouldEmpty;
        }

        var movingTrackKeys = movingAppearances
            .Select(static appearance => appearance.GroupKey!)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var hostRunIds = hostAppearances
            .Select(static appearance => appearance.SourceRunId)
            .Where(static runId => !string.IsNullOrWhiteSpace(runId))
            .Select(static runId => runId!)
            .ToHashSet(StringComparer.Ordinal);

        var detectionHostType = normalizedHostType == "video" ? DetectionHostType.Video : DetectionHostType.Image;
        var hostDetections = await detectionRepo.FindAsync(new DetectionFilter
        {
            HostType = detectionHostType,
            HostId = hostId,
            SourceKey = FaceSourceKey,
            RefKind = "face",
            RefIds = [faceId],
        }, ct);
        var movingDetections = hostDetections.Where(detection => BelongsToTrack(detection.GroupKey, movingTrackKeys)).ToArray();

        var hostSegments = normalizedHostType == "video"
            ? await segmentRepo.FindAsync(new SegmentFilter
            {
                HostType = SegmentHostType.Video,
                HostId = hostId,
                SourceKey = FaceSourceKey,
                RefIds = [faceId],
            }, ct)
            : (IReadOnlyList<Segment>)[];
        var movingSegments = hostSegments
            .Where(segment => BelongsToTrack(ReadMetadata(segment.Payload, TrackKeyMetadataKey), movingTrackKeys))
            .ToArray();

        // Embeddings hang off the face rather than the host, so narrow to this host's runs first.
        var faceEmbeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
        {
            HostType = EmbeddingHostType.Face,
            HostId = faceId,
            SourceKey = FaceSourceKey,
            Modality = EmbeddingModality.Face,
        }, ct);
        var movingEmbeddings = faceEmbeddings
            .Where(embedding => embedding.SourceRunId is not null
                && hostRunIds.Contains(embedding.SourceRunId)
                && BelongsToTrack(ReadMetadata(embedding.Meta, TrackKeyMetadataKey), movingTrackKeys))
            .ToArray();
        var retainedEmbeddings = faceEmbeddings.Except(movingEmbeddings).ToArray();

        var movingCentroid = Centroid(movingEmbeddings.Select(static embedding => embedding.Vector.ToArray()).ToArray());
        var retainedCentroid = Centroid(retainedEmbeddings.Select(static embedding => embedding.Vector.ToArray()).ToArray());

        var sourceFaceKey = face.PrimarySourceKey;
        var existingTarget = await FindMatchingTargetFaceAsync(
            faceRepo, embeddingService, faceId, movingCentroid, settings, ct);

        // The identity graph moves first: it both mints the face key a brand-new target will carry (under
        // the store's own gate, so the ordinal cannot race another run) and rehomes the wrong person's
        // anchors. The Cove rows follow.
        var identitySplit = await SplitIdentityGraphAsync(
            identityStore, sourceFaceKey, existingTarget?.PrimarySourceKey, movingCentroid, retainedCentroid, ct);

        Face target;
        bool createdNew;
        bool mergedIntoExisting;
        string? targetFaceKey;
        if (existingTarget is not null)
        {
            target = existingTarget;
            createdNew = false;
            mergedIntoExisting = true;
            targetFaceKey = existingTarget.PrimarySourceKey;
        }
        else
        {
            targetFaceKey = identitySplit.TargetFaceKey ?? $"face-split-{Guid.NewGuid():N}";
            target = new Face { PrimarySourceKey = targetFaceKey, Label = targetFaceKey };
            faceRepo.AddFace(target);
            await faceRepo.SaveChangesAsync(ct);
            createdNew = true;
            mergedIntoExisting = false;
        }

        // No repository API re-points an individual row, and every Find* here is untracked, so each moved
        // row is deleted and re-inserted against the target. Nothing references these rows by id — faces
        // are referenced the other way round — so the identity of the rows themselves is not meaningful.
        foreach (var appearance in movingAppearances)
        {
            faceRepo.AddAppearance(CloneAppearance(appearance, target.Id));
        }

        faceRepo.RemoveAppearances(movingAppearances);

        foreach (var detection in movingDetections)
        {
            detectionRepo.Add(CloneDetection(detection, target.Id));
        }

        detectionRepo.RemoveRange(movingDetections);

        foreach (var segment in movingSegments)
        {
            segmentRepo.Add(CloneSegment(segment, target.Id));
        }

        segmentRepo.RemoveRange(movingSegments);

        foreach (var embedding in movingEmbeddings)
        {
            embeddingRepo.Add(CloneEmbedding(embedding, target.Id));
        }

        embeddingRepo.RemoveRange(movingEmbeddings);

        await faceRepo.SaveChangesAsync(ct);

        await AiFacesPersistenceService.RefreshFaceStatsAsync(faceRepo, detectionRepo, [faceId, target.Id], ct);
        await faceRepo.SaveChangesAsync(ct);

        // The host now carries both faces; recompute its performer assignments so the split-off tracks
        // stop contributing the source face's performer and start contributing the target's.
        var propagation = scope.ServiceProvider.GetService<IFacePerformerPropagationService>();
        if (propagation is not null)
        {
            try
            {
                await propagation.ReconcileHostAsync(appearanceHostType, hostId, ct);
                await faceRepo.SaveChangesAsync(ct);
            }
            catch
            {
                // Best-effort: a propagation failure must not undo the split that already committed.
            }
        }

        var suggestionMaintenance = scope.ServiceProvider.GetService<IFaceTopSuggestionMaintenance>();
        if (suggestionMaintenance is not null)
        {
            await suggestionMaintenance.InvalidateAsync([faceId, target.Id], ct);
        }

        if (AiFaceIdentityExclusionPair.TryCreate(sourceFaceKey, targetFaceKey) is { } exclusion)
        {
            await exclusionStore.AddAsync([exclusion], ct);
        }

        return new AiFaceSplitResult(
            FaceFound: true,
            HostHadFace: true,
            GroupKeysMatched: true,
            WouldEmptyFace: false,
            MovedAppearanceCount: movingAppearances.Length,
            MovedDetectionCount: movingDetections.Length,
            MovedSegmentCount: movingSegments.Length,
            MovedEmbeddingCount: movingEmbeddings.Length,
            TargetFaceId: target.Id,
            TargetFaceKey: targetFaceKey,
            CreatedNewFace: createdNew,
            MergedIntoExistingFace: mergedIntoExisting,
            RecordedIdentitySplit: identitySplit.Recorded);
    }

    /// <summary>
    /// The face's individual tracks on one host, so a caller can show them side by side and pick the ones
    /// that are the other person. Each entry's <c>GroupKey</c> is what <see cref="SplitAsync"/> takes.
    /// </summary>
    public async Task<IReadOnlyList<AiFaceTrackSummary>> GetHostTracksAsync(
        int faceId,
        string hostType,
        int hostId,
        CancellationToken ct = default)
    {
        var normalizedHostType = NormalizeHostType(hostType);
        if (normalizedHostType is not ("video" or "image") || faceId <= 0 || hostId <= 0)
        {
            return [];
        }

        await using var scope = scopeFactory.CreateAsyncScope();
        var faceRepo = scope.ServiceProvider.GetRequiredService<IFaceRepository>();
        var detectionRepo = scope.ServiceProvider.GetRequiredService<IDetectionRepository>();

        var appearanceHostType = normalizedHostType == "video" ? FaceAppearanceHostType.Video : FaceAppearanceHostType.Image;
        var appearances = await faceRepo.FindAppearancesAsync(new FaceAppearanceFilter
        {
            FaceIds = [faceId],
            HostType = appearanceHostType,
            HostId = hostId,
            SourceKey = FaceSourceKey,
        }, ct);
        if (appearances.Count == 0)
        {
            return [];
        }

        var embeddingRepo = scope.ServiceProvider.GetRequiredService<IEmbeddingRepository>();
        var settings = await AiFacesSettingsRuntime.LoadAsync(ct);

        var detections = await detectionRepo.FindAsync(new DetectionFilter
        {
            HostType = normalizedHostType == "video" ? DetectionHostType.Video : DetectionHostType.Image,
            HostId = hostId,
            SourceKey = FaceSourceKey,
            RefKind = "face",
            RefIds = [faceId],
        }, ct);

        var hostRunIds = appearances
            .Select(static appearance => appearance.SourceRunId)
            .Where(static runId => !string.IsNullOrWhiteSpace(runId))
            .Select(static runId => runId!)
            .ToHashSet(StringComparer.Ordinal);
        var faceEmbeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
        {
            HostType = EmbeddingHostType.Face,
            HostId = faceId,
            SourceKey = FaceSourceKey,
            Modality = EmbeddingModality.Face,
        }, ct);
        var centroidByTrackKey = faceEmbeddings
            .Where(embedding => embedding.SourceRunId is not null && hostRunIds.Contains(embedding.SourceRunId))
            .Select(embedding => (TrackKey: ReadMetadata(embedding.Meta, TrackKeyMetadataKey), embedding.Vector))
            .Where(static item => !string.IsNullOrWhiteSpace(item.TrackKey))
            .GroupBy(static item => item.TrackKey!, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                static group => group.Key,
                static group => Centroid(group.Select(static item => item.Vector.ToArray()).ToArray()),
                StringComparer.OrdinalIgnoreCase);

        var ordered = appearances
            .Where(static appearance => !string.IsNullOrWhiteSpace(appearance.GroupKey))
            .OrderBy(static appearance => appearance.FirstSeenAtSec ?? double.MinValue)
            .ToArray();
        var candidates = ordered
            .Select(appearance => new TrackGroupingCandidate(
                appearance.GroupKey!,
                centroidByTrackKey.GetValueOrDefault(appearance.GroupKey!) ?? [],
                appearance.FirstSeenAtSec,
                appearance.LastSeenAtSec,
                appearance.SampleCount))
            .ToArray();
        var groupByTrackKey = AssignSuggestedGroups(candidates, settings.ConsolidationSameAssetSimilarityThreshold);

        return ordered
            .Select(appearance =>
            {
                var trackKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { appearance.GroupKey! };
                var trackDetections = detections
                    .Where(detection => BelongsToTrack(detection.GroupKey, trackKeys))
                    .ToArray();
                var best = trackDetections.OrderByDescending(static detection => detection.Score).FirstOrDefault();
                return new AiFaceTrackSummary(
                    appearance.GroupKey!,
                    appearance.FirstSeenAtSec,
                    appearance.LastSeenAtSec,
                    appearance.SampleCount,
                    trackDetections.Length,
                    appearance.RepresentativeFrameSec,
                    appearance.TopConfidence,
                    best?.Id,
                    best is null ? null : [best.X, best.Y, best.X + best.W, best.Y + best.H],
                    best?.ObservedAtSec,
                    groupByTrackKey.GetValueOrDefault(appearance.GroupKey!, 0));
            })
            .ToArray();
    }

    private readonly record struct TrackGroupingCandidate(
        string TrackKey,
        float[] Centroid,
        double? StartSeconds,
        double? EndSeconds,
        int SampleCount);

    /// <summary>
    /// Groups a face's appearances on one host into the people they belong to, so the caller can offer
    /// "this person / that person" instead of a pile of undifferentiated clips.
    ///
    /// Agglomerative on embedding similarity, with one hard constraint: appearances whose time ranges
    /// overlap can never join the same group, because a person cannot be in two places at once. That
    /// constraint is what makes this work at all — the two performers are on one face precisely because
    /// their embeddings are close, so similarity alone would happily merge them straight back together.
    /// Average-linkage (not single) keeps one stray frame from chaining two people into one group.
    ///
    /// Groups are numbered by how much of the face they account for, so group 0 is the dominant person
    /// and anything above it is the caller's likely split.
    /// </summary>
    private static Dictionary<string, int> AssignSuggestedGroups(
        IReadOnlyList<TrackGroupingCandidate> candidates,
        double similarityThreshold)
    {
        var groups = candidates.Select(static (_, index) => new List<int> { index }).ToList();

        while (groups.Count > 1)
        {
            var bestScore = double.NegativeInfinity;
            var bestLeft = -1;
            var bestRight = -1;

            for (var left = 0; left < groups.Count; left++)
            {
                for (var right = left + 1; right < groups.Count; right++)
                {
                    if (AnyPairOverlapsInTime(groups[left], groups[right], candidates))
                    {
                        continue;
                    }

                    var score = AverageSimilarity(groups[left], groups[right], candidates);
                    if (score >= similarityThreshold && score > bestScore)
                    {
                        bestScore = score;
                        bestLeft = left;
                        bestRight = right;
                    }
                }
            }

            if (bestLeft < 0)
            {
                break;
            }

            groups[bestLeft].AddRange(groups[bestRight]);
            groups.RemoveAt(bestRight);
        }

        var assignment = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        var ranked = groups
            .OrderByDescending(group => group.Sum(index => candidates[index].SampleCount))
            .ThenBy(group => group.Min(index => candidates[index].StartSeconds ?? double.MaxValue))
            .ToArray();
        for (var groupIndex = 0; groupIndex < ranked.Length; groupIndex++)
        {
            foreach (var candidateIndex in ranked[groupIndex])
            {
                assignment[candidates[candidateIndex].TrackKey] = groupIndex;
            }
        }

        return assignment;
    }

    private static bool AnyPairOverlapsInTime(
        List<int> left,
        List<int> right,
        IReadOnlyList<TrackGroupingCandidate> candidates)
        => left.Any(leftIndex => right.Any(rightIndex => OverlapsInTime(candidates[leftIndex], candidates[rightIndex])));

    private static bool OverlapsInTime(TrackGroupingCandidate left, TrackGroupingCandidate right)
    {
        // A single-moment host (an image) has no timeline: every appearance in it is simultaneous, so
        // every pair conflicts and each face stays its own person — which is exactly right for an image.
        if (left.StartSeconds is null || right.StartSeconds is null)
        {
            return true;
        }

        var leftEnd = left.EndSeconds ?? left.StartSeconds.Value;
        var rightEnd = right.EndSeconds ?? right.StartSeconds.Value;
        return left.StartSeconds.Value <= rightEnd && right.StartSeconds.Value <= leftEnd;
    }

    private static double AverageSimilarity(
        List<int> left,
        List<int> right,
        IReadOnlyList<TrackGroupingCandidate> candidates)
    {
        var scores = new List<double>();
        foreach (var leftIndex in left)
        {
            foreach (var rightIndex in right)
            {
                var leftCentroid = candidates[leftIndex].Centroid;
                var rightCentroid = candidates[rightIndex].Centroid;
                if (leftCentroid.Length == 0 || rightCentroid.Length == 0)
                {
                    // No embedding for one side: not evidence of sameness, so it cannot pull a merge.
                    return double.NegativeInfinity;
                }

                scores.Add(Cosine(leftCentroid, rightCentroid));
            }
        }

        return scores.Count == 0 ? double.NegativeInfinity : scores.Average();
    }

    /// <summary>
    /// An existing face that clearly matches the split-off tracks, if there is one — the split-off person
    /// usually already has a face of their own from another video. Null means a new face is needed.
    /// </summary>
    private static async Task<Face?> FindMatchingTargetFaceAsync(
        IFaceRepository faceRepo,
        IEmbeddingService embeddingService,
        int sourceFaceId,
        float[] movingCentroid,
        AiFacesSettings settings,
        CancellationToken ct)
    {
        if (movingCentroid.Length > 0)
        {
            var matches = await embeddingService.KnnAsync(
                new Vector(movingCentroid),
                k: TargetCandidateK,
                new EmbeddingSearchOptions
                {
                    HostType = EmbeddingHostType.Face,
                    SourceKey = FaceSourceKey,
                    Modality = EmbeddingModality.Face,
                },
                ct);

            var bestSimilarityByFaceId = new Dictionary<int, double>();
            foreach (var match in matches)
            {
                if (match.Embedding.HostId == sourceFaceId)
                {
                    continue;
                }

                var similarity = 1.0 - match.Distance;
                if (!bestSimilarityByFaceId.TryGetValue(match.Embedding.HostId, out var existing) || similarity > existing)
                {
                    bestSimilarityByFaceId[match.Embedding.HostId] = similarity;
                }
            }

            var rankedFaceIds = bestSimilarityByFaceId
                .Where(pair => pair.Value >= settings.ConsolidationSimilarityThreshold)
                .OrderByDescending(pair => pair.Value)
                .Select(pair => pair.Key)
                .ToArray();
            if (rankedFaceIds.Length > 0)
            {
                var candidates = await faceRepo.FindFacesAsync(
                    new FaceFilter { Ids = rankedFaceIds, IsMerged = false, Ignored = false },
                    tracking: true,
                    ct);
                var candidateById = candidates.ToDictionary(static candidate => candidate.Id);
                foreach (var candidateId in rankedFaceIds)
                {
                    if (candidateById.TryGetValue(candidateId, out var candidate))
                    {
                        return candidate;
                    }
                }
            }
        }

        return null;
    }

    /// <summary>
    /// Moves the anchors belonging to the split-off person off the source identity and onto the target's,
    /// minting the target identity (and allocating its face key) when the split needs a new face. Without
    /// this the source identity keeps the wrong person's anchors and goes on attracting them everywhere
    /// else, and the next analysis run simply rebuilds the tangle.
    ///
    /// Runs inside the identity store's own transaction, which serialises against concurrent analysis runs,
    /// so the allocated ordinal cannot collide.
    /// </summary>
    private static async Task<(string? TargetFaceKey, bool Recorded)> SplitIdentityGraphAsync(
        IFaceIdentityStore identityStore,
        string? sourceFaceKey,
        string? existingTargetFaceKey,
        float[] movingCentroid,
        float[] retainedCentroid,
        CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(sourceFaceKey) || movingCentroid.Length == 0)
        {
            return (existingTargetFaceKey, false);
        }

        await using var transaction = await identityStore.BeginIncrementalAsync([movingCentroid], [], TargetCandidateK, ct);
        var snapshot = transaction.Snapshot;
        var source = snapshot.Identities.FirstOrDefault(identity =>
            string.Equals(identity.FaceKey, sourceFaceKey, StringComparison.OrdinalIgnoreCase));
        if (source is null)
        {
            return (existingTargetFaceKey, false);
        }

        var targetFaceKey = string.IsNullOrWhiteSpace(existingTargetFaceKey)
            ? $"face-{snapshot.NextIdentityOrdinal:0000}"
            : existingTargetFaceKey;
        var target = snapshot.Identities.FirstOrDefault(identity =>
            string.Equals(identity.FaceKey, targetFaceKey, StringComparison.OrdinalIgnoreCase));
        var movedAnchors = source.Anchors
            .Where(anchor => Cosine(anchor.Vector.ToArray(), movingCentroid)
                > (retainedCentroid.Length == 0 ? 0.0 : Cosine(anchor.Vector.ToArray(), retainedCentroid)))
            .ToArray();

        // Never strip the source bare: an identity with no anchors can never be matched again.
        if (movedAnchors.Length >= source.Anchors.Count)
        {
            movedAnchors = movedAnchors
                .OrderByDescending(anchor => Cosine(anchor.Vector.ToArray(), movingCentroid))
                .Take(Math.Max(0, source.Anchors.Count - 1))
                .ToArray();
        }

        if (target is null)
        {
            target = new StoredFaceIdentity
            {
                FaceKey = targetFaceKey,
                LifecycleStatus = StoredFaceIdentityLifecycle.Promoted,
                PromotionReason = "user-split",
                QualityScore = source.QualityScore,
                AssetIds = [.. source.AssetIds],
            };
            snapshot.Identities.Add(target);
            snapshot.NextIdentityOrdinal++;
        }

        foreach (var anchor in movedAnchors)
        {
            source.Anchors.Remove(anchor);
            target.Anchors.Add(new StoredFaceAnchor
            {
                ModelKey = anchor.ModelKey,
                QualityScore = anchor.QualityScore,
                Vector = [.. anchor.Vector],
            });
        }

        // A brand-new identity whose source anchors all had to stay behind would be unmatchable; seed it
        // from the split centroid itself so the next run can recognise it.
        if (target.Anchors.Count == 0)
        {
            target.Anchors.Add(new StoredFaceAnchor
            {
                ModelKey = source.Anchors.FirstOrDefault()?.ModelKey ?? string.Empty,
                QualityScore = source.QualityScore,
                Vector = [.. movingCentroid],
            });
        }

        await transaction.CommitAsync(ct);
        return (targetFaceKey, true);
    }

    private static bool BelongsToTrack(string? groupKey, IReadOnlySet<string> trackKeys)
    {
        if (string.IsNullOrWhiteSpace(groupKey))
        {
            return false;
        }

        if (trackKeys.Contains(groupKey))
        {
            return true;
        }

        // Segments and detections are grouped per continuous span of a track ("<trackKey>:span-2").
        var separator = groupKey.LastIndexOf(":span-", StringComparison.Ordinal);
        return separator > 0 && trackKeys.Contains(groupKey[..separator]);
    }

    private static string? ReadMetadata(JsonDocument? document, string key)
        => document is not null
           && document.RootElement.ValueKind == JsonValueKind.Object
           && document.RootElement.TryGetProperty(key, out var value)
           && value.ValueKind == JsonValueKind.String
            ? value.GetString()
            : null;

    private static JsonDocument? CloneJson(JsonDocument? document)
        => document is null ? null : JsonDocument.Parse(document.RootElement.GetRawText());

    private static FaceAppearance CloneAppearance(FaceAppearance source, int targetFaceId)
        => new()
        {
            FaceId = targetFaceId,
            HostType = source.HostType,
            HostId = source.HostId,
            FirstSeenAtSec = source.FirstSeenAtSec,
            LastSeenAtSec = source.LastSeenAtSec,
            SampleCount = source.SampleCount,
            RetainedSpatialSampleCount = source.RetainedSpatialSampleCount,
            SegmentCount = source.SegmentCount,
            RepresentativeFrameSec = source.RepresentativeFrameSec,
            TopConfidence = source.TopConfidence,
            GroupKey = source.GroupKey,
            Payload = CloneJson(source.Payload),
            SourceKey = source.SourceKey,
            SourceRunId = source.SourceRunId,
        };

    private static Detection CloneDetection(Detection source, int targetFaceId)
        => new()
        {
            HostType = source.HostType,
            HostId = source.HostId,
            ObservedAtSec = source.ObservedAtSec,
            FrameWidth = source.FrameWidth,
            FrameHeight = source.FrameHeight,
            Class = source.Class,
            Score = source.Score,
            X = source.X,
            Y = source.Y,
            W = source.W,
            H = source.H,
            Extra = CloneJson(source.Extra),
            RefKind = source.RefKind,
            RefId = targetFaceId,
            GroupKey = source.GroupKey,
            SourceKey = source.SourceKey,
            SourceRunId = source.SourceRunId,
        };

    private static Segment CloneSegment(Segment source, int targetFaceId)
        => new()
        {
            HostType = source.HostType,
            HostId = source.HostId,
            StartSec = source.StartSec,
            EndSec = source.EndSec,
            TagId = source.TagId,
            Kind = source.Kind,
            RefId = targetFaceId,
            Payload = CloneJson(source.Payload),
            SourceKey = source.SourceKey,
            SourceRunId = source.SourceRunId,
            Confidence = source.Confidence,
            Title = source.Title,
            ColorHint = source.ColorHint,
            ImageBlobId = source.ImageBlobId,
        };

    private static Embedding CloneEmbedding(Embedding source, int targetFaceId)
        => new()
        {
            HostType = source.HostType,
            HostId = targetFaceId,
            Kind = source.Kind,
            KindFamily = source.KindFamily,
            Modality = source.Modality,
            IsSemantic = source.IsSemantic,
            Dim = source.Dim,
            Vector = new Vector(source.Vector.ToArray()),
            SectionIndex = source.SectionIndex,
            StartSec = source.StartSec,
            EndSec = source.EndSec,
            SourceKey = source.SourceKey,
            SourceRunId = source.SourceRunId,
            Meta = CloneJson(source.Meta),
        };

    private static float[] Centroid(IReadOnlyList<float[]> vectors)
    {
        if (vectors.Count == 0)
        {
            return [];
        }

        var dimension = vectors[0].Length;
        var buffer = new double[dimension];
        var count = 0;
        foreach (var vector in vectors)
        {
            if (vector.Length != dimension)
            {
                continue;
            }

            for (var index = 0; index < dimension; index++)
            {
                buffer[index] += vector[index];
            }

            count++;
        }

        if (count == 0)
        {
            return [];
        }

        var result = new float[dimension];
        var norm = 0.0;
        for (var index = 0; index < dimension; index++)
        {
            result[index] = (float)(buffer[index] / count);
            norm += result[index] * result[index];
        }

        norm = Math.Sqrt(norm);
        if (norm <= 0.0)
        {
            return [];
        }

        for (var index = 0; index < dimension; index++)
        {
            result[index] = (float)(result[index] / norm);
        }

        return result;
    }

    private static double Cosine(IReadOnlyList<float> left, IReadOnlyList<float> right)
    {
        if (left.Count == 0 || left.Count != right.Count)
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

        return Math.Clamp(dot / (Math.Sqrt(leftNorm) * Math.Sqrt(rightNorm)), -1.0, 1.0);
    }

    private static string NormalizeHostType(string hostType)
        => hostType.Trim().ToLowerInvariant() switch
        {
            "video" or "videos" => "video",
            "image" or "images" => "image",
            var other => other,
        };
}
