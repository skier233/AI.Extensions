using Cove.Core.DTOs;
using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Cove.Data;

using Microsoft.EntityFrameworkCore;

namespace AI.Faces;

internal sealed class AiFaceSuggester(
    CoveContext db,
    IEmbeddingService embeddingService,
    AiFaceReferencePackStore referencePackStore,
    AiFaceReferenceSuggestionDecisionStore referenceSuggestionDecisionStore,
    AiFaceReferencePerformerResolver referencePerformerResolver) : IFaceSuggester
{
    private const int SourceEmbeddingCount = 4;
    private const int CandidateK = 40;
    private const int ReferenceCandidateK = 12;
    private const int EvidencePerSuggestion = 3;

    private readonly CoveContext _db = db;
    private readonly IEmbeddingService _embeddingService = embeddingService;
    private readonly AiFaceReferencePackStore _referencePackStore = referencePackStore;
    private readonly AiFaceReferenceSuggestionDecisionStore _referenceSuggestionDecisionStore = referenceSuggestionDecisionStore;
    private readonly AiFaceReferencePerformerResolver _referencePerformerResolver = referencePerformerResolver;

    public Task<IReadOnlyList<FaceSuggestionDto>> SuggestForAsync(
        int faceId,
        int maxResults,
        CancellationToken cancellationToken = default)
        => SuggestForAsync(faceId, maxResults, new FaceSuggestionOptions(), cancellationToken);

    public async Task<IReadOnlyList<FaceSuggestionDto>> SuggestForAsync(
        int faceId,
        int maxResults,
        FaceSuggestionOptions options,
        CancellationToken cancellationToken = default)
    {
        var batch = await SuggestForBatchAsync([faceId], maxResults, options, cancellationToken);
        return batch.TryGetValue(faceId, out var suggestions) ? suggestions : [];
    }

    public async Task<IReadOnlyDictionary<int, IReadOnlyList<FaceSuggestionDto>>> SuggestForBatchAsync(
        IReadOnlyCollection<int> faceIds,
        int maxResults,
        FaceSuggestionOptions options,
        CancellationToken cancellationToken = default)
    {
        maxResults = Math.Clamp(maxResults, 1, 10);
        var distinctFaceIds = faceIds
            .Where(static faceId => faceId > 0)
            .Distinct()
            .ToArray();
        if (distinctFaceIds.Length == 0)
        {
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();
        }

        var eligibleFaceIds = await _db.Faces
            .AsNoTracking()
            .Where(face => distinctFaceIds.Contains(face.Id) && !face.PerformerId.HasValue && !face.MergedIntoFaceId.HasValue)
            .Select(face => face.Id)
            .ToArrayAsync(cancellationToken);
        if (eligibleFaceIds.Length == 0)
        {
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();
        }

        var sourceEmbeddings = await _db.Embeddings
            .AsNoTracking()
            .Where(embedding =>
                embedding.HostType == EmbeddingHostType.Face
                && eligibleFaceIds.Contains(embedding.HostId)
                && embedding.Modality == EmbeddingModality.Face)
            .OrderByDescending(embedding => embedding.CreatedAt)
            .ToListAsync(cancellationToken);

        var sourceEmbeddingsByFaceId = sourceEmbeddings
            .GroupBy(static embedding => embedding.HostId)
            .ToDictionary(
                static group => group.Key,
                static group => (IReadOnlyList<Embedding>)group.Take(SourceEmbeddingCount).ToArray());
        if (sourceEmbeddingsByFaceId.Count == 0)
        {
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();
        }

        var referenceSuggestionsByFaceId = options.IncludeReferenceMatches
            ? await BuildReferenceSuggestionsByFaceAsync(sourceEmbeddingsByFaceId, maxResults, cancellationToken)
            : new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();
        var suggestionsByFaceIdResult = new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        foreach (var faceId in eligibleFaceIds)
        {
            if (!sourceEmbeddingsByFaceId.TryGetValue(faceId, out var faceSourceEmbeddings) || faceSourceEmbeddings.Count == 0)
            {
                continue;
            }

            var rawMatches = new List<RawFaceMatch>();
            foreach (var sourceEmbedding in faceSourceEmbeddings)
            {
                var nearest = await _embeddingService.KnnAsync(
                    sourceEmbedding.Vector,
                    CandidateK,
                    new EmbeddingSearchOptions
                    {
                        HostType = EmbeddingHostType.Face,
                        KindFamily = sourceEmbedding.KindFamily,
                        var referenceSuggestionsByFaceId = options.IncludeReferenceMatches && sourceEmbeddingsByFaceId.Count > 0
                rawMatches.AddRange(nearest
                    .Where(match => match.Embedding.HostId != faceId)
                    .Select(match => new RawFaceMatch(match.Embedding.HostId, ClampSimilarity(1f - match.Distance))));
            }

            var localSuggestions = rawMatches.Count == 0
            {
                            if (sourceEmbeddingsByFaceId.TryGetValue(faceId, out var faceSourceEmbeddings) && faceSourceEmbeddings.Count > 0)
            }
                                foreach (var sourceEmbedding in faceSourceEmbeddings)
                                {
                                    var nearest = await _embeddingService.KnnAsync(
                                        sourceEmbedding.Vector,
                                        CandidateK,
                                        new EmbeddingSearchOptions
                                        {
                                            HostType = EmbeddingHostType.Face,
                                            KindFamily = sourceEmbedding.KindFamily,
                                            Modality = sourceEmbedding.Modality,
                                            IsSemantic = sourceEmbedding.IsSemantic,
                                            SourceKey = sourceEmbedding.SourceKey,
                                        },
                                        cancellationToken);

                                    rawMatches.AddRange(nearest
                                        .Where(match => match.Embedding.HostId != faceId)
                                        .Select(match => new RawFaceMatch(match.Embedding.HostId, ClampSimilarity(1f - match.Distance))));
                                }
                            }

                            var localSuggestions = rawMatches.Count == 0
                                ? []
                                : await BuildLocalSuggestionsAsync(rawMatches, cancellationToken);
                            referenceSuggestionsByFaceId.TryGetValue(faceId, out var referenceSuggestions);
                            var suggestions = MergeAndRankSuggestions(localSuggestions, referenceSuggestions ?? []);
                            var rankedSuggestions = await ApplySceneEvidenceBoostAsync(faceId, suggestions, maxResults, cancellationToken);
                            if (rankedSuggestions.Count > 0)
                            {
                                suggestionsByFaceIdResult[faceId] = rankedSuggestions;
                            }
                        }

                        return suggestionsByFaceIdResult;
                    }

                    private async Task<IReadOnlyList<FaceSuggestionDto>> ApplySceneEvidenceBoostAsync(
                        int faceId,
                        IReadOnlyList<FaceSuggestionDto> suggestions,
                        int maxResults,
                        CancellationToken cancellationToken)
                    {
                        var sceneIds = await _db.FaceAppearances
                            .AsNoTracking()
                            .Where(appearance => appearance.FaceId == faceId && appearance.HostType == FaceAppearanceHostType.Scene)
                            .Select(appearance => appearance.HostId)
                            .Distinct()
                            .ToListAsync(cancellationToken);

                        if (sceneIds.Count == 0)
                        {
                            sceneIds = await _db.Detections
                                .AsNoTracking()
                                .Where(detection =>
                                    detection.RefId == faceId
                                    && detection.RefKind != null
                                    && detection.RefKind.ToLower() == "face"
                                    && detection.HostType == DetectionHostType.Scene)
                                .Select(detection => detection.HostId)
                                .Distinct()
                                .ToListAsync(cancellationToken);
                        }

                        var imageIds = await _db.FaceAppearances
                            .AsNoTracking()
                            .Where(appearance => appearance.FaceId == faceId && appearance.HostType == FaceAppearanceHostType.Image)
                            .Select(appearance => appearance.HostId)
                            .Distinct()
                            .ToListAsync(cancellationToken);

                        if (imageIds.Count == 0)
                        {
                            imageIds = await _db.Detections
                                .AsNoTracking()
                                .Where(detection =>
                                    detection.RefId == faceId
                                    && detection.RefKind != null
                                    && detection.RefKind.ToLower() == "face"
                                    && detection.HostType == DetectionHostType.Image)
                                .Select(detection => detection.HostId)
                                .Distinct()
                                .ToListAsync(cancellationToken);
                        }

                        var hostKeys = sceneIds.Select(static id => $"scene:{id}")
                            .Concat(imageIds.Select(static id => $"image:{id}"))
                            .Distinct(StringComparer.OrdinalIgnoreCase)
                            .ToArray();
                        if (hostKeys.Length == 0)
                        {
                            return suggestions.Take(maxResults).ToList();
                        }

                        var scenePerformers = await _db.Set<ScenePerformer>()
                            .AsNoTracking()
                            .Include(item => item.Performer)
                                .ThenInclude(performer => performer!.RemoteIds)
                            .Where(item => sceneIds.Contains(item.SceneId) && item.Performer != null)
                            .ToListAsync(cancellationToken);

                        var imagePerformers = await _db.Set<ImagePerformer>()
                            .AsNoTracking()
                            .Include(item => item.Performer)
                                .ThenInclude(performer => performer!.RemoteIds)
                            .Where(item => imageIds.Contains(item.ImageId) && item.Performer != null)
                            .ToListAsync(cancellationToken);

                        var hostPerformers = scenePerformers
                            .Select(item => new HostPerformerEvidence(
                                item.PerformerId,
                                item.Performer!.Name,
                                item.Performer.UpdatedAt,
                                !string.IsNullOrWhiteSpace(item.Performer.ImageBlobId),
                                item.Performer.RemoteIds.Count == 0,
                                $"scene:{item.SceneId}"))
                            .Concat(imagePerformers.Select(item => new HostPerformerEvidence(
                                item.PerformerId,
                                item.Performer!.Name,
                                item.Performer.UpdatedAt,
                                !string.IsNullOrWhiteSpace(item.Performer.ImageBlobId),
                                item.Performer.RemoteIds.Count == 0,
                                $"image:{item.ImageId}")))
                            .ToArray();

                        if (hostPerformers.Length == 0)
                        {
                            return suggestions.Take(maxResults).ToList();
                        }

                        var performerHostKeys = hostPerformers
                            .GroupBy(item => item.PerformerId)
                            .ToDictionary(group => group.Key, group => group.Select(item => item.HostKey).Distinct(StringComparer.OrdinalIgnoreCase).ToArray());
                        var performerDetailsById = hostPerformers
                            .GroupBy(item => item.PerformerId)
                            .ToDictionary(group => group.Key, group => group.OrderByDescending(item => item.PerformerUpdatedAt).First());
                        var normalizedNameHostKeys = hostPerformers
                            .GroupBy(item => NormalizeName(item.PerformerName), StringComparer.OrdinalIgnoreCase)
                            .Where(group => !string.IsNullOrWhiteSpace(group.Key))
                            .ToDictionary(group => group.Key, group => group.Select(item => item.HostKey).Distinct(StringComparer.OrdinalIgnoreCase).ToArray(), StringComparer.OrdinalIgnoreCase);
                        var solePerformerHostKeys = hostPerformers
                            .GroupBy(item => item.HostKey, StringComparer.OrdinalIgnoreCase)
                            .Where(group => group.Select(item => item.PerformerId).Distinct().Count() == 1)
                            .Select(group => group.Key)
                            .ToHashSet(StringComparer.OrdinalIgnoreCase);
                        var exclusivePerformerId = performerHostKeys.Count == 1 && performerHostKeys.Values.Single().Length == hostKeys.Length
                            ? performerHostKeys.Keys.Single()
                            : (int?)null;

                        var boostedSuggestions = suggestions
                            .Select(suggestion => ApplySceneEvidenceBoost(suggestion, performerHostKeys, normalizedNameHostKeys, solePerformerHostKeys, hostKeys.Length, exclusivePerformerId))
                            .ToList();

                        var existingLocalPerformerIds = boostedSuggestions
                            .Select(ResolveLocalPerformerId)
                            .Where(static performerId => performerId.HasValue)
                            .Select(static performerId => performerId!.Value)
                            .ToHashSet();

                        foreach (var performerId in performerHostKeys.Keys)
                        {
                            if (existingLocalPerformerIds.Contains(performerId) || !performerDetailsById.TryGetValue(performerId, out var performer))
                            {
                                continue;
                            }

                            boostedSuggestions.Add(BuildHostEvidenceSuggestion(
                                performer,
                                performerHostKeys[performerId].Length,
                                hostKeys.Length,
                                performerHostKeys[performerId].Count(solePerformerHostKeys.Contains),
                                exclusivePerformerId == performerId));
                        }

                        return boostedSuggestions
                            .GroupBy(GetSuggestionKey, StringComparer.OrdinalIgnoreCase)
                            .Select(group => group
                                .OrderByDescending(item => item.Confidence)
                                .ThenByDescending(item => item.Evidence.Count)
                                .ThenBy(item => item.PerformerName)
                                .First())
                            .OrderByDescending(static suggestion => suggestion.Confidence)
                            .ThenByDescending(static suggestion => suggestion.Evidence.Count)
                            .ThenBy(static suggestion => suggestion.PerformerName)
                            .Take(maxResults)
                            .ToList();
                    }
            boost += 10f;
        }

        if (exclusivePerformerId.HasValue && localPerformerId == exclusivePerformerId.Value)
        {
            boost += 15f;
        }

        var hostSummary = matchedHostKeys.Length == 1
            ? $"already appears on 1 matching host"
            : $"already appears on {matchedHostKeys.Length} matching hosts";
        var fullCoverageSummary = fullCoverage ? "; tagged on every host containing this face" : string.Empty;
        var soleSummary = soleHostKeys.Length > 0
            ? soleHostKeys.Length == 1 ? "; sole performer on 1 of those hosts" : $"; sole performer on {soleHostKeys.Length} of those hosts"
            : string.Empty;

        return suggestion with
        {
            Confidence = MathF.Min(100f, suggestion.Confidence + boost),
            Why = $"{suggestion.Why} Host evidence: {hostSummary}{fullCoverageSummary}{soleSummary}.",
        };
    }

    private static FaceSuggestionDto BuildHostEvidenceSuggestion(HostPerformerEvidence performer, int matchedHostCount, int totalHostCount, int soleHostCount, bool exclusive)
    {
        var coverageRatio = totalHostCount <= 0 ? 0f : matchedHostCount / (float)totalHostCount;
        var soleRatio = matchedHostCount <= 0 ? 0f : soleHostCount / (float)matchedHostCount;
        var hostCountBonus = MathF.Min(15f, matchedHostCount * 3f);
        var confidence = 30f + (coverageRatio * 25f) + (soleRatio * 15f) + hostCountBonus + (exclusive ? 15f : 0f);
        confidence = MathF.Round(MathF.Min(85f, confidence), 1);
        var hostSummary = matchedHostCount == 1
            ? "1 tagged host"
            : $"{matchedHostCount} tagged hosts";
        var coverageSummary = totalHostCount > matchedHostCount
            ? $" out of {totalHostCount} hosts containing this face"
            : totalHostCount == 1 ? string.Empty : " containing this face";
        var why = $"Host evidence only: this performer is assigned on {hostSummary}{coverageSummary}.";

        if (soleHostCount > 0)
        {
            why += soleHostCount == 1
                ? " It is also the sole performer on one matching host."
                : $" It is also the sole performer on {soleHostCount} matching hosts.";
        }

        if (exclusive)
        {
            why += " Every tagged host containing this face points to this performer.";
        }

        return new FaceSuggestionDto(
            performer.PerformerId,
            performer.PerformerName,
            BuildPerformerCoverUrl(performer.PerformerId, performer.PerformerUpdatedAt, performer.PerformerHasImage),
            confidence,
            why,
            [],
            LocalPerformerId: performer.PerformerId,
            LocalPerformerHasImage: performer.PerformerHasImage,
            LocalPerformerIsLocalOnly: performer.PerformerIsLocalOnly);
    }

    private static int? ResolveLocalPerformerId(FaceSuggestionDto suggestion)
        => suggestion.LocalPerformerId ?? (suggestion.PerformerId > 0 ? suggestion.PerformerId : null);

    private async Task<IReadOnlyList<FaceSuggestionDto>> BuildLocalSuggestionsAsync(
        IReadOnlyList<RawFaceMatch> rawMatches,
        CancellationToken cancellationToken)
    {
        var candidateFaceIds = rawMatches.Select(match => match.FaceId).Distinct().ToArray();
        var candidateFaces = await _db.Faces
            .AsNoTracking()
            .Include(item => item.Performer)
                .ThenInclude(performer => performer!.RemoteIds)
            .Where(item =>
                candidateFaceIds.Contains(item.Id)
                && item.PerformerId.HasValue
                && item.MergedIntoFaceId == null
                && !item.Ignored)
            .ToDictionaryAsync(item => item.Id, cancellationToken);
        if (candidateFaces.Count == 0)
        {
            return [];
        }

        var matchedFaceIds = candidateFaces.Keys.ToArray();
        var detections = await _db.Detections
            .AsNoTracking()
            .Where(detection =>
                detection.RefId.HasValue
                && detection.RefKind != null
                && detection.RefKind.ToLower() == "face"
                && matchedFaceIds.Contains((int)detection.RefId.Value))
            .ToListAsync(cancellationToken);
        var bestDetectionsByFaceId = detections
            .GroupBy(detection => (int)detection.RefId!.Value)
            .ToDictionary(
                group => group.Key,
                group => group
                    .OrderByDescending(detection => detection.Score)
                    .ThenByDescending(detection => detection.UpdatedAt)
                    .First());

        var suggestionInputs = rawMatches
            .Where(match => candidateFaces.ContainsKey(match.FaceId))
            .Select(match =>
            {
                var candidate = candidateFaces[match.FaceId];
                return new CandidateFaceMatch(
                    candidate.Id,
                    candidate.PerformerId!.Value,
                    candidate.Performer?.Name ?? $"Performer #{candidate.PerformerId.Value}",
                    candidate.Performer?.UpdatedAt,
                    !string.IsNullOrWhiteSpace(candidate.Performer?.ImageBlobId),
                    candidate.Performer?.RemoteIds.Count == 0,
                    match.Similarity,
                    bestDetectionsByFaceId.GetValueOrDefault(candidate.Id));
            })
            .ToArray();
        if (suggestionInputs.Length == 0)
        {
            return [];
        }

        return suggestionInputs
            .GroupBy(match => new { match.PerformerId, match.PerformerName })
            .Select(group => BuildLocalSuggestion(group.Key.PerformerId, group.Key.PerformerName, group))
            .ToList();
    }

    private async Task<IReadOnlyList<FaceSuggestionDto>> BuildReferenceSuggestionsAsync(
        int faceId,
        IReadOnlyList<Embedding> sourceEmbeddings,
        int maxResults,
        CancellationToken cancellationToken)
    {
        var suggestionsByFaceId = await BuildReferenceSuggestionsByFaceAsync(
            new Dictionary<int, IReadOnlyList<Embedding>>
            {
                [faceId] = sourceEmbeddings,
            },
            maxResults,
            cancellationToken);

        return suggestionsByFaceId.TryGetValue(faceId, out var suggestions) ? suggestions : [];
    }

    private async Task<IReadOnlyDictionary<int, IReadOnlyList<FaceSuggestionDto>>> BuildReferenceSuggestionsByFaceAsync(
        IReadOnlyDictionary<int, IReadOnlyList<Embedding>> sourceEmbeddingsByFaceId,
        int maxResults,
        CancellationToken cancellationToken)
    {
        var pack = await _referencePackStore.GetActivePackAsync(cancellationToken);
        if (pack is null || pack.Identities.Count == 0)
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        var rejectedByFaceId = await _referenceSuggestionDecisionStore.GetRejectedAsync(sourceEmbeddingsByFaceId.Keys.ToArray(), cancellationToken);
        var matchesByFaceId = new Dictionary<int, List<RawReferenceMatch>>();
        foreach (var (faceId, sourceEmbeddings) in sourceEmbeddingsByFaceId)
        {
            rejectedByFaceId.TryGetValue(faceId, out var rejectedIdentityIds);
            foreach (var sourceVector in sourceEmbeddings
                         .Select(embedding => embedding.Vector.ToArray())
                         .Where(vector => vector.Length == pack.Manifest.EmbeddingDim))
            {
                var matches = FindNearestReferenceMatches(sourceVector, pack, ReferenceCandidateK)
                    .Where(match => rejectedIdentityIds is null || !rejectedIdentityIds.Contains(match.Identity.ExternalId));

                if (!matchesByFaceId.TryGetValue(faceId, out var faceMatches))
                {
                    faceMatches = [];
                    matchesByFaceId[faceId] = faceMatches;
                }

                faceMatches.AddRange(matches);
            }
        }

        if (matchesByFaceId.Count == 0)
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        var candidateIdentities = matchesByFaceId.Values
            .SelectMany(static matches => matches)
            .Select(match => match.Identity)
            .DistinctBy(identity => identity.ExternalId, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        var performerMatches = await _referencePerformerResolver.ResolveAsync(candidateIdentities, pack.Manifest.SourceEndpoint, cancellationToken);

        return matchesByFaceId
            .ToDictionary(
                pair => pair.Key,
                pair => (IReadOnlyList<FaceSuggestionDto>)pair.Value
                    .GroupBy(match => match.Identity.ExternalId, StringComparer.OrdinalIgnoreCase)
                    .Select(group => BuildReferenceSuggestion(group, performerMatches.GetValueOrDefault(group.Key), pack.Manifest))
                    .OrderByDescending(item => item.Confidence)
                    .ThenBy(item => item.PerformerName)
                    .Take(maxResults)
                    .ToList());
    }

    private static List<FaceSuggestionDto> MergeAndRankSuggestions(
        IEnumerable<FaceSuggestionDto> localSuggestions,
        IEnumerable<FaceSuggestionDto> referenceSuggestions)
        => localSuggestions
            .Concat(referenceSuggestions)
            .GroupBy(GetSuggestionKey, StringComparer.OrdinalIgnoreCase)
            .Select(group => group
                .OrderByDescending(item => item.Confidence)
                .ThenByDescending(item => item.Evidence.Count)
                .ThenBy(item => item.PerformerName)
                .First())
            .OrderByDescending(static suggestion => suggestion.Confidence)
            .ThenByDescending(static suggestion => suggestion.Evidence.Count)
            .ThenBy(static suggestion => suggestion.PerformerName)
            .ToList();

    private static FaceSuggestionDto BuildLocalSuggestion(
        int performerId,
        string performerName,
        IEnumerable<CandidateFaceMatch> matches)
    {
        var groupedMatches = matches.ToArray();
        var bestPerFace = groupedMatches
            .GroupBy(match => match.FaceId)
            .Select(group => group.OrderByDescending(static item => item.Similarity).First())
            .OrderByDescending(static item => item.Similarity)
            .ToArray();

        var evidence = bestPerFace
            .Take(EvidencePerSuggestion)
            .Select(match => new FaceSuggestionEvidenceDto(
                match.FaceId,
                BuildThumbnailUrl(match.Detection),
                match.Similarity))
            .ToArray();

        var topSimilarity = bestPerFace.Max(static match => match.Similarity);
        var meanSimilarity = bestPerFace.Take(Math.Min(3, bestPerFace.Length)).Average(static match => match.Similarity);
        var uniqueFaceCount = bestPerFace.Length;
        var observationCount = groupedMatches.Length;
        var confidence = ComputeConfidence(topSimilarity, meanSimilarity, uniqueFaceCount, observationCount);
        var why = BuildWhy(uniqueFaceCount, observationCount, topSimilarity, meanSimilarity);

        return new FaceSuggestionDto(
            performerId,
            performerName,
            BuildPerformerCoverUrl(performerId, groupedMatches[0].PerformerUpdatedAt, groupedMatches[0].PerformerHasImage),
            confidence,
            why,
            evidence,
            LocalPerformerId: performerId,
            LocalPerformerHasImage: groupedMatches[0].PerformerHasImage,
            LocalPerformerIsLocalOnly: groupedMatches[0].PerformerIsLocalOnly);
    }

    private static FaceSuggestionDto BuildReferenceSuggestion(
        IGrouping<string, RawReferenceMatch> matches,
        AiFaceReferencePerformerMatch? performerMatch,
        SaieManifest manifest)
    {
        var ordered = matches
            .OrderByDescending(match => match.Similarity)
            .ToArray();
        var identity = ordered[0].Identity;
        var topSimilarity = ordered[0].Similarity;
        var meanSimilarity = ordered.Take(Math.Min(3, ordered.Length)).Average(match => match.Similarity);
        var confidence = ComputeReferenceConfidence(topSimilarity, meanSimilarity, ordered.Length);
        var supportSummary = ordered.Length == 1 ? "1 source embedding" : $"{ordered.Length} source embeddings";
        var sourceLabel = string.IsNullOrWhiteSpace(manifest.PackId) ? "reference pack" : manifest.PackId;
        var why = performerMatch is null
            ? $"{sourceLabel} match; best {Math.Round(topSimilarity * 100)}%, mean {Math.Round(meanSimilarity * 100)}% across {supportSummary}."
            : $"{sourceLabel} matched an existing performer; best {Math.Round(topSimilarity * 100)}%, mean {Math.Round(meanSimilarity * 100)}% across {supportSummary}.";

        return new FaceSuggestionDto(
            performerMatch?.PerformerId ?? AiFaceReferenceSuggestionIds.FromOrdinal(identity.Ordinal),
            performerMatch?.PerformerName ?? identity.DisplayName,
            performerMatch is null
                ? identity.ImageUrl
                : BuildPerformerCoverUrl(performerMatch.PerformerId, performerMatch.PerformerUpdatedAt, performerMatch.LocalPerformerHasImage) ?? identity.ImageUrl,
            confidence,
            why,
            [],
            LocalPerformerId: performerMatch?.PerformerId,
            ExternalUrl: BuildReferenceProfileUrl(manifest.SourceEndpoint, identity.ExternalId),
            LocalPerformerHasImage: performerMatch?.LocalPerformerHasImage ?? false,
            LocalPerformerIsLocalOnly: performerMatch?.LocalPerformerIsLocalOnly ?? false);
    }

    private static float ComputeConfidence(float topSimilarity, double meanSimilarity, int uniqueFaceCount, int observationCount)
    {
        var faceCountWeight = Math.Min(1.0, uniqueFaceCount / 3.0);
        var observationWeight = Math.Min(1.0, observationCount / 6.0);
        var confidence = (topSimilarity * 0.5f)
                         + ((float)meanSimilarity * 0.35f)
                         + ((float)faceCountWeight * 0.1f)
                         + ((float)observationWeight * 0.05f);

        return MathF.Round(Math.Clamp(confidence, 0f, 1f) * 100f, 1);
    }

    private static float ComputeReferenceConfidence(float topSimilarity, double meanSimilarity, int observationCount)
    {
        var observationWeight = Math.Min(1.0, observationCount / 4.0);
        var confidence = (topSimilarity * 0.65f)
                         + ((float)meanSimilarity * 0.3f)
                         + ((float)observationWeight * 0.05f);

        return MathF.Round(Math.Clamp(confidence, 0f, 1f) * 100f, 1);
    }

    private static string BuildWhy(int uniqueFaceCount, int observationCount, float topSimilarity, double meanSimilarity)
    {
        var faceSummary = uniqueFaceCount == 1 ? "1 linked face cluster" : $"{uniqueFaceCount} linked face clusters";
        var observationSummary = observationCount == 1 ? "1 supporting match" : $"{observationCount} supporting matches";
        return $"{faceSummary} agree across {observationSummary}; best {Math.Round(topSimilarity * 100)}%, mean {Math.Round(meanSimilarity * 100)}%.";
    }

    private static string? BuildReferenceProfileUrl(string? sourceEndpoint, string externalId)
    {
        if (string.IsNullOrWhiteSpace(sourceEndpoint) || string.IsNullOrWhiteSpace(externalId))
        {
            return null;
        }

        var baseUrl = sourceEndpoint.Trim().TrimEnd('/');
        if (baseUrl.EndsWith("/graphql", StringComparison.OrdinalIgnoreCase))
        {
            baseUrl = baseUrl[..^"/graphql".Length];
        }

        return $"{baseUrl}/performers/{Uri.EscapeDataString(externalId)}";
    }

    private static string NormalizeName(string? value)
        => string.IsNullOrWhiteSpace(value) ? string.Empty : value.Trim().ToUpperInvariant();

    private static float ClampSimilarity(float similarity)
        => Math.Clamp(similarity, 0f, 1f);

    private static string? BuildPerformerCoverUrl(int performerId, DateTime? performerUpdatedAt, bool performerHasImage)
    {
        if (!performerHasImage || !performerUpdatedAt.HasValue)
        {
            return null;
        }

        return $"/api/performers/{performerId}/image?max=640&v={Uri.EscapeDataString(performerUpdatedAt.Value.ToString("o"))}";
    }

    private static string? BuildThumbnailUrl(Detection? detection)
    {
        if (detection is null)
        {
            return null;
        }

        return detection.HostType switch
        {
            DetectionHostType.Image => $"/api/stream/image/{detection.HostId}/thumbnail?max=320",
            DetectionHostType.Scene => $"/api/stream/scene/{detection.HostId}/screenshot",
            _ => null,
        };
    }

    private static IEnumerable<RawReferenceMatch> FindNearestReferenceMatches(float[] sourceVector, SaieReferencePack pack, int candidateCount)
    {
        if (candidateCount <= 0)
            return [];

        var sourceNorm = ComputeNorm(sourceVector);
        if (sourceNorm <= 0f)
            return [];

        var best = new List<RawReferenceMatch>(candidateCount);
        for (var ordinal = 0; ordinal < pack.Identities.Count; ordinal++)
        {
            var similarity = ComputeCosineSimilarity(sourceVector, sourceNorm, pack.GetCentroid(ordinal), pack.GetCentroidNorm(ordinal));
            if (similarity <= 0f)
                continue;

            var match = new RawReferenceMatch(pack.Identities[ordinal], similarity);
            if (best.Count < candidateCount)
            {
                best.Add(match);
                best.Sort(static (left, right) => left.Similarity.CompareTo(right.Similarity));
                continue;
            }

            if (similarity <= best[0].Similarity)
                continue;

            best[0] = match;
            best.Sort(static (left, right) => left.Similarity.CompareTo(right.Similarity));
        }

        return best.OrderByDescending(match => match.Similarity).ToArray();
    }

    private static float ComputeCosineSimilarity(float[] left, float leftNorm, ReadOnlySpan<float> right, float rightNorm)
    {
        if (left.Length != right.Length || leftNorm <= 0f || rightNorm <= 0f)
            return 0f;

        var dot = 0f;
        for (var index = 0; index < left.Length; index++)
            dot += left[index] * right[index];

        return ClampSimilarity(dot / (leftNorm * rightNorm));
    }

    private static float ComputeNorm(float[] vector)
    {
        var sum = 0f;
        for (var index = 0; index < vector.Length; index++)
            sum += vector[index] * vector[index];

        return MathF.Sqrt(sum);
    }

    private static string GetSuggestionKey(FaceSuggestionDto suggestion)
        => $"performer:{suggestion.PerformerId}";

    private sealed record RawFaceMatch(int FaceId, float Similarity);

    private sealed record RawReferenceMatch(SaieReferenceIdentity Identity, float Similarity);

    private sealed record CandidateFaceMatch(
        int FaceId,
        int PerformerId,
        string PerformerName,
        DateTime? PerformerUpdatedAt,
        bool PerformerHasImage,
        bool PerformerIsLocalOnly,
        float Similarity,
        Detection? Detection);

    private sealed record HostPerformerEvidence(
        int PerformerId,
        string PerformerName,
        DateTime? PerformerUpdatedAt,
        bool PerformerHasImage,
        bool PerformerIsLocalOnly,
        string HostKey);
}