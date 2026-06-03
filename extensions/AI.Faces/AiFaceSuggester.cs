using Cove.Core.DTOs;
using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Faces;

internal sealed class AiFaceSuggester(
    IFaceRepository faceRepository,
    IEmbeddingRepository embeddingRepository,
    IDetectionRepository detectionRepository,
    IVideoRepository videoRepository,
    IImageRepository imageRepository,
    IEmbeddingService embeddingService,
    AiFaceReferencePackStore referencePackStore,
    AiFaceReferenceSuggestionDecisionStore referenceSuggestionDecisionStore,
    AiFaceReferencePerformerResolver referencePerformerResolver) : IFaceSuggester
{
    private const int SourceEmbeddingCount = 4;
    private const int CandidateK = 40;
    private const int ReferenceCandidateK = 12;
    private const int EvidencePerSuggestion = 3;

    public Task<IReadOnlyList<FaceSuggestionDto>> SuggestForAsync(int faceId, int maxResults, CancellationToken cancellationToken = default)
        => SuggestForAsync(faceId, maxResults, new FaceSuggestionOptions(), cancellationToken);

    public async Task<IReadOnlyList<FaceSuggestionDto>> SuggestForAsync(int faceId, int maxResults, FaceSuggestionOptions options, CancellationToken cancellationToken = default)
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
        var distinctFaceIds = faceIds.Where(static id => id > 0).Distinct().ToArray();
        if (distinctFaceIds.Length == 0)
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        var eligibleFaces = await faceRepository.FindFacesAsync(new FaceFilter
        {
            Ids = distinctFaceIds,
            HasPerformer = false,
            IsMerged = false,
        }, tracking: false, cancellationToken);
        var eligibleFaceIds = eligibleFaces.Select(static f => f.Id).ToArray();
        if (eligibleFaceIds.Length == 0)
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        var allSourceEmbeddings = await embeddingRepository.FindAsync(new EmbeddingFilter
        {
            HostType = EmbeddingHostType.Face,
            HostIds = eligibleFaceIds,
            Modality = EmbeddingModality.Face,
        }, cancellationToken);

        var sourceEmbeddingsByFaceId = allSourceEmbeddings
            .OrderByDescending(static e => e.CreatedAt)
            .GroupBy(static e => e.HostId)
            .ToDictionary(static g => g.Key, static g => (IReadOnlyList<Embedding>)g.Take(SourceEmbeddingCount).ToArray());

        var referenceSuggestionsByFaceId = options.IncludeReferenceMatches && sourceEmbeddingsByFaceId.Count > 0
            ? await BuildReferenceSuggestionsByFaceAsync(sourceEmbeddingsByFaceId, maxResults, cancellationToken)
            : new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        var suggestionsByFaceIdResult = new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();
        foreach (var faceId in eligibleFaceIds)
        {
            var rawMatches = new List<RawFaceMatch>();
            if (sourceEmbeddingsByFaceId.TryGetValue(faceId, out var faceSourceEmbeddings) && faceSourceEmbeddings.Count > 0)
            {
                foreach (var sourceEmbedding in faceSourceEmbeddings)
                {
                    var nearest = await embeddingService.KnnAsync(
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

            var localSuggestions = rawMatches.Count == 0 ? [] : await BuildLocalSuggestionsAsync(rawMatches, cancellationToken);
            referenceSuggestionsByFaceId.TryGetValue(faceId, out var referenceSuggestions);
            var suggestions = MergeAndRankSuggestions(localSuggestions, referenceSuggestions ?? []);
            var rankedSuggestions = await ApplyVideoEvidenceBoostAsync(faceId, suggestions, maxResults, cancellationToken);
            if (rankedSuggestions.Count > 0)
                suggestionsByFaceIdResult[faceId] = rankedSuggestions;
        }

        return suggestionsByFaceIdResult;
    }

    private async Task<IReadOnlyList<FaceSuggestionDto>> ApplyVideoEvidenceBoostAsync(
        int faceId,
        IReadOnlyList<FaceSuggestionDto> suggestions,
        int maxResults,
        CancellationToken cancellationToken)
    {
        var videoAppearances = await faceRepository.FindAppearancesAsync(new FaceAppearanceFilter
        {
            FaceIds = [faceId],
            HostType = FaceAppearanceHostType.Video,
        }, cancellationToken);
        var videoIds = videoAppearances.Select(static a => a.HostId).Distinct().ToList();

        if (videoIds.Count == 0)
        {
            var videoDetections = await detectionRepository.FindAsync(new DetectionFilter
            {
                RefKind = "face",
                RefIds = [(long)faceId],
                HostType = DetectionHostType.Video,
            }, cancellationToken);
            videoIds = videoDetections.Select(static d => d.HostId).Distinct().ToList();
        }

        var imageAppearances = await faceRepository.FindAppearancesAsync(new FaceAppearanceFilter
        {
            FaceIds = [faceId],
            HostType = FaceAppearanceHostType.Image,
        }, cancellationToken);
        var imageIds = imageAppearances.Select(static a => a.HostId).Distinct().ToList();

        if (imageIds.Count == 0)
        {
            var imageDetections = await detectionRepository.FindAsync(new DetectionFilter
            {
                RefKind = "face",
                RefIds = [(long)faceId],
                HostType = DetectionHostType.Image,
            }, cancellationToken);
            imageIds = imageDetections.Select(static d => d.HostId).Distinct().ToList();
        }

        var hostKeys = videoIds.Select(static id => $"video:{id}")
            .Concat(imageIds.Select(static id => $"image:{id}"))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (hostKeys.Length == 0)
            return suggestions.Take(maxResults).ToList();

        var videoPerformers = videoIds.Count > 0
            ? await videoRepository.GetVideoPerformersAsync(videoIds, cancellationToken)
            : [];
        var imagePerformers = imageIds.Count > 0
            ? await imageRepository.GetImagePerformersAsync(imageIds, cancellationToken)
            : [];

        var hostPerformers = videoPerformers
            .Where(static vp => vp.Performer != null)
            .Select(vp => new HostPerformerEvidence(
                vp.PerformerId, vp.Performer!.Name, vp.Performer.UpdatedAt,
                !string.IsNullOrWhiteSpace(vp.Performer.ImageBlobId),
                vp.Performer.RemoteIds.Count == 0,
                $"video:{vp.VideoId}"))
            .Concat(imagePerformers
                .Where(static ip => ip.Performer != null)
                .Select(ip => new HostPerformerEvidence(
                    ip.PerformerId, ip.Performer!.Name, ip.Performer.UpdatedAt,
                    !string.IsNullOrWhiteSpace(ip.Performer.ImageBlobId),
                    ip.Performer.RemoteIds.Count == 0,
                    $"image:{ip.ImageId}")))
            .ToArray();

        if (hostPerformers.Length == 0)
            return suggestions.Take(maxResults).ToList();

        var performerHostKeys = hostPerformers.GroupBy(item => item.PerformerId)
            .ToDictionary(g => g.Key, g => g.Select(item => item.HostKey).Distinct(StringComparer.OrdinalIgnoreCase).ToArray());
        var performerDetailsById = hostPerformers.GroupBy(item => item.PerformerId)
            .ToDictionary(g => g.Key, g => g.OrderByDescending(item => item.PerformerUpdatedAt).First());
        var normalizedNameHostKeys = hostPerformers
            .GroupBy(item => NormalizeName(item.PerformerName), StringComparer.OrdinalIgnoreCase)
            .Where(static g => !string.IsNullOrWhiteSpace(g.Key))
            .ToDictionary(g => g.Key, g => g.Select(item => item.HostKey).Distinct(StringComparer.OrdinalIgnoreCase).ToArray(), StringComparer.OrdinalIgnoreCase);
        var solePerformerHostKeys = hostPerformers
            .GroupBy(item => item.HostKey, StringComparer.OrdinalIgnoreCase)
            .Where(static g => g.Select(item => item.PerformerId).Distinct().Count() == 1)
            .Select(static g => g.Key)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var performerCountByHostKey = hostPerformers
            .GroupBy(item => item.HostKey, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => g.Select(item => item.PerformerId).Distinct().Count(), StringComparer.OrdinalIgnoreCase);
        var exclusivePerformerId = performerHostKeys.Count == 1 && performerHostKeys.Values.Single().Length == hostKeys.Length
            ? performerHostKeys.Keys.Single() : (int?)null;

        var boostedSuggestions = suggestions
            .Select(suggestion => ApplyVideoEvidenceBoost(suggestion, performerHostKeys, normalizedNameHostKeys, solePerformerHostKeys, hostKeys.Length, exclusivePerformerId))
            .ToList();

        var existingLocalPerformerIds = boostedSuggestions
            .Select(ResolveLocalPerformerId)
            .Where(static id => id.HasValue)
            .Select(static id => id!.Value)
            .ToHashSet();

        foreach (var performerId in performerHostKeys.Keys)
        {
            if (existingLocalPerformerIds.Contains(performerId) || !performerDetailsById.TryGetValue(performerId, out var performer))
                continue;

            boostedSuggestions.Add(BuildHostEvidenceSuggestion(
                performer,
                performerHostKeys[performerId].Length,
                hostKeys.Length,
                performerHostKeys[performerId].Count(solePerformerHostKeys.Contains),
                performerHostKeys[performerId].Select(hk => performerCountByHostKey.GetValueOrDefault(hk, 1)).DefaultIfEmpty(1).Average(),
                exclusivePerformerId == performerId));
        }

        return boostedSuggestions
            .GroupBy(GetSuggestionKey, StringComparer.OrdinalIgnoreCase)
            .Select(static g => g.OrderByDescending(item => item.Confidence).ThenByDescending(item => item.Evidence.Count).ThenBy(item => item.PerformerName).First())
            .OrderByDescending(static s => s.Confidence).ThenByDescending(static s => s.Evidence.Count).ThenBy(static s => s.PerformerName)
            .Take(maxResults)
            .ToList();
    }

    private async Task<IReadOnlyList<FaceSuggestionDto>> BuildLocalSuggestionsAsync(IReadOnlyList<RawFaceMatch> rawMatches, CancellationToken cancellationToken)
    {
        var candidateFaceIds = rawMatches.Select(static m => m.FaceId).Distinct().ToArray();
        var candidateFaces = await faceRepository.FindFacesAsync(new FaceFilter
        {
            Ids = candidateFaceIds,
            HasPerformer = true,
            Ignored = false,
            IsMerged = false,
            IncludePerformer = true,
        }, tracking: false, cancellationToken);

        if (candidateFaces.Count == 0) return [];
        var candidateFaceDict = candidateFaces.ToDictionary(static f => f.Id);
        var matchedFaceIds = candidateFaceDict.Keys.ToArray();

        var detections = await detectionRepository.FindAsync(new DetectionFilter
        {
            RefKind = "face",
            RefIds = matchedFaceIds.Select(static id => (long)id).ToArray(),
        }, cancellationToken);

        var bestDetectionsByFaceId = detections
            .GroupBy(static d => (int)d.RefId!.Value)
            .ToDictionary(static g => g.Key, static g => g.OrderByDescending(static d => d.Score).ThenByDescending(static d => d.UpdatedAt).First());

        var suggestionInputs = rawMatches
            .Where(match => candidateFaceDict.ContainsKey(match.FaceId))
            .Select(match =>
            {
                var candidate = candidateFaceDict[match.FaceId];
                return new CandidateFaceMatch(
                    candidate.Id, candidate.PerformerId!.Value,
                    candidate.Performer?.Name ?? $"Performer #{candidate.PerformerId.Value}",
                    candidate.Performer?.UpdatedAt,
                    !string.IsNullOrWhiteSpace(candidate.Performer?.ImageBlobId),
                    candidate.Performer?.RemoteIds.Count == 0,
                    candidate.UpdatedAt,
                    !string.IsNullOrWhiteSpace(candidate.CoverBlobId),
                    match.Similarity,
                    bestDetectionsByFaceId.GetValueOrDefault(candidate.Id));
            })
            .ToArray();

        if (suggestionInputs.Length == 0) return [];

        return suggestionInputs
            .GroupBy(static m => new { m.PerformerId, m.PerformerName })
            .Select(static g => BuildLocalSuggestion(g.Key.PerformerId, g.Key.PerformerName, g))
            .ToList();
    }

    private async Task<IReadOnlyDictionary<int, IReadOnlyList<FaceSuggestionDto>>> BuildReferenceSuggestionsByFaceAsync(
        IReadOnlyDictionary<int, IReadOnlyList<Embedding>> sourceEmbeddingsByFaceId,
        int maxResults,
        CancellationToken cancellationToken)
    {
        var pack = await referencePackStore.GetActivePackAsync(cancellationToken);
        if (pack is null || pack.Identities.Count == 0)
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        var rejectedByFaceId = await referenceSuggestionDecisionStore.GetRejectedAsync(sourceEmbeddingsByFaceId.Keys.ToArray(), cancellationToken);
        var matchesByFaceId = new Dictionary<int, List<RawReferenceMatch>>();
        foreach (var (faceId, sourceEmbeddings) in sourceEmbeddingsByFaceId)
        {
            rejectedByFaceId.TryGetValue(faceId, out var rejectedIdentityIds);
            foreach (var sourceVector in sourceEmbeddings.Select(static e => e.Vector.ToArray()).Where(v => v.Length == pack.Manifest.EmbeddingDim))
            {
                var matches = FindNearestReferenceMatches(sourceVector, pack, ReferenceCandidateK)
                    .Where(match => rejectedIdentityIds is null || !rejectedIdentityIds.Contains(match.Identity.ExternalId));

                if (!matchesByFaceId.TryGetValue(faceId, out var faceMatches)) { faceMatches = []; matchesByFaceId[faceId] = faceMatches; }
                faceMatches.AddRange(matches);
            }
        }

        if (matchesByFaceId.Count == 0)
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        var candidateIdentities = matchesByFaceId.Values.SelectMany(static m => m).Select(static m => m.Identity)
            .DistinctBy(static i => i.ExternalId, StringComparer.OrdinalIgnoreCase).ToArray();
        var performerMatches = await referencePerformerResolver.ResolveAsync(candidateIdentities, pack.Manifest.SourceEndpoint, cancellationToken);

        return matchesByFaceId.ToDictionary(
            static pair => pair.Key,
            pair => (IReadOnlyList<FaceSuggestionDto>)pair.Value
                .GroupBy(static m => m.Identity.ExternalId, StringComparer.OrdinalIgnoreCase)
                .Select(g => BuildReferenceSuggestion(g, performerMatches.GetValueOrDefault(g.Key), pack.Manifest))
                .OrderByDescending(static s => s.Confidence).ThenBy(static s => s.PerformerName)
                .Take(maxResults).ToList());
    }

    private static List<FaceSuggestionDto> MergeAndRankSuggestions(IEnumerable<FaceSuggestionDto> local, IEnumerable<FaceSuggestionDto> reference)
        => local.Concat(reference)
            .GroupBy(GetSuggestionKey, StringComparer.OrdinalIgnoreCase)
            .Select(static g => g.OrderByDescending(static s => s.Confidence).ThenByDescending(static s => s.Evidence.Count).ThenBy(static s => s.PerformerName).First())
            .OrderByDescending(static s => s.Confidence).ThenByDescending(static s => s.Evidence.Count).ThenBy(static s => s.PerformerName)
            .ToList();

    private static FaceSuggestionDto ApplyVideoEvidenceBoost(FaceSuggestionDto suggestion, IReadOnlyDictionary<int, string[]> performerHostKeys, IReadOnlyDictionary<string, string[]> normalizedNameHostKeys, ISet<string> solePerformerHostKeys, int totalHostCount, int? exclusivePerformerId)
    {
        var localPerformerId = ResolveLocalPerformerId(suggestion);
        var normalizedName = NormalizeName(suggestion.PerformerName);
        var matchedHostKeys = localPerformerId.HasValue && performerHostKeys.TryGetValue(localPerformerId.Value, out var localHostKeys)
            ? localHostKeys
            : normalizedNameHostKeys.GetValueOrDefault(normalizedName) ?? [];

        if (matchedHostKeys.Length == 0) return suggestion;

        var soleHostKeys = matchedHostKeys.Where(solePerformerHostKeys.Contains).ToArray();
        var fullCoverage = totalHostCount > 0 && matchedHostKeys.Length == totalHostCount;
        var boost = MathF.Min(6f, matchedHostKeys.Length * 2f);
        if (soleHostKeys.Length > 0) boost += MathF.Min(4f, soleHostKeys.Length * 2f);
        if (fullCoverage) boost += 4f;
        if (totalHostCount > 1 && exclusivePerformerId.HasValue && localPerformerId == exclusivePerformerId.Value) boost += 3f;

        var hostSummary = matchedHostKeys.Length == 1 ? "already appears on 1 matching host" : $"already appears on {matchedHostKeys.Length} matching hosts";
        var fullCoverageSummary = fullCoverage ? "; tagged on every host containing this face" : string.Empty;
        var soleSummary = soleHostKeys.Length > 0 ? (soleHostKeys.Length == 1 ? "; sole performer on 1 of those hosts" : $"; sole performer on {soleHostKeys.Length} of those hosts") : string.Empty;

        return suggestion with
        {
            Confidence = MathF.Min(100f, suggestion.Confidence + boost),
            Why = $"{suggestion.Why} Host evidence: {hostSummary}{fullCoverageSummary}{soleSummary}.",
        };
    }

    private static FaceSuggestionDto BuildHostEvidenceSuggestion(HostPerformerEvidence performer, int matchedHostCount, int totalHostCount, int soleHostCount, double averagePerformerCountOnMatchedHosts, bool exclusive)
    {
        var coverageRatio = totalHostCount <= 0 ? 0f : matchedHostCount / (float)totalHostCount;
        var soleRatio = matchedHostCount <= 0 ? 0f : soleHostCount / (float)matchedHostCount;
        var repeatedHostBonus = MathF.Min(20f, Math.Max(0, matchedHostCount - 1) * 6f);
        var coverageBonus = totalHostCount > 1 ? coverageRatio * 8f : 0f;
        var soleConsistencyBonus = totalHostCount > 1 ? soleRatio * 6f : 0f;
        var confidence = ComputeHostEvidenceBaseConfidence(averagePerformerCountOnMatchedHosts) + repeatedHostBonus + coverageBonus + soleConsistencyBonus + (exclusive && totalHostCount > 1 ? 4f : 0f);
        confidence = MathF.Round(MathF.Min(72f, confidence), 1);
        var hostSummary = matchedHostCount == 1 ? "1 tagged host" : $"{matchedHostCount} tagged hosts";
        var coverageSummary = totalHostCount > matchedHostCount ? $" out of {totalHostCount} hosts containing this face" : totalHostCount == 1 ? string.Empty : " containing this face";
        var why = $"Host evidence only: this performer is assigned on {hostSummary}{coverageSummary}.";
        if (soleHostCount > 0) why += soleHostCount == 1 ? " It is also the sole performer on one matching host." : $" It is also the sole performer on {soleHostCount} matching hosts.";
        if (exclusive) why += " Every tagged host containing this face points to this performer.";

        return new FaceSuggestionDto(performer.PerformerId, performer.PerformerName,
            BuildPerformerCoverUrl(performer.PerformerId, performer.PerformerUpdatedAt, performer.PerformerHasImage),
            confidence, why, [],
            LocalPerformerId: performer.PerformerId,
            LocalPerformerHasImage: performer.PerformerHasImage,
            LocalPerformerIsLocalOnly: performer.PerformerIsLocalOnly);
    }

    private static float ComputeHostEvidenceBaseConfidence(double averagePerformerCount)
        => averagePerformerCount <= 1.25 ? 40f : averagePerformerCount <= 2.25 ? 30f : averagePerformerCount >= 4.0 ? 20f : 25f;

    private static int? ResolveLocalPerformerId(FaceSuggestionDto suggestion)
        => suggestion.LocalPerformerId ?? (suggestion.PerformerId > 0 ? suggestion.PerformerId : null);

    private static FaceSuggestionDto BuildLocalSuggestion(int performerId, string performerName, IEnumerable<CandidateFaceMatch> matches)
    {
        var groupedMatches = matches.ToArray();
        var bestPerFace = groupedMatches.GroupBy(static m => m.FaceId)
            .Select(static g => g.OrderByDescending(static m => m.Similarity).First())
            .OrderByDescending(static m => m.Similarity).ToArray();
        var evidence = bestPerFace.Take(EvidencePerSuggestion)
            .Select(static m => new FaceSuggestionEvidenceDto(m.FaceId, BuildFaceCoverUrl(m.FaceId, m.FaceUpdatedAt, m.FaceHasCoverImage) ?? BuildThumbnailUrl(m.Detection), m.Similarity))
            .ToArray();
        var topSimilarity = bestPerFace.Max(static m => m.Similarity);
        var meanSimilarity = bestPerFace.Take(Math.Min(3, bestPerFace.Length)).Average(static m => m.Similarity);
        var uniqueFaceCount = bestPerFace.Length;
        var observationCount = groupedMatches.Length;
        var confidence = ComputeConfidence(topSimilarity, meanSimilarity, uniqueFaceCount, observationCount);
        var why = BuildWhy(uniqueFaceCount, observationCount, topSimilarity, meanSimilarity);
        return new FaceSuggestionDto(performerId, performerName,
            BuildPerformerCoverUrl(performerId, groupedMatches[0].PerformerUpdatedAt, groupedMatches[0].PerformerHasImage),
            confidence, why, evidence,
            LocalPerformerId: performerId,
            LocalPerformerHasImage: groupedMatches[0].PerformerHasImage,
            LocalPerformerIsLocalOnly: groupedMatches[0].PerformerIsLocalOnly);
    }

    private static FaceSuggestionDto BuildReferenceSuggestion(IGrouping<string, RawReferenceMatch> matches, AiFaceReferencePerformerMatch? performerMatch, SaieManifest manifest)
    {
        var ordered = matches.OrderByDescending(static m => m.Similarity).ToArray();
        var identity = ordered[0].Identity;
        var topSimilarity = ordered[0].Similarity;
        var meanSimilarity = ordered.Take(Math.Min(3, ordered.Length)).Average(static m => m.Similarity);
        var confidence = ComputeReferenceConfidence(topSimilarity, meanSimilarity, ordered.Length);
        var supportSummary = ordered.Length == 1 ? "1 source embedding" : $"{ordered.Length} source embeddings";
        var sourceLabel = string.IsNullOrWhiteSpace(manifest.PackId) ? "reference pack" : manifest.PackId;
        var why = performerMatch is null
            ? $"{sourceLabel} match; best {Math.Round(topSimilarity * 100)}%, mean {Math.Round(meanSimilarity * 100)}% across {supportSummary}."
            : $"{sourceLabel} matched an existing performer; best {Math.Round(topSimilarity * 100)}%, mean {Math.Round(meanSimilarity * 100)}% across {supportSummary}.";
        return new FaceSuggestionDto(
            performerMatch?.PerformerId ?? AiFaceReferenceSuggestionIds.FromOrdinal(identity.Ordinal),
            performerMatch?.PerformerName ?? identity.DisplayName,
            performerMatch is null ? identity.ImageUrl : BuildPerformerCoverUrl(performerMatch.PerformerId, performerMatch.PerformerUpdatedAt, performerMatch.LocalPerformerHasImage) ?? identity.ImageUrl,
            confidence, why, [],
            LocalPerformerId: performerMatch?.PerformerId,
            ExternalUrl: BuildReferenceProfileUrl(manifest.SourceEndpoint, identity.ExternalId),
            LocalPerformerHasImage: performerMatch?.LocalPerformerHasImage ?? false,
            LocalPerformerIsLocalOnly: performerMatch?.LocalPerformerIsLocalOnly ?? false);
    }

    private static float ComputeConfidence(float topSimilarity, double meanSimilarity, int uniqueFaceCount, int observationCount)
    {
        var faceCountWeight = Math.Min(1.0, uniqueFaceCount / 3.0);
        var observationWeight = Math.Min(1.0, observationCount / 6.0);
        return MathF.Round(Math.Clamp((topSimilarity * 0.5f) + ((float)meanSimilarity * 0.35f) + ((float)faceCountWeight * 0.1f) + ((float)observationWeight * 0.05f), 0f, 1f) * 100f, 1);
    }

    private static float ComputeReferenceConfidence(float topSimilarity, double meanSimilarity, int observationCount)
    {
        var observationWeight = Math.Min(1.0, observationCount / 4.0);
        return MathF.Round(Math.Clamp((topSimilarity * 0.65f) + ((float)meanSimilarity * 0.3f) + ((float)observationWeight * 0.05f), 0f, 1f) * 100f, 1);
    }

    private static string BuildWhy(int uniqueFaceCount, int observationCount, float topSimilarity, double meanSimilarity)
    {
        var faceSummary = uniqueFaceCount == 1 ? "1 linked face cluster" : $"{uniqueFaceCount} linked face clusters";
        var observationSummary = observationCount == 1 ? "1 supporting match" : $"{observationCount} supporting matches";
        return $"{faceSummary} agree across {observationSummary}; best {Math.Round(topSimilarity * 100)}%, mean {Math.Round(meanSimilarity * 100)}%.";
    }

    private static string? BuildReferenceProfileUrl(string? sourceEndpoint, string externalId)
    {
        if (string.IsNullOrWhiteSpace(sourceEndpoint) || string.IsNullOrWhiteSpace(externalId)) return null;
        var baseUrl = sourceEndpoint.Trim().TrimEnd('/');
        if (baseUrl.EndsWith("/graphql", StringComparison.OrdinalIgnoreCase)) baseUrl = baseUrl[..^"/graphql".Length];
        return $"{baseUrl}/performers/{Uri.EscapeDataString(externalId)}";
    }

    private static string NormalizeName(string? value) => string.IsNullOrWhiteSpace(value) ? string.Empty : value.Trim().ToUpperInvariant();
    private static float ClampSimilarity(float similarity) => Math.Clamp(similarity, 0f, 1f);

    private static string? BuildPerformerCoverUrl(int performerId, DateTime? updatedAt, bool hasImage)
        => hasImage && updatedAt.HasValue ? $"/api/performers/{performerId}/image?max=640&v={Uri.EscapeDataString(updatedAt.Value.ToString("o"))}" : null;

    private static string? BuildThumbnailUrl(Detection? detection)
        => detection is null ? null : $"/api/stream/detection/{detection.Id}/crop?max=320";

    private static string? BuildFaceCoverUrl(int faceId, DateTime? updatedAt, bool hasCoverImage)
        => hasCoverImage && updatedAt.HasValue ? $"/api/faces/{faceId}/image?max=640&v={Uri.EscapeDataString(updatedAt.Value.ToString("o"))}" : null;

    private static IEnumerable<RawReferenceMatch> FindNearestReferenceMatches(float[] sourceVector, SaieReferencePack pack, int candidateCount)
    {
        if (candidateCount <= 0) return [];
        var sourceNorm = ComputeNorm(sourceVector);
        if (sourceNorm <= 0f) return [];
        var best = new List<RawReferenceMatch>(candidateCount);
        for (var ordinal = 0; ordinal < pack.Identities.Count; ordinal++)
        {
            var similarity = ComputeCosineSimilarity(sourceVector, sourceNorm, pack.GetCentroid(ordinal), pack.GetCentroidNorm(ordinal));
            if (similarity <= 0f) continue;
            var match = new RawReferenceMatch(pack.Identities[ordinal], similarity);
            if (best.Count < candidateCount) { best.Add(match); best.Sort(static (l, r) => l.Similarity.CompareTo(r.Similarity)); continue; }
            if (similarity <= best[0].Similarity) continue;
            best[0] = match;
            best.Sort(static (l, r) => l.Similarity.CompareTo(r.Similarity));
        }
        return best.OrderByDescending(static m => m.Similarity).ToArray();
    }

    private static float ComputeCosineSimilarity(float[] left, float leftNorm, ReadOnlySpan<float> right, float rightNorm)
    {
        if (left.Length != right.Length || leftNorm <= 0f || rightNorm <= 0f) return 0f;
        var dot = 0f;
        for (var i = 0; i < left.Length; i++) dot += left[i] * right[i];
        return ClampSimilarity(dot / (leftNorm * rightNorm));
    }

    private static float ComputeNorm(float[] vector)
    {
        var sum = 0f;
        for (var i = 0; i < vector.Length; i++) sum += vector[i] * vector[i];
        return MathF.Sqrt(sum);
    }

    private static string GetSuggestionKey(FaceSuggestionDto suggestion) => $"performer:{suggestion.PerformerId}";

    private sealed record RawFaceMatch(int FaceId, float Similarity);
    private sealed record RawReferenceMatch(SaieReferenceIdentity Identity, float Similarity);
    private sealed record CandidateFaceMatch(int FaceId, int PerformerId, string PerformerName, DateTime? PerformerUpdatedAt, bool PerformerHasImage, bool PerformerIsLocalOnly, DateTime? FaceUpdatedAt, bool FaceHasCoverImage, float Similarity, Detection? Detection);
    private sealed record HostPerformerEvidence(int PerformerId, string PerformerName, DateTime? PerformerUpdatedAt, bool PerformerHasImage, bool PerformerIsLocalOnly, string HostKey);
}
