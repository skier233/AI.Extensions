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
    // When two or more distinct reference identities each clear this confidence for the same face we
    // treat them as conflicting and damp them so neither auto-links; the user decides.
    private const float ReferenceConflictConfidence = 55f;
    private const float ReferenceConflictDampening = 0.85f;

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

        var suggestionsByFaceId = new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();
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
            suggestionsByFaceId[faceId] = MergeAndRankSuggestions(localSuggestions, referenceSuggestions ?? []);
        }

        return await ApplyHostEvidenceBoostForBatchAsync(eligibleFaceIds, suggestionsByFaceId, maxResults, cancellationToken);
    }

    private async Task<IReadOnlyDictionary<int, IReadOnlyList<FaceSuggestionDto>>> ApplyHostEvidenceBoostForBatchAsync(
        IReadOnlyCollection<int> faceIds,
        IReadOnlyDictionary<int, IReadOnlyList<FaceSuggestionDto>> suggestionsByFaceId,
        int maxResults,
        CancellationToken cancellationToken)
    {
        var distinctFaceIds = faceIds.Where(static id => id > 0).Distinct().ToArray();
        if (distinctFaceIds.Length == 0)
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        var videoAppearances = await faceRepository.FindAppearancesAsync(new FaceAppearanceFilter
        {
            FaceIds = distinctFaceIds,
            HostType = FaceAppearanceHostType.Video,
        }, cancellationToken);
        var videoIdsByFaceId = videoAppearances
            .GroupBy(static a => a.FaceId)
            .ToDictionary(static g => g.Key, static g => g.Select(static a => a.HostId).Distinct().ToList());

        var facesMissingVideoAppearances = distinctFaceIds.Where(faceId => !videoIdsByFaceId.ContainsKey(faceId)).ToArray();
        if (facesMissingVideoAppearances.Length > 0)
        {
            var videoDetections = await detectionRepository.FindAsync(new DetectionFilter
            {
                RefKind = "face",
                RefIds = facesMissingVideoAppearances.Select(static id => (long)id).ToArray(),
                HostType = DetectionHostType.Video,
            }, cancellationToken);
            foreach (var group in videoDetections.GroupBy(static d => (int)d.RefId!.Value))
                videoIdsByFaceId[group.Key] = group.Select(static d => d.HostId).Distinct().ToList();
        }

        var imageAppearances = await faceRepository.FindAppearancesAsync(new FaceAppearanceFilter
        {
            FaceIds = distinctFaceIds,
            HostType = FaceAppearanceHostType.Image,
        }, cancellationToken);
        var imageIdsByFaceId = imageAppearances
            .GroupBy(static a => a.FaceId)
            .ToDictionary(static g => g.Key, static g => g.Select(static a => a.HostId).Distinct().ToList());

        var facesMissingImageAppearances = distinctFaceIds.Where(faceId => !imageIdsByFaceId.ContainsKey(faceId)).ToArray();
        if (facesMissingImageAppearances.Length > 0)
        {
            var imageDetections = await detectionRepository.FindAsync(new DetectionFilter
            {
                RefKind = "face",
                RefIds = facesMissingImageAppearances.Select(static id => (long)id).ToArray(),
                HostType = DetectionHostType.Image,
            }, cancellationToken);
            foreach (var group in imageDetections.GroupBy(static d => (int)d.RefId!.Value))
                imageIdsByFaceId[group.Key] = group.Select(static d => d.HostId).Distinct().ToList();
        }

        var allVideoIds = videoIdsByFaceId.Values.SelectMany(static ids => ids).Distinct().ToArray();
        var allImageIds = imageIdsByFaceId.Values.SelectMany(static ids => ids).Distinct().ToArray();

        var videoPerformers = allVideoIds.Length > 0
            ? await videoRepository.GetVideoPerformersAsync(allVideoIds, cancellationToken)
            : [];
        var imagePerformers = allImageIds.Length > 0
            ? await imageRepository.GetImagePerformersAsync(allImageIds, cancellationToken)
            : [];

        var videoPerformerEvidenceByHostId = videoPerformers
            .Where(static vp => vp.Performer != null)
            .Select(vp => new HostPerformerEvidence(
                vp.PerformerId, vp.Performer!.Name, vp.Performer.UpdatedAt,
                !string.IsNullOrWhiteSpace(vp.Performer.ImageBlobId),
                vp.Performer.RemoteIds.Count == 0,
                $"video:{vp.VideoId}"))
            .GroupBy(static evidence => int.Parse(evidence.HostKey["video:".Length..]))
            .ToDictionary(static g => g.Key, static g => g.ToArray());
        var imagePerformerEvidenceByHostId = imagePerformers
            .Where(static ip => ip.Performer != null)
            .Select(ip => new HostPerformerEvidence(
                ip.PerformerId, ip.Performer!.Name, ip.Performer.UpdatedAt,
                !string.IsNullOrWhiteSpace(ip.Performer.ImageBlobId),
                ip.Performer.RemoteIds.Count == 0,
                $"image:{ip.ImageId}"))
            .GroupBy(static evidence => int.Parse(evidence.HostKey["image:".Length..]))
            .ToDictionary(static g => g.Key, static g => g.ToArray());

        var boostedByFaceId = new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();
        foreach (var faceId in distinctFaceIds)
        {
            var videoIds = videoIdsByFaceId.GetValueOrDefault(faceId) ?? [];
            var imageIds = imageIdsByFaceId.GetValueOrDefault(faceId) ?? [];
            var hostKeys = videoIds.Select(static id => $"video:{id}")
                .Concat(imageIds.Select(static id => $"image:{id}"))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            var hostPerformers = videoIds
                .SelectMany(id => videoPerformerEvidenceByHostId.GetValueOrDefault(id) ?? [])
                .Concat(imageIds.SelectMany(id => imagePerformerEvidenceByHostId.GetValueOrDefault(id) ?? []))
                .ToArray();
            suggestionsByFaceId.TryGetValue(faceId, out var suggestions);
            var boosted = ApplyHostEvidenceBoost(suggestions ?? [], hostPerformers, hostKeys, maxResults);
            if (boosted.Count > 0)
                boostedByFaceId[faceId] = boosted;
        }

        return boostedByFaceId;
    }

    private static IReadOnlyList<FaceSuggestionDto> ApplyHostEvidenceBoost(
        IReadOnlyList<FaceSuggestionDto> suggestions,
        IReadOnlyList<HostPerformerEvidence> hostPerformers,
        IReadOnlyCollection<string> hostKeys,
        int maxResults)
    {
        if (hostKeys.Count == 0)
            return suggestions.Take(maxResults).ToList();

        if (hostPerformers.Count == 0)
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
        var exclusivePerformerId = performerHostKeys.Count == 1 && performerHostKeys.Values.Single().Length == hostKeys.Count
            ? performerHostKeys.Keys.Single() : (int?)null;

        var boostedSuggestions = suggestions
            .Select(suggestion => ApplyVideoEvidenceBoost(suggestion, performerHostKeys, normalizedNameHostKeys, solePerformerHostKeys, hostKeys.Count, exclusivePerformerId))
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
                hostKeys.Count,
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
        var packs = await referencePackStore.GetActivePacksAsync(cancellationToken);
        if (packs.Count == 0)
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        var rejectedByFaceId = await referenceSuggestionDecisionStore.GetRejectedAsync(sourceEmbeddingsByFaceId.Keys.ToArray(), cancellationToken);

        // Flatten the work into independent (face, pack, source vector) scans. Each scan is a brute-force
        // KNN over the whole pack and is by far the dominant cost on this path (one full pass over a large
        // reference pack per source embedding), so scoring them is what we parallelize below. Every scan
        // only reads immutable pack data and writes its own result slot, so no synchronization is needed.
        var scoreQueries = new List<ReferenceScoreQuery>();
        foreach (var (faceId, sourceEmbeddings) in sourceEmbeddingsByFaceId)
        {
            var sourceVectors = sourceEmbeddings.Select(static e => e.Vector.ToArray()).ToArray();
            for (var packIndex = 0; packIndex < packs.Count; packIndex++)
            {
                var pack = packs[packIndex];
                if (pack.Identities.Count == 0)
                    continue;

                foreach (var sourceVector in sourceVectors.Where(v => v.Length == pack.Manifest.EmbeddingDim))
                    scoreQueries.Add(new ReferenceScoreQuery(faceId, packIndex, pack, sourceVector));
            }
        }

        // Score each face's source embeddings against every active pack, tagging matches with the pack
        // (by its stable index) they came from. The scans run across all cores; results are reassembled
        // in the original (face, pack, source-vector) order so output ordering stays deterministic.
        var perQueryMatches = new List<RawReferenceMatch>[scoreQueries.Count];
        Parallel.For(
            0,
            scoreQueries.Count,
            new ParallelOptions { CancellationToken = cancellationToken },
            i =>
            {
                var query = scoreQueries[i];
                rejectedByFaceId.TryGetValue(query.FaceId, out var rejectedIdentityIds);
                perQueryMatches[i] = FindNearestReferenceMatches(query.SourceVector, query.Pack, ReferenceCandidateK)
                    .Where(match => rejectedIdentityIds is null || !rejectedIdentityIds.Contains(match.Identity.ExternalId))
                    .Select(match => match with { PackIndex = query.PackIndex, Pack = query.Pack })
                    .ToList();
            });

        var matchesByFaceId = new Dictionary<int, List<RawReferenceMatch>>();
        for (var i = 0; i < scoreQueries.Count; i++)
        {
            if (perQueryMatches[i].Count == 0)
                continue;

            var faceId = scoreQueries[i].FaceId;
            if (!matchesByFaceId.TryGetValue(faceId, out var faceMatches)) { faceMatches = []; matchesByFaceId[faceId] = faceMatches; }
            faceMatches.AddRange(perQueryMatches[i]);
        }

        if (matchesByFaceId.Count == 0)
            return new Dictionary<int, IReadOnlyList<FaceSuggestionDto>>();

        // Resolve reference identities to local performers per pack, because each pack targets its own
        // site/endpoint and the same external id means different people on different sites.
        var performerMatchesByPack = new Dictionary<int, IReadOnlyDictionary<string, AiFaceReferencePerformerMatch>>();
        var allMatches = matchesByFaceId.Values.SelectMany(static m => m).ToArray();
        for (var packIndex = 0; packIndex < packs.Count; packIndex++)
        {
            var index = packIndex;
            var identities = allMatches.Where(m => m.PackIndex == index)
                .Select(static m => m.Identity)
                .DistinctBy(static i => i.ExternalId, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            performerMatchesByPack[index] = identities.Length == 0
                ? new Dictionary<string, AiFaceReferencePerformerMatch>(StringComparer.OrdinalIgnoreCase)
                : await referencePerformerResolver.ResolveAsync(identities, packs[index].Manifest.SourceEndpoint, cancellationToken);
        }

        return matchesByFaceId.ToDictionary(
            static pair => pair.Key,
            pair => BuildReferenceSuggestionsForFace(pair.Value, performerMatchesByPack, maxResults));
    }

    private static IReadOnlyList<FaceSuggestionDto> BuildReferenceSuggestionsForFace(
        IReadOnlyList<RawReferenceMatch> faceMatches,
        IReadOnlyDictionary<int, IReadOnlyDictionary<string, AiFaceReferencePerformerMatch>> performerMatchesByPack,
        int maxResults)
    {
        // Group matches that point at the same performer. A resolved local performer groups by its id;
        // an unresolved identity groups by site endpoint + external id so the same site identity seen in
        // two packs merges (strong corroborating evidence) while genuinely different people stay apart.
        var groups = faceMatches
            .Select(match =>
            {
                AiFaceReferencePerformerMatch? performer = null;
                if (performerMatchesByPack.TryGetValue(match.PackIndex, out var perfMatches))
                    perfMatches.TryGetValue(match.Identity.ExternalId, out performer);
                var key = performer is not null
                    ? $"perf:{performer.PerformerId}"
                    : $"ref:{NormalizeName(match.Pack?.Manifest.SourceEndpoint)}|{match.Identity.ExternalId.ToUpperInvariant()}";
                return (Key: key, Match: match, Performer: performer);
            })
            .GroupBy(static item => item.Key, StringComparer.OrdinalIgnoreCase)
            .ToArray();

        var suggestions = groups
            .Select(group => BuildReferenceSuggestion(
                group.Select(static item => item.Match).ToArray(),
                group.Select(static item => item.Performer).FirstOrDefault(static p => p is not null),
                group.Select(static item => item.Match.PackIndex).Distinct().Count()))
            .ToList();

        // Conflict: more than one distinct strong reference identity competes for this face. Damp each
        // so none auto-links, and surface them all for the user to choose between.
        if (suggestions.Count(static s => s.Confidence >= ReferenceConflictConfidence) > 1)
        {
            // One stable id shared by every competing match for this face, so the UI can group them
            // into a single "possible duplicate" choice instead of showing them as rival suggestions.
            // The conflict itself is surfaced through ConflictGroupId (the UI groups the competing matches
            // and offers the use-one / merge choice), so the evidence text stays clean — no instructional
            // "or merge them" blurb that would just duplicate that UI.
            var conflictGroupId = Guid.NewGuid().ToString("N");
            suggestions = suggestions
                .Select(s => s.Confidence >= ReferenceConflictConfidence
                    ? s with
                    {
                        Confidence = MathF.Round(s.Confidence * ReferenceConflictDampening, 1),
                        ConflictGroupId = conflictGroupId,
                    }
                    : s)
                .ToList();
        }

        return suggestions
            .OrderByDescending(static s => s.Confidence).ThenBy(static s => s.PerformerName)
            .Take(maxResults).ToList();
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

        var hostSummary = matchedHostKeys.Length == 1 ? "already tagged on 1 video or image with this face" : $"already tagged on {matchedHostKeys.Length} videos or images with this face";
        var fullCoverageSummary = fullCoverage ? "; tagged on every video and image with this face" : string.Empty;
        var soleSummary = soleHostKeys.Length > 0 ? (soleHostKeys.Length == 1 ? "; the only performer on 1 of them" : $"; the only performer on {soleHostKeys.Length} of them") : string.Empty;

        return suggestion with
        {
            Confidence = MathF.Min(100f, suggestion.Confidence + boost),
            Why = $"{suggestion.Why}; also {hostSummary}{fullCoverageSummary}{soleSummary}",
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
        var hostSummary = matchedHostCount == 1 ? "1 video or image" : $"{matchedHostCount} videos or images";
        var coverageSummary = totalHostCount > matchedHostCount ? $" of the {totalHostCount} that contain this face" : totalHostCount == 1 ? string.Empty : " that contain this face";
        var why = $"Already tagged on {hostSummary}{coverageSummary} where this face appears";
        if (soleHostCount > 0) why += soleHostCount == 1 ? "; the only performer on 1 of them" : $"; the only performer on {soleHostCount} of them";
        if (exclusive) why += "; every video and image with this face points to this performer";

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

    private static FaceSuggestionDto BuildReferenceSuggestion(IReadOnlyList<RawReferenceMatch> matches, AiFaceReferencePerformerMatch? performerMatch, int supportingPackCount)
    {
        var ordered = matches.OrderByDescending(static m => m.Similarity).ToArray();
        var top = ordered[0];
        var identity = top.Identity;
        var manifest = top.Pack!.Manifest;
        var topSimilarity = ordered[0].Similarity;
        var meanSimilarity = ordered.Take(Math.Min(3, ordered.Length)).Average(static m => m.Similarity);
        var confidence = ComputeReferenceConfidence(topSimilarity, meanSimilarity, ordered.Length);
        // Independent packs from different sites agreeing on the same performer is the strongest signal
        // available, so boost when more than one pack corroborates.
        if (supportingPackCount > 1)
            confidence = MathF.Min(100f, confidence + MathF.Min(12f, (supportingPackCount - 1) * 8f));
        confidence = MathF.Round(confidence, 1);

        var photoSummary = ordered.Length == 1 ? "1 reference photo" : $"{ordered.Length} reference photos";
        var sourceDisplay = BuildSourceDisplay(manifest.SourceEndpoint);
        var packSummary = supportingPackCount > 1 ? $"; confirmed by {supportingPackCount} reference sources" : string.Empty;
        var why = performerMatch is null
            ? $"Found in {sourceDisplay}; {Math.Round(topSimilarity * 100)}% best and {Math.Round(meanSimilarity * 100)}% average visual similarity across {photoSummary}{packSummary}"
            : $"Matches an existing performer from {sourceDisplay}; {Math.Round(topSimilarity * 100)}% best and {Math.Round(meanSimilarity * 100)}% average visual similarity across {photoSummary}{packSummary}";
        return new FaceSuggestionDto(
            performerMatch?.PerformerId ?? AiFaceReferenceSuggestionIds.FromIdentity(top.PackIndex, identity.Ordinal),
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
        var faceSummary = uniqueFaceCount == 1 ? "1 of this performer's saved faces" : $"{uniqueFaceCount} of this performer's saved faces";
        var observationSummary = observationCount == 1 ? "1 comparison" : $"{observationCount} comparisons";
        return $"Looks like {faceSummary}; {Math.Round(topSimilarity * 100)}% best and {Math.Round(meanSimilarity * 100)}% average visual similarity across {observationSummary}";
    }

    // A short, human-readable form of the reference source's URL for evidence text — e.g.
    // "https://stashdb.org" rather than the opaque internal pack id. Strips the GraphQL suffix so the
    // displayed value is the site a user would recognise.
    private static string BuildSourceDisplay(string? sourceEndpoint)
    {
        if (string.IsNullOrWhiteSpace(sourceEndpoint)) return "a reference database";
        var url = sourceEndpoint.Trim().TrimEnd('/');
        if (url.EndsWith("/graphql", StringComparison.OrdinalIgnoreCase)) url = url[..^"/graphql".Length];
        return url;
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
        var sourceNorm = SaieReferencePack.Norm(sourceVector);
        if (sourceNorm <= 0f) return [];
        var best = new List<RawReferenceMatch>(candidateCount);
        for (var ordinal = 0; ordinal < pack.Identities.Count; ordinal++)
        {
            var similarity = pack.CosineSimilarityTo(ordinal, sourceVector, sourceNorm);
            if (similarity <= 0f) continue;
            var match = new RawReferenceMatch(pack.Identities[ordinal], similarity);
            if (best.Count < candidateCount) { best.Add(match); best.Sort(static (l, r) => l.Similarity.CompareTo(r.Similarity)); continue; }
            if (similarity <= best[0].Similarity) continue;
            best[0] = match;
            best.Sort(static (l, r) => l.Similarity.CompareTo(r.Similarity));
        }
        return best.OrderByDescending(static m => m.Similarity).ToArray();
    }

    private static string GetSuggestionKey(FaceSuggestionDto suggestion) => $"performer:{suggestion.PerformerId}";

    private sealed record ReferenceScoreQuery(int FaceId, int PackIndex, SaieReferencePack Pack, float[] SourceVector);

    private sealed record RawFaceMatch(int FaceId, float Similarity);
    private sealed record RawReferenceMatch(SaieReferenceIdentity Identity, float Similarity, int PackIndex = 0, SaieReferencePack? Pack = null);
    private sealed record CandidateFaceMatch(int FaceId, int PerformerId, string PerformerName, DateTime? PerformerUpdatedAt, bool PerformerHasImage, bool PerformerIsLocalOnly, DateTime? FaceUpdatedAt, bool FaceHasCoverImage, float Similarity, Detection? Detection);
    private sealed record HostPerformerEvidence(int PerformerId, string PerformerName, DateTime? PerformerUpdatedAt, bool PerformerHasImage, bool PerformerIsLocalOnly, string HostKey);
}
