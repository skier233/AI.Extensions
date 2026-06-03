using System.Text.Json;

using Cove.Core.Auth;
using Cove.Core.DTOs;
using Cove.Core.Entities;
using Cove.Core.Enums;
using Cove.Core.Interfaces;

using Microsoft.EntityFrameworkCore;
using Pgvector;

namespace AI.Visual;

internal sealed class AiVisualSemanticSearchRequest<TFilter>
    where TFilter : class
{
    public FindFilter? FindFilter { get; init; }

    public TFilter? ObjectFilter { get; init; }
}

internal sealed record AiVisualSimilarVideoDto(
    VideoDto Video,
    float Distance,
    string Kind,
    string? KindFamily,
    bool IsSemantic,
    int SectionIndex,
    double? StartSec,
    double? EndSec);

internal sealed record AiVisualSimilarImageDto(
    ImageDto Image,
    float Distance,
    string Kind,
    string? KindFamily,
    bool IsSemantic,
    int SectionIndex,
    double? StartSec,
    double? EndSec);

internal sealed record AiVisualSimilarPage(int Page, int PerPage);

internal sealed record AiVisualSegmentInterval(double StartSec, double? EndSec);

internal sealed record AiVisualQueryEmbedding(
    Vector Vector,
    string Kind,
    string? KindFamily,
    EmbeddingModality Modality,
    bool IsSemantic,
    double? StartSec,
    double? EndSec);

internal sealed record AiVisualSimilarQuery(
    AiVisualQueryEmbedding Query,
    TargetEmbeddingScope TargetScope,
    double Weight,
    SimilarQueryRole Role);

internal enum SimilarQueryRole
{
    OverallVisual,
    OverallSemantic,
    PartVisual,
    PartSemantic,
}

internal enum TargetEmbeddingScope
{
    Any,
    Asset,
    Section,
}

internal sealed class AiVisualSemanticSearchService(
    IEmbeddingRepository embeddingRepository,
    IEmbeddingService embeddingService,
    ITextEncoderRegistry textEncoderRegistry,
    IVideoRepository videoRepository,
    IImageRepository imageRepository,
    AiVisualLocalTextEncoder localTextEncoder,
    IUserEngagementService? engagementService = null,
    ICurrentPrincipalAccessor? principalAccessor = null)
{
    private const string VisualSourceKey = "ext:ai.visual";
    private const string VisualFeatureKind = "visual.feature.v1";
    private const string FeatureKindFamily = "feature.v1";
    private const string VisualSemanticKind = "visual.semantic.v1";
    private const string SemanticKindFamily = "semantic.v1";
    private const string VisualMatchSort = "visual_match";
    private const float VisualMatchDistanceWindow = 0.03f;
    private const int MinimumVisualMatchResults = 5;
    private const int MaxVisualMatchResults = 500;
    private const int DefaultSimilarPerPage = 12;
    private const int MaxSimilarPerPage = 48;
    private const int MaxSimilarResults = 500;
    private const int MaxFilteredIds = 100_000;
    private const int FastTextEncodeTimeoutMilliseconds = 400;
    private const int LocalTextSeedLimit = 48;
    private const int LocalTextSeedScanLimit = 500;
    private const int MaxPartSimilarityQueries = 4;
    private const int MaxSegmentSimilarityIntervals = 64;
    private const double OverallVisualWeight = 1.0;
    private const double OverallSemanticWeight = 0.42;
    private const double PartVisualWeight = 0.7;
    private const double PartSemanticWeight = 0.25;

    private readonly IEmbeddingRepository _embeddingRepository = embeddingRepository;
    private readonly IEmbeddingService _embeddingService = embeddingService;
    private readonly ITextEncoderRegistry _textEncoderRegistry = textEncoderRegistry;
    private readonly IVideoRepository _videoRepository = videoRepository;
    private readonly IImageRepository _imageRepository = imageRepository;
    private readonly AiVisualLocalTextEncoder _localTextEncoder = localTextEncoder;
    private readonly IUserEngagementService? _engagementService = engagementService;
    private readonly ICurrentPrincipalAccessor? _principalAccessor = principalAccessor;

    private bool CanReadFiles => _principalAccessor?.Current?.Has(Permissions.FilesRead) == true;

    private bool HasUserScopedEngagement => _principalAccessor?.Current?.UserId != null;

    public async Task<PaginatedResponse<VideoDto>> SearchVideosAsync(AiVisualSemanticSearchRequest<VideoFilter> request, CancellationToken ct = default)
    {
        var findFilter = NormalizeFindFilter(request.FindFilter);
        var allowedIds = await ResolveAllowedVideoIdsAsync(request.ObjectFilter, ct);
        var ranked = await SearchRankedMatchesAsync(findFilter.Q, EmbeddingHostType.Video, allowedIds, ct);
        var sorted = await SortVideoMatchesAsync(ranked, findFilter, ct);
        var pageMatches = Page(sorted, findFilter);
        var items = await BuildVideoResultsAsync(pageMatches, ct);

        return new PaginatedResponse<VideoDto>(items, sorted.Count, findFilter.Page, findFilter.PerPage);
    }

    public async Task<PaginatedResponse<ImageDto>> SearchImagesAsync(AiVisualSemanticSearchRequest<ImageFilter> request, CancellationToken ct = default)
    {
        var findFilter = NormalizeFindFilter(request.FindFilter);
        var allowedIds = await ResolveAllowedImageIdsAsync(request.ObjectFilter, ct);
        var ranked = await SearchRankedMatchesAsync(findFilter.Q, EmbeddingHostType.Image, allowedIds, ct);
        var sorted = await SortImageMatchesAsync(ranked, findFilter, ct);
        var pageMatches = Page(sorted, findFilter);
        var items = await BuildImageResultsAsync(pageMatches, ct);

        return new PaginatedResponse<ImageDto>(items, sorted.Count, findFilter.Page, findFilter.PerPage);
    }

    public async Task<PaginatedResponse<AiVisualSimilarVideoDto>> SimilarVideosForVideoAsync(int videoId, int page = 1, int perPage = DefaultSimilarPerPage, CancellationToken ct = default)
    {
        var pageInfo = NormalizeSimilarPage(page, perPage);
        var queries = await LoadAssetSimilarityQueriesAsync(EmbeddingHostType.Video, videoId, TargetEmbeddingScope.Asset, includeSourceSections: true, ct);
        if (queries.Count == 0)
        {
            return EmptySimilarVideoResponse(pageInfo);
        }

        var ranked = await SearchSimilarAsync(queries, EmbeddingHostType.Video, excludeHostId: videoId, ct);
        return await BuildSimilarVideoResponseAsync(ranked, pageInfo, ct);
    }

    public async Task<PaginatedResponse<AiVisualSimilarImageDto>> SimilarImagesForVideoAsync(int videoId, int page = 1, int perPage = DefaultSimilarPerPage, CancellationToken ct = default)
    {
        var pageInfo = NormalizeSimilarPage(page, perPage);
        var queries = await LoadAssetSimilarityQueriesAsync(EmbeddingHostType.Video, videoId, TargetEmbeddingScope.Asset, includeSourceSections: false, ct);
        if (queries.Count == 0)
        {
            return EmptySimilarImageResponse(pageInfo);
        }

        var ranked = await SearchSimilarAsync(queries, EmbeddingHostType.Image, excludeHostId: null, ct);
        return await BuildSimilarImageResponseAsync(ranked, pageInfo, ct);
    }

    public async Task<PaginatedResponse<AiVisualSimilarVideoDto>> SimilarVideosForImageAsync(int imageId, int page = 1, int perPage = DefaultSimilarPerPage, CancellationToken ct = default)
    {
        var pageInfo = NormalizeSimilarPage(page, perPage);
        var queries = await LoadAssetSimilarityQueriesAsync(EmbeddingHostType.Image, imageId, TargetEmbeddingScope.Any, includeSourceSections: false, ct);
        if (queries.Count == 0)
        {
            return EmptySimilarVideoResponse(pageInfo);
        }

        var ranked = await SearchSimilarAsync(queries, EmbeddingHostType.Video, excludeHostId: null, ct);
        return await BuildSimilarVideoResponseAsync(ranked, pageInfo, ct);
    }

    public async Task<PaginatedResponse<AiVisualSimilarImageDto>> SimilarImagesForImageAsync(int imageId, int page = 1, int perPage = DefaultSimilarPerPage, CancellationToken ct = default)
    {
        var pageInfo = NormalizeSimilarPage(page, perPage);
        var queries = await LoadAssetSimilarityQueriesAsync(EmbeddingHostType.Image, imageId, TargetEmbeddingScope.Asset, includeSourceSections: false, ct);
        if (queries.Count == 0)
        {
            return EmptySimilarImageResponse(pageInfo);
        }

        var ranked = await SearchSimilarAsync(queries, EmbeddingHostType.Image, excludeHostId: imageId, ct);
        return await BuildSimilarImageResponseAsync(ranked, pageInfo, ct);
    }

    public async Task<PaginatedResponse<AiVisualSimilarVideoDto>> SimilarVideosForVideoSegmentAsync(
        int videoId,
        double startSec,
        double? endSec = null,
        int page = 1,
        int perPage = DefaultSimilarPerPage,
        CancellationToken ct = default)
    {
        if (!double.IsFinite(startSec) || startSec < 0)
        {
            throw new ArgumentException("startSec must be a non-negative finite value.", nameof(startSec));
        }

        if (endSec.HasValue && (!double.IsFinite(endSec.Value) || endSec.Value < startSec))
        {
            throw new ArgumentException("endSec must be greater than or equal to startSec.", nameof(endSec));
        }

        var pageInfo = NormalizeSimilarPage(page, perPage);
        return await SimilarVideosForVideoSegmentAsync(videoId, [new AiVisualSegmentInterval(startSec, endSec)], pageInfo.Page, pageInfo.PerPage, ct);
    }

    public async Task<PaginatedResponse<AiVisualSimilarVideoDto>> SimilarVideosForVideoSegmentAsync(
        int videoId,
        IReadOnlyList<AiVisualSegmentInterval> intervals,
        int page = 1,
        int perPage = DefaultSimilarPerPage,
        CancellationToken ct = default)
    {
        var pageInfo = NormalizeSimilarPage(page, perPage);
        var queries = await LoadSegmentSimilarityQueriesAsync(videoId, NormalizeSegmentIntervals(intervals), ct);
        if (queries.Count == 0)
        {
            return EmptySimilarVideoResponse(pageInfo);
        }

        var ranked = await SearchSimilarAsync(queries, EmbeddingHostType.Video, excludeHostId: videoId, ct);
        return await BuildSimilarVideoResponseAsync(ranked, pageInfo, ct);
    }

    private async Task<IReadOnlyList<AiVisualSimilarQuery>> LoadAssetSimilarityQueriesAsync(
        EmbeddingHostType hostType,
        int hostId,
        TargetEmbeddingScope overallTargetScope,
        bool includeSourceSections,
        CancellationToken ct)
    {
        var queries = new List<AiVisualSimilarQuery>();
        var feature = await LoadEmbeddingAsync(hostType, hostId, VisualFeatureKind, FeatureKindFamily, isSemantic: false, sectionIndex: 0, ct);
        if (feature is not null)
        {
            queries.Add(new AiVisualSimilarQuery(ToQueryEmbedding(feature), overallTargetScope, OverallVisualWeight, SimilarQueryRole.OverallVisual));
        }

        var semantic = await LoadEmbeddingAsync(hostType, hostId, VisualSemanticKind, SemanticKindFamily, isSemantic: true, sectionIndex: 0, ct);
        if (semantic is not null)
        {
            queries.Add(new AiVisualSimilarQuery(ToQueryEmbedding(semantic), overallTargetScope, OverallSemanticWeight, SimilarQueryRole.OverallSemantic));
        }

        if (includeSourceSections && hostType == EmbeddingHostType.Video)
        {
            foreach (var section in await LoadRepresentativeSectionEmbeddingsAsync(hostId, VisualFeatureKind, FeatureKindFamily, isSemantic: false, ct))
            {
                queries.Add(new AiVisualSimilarQuery(ToQueryEmbedding(section), TargetEmbeddingScope.Section, PartVisualWeight, SimilarQueryRole.PartVisual));
            }

            foreach (var section in await LoadRepresentativeSectionEmbeddingsAsync(hostId, VisualSemanticKind, SemanticKindFamily, isSemantic: true, ct))
            {
                queries.Add(new AiVisualSimilarQuery(ToQueryEmbedding(section), TargetEmbeddingScope.Section, PartSemanticWeight, SimilarQueryRole.PartSemantic));
            }
        }

        return queries;
    }

    private async Task<Embedding?> LoadEmbeddingAsync(
        EmbeddingHostType hostType,
        int hostId,
        string kind,
        string kindFamily,
        bool isSemantic,
        int sectionIndex,
        CancellationToken ct)
    {
        var results = await _embeddingRepository.FindAsync(new EmbeddingFilter
        {
            HostType = hostType, HostId = hostId, SourceKey = VisualSourceKey,
            Kind = kind, KindFamily = kindFamily, Modality = EmbeddingModality.Visual,
            IsSemantic = isSemantic,
        }, ct);
        return results.Where(e => e.SectionIndex == sectionIndex).OrderByDescending(static e => e.CreatedAt).FirstOrDefault();
    }

    private async Task<IReadOnlyList<Embedding>> LoadRepresentativeSectionEmbeddingsAsync(
        int videoId,
        string kind,
        string kindFamily,
        bool isSemantic,
        CancellationToken ct)
    {
        var sections = await LoadSectionEmbeddingsAsync(videoId, kind, kindFamily, isSemantic, ct);
        return SelectRepresentativeEmbeddings(sections, MaxPartSimilarityQueries);
    }

    private async Task<IReadOnlyList<Embedding>> LoadSectionEmbeddingsAsync(
        int videoId,
        string kind,
        string kindFamily,
        bool isSemantic,
        CancellationToken ct)
    {
        var results = await _embeddingRepository.FindAsync(new EmbeddingFilter
        {
            HostType = EmbeddingHostType.Video, HostId = videoId, SourceKey = VisualSourceKey,
            Kind = kind, KindFamily = kindFamily, Modality = EmbeddingModality.Visual,
            IsSemantic = isSemantic, SectionIndexGreaterThan = 0,
        }, ct);
        return results.OrderBy(static e => e.StartSec ?? double.MaxValue).ThenBy(static e => e.SectionIndex).ToList();
    }

    private async Task<IReadOnlyList<AiVisualSimilarQuery>> LoadSegmentSimilarityQueriesAsync(
        int videoId,
        IReadOnlyList<AiVisualSegmentInterval> intervals,
        CancellationToken ct)
    {
        var queries = new List<AiVisualSimilarQuery>();

        var feature = await BuildSegmentQueryEmbeddingAsync(videoId, intervals, VisualFeatureKind, FeatureKindFamily, isSemantic: false, ct);
        if (feature is not null)
        {
            queries.Add(new AiVisualSimilarQuery(feature, TargetEmbeddingScope.Section, OverallVisualWeight, SimilarQueryRole.PartVisual));
        }

        var semantic = await BuildSegmentQueryEmbeddingAsync(videoId, intervals, VisualSemanticKind, SemanticKindFamily, isSemantic: true, ct);
        if (semantic is not null)
        {
            queries.Add(new AiVisualSimilarQuery(semantic, TargetEmbeddingScope.Section, OverallSemanticWeight, SimilarQueryRole.PartSemantic));
        }

        if (intervals.Count > 1)
        {
            foreach (var interval in SelectRepresentativeIntervals(intervals, MaxPartSimilarityQueries))
            {
                var intervalFeature = await BuildSegmentQueryEmbeddingAsync(videoId, [interval], VisualFeatureKind, FeatureKindFamily, isSemantic: false, ct);
                if (intervalFeature is not null)
                {
                    queries.Add(new AiVisualSimilarQuery(intervalFeature, TargetEmbeddingScope.Section, PartVisualWeight, SimilarQueryRole.PartVisual));
                }

                var intervalSemantic = await BuildSegmentQueryEmbeddingAsync(videoId, [interval], VisualSemanticKind, SemanticKindFamily, isSemantic: true, ct);
                if (intervalSemantic is not null)
                {
                    queries.Add(new AiVisualSimilarQuery(intervalSemantic, TargetEmbeddingScope.Section, PartSemanticWeight, SimilarQueryRole.PartSemantic));
                }
            }
        }

        return queries;
    }

    private async Task<AiVisualQueryEmbedding?> BuildSegmentQueryEmbeddingAsync(
        int videoId,
        IReadOnlyList<AiVisualSegmentInterval> intervals,
        string kind,
        string kindFamily,
        bool isSemantic,
        CancellationToken ct)
    {
        var sections = await LoadSectionEmbeddingsAsync(videoId, kind, kindFamily, isSemantic, ct);

        if (sections.Count == 0)
        {
            return null;
        }

        var weighted = sections
            .Select(section => (Embedding: section, Weight: GetSectionIntervalWeight(section, intervals)))
            .Where(static item => item.Weight > 0d)
            .ToArray();

        if (weighted.Length == 0)
        {
            var midpoint = GetIntervalMidpoint(intervals[0]);
            var nearest = sections
                .OrderBy(section => GetDistanceFromSection(section, midpoint))
                .ThenBy(static section => section.SectionIndex)
                .First();
            return ToQueryEmbedding(nearest);
        }

        if (weighted.Length == 1)
        {
            return ToQueryEmbedding(weighted[0].Embedding);
        }

        return BuildWeightedQueryEmbedding(weighted, kind, kindFamily, isSemantic);
    }

    private async Task<IReadOnlyList<EmbeddingSearchResult>> SearchSimilarAsync(
        IReadOnlyList<AiVisualSimilarQuery> queries,
        EmbeddingHostType targetHostType,
        int? excludeHostId,
        CancellationToken ct)
    {
        var candidates = new Dictionary<int, SimilarCandidate>();

        foreach (var query in queries.Where(static query => query.Weight > 0d))
        {
            var matches = await _embeddingService.KnnAsync(
                query.Query.Vector,
                int.MaxValue,
                new EmbeddingSearchOptions
                {
                    HostType = targetHostType,
                    Kind = query.Query.Kind,
                    KindFamily = query.Query.KindFamily,
                    Modality = query.Query.Modality,
                    IsSemantic = query.Query.IsSemantic,
                    SourceKey = VisualSourceKey,
                },
                ct);

            var bestPerHost = matches
                .Where(match => excludeHostId is null || match.Embedding.HostId != excludeHostId.Value)
                .Where(match => MatchesTargetScope(match.Embedding, query.TargetScope))
                .GroupBy(static match => match.Embedding.HostId)
                .Select(static group => group
                    .OrderBy(static match => match.Distance)
                    .ThenBy(static match => match.Embedding.SectionIndex)
                    .ThenBy(static match => match.Embedding.StartSec ?? double.MaxValue)
                    .First());

            foreach (var match in bestPerHost)
            {
                if (!candidates.TryGetValue(match.Embedding.HostId, out var candidate))
                {
                    candidate = new SimilarCandidate(match.Embedding.HostId);
                    candidates[match.Embedding.HostId] = candidate;
                }

                candidate.Add(match, query);
            }
        }

        return DiversifySimilarResults(candidates.Values)
            .Take(MaxSimilarResults)
            .ToArray();
    }

    private async Task<PaginatedResponse<AiVisualSimilarVideoDto>> BuildSimilarVideoResponseAsync(IReadOnlyList<EmbeddingSearchResult> ranked, AiVisualSimilarPage pageInfo, CancellationToken ct)
    {
        var pageMatches = PageSimilar(ranked, pageInfo);
        var videos = await BuildVideoResultsAsync(pageMatches, ct);
        var videoById = videos.ToDictionary(static video => video.Id);
        var items = pageMatches
            .Where(match => videoById.ContainsKey(match.Embedding.HostId))
            .Select(match => new AiVisualSimilarVideoDto(
                videoById[match.Embedding.HostId],
                match.Distance,
                match.Embedding.Kind,
                match.Embedding.KindFamily,
                match.Embedding.IsSemantic,
                match.Embedding.SectionIndex,
                match.Embedding.StartSec,
                match.Embedding.EndSec))
            .ToList();

        return new PaginatedResponse<AiVisualSimilarVideoDto>(items, ranked.Count, pageInfo.Page, pageInfo.PerPage);
    }

    private async Task<PaginatedResponse<AiVisualSimilarImageDto>> BuildSimilarImageResponseAsync(IReadOnlyList<EmbeddingSearchResult> ranked, AiVisualSimilarPage pageInfo, CancellationToken ct)
    {
        var pageMatches = PageSimilar(ranked, pageInfo);
        var images = await BuildImageResultsAsync(pageMatches, ct);
        var imageById = images.ToDictionary(static image => image.Id);
        var items = pageMatches
            .Where(match => imageById.ContainsKey(match.Embedding.HostId))
            .Select(match => new AiVisualSimilarImageDto(
                imageById[match.Embedding.HostId],
                match.Distance,
                match.Embedding.Kind,
                match.Embedding.KindFamily,
                match.Embedding.IsSemantic,
                match.Embedding.SectionIndex,
                match.Embedding.StartSec,
                match.Embedding.EndSec))
            .ToList();

        return new PaginatedResponse<AiVisualSimilarImageDto>(items, ranked.Count, pageInfo.Page, pageInfo.PerPage);
    }

    private static IReadOnlyList<EmbeddingSearchResult> PageSimilar(IReadOnlyList<EmbeddingSearchResult> ranked, AiVisualSimilarPage pageInfo)
        => ranked
            .Skip((pageInfo.Page - 1) * pageInfo.PerPage)
            .Take(pageInfo.PerPage)
            .ToArray();

    private static AiVisualQueryEmbedding ToQueryEmbedding(Embedding embedding)
        => new(
            embedding.Vector,
            embedding.Kind,
            embedding.KindFamily,
            embedding.Modality,
            embedding.IsSemantic,
            embedding.StartSec,
            embedding.EndSec);

    private static AiVisualQueryEmbedding BuildWeightedQueryEmbedding(
        IReadOnlyList<(Embedding Embedding, double Weight)> weighted,
        string kind,
        string kindFamily,
        bool isSemantic)
    {
        var first = weighted[0].Embedding;
        var values = first.Vector.ToArray();
        var sums = new float[values.Length];
        var totalWeight = 0d;
        double? startSec = null;
        double? endSec = null;

        foreach (var (embedding, weight) in weighted)
        {
            var vector = embedding.Vector.ToArray();
            if (vector.Length != sums.Length)
            {
                continue;
            }

            for (var index = 0; index < vector.Length; index++)
            {
                sums[index] += vector[index] * (float)weight;
            }

            totalWeight += weight;
            startSec = startSec is null ? embedding.StartSec : Math.Min(startSec.Value, embedding.StartSec ?? startSec.Value);
            endSec = endSec is null ? embedding.EndSec : Math.Max(endSec.Value, embedding.EndSec ?? endSec.Value);
        }

        if (totalWeight <= 0d)
        {
            return ToQueryEmbedding(first);
        }

        for (var index = 0; index < sums.Length; index++)
        {
            sums[index] /= (float)totalWeight;
        }

        return new AiVisualQueryEmbedding(new Vector(sums), kind, kindFamily, EmbeddingModality.Visual, isSemantic, startSec, endSec);
    }

    private static bool MatchesTargetScope(Embedding embedding, TargetEmbeddingScope targetScope)
        => targetScope switch
        {
            TargetEmbeddingScope.Asset => embedding.SectionIndex == 0,
            TargetEmbeddingScope.Section => embedding.SectionIndex > 0,
            _ => true,
        };

    private static bool ContainsTime(Embedding embedding, double seconds)
    {
        var start = embedding.StartSec ?? 0d;
        var end = embedding.EndSec ?? start;
        return seconds >= start && seconds <= end;
    }

    private static double GetSectionOverlap(Embedding embedding, double startSec, double endSec)
    {
        var start = embedding.StartSec ?? 0d;
        var end = embedding.EndSec ?? start;
        return Math.Max(0d, Math.Min(end, endSec) - Math.Max(start, startSec));
    }

    private static double GetDistanceFromSection(Embedding embedding, double seconds)
    {
        var start = embedding.StartSec ?? 0d;
        var end = embedding.EndSec ?? start;
        if (seconds >= start && seconds <= end)
        {
            return 0d;
        }

        return seconds < start ? start - seconds : seconds - end;
    }

    private static double GetSectionIntervalWeight(Embedding embedding, IReadOnlyList<AiVisualSegmentInterval> intervals)
    {
        var weight = 0d;
        foreach (var interval in intervals)
        {
            var start = Math.Max(0d, interval.StartSec);
            var end = interval.EndSec.HasValue ? Math.Max(start, interval.EndSec.Value) : start;
            weight += end > start
                ? GetSectionOverlap(embedding, start, end)
                : ContainsTime(embedding, start) ? 1d : 0d;
        }

        return weight;
    }

    private static double GetIntervalMidpoint(AiVisualSegmentInterval interval)
    {
        var start = Math.Max(0d, interval.StartSec);
        var end = interval.EndSec.HasValue ? Math.Max(start, interval.EndSec.Value) : start;
        return end > start ? (start + end) / 2d : start;
    }

    private static IReadOnlyList<AiVisualSegmentInterval> NormalizeSegmentIntervals(IReadOnlyList<AiVisualSegmentInterval> intervals)
    {
        if (intervals.Count == 0)
        {
            throw new ArgumentException("At least one interval is required.", nameof(intervals));
        }

        var normalized = intervals
            .Select(static interval =>
            {
                if (!double.IsFinite(interval.StartSec) || interval.StartSec < 0)
                {
                    throw new ArgumentException("Interval startSec values must be non-negative finite values.", nameof(intervals));
                }

                if (interval.EndSec.HasValue && (!double.IsFinite(interval.EndSec.Value) || interval.EndSec.Value < interval.StartSec))
                {
                    throw new ArgumentException("Interval endSec values must be greater than or equal to startSec.", nameof(intervals));
                }

                return new AiVisualSegmentInterval(interval.StartSec, interval.EndSec ?? interval.StartSec);
            })
            .OrderBy(static interval => interval.StartSec)
            .ThenBy(static interval => interval.EndSec)
            .ToArray();

        var merged = new List<AiVisualSegmentInterval>();
        foreach (var interval in normalized)
        {
            if (merged.Count == 0)
            {
                merged.Add(interval);
                continue;
            }

            var previous = merged[^1];
            if (interval.StartSec <= (previous.EndSec ?? previous.StartSec))
            {
                merged[^1] = new AiVisualSegmentInterval(previous.StartSec, Math.Max(previous.EndSec ?? previous.StartSec, interval.EndSec ?? interval.StartSec));
                continue;
            }

            merged.Add(interval);
        }

        return SelectRepresentativeIntervals(merged, MaxSegmentSimilarityIntervals);
    }

    private static IReadOnlyList<Embedding> SelectRepresentativeEmbeddings(IReadOnlyList<Embedding> embeddings, int maxCount)
    {
        if (embeddings.Count <= maxCount)
        {
            return embeddings;
        }

        return SelectRepresentativeIndices(embeddings.Count, maxCount)
            .Select(index => embeddings[index])
            .ToArray();
    }

    private static IReadOnlyList<AiVisualSegmentInterval> SelectRepresentativeIntervals(IReadOnlyList<AiVisualSegmentInterval> intervals, int maxCount)
    {
        if (intervals.Count <= maxCount)
        {
            return intervals;
        }

        return SelectRepresentativeIndices(intervals.Count, maxCount)
            .Select(index => intervals[index])
            .ToArray();
    }

    private static IEnumerable<int> SelectRepresentativeIndices(int count, int maxCount)
    {
        if (maxCount <= 0 || count <= 0)
        {
            yield break;
        }

        if (maxCount == 1)
        {
            yield return 0;
            yield break;
        }

        var lastIndex = count - 1;
        var seen = new HashSet<int>();
        for (var index = 0; index < maxCount; index++)
        {
            var representative = (int)Math.Round(index * (lastIndex / (double)(maxCount - 1)));
            if (seen.Add(representative))
            {
                yield return representative;
            }
        }
    }

    private static IReadOnlyList<EmbeddingSearchResult> DiversifySimilarResults(IEnumerable<SimilarCandidate> candidates)
    {
        var ranked = candidates
            .Select(static candidate => candidate.ToRankedResult())
            .OrderBy(static item => item.Result.Distance)
            .ThenBy(static item => item.Result.Embedding.HostId)
            .ToList();

        if (ranked.Count <= 2)
        {
            return ranked.Select(static item => item.Result).ToArray();
        }

        var selected = new HashSet<int>();
        var results = new List<EmbeddingSearchResult>(ranked.Count);
        foreach (var item in ranked.Take(2))
        {
            if (selected.Add(item.Result.Embedding.HostId))
            {
                results.Add(item.Result);
            }
        }

        var roleOrder = new[]
        {
            SimilarQueryRole.PartVisual,
            SimilarQueryRole.OverallVisual,
            SimilarQueryRole.OverallSemantic,
            SimilarQueryRole.PartSemantic,
        };

        while (results.Count < ranked.Count)
        {
            var added = false;
            foreach (var role in roleOrder)
            {
                var next = ranked.FirstOrDefault(item => item.Role == role && !selected.Contains(item.Result.Embedding.HostId));
                if (next is null)
                {
                    continue;
                }

                selected.Add(next.Result.Embedding.HostId);
                results.Add(next.Result);
                added = true;
                if (results.Count >= ranked.Count)
                {
                    break;
                }
            }

            if (!added)
            {
                var next = ranked.FirstOrDefault(item => !selected.Contains(item.Result.Embedding.HostId));
                if (next is null)
                {
                    break;
                }

                selected.Add(next.Result.Embedding.HostId);
                results.Add(next.Result);
            }
        }

        return results;
    }

    private static AiVisualSimilarPage NormalizeSimilarPage(int page, int perPage)
        => new(Math.Max(1, page), Math.Clamp(perPage <= 0 ? DefaultSimilarPerPage : perPage, 1, MaxSimilarPerPage));

    private static PaginatedResponse<AiVisualSimilarVideoDto> EmptySimilarVideoResponse(AiVisualSimilarPage pageInfo)
        => new([], 0, pageInfo.Page, pageInfo.PerPage);

    private static PaginatedResponse<AiVisualSimilarImageDto> EmptySimilarImageResponse(AiVisualSimilarPage pageInfo)
        => new([], 0, pageInfo.Page, pageInfo.PerPage);

    private async Task<IReadOnlyList<EmbeddingSearchResult>> SearchRankedMatchesAsync(string? queryText, EmbeddingHostType hostType, IReadOnlySet<int>? allowedIds, CancellationToken ct)
    {
        var query = CleanQuery(queryText);
        if (allowedIds is { Count: 0 })
        {
            return [];
        }

        var queryVector = await EncodeQueryVectorAsync(query, hostType, allowedIds, ct);
        if (queryVector is null)
        {
            return [];
        }

        var matches = await _embeddingService.KnnAsync(
            queryVector,
            int.MaxValue,
            new EmbeddingSearchOptions
            {
                HostType = hostType,
                Kind = VisualSemanticKind,
                KindFamily = SemanticKindFamily,
                Modality = EmbeddingModality.Visual,
                IsSemantic = true,
                SourceKey = VisualSourceKey,
            },
            ct);

        var ranked = matches.AsEnumerable();
        if (allowedIds is { Count: > 0 })
        {
            ranked = ranked.Where(match => allowedIds.Contains(match.Embedding.HostId));
        }
        else if (allowedIds is { Count: 0 })
        {
            return [];
        }

        var grouped = ranked
            .GroupBy(static match => match.Embedding.HostId)
            .Select(static group => group.OrderBy(static match => match.Distance).ThenBy(static match => match.Embedding.SectionIndex).First())
            .OrderBy(static match => match.Distance)
            .ThenBy(static match => match.Embedding.HostId)
            .ToArray();

        return ApplyVisualCandidateCutoff(grouped);
    }

    private async Task<Vector?> EncodeQueryVectorAsync(string query, EmbeddingHostType targetHostType, IReadOnlySet<int>? allowedIds, CancellationToken ct)
    {
        var fastVector = await TryEncodeWithRegisteredEncoderAsync(query, ct);
        if (fastVector is not null)
        {
            return fastVector;
        }

        var localVector = _localTextEncoder.TryEncode(query);
        if (localVector is not null)
        {
            return localVector;
        }

        return null;
    }

    private async Task<Vector?> TryEncodeWithRegisteredEncoderAsync(string query, CancellationToken ct)
    {
        var encoder = _textEncoderRegistry.Resolve(SemanticKindFamily);
        if (encoder is null)
        {
            return null;
        }

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromMilliseconds(FastTextEncodeTimeoutMilliseconds));

        try
        {
            return await encoder.EncodeAsync(query, timeout.Token);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            return null;
        }
        catch (HttpRequestException)
        {
            return null;
        }
        catch (InvalidOperationException)
        {
            return null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private async Task<Vector?> TryBuildLocalQueryVectorAsync(string query, EmbeddingHostType targetHostType, IReadOnlySet<int>? allowedIds, CancellationToken ct)
    {
        var seeds = await LoadLocalTextSeedsAsync(query, targetHostType, allowedIds, ct);
        if (seeds.Count == 0)
        {
            return null;
        }

        var videoIds = seeds.Where(static seed => seed.HostType == EmbeddingHostType.Video).Select(static seed => seed.HostId).Distinct().ToArray();
        var imageIds = seeds.Where(static seed => seed.HostType == EmbeddingHostType.Image).Select(static seed => seed.HostId).Distinct().ToArray();
        if (videoIds.Length == 0 && imageIds.Length == 0)
        {
            return null;
        }

        var videoEmbeddings = videoIds.Length > 0
            ? await _embeddingRepository.FindAsync(new EmbeddingFilter
            {
                HostType = EmbeddingHostType.Video, HostIds = videoIds, SourceKey = VisualSourceKey,
                Kind = VisualSemanticKind, KindFamily = SemanticKindFamily,
                Modality = EmbeddingModality.Visual, IsSemantic = true,
            }, ct)
            : [];
        var imageEmbeddings = imageIds.Length > 0
            ? await _embeddingRepository.FindAsync(new EmbeddingFilter
            {
                HostType = EmbeddingHostType.Image, HostIds = imageIds, SourceKey = VisualSourceKey,
                Kind = VisualSemanticKind, KindFamily = SemanticKindFamily,
                Modality = EmbeddingModality.Visual, IsSemantic = true,
            }, ct)
            : [];
        var embeddings = videoEmbeddings.Where(static e => e.SectionIndex == 0)
            .Concat(imageEmbeddings.Where(static e => e.SectionIndex == 0)).ToList();

        if (embeddings.Count == 0)
        {
            return null;
        }

        var seedRank = seeds
            .Select(static (seed, index) => (Key: new EmbeddingHostKey(seed.HostType, seed.HostId), Rank: index))
            .GroupBy(static item => item.Key)
            .ToDictionary(static group => group.Key, static group => group.Min(static item => item.Rank));

        var dimensionGroup = embeddings
            .GroupBy(static embedding => embedding.Dim)
            .OrderByDescending(static group => group.Count())
            .ThenByDescending(static group => group.Key)
            .First();

        var weighted = dimensionGroup
            .Select(embedding =>
            {
                var rank = seedRank.GetValueOrDefault(new EmbeddingHostKey(embedding.HostType, embedding.HostId), LocalTextSeedLimit);
                return (Embedding: embedding, Weight: 1d / (rank + 1d));
            })
            .ToArray();

        return BuildNormalizedCentroid(weighted);
    }

    private async Task<IReadOnlyList<LocalTextSeed>> LoadLocalTextSeedsAsync(string query, EmbeddingHostType targetHostType, IReadOnlySet<int>? allowedIds, CancellationToken ct)
    {
        var normalized = query.Trim().ToLowerInvariant();
        if (string.IsNullOrWhiteSpace(normalized))
        {
            return [];
        }

        var (videoResults, _) = await _videoRepository.FindAsync(
            null,
            new FindFilter { Q = normalized, PerPage = LocalTextSeedScanLimit, Sort = "updated_at", Direction = SortDirection.Desc },
            ct);
        var videoCandidates = videoResults.Select(static video => new
        {
            video.Id, video.Title, video.Details, video.Code, video.FileSearchText, video.UpdatedAt,
        }).ToList();

        var (imageResults, _) = await _imageRepository.FindAsync(
            null,
            new FindFilter { Q = normalized, PerPage = LocalTextSeedScanLimit, Sort = "updated_at", Direction = SortDirection.Desc },
            ct);
        var imageCandidates = imageResults.Select(static image => new
        {
            image.Id, image.Title, image.Details, image.Code,
            Photographer = (string?)null, image.FileSearchText, image.UpdatedAt,
        }).ToList();

        var videoSeeds = videoCandidates.Select(video => new LocalTextSeed(
            EmbeddingHostType.Video,
            video.Id,
            GetTextScore(normalized, video.Title, video.Details, video.Code, video.FileSearchText),
            video.UpdatedAt));

        var imageSeeds = imageCandidates.Select(image => new LocalTextSeed(
            EmbeddingHostType.Image,
            image.Id,
            GetTextScore(normalized, image.Title, image.Details, image.Code, image.Photographer, image.FileSearchText),
            image.UpdatedAt));

        return videoSeeds
            .Concat(imageSeeds)
            .Where(seed => allowedIds is null || seed.HostType != targetHostType || allowedIds.Contains(seed.HostId))
            .OrderBy(static seed => seed.Score)
            .ThenByDescending(static seed => seed.UpdatedAt)
            .ThenBy(static seed => seed.HostType)
            .ThenBy(static seed => seed.HostId)
            .Take(LocalTextSeedLimit)
            .ToArray();
    }

    private static int GetTextScore(string query, params string?[] fields)
    {
        var score = 100;
        for (var index = 0; index < fields.Length; index++)
        {
            var value = fields[index];
            if (string.IsNullOrWhiteSpace(value))
            {
                continue;
            }

            var normalized = value.Trim().ToLowerInvariant();
            var fieldScore = normalized.Equals(query, StringComparison.OrdinalIgnoreCase)
                ? 0
                : normalized.StartsWith(query, StringComparison.OrdinalIgnoreCase)
                    ? 1
                    : normalized.Contains(query, StringComparison.OrdinalIgnoreCase)
                        ? 2
                        : 100;

            if (fieldScore < 100)
            {
                score = Math.Min(score, fieldScore + (index * 4));
            }
        }

        return score;
    }

    private static Vector? BuildNormalizedCentroid(IReadOnlyList<(Embedding Embedding, double Weight)> weighted)
    {
        if (weighted.Count == 0)
        {
            return null;
        }

        var length = weighted[0].Embedding.Vector.ToArray().Length;
        if (length == 0)
        {
            return null;
        }

        var sums = new double[length];
        var totalWeight = 0d;
        foreach (var (embedding, weight) in weighted)
        {
            var values = embedding.Vector.ToArray();
            if (values.Length != length || weight <= 0d)
            {
                continue;
            }

            for (var index = 0; index < values.Length; index++)
            {
                sums[index] += values[index] * weight;
            }

            totalWeight += weight;
        }

        if (totalWeight <= 0d)
        {
            return null;
        }

        var vector = new float[length];
        var norm = 0d;
        for (var index = 0; index < sums.Length; index++)
        {
            var value = sums[index] / totalWeight;
            vector[index] = (float)value;
            norm += value * value;
        }

        if (norm <= 0d)
        {
            return null;
        }

        var scale = Math.Sqrt(norm);
        for (var index = 0; index < vector.Length; index++)
        {
            vector[index] = (float)(vector[index] / scale);
        }

        return new Vector(vector);
    }

    private async Task<IReadOnlyList<EmbeddingSearchResult>> SortVideoMatchesAsync(IReadOnlyList<EmbeddingSearchResult> matches, FindFilter findFilter, CancellationToken ct)
    {
        if (IsVisualMatchSort(findFilter.Sort))
        {
            return SortByVisualMatch(matches, ShouldSortVisualMatchDescending(findFilter));
        }

        var ids = matches.Select(static match => match.Embedding.HostId).Distinct().ToList();
        var (videoItems, _) = await _videoRepository.FindAsync(new VideoFilter { Ids = ids }, null, ct);
        var videos = videoItems.ToDictionary(static v => v.Id);

        var visualOrder = BuildVisualOrder(matches);
        var sortable = matches.Where(match => videos.ContainsKey(match.Embedding.HostId)).ToArray();
        var snapshots = _engagementService == null
            ? null
            : await _engagementService.GetVideoSnapshotsAsync(ids, ct);
        var desc = findFilter.Direction == SortDirection.Desc;
        var sort = findFilter.Sort?.Trim().ToLowerInvariant();

        return sort switch
        {
            "title" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].Title, desc, visualOrder),
            "date" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].Date, desc, visualOrder),
            "rating" => OrderMatchesByKey(sortable, match => snapshots?.GetValueOrDefault(match.Embedding.HostId)?.Rating, desc, visualOrder),
            "play_count" => OrderMatchesByKey(sortable, match => snapshots?.GetValueOrDefault(match.Embedding.HostId)?.PlayCount, desc, visualOrder),
            "o_counter" or "like_counter" => OrderMatchesByKey(sortable, match => snapshots?.GetValueOrDefault(match.Embedding.HostId)?.LikeCount, desc, visualOrder),
            "last_o_at" or "last_like_at" => OrderMatchesByKey(sortable, match => GetLastFavoriteAt(videos[match.Embedding.HostId]), desc, visualOrder),
            "organized" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].Organized, desc, visualOrder),
            "last_played_at" => OrderMatchesByKey(sortable, match => snapshots?.GetValueOrDefault(match.Embedding.HostId)?.LastPlayedAt, desc, visualOrder),
            "play_duration" => OrderMatchesByKey(sortable, match => snapshots?.GetValueOrDefault(match.Embedding.HostId)?.PlayDuration, desc, visualOrder),
            "resume_time" => OrderMatchesByKey(sortable, match => snapshots?.GetValueOrDefault(match.Embedding.HostId)?.ResumeTime, desc, visualOrder),
            "random" => OrderMatchesByRandom(sortable, findFilter.Seed, desc),
            "duration" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].MaxDuration, desc, visualOrder),
            "file_size" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].MaxFileSize, desc, visualOrder),
            "file_mod_time" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].MaxFileModTime, desc, visualOrder),
            "file_count" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].FileCount, desc, visualOrder),
            "path" => OrderMatchesByKey(sortable, match => desc ? videos[match.Embedding.HostId].MaxPath : videos[match.Embedding.HostId].MinPath, desc, visualOrder),
            "resolution" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].MaxHeight, desc, visualOrder),
            "framerate" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].MaxFrameRate, desc, visualOrder),
            "bitrate" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].MaxBitRate, desc, visualOrder),
            "phash" or "perceptual_similarity" => OrderMatchesByKey(sortable, match => GetVideoPhash(videos[match.Embedding.HostId], desc), desc, visualOrder),
            "tag_count" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].TagIds.Length, desc, visualOrder),
            "performer_count" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].PerformerIds.Length, desc, visualOrder),
            "performer_age" => OrderMatchesByKey(sortable, match => GetVideoPerformerAge(videos[match.Embedding.HostId], desc), desc, visualOrder),
            "studio" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].Studio?.Name, desc, visualOrder),
            "code" or "studio_code" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].Code, desc, visualOrder),
            "created_at" => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].CreatedAt, desc, visualOrder),
            _ => OrderMatchesByKey(sortable, match => videos[match.Embedding.HostId].UpdatedAt, desc, visualOrder),
        };
    }

    private async Task<IReadOnlyList<EmbeddingSearchResult>> SortImageMatchesAsync(IReadOnlyList<EmbeddingSearchResult> matches, FindFilter findFilter, CancellationToken ct)
    {
        if (IsVisualMatchSort(findFilter.Sort))
        {
            return SortByVisualMatch(matches, ShouldSortVisualMatchDescending(findFilter));
        }

        var ids = matches.Select(static match => match.Embedding.HostId).Distinct().ToList();
        var (imageItems, _) = await _imageRepository.FindAsync(new ImageFilter { Ids = ids }, null, ct);
        var images = imageItems.ToDictionary(static i => i.Id);

        var visualOrder = BuildVisualOrder(matches);
        var sortable = matches.Where(match => images.ContainsKey(match.Embedding.HostId)).ToArray();
        var snapshots = _engagementService == null
            ? null
            : await _engagementService.GetSnapshotsAsync(AffinityHostType.Image, ids, ct);
        var desc = findFilter.Direction == SortDirection.Desc;
        var sort = findFilter.Sort?.Trim().ToLowerInvariant();

        return sort switch
        {
            "title" => OrderMatchesByKey(sortable, match => GetImageDisplayTitle(images[match.Embedding.HostId], desc), desc, visualOrder),
            "date" => OrderMatchesByKey(sortable, match => images[match.Embedding.HostId].Date, desc, visualOrder),
            "rating" => OrderMatchesByKey(sortable, match => snapshots?.GetValueOrDefault(match.Embedding.HostId)?.Rating, desc, visualOrder),
            "o_counter" or "like_counter" => OrderMatchesByKey(sortable, match => snapshots?.GetValueOrDefault(match.Embedding.HostId)?.LikeCount, desc, visualOrder),
            "random" => OrderMatchesByRandom(sortable, findFilter.Seed, desc),
            "file_mod_time" => OrderMatchesByKey(sortable, match => images[match.Embedding.HostId].MaxFileModTime, desc, visualOrder),
            "file_size" => OrderMatchesByKey(sortable, match => images[match.Embedding.HostId].MaxFileSize, desc, visualOrder),
            "path" => OrderMatchesByKey(sortable, match => desc ? images[match.Embedding.HostId].MaxPath : images[match.Embedding.HostId].MinPath, desc, visualOrder),
            "tag_count" => OrderMatchesByKey(sortable, match => images[match.Embedding.HostId].TagCount, desc, visualOrder),
            "performer_count" => OrderMatchesByKey(sortable, match => images[match.Embedding.HostId].PerformerCount, desc, visualOrder),
            "created_at" => OrderMatchesByKey(sortable, match => images[match.Embedding.HostId].CreatedAt, desc, visualOrder),
            _ => OrderMatchesByKey(sortable, match => images[match.Embedding.HostId].UpdatedAt, desc, visualOrder),
        };
    }

    private async Task<IReadOnlySet<int>?> ResolveAllowedVideoIdsAsync(VideoFilter? filter, CancellationToken ct)
    {
        if (filter is null)
        {
            return null;
        }

        var (items, _) = await _videoRepository.FindAsync(filter, new FindFilter { Page = 1, PerPage = MaxFilteredIds }, ct);
        return items.Select(static item => item.Id).ToHashSet();
    }

    private async Task<IReadOnlySet<int>?> ResolveAllowedImageIdsAsync(ImageFilter? filter, CancellationToken ct)
    {
        if (filter is null)
        {
            return null;
        }

        var (items, _) = await _imageRepository.FindAsync(filter, new FindFilter { Page = 1, PerPage = MaxFilteredIds }, ct);
        return items.Select(static image => image.Id).ToHashSet();
    }

    private async Task<List<VideoDto>> BuildVideoResultsAsync(IReadOnlyList<EmbeddingSearchResult> ranked, CancellationToken ct)
    {
        var hostIds = ranked.Select(static match => match.Embedding.HostId).Distinct().ToList();
        var (videos, _) = await _videoRepository.FindAsync(new VideoFilter { Ids = hostIds }, null, ct);
        var videoMap = videos.ToDictionary(static v => v.Id);

        var results = new List<VideoDto>(ranked.Count);
        foreach (var match in ranked)
        {
            if (videoMap.TryGetValue(match.Embedding.HostId, out var video))
            {
                results.Add(MapVideo(video, null, false));
            }
        }

        return results;
    }

    private async Task<List<ImageDto>> BuildImageResultsAsync(IReadOnlyList<EmbeddingSearchResult> ranked, CancellationToken ct)
    {
        var hostIds = ranked.Select(static match => match.Embedding.HostId).Distinct().ToList();
        var (images, _) = await _imageRepository.FindAsync(new ImageFilter { Ids = hostIds }, null, ct);
        var imageMap = images.ToDictionary(static i => i.Id);

        var results = new List<ImageDto>(ranked.Count);
        foreach (var match in ranked)
        {
            if (imageMap.TryGetValue(match.Embedding.HostId, out var image))
            {
                results.Add(MapImage(image));
            }
        }

        return results;
    }

    private static FindFilter NormalizeFindFilter(FindFilter? filter)
        => new()
        {
            Q = filter?.Q,
            Page = Math.Max(1, filter?.Page ?? 1),
            PerPage = Math.Clamp(filter?.PerPage ?? 40, 1, 500),
            Sort = CleanSort(filter?.Sort),
            Direction = filter?.Direction ?? SortDirection.Asc,
            Seed = filter?.Seed,
        };

    private static IReadOnlyList<EmbeddingSearchResult> Page(IReadOnlyList<EmbeddingSearchResult> ranked, FindFilter findFilter)
        => ranked
            .Skip((findFilter.Page - 1) * findFilter.PerPage)
            .Take(findFilter.PerPage)
            .ToArray();

    private VideoDto MapVideo(Video video, UserEngagementSnapshot? engagement, bool preferUserSnapshot)
        => new(
            video.Id,
            video.Title,
            video.Code,
            video.Details,
            video.Director,
            video.Date?.ToString("yyyy-MM-dd"),
            video.Organized,
            video.IsVr,
            video.StudioId,
            video.Studio?.Name,
            video.Captions,
            video.InteractiveSpeed,
            video.Urls.Select(static url => url.Url).ToList(),
            video.VideoTags.Where(static vt => vt.Tag != null).Select(static vt => new TagDto(vt.Tag!.Id, vt.Tag.Name, vt.Tag.Description, vt.Tag.Favorite, vt.Tag.IgnoreAutoTag, [])).ToList(),
            video.VideoPerformers.Where(static vp => vp.Performer != null).Select(static vp => MapPerformer(vp.Performer!)).ToList(),
            video.Files.Select(file => new VideoFileDto(
                file.Id,
                CanReadFiles ? file.Path : string.Empty,
                GetVisibleBasename(file.Path, file.Basename),
                file.Format,
                file.Width,
                file.Height,
                file.Duration,
                file.VideoCodec,
                file.AudioCodec,
                file.FrameRate,
                file.BitRate,
                file.Size,
                [],
                [])).ToList(),
            MapVideoGroups(video),
            video.VideoGalleries.Where(static vg => vg.Gallery != null).Select(static vg => new GallerySummaryDto(vg.Gallery!.Id, vg.Gallery.Title, vg.Gallery.Date?.ToString("yyyy-MM-dd"))).ToList(),
            [],
            video.CustomFields,
            video.CreatedAt.ToString("o"),
            video.UpdatedAt.ToString("o"));

    private ImageDto MapImage(Image image)
        => new(
            image.Id,
            image.Title,
            image.Code,
            image.Details,
            image.Photographer,
            image.Organized,
            image.StudioId,
            image.Studio?.Name,
            image.Date?.ToString("yyyy-MM-dd"),
            image.Urls.Select(static url => url.Url).ToList(),
            image.ImageTags.Where(static imageTag => imageTag.Tag != null).Select(static imageTag => new TagDto(imageTag.Tag!.Id, imageTag.Tag.Name, imageTag.Tag.Description, imageTag.Tag.Favorite, imageTag.Tag.IgnoreAutoTag, [])).ToList(),
            image.ImagePerformers.Where(static imagePerformer => imagePerformer.Performer != null).Select(static imagePerformer => MapPerformer(imagePerformer.Performer!)).ToList(),
            image.GalleryCount,
            image.ImageGalleries?.Select(static imageGallery => imageGallery.GalleryId).ToList() ?? [],
            image.ImageGalleries?.Where(static imageGallery => imageGallery.Gallery != null).Select(static imageGallery => new GallerySummaryDto(imageGallery.GalleryId, imageGallery.Gallery!.Title, imageGallery.Gallery.Date?.ToString("yyyy-MM-dd"))).ToList() ?? [],
            [],
            image.Files?.Select(file => new ImageFileDto(
                file.Id,
                CanReadFiles ? file.Path : string.Empty,
                GetVisibleBasename(file.Path, file.Basename),
                file.Format ?? string.Empty,
                file.Width,
                file.Height,
                file.Size)).ToList() ?? [],
            image.CustomFields,
            image.CreatedAt.ToString("o"),
            image.UpdatedAt.ToString("o"));

    private static PerformerSummaryDto MapPerformer(Performer performer)
        => new(
            performer.Id,
            performer.Name,
            performer.Disambiguation,
            performer.Gender?.ToString(),
            performer.Birthdate?.ToString("yyyy-MM-dd"),
            performer.Favorite,
            performer.ImageBlobId != null ? BuildEntityImageUrl($"/api/performers/{performer.Id}/image", performer.UpdatedAt) : null);

    private static List<GroupSummaryDto> MapVideoGroups(Video video)
        => video.GroupItems
            .Where(static item => item.Kind == GroupItemKind.Video && item.Group != null)
            .OrderBy(static item => item.OrderIndex)
            .Select(static item => new GroupSummaryDto(item.Group!.Id, item.Group.Name, item.OrderIndex))
            .ToList();

    private static string CleanQuery(string? query)
    {
        var cleaned = query?.Trim();
        if (string.IsNullOrWhiteSpace(cleaned))
        {
            throw new ArgumentException("Query is required.", nameof(query));
        }

        return cleaned;
    }

    private static string GetVisibleBasename(string path, string basename)
        => string.IsNullOrWhiteSpace(basename) ? Path.GetFileName(path) : basename;

    private static string BuildEntityImageUrl(string path, DateTime updatedAt, int maxDimension = 640)
        => $"{path}?max={maxDimension}&v={Uri.EscapeDataString(updatedAt.ToString("o"))}";

    private static IReadOnlyList<EmbeddingSearchResult> ApplyVisualCandidateCutoff(IReadOnlyList<EmbeddingSearchResult> ranked)
    {
        if (ranked.Count <= MinimumVisualMatchResults)
        {
            return ranked;
        }

        var cutoff = ranked[0].Distance + VisualMatchDistanceWindow;
        return ranked
            .Where((match, index) => index < MinimumVisualMatchResults || match.Distance <= cutoff)
            .Take(MaxVisualMatchResults)
            .ToArray();
    }

    private static IReadOnlyList<EmbeddingSearchResult> SortByVisualMatch(IReadOnlyList<EmbeddingSearchResult> matches, bool desc)
        => desc
            ? matches.OrderBy(static match => match.Distance).ThenBy(static match => match.Embedding.HostId).ToArray()
            : matches.OrderByDescending(static match => match.Distance).ThenBy(static match => match.Embedding.HostId).ToArray();

    private static IReadOnlyDictionary<int, int> BuildVisualOrder(IReadOnlyList<EmbeddingSearchResult> matches)
        => matches.Select(static (match, index) => (match.Embedding.HostId, index))
            .GroupBy(static item => item.HostId)
            .ToDictionary(static group => group.Key, static group => group.Min(static item => item.index));

    private static IReadOnlyList<EmbeddingSearchResult> OrderMatchesByKey(
        IReadOnlyList<EmbeddingSearchResult> matches,
        Func<EmbeddingSearchResult, IComparable?> keySelector,
        bool desc,
        IReadOnlyDictionary<int, int> visualOrder)
    {
        IOrderedEnumerable<EmbeddingSearchResult> ordered = desc
            ? matches.OrderBy(match => keySelector(match) is null).ThenByDescending(keySelector, NullableComparableComparer.Instance)
            : matches.OrderBy(match => keySelector(match) is null).ThenBy(keySelector, NullableComparableComparer.Instance);

        return ordered
            .ThenBy(match => GetVisualOrder(visualOrder, match.Embedding.HostId))
            .ThenBy(static match => match.Embedding.HostId)
            .ToArray();
    }

    private static IReadOnlyList<EmbeddingSearchResult> OrderMatchesByRandom(IReadOnlyList<EmbeddingSearchResult> matches, int? seed, bool desc)
        => desc
            ? matches.OrderByDescending(match => GetSeededSortKey(match.Embedding.HostId, seed)).ThenByDescending(static match => match.Embedding.HostId).ToArray()
            : matches.OrderBy(match => GetSeededSortKey(match.Embedding.HostId, seed)).ThenBy(static match => match.Embedding.HostId).ToArray();

    private static DateTime? GetLastFavoriteAt(Video video)
        => video.LikeHistory.Select(static history => (DateTime?)history.OccurredAt).Max();

    private static string? GetVideoPhash(Video video, bool desc)
    {
        var values = video.Files
            .SelectMany(static file => file.Fingerprints)
            .Where(static fingerprint => fingerprint.Type == "phash" && fingerprint.Value != string.Empty)
            .Select(static fingerprint => fingerprint.Value);

        return desc ? values.OrderByDescending(static value => value).FirstOrDefault() : values.OrderBy(static value => value).FirstOrDefault();
    }

    private static int? GetVideoPerformerAge(Video video, bool desc)
    {
        if (video.Date is null)
        {
            return null;
        }

        var ages = video.VideoPerformers
            .Where(static videoPerformer => videoPerformer.Performer?.Birthdate is not null)
            .Select(videoPerformer =>
            {
                var birthdate = videoPerformer.Performer!.Birthdate!.Value;
                var date = video.Date.Value;
                return date.Year - birthdate.Year - ((date.Month < birthdate.Month || (date.Month == birthdate.Month && date.Day < birthdate.Day)) ? 1 : 0);
            })
            .ToArray();

        return ages.Length == 0 ? null : desc ? ages.Max() : ages.Min();
    }

    private static string? GetImageDisplayTitle(Image image, bool desc)
    {
        if (!string.IsNullOrWhiteSpace(image.Title))
        {
            return image.Title;
        }

        var basenames = image.Files.Select(static file => file.Basename);
        return desc ? basenames.OrderByDescending(static value => value).FirstOrDefault() : basenames.OrderBy(static value => value).FirstOrDefault();
    }

    private static int GetVisualOrder(IReadOnlyDictionary<int, int> visualOrder, int hostId)
        => visualOrder.TryGetValue(hostId, out var order) ? order : int.MaxValue;

    private static int GetSeededSortKey(int id, int? seed)
    {
        unchecked
        {
            var value = (uint)id;
            value ^= (uint)(seed ?? 0) + 0x9E3779B9u + (value << 6) + (value >> 2);
            value *= 0x85EBCA6Bu;
            value ^= value >> 13;
            value *= 0xC2B2AE35u;
            value ^= value >> 16;
            return (int)value;
        }
    }

    private static bool IsVisualMatchSort(string? sort)
        => string.IsNullOrWhiteSpace(sort) || string.Equals(sort, VisualMatchSort, StringComparison.OrdinalIgnoreCase);

    private static bool ShouldSortVisualMatchDescending(FindFilter findFilter)
        => string.IsNullOrWhiteSpace(findFilter.Sort) || findFilter.Direction == SortDirection.Desc;

    private static string? CleanSort(string? sort)
        => string.IsNullOrWhiteSpace(sort) ? null : sort.Trim();

    private sealed record LocalTextSeed(EmbeddingHostType HostType, int HostId, int Score, DateTime UpdatedAt);

    private readonly record struct EmbeddingHostKey(EmbeddingHostType HostType, int HostId);

    private sealed record SimilarRankedResult(EmbeddingSearchResult Result, SimilarQueryRole Role);

    private sealed class SimilarCandidate(int hostId)
    {
        private readonly HashSet<SimilarQueryRole> _roles = [];
        private EmbeddingSearchResult? _bestResult;
        private double _bestWeightedSimilarity = double.NegativeInfinity;
        private SimilarQueryRole _primaryRole = SimilarQueryRole.OverallVisual;
        private double _score;
        private double _weight;

        public int HostId { get; } = hostId;

        public void Add(EmbeddingSearchResult match, AiVisualSimilarQuery query)
        {
            var similarity = Math.Clamp(1d - match.Distance, -1d, 1d);
            var weightedSimilarity = similarity * query.Weight;

            _score += weightedSimilarity;
            _weight += query.Weight;
            _roles.Add(query.Role);

            if (weightedSimilarity <= _bestWeightedSimilarity)
            {
                return;
            }

            _bestWeightedSimilarity = weightedSimilarity;
            _bestResult = match;
            _primaryRole = query.Role;
        }

        public SimilarRankedResult ToRankedResult()
        {
            var result = _bestResult ?? throw new InvalidOperationException($"Similarity candidate {HostId} has no embedding result.");
            var averageSimilarity = _weight <= 0d ? 0d : _score / _weight;
            var coverageBonus = Math.Min(0.04d, Math.Max(0, _roles.Count - 1) * 0.012d);
            var distance = (float)Math.Clamp(1d - averageSimilarity - coverageBonus, 0d, 2d);
            return new SimilarRankedResult(new EmbeddingSearchResult(result.Embedding, distance), _primaryRole);
        }
    }

    private sealed class NullableComparableComparer : IComparer<IComparable?>
    {
        public static readonly NullableComparableComparer Instance = new();

        public int Compare(IComparable? x, IComparable? y)
        {
            if (ReferenceEquals(x, y))
            {
                return 0;
            }

            if (x is null)
            {
                return 1;
            }

            if (y is null)
            {
                return -1;
            }

            return x.CompareTo(y);
        }
    }
}