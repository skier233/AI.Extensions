using Cove.Core.Auth;
using Cove.Core.DTOs;
using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Audio;

internal sealed record AiAudioSimilarVideoDto(
    VideoDto Video,
    float Distance,
    string Kind,
    string? KindFamily,
    int SectionIndex,
    double? StartSec,
    double? EndSec);

internal sealed record AiAudioSimilarPage(int Page, int PerPage);

internal sealed record AiAudioSimilarQuery(Embedding Embedding, TargetEmbeddingScope TargetScope, double Weight);

internal enum TargetEmbeddingScope
{
    Any,
    Asset,
    Section,
}

internal sealed class AiAudioSimilarityService(
    IEmbeddingRepository embeddingRepository,
    IVideoRepository videoRepository,
    IEmbeddingService embeddingService,
    IUserEngagementService? engagementService = null,
    ICurrentPrincipalAccessor? principalAccessor = null)
{
    private const string AudioSourceKey = "ext:ai.audio";
    private const string AudioEmbeddingKind = "audio.embed.v1";
    private const string AudioEmbeddingKindFamily = "audio.v1";
    private const int DefaultSimilarPerPage = 12;
    private const int MaxSimilarPerPage = 48;
    private const int MaxSimilarResults = 500;
    private const int MaxSectionQueries = 4;
    private const double AssetWeight = 1.0;
    private const double SectionWeight = 0.65;

    private readonly IEmbeddingRepository _embeddingRepository = embeddingRepository;
    private readonly IVideoRepository _videoRepository = videoRepository;
    private readonly IEmbeddingService _embeddingService = embeddingService;
    private readonly IUserEngagementService? _engagementService = engagementService;
    private readonly ICurrentPrincipalAccessor? _principalAccessor = principalAccessor;

    private bool CanReadFiles => _principalAccessor?.Current?.Has(Permissions.FilesRead) == true;
    private bool HasUserScopedEngagement => _principalAccessor?.Current?.UserId != null;

    // Cheap existence check for whether a video has any audio embeddings, used to decide if the
    // audio-similarity tab should appear — far faster than running the full similarity search with
    // perPage=1 (whose cost is the KNN scan, which perPage does not reduce).
    public Task<bool> HasAudioEmbeddingsAsync(int videoId, CancellationToken ct = default)
        => _embeddingRepository.ExistsAsync(new EmbeddingFilter
        {
            HostType = EmbeddingHostType.Video,
            HostId = videoId,
            SourceKey = AudioSourceKey,
            Modality = EmbeddingModality.Audio,
        }, ct);

    public async Task<PaginatedResponse<AiAudioSimilarVideoDto>> SimilarVideosForVideoAsync(int videoId, int page = 1, int perPage = DefaultSimilarPerPage, CancellationToken ct = default)
    {
        var pageInfo = NormalizeSimilarPage(page, perPage);
        var queries = await LoadSimilarityQueriesAsync(videoId, ct);
        if (queries.Count == 0)
        {
            return new PaginatedResponse<AiAudioSimilarVideoDto>([], 0, pageInfo.Page, pageInfo.PerPage);
        }

        var ranked = await SearchSimilarAsync(queries, videoId, ct);
        var pageMatches = ranked
            .Skip((pageInfo.Page - 1) * pageInfo.PerPage)
            .Take(pageInfo.PerPage)
            .ToArray();
        var videos = await BuildVideoResultsAsync(pageMatches, ct);
        var videoById = videos.ToDictionary(static v => v.Id);
        var items = pageMatches
            .Where(match => videoById.ContainsKey(match.Embedding.HostId))
            .Select(match => new AiAudioSimilarVideoDto(
                videoById[match.Embedding.HostId],
                match.Distance,
                match.Embedding.Kind,
                match.Embedding.KindFamily,
                match.Embedding.SectionIndex,
                match.Embedding.StartSec,
                match.Embedding.EndSec))
            .ToList();

        return new PaginatedResponse<AiAudioSimilarVideoDto>(items, ranked.Count, pageInfo.Page, pageInfo.PerPage);
    }

    private async Task<IReadOnlyList<AiAudioSimilarQuery>> LoadSimilarityQueriesAsync(int videoId, CancellationToken ct)
    {
        var embeddings = await _embeddingRepository.FindAsync(new EmbeddingFilter
        {
            HostType = EmbeddingHostType.Video,
            HostId = videoId,
            SourceKey = AudioSourceKey,
            Kind = AudioEmbeddingKind,
            KindFamily = AudioEmbeddingKindFamily,
            Modality = EmbeddingModality.Audio,
            IsSemantic = false,
        }, ct);

        var sorted = embeddings
            .OrderBy(static e => e.SectionIndex)
            .ThenBy(static e => e.StartSec ?? double.MaxValue)
            .ToArray();

        var queries = new List<AiAudioSimilarQuery>();
        var asset = sorted.FirstOrDefault(static e => e.SectionIndex == 0);
        if (asset is not null)
        {
            queries.Add(new AiAudioSimilarQuery(asset, TargetEmbeddingScope.Asset, AssetWeight));
        }

        foreach (var section in SelectRepresentativeEmbeddings(sorted.Where(static e => e.SectionIndex > 0).ToArray(), MaxSectionQueries))
        {
            queries.Add(new AiAudioSimilarQuery(section, TargetEmbeddingScope.Section, SectionWeight));
        }

        return queries;
    }

    private async Task<IReadOnlyList<EmbeddingSearchResult>> SearchSimilarAsync(IReadOnlyList<AiAudioSimilarQuery> queries, int videoId, CancellationToken ct)
    {
        var candidates = new Dictionary<int, SimilarCandidate>();
        foreach (var query in queries.Where(static q => q.Weight > 0d))
        {
            var matches = await _embeddingService.KnnAsync(
                query.Embedding.Vector,
                int.MaxValue,
                new EmbeddingSearchOptions
                {
                    HostType = EmbeddingHostType.Video,
                    Kind = query.Embedding.Kind,
                    KindFamily = query.Embedding.KindFamily,
                    Modality = EmbeddingModality.Audio,
                    IsSemantic = false,
                    SourceKey = AudioSourceKey,
                },
                ct);

            var bestPerHost = matches
                .Where(match => match.Embedding.HostId != videoId)
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

                candidate.Add(match, query.Weight);
            }
        }

        return candidates.Values
            .Select(static candidate => candidate.ToResult())
            .Where(static result => result is not null)
            .Select(static result => result!)
            .OrderBy(static result => result.Distance)
            .ThenBy(static result => result.Embedding.HostId)
            .Take(MaxSimilarResults)
            .ToArray();
    }

    private async Task<List<VideoDto>> BuildVideoResultsAsync(IReadOnlyList<EmbeddingSearchResult> ranked, CancellationToken ct)
    {
        var hostIds = ranked.Select(static match => match.Embedding.HostId).Distinct().ToList();
        var (videos, _) = await _videoRepository.FindAsync(new VideoFilter { Ids = hostIds }, null, ct);
        var videoMap = videos.ToDictionary(static v => v.Id);

        var engagement = _engagementService == null
            ? []
            : await _engagementService.GetVideoSnapshotsAsync(hostIds.ToArray(), ct);

        var results = new List<VideoDto>(ranked.Count);
        foreach (var match in ranked)
        {
            if (videoMap.TryGetValue(match.Embedding.HostId, out var video))
            {
                results.Add(MapVideo(video, engagement.GetValueOrDefault(video.Id), HasUserScopedEngagement));
            }
        }

        return results;
    }

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
            video.Urls.Select(static u => u.Url).ToList(),
            video.VideoTags.Where(static vt => vt.Tag != null).Select(static vt => new TagDto(vt.Tag!.Id, vt.Tag.Name, vt.Tag.Description, vt.Tag.Favorite, [])).ToList(),
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

    private static bool MatchesTargetScope(Embedding embedding, TargetEmbeddingScope targetScope)
        => targetScope switch
        {
            TargetEmbeddingScope.Asset => embedding.SectionIndex == 0,
            TargetEmbeddingScope.Section => embedding.SectionIndex > 0,
            _ => true,
        };

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

    private static IEnumerable<int> SelectRepresentativeIndices(int count, int maxCount)
    {
        if (maxCount <= 0 || count <= 0) yield break;
        if (maxCount == 1) { yield return 0; yield break; }

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

    private static AiAudioSimilarPage NormalizeSimilarPage(int page, int perPage)
        => new(Math.Max(1, page), Math.Clamp(perPage <= 0 ? DefaultSimilarPerPage : perPage, 1, MaxSimilarPerPage));

    private static string GetVisibleBasename(string path, string basename)
        => string.IsNullOrWhiteSpace(basename) ? Path.GetFileName(path) : basename;

    private static string BuildEntityImageUrl(string path, DateTime updatedAt, int maxDimension = 640)
        => $"{path}?max={maxDimension}&v={Uri.EscapeDataString(updatedAt.ToString("o"))}";

    private sealed class SimilarCandidate(int hostId)
    {
        private double _weightedDistance;
        private double _totalWeight;
        private EmbeddingSearchResult? _bestMatch;

        public int HostId { get; } = hostId;

        public void Add(EmbeddingSearchResult match, double weight)
        {
            _weightedDistance += match.Distance * weight;
            _totalWeight += weight;
            if (_bestMatch is null || match.Distance < _bestMatch.Distance)
            {
                _bestMatch = match;
            }
        }

        public EmbeddingSearchResult? ToResult()
        {
            if (_bestMatch is null || _totalWeight <= 0d) return null;
            return new EmbeddingSearchResult(_bestMatch.Embedding, (float)(_weightedDistance / _totalWeight));
        }
    }
}
