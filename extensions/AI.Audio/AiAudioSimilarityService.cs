using Cove.Core.Auth;
using Cove.Core.DTOs;
using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Cove.Data;

using Microsoft.EntityFrameworkCore;

namespace AI.Audio;

internal sealed record AiAudioSimilarSceneDto(
    SceneDto Scene,
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
    CoveContext db,
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

    private readonly CoveContext _db = db;
    private readonly IEmbeddingService _embeddingService = embeddingService;
    private readonly IUserEngagementService? _engagementService = engagementService;
    private readonly ICurrentPrincipalAccessor? _principalAccessor = principalAccessor;

    private bool CanReadFiles => _principalAccessor?.Current?.Has(Permissions.FilesRead) == true;

    private bool HasUserScopedEngagement => _principalAccessor?.Current?.UserId != null;

    public async Task<PaginatedResponse<AiAudioSimilarSceneDto>> SimilarScenesForSceneAsync(int sceneId, int page = 1, int perPage = DefaultSimilarPerPage, CancellationToken ct = default)
    {
        var pageInfo = NormalizeSimilarPage(page, perPage);
        var queries = await LoadSimilarityQueriesAsync(sceneId, ct);
        if (queries.Count == 0)
        {
            return new PaginatedResponse<AiAudioSimilarSceneDto>([], 0, pageInfo.Page, pageInfo.PerPage);
        }

        var ranked = await SearchSimilarAsync(queries, sceneId, ct);
        var pageMatches = ranked
            .Skip((pageInfo.Page - 1) * pageInfo.PerPage)
            .Take(pageInfo.PerPage)
            .ToArray();
        var scenes = await BuildSceneResultsAsync(pageMatches, ct);
        var sceneById = scenes.ToDictionary(static scene => scene.Id);
        var items = pageMatches
            .Where(match => sceneById.ContainsKey(match.Embedding.HostId))
            .Select(match => new AiAudioSimilarSceneDto(
                sceneById[match.Embedding.HostId],
                match.Distance,
                match.Embedding.Kind,
                match.Embedding.KindFamily,
                match.Embedding.SectionIndex,
                match.Embedding.StartSec,
                match.Embedding.EndSec))
            .ToList();

        return new PaginatedResponse<AiAudioSimilarSceneDto>(items, ranked.Count, pageInfo.Page, pageInfo.PerPage);
    }

    private async Task<IReadOnlyList<AiAudioSimilarQuery>> LoadSimilarityQueriesAsync(int sceneId, CancellationToken ct)
    {
        var embeddings = await _db.Embeddings
            .AsNoTracking()
            .Where(embedding =>
                embedding.HostType == EmbeddingHostType.Scene &&
                embedding.HostId == sceneId &&
                embedding.SourceKey == AudioSourceKey &&
                embedding.Kind == AudioEmbeddingKind &&
                embedding.KindFamily == AudioEmbeddingKindFamily &&
                embedding.Modality == EmbeddingModality.Audio &&
                !embedding.IsSemantic)
            .OrderBy(static embedding => embedding.SectionIndex)
            .ThenBy(static embedding => embedding.StartSec ?? double.MaxValue)
            .ToListAsync(ct);

        var queries = new List<AiAudioSimilarQuery>();
        var asset = embeddings.FirstOrDefault(static embedding => embedding.SectionIndex == 0);
        if (asset is not null)
        {
            queries.Add(new AiAudioSimilarQuery(asset, TargetEmbeddingScope.Asset, AssetWeight));
        }

        foreach (var section in SelectRepresentativeEmbeddings(embeddings.Where(static embedding => embedding.SectionIndex > 0).ToArray(), MaxSectionQueries))
        {
            queries.Add(new AiAudioSimilarQuery(section, TargetEmbeddingScope.Section, SectionWeight));
        }

        return queries;
    }

    private async Task<IReadOnlyList<EmbeddingSearchResult>> SearchSimilarAsync(IReadOnlyList<AiAudioSimilarQuery> queries, int sceneId, CancellationToken ct)
    {
        var candidates = new Dictionary<int, SimilarCandidate>();
        foreach (var query in queries.Where(static query => query.Weight > 0d))
        {
            var matches = await _embeddingService.KnnAsync(
                query.Embedding.Vector,
                int.MaxValue,
                new EmbeddingSearchOptions
                {
                    HostType = EmbeddingHostType.Scene,
                    Kind = query.Embedding.Kind,
                    KindFamily = query.Embedding.KindFamily,
                    Modality = EmbeddingModality.Audio,
                    IsSemantic = false,
                    SourceKey = AudioSourceKey,
                },
                ct);

            var bestPerHost = matches
                .Where(match => match.Embedding.HostId != sceneId)
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

    private async Task<List<SceneDto>> BuildSceneResultsAsync(IReadOnlyList<EmbeddingSearchResult> ranked, CancellationToken ct)
    {
        var hostIds = ranked.Select(static match => match.Embedding.HostId).Distinct().ToArray();
        var scenes = await _db.Scenes
            .AsNoTracking()
            .Include(static scene => scene.Studio)
            .Include(static scene => scene.Urls)
            .Include(static scene => scene.SceneTags).ThenInclude(static sceneTag => sceneTag.Tag)
            .Include(static scene => scene.ScenePerformers).ThenInclude(static scenePerformer => scenePerformer.Performer)
            .Include(static scene => scene.SceneGalleries).ThenInclude(static sceneGallery => sceneGallery.Gallery)
            .Include(static scene => scene.GroupItems).ThenInclude(static groupItem => groupItem.Group)
            .Include(static scene => scene.Files)
            .AsSplitQuery()
            .Where(scene => hostIds.Contains(scene.Id))
            .ToDictionaryAsync(static scene => scene.Id, ct);

        var engagement = _engagementService == null
            ? []
            : await _engagementService.GetSceneSnapshotsAsync(hostIds, ct);

        var results = new List<SceneDto>(ranked.Count);
        foreach (var match in ranked)
        {
            if (scenes.TryGetValue(match.Embedding.HostId, out var scene))
            {
                results.Add(MapScene(scene, engagement.GetValueOrDefault(scene.Id), HasUserScopedEngagement));
            }
        }

        return results;
    }

    private SceneDto MapScene(Scene scene, UserEngagementSnapshot? engagement, bool preferUserSnapshot)
        => new(
            scene.Id,
            scene.Title,
            scene.Code,
            scene.Details,
            scene.Director,
            scene.Date?.ToString("yyyy-MM-dd"),
            scene.Organized,
            scene.IsVr,
            scene.StudioId,
            scene.Studio?.Name,
            scene.Captions,
            scene.InteractiveSpeed,
            scene.Urls.Select(static url => url.Url).ToList(),
            scene.SceneTags.Where(static sceneTag => sceneTag.Tag != null).Select(static sceneTag => new TagDto(sceneTag.Tag!.Id, sceneTag.Tag.Name, sceneTag.Tag.Description, sceneTag.Tag.Favorite, sceneTag.Tag.IgnoreAutoTag, [])).ToList(),
            scene.ScenePerformers.Where(static scenePerformer => scenePerformer.Performer != null).Select(static scenePerformer => MapPerformer(scenePerformer.Performer!)).ToList(),
            scene.Files.Select(file => new VideoFileDto(
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
            MapSceneGroups(scene),
            scene.SceneGalleries.Where(static sceneGallery => sceneGallery.Gallery != null).Select(static sceneGallery => new GallerySummaryDto(sceneGallery.Gallery!.Id, sceneGallery.Gallery.Title, sceneGallery.Gallery.Date?.ToString("yyyy-MM-dd"))).ToList(),
            [],
            scene.CustomFields,
            scene.CreatedAt.ToString("o"),
            scene.UpdatedAt.ToString("o"));

    private static PerformerSummaryDto MapPerformer(Performer performer)
        => new(
            performer.Id,
            performer.Name,
            performer.Disambiguation,
            performer.Gender?.ToString(),
            performer.Birthdate?.ToString("yyyy-MM-dd"),
            performer.Favorite,
            performer.ImageBlobId != null ? BuildEntityImageUrl($"/api/performers/{performer.Id}/image", performer.UpdatedAt) : null);

    private static List<GroupSummaryDto> MapSceneGroups(Scene scene)
        => scene.GroupItems
            .Where(static item => item.Kind == GroupItemKind.Scene && item.Group != null)
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
            if (_bestMatch is null || _totalWeight <= 0d)
            {
                return null;
            }

            return new EmbeddingSearchResult(_bestMatch.Embedding, (float)(_weightedDistance / _totalWeight));
        }
    }
}