using AI.Extensions.Abstractions;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Cove.Plugins;
using Cove.Sdk;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace AI.Audio;

public sealed class AiAudioExtension : FullExtensionBase
{
    public override string Id => "cove.community.ai.audio";

    public override string Name => "AI Audio";

    public override string Version => "0.1.0";

    public override string Description => "Contributes audio embedding and classification claims for AI workflows.";

    public override string Author => "skier233";

    public override string Url => "https://github.com/skier233/AI.Extensions";

    public override string MinCoveVersion => "0.6.0";

    public override IReadOnlyList<string> Categories =>
    [
        ExtensionCategories.Metadata,
        ExtensionCategories.Automation,
        "ai",
        "audio",
    ];

    public override IReadOnlyDictionary<string, string> Dependencies => new Dictionary<string, string>
    {
        ["cove.community.ai.core"] = ">=0.1.0",
    };

    public override UIManifest GetUIManifest()
        => ManifestBuilder()
            .AddFeature("audio-similarity", new Dictionary<string, string>
            {
                ["apiBasePath"] = "/api/ext/ai-audio",
            })
            .Build();

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        services.AddSingleton<AiAudioPreparationService>();
        services.AddSingleton<AiAudioPersistenceService>();
        services.AddScoped<AiAudioSimilarityService>();
        services.AddSingleton<IAiCapabilityContributor, AiAudioContributor>();
    }

    public override Task InitializeAsync(IServiceProvider services, CancellationToken ct = default)
    {
        PublishContributions<IAiCapabilityContributor>(services);
        return Task.CompletedTask;
    }

    public override void MapEndpoints(IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("/api/ext/ai-audio").WithTags("AI.Audio");

        group.MapGet("/videos/{videoId:int}/similar-videos", async (int videoId, int? page, int? perPage, AiAudioSimilarityService searchService, CancellationToken ct) =>
            Results.Ok(await searchService.SimilarVideosForVideoAsync(videoId, page ?? 1, perPage ?? 12, ct)));

        // Cheap "does this video have audio embeddings?" check so the UI can decide whether to show the
        // audio-similarity tab without running a full (slow) similarity search.
        group.MapGet("/videos/{videoId:int}/has-embeddings", async (int videoId, AiAudioSimilarityService searchService, CancellationToken ct) =>
            Results.Ok(new { hasEmbeddings = await searchService.HasAudioEmbeddingsAsync(videoId, ct) }));
    }
}

internal sealed class AiAudioContributor(
    AiAudioPreparationService preparationService,
    AiAudioPersistenceService persistenceService,
    ILogger<AiAudioContributor> logger) : IAiCapabilityContributor
{
    private readonly AiAudioPreparationService _preparationService = preparationService;
    private readonly AiAudioPersistenceService _persistenceService = persistenceService;
    private readonly ILogger<AiAudioContributor> _logger = logger;

    private static readonly AiCapabilityDescriptor Descriptor = new(
        "cove.community.ai.audio",
        "AI Audio",
        [
            new AiCapabilityClaim(
                "audio.asset.embedding",
                "Audio Embeddings",
                AiMediaKinds.Audio,
                "embedding",
                "asset",
                "embeddings",
                PreferredModels: ["audioembed"],
                Description: "Extract audio embeddings for speaker and similarity workflows.")
            {
                CapabilityId = "audio.embedding",
                ModelBindingSlotId = "embedder",
            },
            new AiCapabilityClaim(
                "audio.asset.classification",
                "Audio Classification",
                AiMediaKinds.Audio,
                "classification",
                "asset",
                "categories",
                PreferredModels: ["audioclass"],
                Description: "Classify audio content for filtering and downstream routing.")
            {
                CapabilityId = "audio.classification",
                ModelBindingSlotId = "classifier",
            },
        ])
    {
        Capabilities =
        [
            new AiCapabilityFeature(
                "audio.embedding",
                "Audio Embeddings",
                ["audio.asset.embedding"],
                [
                    new AiModelBindingSlot(
                        "embedder",
                        "Audio embedding model",
                        "embedding",
                        RequiredCapabilities: ["embedding"],
                        RequiredScopes: ["asset"],
                        RequiredCategories: ["audio_embeddings_audioembed"],
                        DefaultModels: ["audioembed"]),
                ],
                "Extract embeddings for speaker and audio similarity workflows."),
            new AiCapabilityFeature(
                "audio.classification",
                "Audio Classification",
                ["audio.asset.classification"],
                [
                    new AiModelBindingSlot(
                        "classifier",
                        "Audio classification model",
                        "classification",
                        RequiredCapabilities: ["classification"],
                        RequiredScopes: ["asset"],
                        RequiredCategories: ["audio_classification_audioclass"],
                        DefaultModels: ["audioclass"]),
                ],
                "Classify audio content for filtering and downstream routing."),
        ],
    };

    public AiCapabilityDescriptor Describe() => Descriptor;

    public async Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default)
    {
        _logger.LogInformation(
            "AI.Audio dispatch starting for run {RunId} host {HostType}:{HostId} media {MediaKind} claims {ClaimIds}. Windows={WindowCount}, rawEmbeddings={RawEmbeddingCount}, rawClassifications={RawClassificationCount}.",
            request.Context.RunId,
            request.Context.HostEntityType ?? "<none>",
            request.Context.HostEntityId,
            request.Result.MediaKind,
            string.Join(", ", request.Claims.Select(static claim => claim.ClaimId)),
            request.Result.Windows.Count,
            request.Result.Windows.Sum(static window => window.Analysis.Embeddings.Count),
            request.Result.Windows.Sum(static window => window.Analysis.Classifications.Count));

        var batch = _preparationService.Prepare(request);
        _logger.LogInformation(
            "AI.Audio prepared {EmbeddingCount} embedding(s), {SegmentCount} segment(s), and {NoteCount} note(s) for run {RunId}.",
            batch.Embeddings.Count,
            batch.Segments.Count,
            batch.Notes.Count,
            request.Context.RunId);

        if (batch.Embeddings.Count == 0 || batch.Segments.Count == 0)
        {
            _logger.LogInformation(
                "AI.Audio run {RunId} window summary: {WindowSummary}",
                request.Context.RunId,
                BuildWindowSummary(request.Result.Windows));
        }

        if (batch.Notes.Count > 0)
        {
            _logger.LogInformation(
                "AI.Audio preparation notes for run {RunId}: {Notes}",
                request.Context.RunId,
                string.Join(" | ", batch.Notes));
        }

        var notes = new List<string>(batch.Notes);
        notes.AddRange(await _persistenceService.PersistAsync(request, batch, ct));

        _logger.LogInformation(
            "AI.Audio dispatch finished for run {RunId}. Persistence notes: {Notes}",
            request.Context.RunId,
            string.Join(" | ", notes));

        return new AiDispatchResult(
            Descriptor.ExtensionId,
            request.Claims.Count,
            batch.ToPreparedCounts(),
            notes);
    }

    private static string BuildWindowSummary(IReadOnlyList<AiTemporalSlice> windows)
    {
        if (windows.Count == 0)
        {
            return "<no windows>";
        }

        var preview = windows
            .Take(12)
            .Select(window =>
            {
                var index = window.Index?.ToString(System.Globalization.CultureInfo.InvariantCulture) ?? "?";
                var start = window.StartSeconds?.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture) ?? "?";
                var end = window.EndSeconds?.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture) ?? "?";
                return $"#{index}({start}-{end}) emb={window.Analysis.Embeddings.Count} cls={window.Analysis.Classifications.Count}";
            });

        var summary = string.Join("; ", preview);
        return windows.Count > 12
            ? $"{summary}; +{windows.Count - 12} more"
            : summary;
    }
}

