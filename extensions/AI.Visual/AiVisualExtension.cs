using AI.Extensions.Abstractions;

using System.Text.Json;
using System.Text.Json.Serialization;

using Cove.Core.Interfaces;
using Cove.Plugins;
using Cove.Sdk;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Visual;

public sealed class AiVisualExtension : FullExtensionBase
{
    public override string Id => "cove.ai.visual";

    public override string Name => "AI Visual";

    public override string Version => "0.1.0";

    public override string Description => "Contributes general visual detection claims for image and video AI similarity and recommendation systems.";

    public override string Author => "Cove Team";

    public override string Url => "https://github.com/yourcove/AI.Extensions";

    public override string MinCoveVersion => "0.0.10";

    public override IReadOnlyList<string> Categories =>
    [
        ExtensionCategories.Metadata,
        ExtensionCategories.Automation,
        "ai",
        "visual",
    ];

    public override IReadOnlyDictionary<string, string> Dependencies => new Dictionary<string, string>
    {
        ["cove.ai.core"] = ">=0.1.0",
    };

    public override UIManifest GetUIManifest()
        => ManifestBuilder()
            .AddFeature("visual-similarity", new Dictionary<string, string>
            {
                ["apiBasePath"] = "/api/ext/ai-visual",
            })
            .Build();

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        services.AddSingleton<AiVisualPreparationService>();
        services.AddSingleton<AiVisualPersistenceService>();
        services.AddSingleton<AiVisualLocalTextEncoder>();
        services.AddScoped<AiVisualSemanticSearchService>();
        services.AddSingleton<IAiCapabilityContributor, AiVisualContributor>();
    }

    public override void MapEndpoints(IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("/api/ext/ai-visual").WithTags("AI.Visual");

        group.MapPost("/videos/search", async (HttpRequest httpRequest, AiVisualSemanticSearchService searchService, CancellationToken ct) =>
        {
            try
            {
                var request = await AiVisualSearchRequestReader.ReadAsync<VideoFilter>(httpRequest, ct);
                return Results.Ok(await searchService.SearchVideosAsync(request, ct));
            }
            catch (ArgumentException ex)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
            catch (InvalidOperationException ex)
            {
                return Results.Conflict(new { error = ex.Message });
            }
            catch (JsonException ex)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
        });

        group.MapPost("/images/search", async (HttpRequest httpRequest, AiVisualSemanticSearchService searchService, CancellationToken ct) =>
        {
            try
            {
                var request = await AiVisualSearchRequestReader.ReadAsync<ImageFilter>(httpRequest, ct);
                return Results.Ok(await searchService.SearchImagesAsync(request, ct));
            }
            catch (ArgumentException ex)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
            catch (InvalidOperationException ex)
            {
                return Results.Conflict(new { error = ex.Message });
            }
            catch (JsonException ex)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
        });

        group.MapGet("/videos/{videoId:int}/similar-videos", async (int videoId, int? page, int? perPage, AiVisualSemanticSearchService searchService, CancellationToken ct) =>
            Results.Ok(await searchService.SimilarVideosForVideoAsync(videoId, page ?? 1, perPage ?? 12, ct)));

        group.MapGet("/videos/{videoId:int}/similar-images", async (int videoId, int? page, int? perPage, AiVisualSemanticSearchService searchService, CancellationToken ct) =>
            Results.Ok(await searchService.SimilarImagesForVideoAsync(videoId, page ?? 1, perPage ?? 12, ct)));

        group.MapGet("/images/{imageId:int}/similar-videos", async (int imageId, int? page, int? perPage, AiVisualSemanticSearchService searchService, CancellationToken ct) =>
            Results.Ok(await searchService.SimilarVideosForImageAsync(imageId, page ?? 1, perPage ?? 12, ct)));

        group.MapGet("/images/{imageId:int}/similar-images", async (int imageId, int? page, int? perPage, AiVisualSemanticSearchService searchService, CancellationToken ct) =>
            Results.Ok(await searchService.SimilarImagesForImageAsync(imageId, page ?? 1, perPage ?? 12, ct)));

        group.MapGet("/videos/{videoId:int}/similar-videos/segment", async (int videoId, double? startSec, double? endSec, int? page, int? perPage, AiVisualSemanticSearchService searchService, CancellationToken ct) =>
        {
            if (startSec is null)
            {
                return Results.BadRequest(new { error = "startSec is required." });
            }

            try
            {
                return Results.Ok(await searchService.SimilarVideosForVideoSegmentAsync(videoId, startSec.Value, endSec, page ?? 1, perPage ?? 12, ct));
            }
            catch (ArgumentException ex)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
        });

        group.MapPost("/videos/{videoId:int}/similar-videos/segment", async (int videoId, HttpRequest httpRequest, AiVisualSemanticSearchService searchService, CancellationToken ct) =>
        {
            try
            {
                var request = await AiVisualSearchRequestReader.ReadSegmentSimilarityAsync(httpRequest, ct);
                return Results.Ok(await searchService.SimilarVideosForVideoSegmentAsync(
                    videoId,
                    request.Intervals ?? [],
                    request.Page ?? 1,
                    request.PerPage ?? 12,
                    ct));
            }
            catch (ArgumentException ex)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
            catch (JsonException ex)
            {
                return Results.BadRequest(new { error = ex.Message });
            }
        });
    }
}

internal sealed record AiVisualSegmentSimilarityRequest(
    IReadOnlyList<AiVisualSegmentInterval>? Intervals,
    int? Page,
    int? PerPage);

internal static class AiVisualSearchRequestReader
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        Converters =
        {
            new JsonStringEnumConverter(JsonNamingPolicy.CamelCase),
        },
    };

    public static async Task<AiVisualSemanticSearchRequest<TFilter>> ReadAsync<TFilter>(HttpRequest request, CancellationToken ct)
        where TFilter : class
        => await JsonSerializer.DeserializeAsync<AiVisualSemanticSearchRequest<TFilter>>(request.Body, JsonOptions, ct)
           ?? new AiVisualSemanticSearchRequest<TFilter>();

    public static async Task<AiVisualSegmentSimilarityRequest> ReadSegmentSimilarityAsync(HttpRequest request, CancellationToken ct)
        => await JsonSerializer.DeserializeAsync<AiVisualSegmentSimilarityRequest>(request.Body, JsonOptions, ct)
           ?? new AiVisualSegmentSimilarityRequest(null, null, null);
}

internal sealed class AiVisualContributor(
    AiVisualPreparationService preparationService,
    AiVisualPersistenceService persistenceService) : IAiCapabilityContributor
{
    private readonly AiVisualPreparationService _preparationService = preparationService;
    private readonly AiVisualPersistenceService _persistenceService = persistenceService;

    private static readonly AiCapabilityDescriptor Descriptor = new(
        "cove.ai.visual",
        "AI Visual",
        [
            new AiCapabilityClaim(
                "visual.image.feature",
                "Image Feature Embeddings",
                AiMediaKinds.Image,
                "embedding",
                "asset",
                "embeddings",
                PreferredModels: ["visual"],
                Description: "Generate self-supervised feature embeddings for still images.")
            {
                CapabilityId = "visual.feature",
                ModelBindingSlotId = "embedder",
            },
            new AiCapabilityClaim(
                "visual.image.semantic",
                "Image Semantic Embeddings",
                AiMediaKinds.Image,
                "embedding",
                "asset",
                "embeddings",
                PreferredModels: ["semvisual"],
                Description: "Generate semantic image embeddings that can pair with a text encoder.")
            {
                CapabilityId = "visual.semantic",
                ModelBindingSlotId = "embedder",
            },
            new AiCapabilityClaim(
                "visual.video.feature",
                "Video Feature Embeddings",
                AiMediaKinds.Video,
                "embedding",
                "frame",
                "frames",
                PreferredModels: ["visual"],
                Description: "Generate self-supervised feature embeddings for sampled video frames.")
            {
                CapabilityId = "visual.feature",
                ModelBindingSlotId = "embedder",
            },
            new AiCapabilityClaim(
                "visual.video.semantic",
                "Video Semantic Embeddings",
                AiMediaKinds.Video,
                "embedding",
                "frame",
                "frames",
                PreferredModels: ["semvisual"],
                Description: "Generate semantic video embeddings and section centroids.")
            {
                CapabilityId = "visual.semantic",
                ModelBindingSlotId = "embedder",
            },
        ])
    {
        Capabilities =
        [
            new AiCapabilityFeature(
                "visual.feature",
                "Visual Feature Embeddings",
                ["visual.image.feature", "visual.video.feature"],
                [
                    new AiModelBindingSlot(
                        "embedder",
                        "Feature embedding model",
                        "embedding",
                        RequiredCapabilities: ["embedding"],
                        RequiredScopes: ["asset", "frame"],
                        RequiredCategories: ["visual_embeddings_visual"],
                        DefaultModels: ["visual"]),
                ],
                "Generate feature embeddings for similarity and recommendation systems."),
            new AiCapabilityFeature(
                "visual.semantic",
                "Semantic Visual Search",
                ["visual.image.semantic", "visual.video.semantic"],
                [
                    new AiModelBindingSlot(
                        "embedder",
                        "Semantic embedding model",
                        "embedding",
                        RequiredCapabilities: ["embedding"],
                        RequiredScopes: ["asset", "frame"],
                        RequiredCategories: ["visual_embeddings_semvisual"],
                        DefaultModels: ["semvisual"]),
                ],
                "Generate CLIP-compatible embeddings for semantic search."),
        ],
    };

    public AiCapabilityDescriptor Describe() => Descriptor;

    public async Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default)
    {
        var batch = _preparationService.Prepare(request);
        var notes = new List<string>(batch.Notes);
        notes.AddRange(await _persistenceService.PersistAsync(request, batch, ct));

        return new AiDispatchResult(
            Descriptor.ExtensionId,
            request.Claims.Count,
            batch.ToPreparedCounts(),
            notes);
    }
}

