using System.Net;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Auth;
using Cove.Core.Interfaces;
using Cove.Plugins;
using Cove.Sdk;

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;

namespace AI.Core;

public sealed class AiCoreExtension : FullExtensionBase, IPermissionContributor
{
    private const string SettingsStoreKey = "settings";
    public const string RunPermission = "cove.community.ai.core.runs.run";
    public const string ManageModelsPermission = "cove.community.ai.core.models.manage";
    public const string ManagePipelinesPermission = "cove.community.ai.core.pipelines.manage";
    public const string WriteSettingsPermission = "cove.community.ai.core.settings.write";

    private static readonly JsonSerializerOptions StateJson = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    private IServiceProvider? _services;

    public override string Id => "cove.community.ai.core";

    public override string Name => "AI Core";

    public override string Version => "0.0.2";

    public override string Description => "AI orchestration, model lifecycle management, and nsfw_ai_server v4 integration for Cove.";

    public override string Author => "Cove Team";

    public override string Url => "https://github.com/yourcove/AI.Extensions";

    public override IReadOnlyList<string> Categories =>
    [
        ExtensionCategories.Tools,
        ExtensionCategories.Automation,
        ExtensionCategories.Metadata,
        "ai",
    ];

    public override string MinCoveVersion => "0.1.0";

    public override void ConfigureServices(IServiceCollection services, ExtensionContext context)
    {
        services.AddHttpClient<INsfwAiServerClient, NsfwAiServerClient>(static client =>
        {
            client.Timeout = Timeout.InfiniteTimeSpan;
        });
        services.AddScoped<ITextEncoder, AiCoreSemanticTextEncoder>();
        services.AddScoped<IAiRunJournal, AiRunJournal>();
        services.AddScoped<IAiRunPlanner, AiRunPlanner>();
        services.AddScoped<IAiArtifactReplaceService, AiArtifactReplaceService>();
        services.AddScoped<IAiCoreOrchestrator, AiCoreOrchestrator>();
        services.AddScoped<IAiRunTargetResolver, AiRunTargetResolver>();
        services.AddScoped<IAiRunQueueService, AiRunQueueService>();
    }

    public override Task InitializeAsync(IServiceProvider services, CancellationToken ct = default)
    {
        _services = services;
        return Task.CompletedTask;
    }

    public override UIManifest GetUIManifest()
        => ManifestBuilder()
            .AddPage("ai", "AI", "AiCorePage", showInNav: true, navOrder: 95)
            .AddSettingsTab(
                "extensions/ai",
                "AI",
                order: 100,
                icon: "database",
                description: "Core AI extension settings and model orchestration defaults.",
                searchKeywords: ["ai", "model server", "nsfw", "path mappings", "models"],
                aliases: ["extensions-ai", "extensions/ai/core"])
            .AddSettingsPanel(new UISettingsPanel("ai-core-settings", "AI Core", Id, "AiCoreSettingsPanel", 40, TargetTab: "extensions/ai"))
            .AddAction("ai-core-run-video-toolbar", "Run AI", "toolbar", ["video"], icon: null, apiEndpoint: "/api/ext/ai-core/actions/run", handlerName: "openRunAiDialog", order: 20, requiredPermission: RunPermission, suppressSuccessAlert: true)
            .AddAction("ai-core-run-image-toolbar", "Run AI", "toolbar", ["image"], icon: null, apiEndpoint: "/api/ext/ai-core/actions/run", handlerName: "openRunAiDialog", order: 20, requiredPermission: RunPermission, suppressSuccessAlert: true)
            .AddAction("ai-core-run-videos-bulk", "Run AI", "bulk", ["video"], icon: null, apiEndpoint: "/api/ext/ai-core/actions/run", handlerName: "openRunAiDialog", order: 20, requiredPermission: RunPermission, suppressSuccessAlert: true)
            .AddAction("ai-core-run-images-bulk", "Run AI", "bulk", ["image"], icon: null, apiEndpoint: "/api/ext/ai-core/actions/run", handlerName: "openRunAiDialog", order: 20, requiredPermission: RunPermission, suppressSuccessAlert: true)
            .Build();

    public IEnumerable<PermissionDefinition> ContributePermissions()
    {
        var source = $"extension:{Id}";
        return
        [
            new(RunPermission, "AI Core", "Queue AI runs from Cove selections and direct run endpoints.", Dangerous: true, Implies: [Cove.Core.Auth.Permissions.JobsRun], Source: source, GrantToAdminsByDefault: true),
            new(ManageModelsPermission, "AI Core", "Load and unload AI server models.", Dangerous: true, Source: source, GrantToAdminsByDefault: true),
            new(ManagePipelinesPermission, "AI Core", "Create, update, delete, and sync custom AI server pipelines.", Dangerous: true, Source: source, GrantToAdminsByDefault: true),
            new(WriteSettingsPermission, "AI Core", "Change AI Core connection and run settings.", Dangerous: true, Source: source, GrantToAdminsByDefault: true),
        ];
    }

    protected override void DefineJobs()
    {
        Job(
            id: "run-selection",
            name: "Run AI",
            handler: RunSelectionJobAsync,
            description: "Runs the AI pipeline over selected Video/Image entities or explicit media paths.",
            supportsParameters: true);
    }

    public override void MapEndpoints(IEndpointRouteBuilder endpoints)
    {
        var group = endpoints.MapGroup("/api/ext/ai-core").WithTags("AI.Core");

        group.MapGet("/settings", async (CancellationToken ct) => Results.Ok(await LoadSettingsAsync(ct)));

        group.MapPut("/settings", async (AiCoreConnectionSettings settings, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, WriteSettingsPermission) is { } denied)
                return denied;

            try
            {
                var normalized = settings.Normalize();
                await SaveSettingsAsync(normalized, ct);
                return Results.Ok(normalized);
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapGet("/capabilities", async (IAiCoreOrchestrator orchestrator, CancellationToken ct) =>
        {
            var settings = await LoadSettingsAsync(ct);
            return Results.Ok(new
            {
                extensions = orchestrator.GetCapabilities(),
                customPipelines = settings.CustomPipelines,
                runPresets = settings.RunPresets,
                modelBindings = settings.CapabilityModelBindings,
            });
        });

        group.MapGet("/health", async (INsfwAiServerClient client, IAiCoreOrchestrator orchestrator, CancellationToken ct) =>
        {
            var settings = await LoadSettingsAsync(ct);
            try
            {
                var models = await client.GetModelCatalogAsync(settings, ct);
                return Results.Ok(new
                {
                    status = "reachable",
                    serverBaseUrl = settings.ServerBaseUrl,
                    modelCount = models.Count,
                    contributorCount = orchestrator.GetCapabilities().Count,
                    pathMappingCount = settings.PathMappings.Count,
                    maxInFlight = settings.MaxInFlight,
                });
            }
            catch (Exception ex)
            {
                return Results.Ok(new
                {
                    status = "unreachable",
                    serverBaseUrl = settings.ServerBaseUrl,
                    contributorCount = orchestrator.GetCapabilities().Count,
                    pathMappingCount = settings.PathMappings.Count,
                    maxInFlight = settings.MaxInFlight,
                    detail = ex.Message,
                });
            }
        });

        group.MapGet("/models/catalog", async (INsfwAiServerClient client, CancellationToken ct) =>
        {
            try
            {
                var settings = await LoadSettingsAsync(ct);
                var models = await client.GetModelCatalogAsync(settings, ct);
                return Results.Ok(new { models });
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapGet("/models/loaded", async (INsfwAiServerClient client, CancellationToken ct) =>
        {
            try
            {
                var settings = await LoadSettingsAsync(ct);
                var models = await client.GetLoadedModelsAsync(settings, ct);
                return Results.Ok(new { models });
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapPost("/models/load", async (AiModelSelectionRequest request, INsfwAiServerClient client, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, ManageModelsPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                var models = await client.LoadModelsAsync(settings, request, ct);
                return Results.Ok(new { models });
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapPost("/models/unload", async (AiModelSelectionRequest request, INsfwAiServerClient client, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, ManageModelsPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                var models = await client.UnloadModelsAsync(settings, request, ct);
                return Results.Ok(new { models });
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapGet("/pipelines/custom", async (CancellationToken ct) =>
        {
            var settings = await LoadSettingsAsync(ct);
            return Results.Ok(new { pipelines = settings.CustomPipelines });
        });

        group.MapPost("/pipelines/custom", async (AiCustomPipelineDefinition pipeline, INsfwAiServerClient client, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, ManagePipelinesPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                var normalized = pipeline.Normalize();
                var sync = await client.RegisterCustomPipelineAsync(settings, normalized, ct);
                var updated = UpsertCustomPipeline(settings, normalized).Normalize();
                await SaveSettingsAsync(updated, ct);
                return Results.Ok(new { pipeline = normalized, sync });
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapPut("/pipelines/custom/{pipelineName}", async (string pipelineName, AiCustomPipelineDefinition pipeline, INsfwAiServerClient client, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, ManagePipelinesPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                var normalized = (pipeline with { PipelineName = pipelineName }).Normalize();
                var sync = await client.RegisterCustomPipelineAsync(settings, normalized, ct);
                var updated = UpsertCustomPipeline(settings, normalized).Normalize();
                await SaveSettingsAsync(updated, ct);
                return Results.Ok(new { pipeline = normalized, sync });
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapPost("/pipelines/custom/{pipelineName}/sync", async (string pipelineName, INsfwAiServerClient client, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, ManagePipelinesPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                var pipeline = settings.CustomPipelines.FirstOrDefault(item => string.Equals(item.PipelineName, pipelineName, StringComparison.OrdinalIgnoreCase));
                if (pipeline is null)
                {
                    return Results.NotFound(new { message = $"Custom pipeline '{pipelineName}' was not found." });
                }

                var sync = await client.RegisterCustomPipelineAsync(settings, pipeline, ct);
                return Results.Ok(new { pipeline, sync });
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapDelete("/pipelines/custom/{pipelineName}", async (string pipelineName, INsfwAiServerClient client, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, ManagePipelinesPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                var sync = await client.DeleteCustomPipelineAsync(settings, pipelineName, ct);
                var updated = settings with
                {
                    CustomPipelines = settings.CustomPipelines
                        .Where(item => !string.Equals(item.PipelineName, pipelineName, StringComparison.OrdinalIgnoreCase))
                        .ToList(),
                };
                await SaveSettingsAsync(updated.Normalize(), ct);
                return Results.Ok(new { pipelineName, sync });
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapPost("/run/images", async (AiRunImagesRequest request, IAiCoreOrchestrator orchestrator, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, RunPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                return Results.Ok(await orchestrator.RunImagesAsync(settings, request, ct));
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapPost("/run/video", async (AiRunVideoRequest request, IAiCoreOrchestrator orchestrator, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, RunPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                return Results.Ok(await orchestrator.RunVideoAsync(settings, request, ct));
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapPost("/run/audio", async (AiRunAudioRequest request, IAiCoreOrchestrator orchestrator, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, RunPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                return Results.Ok(await orchestrator.RunAudioAsync(settings, request, ct));
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapPost("/jobs/queue", async (AiQueueRunRequest request, IAiRunQueueService runQueueService, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, RunPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                var queued = await runQueueService.QueueAsync(settings, request, ct);
                return Results.Accepted($"/api/jobs/{queued.JobId}", queued);
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });

        group.MapPost("/actions/run", async (JsonElement payload, IAiRunQueueService runQueueService, ICurrentPrincipalAccessor principalAccessor, CancellationToken ct) =>
        {
            if (RequirePermission(principalAccessor, RunPermission) is { } denied)
                return denied;

            try
            {
                var settings = await LoadSettingsAsync(ct);
                var queued = await runQueueService.QueueAsync(settings, AiQueueRunRequest.FromActionPayload(payload), ct);
                return Results.Accepted($"/api/jobs/{queued.JobId}", queued);
            }
            catch (Exception ex)
            {
                return ToProblem(ex);
            }
        });
    }

    private async Task<AiCoreConnectionSettings> LoadSettingsAsync(CancellationToken ct)
    {
        var payload = await Store.GetAsync(SettingsStoreKey, ct);
        if (string.IsNullOrWhiteSpace(payload))
        {
            return new AiCoreConnectionSettings().Normalize();
        }

        var settings = JsonSerializer.Deserialize<AiCoreConnectionSettings>(payload, StateJson) ?? new AiCoreConnectionSettings();
        return settings.Normalize();
    }

    private Task SaveSettingsAsync(AiCoreConnectionSettings settings, CancellationToken ct)
        => Store.SetAsync(SettingsStoreKey, JsonSerializer.Serialize(settings, StateJson), ct);

    private static AiCoreConnectionSettings UpsertCustomPipeline(AiCoreConnectionSettings settings, AiCustomPipelineDefinition pipeline)
    {
        var pipelines = settings.CustomPipelines
            .Where(item => !string.Equals(item.PipelineName, pipeline.PipelineName, StringComparison.OrdinalIgnoreCase))
            .ToList();
        pipelines.Add(pipeline);
        return settings with { CustomPipelines = pipelines };
    }

    private static IResult? RequirePermission(ICurrentPrincipalAccessor principalAccessor, string permission)
    {
        var principal = principalAccessor.Current;
        if (principal?.Has(permission) == true)
            return null;

        if (principal is null || principal.Kind == PrincipalKind.Anonymous)
            return Results.Unauthorized();

        return Results.Json(
            new { code = "FORBIDDEN", message = $"Permission '{permission}' is required." },
            statusCode: StatusCodes.Status403Forbidden);
    }

    private async Task RunSelectionJobAsync(IReadOnlyDictionary<string, string>? parameters, Cove.Plugins.IJobProgress progress, CancellationToken ct)
    {
        var services = _services ?? throw new InvalidOperationException("AI.Core has not been initialized.");

        using var scope = services.CreateScope();
        var runQueueService = scope.ServiceProvider.GetRequiredService<IAiRunQueueService>();
        var settings = await LoadSettingsAsync(ct);
        await runQueueService.ExecuteAsync(settings, AiQueueRunRequest.FromJobParameters(parameters), new PluginJobProgressAdapter(progress), ct);
    }

    private sealed class PluginJobProgressAdapter(Cove.Plugins.IJobProgress inner) : Cove.Core.Interfaces.IJobProgress
    {
        public void Report(double progress, string? subTask = null) => inner.Report(progress, subTask);
    }

    private static IResult ToProblem(Exception exception)
    {
        var statusCode = exception switch
        {
            ArgumentException => StatusCodes.Status400BadRequest,
            InvalidOperationException => StatusCodes.Status409Conflict,
            HttpRequestException httpException when httpException.StatusCode is HttpStatusCode status => (int)status,
            _ => StatusCodes.Status500InternalServerError,
        };

        return Results.Problem(
            title: "AI.Core request failed",
            detail: exception.Message,
            statusCode: statusCode);
    }
}
