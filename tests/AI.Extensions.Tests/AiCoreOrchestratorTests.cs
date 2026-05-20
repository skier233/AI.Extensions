using System.Text.Json;

using AI.Core;
using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Data;

using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Pgvector;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiCoreOrchestratorTests
{
    [Fact]
    public async Task RunImagesAsync_UsesConfiguredTaggingModelPerCategory()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels =
            [
                CreateTaggingModel("tagger-actions-old", ["Actions"]),
                CreateTaggingModel("tagger-actions-best", ["Actions", "Pose"]),
                CreateTaggingModel("tagger-pose-fast", ["Pose"]),
                CreateTaggingModel("tagger-body", ["Body"]),
            ],
        };
        var orchestrator = CreateOrchestrator(client, CreateTaggingContributor("tagging.image.asset", AiMediaKinds.Image, "asset"));

        await orchestrator.RunImagesAsync(
            new AiCoreConnectionSettings
            {
                TaggingModelPreferences =
                [
                    new AiTaggingModelPreference { Scope = "asset", Category = "Actions", Model = "tagger-actions-best" },
                    new AiTaggingModelPreference { Scope = "asset", Category = "Pose", Model = "tagger-actions-best" },
                    new AiTaggingModelPreference { Scope = "asset", Category = "Body", Model = "tagger-body" },
                ],
            }.Normalize(),
            new AiRunImagesRequest
            {
                Paths = ["E:/media/example.jpg"],
                ClaimIds = ["tagging.image.asset"],
                DispatchResults = false,
            });

        var request = Assert.IsType<ImageAnalyzeRequest>(client.LastAnalyzeRequest);
        var want = Assert.Single(request.Want ?? []);

        Assert.Equal("tagging", want.Capability);
        Assert.Equal("asset", want.Scope);
        Assert.Equal(["tagger-actions-best", "tagger-body"], want.Models);
    }

    [Fact]
    public async Task RunImagesAsync_FallsBackToSingleTaggingModelPerCategory()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels =
            [
                CreateTaggingModel("tagger-actions-slow", ["Actions"], loaded: true),
                CreateTaggingModel("tagger-actions-pinned", ["Actions"], loaded: true, pinned: true),
                CreateTaggingModel("tagger-body", ["Body"], loaded: true),
            ],
        };
        var orchestrator = CreateOrchestrator(client, CreateTaggingContributor("tagging.image.asset", AiMediaKinds.Image, "asset"));

        await orchestrator.RunImagesAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunImagesRequest
            {
                Paths = ["E:/media/example.jpg"],
                ClaimIds = ["tagging.image.asset"],
                DispatchResults = false,
            });

        var request = Assert.IsType<ImageAnalyzeRequest>(client.LastAnalyzeRequest);
        var want = Assert.Single(request.Want ?? []);

        Assert.Equal(["tagger-actions-pinned", "tagger-body"], want.Models);
        Assert.DoesNotContain("tagger-actions-slow", want.Models ?? []);
    }

    [Fact]
    public async Task RunVideoAsync_PersistsAiRunForResolvedHostEntity()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var client = new RecordingAiServerClient
        {
                        CatalogModels = [CreateTaggingModel("tagger-actions-best", ["Actions"], scope: "frame")],
                        EchoRequestedVideoModels = true,
            VideoAnalyzeResponse = JsonDocument.Parse("""
            {
              "asset_id": "E:/media/example.mp4",
              "duration_seconds": 12.5,
              "frame_interval_seconds": 2.0,
              "models": [
                {
                  "config_name": "tagger-actions-best",
                  "name": "tagger-actions-best",
                  "categories": ["Actions"],
                  "capabilities": ["tagging"],
                  "supported_scopes": ["frame"]
                }
              ]
            }
            """).RootElement.Clone(),
        };
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, client, CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"));

        var response = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                FrameInterval = 2.0,
                DispatchResults = false,
            });

        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var run = await db.AiRuns.SingleAsync();

        Assert.Equal(response.RunId, run.RunKey);
        Assert.Equal("ext:ai.core", run.SourceKey);
        Assert.Equal(AiRunTargetType.Scene, run.TargetType);
        Assert.Equal(42, run.TargetId);
        Assert.Equal(AiRunStatus.Completed, run.Status);
        Assert.Equal(2.0, run.FrameIntervalSec);
        Assert.NotNull(run.Models);
        Assert.NotNull(run.Summary);
    }

    [Fact]
    public async Task RecordFailureAsync_WithCancelledTokenMarksRunCancelled()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var journal = scope.ServiceProvider.GetRequiredService<IAiRunJournal>();

        await journal.RecordStartAsync(new AiRunJournalStart(
            "cancelled-run",
            "scene",
            42,
            "AI.Core",
            "load_or_fail",
            10,
            false,
            new { path = "E:/media/example.mp4" }));

        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await journal.RecordFailureAsync("cancelled-run", new OperationCanceledException(), cancelled.Token);

        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var run = await db.AiRuns.SingleAsync();
        Assert.Equal(AiRunStatus.Cancelled, run.Status);
        Assert.NotNull(run.CompletedAt);
    }

    [Fact]
    public async Task RunVideoAsync_SkipsSatisfiedTaggingRunWithoutCallingServerAgain()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        db.Tags.Add(new Tag { Id = 7, Name = "Action", SortName = "Action" });
        await db.SaveChangesAsync();

        var client = CreateVideoTaggingClient("tagger-actions-best", "Actions");
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, client, CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"));

        var first = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = true,
            });

        db.TagApplications.Add(new TagApplication
        {
            HostType = AffinityHostType.Scene,
            HostId = 42,
            TagId = 7,
            SourceKey = "ext:ai.tagging",
            SourceRunId = first.RunId,
            ModelKey = "Actions",
            Confidence = 0.9f,
        });
        await db.SaveChangesAsync();

        var second = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = true,
            });

        Assert.Equal(1, client.AnalyzeVideoCallCount);
        Assert.Equal(AiRunPlanDecision.Skip, Assert.Single(second.Plan).Decision);
        Assert.Equal("skipped", second.Analysis.GetProperty("status").GetString());
    }

    [Fact]
    public async Task RunVideoAsync_SkipsSatisfiedTaggingRunHistoryWithoutPersistedArtifacts()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();

        var client = CreateVideoTaggingClient("tagger-actions-best", "Actions");
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, client, CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"));

        await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = false,
            });

        var second = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = false,
            });

        Assert.Equal(1, client.AnalyzeVideoCallCount);
        Assert.Equal(AiRunPlanDecision.Skip, Assert.Single(second.Plan).Decision);
        Assert.Equal("skipped", second.Analysis.GetProperty("status").GetString());
    }

    [Fact]
    public async Task RunVideoAsync_SkipsSatisfiedFaceDetectionAndEmbeddingRunWithoutCallingServer()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        db.Faces.Add(new Face { Id = 100, Label = "Existing face", PrimarySourceKey = "ext:ai.faces" });
        db.FaceAppearances.Add(new FaceAppearance
        {
            FaceId = 100,
            HostType = FaceAppearanceHostType.Scene,
            HostId = 42,
            SourceKey = "ext:ai.faces",
            Payload = JsonDocument.Parse("""{"modelKey":"face_detector_torchexport"}"""),
            SampleCount = 1,
        });
        db.Embeddings.Add(new Embedding
        {
            HostType = EmbeddingHostType.Face,
            HostId = 100,
            Kind = "face",
            KindFamily = "face",
            Modality = EmbeddingModality.Face,
            Dim = 2,
            Vector = new Vector(new float[] { 0.1f, 0.2f }),
            SourceKey = "ext:ai.faces",
            Meta = JsonDocument.Parse("""{"modelKey":"face_embedding_torchexport"}"""),
        });
        await db.SaveChangesAsync();

        var client = new RecordingAiServerClient();
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, client, CreateFacesContributor());

        var result = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["faces.video.detection", "faces.video.embedding"],
                DispatchResults = true,
            });

        Assert.Equal(0, client.AnalyzeVideoCallCount);
        Assert.All(result.Plan, plan => Assert.Equal(AiRunPlanDecision.Skip, plan.Decision));
        Assert.Equal("skipped", result.Analysis.GetProperty("status").GetString());
    }

    [Fact]
    public async Task RunVideoAsync_SkipsSatisfiedFaceDetectionAndEmbeddingRunHistoryWithoutPersistedArtifacts()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();

        var client = new RecordingAiServerClient
        {
            CatalogModels =
            [
                CreateModel("face_detector_torchexport", ["face_detections"], "detection", "frame"),
                CreateModel("face_embedding_torchexport", ["face_embeddings"], "embedding", "region"),
            ],
            EchoRequestedVideoModels = true,
        };
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, client, CreateFacesContributor());

        await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["faces.video.detection", "faces.video.embedding"],
                DispatchResults = false,
            });

        var second = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["faces.video.detection", "faces.video.embedding"],
                DispatchResults = false,
            });

        Assert.Equal(1, client.AnalyzeVideoCallCount);
        Assert.All(second.Plan, plan => Assert.Equal(AiRunPlanDecision.Skip, plan.Decision));
        Assert.Equal("skipped", second.Analysis.GetProperty("status").GetString());
    }

    [Fact]
    public async Task RunVideoAsync_ForceClaimOverrideRerunsAndCreatesNewAiRun()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        db.Tags.Add(new Tag { Id = 8, Name = "Action", SortName = "Action" });
        await db.SaveChangesAsync();

        var client = CreateVideoTaggingClient("tagger-actions-best", "Actions");
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, client, CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"));

        var first = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = true,
            });

        db.TagApplications.Add(new TagApplication
        {
            HostType = AffinityHostType.Scene,
            HostId = 42,
            TagId = 8,
            SourceKey = "ext:ai.tagging",
            SourceRunId = first.RunId,
            ModelKey = "Actions",
            Confidence = 0.9f,
        });
        await db.SaveChangesAsync();

        var second = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                ForceClaimIds = ["tagging.video.frame"],
                DispatchResults = true,
            });

        Assert.Equal(2, client.AnalyzeVideoCallCount);
        Assert.Equal(AiRunPlanDecision.Rerun, Assert.Single(second.Plan).Decision);
        Assert.Equal(2, await db.AiRuns.CountAsync());
    }

    [Fact]
    public async Task RunVideoAsync_ChangingPreferredTaggingModelRerunsOnlyTheChangedCategory()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        db.Tags.Add(new Tag { Id = 9, Name = "Action", SortName = "Action" });
        await db.SaveChangesAsync();

        var client = new RecordingAiServerClient
        {
            CatalogModels =
            [
                CreateTaggingModel("tagger-actions-v1", ["Actions"], scope: "frame"),
                CreateTaggingModel("tagger-actions-v2", ["Actions"], scope: "frame"),
                CreateTaggingModel("tagger-body", ["Body"], scope: "frame"),
            ],
                        EchoRequestedVideoModels = true,
        };
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, client, CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"));

        var initialSettings = new AiCoreConnectionSettings
        {
            TaggingModelPreferences =
            [
                new AiTaggingModelPreference { Scope = "frame", Category = "Actions", Model = "tagger-actions-v1" },
                new AiTaggingModelPreference { Scope = "frame", Category = "Body", Model = "tagger-body" },
            ],
        }.Normalize();

        var first = await orchestrator.RunVideoAsync(
            initialSettings,
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = true,
            });

        db.TagApplications.AddRange(
            new TagApplication
            {
                HostType = AffinityHostType.Scene,
                HostId = 42,
                TagId = 9,
                SourceKey = "ext:ai.tagging",
                SourceRunId = first.RunId,
                ModelKey = "Actions",
                Confidence = 0.9f,
            },
            new TagApplication
            {
                HostType = AffinityHostType.Scene,
                HostId = 42,
                TagId = 9,
                SourceKey = "ext:ai.tagging",
                SourceRunId = first.RunId,
                ModelKey = "Body",
                Confidence = 0.9f,
            });
        await db.SaveChangesAsync();

        var rerunSettings = new AiCoreConnectionSettings
        {
            TaggingModelPreferences =
            [
                new AiTaggingModelPreference { Scope = "frame", Category = "Actions", Model = "tagger-actions-v2" },
                new AiTaggingModelPreference { Scope = "frame", Category = "Body", Model = "tagger-body" },
            ],
        }.Normalize();

        var second = await orchestrator.RunVideoAsync(
            rerunSettings,
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "scene",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = true,
            });

        var analyzeRequest = Assert.IsType<VideoAnalyzeRequest>(client.LastAnalyzeRequest);
        var want = Assert.Single(analyzeRequest.Want ?? []);

        Assert.Equal(["tagger-actions-v2"], want.Models);
        Assert.Equal(AiRunPlanDecision.Rerun, Assert.Single(second.Plan).Decision);
        Assert.Equal(0, await db.TagApplications.CountAsync(application => application.SourceRunId == first.RunId && application.ModelKey == "Actions"));
        Assert.Equal(1, await db.TagApplications.CountAsync(application => application.SourceRunId == first.RunId && application.ModelKey == "Body"));
    }

    private static AiCoreOrchestrator CreateOrchestrator(INsfwAiServerClient client, params IAiCapabilityContributor[] contributors)
        => new(
            client,
            contributors,
            NoOpAiRunJournal.Instance,
            NoOpAiRunPlanner.Instance,
            NoOpAiArtifactReplaceService.Instance,
            NullLogger<AiCoreOrchestrator>.Instance);

    private static AiCoreOrchestrator CreatePlannerOrchestrator(IServiceProvider services, INsfwAiServerClient client, params IAiCapabilityContributor[] contributors)
        => new(
            client,
            contributors,
            services.GetRequiredService<IAiRunJournal>(),
            services.GetRequiredService<IAiRunPlanner>(),
            services.GetRequiredService<IAiArtifactReplaceService>(),
            NullLogger<AiCoreOrchestrator>.Instance);

    private static ServiceProvider CreateProvider()
    {
        var services = new ServiceCollection();
        var databaseName = $"ai-core-orchestrator-{Guid.NewGuid():N}";
        var databaseRoot = new InMemoryDatabaseRoot();
        services.AddDbContext<CoveContext>(options => options.UseInMemoryDatabase(databaseName, databaseRoot));
        services.AddScoped<IAiRunJournal, AiRunJournal>();
        services.AddScoped<IAiRunPlanner, AiRunPlanner>();
        services.AddScoped<IAiArtifactReplaceService, AiArtifactReplaceService>();
        return services.BuildServiceProvider();
    }

    private static RecordingAiServerClient CreateVideoTaggingClient(string modelKey, string category)
        => new()
        {
            CatalogModels = [CreateTaggingModel(modelKey, [category], scope: "frame")],
            VideoAnalyzeResponse = JsonDocument.Parse($$"""
            {
              "asset_id": "E:/media/example.mp4",
              "duration_seconds": 12.5,
              "frame_interval_seconds": 2.0,
              "models": [
                {
                  "config_name": "{{modelKey}}",
                  "name": "{{modelKey}}",
                  "categories": ["{{category}}"],
                  "capabilities": ["tagging"],
                  "supported_scopes": ["frame"]
                }
              ]
            }
            """
            ).RootElement.Clone(),
        };

    private static IAiCapabilityContributor CreateTaggingContributor(string claimId, string mediaKind, string scope)
        => new StubContributor(new AiCapabilityDescriptor(
            "ext:ai.tagging",
            "AI Tagging",
            [new AiCapabilityClaim(claimId, "Tagging", mediaKind, "tagging", scope, "tags") ]));

    private static IAiCapabilityContributor CreateFacesContributor()
        => new StubContributor(new AiCapabilityDescriptor(
            "cove.ai.faces",
            "AI Faces",
            [
                new AiCapabilityClaim(
                    "faces.video.detection",
                    "Video Face Detection",
                    AiMediaKinds.Video,
                    "detection",
                    "frame",
                    "frames",
                    PreferredModels: ["face_detector_torchexport"]),
                new AiCapabilityClaim(
                    "faces.video.embedding",
                    "Video Face Identity Embeddings",
                    AiMediaKinds.Video,
                    "embedding",
                    "region",
                    "regions",
                    PreferredModels: ["face_embedding_torchexport"],
                    FromDetection: "face_detector_torchexport"),
            ]));

    private static AiModelCatalogEntry CreateTaggingModel(
        string configName,
        IReadOnlyList<string> categories,
        bool loaded = false,
        bool pinned = false,
        bool active = true,
        string scope = "asset")
        => new()
        {
            ConfigName = configName,
            Name = configName,
            Categories = categories.ToList(),
            Capabilities = ["tagging"],
            SupportedScopes = [scope],
            Loaded = loaded,
            Pinned = pinned,
            Active = active,
        };

    private static AiModelCatalogEntry CreateModel(string configName, IReadOnlyList<string> categories, string capability, string scope)
        => new()
        {
            ConfigName = configName,
            Name = configName,
            Categories = categories.ToList(),
            Capabilities = [capability],
            SupportedScopes = [scope],
            Active = true,
        };

    private sealed class StubContributor(AiCapabilityDescriptor descriptor) : IAiCapabilityContributor
    {
        public AiCapabilityDescriptor Describe() => descriptor;

        public Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default)
            => Task.FromResult(new AiDispatchResult(descriptor.ExtensionId, request.Claims.Count));
    }

    private sealed class RecordingAiServerClient : INsfwAiServerClient
    {
        public IReadOnlyList<AiModelCatalogEntry> CatalogModels { get; init; } = [];

        public JsonElement VideoAnalyzeResponse { get; init; } = EmptyJson();

        public bool EchoRequestedVideoModels { get; init; }

        public object? LastAnalyzeRequest { get; private set; }

        public int AnalyzeVideoCallCount { get; private set; }

        public Task<IReadOnlyList<AiModelCatalogEntry>> GetModelCatalogAsync(AiCoreConnectionSettings settings, CancellationToken ct = default)
            => Task.FromResult(CatalogModels);

        public Task<IReadOnlyList<AiModelCatalogEntry>> GetLoadedModelsAsync(AiCoreConnectionSettings settings, CancellationToken ct = default)
            => Task.FromResult<IReadOnlyList<AiModelCatalogEntry>>(CatalogModels.Where(static model => model.Loaded).ToArray());

        public Task<IReadOnlyList<AiModelCatalogEntry>> LoadModelsAsync(AiCoreConnectionSettings settings, AiModelSelectionRequest request, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task<IReadOnlyList<AiModelCatalogEntry>> UnloadModelsAsync(AiCoreConnectionSettings settings, AiModelSelectionRequest request, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task<IReadOnlyList<AiModelCatalogEntry>> PinModelsAsync(AiCoreConnectionSettings settings, AiModelPinRequest request, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task<JsonElement> AnalyzeImagesAsync(AiCoreConnectionSettings settings, ImageAnalyzeRequest request, CancellationToken ct = default)
        {
            LastAnalyzeRequest = request;
            return Task.FromResult(EmptyJson());
        }

        public Task<JsonElement> AnalyzeVideoAsync(AiCoreConnectionSettings settings, VideoAnalyzeRequest request, CancellationToken ct = default)
        {
            LastAnalyzeRequest = request;
            AnalyzeVideoCallCount++;
            return Task.FromResult(EchoRequestedVideoModels ? BuildVideoResponse(request, CatalogModels) : VideoAnalyzeResponse);
        }

        public Task<JsonElement> AnalyzeAudioAsync(AiCoreConnectionSettings settings, AudioAnalyzeRequest request, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task<TextEncodeResponse> EncodeTextAsync(AiCoreConnectionSettings settings, TextEncodeRequest request, CancellationToken ct = default)
            => throw new NotSupportedException();

        private static JsonElement EmptyJson()
            => JsonDocument.Parse("{}").RootElement.Clone();

        private static JsonElement BuildVideoResponse(VideoAnalyzeRequest request, IReadOnlyList<AiModelCatalogEntry> catalogModels)
        {
            var requestedModels = request.Want?
                .SelectMany(static want => want.Models ?? [])
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray()
                ?? [];
            var models = catalogModels
                .Where(model => requestedModels.Contains(model.ConfigName, StringComparer.OrdinalIgnoreCase))
                .Select(model => new
                {
                    config_name = model.ConfigName,
                    name = model.Name,
                    categories = model.Categories,
                    capabilities = model.Capabilities,
                    supported_scopes = model.SupportedScopes,
                })
                .ToArray();

            return JsonDocument.Parse(JsonSerializer.Serialize(new
            {
                asset_id = "E:/media/example.mp4",
                duration_seconds = 12.5,
                frame_interval_seconds = request.FrameInterval,
                models,
            })).RootElement.Clone();
        }
    }

    private sealed class NoOpAiRunJournal : IAiRunJournal
    {
        public static NoOpAiRunJournal Instance { get; } = new();

        public Task RecordStartAsync(AiRunJournalStart entry, CancellationToken ct = default)
            => Task.CompletedTask;

        public Task RecordCompletionAsync(AiRunJournalCompletion completion, CancellationToken ct = default)
            => Task.CompletedTask;

        public Task RecordFailureAsync(string runKey, Exception exception, CancellationToken ct = default)
            => Task.CompletedTask;
    }

    private sealed class NoOpAiRunPlanner : IAiRunPlanner
    {
        public static NoOpAiRunPlanner Instance { get; } = new();

        public Task<IReadOnlyList<AiRunExecutionPlan>> PlanAsync(AiCoreConnectionSettings settings, string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunPlannerWant> wants, IReadOnlyList<string>? forceClaimIds, double? frameIntervalSeconds = null, double? threshold = null, CancellationToken ct = default)
            => Task.FromResult<IReadOnlyList<AiRunExecutionPlan>>(
                wants.Select(want => new AiRunExecutionPlan(
                    want.ExtensionId,
                    want.Capability,
                    want.Scope,
                    want.FromDetection,
                    want.Claims,
                    want.Models.Select(static model => model.ModelKey).ToArray(),
                    want.Models.Select(static model => model.ModelKey).ToArray(),
                    [],
                    AiRunPlanDecision.Run,
                    ["No-op planner executed all requested models."],
                    false)).ToArray());
    }

    private sealed class NoOpAiArtifactReplaceService : IAiArtifactReplaceService
    {
        public static NoOpAiArtifactReplaceService Instance { get; } = new();

        public Task ReplaceAsync(string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunExecutionPlan> plans, CancellationToken ct = default)
            => Task.CompletedTask;
    }
}