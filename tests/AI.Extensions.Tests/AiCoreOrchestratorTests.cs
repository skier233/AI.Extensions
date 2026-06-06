using System.Text.Json;

using AI.Core;
using AI.Extensions.Abstractions;

using Cove.Plugins;

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
                CapabilityModelBindings =
                [
                    new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "asset", Category = "Actions", Model = "tagger-actions-best" },
                    new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "asset", Category = "Pose", Model = "tagger-actions-best" },
                    new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "asset", Category = "Body", Model = "tagger-body" },
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
                CreateTaggingModel("tagger-actions-slow", ["Actions"], loaded: false),
                CreateTaggingModel("tagger-actions-loaded", ["Actions"], loaded: true),
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

        Assert.Equal(["tagger-actions-loaded", "tagger-body"], want.Models);
        Assert.DoesNotContain("tagger-actions-slow", want.Models ?? []);
    }

    [Fact]
    public async Task RunImagesAsync_ExpandsCapabilityIdsToClaims()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels = [CreateTaggingModel("tagger-actions-best", ["Actions"])],
        };
        var orchestrator = CreateOrchestrator(client, CreateTaggingContributor("tagging.image.asset", AiMediaKinds.Image, "asset"));

        await orchestrator.RunImagesAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunImagesRequest
            {
                Paths = ["E:/media/example.jpg"],
                CapabilityIds = ["tagging"],
                DispatchResults = false,
            });

        var request = Assert.IsType<ImageAnalyzeRequest>(client.LastAnalyzeRequest);
        var want = Assert.Single(request.Want ?? []);

        Assert.Equal("tagging", want.Capability);
        Assert.Equal(["tagger-actions-best"], want.Models);
    }

    [Fact]
    public async Task RunVideoAsync_UsesLoadedAutoBindingSlotModel()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels =
            [
                CreateModel("semvisual", ["visual_embeddings_semvisual"], "embedding", "frame", loaded: false, active: false),
                CreateModel("semvisual_trt", ["visual_embeddings_semvisual"], "embedding", "frame", loaded: true),
            ],
        };
        var orchestrator = CreateOrchestrator(client, CreateVisualContributor());

        await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings { DefaultLoadPolicy = AiLoadPolicies.UseLoaded }.Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                CapabilityIds = ["visual.semantic"],
                DispatchResults = false,
            });

        var request = Assert.IsType<VideoAnalyzeRequest>(client.LastAnalyzeRequest);
        var want = Assert.Single(request.Want ?? []);

        Assert.Equal("embedding", want.Capability);
        Assert.Equal(["semvisual_trt"], want.Models);
    }

    [Fact]
    public async Task RunVideoAsync_TranslatesPresetClaimIdsAcrossMediaKinds()
    {
        var client = CreateVideoTaggingClient("tagger-actions-best", "Actions");
        var orchestrator = CreateOrchestrator(
            client,
            CreateTaggingContributor("tagging.image.asset", AiMediaKinds.Image, "asset"),
            CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"));

        await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings
            {
                RunPresets =
                [
                    new AiRunPreset
                    {
                        PresetId = "default",
                        DisplayName = "Default",
                        ClaimIds = ["tagging.image.asset"],
                    },
                ],
            }.Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                PresetId = "default",
                FrameInterval = 2.0,
                DispatchResults = false,
            });

        var request = Assert.IsType<VideoAnalyzeRequest>(client.LastAnalyzeRequest);
        var want = Assert.Single(request.Want ?? []);

        Assert.Equal("tagging", want.Capability);
        Assert.Equal("frame", want.Scope);
        Assert.Equal(["tagger-actions-best"], want.Models);
    }

    [Fact]
    public async Task RunImagesAsync_CustomPipelineCapabilitySetsPipelineAndIncludedCapabilities()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels = [CreateTaggingModel("tagger-actions-best", ["Actions"])],
        };
        var orchestrator = CreateOrchestrator(client, CreateTaggingContributor("tagging.image.asset", AiMediaKinds.Image, "asset"));

        await orchestrator.RunImagesAsync(
            new AiCoreConnectionSettings
            {
                CustomPipelines =
                [
                    new AiCustomPipelineDefinition
                    {
                        PipelineName = "custom_image_tags",
                        MediaKind = AiMediaKinds.Image,
                        CapabilityId = "custom.image-tags",
                        CapabilityIds = ["tagging"],
                        FullImageModels = ["tagger-actions-best"],
                    },
                ],
            }.Normalize(),
            new AiRunImagesRequest
            {
                Paths = ["E:/media/example.jpg"],
                CapabilityIds = ["custom.image-tags"],
                DispatchResults = false,
            });

        var request = Assert.IsType<ImageAnalyzeRequest>(client.LastAnalyzeRequest);

        Assert.Equal("custom_image_tags", request.PipelineName);
        Assert.Equal(["tagger-actions-best"], Assert.Single(request.Want ?? []).Models);
    }

    [Fact]
    public async Task RunAudioAsync_UsesOnlyAudioPreferredModelsFromCatalog()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels =
            [
                CreateModel("face_embedding_torchexport", ["face_embeddings"], "embedding", "asset"),
                CreateModel("audioembed", ["audio_embeddings_audioembed"], "embedding", "asset"),
                CreateModel("audioclass", ["audio_classification_audioclass"], "classification", "asset"),
            ],
        };
        var orchestrator = CreateOrchestrator(client, CreateAudioContributor());

        await orchestrator.RunAudioAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunAudioRequest
            {
                Paths = ["E:/media/example.mp4"],
                CapabilityIds = ["audio.embedding", "audio.classification"],
                DispatchResults = false,
            });

        var request = Assert.IsType<AudioAnalyzeRequest>(client.LastAnalyzeRequest);
        Assert.NotNull(request.Want);
        var wants = request.Want!;

        Assert.Contains(wants, want => want.Capability == "embedding" && Assert.Single(want.Models ?? []) == "audioembed");
        Assert.Contains(wants, want => want.Capability == "classification" && Assert.Single(want.Models ?? []) == "audioclass");
        Assert.DoesNotContain(wants.SelectMany(static want => want.Models ?? []), model => model == "face_embedding_torchexport");
    }

    [Fact]
    public async Task RunAudioAsync_SkipsWhenNoAudioModelsAreInCatalog()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels = [CreateModel("face_embedding_torchexport", ["face_embeddings"], "embedding", "asset")],
        };
        var orchestrator = CreateOrchestrator(client, CreateAudioContributor());

        var response = await orchestrator.RunAudioAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunAudioRequest
            {
                Paths = ["E:/media/example.mp4"],
                CapabilityIds = ["audio.embedding", "audio.classification"],
                DispatchResults = false,
            });

        Assert.Equal(0, client.AnalyzeAudioCallCount);
        Assert.Equal("skipped", response.Analysis.GetProperty("status").GetString());
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
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                FrameInterval = 2.0,
                DispatchResults = false,
            });

        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var run = await db.AiRuns.SingleAsync();

        Assert.Equal(response.RunId, run.RunKey);
        Assert.Equal("ext:ai.core", run.SourceKey);
        Assert.Equal(AiRunTargetType.Video, run.TargetType);
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
            "video",
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
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = true,
            });

        db.TagApplications.Add(new TagApplication
        {
            HostType = AffinityHostType.Video,
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
                EntityType = "video",
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
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = false,
            });

        var second = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
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
            HostType = FaceAppearanceHostType.Video,
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
                EntityType = "video",
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
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["faces.video.detection", "faces.video.embedding"],
                DispatchResults = false,
            });

        var second = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
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
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = true,
            });

        db.TagApplications.Add(new TagApplication
        {
            HostType = AffinityHostType.Video,
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
                EntityType = "video",
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
            CapabilityModelBindings =
            [
                new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "frame", Category = "Actions", Model = "tagger-actions-v1" },
                new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "frame", Category = "Body", Model = "tagger-body" },
            ],
        }.Normalize();

        var first = await orchestrator.RunVideoAsync(
            initialSettings,
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame"],
                DispatchResults = true,
            });

        db.TagApplications.AddRange(
            new TagApplication
            {
                HostType = AffinityHostType.Video,
                HostId = 42,
                TagId = 9,
                SourceKey = "ext:ai.tagging",
                SourceRunId = first.RunId,
                ModelKey = "Actions",
                Confidence = 0.9f,
            },
            new TagApplication
            {
                HostType = AffinityHostType.Video,
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
            CapabilityModelBindings =
            [
                new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "frame", Category = "Actions", Model = "tagger-actions-v2" },
                new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "frame", Category = "Body", Model = "tagger-body" },
            ],
        }.Normalize();

        var second = await orchestrator.RunVideoAsync(
            rerunSettings,
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
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

    private static IExtensionServiceExchange CreateExchange(IReadOnlyList<IAiCapabilityContributor> contributors)
    {
        var exchange = new ExtensionServiceExchange();
        foreach (var contributor in contributors)
            exchange.Publish<IAiCapabilityContributor>(contributor.Describe().ExtensionId, contributor);
        return exchange;
    }

    private static AiCoreOrchestrator CreateOrchestrator(INsfwAiServerClient client, params IAiCapabilityContributor[] contributors)
        => new(
            client,
            CreateExchange(contributors),
            NoOpAiRunJournal.Instance,
            NoOpAiRunPlanner.Instance,
            NoOpAiArtifactReplaceService.Instance,
            NullLogger<AiCoreOrchestrator>.Instance);

    private static AiCoreOrchestrator CreatePlannerOrchestrator(IServiceProvider services, INsfwAiServerClient client, params IAiCapabilityContributor[] contributors)
        => new(
            client,
            CreateExchange(contributors),
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
            [
                new AiCapabilityClaim(claimId, "Tagging", mediaKind, "tagging", scope, "tags")
                {
                    CapabilityId = "tagging",
                    ModelBindingSlotId = "category",
                },
            ])
        {
            Capabilities =
            [
                new AiCapabilityFeature(
                    "tagging",
                    "Content Tagging",
                    [claimId],
                    [
                        new AiModelBindingSlot(
                            "category",
                            "Tagging category model",
                            "tagging",
                            RequiredCapabilities: ["tagging"],
                            RequiredScopes: ["asset", "frame"],
                            CategoryScoped: true),
                    ]),
            ],
        });

    private static IAiCapabilityContributor CreateFacesContributor()
        => new StubContributor(new AiCapabilityDescriptor(
            "cove.community.ai.faces",
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

    private static IAiCapabilityContributor CreateAudioContributor()
        => new StubContributor(new AiCapabilityDescriptor(
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
                    PreferredModels: ["audioembed"])
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
                    PreferredModels: ["audioclass"])
                {
                    CapabilityId = "audio.classification",
                    ModelBindingSlotId = "classifier",
                },
            ])
        {
            Capabilities =
            [
                new AiCapabilityFeature("audio.embedding", "Audio Embeddings", ["audio.asset.embedding"]),
                new AiCapabilityFeature("audio.classification", "Audio Classification", ["audio.asset.classification"]),
            ],
        });

    private static IAiCapabilityContributor CreateVisualContributor()
        => new StubContributor(new AiCapabilityDescriptor(
            "cove.community.ai.visual",
            "AI Visual",
            [
                new AiCapabilityClaim(
                    "visual.video.semantic",
                    "Video Semantic Embeddings",
                    AiMediaKinds.Video,
                    "embedding",
                    "frame",
                    "frames",
                    PreferredModels: ["semvisual"])
                {
                    CapabilityId = "visual.semantic",
                    ModelBindingSlotId = "embedder",
                },
            ])
        {
            Capabilities =
            [
                new AiCapabilityFeature(
                    "visual.semantic",
                    "Semantic Visual Search",
                    ["visual.video.semantic"],
                    [
                        new AiModelBindingSlot(
                            "embedder",
                            "Semantic embedding model",
                            "embedding",
                            RequiredCapabilities: ["embedding"],
                            RequiredScopes: ["asset", "frame"],
                            RequiredCategories: ["visual_embeddings_semvisual"],
                            DefaultModels: ["semvisual"]),
                    ]),
            ],
        });

    private static AiModelCatalogEntry CreateTaggingModel(
        string configName,
        IReadOnlyList<string> categories,
        bool loaded = true,
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
            Active = active,
        };

    private static AiModelCatalogEntry CreateModel(string configName, IReadOnlyList<string> categories, string capability, string scope, bool loaded = true, bool active = true)
        => new()
        {
            ConfigName = configName,
            Name = configName,
            Categories = categories.ToList(),
            Capabilities = [capability],
            SupportedScopes = [scope],
            Loaded = loaded,
            Active = active,
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

        public int AnalyzeAudioCallCount { get; private set; }

        public Task<IReadOnlyList<AiModelCatalogEntry>> GetModelCatalogAsync(AiCoreConnectionSettings settings, CancellationToken ct = default)
            => Task.FromResult(CatalogModels);

        public Task<IReadOnlyList<AiModelCatalogEntry>> GetLoadedModelsAsync(AiCoreConnectionSettings settings, CancellationToken ct = default)
            => Task.FromResult<IReadOnlyList<AiModelCatalogEntry>>(CatalogModels.Where(static model => model.Loaded).ToArray());

        public Task<IReadOnlyList<AiModelCatalogEntry>> LoadModelsAsync(AiCoreConnectionSettings settings, AiModelSelectionRequest request, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task<IReadOnlyList<AiModelCatalogEntry>> UnloadModelsAsync(AiCoreConnectionSettings settings, AiModelSelectionRequest request, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task<AiCustomPipelineSyncResponse> RegisterCustomPipelineAsync(AiCoreConnectionSettings settings, AiCustomPipelineDefinition pipeline, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task<AiCustomPipelineSyncResponse> DeleteCustomPipelineAsync(AiCoreConnectionSettings settings, string pipelineName, CancellationToken ct = default)
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
        {
            LastAnalyzeRequest = request;
            AnalyzeAudioCallCount++;
            return Task.FromResult(EmptyJson());
        }

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