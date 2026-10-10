using System.Text.Json;

using AI.Core;
using AI.Extensions.Abstractions;

using Cove.Plugins;

using Cove.Core.Entities;
using Cove.Core.Interfaces;
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
    public async Task RunVideoAsync_FaceArtifactsWithoutPriorRunDoNotSatisfyAndStillRun()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();

        // Persisted face artifacts exist, but there is no completed ext:ai.core run record.
        // The planner decides skip/run solely from prior run records, so these artifacts must
        // NOT make it skip — it should still run.
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
        await db.SaveChangesAsync();

        var client = new RecordingAiServerClient
        {
            CatalogModels =
            [
                CreateModel("face_detector_torchexport", ["face_detections"], "detection", "frame"),
                CreateModel("face_embedding_torchexport", ["face_embeddings"], "embedding", "region"),
            ],
        };
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

        Assert.Equal(1, client.AnalyzeVideoCallCount);
        Assert.Contains(result.Plan, plan => plan.Decision == AiRunPlanDecision.Run);
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
    public async Task RunVideoAsync_DoesNotCallServerWhenCategoriesToSkipCoversEveryModel()
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

        var result = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["faces.video.detection", "faces.video.embedding"],
                CategoriesToSkip = ["face_detections", "face_embeddings"],
                DispatchResults = true,
            });

        // Every requested model's categories are in categories_to_skip, so the server
        // would only preprocess and run nothing — cove must not call it at all.
        Assert.Equal(0, client.AnalyzeVideoCallCount);
        Assert.Equal("skipped", result.Analysis.GetProperty("status").GetString());
    }

    [Fact]
    public async Task RunVideoAsync_StillCallsServerWhenAModelHasAnUnskippedCategory()
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

        var result = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["faces.video.detection", "faces.video.embedding"],
                CategoriesToSkip = ["face_detections"],
                DispatchResults = true,
            });

        // The embedding model's category is not skipped, so there is real work to do.
        Assert.Equal(1, client.AnalyzeVideoCallCount);
    }

    [Fact]
    public async Task ClearingHostFaceEvidence_LetsFaceRecognitionRunAgain()
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

        AiRunVideoRequest Request() => new()
        {
            Path = "E:/media/example.mp4",
            EntityType = "video",
            EntityId = 42,
            ClaimIds = ["faces.video.detection", "faces.video.embedding"],
            DispatchResults = false,
        };

        await orchestrator.RunVideoAsync(new AiCoreConnectionSettings().Normalize(), Request());
        var skipped = await orchestrator.RunVideoAsync(new AiCoreConnectionSettings().Normalize(), Request());
        Assert.Equal(1, client.AnalyzeVideoCallCount);
        Assert.All(skipped.Plan, plan => Assert.Equal(AiRunPlanDecision.Skip, plan.Decision));

        // The face lifecycle reports cleared work by category; this must prune the run evidence so the
        // planner stops treating face recognition as satisfied.
        var participant = new AiCoreFaceRunEvidenceParticipant(scope.ServiceProvider.GetRequiredService<IServiceScopeFactory>());
        await participant.OnHostFacesClearedAsync(new FaceRunEvidenceCleared(DetectionHostType.Video, 42, ["face_detections", "face_embeddings"]));

        var rerun = await orchestrator.RunVideoAsync(new AiCoreConnectionSettings().Normalize(), Request());
        Assert.Equal(2, client.AnalyzeVideoCallCount);
        Assert.All(rerun.Plan, plan => Assert.Equal(AiRunPlanDecision.Run, plan.Decision));
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
    public async Task RunVideoAsync_UpgradingTaggingModelVersionRerunsOnlyTheChangedCategory()
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
                CreateTaggingModel("tagger-actions-v1", ["Actions"], scope: "frame", version: "1.0"),
                CreateTaggingModel("tagger-actions-v2", ["Actions"], scope: "frame", version: "2.0"),
                CreateTaggingModel("tagger-body", ["Body"], scope: "frame", version: "1.0"),
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

    [Fact]
    public async Task RunVideoAsync_PlanningContributorSkipRemovesOnlyItsWantFromTheServerRequest()
    {
        var client = CreateTaggingAndShotsClient();
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Skip);
        var orchestrator = CreateOrchestrator(client, CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"), shots);

        var response = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame", PlanningStubContributor.ClaimId],
                DispatchResults = true,
            });

        var analyzeRequest = Assert.IsType<VideoAnalyzeRequest>(client.LastAnalyzeRequest);
        var want = Assert.Single(analyzeRequest.Want ?? []);
        Assert.Equal("tagging", want.Capability);
        Assert.Equal(AiRunPlanDecision.Skip, response.Plan.Single(item => item.ClaimId == PlanningStubContributor.ClaimId).Decision);
        Assert.Equal(AiRunPlanDecision.Run, response.Plan.Single(item => item.ClaimId == "tagging.video.frame").Decision);
        Assert.Empty(shots.Dispatched);
    }

    [Fact]
    public async Task RunVideoAsync_PlanningContributorSkipWithNothingElseMakesNoServerCall()
    {
        var client = CreateTaggingAndShotsClient();
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Skip);
        var orchestrator = CreateOrchestrator(client, shots);

        var response = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = [PlanningStubContributor.ClaimId],
            });

        Assert.Equal(0, client.AnalyzeVideoCallCount);
        Assert.Equal("skipped", response.Analysis.GetProperty("status").GetString());
        Assert.Equal("stub Skip", Assert.Single(Assert.Single(response.Plan).Reasons));
    }

    [Fact]
    public async Task RunVideoAsync_PlanningContributorRunIgnoresCompletedRunHistory()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var client = CreateTaggingAndShotsClient();
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Run);
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, client, shots);
        var request = new AiRunVideoRequest
        {
            Path = "E:/media/example.mp4",
            EntityType = "video",
            EntityId = 42,
            ClaimIds = [PlanningStubContributor.ClaimId],
            DispatchResults = true,
        };

        await orchestrator.RunVideoAsync(new AiCoreConnectionSettings().Normalize(), request);
        var second = await orchestrator.RunVideoAsync(new AiCoreConnectionSettings().Normalize(), request);

        // The run history now records this claim with the same model, which would skip a history-planned claim.
        Assert.Equal(2, client.AnalyzeVideoCallCount);
        Assert.Equal(AiRunPlanDecision.Run, Assert.Single(second.Plan).Decision);
        Assert.Equal(2, shots.Dispatched.Count);
    }

    [Fact]
    public async Task RunVideoAsync_PlanningContributorSkipIgnoresThresholdIntervalAndVersionChanges()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var verdict = AiClaimPlanningVerdict.Run;
        var shots = new PlanningStubContributor(target => target.Claims
            .Select(claim => new AiClaimPlanningDecision { ClaimId = claim.ClaimId, Verdict = verdict })
            .ToArray());
        var firstClient = CreateTaggingAndShotsClient(shotsVersion: "1.0");
        await CreatePlannerOrchestrator(scope.ServiceProvider, firstClient, shots).RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = [PlanningStubContributor.ClaimId],
                FrameInterval = 2.0,
                Threshold = 0.5,
            });

        verdict = AiClaimPlanningVerdict.Skip;
        var secondClient = CreateTaggingAndShotsClient(shotsVersion: "2.0");
        var second = await CreatePlannerOrchestrator(scope.ServiceProvider, secondClient, shots).RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = [PlanningStubContributor.ClaimId],
                FrameInterval = 3.0,
                Threshold = 0.7,
            });

        Assert.Equal(0, secondClient.AnalyzeVideoCallCount);
        Assert.Equal(AiRunPlanDecision.Skip, Assert.Single(second.Plan).Decision);
    }

    [Fact]
    public async Task RunVideoAsync_ForcedPlanningClaimReachesTargetAndMetadataWithoutArtifactReplacement()
    {
        var client = CreateTaggingAndShotsClient();
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Run, replacesExistingResults: true);
        var replace = new RecordingAiArtifactReplaceService();
        var orchestrator = CreateOrchestrator(client, NoOpAiRunPlanner.Instance, replace, shots);

        var response = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                FileId = 77,
                ClaimIds = [PlanningStubContributor.ClaimId],
                ForceClaimIds = [PlanningStubContributor.ClaimId],
                DispatchResults = true,
            });

        var target = Assert.Single(shots.Targets);
        Assert.Equal([PlanningStubContributor.ClaimId], target.ForcedClaimIds);
        Assert.Equal(77, target.HostFileId);
        Assert.Equal("E:/media/example.mp4", target.Subject);
        Assert.Equal(response.RunId, target.RunId);

        var plan = Assert.Single(response.Plan);
        Assert.Equal(AiRunPlanDecision.Rerun, plan.Decision);
        Assert.True(plan.Forced);
        var replacedPlan = Assert.Single(Assert.Single(replace.Calls));
        Assert.Empty(replacedPlan.ReplacementArtifactKeys);

        var dispatched = Assert.Single(shots.Dispatched);
        Assert.True(dispatched.IsForced(PlanningStubContributor.ClaimId));
        Assert.True(dispatched.TryGetHostFileId(out var fileId));
        Assert.Equal(77, fileId);
    }

    [Fact]
    public async Task RunVideoAsync_ForcedPlanningClaimWithNothingToReplaceIsARun()
    {
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Run);
        var orchestrator = CreateOrchestrator(CreateTaggingAndShotsClient(), shots);

        var response = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = [PlanningStubContributor.ClaimId],
                ForceClaimIds = [PlanningStubContributor.ClaimId],
            });

        var plan = Assert.Single(response.Plan);
        Assert.Equal(AiRunPlanDecision.Run, plan.Decision);
        Assert.True(plan.Forced);
    }

    [Fact]
    public async Task RunVideoAsync_HistoryPlansGoBackToTheirOwnWantsAroundAPlanningContributor()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels =
            [
                CreateTaggingModel("tagger-actions-best", ["Actions"], scope: "frame"),
                CreateModel("face_detector_torchexport", ["face_detections"], "detection", "frame"),
                CreateModel("face_embedding_torchexport", ["face_embeddings"], "embedding", "region"),
                CreateModel("shots", ["shot_boundaries"], "temporal_segmentation", "asset"),
            ],
            EchoRequestedVideoModels = true,
        };
        var planner = new SpyAiRunPlanner(new LabellingAiRunPlanner());
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Run);
        var orchestrator = CreateOrchestrator(
            client,
            planner,
            NoOpAiArtifactReplaceService.Instance,
            CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"),
            CreateFacesContributor(),
            shots);

        var response = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ForceClaimIds = ["tagging.video.frame", "faces.video.detection", PlanningStubContributor.ClaimId],
            });

        // Wants are ordered faces detection, faces embedding, shots, tagging: the planning want sits between history
        // wants, so every history plan must be matched back to its own want, not to its position in the planner call.
        var historyWants = Assert.Single(planner.Calls).Wants;
        Assert.Equal(["faces.video.detection", "faces.video.embedding", "tagging.video.frame"], historyWants.SelectMany(static want => want.Claims).Select(static claim => claim.ClaimId));
        Assert.Equal(
            ["faces.video.detection", "faces.video.embedding", PlanningStubContributor.ClaimId, "tagging.video.frame"],
            response.Plan.Select(static item => item.ClaimId));
        foreach (var item in response.Plan.Where(static item => item.ClaimId != PlanningStubContributor.ClaimId))
        {
            Assert.Equal([$"planned for {item.ClaimId}"], item.Reasons);
        }

        // Each want is sent to the server with its own plan's models.
        var serverWants = Assert.IsType<VideoAnalyzeRequest>(client.LastAnalyzeRequest).Want!;
        Assert.Equal(
            [
                "detection/frame: face_detector_torchexport",
                "embedding/region: face_embedding_torchexport",
                "temporal_segmentation/asset: shots",
                "tagging/frame: tagger-actions-best",
            ],
            serverWants.Select(static want => $"{want.Capability}/{want.Scope}: {string.Join(",", want.Models ?? [])}"));

        Assert.Equal(["stub Run"], response.Plan.Single(static item => item.ClaimId == PlanningStubContributor.ClaimId).Reasons);
        Assert.Equal([PlanningStubContributor.ClaimId], Assert.Single(shots.Targets).ForcedClaimIds);
    }

    [Fact]
    public async Task RunVideoAsync_TheAnalysedFileIdIsNotRecordedInTheRunRequest()
    {
        await using var provider = CreateProvider();
        await using var scope = provider.CreateAsyncScope();
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, CreateTaggingAndShotsClient(), CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"));

        await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                FileId = 77,
                ClaimIds = ["tagging.video.frame"],
            });

        var run = await scope.ServiceProvider.GetRequiredService<CoveContext>().AiRuns.SingleAsync();
        Assert.False(run.Request!.RootElement.TryGetProperty("FileId", out _));
    }

    [Fact]
    public void AiRunVideoRequest_FileIdIsNeverReadFromJson()
    {
        var request = JsonSerializer.Deserialize<AiRunVideoRequest>(
            """{ "path": "E:/media/example.mp4", "fileId": 77, "FileId": 78 }""",
            new JsonSerializerOptions(JsonSerializerDefaults.Web))!;

        Assert.Equal("E:/media/example.mp4", request.Path);
        Assert.Null(request.FileId);
        Assert.DoesNotContain("77", JsonSerializer.Serialize(new AiRunVideoRequest { Path = "a", FileId = 77 }));
    }

    [Fact]
    public async Task RunVideoAsync_OrdinaryContributorsKeepTheirDispatchMetadata()
    {
        var client = CreateTaggingAndShotsClient();
        var tagging = new RecordingContributor(CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame").Describe());
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Run);
        var orchestrator = CreateOrchestrator(client, tagging, shots);

        await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                FileId = 77,
                ClaimIds = ["tagging.video.frame", PlanningStubContributor.ClaimId],
                ForceClaimIds = ["tagging.video.frame", PlanningStubContributor.ClaimId],
                DispatchResults = true,
            });

        var taggingMetadata = Assert.Single(tagging.Dispatched).Metadata!;
        Assert.Equal(["extensionId", "source"], taggingMetadata.Keys.Order(StringComparer.Ordinal));
        var shotsMetadata = Assert.Single(shots.Dispatched).Metadata!;
        Assert.Equal(PlanningStubContributor.ClaimId, shotsMetadata[AiDispatchMetadataKeys.ForcedClaimIds]);
        Assert.Equal("77", shotsMetadata[AiDispatchMetadataKeys.HostFileId]);
    }

    [Fact]
    public async Task RunVideoAsync_PlanningContributorIsToldWhenNothingWasForced()
    {
        var client = CreateTaggingAndShotsClient();
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Run);
        var orchestrator = CreateOrchestrator(client, shots);

        await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = [PlanningStubContributor.ClaimId],
                DispatchResults = true,
            });

        Assert.Empty(Assert.Single(shots.Targets).ForcedClaimIds);
        var dispatched = Assert.Single(shots.Dispatched);
        Assert.Equal(string.Empty, dispatched.Metadata![AiDispatchMetadataKeys.ForcedClaimIds]);
        Assert.False(dispatched.IsForced(PlanningStubContributor.ClaimId));
        Assert.False(dispatched.TryGetHostFileId(out _));
    }

    [Fact]
    public async Task RunVideoAsync_WithoutPlanningContributorsTheHistoryPlannerGetsOneCallWithEveryWant()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels =
            [
                CreateTaggingModel("tagger-actions-best", ["Actions"], scope: "frame"),
                CreateModel("face_detector_torchexport", ["face_detections"], "detection", "frame"),
                CreateModel("face_embedding_torchexport", ["face_embeddings"], "embedding", "region"),
            ],
            EchoRequestedVideoModels = true,
        };
        var planner = new SpyAiRunPlanner(NoOpAiRunPlanner.Instance);
        var orchestrator = CreateOrchestrator(
            client,
            planner,
            NoOpAiArtifactReplaceService.Instance,
            CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"),
            CreateFacesContributor());

        await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                FrameInterval = 2.0,
                Threshold = 0.4,
                ForceClaimIds = ["tagging.video.frame"],
            });

        var call = Assert.Single(planner.Calls);
        Assert.Equal("video", call.HostEntityType);
        Assert.Equal(42, call.HostEntityId);
        Assert.Equal(["tagging.video.frame"], call.ForceClaimIds);
        Assert.Equal(2.0, call.FrameIntervalSeconds);
        Assert.Equal(0.4, call.Threshold);
        Assert.Equal(["cove.community.ai.faces", "ext:ai.tagging"], call.Wants.Select(static want => want.ExtensionId).Distinct().Order(StringComparer.Ordinal));
    }

    [Fact]
    public async Task RunVideoAsync_HistoryPlannedWantsAreUnchangedByAPlanningContributor()
    {
        async Task<PlannerCall> PlanWith(params IAiCapabilityContributor[] extra)
        {
            var planner = new SpyAiRunPlanner(NoOpAiRunPlanner.Instance);
            var contributors = new IAiCapabilityContributor[] { CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame") }.Concat(extra).ToArray();
            await CreateOrchestrator(CreateTaggingAndShotsClient(), planner, NoOpAiArtifactReplaceService.Instance, contributors).RunVideoAsync(
                new AiCoreConnectionSettings().Normalize(),
                new AiRunVideoRequest
                {
                    Path = "E:/media/example.mp4",
                    EntityType = "video",
                    EntityId = 42,
                    FrameInterval = 2.0,
                    Threshold = 0.4,
                    ForceClaimIds = ["tagging.video.frame", PlanningStubContributor.ClaimId],
                });
            return Assert.Single(planner.Calls);
        }

        var without = await PlanWith();
        var with = await PlanWith(PlanningStubContributor.Always(AiClaimPlanningVerdict.Skip));

        Assert.Equal(Describe(without), Describe(with));

        static string Describe(PlannerCall call)
            => JsonSerializer.Serialize(new
            {
                call.HostEntityType,
                call.HostEntityId,
                call.ForceClaimIds,
                call.FrameIntervalSeconds,
                call.Threshold,
                Wants = call.Wants.Select(static want => new
                {
                    want.ExtensionId,
                    want.Capability,
                    want.Scope,
                    want.FromDetection,
                    Claims = want.Claims.Select(static claim => claim.ClaimId),
                    Models = want.Models.Select(static model => new { model.ModelKey, model.ArtifactKeys, model.Category, model.Identifier, model.Version, model.Name, model.Categories }),
                    want.AllowPartialExecution,
                }),
            });
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RunVideoAsync_TaggingDecisionsAreIdenticalWithAndWithoutAPlanningContributor(bool forceTagging)
    {
        var without = await RunTaggingUpgradeScenarioAsync(forceTagging, withPlanningContributor: false);
        var with = await RunTaggingUpgradeScenarioAsync(forceTagging, withPlanningContributor: true);

        Assert.Equal(without, with);
    }

    [Fact]
    public async Task RunVideoAsync_ThrowingPlanningContributorDoesNotBlockTagging()
    {
        var client = CreateTaggingAndShotsClient();
        var shots = new PlanningStubContributor(_ => throw new InvalidOperationException("storage offline"));
        var orchestrator = CreateOrchestrator(client, CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"), shots);

        var response = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = ["tagging.video.frame", PlanningStubContributor.ClaimId],
                DispatchResults = true,
            });

        var want = Assert.Single(Assert.IsType<VideoAnalyzeRequest>(client.LastAnalyzeRequest).Want ?? []);
        Assert.Equal("tagging", want.Capability);
        var shotsPlan = response.Plan.Single(item => item.ClaimId == PlanningStubContributor.ClaimId);
        Assert.Equal(AiRunPlanDecision.Skip, shotsPlan.Decision);
        Assert.Contains("storage offline", Assert.Single(shotsPlan.Reasons));
    }

    [Fact]
    public async Task RunVideoAsync_DefaultVerdictFallsBackToRunHistory()
    {
        var client = CreateTaggingAndShotsClient();
        var planner = new SpyAiRunPlanner(NoOpAiRunPlanner.Instance);
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Default);
        var orchestrator = CreateOrchestrator(client, planner, NoOpAiArtifactReplaceService.Instance, shots);

        var response = await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = [PlanningStubContributor.ClaimId],
            });

        var call = Assert.Single(planner.Calls);
        Assert.Equal(PlanningStubContributor.ClaimId, Assert.Single(Assert.Single(call.Wants).Claims).ClaimId);
        Assert.Equal("No-op planner executed all requested models.", Assert.Single(Assert.Single(response.Plan).Reasons));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RunVideoAsync_PlanningContributorLearnsWhetherClaimsWereSelected(bool selectClaims)
    {
        var shots = PlanningStubContributor.Always(AiClaimPlanningVerdict.Skip);
        var orchestrator = CreateOrchestrator(CreateTaggingAndShotsClient(), shots);

        await orchestrator.RunVideoAsync(
            new AiCoreConnectionSettings().Normalize(),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = selectClaims ? [PlanningStubContributor.ClaimId] : null,
                DispatchResults = false,
            });

        var target = Assert.Single(shots.Targets);
        Assert.Equal(selectClaims, target.ClaimsExplicitlySelected);
        Assert.False(target.DispatchResults);
    }

    [Fact]
    public async Task RunImageBatchAsync_ForcedPlanningClaimReachesBatchDispatchMetadata()
    {
        var client = new RecordingAiServerClient
        {
            CatalogModels = [CreateModel("image-shots", ["image_shots"], "temporal_segmentation", "asset")],
        };
        var shots = new PlanningStubContributor(
            target => target.Claims.Select(static claim => new AiClaimPlanningDecision { ClaimId = claim.ClaimId, Verdict = AiClaimPlanningVerdict.Run }).ToArray(),
            AiMediaKinds.Image,
            "image_shots");
        var orchestrator = CreateOrchestrator(client, shots);

        await orchestrator.RunImageBatchAsync(
            new AiCoreConnectionSettings().Normalize(),
            [new AiRunImageTarget("E:/media/a.jpg", "image", 5)],
            new AiRunImagesRequest
            {
                ClaimIds = [PlanningStubContributor.ClaimId],
                ForceClaimIds = [PlanningStubContributor.ClaimId],
                DispatchResults = true,
            });

        Assert.Null(Assert.Single(shots.Targets).HostFileId);
        Assert.True(Assert.Single(shots.Dispatched).IsForced(PlanningStubContributor.ClaimId));
    }

    private static async Task<string> RunTaggingUpgradeScenarioAsync(bool forceTagging, bool withPlanningContributor)
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
                CreateTaggingModel("tagger-actions-v1", ["Actions"], scope: "frame", version: "1.0"),
                CreateTaggingModel("tagger-actions-v2", ["Actions"], scope: "frame", version: "2.0"),
                CreateTaggingModel("tagger-body", ["Body"], scope: "frame", version: "1.0"),
                CreateModel("shots", ["shot_boundaries"], "temporal_segmentation", "asset"),
            ],
            EchoRequestedVideoModels = true,
        };
        var contributors = withPlanningContributor
            ? new[] { CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame"), PlanningStubContributor.Always(AiClaimPlanningVerdict.Skip) }
            : new[] { CreateTaggingContributor("tagging.video.frame", AiMediaKinds.Video, "frame") };
        var orchestrator = CreatePlannerOrchestrator(scope.ServiceProvider, client, contributors);
        var claimIds = withPlanningContributor ? new List<string> { "tagging.video.frame", PlanningStubContributor.ClaimId } : ["tagging.video.frame"];

        AiCoreConnectionSettings Bindings(string actionsModel) => new AiCoreConnectionSettings
        {
            CapabilityModelBindings =
            [
                new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "frame", Category = "Actions", Model = actionsModel },
                new AiCapabilityModelBinding { CapabilityId = "tagging", SlotId = "category", Scope = "frame", Category = "Body", Model = "tagger-body" },
            ],
        }.Normalize();

        var first = await orchestrator.RunVideoAsync(
            Bindings("tagger-actions-v1"),
            new AiRunVideoRequest { Path = "E:/media/example.mp4", EntityType = "video", EntityId = 42, ClaimIds = claimIds, DispatchResults = true });

        db.TagApplications.AddRange(
            new TagApplication { HostType = AffinityHostType.Video, HostId = 42, TagId = 9, SourceKey = "ext:ai.tagging", SourceRunId = first.RunId, ModelKey = "Actions", Confidence = 0.9f },
            new TagApplication { HostType = AffinityHostType.Video, HostId = 42, TagId = 9, SourceKey = "ext:ai.tagging", SourceRunId = first.RunId, ModelKey = "Body", Confidence = 0.9f });
        await db.SaveChangesAsync();

        var second = await orchestrator.RunVideoAsync(
            Bindings("tagger-actions-v2"),
            new AiRunVideoRequest
            {
                Path = "E:/media/example.mp4",
                EntityType = "video",
                EntityId = 42,
                ClaimIds = claimIds,
                ForceClaimIds = forceTagging ? ["tagging.video.frame"] : null,
                DispatchResults = true,
            });

        var taggingPlan = second.Plan.Single(item => item.ClaimId == "tagging.video.frame");
        var taggingWant = (Assert.IsType<VideoAnalyzeRequest>(client.LastAnalyzeRequest).Want ?? []).Single(want => want.Capability == "tagging");
        return JsonSerializer.Serialize(new
        {
            taggingPlan.Decision,
            taggingPlan.DesiredModels,
            taggingPlan.ExecutionModels,
            taggingPlan.Reasons,
            taggingPlan.Forced,
            ServerModels = taggingWant.Models,
            ActionsLeft = await db.TagApplications.CountAsync(application => application.SourceRunId == first.RunId && application.ModelKey == "Actions"),
            BodyLeft = await db.TagApplications.CountAsync(application => application.SourceRunId == first.RunId && application.ModelKey == "Body"),
        });
    }

    private static RecordingAiServerClient CreateTaggingAndShotsClient(string? shotsVersion = null)
        => new()
        {
            CatalogModels =
            [
                CreateTaggingModel("tagger-actions-best", ["Actions"], scope: "frame"),
                CreateModel("shots", ["shot_boundaries"], "temporal_segmentation", "asset", version: shotsVersion),
            ],
            EchoRequestedVideoModels = true,
        };

    private static AiCoreOrchestrator CreateOrchestrator(
        INsfwAiServerClient client,
        IAiRunPlanner planner,
        IAiArtifactReplaceService replaceService,
        params IAiCapabilityContributor[] contributors)
        => new(
            client,
            CreateExchange(contributors),
            NoOpAiRunJournal.Instance,
            planner,
            replaceService,
            NullLogger<AiCoreOrchestrator>.Instance);

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
        services.AddScoped<Cove.Core.Interfaces.IAiRunRepository, Cove.Data.Repositories.AiRunRepository>();
        services.AddScoped<Cove.Core.Interfaces.IEmbeddingRepository, Cove.Data.Repositories.EmbeddingRepository>();
        services.AddScoped<Cove.Core.Interfaces.IDetectionRepository, Cove.Data.Repositories.DetectionRepository>();
        services.AddScoped<Cove.Core.Interfaces.IFaceRepository, Cove.Data.Repositories.FaceRepository>();
        services.AddScoped<Cove.Core.Interfaces.ITagApplicationRepository, Cove.Data.Repositories.TagApplicationRepository>();
        services.AddScoped<Cove.Core.Interfaces.ISegmentRepository, Cove.Data.Repositories.SegmentRepository>();
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
        string scope = "asset",
        string? version = null,
        int? identifier = null)
        => new()
        {
            ConfigName = configName,
            Name = configName,
            Categories = categories.ToList(),
            Capabilities = ["tagging"],
            SupportedScopes = [scope],
            Loaded = loaded,
            Active = active,
            Version = version,
            Identifier = identifier,
        };

    private static AiModelCatalogEntry CreateModel(string configName, IReadOnlyList<string> categories, string capability, string scope, bool loaded = true, bool active = true, string? version = null, int? identifier = null)
        => new()
        {
            ConfigName = configName,
            Name = configName,
            Categories = categories.ToList(),
            Capabilities = [capability],
            SupportedScopes = [scope],
            Loaded = loaded,
            Active = active,
            Version = version,
            Identifier = identifier,
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
                    identifier = model.Identifier,
                    version = model.Version,
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

        public Task RecordStartsAsync(IReadOnlyList<AiRunJournalStart> entries, CancellationToken ct = default)
            => Task.CompletedTask;

        public Task RecordCompletionsAsync(IReadOnlyList<AiRunJournalCompletion> completions, CancellationToken ct = default)
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

    // A contributor that also plans its own claim, standing in for a shot-boundary extension.
    private sealed class PlanningStubContributor(
        Func<AiClaimPlanningTarget, IReadOnlyList<AiClaimPlanningDecision>> plan,
        string mediaKind = AiMediaKinds.Video,
        string category = "shot_boundaries") : IAiCapabilityContributor, IAiClaimPlanningContributor
    {
        public const string ClaimId = "shots.asset";

        public List<AiClaimPlanningTarget> Targets { get; } = [];

        public List<AiDispatchRequest> Dispatched { get; } = [];

        public static PlanningStubContributor Always(AiClaimPlanningVerdict verdict, bool replacesExistingResults = false)
            => new(target => target.Claims
                .Select(claim => new AiClaimPlanningDecision
                {
                    ClaimId = claim.ClaimId,
                    Verdict = verdict,
                    Reason = $"stub {verdict}",
                    ReplacesExistingResults = replacesExistingResults,
                })
                .ToArray());

        public AiCapabilityDescriptor Describe()
            => new(
                "cove.community.ai.shots",
                "AI Shots",
                [
                    new AiCapabilityClaim(ClaimId, "Shot Boundaries", mediaKind, "temporal_segmentation", "asset", "shot_boundaries")
                    {
                        CapabilityId = "shots",
                        ModelBindingSlotId = "detector",
                    },
                ])
            {
                Capabilities =
                [
                    new AiCapabilityFeature(
                        "shots",
                        "Shot Boundaries",
                        [ClaimId],
                        [
                            new AiModelBindingSlot(
                                "detector",
                                "Shot boundary model",
                                "temporal_segmentation",
                                RequiredCapabilities: ["temporal_segmentation"],
                                RequiredScopes: ["asset"],
                                RequiredCategories: [category]),
                        ]),
                ],
            };

        public Task<IReadOnlyList<AiClaimPlanningDecision>> PlanClaimsAsync(AiClaimPlanningTarget target, CancellationToken ct = default)
        {
            Targets.Add(target);
            return Task.FromResult(plan(target));
        }

        public Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default)
        {
            Dispatched.Add(request);
            return Task.FromResult(new AiDispatchResult("cove.community.ai.shots", request.Claims.Count));
        }
    }

    private sealed class RecordingContributor(AiCapabilityDescriptor descriptor) : IAiCapabilityContributor
    {
        public List<AiDispatchRequest> Dispatched { get; } = [];

        public AiCapabilityDescriptor Describe() => descriptor;

        public Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default)
        {
            Dispatched.Add(request);
            return Task.FromResult(new AiDispatchResult(descriptor.ExtensionId, request.Claims.Count));
        }
    }

    private sealed record PlannerCall(
        string? HostEntityType,
        int? HostEntityId,
        IReadOnlyList<AiRunPlannerWant> Wants,
        IReadOnlyList<string>? ForceClaimIds,
        double? FrameIntervalSeconds,
        double? Threshold);

    private sealed class SpyAiRunPlanner(IAiRunPlanner inner) : IAiRunPlanner
    {
        public List<PlannerCall> Calls { get; } = [];

        public Task<IReadOnlyList<AiRunExecutionPlan>> PlanAsync(AiCoreConnectionSettings settings, string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunPlannerWant> wants, IReadOnlyList<string>? forceClaimIds, double? frameIntervalSeconds = null, double? threshold = null, CancellationToken ct = default)
        {
            Calls.Add(new PlannerCall(hostEntityType, hostEntityId, wants.ToArray(), forceClaimIds?.ToArray(), frameIntervalSeconds, threshold));
            return inner.PlanAsync(settings, hostEntityType, hostEntityId, wants, forceClaimIds, frameIntervalSeconds, threshold, ct);
        }
    }

    // Plans every want to run, with a reason naming the want's claims, so a test can tell which plan went where.
    private sealed class LabellingAiRunPlanner : IAiRunPlanner
    {
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
                    [$"planned for {string.Join(",", want.Claims.Select(static claim => claim.ClaimId))}"],
                    false)).ToArray());
    }

    private sealed class RecordingAiArtifactReplaceService : IAiArtifactReplaceService
    {
        public List<IReadOnlyList<AiRunExecutionPlan>> Calls { get; } = [];

        public Task ReplaceAsync(string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunExecutionPlan> plans, CancellationToken ct = default)
        {
            Calls.Add(plans);
            return Task.CompletedTask;
        }

        public Task ReplaceBatchAsync(IReadOnlyList<AiArtifactReplaceTarget> targets, CancellationToken ct = default)
        {
            Calls.AddRange(targets.Select(static target => target.Plans));
            return Task.CompletedTask;
        }
    }

    private sealed class NoOpAiArtifactReplaceService : IAiArtifactReplaceService
    {
        public static NoOpAiArtifactReplaceService Instance { get; } = new();

        public Task ReplaceAsync(string? hostEntityType, int? hostEntityId, IReadOnlyList<AiRunExecutionPlan> plans, CancellationToken ct = default)
            => Task.CompletedTask;

        public Task ReplaceBatchAsync(IReadOnlyList<AiArtifactReplaceTarget> targets, CancellationToken ct = default)
            => Task.CompletedTask;
    }
}