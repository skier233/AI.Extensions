using AI.Extensions.Abstractions;

using AI.Faces;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using System.IO;
using System.IO.Compression;
using System.Text;
using System.Text.Json;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiFacesPreparationServiceTests
{
    [Fact]
    public async Task Prepare_DoesNotCreateIdentityForLowQualityUnmatchedFace()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store);
        var request = BuildImageRequest(
            "image-1",
            detectionScore: 0.45,
            embeddingNorm: 12.0,
            vector: [1f, 0f]);

        var batch = await service.PrepareAsync(request);

        Assert.Empty(batch.Faces);
        Assert.Single(batch.Detections);
        Assert.Empty((await store.LoadAsync()).Identities);
    }

    [Fact]
    public async Task Prepare_ReusesExistingIdentityForHighQualityMatchingFace()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());

        var firstBatch = await service.PrepareAsync(BuildImageRequest("image-a", 0.95, 24.0, [1f, 0f]));
        var secondBatch = await service.PrepareAsync(BuildImageRequest("image-b", 0.94, 23.5, [0.999f, 0.001f]));
        var snapshot = await store.LoadAsync();

        Assert.Single(snapshot.Identities);
        Assert.Single(firstBatch.Faces);
        Assert.Single(secondBatch.Faces);
        Assert.Equal(firstBatch.Faces[0].FaceKey, secondBatch.Faces[0].FaceKey);
    }

    [Fact]
    public async Task Prepare_VideoTrackPrefersSharperLargerExemplarForCover()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "video-cover-rank",
            FrameIntervalSeconds = 1,
            Frames =
            [
                new AiTemporalSlice(
                    "frame",
                    1,
                    1,
                    1,
                    1,
                    new AiAnalysisNode
                    {
                        Detections =
                        [
                            new AiDetectionObservation("face_detector_torchexport", 0, "face", 0.98, new AiBoundingBox(0.20, 0.20, 0.38, 0.38)),
                        ],
                        RegionBranches =
                        [
                            new AiRegionBranch(
                                "regions__face_detector_torchexport",
                                0,
                                new AiAnalysisNode
                                {
                                    Embeddings =
                                    [
                                        new AiEmbeddingObservation("face_embedding_torchexport", "region", [1f, 0f], 12.0, 0),
                                    ],
                                }),
                        ],
                    }),
                new AiTemporalSlice(
                    "frame",
                    2,
                    2,
                    2,
                    2,
                    new AiAnalysisNode
                    {
                        Detections =
                        [
                            new AiDetectionObservation("face_detector_torchexport", 0, "face", 0.90, new AiBoundingBox(0.18, 0.18, 0.42, 0.42)),
                        ],
                        RegionBranches =
                        [
                            new AiRegionBranch(
                                "regions__face_detector_torchexport",
                                0,
                                new AiAnalysisNode
                                {
                                    Embeddings =
                                    [
                                        new AiEmbeddingObservation("face_embedding_torchexport", "region", [1f, 0f], 26.0, 0),
                                    ],
                                }),
                        ],
                    }),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "video-cover-rank"));

        var face = Assert.Single(batch.Faces);
        Assert.Equal(new AiBoundingBox(0.18, 0.18, 0.42, 0.42), face.CoverBoundingBox);
    }

    [Fact]
    public async Task Prepare_CreatesIdentityForStrongFaceDespiteLowPoseOrBlurriness()
    {
        // A strong embedding (high ArcFace norm) from a confident detection is a real, recognizable face
        // and must be created even when the per-frame pose/image-quality heuristics are low. Those signals
        // only rank cover/representative selection — they no longer gate a face's existence, so a clear
        // single still is never silently dropped for not being perfectly frontal or razor-sharp.
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store);
        var request = BuildImageRequest(
            "image-low-quality",
            detectionScore: 0.96,
            embeddingNorm: 24.0,
            vector: [1f, 0f],
            embeddingMetadata: new Dictionary<string, string>
            {
                ["pose_quality"] = "0.42",
                ["image_quality"] = "0.08",
            });

        var batch = await service.PrepareAsync(request);

        Assert.Single(batch.Faces);
        Assert.Single((await store.LoadAsync()).Identities);
    }

    [Fact]
    public async Task Prepare_KeepsSingleFrameUnmatchedVideoTrackProvisional()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store);
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "single-frame-fragment",
            FrameIntervalSeconds = 1,
            Frames = [CreateVideoFrame(1, 1, new AiBoundingBox(0.20, 0.20, 0.36, 0.36))],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "single-frame-fragment"));
        var snapshot = await store.LoadAsync();

        Assert.Empty(batch.Faces);
        Assert.Single(batch.Detections);
        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal(StoredFaceIdentityLifecycle.Provisional, identity.LifecycleStatus);
    }

    [Fact]
    public async Task Prepare_PromotesSparseVideoTrackByRepresentedEvidenceTime()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store);
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "sparse-video-evidence",
            DurationSeconds = 180,
            FrameIntervalSeconds = 60,
            Frames =
            [
                CreateVideoFrame(1, 0, new AiBoundingBox(0.20, 0.20, 0.36, 0.36), [1f, 0f]),
                CreateVideoFrame(2, 60, new AiBoundingBox(0.21, 0.20, 0.37, 0.36), [1f, 0f]),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "sparse-video-evidence"));

        var face = Assert.Single(batch.Faces);
        Assert.Equal("video-evidence", face.Metadata!["promotionReason"]);
        Assert.Contains(batch.Notes, note => note.Contains("videoPromotionEvidenceSeconds=24", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Prepare_VideoTrackUsesSampleOrderForSparseSourceFrameIndexes()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "sparse-source-frame-indexes",
            DurationSeconds = 180,
            Frames =
            [
                CreateVideoFrame(1, 0, new AiBoundingBox(0.20, 0.20, 0.36, 0.36), [1f, 0f]),
                CreateVideoFrame(1801, 60, new AiBoundingBox(0.21, 0.20, 0.37, 0.36), [1f, 0f]),
                CreateVideoFrame(3601, 120, new AiBoundingBox(0.22, 0.20, 0.38, 0.36), [1f, 0f]),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "sparse-source-frame-indexes"));

        Assert.Single(batch.Faces);
        var segment = Assert.Single(batch.Segments);
        Assert.Equal(0d, segment.StartSeconds);
        Assert.Equal(180d, segment.EndSeconds);
        Assert.DoesNotContain(":span-", segment.Metadata!["trackKey"], StringComparison.Ordinal);
    }

    [Fact]
    public async Task Prepare_KeepsLowCoverageSparseVideoTrackProvisionalInLongVideo()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store);
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "long-sparse-low-coverage",
            DurationSeconds = 1000,
            FrameIntervalSeconds = 10,
            Frames =
            [
                CreateVideoFrame(1, 0, new AiBoundingBox(0.20, 0.20, 0.36, 0.36), [1f, 0f]),
                CreateVideoFrame(2, 100, new AiBoundingBox(0.21, 0.20, 0.37, 0.36), [1f, 0f]),
                CreateVideoFrame(3, 200, new AiBoundingBox(0.22, 0.20, 0.38, 0.36), [1f, 0f]),
                CreateVideoFrame(4, 300, new AiBoundingBox(0.23, 0.20, 0.39, 0.36), [1f, 0f]),
                CreateVideoFrame(5, 400, new AiBoundingBox(0.24, 0.20, 0.40, 0.36), [1f, 0f]),
                CreateVideoFrame(6, 500, new AiBoundingBox(0.25, 0.20, 0.41, 0.36), [1f, 0f]),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "long-sparse-low-coverage"));
        var snapshot = await store.LoadAsync();

        Assert.Empty(batch.Faces);
        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal(StoredFaceIdentityLifecycle.Provisional, identity.LifecycleStatus);
        Assert.Contains(batch.Notes, note => note.Contains("sparseVideoPromotionCoverageRatio=0.1", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Prepare_DoesNotReemitSingleAssetVideoEvidenceIdentityWithoutCurrentCoverage()
    {
        var store = new InMemoryFaceIdentityStore();
        await store.SaveAsync(new FaceIdentitySnapshot
        {
            Identities =
            [
                new StoredFaceIdentity
                {
                    FaceKey = "face-stale",
                    LifecycleStatus = StoredFaceIdentityLifecycle.Promoted,
                    PromotionReason = "video-evidence",
                    AssetIds = ["stale-asset"],
                    Anchors =
                    [
                        new StoredFaceAnchor
                        {
                            ModelKey = "face_embedding_torchexport",
                            QualityScore = 1,
                            Vector = [1f, 0f],
                        },
                    ],
                },
            ],
        });
        var service = CreateService(store);
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "stale-asset",
            DurationSeconds = 1000,
            FrameIntervalSeconds = 10,
            Frames =
            [
                CreateVideoFrame(1, 0, new AiBoundingBox(0.20, 0.20, 0.36, 0.36), [1f, 0f]),
                CreateVideoFrame(2, 100, new AiBoundingBox(0.21, 0.20, 0.37, 0.36), [1f, 0f]),
                CreateVideoFrame(3, 200, new AiBoundingBox(0.22, 0.20, 0.38, 0.36), [1f, 0f]),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "stale-asset"));

        Assert.Empty(batch.Faces);
        Assert.NotEmpty(batch.Detections);
        Assert.All(batch.Detections, detection => Assert.Null(detection.RefKey));
    }

    [Fact]
    public async Task Prepare_PromotesSingleSparseFrameWhenVideoIsShorterThanInterval()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store);
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "short-sparse-video",
            DurationSeconds = 30,
            FrameIntervalSeconds = 60,
            Frames = [CreateVideoFrame(1, 0, new AiBoundingBox(0.20, 0.20, 0.36, 0.36), [1f, 0f])],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "short-sparse-video"));

        var face = Assert.Single(batch.Faces);
        Assert.Equal("video-evidence", face.Metadata!["promotionReason"]);
    }

    [Fact]
    public async Task Prepare_KeepsSingleSparseFrameInLongVideoProvisional()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store);
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "long-sparse-video",
            DurationSeconds = 600,
            FrameIntervalSeconds = 60,
            Frames = [CreateVideoFrame(1, 0, new AiBoundingBox(0.20, 0.20, 0.36, 0.36), [1f, 0f])],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "long-sparse-video"));
        var snapshot = await store.LoadAsync();

        Assert.Empty(batch.Faces);
        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal(StoredFaceIdentityLifecycle.Provisional, identity.LifecycleStatus);
    }

    [Fact]
    public async Task Prepare_KeepsShortUnmatchedVideoClusterProvisionalByDefault()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store);
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "short-video-fragment",
            FrameIntervalSeconds = 1,
            Frames =
            [
                CreateVideoFrame(1, 1, new AiBoundingBox(0.20, 0.20, 0.36, 0.36), [1f, 0f]),
                CreateVideoFrame(2, 2, new AiBoundingBox(0.21, 0.20, 0.37, 0.36), [1f, 0f]),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "short-video-fragment"));
        var snapshot = await store.LoadAsync();

        Assert.Empty(batch.Faces);
        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal(StoredFaceIdentityLifecycle.Provisional, identity.LifecycleStatus);
        Assert.Contains(batch.Notes, note => note.Contains("AI.Faces telemetry", StringComparison.Ordinal));
        Assert.Contains(batch.Notes, note => note.Contains("videoPromotionSamples=24", StringComparison.Ordinal));
    }

    [Fact]
    public async Task Prepare_ReusesSameAssetIdentityForBorderlineVideoFragment()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "same-asset-fragments",
            FrameIntervalSeconds = 1,
            Frames =
            [
                CreateVideoFrame(1, 1, new AiBoundingBox(0.10, 0.10, 0.30, 0.35), [1f, 0f]),
                CreateVideoFrame(2, 2, new AiBoundingBox(0.10, 0.10, 0.30, 0.35), [1f, 0f]),
                CreateVideoFrame(8, 8, new AiBoundingBox(0.60, 0.10, 0.80, 0.35), [0.50f, 0.8660254f]),
                CreateVideoFrame(9, 9, new AiBoundingBox(0.60, 0.10, 0.80, 0.35), [0.50f, 0.8660254f]),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "same-asset-fragments"));
        var snapshot = await store.LoadAsync();

        var face = Assert.Single(batch.Faces);
        Assert.Single(snapshot.Identities);
        var appearance = Assert.Single(batch.FaceAppearances);
        Assert.Equal(face.FaceKey, appearance.RefKey);
        Assert.Equal(4, appearance.SampleCount);
    }

    [Fact]
    public async Task Prepare_VideoTrackerBridgesLowIoUWhenEmbeddingMatches()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "low-iou-same-face",
            FrameIntervalSeconds = 1,
            Frames =
            [
                CreateVideoFrame(1, 1, new AiBoundingBox(0.05, 0.10, 0.20, 0.30), [1f, 0f]),
                CreateVideoFrame(2, 2, new AiBoundingBox(0.70, 0.12, 0.86, 0.32), [1f, 0f]),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "low-iou-same-face"));

        Assert.Single(batch.Faces);
        var appearance = Assert.Single(batch.FaceAppearances);
        Assert.Equal(2, appearance.SampleCount);
        var segment = Assert.Single(batch.Segments);
        Assert.Equal(1d, segment.StartSeconds);
        Assert.Equal(3d, segment.EndSeconds);
    }

    [Fact]
    public async Task Prepare_VideoTrackerKeepsConcurrentDifferentEmbeddingsSeparate()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "concurrent-different-faces",
            FrameIntervalSeconds = 1,
            Frames =
            [
                CreateVideoFrame(
                    1,
                    1,
                    [
                        (new AiBoundingBox(0.10, 0.10, 0.30, 0.32), (IReadOnlyList<float>)new float[] { 1f, 0f }),
                        (new AiBoundingBox(0.55, 0.10, 0.75, 0.32), (IReadOnlyList<float>)new float[] { 0f, 1f }),
                    ]),
                CreateVideoFrame(
                    2,
                    2,
                    [
                        (new AiBoundingBox(0.12, 0.10, 0.32, 0.32), (IReadOnlyList<float>)new float[] { 1f, 0f }),
                        (new AiBoundingBox(0.57, 0.10, 0.77, 0.32), (IReadOnlyList<float>)new float[] { 0f, 1f }),
                    ]),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "concurrent-different-faces"));

        Assert.Equal(2, batch.Faces.Count);
        Assert.Equal(2, batch.FaceAppearances.Count);
        Assert.All(batch.FaceAppearances, appearance => Assert.Equal(2, appearance.SampleCount));
    }

    [Fact]
    public async Task Prepare_DoesNotMarkBrieflyPresentFaceAsPresentInVideo()
    {
        // A long video with a main face (present for the majority) and a second, different face that
        // only appears for ~2 seconds — the mis-attributed-single-detection / intro-cameo shape. The
        // brief face must not be marked present even though it would otherwise clear promotion.
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var frames = new List<AiTemporalSlice>();
        for (var i = 0; i < 10; i++)
        {
            frames.Add(CreateVideoFrame(i + 1, i, new AiBoundingBox(0.20, 0.20, 0.40, 0.40), [1f, 0f]));
        }

        // Different person, present only at 20s and 21s (~2s total).
        frames.Add(CreateVideoFrame(11, 20, new AiBoundingBox(0.60, 0.20, 0.80, 0.40), [0f, 1f]));
        frames.Add(CreateVideoFrame(12, 21, new AiBoundingBox(0.60, 0.20, 0.80, 0.40), [0f, 1f]));

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "brief-presence-video",
            DurationSeconds = 60,
            FrameIntervalSeconds = 1,
            Frames = frames,
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "brief-presence-video"));

        var face = Assert.Single(batch.Faces);
        Assert.All(batch.FaceAppearances, appearance => Assert.Equal(face.FaceKey, appearance.RefKey));
        // The brief face's detections are still recorded, just not attributed to a present face.
        Assert.Contains(batch.Detections, detection => detection.RefKey is null);
    }

    [Fact]
    public async Task Prepare_DoesNotCreateIdentityForUsableNonAnchorFace()
    {
        // A face that clears the identity floor (strong norm + confident detection) but never produces an
        // anchor-grade embedding — here because the detection is too small to anchor — can still match an
        // existing identity, but must not mint a new one. Tiny crops are the dominant source of junk faces
        // the user just deletes.
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        var request = BuildImageRequest(
            "image-tiny-crop",
            detectionScore: 0.96,
            embeddingNorm: 24.0,
            vector: [1f, 0f],
            // Area below MinimumNormalizedAnchorArea (0.01): 0.05 * 0.06 = 0.003.
            boundingBox: new AiBoundingBox(0.10, 0.10, 0.15, 0.16));

        var batch = await service.PrepareAsync(request);
        var snapshot = await store.LoadAsync();

        Assert.Empty(batch.Faces);
        Assert.Empty(snapshot.Identities);
        Assert.Single(batch.Detections);
        Assert.Null(batch.Detections[0].RefKey);
    }

    [Fact]
    public async Task Prepare_MatchesNonAnchorFaceOntoExistingIdentity()
    {
        var store = new InMemoryFaceIdentityStore();
        await store.SaveAsync(new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 2,
            Identities =
            [
                new StoredFaceIdentity
                {
                    FaceKey = "face-0001",
                    LifecycleStatus = StoredFaceIdentityLifecycle.Promoted,
                    PromotionReason = "image",
                    QualityScore = 32.0,
                    AssetIds = ["image-original"],
                    Anchors =
                    [
                        new StoredFaceAnchor
                        {
                            ModelKey = "face_embedding_torchexport",
                            QualityScore = 20.0,
                            Vector = [1f, 0f],
                        },
                    ],
                },
            ],
        });
        var service = CreateService(store, ShortVideoPromotionSettings());
        var request = BuildImageRequest(
            "image-soft-profile",
            detectionScore: 0.96,
            embeddingNorm: 24.0,
            vector: [1f, 0f],
            embeddingMetadata: new Dictionary<string, string>
            {
                ["pose_quality"] = "0.68",
                ["image_quality"] = "0.40",
            });

        var batch = await service.PrepareAsync(request);
        var snapshot = await store.LoadAsync();

        var face = Assert.Single(batch.Faces);
        Assert.Equal("face-0001", face.FaceKey);
        var identity = Assert.Single(snapshot.Identities);
        Assert.Contains("image-soft-profile", identity.AssetIds);
    }

    [Fact]
    public async Task Prepare_UsesQualityMetadataForCoverAndCentroidSelection()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "video-quality-rank",
            FrameIntervalSeconds = 1,
            Frames =
            [
                new AiTemporalSlice(
                    "frame",
                    1,
                    1,
                    1,
                    1,
                    new AiAnalysisNode
                    {
                        Detections =
                        [
                            new AiDetectionObservation("face_detector_torchexport", 0, "face", 0.92, new AiBoundingBox(0.20, 0.20, 0.34, 0.34)),
                        ],
                        RegionBranches =
                        [
                            new AiRegionBranch(
                                "regions__face_detector_torchexport",
                                0,
                                new AiAnalysisNode
                                {
                                    Embeddings =
                                    [
                                        new AiEmbeddingObservation(
                                            "face_embedding_torchexport",
                                            "region",
                                            [1f, 0f],
                                            18.0,
                                            0,
                                            Metadata: new Dictionary<string, string>
                                            {
                                                ["pose_quality"] = "0.95",
                                                ["image_quality"] = "0.95",
                                            }),
                                    ],
                                }),
                        ],
                    }),
                new AiTemporalSlice(
                    "frame",
                    2,
                    2,
                    2,
                    2,
                    new AiAnalysisNode
                    {
                        Detections =
                        [
                            new AiDetectionObservation("face_detector_torchexport", 0, "face", 0.98, new AiBoundingBox(0.19, 0.19, 0.37, 0.37)),
                        ],
                        RegionBranches =
                        [
                            new AiRegionBranch(
                                "regions__face_detector_torchexport",
                                0,
                                new AiAnalysisNode
                                {
                                    Embeddings =
                                    [
                                        new AiEmbeddingObservation(
                                            "face_embedding_torchexport",
                                            "region",
                                            [0f, 1f],
                                            26.0,
                                            0,
                                            Metadata: new Dictionary<string, string>
                                            {
                                                ["pose_quality"] = "0.71",
                                                ["image_quality"] = "0.19",
                                            }),
                                    ],
                                }),
                        ],
                    }),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "video-quality-rank"));

        var face = Assert.Single(batch.Faces);
        Assert.Equal(new AiBoundingBox(0.20, 0.20, 0.34, 0.34), face.CoverBoundingBox);

        var embedding = Assert.Single(batch.Embeddings);
        Assert.True(embedding.Vector[0] > 0.95f, $"Expected weighted centroid to stay near the sharper frontal sample, got [{embedding.Vector[0]}, {embedding.Vector[1]}].");
        Assert.True(embedding.Vector[1] < 0.3f, $"Expected weighted centroid to down-rank the weaker sample, got [{embedding.Vector[0]}, {embedding.Vector[1]}].");
    }

    [Fact]
    public async Task Prepare_VideoTrackPersistsOnlyRepresentativeDetectionKeyframes()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "video-keyframes",
            FrameIntervalSeconds = 1,
            Frames =
            [
                CreateVideoFrame(1, 1, new AiBoundingBox(0.20, 0.20, 0.36, 0.36)),
                CreateVideoFrame(2, 2, new AiBoundingBox(0.20, 0.20, 0.36, 0.36)),
                CreateVideoFrame(3, 3, new AiBoundingBox(0.202, 0.201, 0.362, 0.361)),
                CreateVideoFrame(4, 4, new AiBoundingBox(0.24, 0.20, 0.40, 0.36)),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "video-keyframes"));

        Assert.Single(batch.Faces);
        Assert.Equal(2, batch.Detections.Count);
        Assert.Equal([1d, 4d], batch.Detections.Select(static detection => detection.ObservedAtSeconds).ToArray());

        var segment = Assert.Single(batch.Segments);
        Assert.Equal(1d, segment.StartSeconds);
        Assert.Equal(5d, segment.EndSeconds);
    }

    [Fact]
    public async Task Prepare_VideoTrackRetainsTemporalKeyframesForLongStableBoxes()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, new AiFacesSettings
        {
            PromotionMinimumVideoSamples = 2,
            DetectionKeyframeMaxGapSeconds = 2.5,
            MaxDetectionKeyframesPerTrack = 20,
        });
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "video-keyframe-gap",
            FrameIntervalSeconds = 1,
            Frames = Enumerable.Range(1, 10)
                .Select(frame => CreateVideoFrame(frame, frame, new AiBoundingBox(0.20, 0.20, 0.36, 0.36)))
                .ToArray(),
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "video-keyframe-gap"));

        Assert.Single(batch.Faces);
        Assert.Equal([1d, 4d, 7d, 10d], batch.Detections.Select(static detection => detection.ObservedAtSeconds).ToArray());

        var segment = Assert.Single(batch.Segments);
        Assert.Equal("4", segment.Metadata?["retainedSpatialSampleCount"]);
        Assert.Equal("2.5", segment.Metadata?["detectionKeyframeMaxGapSec"]);
    }

    [Fact]
    public async Task Prepare_SplitsFaceSegmentsAcrossLargeObservationGapsInSameAssetCluster()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, ShortVideoPromotionSettings());
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
            new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
        ];

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "video-split-segments",
            FrameIntervalSeconds = 2,
            Frames =
            [
                CreateVideoFrame(1, 1, new AiBoundingBox(0.20, 0.20, 0.36, 0.36), [1f, 0f]),
                CreateVideoFrame(2, 3, new AiBoundingBox(0.21, 0.20, 0.37, 0.36), [1f, 0f]),
                CreateVideoFrame(40, 80, new AiBoundingBox(0.58, 0.18, 0.74, 0.36), [1f, 0f]),
                CreateVideoFrame(41, 82, new AiBoundingBox(0.59, 0.18, 0.75, 0.36), [1f, 0f]),
            ],
        };

        var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "video-split-segments"));

        Assert.Single(batch.Faces);
        Assert.Equal(2, batch.Segments.Count);
        Assert.Equal([(1d, 5d), (80d, 84d)], batch.Segments.Select(static segment => (segment.StartSeconds, segment.EndSeconds ?? segment.StartSeconds)).ToArray());
        Assert.Equal(2, batch.Segments.Select(static segment => segment.Metadata!["trackKey"]).Distinct(StringComparer.Ordinal).Count());
        Assert.All(batch.Detections, detection => Assert.Contains(":span-", detection.GroupKey, StringComparison.Ordinal));
    }

    [Fact]
    public async Task Prepare_UsesConfiguredClusterMatchThresholdToCreateNewIdentity()
    {
        var store = new InMemoryFaceIdentityStore();
        var service = CreateService(store, new AiFacesSettings
        {
            IdentityMatchThreshold = 0.99,
            IdentityAmbiguityMargin = 0.01,
            // Pin consolidation above the test vectors' similarity so reconciliation doesn't
            // immediately re-merge the two identities this test expects to stay distinct.
            ConsolidationSimilarityThreshold = 0.99,
            ConsolidationPromotedSimilarityThreshold = 0.99,
        });

        var firstBatch = await service.PrepareAsync(BuildImageRequest("image-first", 0.95, 24.0, [1f, 0f]));
        var secondBatch = await service.PrepareAsync(BuildImageRequest("image-second", 0.95, 24.0, [0.88f, 0.475f]));
        var snapshot = await store.LoadAsync();

        Assert.Equal(2, snapshot.Identities.Count);
        Assert.Single(firstBatch.Faces);
        Assert.Single(secondBatch.Faces);
        Assert.NotEqual(firstBatch.Faces[0].FaceKey, secondBatch.Faces[0].FaceKey);
    }

    [Fact]
    public async Task Prepare_UsesReferencePackMatchToPromoteAndLabelIdentity()
    {
        var tempRoot = Path.Combine(Path.GetTempPath(), $"ai-faces-prep-reference-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);

        try
        {
            var referencePackStore = await CreateReferencePackStoreAsync(tempRoot);
            var store = new InMemoryFaceIdentityStore();
            var service = CreateService(store, referencePackStore: referencePackStore);

            var batch = await service.PrepareAsync(BuildImageRequest("image-reference-seed", 0.95, 24.0, [1f, 0f]));
            var snapshot = await store.LoadAsync();

            var face = Assert.Single(batch.Faces);
            Assert.Equal("Reference Performer", face.Label);
            Assert.False(face.IsProvisional);
            Assert.NotNull(face.Metadata);
            Assert.Equal("reference-performer-1", face.Metadata!["referenceExternalId"]);

            var identity = Assert.Single(snapshot.Identities);
            Assert.Equal(StoredFaceIdentityLifecycle.Promoted, identity.LifecycleStatus);
            Assert.Equal("reference", identity.PromotionReason);
            Assert.Equal("reference-performer-1", identity.ReferenceExternalId);
        }
        finally
        {
            if (Directory.Exists(tempRoot))
            {
                Directory.Delete(tempRoot, recursive: true);
            }
        }
    }

    [Fact]
    public async Task Prepare_ReplacesMalformedStoredCoverWhenMatchingIdentity()
    {
        var store = new InMemoryFaceIdentityStore();
        await store.SaveAsync(new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 2,
            Identities =
            [
                new StoredFaceIdentity
                {
                    FaceKey = "face-0001",
                    LifecycleStatus = StoredFaceIdentityLifecycle.Promoted,
                    QualityScore = 32.0,
                    CoverAssetId = "stale-image",
                    CoverBoundingBox = new StoredBoundingBox(0.31, 0.14, 0.07, 0.16),
                    Anchors =
                    [
                        new StoredFaceAnchor
                        {
                            ModelKey = "arcface_512",
                            QualityScore = 32.0,
                            Vector = [1f, 0f],
                        },
                    ],
                },
            ],
        });

        var service = CreateService(store);

        var batch = await service.PrepareAsync(BuildImageRequest("fresh-image", 0.95, 24.0, [1f, 0f]));

        var face = Assert.Single(batch.Faces);
        Assert.Equal("face-0001", face.FaceKey);
        Assert.Equal("fresh-image", face.CoverAssetId);
        Assert.Equal(new AiBoundingBox(0.1, 0.1, 0.3, 0.4), face.CoverBoundingBox);

        var snapshot = await store.LoadAsync();
        var identity = Assert.Single(snapshot.Identities);
        Assert.Equal("fresh-image", identity.CoverAssetId);
        Assert.NotNull(identity.CoverBoundingBox);
        Assert.True(identity.CoverBoundingBox!.X2 > identity.CoverBoundingBox.X1);
        Assert.True(identity.CoverBoundingBox.Y2 > identity.CoverBoundingBox.Y1);
    }

    [Fact]
    public async Task CreateAsync_RepairsWidthHeightStyleBoundingBoxForImageCover()
    {
        var imagePath = Path.Combine(Path.GetTempPath(), $"ai-face-cover-test-{Guid.NewGuid():N}.png");

        try
        {
            using (var image = new Image<Rgba32>(100, 100, new Rgba32(255, 255, 255)))
            {
                for (var y = 14; y < 30; y++)
                {
                    for (var x = 31; x < 38; x++)
                    {
                        image[x, y] = new Rgba32(255, 0, 0);
                    }
                }

                await image.SaveAsPngAsync(imagePath);
            }

            var preparedFace = new AiPreparedFaceIdentity(
                "face-0001",
                "ext:ai.faces",
                Label: null,
                IsProvisional: true,
                QualityScore: 1.0,
                CoverAssetId: imagePath,
                CoverBoundingBox: new AiBoundingBox(0.31, 0.14, 0.07, 0.16),
                Metadata: null);

            await using var coverStream = await AiFaceCoverGenerator.CreateAsync("image", preparedFace, coverDetection: null, configuration: null);

            Assert.NotNull(coverStream);

            using var cover = await Image.LoadAsync<Rgba32>(coverStream!);
            var centerPixel = cover[cover.Width / 2, cover.Height / 2];

            Assert.Equal(768, cover.Width);
            Assert.Equal(768, cover.Height);
            Assert.True(centerPixel.R > 200 && centerPixel.G < 80 && centerPixel.B < 80,
                $"Expected repaired crop center to land on the red face marker, got R={centerPixel.R}, G={centerPixel.G}, B={centerPixel.B}.");
        }
        finally
        {
            if (File.Exists(imagePath))
            {
                File.Delete(imagePath);
            }
        }
    }

    [Fact]
    public async Task CreateAsync_ReturnsNullForBlankBlackImageCover()
    {
        var imagePath = Path.Combine(Path.GetTempPath(), $"ai-face-cover-blank-{Guid.NewGuid():N}.png");

        try
        {
            using (var image = new Image<Rgba32>(100, 100, new Rgba32(0, 0, 0)))
            {
                await image.SaveAsPngAsync(imagePath);
            }

            var preparedFace = new AiPreparedFaceIdentity(
                "face-0001",
                "ext:ai.faces",
                Label: null,
                IsProvisional: true,
                QualityScore: 1.0,
                CoverAssetId: imagePath,
                CoverBoundingBox: new AiBoundingBox(0.20, 0.20, 0.36, 0.36),
                Metadata: null);

            await using var coverStream = await AiFaceCoverGenerator.CreateAsync("image", preparedFace, coverDetection: null, configuration: null);

            Assert.Null(coverStream);
        }
        finally
        {
            if (File.Exists(imagePath))
            {
                File.Delete(imagePath);
            }
        }
    }

        [Fact]
        public async Task Prepare_VideoTrackCreatesIdentityWhenBranchDetectionIndexLivesInOtherBlock()
        {
                using var document = JsonDocument.Parse(
                        """
                        {
                            "asset_id": "video-23",
                            "frame_interval_seconds": 60,
                            "frames": [
                                {
                                    "frame_index": 120,
                                    "time_seconds": 120,
                                    "analysis": {
                                        "capabilities": {
                                            "detection": {
                                                "face_detections": [
                                                    {
                                                        "bbox": [0.3163, 0.1435, 0.3897, 0.3070],
                                                        "score": 0.874,
                                                        "detector": "face_rec_detector"
                                                    }
                                                ]
                                            }
                                        },
                                        "region_branches": {
                                            "regions__face_detector_torchexport": [
                                                {
                                                    "capabilities": {
                                                        "embedding": {
                                                            "face_embeddings": [
                                                                {
                                                                    "vector": [1.0, 0.0],
                                                                    "norm": 22.5,
                                                                    "embedder": "face_rec_embedder"
                                                                }
                                                            ]
                                                        }
                                                    },
                                                    "other": {
                                                        "detection_index": 0
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                },
                                {
                                    "frame_index": 121,
                                    "time_seconds": 121,
                                    "analysis": {
                                        "capabilities": {
                                            "detection": {
                                                "face_detections": [
                                                    {
                                                        "bbox": [0.3163, 0.1435, 0.3897, 0.3070],
                                                        "score": 0.874,
                                                        "detector": "face_rec_detector"
                                                    }
                                                ]
                                            }
                                        },
                                        "region_branches": {
                                            "regions__face_detector_torchexport": [
                                                {
                                                    "capabilities": {
                                                        "embedding": {
                                                            "face_embeddings": [
                                                                {
                                                                    "vector": [1.0, 0.0],
                                                                    "norm": 22.5,
                                                                    "embedder": "face_rec_embedder"
                                                                }
                                                            ]
                                                        }
                                                    },
                                                    "other": {
                                                        "detection_index": 0
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                }
                            ]
                        }
                        """);

                var result = AiAnalyzeResultParser.Parse(AiMediaKinds.Video, document.RootElement);
                var embeddings = result.Frames[0].Analysis.EnumerateAllEmbeddings().ToArray();

                Assert.Single(embeddings);
                Assert.Equal(0, embeddings[0].DetectionIndex);

                var store = new InMemoryFaceIdentityStore();
                var service = CreateService(store, ShortVideoPromotionSettings());
                IReadOnlyList<AiCapabilityClaim> claims =
                [
                        new AiCapabilityClaim("faces.video.detection", "Video Face Detection", AiMediaKinds.Video, "detection", "frame", "frames"),
                        new AiCapabilityClaim("faces.video.embedding", "Video Face Identity Embeddings", AiMediaKinds.Video, "embedding", "region", "regions", FromDetection: "face_detector_torchexport"),
                ];

                var batch = await service.PrepareAsync(AiTestData.CreateRequest(AiMediaKinds.Video, claims, result, "video-23"));

                Assert.Single(batch.Faces);
                Assert.Single(batch.Embeddings);
                Assert.Single(batch.Detections);
        }

    private static AiDispatchRequest BuildImageRequest(
        string assetId,
        double detectionScore,
        double embeddingNorm,
        IReadOnlyList<float> vector,
        IReadOnlyDictionary<string, string>? embeddingMetadata = null,
        AiBoundingBox? boundingBox = null)
    {
        var bbox = boundingBox ?? new AiBoundingBox(0.1, 0.1, 0.3, 0.4);
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("faces.image.detection", "Image Face Detection", AiMediaKinds.Image, "detection", "asset", "regions"),
            new AiCapabilityClaim("faces.image.embedding", "Image Face Identity Embeddings", AiMediaKinds.Image, "embedding", "region", "regions", FromDetection: "face"),
        ];
        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Image,
            AssetId = assetId,
            AssetAnalysis = new AiAnalysisNode
            {
                Detections = [new AiDetectionObservation("scrfd_face", 0, "face", detectionScore, bbox)],
                RegionBranches =
                [
                    new AiRegionBranch(
                        "regions__face",
                        0,
                        new AiAnalysisNode
                        {
                            Embeddings = [new AiEmbeddingObservation("arcface_512", "region", vector, embeddingNorm, 0, Metadata: embeddingMetadata)],
                        }),
                ],
            },
        };

        return AiTestData.CreateRequest(AiMediaKinds.Image, claims, result, assetId);
    }

    private static AiTemporalSlice CreateVideoFrame(int frameIndex, double timeSeconds, AiBoundingBox boundingBox, IReadOnlyList<float>? vector = null)
        => new(
            "frame",
            frameIndex,
            timeSeconds,
            timeSeconds,
            timeSeconds,
            new AiAnalysisNode
            {
                Detections =
                [
                    new AiDetectionObservation("face_detector_torchexport", 0, "face", 0.97, boundingBox),
                ],
                RegionBranches =
                [
                    new AiRegionBranch(
                        "regions__face_detector_torchexport",
                        0,
                        new AiAnalysisNode
                        {
                            Embeddings =
                            [
                                new AiEmbeddingObservation("face_embedding_torchexport", "region", vector ?? [1f, 0f], 24.0, 0),
                            ],
                        }),
                ],
            });

    private static AiTemporalSlice CreateVideoFrame(int frameIndex, double timeSeconds, IReadOnlyList<(AiBoundingBox BoundingBox, IReadOnlyList<float> Vector)> faces)
        => new(
            "frame",
            frameIndex,
            timeSeconds,
            timeSeconds,
            timeSeconds,
            new AiAnalysisNode
            {
                Detections = faces
                    .Select((face, index) => new AiDetectionObservation("face_detector_torchexport", index, "face", 0.97, face.BoundingBox))
                    .ToArray(),
                RegionBranches = faces
                    .Select((face, index) => new AiRegionBranch(
                        "regions__face_detector_torchexport",
                        index,
                        new AiAnalysisNode
                        {
                            Embeddings =
                            [
                                new AiEmbeddingObservation("face_embedding_torchexport", "region", face.Vector, 24.0, index),
                            ],
                        }))
                    .ToArray(),
            });

    private static AiFacePreparationService CreateService(
        InMemoryFaceIdentityStore store,
        AiFacesSettings? settings = null,
        AiFaceReferencePackStore? referencePackStore = null)
    {
        AiFacesSettingsRuntime.Attach(new InMemoryAiFacesSettingsStore(settings));
        return new AiFacePreparationService(store, new AiAssetFaceClusterer(), new AiFaceIdentityReconciler(), referencePackStore);
    }

    private static AiFacesSettings ShortVideoPromotionSettings()
        => new()
        {
            PromotionMinimumVideoSamples = 2,
        };

    private static async Task<AiFaceReferencePackStore> CreateReferencePackStoreAsync(string root)
    {
        var archivePath = Path.Combine(root, "reference-pack.saie");
        CreateArchive(
            archivePath,
            new
            {
                version = 1,
                embedder = "test-face-embedder",
                embedding_dim = 2,
                pack_id = "test-reference-pack",
                source_endpoint = "https://stashdb.org/graphql",
                performer_count = 2,
                created_at = "2026-03-24T02:43:03Z",
            },
            [
                new
                {
                    stashdb_id = "reference-performer-1",
                    name = "Reference Performer",
                    aliases = Array.Empty<string>(),
                    disambiguation = (string?)null,
                    sample_count = 4,
                    quality_score = 20.0,
                    image_url = "https://stashdb.org/images/reference-1.jpg",
                },
                new
                {
                    stashdb_id = "reference-performer-2",
                    name = "Other Performer",
                    aliases = Array.Empty<string>(),
                    disambiguation = (string?)null,
                    sample_count = 4,
                    quality_score = 20.0,
                    image_url = "https://stashdb.org/images/reference-2.jpg",
                },
            ],
            [1f, 0f, 0f, 1f],
            2,
            2);

        var store = new AiFaceReferencePackStore(Path.Combine(root, "reference"), new SaieArchiveReader());
        store.Attach(new TestExtensionStore());
        await using var source = File.OpenRead(archivePath);
        var stagedPath = await store.StageUploadAsync(source, Path.GetFileName(archivePath));
        await store.ImportStagedAsync(stagedPath, Path.GetFileName(archivePath));
        return store;
    }

    private static void CreateArchive(string path, object manifest, IReadOnlyList<object> performers, IReadOnlyList<float> centroids, int rows, int columns)
    {
        using var file = File.Create(path);
        using var archive = new ZipArchive(file, ZipArchiveMode.Create, leaveOpen: false);

        var manifestEntry = archive.CreateEntry("manifest.json", CompressionLevel.Fastest);
        using (var writer = new StreamWriter(manifestEntry.Open(), Encoding.UTF8, leaveOpen: false))
        {
            writer.Write(JsonSerializer.Serialize(manifest));
        }

        var performersEntry = archive.CreateEntry("performers.jsonl", CompressionLevel.Fastest);
        using (var writer = new StreamWriter(performersEntry.Open(), Encoding.UTF8, leaveOpen: false))
        {
            foreach (var performer in performers)
            {
                writer.WriteLine(JsonSerializer.Serialize(performer));
            }
        }

        var centroidsEntry = archive.CreateEntry("centroids.npy", CompressionLevel.NoCompression);
        using var stream = centroidsEntry.Open();
        WriteNpy(stream, centroids, rows, columns);
    }

    private static void WriteNpy(Stream stream, IReadOnlyList<float> values, int rows, int columns)
    {
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);

        writer.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 0x01, 0x00 });

        var header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {columns}), }}";
        var preambleLength = 10;
        var paddingLength = 16 - ((preambleLength + header.Length + 1) % 16);
        if (paddingLength == 16)
        {
            paddingLength = 0;
        }

        var paddedHeader = header + new string(' ', paddingLength) + '\n';
        var headerBytes = Encoding.ASCII.GetBytes(paddedHeader);

        writer.Write((ushort)headerBytes.Length);
        writer.Write(headerBytes);

        foreach (var value in values)
        {
            writer.Write(value);
        }
    }

    private sealed class TestExtensionStore : Cove.Plugins.IExtensionStore
    {
        private readonly Dictionary<string, string> _entries = new(StringComparer.OrdinalIgnoreCase);

        public Task<string?> GetAsync(string key, CancellationToken ct = default)
            => Task.FromResult<string?>(_entries.TryGetValue(key, out var value) ? value : null);

        public Task SetAsync(string key, string value, CancellationToken ct = default)
        {
            _entries[key] = value;
            return Task.CompletedTask;
        }

        public Task DeleteAsync(string key, CancellationToken ct = default)
        {
            _entries.Remove(key);
            return Task.CompletedTask;
        }

        public Task<Dictionary<string, string>> GetAllAsync(CancellationToken ct = default)
            => Task.FromResult(new Dictionary<string, string>(_entries, StringComparer.OrdinalIgnoreCase));
    }

    private sealed class InMemoryAiFacesSettingsStore(AiFacesSettings? settings = null) : IAiFacesSettingsStore
    {
        private AiFacesSettings _settings = (settings ?? new AiFacesSettings()).Normalize();

        public Task<AiFacesSettings> LoadAsync(CancellationToken ct = default)
            => Task.FromResult(new AiFacesSettings
            {
                IdentityMatchThreshold = _settings.IdentityMatchThreshold,
                IdentityAmbiguityMargin = _settings.IdentityAmbiguityMargin,
                AssetClusterSimilarityThreshold = _settings.AssetClusterSimilarityThreshold,
                AssetClusterAmbiguityMargin = _settings.AssetClusterAmbiguityMargin,
                ReferenceMatchThreshold = _settings.ReferenceMatchThreshold,
                ReferenceAmbiguityMargin = _settings.ReferenceAmbiguityMargin,
                PromotionMinimumVideoSamples = _settings.PromotionMinimumVideoSamples,
                PromotionMinimumVideoEvidenceSeconds = _settings.PromotionMinimumVideoEvidenceSeconds,
                PromotionMinimumSparseVideoSamples = _settings.PromotionMinimumSparseVideoSamples,
                SparseVideoPromotionFrameIntervalSeconds = _settings.SparseVideoPromotionFrameIntervalSeconds,
                ConsolidationSimilarityThreshold = _settings.ConsolidationSimilarityThreshold,
                ConsolidationSameAssetSimilarityThreshold = _settings.ConsolidationSameAssetSimilarityThreshold,
                ConsolidationAmbiguityMargin = _settings.ConsolidationAmbiguityMargin,
                DetectionKeyframeIoUThreshold = _settings.DetectionKeyframeIoUThreshold,
                MaxDetectionKeyframesPerTrack = _settings.MaxDetectionKeyframesPerTrack,
                DetectionKeyframeMaxGapSeconds = _settings.DetectionKeyframeMaxGapSeconds,
            }.Normalize());

        public Task SaveAsync(AiFacesSettings settings, CancellationToken ct = default)
        {
            _settings = settings.Normalize();
            return Task.CompletedTask;
        }
    }

    // In-memory IFaceIdentityStore for unit tests. Loads the full snapshot into a deep-cloned working
    // copy on Begin and writes it back on Commit (so a test fails if prep forgets to commit). LoadAsync/
    // SaveAsync remain as test seed/assert helpers.
    private sealed class InMemoryFaceIdentityStore : IFaceIdentityStore
    {
        private FaceIdentitySnapshot _snapshot = new();

        public Task<FaceIdentitySnapshot> LoadAsync(CancellationToken ct = default)
            => Task.FromResult(_snapshot);

        public Task SaveAsync(FaceIdentitySnapshot snapshot, CancellationToken ct = default)
        {
            _snapshot = snapshot;
            return Task.CompletedTask;
        }

        public Task<FaceIdentityTransaction> BeginIncrementalAsync(
            IReadOnlyList<IReadOnlyList<float>> queryVectors,
            IReadOnlyCollection<string> referenceExternalIds,
            int candidateK,
            CancellationToken ct = default)
            => Begin();

        public Task<FaceIdentityTransaction> BeginFullAsync(CancellationToken ct = default)
            => Begin();

        public Task DeleteByFaceKeyAsync(string faceKey, CancellationToken ct = default)
        {
            _snapshot.Identities.RemoveAll(identity => string.Equals(identity.FaceKey, faceKey, StringComparison.Ordinal));
            return Task.CompletedTask;
        }

        public Task ClearAllAsync(CancellationToken ct = default)
        {
            _snapshot = new FaceIdentitySnapshot();
            return Task.CompletedTask;
        }

        private Task<FaceIdentityTransaction> Begin()
            => Task.FromResult<FaceIdentityTransaction>(new InMemoryTransaction(Clone(_snapshot), committed => _snapshot = committed));

        private static FaceIdentitySnapshot Clone(FaceIdentitySnapshot source)
            => new()
            {
                NextIdentityOrdinal = source.NextIdentityOrdinal,
                Identities = source.Identities.Select(static identity => new StoredFaceIdentity
                {
                    FaceKey = identity.FaceKey,
                    Label = identity.Label,
                    LifecycleStatus = identity.LifecycleStatus,
                    PromotionReason = identity.PromotionReason,
                    ReferenceExternalId = identity.ReferenceExternalId,
                    ReferenceDisplayName = identity.ReferenceDisplayName,
                    ReferencePackId = identity.ReferencePackId,
                    ReferenceSuggestionId = identity.ReferenceSuggestionId,
                    QualityScore = identity.QualityScore,
                    CoverAssetId = identity.CoverAssetId,
                    CoverBoundingBox = identity.CoverBoundingBox,
                    CoverQualityScore = identity.CoverQualityScore,
                    ObservationCount = identity.ObservationCount,
                    AssetIds = [.. identity.AssetIds],
                    Anchors = identity.Anchors.Select(static anchor => new StoredFaceAnchor
                    {
                        ModelKey = anchor.ModelKey,
                        QualityScore = anchor.QualityScore,
                        Vector = [.. anchor.Vector],
                    }).ToList(),
                }).ToList(),
            };

        private sealed class InMemoryTransaction(FaceIdentitySnapshot snapshot, Action<FaceIdentitySnapshot> commit) : FaceIdentityTransaction
        {
            private readonly Action<FaceIdentitySnapshot> _commit = commit;

            public override FaceIdentitySnapshot Snapshot { get; } = snapshot;

            public override Task CommitAsync(CancellationToken ct = default)
            {
                _commit(Snapshot);
                return Task.CompletedTask;
            }

            public override ValueTask DisposeAsync() => ValueTask.CompletedTask;
        }
    }
}