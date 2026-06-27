using AI.Audio;
using AI.Extensions.Abstractions;
using AI.Faces;
using AI.Tagging;
using AI.Visual;

using Cove.Core.DTOs;
using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Cove.Data;
using Cove.Data.Repositories;
using Cove.Plugins;

using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiPersistenceServiceTests
{
    [Fact]
    public async Task PersistFaces_WritesFaceDetectionsSegmentsAndEmbeddings()
    {
        await using var provider = CreateProvider();
        var service = new AiFacesPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
        var request = CreateRequest(AiMediaKinds.Video, "video-asset-1", "video", 42, "run-faces");
        var batch = new AiPreparedArtifactBatch();

        batch.Faces.Add(new AiPreparedFaceIdentity(
            "face-1",
            "ext:ai.faces",
            Label: "Face 1",
            IsProvisional: false,
            QualityScore: 0.91,
            CoverAssetId: "video-asset-1",
            CoverBoundingBox: new AiBoundingBox(0.1, 0.2, 0.3, 0.4)));
        batch.Detections.Add(new AiPreparedDetection(
            "video-asset-1",
            "ext:ai.faces",
            Class: "face",
            ObservedAtSeconds: 10.0,
            Score: 0.9,
            BoundingBox: new AiBoundingBox(0.1, 0.2, 0.3, 0.4),
            ModelKey: "scrfd_face",
            RefKind: "face",
            RefKey: "face-1",
            GroupKey: "track-1"));
        batch.Embeddings.Add(new AiPreparedEmbedding(
            "video-asset-1",
            "ext:ai.faces",
            "face.embed.v1",
            "face.v1",
            "Face",
            false,
            [1f, 0f],
            1.0,
            HostRefKind: "face",
            HostRefKey: "face-1",
            StartSeconds: 10.0,
            EndSeconds: 12.0,
            ModelKey: "arcface_512"));
        batch.Segments.Add(new AiPreparedSegment(
            "video-asset-1",
            "ext:ai.faces",
            Kind: "face",
            StartSeconds: 10.0,
            EndSeconds: 12.0,
            Title: "Face 1",
            Confidence: 0.9,
            RefKind: "face",
            RefKey: "face-1"));

        var notes = await service.PersistAsync(request, batch);

        await using var scope = provider.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var face = await db.Faces.SingleAsync();
        var detection = await db.Detections.SingleAsync();
        var embedding = await db.Embeddings.SingleAsync();
        var segment = await db.Segments.SingleAsync();

        Assert.Equal("face-1", face.PrimarySourceKey);
        Assert.Equal(1, face.DetectionCount);
        Assert.Equal(1, face.VideoCount);
        Assert.Equal(0, face.ImageCount);

        Assert.Equal(DetectionHostType.Video, detection.HostType);
        Assert.Equal(42, detection.HostId);
        Assert.Equal((long)face.Id, detection.RefId);
        Assert.Equal(1, detection.FrameWidth);
        Assert.Equal(1, detection.FrameHeight);
        Assert.InRange(detection.X, 0.099f, 0.101f);
        Assert.InRange(detection.W, 0.199f, 0.201f);

        Assert.Equal(EmbeddingHostType.Face, embedding.HostType);
        Assert.Equal(face.Id, embedding.HostId);
        Assert.Equal(EmbeddingModality.Face, embedding.Modality);
        Assert.Equal("face.v1", embedding.KindFamily);

        Assert.Equal(SegmentHostType.Video, segment.HostType);
        Assert.Equal(42, segment.HostId);
        Assert.Equal((long)face.Id, segment.RefId);
        Assert.Equal("face", segment.Kind);

        Assert.Contains(notes, note => note.Contains("Persisted 1 retained AI-generated face spatial sample", StringComparison.Ordinal));
        Assert.Contains(notes, note => note.Contains("Persisted 1 face embedding", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PersistFaces_SkipsExplicitlyProvisionalFaceRows()
    {
        await using var provider = CreateProvider();
        var service = new AiFacesPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
        var request = CreateRequest(AiMediaKinds.Video, "video-asset-provisional", "video", 43, "run-provisional");
        var batch = new AiPreparedArtifactBatch();

        batch.Faces.Add(new AiPreparedFaceIdentity(
            "face-provisional-1",
            "ext:ai.faces",
            Label: "Provisional Face",
            IsProvisional: true,
            QualityScore: 0.91,
            Metadata: new Dictionary<string, string>
            {
                ["lifecycle"] = "provisional",
            }));
        batch.Detections.Add(new AiPreparedDetection(
            "video-asset-provisional",
            "ext:ai.faces",
            Class: "face",
            ObservedAtSeconds: 10.0,
            Score: 0.9,
            BoundingBox: new AiBoundingBox(0.1, 0.2, 0.3, 0.4),
            ModelKey: "scrfd_face",
            RefKind: "face",
            RefKey: "face-provisional-1",
            GroupKey: "track-1"));

        await service.PersistAsync(request, batch);

        await using var scope = provider.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        Assert.Empty(await db.Faces.ToListAsync());
        Assert.Empty(await db.Detections.ToListAsync());
    }

    [Fact]
    public async Task PersistVisual_WritesVideoEmbeddings()
    {
        await using var provider = CreateProvider();
        var service = new AiVisualPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
        var request = CreateRequest(AiMediaKinds.Video, "video-visual", "video", 7, "run-visual");
        var batch = new AiPreparedArtifactBatch();

        batch.Embeddings.Add(new AiPreparedEmbedding(
            "video-visual",
            "ext:ai.visual",
            "visual.feature.v1",
            "feature.v1",
            "Visual",
            false,
            [1f, 0f],
            1.0,
            SectionIndex: 0,
            ModelKey: "visual"));
        batch.Embeddings.Add(new AiPreparedEmbedding(
            "video-visual",
            "ext:ai.visual",
            "visual.feature.v1",
            "feature.v1",
            "Visual",
            false,
            [0f, 1f],
            1.0,
            SectionIndex: 1,
            StartSeconds: 0.0,
            EndSeconds: 3.0,
            ModelKey: "visual"));

        var notes = await service.PersistAsync(request, batch);

        await using var scope = provider.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var embeddings = await db.Embeddings.OrderBy(embedding => embedding.SectionIndex).ToListAsync();

        Assert.Equal(2, embeddings.Count);
        Assert.All(embeddings, embedding =>
        {
            Assert.Equal(EmbeddingHostType.Video, embedding.HostType);
            Assert.Equal(7, embedding.HostId);
            Assert.Equal(EmbeddingModality.Visual, embedding.Modality);
            Assert.Equal("ext:ai.visual", embedding.SourceKey);
        });
        Assert.Equal(1, embeddings[1].SectionIndex);
        Assert.Equal(0.0, embeddings[1].StartSec);
        Assert.Equal(3.0, embeddings[1].EndSec);
        Assert.Contains(notes, note => note.Contains("Persisted 2 AI-generated visual embedding", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PersistAudio_WritesEmbeddingsAndClearsLegacySegments()
    {
        await using var provider = CreateProvider();
        var service = new AiAudioPersistenceService(
            provider.GetRequiredService<IServiceScopeFactory>(),
            NullLogger<AiAudioPersistenceService>.Instance);
        var request = CreateRequest(AiMediaKinds.Audio, "video-audio", "video", 9, "run-audio");

        // Seed a leftover classification segment from an older AI.Audio run. Re-processing should
        // sweep it away now that AI.Audio is embeddings-only.
        await using (var seedScope = provider.CreateAsyncScope())
        {
            var seedDb = seedScope.ServiceProvider.GetRequiredService<CoveContext>();
            seedDb.Segments.Add(new Segment
            {
                HostType = SegmentHostType.Video,
                HostId = 9,
                StartSec = 4.0,
                EndSec = 7.0,
                Kind = "audio-classification",
                SourceKey = "ext:ai.audio",
                Title = "speech",
            });
            await seedDb.SaveChangesAsync();
        }

        var batch = new AiPreparedArtifactBatch();
        batch.Embeddings.Add(new AiPreparedEmbedding(
            "video-audio",
            "ext:ai.audio",
            "audio.embed.v1",
            "audio.v1",
            "Audio",
            false,
            [0.25f, 0.75f],
            0.79,
            SectionIndex: 1,
            StartSeconds: 4.0,
            EndSeconds: 7.0,
            ModelKey: "audioembed"));

        var notes = await service.PersistAsync(request, batch);

        await using var scope = provider.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var embedding = await db.Embeddings.SingleAsync();

        Assert.Equal(EmbeddingHostType.Video, embedding.HostType);
        Assert.Equal(9, embedding.HostId);
        Assert.Equal(EmbeddingModality.Audio, embedding.Modality);
        Assert.Equal("audio.v1", embedding.KindFamily);
        Assert.Equal("ext:ai.audio", embedding.SourceKey);

        Assert.False(await db.Segments.AnyAsync());

        Assert.Contains(notes, note => note.Contains("Persisted 1 AI-generated audio embedding", StringComparison.Ordinal));
        Assert.Contains(notes, note => note.Contains("Removed 1 legacy audio classification segment", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PersistTagging_WritesImageTagProvenance()
    {
        await using var provider = CreateProvider(services =>
        {
            services.AddScoped<ITagProvenanceService, TestTagProvenanceService>();
        });

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Images.Add(new Cove.Core.Entities.Image { Id = 11, Title = "Tagged still" });
            await db.SaveChangesAsync();
        }

        var service = new AiTaggingPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
        var request = CreateRequest(AiMediaKinds.Image, "image-tagging", "image", 11, "run-tagging");
        var batch = new AiPreparedArtifactBatch();

        batch.TagLinks.Add(new AiPreparedTagLink(
            "image-tagging",
            "ext:ai.tagging",
            "Action",
            0.82,
            "tagger-v1",
            AiMediaKinds.Image));

        var notes = await service.PersistAsync(request, batch);

        await using var verifyScope = provider.CreateAsyncScope();
        var verifyDb = verifyScope.ServiceProvider.GetRequiredService<CoveContext>();
        var tag = await verifyDb.Tags.SingleAsync();
        var imageTag = await verifyDb.Set<ImageTag>().SingleAsync();
        var provenance = await verifyDb.TagApplications.SingleAsync();

        Assert.Equal(tag.Id, imageTag.TagId);
        Assert.Equal(11, imageTag.ImageId);
        Assert.Equal(AffinityHostType.Image, provenance.HostType);
        Assert.Equal(11, provenance.HostId);
        Assert.Equal(tag.Id, provenance.TagId);
        Assert.Equal("ext:ai.tagging", provenance.SourceKey);
        Assert.Equal("run-tagging", provenance.SourceRunId);
        Assert.Equal("tagger-v1", provenance.ModelKey);
        Assert.Equal(0.82f, provenance.Confidence);
        Assert.Contains(notes, note => note.Contains("Persisted 1 AI-generated tag evidence", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PersistTagging_AppliesConfiguredImageTagNameOverrides()
    {
        var storeFactory = new TestExtensionStoreFactory();
        await AiTaggingSettingsStore.SaveAsync(
            storeFactory.CreateStore(AiTaggingSettingsStore.ExtensionId),
            new AiTaggingSettings
            {
                TagNameOverrides =
                [
                    new AiTagNameOverride { SourceTagName = "1girl", TargetTagName = "Solo female" },
                ],
            });

        await using var provider = CreateProvider(services =>
        {
            services.AddSingleton<IExtensionStoreFactory>(storeFactory);
            services.AddScoped<ITagProvenanceService, TestTagProvenanceService>();
        });

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Images.Add(new Cove.Core.Entities.Image { Id = 14, Title = "Tagged still" });
            await db.SaveChangesAsync();
        }

        var service = new AiTaggingPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
        var request = CreateRequest(AiMediaKinds.Image, "image-tagging", "image", 14, "run-tagging-override");
        var batch = new AiPreparedArtifactBatch();

        batch.TagLinks.Add(new AiPreparedTagLink(
            "image-tagging",
            "ext:ai.tagging",
            "1girl",
            0.82,
            "tagger-v1",
            AiMediaKinds.Image));

        await service.PersistAsync(request, batch);

        await using var verifyScope = provider.CreateAsyncScope();
        var verifyDb = verifyScope.ServiceProvider.GetRequiredService<CoveContext>();
        var tag = await verifyDb.Tags.SingleAsync();
        var imageTag = await verifyDb.Set<ImageTag>().SingleAsync();
        var provenance = await verifyDb.TagApplications.SingleAsync();

        Assert.Equal("Solo female", tag.Name);
        Assert.Equal(tag.Id, imageTag.TagId);
        Assert.Equal(tag.Id, provenance.TagId);
    }

    [Fact]
    public async Task PersistTagging_WritesVideoTagLinksIntoProvenanceWithoutDirectTags()
    {
        await using var provider = CreateProvider(services =>
        {
            services.AddScoped<ITagProvenanceService, TestTagProvenanceService>();
        });

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Videos.Add(new Video { Id = 13, Title = "Tagged video link" });
            await db.SaveChangesAsync();
        }

        var service = new AiTaggingPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
        var request = CreateRequest(AiMediaKinds.Video, "video-tag-link", "video", 13, "run-tagging-link", durationSeconds: 20d);
        var batch = new AiPreparedArtifactBatch();

        batch.TagLinks.Add(new AiPreparedTagLink(
            "video-tag-link",
            "ext:ai.tagging",
            "Outdoors",
            0.72,
            "tagger-v1",
            AiMediaKinds.Video));

        var notes = await service.PersistAsync(request, batch);

        await using var verifyScope = provider.CreateAsyncScope();
        var verifyDb = verifyScope.ServiceProvider.GetRequiredService<CoveContext>();
        var tag = await verifyDb.Tags.SingleAsync();
        var provenance = await verifyDb.TagApplications.SingleAsync();

        Assert.Empty(await verifyDb.Set<VideoTag>().ToListAsync());
        Assert.Equal(AffinityHostType.Video, provenance.HostType);
        Assert.Equal(13, provenance.HostId);
        Assert.Equal(tag.Id, provenance.TagId);
        Assert.Equal("ext:ai.tagging", provenance.SourceKey);
        Assert.Equal("run-tagging-link", provenance.SourceRunId);
        Assert.Equal("tagger-v1", provenance.ModelKey);
        Assert.Equal(0.72f, provenance.Confidence);
        Assert.Null(provenance.TotalDurationSec);
        Assert.Equal(20.0, provenance.HostDurationSec.GetValueOrDefault(), 3);
        Assert.Contains(notes, note => note.Contains("Persisted 1 AI-generated tag evidence", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PersistTagging_WritesVideoTagDurationsIntoProvenance()
    {
        await using var provider = CreateProvider(services =>
        {
            services.AddScoped<ITagProvenanceService, TestTagProvenanceService>();
        });

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Videos.Add(new Video { Id = 12, Title = "Tagged clip" });
            await db.SaveChangesAsync();
        }

        var service = new AiTaggingPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
        var request = CreateRequest(AiMediaKinds.Video, "video-tagging", "video", 12, "run-tagging-video", durationSeconds: 20d);
        var batch = new AiPreparedArtifactBatch();

        batch.Segments.Add(new AiPreparedSegment(
            "video-tagging",
            "ext:ai.tagging",
            Kind: "tag",
            StartSeconds: 1.0,
            EndSeconds: 2.5,
            TagName: "Action",
            Title: "Action",
            Confidence: 0.82,
            Metadata: new Dictionary<string, string>
            {
                ["modelKey"] = "tagger-v1",
            }));
        batch.Segments.Add(new AiPreparedSegment(
            "video-tagging",
            "ext:ai.tagging",
            Kind: "tag",
            StartSeconds: 5.0,
            EndSeconds: 7.5,
            TagName: "Action",
            Title: "Action",
            Confidence: 0.91,
            Metadata: new Dictionary<string, string>
            {
                ["modelKey"] = "tagger-v1",
            }));

        var notes = await service.PersistAsync(request, batch);

        await using var verifyScope = provider.CreateAsyncScope();
        var verifyDb = verifyScope.ServiceProvider.GetRequiredService<CoveContext>();
        var tag = await verifyDb.Tags.SingleAsync();
        var provenance = await verifyDb.TagApplications.SingleAsync();

        Assert.Empty(await verifyDb.Set<VideoTag>().ToListAsync());
        Assert.Equal(AffinityHostType.Video, provenance.HostType);
        Assert.Equal(12, provenance.HostId);
        Assert.Equal(tag.Id, provenance.TagId);
        Assert.Equal("ext:ai.tagging", provenance.SourceKey);
        Assert.Equal("run-tagging-video", provenance.SourceRunId);
        Assert.Equal("tagger-v1", provenance.ModelKey);
        Assert.Equal(0.91f, provenance.Confidence);
        Assert.NotNull(provenance.TotalDurationSec);
        Assert.NotNull(provenance.HostDurationSec);
        Assert.Equal(4.0, provenance.TotalDurationSec.Value, 3);
        Assert.Equal(20.0, provenance.HostDurationSec.Value, 3);
        Assert.Contains(notes, note => note.Contains("Persisted 1 AI-generated tag evidence", StringComparison.Ordinal));
        Assert.Contains(notes, note => note.Contains("Persisted 2 AI-generated tagging segment", StringComparison.Ordinal));
    }

    [Fact]
    public async Task PersistTagging_AppliesConfiguredVideoSegmentTagNameOverrides()
    {
        var storeFactory = new TestExtensionStoreFactory();
        await AiTaggingSettingsStore.SaveAsync(
            storeFactory.CreateStore(AiTaggingSettingsStore.ExtensionId),
            new AiTaggingSettings
            {
                TagNameOverrides =
                [
                    new AiTagNameOverride { SourceTagName = "1girl", TargetTagName = "Solo female" },
                ],
            });

        await using var provider = CreateProvider(services =>
        {
            services.AddSingleton<IExtensionStoreFactory>(storeFactory);
            services.AddScoped<ITagProvenanceService, TestTagProvenanceService>();
        });

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Videos.Add(new Video { Id = 15, Title = "Tagged clip" });
            await db.SaveChangesAsync();
        }

        var service = new AiTaggingPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
        var request = CreateRequest(AiMediaKinds.Video, "video-tagging", "video", 15, "run-tagging-video-override", durationSeconds: 20d);
        var batch = new AiPreparedArtifactBatch();

        batch.Segments.Add(new AiPreparedSegment(
            "video-tagging",
            "ext:ai.tagging",
            Kind: "tag",
            StartSeconds: 1.0,
            EndSeconds: 2.5,
            TagName: "1girl",
            Title: "1girl",
            Confidence: 0.82,
            Metadata: new Dictionary<string, string>
            {
                ["modelKey"] = "tagger-v1",
            }));

        await service.PersistAsync(request, batch);

        await using var verifyScope = provider.CreateAsyncScope();
        var verifyDb = verifyScope.ServiceProvider.GetRequiredService<CoveContext>();
        var tag = await verifyDb.Tags.SingleAsync();
        var segment = await verifyDb.Segments.SingleAsync();
        var provenance = await verifyDb.TagApplications.SingleAsync();

        Assert.Equal("Solo female", tag.Name);
        Assert.Equal(tag.Id, segment.TagId);
        Assert.Equal("Solo female", segment.Title);
        Assert.Equal(tag.Id, provenance.TagId);
    }

    [Fact]
    public async Task PersistFaces_GeneratesFaceCoverForImageHost()
    {
        var tempRoot = Path.Combine(Path.GetTempPath(), $"ai-faces-cover-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);

        try
        {
            var imagePath = Path.Combine(tempRoot, "cover-source.jpg");
            using (var image = new Image<Rgba32>(640, 480, new Rgba32(220, 220, 220)))
            {
                await image.SaveAsJpegAsync(imagePath);
            }

            await using var provider = CreateProvider(services =>
            {
                services.AddSingleton<IBlobService, TestBlobService>();
                services.AddSingleton(new CoveConfiguration());
            });

            var service = new AiFacesPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
            var request = CreateRequest(AiMediaKinds.Image, imagePath, "image", 17, "run-face-cover");
            var batch = new AiPreparedArtifactBatch();

            batch.Faces.Add(new AiPreparedFaceIdentity(
                "face-cover-1",
                "ext:ai.faces",
                Label: "Cover Face",
                IsProvisional: false,
                QualityScore: 0.97,
                CoverAssetId: imagePath,
                CoverBoundingBox: new AiBoundingBox(0.25, 0.15, 0.75, 0.85)));

            var notes = await service.PersistAsync(request, batch);

            await using var scope = provider.CreateAsyncScope();
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var blobService = (TestBlobService)scope.ServiceProvider.GetRequiredService<IBlobService>();
            var face = await db.Faces.SingleAsync();

            Assert.False(string.IsNullOrWhiteSpace(face.CoverBlobId));
            Assert.True(blobService.StoredBlobs.ContainsKey(face.CoverBlobId!));
            Assert.Contains(notes, note => note.Contains("Generated 1 face cover image", StringComparison.Ordinal));
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    public async Task PersistFaces_GeneratesImageCoverFromContextSubjectWhenAssetIdIsMapped()
    {
        var tempRoot = Path.Combine(Path.GetTempPath(), $"ai-faces-cover-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);

        try
        {
            var imagePath = Path.Combine(tempRoot, "cover-source.jpg");
            using (var image = new Image<Rgba32>(640, 480, new Rgba32(220, 220, 220)))
            {
                await image.SaveAsJpegAsync(imagePath);
            }

            await using var provider = CreateProvider(services =>
            {
                services.AddSingleton<IBlobService, TestBlobService>();
                services.AddSingleton(new CoveConfiguration());
            });

            var service = new AiFacesPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
            var request = CreateRequest(AiMediaKinds.Image, "/mapped/cover-source.jpg", "image", 17, "run-face-cover", subject: imagePath);
            var batch = new AiPreparedArtifactBatch();

            batch.Faces.Add(new AiPreparedFaceIdentity(
                "face-cover-1",
                "ext:ai.faces",
                Label: "Cover Face",
                IsProvisional: false,
                QualityScore: 0.97,
                CoverAssetId: "/mapped/cover-source.jpg",
                CoverBoundingBox: new AiBoundingBox(0.25, 0.15, 0.75, 0.85)));

            await service.PersistAsync(request, batch);

            await using var scope = provider.CreateAsyncScope();
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var blobService = (TestBlobService)scope.ServiceProvider.GetRequiredService<IBlobService>();
            var face = await db.Faces.SingleAsync();

            Assert.False(string.IsNullOrWhiteSpace(face.CoverBlobId));
            Assert.True(blobService.StoredBlobs.ContainsKey(face.CoverBlobId!));
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    public async Task PersistFaces_ReplacesFaceCoverWhenNewSampleIsBetter()
    {
        var tempRoot = Path.Combine(Path.GetTempPath(), $"ai-faces-cover-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);

        try
        {
            var imagePath = Path.Combine(tempRoot, "cover-source.jpg");
            using (var image = new Image<Rgba32>(640, 480, new Rgba32(220, 220, 220)))
            {
                await image.SaveAsJpegAsync(imagePath);
            }

            await using var provider = CreateProvider(services =>
            {
                services.AddSingleton<IBlobService, TestBlobService>();
                services.AddSingleton(new CoveConfiguration());
            });

            var service = new AiFacesPersistenceService(provider.GetRequiredService<IServiceScopeFactory>());
            var firstRequest = CreateRequest(AiMediaKinds.Image, imagePath, "image", 18, "run-face-cover-first");
            var firstBatch = new AiPreparedArtifactBatch();
            firstBatch.Faces.Add(new AiPreparedFaceIdentity(
                "face-cover-replace",
                "ext:ai.faces",
                Label: "Cover Face",
                IsProvisional: false,
                QualityScore: 0.80,
                CoverAssetId: imagePath,
                CoverBoundingBox: new AiBoundingBox(0.25, 0.15, 0.75, 0.85),
                Metadata: new Dictionary<string, string>
                {
                    ["coverQualityScore"] = "0.40",
                }));

            await service.PersistAsync(firstRequest, firstBatch);

            string firstBlobId;
            await using (var scope = provider.CreateAsyncScope())
            {
                var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
                firstBlobId = (await db.Faces.SingleAsync()).CoverBlobId!;
            }

            var secondRequest = CreateRequest(AiMediaKinds.Image, imagePath, "image", 18, "run-face-cover-second");
            var secondBatch = new AiPreparedArtifactBatch();
            secondBatch.Faces.Add(new AiPreparedFaceIdentity(
                "face-cover-replace",
                "ext:ai.faces",
                Label: "Cover Face",
                IsProvisional: false,
                QualityScore: 0.95,
                CoverAssetId: imagePath,
                CoverBoundingBox: new AiBoundingBox(0.20, 0.10, 0.80, 0.90),
                Metadata: new Dictionary<string, string>
                {
                    ["coverQualityScore"] = "0.95",
                }));

            await service.PersistAsync(secondRequest, secondBatch);

            await using var verifyScope = provider.CreateAsyncScope();
            var verifyDb = verifyScope.ServiceProvider.GetRequiredService<CoveContext>();
            var blobService = (TestBlobService)verifyScope.ServiceProvider.GetRequiredService<IBlobService>();
            var face = await verifyDb.Faces.SingleAsync();

            Assert.NotEqual(firstBlobId, face.CoverBlobId);
            Assert.False(blobService.StoredBlobs.ContainsKey(firstBlobId));
            Assert.True(blobService.StoredBlobs.ContainsKey(face.CoverBlobId!));
            var storedQuality = await verifyDb.CustomFieldValues
                .Include(value => value.Definition)
                .SingleAsync(value => value.EntityType == CustomFieldEntityTypes.Face
                    && value.EntityId == face.Id
                    && value.Definition != null
                    && value.Definition.Key == "ai.faces.coverBlobQualityScore");
            Assert.Equal(0.95m, storedQuality.NumberValue!.Value, precision: 3);
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    // The DbFaceIdentityStore delete path uses ExecuteDelete, which the EF in-memory provider cannot
    // translate, so the participant's branching logic is verified here against a recording store. The
    // store's one-line key/clear deletes are covered by the relational provider itself.
    [Fact]
    public async Task FaceDeleteParticipant_DeletesIdentityForDeletedFaceAndClearsOnEntireSourcePurge()
    {
        var store = new RecordingFaceIdentityStore();
        var participant = new AiFacesDeleteParticipant(store);

        // A deleted Cove face removes its identity by key.
        await participant.OnDeletingAsync(new Face { Id = 1, PrimarySourceKey = "face-1" });
        Assert.Equal("face-1", Assert.Single(store.DeletedFaceKeys));

        // A face with no source key has no identity to remove.
        await participant.OnDeletingAsync(new Face { Id = 2, PrimarySourceKey = null });
        Assert.Single(store.DeletedFaceKeys);

        // An entire-source purge for this extension clears the provisional identity graph.
        await participant.OnFacesPurgedAsync(new FacePurgeScope("ext:ai.faces", null, null, true, []));
        Assert.Equal(1, store.ClearAllCount);

        // A narrowed purge leaves the identity graph intact.
        await participant.OnFacesPurgedAsync(new FacePurgeScope("ext:ai.faces", "video", 5, false, [5]));
        Assert.Equal(1, store.ClearAllCount);
    }

    private sealed class RecordingFaceIdentityStore : IFaceIdentityStore
    {
        public List<string> DeletedFaceKeys { get; } = [];

        public int ClearAllCount { get; private set; }

        public Task<FaceIdentityTransaction> BeginIncrementalAsync(IReadOnlyList<IReadOnlyList<float>> queryVectors, IReadOnlyCollection<string> referenceExternalIds, int candidateK, CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task<FaceIdentityTransaction> BeginFullAsync(CancellationToken ct = default)
            => throw new NotSupportedException();

        public Task DeleteByFaceKeyAsync(string faceKey, CancellationToken ct = default)
        {
            DeletedFaceKeys.Add(faceKey);
            return Task.CompletedTask;
        }

        public Task ClearAllAsync(CancellationToken ct = default)
        {
            ClearAllCount++;
            return Task.CompletedTask;
        }
    }

    private static ServiceProvider CreateProvider(Action<ServiceCollection>? configure = null)
    {
        var services = new ServiceCollection();
        var databaseName = $"ai-persistence-{Guid.NewGuid():N}";
        var databaseRoot = new InMemoryDatabaseRoot();
        services.AddDbContext<CoveContext>(options => options.UseInMemoryDatabase(databaseName, databaseRoot));
        services.AddScoped<IFaceRepository, FaceRepository>();
        services.AddScoped<IVideoRepository, VideoRepository>();
        services.AddScoped<IImageRepository, ImageRepository>();
        services.AddScoped<IDetectionRepository, DetectionRepository>();
        services.AddScoped<ISegmentRepository, SegmentRepository>();
        services.AddScoped<IEmbeddingRepository, EmbeddingRepository>();
        services.AddScoped<ICustomFieldRepository, CustomFieldRepository>();
        services.AddScoped<ITagRepository, TagRepository>();
        services.AddScoped<ITagApplicationRepository, TagApplicationRepository>();
        configure?.Invoke(services);
        return services.BuildServiceProvider();
    }

    private static AiDispatchRequest CreateRequest(string mediaKind, string assetId, string hostEntityType, int hostEntityId, string runId, double? durationSeconds = null, string? subject = null)
    {
        return new AiDispatchRequest(
            new AiRunContext(runId, mediaKind, assetId, subject ?? assetId, hostEntityType, hostEntityId, durationSeconds),
            [],
            new AiAnalyzeResult
            {
                MediaKind = mediaKind,
                AssetId = assetId,
                DurationSeconds = durationSeconds,
            });
    }

    private sealed class TestBlobService : IBlobService
    {
        public Dictionary<string, byte[]> StoredBlobs { get; } = new(StringComparer.Ordinal);

        public async Task<string> StoreBlobAsync(Stream data, string contentType, CancellationToken ct = default)
        {
            var blobId = Guid.NewGuid().ToString("N");
            await using var buffer = new MemoryStream();
            await data.CopyToAsync(buffer, ct);
            StoredBlobs[blobId] = buffer.ToArray();
            return blobId;
        }

        public Task<(Stream Stream, string ContentType)?> GetBlobAsync(string blobId, CancellationToken ct = default)
        {
            if (!StoredBlobs.TryGetValue(blobId, out var bytes))
            {
                return Task.FromResult<(Stream, string)?>(null);
            }

            return Task.FromResult<(Stream, string)?>((new MemoryStream(bytes, writable: false), "image/jpeg"));
        }

        public Task DeleteBlobAsync(string blobId, CancellationToken ct = default)
        {
            StoredBlobs.Remove(blobId);
            return Task.CompletedTask;
        }
    }

    private sealed class TestExtensionStore : IExtensionStore
    {
        private readonly Dictionary<string, string> _values = new(StringComparer.Ordinal);

        public Task<string?> GetAsync(string key, CancellationToken ct = default)
            => Task.FromResult(_values.TryGetValue(key, out var value) ? value : null);

        public Task SetAsync(string key, string value, CancellationToken ct = default)
        {
            _values[key] = value;
            return Task.CompletedTask;
        }

        public Task DeleteAsync(string key, CancellationToken ct = default)
        {
            _values.Remove(key);
            return Task.CompletedTask;
        }

        public Task<Dictionary<string, string>> GetAllAsync(CancellationToken ct = default)
            => Task.FromResult(new Dictionary<string, string>(_values, StringComparer.Ordinal));
    }

    private sealed class TestExtensionStoreFactory : IExtensionStoreFactory
    {
        private readonly Dictionary<string, TestExtensionStore> _stores = new(StringComparer.OrdinalIgnoreCase);

        public IExtensionStore CreateStore(string extensionId)
        {
            if (!_stores.TryGetValue(extensionId, out var store))
            {
                store = new TestExtensionStore();
                _stores[extensionId] = store;
            }

            return store;
        }
    }

    private sealed class TestTagProvenanceService(CoveContext db) : ITagProvenanceService
    {
        public Task RecordAsync(AffinityHostType hostType, int hostId, int tagId, string sourceKey, string? sourceRunId = null, string? modelKey = null, float? confidence = null, string? contextType = null, int? contextId = null, double? totalDurationSec = null, double? hostDurationSec = null, CancellationToken cancellationToken = default)
        {
            db.TagApplications.Add(new TagApplication
            {
                HostType = hostType,
                HostId = hostId,
                TagId = tagId,
                SourceKey = sourceKey,
                SourceRunId = sourceRunId ?? string.Empty,
                ModelKey = modelKey ?? string.Empty,
                Confidence = confidence,
                ContextType = contextType,
                ContextId = contextId,
                TotalDurationSec = totalDurationSec,
                HostDurationSec = hostDurationSec,
            });
            return Task.CompletedTask;
        }

        public Task RecordAsync(AffinityHostType hostType, int hostId, Tag tag, string sourceKey, string? sourceRunId = null, string? modelKey = null, float? confidence = null, string? contextType = null, int? contextId = null, double? totalDurationSec = null, double? hostDurationSec = null, CancellationToken cancellationToken = default)
        {
            db.TagApplications.Add(new TagApplication
            {
                HostType = hostType,
                HostId = hostId,
                Tag = tag,
                SourceKey = sourceKey,
                SourceRunId = sourceRunId ?? string.Empty,
                ModelKey = modelKey ?? string.Empty,
                Confidence = confidence,
                ContextType = contextType,
                ContextId = contextId,
                TotalDurationSec = totalDurationSec,
                HostDurationSec = hostDurationSec,
            });
            return Task.CompletedTask;
        }

        public Task SyncTagSetAsync(AffinityHostType hostType, int hostId, IReadOnlyCollection<int> previousTagIds, IReadOnlyCollection<int> currentTagIds, string sourceKey = "user", CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public Task RemoveForHostAsync(AffinityHostType hostType, int hostId, CancellationToken cancellationToken = default)
            => Task.CompletedTask;

        public Task<IReadOnlyDictionary<int, List<TagProvenanceDto>>> GetLookupAsync(AffinityHostType hostType, int hostId, IReadOnlyCollection<int> tagIds, CancellationToken cancellationToken = default)
            => Task.FromResult<IReadOnlyDictionary<int, List<TagProvenanceDto>>>(new Dictionary<int, List<TagProvenanceDto>>());
    }
}
