using AI.Faces;

using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Cove.Data;
using Cove.Data.Repositories;

using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Microsoft.Extensions.DependencyInjection;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiFaceCoverBackfillServiceTests
{
    [Fact]
    public async Task BackfillAsync_BuildsCoversFromStoredStillSamplesAndSkipsTheRest()
    {
        var tempRoot = Path.Combine(Path.GetTempPath(), $"ai-faces-cover-backfill-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);

        try
        {
            var archivePath = Path.Combine(tempRoot, "gallery.zip");
            await using (var file = File.Create(archivePath))
            using (var archive = new System.IO.Compression.ZipArchive(file, System.IO.Compression.ZipArchiveMode.Create, leaveOpen: false))
            {
                await using var entryStream = archive.CreateEntry("set/017.jpg").Open();
                using var image = new Image<Rgba32>(640, 480, new Rgba32(220, 30, 30));
                await image.SaveAsJpegAsync(entryStream);
            }

            var imagePath = Path.Combine(tempRoot, "still.jpg");
            using (var image = new Image<Rgba32>(640, 480, new Rgba32(30, 200, 30)))
            {
                await image.SaveAsJpegAsync(imagePath);
            }

            await using var provider = CreateProvider();
            int archivedId, imageId, videoId, missingId, missingVideoId, noSampleId, coveredId, mergedId, foreignId;
            await using (var scope = provider.CreateAsyncScope())
            {
                var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
                var archived = new Face { Label = "archived", PrimarySourceKey = "face-0001" };
                var still = new Face { Label = "still", PrimarySourceKey = "face-0002" };
                var video = new Face { Label = "video", PrimarySourceKey = "face-0003" };
                var missing = new Face { Label = "missing", PrimarySourceKey = "face-0004" };
                var missingVideo = new Face { Label = "missing video", PrimarySourceKey = "face-0008" };
                var noSample = new Face { Label = "no sample", PrimarySourceKey = "face-0005" };
                var covered = new Face { Label = "covered", PrimarySourceKey = "face-0006", CoverBlobId = "existing-blob" };
                var foreign = new Face { Label = "other provider", PrimarySourceKey = "other-0001" };
                db.Faces.AddRange(archived, still, video, missing, missingVideo, noSample, covered, foreign);
                await db.SaveChangesAsync();

                var merged = new Face { Label = "merged", PrimarySourceKey = "face-0007", MergedIntoFaceId = archived.Id };
                db.Faces.Add(merged);
                db.Set<ExtAiFacesIdentityEntity>().AddRange(
                    Identity(1, "face-0001", $"{archivePath}/set/017.jpg"),
                    Identity(2, "face-0002", imagePath),
                    Identity(3, "face-0003", Path.Combine(tempRoot, "clip.mp4")),
                    Identity(4, "face-0004", Path.Combine(tempRoot, "gone.jpg")),
                    Identity(5, "face-0005", null),
                    Identity(6, "face-0006", imagePath),
                    Identity(7, "face-0007", imagePath),
                    Identity(8, "face-0008", Path.Combine(tempRoot, "gone.mp4"), coverSeconds: 12.5));
                await db.SaveChangesAsync();

                (archivedId, imageId, videoId, missingId, missingVideoId, noSampleId, coveredId, mergedId, foreignId)
                    = (archived.Id, still.Id, video.Id, missing.Id, missingVideo.Id, noSample.Id, covered.Id, merged.Id, foreign.Id);
            }

            var service = provider.GetRequiredService<AiFaceCoverBackfillService>();
            var progress = new RecordingProgress();
            var result = await service.BackfillAsync(progress);

            Assert.Equal(new AiFaceCoverBackfillResult(
                FacesWithoutCover: 6,
                CreatedCoverCount: 2,
                NotCreatedCount: 0,
                SkippedVideoWithoutTimestampCount: 1,
                SkippedMissingSourceCount: 2,
                SkippedNoSampleCount: 1), result);
            Assert.Equal(result.Summary, progress.Messages[^1]);

            await using (var scope = provider.CreateAsyncScope())
            {
                var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
                var blobs = (TestBlobService)scope.ServiceProvider.GetRequiredService<IBlobService>();
                var faces = await db.Faces.AsNoTracking().ToDictionaryAsync(face => face.Id);

                AssertCoverColor(blobs, faces[archivedId], expectRed: true);
                AssertCoverColor(blobs, faces[imageId], expectRed: false);
                Assert.Null(faces[videoId].CoverBlobId);
                Assert.Null(faces[missingId].CoverBlobId);
                Assert.Null(faces[missingVideoId].CoverBlobId);
                Assert.Null(faces[noSampleId].CoverBlobId);
                Assert.Equal("existing-blob", faces[coveredId].CoverBlobId);
                Assert.Null(faces[mergedId].CoverBlobId);
                Assert.Null(faces[foreignId].CoverBlobId);

                var storedQuality = await db.CustomFieldValues
                    .Include(value => value.Definition)
                    .SingleAsync(value => value.EntityType == CustomFieldEntityTypes.Face
                        && value.EntityId == archivedId
                        && value.Definition != null
                        && value.Definition.Key == "ai.faces.coverBlobQualityScore");
                Assert.Equal(1.25m, storedQuality.NumberValue!.Value, precision: 3);
            }

            var rerun = await service.BackfillAsync(progress: null);
            Assert.Equal(4, rerun.FacesWithoutCover);
            Assert.Equal(0, rerun.CreatedCoverCount);
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    [Fact]
    public void Extension_ListsCoverBackfillOnTheExtensionTasksPage()
    {
        var job = Assert.Single(new AiFacesExtension().Jobs, job => job.Id == AiFacesExtension.BackfillCoversJobId);

        Assert.True(job.ShowInTaskList);
        Assert.False(job.SupportsParameters);
    }

    [FfmpegFact]
    public async Task BackfillAsync_BuildsCoversFromVideoSamplesWithAStoredTimestamp()
    {
        var tempRoot = Path.Combine(Path.GetTempPath(), $"ai-faces-cover-backfill-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempRoot);

        try
        {
            var videoPath = Path.Combine(tempRoot, "clip.mp4");
            TestVideos.CreateTwoColorVideo(videoPath, new Rgba32(220, 30, 30), new Rgba32(30, 200, 30), switchAtSeconds: 2, totalSeconds: 4);

            await using var provider = CreateProvider();
            int lateId, earlyId;
            await using (var scope = provider.CreateAsyncScope())
            {
                var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
                var late = new Face { Label = "late", PrimarySourceKey = "face-0001" };
                var early = new Face { Label = "early", PrimarySourceKey = "face-0002" };
                db.Faces.AddRange(late, early);
                db.Set<ExtAiFacesIdentityEntity>().AddRange(
                    Identity(1, "face-0001", videoPath, coverSeconds: 3.0),
                    Identity(2, "face-0002", videoPath, coverSeconds: 0.5));
                await db.SaveChangesAsync();
                (lateId, earlyId) = (late.Id, early.Id);
            }

            var result = await provider.GetRequiredService<AiFaceCoverBackfillService>().BackfillAsync(progress: null);

            Assert.Equal(2, result.CreatedCoverCount);
            Assert.Equal(0, result.SkippedVideoWithoutTimestampCount);
            await using (var scope = provider.CreateAsyncScope())
            {
                var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
                var blobs = (TestBlobService)scope.ServiceProvider.GetRequiredService<IBlobService>();
                var faces = await db.Faces.AsNoTracking().ToDictionaryAsync(face => face.Id);
                AssertCoverColor(blobs, faces[lateId], expectRed: false);
                AssertCoverColor(blobs, faces[earlyId], expectRed: true);
            }
        }
        finally
        {
            Directory.Delete(tempRoot, recursive: true);
        }
    }

    private static ExtAiFacesIdentityEntity Identity(int ordinal, string faceKey, string? coverAssetId, double? coverSeconds = null)
        => new()
        {
            FaceKey = faceKey,
            Ordinal = ordinal,
            CoverAssetId = coverAssetId,
            CoverObservedAtSeconds = coverSeconds,
            CoverX1 = coverAssetId is null ? null : 0.25,
            CoverY1 = coverAssetId is null ? null : 0.15,
            CoverX2 = coverAssetId is null ? null : 0.75,
            CoverY2 = coverAssetId is null ? null : 0.85,
            CoverQualityScore = 1.25,
            CreatedAt = DateTime.UtcNow,
            UpdatedAt = DateTime.UtcNow,
        };

    private static void AssertCoverColor(TestBlobService blobs, Face face, bool expectRed)
    {
        Assert.False(string.IsNullOrWhiteSpace(face.CoverBlobId));
        using var cover = SixLabors.ImageSharp.Image.Load<Rgba32>(blobs.StoredBlobs[face.CoverBlobId!]);
        var pixel = cover[cover.Width / 2, cover.Height / 2];
        Assert.True(expectRed ? pixel.R > 180 && pixel.G < 80 : pixel.G > 160 && pixel.R < 80, $"Unexpected cover color RGB({pixel.R}, {pixel.G}, {pixel.B}) for face {face.Label}.");
    }

    private static ServiceProvider CreateProvider()
    {
        var services = new ServiceCollection();
        var databaseName = $"ai-face-cover-backfill-{Guid.NewGuid():N}";
        var databaseRoot = new InMemoryDatabaseRoot();
        services.AddDbContext<CoveContext>(options => options.UseInMemoryDatabase(databaseName, databaseRoot));
        services.AddScoped<DbContext>(static sp => sp.GetRequiredService<CoveContext>());
        services.AddScoped<IFaceRepository, FaceRepository>();
        services.AddScoped<ICustomFieldRepository, CustomFieldRepository>();
        services.AddSingleton<IBlobService, TestBlobService>();
        services.AddSingleton(new CoveConfiguration());
        services.AddSingleton<AiFaceCoverBackfillService>();
        return services.BuildServiceProvider();
    }

    private sealed class RecordingProgress : Cove.Plugins.IJobProgress
    {
        public List<string?> Messages { get; } = [];

        public void Report(double percent, string? message = null)
        {
            lock (Messages)
            {
                Messages.Add(message);
            }
        }
    }

    private sealed class TestBlobService : IBlobService
    {
        public System.Collections.Concurrent.ConcurrentDictionary<string, byte[]> StoredBlobs { get; } = new(StringComparer.Ordinal);

        public async Task<string> StoreBlobAsync(Stream data, string contentType, CancellationToken ct = default)
        {
            var blobId = Guid.NewGuid().ToString("N");
            await using var buffer = new MemoryStream();
            await data.CopyToAsync(buffer, ct);
            StoredBlobs[blobId] = buffer.ToArray();
            return blobId;
        }

        public Task<(Stream Stream, string ContentType)?> GetBlobAsync(string blobId, CancellationToken ct = default)
            => Task.FromResult<(Stream Stream, string ContentType)?>(
                StoredBlobs.TryGetValue(blobId, out var bytes) ? (new MemoryStream(bytes), "image/jpeg") : null);

        public Task DeleteBlobAsync(string blobId, CancellationToken ct = default)
        {
            StoredBlobs.TryRemove(blobId, out _);
            return Task.CompletedTask;
        }
    }
}
