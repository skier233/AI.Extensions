using AI.Faces;

using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Cove.Data;
using Cove.Data.Services;

using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Microsoft.Extensions.DependencyInjection;

using Pgvector;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiFaceSuggesterTests
{
    [Fact]
    public async Task SuggestAsync_RanksPerformersAndSuppressesRejectedMatches()
    {
        await using var provider = CreateProvider();

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Performers.AddRange(
                new Performer { Id = 11, Name = "Performer Alpha", ImageBlobId = "alpha-cover" },
                new Performer { Id = 22, Name = "Performer Beta" });
            db.Faces.AddRange(
                new Face { Id = 1, Label = "Target Face", PrimarySourceKey = "face-0001" },
                new Face { Id = 2, Label = "Alpha Match A", PerformerId = 11, PrimarySourceKey = "face-0002" },
                new Face { Id = 3, Label = "Alpha Match B", PerformerId = 11, PrimarySourceKey = "face-0003" },
                new Face { Id = 4, Label = "Beta Match", PerformerId = 22, PrimarySourceKey = "face-0004" });
            db.Embeddings.AddRange(
                CreateFaceEmbedding(1, [1f, 0f]),
                CreateFaceEmbedding(2, [0.98f, 0.02f]),
                CreateFaceEmbedding(3, [0.96f, 0.04f]),
                CreateFaceEmbedding(4, [0.71f, 0.69f]));
            db.Detections.AddRange(
                CreateFaceDetection(2, DetectionHostType.Image, 101, 0.96f),
                CreateFaceDetection(3, DetectionHostType.Video, 202, 0.93f, observedAtSec: 14.0),
                CreateFaceDetection(4, DetectionHostType.Image, 303, 0.88f));

            await db.SaveChangesAsync();
        }

        await using (var scope = provider.CreateAsyncScope())
        {
            var suggester = scope.ServiceProvider.GetRequiredService<IFaceSuggester>();
            var suggestions = await suggester.SuggestForAsync(1, 5);

            Assert.Equal(2, suggestions.Count);
            Assert.Equal(11, suggestions[0].PerformerId);
            Assert.Equal("Performer Alpha", suggestions[0].PerformerName);
            Assert.True(suggestions[0].Confidence > suggestions[1].Confidence);
            Assert.Equal([2, 3], suggestions[0].Evidence.Select(item => item.FaceId).OrderBy(id => id).ToArray());
            Assert.Contains("2 linked face clusters", suggestions[0].Why, StringComparison.Ordinal);
            Assert.NotNull(suggestions[0].CoverImageUrl);
        }
    }

    [Fact]
    public async Task SuggestAsync_ReturnsReferenceIdentityWithSyntheticSuggestionIdWhenLocalMatchesAreMissing()
    {
        var referenceRoot = Path.Combine(Path.GetTempPath(), $"ai-face-suggester-reference-{Guid.NewGuid():N}");
        Directory.CreateDirectory(referenceRoot);

        await using var provider = CreateProvider(referenceRoot);

        try
        {
            await using (var scope = provider.CreateAsyncScope())
            {
                var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
                db.Faces.Add(new Face { Id = 10, Label = "Reference Target", PrimarySourceKey = "face-0010" });
                db.Embeddings.Add(CreateFaceEmbedding(10, [1f, 0f]));
                await db.SaveChangesAsync();

                var packStore = scope.ServiceProvider.GetRequiredService<AiFaceReferencePackStore>();
                var archivePath = CreateReferenceArchive(
                    referenceRoot,
                    new
                    {
                        version = 1,
                        embedder = "nsfw_ai_server:image_pipeline_dynamic_v4",
                        embedding_dim = 2,
                        pack_id = "stashdb-pack",
                        source_endpoint = "https://stashdb.org/graphql",
                        performer_count = 1,
                        created_at = "2026-03-24T02:43:03Z",
                    },
                    [
                        new
                        {
                            stashdb_id = "stashdb-performer-1",
                            name = "Reference Performer",
                            aliases = new[] { "Ref Alias" },
                            disambiguation = (string?)null,
                            sample_count = 3,
                            quality_score = 22.0,
                            image_url = "https://stashdb.org/images/reference.jpg",
                        },
                    ],
                    [1f, 0f],
                    1,
                    2);

                await using var stream = File.OpenRead(archivePath);
                var stagedPath = await packStore.StageUploadAsync(stream, Path.GetFileName(archivePath));
                await packStore.ImportStagedAsync(stagedPath, Path.GetFileName(archivePath), null, CancellationToken.None);
            }

            await using (var scope = provider.CreateAsyncScope())
            {
                var suggester = scope.ServiceProvider.GetRequiredService<IFaceSuggester>();
                var suggestions = await suggester.SuggestForAsync(10, 5);

                var suggestion = Assert.Single(suggestions);
                Assert.Equal(AiFaceReferenceSuggestionIds.FromOrdinal(0), suggestion.PerformerId);
                Assert.Equal("Reference Performer", suggestion.PerformerName);
                Assert.Equal("https://stashdb.org/images/reference.jpg", suggestion.CoverImageUrl);
                Assert.Empty(suggestion.Evidence);
                Assert.Contains("stashdb-pack match", suggestion.Why, StringComparison.OrdinalIgnoreCase);
            }
        }
        finally
        {
            if (Directory.Exists(referenceRoot))
                Directory.Delete(referenceRoot, recursive: true);
        }
    }

    [Fact]
    public async Task SuggestAsync_PrefersExclusiveVideoPerformerWhenFaceHostsAllPointToSamePerformer()
    {
        await using var provider = CreateProvider();

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Performers.AddRange(
                new Performer { Id = 11, Name = "Mia Melon" },
                new Performer { Id = 22, Name = "Other Performer" });
            db.Videos.Add(new Video { Id = 3077, Title = "Video 3077" });
            db.Set<VideoPerformer>().Add(new VideoPerformer { VideoId = 3077, PerformerId = 11 });
            db.Faces.AddRange(
                new Face { Id = 1, Label = "Target Face", PrimarySourceKey = "face-0001" },
                new Face { Id = 2, Label = "Weak Other Match", PerformerId = 22, PrimarySourceKey = "face-0002" });
            db.Embeddings.AddRange(
                CreateFaceEmbedding(1, [1f, 0f]),
                CreateFaceEmbedding(2, [0.15f, 0.85f]));
            db.Detections.AddRange(
                CreateFaceDetection(1, DetectionHostType.Video, 3077, 0.97f, observedAtSec: 5.0),
                CreateFaceDetection(2, DetectionHostType.Video, 4040, 0.62f, observedAtSec: 12.0));

            await db.SaveChangesAsync();
        }

        await using (var scope = provider.CreateAsyncScope())
        {
            var suggester = scope.ServiceProvider.GetRequiredService<IFaceSuggester>();
            var suggestions = await suggester.SuggestForAsync(1, 5);

            Assert.Equal(2, suggestions.Count);
            Assert.Equal(11, suggestions[0].PerformerId);
            Assert.Equal("Mia Melon", suggestions[0].PerformerName);
            Assert.Contains("Host evidence only", suggestions[0].Why, StringComparison.Ordinal);
            Assert.True(suggestions[0].Confidence > suggestions[1].Confidence);
            Assert.Equal(22, suggestions[1].PerformerId);
        }
    }

    [Fact]
    public async Task SuggestAsync_ReturnsHostPerformerSuggestionWithoutFaceEmbeddingMatches()
    {
        await using var provider = CreateProvider();

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Performers.Add(new Performer { Id = 77, Name = "Video Performer" });
            db.Videos.AddRange(
                new Video { Id = 7001, Title = "Tagged Video A" },
                new Video { Id = 7002, Title = "Tagged Video B" });
            db.Set<VideoPerformer>().AddRange(
                new VideoPerformer { VideoId = 7001, PerformerId = 77 },
                new VideoPerformer { VideoId = 7002, PerformerId = 77 });
            db.Faces.Add(new Face { Id = 5, Label = "No Embedding Face", PrimarySourceKey = "face-0005" });
            db.Detections.AddRange(
                CreateFaceDetection(5, DetectionHostType.Video, 7001, 0.91f, observedAtSec: 12.0),
                CreateFaceDetection(5, DetectionHostType.Video, 7002, 0.88f, observedAtSec: 48.0));

            await db.SaveChangesAsync();
        }

        await using (var scope = provider.CreateAsyncScope())
        {
            var suggester = scope.ServiceProvider.GetRequiredService<IFaceSuggester>();
            var suggestions = await suggester.SuggestForAsync(5, 5);

            var suggestion = Assert.Single(suggestions);
            Assert.Equal(77, suggestion.PerformerId);
            Assert.Equal("Video Performer", suggestion.PerformerName);
            Assert.Contains("Host evidence only", suggestion.Why, StringComparison.Ordinal);
            Assert.Contains("2 tagged hosts", suggestion.Why, StringComparison.Ordinal);
        }
    }

    [Fact]
    public async Task SuggestAsync_KeepsSingleHostOnlySuggestionConfidenceWeak()
    {
        await using var provider = CreateProvider();

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Performers.Add(new Performer { Id = 78, Name = "Only Video Performer" });
            db.Videos.Add(new Video { Id = 7101, Title = "Single Tagged Video" });
            db.Set<VideoPerformer>().Add(new VideoPerformer { VideoId = 7101, PerformerId = 78 });
            db.Faces.Add(new Face { Id = 6, Label = "Single Host Face", PrimarySourceKey = "face-0006" });
            db.Detections.Add(CreateFaceDetection(6, DetectionHostType.Video, 7101, 0.91f, observedAtSec: 12.0));

            await db.SaveChangesAsync();
        }

        await using (var scope = provider.CreateAsyncScope())
        {
            var suggester = scope.ServiceProvider.GetRequiredService<IFaceSuggester>();
            var suggestion = Assert.Single(await suggester.SuggestForAsync(6, 5));

            Assert.Equal(78, suggestion.PerformerId);
            Assert.Equal(40f, suggestion.Confidence);
            Assert.Contains("Host evidence only", suggestion.Why, StringComparison.Ordinal);
        }
    }

    private static ServiceProvider CreateProvider(string? referenceRoot = null)
    {
        var services = new ServiceCollection();
        var databaseName = $"ai-face-suggester-{Guid.NewGuid():N}";
        var databaseRoot = new InMemoryDatabaseRoot();
        var extensionStore = new TestExtensionStore();
        referenceRoot ??= Path.Combine(Path.GetTempPath(), $"ai-face-suggester-pack-{Guid.NewGuid():N}");
        Directory.CreateDirectory(referenceRoot);

        services.AddDbContext<CoveContext>(options => options.UseInMemoryDatabase(databaseName, databaseRoot));
        services.AddSingleton<SaieArchiveReader>();
        services.AddSingleton(services =>
        {
            var store = new AiFaceReferencePackStore(referenceRoot, services.GetRequiredService<SaieArchiveReader>());
            store.Attach(extensionStore);
            return store;
        });
        services.AddSingleton(_ =>
        {
            var store = new AiFaceReferenceSuggestionDecisionStore();
            store.Attach(extensionStore);
            return store;
        });
        services.AddScoped<IEmbeddingService, EmbeddingService>();
        services.AddScoped<AiFaceReferencePerformerResolver>();
        services.AddScoped<IFaceSuggester, AiFaceSuggester>();

        return services.BuildServiceProvider();
    }

    private static Embedding CreateFaceEmbedding(int faceId, IReadOnlyList<float> vector)
        => new()
        {
            HostType = EmbeddingHostType.Face,
            HostId = faceId,
            Kind = "face.embed.v1",
            KindFamily = "face.v1",
            Modality = EmbeddingModality.Face,
            SourceKey = "ext:ai.faces",
            Dim = vector.Count,
            Vector = new Vector(vector.ToArray()),
        };

    private static Detection CreateFaceDetection(int faceId, DetectionHostType hostType, int hostId, float score, double? observedAtSec = null)
        => new()
        {
            HostType = hostType,
            HostId = hostId,
            ObservedAtSec = observedAtSec,
            FrameWidth = 1,
            FrameHeight = 1,
            Class = "face",
            Score = score,
            X = 0.1f,
            Y = 0.1f,
            W = 0.3f,
            H = 0.3f,
            RefKind = "face",
            RefId = faceId,
            SourceKey = "ext:ai.faces",
        };

    private static string CreateReferenceArchive(string root, object manifest, IReadOnlyList<object> performers, IReadOnlyList<float> centroids, int rows, int columns)
    {
        var path = Path.Combine(root, $"reference-{Guid.NewGuid():N}.saie");

        using var file = File.Create(path);
        using var archive = new System.IO.Compression.ZipArchive(file, System.IO.Compression.ZipArchiveMode.Create, leaveOpen: false);

        var manifestEntry = archive.CreateEntry("manifest.json", System.IO.Compression.CompressionLevel.Fastest);
        using (var writer = new StreamWriter(manifestEntry.Open()))
        {
            writer.Write(System.Text.Json.JsonSerializer.Serialize(manifest));
        }

        var performersEntry = archive.CreateEntry("performers.jsonl", System.IO.Compression.CompressionLevel.Fastest);
        using (var writer = new StreamWriter(performersEntry.Open()))
        {
            foreach (var performer in performers)
                writer.WriteLine(System.Text.Json.JsonSerializer.Serialize(performer));
        }

        var centroidsEntry = archive.CreateEntry("centroids.npy", System.IO.Compression.CompressionLevel.NoCompression);
        using (var stream = centroidsEntry.Open())
        {
            WriteNpy(stream, centroids, rows, columns);
        }

        return path;
    }

    private static void WriteNpy(Stream stream, IReadOnlyList<float> values, int rows, int columns)
    {
        using var writer = new BinaryWriter(stream, System.Text.Encoding.ASCII, leaveOpen: true);

        writer.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 0x01, 0x00 });

        var header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {columns}), }}";
        var preambleLength = 10;
        var paddingLength = 16 - ((preambleLength + header.Length + 1) % 16);
        if (paddingLength == 16)
            paddingLength = 0;

        var paddedHeader = header + new string(' ', paddingLength) + '\n';
        var headerBytes = System.Text.Encoding.ASCII.GetBytes(paddedHeader);

        writer.Write((ushort)headerBytes.Length);
        writer.Write(headerBytes);

        foreach (var value in values)
            writer.Write(value);
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

}