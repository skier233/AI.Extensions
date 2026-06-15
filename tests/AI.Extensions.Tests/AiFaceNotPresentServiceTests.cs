using AI.Faces;

using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Cove.Data;
using Cove.Data.Repositories;
using Cove.Data.Services;
using Cove.Plugins;

using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Microsoft.Extensions.DependencyInjection;

using Pgvector;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiFaceNotPresentServiceTests
{
    [Fact]
    public async Task MarkNotPresent_SplitsWrongPersonOccurrenceIntoNewFaceAndRecordsSuppression()
    {
        AiFacesSettingsRuntime.Attach(new FixedSettingsStore());
        var extensionStore = new TestExtensionStore();
        await using var provider = CreateProvider(extensionStore);

        int faceId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            // Jane in videos 101 & 102; a different person (Emily) wrongly lumped into the same face from
            // video 103.
            var face = new Face { Label = "Jane", PrimarySourceKey = "face-0001" };
            db.Faces.Add(face);
            await db.SaveChangesAsync();
            faceId = face.Id;

            AddOccurrence(db, faceId, hostId: 101, runId: "run-101", vector: [1f, 0f]);
            AddOccurrence(db, faceId, hostId: 102, runId: "run-102", vector: [0.99f, 0.0447f]);
            AddOccurrence(db, faceId, hostId: 103, runId: "run-103", vector: [0f, 1f]);
            await db.SaveChangesAsync();
        }

        var suppressionStore = provider.GetRequiredService<AiFacePresenceSuppressionStore>();
        var service = new AiFaceNotPresentService(provider.GetRequiredService<IServiceScopeFactory>(), suppressionStore);

        var result = await service.MarkNotPresentAsync(faceId, "video", 103);

        Assert.True(result.FaceFound);
        Assert.True(result.HostHadFace);
        Assert.Equal(1, result.MovedHostCount);
        Assert.True(result.CreatedNewFace);
        Assert.False(result.SourceFaceEmptied);
        Assert.NotNull(result.TargetFaceId);
        Assert.NotEqual(faceId, result.TargetFaceId);

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var appearances = await db.FaceAppearances.OrderBy(a => a.HostId).ToListAsync();

            Assert.Equal(faceId, appearances.Single(a => a.HostId == 101).FaceId);
            Assert.Equal(faceId, appearances.Single(a => a.HostId == 102).FaceId);
            Assert.Equal(result.TargetFaceId, appearances.Single(a => a.HostId == 103).FaceId);

            // The wrong-person embedding moved with its occurrence.
            var movedEmbedding = await db.Embeddings.SingleAsync(e => e.SourceRunId == "run-103");
            Assert.Equal(result.TargetFaceId, movedEmbedding.HostId);

            var newFace = await db.Faces.SingleAsync(f => f.Id == result.TargetFaceId);
            Assert.StartsWith("face-split-", newFace.PrimarySourceKey);
        }

        var suppressed = await suppressionStore.GetSuppressedFaceKeysAsync("video", 103);
        Assert.Contains("face-0001", suppressed);
    }

    [Fact]
    public async Task MarkNotPresent_ReturnsNoFaceOnHostWhenFaceNotPresentThere()
    {
        AiFacesSettingsRuntime.Attach(new FixedSettingsStore());
        var extensionStore = new TestExtensionStore();
        await using var provider = CreateProvider(extensionStore);

        int faceId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var face = new Face { Label = "Jane", PrimarySourceKey = "face-0001" };
            db.Faces.Add(face);
            await db.SaveChangesAsync();
            faceId = face.Id;

            AddOccurrence(db, faceId, hostId: 101, runId: "run-101", vector: [1f, 0f]);
            await db.SaveChangesAsync();
        }

        var service = new AiFaceNotPresentService(
            provider.GetRequiredService<IServiceScopeFactory>(),
            provider.GetRequiredService<AiFacePresenceSuppressionStore>());

        var result = await service.MarkNotPresentAsync(faceId, "video", 999);

        Assert.True(result.FaceFound);
        Assert.False(result.HostHadFace);
    }

    private static void AddOccurrence(CoveContext db, int faceId, int hostId, string runId, IReadOnlyList<float> vector)
    {
        db.FaceAppearances.Add(new FaceAppearance
        {
            FaceId = faceId,
            HostType = FaceAppearanceHostType.Video,
            HostId = hostId,
            SourceKey = "ext:ai.faces",
            SourceRunId = runId,
            SampleCount = 4,
        });
        db.Detections.Add(new Detection
        {
            HostType = DetectionHostType.Video,
            HostId = hostId,
            FrameWidth = 1,
            FrameHeight = 1,
            Class = "face",
            Score = 0.95f,
            X = 0.1f,
            Y = 0.1f,
            W = 0.3f,
            H = 0.3f,
            RefKind = "face",
            RefId = faceId,
            SourceKey = "ext:ai.faces",
            SourceRunId = runId,
        });
        db.Embeddings.Add(new Embedding
        {
            HostType = EmbeddingHostType.Face,
            HostId = faceId,
            Kind = "face.embed.v1",
            KindFamily = "face.v1",
            Modality = EmbeddingModality.Face,
            SourceKey = "ext:ai.faces",
            SourceRunId = runId,
            Dim = vector.Count,
            Vector = new Vector(vector.ToArray()),
        });
    }

    private static ServiceProvider CreateProvider(IExtensionStore extensionStore)
    {
        var services = new ServiceCollection();
        var databaseName = $"ai-face-not-present-{Guid.NewGuid():N}";
        var databaseRoot = new InMemoryDatabaseRoot();
        services.AddDbContext<CoveContext>(options => options.UseInMemoryDatabase(databaseName, databaseRoot));
        services.AddScoped<IFaceRepository, FaceRepository>();
        services.AddScoped<IEmbeddingRepository, EmbeddingRepository>();
        services.AddScoped<IDetectionRepository, DetectionRepository>();
        services.AddScoped<ISegmentRepository, SegmentRepository>();
        services.AddScoped<IEmbeddingService, EmbeddingService>();
        services.AddSingleton(_ =>
        {
            var store = new AiFacePresenceSuppressionStore();
            store.Attach(extensionStore);
            return store;
        });
        return services.BuildServiceProvider();
    }

    private sealed class FixedSettingsStore : IAiFacesSettingsStore
    {
        public Task<AiFacesSettings> LoadAsync(CancellationToken ct = default)
            => Task.FromResult(new AiFacesSettings().Normalize());

        public Task SaveAsync(AiFacesSettings settings, CancellationToken ct = default)
            => Task.CompletedTask;
    }

    private sealed class TestExtensionStore : IExtensionStore
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
