using System.Text.Json;

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

public sealed class AiFaceSplitServiceTests
{
    private const string SourceKey = "ext:ai.faces";

    [Fact]
    public async Task Split_MovesTheNamedTrackOffTheFaceWithinOneVideo()
    {
        // Jane and a second performer both landed on one face inside video 101, plus Jane alone in 102.
        // Not-present cannot help here — it works per host — so the second performer's track is split off.
        AiFacesSettingsRuntime.Attach(new FixedSettingsStore());
        var extensionStore = new TestExtensionStore();
        var identityStore = new InMemoryFaceIdentityStore(new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 2,
            Identities =
            [
                new StoredFaceIdentity
                {
                    FaceKey = "face-0001",
                    Label = "Jane",
                    LifecycleStatus = StoredFaceIdentityLifecycle.Promoted,
                    PromotionReason = "video-evidence",
                    AssetIds = ["video-101", "video-102"],
                    Anchors =
                    [
                        new StoredFaceAnchor { ModelKey = "m", QualityScore = 20, Vector = [1f, 0f] },
                        new StoredFaceAnchor { ModelKey = "m", QualityScore = 18, Vector = [0f, 1f] },
                    ],
                },
            ],
        });
        await using var provider = CreateProvider(extensionStore, identityStore);

        int faceId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var face = new Face { Label = "Jane", PrimarySourceKey = "face-0001" };
            db.Faces.Add(face);
            await db.SaveChangesAsync();
            faceId = face.Id;

            AddTrack(db, faceId, hostId: 101, runId: "run-101", groupKey: "asset-face-1", vector: [1f, 0f]);
            AddTrack(db, faceId, hostId: 101, runId: "run-101", groupKey: "asset-face-2", vector: [0f, 1f]);
            AddTrack(db, faceId, hostId: 102, runId: "run-102", groupKey: "asset-face-1", vector: [0.99f, 0.0447f]);
            await db.SaveChangesAsync();
        }

        var exclusionStore = provider.GetRequiredService<AiFaceIdentityExclusionStore>();
        var service = new AiFaceSplitService(provider.GetRequiredService<IServiceScopeFactory>(), exclusionStore);

        var result = await service.SplitAsync(faceId, "video", 101, ["asset-face-2"]);

        Assert.True(result.FaceFound);
        Assert.True(result.GroupKeysMatched);
        Assert.True(result.CreatedNewFace);
        Assert.True(result.RecordedIdentitySplit);
        Assert.Equal(1, result.MovedAppearanceCount);
        Assert.Equal(1, result.MovedDetectionCount);
        Assert.Equal(1, result.MovedSegmentCount);
        Assert.Equal(1, result.MovedEmbeddingCount);
        Assert.NotEqual(faceId, result.TargetFaceId);

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();

            // The other track on the same video, and the other video, stay put.
            var appearances = await db.FaceAppearances.ToListAsync();
            Assert.Equal(faceId, appearances.Single(a => a.HostId == 101 && a.GroupKey == "asset-face-1").FaceId);
            Assert.Equal(faceId, appearances.Single(a => a.HostId == 102).FaceId);
            Assert.Equal(result.TargetFaceId, appearances.Single(a => a.HostId == 101 && a.GroupKey == "asset-face-2").FaceId);

            var movedDetection = await db.Detections.SingleAsync(d => d.GroupKey == "asset-face-2");
            Assert.Equal(result.TargetFaceId, (int)movedDetection.RefId!.Value);

            var movedSegment = await db.Segments.SingleAsync(s => s.Title == "asset-face-2");
            Assert.Equal(result.TargetFaceId, (int)movedSegment.RefId!.Value);

            var movedEmbedding = await db.Embeddings.SingleAsync(e =>
                e.Meta!.RootElement.GetProperty("trackKey").GetString() == "asset-face-2");
            Assert.Equal(result.TargetFaceId, movedEmbedding.HostId);
        }

        // The identity graph learned the split, so a re-run cannot rebuild the tangle: the wrong person's
        // anchor left the source identity and seeds the new one.
        var snapshot = await identityStore.LoadAsync();
        var source = snapshot.Identities.Single(identity => identity.FaceKey == "face-0001");
        var created = snapshot.Identities.Single(identity => identity.FaceKey != "face-0001");
        Assert.Equal([1f, 0f], Assert.Single(source.Anchors).Vector);
        Assert.Equal([0f, 1f], Assert.Single(created.Anchors).Vector);
        Assert.Equal(result.TargetFaceKey, created.FaceKey);

        var exclusions = await exclusionStore.LoadAsync();
        Assert.True(exclusions.Excludes("face-0001", created.FaceKey));
    }

    [Fact]
    public async Task Split_RefusesToEmptyTheFace()
    {
        AiFacesSettingsRuntime.Attach(new FixedSettingsStore());
        await using var provider = CreateProvider(new TestExtensionStore(), new InMemoryFaceIdentityStore(new FaceIdentitySnapshot()));

        int faceId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var face = new Face { Label = "Jane", PrimarySourceKey = "face-0001" };
            db.Faces.Add(face);
            await db.SaveChangesAsync();
            faceId = face.Id;

            AddTrack(db, faceId, hostId: 101, runId: "run-101", groupKey: "asset-face-1", vector: [1f, 0f]);
            await db.SaveChangesAsync();
        }

        var service = new AiFaceSplitService(
            provider.GetRequiredService<IServiceScopeFactory>(),
            provider.GetRequiredService<AiFaceIdentityExclusionStore>());

        var result = await service.SplitAsync(faceId, "video", 101, ["asset-face-1"]);

        Assert.True(result.WouldEmptyFace);
        Assert.Equal(0, result.MovedAppearanceCount);
    }

    [Fact]
    public async Task GetHostTracks_ListsTheFacesTracksOnThatHostOnly()
    {
        AiFacesSettingsRuntime.Attach(new FixedSettingsStore());
        await using var provider = CreateProvider(new TestExtensionStore(), new InMemoryFaceIdentityStore(new FaceIdentitySnapshot()));

        int faceId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var face = new Face { Label = "Jane", PrimarySourceKey = "face-0001" };
            db.Faces.Add(face);
            await db.SaveChangesAsync();
            faceId = face.Id;

            AddTrack(db, faceId, hostId: 101, runId: "run-101", groupKey: "asset-face-1", vector: [1f, 0f], firstSeen: 1);
            AddTrack(db, faceId, hostId: 101, runId: "run-101", groupKey: "asset-face-2", vector: [0f, 1f], firstSeen: 5);
            AddTrack(db, faceId, hostId: 102, runId: "run-102", groupKey: "asset-face-1", vector: [1f, 0f], firstSeen: 2);
            await db.SaveChangesAsync();
        }

        var service = new AiFaceSplitService(
            provider.GetRequiredService<IServiceScopeFactory>(),
            provider.GetRequiredService<AiFaceIdentityExclusionStore>());

        var tracks = await service.GetHostTracksAsync(faceId, "video", 101);

        Assert.Equal(["asset-face-1", "asset-face-2"], tracks.Select(track => track.GroupKey));
        Assert.All(tracks, track => Assert.NotNull(track.RepresentativeBoundingBox));
    }

    [Fact]
    public async Task GetHostTracks_GroupsTheAppearancesByWhoTheyLookLike()
    {
        // Jane appears three times and a second performer twice. Every pair here scores above the
        // consolidation floor — which is exactly why the two ended up on one face — so similarity alone
        // would put all five in one group. Being on screen together is what forces them apart.
        AiFacesSettingsRuntime.Attach(new FixedSettingsStore());
        await using var provider = CreateProvider(new TestExtensionStore(), new InMemoryFaceIdentityStore(new FaceIdentitySnapshot()));

        int faceId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var face = new Face { Label = "Jane", PrimarySourceKey = "face-0001" };
            db.Faces.Add(face);
            await db.SaveChangesAsync();
            faceId = face.Id;

            // Jane: 0-10s, 20-30s, 60-70s.
            AddTrack(db, faceId, 101, "run-101", "jane-a", [1f, 0f], firstSeen: 0, lastSeen: 10);
            AddTrack(db, faceId, 101, "run-101", "jane-b", [0.99f, 0.1411f], firstSeen: 20, lastSeen: 30);
            AddTrack(db, faceId, 101, "run-101", "jane-c", [0.97f, 0.2431f], firstSeen: 60, lastSeen: 70);
            // The other performer, on screen at the same time as Jane's first two clips.
            AddTrack(db, faceId, 101, "run-101", "other-a", [0.87f, 0.4931f], firstSeen: 2, lastSeen: 9);
            AddTrack(db, faceId, 101, "run-101", "other-b", [0.86f, 0.5104f], firstSeen: 22, lastSeen: 29);
            await db.SaveChangesAsync();
        }

        var service = new AiFaceSplitService(
            provider.GetRequiredService<IServiceScopeFactory>(),
            provider.GetRequiredService<AiFaceIdentityExclusionStore>());

        var tracks = await service.GetHostTracksAsync(faceId, "video", 101);
        var groupByKey = tracks.ToDictionary(track => track.GroupKey, track => track.SuggestedGroup);

        // Jane's three clips land together and the other performer's two land together, without the two
        // people being mixed.
        Assert.Equal(groupByKey["jane-a"], groupByKey["jane-b"]);
        Assert.Equal(groupByKey["jane-a"], groupByKey["jane-c"]);
        Assert.Equal(groupByKey["other-a"], groupByKey["other-b"]);
        Assert.NotEqual(groupByKey["jane-a"], groupByKey["other-a"]);

        // Group 0 is the dominant person, so the pre-selected "odd ones out" are the other performer.
        Assert.Equal(0, groupByKey["jane-a"]);
        Assert.Equal(1, groupByKey["other-a"]);
    }

    [Fact]
    public async Task GetHostTracks_KeepsOneNonOverlappingPersonInASingleGroup()
    {
        // The common case: one performer whose run was broken into several clips. Nothing should be
        // proposed for separation, so the dialog offers no pre-selection.
        AiFacesSettingsRuntime.Attach(new FixedSettingsStore());
        await using var provider = CreateProvider(new TestExtensionStore(), new InMemoryFaceIdentityStore(new FaceIdentitySnapshot()));

        int faceId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var face = new Face { Label = "Jane", PrimarySourceKey = "face-0001" };
            db.Faces.Add(face);
            await db.SaveChangesAsync();
            faceId = face.Id;

            AddTrack(db, faceId, 101, "run-101", "a", [1f, 0f], firstSeen: 0, lastSeen: 5);
            AddTrack(db, faceId, 101, "run-101", "b", [0.99f, 0.1411f], firstSeen: 10, lastSeen: 15);
            AddTrack(db, faceId, 101, "run-101", "c", [0.98f, 0.1987f], firstSeen: 20, lastSeen: 25);
            await db.SaveChangesAsync();
        }

        var service = new AiFaceSplitService(
            provider.GetRequiredService<IServiceScopeFactory>(),
            provider.GetRequiredService<AiFaceIdentityExclusionStore>());

        var tracks = await service.GetHostTracksAsync(faceId, "video", 101);

        Assert.All(tracks, track => Assert.Equal(0, track.SuggestedGroup));
    }

    [Fact]
    public async Task GetHostTracks_SeparatesADissimilarAppearanceEvenWithoutOverlap()
    {
        // Two people who are never on screen together still separate when they simply do not look alike,
        // so the grouping is not relying on the time constraint alone.
        AiFacesSettingsRuntime.Attach(new FixedSettingsStore());
        await using var provider = CreateProvider(new TestExtensionStore(), new InMemoryFaceIdentityStore(new FaceIdentitySnapshot()));

        int faceId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var face = new Face { Label = "Jane", PrimarySourceKey = "face-0001" };
            db.Faces.Add(face);
            await db.SaveChangesAsync();
            faceId = face.Id;

            AddTrack(db, faceId, 101, "run-101", "jane-a", [1f, 0f], firstSeen: 0, lastSeen: 5);
            AddTrack(db, faceId, 101, "run-101", "jane-b", [0.99f, 0.1411f], firstSeen: 10, lastSeen: 15);
            // Well below the 0.72 consolidation floor, and never on screen with the others.
            AddTrack(db, faceId, 101, "run-101", "stranger", [0f, 1f], firstSeen: 30, lastSeen: 35);
            await db.SaveChangesAsync();
        }

        var service = new AiFaceSplitService(
            provider.GetRequiredService<IServiceScopeFactory>(),
            provider.GetRequiredService<AiFaceIdentityExclusionStore>());

        var tracks = await service.GetHostTracksAsync(faceId, "video", 101);
        var groupByKey = tracks.ToDictionary(track => track.GroupKey, track => track.SuggestedGroup);

        Assert.Equal(groupByKey["jane-a"], groupByKey["jane-b"]);
        Assert.NotEqual(groupByKey["jane-a"], groupByKey["stranger"]);
        Assert.Equal(0, groupByKey["jane-a"]);
    }

    private static void AddTrack(
        CoveContext db,
        int faceId,
        int hostId,
        string runId,
        string groupKey,
        IReadOnlyList<float> vector,
        double firstSeen = 1,
        double? lastSeen = null)
    {
        db.FaceAppearances.Add(new FaceAppearance
        {
            FaceId = faceId,
            HostType = FaceAppearanceHostType.Video,
            HostId = hostId,
            GroupKey = groupKey,
            FirstSeenAtSec = firstSeen,
            LastSeenAtSec = lastSeen ?? firstSeen + 3,
            SampleCount = 4,
            SourceKey = SourceKey,
            SourceRunId = runId,
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
            GroupKey = groupKey,
            SourceKey = SourceKey,
            SourceRunId = runId,
        });
        db.Segments.Add(new Segment
        {
            HostType = SegmentHostType.Video,
            HostId = hostId,
            StartSec = firstSeen,
            EndSec = firstSeen + 3,
            Kind = "face",
            RefId = faceId,
            Title = groupKey,
            Payload = JsonDocument.Parse($$"""{"trackKey":"{{groupKey}}"}"""),
            SourceKey = SourceKey,
            SourceRunId = runId,
        });
        db.Embeddings.Add(new Embedding
        {
            HostType = EmbeddingHostType.Face,
            HostId = faceId,
            Kind = "face.embed.v1",
            KindFamily = "face.v1",
            Modality = EmbeddingModality.Face,
            Dim = vector.Count,
            Vector = new Vector(vector.ToArray()),
            Meta = JsonDocument.Parse($$"""{"trackKey":"{{groupKey}}"}"""),
            SourceKey = SourceKey,
            SourceRunId = runId,
        });
    }

    private static ServiceProvider CreateProvider(IExtensionStore extensionStore, IFaceIdentityStore identityStore)
    {
        var services = new ServiceCollection();
        var databaseName = $"ai-face-split-{Guid.NewGuid():N}";
        var databaseRoot = new InMemoryDatabaseRoot();
        services.AddDbContext<CoveContext>(options => options.UseInMemoryDatabase(databaseName, databaseRoot));
        services.AddScoped<IFaceRepository, FaceRepository>();
        services.AddScoped<IEmbeddingRepository, EmbeddingRepository>();
        services.AddScoped<IDetectionRepository, DetectionRepository>();
        services.AddScoped<ISegmentRepository, SegmentRepository>();
        services.AddScoped<IEmbeddingService, EmbeddingService>();
        services.AddSingleton(identityStore);
        services.AddSingleton(_ =>
        {
            var store = new AiFaceIdentityExclusionStore();
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

    private sealed class InMemoryFaceIdentityStore(FaceIdentitySnapshot seed) : IFaceIdentityStore
    {
        private FaceIdentitySnapshot _snapshot = seed;

        public Task<FaceIdentitySnapshot> LoadAsync(CancellationToken ct = default)
            => Task.FromResult(_snapshot);

        public Task<FaceIdentityTransaction> BeginIncrementalAsync(
            IReadOnlyList<IReadOnlyList<float>> queryVectors,
            IReadOnlyCollection<string> referenceExternalIds,
            int candidateK,
            CancellationToken ct = default)
            => Begin();

        public Task<FaceIdentityTransaction> BeginFullAsync(CancellationToken ct = default) => Begin();

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
                    QualityScore = identity.QualityScore,
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
            public override FaceIdentitySnapshot Snapshot { get; } = snapshot;

            public override Task CommitAsync(CancellationToken ct = default)
            {
                commit(Snapshot);
                return Task.CompletedTask;
            }

            public override ValueTask DisposeAsync() => ValueTask.CompletedTask;
        }
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
