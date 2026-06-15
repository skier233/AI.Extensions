using System.Text.Json;

using AI.Faces;

using Cove.Core.Entities;
using Cove.Data;
using Cove.Plugins;

using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Microsoft.Extensions.DependencyInjection;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiFaceReferenceBackfillServiceTests
{
    [Fact]
    public async Task BackfillAsync_MergesPersistedDuplicatesAndRelabelsTargetAfterReferenceImport()
    {
        await using var provider = CreateProvider(services =>
        {
            services.AddScoped<DbContext>(static sp => sp.GetRequiredService<CoveContext>());
            services.AddSingleton<StoreBackedFaceIdentityStateStore>();
            services.AddSingleton<IFaceIdentityStateStore>(static services => services.GetRequiredService<StoreBackedFaceIdentityStateStore>());
            services.AddSingleton<IFaceIdentityStore, DbFaceIdentityStore>();
            services.AddSingleton<StoreBackedAiFacesSettingsStore>();
            services.AddSingleton<IAiFacesSettingsStore>(static services => services.GetRequiredService<StoreBackedAiFacesSettingsStore>());
            services.AddSingleton<AiFaceIdentityReconciler>();
            services.AddSingleton<AiFaceReferenceBackfillService>();
            services.AddScoped<Cove.Core.Interfaces.IFaceRepository, Cove.Data.Repositories.FaceRepository>();
            services.AddScoped<Cove.Core.Interfaces.IEmbeddingRepository, Cove.Data.Repositories.EmbeddingRepository>();
            services.AddScoped<Cove.Core.Interfaces.IDetectionRepository, Cove.Data.Repositories.DetectionRepository>();
            services.AddScoped<Cove.Core.Interfaces.ISegmentRepository, Cove.Data.Repositories.SegmentRepository>();
        });

        var extensionStore = new TestExtensionStore();
        provider.GetRequiredService<StoreBackedFaceIdentityStateStore>().Attach(extensionStore);
        provider.GetRequiredService<StoreBackedAiFacesSettingsStore>().Attach(extensionStore);

        var snapshot = new FaceIdentitySnapshot
        {
            NextIdentityOrdinal = 3,
            Identities =
            [
                CreatePromotedIdentity("face-0001", [1f, 0f], "video-5634"),
                CreatePromotedIdentity("face-0002", [0.55f, 0.8351647f], "video-5634"),
            ],
        };
        await provider.GetRequiredService<IFaceIdentityStateStore>().SaveAsync(snapshot);

        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var target = new Face { Label = "face-0001", PrimarySourceKey = "face-0001" };
            var duplicate = new Face { Label = "face-0002", PrimarySourceKey = "face-0002" };
            db.Faces.AddRange(target, duplicate);
            await db.SaveChangesAsync();

            db.FaceAppearances.AddRange(
                new FaceAppearance { FaceId = target.Id, HostType = FaceAppearanceHostType.Video, HostId = 5634, SourceKey = "ext:ai.faces", SampleCount = 1 },
                new FaceAppearance { FaceId = duplicate.Id, HostType = FaceAppearanceHostType.Video, HostId = 5634, SourceKey = "ext:ai.faces", SampleCount = 1 });
            await db.SaveChangesAsync();
        }

        var service = provider.GetRequiredService<AiFaceReferenceBackfillService>();
        var referencePack = CreateReferencePack();

        var result = await service.BackfillAsync(referencePack);

        // The reconciled identity graph now lives in the DB-backed store (the legacy blob is cleared on
        // import), so verify the merged/relabeled identity there.
        await using (var identityScope = provider.CreateAsyncScope())
        {
            var identityDb = identityScope.ServiceProvider.GetRequiredService<CoveContext>();
            var identity = Assert.Single(await identityDb.Set<ExtAiFacesIdentityEntity>().ToListAsync());
            Assert.Equal("face-0001", identity.FaceKey);
            Assert.Equal("ref-zazie", identity.ReferenceExternalId);
            Assert.Equal("Zazie Skymm", identity.Label);
        }

        await using var verificationScope = provider.CreateAsyncScope();
        var verificationDb = verificationScope.ServiceProvider.GetRequiredService<CoveContext>();
        var faces = await verificationDb.Faces.OrderBy(face => face.Id).ToListAsync();
        var targetFace = Assert.Single(faces, face => face.PrimarySourceKey == "face-0001");
        var duplicateFace = Assert.Single(faces, face => face.PrimarySourceKey == "face-0002");
        var appearances = await verificationDb.FaceAppearances.OrderBy(appearance => appearance.Id).ToListAsync();

        Assert.Equal("Zazie Skymm", targetFace.Label);
        Assert.Equal(targetFace.Id, duplicateFace.MergedIntoFaceId);
        Assert.All(appearances, appearance => Assert.Equal(targetFace.Id, appearance.FaceId));

        Assert.Equal(1, result.MergedIdentityCount);
        Assert.Equal(0, result.ReferencePromotedIdentityCount);
        Assert.Equal(1, result.MergedPersistedFaceCount);
        Assert.Equal(1, result.RelabeledPersistedFaceCount);
    }

    private static StoredFaceIdentity CreatePromotedIdentity(string faceKey, IReadOnlyList<float> vector, string assetId)
        => new()
        {
            FaceKey = faceKey,
            Label = faceKey,
            LifecycleStatus = StoredFaceIdentityLifecycle.Promoted,
            PromotionReason = "video-evidence",
            ObservationCount = 4,
            AssetIds = [assetId],
            Anchors =
            [
                new StoredFaceAnchor
                {
                    ModelKey = "face_embedding_torchexport",
                    QualityScore = 20.0,
                    Vector = vector.ToList(),
                },
            ],
        };

    private static SaieReferencePack CreateReferencePack()
        => new(
            new SaieManifest(1, "nsfw_ai_server:image_pipeline_dynamic_v4", 2, "pack-zazie", null, 1, DateTimeOffset.UtcNow),
            [new SaieReferenceIdentity(0, "ref-zazie", "Zazie Skymm", [], null, 1, 1.0, null)],
            [1f, 0f]);

    private static ServiceProvider CreateProvider(Action<ServiceCollection>? configure = null)
    {
        var services = new ServiceCollection();
        var databaseName = $"ai-face-reference-backfill-{Guid.NewGuid():N}";
        var databaseRoot = new InMemoryDatabaseRoot();
        services.AddDbContext<CoveContext>(options => options.UseInMemoryDatabase(databaseName, databaseRoot));
        configure?.Invoke(services);
        return services.BuildServiceProvider();
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
}