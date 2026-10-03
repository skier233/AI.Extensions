using AI.Faces;

using Cove.Data;

using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Microsoft.Extensions.DependencyInjection;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class DbFaceIdentityTransactionTests
{
    [Fact]
    public async Task CommitAsync_RewritesOnlyIdentitiesThatChanged()
    {
        var services = new ServiceCollection();
        var databaseName = $"ai-face-identity-transaction-{Guid.NewGuid():N}";
        var databaseRoot = new InMemoryDatabaseRoot();
        services.AddDbContext<CoveContext>(options => options.UseInMemoryDatabase(databaseName, databaseRoot));
        services.AddScoped<DbContext>(static sp => sp.GetRequiredService<CoveContext>());
        services.AddSingleton<StoreBackedFaceIdentityStateStore>();
        services.AddSingleton<IFaceIdentityStateStore>(static sp => sp.GetRequiredService<StoreBackedFaceIdentityStateStore>());
        services.AddSingleton<IFaceIdentityStore, DbFaceIdentityStore>();
        await using var provider = services.BuildServiceProvider();

        var seededAt = new DateTime(2026, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            db.Set<ExtAiFacesIdentityEntity>().AddRange(
                DbFaceIdentityStore.NewEntity(CreateIdentity("face-0001", [1f, 0f]), seededAt),
                DbFaceIdentityStore.NewEntity(CreateIdentity("face-0002", [0f, 1f]), seededAt),
                DbFaceIdentityStore.NewEntity(CreateIdentity("face-0003", [0.6f, 0.8f]), seededAt));
            await db.SaveChangesAsync();
        }

        var anchorIdsBefore = await LoadAnchorIdsAsync(provider);

        await using (var transaction = await provider.GetRequiredService<IFaceIdentityStore>().BeginFullAsync())
        {
            // face-0001 is only looked at; face-0002 changes a scalar; face-0003 gains an anchor.
            transaction.Snapshot.Identities.Single(identity => identity.FaceKey == "face-0002").ObservationCount = 9;
            transaction.Snapshot.Identities.Single(identity => identity.FaceKey == "face-0003").Anchors.Add(
                new StoredFaceAnchor { ModelKey = "face_embedding_torchexport", QualityScore = 18.0, Vector = [0.8f, 0.6f] });
            await transaction.CommitAsync();
        }

        await using var verificationScope = provider.CreateAsyncScope();
        var identities = await verificationScope.ServiceProvider.GetRequiredService<CoveContext>()
            .Set<ExtAiFacesIdentityEntity>()
            .Include(identity => identity.Anchors)
            .ToDictionaryAsync(identity => identity.FaceKey);

        var untouched = identities["face-0001"];
        Assert.Equal(seededAt, untouched.UpdatedAt);
        Assert.Equal(anchorIdsBefore["face-0001"], untouched.Anchors.Select(static anchor => anchor.Id).ToArray());

        var scalarChanged = identities["face-0002"];
        Assert.Equal(9, scalarChanged.ObservationCount);
        Assert.True(scalarChanged.UpdatedAt > seededAt);
        Assert.Equal(anchorIdsBefore["face-0002"], scalarChanged.Anchors.Select(static anchor => anchor.Id).ToArray());

        var anchorsChanged = identities["face-0003"];
        Assert.True(anchorsChanged.UpdatedAt > seededAt);
        Assert.Equal(2, anchorsChanged.Anchors.Count);
    }

    private static async Task<Dictionary<string, int[]>> LoadAnchorIdsAsync(ServiceProvider provider)
    {
        await using var scope = provider.CreateAsyncScope();
        return await scope.ServiceProvider.GetRequiredService<CoveContext>()
            .Set<ExtAiFacesIdentityEntity>()
            .Include(identity => identity.Anchors)
            .ToDictionaryAsync(
                identity => identity.FaceKey,
                identity => identity.Anchors.Select(static anchor => anchor.Id).ToArray());
    }

    private static StoredFaceIdentity CreateIdentity(string faceKey, List<float> vector)
        => new()
        {
            FaceKey = faceKey,
            Label = faceKey,
            LifecycleStatus = StoredFaceIdentityLifecycle.Promoted,
            PromotionReason = "video-evidence",
            ObservationCount = 4,
            AssetIds = ["video-1"],
            Anchors =
            [
                new StoredFaceAnchor
                {
                    ModelKey = "face_embedding_torchexport",
                    QualityScore = 20.0,
                    Vector = vector,
                },
            ],
        };
}
