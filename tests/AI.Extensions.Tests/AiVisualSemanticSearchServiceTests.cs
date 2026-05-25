using AI.Visual;

using Cove.Core.Entities;
using Cove.Core.Enums;
using Cove.Core.Interfaces;
using Cove.Data;
using Cove.Data.Repositories;
using Cove.Data.Services;

using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;

using Pgvector;

using System.Text;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiVisualSemanticSearchServiceTests
{
    [Fact]
    public async Task SearchAsync_ReturnsDistinctSceneMatchesOrderedByVisualDistance()
    {
        await using var provider = CreateProvider();
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var nearScene = new Scene { Title = "Blue Room" };
            var farScene = new Scene { Title = "Red Hall" };
            db.Scenes.AddRange(nearScene, farScene);
            await db.SaveChangesAsync();

            db.Embeddings.AddRange(
                CreateVisualEmbedding(EmbeddingHostType.Scene, nearScene.Id, [0.8f, 0.2f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, nearScene.Id, [1f, 0f], sectionIndex: 1, startSec: 12.0, endSec: 18.0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, farScene.Id, [0f, 1f], sectionIndex: 0));
            await db.SaveChangesAsync();
        }

        await using var searchScope = provider.CreateAsyncScope();
        var service = searchScope.ServiceProvider.GetRequiredService<AiVisualSemanticSearchService>();

        var response = await service.SearchScenesAsync(new AiVisualSemanticSearchRequest<SceneFilter>
        {
            FindFilter = new FindFilter { Q = "blue room", Page = 1, PerPage = 10 },
        });
        var results = response.Items;

        Assert.Equal(2, results.Count);
        Assert.Equal(2, response.TotalCount);
        Assert.Equal("Blue Room", results[0].Title);
        Assert.Equal("Red Hall", results[1].Title);
    }

    [Fact]
    public async Task SearchAsync_ReturnsImageMatchesWithThumbnailUrl()
    {
        await using var provider = CreateProvider();
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var image = new Image { Title = "Window Light" };
            db.Images.Add(image);
            await db.SaveChangesAsync();

            db.Embeddings.Add(CreateVisualEmbedding(EmbeddingHostType.Image, image.Id, [1f, 0f], sectionIndex: 0));
            await db.SaveChangesAsync();
        }

        await using var searchScope = provider.CreateAsyncScope();
        var service = searchScope.ServiceProvider.GetRequiredService<AiVisualSemanticSearchService>();

        var response = await service.SearchImagesAsync(new AiVisualSemanticSearchRequest<ImageFilter>
        {
            FindFilter = new FindFilter { Q = "window light", Page = 1, PerPage = 10 },
        });
        var result = Assert.Single(response.Items);

        Assert.Equal(1, response.TotalCount);
        Assert.Equal("Window Light", result.Title);
    }

    [Fact]
    public async Task SearchAsync_TrimsWeakVisualCandidatesBeyondInitialResults()
    {
        await using var provider = CreateProvider();
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var scenes = Enumerable.Range(1, 6).Select(index => new Scene { Title = $"Candidate {index}" }).ToArray();
            db.Scenes.AddRange(scenes);
            await db.SaveChangesAsync();

            db.Embeddings.AddRange(
                CreateVisualEmbedding(EmbeddingHostType.Scene, scenes[0].Id, [1f, 0f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, scenes[1].Id, [0.999f, 0.001f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, scenes[2].Id, [0.998f, 0.002f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, scenes[3].Id, [0.997f, 0.003f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, scenes[4].Id, [0.996f, 0.004f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, scenes[5].Id, [0f, 1f], sectionIndex: 0));
            await db.SaveChangesAsync();
        }

        await using var searchScope = provider.CreateAsyncScope();
        var service = searchScope.ServiceProvider.GetRequiredService<AiVisualSemanticSearchService>();

        var response = await service.SearchScenesAsync(new AiVisualSemanticSearchRequest<SceneFilter>
        {
            FindFilter = new FindFilter { Q = "blue room", Page = 1, PerPage = 10 },
        });

        Assert.Equal(5, response.TotalCount);
        Assert.DoesNotContain(response.Items, scene => scene.Title == "Candidate 6");
    }

    [Fact]
    public async Task SearchAsync_AllowsNormalSortWithinVisualCandidates()
    {
        await using var provider = CreateProvider();
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var closeScene = new Scene { Title = "Zulu" };
            var laterScene = new Scene { Title = "Alpha" };
            db.Scenes.AddRange(closeScene, laterScene);
            await db.SaveChangesAsync();

            db.Embeddings.AddRange(
                CreateVisualEmbedding(EmbeddingHostType.Scene, closeScene.Id, [1f, 0f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, laterScene.Id, [0.9f, 0.1f], sectionIndex: 0));
            await db.SaveChangesAsync();
        }

        await using var searchScope = provider.CreateAsyncScope();
        var service = searchScope.ServiceProvider.GetRequiredService<AiVisualSemanticSearchService>();

        var visualResponse = await service.SearchScenesAsync(new AiVisualSemanticSearchRequest<SceneFilter>
        {
            FindFilter = new FindFilter { Q = "blue room", Page = 1, PerPage = 10, Sort = "visual_match", Direction = SortDirection.Desc },
        });
        var titleResponse = await service.SearchScenesAsync(new AiVisualSemanticSearchRequest<SceneFilter>
        {
            FindFilter = new FindFilter { Q = "blue room", Page = 1, PerPage = 10, Sort = "title" },
        });

        Assert.Equal("Zulu", visualResponse.Items[0].Title);
        Assert.Equal("Alpha", titleResponse.Items[0].Title);
    }

    [Fact]
    public async Task SearchAsync_ReturnsEmptyWhenTextEncoderTimesOutAndLocalEncoderUnavailable()
    {
        await using var provider = CreateProvider(new SlowTextEncoder());
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var seedImage = new Image { Title = "Window Light" };
            var nearImage = new Image { Title = "Near Match" };
            var farImage = new Image { Title = "Far Match" };
            db.Images.AddRange(seedImage, nearImage, farImage);
            await db.SaveChangesAsync();

            db.Embeddings.AddRange(
                CreateVisualEmbedding(EmbeddingHostType.Image, seedImage.Id, [1f, 0f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Image, nearImage.Id, [0.98f, 0.02f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Image, farImage.Id, [0f, 1f], sectionIndex: 0));
            await db.SaveChangesAsync();
        }

        await using var searchScope = provider.CreateAsyncScope();
        var service = searchScope.ServiceProvider.GetRequiredService<AiVisualSemanticSearchService>();

        var response = await service.SearchImagesAsync(new AiVisualSemanticSearchRequest<ImageFilter>
        {
            FindFilter = new FindFilter { Q = "window", Page = 1, PerPage = 10 },
        });

        Assert.Empty(response.Items);
        Assert.Equal(0, response.TotalCount);
    }

    [Fact]
    public async Task SearchAsync_ReturnsEmptyWhenTextEncoderFailsAndNoLocalSeedsExist()
    {
        await using var provider = CreateProvider(new ThrowingTextEncoder());
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var image = new Image { Title = "Unrelated" };
            db.Images.Add(image);
            await db.SaveChangesAsync();

            db.Embeddings.Add(CreateVisualEmbedding(EmbeddingHostType.Image, image.Id, [1f, 0f], sectionIndex: 0));
            await db.SaveChangesAsync();
        }

        await using var searchScope = provider.CreateAsyncScope();
        var service = searchScope.ServiceProvider.GetRequiredService<AiVisualSemanticSearchService>();

        var response = await service.SearchImagesAsync(new AiVisualSemanticSearchRequest<ImageFilter>
        {
            FindFilter = new FindFilter { Q = "missing", Page = 1, PerPage = 10 },
        });

        Assert.Empty(response.Items);
        Assert.Equal(0, response.TotalCount);
    }

    [Fact]
    public async Task SimilarScenesForSceneAsync_BlendsFeatureAndSemanticButLeansFeature()
    {
        await using var provider = CreateProvider();
        int sourceId;
        int featureMatchId;
        int semanticMatchId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var source = new Scene { Title = "Source" };
            var featureMatch = new Scene { Title = "Feature Match" };
            var semanticMatch = new Scene { Title = "Semantic Match" };
            db.Scenes.AddRange(source, featureMatch, semanticMatch);
            await db.SaveChangesAsync();
            sourceId = source.Id;
            featureMatchId = featureMatch.Id;
            semanticMatchId = semanticMatch.Id;

            db.Embeddings.AddRange(
                CreateVisualEmbedding(EmbeddingHostType.Scene, sourceId, [1f, 0f], sectionIndex: 0, isSemantic: false),
                CreateVisualEmbedding(EmbeddingHostType.Scene, sourceId, [1f, 0f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, featureMatchId, [0.99f, 0.01f], sectionIndex: 0, isSemantic: false),
                CreateVisualEmbedding(EmbeddingHostType.Scene, featureMatchId, [0f, 1f], sectionIndex: 0),
                CreateVisualEmbedding(EmbeddingHostType.Scene, semanticMatchId, [0f, 1f], sectionIndex: 0, isSemantic: false),
                CreateVisualEmbedding(EmbeddingHostType.Scene, semanticMatchId, [1f, 0f], sectionIndex: 0));
            await db.SaveChangesAsync();
        }

        await using var searchScope = provider.CreateAsyncScope();
        var service = searchScope.ServiceProvider.GetRequiredService<AiVisualSemanticSearchService>();

        var response = await service.SimilarScenesForSceneAsync(sourceId, page: 1, perPage: 10);

        Assert.Equal(2, response.TotalCount);
        Assert.Equal("Feature Match", response.Items[0].Scene.Title);
        Assert.Equal("Semantic Match", response.Items[1].Scene.Title);
    }

    [Fact]
    public async Task SimilarScenesForSceneSegmentAsync_UsesAllIntervalsForUnionQuery()
    {
        await using var provider = CreateProvider();
        int sourceId;
        await using (var scope = provider.CreateAsyncScope())
        {
            var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
            var source = new Scene { Title = "Union Source" };
            var secondIntervalMatch = new Scene { Title = "Second Interval Match" };
            db.Scenes.AddRange(source, secondIntervalMatch);
            await db.SaveChangesAsync();
            sourceId = source.Id;

            db.Embeddings.AddRange(
                CreateVisualEmbedding(EmbeddingHostType.Scene, source.Id, [1f, 0f], sectionIndex: 1, startSec: 0, endSec: 10, isSemantic: false),
                CreateVisualEmbedding(EmbeddingHostType.Scene, source.Id, [0f, 1f], sectionIndex: 2, startSec: 40, endSec: 50, isSemantic: false),
                CreateVisualEmbedding(EmbeddingHostType.Scene, secondIntervalMatch.Id, [0f, 1f], sectionIndex: 1, startSec: 4, endSec: 12, isSemantic: false));
            await db.SaveChangesAsync();
        }

        await using var searchScope = provider.CreateAsyncScope();
        var service = searchScope.ServiceProvider.GetRequiredService<AiVisualSemanticSearchService>();

        var response = await service.SimilarScenesForSceneSegmentAsync(
            sourceId,
            [new AiVisualSegmentInterval(0, 10), new AiVisualSegmentInterval(40, 50)],
            page: 1,
            perPage: 10);
        var result = Assert.Single(response.Items);

        Assert.Equal("Second Interval Match", result.Scene.Title);
        Assert.True(result.Distance < 0.5f);
    }

    [Fact]
    public async Task ReadAsync_AcceptsCamelCaseFilterCriterionModifiers()
    {
        var context = new DefaultHttpContext();
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(
            """
            {
              "findFilter": { "q": "couch", "page": 1, "perPage": 10 },
              "objectFilter": { "performersCriterion": { "value": [1779], "modifier": "includes" } }
            }
            """));

        var request = await AiVisualSearchRequestReader.ReadAsync<SceneFilter>(context.Request, CancellationToken.None);

        Assert.Equal("couch", request.FindFilter?.Q);
        Assert.NotNull(request.ObjectFilter?.PerformersCriterion);
        Assert.Equal(CriterionModifier.Includes, request.ObjectFilter.PerformersCriterion.Modifier);
        Assert.Equal([1779], request.ObjectFilter.PerformersCriterion.Value);
    }

    private static Embedding CreateVisualEmbedding(
        EmbeddingHostType hostType,
        int hostId,
        float[] vector,
        int sectionIndex,
        double? startSec = null,
        double? endSec = null,
        bool isSemantic = true)
        => new()
        {
            HostType = hostType,
            HostId = hostId,
            Kind = isSemantic ? "visual.semantic.v1" : "visual.feature.v1",
            KindFamily = isSemantic ? "semantic.v1" : "feature.v1",
            Modality = EmbeddingModality.Visual,
            IsSemantic = isSemantic,
            Dim = vector.Length,
            Vector = new Vector(vector),
            SectionIndex = sectionIndex,
            StartSec = startSec,
            EndSec = endSec,
            SourceKey = "ext:ai.visual",
            SourceRunId = "run-visual",
        };

    private static ServiceProvider CreateProvider(ITextEncoder? textEncoder = null)
    {
        var services = new ServiceCollection();
        var databaseName = $"ai-visual-semantic-search-{Guid.NewGuid():N}";
        var databaseRoot = new InMemoryDatabaseRoot();
        services.AddDbContext<CoveContext>(options => options.UseInMemoryDatabase(databaseName, databaseRoot));
        services.AddScoped<ISceneRepository, SceneRepository>();
        services.AddScoped<IImageRepository, ImageRepository>();
        services.AddScoped<ITextEncoder>(_ => textEncoder ?? new FakeTextEncoder());
        services.AddScoped<EmbeddingService>();
        services.AddScoped<IEmbeddingService>(static services => services.GetRequiredService<EmbeddingService>());
        services.AddScoped<ITextEncoderRegistry>(static services => services.GetRequiredService<EmbeddingService>());
        services.AddSingleton<AiVisualLocalTextEncoder>();
        services.AddScoped<AiVisualSemanticSearchService>();
        return services.BuildServiceProvider();
    }

    private sealed class FakeTextEncoder : ITextEncoder
    {
        public string KindFamily => "semantic.v1";

        public Task<Vector> EncodeAsync(string text, CancellationToken cancellationToken = default)
            => Task.FromResult(new Vector(new[] { 1f, 0f }));
    }

    private sealed class SlowTextEncoder : ITextEncoder
    {
        public string KindFamily => "semantic.v1";

        public async Task<Vector> EncodeAsync(string text, CancellationToken cancellationToken = default)
        {
            await Task.Delay(Timeout.InfiniteTimeSpan, cancellationToken);
            return new Vector(new[] { 0f, 1f });
        }
    }

    private sealed class ThrowingTextEncoder : ITextEncoder
    {
        public string KindFamily => "semantic.v1";

        public Task<Vector> EncodeAsync(string text, CancellationToken cancellationToken = default)
            => throw new InvalidOperationException("The fast encoder is unavailable.");
    }
}