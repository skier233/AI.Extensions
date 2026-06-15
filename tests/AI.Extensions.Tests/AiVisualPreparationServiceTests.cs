using AI.Extensions.Abstractions;

using AI.Visual;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiVisualPreparationServiceTests
{
    [Fact]
    public void Prepare_CreatesOverallAndSectionEmbeddingsForDistinctVideoSections()
    {
        var service = new AiVisualPreparationService();
        var claim = new AiCapabilityClaim(
            "visual.video.feature",
            "Video Feature Embeddings",
            AiMediaKinds.Video,
            "embedding",
            "frame",
            "frames",
            PreferredModels: ["visual"]);
        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "video-visual",
            FrameIntervalSeconds = 1.0,
            Frames =
            [
                Frame(0, 0.0, [1f, 0f]),
                Frame(1, 1.0, [0.98f, 0.02f]),
                Frame(2, 2.0, [0.99f, 0.01f]),
                Frame(3, 3.0, [0f, 1f]),
                Frame(4, 4.0, [0.02f, 0.98f]),
                Frame(5, 5.0, [0.01f, 0.99f]),
            ],
        };

        var batch = service.Prepare(AiTestData.CreateRequest(AiMediaKinds.Video, [claim], result, "video-visual"));

        Assert.Equal(3, batch.Embeddings.Count);
        Assert.Contains(batch.Embeddings, embedding => embedding.SectionIndex == 0);
        Assert.Contains(batch.Embeddings, embedding => embedding.SectionIndex == 1 && embedding.StartSeconds == 0.0 && embedding.EndSeconds == 3.0);
        Assert.Contains(batch.Embeddings, embedding => embedding.SectionIndex == 2 && embedding.StartSeconds == 3.0 && embedding.EndSeconds == 6.0);

        static AiTemporalSlice Frame(int index, double timeSeconds, IReadOnlyList<float> vector)
            => new(
                "frame",
                index,
                timeSeconds,
                null,
                null,
                new AiAnalysisNode
                {
                    Embeddings = [new AiEmbeddingObservation("visual", "frame", vector, 1.0)],
                });
    }

    [Fact]
    public void Prepare_RoutesSemanticAndFeatureCategoriesToDistinctKinds()
    {
        // Both the feature ("visual_embeddings_visual") and semantic ("visual_embeddings_semvisual")
        // models run on the same frames. The semantic category also contains the substring "visual",
        // so a loose match would misroute it to the feature target. Each must land on its own kind.
        var service = new AiVisualPreparationService();
        var featureClaim = new AiCapabilityClaim(
            "visual.video.feature", "Video Feature Embeddings",
            AiMediaKinds.Video, "embedding", "frame", "frames",
            PreferredModels: ["visual"]);
        var semanticClaim = new AiCapabilityClaim(
            "visual.video.semantic", "Video Semantic Embeddings",
            AiMediaKinds.Video, "embedding", "frame", "frames",
            PreferredModels: ["semvisual"]);

        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "video-both",
            FrameIntervalSeconds = 1.0,
            Frames =
            [
                Frame(0, 0.0),
                Frame(1, 1.0),
                Frame(2, 2.0),
            ],
        };

        var batch = service.Prepare(AiTestData.CreateRequest(AiMediaKinds.Video, [featureClaim, semanticClaim], result, "video-both"));

        Assert.Contains(batch.Embeddings, e => e.Kind == "visual.feature.v1" && e.KindFamily == "feature.v1" && !e.IsSemantic);
        Assert.Contains(batch.Embeddings, e => e.Kind == "visual.semantic.v1" && e.KindFamily == "semantic.v1" && e.IsSemantic);
        Assert.All(
            batch.Embeddings.Where(e => e.IsSemantic),
            e => Assert.Equal("visual_embeddings_semvisual", e.ModelKey));
        Assert.All(
            batch.Embeddings.Where(e => !e.IsSemantic),
            e => Assert.Equal("visual_embeddings_visual", e.ModelKey));

        static AiTemporalSlice Frame(int index, double timeSeconds)
            => new(
                "frame",
                index,
                timeSeconds,
                null,
                null,
                new AiAnalysisNode
                {
                    Embeddings =
                    [
                        new AiEmbeddingObservation("visual_embeddings_visual", "frame", [1f, 0f], 1.0),
                        new AiEmbeddingObservation("visual_embeddings_semvisual", "frame", [0f, 1f], 1.0),
                    ],
                });
    }
}
