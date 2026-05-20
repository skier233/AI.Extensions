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
            PreferredModels: ["dinov3_base"]);
        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "scene-visual",
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

        var batch = service.Prepare(AiTestData.CreateRequest(AiMediaKinds.Video, [claim], result, "scene-visual"));

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
                    Embeddings = [new AiEmbeddingObservation("dinov3_base", "frame", vector, 1.0)],
                });
    }
}
