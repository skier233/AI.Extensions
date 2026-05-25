using AI.Extensions.Abstractions;

using AI.Audio;

using System.Text.Json;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiAudioPreparationServiceTests
{
    [Fact]
    public void Prepare_BuildsWindowEmbeddingsAndClassificationSegments()
    {
        var service = new AiAudioPreparationService();
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("audio.asset.embedding", "Audio Embeddings", AiMediaKinds.Audio, "embedding", "asset", "embeddings"),
            new AiCapabilityClaim("audio.asset.classification", "Audio Classification", AiMediaKinds.Audio, "classification", "asset", "categories"),
        ];
        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Audio,
            AssetId = "audio-1",
            Windows =
            [
                Window(0, 0.0, 3.0, [1f, 0f], "speech", 0.9),
                Window(1, 3.0, 6.0, [0.98f, 0.02f], "speech", 0.85),
            ],
        };

        var batch = service.Prepare(AiTestData.CreateRequest(AiMediaKinds.Audio, claims, result, "audio-1"));

        Assert.Equal(3, batch.Embeddings.Count);
        Assert.Single(batch.Segments);
        Assert.Equal(0.0, batch.Segments[0].StartSeconds);
        Assert.Equal(6.0, batch.Segments[0].EndSeconds);

        static AiTemporalSlice Window(int index, double start, double end, IReadOnlyList<float> vector, string label, double confidence)
            => new(
                "window",
                index,
                null,
                start,
                end,
                new AiAnalysisNode
                {
                    Embeddings = [new AiEmbeddingObservation("ecapa", "asset", vector, 1.0)],
                    Classifications = [new AiClassificationPrediction("ast", label, confidence)],
                });
    }

        [Fact]
        public void Prepare_BuildsEmbeddingsFromAudioOtherPayload()
        {
                var payload = JsonDocument.Parse("""
                        {
                            "asset_id": "audio-1",
                            "windows": [
                                {
                                    "index": 0,
                                    "start": 0.0,
                                    "end": 4.0,
                                    "analysis": {
                                        "other": {
                                            "audio_embeddings_audioembed": [
                                                { "vector": [0.1, 0.2, 0.3], "norm": 1.0, "dim": 3, "embedder": "audioembed" }
                                            ]
                                        }
                                    }
                                }
                            ]
                        }
                        """).RootElement;
                var result = AiAnalyzeResultParser.Parse(AiMediaKinds.Audio, payload);
                var service = new AiAudioPreparationService();

                var batch = service.Prepare(AiTestData.CreateRequest(
                        AiMediaKinds.Audio,
                        [new AiCapabilityClaim("audio.asset.embedding", "Audio Embeddings", AiMediaKinds.Audio, "embedding", "asset", "embeddings")],
                        result,
                        "audio-1"));

                Assert.Equal(2, batch.Embeddings.Count);
                Assert.Contains(batch.Embeddings, embedding => embedding.SectionIndex == 1 && embedding.Vector.Count == 3);
                Assert.Contains(batch.Embeddings, embedding => embedding.SectionIndex == 0 && embedding.Vector.Count == 3);
        }
}
