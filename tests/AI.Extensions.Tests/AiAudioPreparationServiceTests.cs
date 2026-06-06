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
        Assert.Equal(2, batch.Segments.Count);
        Assert.Equal(0.0, batch.Segments[0].StartSeconds);
        Assert.Equal(3.0, batch.Segments[0].EndSeconds);
        Assert.Equal(3.0, batch.Segments[1].StartSeconds);
        Assert.Equal(6.0, batch.Segments[1].EndSeconds);

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
    public void Prepare_ParsesNestedProbabilityMapsIntoDistinctClassificationRuns()
    {
        var payload = JsonDocument.Parse("""
            {
              "asset_id": "audio-1",
              "windows": [
                {
                  "index": 0,
                  "start": 0.0,
                  "end": 3.0,
                  "analysis": {
                    "capabilities": {
                      "classification": {
                        "ast": [
                          { "probabilities": { "speech": 0.91, "music": 0.09 } }
                        ]
                      }
                    }
                  }
                },
                {
                  "index": 1,
                  "start": 3.0,
                  "end": 6.0,
                  "analysis": {
                    "capabilities": {
                      "classification": {
                        "ast": [
                          { "probabilities": { "speech": 0.88, "music": 0.12 } }
                        ]
                      }
                    }
                  }
                },
                {
                  "index": 2,
                  "start": 6.0,
                  "end": 9.0,
                  "analysis": {
                    "capabilities": {
                      "classification": {
                        "ast": [
                          { "probabilities": { "speech": 0.11, "music": 0.89 } }
                        ]
                      }
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
            [new AiCapabilityClaim("audio.asset.classification", "Audio Classification", AiMediaKinds.Audio, "classification", "asset", "categories")],
            result,
            "audio-1"));

        Assert.Equal(3, batch.Segments.Count);
        Assert.Equal("speech", batch.Segments[0].Title);
        Assert.Equal(0.0, batch.Segments[0].StartSeconds);
        Assert.Equal(3.0, batch.Segments[0].EndSeconds);
        Assert.Equal("speech", batch.Segments[1].Title);
        Assert.Equal(3.0, batch.Segments[1].StartSeconds);
        Assert.Equal(6.0, batch.Segments[1].EndSeconds);
        Assert.Equal("music", batch.Segments[2].Title);
        Assert.Equal(6.0, batch.Segments[2].StartSeconds);
        Assert.Equal(9.0, batch.Segments[2].EndSeconds);
    }

    [Fact]
    public void Prepare_BinsRawAudioClassProbabilitiesIntoWindowSegment()
    {
        var payload = JsonDocument.Parse("""
            {
              "asset_id": "audio-1",
              "windows": [
                {
                  "index": 0,
                  "start": 0.0,
                  "end": 3.0,
                  "analysis": {
                    "capabilities": {
                      "classification": {
                        "audio_classification_audioclass": [
                          {
                            "probabilities": [0.0477, 0.001, 0.002],
                            "top5": [[0, 0.0477], [2, 0.002]],
                            "num_classes": 527,
                            "classifier": "audioclass"
                          }
                        ]
                      }
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
            [new AiCapabilityClaim("audio.asset.classification", "Audio Classification", AiMediaKinds.Audio, "classification", "asset", "categories")],
            result,
            "audio-1"));

        var segment = Assert.Single(batch.Segments);
        Assert.Equal("speech", segment.Title);
        Assert.Null(segment.Confidence);
        Assert.Equal(0.0, segment.StartSeconds);
        Assert.Equal(3.0, segment.EndSeconds);
    }

    [Fact]
    public void Prepare_ParsesAudioSummaryDominantTypeIntoWindowSegment()
    {
        var payload = JsonDocument.Parse("""
            {
              "asset_id": "audio-1",
              "windows": [
                {
                  "index": 0,
                  "start": 0.0,
                  "end": 3.0,
                  "dominant_type": "speech",
                  "scores": { "moan": 0.005, "speech": 0.09, "breath": 0.002 }
                }
              ]
            }
            """).RootElement;

        var result = AiAnalyzeResultParser.Parse(AiMediaKinds.Audio, payload);
        var service = new AiAudioPreparationService();

        var batch = service.Prepare(AiTestData.CreateRequest(
            AiMediaKinds.Audio,
            [new AiCapabilityClaim("audio.asset.classification", "Audio Classification", AiMediaKinds.Audio, "classification", "asset", "categories")],
            result,
            "audio-1"));

        var segment = Assert.Single(batch.Segments);
        Assert.Equal("speech", segment.Title);
        Assert.Null(segment.Confidence);
        Assert.Equal(0.0, segment.StartSeconds);
        Assert.Equal(3.0, segment.EndSeconds);
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
