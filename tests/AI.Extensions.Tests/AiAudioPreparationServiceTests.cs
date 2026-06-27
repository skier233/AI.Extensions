using AI.Extensions.Abstractions;

using AI.Audio;

using System.Text.Json;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiAudioPreparationServiceTests
{
    [Fact]
    public void Prepare_BuildsWindowEmbeddingsAndVoiceCentroid()
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

        // Two window embeddings + one asset centroid. No user-visible segments are produced anymore.
        Assert.Equal(3, batch.Embeddings.Count);
        Assert.Empty(batch.Segments);

        var windows = batch.Embeddings.Where(e => e.SectionIndex > 0).ToArray();
        Assert.Equal(2, windows.Length);
        Assert.All(windows, w => Assert.Equal("speech", w.Metadata!["soundType"]));
        Assert.All(windows, w => Assert.Equal("true", w.Metadata!["voice"]));

        var asset = Assert.Single(batch.Embeddings, e => e.SectionIndex == 0);
        Assert.Equal("voice", asset.Metadata!["basis"]);
        Assert.Equal("2", asset.Metadata!["voiceWindowCount"]);
        Assert.Equal("2", asset.Metadata!["windowCount"]);
    }

    [Fact]
    public void Prepare_ExcludesNonVoiceWindowsFromAssetCentroid()
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
                Window(0, 0.0, 3.0, [1f, 0f], "moan", 0.9),
                Window(1, 3.0, 6.0, [0f, 1f], "music", 0.95),
            ],
        };

        var batch = service.Prepare(AiTestData.CreateRequest(AiMediaKinds.Audio, claims, result, "audio-1"));

        var asset = Assert.Single(batch.Embeddings, e => e.SectionIndex == 0);
        Assert.Equal("voice", asset.Metadata!["basis"]);
        Assert.Equal("1", asset.Metadata!["voiceWindowCount"]);
        Assert.Equal("2", asset.Metadata!["windowCount"]);

        // The centroid is the (normalized) moan window only, so it points along the first axis.
        Assert.True(asset.Vector[0] > asset.Vector[1]);

        var music = Assert.Single(batch.Embeddings, e => e.SectionIndex > 0 && e.Metadata!["soundType"] == "music");
        Assert.Equal("false", music.Metadata!["voice"]);
    }

    [Fact]
    public void Prepare_FallsBackToAllWindowsWhenNoVoiceClassification()
    {
        var service = new AiAudioPreparationService();
        IReadOnlyList<AiCapabilityClaim> claims =
        [
            new AiCapabilityClaim("audio.asset.embedding", "Audio Embeddings", AiMediaKinds.Audio, "embedding", "asset", "embeddings"),
        ];
        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Audio,
            AssetId = "audio-1",
            Windows =
            [
                WindowNoClassification(0, 0.0, 3.0, [1f, 0f]),
                WindowNoClassification(1, 3.0, 6.0, [0f, 1f]),
            ],
        };

        var batch = service.Prepare(AiTestData.CreateRequest(AiMediaKinds.Audio, claims, result, "audio-1"));

        var asset = Assert.Single(batch.Embeddings, e => e.SectionIndex == 0);
        Assert.Equal("all", asset.Metadata!["basis"]);
        Assert.Equal("0", asset.Metadata!["voiceWindowCount"]);
        Assert.Equal("2", asset.Metadata!["windowCount"]);
        Assert.All(batch.Embeddings.Where(e => e.SectionIndex > 0), w => Assert.Equal("unknown", w.Metadata!["soundType"]));
    }

    [Fact]
    public void Parse_ParsesNestedProbabilityMapsIntoDistinctWindowLabels()
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

        Assert.Equal(3, result.Windows.Count);
        Assert.Equal("speech", DominantLabel(result.Windows[0]));
        Assert.Equal("speech", DominantLabel(result.Windows[1]));
        Assert.Equal("music", DominantLabel(result.Windows[2]));
    }

    [Fact]
    public void Parse_BinsRawAudioClassProbabilitiesIntoWindowLabel()
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

        var window = Assert.Single(result.Windows);
        Assert.Equal("speech", DominantLabel(window));
    }

    [Fact]
    public void Parse_ParsesAudioSummaryDominantTypeIntoWindowLabel()
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

        var window = Assert.Single(result.Windows);
        Assert.Equal("speech", DominantLabel(window));
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

    private static string? DominantLabel(AiTemporalSlice window)
        => window.Analysis.Classifications
            .OrderByDescending(static c => c.Confidence ?? 1d)
            .Select(static c => c.Label)
            .FirstOrDefault();

    private static AiTemporalSlice Window(int index, double start, double end, IReadOnlyList<float> vector, string label, double confidence)
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

    private static AiTemporalSlice WindowNoClassification(int index, double start, double end, IReadOnlyList<float> vector)
        => new(
            "window",
            index,
            null,
            start,
            end,
            new AiAnalysisNode
            {
                Embeddings = [new AiEmbeddingObservation("ecapa", "asset", vector, 1.0)],
            });
}
