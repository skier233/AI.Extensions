using AI.Extensions.Abstractions;

using System.Text.Json;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiAnalyzeResultParserTests
{
    private const string VideoWithAssetAnalysis = """
        {
          "asset_id": "video-1",
          "frame_interval_seconds": 1.0,
          "frames": [
            {
              "index": 0,
              "time_seconds": 0.0,
              "analysis": {
                "capabilities": {
                  "tagging": {
                    "nsfw_v3": [["tag-a", 0.9]]
                  }
                }
              }
            }
          ],
          "analysis": {
            "other": {
              "shot_boundaries": {
                "model": "shot_model",
                "fps": 30,
                "boundaries": [
                  { "start": 0.0, "end": 4.5, "transition": "cut" },
                  { "start": 4.5, "end": 9.0, "transition": null }
                ]
              }
            }
          }
        }
        """;

    [Fact]
    public void Parse_ReadsAssetLevelAnalysisForVideo()
    {
        using var document = JsonDocument.Parse(VideoWithAssetAnalysis);

        var result = AiAnalyzeResultParser.Parse(AiMediaKinds.Video, document.RootElement);

        Assert.NotNull(result.AssetAnalysis);
        Assert.Empty(result.AssetAnalysis.Embeddings);
        using var boundaries = JsonDocument.Parse(result.AssetAnalysis.Other["shot_boundaries"]);
        Assert.Equal("shot_model", boundaries.RootElement.GetProperty("model").GetString());
        Assert.Equal(2, boundaries.RootElement.GetProperty("boundaries").GetArrayLength());

        var frame = Assert.Single(result.Frames);
        var tag = Assert.Single(frame.Analysis.Tags);
        Assert.Equal("tag-a", tag.Tag);
    }

    [Fact]
    public void Parse_DoesNotTreatVectorlessAssetLevelArrayAsEmbeddingsForVideo()
    {
        using var document = JsonDocument.Parse("""
            {
              "asset_id": "video-1",
              "frames": [],
              "analysis": {
                "other": {
                  "shot_boundaries": [
                    { "start": 0.0, "end": 4.5 },
                    { "start": 4.5, "end": 9.0 }
                  ]
                }
              }
            }
            """);

        var result = AiAnalyzeResultParser.Parse(AiMediaKinds.Video, document.RootElement);

        Assert.NotNull(result.AssetAnalysis);
        Assert.Empty(result.AssetAnalysis.Embeddings);
        using var boundaries = JsonDocument.Parse(result.AssetAnalysis.Other["shot_boundaries"]);
        Assert.Equal(2, boundaries.RootElement.GetArrayLength());
    }

    [Fact]
    public void Parse_ReturnsEmptyAssetAnalysisForVideoWithoutOne()
    {
        using var document = JsonDocument.Parse("""
            {
              "asset_id": "video-1",
              "frames": [
                { "index": 0, "time_seconds": 0.0, "analysis": {} }
              ]
            }
            """);

        var result = AiAnalyzeResultParser.Parse(AiMediaKinds.Video, document.RootElement);

        Assert.NotNull(result.AssetAnalysis);
        Assert.Empty(result.AssetAnalysis.Tags);
        Assert.Empty(result.AssetAnalysis.Embeddings);
        Assert.Empty(result.AssetAnalysis.Other);
        Assert.Single(result.Frames);
    }

    [Theory]
    [InlineData("{{ \"result\": {0} }}")]
    [InlineData("{{ \"result\": [{0}] }}")]
    public void Parse_ReadsAssetLevelAnalysisForVideoInsideResultWrapper(string wrapper)
    {
        using var document = JsonDocument.Parse(string.Format(wrapper, VideoWithAssetAnalysis));

        var result = AiAnalyzeResultParser.Parse(AiMediaKinds.Video, document.RootElement);

        Assert.Equal("video-1", result.AssetId);
        Assert.NotNull(result.AssetAnalysis);
        Assert.True(result.AssetAnalysis.Other.ContainsKey("shot_boundaries"));
        Assert.Single(result.Frames);
    }
}
