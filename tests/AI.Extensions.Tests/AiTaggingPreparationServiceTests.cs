using AI.Extensions.Abstractions;

using AI.Tagging;

using System.Text.Json;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiTaggingPreparationServiceTests
{
    [Fact]
    public void Prepare_SplitsVideoSegmentsAcrossSilentGap()
    {
        var service = new AiTaggingPreparationService();
        var claim = new AiCapabilityClaim("tagging.video.frame", "Video Tags", AiMediaKinds.Video, "tagging", "frame", "frames");
        var result = new AiAnalyzeResult
        {
            MediaKind = AiMediaKinds.Video,
            AssetId = "video-1",
            FrameIntervalSeconds = 1.0,
            Frames =
            [
                new AiTemporalSlice("frame", 0, 0.0, null, null, new AiAnalysisNode { Tags = [new AiTagPrediction("nsfw_v3", "tag-a", 0.9)] }),
                new AiTemporalSlice("frame", 1, 1.0, null, null, new AiAnalysisNode { Tags = [new AiTagPrediction("nsfw_v3", "tag-a", 0.8)] }),
                new AiTemporalSlice("frame", 2, 2.0, null, null, new AiAnalysisNode()),
                new AiTemporalSlice("frame", 3, 3.0, null, null, new AiAnalysisNode { Tags = [new AiTagPrediction("nsfw_v3", "tag-a", 0.95)] }),
            ],
        };

        var batch = service.Prepare(AiTestData.CreateRequest(AiMediaKinds.Video, [claim], result, "video-1"));

        Assert.Equal(2, batch.Segments.Count);
        Assert.Equal(0.0, batch.Segments[0].StartSeconds);
        Assert.Equal(2.0, batch.Segments[0].EndSeconds);
        Assert.Equal(3.0, batch.Segments[1].StartSeconds);
        Assert.Equal(4.0, batch.Segments[1].EndSeconds);
    }

        [Fact]
        public void Prepare_UsesLegacyFrameIntervalFieldForSingleFrameSegments()
        {
                var service = new AiTaggingPreparationService();
                var claim = new AiCapabilityClaim("tagging.video.frame", "Video Tags", AiMediaKinds.Video, "tagging", "frame", "frames");
                using var document = JsonDocument.Parse(
                        """
                        {
                            "asset_id": "video-legacy",
                            "frame_interval": 2.0,
                            "frames": [
                                {
                                    "time_seconds": 10.0,
                                    "analysis": {
                                        "capabilities": {
                                            "tagging": {
                                                "bodyparts": [
                                                    ["tag-a", 0.9]
                                                ]
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                        """);

                var result = AiAnalyzeResultParser.Parse(AiMediaKinds.Video, document.RootElement);

                var batch = service.Prepare(AiTestData.CreateRequest(AiMediaKinds.Video, [claim], result, "video-legacy"));

                var segment = Assert.Single(batch.Segments);
                Assert.Equal(10.0, segment.StartSeconds);
                Assert.Equal(12.0, segment.EndSeconds);
        }
}
