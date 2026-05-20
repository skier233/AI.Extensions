using AI.Extensions.Abstractions;

namespace AI.Extensions.Tests;

internal static class AiTestData
{
    public static AiDispatchRequest CreateRequest(
        string mediaKind,
        IReadOnlyList<AiCapabilityClaim> claims,
        AiAnalyzeResult result,
        string assetId = "asset-1")
    {
        return new AiDispatchRequest(
            new AiRunContext("run-1", mediaKind, assetId, assetId, null, null, result.DurationSeconds, result.FrameIntervalSeconds),
            claims,
            result);
    }
}
