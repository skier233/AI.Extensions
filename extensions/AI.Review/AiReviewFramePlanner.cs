using System.Text.Json;

namespace AI.Review;

// Picks which frames of a segment to show and export. The AI server sampled the video every
// `interval` seconds and AI.Tagging closes a run at lastSeen + interval, so the frames the model
// actually judged sit at start, start + interval, ... The planner prefers exactly those frames,
// because a false positive is most useful for training when it is the very frame the model got wrong.
internal static class AiReviewFramePlanner
{
    public static IReadOnlyList<double> Plan(double startSec, double? endSec, int observationCount, int maxFrames)
    {
        maxFrames = Math.Clamp(maxFrames, 1, 3);
        var start = Math.Max(0, startSec);
        var end = endSec is { } e && e > start ? e : start;
        var duration = end - start;

        var wanted = duration < 3 ? 1 : duration < 10 ? 2 : 3;
        wanted = Math.Min(wanted, maxFrames);

        if (observationCount > 0 && duration > 0)
        {
            var interval = duration / observationCount;
            var count = Math.Min(wanted, observationCount);
            var indices = PickIndices(observationCount, count);
            return indices.Select(index => Round(start + (index * interval))).ToArray();
        }

        if (duration <= 0)
        {
            return [Round(start)];
        }

        return wanted switch
        {
            1 => [Round(start + (duration * 0.5))],
            2 => [Round(start + (duration / 3)), Round(start + (duration * 2 / 3))],
            _ => [Round(start + (duration * 0.25)), Round(start + (duration * 0.5)), Round(start + (duration * 0.75))],
        };
    }

    // First / middle / last observation, collapsing duplicates for short runs.
    internal static IReadOnlyList<int> PickIndices(int observationCount, int count)
    {
        if (observationCount <= 0)
        {
            return [0];
        }

        var last = observationCount - 1;
        var middle = last / 2;
        IEnumerable<int> picks = count switch
        {
            <= 1 => [middle],
            2 => [0, last],
            _ => [0, middle, last],
        };
        return picks.Distinct().OrderBy(static index => index).ToArray();
    }

    public static int ReadObservationCount(JsonDocument? payload)
    {
        if (payload is null || payload.RootElement.ValueKind != JsonValueKind.Object)
        {
            return 0;
        }

        if (!payload.RootElement.TryGetProperty("observationCount", out var element))
        {
            return 0;
        }

        return element.ValueKind switch
        {
            JsonValueKind.Number when element.TryGetInt32(out var number) => Math.Max(0, number),
            JsonValueKind.String when int.TryParse(element.GetString(), out var parsed) => Math.Max(0, parsed),
            _ => 0,
        };
    }

    public static string? ReadModelKey(JsonDocument? payload)
    {
        if (payload is null || payload.RootElement.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        return payload.RootElement.TryGetProperty("modelKey", out var element) && element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
    }

    private static double Round(double seconds) => Math.Round(seconds, 3);
}
