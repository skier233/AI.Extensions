using AI.Extensions.Abstractions;

namespace AI.Tagging;

internal sealed class AiTaggingPreparationService
{
    private const string SourceKey = "ext:ai.tagging";
    private const double DefaultConfidenceFloor = 0.35;
    private const double DefaultSliceSpanSeconds = 0.5;

    public AiPreparedArtifactBatch Prepare(AiDispatchRequest request)
    {
        var batch = new AiPreparedArtifactBatch();
        if (request.Result.MediaKind == AiMediaKinds.Image)
        {
            PrepareImage(batch, request);
        }
        else if (request.Result.MediaKind == AiMediaKinds.Video)
        {
            PrepareVideo(batch, request);
        }
        else
        {
            batch.Notes.Add($"AI.Tagging does not consume media kind '{request.Result.MediaKind}'.");
        }

        if (batch.TagLinks.Count == 0 && batch.Segments.Count == 0 && batch.Notes.Count == 0)
        {
            batch.Notes.Add("No tagging predictions met the ingest floor.");
        }

        return batch;
    }

    private static void PrepareImage(AiPreparedArtifactBatch batch, AiDispatchRequest request)
    {
        if (request.Result.AssetAnalysis is null)
        {
            batch.Notes.Add("Image analysis did not include an asset-level tagging block.");
            return;
        }

        foreach (var prediction in request.Result.AssetAnalysis.Tags)
        {
            if (!ShouldKeep(prediction.Confidence))
            {
                continue;
            }

            batch.TagLinks.Add(new AiPreparedTagLink(
                request.Context.AssetId,
                SourceKey,
                prediction.Tag,
                prediction.Confidence,
                prediction.ModelKey,
                request.Context.MediaKind,
                new Dictionary<string, string>
                {
                    ["runId"] = request.Context.RunId,
                }));
        }
    }

    private static void PrepareVideo(AiPreparedArtifactBatch batch, AiDispatchRequest request)
    {
        if (request.Result.Frames.Count == 0)
        {
            batch.Notes.Add("Video analysis did not include any frame tagging data.");
            return;
        }

        var sliceSpan = request.Result.FrameIntervalSeconds ?? request.Context.FrameIntervalSeconds ?? DefaultSliceSpanSeconds;
        var activeRuns = new Dictionary<TagRunKey, ActiveTagRun>();
        var orderedFrames = request.Result.Frames
            .OrderBy(static frame => frame.TimeSeconds ?? double.MinValue)
            .ThenBy(static frame => frame.Index ?? int.MinValue)
            .ToArray();

        foreach (var frame in orderedFrames)
        {
            var frameTime = frame.TimeSeconds ?? ((frame.Index ?? 0) * sliceSpan);
            var visibleKeys = new HashSet<TagRunKey>();

            foreach (var prediction in frame.Analysis.Tags)
            {
                if (!ShouldKeep(prediction.Confidence))
                {
                    continue;
                }

                var key = new TagRunKey(prediction.ModelKey, prediction.Tag);
                visibleKeys.Add(key);
                if (!activeRuns.TryGetValue(key, out var activeRun))
                {
                    activeRuns[key] = new ActiveTagRun(frameTime, frameTime, prediction.Confidence, 1);
                    continue;
                }

                activeRuns[key] = activeRun with
                {
                    LastSeenSeconds = frameTime,
                    PeakConfidence = MaxConfidence(activeRun.PeakConfidence, prediction.Confidence),
                    ObservationCount = activeRun.ObservationCount + 1,
                };
            }

            var keysToClose = activeRuns.Keys.Where(key => !visibleKeys.Contains(key)).ToArray();
            foreach (var key in keysToClose)
            {
                AddSegment(batch, request, key, activeRuns[key], sliceSpan);
                activeRuns.Remove(key);
            }
        }

        foreach (var (key, run) in activeRuns)
        {
            AddSegment(batch, request, key, run, sliceSpan);
        }
    }

    private static void AddSegment(AiPreparedArtifactBatch batch, AiDispatchRequest request, TagRunKey key, ActiveTagRun run, double sliceSpan)
    {
        var endSeconds = run.LastSeenSeconds + sliceSpan;
        if (endSeconds <= run.StartSeconds)
        {
            endSeconds = run.StartSeconds + sliceSpan;
        }

        batch.Segments.Add(new AiPreparedSegment(
            request.Context.AssetId,
            SourceKey,
            Kind: "tag",
            StartSeconds: run.StartSeconds,
            EndSeconds: endSeconds,
            TagName: key.Tag,
            Title: key.Tag,
            Confidence: run.PeakConfidence,
            Metadata: new Dictionary<string, string>
            {
                ["modelKey"] = key.ModelKey,
                ["runId"] = request.Context.RunId,
                ["observationCount"] = run.ObservationCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            }));
    }

    private static bool ShouldKeep(double? confidence)
        => confidence is null || confidence.Value >= DefaultConfidenceFloor;

    private static double? MaxConfidence(double? left, double? right)
    {
        if (left is null)
        {
            return right;
        }

        if (right is null)
        {
            return left;
        }

        return Math.Max(left.Value, right.Value);
    }

    private readonly record struct TagRunKey(string ModelKey, string Tag);

    private readonly record struct ActiveTagRun(double StartSeconds, double LastSeenSeconds, double? PeakConfidence, int ObservationCount);
}