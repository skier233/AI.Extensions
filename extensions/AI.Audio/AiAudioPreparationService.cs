using AI.Extensions.Abstractions;

namespace AI.Audio;

internal sealed class AiAudioPreparationService
{
    private const string SourceKey = "ext:ai.audio";
    private const double DefaultClassificationFloor = 0.45;
    private const double DefaultWindowSpanSeconds = 3.0;

    public AiPreparedArtifactBatch Prepare(AiDispatchRequest request)
    {
        var batch = new AiPreparedArtifactBatch();
        if (request.Result.MediaKind != AiMediaKinds.Audio)
        {
            batch.Notes.Add($"AI.Audio does not consume media kind '{request.Result.MediaKind}'.");
            return batch;
        }

        PrepareEmbeddings(batch, request);
        PrepareClassificationSegments(batch, request);

        if (batch.Embeddings.Count == 0 && batch.Segments.Count == 0 && batch.Notes.Count == 0)
        {
            batch.Notes.Add("No audio embeddings or classification windows were found.");
        }

        return batch;
    }

    private static void PrepareEmbeddings(AiPreparedArtifactBatch batch, AiDispatchRequest request)
    {
        var byModel = new Dictionary<string, List<IReadOnlyList<float>>>(StringComparer.OrdinalIgnoreCase);

        foreach (var window in request.Result.Windows.OrderBy(static window => window.StartSeconds ?? double.MinValue).ThenBy(static window => window.Index ?? int.MinValue))
        {
            foreach (var embedding in window.Analysis.Embeddings)
            {
                if (!byModel.TryGetValue(embedding.ModelKey, out var vectors))
                {
                    vectors = [];
                    byModel[embedding.ModelKey] = vectors;
                }

                vectors.Add(embedding.Vector);
                batch.Embeddings.Add(new AiPreparedEmbedding(
                    request.Context.AssetId,
                    SourceKey,
                    "audio.embed.v1",
                    "audio.v1",
                    "Audio",
                    false,
                    embedding.Vector,
                    embedding.Norm,
                    SectionIndex: window.Index ?? 0,
                    StartSeconds: window.StartSeconds,
                    EndSeconds: window.EndSeconds,
                    ModelKey: embedding.ModelKey,
                    Metadata: new Dictionary<string, string>
                    {
                        ["scope"] = "window",
                        ["runId"] = request.Context.RunId,
                    }));
            }
        }

        foreach (var (modelKey, vectors) in byModel)
        {
            var centroid = BuildCentroid(vectors);
            if (centroid is null)
            {
                continue;
            }

            batch.Embeddings.Add(new AiPreparedEmbedding(
                request.Context.AssetId,
                SourceKey,
                "audio.embed.v1",
                "audio.v1",
                "Audio",
                false,
                centroid.Value.Vector,
                centroid.Value.Norm,
                SectionIndex: 0,
                ModelKey: modelKey,
                Metadata: new Dictionary<string, string>
                {
                    ["scope"] = "asset",
                    ["runId"] = request.Context.RunId,
                }));
        }
    }

    private static void PrepareClassificationSegments(AiPreparedArtifactBatch batch, AiDispatchRequest request)
    {
        var activeRuns = new Dictionary<AudioRunKey, ActiveAudioRun>();

        foreach (var window in request.Result.Windows.OrderBy(static window => window.StartSeconds ?? double.MinValue).ThenBy(static window => window.Index ?? int.MinValue))
        {
            var windowStart = window.StartSeconds ?? ((window.Index ?? 0) * DefaultWindowSpanSeconds);
            var windowEnd = window.EndSeconds ?? (windowStart + DefaultWindowSpanSeconds);
            var visible = new HashSet<AudioRunKey>();

            foreach (var prediction in window.Analysis.Classifications)
            {
                if (prediction.Confidence is { } confidence && confidence < DefaultClassificationFloor)
                {
                    continue;
                }

                var key = new AudioRunKey(prediction.ModelKey, prediction.Label);
                visible.Add(key);

                if (!activeRuns.TryGetValue(key, out var activeRun))
                {
                    activeRuns[key] = new ActiveAudioRun(windowStart, windowEnd, prediction.Confidence, 1);
                    continue;
                }

                activeRuns[key] = activeRun with
                {
                    EndSeconds = windowEnd,
                    PeakConfidence = MaxConfidence(activeRun.PeakConfidence, prediction.Confidence),
                    ObservationCount = activeRun.ObservationCount + 1,
                };
            }

            var keysToClose = activeRuns.Keys.Where(key => !visible.Contains(key)).ToArray();
            foreach (var key in keysToClose)
            {
                AddSegment(batch, request, key, activeRuns[key]);
                activeRuns.Remove(key);
            }
        }

        foreach (var (key, run) in activeRuns)
        {
            AddSegment(batch, request, key, run);
        }
    }

    private static void AddSegment(AiPreparedArtifactBatch batch, AiDispatchRequest request, AudioRunKey key, ActiveAudioRun run)
    {
        batch.Segments.Add(new AiPreparedSegment(
            request.Context.AssetId,
            SourceKey,
            Kind: "audio-classification",
            StartSeconds: run.StartSeconds,
            EndSeconds: run.EndSeconds,
            TagName: key.Label,
            Title: key.Label,
            Confidence: run.PeakConfidence,
            Metadata: new Dictionary<string, string>
            {
                ["modelKey"] = key.ModelKey,
                ["runId"] = request.Context.RunId,
                ["observationCount"] = run.ObservationCount.ToString(System.Globalization.CultureInfo.InvariantCulture),
            }));
    }

    private static (IReadOnlyList<float> Vector, double Norm)? BuildCentroid(IEnumerable<IReadOnlyList<float>> vectors)
    {
        var materialized = vectors.Where(static vector => vector.Count > 0).ToArray();
        if (materialized.Length == 0)
        {
            return null;
        }

        var length = materialized[0].Count;
        var buffer = new double[length];
        foreach (var vector in materialized)
        {
            for (var index = 0; index < length; index++)
            {
                buffer[index] += vector[index];
            }
        }

        var averaged = buffer.Select(value => (float)(value / materialized.Length)).ToArray();
        var norm = Math.Sqrt(averaged.Sum(static value => value * value));
        if (norm > 0)
        {
            for (var index = 0; index < averaged.Length; index++)
            {
                averaged[index] = (float)(averaged[index] / norm);
            }
        }

        return (averaged, norm);
    }

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

    private readonly record struct AudioRunKey(string ModelKey, string Label);

    private readonly record struct ActiveAudioRun(double StartSeconds, double EndSeconds, double? PeakConfidence, int ObservationCount);
}