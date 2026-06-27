using System.Globalization;

using AI.Extensions.Abstractions;

namespace AI.Audio;

internal sealed class AiAudioPreparationService
{
    private const string SourceKey = "ext:ai.audio";

    // Minimum classifier confidence for a window's sound-type label to be trusted when deciding
    // whether the window is voice-bearing. Mirrors the floor the legacy segment path used.
    private const double VoiceClassificationFloor = 0.45;

    // Sound types that carry the performer's voice/vocalizations. ECAPA-TDNN speaker embeddings
    // computed over these windows characterize the person; music/silence/breath windows mostly add
    // noise to the speaker centroid, so they are excluded from it.
    private static readonly HashSet<string> VoiceSoundTypes = new(StringComparer.OrdinalIgnoreCase)
    {
        "moan",
        "speech",
    };

    public AiPreparedArtifactBatch Prepare(AiDispatchRequest request)
    {
        var batch = new AiPreparedArtifactBatch();
        if (request.Result.MediaKind != AiMediaKinds.Audio)
        {
            batch.Notes.Add($"AI.Audio does not consume media kind '{request.Result.MediaKind}'.");
            return batch;
        }

        PrepareEmbeddings(batch, request);

        if (batch.Embeddings.Count == 0 && batch.Notes.Count == 0)
        {
            batch.Notes.Add("No audio embeddings were found.");
        }

        return batch;
    }

    private static void PrepareEmbeddings(AiPreparedArtifactBatch batch, AiDispatchRequest request)
    {
        // Per embedding-model accumulation. `voiceByModel` feeds the asset-level speaker centroid;
        // `allByModel` is the fallback for assets that have no confidently voiced windows so we never
        // drop the asset embedding entirely (e.g. when the classifier was not run).
        var voiceByModel = new Dictionary<string, List<IReadOnlyList<float>>>(StringComparer.OrdinalIgnoreCase);
        var allByModel = new Dictionary<string, List<IReadOnlyList<float>>>(StringComparer.OrdinalIgnoreCase);

        foreach (var window in request.Result.Windows
            .OrderBy(static window => window.StartSeconds ?? double.MinValue)
            .ThenBy(static window => window.Index ?? int.MinValue))
        {
            var soundType = ResolveWindowSoundType(window);
            var isVoice = soundType is not null && VoiceSoundTypes.Contains(soundType);

            foreach (var embedding in window.Analysis.Embeddings)
            {
                Accumulate(allByModel, embedding.ModelKey, embedding.Vector);
                if (isVoice)
                {
                    Accumulate(voiceByModel, embedding.ModelKey, embedding.Vector);
                }

                batch.Embeddings.Add(new AiPreparedEmbedding(
                    request.Context.AssetId,
                    SourceKey,
                    "audio.embed.v1",
                    "audio.v1",
                    "Audio",
                    false,
                    embedding.Vector,
                    embedding.Norm,
                    SectionIndex: window.Index.HasValue ? window.Index.Value + 1 : 1,
                    StartSeconds: window.StartSeconds,
                    EndSeconds: window.EndSeconds,
                    ModelKey: embedding.ModelKey,
                    Metadata: new Dictionary<string, string>
                    {
                        ["scope"] = "window",
                        ["runId"] = request.Context.RunId,
                        // Persist the classifier verdict on the embedding itself so voice filtering and
                        // future per-type aggregation can run without re-processing the audio.
                        ["soundType"] = soundType ?? "unknown",
                        ["voice"] = isVoice ? "true" : "false",
                    }));
            }
        }

        foreach (var modelKey in allByModel.Keys)
        {
            var hasVoice = voiceByModel.TryGetValue(modelKey, out var voiceVectors) && voiceVectors.Count > 0;
            var sourceVectors = hasVoice ? voiceVectors! : allByModel[modelKey];
            var centroid = BuildCentroid(sourceVectors);
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
                    // "voice" = centroid over voice-bearing windows only; "all" = fallback over every
                    // window because no confidently voiced windows existed for this model.
                    ["basis"] = hasVoice ? "voice" : "all",
                    ["voiceWindowCount"] = (hasVoice ? voiceVectors!.Count : 0).ToString(CultureInfo.InvariantCulture),
                    ["windowCount"] = allByModel[modelKey].Count.ToString(CultureInfo.InvariantCulture),
                }));
        }
    }

    private static void Accumulate(Dictionary<string, List<IReadOnlyList<float>>> byModel, string modelKey, IReadOnlyList<float> vector)
    {
        if (vector.Count == 0)
        {
            return;
        }

        if (!byModel.TryGetValue(modelKey, out var vectors))
        {
            vectors = [];
            byModel[modelKey] = vectors;
        }

        vectors.Add(vector);
    }

    // Highest-confidence classifier label for a window, or null when the window has no label clearing
    // the floor. Labels without a confidence score are treated as confident (the classifier emitted
    // them deliberately) to match the legacy segment behavior.
    private static string? ResolveWindowSoundType(AiTemporalSlice window)
    {
        string? best = null;
        var bestRank = double.NegativeInfinity;

        foreach (var prediction in window.Analysis.Classifications)
        {
            if (prediction.Confidence is { } confidence && confidence < VoiceClassificationFloor)
            {
                continue;
            }

            var rank = prediction.Confidence ?? 1d;
            if (rank > bestRank)
            {
                bestRank = rank;
                best = prediction.Label?.Trim().ToLowerInvariant();
            }
        }

        return string.IsNullOrWhiteSpace(best) ? null : best;
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
}
