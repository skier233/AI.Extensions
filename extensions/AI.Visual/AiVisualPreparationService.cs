using AI.Extensions.Abstractions;

namespace AI.Visual;

internal sealed class AiVisualPreparationService
{
    private const string SourceKey = "ext:ai.visual";
    private const double DefaultSliceSpanSeconds = 0.5;
    private const double SectionSplitSimilarity = 0.92;
    private const double MinimumSectionDurationSeconds = 1.0;
    private const int MinimumSectionFrames = 3;

    public AiPreparedArtifactBatch Prepare(AiDispatchRequest request)
    {
        var batch = new AiPreparedArtifactBatch();
        var targets = ResolveTargets(request.Claims);
        if (targets.Count == 0)
        {
            batch.Notes.Add("No visual embedding claims were selected.");
            return batch;
        }

        if (request.Result.MediaKind == AiMediaKinds.Image)
        {
            PrepareImage(batch, request, targets);
        }
        else if (request.Result.MediaKind == AiMediaKinds.Video)
        {
            PrepareVideo(batch, request, targets);
        }
        else
        {
            batch.Notes.Add($"AI.Visual does not consume media kind '{request.Result.MediaKind}'.");
        }

        if (batch.Embeddings.Count == 0 && batch.Notes.Count == 0)
        {
            batch.Notes.Add("No visual embeddings were found for the selected claims.");
        }

        return batch;
    }

    private static void PrepareImage(AiPreparedArtifactBatch batch, AiDispatchRequest request, IReadOnlyList<VisualEmbeddingTarget> targets)
    {
        if (request.Result.AssetAnalysis is null)
        {
            batch.Notes.Add("Image analysis did not include an asset-level embedding block.");
            return;
        }

        foreach (var embedding in request.Result.AssetAnalysis.Embeddings)
        {
            if (!TryResolveTarget(embedding.ModelKey, targets, out var target))
            {
                continue;
            }

            batch.Embeddings.Add(BuildEmbedding(
                request,
                target,
                embedding,
                sectionIndex: 0,
                startSeconds: null,
                endSeconds: null,
                metadata: new Dictionary<string, string>
                {
                    ["scope"] = "asset",
                }));
        }
    }

    private static void PrepareVideo(AiPreparedArtifactBatch batch, AiDispatchRequest request, IReadOnlyList<VisualEmbeddingTarget> targets)
    {
        var sliceSpan = request.Result.FrameIntervalSeconds ?? request.Context.FrameIntervalSeconds ?? DefaultSliceSpanSeconds;
        var groupedSamples = new Dictionary<(string TargetKind, string ModelKey), List<VisualFrameSample>>();

        foreach (var frame in request.Result.Frames.OrderBy(static frame => frame.TimeSeconds ?? double.MinValue))
        {
            var timeSeconds = frame.TimeSeconds ?? ((frame.Index ?? 0) * sliceSpan);
            foreach (var embedding in frame.Analysis.Embeddings)
            {
                if (!TryResolveTarget(embedding.ModelKey, targets, out var target))
                {
                    continue;
                }

                var key = (target.Kind, embedding.ModelKey);
                if (!groupedSamples.TryGetValue(key, out var samples))
                {
                    samples = [];
                    groupedSamples[key] = samples;
                }

                samples.Add(new VisualFrameSample(timeSeconds, embedding));
            }
        }

        foreach (var ((targetKind, modelKey), samples) in groupedSamples)
        {
            if (samples.Count == 0)
            {
                continue;
            }

            var target = targets.First(t => string.Equals(t.Kind, targetKind, StringComparison.OrdinalIgnoreCase));
            var overall = BuildCentroid(samples.Select(static sample => sample.Embedding.Vector));
            if (overall is not null)
            {
                batch.Embeddings.Add(new AiPreparedEmbedding(
                    request.Context.AssetId,
                    SourceKey,
                    target.Kind,
                    target.KindFamily,
                    "Visual",
                    target.IsSemantic,
                    overall.Value.Vector,
                    overall.Value.Norm,
                    SectionIndex: 0,
                    ModelKey: modelKey,
                    Metadata: new Dictionary<string, string>
                    {
                        ["scope"] = "asset",
                        ["runId"] = request.Context.RunId,
                    }));
            }

            var sectionIndex = 1;
            foreach (var section in BuildSections(samples, sliceSpan))
            {
                var centroid = BuildCentroid(section.Select(static sample => sample.Embedding.Vector));
                if (centroid is null)
                {
                    continue;
                }

                batch.Embeddings.Add(new AiPreparedEmbedding(
                    request.Context.AssetId,
                    SourceKey,
                    target.Kind,
                    target.KindFamily,
                    "Visual",
                    target.IsSemantic,
                    centroid.Value.Vector,
                    centroid.Value.Norm,
                    SectionIndex: sectionIndex,
                    StartSeconds: section[0].TimeSeconds,
                    EndSeconds: section[^1].TimeSeconds + sliceSpan,
                    ModelKey: modelKey,
                    Metadata: new Dictionary<string, string>
                    {
                        ["scope"] = "section",
                        ["runId"] = request.Context.RunId,
                        ["frameCount"] = section.Count.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    }));
                sectionIndex++;
            }
        }
    }

    private static IReadOnlyList<IReadOnlyList<VisualFrameSample>> BuildSections(IReadOnlyList<VisualFrameSample> samples, double sliceSpan)
    {
        if (samples.Count == 0)
        {
            return [];
        }

        var sections = new List<List<VisualFrameSample>>();
        var current = new List<VisualFrameSample> { samples[0] };
        var centroid = Normalize(samples[0].Embedding.Vector);

        for (var index = 1; index < samples.Count; index++)
        {
            var sample = samples[index];
            var similarity = CosineSimilarity(centroid, Normalize(sample.Embedding.Vector));
            if (similarity >= SectionSplitSimilarity)
            {
                current.Add(sample);
                centroid = Normalize(AverageVectors(current.Select(static item => item.Embedding.Vector)));
                continue;
            }

            sections.Add(current);
            current = [sample];
            centroid = Normalize(sample.Embedding.Vector);
        }

        sections.Add(current);

        return sections
            .Where(section =>
            {
                var duration = (section[^1].TimeSeconds + sliceSpan) - section[0].TimeSeconds;
                return section.Count >= MinimumSectionFrames && duration >= MinimumSectionDurationSeconds;
            })
            .Cast<IReadOnlyList<VisualFrameSample>>()
            .ToArray();
    }

    private static AiPreparedEmbedding BuildEmbedding(
        AiDispatchRequest request,
        VisualEmbeddingTarget target,
        AiEmbeddingObservation embedding,
        int sectionIndex,
        double? startSeconds,
        double? endSeconds,
        IReadOnlyDictionary<string, string>? metadata)
    {
        return new AiPreparedEmbedding(
            request.Context.AssetId,
            SourceKey,
            target.Kind,
            target.KindFamily,
            "Visual",
            target.IsSemantic,
            embedding.Vector,
            embedding.Norm,
            SectionIndex: sectionIndex,
            StartSeconds: startSeconds,
            EndSeconds: endSeconds,
            ModelKey: embedding.ModelKey,
            Metadata: MergeMetadata(metadata, new Dictionary<string, string>
            {
                ["runId"] = request.Context.RunId,
            }));
    }

    private static IReadOnlyList<VisualEmbeddingTarget> ResolveTargets(IReadOnlyList<AiCapabilityClaim> claims)
        => claims
            .Select(claim => new VisualEmbeddingTarget(
                claim.ClaimId.Contains("semantic", StringComparison.OrdinalIgnoreCase) ? "visual.semantic.v1" : "visual.feature.v1",
                claim.ClaimId.Contains("semantic", StringComparison.OrdinalIgnoreCase) ? "semantic.v1" : "feature.v1",
                claim.ClaimId.Contains("semantic", StringComparison.OrdinalIgnoreCase),
                claim.PreferredModels ?? []))
            .Distinct()
            .ToArray();

    private static bool TryResolveTarget(string modelKey, IReadOnlyList<VisualEmbeddingTarget> targets, out VisualEmbeddingTarget target)
    {
        foreach (var candidate in targets)
        {
            if (candidate.PreferredModels.Count > 0 && candidate.PreferredModels.Any(preferred => modelKey.Contains(preferred, StringComparison.OrdinalIgnoreCase) || preferred.Contains(modelKey, StringComparison.OrdinalIgnoreCase)))
            {
                target = candidate;
                return true;
            }
        }

        var semantic = IsSemanticModelKey(modelKey);
        var resolvedTarget = targets.FirstOrDefault(candidate => candidate.IsSemantic == semantic);
        if (resolvedTarget is null)
        {
            target = default!;
            return false;
        }

        target = resolvedTarget;
        return true;
    }

    private static bool IsSemanticModelKey(string modelKey)
        => modelKey.Contains("clip", StringComparison.OrdinalIgnoreCase)
           || modelKey.Contains("meta", StringComparison.OrdinalIgnoreCase)
           || modelKey.Contains("semantic", StringComparison.OrdinalIgnoreCase);

    private static (IReadOnlyList<float> Vector, double Norm)? BuildCentroid(IEnumerable<IReadOnlyList<float>> vectors)
    {
        var averaged = AverageVectors(vectors);
        if (averaged.Count == 0)
        {
            return null;
        }

        var normalized = Normalize(averaged);
        var norm = Math.Sqrt(normalized.Sum(static value => value * value));
        return (normalized, norm);
    }

    private static IReadOnlyList<float> AverageVectors(IEnumerable<IReadOnlyList<float>> vectors)
    {
        var materialized = vectors.Where(static vector => vector.Count > 0).ToArray();
        if (materialized.Length == 0)
        {
            return [];
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

        return buffer.Select(value => (float)(value / materialized.Length)).ToArray();
    }

    private static IReadOnlyList<float> Normalize(IReadOnlyList<float> vector)
    {
        if (vector.Count == 0)
        {
            return [];
        }

        var norm = Math.Sqrt(vector.Sum(static value => value * value));
        if (norm <= 0)
        {
            return vector.ToArray();
        }

        return vector.Select(value => (float)(value / norm)).ToArray();
    }

    private static double CosineSimilarity(IReadOnlyList<float> left, IReadOnlyList<float> right)
    {
        if (left.Count == 0 || right.Count == 0 || left.Count != right.Count)
        {
            return 0.0;
        }

        double dot = 0.0;
        for (var index = 0; index < left.Count; index++)
        {
            dot += left[index] * right[index];
        }

        return dot;
    }

    private static IReadOnlyDictionary<string, string>? MergeMetadata(IReadOnlyDictionary<string, string>? left, IReadOnlyDictionary<string, string>? right)
    {
        if (left is null && right is null)
        {
            return null;
        }

        var merged = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (left is not null)
        {
            foreach (var (key, value) in left)
            {
                merged[key] = value;
            }
        }

        if (right is not null)
        {
            foreach (var (key, value) in right)
            {
                merged[key] = value;
            }
        }

        return merged;
    }

    private sealed record VisualEmbeddingTarget(string Kind, string KindFamily, bool IsSemantic, IReadOnlyList<string> PreferredModels);

    private sealed record VisualFrameSample(double TimeSeconds, AiEmbeddingObservation Embedding);
}