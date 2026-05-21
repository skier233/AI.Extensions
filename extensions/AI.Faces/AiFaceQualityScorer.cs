using AI.Extensions.Abstractions;

namespace AI.Faces;

internal static class AiFaceQualityScorer
{
    public const string PoseQualityMetadataKey = "pose_quality";

    public const string ImageQualityMetadataKey = "image_quality";

    private const string SharpnessMetadataKey = "sharpness";
    private const string BrightnessMetadataKey = "brightness";
    private const string BrightnessQualityMetadataKey = "brightness_quality";
    private const string OcclusionMetadataKey = "occlusion";
    private const string OcclusionQualityMetadataKey = "occlusion_quality";
    private const double MaximumAreaEvidenceBoost = 24.0;

    public static FaceFrameSample SelectBestCoverSample(IReadOnlyList<FaceFrameSample> samples)
    {
        if (samples.Count == 0)
        {
            throw new ArgumentException("At least one face sample is required.", nameof(samples));
        }

        var centroid = BuildCentroid(samples);
        var candidates = samples.Where(IsPlausibleCoverSample).ToArray();
        var rankedSamples = candidates.Length > 0 ? candidates : samples;
        return rankedSamples
            .OrderByDescending(sample => ScoreCoverCandidate(sample, centroid))
            .First();
    }

    public static double ScoreCoverQuality(IReadOnlyList<FaceFrameSample> samples, FaceFrameSample sample)
        => ScoreCoverCandidate(sample, BuildCentroid(samples));

    public static double ScoreIdentityEvidence(IReadOnlyList<FaceFrameSample> samples)
        => samples.Select(ScoreIdentityEvidenceSample).DefaultIfEmpty(0.0).Max();

    public static double GetMetadataQuality(AiEmbeddingObservation embedding, string key)
    {
        if (embedding.Metadata is null)
        {
            return 1.0;
        }

        foreach (var (metadataKey, rawValue) in embedding.Metadata)
        {
            if (!string.Equals(metadataKey, key, StringComparison.OrdinalIgnoreCase) || string.IsNullOrWhiteSpace(rawValue))
            {
                continue;
            }

            if (double.TryParse(rawValue, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var parsed))
            {
                return Math.Clamp(parsed, 0.0, 1.0);
            }
        }

        return 1.0;
    }

    private static double ScoreIdentityEvidenceSample(FaceFrameSample sample)
    {
        var bestEmbeddingNorm = sample.Embeddings
            .Select(static embedding => embedding.Norm ?? 0.0)
            .DefaultIfEmpty(0.0)
            .Max();
        var bestPoseQuality = sample.Embeddings
            .Select(static embedding => GetMetadataQuality(embedding, PoseQualityMetadataKey))
            .DefaultIfEmpty(1.0)
            .Max();
        var bestImageQuality = sample.Embeddings
            .Select(static embedding => GetMetadataQuality(embedding, ImageQualityMetadataKey))
            .DefaultIfEmpty(1.0)
            .Max();
        var areaScore = Math.Min(MaximumAreaEvidenceBoost, Math.Sqrt(Math.Max(1.0, ScaleArea(sample.Detection.BoundingBox.Area))));
        return sample.Detection.Score * Math.Max(1.0, bestEmbeddingNorm) * bestPoseQuality * bestImageQuality * areaScore;
    }

    private static double ScoreCoverCandidate(FaceFrameSample sample, IReadOnlyList<float>? centroid)
    {
        var bestEmbeddingNorm = sample.Embeddings
            .Select(static embedding => embedding.Norm ?? 0.0)
            .DefaultIfEmpty(1.0)
            .Max();
        var poseQuality = sample.Embeddings
            .Select(static embedding => GetMetadataQuality(embedding, PoseQualityMetadataKey))
            .DefaultIfEmpty(1.0)
            .Max();
        var imageQuality = sample.Embeddings
            .Select(static embedding => GetMetadataQuality(embedding, ImageQualityMetadataKey))
            .DefaultIfEmpty(1.0)
            .Max();
        var sharpnessQuality = sample.Embeddings
            .Select(static embedding => GetMetadataQuality(embedding, SharpnessMetadataKey))
            .DefaultIfEmpty(1.0)
            .Max();
        var brightnessQuality = sample.Embeddings
            .Select(GetBrightnessQuality)
            .DefaultIfEmpty(1.0)
            .Max();
        var occlusionQuality = sample.Embeddings
            .Select(GetOcclusionQuality)
            .DefaultIfEmpty(1.0)
            .Max();
        var areaScore = Math.Clamp(Math.Sqrt(Math.Max(1.0, ScaleArea(sample.Detection.BoundingBox.Area))) / 10.0, 0.35, 2.25);
        var representativeScore = centroid is null ? 1.0 : Math.Clamp(0.55 + (0.45 * BestSimilarity(sample, centroid)), 0.55, 1.0);
        var normScore = Math.Clamp(Math.Log(Math.Max(1.0, bestEmbeddingNorm), 32.0), 0.35, 1.15);

        return sample.Detection.Score
               * areaScore
               * poseQuality
               * imageQuality
               * sharpnessQuality
               * brightnessQuality
               * occlusionQuality
               * representativeScore
               * normScore;
    }

    private static bool IsPlausibleCoverSample(FaceFrameSample sample)
    {
        var boundingBox = sample.Detection.BoundingBox;
        var width = Math.Abs(boundingBox.X2 - boundingBox.X1);
        var height = Math.Abs(boundingBox.Y2 - boundingBox.Y1);
        if (width <= 0.0 || height <= 0.0)
        {
            return false;
        }

        var aspectRatio = width / height;
        return aspectRatio is >= 0.45 and <= 1.8;
    }

    private static double GetBrightnessQuality(AiEmbeddingObservation embedding)
    {
        var directQuality = GetMetadataQuality(embedding, BrightnessQualityMetadataKey);
        if (directQuality < 1.0)
        {
            return directQuality;
        }

        if (embedding.Metadata is null || !TryReadMetadataDouble(embedding.Metadata, BrightnessMetadataKey, out var brightness))
        {
            return 1.0;
        }

        return Math.Clamp(1.0 - (Math.Abs(brightness - 0.55) / 0.55), 0.25, 1.0);
    }

    private static double GetOcclusionQuality(AiEmbeddingObservation embedding)
    {
        var directQuality = GetMetadataQuality(embedding, OcclusionQualityMetadataKey);
        if (directQuality < 1.0)
        {
            return directQuality;
        }

        if (embedding.Metadata is null || !TryReadMetadataDouble(embedding.Metadata, OcclusionMetadataKey, out var occlusion))
        {
            return 1.0;
        }

        return Math.Clamp(1.0 - occlusion, 0.15, 1.0);
    }

    private static bool TryReadMetadataDouble(IReadOnlyDictionary<string, string> metadata, string key, out double value)
    {
        foreach (var (metadataKey, rawValue) in metadata)
        {
            if (string.Equals(metadataKey, key, StringComparison.OrdinalIgnoreCase)
                && double.TryParse(rawValue, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out value))
            {
                value = Math.Clamp(value, 0.0, 1.0);
                return true;
            }
        }

        value = 0.0;
        return false;
    }

    private static double ScaleArea(double area)
        => area <= 1.0 ? area * 10000.0 : area;

    private static IReadOnlyList<float>? BuildCentroid(IReadOnlyList<FaceFrameSample> samples)
    {
        var vectors = samples
            .SelectMany(sample => sample.Embeddings.Select(embedding => new
            {
                embedding.Vector,
                Weight = Math.Max(1e-6, sample.Detection.Score * Math.Max(1.0, embedding.Norm ?? 1.0)),
            }))
            .Where(static item => item.Vector.Count > 0)
            .ToArray();
        if (vectors.Length == 0)
        {
            return null;
        }

        var dimension = vectors[0].Vector.Count;
        vectors = vectors.Where(item => item.Vector.Count == dimension).ToArray();
        if (vectors.Length == 0)
        {
            return null;
        }

        var buffer = new double[dimension];
        var totalWeight = 0.0;
        foreach (var vector in vectors)
        {
            totalWeight += vector.Weight;
            for (var index = 0; index < dimension; index++)
            {
                buffer[index] += vector.Vector[index] * vector.Weight;
            }
        }

        if (totalWeight <= 0.0)
        {
            return null;
        }

        var centroid = buffer.Select(value => (float)(value / totalWeight)).ToArray();
        var norm = Math.Sqrt(centroid.Sum(static value => value * value));
        if (norm <= 0.0)
        {
            return null;
        }

        for (var index = 0; index < centroid.Length; index++)
        {
            centroid[index] = (float)(centroid[index] / norm);
        }

        return centroid;
    }

    private static double BestSimilarity(FaceFrameSample sample, IReadOnlyList<float> centroid)
        => sample.Embeddings
            .Select(embedding => CosineSimilarity(embedding.Vector, centroid))
            .DefaultIfEmpty(0.0)
            .Max();

    private static double CosineSimilarity(IReadOnlyList<float> left, IReadOnlyList<float> right)
    {
        if (left.Count == 0 || right.Count == 0 || left.Count != right.Count)
        {
            return 0.0;
        }

        double dot = 0.0;
        double leftNorm = 0.0;
        double rightNorm = 0.0;
        for (var index = 0; index < left.Count; index++)
        {
            dot += left[index] * right[index];
            leftNorm += left[index] * left[index];
            rightNorm += right[index] * right[index];
        }

        if (leftNorm <= 0.0 || rightNorm <= 0.0)
        {
            return 0.0;
        }

        return Math.Clamp(dot / (Math.Sqrt(leftNorm) * Math.Sqrt(rightNorm)), 0.0, 1.0);
    }
}