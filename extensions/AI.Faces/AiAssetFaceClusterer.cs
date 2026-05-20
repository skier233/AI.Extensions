using AI.Extensions.Abstractions;

namespace AI.Faces;

internal sealed class AiAssetFaceClusterer
{
    private const double MinimumClusterEmbeddingNorm = 10.0;
    private const double MinimumClusterDetectionScore = 0.3;

    public IReadOnlyList<PreparedFaceTrack> Cluster(IReadOnlyList<PreparedFaceTrack> tracks, AiFacesSettings settings)
        => ClusterWithDiagnostics(tracks, settings).Tracks;

    public AiAssetFaceClusterResult ClusterWithDiagnostics(IReadOnlyList<PreparedFaceTrack> tracks, AiFacesSettings settings)
    {
        if (tracks.Count <= 1)
        {
            return new AiAssetFaceClusterResult(
                tracks,
                new AiAssetFaceClusterDiagnostics(tracks.Count, tracks.Count, 0, 0, 0, 0));
        }

        var clusters = new List<AssetFaceCluster>();
        var mergedTracks = 0;
        var rejectedByConcurrency = 0;
        var rejectedByThreshold = 0;
        var rejectedByAmbiguity = 0;
        foreach (var track in tracks.OrderByDescending(static track => track.TrackQuality))
        {
            var trackDescriptor = FaceClusterDescriptor.FromSamples(track.Samples);
            var hadConcurrencyConflict = false;
            var rankedClusters = clusters
                .Select(cluster =>
                {
                    var hasConcurrencyConflict = HasConcurrencyConflict(cluster, track);
                    hadConcurrencyConflict |= hasConcurrencyConflict;
                    return new
                    {
                        Cluster = cluster,
                        Score = trackDescriptor is null || cluster.Descriptor is null || hasConcurrencyConflict
                            ? 0.0
                            : CosineSimilarity(trackDescriptor.Centroid, cluster.Descriptor.Centroid),
                    };
                })
                .Where(static candidate => candidate.Score > 0.0)
                .OrderByDescending(static candidate => candidate.Score)
                .ToArray();

            var best = rankedClusters.FirstOrDefault();
            var secondBestScore = rankedClusters.Length > 1 ? rankedClusters[1].Score : 0.0;
            if (best is not null
                && best.Score >= settings.AssetClusterSimilarityThreshold
                && (best.Score - secondBestScore) >= settings.AssetClusterAmbiguityMargin)
            {
                best.Cluster.Add(track);
                mergedTracks++;
                continue;
            }

            if (best is null && hadConcurrencyConflict)
            {
                rejectedByConcurrency++;
            }
            else if (best is not null && best.Score < settings.AssetClusterSimilarityThreshold)
            {
                rejectedByThreshold++;
            }
            else if (best is not null && (best.Score - secondBestScore) < settings.AssetClusterAmbiguityMargin)
            {
                rejectedByAmbiguity++;
            }

            clusters.Add(new AssetFaceCluster(track));
        }

        var outputTracks = clusters
            .OrderBy(static cluster => cluster.FirstFrameOrder)
            .ThenByDescending(static cluster => cluster.TrackQuality)
            .Select((cluster, index) => cluster.ToTrack($"asset-face-{index + 1}"))
            .ToArray();
        return new AiAssetFaceClusterResult(
            outputTracks,
            new AiAssetFaceClusterDiagnostics(
                tracks.Count,
                outputTracks.Length,
                mergedTracks,
                rejectedByConcurrency,
                rejectedByThreshold,
                rejectedByAmbiguity));
    }

    private static bool HasConcurrencyConflict(AssetFaceCluster cluster, PreparedFaceTrack track)
    {
        var existingFrames = cluster.Tracks
            .SelectMany(static existingTrack => existingTrack.Samples)
            .Select(static sample => sample.FrameOrder)
            .ToHashSet();

        return track.Samples.Any(sample => existingFrames.Contains(sample.FrameOrder));
    }

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

        if (leftNorm <= 0 || rightNorm <= 0)
        {
            return 0.0;
        }

        return dot / (Math.Sqrt(leftNorm) * Math.Sqrt(rightNorm));
    }

    private sealed class AssetFaceCluster
    {
        private readonly List<PreparedFaceTrack> _tracks;

        public AssetFaceCluster(PreparedFaceTrack track)
        {
            _tracks = [track];
            Descriptor = FaceClusterDescriptor.FromSamples(track.Samples);
        }

        public IReadOnlyList<PreparedFaceTrack> Tracks => _tracks;

        public FaceClusterDescriptor? Descriptor { get; private set; }

        public int FirstFrameOrder => _tracks
            .SelectMany(static track => track.Samples)
            .Select(static sample => sample.FrameOrder)
            .DefaultIfEmpty(0)
            .Min();

        public double TrackQuality => _tracks.Select(static track => track.TrackQuality).DefaultIfEmpty(0.0).Max();

        public void Add(PreparedFaceTrack track)
        {
            _tracks.Add(track);
            Descriptor = FaceClusterDescriptor.FromSamples(_tracks.SelectMany(static item => item.Samples));
        }

        public PreparedFaceTrack ToTrack(string trackKey)
        {
            var samples = _tracks
                .SelectMany(static track => track.Samples)
                .OrderBy(static sample => sample.TimeSeconds ?? double.MinValue)
                .ThenBy(static sample => sample.FrameOrder)
                .ToArray();
            var bestSample = AiFaceQualityScorer.SelectBestCoverSample(samples);
            var frameIntervalSeconds = _tracks
                .Select(static track => track.FrameIntervalSeconds)
                .FirstOrDefault(static value => value.HasValue);

            return new PreparedFaceTrack(
                trackKey,
                samples,
                samples.Select(static sample => sample.TimeSeconds).Where(static value => value.HasValue).Min(),
                samples.Select(static sample => sample.TimeSeconds).Where(static value => value.HasValue).Max(),
                frameIntervalSeconds,
                TrackQuality,
                bestSample);
        }
    }

    private sealed record FaceClusterDescriptor(IReadOnlyList<float> Centroid)
    {
        public static FaceClusterDescriptor? FromSamples(IEnumerable<FaceFrameSample> sourceSamples)
        {
            var weightedVectors = sourceSamples
                .SelectMany(sample => sample.Embeddings.Select(embedding => new
                {
                    embedding.Vector,
                    Norm = embedding.Norm ?? 0.0,
                    Weight = Math.Max(1e-6, sample.Detection.Score * Math.Max(1.0, embedding.Norm ?? 0.0)),
                    sample.Detection.Score,
                }))
                .Where(static item => item.Vector.Count > 0
                    && item.Norm >= MinimumClusterEmbeddingNorm
                    && item.Score >= MinimumClusterDetectionScore)
                .ToArray();

            if (weightedVectors.Length == 0)
            {
                return null;
            }

            var dimension = weightedVectors[0].Vector.Count;
            if (weightedVectors.Any(item => item.Vector.Count != dimension))
            {
                weightedVectors = weightedVectors.Where(item => item.Vector.Count == dimension).ToArray();
            }

            if (weightedVectors.Length == 0)
            {
                return null;
            }

            var buffer = new double[dimension];
            var totalWeight = 0.0;
            foreach (var item in weightedVectors)
            {
                totalWeight += item.Weight;
                for (var index = 0; index < dimension; index++)
                {
                    buffer[index] += item.Vector[index] * item.Weight;
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

            return new FaceClusterDescriptor(centroid);
        }
    }
}

internal sealed record FaceFrameSample(
    AiDetectionObservation Detection,
    IReadOnlyList<AiEmbeddingObservation> Embeddings,
    double? TimeSeconds,
    int FrameOrder
);

internal sealed record PreparedFaceTrack(
    string TrackKey,
    IReadOnlyList<FaceFrameSample> Samples,
    double? StartSeconds,
    double? EndSeconds,
    double? FrameIntervalSeconds,
    double TrackQuality,
    FaceFrameSample BestSample
);

internal sealed record RepresentativeFaceEmbedding(
    string ModelKey,
    IReadOnlyList<float> Vector,
    double Norm,
    double DetectionScore,
    AiBoundingBox BoundingBox,
    double? TimeSeconds,
    double PoseQuality,
    double ImageQuality,
    double QualityScore,
    bool PassesHardFloor,
    bool PassesIdentityFloor,
    bool IsAnchor
);

internal sealed record AiAssetFaceClusterResult(
    IReadOnlyList<PreparedFaceTrack> Tracks,
    AiAssetFaceClusterDiagnostics Diagnostics
);

internal sealed record AiAssetFaceClusterDiagnostics(
    int InputTrackCount,
    int ClusterCount,
    int MergedTrackCount,
    int RejectedByConcurrencyCount,
    int RejectedByThresholdCount,
    int RejectedByAmbiguityCount
);