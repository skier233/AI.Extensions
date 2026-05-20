using AI.Extensions.Abstractions;
using AI.Faces;

using Xunit;

namespace AI.Extensions.Tests;

public sealed class AiAssetFaceClustererTests
{
    [Fact]
    public void Cluster_MergesNonConcurrentSimilarFragments()
    {
        var clusterer = new AiAssetFaceClusterer();
        var tracks = new[]
        {
            CreateTrack("track-a", 1, 1, [1f, 0f]),
            CreateTrack("track-b", 10, 10, [0.50f, 0.8660254f]),
        };

        var clusters = clusterer.Cluster(tracks, new AiFacesSettings());

        var cluster = Assert.Single(clusters);
        Assert.Equal(2, cluster.Samples.Count);
        Assert.Equal("asset-face-1", cluster.TrackKey);
    }

    [Fact]
    public void Cluster_DoesNotMergeConcurrentFaces()
    {
        var clusterer = new AiAssetFaceClusterer();
        var tracks = new[]
        {
            CreateTrack("track-a", 1, 1, [1f, 0f]),
            CreateTrack("track-b", 1, 1, [0.99f, 0.01f]),
        };

        var clusters = clusterer.Cluster(tracks, new AiFacesSettings());

        Assert.Equal(2, clusters.Count);
        Assert.All(clusters, cluster => Assert.Single(cluster.Samples));
    }

    [Fact]
    public void Cluster_PreservesSamplesWhenMergingFragments()
    {
        var clusterer = new AiAssetFaceClusterer();
        var tracks = new[]
        {
            CreateTrack("track-a", 1, 1, [1f, 0f]),
            CreateTrack("track-b", 8, 8, [0.52f, 0.8541662f]),
            CreateTrack("track-c", 20, 20, [-1f, 0f]),
        };

        var clusters = clusterer.Cluster(tracks, new AiFacesSettings());

        Assert.Equal(2, clusters.Count);
        Assert.Contains(clusters, cluster => cluster.Samples.Count == 2);
        Assert.Contains(clusters, cluster => cluster.Samples.Count == 1);
    }

    private static PreparedFaceTrack CreateTrack(string trackKey, int frameOrder, double timeSeconds, IReadOnlyList<float> vector)
    {
        var detection = new AiDetectionObservation(
            "face_detector_torchexport",
            0,
            "face",
            0.97,
            new AiBoundingBox(0.1, 0.1, 0.3, 0.3));
        var sample = new FaceFrameSample(
            detection,
            [new AiEmbeddingObservation("face_embedding_torchexport", "region", vector, 24.0, 0)],
            timeSeconds,
            frameOrder);

        return new PreparedFaceTrack(
            trackKey,
            [sample],
            timeSeconds,
            timeSeconds,
            1.0,
            24.0,
            sample);
    }
}