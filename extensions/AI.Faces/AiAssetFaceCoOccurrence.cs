namespace AI.Faces;

/// <summary>
/// Asset-local "were these two clusters on screen at the same time" evidence, derived from the frame
/// ordinals the tracks were sampled at.
///
/// <see cref="AiAssetFaceClusterer"/> already refuses to fold two tracks sharing a frame into one cluster,
/// because a frame containing two faces contains two people. That conclusion used to stop at the cluster
/// boundary: once clusters were matched against the stored identity graph, nothing stopped two clusters
/// the clusterer had just proven distinct from binding to the same identity — and nothing stopped
/// reconciliation from merging the identities afterwards. This carries the signal through to both.
///
/// Three graduated relationships, weakest evidence last:
/// <list type="bullet">
/// <item><see cref="AreConcurrent"/> — detected together in enough frames to rule out a detector artefact.
/// A hard veto: no match, no merge, at any similarity.</item>
/// <item><see cref="Overlap"/> — their frame ranges interleave without ever sharing a frame. Weak, so it
/// only withdraws the relaxed same-asset match floor rather than blocking anything.</item>
/// <item>disjoint — the shape of one performer fragmented across a video; treated exactly as before.</item>
/// </list>
/// </summary>
internal sealed class AiAssetFaceCoOccurrence
{
    // A single shared frame is not enough in a video: a detector can emit two boxes for one face, and the
    // tracker will happily carry both as separate tracks. Two genuine co-performers share many frames.
    private const int MinimumSharedFramesInMultiFrameAsset = 2;

    private readonly int _requiredSharedFrames;

    private AiAssetFaceCoOccurrence(int requiredSharedFrames)
    {
        _requiredSharedFrames = requiredSharedFrames;
    }

    public static AiAssetFaceCoOccurrence Build(IReadOnlyList<PreparedFaceTrack> tracks)
    {
        // A single-frame asset (an image) has exactly one moment, so two faces in it are two people and
        // one shared frame is all the evidence there is — or ever will be.
        var distinctFrames = tracks
            .SelectMany(static track => track.Samples)
            .Select(static sample => sample.FrameOrder)
            .Distinct()
            .Take(2)
            .Count();
        return new AiAssetFaceCoOccurrence(distinctFrames <= 1 ? 1 : MinimumSharedFramesInMultiFrameAsset);
    }

    public static FaceTrackFootprint FootprintOf(PreparedFaceTrack track)
        => new(track.Samples.Select(static sample => sample.FrameOrder));

    /// <summary>Detected in the same frames often enough to be two different people.</summary>
    public bool AreConcurrent(FaceTrackFootprint left, FaceTrackFootprint right)
    {
        if (left.IsEmpty || right.IsEmpty)
        {
            return false;
        }

        var shared = 0;
        // Iterate the smaller set so the cost is bounded by the shorter track.
        var (probe, lookup) = left.Frames.Count <= right.Frames.Count ? (left, right) : (right, left);
        foreach (var frame in probe.Frames)
        {
            if (!lookup.Frames.Contains(frame))
            {
                continue;
            }

            shared++;
            if (shared >= _requiredSharedFrames)
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>Their sampled spans interleave, whether or not any single frame is shared.</summary>
    public static bool Overlap(FaceTrackFootprint left, FaceTrackFootprint right)
        => !left.IsEmpty
           && !right.IsEmpty
           && left.MinFrame <= right.MaxFrame
           && right.MinFrame <= left.MaxFrame;
}

/// <summary>The frames an identity has been seen in so far within the asset being processed.</summary>
internal sealed class FaceTrackFootprint
{
    public FaceTrackFootprint(IEnumerable<int> frames)
    {
        Frames = [.. frames];
        MinFrame = Frames.Count == 0 ? 0 : Frames.Min();
        MaxFrame = Frames.Count == 0 ? 0 : Frames.Max();
    }

    public HashSet<int> Frames { get; }

    public int MinFrame { get; private set; }

    public int MaxFrame { get; private set; }

    public bool IsEmpty => Frames.Count == 0;

    public void Add(FaceTrackFootprint other)
    {
        if (other.IsEmpty)
        {
            return;
        }

        var wasEmpty = IsEmpty;
        Frames.UnionWith(other.Frames);
        MinFrame = wasEmpty ? other.MinFrame : Math.Min(MinFrame, other.MinFrame);
        MaxFrame = wasEmpty ? other.MaxFrame : Math.Max(MaxFrame, other.MaxFrame);
    }
}
