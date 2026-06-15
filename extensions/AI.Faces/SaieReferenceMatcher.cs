namespace AI.Faces;

// Finds the best-matching reference performer for a set of face anchor vectors. Uses the pack's
// SIMD-accelerated cosine (SaieReferencePack.CosineSimilarityTo) and a single streaming pass that tracks
// the best and runner-up score, replacing the previous approach that allocated one match object per
// performer and fully sorted them with a scalar cosine. With ~150k performers this is the per-image hot
// path for reference matching (and the bulk path for pack-import backfill), so it stays allocation-light.
internal static class SaieReferenceMatcher
{
    internal readonly record struct ReferenceMatchResult(int Ordinal, double Score, double SecondScore);

    public static ReferenceMatchResult? FindBest(SaieReferencePack pack, IReadOnlyList<IReadOnlyList<float>> anchorVectors)
    {
        if (pack.Identities.Count == 0 || anchorVectors.Count == 0)
        {
            return null;
        }

        var dimension = pack.Manifest.EmbeddingDim;
        var sources = new List<(float[] Vector, float Norm)>(anchorVectors.Count);
        foreach (var vector in anchorVectors)
        {
            if (vector.Count != dimension)
            {
                continue;
            }

            var array = vector as float[] ?? vector.ToArray();
            var norm = SaieReferencePack.Norm(array);
            if (norm > 0f)
            {
                sources.Add((array, norm));
            }
        }

        if (sources.Count == 0)
        {
            return null;
        }

        var bestOrdinal = -1;
        double bestScore = 0.0;
        double secondScore = 0.0;
        var ordinalCount = pack.Identities.Count;
        for (var ordinal = 0; ordinal < ordinalCount; ordinal++)
        {
            var score = ScoreOrdinal(pack, ordinal, sources);
            if (score <= 0.0)
            {
                continue;
            }

            if (score > bestScore)
            {
                secondScore = bestScore;
                bestScore = score;
                bestOrdinal = ordinal;
            }
            else if (score > secondScore)
            {
                secondScore = score;
            }
        }

        return bestOrdinal < 0 ? null : new ReferenceMatchResult(bestOrdinal, bestScore, secondScore);
    }

    // Mirrors the previous scoring: the average of the top-2 anchor-vs-centroid similarities (or the
    // single best when only one anchor scores).
    private static double ScoreOrdinal(SaieReferencePack pack, int ordinal, List<(float[] Vector, float Norm)> sources)
    {
        double best = 0.0;
        double second = 0.0;
        var count = 0;
        foreach (var source in sources)
        {
            double score = pack.CosineSimilarityTo(ordinal, source.Vector, source.Norm);
            if (score <= 0.0)
            {
                continue;
            }

            count++;
            if (score >= best)
            {
                second = best;
                best = score;
            }
            else if (score > second)
            {
                second = score;
            }
        }

        return count == 0 ? 0.0 : count >= 2 ? (best + second) / 2.0 : best;
    }
}
