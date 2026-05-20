namespace AI.Extensions.Abstractions;

public static class AiAnalysisExtensions
{
    public static IEnumerable<AiEmbeddingObservation> EnumerateAllEmbeddings(this AiAnalysisNode analysis)
    {
        foreach (var embedding in analysis.Embeddings)
        {
            yield return embedding;
        }

        foreach (var branch in analysis.RegionBranches)
        {
            foreach (var embedding in branch.Analysis.Embeddings)
            {
                yield return embedding with
                {
                    DetectionIndex = embedding.DetectionIndex ?? branch.DetectionIndex,
                    SourceBranchKey = embedding.SourceBranchKey ?? branch.BranchKey,
                };
            }
        }
    }

    public static IEnumerable<AiDetectionObservation> FindDetections(this AiAnalysisNode analysis, string hint)
    {
        if (string.IsNullOrWhiteSpace(hint))
        {
            return analysis.Detections;
        }

        return analysis.Detections.Where(detection =>
            string.Equals(detection.Label, hint, StringComparison.OrdinalIgnoreCase) ||
            detection.Label.Contains(hint, StringComparison.OrdinalIgnoreCase) ||
            detection.ModelKey.Contains(hint, StringComparison.OrdinalIgnoreCase));
    }

    public static IReadOnlyDictionary<string, int> ToPreparedCounts(this AiPreparedArtifactBatch batch)
    {
        var counts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);

        AddCount("tagLinks", batch.TagLinks.Count);
        AddCount("segments", batch.Segments.Count);
        AddCount("faceAppearances", batch.FaceAppearances.Count);
        AddCount("detections", batch.Detections.Count);
        AddCount("embeddings", batch.Embeddings.Count);
        AddCount("faces", batch.Faces.Count);
        AddCount("deferred", batch.DeferredWorkItems.Count);

        return counts;

        void AddCount(string key, int count)
        {
            if (count > 0)
            {
                counts[key] = count;
            }
        }
    }
}
