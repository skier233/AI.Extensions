using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

using Pgvector;

namespace AI.Faces;

internal sealed record AiFaceNotPresentRequest(string HostType, int HostId);

internal sealed record AiFaceNotPresentResult(
    bool FaceFound,
    bool HostHadFace,
    int MovedHostCount,
    int? TargetFaceId,
    bool CreatedNewFace,
    bool MergedIntoTarget,
    bool SourceFaceEmptied)
{
    public static readonly AiFaceNotPresentResult NotFound = new(false, false, 0, null, false, false, false);
    public static readonly AiFaceNotPresentResult NoFaceOnHost = new(true, false, 0, null, false, false, false);
}

/// <summary>
/// Applies a user's "this face is not actually present here" decision. The marked host's occurrence of
/// the face is treated as a negative exemplar: the face's per-occurrence embeddings (one per video/image
/// it was seen in) are partitioned, the occurrences that clearly match the exemplar are split off, and
/// they are either folded into an existing face that matches them or moved to a new face. The original
/// face keeps the rest. A durable suppression is recorded so a future AI re-run does not re-attach the
/// face to the marked host.
/// </summary>
internal sealed class AiFaceNotPresentService(
    IServiceScopeFactory scopeFactory,
    AiFacePresenceSuppressionStore suppressionStore)
{
    private const string FaceSourceKey = "ext:ai.faces";

    public async Task<AiFaceNotPresentResult> MarkNotPresentAsync(int faceId, string hostType, int hostId, CancellationToken ct = default)
    {
        var normalizedHostType = NormalizeHostType(hostType);
        if (normalizedHostType is not ("video" or "image") || faceId <= 0 || hostId <= 0)
        {
            return AiFaceNotPresentResult.NotFound;
        }

        await using var scope = scopeFactory.CreateAsyncScope();
        var faceRepo = scope.ServiceProvider.GetRequiredService<IFaceRepository>();
        var embeddingRepo = scope.ServiceProvider.GetRequiredService<IEmbeddingRepository>();
        var detectionRepo = scope.ServiceProvider.GetRequiredService<IDetectionRepository>();
        var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();
        var embeddingService = scope.ServiceProvider.GetRequiredService<IEmbeddingService>();
        var settings = await AiFacesSettingsRuntime.LoadAsync(ct);

        var face = await faceRepo.GetFaceAsync(faceId, tracking: true, ct);
        if (face is null)
        {
            return AiFaceNotPresentResult.NotFound;
        }

        var markedHost = new HostRef(normalizedHostType, hostId);

        // Map this face's occurrences to their hosts via the run id each appearance/embedding shares
        // (one run == one processed video/image), then build a per-host centroid of the face embeddings.
        var appearances = await faceRepo.FindAppearancesAsync(new FaceAppearanceFilter { FaceIds = [faceId], SourceKey = FaceSourceKey }, ct);
        var hostByRunId = new Dictionary<string, HostRef>(StringComparer.Ordinal);
        foreach (var appearance in appearances)
        {
            if (!string.IsNullOrWhiteSpace(appearance.SourceRunId))
            {
                hostByRunId[appearance.SourceRunId] = new HostRef(
                    appearance.HostType == FaceAppearanceHostType.Video ? "video" : "image",
                    appearance.HostId);
            }
        }

        var faceEmbeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
        {
            HostType = EmbeddingHostType.Face,
            HostId = faceId,
            SourceKey = FaceSourceKey,
            Modality = EmbeddingModality.Face,
        }, ct);

        var vectorsByHost = new Dictionary<HostRef, List<float[]>>();
        var runIdsByHost = new Dictionary<HostRef, HashSet<string>>();
        foreach (var embedding in faceEmbeddings)
        {
            if (string.IsNullOrWhiteSpace(embedding.SourceRunId) || !hostByRunId.TryGetValue(embedding.SourceRunId, out var host))
            {
                continue;
            }

            var vector = embedding.Vector.ToArray();
            if (vector.Length == 0)
            {
                continue;
            }

            (vectorsByHost.TryGetValue(host, out var list) ? list : vectorsByHost[host] = []).Add(vector);
            (runIdsByHost.TryGetValue(host, out var runIds) ? runIds : runIdsByHost[host] = new HashSet<string>(StringComparer.Ordinal)).Add(embedding.SourceRunId);
        }

        var hostsWithFace = appearances
            .Select(a => new HostRef(a.HostType == FaceAppearanceHostType.Video ? "video" : "image", a.HostId))
            .ToHashSet();
        if (!hostsWithFace.Contains(markedHost))
        {
            return AiFaceNotPresentResult.NoFaceOnHost;
        }

        var centroidByHost = vectorsByHost.ToDictionary(pair => pair.Key, pair => Centroid(pair.Value));

        // Determine which occurrences are split off. The marked host always is; other hosts join it only
        // when they clearly resemble the marked exemplar more than the face's retained identity.
        var removeHosts = new HashSet<HostRef> { markedHost };
        if (centroidByHost.TryGetValue(markedHost, out var markedCentroid) && markedCentroid.Length > 0)
        {
            var otherHosts = centroidByHost.Keys.Where(host => !host.Equals(markedHost)).ToArray();
            var keepCentroid = Centroid(otherHosts.Select(host => centroidByHost[host]).ToList());
            foreach (var host in otherHosts)
            {
                var simToMarked = Cosine(centroidByHost[host], markedCentroid);
                var simToKeep = keepCentroid.Length == 0 ? 0.0 : Cosine(centroidByHost[host], keepCentroid);
                if (simToMarked >= settings.NotPresentSplitSimilarityThreshold
                    && simToMarked > simToKeep + settings.IdentityAmbiguityMargin)
                {
                    removeHosts.Add(host);
                }
            }
        }

        // Every face row (appearance, detection, segment, embedding) carries the run id that produced it,
        // and one run is one processed host, so re-homing the split-off hosts is a uniform re-point by
        // run id across all four row kinds.
        var removeRunIds = appearances
            .Where(a => removeHosts.Contains(new HostRef(a.HostType == FaceAppearanceHostType.Video ? "video" : "image", a.HostId)))
            .Select(a => a.SourceRunId)
            .Concat(removeHosts.SelectMany(host => runIdsByHost.GetValueOrDefault(host) ?? []))
            .Where(runId => !string.IsNullOrWhiteSpace(runId))
            .Select(runId => runId!)
            .ToHashSet(StringComparer.Ordinal);
        var removeCentroid = Centroid(removeHosts
            .Where(centroidByHost.ContainsKey)
            .Select(host => centroidByHost[host])
            .ToList());

        // Re-home the split-off occurrences: fold them into an existing face that matches, or a new face.
        var (target, createdNew, mergedIntoTarget) = await ResolveTargetFaceAsync(
            faceRepo, embeddingService, faceId, removeCentroid, settings, ct);

        var movedAppearanceCount = await faceRepo.ReassignAppearancesByRunAsync(FaceSourceKey, faceId, removeRunIds, target.Id, ct);
        await detectionRepo.ReassignRefByRunAsync(FaceSourceKey, "face", faceId, removeRunIds, target.Id, ct);
        await segmentRepo.ReassignRefByRunAsync(FaceSourceKey, faceId, removeRunIds, target.Id, ct);
        await embeddingRepo.ReassignHostByRunAsync(EmbeddingHostType.Face, FaceSourceKey, faceId, removeRunIds, target.Id, ct);

        var sourceEmptied = movedAppearanceCount >= appearances.Count;
        if (sourceEmptied && face.MergedIntoFaceId is null)
        {
            // Every occurrence was the wrong person; the original face has nothing left. Fold it into the
            // target so it stops showing as an empty cluster.
            face.MergedIntoFaceId = target.Id;
        }

        await faceRepo.SaveChangesAsync(ct);
        await AiFacesPersistenceService.RefreshFaceStatsAsync(faceRepo, detectionRepo, [faceId, target.Id], ct);
        await faceRepo.SaveChangesAsync(ct);

        // The moved hosts now point at the target face; recompute their performer assignments so the
        // wrong face's performer is removed and the target's (if any) is applied.
        var propagation = scope.ServiceProvider.GetService<IFacePerformerPropagationService>();
        if (propagation is not null)
        {
            foreach (var host in removeHosts)
            {
                var appearanceHostType = host.Type == "video" ? FaceAppearanceHostType.Video : FaceAppearanceHostType.Image;
                try
                {
                    await propagation.ReconcileHostAsync(appearanceHostType, host.Id, ct);
                }
                catch
                {
                    // Best-effort: a propagation failure must not undo the split that already committed.
                }
            }

            await faceRepo.SaveChangesAsync(ct);
        }

        var suggestionMaintenance = scope.ServiceProvider.GetService<IFaceTopSuggestionMaintenance>();
        if (suggestionMaintenance is not null)
        {
            await suggestionMaintenance.InvalidateAsync([faceId, target.Id], ct);
        }

        if (!string.IsNullOrWhiteSpace(face.PrimarySourceKey))
        {
            await suppressionStore.AddAsync(
                removeHosts.Select(host => new AiFacePresenceSuppression(face.PrimarySourceKey!, host.Type, host.Id)),
                ct);
        }

        return new AiFaceNotPresentResult(
            FaceFound: true,
            HostHadFace: true,
            MovedHostCount: removeHosts.Count,
            TargetFaceId: target.Id,
            CreatedNewFace: createdNew,
            MergedIntoTarget: mergedIntoTarget,
            SourceFaceEmptied: sourceEmptied);
    }

    private static async Task<(Face Target, bool CreatedNew, bool MergedIntoExisting)> ResolveTargetFaceAsync(
        IFaceRepository faceRepo,
        IEmbeddingService embeddingService,
        int sourceFaceId,
        float[] removeCentroid,
        AiFacesSettings settings,
        CancellationToken ct)
    {
        if (removeCentroid.Length > 0)
        {
            var matches = await embeddingService.KnnAsync(
                new Vector(removeCentroid),
                k: 20,
                new EmbeddingSearchOptions
                {
                    HostType = EmbeddingHostType.Face,
                    SourceKey = FaceSourceKey,
                    Modality = EmbeddingModality.Face,
                },
                ct);

            var bestSimByFaceId = new Dictionary<int, double>();
            foreach (var match in matches)
            {
                if (match.Embedding.HostId == sourceFaceId)
                {
                    continue;
                }

                var similarity = 1.0 - match.Distance;
                if (!bestSimByFaceId.TryGetValue(match.Embedding.HostId, out var existing) || similarity > existing)
                {
                    bestSimByFaceId[match.Embedding.HostId] = similarity;
                }
            }

            var rankedFaceIds = bestSimByFaceId
                .Where(pair => pair.Value >= settings.ConsolidationSimilarityThreshold)
                .OrderByDescending(pair => pair.Value)
                .Select(pair => pair.Key)
                .ToArray();
            if (rankedFaceIds.Length > 0)
            {
                var candidates = await faceRepo.FindFacesAsync(
                    new FaceFilter { Ids = rankedFaceIds, IsMerged = false, Ignored = false },
                    tracking: true,
                    ct);
                var candidateById = candidates.ToDictionary(candidate => candidate.Id);
                foreach (var candidateId in rankedFaceIds)
                {
                    if (candidateById.TryGetValue(candidateId, out var candidate))
                    {
                        return (candidate, false, true);
                    }
                }
            }
        }

        var created = new Face { PrimarySourceKey = $"face-split-{Guid.NewGuid():N}" };
        faceRepo.AddFace(created);
        await faceRepo.SaveChangesAsync(ct);
        return (created, true, false);
    }

    private static float[] Centroid(IReadOnlyList<float[]> vectors)
    {
        if (vectors.Count == 0)
        {
            return [];
        }

        var dimension = vectors[0].Length;
        var buffer = new double[dimension];
        var count = 0;
        foreach (var vector in vectors)
        {
            if (vector.Length != dimension)
            {
                continue;
            }

            for (var i = 0; i < dimension; i++)
            {
                buffer[i] += vector[i];
            }

            count++;
        }

        if (count == 0)
        {
            return [];
        }

        var result = new float[dimension];
        var norm = 0.0;
        for (var i = 0; i < dimension; i++)
        {
            result[i] = (float)(buffer[i] / count);
            norm += result[i] * result[i];
        }

        norm = Math.Sqrt(norm);
        if (norm <= 0.0)
        {
            return [];
        }

        for (var i = 0; i < dimension; i++)
        {
            result[i] = (float)(result[i] / norm);
        }

        return result;
    }

    private static double Cosine(float[] left, float[] right)
    {
        if (left.Length == 0 || left.Length != right.Length)
        {
            return 0.0;
        }

        double dot = 0.0;
        double leftNorm = 0.0;
        double rightNorm = 0.0;
        for (var i = 0; i < left.Length; i++)
        {
            dot += left[i] * right[i];
            leftNorm += left[i] * left[i];
            rightNorm += right[i] * right[i];
        }

        if (leftNorm <= 0.0 || rightNorm <= 0.0)
        {
            return 0.0;
        }

        return Math.Clamp(dot / (Math.Sqrt(leftNorm) * Math.Sqrt(rightNorm)), -1.0, 1.0);
    }

    private static string NormalizeHostType(string hostType)
        => hostType.Trim().ToLowerInvariant() switch
        {
            "video" or "videos" => "video",
            "image" or "images" => "image",
            var other => other,
        };

    private readonly record struct HostRef(string Type, int Id);
}
