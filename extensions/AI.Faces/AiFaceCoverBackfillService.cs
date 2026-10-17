using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

using PluginJobProgress = Cove.Plugins.IJobProgress;

namespace AI.Faces;

internal sealed record AiFaceCoverBackfillResult(
    int FacesWithoutCover,
    int CreatedCoverCount,
    int NotCreatedCount,
    int SkippedVideoWithoutTimestampCount,
    int SkippedMissingSourceCount,
    int SkippedNoSampleCount)
{
    public string Summary
        => $"Created {CreatedCoverCount} of {FacesWithoutCover} missing face cover(s). "
           + $"Not created (unreadable source, blank crop, error, or covered meanwhile): {NotCreatedCount}. "
           + $"Skipped: {SkippedVideoWithoutTimestampCount} whose best sample is a video frame without a stored time, {SkippedMissingSourceCount} whose source file is missing, "
           + $"{SkippedNoSampleCount} without a stored best sample.";
}

/// <summary>
/// Builds covers for faces that have none, from each face's stored best sample (the identity's cover asset, box and,
/// for a video frame, its time), without re-running any model. Covers are otherwise only queued while persisting an
/// AI run, and a normal run skips reusable results before that point, so faces whose earlier cover attempt failed
/// never get another.
/// </summary>
internal sealed class AiFaceCoverBackfillService(IServiceScopeFactory scopeFactory)
{
    // Cover crops decode full-resolution stills or video frames; bound the parallelism so a large backfill can't
    // exhaust memory.
    private const int MaxParallelCovers = 4;
    private const int ProgressInterval = 25;

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;

    public async Task<AiFaceCoverBackfillResult> BackfillAsync(PluginJobProgress? progress, CancellationToken ct = default)
    {
        progress?.Report(0, "Finding faces without a cover...");
        var candidates = await LoadCandidatesAsync(ct);

        var skippedVideo = 0;
        var skippedMissing = 0;
        var skippedNoSample = 0;
        var workItems = new List<FaceCoverWorkItem>(candidates.Count);
        foreach (var candidate in candidates)
        {
            if (candidate.Sample is not { CoverAssetId: { } assetId } sample
                || string.IsNullOrWhiteSpace(assetId)
                || sample.CoverX1 is not { } x1 || sample.CoverY1 is not { } y1 || sample.CoverX2 is not { } x2 || sample.CoverY2 is not { } y2)
            {
                skippedNoSample++;
                continue;
            }

            assetId = assetId.Trim();
            var isStill = AiFaceCoverGenerator.IsStillImagePath(assetId);
            // A video frame can be found again only by its time, which identities stored before it was recorded lack.
            if (!isStill && sample.CoverObservedAtSeconds is null)
            {
                skippedVideo++;
                continue;
            }

            if (!AiFaceCoverGenerator.SourceExists(assetId))
            {
                skippedMissing++;
                continue;
            }

            workItems.Add(new FaceCoverWorkItem(
                candidate.FaceId,
                new FaceCoverSource(assetId, new AiBoundingBox(x1, y1, x2, y2), isStill ? null : sample.CoverObservedAtSeconds),
                IncomingCoverQuality: sample.CoverQualityScore));
        }

        var created = 0;
        var processed = 0;
        var configured = 0;
        await using (var scope = _scopeFactory.CreateAsyncScope())
        {
            configured = scope.ServiceProvider.GetService<CoveConfiguration>()?.MaxParallelTasks ?? 0;
            // Create the cover-quality field once up front; parallel workers would otherwise race to create it.
            if (workItems.Count > 0)
            {
                await AiFacesPersistenceService.EnsureCoverBlobQualityDefinitionAsync(scope.ServiceProvider.GetRequiredService<ICustomFieldRepository>(), ct);
            }
        }

        var parallelism = Math.Clamp(configured <= 0 ? Environment.ProcessorCount : configured, 1, MaxParallelCovers);
        await Parallel.ForEachAsync(workItems, new ParallelOptions { MaxDegreeOfParallelism = parallelism, CancellationToken = ct }, async (item, itemCt) =>
        {
            await using var scope = _scopeFactory.CreateAsyncScope();
            try
            {
                // GenerateAndStoreCoverAsync reloads the face and re-checks that it still needs a cover, so a cover an AI
                // run stored in the meantime is kept rather than overwritten.
                if (await AiFacesPersistenceService.GenerateAndStoreCoverAsync(scope.ServiceProvider, item, itemCt))
                {
                    Interlocked.Increment(ref created);
                }
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                // One face failing (a storage or database error) must not abort the rest of the backfill.
                scope.ServiceProvider.GetService<ILogger<AiFaceCoverBackfillService>>()?.LogWarning(ex, "AI.Faces cover backfill failed for face {FaceId}.", item.FaceId);
            }

            var done = Interlocked.Increment(ref processed);
            if (done % ProgressInterval == 0 || done == workItems.Count)
            {
                progress?.Report((double)done / workItems.Count, $"Building face covers: {done} of {workItems.Count}...");
            }
        });

        var result = new AiFaceCoverBackfillResult(
            candidates.Count,
            created,
            workItems.Count - created,
            skippedVideo,
            skippedMissing,
            skippedNoSample);
        progress?.Report(1, result.Summary);
        return result;
    }

    private async Task<IReadOnlyList<CoverBackfillCandidate>> LoadCandidatesAsync(CancellationToken ct)
    {
        await using var scope = _scopeFactory.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<DbContext>();

        var faces = await db.Set<Face>()
            .AsNoTracking()
            .Where(static face => face.MergedIntoFaceId == null
                && (face.CoverBlobId == null || face.CoverBlobId == "")
                && face.PrimarySourceKey != null && face.PrimarySourceKey != "")
            .OrderBy(static face => face.Id)
            .Select(static face => new { face.Id, FaceKey = face.PrimarySourceKey! })
            .ToListAsync(ct);
        if (faces.Count == 0)
        {
            return [];
        }

        var faceKeys = faces.Select(static face => face.FaceKey).Distinct(StringComparer.Ordinal).ToArray();
        var samples = await db.Set<ExtAiFacesIdentityEntity>()
            .AsNoTracking()
            .Where(identity => faceKeys.Contains(identity.FaceKey))
            .Select(static identity => new CoverSample(
                identity.FaceKey,
                identity.CoverAssetId,
                identity.CoverX1,
                identity.CoverY1,
                identity.CoverX2,
                identity.CoverY2,
                identity.CoverObservedAtSeconds,
                identity.CoverQualityScore))
            .ToListAsync(ct);
        var samplesByKey = samples.ToDictionary(static sample => sample.FaceKey, StringComparer.Ordinal);

        // Faces from other providers have no AI.Faces identity; they aren't ours to cover.
        return faces
            .Where(face => samplesByKey.ContainsKey(face.FaceKey))
            .Select(face => new CoverBackfillCandidate(face.Id, face.FaceKey, samplesByKey[face.FaceKey]))
            .ToList();
    }

    private sealed record CoverSample(
        string FaceKey,
        string? CoverAssetId,
        double? CoverX1,
        double? CoverY1,
        double? CoverX2,
        double? CoverY2,
        double? CoverObservedAtSeconds,
        double CoverQualityScore);

    private sealed record CoverBackfillCandidate(int FaceId, string FaceKey, CoverSample Sample);
}
