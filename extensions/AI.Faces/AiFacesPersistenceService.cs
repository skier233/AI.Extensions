using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

using Pgvector;

namespace AI.Faces;

internal sealed class AiFacesPersistenceService(IServiceScopeFactory scopeFactory, AiFacePresenceSuppressionStore? suppressionStore = null)
{
    private const string FaceSourceKey = "ext:ai.faces";
    private const int NormalizedFrameSize = 1;
    private const string CoverQualityScoreField = "ai.faces.coverQualityScore";
    private const string CoverBlobQualityScoreField = "ai.faces.coverBlobQualityScore";
    private const double CoverReplacementQualityMargin = 0.02;

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;
    private readonly AiFacePresenceSuppressionStore? _suppressionStore = suppressionStore;

    public Task<IReadOnlyList<string>> PersistAsync(AiDispatchRequest request, AiPreparedArtifactBatch batch, CancellationToken ct = default)
        => PersistAsync(request, batch, mergedFaceKeyMap: null, ct);

    public async Task<IReadOnlyList<string>> PersistAsync(
        AiDispatchRequest request,
        AiPreparedArtifactBatch batch,
        IReadOnlyDictionary<string, string>? mergedFaceKeyMap,
        CancellationToken ct = default)
    {
        if (request.Context.HostEntityId is null || string.IsNullOrWhiteSpace(request.Context.HostEntityType))
        {
            return ["AI.Faces prepared artifacts but skipped persistence because no Cove host entity identity was supplied."];
        }

        var hostEntityType = NormalizeHostEntityType(request.Context.HostEntityType);
        if (hostEntityType is not ("video" or "image"))
        {
            return [$"AI.Faces persistence does not support host entity type '{request.Context.HostEntityType}'."];
        }

        await using var scope = _scopeFactory.CreateAsyncScope();
        var faceRepo = scope.ServiceProvider.GetRequiredService<IFaceRepository>();
        var detectionRepo = scope.ServiceProvider.GetRequiredService<IDetectionRepository>();
        var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();
        var embeddingRepo = scope.ServiceProvider.GetRequiredService<IEmbeddingRepository>();
        var customFieldRepo = scope.ServiceProvider.GetRequiredService<ICustomFieldRepository>();

        var hostEntityId = request.Context.HostEntityId.Value;
        var detectionHostType = hostEntityType == "video" ? DetectionHostType.Video : DetectionHostType.Image;
        var appearanceHostType = hostEntityType == "video" ? FaceAppearanceHostType.Video : FaceAppearanceHostType.Image;

        // Faces the user has explicitly marked not-present on this host must not be re-attached by a
        // re-run. Drop their artifacts before persistence so the host's previously-removed appearance
        // stays gone (the detections that would have backed it are simply not attributed here).
        var suppressedFaceKeys = _suppressionStore is null
            ? (IReadOnlySet<string>)new HashSet<string>()
            : await _suppressionStore.GetSuppressedFaceKeysAsync(hostEntityType, hostEntityId, ct);

        var facesByKey = await ResolveFacesAsync(faceRepo, batch.Faces, suppressedFaceKeys, ct);
        var affectedFaceIds = new HashSet<int>(facesByKey.Values.Select(static face => face.Id));

        var existingAppearances = await faceRepo.FindAppearancesAsync(new FaceAppearanceFilter
        {
            HostType = appearanceHostType, HostId = hostEntityId, SourceKey = FaceSourceKey,
        }, ct);
        foreach (var appearance in existingAppearances) affectedFaceIds.Add(appearance.FaceId);

        var existingDetections = await detectionRepo.FindAsync(new DetectionFilter
        {
            HostType = detectionHostType, HostId = hostEntityId, SourceKey = FaceSourceKey,
        }, ct);
        foreach (var detection in existingDetections)
        {
            if (detection.RefKind?.Equals("face", StringComparison.OrdinalIgnoreCase) == true && detection.RefId is { } rid && rid > 0)
                affectedFaceIds.Add((int)rid);
        }

        var existingSegments = hostEntityType == "video"
            ? await segmentRepo.FindAsync(new SegmentFilter { HostType = SegmentHostType.Video, HostId = hostEntityId, SourceKey = FaceSourceKey }, ct)
            : (IReadOnlyList<Segment>)[];
        foreach (var segment in existingSegments)
        {
            if (segment.RefId is { } rid && rid > 0) affectedFaceIds.Add((int)rid);
        }

        foreach (var detection in batch.Detections) { var id = ResolveFaceId(facesByKey, detection.RefKey); if (id > 0) affectedFaceIds.Add(id); }
        foreach (var appearance in batch.FaceAppearances) { var id = ResolveFaceId(facesByKey, appearance.RefKey); if (id > 0) affectedFaceIds.Add(id); }
        foreach (var segment in batch.Segments) { var id = ResolveFaceId(facesByKey, segment.RefKey); if (id > 0) affectedFaceIds.Add(id); }
        foreach (var embedding in batch.Embeddings) { var id = ResolveFaceId(facesByKey, embedding.HostRefKey); if (id > 0) affectedFaceIds.Add(id); }

        var existingEmbeddings = affectedFaceIds.Count == 0
            ? (IReadOnlyList<Embedding>)[]
            : await embeddingRepo.FindAsync(new EmbeddingFilter
            {
                HostType = EmbeddingHostType.Face, HostIds = affectedFaceIds.ToArray(), SourceKey = FaceSourceKey,
            }, ct);

        detectionRepo.RemoveRange(existingDetections);
        segmentRepo.RemoveRange(existingSegments);
        embeddingRepo.RemoveRange(existingEmbeddings);
        faceRepo.RemoveAppearances(existingAppearances);

        var persistedAppearances = PersistFaceAppearances(faceRepo, hostEntityId, appearanceHostType, batch, facesByKey, request);
        var persistedDetections = PersistDetections(detectionRepo, hostEntityId, detectionHostType, batch, facesByKey, request);
        var persistedSegments = hostEntityType == "video" ? PersistSegments(segmentRepo, hostEntityId, batch, facesByKey, request) : 0;
        var persistedEmbeddings = PersistEmbeddings(embeddingRepo, batch, facesByKey, request);
        var persistedFaceCovers = await PersistFaceCoversAsync(scope.ServiceProvider, customFieldRepo, faceRepo, hostEntityType, request, batch, facesByKey, ct);

        await faceRepo.SaveChangesAsync(ct);

        // Reconciliation during preparation may have merged identities whose Cove rows were persisted
        // by earlier runs; apply those merges here so the duplicate face rows don't linger as orphans
        // that keep collecting stale suggestions.
        AiFacePersistedFaceMergeResult? mergeResult = null;
        if (mergedFaceKeyMap is { Count: > 0 })
        {
            mergeResult = await AiFacePersistedFaceMerger.ApplyAsync(
                faceRepo, embeddingRepo, detectionRepo, segmentRepo, FaceSourceKey, mergedFaceKeyMap, ct);
            foreach (var faceId in mergeResult.AffectedFaceIds)
                affectedFaceIds.Add(faceId);
            if (mergeResult.MergedPersistedFaceCount > 0)
                await faceRepo.SaveChangesAsync(ct);
        }

        if (affectedFaceIds.Count > 0)
        {
            await RefreshFaceStatsAsync(faceRepo, detectionRepo, affectedFaceIds, ct);
            await faceRepo.SaveChangesAsync(ct);
        }

        // Apply the performer of any already-linked face that now appears on this host, and remove the
        // performer contributed by faces that no longer appear here after a re-run. The steps above
        // (re)write this host's face appearances but otherwise never touch host performer assignments,
        // so without this a video/image that matches an existing linked face never gets the performer.
        var notes = new List<string>();
        var propagation = scope.ServiceProvider.GetService<IFacePerformerPropagationService>();
        if (propagation is not null)
        {
            // Best-effort: performer propagation must never break face persistence. The face
            // appearances/detections above are already committed, so a failure here only means the
            // host's performer assignments weren't updated this run.
            try
            {
                await propagation.ReconcileHostAsync(appearanceHostType, hostEntityId, ct);

                // A merge that transferred a performer onto its target face must propagate that
                // performer to every host the target now appears on, not just this run's host.
                if (mergeResult is { RelinkedTargetFaceIds.Count: > 0 })
                {
                    var mergedHosts = await faceRepo.FindAppearancesAsync(
                        new FaceAppearanceFilter { FaceIds = mergeResult.RelinkedTargetFaceIds.Distinct().ToArray() }, ct);
                    foreach (var hostGroup in mergedHosts.GroupBy(static appearance => (appearance.HostType, appearance.HostId)))
                        await propagation.ReconcileHostAsync(hostGroup.Key.HostType, hostGroup.Key.HostId, ct);
                }

                await faceRepo.SaveChangesAsync(ct);
            }
            catch (Exception ex)
            {
                notes.Add($"AI.Faces applied face artifacts but performer propagation failed for this {hostEntityType}: {ex.Message}");
            }
        }

        // The faces list reads a materialized top-suggestion projection. Evidence persisted in this
        // run changes what the suggester would say for these faces, so stamp them for recompute —
        // otherwise the list keeps showing a suggestion computed before this run's evidence existed,
        // diverging from the (compute-on-read) detail page.
        if (affectedFaceIds.Count > 0)
        {
            var suggestionMaintenance = scope.ServiceProvider.GetService<IFaceTopSuggestionMaintenance>();
            if (suggestionMaintenance is not null)
            {
                try
                {
                    await suggestionMaintenance.InvalidateAsync(affectedFaceIds.ToArray(), ct);
                }
                catch (Exception ex)
                {
                    notes.Add($"AI.Faces persisted face artifacts but could not invalidate materialized top suggestions: {ex.Message}");
                }
            }
        }

        if (batch.Faces.Count > 0) notes.Add($"Resolved {batch.Faces.Count} AI face identity candidate(s) into Cove face cluster(s).");
        if (mergeResult is { MergedPersistedFaceCount: > 0 }) notes.Add($"Merged {mergeResult.MergedPersistedFaceCount} duplicate face row(s) into their reconciled targets.");
        if (persistedAppearances > 0) notes.Add($"Persisted {persistedAppearances} AI-generated face appearance(s) onto the {hostEntityType}.");
        if (persistedDetections > 0) notes.Add($"Persisted {persistedDetections} retained AI-generated face spatial sample(s) onto the {hostEntityType}.");
        if (persistedSegments > 0) notes.Add($"Persisted {persistedSegments} AI-generated face segment(s) onto the video timeline.");
        if (persistedEmbeddings > 0) notes.Add($"Persisted {persistedEmbeddings} face embedding(s) for similarity and clustering workflows.");
        if (persistedFaceCovers > 0) notes.Add($"Generated {persistedFaceCovers} face cover image(s) for face detail pages.");

        if (notes.Count == 0)
        {
            notes.Add(existingAppearances.Count + existingDetections.Count + existingSegments.Count + existingEmbeddings.Count > 0
                ? "AI.Faces cleared previously persisted rows for this host because the latest run did not emit any current face artifacts."
                : "AI.Faces found no new face rows to persist for this host entity.");
        }

        return notes;
    }

    private static async Task<Dictionary<string, Face>> ResolveFacesAsync(IFaceRepository faceRepo, IReadOnlyList<AiPreparedFaceIdentity> faces, IReadOnlySet<string> suppressedFaceKeys, CancellationToken ct)
    {
        var persistableFaces = faces
            .Where(IsPersistableFace)
            .Where(face => !suppressedFaceKeys.Contains(face.FaceKey))
            .ToArray();
        var faceKeys = persistableFaces
            .Select(static f => f.FaceKey)
            .Where(static k => !string.IsNullOrWhiteSpace(k))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (faceKeys.Length == 0)
            return new Dictionary<string, Face>(StringComparer.OrdinalIgnoreCase);

        var existingFaces = await faceRepo.FindFacesAsync(new FaceFilter { PrimarySourceKeys = faceKeys }, tracking: true, ct);
        var facesByKey = new Dictionary<string, Face>(StringComparer.OrdinalIgnoreCase);

        foreach (var face in existingFaces)
        {
            if (!string.IsNullOrWhiteSpace(face.PrimarySourceKey) && !facesByKey.ContainsKey(face.PrimarySourceKey))
                facesByKey[face.PrimarySourceKey] = face;
        }

        foreach (var preparedFace in persistableFaces)
        {
            if (facesByKey.TryGetValue(preparedFace.FaceKey, out var existing))
            {
                existing.PrimarySourceKey ??= preparedFace.FaceKey;
                if (string.IsNullOrWhiteSpace(existing.Label))
                    existing.Label = Clean(preparedFace.Label) ?? preparedFace.FaceKey;
                existing.CustomFields = MergeCustomFields(existing.CustomFields, preparedFace);
                continue;
            }

            var created = new Face
            {
                Label = Clean(preparedFace.Label) ?? preparedFace.FaceKey,
                PrimarySourceKey = preparedFace.FaceKey,
                CustomFields = MergeCustomFields(null, preparedFace),
            };
            faceRepo.AddFace(created);
            facesByKey[preparedFace.FaceKey] = created;
        }

        await faceRepo.SaveChangesAsync(ct);
        return facesByKey;
    }

    private static bool IsPersistableFace(AiPreparedFaceIdentity face)
    {
        if (!face.IsProvisional) return true;
        if (face.Metadata is null) return false;
        return face.Metadata.TryGetValue("lifecycle", out var lifecycle) && lifecycle.Equals("promoted", StringComparison.OrdinalIgnoreCase);
    }

    private static int PersistFaceAppearances(IFaceRepository faceRepo, int hostEntityId, FaceAppearanceHostType hostType, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, AiDispatchRequest request)
    {
        var inserted = 0;
        foreach (var appearance in batch.FaceAppearances)
        {
            var faceId = ResolveFaceId(facesByKey, appearance.RefKey);
            if (faceId <= 0) continue;
            faceRepo.AddAppearance(new FaceAppearance
            {
                FaceId = faceId, HostType = hostType, HostId = hostEntityId,
                FirstSeenAtSec = appearance.FirstSeenSeconds, LastSeenAtSec = appearance.LastSeenSeconds,
                SampleCount = Math.Max(0, appearance.SampleCount),
                RetainedSpatialSampleCount = Math.Max(0, appearance.RetainedSpatialSampleCount),
                SegmentCount = Math.Max(0, appearance.SegmentCount),
                RepresentativeFrameSec = appearance.RepresentativeFrameSeconds,
                TopConfidence = appearance.TopConfidence is null ? null : (float)appearance.TopConfidence.Value,
                GroupKey = Clean(appearance.GroupKey),
                Payload = SerializeMetadata(appearance.Metadata, new Dictionary<string, string?> { ["assetId"] = appearance.AssetId, ["refKind"] = appearance.RefKind, ["refKey"] = appearance.RefKey, ["runId"] = request.Context.RunId }),
                SourceKey = appearance.SourceKey, SourceRunId = request.Context.RunId,
            });
            inserted++;
        }
        return inserted;
    }

    private static int PersistDetections(IDetectionRepository detectionRepo, int hostEntityId, DetectionHostType hostType, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, AiDispatchRequest request)
    {
        var inserted = 0;
        foreach (var detection in batch.Detections)
        {
            var refId = ResolveFaceId(facesByKey, detection.RefKey);
            if (refId <= 0) continue;
            detectionRepo.Add(new Detection
            {
                HostType = hostType, HostId = hostEntityId,
                ObservedAtSec = detection.ObservedAtSeconds,
                FrameWidth = NormalizedFrameSize, FrameHeight = NormalizedFrameSize,
                Class = Clean(detection.Class) ?? "face",
                Score = (float)detection.Score,
                X = (float)detection.BoundingBox.X1, Y = (float)detection.BoundingBox.Y1,
                W = (float)detection.BoundingBox.Width, H = (float)detection.BoundingBox.Height,
                Extra = SerializeMetadata(detection.Metadata, new Dictionary<string, string?> { ["assetId"] = detection.AssetId, ["modelKey"] = detection.ModelKey, ["refKey"] = detection.RefKey, ["runId"] = request.Context.RunId }),
                RefKind = "face", RefId = refId,
                GroupKey = Clean(detection.GroupKey),
                SourceKey = detection.SourceKey, SourceRunId = request.Context.RunId,
            });
            inserted++;
        }
        return inserted;
    }

    private static int PersistSegments(ISegmentRepository segmentRepo, int videoId, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, AiDispatchRequest request)
    {
        var inserted = 0;
        foreach (var segment in batch.Segments)
        {
            var refId = ResolveFaceId(facesByKey, segment.RefKey);
            var title = ResolveFaceLabel(facesByKey, segment.RefKey) ?? Clean(segment.Title) ?? segment.RefKey;
            segmentRepo.Add(new Segment
            {
                HostType = SegmentHostType.Video, HostId = videoId,
                StartSec = segment.StartSeconds, EndSec = segment.EndSeconds,
                Kind = Clean(segment.Kind), RefId = refId > 0 ? refId : null,
                Payload = SerializeMetadata(segment.Metadata, new Dictionary<string, string?> { ["assetId"] = segment.AssetId, ["refKind"] = segment.RefKind, ["refKey"] = segment.RefKey, ["runId"] = request.Context.RunId }),
                SourceKey = segment.SourceKey, SourceRunId = request.Context.RunId,
                Confidence = segment.Confidence is null ? null : (float)segment.Confidence.Value,
                Title = title,
            });
            inserted++;
        }
        return inserted;
    }

    private static int PersistEmbeddings(IEmbeddingRepository embeddingRepo, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, AiDispatchRequest request)
    {
        var inserted = 0;
        foreach (var embedding in batch.Embeddings)
        {
            var faceId = ResolveFaceId(facesByKey, embedding.HostRefKey);
            if (faceId <= 0) continue;
            embeddingRepo.Add(new Embedding
            {
                HostType = EmbeddingHostType.Face, HostId = faceId,
                Kind = embedding.Kind, KindFamily = Clean(embedding.KindFamily),
                Modality = EmbeddingModality.Face, IsSemantic = embedding.IsSemantic,
                Dim = embedding.Vector.Count, Vector = new Vector(embedding.Vector.ToArray()),
                SectionIndex = embedding.SectionIndex, StartSec = embedding.StartSeconds, EndSec = embedding.EndSeconds,
                SourceKey = embedding.SourceKey, SourceRunId = request.Context.RunId,
                Meta = SerializeMetadata(embedding.Metadata, new Dictionary<string, string?> { ["assetId"] = embedding.AssetId, ["hostRefKind"] = embedding.HostRefKind, ["hostRefKey"] = embedding.HostRefKey, ["modelKey"] = embedding.ModelKey, ["norm"] = embedding.Norm?.ToString(CultureInfo.InvariantCulture), ["runId"] = request.Context.RunId }),
            });
            inserted++;
        }
        return inserted;
    }

    private static async Task<int> PersistFaceCoversAsync(IServiceProvider services, ICustomFieldRepository customFieldRepo, IFaceRepository faceRepo, string hostEntityType, AiDispatchRequest request, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, CancellationToken ct)
    {
        var blobService = services.GetService<IBlobService>();
        if (blobService is null || batch.Faces.Count == 0) return 0;

        var configuration = services.GetService<CoveConfiguration>();
        var generated = 0;

        foreach (var preparedFace in batch.Faces)
        {
            if (!facesByKey.TryGetValue(preparedFace.FaceKey, out var face)) continue;

            var incomingCoverQuality = ReadPreparedCoverQuality(preparedFace);
            var currentCoverQuality = await customFieldRepo.FindNumberValueAsync(CustomFieldEntityTypes.Face, face.Id, CoverBlobQualityScoreField, ct);
            if (!ShouldGenerateCover(face, incomingCoverQuality, currentCoverQuality is null ? null : (double?)currentCoverQuality)) continue;

            var coverDetection = ResolveCoverDetection(batch, preparedFace);
            var coverFace = ResolveCoverFace(hostEntityType, request, preparedFace);

            // When the image cover source was substituted to the current host image (ResolveCoverFace,
            // images only), the crop box must come from a detection on THIS image too. The stored
            // CoverBoundingBox is the face's best-quality sample, which may belong to a *different* image
            // the face also appears in — cropping the current image at those coordinates lands on the
            // wrong region, the result is detected as blank, and the cover is silently skipped (leaving
            // the fingerprint placeholder). Videos don't hit this: they keep asset+box+timestamp together
            // with no substitution. Re-anchor the box to the current host's detection to keep them in sync.
            if (hostEntityType == "image"
                && coverDetection is not null
                && !string.Equals(coverFace.CoverAssetId, preparedFace.CoverAssetId, StringComparison.Ordinal))
            {
                coverFace = coverFace with { CoverBoundingBox = coverDetection.BoundingBox };
            }

            await using var coverStream = await AiFaceCoverGenerator.CreateAsync(hostEntityType, coverFace, coverDetection, configuration, ct);
            if (coverStream is null) continue;

            var previousBlobId = face.CoverBlobId;
            face.CoverBlobId = await blobService.StoreBlobAsync(coverStream, "image/jpeg", ct);
            face.CustomFields = SetCoverBlobQuality(face.CustomFields, incomingCoverQuality);

            if (incomingCoverQuality.HasValue)
            {
                var definition = await customFieldRepo.FindOrCreateDefinitionAsync(new CustomFieldDefinition
                {
                    Key = CoverBlobQualityScoreField, Label = "AI Faces Cover Quality",
                    Type = CustomFieldTypes.Number, EntityTypes = [CustomFieldEntityTypes.Face],
                    Filterable = false, Sortable = false, DisplayOrder = -1000,
                }, ct);
                await customFieldRepo.UpsertNumberValueAsync(CustomFieldEntityTypes.Face, face.Id, definition.Id, Convert.ToDecimal(incomingCoverQuality.Value, CultureInfo.InvariantCulture), ct);
            }

            if (!string.IsNullOrWhiteSpace(previousBlobId) && !string.Equals(previousBlobId, face.CoverBlobId, StringComparison.Ordinal))
                await blobService.DeleteBlobAsync(previousBlobId, ct);

            generated++;
        }

        return generated;
    }

    internal static async Task RefreshFaceStatsAsync(IFaceRepository faceRepo, IDetectionRepository detectionRepo, IReadOnlyCollection<int> faceIds, CancellationToken ct)
    {
        if (faceIds.Count == 0) return;

        var detections = await detectionRepo.FindAsync(new DetectionFilter
        {
            RefKind = "face",
            RefIds = faceIds.Select(static id => (long)id).ToArray(),
        }, ct);

        var retainedDetectionStatsByFaceId = detections
            .GroupBy(static d => (int)d.RefId!.Value)
            .ToDictionary(static g => g.Key, static g => new
            {
                DetectionCount = g.Count(),
                VideoCount = g.Where(static d => d.HostType == DetectionHostType.Video).Select(static d => d.HostId).Distinct().Count(),
                ImageCount = g.Where(static d => d.HostType == DetectionHostType.Image).Select(static d => d.HostId).Distinct().Count(),
            });

        var appearances = await faceRepo.FindAppearancesAsync(new FaceAppearanceFilter { FaceIds = faceIds.ToArray() }, ct);
        var appearanceStatsByFaceId = appearances
            .GroupBy(static a => a.FaceId)
            .ToDictionary(static g => g.Key, static g => new
            {
                FrameSampleCount = g.Sum(static a => a.SampleCount),
                VideoCount = g.Where(static a => a.HostType == FaceAppearanceHostType.Video).Select(static a => a.HostId).Distinct().Count(),
                ImageCount = g.Where(static a => a.HostType == FaceAppearanceHostType.Image).Select(static a => a.HostId).Distinct().Count(),
            });

        var trackedFaces = await faceRepo.FindFacesAsync(new FaceFilter { Ids = faceIds.ToArray() }, tracking: true, ct);
        foreach (var face in trackedFaces)
        {
            retainedDetectionStatsByFaceId.TryGetValue(face.Id, out var detectionStats);
            if (!appearanceStatsByFaceId.TryGetValue(face.Id, out var appearanceStats))
            {
                face.DetectionCount = detectionStats?.DetectionCount ?? 0;
                face.AppearanceCount = detectionStats is null ? 0 : detectionStats.VideoCount + detectionStats.ImageCount;
                face.FrameSampleCount = detectionStats?.DetectionCount ?? 0;
                face.VideoCount = detectionStats?.VideoCount ?? 0;
                face.ImageCount = detectionStats?.ImageCount ?? 0;
                continue;
            }

            face.DetectionCount = detectionStats?.DetectionCount ?? 0;
            // Appearance count is the number of distinct hosts (videos/images) the face appears in —
            // matching the "Appears In" list, the Videos/Images stats, and the no-appearance branch
            // above. A face can have several appearance rows per host (one per track), so the raw row
            // count would overstate it.
            face.AppearanceCount = appearanceStats.VideoCount + appearanceStats.ImageCount;
            face.FrameSampleCount = appearanceStats.FrameSampleCount;
            face.VideoCount = appearanceStats.VideoCount;
            face.ImageCount = appearanceStats.ImageCount;
        }
    }

    private static AiPreparedDetection? ResolveCoverDetection(AiPreparedArtifactBatch batch, AiPreparedFaceIdentity preparedFace)
    {
        var candidates = batch.Detections.Where(d => !string.IsNullOrWhiteSpace(d.RefKey) && d.RefKey.Equals(preparedFace.FaceKey, StringComparison.OrdinalIgnoreCase));
        if (preparedFace.CoverBoundingBox is { } coverBox)
        {
            var exact = candidates.Where(d => BoundingBoxesMatch(d.BoundingBox, coverBox)).OrderByDescending(static d => d.Score).FirstOrDefault();
            if (exact is not null) return exact;
        }
        return candidates.OrderByDescending(static d => d.Score).FirstOrDefault();
    }

    private static AiPreparedFaceIdentity ResolveCoverFace(string hostEntityType, AiDispatchRequest request, AiPreparedFaceIdentity preparedFace)
    {
        if (hostEntityType != "image"
            || string.IsNullOrWhiteSpace(preparedFace.CoverAssetId)
            || File.Exists(preparedFace.CoverAssetId)
            || string.IsNullOrWhiteSpace(request.Context.Subject)
            || !File.Exists(request.Context.Subject))
        {
            return preparedFace;
        }

        return preparedFace with { CoverAssetId = request.Context.Subject };
    }

    private static bool BoundingBoxesMatch(AiBoundingBox left, AiBoundingBox right)
        => Math.Abs(left.X1 - right.X1) <= 0.0001 && Math.Abs(left.Y1 - right.Y1) <= 0.0001 && Math.Abs(left.X2 - right.X2) <= 0.0001 && Math.Abs(left.Y2 - right.Y2) <= 0.0001;

    private static Dictionary<string, object>? MergeCustomFields(Dictionary<string, object>? current, AiPreparedFaceIdentity preparedFace)
    {
        var fields = current is null ? new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase) : new Dictionary<string, object>(current, StringComparer.OrdinalIgnoreCase);
        fields["ai.faces.faceKey"] = preparedFace.FaceKey;
        if (preparedFace.QualityScore.HasValue) fields["ai.faces.qualityScore"] = preparedFace.QualityScore.Value;
        if (!string.IsNullOrWhiteSpace(preparedFace.CoverAssetId)) fields["ai.faces.coverAssetId"] = preparedFace.CoverAssetId;
        if (preparedFace.CoverBoundingBox is { } box) fields["ai.faces.coverBoundingBox"] = new[] { box.X1, box.Y1, box.X2, box.Y2 };
        if (preparedFace.Metadata is not null) foreach (var (key, value) in preparedFace.Metadata) fields[$"ai.faces.{key}"] = value;
        var coverQuality = ReadPreparedCoverQuality(preparedFace);
        if (coverQuality.HasValue) fields[CoverQualityScoreField] = coverQuality.Value;
        return fields.Count == 0 ? null : fields;
    }

    private static bool ShouldGenerateCover(Face face, double? incomingCoverQuality, double? persistedCoverQuality)
    {
        if (string.IsNullOrWhiteSpace(face.CoverBlobId)) return true;
        if (!incomingCoverQuality.HasValue) return false;
        var currentBlobQuality = persistedCoverQuality ?? ReadCustomDouble(face.CustomFields, CoverBlobQualityScoreField);
        return !currentBlobQuality.HasValue || incomingCoverQuality.Value > currentBlobQuality.Value + CoverReplacementQualityMargin;
    }

    private static Dictionary<string, object>? SetCoverBlobQuality(Dictionary<string, object>? current, double? coverQualityScore)
    {
        if (!coverQualityScore.HasValue) return current;
        var fields = current is null ? new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase) : new Dictionary<string, object>(current, StringComparer.OrdinalIgnoreCase);
        fields[CoverBlobQualityScoreField] = coverQualityScore.Value;
        return fields;
    }

    private static double? ReadPreparedCoverQuality(AiPreparedFaceIdentity preparedFace)
        => preparedFace.Metadata is not null && preparedFace.Metadata.TryGetValue("coverQualityScore", out var value)
            && double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) ? parsed : null;

    private static double? ReadCustomDouble(Dictionary<string, object>? fields, string key)
    {
        if (fields is null || !fields.TryGetValue(key, out var value) || value is null) return null;
        return value switch
        {
            double d when double.IsFinite(d) => d,
            float f when float.IsFinite(f) => f,
            decimal dec => (double)dec,
            int i => i,
            long l => l,
            string s when double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var p) => p,
            JsonElement { ValueKind: JsonValueKind.Number } e when e.TryGetDouble(out var p) => p,
            JsonElement { ValueKind: JsonValueKind.String } e when double.TryParse(e.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var p) => p,
            _ => null,
        };
    }

    private static JsonDocument? SerializeMetadata(IReadOnlyDictionary<string, string>? metadata, IReadOnlyDictionary<string, string?>? extras = null)
    {
        var payload = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (metadata is not null) foreach (var (k, v) in metadata) { if (!string.IsNullOrWhiteSpace(k) && !string.IsNullOrWhiteSpace(v)) payload[k] = v; }
        if (extras is not null) foreach (var (k, v) in extras) { if (!string.IsNullOrWhiteSpace(k) && !string.IsNullOrWhiteSpace(v)) payload[k] = v; }
        return payload.Count == 0 ? null : JsonDocument.Parse(JsonSerializer.Serialize(payload));
    }

    private static int ResolveFaceId(IReadOnlyDictionary<string, Face> facesByKey, string? faceKey)
        => !string.IsNullOrWhiteSpace(faceKey) && facesByKey.TryGetValue(faceKey.Trim(), out var face) ? face.Id : 0;

    private static string? ResolveFaceLabel(IReadOnlyDictionary<string, Face> facesByKey, string? faceKey)
        => !string.IsNullOrWhiteSpace(faceKey) && facesByKey.TryGetValue(faceKey.Trim(), out var face)
            ? Clean(face.Label) ?? Clean(face.PrimarySourceKey) : null;

    private static string NormalizeHostEntityType(string hostEntityType)
        => hostEntityType.Trim().ToLowerInvariant() switch { "video" or "videos" => "video", "images" => "image", var n => n };

    private static string? Clean(string? value) => string.IsNullOrWhiteSpace(value) ? null : value.Trim();
}
