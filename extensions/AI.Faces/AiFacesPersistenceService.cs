using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;
using Cove.Data;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;

using Pgvector;

namespace AI.Faces;

internal sealed class AiFacesPersistenceService(IServiceScopeFactory scopeFactory)
{
    private const string FaceSourceKey = "ext:ai.faces";
    private const int NormalizedFrameSize = 1;
    private const string CoverQualityScoreField = "ai.faces.coverQualityScore";
    private const string CoverBlobQualityScoreField = "ai.faces.coverBlobQualityScore";
    private const double CoverReplacementQualityMargin = 0.02;

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;

    public async Task<IReadOnlyList<string>> PersistAsync(AiDispatchRequest request, AiPreparedArtifactBatch batch, CancellationToken ct = default)
    {
        if (request.Context.HostEntityId is null || string.IsNullOrWhiteSpace(request.Context.HostEntityType))
        {
            return ["AI.Faces prepared artifacts but skipped persistence because no Cove host entity identity was supplied."];
        }

        var hostEntityType = NormalizeHostEntityType(request.Context.HostEntityType);
        if (hostEntityType is not ("scene" or "image"))
        {
            return [$"AI.Faces persistence does not support host entity type '{request.Context.HostEntityType}'."];
        }

        await using var scope = _scopeFactory.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var hostEntityId = request.Context.HostEntityId.Value;
        var detectionHostType = hostEntityType == "scene" ? DetectionHostType.Scene : DetectionHostType.Image;
        var appearanceHostType = hostEntityType == "scene" ? FaceAppearanceHostType.Scene : FaceAppearanceHostType.Image;

        var facesByKey = await ResolveFacesAsync(db, batch.Faces, ct);
        var affectedFaceIds = new HashSet<int>(facesByKey.Values.Select(static face => face.Id));

        var existingAppearances = await db.FaceAppearances
            .Where(appearance => appearance.HostType == appearanceHostType && appearance.HostId == hostEntityId && appearance.SourceKey == FaceSourceKey)
            .ToListAsync(ct);
        foreach (var appearance in existingAppearances)
        {
            affectedFaceIds.Add(appearance.FaceId);
        }

        var existingDetections = await db.Detections
            .Where(detection => detection.HostType == detectionHostType && detection.HostId == hostEntityId && detection.SourceKey == FaceSourceKey)
            .ToListAsync(ct);
        foreach (var detection in existingDetections)
        {
            if (detection.RefKind is not null && detection.RefKind.Equals("face", StringComparison.OrdinalIgnoreCase) && detection.RefId is { } refId && refId > 0)
            {
                affectedFaceIds.Add((int)refId);
            }
        }

        var existingSegments = hostEntityType == "scene"
            ? await db.Segments
                .Where(segment => segment.HostType == SegmentHostType.Scene && segment.HostId == hostEntityId && segment.SourceKey == FaceSourceKey)
                .ToListAsync(ct)
            : [];
        foreach (var segment in existingSegments)
        {
            if (segment.RefId is { } refId && refId > 0)
            {
                affectedFaceIds.Add((int)refId);
            }
        }

        foreach (var detection in batch.Detections)
        {
            var faceId = ResolveFaceId(facesByKey, detection.RefKey);
            if (faceId > 0)
            {
                affectedFaceIds.Add(faceId);
            }
        }

        foreach (var appearance in batch.FaceAppearances)
        {
            var faceId = ResolveFaceId(facesByKey, appearance.RefKey);
            if (faceId > 0)
            {
                affectedFaceIds.Add(faceId);
            }
        }

        foreach (var segment in batch.Segments)
        {
            var faceId = ResolveFaceId(facesByKey, segment.RefKey);
            if (faceId > 0)
            {
                affectedFaceIds.Add(faceId);
            }
        }

        foreach (var embedding in batch.Embeddings)
        {
            var faceId = ResolveFaceId(facesByKey, embedding.HostRefKey);
            if (faceId > 0)
            {
                affectedFaceIds.Add(faceId);
            }
        }

        var existingEmbeddings = affectedFaceIds.Count == 0
            ? []
            : await db.Embeddings
                .Where(embedding => embedding.HostType == EmbeddingHostType.Face && affectedFaceIds.Contains(embedding.HostId) && embedding.SourceKey == FaceSourceKey)
                .ToListAsync(ct);

        if (existingDetections.Count > 0)
        {
            db.Detections.RemoveRange(existingDetections);
        }

        if (existingSegments.Count > 0)
        {
            db.Segments.RemoveRange(existingSegments);
        }

        if (existingEmbeddings.Count > 0)
        {
            db.Embeddings.RemoveRange(existingEmbeddings);
        }

        if (existingAppearances.Count > 0)
        {
            db.FaceAppearances.RemoveRange(existingAppearances);
        }

        var persistedAppearances = PersistFaceAppearances(db, hostEntityId, appearanceHostType, batch, facesByKey, request);
        var persistedDetections = PersistDetections(db, hostEntityId, detectionHostType, batch, facesByKey, request);
        var persistedSegments = hostEntityType == "scene"
            ? PersistSegments(db, hostEntityId, batch, facesByKey, request)
            : 0;
        var persistedEmbeddings = PersistEmbeddings(db, batch, facesByKey, request);
        var persistedFaceCovers = await PersistFaceCoversAsync(scope.ServiceProvider, hostEntityType, batch, facesByKey, ct);

        await db.SaveChangesAsync(ct);

        if (affectedFaceIds.Count > 0)
        {
            await RefreshFaceStatsAsync(db, affectedFaceIds, ct);
            await db.SaveChangesAsync(ct);
        }

        var notes = new List<string>();
        if (batch.Faces.Count > 0)
        {
            notes.Add($"Resolved {batch.Faces.Count} AI face identity candidate(s) into Cove face cluster(s).");
        }

        if (persistedAppearances > 0)
        {
            notes.Add($"Persisted {persistedAppearances} AI-generated face appearance(s) onto the {hostEntityType}.");
        }

        if (persistedDetections > 0)
        {
            notes.Add($"Persisted {persistedDetections} retained AI-generated face spatial sample(s) onto the {hostEntityType}.");
        }

        if (persistedSegments > 0)
        {
            notes.Add($"Persisted {persistedSegments} AI-generated face segment(s) onto the scene timeline.");
        }

        if (persistedEmbeddings > 0)
        {
            notes.Add($"Persisted {persistedEmbeddings} face embedding(s) for similarity and clustering workflows.");
        }

        if (persistedFaceCovers > 0)
        {
            notes.Add($"Generated {persistedFaceCovers} face cover image(s) for face detail pages.");
        }

        if (notes.Count == 0)
        {
            notes.Add(existingAppearances.Count + existingDetections.Count + existingSegments.Count + existingEmbeddings.Count > 0
                ? "AI.Faces cleared previously persisted rows for this host because the latest run did not emit any current face artifacts."
                : "AI.Faces found no new face rows to persist for this host entity.");
        }

        return notes;
    }

    private static async Task<Dictionary<string, Face>> ResolveFacesAsync(CoveContext db, IReadOnlyList<AiPreparedFaceIdentity> faces, CancellationToken ct)
    {
        var persistableFaces = faces
            .Where(IsPersistableFace)
            .ToArray();
        var faceKeys = persistableFaces
            .Select(static face => face.FaceKey)
            .Where(static faceKey => !string.IsNullOrWhiteSpace(faceKey))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();

        if (faceKeys.Length == 0)
        {
            return new Dictionary<string, Face>(StringComparer.OrdinalIgnoreCase);
        }

        var loweredKeys = faceKeys
            .Select(static faceKey => faceKey.Trim().ToLowerInvariant())
            .ToArray();

        var existingFaces = await db.Faces
            .Where(face => face.PrimarySourceKey != null && loweredKeys.Contains(face.PrimarySourceKey.ToLower()))
            .ToListAsync(ct);
        var facesByKey = new Dictionary<string, Face>(StringComparer.OrdinalIgnoreCase);

        foreach (var face in existingFaces)
        {
            if (!string.IsNullOrWhiteSpace(face.PrimarySourceKey) && !facesByKey.ContainsKey(face.PrimarySourceKey))
            {
                facesByKey[face.PrimarySourceKey] = face;
            }
        }

        foreach (var preparedFace in persistableFaces)
        {
            if (facesByKey.TryGetValue(preparedFace.FaceKey, out var existing))
            {
                existing.PrimarySourceKey ??= preparedFace.FaceKey;
                if (string.IsNullOrWhiteSpace(existing.Label))
                {
                    existing.Label = Clean(preparedFace.Label) ?? preparedFace.FaceKey;
                }

                existing.CustomFields = MergeCustomFields(existing.CustomFields, preparedFace);
                continue;
            }

            var created = new Face
            {
                Label = Clean(preparedFace.Label) ?? preparedFace.FaceKey,
                PrimarySourceKey = preparedFace.FaceKey,
                CustomFields = MergeCustomFields(null, preparedFace),
            };

            db.Faces.Add(created);
            facesByKey[preparedFace.FaceKey] = created;
        }

        if (db.ChangeTracker.HasChanges())
        {
            await db.SaveChangesAsync(ct);
        }

        return facesByKey;
    }

    private static bool IsPersistableFace(AiPreparedFaceIdentity face)
    {
        if (!face.IsProvisional)
        {
            return true;
        }

        if (face.Metadata is null)
        {
            return false;
        }

        return face.Metadata.TryGetValue("lifecycle", out var lifecycle)
               && lifecycle.Equals("promoted", StringComparison.OrdinalIgnoreCase);
    }

    private static int PersistFaceAppearances(CoveContext db, int hostEntityId, FaceAppearanceHostType hostType, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, AiDispatchRequest request)
    {
        var inserted = 0;
        foreach (var appearance in batch.FaceAppearances)
        {
            var faceId = ResolveFaceId(facesByKey, appearance.RefKey);
            if (faceId <= 0)
            {
                continue;
            }

            db.FaceAppearances.Add(new FaceAppearance
            {
                FaceId = faceId,
                HostType = hostType,
                HostId = hostEntityId,
                FirstSeenAtSec = appearance.FirstSeenSeconds,
                LastSeenAtSec = appearance.LastSeenSeconds,
                SampleCount = Math.Max(0, appearance.SampleCount),
                RetainedSpatialSampleCount = Math.Max(0, appearance.RetainedSpatialSampleCount),
                SegmentCount = Math.Max(0, appearance.SegmentCount),
                RepresentativeFrameSec = appearance.RepresentativeFrameSeconds,
                TopConfidence = appearance.TopConfidence is null ? null : (float)appearance.TopConfidence.Value,
                GroupKey = Clean(appearance.GroupKey),
                Payload = SerializeMetadata(appearance.Metadata, new Dictionary<string, string?>
                {
                    ["assetId"] = appearance.AssetId,
                    ["refKind"] = appearance.RefKind,
                    ["refKey"] = appearance.RefKey,
                    ["runId"] = request.Context.RunId,
                }),
                SourceKey = appearance.SourceKey,
                SourceRunId = request.Context.RunId,
            });
            inserted++;
        }

        return inserted;
    }

    private static int PersistDetections(CoveContext db, int hostEntityId, DetectionHostType hostType, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, AiDispatchRequest request)
    {
        var inserted = 0;
        foreach (var detection in batch.Detections)
        {
            var refId = ResolveFaceId(facesByKey, detection.RefKey);
            if (refId <= 0)
            {
                continue;
            }

            db.Detections.Add(new Detection
            {
                HostType = hostType,
                HostId = hostEntityId,
                ObservedAtSec = detection.ObservedAtSeconds,
                FrameWidth = NormalizedFrameSize,
                FrameHeight = NormalizedFrameSize,
                Class = Clean(detection.Class) ?? "face",
                Score = (float)detection.Score,
                X = (float)detection.BoundingBox.X1,
                Y = (float)detection.BoundingBox.Y1,
                W = (float)detection.BoundingBox.Width,
                H = (float)detection.BoundingBox.Height,
                Extra = SerializeMetadata(detection.Metadata, new Dictionary<string, string?>
                {
                    ["assetId"] = detection.AssetId,
                    ["modelKey"] = detection.ModelKey,
                    ["refKey"] = detection.RefKey,
                    ["runId"] = request.Context.RunId,
                }),
                RefKind = "face",
                RefId = refId,
                GroupKey = Clean(detection.GroupKey),
                SourceKey = detection.SourceKey,
                SourceRunId = request.Context.RunId,
            });
            inserted++;
        }

        return inserted;
    }

    private static int PersistSegments(CoveContext db, int sceneId, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, AiDispatchRequest request)
    {
        var inserted = 0;
        foreach (var segment in batch.Segments)
        {
            var refId = ResolveFaceId(facesByKey, segment.RefKey);
            var title = ResolveFaceLabel(facesByKey, segment.RefKey) ?? Clean(segment.Title) ?? segment.RefKey;

            db.Segments.Add(new Segment
            {
                HostType = SegmentHostType.Scene,
                HostId = sceneId,
                StartSec = segment.StartSeconds,
                EndSec = segment.EndSeconds,
                Kind = Clean(segment.Kind),
                RefId = refId > 0 ? refId : null,
                Payload = SerializeMetadata(segment.Metadata, new Dictionary<string, string?>
                {
                    ["assetId"] = segment.AssetId,
                    ["refKind"] = segment.RefKind,
                    ["refKey"] = segment.RefKey,
                    ["runId"] = request.Context.RunId,
                }),
                SourceKey = segment.SourceKey,
                SourceRunId = request.Context.RunId,
                Confidence = segment.Confidence is null ? null : (float)segment.Confidence.Value,
                Title = title,
            });
            inserted++;
        }

        return inserted;
    }

    private static int PersistEmbeddings(CoveContext db, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, AiDispatchRequest request)
    {
        var inserted = 0;
        foreach (var embedding in batch.Embeddings)
        {
            var faceId = ResolveFaceId(facesByKey, embedding.HostRefKey);
            if (faceId <= 0)
            {
                continue;
            }

            db.Embeddings.Add(new Embedding
            {
                HostType = EmbeddingHostType.Face,
                HostId = faceId,
                Kind = embedding.Kind,
                KindFamily = Clean(embedding.KindFamily),
                Modality = EmbeddingModality.Face,
                IsSemantic = embedding.IsSemantic,
                Dim = embedding.Vector.Count,
                Vector = new Vector(embedding.Vector.ToArray()),
                SectionIndex = embedding.SectionIndex,
                StartSec = embedding.StartSeconds,
                EndSec = embedding.EndSeconds,
                SourceKey = embedding.SourceKey,
                SourceRunId = request.Context.RunId,
                Meta = SerializeMetadata(embedding.Metadata, new Dictionary<string, string?>
                {
                    ["assetId"] = embedding.AssetId,
                    ["hostRefKind"] = embedding.HostRefKind,
                    ["hostRefKey"] = embedding.HostRefKey,
                    ["modelKey"] = embedding.ModelKey,
                    ["norm"] = embedding.Norm?.ToString(CultureInfo.InvariantCulture),
                    ["runId"] = request.Context.RunId,
                }),
            });
            inserted++;
        }

        return inserted;
    }

    private static async Task<int> PersistFaceCoversAsync(IServiceProvider services, string hostEntityType, AiPreparedArtifactBatch batch, IReadOnlyDictionary<string, Face> facesByKey, CancellationToken ct)
    {
        var blobService = services.GetService<IBlobService>();
        if (blobService is null || batch.Faces.Count == 0)
        {
            return 0;
        }

        var configuration = services.GetService<CoveConfiguration>();
        var generated = 0;

        foreach (var preparedFace in batch.Faces)
        {
            if (!facesByKey.TryGetValue(preparedFace.FaceKey, out var face))
            {
                continue;
            }

            var incomingCoverQuality = ReadPreparedCoverQuality(preparedFace);
            if (!ShouldGenerateCover(face, incomingCoverQuality))
            {
                continue;
            }

            var coverDetection = ResolveCoverDetection(batch, preparedFace);
            await using var coverStream = await AiFaceCoverGenerator.CreateAsync(hostEntityType, preparedFace, coverDetection, configuration, ct);
            if (coverStream is null)
            {
                continue;
            }

            var previousBlobId = face.CoverBlobId;
            face.CoverBlobId = await blobService.StoreBlobAsync(coverStream, "image/jpeg", ct);
            face.CustomFields = SetCoverBlobQuality(face.CustomFields, incomingCoverQuality);
            if (!string.IsNullOrWhiteSpace(previousBlobId) && !string.Equals(previousBlobId, face.CoverBlobId, StringComparison.Ordinal))
            {
                await blobService.DeleteBlobAsync(previousBlobId, ct);
            }

            generated++;
        }

        return generated;
    }

    private static AiPreparedDetection? ResolveCoverDetection(AiPreparedArtifactBatch batch, AiPreparedFaceIdentity preparedFace)
    {
        var candidates = batch.Detections
            .Where(detection => !string.IsNullOrWhiteSpace(detection.RefKey)
                && detection.RefKey.Equals(preparedFace.FaceKey, StringComparison.OrdinalIgnoreCase));

        if (preparedFace.CoverBoundingBox is { } coverBoundingBox)
        {
            var exactMatch = candidates
                .Where(detection => BoundingBoxesMatch(detection.BoundingBox, coverBoundingBox))
                .OrderByDescending(detection => detection.Score)
                .FirstOrDefault();
            if (exactMatch is not null)
            {
                return exactMatch;
            }
        }

        return candidates
            .OrderByDescending(detection => detection.Score)
            .FirstOrDefault();
    }

    private static bool BoundingBoxesMatch(AiBoundingBox left, AiBoundingBox right)
        => Math.Abs(left.X1 - right.X1) <= 0.0001
           && Math.Abs(left.Y1 - right.Y1) <= 0.0001
           && Math.Abs(left.X2 - right.X2) <= 0.0001
           && Math.Abs(left.Y2 - right.Y2) <= 0.0001;

    private static async Task RefreshFaceStatsAsync(CoveContext db, IReadOnlyCollection<int> faceIds, CancellationToken ct)
    {
        if (faceIds.Count == 0)
        {
            return;
        }

        var faceIdValues = faceIds.Select(static faceId => (long)faceId).ToArray();
        var detectionRows = await db.Detections
            .Where(detection =>
                detection.RefKind != null
                && detection.RefKind.ToLower() == "face"
                && detection.RefId.HasValue
                && faceIdValues.Contains(detection.RefId.Value))
            .Select(detection => new
            {
                FaceId = detection.RefId!.Value,
                detection.HostType,
                detection.HostId,
            })
            .ToListAsync(ct);

        var retainedDetectionStatsByFaceId = detectionRows
            .GroupBy(static row => (int)row.FaceId)
            .ToDictionary(
                static group => group.Key,
                static group => new
                {
                    DetectionCount = group.Count(),
                    SceneCount = group.Where(static row => row.HostType == DetectionHostType.Scene).Select(static row => row.HostId).Distinct().Count(),
                    ImageCount = group.Where(static row => row.HostType == DetectionHostType.Image).Select(static row => row.HostId).Distinct().Count(),
                });

        var appearanceRows = await db.FaceAppearances
            .Where(appearance => faceIds.Contains(appearance.FaceId))
            .Select(appearance => new
            {
                appearance.FaceId,
                appearance.HostType,
                appearance.HostId,
                appearance.SampleCount,
            })
            .ToListAsync(ct);

        var appearanceStatsByFaceId = appearanceRows
            .GroupBy(static row => row.FaceId)
            .ToDictionary(
                static group => group.Key,
                static group => new
                {
                    AppearanceCount = group.Count(),
                    FrameSampleCount = group.Sum(static row => row.SampleCount),
                    SceneCount = group.Where(static row => row.HostType == FaceAppearanceHostType.Scene).Select(static row => row.HostId).Distinct().Count(),
                    ImageCount = group.Where(static row => row.HostType == FaceAppearanceHostType.Image).Select(static row => row.HostId).Distinct().Count(),
                });

        var trackedFaces = await db.Faces
            .Where(face => faceIds.Contains(face.Id))
            .ToListAsync(ct);
        foreach (var face in trackedFaces)
        {
            retainedDetectionStatsByFaceId.TryGetValue(face.Id, out var detectionStats);
            if (!appearanceStatsByFaceId.TryGetValue(face.Id, out var appearanceStats))
            {
                face.DetectionCount = detectionStats?.DetectionCount ?? 0;
                face.AppearanceCount = detectionStats is null ? 0 : detectionStats.SceneCount + detectionStats.ImageCount;
                face.FrameSampleCount = detectionStats?.DetectionCount ?? 0;
                face.SceneCount = detectionStats?.SceneCount ?? 0;
                face.ImageCount = detectionStats?.ImageCount ?? 0;
                continue;
            }

            face.DetectionCount = detectionStats?.DetectionCount ?? 0;
            face.AppearanceCount = appearanceStats.AppearanceCount;
            face.FrameSampleCount = appearanceStats.FrameSampleCount;
            face.SceneCount = appearanceStats.SceneCount;
            face.ImageCount = appearanceStats.ImageCount;
        }
    }

    private static Dictionary<string, object>? MergeCustomFields(Dictionary<string, object>? current, AiPreparedFaceIdentity preparedFace)
    {
        var fields = current is null
            ? new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase)
            : new Dictionary<string, object>(current, StringComparer.OrdinalIgnoreCase);

        fields["ai.faces.faceKey"] = preparedFace.FaceKey;
        if (preparedFace.QualityScore.HasValue)
        {
            fields["ai.faces.qualityScore"] = preparedFace.QualityScore.Value;
        }

        if (!string.IsNullOrWhiteSpace(preparedFace.CoverAssetId))
        {
            fields["ai.faces.coverAssetId"] = preparedFace.CoverAssetId;
        }

        if (preparedFace.CoverBoundingBox is { } coverBoundingBox)
        {
            fields["ai.faces.coverBoundingBox"] = new[]
            {
                coverBoundingBox.X1,
                coverBoundingBox.Y1,
                coverBoundingBox.X2,
                coverBoundingBox.Y2,
            };
        }

        if (preparedFace.Metadata is not null)
        {
            foreach (var (key, value) in preparedFace.Metadata)
            {
                fields[$"ai.faces.{key}"] = value;
            }
        }

        var coverQualityScore = ReadPreparedCoverQuality(preparedFace);
        if (coverQualityScore.HasValue)
        {
            fields[CoverQualityScoreField] = coverQualityScore.Value;
        }

        return fields.Count == 0 ? null : fields;
    }

    private static bool ShouldGenerateCover(Face face, double? incomingCoverQuality)
    {
        if (string.IsNullOrWhiteSpace(face.CoverBlobId))
        {
            return true;
        }

        if (!incomingCoverQuality.HasValue)
        {
            return false;
        }

        var currentBlobQuality = ReadCustomDouble(face.CustomFields, CoverBlobQualityScoreField);
        return !currentBlobQuality.HasValue || incomingCoverQuality.Value > currentBlobQuality.Value + CoverReplacementQualityMargin;
    }

    private static Dictionary<string, object>? SetCoverBlobQuality(Dictionary<string, object>? current, double? coverQualityScore)
    {
        if (!coverQualityScore.HasValue)
        {
            return current;
        }

        var fields = current is null
            ? new Dictionary<string, object>(StringComparer.OrdinalIgnoreCase)
            : new Dictionary<string, object>(current, StringComparer.OrdinalIgnoreCase);
        fields[CoverBlobQualityScoreField] = coverQualityScore.Value;
        return fields;
    }

    private static double? ReadPreparedCoverQuality(AiPreparedFaceIdentity preparedFace)
        => preparedFace.Metadata is not null && preparedFace.Metadata.TryGetValue("coverQualityScore", out var value)
            && double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : null;

    private static double? ReadCustomDouble(Dictionary<string, object>? fields, string key)
    {
        if (fields is null || !fields.TryGetValue(key, out var value) || value is null)
        {
            return null;
        }

        return value switch
        {
            double doubleValue when double.IsFinite(doubleValue) => doubleValue,
            float floatValue when float.IsFinite(floatValue) => floatValue,
            decimal decimalValue => (double)decimalValue,
            int intValue => intValue,
            long longValue => longValue,
            string stringValue when double.TryParse(stringValue, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) => parsed,
            JsonElement { ValueKind: JsonValueKind.Number } element when element.TryGetDouble(out var parsed) => parsed,
            JsonElement { ValueKind: JsonValueKind.String } element when double.TryParse(element.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed) => parsed,
            _ => null,
        };
    }

    private static JsonDocument? SerializeMetadata(IReadOnlyDictionary<string, string>? metadata, IReadOnlyDictionary<string, string?>? extras = null)
    {
        var payload = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);

        if (metadata is not null)
        {
            foreach (var (key, value) in metadata)
            {
                if (!string.IsNullOrWhiteSpace(key) && !string.IsNullOrWhiteSpace(value))
                {
                    payload[key] = value;
                }
            }
        }

        if (extras is not null)
        {
            foreach (var (key, value) in extras)
            {
                if (!string.IsNullOrWhiteSpace(key) && !string.IsNullOrWhiteSpace(value))
                {
                    payload[key] = value;
                }
            }
        }

        return payload.Count == 0 ? null : JsonDocument.Parse(JsonSerializer.Serialize(payload));
    }

    private static int ResolveFaceId(IReadOnlyDictionary<string, Face> facesByKey, string? faceKey)
        => !string.IsNullOrWhiteSpace(faceKey) && facesByKey.TryGetValue(faceKey.Trim(), out var face) ? face.Id : 0;

    private static string? ResolveFaceLabel(IReadOnlyDictionary<string, Face> facesByKey, string? faceKey)
        => !string.IsNullOrWhiteSpace(faceKey) && facesByKey.TryGetValue(faceKey.Trim(), out var face)
            ? Clean(face.Label) ?? Clean(face.PrimarySourceKey)
            : null;

    private static string NormalizeHostEntityType(string hostEntityType)
        => hostEntityType.Trim().ToLowerInvariant() switch
        {
            "scenes" => "scene",
            "images" => "image",
            var normalized => normalized,
        };

    private static string? Clean(string? value)
        => string.IsNullOrWhiteSpace(value) ? null : value.Trim();
}