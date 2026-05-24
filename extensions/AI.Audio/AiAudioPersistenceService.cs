using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Data;

using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

using Pgvector;

namespace AI.Audio;

internal sealed class AiAudioPersistenceService(IServiceScopeFactory scopeFactory, ILogger<AiAudioPersistenceService> logger)
{
    private const string AudioSourceKey = "ext:ai.audio";

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;
    private readonly ILogger<AiAudioPersistenceService> _logger = logger;

    public async Task<IReadOnlyList<string>> PersistAsync(AiDispatchRequest request, AiPreparedArtifactBatch batch, CancellationToken ct = default)
    {
        if (request.Context.HostEntityId is null || string.IsNullOrWhiteSpace(request.Context.HostEntityType))
        {
            _logger.LogWarning(
                "AI.Audio skipped persistence for run {RunId} because the host identity was missing. HostType={HostType}, HostId={HostId}.",
                request.Context.RunId,
                request.Context.HostEntityType,
                request.Context.HostEntityId);
            return ["AI.Audio prepared artifacts but skipped persistence because no Cove host entity identity was supplied."];
        }

        var hostEntityType = NormalizeHostEntityType(request.Context.HostEntityType);
        if (hostEntityType != "scene")
        {
            _logger.LogWarning(
                "AI.Audio skipped persistence for run {RunId} because host entity type {HostEntityType} is unsupported.",
                request.Context.RunId,
                request.Context.HostEntityType);
            return [$"AI.Audio persistence does not support host entity type '{request.Context.HostEntityType}'."];
        }

        await using var scope = _scopeFactory.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var sceneId = request.Context.HostEntityId.Value;

        _logger.LogInformation(
            "AI.Audio persistence starting for run {RunId} scene {SceneId} with {EmbeddingCount} embedding(s) and {SegmentCount} segment(s).",
            request.Context.RunId,
            sceneId,
            batch.Embeddings.Count,
            batch.Segments.Count);

        var existingEmbeddings = await db.Embeddings
            .Where(embedding => embedding.HostType == EmbeddingHostType.Scene && embedding.HostId == sceneId && embedding.SourceKey == AudioSourceKey)
            .ToListAsync(ct);
        var existingSegments = await db.Segments
            .Where(segment => segment.HostType == SegmentHostType.Scene && segment.HostId == sceneId && segment.SourceKey == AudioSourceKey)
            .ToListAsync(ct);

        _logger.LogInformation(
            "AI.Audio found {ExistingEmbeddingCount} existing embedding row(s) and {ExistingSegmentCount} existing segment row(s) for scene {SceneId}.",
            existingEmbeddings.Count,
            existingSegments.Count,
            sceneId);

        if (existingEmbeddings.Count > 0)
        {
            db.Embeddings.RemoveRange(existingEmbeddings);
        }

        if (existingSegments.Count > 0)
        {
            db.Segments.RemoveRange(existingSegments);
        }

        var insertedEmbeddings = 0;
        foreach (var embedding in batch.Embeddings)
        {
            db.Embeddings.Add(new Embedding
            {
                HostType = EmbeddingHostType.Scene,
                HostId = sceneId,
                Kind = embedding.Kind,
                KindFamily = Clean(embedding.KindFamily),
                Modality = EmbeddingModality.Audio,
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
                    ["modelKey"] = embedding.ModelKey,
                    ["norm"] = embedding.Norm?.ToString(CultureInfo.InvariantCulture),
                    ["runId"] = request.Context.RunId,
                }),
            });
            insertedEmbeddings++;
        }

        var insertedSegments = 0;
        foreach (var segment in batch.Segments)
        {
            db.Segments.Add(new Segment
            {
                HostType = SegmentHostType.Scene,
                HostId = sceneId,
                StartSec = segment.StartSeconds,
                EndSec = segment.EndSeconds,
                Kind = Clean(segment.Kind),
                Payload = SerializeMetadata(segment.Metadata, new Dictionary<string, string?>
                {
                    ["assetId"] = segment.AssetId,
                    ["tagName"] = segment.TagName,
                    ["runId"] = request.Context.RunId,
                }),
                SourceKey = segment.SourceKey,
                SourceRunId = request.Context.RunId,
                Confidence = segment.Confidence is null ? null : (float)segment.Confidence.Value,
                Title = Clean(segment.Title) ?? Clean(segment.TagName),
            });
            insertedSegments++;
        }

        await db.SaveChangesAsync(ct);

        var notes = new List<string>();
        if (insertedEmbeddings > 0)
        {
            notes.Add($"Persisted {insertedEmbeddings} AI-generated audio embedding(s) onto the scene.");
        }

        if (insertedSegments > 0)
        {
            notes.Add($"Persisted {insertedSegments} AI-generated audio classification segment(s) onto the scene timeline.");
        }

        if (notes.Count == 0)
        {
            notes.Add(existingEmbeddings.Count + existingSegments.Count > 0
                ? "AI.Audio cleared previously persisted rows for this scene because the latest run did not emit any current audio artifacts."
                : "AI.Audio found no new rows to persist for this scene.");
        }

        _logger.LogInformation(
            "AI.Audio persistence finished for run {RunId} scene {SceneId}. Inserted embeddings={InsertedEmbeddingCount}, inserted segments={InsertedSegmentCount}. Notes: {Notes}",
            request.Context.RunId,
            sceneId,
            insertedEmbeddings,
            insertedSegments,
            string.Join(" | ", notes));

        return notes;
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