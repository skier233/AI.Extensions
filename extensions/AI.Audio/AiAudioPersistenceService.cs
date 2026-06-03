using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

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
        if (hostEntityType != "video")
        {
            _logger.LogWarning(
                "AI.Audio skipped persistence for run {RunId} because host entity type {HostEntityType} is unsupported.",
                request.Context.RunId,
                request.Context.HostEntityType);
            return [$"AI.Audio persistence does not support host entity type '{request.Context.HostEntityType}'."];
        }

        await using var scope = _scopeFactory.CreateAsyncScope();
        var embeddingRepo = scope.ServiceProvider.GetRequiredService<IEmbeddingRepository>();
        var segmentRepo = scope.ServiceProvider.GetRequiredService<ISegmentRepository>();
        var videoId = request.Context.HostEntityId.Value;

        _logger.LogInformation(
            "AI.Audio persistence starting for run {RunId} video {VideoId} with {EmbeddingCount} embedding(s) and {SegmentCount} segment(s).",
            request.Context.RunId,
            videoId,
            batch.Embeddings.Count,
            batch.Segments.Count);

        var existingEmbeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
        {
            HostType = EmbeddingHostType.Video,
            HostId = videoId,
            SourceKey = AudioSourceKey,
        }, ct);

        var existingSegments = await segmentRepo.FindAsync(new SegmentFilter
        {
            HostType = SegmentHostType.Video,
            HostId = videoId,
            SourceKey = AudioSourceKey,
        }, ct);

        _logger.LogInformation(
            "AI.Audio found {ExistingEmbeddingCount} existing embedding row(s) and {ExistingSegmentCount} existing segment row(s) for video {VideoId}.",
            existingEmbeddings.Count,
            existingSegments.Count,
            videoId);

        if (existingEmbeddings.Count > 0)
        {
            embeddingRepo.RemoveRange(existingEmbeddings);
        }

        if (existingSegments.Count > 0)
        {
            segmentRepo.RemoveRange(existingSegments);
        }

        var insertedEmbeddings = 0;
        foreach (var embedding in batch.Embeddings)
        {
            embeddingRepo.Add(new Embedding
            {
                HostType = EmbeddingHostType.Video,
                HostId = videoId,
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
            segmentRepo.Add(new Segment
            {
                HostType = SegmentHostType.Video,
                HostId = videoId,
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

        await embeddingRepo.SaveChangesAsync(ct);

        var notes = new List<string>();
        if (insertedEmbeddings > 0)
        {
            notes.Add($"Persisted {insertedEmbeddings} AI-generated audio embedding(s) onto the video.");
        }

        if (insertedSegments > 0)
        {
            notes.Add($"Persisted {insertedSegments} AI-generated audio classification segment(s) onto the video timeline.");
        }

        if (notes.Count == 0)
        {
            notes.Add(existingEmbeddings.Count + existingSegments.Count > 0
                ? "AI.Audio cleared previously persisted rows for this video because the latest run did not emit any current audio artifacts."
                : "AI.Audio found no new rows to persist for this video.");
        }

        _logger.LogInformation(
            "AI.Audio persistence finished for run {RunId} video {VideoId}. Inserted embeddings={InsertedEmbeddingCount}, inserted segments={InsertedSegmentCount}. Notes: {Notes}",
            request.Context.RunId,
            videoId,
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
            "video" or "videos" => "video",
            "images" => "image",
            var normalized => normalized,
        };

    private static string? Clean(string? value)
        => string.IsNullOrWhiteSpace(value) ? null : value.Trim();
}
