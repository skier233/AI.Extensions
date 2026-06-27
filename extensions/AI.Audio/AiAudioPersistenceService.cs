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
            "AI.Audio persistence starting for run {RunId} video {VideoId} with {EmbeddingCount} embedding(s).",
            request.Context.RunId,
            videoId,
            batch.Embeddings.Count);

        var existingEmbeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
        {
            HostType = EmbeddingHostType.Video,
            HostId = videoId,
            SourceKey = AudioSourceKey,
        }, ct);

        // AI.Audio no longer emits user-visible classification segments — its value is the speaker
        // embeddings. We still sweep any segments left by older runs so re-processing a video purges
        // the legacy short "moan/speech" rows from the timeline and segment library.
        var legacySegments = await segmentRepo.FindAsync(new SegmentFilter
        {
            HostType = SegmentHostType.Video,
            HostId = videoId,
            SourceKey = AudioSourceKey,
        }, ct);

        _logger.LogInformation(
            "AI.Audio found {ExistingEmbeddingCount} existing embedding row(s) and {LegacySegmentCount} legacy segment row(s) for video {VideoId}.",
            existingEmbeddings.Count,
            legacySegments.Count,
            videoId);

        if (existingEmbeddings.Count > 0)
        {
            embeddingRepo.RemoveRange(existingEmbeddings);
        }

        if (legacySegments.Count > 0)
        {
            segmentRepo.RemoveRange(legacySegments);
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

        await embeddingRepo.SaveChangesAsync(ct);
        if (legacySegments.Count > 0)
        {
            EvictVideoSegmentSpans(scope.ServiceProvider, videoId);
        }

        var notes = new List<string>();
        if (insertedEmbeddings > 0)
        {
            notes.Add($"Persisted {insertedEmbeddings} AI-generated audio embedding(s) onto the video.");
        }

        if (legacySegments.Count > 0)
        {
            notes.Add($"Removed {legacySegments.Count} legacy audio classification segment(s) from the video timeline.");
        }

        if (notes.Count == 0)
        {
            notes.Add(existingEmbeddings.Count > 0
                ? "AI.Audio cleared previously persisted embeddings for this video because the latest run did not emit any current audio artifacts."
                : "AI.Audio found no new rows to persist for this video.");
        }

        _logger.LogInformation(
            "AI.Audio persistence finished for run {RunId} video {VideoId}. Inserted embeddings={InsertedEmbeddingCount}, removed legacy segments={LegacySegmentCount}. Notes: {Notes}",
            request.Context.RunId,
            videoId,
            insertedEmbeddings,
            legacySegments.Count,
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

    private static void EvictVideoSegmentSpans(IServiceProvider services, int videoId)
    {
        var resolverType = Type.GetType("Cove.Data.Services.SegmentSpanResolver, Cove.Data", throwOnError: false);
        if (resolverType is null)
        {
            return;
        }

        var resolver = services.GetService(resolverType);
        if (resolver is null)
        {
            return;
        }

        var evictVideo = resolverType.GetMethod("EvictVideo", [typeof(int)]);
        evictVideo?.Invoke(resolver, [videoId]);
    }
}
