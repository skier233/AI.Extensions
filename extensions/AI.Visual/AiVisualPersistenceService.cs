using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

using Pgvector;

namespace AI.Visual;

internal sealed class AiVisualPersistenceService(IServiceScopeFactory scopeFactory)
{
    private const string VisualSourceKey = "ext:ai.visual";

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;

    public async Task<IReadOnlyList<string>> PersistAsync(AiDispatchRequest request, AiPreparedArtifactBatch batch, CancellationToken ct = default)
    {
        if (request.Context.HostEntityId is null || string.IsNullOrWhiteSpace(request.Context.HostEntityType))
        {
            return ["AI.Visual prepared embeddings but skipped persistence because no Cove host entity identity was supplied."];
        }

        var hostEntityType = NormalizeHostEntityType(request.Context.HostEntityType);
        if (hostEntityType is not ("video" or "image"))
        {
            return [$"AI.Visual persistence does not support host entity type '{request.Context.HostEntityType}'."];
        }

        await using var scope = _scopeFactory.CreateAsyncScope();
        var embeddingRepo = scope.ServiceProvider.GetRequiredService<IEmbeddingRepository>();
        var hostType = hostEntityType == "video" ? EmbeddingHostType.Video : EmbeddingHostType.Image;
        var hostId = request.Context.HostEntityId.Value;

        var existingEmbeddings = await embeddingRepo.FindAsync(new EmbeddingFilter
        {
            HostType = hostType,
            HostId = hostId,
            SourceKey = VisualSourceKey,
        }, ct);

        if (existingEmbeddings.Count > 0)
        {
            embeddingRepo.RemoveRange(existingEmbeddings);
        }

        var inserted = 0;
        foreach (var embedding in batch.Embeddings)
        {
            embeddingRepo.Add(new Embedding
            {
                HostType = hostType,
                HostId = hostId,
                Kind = embedding.Kind,
                KindFamily = Clean(embedding.KindFamily),
                Modality = EmbeddingModality.Visual,
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
            inserted++;
        }

        await embeddingRepo.SaveChangesAsync(ct);

        if (inserted > 0)
        {
            return [$"Persisted {inserted} AI-generated visual embedding(s) onto the {hostEntityType}."];
        }

        return [existingEmbeddings.Count > 0
            ? "AI.Visual cleared previously persisted embeddings for this host because the latest run did not emit any current visual embeddings."
            : "AI.Visual found no new embeddings to persist for this host entity."];
    }

    // Persists the embeddings for a whole batch of host entities in one scope and one SaveChanges. Existing
    // embeddings for every host are bulk-loaded and removed in a single query, then all new embeddings are
    // queued and flushed together — instead of a fresh scope + find + save per image, which is what made
    // large image runs spend minutes "persisting". Returns per-item notes aligned with the input order.
    public async Task<IReadOnlyList<IReadOnlyList<string>>> PersistBatchAsync(
        IReadOnlyList<(AiDispatchRequest Request, AiPreparedArtifactBatch Batch)> items,
        CancellationToken ct = default)
    {
        var notes = new List<string>[items.Count];
        for (var i = 0; i < items.Count; i++)
        {
            notes[i] = [];
        }

        var valid = new List<(int Index, EmbeddingHostType HostType, int HostId, string HostLabel)>();
        for (var i = 0; i < items.Count; i++)
        {
            var request = items[i].Request;
            if (request.Context.HostEntityId is null || string.IsNullOrWhiteSpace(request.Context.HostEntityType))
            {
                notes[i].Add("AI.Visual prepared embeddings but skipped persistence because no Cove host entity identity was supplied.");
                continue;
            }

            var hostEntityType = NormalizeHostEntityType(request.Context.HostEntityType);
            if (hostEntityType is not ("video" or "image"))
            {
                notes[i].Add($"AI.Visual persistence does not support host entity type '{request.Context.HostEntityType}'.");
                continue;
            }

            var hostType = hostEntityType == "video" ? EmbeddingHostType.Video : EmbeddingHostType.Image;
            valid.Add((i, hostType, request.Context.HostEntityId.Value, hostEntityType));
        }

        if (valid.Count == 0)
        {
            return notes.Select(static n => (IReadOnlyList<string>)n).ToArray();
        }

        await using var scope = _scopeFactory.CreateAsyncScope();
        var embeddingRepo = scope.ServiceProvider.GetRequiredService<IEmbeddingRepository>();

        // Bulk-remove existing visual embeddings for every host in the batch (one query per host-type).
        var hadExisting = new HashSet<int>();
        foreach (var typeGroup in valid.GroupBy(static v => v.HostType))
        {
            var hostIds = typeGroup.Select(static v => v.HostId).Distinct().ToList();
            var existing = await embeddingRepo.FindAsync(new EmbeddingFilter
            {
                HostType = typeGroup.Key,
                HostIds = hostIds,
                SourceKey = VisualSourceKey,
            }, ct);

            if (existing.Count > 0)
            {
                embeddingRepo.RemoveRange(existing);
                foreach (var embedding in existing)
                {
                    hadExisting.Add(embedding.HostId);
                }
            }
        }

        foreach (var entry in valid)
        {
            var request = items[entry.Index].Request;
            var batch = items[entry.Index].Batch;
            var inserted = 0;
            foreach (var embedding in batch.Embeddings)
            {
                embeddingRepo.Add(new Embedding
                {
                    HostType = entry.HostType,
                    HostId = entry.HostId,
                    Kind = embedding.Kind,
                    KindFamily = Clean(embedding.KindFamily),
                    Modality = EmbeddingModality.Visual,
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
                inserted++;
            }

            notes[entry.Index].Add(inserted > 0
                ? $"Persisted {inserted} AI-generated visual embedding(s) onto the {entry.HostLabel}."
                : hadExisting.Contains(entry.HostId)
                    ? "AI.Visual cleared previously persisted embeddings for this host because the latest run did not emit any current visual embeddings."
                    : "AI.Visual found no new embeddings to persist for this host entity.");
        }

        await embeddingRepo.SaveChangesAsync(ct);

        return notes.Select(static n => (IReadOnlyList<string>)n).ToArray();
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
