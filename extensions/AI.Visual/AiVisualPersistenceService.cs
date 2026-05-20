using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Data;

using Microsoft.EntityFrameworkCore;
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
        if (hostEntityType is not ("scene" or "image"))
        {
            return [$"AI.Visual persistence does not support host entity type '{request.Context.HostEntityType}'."];
        }

        await using var scope = _scopeFactory.CreateAsyncScope();
        var db = scope.ServiceProvider.GetRequiredService<CoveContext>();
        var hostType = hostEntityType == "scene" ? EmbeddingHostType.Scene : EmbeddingHostType.Image;
        var hostId = request.Context.HostEntityId.Value;

        var existingEmbeddings = await db.Embeddings
            .Where(embedding => embedding.HostType == hostType && embedding.HostId == hostId && embedding.SourceKey == VisualSourceKey)
            .ToListAsync(ct);
        if (existingEmbeddings.Count > 0)
        {
            db.Embeddings.RemoveRange(existingEmbeddings);
        }

        var inserted = 0;
        foreach (var embedding in batch.Embeddings)
        {
            db.Embeddings.Add(new Embedding
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

        await db.SaveChangesAsync(ct);

        if (inserted > 0)
        {
            return [$"Persisted {inserted} AI-generated visual embedding(s) onto the {hostEntityType}."];
        }

        return [existingEmbeddings.Count > 0
            ? "AI.Visual cleared previously persisted embeddings for this host because the latest run did not emit any current visual embeddings."
            : "AI.Visual found no new embeddings to persist for this host entity."];
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