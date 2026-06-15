using System.Text.Json;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;

namespace AI.Core;

/// <summary>
/// Prunes AI.Core's run history when a host's faces are deleted. The host (Cove) reports, via the face
/// lifecycle, that a host no longer has any faces and which model keys were cleared; this extension — which
/// owns the <c>ext:ai.core</c> run records and the run <c>Models</c> shape — removes the matching models from
/// the host's completed runs so the planner stops treating that work as satisfied and a re-run redoes it.
/// Other capabilities in the same run keep their models (and skip via their still-present artifacts).
///
/// Resolves <see cref="IAiRunRepository"/> per call through a fresh scope (the repository is scoped/DbContext
/// backed) because this participant is published to the host as a singleton across the extension-isolation
/// boundary.
/// </summary>
internal sealed class AiCoreFaceRunEvidenceParticipant(IServiceScopeFactory scopeFactory) : IFaceLifecycleParticipant
{
    private const string SourceKey = "ext:ai.core";

    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;

    public Task OnDeletingAsync(Face face, CancellationToken cancellationToken = default) => Task.CompletedTask;

    public async Task OnHostFacesClearedAsync(FaceRunEvidenceCleared cleared, CancellationToken cancellationToken = default)
    {
        if (cleared.ModelKeys.Count == 0 || !TryResolveTarget(cleared.HostType, out var targetType))
            return;

        var modelKeys = cleared.ModelKeys
            .Where(key => !string.IsNullOrWhiteSpace(key))
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        if (modelKeys.Count == 0)
            return;

        using var scope = _scopeFactory.CreateScope();
        var runRepo = scope.ServiceProvider.GetRequiredService<IAiRunRepository>();

        var runs = await runRepo.GetCompletedAsync(targetType, cleared.HostId, SourceKey, cancellationToken);
        foreach (var run in runs)
        {
            if (TryStripModelsFromRun(run, modelKeys))
                await runRepo.UpdateAsync(run, cancellationToken);
        }
    }

    private static bool TryResolveTarget(DetectionHostType hostType, out AiRunTargetType targetType)
    {
        switch (hostType)
        {
            case DetectionHostType.Video:
                targetType = AiRunTargetType.Video;
                return true;
            case DetectionHostType.Image:
                targetType = AiRunTargetType.Image;
                return true;
            default:
                targetType = default;
                return false;
        }
    }

    // Removes any model entries matching <paramref name="modelKeys"/> from the run's stored models array.
    // Returns true (and rewrites run.Models) when at least one model was removed.
    private static bool TryStripModelsFromRun(AiRun run, IReadOnlySet<string> modelKeys)
    {
        if (run.Models is null || run.Models.RootElement.ValueKind != JsonValueKind.Array)
            return false;

        var kept = new List<JsonElement>();
        var removedAny = false;
        foreach (var model in run.Models.RootElement.EnumerateArray())
        {
            if (ModelMatches(model, modelKeys))
                removedAny = true;
            else
                kept.Add(model.Clone());
        }

        if (!removedAny)
            return false;

        run.Models = JsonSerializer.SerializeToDocument(kept);
        return true;
    }

    // The face lifecycle reports cleared work by the category the AI server keys results under
    // (e.g. "face_detections"/"face_embeddings"), which is also how the run's models are paired against
    // the planner's per-category slots. Match a run model when any of its config_name, name, or categories
    // is in the cleared set so the matching face evidence is actually removed.
    private static bool ModelMatches(JsonElement model, IReadOnlySet<string> modelKeys)
    {
        if (model.ValueKind != JsonValueKind.Object)
            return false;

        if (model.TryGetProperty("config_name", out var configElement) && configElement.GetString() is { } configName && modelKeys.Contains(configName))
            return true;
        if (model.TryGetProperty("name", out var nameElement) && nameElement.GetString() is { } name && modelKeys.Contains(name))
            return true;
        if (model.TryGetProperty("categories", out var categoriesElement) && categoriesElement.ValueKind == JsonValueKind.Array)
        {
            foreach (var category in categoriesElement.EnumerateArray())
            {
                if (category.GetString() is { } categoryName && modelKeys.Contains(categoryName))
                    return true;
            }
        }

        return false;
    }
}
