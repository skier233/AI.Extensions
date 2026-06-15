using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Core;

public sealed record AiRunPlannerModel(
    string ModelKey,
    IReadOnlyList<string> ArtifactKeys,
    string? Category = null,
    int? Identifier = null,
    string? Version = null,
    string? Name = null,
    IReadOnlyList<string>? Categories = null);

public sealed record AiRunPlannerWant(
    string ExtensionId,
    string Capability,
    string Scope,
    string? FromDetection,
    IReadOnlyList<AiCapabilityClaim> Claims,
    IReadOnlyList<AiRunPlannerModel> Models,
    bool AllowPartialExecution);

public sealed record AiRunExecutionPlan(
    string ExtensionId,
    string Capability,
    string Scope,
    string? FromDetection,
    IReadOnlyList<AiCapabilityClaim> Claims,
    IReadOnlyList<string> DesiredModels,
    IReadOnlyList<string> ExecutionModels,
    IReadOnlyList<string> ReplacementArtifactKeys,
    AiRunPlanDecision Decision,
    IReadOnlyList<string> Reasons,
    bool Forced);

public interface IAiRunPlanner
{
    Task<IReadOnlyList<AiRunExecutionPlan>> PlanAsync(
        AiCoreConnectionSettings settings,
        string? hostEntityType,
        int? hostEntityId,
        IReadOnlyList<AiRunPlannerWant> wants,
        IReadOnlyList<string>? forceClaimIds,
        double? frameIntervalSeconds = null,
        double? threshold = null,
        CancellationToken ct = default);
}

// Determines, per requested model, whether a prior AI run for the same Cove host
// already satisfies it. The decision is based ONLY on what a previous run recorded
// about its model — never on whether artifacts are currently present, since an AI run
// producing zero faces/tags/etc. is a valid outcome we must not keep re-running.
//
// A requested model is paired with the most recent prior run that produced the same
// model_category (the "slot"). Given that pair, we decide:
//   * higher model_version wins (newer dataset) -> rerun to replace
//   * same version, lower model_identifier wins (a lower id is a larger/more accurate
//     variant) -> rerun to replace
//   * exact same model -> only rerun if the prior run does not cover the requested run
//     parameters (a different threshold, or a frame interval the prior run cannot be
//     evenly subsampled into).
internal sealed class AiRunPlanner(IAiRunRepository runRepo) : IAiRunPlanner
{
    private readonly IAiRunRepository _runRepo = runRepo;

    private const double Tolerance = 1e-6;

    public async Task<IReadOnlyList<AiRunExecutionPlan>> PlanAsync(
        AiCoreConnectionSettings settings,
        string? hostEntityType,
        int? hostEntityId,
        IReadOnlyList<AiRunPlannerWant> wants,
        IReadOnlyList<string>? forceClaimIds,
        double? frameIntervalSeconds = null,
        double? threshold = null,
        CancellationToken ct = default)
    {
        if (wants.Count == 0)
        {
            return [];
        }

        var forceSet = new HashSet<string>(
            (forceClaimIds ?? []).Where(static item => !string.IsNullOrWhiteSpace(item)).Select(static item => item.Trim()),
            StringComparer.OrdinalIgnoreCase);

        if (!TryResolveTarget(hostEntityType, hostEntityId, out var targetType, out var targetId))
        {
            return wants.Select(want => BuildRunPlan(want, forced: OverlapsForce(want, forceSet), ["No persisted Cove host identity was supplied, so sufficiency cannot be evaluated."])).ToArray();
        }

        var completedRuns = (await _runRepo.GetCompletedAsync(targetType, targetId, "ext:ai.core", ct) ?? [])
            .OrderByDescending(run => run.CompletedAt ?? run.StartedAt)
            .ToList();

        var results = new List<AiRunExecutionPlan>(wants.Count);
        foreach (var want in wants)
        {
            var forced = OverlapsForce(want, forceSet);
            var claimIds = want.Claims.Select(static claim => claim.ClaimId).ToHashSet(StringComparer.OrdinalIgnoreCase);
            var relevantRuns = completedRuns
                .Where(run => HasAnyClaim(run, claimIds))
                .ToArray();

            var perModel = want.Models
                .Select(model => PlanModel(model, relevantRuns, forced, frameIntervalSeconds, threshold))
                .ToArray();

            if (!want.AllowPartialExecution && perModel.Any(static item => item.Decision != AiRunPlanDecision.Skip))
            {
                var executionModels = want.Models.Select(static model => model.ModelKey).Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
                var replacementArtifactKeys = perModel
                    .Where(static item => item.Decision == AiRunPlanDecision.Rerun)
                    .SelectMany(static item => item.ReplacementArtifactKeys)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToArray();
                var reasons = perModel
                    .Where(static item => item.Decision != AiRunPlanDecision.Skip)
                    .SelectMany(static item => item.Reasons)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToArray();

                results.Add(new AiRunExecutionPlan(
                    want.ExtensionId,
                    want.Capability,
                    want.Scope,
                    want.FromDetection,
                    want.Claims,
                    want.Models.Select(static model => model.ModelKey).ToArray(),
                    executionModels,
                    replacementArtifactKeys,
                    replacementArtifactKeys.Length > 0 ? AiRunPlanDecision.Rerun : AiRunPlanDecision.Run,
                    reasons.Length > 0 ? reasons : ["One or more required models are not satisfied by a prior AI run for this host."],
                    forced));
                continue;
            }

            var execution = perModel
                .Where(static item => item.Decision != AiRunPlanDecision.Skip)
                .Select(static item => item.ModelKey)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            var replacements = perModel
                .Where(static item => item.Decision == AiRunPlanDecision.Rerun)
                .SelectMany(static item => item.ReplacementArtifactKeys)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            var reasonsForExecution = perModel
                .Where(static item => item.Decision != AiRunPlanDecision.Skip)
                .SelectMany(static item => item.Reasons)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            results.Add(new AiRunExecutionPlan(
                want.ExtensionId,
                want.Capability,
                want.Scope,
                want.FromDetection,
                want.Claims,
                want.Models.Select(static model => model.ModelKey).ToArray(),
                execution,
                replacements,
                execution.Length == 0
                    ? AiRunPlanDecision.Skip
                    : replacements.Length > 0 ? AiRunPlanDecision.Rerun : AiRunPlanDecision.Run,
                execution.Length == 0
                    ? ["A prior AI run already satisfies this request."]
                    : reasonsForExecution.Length > 0 ? reasonsForExecution : ["One or more required models are not satisfied by a prior AI run for this host."],
                forced));
        }

        return results;
    }

    private static AiRunExecutionPlan BuildRunPlan(AiRunPlannerWant want, bool forced, IReadOnlyList<string> reasons)
        => new(
            want.ExtensionId,
            want.Capability,
            want.Scope,
            want.FromDetection,
            want.Claims,
            want.Models.Select(static model => model.ModelKey).ToArray(),
            want.Models.Select(static model => model.ModelKey).ToArray(),
            [],
            AiRunPlanDecision.Run,
            reasons,
            forced);

    private static PlannedModelDecision PlanModel(
        AiRunPlannerModel desiredModel,
        IReadOnlyList<AiRun> relevantRuns,
        bool forced,
        double? requestedFrameIntervalSeconds,
        double? requestedThreshold)
    {
        // Pair the requested model against the most recent prior run for each of its
        // categories (the per-category slot for tagging, the face_detections /
        // face_embeddings slot for faces, etc.).
        var slotPrevious = GetSlotKeys(desiredModel)
            .Select(slot => FindMostRecentPrevious(relevantRuns, slot))
            .ToArray();

        if (forced)
        {
            var forcedReplacements = slotPrevious
                .Where(static previous => previous is not null)
                .SelectMany(static previous => previous!.ReplacementKeys)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            return new PlannedModelDecision(
                desiredModel.ModelKey,
                forcedReplacements.Length > 0 ? AiRunPlanDecision.Rerun : AiRunPlanDecision.Run,
                forcedReplacements,
                [$"Force rerun requested for model '{desiredModel.ModelKey}'."]);
        }

        var replacementKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var reasons = new List<string>();
        var needsRun = false;

        foreach (var previous in slotPrevious)
        {
            if (previous is null)
            {
                needsRun = true;
                reasons.Add($"No prior AI run produced model '{desiredModel.ModelKey}' for this host.");
                continue;
            }

            if (DecideAgainstPrevious(desiredModel, previous, requestedFrameIntervalSeconds, requestedThreshold, out var reason) == AiRunPlanDecision.Skip)
            {
                continue;
            }

            needsRun = true;
            foreach (var key in previous.ReplacementKeys)
            {
                replacementKeys.Add(key);
            }

            reasons.Add(reason);
        }

        if (!needsRun)
        {
            return new PlannedModelDecision(desiredModel.ModelKey, AiRunPlanDecision.Skip, [], []);
        }

        return new PlannedModelDecision(
            desiredModel.ModelKey,
            replacementKeys.Count > 0 ? AiRunPlanDecision.Rerun : AiRunPlanDecision.Run,
            replacementKeys.ToArray(),
            reasons.Count > 0 ? reasons : [$"Model '{desiredModel.ModelKey}' is not satisfied by a prior AI run for this host."]);
    }

    // Skip / Rerun verdict for a requested model against one prior run for the same slot.
    private static AiRunPlanDecision DecideAgainstPrevious(
        AiRunPlannerModel desiredModel,
        PreviousModel previous,
        double? requestedFrameIntervalSeconds,
        double? requestedThreshold,
        out string reason)
    {
        var quality = CompareModelQuality(desiredModel, previous);
        if (quality > 0)
        {
            reason = $"Requested model '{desiredModel.ModelKey}' supersedes prior model '{previous.ModelKey}' (newer version or larger variant).";
            return AiRunPlanDecision.Rerun;
        }

        if (quality < 0)
        {
            reason = string.Empty;
            return AiRunPlanDecision.Skip;
        }

        // Same model. Re-run only if the prior run does not cover the requested run parameters.
        if (!ThresholdMatches(previous.Threshold, requestedThreshold))
        {
            reason = $"Requested threshold differs from the prior run for model '{desiredModel.ModelKey}'.";
            return AiRunPlanDecision.Rerun;
        }

        if (!FrameIntervalCovers(previous.FrameIntervalSeconds, requestedFrameIntervalSeconds))
        {
            reason = $"Requested frame interval is not covered by the prior run for model '{desiredModel.ModelKey}'.";
            return AiRunPlanDecision.Rerun;
        }

        reason = string.Empty;
        return AiRunPlanDecision.Skip;
    }

    // > 0 : the requested model is preferred (rerun)   < 0 : the prior model is preferred (skip)   0 : same model.
    // Higher model_version wins; on a version tie the lower model_identifier wins
    // (a lower identifier denotes a larger / more accurate model variant).
    private static int CompareModelQuality(AiRunPlannerModel desired, PreviousModel previous)
    {
        var desiredVersion = ParseVersion(desired.Version);
        var previousVersion = ParseVersion(previous.Version);
        if (!AreClose(desiredVersion, previousVersion))
        {
            return desiredVersion > previousVersion ? 1 : -1;
        }

        var desiredIdentifier = desired.Identifier ?? int.MaxValue;
        var previousIdentifier = previous.Identifier ?? int.MaxValue;
        if (desiredIdentifier == previousIdentifier)
        {
            return 0;
        }

        return desiredIdentifier < previousIdentifier ? 1 : -1;
    }

    // The prior run's frame interval must evenly divide into the requested one, i.e. the
    // requested sampling is a whole-number multiple of what we already have (e.g. a prior
    // 1.1s run covers a 2.2s request, but not a 2.0s request). Equal intervals always cover.
    private static bool FrameIntervalCovers(double? previousInterval, double? requestedInterval)
    {
        var previous = previousInterval ?? 0d;
        var requested = requestedInterval ?? 0d;

        if (previous <= 0d || requested <= 0d)
        {
            return AreClose(previous, requested);
        }

        var remainder = requested % previous;
        return remainder < Tolerance || previous - remainder < Tolerance;
    }

    private static IReadOnlyList<string> GetSlotKeys(AiRunPlannerModel model)
    {
        var categories = (model.Categories ?? [])
            .Where(static category => !string.IsNullOrWhiteSpace(category))
            .Select(static category => category.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToArray();
        return categories.Length > 0 ? categories : [model.ModelKey];
    }

    private static PreviousModel? FindMostRecentPrevious(IReadOnlyList<AiRun> relevantRuns, string slot)
    {
        // relevantRuns is ordered most-recent first, so the first match reflects the
        // model whose output currently occupies this slot.
        foreach (var run in relevantRuns)
        {
            foreach (var model in GetRunModels(run))
            {
                var modelSlots = model.Categories.Count > 0 ? model.Categories : [model.ModelKey];
                if (modelSlots.Contains(slot, StringComparer.OrdinalIgnoreCase))
                {
                    return new PreviousModel(
                        model.ModelKey,
                        model.Identifier,
                        model.Version,
                        run.FrameIntervalSec,
                        ExtractThreshold(run),
                        BuildReplacementKeys(model));
                }
            }
        }

        return null;
    }

    // Keys the prior run's artifacts may be stored under, so a replacing run can clear them.
    // Faces store the model config_name; tagging stores the category — include both.
    private static IReadOnlyList<string> BuildReplacementKeys(RunModelInfo model)
    {
        var keys = new HashSet<string>(StringComparer.OrdinalIgnoreCase) { model.ModelKey };
        foreach (var category in model.Categories)
        {
            keys.Add(category);
        }

        return keys.ToArray();
    }

    // A different threshold means a re-run. If either side has no recorded threshold we
    // can't claim a difference, so we treat it as a match rather than churn.
    private static bool ThresholdMatches(double? previousThreshold, double? requestedThreshold)
        => !previousThreshold.HasValue || !requestedThreshold.HasValue || AreClose(previousThreshold.Value, requestedThreshold.Value);

    private static double ParseVersion(string? rawVersion)
        => double.TryParse(rawVersion, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) ? value : 0d;

    private static bool AreClose(double left, double right)
        => Math.Abs(left - right) < Tolerance;

    private static bool HasAnyClaim(AiRun run, IReadOnlySet<string> claimIds)
    {
        if (run.Summary is null || !run.Summary.RootElement.TryGetProperty("claimIds", out var claimIdElement) || claimIdElement.ValueKind != JsonValueKind.Array)
        {
            return false;
        }

        return claimIdElement.EnumerateArray()
            .Select(static item => item.GetString())
            .Where(static item => !string.IsNullOrWhiteSpace(item))
            .Any(claimId => claimId is not null && claimIds.Contains(claimId));
    }

    private static IReadOnlyList<RunModelInfo> GetRunModels(AiRun run)
    {
        if (run.Models is null || run.Models.RootElement.ValueKind != JsonValueKind.Array)
        {
            return [];
        }

        return run.Models.RootElement.EnumerateArray()
            .Select(static model =>
            {
                string? resolvedModelKey = null;
                int? resolvedIdentifier = null;
                string? resolvedVersion = null;
                if (model.ValueKind == JsonValueKind.Object)
                {
                    if (model.TryGetProperty("config_name", out var configNameElement))
                    {
                        resolvedModelKey = configNameElement.GetString();
                    }

                    if (string.IsNullOrWhiteSpace(resolvedModelKey) && model.TryGetProperty("name", out var nameElement))
                    {
                        resolvedModelKey = nameElement.GetString();
                    }

                    if (model.TryGetProperty("identifier", out var identifierElement))
                    {
                        if (identifierElement.ValueKind == JsonValueKind.Number && identifierElement.TryGetInt32(out var identifier))
                        {
                            resolvedIdentifier = identifier;
                        }
                        else if (identifierElement.ValueKind == JsonValueKind.String
                            && int.TryParse(identifierElement.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out identifier))
                        {
                            resolvedIdentifier = identifier;
                        }
                    }

                    if (model.TryGetProperty("version", out var versionElement) && versionElement.ValueKind != JsonValueKind.Null)
                    {
                        resolvedVersion = versionElement.ToString();
                    }
                }

                var categories = new List<string>();
                if (model.ValueKind == JsonValueKind.Object
                    && model.TryGetProperty("categories", out var categoriesElement)
                    && categoriesElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var category in categoriesElement.EnumerateArray().Select(static item => item.GetString()))
                    {
                        if (!string.IsNullOrWhiteSpace(category))
                        {
                            categories.Add(category.Trim());
                        }
                    }
                }

                return string.IsNullOrWhiteSpace(resolvedModelKey)
                    ? null
                    : new RunModelInfo(
                        resolvedModelKey.Trim(),
                        categories,
                        resolvedIdentifier,
                        string.IsNullOrWhiteSpace(resolvedVersion) ? null : resolvedVersion.Trim());
            })
            .Where(static model => model is not null)
            .Cast<RunModelInfo>()
            .ToArray();
    }

    private static double? ExtractThreshold(AiRun run)
    {
        if (run.Request is null || run.Request.RootElement.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        if (!TryGetPropertyIgnoreCase(run.Request.RootElement, "threshold", out var thresholdElement))
        {
            return null;
        }

        if (thresholdElement.ValueKind == JsonValueKind.Number && thresholdElement.TryGetDouble(out var threshold))
        {
            return threshold;
        }

        return thresholdElement.ValueKind == JsonValueKind.String
            && double.TryParse(thresholdElement.GetString(), NumberStyles.Float, CultureInfo.InvariantCulture, out threshold)
            ? threshold
            : null;
    }

    private static bool TryGetPropertyIgnoreCase(JsonElement element, string propertyName, out JsonElement property)
    {
        foreach (var candidate in element.EnumerateObject())
        {
            if (string.Equals(candidate.Name, propertyName, StringComparison.OrdinalIgnoreCase))
            {
                property = candidate.Value;
                return true;
            }
        }

        property = default;
        return false;
    }

    private static bool OverlapsForce(AiRunPlannerWant want, IReadOnlySet<string> forceSet)
        => forceSet.Count > 0 && want.Claims.Select(static claim => claim.ClaimId).Any(forceSet.Contains);

    private static bool TryResolveTarget(string? hostEntityType, int? hostEntityId, out AiRunTargetType targetType, out int targetId)
    {
        targetType = default;
        targetId = default;
        if (!hostEntityId.HasValue || string.IsNullOrWhiteSpace(hostEntityType))
        {
            return false;
        }

        switch (hostEntityType.Trim().ToLowerInvariant())
        {
            case "video":
                targetType = AiRunTargetType.Video;
                targetId = hostEntityId.Value;
                return true;
            case "image":
                targetType = AiRunTargetType.Image;
                targetId = hostEntityId.Value;
                return true;
            case "performer":
                targetType = AiRunTargetType.Performer;
                targetId = hostEntityId.Value;
                return true;
            case "face":
                targetType = AiRunTargetType.Face;
                targetId = hostEntityId.Value;
                return true;
            default:
                return false;
        }
    }

    private sealed record PlannedModelDecision(
        string ModelKey,
        AiRunPlanDecision Decision,
        IReadOnlyList<string> ReplacementArtifactKeys,
        IReadOnlyList<string> Reasons);

    private sealed record RunModelInfo(
        string ModelKey,
        IReadOnlyList<string> Categories,
        int? Identifier,
        string? Version);

    private sealed record PreviousModel(
        string ModelKey,
        int? Identifier,
        string? Version,
        double? FrameIntervalSeconds,
        double? Threshold,
        IReadOnlyList<string> ReplacementKeys);
}
