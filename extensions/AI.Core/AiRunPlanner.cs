using System.Globalization;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Data;

using Microsoft.EntityFrameworkCore;

namespace AI.Core;

public sealed record AiRunPlannerModel(
    string ModelKey,
    IReadOnlyList<string> ArtifactKeys,
    string? Category = null,
    int? Identifier = null,
    string? Version = null,
    string? Name = null);

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

internal sealed class AiRunPlanner(CoveContext dbContext) : IAiRunPlanner
{
    private readonly CoveContext _dbContext = dbContext;

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

        var completedRuns = await _dbContext.AiRuns
            .AsNoTracking()
            .Where(run => run.TargetType == targetType
                && run.TargetId == targetId
                && run.Status == AiRunStatus.Completed
                && run.SourceKey == "ext:ai.core")
            .OrderByDescending(run => run.CompletedAt ?? run.StartedAt)
            .ToListAsync(ct);

        var currentArtifactKeyCache = new Dictionary<string, HashSet<string>>(StringComparer.OrdinalIgnoreCase);
        var results = new List<AiRunExecutionPlan>(wants.Count);
        foreach (var want in wants)
        {
            var forced = OverlapsForce(want, forceSet);
            var claimIds = want.Claims.Select(static claim => claim.ClaimId).ToHashSet(StringComparer.OrdinalIgnoreCase);
            var relevantRuns = completedRuns
                .Where(run => HasAnyClaim(run, claimIds))
                .ToArray();

            var currentArtifactKeys = await GetCurrentArtifactKeysAsync(currentArtifactKeyCache, hostEntityType!, targetId, ResolveArtifactSourceKey(want.ExtensionId), ct);
            var perModel = want.Models.Select(model => PlanModel(settings, want, model, relevantRuns, currentArtifactKeys, forced, frameIntervalSeconds, threshold)).ToArray();

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
                    reasons.Length > 0 ? reasons : ["One or more required models are missing current artifacts for this host."],
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
                    ? ["Existing AI run records or current artifacts already satisfy this request."]
                    : reasonsForExecution.Length > 0 ? reasonsForExecution : ["One or more required models are not satisfied by historical AI runs or current artifacts for this host."],
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

    private PlannedModelDecision PlanModel(
        AiCoreConnectionSettings settings,
        AiRunPlannerWant want,
        AiRunPlannerModel desiredModel,
        IReadOnlyList<AiRun> relevantRuns,
        IReadOnlySet<string> currentArtifactKeys,
        bool forced,
        double? currentFrameIntervalSeconds,
        double? currentThreshold)
    {
        var desiredArtifactKeys = NormalizeArtifactKeys(desiredModel.ArtifactKeys, desiredModel.ModelKey);
        var exactArtifactsPresent = desiredArtifactKeys.All(currentArtifactKeys.Contains);
        var exactRun = relevantRuns.FirstOrDefault(run => RunHasModel(run, desiredModel.ModelKey));
        var modelKeyPresent = currentArtifactKeys.Contains(desiredModel.ModelKey);
        var historicalMatch = FindBestHistoricalMatch(desiredModel, relevantRuns);

        if (forced)
        {
            var shouldReplace = exactArtifactsPresent || historicalMatch is not null;
            return new PlannedModelDecision(
                desiredModel.ModelKey,
                shouldReplace ? AiRunPlanDecision.Rerun : AiRunPlanDecision.Run,
                exactArtifactsPresent ? desiredArtifactKeys : [],
                [shouldReplace
                    ? $"Force rerun requested for model '{desiredModel.ModelKey}'."
                    : $"Force rerun requested and no historical run or current artifacts were found for model '{desiredModel.ModelKey}'."]);
        }

        if (historicalMatch is not null && HistoricalModelSatisfies(desiredModel, historicalMatch, currentFrameIntervalSeconds, currentThreshold))
        {
            return new PlannedModelDecision(desiredModel.ModelKey, AiRunPlanDecision.Skip, [], []);
        }

        if (exactArtifactsPresent && (modelKeyPresent || exactRun is not null))
        {
            return new PlannedModelDecision(desiredModel.ModelKey, AiRunPlanDecision.Skip, [], []);
        }

        var supersedingMatch = relevantRuns
            .SelectMany(run => GetRunModels(run).Select(model => new { Run = run, Model = model }))
            .FirstOrDefault(candidate => IsSupersedingCandidate(settings, want, desiredModel, candidate.Model, currentArtifactKeys));
        if (supersedingMatch is not null)
        {
            return new PlannedModelDecision(
                desiredModel.ModelKey,
                AiRunPlanDecision.Skip,
                [],
                [$"Current artifacts from '{supersedingMatch.Model.ModelKey}' already supersede requested model '{desiredModel.ModelKey}'."]);
        }

        var staleMatch = relevantRuns
            .SelectMany(run => GetRunModels(run).Select(model => new { Run = run, Model = model }))
            .FirstOrDefault(candidate => IsReplacementCandidate(settings, want, desiredModel, candidate.Model, currentArtifactKeys));
        if (staleMatch is not null)
        {
            var replacementArtifactKeys = staleMatch.Model.ArtifactKeys
                .Where(currentArtifactKeys.Contains)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            var reason = desiredModel.Category is { Length: > 0 } category
                ? $"Category '{category}' is currently backed by '{staleMatch.Model.ModelKey}' and should be rerun with '{desiredModel.ModelKey}'."
                : $"Current artifacts were produced by '{staleMatch.Model.ModelKey}' and are superseded by '{desiredModel.ModelKey}'.";

            return new PlannedModelDecision(desiredModel.ModelKey, AiRunPlanDecision.Rerun, replacementArtifactKeys, [reason]);
        }

        return new PlannedModelDecision(
            desiredModel.ModelKey,
            AiRunPlanDecision.Run,
            [],
            [$"No completed AI run record or current artifact was found for model '{desiredModel.ModelKey}'."]);
    }

    private async Task<HashSet<string>> GetCurrentArtifactKeysAsync(
        IDictionary<string, HashSet<string>> cache,
        string hostEntityType,
        int hostEntityId,
        string sourceKey,
        CancellationToken ct)
    {
        var cacheKey = $"{hostEntityType}\u001F{hostEntityId}\u001F{sourceKey}";
        if (cache.TryGetValue(cacheKey, out var existing))
        {
            return existing;
        }

        var resolved = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var normalizedHostType = hostEntityType.Trim().ToLowerInvariant();

        if (normalizedHostType is "scene" or "image")
        {
            var affinityHostType = normalizedHostType == "scene" ? AffinityHostType.Scene : AffinityHostType.Image;
            var tagModels = await _dbContext.TagApplications
                .AsNoTracking()
                .Where(application => application.SourceKey == sourceKey && application.HostType == affinityHostType && application.HostId == hostEntityId)
                .Select(application => application.ModelKey)
                .ToListAsync(ct);
            foreach (var model in tagModels)
            {
                if (!string.IsNullOrWhiteSpace(model))
                {
                    resolved.Add(model.Trim());
                }
            }
        }

        if (normalizedHostType == "scene")
        {
            var segmentModels = await _dbContext.Segments
                .AsNoTracking()
                .Where(segment => segment.SourceKey == sourceKey && segment.HostType == SegmentHostType.Scene && segment.HostId == hostEntityId)
                .Select(segment => segment.Payload)
                .ToListAsync(ct);
            foreach (var payload in segmentModels)
            {
                if (TryExtractModelKey(payload, out var modelKey))
                {
                    resolved.Add(modelKey);
                }
            }
        }

        if (TryResolveEmbeddingHostType(normalizedHostType, out var embeddingHostType))
        {
            var embeddingModels = await _dbContext.Embeddings
                .AsNoTracking()
                .Where(embedding => embedding.SourceKey == sourceKey && embedding.HostType == embeddingHostType && embedding.HostId == hostEntityId)
                .Select(embedding => embedding.Meta)
                .ToListAsync(ct);
            foreach (var meta in embeddingModels)
            {
                if (TryExtractModelKey(meta, out var modelKey))
                {
                    resolved.Add(modelKey);
                }
            }
        }

        if (TryResolveDetectionHostType(normalizedHostType, out var detectionHostType))
        {
            var detectionModels = await _dbContext.Set<Detection>()
                .AsNoTracking()
                .Where(detection => detection.SourceKey == sourceKey && detection.HostType == detectionHostType && detection.HostId == hostEntityId)
                .Select(detection => detection.Extra)
                .ToListAsync(ct);
            foreach (var extra in detectionModels)
            {
                if (TryExtractModelKey(extra, out var modelKey))
                {
                    resolved.Add(modelKey);
                }
            }
        }

        if (normalizedHostType is "scene" or "image")
        {
            var appearanceHostType = normalizedHostType == "scene" ? FaceAppearanceHostType.Scene : FaceAppearanceHostType.Image;
            var faceAppearances = await _dbContext.FaceAppearances
                .AsNoTracking()
                .Where(appearance => appearance.SourceKey == sourceKey && appearance.HostType == appearanceHostType && appearance.HostId == hostEntityId)
                .Select(appearance => new { appearance.FaceId, appearance.Payload })
                .ToListAsync(ct);
            foreach (var payload in faceAppearances.Select(static appearance => appearance.Payload))
            {
                if (TryExtractModelKey(payload, out var modelKey))
                {
                    resolved.Add(modelKey);
                }
            }

            var faceIds = faceAppearances
                .Select(static appearance => appearance.FaceId)
                .Distinct()
                .ToArray();
            if (faceIds.Length > 0)
            {
                var faceEmbeddingModels = await _dbContext.Embeddings
                    .AsNoTracking()
                    .Where(embedding => embedding.SourceKey == sourceKey && embedding.HostType == EmbeddingHostType.Face && faceIds.Contains(embedding.HostId))
                    .Select(embedding => embedding.Meta)
                    .ToListAsync(ct);
                foreach (var meta in faceEmbeddingModels)
                {
                    if (TryExtractModelKey(meta, out var modelKey))
                    {
                        resolved.Add(modelKey);
                    }
                }
            }
        }

        cache[cacheKey] = resolved;
        return resolved;
    }

    private static IReadOnlyList<string> NormalizeArtifactKeys(IReadOnlyList<string> artifactKeys, string modelKey)
    {
        var resolved = artifactKeys
            .Where(static key => !string.IsNullOrWhiteSpace(key))
            .Select(static key => key.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();
        if (resolved.Count == 0)
        {
            resolved.Add(modelKey);
        }

        return resolved;
    }

    private static HistoricalModelInfo? FindBestHistoricalMatch(AiRunPlannerModel desiredModel, IReadOnlyList<AiRun> relevantRuns)
    {
        var desiredArtifactKeys = NormalizeArtifactKeys(desiredModel.ArtifactKeys, desiredModel.ModelKey);
        HistoricalModelInfo? best = null;

        foreach (var run in relevantRuns)
        {
            foreach (var model in GetRunModels(run))
            {
                if (!model.ArtifactKeys.Any(desiredArtifactKeys.Contains))
                {
                    continue;
                }

                var candidate = new HistoricalModelInfo(model, run.FrameIntervalSec, ExtractThreshold(run));
                if (best is null || !ShouldSkipComparison(best.ToComparison(), candidate.ToComparison()))
                {
                    best = candidate;
                }
            }
        }

        return best;
    }

    private static bool HistoricalModelSatisfies(
        AiRunPlannerModel desiredModel,
        HistoricalModelInfo historical,
        double? currentFrameIntervalSeconds,
        double? currentThreshold)
        => ShouldSkipComparison(
            historical.ToComparison(),
            new ModelComparison(
                desiredModel.Name ?? desiredModel.ModelKey,
                desiredModel.Identifier,
                desiredModel.Version,
                currentFrameIntervalSeconds,
                currentThreshold));

    private static bool ShouldSkipComparison(ModelComparison previous, ModelComparison current)
    {
        var previousThreshold = previous.Threshold ?? 0.5;
        var currentThreshold = current.Threshold ?? 0.5;
        if (!AreClose(previousThreshold, currentThreshold))
        {
            return false;
        }

        var previousFrameInterval = previous.FrameIntervalSeconds ?? 2.0;
        var currentFrameInterval = current.FrameIntervalSeconds ?? 2.0;
        if (!AreClose(previousFrameInterval, currentFrameInterval)
            && !IsCompatibleFrameInterval(previousFrameInterval, currentFrameInterval))
        {
            return false;
        }

        var versionComparison = CompareVersion(previous.Version, current.Version);
        if (versionComparison > 0)
        {
            return true;
        }

        if (versionComparison < 0)
        {
            return false;
        }

        if (string.Equals(previous.Name, current.Name, StringComparison.OrdinalIgnoreCase)
            && previous.Identifier == current.Identifier)
        {
            return true;
        }

        return previous.Identifier.HasValue
            && current.Identifier.HasValue
            && current.Identifier.Value >= previous.Identifier.Value;
    }

    private static int CompareVersion(string? previousVersion, string? currentVersion)
    {
        if (TryParseVersionNumber(previousVersion, out var previousNumeric)
            && TryParseVersionNumber(currentVersion, out var currentNumeric))
        {
            return previousNumeric.CompareTo(currentNumeric);
        }

        if (!string.IsNullOrWhiteSpace(previousVersion)
            && !string.IsNullOrWhiteSpace(currentVersion)
            && string.Equals(previousVersion.Trim(), currentVersion.Trim(), StringComparison.OrdinalIgnoreCase))
        {
            return 0;
        }

        return 0;
    }

    private static bool TryParseVersionNumber(string? rawVersion, out double value)
        => double.TryParse(rawVersion, NumberStyles.Float, CultureInfo.InvariantCulture, out value);

    private static bool AreClose(double left, double right)
        => Math.Abs(left - right) < 0.000001d;

    private static bool IsCompatibleFrameInterval(double previous, double current)
    {
        if (current <= 0d)
        {
            return false;
        }

        var multiple = previous / current;
        return Math.Abs(multiple - Math.Round(multiple)) < 0.000001d;
    }

    private static bool IsReplacementCandidate(
        AiCoreConnectionSettings settings,
        AiRunPlannerWant want,
        AiRunPlannerModel desiredModel,
        RunModelInfo existingModel,
        IReadOnlySet<string> currentArtifactKeys)
    {
        if (string.Equals(existingModel.ModelKey, desiredModel.ModelKey, StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        if (!existingModel.ArtifactKeys.Any(currentArtifactKeys.Contains))
        {
            return false;
        }

        if (desiredModel.ArtifactKeys.Any(existingModel.ArtifactKeys.Contains))
        {
            return true;
        }

        var desiredIndex = FindSupersessionIndex(settings, want, desiredModel.ModelKey, desiredModel.Category);
        var existingIndex = FindSupersessionIndex(settings, want, existingModel.ModelKey, desiredModel.Category);
        return desiredIndex > existingIndex && existingIndex >= 0;
    }

    private static bool IsSupersedingCandidate(
        AiCoreConnectionSettings settings,
        AiRunPlannerWant want,
        AiRunPlannerModel desiredModel,
        RunModelInfo existingModel,
        IReadOnlySet<string> currentArtifactKeys)
    {
        if (string.Equals(existingModel.ModelKey, desiredModel.ModelKey, StringComparison.OrdinalIgnoreCase)
            || !existingModel.ArtifactKeys.Any(currentArtifactKeys.Contains))
        {
            return false;
        }

        var desiredIndex = FindSupersessionIndex(settings, want, desiredModel.ModelKey, desiredModel.Category);
        var existingIndex = FindSupersessionIndex(settings, want, existingModel.ModelKey, desiredModel.Category);
        return existingIndex > desiredIndex && desiredIndex >= 0;
    }

    private static int FindSupersessionIndex(AiCoreConnectionSettings settings, AiRunPlannerWant want, string modelKey, string? category)
    {
        foreach (var rule in settings.ModelSupersessions)
        {
            if (!string.Equals(rule.Capability, want.Capability, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(rule.Scope, want.Scope, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!string.IsNullOrWhiteSpace(rule.Category)
                && !string.Equals(rule.Category, category, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            for (var index = 0; index < rule.Models.Count; index++)
            {
                if (string.Equals(rule.Models[index], modelKey, StringComparison.OrdinalIgnoreCase))
                {
                    return index;
                }
            }
        }

        return -1;
    }

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

    private static bool RunHasModel(AiRun run, string modelKey)
        => GetRunModels(run).Any(model => string.Equals(model.ModelKey, modelKey, StringComparison.OrdinalIgnoreCase));

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
                string? resolvedName = null;
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

                    if (model.TryGetProperty("name", out var resolvedNameElement) && resolvedNameElement.ValueKind == JsonValueKind.String)
                    {
                        resolvedName = resolvedNameElement.GetString();
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

                var artifactKeys = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                if (!string.IsNullOrWhiteSpace(resolvedModelKey))
                {
                    artifactKeys.Add(resolvedModelKey.Trim());
                }

                if (model.ValueKind == JsonValueKind.Object
                    && model.TryGetProperty("categories", out var categoriesElement)
                    && categoriesElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var category in categoriesElement.EnumerateArray().Select(static item => item.GetString()))
                    {
                        if (!string.IsNullOrWhiteSpace(category))
                        {
                            artifactKeys.Add(category.Trim());
                        }
                    }
                }

                return string.IsNullOrWhiteSpace(resolvedModelKey)
                    ? null
                    : new RunModelInfo(
                        resolvedModelKey.Trim(),
                        artifactKeys.ToArray(),
                        string.IsNullOrWhiteSpace(resolvedName) ? resolvedModelKey.Trim() : resolvedName.Trim(),
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

    private static bool TryExtractModelKey(JsonDocument? document, out string modelKey)
    {
        modelKey = string.Empty;
        if (document is null || document.RootElement.ValueKind != JsonValueKind.Object || !document.RootElement.TryGetProperty("modelKey", out var element))
        {
            return false;
        }

        var raw = element.GetString();
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        modelKey = raw.Trim();
        return true;
    }

    private static string ResolveArtifactSourceKey(string extensionId)
    {
        var normalized = (extensionId ?? string.Empty).Trim();
        if (normalized.StartsWith("ext:", StringComparison.OrdinalIgnoreCase))
        {
            return normalized;
        }

        return normalized.StartsWith("cove.ai.", StringComparison.OrdinalIgnoreCase)
            ? $"ext:ai.{normalized["cove.ai.".Length..]}"
            : normalized;
    }

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
            case "scene":
                targetType = AiRunTargetType.Scene;
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

    private static bool TryResolveEmbeddingHostType(string hostEntityType, out EmbeddingHostType hostType)
    {
        switch (hostEntityType)
        {
            case "scene":
                hostType = EmbeddingHostType.Scene;
                return true;
            case "image":
                hostType = EmbeddingHostType.Image;
                return true;
            case "face":
                hostType = EmbeddingHostType.Face;
                return true;
            default:
                hostType = default;
                return false;
        }
    }

    private static bool TryResolveDetectionHostType(string hostEntityType, out DetectionHostType hostType)
    {
        switch (hostEntityType)
        {
            case "scene":
                hostType = DetectionHostType.Scene;
                return true;
            case "image":
                hostType = DetectionHostType.Image;
                return true;
            default:
                hostType = default;
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
        IReadOnlyList<string> ArtifactKeys,
        string Name,
        int? Identifier,
        string? Version);

    private sealed record HistoricalModelInfo(RunModelInfo Model, double? FrameIntervalSeconds, double? Threshold)
    {
        public ModelComparison ToComparison()
            => new(Model.Name, Model.Identifier, Model.Version, FrameIntervalSeconds, Threshold);
    }

    private sealed record ModelComparison(
        string Name,
        int? Identifier,
        string? Version,
        double? FrameIntervalSeconds,
        double? Threshold);
}