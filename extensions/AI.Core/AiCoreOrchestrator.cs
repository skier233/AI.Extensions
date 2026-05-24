using System.Text.Json;

using AI.Extensions.Abstractions;

using Microsoft.Extensions.Logging;

namespace AI.Core;

public interface IAiCoreOrchestrator
{
    IReadOnlyList<AiCapabilityDescriptor> GetCapabilities();

    Task<AiRunResponse> RunImagesAsync(AiCoreConnectionSettings settings, AiRunImagesRequest request, CancellationToken ct = default);

    Task<AiRunResponse> RunVideoAsync(AiCoreConnectionSettings settings, AiRunVideoRequest request, CancellationToken ct = default);

    Task<AiRunResponse> RunAudioAsync(AiCoreConnectionSettings settings, AiRunAudioRequest request, CancellationToken ct = default);
}

public sealed class AiCoreOrchestrator(
    INsfwAiServerClient aiServerClient,
    IEnumerable<IAiCapabilityContributor> contributors,
    IAiRunJournal aiRunJournal,
    IAiRunPlanner aiRunPlanner,
    IAiArtifactReplaceService aiArtifactReplaceService,
    ILogger<AiCoreOrchestrator> logger) : IAiCoreOrchestrator
{
    private readonly INsfwAiServerClient _aiServerClient = aiServerClient;
    private readonly IAiRunJournal _aiRunJournal = aiRunJournal;
    private readonly IAiRunPlanner _aiRunPlanner = aiRunPlanner;
    private readonly IAiArtifactReplaceService _aiArtifactReplaceService = aiArtifactReplaceService;
    private readonly ILogger<AiCoreOrchestrator> _logger = logger;
    private readonly IReadOnlyList<ResolvedContributor> _contributors = contributors
        .Select(static contributor => new ResolvedContributor(contributor, contributor.Describe()))
        .OrderBy(static item => item.Descriptor.ExtensionId, StringComparer.OrdinalIgnoreCase)
        .ToArray();

    public IReadOnlyList<AiCapabilityDescriptor> GetCapabilities()
        => _contributors.Select(static item => item.Descriptor).ToArray();

    public async Task<AiRunResponse> RunImagesAsync(AiCoreConnectionSettings settings, AiRunImagesRequest request, CancellationToken ct = default)
    {
        if (request.Paths.Count == 0)
        {
            throw new ArgumentException("At least one image path is required.", nameof(request));
        }

        var runId = Guid.NewGuid().ToString("n");
        var mappedPaths = AiPathMapper.MapPaths(settings.PathMappings, request.Paths);
        var selection = ResolveRunSelection(settings, AiMediaKinds.Image, request.PresetId, request.CapabilityIds, request.ClaimIds, request.CategoriesToSkip, request.LoadPolicy, request.PipelineName);
        var claims = SelectClaims(AiMediaKinds.Image, selection.ClaimIds, selection.CapabilityIds);
        var resolvedLoadPolicy = ResolveLoadPolicy(settings, selection.LoadPolicy);
        var wants = await BuildWantAsync(settings, claims, selection.CategoriesToSkip, resolvedLoadPolicy, ct);
        var plans = await _aiRunPlanner.PlanAsync(
            settings,
            request.EntityType,
            request.EntityId,
            wants.Select(static want => want.ToPlannerWant()).ToArray(),
            request.ForceClaimIds,
            frameIntervalSeconds: null,
            threshold: request.Threshold ?? settings.DefaultThreshold,
            ct: ct);
        var responsePlan = BuildResponsePlan(plans);
        var execution = BuildExecution(wants, plans);
        if (execution.Wants.Count == 0)
        {
            return new AiRunResponse(
                runId,
                AiMediaKinds.Image,
                claims.Select(static item => item.Claim).ToArray(),
                CreateSkippedAnalysis(AiMediaKinds.Image, $"{request.Paths.Count} image(s)"),
                [],
                responsePlan);
        }

        var analyzeRequest = new ImageAnalyzeRequest
        {
            Paths = mappedPaths.ToList(),
            Threshold = request.Threshold ?? settings.DefaultThreshold,
            ReturnConfidence = request.ReturnConfidence ?? true,
            CategoriesToSkip = selection.CategoriesToSkip?.ToList(),
            Want = execution.Wants.ToList(),
            LoadPolicy = resolvedLoadPolicy,
            PipelineName = selection.PipelineName,
        };

        await _aiRunJournal.RecordStartAsync(
            new AiRunJournalStart(runId, request.EntityType, request.EntityId, "AI.Core", resolvedLoadPolicy, null, null, request),
            ct);

        try
        {
            var response = await _aiServerClient.AnalyzeImagesAsync(settings, analyzeRequest, ct);

            await _aiArtifactReplaceService.ReplaceAsync(request.EntityType, request.EntityId, plans, ct);

            var dispatchResults = await MaybeDispatchAsync(
                settings,
                request.DispatchResults,
                AiMediaKinds.Image,
                $"{request.Paths.Count} image(s)",
                runId,
                request.EntityType,
                request.EntityId,
                execution.Claims,
                response,
                ct);

            await _aiRunJournal.RecordCompletionAsync(
                new AiRunJournalCompletion(runId, AiMediaKinds.Image, response, execution.Claims.Select(static item => item.Claim.ClaimId).ToArray(), dispatchResults.Count),
                ct);

            return new AiRunResponse(runId, AiMediaKinds.Image, claims.Select(static item => item.Claim).ToArray(), response, dispatchResults, responsePlan);
        }
        catch (Exception ex)
        {
            await _aiRunJournal.RecordFailureAsync(runId, ex, ct);
            throw;
        }
    }

    public async Task<AiRunResponse> RunVideoAsync(AiCoreConnectionSettings settings, AiRunVideoRequest request, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(request.Path))
        {
            throw new ArgumentException("A video path is required.", nameof(request));
        }

        var runId = Guid.NewGuid().ToString("n");
        var mappedPath = AiPathMapper.MapPath(settings.PathMappings, request.Path);
        var selection = ResolveRunSelection(settings, AiMediaKinds.Video, request.PresetId, request.CapabilityIds, request.ClaimIds, request.CategoriesToSkip, request.LoadPolicy, request.PipelineName);
        var claims = SelectClaims(AiMediaKinds.Video, selection.ClaimIds, selection.CapabilityIds);
        var resolvedLoadPolicy = ResolveLoadPolicy(settings, selection.LoadPolicy);
        var wants = await BuildWantAsync(settings, claims, selection.CategoriesToSkip, resolvedLoadPolicy, ct);
        var plans = await _aiRunPlanner.PlanAsync(
            settings,
            request.EntityType,
            request.EntityId,
            wants.Select(static want => want.ToPlannerWant()).ToArray(),
            request.ForceClaimIds,
            frameIntervalSeconds: request.FrameInterval,
            threshold: request.Threshold ?? settings.DefaultThreshold,
            ct: ct);
        var responsePlan = BuildResponsePlan(plans);
        var execution = BuildExecution(wants, plans);
        if (execution.Wants.Count == 0)
        {
            return new AiRunResponse(
                runId,
                AiMediaKinds.Video,
                claims.Select(static item => item.Claim).ToArray(),
                CreateSkippedAnalysis(AiMediaKinds.Video, request.Path),
                [],
                responsePlan);
        }

        var analyzeRequest = new VideoAnalyzeRequest
        {
            Path = mappedPath,
            FrameInterval = request.FrameInterval,
            Threshold = request.Threshold ?? settings.DefaultThreshold,
            ReturnConfidence = request.ReturnConfidence ?? true,
            VrVideo = request.VrVideo,
            CategoriesToSkip = selection.CategoriesToSkip?.ToList(),
            Want = execution.Wants.ToList(),
            LoadPolicy = resolvedLoadPolicy,
            PipelineName = selection.PipelineName,
        };

        await _aiRunJournal.RecordStartAsync(
            new AiRunJournalStart(runId, request.EntityType, request.EntityId, "AI.Core", resolvedLoadPolicy, request.FrameInterval, request.VrVideo, request),
            ct);

        try
        {
            var response = await _aiServerClient.AnalyzeVideoAsync(settings, analyzeRequest, ct);

            await _aiArtifactReplaceService.ReplaceAsync(request.EntityType, request.EntityId, plans, ct);

            var dispatchResults = await MaybeDispatchAsync(
                settings,
                request.DispatchResults,
                AiMediaKinds.Video,
                request.Path,
                runId,
                request.EntityType,
                request.EntityId,
                execution.Claims,
                response,
                ct);

            await _aiRunJournal.RecordCompletionAsync(
                new AiRunJournalCompletion(runId, AiMediaKinds.Video, response, execution.Claims.Select(static item => item.Claim.ClaimId).ToArray(), dispatchResults.Count),
                ct);

            return new AiRunResponse(runId, AiMediaKinds.Video, claims.Select(static item => item.Claim).ToArray(), response, dispatchResults, responsePlan);
        }
        catch (Exception ex)
        {
            await _aiRunJournal.RecordFailureAsync(runId, ex, ct);
            throw;
        }
    }

    public async Task<AiRunResponse> RunAudioAsync(AiCoreConnectionSettings settings, AiRunAudioRequest request, CancellationToken ct = default)
    {
        if (request.Paths.Count == 0)
        {
            throw new ArgumentException("At least one audio path is required.", nameof(request));
        }

        var runId = Guid.NewGuid().ToString("n");
        var mappedPaths = AiPathMapper.MapPaths(settings.PathMappings, request.Paths);
        var selection = ResolveRunSelection(settings, AiMediaKinds.Audio, request.PresetId, request.CapabilityIds, request.ClaimIds, categoriesToSkip: null, request.LoadPolicy, request.PipelineName);
        var claims = SelectClaims(AiMediaKinds.Audio, selection.ClaimIds, selection.CapabilityIds);
        var resolvedLoadPolicy = ResolveLoadPolicy(settings, selection.LoadPolicy);
        var wants = await BuildWantAsync(settings, claims, categoriesToSkip: null, resolvedLoadPolicy, ct);
        var plans = await _aiRunPlanner.PlanAsync(
            settings,
            request.EntityType,
            request.EntityId,
            wants.Select(static want => want.ToPlannerWant()).ToArray(),
            request.ForceClaimIds,
            frameIntervalSeconds: null,
            threshold: request.Threshold ?? settings.DefaultThreshold,
            ct: ct);
        var responsePlan = BuildResponsePlan(plans);
        var execution = BuildExecution(wants, plans);
        if (execution.Wants.Count == 0)
        {
            return new AiRunResponse(
                runId,
                AiMediaKinds.Audio,
                claims.Select(static item => item.Claim).ToArray(),
                CreateSkippedAnalysis(AiMediaKinds.Audio, $"{request.Paths.Count} audio file(s)"),
                [],
                responsePlan);
        }

        var analyzeRequest = new AudioAnalyzeRequest
        {
            Paths = mappedPaths.ToList(),
            Threshold = request.Threshold ?? settings.DefaultThreshold,
            Want = execution.Wants.ToList(),
            LoadPolicy = resolvedLoadPolicy,
            PipelineName = selection.PipelineName,
        };

        await _aiRunJournal.RecordStartAsync(
            new AiRunJournalStart(runId, request.EntityType, request.EntityId, "AI.Core", resolvedLoadPolicy, null, null, request),
            ct);

        try
        {
            var response = await _aiServerClient.AnalyzeAudioAsync(settings, analyzeRequest, ct);

            await _aiArtifactReplaceService.ReplaceAsync(request.EntityType, request.EntityId, plans, ct);

            var dispatchResults = await MaybeDispatchAsync(
                settings,
                request.DispatchResults,
                AiMediaKinds.Audio,
                $"{request.Paths.Count} audio file(s)",
                runId,
                request.EntityType,
                request.EntityId,
                execution.Claims,
                response,
                ct);

            await _aiRunJournal.RecordCompletionAsync(
                new AiRunJournalCompletion(runId, AiMediaKinds.Audio, response, execution.Claims.Select(static item => item.Claim.ClaimId).ToArray(), dispatchResults.Count),
                ct);

            return new AiRunResponse(runId, AiMediaKinds.Audio, claims.Select(static item => item.Claim).ToArray(), response, dispatchResults, responsePlan);
        }
        catch (Exception ex)
        {
            await _aiRunJournal.RecordFailureAsync(runId, ex, ct);
            throw;
        }
    }

    private RunSelection ResolveRunSelection(
        AiCoreConnectionSettings settings,
        string mediaKind,
        string? presetId,
        IReadOnlyList<string>? requestCapabilityIds,
        IReadOnlyList<string>? requestClaimIds,
        IReadOnlyList<string>? categoriesToSkip,
        string? loadPolicy,
        string? pipelineName)
    {
        var preset = string.IsNullOrWhiteSpace(presetId)
            ? null
            : settings.RunPresets.FirstOrDefault(item => string.Equals(item.PresetId, presetId.Trim(), StringComparison.OrdinalIgnoreCase));
        if (!string.IsNullOrWhiteSpace(presetId) && preset is null)
        {
            throw new ArgumentException($"Unknown AI run preset '{presetId}'.", nameof(presetId));
        }

        var capabilityIds = NormalizeStringList(requestCapabilityIds);
        if (capabilityIds.Count == 0 && preset?.CapabilityIds is { Count: > 0 } presetCapabilityIds)
        {
            capabilityIds = NormalizeStringList(presetCapabilityIds);
        }

        var claimIds = NormalizeStringList(requestClaimIds);
        if (claimIds.Count == 0 && preset?.ClaimIds is { Count: > 0 } presetClaimIds)
        {
            claimIds = NormalizeStringList(presetClaimIds);
        }

        var resolvedPipelineName = string.IsNullOrWhiteSpace(pipelineName) ? preset?.PipelineName : pipelineName.Trim();
        var customCapabilityIds = settings.CustomPipelines
            .Where(pipeline => !string.IsNullOrWhiteSpace(pipeline.CapabilityId) && capabilityIds.Contains(pipeline.CapabilityId, StringComparer.OrdinalIgnoreCase))
            .ToArray();
        if (customCapabilityIds.Length > 1)
        {
            throw new ArgumentException("Only one custom AI pipeline capability can be selected per run.", nameof(requestCapabilityIds));
        }

        if (customCapabilityIds.Length == 1)
        {
            var customPipeline = customCapabilityIds[0];
            if (!string.Equals(customPipeline.MediaKind, mediaKind, StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"Custom pipeline '{customPipeline.PipelineName}' supports media kind '{customPipeline.MediaKind}', not '{mediaKind}'.", nameof(requestCapabilityIds));
            }

            resolvedPipelineName ??= customPipeline.PipelineName;
            capabilityIds = capabilityIds
                .Where(id => !string.Equals(id, customPipeline.CapabilityId, StringComparison.OrdinalIgnoreCase))
                .Concat(customPipeline.CapabilityIds)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
            claimIds = claimIds
                .Concat(customPipeline.ClaimIds)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
        }

        return new RunSelection(
            capabilityIds.Count == 0 ? null : capabilityIds,
            claimIds.Count == 0 ? null : claimIds,
            categoriesToSkip is null ? preset?.CategoriesToSkip : NormalizeStringList(categoriesToSkip),
            string.IsNullOrWhiteSpace(loadPolicy) ? preset?.LoadPolicy : loadPolicy.Trim(),
            resolvedPipelineName);
    }

    private IReadOnlyList<ResolvedClaim> SelectClaims(string mediaKind, IReadOnlyList<string>? requestedClaimIds, IReadOnlyList<string>? requestedCapabilityIds)
    {
        var contributorClaims = _contributors
            .SelectMany(static contributor => contributor.Descriptor.Claims.Select(claim => new ResolvedClaim(contributor.Contributor, contributor.Descriptor, claim)))
            .ToArray();
        var allClaims = contributorClaims
            .Where(claim => string.Equals(claim.Claim.MediaKind, mediaKind, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        var mediaClaimIds = allClaims
            .Select(static claim => claim.Claim.ClaimId)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);

        var requestedFeatures = new HashSet<string>(
            requestedCapabilityIds?.Where(static item => !string.IsNullOrWhiteSpace(item)).Select(static item => item.Trim()) ?? [],
            StringComparer.OrdinalIgnoreCase);
        var selectedClaimIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var unresolvedClaimIds = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var claimId in requestedClaimIds?.Where(static item => !string.IsNullOrWhiteSpace(item)).Select(static item => item.Trim()) ?? [])
        {
            if (mediaClaimIds.Contains(claimId))
            {
                selectedClaimIds.Add(claimId);
                continue;
            }

            var translatedCapabilityId = contributorClaims
                .Where(claim => string.Equals(claim.Claim.ClaimId, claimId, StringComparison.OrdinalIgnoreCase))
                .Select(claim => claim.Claim.CapabilityId)
                .FirstOrDefault(static capabilityId => !string.IsNullOrWhiteSpace(capabilityId));
            if (!string.IsNullOrWhiteSpace(translatedCapabilityId))
            {
                requestedFeatures.Add(translatedCapabilityId.Trim());
                continue;
            }

            unresolvedClaimIds.Add(claimId);
        }

        if (requestedFeatures.Count > 0)
        {
            var knownFeatures = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var contributor in _contributors)
            {
                foreach (var feature in contributor.Descriptor.Capabilities)
                {
                    knownFeatures.Add(feature.CapabilityId);
                    if (!requestedFeatures.Contains(feature.CapabilityId))
                    {
                        continue;
                    }

                    foreach (var claimId in feature.ClaimIds.Where(mediaClaimIds.Contains))
                    {
                        selectedClaimIds.Add(claimId);
                    }
                }
            }

            var missingFeatures = requestedFeatures
                .Where(featureId => !knownFeatures.Contains(featureId))
                .OrderBy(static id => id, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            if (missingFeatures.Length > 0)
            {
                throw new ArgumentException($"Unknown AI capability id(s): {string.Join(", ", missingFeatures)}.", nameof(requestedCapabilityIds));
            }
        }

        if (selectedClaimIds.Count == 0 && requestedFeatures.Count == 0 && unresolvedClaimIds.Count == 0)
        {
            if (allClaims.Length == 0)
            {
                throw new InvalidOperationException($"No AI capability claims are registered for media kind '{mediaKind}'.");
            }

            return allClaims;
        }

        var selected = allClaims.Where(claim => selectedClaimIds.Contains(claim.Claim.ClaimId)).ToArray();
        var matched = new HashSet<string>(selected.Select(static item => item.Claim.ClaimId), StringComparer.OrdinalIgnoreCase);
        var missing = unresolvedClaimIds
            .Concat(selectedClaimIds.Where(id => !matched.Contains(id)))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(static id => id, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (missing.Length > 0)
        {
            throw new ArgumentException($"Unknown AI claim id(s) for media kind '{mediaKind}': {string.Join(", ", missing)}.", nameof(requestedClaimIds));
        }

        if (selected.Length == 0)
        {
            throw new InvalidOperationException($"No AI capability claims were selected for media kind '{mediaKind}'.");
        }

        return selected;
    }

    private async Task<IReadOnlyList<AiDispatchResult>> MaybeDispatchAsync(
        AiCoreConnectionSettings settings,
        bool? dispatchOverride,
        string mediaKind,
        string subject,
        string runId,
        string? hostEntityType,
        int? hostEntityId,
        IReadOnlyList<ResolvedClaim> claims,
        JsonElement response,
        CancellationToken ct)
    {
        var shouldDispatch = dispatchOverride ?? settings.DispatchResultsByDefault;
        if (!shouldDispatch)
        {
            return [];
        }

        var parsedResult = AiAnalyzeResultParser.Parse(mediaKind, response);
        var assetId = response.ValueKind == JsonValueKind.Object && response.TryGetProperty("asset_id", out var assetIdElement)
            ? assetIdElement.GetString() ?? subject
            : subject;
        var runContext = new AiRunContext(
            runId,
            mediaKind,
            assetId,
            subject,
            hostEntityType,
            hostEntityId,
            parsedResult.DurationSeconds,
            parsedResult.FrameIntervalSeconds,
            new Dictionary<string, string>
            {
                ["source"] = "cove.ai.core",
            });

        var results = new List<AiDispatchResult>();
        foreach (var group in claims.GroupBy(static claim => claim.Descriptor.ExtensionId, StringComparer.OrdinalIgnoreCase))
        {
            var first = group.First();
            var dispatchRequest = new AiDispatchRequest(
                runContext,
                group.Select(static item => item.Claim).ToArray(),
                parsedResult,
                new Dictionary<string, string>
                {
                    ["source"] = "cove.ai.core",
                    ["extensionId"] = first.Descriptor.ExtensionId,
                });

            _logger.LogInformation(
                "Dispatching AI response for {MediaKind} to {ExtensionId} with {ClaimCount} claim(s)",
                mediaKind,
                first.Descriptor.ExtensionId,
                dispatchRequest.Claims.Count);

            results.Add(await first.Contributor.DispatchAsync(dispatchRequest, ct));
        }

        return results;
    }

    private async Task<List<ResolvedWant>> BuildWantAsync(
        AiCoreConnectionSettings settings,
        IReadOnlyList<ResolvedClaim> claims,
        IReadOnlyList<string>? categoriesToSkip,
        string loadPolicy,
        CancellationToken ct)
    {
        var catalogModels = claims.Count > 0
            ? await _aiServerClient.GetModelCatalogAsync(settings, ct)
            : [];
        var catalogModelLookup = BuildCatalogModelLookup(catalogModels);
        var bindingLookup = BuildBindingLookup(settings.CapabilityModelBindings);

        IReadOnlyDictionary<string, List<AiModelCatalogEntry>>? taggingModelsByScope = null;
        if (claims.Any(static claim => string.Equals(claim.Claim.WantCapability, "tagging", StringComparison.OrdinalIgnoreCase)))
        {
            var modelSource = string.Equals(loadPolicy, AiLoadPolicies.UseLoaded, StringComparison.Ordinal)
                ? await _aiServerClient.GetLoadedModelsAsync(settings, ct)
                : catalogModels;
            taggingModelsByScope = ResolveTaggingModelsByScope(modelSource, categoriesToSkip, settings.CapabilityModelBindings);
        }

        return claims
            .GroupBy(static claim => new WantKey(
                claim.Descriptor.ExtensionId,
                claim.Claim.WantCapability,
                claim.Claim.WantScope,
                claim.Claim.FromDetection,
                claim.Claim.CapabilityId ?? string.Empty,
                claim.Claim.ModelBindingSlotId ?? string.Empty,
                claim.Claim.PreferredModels is { Count: > 0 } preferredModels
                    ? string.Join("\u001F", preferredModels)
                    : string.Empty))
            .Select(static group => group.ToArray())
            .Select(group =>
            {
                var first = group[0];
                return new ResolvedWant(
                    first.Descriptor.ExtensionId,
                    first.Claim.WantCapability,
                    first.Claim.WantScope,
                    first.Claim.FromDetection,
                    group,
                        ResolveModels(first.Claim, taggingModelsByScope, catalogModelLookup, bindingLookup),
                    IsTaggingExtensionId(first.Descriptor.ExtensionId));
            })
            .Where(static want => want.Models.Count > 0)
            .ToList();

        static IReadOnlyList<ResolvedWantModel> ResolveModels(
            AiCapabilityClaim claim,
            IReadOnlyDictionary<string, List<AiModelCatalogEntry>>? taggingModelsByScope,
            IReadOnlyDictionary<string, AiModelCatalogEntry> catalogModelLookup,
            IReadOnlyDictionary<string, List<string>> bindingLookup)
        {
            var boundModels = ResolveBoundModels(claim, bindingLookup);
            if (boundModels.Count > 0)
            {
                return boundModels
                    .Where(static model => !string.IsNullOrWhiteSpace(model))
                    .Select(model =>
                    {
                        var trimmed = model.Trim();
                        catalogModelLookup.TryGetValue(trimmed, out var catalogModel);
                        return CreateResolvedWantModel(trimmed, [trimmed], null, catalogModel);
                    })
                    .ToArray();
            }

            if (claim.PreferredModels is { Count: > 0 } preferredModels)
            {
                return preferredModels
                    .Where(static model => !string.IsNullOrWhiteSpace(model))
                    .Select(model =>
                    {
                        var trimmed = model.Trim();
                        catalogModelLookup.TryGetValue(trimmed, out var catalogModel);
                        return catalogModel is null ? null : CreateResolvedWantModel(trimmed, [trimmed], null, catalogModel);
                    })
                    .OfType<ResolvedWantModel>()
                    .ToArray();
            }

            if (!string.Equals(claim.WantCapability, "tagging", StringComparison.OrdinalIgnoreCase) || taggingModelsByScope is null)
            {
                return catalogModelLookup.Values
                    .Where(model => model.Capabilities.Contains(claim.WantCapability, StringComparer.OrdinalIgnoreCase))
                    .Where(model => model.SupportedScopes.Count == 0 || model.SupportedScopes.Contains(claim.WantScope, StringComparer.OrdinalIgnoreCase))
                    .Where(static model => !string.IsNullOrWhiteSpace(model.ConfigName))
                    .DistinctBy(static model => model.ConfigName, StringComparer.OrdinalIgnoreCase)
                    .Select(model => CreateResolvedWantModel(model.ConfigName, [model.ConfigName], null, model))
                    .ToArray();
            }

            if (!taggingModelsByScope.TryGetValue(claim.WantScope, out var models) || models.Count == 0)
            {
                return [];
            }

            return models
                .Where(static model => !string.IsNullOrWhiteSpace(model.ConfigName))
                .DistinctBy(static model => model.ConfigName, StringComparer.OrdinalIgnoreCase)
                .Select(model =>
                {
                    var artifactKeys = model.Categories.Count > 0
                        ? model.Categories
                            .Where(static category => !string.IsNullOrWhiteSpace(category))
                            .Select(static category => category.Trim())
                            .Distinct(StringComparer.OrdinalIgnoreCase)
                            .ToArray()
                        : [model.ConfigName];
                    return CreateResolvedWantModel(model.ConfigName, artifactKeys, artifactKeys.Length == 1 ? artifactKeys[0] : null, model);
                })
                .ToArray();
        }

        static ResolvedWantModel CreateResolvedWantModel(string modelKey, IReadOnlyList<string> artifactKeys, string? category, AiModelCatalogEntry? catalogModel)
            => new(
                modelKey,
                artifactKeys,
                category,
                catalogModel?.Identifier,
                catalogModel?.Version,
                string.IsNullOrWhiteSpace(catalogModel?.Name) ? modelKey : catalogModel.Name);

        static IReadOnlyDictionary<string, AiModelCatalogEntry> BuildCatalogModelLookup(IReadOnlyList<AiModelCatalogEntry> models)
        {
            var lookup = new Dictionary<string, AiModelCatalogEntry>(StringComparer.OrdinalIgnoreCase);
            foreach (var model in models)
            {
                if (!string.IsNullOrWhiteSpace(model.ConfigName) && !lookup.ContainsKey(model.ConfigName))
                {
                    lookup[model.ConfigName] = model;
                }

                if (!string.IsNullOrWhiteSpace(model.Name) && !lookup.ContainsKey(model.Name))
                {
                    lookup[model.Name] = model;
                }
            }

            return lookup;
        }

        static IReadOnlyDictionary<string, List<string>> BuildBindingLookup(IReadOnlyList<AiCapabilityModelBinding> bindings)
        {
            var lookup = new Dictionary<string, List<string>>(StringComparer.OrdinalIgnoreCase);
            foreach (var binding in bindings.Select(static binding => binding.Normalize()))
            {
                var key = CreateBindingKey(binding.CapabilityId, binding.SlotId, binding.Scope, binding.Category);
                if (!lookup.TryGetValue(key, out var models))
                {
                    models = [];
                    lookup[key] = models;
                }

                models.Add(binding.Model);
            }

            return lookup;
        }

        static IReadOnlyList<string> ResolveBoundModels(AiCapabilityClaim claim, IReadOnlyDictionary<string, List<string>> bindingLookup)
        {
            if (string.IsNullOrWhiteSpace(claim.CapabilityId) || string.IsNullOrWhiteSpace(claim.ModelBindingSlotId))
            {
                return [];
            }

            var exactKey = CreateBindingKey(claim.CapabilityId, claim.ModelBindingSlotId, claim.WantScope, category: null);
            if (bindingLookup.TryGetValue(exactKey, out var exactModels))
            {
                return exactModels.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();
            }

            var genericKey = CreateBindingKey(claim.CapabilityId, claim.ModelBindingSlotId, scope: null, category: null);
            return bindingLookup.TryGetValue(genericKey, out var genericModels)
                ? genericModels.Distinct(StringComparer.OrdinalIgnoreCase).ToArray()
                : [];
        }
    }

    private static ExecutionPlan BuildExecution(IReadOnlyList<ResolvedWant> wants, IReadOnlyList<AiRunExecutionPlan> plans)
    {
        var analyzeWants = new List<AnalyzeWantRequest>();
        var executionClaims = new List<ResolvedClaim>();
        for (var index = 0; index < wants.Count; index++)
        {
            var want = wants[index];
            var plan = plans[index];
            if (plan.ExecutionModels.Count == 0)
            {
                continue;
            }

            analyzeWants.Add(new AnalyzeWantRequest
            {
                Capability = want.Capability,
                Scope = want.Scope,
                Models = plan.ExecutionModels.ToList(),
                FromDetection = want.FromDetection,
            });
            executionClaims.AddRange(want.Claims);
        }

        return new ExecutionPlan(analyzeWants, executionClaims);
    }

    private static IReadOnlyList<AiRunPlanItem> BuildResponsePlan(IReadOnlyList<AiRunExecutionPlan> plans)
        => plans
            .SelectMany(plan => plan.Claims.Select(claim => new AiRunPlanItem(
                claim.ClaimId,
                plan.ExtensionId,
                plan.Capability,
                plan.Scope,
                plan.DesiredModels,
                plan.ExecutionModels,
                plan.Decision,
                plan.Reasons,
                plan.Forced)))
            .ToArray();

    private static JsonElement CreateSkippedAnalysis(string mediaKind, string subject)
        => JsonDocument.Parse(JsonSerializer.Serialize(new
        {
            status = "skipped",
            mediaKind,
            subject,
        })).RootElement.Clone();

    private static IReadOnlyDictionary<string, List<AiModelCatalogEntry>> ResolveTaggingModelsByScope(
        IReadOnlyList<AiModelCatalogEntry> models,
        IReadOnlyList<string>? categoriesToSkip,
        IReadOnlyList<AiCapabilityModelBinding>? bindings)
    {
        var skippedCategories = new HashSet<string>(
            categoriesToSkip?.Where(static category => !string.IsNullOrWhiteSpace(category)).Select(static category => category.Trim())
                ?? [],
            StringComparer.OrdinalIgnoreCase);

        var preferenceMap = (bindings ?? [])
            .Where(static binding => string.Equals(binding.CapabilityId, "tagging", StringComparison.OrdinalIgnoreCase)
                && string.Equals(binding.SlotId, "category", StringComparison.OrdinalIgnoreCase)
                && !string.IsNullOrWhiteSpace(binding.Scope)
                && !string.IsNullOrWhiteSpace(binding.Category)
                && !string.IsNullOrWhiteSpace(binding.Model))
            .Select(static binding => binding.Normalize())
            .GroupBy(static binding => CreateTaggingPreferenceKey(binding.Scope!, binding.Category!), StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                static group => group.Key,
                static group => group.First().Model,
                StringComparer.OrdinalIgnoreCase);

        var categorizedCandidatesByScope = new Dictionary<string, Dictionary<string, List<AiModelCatalogEntry>>>(StringComparer.OrdinalIgnoreCase);
        var uncategorizedCandidatesByScope = new Dictionary<string, List<AiModelCatalogEntry>>(StringComparer.OrdinalIgnoreCase);

        foreach (var model in models)
        {
            if (string.IsNullOrWhiteSpace(model.ConfigName)
                || !model.Capabilities.Contains("tagging", StringComparer.OrdinalIgnoreCase))
            {
                continue;
            }

            var scopes = (model.SupportedScopes.Count > 0 ? model.SupportedScopes : ["asset", "frame", "region"])
                .Select(NormalizeTaggingScope)
                .Where(static scope => !string.IsNullOrWhiteSpace(scope))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();
            if (scopes.Length == 0)
            {
                continue;
            }

            var categories = model.Categories
                .Where(static category => !string.IsNullOrWhiteSpace(category))
                .Select(static category => category.Trim())
                .Where(category => !skippedCategories.Contains(category))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            if (categories.Length == 0)
            {
                foreach (var scope in scopes)
                {
                    if (!uncategorizedCandidatesByScope.TryGetValue(scope, out var candidates))
                    {
                        candidates = [];
                        uncategorizedCandidatesByScope[scope] = candidates;
                    }

                    candidates.Add(model);
                }

                continue;
            }

            foreach (var scope in scopes)
            {
                if (!categorizedCandidatesByScope.TryGetValue(scope, out var categoriesByName))
                {
                    categoriesByName = new Dictionary<string, List<AiModelCatalogEntry>>(StringComparer.OrdinalIgnoreCase);
                    categorizedCandidatesByScope[scope] = categoriesByName;
                }

                foreach (var category in categories)
                {
                    if (!categoriesByName.TryGetValue(category, out var candidates))
                    {
                        candidates = [];
                        categoriesByName[category] = candidates;
                    }

                    candidates.Add(model);
                }
            }
        }

        var resolved = new Dictionary<string, List<AiModelCatalogEntry>>(StringComparer.OrdinalIgnoreCase);
        var resolvedScopes = categorizedCandidatesByScope.Keys
            .Concat(uncategorizedCandidatesByScope.Keys)
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(static scope => scope, StringComparer.OrdinalIgnoreCase);

        foreach (var scope in resolvedScopes)
        {
            var selectedModels = new Dictionary<string, AiModelCatalogEntry>(StringComparer.OrdinalIgnoreCase);

            if (categorizedCandidatesByScope.TryGetValue(scope, out var categoriesByName))
            {
                foreach (var categoryEntry in categoriesByName.OrderBy(static item => item.Key, StringComparer.OrdinalIgnoreCase))
                {
                    var selectedModel = SelectTaggingModel(scope, categoryEntry.Key, categoryEntry.Value, preferenceMap);
                    if (!string.IsNullOrWhiteSpace(selectedModel?.ConfigName))
                    {
                        selectedModels[selectedModel.ConfigName] = selectedModel;
                    }
                }
            }

            if (selectedModels.Count == 0 && uncategorizedCandidatesByScope.TryGetValue(scope, out var uncategorizedCandidates))
            {
                var selectedModel = SelectTaggingModel(scope, "*", uncategorizedCandidates, preferenceMap);
                if (!string.IsNullOrWhiteSpace(selectedModel?.ConfigName))
                {
                    selectedModels[selectedModel.ConfigName] = selectedModel;
                }
            }

            if (selectedModels.Count > 0)
            {
                resolved[scope] = selectedModels
                    .OrderBy(static model => model.Key, StringComparer.OrdinalIgnoreCase)
                    .Select(static model => model.Value)
                    .ToList();
            }
        }

        return resolved;

        static string CreateTaggingPreferenceKey(string scope, string category)
            => $"{NormalizeTaggingScope(scope)}\u001F{(category ?? string.Empty).Trim()}";

        static string NormalizeTaggingScope(string? scope)
            => (scope ?? string.Empty).Trim().ToLowerInvariant();

        static AiModelCatalogEntry? SelectTaggingModel(
            string scope,
            string category,
            IReadOnlyList<AiModelCatalogEntry> candidates,
            IReadOnlyDictionary<string, string> preferenceMap)
        {
            var distinctCandidates = candidates
                .Where(static candidate => !string.IsNullOrWhiteSpace(candidate.ConfigName))
                .DistinctBy(static candidate => candidate.ConfigName, StringComparer.OrdinalIgnoreCase)
                .ToArray();
            if (distinctCandidates.Length == 0)
            {
                return null;
            }

            if (preferenceMap.TryGetValue(CreateTaggingPreferenceKey(scope, category), out var preferredModel))
            {
                return distinctCandidates.FirstOrDefault(candidate => string.Equals(candidate.ConfigName, preferredModel, StringComparison.OrdinalIgnoreCase));
            }

            return distinctCandidates
                .OrderByDescending(static candidate => candidate.Pinned)
                .ThenByDescending(static candidate => candidate.Loaded)
                .ThenByDescending(static candidate => candidate.Active)
                .ThenBy(static candidate => candidate.ConfigName, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault();
        }
    }

    private static string ResolveLoadPolicy(AiCoreConnectionSettings settings, string? loadPolicy)
    {
        var resolved = string.IsNullOrWhiteSpace(loadPolicy) ? settings.DefaultLoadPolicy : loadPolicy.Trim();
        if (!AiLoadPolicies.All.Contains(resolved))
        {
            throw new ArgumentException($"Unsupported load policy '{resolved}'.", nameof(loadPolicy));
        }

        return resolved;
    }

    private static bool IsTaggingExtensionId(string extensionId)
        => string.Equals(extensionId, "cove.ai.tagging", StringComparison.OrdinalIgnoreCase)
            || string.Equals(extensionId, "ext:ai.tagging", StringComparison.OrdinalIgnoreCase);

    private static string CreateBindingKey(string capabilityId, string slotId, string? scope, string? category)
        => $"{(capabilityId ?? string.Empty).Trim().ToLowerInvariant()}\u001F{(slotId ?? string.Empty).Trim().ToLowerInvariant()}\u001F{(scope ?? string.Empty).Trim().ToLowerInvariant()}\u001F{(category ?? string.Empty).Trim()}";

    private static List<string> NormalizeStringList(IEnumerable<string>? values)
        => (values ?? [])
            .Where(static value => !string.IsNullOrWhiteSpace(value))
            .Select(static value => value.Trim())
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .ToList();

    private readonly record struct ResolvedContributor(IAiCapabilityContributor Contributor, AiCapabilityDescriptor Descriptor);

    private readonly record struct ResolvedClaim(IAiCapabilityContributor Contributor, AiCapabilityDescriptor Descriptor, AiCapabilityClaim Claim);

    private sealed record RunSelection(
        IReadOnlyList<string>? CapabilityIds,
        IReadOnlyList<string>? ClaimIds,
        IReadOnlyList<string>? CategoriesToSkip,
        string? LoadPolicy,
        string? PipelineName);

    private sealed record ExecutionPlan(IReadOnlyList<AnalyzeWantRequest> Wants, IReadOnlyList<ResolvedClaim> Claims);

    private readonly record struct WantKey(string ExtensionId, string Capability, string Scope, string? FromDetection, string CapabilityId, string SlotId, string PreferredModelsKey);

    private sealed record ResolvedWantModel(
        string ModelKey,
        IReadOnlyList<string> ArtifactKeys,
        string? Category,
        int? Identifier,
        string? Version,
        string? Name)
    {
        public AiRunPlannerModel ToPlannerModel()
            => new(ModelKey, ArtifactKeys, Category, Identifier, Version, Name);
    }

    private sealed record ResolvedWant(
        string ExtensionId,
        string Capability,
        string Scope,
        string? FromDetection,
        IReadOnlyList<ResolvedClaim> Claims,
        IReadOnlyList<ResolvedWantModel> Models,
        bool AllowPartialExecution)
    {
        public AiRunPlannerWant ToPlannerWant()
            => new(ExtensionId, Capability, Scope, FromDetection, Claims.Select(static claim => claim.Claim).ToArray(), Models.Select(static model => model.ToPlannerModel()).ToArray(), AllowPartialExecution);
    }
}
