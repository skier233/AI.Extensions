using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Plugins;

using Microsoft.Extensions.Logging;

namespace AI.Core;

public interface IAiCoreOrchestrator
{
    IReadOnlyList<AiCapabilityDescriptor> GetCapabilities();

    Task<AiRunResponse> RunImagesAsync(AiCoreConnectionSettings settings, AiRunImagesRequest request, CancellationToken ct = default);

    Task<IReadOnlyList<AiRunResponse>> RunImageBatchAsync(AiCoreConnectionSettings settings, IReadOnlyList<AiRunImageTarget> targets, AiRunImagesRequest template, CancellationToken ct = default);

    Task<AiRunResponse> RunVideoAsync(AiCoreConnectionSettings settings, AiRunVideoRequest request, CancellationToken ct = default);

    Task<AiRunResponse> RunAudioAsync(AiCoreConnectionSettings settings, AiRunAudioRequest request, CancellationToken ct = default);
}

public sealed class AiCoreOrchestrator(
    INsfwAiServerClient aiServerClient,
    IExtensionServiceExchange exchange,
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
    private readonly IExtensionServiceExchange _exchange = exchange;

    // Capability contributors are resolved LIVE from the cross-extension exchange on each use, because
    // feature extensions (AI Tagging, Faces, Visual, Audio) live in their own isolated containers and
    // can be installed or removed at runtime. They publish their IAiCapabilityContributor to the
    // exchange on initialize; the host withdraws it on disable/uninstall.
    private IReadOnlyList<ResolvedContributor> ResolveContributors()
        => _exchange.GetAll<IAiCapabilityContributor>()
            .Select(static contributor => new ResolvedContributor(contributor, contributor.Describe()))
            .OrderBy(static item => item.Descriptor.ExtensionId, StringComparer.OrdinalIgnoreCase)
            .ToArray();

    public IReadOnlyList<AiCapabilityDescriptor> GetCapabilities()
        => ResolveContributors().Select(static item => item.Descriptor).ToArray();

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
                request.Paths.Count == 1 ? request.Paths[0] : $"{request.Paths.Count} image(s)",
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

    // Processes many images in a single AI-server call. Selection/claims/wants are resolved once (they are
    // run-level), planning is per entity, and targets are grouped by their resolved execution want-set so a
    // homogeneous batch (the common case) becomes one server call. The batched response's per-image results
    // are split back to each entity for the existing per-entity artifact replace, dispatch, and journaling.
    public async Task<IReadOnlyList<AiRunResponse>> RunImageBatchAsync(
        AiCoreConnectionSettings settings,
        IReadOnlyList<AiRunImageTarget> targets,
        AiRunImagesRequest template,
        CancellationToken ct = default)
    {
        if (targets.Count == 0)
        {
            return [];
        }

        var selection = ResolveRunSelection(settings, AiMediaKinds.Image, template.PresetId, template.CapabilityIds, template.ClaimIds, template.CategoriesToSkip, template.LoadPolicy, template.PipelineName);
        var claims = SelectClaims(AiMediaKinds.Image, selection.ClaimIds, selection.CapabilityIds);
        var resolvedLoadPolicy = ResolveLoadPolicy(settings, selection.LoadPolicy);
        var wants = await BuildWantAsync(settings, claims, selection.CategoriesToSkip, resolvedLoadPolicy, ct);
        var threshold = template.Threshold ?? settings.DefaultThreshold;
        var claimDescriptors = claims.Select(static item => item.Claim).ToArray();

        var planned = new List<PlannedImageTarget>(targets.Count);
        foreach (var target in targets)
        {
            var mappedPath = AiPathMapper.MapPath(settings.PathMappings, target.Path);
            var plans = await _aiRunPlanner.PlanAsync(
                settings,
                target.EntityType,
                target.EntityId,
                wants.Select(static want => want.ToPlannerWant()).ToArray(),
                template.ForceClaimIds,
                frameIntervalSeconds: null,
                threshold: threshold,
                ct: ct);
            planned.Add(new PlannedImageTarget(target, mappedPath, plans, BuildExecution(wants, plans), BuildResponsePlan(plans)));
        }

        var responses = new List<AiRunResponse>(targets.Count);

        // Fully-satisfied targets need no server call.
        foreach (var item in planned.Where(static item => item.Execution.Wants.Count == 0))
        {
            responses.Add(new AiRunResponse(
                Guid.NewGuid().ToString("n"),
                AiMediaKinds.Image,
                claimDescriptors,
                CreateSkippedAnalysis(AiMediaKinds.Image, item.Target.Path),
                [],
                item.ResponsePlan));
        }

        foreach (var group in planned.Where(static item => item.Execution.Wants.Count > 0).GroupBy(static item => BuildWantSignature(item.Execution.Wants)))
        {
            var members = group.ToArray();
            var analyzeRequest = new ImageAnalyzeRequest
            {
                Paths = members.Select(static item => item.MappedPath).ToList(),
                Threshold = threshold,
                ReturnConfidence = template.ReturnConfidence ?? true,
                CategoriesToSkip = selection.CategoriesToSkip?.ToList(),
                Want = members[0].Execution.Wants.ToList(),
                LoadPolicy = resolvedLoadPolicy,
                PipelineName = selection.PipelineName,
            };

            var runIds = new string[members.Length];
            for (var index = 0; index < members.Length; index++)
            {
                var member = members[index];
                runIds[index] = Guid.NewGuid().ToString("n");
                await _aiRunJournal.RecordStartAsync(
                    new AiRunJournalStart(runIds[index], member.Target.EntityType, member.Target.EntityId, "AI.Core", resolvedLoadPolicy, null, null, BuildTargetRequest(template, member.Target)),
                    ct);
            }

            JsonElement response;
            try
            {
                response = await _aiServerClient.AnalyzeImagesAsync(settings, analyzeRequest, ct);
            }
            catch (Exception ex)
            {
                foreach (var runId in runIds)
                {
                    await _aiRunJournal.RecordFailureAsync(runId, ex, ct);
                }

                throw;
            }

            var perImageResults = ExtractImageResults(response, members.Length);
            for (var index = 0; index < members.Length; index++)
            {
                var member = members[index];
                var runId = runIds[index];
                var perImage = perImageResults[index];
                if (TryGetImageError(perImage, out var error))
                {
                    await _aiRunJournal.RecordFailureAsync(runId, new InvalidOperationException($"AI server reported an error for '{member.Target.Path}': {error}"), ct);
                    responses.Add(new AiRunResponse(runId, AiMediaKinds.Image, claimDescriptors, perImage, [], member.ResponsePlan));
                    continue;
                }

                try
                {
                    await _aiArtifactReplaceService.ReplaceAsync(member.Target.EntityType, member.Target.EntityId, member.Plans, ct);
                    var dispatchResults = await MaybeDispatchAsync(
                        settings,
                        template.DispatchResults,
                        AiMediaKinds.Image,
                        member.Target.Path,
                        runId,
                        member.Target.EntityType,
                        member.Target.EntityId,
                        member.Execution.Claims,
                        perImage,
                        ct);
                    await _aiRunJournal.RecordCompletionAsync(
                        new AiRunJournalCompletion(runId, AiMediaKinds.Image, perImage, member.Execution.Claims.Select(static item => item.Claim.ClaimId).ToArray(), dispatchResults.Count),
                        ct);
                    responses.Add(new AiRunResponse(runId, AiMediaKinds.Image, claimDescriptors, perImage, dispatchResults, member.ResponsePlan));
                }
                catch (Exception ex)
                {
                    await _aiRunJournal.RecordFailureAsync(runId, ex, ct);
                    throw;
                }
            }
        }

        return responses;
    }

    private static string BuildWantSignature(IReadOnlyList<AnalyzeWantRequest> wants)
        => string.Join("", wants.Select(static want => string.Join(
            "",
            want.Capability ?? string.Empty,
            want.Scope ?? string.Empty,
            want.FromDetection ?? string.Empty,
            want.Models is { Count: > 0 } models ? string.Join(",", models) : string.Empty)));

    private static AiRunImagesRequest BuildTargetRequest(AiRunImagesRequest template, AiRunImageTarget target)
        => new()
        {
            Paths = [target.Path],
            EntityType = target.EntityType,
            EntityId = target.EntityId,
            PresetId = template.PresetId,
            CapabilityIds = template.CapabilityIds,
            ClaimIds = template.ClaimIds,
            PipelineName = template.PipelineName,
            Threshold = template.Threshold,
            ReturnConfidence = template.ReturnConfidence,
            CategoriesToSkip = template.CategoriesToSkip,
            LoadPolicy = template.LoadPolicy,
            DispatchResults = template.DispatchResults,
            ForceClaimIds = template.ForceClaimIds,
        };

    // The v4 /analyze/images response is { "result": [ <per-image>, ... ] } ordered to match the input
    // paths. Fall back to treating the whole payload as a single result if the shape is unexpected.
    private static IReadOnlyList<JsonElement> ExtractImageResults(JsonElement response, int expectedCount)
    {
        if (response.ValueKind == JsonValueKind.Object
            && response.TryGetProperty("result", out var result)
            && result.ValueKind == JsonValueKind.Array)
        {
            var items = result.EnumerateArray().Select(static element => element.Clone()).ToArray();
            if (items.Length == expectedCount)
            {
                return items;
            }
        }

        // Unexpected shape: hand the same payload to every member so nothing silently drops.
        var fallback = response.Clone();
        return Enumerable.Repeat(fallback, expectedCount).ToArray();
    }

    private static bool TryGetImageError(JsonElement perImage, out string? error)
    {
        if (perImage.ValueKind == JsonValueKind.Object
            && perImage.TryGetProperty("error", out var errorElement)
            && errorElement.ValueKind == JsonValueKind.String)
        {
            error = errorElement.GetString();
            return true;
        }

        error = null;
        return false;
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
        var contributors = ResolveContributors();
        var contributorClaims = contributors
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
            foreach (var contributor in contributors)
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
                ["source"] = "cove.community.ai.core",
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
                    ["source"] = "cove.community.ai.core",
                    ["extensionId"] = first.Descriptor.ExtensionId,
                });

            _logger.LogDebug(
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
                    FilterFullySkippedModels(
                        ResolveModels(first.Claim, first.Descriptor, loadPolicy, catalogModels, taggingModelsByScope, catalogModelLookup, bindingLookup),
                        categoriesToSkip),
                    IsTaggingExtensionId(first.Descriptor.ExtensionId));
            })
            .Where(static want => want.Models.Count > 0)
            .ToList();

        static IReadOnlyList<ResolvedWantModel> ResolveModels(
            AiCapabilityClaim claim,
            AiCapabilityDescriptor descriptor,
            string loadPolicy,
            IReadOnlyList<AiModelCatalogEntry> catalogModels,
            IReadOnlyDictionary<string, List<AiModelCatalogEntry>>? taggingModelsByScope,
            IReadOnlyDictionary<string, AiModelCatalogEntry> catalogModelLookup,
            IReadOnlyDictionary<string, List<string>> bindingLookup)
        {
            if (string.Equals(claim.WantCapability, "tagging", StringComparison.OrdinalIgnoreCase))
            {
                if (taggingModelsByScope is null || !taggingModelsByScope.TryGetValue(claim.WantScope, out var taggingModels) || taggingModels.Count == 0)
                {
                    return [];
                }

                return taggingModels
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

            var autoModels = ResolveAutoModels(claim, descriptor, catalogModels, catalogModelLookup);
            if (autoModels.HasSlot)
            {
                if (autoModels.Models.Count > 0)
                {
                    return autoModels.Models
                        .Select(model => CreateResolvedWantModel(model.ConfigName, [model.ConfigName], null, model))
                        .ToArray();
                }

                if (string.Equals(loadPolicy, AiLoadPolicies.UseLoaded, StringComparison.Ordinal))
                {
                    return [];
                }
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

            return catalogModelLookup.Values
                .Where(model => model.Capabilities.Contains(claim.WantCapability, StringComparer.OrdinalIgnoreCase))
                .Where(model => model.SupportedScopes.Count == 0 || model.SupportedScopes.Contains(claim.WantScope, StringComparer.OrdinalIgnoreCase))
                .Where(static model => !string.IsNullOrWhiteSpace(model.ConfigName))
                .DistinctBy(static model => model.ConfigName, StringComparer.OrdinalIgnoreCase)
                .Select(model => CreateResolvedWantModel(model.ConfigName, [model.ConfigName], null, model))
                .ToArray();
        }

        static ResolvedWantModel CreateResolvedWantModel(string modelKey, IReadOnlyList<string> artifactKeys, string? category, AiModelCatalogEntry? catalogModel)
            => new(
                modelKey,
                artifactKeys,
                category,
                catalogModel?.Identifier,
                catalogModel?.Version,
                string.IsNullOrWhiteSpace(catalogModel?.Name) ? modelKey : catalogModel.Name,
                catalogModel?.Categories is { Count: > 0 } categories ? categories.ToArray() : []);

        // Mirror the AI server's per-model skip gate: a model is only skipped server-side
        // when it has categories AND every one of them is in categories_to_skip. Models with
        // no categories (e.g. embeddings, audio, face models) are never gated by this list.
        // Dropping fully-skipped models here means a want that resolves to nothing-to-run is
        // pruned, so the orchestrator's "no execution wants" path skips the server call entirely
        // instead of issuing a request that only preprocesses and runs no models.
        static IReadOnlyList<ResolvedWantModel> FilterFullySkippedModels(
            IReadOnlyList<ResolvedWantModel> models,
            IReadOnlyList<string>? categoriesToSkip)
        {
            if (models.Count == 0 || categoriesToSkip is null || categoriesToSkip.Count == 0)
            {
                return models;
            }

            var skipped = new HashSet<string>(
                categoriesToSkip
                    .Where(static category => !string.IsNullOrWhiteSpace(category))
                    .Select(static category => category.Trim()),
                StringComparer.OrdinalIgnoreCase);
            if (skipped.Count == 0)
            {
                return models;
            }

            return models
                .Where(model =>
                {
                    var categories = model.Categories
                        .Where(static category => !string.IsNullOrWhiteSpace(category))
                        .Select(static category => category.Trim())
                        .ToArray();
                    return categories.Length == 0 || !categories.All(skipped.Contains);
                })
                .ToList();
        }

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

        static (bool HasSlot, IReadOnlyList<AiModelCatalogEntry> Models) ResolveAutoModels(
            AiCapabilityClaim claim,
            AiCapabilityDescriptor descriptor,
            IReadOnlyList<AiModelCatalogEntry> catalogModels,
            IReadOnlyDictionary<string, AiModelCatalogEntry> catalogModelLookup)
        {
            if (string.IsNullOrWhiteSpace(claim.CapabilityId) || string.IsNullOrWhiteSpace(claim.ModelBindingSlotId))
            {
                return (false, []);
            }

            var slot = descriptor.Capabilities
                .Where(feature => string.Equals(feature.CapabilityId, claim.CapabilityId, StringComparison.OrdinalIgnoreCase)
                    && feature.ClaimIds.Contains(claim.ClaimId, StringComparer.OrdinalIgnoreCase))
                .SelectMany(static feature => feature.ModelBindingSlots ?? [])
                .FirstOrDefault(slot => string.Equals(slot.SlotId, claim.ModelBindingSlotId, StringComparison.OrdinalIgnoreCase));
            if (slot is null)
            {
                return (false, []);
            }

            var models = catalogModels
                .Where(static model => model.Loaded)
                .Where(model => ModelMatchesSlot(model, slot))
                .Where(static model => !string.IsNullOrWhiteSpace(model.ConfigName))
                .ToList();

            foreach (var defaultModelName in slot.DefaultModels ?? [])
            {
                var trimmed = defaultModelName?.Trim();
                if (string.IsNullOrWhiteSpace(trimmed)
                    || !catalogModelLookup.TryGetValue(trimmed, out var defaultModel)
                    || !defaultModel.Loaded
                    || string.IsNullOrWhiteSpace(defaultModel.ConfigName)
                    || models.Any(model => string.Equals(model.ConfigName, defaultModel.ConfigName, StringComparison.OrdinalIgnoreCase)))
                {
                    continue;
                }

                models.Add(defaultModel);
            }

            var resolved = models
                .DistinctBy(static model => model.ConfigName, StringComparer.OrdinalIgnoreCase)
                .OrderByDescending(static model => model.Active)
                .ThenBy(static model => model.ConfigName, StringComparer.OrdinalIgnoreCase)
                .ToArray();

            return (true, slot.AllowMultiple ? resolved : resolved.Take(1).ToArray());
        }

        static bool ModelMatchesSlot(AiModelCatalogEntry model, AiModelBindingSlot slot)
        {
            if (!HasAnyRequired(slot.RequiredCapabilities, model.Capabilities))
            {
                return false;
            }

            if ((slot.RequiredScopes?.Count ?? 0) > 0
                && model.SupportedScopes.Count > 0
                && !HasAnyRequired(slot.RequiredScopes, model.SupportedScopes))
            {
                return false;
            }

            return HasAnyRequired(slot.RequiredCategories, model.Categories);
        }

        static bool HasAnyRequired(IReadOnlyList<string>? required, IReadOnlyList<string>? actual)
        {
            var requiredValues = required?.Where(static value => !string.IsNullOrWhiteSpace(value)).Select(static value => value.Trim()).ToArray() ?? [];
            if (requiredValues.Length == 0)
            {
                return true;
            }

            return actual?.Any(value => requiredValues.Contains(value, StringComparer.OrdinalIgnoreCase)) == true;
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
                .OrderByDescending(static candidate => candidate.Loaded)
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
        => string.Equals(extensionId, "cove.community.ai.tagging", StringComparison.OrdinalIgnoreCase)
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

    private sealed record PlannedImageTarget(
        AiRunImageTarget Target,
        string MappedPath,
        IReadOnlyList<AiRunExecutionPlan> Plans,
        ExecutionPlan Execution,
        IReadOnlyList<AiRunPlanItem> ResponsePlan);

    private readonly record struct WantKey(string ExtensionId, string Capability, string Scope, string? FromDetection, string CapabilityId, string SlotId, string PreferredModelsKey);

    private sealed record ResolvedWantModel(
        string ModelKey,
        IReadOnlyList<string> ArtifactKeys,
        string? Category,
        int? Identifier,
        string? Version,
        string? Name,
        IReadOnlyList<string> Categories)
    {
        public AiRunPlannerModel ToPlannerModel()
            => new(ModelKey, ArtifactKeys, Category, Identifier, Version, Name, Categories);
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
