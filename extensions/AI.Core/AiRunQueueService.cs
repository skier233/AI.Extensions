using System.IO;

using AI.Extensions.Abstractions;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace AI.Core;

public interface IAiRunTargetResolver
{
    Task<IReadOnlyList<AiResolvedRunTarget>> ResolveAsync(AiQueueRunRequest request, CancellationToken ct = default);
}

public interface IAiRunQueueService
{
    Task<AiQueuedRunResponse> QueueAsync(AiCoreConnectionSettings settings, AiQueueRunRequest request, CancellationToken ct = default);

    Task ExecuteAsync(AiCoreConnectionSettings settings, AiQueueRunRequest request, IJobProgress progress, CancellationToken ct = default);
}

public sealed class AiRunTargetResolver(IVideoRepository videoRepository, IImageRepository imageRepository) : IAiRunTargetResolver
{
    private readonly IVideoRepository _videoRepository = videoRepository;
    private readonly IImageRepository _imageRepository = imageRepository;

    public async Task<IReadOnlyList<AiResolvedRunTarget>> ResolveAsync(AiQueueRunRequest request, CancellationToken ct = default)
    {
        var normalized = request.Normalize();
        var results = new List<AiResolvedRunTarget>();
        var seenPaths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var path in normalized.Paths)
        {
            var normalizedPath = AiPathMapper.NormalizePath(path);
            if (seenPaths.Add(normalizedPath))
            {
                results.Add(new AiResolvedRunTarget(
                    UnitId: $"path:{normalizedPath}",
                    Label: Path.GetFileName(normalizedPath) is { Length: > 0 } basename ? basename : normalizedPath,
                    Path: normalizedPath));
            }
        }

        if (normalized.EntityIds.Count == 0)
        {
            return results;
        }

        var entityTargets = normalized.EntityType switch
        {
            "video" => await ResolveVideoTargetsAsync(normalized, ct),
            "image" => await ResolveImageTargetsAsync(normalized, ct),
            _ => throw new ArgumentException($"Unsupported entity type '{normalized.EntityType}'.", nameof(request)),
        };

        foreach (var target in entityTargets)
        {
            if (seenPaths.Add(target.Path))
            {
                results.Add(target);
            }
        }

        return results;
    }

    private async Task<IReadOnlyList<AiResolvedRunTarget>> ResolveVideoTargetsAsync(AiQueueRunRequest request, CancellationToken ct)
    {
        var (videos, _) = await _videoRepository.FindAsync(
            new VideoFilter { Ids = request.EntityIds.ToList() },
            new FindFilter { PerPage = request.EntityIds.Count },
            ct);
        var videoById = videos.ToDictionary(static v => v.Id);

        var results = new List<AiResolvedRunTarget>();
        foreach (var entityId in request.EntityIds)
        {
            if (!videoById.TryGetValue(entityId, out var video))
            {
                continue;
            }

            var file = video.Files
                .Where(static file => !string.IsNullOrWhiteSpace(file.Path))
                .OrderByDescending(static file => file.Duration)
                .ThenBy(static file => file.Path, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault();

            if (file is null)
            {
                continue;
            }

            var normalizedPath = AiPathMapper.NormalizePath(file.Path);
            var label = !string.IsNullOrWhiteSpace(video.Title)
                ? video.Title!
                : Path.GetFileName(normalizedPath) is { Length: > 0 } basename ? basename : $"Video {video.Id}";

            results.Add(new AiResolvedRunTarget(
                UnitId: $"video:{video.Id}",
                Label: label,
                Path: normalizedPath,
                EntityId: video.Id,
                EntityType: "video"));
        }

        return results;
    }

    private async Task<IReadOnlyList<AiResolvedRunTarget>> ResolveImageTargetsAsync(AiQueueRunRequest request, CancellationToken ct)
    {
        var (images, _) = await _imageRepository.FindAsync(
            new ImageFilter { Ids = request.EntityIds.ToList() },
            new FindFilter { PerPage = request.EntityIds.Count },
            ct);
        var imageById = images.ToDictionary(static i => i.Id);

        var results = new List<AiResolvedRunTarget>();
        foreach (var entityId in request.EntityIds)
        {
            if (!imageById.TryGetValue(entityId, out var image))
            {
                continue;
            }

            var file = image.Files
                .Where(static file => !string.IsNullOrWhiteSpace(file.Path))
                .OrderByDescending(static file => (long)file.Width * file.Height)
                .ThenBy(static file => file.Path, StringComparer.OrdinalIgnoreCase)
                .FirstOrDefault();

            if (file is null)
            {
                continue;
            }

            var normalizedPath = AiPathMapper.NormalizePath(file.Path);
            var label = !string.IsNullOrWhiteSpace(image.Title)
                ? image.Title!
                : Path.GetFileName(normalizedPath) is { Length: > 0 } basename ? basename : $"Image {image.Id}";

            results.Add(new AiResolvedRunTarget(
                UnitId: $"image:{image.Id}",
                Label: label,
                Path: normalizedPath,
                EntityId: image.Id,
                EntityType: "image"));
        }

        return results;
    }
}

public sealed class AiRunQueueService(
    IAiCoreOrchestrator orchestrator,
    IAiRunTargetResolver targetResolver,
    IJobService jobService,
    IServiceScopeFactory scopeFactory,
    ILogger<AiRunQueueService> logger) : IAiRunQueueService
{
    private readonly IAiCoreOrchestrator _orchestrator = orchestrator;
    private readonly IAiRunTargetResolver _targetResolver = targetResolver;
    private readonly IJobService _jobService = jobService;
    private readonly IServiceScopeFactory _scopeFactory = scopeFactory;
    private readonly ILogger<AiRunQueueService> _logger = logger;

    public async Task<AiQueuedRunResponse> QueueAsync(AiCoreConnectionSettings settings, AiQueueRunRequest request, CancellationToken ct = default)
    {
        var normalized = request.Normalize();
        var targets = await _targetResolver.ResolveAsync(normalized, ct);
        if (targets.Count == 0)
        {
            throw new InvalidOperationException("No AI targets could be resolved from the supplied selection.");
        }

        var description = BuildDescription(normalized, targets);
        var jobId = _jobService.Enqueue(
            type: "ai.run",
            description: description,
            work: (progress, jobCt) => ExecuteResolvedAsync(_scopeFactory, _jobService, _logger, settings, normalized, targets, progress, jobCt),
            exclusive: true);

        return new AiQueuedRunResponse(
            JobId: jobId,
            Description: description,
            TargetCount: targets.Count,
            MediaKind: normalized.MediaKind,
            ClaimIds: normalized.ClaimIds ?? []);
    }

    public async Task ExecuteAsync(AiCoreConnectionSettings settings, AiQueueRunRequest request, IJobProgress progress, CancellationToken ct = default)
    {
        var normalized = request.Normalize();
        var targets = await _targetResolver.ResolveAsync(normalized, ct);
        if (targets.Count == 0)
        {
            throw new InvalidOperationException("No AI targets could be resolved from the supplied selection.");
        }

        await ExecuteResolvedAsync(_scopeFactory, _jobService, _logger, settings, normalized, targets, progress, ct);
    }

    private static async Task ExecuteResolvedAsync(
        IServiceScopeFactory scopeFactory,
        IJobService jobService,
        ILogger<AiRunQueueService> logger,
        AiCoreConnectionSettings settings,
        AiQueueRunRequest request,
        IReadOnlyList<AiResolvedRunTarget> targets,
        IJobProgress progress,
        CancellationToken ct)
    {
        var executionSettings = settings.RequestTimeoutSeconds == 0
            ? settings
            : settings with { RequestTimeoutSeconds = 0 };

        // Images are sent to the AI server in batches (one server call per ImageBatchSize images), each
        // batch counting as a single in-flight unit. Video/audio stay one server call per target.
        if (request.MediaKind == AiMediaKinds.Image)
        {
            await ExecuteImageBatchesAsync(scopeFactory, jobService, logger, settings, executionSettings, request, targets, progress, ct);
            return;
        }

        await ExecutePerTargetAsync(scopeFactory, jobService, logger, executionSettings, request, targets, progress, ct);
    }

    private static async Task ExecutePerTargetAsync(
        IServiceScopeFactory scopeFactory,
        IJobService jobService,
        ILogger<AiRunQueueService> logger,
        AiCoreConnectionSettings executionSettings,
        AiQueueRunRequest request,
        IReadOnlyList<AiResolvedRunTarget> targets,
        IJobProgress progress,
        CancellationToken ct)
    {
        progress.Report(0d, $"Preparing {targets.Count} target(s)");
        RegisterBatchUnits(progress, targets);

        var unitPositions = targets
            .Select((target, index) => new { target.UnitId, Position = index + 1 })
            .ToDictionary(static item => item.UnitId, static item => item.Position, StringComparer.Ordinal);
        var unitKind = BuildUnitKind(request);

        var batchResult = await jobService.RunBatchAsync(
            targets,
            executionSettings.MaxInFlight,
            async (target, unit, unitCt) =>
            {
                using var scope = scopeFactory.CreateScope();
                var orchestrator = scope.ServiceProvider.GetRequiredService<IAiCoreOrchestrator>();
                var position = unitPositions[target.UnitId];
                var processingMessage = BuildProcessingMessage(unitKind, position, targets.Count, target.Label);
                unit.Report(0.05d, processingMessage);
                string runId;
                try
                {
                    runId = await ExecuteTargetAsync(orchestrator, logger, executionSettings, request, target, unitCt);
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "AI queue item {UnitId} failed for {Path}", target.UnitId, target.Path);
                    throw;
                }
                unit.Report(1d, $"Completed {unitKind} {position} of {targets.Count}: {target.Label} (run {runId})");
            },
            progress,
            unitIdFactory: static (target, _) => target.UnitId,
            labelFactory: static target => target.Label,
            ct: ct);

        progress.Report(1d, batchResult.Summary);
    }

    private static async Task ExecuteImageBatchesAsync(
        IServiceScopeFactory scopeFactory,
        IJobService jobService,
        ILogger<AiRunQueueService> logger,
        AiCoreConnectionSettings settings,
        AiCoreConnectionSettings executionSettings,
        AiQueueRunRequest request,
        IReadOnlyList<AiResolvedRunTarget> targets,
        IJobProgress progress,
        CancellationToken ct)
    {
        var batchSize = Math.Max(1, settings.ImageBatchSize);
        var batches = new List<ImageBatch>();
        for (var start = 0; start < targets.Count; start += batchSize)
        {
            var slice = targets.Skip(start).Take(batchSize).ToArray();
            batches.Add(new ImageBatch(slice, start + 1, start + slice.Length));
        }

        progress.Report(0d, $"Preparing {targets.Count} image(s) in {batches.Count} batch(es)");
        foreach (var batch in batches)
        {
            progress.StartUnit(BatchUnitId(batch), BatchLabel(batch, targets.Count)).Dispose();
        }

        var template = BuildImageTemplate(request);
        var batchResult = await jobService.RunBatchAsync(
            batches,
            settings.MaxInFlight,
            async (batch, unit, unitCt) =>
            {
                using var scope = scopeFactory.CreateScope();
                var orchestrator = scope.ServiceProvider.GetRequiredService<IAiCoreOrchestrator>();
                unit.Report(0.05d, $"Processing {BatchLabel(batch, targets.Count)}");
                try
                {
                    var imageTargets = batch.Targets
                        .Select(static target => new AiRunImageTarget(target.Path, target.EntityType, target.EntityId))
                        .ToList();
                    await orchestrator.RunImageBatchAsync(executionSettings, imageTargets, template, unitCt);
                }
                catch (Exception ex)
                {
                    logger.LogError(ex, "AI image batch {UnitId} failed", BatchUnitId(batch));
                    throw;
                }
                unit.Report(1d, $"Completed {BatchLabel(batch, targets.Count)}");
            },
            progress,
            unitIdFactory: static (batch, _) => BatchUnitId(batch),
            labelFactory: batch => BatchLabel(batch, targets.Count),
            ct: ct);

        progress.Report(1d, batchResult.Summary);
    }

    private static string BatchUnitId(ImageBatch batch) => $"imagebatch:{batch.StartPosition}";

    private static string BatchLabel(ImageBatch batch, int total)
        => batch.StartPosition == batch.EndPosition
            ? $"image {batch.StartPosition} of {total}"
            : $"images {batch.StartPosition}–{batch.EndPosition} of {total}";

    private static AiRunImagesRequest BuildImageTemplate(AiQueueRunRequest request)
        => new()
        {
            Paths = [],
            PresetId = request.PresetId,
            CapabilityIds = request.CapabilityIds,
            ClaimIds = request.ClaimIds,
            PipelineName = request.PipelineName,
            ForceClaimIds = request.ForceClaimIds,
            Threshold = request.Threshold,
            ReturnConfidence = request.ReturnConfidence,
            CategoriesToSkip = request.CategoriesToSkip,
            LoadPolicy = request.LoadPolicy,
            DispatchResults = request.DispatchResults,
        };

    private sealed record ImageBatch(IReadOnlyList<AiResolvedRunTarget> Targets, int StartPosition, int EndPosition);

    private static void RegisterBatchUnits(IJobProgress progress, IReadOnlyList<AiResolvedRunTarget> targets)
    {
        foreach (var target in targets)
        {
            progress.StartUnit(target.UnitId, target.Label).Dispose();
        }
    }

    private static string BuildUnitKind(AiQueueRunRequest request)
        => request.EntityType switch
        {
            "video" => "Video",
            "image" => "Image",
            _ => request.MediaKind switch
            {
                AiMediaKinds.Image => "Image",
                AiMediaKinds.Audio => "Audio",
                _ => "Video",
            },
        };

    private static string BuildProcessingMessage(string unitKind, int position, int total, string label)
        => string.IsNullOrWhiteSpace(label)
            ? $"Processing {unitKind} {position} of {total}"
            : $"Processing {unitKind} {position} of {total}: {label}";

    private static async Task<string> ExecuteTargetAsync(
        IAiCoreOrchestrator orchestrator,
        ILogger<AiRunQueueService> logger,
        AiCoreConnectionSettings settings,
        AiQueueRunRequest request,
        AiResolvedRunTarget target,
        CancellationToken ct)
    {
        logger.LogInformation(
            "Running AI queue item {UnitId} ({MediaKind}) against {Path}",
            target.UnitId,
            request.MediaKind,
            target.Path);

        return request.MediaKind switch
        {
            AiMediaKinds.Image => (await orchestrator.RunImagesAsync(
                settings,
                new AiRunImagesRequest
                {
                    Paths = [target.Path],
                    EntityType = target.EntityType,
                    EntityId = target.EntityId,
                    PresetId = request.PresetId,
                    CapabilityIds = request.CapabilityIds,
                    ClaimIds = request.ClaimIds,
                    PipelineName = request.PipelineName,
                    ForceClaimIds = request.ForceClaimIds,
                    Threshold = request.Threshold,
                    ReturnConfidence = request.ReturnConfidence,
                    CategoriesToSkip = request.CategoriesToSkip,
                    LoadPolicy = request.LoadPolicy,
                    DispatchResults = request.DispatchResults,
                },
                ct)).RunId,
            AiMediaKinds.Video => (await orchestrator.RunVideoAsync(
                settings,
                new AiRunVideoRequest
                {
                    Path = target.Path,
                    EntityType = target.EntityType,
                    EntityId = target.EntityId,
                    PresetId = request.PresetId,
                    CapabilityIds = request.CapabilityIds,
                    ClaimIds = request.ClaimIds,
                    PipelineName = request.PipelineName,
                    ForceClaimIds = request.ForceClaimIds,
                    FrameInterval = request.FrameInterval,
                    Threshold = request.Threshold,
                    ReturnConfidence = request.ReturnConfidence,
                    VrVideo = request.VrVideo,
                    CategoriesToSkip = request.CategoriesToSkip,
                    LoadPolicy = request.LoadPolicy,
                    DispatchResults = request.DispatchResults,
                },
                ct)).RunId,
            AiMediaKinds.Audio => (await orchestrator.RunAudioAsync(
                settings,
                new AiRunAudioRequest
                {
                    Paths = [target.Path],
                    EntityType = target.EntityType,
                    EntityId = target.EntityId,
                    PresetId = request.PresetId,
                    CapabilityIds = request.CapabilityIds,
                    ClaimIds = request.ClaimIds,
                    PipelineName = request.PipelineName,
                    ForceClaimIds = request.ForceClaimIds,
                    Threshold = request.Threshold,
                    LoadPolicy = request.LoadPolicy,
                    DispatchResults = request.DispatchResults,
                },
                ct)).RunId,
            _ => throw new ArgumentException($"Unsupported media kind '{request.MediaKind}'.", nameof(request)),
        };
    }

    private static string BuildDescription(AiQueueRunRequest request, IReadOnlyList<AiResolvedRunTarget> targets)
    {
        var noun = request.EntityType switch
        {
            "video" when request.MediaKind == AiMediaKinds.Audio => "selected video audio track(s)",
            "video" => "selected video(s)",
            "image" => "selected image(s)",
            _ => request.MediaKind switch
            {
                AiMediaKinds.Image => "image file(s)",
                AiMediaKinds.Audio => "audio file(s)",
                _ => "video file(s)",
            },
        };

        return $"Run AI on {targets.Count} {noun}";
    }
}
