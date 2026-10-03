using System.Globalization;
using System.Text;
using System.Text.Json;

using AI.Extensions.Abstractions;

using Cove.Core.Common;
using Cove.Core.DTOs;

using Microsoft.Extensions.Logging;

namespace AI.Shots;

/// <summary>
/// Runs the shot-boundary model and stores its partition of each video file in Cove. It plans its own claim: a file
/// that already has shots is skipped before the (every-frame, whole-video) analysis is requested, and existing shots
/// are only replaced — hand edits included — when the run is forced. The Run AI dialog never ticks it on the user's
/// behalf.
/// </summary>
internal sealed class AiShotsContributor(
    AiShotsPersistenceService persistence,
    ILogger<AiShotsContributor> logger) : IAiCapabilityContributor, IAiClaimPlanningContributor
{
    public const string ExtensionId = "cove.community.ai.shots";
    public const string ClaimId = "shots.video.asset";
    public const string CapabilityId = "shots";
    public const string SlotId = "detector";

    private readonly AiShotsPersistenceService _persistence = persistence;
    private readonly ILogger<AiShotsContributor> _logger = logger;

    private static readonly AiCapabilityDescriptor Descriptor = new(
        ExtensionId,
        "AI Shots",
        [
            new AiCapabilityClaim(
                ClaimId,
                "Shot Boundaries",
                AiMediaKinds.Video,
                "temporal_segmentation",
                "asset",
                AiShotBoundaryPayload.OutputKey,
                Description: "Split each video file into its shots.")
            {
                CapabilityId = CapabilityId,
                ModelBindingSlotId = SlotId,
            },
        ])
    {
        Capabilities =
        [
            new AiCapabilityFeature(
                CapabilityId,
                "Shot Boundaries",
                [ClaimId],
                [
                    new AiModelBindingSlot(
                        SlotId,
                        "Shot boundary model",
                        "temporal_segmentation",
                        RequiredCapabilities: ["temporal_segmentation"],
                        RequiredScopes: ["asset"],
                        RequiredCategories: [AiShotBoundaryPayload.OutputKey]),
                ],
                "Split each video file into its shots. Reads every frame, so it is never ticked for you. Files that already have shots are skipped; Force rerun replaces them, including shots edited by hand.")
            {
                // Every frame of every selected video is decoded, and a forced rerun of anything else would replace
                // shots edited by hand: the user ticks it for each run that should include it.
                SelectedByDefault = false,
            },
        ],
    };

    public AiCapabilityDescriptor Describe() => Descriptor;

    public async Task<IReadOnlyList<AiClaimPlanningDecision>> PlanClaimsAsync(AiClaimPlanningTarget target, CancellationToken ct = default)
    {
        var decisions = new List<AiClaimPlanningDecision>();
        foreach (var claim in target.Claims.Where(static claim => string.Equals(claim.ClaimId, ClaimId, StringComparison.OrdinalIgnoreCase)))
        {
            decisions.Add(await PlanAsync(claim.ClaimId, target, ct));
        }

        return decisions;
    }

    // Nothing runs unless the result can be stored for a Cove video. A run must also have asked for shots: forced, or
    // selected rather than included by an implicit "all claims" run, which would start a whole-video decode nobody
    // asked for. Only then is the file looked up: a forced run replaces its shots, any other run skips a file that
    // already has shots.
    private async Task<AiClaimPlanningDecision> PlanAsync(string claimId, AiClaimPlanningTarget target, CancellationToken ct)
    {
        if (!target.DispatchResults)
        {
            return Skip(claimId, "Results are not being stored for this run.");
        }

        if (!IsVideo(target.HostEntityType) || target.HostEntityId is not int videoId)
        {
            return Skip(claimId, "Shot boundaries are stored for the files of Cove videos only.");
        }

        var forced = target.ForcedClaimIds.Contains(claimId, StringComparer.OrdinalIgnoreCase);
        if (!forced && !target.ClaimsExplicitlySelected)
        {
            return Skip(claimId, "Shot boundaries run only when selected: the model decodes every frame of the video.");
        }

        var file = await _persistence.ResolveFileAsync(videoId, target.HostFileId, target.Subject, ct);
        if (file is null)
        {
            return Skip(claimId, "The analysed file could not be identified.");
        }

        var existing = await _persistence.GetExistingSetAsync(file.FileId, ct);
        if (forced)
        {
            return existing is null
                ? Run(claimId, "Force rerun: the file has no shots yet.", replacesExistingResults: false)
                : Run(claimId, $"Force rerun: the file's {existing.ShotCount} shots will be replaced, including any edited by hand.", replacesExistingResults: true);
        }

        return existing is null
            ? Run(claimId, "The file has no shots yet.", replacesExistingResults: false)
            : Skip(claimId, $"The file already has {existing.ShotCount} shots; Force rerun replaces them.");
    }

    public async Task<AiDispatchResult> DispatchAsync(AiDispatchRequest request, CancellationToken ct = default)
    {
        var claims = request.Claims.Where(static claim => string.Equals(claim.ClaimId, ClaimId, StringComparison.OrdinalIgnoreCase)).ToArray();
        if (claims.Length == 0)
        {
            return new AiDispatchResult(ExtensionId, 0);
        }

        try
        {
            return await DispatchShotsAsync(request, claims.Length, ct);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            throw;
        }
        catch (Exception ex)
        {
            // Contributors are dispatched one after another; throwing would lose every later contributor's results.
            _logger.LogError(ex, "AI.Shots could not store shot boundaries for run {RunId}", request.Context.RunId);
            return Note(claims.Length, $"Shot boundaries were not stored: {ex.Message}");
        }
    }

    private async Task<AiDispatchResult> DispatchShotsAsync(AiDispatchRequest request, int claimCount, CancellationToken ct)
    {
        var context = request.Context;
        if (!IsVideo(context.HostEntityType) || context.HostEntityId is not int videoId)
        {
            return Note(claimCount, "Shot boundaries were not stored: the run has no Cove video.");
        }

        string? json = null;
        request.Result.AssetAnalysis?.Other.TryGetValue(AiShotBoundaryPayload.OutputKey, out json);
        var read = AiShotBoundaryPayload.Read(json);
        if (read.Node is null)
        {
            _logger.LogWarning("AI.Shots stored nothing for run {RunId}: {Reason}", context.RunId, read.Message);
            return Note(claimCount, read.Message!);
        }

        var file = await _persistence.ResolveFileAsync(
            videoId,
            request.TryGetHostFileId(out var namedFileId) ? namedFileId : null,
            context.Subject,
            ct);
        if (file is null)
        {
            _logger.LogWarning("AI.Shots stored nothing for run {RunId}: no file of video {VideoId} matches the analysed file", context.RunId, videoId);
            return Note(claimCount, "Shot boundaries were not stored: the analysed file could not be identified.");
        }

        var forced = request.IsForced(ClaimId);
        var write = BuildWrite(request, read.Node, file.FileId, forced);
        var result = await _persistence.WriteAsync(write, ct);

        _logger.LogInformation(
            "AI.Shots {Outcome} shot boundaries for run {RunId} video {VideoId} file {FileId}: {ShotCount} shots from {Model}",
            result.Outcome,
            context.RunId,
            videoId,
            file.FileId,
            write.Shots.Count,
            write.Model ?? "an unnamed model");

        switch (result.Outcome)
        {
            case VideoShotWriteOutcome.Written:
                return Counted(claimCount, "shots", result.Set?.ShotCount ?? write.Shots.Count, null);

            case VideoShotWriteOutcome.Replaced:
                if (result.ReplacedEditedSet)
                {
                    _logger.LogWarning("AI.Shots replaced hand-edited shot boundaries of file {FileId} (forced run {RunId})", file.FileId, context.RunId);
                }

                return Counted(
                    claimCount,
                    "shots",
                    result.Set?.ShotCount ?? write.Shots.Count,
                    result.ReplacedEditedSet ? "Replaced shot boundaries that had been edited by hand." : null);

            case VideoShotWriteOutcome.SkippedExisting:
                return Counted(claimCount, "shotSetsSkipped", 1, "The file already had shot boundaries; they were kept.");

            default:
                _logger.LogWarning(
                    "AI.Shots could not store shot boundaries for run {RunId} file {FileId}: {Outcome} {Reason}",
                    context.RunId,
                    file.FileId,
                    result.Outcome,
                    result.Reason);
                var durationProblem = result.Outcome == VideoShotWriteOutcome.Invalid
                    ? DescribeDurationProblem(read.Node.DurationSeconds, file.Duration)
                    : null;
                return Note(claimCount, durationProblem ?? $"Shot boundaries were not stored ({result.Outcome}): {result.Reason}");
        }
    }

    /// <summary>
    /// Why Cove refuses an analysis whose duration does not fit the file, or null when it fits. Mirrors Cove's rule
    /// (<see cref="VideoShotRules"/>): an analysis may run only slightly past the file's probed duration, and may fall
    /// short of it down to half, because a video stream can end before the container's audio does.
    /// </summary>
    internal static string? DescribeDurationProblem(double analysedSeconds, double fileSeconds)
    {
        if (!double.IsFinite(fileSeconds) || fileSeconds <= 0)
        {
            return null;
        }

        var overrun = Math.Max(VideoShotRules.DurationSanityMinSec, fileSeconds * VideoShotRules.DurationSanityFraction);
        if (analysedSeconds > fileSeconds + overrun)
        {
            return string.Create(
                CultureInfo.InvariantCulture,
                $"Shot boundaries were not stored: the analysis covered {analysedSeconds:0.###}s, longer than the {fileSeconds:0.###}s Cove has for the file, so the file has probably changed since Cove scanned it. Rescan the file, then run Shot Boundaries again.");
        }

        if (analysedSeconds < fileSeconds * VideoShotRules.DurationSanityShortestFraction)
        {
            return string.Create(
                CultureInfo.InvariantCulture,
                $"Shot boundaries were not stored: the analysis covered only {analysedSeconds:0.###}s, less than half of the {fileSeconds:0.###}s Cove has for the file. Either the file has changed since Cove scanned it, or its video ends long before its audio. Rescan the file and check that it plays to the end, then run Shot Boundaries again.");
        }

        return null;
    }

    internal static VideoShotSetWrite BuildWrite(AiDispatchRequest request, ShotBoundaryNode node, int fileId, bool forced)
    {
        // Cove stores frames for every shot or for none, and only with the frame rate that relates them to seconds.
        var withFrames = node.Fps is > 0
            && node.Boundaries.All(static element => element.StartFrame is not null && element.EndFrame is not null);
        var modelVersion = node.ModelVersion
            ?? request.Result.Models
                .FirstOrDefault(model => string.Equals(model.ConfigName, node.Model, StringComparison.OrdinalIgnoreCase)
                    || string.Equals(model.Name, node.Model, StringComparison.OrdinalIgnoreCase))
                ?.Version;

        return new VideoShotSetWrite
        {
            FileId = fileId,
            WriteMode = forced ? VideoShotWriteMode.Replace : VideoShotWriteMode.SkipIfExists,
            SourceKey = AiShotsPersistenceService.SourceKey,
            SourceRunId = request.Context.RunId,
            Model = Clean(node.Model, VideoShotRules.MaxKeyLength),
            ModelVersion = Clean(modelVersion, VideoShotRules.MaxLabelLength),
            Mode = Clean(node.Mode, VideoShotRules.MaxLabelLength),
            DecodeBackend = Clean(node.DecodeBackend, VideoShotRules.MaxLabelLength),
            Fps = node.Fps is > 0 ? node.Fps : null,
            FrameCount = withFrames && node.SourceFrameCount is > 0 ? node.SourceFrameCount : null,
            DurationSec = node.DurationSeconds,
            Shots = node.Boundaries
                .Select(element => new VideoShotInput
                {
                    StartSec = element.StartSeconds,
                    EndSec = element.EndSeconds,
                    StartFrame = withFrames ? element.StartFrame : null,
                    EndFrame = withFrames ? element.EndFrame : null,
                    ShotType = Clean(element.ShotType, VideoShotRules.MaxLabelLength),
                    TransitionIn = Clean(element.TransitionIn, VideoShotRules.MaxLabelLength),
                })
                .ToArray(),
            Payload = BuildPayload(node),
        };
    }

    // Model details Cove has no column for. The label counts are only a summary of the shots, so they are left out
    // rather than let Cove refuse the whole set: when the payload would exceed Cove's limit, measured as Cove measures
    // it, or when a label carries U+0000, which PostgreSQL cannot store.
    private static JsonElement BuildPayload(ShotBoundaryNode node)
    {
        var payload = JsonSerializer.SerializeToElement(new
        {
            schemaVersion = node.SchemaVersion,
            windowFrames = node.WindowFrames,
            contextFrames = node.ContextFrames,
            durationMismatchSeconds = node.DurationMismatchSeconds,
            filledShots = node.Boundaries.Count(static element => element.Filled),
            labelCounts = node.LabelCounts,
        });
        if (node.LabelCounts is null || (Encoding.UTF8.GetByteCount(payload.GetRawText()) <= VideoShotRules.MaxPayloadBytes && !ContainsNul(payload)))
        {
            return payload;
        }

        return JsonSerializer.SerializeToElement(new
        {
            schemaVersion = node.SchemaVersion,
            windowFrames = node.WindowFrames,
            contextFrames = node.ContextFrames,
            durationMismatchSeconds = node.DurationMismatchSeconds,
            filledShots = node.Boundaries.Count(static element => element.Filled),
        });
    }

    private static bool IsVideo(string? entityType)
        => string.Equals(entityType?.Trim(), "video", StringComparison.OrdinalIgnoreCase);

    // Drops U+0000, which PostgreSQL cannot store, and truncates to maxLength UTF-16 code units without splitting a
    // surrogate pair.
    internal static string? Clean(string? value, int maxLength)
    {
        if (value is null)
        {
            return null;
        }

        if (value.Contains('\0'))
        {
            value = value.Replace("\0", string.Empty);
        }

        if (value.Length <= maxLength)
        {
            return value;
        }

        var length = char.IsHighSurrogate(value[maxLength - 1]) ? maxLength - 1 : maxLength;
        return value[..length];
    }

    private static bool ContainsNul(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.String => element.GetString()!.Contains('\0'),
        JsonValueKind.Array => element.EnumerateArray().Any(ContainsNul),
        JsonValueKind.Object => element.EnumerateObject().Any(static property => property.Name.Contains('\0') || ContainsNul(property.Value)),
        _ => false,
    };

    private static AiClaimPlanningDecision Run(string claimId, string reason, bool replacesExistingResults)
        => new() { ClaimId = claimId, Verdict = AiClaimPlanningVerdict.Run, Reason = reason, ReplacesExistingResults = replacesExistingResults };

    private static AiClaimPlanningDecision Skip(string claimId, string reason)
        => new() { ClaimId = claimId, Verdict = AiClaimPlanningVerdict.Skip, Reason = reason };

    private static AiDispatchResult Counted(int claimCount, string key, int count, string? note)
        => new(
            ExtensionId,
            claimCount,
            new Dictionary<string, int>(StringComparer.Ordinal) { [key] = count },
            note is null ? null : [note]);

    private static AiDispatchResult Note(int claimCount, string note)
        => new(ExtensionId, claimCount, null, [note]);
}
