using System.Globalization;

namespace AI.Extensions.Abstractions;

/// <summary>
/// Metadata keys AI Core adds to <see cref="AiDispatchRequest.Metadata"/>, only for contributors that implement
/// <see cref="IAiClaimPlanningContributor"/>. Other contributors keep receiving exactly the metadata they always did.
/// </summary>
public static class AiDispatchMetadataKeys
{
    /// <summary>Comma-separated ids of this dispatch's claims that the run was asked to force (Run AI's
    /// "Force rerun and replace stored data").</summary>
    public const string ForcedClaimIds = "forcedClaimIds";

    /// <summary>The Cove file that was analysed, as an invariant integer, when AI Core knows it.</summary>
    public const string HostFileId = "hostFileId";
}

public static class AiDispatchMetadata
{
    public static bool TryGetHostFileId(this AiDispatchRequest request, out int fileId)
    {
        fileId = 0;
        return request.Metadata is not null
            && request.Metadata.TryGetValue(AiDispatchMetadataKeys.HostFileId, out var value)
            && int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out fileId);
    }

    public static bool IsForced(this AiDispatchRequest request, string claimId)
    {
        if (request.Metadata is null
            || string.IsNullOrWhiteSpace(claimId)
            || !request.Metadata.TryGetValue(AiDispatchMetadataKeys.ForcedClaimIds, out var value)
            || string.IsNullOrWhiteSpace(value))
        {
            return false;
        }

        return value
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Contains(claimId.Trim(), StringComparer.OrdinalIgnoreCase);
    }
}

/// <summary>What AI Core knows about a target when it asks a contributor to plan that contributor's claims.</summary>
public sealed record AiClaimPlanningTarget
{
    public string? RunId { get; init; }

    public required string MediaKind { get; init; }

    /// <summary>What is being analysed: the Cove path of the media file, before AI Core's path mapping, when a single
    /// file is analysed (every video run), otherwise a summary such as "3 image(s)".</summary>
    public required string Subject { get; init; }

    public string? HostEntityType { get; init; }

    public int? HostEntityId { get; init; }

    /// <summary>The Cove file that will be analysed, when AI Core resolved one.</summary>
    public int? HostFileId { get; init; }

    /// <summary>This contributor's claims that survived model resolution for this run.</summary>
    public required IReadOnlyList<AiCapabilityClaim> Claims { get; init; }

    /// <summary>The ids of <see cref="Claims"/> that the run was asked to force.</summary>
    public IReadOnlyList<string> ForcedClaimIds { get; init; } = [];

    /// <summary>False when the caller named no claims or capabilities and AI Core ran every registered claim by default.</summary>
    public bool ClaimsExplicitlySelected { get; init; }

    /// <summary>Whether results will be dispatched to contributors after analysis.</summary>
    public bool DispatchResults { get; init; } = true;
}

public enum AiClaimPlanningVerdict
{
    /// <summary>No opinion: AI Core plans the claim from its run history, as it does for every other claim.</summary>
    Default = 0,

    /// <summary>Run the claim's models, whatever earlier runs recorded.</summary>
    Run = 1,

    /// <summary>Do not run the claim's models for this target.</summary>
    Skip = 2,
}

public sealed record AiClaimPlanningDecision
{
    public required string ClaimId { get; init; }

    public required AiClaimPlanningVerdict Verdict { get; init; }

    public string? Reason { get; init; }

    /// <summary>With <see cref="AiClaimPlanningVerdict.Run"/>, that the run will replace results the contributor already
    /// stored for this target, so AI Core reports the claim as a rerun rather than a first run.</summary>
    public bool ReplacesExistingResults { get; init; }
}

/// <summary>
/// Optional companion to <see cref="IAiCapabilityContributor"/> for contributors whose results should be planned from
/// what the contributor has stored rather than from AI Core's run history. AI Core asks it once per target about its
/// claims that are about to run. A want (the claims that share a model request) is run when any of its claims is
/// answered <see cref="AiClaimPlanningVerdict.Run"/> and skipped when all of them are answered
/// <see cref="AiClaimPlanningVerdict.Skip"/>; such a want bypasses the run history entirely (no reruns for a newer
/// model version, a different threshold or frame interval) and never queues AI Core's artifact replacement, because
/// the contributor replaces its own results when it is dispatched. Any other want, including one answered
/// <see cref="AiClaimPlanningVerdict.Default"/>, is planned from run history like the claims of contributors that do
/// not implement this interface.
/// </summary>
public interface IAiClaimPlanningContributor
{
    Task<IReadOnlyList<AiClaimPlanningDecision>> PlanClaimsAsync(AiClaimPlanningTarget target, CancellationToken ct = default);
}
