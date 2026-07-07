using System.Text.Json;

using Cove.Core.Entities;
using Cove.Core.Interfaces;

namespace AI.Core;

public interface IAiRunJournal
{
    Task RecordStartAsync(AiRunJournalStart entry, CancellationToken ct = default);

    Task RecordCompletionAsync(AiRunJournalCompletion completion, CancellationToken ct = default);

    Task RecordFailureAsync(string runKey, Exception exception, CancellationToken ct = default);

    /// <summary>Batch form of <see cref="RecordStartAsync"/>: writes all of a batch's run-start rows in a
    /// couple of round-trips instead of two per entity.</summary>
    Task RecordStartsAsync(IReadOnlyList<AiRunJournalStart> entries, CancellationToken ct = default);

    /// <summary>Batch form of <see cref="RecordCompletionAsync"/>.</summary>
    Task RecordCompletionsAsync(IReadOnlyList<AiRunJournalCompletion> completions, CancellationToken ct = default);
}

public sealed record AiRunJournalStart(
    string RunKey,
    string? HostEntityType,
    int? HostEntityId,
    string? Trigger,
    string? LoadPolicy,
    double? FrameIntervalSec,
    bool? Vr,
    object RequestPayload);

public sealed record AiRunJournalCompletion(
    string RunKey,
    string MediaKind,
    JsonElement Response,
    IReadOnlyList<string> ClaimIds,
    int DispatchResultCount);

internal sealed class AiRunJournal(IAiRunRepository runRepo) : IAiRunJournal
{
    private const string SourceKey = "ext:ai.core";

    private readonly IAiRunRepository _runRepo = runRepo;

    public async Task RecordStartAsync(AiRunJournalStart entry, CancellationToken ct = default)
    {
        if (!TryResolveTarget(entry.HostEntityType, entry.HostEntityId, out var targetType, out var targetId))
        {
            return;
        }

        var run = await _runRepo.FindOrCreateAsync(entry.RunKey, SourceKey, targetType, targetId, AiRunStatus.Running, ct);

        run.Trigger = entry.Trigger;
        run.Status = AiRunStatus.Running;
        run.LoadPolicy = entry.LoadPolicy;
        run.FrameIntervalSec = entry.FrameIntervalSec;
        run.Vr = entry.Vr;
        run.Request = Serialize(entry.RequestPayload);
        if (run.StartedAt == default)
        {
            run.StartedAt = DateTime.UtcNow;
        }

        await _runRepo.UpdateAsync(run, ct);
    }

    public async Task RecordCompletionAsync(AiRunJournalCompletion completion, CancellationToken ct = default)
    {
        if (!TryResolveRunKey(completion.RunKey, out var runKey))
        {
            return;
        }

        var run = await _runRepo.FindOrCreateAsync(runKey, SourceKey, default, 0, AiRunStatus.Running, ct);
        if (run is null)
        {
            return;
        }

        run.Status = AiRunStatus.Completed;
        run.CompletedAt = DateTime.UtcNow;
        run.Models = ExtractProperty(completion.Response, "models");
        run.Summary = BuildSummary(completion);
        run.Error = null;

        await _runRepo.UpdateAsync(run, ct);
    }

    public async Task RecordStartsAsync(IReadOnlyList<AiRunJournalStart> entries, CancellationToken ct = default)
    {
        var resolved = new List<(AiRunJournalStart Entry, AiRunTargetType Type, int Id)>(entries.Count);
        foreach (var entry in entries)
        {
            if (TryResolveTarget(entry.HostEntityType, entry.HostEntityId, out var targetType, out var targetId))
            {
                resolved.Add((entry, targetType, targetId));
            }
        }

        if (resolved.Count == 0)
        {
            return;
        }

        var runs = await _runRepo.FindOrCreateManyAsync(
            resolved.Select(static item => (item.Entry.RunKey, item.Type, item.Id)).ToArray(),
            SourceKey,
            AiRunStatus.Running,
            ct);

        var now = DateTime.UtcNow;
        var toUpdate = new List<AiRun>(resolved.Count);
        foreach (var (entry, _, _) in resolved)
        {
            if (!runs.TryGetValue(entry.RunKey, out var run))
            {
                continue;
            }

            run.Trigger = entry.Trigger;
            run.Status = AiRunStatus.Running;
            run.LoadPolicy = entry.LoadPolicy;
            run.FrameIntervalSec = entry.FrameIntervalSec;
            run.Vr = entry.Vr;
            run.Request = Serialize(entry.RequestPayload);
            if (run.StartedAt == default)
            {
                run.StartedAt = now;
            }

            toUpdate.Add(run);
        }

        await _runRepo.UpdateManyAsync(toUpdate, ct);
    }

    public async Task RecordCompletionsAsync(IReadOnlyList<AiRunJournalCompletion> completions, CancellationToken ct = default)
    {
        var valid = completions
            .Where(static completion => TryResolveRunKey(completion.RunKey, out _))
            .ToArray();
        if (valid.Length == 0)
        {
            return;
        }

        var runs = await _runRepo.FindOrCreateManyAsync(
            valid.Select(static completion => (completion.RunKey, default(AiRunTargetType), 0)).ToArray(),
            SourceKey,
            AiRunStatus.Running,
            ct);

        var now = DateTime.UtcNow;
        var toUpdate = new List<AiRun>(valid.Length);
        foreach (var completion in valid)
        {
            if (!runs.TryGetValue(completion.RunKey, out var run))
            {
                continue;
            }

            run.Status = AiRunStatus.Completed;
            run.CompletedAt = now;
            run.Models = ExtractProperty(completion.Response, "models");
            run.Summary = BuildSummary(completion);
            run.Error = null;
            toUpdate.Add(run);
        }

        await _runRepo.UpdateManyAsync(toUpdate, ct);
    }

    public async Task RecordFailureAsync(string runKey, Exception exception, CancellationToken ct = default)
    {
        var effectiveCt = ct.IsCancellationRequested ? CancellationToken.None : ct;

        if (!TryResolveRunKey(runKey, out var resolvedRunKey))
        {
            return;
        }

        var run = await _runRepo.FindOrCreateAsync(resolvedRunKey, SourceKey, default, 0, AiRunStatus.Running, effectiveCt);
        if (run is null)
        {
            return;
        }

        run.Status = exception is OperationCanceledException ? AiRunStatus.Cancelled : AiRunStatus.Failed;
        run.CompletedAt = DateTime.UtcNow;
        run.Error = exception.Message;

        await _runRepo.UpdateAsync(run, effectiveCt);
    }

    private static bool TryResolveRunKey(string? runKey, out string resolved)
    {
        resolved = runKey ?? string.Empty;
        return !string.IsNullOrWhiteSpace(runKey);
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

    private static JsonDocument Serialize(object value)
        => JsonDocument.Parse(JsonSerializer.Serialize(value));

    private static JsonDocument? ExtractProperty(JsonElement element, string propertyName)
    {
        if (element.ValueKind != JsonValueKind.Object || !element.TryGetProperty(propertyName, out var property))
        {
            return null;
        }

        return JsonDocument.Parse(property.GetRawText());
    }

    private static JsonDocument BuildSummary(AiRunJournalCompletion completion)
    {
        double? durationSeconds = null;
        if (completion.Response.ValueKind == JsonValueKind.Object
            && completion.Response.TryGetProperty("duration_seconds", out var durationElement)
            && durationElement.ValueKind == JsonValueKind.Number
            && durationElement.TryGetDouble(out var durationValue))
        {
            durationSeconds = durationValue;
        }

        double? frameIntervalSeconds = null;
        if (completion.Response.ValueKind == JsonValueKind.Object
            && completion.Response.TryGetProperty("frame_interval_seconds", out var frameIntervalElement)
            && frameIntervalElement.ValueKind == JsonValueKind.Number
            && frameIntervalElement.TryGetDouble(out var frameIntervalValue))
        {
            frameIntervalSeconds = frameIntervalValue;
        }

        return Serialize(new
        {
            mediaKind = completion.MediaKind,
            claimIds = completion.ClaimIds,
            dispatchResultCount = completion.DispatchResultCount,
            durationSeconds,
            frameIntervalSeconds,
        });
    }
}
