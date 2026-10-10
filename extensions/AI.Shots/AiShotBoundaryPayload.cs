using System.Text.Json;

namespace AI.Shots;

/// <summary>One element of the model server's shot-boundary partition.</summary>
internal sealed record ShotBoundaryElement(
    double StartSeconds,
    double EndSeconds,
    int? StartFrame,
    int? EndFrame,
    string? ShotType,
    string? TransitionIn,
    bool Filled);

/// <summary>The <c>analysis.other.shot_boundaries</c> node, contract v2 (see the model server's
/// <c>docs/asset-scope-models.md</c>).</summary>
internal sealed record ShotBoundaryNode(
    int SchemaVersion,
    string? Model,
    string? ModelVersion,
    string? Mode,
    double? Fps,
    double DurationSeconds,
    int? SourceFrameCount,
    int? WindowFrames,
    int? ContextFrames,
    string? DecodeBackend,
    double? DurationMismatchSeconds,
    IReadOnlyList<ShotBoundaryElement> Boundaries,
    JsonElement? LabelCounts);

internal enum ShotBoundaryReadError
{
    None = 0,
    Missing,
    Malformed,
    UnsupportedSchema,
}

internal sealed record ShotBoundaryReadResult(ShotBoundaryNode? Node, ShotBoundaryReadError Error, string? Message)
{
    public static ShotBoundaryReadResult Failure(ShotBoundaryReadError error, string message) => new(null, error, message);
}

/// <summary>
/// Reads the shot-boundary node that AI Core hands contributors as a raw JSON string in
/// <c>AssetAnalysis.Other["shot_boundaries"]</c>. It checks the node's shape only; Cove validates the partition itself.
/// </summary>
internal static class AiShotBoundaryPayload
{
    public const string OutputKey = "shot_boundaries";

    /// <summary>Version 1 was only ever sent by pre-release model servers; it had no frames and a misnamed
    /// transition label, so it is refused rather than guessed at.</summary>
    public const int MinimumSchemaVersion = 2;

    public static ShotBoundaryReadResult Read(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return ShotBoundaryReadResult.Failure(ShotBoundaryReadError.Missing, "The AI server returned no shot-boundary output.");
        }

        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(json);
        }
        catch (JsonException ex)
        {
            return ShotBoundaryReadResult.Failure(ShotBoundaryReadError.Malformed, $"The shot-boundary output is not valid JSON: {ex.Message}");
        }

        using (document)
        {
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                return ShotBoundaryReadResult.Failure(ShotBoundaryReadError.Malformed, "The shot-boundary output is not a JSON object.");
            }

            var schemaVersion = GetInt(root, "schema_version") ?? 1;
            if (schemaVersion < MinimumSchemaVersion)
            {
                return ShotBoundaryReadResult.Failure(
                    ShotBoundaryReadError.UnsupportedSchema,
                    $"The AI server sent shot boundaries in contract version {schemaVersion}; update it to a version that sends version {MinimumSchemaVersion} or later.");
            }

            var duration = GetDouble(root, "duration_seconds");
            if (duration is not > 0)
            {
                return ShotBoundaryReadResult.Failure(ShotBoundaryReadError.Malformed, "The shot-boundary output has no positive duration_seconds.");
            }

            if (!root.TryGetProperty("boundaries", out var boundariesElement) || boundariesElement.ValueKind != JsonValueKind.Array)
            {
                return ShotBoundaryReadResult.Failure(ShotBoundaryReadError.Malformed, "The shot-boundary output has no boundaries array.");
            }

            var boundaries = new List<ShotBoundaryElement>(boundariesElement.GetArrayLength());
            var index = 0;
            foreach (var element in boundariesElement.EnumerateArray())
            {
                var start = element.ValueKind == JsonValueKind.Object ? GetDouble(element, "start_seconds") : null;
                var end = element.ValueKind == JsonValueKind.Object ? GetDouble(element, "end_seconds") : null;
                if (start is null || end is null)
                {
                    return ShotBoundaryReadResult.Failure(ShotBoundaryReadError.Malformed, $"Shot-boundary element {index} has no start_seconds or end_seconds.");
                }

                var filled = element.TryGetProperty("filled", out var filledElement) && filledElement.ValueKind == JsonValueKind.True;
                boundaries.Add(new ShotBoundaryElement(
                    start.Value,
                    end.Value,
                    GetInt(element, "start_frame"),
                    GetInt(element, "end_frame"),
                    filled ? null : GetString(element, "shot_type"),
                    filled ? null : GetString(element, "transition_in"),
                    filled));
                index++;
            }

            if (boundaries.Count == 0)
            {
                return ShotBoundaryReadResult.Failure(ShotBoundaryReadError.Malformed, "The shot-boundary output has an empty boundaries array.");
            }

            return new ShotBoundaryReadResult(
                new ShotBoundaryNode(
                    schemaVersion,
                    GetString(root, "model"),
                    GetString(root, "model_version"),
                    GetString(root, "mode"),
                    GetDouble(root, "fps"),
                    duration.Value,
                    GetInt(root, "source_frame_count"),
                    GetInt(root, "window_frames"),
                    GetInt(root, "context_frames"),
                    GetString(root, "decode_backend"),
                    GetDouble(root, "duration_mismatch_seconds"),
                    boundaries,
                    root.TryGetProperty("label_counts", out var labelCounts) && labelCounts.ValueKind == JsonValueKind.Object
                        ? labelCounts.Clone()
                        : null),
                ShotBoundaryReadError.None,
                null);
        }
    }

    private static double? GetDouble(JsonElement element, string name)
        => element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var number) && double.IsFinite(number)
            ? number
            : null;

    private static int? GetInt(JsonElement element, string name)
        => element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out var number)
            ? number
            : null;

    private static string? GetString(JsonElement element, string name)
        => element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(value.GetString())
            ? value.GetString()!.Trim()
            : null;
}
