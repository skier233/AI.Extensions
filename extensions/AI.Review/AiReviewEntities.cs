namespace AI.Review;

// One row per reviewed AI segment. The row outlives the segment: once a false positive is applied the
// Cove segment is deleted, but the verdict, the segment's original fields (for undo) and the exported
// frame paths stay here. AI re-runs replace segments by model key, so review state must never live on
// the segment itself.
internal sealed class ExtAiReviewMarkEntity
{
    public int Id { get; set; }

    public int VideoId { get; set; }

    // Null once the segment has been deleted from Cove (see SegmentDeletedAt). Restoring creates a new
    // segment and re-points this.
    public int? SegmentId { get; set; }

    public int TagId { get; set; }

    public string TagName { get; set; } = string.Empty;

    public string? ModelKey { get; set; }

    public string SourceKey { get; set; } = AiReviewSettings.DefaultSourceKey;

    public string? SourceRunId { get; set; }

    public string? PayloadJson { get; set; }

    public double StartSec { get; set; }

    public double EndSec { get; set; }

    public double? Confidence { get; set; }

    // "fp" (false positive: the tag is NOT present) or "tp" (true positive: the tag IS present).
    public string Verdict { get; set; } = AiReviewVerdicts.FalsePositive;

    public int? ReviewerUserId { get; set; }

    public string? ReviewerName { get; set; }

    public string? Note { get; set; }

    // Frame times (seconds) chosen for this segment; the review grid shows these same frames.
    public string FrameTimesJson { get; set; } = "[]";

    // "pending" | "done" | "failed" | "skipped"
    public string ExportStatus { get; set; } = AiReviewExportStatus.Pending;

    // JSON array of absolute file paths written by the exporter.
    public string ExportedFilesJson { get; set; } = "[]";

    public string? ExportError { get; set; }

    public DateTime? ExportedAt { get; set; }

    public DateTime? SegmentDeletedAt { get; set; }

    public DateTime CreatedAt { get; set; }

    public DateTime UpdatedAt { get; set; }
}

internal static class AiReviewVerdicts
{
    public const string FalsePositive = "fp";
    public const string TruePositive = "tp";

    public static bool IsValid(string? verdict)
        => verdict is FalsePositive or TruePositive;
}

internal static class AiReviewExportStatus
{
    public const string Pending = "pending";
    public const string Done = "done";
    public const string Failed = "failed";
    public const string Skipped = "skipped";
}
