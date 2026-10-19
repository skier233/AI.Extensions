using Cove.Core.DTOs;
using Cove.Core.Interfaces;

namespace AI.Review;

public sealed record AiReviewTagSummary(
    int TagId,
    string TagName,
    int SegmentCount,
    int UnreviewedCount,
    int FalsePositiveCount,
    int TruePositiveCount,
    int PendingDeleteCount);

public sealed record AiReviewMarkDto(
    int Id,
    int? SegmentId,
    string Verdict,
    string ExportStatus,
    int ExportedFrameCount,
    string? ExportError,
    DateTime? SegmentDeletedAt,
    string? ReviewerName,
    DateTime UpdatedAt);

// The segment exactly as GET /api/segments returns it, so the UI can hand it to Cove's own segment
// tiles and rows, plus the review verdict. A mark whose segment Cove has deleted carries a synthetic
// record with the negated mark id, so it still has a stable id in the list.
public sealed record AiReviewQueueItem(SegmentRecordDto Segment, AiReviewMarkDto? Mark);

public sealed record AiReviewQueuePage(
    IReadOnlyList<AiReviewQueueItem> Items,
    int TotalCount,
    int Page,
    int PerPage);

public static class AiReviewQueueFilters
{
    public const string Unreviewed = "unreviewed";
    public const string All = "all";
    public const string FalsePositive = "fp";
    public const string TruePositive = "tp";
    // False positives whose segment Cove has already deleted; they live only in the marks table.
    public const string Deleted = "deleted";
}

/// <summary>
/// The review queue request. The filter dialog's object filter arrives as the very
/// <see cref="SegmentFilter"/> the Segments page posts to <c>/api/segments/find</c>, and the search,
/// sort, seed and paging as its <see cref="FindFilter"/>; the review adds the tag, the verdict scope
/// and the frame-export state on top.
/// </summary>
public sealed class AiReviewQueueRequest
{
    public int? VideoId { get; set; }
    public int TagId { get; set; }
    // Verdict scope: unreviewed (default) | all | fp | tp | deleted
    public string Filter { get; set; } = AiReviewQueueFilters.Unreviewed;
    public SegmentFilter? ObjectFilter { get; set; }
    public FindFilter? FindFilter { get; set; }
    // Frame export state of the mark: pending | done | failed | skipped (reviewed segments only).
    public string? ExportStatus { get; set; }
}

public sealed record AiReviewSetVerdictRequest(int SegmentId, string Verdict, string? Note);

public sealed record AiReviewBulkVerdictRequest(IReadOnlyList<int> SegmentIds, string Verdict);

public sealed record AiReviewBulkClearRequest(IReadOnlyList<int> MarkIds);

public sealed record AiReviewClearVerdictRequest(int MarkId);

public sealed record AiReviewApplyRequest(int? VideoId, int? TagId);

public sealed record AiReviewApplyResult(int DeletedSegments, int AffectedVideos, int RemovedTagApplications);

public sealed record AiReviewExportStatusDto(int Pending, int Done, int Failed, int Skipped, string ExportRootPath, string ManifestPath);
