import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, FolderOpen, Library, Loader2, Play, RotateCcw, Trash2, X } from "lucide-react";
import { extensionFetch } from "@cove/runtime/api";
import {
  ConfirmDialog,
  DetailListPagination,
  DetailListToolbar,
  Field,
  InfiniteScrollSentinel,
  ListPage,
  NumberInput,
  RAW_SEGMENT_SORT_OPTIONS,
  RawSegmentListHeader,
  RawSegmentListRow,
  SaveButton,
  SegmentTile,
  SelectInput,
  TextInput,
  VirtualizedEntityGrid,
  buildSegmentFilteredQuery,
  createSegmentCriteria,
  formatSegmentCardEyebrow,
  getEntityCardMinWidthPx,
  normalizeCriterionPayload,
  readMultiIdCriterionDepth,
  readMultiIdCriterionIds,
  readRawSegmentListFilter,
  useExtensionKeyboardBindings,
  useListPageCardMinWidthPx,
  useListUrlState,
  useMultiSelect,
  usePaginatedInfiniteQuery,
  useRegisterExtensionKeyboardActions,
  useSegmentListDensity,
  type CriterionDefinition,
  type RawSegmentItem,
  type SegmentFilteredQueryRequest,
} from "@cove/runtime/components";

// The review board is Cove's segment list with verdicts on top. Cove owns the list chrome (search,
// filter dialog, sort, display modes, card size, selection bar, paging, infinite scroll), the cards and
// rows (segment tiles with their thumbnails and hover previews) and the keyboard shortcut system; the
// extension adds the tag queue, the verdict state, keyboard focus and the review actions.
//
// Styling: the extension's Tailwind bundle is scoped to `.ai-review-shell` and loads after Cove's CSS,
// so anything inside that class would see the extension's copy of a utility win over Cove's responsive
// variants. Only elements the extension renders itself carry the class; Cove's list page, toolbar,
// tiles and rows are never placed inside it.

// ─── Constants ─────────────────────────────────────────────────────────────

const EXTENSION_ID = "cove.community.ai.review";
const API_BASE = "/api/ext/ai-review";
const DEFAULT_PER_PAGE = 48;
const VERDICT_FP = "fp";
const VERDICT_TP = "tp";
const VERDICT_UNREVIEWED = "unreviewed";
const VERDICT_DELETED = "deleted";
const CARD_SIZE_ENTITY_TYPE = "segment";

// Keep in sync with AiReviewExtension.GetUIManifest. The first key of each list is the one the hints
// show; the video page's player owns F and the arrow keys, so the primaries avoid them.
const DEFAULT_BINDINGS: Record<string, string[]> = {
  "review-false-positive": ["X", "F"],
  "review-true-positive": ["T"],
  "review-clear": ["U"],
  "review-open": ["Enter"],
  "review-next": ["N", "ArrowRight"],
  "review-previous": ["P", "ArrowLeft"],
  "review-down": ["ArrowDown"],
  "review-up": ["ArrowUp"],
};
const KEY_LABELS: Record<string, string> = { ArrowRight: "→", ArrowLeft: "←", ArrowDown: "↓", ArrowUp: "↑" };

// Cove's segment sorts, minus the ones that are constant inside one tag's queue (tag, kind, source) or
// are about faces and performers.
const SORT_KEYS = new Set(["random", "confidence", "duration", "start_sec", "end_sec", "video_title", "created_at", "updated_at"]);
const SORT_OPTIONS = RAW_SEGMENT_SORT_OPTIONS.filter((option) => SORT_KEYS.has(option.value));

const VERDICT_OPTIONS = [
  { value: VERDICT_UNREVIEWED, label: "Unreviewed" },
  { value: VERDICT_FP, label: "False positive" },
  { value: VERDICT_TP, label: "True positive" },
  { value: VERDICT_DELETED, label: "False positive, deleted" },
];
const EXPORT_STATUS_OPTIONS = [
  { value: "pending", label: "Pending" },
  { value: "done", label: "Exported" },
  { value: "failed", label: "Failed" },
  { value: "skipped", label: "Skipped" },
];

// Cove's own segment criteria that make sense inside one tag's queue. Their object-filter keys are the
// Segments page's, so Cove's helpers turn them into the SegmentFilter the server applies.
const SEGMENT_CRITERION_IDS = new Set(["videoTitle", "videos", "videoTags", "createdAt", "updatedAt", "startSec", "endSec", "performers", "confidence", "duration"]);

function buildReviewCriteria(tags: ReviewTag[]): CriterionDefinition[] {
  return [
    {
      id: "tag",
      label: "Tag",
      type: "enum",
      filterKey: "tagCriterion",
      modifiers: ["EQUALS"],
      options: tags.map((tag) => ({ value: String(tag.tagId), label: `${tag.tagName} (${tag.unreviewedCount} unreviewed)` })),
    },
    { id: "verdict", label: "Verdict", type: "enum", filterKey: "verdictCriterion", modifiers: ["EQUALS"], options: VERDICT_OPTIONS },
    { id: "exportStatus", label: "Frame export", type: "enum", filterKey: "exportStatusCriterion", modifiers: ["EQUALS"], options: EXPORT_STATUS_OPTIONS },
    ...createSegmentCriteria().filter((criterion) => SEGMENT_CRITERION_IDS.has(criterion.id)),
  ];
}

const DEFAULT_FILTER = { page: 1, perPage: DEFAULT_PER_PAGE, sort: "confidence", direction: "asc" as const };
const DEFAULT_OBJECT_FILTER = { verdictCriterion: { modifier: "EQUALS", value: VERDICT_UNREVIEWED } };

// Cove's button looks, as its list pages use them.
const BTN = "inline-flex items-center gap-1.5 rounded-lg border border-border bg-card/70 px-2.5 py-1.5 text-xs text-muted transition-colors hover:text-foreground disabled:cursor-not-allowed disabled:opacity-60";
const BTN_DANGER = "inline-flex items-center gap-1.5 rounded-lg border border-red-500/40 bg-card/70 px-2.5 py-1.5 text-xs text-red-400 transition-colors hover:bg-red-900/20 hover:text-red-300 disabled:cursor-not-allowed disabled:opacity-60";
const SELECTION_ACTION = "flex items-center gap-1 rounded px-2 py-0.5 text-xs hover:bg-surface/60";

// ─── Types ─────────────────────────────────────────────────────────────────

interface ReviewMark {
  id: number;
  segmentId: number | null;
  verdict: string;
  exportStatus: string;
  exportedFrameCount: number;
  exportError: string | null;
  segmentDeletedAt: string | null;
  reviewerName: string | null;
  updatedAt: string;
}

/** What the queue endpoint returns: the segment as GET /api/segments has it, plus the verdict. */
interface QueueItem {
  segment: Omit<RawSegmentItem, "key" | "videoId" | "videoTitle">;
  mark: ReviewMark | null;
}

/** A queue item as the board holds it: a stable id (negative for a deleted segment) and Cove's row shape. */
interface ReviewEntry {
  id: number;
  segment: RawSegmentItem;
  mark: ReviewMark | null;
}

interface ReviewTag {
  tagId: number;
  tagName: string;
  segmentCount: number;
  unreviewedCount: number;
  falsePositiveCount: number;
  truePositiveCount: number;
  pendingDeleteCount: number;
}

interface EntriesPage {
  items: ReviewEntry[];
  totalCount: number;
  page: number;
  perPage: number;
}

// ─── API ───────────────────────────────────────────────────────────────────

async function api<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await extensionFetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });

  const text = await response.text();
  let data: any = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const detail =
      data && typeof data === "object" && data.detail
        ? data.detail
        : data && typeof data === "object" && data.message
          ? data.message
          : typeof data === "string" && data.length > 0
            ? data
            : response.statusText;
    throw new Error(detail || "Request failed.");
  }

  return data as T;
}

function toEntry(item: QueueItem): ReviewEntry {
  // The same row shape the Segments page builds from a segment record.
  const hostId = item.segment.hostId;
  return {
    id: item.segment.id,
    segment: {
      ...item.segment,
      key: `segment:${item.segment.id}`,
      videoId: hostId,
      videoTitle: item.segment.hostTitle?.trim() || `Video #${hostId}`,
    },
    mark: item.mark,
  };
}

function isEditableTarget(target: EventTarget | null) {
  if (!target || !(target instanceof Element)) return false;
  const tag = target.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return true;
  return (target as HTMLElement).isContentEditable;
}

function keyLabel(binding: string) {
  return KEY_LABELS[binding] || (binding === "Escape" ? "Esc" : binding);
}

function readEnumCriterion(value: unknown): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const candidate = (value as { value?: unknown }).value;
  return typeof candidate === "string" && candidate ? candidate : undefined;
}

// ─── Small pieces ──────────────────────────────────────────────────────────

function Keycap({ children }: { children: React.ReactNode }) {
  return <span className="ai-review-keycap">{children}</span>;
}

function KeyHints({ bindings, libraryMode }: { bindings: Record<string, string[]> | null; libraryMode: boolean }) {
  const keysFor = (id: string, all = false) => {
    const custom = bindings && bindings[id] && bindings[id].length > 0 ? bindings[id] : null;
    const list = custom || (all ? DEFAULT_BINDINGS[id] : DEFAULT_BINDINGS[id].slice(0, 1));
    return list.map(keyLabel);
  };
  const hint = (keys: string[], label: string) => (
    <span className="inline-flex items-center gap-1 whitespace-nowrap">
      {keys.map((key, index) => (
        <Keycap key={`${key}-${index}`}>{key}</Keycap>
      ))}
      <span className="text-muted">{label}</span>
    </span>
  );
  const moveKeys = libraryMode
    ? [...keysFor("review-previous", true), ...keysFor("review-next", true), ...keysFor("review-up"), ...keysFor("review-down")]
    : [...keysFor("review-previous"), ...keysFor("review-next")];
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
      {hint(keysFor("review-false-positive"), "false positive")}
      {hint(keysFor("review-true-positive"), "true positive")}
      {hint(keysFor("review-clear"), "unmark")}
      {hint(keysFor("review-open"), "open segment")}
      {hint(moveKeys, libraryMode ? "move" : "previous / next")}
      {libraryMode ? null : <span className="text-muted">(arrow keys and F belong to the player here)</span>}
    </div>
  );
}

function verdictText(entry: ReviewEntry) {
  const verdict = entry.mark ? entry.mark.verdict : null;
  if (!verdict) return null;
  if (verdict === VERDICT_TP) return "True positive";
  return entry.mark && entry.mark.segmentDeletedAt ? "False positive · deleted" : "False positive";
}

function verdictColor(entry: ReviewEntry) {
  const verdict = entry.mark ? entry.mark.verdict : null;
  return verdict === VERDICT_FP ? "text-red-400" : verdict === VERDICT_TP ? "text-emerald-400" : "text-secondary";
}

function VerdictBadge({ entry }: { entry: ReviewEntry }) {
  const verdict = entry.mark ? entry.mark.verdict : null;
  if (!verdict) return null;
  return (
    <span
      className={`ai-review-shell pointer-events-none absolute right-2 top-2 z-20 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wide text-white ${verdict === VERDICT_FP ? "bg-red-600" : "bg-emerald-600"}`}
    >
      {verdictText(entry)}
    </span>
  );
}

function ExportStatus({ entry }: { entry: ReviewEntry }) {
  if (!entry.mark) return null;
  if (entry.mark.exportStatus === "failed") {
    return (
      <span className="ai-review-shell text-red-400" title={entry.mark.exportError || "Frame export failed"}>
        export failed
      </span>
    );
  }
  if (entry.mark.exportStatus === "pending") return <span className="ai-review-shell text-muted">exporting…</span>;
  return null;
}

const RING_ACCENT = "var(--color-accent, #7c5cff)";
const RING_FP = "rgba(229, 72, 77, 0.7)";
const RING_TP = "rgba(48, 164, 108, 0.7)";

/** Keyboard focus and verdict as a ring around Cove's tile (or a rail on Cove's row). */
function reviewStateStyle(entry: ReviewEntry, focused: boolean, row: boolean): React.CSSProperties {
  const verdict = entry.mark ? entry.mark.verdict : null;
  const verdictColor = verdict === VERDICT_FP ? RING_FP : verdict === VERDICT_TP ? RING_TP : null;
  const color = focused ? RING_ACCENT : verdictColor;
  const style: React.CSSProperties = {};
  if (color) style.boxShadow = row ? `inset 3px 0 0 ${color}` : `0 0 0 2px ${color}`;
  if (entry.id < 0) {
    style.opacity = 0.55;
    style.filter = "grayscale(0.8)";
  }
  return style;
}

/** Where a tile or row leads: the segment page, or the video for a segment Cove already deleted. */
function entryRoute(entry: ReviewEntry) {
  return entry.id > 0 ? { page: "segment", id: entry.id } : { page: "video", id: entry.segment.hostId, seekTo: entry.segment.startSec };
}

// ─── Cards and rows: Cove's, with the review state layered on ──────────────

interface EntryProps {
  entry: ReviewEntry;
  focused: boolean;
  selected: boolean;
  selecting: boolean;
  libraryMode: boolean;
  onFocus: () => void;
  onToggle: (options?: any) => void;
  onNavigate: (route: any) => void;
}

function ReviewCard({ entry, focused, selected, selecting, libraryMode, onFocus, onToggle, onNavigate }: EntryProps) {
  return (
    <div
      data-item-id={String(entry.id)}
      className="relative h-full rounded"
      style={reviewStateStyle(entry, focused, false)}
      onMouseDown={(event) => {
        if (event.button === 0) onFocus();
      }}
    >
      <SegmentTile
        segment={entry.segment}
        route={entryRoute(entry)}
        label={`Review ${entry.segment.videoTitle}`}
        eyebrow={formatSegmentCardEyebrow(entry.segment.startSec, entry.segment.endSec)}
        onClick={(options) => (selecting ? onToggle(options) : onNavigate(entryRoute(entry)))}
        selected={selected}
        selecting={selecting}
        onSelect={onToggle}
        footer={
          <div className="flex items-center justify-between gap-2">
            <span className={`ai-review-shell ${verdictColor(entry)}`}>{verdictText(entry) || "Unreviewed"}</span>
            <span className="flex items-center gap-2">
              <ExportStatus entry={entry} />
              {libraryMode ? (
                <button
                  type="button"
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    onNavigate({ page: "video", id: entry.segment.hostId, seekTo: entry.segment.startSec });
                  }}
                  className="inline-flex items-center gap-1 text-accent hover:underline"
                >
                  <FolderOpen className="h-3.5 w-3.5" />
                  Open video
                </button>
              ) : null}
            </span>
          </div>
        }
      />
      <VerdictBadge entry={entry} />
    </div>
  );
}

function ReviewRow({ entry, focused, selected, selecting, libraryMode, onFocus, onToggle, onNavigate }: EntryProps) {
  const density = useSegmentListDensity();
  return (
    <div
      data-item-id={String(entry.id)}
      className="relative"
      style={reviewStateStyle(entry, focused, true)}
      onMouseDown={(event) => {
        if (event.button === 0) onFocus();
      }}
    >
      <RawSegmentListRow
        item={entry.segment}
        canReadVideos={libraryMode}
        onNavigate={(route: any) => onNavigate(route && route.page === "segment" && entry.id < 0 ? entryRoute(entry) : route)}
        selected={selected}
        onToggle={onToggle}
        selecting={selecting}
        density={density}
      />
      <VerdictBadge entry={entry} />
    </div>
  );
}

// ─── Data ──────────────────────────────────────────────────────────────────

function useTagsQuery(videoId: number | null) {
  return useQuery<ReviewTag[]>({
    queryKey: ["ai-review", "tags", videoId == null ? "library" : videoId],
    queryFn: () => api(`/tags${videoId == null ? "" : `?videoId=${videoId}`}`),
    staleTime: 30_000,
  });
}

interface QueueScope {
  videoId: number | null;
  tagId: number | null;
  verdictFilter: string;
  exportStatus: string | undefined;
  objectFilter: Record<string, unknown>;
  filter: { q?: string; sort?: string; direction?: "asc" | "desc"; seed?: number };
}

/**
 * The review queue request. The filter dialog's object filter becomes the very SegmentFilter the
 * Segments page posts (Cove's helpers do that mapping), and the review's own tag, verdict and
 * export-status criteria ride alongside it.
 */
function buildQueueBody(scope: QueueScope, page: number, perPage: number) {
  const raw = readRawSegmentListFilter(scope.objectFilter);
  const request: SegmentFilteredQueryRequest = buildSegmentFilteredQuery({
    pageNumber: page,
    perPage,
    q: scope.filter.q || "",
    videoTitle: raw.videoTitle ?? "",
    videoTagIds: readMultiIdCriterionIds(scope.objectFilter.videoTagsCriterion),
    videoTagDepth: readMultiIdCriterionDepth(scope.objectFilter.videoTagsCriterion),
    sort: scope.filter.sort || "confidence",
    direction: scope.filter.direction || "asc",
    seed: scope.filter.seed,
    includeVideoIds: raw.videoIds,
    excludeVideoIds: raw.excludeVideoIds,
    rawSegmentIds: [],
    rawFilter: raw,
  });
  return {
    tagId: scope.tagId ?? 0,
    videoId: scope.videoId,
    filter: scope.verdictFilter,
    exportStatus: scope.exportStatus,
    objectFilter: request.objectFilter,
    findFilter: request.findFilter,
  };
}

async function fetchQueuePage(scope: QueueScope, page: number, perPage: number): Promise<EntriesPage> {
  const result = await api<{ items: QueueItem[]; totalCount: number; page: number; perPage: number }>("/queue", {
    method: "POST",
    body: JSON.stringify(normalizeCriterionPayload(buildQueueBody(scope, page, perPage))),
  });
  return { items: result.items.map(toEntry), totalCount: result.totalCount, page: result.page, perPage: result.perPage };
}

interface QueueState {
  items: ReviewEntry[];
  totalCount: number;
  isPending: boolean;
  error: Error | null;
  refetch: () => void;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  loadMore: () => void;
}

/** One hook for both of Cove's paging styles: a page at a time, or Cove's infinite query where pages accumulate. */
function useQueue(scope: QueueScope, infinite: boolean, page: number, perPage: number): QueueState {
  const enabled = scope.tagId != null;
  const keyBase = ["ai-review", "queue", JSON.stringify(scope)];

  const paged = useQuery<EntriesPage>({
    queryKey: [...keyBase, "page", page, perPage],
    queryFn: () => fetchQueuePage(scope, page, perPage),
    enabled: enabled && !infinite,
    placeholderData: (previous) => previous,
  });

  const endless = usePaginatedInfiniteQuery<ReviewEntry>({
    queryKey: [...keyBase, "infinite"],
    queryFn: (pageNumber, chunk) => fetchQueuePage(scope, pageNumber, chunk),
    enabled: enabled && infinite,
    chunkSize: DEFAULT_PER_PAGE,
  });

  if (!enabled) {
    return { items: [], totalCount: 0, isPending: false, error: null, refetch: () => {}, hasNextPage: false, isFetchingNextPage: false, loadMore: () => {} };
  }

  if (infinite) {
    return {
      items: endless.items,
      totalCount: endless.totalCount,
      isPending: endless.isPending,
      error: endless.error,
      refetch: () => void endless.refetch(),
      hasNextPage: Boolean(endless.hasNextPage),
      isFetchingNextPage: endless.isFetchingNextPage,
      loadMore: () => {
        if (endless.hasNextPage && !endless.isFetchingNextPage) void endless.fetchNextPage();
      },
    };
  }

  return {
    items: paged.data ? paged.data.items : [],
    totalCount: paged.data ? paged.data.totalCount : 0,
    isPending: paged.isPending,
    error: paged.error,
    refetch: () => void paged.refetch(),
    hasNextPage: false,
    isFetchingNextPage: false,
    loadMore: () => {},
  };
}

// ─── Review board ──────────────────────────────────────────────────────────

interface ReviewBoardProps {
  videoId: number | null;
  onNavigate?: (route: any) => void;
  zoomLevel?: number;
  onZoomChange?: (level: number) => void;
}

function ReviewBoard({ videoId, onNavigate, zoomLevel, onZoomChange }: ReviewBoardProps) {
  const libraryMode = videoId == null;
  const queryClient = useQueryClient();

  // Cove list state. On the library page it lives in the URL (shareable); inside the video tab it is local.
  const listState = useListUrlState({
    resetKey: libraryMode ? "ai-review" : `ai-review-video-${videoId}`,
    defaultFilter: DEFAULT_FILTER,
    defaultObjectFilter: DEFAULT_OBJECT_FILTER,
    defaultDisplayMode: "grid",
    allowedDisplayModes: ["grid", "list"],
    allowInfinitePageSize: true,
    enabled: libraryMode,
  });
  const { filter, setFilter, objectFilter, setObjectFilter, displayMode, setDisplayMode } = listState;
  // Cove's toolbars offer every display mode in their type; this board only renders grid and list.
  const changeDisplayMode = useCallback(
    (mode: string) => {
      if (mode === "grid" || mode === "list") setDisplayMode(mode);
    },
    [setDisplayMode],
  );
  const tagCriterionValue = readEnumCriterion(objectFilter.tagCriterion);
  const tagId: number | null = tagCriterionValue && Number.isFinite(Number(tagCriterionValue)) ? Number(tagCriterionValue) : null;
  const verdictFilter = readEnumCriterion(objectFilter.verdictCriterion) || "all";
  const exportStatus = readEnumCriterion(objectFilter.exportStatusCriterion);
  const perPage: number = filter.perPage == null ? DEFAULT_PER_PAGE : filter.perPage;
  const infinite = perPage === 0;
  const page: number = filter.page || 1;

  const setTagId = useCallback(
    (next: number | null) => {
      setObjectFilter({ ...objectFilter, tagCriterion: next == null ? undefined : { modifier: "EQUALS", value: String(next) } });
      setFilter({ ...filter, page: 1 });
    },
    [objectFilter, setObjectFilter, filter, setFilter],
  );

  // Tags with counts; verdict changes adjust the counts locally until the next refetch.
  const tagsQuery = useTagsQuery(videoId);
  const [countAdjust, setCountAdjust] = useState<{ at: number; byTag: Record<number, any> }>({ at: 0, byTag: {} });
  const tags = useMemo<ReviewTag[]>(() => {
    const base = Array.isArray(tagsQuery.data) ? tagsQuery.data : [];
    const adjust = countAdjust.at === tagsQuery.dataUpdatedAt ? countAdjust.byTag : {};
    return base.map((tag) => {
      const delta = adjust[tag.tagId];
      return delta
        ? {
            ...tag,
            unreviewedCount: Math.max(0, tag.unreviewedCount + delta.unreviewed),
            falsePositiveCount: tag.falsePositiveCount + delta.fp,
            truePositiveCount: tag.truePositiveCount + delta.tp,
            pendingDeleteCount: Math.max(0, tag.pendingDeleteCount + delta.pending),
            segmentCount: tag.segmentCount + delta.segments,
          }
        : tag;
    });
  }, [tagsQuery.data, tagsQuery.dataUpdatedAt, countAdjust]);
  const bumpTag = useCallback(
    (id: number, delta: { unreviewed?: number; fp?: number; tp?: number; pending?: number; segments?: number }) => {
      setCountAdjust((current) => {
        const byTag = current.at === tagsQuery.dataUpdatedAt ? { ...current.byTag } : {};
        const previous = byTag[id] || { unreviewed: 0, fp: 0, tp: 0, pending: 0, segments: 0 };
        byTag[id] = {
          unreviewed: previous.unreviewed + (delta.unreviewed || 0),
          fp: previous.fp + (delta.fp || 0),
          tp: previous.tp + (delta.tp || 0),
          pending: previous.pending + (delta.pending || 0),
          segments: previous.segments + (delta.segments || 0),
        };
        return { at: tagsQuery.dataUpdatedAt, byTag };
      });
    },
    [tagsQuery.dataUpdatedAt],
  );

  // Pick a tag once the list arrives (first with unreviewed segments).
  useEffect(() => {
    if (!Array.isArray(tagsQuery.data) || tagsQuery.data.length === 0) return;
    if (tagId != null && tagsQuery.data.some((tag) => tag.tagId === tagId)) return;
    const first = tagsQuery.data.find((tag) => tag.unreviewedCount > 0) || tagsQuery.data[0];
    setTagId(first.tagId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tagsQuery.data, tagId]);
  const selectedTag = useMemo(() => tags.find((tag) => tag.tagId === tagId) || null, [tags, tagId]);
  const scopePending = useMemo(() => tags.reduce((sum, tag) => sum + (tag.pendingDeleteCount || 0), 0), [tags]);
  const criteriaDefinitions = useMemo(() => buildReviewCriteria(tags), [tags]);

  // Queue: Cove-style paging or Cove's infinite query, plus local overrides for just-changed entries.
  const queueScope = useMemo<QueueScope>(
    () => ({
      videoId,
      tagId,
      verdictFilter,
      exportStatus,
      objectFilter,
      filter: { q: libraryMode ? filter.q : undefined, sort: filter.sort, direction: filter.direction, seed: filter.seed },
    }),
    [videoId, tagId, verdictFilter, exportStatus, objectFilter, libraryMode, filter.q, filter.sort, filter.direction, filter.seed],
  );
  const queue = useQueue(queueScope, infinite, page, perPage);
  const scopeKey = JSON.stringify({ ...queueScope, page: infinite ? "inf" : page, perPage });
  const [overrides, setOverrides] = useState<Record<number, ReviewEntry>>({});
  const overridesScopeRef = useRef("");
  if (overridesScopeRef.current !== scopeKey) {
    overridesScopeRef.current = scopeKey;
    if (Object.keys(overrides).length > 0) setOverrides({});
  }
  const items = useMemo(() => queue.items.map((entry) => overrides[entry.id] || entry), [queue.items, overrides]);
  const totalCount = queue.totalCount;
  const itemsRef = useRef(items);
  itemsRef.current = items;

  const replaceEntry = useCallback((updated: ReviewEntry, previousId: number) => {
    setOverrides((current) => ({ ...current, [previousId]: updated }));
  }, []);

  // Selection (Cove's multi-select + selection bar).
  const multiSelect = useMultiSelect(items, { preserveOnItemsChange: true, resetKey: scopeKey });
  const selectedIds = multiSelect.selectedIds;
  const selecting = selectedIds.size > 0;
  const orderedIds = useMemo(() => items.map((entry) => entry.id), [items]);
  const toggleSelection = useCallback(
    (id: number, options?: any) => {
      multiSelect.toggle(id, { ...(options || {}), orderedIds });
    },
    [multiSelect, orderedIds],
  );

  // Keyboard focus.
  const [focusIndex, setFocusIndex] = useState(0);
  const focusRef = useRef(focusIndex);
  focusRef.current = focusIndex;
  // Only a keyboard move scrolls the list; a page of results arriving must not pull it back up.
  const scrollPendingRef = useRef(false);
  const [notice, setNotice] = useState("");
  const [actionError, setActionError] = useState("");
  const [applying, setApplying] = useState(false);
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const gridRef = useRef<HTMLDivElement | null>(null);
  const contextCardMinWidthPx = useListPageCardMinWidthPx();
  const cardMinWidthPx = contextCardMinWidthPx ?? getEntityCardMinWidthPx(CARD_SIZE_ENTITY_TYPE, zoomLevel ?? 1);

  useEffect(() => {
    setFocusIndex(0);
  }, [scopeKey]);

  // Once the first results are in, move DOM focus into the results so keys drive the board instead
  // of whatever was clicked last (the video page's tab strip eats ArrowLeft/ArrowRight).
  const hadItemsRef = useRef(false);
  useEffect(() => {
    if (items.length === 0) {
      hadItemsRef.current = false;
      return;
    }
    if (hadItemsRef.current) return;
    hadItemsRef.current = true;
    const grid = gridRef.current;
    if (!grid) return;
    const active = document.activeElement;
    if ((active && grid.contains(active)) || isEditableTarget(active)) return;
    try {
      grid.focus({ preventScroll: true });
    } catch {
      /* ignore */
    }
  }, [items.length]);

  useEffect(() => {
    if (!scrollPendingRef.current) return;
    const grid = gridRef.current;
    const entry = items[focusIndex];
    if (!grid || !entry) return;
    scrollPendingRef.current = false;
    const node = grid.querySelector(`[data-item-id="${String(entry.id)}"]`);
    if (node && typeof (node as HTMLElement).scrollIntoView === "function") {
      (node as HTMLElement).scrollIntoView({ block: "nearest", inline: "nearest" });
    }
  }, [focusIndex, items]);

  // Infinite mode: keep a page of headroom ahead of the keyboard focus.
  useEffect(() => {
    if (!infinite) return;
    if (items.length - focusIndex <= 12) queue.loadMore();
  }, [infinite, focusIndex, items.length, queue]);

  // Cove's grid lays cards out with auto-fill columns of the card size, so the column count follows
  // from the grid width rather than from measuring cards (which may be virtualized away).
  const columns = useCallback(() => {
    const grid = gridRef.current;
    if (!grid || displayMode !== "grid") return 1;
    const gap = 12;
    return Math.max(1, Math.floor((grid.clientWidth + gap) / (cardMinWidthPx + gap)));
  }, [displayMode, cardMinWidthPx]);

  const moveFocus = useCallback(
    (delta: number) => {
      const list = itemsRef.current;
      const current = focusRef.current;
      if (list.length === 0) return;
      scrollPendingRef.current = true;
      const next = current + delta;
      if (next >= list.length) {
        // Paged mode: step onto the next page instead of stopping at the edge.
        if (!infinite && page * perPage < totalCount) {
          setFilter({ ...filter, page: page + 1 });
          setFocusIndex(0);
        } else {
          setFocusIndex(list.length - 1);
        }
        return;
      }
      if (next < 0) {
        if (!infinite && page > 1) {
          setFilter({ ...filter, page: page - 1 });
          setFocusIndex(0);
        } else {
          setFocusIndex(0);
        }
        return;
      }
      setFocusIndex(next);
    },
    [infinite, page, perPage, totalCount, filter, setFilter],
  );

  const invalidateVideo = useCallback(
    (id: number) => {
      try {
        queryClient.invalidateQueries({ queryKey: ["video", id] });
      } catch {
        /* ignore */
      }
    },
    [queryClient],
  );

  const applyVerdictResult = useCallback(
    (previous: ReviewEntry, saved: QueueItem, verdict: string) => {
      const entry = toEntry(saved);
      const previousVerdict = previous.mark ? previous.mark.verdict : null;
      replaceEntry(entry, previous.id);
      const deletedNow = Boolean(entry.mark && entry.mark.segmentDeletedAt) && !(previous.mark && previous.mark.segmentDeletedAt);
      bumpTag(previous.segment.tagId ?? 0, {
        unreviewed: previousVerdict ? 0 : -1,
        fp: (verdict === VERDICT_FP ? 1 : 0) - (previousVerdict === VERDICT_FP ? 1 : 0),
        tp: (verdict === VERDICT_TP ? 1 : 0) - (previousVerdict === VERDICT_TP ? 1 : 0),
        pending:
          (verdict === VERDICT_FP && !(entry.mark && entry.mark.segmentDeletedAt) ? 1 : 0) -
          (previousVerdict === VERDICT_FP && !(previous.mark && previous.mark.segmentDeletedAt) ? 1 : 0),
        segments: deletedNow ? -1 : 0,
      });
      if (deletedNow) invalidateVideo(previous.segment.hostId);
    },
    [replaceEntry, bumpTag, invalidateVideo],
  );

  const markEntry = useCallback(
    async (entry: ReviewEntry | undefined, verdict: string) => {
      if (!entry || entry.id <= 0) return;
      const previous = entry.mark ? entry.mark.verdict : null;
      if (previous === verdict) return;
      replaceEntry({ ...entry, mark: { ...(entry.mark || ({} as ReviewMark)), verdict, exportStatus: "pending" } as ReviewMark }, entry.id);
      try {
        const saved = await api<QueueItem>("/marks", { method: "POST", body: JSON.stringify({ segmentId: entry.id, verdict }) });
        applyVerdictResult(entry, saved, verdict);
      } catch (markError: any) {
        replaceEntry(entry, entry.id);
        setActionError(markError.message || "Failed to save verdict.");
      }
    },
    [replaceEntry, applyVerdictResult],
  );

  const markCurrent = useCallback(
    (verdict: string) => {
      const entry = itemsRef.current[focusRef.current];
      if (!entry) return;
      moveFocus(1);
      void markEntry(entry, verdict);
    },
    [markEntry, moveFocus],
  );

  const clearEntry = useCallback(
    async (entry: ReviewEntry | undefined) => {
      if (!entry || !entry.mark) return;
      if (entry.mark.segmentDeletedAt) {
        try {
          const restored = await api<QueueItem>(`/marks/${entry.mark.id}/restore`, { method: "POST" });
          replaceEntry(toEntry(restored), entry.id);
          bumpTag(entry.segment.tagId ?? 0, { pending: 1, segments: 1 });
          invalidateVideo(entry.segment.hostId);
          setNotice("Segment restored in Cove. Press U again to clear the verdict.");
        } catch (restoreError: any) {
          setActionError(restoreError.message || "Failed to restore segment.");
        }
        return;
      }
      const previous = entry.mark.verdict;
      replaceEntry({ ...entry, mark: null }, entry.id);
      try {
        await api(`/marks/${entry.mark.id}`, { method: "DELETE" });
        bumpTag(entry.segment.tagId ?? 0, { unreviewed: 1, fp: previous === VERDICT_FP ? -1 : 0, tp: previous === VERDICT_TP ? -1 : 0, pending: previous === VERDICT_FP ? -1 : 0 });
      } catch (clearError: any) {
        replaceEntry(entry, entry.id);
        setActionError(clearError.message || "Failed to clear verdict.");
      }
    },
    [replaceEntry, bumpTag, invalidateVideo],
  );

  const unmarkCurrent = useCallback(() => void clearEntry(itemsRef.current[focusRef.current]), [clearEntry]);

  const openCurrent = useCallback(() => {
    const entry = itemsRef.current[focusRef.current];
    if (entry && typeof onNavigate === "function") onNavigate(entryRoute(entry));
  }, [onNavigate]);

  const markSelected = useCallback(
    async (verdict: string) => {
      const chosen = items.filter((entry) => selectedIds.has(entry.id) && entry.id > 0);
      if (chosen.length === 0) return;
      try {
        const saved = await api<QueueItem[]>("/marks/bulk", { method: "POST", body: JSON.stringify({ segmentIds: chosen.map((entry) => entry.id), verdict }) });
        const byId = new Map(saved.map((item) => [item.segment.id, item]));
        chosen.forEach((entry) => {
          const updated = byId.get(entry.id);
          if (updated) applyVerdictResult(entry, updated, verdict);
        });
        multiSelect.selectNone();
      } catch (bulkError: any) {
        setActionError(bulkError.message || "Failed to save verdicts.");
      }
    },
    [items, selectedIds, applyVerdictResult, multiSelect],
  );

  const clearSelected = useCallback(async () => {
    const chosen = items.filter((entry) => selectedIds.has(entry.id) && entry.mark && !entry.mark.segmentDeletedAt);
    if (chosen.length === 0) return;
    try {
      await api("/marks/bulk-clear", { method: "POST", body: JSON.stringify({ markIds: chosen.map((entry) => entry.mark!.id) }) });
      chosen.forEach((entry) => {
        replaceEntry({ ...entry, mark: null }, entry.id);
        bumpTag(entry.segment.tagId ?? 0, { unreviewed: 1, fp: entry.mark!.verdict === VERDICT_FP ? -1 : 0, tp: entry.mark!.verdict === VERDICT_TP ? -1 : 0, pending: entry.mark!.verdict === VERDICT_FP ? -1 : 0 });
      });
      multiSelect.selectNone();
    } catch (bulkError: any) {
      setActionError(bulkError.message || "Failed to clear verdicts.");
    }
  }, [items, selectedIds, replaceEntry, bumpTag, multiSelect]);

  const applyDeletes = useCallback(async () => {
    setApplying(true);
    setActionError("");
    try {
      const result = await api("/apply", { method: "POST", body: JSON.stringify({ videoId: libraryMode ? null : videoId, tagId: null }) });
      setNotice(`Deleted ${result.deletedSegments} segment${result.deletedSegments === 1 ? "" : "s"} across ${result.affectedVideos} video${result.affectedVideos === 1 ? "" : "s"}.`);
      if (libraryMode) {
        try {
          queryClient.invalidateQueries({ queryKey: ["video"] });
        } catch {
          /* ignore */
        }
      } else {
        invalidateVideo(videoId);
      }
      setOverrides({});
      await queryClient.invalidateQueries({ queryKey: ["ai-review"] });
      setConfirmDeleteOpen(false);
    } catch (applyError: any) {
      setActionError(applyError.message || "Failed to delete segments.");
    } finally {
      setApplying(false);
    }
  }, [libraryMode, videoId, queryClient, invalidateVideo]);

  // Keyboard driving through Cove's shortcut system (manifest-declared actions, rebindable) with a
  // fallback listener for hosts where those actions ended up unbound.
  const bindings = useExtensionKeyboardBindings(EXTENSION_ID) as Record<string, string[]> | null;
  const bindingsKey = JSON.stringify(bindings || {});
  const keyboardActions = useMemo(() => {
    const effective = bindings || {};
    const bind = (id: string) => (effective[id] && effective[id].length > 0 ? effective[id] : DEFAULT_BINDINGS[id]);
    return [
      { id: "review-false-positive", surface: "local" as const, bindings: bind("review-false-positive"), action: () => markCurrent(VERDICT_FP) },
      { id: "review-true-positive", surface: "local" as const, bindings: bind("review-true-positive"), action: () => markCurrent(VERDICT_TP) },
      { id: "review-clear", surface: "local" as const, bindings: bind("review-clear"), action: () => unmarkCurrent() },
      { id: "review-open", surface: "local" as const, bindings: bind("review-open"), action: () => openCurrent() },
      { id: "review-next", surface: "local" as const, bindings: bind("review-next"), action: () => moveFocus(1) },
      { id: "review-previous", surface: "local" as const, bindings: bind("review-previous"), action: () => moveFocus(-1) },
      { id: "review-down", surface: "local" as const, bindings: bind("review-down"), action: () => moveFocus(columns()) },
      { id: "review-up", surface: "local" as const, bindings: bind("review-up"), action: () => moveFocus(-columns()) },
    ];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markCurrent, unmarkCurrent, openCurrent, moveFocus, columns, bindingsKey]);
  useRegisterExtensionKeyboardActions(EXTENSION_ID, keyboardActions);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (isEditableTarget(event.target)) return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      let handled = true;
      switch (event.key) {
        case "ArrowRight":
        case "n":
        case "N":
          moveFocus(1);
          break;
        case "ArrowLeft":
        case "p":
        case "P":
          moveFocus(-1);
          break;
        case "ArrowDown":
          moveFocus(columns());
          break;
        case "ArrowUp":
          moveFocus(-columns());
          break;
        case "x":
        case "X":
        case "f":
        case "F":
          markCurrent(VERDICT_FP);
          break;
        case "t":
        case "T":
          markCurrent(VERDICT_TP);
          break;
        case "u":
        case "U":
          unmarkCurrent();
          break;
        case "Enter":
          openCurrent();
          break;
        default:
          handled = false;
      }
      if (handled) {
        event.preventDefault();
        event.stopPropagation();
      }
    };
    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [moveFocus, columns, markCurrent, unmarkCurrent, openCurrent]);

  useEffect(() => {
    if (!notice) return undefined;
    const timer = setTimeout(() => setNotice(""), 6000);
    return () => clearTimeout(timer);
  }, [notice]);

  const navigate = useCallback(
    (route: any) => {
      if (typeof onNavigate === "function") onNavigate(route);
    },
    [onNavigate],
  );

  // ── Pieces handed to Cove's list chrome ──

  const operations = (
    <div className="ai-review-shell flex flex-wrap items-center gap-2">
      <button
        type="button"
        className={scopePending > 0 ? BTN_DANGER : BTN}
        disabled={applying || scopePending === 0}
        onClick={() => setConfirmDeleteOpen(true)}
        title="Delete every segment marked as a false positive from Cove"
      >
        {applying ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
        {applying ? "Deleting…" : `Delete ${scopePending} FP`}
      </button>
      {libraryMode ? null : (
        <button type="button" className={BTN} onClick={() => navigate({ page: "ai-review" })} title="Review this tag across the whole library">
          <Library className="h-3.5 w-3.5" />
          Library
        </button>
      )}
    </div>
  );

  const byline = selectedTag ? (
    <span
      className="ai-review-shell whitespace-nowrap"
      title={`${selectedTag.unreviewedCount.toLocaleString()} unreviewed${selectedTag.pendingDeleteCount > 0 ? `, ${selectedTag.pendingDeleteCount} awaiting deletion` : ""}`}
    >
      <span className="text-red-400">{selectedTag.falsePositiveCount} FP</span>
      {" · "}
      <span className="text-emerald-400">{selectedTag.truePositiveCount} TP</span>
    </span>
  ) : null;

  const tagPicker = (
    <div className="w-72 shrink-0" title="Tag under review">
      <SelectInput
        value={tagId == null ? "" : String(tagId)}
        onChange={(value) => setTagId(value ? Number(value) : null)}
        options={
          tags.length === 0
            ? [{ value: "", label: tagsQuery.isPending ? "Loading tags…" : "No AI segments" }]
            : tags.map((tag) => ({ value: String(tag.tagId), label: `${tag.tagName} (${tag.unreviewedCount.toLocaleString()} unreviewed)` }))
        }
      />
    </div>
  );

  const selectionActions = (
    <span className="ai-review-shell contents">
      <button type="button" className={`${SELECTION_ACTION} text-red-400 hover:bg-red-900/20 hover:text-red-300`} onClick={() => void markSelected(VERDICT_FP)}>
        <X className="h-3 w-3" />
        False positive
      </button>
      <button type="button" className={`${SELECTION_ACTION} text-emerald-400 hover:bg-emerald-900/20 hover:text-emerald-300`} onClick={() => void markSelected(VERDICT_TP)}>
        <Check className="h-3 w-3" />
        True positive
      </button>
      <button type="button" className={`${SELECTION_ACTION} text-secondary hover:text-foreground`} onClick={() => void clearSelected()}>
        <RotateCcw className="h-3 w-3" />
        Clear verdict
      </button>
    </span>
  );

  const renderEntry = (entry: ReviewEntry, index: number) => {
    const shared: EntryProps = {
      entry,
      focused: index === focusIndex,
      selected: selectedIds.has(entry.id),
      selecting,
      libraryMode,
      onFocus: () => setFocusIndex(index),
      onToggle: (options?: any) => toggleSelection(entry.id, options),
      onNavigate: navigate,
    };
    return displayMode === "list" ? <ReviewRow key={entry.id} {...shared} /> : <ReviewCard key={entry.id} {...shared} />;
  };

  const emptyMessage =
    tagId == null
      ? tagsQuery.isPending
        ? "Loading tags…"
        : libraryMode
          ? "No AI tag segments found. Run AI tagging on some videos first."
          : "This video has no AI tag segments."
      : verdictFilter === VERDICT_UNREVIEWED
        ? "Nothing left to review for this tag. Pick another tag or change the verdict filter."
        : verdictFilter === VERDICT_DELETED
          ? "No deleted false positives for this tag."
          : "No segments match.";

  // Cove's grid owns infinite loading in grid mode (it renders the sentinel inside the virtualized
  // list); in list mode the surrounding page or toolbar provides it.
  const results =
    items.length === 0 && !queue.isPending ? (
      <p className="ai-review-shell rounded-md border border-dashed border-border bg-surface/40 p-4 text-sm text-muted">{emptyMessage}</p>
    ) : displayMode === "list" ? (
      <div ref={gridRef} tabIndex={-1} className="overflow-hidden rounded-xl border border-border bg-card outline-none">
        <RawSegmentListHeader />
        <div className="divide-y divide-border">{items.map(renderEntry)}</div>
      </div>
    ) : (
      <div ref={gridRef} tabIndex={-1} className="outline-none">
        <VirtualizedEntityGrid
          items={items}
          getItemKey={(entry) => entry.id}
          minCardWidth="var(--card-min-width, 225px)"
          estimateRowHeight={300}
          infinitePageSize={infinite}
          hasNextPage={queue.hasNextPage}
          isFetchingNextPage={queue.isFetchingNextPage}
          loadMore={queue.loadMore}
          renderItem={renderEntry}
        />
      </div>
    );

  const underToolbar = (
    <div className="flex flex-wrap items-center gap-3">
      {tagPicker}
      <div className="ai-review-shell flex flex-wrap items-center gap-x-4 gap-y-1">
        <KeyHints bindings={bindings} libraryMode={libraryMode} />
        {notice ? <span className="text-xs text-foreground">{notice}</span> : null}
        {actionError ? <span className="text-xs text-red-400">{actionError}</span> : null}
      </div>
    </div>
  );

  const confirmDelete = (
    <ConfirmDialog
      open={confirmDeleteOpen}
      title="Delete false positives"
      message={`Delete ${scopePending} false-positive segment${scopePending === 1 ? "" : "s"} from ${libraryMode ? "the whole library" : "this video"} in Cove? Frames stay exported and each deletion can be undone from the deleted view.`}
      confirmLabel="Delete"
      onConfirm={() => void applyDeletes()}
      onCancel={() => setConfirmDeleteOpen(false)}
      isPending={applying}
      errorMessage={actionError || null}
    />
  );

  const infiniteScroll = infinite
    ? { hasNextPage: queue.hasNextPage, isFetchingNextPage: queue.isFetchingNextPage, onLoadMore: queue.loadMore, loadedCount: items.length, totalCount }
    : undefined;

  if (libraryMode) {
    return (
      <>
        <ListPage
          title="AI Review"
          pageKey="ai-review"
          cardSizeEntityType={CARD_SIZE_ENTITY_TYPE}
          filter={filter}
          onFilterChange={setFilter}
          totalCount={totalCount}
          isLoading={queue.isPending && tagId != null}
          error={queue.error || null}
          onRetry={() => queue.refetch()}
          sortOptions={SORT_OPTIONS}
          disableRelevanceSort
          searchPlaceholder="Search video titles…"
          displayMode={displayMode}
          onDisplayModeChange={changeDisplayMode}
          availableDisplayModes={["grid", "list"]}
          allowInfinitePageSize
          showPagingControls={!infinite}
          infiniteScroll={infiniteScroll}
          criteriaDefinitions={criteriaDefinitions}
          objectFilter={objectFilter}
          onObjectFilterChange={setObjectFilter}
          selectedIds={selectedIds}
          onSelectAll={() => multiSelect.selectAll()}
          onSelectNone={() => multiSelect.selectNone()}
          onInvertSelection={() => multiSelect.invertSelection()}
          selectionActions={selectionActions}
          metadataByline={byline}
          renderOperations={() => operations}
          onNavigate={onNavigate}
          manageDocumentTitle
        >
          <div className="mb-3">{underToolbar}</div>
          {results}
        </ListPage>
        {confirmDelete}
      </>
    );
  }

  // Video tab: the embedded-list chrome Cove uses on detail pages.
  return (
    <div className="space-y-3">
      <DetailListToolbar
        filter={filter}
        onFilterChange={setFilter}
        totalCount={totalCount}
        sortOptions={SORT_OPTIONS}
        showSearch={false}
        cardSizeEntityType={CARD_SIZE_ENTITY_TYPE}
        zoomLevel={zoomLevel}
        onZoomChange={onZoomChange}
        displayMode={displayMode}
        onDisplayModeChange={changeDisplayMode}
        availableDisplayModes={["grid", "list"]}
        allowInfinitePageSize
        showPagingControls={!infinite}
        criteriaDefinitions={criteriaDefinitions}
        objectFilter={objectFilter}
        onObjectFilterChange={setObjectFilter}
        paginationAriaLabel="AI review pagination above results"
        selectedCount={selectedIds.size}
        onSelectAll={() => multiSelect.selectAll()}
        onSelectNone={() => multiSelect.selectNone()}
        selectionActions={selectionActions}
        metadataByline={byline}
      />
      <div className="flex flex-wrap items-center gap-3">
        {operations}
        {underToolbar}
      </div>
      <div className="list-page-content" style={{ "--card-min-width": `${cardMinWidthPx}px` } as React.CSSProperties}>
        {results}
        {infinite ? (
          displayMode === "list" ? (
            <InfiniteScrollSentinel
              hasMore={queue.hasNextPage}
              isLoading={queue.isFetchingNextPage}
              onLoadMore={queue.loadMore}
              loadedCount={items.length}
              totalCount={totalCount}
            />
          ) : null
        ) : (
          <DetailListPagination filter={filter} onFilterChange={setFilter} totalCount={totalCount} allowInfinitePageSize ariaLabel="AI review pagination below results" />
        )}
      </div>
      {confirmDelete}
    </div>
  );
}

// ─── Entry components ──────────────────────────────────────────────────────

function AiReviewVideoTab(props: { entityId: number | string; onNavigate?: (route: any) => void }) {
  const entityId = Number(props.entityId);
  const [zoomLevel, setZoomLevel] = useState(1);
  if (!Number.isFinite(entityId)) {
    return <p className="text-sm text-muted">No video selected.</p>;
  }
  return <ReviewBoard videoId={entityId} onNavigate={props.onNavigate} zoomLevel={zoomLevel} onZoomChange={setZoomLevel} />;
}

function AiReviewPage(props: { onNavigate?: (route: any) => void }) {
  return <ReviewBoard videoId={null} onNavigate={props.onNavigate} />;
}

// ─── Settings ──────────────────────────────────────────────────────────────

interface ReviewSettings {
  exportRootPath: string;
  thumbnailWidth: number;
  exportFullResolution: boolean;
  maxFramesPerSegment: number;
  exportTruePositives: boolean;
  deleteOnMark: boolean;
  removeTagFromVideoWhenEmpty: boolean;
  sourceKey: string;
}

function copySettings(loaded: any): ReviewSettings {
  const settings = (loaded && loaded.settings) || {};
  return {
    exportRootPath: settings.exportRootPath || "",
    thumbnailWidth: settings.thumbnailWidth || 480,
    exportFullResolution: settings.exportFullResolution !== false,
    maxFramesPerSegment: settings.maxFramesPerSegment || 3,
    exportTruePositives: settings.exportTruePositives !== false,
    deleteOnMark: Boolean(settings.deleteOnMark),
    removeTagFromVideoWhenEmpty: settings.removeTagFromVideoWhenEmpty !== false,
    sourceKey: settings.sourceKey || "ext:ai.tagging",
  };
}

function Help({ children }: { children: React.ReactNode }) {
  return <p className="mt-1 text-xs text-muted">{children}</p>;
}

function AiReviewSettingsPanel() {
  const [settings, setSettings] = useState<ReviewSettings>(copySettings(null));
  const [paths, setPaths] = useState({ effectiveExportRootPath: "", manifestPath: "" });
  const [exportStatus, setExportStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  const refreshStatus = useCallback(async () => {
    try {
      setExportStatus(await api("/export/status"));
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const loaded = await api("/settings");
        if (!cancelled) {
          setSettings(copySettings(loaded));
          setPaths({ effectiveExportRootPath: loaded.effectiveExportRootPath, manifestPath: loaded.manifestPath });
        }
      } catch (loadError: any) {
        if (!cancelled) setMessage(loadError.message || "Failed to load AI Review settings.");
      } finally {
        if (!cancelled) setLoading(false);
      }
      void refreshStatus();
    })();
    return () => {
      cancelled = true;
    };
  }, [refreshStatus]);

  function patch<K extends keyof ReviewSettings>(field: K, value: ReviewSettings[K]) {
    setSettings((current) => ({ ...current, [field]: value }));
  }

  async function save() {
    setSaving(true);
    setMessage("");
    try {
      const saved = await api("/settings", { method: "PUT", body: JSON.stringify(settings) });
      setSettings(copySettings(saved));
      setPaths({ effectiveExportRootPath: saved.effectiveExportRootPath, manifestPath: saved.manifestPath });
      setMessage("Settings saved.");
    } catch (saveError: any) {
      setMessage(saveError.message || "Failed to save AI Review settings.");
    } finally {
      setSaving(false);
    }
  }

  async function retryExports() {
    try {
      const result = await api("/export/retry", { method: "POST" });
      setMessage(`Queued ${result.queued} export${result.queued === 1 ? "" : "s"}.`);
      setTimeout(() => void refreshStatus(), 1500);
    } catch (retryError: any) {
      setMessage(retryError.message || "Failed to queue exports.");
    }
  }

  async function rewriteManifest() {
    try {
      const result = await api("/export/manifest", { method: "POST" });
      setMessage(`Manifest written to ${result.manifestPath}`);
    } catch (manifestError: any) {
      setMessage(manifestError.message || "Failed to write manifest.");
    }
  }

  if (loading) {
    return <p className="text-sm text-muted">Loading AI Review settings...</p>;
  }

  const checkbox = (field: keyof ReviewSettings, label: string) => (
    <label className="flex items-center gap-2 text-sm text-foreground">
      <input type="checkbox" checked={Boolean(settings[field])} onChange={(event) => patch(field, event.target.checked as any)} />
      {label}
    </label>
  );

  return (
    <div className="ai-review-shell max-w-2xl">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-foreground">Review and export</h3>
        <p className="text-sm text-muted">
          Open the AI Review tab of a video, or the AI Review page for the whole library, to flip through the segments of a tag. Verdicts are
          saved immediately and frames are exported in the background.
        </p>
      </div>
      <Field label="Export folder">
        <TextInput value={settings.exportRootPath} placeholder={paths.effectiveExportRootPath} onChange={(value) => patch("exportRootPath", value)} />
        <Help>
          Frames go to &lt;folder&gt;/&lt;tag&gt;/&lt;fp|tp&gt;/, with manifest.csv at the root. Currently: {paths.effectiveExportRootPath}
        </Help>
      </Field>
      <Field label="Frames per segment">
        <SelectInput
          value={String(settings.maxFramesPerSegment)}
          onChange={(value) => patch("maxFramesPerSegment", Number(value))}
          options={[1, 2, 3].map((n) => ({ value: String(n), label: n === 1 ? "1" : `up to ${n} (by segment length)` }))}
        />
        <Help>Frames are taken at the times the AI model actually sampled, so a false positive is the frame the model got wrong.</Help>
      </Field>
      <Field label="Exported frame width">
        <NumberInput value={settings.thumbnailWidth} min={160} max={1920} onChange={(value) => patch("thumbnailWidth", value ?? 480)} />
        <Help>Used when Export full resolution is off.</Help>
      </Field>
      <Field label="Export">
        <div className="space-y-2">
          {checkbox("exportFullResolution", "Export full resolution")}
          {checkbox("exportTruePositives", "Export true positives too, so a training pass has both classes")}
        </div>
      </Field>
      <Field label="Deleting">
        <div className="space-y-2">
          {checkbox("deleteOnMark", "Delete a false-positive segment from Cove the moment it is marked")}
          {checkbox("removeTagFromVideoWhenEmpty", "Remove the tag from the video when its last AI segment goes")}
        </div>
        <Help>With delete-on-mark off, segments are deleted when you press the Delete button on the review board.</Help>
      </Field>
      <Field label="Segment source">
        <TextInput value={settings.sourceKey} onChange={(value) => patch("sourceKey", value)} />
        <Help>Which segments are reviewed. AI Tagging writes ext:ai.tagging.</Help>
      </Field>
      <div className="flex items-center justify-end gap-3">
        {message ? <span className="text-sm text-muted">{message}</span> : null}
        <SaveButton loading={saving} onClick={() => void save()} />
      </div>
      <div className="mt-4 rounded-xl border border-border bg-surface p-3 text-sm">
        <div className="mb-2 font-medium text-foreground">Frame export</div>
        {exportStatus ? (
          <p className="text-muted">
            {exportStatus.done} exported · {exportStatus.pending} pending · {exportStatus.failed} failed · {exportStatus.skipped} skipped
          </p>
        ) : (
          <p className="text-muted">Status unavailable.</p>
        )}
        <p className="text-xs text-muted">Manifest: {paths.manifestPath}</p>
        <div className="mt-2 flex flex-wrap gap-2">
          <button className={BTN} type="button" onClick={() => void retryExports()}>
            <Play className="h-3.5 w-3.5" />
            Retry pending and failed
          </button>
          <button className={BTN} type="button" onClick={() => void rewriteManifest()}>
            Rewrite manifest
          </button>
          <button className={BTN} type="button" onClick={() => void refreshStatus()}>
            Refresh
          </button>
        </div>
      </div>
    </div>
  );
}

export default {
  components: {
    AiReviewVideoTab,
    AiReviewPage,
    AiReviewSettingsPanel,
  },
};
