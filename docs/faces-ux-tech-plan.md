# Faces UX Tech Plan — Closing the Gap vs `stash-ai-server`

> **Purpose.** A self-contained, agent-implementable plan that closes the known UX and storage gaps in the new Cove faces stack vs. the legacy `stash-ai-server` / `skier_aitagging` implementation. This now includes the 13 user-reported items, the detection-retention correction, and additional workflow gaps found during a second review.
>
> **Audience.** A focused implementation agent. Read §0 first, then pick items in the suggested order from §15.

---

## 0. Repo map (current → reference)

Current stack ("Cove" + AI extensions):

- Cove core (server) — `C:\Users\tyler\source\repos\cove`
  - `src\Cove.Core\Entities\Face.cs` — `Face` entity (identity bucket).
  - `src\Cove.Core\Entities\Segment.cs` — `Segment` and `Detection` entities. `Segment` remains the temporal range record for scene/audio timelines; `Detection` must be retained sparingly for spatial/keyframe/debug data, not as one row per sampled frame forever.
  - `src\Cove.Core\Entities\FaceAppearance.cs` — new canonical face appearance record proposed in this plan (`FaceId + host` with optional temporal fields for scenes, null temporal fields for images).
  - `src\Cove.Api\Controllers\FacesController.cs` — REST surface for `Face` (`GET /api/faces`, `GET /api/faces/{id}`, `GET /api/faces/{id}/detections`, `GET /api/faces/{id}/suggestions`, `POST /api/faces/{id}/link`, `merge-into`, `ignore`, `similar`, etc.).
  - `src\Cove.Api\Controllers\EntityImageUrls.cs` — image URL helpers (face cover URL builder).
- Cove core (UI, React/TS) — `C:\Users\tyler\source\repos\cove\ui`
  - `src\pages\FacesPage.tsx` — list page (currently shows a `merged` tri-state filter).
  - `src\pages\FaceDetailPage.tsx` — detail page (cover hero + suggestions + detections grid).
  - `src\components\FaceSuggestionsPanel.tsx` — suggestion cards with confidence/evidence + accept/reject.
- AI extensions — `C:\Users\tyler\source\repos\AI.Extensions`
  - `extensions\AI.Faces\AiFacesExtension.cs` — extension entry, endpoints (`/api/ext/ai-faces/*`), DI.
  - `extensions\AI.Faces\AiFacesPersistenceService.cs` — currently deletes/reinserts all source detections for a host and writes every prepared detection sample; this is the main storage issue to fix.
  - `extensions\AI.Faces\AiFacePreparationService.cs` — already builds IoU-based tracks and emits one `Segment` per track; do not reimplement this from scratch, audit and tune it.
  - `extensions\AI.Faces\AiFaceSuggester.cs` — `IFaceSuggester` impl (KNN over reference pack vectors).
  - `extensions\AI.Faces\AiFaceCoverGenerator.cs` — face cover thumb production.
  - `extensions\AI.Faces\AiFaceReferencePackStore.cs` / `SaieArchiveReader.cs` — `.saie` reference pack import.
  - `extensions\AI.Faces\AiFaceReferencePerformerResolver.cs` — "Import as performer" path.
- Cove SDK contracts (consumed here) — `C:\Users\tyler\source\repos\cove\src\Cove.Sdk\*` (currently `IFaceSuggester`, `IFaceLifecycleParticipant`, and suggestion-decision contracts; link/unlink and merge participant hooks are planned additions in this document).

Reference / legacy stack ("stash-ai-server" + `skier_aitagging`):

- `C:\Coding\Testing\Stash-PornServer\plugins\Stash-AIServer\backend\plugins\skier_aitagging\face_processor.py` — track building, exemplar selection, auto-apply propagation.
- `C:\Coding\Testing\Stash-PornServer\plugins\Stash-AIServer\backend\plugins\skier_aitagging\face_api.py` — cluster API, link/unlink with propagation + cascade un-assign.
- `C:\Coding\Testing\Stash-PornServer\plugins\Stash-AIServer\frontend\src\FacesHub.tsx` — list page with batch ops + suggested-match preview.
- `C:\Coding\Testing\Stash-PornServer\plugins\Stash-AIServer\frontend\src\FaceReviewPanel.tsx` — side-by-side review pane.
- `C:\Coding\Testing\Stash-PornServer\plugins\Stash-AIServer\frontend\src\PerformerFacesPanel.tsx`, `SceneFacesPanel.tsx`, `ImageFacesPanel.tsx` — per-entity face affordances.

Architectural ground rules (from `docs\cove_extensions_design.md` v5):

1. `Face` in core is intentionally minimal (identity bucket only). Algorithm-specific state stays in extension-private tables (`IDataExtension`).
2. `Segment` owns temporal ranges; `Detection` owns optional spatial samples. `Segment` should not be overloaded to represent image appearances. **The UI should drive face presence from a new canonical `FaceAppearance` record, with `Segment` remaining the scene timeline detail and `Detection` remaining sparse spatial evidence.**
3. Current Cove exposes `IFaceLifecycleParticipant`, `IFaceSuggester`, and suggestion-decision plumbing. A link/unlink participant hook does **not** appear to exist yet and must be added before propagation/relink cleanup can be implemented cleanly.
4. AI.* extensions only depend on AI.Core + Cove SDK; they do not call each other.

---

## 0.1 Key correction: detection rows are not the canonical face record

The first version of this plan was too generous about keeping per-frame `Detection` rows. The user's storage concern is correct: for videos, a sampled face detection produces timestamp + 4 coordinate values + JSON metadata for every frame. At `frame_interval=2s`, a one-hour video can produce 1,800 samples per visible face track. Across a library this becomes noisy and expensive, especially because the UI should not show detection thumbnails and most workflows only need the temporal appearance span plus a handful of spatial keyframes.

Current reality:

- `AiFacePreparationService.BuildVideoTracks` already builds IoU tracks with `MaxGapFrames`, `IoUMatchThreshold`, and `OpenTrack`; `EmitSegment` already emits one face `Segment` per track.
- `EmitDetections` still emits **every sample in every track** into `batch.Detections`.
- `AiFacesPersistenceService.PersistDetections` persists every prepared sample as a `Detection` row.
- `RefreshFaceStatsAsync` derives `DetectionCount`, `SceneCount`, and `ImageCount` from `Detections`, so raw sample volume leaks into user-facing counts.
- `FacesController.GetDetections` and `FaceDetailPage.tsx` make those rows the user's primary evidence, which is exactly the wrong abstraction.

Target model:

1. Add a new core `FaceAppearance` row as the canonical answer to "this face appears in this host." It unifies scenes and images without pretending images are temporal segments:
  ```jsonc
  {
    "faceId": 7,
    "hostType": "scene" | "image",
    "hostId": 42,
    "firstSeenAtSec": 778.0,
    "lastSeenAtSec": 784.0,
    "sampleCount": 108,
    "retainedSpatialSampleCount": 3,
    "topConfidence": 0.83,
    "representativeDetectionId": 99123,
    "representativeFrameSec": 780.0
  }
  ```
  For images, `firstSeenAtSec` / `lastSeenAtSec` are `null` and `sampleCount` is normally `1`.
2. `Segment` remains the temporal refinement for scene appearances only. It should carry enough compact payload to support timeline/X-Ray display:
  ```jsonc
  {
    "trackKey": "video-face-7",
    "sampleCount": 108,
    "frameIntervalSec": 2.0,
    "bestScore": 0.83,
    "bestTimeSec": 780.0,
    "bestBbox": [0.42, 0.18, 0.10, 0.14],
    "keyframes": [
     { "t": 778.0, "bbox": [0.41, 0.18, 0.10, 0.14], "score": 0.79 },
     { "t": 780.0, "bbox": [0.42, 0.18, 0.10, 0.14], "score": 0.83 },
     { "t": 784.0, "bbox": [0.43, 0.19, 0.10, 0.14], "score": 0.76 }
    ]
  }
  ```
3. Persist **keyframe detections only** by default, not every sampled frame. Use `Detection.GroupKey = trackKey` and `Detection.Extra.role = first|best|last|motion-keyframe`. For a normal track this should be 1-5 rows, not 108.
4. Add a configurable retention policy for face detections:
  - `None` — no `Detection` rows for faces; only `Segment.Payload.keyframes` and exemplars are kept. Good for users who only care about matching and performer propagation.
  - `Keyframes` — default; persist first/best/last and any significant bbox-motion keyframes.
  - `AllSamples` — debug / research mode only; persists the current full per-frame samples.
5. Face list/detail counters should be renamed conceptually:
  - `AppearanceCount` = number of `FaceAppearance` rows.
  - `FrameSampleCount` = sum of `sampleCount` across `FaceAppearance` rows (optional, mostly debug).
  - `SceneCount` / `ImageCount` = distinct hosts from `FaceAppearance`.
  - Avoid displaying "DetectionCount" as the primary user-facing number.
6. The X-Ray overlay should consume scene-segment keyframes and interpolate/nearest-neighbor bboxes when needed. It should not require retaining every sampled detection row just to draw a box.
7. `/api/faces/{id}/detections` should become an advanced/debug endpoint (or be replaced by `/api/faces/{id}/spatial-samples`) and should clearly return retained samples, not all original model outputs. The normal UX should use `/api/faces/{id}/appearances` backed by `FaceAppearance`.

Acceptance criteria for the retention correction:

- Running the same scene that currently produces 108 face detection rows produces 1-2 `FaceAppearance` rows, 1-2 face `Segment` rows, and no more than 5 retained `Detection` rows per track under the default `Keyframes` policy.
- Face detail and face list do not require `GET /api/faces/{id}/detections`.
- `DetectionCount` is no longer the primary user-visible count; appearances and distinct scenes/images are.
- A config toggle can temporarily restore full per-sample retention for debugging without changing UI behavior.

## 0.2 Why images use `FaceAppearance`, not `Segment`

Images are not time-based media, so treating an image as a pseudo-segment (`StartSec=0`, `EndSec=null`) would make the schema lie even if it technically works. The clean split is:

- `FaceAppearance` = canonical "face X appears in host Y" row for both scenes and images.
- `Segment` = temporal refinement for scene appearances only.
- `Detection` = optional sparse spatial samples for overlays/debug.

This avoids special cases in the UI while keeping each entity semantically honest. Every face-driven screen can query `FaceAppearance`; timeline/X-Ray and scene seeking can additionally use the linked scene segment data.

---

## 1. Stop showing "detection" thumbnails for scenes / images (replace with scene-or-image thumbnails)

**Problem.** Today `FaceDetailPage` and the per-scene/per-image evidence grids render one tile per `Detection` row. For a 17-second scene that produced 108 detections this is wasteful, slow, and visually noisy. We do not have (and don't want to extract) a unique thumbnail per detection — they would all be slight variants of the same scene frame. After §0.1, there also should not be 108 persisted detection rows to render.

**Prior reference.** `skier_aitagging` rendered evidence as **per-entity cards** (one per scene / image where the cluster appeared) using the existing scene/image cover thumbnail — see `frontend\src\FacesHub.tsx` ("clusters" and the per-cluster `entity_pairs` rendering) and `face_api.py:get_cluster_entity_pairs` which returns `(entity_type, entity_id)` pairs, not per-detection rows.

**Target files.**
- Backend: `src\Cove.Api\Controllers\FacesController.cs` — add a new endpoint or expand an existing one (see §11 below for the shared shape).
- Frontend: `ui\src\pages\FaceDetailPage.tsx`, plus `FaceSuggestionsPanel.tsx` evidence rendering.

**Approach.**
1. Add `GET /api/faces/{id}/appearances` returning the canonical `FaceAppearance` list:
   ```jsonc
   {
     "items": [
       {
        "appearanceId": 17,
         "hostType": "scene" | "image",
         "hostId": 42,
         "title": "...",
         "thumbnailUrl": "/api/scenes/42/cover?max=320&v=...",
         "frameSampleCount": 108,
         "retainedSpatialSampleCount": 3,
        "segmentCount": 1,
         "firstSeenAtSec": 778.2,
         "lastSeenAtSec": 794.1,
         "topConfidence": 0.83
       },
       ...
     ],
     "totalScenes": N, "totalImages": M
   }
   ```
  Implementation: read directly from `FaceAppearance` rows and join scene-only `Segment` rows for range metadata / counts. Use the standard scene/image cover URL builders (see `EntityImageUrls.cs`). No new thumbnails are produced.
2. The existing `GET /api/faces/{id}/detections` endpoint stays only as an advanced/debug retained-spatial-sample endpoint. The detail page no longer calls it by default.
3. `FaceDetailPage.tsx` "Detections" section becomes "Appears in" with `hostType`-aware cards (scene poster, scene title, count badge "108 analyzed frames" only when useful). Image card click → image page; scene card click → scene page (and later, with #11 timestamp-jump deep-link).

**Acceptance criteria.**
- Detail page makes one `appearances` call instead of one detection call returning 108 rows.
- No detection-bbox crops are rendered as tiles.
- Clicking a card navigates to the scene/image page.
- Empty face cluster shows "No appearances yet."

---

## 2. Group per-frame face detections into temporal segments (one "appearance," not 108)

**Problem.** The current preparation code already builds tracks and emits one `Segment` per track, but persistence still stores one `Detection` per face per analyzed frame, and the UI/API still treat those raw samples as the primary evidence. We need the track/appearance abstraction to be the storage and UX center: contiguous appearances of the same face on screen should be **one** scene `FaceAppearance` + one linked `Segment`, with optional gap merging (e.g. `frame_interval=2s` → merge gaps ≤ ~6s), plus a small set of retained keyframe spatial samples.

**Prior reference.**
- Track building (the canonical algorithm): `skier_aitagging\face_processor.py`
  - `class _OpenTrack` — a mutable per-track accumulator (start_s, end_s, last_frame_idx, keyframes, embeddings).
  - `def build_tracks(frames, frame_interval, *, iou_threshold, keyframe_iou_threshold, max_gap_frames) -> list[TrackCandidate]` (≈ line 176). Greedy IoU matching across consecutive frames; closes tracks once the gap exceeds `max_gap_frames`.
  - `DEFAULT_MAX_TRACK_GAP_FRAMES`, `DEFAULT_IOU_THRESHOLD` constants near top of the file.
- Per-track exemplar selection: `select_representative_embeddings` (≈ line 350) — quality + temporal-spread combination, used both for embeddings and for face cover candidates.

**Target files.**
- `src\Cove.Core\Entities\FaceAppearance.cs` — new canonical appearance entity.
- `extensions\AI.Faces\AiFacePreparationService.cs` — audit/tune existing `BuildVideoTracks`, `OpenTrack`, `EmitSegment`, and `EmitDetections`; extend track output with keyframes and sample counts.
- `extensions\AI.Faces\AiFacesPersistenceService.cs` — persist **one** `FaceAppearance` per (face, host), plus **one** scene `Segment` per contiguous appearance for scene hosts, with `Payload` carrying keyframes/sample counts; persist retained keyframe detections only under the retention policy.
- `src\Cove.Api\Controllers\FacesController.cs` — counts and "appears in" data should be derived from `FaceAppearance` primarily, with detection counts as a secondary/debug number.
- `src\Cove.Core\Entities\Face.cs` — denormalized counters need a rethink: add `AppearanceCount`; keep `DetectionCount` only as retained spatial-sample count or migrate it to `FrameSampleCount` if we still want the raw analyzer sample total.

**Approach.**
1. Keep the existing C# IoU tracker but make the constants configurable via `IConfiguration` under `AI:Faces:Tracking:*`:
  - `IoUMatchThreshold = 0.5` initially (current code); evaluate `0.3` if tracks split too often.
  - `KeyframeIoUThreshold = 0.5` (new; append keyframe when bbox motion is meaningful).
  - `MaxGapFrames = 3` (current code; at `frame_interval=2s` this tolerates up to ~6s of occlusion).
2. Extend `PreparedFaceTrack` to carry `SampleCount`, `FrameIntervalSec`, `BestSample`, and `Keyframes` (`t`, `bbox`, `score`, and optional embedding quality).
3. For scenes, write one `FaceAppearance` per contiguous face track plus one linked `Segment` carrying temporal detail. For images (single frame), write one image `FaceAppearance` and one retained spatial sample. Images do not get pseudo-segments.
4. Persistence: write one scene `Segment` per track (linked to the cluster face via `RefKind="face"`, `RefId=face.Id`) with keyframe/sample-count payload. Persist retained `Detection` rows according to §0.1 (`None`, `Keyframes`, `AllSamples`), defaulting to `Keyframes`.
5. Recompute `Face.AppearanceCount`, `SceneCount`, and `ImageCount` from `FaceAppearance`. Stop deriving user-visible counts only from `Detections`.

**Acceptance criteria.**
- A scene with face A on-screen continuously from `12:58.0` to `13:04.0` (frame interval 2s, frames at 12:58, 13:00, 13:02, 13:04) produces **1** `FaceAppearance` and **1** `Segment` (`StartSec=778`, `EndSec=784`), not 4 detection-owned rows.
- A scene where face A appears, leaves for 4s (one missed sample), then reappears for 6s produces 1 segment (within `max_gap_frames` tolerance) — covered by a unit test fixture.
- A scene where face A is gone for 30s before reappearing produces 2 segments — covered by a unit test fixture.
- `Face.AppearanceCount`, `SceneCount`, and `ImageCount` are populated and surfaced in `FaceDto`.
- Under default retention, the same track persists only a handful of `Detection` keyframes, not every sample.

---

## 3. Face cover hero is too large on the detail page

**Problem.** `FaceDetailPage.tsx` renders the face cover at near-screen-height; the user has to scroll past it to reach suggestions, controls, and appearances.

**Target files.** `ui\src\pages\FaceDetailPage.tsx` (and any shared CSS).

**Approach.** Convert the page to a two-column layout above the fold:
- Left column (~280–320 px fixed on desktop, full-width on mobile): cover thumbnail (capped at e.g. `max-h-[320px]`), label, primary metadata, controls (Link / Merge / Ignore / Delete).
- Right column: Suggested Matches (top 3 cards visible without scrolling), then "Appears in" (#1) below.
- On mobile the cover collapses to a `~160px` square left of label/controls.

Use existing layout primitives in `ui\src\components\` (e.g. the same patterns as the Performer detail page if available). No new components beyond layout glue.

**Acceptance criteria.**
- Above-the-fold on a 1080p browser shows: cover, label, controls, top-3 suggestion cards.
- Cover never exceeds 320px tall on desktop.
- Existing functionality (cover image fallback, edit label) is preserved.

---

## 4. Suggested-match images are too small; clicking them must navigate

**Problem.** Suggestion cards in `FaceSuggestionsPanel.tsx` use thumbnails that are too small to evaluate visually, and clicking them does nothing useful (no link to local performer page, no link to remote provider).

**Target files.** `ui\src\components\FaceSuggestionsPanel.tsx`; `src\Cove.Api\Controllers\FacesController.cs` (`GET /api/faces/{id}/suggestions` DTO).

**Approach.**
1. Bump suggestion thumbnail to ~160 px square on desktop (consistent with Performer cover sizing elsewhere).
2. Extend `FaceSuggestionDto` to always carry:
   - `localPerformerId: int | null` — set when the suggestion is already a Cove performer.
   - `externalUrl: string | null` — set when the suggestion comes from a reference pack (`.saie`) and the pack metadata includes a source URL (StashDB / TPDB / etc.). The reference pack already carries the source endpoint (`AiFaceReferencePackStore.GetStatusAsync().SourceEndpoint`); ensure per-identity `externalUrl` is plumbed through `AiFaceReferencePackStore` → `AiFaceSuggester` → `FaceSuggestionDto`.
3. Frontend wraps the thumbnail (and the name) in:
   - An `<a href="/performers/:id">` when `localPerformerId` is set.
   - An `<a href={externalUrl} target="_blank" rel="noopener">` when only `externalUrl` is set.
   - A non-link `<div>` when neither is set (with hint text "No linked performer page").
4. Use a real `<a>` (not `onClick`) so middle-click / Ctrl-click open in new tab natively, satisfying the "middle click = new tab" requirement.

**Acceptance criteria.**
- Suggestion thumbnails are visually large enough to evaluate (≥ 160 px square on desktop).
- Left-click on a suggestion that's a local performer routes to `/performers/:id`.
- Middle-click opens that route in a new tab.
- Suggestions that are reference-only with `externalUrl` open the remote URL in a new tab.

---

## 5. Side-by-side comparison: face-in-question vs suggested match

**Problem.** Users cannot easily judge whether a suggested match is correct without seeing the cluster cover and the candidate side by side.

**Prior reference.** `frontend\src\FaceReviewPanel.tsx` in stash-ai-server renders the unlinked cluster cover next to candidate exemplars in a comparison strip with accept/reject.

**Target files.** New component `ui\src\components\FaceCompareDialog.tsx`; integrated into `FaceSuggestionsPanel.tsx` and from the list page (#6).

**Approach.**
1. New dialog/pane that takes `{ faceId, suggestionId }` and renders two columns:
   - Left: face cover + (if implemented, see #7) up to 5 exemplars from the cluster.
   - Right: suggestion thumbnail + (if `localPerformerId`) the performer's existing reference images / cover.
2. Buttons: **Confirm link** (calls `POST /api/faces/{id}/link` with the suggestion's performerId), **Reject** (calls `/suggestions/decision` with `decision="reject"`), **Open performer** / **Open external**.
3. Keep this purely client-side composition over existing endpoints — no new backend except whatever's needed for #7 (exemplars).

**Acceptance criteria.**
- "Compare" button on each suggestion card opens the dialog with cover + suggestion side by side.
- Accept link works without a page refresh; the face moves to "Linked" state and suggestions list updates.
- Reject persists and removes the suggestion from the list.

---

## 6. Show top suggestion on the faces list page with a "Link" affordance

**Problem.** `FacesPage.tsx` shows the cover, label, and counts but the user must drill into each face to see its top suggestion. The user wants the top suggestion visible on each card with a one-click way to confirm.

**Prior reference.** `FacesHub.tsx` in stash-ai-server shows top candidate name + score on each cluster card and offers an inline "link" action.

**Target files.**
- Backend: `src\Cove.Api\Controllers\FacesController.cs` (`List` endpoint and `FaceDto`).
- Frontend: `ui\src\pages\FacesPage.tsx`, plus the new `FaceCompareDialog` from #5.

**Approach.**
1. `GET /api/faces` returns each face with an optional `topSuggestion`:
   ```jsonc
   { "id": 7, "label": null, "performerId": null, "coverImageUrl": "...",
    "appearanceCount": 12, "frameSampleCount": 108,
     "topSuggestion": {
       "suggestionId": "...", "displayName": "Halmia",
       "thumbnailUrl": "...", "confidence": 0.75,
       "localPerformerId": 580, "externalUrl": null
     } }
   ```
  Implementation: in the list query, for each unlinked face, run the same suggester as `GetSuggestions` (with `take=1`). Cache with an **extension-private** top-suggestion cache table rather than adding suggestion JSON to the core `Face` entity; suggestions are provider/algorithm-specific and should not leak into core schema.
2. List card grows a small footer: thumbnail (small) + name + confidence + **Link** button. Clicking **Link** opens the `FaceCompareDialog` (from #5). Cards for already-linked faces just show the linked performer link instead.

**Acceptance criteria.**
- Faces list shows the top suggestion for each unlinked cluster.
- Clicking Link opens the side-by-side dialog scoped to that face + that suggestion.
- Performance: list page render time stays within 1.5× of current on a 100-face fixture (verify with the cache).

---

## 7. Up to 5 cluster exemplars surfaced on the detail page

**Problem.** Today the cover is a single image. If it's a poor angle the user has no other reference to evaluate suggestions. The legacy stack tracked exemplars per cluster; we should too.

**Prior reference.**
- Selection algorithm: `face_processor.py::select_representative_embeddings` (≈ line 350): sort by `score * norm`, greedy dedup at cosine ≥ 0.85, then split surviving set into top-half-by-quality + remaining evenly spaced over the temporal span. Cap at `DEFAULT_MAX_EMBEDDINGS_PER_TRACK`.
- Storage: `face_processor.py` stores per-cluster representative embeddings; `face_api.py::_set_performer_image_from_cluster` (≈ line 217) reuses the same exemplar pool to set a performer cover from the cluster.

**Target files.**
- New extension-private table `ext_ai_faces_exemplars` (rows: `Id, FaceId, BlobId, FrameSec, Score, Norm, CreatedAt`) via `IDataExtension` migration in `AI.Faces`.
- `extensions\AI.Faces\AiFacesPersistenceService.cs` — when persisting tracks (#2), also persist up to `MaxExemplarsPerFace` (default 5) cropped face-region thumbnails as exemplar blobs.
- `extensions\AI.Faces\AiFaceCoverGenerator.cs` — likely shares the crop logic; refactor to produce both the cover blob and exemplar blobs from the same crop pipeline.
- New endpoint `GET /api/ext/ai-faces/faces/{id}/exemplars` → `[ { "blobUrl": "...", "frameSec": ..., "sourceHostType": "scene", "sourceHostId": 10 } ]`.
- Frontend: `FaceDetailPage.tsx` adds an "Exemplars" strip under the cover (small horizontal scroller, max 5). The compare dialog (#5) also pulls exemplars from this endpoint.

**Approach.**
1. Port the selection algorithm from `select_representative_embeddings` to C# in a new `extensions\AI.Faces\AiFaceExemplarSelector.cs`. Use the embedding `score * norm` quality metric already available on prepared frame embeddings.
2. On persist: after assigning a track to a face, run the selector over (existing exemplars for that face) ∪ (this run's track embeddings), keep the top `MaxExemplars=5` (configurable). Materialize crops into blob storage with a stable hash to avoid duplicate blobs across re-runs.
3. Cover generation continues as today (best single exemplar). Expose exemplars list via the new endpoint.

**Acceptance criteria.**
- A face cluster with ≥ 5 high-quality embeddings produces 5 exemplar blobs.
- Detail page shows ≤ 5 exemplar thumbs in a strip; click opens fullsize.
- Re-running detection on the same scene does not multiply exemplar blobs (idempotent on hash).

---

## 8. "Import as performer" sometimes silently no-ops (after unlink/retry)

**Problem.** After unlinking a performer and trying again, `POST /api/ext/ai-faces/reference/faces/{faceId}/import-performer` does nothing. Reading the current handler (`AiFacesExtension.cs`, the `import-performer` route): it returns `400` if `face.PerformerId.HasValue`. After an unlink the `Face.PerformerId` is cleared, so the early-out shouldn't fire — but the handler also depends on `AiFaceReferencePerformerResolver.FindOrCreateAsync`. Two likely failure modes worth verifying as part of the fix:
1. The reference suggestion decision store still has a `Reject` decision recorded for `(faceId, identity.ExternalId)` from a prior cycle, and the resolver / suggester is short-circuiting on it.
2. The resolver finds an existing performer that was created by the prior import and does not re-link the face to it — but **also doesn't return an error**, so the response is `204` and the caller thinks it worked.

**Target files.** `extensions\AI.Faces\AiFacesExtension.cs` (the `/reference/faces/{faceId}/import-performer` handler), `AiFaceReferencePerformerResolver.cs`, `AiFaceReferenceSuggestionDecisionStore.cs`.

**Approach.**
1. Add structured logging at the handler entry: faceId, suggestionId, currentPerformerId, packStatus.
2. On unlink (`POST /api/faces/{id}/link` with `performerId=null`): also clear stale import/reject state for that face from `AiFaceReferenceSuggestionDecisionStore`. Add a new core/SDK `IFaceLinkParticipant` hook (it does not appear to exist yet) and have `AI.Faces` observe unlink/relink to purge stale decision rows.
3. Have the resolver explicitly return a domain result (`{ Performer, WasCreated, WasReused }`) and have the handler always update `face.PerformerId` and `face.PrimarySourceKey` on success and return `200` with the updated `FaceDto`.
4. Add an integration test: link → unlink → re-link via reference must succeed and result in `Face.PerformerId == previousId`.

**Acceptance criteria.**
- After unlink + retry import, `Face.PerformerId` is set and the UI shows the linked performer.
- Test case in `tests\AI.Extensions.Tests\` covers the link → unlink → re-link sequence.

---

## 9. Replace "Merged + primary" filter with "Linked / Unlinked"

**Problem.** The current tri-state on `FacesPage.tsx` exposes `merged` semantics (relevant for soft-merge dedup) but the user's day-to-day filter is "show me faces that still need a performer linked vs ones already done."

**Target files.**
- Backend: `src\Cove.Api\Controllers\FacesController.cs` `List` query.
- Frontend: `ui\src\pages\FacesPage.tsx`.

**Approach.**
1. Add `linked: bool? = null` to the `List` query parameters: `null` = all, `true` = `PerformerId IS NOT NULL`, `false` = `PerformerId IS NULL`.
2. Replace the existing `merged` `<select>` with a `linked` tri-state: "All / Linked / Unlinked" (default Unlinked, since that's the actionable state).
3. Move the `merged` filter into an "Advanced" disclosure (or drop it — soft-merged faces have `MergedIntoFaceId IS NOT NULL` and are usually hidden from the default view anyway; confirm current default-list behavior in the controller and keep parity).
4. URL state in `useListUrlState`: rename `merged` query key to `linked` (with a one-release back-compat alias if anyone bookmarked the old form).

**Acceptance criteria.**
- Default list view shows unlinked faces.
- Toggle to "Linked" shows linked-to-performer faces.
- Toggle to "All" matches existing behavior (minus merged-state filtering).

---

## 10. Batch operations on the faces list: "Link to suggested" + "Delete"

**Problem.** Selecting multiple faces does nothing actionable. Users want to bulk-confirm top suggestions and bulk-delete junk clusters.

**Prior reference.** `FacesHub.tsx` had bulk "accept top suggestion" and bulk "delete" toolbars over selected clusters.

**Target files.**
- Backend: `src\Cove.Api\Controllers\FacesController.cs` — add `POST /api/faces/batch/link-top-suggestion` and `POST /api/faces/batch/delete`.
- Frontend: `ui\src\pages\FacesPage.tsx` — extend the existing selection toolbar (already imports `useMultiSelect`).

**Approach.**
1. `POST /api/faces/batch/link-top-suggestion` body `{ faceIds: number[], minConfidence?: number }`.
   - Server iterates: for each face that's unlinked and has a top suggestion ≥ `minConfidence` (default 0.6) with a `localPerformerId` (skip pure-external suggestions in batch — those need user choice), call the same code path as `Link`.
   - Returns `{ linked: [...], skipped: [{faceId, reason}], failed: [...] }`. Surface this as a toast in the UI.
2. `POST /api/faces/batch/delete` body `{ faceIds: number[] }`.
   - Reuses `Delete` per-face logic, runs in a single transaction with `IFaceLifecycleParticipant` cascade hooks (covers, exemplars, segments, detections, embeddings).
   - Returns counts.
3. Selection toolbar in `FacesPage.tsx` adds two buttons (visible only when `selecting`). The link batch button shows a confirmation modal listing how many will be linked, how many skipped, and a "min confidence" slider.

**Acceptance criteria.**
- Selecting 10 unlinked faces with valid suggestions and clicking "Link top suggestion" links them all in one request.
- Selecting and deleting 10 faces removes them and their owned artifacts (covers, exemplars, segments, detections, embeddings).
- Toast summarizes per-face outcomes (linked/skipped/failed counts).

---

## 11. Replace detection thumbnails on the detail page with scene/image thumbnails

**Problem.** Same root cause as #1 but specifically for the detail page's "Detections" section: it should show "Appears in" (scenes/images), not bbox crops.

**Target files.** Same as #1 + `ui\src\pages\FaceDetailPage.tsx`.

**Approach.** This is the consumer of #1's `appearances` endpoint. Specifics for the detail page:
- Group cards by `hostType`: "Scenes (N)" then "Images (M)".
- Each card: scene/image thumbnail, title, "Appears at 12:58–13:04 (3 segments, 108 frames)" subtitle.
- Clicking a scene card deep-links to the scene player and seeks to `firstSeenAtSec` (use the existing scene route's `?t=` param if supported; if not, leave a TODO and link to the scene page).
- Clicking an image card opens the image page.

**Acceptance criteria.**
- The detail page's evidence section shows one card per appearance host, not per detection row.
- Card content reflects segment-derived ranges (depends on #2 landing).

---

## 12. Use existing scene performers as evidence to boost suggestion confidence

**Problem.** Today suggestions are pure visual KNN. We have additional structured signal: the scenes the face appears in already have curated performers. If the suggestion's name (or `localPerformerId`) matches a performer already on one of those scenes, that's strong corroborating evidence; if it matches the **only** performer on the scene, even stronger.

**Prior reference.** `face_processor.py::_check_stashdb_matches_for_clusters` (≈ line 1199) and `_auto_link_cluster_to_stashdb` (≈ line 1277) used scene-side evidence (existing scene scrapers, performer co-occurrence) to bias auto-linking decisions.

**Target files.**
- `extensions\AI.Faces\AiFaceSuggester.cs` — extend the suggestion pipeline.
- `src\Cove.Api\Controllers\FacesController.cs` (`FaceSuggestionDto`) — add `evidence` array field.

**Approach.**
1. After the visual KNN produces a base score, fetch the scenes the face appears in (use #1's appearances data) and their performers.
2. For each candidate suggestion:
   - If candidate has a `localPerformerId` already on ≥ 1 of those scenes: add `+0.10` to confidence and append evidence `"Already on scene #42"` (cap at +0.20 across multiple scenes).
   - If candidate is the **sole** performer on at least one scene: additional `+0.10` and evidence `"Sole performer on scene #42"`.
   - For external (reference-pack) suggestions, fall back to **name match** against existing scene performers (case-insensitive normalized): smaller bump (`+0.05`) since the name match alone is weaker than an FK match.
3. Cap final confidence at `1.0`. Always preserve the raw visual score as `visualConfidence` so the UI can show "85% visual + 10% scene-evidence = 95%".
4. Evidence lines feed the existing "Evidence" rendering in `FaceSuggestionsPanel.tsx` (it already shows a list).

**Acceptance criteria.**
- Suggestion confidence is boosted (and evidence lines added) when the candidate is already a performer on the face's appearance scenes.
- Larger boost when the candidate is the sole performer on a scene.
- A unit test fixture covers each evidence path.

---

## 13. Linking a face propagates the performer to its scenes/images; unlink reverses

**Problem.** Today linking only sets `Face.PerformerId`. The user expects linking to also add that performer to all scenes/images where the face appears, and unlinking (or re-linking to a different performer) to roll back any **previously-AI-applied** performer assignments.

**Prior reference.**
- `skier_aitagging\face_api.py::_apply_performer_to_cluster_entities` (≈ line 61) — adds performer to scenes/images, **records** which entities did NOT already have the performer in `face_performer_assignments`, so we never remove an organically-added performer later.
- `skier_aitagging\face_api.py::_cascade_performer_unassignment` (≈ line 1427) — on unlink/relink, removes performer only from entities that:
  1. No longer have any face row in the cluster, AND
  2. Have no other cluster's assignment record for the performer, AND
  3. Have an assignment record from us (proving we were the original applier).

**Target files.**
- New extension-private table `ext_ai_faces_performer_assignments` (`FaceId, PerformerId, HostType, HostId, CreatedAt`) via `IDataExtension` migration in `AI.Faces`.
- New core/SDK `IFaceLinkParticipant` contract plus an `AI.Faces` implementation (e.g. `AiFacePerformerPropagation.cs`). This hook is not present in the current codebase, so add it and invoke it from `FacesController.Link`, `AcceptSuggestion`, any reference-import link path, and relink/unlink flows. It should receive at least `faceId`, `oldPerformerId`, `newPerformerId`, and the affected entity set or a way to query it.
- Cove SDK / core: confirm there are repository-level operations to add/remove a performer to/from a Scene/Image (e.g. `IScenePerformerService` or a `db.ScenePerformers` join entity). If not present, add a minimal core service so extensions don't have to reach into EF directly.

**Approach.**
1. **Apply on link** (`OnLink(faceId, performerId)` where `performerId != null`):
   - Resolve appearance entities (`(hostType, hostId)` pairs) for the face — same query as #1.
   - For each entity, check whether `performerId` is already assigned in core (don't double-add and don't record an assignment).
   - For entities where it was newly added, insert `ext_ai_faces_performer_assignments` rows.
2. **Cascade on unlink / relink** (`OnLink(faceId, newPerformerId)` where `newPerformerId == null` OR differs from prior):
   - Compute set `S` = entities where the face previously appeared.
   - For each `(faceId, oldPerformerId, hostType, hostId)` row in our assignment table:
     - If no other face cluster's assignment row in our table also covers `(performerId, hostType, hostId)`: remove the performer from the entity (via the core service). Do not require the face to have lost the entity entirely when the user is unlinking/relinking the face; the old performer should be removed from AI-owned assignments even though the face still appears there.
     - Always delete our assignment row for that face/performer/entity.
3. **Cascade on face delete** — already partially handled by `AiFacesDeleteParticipant`; extend it to also run the cascade-unassign for the face's prior `performerId`.
4. **Cascade on segment/detection removal that empties a face's coverage of an entity** — if a re-run removes a face's last appearance on a scene, we should treat that the same as unlink-for-that-entity. Implement via a small post-persist sweep in `AiFacesPersistenceService` that compares pre/post entity sets and runs the cascade on the diff.
5. **Settings toggle.** Add `AI:Faces:AutoApplyPerformers` (default `true`) and an equivalent in the AI.Faces settings panel so users can turn off propagation if they want manual control. (Mirrors `service.auto_apply_performers` in the legacy stack.)

**Acceptance criteria.**
- Linking face A to performer P1 adds P1 to every scene/image where A appears, except where P1 was already present.
- Unlinking removes P1 from those entities — and only those — that we originally added it to (verified by `ext_ai_faces_performer_assignments`), never from entities where the user had P1 manually.
- Relinking face A from P1 to P2 cascades a full unassignment of P1 (per the above) and a fresh assignment of P2.
- Deleting face A cascades the same as an unlink.
- Re-running detection that drops face A from scene X (no segment remains) removes our P1 assignment from scene X.
- Setting `AutoApplyPerformers=false` skips both apply and cascade.

---

## 14. Cross-cutting infrastructure changes (do these first)

These are prerequisites for the above; sequence them at the start.

### 14.1 New `Face` denorm columns
Add (or repurpose) on `Face`:
- `int AppearanceCount` — count of `FaceAppearance` rows for the face (drives "appears in N places" in UI; needed for #1, #2, #6, #11).
- `int FrameSampleCount` — optional raw analyzer sample total, derived from `FaceAppearance.SampleCount`; hide by default or expose only as an advanced/debug stat.
- `int SceneCount`, `int ImageCount` — distinct host counts from `FaceAppearance` (already declared in design doc; verify presence in `Face.cs` and add to migration if missing).

Do **not** add provider-specific fields such as top-suggestion JSON, StashDB match score, or reference-provider IDs to `Face`. Store those in extension-private tables or `CustomFields` only when they are intentionally part of an extension-owned display payload.

Also add the new core `FaceAppearance` entity + EF mapping + migration in `cove` (`src\Cove.Core\Migrations\` — match existing naming convention).

### 14.2 New `appearances` endpoint
Add `GET /api/faces/{id}/appearances` per #1 schema. Replace detail-page detection-grid usage with this. Keep `/detections` only for retained spatial samples/debug; X-Ray should be able to use segment keyframes.

Add a sibling shape for per-host views:

- `GET /api/scenes/{id}/faces` — compact face chips for a scene, grouped by face and segment.
- `GET /api/images/{id}/faces` — compact face chips for an image.

These endpoints unlock the scene/image face strips in §17.3 and prevent those pages from querying all detections manually.

Implementation note: `GET /api/faces/{id}/appearances` should read `FaceAppearance` as the canonical table and optionally join scene `Segment` rows for richer timeline metadata. The API contract stays uniform across scenes and images even though only scenes have segment ranges.

### 14.2.1 Detection retention policy
Add `AI:Faces:DetectionRetention = Keyframes | None | AllSamples` (default `Keyframes`) and implement it in `AiFacePreparationService.EmitDetections`/`AiFacesPersistenceService.PersistDetections`.

- `Keyframes`: persist first, best, last, and bbox-motion keyframes only.
- `None`: persist no face `Detection` rows for scenes; retain image detections only if the image page overlay needs them. All timeline/evidence data comes from `Segment.Payload`.
- `AllSamples`: current behavior, for debugging only.

Update `RefreshFaceStatsAsync` so it does not use retained detection row count as the user-facing face count; drive denorms from `FaceAppearance` instead.

### 14.3 Extension-private tables in `AI.Faces`
New tables registered via `IDataExtension`:
- `ext_ai_faces_exemplars` (#7).
- `ext_ai_faces_performer_assignments` (#13).
- `ext_ai_faces_top_suggestion_cache` (#6).
- `ext_ai_faces_track_samples` only if a future debug UI needs full original model samples without bloating `Detections`; store compressed JSON per track, not one SQL row per frame.
- (Optionally reuse the existing decision store table for #8 cleanup logic.)

### 14.4 Core "performer ↔ entity" service
Confirm existence; if absent, add `IPerformerEntityService { Task AddAsync(int performerId, EntityRef entity, ...); Task RemoveAsync(...); Task<IReadOnlyList<int>> GetPerformerIdsAsync(EntityRef entity, ...); }` in `Cove.Core` so #13 doesn't reach into `CoveContext` join tables directly from an extension.

---

## 15. Suggested implementation order

Backend foundations first, then UI:

1. §14.1 — `Face` denorm columns + migration.
2. §14.2.1 + §2 — Detection retention + `FaceAppearance`/track payloads (the primary correctness/storage fix; everything downstream renders better with this).
3. §14.2 — `appearances` + per-host face endpoints.
4. §1 + §11 — Detail/list use `appearances` (simple FE swap once endpoint exists).
5. §9 — Linked/Unlinked filter (small, isolated; quick win).
6. §3 — Detail page two-column layout (small, isolated).
7. §7 — Exemplars (extension-private table + endpoint + crop pipeline).
8. §4 — Suggestion DTO `localPerformerId`/`externalUrl` + larger thumbs + native links.
9. §5 — Side-by-side compare dialog (consumes #4 + #7).
10. §12 — Scene-performer evidence in suggester.
11. §6 — Top suggestion on list cards (uses #4, #5, #12; needs cache).
12. §10 — Batch link / batch delete.
13. §14.4 + §13 — Performer-link propagation (largest cross-cutting backend change; do after appearances and participant hooks are stable).
14. §8 — Diagnose + fix import-performer no-op (partly depends on link participant hooks landing in #13).
15. §17.1 — Post-run face review modal (high UX value once suggestions/exemplars/linking are solid).
16. §17.2 + §17.3 — Performer and scene/image face panels.
17. §17.4 through §17.8 — Secondary workflow polish (detach/split, quality filters, create performer, set performer image, metadata hydration, merge audit).

### 15.1 Current three-phase slice for quality-aware clustering

Use the following near-term slice when continuing face-clustering quality work:

1. Embedder metadata emission and pass-through.
  - Emit `pose_quality` (frontalness) and `image_quality` (sharpness / blur proxy) with each face embedding result.
  - Preserve those metadata fields end-to-end so `AI.Faces` can use them without re-running pose/blur analysis in the extension.
2. Quality-aware identity preparation and matching.
  - Use `pose_quality` + `image_quality` when deciding whether an embedding passes the hard floor, when ranking representative embeddings, when weighting centroid construction, and when choosing the face cluster's primary image / cover sample.
  - Keep the cluster matcher conservative, but move the "reuse existing cluster vs create new cluster" decision behind explicit knobs instead of fixed constants.
  - Minimum knobs required: `MinimumPoseQuality`, `MinimumImageQuality`, `IdentityMatchThreshold`, and `IdentityAmbiguityMargin`.
3. Operator-facing tuning UI.
  - Surface those knobs in the AI Data settings flow under the existing `AI Faces` section so users can tune cluster reuse aggressiveness without rebuilding or editing config files.
  - The settings UI should call the extension-owned settings endpoints, not reach into app-level config directly.

---

## 16. Test coverage expectations

- New unit tests under `C:\Users\tyler\source\repos\AI.Extensions\tests\AI.Extensions.Tests\`:
  - `FaceTrackBuilderTests` — fixtures for contiguous, gap-tolerant, gap-exceeded tracks (#2).
  - `FaceDetectionRetentionTests` — `None`, `Keyframes`, and `AllSamples` retention policies; default keyframe count is bounded (#0.1/#14.2.1).
  - `FaceExemplarSelectorTests` — quality + temporal-spread selection, dedup threshold (#7).
  - `FacePerformerPropagationTests` — apply, cascade-on-unlink, cascade-on-delete, no-double-add, no-remove-organic (#13).
  - `FaceSuggesterEvidenceTests` — boosts for shared / sole performer (#12).
  - `ReferenceImportRelinkTests` — link → unlink → re-link succeeds (#8).
- New integration tests under `C:\Users\tyler\source\repos\cove\src\Cove.Tests\`:
  - Extend `AiCoreControllerTests.cs` / `FaceSuggestionControllerTests.cs` with `appearances`, `linked` filter, batch endpoints.
- UI tests:
  - Extend `ui\src\test\FaceDetailPage.test.tsx` and `FaceSuggestionsPanel.test.tsx` for new layout, side-by-side dialog, native link affordances.
  - Add coverage that `FaceDetailPage` no longer calls the detections endpoint for normal rendering.
  - Add coverage for the review modal and per-host face panels when §17 lands.

Each item's acceptance criteria above doubles as the test plan for that item.

---

## 17. Additional UX gaps found during second review

These were present in some form in `stash-ai-server` / `skier_aitagging` and are still missing or thin in Cove. They are not all required to close the initial 13 bullets, but they matter for a complete face-review workflow.

### 17.1 Post-run face review modal

**Gap.** After a face run completes, the user is not guided through newly found unlinked faces. They must find the Faces page manually and inspect clusters one by one.

**Prior reference.** `frontend\src\FaceReviewPanel.tsx` — task-completion overlay with newly discovered clusters, top suggestion, side-by-side preview, accept/reject/defer/delete/create-performer actions, and completion summary.

**Target.** New `ui\src\components\FaceReviewModal.tsx` plus an `AI.Faces`/`AI.Core` job-result hook or notification payload that tells the UI which face IDs were created/updated by a run.

**Approach.**
1. Extend `AiDispatchResult`/job notes or add an extension endpoint `GET /api/ext/ai-faces/runs/{runId}/review` returning newly-created/unlinked faces with top suggestions.
2. On job completion, show a non-blocking "Review new faces" notification; opening it renders a modal queue.
3. Actions per row: compare/link, reject suggestion, defer, delete, import/create performer.

**Acceptance criteria.** Running face detection on a scene with new clusters offers a direct review workflow without requiring manual navigation to the Faces page.

### 17.2 Performer detail: Faces tab / linked clusters

**Gap.** Performer pages do not show linked face clusters, exemplars, or quick unlink. This makes it hard to audit whether a performer has bad face links.

**Prior reference.** `frontend\src\PerformerFacesPanel.tsx` — performer "Faces" tab with face grid, zoom levels, exemplar cycling, quick unlink, and cluster navigation.

**Target.** `ui\src\pages\PerformerDetailPage.tsx` plus backend query `GET /api/performers/{id}/faces` or reuse `GET /api/faces?performerId=` with richer DTO/exemplars.

**Approach.** Add a compact Faces tab/panel that shows linked clusters, top exemplars, appearance counts, quality score, and unlink/compare actions. Keep it secondary to the existing performer media/metadata layout.

**Acceptance criteria.** A linked performer page shows all linked face clusters and can unlink a bad cluster without visiting the global Faces page.

### 17.3 Scene/Image detail: compact face strip and context actions

**Gap.** Scene/image pages should show which faces were detected in that asset. The design doc already called for a subtle "Faces in this scene/image" strip, but the current UX doesn't expose it.

**Prior reference.** `frontend\src\SceneFacesPanel.tsx` and `ImageFacesPanel.tsx` — injected panels with face chips and actions.

**Target.** `ui\src\pages\SceneDetailPage.tsx`, `ui\src\pages\ImageDetailPage.tsx`, and the per-host endpoints from §14.2.

**Approach.** Add a small horizontal strip of face chips below existing performer metadata: cover/exemplar thumb, linked performer or label, confidence/appearance span. Context menu: open face, compare top suggestion, link/unlink, detach from this scene/image (§17.4), delete cluster.

**Acceptance criteria.** Scene/image detail pages expose detected faces without competing visually with curated performer cards.

### 17.4 Detach / split a face from one scene or image

**Gap.** If a cluster is mostly correct but one scene/image assignment is wrong, the only obvious current choices are global delete or merge/unlink. Legacy supported removing a face from a specific scene/image.

**Prior reference.** `face_api.py` scene/image detach endpoints and `SceneFacesPanel.tsx` context menu.

**Target.** `FacesController.cs` or `AI.Faces` endpoint: `POST /api/faces/{faceId}/appearances/{hostType}/{hostId}/detach`.

**Approach.** Remove or unassign the face's `Segment`/retained detection rows for that host, decrement/recompute counters, remove AI-owned performer assignment for that host if propagation had added it, and optionally mark the removed samples as ignored in extension-private state so reruns don't immediately reattach them unless forced.

**Acceptance criteria.** Detaching a face from scene X removes that scene from the face's appearances without deleting the whole face cluster.

### 17.5 Quality score, filters, and sorting

**Gap.** The preparation service has quality concepts (score/norm/anchor thresholds), but the UI cannot filter obvious low-quality clusters or sort by "needs attention."

**Prior reference.** `face_processor.py` quality gates and `face_api.py` query params for `quality_score` / sort (`sample_count`, `quality_score`, `suggestion_confidence`, `scene_count`, `image_count`).

**Target.** Store extension-owned `qualityScore` in `Face.CustomFields` or an extension table, expose it in `FaceDto` as an optional algorithm-owned display metric, and add list sort/filter controls in `FacesPage.tsx`.

**Approach.** Compute cluster quality from track/exemplar evidence, but stop treating it as only detection score + embedding norm. The quality model for matching and review should incorporate:

- `pose_quality` from face landmarks as a frontalness proxy.
- `image_quality` from the face crop as a sharpness / blur proxy.
- Detection confidence, embedding norm, sample/anchor count, and bbox size as secondary signals.

Use those signals in four distinct places:

1. Hard-floor filtering.
  - A sample should not become a representative anchor when it is too profile-heavy or too blurry, even if detection score and embedding norm are high.
2. Representative/centroid selection.
  - Cluster centroids should be weighted toward frontal, sharp exemplars rather than averaging all samples equally.
  - The cluster's primary image / cover should come from the best-known exemplar, not merely the latest request that touched the identity.
3. Cluster reuse vs. new-cluster creation.
  - Replace fixed matcher constants with extension-owned settings for minimum pose quality, minimum image quality, identity match threshold, and ambiguity margin.
  - These knobs define when a new sample is allowed to attach to an existing face cluster vs. when it must create a new provisional face.
4. Review/list sorting.
  - Default unlinked-face sorting should still prioritize high-confidence suggestions first, then high overall quality, then appearance count.
  - Add cleanup-oriented filtering such as "Only low quality" once the backend metric is exposed in `FaceDto`.

The tuning surface for these thresholds should live in the AI.Faces settings UI under the AI Data tab, backed by extension-owned persisted settings rather than global app config.

**Acceptance criteria.**

- Users can quickly review high-confidence suggestions and separately clean up low-quality/noisy clusters.
- Face covers and exported centroids prefer frontal, sharp exemplars over blurrier/profile-heavy samples from the same cluster.
- Operators can tune when AI.Faces reuses an existing cluster vs. creates a new one without code changes.

### 17.6 Create performer from face and set performer image from exemplar

**Gap.** If no metadata-provider match exists, the user needs a fast way to create a performer from the face cluster and optionally set the performer image from an exemplar.

**Prior reference.** `face_api.py` create-performer/link flow and `_set_performer_image_from_cluster`; `FacesHub.tsx` link/create dialog with "set performer image" and thumbnail index selection.

**Target.** `AiFacesExtension.cs` or Cove `PerformersController.cs`, `FaceDetailPage.tsx`, `FaceCompareDialog.tsx`.

**Approach.** Add "Create performer from face" action with name input, exemplar picker, and optional "set performer image" toggle. Use the existing crop/exemplar blob pipeline; do not create another crop cache.

**Acceptance criteria.** From a face detail/compare modal, a user can create a new performer, set the chosen face exemplar as the performer image, and link the face in one flow.

### 17.7 Metadata-provider hydration from reference matches

**Gap.** Reference-pack matches can create/import performers, but the workflow should optionally hydrate performer metadata and remote URLs from the metadata provider.

**Prior reference.** `face_api.py` StashDB hydrate/apply helpers and link-dialog "Hydrate from StashDB" toggle.

**Target.** `AiFaceReferencePerformerResolver.cs`, Cove metadata server services, `FaceCompareDialog.tsx` / import UI.

**Approach.** When a suggestion has a reference-provider identity, offer "Import performer metadata" and route through Cove's existing metadata-server performer merge/import service. Make the toggle visible in compare/import flows; default on when enough provider data is present.

**Acceptance criteria.** Importing a reference-only match can create/link a local performer and populate name, URLs, image, aliases, and other supported fields.

### 17.8 Merge history and recovery UX

**Gap.** `Face.MergedIntoFaceId` exists, but the UI does not make merge chains auditable or reversible. This matters when auto/quick cleanup merges the wrong clusters.

**Prior reference.** Legacy cluster list exposed merged status and retained `merged_into_id` for cleanup queries.

**Target.** `FacesPage.tsx`, `FaceDetailPage.tsx`, and `FacesController.cs`.

**Approach.** Add merged badge, "show merged into" link, and "unmerge" action on the source face. Keep merged faces hidden by default but accessible through an advanced filter.

**Acceptance criteria.** A user can find merged faces, navigate to the target cluster, and undo an accidental merge.
