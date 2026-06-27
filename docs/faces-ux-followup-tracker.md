# Face UX Follow-up Tracker

Created: 2026-05-03

## Scope

Track the follow-up issues reported after the face UX/backend pass. Update each item as it is investigated, implemented, and verified.

## Items

- [x] AI run skip behavior: rerunning a scene/image with the same models or lower-accuracy models should skip already-satisfied work, and skip the whole run when no claims need work.
- [x] Face detail suggestion evidence: reference suggestions with scene/sole-performer evidence must not display "No supporting face evidence was returned."
- [x] X-ray duplicate face boxes: avoid stacked duplicate face bounding boxes for the same face at the same timestamp.
- [x] Faces-over-time mismatch: make the timeline face bar agree with visible face boxes at the current timestamp.
- [x] Remove the blue bar at the bottom of the scene player/timeline if it is not useful.
- [x] AI run dialog persistence: save selected options reliably between runs/entity types.
- [x] AI run dialog styling: align the extension UI with Cove theme/classes.
- [x] AI run queue feedback: do not show blocking alert dialogs when Run AI queues a job.
- [x] AI run dialog force mode: expose a checkbox for "force-run and replace stored data."
- [x] AI run dialog load policy: remove the load policy option.
- [x] AI run dialog frame interval: default frame interval should be 2 seconds, not 60.
- [x] AI run dialog loaded status: show loaded state correctly for all loaded models, not only tagging models.
- [x] Post-run review dialog: show only faces from the just-completed single scene/image run.
- [x] Post-run review compare flow: open the compare dialog for suggested performers, and omit faces with no suggestion.
- [x] Post-run review multi-host rule: do not show the dialog when the AI job covered more than one scene/image.
- [x] AI completion feedback: do not open automatic AI completion/review dialogs; rely on the Jobs drawer instead.
- [x] Segment detail page layout: restore a usable sidebar/tabs structure and expose segment information through tabs.
- [x] Faces page controls: remove the "Primary only" dropdown option and remove the Clear button.
- [x] Faces page filters/sorts: add useful filters/sort options such as suggested match confidence and recently created.
- [x] Face detail appearances: "Appears In" should list scenes/images where the face appears, not retained detections.
- [x] Face timeline seeding: usable non-anchor detections should still create provisional face appearances so Faces over time lanes are not dropped.
- [x] Scene/image detail face strips: list unique faces only, not per-time-window appearances.
- [x] Scene/image detail face strip density: make face chips subtler and more compact.

## Verification

- 2026-05-03: `dotnet test tests\AI.Extensions.Tests\AI.Extensions.Tests.csproj --no-restore` passed: 38/38 tests.
- 2026-05-03: `dotnet build src\Cove.Api\Cove.Api.csproj --no-restore -o temp\build-verify\Cove.Api` passed.
- 2026-05-03: `npm run build` in `cove\ui` passed; existing Rollup/chunk-size warnings remain.
- 2026-05-04: `dotnet test .\tests\AI.Extensions.Tests\AI.Extensions.Tests.csproj --filter "FullyQualifiedName~AiCoreOrchestratorTests"` passed after the run-history planner refactor.
- 2026-05-04: `dotnet test .\tests\AI.Extensions.Tests\AI.Extensions.Tests.csproj --filter "FullyQualifiedName~AiCoreOrchestratorTests"` passed after tightening the face-history skip rule so `AI.Faces` reruns when run history exists but the host has no persisted face artifacts.
- 2026-05-04: Live `POST /api/ext/ai-core/run/video` for scene `5160` returned `analysis.status = "skipped"` without calling the analyze path when completed run history already satisfied the request.
- 2026-05-04: Live `/face/68` verification showed the `Appears In` tab listing scene/image appearances instead of retained detections.
- 2026-05-04: Live `/scene/5160/span/149d539f8a444e79?profile=5` verification showed the restored tab rail with working `Intervals` and `Provenance` tabs.
- 2026-05-04: `npm run build` in `cove\ui` passed after the Faces page control update; existing Rollup warnings remain.
- 2026-05-04: Live `/faces` verification showed sort options including `Suggested match confidence`, `Recently created`, and `Recently updated`, no `Clear all` chip action, and a merge-state filter offering `Primary and merged` and `Merged only`.
- 2026-05-04: `dotnet test .\tests\AI.Extensions.Tests\AI.Extensions.Tests.csproj --filter "FullyQualifiedName~AiFacesPreparationServiceTests"` passed after allowing usable non-anchor face samples to seed provisional identities/appearances.
- 2026-05-04: Live `/scene/5160` verification showed the Run AI dialog using Cove dialog/theme classes, no queue alert after clicking `Run AI`, no automatic review/completion dialog, and `/api/jobs` returning `[]` after the run settled.
- 2026-05-04: Live `/scene/5160` verification showed `Faces over time` present with a `Show faces` toggle and two visible face lanes when expanded.
- 2026-05-04: Live `/scene/1455` verification showed the pre-run state had no scene faces, no `Faces in this scene`, and no `Faces over time`, while `/api/scenes/1455/faces` returned `[]`.
- 2026-05-04: Live `/scene/1455` Run AI verification on theme `dark-rose` with component style `glass animated gradient` showed the dialog using Cove-responsive surfaces and shape classes; a temporary `data-component-style` probe changed the dialog border radius from `4px` (`minimal`) to `12px` (`rounded`), confirming the dialog now follows Cove theme/style selectors instead of fixed extension-owned styling.
- 2026-05-04: Live rerun of scene `1455` completed with face detector and face embedder models in the persisted `AiRun`, after which `/api/scenes/1455/faces` returned `7` faces, `/api/embeddings?hostType=scene&hostId=1455&modality=face` returned `59` embeddings, `/api/scenes/1455/detections` returned `78` detections, and the scene page showed both `Faces in this scene` and `Faces over time · 3 lanes`.

## Notes

- Exemplars remain intentionally skipped for now.

## 2026-05-20 Follow-up Items

Capture of the current unresolved faces, detections, segments, and UI issues. Only mark these complete after the implementation is actually changed and verified.

### Faces, Detections, And Segment Windows

- [x] Face-backed video segments should cover the visible face span, not only tiny sparse keyframes. For a face visible across an 11:47 scene, the expected result is one face segment covering the scene, with retained detections/keyframes only where bbox movement warrants them.
- [x] Face-backed scene player overlays should consume persisted segment `bestBbox`/`keyframes` even when retained sparse detections are absent for the current timestamp.
- [x] Face swimlanes should show continuous persisted face segment windows, and linked faces should prefer performer names on swimlanes and player bboxes.
- [x] X-ray/player toggle text should be `X-ray`, not `hide face boxes on video`, and should support future non-face detection boxes rather than face-only wording.

### Segment List Cards, Filters, And Sorting

- [ ] Raw and resolved segment cards should use the exact same reusable card layout. Raw-only actions may include delete; derived/resolved cards should not carry a card-level `View raw segments` button.
- [x] Segment cards should clearly display segment kind, provider/source, tag, performer, face/reference data, and confidence with human-readable labels instead of unlabeled boxes.
- [ ] Segment list should support filtering and sorting by confidence, kind, tag, performer, face, and provider/source category (`user` vs `extensions`).

### Segment Detail Pages

- [x] Segment detail pages should use the scene-style `MediaDetailLayout`/detail action patterns, including an overflow `...` menu like scenes.
- [x] Segment detail pages should remove page-owned playback controls such as progress bar, restart, loop, and time jump controls because those belong to the player.
- [x] Remove useless instructional copy such as `Resolved Span Playback` / `Playback follows the resolved span intervals and automatically skips the gaps between them.`
- [ ] Union, resolved, and raw segment detail pages should be aligned into one reusable page with only small kind-specific differences.
- [ ] Segment detail pages should hide payload and overwhelming internals, and show human-readable user-facing details only: range, created/updated, scene jump, tag/performer/detection, provider/source, make scene from segment, edit, next same-tag segment, intersecting segments, union/intersection details where applicable, and similar scenes by segment.

### Face Cards And Face List

- [x] Face cards should show the top suggestion above the bottom icon/count row.
- [ ] Face cards should use the standard reusable entity card behavior, including normal hoverable image/scene icons.
- [x] Host-only performer suggestion confidence should be much lower: roughly 40% for the only performer on one host, about 30% with two performers, and about 20% with four or more performers; confidence should rise with repeated host evidence but remain weaker than SAIE/reference matches.
- [x] Face list action should read `Link suggested`, not `Link suggested 60%+`; confidence threshold should come only from the active filter.
- [x] Suggestion confidence filter UI should use the standard numeric filter input pattern, like performer image-count filters, rather than a custom nonstandard input.
- [x] Face cards should hide scene/image icon counters when those counts are zero.
- [x] Face batch `Link suggested` and `Delete` actions should use standard compact batch-action styling.
- [x] Face list filters should support linked performer filtering and top-suggestion performer filtering through the standard performer multi-select criterion.

### Face Detail

- [x] `Best Samples` should not be inserted as a separate section in the current position; cover/compare panes should allow left/right navigation through real face crops.
- [x] Compare popup images should use actual face images/crops rather than generic scene thumbnails.
- [x] `Appears In` should use standard scene/image cards with additional face-specific info such as frames and segment ranges.
- [x] `Merge`, `Create performer`, and similar actions should live in the overflow `...` menu.
- [x] Face detail edit button styling should match standard `EntityHeroLayout` action styling.
- [x] Face detail cover should support left/right navigation through the available face cover/crop images.
- [x] Face edit should expose only the title field, with the field labeled `Title` instead of `Label`.

### 2026-05-20 Verification

- `dotnet build extensions\AI.Faces\AI.Faces.csproj` passed after changing sparse video tracking to use analyzed-slice order and inferred sampling intervals.
- `dotnet test tests\AI.Extensions.Tests\AI.Extensions.Tests.csproj --filter "AiFaceSuggesterTests|AiFacesPreparationServiceTests"` passed: 29/29 tests.
- `npm run build` in `cove\ui` passed after the player, face card/list/detail, segment card, and segment detail changes; existing Rollup/chunk-size warnings remain.
- `npm run build` in `cove\ui` passed again after switching face suggestion confidence to the reusable numeric editor and adding face-card scene/image hover popovers.
- `dotnet test tests\AI.Extensions.Tests\AI.Extensions.Tests.csproj --filter "AiFaceSuggesterTests|AiFacesPreparationServiceTests"` passed: 29/29 tests after changing suggestion evidence URLs to cropped detection media.
- `dotnet build src\Cove.Api\Cove.Api.csproj -o artifacts\verify\face-crops -p:UseAppHost=false` passed after adding `/api/stream/detection/{id}/crop`; the temporary output was removed.
- A normal Cove API Debug build was blocked by the running `Cove.Api` process locking Debug DLLs; the API compile was re-verified with isolated output `artifacts\verify\face-crops-final`, then removed.
- `npm run build` in `cove\ui` passed after moving face crop samples into the compare dialog carousel and removing the separate `Best Samples` block.
- `dotnet build src\Cove.Api\Cove.Api.csproj -o artifacts\verify\ai-ext-followup -p:UseAppHost=false` passed, and the temporary output was removed.
- `scripts\stage-local-extensions.ps1 -Configuration Debug` passed and restaged all six local AI extensions to `%LocalAppData%\cove\extensions` after the face evidence crop URL update.
- The rebuilt Cove UI assets were copied from `src\Cove.Api\wwwroot` to `src\Cove.Api\bin\Debug\net10.0\wwwroot` for the Debug runtime after the latest UI build.
- 2026-05-21: `npm run build` in `cove\ui` passed after the face carousel/filter/action/card fixes, segment card label unification, scene face-segment overlay wiring, and segment detail action alignment; existing Rollup/chunk-size warnings remain.
- 2026-05-21: `dotnet build src\Cove.Api\Cove.Api.csproj -o artifacts\verify\ai-extensions-cove-api -p:UseAppHost=false` passed after adding performer/top-suggestion performer and numeric suggestion confidence face filters.
- 2026-05-21: `dotnet build AI.Extensions.slnx` passed after changing AI.Faces suggestion evidence URLs and cover-quality persistence.
- 2026-05-21: `dotnet test AI.Extensions.slnx --no-build` passed: 82/82 tests after moving AI.Faces cover-quality metadata to Cove custom field rows.
- 2026-05-21: The rebuilt Cove UI assets were mirrored from `src\Cove.Api\wwwroot` to `src\Cove.Api\bin\Debug\net10.0\wwwroot` for the Debug runtime.
- 2026-05-21: `scripts\stage-local-extensions.ps1 -Configuration Debug` passed and restaged all six local AI extensions to `%LocalAppData%\cove\extensions`.

## 2026-05-21 Follow-up Items

Capture of the latest live-review fixes from screenshots and segment/face testing. Only mark these complete after implementation and verification.

### Bbox Keyframes And Face Crop Quality

- [x] Bbox keyframe generation should be more aggressive when a face moves, using configurable error-over-time style scoring rather than a tiny cap of sparse detections for long scenes.
- [x] Bbox aggressiveness should be configurable per detection/bbox kind and use case, with the reusable scoring logic living in Cove where future non-face detections can share it.
- [x] Face sample images should be investigated and fixed so the three carousel/compare images are actual good face crops rather than black frames, legs, or unrelated scene regions.

### Scene Timeline And Segment Cards

- [x] Face-backed segments should be hidden from the generic segments swimlane when they already appear in the dedicated face swimlane.
- [x] Segment cards should show only a right-aligned range such as `0:04 - 0:20`, without a duration suffix.
- [x] Segment list cards should support hover playback/preview from the list page.

### Segment Detail Pages

- [x] Segment detail pages should be flat like scene detail pages and use the standard action layout with `Open parent scene` as a primary action plus an overflow `...` menu.
- [x] `Make scene` should live in the segment detail overflow menu, not as a standalone button.
- [x] Segment detail should remove the `Context` and `Resolved spans` tabs.
- [x] Segment detail should not have an edit action button because edit is already a tab.
- [x] Raw segment detail should put the `Edit` tab at the bottom of the tab list.
- [x] The segment overview should show the parent scene using a card-style presentation instead of a bare `Scene` title.

### Segment Search, Filters, And Derived Spans

- [x] Segment search should match tag names and aliases, performer names and aliases, face titles/labels, and linked face performer names/aliases.
- [x] Segment filters should include segment type/kind filters such as tag, performer, face, and other segment kinds.
- [x] Segment filters should support filtering directly by faces, tags, and performers.
- [x] Derived span filters should be simplified so only the selector for the selected kind appears, rather than showing tag, performer, and face selectors at the same time.
- [x] Typing in tag/entity filter selectors should not make the filter dialog flash or remount on each letter.
- [x] Union/intersection/difference derived spans should explain their operands clearly on the overview page.
- [x] Provenance should not be a separate tab on segment pages; user-facing provenance details should be folded into the overview tab.

### Face List And Face Detail Polish

- [x] Face filters should not render a divider between filter groups.
- [x] Face detail should use the same standard edit button styling as performer detail pages.

### 2026-05-21 Verification

- `npm run build` in `cove\ui` passed after segment list/detail, derived-span, face-list, face-detail, and face sample ranking changes; existing Rollup/chunk-size warnings remain.
- `dotnet build src\Cove.Api\Cove.Api.csproj -o artifacts\verify\segments-faces-pass -p:UseAppHost=false` passed after segment alias/direct-performer search and filter changes; the temporary output was removed.
- `dotnet test AI.Extensions.slnx` passed: 84/84 tests after the AI.Faces keyframe/crop-quality changes.
- `dotnet build src\Cove.Api\Cove.Api.csproj` passed after stopping the running Debug Cove API so the new Cove.Core bbox keyframe selector could be copied into the normal runtime output.
- `scripts\stage-local-extensions.ps1 -Configuration Debug` passed and restaged all six local AI extensions to `%LocalAppData%\cove\extensions`.
- The rebuilt Cove UI assets were copied from `src\Cove.Api\wwwroot` to `src\Cove.Api\bin\Debug\net10.0\wwwroot` for the Debug runtime.
- Cove API was restarted from `src\Cove.Api\bin\Debug\net10.0\Cove.Api.exe`; startup initialized `cove.community.ai.faces` successfully and reported Cove on port 9999.

## 2026-05-21 Segment Detail And Filter Follow-up Items

Capture of the latest live-review fixes from segment detail, derived segment, face sample, and global icon feedback. Only mark these complete after implementation and verification.

### Face Samples And Segment Preview Playback

- [x] Non-cover face captures in the face detail carousel/compare flow should use higher-resolution crops comparable to the primary cover image.
- [x] Segment list hover previews should only play while the segment item is hovered, matching the scene page hover-preview behavior, and should not keep playback/CPU work running after hover ends.

### Segment Detail Layout And Actions

- [x] Raw and derived segment detail pages should be flat like scene detail pages, without extra visual container cards around ordinary detail data.
- [x] Segment detail pages should not show the word `Provenance`; user-facing labels should describe the source/details directly.
- [x] The scene reference at the top of segment detail pages should be presented as a scene card instead of plain scene title text.
- [x] Segment `Set cover` should affect only that segment, not the parent scene cover.
- [x] Derived/intersection/union/difference pages should show the same kind of user-facing segment info as raw segment pages where it makes sense.

### Segment Search, Filters, And Cards

- [x] Segment search should work for derived/union/intersection/difference segments, not only raw segments.
- [x] Segment type/source filters should use dropdowns/selectors instead of freeform string text boxes.
- [x] Duration filters should use standard `mm:ss` inputs.
- [x] Duration and confidence filters should expose the standard numeric comparison options, including less-than style operators.
- [x] Intersection/union/difference cards should show what operands they are built from directly on the cards.
- [x] Raw and resolved/derived segment filters should use the same filter definitions, with raw vs resolved remaining as a mode/toggle rather than separate filter experiences.

### Global Cleanup

- [x] Across Cove, use the sparkles/similar icon for similar actions/views instead of the text `S` glyph currently used in many places.
- [x] Ignored face state and merge state UI should be removed; ignored faces should be deleted instead, and merged faces should not be exposed as a user-facing list/filter state.

### 2026-05-21 Segment Detail And Filter Verification

- `npm run build` in `cove\ui` passed after the segment preview, face sample, segment detail, segment filter/search, derived card, similar icon, and face state UI changes; existing Rollup/chunk-size warnings remain.
- `dotnet build src\Cove.Api\Cove.Api.csproj -o artifacts\verify\segments-filters-pass -p:UseAppHost=false` passed after the derived span search/filter backend changes; the temporary output was removed.
- `dotnet build src\Cove.Api\Cove.Api.csproj` passed after stopping the running Debug Cove API so updated Cove.Core/Cove.Data/Cove.Api DLLs could be copied into the normal runtime output.
- The rebuilt Cove UI assets were copied from `src\Cove.Api\wwwroot` to `src\Cove.Api\bin\Debug\net10.0\wwwroot` for the Debug runtime.
- Cove API was restarted from `src\Cove.Api\bin\Debug\net10.0\Cove.Api.exe`; startup initialized the AI extensions and reported Cove on port 9999.
- `dotnet build src\Cove.Api\Cove.Api.csproj -o artifacts\verify\segment-cover-pass -p:UseAppHost=false` passed after adding segment-owned cover storage, image endpoints, and the duration-only derived-filter trigger; the temporary output was removed.
- `npm run build` in `cove\ui` passed after restoring Segment `Set Cover` through segment-owned cover endpoints; existing Rollup/chunk-size warnings remain.
- `dotnet build src\Cove.Api\Cove.Api.csproj` passed for the normal Debug runtime after the segment cover migration/API changes.
- `POST /api/database/migrate` applied migration `20260521181800_SegmentCoverImage` successfully, with backup `cove_backup_20260521_173726_pre_migration.sql` created before migration.
- Cove API was restarted again from `src\Cove.Api\bin\Debug\net10.0\Cove.Api.exe`; startup reported `Database is up to date`, pre-warmed EF Core, initialized AI extensions, and started on port 9999.
- Live `GET /api/segments/47559/image?max=64` returned image bytes, confirming the segment-owned cover image route is reachable after migration.