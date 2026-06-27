# Run-AI UX and Pipeline Composition Proposal

A look at how AI.Core / AI.Extensions and `nsfw_ai_server` currently
collaborate, what makes the "Run AI" experience confusing, and several
design options that improve it for both casual users and power users.

---

## 1. Current System (as built today)

### 1.1 `nsfw_ai_server`

Cove treats the Python server as an API boundary. AI.Extensions does not
read or write the server's local YAML configuration; the shared contract is
the v4 model and pipeline API:

- `/v4/models/catalog` returns model metadata including config name,
  capabilities, supported scopes, categories, active state, and loaded state.
- `/v4/models/loaded`, `/v4/models/load`, and `/v4/models/unload` expose the
  current serving-model state.
- `/v4/pipelines/custom` lets AI.Core register or delete user-defined
  pipeline definitions at runtime.

The analyze contract exposed to clients is:

- Pick a pipeline (`image_pipeline_dynamic_v4`, `video_pipeline_dynamic_v4`,
  `audio_pipeline_v4`, …).
- Pass `requested_model_names`, `skipped_categories`, `threshold`,
  `time_interval`, `vr_video`, `load_policy`, etc.
- The server returns nested `childrenResults` with tags / detections /
  embeddings keyed by `model_category`.

The levers for "run this subset of models" are requested model names, load
policy, AI.Core capability/preset settings, and custom pipelines registered
through the server API. Legacy server-side capability YAML is not part of the
Cove/extension contract.

### 1.2 `AI.Core`

`AI.Core` is the broker between Cove and the Python server.

- `INsfwAiServerClient` (`NsfwAiServerClient.cs`) calls the server's model,
  pipeline, analyze, and text-encoding APIs and parses results through
  `AiAnalyzeResultParser`.
- `IAiCapabilityContributor` (in `AI.Extensions.Abstractions`) is the
  plugin point. Each extension publishes an `AiCapabilityDescriptor`
  with one or more `AiCapabilityClaim`s. A claim is the tuple
  `(claimId, mediaKind, wantCapability, wantScope, outputKey,
   PreferredModels[], FromDetection?)`.
- `AiCoreOrchestrator.SelectClaims` resolves the requested claim IDs,
  calls `AiRunPlanner` to figure out what needs (re)running based on
  prior `AiRun` history + currently-persisted artifacts, then issues
  one call to the server with the consolidated `Want` list and load
  policy. Results are dispatched back to each contributor's
  `DispatchAsync` for persistence.
- `AiCoreConnectionSettings` exposes `CapabilityModelBindings`,
  `RunPresets`, `CustomPipelines`, and `ModelSupersessionRule`s. Claim
  `PreferredModels` remain compatibility/default hints; actual model choice
  is resolved from bindings plus the server catalog.

### 1.3 The other extensions

- `AI.Faces` ships **four** claims:
  `faces.image.detection`, `faces.image.embedding`,
  `faces.video.detection`, `faces.video.embedding`.
  Detection and embedding are independent claims linked only by
  `FromDetection: "face_detector_torchexport"`.
- `AI.Visual` ships `visual.{image,video}.{feature,semantic}` (visual /
  semvisual embeddings).
- `AI.Tagging` ships `tagging.image.asset` and `tagging.video.frame`
  – one claim per media kind, with category selection and actual model
  choice resolved through catalog metadata, `CapabilityModelBindings`, and
  `CategoriesToSkip`.
- `AI.Audio` ships audio analysis claims.

### 1.4 What the Run AI dialog actually toggles

The dialog operates on **`ClaimIds`**. So a user sees toggles like:

- ☑ Image Face Detection
- ☑ Image Face Identity Embeddings
- ☑ Image Feature Embeddings
- ☑ Image Semantic Embeddings
- ☑ Image Tags
- ☐ … etc.

And the same list duplicated for video. There is no concept of a
"feature" that bundles related claims, no concept of "which tagging
model do I want today", and no way to express "scan for faces but
only embed them if confidence > X."

---

## 2. Why it's confusing

1. **Claims leak implementation detail.** "Detection" vs "Embedding"
   is a stage of the face-recognition pipeline, not a thing a user
   recognizes. The dialog lets you turn embeddings off while leaving
   detection on (or vice versa), which is never a sensible runtime
   choice.
2. **Image vs Video duplication.** Every feature appears twice (`...image.*`
   and `...video.*`). Users are picking media-kind tags that the
   orchestrator already knows from the entity type.
3. **Tagging needs category-aware selection.** Tagging has *one* claim per
  kind but can fan out to multiple category models. The run dialog and
  planner need to keep category choices visible so selecting tagging does
  not accidentally collapse to one model.
4. **Model selection must have one Cove-side home.** The current design keeps
  user choices in `CapabilityModelBindings` and treats server active/loaded
  model state as API data. `PreferredModels` are only default hints for
  compatibility and planner identity.
5. **No grouping / no presets.** The dialog can't say "Standard run",
   "Faces only", "Search-index rebuild" without the user re-ticking
   eight checkboxes every time.
6. **Last-used isn't saved per user / per host type.** Every dialog
   open is a fresh decision.
7. **No exposure of advanced wiring.** Power users who want
   "run detector X, embed with Y, run classifier Z only on region
   branches whose detection score > 0.6" have nowhere to express it.
  That wiring belongs in AI.Core custom pipeline definitions synchronized
  through the server API.

---

## 3. Design goals

| Audience  | Needs                                                                                    |
| --------- | ---------------------------------------------------------------------------------------- |
| Casual    | Pick **features** ("Facial recognition", "Tagging → Actions, Body"), not models or stages. |
| Casual    | Pick a small number of global presets per host kind.                                      |
| Casual    | Settings page where they choose which model powers each feature.                          |
| Advanced  | Define their own feature/preset, including model choice per stage.                        |
| Advanced  | Compose pipelines that bind a model to a detection sub-image / region branch.            |
| Both      | Plan & history (`AiRunPlanner`) still works; reruns still skip work already done.        |

---

## 4. Core concept changes

Three concept renames / additions unlock the whole UX redesign:

### 4.1 Promote the user-facing thing to **"Capability"** (or "Feature")

A capability is something the user understands:

- `faces` – facial recognition (detect + embed + identify)
- `tagging` – content tagging (actions, body, etc.)
- `visual.search` – semantic search index (semvisual)
- `visual.dedupe` – feature embedding (visual)
- `audio.transcribe`
- `audio.classify`

Each capability is owned by one extension and internally maps to one
or more existing `AiCapabilityClaim`s. The current claims keep
existing (planner cares about them) but become a hidden implementation
detail. The Run AI dialog and Settings page **only show capabilities**.

### 4.2 Bind each capability slot to a **`ModelBinding`** in settings

```jsonc
{
  "capabilityBindings": {
    "faces": {
      "detector":  "face_detector_torchexport",
      "embedder":  "face_embedding_torchexport"
    },
    "tagging": {
      "actions":   "tagging-actions",
      "body":      "tagging-body",
      "scene":     "tagging-scene",
      "people":    "tagging-people"
    },
    "visual.search": { "embedder": "semvisual" },
    "visual.dedupe": { "embedder": "visual" }
  }
}
```

Each binding *slot* is declared by the extension and constrained by the
catalog returned from `/models/catalog` (we already have
capability, scope, category, active, and loaded metadata to filter by). The
settings page renders a dropdown per slot populated from compatible models.
This replaces legacy category-specific preferences and hard-coded
`PreferredModels` lists as the source of user choice.

### 4.3 Add **Run Presets**

A preset is `(name, capabilities[], claim overrides, pipeline override,
categories to skip, load policy)`. Presets are global Cove settings.

- **Built-in presets** shipped by each extension (e.g. `AI.Faces`
  ships `"Facial recognition"`).
- **Custom presets** persisted in AI.Core settings and shared by all
  users.

This is the casual-user end of the spectrum.

---

## 5. Proposal options for the run-time experience

The chosen implementation is Option A for the normal-user flow plus
Option C1 for the advanced composition flow.

### Option A — "Capabilities + Presets" (minimum-change recommendation)

- Add the `Capability` concept as a thin layer over the existing claim
  system. An extension declares capabilities and what claims each
  capability implies (filtered by host media kind).
- Add a global `RunPreset` store in AI.Core settings.
- Settings page gains a **"Models"** section with one dropdown per
  declared `ModelBinding` slot, sourced from `/models/catalog`.
- Run dialog becomes:
  ```
  Preset:  [ Standard ▾ ]   [ Save as… ]
  ─────────────────────────────────────
  ☑ Facial recognition
  ☑ Content tagging      [ Actions • Body • Scene ▾ ]
  ☐ Semantic search index
  ☐ Feature embeddings
  ☐ Audio transcription
  [Advanced ▸]   [Run]
  ```
- "Advanced ▸" reveals the old claim toggles + load policy + threshold
  + frame interval.
- Wire change is minimal: the dialog can POST `capabilityIds`; capability
  → claim expansion happens in `AiCoreOrchestrator`. Raw `claimIds`
  still exist for the advanced path.

**Pros:** small surface change, instantly fixes "remove detector but
keep embedder" footgun (capability is atomic), keeps the existing
planner / journal contracts untouched.

**Cons:** still single-pipeline per run. No power-user composition
beyond what `nsfw_ai_server` already exposes.

### Deferred: Runtime Knobs

Per-capability knobs are not part of this implementation. The useful
runtime decisions are which capabilities/presets/pipelines to run;
model and pipeline details belong in global settings or advanced custom
pipelines.

### Option C — "Composable Pipelines" (the power-user escape hatch)

Option C lets Cove ship user-defined pipelines to the server through the
runtime custom-pipeline API (or model the same composition entirely on the
Cove side and call the server piecemeal).

Two sub-options for *where* composition lives:

#### C1. Cove registers pipeline definitions with the server

- Keep the `/v4/pipelines/custom` API that accepts a user pipeline
  `{ full_image_models, detector_models, region_models, audio_models }` and
  registers it at runtime.
- `AI.Core` gets a `CustomPipelineStore` and a UI page
  ("Pipelines") where advanced users describe:
  - full-image models to run,
  - detector model(s),
  - per-detector region branches: which models to run on each crop,
  - optional filters (`min_score`, `min_area`, `class IN (...)`).
- A user-defined pipeline appears in the Run dialog as a **custom
  capability**.

This matches `nsfw_ai_server`'s native vocabulary almost 1:1 and
keeps execution in one round-trip.

#### C2. Cove orchestrates multi-pass runs

- AI.Core gains a `Workflow` graph type: nodes are capability calls,
  edges are "run B on the region outputs of A" or "run C only if A
  emits class=person".
- Execute by issuing successive calls to `nsfw_ai_server`
  pipelines (or, where supported, a single region pipeline) and
  passing intermediate artifacts as the region targets for the next
  pass.
- Strictly more flexible than C1 (e.g. supports "run tagging model X
  on the *largest* face from detector Y") but requires multiple server
  round-trips and a real workflow runner in AI.Core.

**Recommendation for C:** start with **C1**. We already speak its
schema; the work is mostly UI + API synchronization. Promote to C2 only if
real workflows can't be expressed in a single server pipeline.

---

## 6. Recommended path

1. **Implemented foundation (Option A):**
   - Introduce `AiCapability` / `AiModelBindingSlot` abstractions
     alongside the existing `AiCapabilityClaim`.
   - Migrate Faces, Visual, Tagging, Audio extensions to declare
     capabilities (claims become an internal expansion).
   - Use unified global `CapabilityModelBindings`; tagging categories are discovered from
     model catalog metadata, so adding a new tagging category does not
     require Cove extension code changes.
   - Add global `RunPresets` and request-side `capabilityIds`,
     `presetId`, and `pipelineName` fields.

2. **Implemented backend foundation (Option C1):**
   - Add AI.Core endpoints for custom detector / region pipelines,
     persisted in Cove settings and pushed to `nsfw_ai_server` as
     runtime-registered pipelines.
   - Custom pipelines surface as additional capabilities in the Run
     dialog under a "Custom" group.
   - Manage custom pipelines with a separate
     `cove.community.ai.core.pipelines.manage` permission.

The current implementation preserves the planner / journal contracts so
reruns continue to behave correctly.

---

## 7. Open questions

- **UI follow-up.** Build the Run AI dialog around `capabilityIds`,
  global presets, and custom pipeline capabilities.
- **Settings UI follow-up.** Render model-binding dropdowns from the
  extension-declared slots plus `/models/catalog` metadata.
- **Advanced pipeline UI follow-up.** Build an editor for
  `full_image_models`, `detector_models`, `region_models`, and
  `audio_models`, then call the new AI.Core custom pipeline endpoints.
