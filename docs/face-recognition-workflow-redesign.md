# AI.Faces Recognition Workflow Redesign

## Problem

The current AI.Faces pipeline creates too many global face identities from one video asset. A scene with two real performers can end up with dozens of visible `Face` rows because the pipeline treats short visual fragments as independent people.

The observed failure mode is strongest when performers move through varied poses, angles, camera distances, occlusions, cuts, and position swaps. The tracker fragments the video, the global matcher is too strict for clustering, and every unmatched fragment can become a first-class face.

## Current Pipeline

The current flow is effectively:

1. Read detections and embeddings from an AI run.
2. Build image tracks or video tracks.
3. Select representative embeddings from each track.
4. Match representatives against stored global identities.
5. Create a new `face-####` identity if no match is found.
6. Persist prepared faces, detections, segments, appearances, and covers.

That design makes early tracking mistakes permanent. Once a video fragment becomes a global identity and a Cove `Face` row, later stages must clean up a polluted identity list rather than preventing the pollution.

## Root Causes

### IoU-only Tracking

Video tracking currently depends primarily on box overlap. This works for smooth motion, but it breaks when a performer turns, jumps in scale, crosses another performer, leaves frame briefly, or reappears after a cut. Low overlap does not mean a new person.

### Authentication-style Matching

Global identity matching uses a threshold that is closer to face verification than asset-local clustering. It is conservative enough to avoid some false merges, but too conservative for same-person fragments across pose and quality changes.

### Anchor Bias

Representative selection favors frontal, sharp, large faces. Those samples are good covers, but they are not enough to describe a person across profiles, partial occlusion, side angles, distance, and lighting changes.

### No Per-asset Clustering

The pipeline jumps from tracks directly to global identity matching. It does not first ask the asset-level question: within this scene, how many recurring people do these tracks represent?

### Eager Face Creation

Unmatched fragments are promoted immediately into global identities and visible `Face` rows. This turns uncertainty into user-facing clutter.

### SAIE Used Too Late

SAIE references are useful performer anchors, but they are currently used mostly for post-hoc suggestions. They should help cluster and label known performers before the system mints unknown identities.

## Target Architecture

Replace the single-pass `track -> match-or-create -> persist Face` flow with a tiered workflow:

1. Tier 1: embedding-aware video tracker.
2. Tier 2: per-asset face clusterer.
3. Tier 3: seeded global identity reconciler and promotion gate.

The key rule is that a video fragment should not become a durable global face until it has enough evidence, matches a known performer/reference, or has been reviewed.

## Phase 0: Stop-the-bleeding Baseline

Goals:

- Remove the short-track suppression patch that hides evidence but does not solve clustering.
- Lower the default identity match threshold from strict verification behavior toward clustering behavior.
- Lower the ambiguity margin slightly so a clear nearest same-person match is accepted more often.
- Prefer same-run and same-asset merge opportunities before creating new global identities.

Expected result:

- Fewer new identities are minted for same-person fragments.
- No detections are silently discarded just because a track is short.
- Existing tests that encode old suppression behavior are updated to assert merging instead.

## Phase 1: Embedding-aware Tracker

Goals:

- Replace IoU-only video track assignment with a combined score using box overlap, embedding similarity, frame gap, and temporal continuity.
- Allow lower IoU when embedding continuity is strong.
- Allow a longer gap for re-identifying the same performer across brief disappearances or cuts.
- Prevent nearby different performers from merging when embeddings disagree.

Core rules:

- High IoU can continue a track even if embedding evidence is weak.
- High embedding similarity can continue a track even if IoU is low.
- Longer gaps require stronger embedding evidence.
- Two detections in the same frame cannot belong to the same track.

Expected tests:

- Same embedding with low IoU bridges into one track.
- Different embeddings with overlapping or nearby boxes remain separate.
- Longer gaps require stronger embedding similarity.

## Phase 2: Per-asset Clusterer

Goals:

- Introduce an asset-local clustering stage after track building and before global identity matching.
- Merge tracks that represent the same person within the same scene or image batch.
- Use cluster centroids and pose-diverse representatives, not only a single best frontal sample.
- Enforce a concurrency veto so two faces seen at the same time are not merged into one person.

Cluster evidence:

- Representative embedding centroid.
- Best cover candidate.
- Detection count and frame/time range.
- Source track ids.
- Concurrent frame conflicts.
- Pose and quality spread.

Expected tests:

- Multiple fragmented tracks with similar embeddings become one asset cluster.
- Simultaneous different performers do not merge even if embeddings are moderately similar.
- Cluster output preserves all detections and appearances.

## Phase 3: SAIE-seeded Matching

Goals:

- Use known scene performers and SAIE reference packs as matching seeds before creating unknown identities.
- Bind a local asset cluster to a known performer when it clearly matches that performer's references.
- Reuse or create a performer-backed global identity for seeded matches.

Core rules:

- Scene performer references are considered before generic global unknown identities.
- A strong SAIE/reference match can promote immediately because it is tied to a known performer.
- Rejected suggestion decisions must still be respected.
- Ambiguous seeded matches stay provisional.

Expected tests:

- A cluster matching a scene performer's SAIE references resolves to that performer-backed identity.
- Ambiguous matches between two known performers do not auto-bind.
- Prior rejected suggestions block seeded auto-binding.

## Phase 4: Global Reconciler and Promotion Gate

Goals:

- Add identity lifecycle state: provisional and promoted.
- Stop writing first-class Cove `Face` rows for unpromoted unknown clusters.
- Promote identities when evidence is strong enough or when they are known performer-backed.
- Keep provisional detections and clusters available for review/future reconciliation without cluttering the normal Faces list.

Promotion candidates:

- SAIE or performer-backed identity with clear confidence.
- Unknown identity observed across enough detections in a single asset and passing ambiguity checks.
- Unknown identity observed across multiple distinct assets.
- User-reviewed cluster.

Expected result:

- Main Faces list contains durable, useful identities.
- Single asset fragments can be retained as provisional evidence without becoming dozens of faces.
- Future background consolidation can promote or merge provisional clusters when more evidence arrives.

## Later Phases

### Phase 5: Quality-aware Cover Selection

Cover selection should be separate from identity matching. The best cover should balance sharpness, face size, pose, occlusion, brightness, and representativeness. A cover should not drive identity clustering by itself.

### Phase 6: Cove Review Queue

Cove should expose provisional clusters in a review flow where users can confirm, merge, bind to a performer, reject, or hide clusters without polluting the normal Faces list.

### Phase 7: Background Consolidation

A background reconciler should revisit provisional clusters when new assets, references, or user decisions arrive. This allows conservative initial processing while still improving over time.

### Phase 8: Telemetry and Calibration

Add diagnostics for track count, cluster count, created identities, promoted identities, rejected merges, seeded matches, ambiguity decisions, and thresholds. Keep anonymized or local-only metrics suitable for calibration tests.

## Implementation Order

1. Save this design document.
2. Inspect current AI.Faces preparation, identity state, reference pack, and persistence contracts.
3. Implement phase 0 and update tests.
4. Implement phase 1 and add focused tracker tests.
5. Implement phase 2 and add clusterer tests.
6. Implement phase 3 and add seeded matching tests.
7. Implement phase 4 with the smallest persistence change that prevents unknown provisional clutter.
8. Run focused tests, then the AI.Extensions test suite.

## Non-goals For Phases 0-4

- Do not build the full Cove review UI yet.
- Do not rewrite all face persistence tables unless phase 4 proves it is unavoidable.
- Do not discard detections just to reduce visible face counts.
- Do not tune thresholds blindly without tests showing the intended behavior.