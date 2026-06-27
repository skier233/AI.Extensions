# Semantic Visual Search — Debugging Handoff

## Symptom
Semantic visual search in cove returns **no results**, even though many videos have
semantic visual embeddings stored (modality `visual`, source `ext:ai.visual`, model
`visual_embeddings_semvisual`, host type Video).

## Repos
- `C:\Users\tyler\source\repos\AI.Extensions` (the AI extensions: AI.Core, AI.Visual, …)
- `C:\Users\tyler\source\repos\cove` (the host app + SDK: Cove.Core, Cove.Data, Cove.Plugins, Cove.Sdk, Cove.Api)
- cove dev server runs at `http://127.0.0.1:5173` (Vite frontend, `/api/*` proxied to backend). Authenticated session is live in Vivaldi.

---

## Root causes already found & FIXED (confirmed working: encode now reaches nsfw_ai_server)

Since the "extensions-runtime redesign," each extension runs in its **own isolated DI
container** (`Cove.Plugins/ExtensionServiceOverlay.cs`). Cross-extension/host services are
shared only through `IExtensionServiceExchange` (publish via
`CoveExtensionBase.PublishContributions<T>`; consume via `serviceExchange.GetAll<T>()`).
The text-encoder path was never migrated to this model, so AI.Visual (running a search in
its own container) could not see AI.Core's `ITextEncoder`, and the query was never encoded
→ never sent to nsfw_ai_server.

Fixes applied:

1. **`AI.Extensions/extensions/AI.Core/ExtensionSemanticTextEncoder.cs`** (NEW) — singleton
   `ITextEncoder` bridge that opens a scope per call and delegates to the real scoped
   `AiCoreSemanticTextEncoder` (mirrors `AI.Faces/ExtensionFaceContributions.cs`). Must be a
   singleton because `PublishContributions` resolves from the extension's root provider.
2. **`AI.Extensions/extensions/AI.Core/AiCoreSemanticTextEncoder.cs`** — added
   `public const string SemanticKindFamily = "semantic.v1";` and `KindFamily => SemanticKindFamily`.
3. **`AI.Extensions/extensions/AI.Core/AiCoreExtension.cs`** —
   `ConfigureServices`: `AddScoped<AiCoreSemanticTextEncoder>()` +
   `AddSingleton<ITextEncoder, ExtensionSemanticTextEncoder>()`;
   `InitializeAsync`: added `PublishContributions<ITextEncoder>(services);`.
4. **`cove/src/Cove.Data/Services/EmbeddingService.cs`** (the `ITextEncoderRegistry`) — the
   ACTUAL root cause. `Resolve()` only read its per-container injected
   `IEnumerable<ITextEncoder>`, never the exchange. Now it also merges
   `serviceExchange.GetAll<ITextEncoder>()` (constructor takes optional
   `IExtensionServiceExchange? serviceExchange = null`), mirroring
   `Cove.Api/Controllers/FacesController.ActiveSuggesters()`. Test call sites
   `new EmbeddingService(context, [])` still compile (param is optional).

Result after rebuild of cove + AI.Core: the query text now reaches `/v4/encode/text`. ✅

---

## REMAINING PROBLEM (this is what the next agent needs to solve)
Encode works, server is hit, a vector comes back — but the KNN lookup returns nothing.

Every filter the search applies has been traced. The search
(`AI.Extensions/extensions/AI.Visual/AiVisualSemanticSearchService.cs`,
`SearchRankedMatchesAsync`) calls `KnnAsync` with:
`HostType=Video`, `Kind="visual.semantic.v1"`, `KindFamily="semantic.v1"`,
`Modality=Visual`, `IsSemantic=true`, `SourceKey="ext:ai.visual"`.
- `ResolveAllowedVideoIdsAsync` returns null when no object filter → not the cause.
- `ApplyVisualCandidateCutoff` never empties a non-empty set → not the cause.
- The prep service (`AiVisualPreparationService` + `AiVisualPersistenceService`) writes
  exactly `Kind="visual.semantic.v1"`, `KindFamily="semantic.v1"`, `IsSemantic=true`,
  `Modality=Visual`, `SourceKey="ext:ai.visual"` — matches the query (for embeddings made by
  the current code).

### Prime suspect: dimension mismatch (silent, total filter)
`cove/src/Cove.Data/Services/EmbeddingService.cs` → `KnnAsync`:
```csharp
var dimensions = query.ToArray().Length;          // text-embedding dim
var embeddings = ApplyFilters(...)
    .Where(embedding => embedding.Dim == dimensions);  // drops EVERY row of a different dim
```
Cross-modal search compares a **text** embedding (query, `semantic.v1` text model) against
stored **image/frame** embeddings (`semvisual` model). If the text model's output dim ≠ the
stored `semvisual` dim, this line removes all candidates → zero results, no error.

---

## NEXT STEPS — diagnostics to run (need DB and/or authenticated-app access)

### 1. Check stored embedding identifiers + dimension
Run against cove's Postgres:
```sql
SELECT "Kind","KindFamily","Modality","IsSemantic","Dim", count(*)
FROM embeddings
WHERE "SourceKey" = 'ext:ai.visual'
GROUP BY "Kind","KindFamily","Modality","IsSemantic","Dim"
ORDER BY count(*) DESC;
```
(Table `embeddings`; columns are quoted PascalCase under Npgsql.)

### 2. Find the query text-embedding dimension
Either read it from the nsfw_ai_server response/logs for `/v4/encode/text` (kind family
`semantic.v1`), or use the cove API (authenticated, app at `http://127.0.0.1:5173`):
`POST /api/embeddings/search` with body
`{ "queryText":"test", "kindFamily":"semantic.v1", "kind":"visual.semantic.v1",
   "sourceKey":"ext:ai.visual", "modality":<Visual>, "isSemantic":true,
   "hostType":<Video>, "k":10 }`
This runs the *same* encode + KNN path (see `Cove.Api/Controllers/EmbeddingsController.Search`).
- To isolate encode-dim vs filters: also POST with `queryVector` = an array of zeros whose
  length equals the stored `Dim`, same filters. If that returns rows but `queryText` doesn't,
  it's a dim mismatch. If neither returns rows, a filter value (Kind/KindFamily/IsSemantic)
  doesn't match the stored rows.

### Interpretation
- **Stored `Dim` ≠ encode dim** → nsfw_ai_server model-pairing/config problem: the text
  encoder bound to `semantic.v1` must be the SAME CLIP-family model (same dim, same vector
  space) as the `semvisual` image embedder. Fix the server model config. NOT a code bug in
  these repos. (Check the `nsfw_ai_model_server` config.)
- **Stored `Kind`/`KindFamily`/`IsSemantic` differ** from `visual.semantic.v1` /
  `semantic.v1` / `true` → embeddings are stale (generated by older extension code).
  Re-run AI Visual on the library, or reconcile the mapping in `AiVisualPreparationService`.

---

## Secondary (not the current blocker, but worth fixing)
`AiVisualSemanticSearchService.cs` has `FastTextEncodeTimeoutMilliseconds = 400` and a
`TryEncodeWithRegisteredEncoderAsync` that **silently swallows** all encode failures
(OperationCanceled/HttpRequest/InvalidOperation/Json) and falls back to a local CPU encoder
(usually unavailable). The service takes no `ILogger`. Recommend: inject `ILogger`, log each
branch (encoder resolved/null, server-call attempted, exception), and widen/separate the
400 ms budget (it also wraps the settings-store read). This is why the failure was invisible
for so long.
