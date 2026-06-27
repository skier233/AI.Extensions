---
name: ai-extensions-repo-context
description: AI.Extensions repo context, source map, nsfw_ai_server relationship, and multi-repo paths. Use when working on Cove AI extensions or AI-related Cove integration.
---

# AI.Extensions Repo Context

## Purpose

`AI.Extensions` is the extension-side half of the Cove AI stack. It provides Cove extensions for AI orchestration, tagging, faces, visual semantic search, audio analysis, and a bundle extension that installs the family together.

## Important Local Path

`C:\Users\tyler\source\repos\AI.Extensions`

## Code Map

- `extensions/AI.Core/` owns nsfw_ai_server v4 connectivity, settings, model lifecycle calls, orchestration, run planning, run queues, path mapping, AI page/settings UI, and cross-extension dispatch.
- `extensions/AI.Core/NsfwAiServerClient.cs` is the important client boundary to the Python model server.
- `extensions/AI.Tagging/` contributes tagging capability claims, tagging settings, persistence, and preparation services.
- `extensions/AI.Faces/` contributes face detection/identity/reference-pack features, suggestion decisions, clustering/reconciliation, quality scoring, and face-specific persistence.
- `extensions/AI.Visual/` contributes visual preparation, visual persistence, semantic search, local text encoding, and CLIP tokenizer code.
- `extensions/AI.Audio/` contributes audio preparation, similarity, persistence, and audio capability hooks.
- `extensions/AI.Full/` is a manifest/code bundle that activates the whole AI family.
- `extensions/AI.Extensions.Abstractions/` contains shared contracts used by the AI extension family without coupling implementation assemblies.
- `extensions/catalog.json` describes packages/release metadata for the multi-extension repo.
- `tests/AI.Extensions.Tests/` contains unit tests for orchestration, persistence, path mapping, preparation services, face logic, and server-facing behavior.
- `scripts/build-extension-ui.mjs` builds extension UI assets from each extension's `ui/` folder into checked-in `dist/` runtime assets.
- `scripts/stage-local-extensions.ps1` stages built extensions into `%LOCALAPPDATA%\cove\extensions\<extension-id>` for Cove runtime testing.

## Related Repos On This PC

- `C:\Users\tyler\source\repos\cove` is the Cove host and SDK source. Modify it with AI.Extensions when host contracts, extension UI actions, AI hooks, semantic search registration, face suggestion hooks, plugin loading, or `Cove.Sdk` APIs change.
- `C:\Coding\Testing\PyTorch\MultiLabelClassification_Patreon\github\nsfw_ai_model_server` is the Python server that `AI.Core` calls. Modify it with AI.Extensions when endpoints, request/response JSON, model capabilities, result formats, or lifecycle behavior change.
- `C:\Users\tyler\source\repos\officialextensionregistry` contains published metadata for AI extension entries. Modify it when AI extension versions, source manifests, dependencies, release ZIP URLs, or registry-visible metadata change.
- `C:\Users\tyler\source\repos\communitydownloaders` and `C:\Users\tyler\source\repos\communityscrapers` are not AI-specific, but may need updates if shared extension packaging, manifest validation, registry conventions, or Cove SDK extension contracts change.

## Common Commands

- Build all .NET projects: `dotnet build AI.Extensions.slnx`.
- Test: `dotnet test AI.Extensions.slnx`.
- Build extension UI assets: `npm install` if needed, then `npm run build:ui`.
- Stage locally for Cove: `./scripts/stage-local-extensions.ps1`.

## Coordination Notes

- This repo prefers a sibling Cove checkout at `..\cove` and references Cove source contracts when present.
- Edit extension UI source under each extension's `ui/` folder, then regenerate `dist/`; do not hand-edit checked-in `dist/bundle.js` or `dist/bundle.css` unless the build output itself is the intended artifact.
- Treat `NsfwAiServerClient.cs` and server response DTOs as the API seam with `nsfw_ai_model_server`.
- If an AI extension needs new host behavior, update Cove first or in the same change set, then update this repo's contracts/tests.
