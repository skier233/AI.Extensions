# Cove AI Extensions

A family of AI add-ons for [Cove](https://yourcove.net) that read your library and add things you can actually use: automatic tags, facial recognition, search-by-meaning, and audio matching. The models run on a small AI server you host yourself, and Cove just connects to it.

Most of the models and their development are funded through Patreon, but **facial recognition is completely free** as well as one of the simple tagging models, so you can use real AI features without paying anything.

---

## What you get

| Extension | What it does | Tier |
| --- | --- | --- |
| **AI Core** | Connects Cove to your AI server and runs the work. Required by everything else. | Free |
| **AI Faces** | Recognizes the people in your videos and images, groups each person together, and suggests which performer they are. | **Free** |
| **AI Tagging** | Automatically tags your content, with timing so tags can show as segments on the player bar. | Free model included; main models on Patreon |
| **AI Visual** | Search your library by meaning ("sunset on a beach") and find look-alike videos and images. | Patreon |
| **AI Audio** | Finds other videos that sound like the one you're watching. | Patreon |

**Free vs. paid, plainly:** AI Faces is free. AI Tagging ships with one free model; its main tagging models, plus AI Visual and AI Audio, are part of the Patreon that funds ongoing model development. The extensions themselves are free to install. The extensions as well as the AI server are open source.

---

## Getting started

1. **Set up the AI server.** The extensions talk to a self-hosted model server. Follow the quickstart to install it and download models:
   - 👉 **[AI server quickstart guide](https://github.com/skier233/nsfw_ai_model_server/wiki/NSFW-AI-Tagging-Quickstart-Guide)**
2. **Install the extensions in Cove.** Open **Settings → Extensions → Discover** and install **AI Tagging** or **AI Faces** first. Then add additional capabilities you want — AI Faces, AI Tagging, AI Visual, AI Audio.
3. **Connect AI Core to your server.** Go to **Settings → Extensions → AI**, enter your server URL, and confirm it shows **Server reachable**. Load the models you plan to use.
4. **Run it.** Select one or more videos or images, use **Run AI**, and pick the capabilities to run. Start small and review the results before doing your whole library.

Every extension also ships an in-app guide. After installing one, open the **?** icon in the top right of Cove for step-by-step screenshots and info.

---

## A quick tour

- **Faces** — After a run, open the **Faces** page. Each tile is one person; link them to performers with one click, or **Compare** side by side first. Import a free **SAIE reference pack** to have known performers suggested automatically — [browse face packs](https://github.com/skier233/nsfw_ai_model_server/wiki/AI-Models#face-packs).
- **Tagging** — Tags come with timing, so you can filter by how long a tag lasts and show tags as colored segments on the player bar. Tune noisy tags per-tag, or rename/merge AI tags to match your own.
- **Visual** — On the Videos or Images page, switch the search box from **Text** to **Visual** and describe what you want. Open any item's **Similar** tab to find look-alikes.
- **Audio** — Open a video's **Audio Similar** tab to see other videos ranked by how much they sound alike.

---

## Links

- 🧠 **AI server (setup & models):** https://github.com/skier233/nsfw_ai_model_server
- 🚀 **Quickstart guide:** https://github.com/skier233/nsfw_ai_model_server/wiki/NSFW-AI-Tagging-Quickstart-Guide
- 🧩 **This repo:** https://github.com/skier233/AI.Extensions
- 📖 **Cove docs:** https://yourcove.net/docs/

---

<details>
<summary>Building from source (developers)</summary>

The repo prefers a sibling Cove checkout at `../cove` and references `Cove.Sdk` from source when present.

```powershell
# build the .NET extensions
dotnet build AI.Extensions.slnx

# rebuild the extension UI bundles after editing anything under each extension's ui/
npm install
npm run build:ui
```

Don't hand-edit `dist/bundle.js` or `dist/bundle.css` — they're checked in as generated assets. To stage builds into Cove's local runtime folder for testing:

```powershell
./scripts/stage-local-extensions.ps1
```

This copies outputs into `%LOCALAPPDATA%\cove\extensions\<extension-id>` so the Cove host discovers them on startup.

</details>
