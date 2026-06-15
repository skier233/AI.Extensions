import React from "react";

const { useEffect, useState } = React;
const h = React.createElement;

const API_BASE = "/api/ext/ai-faces";
const JOBS_BASE = "/api/jobs";

async function request(url, options = {}) {
  const isFormData = typeof FormData !== "undefined" && options.body instanceof FormData;
  const response = await fetch(url, {
    ...options,
    headers: {
      ...(isFormData ? {} : { "Content-Type": "application/json" }),
      ...(options.headers || {}),
    },
  });

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }
  }

  if (!response.ok) {
    const detail = data && typeof data === "object" && data.detail
      ? data.detail
      : data && typeof data === "object" && data.error
        ? data.error
        : data && typeof data === "object" && data.message
          ? data.message
          : typeof data === "string" && data.length > 0
            ? data
            : response.statusText;
    throw new Error(detail || "Request failed.");
  }

  return data;
}

function api(path, options = {}) {
  return request(`${API_BASE}${path}`, options);
}

function getJob(jobId) {
  return request(`${JOBS_BASE}/${jobId}`);
}

function isTerminalJobStatus(status) {
  return status === "completed" || status === "failed" || status === "cancelled";
}

function formatDate(value) {
  if (!value) {
    return "Unknown";
  }

  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatProgress(value) {
  const scaled = value <= 1 ? value * 100 : value;
  return Math.max(0, Math.min(100, Math.round(scaled)));
}

function metricCard(label, value, detail) {
  return h("div", { className: "rounded-xl border border-border bg-card p-4 text-sm text-secondary" }, [
    h("div", { key: "label", className: "text-[11px] font-semibold uppercase tracking-wide text-muted" }, label),
    h("div", { key: "value", className: "mt-2 text-base font-semibold text-foreground" }, value),
    detail ? h("div", { key: "detail", className: "mt-1" }, detail) : null,
  ]);
}

function Notice({ tone, children }) {
  const toneClass = tone === "error"
    ? "border-rose-500/40 bg-rose-500/10 text-rose-100"
    : "border-emerald-500/40 bg-emerald-500/10 text-emerald-100";

  return h("div", { className: `rounded-xl border p-4 text-sm ${toneClass}` }, children);
}

function ActionButton({ label, busyLabel, pending, onClick, tone = "default" }) {
  const toneClass = tone === "accent"
    ? "bg-accent text-white hover:bg-accent-hover"
    : "border border-border bg-card text-foreground hover:border-accent";

  return h(
    "button",
    {
      type: "button",
      onClick,
      disabled: pending,
      className: `inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50 ${toneClass}`,
    },
    pending ? busyLabel : label,
  );
}

const DEFAULT_FACE_SETTINGS = {
  identityMatchThreshold: 0.5,
  identityAmbiguityMargin: 0.05,
  minimumVideoFacePresenceSeconds: 8,
  updateExistingPerformersFromMetadataServers: true,
};

const FACE_SETTINGS_FIELDS = [
  {
    key: "identityMatchThreshold",
    label: "Existing cluster match threshold",
    hint: "Higher values create new provisional faces more aggressively instead of reusing an existing cluster.",
    step: "0.01",
  },
  {
    key: "identityAmbiguityMargin",
    label: "Ambiguity margin",
    hint: "Require the best candidate to clearly beat the runner-up before attaching to an existing face.",
    step: "0.01",
  },
  {
    key: "minimumVideoFacePresenceSeconds",
    label: "Minimum video presence (seconds)",
    hint: "A face must have at least this much screen time in a video to be marked present, filtering out brief mis-attributions and intro-clip cameos. Capped at half the video length for short clips; images are never gated. Set to 0 to disable.",
    step: "1",
  },
];

function readFaceSetting(settings, camelKey, pascalKey) {
  if (settings && settings[camelKey] != null) {
    return settings[camelKey];
  }

  if (settings && settings[pascalKey] != null) {
    return settings[pascalKey];
  }

  return DEFAULT_FACE_SETTINGS[camelKey];
}

function readFaceBoolSetting(settings, camelKey, pascalKey) {
  if (settings && settings[camelKey] != null) {
    return settings[camelKey] === true;
  }

  if (settings && settings[pascalKey] != null) {
    return settings[pascalKey] === true;
  }

  return DEFAULT_FACE_SETTINGS[camelKey];
}

function toFaceSettingsDraft(settings) {
  return {
    identityMatchThreshold: String(readFaceSetting(settings, "identityMatchThreshold", "IdentityMatchThreshold")),
    identityAmbiguityMargin: String(readFaceSetting(settings, "identityAmbiguityMargin", "IdentityAmbiguityMargin")),
    minimumVideoFacePresenceSeconds: String(readFaceSetting(settings, "minimumVideoFacePresenceSeconds", "MinimumVideoFacePresenceSeconds")),
    updateExistingPerformersFromMetadataServers: readFaceBoolSetting(
      settings,
      "updateExistingPerformersFromMetadataServers",
      "UpdateExistingPerformersFromMetadataServers",
    ),
  };
}

function parseFaceSetting(value, fallback) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function SettingField({ label, hint, value, step, onInput, onCommit }) {
  return h("label", { className: "space-y-2 rounded-xl border border-border bg-card p-4" }, [
    h("div", { key: "label", className: "text-sm font-medium text-foreground" }, label),
    h("p", { key: "hint", className: "text-sm text-secondary" }, hint),
    h("input", {
      key: "input",
      type: "number",
      min: 0,
      max: 1,
      step,
      value,
      inputMode: "decimal",
      onInput,
      // Settings pages auto-save: commit when the field loses focus.
      onBlur: onCommit,
      className: "w-full rounded-xl border border-border bg-input px-3 py-2 text-sm text-foreground outline-none transition focus:border-accent",
    }),
  ]);
}

function ToggleField({ label, hint, checked, onToggle }) {
  return h("label", { className: "flex items-start gap-3 rounded-xl border border-border bg-card p-4" }, [
    h("input", {
      key: "input",
      type: "checkbox",
      checked,
      onChange: (event) => onToggle(event.target.checked),
      className: "mt-1 rounded border-border bg-input accent-accent",
    }),
    h("div", { key: "copy", className: "space-y-1" }, [
      h("div", { key: "label", className: "text-sm font-medium text-foreground" }, label),
      h("p", { key: "hint", className: "text-sm text-secondary" }, hint),
    ]),
  ]);
}

function AiFacesSettingsPanel() {
  const [packs, setPacks] = useState([]);
  const [statusLoaded, setStatusLoaded] = useState(false);
  const [statusError, setStatusError] = useState("");
  const [settings, setSettings] = useState(() => toFaceSettingsDraft());
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [settingsError, setSettingsError] = useState("");
  const [settingsSaveState, setSettingsSaveState] = useState("idle");
  const [notice, setNotice] = useState("");
  const [job, setJob] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [clearingPackId, setClearingPackId] = useState(null);

  async function loadStatus() {
    setStatusError("");
    try {
      const nextPacks = await api("/reference/packs");
      setPacks(Array.isArray(nextPacks) ? nextPacks : []);
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : "Failed to load AI.Faces reference pack status.");
    } finally {
      setStatusLoaded(true);
    }
  }

  async function loadSettings() {
    setSettingsError("");
    try {
      const nextSettings = await api("/settings");
      setSettings(toFaceSettingsDraft(nextSettings));
    } catch (error) {
      setSettingsError(error instanceof Error ? error.message : "Failed to load AI.Faces clustering settings.");
    } finally {
      setSettingsLoaded(true);
    }
  }

  async function pollJob(jobId) {
    try {
      const nextJob = await getJob(jobId);
      setJob(nextJob);
      if (isTerminalJobStatus(nextJob.status)) {
        setUploading(false);
        if (nextJob.status === "completed") {
          setNotice("Reference pack imported.");
          await loadStatus();
        } else if (nextJob.error) {
          setStatusError(nextJob.error);
        }
      }
    } catch (error) {
      setUploading(false);
      setStatusError(error instanceof Error ? error.message : "Failed to load reference import job status.");
    }
  }

  useEffect(() => {
    void loadStatus();
    void loadSettings();
  }, []);

  useEffect(() => {
    if (!job || !job.id || isTerminalJobStatus(job.status)) {
      return undefined;
    }

    const handle = window.setTimeout(() => {
      void pollJob(job.id);
    }, 1500);

    return () => window.clearTimeout(handle);
  }, [job]);

  async function handleFileChange(event) {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file) {
      return;
    }

    setUploading(true);
    setStatusError("");
    setNotice("");
    setJob(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const result = await api("/reference/import", { method: "POST", body: formData });
      if (result && result.jobId) {
        await pollJob(result.jobId);
      } else {
        setUploading(false);
        await loadStatus();
      }
    } catch (error) {
      setUploading(false);
      setStatusError(error instanceof Error ? error.message : "Failed to import the selected .saie pack.");
    }
  }

  async function handleRemovePack(pack) {
    const label = pack && pack.packId ? pack.packId : "this";
    if (!window.confirm(`Remove the "${label}" reference pack and clear its cached suggestion source?`)) {
      return;
    }

    setClearingPackId(pack.packId);
    setStatusError("");
    setNotice("");

    try {
      await api(`/reference?packId=${encodeURIComponent(pack.packId)}`, { method: "DELETE" });
      setJob(null);
      setNotice(`Reference pack "${label}" removed.`);
      await loadStatus();
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : "Failed to remove the reference pack.");
    } finally {
      setClearingPackId(null);
    }
  }

  function handleSettingInput(key, value) {
    setSettings((current) => ({
      ...current,
      [key]: value,
    }));
  }

  // The boolean toggle commits immediately on change. We thread the new value through explicitly because
  // the setSettings update above has not flushed yet when we build the PUT payload.
  function handleToggleSetting(key, checked) {
    setSettings((current) => ({ ...current, [key]: checked }));
    void persistSettings({ ...settings, [key]: checked });
  }

  // Settings pages auto-save. Each field commits on blur; we PUT the full settings object built from
  // the latest draft (numbers fall back to defaults when a field is left blank/invalid).
  function commitSettings() {
    void persistSettings(settings);
  }

  async function persistSettings(draft) {
    if (!settingsLoaded) {
      return;
    }

    setSettingsSaveState("saving");
    setSettingsError("");

    const payload = {
      identityMatchThreshold: parseFaceSetting(draft.identityMatchThreshold, DEFAULT_FACE_SETTINGS.identityMatchThreshold),
      identityAmbiguityMargin: parseFaceSetting(draft.identityAmbiguityMargin, DEFAULT_FACE_SETTINGS.identityAmbiguityMargin),
      minimumVideoFacePresenceSeconds: parseFaceSetting(draft.minimumVideoFacePresenceSeconds, DEFAULT_FACE_SETTINGS.minimumVideoFacePresenceSeconds),
      updateExistingPerformersFromMetadataServers: draft.updateExistingPerformersFromMetadataServers === true,
    };

    try {
      const savedSettings = await api("/settings", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      setSettings(toFaceSettingsDraft(savedSettings || payload));
      setSettingsSaveState("saved");
    } catch (error) {
      setSettingsSaveState("error");
      setSettingsError(error instanceof Error ? error.message : "Failed to save AI.Faces clustering settings.");
    }
  }

  const packSection = !statusLoaded
    ? h("div", { className: "rounded-xl border border-border bg-card p-4 text-sm text-secondary" }, "Loading reference pack status...")
    : packs.length > 0
      ? h("div", { className: "space-y-3" },
          packs.map((pack) => h("div", {
            key: pack.packId,
            className: "space-y-3 rounded-2xl border border-border bg-card p-4",
          }, [
            h("div", { key: "cards", className: "grid gap-3 md:grid-cols-2" }, [
              metricCard("Pack", pack.packId, `${(pack.performerCount || 0).toLocaleString()} identities - ${pack.embeddingDim} dims`),
              metricCard("Source", pack.sourceEndpoint || "Unknown source", `Imported ${formatDate(pack.importedAt)}`),
            ]),
            h("div", { key: "actions", className: "flex flex-wrap justify-end gap-2" }, [
              h(ActionButton, {
                key: "remove",
                label: "Remove pack",
                busyLabel: "Removing...",
                pending: clearingPackId === pack.packId,
                onClick: () => handleRemovePack(pack),
              }),
            ]),
          ])))
      : h("div", { className: "rounded-xl border border-dashed border-border px-4 py-6 text-sm text-secondary" }, "No .saie reference packs are currently imported. Upload packs from one or more sites to match faces against them.");

  const jobSection = job
    ? h("div", { className: "rounded-xl border border-border bg-card p-4 text-sm text-secondary" }, [
        h("div", { key: "summary" }, `Import job ${job.id} is ${job.status} at ${formatProgress(job.progress)}%.`),
        job.subTask ? h("div", { key: "task", className: "mt-1" }, job.subTask) : null,
        job.error ? h("div", { key: "error", className: "mt-2 text-rose-200" }, job.error) : null,
      ])
    : null;

  const settingsSection = !settingsLoaded
    ? h("div", { className: "rounded-xl border border-border bg-card p-4 text-sm text-secondary" }, "Loading clustering settings...")
    : h("section", { className: "space-y-4 rounded-2xl border border-border bg-card/50 p-4" }, [
        h("div", { key: "intro", className: "flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between" }, [
          h("div", { key: "copy" }, [
            h("div", { key: "title", className: "text-sm font-semibold uppercase tracking-wide text-muted" }, "Face Clustering"),
            h("p", { key: "body", className: "mt-1 text-sm text-secondary" }, "Tune how aggressively AI.Faces reuses an existing face cluster versus creating a new provisional face. Frontal, sharper samples also drive cover and centroid selection. Changes save automatically."),
          ]),
          h("div", {
            key: "savestate",
            className: "text-xs font-medium text-muted",
          }, settingsSaveState === "saving" ? "Saving..." : settingsSaveState === "saved" ? "Saved" : settingsSaveState === "error" ? "Save failed" : ""),
        ]),
        settingsError ? h(Notice, { key: "error", tone: "error" }, settingsError) : null,
        h("div", { key: "grid", className: "grid gap-3 md:grid-cols-2" },
          FACE_SETTINGS_FIELDS.map((field) => h(SettingField, {
            key: field.key,
            label: field.label,
            hint: field.hint,
            value: settings[field.key],
            step: field.step,
            onInput: (event) => handleSettingInput(field.key, event.target.value),
            onCommit: commitSettings,
          }))),
        h(ToggleField, {
          key: "updateExisting",
          label: "Update existing performers from metadata servers",
          hint: "When accepting a reference (metadata-server) face match for a performer that already exists locally, also scrape that performer from the originating server to refresh their image, bio, and aliases. The remote id is linked either way.",
          checked: settings.updateExistingPerformersFromMetadataServers === true,
          onToggle: (checked) => handleToggleSetting("updateExistingPerformersFromMetadataServers", checked),
        }),
      ]);

  return h("div", { className: "ai-faces-settings-panel space-y-5" }, [
    notice ? h(Notice, { key: "notice", tone: "success" }, notice) : null,
    h("section", { key: "reference", className: "space-y-4" }, [
      h("div", { key: "intro", className: "flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between" }, [
        h("div", { key: "copy" }, [
          h("div", { key: "title", className: "text-sm font-semibold uppercase tracking-wide text-muted" }, "Reference Face DB"),
          h("p", { key: "body", className: "mt-1 text-sm text-secondary" }, "Import a .saie archive so AI.Faces can suggest performers from an external face reference database."),
        ]),
        h("label", { key: "upload", className: "inline-flex cursor-pointer items-center gap-2 rounded-xl border border-border bg-card px-3 py-2 text-sm font-medium text-foreground transition hover:border-accent" }, [
          h("span", { key: "label" }, uploading ? "Uploading..." : "Upload .saie pack"),
          h("input", {
            key: "input",
            type: "file",
            accept: ".saie",
            className: "hidden",
            disabled: uploading,
            onChange: handleFileChange,
          }),
        ]),
      ]),
      jobSection,
      statusError ? h(Notice, { key: "error", tone: "error" }, statusError) : null,
      packSection,
    ]),
    h("div", { key: "settings" }, settingsSection),
  ]);
}

export default {
  components: {
    AiFacesSettingsPanel,
  },
};
