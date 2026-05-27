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
  minimumPoseQuality: 0.78,
  minimumImageQuality: 0.22,
  identityMatchThreshold: 0.54,
  identityAmbiguityMargin: 0.03,
};

const FACE_SETTINGS_FIELDS = [
  {
    key: "minimumPoseQuality",
    label: "Minimum pose quality",
    hint: "Reject profile-heavy samples from anchor, centroid, and cover selection.",
    step: "0.01",
  },
  {
    key: "minimumImageQuality",
    label: "Minimum image quality",
    hint: "Reject blurry face crops before they influence clustering or cover choice.",
    step: "0.01",
  },
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

function toFaceSettingsDraft(settings) {
  return {
    minimumPoseQuality: String(readFaceSetting(settings, "minimumPoseQuality", "MinimumPoseQuality")),
    minimumImageQuality: String(readFaceSetting(settings, "minimumImageQuality", "MinimumImageQuality")),
    identityMatchThreshold: String(readFaceSetting(settings, "identityMatchThreshold", "IdentityMatchThreshold")),
    identityAmbiguityMargin: String(readFaceSetting(settings, "identityAmbiguityMargin", "IdentityAmbiguityMargin")),
  };
}

function parseFaceSetting(value, fallback) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function SettingField({ label, hint, value, step, onInput }) {
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
      className: "w-full rounded-xl border border-border bg-input px-3 py-2 text-sm text-foreground outline-none transition focus:border-accent",
    }),
  ]);
}

function AiFacesSettingsPanel() {
  const [status, setStatus] = useState(null);
  const [statusLoaded, setStatusLoaded] = useState(false);
  const [statusError, setStatusError] = useState("");
  const [settings, setSettings] = useState(() => toFaceSettingsDraft());
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  const [settingsError, setSettingsError] = useState("");
  const [savingSettings, setSavingSettings] = useState(false);
  const [notice, setNotice] = useState("");
  const [job, setJob] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [clearing, setClearing] = useState(false);

  async function loadStatus() {
    setStatusError("");
    try {
      const nextStatus = await api("/reference/status");
      setStatus(nextStatus);
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

  async function handleClear() {
    if (!window.confirm("Remove the imported AI.Faces reference pack and clear its cached suggestion source?")) {
      return;
    }

    setClearing(true);
    setStatusError("");
    setNotice("");

    try {
      await api("/reference", { method: "DELETE" });
      setJob(null);
      setStatus(null);
      setNotice("Reference pack cleared.");
      await loadStatus();
    } catch (error) {
      setStatusError(error instanceof Error ? error.message : "Failed to clear the imported reference pack.");
    } finally {
      setClearing(false);
    }
  }

  function handleSettingInput(key, value) {
    setSettings((current) => ({
      ...current,
      [key]: value,
    }));
  }

  async function handleSaveSettings() {
    setSavingSettings(true);
    setSettingsError("");
    setNotice("");

    const payload = {
      minimumPoseQuality: parseFaceSetting(settings.minimumPoseQuality, DEFAULT_FACE_SETTINGS.minimumPoseQuality),
      minimumImageQuality: parseFaceSetting(settings.minimumImageQuality, DEFAULT_FACE_SETTINGS.minimumImageQuality),
      identityMatchThreshold: parseFaceSetting(settings.identityMatchThreshold, DEFAULT_FACE_SETTINGS.identityMatchThreshold),
      identityAmbiguityMargin: parseFaceSetting(settings.identityAmbiguityMargin, DEFAULT_FACE_SETTINGS.identityAmbiguityMargin),
    };

    try {
      const savedSettings = await api("/settings", {
        method: "PUT",
        body: JSON.stringify(payload),
      });
      setSettings(toFaceSettingsDraft(savedSettings || payload));
      setNotice("AI.Faces clustering settings saved.");
    } catch (error) {
      setSettingsError(error instanceof Error ? error.message : "Failed to save AI.Faces clustering settings.");
    } finally {
      setSavingSettings(false);
    }
  }

  const packSection = !statusLoaded
    ? h("div", { className: "rounded-xl border border-border bg-card p-4 text-sm text-secondary" }, "Loading reference pack status...")
    : status
      ? h("div", { className: "space-y-4" }, [
          h("div", { key: "cards", className: "grid gap-3 md:grid-cols-2" }, [
            metricCard("Pack", status.packId, `${(status.performerCount || 0).toLocaleString()} identities - ${status.embeddingDim} dims`),
            metricCard("Source", status.sourceEndpoint || "Unknown source", `Imported ${formatDate(status.importedAt)}`),
          ]),
          h("div", { key: "actions", className: "flex flex-wrap gap-2" }, [
            h(ActionButton, {
              key: "clear",
              label: "Clear imported reference pack",
              busyLabel: "Clearing...",
              pending: clearing,
              onClick: handleClear,
            }),
          ]),
        ])
      : h("div", { className: "rounded-xl border border-dashed border-border px-4 py-6 text-sm text-secondary" }, "No .saie reference pack is currently imported.");

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
            h("p", { key: "body", className: "mt-1 text-sm text-secondary" }, "Tune how aggressively AI.Faces reuses an existing face cluster versus creating a new provisional face. Frontal, sharper samples also drive cover and centroid selection."),
          ]),
          h(ActionButton, {
            key: "save",
            label: "Save clustering settings",
            busyLabel: "Saving...",
            pending: savingSettings,
            onClick: handleSaveSettings,
            tone: "accent",
          }),
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
          }))),
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
