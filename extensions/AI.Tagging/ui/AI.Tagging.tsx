import React from "react";

const { useEffect, useState } = React;
const h = React.createElement;

const API_BASE = "/api/ext/ai-tagging";

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
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
      : data && typeof data === "object" && data.message
        ? data.message
        : typeof data === "string" && data.length > 0
          ? data
          : response.statusText;
    throw new Error(detail || "Request failed.");
  }

  return data;
}

function copySettings(settings) {
  return {
    tagNameOverrides: (settings?.tagNameOverrides || []).map((override) => ({
      sourceTagName: override.sourceTagName || "",
      targetTagName: override.targetTagName || "",
    })),
  };
}

function cleanSettings(settings) {
  const seen = new Set();
  return {
    tagNameOverrides: (settings.tagNameOverrides || [])
      .map((override) => ({
        sourceTagName: (override.sourceTagName || "").trim(),
        targetTagName: (override.targetTagName || "").trim(),
      }))
      .filter((override) => override.sourceTagName && override.targetTagName)
      .filter((override) => {
        const key = override.sourceTagName.toLocaleLowerCase();
        if (seen.has(key)) {
          return false;
        }
        seen.add(key);
        return true;
      }),
  };
}

function AiTaggingSettingsPanel() {
  const [settings, setSettings] = useState(copySettings(null));
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const loaded = await api("/settings");
        if (!cancelled) {
          setSettings(copySettings(loaded));
        }
      } catch (error) {
        if (!cancelled) {
          setMessage(error.message || "Failed to load AI Tagging settings.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  function patchOverride(index, field, value) {
    setSettings((current) => ({
      ...current,
      tagNameOverrides: current.tagNameOverrides.map((override, overrideIndex) => (
        overrideIndex === index ? { ...override, [field]: value } : override
      )),
    }));
  }

  function addOverride() {
    setSettings((current) => ({
      ...current,
      tagNameOverrides: [
        ...current.tagNameOverrides,
        { sourceTagName: "", targetTagName: "" },
      ],
    }));
  }

  function removeOverride(index) {
    setSettings((current) => ({
      ...current,
      tagNameOverrides: current.tagNameOverrides.filter((_, overrideIndex) => overrideIndex !== index),
    }));
  }

  async function saveSettings() {
    setSaving(true);
    setMessage("");
    try {
      const saved = await api("/settings", {
        method: "PUT",
        body: JSON.stringify(cleanSettings(settings)),
      });
      setSettings(copySettings(saved));
      setMessage("Settings saved.");
    } catch (error) {
      setMessage(error.message || "Failed to save AI Tagging settings.");
    } finally {
      setSaving(false);
    }
  }

  if (loading) {
    return h("p", { className: "text-sm text-muted-foreground" }, "Loading AI Tagging settings...");
  }

  const rows = settings.tagNameOverrides || [];

  return h("div", { className: "ai-tagging-settings-panel space-y-3" }, [
    h("div", { className: "flex items-center justify-between gap-3", key: "header" }, [
      h("div", null, [
        h("h3", { className: "text-base font-semibold text-foreground" }, "Tag name overrides"),
        h("p", { className: "text-sm text-muted-foreground" }, "Map server labels to the Cove tag names that should be created or reused."),
      ]),
      h("button", {
        className: "rounded-md border border-border bg-surface px-3 py-1.5 text-sm font-medium text-foreground hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60",
        type: "button",
        onClick: addOverride,
        disabled: saving,
      }, "Add"),
    ]),
    rows.length === 0
      ? h("p", { className: "rounded-md border border-dashed border-border bg-muted/30 p-3 text-sm text-muted-foreground", key: "empty" }, "No tag name overrides configured.")
      : h("div", { className: "overflow-x-auto rounded-md border border-border bg-surface", key: "rows" }, [
          h("table", { className: "min-w-full border-collapse text-sm" }, [
            h("thead", { key: "head", className: "bg-muted/40" }, [
              h("tr", { className: "text-left text-xs font-medium uppercase tracking-wide text-muted-foreground" }, [
                h("th", { className: "px-3 py-2 font-medium" }, "Original tag"),
                h("th", { className: "px-3 py-2 font-medium" }, "Override"),
                h("th", { className: "w-1 px-3 py-2 font-medium text-right" }, ""),
              ]),
            ]),
            h("tbody", { key: "body" }, rows.map((override, index) => h("tr", {
              className: "border-t border-border align-middle first:border-t-0",
              key: index,
            }, [
              h("td", { className: "px-3 py-2" }, [
                h("input", {
                  className: "w-full min-w-[12rem] rounded-md border border-border bg-input px-2 py-1.5 text-sm text-foreground outline-none focus:border-accent",
                  value: override.sourceTagName,
                  onChange: (event) => patchOverride(index, "sourceTagName", event.target.value),
                  placeholder: "Server tag",
                  "aria-label": "Original tag",
                }),
              ]),
              h("td", { className: "px-3 py-2" }, [
                h("input", {
                  className: "w-full min-w-[14rem] rounded-md border border-border bg-input px-2 py-1.5 text-sm text-foreground outline-none focus:border-accent",
                  value: override.targetTagName,
                  onChange: (event) => patchOverride(index, "targetTagName", event.target.value),
                  placeholder: "Override tag",
                  "aria-label": "Override tag",
                }),
              ]),
              h("td", { className: "px-3 py-2 text-right" }, [
                h("button", {
                  className: "rounded-md border border-border bg-surface px-2.5 py-1.5 text-sm font-medium text-foreground hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60",
                  type: "button",
                  onClick: () => removeOverride(index),
                  disabled: saving,
                }, "Delete"),
              ]),
            ]))),
          ]),
        ]),
    h("div", { className: "flex items-center justify-end gap-3", key: "actions" }, [
      message ? h("span", { className: "text-sm text-muted-foreground" }, message) : null,
      h("button", {
        className: "rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60",
        type: "button",
        disabled: saving,
        onClick: saveSettings,
      }, saving ? "Saving..." : "Save"),
    ]),
  ]);
}

export default {
  components: {
    AiTaggingSettingsPanel,
  },
};