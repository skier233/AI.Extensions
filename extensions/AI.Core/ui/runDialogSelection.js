// Which features the Run AI dialog selects on the user's behalf: when it opens, restores the last run's selection,
// changes media kind or clears the preset. A feature marked `selectedByDefault: false` (for example one that decodes
// every frame of every video, or whose forced rerun replaces results users edit by hand) is never selected this way,
// so it only runs when the user ticks it for that run. "Select all" and presets still select it.

export function isSelectedByDefault(feature) {
  return feature?.selectedByDefault !== false;
}

/** The features selected when nothing else says otherwise. */
export function defaultCapabilityIds(features) {
  return (features || []).filter(isSelectedByDefault).map((feature) => feature.capabilityId);
}

/** The remembered selection, limited to features still offered and never including one that is not selected by default. */
export function rememberedCapabilityIds(rememberedIds, features) {
  const restorable = new Set((features || []).filter(isSelectedByDefault).map((feature) => feature.capabilityId));
  return (Array.isArray(rememberedIds) ? rememberedIds : [])
    .filter((capabilityId) => typeof capabilityId === "string" && restorable.has(capabilityId));
}
