import React from "react";
import ReactDOMClient from "react-dom/client";

const { useEffect, useRef, useState } = React;
const { createRoot } = ReactDOMClient;
const h = React.createElement;

const API_BASE = "/api/ext/ai-core";
const RUN_DIALOG_ROOT_ID = "cove-ai-core-run-dialog-root";
const RUN_DIALOG_STORAGE_PREFIX = "cove.aiCore.runDialog.v1";
const LOAD_POLICIES = [
  { value: "use_loaded", label: "Use loaded" },
  { value: "load_if_cheap", label: "Load if cheap" },
  { value: "load_or_fail", label: "Load or fail" },
];
const MEDIA_KINDS = [
  { value: "video", label: "Video" },
  { value: "image", label: "Image" },
  { value: "audio", label: "Audio" },
];

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

function splitList(raw) {
  return (raw || "")
    .split(/\r?\n|,/)
    .map((value) => value.trim())
    .filter(Boolean);
}

function joinList(values) {
  return (values || []).join(", ");
}

function formatRegionModels(regionModels) {
  return Object.entries(regionModels || {})
    .map(([detector, models]) => `${detector}: ${(models || []).join(", ")}`)
    .join("\n");
}

function parseRegionModels(raw) {
  const result = {};
  (raw || "").split(/\r?\n/).forEach((line) => {
    const [detector, ...rest] = line.split(":");
    const detectorName = (detector || "").trim();
    const models = splitList(rest.join(":"));
    if (detectorName && models.length > 0) {
      result[detectorName] = models;
    }
  });
  return result;
}

function statusTone(status) {
  return status === "reachable" ? "success" : status === "unreachable" ? "warning" : "muted";
}

function prettyModelGroup(model) {
  if (model.capabilities && model.capabilities.length > 0) {
    return model.capabilities[0];
  }
  if (model.type) {
    return model.type;
  }
  return "other";
}

function groupModels(models) {
  const groups = new Map();
  (models || []).forEach((model) => {
    const groupKey = prettyModelGroup(model);
    if (!groups.has(groupKey)) {
      groups.set(groupKey, []);
    }
    groups.get(groupKey).push(model);
  });

  return Array.from(groups.entries())
    .sort((left, right) => left[0].localeCompare(right[0]))
    .map(([groupKey, items]) => ({
      groupKey,
      items: items.sort((left, right) => (left.name || left.configName).localeCompare(right.name || right.configName)),
    }));
}

function collectClaims(capabilities, mediaKind) {
  return (capabilities || [])
    .flatMap((descriptor) => (descriptor.claims || []).map((claim) => ({ ...claim, extensionId: descriptor.extensionId, extensionLabel: descriptor.displayName })))
    .filter((claim) => claim.mediaKind === mediaKind);
}

function collectCapabilityFeatures(capabilities, mediaKind, customPipelines = [], catalog = null) {
  const claims = collectClaims(capabilities, mediaKind);
  const claimMap = new Map(claims.map((claim) => [claim.claimId, claim]));
  const features = [];

  (capabilities || []).forEach((descriptor) => {
    (descriptor.capabilities || []).forEach((feature) => {
      const claimIds = (feature.claimIds || []).filter((claimId) => claimMap.has(claimId));
      if (claimIds.length === 0) {
        return;
      }

      features.push({
        ...feature,
        extensionId: descriptor.extensionId,
        extensionLabel: descriptor.displayName,
        claimIds,
        customPipeline: false,
      });
    });
  });

  (customPipelines || []).forEach((pipeline) => {
    if (pipeline.mediaKind !== mediaKind) {
      return;
    }

    features.push({
      capabilityId: pipeline.capabilityId || `custom.${pipeline.pipelineName}`,
      displayName: pipeline.displayName || pipeline.pipelineName,
      description: pipeline.description || "Custom AI pipeline",
      claimIds: pipeline.claimIds || [],
      includedCapabilityIds: pipeline.capabilityIds || [],
      extensionId: "custom",
      extensionLabel: "Custom",
      customPipeline: true,
      pipelineName: pipeline.pipelineName,
    });
  });

  const seen = new Set();
  const distinctFeatures = features.filter((feature) => {
    const id = (feature.capabilityId || "").toLowerCase();
    if (!id || seen.has(id)) {
      return false;
    }
    seen.add(id);
    return true;
  });

  if (!catalog || !Array.isArray(catalog.models)) {
    return [];
  }

  return distinctFeatures.filter((feature) => featureHasActiveServingModels(catalog, distinctFeatures, claims, feature));
}

function collectSelectedClaimIds(features, claims, selectedCapabilityIds) {
  const selected = new Set(selectedCapabilityIds || []);
  const claimIds = new Set();
  const featureById = new Map((features || []).map((feature) => [feature.capabilityId, feature]));

  (features || []).forEach((feature) => {
    if (!selected.has(feature.capabilityId)) {
      return;
    }

    (feature.claimIds || []).forEach((claimId) => claimIds.add(claimId));
    (feature.includedCapabilityIds || []).forEach((includedId) => {
      const included = featureById.get(includedId);
      (included?.claimIds || []).forEach((claimId) => claimIds.add(claimId));
    });
  });

  return Array.from(claimIds);
}

function normalizeCapabilityBinding(binding) {
  return {
    capabilityId: (binding?.capabilityId || "").trim().toLowerCase(),
    slotId: (binding?.slotId || "").trim().toLowerCase(),
    scope: (binding?.scope || "").trim().toLowerCase() || null,
    category: (binding?.category || "").trim() || null,
    model: (binding?.model || "").trim(),
  };
}

function createModelBindingKey(capabilityId, slotId, scope = null, category = null) {
  return `${(capabilityId || "").trim().toLowerCase()}\u001F${(slotId || "").trim().toLowerCase()}\u001F${(scope || "").trim().toLowerCase()}\u001F${(category || "").trim()}`;
}

function findModelBinding(bindings, capabilityId, slotId, scope = null, category = null) {
  const key = createModelBindingKey(capabilityId, slotId, scope, category);
  return (bindings || [])
    .map(normalizeCapabilityBinding)
    .find((binding) => createModelBindingKey(binding.capabilityId, binding.slotId, binding.scope, binding.category) === key) || null;
}

function upsertModelBinding(bindings, capabilityId, slotId, scope, category, model) {
  const key = createModelBindingKey(capabilityId, slotId, scope, category);
  const next = (bindings || [])
    .map(normalizeCapabilityBinding)
    .filter((binding) => createModelBindingKey(binding.capabilityId, binding.slotId, binding.scope, binding.category) !== key);
  const trimmedModel = (model || "").trim();
  if (trimmedModel) {
    next.push({
      capabilityId: (capabilityId || "").trim().toLowerCase(),
      slotId: (slotId || "").trim().toLowerCase(),
      scope: scope || null,
      category: category || null,
      model: trimmedModel,
    });
  }
  return next;
}

function modelMatchesSlot(model, slot) {
  const requiredCapabilities = slot?.requiredCapabilities || [];
  const requiredScopes = slot?.requiredScopes || [];
  const requiredCategories = slot?.requiredCategories || [];
  const capabilities = model?.capabilities || [];
  const scopes = model?.supportedScopes || [];
  const categories = model?.categories || [];

  if (requiredCapabilities.length > 0 && !requiredCapabilities.some((capability) => capabilities.includes(capability))) {
    return false;
  }
  if (requiredScopes.length > 0 && scopes.length > 0 && !requiredScopes.some((scope) => scopes.includes(scope))) {
    return false;
  }
  if (requiredCategories.length > 0 && !requiredCategories.some((category) => categories.includes(category))) {
    return false;
  }
  return true;
}

function getSlotModelCandidates(catalog, slot, options = {}) {
  const matchesModelState = options.activeOnly ? isActiveCatalogModel : isLoadedCatalogModel;
  const models = (catalog?.models || [])
    .filter(matchesModelState)
    .filter((model) => modelMatchesSlot(model, slot))
    .map(createTaggingModelCandidate)
    .filter(Boolean);
  (slot?.defaultModels || []).forEach((modelName) => {
    if (!models.some((model) => model.configName.toLowerCase() === modelName.toLowerCase())) {
      const catalogModel = findCatalogModel(catalog, modelName);
      if (matchesModelState(catalogModel)) {
        models.push({
          configName: modelName,
          loaded: catalogModel?.loaded === true ? true : catalogModel?.loaded === false ? false : undefined,
          active: catalogModel?.active !== false,
          unavailable: false,
        });
      }
    }
  });
  return sortTaggingCandidates(models);
}

function modelMatchesClaim(model, claim) {
  const capabilities = model?.capabilities || [];
  const scopes = model?.supportedScopes || [];

  if (claim?.wantCapability && capabilities.length > 0 && !capabilities.includes(claim.wantCapability)) {
    return false;
  }

  if (claim?.wantScope && scopes.length > 0 && !scopes.includes(claim.wantScope)) {
    return false;
  }

  return true;
}

function claimHasActiveServingModel(catalog, features, claim) {
  if (!catalog || !Array.isArray(catalog.models) || !claim) {
    return false;
  }

  if (claim.wantCapability === "tagging") {
    return collectTaggingCategories(catalog, claim.mediaKind).length > 0;
  }

  const slot = findModelBindingSlotForClaim(features, claim);
  if (slot) {
    return getSlotModelCandidates(catalog, slot, { activeOnly: true }).length > 0;
  }

  if (Array.isArray(claim.preferredModels) && claim.preferredModels.length > 0) {
    return claim.preferredModels.some((modelName) => isActiveCatalogModel(findCatalogModel(catalog, modelName)));
  }

  return (catalog.models || []).some((model) => isActiveCatalogModel(model) && modelMatchesClaim(model, claim));
}

function featureHasActiveServingModels(catalog, features, claims, feature, visited = new Set()) {
  const capabilityId = (feature?.capabilityId || "").trim().toLowerCase();
  if (!feature || !capabilityId || visited.has(capabilityId)) {
    return false;
  }

  const nextVisited = new Set(visited);
  nextVisited.add(capabilityId);
  const claimById = new Map((claims || []).map((claim) => [claim.claimId, claim]));
  const directClaims = (feature.claimIds || []).map((claimId) => claimById.get(claimId)).filter(Boolean);
  const includedIds = new Set((feature.includedCapabilityIds || []).map((id) => (id || "").trim().toLowerCase()).filter(Boolean));
  const includedFeatures = (features || []).filter((candidate) => includedIds.has((candidate.capabilityId || "").trim().toLowerCase()));

  const directClaimsServed = directClaims.length === 0
    ? true
    : directClaims.every((claim) => claimHasActiveServingModel(catalog, features, claim));
  const includedFeaturesServed = includedIds.size === 0
    ? true
    : includedFeatures.length > 0 && includedFeatures.every((included) => featureHasActiveServingModels(catalog, features, claims, included, nextVisited));

  return (directClaims.length > 0 || includedIds.size > 0 || feature.customPipeline) && directClaimsServed && includedFeaturesServed;
}

function findModelBindingSlotForClaim(features, claim) {
  if (!claim?.capabilityId || !claim?.modelBindingSlotId) {
    return null;
  }

  return (features || [])
    .filter((feature) => feature.capabilityId === claim.capabilityId && (feature.claimIds || []).includes(claim.claimId))
    .flatMap((feature) => feature.modelBindingSlots || [])
    .find((slot) => slot.slotId === claim.modelBindingSlotId) || null;
}

function collectAutoSlotModels(catalog, features, claim, options = {}) {
  const slot = findModelBindingSlotForClaim(features, claim);
  if (!slot) {
    return { hasSlot: false, models: [] };
  }

  const models = getSlotModelCandidates(catalog, slot, options);
  return {
    hasSlot: true,
    models: slot.allowMultiple ? models : models.slice(0, 1),
  };
}

function getModelBindingRows(capabilities, catalog, settings) {
  return (capabilities || []).flatMap((descriptor) => (descriptor.capabilities || []).flatMap((feature) => (
    (feature.modelBindingSlots || [])
      .filter((slot) => !slot.categoryScoped)
      .map((slot) => {
        const selected = findModelBinding(settings.capabilityModelBindings, feature.capabilityId, slot.slotId);
        const models = getSlotModelCandidates(catalog, slot);
        const selectedModelAvailable = !selected?.model || models.some((model) => model.configName.toLowerCase() === selected.model.toLowerCase());
        const defaultModel = (slot.defaultModels || [])
          .map((modelName) => models.find((model) => model.configName.toLowerCase() === modelName.toLowerCase()))
          .filter(Boolean)[0]?.configName || models[0]?.configName || "";
        return {
          key: createModelBindingKey(feature.capabilityId, slot.slotId),
          extensionLabel: descriptor.displayName,
          capabilityId: feature.capabilityId,
          capabilityLabel: feature.displayName,
          slotId: slot.slotId,
          slotLabel: slot.displayName,
          description: slot.description || feature.description,
          selectedModel: selectedModelAvailable ? selected?.model || "" : "",
          savedModelUnavailable: !!selected?.model && !selectedModelAvailable,
          defaultModel,
          models,
        };
      })
  ))).sort((left, right) => left.capabilityLabel.localeCompare(right.capabilityLabel) || left.slotLabel.localeCompare(right.slotLabel));
}

function isTaggingModel(model) {
  return Array.isArray(model?.capabilities) && model.capabilities.includes("tagging");
}

function normalizeTaggingScope(scope) {
  return (scope || "").trim().toLowerCase();
}

function normalizeTaggingCategory(category) {
  return (category || "").trim();
}

function normalizeTaggingCategoryKey(category) {
  return normalizeTaggingCategory(category).toLowerCase();
}

function createTaggingPreferenceKey(scope, category) {
  return `${normalizeTaggingScope(scope)}\u001F${normalizeTaggingCategoryKey(category)}`;
}

function getTaggingScopes(model) {
  const uniqueScopes = new Map();
  const scopes = Array.isArray(model?.supportedScopes) && model.supportedScopes.length > 0
    ? model.supportedScopes
    : ["asset", "frame", "region"];

  scopes.forEach((scope) => {
    const normalized = normalizeTaggingScope(scope);
    if (normalized && !uniqueScopes.has(normalized)) {
      uniqueScopes.set(normalized, normalized);
    }
  });

  return Array.from(uniqueScopes.values());
}

function getTaggingCategoriesForModel(model) {
  const categories = new Map();

  (Array.isArray(model?.categories) ? model.categories : []).forEach((category) => {
    const normalized = normalizeTaggingCategory(category);
    const key = normalizeTaggingCategoryKey(category);
    if (normalized && key && !categories.has(key)) {
      categories.set(key, normalized);
    }
  });

  return Array.from(categories.values());
}

function getModelConfigName(model) {
  return model?.configName || model?.name || "";
}

function findCatalogModel(catalog, modelName) {
  const normalized = (modelName || "").trim().toLowerCase();
  if (!normalized || !catalog || !Array.isArray(catalog.models)) {
    return null;
  }

  return catalog.models.find((model) => {
    const configName = (model?.configName || "").trim().toLowerCase();
    const name = (model?.name || "").trim().toLowerCase();
    return configName === normalized || name === normalized;
  }) || null;
}

function createTaggingModelCandidate(model) {
  const configName = getModelConfigName(model).trim();
  if (!configName) {
    return null;
  }

  return {
    configName,
    loaded: model?.loaded === true ? true : model?.loaded === false ? false : undefined,
    active: model?.active !== false,
  };
}

function isLoadedCatalogModel(model) {
  return model?.loaded === true;
}

function isActiveCatalogModel(model) {
  return model?.active === true;
}

function buildTaggingPreferenceMap(preferences) {
  const map = new Map();

  (preferences || []).forEach((preference) => {
    const binding = normalizeCapabilityBinding(preference);
    const scope = normalizeTaggingScope(binding.scope);
    const category = normalizeTaggingCategory(binding.category);
    const model = (binding.model || "").trim();
    const key = createTaggingPreferenceKey(scope, category);
    if (binding.capabilityId === "tagging" && binding.slotId === "category" && scope && category && model && !map.has(key)) {
      map.set(key, { scope, category, model });
    }
  });

  return map;
}

function dedupeTaggingCandidates(candidates) {
  const map = new Map();

  (candidates || []).forEach((candidate) => {
    const configName = (candidate?.configName || "").trim();
    const key = configName.toLowerCase();
    if (configName && !map.has(key)) {
      map.set(key, { ...candidate, configName });
    }
  });

  return Array.from(map.values());
}

function sortTaggingCandidates(candidates) {
  return dedupeTaggingCandidates(candidates).sort((left, right) => {
    if (left.loaded === true && right.loaded !== true) {
      return -1;
    }
    if (right.loaded === true && left.loaded !== true) {
      return 1;
    }
    if (!!left.active !== !!right.active) {
      return left.active ? -1 : 1;
    }

    return left.configName.localeCompare(right.configName);
  });
}

function selectTaggingModelCandidate(candidates, preferenceMap, scope, category) {
  const distinctCandidates = dedupeTaggingCandidates(candidates);
  if (distinctCandidates.length === 0) {
    return null;
  }

  const preference = preferenceMap.get(createTaggingPreferenceKey(scope, category));
  if (preference) {
    return distinctCandidates.find((candidate) => candidate.configName.toLowerCase() === preference.model.toLowerCase()) || null;
  }

  return sortTaggingCandidates(distinctCandidates)[0] || null;
}

function collectRequestedTaggingModels(catalog, scope, selectedTaggingCategories, capabilityModelBindings) {
  if (!catalog || !Array.isArray(catalog.models)) {
    return [];
  }

  const resolvedScope = normalizeTaggingScope(scope);
  if (!resolvedScope) {
    return [];
  }

  const selectedCategoryKeys = new Set((selectedTaggingCategories || []).map((category) => normalizeTaggingCategoryKey(category)).filter(Boolean));
  const preferenceMap = buildTaggingPreferenceMap(capabilityModelBindings);
  const categorizedCandidates = new Map();
  const uncategorizedCandidates = [];

  (catalog.models || []).forEach((model) => {
    if (!isActiveCatalogModel(model) || !isTaggingModel(model) || !getTaggingScopes(model).includes(resolvedScope)) {
      return;
    }

    const candidate = createTaggingModelCandidate(model);
    if (!candidate) {
      return;
    }

    const categories = getTaggingCategoriesForModel(model);
    if (categories.length === 0) {
      uncategorizedCandidates.push(candidate);
      return;
    }

    categories.forEach((category) => {
      const categoryKey = normalizeTaggingCategoryKey(category);
      if (selectedCategoryKeys.size > 0 && !selectedCategoryKeys.has(categoryKey)) {
        return;
      }

      if (!categorizedCandidates.has(categoryKey)) {
        categorizedCandidates.set(categoryKey, { category, candidates: [] });
      }

      categorizedCandidates.get(categoryKey).candidates.push(candidate);
    });
  });

  const selectedModels = new Map();

  Array.from(categorizedCandidates.values())
    .sort((left, right) => left.category.localeCompare(right.category))
    .forEach((entry) => {
      const chosen = selectTaggingModelCandidate(entry.candidates, preferenceMap, resolvedScope, entry.category);
      if (chosen) {
        selectedModels.set(chosen.configName.toLowerCase(), chosen);
      }
    });

  if (selectedModels.size === 0 && uncategorizedCandidates.length > 0) {
    const chosen = selectTaggingModelCandidate(uncategorizedCandidates, preferenceMap, resolvedScope, "*");
    if (chosen) {
      selectedModels.set(chosen.configName.toLowerCase(), chosen);
    }
  }

  return Array.from(selectedModels.values()).sort((left, right) => left.configName.localeCompare(right.configName));
}

function collectTaggingPreferenceRows(catalog, preferences) {
  const rows = new Map();
  const preferenceMap = buildTaggingPreferenceMap(preferences);

  (catalog?.models || []).forEach((model) => {
    if (!isTaggingModel(model)) {
      return;
    }

    if (!isLoadedCatalogModel(model)) {
      return;
    }

    const candidate = createTaggingModelCandidate(model);
    if (!candidate) {
      return;
    }

    const categories = getTaggingCategoriesForModel(model);
    if (categories.length === 0) {
      return;
    }

    getTaggingScopes(model).forEach((scope) => {
      categories.forEach((category) => {
        const key = createTaggingPreferenceKey(scope, category);
        if (!rows.has(key)) {
          rows.set(key, { key, scope, category, models: [] });
        }

        rows.get(key).models.push(candidate);
      });
    });
  });

  preferenceMap.forEach((preference) => {
    const key = createTaggingPreferenceKey(preference.scope, preference.category);
    if (!rows.has(key)) {
      rows.set(key, { key, scope: preference.scope, category: preference.category, models: [] });
    }
  });

  return Array.from(rows.values())
    .map((row) => {
      const models = sortTaggingCandidates(row.models);
      const selectedModel = preferenceMap.get(row.key)?.model || "";
      const selectedModelUnavailable = !!selectedModel && !models.some((model) => model.configName.toLowerCase() === selectedModel.toLowerCase());

      return {
        ...row,
        models,
        selectedModel: selectedModelUnavailable ? "" : selectedModel,
        selectedModelUnavailable,
        defaultModel: selectTaggingModelCandidate(models, new Map(), row.scope, row.category),
      };
    })
    .sort((left, right) => left.scope.localeCompare(right.scope) || left.category.localeCompare(right.category));
}

function formatTaggingModelOptionLabel(model) {
  const flags = [];
  if (model?.unavailable) {
    flags.push("unavailable");
  } else {
    if (model?.loaded === true) {
      flags.push("loaded");
    }
    if (model?.loaded === false) {
      flags.push("not loaded");
    }
  }

  return flags.length > 0 ? `${model.configName} (${flags.join(", ")})` : model.configName;
}

function collectTaggingCategories(catalog, mediaKind) {
  if (!catalog || !Array.isArray(catalog.models) || (mediaKind !== "image" && mediaKind !== "video")) {
    return [];
  }

  const requiredScope = mediaKind === "image" ? "asset" : "frame";
  const categories = new Map();

  (catalog.models || []).forEach((model) => {
    if (!isActiveCatalogModel(model) || !isTaggingModel(model) || !getTaggingScopes(model).includes(requiredScope)) {
      return;
    }

    getTaggingCategoriesForModel(model).forEach((category) => {
      const key = normalizeTaggingCategoryKey(category);
      if (key && !categories.has(key)) {
        categories.set(key, category);
      }
    });
  });

  return Array.from(categories.values()).sort((left, right) => left.localeCompare(right));
}

function collectRequestedModels(catalog, features, claims, mediaKind, selectedClaimIds, selectedTaggingCategories, capabilityModelBindings, loadPolicy) {
  const requested = new Map();
  const selectedClaimIdSet = new Set(selectedClaimIds || []);

  (claims || []).forEach((claim) => {
    if (!selectedClaimIdSet.has(claim.claimId)) {
      return;
    }

    if (claim.wantCapability === "tagging" && catalog && Array.isArray(catalog.models)) {
      collectRequestedTaggingModels(catalog, claim.wantScope, selectedTaggingCategories, capabilityModelBindings).forEach((model) => {
        requested.set(model.configName, {
          configName: model.configName,
          source: claim.displayName || claim.claimId,
          loaded: model.loaded,
        });
      });
      return;
    }

    const binding = claim.capabilityId && claim.modelBindingSlotId
      ? findModelBinding(capabilityModelBindings, claim.capabilityId, claim.modelBindingSlotId, claim.wantScope, null)
        || findModelBinding(capabilityModelBindings, claim.capabilityId, claim.modelBindingSlotId, null, null)
      : null;
    if (binding?.model) {
      const catalogModel = findCatalogModel(catalog, binding.model);
      if (isActiveCatalogModel(catalogModel)) {
        requested.set(binding.model, {
          configName: binding.model,
          source: claim.displayName || claim.claimId,
          loaded: catalogModel?.loaded === true ? true : catalogModel?.loaded === false ? false : undefined,
        });
        return;
      }
    }

    const autoSelection = collectAutoSlotModels(catalog, features, claim, { activeOnly: true });
    if (autoSelection.models.length > 0) {
      autoSelection.models.forEach((model) => {
        requested.set(model.configName, {
          configName: model.configName,
          source: claim.displayName || claim.claimId,
          loaded: model.loaded,
        });
      });
      return;
    }

    if (autoSelection.hasSlot && (loadPolicy || "use_loaded") === "use_loaded") {
      return;
    }

    if (Array.isArray(claim.preferredModels) && claim.preferredModels.length > 0) {
      claim.preferredModels.forEach((modelName) => {
        if (typeof modelName === "string" && modelName.trim().length > 0) {
          const catalogModel = findCatalogModel(catalog, modelName);
          if (!isActiveCatalogModel(catalogModel)) {
            return;
          }
          requested.set(modelName, {
            configName: modelName,
            source: claim.displayName || claim.claimId,
            loaded: catalogModel?.loaded === true ? true : catalogModel?.loaded === false ? false : undefined,
          });
        }
      });
      return;
    }

  });

  return Array.from(requested.values()).sort((left, right) => left.configName.localeCompare(right.configName));
}

function summarizeRequestedModels(requestedModels, loadPolicy) {
  const loadedCount = requestedModels.filter((model) => model.loaded).length;
  const missingCount = requestedModels.filter((model) => model.loaded === false).length;
  const unknownCount = requestedModels.filter((model) => model.loaded !== true && model.loaded !== false).length;

  if (requestedModels.length === 0) {
    return "No concrete models are currently implied by this selection.";
  }

  if (unknownCount > 0) {
    return loadPolicy === "use_loaded"
      ? `${requestedModels.length} model(s) are implied by this selection; their loaded state is unknown until the server catalog is reachable.`
      : `${requestedModels.length} model(s) are implied by this selection; model availability is unknown until the server catalog is reachable.`;
  }

  if (loadPolicy === "use_loaded") {
    return missingCount > 0
      ? `${requestedModels.length} model(s) match this selection; ${loadedCount} currently loaded, ${missingCount} not loaded.`
      : `All ${requestedModels.length} requested model(s) are already loaded.`;
  }

  return missingCount > 0
    ? `${requestedModels.length} model(s) match this selection; ${missingCount} may be loaded on demand before the run starts.`
    : `All ${requestedModels.length} requested model(s) are already loaded.`;
}

function copySettings(settings) {
  return {
    serverBaseUrl: settings?.serverBaseUrl || "http://127.0.0.1:8000",
    defaultLoadPolicy: settings?.defaultLoadPolicy || "use_loaded",
    defaultThreshold: settings?.defaultThreshold ?? null,
    requestTimeoutSeconds: settings?.requestTimeoutSeconds ?? 120,
    maxInFlight: settings?.maxInFlight ?? 2,
    dispatchResultsByDefault: settings?.dispatchResultsByDefault ?? true,
    pathMappings: (settings?.pathMappings || []).map((mapping) => ({
      fromPrefix: mapping.fromPrefix || "",
      toPrefix: mapping.toPrefix || "",
    })),
    capabilityModelBindings: (settings?.capabilityModelBindings || []).map(normalizeCapabilityBinding).filter((binding) => binding.capabilityId && binding.slotId && binding.model),
    runPresets: (settings?.runPresets || []).map((preset) => ({
      presetId: (preset.presetId || "").trim().toLowerCase(),
      displayName: (preset.displayName || "").trim(),
      capabilityIds: Array.isArray(preset.capabilityIds) ? preset.capabilityIds.filter(Boolean) : [],
      claimIds: Array.isArray(preset.claimIds) ? preset.claimIds.filter(Boolean) : [],
      pipelineName: preset.pipelineName || null,
      categoriesToSkip: Array.isArray(preset.categoriesToSkip) ? preset.categoriesToSkip.filter(Boolean) : null,
      loadPolicy: preset.loadPolicy || null,
    })).filter((preset) => preset.presetId && preset.displayName),
    customPipelines: (settings?.customPipelines || []).map((pipeline) => ({
      pipelineName: pipeline.pipelineName || "",
      displayName: pipeline.displayName || pipeline.pipelineName || "",
      mediaKind: pipeline.mediaKind || "video",
      capabilityId: pipeline.capabilityId || (pipeline.pipelineName ? `custom.${String(pipeline.pipelineName).toLowerCase()}` : ""),
      description: pipeline.description || "",
      capabilityIds: Array.isArray(pipeline.capabilityIds) ? pipeline.capabilityIds.filter(Boolean) : [],
      claimIds: Array.isArray(pipeline.claimIds) ? pipeline.claimIds.filter(Boolean) : [],
      useAllFullImageModels: !!pipeline.useAllFullImageModels,
      fullImageModels: Array.isArray(pipeline.fullImageModels) ? pipeline.fullImageModels.filter(Boolean) : [],
      detectorModels: Array.isArray(pipeline.detectorModels) ? pipeline.detectorModels.filter(Boolean) : [],
      regionModels: pipeline.regionModels || {},
      useAllAudioModels: !!pipeline.useAllAudioModels,
      audioModels: Array.isArray(pipeline.audioModels) ? pipeline.audioModels.filter(Boolean) : [],
    })).filter((pipeline) => pipeline.pipelineName),
  };
}

function getRunDialogStorageKey(selection) {
  const entityType = selection?.entityType || "paths";
  return `${RUN_DIALOG_STORAGE_PREFIX}:${entityType}`;
}

function readRunDialogDefaults(selection) {
  if (typeof window === "undefined") {
    return {};
  }

  try {
    const raw = window.localStorage.getItem(getRunDialogStorageKey(selection));
    const parsed = raw ? JSON.parse(raw) : null;
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writeRunDialogDefaults(selection, form) {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(getRunDialogStorageKey(selection), JSON.stringify({
      mediaKind: form.mediaKind,
      capabilityIds: form.capabilityIds,
      presetId: form.presetId,
      frameInterval: form.frameInterval,
      taggingCategories: form.taggingCategories,
    }));
  } catch {
    // Ignore storage failures; the run should still queue normally.
  }
}

function normalizeEntityType(entityType) {
  const normalized = (entityType || "").trim().toLowerCase();
  if (normalized === "scenes") {
    return "scene";
  }
  if (normalized === "images") {
    return "image";
  }
  return normalized;
}

function normalizeSelectionPayload(payload) {
  if (!payload || typeof payload !== "object") {
    return null;
  }

  const entityType = normalizeEntityType(payload.entityType || payload.pageType);
  const entityIds = Array.from(new Set(
    ((payload.entityIds || payload.selectedIds || payload.ids || []) || [])
      .map((value) => Number(value))
      .filter((value) => Number.isInteger(value) && value > 0)
  ));

  if (!entityType || entityIds.length === 0) {
    return null;
  }

  return {
    entityType,
    entityIds,
    count: entityIds.length,
  };
}

function getMediaKindsForSelection(selection) {
  if (!selection) {
    return MEDIA_KINDS;
  }

  if (selection.entityType === "image") {
    return MEDIA_KINDS.filter((kind) => kind.value === "image");
  }

  if (selection.entityType === "scene") {
    return MEDIA_KINDS.filter((kind) => kind.value !== "image");
  }

  return MEDIA_KINDS;
}

function formatSelectionTitle(selection) {
  if (!selection) {
    return "Run AI";
  }

  const noun = selection.entityType === "image" ? "image" : "scene";
  const count = selection.count || selection.entityIds.length;
  return `Run AI on ${count} ${noun}${count === 1 ? "" : "s"}`;
}

function formatSelectionDescription(selection) {
  if (!selection) {
    return "";
  }

  if (selection.entityType === "image") {
    return "The selected Cove images will be resolved to concrete files before the AI run is queued.";
  }

  return "The selected Cove scenes will be resolved to their source media before the AI run is queued.";
}

  function useAutosaveSettings(settings, enabled, onStart, onSaved, onError) {
    const serialized = JSON.stringify(settings);
    const initializedRef = useRef(false);
    const lastSavedRef = useRef("");

    useEffect(() => {
      if (!enabled) {
        return undefined;
      }
      if (!initializedRef.current) {
        initializedRef.current = true;
        lastSavedRef.current = serialized;
        return undefined;
      }
      if (serialized === lastSavedRef.current) {
        return undefined;
      }

      let cancelled = false;
      const handle = window.setTimeout(async () => {
        onStart();
        try {
          const saved = copySettings(await api("/settings", {
            method: "PUT",
            body: serialized,
          }));
          lastSavedRef.current = JSON.stringify(saved);
          if (!cancelled) {
            onSaved(saved);
          }
        } catch (error) {
          if (!cancelled) {
            onError(error);
          }
        }
      }, 500);

      return () => {
        cancelled = true;
        window.clearTimeout(handle);
      };
    }, [serialized, enabled]);
  }

function SettingsEditor({ settings, health, catalog, capabilities = [], busy, message, pipelineBusy = "", onChange, onSyncPipeline = null, onDeletePipeline = null }) {
  const mappings = settings.pathMappings || [];
  const modelBindingRows = getModelBindingRows(capabilities, catalog, settings);
  const customPipelines = settings.customPipelines || [];

  function patch(next) {
    onChange({ ...settings, ...next });
  }

  function patchMapping(index, field, value) {
    patch({
      pathMappings: mappings.map((mapping, mappingIndex) => (
        mappingIndex === index ? { ...mapping, [field]: value } : mapping
      )),
    });
  }

  function patchTaggingPreference(scope, category, model) {
    patch({ capabilityModelBindings: upsertModelBinding(settings.capabilityModelBindings, "tagging", "category", scope, category, model) });
  }

  function patchModelBinding(row, model) {
    patch({ capabilityModelBindings: upsertModelBinding(settings.capabilityModelBindings, row.capabilityId, row.slotId, null, null, model) });
  }

  function patchPipeline(index, next) {
    patch({
      customPipelines: customPipelines.map((pipeline, pipelineIndex) => (
        pipelineIndex === index ? { ...pipeline, ...next } : pipeline
      )),
    });
  }

  function addPipeline() {
    const nextIndex = customPipelines.length + 1;
    patch({
      customPipelines: [
        ...customPipelines,
        {
          pipelineName: `custom_pipeline_${nextIndex}`,
          displayName: `Custom pipeline ${nextIndex}`,
          mediaKind: "video",
          capabilityId: `custom.custom-pipeline-${nextIndex}`,
          description: "",
          capabilityIds: [],
          claimIds: [],
          useAllFullImageModels: false,
          fullImageModels: [],
          detectorModels: [],
          regionModels: {},
          useAllAudioModels: false,
          audioModels: [],
        },
      ],
    });
  }

  function removePipeline(index) {
    patch({ customPipelines: customPipelines.filter((_, pipelineIndex) => pipelineIndex !== index) });
  }

  return h("div", { className: "ai-core-stack" }, [
    h("div", { className: "ai-core-grid ai-core-grid-compact", key: "fields" }, [
      h("label", { className: "ai-core-field", key: "serverBaseUrl" }, [
        h("span", { className: "ai-core-label" }, "Server URL"),
        h("input", {
          className: "ai-core-input",
          type: "text",
          value: settings.serverBaseUrl,
          onChange: (event) => patch({ serverBaseUrl: event.target.value }),
          placeholder: "http://127.0.0.1:8000",
        }),
      ]),
      h("label", { className: "ai-core-field", key: "loadPolicy" }, [
        h("span", { className: "ai-core-label" }, "Default load policy"),
        h("select", {
          className: "ai-core-select",
          value: settings.defaultLoadPolicy,
          onChange: (event) => patch({ defaultLoadPolicy: event.target.value }),
        }, LOAD_POLICIES.map((policy) => h("option", { key: policy.value, value: policy.value }, policy.label))),
      ]),
      h("label", { className: "ai-core-field", key: "defaultThreshold" }, [
        h("span", { className: "ai-core-label" }, "Default threshold"),
        h("input", {
          className: "ai-core-input",
          type: "number",
          step: "0.01",
          min: "0",
          max: "1",
          value: settings.defaultThreshold ?? "",
          onChange: (event) => patch({ defaultThreshold: event.target.value === "" ? null : Number(event.target.value) }),
          placeholder: "Use server default",
        }),
      ]),
      h("label", { className: "ai-core-field", key: "requestTimeoutSeconds" }, [
        h("span", { className: "ai-core-label" }, "Request timeout (seconds)"),
        h("input", {
          className: "ai-core-input",
          type: "number",
          min: "0",
          value: settings.requestTimeoutSeconds,
          onChange: (event) => patch({ requestTimeoutSeconds: Number(event.target.value || 0) }),
        }),
      ]),
      h("label", { className: "ai-core-field", key: "maxInFlight" }, [
        h("span", { className: "ai-core-label" }, "Max in flight"),
        h("input", {
          className: "ai-core-input",
          type: "number",
          min: "1",
          value: settings.maxInFlight,
          onChange: (event) => patch({ maxInFlight: Number(event.target.value || 1) }),
        }),
      ]),
      h("label", { className: "ai-core-field ai-core-checkbox-field", key: "dispatchResultsByDefault" }, [
        h("input", {
          className: "ai-core-checkbox",
          type: "checkbox",
          checked: settings.dispatchResultsByDefault,
          onChange: (event) => patch({ dispatchResultsByDefault: event.target.checked }),
        }),
        h("span", { className: "ai-core-label" }, "Dispatch results to installed AI extensions by default"),
      ]),
    ]),
    h("div", { className: "ai-core-section", key: "pathMappings" }, [
      h("div", { className: "ai-core-section-header" }, [
        h("div", null, [
          h("h4", { className: "ai-core-subtitle" }, "Path mappings"),
          h("p", { className: "ai-core-muted" }, "Translate Cove-visible paths into paths the model server can open."),
        ]),
        h("button", {
          className: "ai-core-button ai-core-button-secondary",
          type: "button",
          onClick: () => patch({ pathMappings: [...mappings, { fromPrefix: "", toPrefix: "" }] }),
        }, "Add mapping"),
      ]),
      mappings.length === 0
        ? h("p", { className: "ai-core-empty" }, "No path mappings configured.")
        : h("div", { className: "ai-core-stack" }, mappings.map((mapping, index) => h("div", { className: "ai-core-grid ai-core-grid-compact ai-core-mapping-row", key: `${mapping.fromPrefix}:${mapping.toPrefix}:${index}` }, [
            h("label", { className: "ai-core-field", key: `from-${index}` }, [
              h("span", { className: "ai-core-label" }, "From prefix"),
              h("input", {
                className: "ai-core-input ai-core-code",
                type: "text",
                value: mapping.fromPrefix,
                onChange: (event) => patchMapping(index, "fromPrefix", event.target.value),
                placeholder: "C:/Library",
              }),
            ]),
            h("label", { className: "ai-core-field", key: `to-${index}` }, [
              h("span", { className: "ai-core-label" }, "To prefix"),
              h("input", {
                className: "ai-core-input ai-core-code",
                type: "text",
                value: mapping.toPrefix,
                onChange: (event) => patchMapping(index, "toPrefix", event.target.value),
                placeholder: "/mnt/library",
              }),
            ]),
            h("div", { className: "ai-core-field ai-core-field-actions", key: `remove-${index}` }, [
              h("span", { className: "ai-core-label" }, " "),
              h("button", {
                className: "ai-core-button ai-core-button-secondary",
                type: "button",
                onClick: () => patch({ pathMappings: mappings.filter((_, mappingIndex) => mappingIndex !== index) }),
              }, "Remove"),
            ]),
          ]))),
    ]),
    h("div", { className: "ai-core-section", key: "modelBindings" }, [
      h("div", { className: "ai-core-section-header" }, [
        h("div", null, [
          h("h4", { className: "ai-core-subtitle" }, "AI feature model bindings"),
          h("p", { className: "ai-core-muted" }, "Choose which concrete model powers each user-facing AI feature slot."),
        ]),
      ]),
      modelBindingRows.length === 0
        ? h("p", { className: "ai-core-empty" }, "No feature model slots are available from installed AI extensions.")
        : h("div", { className: "ai-core-claim-list" }, modelBindingRows.map((row) => h("div", { className: "ai-core-model-row", key: row.key }, [
            h("div", { className: "ai-core-model-copy" }, [
              h("div", { className: "ai-core-model-title" }, `${row.capabilityLabel}: ${row.slotLabel}`),
              h("div", { className: "ai-core-model-meta" }, [
                h("span", { className: "ai-core-pill ai-core-pill-muted", key: "extension" }, row.extensionLabel),
                row.defaultModel ? h("span", { key: "default" }, `Auto -> ${row.defaultModel}`) : h("span", { key: "default" }, "No automatic model available"),
                row.description ? h("span", { key: "description" }, row.description) : null,
                row.savedModelUnavailable ? h("span", { className: "ai-core-pill ai-core-pill-warning", key: "missing" }, "saved model not loaded") : null,
              ]),
            ]),
            h("label", { className: "ai-core-field ai-core-preference-field" }, [
              h("span", { className: "ai-core-label" }, "Selected model"),
              h("select", {
                className: "ai-core-select ai-core-preference-select",
                value: row.selectedModel || "",
                onChange: (event) => patchModelBinding(row, event.target.value),
              }, [
                h("option", { key: "auto", value: "" }, row.defaultModel ? `Auto (${row.defaultModel})` : "Auto"),
                ...row.models.map((model) => h("option", { key: model.configName, value: model.configName }, formatTaggingModelOptionLabel(model))),
              ]),
            ]),
          ]))),
    ]),
    h("div", { className: "ai-core-section", key: "customPipelines" }, [
      h("div", { className: "ai-core-section-header" }, [
        h("div", null, [
          h("h4", { className: "ai-core-subtitle" }, "Custom AI pipelines"),
          h("p", { className: "ai-core-muted" }, "Compose advanced full-image, detector, region-branch, or audio pipelines and sync them to nsfw_ai_server."),
        ]),
        h("button", {
          className: "ai-core-button ai-core-button-secondary",
          type: "button",
          onClick: addPipeline,
        }, "Add pipeline"),
      ]),
      customPipelines.length === 0
        ? h("p", { className: "ai-core-empty" }, "No custom pipelines configured.")
        : h("div", { className: "ai-core-stack" }, customPipelines.map((pipeline, index) => {
            const busyPipeline = pipelineBusy === pipeline.pipelineName;
            return h("div", { className: "ai-core-custom-pipeline rounded-xl border border-border bg-card p-4", key: `${pipeline.pipelineName}:${index}` }, [
              h("div", { className: "ai-core-grid ai-core-grid-compact" }, [
                h("label", { className: "ai-core-field", key: "pipelineName" }, [
                  h("span", { className: "ai-core-label" }, "Pipeline name"),
                  h("input", {
                    className: "ai-core-input ai-core-code",
                    type: "text",
                    value: pipeline.pipelineName,
                    onChange: (event) => patchPipeline(index, { pipelineName: event.target.value, capabilityId: pipeline.capabilityId || `custom.${event.target.value}` }),
                  }),
                ]),
                h("label", { className: "ai-core-field", key: "displayName" }, [
                  h("span", { className: "ai-core-label" }, "Display name"),
                  h("input", {
                    className: "ai-core-input",
                    type: "text",
                    value: pipeline.displayName,
                    onChange: (event) => patchPipeline(index, { displayName: event.target.value }),
                  }),
                ]),
                h("label", { className: "ai-core-field", key: "mediaKind" }, [
                  h("span", { className: "ai-core-label" }, "Media kind"),
                  h("select", {
                    className: "ai-core-select",
                    value: pipeline.mediaKind,
                    onChange: (event) => patchPipeline(index, { mediaKind: event.target.value }),
                  }, MEDIA_KINDS.map((kind) => h("option", { key: kind.value, value: kind.value }, kind.label))),
                ]),
                h("label", { className: "ai-core-field", key: "capabilityId" }, [
                  h("span", { className: "ai-core-label" }, "Capability id"),
                  h("input", {
                    className: "ai-core-input ai-core-code",
                    type: "text",
                    value: pipeline.capabilityId,
                    onChange: (event) => patchPipeline(index, { capabilityId: event.target.value }),
                  }),
                ]),
              ]),
              h("label", { className: "ai-core-field", key: "description" }, [
                h("span", { className: "ai-core-label" }, "Description"),
                h("input", {
                  className: "ai-core-input",
                  type: "text",
                  value: pipeline.description || "",
                  onChange: (event) => patchPipeline(index, { description: event.target.value }),
                }),
              ]),
              h("div", { className: "ai-core-grid ai-core-grid-compact" }, [
                h("label", { className: "ai-core-field", key: "capabilityIds" }, [
                  h("span", { className: "ai-core-label" }, "Included capability ids"),
                  h("input", {
                    className: "ai-core-input ai-core-code",
                    type: "text",
                    value: joinList(pipeline.capabilityIds),
                    onChange: (event) => patchPipeline(index, { capabilityIds: splitList(event.target.value) }),
                    placeholder: "tagging, faces",
                  }),
                ]),
                pipeline.mediaKind === "audio"
                  ? h("label", { className: "ai-core-field ai-core-checkbox-field", key: "allAudio" }, [
                      h("input", {
                        className: "ai-core-checkbox",
                        type: "checkbox",
                        checked: !!pipeline.useAllAudioModels,
                        onChange: (event) => patchPipeline(index, { useAllAudioModels: event.target.checked }),
                      }),
                      h("span", { className: "ai-core-label" }, "Use all active audio models"),
                    ])
                  : h("label", { className: "ai-core-field ai-core-checkbox-field", key: "allFullImage" }, [
                      h("input", {
                        className: "ai-core-checkbox",
                        type: "checkbox",
                        checked: !!pipeline.useAllFullImageModels,
                        onChange: (event) => patchPipeline(index, { useAllFullImageModels: event.target.checked }),
                      }),
                      h("span", { className: "ai-core-label" }, "Use all active full-image models"),
                    ]),
              ]),
              pipeline.mediaKind === "audio"
                ? h("label", { className: "ai-core-field", key: "audioModels" }, [
                    h("span", { className: "ai-core-label" }, "Audio models"),
                    h("input", {
                      className: "ai-core-input ai-core-code",
                      type: "text",
                      value: joinList(pipeline.audioModels),
                      disabled: !!pipeline.useAllAudioModels,
                      onChange: (event) => patchPipeline(index, { audioModels: splitList(event.target.value) }),
                    }),
                  ])
                : h("div", { className: "ai-core-grid ai-core-grid-compact", key: "imageVideoModels" }, [
                    h("label", { className: "ai-core-field", key: "fullImageModels" }, [
                      h("span", { className: "ai-core-label" }, "Full-image models"),
                      h("input", {
                        className: "ai-core-input ai-core-code",
                        type: "text",
                        value: joinList(pipeline.fullImageModels),
                        disabled: !!pipeline.useAllFullImageModels,
                        onChange: (event) => patchPipeline(index, { fullImageModels: splitList(event.target.value) }),
                      }),
                    ]),
                    h("label", { className: "ai-core-field", key: "detectorModels" }, [
                      h("span", { className: "ai-core-label" }, "Detector models"),
                      h("input", {
                        className: "ai-core-input ai-core-code",
                        type: "text",
                        value: joinList(pipeline.detectorModels),
                        onChange: (event) => patchPipeline(index, { detectorModels: splitList(event.target.value) }),
                      }),
                    ]),
                    h("label", { className: "ai-core-field", key: "regionModels" }, [
                      h("span", { className: "ai-core-label" }, "Region models"),
                      h("textarea", {
                        className: "ai-core-textarea ai-core-code",
                        rows: 3,
                        value: formatRegionModels(pipeline.regionModels),
                        onChange: (event) => patchPipeline(index, { regionModels: parseRegionModels(event.target.value) }),
                        placeholder: "face_detector_torchexport: face_embedding_torchexport",
                      }),
                    ]),
                  ]),
              h("div", { className: "ai-core-toolbar" }, [
                h("button", {
                  className: "ai-core-button ai-core-button-secondary",
                  type: "button",
                  disabled: busyPipeline || !onSyncPipeline,
                  onClick: () => onSyncPipeline && onSyncPipeline(pipeline),
                }, busyPipeline ? "Syncing..." : "Sync to server"),
                h("button", {
                  className: "ai-core-button ai-core-button-secondary",
                  type: "button",
                  disabled: busyPipeline,
                  onClick: () => {
                    if (onDeletePipeline && pipeline.pipelineName) {
                      onDeletePipeline(pipeline.pipelineName);
                    }
                    removePipeline(index);
                  },
                }, "Remove"),
              ]),
            ]);
          })),
    ]),
    h("div", { className: "ai-core-toolbar", key: "actions" }, [
      h("div", { className: `ai-core-pill ai-core-pill-${statusTone(health?.status)}` }, health?.status === "reachable" ? "Server reachable" : health?.status === "unreachable" ? "Server unreachable" : "Status pending"),
      h("div", { className: "ai-core-toolbar-spacer" }),
      busy ? h("span", { className: "ai-core-muted", key: "saving" }, "Saving...") : null,
      message ? h("span", { className: "ai-core-muted", key: "message" }, message) : null,
    ]),
  ]);
}

function ModelsPanel({ catalog, busyModel, onLoadToggle, onRefresh }) {
  const models = (catalog?.models || []).slice().sort((left, right) => {
    const leftCategory = (left.categories || [""])[0] || "";
    const rightCategory = (right.categories || [""])[0] || "";
    return leftCategory.localeCompare(rightCategory) || (left.configName || "").localeCompare(right.configName || "");
  });
  const availableModels = models.filter((model) => !model.active);
  const activeModels = models.filter((model) => model.active);

  const MODEL_CATEGORY_PREFIX_LABELS = {
    audio_classification: "Audio Classification",
    audio_embeddings: "Audio Embeddings",
    visual_embeddings: "Visual Embeddings",
    face_detections: "Face Detections",
    face_embeddings: "Face Embeddings",
  };

  function humanizeCatalogToken(value) {
    return (value || "")
      .trim()
      .replace(/[_-]+/g, " ")
      .replace(/\b([a-z])/g, (match, char) => char.toUpperCase());
  }

  function splitCategoryParts(category) {
    return (category || "").trim().split("_").filter(Boolean);
  }

  function formatModelCategoryGroup(category) {
    const parts = splitCategoryParts(category);
    if (parts.length >= 2) {
      const prefix = `${parts[0]}_${parts[1]}`.toLowerCase();
      if (MODEL_CATEGORY_PREFIX_LABELS[prefix]) {
        return MODEL_CATEGORY_PREFIX_LABELS[prefix];
      }
    }

    return humanizeCatalogToken(category) || "Uncategorized";
  }

  function formatModelCategoryLabel(category) {
    const parts = splitCategoryParts(category);
    if (parts.length >= 3) {
      const prefix = `${parts[0]}_${parts[1]}`.toLowerCase();
      const prefixLabel = MODEL_CATEGORY_PREFIX_LABELS[prefix];
      if (prefixLabel) {
        return `${prefixLabel} / ${humanizeCatalogToken(parts.slice(2).join(" "))}`;
      }
    }

    return formatModelCategoryGroup(category);
  }

  function getDisplayModelCategories(model) {
    const categories = new Map();
    (Array.isArray(model?.categories) ? model.categories : []).forEach((category) => {
      const normalized = (category || "").trim();
      const key = normalized.toLowerCase();
      const label = formatModelCategoryLabel(normalized);
      if (key && label && !categories.has(key)) {
        categories.set(key, label);
      }
    });
    return Array.from(categories.values());
  }

  function getPrimaryModelCategoryGroup(model) {
    const firstCategory = (Array.isArray(model?.categories) ? model.categories : []).find((category) => (category || "").trim());
    return firstCategory ? formatModelCategoryGroup(firstCategory) : "Uncategorized";
  }

  function formatFacetList(values) {
    const unique = new Map();
    (Array.isArray(values) ? values : []).forEach((value) => {
      const normalized = (value || "").trim();
      const key = normalized.toLowerCase();
      const label = humanizeCatalogToken(normalized);
      if (key && label && !unique.has(key)) {
        unique.set(key, label);
      }
    });
    return Array.from(unique.values()).join(", ") || "-";
  }

  function isTaggingCatalogModel(model) {
    return Array.isArray(model?.capabilities) && model.capabilities.some((capability) => (capability || "").trim().toLowerCase() === "tagging");
  }

  function formatIncompatibilityReason(reason) {
    const text = (reason || "").trim();
    const match = text.match(/^(Category already active|Tagging category already active):\s*(.*)$/i);
    if (!match) {
      return text;
    }

    const categories = match[2]
      .split(",")
      .map((category) => formatModelCategoryLabel(category.trim()))
      .filter(Boolean);
    return `${match[1]}: ${categories.join(", ")}`;
  }

  function collectTaggingConflictReasons(model) {
    if (model?.active || !isTaggingCatalogModel(model)) {
      return [];
    }

    const serverReasons = (model?.incompatibilityReason || "").toLowerCase();
    if (serverReasons.includes("category already active")) {
      return [];
    }

    const candidateCategories = new Map();
    (Array.isArray(model?.categories) ? model.categories : []).forEach((category) => {
      const normalized = (category || "").trim();
      const key = normalized.toLowerCase();
      if (key && !candidateCategories.has(key)) {
        candidateCategories.set(key, formatModelCategoryLabel(normalized));
      }
    });

    if (candidateCategories.size === 0) {
      return [];
    }

    const conflicts = new Map();
    activeModels.filter(isTaggingCatalogModel).forEach((activeModel) => {
      (Array.isArray(activeModel?.categories) ? activeModel.categories : []).forEach((category) => {
        const normalized = (category || "").trim();
        const key = normalized.toLowerCase();
        if (key && candidateCategories.has(key) && !conflicts.has(key)) {
          conflicts.set(key, candidateCategories.get(key));
        }
      });
    });

    return conflicts.size > 0
      ? [`Tagging category already active: ${Array.from(conflicts.values()).join(", ")}`]
      : [];
  }

  function collectModelWarnings(model) {
    const warnings = new Map();
    (model?.incompatibilityReason || "")
      .split(";")
      .map((reason) => formatIncompatibilityReason(reason))
      .map((reason) => reason.trim())
      .filter(Boolean)
      .forEach((reason) => warnings.set(reason.toLowerCase(), reason));

    collectTaggingConflictReasons(model)
      .map((reason) => formatIncompatibilityReason(reason))
      .forEach((reason) => {
        const key = reason.toLowerCase();
        if (!warnings.has(key)) {
          warnings.set(key, reason);
        }
      });

    return Array.from(warnings.values());
  }

  function groupModelsByCategory(items) {
    const groups = new Map();

    (items || []).forEach((model) => {
      const label = getPrimaryModelCategoryGroup(model);
      const key = label.toLowerCase();
      if (!groups.has(key)) {
        groups.set(key, { label, items: [] });
      }
      groups.get(key).items.push(model);
    });

    return Array.from(groups.values())
      .sort((left, right) => left.label.localeCompare(right.label))
      .map((group) => ({
        ...group,
        items: group.items.slice().sort((left, right) => {
          if (!!left.loaded !== !!right.loaded) {
            return left.loaded ? -1 : 1;
          }
          return (left.configName || "").localeCompare(right.configName || "");
        }),
      }));
  }

  function formatSize(model) {
    return model.imageSize ?? model.modelSize ?? "-";
  }

  function renderModelDetail(label, value, code = false) {
    return h("div", { className: "ai-core-model-detail", key: label }, [
      h("span", { className: "ai-core-model-detail-label" }, label),
      h("span", { className: code ? "ai-core-model-detail-value ai-core-code" : "ai-core-model-detail-value" }, value || "-"),
    ]);
  }

  function renderList(title, items, emptyText) {
    const groups = groupModelsByCategory(items);
    return h("section", { className: "ai-core-model-group", key: title }, [
      h("div", { className: "ai-core-model-group-header" }, title),
      items.length === 0
        ? h("p", { className: "ai-core-empty" }, emptyText)
        : h("div", { className: "ai-core-stack ai-core-tight" }, groups.map((group) => h("div", { className: "ai-core-model-category-group", key: `${title}:${group.label}` }, [
            h("div", { className: "ai-core-model-category-header" }, group.label),
            h("div", { className: "ai-core-stack ai-core-tight" }, group.items.map((model) => {
              const busy = busyModel === model.configName;
              const warnings = collectModelWarnings(model);
              const blocked = !model.active && warnings.length > 0;
              const displayCategories = getDisplayModelCategories(model);
              const secondaryName = model.name && model.name !== model.configName ? model.name : "";
              return h("div", { className: "ai-core-model-row", key: model.configName }, [
                h("div", { className: "ai-core-model-copy" }, [
                  h("div", { className: "ai-core-model-heading" }, [
                    h("div", { className: "ai-core-model-title" }, model.configName || model.name || "Unnamed model"),
                    secondaryName
                      ? h("div", { className: "ai-core-model-subtitle ai-core-code" }, secondaryName)
                      : null,
                  ]),
                  h("div", { className: "ai-core-model-pill-row" }, [
                    model.version ? h("span", { className: "ai-core-pill ai-core-pill-muted", key: "version" }, `v${model.version}`) : null,
                    h("span", { className: "ai-core-pill ai-core-pill-muted", key: "size" }, `size ${formatSize(model)}`),
                    model.loaded ? h("span", { className: "ai-core-pill ai-core-pill-success", key: "loaded" }, "loaded") : null,
                    model.active ? h("span", { className: "ai-core-pill ai-core-pill-success", key: "active" }, "active") : null,
                  ]),
                  h("div", { className: "ai-core-model-detail-grid" }, [
                    renderModelDetail("Category", displayCategories.join(", ") || "-"),
                    renderModelDetail("Capability", formatFacetList(model.capabilities)),
                    renderModelDetail("Scope", formatFacetList(model.supportedScopes)),
                  ]),
                  model.info
                    ? h("div", { className: "ai-core-model-note" }, model.info)
                    : null,
                  warnings.length > 0
                    ? h("div", { className: "ai-core-model-warning-list" }, warnings.map((warning, index) => h("span", { className: "ai-core-pill ai-core-pill-warning", key: `${model.configName}:warning:${index}` }, warning)))
                    : null,
                ]),
                h("div", { className: "ai-core-model-actions" }, [
                  h("button", {
                    className: "ai-core-button ai-core-button-secondary",
                    type: "button",
                    disabled: busy || blocked,
                    onClick: () => onLoadToggle(model.configName, !model.active),
                  }, busy ? "Working..." : model.active ? "Unload" : "Load"),
                ]),
              ]);
            })),
          ]))),
    ]);
  }

  return h("div", { className: "ai-core-stack" }, [
    h("div", { className: "ai-core-section-header", key: "header" }, [
      h("div", null, [
        h("h4", { className: "ai-core-subtitle" }, "Model catalog"),
      ]),
      h("button", {
        className: "ai-core-button ai-core-button-secondary",
        type: "button",
        onClick: onRefresh,
      }, "Refresh"),
    ]),
    models.length === 0
      ? h("p", { className: "ai-core-empty", key: "empty" }, "No models were returned by the server.")
      : h("div", { className: "ai-core-stack", key: "groups" }, [
          renderList("Available AI Models", availableModels, "No inactive models are available."),
          renderList("Active AI Models", activeModels, "No models are active."),
        ]),
  ]);
}

function RunComposer({ capabilities, catalog, settings, busy, message, onQueue, selection = null, submitLabel = "Queue AI job" }) {
  const defaultVideoFrameInterval = "2";
  const mediaKinds = getMediaKindsForSelection(selection);
  const savedDefaults = readRunDialogDefaults(selection);
  const hasSavedCapabilityIds = Array.isArray(savedDefaults.capabilityIds) && savedDefaults.capabilityIds.length > 0;
  const hasSavedTaggingCategories = Array.isArray(savedDefaults.taggingCategories) && savedDefaults.taggingCategories.length > 0;
  const savedMediaKind = typeof savedDefaults.mediaKind === "string" && mediaKinds.some((kind) => kind.value === savedDefaults.mediaKind)
    ? savedDefaults.mediaKind
    : null;
  const initialMediaKind = savedMediaKind || (selection?.entityType === "image" ? "image" : "video");
  const initialFeatures = collectCapabilityFeatures(capabilities, initialMediaKind, settings?.customPipelines || [], catalog);
  const initialAvailableTaggingCategories = collectTaggingCategories(catalog, initialMediaKind);
  const savedTaggingCategories = hasSavedTaggingCategories ? savedDefaults.taggingCategories.filter((category) => typeof category === "string") : [];
  const initialFeatureIds = initialFeatures.map((feature) => feature.capabilityId);
  const matchingSavedTaggingCategories = initialAvailableTaggingCategories.length > 0
    ? savedTaggingCategories.filter((category) => initialAvailableTaggingCategories.includes(category))
    : savedTaggingCategories;
  const initialTaggingCategories = matchingSavedTaggingCategories.length > 0
    ? matchingSavedTaggingCategories
    : initialAvailableTaggingCategories;
  const [form, setForm] = useState({
    mediaKind: initialMediaKind,
    pathsText: "",
    presetId: typeof savedDefaults.presetId === "string" ? savedDefaults.presetId : "",
    capabilityIds: hasSavedCapabilityIds
      ? savedDefaults.capabilityIds.filter((capabilityId) => typeof capabilityId === "string" && initialFeatureIds.includes(capabilityId))
      : initialFeatureIds,
    frameInterval: typeof savedDefaults.frameInterval === "string" ? savedDefaults.frameInterval : initialMediaKind === "video" ? defaultVideoFrameInterval : "",
    taggingCategories: initialTaggingCategories,
    forceExisting: false,
  });
  const capabilityDefaultsApplied = useRef(hasSavedCapabilityIds || initialFeatures.length > 0);
  const taggingDefaultsApplied = useRef(initialTaggingCategories.length > 0 || hasSavedTaggingCategories);
  const taggingCategoriesTouched = useRef(false);

  const claims = collectClaims(capabilities, form.mediaKind);
  const features = collectCapabilityFeatures(capabilities, form.mediaKind, settings?.customPipelines || [], catalog);
  const selectedClaimIds = collectSelectedClaimIds(features, claims, form.capabilityIds);
  const availableTaggingCategories = collectTaggingCategories(catalog, form.mediaKind);
  const effectiveTaggingCategories = form.taggingCategories.length > 0 || taggingCategoriesTouched.current
    ? form.taggingCategories
    : availableTaggingCategories;
  const hasSelectedTaggingClaims = claims
    .filter((claim) => claim.wantCapability === "tagging")
    .some((claim) => selectedClaimIds.includes(claim.claimId));
  const requestedModels = collectRequestedModels(catalog, features, claims, form.mediaKind, selectedClaimIds, effectiveTaggingCategories, settings?.capabilityModelBindings || [], settings?.defaultLoadPolicy || "use_loaded");
  const runPresets = settings?.runPresets || [];
  const showMediaKindField = mediaKinds.length > 1;
  const hasSelectionTargets = !!selection && Array.isArray(selection.entityIds) && selection.entityIds.length > 0;
  const pathCount = splitList(form.pathsText).length;

  useEffect(() => {
    if (mediaKinds.length > 0 && !mediaKinds.some((kind) => kind.value === form.mediaKind)) {
      setForm((current) => ({ ...current, mediaKind: mediaKinds[0].value }));
    }
  }, [form.mediaKind, mediaKinds]);

  useEffect(() => {
    setForm((current) => {
      if (current.mediaKind !== "video" || current.frameInterval !== "") {
        return current;
      }

      return {
        ...current,
        frameInterval: defaultVideoFrameInterval,
      };
    });
  }, [defaultVideoFrameInterval, form.mediaKind]);

  useEffect(() => {
    if (features.length === 0) {
      setForm((current) => current.capabilityIds.length === 0
        ? current
        : {
            ...current,
            capabilityIds: [],
          });
      capabilityDefaultsApplied.current = true;
      return;
    }

    const availableIds = features.map((feature) => feature.capabilityId);
    setForm((current) => {
      const currentIds = current.capabilityIds.filter((capabilityId) => availableIds.includes(capabilityId));
      const resolvedCapabilityIds = !capabilityDefaultsApplied.current && currentIds.length === 0
        ? availableIds
        : currentIds;
      if (resolvedCapabilityIds.length === current.capabilityIds.length
        && resolvedCapabilityIds.every((capabilityId, index) => capabilityId === current.capabilityIds[index])) {
        return current;
      }

      return {
        ...current,
        capabilityIds: resolvedCapabilityIds,
      };
    });
    capabilityDefaultsApplied.current = true;
  }, [features]);

  useEffect(() => {
    if (availableTaggingCategories.length === 0) {
      setForm((current) => current.taggingCategories.length === 0
        ? current
        : {
            ...current,
            taggingCategories: [],
          });
      return;
    }

    setForm((current) => {
      const nextCategories = current.taggingCategories.filter((category) => availableTaggingCategories.includes(category));
      const savedCategoriesDoNotMatchCurrentCatalog = taggingDefaultsApplied.current
        && current.taggingCategories.length > 0
        && nextCategories.length === 0;
      const resolvedCategories = (!taggingDefaultsApplied.current || savedCategoriesDoNotMatchCurrentCatalog) && nextCategories.length === 0
        ? availableTaggingCategories
        : nextCategories;

      if (resolvedCategories.length === current.taggingCategories.length
        && resolvedCategories.every((category, index) => category === current.taggingCategories[index])) {
        return current;
      }

      return {
        ...current,
        taggingCategories: resolvedCategories,
      };
    });
    taggingDefaultsApplied.current = true;
  }, [availableTaggingCategories]);

  function patch(next) {
    setForm((current) => ({ ...current, ...next }));
  }

  function changeMediaKind(mediaKind) {
    const nextFeatures = collectCapabilityFeatures(capabilities, mediaKind, settings?.customPipelines || [], catalog);
    const nextTaggingCategories = collectTaggingCategories(catalog, mediaKind);
    taggingCategoriesTouched.current = false;
    taggingDefaultsApplied.current = nextTaggingCategories.length > 0;
    patch({
      mediaKind,
      presetId: "",
      capabilityIds: nextFeatures.map((feature) => feature.capabilityId),
      frameInterval: mediaKind === "video" ? (form.frameInterval || defaultVideoFrameInterval) : "",
      taggingCategories: nextTaggingCategories,
    });
  }

  function selectPreset(presetId) {
    const preset = runPresets.find((item) => item.presetId === presetId);
    if (!preset) {
      patch({ presetId: "", capabilityIds: features.map((feature) => feature.capabilityId) });
      return;
    }

    patch({
      presetId,
      capabilityIds: Array.isArray(preset.capabilityIds) && preset.capabilityIds.length > 0
        ? preset.capabilityIds.filter((capabilityId) => features.some((feature) => feature.capabilityId === capabilityId))
        : form.capabilityIds,
      frameInterval: preset.frameInterval || form.frameInterval,
    });
  }

  async function handleSubmit() {
    const categoriesToSkip = hasSelectedTaggingClaims && availableTaggingCategories.length > 0
      ? availableTaggingCategories.filter((category) => !effectiveTaggingCategories.includes(category))
      : [];

    const payload = {
      mediaKind: form.mediaKind,
      presetId: form.presetId || null,
      capabilityIds: form.capabilityIds,
      frameInterval: form.mediaKind === "video" && form.frameInterval !== "" ? Number(form.frameInterval) : null,
      threshold: settings.defaultThreshold ?? null,
      returnConfidence: form.mediaKind === "audio" ? null : true,
      vrVideo: false,
      categoriesToSkip,
      taggingCategories: effectiveTaggingCategories,
      loadPolicy: settings.defaultLoadPolicy || null,
      dispatchResults: settings.dispatchResultsByDefault ?? true,
      forceClaimIds: form.forceExisting ? selectedClaimIds : [],
    };

    if (hasSelectionTargets) {
      payload.entityType = selection.entityType;
      payload.entityIds = selection.entityIds;
    } else {
      payload.paths = splitList(form.pathsText);
    }

    writeRunDialogDefaults(selection, { ...form, taggingCategories: effectiveTaggingCategories });
    await onQueue(payload);
  }

  return h("div", { className: "ai-core-stack" }, [
    hasSelectionTargets
      ? h("div", { className: "ai-core-selection-summary rounded-xl border border-border bg-card p-4", key: "selection" }, [
          h("span", { className: "ai-core-label" }, selection.entityType === "image" ? "Selected images" : "Selected scenes"),
          h("strong", { className: "ai-core-selection-title" }, formatSelectionTitle(selection)),
          h("p", { className: "ai-core-muted ai-core-selection-copy" }, formatSelectionDescription(selection)),
        ])
      : h("label", { className: "ai-core-field", key: "paths" }, [
          h("span", { className: "ai-core-label" }, "Paths"),
          h("textarea", {
            className: "ai-core-textarea ai-core-code rounded-lg border border-border bg-input px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none",
            rows: 6,
            value: form.pathsText,
            onChange: (event) => patch({ pathsText: event.target.value }),
            placeholder: form.mediaKind === "video"
              ? "One video path per line"
              : form.mediaKind === "image"
                ? "One image path per line"
                : "One audio path per line",
          }),
        ]),
    h("div", { className: "ai-core-grid ai-core-grid-compact", key: "runOptions" }, [
      h("label", { className: "ai-core-field", key: "preset" }, [
        h("span", { className: "ai-core-label" }, "Run preset"),
        h("select", {
          className: "ai-core-select rounded-lg border border-border bg-input px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none",
          value: form.presetId,
          onChange: (event) => selectPreset(event.target.value),
        }, [
          h("option", { key: "standard", value: "" }, "Standard"),
          ...runPresets.map((preset) => h("option", { key: preset.presetId, value: preset.presetId }, preset.displayName)),
        ]),
      ]),
      showMediaKindField
        ? h("label", { className: "ai-core-field", key: "mediaKind" }, [
            h("span", { className: "ai-core-label" }, "Media kind"),
            h("select", {
              className: "ai-core-select rounded-lg border border-border bg-input px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none",
              value: form.mediaKind,
              onChange: (event) => changeMediaKind(event.target.value),
            }, mediaKinds.map((kind) => h("option", { key: kind.value, value: kind.value }, kind.label))),
          ])
        : null,
      form.mediaKind === "video"
        ? h("label", { className: "ai-core-field", key: "frameInterval" }, [
            h("span", { className: "ai-core-label" }, "Frame interval (seconds)"),
            h("input", {
              className: "ai-core-input rounded-lg border border-border bg-input px-3 py-2 text-sm text-foreground focus:border-accent focus:outline-none",
              type: "number",
              min: "0",
              step: "0.1",
              value: form.frameInterval,
              onChange: (event) => patch({ frameInterval: event.target.value }),
              placeholder: defaultVideoFrameInterval,
            }),
          ])
        : null,
      h("label", { className: "ai-core-force-option rounded-xl border border-border bg-card px-3 py-3 text-foreground", key: "forceExisting" }, [
        h("input", {
          className: "ai-core-checkbox",
          type: "checkbox",
          checked: form.forceExisting,
          onChange: (event) => patch({ forceExisting: event.target.checked }),
        }),
        h("span", { className: "ai-core-label" }, "Force rerun and replace stored data"),
      ]),
    ]),
    h("div", { className: "ai-core-run-summary rounded-xl border border-border bg-card p-4", key: "modelSummary" }, [
      h("div", { className: "ai-core-section-header" }, [
        h("div", null, [
          h("h4", { className: "ai-core-subtitle" }, "Model request summary"),
          h("p", { className: "ai-core-muted" }, summarizeRequestedModels(requestedModels, settings.defaultLoadPolicy || "use_loaded")),
        ]),
      ]),
      h("div", { className: "ai-core-toolbar" }, [
        h("span", { className: "ai-core-pill ai-core-pill-muted" }, `${form.capabilityIds.length} selected feature(s)`),
        h("span", { className: "ai-core-pill ai-core-pill-muted" }, `${selectedClaimIds.length} claim(s)`),
        form.forceExisting
          ? h("span", { className: "ai-core-pill ai-core-pill-warning" }, "Existing results will be rerun")
          : h("span", { className: "ai-core-pill ai-core-pill-success" }, "Reusable results will be skipped"),
      ]),
      requestedModels.length > 0
        ? h("div", { className: "ai-core-model-chip-list" }, requestedModels.map((model) => h("div", { className: "ai-core-model-chip rounded-full border border-border bg-surface text-foreground", key: model.configName }, [
            h("span", { className: "ai-core-code" }, model.configName),
            model.loaded === true ? h("span", { className: "ai-core-pill ai-core-pill-success" }, "loaded") : null,
            model.loaded === false ? h("span", { className: "ai-core-pill ai-core-pill-muted" }, "not loaded") : null,
          ])))
        : h("p", { className: "ai-core-empty" }, "Select AI features to see the model set this run will ask the server for."),
    ]),
    h("div", { className: "ai-core-section", key: "features" }, [
      h("div", { className: "ai-core-section-header" }, [
        h("div", null, [
          h("h4", { className: "ai-core-subtitle" }, "AI features"),
          h("p", { className: "ai-core-muted" }, "Choose user-facing workflows instead of detector/embedder implementation stages."),
        ]),
        features.length > 0
          ? h("div", { className: "ai-core-toolbar" }, [
              h("button", {
                className: "text-sm text-accent transition-colors hover:text-accent-hover",
                type: "button",
                onClick: () => patch({ presetId: "", capabilityIds: features.map((feature) => feature.capabilityId) }),
              }, "Select all"),
              h("button", {
                className: "text-sm text-accent transition-colors hover:text-accent-hover",
                type: "button",
                onClick: () => patch({ presetId: "", capabilityIds: [] }),
              }, "Clear"),
            ])
          : null,
      ]),
      features.length === 0
        ? h("p", { className: "ai-core-empty" }, "No installed AI features are available for this media kind.")
        : h("div", { className: "ai-core-claim-list rounded-xl border border-border bg-card" }, features.map((feature) => {
            const selected = form.capabilityIds.includes(feature.capabilityId);
            const featureClaimIds = collectSelectedClaimIds([feature, ...features], claims, [feature.capabilityId]);
            const isTaggingFeature = claims.some((claim) => featureClaimIds.includes(claim.claimId) && claim.wantCapability === "tagging");

            return h("div", { className: "ai-core-claim ai-core-claim-block bg-surface/50", key: feature.capabilityId }, [
              h("label", { className: "ai-core-claim-toggle" }, [
                h("input", {
                  className: "ai-core-checkbox",
                  type: "checkbox",
                  checked: selected,
                  onChange: (event) => patch({
                    presetId: "",
                    capabilityIds: event.target.checked
                      ? [...form.capabilityIds, feature.capabilityId]
                      : form.capabilityIds.filter((capabilityId) => capabilityId !== feature.capabilityId),
                  }),
                }),
                h("div", { className: "ai-core-claim-copy" }, [
                  h("div", { className: "ai-core-claim-title" }, feature.displayName),
                  h("div", { className: "ai-core-model-meta" }, [
                    h("span", { className: "ai-core-pill ai-core-pill-muted", key: "extension" }, feature.extensionLabel || feature.extensionId),
                    feature.customPipeline ? h("span", { className: "ai-core-pill ai-core-pill-warning", key: "custom" }, feature.pipelineName) : null,
                    feature.description ? h("span", { key: "description" }, feature.description) : null,
                  ]),
                ]),
              ]),
              selected && isTaggingFeature && availableTaggingCategories.length > 0
                ? h("div", { className: "ai-core-tagging-group" }, [
                    h("div", { className: "ai-core-tagging-header" }, [
                      h("span", { className: "ai-core-label" }, "Tag categories"),
                      h("div", { className: "ai-core-toolbar" }, [
                        h("button", {
                          className: "text-sm text-accent transition-colors hover:text-accent-hover",
                          type: "button",
                          onClick: () => {
                            taggingCategoriesTouched.current = true;
                            patch({ taggingCategories: availableTaggingCategories });
                          },
                        }, "Select all"),
                        h("button", {
                          className: "text-sm text-accent transition-colors hover:text-accent-hover",
                          type: "button",
                          onClick: () => {
                            taggingCategoriesTouched.current = true;
                            patch({ taggingCategories: [] });
                          },
                        }, "Clear"),
                      ]),
                    ]),
                    h("div", { className: "ai-core-category-grid" }, availableTaggingCategories.map((category) => h("label", { className: "ai-core-category-option rounded-full border border-border bg-surface text-foreground", key: `${feature.capabilityId}:${category}` }, [
                      h("input", {
                        className: "ai-core-checkbox",
                        type: "checkbox",
                        checked: effectiveTaggingCategories.includes(category),
                        onChange: (event) => {
                          taggingCategoriesTouched.current = true;
                          patch({
                            taggingCategories: event.target.checked
                              ? [...effectiveTaggingCategories, category]
                              : effectiveTaggingCategories.filter((value) => value !== category),
                          });
                        },
                      }),
                      h("span", null, category),
                    ]))),
                  ])
                : null,
            ]);
          })),
    ]),
    h("div", { className: "ai-core-toolbar", key: "actions" }, [
      h("div", { className: "ai-core-toolbar-spacer" }),
      message ? h("span", { className: "ai-core-muted", key: "message" }, message) : null,
      h("button", {
        className: "rounded-md bg-accent px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-60",
        type: "button",
        disabled: busy
          || (!hasSelectionTargets && pathCount === 0)
          || form.capabilityIds.length === 0
          || requestedModels.length === 0
          || (hasSelectedTaggingClaims && availableTaggingCategories.length > 0 && effectiveTaggingCategories.length === 0),
        onClick: handleSubmit,
      }, busy ? "Queueing..." : submitLabel),
    ]),
  ]);
}

function AiCoreRunDialog({ selection, onComplete }) {
  const [settings, setSettings] = useState(copySettings(null));
  const [capabilities, setCapabilities] = useState([]);
  const [catalog, setCatalog] = useState({ models: [] });
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const pointerDownOnBackdrop = useRef(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [nextSettings, capabilityEnvelope] = await Promise.all([
          api("/settings"),
          api("/capabilities"),
        ]);
        let nextCatalog = { models: [] };
        try {
          nextCatalog = await api("/models/catalog");
        } catch {
          nextCatalog = { models: [] };
        }
        if (!cancelled) {
          setSettings(copySettings(nextSettings));
          setCapabilities(capabilityEnvelope?.extensions || []);
          setCatalog(nextCatalog || { models: [] });
        }
      } catch (error) {
        if (!cancelled) {
          setMessage(error.message || "Failed to load AI Core run options.");
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

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Escape" && !busy) {
        onComplete({ cancelled: true, suppressToast: true });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [busy, onComplete]);

  async function queueSelection(payload) {
    setBusy(true);
    setMessage("");
    try {
      const queued = await api("/jobs/queue", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      onComplete(queued);
    } catch (error) {
      setMessage(error.message || "Failed to queue AI job.");
    } finally {
      setBusy(false);
    }
  }

  return h("div", {
    className: "ai-core-modal-backdrop fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4",
    onPointerDown: (event) => {
      pointerDownOnBackdrop.current = event.target === event.currentTarget;
    },
    onPointerUp: (event) => {
      if (!busy && pointerDownOnBackdrop.current && event.target === event.currentTarget) {
        onComplete({ cancelled: true, suppressToast: true });
      }
      pointerDownOnBackdrop.current = false;
    },
  }, h("div", {
    className: "ai-core-modal relative w-full rounded-xl border border-border bg-surface text-foreground shadow-xl",
    role: "dialog",
    "aria-modal": "true",
    onClick: (event) => event.stopPropagation(),
  }, [
    h("div", { className: "ai-core-modal-header border-b border-border", key: "header" }, [
      h("div", null, [
        h("h3", { className: "ai-core-title" }, formatSelectionTitle(selection)),
        h("p", { className: "ai-core-muted" }, "Choose what AI should run for this selection."),
      ]),
      h("button", {
        className: "rounded-md border border-border bg-surface px-4 py-2 text-sm text-foreground transition-colors hover:bg-surface/50 disabled:opacity-60",
        type: "button",
        disabled: busy,
        onClick: () => onComplete({ cancelled: true, suppressToast: true }),
      }, "Cancel"),
    ]),
    h("div", { className: "ai-core-modal-body", key: "body" }, loading
      ? h("p", { className: "ai-core-empty" }, "Loading AI run options...")
      : h(RunComposer, {
          capabilities,
          catalog,
          settings,
          busy,
          message,
          selection,
          submitLabel: "Run AI",
          onQueue: queueSelection,
        })),
  ]));
}

function openRunAiDialog(_action, payload) {
  const selection = normalizeSelectionPayload(payload);
  if (!selection) {
    throw new Error("Run AI requires a non-empty Scene or Image selection.");
  }

  return new Promise((resolve) => {
    const existing = document.getElementById(RUN_DIALOG_ROOT_ID);
    if (existing) {
      existing.remove();
    }

    const container = document.createElement("div");
    container.id = RUN_DIALOG_ROOT_ID;
    document.body.appendChild(container);

    const root = createRoot(container);
    const finish = (result) => {
      root.unmount();
      container.remove();
      resolve(result || { cancelled: true, suppressToast: true });
    };

    root.render(h(AiCoreRunDialog, {
      selection,
      onComplete: finish,
    }));
  });
}

function AiCoreSettingsPanel() {
  const [settings, setSettings] = useState(copySettings(null));
  const [health, setHealth] = useState(null);
  const [capabilities, setCapabilities] = useState([]);
  const [catalog, setCatalog] = useState({ models: [] });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [pipelineBusy, setPipelineBusy] = useState("");
  const [message, setMessage] = useState("");
  const [busyModel, setBusyModel] = useState("");

  async function refreshPanel() {
    const [nextSettings, nextHealth, capabilityEnvelope] = await Promise.all([api("/settings"), api("/health"), api("/capabilities")]);
    let nextCatalog = { models: [] };
    try {
      nextCatalog = await api("/models/catalog");
    } catch {
      nextCatalog = { models: [] };
    }
    setSettings(copySettings(nextSettings));
    setHealth(nextHealth);
    setCapabilities(capabilityEnvelope?.extensions || []);
    setCatalog(nextCatalog || { models: [] });
  }

  async function refreshRuntime() {
    const [nextHealth, capabilityEnvelope] = await Promise.all([api("/health"), api("/capabilities")]);
    let nextCatalog = { models: [] };
    try {
      nextCatalog = await api("/models/catalog");
    } catch {
      nextCatalog = { models: [] };
    }
    setHealth(nextHealth);
    setCapabilities(capabilityEnvelope?.extensions || []);
    setCatalog(nextCatalog || { models: [] });
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await refreshPanel();
      } catch (error) {
        if (!cancelled) {
          setMessage(error.message || "Failed to load AI Core settings.");
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

  useAutosaveSettings(
    settings,
    !loading,
    () => {
      setSaving(true);
      setMessage("");
    },
    (saved) => {
      setSaving(false);
      setSettings(saved);
      setMessage("Settings saved.");
      refreshRuntime();
    },
    (error) => {
      setSaving(false);
      setMessage(error.message || "Failed to save AI Core settings.");
    }
  );

  async function syncPipeline(pipeline) {
    setPipelineBusy(pipeline.pipelineName);
    setMessage("");
    try {
      await api("/pipelines/custom", {
        method: "POST",
        body: JSON.stringify(pipeline),
      });
      setSettings(copySettings(await api("/settings")));
      setHealth(await api("/health"));
      setMessage(`Pipeline ${pipeline.pipelineName} synced.`);
    } catch (error) {
      setMessage(error.message || "Failed to sync custom pipeline.");
    } finally {
      setPipelineBusy("");
    }
  }

  async function deletePipeline(pipelineName) {
    setPipelineBusy(pipelineName);
    setMessage("");
    try {
      await api(`/pipelines/custom/${encodeURIComponent(pipelineName)}`, { method: "DELETE" });
      setSettings(copySettings(await api("/settings")));
      setHealth(await api("/health"));
      setMessage(`Pipeline ${pipelineName} removed.`);
    } catch (error) {
      setMessage(error.message || "Failed to remove custom pipeline.");
    } finally {
      setPipelineBusy("");
    }
  }

  async function toggleLoad(configName, shouldLoad) {
    setBusyModel(configName);
    setMessage("");
    try {
      await api(shouldLoad ? "/models/load" : "/models/unload", {
        method: "POST",
        body: JSON.stringify({ models: [configName] }),
      });
      await refreshPanel();
    } catch (error) {
      setMessage(error.message || "Failed to update model state.");
    } finally {
      setBusyModel("");
    }
  }

  if (loading) {
    return h("p", { className: "ai-core-empty" }, "Loading AI Core settings...");
  }

  return h("div", { className: "ai-core-settings-panel" }, [
    h(SettingsEditor, {
      key: "editor",
      settings,
      health,
      capabilities,
      catalog,
      busy: saving,
      message,
      pipelineBusy,
      onChange: setSettings,
      onSyncPipeline: syncPipeline,
      onDeletePipeline: deletePipeline,
    }),
    h("div", { className: "ai-core-section", key: "models" }, h(ModelsPanel, {
      catalog,
      busyModel,
      onLoadToggle: toggleLoad,
      onRefresh: refreshPanel,
    })),
  ]);
}

function AiCorePage() {
  const [settings, setSettings] = useState(copySettings(null));
  const [health, setHealth] = useState(null);
  const [capabilities, setCapabilities] = useState([]);
  const [catalog, setCatalog] = useState({ models: [] });
  const [loading, setLoading] = useState(true);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [settingsMessage, setSettingsMessage] = useState("");
  const [pipelineBusy, setPipelineBusy] = useState("");
  const [queueBusy, setQueueBusy] = useState(false);
  const [queueMessage, setQueueMessage] = useState("");
  const [busyModel, setBusyModel] = useState("");

  async function refresh() {
    const [nextSettings, nextHealth, capabilityEnvelope] = await Promise.all([
      api("/settings"),
      api("/health"),
      api("/capabilities"),
    ]);
    let modelEnvelope = { models: [] };
    try {
      modelEnvelope = await api("/models/catalog");
    } catch {
      modelEnvelope = { models: [] };
    }
    setSettings(copySettings(nextSettings));
    setHealth(nextHealth);
    setCapabilities(capabilityEnvelope?.extensions || []);
    setCatalog(modelEnvelope || { models: [] });
  }

  async function refreshRuntime() {
    const [nextHealth, capabilityEnvelope] = await Promise.all([api("/health"), api("/capabilities")]);
    let modelEnvelope = { models: [] };
    try {
      modelEnvelope = await api("/models/catalog");
    } catch {
      modelEnvelope = { models: [] };
    }
    setHealth(nextHealth);
    setCapabilities(capabilityEnvelope?.extensions || []);
    setCatalog(modelEnvelope || { models: [] });
  }

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await refresh();
      } catch (error) {
        if (!cancelled) {
          setSettingsMessage(error.message || "Failed to load AI Core.");
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

  useAutosaveSettings(
    settings,
    !loading,
    () => {
      setSettingsBusy(true);
      setSettingsMessage("");
    },
    (saved) => {
      setSettingsBusy(false);
      setSettings(saved);
      setSettingsMessage("Settings saved.");
      refreshRuntime();
    },
    (error) => {
      setSettingsBusy(false);
      setSettingsMessage(error.message || "Failed to update settings.");
    }
  );

  async function toggleLoad(configName, shouldLoad) {
    setBusyModel(configName);
    try {
      await api(shouldLoad ? "/models/load" : "/models/unload", {
        method: "POST",
        body: JSON.stringify({ models: [configName] }),
      });
      await refresh();
    } catch (error) {
      setSettingsMessage(error.message || "Failed to update model state.");
    } finally {
      setBusyModel("");
    }
  }

  async function syncPipeline(pipeline) {
    setPipelineBusy(pipeline.pipelineName);
    setSettingsMessage("");
    try {
      await api("/pipelines/custom", {
        method: "POST",
        body: JSON.stringify(pipeline),
      });
      await refresh();
      setSettingsMessage(`Pipeline ${pipeline.pipelineName} synced.`);
    } catch (error) {
      setSettingsMessage(error.message || "Failed to sync custom pipeline.");
    } finally {
      setPipelineBusy("");
    }
  }

  async function deletePipeline(pipelineName) {
    setPipelineBusy(pipelineName);
    setSettingsMessage("");
    try {
      await api(`/pipelines/custom/${encodeURIComponent(pipelineName)}`, { method: "DELETE" });
      await refresh();
      setSettingsMessage(`Pipeline ${pipelineName} removed.`);
    } catch (error) {
      setSettingsMessage(error.message || "Failed to remove custom pipeline.");
    } finally {
      setPipelineBusy("");
    }
  }

  async function queueRun(payload) {
    setQueueBusy(true);
    setQueueMessage("");
    try {
      const queued = await api("/jobs/queue", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      setQueueMessage(`Queued ${queued.description || "AI job"} (${queued.jobId}).`);
    } catch (error) {
      setQueueMessage(error.message || "Failed to queue AI job.");
    } finally {
      setQueueBusy(false);
    }
  }

  if (loading) {
    return h("div", { className: "ai-core-shell" }, h("div", { className: "ai-core-card" }, h("p", { className: "ai-core-empty" }, "Loading AI Core...")));
  }

  return h("div", { className: "ai-core-shell" }, [
    h("section", { className: "ai-core-card", key: "summary" }, [
      h("div", { className: "ai-core-card-header" }, [
        h("div", null, [
          h("h2", { className: "ai-core-title" }, "AI Control"),
          h("p", { className: "ai-core-muted" }, "Manage the nsfw_ai_server connection, active models, and queued AI runs from one place."),
        ]),
        h("button", {
          className: "ai-core-button ai-core-button-secondary",
          type: "button",
          onClick: refresh,
        }, "Refresh"),
      ]),
      h("div", { className: "ai-core-card-body ai-core-grid" }, [
        h("div", { className: "ai-core-metric", key: "status" }, [
          h("span", { className: "ai-core-metric-label" }, "Server"),
          h("strong", { className: "ai-core-metric-value" }, health?.status || "unknown"),
          h("span", { className: `ai-core-pill ai-core-pill-${statusTone(health?.status)}` }, health?.serverBaseUrl || settings.serverBaseUrl),
        ]),
        h("div", { className: "ai-core-metric", key: "contributors" }, [
          h("span", { className: "ai-core-metric-label" }, "Contributor extensions"),
          h("strong", { className: "ai-core-metric-value" }, String(health?.contributorCount ?? capabilities.length)),
        ]),
        h("div", { className: "ai-core-metric", key: "models" }, [
          h("span", { className: "ai-core-metric-label" }, "Catalog models"),
          h("strong", { className: "ai-core-metric-value" }, String((catalog.models || []).length)),
        ]),
        h("div", { className: "ai-core-metric", key: "mappings" }, [
          h("span", { className: "ai-core-metric-label" }, "Path mappings"),
          h("strong", { className: "ai-core-metric-value" }, String((settings.pathMappings || []).length)),
        ]),
      ]),
    ]),
    h("section", { className: "ai-core-card", key: "settings" }, [
      h("div", { className: "ai-core-card-header" }, [
        h("div", null, [
          h("h3", { className: "ai-core-title" }, "Connection and defaults"),
          h("p", { className: "ai-core-muted" }, "This is the primary home for AI Core runtime settings."),
        ]),
      ]),
      h("div", { className: "ai-core-card-body" }, h(SettingsEditor, {
        settings,
        health,
        capabilities,
        catalog,
        busy: settingsBusy,
        message: settingsMessage,
        pipelineBusy,
        onChange: setSettings,
        onSyncPipeline: syncPipeline,
        onDeletePipeline: deletePipeline,
      })),
    ]),
    h("section", { className: "ai-core-card", key: "models" }, [
      h("div", { className: "ai-core-card-header" }, [
        h("div", null, [
          h("h3", { className: "ai-core-title" }, "Model catalog"),
        ]),
      ]),
      h("div", { className: "ai-core-card-body" }, h(ModelsPanel, {
        catalog,
        busyModel,
        onLoadToggle: toggleLoad,
        onRefresh: refresh,
      })),
    ]),
    h("section", { className: "ai-core-card", key: "run" }, [
      h("div", { className: "ai-core-card-header" }, [
        h("div", null, [
          h("h3", { className: "ai-core-title" }, "Queue AI runs"),
          h("p", { className: "ai-core-muted" }, "Compose a multi-capability run and send it through Cove's job system as one parent job."),
        ]),
      ]),
      h("div", { className: "ai-core-card-body" }, h(RunComposer, {
        capabilities,
        catalog,
        settings,
        busy: queueBusy,
        message: queueMessage,
        onQueue: queueRun,
      })),
    ]),
  ]);
}

export default {
  components: {
    AiCorePage,
    AiCoreSettingsPanel,
  },
  actionHandlers: {
    openRunAiDialog,
  },
  handlers: {
    openRunAiDialog,
  },
};