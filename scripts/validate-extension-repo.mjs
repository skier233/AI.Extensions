import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const root = path.resolve(import.meta.dirname, "..");
const catalogPath = path.join(root, "extensions", "catalog.json");
const buildPropsPath = path.join(root, "Directory.Build.props");
const errors = [];

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8").replace(/^﻿/, ""));
}

function isLowerKebab(value) {
  return value === value.toLowerCase() && !value.includes(" ");
}

function readMsBuildProperties(filePath, inherited = {}) {
  if (!fs.existsSync(filePath)) return { ...inherited };

  const props = { ...inherited };
  const content = fs.readFileSync(filePath, "utf8");
  const pattern = /<([A-Za-z_][A-Za-z0-9_.-]*)(?:\s+[^>]*)?>([^<]*)<\/\1>/g;
  for (const match of content.matchAll(pattern)) {
    const [, name, rawValue] = match;
    const value = rawValue.trim().replace(/\$\(([^)]+)\)/g, (_, propertyName) => props[propertyName] ?? `$(${propertyName})`);
    props[name] = value;
  }

  return props;
}

function parseVersion(value) {
  if (typeof value !== "string") return null;
  const match = value.match(/^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+[0-9A-Za-z.-]+)?$/);
  if (!match) return null;
  return {
    core: match.slice(1, 4).map(part => Number.parseInt(part, 10)),
    prerelease: match[4] ? match[4].split(".") : [],
  };
}

// SemVer 2.0 precedence of prerelease identifiers: a release sorts after its prereleases; numeric identifiers
// compare numerically and sort before alphanumeric ones; a longer list wins when every shared identifier is equal.
function comparePrerelease(left, right) {
  if (left.length === 0 || right.length === 0) return right.length - left.length;

  for (let i = 0; i < Math.min(left.length, right.length); i++) {
    const leftNumeric = /^\d+$/.test(left[i]);
    const rightNumeric = /^\d+$/.test(right[i]);
    if (leftNumeric && rightNumeric) {
      const difference = Number(left[i]) - Number(right[i]);
      if (difference !== 0) return difference;
    } else if (leftNumeric !== rightNumeric) {
      return leftNumeric ? -1 : 1;
    } else if (left[i] !== right[i]) {
      return left[i] < right[i] ? -1 : 1;
    }
  }

  return left.length - right.length;
}

function compareVersions(left, right) {
  const leftVersion = parseVersion(left);
  const rightVersion = parseVersion(right);
  if (!leftVersion || !rightVersion) return null;

  for (let i = 0; i < 3; i++) {
    if (leftVersion.core[i] !== rightVersion.core[i]) return leftVersion.core[i] - rightVersion.core[i];
  }

  return comparePrerelease(leftVersion.prerelease, rightVersion.prerelease);
}

function validateVersionFloor(label, field, value, minimum) {
  if (!value) {
    errors.push(label + ": " + field + " is missing");
    return;
  }

  const comparison = compareVersions(value, minimum);
  if (comparison == null) {
    errors.push(label + ": " + field + " must be a semantic version, found " + value);
  } else if (comparison < 0) {
    errors.push(label + ": " + field + " " + value + " is below repo CoveMinVersion " + minimum);
  }
}

function validateExternalDependencies(extensionId, manifest) {
  if (manifest.externalDependencies == null) return;
  if (!Array.isArray(manifest.externalDependencies)) {
    errors.push(extensionId + ": extension.json externalDependencies must be an array");
    return;
  }

  for (const dependency of manifest.externalDependencies) {
    if (!dependency?.id) errors.push(extensionId + ": external dependency missing id");
    if (!dependency?.name) errors.push(extensionId + ": external dependency missing name");
    if (Object.prototype.hasOwnProperty.call(dependency, "optional")) {
      errors.push(extensionId + ": external dependency uses legacy optional; use required");
    }
    if (Object.prototype.hasOwnProperty.call(dependency, "settingsKey")) {
      errors.push(extensionId + ": external dependency uses legacy settingsKey; use configurationKeys");
    }
    if (dependency.configurationKeys != null && !Array.isArray(dependency.configurationKeys)) {
      errors.push(extensionId + ": external dependency configurationKeys must be an array");
    }
  }
}

function validateSettings(extensionId, manifest) {
  if (manifest.settings == null) return;
  if (!Array.isArray(manifest.settings)) {
    errors.push(extensionId + ": extension.json settings must be an array");
    return;
  }

  for (const setting of manifest.settings) {
    if (!setting?.name) errors.push(extensionId + ": setting missing name");
    if (Object.prototype.hasOwnProperty.call(setting, "key")) {
      errors.push(extensionId + ": setting uses legacy key; use name");
    }
    if (Object.prototype.hasOwnProperty.call(setting, "label")) {
      errors.push(extensionId + ": setting uses legacy label; use displayName");
    }
    if (Object.prototype.hasOwnProperty.call(setting, "defaultValue")) {
      errors.push(extensionId + ": setting uses legacy defaultValue; remove it from extension.json");
    }
    if (Object.prototype.hasOwnProperty.call(setting, "scope")) {
      errors.push(extensionId + ": setting uses legacy scope; remove it from extension.json");
    }
  }
}

const catalog = readJson(catalogPath);
const entries = Array.isArray(catalog.extensions) ? catalog.extensions : [];
const buildProps = readMsBuildProperties(buildPropsPath);
const coveMinVersion = buildProps.CoveMinVersion;

if (!catalog.schemaVersion) errors.push("extensions/catalog.json missing schemaVersion");
if (entries.length === 0) errors.push("extensions/catalog.json has no extensions");
if (!coveMinVersion) errors.push("Directory.Build.props missing CoveMinVersion");
if (coveMinVersion) {
  validateVersionFloor("Directory.Build.props", "CoveSdkVersion", buildProps.CoveSdkVersion, coveMinVersion);
  validateVersionFloor("Directory.Build.props", "CoveCoreVersion", buildProps.CoveCoreVersion, coveMinVersion);
}

const ids = new Set();
const tagPrefixes = new Set();
for (const entry of entries) {
  for (const field of ["name", "id", "path", "tagPrefix"]) {
    if (!entry[field]) errors.push((entry.id ?? entry.name ?? "catalog entry") + ": missing " + field);
  }

  if (entry.id && ids.has(entry.id)) errors.push(entry.id + ": duplicate extension id");
  if (entry.id) ids.add(entry.id);

  if (entry.tagPrefix && tagPrefixes.has(entry.tagPrefix)) errors.push(entry.id + ": duplicate tagPrefix " + entry.tagPrefix);
  if (entry.tagPrefix) tagPrefixes.add(entry.tagPrefix);
  if (entry.tagPrefix && !entry.tagPrefix.endsWith("/")) errors.push(entry.id + ": tagPrefix must end with /");

  const extensionDir = path.join(root, entry.path ?? "");
  const manifestPath = path.join(extensionDir, "extension.json");
  const projectPath = path.join(extensionDir, entry.name + ".csproj");
  const isManifestOnly = entry.manifestOnly === true;

  if (!fs.existsSync(extensionDir)) {
    errors.push(entry.id + ": path does not exist: " + entry.path);
    continue;
  }
  if (!fs.existsSync(manifestPath)) {
    errors.push(entry.id + ": missing extension.json at " + entry.path);
    continue;
  }
  if (!isManifestOnly && !fs.existsSync(projectPath)) {
    errors.push(entry.id + ": missing project " + entry.name + ".csproj at " + entry.path);
  }

  const manifest = readJson(manifestPath);
  if (manifest.id !== entry.id) errors.push(entry.id + ": catalog id does not match extension.json id " + manifest.id);
  if (!manifest.version) errors.push(entry.id + ": extension.json missing version");
  if (coveMinVersion) validateVersionFloor(entry.id, "extension.json minCoveVersion", manifest.minCoveVersion, coveMinVersion);
  if (!isManifestOnly && !manifest.entryDll) errors.push(entry.id + ": extension.json missing entryDll");
  if (isManifestOnly && manifest.entryDll) errors.push(entry.id + ": manifestOnly entry must not declare entryDll");
  if (isManifestOnly && !["bundle", "scraper-pack"].includes(manifest.kind)) {
    errors.push(entry.id + ": manifestOnly entries must use kind=bundle or kind=scraper-pack");
  }
  if (!manifest.url) errors.push(entry.id + ": extension.json missing url");
  if (!Array.isArray(manifest.categories) || manifest.categories.length === 0) {
    errors.push(entry.id + ": extension.json missing categories");
  } else {
    for (const category of manifest.categories) {
      if (!isLowerKebab(category)) errors.push(entry.id + ": category must be lowercase kebab-case: " + category);
    }
  }

  // A project may compile against a newer Cove than the repo-wide floor (AI Shots does), but never against a newer
  // Cove than its manifest admits: the host would load it into a Cove missing the API it was built against.
  if (!isManifestOnly && fs.existsSync(projectPath) && manifest.minCoveVersion) {
    const projectProps = readMsBuildProperties(projectPath, buildProps);
    for (const field of ["CoveSdkVersion", "CoveCoreVersion"]) {
      const value = projectProps[field];
      const comparison = value ? compareVersions(value, manifest.minCoveVersion) : null;
      if (comparison != null && comparison > 0) {
        errors.push(entry.id + ": " + entry.name + ".csproj compiles against Cove " + value + " (" + field + ") but extension.json minCoveVersion is " + manifest.minCoveVersion);
      }
    }
  }

  validateExternalDependencies(entry.id, manifest);
  validateSettings(entry.id, manifest);
}

if (errors.length > 0) {
  for (const error of errors) console.error("ERROR: " + error);
  process.exit(1);
}

console.log("Validated " + entries.length + " extension catalog entries.");
