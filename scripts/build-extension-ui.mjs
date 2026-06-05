#!/usr/bin/env node
import * as esbuild from "esbuild";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const runtimeAliases = {
  react: "@cove/runtime/react",
  "react-dom": "@cove/runtime/react-dom",
  "react-dom/client": "@cove/runtime/react-dom-client",
  "react/jsx-runtime": "@cove/runtime/react-jsx-runtime",
  "react/jsx-dev-runtime": "@cove/runtime/react-jsx-dev-runtime",
  "@tanstack/react-query": "@cove/runtime/react-query",
  "lucide-react": "@cove/runtime/lucide-react",
  "@cove/runtime/components": "@cove/runtime/components",
};

const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(scriptDir, "..");
const extensionName = process.argv[2];

if (!extensionName) {
  console.error("Usage: node scripts/build-extension-ui.mjs <ExtensionName>");
  process.exit(1);
}

const extensionDir = path.join(rootDir, "extensions", extensionName);
const uiDir = path.join(extensionDir, "ui");
const distDir = path.join(extensionDir, "dist");
const entryFile = path.join(uiDir, `${extensionName}.tsx`);
const cssEntryFile = path.join(uiDir, `${extensionName}.css`);
const outFile = path.join(distDir, "bundle.js");
const cssOutFile = path.join(distDir, "bundle.css");
const sharedThemeFile = path.join(rootDir, "shared", "theme.css");
const tailwindCliEntry = path.join(rootDir, "node_modules", "@tailwindcss", "cli", "dist", "index.mjs");
const themeScopeSelector = ":where(.ai-core-shell,.ai-core-settings-panel,.ai-core-modal-backdrop,.ai-tagging-settings-panel,.ai-faces-settings-panel)";

function findMatchingBrace(css, openIndex) {
  let depth = 1;
  let quote = "";

  for (let index = openIndex + 1; index < css.length; index += 1) {
    const char = css[index];
    const previous = css[index - 1];

    if (quote) {
      if (char === quote && previous !== "\\") {
        quote = "";
      }
      continue;
    }

    if (char === "\"" || char === "'") {
      quote = char;
      continue;
    }

    if (char === "{") {
      depth += 1;
      continue;
    }

    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return index;
      }
    }
  }

  throw new Error("Unable to find matching CSS brace.");
}

function splitSelectorList(selectorText) {
  const selectors = [];
  let depth = 0;
  let quote = "";
  let start = 0;

  for (let index = 0; index < selectorText.length; index += 1) {
    const char = selectorText[index];
    const previous = selectorText[index - 1];

    if (quote) {
      if (char === quote && previous !== "\\") {
        quote = "";
      }
      continue;
    }

    if (char === "\"" || char === "'") {
      quote = char;
      continue;
    }

    if (char === "(" || char === "[") {
      depth += 1;
      continue;
    }

    if (char === ")" || char === "]") {
      depth = Math.max(0, depth - 1);
      continue;
    }

    if (char === "," && depth === 0) {
      selectors.push(selectorText.slice(start, index));
      start = index + 1;
    }
  }

  selectors.push(selectorText.slice(start));
  return selectors;
}

function scopeSelector(selector) {
  const trimmed = selector.trim();
  if (!trimmed) {
    return [];
  }

  if (trimmed === ":root" || trimmed === ":host") {
    return [themeScopeSelector];
  }

  if (trimmed === "*") {
    return [themeScopeSelector, `${themeScopeSelector} *`];
  }

  if (trimmed === ":before" || trimmed === "::before") {
    return [`${themeScopeSelector}:before`, `${themeScopeSelector} :before`];
  }

  if (trimmed === ":after" || trimmed === "::after") {
    return [`${themeScopeSelector}:after`, `${themeScopeSelector} :after`];
  }

  if (trimmed === "::backdrop") {
    return [`${themeScopeSelector}::backdrop`];
  }

  return [`${themeScopeSelector}${trimmed}`, `${themeScopeSelector} ${trimmed}`];
}

function scopeSelectorPrelude(prelude) {
  const leading = prelude.match(/^\s*/)?.[0] ?? "";
  const trailing = prelude.match(/\s*$/)?.[0] ?? "";
  const selectorText = prelude.trim();
  const scopedSelectors = new Set();

  splitSelectorList(selectorText)
    .flatMap(scopeSelector)
    .forEach((selector) => scopedSelectors.add(selector));

  return `${leading}${Array.from(scopedSelectors).join(",")}${trailing}`;
}

function scopeCssSelectors(css, start = 0, end = css.length) {
  let output = "";
  let cursor = start;

  while (cursor < end) {
    const openIndex = css.indexOf("{", cursor);
    if (openIndex < 0 || openIndex >= end) {
      output += css.slice(cursor, end);
      break;
    }

    const closeIndex = findMatchingBrace(css, openIndex);
    const prelude = css.slice(cursor, openIndex);
    const trimmedPrelude = prelude.trim();
    const innerStart = openIndex + 1;
    const innerEnd = closeIndex;

    if (trimmedPrelude.startsWith("@")) {
      const shouldRecurse = /^@(media|supports|layer|container)\b/.test(trimmedPrelude);
      output += prelude;
      output += "{";
      output += shouldRecurse
        ? scopeCssSelectors(css, innerStart, innerEnd)
        : css.slice(innerStart, innerEnd);
      output += "}";
    } else {
      output += scopeSelectorPrelude(prelude);
      output += "{";
      output += css.slice(innerStart, innerEnd);
      output += "}";
    }

    cursor = closeIndex + 1;
  }

  return output;
}

async function assertFileExists(filePath, label) {
  try {
    await fs.access(filePath);
  } catch {
    throw new Error(`${label} not found: ${filePath}`);
  }
}

async function buildJs() {
  await assertFileExists(entryFile, "UI entry file");
  await esbuild.build({
    entryPoints: [entryFile],
    bundle: true,
    format: "esm",
    outfile: outFile,
    alias: runtimeAliases,
    external: Object.values(runtimeAliases),
    jsx: "automatic",
    jsxImportSource: "react",
    splitting: false,
    sourcemap: false,
    minify: true,
  });
  console.log(`Built ${extensionName} UI bundle: ${outFile}`);
}

async function buildCss() {
  await assertFileExists(sharedThemeFile, "Shared UI theme file");

  const tmpDir = path.join(extensionDir, "obj", "ui-build");
  await fs.mkdir(tmpDir, { recursive: true });
  const tmpCss = path.join(tmpDir, "tailwind-entry.css");
  const relativeTheme = path.relative(tmpDir, sharedThemeFile).replace(/\\/g, "/");
  const relativeUiDir = path.relative(tmpDir, uiDir).replace(/\\/g, "/");
  const imports = [`@import "${relativeTheme}";`];

  try {
    await fs.access(cssEntryFile);
    const relativeCssEntry = path.relative(tmpDir, cssEntryFile).replace(/\\/g, "/");
    imports.push(`@import "${relativeCssEntry}";`);
  } catch {
    // Tailwind utility-only extensions do not need an authored CSS file.
  }

  imports.push(`@source "${relativeUiDir}";`);
  await fs.writeFile(tmpCss, `${imports.join("\n")}\n`);

  try {
    await execFileAsync(process.execPath, [tailwindCliEntry, "--input", tmpCss, "--output", cssOutFile, "--minify"], {
      cwd: rootDir,
      windowsHide: true,
    });
    const css = await fs.readFile(cssOutFile, "utf8");
    await fs.writeFile(cssOutFile, scopeCssSelectors(css));
  } finally {
    await fs.unlink(tmpCss).catch(() => {});
  }

  console.log(`Built ${extensionName} CSS bundle: ${cssOutFile}`);
}

async function build() {
  try {
    await fs.mkdir(distDir, { recursive: true });
    await Promise.all([buildJs(), buildCss()]);
  } catch (error) {
    console.error(`Failed to build ${extensionName} UI:`, error);
    process.exit(1);
  }
}

build();
