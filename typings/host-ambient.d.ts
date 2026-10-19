// Ambient declarations needed only to TYPE-CHECK against the sibling cove checkout's sources.
//
// Cove's UI is built by Vite, which supplies `import.meta.env` via `vite/client`. We aren't a Vite project and
// don't depend on Vite, so we declare the shape ourselves rather than pull in a build tool we never run. Nothing
// here reaches the shipped bundle — it exists purely so `npm run typecheck` sees the same types cove's own build
// does.
interface ImportMetaEnv {
  readonly DEV: boolean;
  readonly PROD: boolean;
  readonly MODE: string;
  readonly [key: string]: unknown;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
  /** Vite's compile-time directory import. Cove uses it to bundle tutorial storyboards. */
  glob(pattern: string, options?: Record<string, unknown>): Record<string, unknown>;
}
