// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Bridge to the Rust inference path compiled to wasm/WebGPU.
//
// The wasm-bindgen `--target web` package (glue JS + `.wasm`) is staged as a
// static asset under `public/pkg/` by `make web/wasm`. We load the glue with a
// *bundler-ignored* dynamic import so neither webpack nor turbopack tries to
// resolve/bundle it: the browser loads it as a real ESM module from `/pkg/…`,
// its `import.meta.url` resolves there, and the default `init()` then fetches
// the sibling `…_bg.wasm` with no path juggling or bundler wasm config. This
// keeps the app a pure static export, exactly like the old Vite build.

export interface RolloutResult {
  t: number[];
  setpoint: number[];
  model_y: number[];
  model_u: number[];
  model_mse: number;
  oracle_y: number[];
  oracle_u: number[];
  oracle_mse: number;
  tau: number;
  gain: number;
  steps: number;
}

interface PkgModule {
  /** wasm-bindgen default init (also installs the Rust panic hook). */
  default: (moduleOrPath?: unknown) => Promise<unknown>;
  /** Run both closed loops in wasm/WebGPU; returns a JSON string. */
  rollout_compare: (
    weights: Uint8Array,
    tau: number,
    gain: number,
    steps: number,
  ) => Promise<string>;
}

let modPromise: Promise<PkgModule> | null = null;
let weightsCache: Uint8Array | null = null;

/** Load + initialise the wasm module exactly once (installs the panic hook). */
async function ensureModule(): Promise<PkgModule> {
  if (!modPromise) {
    modPromise = (async () => {
      // The specifier is an absolute URL resolved at runtime against the page
      // origin (served from public/pkg) — there is no such module in the source
      // tree, so TS can't resolve it and the bundler must skip it. Keep it a
      // single-line string literal so the webpack/turbopack ignore comments are
      // honored (magic comments require a literal, not a variable).
      // @ts-expect-error -- runtime-only module, see note above
      const mod = (await import(/* webpackIgnore: true */ /* turbopackIgnore: true */ '/pkg/tiny_sparse_moe_wgsl.js')) as unknown as PkgModule;
      await mod.default();
      return mod;
    })();
  }
  return modPromise;
}

/** Fetch + cache the weights container bytes. */
async function loadWeights(): Promise<Uint8Array> {
  if (weightsCache) return weightsCache;
  const resp = await fetch('/moe_pid.safetensors');
  if (!resp.ok) {
    throw new Error(`failed to fetch /moe_pid.safetensors — HTTP ${resp.status}`);
  }
  weightsCache = new Uint8Array(await resp.arrayBuffer());
  return weightsCache;
}

/** Run BOTH closed loops in wasm/WebGPU and return the parsed time series. */
export async function runRollout(
  tau: number,
  gain: number,
  steps: number,
): Promise<RolloutResult> {
  const mod = await ensureModule();
  const weights = await loadWeights();
  const json = await mod.rollout_compare(weights, tau, gain, steps);
  return JSON.parse(json) as RolloutResult;
}
