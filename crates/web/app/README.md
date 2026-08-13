# PID Transformer — Model vs. Oracle (WebGPU demo)

A polished **Next.js (App Router)** demo that runs a tiny transformer's PID
controller and an analytically-tuned PID *oracle* in two closed loops, both
driving the same first-order plant on the same setpoint schedule. Everything —
plant physics and both control laws — executes in Rust compiled to WebAssembly +
WebGPU; the page only renders the time series returned by `rollout_compare()`.

UI is built with [shadcn/ui](https://ui.shadcn.com)-style primitives (Radix +
Tailwind, in `components/ui`) over a small bespoke design system in
`app/globals.css`; charts use [recharts](https://recharts.org).

## Requirements

- A **WebGPU-capable browser** (Chrome/Edge 113+, or Firefox/Safari with WebGPU
  enabled). The page shows a friendly notice if `navigator.gpu` is missing.
- Node 20+ + npm (for building/serving).

## Run

```bash
npm install
npm run dev      # Next.js dev server, default http://localhost:3000
```

Static production export + local preview:

```bash
npm run build    # type-checks then exports a static site into ./out
npx serve out    # or any static file server
```

> The whole app is a static client-side bundle (`output: 'export'` in
> `next.config.mjs`) — there is no server runtime, exactly like the old Vite
> build. `make web/dev` / `make web/build` (in `crates/web/`) wrap these and also
> rebuild the wasm.

## How it works

- The wasm-bindgen `--target web` package (`tiny_sparse_moe_wgsl.js` + `.wasm`)
  is staged as a **static asset** under `public/pkg/` by `make web/wasm`.
- `lib/wasm.ts` loads that glue with a *bundler-ignored* dynamic import
  (`webpackIgnore` / `turbopackIgnore`), so neither webpack nor Turbopack tries
  to bundle it: the browser loads it as a real ESM module from `/pkg/…`, its
  `import.meta.url` resolves there, and the default `init()` fetches the sibling
  `…_bg.wasm`. This needs no bundler wasm configuration and keeps the app a pure
  static export.
- After init (which also installs the Rust panic hook), it fetches
  `/moe_pid.safetensors` and calls `rollout_compare(weights, tau, gain, steps)`,
  which returns a JSON string parsed into the series the charts render.
- The weights container is served from `public/moe_pid.safetensors`.

## Layout

```
app/         layout (fonts + metadata), page, globals.css (design system)
components/   demo.tsx (the client app) + ui/ (shadcn-style primitives)
lib/          wasm.ts (Rust bridge), grid.ts (train/val grid), utils.ts (cn)
public/       moe_pid.safetensors, pkg/ (generated wasm — git-ignored)
```

## Plant grid

The transformer was trained on a grid of plants and validated on held-out
interpolation points:

- **Training nodes:** τ ∈ {0.45, 0.65, 0.85}, gain ∈ {1.00, 1.25, 1.50}
- **Validation (unseen):** τ ∈ {0.55, 0.75}, gain ∈ {1.125, 1.375}

A live badge above the sliders reports whether the current (τ, gain) is a
training node, a held-out validation point, or off-grid.

## Notes

- No plant or PID math is re-implemented in JavaScript — the demo is a thin
  renderer over the Rust rollout.
- Runtime behavior (WebGPU execution) cannot be exercised in a headless
  environment; `npm run build` validates the wasm import, bundling, and types.
