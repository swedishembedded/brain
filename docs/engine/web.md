# Browser (WebGPU) demo — model vs. PID

A Next.js (App Router) app (`web/app/`) runs the PID control transformer
**entirely in the browser** (Rust → wasm → WebGPU) and compares it against the
per-plant PID oracle on the same plant and setpoint schedule. The native CLI
(`moe`) is unchanged; this is an additive, feature-gated build.

> **Status: unverified-here.** No browser/GPU exists in the build environment,
> so the wasm has been *built* (`npm run build` is green) but not *run*.
> Validating actual inference + the charts requires a real WebGPU browser
> (Chrome/Edge 113+, or Firefox/Safari with WebGPU enabled), served over
> `http(s)://localhost` (a secure context; not `file://`).

## Quick start

```sh
cd moe-rs
make web/dev      # builds the wasm, stages weights, runs the Next.js dev server
# open http://localhost:5173 in a WebGPU browser
```

`make web/dev`:
1. compiles the Rust inference path to wasm + wasm-bindgen JS into
   `web/app/public/pkg` (`cargo build --target wasm32 --features webgpu`),
2. copies `moe_pid.safetensors` into `web/app/public/` (train one first with
   `./target/release/moe pid train … --out moe_pid.safetensors` if it's missing),
3. installs npm deps if needed and starts the Next.js dev server.

Other targets: `make web/build` (static export into `web/app/out`),
`make web/wasm` (just rebuild the bindings), `make web/clean`. Prerequisites:
`rustup target add wasm32-unknown-unknown` and
`cargo install wasm-bindgen-cli --version 0.2.125` (must match the locked
`wasm-bindgen` crate), plus Node 20+.

## How the comparison stays honest

All physics and control run in **validated Rust**, exposed to JS as one call:

```ts
rollout_compare(weights: Uint8Array, tau, gain, steps): Promise<string>  // JSON
// -> { t, setpoint, model_y, model_u, model_mse,
//                    oracle_y, oracle_u, oracle_mse, tau, gain, steps }
```

The model loop drives the plant purely from the transformer's decisions
(event → DECIDE → effect fed back); the oracle loop uses that plant's
pole-placed velocity-PI. The Next.js app only *renders* what Rust returns — no
plant/PID math is re-implemented in JS, so the model-vs-PID comparison matches
the native rollout. The app lets you move `tau`/`gain` across the training grid
and the interpolated validation points to see generalization live.

See `web/app/README.md` for the app internals. The lower-level wasm runtime
notes (async device init, in-memory checkpoint parse) live in `src/web.rs`,
`src/gpu.rs`, and `src/checkpoint.rs`.
