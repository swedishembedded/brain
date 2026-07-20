# splat — workstream ledger

2026-07-19, branch feat/world-models. Fresh crate; all tests CPU-first with
gated wgpu duplicates (`MOE_SKIP_GPU_TESTS`).

## Done

- **R0 primitives**: `scan_block`/`scan_add` (recursive 256-block exclusive
  scan, ≤16M), `sort_hist`/`sort_scatter` (stable LSD radix pairs,
  per-256-chunk private ranking, column-major histograms — one scan serves
  global digit bases AND per-chunk offsets). Brain's first device sort/scan;
  no atomics, no barriers. Property-tested vs Rust references (sizes 1…4M,
  stability via payload order) on CPU + wgpu.
- **R1**: `splat_project` (gsplat-parity EWA: FOV-clamped Jacobian, +eps2d
  blur & compensation, opacity-aware 3.33σ radius, full cull set),
  `splat_naive` per-pixel oracle, pure-Rust `reference.rs` oracle, Inria PLY
  reader/writer (activations applied/inverted, `f_rest` preserved). Analytic
  single-gaussian goldens; device==oracle ≤2e-4 CPU / 2e-3 wgpu.
- **R2 tiled pipeline**: count → scan → emit (32-bit `tile<<depth_bits |
  truncated-IEEE-depth` keys, 4 radix passes) → ranges → `splat_rasterize`
  (tile = idx/64, 4 px/thread, coherent range walk, T≤1e-4 break with
  gsplat's exclude-terminator semantics) → rgba8 pack. Tiled == oracle within
  1/255 (procedural scenes, edge tiles, behind-camera); **bit-identical under
  input shuffling**; instance-cap overflow degrades gracefully (clamped +
  reported). Default cap `max(8N, 1M)` — the 1M floor matters more than the
  multiple for large screen-space gaussians.
- **R3 viewer**: `brain splat view` — WASD+mouse fly cam (wm-display gained
  SDL relative-mouse + `Input::mouse_dx/dy` + Shift/C keys + Screenshot/
  ToggleMouse UX keys), quality levels 1×/½/¼ at constant window size, depth
  toggle, HUD, screenshots. Headless smoke via `SDL_VIDEODRIVER=dummy`.
- **T2 backward**: `splat_bwd_count/emit` (per-pixel replay; suffix-color
  trick for v_alpha), record sort by gaussian id (reuses the radix kernels;
  segment ranges = `splat_tile_ranges` with depth_bits=0),
  `splat_grad_reduce`, `splat_project_bwd` (full EWA VJP incl. clamped-FOV
  Jacobian terms and the quat-normalization chain). **Gradcheck vs a
  committed float64 torch-autograd golden** (`tools/splat_dump_gradcheck.py`)
  — all 84 grads within 5e-3. Finite differences are deliberately NOT the
  oracle: the 1/255 truncation boundary biases them (verified 2× off where
  autograd confirms our analytic value). `splat fit` (AdamW on packed
  geometry + opacity + colors, projected-gradient clamps) recovers a
  perturbed 12-gaussian scene: MSE ↓ >65% in 120 iters. Limitation:
  backward assumes `antialiased=false` (compensation chain unmodeled) and
  un-chunked per-view grads.

## Measured (steady-state, `splat render --bench 8`, 1M-gaussian shell,
## ~2M sorted instances — a heavy-overdraw worst case)

| Resolution | CPU JIT (22 threads) | Intel iGPU (wgpu) |
|---|---|---|
| 1280×720 | ~3.0 s/frame | ~0.76 s/frame (~1.3 fps) |
| 640×360 | ~2.9 s/frame | ~0.31 s/frame (~3 fps) |
| 320×180 | ~3.4 s/frame | ~0.33 s/frame |

CPU is resolution-insensitive → sort/dispatch-bound; iGPU floors ~0.3 s →
sort + submission overhead dominates below 360p. The plan's 30 fps @720p/1M
target assumed a discrete GPU; this machine's realistic interactive path is
the viewer's ¼-resolution quality level on scenes ≤ a few hundred k
gaussians (a 1-frame WorldMirror scene is ~268k pre-prune). Small scenes:
301 gaussians @720p ≈ 73 ms CPU / 280 ms iGPU single-shot.

## Remaining

- R4 optimization pass: BRAIN_PROFILE stage breakdown, radix chunk tuning,
  per-tile sort experiments, pipelined present (render N while presenting
  N-1 — hooks exist via `flush()`).
- `prune::voxel_merge` landed: reference `prune_gs` voxel merge (weighted
  means/scales/colors, Σw²/Σw opacity, renormalized quats), wired as
  `brain mirror infer|demo --prune VOXEL`.
- SH degree 1–3 color kernel; `.splat`/`.spz` IO; densify/prune-in-training for full
  from-scratch scene training; accumulate-mode attention bwd for chunked fits.
