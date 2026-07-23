# 3D Gaussian Splatting in brain (`crates/splat`)

A from-scratch tiled 3DGS rasterizer as WGSL compute — atomic-free and
barrier-free by construction, so the identical kernel source runs on wgpu
(any GPU) and on the CPU Cranelift JIT. Plus Inria `.ply` scene IO and the
interactive viewer.

```
brain splat info   scene.ply
brain splat render scene.ply --out img.ppm [--eye x,y,z --target x,y,z]
brain splat view   scene.ply         # WASD+mouse fly-through
```

Viewer controls: WASD move, Space/C up/down, Shift sprint, `m` mouse-look,
arrows look, `[`/`]` render quality (1×, ½, ¼ resolution), `v` color/depth,
`p` screenshot, Enter reset, Esc quit.

## Pipeline (gsplat-parity forward)

`project` (EWA perspective with FOV clamping, +0.3 anti-alias blur &
compensation, opacity-aware 3.33σ radius, full cull set) → `tile_count` →
**generic exclusive scan** → `emit` (32-bit keys: `tile_id << depth_bits |
truncated IEEE depth bits`) → **generic LSD radix sort** (per-256-chunk
private ranking, column-major histograms, one scan per pass, stable) →
`tile_ranges` (neighbor compare) → `rasterize` (one 64-thread workgroup per
16×16 tile, 4 px/thread, front-to-back, T≤1e-4 early-out) → `pack_rgba8`.

The scan/sort primitives (`scan_block`/`scan_add`/`sort_hist`/`sort_scatter`)
are brain's first device sort machinery and are model-agnostic — reuse them.

## Correctness

- analytic single-gaussian goldens (closed-form conic/alpha);
- pure-Rust oracle (`reference.rs`) == device naive path == tiled pipeline
  (±1/255) on procedural scenes, CPU and wgpu;
- input-order invariance (bit-identical after shuffling);
- PLY round-trip; rgba8 pack; graceful instance-cap overflow.

No CUDA/gsplat golden on this machine (no NVIDIA GPU): the oracle chain is
analytic → reference.rs → device parity, with all formulas transcribed from
the gsplat CUDA source.

## Data model

Host `Splats` are post-activation (linear scales, [0,1] opacity, decoded
RGB); the PLY reader/writer applies/inverts the Inria on-disk activations
(log-scale, logit-opacity, SH-DC). SH degree 0 renders today; `f_rest_*` is
parsed and preserved for round-trips (rendering higher orders is additive —
colors are a separate buffer).

Camera: OpenCV pinhole (+X right, +Y down, +Z forward), `c2w` rigid,
`viewmat = inv(c2w)`; near 0.01, eps2d 0.3, tile 16.
