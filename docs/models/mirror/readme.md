# WorldMirror-2 in brain (`crates/mirror`)

Multi-view 3D reconstruction: N photos → a navigable **3D Gaussian Splatting
scene** plus per-frame depth/normal/confidence maps and per-frame cameras —
the HY-World 2.0 "WorldMirror-2.0" 1.26B feed-forward transformer, imported
exactly from the reference checkpoint and re-implemented from scratch on the
brain engine (same WGSL kernels on wgpu and the CPU JIT).

```
brain mirror import <model.safetensors|hf_dir> --out out/mirror.safetensors
brain mirror infer  --weights out/mirror.safetensors --images photos/ --maps
brain splat view    out/mirror/scene.ply            # fly through it (WASD+mouse)
brain mirror demo   --weights out/mirror.safetensors --images photos/   # both at once
```

Or via make: `make mirror/import MIRROR_CKPT=…`, `make mirror/infer
MIRROR_IMAGES=…`, `make mirror/demo MIRROR_IMAGES=…`.

## Model

- **Encoder**: full DINOv2 ViT-L/14-reg per frame (24 blocks, LayerScale,
  GELU-erf, LN eps 1e-6) → 1369 patch tokens at 518×518.
- **Trunk**: 24 alternating levels of frame-attention and global-attention
  blocks (dim 1024, 16 heads, per-head-dim LayerNorm QK-norm, DINOv3-style
  normalized 2D RoPE on a 38×38 grid with specials at (0,0)); per-frame token
  layout `[cam, reg×4, pose=0, ray=0, patch×1369]`. Taps at levels
  [4,11,17,23] concatenate frame‖global → 2048.
- **Heads**: four DPT heads (depth+conf+mask, points, normals, GS-depth) and
  the gaussian-parameter convs (12ch/pixel: quat, scale, opacity, SH-DC
  residual, merge weight); an iterative camera head (4 refinement steps →
  `[t, quat_xyzw, fov_v, fov_u]` per frame).
- **Assembly** (host): gaussian means = GS-depth back-projected through the
  predicted camera; one gaussian per source pixel; colors = SH-DC residual +
  image RGB.

Everything is fp32 and runs on `--device cpu|gpu` like any brain model. The
whole forward is recorded once (`crates/mirror/src/model.rs`) and reuses the
shared ViT block builder `model::vit` (also used by the camera head) plus the
`crates/splat` renderer for display.

## Parity (vs the PyTorch reference, `tools/goldens/mirror_dump_reference.py`)

| Gate | What | Status |
|---|---|---|
| T0 | 1545-tensor param layout vs checkpoint header | exact |
| T1 | PIL fixed-point bicubic preprocessing | bit-exact |
| T2 | DINOv2 patch tokens | ≤2e-4 |
| T4 | all 4 trunk taps (48 attention blocks) | ≤3e-3 |
| T5 | dense-head maps + gaussian params + camera vector | see STATUS.md |

Gated tests: `cargo test -p brain-mirror` (device-free T0/T1 always;
T2/T4/T5 need `MIRROR_CKPT=<model.safetensors>` and ~10 min on CPU).

## Limitations (current)

- Square inputs only (the native 518×518 / 37×37 grid). Rectangular inputs
  need DINOv2 pos-embed interpolation — planned follow-up; the CLI says so
  loudly rather than guessing.
- Prior injection (camera/depth prompts) is allocated but not exposed in the
  CLI; the no-prior path matches the reference exactly (zero pose/ray tokens).
- The reference's ONNX sky-segmentation filter is not bundled; use
  `--min-opacity` / `--max-depth` to prune sky artifacts.

## Licensing

The HY-WorldMirror-2.0 weights are under Tencent's community license, which
**excludes the EU** — treat the imported checkpoint as reference/research
material. This implementation is from scratch; the sibling `splat` renderer
follows the (Apache/MIT) gsplat semantics.
