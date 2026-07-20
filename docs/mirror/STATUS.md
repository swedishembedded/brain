# mirror — workstream ledger

2026-07-19, branch feat/world-models. Model = HY-WorldMirror-2.0 (1.26B fp32),
imported exactly; all parity vs the PyTorch reference via
`tools/mirror_dump_reference.py` goldens (committed samples: rms + 64–256
point values per stage).

## Done

- **P0**: `MirrorConfig::param_list()` reproduces the checkpoint's 1545
  tensors exactly (T0, device-free, incl. the 47 aliased `rope.periods`);
  `brain mirror import` converts the 5.05 GB safetensors → `.weights` with
  every tensor consumed + shape-verified. Cross-cutting refactor:
  `layernorm`/`ln_stats`/`layernorm_dx` gained an eps parameter (DINOv2 needs
  1e-6); all 37 call sites updated, gradcheck green.
- **P1**: PIL fixed-point bicubic preprocessing ported **bit-exact** (T1);
  DINOv2 ViT-L/14-reg encoder via the new shared `model::vit` builder —
  patch tokens match the reference ≤2e-4 (T2) on CPU and wgpu.
- **P2**: trunk — 24 alternating frame/global levels, per-head-dim LN QK-norm
  (`ln_head`), normalized 2D RoPE (`rope2d` + host tables porting
  `norm_rope.py`, 38×38 grid, specials at (0,0)), LayerScale, taps [4,11,17,23]
  frame‖global concat. All 4 taps ≤3e-3 after 48 attention blocks (T4).
- **P3**: four DPT heads (deconv = `conv2d_dx` gather form with checkpoint
  ConvT weights verbatim; sinusoidal pos-embeds host-precomputed in f64;
  a/b/t/u fusion ping-pong) + GS parameter convs + iterative camera head
  (adaptive-LN modulation via 3 sliced matmuls of the fused 6144 projection).
  Two bugs found & fixed via the tiny-grid stage-isolation harness
  (`tools/mirror_dump_dpt_tiny.py` + `tests/t3_dpt_tiny.rs`): (1) the DPT
  token→NCHW permute was wired to the inverse kernel; (2) the reference
  `ResidualConvUnit` uses `nn.ReLU(inplace=True)`, which MUTATES the block
  input — the skip connection adds `relu(x)`, not `x` (the classic MiDaS
  quirk; now documented in `dpt.rs::rcu`).
- **P4**: host Gaussian assembly (camera decode incl. scalar-last quats and
  fov intrinsics; per-pixel back-projection on the pixel-index grid; color =
  SH-residual·C0 + rgb), `brain mirror infer` (PLY + cameras.json + depth/
  normal maps), `brain mirror demo` (shared-Gpu handoff to the splat viewer,
  starts at the first predicted pose).
- **T1 (kernels)**: gradchecks for `rope2d` (sign=-1 VJP), `ln_head_dx/dgb`,
  `scale_chan_dg` vs finite differences — green. Block-level `vit_block_bwd`
  composition: cache struct landed; full assembly pending.

## Parity ladder

| Gate | Stage | Result |
|---|---|---|
| T0 | param layout (1545 tensors) | exact |
| T1 | PIL bicubic | bit-exact |
| T2 | DINOv2 patch tokens | ≤2e-4 |
| T4 | trunk taps ×4 | ≤3e-3 |
| T5 | heads + gaussians + camera | ≤5e-3 (all four heads, 12ch gaussian params, cam 9-vec) |

## Performance (honest, CPU JIT release, 22 threads)

- 1-frame 518² full forward (DINOv2 + trunk + 4 heads + cam): ~10–14 min.
  The engine is unoptimized for this model class (naive matmul kernel
  dominates); `matmul_tile` routing, wgpu, and the NPU export (P6) are the
  acceleration paths. Import: 5 GB in ~15 s.

## End-to-end demo (verified)

`brain mirror infer --images out/mirror-test/frame_00.ppm` (a 640² kitchen
photo): 268,324 gaussians + predicted camera + depth/normal maps in 547 s
(CPU JIT, 1 frame). `brain splat render` reproduces the input from the
predicted pose and shows correct parallax from novel views (renders saved in
`out/mirror-scene/`); `brain mirror demo` opens the interactive fly-through.
NPU: the DINOv2 encoder exports to ONNX and matches the reference under
OpenVINO-CPU to 1e-6 (`brain mirror export-npu` +
`tools/mirror_check_onnx.py`, runnable with the `NPU` device argument).

## Remaining

- Pos-embed interpolation for non-square/non-native grids (CLI errors
  loudly today).
- T1 block-level `vit_block_bwd` assembly (kernel-level gradchecks green;
  `VitBlockCache` scaffold landed; cross-attn bwd kernels assign rather than
  accumulate → training bwd needs chunk ≥ span).
- P6b/6c: trunk + DPT-head ONNX export (6a DINOv2 verified; same emitter
  pattern, RoPE tables as initializers).
- Perf: matmul_tile routing for [T,C]×[C,3C] shapes, wgpu memory budget
  autotune, frame pipelining.
