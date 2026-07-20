# mirror — workstream ledger

2026-07-20, branch feat/world-models. Model = HY-WorldMirror-2.0 (1.26B fp32),
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
  landed and verified against committed float64 torch-autograd goldens
  (finite differences are noise-dominated through attention softmax), both
  trunk-like and DINOv2-like configs.
- **Rect inputs (T7)**: non-native grids bicubic-interpolate the DINOv2
  pos-embed with torch-antialias semantics (t6 gates the resampler — torch's
  AA bicubic uses PIL's a=-0.5, not -0.75); a full 392×518 forward matches
  fresh reference goldens (taps/depth/camera) at the T4/T5 tolerances. The
  CLI accepts any aspect ratio.
- **Multi-frame (S>1) fixed**: the first S=3 run produced an all-NaN scene —
  every parity gate was single-frame. Root causes: the patch-conv bias went
  through `add_chan_bcast` at N=s (a per-image [N,C] kernel; the shared [C]
  bias overran for frames ≥1) and `dpt.rs` hardcoded the reference's
  32-channel `output_conv2.0` where `param_list` says `feat/8`. Now gated by
  a tiny-config S=3 smoke test (t8: every stage finite for every frame),
  chunk-invariance tests for the query-chunked attention, and a batched
  conv fast-path parity test in backend-cpu.
- **Voxel prune**: `--prune 0.002` collapses multi-view duplicate gaussians
  with the reference `prune_gs` weighted-merge semantics (`splat::prune`).
- **P6b/6c**: trunk + all four DPT heads export to ONNX (`export-npu --stage
  trunk|heads`); RoPE tables/pos-embeds as initializers, structural tests
  green; `tools/mirror_check_onnx.py --stage trunk|heads` verifies against
  the T4/T5 goldens under OpenVINO (chained trunk→heads).

## Parity ladder

| Gate | Stage | Result |
|---|---|---|
| T0 | param layout (1545 tensors) | exact |
| T1 | PIL bicubic | bit-exact |
| T2 | DINOv2 patch tokens | ≤2e-4 |
| T4 | trunk taps ×4 | ≤3e-3 |
| T5 | heads + gaussians + camera | ≤5e-3 (all four heads, 12ch gaussian params, cam 9-vec) |
| T7 | rectangular 392×518 forward (pos-embed interp) | taps ≤3e-3, depth/cam ≤5e-3 |

## Performance (honest, CPU JIT release, 22 threads)

- 1-frame 518² full forward (DINOv2 + trunk + 4 heads + cam): ~9–14 min.
- 3-frame 518² forward + assembly: 4585 s (~76 min). `matmul_rows` bought
  ~8% at S=3 (5003 s → 4585 s on the same input) — real but modest, because
  global attention over 3×1376 tokens, not the linears, dominates at this
  frame count. wgpu and the NPU exports (P6) remain the real acceleration
  paths. Import: 5 GB in ~15 s.

## End-to-end demo (verified)

`brain mirror infer --images out/mirror-test/frame_00.ppm` (a 640² kitchen
photo): 268,324 gaussians + predicted camera + depth/normal maps in 547 s
(CPU JIT, 1 frame). `brain splat render` reproduces the input from the
predicted pose and shows correct parallax from novel views (renders saved in
`out/mirror-scene/`); `brain mirror demo` opens the interactive fly-through.
Multi-view verified: three kitchen frames → 804,972 gaussians, voxel-pruned
to 285,536 with `--prune 0.002`, three distinct predicted camera poses
(baseline ~0.16 in scene units). Novel views render correctly and show
geometry fused from the non-reference frames (stools and a chair absent
from frame 0), at ~0.6 s per 518² frame on the CPU backend.

## NPU / OpenVINO (measured, OpenVINO 2026.2, Intel NPU)

All three stages export and were run for real, on both devices:

| Graph | OpenVINO CPU (fp32) | Intel NPU (fp16-only) |
|---|---|---|
| DINOv2 encoder (6a) | 1.0e-6 | median rel 1.3e-3, per-token cosine ≥0.99985 |
| Trunk, 24 levels (6b) | taps ≤1.3e-6 | tap0 5e-4 → tap3 2.5e-2 median rel, cosine ≥0.9990 |
| 4 DPT heads (6c) | ≤1.0e-5 | median rel ~3e-4 from clean taps |
| Full NPU chain (trunk→heads) | — | depth head rms within **0.039%** of the fp32 golden |

The NPU executes fp16 only (it rejects `INFERENCE_PRECISION_HINT: f32`), so
the deviations above are precision, not error, and the check tool uses
measurement-justified per-device tolerances.

**Bug found and fixed — it was our graph construction, not the plugin.** The
trunk first came back badly wrong on NPU (last-level tap 38% rms error, its
global half holding the frame half's values). Bisecting per level showed
levels 1–21 clean and only the *last* level broken, and re-exporting a
22-level trunk moved the breakage to level 21 — i.e. "last level", not
"level 23". The cause was emitting the tap as `Concat(frame_out, x)` writing
straight into a graph output, where `frame_out` is an ancestor of `x` and
nothing else consumes `x`: the NPU plugin resolved the concat's second input
wrongly. The identical tensor routed out on its own was correct to 0.18%.
Inserting an `Identity` did not help (the plugin folds it); emitting the two
halves as separate `tap{i}_frame` / `tap{i}_global` outputs and
concatenating in the consumer fixed it completely. Worth remembering as a
general ONNX-authoring rule: **do not let a `Concat` write directly into a
graph output when one of its inputs is an ancestor of another.**

## Known unrelated failure

`brain-wm-genie --test blocks genie_geglu` fails a 1e-4 tolerance at
1.38e-3 against a host reference that uses an Abramowitz-Stegun erf
approximation. **Pre-existing and unrelated**: byte-identical failure
(0.0013809204) at 3a6169c, the commit before this workstream, and
wm-genie depends only on gpu-core + kernels. Rest of the workspace: 713
passed.

## Remaining

- NPU timings/throughput are not measured yet (parity is); the fp16 drift at
  the deepest tap (2.5e-2 median relative) is worth revisiting if the NPU
  path is used for anything beyond preview-quality geometry.
- Cross-attn bwd kernels assign rather than accumulate → training bwd needs
  chunk ≥ span (documented constraint, fine for the tiny-config path).
- Perf: wgpu memory budget autotune, frame pipelining; further CPU matmul
  blocking (column tiles) if the forward needs another step.
