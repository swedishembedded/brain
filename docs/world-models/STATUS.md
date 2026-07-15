# World-model workstream — status

Goal: complete world-model support in brain — train / inspect / infer
action-conditioned video world models on CPU/GPU/NPU, load open pretrained
models (DIAMOND-Atari first, then GenieRedux-CoinRun, iVideoGPT, open-oasis),
play them in realtime in an SDL window with WASD, record play sessions as
training data, and fine-tune (full FT + LoRA). Reference material:
`/data/workspace/resources/world-models/` (read-only).

Mode: direct implementation on this branch (based on origin/main), gated by
the repo's standard discipline — `make build`, headless
`BRAIN_DEVICE=cpu MOE_SKIP_GPU_TESTS=1 make test` (with `DISPLAY=` unset:
a stale X11 DISPLAY breaks Vulkan enumeration), `make gradcheck`, tests
written from the specs in `docs/world-models/specs/`.

## Done
- Kernel registry mechanically derivable: `make kernels-regen`
  (`scripts/kernels-regen.sh`); registry canonicalized, 161 kernels.
- GroupNorm kernel family (`gn_stats/apply/dsum/dx/dgamma/dbeta`) fwd+bwd
  + host dispatch `wm_core::gn` — spec P1.gn.md, 9 tests incl. FD backward.
- FiLM/adaLN family (`film_chan*`, `film_row*` w/ rows_per_cond
  diffusion-forcing grouping, `gate_row*` adaLN-Zero) + `wm_core::film` —
  spec P1.film.md, 12 tests incl. FD checks.
- Glue kernels (`mul`, `scale_row`, `edm_mix`, `mse_value_w`/`mse_grad_w`,
  `pad2d`/`crop2d` adjoint pair, `nchw_nlc`/`nlc_nchw` permutes) — spec
  P1.glue.md, 15 tests incl. adjointness + FD.
- `wm-core` crate: object-safe `WorldModel` trait (CHW f32 frames, discrete
  actions, `set_nfe` quality knob) + deterministic GPU-free `FakeWorldModel`
  — spec P1.worldmodel-trait.md, 23 tests.
- DIAMOND parity fixtures: regenerable via `make wm-fixtures` (not in git);
  `crates/wm-diamond` stub crate reserves the layout.

- `wm-display` crate: SDL2 software-blit window (hand-rolled ~15-fn FFI,
  feature `sdl`, links system libSDL2; iGPU stays free for compute),
  longest-chord-wins keymap, fixed-timestep pacing with mock-clock tests and
  adaptive-quality hysteresis, headless/hash/PPM/tee sinks, `PlayIo` seam.
- `brain wm play` (windowed via `make build/wm`; `--headless` deterministic
  scripted rollouts with fnv1a golden hashes) and `brain wm bench`, running
  the FakeWorldModel end-to-end. 12 display tests, all headless.

- Pure-Rust torch `.pt` reader (`checkpoint::{zipread,torchpt}`): zip +
  pickle state_dicts, f16/bf16/f64 -> f32, strided views, full-coverage
  contract; 13 synthetic-bytes tests + bit-exact validation vs torch 2.12.
- **DIAMOND playable (P2 core)**: `crates/wm-diamond` — reference-parity
  UNet as one pre-recorded Step graph (AdaGN via gn_stats/gn_apply with
  gamma=1+scale), host EDM/Fourier/cond path, Karras/Euler sampler with the
  reference's exact quirks (unit-noise init, byte-truncation quantization,
  attention residual on the NORMED input), `brain wm import` (236 tensors,
  full coverage) and `brain wm play --model diamond --weights ... 
  [--seed-context <ppm dir>]`. Forward parity < 1e-4 end-to-end and
  per-module vs `make wm-fixtures`. Real Breakout checkpoint imported and
  playable: coherent Breakout frames, paddle follows actions.

## Measured (Breakout, 3 denoise steps, 64x64; warm/idle best-of-3 —
## this 155H throttles 2-3x run-to-run, see docs/PERFORMANCE.md)
- **Intel NPU (OpenVINO, fp32): ~60-63 ms/frame = 16 fps — realtime target
  MET at full quality** (23.7 ms/UNet-inference on the NPU silicon;
  parity vs brain engine 2.6e-4, fp16 internals).
- CPU (wgsl-cpu + native fast paths): ~146-166 ms/frame (6 fps; ~18 fps at
  1 denoise step). Was 440 before optimization.
- iGPU (wgpu/Vulkan): ~140 ms/frame rested, 350-490 throttled. Was 2 392.
- Levers (measurement-driven, docs/PERFORMANCE.md): parallel GroupNorm
  reduction (gn_part/gn_stats2 — gn_stats was 77.6% of GPU frame time),
  register-tiled conv_bias_reg (~21x GPU conv), native CPU gn fast paths,
  on-device denoise loop (one readback/frame). scripts/wm-perf-gate.sh
  guards order-of-magnitude regressions (x3 band; tighter flaps thermally).
- NPU path: `brain wm export --arch diamond` -> fp32 ONNX (Gemm+Split
  AdaGN, decomposed GN, attention as MatMul/Softmax) -> `--device npu
  --onnx out/diamond.onnx` in play/bench; sampler host-side.

- **Training (P3)**: full-UNet backward as a second SSA graph
  (crates/wm-diamond/src/train.rs; all conv weights+biases trainable,
  cond path frozen, gradients flow through GroupNorm), F-space EDM loss,
  AdamW+clip; gates: training-fwd == inference-fwd exactly, FD gradcheck
  across the whole net, 100-step overfit halves loss. `brain wm finetune`.
- **Data (P3)**: episode datasets (atomic writer, boundary-safe windows,
  split-by-episode), `brain data gen pong` (deterministic fixed-point env),
  `--record` on wm play + `wm replay --verify` (exact roundtrip).

## Next
- INT8 PTQ for the NPU graph (existing quant.rs machinery) — likely 2x more.
- GPU: fold 3 per-NFE submits into 1 (3 pre-written gb sets); coopmat fp16.
- Batched training (n>1) + backward-pass GPU tiling (conv2d_dx/dw are naive).
- P4: GenieRedux (VQ tokenizer + MaskGIT) on the same seams.
- P3: episode dataset + gen_pong + record/replay + DIAMOND training
  (EDM loss, check_wm_unet) + fine-tune.
- P1 remainder: vq kernels, dwconv3d, maskgit host decode, EDM host math
  already partially landed in wm-diamond (generalize into wm-core when
  GenieRedux needs it).

Backups of the pre-restructure orchestration experiment:
`backup/wm-orchestration-v1`, `backup/wm-p1-*` branches.
