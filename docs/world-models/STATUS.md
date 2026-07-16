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
- `brain wm play` (windowed via the standard build; `--headless` deterministic
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

- **Fine-tuning verified end-to-end (P3)**: `brain data gen pong` -> 300-step
  `brain wm finetune` on the Breakout model adapts it to pong (loss 0.115 ->
  0.013, no divergence); base imposes a Breakout prior on pong frames, the
  tuned model renders a clean pong court. Crux fix: the training graph's
  shared residual-copy zeros buffer was sized to the model input, not the
  widest activation -> OOB reads corrupted every gradient above input width
  (why gradcheck only passed at tiny channel counts). Now: FD gradcheck
  clean at cpg 8/16/32 + 1/2 groups; real Breakout 136/136 per-param scan.
- **Training (P3)**: full-UNet backward as a second SSA graph
  (crates/wm-diamond/src/train.rs; all conv weights+biases trainable,
  cond path frozen, gradients flow through GroupNorm), F-space EDM loss,
  AdamW+clip; gates: training-fwd == inference-fwd exactly, FD gradcheck
  across the whole net, 100-step overfit halves loss. `brain wm finetune`.
- **Data (P3)**: episode datasets (atomic writer, boundary-safe windows,
  split-by-episode), `brain data gen pong` (deterministic fixed-point env),
  `--record` on wm play + `wm replay --verify` (exact roundtrip).

## P1 remainder (done)
- VQ nearest-codebook: `vq_argmin` (Euclidean), `vq_argmax_dot` (cosine);
  `wm_core::vq` dispatch + straight-through/EMA/usage host math. Tested.
- Depthwise conv3d PEG: `dwconv3d` + `_dx` + `_dw`, FD-gradchecked.
- (registry 170 kernels)

## P4 GenieRedux-G CoinRun — spec ready (agent analysis)
Checkpoints downloaded (gitignored scratch): tokenizer 101.7M + dynamics 80.8M
(HF INSAIT-Institute/GenieRedux, `neurips` branch case study). Tokenizer =
frozen ST-ViViT + cosine-VQ; dynamics = guided MaskGIT (12 blocks @ dim 519
after one-hot(7) action concat). Both share ONE ST transformer (spatial "st" /
temporal "ts" reshapes, per-block PEG dwconv3d + attention + GEGLU FF).
brain HAS: cosine VQ, dwconv3d, leaky_relu, embed, matmul, layernorm, gelu/mul.
TWO REAL KERNEL GAPS — BOTH DONE (registry -> 178, FD-gradchecked):
1. [done] additive per-head score bias + configurable scale:
   attn_scores_bidir_bias (spatial) + attn_scores_causal_bias (temporal);
   backward attn_bwd_dq_bias / _dk_bias / _dbias. softmax/apply/dscores/dv reuse
   the existing bidir kernels (causal mask rides through softmax as prob 0).
2. [done] QK-L2-norm + learnable per-dim scale: l2norm_scale (+ _dx, _dg),
   applied to q,k as [tokens*heads, head_dim] before the scale-8 scores kernel.
Remaining for the STBlock: GEGLU (compose gelu*mul), causal-PEG asymmetric
temporal pad (2,0) via
host pre-pad, gradient-shrink (backward grad*0.1), Gumbel/cosine/topk host
decode. Build order: kernels -> shared STBlock -> tokenizer (load 100M for
parity) -> dynamics (load 80M) -> MaskGIT sampler. Data: CoinRun jpg frames +
actions.json (7 actions), convert via wm-ingest.

## P4 progress
- [done] wm-genie crate created. STBlock's two learnable sub-modules implemented
  + verified vs exact host refs (crates/wm-genie/tests/blocks.rs, <1e-4):
    * attn_forward — GenieRedux Attention (num_null_kv=0): pre-norm, fused to_kv,
      QK-norm (l2norm_scale x q/k_scale), scale-8 biased scores (bidir CPB /
      causal ALiBi), softmax, apply, to_out.
    * geglu_forward — FeedForward: pre-norm -> gelu(gate)*x -> out-proj.
  Verified architecture in docs/world-models/specs/P4.genie.md (from real ckpts
  + reference source).
- PARITY NOTE: brain gelu = tanh approx; GenieRedux uses torch exact-erf F.gelu.
  Full checkpoint parity likely needs an erf-gelu kernel (small, add at parity
  step). ContinuousPositionBias/ALiBi MLP outputs are precomputed host-side and
  fed as the bias buffer.

## P4 progress (cont.)
- [done] dwconv3d generalized to independent spatial/temporal pad (causal PEG).
- [done] STBlock assembled + host-verified (<2e-4): spatial(PEG->bidir attn CPB
  ->GEGLU) then temporal(PEG->causal attn ALiBi->GEGLU), 6 residual stages, with
  the (b t)(h w) <-> (b h w) t reshape. crates/wm-genie: peg_forward_w,
  stblock_forward.
- [done] STTransformer stack (N blocks + norm_out) — sttransformer_forward.
  This is the shared body of tokenizer enc/dec (8) and dynamics (12).

## P4 progress (cont.)
- [done] Tokenizer FORWARD path complete + component-verified (crates/wm-genie):
    * patch_embed / to_pixels (patchify 4x4x3, LN->Linear->LN with bias) <1e-4
    * vq_quantize (cosine VQ: project_in -> l2norm -> argmax vs normed codebook
      -> gather -> project_out) <1e-4 + exact indices
    * bias::alibi_bias (temporal) + bias::cpb_bias (spatial CPB MLP)
    * tokenizer_forward: patch(first+rest)->encoder(8,"st")->VQ->decoder(8,"ts")
      ->to_pixels(first+rest); end-to-end shape/finite/determinism/index-range.
  Kernel table: +bias_add, embed, vq_argmax_dot (all pre-existing kernels).

## P4 progress (cont.)
- [done] erf-gelu kernel (gelu_erf) matching torch F.gelu; wm-genie GEGLU uses
  it. Closed the known numeric parity gap.
- [done] TWO parity fixes found preparing import: (a) attention k,v project from
  the UN-normed x (kv_input captured before norm); (b) FeedForward pre-norm is
  nn.LayerNorm WITH bias (added FfWeights.norm_beta). Both host-verified.
- [done] Tokenizer IMPORT (import::import_tokenizer): 514 model.* tensors ->
  TokenizerWeights, full coverage (missing/leftover = hard error). Splits fused
  to_kv->to_k|to_v and GEGLU in-proj->w_x|w_gate; asserts custom-LN beta / unused
  context_norm are zero; drops VQ EMA buffers. VERIFIED against the real 1.2GB
  checkpoint: imports clean, all 514 consumed, 8+8 blocks, shapes correct.
  (pure-Rust torchpt reader handles the 1.2GB .pt in ~145s.)

- [DONE] TOKENIZER PARITY-EXACT vs the reference on the real 100M checkpoint:
  codebook indices 1280/1280 exact, reconstruction max abs ~1e-6. Full pipeline
  (patch->8 enc->cosine VQ->8 dec->pixels) verified. Final VQ fix: argmax
  (l2norm(input) . embed_RAW) — codebook used raw (only ~unit-norm). Tooling:
  scripts/parity-dump/genie_tokenizer.py + tests/parity_tokenizer.rs (ignored).
  (CPU forward is slow ~19min for f=5 64x64; correctness milestone, not perf.)

- [DONE] DYNAMICS (guided MaskGIT) forward + import + PARITY-EXACT: predicted
  argmax 1280/1280, logit max abs ~4e-5 vs the reference on the real 80M ckpt.
  dynamics_forward (token use_token-blend + pos_emb + action concat dim 512->519
  -> STTransformer(12) -> to_logits) + import_dynamics (372 tensors, full
  coverage). Passed on the FIRST parity run — reused the verified stack.
  BOTH GenieRedux models are now parity-exact in brain.

## Next (P4 remaining)
1. MaskGIT sampler (host decode loop): iterative confidence-based unmask,
   cosine schedule, Gumbel/top-k over dynamics_forward logits -> next-frame
   tokens -> tokenizer.decode -> frame. Then WorldModel wrap.
2. CoinRun ingest (jpg frames + actions.json, 7 actions) -> wm-ingest.
3. WorldModel wrap (tokenizer+dynamics+sampler) -> interactive SDL/WASD.
4. PERF (needed for interactive; correctness is done): wm-genie forward is
   host-round-trip heavy (gpu.read between every op) + naive matmul (~7-19min
   CPU). Move to a single on-device graph / GPU backend; convert .pt -> .weights
   once (import re-read ~145s tokenizer / ~100s dynamics).

## Backlog (user-reported)
- [#8 done] SDL window always compiled (no wm-sdl feature / build/wm).
- [#9] `wm play` Enter/reset must restore the INITIAL --seed-context + re-seed
  the NormalRng (currently resets to zeros -> random dream, not the start).

## Perf backlog
- INT8 PTQ for the NPU graph (existing quant.rs machinery) — likely 2x more.
- GPU: fold 3 per-NFE submits into 1 (3 pre-written gb sets); coopmat fp16.
- Batched training (n>1) + backward-pass GPU tiling (conv2d_dx/dw are naive).
- P3: episode dataset + gen_pong + record/replay + DIAMOND training
  (EDM loss, check_wm_unet) + fine-tune.
- P1 remainder: vq kernels, dwconv3d, maskgit host decode, EDM host math
  already partially landed in wm-diamond (generalize into wm-core when
  GenieRedux needs it).

Backups of the pre-restructure orchestration experiment:
`backup/wm-orchestration-v1`, `backup/wm-p1-*` branches.
