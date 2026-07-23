# Weekly market fine-tuning pipeline — Chronos-2 + Kronos

See DATASETS.md for the data inventory + the fetch & fine-tune commands.

Goal: take the whole market's latest OHLCV, fine-tune both forecasting models, and
produce a **dated weekly checkpoint** that adapts to recent market behavior so the
next week's cross-sectional ranking is above average — **without overfitting**, on
**leak-free** datasets. Full backprop (not PEFT-only), validated to work correctly.

## Hard requirement: correctness, proven three ways

Every backward pass ships only when ALL of these are green, on **CPU and GPU**:
1. **gradcheck** — `gradcheck::directional_check` (analytic vs central finite-difference),
   tol `atol=4e-3, rtol=8e-2`, `eps=5e-3`, `n_dirs=4`. Matches the bar every other
   brain model clears.
2. **from-scratch learning test** — a tiny config trained from random init on a small
   *learnable* synthetic task; assert the loss collapses (memorizes / final ≪ initial).
   Proves the gradients don't just match FD locally but actually drive learning.
3. **CPU↔GPU parity** — forward loss and gradients agree across the wgsl-cpu (Cranelift
   JIT) and wgpu backends (same WGSL source), per `gradcheck/tests/backend_parity` style.

Training runs on **CPU and GPU** (both brain backends execute the same WGSL kernels).
**NPU is inference-only** (OpenVINO exported ONNX cannot backprop); the fine-tuned weekly
checkpoint stays NPU-exportable for fast serving via the existing kronos/chronos2 export.

## Why full backprop needs new work

Both models are today **inference-only** ports: parity-exact forward via immediate GPU
submits, plain `w: HashMap`, host readbacks mid-graph — no ParamStore, no Step tape, no
`impl model::Model`, no backward, no gradcheck. brain HAS all training infra for
GPT/Qwen/MoE/GLM (gradcheck-passing, incl. gradcheck-validated Qwen LoRA); the two
forecasters must be rebuilt as differentiable `model::Model`s.

Blueprint = `crates/qwen/src/model.rs` (ParamStore roles, `forward_steps`/
`build_backward_steps` tape, `lora_fwd`/`proj_bwd`, shared `model::block` builders for
RMSNorm/RoPE/GQA/SwiGLU fwd+bwd, CE loss+grad kernels, `impl Model`).

## Kronos block = Qwen block (verified from nn.rs)

Pre-norm RMSNorm(norm1) → q/k/v linear **+bias** → NeoX RoPE(q,k) → causal scaled MHA
(n_heads=8, head_dim=64, scale 1/√hd) → out_proj **+bias** → residual → RMSNorm(norm2)
→ w1(gate),w3(up) no-bias → SwiGLU → w2(down) → residual. Maps onto `model::block` GQA
builders with n_kv=n_heads. Deltas vs Qwen: **projection biases** (Kronos has them, Qwen
uses q/k/v-norm instead), **no q/k-norm**, hierarchical embedding, dep cross-attention,
dual head. A forward-parity test (GQA-builder forward vs the existing ATTN_SCORES_QK
inference forward on the same weights) GATES that fine-tuned weights stay
inference-compatible.

## Phases (TDD, each gated by the three checks above)

- **P0 (done): leak-safe training-data pipeline** — `forecast::train_data`: whole-market
  OHLCV → (context→horizon) windows, **temporal split with an embargo/purge gap** ≥ horizon
  (the reference Kronos recipe OVERLAPS its splits — a leak we fix), calendar stamps.
  Per-model normalization stays in each model's existing past-only preprocessing.
- **P1: Kronos decoder as differentiable `model::Model`** (`crates/kronos/src/train.rs`).
  Milestones, each gradchecked before the next: (A) block stack + proj_s1 + CE_s1
  (embeddings frozen input); (B) trainable hierarchical + temporal embeddings; (C) dep
  cross-attention + proj_s2 + CE_s2 with the exposure-bias s1-sample; (D) LoRA. Tokenizer
  stays **frozen** (encode under no-grad), matching the reference. Loss `(CE_s1+CE_s2)/2`.
- **P2: Chronos-2 as differentiable `model::Model`.** New **pinball/quantile loss kernel
  + grad** (none exists today), group-attention backward, patch-embed + quantile-head
  backward. Same milestone/gradcheck discipline.
- **P3: fit + walk-forward PROMOTION GATE.** Fine-tune on train, evaluate held-out RankIC
  + val loss vs the BASE checkpoint; **promote the dated weekly checkpoint IFF it beats
  base out-of-sample**. Honest: many weeks it will keep base. Data source = trademiner
  `stocks.db` (whole liquid market, refreshed via `make weekly`).
- **P4: CLI + weekly wiring.** `brain forecast finetune --{kronos,chronos2} --data … --out
  <dated ckpt>` + gate report; NPU export of the promoted checkpoint for fast serving.

## Anti-overfit measures (baked in)

Small fine-tune LR (4e-5) + OneCycle warmup, weight decay 0.1, grad-clip 3.0, dropout,
exposure-bias s1-sampling, best-val-loss checkpoint on a **held-out future** window, the
embargo/purge split, multi-symbol pooling + shuffling, and above all the **promotion
gate** — a fine-tune only ships if it demonstrably beats base out-of-sample. LoRA is the
default fine-tune surface (few params) once the full backward is validated; full-param
training is used for the from-scratch correctness tests.
