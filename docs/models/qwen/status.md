# qwen — workstream ledger

Qwen3 dense decoder Transformer (GQA + QK-norm + RoPE + SwiGLU), fp32/WGSL
engine. The concurrent paged-KV serving engine is a major workstream
in brain.

## Done

- **P0 — config + param layout**: `QwenConfig` with GQA (`n_kv_heads < n_heads`),
  per-head QK-norm, decoupled `head_dim`, SwiGLU, no biases. `param_list()`
  verified against HF Qwen3-0.6B checkpoints (exact tensor count + shape match).
- **P1 — forward + backward**: full decoder forward/backprop as WGSL dispatches
  via `model::block`. Gradchecked via `gradcheck::check_qwen`.
- **P2 — HF import**: `brain qwen import --hf <dir>` reads single or sharded
  safetensors, maps HF tensor names 1:1, handles the GQA de-interleave for
  `k_proj`/`v_proj` when `n_kv_heads ≠ n_heads`.
- **P3 — training**: `brain qwen train` — end-to-end training loop with the
  shared `optim::Optim` (AdamW + grad-norm clip). Convergence tested.
- **P4 — LoRA finetuning**: frozen base + trainable `A`/`B` adapters.
  `gradcheck::check_qwen_lora` gates the backward. See P11 for the full
  named-adapter workstream (`--lora`, model-store integration, eval, serving)
  this landed into.
- **P5 — INT8 weight quantization**: `crates/qwen/src/q8.rs` — per-channel
  symmetric weights packed 4-per-u32, DP4A GEMMs. ~4× memory reduction with
  bounded accuracy loss.
- **P6 — sharding**: tensor parallel (`shard.rs`) + data parallel
  (`DataParallel<Qwen>`). `dp_parity` and `shard_parity` tests verify
  cross-device numerical agreement.
- **P7 — paged-KV serving engine** (`serve.rs`):
  - Paged KV cache with shared block pools (`model::paged::BlockAllocator`)
  - Batched ragged paged decode (one forward per iteration for all active sequences)
  - Chunked prefill (long prompts split into bounded chunks)
  - Int8 paged KV (on-read dequant, ~4× smaller pool)
  - Speculative decoding
  - Device-side greedy head (`ARGMAX_ROW/PART/FINAL`) — no `[batch, vocab]`
    transfer to host
  - Decode-regime kernel selection by row count
  - Int8 weight path A0 (per-token activation quant + DP4A)
  - On-device decode window A4 (argmax → next input without host round-trip)
  - Continuous batching scheduler + throughput benchmark
- **P8 — tool-call eval**: `brain qwen toolcall` — structured tool-call
  evaluation against ground-truth targets.
- **P9 — encoder parity**: cross-backend parity (CPU == wgpu == Vulkan) for the
  encoder path used by FLUX.2 text conditioning (`encoder_parity.rs`).
- **P10 — bench train P40**: measured training throughput on Tesla P40
  (`bench_train_p40.rs`).
- **P11 — named LoRA adapters, end to end**: `brain qwen finetune --lora RANK
  --weights BASE --adapter OWNER/NAME[:TAG] --dataset DIR` trains a named
  adapter from a bench-exported `generic-messages-v2` dataset (real Qwen3
  chat template, `data::chat_template`, never a hand-rolled one) and saves
  ONLY the adapter tensors into the model store
  (`<vendor>/<repo>/adapters/<owner>/<name>/<tag>/`, `crates/modelref`'s
  `vendor/repo:owner:name:tag` grammar) — retraining with the same
  `--adapter` overwrites the tag in place. `brain qwen eval --weights BASE
  [--adapter ...] --jsonl FILE` reports held-out teacher-forced loss/token
  accuracy, base alone or base-vs-adapter side by side
  (`qwen::eval::score_chat`). `QwenResident` folds a named adapter into the
  base at `activate` (`Qwen::from_tensors_decode`) so `brain caps`/`brain do`
  serve it as its own catalog entry, zero extra per-token cost once folded.
  See `docs/guides/training.md` for the full ledger (dataset contract, both
  learning gates, what's still planned).
- **P12 — calibrated INT8 KV cache**: `serve.rs`'s existing int8 KV path
  quantizes with a purely ONLINE per-(token, kv-head) absmax — no
  calibration, no offline pass. `Engine::calibrate_kv` (offline only, never
  on the hot `run_batched_submit` path) prefills a representative prompt set
  and reads back every K (post-RoPE)/V row into `model::actstats::Collector`,
  keyed `layer{L:02}.{k|v}.head{H}`. `brain qwen calib --weights --jsonl
  [--report] [--out report.json] [--clip-out kv_calib.json --percentile Q]`
  prints/writes the `absmax`/`p99`/`p99.99`/`outlier_ratio` report and, with
  `--clip-out`, the actual `model::kvcalib::KvCalib` table the engine loads
  (architecture-agnostic — lives in `crates/model`, not `crates/qwen`, since
  nothing about the type or file format is Qwen-specific; qwen is just the
  first consumer). `brain qwen eval --kv fp32,int8,int8-calib` scores held-out
  loss/accuracy through the SAME paged engine `brain serve` runs
  (`qwen::eval::score_chat_paged`, gated bit-for-bit-in-spirit against the
  legacy `score_chat` backend at `fp32` — see below), so the calibration
  decision is judged on the number that actually matters, not a proxy.

  **Measured on the real Qwen3-0.6B checkpoint** (10 short calibration
  prompts, `--percentile 0.999`): 448 streams (28 layers × 8 kv-heads × K+V),
  **worst `outlier_ratio` 1.72** (`layer26.v.head5`), **median 1.07** — the
  handful of streams above ~1.3 cluster almost entirely in LATE-layer V heads
  (23–27), not K, and not early layers. A much tighter distribution than the
  "long-tailed activations" case calibration exists to fix.

  **`brain qwen eval --kv fp32,int8,int8-calib`, same checkpoint, a held-out
  10-sample chat set, `--block 256` (228 scored positions)**:

  | `--kv` | loss | Δ vs legacy fp32 | token-acc |
  |---|---:|---:|---:|
  | `fp32` (paged engine) | 4.2225 | **+0.0000** | 54.8% |
  | `int8` (uncalibrated, online absmax) | 4.2379 | +0.0154 | 55.3% |
  | `int8-calib` (p99.9, this calibration set) | 5.5618 | **+1.3392** | 40.8% |

  Two findings, both load-bearing for W3.5's default decision:
  1. **The paged fp32 backend matches the legacy backend EXACTLY** (delta
     `0.0000`, not just "close") — confirms `score_chat_paged`/
     `Engine::score_positions` are correct, independent of the KV-dtype
     question, on a real checkpoint (the tiny-synthetic-checkpoint gate test
     only proves the mechanism; this proves it at the real shape).
  2. **Plain uncalibrated int8 is nearly free** (+0.0154 loss, token-acc
     *higher*) — consistent with the low outlier ratios above. **Calibrated
     int8 at p99.9 is measurably WORSE** — not a wash, a real regression
     (+1.34 loss, −14 points of token accuracy). This is the clip mechanism
     working exactly as designed (`scale = min(token_absmax, clip[head]) /
     127` hard-truncates any value above `clip[head]`) applied to
     UNDER-calibrated ceilings: 10 short prompts underestimate the true
     p99.9 a held-out set actually needs, so real signal on the eval set gets
     truncated away. **Not evidence against the calibration mechanism** — it
     is evidence that a 10-prompt calibration set is too small at this
     percentile. A larger/more representative calibration set (or a less
     aggressive percentile, e.g. p99.99) is very likely to close most or all
     of this gap, but that is unverified — no larger sweep was run in this
     pass.

  **Conclusion for the default (W3.5)**: ship plain (uncalibrated) int8 KV as
  the serving default — the data supports it as a clear, nearly-free win.
  Calibrated int8 stays available (`--kv-calib` / `kv_calib.json` next to a
  checkpoint) but is NOT defaulted to: shipping a default that measurably
  degrades quality on the only real measurement taken would be irresponsible,
  regardless of the mechanism's promise. `KvCalib::from_model_dir`
  auto-discovers `kv_calib.json` beside a checkpoint when a caller opts in;
  absent, serving stays uncalibrated (`KvCalib::disabled`'s `f32::MAX`
  sentinel makes the clipped kernel bit-identical to the plain one).

## Parity ladder

| Gate | What | Result |
|---|---|---|
| P1 | forward vs HF reference | cosine ≥0.9999 |
| P1 | backward (gradcheck) | all params pass `check_qwen` |
| P2 | HF import (single + sharded) | exact tensor match |
| P4 | LoRA backward (gradcheck) | passes `check_qwen_lora` |
| P6 | DP parity (2 GPUs) | bit-identical |
| P7 | paged-KV decode vs naive | bit-identical per-token |
| P11 | folded-adapter decode-only generation vs live unfolded trained forward | exact token match (`lora_serve_fold.rs`) |
| P11 | adapter beats base on held-out inputs, from a RELOADED checkpoint | `lora_learning_gate.rs` (synthetic, CPU) + `qwen_eval.rs` (real Qwen3-0.6B, `#[ignore]`) |

## Remaining

- Prefix caching across requests (infrastructure exists in `model::paged`,
  not yet wired into `serve.rs`).
- FP8 / E4M3 weight path (the INT8 path is the current quant tool).
- Mixture-of-Experts serving (dense configs only today).
