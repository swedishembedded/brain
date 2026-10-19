# qwen3 - roadmap

Qwen3 dense decoder transformer (GQA, QK-norm, RoPE, SwiGLU) with a
concurrent paged-KV serving engine. Training, LoRA fine-tuning, INT8 weights,
INT8 KV cache, sharding, and continuous batching are built and verified
against the reference.

## Sizing: Qwen3-8B LoRA (rank 16, batch 1) on two Tesla P40

Release binary: a bf16 base trains 12288 tokens (21.1 / 20.0 GiB peak) and an
int8 base trains 16384 (22.6 / 21.0 GiB peak, 418 s per step); 24576 does not
fit two cards. The 8192-token guard panic was a wasted int8 activation
packing on bf16 bases, and the 12288 refusal an uneven pipeline cut; both are
fixed. Numbers, method and the remaining guard limitation: knowledge entry 215.

## Fixed: `brain serve --openai` panicked on its first real request

`brain serve --openai PORT` against a real, already-fetched-and-converted
`Qwen/Qwen3-0.6B` checkpoint used to start cleanly (model-dir scan finds it,
HTTP listener binds, `--ready-file` fires) but panic on the FIRST chat-
completion request, inside the serving lane:

```
thread 'brain-lane-Gpu(0)' panicked at crates/qwen3/src/serve.rs:643:80:
serve: Ops::new: Ops::new: kernel 'embed' is not registered on this Gpu --
every model that builds an `Ops` must register the full façade kernel set
(["matmul", ..., "embed", "embed#emb=bf16", "embed#emb=f16", ...]), not just
the tiers it plans to use
```

**Root cause**: `crates/qwen3/src/serve.rs`'s `ops_kernel_list()` - the
hand-maintained kernel list `Engine::from_map_with_gpu` builds its side `Ops`
façade from - had silently drifted 15 kernels short of `model::ops::
REQUIRED_KERNELS`: missing `embed`, `moe_linear_gated`, every
`paged_*_batched` bf16 storage tier, and `matmul_dx`/`matmul_dw` (plus their
own bf16/f16 variants). The gap was never caught by `cargo test` because
`Engine::from_map_with_gpu` is only reached lazily, through the residency
pool's `activate()` (GPU activation on demand, so many resident models can
share one GPU) - not eagerly at `brain serve` startup or in any existing
unit test, so it only ever surfaced on a live server's first real request.
Fixed by completing `ops_kernel_list()` to match `REQUIRED_KERNELS` exactly,
and closed permanently with a new unit test,
`serve::tests::ops_kernel_list_has_every_kernel_ops_new_requires`
(`crates/qwen3/src/serve.rs`), that does a pure name-set comparison against
`model::ops::assert_kernel_list_complete` - no GPU device required, so it
runs even where every other test in that module needs real hardware, and it
is exactly the check that would have caught this at `cargo test` time
instead of on a live server's first request.

Verified against real hardware end to end: `brain serve --openai` +
`curl .../v1/chat/completions` against the real checkpoint now returns a
real completion (`docs/getting-started/img/serve.txt`).

**This exact bug shape can recur** wherever a model hand-maintains its own
kernel list instead of deriving it from `model::ops::REQUIRED_KERNELS`. The
new `assert_kernel_list_complete` helper in `crates/model/src/ops.rs` is now
wired into a no-GPU-needed test at each of the three real `Ops::new` call
sites in the codebase (`qwen3::serve`, `qwen3::model`, `gradcheck::
bf16_train`), so any future drift in any of them fails at `cargo test -p
<crate>` time instead of on a live server's first request. A model added
later that builds its own `Ops` should add the same one-line test rather than
relying on someone noticing the drift by eye.

## Fixed (not independently re-verified end to end): full-parameter finetune OOM at real Qwen3-0.6B scale

`brain qwen3 finetune <toolcall-data> --weights <real Qwen3-0.6B
checkpoint> --block 2048 --batch 1` used to panic with `wgpu error: Out of
Memory` on a fully idle 24 GiB Tesla P40, even at `--batch 1`. Two real,
distinct root causes, both fixed together (`crates/cli/src/qwen_cli.rs`'s
`train()`, `crates/paramstore`): (1) `ParamStore::new`'s grad/Adam-moment
buffer init wrote an explicit zero vector via `gpu.storage_init`, and this
backend's resident-cost accounting charges 2x for any buffer that is ever
WRITTEN (vs. 1x for allocated-but-untouched) - three such buffers per
trainable tensor (grad, adam_m, adam_v) made this the dominant real VRAM
cost of a full finetune, invisible from the nominal parameter count; switched
to `gpu.storage(numel)`, which allocates without writing. (2) `qwen_cli::train`
never set `BRAIN_OFFLOAD_ADAM`, so `qwen3::finetune::Mode::FullOffload`
(keeps AdamW moments in system RAM instead of on-GPU, trading weight+grad+m+v
[4x] for weight+grad [2x] GPU-resident) was dead code - only the `--lora`
path ever exercised it. `train()` now sets it around a real (`base: Some`)
finetune.

**Not independently re-verified end to end in this pass**: this fix was
reviewed (the diff is mechanically sound and its stated reasoning checks out
against `ParamStore`'s and `qwen3::finetune`'s actual code) but not re-run
against a real checkpoint here, since it requires a real BPE-tokenized,
vocab-matched dataset (`model::train::load`'s resume path asserts the
dataset's vocab against the checkpoint's) that does not exist as a committed
fixture, and the README's Quick start does not include a full-parameter text
finetune line (LoRA is demonstrated via `qwen3tts finetune` instead, at
TTS-Talker scale). Re-verify with a real run before relying on this being
fully closed.

## Fixed: a batch bigger than the KV pool killed the serving lane

`Scheduler` admits a request when its PROMPT fits, and `pool_sizing` gives the
pool room for roughly two full contexts while the batch has
`BRAIN_QWEN_MAX_BATCH` (16) slots. So a busy server admits sessions it cannot
keep cached, and the first decode step that needed a block it did not have
reached `serve.rs`'s `append(...).expect("KV pool exhausted")`. Reproduced
before fixing: four sessions against an 8-block pool panic there.

Sequences are now preempted to host RAM instead (`model::kv_offload`,
`BRAIN_QWEN_KV_OFFLOAD_GB`, off by default). Whole sequences, between
scheduler turns - never blocks inside the decode loop, which for standard
causal attention would mean streaming a sequence's entire cache across the
bus once per token.

Measured on 2x Tesla P40 (`cargo test --release -p brain-qwen3
--test kv_offload_real -- --ignored`), at Qwen3-8B's exact KV geometry
(36 layers, 8 KV heads, `head_dim` 128, int8 KV + scales):

| | measured |
|---|---|
| KV per cached token | 74.2 KiB (confirmed against the real 8B checkpoint's own config) |
| park a 1024-token session (74.2 MiB) | 140 ms, 0.55 GB/s |
| revive it | 17.9 ms, 4.35 GB/s |
| park a 4096-token session (297 MiB) | 452 ms, 0.69 GB/s |
| revive it | 56.6 ms, 5.50 GB/s |

Context for those rates: the raw host bus measures 4.26 GB/s up
and 1.24 GB/s down (`gpu-core/tests/pcie_handoff.rs`, 216 MiB), against
287 GB/s of device DRAM (`gpu_core::roof`). Swap-IN is at the bus; swap-OUT
runs at roughly half of it, the gap being the `2 * n_layers` (int8:
`4 * n_layers`) gather dispatches and the per-chunk readback - real headroom
if it ever matters, and not on any hot path today.

Those same numbers are why per-block swapping during decode is not
implemented: a decoding sequence re-reads its whole KV every step, so a
block-level tier would fetch 74.2 KiB/token across a bus 67x slower than the
memory the attention would otherwise read it from.

The end-to-end arm of that test - the real 8B checkpoint, eight sessions
against a pool sized for four, with and without offload - is written and
committed but **has not run to completion**. It used to die in
`create_bind_group`: `Engine`'s fp32 embedding table at Qwen3-8B's vocabulary
is 2.49 GB, over a card's 2 GiB storage-binding limit. `Engine::embed_tiled`
now gathers it in vocab sub-ranges, each under the device's binding limit, so
that failure is gone; the arm itself still has to be re-run on a card with
the 8B checkpoint resident. Everything above comes from the geometry fixture
in the same file, which needs no embedding table that size.

## Prefix caching across requests

Wired: `Engine::prefill` adopts the longest cached chain of full prompt
blocks (`model::paged::PrefixCache`, refcounted, keyed inside the
sequence's adapter namespace) and computes only the tail. `prefix_stats`
reports hit and lookup tokens. Gates: `random_shared_prefixes_stay_exact`,
`sample_n_shares_the_prompts_kv_through_the_prefix_cache` (`serve.rs`),
`tests/engine_prefill_mixed.rs` and `tests/engine_adapter_bank.rs` (no
cross-adapter reuse).

## Forward-only batched builds run flash attention

A batched build that is not decode-only (training or inference, including
the held-out scorer and the encoders) routes causal attention to the flash
kernels wherever the device runs them, so no `[H,T,T]` score pair exists, and
`logits_all`/`encode_hidden(s)` submit the forward one layer at a time (the
tape training uses) instead of as one submission against the GPU wait bound.
`footprint::runs_flash` states the same condition for the estimator.
The CPU JIT keeps the materialised chain. Gates run without a device:
`footprint::tests::inference_memory_is_linear_in_the_context_on_the_flash_path`.
Still to run on a card: `model::kv_tests::inference_flash_forward_matches_the_materialised_chain`
(flash against the chain on the device) and
`crates/sdk/tests/score_long_record_real.rs` (a multi-thousand-token record on
Qwen3-0.6B completes).

## Tensor cores by default (serving and LoRA training)

On a device with tensor cores and the vendor library, the engine chooses them on
queried capability (`GemmPrecision::Auto`, the default of `Engine`, `FitOpts` and
`ChatFineTune`); `GemmPrecision::Fp32Reference` (`--qwen-gemm-fp32`, SDK
`gemm_precision`) pins the exact fp32 kernels and bars the native attention too.

- Linears of fp32 weights with at least 8 rows (prefill and batched decode) run as TF32
  products (`LibraryGemmProvider::forward_tf32`/`dx_tf32`): operands keep their
  binary32 storage and are rounded to a 10-bit mantissa inside the multiply, fp32
  accumulation, no copy of the weights. bf16 weights run as bf16 (activation
  rounded to bf16). The training tapes are recorded at construction, so the tier is
  chosen by `Qwen::new_shard_gemm`/`new_lora_shard_dt_gemm`, not afterwards.
- `--qwen-weights-bf16` stores the linears and the head as bf16 (the checkpoint's own
  values when it ships bf16): half the bytes in memory and per decode step.
- The paged flash prefill runs on fp16 `mma` at `head_dim` 128 as well as 256, for
  fp32, bf16 and int8 KV (`flash_prefill_f16_hd128{,_bf16,_kv8}`).
- The training attention runs on fp16 `mma` too: the dense forward (`flash_attn_fwd_f16_hd{128,256}`, also writing
  the log-sum-exp) and, at `head_dim` 128, the backward (`flash_bwd_{scale,dq,dkv}`; dO is multiplied by one
  power of two per call so a mean-reduced loss's gradients keep fp16's precision). `FlashGqaIds::native` is the
  switch; `Fp32Reference` pins the portable fp32 trio.
- Accuracy on the real 8B weights, throughput before/after and the open items: knowledge #326.

## Not yet done

- [ ] Faster KV swap-out: the gather side runs at about half the measured
      host-bus rate (see above), dispatch-bound rather than bandwidth-bound
- [ ] An FP8 / E4M3 weight path (INT8 is the only quantized weight format
      today)
- [ ] Mixture-of-Experts serving - only dense configurations are supported
      end to end
- [ ] Serving a GGUF at its own quantization. The serving engine
      (`serve::Engine::from_map_with_gpu`) builds every linear from a
      whole-model f32 host map and its GEMM dispatch (`Engine::linear`) has
      F32 and I8 arms only, so a Q4_K/Q5_K/Q6_K file is expanded and, on the
      int8 tier, requantized; a Q8_0 file round-trips through f32 to the same
      int8 grid. The device side exists (`model::ops::Weight::KQuant`,
      `matmul_kq_dyn`/`matmul_kq_gemv`, `gguf::try_kq_rect`'s lossless host
      relayout) but no decoder builds it. Needed: stream each linear from
      the source (`model::int8::quantize_from`'s Q8_0 byte repack,
      `try_kq_rect` for K-quants), a KQuant arm in `Engine::linear` with the
      activation group-sum prepass, and a per-tensor tier instead of one
      engine-wide `weights_int8`.
- [ ] A hard reasoning budget: Anthropic's `thinking.budget_tokens` is
      validated and treated as the target Anthropic defines it as, with
      `max_tokens` the only bound. Enforcing it would mean closing the think
      block (forcing the `</think>` tokens) once the reasoning reaches the
      budget, a decode-side token-forcing step the engine does not have.
