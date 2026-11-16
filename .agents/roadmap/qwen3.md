# qwen3 - roadmap

Scope: Qwen3 dense decoder and its paged-KV serving engine (`crates/qwen3`).

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

`brain qwen3 finetune --data <toolcall-data> --weights <real Qwen3-0.6B
checkpoint> --block 2048 --batch 1` used to panic with `wgpu error: Out of
Memory` on a fully idle 24 GiB Tesla P40, even at `--batch 1`. Two real,
distinct root causes, both fixed together (the full fine-tune path,
`crates/paramstore`): (1) `ParamStore::new`'s grad/Adam-moment
buffer init wrote an explicit zero vector via `gpu.storage_init`, and this
backend's resident-cost accounting charges 2x for any buffer that is ever
WRITTEN (vs. 1x for allocated-but-untouched) - three such buffers per
trainable tensor (grad, adam_m, adam_v) made this the dominant real VRAM
cost of a full finetune, invisible from the nominal parameter count; switched
to `gpu.storage(numel)`, which allocates without writing. (2) the full fine-tune
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
`crates/sdk/tests/it/score_long_record_real.rs` (a multi-thousand-token record on
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

## Served at the stored precision (default weight tier)

One rule decides the weight tier for `brain serve` (catalog resident), the SDK
`TextGenerationPipeline` and the `brain perf` `qwen:`/`qwen-synth:` targets:
`qwen3::footprint::default_weight_tier` keeps the floating tier the checkpoint stores
its linears in (`qwen3::stored_weight_tier`, read from the tensor headers) and falls to
int8 only for 6B+ parameters where those weights plus 8 GiB do not fit the card;
`--weight-format` always wins (`catalog::resident_llm::qwen3_weight_policy`). Knowledge #468.

Qwen3-4B (bf16 release), GH200 gpu3, `brain perf run longctx --target qwen:<4B> --context 32768
--ladder 1,16,31 --steps 4 --seed 1`, int8 KV, one run each (2026-11-15):

| default tier | batch 1 ms/step (tok/s) | batch 16 tok/s | largest batch admitted at 32K |
|---|---|---|---|
| fp32 (before) | 10.60 (94) | 485 | 30 |
| bf16, the stored tier (after) | 7.92 (126) | 527 | 35 (batch 31: 496 tok/s) |

Accuracy, `qwen_bench serve-accuracy --weights <4B> --t 256 --gen 24`, KL of the last-token
distribution from exact fp32 (fp32 weights, fp32 kernels; its repeat KL 0), int8 KV:

| text, prompts | fp32 weights, tensor cores (old default) KL mean / max, top-1 | bf16 weights, tensor cores (new default) | bf16 weights, fp32 kernels | int8 weights |
|---|---|---|---|---|
| book, 96 | 2.68e-4 / 3.83e-3, 95/96 | 3.12e-4 / 4.66e-3, 94/96 | 1.85e-4 / 2.93e-3, 94/96 | 1.28e-3 / 1.05e-2, 93/96 |
| knowledge notes, 96 | 4.85e-4 / 5.82e-3, 96/96 | 5.67e-4 / 5.03e-3, 96/96 | 3.29e-4 / 3.99e-3, 96/96 | 1.92e-3 / 1.45e-2, 95/96 |
| book, 16 (the first 16 of the 96) | 3.08e-4 / 1.76e-3, 15/16 | 3.59e-4 / 1.67e-3, 14/16 | 1.61e-4 / 1.20e-3, 15/16 | 1.26e-3 / 3.45e-3, 14/16 |
| knowledge notes 300-349, 16 | 2.71e-4 / 1.33e-3, 16/16 | 9.73e-4 / 1.06e-2, 16/16 | 3.32e-4 / 2.60e-3, 16/16 | 1.41e-3 / 4.16e-3, 16/16 |

## Decode attention through the fused GQA decode (2026-11-15)

The serving engine decodes attention through `model::kv_tier::FlashDecode` (the split-key GQA decode the 27B uses)
at every batch size and at its f32, int8 and FP8 KV tiers, for the dense family's head shapes (`DECODE_SHAPES`:
width 128, groups 2/4/5/8); the exact-decode regime pins the single-sequence split count. Its native CUDA forms
(`paged_flash_decode_gqa_hd128_{kv8,kvfp8,f32}`) replaced the split/combine pairs, which were removed. Knowledge #469
(the block order the 8-bit pools' row scales need).

Measured, GH200 gpu3, Qwen3-4B fp32 weights, `perf run longctx --context 32768 --steps 4 --kernels`, attention
device time per step (36 layers, split + merge) and its rate on the K/V bytes (scales included) against 3.4 TB/s:

| KV, batch | before | after | rate after |
|---|---|---|---|
| int8, 1 | 2.17 + 1.76 ms | 1.23 + 0.41 ms | 2.0 TB/s split alone (60 %) |
| int8, 16 | 26.15 + 0.39 ms (step 32.9 ms) | 12.81 + 0.29 ms (step 20.0 ms) | 3.1 TB/s (92 %) |
| int8, 30 (planner max) | - | 22.95 + 0.42 ms (step 30.4 ms) | 3.3 TB/s (96 %) |
| f32, 1 | 189.7 + 0.24 ms (step 199 ms) | 4.14 + 0.27 ms (step 11.1 ms) | 2.3 TB/s (69 %) |
| f32, 7 (planner max) | 147.0 ms (step 157 ms) | 22.78 + 0.33 ms (step 33.2 ms) | 3.0 TB/s (87 %) |

The per-kernel timer adds a few microseconds a launch; nsys showed the int8 batch-1 split at 28.5 us and the
merge at 4.2 us a layer (62 % of the roof together).

- [ ] Batch 1 at 32K is 60 % of the roof (int8, split pass alone): one wave of 512 blocks of eight 16-key chunks a
      warp, with the merge a fixed 4 us a layer. Folding the merge into the last split block, and cutting the merge's
      split sum over four threads a channel, both measured slower (knowledge #472); the split pass itself is the gap.
- [ ] bf16 KV and an FP8 target in `perf run longctx` for this engine (the kernels are registered at FP8; bf16 is
      not an engine tier).

## Serving keeps the card busy (2026-11-16)

Qwen3-4B, bf16 weights, int8 KV, GH200 gpu3, `perf run serve --target qwen:<4B> --input 8192 --output 128 --warmup 2
--seed 1` (c=16: 32 requests, c=64: 128); client numbers from runs without the profiler. Card busy is the union of
kernels, copies and memsets from the first `serve.*` NVTX range to the last kernel of an nsys
`-t cuda,nvtx --cuda-graph-trace=node` run (the trace loses the kernel records of its last ~0.9 s, while the host is
still waiting on graph launches, so the NVTX end overstates idle time). Glue is `concat_split`, `silu_mul*`,
`cast_f32_bf16`, `qknorm_rope_*`, `add_rmsnorm_fwd`, the KV appends, norms, embedding and argmax kernels, by the NVTX
phase their launch ran in.

| step | c=16 req/s | c=16 TTFT p50 / p95 | c=16 busy | c=64 req/s | c=64 TTFT p50 / p95 | c=64 busy | c=64 glue: per prefill token, per decode step |
|---|---|---|---|---|---|---|---|
| main 86d37a54c | 2.42 | 604 / 5070 ms | 84.2 % | 2.66 | 12.1 / 20.1 s | 79.0 % | 12.1 ns, 2.04 ms |
| per-layer prefill submission, last-row read (#476) | 2.75 | 524 / 4262 ms | 97.1 % | 3.08 | 10.5 / 17.2 s | 96.4 % | 12.2 ns, 2.06 ms |
| per-part library products, SwiGLU writes the bf16 operand (#478) | 3.07 | 493 / 3626 ms | 93.0 % | 3.46 | 9.3 / 15.2 s | 95.2 % | 5.8 ns, 1.22 ms |
| native QK norm and rotation, four heads a block (#479) | 3.07 (2.98-3.17) | 457 / 3548 ms | 96.6 % | 3.66 | 8.9 / 14.1 s | 95.9 % | 4.9 ns, 1.23 ms |
| residual norms write the bf16 operand of the products that read them | 3.23 (3.21-3.24) | 425 / 3442 ms | 96.4 % | 3.70 | 8.8 / 13.9 s | 96.2 % | 4.4 ns, 1.11 ms |
| query+key norm and K+V append one launch each, SwiGLU without fp32 `h`, one block-table row copied (#485) | 3.09 (3.08-3.10)* | 447 / 3595 ms | 95.3 % | 3.53* | 9.2 / 14.6 s | 96.6 % | 4.0 ns, 0.87 ms |

| decode tapes of 1..=32 rows recorded and frozen when the scheduler is made | (see below) | | 96.4 % | | | 96.7 % | 4.1 ns, 0.88 ms |

\* Client runs taken while another lane's compiler held the cores the benchmark is pinned to; the traced c=64 run of
the same binary served its 128 requests in 33.9 s against 34.9 s for the row above, prefill kernels 200 against 203 ms
a prompt. The last row's client runs (c=16 2.99-3.25, c=64 3.28 req/s) ran under the same load with every kernel of the
traced run ~10 % slower than the row above's (the module's shared power budget); its own gain is the idle time behind
first captures, 0.11-0.27 s (c=16) and 0.19 s (c=64) -> 0, for 0.2-0.5 s of recording at start. The graphs are the
ones a server instantiated on first use anyway.

- [ ] Glue left per 8K-row prefill layer: `silu_mul_to_bf16` 0.25 ms, `add_rmsnorm_fwd_bf16` 2 x 0.09 ms, the
      attention output's `cast_f32_bf16`, `qknorm_rope_qk` 0.36 ms (a block per token was faster at prefill but starved
      decode, knowledge 479).
- Memory (2026-11-16): the fused `qkv_pre` / `gateup_pre` outputs are allocated only when a path that writes them
  runs (adapter rows, a QKV bias, the exact-decode regime, the reference tier); the bf16 library path never does.
  Qwen3-4B at an 8K-row round: 8192 x (6144 + 19456) words, 800 MiB, no longer held (`engine_glue`
  `a_bf16_library_engine_holds_no_fused_product_scratch`). The 32 decode tapes `prepare_decode` records and freezes
  hold ~2 MiB of driver memory each (driver free before and after each freeze, 0-6 MiB in 2 MiB steps), ~64 MiB for
  all: kept as is. The residual stream is two buffers instead of one per layer plus one: Qwen3-4B at an 8K-row round
  2800 MiB less (the engine's live bytes after build 27.78 -> 24.84 GB, driver free +2800 MiB, two A/B pairs in one
  locked run).
- The decode merge writes the output projection's bf16 operand itself (`paged_flash_decode_gqa_combine_bf16`, gate
  `flash_combine_bf16_native`): interleaved A/B traces at c=16 (base, new, base, new): decode glue 784 / 834 -> 738 /
  768 us a step, the context cast 62 -> 3 us a step, serving window 10.29 / 11.02 -> 9.76 / 9.99 s.
- [ ] A decode step spends ~0.8 ms of its ~13 ms (c=64) in glue launches of a few microseconds each (two norms, the
      QK norm, the append, the attention-output cast and the SwiGLU per layer): the residual add into the down and o
      products' epilogue, or the cast into the attention merge, are the next launches to remove.
- [ ] Prefill chunks still upload a per-row copy of the block table (17 MB at 8K rows); the attention kernels could
      take one table per sequence and a row-to-sequence map.

## 16 concurrent 32K prompts (2026-11-16)

Qwen3-4B, bf16 weights, int8 KV, GH200 gpu3, `perf run serve --target qwen:<4B> --concurrency 16 --input 32768
--output 128 --requests 16 --warmup 1 --seed 7` (knowledge 498). Busy as in the section above, from an nsys
`-t cuda,nvtx --cuda-graph-trace=node` run of the same command; the 27B and the MoE rows are the same measurement of
their own targets (`~/tmp/perf/final`, re-read with the same busy script).

| target | req/s | TTFT p50 / p95 | busy |
|---|---|---|---|
| Qwen3-4B before (pool of 32 whole contexts, no card in the sum) | exits 101 while loading | - | - |
| Qwen3-4B, pool cut to the card (31 contexts), the plan's charges the engine's | 0.63 | 10.5 / 21.0 s | 95.5 % |
| Qwen3.8-27B-FP8 | 0.33 | 24.6 / 43.9 s | 97.2 % |
| Qwen3.6-35B-A3B Q8_0 | 0.32 | 22.5 / 45.0 s | 97.9 % |

`brain serve` (HTTP, 16 streamed chat completions of 32.2-32.4K prompt tokens, 128 out, after one warm-up request):
before, every request was refused (`needs 33498 tokens, engine capacity is 24576`: the auto context charged the
never-fused attention slab); after, 16 of 16 complete, 0.46 req/s, TTFT p50 20.8 s, p95 34.1 s.

- [ ] `brain serve` reads a Qwen3 prompt in rounds of at most 512 rows (`QwenServeConfig::max_prefill_cap`, which
      bounds the never-fused attention slab), where the perf target reads it in one round of the whole prompt; on a GPU
      the slab is fused away, so the served round could take the tensor-core round (`model::serve::
      prefill_round_rows_for`) with its activation charge (`qwen3::serve::activation_bytes`) in the plan.

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
