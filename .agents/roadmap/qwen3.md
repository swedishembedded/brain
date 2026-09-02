# qwen3 - roadmap

Qwen3 dense decoder transformer (GQA, QK-norm, RoPE, SwiGLU) with a
concurrent paged-KV serving engine. Training, LoRA fine-tuning, INT8 weights,
INT8 KV cache, sharding, and continuous batching are built and verified
against the reference.

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
real completion (`docs/quickstart/img/serve.txt`).

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

Measured on this box (2x Tesla P40, `cargo test --release -p brain-qwen3
--test kv_offload_real -- --ignored`), at Qwen3-8B's exact KV geometry
(36 layers, 8 KV heads, `head_dim` 128, int8 KV + scales):

| | measured |
|---|---|
| KV per cached token | 74.2 KiB (confirmed against the real 8B checkpoint's own config) |
| park a 1024-token session (74.2 MiB) | 140 ms, 0.55 GB/s |
| revive it | 17.9 ms, 4.35 GB/s |
| park a 4096-token session (297 MiB) | 452 ms, 0.69 GB/s |
| revive it | 56.6 ms, 5.50 GB/s |

Context for those rates: the raw host bus on this box measures 4.26 GB/s up
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
committed but **has not run to completion**: `Engine`'s fp32 embedding table
at Qwen3-8B's vocabulary is 2.49 GB, over this card's 2 GiB storage-binding
limit, so the run dies in `create_bind_group` before it reaches the workload
(a separate, unrelated defect being fixed elsewhere). Everything above comes
from the geometry fixture in the same file, which needs no embedding table
that size. Re-run it once that lands.

## Not yet done

- [ ] Prefix caching across requests (the underlying infrastructure exists
      but is not wired into the serving engine)
- [ ] Faster KV swap-out: the gather side runs at about half the measured
      host-bus rate (see above), dispatch-bound rather than bandwidth-bound
- [ ] An FP8 / E4M3 weight path (INT8 is the only quantized weight format
      today)
- [ ] Mixture-of-Experts serving - only dense configurations are supported
      end to end
