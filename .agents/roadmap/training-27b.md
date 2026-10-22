# Training the 27B on one card, generically - roadmap

Scope: Training Qwen3.8-27B on one card.

Ledger for training Qwen3.8-27B (crate `qwen35`) with LoRA on a frozen bf16 base
("normal LoRA") and on a frozen 4-bit NF4 base ("QLoRA"), on ONE CUDA card (GH200,
95 GiB HBM) at the longest sequence length the card allows, and for making every
piece of that generic so another architecture gets the same training by
implementing a small set of capability traits. Also holds the on-policy data and
GRPO work that the 27B needs. Update this file in the same commit that finishes or
changes an item. Dated measurements live in `.agents/knowledge/` (240, 250-252, 260).

## Hardware and limits this plan is built for (measured 2026-10-16)

- One GH200: 95.0 GiB HBM, 860 TFLOPS bf16 matmul, 399 GB/s device reading pinned host memory.
- Host memory cap of the job is 102.6 GiB (not the machine's RAM). Two concurrent
  builds are the safe maximum.
- CUDA binds a single buffer of at most 4 GiB. Any n x V, n x ff or n x 2*qd buffer
  that crosses it needs chunking: logits at T = 4,324; MLP/LoRA scratch at T about
  61.7K; `q_full` at T about 87K.
- The project drive is small and shared: build with CARGO_INCREMENTAL=0 in scratch
  target dirs, never run a bare `cargo test -p <crate>` (each integration-test binary
  is about 470 MB).

## Landed

- Streamed FP8 checkpoint source for qwen35 (`fp8_source`, `base_source`): host peak
  about 1.1-1.5 GiB for the whole 27B (851 tensors, 107.6 GB of f32 streamed in
  82-184 s). CLI flags `--base-dtype f32|bf16|nf4`, `--lora-rank`, `--lora-alpha`,
  `--seq-len`, `--recompute on|off` on `brain qwen35 finetune` (training on a bf16/nf4
  base returns an explicit unsupported error until the trainer tier below lands).
- Flash attention trio trains at head_dim 256 (parameterised, qwen3 behaviour
  unchanged); the GQA mixer saves LSE instead of T x T probabilities.
  Generated-from-WGSL cost at real width: see knowledge 240.
- Gated DeltaNet training over segments (`gdn_segment`): the per-layer peak is
  bounded by the segment, not the sequence; four per-chunk buffers are recomputed in
  backward. Measured at real width: about 1.3 MiB/token one-shot vs 113 KiB/token at
  T = 131072 with 2048-token segments (knowledge 250-252).

## Landed since (measured; see knowledge 233 and 234)

- Frozen-base tiers for the qwen35 trainer: bf16 and NF4 bases train LoRA with
  per-layer recompute, the chunked LM head, a bf16 embedding table and head,
  flash attention, segmented Gated-DeltaNet layers, a bf16 residual stream and an MLP in
  token chunks (`TrainOpts`); `brain qwen35 finetune --base-dtype bf16|nf4` streams the
  FP8 checkpoint (host peak below 7 GiB). Real 27B steps ran on one card for both tiers.
- The shared NF4 weight type and the shared chunked head loss; the generic training
  capability layer (`TokenObjectives`, `LoraSites`, architecture-table dispatch,
  `score_tokens`, `make test/conformance`).

## Next (dependency order)

- Adapter on the real 27B in serving: `brain qwen35 sample-adapter` shows a trained adapter changing
  the greedy answer on the NF4 base (knowledge 280, Q6 is partial: the facts change, but nothing was served and
  the control question was overwritten, so retention is not shown); serving it still needs a LoRA bank or an on-device fold on the resident engine, and the
  decode is repeated full forwards (no KV cache).
- Long rows: the real-checkpoint NF4 step at 8192 tokens ran through the CLI (knowledge 281); 16384 on the
  real checkpoint and a throughput pass are open. Throughput could not be tuned on this shared card: the same
  probe step varied 4x with a neighbour job (knowledge 281). Start from the launch counts there (segment-128
  GDN issues thousands of tiny `concat_split`/`bmm_tiled` launches per layer-step) and measure on a quiet
  card or with `program_table` replays. A tuned native CUDA flash kernel for head_dim 256 is also open.
- Embedding table and head are always bf16, never NF4; MTP builds keep the old fp32 head
  and cannot use the bf16 residual stream; host offload of residuals for 64K and beyond.
- `TokenObjectives` and `LoraSites` are implemented for qwen35 and qwen35moe
  (`objectives.rs`, `lora_sites.rs`) and both are conformance subjects. Still open: DPO/GRPO
  on the 27B (gates Q7-Q10).
- Move qwen3's int8 frozen-base input gradient into the shared `Ops::matmul_dx`, delete the
  local copy; double quantisation of the NF4 scale plane (skipped).
- `make test/parity` is diagnosed (knowledge 300): the Vulkan failures are a concurrency race
  on the shared pooled device (open; reproducer `crates/vulkan/tests/shared_device_two_threads.rs`),
  so the gate runs Vulkan serially and `bf16_train` stays CPU-only; the four qwen3 fused-attention
  dispatch-count tests were fixed for the CPU backend; the `random_shared_prefixes_stay_exact`
  flake is knowledge 301.

## Generic training capabilities (any model that implements them gets SFT/DPO/GRPO)

- `TokenObjectives` capability on `Model` replacing the `unimplemented!()` defaults
  (`enable_weighted_loss`, `set_loss_weights`, `set_loss_denominator`,
  `batch_token_logprobs`, plus `set_logit_scale(1/T)`); a typed `Unsupported` error
  raised at configuration time; objectives query it.
- `LoraSites` per model config and the generic adapter tensor naming/validation.
- SDK fine-tune types dispatch through the `ARCHS` table (family from the entry);
  `score_tokens` public per-token scoring with logit scale and adapter option.
- Conformance suite (`make test/conformance`, must fail on zero models): logprobs equal
  log-softmax of logits; weighted loss with weights 1 equals plain CE; finite-difference
  gradcheck at distinct dims; LoRA B = 0 leaves logits bit-identical; save/reload;
  one step of each objective lowers its own loss; chunked head equals unchunked; each
  frozen-base tier within tolerance of f32. Run for qwen3, gpt2, qwen35, qwen35moe.
- Move the serving `AdapterBank` behind `LoraSites` so qwen35 serves adapters
  (today qwen3 only; qwen35 folds an adapter into the base).
- Move qwen3's int8 frozen-base dx into the shared `Ops::matmul_dx` and delete the
  local copy (one implementation).
- In-process device high-water API (`Gpu::peak_bytes`) used by every memory gate.

## Server and SDK: token ids and logprobs (the on-policy data contract)

Done (HTTP chat, `docs/reference/api/http.md`): `logprobs: true` and `return_token_ids: true`
return the sampled token ids, each token's log-probability under its decoding
distribution and the prompt ids; the seed used, the answering adapter, the base
digest, backend and engine are echoed in the reply's `brain` object. Still needed,
with tests in the api-conformance e2e:
- accept token-id prompts; `system_fingerprint` as a digest of {base, adapter,
  tokenizer, template}; `base_digest`, `tokenizer_digest`, `template_digest` in
  `/v1/models`.
- SDK: `ChatRequest::logprobs`, `ChatResponse.{prompt_ids, tokens, logprobs, sampling,
  identity}`, `ModelIdentity.{tokenizer, template}`.
- A `generic-rollout-v1` dataset format and reader beside the preference reader; the
  GRPO external-records path recomputes pi_old with the training forward at policy
  version v, keeps behaviour logprobs for the mismatch metric and truncated importance
  sampling, refuses rows whose policy version is older than the staleness bound,
  and refuses a tokenizer/template mismatch.

## GRPO fixes (found by reading; none is yet shown to explain the documented negative)

1. Ratio bias at temperature != 1: stored sampling logprobs use the sampling
   temperature and top-k/top-p, trained logprobs do not; at T = 1.5 (the 12-cycle study)
   confident tokens start with ratio > 1 and positive-advantage rows are clipped to
   weight 0. All tests use T = 1.0. Fix: logit-scale hook, recompute pi_old with the
   training forward, untruncated sampling by default.
2. Zero-variance groups are dropped, not replaced, and the optimizer still steps on an
   empty batch (73.6% of groups in the 12-cycle run). Fix: dynamic sampling to a
   bounded resample factor; never step on zero rows.
3. Advantage options (mean-centred default, z-score kept for reproduction); token-level
   loss aggregation; clip-higher 0.2/0.28; truncated completions masked and rewarded on
   the trained span; EOS trained when sampled; non-finite guard (skip step, abort after
   3 in a row) and log-space ratio cap; KL to the adapter-disabled reference always
   logged (weight 0 by default); loss probe must not consume rollout rows; rollouts
   through the KV-cached decode engine instead of re-prefilling per token.
4. Done: `make test/rl` no longer names the deleted `continuous_cycle`, and naming a test
   target that does not exist fails the build.
Pre-registered tests (CPU, tiny models): on-policy identity at T in {0.7, 1.0, 1.5};
advantage fixtures; no step on zero rows with 80% zero-variance groups; length
invariance under token-sum; truncation; EOS; overflow/NaN; version refusal; analytic
KL; the same identity test on tiny qwen35 once it implements `TokenObjectives`.
Pre-registered experiment: replicate the 12-cycle continual GRPO study on 3 drawn
seeds with arms legacy / ratio fix only / plus dynamic sampling / plus the rest; the
ratio fix counts as a contributing cause only if it improves accuracy over legacy on
all 3 seeds. Sequential SFT with rehearsal remains the validated continual-learning
path; GRPO stays opt-in until it beats rejection-sampling SFT on a verifiable family.

## Qualification gates for the 27B (each a pass condition and a command; all GPU gates
run under the shared GPU lock, backend passed as a flag)

| Gate | Pass condition |
|---|---|
| Q1 conformance | the suite above passes for qwen3, gpt2, qwen35, qwen35moe |
| Q2 gradcheck | qwen35 weighted loss / logprob / logit scale at distinct dims, CPU and CUDA |
| Q3 real-width probe | 4 real-width layers (1 GQA + 3 GDN), real vocab, rank 16: finite loss, peak within 15% of the memory budget scaled to 4 layers |
| Q4 streamed load | 64-layer base from the FP8 source: host peak < 8 GiB, device 51 +/- 2 GiB (bf16) or about 17 GiB (NF4) |
| Q5 one SFT step | T = 4096, rank 16, 12 targets: finite, peak < 70 GiB (bf16), time recorded; then the longest T that fits for bf16 and NF4 |
| Q6 content gate (PARTIAL, knowledge 280: facts changed after reload, but not served and the France control was overwritten) | 50 steps on 8 examples teaching one fact; after save/reload and serving, the greedy answer contains the fact and the base without the adapter does not |
| Q7 DPO | chosen-minus-rejected margin rises; reference = adapter disabled |
| Q8 batched rollout | batch 32: aggregate >= 1,000 tok/s with per-token logprobs and the seed echoed |
| Q9 on-policy identity | mean |ratio - 1| < 1e-3 at T in {0.7, 1.0} on the first micro-step |
| Q10 GRPO step | 8 prompts x G = 8 completes; zero-variance fraction, KL, adapter version logged; no non-finite |
| Q11 endurance and resume | a run crossing a forced stop resumes to the identical adapter; no device memory growth |

## Deliberately not built

Full-parameter 27B training (does not fit this allocation); tensor/pipeline parallelism
and NCCL (one card); Vulkan/wgpu 27B training (2 GiB buffer clamp; CUDA is the lane);
training the MTP head or the vision tower; a value model; asynchronous off-policy
rollout/training overlap.

## Retirement candidates (remove once the replacement is proven by ablation)

Each item is removed in its own commit, after a grep of its consumers and an
ablation or parity run showing the replacement is no worse. Nothing here is
deleted yet.

- fp32 resident qwen35 trainer path (108 GB base, all activations kept, TxT scores,
  n x V logits): superseded by the tiered recompute trainer. Keep only while a
  parity test needs it as a reference.
- Full-logits cross-entropy in qwen35: superseded by chunked head CE.
- Pre-fix GRPO behaviour: z-score advantages, length-biased loss, dropped
  zero-variance groups, O(T^2) rollouts, the loss probe that consumes rows.
- qwen3-only adapter bank: fold into the generic LoRA layer.

## Throughput plan (originally 19-20 tok/s untuned; 635 tokens/s NF4 on the real checkpoint, knowledge 312)

`throughput-27b.md` is the current plan and its landed table; the list below is the original
order of payoff.


Budget: about 216 GFLOP per trained token (6 x 27e9, plus recompute). At 30-40 %
of GH200 bf16 peak that is on the order of 1-2K tok/s, so the current number is
far below the hardware. Profile first (`BRAIN_PROFILE=1`, per-kernel device time per
step), then in expected order of payoff:

1. Tensor-core GEMMs for the bf16 tier and for NF4 dequant-in-registers GEMMs
   (check what the generated kernels use today).
2. Tensor-core flash attention at head_dim 256 (forward and backward).
3. Parallel-scan or chunked GDN forward/backward instead of a sequential
   per-token recurrence.
4. Fewer launches: fused RMSNorm+residual, fused SwiGLU, fused LoRA A/B GEMM,
   CUDA graphs per layer.
5. Remove host syncs and overlap the FP8 stream, dequantise and compute on separate
   streams; keep the adapter and optimizer state resident.
6. Pack several short rows into one segment (token-budget batching) so GEMMs are
   large; raise recompute granularity only where memory allows.
7. Cheaper recompute: save the cheap activations, recompute only attention/GDN;
   FP8 activations for layer-boundary residuals.

Each step needs a recorded before/after tok/s on the same workload.
