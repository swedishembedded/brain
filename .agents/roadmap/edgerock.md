# edgerock - roadmap

Scope: EdgeRock-4B windowed hybrid decoder with a bounded anchor memory (`crates/qwen35` config variant, `model::anchor_memory`).

EdgeRock (`rockmatrix-ai/EdgeRock`, HF `EdgeRockForCausalLM`, text only, tied head,
546 tensors / 4.58B parameters) is the Qwen3.5 hybrid decoder (`crates/qwen35`) with
two changes on every full-attention layer: a causal sliding window on the attention
(`swa_window`, 1024), and an **anchor memory** beside it (`anchor_memory`: capacity
1024 records per KV group, commit blocks of `min(256, window)` tokens). The memory
reads the layer's residual input, keeps per KV group the best-scoring written
records of completed commit blocks (score descending, position ascending), reads
them by softmax attention with a zero sink, gates the read per head and projects it
with its own `o_proj`; the result is added to the attention output before the
residual add. It is a row in the arch registry (`edgerock`, package `brain-qwen35`),
not a crate: the memory math is generic (`model::anchor_memory`, four WGSL kernels).

The reference is `tools/goldens/edgerock_dump_reference.py` (an independent torch
implementation on the installed `transformers.models.qwen3_5`; the upstream code is
never run). `python3 tools/goldens/edgerock_dump_reference.py --seed 7` writes the
goldens under `testdata/golden/edgerock/{tiny_text,tiny_ties}/`.

## Done

- [x] M0: goldens dumper (`tiny_text`: random weights, the bank evicts; `tiny_ties`:
  `admission_up` zeroed so every score is 0 and the tie rule alone decides).
- [x] M1: arch row; `Qwen35Config::{swa_window, anchor_memory}` + `edgerock_tiny()`
  (pinned to the dumper) + `edgerock_4b()`; HF `config.json` parsing that refuses an
  unknown `fixed_memory_revision`, a `fixed_memory_swa_layers` that is not exactly
  the full-attention layers, and every missing/zero `independent_memory` field;
  `param_list()` `blocks.{l}.memory.*` (the three memory RMSNorms are `(1+w)` norms
  and take the import fold); the text-only `model.layers.*` layout through the one
  `import::classify` (also the lazy `fp8_source`); init; forward through
  `Qwen35::logits_all` (windowed `gqa_mixer_fwd_windowed`, memory added before the
  residual add); paths that implement neither the window nor the memory refuse the
  config loudly (`Qwen35Config::require_plain_decoder`).
  Gates: `crates/qwen35/tests/edgerock_golden_parity.rs` (both variants, CPU and
  default backend), `edgerock_config.rs`, `edgerock_hf_fixture.rs` (the release
  layout through config, lazy source, eager import and a forward),
  `crates/model/tests/anchor_memory.rs` (selection and indexed read against host
  definitions), the import and init unit tests.

Parity at the tiny dims (`edgerock_golden_parity`, per stage cosine / max_abs /
rel_l2, 24 stages per variant): cosine 1.0000000000 at every stage; worst max_abs
4.1e-5 (CPU backend) and 2.9e-5 (default backend, wgpu), worst rel_l2 4.0e-6. The
bank of both memory layers (`layer{3,7}.mem.refs`, 360 slots each) equals the
reference exactly in both variants on both backends; flipping the tie rule in
`anchor_commit.wgsl` (formerly `anchor_select`) fails the ties variant and the model-crate selection test.

- [x] M2: training at tiny scale. `model::anchor_memory::anchor_memory_bwd` (generic) differentiates the
  memory end to end: the output projection and per-head gate, the indexed read (dP, softmax backward,
  query gradient, then key/value gradients **gathered** per record through an inverse bank map built
  by `anchor_slot_inverse`, so no atomics and `O(rows * heads * capacity * head_dim)`), the rotary and
  per-head norms, the writer MLP and the input norm, with both `u` paths (query/gate and writer/key/value)
  joined. The admission scorer gets no gradient and is `Role::Frozen` (a discrete sort reads it; weight
  decay must not touch it either). Six new kernels (`anchor_read_bwd_{dp,q,kv}`, `anchor_slot_inverse`,
  `anchor_read_gate_{bwd,wgrad}`) around `softmax_k_dx`, `row_dot`, `rmsnorm_bwd`, `swiglu_bwd`. The
  windowed attention backward needs nothing new: its saved probabilities are zero outside the window.
  Wired into `backward_over` for Full layers (the memory's `d_h` joins the layer's residual gradient;
  the recompute policy rebuilds the memory's activations through `layer_forward`). The training
  refusal is gone. LoRA targets `memory.{q,k,v,o}_proj` and `memory.write_{gate,up,down}_proj`
  (`memory_lora_targets()`, asked for through `lora_cfg_for(&cfg, rank, alpha)`; `LoraSites`,
  `hf_linear_name`), never the scorer. The arch row declares
  `full_ft` and `lora(F32)` with the matrix rows that back them.
  Gates: `gradcheck::check_edgerock` (b = 2, every trainable tensor of the memory and the windowed
  layer, banks pinned: a finite-difference step that moves a bank fails loudly instead of measuring a
  jump; rel error 1e-4 to 3e-2 over the memory tensors, no hollow memory entry),
  `check_edgerock_lora` (all seven memory adapter pairs), `model/tests/gqa_mixer_windowed.rs`
  (the window's forward and backward against finite differences), `model/tests/anchor_memory.rs`
  (the read backward against an explicit scatter), `qwen35/tests/edgerock_training.rs` (a batch of two
  sequences equals two single batches in loss and gradients, a fixed-batch loop cuts the loss by 40
  percent, zeroing `read_scale` moves the trained loss, the scorer is never optimised),
  `lora_freezes_base.rs` (memory base, norms, gate and scorer unchanged under LoRA).
- [x] CUDA parity. The memory's projections are `MATMUL` kernel steps, exact fp32 on every backend; only
  `Ops::matmul` takes the TF32 tensor-core products, and those perturb the *residual stream* the
  scorer reads, not the scorer's own arithmetic. So the banks are exact exactly where the whole model
  is: `edgerock_golden_parity` runs CUDA pinned to `GemmPrecision::Fp32Reference` (stages to 1e-4, banks
  exactly equal, both variants), and at the default tier (TF32) with the `lora_gemm_precision`
  convention (cosine 0.999) and at least 95 percent of bank slots equal: near-tied scores swap rank
  there (1 to 2 percent of slots on `tiny_text`), which the released bf16 model does as well.

- [x] M3: inference state. A sliding-window config decodes and prefills on **fixed-size** per-sequence
  state; nothing in it grows with the sequence:
  - the window's KV is a ring of exactly `window` rows per sequence and layer (position `p` in row
    `p % window`, `model::gqa_mixer::{ring_row, ring_live}`); decode attends `min(pos + 1, window)` rows
    through the unchanged paged decode kernels (`KvWindow::Ring`); a prefill chunk gathers the ring in
    logical order, appends its own roped rows, attends under the window mask and writes its last
    `window` rows back (`gqa_mixer_ring_chunk_fwd`, generic, any sliding-window model);
  - the anchor bank per sequence slot and memory layer (`model::anchor_memory::AnchorBankPool`):
    committed `pos`/`score` `[G,K]` and `key`/`value` `[K,G*D]`, pending `key`/`value`/`score` `[C,*]`;
    the pending count is `position % C`, so no counter is stored. **One ordering implementation**:
    `anchor_commit` merges a bank with explicit (score, position) candidates; the sequence form commits
    block by block through it (ping-pong banks, refs now block-major `[blocks, seq, G, K]`) and a decode
    step or a chunk commits through the same kernel, `anchor_bank_gather` moving keys and values.
    A decode step dispatches a commit for every sequence (empty unless its block fills) so the step is
    one list of dispatches; the read kernels address a pool by slot row. `reset_row` / `copy_row` are the
    slot-reuse and snapshot operations;
  - the first partial block of a chunk reads the state's bank, completed blocks commit, the tail
    stays pending (`anchor_memory_chunk_fwd`, segment by block).
  Wired into `step`, `prefill_chunked`, `prefill_embeds` and `run_decode_batch` (rings by position, banks
  and `ReadPlan`/`WritePlan` by slot row); `SlotPool` (`new_slot_pool`, `slot_prefill`, `slot_decode`,
  `slot_reset`, `slot_copy`) is the per-slot-row seam the M4 engine needs; `decode_snapshot` /
  `decode_restore` copy the whole fixed state. Still refusing the config loudly (the same message):
  the layer-streaming forward and trainer, paged multi-sequence prefill (`run_prefill_multi_stage`),
  verify / speculative rounds (several rows of one sequence), and the f32 `serve::Engine` (M4); the
  int8 GGUF resident cannot hold an EdgeRock config. A short chunk never takes the exact-round decode
  tape for a windowed config.
  Gates: `qwen35/tests/edgerock_decode.rs` on the CPU JIT, the default device and CUDA at the fp32 tier,
  both golden variants: step-by-step decode equals `logits_all` at every one of the 48 positions;
  chunked prefill at 1, 3, 5, 13 and 48 rows (then decode) equals it; a batch of three sequences at
  positions 7, 19 and 30 (batched in a different order than pool order) equals each alone and the
  whole sequence; a 133-token run (twelve windows) leaves every persistent buffer's identity and size
  unchanged; snapshot, a detour, restore and continue reproduces the continuation bit for bit; a reset
  slot behaves exactly as a fresh one while an unreset slot does not (the control), and a copied slot
  continues identically. `model/tests/anchor_memory.rs` checks the state form against the sequence form
  (every bank after every block with its keys, and every read), reset and copy.
  Open: the decode step moves every sequence's bank through a scratch bank on every token (the commit is
  unconditional to keep the step one list of dispatches), and the ring chunk attention is the
  materialised chain over `m + n` rows; both are M6 work.

- [x] M4: serving. `qwen35::serve_ring::RingEngine` implements `model::serve::PagedDecoder` over the
  slot pool, a second engine rather than a mode of `serve::Engine` (whose pool, block table and
  prefix story all assume KV that grows with the sequence; the module doc justifies it): one block
  per live sequence whose id is its slot, `max_seq_len` = the position limit (262144), batched
  decode of all live slots in one dispatch set (`Qwen35::slot_decode_dev`), chunked prefill under
  the scheduler budget, device heads shared with `Engine`. Prefix reuse is the generic
  `SnapshotPrefixCache`: an entry is a snapshot slot (`slot_copy` of the whole GDN + ring + bank
  state) plus the last hidden row, with a table that carries only the namespace; `reclaim_prefix`
  frees and evicts nothing because the cache holds no pool block. `tests/edgerock_serve.rs`:
  generic conformance on the CPU and the default device (CUDA under the build environment),
  scheduler-batched generation equal to per-sequence `step` decoding, a prefix hit equal to the
  cold run and reading only the suffix (asserted from `prefix_stats`), namespaces, LRU,
  reclaim, slot reuse after a cancel, fixed state bytes equal to `slot_state_bytes`.
  Catalog: `catalog::resident_edgerock` (`EdgeRockResident` over the pulled directory read in place
  through `fp8_source::open_dir`, bf16 weight tier, slots sized from the emptiest card's free
  memory by `ring_slots`; `EdgeRockProvider` for `brain do edgerock generate`),
  `qwen35::spec::EdgeRockSpec` (roles `weights` = the directory, `tokenizer` = its
  `tokenizer.json`), the `model_dir` arm for the `edgerock` family, sampling defaults from
  `generation_config.json`, chat through the existing Qwen3.5 flavour, `--ctx` honoured as a cap
  (`Qwen35Sizing::ctx_explicit`). `tests/edgerock_served.rs` serves a tiny pulled directory through
  the residency adapter (batch equals single, prefix hits, footprint = weights + fixed slots);
  `tests/edgerock_real_served.rs` (ignored) asks the released model the vault-code question.
  Sample `samples/python/llm/edgerock`; model page promoted to served. A device out-of-memory now
  says which allocation was refused before it unwinds (`DeviceOutOfMemory::raise`).
- [x] M6a (partial): inference speed, profile-driven. Tool: `cargo run --release -p brain-qwen35 --bin
  edgerock_decode_profile -- <model dir | synth[:layers]> --batch N --position P --device gpu1 --backend cuda`
  (per-kernel device table, state bytes per sequence, wall per step; `RingEngine::seed_synthetic` puts the
  slots at a decode position with full ring and bank for one prefill). Measured on one NVIDIA GH200 (GPU1),
  real weights, position 2048 at batch 1.
  <!-- perf-number: measured with edgerock_decode_profile and tps.py on the GH200, 2026-10-27; reproduce with those tools -->
  Before: 56.3 ms per step (17.8 tok/s), 72% of device time in `matmul_f32_reg` (the fp32 tied head, read every
  step at 2.5 GB, and the memory's fp32 projections and scorer: 73 calls). Fixes, each re-profiled:
  (1) the head, the embedding and the memory's seven read-side linears follow the bf16 weight tier of an
  inference build (`table_tier`, `memory_linear_bf16`, `Qwen35::head_matmul`), the admission scorer stays exact
  fp32 on the skinny-M kernel `matmul_gemv` (groups of 32 rows; the library fp32 path is TF32 and was rejected):
  19.2 ms. (2) Only slots whose pending block fills commit their bank (boundary-only `Commit`s): no bank copies
  on other steps. (3) `FLASH_SHAPES` gains group 4: the fused split-key decode attention replaces the
  scores/softmax/apply triad over the ring: 12.0 ms. (4) `anchor_read_softmax_apply`: the read's softmax and
  apply as one workgroup-cooperative kernel (devices with workgroup reductions; the CPU keeps the two
  kernels, the reference provider): 7.7 ms. (5) The slot pool keeps one buffer of slot rows per recurrent layer
  and decode runs the pooled in-place recurrence (`gdn_decode_pool`) instead of splicing every sequence's state
  in and out: at batch 64 this removed 44 ms of `splice` and `concat_split` per step; prefill stages a slot's
  row through a work buffer. (6) `anchor_read_scores_group` reads each key row once for the heads of its group,
  bit-identical to the old scores; the softmax-apply kernel takes the group's heads together.
  Through the API (`brain --device gpu1 --backend cuda serve --openai`, `tps.py`, greedy, 62-token prompt,
  aggregate tok/s), before on 664033238 and after: b1 19.4 -> 138, b8 121.5 -> 846, b32 230.7 -> 2093,
  b64 261.9 -> 2544, b256 2393 (14.5 per stream); prefill of a 4126-token prompt 3.06 s (about 1350 tok/s,
  about 1054 before); 67.5 GB at 256 live slots. `ring_slots` now keeps at most 8 prompts beside the live
  sequences (a cached prompt costs a whole slot).
  Parity after every step: `edgerock_golden_parity`, `edgerock_decode` (CPU and CUDA, batched sequences at different
  positions, chunk sizes, snapshots), `edgerock_serve` (conformance), `edgerock_training`, `model/anchor_memory`, and
  the real-weight test on CUDA at the bf16 tier: logits cosine 0.999993, same greedy answer; banks compared as
  per-(block, group) sets (99.94-99.99% overlap; slot-wise equality is 54-67% on the tensor-core tier because a
  rank shift moves every later slot), exact slot equality only on the fp32 tier.
  Ceilings reached: at batch 64 (23 ms per step) every attention/state kernel runs at roughly 1 TB/s of the
  card's HBM because the per-sequence state is fp32 (194.5 MiB per slot: ring 64, bank 64+, recurrent state
  about 50) and read once per step; b256 is bandwidth-bound at about 90 ms per step. b1 is at 7.4 ms against
  a 2.5 ms weight-streaming floor: about 800 dispatches per step, no decode tape for windowed configs.
  Not done: bf16 storage of the ring, the bank and the recurrent state (halves the bytes read and the slot
  size, needed for 512 slots in 60 GB; the ring prefill path is f32-only, the bank kernels f32, the GDN decode
  kernel f32 state); windowed prefill attention on CUDA (prefill still runs the materialised path); the
  decode tape/graph for windowed configs; the batched anchor read and the GDN pooled kernel are at about 1 TB/s
  and are the next targets by share.
- [x] M6b: precision matrix, one NVIDIA GH200 (GPU1), real weights, the bf16 checkpoint quantized on load group-wise through
  `TierPolicy` (`edgerock_decode_profile --weights <tier>`; the tables follow the tier at 16 bits, the memory's
  read-side projections follow the tier, the scorer, norms and read gate stay exact fp32). Step times are decode
  steps at position 4096 with full ring and bank, fp32 state (194.5 MiB per slot), greedy.
  <!-- perf-number: measured with edgerock_decode_profile on the GH200, 2026-10-27; reproduce with that tool and `precision_matrix_against_the_reference_forward` -->
  | tier | weights+tables GiB | b1 tok/s | b64 agg tok/s | b256 agg tok/s | layer cos (worst) | logits cos | logits rel_l2 | greedy | bank set overlap |
  |---|---|---|---|---|---|---|---|---|---|
  | f32 | 17.29 | 59 | 1360 | 1732 | 0.999994 | 0.9999999 | 4.8e-4 | same | 99.99% |
  | bf16 | 8.78 | 126 | 2716 | 3646 | 0.99917 | 0.999993 | 3.6e-3 | same | 99.94% |
  | f16 | 8.78 | 41 | 163 | 533 | 1.000000 | 1.000000 | 3.3e-6 | same | 100% |
  | f8e4m3 | 5.11 | 112 | 2144 | 3328 | 0.99501 | 0.99889 | 4.9e-2 | same | 99.67% |
  | i8 (q8) | 5.57 | 145 | 2336 | 3165 | 0.99157 | 0.99963 | 2.7e-2 | same | 99.76% |
  | q4 | 3.74 | 159 | 463 | 1270 | 0.95162 | 0.98533 | 1.7e-1 | same | 98.95% |
  | nf4 | 3.74 | 69 | 2166 | 3332 | 0.95794 | 0.98632 | 1.7e-1 | same | 99.04% |
  | f4e2m1 (fp4) | - | - | - | - | cannot run: `Weight::upload: F4E2M1 weights are not implemented yet` | | | | |
  Gaps, not fixed here: (1) fp4 E2M1 has no weight upload (a generic gap in `model::ops`; `matmul_nvfp4_gemv`
  exists for pre-packed NVFP4 checkpoints only). (2) f16 has no fast batched GEMM at this shape: b1 is 3x slower
  than bf16 and b64 17x slower. (3) q4 has a fast GEMV but no batched tensor-core GEMM: b64 is 6x slower than
  bf16. (4) nf4 has no fast GEMV: b1 is 2x slower than q4 of the same size, while its batched path is on par with
  bf16. (5) Every tier is bounded at large batch by the fp32 per-sequence state (about 1 TB/s of HBM through the
  attention and recurrence kernels), so b256 does not separate the tiers below bf16. A serve-level option for the
  weight tier of this model is not wired (`Qwen35Sizing` has no tier): the per-tier numbers are from the profile
  tool at the engine, not through the API. KV and state tiers: ring KV bf16/fp8/int8 exists in `KvTier` for the
  fused decode, but the ring prefill path is f32-only (`gqa_mixer_ring_chunk_fwd` asserts it); the bank and the
  recurrent state have no storage tiers (kernels are f32). Not done.
- [x] M6c (partial): state formats and prefill. One NVIDIA GH200 (GPU1, 96 GiB), real weights.
  <!-- perf-number: measured with edgerock_decode_profile and tps.py on the GH200, 2026-10-27; reproduce with those tools -->
  (1) The slot pool stores each ring as f32 or bf16 (`SlotPool` takes a `KvTier`; the decode flash kernel already
  reads bf16 planes, the ring prefill gathers and writes back through `pack_bf16`/`unpack_bf16`, a generic change in
  `gqa_mixer_ring_chunk_fwd`). EdgeRock serves with a bf16 ring: 162.5 MiB per slot (was 194.5). Real weights, the
  1400-token golden prompt through the slot: logits cosine 0.9999925 (f32 ring) and 0.9999937 (bf16 ring), the
  reference's greedy token with both (`the_serving_slot_matches_the_reference_with_a_bf16_ring`). Decode at batch 64
  is 22.1 ms (was 23.2). (2) The windowed attention of a prefill round is one workgroup-cooperative kernel
  (`gqa_ring_chunk_attn`: the n query rows against their window, no full-sequence slabs; WGSL three-kernel path
  stays the CPU reference): the attention of a 4096-token prefill went from about 1.8 s to 0.25 s, the prefill from
  3.05 s to 1.6 s (about 2500 tok/s; was 1350). The kernel is still not tensor-core: the next prefill costs are the
  anchor read (320 ms), the scorer products (75 ms) and host submits (wall 1.6 s against 1.0 s of device time).
  Through the API (4126-token prompt, `l370`, max_tokens 2048; the API does not honour `ignore_eos`, so streams end
  at the model's EOS, about 430-470 tokens, and the identical prompts hit the prefix cache): per-stream decode tok/s
  b1 124.5, b8 103.4, b32 64.0, b64 39.2, b128 22.5, b256 12.9, i.e. decode-only aggregate b8 827, b32 2048,
  b64 2509, b128 2880, b256 3300. `--max-batch 480` loads (89 GB) but then runs out of memory in a decode step at
  256 concurrent (`brain: device out of memory allocating 252293120 bytes for storage` is now printed, the request
  fails with "non-string panic payload": the decode path does not classify the typed OOM); `--max-batch 320` serves
  256 streams (72 GB), 320 concurrent loses 64 requests. 512 slots do not fit (95.4 GiB estimated).
  Not done: bf16 bank keys/values (the read kernels are f32; the templated bf16 loads need scalar indices and the
  grouped kernels read 4-wide), bf16 recurrent state (native CUDA `gdn_decode_pool`, verify/commit entries and the
  WGSL fallbacks all f32), a decode tape/graph for windowed configs (the step is 493 submits and 863 dispatches
  with 13 host writes, built from fresh plan buffers each step), the tensor-core windowed prefill kernel, the
  weight-tier serve option, the generic tier gaps, the longctx perf target. Ceiling now: per-slot state is 162.5 MiB
  and every attention/recurrence kernel runs at about 1 TB/s, so b256 decode is about 70 ms per step.
- [x] M6d (partial): protocol support and the honest ladder. `ignore_eos` is parsed in the shared API sampling layer
  (OpenAI and Anthropic), declared on the generate action of the scheduled text models and honoured by the
  scheduler front end (the end-of-sequence ids are not stop ids for that request). A decode step that runs the card
  out of memory now refuses the sequences it carried with `RejectReason::DeviceOutOfMemory` and returns their blocks,
  as the prefill phase already did (`serve_device_oom` has both). Through the API with a distinct prompt per request
  (request index prefixed to the 4131-token prompt, so the prefix cache cannot answer it), max_tokens 2048 with
  ignore_eos, `--max-batch 320`, one GH200:
  <!-- perf-number: measured with tps.py on the GH200, 2026-10-27; reproduce with that client -->
  | concurrency | end-to-end aggregate tok/s | per-stream decode tok/s | mean TTFT |
  |---|---|---|---|
  | 1 | 113 | 122 | 1.3 s |
  | 8 | 509 | 78 | 5.1 s |
  | 32 | 832 | 35 | 14 s |
  | 64 | 697 | 18 | 53 s |
  | 128 | 536 | 7.8 | 154 s |
  The ladder is bound by serving-time prefill, not decode: eight 4K prompts take 10 s to read in (about 3.3k tok/s
  alone, less when interleaved with decode and with many queued), and every decode step is slowed by the prefill
  slices interleaved with it. The 256 and 512 rungs were not run. Not done from the M6d scope: the decode tape
  (b1 stays at 7.4 ms, 493 submits and 863 dispatches per step), bf16 bank and recurrent state, and the conversion
  of the M6a-c kernels into roofline-selected candidates (they are still wired by device capability checks in
  `anchor_memory.rs`, `gqa_mixer.rs` and the qwen35 model, the pattern the new rule forbids).
- [x] M6e (partial): the decode step as a recording. `Qwen35::slot_decode` replays a `gpu_core::tape` recording of the
  windowed step (`qwen35::decode_tape`): every per-step input (token ids, rope rows, ring row and live count, slot rows,
  block tables, the anchor memory's read and push plans) lives in persistent buffers (`DecodeInputs`) rewritten by 11 small
  host writes, the ring attention has one scratch size and split count at every position, and the bank commits (the only
  position-dependent dispatches) moved out of the step: they are issued after it from the pool's pending blocks
  (`CommitPlan`, `state_commit_steps`; one implementation for the recorded and the ordinary step). A recording is per batch
  size (any mix of slots replays it), taken once a size has held 3 steps, at most 4 kept per pool (0.74 GiB each at b64);
  `Qwen35::set_slot_decode_tape(on)` (default off) selects it; off is the reference path. Parity: recorded hidden states equal the ordinary step bit for
  bit at b1 (48 positions, four ring wraps, twelve commits) and b3 with changing composition, on CPU, default device and CUDA
  fp32 (`edgerock_decode`); `edgerock_serve`, `edgerock_golden_parity`, catalog `edgerock_served` and `edgerock_real_weights
  -- --ignored` pass.
  <!-- perf-number: measured with edgerock_decode_profile (real weights, position 2048, bf16) and tps.py on one NVIDIA GH200 (GPU1), 2026-10-27; reproduce with those tools -->
  | batch | ms/step ordinary | ms/step recorded | submits/step | graph alone, no host (layers) |
  |---|---|---|---|---|
  | 1 | 7.97 | 7.83 | 493 -> 3 | 7.18 |
  | 8 | 9.21 | 9.09 | 493 -> 3 | 8.41 |
  | 64 | 21.95 | 21.96 | 493 -> 3 | 21.14 |
  **Finding: the host was not the bottleneck.** The pass already issued each step asynchronously, so wall time equals the
  card's time (the layers replayed back to back take 7.18 ms; head and readback are the other 0.65 ms). The recording takes
  the host out (3 submits instead of 493) but buys 1 to 2 percent; through the API (`tps.py`, 62-token prompts, 512 tokens)
  it is neutral at b1/b8 (138 / 881-896 tok/s against 135-138 / 892-923 ordinary) and 8 percent BEHIND at b64 (2970-2993
  against 3232-3272 aggregate; waiting for a size to hold did not close it, cause not isolated: the b64 profile is equal, so
  suspect the interleaved prefill rounds or the ramp). Hence `set_slot_decode_tape` defaults to OFF; the recording is
  there for the device-side work below, which is what can make it pay. The 4.3 ms target needs fewer and faster device dispatches, not less host. Graph
  breakdown at b1 (859 dispatches, 8.4 us each on average; event stamps add about 2.9 us per node to the profile table):
  library GEMMs 304 calls, about 3.4 ms against a 2.3 ms weight-streaming floor (about 8.7 GB); the anchor memory read
  (`anchor_read_softmax_apply` 8 x 110 us, `anchor_read_scores_group` 8 x 59 us, gate) about 1.5 ms, whose banks are only
  64 MiB, so it is a grid too small for the card; 184 `cast_f32_bf16` and 64 `add_rms_quant` / `quant_epilogue_cluster`
  launches about 1.1 ms; 49 `rmsnorm_rows` 0.3 ms; GDN decode 24 x 14 us. Next: split the memory read across more
  workgroups (roofline candidate), fuse the activation casts into the producing norm/epilogue, run independent projections
  of a layer as parallel graph branches, and put the head in the recording.
- [x] M6f (partial, in progress): roofline round after the recording. (1) Anchor read split over the bank's slots
  (`anchor_read_split` + `anchor_read_merge`, flash-decoding: 32 slots per workgroup, partial softmax records, a merge that
  adds the sink): replaces `anchor_read_scores_group` (deleted) + `anchor_read_softmax_apply` for pooled reads on devices with
  workgroup reductions. Per-kernel device time at b1, 8 layers (graph-profiled, minimum over replays, a peer process shares
  the card so wall times are not comparable): 1.37 ms -> 0.31 ms (split 0.23, merge 0.08, event stamps included). Still
  about 26 us per split call against ~3 us of bank bytes: the phases inside a workgroup are serial.
  Parity: `model/tests/anchor_memory` (new `the_pooled_read_matches_the_definition_across_split_widths` against a host
  evaluation, caps 5 to 1024), `edgerock_decode`, `edgerock_serve`, `edgerock_golden_parity`. The pooled read no longer
  allocates the `logits`/`probs` scratch it does not use.
  (2) Dense bf16 decode GEMV (`matmul_bf16_gemv`: WGSL reference + native `cu/bf16_gemv.cu`, a warp per weight row, ten
  16-byte streaming loads in flight per lane, f32 activations read as they are). `Ops::dense_small_m_gemv` takes the single
  decode token of a `Weight::BF16` before the library product, which removes the 184 `cast_f32_bf16` launches (the library
  casts the activation to bf16 first) and the 304 library calls: dispatches per step 859 -> 675. Gate
  `gpu-core/tests/bf16_gemv_native.rs` (native and WGSL against an f64 oracle at ragged shapes, native selected). Cold-weight
  device time per call (same file, `--ignored`): [9216,2560] 17.6 us = 2.7 TB/s, [8192,2560] 15.8 us, [2560,9216] 17.4 us,
  [2560,4096] 10.8 us (1.9 TB/s), [4096,2560] 10.3 us, LM head [248320,2560] 335 us = 3.8 TB/s; the library product measured
  21 us on the first. Graph-profiled device time of the layers at b1: 8.63 -> 7.49 ms (event stamps included, minimum over
  replays). Eight rows: the same kernel is three times slower than the library's tensor-core product (13.8 vs 4.6 ms per step),
  so it served one row only. Native builds for 2, 4 and 8 rows (`matmul_bf16_gemv_m2/_m4/_m8`: activations staged in shared
  memory, a warp walks four weight rows per chunk) and `Fused::Bf16GemvMulti` (up to four projections of one activation in
  one launch, bit-identical to the singles, wired through `Ops::matmul_group`: b1 dispatches 675 -> 555) are in and gated
  (`bf16_gemv_native.rs`, m = 1..8 ragged); the multi-row crossover against the library product is not yet measured on a quiet
  card, so `DENSE_GEMV_MAX_ROWS` stays 1. Small projections sit at a 6.5 us latency floor
  (the 32-row in_proj_a/b, 1024-row k projections).
- [x] M6g (partial): bf16 storage tiers for the slot pool's bank and recurrent state (`SlotTiers { ring, bank, recurrent }`,
  default bf16 for all three, `SlotTiers::EXACT` is the f32 reference). The anchor bank's key/value and pending block store
  bf16 (`AnchorBankPool::tier`; push rounds to nearest even, the commit gather and row copies move words, the reads decode:
  the split read and the triad's score/apply are the f32 kernels with their key/value bindings rewritten by
  `dtype_variant`, so the CPU runs them too; `anchor_read_softmax_apply` is gone). The Gated-DeltaNet state and conv history
  store bf16 (`gdn_decode_pool_bf16`: the shared body templated on the state element, widened on load, rounded on store;
  `pool_rows_gather2` decodes and `pool_rows_scatter2_bf16` rounds for the WGSL chain and the prefill staging; a speculative
  verify round still needs the f32 pool and refuses a compact one). Slot bytes from the allocated buffers (real config, bf16
  ring): 194 MiB all f32, 162 ring bf16, 122 + bank, 137 + recurrent, **97 MiB** all three (`RingEngine::state_bytes()`).
  Parity on the real weights (`edgerock_real_weights the_serving_slot`, 1400-token prompt, then 48 greedy decode steps):
  prefill logits cosine 0.9999938 for every tier (reference 220 = greedy), worst bank overlap against the f32 state 99.51%,
  and the 48 decode tokens identical to the f32 state's. Tiny random model (`edgerock_decode`, CPU, default device, CUDA fp32):
  bank bf16 alone moves the hidden state by 7e-4 relative L2, recurrent bf16 by 1e-2 to 5e-2 per step (the random tiny state is
  ill-conditioned; the real model does not show it). `model/tests/anchor_memory` runs the state-form scenario for both bank
  tiers at hd 6 (triad) and 8 (split) against the sequence form.
  Measured effect (full card, position 4096, tape on): slots 194 -> 97 MiB, b256 53.3 -> 51.2 ms; `gdn_decode_pool` 17.4 -> 14.8 ms
  and `anchor_read_split` 14.4 -> 14.5 ms, i.e. both are latency-bound (a 128-register serial FP chain; scalar bf16 loads and
  repeated q loads), not bandwidth-bound; the capacity unlock (256 slots = 25 GiB) is the gain, the speed needs those kernels.
  The memory's fp32 scorer at more than 32 rows takes `matmul_tile16` (exact fp32, k ascending, 16 x 16 tiles, 64-wide k stages,
  padded shared columns; gate `gpu-core/tests/matmul_tile16.rs` against f64): b256 `matmul_gemv_reg` 4.46 ms -> 1.35 ms.
- [x] M6h (partial): the pooled Gated-DeltaNet decode as an occupancy-built candidate (`gdn_decode_pool_wide[_bf16]`,
  `Fused::GdnDecodePoolWide[Bf16]`, `Qwen35::set_gdn_pool_wide`, default on; the bit-identical body stays as the reference
  candidate). A pair of lanes shares each state column (64 registers a thread), the key and query contractions fold with one
  shuffle, the gated norm sums by warp tree, products are fused multiply-adds; the conv windows and gates are the exact
  body's code. Verified against the exact body (`model/tests/gdn_decode_native`): layer output and state within 2e-5
  relative L2 on an f32 pool, within 2e-3 (output) and 6e-3 (state) on a bf16 pool, windows bit for bit. b256 (position 4096,
  bf16 pools): 14.75 -> 11.55 ms for the 24 layers (1.1 TB/s of state traffic, not the 2.5 TB/s target: block latency is
  still about 15 us; unprofiled, no counter tool on the box); at b1 0.38 ms. Forcing 64 registers (two blocks per SM) spills
  and doubles the time.
- [x] M5 (forward part): the released weights against the independent reference
  forward (`tools/goldens/edgerock_dump_real_reference.py`, fp32 CPU torch, a
  1400-token prompt that slides the window and commits five blocks, then asks about
  a fact stated at its start). `crates/qwen35/tests/edgerock_real_weights.rs`
  (ignored: needs the store checkpoint and the dumped golden), bf16 weight tier on
  the CPU backend: every `layer{i}.out` cosine >= 0.9999999997 (worst max_abs
  1.9e-3 at layer 26, rel_l2 <= 2.3e-5), tail logits rel_l2 4.1e-6, the same greedy
  answer, and 99.54-99.85% of the slots of every memory bank equal (near-tied
  scores swap rank). The reference itself answers the probe correctly with the
  memory on (" 4817.", the fact sits ~1380 tokens back, outside the window) and
  wrongly with `read_scale` zeroed (" 48617").

## Not yet done

- [ ] Serving: the tied embedding table is held in fp32 on the device (2.5 GB at the published shape,
  and the head matmul reads it every decode step); a bf16 table needs the head and the embedding
  gather routed through `Ops` like the linears (`table_tier` does it for narrow LoRA training only).
  The memory's projections stay fp32 by design; `infer_footprint_bytes` charges both.
- [ ] Serving: no adapters (the engine reports none, `adapter_availability` refuses), no speculation,
  no embedding-row prompts, no KV offload (nothing to offload: the state is fixed). A prefix entry
  serves one prompt length (the prompt that was filed); there is no mid-prompt checkpoint.
- [ ] Serving: a GPU run of the real model (`edgerock_real_served`, then the sample over HTTP) and the
  throughput of the batched decode at the published shape are unmeasured.
- [ ] M5: real weights (BF16 load, reference parity with the CPU torch reference,
  generation sanity, long-context retrieval).
- [ ] M6: performance, training included: real-weight training throughput and memory (the saved `probs`
  are `[rows * heads, capacity + 1]` per memory layer, held until the backward or recomputed).
  Inference and training: The memory read materialises `[rows * heads, capacity + 1]`
  logits and probabilities and the windowed attention the `[B,H,T,T]` slabs (no
  windowed flash kernel yet); `anchor_commit` ranks by counting, `O((capacity + chunk)^2)`
  per group and block; both want the cooperative/flash treatment per `kernels.md` F.
- [ ] The memory's projections are plain fp32 matmuls over the parameter store: no
  quantized tier, no adapter site. Quantizing them is only safe for the read side
  (`q/k/v/o`, writer); the scorer stays fp32.
- [ ] A real-weight parity run must first confirm `read_scale != 0` on the memory
  layers: a zero read gate silences the memory entirely, and parity would then pass
  without exercising it (the tiny goldens randomise every parameter for the same
  reason).
