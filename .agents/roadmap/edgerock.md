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
