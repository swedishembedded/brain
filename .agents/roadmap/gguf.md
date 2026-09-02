# gguf - roadmap

GGUF, from container/interchange format to execution representation. Brain's
GGUF reader (`crates/checkpoint/src/gguf.rs`) has been a solid mmap-backed
loader for a while: full v2/v3 KV metadata, bounded per-tensor and per-range
decode, and every mainstream GGML block format decoded to fp32. What it was
missing is the thing llama.cpp treats as basic: a Q8_0/Q4_K/etc. tensor does
not have to become fp32 (or brain's own INT8) to be read - it can stay in, or
land in a lossless device-native repack of, its own quantized representation
all the way to the GPU.

This ledger tracks that workstream plus the container-completeness gaps found
alongside it (split GGUF, missing ggml dtypes, embedded chat templates, a
`brain gguf inspect` command). Full design rationale, the corrected premises,
and the milestone-by-milestone plan live in the session that authored this
ledger; this file is the terse, load-bearing status record per `AGENTS.md`'s
own rule that this is where "work still outstanding on ONE model [or
subsystem]" belongs, not `docs/`.

Two facts worth keeping visible because they contradict what an external
review of this codebase assumed:

- `gguf::int8_direct::try_i8_rect` (Q8_0 → brain packed-int8, byte-exact, no
  fp32 anywhere) already existed before this ledger started, with exactly one
  caller (`flux2`). This workstream generalizes it, it does not invent it.
- `qwen35::int8_gguf_resident` does **not** use that fast path - it dequantizes
  every tensor to fp32 before requantizing, same as every other GGUF-backed
  model. It only avoids a *whole-model* fp32 intermediate, not a per-tensor one.

## Done

- [x] M0: `GgmlType` enum (`crates/checkpoint/src/gguf.rs`) collapsing the
      four independent `match ty: u32` tables (`ggml_type_name`,
      `block_geometry`, `tensor_nbytes`'s `block_geometry` call, `dequantize`)
      into one vocabulary with `from_id`/`id`/`name`/`block_elems`/
      `block_bytes`. K-quant variants are spelled `Q4K`/`Q5K`/`Q6K` (no
      underscore before `K`) solely to satisfy `non_camel_case_types` -
      `GgmlType::name()` still returns the conventional `"Q4_K"` spelling.
      `q8_0_expand` generalized to `block_expand(ty: GgmlType, ..)`, with
      `q8_0_expand` kept as a thin `block_expand(GgmlType::Q8_0, ..)` wrapper
      (no call sites touched). `file_type_name` now returns `Option<&'static
      str>` and covers ids 0-32/36/37 (was 0-18); `quant_label` fixed to fall
      back to `general.quantization_version` when `general.file_type` is
      *present but unrecognized*, not only when absent - the old behavior
      returned a fabricated `"unknown"` string that shadowed the fallback.
      `checkpoint::quantize::Tier` extended with `Q4_0`/`Q5_0`/`Q4K`/`Q5K`/
      `Q6K` (was `Q8_0`-only) - `plan`/`convert`/`crate::quant::quantize_par`
      were already generic over block geometry, proven by a new end-to-end
      `convert()` → `MmapGguf` round trip for `Tier::Q4K` (cosine > 0.99,
      rel_l2 < 0.15 - a wiring gate, not a quality claim; `crate::quant`'s own
      `every_type_meets_its_quality_floor` and the M8 relayout gate own
      quality). Gated: `ggml_type_round_trips_and_agrees_with_the_wrapper_fns`
      (every id ↔ `GgmlType` round trip, unknown ids decline), `block_expand_
      matches_whole_tensor_dequantize_for_q4_k` (mid-tensor range, not just
      block 0 - the class of bug a Q8_0-only test suite could not catch),
      `quant_label_falls_back_past_an_unrecognized_file_type`. `make test -p
      brain-checkpoint`: 104 passed, 0 failed.
- [x] M1: `TensorSource::raw_blocks` seam - one new defaulted trait method
      (`crates/checkpoint/src/lib.rs`), so every existing implementor (13
      across the workspace) compiles unchanged. `MmapGguf` implements it for
      real (`gguf.rs`); `raw_tensor_bytes` deliberately stays a SEPARATE,
      independent accessor rather than being reimplemented on top of
      `raw_blocks` - the two have different contracts (`raw_blocks` declines
      for a type `GgmlType` does not recognize; `raw_tensor_bytes` must keep
      working for an unsupported quant type too, because
      `weightio::WeightReader::nbytes` depends on that). `WeightReader` and
      `RemapSource` forward it (`RemapSource::Fetch::Slice` only when the
      slice lands on whole blocks, `Fetch::Concat` always declines - no
      single contiguous byte range to lend). `qwen35::int8_gguf_resident::
      SsmALogFix` gets an EXPLICIT refusal (not just the inherited default)
      per lesson #70: a zero-copy block lend for the transformed
      `linear_attn.A_log` leaf would bypass `ElemOp::LnNeg` and hand a caller
      llama.cpp's untransformed bytes. New `checkpoint::srccheck::
      assert_read_paths_agree` decodes a tensor through every path a source
      offers and asserts they agree - the mechanical version of "did every
      wrapper apply its transform everywhere", proven against both a real
      GGUF-backed source and a deliberately-lying `TensorSource` impl (the
      lying one must be CAUGHT, not just pass). Gated:
      `raw_blocks_forwards_whole_slices_block_aligned_and_declines_concat`
      (`remap.rs`), `raw_blocks_never_lends_the_untransformed_a_log_blocks`
      (`qwen35`), `srccheck::tests::{agreeing_paths_pass,
      a_disagreeing_raw_words_is_caught, a_real_gguf_source_agrees_on_every_
      path}`. `make test -p brain-checkpoint -p brain-qwen35`: green.
- [x] M2: `gguf::int8_direct::try_i8_rect` takes `&dyn TensorSource` instead
      of `&MmapGguf`, reading through `raw_blocks` (M1) instead of
      `MmapGguf::raw_tensor_bytes` directly - now reachable through
      `RemapSource` and any future name-translating shim, not only a bare
      `MmapGguf`. The only call site (`flux2::weights::DitWeights::
      try_i8_rect`) needed one change, a `*gguf` deref: matching
      `DitWeights::Gguf { gguf, .. }` through `&self` binds `gguf` as
      `&&'a MmapGguf` (match ergonomics through the outer reference), and
      `&dyn TensorSource` coercion needs exactly one layer of reference, not
      two - caught by the compiler, not assumed. Behavior and the
      `assert_eq!`-gated bit-exactness proof in
      `flux2/tests/gguf_direct_int8.rs` are unchanged; that test file needed
      no edits at all, which is the point of routing through the trait.
      `make test -p brain-flux2`: green (24 test-result groups, 0 failed).
- [x] M3: `model::int8::{q8_0_rect, quantize_from, quantize_rect_from,
      upload_quantized, upload_rect}` - the one quantize-and-upload helper.
      `q8_0_rect` is the ONE canonical implementation of the Q8_0 byte
      repack; `gguf::int8_direct::try_i8_rect` cannot be it because
      `crates/gguf` depends on `model`, not the reverse, so the algorithm
      has to live in `model` to be shared (a follow-up can turn
      `try_i8_rect` into a thin wrapper over this - not done in this
      milestone, to keep the diff to "add a helper", not "also touch a
      previously-landed, still-correct call site"). `quantize_from` (whole
      tensor) tries the byte repack first, else a row-bounded fp32 route
      (`with_tensor_chunks` cut on row boundaries) - peak host allocation
      one row block, not `n*k`. `quantize_rect_from` generalizes to a
      rectangle: delegates to `quantize_from` for the whole-tensor case,
      else tries the byte repack, else falls back to a whole-tensor fp32
      materialize-and-slice - the same cost `flux2::weights::DitWeights::
      with_f32` already pays for a rect fallback, not a new one this
      introduces. `upload_quantized`/`upload_rect` land the result on the
      device through `paramstore::upload::Uploader`'s `account`/
      `maybe_drain` staging discipline (closing the gap `Weight::upload`
      bypasses by writing through `Gpu::write*_at` directly with no
      accounting). Gated: `q8_0_rect_is_bit_identical_to_the_fp32_round_
      trip` (whole tensor + a nonzero row offset), `q8_0_rect_declines_
      unaligned_bounds`, `quantize_from_matches_across_both_routes` (a
      Q8_0-backed source and a `HashMap` holding the SAME tensor's own
      dequantized values land on identical packed bytes), `quantize_rect_
      from_a_genuine_subrectangle_matches_manual_slicing`, `upload_
      quantized_and_upload_rect_read_back_correctly` (actual device
      read-back, not just "the call returned Ok"). `make test -p
      brain-model`: 163 lib tests passed, 0 failed (one unrelated,
      pre-existing integration-test failure in `moe_compact_parity` - a
      `Gpu::submit` count assertion in `model::moe`, code this milestone
      never touches; almost certainly backend/device-dependent on this
      box, not a regression from this work).

## Not yet done

- [x] M4: migrated `qwen3::q8::Q8::build`, `qwen35moe::q8::Qwen35Q8::build`
      and `wan::block::QLinear::upload` (its Int8 arm only - Int4 has no
      analogous zero-fp32 GGUF fast path, so it keeps its own route) onto
      `model::int8::upload_quantized`. Deleted `qwen35::stream::
      quantize_i8_rows` outright (its body WAS route 2, now generalized);
      both its call sites (`stream::generate_with_stats`'s lm_head,
      `int8_gguf_resident::activate_owned`'s lm_head) now build a
      `paramstore::upload::Uploader` and call `model::int8::
      upload_quantized` directly, reconstructing `Weight::I8` from its
      `(DeviceBuffer, DeviceBuffer)` return. Found and fixed one real gap
      while migrating: `quantize_from`'s bounded fp32 fallback (M3) cut
      chunks at exactly one row (`max_elems = k`), while the code it
      replaces let each caller tune `rows_per_chunk` (4096 at both real call
      sites) - fixed to derive a chunk size from `paramstore::
      UPLOAD_CHUNK_WORDS` instead of hardcoding to one row, matching
      `Uploader`'s own chunk sizing precedent rather than either the
      one-row-at-a-time default or a per-caller tunable. Gated: each of the
      five touched crates individually (`make test -p brain-qwen3`,
      `-p brain-qwen35moe`, `-p brain-wan`, `-p brain-qwen35`, plus M3's own
      `-p brain-model` pass) - all green, 0 failed. (Run individually, not
      batched: `make test`'s 2400s deadlock-guard timeout is a wall-clock
      budget per invocation, and five crates' full suites - unit +
      integration + real-weight parity tests - together exceeded it even
      though none of them hung; that is an orchestration constraint on how
      this ledger's own verification is run, not a property of the code.)
- [x] M5: `checkpoint::gguf_src::GgufSource` (new `crates/checkpoint/src/
      gguf_src.rs`) absorbing wan/ltxv/gemma4's three near-duplicate
      `gguf_src.rs` files, which differed in exactly one expression (the
      name translation) and were otherwise five verbatim forwarding methods
      each. Two constructors cover both shapes found: `renaming(mg, plan:
      HashMap<String,String>)` (wan's lookup table, gemma4's `model.`-prefix
      rewrite, both reduced to building a plan up front) and
      `identity(mg)` (ltxv, no rename step at all). Each model crate keeps
      only its own knowledge - wan's `dit_config_from_shapes`/`source_map`,
      ltxv's `av_dit_config_from_kv`/`validate_av_dit_gguf_shapes`,
      gemma4's architecture check, two-way manifest validation,
      `dtype`/`tokenizer_json`/`tokenizer` - behind a thin wrapper struct
      that forwards `TensorSource` to the shared type; none of the three
      public APIs (`open`/`from_mmap`/`config`) changed shape, so no
      downstream caller needed touching. gemma4 gains `raw_words`,
      `with_tensor_chunks` AND `raw_blocks` it did not have before -
      every GGUF tensor it reads was materializing whole on every read
      path, unnoticed because the boilerplate that would have caught it
      was written three separate times. Gated: three new tests on
      `GgufSource` itself (`renaming_translates_every_read_path_including_
      raw_blocks`, `identity_needs_no_plan_entries_written_by_hand`,
      `raw_blocks_is_reachable_through_a_rename` - the last one is the
      capability none of the three original wrappers had), plus every
      pre-existing wan/ltxv/gemma4 `gguf_src` test unchanged and still
      green. `make test -p brain-checkpoint -p brain-wan -p brain-ltxv
      -p brain-gemma4`: green, 0 failed (run as one combined invocation at
      39m3s - close to the 2400s/40min deadlock-guard budget; the next
      milestone should go back to separate per-crate runs).
- [x] M6: `flux2::weights::DitWeights` implements `checkpoint::TensorSource`
      directly. `with_f32` (the panic-on-missing inherent method) is now
      built ON `with_tensor` (the bool-returning trait method) rather than
      duplicating its decode-and-LoRA-fold-and-cache body - the two used to
      be two independent copies of the same logic with different failure
      contracts. `try_i8_rect`'s LoRA-touched decline moved into `raw_blocks`
      (the ONE place it is now checked - `try_i8_rect` itself just calls
      `gguf::try_i8_rect(self, ..)` and inherits the decline for free,
      instead of re-checking `lora.touches(..)` a second time). Deviated
      from the plan's literal "`lin_rect` -> `upload_rect`" for
      `crates/flux2/src/model.rs`'s actual per-linear upload closure: it
      carries real, consumed profiling instrumentation (`quant_ns`/
      `write_ns`/`split_ns`/`flush_ns`, printed as a load-time perf
      breakdown at the end of the build) that `model::int8::upload_rect`
      has no equivalent for, and swapping it in would delete that
      observability for a real, expensive, already-tuned hot path with no
      correctness upside - `lin_rect` is not a "no-policy" site the way
      M4's targets were. Left it as-is; `try_i8_rect` was already the
      generalized primitive, and `DitWeights` implementing `TensorSource`
      is what "proves the seam" - any future generic caller
      (`model::int8::upload_rect` included) now works against a
      `DitWeights` directly. Gated: two new tests in `weights.rs`'s own
      `mod tests` (none existed before) - `dit_weights_is_usable_as_a_
      trait_object` and `a_lora_touched_tensor_declines_the_direct_path_
      but_still_folds_via_with_f32`, the latter closing a real, previously
      untested gap (no test anywhere exercised the LoRA-touched decline
      path before this, via a `PendingLora` built directly with public
      `model::lora::ExternalPair` fields rather than a file-based fixture).
      `make test -p brain-flux2`: green (exit 0; every pre-existing test,
      including the `gguf_direct_int8.rs` bit-exactness suite, unaffected).
- [x] M7 (s3dit slice): `crates/s3dit/src/block.rs`'s `quantize_block` was
      already the most migrated of the three `&Tensors` → `&dyn TensorSource`
      targets - its signature already took `t: &dyn checkpoint::TensorSource`
      - but its inner `q` closure still forced every linear through the fp32
      route unconditionally (`t.with_tensor(&name, &mut |data| .. quantize_
      weight(data, no, k))`), never trying the zero-fp32 Q8_0 byte repack even
      when `t` could serve one directly. Swapped that hand-rolled `with_tensor`
      + `quantize_weight` dance for a single call to `model::int8::quantize_
      from(t, &name, no, k)`, which tries the byte repack first and only falls
      back to the bounded fp32 route when the source can't serve one - the
      same choice M4 already wired into `qwen3`/`qwen35moe`/`wan`, now reaching
      this crate too. `crate::int8::quantize_weight`'s re-export
      (`crates/s3dit/src/int8.rs`) stays - `tests/int8_matmul.rs` calls it
      directly and nothing else in the crate needed touching; grepped for any
      other hand-rolled `with_tensor` + `quantize_weight`/`quantize_weight_q4`
      pattern outside `quantize_block` and found none. `make test -p
      brain-s3dit`: green, 0 failed. ltxv and gemma4 still need their own
      `&Tensors` → `&dyn TensorSource` migration (plus this same `quantize_
      from` wiring once migrated) - M7 stays open until those land.
- [x] M7 (ltxv slice): `crates/ltxv/src/block.rs`'s `QLinear::quantize_host`
      took `t: &Tensors` - the eager, wholly-materialized checkpoint map -
      even for the int8 tier, so a GGUF-backed `Q8_0` tensor had to be
      decoded to fp32 in full before this function ever saw it. Changed the
      signature to `t: &dyn checkpoint::TensorSource`; `Tensors` already
      implements `TensorSource` via `checkpoint::lib`'s blanket impl for
      `HashMap<String, (Vec<usize>, Vec<f32>)>`, so every existing caller
      (`QAttnWeights::quantize_host`, `QFfWeights::quantize_host`,
      `QBlockWeights::quantize_host`/`quantize_host_stream`, all still typed
      `&Tensors`) kept compiling unchanged - Rust unsize-coerces `&Tensors`
      to `&dyn TensorSource` at the call site, no caller edits needed. Inside
      the function, the `QTier::Int8` arm's hand-rolled `tget` (Tensors-only)
      + `model::int8::quantize_weight(data, out_dim, in_dim)` pair became one
      call to `model::int8::quantize_from(t, &weight_name, out_dim, in_dim)`,
      which tries the zero-fp32 `Q8_0` byte repack first and only falls back
      to the bounded fp32 route when the source can't serve one directly -
      the same choice M4 wired into `qwen3`/`qwen35moe`/`wan` and the s3dit
      slice above wired into `s3dit`, now reaching this crate too; a `None`
      return panics with a clear message, matching `tget`'s own
      panic-on-missing convention. `QTier::Int4` has no such shared fast
      path (`model::int4` has nothing analogous to `quantize_from`), so it
      keeps reading via `t.with_tensor(&weight_name, |data| model::int4::
      quantize_weight_q4(data, out_dim, in_dim))` - the same choice M4 left
      `wan::block::QLinear::upload`'s Int4 arm making. The bias read also
      moved off `tget` (Tensors-only) to `t.with_tensor`, panicking if
      `has_bias` is true but the tensor is absent (a static flag mismatching
      the checkpoint is a real bug, not a normal absence). Grepped
      `crates/ltxv/src/dit.rs` and `crates/ltxv/src/na_decoder.rs` (each
      defines its own local `tget` over the same `Tensors` type) for the same
      hand-rolled `with_tensor` + `quantize_weight`-by-hand shape and found
      none - every `tget` call in both files is a plain fp32 norm/bias read,
      no quantization involved. `make test -p brain-ltxv`: green, 0 failed
      (47 test binaries; the quantization-relevant suites - `gguf_quant_real`,
      `int8_compute` (including `real_q8_0_block0_int8_compute_matches_fp32`),
      `int8_storage`, `dit_parity` (including `real_weight::
      ltxv_real_dit_tiny_layers_matches_reference`), `na_decoder_parity` -
      all pass). gemma4 still needs its own `&Tensors` → `&dyn TensorSource`
      migration - M7 stays open until that lands too.
- [ ] M7: gemma4 `&Tensors` → `&dyn TensorSource` (s3dit's and ltxv's slices
      are done - see the M7 entries above).
- [ ] M8: host relayout (`crates/gguf/src/kquant.rs`) for Q4_K/Q5_K/Q6_K/
      Q4_0/Q5_0/Q8_0 into brain's device K-quant layout. Gate: `assert_eq!`
      round trip against `deq_*`, no tolerance.
- [ ] M9: `quant_group_sum.wgsl` int8-activation prepass (`xgs`) - the affine
      correction term's `Σ xq` piece, computed once, not per column tile.
- [ ] M10: group-16 (Q6_K) + legacy types through EXISTING kernels via
      template knobs (`WPG`→4, new `QPG` knob on `matmul_i8_dyn`) - zero new
      `.wgsl` files.
- [ ] M11: `matmul_kq_dyn.wgsl` + `matmul_kq_gemv.wgsl` - the new affine
      Q4_K/Q5_K GEMM/GEMV kernels.
- [ ] M12: the selection seam - `DType::{Q4K,Q8K}`, `Weight::KQuant`,
      `Ops::{bind,threads,matmul}` (`threads` must route the new dtypes to
      `tile()`, not the `_ => m*n` arm - that under-dispatch corrupts output,
      not just wastes workgroups).
- [ ] M13: `moe_linear_gated_kq.wgsl` + `matmul_kq_gemv_reg.wgsl` +
      `gpu_core::upgrade` row.
- [ ] M14: byte compression (packed `sc`/`m` + f16 `d`; Q4_K → 1.03× GGUF).
- [ ] M15: split GGUF in `MmapGguf` (`mmaps: Vec<Mmap>`, part in the tensor
      index, one `LoadMeter` over summed bytes - all 17 method signatures
      stay byte-identical, no `Inner::GgufSharded` arm needed).
- [ ] M16: split GGUF in `modelstore`/CLI (hoisted `-NNNNN-of-NNNNN` parser;
      `quant_of_gguf`/`pick_gguf`/`local_quant`/`scan_repo_dir`).
- [ ] M17: ggml ids 9 (Q8_1) and 24-28 (I8/I16/I32/I64/F64) - these fail
      `MmapGguf::open` OUTRIGHT today via `tensor_nbytes` → `None`, a
      whole-file refusal for a type usually carried by a metadata tensor
      nobody reads. Worst failure shape in the set.
- [ ] M18: codebook families - MXFP4 first (gpt-oss's native release format,
      simplest of the set), then IQ4_XS/IQ4_NL, then the rest, then TQ.
- [ ] M19: write side - `Tier` gains the 10 variants
      `crate::quant::quantize`/`quantize_par` already encode;
      `quantize_cli.rs` accepts them by name; `general.file_type` derived from
      the tier via M0's table instead of the hardcoded `7`.
- [ ] M20: `GgufTokenizer` completeness (`chat_template` + named variants,
      `add_bos_token`/`add_eos_token`, eot/eom, `scores`,
      `precompiled_charsmap`, `fim_*`); `ChatTemplate::from_gguf` as
      `from_model_dir`'s sibling.
- [ ] M21: wire the GGUF tokenizer fallback into `resident_qwen35.rs`/
      `resident_qwen35moe.rs` (`resident_llm.rs` already has it for qwen3).
- [ ] M22: `brain gguf inspect PATH [--json]` - no `kv()` is reachable from
      the CLI today (`brain models info` goes through `WeightReader`, which
      exposes no KV accessor).

Docs to fix alongside the relevant milestones: `docs/models/qwen3.md:33-52`
still demonstrates converting Qwen3 GGUF → safetensors even though the
registry declares qwen3 `[direct]`; `docs/using/models-and-weights.md`
needs the native K-quant tier documented once M12 lands.

## Recorded gaps (this development machine has no MXFP4/IQ fixture and only
one small Q8_0 in the model store)

- The store holds only `Qwen/Qwen3-0.6B/Q8_0.gguf` (610 MB). There is a
  17 GB Q4_K_M under `~/Downloads/MiniMax-H3/` but it is not in the store, and
  there is no MXFP4 or IQ*-quantized file anywhere on this box.
- Every gate through M14 is synthetic and exactly-known by construction (per
  the user's explicit instruction - validate the math, do not fetch real
  checkpoints for this workstream). The end-to-end forward-parity rung (rung F
  in the design: `Weight::upload(F32, deq(gguf))` vs `from_kquant`, gated
  behind `BRAIN_*_GGUF`) is written and will self-skip on this box until a
  real K-quant checkpoint is available.
- `PARITY_STRICT_SUITES` in the root `Makefile` does not yet include
  `brain-qwen3:gguf_vs_safetensors_real` or `brain-ltxv:gguf_quant_real` -
  candidates to add once real checkpoints are on hand.
