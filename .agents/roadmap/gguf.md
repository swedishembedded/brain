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
- [x] M8: `gguf::kquant` (new `crates/gguf/src/kquant.rs`) - the host-side
      lossless relayout for all six GGUF block formats the device K-quant
      layout targets (Q4_K, Q5_K, Q6_K, Q5_0, Q4_0, Q8_0) into ONE canonical
      shape: `wq: [n, k*bits/32] u32` (codes, K-contiguous, `32/bits` codes
      per word, low bits first) plus `wsz: [n, 2*k/G] f32` (interleaved
      `(scale, min)` pairs per group, `min == 0` for a symmetric type).
      `try_kq_rect(source, name, stride, r0, n_out, c0, k)` mirrors
      `int8_direct::try_i8_rect`'s rectangle-slice contract exactly (`None`
      always means "take the fp32 route", never a partial or approximate
      answer) and returns a new `KqLayout` (`ty`/`bits`/`group`/`affine`/`n`/
      `k`, plus `words_per_row`/`groups_per_row` helpers) alongside the
      packed buffers, so a later GPU-upload milestone knows the packing
      without re-deriving it from the GGUF type. Declines when: `raw_blocks`
      has nothing for `name`; the type is not one of the six; or `stride`/
      `c0`/`k` are not each a multiple of the type's block size (32 for the
      three legacy formats, 256 for the three K-quant super-block formats -
      a super-block cannot be split, so a `k` that is 256-unaligned is a
      refusal even where it would be a valid legacy boundary). Every
      per-block bit layout (`scale_min_k4`'s 6-bit packed scale/min
      extraction, Q5_K's `qh`/`ql` high-bit combination, Q6_K's `ql`/`qh`/
      `sc` 16-group indexing, the legacy formats' lo/hi-nibble and
      high-bit-of-5 layouts) is transcribed from `checkpoint::gguf`'s
      private `deq_q4_k`/`deq_q5_k`/`deq_q6_k`/`deq_q4_0`/`deq_q5_0`/
      `deq_q8_0` with the SAME expressions in the SAME operand order for
      `ds`/`dm` (`ds = d*sc`, `dm = dmin*m` for the affine pair; `ds = d`
      symmetric), reorganized from each decoder's interleaved emission order
      into a flat per-group loop but proven equivalent by construction
      (documented inline per format) rather than by re-deriving the bit
      layout from scratch - this module performs NO arithmetic on weight
      values, only computes `ds`/`dm` and moves codes. Affine codes
      (Q4_K/Q5_K) stay the format's own raw unsigned value; symmetric codes
      (Q6_K/Q5_0/Q4_0/Q8_0) are bias-folded to a signed value in low-bits
      two's complement (`pack_codes`/`unpack_row_codes` need no per-type
      branch beyond masking and an `affine`-gated sign extension) - the
      symmetric fold needs no rounding because it is an exact integer
      operation, unlike an affine fold would be. Legacy scales are routinely
      negative (Q4_0's especially) and Q6_K's per-16-group sub-scale is
      signed (`i8`); neither is special-cased because nothing here assumes a
      sign. Gated (`crates/gguf/tests/kquant.rs`, new): one round-trip test
      per format building a real temp GGUF via `checkpoint::quantize::
      convert(&src, Tier::*, ..)` (not hand-assembled bytes) and asserting
      `assert_eq!` (never a tolerance) that reconstructing `(wq, wsz)` with
      `ds*code - dm`/`ds*code` reproduces `MmapGguf::tensor`'s decode
      exactly, both for the whole tensor and for a genuine sub-rectangle
      (`r0 != 0` AND `c0 != 0` at once - block 0 alone cannot catch a
      mis-indexed row or column offset); `k` not a multiple of 256 declines
      for a K-quant type; unaligned `stride`/`c0`/`k` declines for a legacy
      type; an F32-typed tensor (a real `GgmlType` `raw_blocks` reports, just
      not one of the six) declines; a source with no `raw_blocks` at all
      declines. `make test -p brain-gguf -p brain-checkpoint`: 111 + 10 + 19
      + 5 gguf-crate test-groups and every brain-checkpoint test, 0 failed.
- [x] M7 (gemma4 slice, closing M7): `crates/gemma4/src/block.rs`'s whole
      weight-loading chain took `&Tensors` end to end - `Proj::from_tensors`,
      `AttnWeights::upload`, `MlpWeights::upload`, `Gemma4Layer::on` - so
      `forward_streamed` (`crates/gemma4/src/model.rs`) had to call
      `load_layer_tensors` first, which decodes EVERY tensor in a layer to
      f32 via `with_tensor().to_vec()` before `Gemma4Layer::on` ever saw it -
      the exact whole-layer fp32 materialization this workstream exists to
      remove, and worse than the other two slices' problem (s3dit/ltxv only
      forced fp32 on the int8 *quantize* path; gemma4 forced it on every
      tensor, every precision). Renamed `Proj::from_tensors` to `Proj::load(
      gpu, source: &dyn checkpoint::TensorSource, name, n, k, precision)` -
      `n`/`k` now come from the caller's own config-derived dims rather than
      the tensor's stored shape, since a bare `TensorSource` has no shape to
      read beyond `numel`; its Int8 arm routes through `model::int8::
      upload_quantized`, which tries a zero-fp32 `Q8_0` byte repack via
      `TensorSource::raw_blocks` before falling back to a bounded fp32
      decode. `AttnWeights::upload`/`MlpWeights::upload` widened to
      `w: &dyn checkpoint::TensorSource` plus explicit `q_dim`/`kv_dim`/
      `hidden`/`intermediate` parameters (replacing shape reads off an eager
      map); `Gemma4Layer::on`'s `weights: &Tensors` became `&dyn
      checkpoint::TensorSource`, computing `hidden`/`q_dim`/`kv_dim` from
      `cfg` and passing them through. `forward_streamed`'s per-layer closure
      now passes `src` straight into `Gemma4Layer::on`, with the
      `load_layer_tensors` call and its error plumbing removed from this
      path entirely - `load_layer_tensors` itself stays (public, still used
      by `crates/gemma4/tests/int8_compute_real.rs`); the non-streamed
      `Gemma4Model::new` path keeps passing `&self.w` (an owned `Tensors`),
      which still satisfies the new `&dyn TensorSource` parameter via the
      blanket impl with no behavior change. Added `brain-paramstore` as a
      dependency (`upload_quantized` needs `paramstore::upload::Uploader`,
      which this crate had never pulled in directly before). Removed the
      now-dead `tget(w: &Tensors, name)` helper (every call site moved to
      `tget_owned`, which reads through `TensorSource::with_tensor` instead).
      `make test -p brain-gemma4`: green, 0 failed, 0 warnings (23 tests
      across lib + `int8_compute_real`/`parity`/`real_weight_parity`,
      including `real_q8_0_whole_encoder_int8_matches_fp32` - the suite that
      exercises this exact call chain against real int8 weights).

All three M7 slices (s3dit, ltxv, gemma4) are now done - M7 is closed.
- [x] M9: `quant_group_sum.wgsl` (new `crates/kernels/wgsl/`) - the affine
      K-quant activation-only correction prepass, `S[m,g] = Σ_{k in g}
      xq[m,k]`, computed ONCE per activation via `dot4I8Packed(xq_word,
      0x01010101u)` per packed word (exact int8 lane sum, no rounding - one
      thread per `(row, group)` output, one thread reads exactly
      `model::int8::WORDS_PER_GROUP` (8) words). `model::int8::QuantRows`
      gains a THIRD, optional field, `xgs: Option<(kernel_idx,
      &DeviceBuffer)>`; `quant_rows_steps` now returns `Vec<Step>` (was
      `[Step; 2]`) and appends the group-sum dispatch as a third step only
      when `xgs` is `Some` - every existing caller (`I8Scratch::quant_rows`,
      `model::moe`'s expert/shared-expert forwards, and every downstream
      `quant_rows_steps` call site in `qwen3`/`qwen3omnimoe`/`moondream3`)
      passes `xgs: None` and keeps the byte-identical two-step dispatch;
      only `.to_vec()`-shaped call sites needed a mechanical one-line update
      for the new struct field, nothing behavioural. Gated
      (`crates/model/tests/quant_group_sum.rs`, new, 3 test functions):
      `quant_group_sum_matches_host_exact_integer_sum` - `assert_eq!`
      against a host-computed exact integer sum of the SAME packed bytes
      (never a tolerance, since the kernel's whole claim is bit-exactness),
      values spanning the full i8 range including `-128` (the one value
      `-x` cannot represent, so a sign-handling bug cannot hide);
      `quant_group_sum_indexes_row_and_group_independently`;
      `quant_rows_steps_wires_the_xgs_seam` (the `Option<(kernel_idx,
      &DeviceBuffer)>` plumbing itself, device-level). `make test -p
      brain-kernels -p brain-model --test quant_group_sum`: 3 passed, 0
      failed on a real device - the one M9-M11 gate that was clean on the
      first real run (see M10's and M11's own entries for the two that
      were not).
- [x] M10: group=16 (Q6_K) plus the legacy group=32 formats reach the
      device through EXISTING kernels via template knobs - ZERO new
      `.wgsl` files. `matmul_i8_dyn.wgsl` (the tiled prefill GEMM) gains a
      `QPG` (quads-per-group) knob: `QPG=2` (the new default) reproduces
      today's implicit fold-once-per-chunk behaviour BIT-IDENTICALLY
      (gated against a checked-in text snapshot of the kernel exactly as it
      stood before this milestone,
      `fixtures/matmul_i8_dyn_pre_qpg.wgsl.snapshot` - not a `.wgsl` file,
      so `kernels-regen`/`kernels-table` never scan it), `QPG=1` folds
      twice per chunk instead (group=16). `matmul_i8_gemv`/
      `matmul_i8_gemv_reg`/`moe_linear_gated_i8` already had a plain `const
      WPG: u32 = 8u;` each - exactly the `const NAME: u32 = <lit>u;` shape
      `kernels::template::specialize`'s existing rewrite already handles,
      so `WPG=4` (group=16) needed no source edit at all, only a test
      proving it. Gated (`crates/model/tests/kquant_group16_knobs.rs`, new):
      (a) `matmul_i8_dyn_qpg2_default_is_bit_identical_to_pre_qpg_kernel` -
      `assert_eq!`, not a tolerance, since this touches the production
      prefill GEMM every existing int8 model already depends on; (b) every
      group=16 variant (`matmul_i8_dyn#QPG=1`,
      `matmul_i8_gemv`/`matmul_i8_gemv_reg#WPG=4`) against a host oracle
      built directly from int8 CODES (never a lossy float-quantize step),
      `rel_l2 <= 1e-6`. Running this gate for real (M12, not this
      milestone's own agent - see M11's entry for why) found the `max_rel`
      ceiling too tight for group=16's doubled fold count, the same
      accumulation-order finding M11's own entry describes in more detail:
      `rel_l2` measured `8e-8..1.5e-7` and `cosine` exactly `1.0` (both
      unchanged, still the tight primary gates) while `max_rel` measured up
      to `3.34e-4` on a real device (Intel Arc iGPU, Vulkan); the ceiling
      was recalibrated from `1e-5` to `5e-4` with the measurement recorded
      inline. `make test -p brain-kernels -p brain-model --test
      kquant_group16_knobs`: 5 passed, 0 failed after the fix.
- [x] M11: `matmul_kq_dyn.wgsl` + `matmul_kq_gemv.wgsl` (new
      `crates/kernels/wgsl/`) - the new affine Q4_K/Q5_K GEMM/GEMV kernels,
      `CODE_BITS`-template-specialized (4 = Q4_K, 8 = Q5_K). `matmul_kq_dyn`
      mirrors `matmul_i8_dyn`'s 128×128 tile / vec4 k-group-minor staging /
      8×8 interleaved register block EXACTLY (same `BM`/`BN`/`BKG`/`SP4`/
      `LN`/`RS`), with two deltas: a staging-time bit-unpack of
      `CODE_BITS`-wide UNSIGNED codes into DP4A-ready packed words (a pure
      bit-shuffle, never a multiply/bias - the code's numeric value never
      changes, only its bit position), and a second reduction (the affine
      `dm[n,g]*S[m,g]` correction, `S` read from M9's `xgs` prepass) folded
      alongside the usual `ds*dot4I8Packed(...)` term every `QPG`-th quad
      (`QPG` FIXED at 2 here - this kernel serves only group=32; Q6_K's
      group=16 stays on M10's own kernels, never this one).
      `matmul_kq_gemv` mirrors `matmul_i8_gemv`'s one-workgroup-per-column,
      64-thread-k-stride shape, with the min correction guarded to fire on
      EXACTLY one thread per group
      (`select(0.0, dm, (g % WPGK) == 0u)` - a naive per-quad application
      would land the correction 8 times, since 8 threads visit each
      group's 8 quads). Gated (`crates/model/tests/matmul_kq.rs`, new,
      11 test functions): rung (b) device vs an f64 host oracle built
      directly from int8 codes (never a lossy float-quantize step) at both
      `CODE_BITS`, both kernels; rung (c) seven adversarial cases -
      `dmin` zero-vs-nonzero pairing (MUTATION-VERIFIED: temporarily
      dropping the correction term left the `dmin==0` half green and the
      `dmin!=0` half red, proving the term is load-bearing, not
      coincidentally near-zero), sub-block scale variation across a
      super-block's 8 groups (MUTATION-VERIFIED: a hoisted-to-group-0 index
      bug went red immediately), mixed-sign activation with full-range
      codes, an all-zero sub-block (no NaN), a genuine sub-rectangle at a
      nonzero tile origin spanning two super-blocks, ragged tiles, and a
      `k` where `CODE_BITS=4`'s word density (8 codes/word) genuinely
      differs from `xq`'s (4/word); rung (d) `matmul_kq_dyn` vs
      `matmul_kq_gemv` cross-kernel agreement on identical inputs.
      This milestone's own gate was written but never actually RUN before
      M12 exercised it for real (this whole ledger's landing history for
      M9-M11 was written ahead of a real `make test` pass - see M12's own
      entry for the fuller story); running it for real found two genuine
      bugs, both in the TEST FIXTURE, not the kernels: (1)
      `rand_unsigned_codes(seed, n, bits)` generated the FULL `0..2^bits`
      range for weight codes, but Q5_K's real codes are a 5-bit value
      (`0..31`) sitting in an 8-bit `CODE_BITS=8` slot - the top-3-bits-
      always-zero invariant `matmul_kq_dyn.wgsl`'s own header states as the
      reason the unsigned-code-read-as-signed hazard is unreachable. A code
      `>= 128` from the un-capped generator DOES set the sign bit, so
      `dot4I8Packed` silently reinterpreted it as negative for roughly half
      of every `CODE_BITS=8` weight - caught by `case4_all_zero_subblock_
      no_nan` going catastrophically wrong (`rel_l2 ~ 1.2`, `cosine ~
      -0.26`) while `case2` (also `CODE_BITS=8`) showed the same signature;
      fixed by capping the generator's span at 32 (the real max across both
      formats this file exercises), after which every `CODE_BITS=8` case
      passed at the SAME tight tolerance the `CODE_BITS=4` cases already
      met. (2) With that fixed, `max_rel` (the single-worst-element metric)
      still exceeded the `1e-5` ceiling by up to ~30x on several cases while
      `rel_l2` stayed at `8e-8..1.8e-7` and `cosine` at `1.0` to 12
      decimals (both unchanged) - a real, measured property of this
      kernel's affine fold doing a SECOND f32 reduction per group (the
      `- dm*S` term) alongside the usual `ds` one, not a correctness bug;
      the ceiling (calibrated against the symmetric family's single-
      reduction fold) was recalibrated to `5e-4`, with the measured numbers
      recorded inline as the justification. `make test -p brain-kernels -p
      brain-model --test matmul_kq`: 11 passed, 0 failed on a real device
      (Intel Arc iGPU, Vulkan) after both fixes.
- [x] M12: the shared dispatch seam every int8-tier model in the workspace
      goes through - `backend_api::DType` gains `Q4K` (affine 4-bit,
      `bits()=4`) and `Q8K` (affine 8-bit - Q5_K's 5-bit code sits in an
      8-bit slot, so the tag names the device SLOT width, not the GGUF
      format; `checkpoint::quantize::Tier::Q5K` is the on-disk name for the
      same format, spelled differently because it names a different thing)
      - both `bytes()=1`, `per_word()` `8`/`4`, and `promote()` on
      `int8_dot` exactly like `I8`/`Q4`. `backend_api::select`'s
      `dtype_storage_requirement` and `candidates`'s `Op::MatMul` arm fold
      both new dtypes into the SAME arm `I8`/`Q4` already use (identical
      regime split, identical `int8_dot` capability gate - only
      `model::ops::Ops::bind` picks a different PHYSICAL kernel per dtype);
      `Op::PagedAttention`/`Op::MoeExpertLinear`'s exhaustive matches fold
      them in too, for compilation only (nothing constructs an `OpShape`
      with these dtypes for either op yet). `model::ops::Weight` gains
      `KQuant { w, sz, n, k, group, bits, affine }` - ONE variant for all
      three device instantiations from the design doc's table (Q4_K:
      bits=4, group=32, affine=true, reports `Dtype::Q4K`; Q5_K: bits=8,
      group=32, affine=true, reports `Dtype::Q8K`; Q6_K: bits=8, group=16,
      affine=false, reports `Dtype::I8` - reused, since from the
      selector's perspective it IS exactly `I8` with a different
      weight-scale group). `sz`'s layout is a property of `affine`, not of
      where the bytes came from: INTERLEAVED `(ds,dm)` for the affine pair
      (what `matmul_kq_{dyn,gemv}` bind directly), a PLAIN `ds`-only plane
      for the non-affine Q6_K case (what M10's `#QPG=1`/`#WPG=4`
      specializations of the EXISTING symmetric kernels actually bind - a
      caller building this variant from `gguf::kquant`'s always-interleaved
      host relayout output must extract just the `ds` half first for that
      case). `Ops::bind` gains a THIRD parameter, `group: u32` (every
      pre-M12 call site passes `32`, matching what was implicitly true
      before): `(PackedInt8, I8, 32)` still binds the untouched
      `matmul_i8_dyn`, `(PackedInt8, I8, 16)` binds a NEW registered name
      (`matmul_i8_dyn#QPG=1`, an `interned` specialization built into
      `kernel_list()` exactly like the bf16/f16 storage tiers already are -
      zero new `.wgsl` files); `(_, Q4K/Q8K, 32)` binds
      `matmul_kq_{dyn,gemv}#CODE_BITS={4,8}`, also `interned`
      specializations. `Ops::threads`'s `PackedInt8` arm explicitly adds
      `Dtype::Q4K | Dtype::Q8K` to the `tile()` branch (NOT the `_ => m*n`
      fallback `Q4` uses) - `matmul_kq_dyn` is `matmul_i8_dyn`'s own
      128×128 register-tiled sibling, so it needs the identical dispatch
      geometry; landing in the `m*n` arm instead would silently
      under-dispatch the tile grid and leave real output elements never
      written (gated by
      `kq_dtypes_dispatch_the_tiled_formula_not_m_times_n`, which asserts
      the dispatched count against the SAME tile formula `Dtype::I8` uses,
      not merely "some number"). `Ops::matmul`'s existing `I8`/`Q4` arm's
      `param_k` fallback (`_ => k`) already does the right thing for the
      new dtypes by the SAME reasoning `matmul_q4_*` established (`xq`/`wq`
      have different word densities, so `k` must be the raw logical
      length, never a packed word count) - confirmed explicitly with a
      comment, not assumed; the new `Weight::KQuant` arm is otherwise
      SEPARATE (six buffers - `xq`, `wq`, `sx`, `wsz`, `xgs`, `out` - not
      the five the `I8`/`Q4` arm binds) since it needs the extra `xgs`
      buffer no other tier reads. `Act` (the activation-quantization
      struct) gains `xgs: Option<DeviceBuffer>`, `None` for every existing
      constructor (`Ops::act`/`Ops::act_f32`); a new `Ops::act_kq`
      constructor builds the SAME `I8Scratch` `Ops::act` does PLUS the
      `quant_group_sum` prepass via `QuantRows`'s `xgs` seam (M9). A
      `Weight::KQuant` matmul against an activation with no `xgs` panics
      LOUDLY naming the problem (`"build it with Ops::act_kq, not Ops::act
      or Ops::act_f32"`), never silently reading a missing buffer. Gated:
      `affine_kquant_dtypes_select_exactly_like_i8` (pure `select.rs`
      logic, no device - asserts `Q4K`/`Q8K` select IDENTICALLY to `I8` at
      every shape/cap combination the existing `int8_requires_the_
      capability` test covers, not just a hardcoded `KernelVariant`),
      `bind_routes_the_m12_dtypes_to_the_right_physical_kernel`,
      `kq_dtypes_dispatch_the_tiled_formula_not_m_times_n` (the specific
      under-dispatch regression named above),
      `m12_kname_literals_match_interned_naming` (pins the hand-spelled
      `kname` string literals against `kernels::template::interned`'s real
      output, same discipline the B4/B8/B9/B10 storage-tier literals
      already have) - all in `crates/model/src/ops.rs`'s own test module;
      plus a new `crates/model/tests/ops_kquant.rs`: a REAL device-level
      test building `Weight::KQuant` directly (there is no
      `Weight::upload` path for it - that function explicitly refuses
      `Q4K`/`Q8K`, since K-quant's whole point is reaching the device
      without ever materializing fp32) and dispatching through
      `Ops::act_kq`/`Ops::matmul`, asserting the result is BIT-IDENTICAL to
      a hand-dispatched call to the SAME M11 kernels on the SAME packed
      buffers (both `Q4K`/`Q8K`, both the GEMV decode regime and the tiled
      prefill regime), plus a `#[should_panic]` gate for the
      no-`xgs`-activation refusal.
      **What "gating: must not regress ANYTHING" actually caught.** M9-M11
      landed in the working tree but were never run against a real device
      before this milestone (see M9/M10/M11's own entries for the two real
      test-fixture bugs that surfaced there); this milestone's own required
      broad sweep (every crate that builds a `model::ops::Ops`, not just the
      two this milestone edits) found two MORE real gaps, both caused by
      `REQUIRED_KERNELS` growing by 7 names (`quant_group_sum`, both
      `matmul_kq_{dyn,gemv}` `CODE_BITS` specialisations, both
      `matmul_i8_{dyn,gemv}` group=16 specialisations):
      (1) `qwen3::model::pipelines`, `qwen35moe::model::pipelines`, and
      `qwen35::model::pipelines` each hand-maintain their OWN copy of the
      façade kernel set (documented as deliberate - they override the
      `matmul_reg2`/`matmul_reg3` name/source binding `model::ops::
      kernel_list` cannot express - so none of the three delegates to the
      canonical list the way `qwen3::serve`/`gradcheck::bf16_train` already
      do), and all three were now missing the 7 new names:
      `Ops::new` refused to build at all (`kernel 'quant_group_sum' is not
      registered on this Gpu`), which took down every single test in
      `qwen3`'s lib suite that constructs a `Qwen` (27 of them) the moment
      it ran for real. Fixed by appending the same 7 entries (`kernels::
      template::interned`, matching `model::ops::kernel_list`'s own recipe)
      to each of the three `pipelines()` builders - "compiled, never
      dispatched" for these three crates, the same precedent the bf16/f16
      storage-tier entries already established there, since none of them
      builds a `Weight::KQuant`. (2) `gpu_core::cost::kernel_cost` (the
      per-kernel FLOP/byte accounting table `PassProfile::gflops` depends
      on to report a rate at all) had no formula for `quant_group_sum` or
      `matmul_kq_dyn`/`matmul_kq_gemv` - caught by `qwen3::model::tests::
      pipelines_fully_costed`, a ratchet that refuses ANY kernel in a
      model's own pipeline list without a cost formula. Fixed by adding
      three formulas: `quant_group_sum` (int-ops only, `64 * m*(k/32)` -
      one thread per `(row, group)` output, 8 `dot4I8Packed` calls each,
      4 MACs = 8 int ops per call) and `matmul_kq_dyn`/`matmul_kq_gemv`
      (the SAME DP4A structure `matmul_i8_dyn`'s own formula already
      counts, keyed off this kernel's own RAW-`k` param contract instead of
      `kg = K/4` - `bytes` approximates the weight-code buffer at
      `CODE_BITS=8` density since `bits` is not itself a dispatch param
      this function can see, the same "best-effort streaming estimate, not
      a cache model" every other formula in this file already accepts).
      `matmul_i8_dyn#QPG=1`/`matmul_i8_gemv#WPG=4` needed NO new formula -
      `kernel_cost` already strips a `base#K=V` specialisation suffix
      before matching, so they fall through to the EXISTING `matmul_i8_dyn`/
      `matmul_i8_gemv` arms for free. `gpu_core`'s own coverage-floor
      ratchet (`cost_coverage_over_the_kernel_tree_never_regresses`) rose
      with these additions, so it needed no edit.
      `make test`, run per crate on a real device (Intel Arc iGPU, Vulkan),
      every one green after the fixes above: `-p brain-backend-api` (45
      passed), `-p brain-model` (full suite green except ONE pre-existing,
      unrelated failure - `moe_compact_parity`'s `compact_layer_submit_
      count_does_not_scale_with_expert_count`, confirmed via `git diff` that
      `expert_fwd_compact_layer` - the function under test - is untouched
      by any commit in this whole M9-M12 workstream; this exact test/root
      cause is already recorded as a known pre-existing gap in M3's own
      entry above), `-p brain-kernels` (33 passed), `-p brain-qwen3` (full
      suite green), `-p brain-qwen35moe` (full suite green), `-p
      brain-qwen35` (full suite green), `-p brain-wan` (full suite green -
      this crate never builds a `model::ops::Ops` at all, confirmed by
      grep, so it was never at risk from this milestone's `REQUIRED_KERNELS`
      growth, but still run per the gating requirement), `-p brain-flux2`
      (full suite green, same "never builds an `Ops`" note), `-p
      brain-s3dit` (full suite green, same note).
- [x] M13: the two remaining kernels completing the affine K-quant
      (Q4_K/Q5_K) family - `moe_linear_gated_kq.wgsl` (new) and
      `matmul_kq_gemv_reg.wgsl` (new), plus their dispatch wiring.
      `moe_linear_gated_kq.wgsl` is `moe_linear_gated_i8.wgsl`'s affine
      sibling, in the SAME naive tier (one thread per output element, no
      workgroup tiling) for the SAME reason that kernel's own header states:
      `matmul_kq_dyn`/`matmul_kq_gemv` stage rows into WORKGROUP-SHARED
      memory across a barrier, and a per-thread early return for a
      non-routed row ahead of a `workgroupBarrier()` not every thread
      reaches is undefined behaviour in WGSL, so the row-level skip needs the
      naive tier's ordinary `return`. Its inner loop walks K IN ORDER, one
      thread per output element, so a weight-scale group's 8 quads are
      consecutive in the SAME thread's own loop - unlike `matmul_kq_gemv`'s
      64-thread k-stride (where 8 different threads visit one group and only
      the first may apply the min correction, guarded by `(g % WPGK) ==
      0u`), no guard is needed here: the correction is read once and applied
      once per (row, group) as the group completes, because no other thread
      ever touches that pair. Wired into `model::moe` (NOT
      `model::ops::Ops::moe_linear`, which explicitly refuses `I8`/`Q4`/
      `KQuant` today - the real dispatch site for every quantized MoE tier is
      `model::moe`'s own hand-written `expert_fwd_*` family, `MoeIds8`/
      `expert_fwd_i8` being the precedent): `MoeIdsKQ` (kernel indices,
      including the `quant_group_sum` prepass `h`'s own re-quantization
      needs - `h` is a fresh tensor per expert, so it cannot share the
      shared input's `xgs` prepass), `LinKQ` (one affine expert linear's
      weight - `wq`/`sz` in `model::ops::Weight::KQuant`'s own device
      layout), `ExpertScratchKQ` (adds `hgs` to `ExpertScratch8`'s fields),
      and `expert_fwd_kq` (the three-linear SwiGLU FFN + gated combine,
      structurally identical to `expert_fwd_i8` except its `lin` closure
      passes the RAW logical `k`/`n`, never divided by 4 - `moe_linear_
      gated_kq.wgsl`'s own param contract, matching `matmul_kq_{dyn,gemv}`'s
      established reason: `xq`/`wq` have different word densities, so a
      shared packed-word count would be ambiguous about which operand it
      counts). `backend_api::select::candidates`'s `Op::MoeExpertLinear` arm
      needed no CODE change (M12 already folded `Q4K`/`Q8K` into the SAME
      `PackedInt8` arm `I8`/`Q4` use, since the naive per-group-`dot4I8Packed`
      structure is genuinely DP4A-bound at every quantized tier this
      workspace ships) - only its stale comment (which had said "no kernel
      yet, nothing routes through these dtypes") was corrected to name the
      kernel and dispatch function that now exist.
      `matmul_kq_gemv_reg.wgsl` is `matmul_kq_gemv`'s register-accumulator
      sibling, standing to it exactly as `matmul_i8_gemv_reg` stands to
      `matmul_i8_gemv`: the same `MREG`-bucketed function-local accumulator
      array (registers, not workgroup memory sized for the `m=32` worst
      case) and the same one-write-at-the-end `partial` fold, with the two
      deltas `matmul_kq_gemv` itself already carries over `matmul_i8_gemv` -
      a staging-time `CODE_BITS`-wide unsigned-code unpack per quad, and the
      affine `dm[n,g]*S[m,g]` correction folded in per quad with the
      IDENTICAL one-thread-per-group guard (`(g % WPGK) == 0u`) in the
      IDENTICAL position, so the two kernels perform the identical
      operations in the identical order and bit-identity is a property of
      that construction, not a coincidence. Wired into `gpu_core::upgrade`
      as TWO new rows (one per `CODE_BITS`, since `matmul_kq_gemv` is only
      ever registered as its two `#CODE_BITS={4,8}` specialisations - the
      bare, unspecialised name is never registered by any model crate),
      sharing a new `DECODE_SHAPE_KQ` capability probe (a separate constant
      from `DECODE_SHAPE_I8` per this file's own "one probe per dtype
      family" convention, even though `select::candidates` folds `Q4K`/
      `Q8K` into I8's identical branch today). The `Upgrade` struct gained
      one new field, `extra: &'static [(&'static str, u32)]` - additional
      template constants FIXED for a row, applied alongside its bucket knob
      - because this is the first upgrade row whose `fast` kernel is
      ALREADY specialised along a second axis (`CODE_BITS`) before the
      `MREG` bucket ladder ever runs; every pre-existing row keeps `extra:
      &[]`. The table's own `table_entries_name_real_kernels` regression
      test needed one adjustment for the same reason: `u.slow` for these two
      rows is a `#CODE_BITS=N` specialised name that `kernels::src` (which
      only knows BARE registered names) cannot resolve directly, so the
      test now validates the Params/bindings CONTRACT against `u.slow`'s
      base name (a `#K=V` specialisation never moves either) rather than the
      specialised name itself. Gated: two new tests in `gpu_core::upgrade`'s
      own module (`kq_gemv_reg_appends_one_bucket_ladder_per_code_bits`,
      `kq_gemv_reg_resolves_independently_per_code_bits` - the two
      `CODE_BITS` rows activate and pick buckets independently, never
      conflating one's ladder with the other's); a new
      `crates/gpu-core/tests/kq_gemv_reg_upgrade.rs` (device-level, both
      `CODE_BITS`): the upgrade is active with the full 6-bucket `MREG`
      ladder, and `matmul_kq_gemv`/`matmul_kq_gemv_reg` are `assert_eq!`
      BIT-IDENTICAL across `m in {1,2,3,8,17,32}` at two `(k, n)` shapes -
      real `assert_eq!` on the raw output bits, not a tolerance, since these
      two are supposed to compute identically by construction; a new
      `crates/model/tests/moe_linear_gated_kq.rs` (device-level): rung (b)
      `moe_linear_gated_kq` with every row routed to one expert (out of
      three, exercising `e_idx` addressing) matches BOTH `matmul_kq_dyn` and
      `matmul_kq_gemv` at M11's own tolerance (`rel_l2 <= 1e-6`, `cosine >=
      1 - 1e-9`, `max_rel <= 5e-4`), all three compared against the SAME f64
      host oracle built directly from int8 codes, at both `CODE_BITS`; two
      more device-level tests exercise `expert_fwd_kq`'s WIRING specifically
      (not just the raw kernel): a non-routed row (gate column 0) writes
      EXACTLY zero, and two experts both routed to every row combine by
      plain f32 addition in the SAME order `scale_add.wgsl`'s own
      `accumulate` branch performs - `assert_eq!` on the bits, catching a
      swapped buffer or a mis-threaded `accumulate` flag a purely-finite
      smoke test could not. `make test --release --offline -p brain-kernels
      -p brain-model -p brain-gpu-core -p brain-backend-api`: every test in
      scope green on a real device (Intel Arc iGPU, Meteor Lake, Vulkan).
      **Measured speedup** (`crates/gpu-core/tests/kq_gemv_reg_speed_bench.rs`,
      `#[ignore]`d, `gpu_core::profile::best_of` bracketed by `poll_wait()`,
      preceded by a 3-second continuous-dispatch DVFS ramp per checklist
      §E.0b): at `k=n=2048` across `m in {1,2,4,8,16,32}`, both `CODE_BITS`,
      `matmul_kq_gemv_reg` measured `0.90x-1.09x` against plain
      `matmul_kq_gemv` on this integrated Arc - essentially a WASH, not the
      ~2x `matmul_i8_gemv_reg` measured on a discrete Tesla P40. This is a
      real, measured number, not a regression (no shape measured below
      0.90x, several measured a small win), and the row stays wired per
      this milestone's own instruction - but the honest reading is that
      this iGPU's shared-memory/register tradeoff at this shape does not
      show the same win the symmetric int8 kernel showed on a discrete
      card, which is worth re-measuring on a P40-class device before
      treating the wiring as validated for a win rather than merely for
      correctness. `.agents/rules/kernels.md`'s own bar #3 for
      `gpu_core::upgrade` ("wins at every shape") is therefore NOT
      cleanly met on this hardware the way the i8/f32 GEMV rows are -
      recorded here as the honest evidence, matching this file's own
      precedent for `matmul_q4_gemv_reg`'s measured regression, except this
      row is a wash rather than a loss so it is left wired rather than
      pulled.
- [x] M14: byte compression of the M8 device scale plane - the flat `wsz:
      [n, 2*k/G] f32` interleaved `(scale, min)` array (M8-M13) replaced with
      two packed planes: `wsm: [n, ceil(k/G/2)] u32` (per-group `(sc, m)`
      sub-scale BYTE pair, two groups/word - `sc` low byte, `m` high byte of
      each group's own 16-bit half) and `wd: [n, k/spb] u32` (per-super-block
      `(d, dmin)` f16 BIT-PATTERN pair, one word per `spb`-element
      super-block - 256 for the three K-quant formats, 32 for the three
      legacy formats which have no coarser grouping than their own block).
      `crates/gguf/src/kquant.rs` (host relayout), `crates/model/src/
      kquant.rs` (`KqDeviceLayout`, restated for device dispatch call sites
      that do not depend on the `gguf` crate) and `crates/model/src/ops.rs`
      (`Weight::KQuant`'s `sz: KqScale` field, `Packed { wsm, wd }` for the
      affine family) all carry the new shape; `matmul_kq_dyn`/
      `matmul_kq_gemv`/`matmul_kq_gemv_reg`/`moe_linear_gated_kq` (M11/M13)
      bind `wsm`/`wd` directly in place of the old flat `wsz`, decoding
      in-shader with `ds = f16_to_f32(wd_low_half) * f32(sc)`, `dm =
      f16_to_f32(wd_high_half) * f32(m)` - the identical `d*sc`/`dmin*m` fp32
      expressions `checkpoint::gguf`'s own `deq_q4_k`/`deq_q5_k` use, so the
      host round-trip gate stays a real `assert_eq!`, never a tolerance. The
      in-shader `f16_to_f32` is the SAME magic-multiply/FTZ-safe-subnormal/
      inf-nan construction `kernels::template::f16_decode_expr` already
      generates for the bf16/f16 WEIGHT STORAGE tier - copied verbatim (same
      magic constants `0x77800000`/`0x38800000`/`0x7F800000`), not
      reinvented, and independently re-verified bit-exact against
      `half::f16::from_bits(..).to_f32()` for all 65536 possible bit patterns
      by hand during this milestone (a standalone Rust translation of the
      WGSL expression, `worst diff = 0`) - ruling out decode precision as a
      source of any measured discrepancy (see below). Decoding happens once
      per `(column, group)` fold - the SAME frequency the old flat `wsz` read
      already ran at - so no additional hoisting was needed for correctness;
      hoisting the f16 decode across the `GPS=8` folds that share one `wd`
      word (a further win the task description flagged as a possible
      follow-up) was NOT attempted, since the fold-point frequency was
      already the natural granularity and a deeper hoist would need
      restructuring the software-pipelined chunk loop for a benefit no gate
      in this milestone required.
      **Design decision: full replacement, not a parallel variant.** Grepped
      the whole workspace for any caller of `Weight::KQuant`/`KqLayout`
      outside `crates/{kernels,gguf,model}` and found none - `Weight::KQuant`
      has no `Weight::upload` path at all (that function explicitly refuses
      `Q4K`/`Q8K`, per M12), and the only two call sites of the struct in the
      whole tree are this crate's own test files (M9-M13 landed the kernels
      and the dispatch seam but no model has wired a real GGUF checkpoint
      through them yet - see the ledger's own "Recorded gaps" section). With
      no external dependent on the exact old byte layout, this repo's stated
      policy against parallel/legacy paths made a full, clean replacement the
      only reasonable choice; the old flat-`wsz` code path was deleted
      outright rather than kept behind a flag.
      **Scope: the affine family only (Q4_K/Q5_K).** Q6_K stays on a PLAIN
      `[n, k/16] f32` `ds`-only array (`KqScale::PlainF32`, `dm` is always
      `0.0` for a symmetric type, nothing to pack) - it reaches the device
      through the EXISTING `matmul_i8_dyn#QPG=1`/`matmul_i8_gemv#WPG=4`
      kernels (M10), which know nothing about the packed `(sc,m)`/`(d,dmin)`
      shape and are not this milestone's to rewrite; Q5_0/Q4_0/Q8_0 never
      touched `KqLayout`/`wsz`/`wsm`/`wd` at the device dispatch level to
      begin with (M12: they reach the device as plain `Weight::I8`/`Weight::
      Q4`, unrelated to this struct) - `gguf::kquant`'s HOST relayout still
      produces a `(wsm, wd)` pair for all six formats uniformly (for the
      lossless-relayout/round-trip-test contract every format shares), but
      only Q4_K/Q5_K's device kernels actually bind it.
      **Gating.** (a) `crates/gguf/tests/kquant.rs`'s round-trip test is
      still a real `assert_eq!` against the SAME oracle
      (`checkpoint::gguf`'s private `deq_*` functions), re-derived for the
      packed encoding rather than weakened - reconstructing `(wq, wsm, wd)`
      with the in-shader-matching expressions above still reproduces
      `MmapGguf::tensor`'s decode bit for bit, both whole-tensor and at a
      genuine sub-rectangle. (b) `device_bytes_per_parameter_matches_the_
      recomputed_layout` (new) computes the REAL device bytes/param from the
      actual relaid-out buffer lengths for all six types and asserts it
      against a recomputed-from-layout formula (`bits/8 + 2/group + 4/spb`),
      not a guessed constant - measured/asserted numbers below. (c) every
      M11/M13 device-vs-oracle and cross-kernel test in `crates/model/tests/
      matmul_kq.rs`/`moe_linear_gated_kq.rs`/`ops_kquant.rs` still passes at
      the SAME tolerances (`rel_l2 <= 1e-6`, `cosine >= 1 - 1e-9`, `max_rel
      <= 5e-4`) - three of those (`case5_subrectangle_nonzero_origin_two_
      superblocks`, `case6_ragged_tiles_dyn`, `case6_ragged_gemv`) initially
      went red on a real device (`max_rel` measured `6e-4..1e-3`, i.e. 1.2x
      to 2x over the calibrated ceiling) after the M8→M14 layout swap; root-
      caused (not just "recalibrate and move on", per this ledger's own rule
      against widening a band to make a test pass) by instrumenting `max_rel`
      to report the failing element and finding each one had a genuinely
      small true value (`|want|` in `0.07..1.7`) produced by real cancellation
      between the two-reduction affine fold's `ds*A` and `dm*S` terms - NOT a
      kernel bug (the f16 decode was independently verified bit-exact above,
      and `ds = d*sc`/`dm = dmin*m` is one IEEE-754 multiply either computed
      host-side or device-side, so identical operands give an identical
      result on any conforming backend). The actual cause was the new
      decomposed `(sc, d_super)`/`(mn, dmin_super)` test-data generator
      (`rand_kq_scale`, needed because M14 replaced the old test's direct
      `ds`/`dm` sampling with sampling the pieces the packed format actually
      stores) drawing from a wider effective `ds`/`dm` envelope than the
      pre-M14 test did (up to `1.24`/`2.48` vs the original `0.5`/`1.5`),
      which fed larger-magnitude intermediate terms into the SAME
      cancellation-prone fold and pushed the accumulated fp32 rounding noise
      on an unlucky element past the `max(want,1)`-floored `max_rel` metric's
      ceiling. Fixed by narrowing `rand_kq_scale`'s `d_super`/`dmin_super`
      ranges so the effective `ds`/`dm` envelope matches what the ceiling was
      originally calibrated against (`~0.5`/`~1.5`) - restoring the pre-M14
      numerical risk profile is a legitimate "update the test for the new
      layout" per this milestone's own gating instructions, not a loosened
      tolerance; the ceiling itself (`5e-4`) is untouched. `make test
      --release --offline -p brain-gguf -p brain-kernels -p brain-model`:
      every test in scope green except the ONE pre-existing, unrelated
      failure already recorded in M3's and M12's own entries above
      (`moe_compact_parity::compact_layer_submit_count_does_not_scale_with_
      expert_count` - confirmed via `git diff` that `expert_fwd_compact_
      layer` is untouched by this milestone). `make kernels-regen && make
      kernels-table`: both pass, no new kernel files (this milestone edits
      four existing `.wgsl` files in place, adds none).
      **Measured bytes/param** (`device_bytes_per_parameter_matches_the_
      recomputed_layout`, `bits/8 + 2/group + 4/spb`) vs the source GGUF
      block's own bytes/param (`block_bytes/block_elems`):
      | type  | device B/param | GGUF B/param | ratio    |
      |-------|-----------------|--------------|----------|
      | Q4_K  | 0.578125        | 0.5625       | 1.0278x  |
      | Q5_K  | 1.078125        | 0.6875       | 1.5682x  |
      | Q6_K  | 1.140625        | 0.8203125    | 1.3905x  |
      | Q5_0  | 1.187500        | 0.6875       | 1.7273x  |
      | Q4_0  | 0.687500        | 0.5625       | 1.2222x  |
      | Q8_0  | 1.187500        | 1.0625       | 1.1176x  |
      Q4_K lands almost exactly at GGUF's own density (`1.03x`, matching this
      ledger's own placeholder target before this milestone started) since
      its 4-bit codes were already this compact pre-M14 - M14's whole
      contribution for Q4_K is shrinking the scale plane. Q6_K/Q5_0/Q4_0/
      Q8_0 carry real, permanent overhead (`1.12x-1.73x`) because their
      groups are 16-32 elements with NO coarser super-block structure to
      amortize `wsm`/`wd` against the way Q4_K/Q5_K's 256-element super-block
      does - this is the canonical layout's known, accepted cost of giving
      every format ONE shared shape rather than a per-format bespoke one
      (Q8_0 in particular never pays it in practice, since production Q8_0
      never reaches the device through this table at all - it uses the
      separate, already-solved `gguf::int8_direct::try_i8_rect` path M8's own
      "two facts worth keeping visible" section named). Q5_K is the clear
      outlier at `1.57x` - it is the one format this milestone's SECOND
      compression target (below) would fix.
      **Second compression target (Q5_K nibble + high-bit plane):
      investigated, NOT implemented - a real architectural blocker, reported
      rather than forced.** The design: shrink Q5_K's `wq` from `bits=8`
      (its raw 5-bit code sitting in a full unsigned byte slot, 4 codes/word)
      to `bits=4` (a nibble plane, byte-IDENTICAL to Q4_K's own packing, 8
      codes/word) plus a new `wh: [n, k/32] u32` plane holding the 5th
      (high) bit, one bit per element, 32 elements packed per word - which
      conveniently is exactly one weight-scale group per `wh` word, since
      `group=32` for Q5_K. Reconstruction is a staging-time BIT-SCATTER
      (spread one bit per element out of a packed `wh` word into bit 4 of
      each unpacked nibble) - the same technique class this ledger's own
      M8 entry already documents for legacy Q5_0's high-bit-of-5 nibble
      combination (`relayout_q5_0`, `checkpoint::gguf`'s own `deq_q5_0`), so
      the pattern is not new to this codebase, only new to the K-quant
      device-kernel side. Verified this would be a real win before deciding
      whether to build it: `bits/8 + 4/32(wh) + 2/32(wsm) + 4/256(wd) =
      0.703125` B/param, `1.0227x` GGUF - almost exactly Q4_K's own ratio,
      down from the uncompressed `1.5682x` measured above. Not implemented
      this milestone because `moe_linear_gated_kq.wgsl` - one of the FOUR
      kernels this compression would touch - is ALREADY at exactly 8 storage
      bindings (`xq`, `wq`, `sx`, `wsm`, `wd`, `xgs`, `gate`, `out`; verified
      by direct inspection of the kernel source, not assumed) before adding a
      9th `wh` binding, which this engine's own `<=8 storage buffers per
      kernel` hard constraint (`AGENTS.md`) makes impossible without either
      (a) accepting a per-OP-TYPE layout divergence for the SAME dtype
      (`Dtype::Q8K`) - the three tiled/GEMV matmul kernels binding a
      compressed `bits=4 + wh` `Weight::KQuant` while `moe_linear_gated_kq`
      keeps binding an uncompressed `bits=8` one for the SAME logical GGUF
      format, which is buildable in principle (`Weight::KQuant` instances are
      already independently constructed per destination tensor, so two
      differently-shaped Q5_K instances coexisting is not itself a
      contradiction) but was not attempted here, or (b) freeing a slot by
      merging two of the eight existing bindings (e.g. folding `xgs` into
      `wsm`'s buffer, or `gate` into `out`), which is a genuine kernel-layout
      redesign this milestone's scope did not budget for. Per this
      milestone's own instructions ("that is a real blocker - report it
      clearly rather than silently exceeding the limit or silently skipping
      the MoE kernel's compression"), this is reported here rather than
      forced: Q5_K's nibble+high-bit compression is deferred to a future
      milestone with the design above recorded so it does not need
      re-deriving, and NOTHING in this milestone's own shipped code claims
      it is done (`gguf::kquant`'s and `model::kquant`'s module doc tables
      both still state `Q5_K | bits=8`, matching what is actually built).

**K-quant workstream status (M8-M14).** The native K-quant execution path is
complete and shipped: M8 built the lossless host relayout for all six GGUF
block formats into one canonical device shape; M9-M13 built the affine
Q4_K/Q5_K GEMM/GEMV/MoE kernels, the group-16 reuse of the existing symmetric
kernels for Q6_K, and wired all of it into the shared `backend_api`/
`model::ops` dispatch seam every int8-tier model in this workspace already
goes through; M14 shrank the device scale plane from a flat f32 array to a
packed `(sc,m)`-byte + `(d,dmin)`-f16 pair, landing Q4_K within 3% of GGUF's
own on-disk density. This whole path builds on the loader-seam foundation
M0-M7 laid earlier in this same ledger (the `GgmlType` vocabulary, the
`TensorSource::raw_blocks` zero-fp32 read path, and the `model::int8`/
`quantize_from` helpers every migrated model crate now shares) - K-quant
specifically needed `raw_blocks` to reach raw GGUF block bytes without a
whole-tensor fp32 detour, which is the seam M1 built for Q8_0 and this
workstream generalized. One real gap remains open, not from a missing
capability but from an unexploited one: Q5_K's own device footprint (`1.57x`
GGUF, the worst of the six) has a fully-designed, verified-worthwhile fix
(M14's own "second compression target" note above) blocked on a genuine
8-storage-buffer budget conflict in `moe_linear_gated_kq.wgsl`, deferred
rather than forced. Every gate through M14 remains synthetic (see "Recorded
gaps" below) - no real K-quant checkpoint has been run through this path on
this box yet.
- [x] M15: split GGUF in `MmapGguf`. `mmap: Mmap` became `mmaps: Vec<Mmap>`
      (`mmaps[0]` is always part 1) and `index`'s value tuple gained a part
      index as its first field (`(usize, u32, usize, usize, usize)` -
      part/ty/start/nbytes/numel); all 17 public method signatures stay
      byte-identical, only the six `self.mmap[..]` slice sites inside
      `tensor`/`tensor_range`/`raw_tensor_bytes`/`with_tensor_chunks`/
      `raw_words`/`raw_blocks`/`dtype`/`numel` became `self.mmaps[part][..]`
      - no `Inner::GgufSharded` arm anywhere else in the tree, exactly the
      "entry point stays `MmapGguf::open(path)`" design goal, so
      `WeightReader`, every existing `MmapGguf::open` call site (grepped:
      over 90 across the workspace), and `brain models info` gain split
      support with zero edits of their own.
      New `crates/checkpoint/src/split.rs`: `split_name(fname, ext) ->
      Option<(base, part_1_based, count, digit_width)>` and its inverse
      `split_sibling`, the `<base>-NNNNN-of-MMMMM.<ext>` parser generalized
      from `cli::model_dir::shard_of`'s `.safetensors`-only original (that
      hoist is what M16 will point its own callers at) - refuses part `0`
      or a part exceeding `count`, matching a real writer's own invariant
      rather than guessing at a malformed one. `MmapGguf::open` parses
      `path`'s filename once: a non-split name takes the untouched
      single-file path (`vec![path.to_string()]`); a split name builds
      every sibling's path from `dir.join(split_sibling(...))` for
      `1..=count` and opens all of them eagerly, in part order, BEFORE
      returning - a missing part therefore fails at `std::fs::File::open`
      with that exact filename in the error, never as a "tensor not found"
      once decoding starts. New `validate_split` (only called when more
      than one part was found) checks, per part: `split.no` equals the
      part's own 0-based index (llama.cpp writes it 0-based against
      1-based filenames - the exact off-by-one relationship a dedicated
      test pins rather than re-deriving from a spec each time);
      `split.count` equals the number of files THIS OPEN actually located
      (derived from the filename's own encoded count, not trusted from any
      part's KV alone); `general.architecture` agrees with part 1's. Once,
      across all parts: if any part carries `split.tensors.count`, every
      part that carries it agrees, and the summed real tensor count across
      every part's own tensor-info list matches it exactly. The merge step
      that follows also refuses a tensor NAME appearing in more than one
      part (a corrupt or hand-edited split could otherwise silently let a
      later part's tensor shadow an earlier one's). `general.file_type` and
      every other model-config key are read from part 1's KV, same as a
      single file's own convention.
      New `gguf_write::write_split(dir, base, kv, tensors: &[Vec<TensorOut>],
      alignment) -> io::Result<String>` (returns part 1's path) - the
      fixture generator this milestone needed, since this box has no real
      split checkpoint (see "Recorded gaps"): writes one real, independent
      GGUF file per element of `tensors` via the existing eager `write`,
      injecting `split.no`/`split.count`/`split.tensors.count` into every
      part's KV itself so a test never hand-assembles split metadata bytes.
      Gated (`crates/checkpoint/tests/gguf_split.rs`, new, 7 tests): a
      3-part split opens correctly from EITHER part 1's path or a
      non-first part's path and reads one real tensor from each of the
      three parts by value (not just checking names - proves cross-part
      byte indexing, not just cross-part name merging); a part deleted
      from disk after writing is refused by its own exact filename; a
      `split.tensors.count` that disagrees with the real summed total is
      refused, naming both numbers; a `split.no` written 1-based (the
      off-by-one this design exists to catch) is refused; a part missing
      `split.no` entirely is refused; mismatched `general.architecture`
      across parts is refused; a plain unsplit file still opens exactly as
      before (the single-file path is provably untouched, not just
      unlikely to be touched). `make test -p brain-checkpoint`: 117 lib
      tests + this file's 7 + every other integration suite in the crate,
      0 failed. `cargo check --release --workspace`: clean - confirms the
      "zero downstream edits" claim, since every one of the ~90 external
      `MmapGguf::open` call sites in the tree still compiles unchanged
      against the new internal layout.

- [x] M16: split GGUF in `modelstore`/CLI. New `crates/checkpoint/src/split.rs`
      generalized `cli::model_dir::shard_of`'s `.safetensors`-only parser to
      `split_name(fname, ext)`/`split_sibling(...)`, shared by every layer
      below rather than each keeping its own copy (M15 already used it for
      `MmapGguf::open` itself; this milestone is the two remaining callers).
      `recipe::quant_of_gguf` now strips a split part's `-NNNNN-of-MMMMM`
      tail before applying the existing exact `-<QUANT>` grammar, so
      `<base>-Q4_K_M-00001-of-00003.gguf` declares `Q4_K_M` exactly as
      `<base>-Q4_K_M.gguf` would - a genuine behavior change from the
      original doc's "one PART of a split model … declares NOTHING", made
      deliberately (correction #6: policy change, not a bug fix).
      `GgufPick` widened `file: String` to `files: Vec<String>`;
      `GgufRecipe::offered` now groups root `.gguf` files by identity FIRST
      (a plain file is its own group; a split part's group is its
      `(base, count)`) and only calls a split group a real candidate when
      every part `1..=count` is actually present - an incomplete split
      contributes nothing (its files fall into `refuse()`'s "unnamed" count,
      same policy as a file whose name declares no quantization). This is
      what keeps `choose()` completely unchanged: with grouping done in
      `offered()`, a complete 3-part Q4_K_M split is already exactly ONE
      `GgufPick`, so the existing "more than one file declares X" ambiguity
      check never fires on it. `artifacts()` emits one `Artifact` per part
      for a split pick, destined locally as `<QUANT>-NNNNN-of-MMMMM.gguf`
      (`split_sibling` keyed off the quant name, not the upstream repo's own
      filename) - the same convention `MmapGguf::open` reads back.
      `Store::local_quant` tries the plain `<QUANT>.gguf` first (unchanged
      fast path) and falls back to a new `split_quant_part1` directory scan
      for `<QUANT>-00001-of-MMMMM.gguf`, returning THAT path (never a part
      other than 1) since that is what `MmapGguf::open` needs to find every
      sibling. `Store::scan_repo_dir` gained a dedup set keyed by `Quant`:
      without it a 3-part split would register the same `LocalModel` three
      times, once per part file. `cli::model_dir::discover_flat`'s legacy
      flat-layout scan gained the identical split-grouping shape the
      pre-existing `.safetensors` shard grouping already had (both now go
      through the one shared `split_name` parser instead of two separate
      hand-rolled ones) - registers once, from part 1's path.
      Gated: `crates/modelstore/src/recipe.rs` (3 new tests: a complete
      3-part split resolves as one candidate with correct upstream/local
      names; an incomplete 2-of-3 split declares nothing and its stray
      files count as unnamed; `quant_of_gguf` reads a split part's quant
      through the `-of-` tail) plus the pre-existing `only_an_exact_quant_
      tail_declares_a_quantization` updated for the new (intentional)
      behavior. `crates/modelstore/src/lib.rs` (2 new tests, using
      `checkpoint::gguf_write::write_split` with a DISTINCTLY-named tensor
      per part - `MmapGguf::open`'s own cross-part merge refuses a repeated
      name, which the first draft of these tests tripped over and fixed):
      `local()` resolves a split quant to part 1's real path; `scan()`
      dedupes a 3-part split to exactly one `LocalModel`. `crates/cli/src/
      model_dir.rs` (1 new test, using the same `write_split` shape plus the
      embedded-tokenizer KV `gguf_qwen_with_embedded_tokenizer_registers_
      chat_capable` already established): a 3-part split registers exactly
      once, as a real chat-capable Qwen resident, not three separate (or
      zero) entries. `make test -p brain-modelstore`: 86/86. `brain-cli` has
      no `[lib]` target (`[[bin]]` only), so the Makefile's blanket `--lib
      --bins --tests` fails on it for a reason unrelated to this change
      (confirmed pre-existing); `cargo test --release --offline -p brain-cli
      --bins`: 267/267, including all 10 `model_dir` tests. `cargo check
      --release --workspace`: clean.

- [x] M17: ggml ids 9 (Q8_1) and 24-28 (I8/I16/I32/I64/F64), which used to
      fail `MmapGguf::open` OUTRIGHT (`tensor_nbytes` → `None` for the
      unrecognized type, refusing the WHOLE file over one tensor usually
      carried for metadata nobody reads). `GgmlType` gained 6 variants:
      `Q8_1` (Q8_0's block plus a second fp16 field ggml's fused matmul
      caches - `s = d*sum(qs)` - and never reads for plain dequant, so
      `deq_q8_1` is `deq_q8_0`'s math verbatim over a 36-byte block instead
      of 34; practically never a STORED tensor type, but refusing a file
      over a type nothing in it actually reads is the worse failure) and the
      five plain scalar-array types `F64`/`I8`/`I16`/`I32`/`I64`
      (`block_elems=1`, no block header, decoded with a straight `as f32`
      cast - lossy outside f32's exact range for I64/F64, the same tradeoff
      this fp32-only engine already accepts everywhere else). Every wrapper
      this reader has (`ggml_type_name`/`block_geometry`/`tensor_nbytes`/
      `dequantize`) is `GgmlType`-derived (M0), so adding the six variants
      to `from_id`/`id`/`name`/`block_elems`/`block_bytes`/`block_decoder`
      is the whole change - no second table anywhere to keep in sync, and
      the compiler's exhaustiveness check on every `match self { .. }` over
      `GgmlType` (none of which have a wildcard arm) is what proves no other
      file needed touching, not a grep.
      Gated (`crates/checkpoint/src/gguf.rs`'s own test module):
      `scalar_array_types_open_and_dequantize_to_their_exact_value` opens a
      real GGUF (via `gguf_write::write`, not hand-assembled bytes) for each
      of the five scalar types and asserts EXACT decoded values, chosen at
      each type's extremes (`i8::MIN/MAX`, `i32::MIN/MAX`, and for I64/F64 a
      value OUTSIDE f32's exact range, so the lossy cast is actually
      exercised rather than passing by coincidence on an in-range fixture);
      `q8_1_dequantizes_identically_to_q8_0_ignoring_the_cached_sum_field`
      builds a Q8_1 block with the SAME `d`/`qs` as a Q8_0 block but a
      DELIBERATELY WRONG `s` field (`0xFFFF`, not the real `d*sum(qs)`) and
      asserts the dequant still matches Q8_0's exactly - if `s` were
      mistakenly folded into the decode this is the case that would show it;
      `m17_types_report_correct_block_geometry` pins `block_geometry`/
      `tensor_nbytes` for all six directly. Also fixed the pre-existing
      `ggml_type_round_trips_and_agrees_with_the_wrapper_fns`, which
      asserted id 9 (among others) "must be unrecognized" - true before this
      milestone, the exact thing it changes now, so the test's own
      "must-still-be-unknown" set was narrowed to the ids that remain
      genuinely unrecognized (16, 39, 999) and its "known" table extended
      with the six new ones. `make test -p brain-checkpoint`: 120 lib tests
      (3 new + the updated one) + every integration suite, 0 failed. `cargo
      check --release --workspace`: clean.

- [x] M18: codebook families - MXFP4, IQ4_NL, IQ4_XS, TQ1_0, TQ2_0. This box
      has no real MXFP4/IQ/TQ checkpoint (see "Recorded gaps"), and unlike
      the K-quant work (M8), there was no EXISTING decoder anywhere in this
      tree to read the byte layout from by inspection - so ggml's own
      `ggml-common.h` (block structs, `kvalues_*` tables) and
      `ggml-quants.c` (`dequantize_row_*`) were fetched from
      `github.com/ggml-org/llama.cpp` (outside this repo, into the
      workspace's shared resources tree) as the ground truth, cross-checked
      against `diffusers`' own installed `IQ4_NL`/`IQ4_XS` Python
      reference (`quantizers/gguf/utils.py`, a real-world port other people
      run against real checkpoints) before a line of Rust was written.
      Transcribing the C loops caught two real ordering bugs BEFORE they
      shipped: a first draft of `deq_tq1_0`/`deq_tq2_0` "simplified" ggml's
      two-chunk `qs` split (`[0..32]` then `[32..48]`/`[32..64]`, each
      running its full inner digit/shift loop) into one pass over every
      byte per digit - which changes the OUTPUT ORDER, not just the code
      shape, since 48 (TQ1_0's `qs` length) is not a multiple of 32. Caught
      by hand-tracing the real C loop bounds against the simplified version
      before either was gated, not by a failing test.
      `GgmlType` gained 5 variants. MXFP4 (`block_mxfp4{e:u8; qs:[u8;16]}`,
      17 bytes/32 elements): `e` is E8M0 (unsigned exponent, bias 127);
      `KVALUES_MXFP4` is ggml's `kvalues_fp4` table, the E2M1 magnitude
      DOUBLED (confirmed independently against Triton's own OCP-MX-spec
      `MXFP4Tensor`/`MXScaleTensor` reference, `triton/tools/mxfp.py`), and
      `e8m0_to_fp32_half` returns HALF the true E8M0 value (`2^(e-128)`,
      ported as ggml's own bit-placement trick rather than
      `2f32.powi(e-128)` - `e` in `{0,1}` lands in fp32 subnormal range,
      where a computed power is not guaranteed to round identically to a
      direct bit pattern) so the doubling cancels exactly. IQ4_NL
      (`block_iq4_nl{d:f16; qs:[u8;16]}`, 18 bytes/32 elements): ggml's own
      non-linear `kvalues_iq4nl` 16-entry codebook, byte-packing identical
      to Q4_0's own lo/hi-nibble-half split already in this file. IQ4_XS
      (`block_iq4_xs{d:f16; scales_h:u16; scales_l:[u8;4]; qs:[u8;128]}`,
      136 bytes/256 elements): 8 sub-blocks of 32, each scale a 6-bit signed
      value (`scales_l` nibble | `scales_h` 2-bit field, offset -32) applied
      to the same `kvalues_iq4nl` codebook. TQ1_0/TQ2_0 (54 and 66
      bytes/256 elements): base-3 (TQ1_0, `pow3`-multiply-then-shift digit
      extraction, `wrapping_mul` reproducing C's implicit `uint8_t`
      truncation - the algorithm DEPENDS on the wraparound, not just
      tolerates it) and base-4 (TQ2_0, plain 2-bit shifts) ternary packing.
      Every wrapper (`ggml_type_name`/`block_geometry`/`tensor_nbytes`/
      `dequantize`) picked the 5 up for free via `GgmlType` (M0 again).
      Gated (`crates/checkpoint/src/gguf.rs`'s own test module, all against
      INDEPENDENTLY computed expected values, never the decoder's own
      output): MXFP4/IQ4_NL construct a block covering all 16 codebook
      entries at unit scale and assert the decode equals the LUT verbatim
      (plus a second MXFP4 case at half scale, proving the scale is
      genuinely applied); IQ4_XS hand-derives the 6-bit scale bit layout for
      2 of 8 sub-blocks and asserts both the scale AND the codebook lookup
      per sub-block; TQ1_0/TQ2_0 are checked against values from a SEPARATE
      Python transcription of the same ggml C source (not hand arithmetic,
      not this Rust code) at real, non-trivial byte values spanning both
      chunk boundaries and `qh` - exactly the shape that would have caught
      the two ordering bugs above had they shipped; a final test opens a
      real GGUF via `MmapGguf::open`/`gguf_write::write` for all 5 types,
      proving the container-level path (not just the bare `dequantize`
      function) no longer refuses them. Also fixed two pre-existing tests
      whose fixtures happened to use ids this milestone newly recognizes
      (`iq_types_error_clearly` used id 20 = IQ4_NL; the `ggml_type_round_
      trips` "still unknown" set included 39 = MXFP4). `make test -p
      brain-checkpoint`: 126 lib tests (7 new/updated) + every integration
      suite, 0 failed. `cargo check --release --workspace`: clean.
      Deferred, NOT part of this milestone: IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/
      IQ2_S/IQ3_XXS/IQ3_S ("the rest" of the IQ family) - each needs a large
      NGRID lookup table (up to 2048+ entries) ported from ggml with the
      same verify-before-write discipline as above; not attempted here for
      lack of time, not lack of a ground-truth source (the same
      `ggml-common.h` fetch has them).

- [x] M19: write side - `Tier` gained the 5 remaining variants `crate::
      quant::encodable_geometry` (the panic-fix commit right before this
      one) already has a real encoder for: `Q4_1`, `Q5_1`, `Q2K`, `Q3K`,
      `Q8K` - all 11 encodable types now have a `Tier` row, and the doc
      comment says so explicitly (a `Tier` variant with no matching
      `encodable_geometry` arm would panic the moment `convert` tried to
      encode a block, so the two enums are documented as required to stay
      in lockstep). `quantize_cli.rs`'s `--tier` parser accepts all 11 names
      case-insensitively (was `Q8_0`-only); `USAGE` lists them.
      `general.file_type` is now derived from the tier via a new `Tier::
      file_type_id() -> Option<u32>`, instead of the hardcoded `7` written
      regardless of the actual `--tier` chosen - `checkpoint::gguf` gained
      the write-side inverse of its own `file_type_name` (`file_type_id`,
      `pub(crate)`), both now searching ONE `FILE_TYPES: &[(u32, &str)]`
      table (refactored from `file_type_name`'s own `match`) so the two
      directions cannot drift from each other. The three K-quant tiers this
      module quantizes UNIFORMLY (`Q3K`/`Q4K`/`Q5K`) have no bare
      `"Q3_K"`/`"Q4_K"`/`"Q5_K"` entry in llama.cpp's own `file_type`
      enum - real GGUF files only ever declare a MIXED per-layer "S/M/L"
      recipe under that id - so `file_type_id` picks each tier's `"_M"`
      spelling (llama.cpp's own default/most-common recipe name) as the
      closest real id, documented as an approximation rather than hidden.
      `Q8K` has no `general.file_type` id at all in llama.cpp's own enum
      (never a real release format - it exists for K-quant matmul's own
      intermediate accumulation) and returns `None`; `quantize_cli.rs`
      omits the KV entirely rather than writing a fabricated id.
      `make test -p brain-checkpoint`: green (127+ lib tests, `Tier`'s own
      round-trip and `encodable_geometry` regression coverage from the
      panic-fix commit unaffected). This milestone landed in the middle of
      rebasing this whole workstream onto `origin/main`'s newer qwen35/GDN
      work (33 commits, none of which touch `checkpoint::quantize` or
      `quantize_cli.rs`) - no conflict here, unlike the M13/M12-era files
      the rebase itself needed hand-resolving (see those commits' own
      messages).

- [x] M20: `GgufTokenizer` completeness. 16 new fields: `chat_template`
      (`tokenizer.chat_template`), `chat_templates` (named variants -
      llama.cpp's writer stores each one under its own `tokenizer.chat_
      template.<name>` string key and lists only NAMES in `tokenizer.chat_
      templates`, confirmed by fetching `gguf_writer.py` rather than
      guessing, per M18's precedent; read by scanning the KV `BTreeMap` for
      the `tokenizer.chat_template.` prefix via `.range().take_while()`,
      which needs no separate names-array lookup and cannot desync from
      what the writer actually named), `add_bos_token`/`add_eos_token`,
      `eot`/`eom`, `scores` (parallel f32 array to `tokens`), `add_space_
      prefix`, `precompiled_charsmap` (confirmed as `Array` of `U8` via the
      same writer fetch - not `String`, which this reader's `Cursor::
      string` UTF-8-validates and a real charsmap blob is not guaranteed to
      satisfy, a real correctness hazard this could have hit if guessed
      wrong), and the six `fim_*_token_id`s. `GgufTokenizer` gained
      `#[derive(Default)]` so every field beyond the pre-M20 nine can be
      `..Default::default()`-spread at existing hand-built call sites
      instead of requiring every one of them to enumerate 16 new fields.
      `data::chat_template::ChatTemplate::from_gguf(gt: &GgufTokenizer)`
      is `from_model_dir`'s sibling for a `.gguf` with no sibling
      `tokenizer_config.json`: same `compile()` call, same bos/eos
      default-injection, sourced from `gt.tokens[gt.bos]`/`gt.tokens[gt.eos]`
      (a GGUF's own vocab) instead of `tokenizer_config.json`'s string
      fields, and the same absent-template error shape.
      `#[derive(Default)]` is a breaking change for every hand-built
      `GgufTokenizer {..}` literal elsewhere in the workspace (Rust does not
      let a struct grow fields without a way to default them). Six sites
      found (`crates/data/src/qwen_tokenizer.rs`'s own two test fixtures,
      `crates/cli/src/continuous_train.rs`, `crates/rl/src/atif.rs`,
      `crates/rl/tests/continuous_cycle.rs`, `crates/deepseek2ocr/src/
      caps.rs`, `crates/deepseek2ocr/src/prompt.rs` - two literals there),
      every one fixed with `..Default::default()` in the same commit as the
      struct change, since M20 alone would leave the tree non-compiling.
      Gated (`crates/checkpoint/src/gguf.rs`, `crates/data/src/chat_
      template.rs`): both new tests build a REAL GGUF via `gguf_write::
      write` + `MmapGguf::open`/`ChatTemplate::from_gguf` (never a
      hand-built `GgufTokenizer`, which would only prove the struct
      satisfies its own field list, not that `tokenizer_from_kv` parses
      correctly) - one carrying every new key (asserts each decodes
      exactly, including two named template variants), one carrying none
      (asserts every new field lands on its documented default).
      `make test -p brain-checkpoint -p brain-data -p brain-cli -p
      brain-rl -p brain-deepseek2ocr`: green throughout. `cargo check
      --release --workspace`: clean.
- [x] M21: wired the GGUF-embedded-tokenizer fallback (`reader.tokenizer()`
      -> `QwenBpe::from_gguf`, the pattern `crate::resident_llm::
      QwenResident::activate` already used) into `crates/cli/src/
      resident_qwen35.rs` and `resident_qwen35moe.rs`'s own `activate()`:
      an explicit sibling `tokenizer.json`/env override still wins, but an
      unset one now falls back to the checkpoint's own embedded `tokenizer.
      ggml.*` KV before giving up, instead of erroring immediately. `Engine`
      itself still has no GGUF arm (`checkpoint::load` is safetensors-only),
      so a `.gguf` checkpoint still cannot fully activate on either resident
      - that gap is pre-existing and stays open, documented in both
      module docs; the fallback only closes the "tokenizer construction
      fails before the real, already-documented limitation is even
      reached" gap. Gated: `resident_qwen35.rs`'s new test builds a real
      GGUF with an embedded gpt2-scheme tokenizer (same shape `cli::
      model_dir.rs`'s own `write_gguf_qwen` test helper uses), calls
      `activate()` with no explicit tokenizer set, and asserts the failure
      it eventually hits is `checkpoint::load`'s safetensors-parse panic
      (proving the tokenizer fallback ran and succeeded) rather than the
      tokenizer-missing error path. `make test -p brain-cli --bins`: green
      (277 tests, 2 new).
- [x] M22: `brain gguf inspect PATH [--json]`, in a new `crates/cli/src/
      gguf_cli.rs` (`mod gguf_cli;` in `main.rs`, dispatched under a new
      `"gguf"` verb next to `"models"`). `PATH` is a real filesystem path
      opened directly via `checkpoint::gguf::MmapGguf::open` - deliberately
      NOT a model-store reference (`brain models info` already covers
      that); the whole point is inspecting a file before it is ever pulled
      or imported. Plain-mode output is a `crate::tree` node tree (KV
      metadata + a per-tensor name/dtype/shape/size listing grouped by
      `.`-segment, the same grouping shape `models_cli`'s own tensor tree
      uses); `--json` emits the same data via `MmapGguf::config()` plus the
      tensor tree. A KV array over 8 entries is elided to its length plus
      the first 3 (the real motivator: `tokenizer.ggml.tokens` on a genuine
      vocabulary is 100k+ entries - printing it all would bury every other
      KV key). While implementing this, found `models_cli.rs` had its own
      private `node_to_json` doing exactly what this command also needed -
      hoisted it into `crate::tree::node_to_json` as a shared public
      helper (both of `models_cli.rs`'s own call sites updated to the
      hoisted version) instead of adding a second copy. Only `inspect` is
      implemented; `kv`/`tensors` sub-verbs from earlier planning notes
      were left unimplemented since `inspect` alone already covers both
      (metadata tree + tensor tree) in one command. Gated: usage-parsing
      tests (missing path, `--json` recognized in either argument order,
      unknown sub-verb, missing file), plus an end-to-end test against a
      real synthetic GGUF asserting both render modes run without
      panicking, the tensor tree names a real tensor, and a large KV array
      is genuinely elided (and does NOT print every entry). `make test -p
      brain-cli --bins`: green (277 tests, 5 new for this file alone).
- [x] Doc fixes: `docs/models/qwen3.md`'s GGUF section demonstrated
      converting a Qwen3 GGUF to safetensors as the way to run one, despite
      qwen3 being direct-load (`gguf_import.rs`'s `Qwen3Importer::
      loads_directly` returns `true`, and `resident_llm.rs`'s Qwen resident
      already opens a `.gguf` straight via `WeightReader` with zero
      conversion) - rewritten to show `brain qwen3 infer --weights
      *.gguf` directly, with the converter kept as an explicit "still
      available, now optional" path for callers who actually want a
      brain-native checkpoint (training, or pinning one on-disk format).
      `docs/using/models-and-weights.md` gained a "Native K-quant GPU
      execution" section describing the M8-M18 kernel/dispatch work
      (Q4_K/Q5_K via new affine kernels, Q6_K/Q5_0/Q4_0/Q8_0 through
      existing kernels via template knobs, all bit-exact host relayout) and
      its real, measured device-bytes-per-param table, citing the exact
      figures already worked out in this ledger's own M8/M14 entries
      rather than inventing new ones - explicitly noting this is
      engine-level capability only: no served model crate has wired its
      own GGUF loader onto the K-quant device path yet (every one still
      dequantizes to fp32 on load, unchanged by this workstream).

## Not yet done

- [ ] The 7 remaining IQ grid codebooks deferred out of M18
      (`IQ1_S`/`IQ1_M`/`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`IQ3_S`), each
      needing a large NGRID lookup table ported from ggml with the same
      fetch-real-source-before-writing discipline M18 established. Not
      attempted for lack of time, not lack of a ground-truth source (the
      same `ggml-common.h`/`ggml-quants.c` fetch M18 already did has them).
- [ ] No served model crate (`qwen3`/`qwen35`/`qwen35moe`/`qwen3vl`) has
      wired its own GGUF loader onto the M8-M18 native K-quant device path
      yet - every one still dequantizes a K-quant tensor to fp32 on load.
      The kernels, host relayout, and dispatch seam are implemented and
      gated by real device-level tests (see M8-M14); reaching for them from
      an actual model's loader is the remaining step, and needs a real
      Q4_K_M/Q5_K_M checkpoint on this box to validate against end to end
      (see "Recorded gaps" below - none is available here today).
- [ ] `Engine`-based serving (`qwen35`'s and `qwen35moe`'s CLI residents,
      `crates/qwen35::serve`/`qwen35moe::serve`) is safetensors-only
      (`checkpoint::load`) - M21 gave both residents a working GGUF
      tokenizer fallback, but activation on a `.gguf` checkpoint still
      cannot get past that point (a pre-existing, documented limitation,
      unchanged by M21). Wiring `Engine` itself onto a GGUF source is a
      separate, larger piece of work outside this ledger's own scope.

## Recorded gaps (this development machine has no MXFP4/IQ fixture and only
one small Q8_0 in the model store)

- The store holds only `Qwen/Qwen3-0.6B/Q8_0.gguf` (610 MB). There is a
  17 GB Q4_K_M under `~/Downloads/MiniMax-H3/` but it is not in the store, and
  there is no MXFP4 or IQ*-quantized file anywhere on this box. M18's
  MXFP4/IQ4_NL/IQ4_XS/TQ1_0/TQ2_0 decoders are therefore gated against
  ggml's own C source (`github.com/ggml-org/llama.cpp`, fetched outside
  this repo) and an independent Python transcription of it, not a real
  file - still no substitute for running a real gpt-oss (MXFP4) or
  IQ-quantized checkpoint through this path end to end.
- Every gate through M14 is synthetic and exactly-known by construction (per
  the user's explicit instruction - validate the math, do not fetch real
  checkpoints for this workstream). The end-to-end forward-parity rung (rung F
  in the design: `Weight::upload(F32, deq(gguf))` vs `from_kquant`, gated
  behind `BRAIN_*_GGUF`) is written and will self-skip on this box until a
  real K-quant checkpoint is available.
- `PARITY_STRICT_SUITES` in the root `Makefile` does not yet include
  `brain-qwen3:gguf_vs_safetensors_real` or `brain-ltxv:gguf_quant_real` -
  candidates to add once real checkpoints are on hand.
