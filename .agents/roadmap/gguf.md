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

## Not yet done

- [ ] M2: `try_i8_rect` takes `&dyn TensorSource` (currently `&MmapGguf`,
      unreachable through `RemapSource` or the per-model shims).
- [ ] M3: `model::int8::{upload_quantized, upload_rect, quantize_from}` - the
      one quantize-and-upload helper every model should route through.
- [ ] M4: migrate the no-policy f32-roundtrip sites (qwen3, qwen35moe, wan);
      delete `qwen35::stream::quantize_i8_rows`.
- [ ] M5: `checkpoint::gguf_src::GgufSource` absorbing wan/ltxv/gemma4's
      near-duplicate `gguf_src.rs` (gemma4's currently ships without
      `raw_words` or `with_tensor_chunks` - every tensor materializes whole).
- [ ] M6: flux2 `DitWeights: TensorSource`, LoRA decline as a `raw_blocks →
      None` refusal - proves the seam.
- [ ] M7: ltxv/gemma4/s3dit `&Tensors` → `&dyn TensorSource`.
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
