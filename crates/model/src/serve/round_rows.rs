// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How many prompt rows one prefill round pushes through a decoder's layer stack: the policy every engine of the
//! hybrid decoders (`qwen35`, `qwen35moe`) takes, so a round size measured on one is not re-chosen by hand for the
//! other. A round amortises each layer's per-dispatch cost over its rows; on tensor cores that cost is a large share
//! of a narrow round, so the round is wider there.
//!
//! Swedish Embedded AB implements solutions for prefill throughput of served language models for its clients. If
//! your team needs expertise in GPU inference servers then you can procure our services by sending an email to
//! info@swedishembedded.com.

/// How many prompt tokens a prefill round pushes through the layer stack on a device without tensor cores. Measured
/// on `qwen35::int8_gguf_resident` (`Qwen35GgufInstance::generate`), whose history follows.
///
/// The upside is `crate::serve`'s and applies here twice over: a round
/// amortises each stage's per-layer dispatch overhead over `n` rows AND
/// replaces `n` per-token stage hand-offs (there are `n_stages` of those per
/// token) with one.
///
/// The same 256 `crate::serve::Engine` uses - but RE-MEASURED on this model
/// rather than inherited, because the first honest attempt at it did not run
// perf-number: hardware count (2x P40), not a rate
/// at all. Measured on 2x Tesla P40, real Q4_K_M checkpoint at the INT8 tier,
/// 1731-token prompt; `prefill_seconds` from the instance's own metrics, peak
/// card occupancy sampled from `nvidia-smi`:
///
/// | chunk | prefill | tok/s | peak VRAM |
/// |-------|---------|-------|-----------|
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | (per token) | 262.8 s | 6.6 | - |
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 64 | 36.5 s | 47.4 | 16.2 GiB |
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 128 | 26.6 s | 65.0 | 17.8 GiB |
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 192 | 26.4 s | 65.4 | 20.3 GiB |
/// | 256 | OOM | - | (> 24 GiB) |
///
/// A round's transients are not just its GQA scratch: EVERY intermediate a
/// layer allocates stays alive until something drains the queue, and a round
/// used to drain only at its terminal readback - so a stage held all ~32 of
/// its layers' worth at once, at the real 27B widths (`d_model` 5120,
/// `intermediate_size` 17408, `conv_dim` 10240). That is a DEPTH problem, not
/// a chunk-size problem, and capping the chunk is the wrong lever for it: a
/// single-card plan (the Q4 tier fits one) is 64 layers deep and overruns the
/// same budget at half the chunk.
///
/// So the fix went where the defect was - `Qwen35::run_prefill_chunk_stage`'s
/// `DRAIN_EVERY_N_LAYERS` - and the table became:
///
/// | chunk | prefill | tok/s | peak VRAM |
/// |-------|---------|-------|-----------|
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 256 | 26.5 s | 65.4 | 15.3 GiB |
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// | 512 | 27.6 s | 62.8 | 16.6 GiB |
///
/// 256 is now both the fastest measured round size AND cheaper in memory than
/// 64 was before the drain, and 512 is slower - so this is a real optimum
/// rather than a memory-driven compromise.
///
/// One thing that is NOT a consideration here and is in `crate::serve`: the
/// per-stage buffers a round needs are all allocated per call from `n`
/// (`Qwen35::run_prefill_chunk_stage` sizes every one of them), so a stage
/// built at `b = t = 1` carries no FIXED cost for this at all.
///
/// **This constant now applies to a Q4 build too** - see
/// `Qwen35GgufInstance::replay_prompt` and
/// `Qwen35::chunked_prefill_is_profitable`. It did not always: the Q4 tier's
/// `m > 1` GEMM (`matmul_q4_dyn.wgsl`) used to resolve to the naive
/// one-thread-per-output kernel its own header describes, and a round
/// through it cost far MORE than the same tokens through `matmul_q4_gemv`'s
/// coalesced `m = 1` path - measured on the real checkpoint, uniform Q4 on
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// ONE P40, 1555-token prompt: 152.1 s (10.2 tok/s) per token, 1108.3 s
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// (1.4 tok/s) in rounds of 256, a 7.3x REGRESSION, where the same change on
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// the INT8 two-card stack was a 9.9x win. Same host code, same round size;
/// the only difference was which GEMM the weight tier bound.
///
/// Fixed by wiring `matmul_q4_dyn_reg` (128x128 register-tiled, already
/// proven bit-identical to `matmul_q4_dyn`, kernel-performance ledger M5.5:
// perf-number: recorded in kernel-performance ledger M5.5 (matmul_q4_dyn_reg vs matmul_q4_dyn)
/// 2.02x at `m = 32` rising to 12.56x at `m = 2048`) into `model::ops::
/// Ops::bind`. Re-measured on the same real checkpoint, same one P40, the
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// same 1555-token prompt, same round size: **22.9 s (68.0 tok/s)** - a
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// ~6.6x win over the per-token baseline, and slightly FASTER than the
// perf-number: measured by `tests/gguf_resident_real.rs` and `qwen35_gguf_prefill_profile`
/// INT8 stack's own chunked result (64.6-65.4 tok/s) despite streaming
/// fewer bytes per round, not more.
pub const PREFILL_ROUND_ROWS: u32 = 256;

/// The round size on a stack whose every stage runs the chunk round on tensor
/// cores (`gpu_core::provider::cuda::tensor_core_kernels_enabled`).
///
/// 256 is the P40-era optimum above: there the round is arithmetic-bound and
/// a bigger one only costs memory. On tensor cores the arithmetic is cheap
/// enough that what a round pays per DISPATCH - every step is a launch plus a
/// parameter upload, a few thousand of them per round whatever its width - is
/// a real fraction of it, so a wider round amortises that. Measured on a
/// GH200 (`qwen35_gguf_prefill_profile`, production rounds at depth 256, the
/// fastest of repeated rounds on a shared card):
///
/// | rows | ms/round | tok/s |
/// |------|----------|-------|
/// | 256  | 188      | 1363  |
/// | 512  | 309      | 1659  |
/// | 768  | 434      | 1770  |
/// | 1024 | 610      | 1679  |
/// | 2048 | 1028     | 1993  |
///
/// Re-measured with the prefill GEMM chosen per shape class (the Hopper
/// warpgroup candidate wins from about 512 rows): the 2048-row round is
/// faster than 1024 on `qwen35_gguf_prefill_profile` and on `brain perf run
/// longctx` prompts of 1K/8K/32K, and 4096 rows buy only a little more for
/// twice the scratch (the cuda.md ledger has the numbers). With the FP8
/// stack's GEMMs of N = 5120 (down and output projections) a 2048-row round
/// is 16 x 20 = 320 tiles of 128 x 256, 2.4 waves of the 132 multiprocessors
/// run as 3; 4096 rows are 640 tiles, 4.8 waves run as 5. One uncached 32.5K
/// prompt's first token on a GH200 gains little past 4096 rows, while a wider
/// round takes a larger share of the default context in scratch memory.
pub const TENSOR_CORE_PREFILL_ROUND_ROWS: u32 = 4096;

/// Rows of one prefill round, on a stack whose stages all run it on tensor cores or not.
pub fn prefill_round_rows_for(tensor_cores: bool) -> u32 {
    if tensor_cores {
        TENSOR_CORE_PREFILL_ROUND_ROWS
    } else {
        PREFILL_ROUND_ROWS
    }
}
