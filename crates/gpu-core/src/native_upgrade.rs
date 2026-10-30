// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Transparent NATIVE kernel upgrades - a native kernel a model inherits
//! without knowing it exists, on the devices whose backend can compile it.
//!
//! Swedish Embedded AB implements memory-bandwidth-bound inference kernels
//! for its clients. If your team needs expertise in getting quantised LLM
//! decode onto the roofline of the memory system, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! [`crate::upgrade`] redirects a registered WGSL kernel to a faster WGSL
//! sibling. This module is its second tier: a row here may redirect a
//! dispatch to a registry kernel from `kernels_cuda` instead, through
//! `Backend::register_native`/`step_native`. The contract is `upgrade`'s -
//! same uniform, same bindings, same output layout - so no call site sees the
//! substitution. The int8 rows also keep the WGSL tier's reduction order and
//! are gated on the raw bits; the dense conv rows accumulate the same
//! reduction with fused multiply-adds and are gated on the fp32 summation
//! bound against an f64 oracle and on exact agreement where the arithmetic is
//! exact (`tests/conv2d_native.rs`); the flash attention row reorders its sums
//! and is held to a stated tolerance against an f64 oracle and to no worse
//! accuracy than the WGSL tier (`tests/flash_bidir_native.rs`).
//!
//! Three conditions, each a queried fact and none a backend name:
//!
//! 1. the WGSL upgrade for the same kernel is ACTIVE on this device, so the
//!    capability policy in `backend_api::select` has already said this
//!    regime wants a workgroup-per-output GEMV at all;
//! 2. the device reports a compute capability, and the registry has a kernel
//!    for this operator and weight tier at or below it;
//! 3. the backend accepts the kernel's source (`register_native` is `Some`).
//!    Every backend that cannot compile CUDA C++ answers `None`, so the
//!    redirect declines there by the same mechanism the CUDA provider does.
//!
//! A dispatch whose shape is outside what the native kernel serves keeps the
//! WGSL tier: the redirect is per dispatch, not per handle.
//!
//! [`set_native_kernels`]`(false)` - which an application calls before it
//! builds any handle, the CLI from its `--no-native-kernels` flag - pins every
//! dispatch to the WGSL tier: the A/B switch the native kernels' own
//! measurements are taken with.

use backend_api::select::{Dtype, Op};
use backend_api::{BindKind, Backend, CudaLaunch, NativeId, NativeSpec};

/// How a [`Row`] names the registry entry it redirects to.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Target {
    /// The registry's best entry for an operator over a weight tier at the
    /// device's capability ([`kernels_cuda::find`]).
    Operator { op: Op, weight: Dtype },
    /// One named entry - a kernel asked for by name rather than resolved by
    /// operator - offered when the device meets its own floor.
    Named(&'static str),
}

impl Target {
    fn resolve(self, cc: kernels_cuda::Cc) -> Option<&'static kernels_cuda::CudaKernel> {
        match self {
            Target::Operator { op, weight } => kernels_cuda::find(op, weight, cc),
            Target::Named(name) => kernels_cuda::get(name).filter(|k| k.min_cc <= cc),
        }
    }
}

/// One drop-in native replacement for a registered kernel.
pub(crate) struct Row {
    /// The kernel name models register and dispatch by index.
    pub slow: &'static str,
    /// The `kernels_cuda` registry entry the dispatch is redirected to.
    pub target: Target,
    /// The storage/uniform bindings of `slow`, in order - the native kernel
    /// takes the identical list.
    pub bindings: &'static [BindKind],
    /// Whether the native kernel serves a dispatch of these caller params.
    pub serves: fn(&[u32]) -> bool,
    /// The BLOCK count a served dispatch launches, from the caller's params
    /// and the registry entry's declared tile. The kernel reconstructs its
    /// own tile from the flat block index by the same rule.
    pub blocks: fn(&[u32], (u32, u32)) -> u32,
    /// Whether `slow` must also have an ACTIVE [`crate::upgrade`] row on this
    /// device (condition 1 of the module doc). A kernel with no WGSL sibling to
    /// upgrade to - there is no regime choice to defer to - says `false`.
    pub requires_wgsl_upgrade: bool,
    /// Words a binding window must start on for this kernel to run (`1` = any). A candidate that copies 16 bytes
    /// at a time cannot take a sliced binding that starts mid-vector; the dispatch then keeps a candidate that can.
    pub align_words: u64,
    /// The activation and weight forms the native kernel consumes: what the dispatch must hand it and what the
    /// tuner grades its error against ([`Input`]).
    pub input: Input,
    /// Launch shape beyond the source (cluster, dynamic shared memory); `programmatic` comes from the registry entry.
    pub launch: CudaLaunch,
}

/// The form a candidate of an int8-weight linear reads its activations in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActFormat {
    /// The activations quantised to int8 with one scale per 32 elements (`quant_pack`), the form every integer-MMA
    /// candidate consumes. Its storage resolution is half a quantisation step of the group.
    Int8Group32,
    /// The fp32 activations rounded to bf16 (round-to-nearest-even, one `cast_f32_bf16`), the form a bf16
    /// tensor-core candidate consumes. No quantisation step and no scale operand: its resolution is the bf16 ulp.
    Bf16,
    /// The fp32 activations quantised to E4M3 with one f32 scale per row per 128 elements (`quant_e4m3_128`): the form of the
    /// FP8 tensor-core GEMM. Its storage resolution is E4M3's (three mantissa bits) at the block's scale.
    Fp8Group128,
}

/// How a candidate reads the stored weights (int8 codes with one f32 scale per 32 elements).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRead {
    /// Codes and scales are consumed separately and exactly (an integer product per group, scaled afterwards).
    Stored,
    /// `code * scale` is formed in registers per element and rounded to bf16 for the tensor core.
    Bf16,
}

/// The input formats of one candidate. They decide which buffers the dispatch binds (the packed activations and their
/// scales, or the bf16 rows) and the storage resolution `native_tune` measures a candidate's error against: a
/// candidate is admitted when its error stays within a factor of the error of rounding ITS inputs to these formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Input {
    pub act: ActFormat,
    pub weight: WeightRead,
}

impl Input {
    /// The integer-MMA candidates: int8 group-quantised activations, stored weights.
    pub const PACKED: Input = Input { act: ActFormat::Int8Group32, weight: WeightRead::Stored };
    /// The dequantize-in-register candidates: bf16 activations, weights `code * scale` rounded to bf16.
    pub const DEQUANT_BF16: Input = Input { act: ActFormat::Bf16, weight: WeightRead::Bf16 };
    /// The FP8 tensor-core candidates: E4M3 activations with a scale per 128, the weights as stored.
    pub const FP8_NATIVE: Input = Input { act: ActFormat::Fp8Group128, weight: WeightRead::Stored };
}

/// The bindings of a [`Input::DEQUANT_BF16`] candidate: params, the bf16 activation rows, the int8 weight codes, the
/// weight scales, the output. There is no activation scale: nothing is quantised.
pub(crate) const I8_DEQUANT_BINDINGS: &[BindKind] =
    &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `Params { m, .., n }`: one block covers `tile.0` rows of x by `tile.1`
/// weight rows, the count `CudaKernel::blocks_for` states.
fn gemm_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[0].div_ceil(tile.0) * p[2].div_ceil(tile.1)
}

/// `matmul_i8_gemv.wgsl`'s bindings: params, `xq`, `wq`, `sx`, `sw`, `out`.
const I8_GEMV_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `matmul_kq_gemv.wgsl`'s and `matmul_kq_dyn.wgsl`'s bindings: params, `xq`,
/// `wq`, `sx`, `wsm`, `wd`, `xgs`, `out`.
const KQ_GEMV_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `Params { m, kg, n }`. The kernel needs `K` to be a whole number of 32
/// element weight-scale groups (`kg % 8 == 0`, the WGSL kernel's own
/// contract). It serves the whole decode regime the WGSL GEMV does: a block
/// covers one tile of rows, and more rows take more blocks of the grid, each
/// re-streaming the weights - four passes at 32 rows is still far less than the
/// register-tiled prefill kernel's mostly idle 128x128 tiles cost there.
fn serves_i8_gemv(p: &[u32]) -> bool {
    let (m, kg, n) = match p {
        [m, kg, n, ..] => (*m, *kg, *n),
        _ => return false,
    };
    (1..=backend_api::select::DECODE_REGIME_MAX_ROWS).contains(&m) && n >= 1 && kg >= 8 && kg % 8 == 0
}

/// The batched decode candidates of the int8 slot: the tensor-core block of the GEMV and the tiled tensor-core
/// GEMMs serve from the first row count whose result is not bit-identical to the single-row dp4a block
/// ([`kernels_cuda::I8_GEMV_MMA_MIN_ROWS`]), so a group of rows the exact-decode contract keeps on the dp4a
/// block never reaches a candidate that would change its bits.
fn serves_i8_batched(p: &[u32], max_rows: u32, kg_multiple: u32) -> bool {
    match p {
        [m, kg, n, ..] => {
            (kernels_cuda::I8_GEMV_MMA_MIN_ROWS..=max_rows).contains(m) && *n >= 1 && *kg >= 8 && kg % kg_multiple == 0
        }
        _ => false,
    }
}

/// The wide tensor-core GEMV: any whole number of scale groups.
fn serves_i8_gemv_wide(p: &[u32]) -> bool {
    serves_i8_batched(p, backend_api::select::DECODE_REGIME_MAX_ROWS, 8)
}

/// The weight-streaming candidate serves the batched rows the tensor-core block serves, whole scale groups of K.
fn serves_i8_stream(p: &[u32]) -> bool {
    serves_i8_batched(p, backend_api::select::DECODE_REGIME_MAX_ROWS, 8)
}

/// Blocks of the weight-streaming candidate split over `SPLIT` blocks of a cluster: one tile of weight rows a cluster.
fn i8_stream_blocks<const SPLIT: u32>(p: &[u32], tile: (u32, u32)) -> u32 {
    p[2].div_ceil(tile.1) * SPLIT
}

/// Cluster shape and dynamic shared memory of the weight-streaming candidate at a split of `split` blocks, `bn` weight
/// rows a block, room for `rows` rows of x and `stages` stages.
const fn i8_stream_launch(split: u32, bn: u32, rows: u32, stages: u32) -> CudaLaunch {
    CudaLaunch {
        dynamic_shared_bytes: kernels_cuda::i8_stream_dynamic_shared_bytes(bn, rows, stages),
        shared_opt_in: true,
        cluster: if split > 1 { Some([split, 1, 1]) } else { None },
        ..CudaLaunch::NONE
    }
}

/// Cluster shape and dynamic shared memory of the dequantize-in-register streaming candidate at a split of `split`
/// blocks, `bn` weight rows a block, room for `rows` rows of x and `stages` stages.
const fn i8_dq_stream_launch(split: u32, bn: u32, rows: u32, stages: u32) -> CudaLaunch {
    CudaLaunch {
        dynamic_shared_bytes: kernels_cuda::i8_dq_stream_dynamic_shared_bytes(bn, rows, stages),
        shared_opt_in: true,
        cluster: if split > 1 { Some([split, 1, 1]) } else { None },
        ..CudaLaunch::NONE
    }
}

/// The many-row weight-streaming candidates serve the rows past the decode regime up to
/// [`kernels_cuda::I8_STREAM_MANY_ROWS`]: a step of a large batch is still bound by reading the weights once.
fn serves_i8_stream_many(p: &[u32]) -> bool {
    match p {
        [m, kg, n, ..] => {
            (backend_api::select::DECODE_REGIME_MAX_ROWS + 1..=kernels_cuda::I8_STREAM_MANY_ROWS).contains(m) && *n >= 1 && *kg >= 8 && kg % 8 == 0
        }
        _ => false,
    }
}

/// The warpgroup weight-streaming candidates serve up to `ROWS` rows of x past the decode regime, whole 128-byte k-tiles.
fn serves_i8_wgstream<const ROWS: u32>(p: &[u32]) -> bool {
    match p {
        [m, kg, n, ..] => (backend_api::select::DECODE_REGIME_MAX_ROWS + 1..=ROWS).contains(m) && *n >= 1 && *kg >= 32 && kg % 32 == 0,
        _ => false,
    }
}

/// Launch of the warpgroup weight-streaming candidate for `xn` rows of x with K split over a cluster of `split` blocks.
const fn i8_wgstream_launch(split: u32, xn: u32) -> CudaLaunch {
    CudaLaunch {
        dynamic_shared_bytes: kernels_cuda::i8_wgstream_dynamic_shared_bytes(xn),
        shared_opt_in: true,
        cluster: Some([split, 1, 1]),
        arch: backend_api::ArchFeatures::Required,
        ..CudaLaunch::NONE
    }
}

/// The tiled tensor-core GEMM at a 16-row tile: `K` a multiple of 64 (two scale groups per staged k-tile).
fn serves_i8_mma_m16(p: &[u32]) -> bool {
    serves_i8_batched(p, 16, 16)
}

/// The tiled tensor-core GEMM at a 32-row tile.
fn serves_i8_mma_m32(p: &[u32]) -> bool {
    serves_i8_batched(p, backend_api::select::DECODE_REGIME_MAX_ROWS, 16)
}

/// The tuning slot of the prefill-sized int8 GEMM. The CUDA provider lowers the int8 prefill GEMM
/// (`provider::cuda`), not a redirect of a model-registered kernel, so its candidates are not rows of the
/// kernel `matmul_i8_dyn` (whose rows must stay bit-identical to the WGSL tier): they live under this virtual slot,
/// which a handle holds when its model registers `matmul_i8_dyn` (the provider's reference kernel), is tuned
/// by [`crate::native_tune`] like any slot, and is asked for by the provider through [`prefill_gemm`].
pub(crate) const PREFILL_SLOT: &str = "matmul_i8_prefill";

/// The pipeline-slot number of [`PREFILL_SLOT`]: past every model kernel index, so no dispatch redirect matches it.
pub(crate) const PREFILL_KIND: usize = usize::MAX;

/// The tuning slot of the FP8 linear (E4M3 weights with 128 x 128 block scales), every row count: like [`PREFILL_SLOT`] it is
/// not a redirect of a model-registered kernel but a virtual slot the handle holds when its model registers the FP8 GEMV
/// (`matmul_fp8_gemv*`), tuned by [`crate::native_tune`] and asked for through [`crate::Gpu::fp8_linear_step`].
pub(crate) const FP8_SLOT: &str = "matmul_fp8_native";

/// The pipeline-slot number of [`FP8_SLOT`]: past every model kernel index and [`PREFILL_KIND`].
pub(crate) const FP8_KIND: usize = usize::MAX - 1;

/// Rows of the prefill-sized GEMM candidates that need no more than the smallest tile: past the decode regime, up
/// to where a bigger tile fills the device.
const PREFILL_SMALL_MAX_ROWS: u32 = 128;

/// Whether `p` is a prefill-sized int8 GEMM `Params { m, kg, n }` every prefill candidate can run: rows past the
/// decode regime (the GEMV slot owns those), `K` a whole number of 64-element staged tiles, an output the 32-bit
/// row arithmetic covers.
fn serves_i8_prefill(p: &[u32]) -> bool {
    match p {
        [m, kg, n, ..] => {
            *m > backend_api::select::DECODE_REGIME_MAX_ROWS && *n >= 1 && *kg >= 16 && kg % 16 == 0 && u64::from(*m) * u64::from(*n) <= u64::from(u32::MAX)
        }
        _ => false,
    }
}

/// An FP8 linear's `Params { m, kg, n }`: K a whole number of 128-wide scale blocks (`kg % 32 == 0`), an output and a weight
/// the 32-bit offsets of the kernels cover.
fn fp8_shape(p: &[u32]) -> Option<(u32, u32, u32)> {
    match p {
        [m, kg, n, ..] if *m >= 1 && *n >= 1 && *kg >= 32 && kg % 32 == 0 && dq_offsets_fit(*kg, *n) && u64::from(*m) * u64::from(*n) <= u64::from(u32::MAX) => Some((*m, *kg, *n)),
        _ => None,
    }
}

/// The dequantize-in-register streaming candidates of the FP8 linear: up to 64 rows with 64 weight rows a block.
fn serves_fp8_stream_narrow(p: &[u32]) -> bool {
    fp8_shape(p).is_some_and(|(m, ..)| m <= kernels_cuda::I8_STREAM_MANY_ROWS)
}

/// The wide FP8 streaming candidates: up to [`kernels_cuda::I8_DQ_STREAM_WIDE_ROWS`] rows with 128 weight rows a block.
fn serves_fp8_stream_wide(p: &[u32]) -> bool {
    fp8_shape(p).is_some_and(|(m, ..)| m <= kernels_cuda::I8_DQ_STREAM_WIDE_ROWS)
}

/// The f16-MMA FP8 streaming candidates: up to [`kernels_cuda::FP8_GEMV_F16_ROWS`] rows, a weight row a whole number of 128-wide scale blocks.
fn serves_fp8_gemv_f16(p: &[u32]) -> bool {
    fp8_shape(p).is_some_and(|(m, ..)| m <= kernels_cuda::FP8_GEMV_F16_ROWS)
}

/// One block per `tile.1` weight rows: the candidate keeps all of K inside the block.
fn fp8_gemv_f16_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[2].div_ceil(tile.1)
}

/// The Hopper FP8 GEMM: the rows a 128-row tile is worth filling, K a whole number of 128-wide scale blocks.
fn serves_fp8_wgmma(p: &[u32]) -> bool {
    fp8_shape(p).is_some_and(|(m, ..)| m > backend_api::select::DECODE_REGIME_MAX_ROWS)
}

/// Cluster shape and dynamic shared memory of the FP8 streaming candidate at a split of `split` blocks, `bn` weight rows
/// a block, room for `rows` rows of x and `stages` stages.
const fn fp8_dq_stream_launch(split: u32, bn: u32, rows: u32, stages: u32) -> CudaLaunch {
    CudaLaunch {
        dynamic_shared_bytes: kernels_cuda::fp8_dq_stream_dynamic_shared_bytes(bn, rows, stages),
        shared_opt_in: true,
        cluster: if split > 1 { Some([split, 1, 1]) } else { None },
        ..CudaLaunch::NONE
    }
}

/// The dequantize-in-register kernels address a weight element with a 32-bit offset (`n * K` bytes).
fn dq_offsets_fit(kg: u32, n: u32) -> bool {
    u64::from(kg) * 4 * u64::from(n) <= u64::from(u32::MAX)
}

/// The dequantize-in-register streaming candidates past the decode regime.
fn serves_i8_dq_stream_many(p: &[u32]) -> bool {
    serves_i8_stream_many(p) && dq_offsets_fit(p[1], p[2])
}

/// The wide dequantize-in-register streaming candidates: up to [`kernels_cuda::I8_DQ_STREAM_WIDE_ROWS`] rows of x past
/// the decode regime.
fn serves_i8_dq_stream_wide(p: &[u32]) -> bool {
    match p {
        [m, kg, n, ..] => {
            (backend_api::select::DECODE_REGIME_MAX_ROWS + 1..=kernels_cuda::I8_DQ_STREAM_WIDE_ROWS).contains(m) && *n >= 1 && *kg >= 8 && kg % 8 == 0 && dq_offsets_fit(*kg, *n)
        }
        _ => false,
    }
}

/// The dequantize-in-register prefill GEMM's shapes: rows past the decode regime, whole scale groups of K, an output the
/// 32-bit row arithmetic covers.
fn serves_i8_dq_prefill(p: &[u32]) -> bool {
    match p {
        [m, kg, n, ..] => *m > backend_api::select::DECODE_REGIME_MAX_ROWS && *n >= 1 && *kg >= 8 && kg % 8 == 0 && u64::from(*m) * u64::from(*n) <= u64::from(u32::MAX) && dq_offsets_fit(*kg, *n),
        _ => false,
    }
}

/// The Hopper dequantize-in-shared-memory prefill GEMM: those shapes with whole 64-deep k-tiles.
fn serves_i8_dq_wgmma(p: &[u32]) -> bool {
    serves_i8_dq_prefill(p) && p[1].is_multiple_of(16)
}

/// The small-tile prefill candidate: the rows that do not fill a bigger tile.
fn serves_i8_prefill_small(p: &[u32]) -> bool {
    serves_i8_prefill(p) && p[0] <= PREFILL_SMALL_MAX_ROWS
}

/// The Hopper warpgroup candidate: whole 128-deep k-tiles.
fn serves_i8_prefill_wgmma(p: &[u32]) -> bool {
    serves_i8_prefill(p) && p[1].is_multiple_of(32)
}

/// `matmul_q4_gemv`'s (and `matmul_kq_gemv`'s) `Params { m, k, n }` with the
/// RAW `k`: the int8 GEMV's decode regime, `K` a whole number of 32-element
/// weight-scale groups.
fn serves_q4_gemv(p: &[u32]) -> bool {
    match p {
        [m, k, n, ..] => (1..=backend_api::select::DECODE_REGIME_MAX_ROWS).contains(m) && *n >= 1 && *k >= 32 && k % 32 == 0,
        _ => false,
    }
}

/// Rows of x from which the K-quant GEMV runs its wide block (four weight rows
/// a thread): below it the weights' bandwidth bounds the narrow block, above
/// it the math and shared reads per row do (`tests/quant_crossover_bench.rs`).
pub(crate) const KQ_GEMV_WIDE_MIN_ROWS: u32 = 9;

/// The K-quant GEMV's narrow block: [`serves_q4_gemv`] below
/// [`KQ_GEMV_WIDE_MIN_ROWS`].
fn serves_kq_gemv_narrow(p: &[u32]) -> bool {
    serves_q4_gemv(p) && p[0] < KQ_GEMV_WIDE_MIN_ROWS
}

/// The K-quant GEMV's wide block: [`serves_q4_gemv`] from [`KQ_GEMV_WIDE_MIN_ROWS`].
fn serves_kq_gemv_wide(p: &[u32]) -> bool {
    serves_q4_gemv(p) && p[0] >= KQ_GEMV_WIDE_MIN_ROWS
}

/// `matmul_kq_dyn`'s `Params { m, k, n }` with the RAW `k`: any non-empty
/// output whose `K` is a whole number of 32-element groups, inside the
/// kernel's 32-bit row arithmetic.
fn serves_kq_dyn(p: &[u32]) -> bool {
    match p {
        [m, k, n] => *m >= 1 && *n >= 1 && *k >= 32 && k % 32 == 0 && u64::from(*m) * u64::from(*n) <= u64::from(u32::MAX),
        _ => false,
    }
}

/// `matmul_i8_dyn`'s `Params { m, kg, n }` (same bindings as the GEMV): any
/// non-empty output whose `K` is a whole number of 32-element weight-scale
/// groups, the WGSL kernel's own contract, with every operand inside the
/// kernel's 32-bit row arithmetic.
fn serves_i8_dyn(p: &[u32]) -> bool {
    match p {
        [m, kg, n] => {
            let limit = u64::from(u32::MAX);
            *m >= 1 && *n >= 1 && *kg >= 8 && kg % 8 == 0 && u64::from(*m) * u64::from(*n) <= limit
        }
        _ => false,
    }
}

/// The fp32 GEMM family's bindings (`matmul_reg3.wgsl`): params, `x`, `w`, `out`.
const F32_GEMM_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `Params { m, k, n }`: any non-empty product whose output fits the kernel's
/// 32-bit element arithmetic.
fn serves_f32_gemm(p: &[u32]) -> bool {
    matches!(p, [m, k, n] if *m >= 1 && *k >= 1 && *n >= 1 && u64::from(*m) * u64::from(*n) <= u64::from(u32::MAX))
}

/// `matmul_dx_reg`'s `Params { m, k, n, accumulate }`: any non-empty product
/// whose `m x k` output fits the kernel's 32-bit element arithmetic.
fn serves_f32_dx(p: &[u32]) -> bool {
    matches!(p, [m, k, n, _] if *m >= 1 && *k >= 1 && *n >= 1 && u64::from(*m) * u64::from(*k) <= u64::from(u32::MAX))
}

/// The input gradient's output is `m x k`: `params[1]` is its width.
fn dx_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[0].div_ceil(tile.0) * p[1].div_ceil(tile.1)
}

/// `matmul_dw_reg`'s `Params { m, k, n }`: any non-empty product whose
/// `n x k` output fits the kernel's 32-bit element arithmetic.
fn serves_f32_dw(p: &[u32]) -> bool {
    matches!(p, [m, k, n] if *m >= 1 && *k >= 1 && *n >= 1 && u64::from(*n) * u64::from(*k) <= u64::from(u32::MAX))
}

/// The weight gradient's output is `n x k`.
fn dw_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[2].div_ceil(tile.0) * p[1].div_ceil(tile.1)
}

/// `moe_i8_grouped.wgsl`'s bindings: params, `xq`, `sx`, `tab`, `perm`, `wq`, `sw`,
/// `out`.
const MOE_I8_GROUPED_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `Params { tiles, kg, n, xdiv, ne }`. The kernel needs `K` to be a whole number
/// of 32-element weight-scale groups (`kg % 8 == 0`, the WGSL kernel's own
/// contract) and the routing tables `ne` experts wide.
fn serves_moe_i8_grouped(p: &[u32]) -> bool {
    match p {
        [tiles, kg, n, xdiv, ne, ..] => *tiles >= 1 && *n >= 1 && *kg >= 8 && kg % 8 == 0 && *xdiv >= 1 && *ne >= 1,
        _ => false,
    }
}

/// The dense conv family's bindings (`conv2d.wgsl`, `conv2d_dx.wgsl`): params,
/// two read operands, the written output.
const CONV2D_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `conv_bias.wgsl`'s and `conv_bias_reg.wgsl`'s bindings: params, `x`, `w`,
/// `bias`, the written `y`.
const CONV_BIAS_BINDINGS: &[BindKind] =
    &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// The dense conv uniform `[N, Cin, H, W, Cout, K, stride, pad, Ho, Wo]`, when
/// it describes a convolution the native kernels serve: every extent non-zero,
/// the output extent the one the input, kernel, stride and padding give, and
/// every tensor small enough for the kernels' 32-bit signed element offsets.
/// The grouped/dilated ABI is twelve words long and never matches.
fn conv_shape(p: &[u32]) -> Option<[u64; 10]> {
    let p: [u32; 10] = p.get(..10)?.try_into().ok()?;
    let [n, cin, h, w, cout, k, s, pad, ho, wo] = p.map(u64::from);
    if [n, cin, h, w, cout, k, s, ho, wo].contains(&0) || h + 2 * pad < k || w + 2 * pad < k {
        return None;
    }
    if (h + 2 * pad - k) / s + 1 != ho || (w + 2 * pad - k) / s + 1 != wo {
        return None;
    }
    let limit = i32::MAX as u64;
    (n * cin * h * w <= limit && n * cout * ho * wo <= limit && cout * cin * k * k <= limit).then_some([n, cin, h, w, cout, k, s, pad, ho, wo])
}

/// The ten-word dense conv uniform, and nothing longer: a dispatch that
/// carries more words is a different ABI.
fn serves_conv(p: &[u32]) -> bool {
    p.len() == 10 && conv_shape(p).is_some()
}

/// Output channels (forward, weight gradient) or input channels (input
/// gradient) one block of the conv kernels covers: the smallest of 16/32/64
/// that holds them. The SAME rule as `brain_cv_bm` in `cu/conv2d_f32.cu`; a
/// disagreement leaves a tail of the output unwritten, which the conv gate's
/// 16- and 32-channel shapes would show.
fn conv_rows_tile(rows: u32) -> u32 {
    if rows <= 16 {
        16
    } else if rows <= 32 {
        32
    } else {
        64
    }
}

/// `brain_conv2d_fwd`: channel tiles x position tiles of `tile.1`.
fn conv2d_fwd_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    let (n, cout, ho, wo) = (p[0], p[4], p[8], p[9]);
    cout.div_ceil(conv_rows_tile(cout)) * (n * ho * wo).div_ceil(tile.1)
}

/// `brain_conv2d_dx`: input-channel tiles x stride classes x position tiles
/// of the LARGEST class (`ceil(H/s) * ceil(W/s)` per image); blocks of the
/// smaller classes past their own extent exit at once.
fn conv2d_dx_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    let (n, cin, h, w, s) = (p[0], p[1], p[2], p[3], p[6]);
    cin.div_ceil(conv_rows_tile(cin)) * s * s * (n * h.div_ceil(s) * w.div_ceil(s)).div_ceil(tile.1)
}

/// Output channels and output positions (a square patch, side
/// `CONV3X3_SIDE`) one block of the direct 3x3 forward covers:
/// `BRAIN_C3_CO`, `BRAIN_C3_TH` and `BRAIN_C3_TW` in `cu/conv2d_f32.cu`.
const CONV3X3_CHANNELS: u32 = 64;
const CONV3X3_SIDE: u32 = 16;

/// A dense conv the direct 3x3 forward serves: 3x3, stride 1, pad 1 (so the
/// output extent is the input's) and at least one full channel tile wide -
/// `brain_c3_serves` in `cu/conv2d_f32.cu`.
fn serves_conv3x3(p: &[u32]) -> bool {
    serves_conv(p) && p[5] == 3 && p[6] == 1 && p[7] == 1 && p[4] >= CONV3X3_CHANNELS
}

/// Channel tiles x images x patch rows x patch columns.
fn conv3x3_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    let (n, h, w, cout) = (p[0], p[2], p[3], p[4]);
    cout.div_ceil(CONV3X3_CHANNELS) * n * h.div_ceil(CONV3X3_SIDE) * w.div_ceil(CONV3X3_SIDE)
}

/// `paged_kv_append_i8_clipped_batched.wgsl`'s bindings: params, `src`, `blocks`, `offsets`, `clip`, `pool`, `scales`.
const KV_APPEND_I8_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// `Params { batch, kv_stride, block_size, head_dim }`: heads of a whole number of four-value words.
fn serves_kv_append_i8(p: &[u32]) -> bool {
    matches!(p, [batch, kv_stride, block_size, hd] if *batch >= 1 && *block_size >= 1 && *hd >= 4 && hd % 4 == 0 && kv_stride % hd == 0 && *kv_stride >= *hd)
}

/// One block covers four (token, kv head) pairs.
fn kv_append_i8_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    (p[0] * (p[1] / p[3])).div_ceil(tile.0)
}

/// `paged_flash_decode_gqa_hd256.wgsl`'s bindings for a scaled 8-bit tier (int8, FP8): params, `q`, the two pools,
/// `block_tables`, `seq_lens`, the written partial records, then the key and value row scales.
const FLASH_DECODE_KV8_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
];

/// `paged_flash_decode_gqa_hd256.wgsl`'s bindings for an unscaled tier (f32, bf16): params, `q`, the two pools,
/// `block_tables`, `seq_lens`, the written partial records.
const FLASH_DECODE_BINDINGS: &[BindKind] =
    &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// The query heads per kv head the native int8 decode kernel is compiled for
/// (`BRAIN_GRP` in its registry entry).
const FLASH_DECODE_GROUP: u32 = 6;

/// `[batch, n_heads, n_kv_heads, head_dim, group, block_size, max_bt,
/// n_splits, tiles_per_split]` at head_dim 256 and the compiled group, with a
/// grid the 32-bit block index covers.
fn serves_flash_decode(p: &[u32]) -> bool {
    matches!(p, [batch, heads, nkv, 256, FLASH_DECODE_GROUP, bs, bt, splits, per]
        if *batch >= 1 && *nkv >= 1 && *heads == nkv * FLASH_DECODE_GROUP && *bs >= 1 && *bt >= 1 && *splits >= 1 && *per >= 1
            && u64::from(*batch) * u64::from(*nkv) * u64::from(*splits) <= u64::from(u32::MAX))
}

/// One block per (sequence, kv head, key split).
fn flash_decode_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[0] * p[2] * p[7]
}

/// Splits the native combine holds weights for in shared memory (`MAX_SPLITS` in its source).
const FLASH_COMBINE_MAX_SPLITS: u32 = 4096;

/// `[batch, n_heads, head_dim, n_splits]` with a head of at most one block's 256 channels and splits that fit.
fn serves_flash_combine(p: &[u32]) -> bool {
    matches!(p, [batch, heads, hd, splits] if *batch >= 1 && *heads >= 1 && (1..=256).contains(hd) && (1..=FLASH_COMBINE_MAX_SPLITS).contains(splits)
        && u64::from(*batch) * u64::from(*heads) * 4 <= u64::from(u32::MAX))
}

/// One block per 64 channels (`THREADS` in the kernel) of a (sequence, head).
fn flash_combine_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[0] * p[1] * p[2].div_ceil(64)
}

/// `matmul_fp8_gemv.wgsl`'s and `matmul_nvfp4_gemv.wgsl`'s bindings: params, `x`, `w`, `s`, `y`.
const QGEMV_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `matmul_bf16_gemv.wgsl`'s bindings: params, `x`, `w`, `y`.
const BF16_GEMV_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// The bf16 GEMVs' `[m, k, count, first, xr0, rpw, pad, pad]`: up to `rows` activation rows, `k` whole 8-weight chunks.
fn serves_bf16_gemv(p: &[u32], rows: u32) -> bool {
    matches!(p, [m, k, count, ..] if (1..=rows).contains(m) && *count >= 1 && *k >= 8 && k % 8 == 0)
}
fn serves_bf16_gemv_m1(p: &[u32]) -> bool {
    serves_bf16_gemv(p, 1)
}
fn serves_bf16_gemv_m2(p: &[u32]) -> bool {
    serves_bf16_gemv(p, 2)
}
fn serves_bf16_gemv_m4(p: &[u32]) -> bool {
    serves_bf16_gemv(p, 4)
}
fn serves_bf16_gemv_m8(p: &[u32]) -> bool {
    serves_bf16_gemv(p, 8)
}

/// Weight rows one block of the native bf16 GEMV covers (`BRAIN_BG_COLS`): a warp per row.
const BF16_GEMV_COLS: u32 = 8;

/// Weight rows one block of the staged native bf16 GEMV covers: eight warps of four rows.
const BF16_GEMV_STAGED_COLS: u32 = 32;

/// One block per 32 weight rows.
fn bf16_gemv_staged_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[2].div_ceil(BF16_GEMV_STAGED_COLS)
}

/// One block per 8 weight rows.
fn bf16_gemv_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[2].div_ceil(BF16_GEMV_COLS)
}

/// `moe_nvfp4_gemv_gather.wgsl`'s bindings: params, `x`, `ids`, `wq`, `sc`, `gl`, `out`.
const MOE_NVFP4_GATHER_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `moe_nvfp4_gateup_silu`'s bindings: params, `x`, `ids`, `wg`, `sg`, `gg`, `wu`, `su`, `gu`, `h`.
const MOE_NVFP4_GATEUP_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `moe_router_topk.wgsl`'s bindings: params, `logits`, `ids`, `weight`.
const MOE_ROUTER_TOPK_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite, BindKind::StorageReadWrite];

/// The router head's `[rows, n_experts, top_k, has_shared, norm, scale]`: one 256-thread block holds at most 4 experts per thread and 16 picks.
fn serves_moe_router_topk(p: &[u32]) -> bool {
    matches!(p, [rows, n, k, shared, norm, _scale] if *rows >= 1 && (1..=1024).contains(n) && (1..=16).contains(k) && *shared <= 1 && *norm <= 1)
}

/// One block per row.
fn moe_router_topk_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[0]
}

/// `moe_nvfp4_grouped.wgsl`'s bindings: params, `x`, `tab`, `perm`, `wq`, `sc`, `gl`, `out`.
const MOE_NVFP4_GROUPED_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// Weight rows one block of the native grouped NVFP4 kernel covers.
const MOE_NVFP4_GROUPED_COLS: u32 = 64;

/// The grouped kernel's `[tiles, k, n, xdiv, ne]`, with the grid the 32-bit block index covers.
fn serves_moe_nvfp4_grouped(p: &[u32]) -> bool {
    matches!(p, [tiles, k, n, xdiv, ne] if *tiles >= 1 && *n >= 1 && *k >= 64 && k % 64 == 0 && *xdiv >= 1 && *ne >= 1
        && u64::from(*tiles) * u64::from(n.div_ceil(MOE_NVFP4_GROUPED_COLS)) <= u64::from(u32::MAX))
}

/// Tiles x column tiles of 64 weight rows.
fn moe_nvfp4_grouped_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[0] * p[2].div_ceil(MOE_NVFP4_GROUPED_COLS)
}

/// Weight rows one block of the stored-format GEMV kernels covers (`BRAIN_QG_COLS`).
const QGEMV_COLS: u32 = 16;

/// The FP8 GEMV's `[m, k, count, first, block, cb, xr0, rpw]`: up to 8 activation rows, `k` and the scale block whole
/// 16-weight chunks.
fn serves_fp8_gemv(p: &[u32]) -> bool {
    matches!(p, [m, k, count, _, block, _, _, _] if (1..=8).contains(m) && *count >= 1 && *k >= 16 && k % 16 == 0 && *block >= 16 && block % 16 == 0)
}

/// The NVFP4 GEMV's `[m, k, count, first, xr0, global_bits, rpw, pad]`: up to 8 activation rows, `k` whole 64-weight rows
/// (the scale bytes of a row are four to a word).
fn serves_nvfp4_gemv(p: &[u32]) -> bool {
    matches!(p, [m, k, count, ..] if (1..=8).contains(m) && *count >= 1 && *k >= 64 && k % 64 == 0)
}

/// One block per 16 weight rows.
fn qgemv_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[2].div_ceil(QGEMV_COLS)
}

/// The gather's `[slots, k, n, xdiv]`.
fn serves_moe_nvfp4_gather(p: &[u32]) -> bool {
    matches!(p, [slots, k, n, xdiv] if *slots >= 1 && *n >= 1 && *k >= 64 && k % 64 == 0 && *xdiv >= 1
        && u64::from(*slots) * u64::from(n.div_ceil(QGEMV_COLS)) <= u64::from(u32::MAX))
}

/// One block per 16 weight rows of a slot, slot-major.
fn moe_nvfp4_gather_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[0] * p[2].div_ceil(QGEMV_COLS)
}

/// The bindings of an FP8 candidate: params, the bf16 activation rows, the E4M3 weight bytes, the 128 x 128 block scales, the
/// output. The same four buffers as the int8 dequantize candidates, so one dispatch shape serves both.
const FP8_BINDINGS: &[BindKind] = I8_DEQUANT_BINDINGS;

/// The candidates of [`FP8_SLOT`], the FP8 linear, in preference order: the first row that serves a dispatch is the
/// untuned default, the rest are what [`crate::native_tune`] times against it. The dequantize-in-register candidates read
/// bf16 activations and decode the weight bytes times their block scale ([`Input::DEQUANT_BF16`]); the tensor-core GEMM
/// reads E4M3 activations and the weights as stored ([`Input::FP8_NATIVE`]); the bf16 forms come first, so an untuned
/// device keeps the activations at the finer resolution.
pub(crate) const FP8_ROWS: &[Row] = &[
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_dq_stream_r1"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_stream_narrow,
        blocks: i8_stream_blocks::<1>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: fp8_dq_stream_launch(1, 64, kernels_cuda::I8_STREAM_MANY_ROWS, 3),
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_dq_stream_r2"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_stream_narrow,
        blocks: i8_stream_blocks::<2>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: fp8_dq_stream_launch(2, 64, kernels_cuda::I8_STREAM_MANY_ROWS, 3),
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_dq_stream_r4"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_stream_narrow,
        blocks: i8_stream_blocks::<4>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: fp8_dq_stream_launch(4, 64, kernels_cuda::I8_STREAM_MANY_ROWS, 3),
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_dq_stream_q2"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_stream_wide,
        blocks: i8_stream_blocks::<2>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: fp8_dq_stream_launch(2, 128, kernels_cuda::I8_DQ_STREAM_WIDE_ROWS, 3),
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_dq_stream_q4"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_stream_wide,
        blocks: i8_stream_blocks::<4>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: fp8_dq_stream_launch(4, 128, kernels_cuda::I8_DQ_STREAM_WIDE_ROWS, 3),
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_gemv_f16_a1w4d3b3"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_gemv_f16,
        blocks: fp8_gemv_f16_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: CudaLaunch {
            dynamic_shared_bytes: kernels_cuda::fp8_gemv_f16_dynamic_shared_bytes(1, 4, 3),
            shared_opt_in: true,
            ..CudaLaunch::NONE
        },
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_gemv_f16_a1w4d4b5"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_gemv_f16,
        blocks: fp8_gemv_f16_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: CudaLaunch {
            dynamic_shared_bytes: kernels_cuda::fp8_gemv_f16_dynamic_shared_bytes(1, 4, 4),
            shared_opt_in: true,
            ..CudaLaunch::NONE
        },
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_gemv_f16_a1w2d6b6"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_gemv_f16,
        blocks: fp8_gemv_f16_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: CudaLaunch {
            dynamic_shared_bytes: kernels_cuda::fp8_gemv_f16_dynamic_shared_bytes(1, 2, 6),
            shared_opt_in: true,
            ..CudaLaunch::NONE
        },
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_gemv_f16_a2w4d4b3"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_gemv_f16,
        blocks: fp8_gemv_f16_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: CudaLaunch {
            dynamic_shared_bytes: kernels_cuda::fp8_gemv_f16_dynamic_shared_bytes(2, 4, 4),
            shared_opt_in: true,
            ..CudaLaunch::NONE
        },
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_dq_wgmma"),
        bindings: FP8_BINDINGS,
        serves: serves_fp8_wgmma,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: CudaLaunch {
            dynamic_shared_bytes: kernels_cuda::FP8_DQ_WGMMA_DYNAMIC_SHARED_BYTES,
            shared_opt_in: true,
            arch: backend_api::ArchFeatures::Required,
            ..CudaLaunch::NONE
        },
    },
    Row {
        slow: "matmul_fp8_gemv",
        target: Target::Named("matmul_fp8_wgmma"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_fp8_wgmma,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::FP8_NATIVE,
        launch: CudaLaunch {
            dynamic_shared_bytes: kernels_cuda::FP8_WGMMA_DYNAMIC_SHARED_BYTES,
            shared_opt_in: true,
            arch: backend_api::ArchFeatures::Required,
            ..CudaLaunch::NONE
        },
    },
];

/// The candidates of [`PREFILL_SLOT`], the prefill-sized int8 GEMM, in preference order: the first row that serves
/// a dispatch is the untuned default (the routing the provider had before the slot was measured), the rest are
/// what [`crate::native_tune`] times against it. Every row states the capability it needs through its registry
/// entry's floor, so a device only ever sees the candidates it can run. `slow` names the kernel a handle must
/// register for these rows to exist: the provider's reference kernel.
pub(crate) const PREFILL_ROWS: &[Row] = &[
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_mma_m128"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_prefill_small,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_mma"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_prefill,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_dq_wgmma"),
        bindings: I8_DEQUANT_BINDINGS,
        serves: serves_i8_dq_wgmma,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: CudaLaunch {
            dynamic_shared_bytes: kernels_cuda::I8_DQ_WGMMA_DYNAMIC_SHARED_BYTES,
            shared_opt_in: true,
            arch: backend_api::ArchFeatures::Required,
            ..CudaLaunch::NONE
        },
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_dq_stream_q2"),
        bindings: I8_DEQUANT_BINDINGS,
        serves: serves_i8_dq_stream_wide,
        blocks: i8_stream_blocks::<2>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: i8_dq_stream_launch(2, 128, kernels_cuda::I8_DQ_STREAM_WIDE_ROWS, 3),
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_dq_stream_q4"),
        bindings: I8_DEQUANT_BINDINGS,
        serves: serves_i8_dq_stream_wide,
        blocks: i8_stream_blocks::<4>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: i8_dq_stream_launch(4, 128, kernels_cuda::I8_DQ_STREAM_WIDE_ROWS, 3),
    },
    // The weight-streaming candidates of the many-row decode step (33-128 rows of x): the structure of the decode
    // linear's streaming set (`matmul_i8_stream_*`) with room for 128 rows of x.
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_stream_r1"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream_many,
        blocks: i8_stream_blocks::<1>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(1, 64, kernels_cuda::I8_STREAM_MANY_ROWS, 3),
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_stream_r2"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream_many,
        blocks: i8_stream_blocks::<2>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(2, 64, kernels_cuda::I8_STREAM_MANY_ROWS, 3),
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_stream_r4"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream_many,
        blocks: i8_stream_blocks::<4>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(4, 64, kernels_cuda::I8_STREAM_MANY_ROWS, 3),
    },
    // The same for the many-row decode step (33-64 rows of x).
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_dq_stream_r1"),
        bindings: I8_DEQUANT_BINDINGS,
        serves: serves_i8_dq_stream_many,
        blocks: i8_stream_blocks::<1>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: i8_dq_stream_launch(1, 64, 64, 3),
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_dq_stream_r2"),
        bindings: I8_DEQUANT_BINDINGS,
        serves: serves_i8_dq_stream_many,
        blocks: i8_stream_blocks::<2>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: i8_dq_stream_launch(2, 64, 64, 3),
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_dq_stream_r4"),
        bindings: I8_DEQUANT_BINDINGS,
        serves: serves_i8_dq_stream_many,
        blocks: i8_stream_blocks::<4>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::DEQUANT_BF16,
        launch: i8_dq_stream_launch(4, 64, 64, 3),
    },
    // The same, with the weights as the A operand of a warpgroup `wgmma` (`matmul_i8_wgstream.cu`).
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_wgstream_n64_s2"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_wgstream::<64>,
        blocks: i8_stream_blocks::<2>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_wgstream_launch(2, 64),
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_wgstream_n64_s4"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_wgstream::<64>,
        blocks: i8_stream_blocks::<4>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_wgstream_launch(4, 64),
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_wgstream_n128_s2"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_wgstream::<128>,
        blocks: i8_stream_blocks::<2>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_wgstream_launch(2, 128),
    },
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_wgstream_n128_s4"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_wgstream::<128>,
        blocks: i8_stream_blocks::<4>,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_wgstream_launch(4, 128),
    },
    // The Hopper warpgroup GEMM: `wgmma` is arch-specific (`sm_90a`), so the launch requires it, and its ring of
    // stages is dynamic shared memory above the default limit.
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_wgmma"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_prefill_wgmma,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 4,
        input: Input::PACKED,
        launch: CudaLaunch {
            dynamic_shared_bytes: kernels_cuda::I8_WGMMA_DYNAMIC_SHARED_BYTES,
            shared_opt_in: true,
            arch: backend_api::ArchFeatures::Required,
            ..CudaLaunch::NONE
        },
    },
];

pub(crate) const ROWS: &[Row] = &[
    Row {
        slow: "matmul_bf16_gemv#MAXM=1",
        target: Target::Named("matmul_bf16_gemv_m1"),
        bindings: BF16_GEMV_BINDINGS,
        serves: serves_bf16_gemv_m1,
        blocks: bf16_gemv_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_bf16_gemv#MAXM=2",
        target: Target::Named("matmul_bf16_gemv_m2"),
        bindings: BF16_GEMV_BINDINGS,
        serves: serves_bf16_gemv_m2,
        blocks: bf16_gemv_staged_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_bf16_gemv#MAXM=4",
        target: Target::Named("matmul_bf16_gemv_m4"),
        bindings: BF16_GEMV_BINDINGS,
        serves: serves_bf16_gemv_m4,
        blocks: bf16_gemv_staged_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_bf16_gemv#MAXM=8",
        target: Target::Named("matmul_bf16_gemv_m8"),
        bindings: BF16_GEMV_BINDINGS,
        serves: serves_bf16_gemv_m8,
        blocks: bf16_gemv_staged_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_fp8_gemv#MAXM=1",
        target: Target::Named("matmul_fp8_gemv_m1"),
        bindings: QGEMV_BINDINGS,
        serves: serves_fp8_gemv,
        blocks: qgemv_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_fp8_gemv#MAXM=2",
        target: Target::Named("matmul_fp8_gemv"),
        bindings: QGEMV_BINDINGS,
        serves: serves_fp8_gemv,
        blocks: qgemv_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_fp8_gemv#MAXM=4",
        target: Target::Named("matmul_fp8_gemv"),
        bindings: QGEMV_BINDINGS,
        serves: serves_fp8_gemv,
        blocks: qgemv_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_fp8_gemv#MAXM=8",
        target: Target::Named("matmul_fp8_gemv"),
        bindings: QGEMV_BINDINGS,
        serves: serves_fp8_gemv,
        blocks: qgemv_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_nvfp4_gemv#MAXM=1",
        target: Target::Named("matmul_nvfp4_gemv_m1"),
        bindings: QGEMV_BINDINGS,
        serves: serves_nvfp4_gemv,
        blocks: qgemv_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_nvfp4_gemv#MAXM=2",
        target: Target::Named("matmul_nvfp4_gemv"),
        bindings: QGEMV_BINDINGS,
        serves: serves_nvfp4_gemv,
        blocks: qgemv_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_nvfp4_gemv#MAXM=4",
        target: Target::Named("matmul_nvfp4_gemv"),
        bindings: QGEMV_BINDINGS,
        serves: serves_nvfp4_gemv,
        blocks: qgemv_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_nvfp4_gemv#MAXM=8",
        target: Target::Named("matmul_nvfp4_gemv"),
        bindings: QGEMV_BINDINGS,
        serves: serves_nvfp4_gemv,
        blocks: qgemv_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "moe_router_topk",
        target: Target::Named("moe_router_topk"),
        bindings: MOE_ROUTER_TOPK_BINDINGS,
        serves: serves_moe_router_topk,
        blocks: moe_router_topk_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "moe_nvfp4_grouped",
        target: Target::Named("moe_nvfp4_grouped"),
        bindings: MOE_NVFP4_GROUPED_BINDINGS,
        serves: serves_moe_nvfp4_grouped,
        blocks: moe_nvfp4_grouped_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "moe_nvfp4_gemv_gather",
        target: Target::Named("moe_nvfp4_gemv_gather"),
        bindings: MOE_NVFP4_GATHER_BINDINGS,
        serves: serves_moe_nvfp4_gather,
        blocks: moe_nvfp4_gather_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "paged_kv_append_i8_clipped_batched",
        target: Target::Named("kv_append_i8_clipped"),
        bindings: KV_APPEND_I8_BINDINGS,
        serves: serves_kv_append_i8,
        blocks: kv_append_i8_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "paged_flash_decode_gqa_hd256#GRP=6#kv=int8",
        target: Target::Named("paged_flash_decode_gqa_hd256_kv8"),
        bindings: FLASH_DECODE_KV8_BINDINGS,
        serves: serves_flash_decode,
        blocks: flash_decode_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "paged_flash_decode_gqa_hd256#GRP=6",
        target: Target::Named("paged_flash_decode_gqa_hd256_f32"),
        bindings: FLASH_DECODE_BINDINGS,
        serves: serves_flash_decode,
        blocks: flash_decode_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "paged_flash_decode_gqa_hd256#GRP=6#kv=bf16",
        target: Target::Named("paged_flash_decode_gqa_hd256_bf16"),
        bindings: FLASH_DECODE_BINDINGS,
        serves: serves_flash_decode,
        blocks: flash_decode_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "paged_flash_decode_gqa_hd256#GRP=6#kv=fp8",
        target: Target::Named("paged_flash_decode_gqa_hd256_kvfp8"),
        bindings: FLASH_DECODE_KV8_BINDINGS,
        serves: serves_flash_decode,
        blocks: flash_decode_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "paged_flash_decode_gqa_combine",
        target: Target::Named("paged_flash_decode_gqa_combine"),
        bindings: COPY_BINDINGS,
        serves: serves_flash_combine,
        blocks: flash_combine_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The int8 decode linear is a CANDIDATE SET, not one redirect: every row below that serves a dispatch is a
    // candidate, `crate::native_tune` times them per device and shape class and `serving` takes its choice. The
    // first row is the untuned default - the dp4a/tensor-core GEMV, valid on every device that has `__dp4a`.
    // Each states the compute capability it needs through its registry entry's floor (`Target::Named` resolves
    // only at or above it) and the row counts and `K` multiples it serves through `serves`.
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_gemv"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_gemv,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: true,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_gemv_wide"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_gemv_wide,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: true,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_mma_m16"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_mma_m16,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_mma_m32"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_mma_m32,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The weight-streaming candidates (cp.async ring, ldmatrix, K split over a cluster of 1, 2, 4 or 8 blocks).
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_stream_s1"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream,
        blocks: i8_stream_blocks::<1>,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(1, 64, 32, 5),
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_stream_s2"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream,
        blocks: i8_stream_blocks::<2>,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(2, 64, 32, 5),
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_stream_s4"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream,
        blocks: i8_stream_blocks::<4>,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(4, 64, 32, 5),
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_stream_s8"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream,
        blocks: i8_stream_blocks::<8>,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(8, 64, 32, 5),
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_stream_w1"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream,
        blocks: i8_stream_blocks::<1>,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(1, 128, 32, 4),
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_stream_w2"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream,
        blocks: i8_stream_blocks::<2>,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(2, 128, 32, 4),
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_stream_w4"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream,
        blocks: i8_stream_blocks::<4>,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(4, 128, 32, 4),
    },
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Named("matmul_i8_stream_w8"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_stream,
        blocks: i8_stream_blocks::<8>,
        requires_wgsl_upgrade: true,
        align_words: 4,
        input: Input::PACKED,
        launch: i8_stream_launch(8, 128, 32, 4),
    },
    // The q4 (W4A8) decode GEMV: the int8 row's structure over 4-bit weights,
    // bit-identical to `matmul_q4_gemv_reg` (`tests/q4_gemv_native.rs`).
    Row {
        slow: "matmul_q4_gemv",
        target: Target::Operator { op: Op::MatMul, weight: Dtype::Q4 },
        bindings: I8_GEMV_BINDINGS,
        serves: serves_q4_gemv,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: true,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The affine K-quant (Q4_K / Q5_K) decode GEMVs. They round once per
    // 32-element group where the WGSL tier rounds per quad, so they are held
    // to the fp32 summation bound against an f64 oracle rather than to the
    // WGSL tier's bits (`tests/kq_native.rs`). No WGSL upgrade precedes
    // them: the WGSL kernel is already the GEMV. Few rows take the block with
    // one weight row a thread; batched rows the one with four
    // ([`KQ_GEMV_WIDE_MIN_ROWS`]).
    Row {
        slow: "matmul_kq_gemv#CODE_BITS=4",
        target: Target::Operator { op: Op::MatMul, weight: Dtype::Q4K },
        bindings: KQ_GEMV_BINDINGS,
        serves: serves_kq_gemv_narrow,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_kq_gemv#CODE_BITS=4",
        target: Target::Named("matmul_kq4_gemv_wide"),
        bindings: KQ_GEMV_BINDINGS,
        serves: serves_kq_gemv_wide,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_kq_gemv#CODE_BITS=8",
        target: Target::Operator { op: Op::MatMul, weight: Dtype::Q8K },
        bindings: KQ_GEMV_BINDINGS,
        serves: serves_kq_gemv_narrow,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_kq_gemv#CODE_BITS=8",
        target: Target::Named("matmul_kq8_gemv_wide"),
        bindings: KQ_GEMV_BINDINGS,
        serves: serves_kq_gemv_wide,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The prefill/DiT GEMM, at every shape the WGSL kernel serves. Its
    // grouped int32 sums are exact and folded in the reference's order with
    // the reference's two roundings, so this row is bit-identical too
    // (`tests/i8_dyn_native.rs`).
    Row {
        slow: "matmul_i8_dyn",
        target: Target::Named("matmul_i8_dp4a"),
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_dyn,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The affine K-quant prefill GEMMs: the int8 GEMM's block over K-quant
    // codes, held to the fp32 summation bound against an f64 oracle rather
    // than to the WGSL tier's bits (`tests/kq_native.rs`).
    Row {
        slow: "matmul_kq_dyn#CODE_BITS=4",
        target: Target::Named("matmul_kq4_dp4a"),
        bindings: KQ_GEMV_BINDINGS,
        serves: serves_kq_dyn,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "matmul_kq_dyn#CODE_BITS=8",
        target: Target::Named("matmul_kq8_dp4a"),
        bindings: KQ_GEMV_BINDINGS,
        serves: serves_kq_dyn,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The plain one-thread-per-output fp32 GEMM: what a small output (a ViT tile's 101 rows, a conditioning row)
    // gets from `model::block::pick_gemm`, and ~100x slower than the register-tiled kernel on a CUDA device. The
    // same arithmetic (one fp32 accumulator per output, product and sum each rounded, k ascending), so the same
    // native kernel serves it bit for bit (`tests/f32_reg_native.rs`).
    Row {
        slow: "matmul",
        target: Target::Named("matmul_f32_reg"),
        bindings: F32_GEMM_BINDINGS,
        serves: serves_f32_gemm,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The register-tiled fp32 GEMM half the workspace dispatches. One fp32
    // accumulator per output, product and sum each rounded, k ascending: the
    // reference's own arithmetic, so bit-identical (`tests/f32_reg_native.rs`).
    Row {
        slow: "matmul_reg3",
        target: Target::Named("matmul_f32_reg"),
        bindings: F32_GEMM_BINDINGS,
        serves: serves_f32_gemm,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // Its input gradient: `Params { m, k, n, accumulate }`, the output `m x k`.
    Row {
        slow: "matmul_dx_reg",
        target: Target::Named("matmul_f32_dx_reg"),
        bindings: F32_GEMM_BINDINGS,
        serves: serves_f32_dx,
        blocks: dx_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The accumulating weight gradient: `Params { m, k, n }`, the output `n x k`.
    Row {
        slow: "matmul_dw_reg",
        target: Target::Named("matmul_f32_dw_reg"),
        bindings: F32_GEMM_BINDINGS,
        serves: serves_f32_dw,
        blocks: dw_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    Row {
        slow: "moe_i8_grouped",
        target: Target::Operator { op: Op::MoeExpertLinear, weight: Dtype::I8 },
        bindings: MOE_I8_GROUPED_BINDINGS,
        serves: serves_moe_i8_grouped,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The 3x3 / stride-1 / pad-1 forward of a layer at least one 64-channel
    // tile wide, as a direct convolution: the same bits as the implicit GEMM
    // rows below (same reduction order, same fused multiply-adds), which
    // serve every other shape. These come FIRST: `apply` takes the first row
    // of a slot that serves the dispatch.
    Row { slow: "conv2d", target: Target::Named("conv2d_fwd3x3_f32"), bindings: CONV2D_BINDINGS, serves: serves_conv3x3, blocks: conv3x3_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "conv_bias", target: Target::Named("conv2d_bias_fwd3x3_f32"), bindings: CONV_BIAS_BINDINGS, serves: serves_conv3x3, blocks: conv3x3_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "conv_bias_reg", target: Target::Named("conv2d_bias_fwd3x3_f32"), bindings: CONV_BIAS_BINDINGS, serves: serves_conv3x3, blocks: conv3x3_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    // The dense convolution and its input gradient. Unlike the two rows above
    // these do NOT reproduce the WGSL kernel's bits: they sum the same
    // reduction in the same order with a fused multiply-add, one rounding
    // where the reference has two. Their gate (`tests/conv2d_native.rs`)
    // holds them to the fp32 summation-order bound instead of to raw bits.
    Row { slow: "conv2d", target: Target::Named("conv2d_fwd_f32"), bindings: CONV2D_BINDINGS, serves: serves_conv, blocks: conv2d_fwd_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "conv2d_dx", target: Target::Named("conv2d_dx_f32"), bindings: CONV2D_BINDINGS, serves: serves_conv, blocks: conv2d_dx_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    // The biased forward, and its register-tiled WGSL sibling (same uniform,
    // same bindings - only the WGSL dispatch geometry differs, and the native
    // kernel derives its own block count from the uniform).
    Row { slow: "conv_bias", target: Target::Named("conv2d_bias_fwd_f32"), bindings: CONV_BIAS_BINDINGS, serves: serves_conv, blocks: conv2d_fwd_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "conv_bias_reg", target: Target::Named("conv2d_bias_fwd_f32"), bindings: CONV_BIAS_BINDINGS, serves: serves_conv, blocks: conv2d_fwd_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    // BatchNorm's per-channel reductions: one block per channel instead of
    // one thread. Re-associated sums, gated like the conv rows
    // (`tests/bn_native.rs`).
    Row { slow: "bn_stats", target: Target::Named("bn_stats_f32"), bindings: BN_STATS_BINDINGS, serves: serves_bn, blocks: bn_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "bn_dstats", target: Target::Named("bn_dstats_f32"), bindings: BN_FOUR_BINDINGS, serves: serves_bn, blocks: bn_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "bn_dgamma", target: Target::Named("bn_dgamma_f32"), bindings: BN_FOUR_BINDINGS, serves: serves_bn, blocks: bn_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "bn_dbeta", target: Target::Named("bn_dbeta_f32"), bindings: BN_DBETA_BINDINGS, serves: serves_bn, blocks: bn_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    // BatchNorm's elementwise passes: the same arithmetic per element as the
    // WGSL kernels, so these ARE bit-identical, and gated so.
    Row { slow: "bn_train", target: Target::Named("bn_train_f32"), bindings: BN_FOUR_BINDINGS, serves: serves_bn, blocks: bn_elem_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "bn_dx", target: Target::Named("bn_dx_f32"), bindings: BN_FOUR_BINDINGS, serves: serves_bn, blocks: bn_elem_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    // GroupNorm's apply pass, plane-blocked like BatchNorm's: the same
    // arithmetic per element, bit-identical (`tests/gn_apply_native.rs`).
    Row { slow: "gn_apply", target: Target::Named("gn_apply_f32"), bindings: BN_FOUR_BINDINGS, serves: serves_gn, blocks: bn_elem_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    // The row softmax with each lane's columns in registers: the WGSL
    // kernel's partition and fold order, bit-identical
    // (`tests/softmax_rows_native.rs`). One row per register bucket, the
    // smallest first - `apply` takes the first that serves the row length.
    // The cross-entropy backward's per-row statistics: a block per row. The sum is
    // reassociated, so it is held to the fp32 summation bound (`tests/ce_stats_native.rs`).
    // Gated DeltaNet's decay-scale gradient: a block per head (`tests/gdn_dscale_native.rs`).
    Row { slow: "gdn_state_decay_bwd_dscale", target: Target::Named("gdn_state_decay_bwd_dscale_f32"), bindings: DSCALE_BINDINGS, serves: serves_dscale, blocks: dscale_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "ce_stats", target: Target::Named("ce_stats_f32"), bindings: CE_STATS_BINDINGS, serves: serves_ce_stats, blocks: ce_stats_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "softmax_rows", target: Target::Named("softmax_rows_f32_k8"), bindings: COPY_BINDINGS, serves: serves_softmax_rows_k8, blocks: softmax_rows_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "softmax_rows", target: Target::Named("softmax_rows_f32_k16"), bindings: COPY_BINDINGS, serves: serves_softmax_rows_k16, blocks: softmax_rows_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "softmax_rows", target: Target::Named("softmax_rows_f32_k32"), bindings: COPY_BINDINGS, serves: serves_softmax_rows_k32, blocks: softmax_rows_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "softmax_rows", target: Target::Named("softmax_rows_f32_k64"), bindings: COPY_BINDINGS, serves: serves_softmax_rows_k64, blocks: softmax_rows_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    // The CSP plumbing and the SiLU pair: copies and elementwise expressions,
    // bit-identical, gated so (`tests/plumb_native.rs`).
    Row { slow: "concat_split", target: Target::Named("concat_split_f32"), bindings: COPY_BINDINGS, serves: serves_chan_window, blocks: chan_window_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "chan_place", target: Target::Named("chan_place_f32"), bindings: COPY_BINDINGS, serves: serves_chan_window, blocks: chan_window_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "concat2", target: Target::Named("concat2_f32"), bindings: CONCAT2_BINDINGS, serves: serves_concat2, blocks: concat2_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    // The transformer glue: bias add, residual add and the erf GELU, four elements a thread, bit-identical
    // (`tests/elementwise_native.rs`).
    Row { slow: "bias_add", target: Target::Named("bias_add_f32"), bindings: BIAS_ADD_BINDINGS, serves: serves_bias_add, blocks: bias_add_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "add2", target: Target::Named("add2_f32"), bindings: CONCAT2_BINDINGS, serves: serves_flat, blocks: flat_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "gelu_erf", target: Target::Named("gelu_erf_f32"), bindings: COPY_BINDINGS, serves: serves_flat, blocks: flat_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "silu", target: Target::Named("silu_f32"), bindings: COPY_BINDINGS, serves: serves_flat, blocks: flat_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    Row { slow: "silu_bwd", target: Target::Named("silu_bwd_f32"), bindings: CONCAT2_BINDINGS, serves: serves_flat, blocks: flat_blocks, requires_wgsl_upgrade: false, align_words: 1, input: Input::PACKED, launch: CudaLaunch::NONE },
    // Bidirectional flash attention at head_dim 128. A reordered sum, so not
    // bit-identical: the gate holds both tiers to one stated tolerance
    // against an f64 oracle (`tests/flash_bidir_native.rs`).
    Row {
        slow: "flash_attn_bidir_reg2",
        target: Target::Named("flash_bidir_f32"),
        bindings: FLASH_BIDIR_BINDINGS,
        serves: serves_flash_bidir,
        blocks: flash_bidir_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // The same at head_dim 64 (CLIP) on fp16 tensor cores: fp16 operands, fp32 accumulation, held to its own stated
    // tolerance (`tests/flash_attn_f16_native.rs`).
    Row {
        slow: "flash_attn_bidir_reg2",
        target: Target::Named("flash_bidir_f16_hd64"),
        bindings: FLASH_BIDIR_BINDINGS,
        serves: serves_flash_bidir_hd64,
        blocks: flash_bidir_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
    // SAM's relative-position-bias attention (windowed and global spans) at head_dim 64 on fp16 tensor cores.
    Row {
        slow: "flash_attn_cross_relpos",
        target: Target::Named("flash_cross_relpos_f16_hd64"),
        bindings: FLASH_CROSS_RELPOS_BINDINGS,
        serves: serves_flash_cross_relpos_hd64,
        blocks: flash_cross_relpos_blocks,
        requires_wgsl_upgrade: false,
        align_words: 1,
        input: Input::PACKED,
        launch: CudaLaunch::NONE,
    },
];

/// `flash_attn_cross_relpos.wgsl`'s bindings: params, `q`, `k`, `v`, `rel_h`, `rel_w`, the written `out`.
const FLASH_CROSS_RELPOS_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// The head width the fp16 tensor-core flash kernels are compiled for.
const FLASH_F16_HEAD_DIM: u32 = 64;

/// Query rows one block of the fp16 flash kernels covers - their registry tile.
const FLASH_F16_ROWS: u32 = 64;

/// The packed-slab params of `serves_flash_bidir`, at head_dim 64, with every row start and head offset a whole
/// number of 16-byte vectors (the kernel loads rows as float4).
fn serves_flash_bidir_hd64(p: &[u32]) -> bool {
    matches!(p, [bsz, heads, t, hd, stride, q_off, k_off, v_off, d_model]
        if *bsz >= 1 && *heads >= 1 && *t >= 1 && *hd == FLASH_F16_HEAD_DIM
        && [*stride, *q_off, *k_off, *v_off, *d_model].iter().all(|w| w % 4 == 0)
        && u64::from(*bsz) * u64::from(*heads) * u64::from(t.div_ceil(FLASH_F16_ROWS)) <= u64::from(u32::MAX))
}

/// `[bsz, n_heads, t_dec, t_enc, head_dim, q_stride, q_off, k_stride, k_off, v_stride, v_off, d_model, kh_ext, kw_ext]`
/// at head_dim 64 with 16-byte aligned rows and a key grid that covers the keys.
fn serves_flash_cross_relpos_hd64(p: &[u32]) -> bool {
    matches!(p, [bsz, heads, t_dec, t_enc, hd, qs, qo, ks, ko, vs, vo, d_model, kh, kw]
        if *bsz >= 1 && *heads >= 1 && *t_dec >= 1 && *t_enc >= 1 && *hd == FLASH_F16_HEAD_DIM && *kh >= 1 && *kw >= 1
        && u64::from(*kh) * u64::from(*kw) >= u64::from(*t_enc)
        && [*qs, *qo, *ks, *ko, *vs, *vo, *d_model].iter().all(|w| w % 4 == 0)
        && u64::from(*bsz) * u64::from(*heads) * u64::from(t_dec.div_ceil(FLASH_F16_ROWS)) <= u64::from(u32::MAX))
}

/// One block per `tile.0` query rows of one (span, head).
fn flash_cross_relpos_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[0] * p[1] * p[2].div_ceil(tile.0)
}

/// The flash attention family's bindings: params, the packed `qkv` slab, the
/// written `out`.
const FLASH_BIDIR_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite];

/// The head width the native flash kernel is compiled for.
const FLASH_BIDIR_HEAD_DIM: u32 = 128;

/// Query rows one block of the native flash kernel covers - its registry
/// tile, which a unit test holds this to.
const FLASH_BIDIR_ROWS: u32 = 64;

/// `[bsz, n_heads, T, head_dim, qkv_stride, q_off, k_off, v_off, d_model]`
/// with something to attend, at the one head width the kernel serves, and a
/// grid the 32-bit block index covers.
fn serves_flash_bidir(p: &[u32]) -> bool {
    matches!(p, [bsz, heads, t, hd, _, _, _, _, _] if *bsz >= 1 && *heads >= 1 && *t >= 1 && *hd == FLASH_BIDIR_HEAD_DIM
        && u64::from(*bsz) * u64::from(*heads) * u64::from(t.div_ceil(FLASH_BIDIR_ROWS)) <= u64::from(u32::MAX))
}

/// One block per `tile.0` query rows of one (sample, head).
fn flash_bidir_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[0] * p[1] * p[2].div_ceil(tile.0)
}

/// One read operand, one written output (`concat_split`, `chan_place`, `silu`).
const COPY_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite];

/// Two read operands, one written output (`concat2`, `silu_bwd`).
const CONCAT2_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `ce_stats.wgsl`'s bindings: params, logits, targets, the written statistics.
const CE_STATS_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `gdn_state_decay_bwd_dscale.wgsl`'s bindings: params, `d_state`, `state_in`, `g_cs`, the accumulated `d_g_cs`.
const DSCALE_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `[bh, dk, dv, c_len, g_cs_off, state_off]`, every extent non-zero.
fn serves_dscale(p: &[u32]) -> bool {
    matches!(p, [bh, dk, dv, c, _, _] if *bh >= 1 && *dk >= 1 && *dv >= 1 && *c >= 1)
}

/// A block per head.
fn dscale_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[0]
}

/// `[n_rows, u_bins, ignore]`, both extents non-zero.
fn serves_ce_stats(p: &[u32]) -> bool {
    matches!(p, [rows, bins, _] if *rows >= 1 && *bins >= 1 && u64::from(*rows) * u64::from(*bins) <= u64::from(u32::MAX))
}

/// A block per row.
fn ce_stats_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[0]
}

/// `[N, Ctot, Csrc, c_off, H, W]` with the window inside the wider map.
fn serves_chan_window(p: &[u32]) -> bool {
    matches!(p, [n, ctot, csrc, off, h, w] if *n >= 1 && *csrc >= 1 && *h >= 1 && *w >= 1
        && u64::from(*off) + u64::from(*csrc) <= u64::from(*ctot)
        && u64::from(*n) * u64::from(*ctot) * u64::from(*h) * u64::from(*w) <= u64::from(u32::MAX))
}

/// A block per run of `tile.1` floats of one window plane, plane-major (or per
/// run of that many consecutive elements when the planes are tiny).
fn chan_window_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    let hw = p[4] * p[5];
    if hw < SMALL_PLANE {
        return (p[0] * p[2] * hw).div_ceil(tile.1);
    }
    p[0] * p[2] * hw.div_ceil(tile.1)
}

/// Planes of fewer floats than this are copied flat, a block per run of
/// `tile.1` consecutive elements across planes (the kernels do the same
/// split on the same bound).
const SMALL_PLANE: u32 = 32;

/// `[N, Ca, Cb, H, W]`, both inputs non-empty.
fn serves_concat2(p: &[u32]) -> bool {
    matches!(p, [n, ca, cb, h, w] if *n >= 1 && *ca >= 1 && *cb >= 1 && *h >= 1 && *w >= 1
        && u64::from(*n) * (u64::from(*ca) + u64::from(*cb)) * u64::from(*h) * u64::from(*w) <= u64::from(u32::MAX))
}

/// A block per run of `tile.1` floats of one output plane, plane-major.
fn concat2_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    let hw = p[3] * p[4];
    if hw < SMALL_PLANE {
        return (p[0] * (p[1] + p[2]) * hw).div_ceil(tile.1);
    }
    p[0] * (p[1] + p[2]) * hw.div_ceil(tile.1)
}

/// `[total]`, non-empty.
fn serves_flat(p: &[u32]) -> bool {
    matches!(p, [total] if *total >= 1)
}

/// `bias_add.wgsl`'s bindings: params, the biased `out` (read and written), the `bias` row.
const BIAS_ADD_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageReadWrite, BindKind::StorageRead];

/// `[m, n]` with something to add; the whole product must index with 32 bits.
fn serves_bias_add(p: &[u32]) -> bool {
    matches!(p, [m, n] if *m >= 1 && *n >= 1 && u64::from(*m) * u64::from(*n) <= u64::from(u32::MAX))
}

/// A block per run of `tile.1` elements of the `m * n` array.
fn bias_add_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    (p[0] * p[1]).div_ceil(tile.1)
}

/// A block per run of `tile.1` elements.
fn flat_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[0].div_ceil(tile.1)
}

/// `brain_bn_train`/`brain_bn_dx`: one block per run of `tile.1` elements of
/// one `(n, c)` plane - the kernels number them plane-major.
fn bn_elem_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[0] * p[1] * (p[2] * p[3]).div_ceil(tile.1)
}

/// `bn_stats.wgsl`'s bindings: params, `x`, then the written `mean` and `var`.
const BN_STATS_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite, BindKind::StorageReadWrite];

/// `bn_dstats.wgsl`'s and `bn_dgamma.wgsl`'s bindings: params, three read
/// operands, one written output.
const BN_FOUR_BINDINGS: &[BindKind] =
    &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `bn_dbeta.wgsl`'s bindings: params, `dy`, the accumulated `dbeta`.
const BN_DBETA_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite];

/// The BatchNorm uniform `[N, C, H, W]` with every extent non-zero.
fn serves_bn(p: &[u32]) -> bool {
    matches!(p, [n, c, h, w] if *n >= 1 && *c >= 1 && *h >= 1 && *w >= 1)
        && p.iter().map(|&v| u64::from(v)).product::<u64>() <= u64::from(u32::MAX)
}

/// The GroupNorm uniform `[N, C, H, W, G]`: every extent non-zero and the
/// channels a whole number of groups.
fn serves_gn(p: &[u32]) -> bool {
    matches!(p, [n, c, h, w, g] if *n >= 1 && *c >= 1 && *h >= 1 && *w >= 1 && *g >= 1 && c % g == 0)
        && p[..4].iter().map(|&v| u64::from(v)).product::<u64>() <= u64::from(u32::MAX)
}

/// Lanes a row of the native row softmax spreads over, and rows per block
/// (`BRAIN_SMR_LANES`, `BRAIN_SMR_ROWS` in `cu/softmax_rows_f32.cu`).
const SOFTMAX_ROWS_LANES: u32 = 64;
const SOFTMAX_ROWS_PER_BLOCK: u32 = 4;

/// `[rows, cols]` with `cols` in the register bucket `(lo, hi]` of `kmax`
/// values a lane.
fn serves_softmax_rows_bucket(p: &[u32], kmax: u32) -> bool {
    let (lo, hi) = ((kmax / 2) * SOFTMAX_ROWS_LANES, kmax * SOFTMAX_ROWS_LANES);
    let lo = if kmax == 8 { 0 } else { lo };
    matches!(p, [rows, cols] if *rows >= 1 && *cols > lo && *cols <= hi
        && u64::from(*rows) * u64::from(*cols) <= u64::from(u32::MAX))
}
fn serves_softmax_rows_k8(p: &[u32]) -> bool {
    serves_softmax_rows_bucket(p, 8)
}
fn serves_softmax_rows_k16(p: &[u32]) -> bool {
    serves_softmax_rows_bucket(p, 16)
}
fn serves_softmax_rows_k32(p: &[u32]) -> bool {
    serves_softmax_rows_bucket(p, 32)
}
fn serves_softmax_rows_k64(p: &[u32]) -> bool {
    serves_softmax_rows_bucket(p, 64)
}

/// Four rows per block.
fn softmax_rows_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[0].div_ceil(SOFTMAX_ROWS_PER_BLOCK)
}

/// One block per channel./// One block per channel.
fn bn_blocks(p: &[u32], _tile: (u32, u32)) -> u32 {
    p[1]
}

/// A native kernel with no WGSL twin: it fuses a chain of dispatches the MODEL
/// builds, so there is nothing to redirect and a model asks for it by name -
/// [`crate::Gpu::fused_step`] - and keeps its own chain of WGSL dispatches for
/// every device that is not offered it.
///
/// The conditions are [`resolve`]'s, minus the first (there is no WGSL upgrade
/// to be active): a queried compute capability at or above the kernel's own
/// floor, a backend that accepts the source, and native kernels not withheld
/// ([`set_native_kernels`]). A shape outside [`Fused::serves`] is the caller's to route to its
/// WGSL chain, exactly like a redirected dispatch the native kernel declines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fused {
    /// `add_rms_quant`: residual add + RMSNorm + per-row int8 scale + pack.
    /// Params `[d, rows, eps (f32 bits), flags]` (flags bit 0: add `b` first);
    /// bindings `a, b, w` read, `sum, xn, xq, sx` written.
    AddRmsQuant,
    /// `quant_epilogue`: an elementwise producer (`mode` 0 plain, 1 `silu(a) *
    /// b`, 2 `a * sigmoid(b)`) + per-row int8 scale + pack. Params `[k, rows,
    /// mode, 0]`; bindings `a, b` read, `y, xq, sx` written.
    QuantEpilogue,
    /// `gdn_decode`: `rows` consecutive Gated DeltaNet decode steps (conv, SiLU,
    /// L2 norm, gates, delta-rule state update, gated RMSNorm) of ONE sequence
    /// with 128-wide key and value heads and a 4-tap conv, taken in order in one
    /// launch. Params `[nkh, nvh, group, l2_eps, rms_eps, q_scale, rows, 0]`
    /// (floats as bits); `mixed`, `bproj`, `aproj`, `z` and `gated` carry one
    /// row per step. Bindings `mixed,
    /// conv_w, hist (RW), bproj, aproj, a_log, dt_bias, state (RW), z, norm_w`
    /// read, `gated` written. The head and conv shapes are the CALLER's to
    /// check - the params cannot say them.
    GdnDecode,
    /// `gdn_decode_pool`: [`Fused::GdnDecode`] for a BATCH of sequences whose
    /// recurrent state and conv window are rows of two pools: block (key head,
    /// batch row) runs the single-sequence body for one step on that row's
    /// inputs and pool row `rows[bi]`, updating the pools in place - no state is
    /// staged in or out. Params `[nkh, nvh, group, l2_eps, rms_eps, q_scale, b,
    /// 0]` (index 6 is the batch, not a step count); bindings as
    /// [`Fused::GdnDecode`] (`mixed`, `bproj`, `aproj`, `z`, `gated` carry a row
    /// per sequence; `hist` and `state` are the pools) plus `rows` read, last.
    GdnDecodePool,
    /// `gdn_decode_pool_bf16`: [`Fused::GdnDecodePool`] over pools that store the state and the conv window as bf16
    /// (`unsigned short` elements: widened on load, rounded to nearest even on store); the same params and bindings.
    GdnDecodePoolBf16,
    /// `gdn_conv_pool` and `gdn_conv_pool_bf16`: the first half of the split pooled decode - the causal conv + SiLU of every
    /// channel of the batch's sequences, windows slid in place in the pool rows, the activated channels written to a scratch
    /// `conv_out` (`[b, conv_dim]`). Params `[nkh, nvh, b]`; bindings `mixed`, `conv_w` read, `hist` (RW), `conv_out` written, `rows` read.
    GdnConvPool,
    GdnConvPoolBf16,
    /// `gdn_recur_pool` and `gdn_recur_pool_bf16`: the second half - per (sequence, value head) the L2 norms, gates, recurrence and
    /// gated norm on the sequence's pool row of `state`. Params as [`Fused::GdnDecodePool`]; bindings `conv_out`, `bproj`, `aproj`,
    /// `a_log`, `dt_bias` read, `state` (RW), `z`, `norm_w` read, `gated` written, `rows` read. Bit-identical to the pair
    /// [`Fused::GdnDecodePool`] replaces, with a block per value head and the conv outside, so many more blocks run at once.
    GdnRecurPool,
    GdnRecurPoolBf16,
    /// `gdn_decode_pool_wide` and `gdn_decode_pool_wide_bf16`: [`Fused::GdnDecodePool`] built for occupancy (a lane pair per state
    /// column, 64 registers a thread), agreeing with the exact body to f32 reassociation rather than bit for bit; at most two
    /// value heads per key head. The same params and bindings.
    GdnDecodePoolWide,
    GdnDecodePoolWideBf16,
    /// `gdn_verify_pool`: [`Fused::GdnDecodePool`] for a speculative verify - each
    /// sequence of the batch takes `counts[bi]` (1..=8) consecutive steps, its
    /// inputs and outputs the rows `offs[bi] ..` of packed token arrays, and the
    /// state and conv window after every row but the last are written to
    /// `snap_state` / `snap_hist` row `offs[bi] + r`. Params as
    /// [`Fused::GdnDecodePool`] (index 6 is the number of sequences); bindings
    /// those of [`Fused::GdnDecodePool`] (through `rows`) plus `counts`, `offs`
    /// read and `snap_state`, `snap_hist` written. The caller guarantees every
    /// count is in 1..=8.
    GdnVerifyPool,
    /// `gdn_commit_pool`: for each sequence with a selected snapshot row, copy
    /// that row of `snap_state` / `snap_hist` over its pool row. Params
    /// `[b, state_len, hist_len, 0]`; bindings `pool_state`, `pool_hist` (RW),
    /// `snap_state`, `snap_hist`, `rows`, `sel` read; `sel` of `u32::MAX` skips
    /// a sequence.
    GdnCommitPool,
    /// `gqa_decode_prep`: for each of `rows` tokens of a gated-attention layer,
    /// the `[value|gate]` split, the per-head QK RMSNorm, the partial rotary
    /// rotation and the K/V append into the paged pools. Params `[nh, nkv,
    /// head_dim, half, eps (f32 bits), block_size, rows, 0]`; bindings `q_full,
    /// k, v, q_norm, k_norm, cos, sin, blocks, offsets` read (all but the two
    /// norm gains carry one row per token), `q_out, q_gate, pool_k, pool_v`
    /// written.
    GqaDecodePrep,
    /// `moe_nvfp4_gateup_silu`: the gate and up projections of an NVFP4 expert bank for the same slots, and `silu(gate) * up`,
    /// in one launch. Params `[slots, k, n, xdiv]`; bindings `x, ids, wg, sg, gg, wu, su, gu` read, `h` written.
    MoeNvfp4GateUp,
    /// `matmul_i8_gemv_multi`: up to four int8 matrices that read ONE packed
    /// activation, multiplied in a single launch - each output bit-identical to
    /// the single `matmul_i8_gemv` on that matrix. Params `[m, kg, n0, n1, n2,
    /// n3]` (`n_i == 0` leaves set `i` unused); bindings `xq`, `sx`, then for
    /// each set its `wq`, `sw` (read) and `out` (written).
    I8GemvMulti,
    /// `matmul_q4_gemv_multi`: [`Fused::I8GemvMulti`] for q4 (W4A8) weights - up
    /// to four q4 matrices that read ONE packed int8 activation, in one launch,
    /// each output bit-identical to the single `matmul_q4_gemv`. Params `[m, k,
    /// n0, n1, n2, n3]` with the RAW `k` (the q4 GEMV's own contract); bindings
    /// as [`Fused::I8GemvMulti`].
    Q4GemvMulti,
    /// `matmul_bf16_gemv_multi`: up to four dense bf16 matrices that read ONE f32 activation (up to eight rows), multiplied
    /// in a single launch - each output bit-identical to the single `matmul_bf16_gemv` on that matrix. Params `[m, k, n0,
    /// n1, n2, n3]` (`n_i == 0` leaves set `i` unused); bindings `x`, then for each set its `w` (read) and `y` (written).
    Bf16GemvMulti,
    /// `conv2d_dw_partial`: the dense conv weight gradient over `S` slices of
    /// `chunk` output positions each. Params: the conv uniform `[N, Cin, H, W,
    /// Cout, K, stride, pad, Ho, Wo]` then `[S, chunk]`; bindings `dy`, `x`
    /// read, `out` written. With `S == 1` `out` IS `dw` and is accumulated
    /// into, exactly like `conv2d_dw`; with `S > 1` it is an `[S, Cout,
    /// Cin*K*K]` scratch [`Fused::Conv2dDwReduce`] folds into `dw`.
    /// [`conv2d_dw_split`] picks `S` and `chunk`.
    Conv2dDwPartial,
    /// `conv2d_dw_reduce`: `dw[i] += part[0][i] + .. + part[S-1][i]`, the
    /// slices added in ascending order so the result does not depend on how
    /// the partial kernel's blocks were scheduled. Params `[total, S]`;
    /// bindings `part` read, `dw` read-written.
    Conv2dDwReduce,
    /// `matmul_i8w_dx`: the input gradient of a linear whose frozen weight is
    /// resident as packed int8, `out[m,k] (+)= sum_n dy[m,n] * q[n,k] *
    /// sw[n,k/32]` - the weight dequantised exactly as it is staged, fp32
    /// fused multiply-adds. Params `[m, k, n, accumulate]` (K a multiple of
    /// 32); bindings `dy`, `wq` (`[n, k/4]` u32), `sw` (`[n, k/32]`) read,
    /// `out` read-written.
    I8wDx,
    /// `matmul_f32_fma`: `matmul_reg3`'s contract (params `[m, k, n]`,
    /// bindings `x`, `w` read, `out` written) with one fused multiply-add per
    /// term - for a trainer, whose gate is its host reference, not bits.
    F32Fma,
    /// `matmul_f32_dx_fma`: `matmul_dx_reg`'s contract (params `[m, k, n,
    /// accumulate]`) with fused multiply-adds.
    F32DxFma,
    /// `matmul_f32_dw_fma`: `matmul_dw_reg`'s contract (params `[m, k, n]`,
    /// always accumulating into the `n x k` output) with fused multiply-adds.
    F32DwFma,
    /// `softmax_rows_dx`: the softmax Jacobian `dx = y * (dy - sum(dy * y))`
    /// over `rows` contiguous rows of `k`, one warp per row. Params `[rows,
    /// k]`; bindings `y`, `dy` read, `dx` written. A reordered reduction of
    /// `softmax_k_dx`'s `M = 1` case - for a trainer, not a redirect.
    SoftmaxRowsDx,
    /// `flash_gqa_kmask_f32`: causal grouped-query attention with an additive
    /// per-key mask - `gqa_scores_kmask`, a row softmax and `gqa_apply` in one
    /// launch, at head_dim 128. Params `[bsz, n_heads, n_kv_heads, T,
    /// head_dim, group]` (the chain's own uniform); bindings `q`, `k`,
    /// `kmask`, `v` read, `out` written. Every query row must see a live key
    /// (key 0 live is the padded encoder's case). A reordered reduction of the
    /// chain, gated against an f64 oracle (`tests/flash_gqa_kmask_native.rs`).
    FlashGqaKmask,
}

/// Output positions one block of [`Fused::Conv2dDwPartial`] stages per
/// iteration - its slices are a whole number of these. `BRAIN_CV_DW_BP` in
/// `cu/conv2d_f32.cu`.
const CONV_DW_STAGE: u32 = 32;

/// Positions below which a slice of the weight-gradient reduction is not
/// worth its own block: each slice costs a full `[Cout, Cin*K*K]` plane of
/// scratch written once and read once by the reduction, so a slice must
/// carry enough multiply-adds to pay for that traffic many times over.
const CONV_DW_MIN_SLICE: u32 = 512;

/// Resident blocks per multiprocessor the slice count aims to fill several
/// waves of: the weight-gradient block's shared memory and registers allow
/// two per multiprocessor on any device whose limits are the portable ones.
const CONV_DW_WAVES_BLOCKS_PER_UNIT: u32 = 8;

/// How the dense conv weight gradient of `params` (the ten-word conv uniform)
/// is split over output positions on a device with `compute_units`
/// multiprocessors: `(S, chunk)`. The output alone is often a handful of
/// blocks (a 16-channel 3x3 layer's gradient is 16 x 144), while the reduction
/// runs over every position of the batch, so the positions are split until
/// the grid fills the device several times over - and no further than
/// [`CONV_DW_MIN_SLICE`] positions a slice. `None` for a shape the native
/// kernel does not serve.
pub fn conv2d_dw_split(params: &[u32], compute_units: u32) -> Option<(u32, u32)> {
    if !serves_conv(params) {
        return None;
    }
    let (n, cin, cout, k, ho, wo) = (params[0], params[1], params[4], params[5], params[8], params[9]);
    let positions = n * ho * wo;
    let tiles = cout.div_ceil(conv_rows_tile(cout)) * (cin * k * k).div_ceil(CONV_DW_COLS);
    let target = compute_units.max(1) * CONV_DW_WAVES_BLOCKS_PER_UNIT;
    let max_splits = (positions / CONV_DW_MIN_SLICE).max(1);
    let splits = target.div_ceil(tiles).clamp(1, max_splits);
    let chunk = positions.div_ceil(splits).div_ceil(CONV_DW_STAGE) * CONV_DW_STAGE;
    // Re-derive the count from the rounded slice so no slice is empty.
    Some((positions.div_ceil(chunk), chunk))
}

/// Weight-gradient columns `(ci, kh, kw)` one block of
/// [`Fused::Conv2dDwPartial`] covers: `BRAIN_CV_DW_BN` in `cu/conv2d_f32.cu`.
const CONV_DW_COLS: u32 = 64;

/// State elements one block of [`Fused::GdnCommitPool`] copies:
/// `BRAIN_GDN_COMMIT_TILE` in `cu/gdn_decode.cu`.
const GDN_COMMIT_TILE: u32 = 4096;

/// Threads per block of [`Fused::Conv2dDwReduce`], one weight element each.
const CONV_DW_REDUCE_BLOCK: u32 = 256;

/// Blocks of one `quant_epilogue` row (`BRAIN_QE_CLUSTER` in its source).
const QUANT_EPILOGUE_CLUSTER: u32 = 4;

/// `add_rms_quant`'s bindings: params, `a`, `b`, `w`, `sum`, `xn`, `xq`, `sx`.
const ADD_RMS_QUANT_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// `quant_epilogue`'s bindings: params, `a`, `b`, `y`, `xq`, `sx`.
const QUANT_EPILOGUE_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// `gdn_decode`'s bindings: params, `mixed`, `conv_w`, `hist`, `bproj`,
/// `aproj`, `a_log`, `dt_bias`, `state`, `z`, `norm_w`, `gated`.
const GDN_DECODE_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `gdn_decode_pool`'s bindings: `gdn_decode`'s, then `rows`.
const GDN_DECODE_POOL_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
];

/// `gdn_conv_pool`'s bindings: params, `mixed`, `conv_w`, `hist` (RW), `conv_out`, `rows`.
const GDN_CONV_POOL_BINDINGS: &[BindKind] =
    &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite, BindKind::StorageReadWrite, BindKind::StorageRead];

/// `gdn_recur_pool`'s bindings: params, `conv_out`, `bproj`, `aproj`, `a_log`, `dt_bias`, `state` (RW), `z`, `norm_w`, `gated`, `rows`.
const GDN_RECUR_POOL_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
];

/// `gqa_decode_prep`'s bindings: params, `q_full`, `k`, `v`, `q_norm`,
/// `k_norm`, `cos`, `sin`, `blocks`, `offsets`, `q_out`, `q_gate`, `pool_k`,
/// `pool_v`.
const GQA_DECODE_PREP_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// `matmul_i8_gemv_multi`'s bindings: params, `xq`, `sx`, then `wq`, `sw`, `out`
/// for each of four sets.
/// `matmul_bf16_gemv_multi`'s bindings: params, `x`, then `w` and `y` of each of four sets.
const BF16_GEMV_MULTI_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

const I8_GEMV_MULTI_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
];

/// `gdn_verify_pool`'s bindings: `gdn_decode_pool`'s, then `counts`, `offs`,
/// `snap_state`, `snap_hist`.
const GDN_VERIFY_POOL_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
];

/// `gdn_commit_pool`'s bindings: params, `pool_state`, `pool_hist`,
/// `snap_state`, `snap_hist`, `rows`, `sel`.
const GDN_COMMIT_POOL_BINDINGS: &[BindKind] = &[
    BindKind::Uniform,
    BindKind::StorageReadWrite,
    BindKind::StorageReadWrite,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
    BindKind::StorageRead,
];

/// `matmul_i8w_dx`'s bindings: params, `dy`, `wq`, `sw`, `out`.
const I8W_DX_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `flash_gqa_kmask_f32`'s bindings: params, `q`, `k`, `kmask`, `v`, `out`.
const FLASH_GQA_KMASK_BINDINGS: &[BindKind] =
    &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `conv2d_dw_partial`'s bindings: params, `dy`, `x`, `out`.
const CONV2D_DW_PARTIAL_BINDINGS: &[BindKind] = CONV2D_BINDINGS;

/// `conv2d_dw_reduce`'s bindings: params, `part`, `dw`.
const CONV2D_DW_REDUCE_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite];

impl Fused {
    /// The `kernels_cuda` registry entry's name.
    pub fn registry_name(self) -> &'static str {
        match self {
            Fused::AddRmsQuant => "add_rms_quant",
            Fused::QuantEpilogue => "quant_epilogue",
            Fused::GdnDecode => "gdn_decode",
            Fused::GdnDecodePool => "gdn_decode_pool",
            Fused::GdnDecodePoolBf16 => "gdn_decode_pool_bf16",
            Fused::GdnConvPool => "gdn_conv_pool",
            Fused::GdnConvPoolBf16 => "gdn_conv_pool_bf16",
            Fused::GdnRecurPool => "gdn_recur_pool",
            Fused::GdnRecurPoolBf16 => "gdn_recur_pool_bf16",
            Fused::GdnDecodePoolWide => "gdn_decode_pool_wide",
            Fused::GdnDecodePoolWideBf16 => "gdn_decode_pool_wide_bf16",
            Fused::GdnVerifyPool => "gdn_verify_pool",
            Fused::GdnCommitPool => "gdn_commit_pool",
            Fused::GqaDecodePrep => "gqa_decode_prep",
            Fused::MoeNvfp4GateUp => "moe_nvfp4_gateup_silu",
            Fused::I8GemvMulti => "matmul_i8_gemv_multi",
            Fused::Q4GemvMulti => "matmul_q4_gemv_multi",
            Fused::Bf16GemvMulti => "matmul_bf16_gemv_multi",
            Fused::Conv2dDwPartial => "conv2d_dw_partial_f32",
            Fused::Conv2dDwReduce => "conv2d_dw_reduce_f32",
            Fused::I8wDx => "matmul_i8w_dx",
            Fused::F32Fma => "matmul_f32_fma",
            Fused::F32DxFma => "matmul_f32_dx_fma",
            Fused::F32DwFma => "matmul_f32_dw_fma",
            Fused::SoftmaxRowsDx => "softmax_rows_dx",
            Fused::FlashGqaKmask => "flash_gqa_kmask_f32",
        }
    }

    fn bindings(self) -> &'static [BindKind] {
        match self {
            Fused::AddRmsQuant => ADD_RMS_QUANT_BINDINGS,
            Fused::QuantEpilogue => QUANT_EPILOGUE_BINDINGS,
            Fused::GdnDecode => GDN_DECODE_BINDINGS,
            Fused::GdnDecodePool | Fused::GdnDecodePoolBf16 | Fused::GdnDecodePoolWide | Fused::GdnDecodePoolWideBf16 => GDN_DECODE_POOL_BINDINGS,
            Fused::GdnConvPool | Fused::GdnConvPoolBf16 => GDN_CONV_POOL_BINDINGS,
            Fused::GdnRecurPool | Fused::GdnRecurPoolBf16 => GDN_RECUR_POOL_BINDINGS,
            Fused::GdnVerifyPool => GDN_VERIFY_POOL_BINDINGS,
            Fused::GdnCommitPool => GDN_COMMIT_POOL_BINDINGS,
            Fused::GqaDecodePrep => GQA_DECODE_PREP_BINDINGS,
            Fused::MoeNvfp4GateUp => MOE_NVFP4_GATEUP_BINDINGS,
            Fused::I8GemvMulti | Fused::Q4GemvMulti => I8_GEMV_MULTI_BINDINGS,
            Fused::Bf16GemvMulti => BF16_GEMV_MULTI_BINDINGS,
            Fused::Conv2dDwPartial => CONV2D_DW_PARTIAL_BINDINGS,
            Fused::Conv2dDwReduce => CONV2D_DW_REDUCE_BINDINGS,
            Fused::I8wDx => I8W_DX_BINDINGS,
            Fused::F32Fma | Fused::F32DxFma | Fused::F32DwFma | Fused::SoftmaxRowsDx => F32_GEMM_BINDINGS,
            Fused::FlashGqaKmask => FLASH_GQA_KMASK_BINDINGS,
        }
    }

    /// Whether the kernel serves a dispatch with these `params`. A `false`
    /// means "use the WGSL chain", never an error.
    pub fn serves(self, params: &[u32]) -> bool {
        match self {
            // One 64-thread block per row keeps `d / 64` elements per thread in
            // registers: up to 80 of them (5120 wide), and a whole number of
            // 32-element scale groups.
            Fused::AddRmsQuant => matches!(params, [d, rows, _, _] if *rows >= 1 && *d >= 32 && d % 32 == 0 && *d <= 64 * 80),
            // 1024 threads per row keep `k / 1024` elements each in registers:
            // up to 18 of them, in whole 32-element scale groups.
            Fused::QuantEpilogue => matches!(params, [k, rows, mode, _] if *rows >= 1 && *k >= 32 && k % 32 == 0 && *k <= 1024 * 18 && *mode <= 2),
            // A block is one key head's `group` value heads, 128 threads each
            // (3 is what one SM's registers hold at 128 live state words).
            Fused::GdnDecode => {
                matches!(params, [nkh, nvh, group, _, _, _, rows, _] if *nkh >= 1 && (1..=3).contains(group) && *nvh == nkh * group && (1..=8).contains(rows))
            }
            Fused::GdnDecodePoolWide | Fused::GdnDecodePoolWideBf16 => {
                matches!(params, [nkh, nvh, group, _, _, _, b, _] if *nkh >= 1 && (1..=2).contains(group) && *nvh == nkh * group && *b >= 1)
            }
            Fused::GdnDecodePool | Fused::GdnDecodePoolBf16 | Fused::GdnVerifyPool | Fused::GdnRecurPool | Fused::GdnRecurPoolBf16 => {
                matches!(params, [nkh, nvh, group, _, _, _, b, _] if *nkh >= 1 && (1..=3).contains(group) && *nvh == nkh * group && *b >= 1)
            }
            Fused::GdnConvPool | Fused::GdnConvPoolBf16 => matches!(params, [nkh, nvh, b] if *nkh >= 1 && *nvh >= *nkh && nvh % nkh == 0 && *b >= 1),
            Fused::GdnCommitPool => matches!(params, [b, state_len, hist_len, _] if *b >= 1 && *state_len >= 4 && state_len % 4 == 0 && hist_len % 4 == 0 && hist_len <= state_len),
            // A head is one 256-thread block holding up to 512 values; the
            // rotated span `2 * half` has to fit inside it.
            Fused::GqaDecodePrep => {
                matches!(params, [nh, nkv, hd, half, _, _, rows, _] if *nh >= 1 && *nkv >= 1 && nh % nkv == 0 && (2..=512).contains(hd) && *half >= 1 && 2 * half <= *hd && *rows >= 1)
            }
            // The single GEMV's own contract for every matrix: one tile of x
            // rows per weight pass and K a whole number of 32-element groups.
            Fused::I8GemvMulti => {
                let rows = kernels_cuda::get("matmul_i8_gemv").map_or(0, |k| k.tile.0);
                matches!(params, [m, kg, n0, ..] if params.len() == 6 && *m >= 1 && *m <= rows && *kg >= 8 && kg % 8 == 0 && *n0 >= 1)
            }
            Fused::Q4GemvMulti => {
                let rows = self.tile().0;
                matches!(params, [m, k, n0, ..] if params.len() == 6 && *m >= 1 && *m <= rows && *k >= 32 && k % 32 == 0 && *n0 >= 1)
            }
            Fused::Bf16GemvMulti => matches!(params, [m, k, n0, ..] if params.len() == 6 && (1..=8).contains(m) && *k >= 8 && k % 8 == 0 && *n0 >= 1),
            // The dense conv uniform plus a split whose slices are whole
            // stages and together cover every position, none of them empty.
            Fused::Conv2dDwPartial => {
                params.len() == 12 && conv_shape(params).is_some() && {
                    let (splits, chunk) = (params[10], params[11]);
                    let positions = params[0] * params[8] * params[9];
                    splits >= 1 && chunk >= CONV_DW_STAGE && chunk % CONV_DW_STAGE == 0 && positions.div_ceil(chunk) == splits
                }
            }
            Fused::Conv2dDwReduce => matches!(params, [total, splits] if *total >= 1 && *splits >= 1),
            Fused::I8wDx => matches!(params, [m, k, n, _] if *m >= 1 && *n >= 1 && *k >= 32 && k % 32 == 0
                && u64::from(*m) * u64::from(*k) <= u64::from(u32::MAX)),
            Fused::F32Fma => serves_f32_gemm(params),
            Fused::F32DxFma => serves_f32_dx(params),
            Fused::F32DwFma => serves_f32_dw(params),
            Fused::SoftmaxRowsDx => {
                matches!(params, [rows, k] if *rows >= 1 && *k >= 1 && u64::from(*rows) * u64::from(*k) <= u64::from(u32::MAX))
            }
            Fused::MoeNvfp4GateUp => serves_moe_nvfp4_gather(params),
            Fused::FlashGqaKmask => matches!(params, [bsz, nh, nkv, t, hd, group]
                if *bsz >= 1 && *nkv >= 1 && *t >= 1 && *hd == FLASH_BIDIR_HEAD_DIM && nh % nkv == 0 && *group == nh / nkv && *group >= 1
                    && u64::from(*bsz) * u64::from(*nh) * u64::from(t.div_ceil(FLASH_BIDIR_ROWS)) <= u64::from(u32::MAX)),
        }
    }

    /// Blocks a dispatch with these `params` launches.
    pub(crate) fn blocks(self, params: &[u32]) -> u32 {
        match self {
            Fused::AddRmsQuant | Fused::QuantEpilogue => params[1],
            Fused::GdnDecode => params[0],
            Fused::GdnDecodePool | Fused::GdnDecodePoolBf16 | Fused::GdnDecodePoolWide | Fused::GdnDecodePoolWideBf16 | Fused::GdnVerifyPool => params[0] * params[6],
            // A block a (sequence, value head), and 256 channels a block.
            Fused::GdnRecurPool | Fused::GdnRecurPoolBf16 => params[1] * params[6],
            Fused::GdnConvPool | Fused::GdnConvPoolBf16 => (params[2] * (2 * params[0] + params[1]) * 128).div_ceil(256),
            Fused::GdnCommitPool => params[0] * params[1].div_ceil(GDN_COMMIT_TILE),
            Fused::GqaDecodePrep => (params[0] + params[1]) * params[6],
            Fused::MoeNvfp4GateUp => params[0] * params[2].div_ceil(QGEMV_COLS),
            Fused::I8GemvMulti | Fused::Q4GemvMulti => {
                let (rows, cols) = self.tile();
                params[0].div_ceil(rows) * params[2..6].iter().map(|n| n.div_ceil(cols)).sum::<u32>()
            }
            // Channel tiles x column tiles x slices; the kernel numbers them
            // channel-tile fastest.
            Fused::Bf16GemvMulti => {
                let cols = if params[0] == 1 { BF16_GEMV_COLS } else { BF16_GEMV_STAGED_COLS };
                params[2..6].iter().map(|n| n.div_ceil(cols)).sum::<u32>()
            }
            Fused::Conv2dDwPartial => {
                let (cin, cout, k) = (params[1], params[4], params[5]);
                cout.div_ceil(conv_rows_tile(cout)) * (cin * k * k).div_ceil(CONV_DW_COLS) * params[10]
            }
            Fused::Conv2dDwReduce => params[0].div_ceil(CONV_DW_REDUCE_BLOCK),
            Fused::I8wDx | Fused::F32DxFma => dx_blocks(params, self.tile()),
            Fused::F32Fma => gemm_blocks(params, self.tile()),
            Fused::F32DwFma => dw_blocks(params, self.tile()),
            Fused::SoftmaxRowsDx => params[0].div_ceil(self.tile().0),
            Fused::FlashGqaKmask => params[0] * params[1] * params[3].div_ceil(self.tile().0),
        }
    }

    /// Whether `which` runs as clusters of [`QUANT_EPILOGUE_CLUSTER`] blocks per row on a device of capability `cc`:
    /// `quant_epilogue` from Hopper, where a block cluster exists (`quant_epilogue_cluster`).
    fn clustered(self, cc: kernels_cuda::Cc) -> bool {
        self == Fused::QuantEpilogue && cc >= kernels_cuda::CLUSTER_MIN_CC
    }

    /// The registry entry this kernel resolves to on a device of capability `cc`.
    fn registry_name_on(self, cc: kernels_cuda::Cc) -> &'static str {
        if self.clustered(cc) { "quant_epilogue_cluster" } else { self.registry_name() }
    }

    /// How it is launched on a device of capability `cc`. A device that cannot launch clusters declines the
    /// registration, and the caller keeps the WGSL chain.
    fn launch_on(self, cc: kernels_cuda::Cc) -> CudaLaunch {
        match self {
            _ if self.clustered(cc) => CudaLaunch { cluster: Some([QUANT_EPILOGUE_CLUSTER, 1, 1]), ..CudaLaunch::NONE },
            // The recurrence stages a head's whole 128 x 128 state in dynamic shared memory (f32, or bf16 at half the bytes).
            Fused::GdnRecurPool => CudaLaunch { dynamic_shared_bytes: kernels_cuda::GDN_RECUR_TILE_F32_BYTES, shared_opt_in: true, ..CudaLaunch::NONE },
            Fused::GdnRecurPoolBf16 => CudaLaunch { dynamic_shared_bytes: kernels_cuda::GDN_RECUR_TILE_F32_BYTES / 2, shared_opt_in: true, ..CudaLaunch::NONE },
            _ => CudaLaunch::NONE,
        }
    }

    /// Blocks a dispatch with these `params` launches on a device of capability `cc` ([`Self::blocks`], times the
    /// cluster where the kernel runs as one).
    pub(crate) fn blocks_on(self, params: &[u32], cc: Option<kernels_cuda::Cc>) -> u32 {
        let per = if cc.is_some_and(|cc| self.clustered(cc)) { QUANT_EPILOGUE_CLUSTER } else { 1 };
        self.blocks(params) * per
    }

    /// Whether a CUDA device of capability `cc` is offered this kernel - what a
    /// memory plan asks before any device is opened.
    pub fn offered_at(self, cc: kernels_cuda::Cc) -> bool {
        !disabled() && kernels_cuda::get(self.registry_name_on(cc)).is_some_and(|k| k.min_cc <= cc)
    }

    /// The registry entry's output tile.
    fn tile(self) -> (u32, u32) {
        kernels_cuda::get(self.registry_name()).map_or((128, 128), |k| k.tile)
    }
}

/// The native id of `which` on this backend, or `None` where it is not
/// offered. Compiles on first use per backend, so a handle caches the answer.
pub(crate) fn resolve_fused(backend: &dyn Backend, which: Fused) -> Option<NativeId> {
    if disabled() {
        return None;
    }
    let cc = backend.caps().arch.compute_capability?;
    if !which.offered_at(cc) {
        return None;
    }
    let k = kernels_cuda::get(which.registry_name_on(cc))?;
    backend.register_native(&NativeSpec::Cuda {
        name: k.reported,
        src: k.src,
        entry: k.entry,
        block_dim: k.block_dim,
        bindings: which.bindings(),
        shared_bytes: k.shared_bytes,
        launch: CudaLaunch { programmatic: k.programmatic, ..which.launch_on(cc) },
    })
}

/// Whether native kernels are withheld - see [`set_native_kernels`].
static WITHHELD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Offer native kernels (`true`, the default) or withhold them process-wide
/// (`false`), so every dispatch stays on the generated WGSL tier. A handle
/// settles which native kernels it uses when it is built and when it first
/// asks for a fused one, so this is set by the APPLICATION before it builds
/// any `Gpu` - the CLI from its `--no-native-kernels` flag, a benchmark that
/// measures the generated tier as its first statement. Never read from the
/// environment.
pub fn set_native_kernels(enabled: bool) {
    WITHHELD.store(!enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Whether native kernels are offered (not withheld by
/// [`set_native_kernels`]). A test that asserts a native kernel ran asks this
/// rather than assuming it.
pub fn native_kernels_enabled() -> bool {
    !disabled()
}

fn disabled() -> bool {
    WITHHELD.load(std::sync::atomic::Ordering::Relaxed)
}

/// Whether a row belongs to a virtual tuning slot (a candidate set no model-registered kernel is redirected to) and which.
#[derive(Clone, Copy)]
enum Virtual {
    No,
    Prefill,
    Fp8,
}

/// One resolved redirect for a handle.
#[derive(Clone, Debug)]
pub(crate) struct Active {
    /// The pipeline slot the model registered and dispatches by index.
    pub slow: usize,
    /// The kernel name the slot was registered under.
    pub slot: &'static str,
    pub id: NativeId,
    /// The registry entry's name, for diagnostics.
    pub kernel: &'static str,
    /// The name the backend times its launches under.
    pub reported: &'static str,
    /// Output tile a block covers, as the registry entry declares it.
    pub tile: (u32, u32),
    pub serves: fn(&[u32]) -> bool,
    pub blocks: fn(&[u32], (u32, u32)) -> u32,
    pub align_words: u64,
    pub input: Input,
}

/// The active native redirects for a handle. Empty on every backend that
/// cannot compile the registry's source and on a device below the kernel's
/// capability floor. `wgsl_upgrades` is [`crate::upgrade::resolve`]'s answer
/// for the same handle (condition 1 in the module doc).
pub(crate) fn resolve(
    names: &[String],
    backend: &dyn Backend,
    wgsl_upgrades: &[crate::upgrade::Active],
) -> Vec<Active> {
    let Some(cc) = backend.caps().arch.compute_capability else {
        return Vec::new();
    };
    let prefill = names.iter().any(|n| n == "matmul_i8_dyn");
    // The FP8 linear's slot exists for a model that registers the FP8 GEMV (every variant carries the base name).
    let fp8 = names.iter().any(|n| n.starts_with("matmul_fp8_gemv"));
    ROWS.iter()
        .filter(|_| !disabled())
        .map(|row| (row, Virtual::No))
        .chain(PREFILL_ROWS.iter().filter(|_| prefill).map(|row| (row, Virtual::Prefill)))
        .chain(FP8_ROWS.iter().filter(|_| fp8 && !disabled()).map(|row| (row, Virtual::Fp8)))
        .filter_map(|(row, virtual_slot)| {
            let slow = match virtual_slot {
                Virtual::No => names.iter().position(|n| n == row.slow)?,
                Virtual::Prefill => PREFILL_KIND,
                Virtual::Fp8 => FP8_KIND,
            };
            if row.requires_wgsl_upgrade {
                wgsl_upgrades.iter().find(|a| a.slow == slow)?;
            }
            let k = row.target.resolve(cc)?;
            let id = backend.register_native(&NativeSpec::Cuda {
                name: k.reported,
                src: k.src,
                entry: k.entry,
                block_dim: k.block_dim,
                bindings: row.bindings,
                shared_bytes: k.shared_bytes,
                launch: CudaLaunch { programmatic: k.programmatic, ..row.launch },
            })?;
            let slot = match virtual_slot {
                Virtual::No => row.slow,
                Virtual::Prefill => PREFILL_SLOT,
                Virtual::Fp8 => FP8_SLOT,
            };
            Some(Active { slow, slot, id, kernel: k.name, reported: k.reported, tile: k.tile, serves: row.serves, blocks: row.blocks, align_words: row.align_words, input: row.input })
        })
        .collect()
}

/// The native pipeline and BLOCK count to dispatch instead of `kind`, or
/// `None` to keep the WGSL tier. `params` is the caller's own uniform; a call
/// without one (`Gpu::step_buf`, whose uniform lives in a caller-owned
/// buffer) is never redirected, exactly as a shape-specialised
/// [`crate::upgrade`] row is not. A slot may have several rows; `serving`
/// says which one a dispatch takes. A redirect keeps the WGSL kernel's own
/// bindings, so it only ever takes a candidate that reads the packed int8
/// activations.
#[inline]
pub(crate) fn apply(active: &[Active], kind: usize, params: &[u32], tuned: Option<&crate::native_tune::Tuned>, offsets: Option<&[(u64, u64)]>) -> Option<(NativeId, u32)> {
    let a = serving(active, kind, params, tuned, offsets, Some(ActFormat::Int8Group32))?;
    Some((a.id, (a.blocks)(params, a.tile)))
}

/// The active row of slot `kind` that serves `params`: the candidate the
/// device's tuned table names for this shape class when it names one this
/// dispatch can run, otherwise the first row that serves (the untuned
/// default, and the answer for every slot with a single candidate).
///
/// `supplies` is the activation form the caller can hand over (`None`: any). Only rows reading it are considered, so
/// a caller holding packed int8 activations never gets a candidate that needs the bf16 rows, and the table's choice
/// among the rest is the fastest candidate that reads that form.
///
/// Cost per dispatch: O(rows of slot `kind`) for the `serves` tests, plus the tuned table's O(1) class lookup
/// ([`crate::native_tune::Tuned::choice`]). Pass `active` as the slot's own rows ([`crate::Gpu`] does) so a
/// dispatch of a kernel with no native row pays one lookup of the slot, not a scan of every row on the handle.
#[inline]
pub(crate) fn serving<'a>(
    active: &'a [Active],
    kind: usize,
    params: &[u32],
    tuned: Option<&crate::native_tune::Tuned>,
    offsets: Option<&[(u64, u64)]>,
    supplies: Option<ActFormat>,
) -> Option<&'a Active> {
    // A candidate that needs aligned bindings is skipped when a sliced binding starts off its alignment.
    let runs = |a: &&Active| offsets.is_none_or(|o| o.iter().all(|&(off, _)| off % a.align_words == 0));
    let reads = |a: &Active| supplies.is_none_or(|f| a.input.act == f);
    let mut serving = active.iter().filter(|a| a.slow == kind && reads(a) && (a.serves)(params)).filter(runs);
    let first = serving.next()?;
    let accepts = |name: &str| active.iter().any(|a| a.slow == kind && a.kernel == name && reads(a));
    let Some(choice) = tuned.and_then(|t| t.choice(first.slot, params, &accepts)) else { return Some(first) };
    if first.kernel == &*choice {
        return Some(first);
    }
    Some(serving.find(|a| a.kernel == &*choice).unwrap_or(first))
}

/// Every candidate row of slot `kind` that serves `params` on this handle, in preference order.
pub(crate) fn candidates<'a>(active: &'a [Active], kind: usize, params: &'a [u32]) -> impl Iterator<Item = &'a Active> {
    active.iter().filter(move |a| a.slow == kind && (a.serves)(params))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every row of a slot is a CANDIDATE the autotuner may choose: it names a registry entry that exists, states the
    /// capability it needs through that entry's floor, and the int8 decode linear offers a dp4a GEMV plus the
    /// tensor-core GEMV and GEMMs, so a device with tensor cores has a choice and one without keeps the GEMV.
    #[test]
    fn the_int8_decode_slot_is_a_candidate_set_with_declared_capabilities() {
        let rows: Vec<&Row> = ROWS.iter().filter(|r| r.slow == "matmul_i8_gemv").collect();
        let names: Vec<&str> = rows
            .iter()
            .map(|r| match r.target {
                Target::Named(n) => n,
                Target::Operator { .. } => panic!("the int8 slot's candidates are named"),
            })
            .collect();
        assert_eq!(names, ["matmul_i8_gemv", "matmul_i8_gemv_wide", "matmul_i8_mma_m16", "matmul_i8_mma_m32", "matmul_i8_stream_s1", "matmul_i8_stream_s2", "matmul_i8_stream_s4", "matmul_i8_stream_s8", "matmul_i8_stream_w1", "matmul_i8_stream_w2", "matmul_i8_stream_w4", "matmul_i8_stream_w8"], "first is the untuned default");
        for r in &rows {
            // The cluster-split streaming candidates need Hopper; every other candidate runs at the tensor-core floor.
            let at = if r.launch.cluster.is_some() { kernels_cuda::CLUSTER_MIN_CC } else { kernels_cuda::MMA_S8_MIN_CC };
            let k = r.target.resolve(at).expect("every candidate exists at its declared floor");
            assert!(k.min_cc <= at);
            if r.launch.cluster.is_none() {
                assert!(r.target.resolve(kernels_cuda::MMA_S8_MIN_CC).is_some());
            }
        }
        // A device below the tensor-core floor is offered the dp4a GEMV alone.
        let offered = rows.iter().filter(|r| r.target.resolve(kernels_cuda::DP4A_MIN_CC).is_some()).count();
        assert_eq!(offered, 1, "only the dp4a GEMV runs below the tensor-core floor");
        // The exact-decode contract: rows the single-row block computes bit for bit are served by it alone.
        for m in 1..kernels_cuda::I8_GEMV_MMA_MIN_ROWS {
            let p = [m, 1280, 1024];
            assert_eq!(rows.iter().filter(|r| (r.serves)(&p)).count(), 1, "m={m}");
        }
        for m in kernels_cuda::I8_GEMV_MMA_MIN_ROWS..=backend_api::select::DECODE_REGIME_MAX_ROWS {
            let p = [m, 1280, 1024];
            assert!(rows.iter().filter(|r| (r.serves)(&p)).count() >= 3, "m={m} should have a choice");
        }
    }

    /// The prefill slot's rows are candidates like the decode slot's: each names a registry entry that exists at
    /// the capability it declares, a device below a candidate's floor is never offered it, the first row that serves a
    /// shape is the untuned default (the routing the provider had before the slot was measured), and the decode
    /// regime - the GEMV slot's - is served by none of them.
    #[test]
    fn the_prefill_slot_is_a_candidate_set_with_declared_capabilities() {
        let name = |r: &Row| match r.target {
            Target::Named(n) => n,
            Target::Operator { .. } => panic!("the prefill candidates are named"),
        };
        let names: Vec<&str> = PREFILL_ROWS.iter().map(name).collect();
        assert_eq!(
            names,
            [
                "matmul_i8_mma_m128",
                "matmul_i8_mma",
                "matmul_i8_dq_wgmma",
                "matmul_i8_dq_stream_q2",
                "matmul_i8_dq_stream_q4",
                "matmul_i8_stream_r1",
                "matmul_i8_stream_r2",
                "matmul_i8_stream_r4",
                "matmul_i8_dq_stream_r1",
                "matmul_i8_dq_stream_r2",
                "matmul_i8_dq_stream_r4",
                "matmul_i8_wgstream_n64_s2",
                "matmul_i8_wgstream_n64_s4",
                "matmul_i8_wgstream_n128_s2",
                "matmul_i8_wgstream_n128_s4",
                "matmul_i8_wgmma"
            ],
            "first is the untuned default"
        );
        let offered = |cc| PREFILL_ROWS.iter().filter(|r| r.target.resolve(cc).is_some()).map(name).collect::<Vec<_>>();
        assert!(offered(kernels_cuda::DP4A_MIN_CC).is_empty(), "no prefill candidate runs below the tensor-core floor");
        assert_eq!(offered(kernels_cuda::MMA_S8_MIN_CC), ["matmul_i8_mma_m128", "matmul_i8_mma", "matmul_i8_stream_r1", "matmul_i8_dq_stream_r1"]);
        assert_eq!(offered(kernels_cuda::WGMMA_MIN_CC), names, "Hopper offers the warpgroup GEMM too");
        // The untuned default is the first row reading the packed int8 activations, which is what the provider holds.
        let first = |p: &[u32]| PREFILL_ROWS.iter().find(|r| r.input == Input::PACKED && (r.serves)(p)).map(name);
        assert_eq!(first(&[33, 1280, 1024]), Some("matmul_i8_mma_m128"));
        assert_eq!(first(&[128, 1280, 1024]), Some("matmul_i8_mma_m128"));
        assert_eq!(first(&[129, 1280, 1024]), Some("matmul_i8_mma"));
        assert_eq!(first(&[4096, 1280, 1024]), Some("matmul_i8_mma"));
        assert_eq!(first(&[32, 1280, 1024]), None, "the decode regime is the GEMV slot's");
        assert_eq!(first(&[256, 1288, 1024]), None, "K must be a whole number of 64-element staged tiles (K = 5152 is not)");
        // The many-row streaming candidates serve the step of a large batch (33-64 or 33-128 rows) and nothing past it.
        for r in PREFILL_ROWS.iter().filter(|r| name(r).contains("stream_")) {
            let top = if name(r).contains("n128") || name(r).contains("dq_stream_q") { 128 } else { kernels_cuda::I8_STREAM_MANY_ROWS };
            let k = if name(r).contains("wgstream") { 1280 } else { 1288 };
            assert!((r.serves)(&[33, k, 1024]) && (r.serves)(&[top, k, 1024]), "{}", name(r));
            assert!(!(r.serves)(&[32, k, 1024]) && !(r.serves)(&[top + 1, k, 1024]), "{}", name(r));
        }
        let wgmma = PREFILL_ROWS.iter().find(|r| name(r) == "matmul_i8_wgmma").expect("the row");
        assert!((wgmma.serves)(&[256, 1280, 1024]));
        assert!(!(wgmma.serves)(&[256, 1296, 1024]), "the warpgroup GEMM wants whole 128-deep k-tiles (K = 5184 is 81 x 64)");
        assert_eq!(wgmma.launch.arch, backend_api::ArchFeatures::Required);
        assert_eq!(wgmma.launch.dynamic_shared_bytes, kernels_cuda::I8_WGMMA_DYNAMIC_SHARED_BYTES);
        for r in PREFILL_ROWS {
            assert_eq!(r.slow, "matmul_i8_dyn", "the rows exist on a handle whose model registers the provider's reference kernel");
            let bindings = if r.input == Input::PACKED { I8_GEMV_BINDINGS } else { I8_DEQUANT_BINDINGS };
            assert_eq!(r.bindings.len(), bindings.len(), "{}: a candidate takes the bindings of the form it reads", name(r));
        }
    }

    /// A candidate reading bf16 rows is bound to the five-binding list without an activation scale; every other
    /// candidate of the int8 slots keeps the WGSL kernel's six. The activation form a caller can supply decides which
    /// the dispatch considers, so a redirect of the WGSL pipeline never lands on a candidate it cannot feed.
    #[test]
    fn a_dispatch_only_considers_candidates_reading_the_form_it_supplies() {
        let rows = || ROWS.iter().chain(PREFILL_ROWS).filter(|r| r.slow == "matmul_i8_dyn" || r.slow == "matmul_i8_gemv");
        for r in rows() {
            let expected = if r.input.act == ActFormat::Bf16 { I8_DEQUANT_BINDINGS.len() } else { I8_GEMV_BINDINGS.len() };
            assert_eq!(r.bindings.len(), expected);
        }
        assert!(rows().any(|r| r.input == Input::DEQUANT_BF16), "the slots offer dequantize-in-register candidates");
        assert!(rows().filter(|r| r.input == Input::DEQUANT_BF16).all(|r| matches!(r.target, Target::Named(n) if n.starts_with("matmul_i8_dq_"))));
    }

    #[test]
    fn the_int8_gemv_row_names_a_registry_entry_with_its_own_bindings() {
        let row = ROWS.iter().find(|r| r.slow == "matmul_i8_gemv").expect("the row");
        let k = kernels_cuda::get("matmul_i8_gemv").expect("registry entry");
        assert_eq!(row.target.resolve(kernels_cuda::DP4A_MIN_CC).map(|r| r.name), Some(k.name));
        // Same binding list as the WGSL kernel's own declaration: one uniform
        // and five storage buffers.
        let wgsl = kernels::MATMUL_I8_GEMV_REG;
        let declared = wgsl.lines().filter(|l| l.trim_start().starts_with("@group(")).count();
        assert_eq!(declared, row.bindings.len());
    }

    #[test]
    fn the_grouped_moe_row_names_a_registry_entry_with_its_own_bindings() {
        let row = ROWS.iter().find(|r| r.slow == "moe_i8_grouped").expect("the row");
        let k = kernels_cuda::get("moe_i8_grouped_mma").expect("registry entry");
        assert_eq!(row.target.resolve(kernels_cuda::MMA_S8_MIN_CC).map(|r| r.name), Some(k.name));
        let declared = kernels::MOE_I8_GROUPED.lines().filter(|l| l.trim_start().starts_with("@group(")).count();
        assert_eq!(declared, row.bindings.len());
        // No WGSL sibling exists to upgrade to, so the row must not wait for one.
        assert!(!row.requires_wgsl_upgrade);
        assert!(serves_moe_i8_grouped(&[40, 512, 2048, 9, 257]));
        assert!(!serves_moe_i8_grouped(&[40, 516, 2048, 9, 257]), "K not a whole number of scale groups");
        assert!(!serves_moe_i8_grouped(&[40, 512, 2048]), "short params");
    }

    #[test]
    fn the_flash_row_agrees_with_its_registry_entry() {
        let row = ROWS.iter().find(|r| r.slow == "flash_attn_bidir_reg2").expect("the row");
        let k = row.target.resolve(kernels_cuda::BASELINE_MIN_CC).expect("registry entry");
        assert_eq!(k.tile.0, FLASH_BIDIR_ROWS);
        let declared = kernels::FLASH_ATTN_BIDIR_REG2.lines().filter(|l| l.trim_start().starts_with("@group(")).count();
        assert_eq!(declared, row.bindings.len());
        assert_eq!(flash_bidir_blocks(&[2, 24, 1792, 128, 9216, 0, 3072, 6144, 3072], k.tile), 2 * 24 * 28);
        assert!(!serves_flash_bidir(&[1, 24, 1792, 64, 9216, 0, 3072, 6144, 3072]), "another head width");
    }

    #[test]
    fn only_shapes_the_kernel_serves_are_redirected() {
        assert!(serves_i8_gemv(&[1, 1280, 17408]));
        assert!(serves_i8_gemv(&[8, 8, 1]));
        assert!(!serves_i8_gemv(&[0, 1280, 64]), "no rows");
        assert!(serves_i8_gemv(&[9, 1280, 64]), "a second tile of x rows is more blocks, not another kernel");
        assert!(serves_i8_gemv(&[32, 1280, 64]));
        assert!(!serves_i8_gemv(&[33, 1280, 64]), "past the decode regime");
        assert!(!serves_i8_gemv(&[1, 1284, 64]), "K not a whole number of scale groups");
        assert!(!serves_i8_gemv(&[1, 1280]), "short params");
    }

    /// The FP8 linear's candidate set: every row is a registered FP8 kernel reading bf16 rows, the decode streaming kernels
    /// serve the rows they are sized for, the Hopper GEMM the rows past the decode regime, and nothing serves a width that is
    /// not a whole number of 128-wide scale blocks.
    #[test]
    fn the_fp8_linear_candidates_serve_the_shapes_they_are_built_for() {
        let name = |r: &Row| match r.target {
            Target::Named(n) => n,
            Target::Operator { .. } => unreachable!("named rows"),
        };
        for r in FP8_ROWS {
            let k = kernels_cuda::get(name(r)).unwrap_or_else(|| panic!("{} is not a registered kernel", name(r)));
            assert_eq!(k.weight, Dtype::F8E4M3, "{}", name(r));
            if r.input == Input::FP8_NATIVE {
                assert_eq!(r.bindings, I8_GEMV_BINDINGS, "{}", name(r));
            } else {
                assert_eq!(r.input, Input::DEQUANT_BF16, "{}", name(r));
                assert_eq!(r.bindings, FP8_BINDINGS);
            }
        }
        let serving = |p: &[u32]| FP8_ROWS.iter().filter(|r| (r.serves)(p)).map(name).collect::<Vec<_>>();
        // Decode of one token over a 5120 x 5120 weight (K = 5120 = 1280 words): the streaming kernels only.
        let one = serving(&[1, 1280, 5120]);
        assert!(!one.is_empty() && one.iter().all(|n| n.contains("stream") || n.contains("gemv_f16")), "{one:?}");
        // The f16 weight streams stop at 16 rows.
        assert!(serving(&[16, 1280, 5120]).iter().any(|n| n.contains("gemv_f16")));
        assert!(!serving(&[17, 1280, 5120]).iter().any(|n| n.contains("gemv_f16")));
        // 100 rows: the wide streaming kernels and the GEMM, not the narrow ones.
        let hundred = serving(&[100, 1280, 5120]);
        assert!(hundred.contains(&"matmul_fp8_dq_stream_q2") && hundred.contains(&"matmul_fp8_dq_wgmma") && !hundred.contains(&"matmul_fp8_dq_stream_r1"), "{hundred:?}");
        // A prefill round: the two GEMMs (bf16 operands decoded in shared memory, E4M3 operands as stored).
        assert_eq!(serving(&[1024, 1280, 17408]), vec!["matmul_fp8_dq_wgmma", "matmul_fp8_wgmma"]);
        // K = 5000 is not whole 128-wide blocks.
        assert!(serving(&[1, 1250, 64]).is_empty());
        // The weight must stay addressable with 32-bit byte offsets.
        assert!(serving(&[1, 1280, 1 << 20]).is_empty());
    }

    /// A window of single-element planes (the channel-window ops over a row-major
    /// matrix, `H = W = 1`) shares blocks across planes instead of launching one
    /// block per element.
    #[test]
    fn single_element_planes_share_blocks() {
        let tile = (1, 1024);
        // 4096 rows of a 20-channel matrix, a 7-channel window.
        assert!(chan_window_blocks(&[4096, 20, 7, 5, 1, 1], tile) <= (4096 * 7u32).div_ceil(1024));
        assert!(concat2_blocks(&[4096, 7, 9, 1, 1], tile) <= (4096 * 16u32).div_ceil(1024));
        // Large planes keep a block per run of one plane.
        assert_eq!(chan_window_blocks(&[8, 64, 32, 0, 64, 64], tile), 8 * 32 * 4);
        assert_eq!(concat2_blocks(&[8, 64, 128, 16, 16], tile), 8 * 192);
    }
}
