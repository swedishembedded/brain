// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Transparent NATIVE kernel upgrades - a hand-written kernel a model inherits
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
//! exact (`tests/conv2d_native.rs`).
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
//! `BRAIN_NO_NATIVE_KERNELS=1` pins every dispatch to the WGSL tier - the A/B
//! switch the native kernel's own measurements are taken with.

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
}

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

pub(crate) const ROWS: &[Row] = &[
    Row {
        slow: "matmul_i8_gemv",
        target: Target::Operator { op: Op::MatMul, weight: Dtype::I8 },
        bindings: I8_GEMV_BINDINGS,
        serves: serves_i8_gemv,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: true,
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
    },
    // Its input gradient: `Params { m, k, n, accumulate }`, the output `m x k`.
    Row {
        slow: "matmul_dx_reg",
        target: Target::Named("matmul_f32_dx_reg"),
        bindings: F32_GEMM_BINDINGS,
        serves: serves_f32_dx,
        blocks: dx_blocks,
        requires_wgsl_upgrade: false,
    },
    // The accumulating weight gradient: `Params { m, k, n }`, the output `n x k`.
    Row {
        slow: "matmul_dw_reg",
        target: Target::Named("matmul_f32_dw_reg"),
        bindings: F32_GEMM_BINDINGS,
        serves: serves_f32_dw,
        blocks: dw_blocks,
        requires_wgsl_upgrade: false,
    },
    Row {
        slow: "moe_i8_grouped",
        target: Target::Operator { op: Op::MoeExpertLinear, weight: Dtype::I8 },
        bindings: MOE_I8_GROUPED_BINDINGS,
        serves: serves_moe_i8_grouped,
        blocks: gemm_blocks,
        requires_wgsl_upgrade: false,
    },
    // The dense convolution and its input gradient. Unlike the two rows above
    // these do NOT reproduce the WGSL kernel's bits: they sum the same
    // reduction in the same order with a fused multiply-add, one rounding
    // where the reference has two. Their gate (`tests/conv2d_native.rs`)
    // holds them to the fp32 summation-order bound instead of to raw bits.
    Row { slow: "conv2d", target: Target::Named("conv2d_fwd_f32"), bindings: CONV2D_BINDINGS, serves: serves_conv, blocks: conv2d_fwd_blocks, requires_wgsl_upgrade: false },
    Row { slow: "conv2d_dx", target: Target::Named("conv2d_dx_f32"), bindings: CONV2D_BINDINGS, serves: serves_conv, blocks: conv2d_dx_blocks, requires_wgsl_upgrade: false },
    // The biased forward, and its register-tiled WGSL sibling (same uniform,
    // same bindings - only the WGSL dispatch geometry differs, and the native
    // kernel derives its own block count from the uniform).
    Row { slow: "conv_bias", target: Target::Named("conv2d_bias_fwd_f32"), bindings: CONV_BIAS_BINDINGS, serves: serves_conv, blocks: conv2d_fwd_blocks, requires_wgsl_upgrade: false },
    Row { slow: "conv_bias_reg", target: Target::Named("conv2d_bias_fwd_f32"), bindings: CONV_BIAS_BINDINGS, serves: serves_conv, blocks: conv2d_fwd_blocks, requires_wgsl_upgrade: false },
    // BatchNorm's per-channel reductions: one block per channel instead of
    // one thread. Re-associated sums, gated like the conv rows
    // (`tests/bn_native.rs`).
    Row { slow: "bn_stats", target: Target::Named("bn_stats_f32"), bindings: BN_STATS_BINDINGS, serves: serves_bn, blocks: bn_blocks, requires_wgsl_upgrade: false },
    Row { slow: "bn_dstats", target: Target::Named("bn_dstats_f32"), bindings: BN_FOUR_BINDINGS, serves: serves_bn, blocks: bn_blocks, requires_wgsl_upgrade: false },
    Row { slow: "bn_dgamma", target: Target::Named("bn_dgamma_f32"), bindings: BN_FOUR_BINDINGS, serves: serves_bn, blocks: bn_blocks, requires_wgsl_upgrade: false },
    Row { slow: "bn_dbeta", target: Target::Named("bn_dbeta_f32"), bindings: BN_DBETA_BINDINGS, serves: serves_bn, blocks: bn_blocks, requires_wgsl_upgrade: false },
    // BatchNorm's elementwise passes: the same arithmetic per element as the
    // WGSL kernels, so these ARE bit-identical, and gated so.
    Row { slow: "bn_train", target: Target::Named("bn_train_f32"), bindings: BN_FOUR_BINDINGS, serves: serves_bn, blocks: bn_elem_blocks, requires_wgsl_upgrade: false },
    Row { slow: "bn_dx", target: Target::Named("bn_dx_f32"), bindings: BN_FOUR_BINDINGS, serves: serves_bn, blocks: bn_elem_blocks, requires_wgsl_upgrade: false },
    // The CSP plumbing and the SiLU pair: copies and elementwise expressions,
    // bit-identical, gated so (`tests/plumb_native.rs`).
    Row { slow: "concat_split", target: Target::Named("concat_split_f32"), bindings: COPY_BINDINGS, serves: serves_chan_window, blocks: chan_window_blocks, requires_wgsl_upgrade: false },
    Row { slow: "chan_place", target: Target::Named("chan_place_f32"), bindings: COPY_BINDINGS, serves: serves_chan_window, blocks: chan_window_blocks, requires_wgsl_upgrade: false },
    Row { slow: "concat2", target: Target::Named("concat2_f32"), bindings: CONCAT2_BINDINGS, serves: serves_concat2, blocks: concat2_blocks, requires_wgsl_upgrade: false },
    Row { slow: "silu", target: Target::Named("silu_f32"), bindings: COPY_BINDINGS, serves: serves_flat, blocks: flat_blocks, requires_wgsl_upgrade: false },
    Row { slow: "silu_bwd", target: Target::Named("silu_bwd_f32"), bindings: CONCAT2_BINDINGS, serves: serves_flat, blocks: flat_blocks, requires_wgsl_upgrade: false },
];

/// One read operand, one written output (`concat_split`, `chan_place`, `silu`).
const COPY_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageReadWrite];

/// Two read operands, one written output (`concat2`, `silu_bwd`).
const CONCAT2_BINDINGS: &[BindKind] = &[BindKind::Uniform, BindKind::StorageRead, BindKind::StorageRead, BindKind::StorageReadWrite];

/// `[N, Ctot, Csrc, c_off, H, W]` with the window inside the wider map.
fn serves_chan_window(p: &[u32]) -> bool {
    matches!(p, [n, ctot, csrc, off, h, w] if *n >= 1 && *csrc >= 1 && *h >= 1 && *w >= 1
        && u64::from(*off) + u64::from(*csrc) <= u64::from(*ctot)
        && u64::from(*n) * u64::from(*ctot) * u64::from(*h) * u64::from(*w) <= u64::from(u32::MAX))
}

/// A block per run of `tile.1` floats of one window plane, plane-major.
fn chan_window_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[0] * p[2] * (p[4] * p[5]).div_ceil(tile.1)
}

/// `[N, Ca, Cb, H, W]`, both inputs non-empty.
fn serves_concat2(p: &[u32]) -> bool {
    matches!(p, [n, ca, cb, h, w] if *n >= 1 && *ca >= 1 && *cb >= 1 && *h >= 1 && *w >= 1
        && u64::from(*n) * (u64::from(*ca) + u64::from(*cb)) * u64::from(*h) * u64::from(*w) <= u64::from(u32::MAX))
}

/// A block per run of `tile.1` floats of one output plane, plane-major.
fn concat2_blocks(p: &[u32], tile: (u32, u32)) -> u32 {
    p[0] * (p[1] + p[2]) * (p[3] * p[4]).div_ceil(tile.1)
}

/// `[total]`, non-empty.
fn serves_flat(p: &[u32]) -> bool {
    matches!(p, [total] if *total >= 1)
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

/// One block per channel.
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
/// floor, a backend that accepts the source, and `BRAIN_NO_NATIVE_KERNELS`
/// unset. A shape outside [`Fused::serves`] is the caller's to route to its
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
    /// `gqa_decode_prep`: for each of `rows` tokens of a gated-attention layer,
    /// the `[value|gate]` split, the per-head QK RMSNorm, the partial rotary
    /// rotation and the K/V append into the paged pools. Params `[nh, nkv,
    /// head_dim, half, eps (f32 bits), block_size, rows, 0]`; bindings `q_full,
    /// k, v, q_norm, k_norm, cos, sin, blocks, offsets` read (all but the two
    /// norm gains carry one row per token), `q_out, q_gate, pool_k, pool_v`
    /// written.
    GqaDecodePrep,
    /// `matmul_i8_gemv_multi`: up to four int8 matrices that read ONE packed
    /// activation, multiplied in a single launch - each output bit-identical to
    /// the single `matmul_i8_gemv` on that matrix. Params `[m, kg, n0, n1, n2,
    /// n3]` (`n_i == 0` leaves set `i` unused); bindings `xq`, `sx`, then for
    /// each set its `wq`, `sw` (read) and `out` (written).
    I8GemvMulti,
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

/// Threads per block of [`Fused::Conv2dDwReduce`], one weight element each.
const CONV_DW_REDUCE_BLOCK: u32 = 256;

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
            Fused::GqaDecodePrep => "gqa_decode_prep",
            Fused::I8GemvMulti => "matmul_i8_gemv_multi",
            Fused::Conv2dDwPartial => "conv2d_dw_partial_f32",
            Fused::Conv2dDwReduce => "conv2d_dw_reduce_f32",
        }
    }

    fn bindings(self) -> &'static [BindKind] {
        match self {
            Fused::AddRmsQuant => ADD_RMS_QUANT_BINDINGS,
            Fused::QuantEpilogue => QUANT_EPILOGUE_BINDINGS,
            Fused::GdnDecode => GDN_DECODE_BINDINGS,
            Fused::GdnDecodePool => GDN_DECODE_POOL_BINDINGS,
            Fused::GqaDecodePrep => GQA_DECODE_PREP_BINDINGS,
            Fused::I8GemvMulti => I8_GEMV_MULTI_BINDINGS,
            Fused::Conv2dDwPartial => CONV2D_DW_PARTIAL_BINDINGS,
            Fused::Conv2dDwReduce => CONV2D_DW_REDUCE_BINDINGS,
        }
    }

    /// Whether the kernel serves a dispatch with these `params`. A `false`
    /// means "use the WGSL chain", never an error.
    pub fn serves(self, params: &[u32]) -> bool {
        match self {
            // One 64-thread block per row keeps `d / 64` elements per thread in
            // registers: up to 80 of them (5120 wide), and a whole number of
            // int8 words.
            Fused::AddRmsQuant => matches!(params, [d, rows, _, _] if *rows >= 1 && *d >= 4 && d % 4 == 0 && *d <= 64 * 80),
            // 1024 threads per row keep `k / 1024` elements each in registers:
            // up to 18 of them.
            Fused::QuantEpilogue => matches!(params, [k, rows, mode, _] if *rows >= 1 && *k >= 4 && k % 4 == 0 && *k <= 1024 * 18 && *mode <= 2),
            // A block is one key head's `group` value heads, 128 threads each
            // (3 is what one SM's registers hold at 128 live state words).
            Fused::GdnDecode => {
                matches!(params, [nkh, nvh, group, _, _, _, rows, _] if *nkh >= 1 && (1..=3).contains(group) && *nvh == nkh * group && (1..=8).contains(rows))
            }
            Fused::GdnDecodePool => matches!(params, [nkh, nvh, group, _, _, _, b, _] if *nkh >= 1 && (1..=3).contains(group) && *nvh == nkh * group && *b >= 1),
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
        }
    }

    /// Blocks a dispatch with these `params` launches.
    pub(crate) fn blocks(self, params: &[u32]) -> u32 {
        match self {
            Fused::AddRmsQuant | Fused::QuantEpilogue => params[1],
            Fused::GdnDecode => params[0],
            Fused::GdnDecodePool => params[0] * params[6],
            Fused::GqaDecodePrep => (params[0] + params[1]) * params[6],
            Fused::I8GemvMulti => {
                let (rows, cols) = kernels_cuda::get("matmul_i8_gemv").map_or((1, 1), |k| k.tile);
                params[0].div_ceil(rows) * params[2..6].iter().map(|n| n.div_ceil(cols)).sum::<u32>()
            }
            // Channel tiles x column tiles x slices; the kernel numbers them
            // channel-tile fastest.
            Fused::Conv2dDwPartial => {
                let (cin, cout, k) = (params[1], params[4], params[5]);
                cout.div_ceil(conv_rows_tile(cout)) * (cin * k * k).div_ceil(CONV_DW_COLS) * params[10]
            }
            Fused::Conv2dDwReduce => params[0].div_ceil(CONV_DW_REDUCE_BLOCK),
        }
    }
}

/// The native id of `which` on this backend, or `None` where it is not
/// offered. Compiles on first use per backend, so a handle caches the answer.
pub(crate) fn resolve_fused(backend: &dyn Backend, which: Fused) -> Option<NativeId> {
    if disabled() {
        return None;
    }
    let cc = backend.caps().arch.compute_capability?;
    let k = kernels_cuda::get(which.registry_name())?;
    if k.min_cc > cc {
        return None;
    }
    backend.register_native(&NativeSpec::Cuda {
        src: k.src,
        entry: k.entry,
        block_dim: k.block_dim,
        bindings: which.bindings(),
        shared_bytes: k.shared_bytes,
        launch: CudaLaunch::NONE,
    })
}

/// `BRAIN_NO_NATIVE_KERNELS=1` disables the tier. Read once: the policy must
/// stay fixed for a given process.
fn disabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("BRAIN_NO_NATIVE_KERNELS").map(|v| v != "0").unwrap_or(false))
}

/// One resolved redirect for a handle.
#[derive(Clone, Debug)]
pub(crate) struct Active {
    /// The pipeline slot the model registered and dispatches by index.
    pub slow: usize,
    pub id: NativeId,
    /// The registry entry's name, for diagnostics.
    pub kernel: &'static str,
    /// Output tile a block covers, as the registry entry declares it.
    pub tile: (u32, u32),
    serves: fn(&[u32]) -> bool,
    blocks: fn(&[u32], (u32, u32)) -> u32,
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
    if disabled() {
        return Vec::new();
    }
    let Some(cc) = backend.caps().arch.compute_capability else {
        return Vec::new();
    };
    ROWS.iter()
        .filter_map(|row| {
            let slow = names.iter().position(|n| n == row.slow)?;
            if row.requires_wgsl_upgrade {
                wgsl_upgrades.iter().find(|a| a.slow == slow)?;
            }
            let k = row.target.resolve(cc)?;
            let id = backend.register_native(&NativeSpec::Cuda {
                src: k.src,
                entry: k.entry,
                block_dim: k.block_dim,
                bindings: row.bindings,
                shared_bytes: k.shared_bytes,
                launch: CudaLaunch::NONE,
            })?;
            Some(Active { slow, id, kernel: k.name, tile: k.tile, serves: row.serves, blocks: row.blocks })
        })
        .collect()
}

/// The native pipeline and BLOCK count to dispatch instead of `kind`, or
/// `None` to keep the WGSL tier. `params` is the caller's own uniform; a call
/// without one (`Gpu::step_buf`, whose uniform lives in a caller-owned
/// buffer) is never redirected, exactly as a shape-specialised
/// [`crate::upgrade`] row is not.
#[inline]
pub(crate) fn apply(active: &[Active], kind: usize, params: &[u32]) -> Option<(NativeId, u32)> {
    let a = active.iter().find(|a| a.slow == kind)?;
    if !(a.serves)(params) {
        return None;
    }
    Some((a.id, (a.blocks)(params, a.tile)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_int8_gemv_row_names_a_registry_entry_with_its_own_bindings() {
        let row = &ROWS[0];
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
}
