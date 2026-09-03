// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared fp32/int8 linear-dispatch machinery for DiT-style forwards -
//! hoisted from `flux1::model` / `flux2::model`, which each carried a
//! near-verbatim copy (both copies even doc-commented "the ONE name→tier
//! map"). The models genuinely differ (per-block vs global modulation,
//! 3- vs 4-axis RoPE, biased vs bias-free linears), but the NUMERIC-TIER
//! machinery - the Precision map, the packed-int8 resident weight, the
//! K-keyed packed-activation scratch, per-token activation quant, and the
//! DP4A GEMM dispatch with `sx·sw` dequant - is model-agnostic; an int8 fix
//! previously had to land twice.
//!
//! The quantizer itself was already shared (`model::int8`); this module is
//! the DISPATCH layer over it: which buffer a K-wide activation packs into,
//! and how a linear at either tier turns into one `Gpu::step_sliced`.
//! Kernel INDICES stay caller-owned (each model registers its own pipeline
//! list), passed in per call - the same convention as `model::block`.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::block::{gemm_variant, GemmVariants};
use crate::int8::{quant_rows_steps, QuantRows};

/// DiT numeric tier: fp32 is the parity reference; int8 quantizes every
/// in-block linear (group-wise symmetric weights + dynamic per-token
/// activation quant, DP4A GEMM - GPU only). Norms/RoPE/attention/activations
/// always stay f32 (the engine's fp32-only core-compute rule).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Precision {
    F32,
    Int8,
}

impl Precision {
    /// The CLI/capability enum (`fp32 | int8`) - the ONE name→tier map.
    pub fn from_name(s: &str) -> Result<Precision, String> {
        match s {
            "fp32" => Ok(Precision::F32),
            "int8" => Ok(Precision::Int8),
            other => Err(format!("unknown precision {other} (fp32|int8)")),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Precision::F32 => "fp32",
            Precision::Int8 => "int8",
        }
    }
}

/// One linear's resident weight: fp32, or int8 (packed `[n, k/4]` u32 +
/// group-wise scale `[n, k/32]` - `model::int8::quantize_weight` layout). A
/// model whose linears carry biases wraps this with its own bias buffer.
pub enum LinW {
    F32(DeviceBuffer),
    I8(DeviceBuffer, DeviceBuffer),
}

impl LinW {
    /// Whether this linear consumes the packed int8 activation (i.e. its
    /// activation must be quantized before it runs).
    pub fn is_i8(&self) -> bool {
        matches!(self, LinW::I8(..))
    }
}

/// Int8 activation-quantization scratch: the per-token dynamic scale plus one
/// packed-activation buffer per contraction width. ONE quant feeds every
/// linear reading that activation. The widths are distinct within a model,
/// so K keys the buffer.
pub struct I8Scratch {
    /// `[rows]` per-token activation scale (`max_abs_row` output).
    pub sx: DeviceBuffer,
    slots: Vec<(u32, DeviceBuffer)>,
}

impl I8Scratch {
    /// `sx_rows` per-token scales; one `rows·K/4`-word packed buffer per
    /// width in `widths`.
    pub fn new(g: &Gpu, sx_rows: u64, rows: u64, widths: &[u32]) -> I8Scratch {
        I8Scratch {
            sx: g.storage(sx_rows),
            slots: widths.iter().map(|&k| (k, g.storage(rows * k as u64 / 4))).collect(),
        }
    }

    /// The packed-activation scratch for a K-wide activation.
    pub fn xq_for(&self, k: u32) -> &DeviceBuffer {
        self.slots
            .iter()
            .find(|(w, _)| *w == k)
            .map(|(_, b)| b)
            .unwrap_or_else(|| panic!("no int8 activation scratch for K={k}"))
    }

    /// Quantize rows `r0..r1` of `x` `[.., k]` into the K-matched packed
    /// scratch with fresh per-token scales (`max_abs_row` → `quant_pack`;
    /// `quant` is those two kernel indices).
    pub fn quant_rows(&self, g: &Gpu, quant: [usize; 2], s: &mut Vec<Step>, x: &DeviceBuffer, r0: u32, r1: u32, k: u32) {
        s.extend(quant_rows_steps(g, QuantRows { kernels: quant, x, sx: &self.sx, xq: self.xq_for(k), xgs: None }, r0, r1, k));
    }
}

/// Sliced fp32 matmul: rows `xr0..xr0+m` of `x` `[.., k]` → the `m·n` floats
/// of `o` at float offset `ooff`. Kernel choice is
/// [`crate::block::gemm_variant`] over the caller's tier.
#[allow(clippy::too_many_arguments)]
pub fn mm_rows_off(g: &Gpu, tier: GemmVariants, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, xr0: u32, ooff: u64, m: u32, k: u32, n: u32) -> Step {
    let xo = (xr0 as u64 * k as u64, m as u64 * k as u64);
    let oo = (ooff, m as u64 * n as u64);
    let (kind, threads) = gemm_variant(tier, m, n);
    g.step_sliced(kind, &[x, w, o], &[xo, (0, 0), oo], &[m, k, n], threads)
}

/// Int8 DP4A matmul over pre-quantized rows `xr0..xr0+m` of the K-matched
/// packed scratch, writing the `m·n` floats of `o` at float offset `ooff`.
/// Dequantizes with the per-token `sx` (sliced at `xr0`) × the group-wise
/// `sw` (bound whole; the kernel derives the group count from `k/4`).
/// `i8_tier` is the DP4A GEMM family (same selection rule and dispatch
/// geometry as the fp32 tier; GPU-only, hence always a `Fast` arm).
#[allow(clippy::too_many_arguments)]
pub fn mm8_rows_off(g: &Gpu, i8_tier: GemmVariants, scr: &I8Scratch, wq: &DeviceBuffer, sw: &DeviceBuffer, o: &DeviceBuffer, xr0: u32, ooff: u64, m: u32, k: u32, n: u32) -> Step {
    let kg = k as u64 / 4;
    let xo = (xr0 as u64 * kg, m as u64 * kg);
    let so = (xr0 as u64, m as u64);
    let oo = (ooff, m as u64 * n as u64);
    let (kind, threads) = gemm_variant(i8_tier, m, n);
    g.step_sliced(kind, &[scr.xq_for(k), wq, &scr.sx, sw, o], &[xo, (0, 0), so, (0, 0), oo], &[m, k / 4, n], threads)
}

/// Int4 (q4, W4A8) DP4A matmul over pre-quantized rows `xr0..xr0+m` of the
/// K-matched packed INT8 activation scratch (q4 is W4A8 - only the weight
/// narrows further, the activation quantizer is unchanged from the int8
/// tier), writing the `m·n` floats of `o` at float offset `ooff`.
///
/// The q4 sibling of [`mm8_rows_off`] - deliberately NOT a thin wrapper over
/// it, because `matmul_q4_{dyn,gemv}.wgsl`'s own `k` PARAM contract differs
/// from the int8 family's: those kernels take the RAW logical `k`,
/// un-divided, since the packed activation (int8, 4 values/word) and the
/// packed weight (int4, 8 values/word) have DIFFERENT word densities for the
/// same `k` - a single shared `kg` the way [`mm8_rows_off`] passes would be
/// ambiguous about which operand it counts (see `matmul_q4_dyn.wgsl`'s own
/// header comment). Passing `mm8_rows_off`'s `k/4` to a q4 kernel - or this
/// function's raw `k` to an int8 kernel - is exactly the silently-wrong
/// arithmetic `model::int8`'s own module doc warns about, not a crash.
///
/// The BUFFER OFFSETS are otherwise identical to `mm8_rows_off`'s: the packed
/// activation is still read at the int8 word offset (`k/4`) regardless.
/// `q4_tier` is the DP4A GEMM family, same selection rule as the int8 tier.
///
/// **Thread count - the SECOND place q4 and i8 disagree.**
/// [`crate::block::gemm_variant`]'s `tiled` slot assumes whatever kernel a
/// caller registered there is the 128×128-tile/256-thread family (true for
/// `matmul_reg2`/`matmul_reg3`/`matmul_i8_dyn` - its `_ => (tiled,
/// m.div_ceil(128) * n.div_ceil(128) * 256)` fallback is exactly that
/// formula). `matmul_q4_dyn.wgsl` breaks that assumption: its own header
/// states it is deliberately the NAIVE, non-tiled tier ("the correct-first,
/// non-tiled q4 GEMM… a register-tiled `matmul_q4_dyn`… is the documented
/// follow-on optimization… not attempted here"), so it needs `m*n`
/// invocations like [`mm_rows_off`]'s `Reference` kernel, not the tile
/// formula. Using `gemm_variant`'s own thread count here under-dispatches
/// `matmul_q4_dyn` (leaves most of the output buffer never written) -
/// detected by comparing which kernel slot `gemm_variant` actually chose
/// (`kind == tiled`) rather than trusting its thread count blindly.
#[allow(clippy::too_many_arguments)]
pub fn mm4_rows_off(g: &Gpu, q4_tier: GemmVariants, scr: &I8Scratch, wq: &DeviceBuffer, sw: &DeviceBuffer, o: &DeviceBuffer, xr0: u32, ooff: u64, m: u32, k: u32, n: u32) -> Step {
    let kg = k as u64 / 4;
    let xo = (xr0 as u64 * kg, m as u64 * kg);
    let so = (xr0 as u64, m as u64);
    let oo = (ooff, m as u64 * n as u64);
    let (kind, tile_threads) = gemm_variant(q4_tier, m, n);
    let threads = match q4_tier {
        GemmVariants::Fast { tiled, .. } if kind == tiled => m * n,
        _ => tile_threads,
    };
    g.step_sliced(kind, &[scr.xq_for(k), wq, &scr.sx, sw, o], &[xo, (0, 0), so, (0, 0), oo], &[m, k, n], threads)
}
