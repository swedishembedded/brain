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

/// One linear's RUNTIME low-rank (LoRA) correction: `y = W·x + B·(A·x)`,
/// evaluated beside the base matmul instead of being folded into `W`.
///
/// ## Why a runtime correction exists at all
///
/// Folding is the obvious deployment: rebuild `W' = W + s·B·A` once, quantize
/// it, and the forward is unchanged. That is correct at fp32 and **silently
/// destructive at int8**. An int8 weight has 256 levels; a trained adapter's
/// delta is typically a fraction of ONE level, so requantizing `W + s·B·A`
/// rounds the overwhelming majority of the adapted weights straight back to
/// the base code - the correction is discarded - while the minority that do
/// move deliver a badly distorted version of it. `flux2`'s
/// `tests/lora_requant_int8.rs` measures both halves of that on fixtures:
/// at a delta 0.5% of the weight magnitude, 94.8% of weights round back
/// unchanged and the delivered delta's cosine to the intended one is 0.44.
/// The residue is a token-CONSTANT bias in the matmul output, which is what
/// turns into a visible periodic artifact once a VAE decoder's upsampling
/// stages get hold of it.
///
/// Keeping the base weight exactly as the checkpoint quantized it and adding
/// `s·B·(A·x)` in fp32 at runtime has no such failure mode: nothing is ever
/// requantized, and the correction arrives in full. It costs one extra skinny
/// GEMM (`n = r`) plus one accumulate pass over the output - rank is 8-32
/// against hidden dims of 3072-12288, so a few percent of the linear's work.
///
/// ## Layout
///
/// `a` is `A [r, k]` row-major - already a GEMM weight in this engine's
/// `out = x @ Wᵀ` convention, so `A·x` needs no new kernel. `bt` is `B [n, r]`
/// stored **transposed** as `[r, n]`, with the adapter's `α/r` scale and the
/// caller's strength already multiplied in, which is what makes the
/// accumulate kernel's `B` read coalesced (see `lora_delta.wgsl`).
///
/// A STACK of adapters over the same linear is ONE `LoraW`, not several:
/// `Σᵢ sᵢ·Bᵢ·Aᵢ = [s₁B₁ | s₂B₂ | …]·[A₁; A₂; …]`, so the pairs concatenate
/// along the rank axis and the sum the fold documents falls out of a single
/// rank-`Σrᵢ` correction.
pub struct LoraW {
    /// `A [r, k]`, row-major.
    pub a: DeviceBuffer,
    /// `B [n, r]` transposed to `[r, n]`, scale folded in.
    pub bt: DeviceBuffer,
    /// The (possibly concatenated) rank.
    pub r: u32,
}

/// Record the two steps of one runtime LoRA correction over rows
/// `xr0..xr0+m` of `x` `[.., k]`, accumulating into the `m·n` floats of `o` at
/// float offset `ooff` - the same row/offset contract as [`mm_rows_off`], so a
/// correction lands exactly where its base linear wrote.
///
/// `tscr` is a `[rows, r]` scratch the caller owns; rows `0..m` of it are
/// overwritten and consumed by the second step, so ONE buffer serves every
/// corrected linear in a forward (the steps of one correction are adjacent in
/// the submitted list, like [`I8Scratch`]'s packed activation).
///
/// `tier` is the fp32 GEMM family for `A·x`; `lora_kernel` is the index the
/// caller registered `kernels::LORA_DELTA` at.
#[allow(clippy::too_many_arguments)]
pub fn lora_rows_off(
    g: &Gpu,
    tier: GemmVariants,
    lora_kernel: usize,
    lo: &LoraW,
    tscr: &DeviceBuffer,
    x: &DeviceBuffer,
    o: &DeviceBuffer,
    xr0: u32,
    ooff: u64,
    m: u32,
    k: u32,
    n: u32,
) -> [Step; 2] {
    let proj = mm_rows_off(g, tier, x, &lo.a, tscr, xr0, 0, m, k, lo.r);
    let oo = (ooff, m as u64 * n as u64);
    let to = (0u64, m as u64 * lo.r as u64);
    let add = g.step_sliced(lora_kernel, &[o, tscr, &lo.bt], &[oo, to, (0, 0)], &[m, n, lo.r], m * n);
    [proj, add]
}

/// The slot value that means "this row uses no adapter". The bank kernels treat
/// ANY slot `>= LoraBank::slots` the same way (the row is left untouched), so a
/// stale or corrupt slot can never index outside the bank.
pub const LORA_SLOT_NONE: u32 = u32::MAX;

/// A device-resident bank of LoRA adapters for ONE linear, so a single batch can
/// apply a different adapter (or none) to each row: the multi-adapter sibling of
/// [`LoraW`], which holds exactly one.
///
/// Every slot holds one adapter in [`LoraW`]'s layout, padded to `r_max` rows:
///
/// * `a`: `[slots, r_max, k]` - slot `s`'s `A [r, k]` at `s·r_max·k`;
/// * `bt`: `[slots, r_max, n]` - slot `s`'s `B` transposed to `[r, n]`, `α/r`
///   (and any strength) folded in, at `s·r_max·n`;
/// * `rank`: `[slots]` u32 - the live rank of each slot.
///
/// Rows `rank..r_max` of a slot are never read, so they need not be zeroed and
/// loading a lower-rank adapter into a slot only overwrites its own rows. A rank
/// above `r_max` is clamped to it. Loading or unloading an adapter is a buffer
/// write; no recorded step has to change.
pub struct LoraBank {
    /// `A_bank [slots, r_max, k]`.
    pub a: DeviceBuffer,
    /// `Bt_bank [slots, r_max, n]`.
    pub bt: DeviceBuffer,
    /// Live rank of each slot, `[slots]` u32.
    pub rank: DeviceBuffer,
    /// Number of slots.
    pub slots: u32,
    /// Rank capacity of every slot.
    pub r_max: u32,
}

/// The three kernels [`lora_bank_rows`] dispatches, as `(name, source)` pairs
/// for a model's pipeline list (they are resolved by name,
/// [`LoraBankKernels::resolve`], so their position carries no meaning). A
/// model that serves an adapter bank registers these.
pub const LORA_BANK_PIPELINES: [(&str, &str); 4] = [
    ("lora_xa_rows", kernels::LORA_XA_ROWS),
    ("lora_xa_rows_serial", kernels::LORA_XA_ROWS_SERIAL),
    ("lora_delta_rows", kernels::LORA_DELTA_ROWS),
    ("lora_delta_rows_multi", kernels::LORA_DELTA_ROWS_MULTI),
];

/// The three pipeline indices [`lora_bank_rows`] chooses among, as the caller
/// registered them.
#[derive(Clone, Copy, Debug)]
pub struct LoraBankKernels {
    /// `lora_xa_rows` (workgroup per row and rank index).
    pub xa: usize,
    /// `lora_xa_rows_serial` (the barrier-free twin for devices without
    /// workgroup reductions).
    pub xa_serial: usize,
    /// `lora_delta_rows`.
    pub delta: usize,
    /// `lora_delta_rows_multi`, the epilogue of a group of linears that share
    /// their input ([`lora_bank_group_rows`]); `None` when the model did not
    /// register it, in which case a group is corrected site by site.
    pub delta_multi: Option<usize>,
}

impl LoraBankKernels {
    /// Look the three kernels up by name on `g`; `None` when the model did not
    /// register all of them. All three are needed because which projection
    /// runs is decided at record time from the device, not by the caller.
    pub fn resolve(g: &Gpu) -> Option<LoraBankKernels> {
        Some(LoraBankKernels {
            xa: g.kernel_index("lora_xa_rows")?,
            xa_serial: g.kernel_index("lora_xa_rows_serial")?,
            delta: g.kernel_index("lora_delta_rows")?,
            delta_multi: g.kernel_index("lora_delta_rows_multi"),
        })
    }
}

/// One per-row-adapter LoRA correction over a window of rows: what
/// [`lora_bank_rows`] records.
pub struct LoraBankRows<'a> {
    /// Pipeline indices of the bank kernels.
    pub kernels: LoraBankKernels,
    /// The adapters of this linear.
    pub bank: &'a LoraBank,
    /// The u32 slot of every row, indexed like `x` (the window is
    /// `row0..row0 + rows`); [`LORA_SLOT_NONE`] or any slot `>= bank.slots`
    /// leaves the row untouched.
    pub slot: &'a DeviceBuffer,
    /// `[rows, bank.r_max]` scratch the caller owns, overwritten by the
    /// projection and consumed by the epilogue (the two steps are adjacent, so
    /// one buffer serves every corrected linear in a forward).
    pub tscr: &'a DeviceBuffer,
    /// The fp32 activation `[.., k]` the base linear read.
    pub x: &'a DeviceBuffer,
    /// The base linear's output, `out_stride` floats per row.
    pub out: &'a DeviceBuffer,
    /// First row of the window in `x`, `slot` and `out`.
    pub row0: u32,
    /// Rows in the window.
    pub rows: u32,
    /// Contraction width of the linear.
    pub k: u32,
    /// Output width of THIS projection.
    pub n: u32,
    /// Floats between consecutive output rows (`>= out_col0 + n`).
    pub out_stride: u32,
    /// First output column of this projection inside a row: 0 for a plain
    /// linear, and the q/k/v (or gate/up) offset when the base wrote a fused
    /// projection, so the correction lands in place with no gather or split.
    pub out_col0: u32,
}

/// Record the two steps of a per-row-adapter LoRA correction:
/// `out[i, out_col0 + j] += s·B_slot(i)·(A_slot(i)·x_i)` for every row `i` of the
/// window, with the adapter chosen per row by `slot` (see [`LoraBank`]).
///
/// Rows need not be contiguous or sorted by adapter, and the number of
/// dispatches is two whatever number of distinct adapters the window holds. A
/// row whose slot is [`LORA_SLOT_NONE`] (or out of range, or whose slot has rank
/// 0) is not touched: its output bits are exactly what the base linear wrote.
/// Each corrected row equals [`lora_rows_off`] run on that row alone with its
/// adapter, up to fp32 accumulation order in `A·x`.
///
/// The projection runs as the cooperative `lora_xa_rows` on a device with
/// workgroup reductions and as `lora_xa_rows_serial` otherwise - the choice is
/// `backend_api::select`'s `Op::LoraBankProject`, keyed on the queried
/// capabilities.
///
/// # Panics
///
/// On a window whose geometry cannot be right: zero `rows`, `k`, `n`, or bank
/// extent, or `out_col0 + n > out_stride`.
pub fn lora_bank_rows(g: &Gpu, rq: &LoraBankRows) -> [Step; 2] {
    use gpu_core::select::{Dtype, KernelSelector, KernelVariant, Op, OpShape};

    let LoraBankRows { kernels, bank, slot, tscr, x, out, row0, rows, k, n, out_stride, out_col0 } = *rq;
    assert!(rows > 0 && k > 0 && n > 0 && bank.slots > 0 && bank.r_max > 0, "lora_bank_rows: empty window or bank");
    assert!(out_col0 as u64 + n as u64 <= out_stride as u64, "lora_bank_rows: columns {out_col0}..{} overrun the {out_stride}-float output row", out_col0 + n);

    let shape = OpShape { m: rows, n: bank.r_max, k, dtype: Dtype::F32 };
    let (xa_kernel, xa_grid) = match gpu_core::select::DefaultSelector.select(Op::LoraBankProject, shape, &g.caps()) {
        KernelVariant::WorkgroupPerOutput => (kernels.xa, gpu_core::Dispatch::Workgroups(rows * bank.r_max)),
        _ => (kernels.xa_serial, gpu_core::Dispatch::Threads(rows * bank.r_max)),
    };
    // Every buffer is bound whole and the window is a Param: a bind offset
    // must respect the device's storage-buffer offset alignment (256 bytes),
    // which an arbitrary `row0` does not.
    let project = g.dispatch(xa_kernel, &[x, &bank.a, slot, &bank.rank, tscr], &[rows, k, bank.r_max, bank.slots, row0], xa_grid);
    let add = g.dispatch(
        kernels.delta,
        &[out, tscr, &bank.bt, slot, &bank.rank],
        &[rows, n, bank.r_max, bank.slots, out_stride, out_col0, row0],
        gpu_core::Dispatch::Threads(rows * n),
    );
    [project, add]
}

/// The largest group of linears [`lora_bank_group_rows`] corrects in one
/// epilogue dispatch (the kernel's output bindings).
pub const LORA_GROUP_MAX: usize = 4;

/// One output of a group correction: the base linear's output buffer, its width
/// and where the correction lands in a row of it.
pub struct LoraGroupOut<'a> {
    /// The linear's output, `stride` floats per row.
    pub out: &'a DeviceBuffer,
    /// This linear's output width `n`.
    pub n: u32,
    /// Floats between consecutive output rows.
    pub stride: u32,
    /// First output column of the correction inside a row.
    pub col0: u32,
}

/// A group's bank: the A matrices of linears that read ONE activation, stacked
/// along the rank axis, and their `Bt` matrices laid end to end.
pub struct LoraGroupBank {
    /// `[slots, g·r_max, k]`: site `s`'s `A [r_max, k]` at rows `s·r_max`,
    /// zero-padded past its rank.
    pub a: DeviceBuffer,
    /// `[slots, r_max · Σ n]`: site `s`'s `Bt [r_max, n_s]` at `r_max · Σ_{h<s} n_h`,
    /// zero-padded past its rank.
    pub bt: DeviceBuffer,
    /// Per slot, non-zero when its adapter corrects any site of the group.
    pub rank: DeviceBuffer,
    /// Number of slots.
    pub slots: u32,
    /// Rank capacity of every site.
    pub r_max: u32,
    /// Two one-float buffers bound to the output slots a group of fewer than
    /// four leaves unused (a buffer cannot be bound writable twice in a dispatch).
    pub spare: [DeviceBuffer; 2],
}

/// One per-row-adapter correction of a group of 2..=[`LORA_GROUP_MAX`] linears
/// that read the same activation `x`: the work of [`lora_bank_rows`] for each of
/// them in TWO dispatches instead of two each. Rows whose slot is
/// [`LORA_SLOT_NONE`] (or out of range, or whose slot corrects no site of the
/// group) are not touched, so their output bits are what the base wrote.
///
/// # Panics
///
/// On a group the kernel cannot take (`outs` outside `2..=LORA_GROUP_MAX`, an
/// empty window or bank), or when the model did not register
/// `lora_delta_rows_multi`.
#[allow(clippy::too_many_arguments)]
pub fn lora_bank_group_rows(g: &Gpu, kernels: LoraBankKernels, bank: &LoraGroupBank, slot: &DeviceBuffer, tscr: &DeviceBuffer, x: &DeviceBuffer, outs: &[LoraGroupOut], rows: u32, k: u32) -> [Step; 2] {
    use gpu_core::select::{Dtype, KernelSelector, KernelVariant, Op, OpShape};

    let sites = outs.len() as u32;
    assert!((2..=LORA_GROUP_MAX as u32).contains(&sites), "lora_bank_group_rows: {sites} sites");
    assert!(rows > 0 && k > 0 && bank.slots > 0 && bank.r_max > 0, "lora_bank_group_rows: empty window or bank");
    let delta_multi = kernels.delta_multi.expect("lora_bank_group_rows: the model did not register lora_delta_rows_multi");
    let stacked = sites * bank.r_max;
    let shape = OpShape { m: rows, n: stacked, k, dtype: Dtype::F32 };
    let (xa_kernel, xa_grid) = match gpu_core::select::DefaultSelector.select(Op::LoraBankProject, shape, &g.caps()) {
        KernelVariant::WorkgroupPerOutput => (kernels.xa, gpu_core::Dispatch::Workgroups(rows * stacked)),
        _ => (kernels.xa_serial, gpu_core::Dispatch::Threads(rows * stacked)),
    };
    // The projection runs over the stacked A bank as if it were one adapter of rank `g·r_max`.
    let project = g.dispatch(xa_kernel, &[x, &bank.a, slot, &bank.rank, tscr], &[rows, k, stacked, bank.slots, 0], xa_grid);

    let mut params = vec![rows, sites, bank.r_max, bank.slots, 0];
    let mut off = 0u32;
    for o in outs {
        params.extend([o.n, o.stride, o.col0, off]);
        off += bank.r_max * o.n;
    }
    params.resize(5 + 4 * LORA_GROUP_MAX, 0);
    let total_n: u32 = outs.iter().map(|o| o.n).sum();
    let bound = |i: usize| outs.get(i).map_or_else(|| &bank.spare[i - 2], |o| o.out);
    let add = g.dispatch(delta_multi, &[tscr, &bank.bt, slot, &bank.rank, bound(0), bound(1), bound(2), bound(3)], &params, gpu_core::Dispatch::Threads(rows * total_n));
    [project, add]
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

    /// A scratch whose per-row scales and packed activation were produced by
    /// something other than [`Self::quant_rows`] - a fused kernel that quantises
    /// as part of the step that made the activation. `sx` is `[rows]` and `xq`
    /// `[rows, k/4]`, the layout [`Self::new`] sizes for.
    pub fn from_parts(sx: DeviceBuffer, k: u32, xq: DeviceBuffer) -> I8Scratch {
        I8Scratch { sx, slots: vec![(k, xq)] }
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
    let (kind, grid) = gemm_variant(tier, m, n);
    g.dispatch_sliced(kind, &[x, w, o], &[xo, (0, 0), oo], &[m, k, n], grid)
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
    let (kind, grid) = gemm_variant(i8_tier, m, n);
    g.dispatch_sliced(kind, &[scr.xq_for(k), wq, &scr.sx, sw, o], &[xo, (0, 0), so, (0, 0), oo], &[m, k / 4, n], grid)
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
    let (kind, tile_grid) = gemm_variant(q4_tier, m, n);
    let grid = match q4_tier {
        GemmVariants::Fast { tiled, .. } if kind == tiled => gpu_core::Dispatch::Threads(m * n),
        _ => tile_grid,
    };
    g.dispatch_sliced(kind, &[scr.xq_for(k), wq, &scr.sx, sw, o], &[xo, (0, 0), so, (0, 0), oo], &[m, k, n], grid)
}
