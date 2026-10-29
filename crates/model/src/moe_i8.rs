// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sparse mixture-of-experts inference that
// stays on the memory roofline at both decode and prefill. If your team needs
// expertise in int8 expert banks, device-side routing and tensor-core grouped
// GEMMs then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The int8 sparse-MoE sublayer without a host synchronisation and without a
//! per-expert dispatch: every expert a row selects is a `slot`, the slot's
//! expert is read out of one fused bank by id, and a whole layer is a handful
//! of dispatches whatever the expert count.
//!
//! ```text
//! xn --quant--> xq, sx
//! router logits [rows, E]  (computed by the caller: a hard top-k decision is kept in fp32)
//! moe_router_topk  --> ids[rows, K], weight[rows, K]
//! (rows >= grouped_min_rows)  moe_route_{count,scan,emit}: slots ordered by expert
//! gate, up : gather GEMV (decode) or grouped GEMM (prefill) over the fused banks
//! moe_swiglu_quant --> hq, sh        SiLU(gate) * up, requantised per slot
//! down     : same --> y[rows * K, d]
//! moe_slot_combine --> out[rows, d] = sum_s weight * y
//! ```
//!
//! At decode a token's experts are mostly distinct matrices, so each slot reads
//! its own weight rows (`moe_i8_gemv_gather`) and the layer streams exactly the
//! bytes the selected experts hold. From a few rows up an expert has several
//! slots and the grouped kernel (`moe_i8_grouped`, native int8 tensor cores on
//! CUDA) reads a weight row once for up to eight of them.
//!
//! The router weights are DeepSeek-V2's: the raw softmax probabilities of the
//! chosen experts when `norm` is off, the renormalised ones when on, times
//! `scale` ([`SlotRouter`]).
//!
//! A shared expert that has the routed experts' shape rides along as one more
//! slot of every row ([`SlotShape::shared`]): it is block `n_experts` of each
//! bank, the router matmul grows by one row (the shared expert's gate) and the
//! router kernel emits `sigmoid(gate . x)` as that slot's combine weight. That
//! is the Qwen3.5/3.6 MoE; a shared expert of another width is the caller's to
//! add on top of the routed output.

use gpu_core::{DeviceBuffer, Dispatch, Gpu, Step};

use crate::int8::{act_scale_len, GROUP};

/// Weight rows one `moe_i8_gemv_gather` workgroup covers (64 threads, 16 lanes per row): the kernel's own
/// constant, restated to size its grid.
pub const GATHER_COLS: u32 = 4;
/// Slots per tile of `moe_i8_grouped`: the kernel's own constant, restated to size its grid and the routing table.
pub const GROUPED_MR: u32 = 8;
/// Weight rows per workgroup of `moe_i8_grouped`.
pub const GROUPED_COLS: u32 = 4;
/// Rows at which the expert GEMMs switch from one slot per weight pass to the grouped kernel. Below it a token's
/// experts are mostly distinct matrices and there is nothing to share (swept on a GH200 on a 35B-A3B decode step).
pub const GROUPED_MIN_ROWS: u32 = 7;

/// The kernels the int8 slot sublayer dispatches, by pipeline index on one [`Gpu`].
#[derive(Clone, Copy, Debug)]
pub struct SlotKernels {
    pub router_topk: usize,
    pub route_count: usize,
    pub route_scan: usize,
    pub route_emit: usize,
    pub i8_grouped: usize,
    pub i8_gather: usize,
    pub swiglu_quant: usize,
    pub slot_combine: usize,
    pub max_abs_row: usize,
    pub quant_pack: usize,
}

impl SlotKernels {
    /// The `(name, source)` pairs a model adds to its pipeline list.
    pub fn pipelines() -> Vec<(&'static str, &'static str)> {
        vec![
            ("moe_router_topk", kernels::MOE_ROUTER_TOPK),
            ("moe_route_count", kernels::MOE_ROUTE_COUNT),
            ("moe_route_scan", kernels::MOE_ROUTE_SCAN),
            ("moe_route_emit", kernels::MOE_ROUTE_EMIT),
            ("moe_i8_grouped", kernels::MOE_I8_GROUPED),
            ("moe_i8_gemv_gather", kernels::MOE_I8_GEMV_GATHER),
            ("moe_swiglu_quant", kernels::MOE_SWIGLU_QUANT),
            ("moe_slot_combine", kernels::MOE_SLOT_COMBINE),
            ("max_abs_row", kernels::MAX_ABS_ROW),
            ("quant_pack", kernels::QUANT_PACK),
        ]
    }

    /// Look the kernels up by name on `g`.
    pub fn resolve(g: &Gpu) -> Result<SlotKernels, String> {
        let find = |name: &str| g.kernel_index(name).ok_or_else(|| format!("int8 slot MoE: kernel '{name}' is not registered on this Gpu (add moe_i8::SlotKernels::pipelines)"));
        Ok(SlotKernels {
            router_topk: find("moe_router_topk")?,
            route_count: find("moe_route_count")?,
            route_scan: find("moe_route_scan")?,
            route_emit: find("moe_route_emit")?,
            i8_grouped: find("moe_i8_grouped")?,
            i8_gather: find("moe_i8_gemv_gather")?,
            swiglu_quant: find("moe_swiglu_quant")?,
            slot_combine: find("moe_slot_combine")?,
            max_abs_row: find("max_abs_row")?,
            quant_pack: find("quant_pack")?,
        })
    }
}

/// One projection of every expert of a layer as ONE fused int8 bank: expert `e`'s `[n, k]` matrix is rows
/// `e*n .. (e+1)*n` of a `[blocks * n, k/4]` packed-word buffer with a `[blocks * n, k/32]` group-scale sibling
/// (llama.cpp's stacked-expert layout, so a Q8_0 checkpoint becomes this by a byte repack).
pub struct I8Bank {
    pub packed: DeviceBuffer,
    pub scale: DeviceBuffer,
    /// Output width of one expert's matrix.
    pub n: u32,
    /// Contraction width.
    pub k: u32,
    /// Stacked matrices (experts) in the bank.
    pub blocks: u32,
}

impl I8Bank {
    /// Quantise tensor `name` (`blocks` stacked `[n, k]` matrices, `blocks * n * k` values) from `source` into a
    /// bank on `g`, or `None` when the source does not hold it. `k` must be a multiple of [`GROUP`].
    pub fn from_source(g: &Gpu, source: &dyn checkpoint::TensorSource, name: &str, blocks: usize, n: usize, k: usize) -> Option<I8Bank> {
        let (words, scales) = crate::int8::quantize_from(source, name, blocks * n, k)?;
        let packed = g.storage(words.len() as u64);
        let scale = g.storage(scales.len() as u64);
        g.write(&packed, &words);
        g.write_f32(&scale, &scales);
        Some(I8Bank { packed, scale, n: n as u32, k: k as u32, blocks: blocks as u32 })
    }

    /// Device bytes the bank holds.
    pub fn bytes(&self) -> u64 {
        let (rows, k) = (self.blocks as u64 * self.n as u64, self.k as u64);
        rows * (k + 4 * (k / GROUP as u64))
    }
}

/// The three projections of one MoE layer's routed experts.
pub struct I8Experts {
    pub gate: I8Bank,
    pub up: I8Bank,
    pub down: I8Bank,
}

/// How the router's probabilities become combine weights.
#[derive(Clone, Copy, Debug)]
pub struct SlotRouter {
    /// Divide the kept probabilities by their own sum (`norm_topk_prob`).
    pub norm: bool,
    /// DeepSeek's `routed_scaling_factor`, applied last (1.0 = none).
    pub scale: f32,
}

/// The rows, widths and expert counts one sublayer call runs over.
#[derive(Clone, Copy, Debug)]
pub struct SlotShape {
    pub rows: u32,
    pub d_model: u32,
    pub moe_ff: u32,
    pub n_experts: u32,
    pub top_k: u32,
    /// 1 when the shared expert is block `n_experts` of every bank and one more slot per row, else 0. The router
    /// `logits` then have `n_experts + 1` columns, the last being the shared expert's gate.
    pub shared: u32,
}

impl SlotShape {
    /// Experts a row can address: the routed ones plus the shared one in the bank.
    fn banked(&self) -> u32 {
        self.n_experts + self.shared
    }

    /// Slots one row occupies: its `top_k` routed experts plus the shared one.
    fn slots_per_row(&self) -> u32 {
        self.top_k + self.shared
    }

    fn slots(&self) -> u32 {
        self.rows * self.slots_per_row()
    }
}

/// Device scratch for one sublayer call, sized once for the widest call (`max_rows`) and reused by every layer.
pub struct SlotScratch {
    max_rows: u32,
    pub xq: DeviceBuffer,
    pub sx: DeviceBuffer,
    ids: DeviceBuffer,
    weight: DeviceBuffer,
    gate_out: DeviceBuffer,
    up_out: DeviceBuffer,
    hq: DeviceBuffer,
    sh: DeviceBuffer,
    y: DeviceBuffer,
    counts: DeviceBuffer,
    route_tab: DeviceBuffer,
    route_perm: DeviceBuffer,
}

impl SlotScratch {
    pub fn new(g: &Gpu, max_rows: u32, d_model: u32, moe_ff: u32, n_experts: u32, top_k: u32, shared: u32) -> SlotScratch {
        let slots = (max_rows * (top_k + shared)) as usize;
        let banked = (n_experts + shared) as usize;
        let st = |n: usize| g.storage(n.max(1) as u64);
        SlotScratch {
            max_rows,
            xq: st(max_rows as usize * d_model as usize / 4),
            sx: st(act_scale_len(max_rows as usize, d_model as usize)),
            ids: st(slots),
            weight: st(slots),
            gate_out: st(slots * moe_ff as usize),
            up_out: st(slots * moe_ff as usize),
            hq: st(slots * moe_ff as usize / 4),
            sh: st(act_scale_len(slots, moe_ff as usize)),
            y: st(slots * d_model as usize),
            counts: st(banked),
            route_tab: st(2 * (banked + 1)),
            route_perm: st(slots),
        }
    }
}

/// One MoE layer's experts over `shape.rows` rows of `xn` (`[rows, d_model]`, already normalised), with
/// the router's `logits` (`[rows, n_experts + shape.shared]`) computed by the caller; the combined output lands in
/// `out` (`[rows, d_model]`, overwritten). `grouped_min_rows` is the row count from which slots are ordered by expert and
/// the grouped GEMM runs ([`GROUPED_MIN_ROWS`] unless a model measured another).
pub fn expert_fwd_i8_slots(g: &Gpu, k: &SlotKernels, shape: &SlotShape, router: SlotRouter, xn: &DeviceBuffer, logits: &DeviceBuffer, experts: &I8Experts, sc: &SlotScratch, grouped_min_rows: u32, out: &DeviceBuffer) -> Vec<Step> {
    let SlotShape { rows, d_model: d, moe_ff: ff, n_experts: e, top_k, shared } = *shape;
    assert!(rows >= 1 && rows <= sc.max_rows, "int8 slot MoE: {rows} rows exceeds the scratch's {}", sc.max_rows);
    assert!(d.is_multiple_of(GROUP as u32) && ff.is_multiple_of(GROUP as u32), "int8 slot MoE: d_model {d} and moe_ff {ff} must be multiples of {GROUP}");
    assert!(shared <= 1, "int8 slot MoE: at most one shared expert rides in the bank, got {shared}");
    let (slots, per_row, banked) = (shape.slots(), shape.slots_per_row(), shape.banked());
    let grouped = rows >= grouped_min_rows;
    let gather = |bank: &I8Bank, xq: &DeviceBuffer, sx: &DeviceBuffer, out: &DeviceBuffer, xdiv: u32| {
        if grouped {
            // Worst-case tile count (an expert's last tile may be partial); a workgroup past the real count finds
            // the table's end and exits.
            let tiles = banked + slots.div_ceil(GROUPED_MR);
            g.dispatch(
                k.i8_grouped,
                &[xq, sx, &sc.route_tab, &sc.route_perm, &bank.packed, &bank.scale, out],
                &[tiles, bank.k / 4, bank.n, xdiv, banked],
                Dispatch::Workgroups(tiles * bank.n.div_ceil(GROUPED_COLS)),
            )
        } else {
            g.dispatch(k.i8_gather, &[xq, sx, &sc.ids, &bank.packed, &bank.scale, out], &[slots, bank.k / 4, bank.n, xdiv], Dispatch::Workgroups(slots * bank.n.div_ceil(GATHER_COLS)))
        }
    };
    let mut steps = Vec::with_capacity(16);
    steps.push(g.step(k.max_abs_row, &[xn, &sc.sx], &[rows, d], rows));
    steps.push(g.step(k.quant_pack, &[xn, &sc.sx, &sc.xq], &[rows, d], rows * d / 4));
    steps.push(g.dispatch(k.router_topk, &[logits, &sc.ids, &sc.weight], &[rows, e, top_k, shared, u32::from(router.norm), router.scale.to_bits()], Dispatch::Workgroups(rows)));
    if grouped {
        steps.push(g.dispatch(k.route_count, &[&sc.ids, &sc.counts], &[slots, banked], Dispatch::Workgroups(banked)));
        steps.push(g.dispatch(k.route_scan, &[&sc.counts, &sc.route_tab], &[banked, GROUPED_MR], Dispatch::Workgroups(1)));
        steps.push(g.dispatch(k.route_emit, &[&sc.ids, &sc.route_tab, &sc.route_perm], &[slots, banked], Dispatch::Workgroups(banked)));
    }
    steps.push(gather(&experts.gate, &sc.xq, &sc.sx, &sc.gate_out, per_row));
    steps.push(gather(&experts.up, &sc.xq, &sc.sx, &sc.up_out, per_row));
    steps.push(g.dispatch(k.swiglu_quant, &[&sc.gate_out, &sc.up_out, &sc.hq, &sc.sh], &[slots, ff], Dispatch::Workgroups(slots)));
    steps.push(gather(&experts.down, &sc.hq, &sc.sh, &sc.y, 1));
    steps.push(g.step(k.slot_combine, &[&sc.y, &sc.weight, out], &[rows, d, per_row], rows * d));
    steps
}
