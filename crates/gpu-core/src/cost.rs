// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-kernel FLOP/OPS/bytes accounting — the single cost registry for every
//! backend and every model.
//!
//! Every model emits device work as `Step`s through `Gpu::step*`, so cost is
//! hung on that seam, not per model: [`kernel_cost`] maps one recorded dispatch
//! (kernel NAME + uniform params + thread count) to a [`Cost`], and
//! [`tally`] folds a step list into a [`CostReport`]. The same formulas serve
//! both directions:
//!
//! * OFFLINE — `Gpu::cost_of(&steps)` walks a recorded step list without
//!   executing anything.
//! * ONLINE — `Gpu::submit` folds every submitted step into that handle's
//!   counters (`Gpu::ops_counters`), so the runtime number reflects exactly the
//!   kernel variants that were dispatched (int8 GEMMs count `int_ops`, fp32
//!   count `flops`). One `Gpu` handle is one device context, so per-`Gpu`
//!   counters are per-device numbers; a sharded pipeline reads each stage's
//!   handle.
//!
//! ## Conventions (the numbers are meaningless without them)
//!
//! * One multiply-accumulate = 2 ops. `flops` are fp32 ops; `int_ops` are
//!   integer ops (the DP4A int8 GEMMs); a kernel can have both (int8 GEMM
//!   epilogue dequantizes in fp32).
//! * Transcendentals / div / sqrt count 1 — a device-neutral floor, not a
//!   per-architecture instruction count.
//! * Loop trip counts are exact for the recorded shape: causal attention costs
//!   t(t+1)/2 pairs, not t². Workgroup-cooperative variants (`rmsnorm_rows`,
//!   `matmul_gemv`) count the row's math once — cooperative fan-out/fold
//!   redundancy is a micro-architectural detail, and counting it would make the
//!   same model math cost different amounts per variant.
//! * `bytes` is best-effort streaming traffic (each logical operand read or
//!   written once, 4 B/element) — a roofline denominator, not a cache model.
//! * An unknown kernel yields `None` — reported as UNCOVERED, never as zero.
//!   [`CostReport::coverage`] states how much of a run the totals actually
//!   describe (brain's rule: unmeasured is null, never 0-pretending-complete).
//!
//! Template-specialised variants (`base#K=V,...` from `kernels::template`) cost
//! as their base kernel: specialisation changes tiling constants, not math.

use std::collections::BTreeMap;
use std::fmt;

use backend_api::Step;

/// The cost of one dispatch (or a sum of them).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cost {
    /// fp32 floating-point ops (1 MAC = 2).
    pub flops: u64,
    /// Integer ops (1 int8 MAC = 2) — the DP4A `matmul_i8*` path.
    pub int_ops: u64,
    /// Approximate bytes moved (streaming estimate).
    pub bytes: u64,
}

impl Cost {
    pub fn add(&mut self, o: Cost) {
        self.flops += o.flops;
        self.int_ops += o.int_ops;
        self.bytes += o.bytes;
    }
}

/// Per-kernel slice of a [`CostReport`].
#[derive(Clone, Copy, Debug, Default)]
pub struct KernelCost {
    pub calls: u64,
    pub cost: Cost,
}

/// Summed cost over a step list (offline) or over everything submitted through
/// one `Gpu` handle (online).
#[derive(Clone, Debug, Default)]
pub struct CostReport {
    /// Totals over the COVERED steps only.
    pub total: Cost,
    /// Steps seen.
    pub steps: u64,
    /// Steps with a cost formula.
    pub covered: u64,
    /// Covered kernels: name -> calls + summed cost.
    pub by_kernel: BTreeMap<String, KernelCost>,
    /// Uncovered kernels: name -> calls. Their work is NOT in `total`.
    pub uncovered: BTreeMap<String, u64>,
}

impl CostReport {
    /// Fold one dispatch in.
    pub fn record(&mut self, name: &str, cost: Option<Cost>) {
        self.steps += 1;
        match cost {
            Some(c) => {
                self.covered += 1;
                self.total.add(c);
                // get_mut-first keeps the hot online path allocation-free.
                if let Some(e) = self.by_kernel.get_mut(name) {
                    e.calls += 1;
                    e.cost.add(c);
                } else {
                    self.by_kernel.insert(name.to_string(), KernelCost { calls: 1, cost: c });
                }
            }
            None => {
                if let Some(v) = self.uncovered.get_mut(name) {
                    *v += 1;
                } else {
                    self.uncovered.insert(name.to_string(), 1);
                }
            }
        }
    }

    /// Fraction of steps with a cost formula (1.0 when empty: nothing missing).
    pub fn coverage(&self) -> f64 {
        if self.steps == 0 { 1.0 } else { self.covered as f64 / self.steps as f64 }
    }

    /// Sum another report into this one (e.g. fwd + bwd, or across stages).
    pub fn merge(&mut self, o: &CostReport) {
        self.total.add(o.total);
        self.steps += o.steps;
        self.covered += o.covered;
        for (k, v) in &o.by_kernel {
            let e = self.by_kernel.entry(k.clone()).or_default();
            e.calls += v.calls;
            e.cost.add(v.cost);
        }
        for (k, v) in &o.uncovered {
            *self.uncovered.entry(k.clone()).or_default() += v;
        }
    }
}

fn eng(x: u64) -> String {
    let f = x as f64;
    if f >= 1e12 {
        format!("{:8.3} T", f / 1e12)
    } else if f >= 1e9 {
        format!("{:8.3} G", f / 1e9)
    } else if f >= 1e6 {
        format!("{:8.3} M", f / 1e6)
    } else if f >= 1e3 {
        format!("{:8.3} k", f / 1e3)
    } else {
        format!("{f:8.0}  ")
    }
}

impl fmt::Display for CostReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:<24} {:>8} {:>11} {:>11} {:>11}", "kernel", "calls", "flops", "int_ops", "bytes")?;
        for (name, k) in &self.by_kernel {
            writeln!(
                f,
                "{:<24} {:>8} {} {} {}",
                name,
                k.calls,
                eng(k.cost.flops),
                eng(k.cost.int_ops),
                eng(k.cost.bytes)
            )?;
        }
        for (name, calls) in &self.uncovered {
            writeln!(f, "{name:<24} {calls:>8} {:>10} {:>10} {:>10}", "?", "?", "?")?;
        }
        writeln!(
            f,
            "{:<24} {:>8} {} {} {}",
            "TOTAL (covered)",
            self.steps,
            eng(self.total.flops),
            eng(self.total.int_ops),
            eng(self.total.bytes)
        )?;
        write!(f, "coverage: {}/{} steps ({:.1}%)", self.covered, self.steps, self.coverage() * 100.0)
    }
}

/// True iff `name` has a cost formula (all-ones probe shape; formulas are
/// polynomial in their params, so shape never changes coverage).
pub fn covers(name: &str) -> bool {
    kernel_cost(name, Some(&[1; 16]), 1).is_some()
}

/// Fold `steps` into `report`, resolving kernel indices through `names` (the
/// pipeline set the recording `Gpu` was built with). Steps without meta (built
/// behind the facade's back) count as uncovered.
pub fn tally(report: &mut CostReport, names: &[String], steps: &[Step]) {
    for s in steps {
        match s.meta() {
            Some(m) => {
                let name = names.get(m.kernel).map(|s| s.as_str()).unwrap_or("<unknown-kernel>");
                report.record(name, kernel_cost(name, m.params.as_deref(), m.threads));
            }
            None => report.record("<no-meta>", None),
        }
    }
}

/// The cost of one dispatch of kernel `name` with the given uniform `params`
/// and dispatch `threads`, or `None` if the kernel has no formula.
///
/// `params` is `None` for `step_buf` dispatches (uniform in a caller-owned
/// buffer); kernels whose dispatch count fixes the shape (elementwise, and
/// `ce_grad_stats` where threads = rows·vocab) still cost exactly, the rest
/// return `None`. Param layouts mirror each kernel's WGSL `struct Params` —
/// see `crates/kernels/wgsl/<base>.wgsl`.
pub fn kernel_cost(name: &str, params: Option<&[u32]>, threads: u32) -> Option<Cost> {
    // A specialised variant (`base#K=V`) has its base kernel's math.
    let base = name.split('#').next().unwrap_or(name);
    // Uniform word i as u64; None when absent (short slice or step_buf).
    let p = |i: usize| -> Option<u64> { params.and_then(|s| s.get(i).copied()).map(u64::from) };
    // Elementwise kernels: params[0] is the element count and equals the
    // dispatch width, so `threads` is an exact fallback for step_buf.
    let n0 = || p(0).unwrap_or(threads as u64);
    // Causal pair count: sum_{i<t}(i+1).
    let tri = |t: u64| t * (t + 1) / 2;
    let c = |flops: u64, int_ops: u64, bytes: u64| Some(Cost { flops, int_ops, bytes });
    let f = |flops: u64, bytes: u64| Some(Cost { flops, int_ops: 0, bytes });

    match base {
        // ---- GEMMs: out = x[m,k] @ w[n,k]ᵀ (contraction MACs only) ----------
        "matmul" | "matmul_reg" | "matmul_reg2" | "matmul_reg3" | "matmul_reg3_splitk"
        | "matmul_gemv" => {
            let (m, k, n) = (p(0)?, p(1)?, p(2)?);
            f(2 * m * k * n, 4 * (m * k + n * k + m * n))
        }
        // Column tile of a wide output: params [m, k, n_full, n_off, n_tile].
        "matmul_tile" => {
            let (m, k, nt) = (p(0)?, p(1)?, p(4)?);
            f(2 * m * k * nt, 4 * (m * k + nt * k + m * nt))
        }
        // Sparse-MoE expert linear: params [m, k, n, n_experts, e_idx] - same
        // (m, k, n) GEMM shape as `matmul` in the first three slots, plus two
        // routing params that do not change the dispatch shape. A non-routed
        // row early-exits before the K-reduction (moe_linear_gated.wgsl's own
        // header), so the REAL flop count is data-dependent (scales with how
        // many of the `m` rows are actually routed to this expert) - this
        // formula reports the dense upper bound the dispatch shape implies,
        // the same convention `kernel_cost`'s own doc uses for every other
        // kernel here (a pure function of params/threads, never buffer
        // contents).
        "moe_linear_gated" => {
            let (m, k, n) = (p(0)?, p(1)?, p(2)?);
            f(2 * m * k * n, 4 * (m * k + n * k + m * n))
        }
        // dX[m,k] = dY[m,n]·W[n,k]; dW[n,k] += dY[m,n]ᵀ·X[m,k].
        "matmul_dx" | "matmul_dx_reg" => {
            let (m, k, n) = (p(0)?, p(1)?, p(2)?);
            f(2 * m * k * n, 4 * (m * n + n * k + m * k))
        }
        // `_tn` differs only in how dY is INDEXED (already transposed), not in
        // what it reads or computes, so the cost is identical.
        "matmul_dw" | "matmul_dw_reg" | "matmul_dw_reg_tn" | "matmul_dw_reg_splitk" => {
            let (m, k, n) = (p(0)?, p(1)?, p(2)?);
            f(2 * m * k * n, 4 * (m * n + m * k + 2 * n * k))
        }
        // int8 DP4A GEMMs: params [m, kg = K/4, n]. The MACs are INTEGER ops
        // (K = 4·kg of them per output); the dequant epilogue (acc·sx·sw) is fp32.
        // `matmul_i8` is the static-scale sibling of the two below and was the
        // one member of the family without a formula — so any pass using it
        // (the batched int8 forward) could not report a rate at all.
        "matmul_i8" | "matmul_i8_dyn" | "matmul_i8_gemv" => {
            let (m, kg, n) = (p(0)?, p(1)?, p(2)?);
            c(2 * m * n, 8 * m * kg * n, 4 * (m * kg + n * kg + m * n + m + n))
        }
        // q4 W4A8 GEMMs (int8 activation, int4 weight): params [m, k, n] with
        // `k` the LOGICAL (un-divided) K, unlike the int8 family's `kg` --
        // x and w pack a different number of values per u32 for the same K
        // (4 vs 8), so one shared "kg" would be ambiguous about which operand
        // it counts. The MACs are the same K per output as int8 (a nibble
        // multiply-add costs the same logical MAC as a byte one in this
        // roofline accounting); bytes reflect x's [m, k/4] u32 footprint and
        // w's HALF-that [n, k/8] u32 footprint.
        "matmul_q4_dyn" | "matmul_q4_gemv" => {
            let (m, k, n) = (p(0)?, p(1)?, p(2)?);
            c(2 * m * n, 2 * m * k * n, 4 * (m * (k / 4) + n * (k / 8) + m * n + m + n))
        }
        // Activation quantization: params [m, k]; q = clamp(round(x/sx)).
        "quant_pack" => {
            let (m, k) = (p(0)?, p(1)?);
            f(3 * m * k, 5 * m * k + 4 * m)
        }
        // Per-row max|x| scale: params [m, k]. `max_abs_rows` is the
        // cooperative (workgroup-per-row) variant of the same op — same reads,
        // same writes, only the thread mapping differs.
        "max_abs_row" | "max_abs_rows" => {
            let (m, k) = (p(0)?, p(1)?);
            f(m * k, 4 * (m * k + m))
        }

        // ---- paged KV decode/prefill (the SERVING tape) ---------------------
        //
        // These were all uncovered, so `qwen3::serve`'s step could not report a
        // rate at all — and an unrateable pass cannot be ranked.
        //
        // CONVENTION, because these kernels are DATA-dependent in a way the
        // others are not: the work per sequence is `seq_lens[b]`, which lives in
        // a storage buffer, not in `Params`, so `kernel_cost` cannot see it.
        // The formulas therefore use `cap` — the row stride the dispatch is
        // sized for, and the length every sequence actually reaches at full
        // context. That is EXACT for steady-state decode (the case worth
        // optimising) and an OVER-estimate for short sequences, which flatters
        // the rate. Anything profiling a ramp of short sequences must say so.
        //
        // params: [batch, n_heads, group, head_dim, block_size, kv_stride, cap, max_bt(, scale)]
        "paged_decode_scores_batched" | "paged_decode_scores" | "paged_decode_scores_wg" => {
            let (b, nh, grp, hd, cap) = (p(0)?, p(1)?, p(2)?, p(3)?, p(6)?);
            // GQA: `group` query heads SHARE one kv head, so the KV cache is
            // read `nh/group` times, not `nh`. Counting per query head
            // overstated it by exactly `group` (2x here) — and `bytes` is a
            // streaming estimate, "each logical operand once", not a count of
            // per-thread loads.
            let nkv = nh / grp.max(1);
            f(2 * b * nh * cap * hd, 4 * (b * nh * hd + b * nkv * cap * hd + b * nh * cap))
        }
        "paged_decode_apply_batched" | "paged_decode_apply" => {
            let (b, nh, grp, hd, cap) = (p(0)?, p(1)?, p(2)?, p(3)?, p(6)?);
            let nkv = nh / grp.max(1);
            f(2 * b * nh * cap * hd, 4 * (b * nh * cap + b * nkv * cap * hd + b * nh * hd))
        }
        // int8 paged KV: same math, dequantized on read, so the K/V half of the
        // traffic is 1 byte per element plus a per-block scale.
        "paged_decode_scores_i8_batched" => {
            let (b, nh, grp, hd, cap) = (p(0)?, p(1)?, p(2)?, p(3)?, p(6)?);
            let nkv = nh / grp.max(1);
            f(2 * b * nh * cap * hd, 4 * (b * nh * hd + b * nh * cap) + b * nkv * cap * hd)
        }
        "paged_decode_apply_i8_batched" => {
            let (b, nh, grp, hd, cap) = (p(0)?, p(1)?, p(2)?, p(3)?, p(6)?);
            let nkv = nh / grp.max(1);
            f(2 * b * nh * cap * hd, 4 * (b * nh * cap + b * nh * hd) + b * nkv * cap * hd)
        }
        // params: [batch, kv_stride, block_size] — a copy into the paged pool.
        // `paged_kv_append_batched_word` (B9's bf16-packed, one-thread-per-
        // token sibling - see that kernel's own doc comment) shares this
        // EXACT `Params` struct (batch, kv_stride, block_size) and moves the
        // same batch*kv_stride elements, just with a different thread
        // granularity, so the same formula applies.
        "paged_kv_append_batched" | "paged_kv_append" | "paged_kv_append_batched_word" => {
            f(0, 8 * p(0)? * p(1)?)
        }
        "paged_kv_append_i8_clipped_batched" => {
            let (b, kv) = (p(0)?, p(1)?);
            f(2 * b * kv, 5 * b * kv)
        }
        // params: [batch, n_heads, cap] — one softmax row per (b, head).
        "decode_softmax_batched" | "decode_softmax" => {
            let n = p(0)? * p(1)? * p(2)?;
            f(4 * n, 8 * n)
        }
        // params: [n_rows, n_heads, head_dim, row_stride, base] — in-place RoPE
        // on the newly appended token of each sequence.
        "rope_paged" => {
            let n = p(0)? * p(1)? * p(2)?;
            f(4 * n, 8 * n)
        }

        // ---- gathers / scatters / copies (bytes, no flops) ------------------
        // embed: params [d_model, seq_len]; embed_tile adds [v0, v_count].
        "embed" | "embed_tile" => {
            let (d, s) = (p(0)?, p(1)?);
            f(0, 8 * s * d + 4 * s)
        }
        // params [n_rows, d_model, vocab]: per (v,c) scans tokens; adds fire
        // once per looked-up row plus the final accumulate.
        "emb_bwd" => {
            let (rows, d, v) = (p(0)?, p(1)?, p(2)?);
            f(d * (rows + v), 4 * (2 * v * d + rows * d + rows))
        }
        // params [n_idx, d, n_rows_out].
        "row_scatter" => {
            let (ni, d) = (p(0)?, p(1)?);
            f(0, 8 * ni * d + 4 * ni)
        }
        // params [n, base] (compact block copy into/out of the residual).
        "splice" => f(0, 8 * n0()),
        "splice_bwd" => f(0, 12 * n0()),
        "splice_add" => f(n0(), 12 * n0()),
        // params [n, src_base, dst_base]: same op as splice_add, independent offsets.
        "splice_add_offset_src" => f(n0(), 12 * n0()),
        // params [rows, n_experts, top_k]: one thread per row scans up to
        // n_experts gate entries (int compares, no flops) and writes up to
        // top_k ids -- see router_topk_compact.wgsl's own doc.
        "router_topk_compact" => {
            let (rows, e, k) = (p(0)?, p(1)?, p(2)?);
            c(0, rows * e, 4 * (rows * e + rows * k))
        }
        // params [width, row]: one cache-row write.
        "kv_append" => f(0, 8 * p(0)?),
        // params [rows, heads_out, group, hd, ...]: replicate kv heads (copy);
        // backward is the group-sum adjoint.
        "kv_expand" => {
            let (rows, ho, hd) = (p(0)?, p(1)?, p(3)?);
            f(0, 8 * rows * ho * hd)
        }
        "kv_expand_bwd" => {
            let (rows, ho, g, hd) = (p(0)?, p(1)?, p(2)?, p(3)?);
            f(rows * ho * hd, 4 * (rows * ho * hd + rows * ho * hd / g.max(1)))
        }
        // params [total, c, hw] (layout permutations).
        "nlc_nchw" | "nchw_nlc" => f(0, 8 * p(0)?),

        // ---- norms ----------------------------------------------------------
        // params [d_model, rows(, eps)].
        "rmsnorm" | "rmsnorm_eps" | "rmsnorm_rows" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(rows * (4 * d + 2), 4 * (2 * rows * d + d))
        }
        "rms_inv" | "rms_inv_eps" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(rows * (2 * d + 2), 4 * (rows * d + rows))
        }
        "rmsnorm_dx" | "rmsnorm_dx_eps" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(rows * (9 * d + 7), 4 * (4 * rows * d + d))
        }
        "rmsnorm_dw" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(3 * rows * d + d, 4 * (2 * rows * d + rows + 2 * d))
        }
        // params [d_model, n_rows, eps].
        // `*_rows` are the workgroup-per-row variants: same math, same traffic,
        // only the thread mapping differs — so they cost the same (mirrors
        // `rmsnorm` / `rmsnorm_rows` above).
        "layernorm" | "layernorm_rows" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(rows * (8 * d + 5), 4 * (2 * rows * d + 2 * d))
        }
        "ln_stats" | "ln_stats_rows" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(rows * (4 * d + 3), 4 * (rows * d + 2 * rows))
        }
        "layernorm_dx" | "layernorm_dx_rows" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(rows * (15 * d + 8), 4 * (4 * rows * d + d))
        }
        // params [d_model, n_rows].
        "layernorm_dgamma" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(4 * rows * d + d, 4 * (2 * rows * d + 2 * rows + 2 * d))
        }
        "layernorm_dbeta" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(rows * d + d, 4 * (rows * d + 2 * d))
        }

        // ---- RoPE: rows·heads·hd/2 pairs; angle (pow+mul) + cos + sin + the
        // 6-op rotation = 10 per pair. rope2d reads its angles from tables (7).
        "rope_base" | "rope_base_bwd" | "rope_at" => {
            let (rows, h, hd) = (p(0)?, p(1)?, p(2)?);
            f(5 * rows * h * hd, 8 * rows * h * hd)
        }
        // params [rows, heads, half, ...]. rope2d_partial is the same per-pair
        // cost over fewer pairs (half = rot_dim/2 < head_dim/2).
        "rope2d" | "rope2d_partial" => {
            let (rows, h, half) = (p(0)?, p(1)?, p(2)?);
            f(7 * rows * h * half, 24 * rows * h * half)
        }

        // ---- attention, causal (t(t+1)/2 pairs) -----------------------------
        // GQA params [b, h, hkv, t, hd, group]; dense params [b, h, t, hd, ...].
        // `gqa_scores_kmask` = gqa_scores + one additive per-key mask read/add
        // per causal pair (params identical).
        "gqa_scores" | "attn_scores" | "gqa_scores_kmask" => {
            let gqa = base != "attn_scores";
            let kmask = base == "gqa_scores_kmask";
            let (b, h) = (p(0)?, p(1)?);
            let (t, hd) = if gqa { (p(3)?, p(4)?) } else { (p(2)?, p(3)?) };
            let kvh = if gqa { p(2)? } else { h };
            let extra = if kmask { b * h * tri(t) } else { 0 };
            f(
                b * h * tri(t) * (2 * hd + 1) + extra,
                4 * (b * h * t * t + b * t * hd * (h + kvh)) + if kmask { 4 * t } else { 0 },
            )
        }
        // params [b, h, t]: per row i — max, exp-sum, and the exp·inv writes.
        "attn_softmax" => {
            let (b, h, t) = (p(0)?, p(1)?, p(2)?);
            f(b * h * (6 * tri(t) + t), 8 * b * h * t * t)
        }
        "gqa_apply" | "attn_apply" => {
            let gqa = base == "gqa_apply";
            let (b, h) = (p(0)?, p(1)?);
            let (t, hd) = if gqa { (p(3)?, p(4)?) } else { (p(2)?, p(3)?) };
            let kvh = if gqa { p(2)? } else { h };
            f(2 * b * h * hd * tri(t), 4 * (b * h * t * t + b * t * hd * (h + kvh)))
        }
        "gqa_bwd_dscores" | "attn_bwd_dscores" => {
            let gqa = base == "gqa_bwd_dscores";
            let (b, h) = (p(0)?, p(1)?);
            let (t, hd) = if gqa { (p(3)?, p(4)?) } else { (p(2)?, p(3)?) };
            let kvh = if gqa { p(2)? } else { h };
            f(b * h * tri(t) * (2 * hd + 4), 4 * (2 * b * h * t * t + b * t * hd * (h + kvh)))
        }
        "gqa_bwd_dv" | "attn_bwd_dv" => {
            let gqa = base == "gqa_bwd_dv";
            let (b, h) = (p(0)?, p(1)?);
            let (t, hd) = if gqa { (p(3)?, p(4)?) } else { (p(2)?, p(3)?) };
            let kvh = if gqa { p(2)? } else { h };
            f(2 * b * h * hd * tri(t), 4 * (b * h * t * t + b * t * hd * (h + kvh)))
        }
        "gqa_bwd_dq" | "attn_bwd_dq" => {
            let gqa = base == "gqa_bwd_dq";
            let (b, h) = (p(0)?, p(1)?);
            let (t, hd) = if gqa { (p(3)?, p(4)?) } else { (p(2)?, p(3)?) };
            let kvh = if gqa { p(2)? } else { h };
            f(b * h * hd * (2 * tri(t) + t), 4 * (b * h * t * t + b * t * hd * (h + kvh)))
        }
        "gqa_bwd_dk" | "attn_bwd_dk" => {
            let gqa = base == "gqa_bwd_dk";
            let (b, h) = (p(0)?, p(1)?);
            let (t, hd) = if gqa { (p(3)?, p(4)?) } else { (p(2)?, p(3)?) };
            let kvh = if gqa { p(2)? } else { h };
            f(2 * b * h * hd * tri(t) + b * kvh * hd * t, 4 * (b * h * t * t + b * t * hd * (h + kvh)))
        }

        // ---- attention, bidirectional (t² pairs); params [b, h, t(, hd, ...)].
        "attn_scores_bidir" => {
            let (b, h, t, hd) = (p(0)?, p(1)?, p(2)?, p(3)?);
            f(b * h * t * t * (2 * hd + 1), 4 * (b * h * t * t + 2 * b * t * h * hd))
        }
        "attn_softmax_bidir" => {
            let (b, h, t) = (p(0)?, p(1)?, p(2)?);
            f(b * h * t * (6 * t + 1), 8 * b * h * t * t)
        }
        "attn_apply_bidir" | "attn_bwd_dv_bidir" => {
            let (b, h, t, hd) = (p(0)?, p(1)?, p(2)?, p(3)?);
            f(2 * b * h * hd * t * t, 4 * (b * h * t * t + 2 * b * t * h * hd))
        }
        "attn_bwd_dscores_bidir" => {
            let (b, h, t, hd) = (p(0)?, p(1)?, p(2)?, p(3)?);
            f(b * h * t * t * (2 * hd + 4), 4 * (2 * b * h * t * t + 2 * b * t * h * hd))
        }
        "attn_bwd_dq_bidir" | "attn_bwd_dk_bidir" => {
            let (b, h, t, hd) = (p(0)?, p(1)?, p(2)?, p(3)?);
            f(b * h * hd * t * (2 * t + 1), 4 * (b * h * t * t + 2 * b * t * h * hd))
        }
        // Head-major packing for GEMM attention; params [rows, heads_out, group,
        // hd, ...]. One scale-mul + copy per packed element (head_pack_t is the
        // transposed write of the same elements; head_unpack params are
        // [rows, heads, hd, ...] — same element count, pure copy).
        "head_pack" | "head_pack_t" => {
            let (rows, ho, hd) = (p(0)?, p(1)?, p(3)?);
            f(rows * ho * hd, 8 * rows * ho * hd)
        }
        "head_unpack" => {
            let (rows, h, hd) = (p(0)?, p(1)?, p(2)?);
            f(0, 8 * rows * h * hd)
        }
        // Workgroup-cooperative row softmax; params [rows, cols] — the same math
        // as attn_softmax_* (max, exp+sum, normalize per row).
        "softmax_rows" => {
            let (rows, cols) = (p(0)?, p(1)?);
            f(rows * (6 * cols + 1), 8 * rows * cols)
        }
        // Fused flash attention (scores -> softmax -> apply in one tiled pass);
        // params [bsz, n_heads, t, head_dim, ...]. FLOPs are the fused trio's sum;
        // bytes are the ideal-tiling traffic the kernel exists to achieve — the
        // packed QKV read once, O written once, and NO materialised [T,T]
        // scores/probs (that absence is the whole point of the kernel).
        // The whole family shares one formula: same Params, same fused trio,
        // same ideal traffic. They differ only in how the inner loops are
        // scheduled (see `model::block::FlashIds`), which is a constant factor
        // on the achieved rate, not on the work.
        "flash_attn_bidir" | "flash_attn_bidir_split" | "flash_attn_bidir_reg" | "flash_attn_bidir_reg2" => {
            let (b, h, t, hd) = (p(0)?, p(1)?, p(2)?, p(3)?);
            f(
                b * h * t * t * (2 * hd + 1) + b * h * t * (6 * t + 1) + 2 * b * h * hd * t * t,
                4 * (3 * b * t * h * hd + b * t * h * hd),
            )
        }

        // ---- attention, cross (t_dec × t_enc); params [b, h, td, te, hd, ...].
        "attn_scores_cross" => {
            let (b, h, td, te, hd) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?);
            f(b * h * td * te * (2 * hd + 1), 4 * (b * h * td * te + b * hd * h * (td + te)))
        }
        // params [b, h, td, te].
        "attn_softmax_cross" => {
            let (b, h, td, te) = (p(0)?, p(1)?, p(2)?, p(3)?);
            f(b * h * td * (6 * te + 1), 8 * b * h * td * te)
        }
        "attn_apply_cross" | "attn_bwd_dv_cross_acc" => {
            let (b, h, td, te, hd) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?);
            f(2 * b * h * hd * td * te, 4 * (b * h * td * te + b * hd * h * (td + te)))
        }
        "attn_bwd_dscores_cross" => {
            let (b, h, td, te, hd) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?);
            f(b * h * td * te * (2 * hd + 4), 4 * (2 * b * h * td * te + b * hd * h * (td + te)))
        }
        "attn_bwd_dq_cross" => {
            let (b, h, td, te, hd) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?);
            f(b * h * hd * td * (2 * te + 1), 4 * (b * h * td * te + b * hd * h * (td + te)))
        }
        "attn_bwd_dk_cross_acc" => {
            let (b, h, td, te, hd) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?);
            f(b * h * hd * te * (2 * td + 1), 4 * (b * h * td * te + b * hd * h * (td + te)))
        }

        // ---- attention, KV-cache decode (1 query vs t cached) ---------------
        // params [n_heads, group, head_dim, t, cap, kv_stride(, scale)].
        "attn_decode_scores" => {
            let (h, hd, t) = (p(0)?, p(2)?, p(3)?);
            f(h * t * (2 * hd + 1), 4 * (h * hd + t * p(5)? + h * t))
        }
        // params [n_heads, t, cap].
        "attn_decode_apply" => {
            let (h, hd, t) = (p(0)?, p(2)?, p(3)?);
            f(2 * h * hd * t, 4 * (h * t + t * p(5)? + h * hd))
        }

        // ---- elementwise (params [total] == threads) ------------------------
        "add2" | "mul" | "pos_add" | "grad_scale" | "grad_scale_buf" => f(n0(), 12 * n0()),
        "axpy" => f(2 * n0(), 12 * n0()),
        "silu_mul" | "silu_bwd_db" => f(5 * n0(), 12 * n0()),
        "silu_bwd_da" => f(8 * n0(), 16 * n0()),
        "gelu" => f(10 * n0(), 8 * n0()),
        "gelu_bwd" => f(15 * n0(), 12 * n0()),
        // `silu`/`silu_bwd` were uncovered, which is why a VQGAN forward could
        // not report a whole-pass rate at all: one kind without a formula makes
        // the pass numerator partial, and a partial numerator over the full
        // denominator under-reports rather than admitting it cannot tell.
        // x*sigmoid(x): exp + reciprocal + multiply, read one write one.
        "silu" => f(4 * n0(), 8 * n0()),
        // dx = dy * s * (1 + x*(1-s)) — the same sigmoid plus the product rule.
        "silu_bwd" => f(8 * n0(), 12 * n0()),
        // params [total, c, inner]: out[i] *= scale[chan(i)].
        "scale_chan" => {
            let total = p(0)?;
            f(total, 4 * (2 * total + p(1)?))
        }
        // params [total, m]: y[i] = s[i/m] * x[i] (per-row scalar scale, EDM
        // c_in/c_skip/c_out/lambda(sigma) row factors - see scale_row.wgsl).
        // One multiply per element; reads x (total) + the per-row scale array
        // (total/m rows) once each, writes y (total) once.
        "scale_row" => {
            let total = p(0)?;
            let m = p(1)?.max(1);
            f(total, 4 * (2 * total + total / m))
        }
        // params [m, n]: out[m,n] += bias[n] / dbias[n] += Σ_m dy[m,n].
        "bias_add" => {
            let (m, n) = (p(0)?, p(1)?);
            f(m * n, 4 * (2 * m * n + n))
        }
        "bias_grad" => {
            let (m, n) = (p(0)?, p(1)?);
            f(m * n, 4 * (m * n + 2 * n))
        }
        // params [rows, dim, rows_per_cond]: y[r,d] = x[r,d] + g[k,d]*h[r,d],
        // k = r/rows_per_cond - the per-token/per-forward gated residual add
        // `ltxv::block::gate_row` dispatches (one gate row per token at
        // rows_per_cond=1, one shared row at rows_per_cond=rows).
        "gate_row" => {
            let (rows, dim, rpc) = (p(0)?, p(1)?, p(2)?.max(1));
            let g_rows = rows.div_ceil(rpc);
            f(2 * rows * dim, 4 * (3 * rows * dim + g_rows * dim))
        }
        // params [b, t, d_model]: dpos[i,c] += Σ_b dx.
        "pos_bwd" => {
            let (b, t, d) = (p(0)?, p(1)?, p(2)?);
            f(t * d * (b + 1), 4 * (b * t * d + 2 * t * d))
        }

        // ---- cross-entropy; v = vocab / u_bins ------------------------------
        // params [n_rows, u_bins, ignore].
        "ce_value" | "ce_value_masked" => {
            let (rows, v) = (p(0)?, p(1)?);
            f(rows * (4 * v + 3), 4 * (rows * v + 2 * rows))
        }
        // params [n_rows, u_bins, ignore, count]: per ELEMENT softmax recompute
        // — genuinely O(rows·v²) as dispatched (ce_grad_stats is the O(rows·v) fix).
        "ce_grad" | "ce_grad_masked" => {
            let (rows, v) = (p(0)?, p(1)?);
            f(rows * v * (4 * v + 6), 4 * (2 * rows * v + rows))
        }
        "ce_stats" => {
            let (rows, v) = (p(0)?, p(1)?);
            f(4 * rows * v, 4 * (rows * v + 3 * rows))
        }
        // O(1) per element; threads = rows·v exactly, so the step_buf path
        // (params in a reused uniform buffer) still costs exactly.
        "ce_grad_stats" => {
            let total = p(0).and_then(|r| p(1).map(|v| r * v)).unwrap_or(threads as u64);
            f(5 * total, 8 * total)
        }

        // ---- optimizer / grad plumbing --------------------------------------
        // params [numel, pad, lr, b1, b2, eps, wd, bc1, bc2].
        "adamw" => f(12 * p(0)?, 28 * p(0)?),
        // params [numel, slot].
        "gradnorm_sq" => f(2 * p(0)?, 4 * p(0)?),
        // The roofline probe itself (params [n, iters, c, d]): `iters` passes of
        // 8 independent FMAs per thread, one read and one write of traffic. It
        // is costed like anything else so the probe is measured by the same
        // accounting it produces the denominator for.
        "roof_fma" => {
            let (n, iters) = (p(0)?, p(1)?);
            f(n * iters * 16, 8 * n)
        }
        // The int8 sibling: 8 `dot4I8Packed` per iteration, each a 4-wide dot
        // (4 multiplies + 4 adds) = 64 INTEGER ops, counted the same way the
        // int8 GEMMs are so the two are comparable.
        "roof_dp4a" => {
            let (n, iters) = (p(0)?, p(1)?);
            c(0, n * iters * 64, 8 * n)
        }
        // Scalar loss values: one serial reduction over the whole tensor.
        // `mse_value` reads prediction and target, `masked_l1` reads both plus
        // its mask. Their FLOPs are trivial; they are here so a pass containing
        // them can still report a rate at all.
        "mse_value" => f(3 * p(0)?, 8 * p(0)?),
        "masked_l1" => f(2 * n0(), 12 * n0()),
        // …and their adjoints. `mse_grad` is 2(pred-target)/n over two reads
        // and a write; `masked_l1_grad` is sign(diff)*scale over three.
        "mse_grad" => f(3 * p(0)?, 12 * p(0)?),
        "masked_l1_grad" => f(3 * n0(), 16 * n0()),
        // params [numel, out_off, n_wg] — same work as gradnorm_sq, spread over
        // n_wg workgroups; the n_wg partials it writes are the only extra bytes.
        "gradnorm_part" => f(2 * p(0)?, 4 * p(0)? + 4 * p(2)?),
        // params [n_params, max_norm, extra_scale].
        "clip_coef" => f(p(0)? + 5, 4 * p(0)?),
        // params [n_parts, max_norm, extra_scale] — same fold, 64 threads.
        "clip_coef_wg" => f(p(0)? + 5, 4 * p(0)?),

        // ---- conv2d family: params [N, Cin, H, W, Cout, K, stride, pad, Ho, Wo].
        //
        // These were UNCOVERED until 2026-08-06, which meant `conv_bias_reg` —
        // 89.5% of a VQGAN forward — reported no rate at all and the pass-level
        // GFLOP/s was a fiction. Formulas mirror the conv1d family above.
        "conv2d" | "conv_bias" | "conv_bias_reg" | "conv_act" | "conv_act_bn" => {
            let (n, cin, h, w, cout, k, ho, wo) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(5)?, p(8)?, p(9)?);
            f(2 * n * cout * ho * wo * cin * k * k, 4 * (n * cin * h * w + cout * cin * k * k + n * cout * ho * wo))
        }
        // The GATHER form: one invocation per INPUT element, reducing Cout*K*K.
        // Counted as the work actually done (as `conv1d_dx` does), not the
        // algorithmic minimum — the point of the number is to expose the cost.
        "conv2d_dx" => {
            let (n, cin, h, w, cout, k, ho, wo) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(5)?, p(8)?, p(9)?);
            f(2 * n * cin * h * w * cout * k * k, 4 * (n * cout * ho * wo + cout * cin * k * k + n * cin * h * w))
        }
        "conv2d_dw" => {
            let (n, cin, h, w, cout, k, ho, wo) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(5)?, p(8)?, p(9)?);
            f(2 * cout * cin * k * k * n * ho * wo, 4 * (n * cout * ho * wo + n * cin * h * w + 2 * cout * cin * k * k))
        }
        // im2col / col2im are LAYOUT, not arithmetic: `im2col_at` moves
        // cnt*cinkk floats, `col2im` sums K*K taps per input element.
        // params: im2col_at [cin,h,w,k,stride,pad,ho,wo,cinkk,pos0,cnt]
        //         col2im   [N,Cin,H,W,K,stride,pad,Ho,Wo,cinkk]
        "im2col_at" => {
            let (cinkk, cnt) = (p(8)?, p(10)?);
            f(0, 8 * cnt * cinkk)
        }
        "im2col" => f(0, 8 * n0()),
        // params [rc, slices]: fold the split-K weight-gradient partials.
        // Reads `rc*slices` and writes `rc` — the tail the split-K dW GEMM pays
        // to avoid atomics.
        "dw_splitk_reduce" => {
            let (rc, slices) = (p(0)?, p(1)?);
            f(rc * slices, 4 * (rc * slices + rc))
        }
        "col2im" => {
            let (n, cin, h, w, k) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?);
            let elems = n * cin * h * w;
            f(elems * k * k, 4 * (elems * k * k + elems))
        }

        // ---- GroupNorm family: params [N, C, H, W, G(, ...)].
        // Two passes over the data for the statistics, an affine for the apply.
        "gn_stats" | "gn_stats_wg" | "gn_part" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)?;
            f(2 * n, 4 * n)
        }
        "gn_stats2" => {
            let (g, pp) = (p(4)?, p(5)?);
            f(2 * g * pp, 8 * g * pp)
        }
        "gn_apply" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)?;
            f(3 * n, 4 * (2 * n + 2 * p(1)?))
        }
        // The backward reductions: gn_dsum reads x and dy, the per-channel pair
        // reduce N*H*W each, gn_dx recombines.
        "gn_dsum" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)?;
            f(4 * n, 8 * n)
        }
        "gn_dgamma" | "gn_dbeta" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)?;
            f(2 * n, 8 * n)
        }
        // The two-stage barrier-free rewrites of the three reductions above
        // (params [N,C,H,W,G,P], P = partials per group). Stage 1 streams the
        // data exactly as its serial ancestor did; stage 2 folds G*P partials,
        // so the pair costs the ancestor's traffic plus a negligible tail.
        // Uncovered until now, which cost the VQGAN backward its pass rate —
        // `gn_dsum_part` + `gn_dgb_part` alone are 5.5% of that pass.
        "gn_dsum_part" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)?;
            f(4 * n, 8 * n)
        }
        "gn_dgb_part" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)?;
            f(2 * n, 8 * n)
        }
        // Stage 2 of each pair folds `G*P` (resp. `C*P`) partials.
        "gn_dsum2" => {
            let (g, pp) = (p(4)?, p(5)?);
            f(2 * g * pp, 8 * g * pp)
        }
        "gn_dgb2" => {
            let (c, pp) = (p(1)?, p(5)?);
            f(2 * c * pp, 8 * c * pp)
        }
        "gn_dx" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)?;
            f(5 * n, 12 * n)
        }
        "upsample2" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)?;
            f(0, 4 * (n + 4 * n))
        }
        // Its adjoint sums the 4 upsampled taps back into each input element:
        // reads 4n, writes n, 3 adds. Params are the INPUT shape [N,C,H,W].
        "upsample2_dx" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)?;
            f(3 * n, 4 * (4 * n + n))
        }
        // Pure movement along the channel axis. `concat2` copies both sources
        // once ([N,Ca,Cb,H,W]); `concat_split` copies one slice
        // ([N,Ctot,Csrc,c_off,H,W]) — read once, write once, no arithmetic.
        "concat2" => {
            let n = p(0)? * (p(1)? + p(2)?) * p(3)? * p(4)?;
            f(0, 8 * n)
        }
        "concat_split" => {
            let n = p(0)? * p(2)? * p(4)? * p(5)?;
            f(0, 8 * n)
        }

        // ---- conv1d family: params [N, Cin, L, Cout, K, stride, pad, dil, G, Lo].
        "conv1d" => {
            let (n, cin, l, cout, k, g, lo) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(8)?, p(9)?);
            let cin_g = cin / g.max(1);
            f(2 * n * cout * lo * cin_g * k, 4 * (n * cin * l + cout * cin_g * k + n * cout * lo))
        }
        "conv1d_dx" => {
            let (n, cin, l, cout, k, g, lo) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(8)?, p(9)?);
            let (cin_g, cout_g) = (cin / g.max(1), cout / g.max(1));
            f(2 * n * cin * l * cout_g * k, 4 * (n * cout * lo + cout * cin_g * k + n * cin * l))
        }
        "conv1d_dw" => {
            let (n, cin, l, cout, k, g, lo) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(8)?, p(9)?);
            let cin_g = cin / g.max(1);
            f(2 * cout * cin_g * k * n * lo, 4 * (n * cout * lo + n * cin * l + 2 * cout * cin_g * k))
        }
        // Causal depthwise Conv1d, single-token decode step: params [n, c, k].
        // `n*c = rows` threads, each a `K`-tap MAC (the `conv1d` family's
        // `Cin_g=1` depthwise case at `L=1`, `w`'s footprint counted once as
        // `c*k` -- same convention as `conv1d`'s own weight term, not
        // multiplied out per row) plus a `K-1`-element ring-buffer shift
        // (`hist` read AND written, so its footprint counts twice).
        "causal_conv1d_step" => {
            let (n, c, k) = (p(0)?, p(1)?, p(2)?);
            let rows = n * c;
            let km1 = k.saturating_sub(1);
            f(2 * rows * k, 4 * (2 * rows + c * k + 2 * rows * km1))
        }

        // ---- bmm/bmm_acc: batched matmul, both operands vary per batch.
        // Params [batch, m, k, n, trans_a, trans_b, alpha, a_off, b_off, out_off].
        // Same contraction-MAC accounting as the GEMM family above, times `batch`.
        "bmm" | "bmm_acc" => {
            let (batch, m, k, n) = (p(0)?, p(1)?, p(2)?, p(3)?);
            f(2 * batch * m * k * n, 4 * batch * (m * k + n * k + m * n))
        }

        // ---- Gated DeltaNet (GDN) chunk recurrence -------------------------
        // `exp`/`sub`: plain elementwise, `total` is params[0] (or `n0()` for
        // `sub`, whose extra offset params don't shift its position).
        "exp" => f(n0(), 8 * n0()),
        "sub" => f(n0(), 12 * n0()),
        // One row of the per-chunk cumsum: params [bhc, c_len, i]; one add per
        // row, dispatch width is `bhc` (params[0]), NOT threads (`i` grows
        // per host-loop call so `threads` alone would undercount the row width).
        "gdn_chunk_cumsum_step" => {
            let bhc = p(0)?;
            f(bhc, 12 * bhc)
        }
        // Pure data movement, token-major <-> chunk-major (qwen35's GDN layer
        // boundary): params [b, h, n_chunks, c, d, to_chunk_major].
        "gdn_layout_permute" => {
            let n = p(0)? * p(1)? * p(2)? * p(3)? * p(4)?;
            f(0, 8 * n)
        }
        // g[row,h] = -exp(A_log[h]) * softplus(a_proj[row,h]+dt_bias[h]):
        // params [rows, num_v_heads]; one add, one stable-softplus (max+abs+exp+
        // log), one exp, one mul per output element.
        "gdn_decay_gate" => {
            let n = p(0)? * p(1)?;
            f(6 * n, 4 * (2 * n + n))
        }
        // Backward of `gdn_decay_gate` w.r.t. `a_proj`: params [rows,
        // num_v_heads]; one add (x), one sigmoid (~3 ops), one exp(A_log),
        // one mul, one negate per output element. Reads a_proj/A_log/
        // dt_bias/d_g (4 operands, though A_log/dt_bias are only
        // num_v_heads-wide), writes d_a_proj (1).
        "gdn_decay_gate_bwd" => {
            let n = p(0)? * p(1)?;
            f(7 * n, 4 * (4 * n + n))
        }
        // decay_mask[row,i,j] = exp(g_cs[i]-g_cs[j]) for j<=i: params [bhc, c_len].
        // Every element is written; roughly half the exp/sub pairs are live.
        "gdn_decay_mask" => {
            let (bhc, c) = (p(0)?, p(1)?);
            let n = bhc * c * c;
            f(2 * n, 4 * (2 * n))
        }
        // Elementwise mask+multiply over the same [bhc,c,c] shape.
        "gdn_mask_strict_lower" => {
            let (bhc, c) = (p(0)?, p(1)?);
            let n = bhc * c * c;
            f(n, 4 * (3 * n))
        }
        // One row `i` of the UT-transform forward substitution: params
        // [bhc, c_len, i]; threads = bhc*i, and each thread's serial reduction
        // is at most `i` MACs (exact per this row's dispatch width).
        "gdn_ut_step" => {
            let (bhc, i) = (p(0)?, p(2)?);
            f(2 * bhc * i * i, 4 * (2 * bhc * i * i))
        }
        "gdn_add_identity" => {
            let (bhc, c) = (p(0)?, p(1)?);
            f(bhc * c, 8 * bhc * c)
        }
        // params [total, m, x_off, s_off, alpha]; one mul + one scale-mul.
        "gdn_row_scale_off" => f(2 * n0(), 12 * n0()),
        // params [bh, c_len, g_cs_off]; one sub + one exp per row.
        "gdn_decay_scale" => {
            let (bh, c) = (p(0)?, p(1)?);
            f(2 * bh * c, 4 * (2 * bh * c))
        }
        // params [bh, dk, dv, c_len, g_cs_off]; one exp (amortised over dk*dv
        // threads, but every thread still costs one mul) + one mul per element.
        "gdn_state_decay" => {
            let (bh, dk, dv) = (p(0)?, p(1)?, p(2)?);
            let n = bh * dk * dv;
            f(2 * n, 8 * n)
        }

        // ---- Gated DeltaNet (GDN) backward -------------------------------
        // Generic per-row dot product: params [rows, d, a_off, b_off, alpha];
        // one MAC per element plus one closing scalar multiply per row.
        "row_dot" => {
            let (rows, d) = (p(0)?, p(1)?);
            f(2 * rows * d + rows, 4 * (2 * rows * d + rows))
        }
        // Reverse of the per-chunk cumsum: identical shape to
        // `gdn_chunk_cumsum_step` (one add per row, dispatch width `bhc`).
        "gdn_chunk_reverse_cumsum_step" => {
            let bhc = p(0)?;
            f(bhc, 12 * bhc)
        }
        // UT-transform backward, both halves: params [bhc, c_len, i]; same
        // triangular dispatch shape as `gdn_ut_step` (threads = bhc*i, each
        // thread's serial loop at most `i` MACs).
        "gdn_ut_bwd_dattn0" | "gdn_ut_bwd_dtmat" => {
            let (bhc, i) = (p(0)?, p(2)?);
            f(2 * bhc * i * i, 4 * (2 * bhc * i * i))
        }
        // Elementwise mask-multiply backward over [bhc,c,c]; two muls + one
        // add live on the strictly-lower half, nothing outside it.
        "gdn_mask_strict_lower_bwd" => {
            let (bhc, c) = (p(0)?, p(1)?);
            let n = bhc * c * c;
            f(2 * n, 4 * (4 * n))
        }
        // Row-sum or column-sum over [bhc,c,c] (params [bhc, c_len, mode]);
        // dispatched twice, each call a triangular reduction bounded by c_len.
        "gdn_decay_mask_bwd" => {
            let (bhc, c) = (p(0)?, p(1)?);
            f(2 * bhc * c * c, 4 * (2 * bhc * c * c))
        }
        // params [bh, c_len, g_cs_off]; one mul + one sub per element.
        "gdn_decay_scale_bwd" => {
            let (bh, c) = (p(0)?, p(1)?);
            let n = bh * c;
            f(2 * n, 4 * (3 * n))
        }
        // params [bh, c_len, g_cs_off]; same per-row total work as
        // `gdn_decay_scale`'s own forward (one thread per bh, loop c_len).
        "gdn_decay_scale_bwd_last" => {
            let (bh, c) = (p(0)?, p(1)?);
            f(2 * bh * c, 4 * (2 * bh * c))
        }
        // params [bh, dk, dv, c_len, g_cs_off]; one thread per bh, loop
        // dk*dv MACs plus one exp + one mul + one add for the closing term.
        "gdn_state_decay_bwd_dscale" => {
            let (bh, dk, dv) = (p(0)?, p(1)?, p(2)?);
            let n = bh * dk * dv;
            f(2 * n + 3 * bh, 4 * (2 * n + 3 * bh))
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(name: &str, params: &[u32], threads: u32) -> Cost {
        kernel_cost(name, Some(params), threads).expect(name)
    }

    /// Hand-computed expectations for the load-bearing formulas.
    #[test]
    fn gemm_costs() {
        // out[8,6] = x[8,4] @ w[6,4]ᵀ: 8*6 outputs × 4 MACs × 2 = 384 flops.
        assert_eq!(cost("matmul", &[8, 4, 6], 48).flops, 384);
        assert_eq!(cost("matmul_reg2", &[8, 4, 6], 48).flops, 384);
        assert_eq!(cost("matmul_gemv", &[8, 4, 6], 48).flops, 384);
        // A specialised variant costs as its base.
        assert_eq!(cost("matmul_reg2#TILE=64", &[8, 4, 6], 48).flops, 384);
        // dX/dW backward GEMMs contract the same volume.
        assert_eq!(cost("matmul_dx", &[8, 4, 6, 0], 32).flops, 384);
        assert_eq!(cost("matmul_dw", &[8, 4, 6], 24).flops, 384);
        // Column tile: n_tile (last param) is the width, not n_full.
        assert_eq!(cost("matmul_tile", &[8, 4, 100, 0, 6], 48).flops, 384);
        // Bytes: each operand streamed once.
        assert_eq!(cost("matmul", &[8, 4, 6], 48).bytes, 4 * (32 + 24 + 48));
    }

    #[test]
    fn int8_gemm_is_int_ops() {
        // [m=2, kg=8 (K=32), n=3]: 2*3 outputs × 32 int8 MACs × 2 = 384 int ops;
        // fp32 only in the dequant epilogue (2 per output).
        let c = cost("matmul_i8_dyn", &[2, 8, 3], 6);
        assert_eq!(c.int_ops, 384);
        assert_eq!(c.flops, 12);
        let g = cost("matmul_i8_gemv", &[2, 8, 3], 3 * 64);
        assert_eq!(g.int_ops, 384);
    }

    #[test]
    fn q4_gemm_is_int_ops() {
        // [m=2, k=32 (LOGICAL, not kg), n=3]: 2*3 outputs x 32 int MACs x 2 =
        // 384 int ops -- same MAC count as int8_gemm_is_int_ops's [2,8,3]
        // (kg=8 there means K=32 too), because a q4 kernel still contracts
        // every logical K element once, just via a nibble instead of a byte.
        let c = cost("matmul_q4_dyn", &[2, 32, 3], 6);
        assert_eq!(c.int_ops, 384);
        assert_eq!(c.flops, 12);
        let g = cost("matmul_q4_gemv", &[2, 32, 3], 3 * 64);
        assert_eq!(g.int_ops, 384);
        // Bytes: x is [m, k/4] u32 (int8), w is HALF that density, [n, k/8].
        assert_eq!(c.bytes, 4 * (2 * 8 + 3 * 4 + 2 * 3 + 2 + 3));
    }

    #[test]
    fn attention_costs_are_causal() {
        // gqa_scores [b=1, h=2, hkv=1, t=4, hd=8, g=2]: 2 heads × 10 causal
        // pairs × (2*8+1) = 340.
        assert_eq!(cost("gqa_scores", &[1, 2, 1, 4, 8, 2], 32).flops, 2 * 10 * 17);
        // Dense layout [b, h, t, hd, ...] costs identically at hkv == h.
        assert_eq!(cost("attn_scores", &[1, 2, 4, 8, 96, 0, 32], 32).flops, 2 * 10 * 17);
        // Bidirectional pays t² not t(t+1)/2.
        assert_eq!(cost("attn_scores_bidir", &[1, 2, 4, 8, 96, 0, 32], 32).flops, 2 * 16 * 17);
        // apply: 2·hd per causal pair.
        assert_eq!(cost("gqa_apply", &[1, 2, 1, 4, 8, 2], 64).flops, 2 * 2 * 8 * 10);
        // softmax: 6 ops per causal pair + 1 div per row.
        assert_eq!(cost("attn_softmax", &[1, 2, 4], 8).flops, 2 * (6 * 10 + 4));
        // cross: t_dec × t_enc rectangle.
        assert_eq!(cost("attn_scores_cross", &[1, 2, 3, 5, 8, 96, 64, 0, 0], 30).flops, 2 * 15 * 17);
    }

    #[test]
    fn norm_and_elementwise_costs() {
        // rmsnorm [d=16, rows=3]: 3 × (4*16+2) = 198.
        assert_eq!(cost("rmsnorm", &[16, 3], 3).flops, 198);
        assert_eq!(cost("rmsnorm_rows", &[16, 3], 3 * 64).flops, 198, "cooperative variant, same math");
        assert_eq!(cost("silu_mul", &[100], 100).flops, 500);
        assert_eq!(cost("add2", &[7], 7).flops, 7);
        // embed moves bytes, no flops.
        let e = cost("embed", &[8, 5], 40);
        assert_eq!((e.flops, e.bytes), (0, 8 * 5 * 8 + 4 * 5));
    }

    #[test]
    fn conv1d_costs() {
        // [N=2, Cin=6, L=10, Cout=4, K=3, stride=1, pad=1, dil=1, G=2, Lo=10]:
        // 2*4*10 outputs × (6/2)*3 taps × 2 = 1440.
        assert_eq!(cost("conv1d", &[2, 6, 10, 4, 3, 1, 1, 1, 2, 10], 80).flops, 1440);

        // conv2d: [N,Cin,H,W,Cout,K,stride,pad,Ho,Wo]. 2*N*Cout*Ho*Wo*Cin*K*K
        // = 2*1*4*8*8*2*3*3 = 9216.
        assert_eq!(cost("conv_bias_reg", &[1, 2, 8, 8, 4, 3, 1, 1, 8, 8], 256).flops, 9216);
        // dW does the same MACs, reduced over output positions instead.
        assert_eq!(cost("conv2d_dw", &[1, 2, 8, 8, 4, 3, 1, 1, 8, 8], 72).flops, 9216);
        // dX is the GATHER form: per INPUT element, reduce Cout*K*K.
        assert_eq!(cost("conv2d_dx", &[1, 2, 8, 8, 4, 3, 1, 1, 8, 8], 128).flops, 9216);
        // col2im sums K*K taps per input element: 1*2*8*8*9 = 1152.
        assert_eq!(cost("col2im", &[1, 2, 8, 8, 3, 1, 1, 8, 8, 18], 128).flops, 1152);
        // im2col_at is pure movement: 8 bytes per moved float (read + write).
        assert_eq!(cost("im2col_at", &[2, 8, 8, 3, 1, 1, 8, 8, 18, 0, 64], 1152).bytes, 8 * 64 * 18);
        assert_eq!(cost("im2col_at", &[2, 8, 8, 3, 1, 1, 8, 8, 18, 0, 64], 1152).flops, 0);
        // GroupNorm statistics: two passes over N*C*H*W = 128.
        assert_eq!(cost("gn_stats_wg", &[1, 2, 8, 8, 2], 512).flops, 256);
        assert_eq!(cost("gn_apply", &[1, 2, 8, 8, 2], 128).flops, 384);

        // The whole point of F0: every kernel the conv models dispatch must
        // have a formula, or a profile of them reports a fiction.
        //
        // The second row is what a *coverage-honest* whole-pass rate exposed:
        // with any one kind uncovered the pass numerator is partial, and a
        // partial numerator over the full denominator silently under-reports.
        // Ten of the VQGAN backward's 26 kinds were in that state, which is why
        // its published "5.4% of peak" was a fiction in the other direction.
        for k in [
            "conv_bias_reg", "conv2d_dx", "conv2d_dw", "col2im", "im2col_at", "gn_stats",
            "gn_stats_wg", "gn_part", "gn_stats2", "gn_apply", "gn_dsum", "gn_dgamma",
            "gn_dbeta", "gn_dx", "upsample2",
            // the VQGAN training step's remaining kinds
            "silu", "silu_bwd", "scale_chan", "dw_splitk_reduce", "gn_dsum_part", "gn_dsum2",
            "gn_dgb_part", "gn_dgb2", "mse_value", "masked_l1", "upsample2_dx", "concat2",
            "concat_split", "matmul_i8", "roof_fma", "roof_dp4a", "mse_grad", "masked_l1_grad",
            "matmul_reg3_splitk",
            // the served paged tape
            "paged_decode_scores_batched", "paged_decode_scores_wg", "paged_decode_apply_batched",
            "paged_kv_append_batched", "decode_softmax_batched", "rope_paged",
            "paged_decode_scores_i8_batched", "paged_decode_apply_i8_batched",
            "paged_kv_append_i8_clipped_batched",
        ] {
            assert!(covers(k), "kernel `{k}` has no cost formula");
        }
        assert_eq!(cost("conv1d_dw", &[2, 6, 10, 4, 3, 1, 1, 1, 2, 10], 36).flops, 1440);
    }

    /// A RATCHET over the whole kernel tree, not another hand-maintained list.
    ///
    /// The list above only fails when someone remembers to add a name to it, so
    /// on its own it cannot stop a new kernel landing unmeasurable — and an
    /// unmeasurable kernel silently removes its pass's ability to report a rate
    /// at all (see `profile::PassProfile::gflops`). This asserts the *fraction*
    /// of `kernels::ALL` with a formula never falls, so adding kernels without
    /// formulas is visible the moment it happens rather than at the next
    /// profile. Raise `FLOOR` whenever coverage rises; never lower it.
    #[test]
    fn cost_coverage_over_the_kernel_tree_never_regresses() {
        // Measured 2026-08-06: 150 of 357. Deliberately a floor and not an
        // equality — adding a formula must not require editing a test.
        const FLOOR: usize = 150;
        let total = kernels::ALL.len();
        let covered = kernels::ALL.iter().filter(|(n, _)| covers(n)).count();
        let uncovered: Vec<&str> =
            kernels::ALL.iter().filter(|(n, _)| !covers(n)).map(|(n, _)| *n).collect();
        println!(
            "cost coverage: {covered}/{total} kernels ({:.1}%)\nuncovered: {uncovered:?}",
            100.0 * covered as f64 / total as f64
        );
        assert!(
            covered >= FLOOR,
            "cost coverage fell to {covered}/{total}; floor is {FLOOR}. \
             A kernel without a formula makes every pass that dispatches it unable \
             to report a rate. Uncovered: {uncovered:?}"
        );
    }

    #[test]
    fn step_buf_fallbacks_and_unknowns() {
        // ce_grad_stats without params (step_buf): threads = rows·v exactly.
        assert_eq!(kernel_cost("ce_grad_stats", None, 60).unwrap().flops, 300);
        // With params, identical.
        assert_eq!(kernel_cost("ce_grad_stats", Some(&[12, 5, 0xFFFF_FFFF, 0]), 60).unwrap().flops, 300);
        // Unknown kernels are None — uncovered, never zero.
        assert!(kernel_cost("no_such_kernel", Some(&[1, 2, 3]), 10).is_none());
        assert!(!covers("no_such_kernel"));
        assert!(covers("matmul"));
        // A GEMM recorded without params cannot be costed.
        assert!(kernel_cost("matmul", None, 48).is_none());
    }

    #[test]
    fn report_accounting_is_honest() {
        let mut r = CostReport::default();
        r.record("matmul", kernel_cost("matmul", Some(&[2, 3, 4]), 8));
        r.record("mystery", None);
        assert_eq!(r.steps, 2);
        assert_eq!(r.covered, 1);
        assert_eq!(r.total.flops, 48);
        assert_eq!(r.uncovered.get("mystery"), Some(&1));
        assert!((r.coverage() - 0.5).abs() < 1e-12);
        let mut m = CostReport::default();
        m.merge(&r);
        m.merge(&r);
        assert_eq!(m.total.flops, 96);
        assert_eq!(m.by_kernel.get("matmul").unwrap().calls, 2);
    }
}
