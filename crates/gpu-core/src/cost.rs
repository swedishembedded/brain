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
        "matmul" | "matmul_reg" | "matmul_reg2" | "matmul_gemv" => {
            let (m, k, n) = (p(0)?, p(1)?, p(2)?);
            f(2 * m * k * n, 4 * (m * k + n * k + m * n))
        }
        // Column tile of a wide output: params [m, k, n_full, n_off, n_tile].
        "matmul_tile" => {
            let (m, k, nt) = (p(0)?, p(1)?, p(4)?);
            f(2 * m * k * nt, 4 * (m * k + nt * k + m * nt))
        }
        // dX[m,k] = dY[m,n]·W[n,k]; dW[n,k] += dY[m,n]ᵀ·X[m,k].
        "matmul_dx" | "matmul_dx_reg" => {
            let (m, k, n) = (p(0)?, p(1)?, p(2)?);
            f(2 * m * k * n, 4 * (m * n + n * k + m * k))
        }
        "matmul_dw" | "matmul_dw_reg" => {
            let (m, k, n) = (p(0)?, p(1)?, p(2)?);
            f(2 * m * k * n, 4 * (m * n + m * k + 2 * n * k))
        }
        // int8 DP4A GEMMs: params [m, kg = K/4, n]. The MACs are INTEGER ops
        // (K = 4·kg of them per output); the dequant epilogue (acc·sx·sw) is fp32.
        "matmul_i8_dyn" | "matmul_i8_gemv" => {
            let (m, kg, n) = (p(0)?, p(1)?, p(2)?);
            c(2 * m * n, 8 * m * kg * n, 4 * (m * kg + n * kg + m * n + m + n))
        }
        // Activation quantization: params [m, k]; q = clamp(round(x/sx)).
        "quant_pack" => {
            let (m, k) = (p(0)?, p(1)?);
            f(3 * m * k, 5 * m * k + 4 * m)
        }
        // Per-row max|x| scale: params [m, k].
        "max_abs_row" => {
            let (m, k) = (p(0)?, p(1)?);
            f(m * k, 4 * (m * k + m))
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
        "layernorm" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(rows * (8 * d + 5), 4 * (2 * rows * d + 2 * d))
        }
        "ln_stats" => {
            let (d, rows) = (p(0)?, p(1)?);
            f(rows * (4 * d + 3), 4 * (rows * d + 2 * rows))
        }
        "layernorm_dx" => {
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
        // params [rows, heads, half, ...].
        "rope2d" => {
            let (rows, h, half) = (p(0)?, p(1)?, p(2)?);
            f(7 * rows * h * half, 24 * rows * h * half)
        }

        // ---- attention, causal (t(t+1)/2 pairs) -----------------------------
        // GQA params [b, h, hkv, t, hd, group]; dense params [b, h, t, hd, ...].
        "gqa_scores" | "attn_scores" => {
            let gqa = base == "gqa_scores";
            let (b, h) = (p(0)?, p(1)?);
            let (t, hd) = if gqa { (p(3)?, p(4)?) } else { (p(2)?, p(3)?) };
            let kvh = if gqa { p(2)? } else { h };
            f(b * h * tri(t) * (2 * hd + 1), 4 * (b * h * t * t + b * t * hd * (h + kvh)))
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
        "decode_softmax" => {
            let (h, t) = (p(0)?, p(1)?);
            f(h * (6 * t + 1), 8 * h * t)
        }
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
        // params [m, n]: out[m,n] += bias[n] / dbias[n] += Σ_m dy[m,n].
        "bias_add" => {
            let (m, n) = (p(0)?, p(1)?);
            f(m * n, 4 * (2 * m * n + n))
        }
        "bias_grad" => {
            let (m, n) = (p(0)?, p(1)?);
            f(m * n, 4 * (m * n + 2 * n))
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
        // params [n_params, max_norm, extra_scale].
        "clip_coef" => f(p(0)? + 5, 4 * p(0)?),

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
        assert_eq!(cost("conv1d_dw", &[2, 6, 10, 4, 3, 1, 1, 1, 2, 10], 36).flops, 1440);
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
