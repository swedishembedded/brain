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

    /// `k` repetitions of the same work - a denoise loop's per-step cost times
    /// its step count.
    pub fn scaled(self, k: u64) -> Cost {
        Cost { flops: self.flops * k, int_ops: self.int_ops * k, bytes: self.bytes * k }
    }

    /// `self - o`, or `None` if `o` is not contained in `self`.
    pub fn checked_sub(self, o: Cost) -> Option<Cost> {
        Some(Cost {
            flops: self.flops.checked_sub(o.flops)?,
            int_ops: self.int_ops.checked_sub(o.int_ops)?,
            bytes: self.bytes.checked_sub(o.bytes)?,
        })
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

    /// `k` repetitions of this whole report - what turns one denoise step into
    /// a denoise LOOP without re-recording the graph `k` times.
    pub fn scaled(&self, k: u64) -> CostReport {
        CostReport {
            total: self.total.scaled(k),
            steps: self.steps * k,
            covered: self.covered * k,
            by_kernel: self
                .by_kernel
                .iter()
                .map(|(n, v)| (n.clone(), KernelCost { calls: v.calls * k, cost: v.cost.scaled(k) }))
                .collect(),
            uncovered: self.uncovered.iter().map(|(n, c)| (n.clone(), c * k)).collect(),
        }
    }

    /// `self - o`, per kernel, or `None` if `o` is not contained in `self`.
    ///
    /// This is what makes a whole-model cost DERIVABLE from small probes: the
    /// per-block cost of a transformer is the difference between the graphs of
    /// a depth-N and a depth-(N-1) build of the SAME config, so a 4B model's
    /// denoise cost can be computed exactly without ever holding 4B weights.
    /// `None` - a kernel in `o` that `self` does not have, or a negative
    /// difference - means the two graphs are not nested, which invalidates the
    /// derivation rather than merely perturbing it, so it is not a saturating
    /// subtraction.
    pub fn checked_sub(&self, o: &CostReport) -> Option<CostReport> {
        let mut out = CostReport {
            total: self.total.checked_sub(o.total)?,
            steps: self.steps.checked_sub(o.steps)?,
            covered: self.covered.checked_sub(o.covered)?,
            by_kernel: self.by_kernel.clone(),
            uncovered: self.uncovered.clone(),
        };
        for (k, v) in &o.by_kernel {
            let e = out.by_kernel.get_mut(k)?;
            e.calls = e.calls.checked_sub(v.calls)?;
            e.cost = e.cost.checked_sub(v.cost)?;
            if e.calls == 0 {
                out.by_kernel.remove(k);
            }
        }
        for (k, c) in &o.uncovered {
            let e = out.uncovered.get_mut(k)?;
            *e = e.checked_sub(*c)?;
            if *e == 0 {
                out.uncovered.remove(k);
            }
        }
        Some(out)
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

/// Thread-scoped recording of every dispatch submitted through ANY `Gpu`
/// handle on this thread, optionally with execution suppressed.
///
/// [`crate::Gpu::cost_of`] prices a step list a model already holds, which
/// serves the decoder LMs (`fwd_steps`) and the VAE decoders (`steps()`). A
/// diffusion transformer has no such list: flux2 and ltxv build their
/// dispatches inside `forward()` and submit them there, and several models
/// (the VAE decoders among them) open their own device, so the caller does not
/// even hold the handle whose counters would see them. A recording is scoped to
/// the calling THREAD rather than to a handle for exactly that reason - a graph
/// is built and submitted synchronously by whoever called `forward()`, so a
/// thread scope catches every handle that call touches and nothing else. Work a
/// model hands to its own threads (a cross-device sharded stage) is NOT seen,
/// and a caller relying on this must say so.
///
/// [`Recording::dry`] additionally suppresses execution: `submit` folds the
/// steps in and then drops them, so a whole image or video generation can be
/// priced with zero device work. That is only sound because a dispatch sequence
/// is a function of SHAPES, not of buffer contents - true of every graph in
/// this workspace, and gated in `tests/dry_run_recording.rs` (same report as a
/// real run, and the output buffer provably untouched).
///
/// One recording per thread; nesting panics rather than silently merging two
/// callers' numbers into one.
pub struct Recording {
    /// Not `Send`: a recording belongs to the thread that opened it.
    _not_send: std::marker::PhantomData<*const ()>,
}

struct Active {
    dry: bool,
    report: CostReport,
}

thread_local! {
    static RECORDING: std::cell::RefCell<Option<Active>> = const { std::cell::RefCell::new(None) };
}

impl Recording {
    /// Record AND execute - the graph runs as usual.
    pub fn live() -> Recording {
        Recording::start(false)
    }

    /// Record WITHOUT executing: nothing reaches the device and no buffer
    /// changes. This is what makes a cost report a prediction.
    pub fn dry() -> Recording {
        Recording::start(true)
    }

    fn start(dry: bool) -> Recording {
        RECORDING.with(|r| {
            let mut r = r.borrow_mut();
            assert!(r.is_none(), "a recording is already open on this thread");
            *r = Some(Active { dry, report: CostReport::default() });
        });
        Recording { _not_send: std::marker::PhantomData }
    }

    /// Close the recording and take what it saw.
    pub fn take(self) -> CostReport {
        let r = RECORDING.with(|r| r.borrow_mut().take()).expect("recording open");
        std::mem::forget(self); // Drop would clear an already-taken slot.
        r.report
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        RECORDING.with(|r| *r.borrow_mut() = None);
    }
}

/// Fold `steps` into this thread's recording, if any. Returns true when the
/// caller must SKIP execution (a dry recording).
///
/// The fast path is one thread-local `Option` check, so a production submit
/// pays a predictable-branch and nothing else.
pub fn record_submitted(names: &[String], steps: &[Step]) -> bool {
    RECORDING.with(|r| match &mut *r.borrow_mut() {
        Some(a) => {
            tally(&mut a.report, names, steps);
            a.dry
        }
        None => false,
    })
}

/// True iff a recording is open on this thread.
pub fn is_recording() -> bool {
    RECORDING.with(|r| r.borrow().is_some())
}

/// True iff `name` has a cost formula (all-ones probe shape; formulas are
/// polynomial in their params, so shape never changes coverage).
///
/// 32 slots, not 16: `conv3d`/`conv3d_dx`/`conv3d_dw`/`im2col3d_at`'s `Params`
/// structs run to 19 fields (the NCTHW conv family plus its im2col lowering),
/// so a 16-slot probe silently reported them UNCOVERED even with a correct
/// formula in place - the probe's shape must cover the widest real `Params`
/// struct, not just the kernels already in this file when it was 16.
pub fn covers(name: &str) -> bool {
    kernel_cost(name, Some(&[1; 32]), 1).is_some()
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
            // Weight scales are GROUP-wise (`model::int8::GROUP` = 32 int8 =
            // 8 packed words), so the `sw` term is `n * kg/8`, not `n`.
            c(2 * m * n, 8 * m * kg * n, 4 * (m * kg + n * kg + m * n + m + n * kg / 8))
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
            c(2 * m * n, 2 * m * k * n, 4 * (m * (k / 4) + n * (k / 8) + m * n + m + n * (k / 32)))
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
        // Replicate pad on NCL, params [total, l, left, right]. Pure movement,
        // no arithmetic: every output element is written once and reads one
        // source element, so the streaming traffic is 4 B in + 4 B out per
        // OUTPUT element - the pad regions re-read the same edge sample, which
        // this best-effort model counts as read traffic rather than modelling
        // the cache that would serve it.
        "pad1d_edge" => f(0, 8 * p(0)?),

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
        // `rmsnorm_dx_rows` is the workgroup-per-row variant: same math, same
        // traffic, only the thread mapping differs (mirrors
        // `layernorm_dx` / `layernorm_dx_rows` below).
        "rmsnorm_dx" | "rmsnorm_dx_eps" | "rmsnorm_dx_rows" => {
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
        // Per-row L2 norm with a learnable per-dim scale (GenieRedux QK-norm):
        // params [n, d, eps_bits]. One thread per (n,d) output element, EACH
        // thread redoing the whole row's sum-of-squares (no cross-thread
        // reduction) - real duplicated work, so flops scale as n*d*d, not
        // n*d. Per thread: d*(mul+add) + rsqrt(SFU, 1) + 2 closing muls =
        // 2*d+3. Bytes are the idealized "each operand once" streaming total
        // (x and y are [n,d], g is [d]), not the per-thread redundant reads.
        "l2norm_scale" => {
            let (n, d) = (p(0)?, p(1)?);
            f(n * d * (2 * d + 3), 4 * (2 * n * d + d))
        }
        // The FUSED channels-first twin: params [N, C, HW, eps_bits]. One
        // thread per (n, hw) position, so the sum of squares is computed ONCE
        // per position instead of once per output element. Per thread:
        // C*(mul+add) for the sum, rsqrt (1), then C*(2 muls) to scale =
        // 4*C + 1. Bytes are the same idealized streaming total as the
        // composed form's middle stage (x and y are [N,C,HW], g is [C]).
        "l2norm_scale2d" => {
            let (n, c, hw) = (p(0)?, p(1)?, p(2)?);
            f(n * hw * (4 * c + 1), 4 * (2 * n * c * hw + c))
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
        // Interleaved-pair variants sharing rope_base's exact per-pair op count
        // (pow + cos + sin + 4 mul + 2 add/sub = 10 ops/pair, 8 B/pair): `rope`
        // (in-place fused-qkv, analytic angle), `rope_neox` (half-split,
        // configurable theta - subsumed by rope_partial at rot_dim==head_dim
        // per that kernel's own seam note, but still dispatched by
        // kronos/chronos2), `rope_train`/`rope_train_bwd` (batched, within-
        // sequence position). All four share Params[rows, heads, head_dim, ...]
        // at slots 0,1,2.
        "rope" | "rope_neox" | "rope_train" | "rope_train_bwd" => {
            let (rows, h, hd) = (p(0)?, p(1)?, p(2)?);
            f(5 * rows * h * hd, 8 * rows * h * hd)
        }
        // Moondream partial RoPE fwd/bwd: same 10-op/pair rotation as rope_base,
        // over only `rot_dim` (not the full head_dim) channels of each head.
        // Params[n_rows, n_heads, head_dim, row_stride, base_off, tcols,
        // rope_base, rot_dim] - rot_dim at slot 7.
        "rope_partial" | "rope_partial_bwd" => {
            let (rows, h, rot_dim) = (p(0)?, p(1)?, p(7)?);
            f(5 * rows * h * rot_dim, 8 * rows * h * rot_dim)
        }
        // DSA-indexer interleaved RoPE on the first `rope_dim` channels of a
        // [rope|pass] head. Same per-pair cost as rope_train, over `rope_dim`
        // (slot 3) instead of the full head_dim.
        "rope_sub" => {
            let (rows, h, rope_dim) = (p(0)?, p(1)?, p(3)?);
            f(5 * rows * h * rope_dim, 8 * rows * h * rope_dim)
        }
        // Table-driven interleaved RoPE (Z-Image): angle comes from a
        // host-precomputed cos/sin table, so there is no pow/cos/sin per pair -
        // just 4 mul + 2 add/sub = 6 flops/pair. Params[seq_len, n_heads,
        // head_dim, half] gives `half` (pair count per head) directly at slot 3.
        // Bytes: x[seq,heads,hd] read once, cos/sin tables [seq,half] read once
        // each (shared across heads, so NOT multiplied by n_heads), y write once.
        "rope_interleave_table" => {
            let (rows, h, half) = (p(0)?, p(1)?, p(3)?);
            f(6 * rows * h * half, 4 * (4 * rows * h * half + 2 * rows * half))
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
        // Fuse three separate [seq, d] projections into one [seq, 3d] buffer
        // for the bidirectional/flash attention trio; params [seq_len,
        // d_model]. Pure movement: every one of the `seq*3*d` output elements
        // is written once and reads one source element, no arithmetic. This
        // was the one uncovered kernel on the flux2/ltxv/wan DiT attention
        // path, and one uncovered kernel is enough to make a whole-generation
        // total a partial number wearing a complete number's clothes.
        "pack_qkv" => {
            let (seq, d) = (p(0)?, p(1)?);
            f(0, 24 * seq * d)
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

        // Fused flash CROSS-attention; params [bsz, n_heads, t_dec, t_enc,
        // head_dim, ...]. Same fused trio as the bidirectional family above,
        // with the two lengths independent: the scores/apply MACs run over
        // td*te pairs and the softmax over td rows of te entries. Bytes are the
        // ideal-tiling traffic the kernel exists to achieve - q and out once
        // over td rows, k and v once over te rows, and NO materialised
        // [h, td, te] scores/probs (that absence is the whole point).
        "flash_attn_cross_reg2" => {
            let (b, h, td, te, hd) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?);
            f(
                b * h * td * te * (2 * hd + 1) + b * h * td * (6 * te + 1) + 2 * b * h * hd * td * te,
                4 * (2 * b * te * h * hd + 2 * b * td * h * hd),
            )
        }

        // ---- attention, cross (t_dec × t_enc); params [b, h, td, te, hd, ...].
        // `attn_scores_cross_kt` (params [bsz,n_heads,t_dec,t_enc,head_dim,
        // q_stride,q_off] - the same first 5 slots) is the coalesced twin
        // reading a key-minor `kt` instead of the fused `kv` slab: same
        // output values, same MAC count, same total q/k/scores element
        // counts (`kt` has exactly `k`'s element count, just transposed), so
        // it costs identically - ltxv's Phase 8 profiling adopted it in
        // place of `attn_scores_cross` for every attention call in that
        // crate (see `crates/ltxv/src/block.rs::attn_scores_kt`).
        "attn_scores_cross" | "attn_scores_cross_kt" => {
            let (b, h, td, te, hd) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?);
            f(b * h * td * te * (2 * hd + 1), 4 * (b * h * td * te + b * hd * h * (td + te)))
        }
        // The transpose `attn_scores_cross_kt` reads from: params [t_enc,
        // d_model, kv_stride, k_off]. Pure movement, one thread per (c,j)
        // output element - the same "read+write each moved element once"
        // accounting as `im2col_at`/`concat2`.
        "kv_k_headt" => f(0, 8 * p(0)? * p(1)?),
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
        // Exact erf-GELU (A&S 7.1.26 rational approximation, matching torch's
        // default F.gelu): arg-scale(1) + abs(1) + t=1/(1+c*ax)(3) + 4-term
        // Horner poly(8) + ax*ax+negate+exp+poly*exp+1-.+s*.(6) + 0.5*v*(1+erf)
        // (3) = 24 ops/element (transcendentals/div count 1, per this file's
        // own convention).
        "gelu_erf" => f(24 * n0(), 8 * n0()),
        // Moondream MoE expert activation: gelu_erf(h) * (g+1) - the same 24
        // ops plus one read of g, one add, one mul.
        "geglu_shift" => f(26 * n0(), 12 * n0()),
        // SnakeBeta (codec SEANet/BigVGAN vocoder activation): params
        // [total, c, inner, eps]. y = x + exp(-beta)*sin(exp(alpha)*x)^2, with
        // alpha/beta indexed per-channel and RECOMPUTED (exp) by every element
        // sharing that channel - real work, so flops scale with `total` not
        // `c`. 9 ops/element: 2 exp + 1 add(eps) + 1 mul(x*a) + 1 sin(SFU) +
        // 1 mul(s*s) + 1 div(1/b) + 1 mul + 1 add. Bytes: x/out are [total],
        // alpha/beta are [c] (each operand's true size once).
        "snake_beta" => {
            let (total, ch) = (p(0)?, p(1)?);
            f(9 * total, 4 * (2 * total + 2 * ch))
        }
        // Snake, the SINGLE-parameter DAC form: params [total, c, inner, eps].
        // y = x + (alpha+eps)^-1 * sin(alpha*x)^2, alpha per channel and used
        // un-exponentiated. 7 ops/element: 1 add(eps) + 1 mul(x*a) + 1 sin
        // (SFU) + 1 mul(s*s) + 1 div + 1 mul + 1 add - two fewer than
        // `snake_beta`, which pays two `exp` this form does not have. Bytes:
        // x/out are [total], alpha is [c].
        "snake1d" => {
            let (total, ch) = (p(0)?, p(1)?);
            f(7 * total, 4 * (2 * total + ch))
        }
        // `silu`/`silu_bwd` were uncovered, which is why a VQGAN forward could
        // not report a whole-pass rate at all: one kind without a formula makes
        // the pass numerator partial, and a partial numerator over the full
        // denominator under-reports rather than admitting it cannot tell.
        // x*sigmoid(x): exp + reciprocal + multiply, read one write one.
        "silu" => f(4 * n0(), 8 * n0()),
        // y = 1/(1+exp(-x)): exp + add + divide = 3, one op short of `silu`
        // above, which is that sigmoid times x. params [total].
        "sigmoid" => f(3 * n0(), 8 * n0()),
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
        // params [R, D, NR, row, plus_one]: one PixArt/adaLN-single modulation
        // vector, `out[r,d] = tbl[row,d] + tab[map[r],row,d]` and one more add
        // when `plus_one`. Bytes are the streaming form: `D` gathered floats
        // and `D` written per token, plus the block's own `[NR,D]` row and the
        // `R`-entry u32 row map read once each. The gather is counted at what
        // the dispatch issues (`R*D`), not at the distinct rows behind it -
        // this file's `bytes` is a roofline denominator, not a cache model,
        // and `U` is not in the uniform params anyway.
        "adaln_row" => {
            let (rows, dim, plus_one) = (p(0)?, p(1)?, p(4).unwrap_or(0).min(1));
            f(rows * dim * (1 + plus_one), 4 * (2 * rows * dim + dim + rows))
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
        // These were UNCOVERED until 2026-08-06, which meant `conv_bias_reg` -
        // nearly the whole VQGAN forward - reported no rate at all and the
        // pass-level GFLOP/s was a fiction. Formulas mirror the conv1d family above.
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
        // The 1D lowering's own pair. `im2col1d_at` is the same pure movement
        // one spatial axis down: params [cin,l,k,stride,pad,dilation,cink,pos0,cnt],
        // `cnt*cink` floats read and written.
        "im2col1d_at" => {
            let (cink, cnt) = (p(6)?, p(8)?);
            f(0, 8 * cnt * cink)
        }
        // `col2im1d_bias` gathers the [Cout*K, L] GEMM output into [Cout, Lo]
        // and adds the bias: params [l, cout, k, stride, pad, dilation, lo].
        // One add per tap that actually lands (`K/stride` of the `K` the loop
        // walks, the same discard `convtr1d`'s own formula accounts for), plus
        // one for the bias. Reads only the taps it uses - `Lo*(K/stride)` per
        // channel - and writes `Cout*Lo`.
        "col2im1d_bias" => {
            let (cout, k, stride, lo) = (p(1)?, p(2)?, p(3)?.max(1), p(6)?);
            let taps = cout * lo * (k / stride).max(1);
            f(taps + cout * lo, 4 * (taps + cout * lo + cout))
        }
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
        // Uncovered until now, which cost the VQGAN backward its pass rate -
        // `gn_dsum_part` + `gn_dgb_part` alone are a real share of that pass.
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
        // `concat_split`'s inverse - copy a source tensor INTO a channel range
        // of a bigger NCHW destination. Identical `Params` layout
        // ([N, Ctot, Csrc, c_off, H, W]) and identical traffic: one dispatch
        // per SOURCE element, read once, written once.
        "chan_place" => {
            let n = p(0)? * p(2)? * p(4)? * p(5)?;
            f(0, 8 * n)
        }

        // ---- conv1d family: params [N, Cin, L, Cout, K, stride, pad, dil, G, Lo].
        "conv1d" => {
            let (n, cin, l, cout, k, g, lo) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(8)?, p(9)?);
            let cin_g = cin / g.max(1);
            f(2 * n * cout * lo * cin_g * k, 4 * (n * cin * l + cout * cin_g * k + n * cout * lo))
        }
        // ConvTranspose1d forward. One thread per OUTPUT element gathers the
        // inputs that land on it, so the exact MAC count is the FORWARD conv's
        // seen from the other side: every (input element, output channel in its
        // group, tap) triple contributes one MAC, i.e. `N*Cin*L*(Cout/G)*K`,
        // independent of stride/pad. Counting `N*Cout*Lo*(Cin/G)*K` instead -
        // the naive "output elements times the loop bound" - overstates it by
        // the stride, since the `(lo+pad-kw*d) % stride != 0` branch discards
        // most taps at an upsampling stride. The vocoder's stages run at
        // stride 8/8/4/2, so that error would have been up to 8x on the single
        // most expensive kernel in the decoder.
        "convtr1d" => {
            let (n, cin, l, cout, k, g, lo) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(8)?, p(9)?);
            let cout_g = cout / g.max(1);
            f(2 * n * cin * l * cout_g * k, 4 * (n * cin * l + cin * cout_g * k + n * cout * lo))
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

        // ---- LTX-2.5 (ltxv): 3D conv, 3D neighborhood attention, layout ----
        //
        // conv3d family: params [N, Cin, T, H, W, Cout, KT, KH, KW, st, sh, sw,
        // pt, ph, pw, groups, To, Ho, Wo] - conv2d's family lifted to NCTHW,
        // WITH grouping (conv2d has none). Bias is tiny and omitted, matching
        // the conv2d precedent above. `cin_g`/`cout_g` are the per-group
        // channel counts the WGSL loops actually bound over.
        "conv3d" => {
            let (n, cin, t, h, w, cout, kt, kh, kw, groups, to, ho, wo) = (
                p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(5)?, p(6)?, p(7)?, p(8)?, p(15)?, p(16)?,
                p(17)?, p(18)?,
            );
            let cin_g = cin / groups.max(1);
            let ktkhkw = kt * kh * kw;
            f(
                2 * n * cout * to * ho * wo * cin_g * ktkhkw,
                4 * (n * cin * t * h * w + cout * cin_g * ktkhkw + n * cout * to * ho * wo),
            )
        }
        // GATHER form (one thread per INPUT element, reduces cout_g*KT*KH*KW) -
        // same Params as conv3d.
        "conv3d_dx" => {
            let (n, cin, t, h, w, cout, kt, kh, kw, groups, to, ho, wo) = (
                p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(5)?, p(6)?, p(7)?, p(8)?, p(15)?, p(16)?,
                p(17)?, p(18)?,
            );
            let cin_g = cin / groups.max(1);
            let cout_g = cout / groups.max(1);
            let ktkhkw = kt * kh * kw;
            f(
                2 * n * cin * t * h * w * cout_g * ktkhkw,
                4 * (n * cout * to * ho * wo + cout * cin_g * ktkhkw + n * cin * t * h * w),
            )
        }
        // Accumulating weight gradient (dw is read AND written, like
        // conv2d_dw) - same Params as conv3d.
        "conv3d_dw" => {
            let (n, cin, t, h, w, cout, kt, kh, kw, groups, to, ho, wo) = (
                p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(5)?, p(6)?, p(7)?, p(8)?, p(15)?, p(16)?,
                p(17)?, p(18)?,
            );
            let cin_g = cin / groups.max(1);
            let ktkhkw = kt * kh * kw;
            f(
                2 * cout * cin_g * ktkhkw * n * to * ho * wo,
                4 * (n * cout * to * ho * wo + n * cin * t * h * w + 2 * cout * cin_g * ktkhkw),
            )
        }
        // 3D im2col over a position RANGE - im2col_at lifted to NCTHW; pure
        // movement, same "read+write each moved element once" accounting.
        // Params: [cin,t,h,w,kt,kh,kw,st,sh,sw,pt,ph,pw,to,ho,wo,cinkkk,pos0,cnt].
        "im2col3d_at" => {
            let (cinkkk, cnt) = (p(16)?, p(18)?);
            f(0, 8 * cnt * cinkkk)
        }
        // space_to_depth3d / depth_to_space3d: pure rearrange, element count
        // preserved (Cout*To*Ho*Wo == Cin*T*H*W by construction), so both cost
        // the same "read once, write once" traffic over the INPUT's element
        // count - params share [Cin,T,H,W,...] at slots 0..3.
        "space_to_depth3d" | "depth_to_space3d" => {
            let (cin, t, h, w) = (p(0)?, p(1)?, p(2)?, p(3)?);
            f(0, 8 * cin * t * h * w)
        }
        // Channels-last 3D pixel shuffle (depth-to-space): params
        // [T,H,W,Cout,p1,p2,p3]; element count = Cin*T*H*W = Cout*p1*p2*p3*T*H*W.
        "pixel_shuffle3d_cl" => {
            let (t, h, w, cout, p1, p2, p3) = (p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(5)?, p(6)?);
            f(0, 8 * t * h * w * cout * p1 * p2 * p3)
        }
        // 3D neighborhood-attention (NATTEN-style windowed self-attention)
        // scores: params [t,h,w,heads,head_dim,kt,kh,kw]. One thread per
        // (head,query,window-slot), serial hd-deep dot product. q/k are the
        // SAME [t*h*w,heads,head_dim] volume (self-attention).
        "na3d_scores" => {
            let (t, h, w, heads, hd, kt, kh, kw) = (
                p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(5)?, p(6)?, p(7)?,
            );
            let nq = t * h * w;
            let window = kt * kh * kw;
            let total = heads * nq * window;
            f(2 * hd * total, 4 * (2 * nq * heads * hd + total))
        }
        // na3d_scores' twin: probs*V apply. params identical; one thread per
        // (query,head,dim), serial window-deep reduction.
        "na3d_apply" => {
            let (t, h, w, heads, hd, kt, kh, kw) = (
                p(0)?, p(1)?, p(2)?, p(3)?, p(4)?, p(5)?, p(6)?, p(7)?,
            );
            let nq = t * h * w;
            let window = kt * kh * kw;
            let total_out = nq * heads * hd;
            f(2 * window * total_out, 4 * (heads * nq * window + 2 * nq * heads * hd))
        }
        // NLC -> NCHW transpose with a fused per-channel bias (64x64 tiled,
        // the conv-lowering epilogue). Params [total(unused), c, l]; x/y are
        // [l,c]/[c,l] (same size), bias is [c].
        "nlc_bias_nchw" => {
            let (c, l) = (p(1)?, p(2)?);
            f(c * l, 4 * (2 * c * l + c))
        }
        // Per-(image,channel) scalar broadcast-add, NCHW: params [N,C,HW].
        // x/y are [N,C,HW], v is [N,C].
        "add_chan_bcast" => {
            let (n, ch, hw) = (p(0)?, p(1)?, p(2)?);
            f(n * ch * hw, 4 * (2 * n * ch * hw + n * ch))
        }
        // In-place per-CHANNEL bias, NCHW: params [total = N*C*HW, C, HW].
        // One add per element; `out` is read AND written through the single
        // read_write binding, and the bias vector is `C` long.
        "add_chan_inplace" => {
            let (total, ch) = (p(0)?, p(1)?);
            f(total, 4 * (2 * total + ch))
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

    /// Report arithmetic: a whole-model cost is DERIVED from small probes by
    /// subtracting nested graphs and scaling the difference, so both operations
    /// have to be exact per kernel, not just in the totals.
    #[test]
    fn report_arithmetic_is_exact_per_kernel() {
        let mut small = CostReport::default();
        let mut big = CostReport::default();
        for r in [&mut small, &mut big] {
            r.record("matmul", kernel_cost("matmul", Some(&[8, 4, 6]), 48));
            r.record("nosuch_kernel", None);
        }
        big.record("matmul", kernel_cost("matmul", Some(&[8, 4, 6]), 48));
        big.record("silu", kernel_cost("silu", Some(&[100]), 100));

        let d = big.checked_sub(&small).expect("big contains small");
        assert_eq!(d.steps, 2);
        assert_eq!(d.by_kernel["matmul"].calls, 1);
        assert_eq!(d.by_kernel["silu"].calls, 1);
        assert!(!d.by_kernel.contains_key("nosuch_kernel"));
        assert!(d.uncovered.is_empty(), "the uncovered call cancels too");
        assert_eq!(d.total, Cost { flops: 384 + 400, int_ops: 0, bytes: 4 * 104 + 800 });

        // Scaling a report is scaling every kernel row, not just the total.
        let x3 = d.scaled(3);
        assert_eq!(x3.total, d.total.scaled(3));
        assert_eq!(x3.steps, 6);
        assert_eq!(x3.by_kernel["silu"].calls, 3);
        assert_eq!(x3.by_kernel["silu"].cost, d.by_kernel["silu"].cost.scaled(3));

        // Non-nested graphs are refused, not saturated: an unsound derivation
        // must fail loudly rather than quietly report a smaller model.
        assert!(small.checked_sub(&big).is_none(), "subtracting a superset must refuse");
    }

    /// The three kernels the diffusion image/video graphs dispatch that had no
    /// formula, hand-computed. `pack_qkv` and `chan_place` move bytes and do no
    /// arithmetic, so a "flops" assertion alone would pass on a formula that
    /// forgot the traffic entirely - both numbers are asserted.
    #[test]
    fn diffusion_movement_and_gate_costs() {
        // pack_qkv [seq=5, d=8]: 5*3*8 = 120 elements, read once + written
        // once = 8 B each.
        let pk = cost("pack_qkv", &[5, 8], 120);
        assert_eq!((pk.flops, pk.int_ops, pk.bytes), (0, 0, 8 * 120));
        // sigmoid: exp + add + divide per element, one fewer than silu's 4.
        assert_eq!(cost("sigmoid", &[100], 100).flops, 300);
        assert_eq!(cost("sigmoid", &[100], 100).bytes, 800);
        assert_eq!(cost("silu", &[100], 100).flops - cost("sigmoid", &[100], 100).flops, 100);
        // chan_place [N=2, Ctot=10, Csrc=3, c_off=4, H=5, W=7]: 2*3*5*7 = 210
        // SOURCE elements - `Ctot` and `c_off` size the destination, not the
        // work, so a formula that reached for Ctot would report 700.
        let cp = cost("chan_place", &[2, 10, 3, 4, 5, 7], 210);
        assert_eq!((cp.flops, cp.bytes), (0, 8 * 210));
        assert_eq!(cp.bytes, cost("concat_split", &[2, 10, 3, 4, 5, 7], 210).bytes, "chan_place is concat_split's inverse: same traffic");
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
        // its published share of peak was a fiction in the other direction.
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

    /// Hand-computed expectations for the ltxv (Phase 8) formulas: 3D conv,
    /// 3D neighborhood attention, the rope*/gelu_erf/geglu_shift/snake_beta/
    /// l2norm_scale/nlc_bias_nchw/add_chan_bcast family, and the 3D layout
    /// kernels (space_to_depth3d/depth_to_space3d/pixel_shuffle3d_cl/
    /// im2col3d_at).
    #[test]
    fn ltxv_kernel_costs() {
        // conv3d [N=1,Cin=1,T=3,H=1,W=1,Cout=1,KT=3,KH=1,KW=1,st=1,sh=1,sw=1,
        // pt=0,ph=0,pw=0,groups=1,To=1,Ho=1,Wo=1]: 1 output x (1 cin_g x 3 KT)
        // taps x 2 = 6 flops; bytes = 4*(x=3 + wt=3 + y=1) = 28.
        let p3d = [1u32, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1];
        assert_eq!(cost("conv3d", &p3d, 1).flops, 6);
        assert_eq!(cost("conv3d", &p3d, 1).bytes, 28);
        // dX gather form: 2*(N*Cin*T*H*W)*cout_g*KT*KH*KW = 2*3*1*3 = 18.
        assert_eq!(cost("conv3d_dx", &p3d, 1).flops, 18);
        assert_eq!(cost("conv3d_dx", &p3d, 1).bytes, 28);
        // dW: 2*Cout*cin_g*KT*KH*KW*N*To*Ho*Wo = 2*3*1 = 6; dw is read+written.
        assert_eq!(cost("conv3d_dw", &p3d, 1).flops, 6);
        assert_eq!(cost("conv3d_dw", &p3d, 1).bytes, 40);

        // na3d_scores/apply [t=2,h=1,w=1,heads=1,head_dim=2,kt=2,kh=1,kw=1]:
        // nq=2, window=2, total=4 -> 2*hd*total = 16 flops each.
        let pna = [2u32, 1, 1, 1, 2, 2, 1, 1];
        assert_eq!(cost("na3d_scores", &pna, 4).flops, 16);
        assert_eq!(cost("na3d_scores", &pna, 4).bytes, 48);
        assert_eq!(cost("na3d_apply", &pna, 4).flops, 16);
        assert_eq!(cost("na3d_apply", &pna, 4).bytes, 48);

        // rope [seq_len=2,n_heads=1,head_dim=4,row_stride=4,base_off=0]:
        // 5*rows*h*hd = 40 flops, 8*rows*h*hd = 64 bytes - the rope_base
        // family's exact per-pair accounting, shared by rope/rope_neox/
        // rope_train/rope_train_bwd (same Params slots 0,1,2).
        let prope = [2u32, 1, 4, 4, 0];
        assert_eq!(cost("rope", &prope, 4).flops, 40);
        assert_eq!(cost("rope_neox", &[2, 1, 4, 4, 0, 10000], 4).flops, 40);
        assert_eq!(cost("rope_train", &[2, 1, 4, 4, 0, 2], 4).flops, 40);
        assert_eq!(cost("rope_train_bwd", &[2, 1, 4, 4, 0, 2], 4).flops, 40);
        // rope_partial: rot_dim (slot 7) stands in for head_dim.
        let pp = [2u32, 1, 8, 8, 0, 2, 10000, 4];
        assert_eq!(cost("rope_partial", &pp, 4).flops, 40);
        assert_eq!(cost("rope_partial_bwd", &pp, 4).flops, 40);
        // rope_sub: rope_dim at slot 3.
        assert_eq!(cost("rope_sub", &[2, 1, 8, 4, 8, 2], 4).flops, 40);
        // rope_interleave_table: table lookup, no pow/cos/sin -> 6 flops/pair.
        let pt = [2u32, 1, 4, 2];
        assert_eq!(cost("rope_interleave_table", &pt, 4).flops, 24);
        assert_eq!(cost("rope_interleave_table", &pt, 4).bytes, 96);

        // gelu_erf/geglu_shift/snake_beta: elementwise, params[total(,c,...)].
        assert_eq!(cost("gelu_erf", &[10], 10).flops, 240);
        assert_eq!(cost("gelu_erf", &[10], 10).bytes, 80);
        assert_eq!(cost("geglu_shift", &[10], 10).flops, 260);
        assert_eq!(cost("snake_beta", &[8, 2, 1, 0], 8).flops, 72);
        assert_eq!(cost("snake_beta", &[8, 2, 1, 0], 8).bytes, 80);

        // l2norm_scale [n=2,d=4]: n*d*(2*d+3) = 8*11 = 88 flops (real
        // per-thread duplicated work); bytes are the idealized per-operand-
        // once total = 4*(2*8+4) = 80.
        assert_eq!(cost("l2norm_scale", &[2, 4, 0], 8).flops, 88);
        assert_eq!(cost("l2norm_scale", &[2, 4, 0], 8).bytes, 80);

        // l2norm_scale2d [N=1,C=4,HW=2]: the same 8 elements normalized over
        // the same axis, one thread per POSITION - N*HW*(4*C+1) = 2*17 = 34
        // flops against the composed form's 88, which is the duplicated work
        // the fusion removes. Bytes match, because both move each operand once.
        assert_eq!(cost("l2norm_scale2d", &[1, 4, 2, 0], 2).flops, 34);
        assert_eq!(cost("l2norm_scale2d", &[1, 4, 2, 0], 2).bytes, 80);

        // nlc_bias_nchw [total(unused)=12,c=3,l=4]: c*l = 12 flops.
        assert_eq!(cost("nlc_bias_nchw", &[12, 3, 4], 12).flops, 12);
        assert_eq!(cost("nlc_bias_nchw", &[12, 3, 4], 12).bytes, 108);
        // add_chan_bcast [N=2,C=3,HW=4]: N*C*HW = 24 flops.
        assert_eq!(cost("add_chan_bcast", &[2, 3, 4], 24).flops, 24);
        assert_eq!(cost("add_chan_bcast", &[2, 3, 4], 24).bytes, 216);

        // 3D layout: pure movement, element count = Cin*T*H*W (same on both
        // sides of the rearrange).
        assert_eq!(
            cost("space_to_depth3d", &[2, 4, 1, 1, 2, 1, 1, 2, 1, 1], 8).bytes,
            64
        );
        assert_eq!(cost("depth_to_space3d", &[2, 4, 1, 1, 2, 1, 1, 1], 8).bytes, 64);
        assert_eq!(cost("pixel_shuffle3d_cl", &[2, 1, 1, 1, 2, 1, 1], 4).bytes, 32);
        // im2col3d_at: cinkkk (slot 16) x cnt (slot 18) moved elements.
        let pim = [1u32, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 6, 0, 4];
        assert_eq!(cost("im2col3d_at", &pim, 24).bytes, 192);

        // attn_scores_cross_kt shares attn_scores_cross's formula exactly
        // (same first-5 param slots, same MAC count, same total q/k/scores
        // element counts - kt is k transposed, not resized).
        let pcross = [1u32, 2, 3, 5, 8, 96, 0];
        assert_eq!(cost("attn_scores_cross_kt", &pcross, 30).flops, 2 * 15 * 17);
        // kv_k_headt [t_enc=4, d_model=8, kv_stride=16, k_off=0]: pure
        // movement, 8 bytes per moved element x (4*8) elements = 256.
        assert_eq!(cost("kv_k_headt", &[4, 8, 16, 0], 32).bytes, 256);

        for k in [
            "na3d_scores", "na3d_apply", "conv3d", "conv3d_dx", "conv3d_dw", "im2col3d_at",
            "space_to_depth3d", "depth_to_space3d", "pixel_shuffle3d_cl", "l2norm_scale",
            "l2norm_scale2d",
            "nlc_bias_nchw", "add_chan_bcast", "rope", "rope_neox", "rope_train",
            "rope_train_bwd", "rope_partial", "rope_partial_bwd", "rope_sub",
            "rope_interleave_table", "gelu_erf", "geglu_shift", "snake_beta",
            "attn_scores_cross_kt", "kv_k_headt", "convtr1d", "snake1d", "add_chan_inplace",
            "im2col1d_at", "col2im1d_bias", "matmul_dw_reg_splitk",
        ] {
            assert!(covers(k), "kernel `{k}` has no cost formula");
        }
    }

    /// The DAC-vocoder trio: `convtr1d`, `snake1d`, `add_chan_inplace`.
    ///
    /// These three were the ONLY uncovered kinds in `minimaxmusic3`'s vocoder -
    /// and they are its upsample path, its activation and its conv bias, i.e.
    /// most of the decoder. One uncovered kind makes the whole pass unable to
    /// report a rate at all ([`super::super::profile::PassProfile::gflops`] is
    /// `None` the moment a single row is uncovered), so the vocoder could not
    /// be profiled against a roofline until these landed.
    #[test]
    fn the_dac_vocoder_upsample_trio_is_covered() {
        // convtr1d [N=1,Cin=2,L=4,Cout=3,K=4,stride=2,pad=1,dil=1,G=1,Lo=8].
        // Every (input element, output channel, tap) is exactly one MAC:
        // 2 * 1*2*4*3*4 = 192 flops.
        assert_eq!(cost("convtr1d", &[1, 2, 4, 3, 4, 2, 1, 1, 1, 8], 24).flops, 192);
        // The trap this formula exists to avoid: the MAC count does NOT scale
        // with the upsampling stride. Doubling stride doubles `Lo` and doubles
        // the thread count, but every extra thread's taps land on the
        // `(lo+pad-kw*d) % stride != 0` branch and do no work.
        assert_eq!(cost("convtr1d", &[1, 2, 4, 3, 4, 4, 1, 1, 1, 16], 48).flops, 192);
        // bytes: x [1*2*4] + w [2*3*4] + y [1*3*8] = 56 floats.
        assert_eq!(cost("convtr1d", &[1, 2, 4, 3, 4, 2, 1, 1, 1, 8], 24).bytes, 4 * 56);

        // snake1d [total=8, c=2, inner=4, eps]: 7 ops/element - two fewer than
        // `snake_beta`, which pays two `exp` this single-parameter form has
        // not got - and ONE alpha vector, not two.
        assert_eq!(cost("snake1d", &[8, 2, 4, 0], 8).flops, 56);
        assert_eq!(cost("snake1d", &[8, 2, 4, 0], 8).bytes, 4 * (2 * 8 + 2));

        // add_chan_inplace [total=24, c=3, hw=4]: one add per element, with
        // `out` read AND written through its single read_write binding.
        assert_eq!(cost("add_chan_inplace", &[24, 3, 4], 24).flops, 24);
        assert_eq!(cost("add_chan_inplace", &[24, 3, 4], 24).bytes, 4 * (2 * 24 + 3));
    }

    /// The 1D conv-as-GEMM lowering's own pair. Without these the vocoder's
    /// pass stops reporting a rate the moment the lowered path is selected -
    /// the same coverage hole [`the_dac_vocoder_upsample_trio_is_covered`]
    /// closed for the direct kernels.
    #[test]
    fn the_1d_conv_lowering_pair_is_covered() {
        // im2col1d_at [cin=8,l=16,k=3,stride=1,pad=1,dil=1,cink=24,pos0=0,cnt=16]:
        // pure movement, 8 bytes per moved float (read + write), no arithmetic.
        assert_eq!(cost("im2col1d_at", &[8, 16, 3, 1, 1, 1, 24, 0, 16], 384).bytes, 8 * 16 * 24);
        assert_eq!(cost("im2col1d_at", &[8, 16, 3, 1, 1, 1, 24, 0, 16], 384).flops, 0);

        // col2im1d_bias [l=4,cout=3,k=4,stride=2,pad=1,dil=1,lo=8]: K/stride = 2
        // taps actually land per output element, plus the bias add:
        // 3*8*2 + 3*8 = 72 adds.
        assert_eq!(cost("col2im1d_bias", &[4, 3, 4, 2, 1, 1, 8], 24).flops, 72);
        // bytes: the 48 tap reads, the 24 output writes and the 3-long bias.
        assert_eq!(cost("col2im1d_bias", &[4, 3, 4, 2, 1, 1, 8], 24).bytes, 4 * (48 + 24 + 3));
        // The same trap `convtr1d`'s formula documents, from the epilogue's
        // side: doubling the stride doubles `Lo` and the thread count but NOT
        // the taps that land, so the per-element tap count must divide by the
        // stride rather than stay at `K`.
        // K/stride = 1 tap per output element now, plus the bias: 2*3*16.
        assert_eq!(cost("col2im1d_bias", &[4, 3, 4, 4, 1, 1, 16], 48).flops, 2 * 3 * 16);
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
        // Measured 2026-08-19: 225 of 416 (ltxv's na3d/conv3d/rope-family/
        // gelu_erf/geglu_shift/snake_beta/l2norm_scale/nlc_bias_nchw/
        // add_chan_bcast pass added here, plus widening `covers`'s probe to
        // 32 params so conv3d's 19-field Params struct stops reporting
        // UNCOVERED despite having a formula, plus `attn_scores_cross_kt`/
        // `kv_k_headt` once ltxv's Phase 8 optimization pass adopted them).
        // Deliberately a floor and not an equality - adding a formula must
        // not require editing a test.
        //
        // 225 -> 228: `convtr1d`/`snake1d`/`add_chan_inplace`, the DAC-style
        // vocoder's upsample, activation and conv-bias kernels. Without them
        // `minimaxmusic3`'s vocoder pass could not report a rate at all.
        const FLOOR: usize = 228;
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
