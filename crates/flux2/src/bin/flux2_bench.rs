// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 DiT profiler: where the forward's time actually goes, per matmul
//! shape and per step kind, against the Tesla P40 fp32 peak (11.76 TFLOP/s).
//!
//! The DiT forward is a fixed sequence of dispatches whose *cost depends only
//! on shape*, so this drives correctly-shaped scratch instead of the 15.5 GiB
//! checkpoint (same idea as `zimage_bench train`). That makes the profile
//! runnable in seconds, with no weights, and lets each shape class be timed in
//! isolation. Total analytic FLOP is asserted against the real graph's
//! 10.17 TFLOP so the replay cannot silently drift from `model.rs`.
//!
//! Usage:
//!   flux2_bench mm      [reps]        standalone matmul at the DiT's shapes
//!   flux2_bench floor   [n]           per-dispatch floor (tiny kernel × n)
//!   flux2_bench replay  [reps] [h w]  the whole DiT graph, per shape + kind
//!
//! `BRAIN_GPU_INDEX=0` selects a card. Per-group timings drain the queue with a
//! `poll_wait` between groups; that adds one queue round-trip (~the `floor`
//! number) per group, which is reported so it can be subtracted.

use std::time::Instant;

use flux2::Flux2Config;
use gpu_core::{DeviceBuffer, Gpu, Step};

const P40_FP32_TFLOPS: f64 = 11.76;

/// Kernel slots — the subset of `flux2::KERNELS` this bench dispatches, in the
/// same order so a slot index means the same thing in both.
const KERNELS: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pack_qkv", kernels::PACK_QKV),
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT),
    ("silu_mul", kernels::SILU_MUL),
    ("gate_row", kernels::GATE_ROW),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
];
const K_LN: usize = 0;
const K_MM: usize = 1;
const K_RMS: usize = 2;
const K_ROPE: usize = 3;
const K_PACK: usize = 4;
const K_FLASH: usize = 5;
const K_SPLIT: usize = 6;
const K_SILU: usize = 7;
const K_GATE: usize = 8;
const K_MM3: usize = 9;
const K_RMS_ROWS: usize = 10;

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// Best-of-`reps` wall seconds for one submitted-and-drained step list.
fn time_steps(gpu: &Gpu, steps: &[Step], reps: usize) -> f64 {
    gpu.submit(&[], steps);
    gpu.poll_wait();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        gpu.submit(&[], steps);
        gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    best
}

fn pct(gflops: f64) -> f64 {
    100.0 * gflops / (P40_FP32_TFLOPS * 1e3)
}

// ---------------------------------------------------------------- mm ------

/// Every distinct matmul shape the klein-4B graph runs at 512²/1536 tokens,
/// with its dispatch count per forward. (m, k, n, count, label)
fn mm_shapes(cfg: &Flux2Config, nt: u32, ni: u32) -> Vec<(u32, u32, u32, u32, &'static str)> {
    let d = cfg.hidden as u32;
    let mlp = cfg.mlp_hidden() as u32;
    let cin = cfg.in_channels as u32;
    let ctxd = cfg.context_in_dim as u32;
    let n = nt + ni;
    let nd = cfg.depth_double as u32;
    let ns = cfg.depth_single as u32;
    vec![
        // single-stream blocks (the bulk)
        (n, d, d, 4 * ns, "sgl qkv+wo_a"),
        (n, d, mlp, 2 * ns, "sgl w1+w3"),
        (n, mlp, d, ns, "sgl wo_b"),
        // double-stream blocks, image rows
        (ni, d, d, 4 * nd, "dbl img qkv+wo"),
        (ni, d, mlp, 2 * nd, "dbl img w1+w3"),
        (ni, mlp, d, nd, "dbl img w2"),
        // double-stream blocks, text rows
        (nt, d, d, 4 * nd, "dbl txt qkv+wo"),
        (nt, d, mlp, 2 * nd, "dbl txt w1+w3"),
        (nt, mlp, d, nd, "dbl txt w2"),
        // boundary linears
        (nt, ctxd, d, 1, "txt_in"),
        (ni, cin, d, 1, "img_in"),
        (ni, d, cin, 1, "final"),
    ]
}

fn mm_step(gpu: &Gpu, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
    gpu.step(K_MM, &[x, w, o], &[m, k, n], m.div_ceil(128) * n.div_ceil(128) * 256)
}

fn bench_mm(gpu: &Gpu, reps: usize) {
    let cfg = Flux2Config::klein_4b();
    let shapes = mm_shapes(&cfg, 512, 1024);
    println!("\n=== matmul_reg2 standalone, klein-4B DiT shapes (1×P40, best of {reps}) ===");
    println!(
        "{:<22} {:>6} {:>6} {:>6} {:>5} {:>10} {:>10} {:>7} {:>9}",
        "shape", "m", "k", "n", "disp", "ms/disp", "GFLOP/s", "%peak", "fwd ms"
    );
    let mut total_ms = 0.0;
    let mut total_gflop = 0.0;
    for &(m, k, n, count, label) in &shapes {
        let x = gpu.storage((m as u64) * k as u64);
        let w = gpu.storage((n as u64) * k as u64);
        let o = gpu.storage((m as u64) * n as u64);
        // 4 back-to-back identical dispatches per timed submit: amortises the
        // one queue round-trip so the number is kernel time, not launch time.
        let batch = 4usize;
        let steps: Vec<Step> = (0..batch).map(|_| mm_step(gpu, &x, &w, &o, m, k, n)).collect();
        let s = time_steps(gpu, &steps, reps) / batch as f64;
        let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
        let gflops = gflop / s;
        total_ms += s * 1e3 * count as f64;
        total_gflop += gflop * count as f64;
        println!(
            "{label:<22} {m:>6} {k:>6} {n:>6} {count:>5} {:>10.3} {:>10.0} {:>6.1}% {:>9.1}",
            s * 1e3,
            gflops,
            pct(gflops),
            s * 1e3 * count as f64
        );
    }
    println!(
        "TOTAL matmul: {:.0} GFLOP in {:.0} ms -> {:.0} GFLOP/s ({:.1}% peak)",
        total_gflop,
        total_ms,
        total_gflop / (total_ms / 1e3),
        pct(total_gflop / (total_ms / 1e3))
    );
}

// ------------------------------------------------------------- floor ------

/// Per-dispatch floor: `n` one-workgroup matmuls in ONE submit. Isolates
/// pipeline-set + bind-group + dispatch cost from any real work.
fn bench_floor(gpu: &Gpu, n: usize) {
    let x = gpu.storage(256);
    let w = gpu.storage(256);
    let o = gpu.storage(256);
    let steps: Vec<Step> = (0..n).map(|_| mm_step(gpu, &x, &w, &o, 1, 1, 1)).collect();
    let s = time_steps(gpu, &steps, 20);
    let one = time_steps(gpu, &steps[..1], 20);
    println!("\n=== dispatch floor (1×1×1 matmul_reg2, one workgroup) ===");
    println!("  1 dispatch + queue round-trip: {:.3} ms", one * 1e3);
    println!("  {n} dispatches in one submit:   {:.3} ms total, {:.4} ms/dispatch", s * 1e3, s * 1e3 / n as f64);
}

// ------------------------------------------------------------ replay ------

/// Scratch sized for the real graph, weights shared per distinct shape (the
/// timing depends only on shape, so one buffer per (n_out, k) suffices).
struct Replay {
    steps: Vec<Step>,
    /// (kind label, shape label, index range into `steps`) per recorded group.
    groups: Vec<(&'static str, String, usize, usize)>,
}

#[allow(clippy::too_many_arguments)]
fn build_replay(gpu: &Gpu, cfg: &Flux2Config, nt: u32, ni: u32) -> (Replay, f64) {
    let d = cfg.hidden as u32;
    let mlp = cfg.mlp_hidden() as u32;
    let cin = cfg.in_channels as u32;
    let ctxd = cfg.context_in_dim as u32;
    let hd = cfg.head_dim() as u32;
    let nh = cfg.n_heads as u32;
    let n = nt + ni;
    let (nu, du, mu) = (n as u64, d as u64, mlp as u64);

    let a = |len: u64| gpu.storage(len);
    let x0 = a(nu * du);
    let x1 = a(nu * du);
    let n1 = a(nu * du);
    let (q, k, v) = (a(nu * du), a(nu * du), a(nu * du));
    let (qn, kn) = (a(nu * du), a(nu * du));
    let (qr, kr) = (a(nu * du), a(nu * du));
    let qkv = a(nu * 3 * du);
    let ctx = a(nu * du);
    let proj = a(nu * du);
    let (h1, h2, hs) = (a(nu * mu), a(nu * mu), a(nu * mu));
    let mlpb = a(nu * du);
    let out = a(nu * cin as u64);
    let (cos, sin) = (a(nu * hd as u64 / 2), a(nu * hd as u64 / 2));
    let tok_in = a(nu * cin as u64);
    let ctx_in = a(nt as u64 * ctxd as u64);
    let vecd = a(du);
    // One weight buffer per distinct (n_out, k) — contents are irrelevant.
    let w_dd = a(du * du);
    let w_dm = a(mu * du);
    let w_md = a(du * mu);
    let w_txtin = a(du * ctxd as u64);
    let w_imgin = a(du * cin as u64);
    let w_final = a(cin as u64 * du);
    let nscale = a(hd as u64);

    // BRAIN_FLUX2_BENCH_BASELINE=1 profiles the PRE-FIX kernel set
    // (flash_attn_bidir + matmul_reg2 + rmsnorm_eps), so the before and after
    // tables come from one binary on one device in one run.
    let base = std::env::var("BRAIN_FLUX2_BENCH_BASELINE").as_deref() == Ok("1");
    let baseline_flash = base;
    let kmm = if base { K_MM } else { K_MM3 };

    let mut steps: Vec<Step> = Vec::new();
    let mut groups: Vec<(&'static str, String, usize, usize)> = Vec::new();
    let mut flop = 0.0f64;
    let mut open: Option<(&'static str, String, usize)> = None;
    macro_rules! grp {
        ($kind:expr, $shape:expr) => {
            if let Some((kk, ss, st)) = open.take() {
                groups.push((kk, ss, st, steps.len()));
            }
            open = Some(($kind, $shape, steps.len()));
        };
    }

    let mm = |steps: &mut Vec<Step>, flop: &mut f64, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, m: u32, kk: u32, nn: u32| {
        steps.push(gpu.step(kmm, &[x, w, o], &[m, kk, nn], m.div_ceil(128) * nn.div_ceil(128) * 256));
        *flop += 2.0 * m as f64 * kk as f64 * nn as f64;
    };
    // Sliced matmul, exactly as `Flux2Model::mm_rows`.
    let mm_rows = |steps: &mut Vec<Step>, flop: &mut f64, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, r0: u32, r1: u32, kk: u32, nn: u32| {
        let m = r1 - r0;
        let xo = (r0 as u64 * kk as u64, m as u64 * kk as u64);
        let oo = (r0 as u64 * nn as u64, m as u64 * nn as u64);
        steps.push(gpu.step_sliced(kmm, &[x, w, o], &[xo, (0, 0), oo], &[m, kk, nn], m.div_ceil(128) * nn.div_ceil(128) * 256));
        *flop += 2.0 * m as f64 * kk as f64 * nn as f64;
    };
    let ln_rows = |steps: &mut Vec<Step>, x: &DeviceBuffer, o: &DeviceBuffer, r0: u32, r1: u32| {
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        steps.push(gpu.step_sliced(K_LN, &[x, &vecd, &vecd, o], &[off, (0, 0), (0, 0), off], &[d, m, f(1e-6)], m));
    };
    let gate_rows = |steps: &mut Vec<Step>, x: &DeviceBuffer, h: &DeviceBuffer, y: &DeviceBuffer, r0: u32, r1: u32| {
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        steps.push(gpu.step_sliced(K_GATE, &[x, &vecd, h, y], &[off, (0, 0), off, off], &[m, d, m], m * d));
    };
    let qknorm_rows = |steps: &mut Vec<Step>, x: &DeviceBuffer, o: &DeviceBuffer, r0: u32, r1: u32| {
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        let rows = m * nh;
        let (kind, th) = if base { (K_RMS, rows) } else { (K_RMS_ROWS, rows * 64) };
        steps.push(gpu.step_sliced(kind, &[x, &nscale, o], &[off, (0, 0), off], &[hd, rows, f(1e-6)], th));
    };
    let rope_pack = |steps: &mut Vec<Step>| {
        let half = hd / 2;
        steps.push(gpu.step(K_ROPE, &[&qn, &cos, &sin, &qr], &[n, nh, hd, half], n * nh * half));
        steps.push(gpu.step(K_ROPE, &[&kn, &cos, &sin, &kr], &[n, nh, hd, half], n * nh * half));
        steps.push(gpu.step(K_PACK, &[&qr, &kr, &v, &qkv], &[n, d], n * 3 * d));
    };
    let flash = |steps: &mut Vec<Step>, flop: &mut f64| {
        let nwg = nh * n.div_ceil(64);
        if baseline_flash {
            steps.push(gpu.step(K_FLASH, &[&qkv, &ctx], &[1, nh, n, hd, 3 * d, 0, d, 2 * d, d], nwg * 64));
        } else {
            steps.push(gpu.step(K_SPLIT, &[&qkv, &ctx], &[1, nh, n, hd, 3 * d, 0, d, 2 * d, d], nwg * 256));
        }
        *flop += 4.0 * n as f64 * n as f64 * d as f64;
    };

    // --- boundary embeds -------------------------------------------------
    grp!("matmul", format!("txt_in {nt}x{ctxd}x{d}"));
    mm_rows(&mut steps, &mut flop, &ctx_in, &w_txtin, &x0, 0, nt, ctxd, d);
    grp!("matmul", format!("img_in {ni}x{cin}x{d}"));
    {
        let xo = (0u64, ni as u64 * cin as u64);
        let oo = (nt as u64 * d as u64, ni as u64 * d as u64);
        steps.push(gpu.step_sliced(kmm, &[&tok_in, &w_imgin, &x0], &[xo, (0, 0), oo], &[ni, cin, d], ni.div_ceil(128) * d.div_ceil(128) * 256));
        flop += 2.0 * ni as f64 * cin as f64 * d as f64;
    }

    let (mut xa, mut xb) = (&x0, &x1);
    for _ in 0..cfg.depth_double {
        grp!("layernorm", format!("dbl ln {nt}|{ni}x{d}"));
        ln_rows(&mut steps, xa, &n1, 0, nt);
        grp!("matmul", format!("dbl txt qkv {nt}x{d}x{d}"));
        for o in [&q, &k, &v] {
            mm_rows(&mut steps, &mut flop, &n1, &w_dd, o, 0, nt, d, d);
        }
        grp!("layernorm", format!("dbl ln img {ni}x{d}"));
        ln_rows(&mut steps, xa, &n1, nt, n);
        grp!("matmul", format!("dbl img qkv {ni}x{d}x{d}"));
        for o in [&q, &k, &v] {
            mm_rows(&mut steps, &mut flop, &n1, &w_dd, o, nt, n, d, d);
        }
        grp!("rmsnorm", format!("dbl qknorm {nt}|{ni}"));
        qknorm_rows(&mut steps, &q, &qn, 0, nt);
        qknorm_rows(&mut steps, &k, &kn, 0, nt);
        qknorm_rows(&mut steps, &q, &qn, nt, n);
        qknorm_rows(&mut steps, &k, &kn, nt, n);
        grp!("rope+pack", format!("dbl rope+pack n={n}"));
        rope_pack(&mut steps);
        grp!("flash_attn", format!("dbl flash n={n} hd={hd} nh={nh}"));
        flash(&mut steps, &mut flop);
        grp!("matmul", format!("dbl wo {nt}|{ni}x{d}x{d}"));
        mm_rows(&mut steps, &mut flop, &ctx, &w_dd, &proj, 0, nt, d, d);
        mm_rows(&mut steps, &mut flop, &ctx, &w_dd, &proj, nt, n, d, d);
        grp!("gate_row", format!("dbl gate {nt}|{ni}x{d}"));
        gate_rows(&mut steps, xa, &proj, xb, 0, nt);
        gate_rows(&mut steps, xa, &proj, xb, nt, n);
        std::mem::swap(&mut xa, &mut xb);
        grp!("layernorm", format!("dbl ln2 {nt}|{ni}x{d}"));
        ln_rows(&mut steps, xa, &n1, 0, nt);
        grp!("matmul", format!("dbl txt w1w3 {nt}x{d}x{mlp}"));
        mm_rows(&mut steps, &mut flop, &n1, &w_dm, &h1, 0, nt, d, mlp);
        mm_rows(&mut steps, &mut flop, &n1, &w_dm, &h2, 0, nt, d, mlp);
        grp!("layernorm", format!("dbl ln2 img {ni}x{d}"));
        ln_rows(&mut steps, xa, &n1, nt, n);
        grp!("matmul", format!("dbl img w1w3 {ni}x{d}x{mlp}"));
        mm_rows(&mut steps, &mut flop, &n1, &w_dm, &h1, nt, n, d, mlp);
        mm_rows(&mut steps, &mut flop, &n1, &w_dm, &h2, nt, n, d, mlp);
        grp!("silu_mul", format!("dbl silu {n}x{mlp}"));
        steps.push(gpu.step(K_SILU, &[&h1, &h2, &hs], &[n * mlp], n * mlp));
        grp!("matmul", format!("dbl w2 {nt}|{ni}x{mlp}x{d}"));
        mm_rows(&mut steps, &mut flop, &hs, &w_md, &mlpb, 0, nt, mlp, d);
        mm_rows(&mut steps, &mut flop, &hs, &w_md, &mlpb, nt, n, mlp, d);
        grp!("gate_row", format!("dbl gate2 {nt}|{ni}x{d}"));
        gate_rows(&mut steps, xa, &mlpb, xb, 0, nt);
        gate_rows(&mut steps, xa, &mlpb, xb, nt, n);
        std::mem::swap(&mut xa, &mut xb);
    }

    for _ in 0..cfg.depth_single {
        grp!("layernorm", format!("sgl ln {n}x{d}"));
        ln_rows(&mut steps, xa, &n1, 0, n);
        grp!("matmul", format!("sgl qkv {n}x{d}x{d}"));
        for o in [&q, &k, &v] {
            mm(&mut steps, &mut flop, &n1, &w_dd, o, n, d, d);
        }
        grp!("rmsnorm", format!("sgl qknorm {n}"));
        qknorm_rows(&mut steps, &q, &qn, 0, n);
        qknorm_rows(&mut steps, &k, &kn, 0, n);
        grp!("rope+pack", format!("sgl rope+pack n={n}"));
        rope_pack(&mut steps);
        grp!("flash_attn", format!("sgl flash n={n} hd={hd} nh={nh}"));
        flash(&mut steps, &mut flop);
        grp!("matmul", format!("sgl w1w3 {n}x{d}x{mlp}"));
        mm(&mut steps, &mut flop, &n1, &w_dm, &h1, n, d, mlp);
        mm(&mut steps, &mut flop, &n1, &w_dm, &h2, n, d, mlp);
        grp!("silu_mul", format!("sgl silu {n}x{mlp}"));
        steps.push(gpu.step(K_SILU, &[&h1, &h2, &hs], &[n * mlp], n * mlp));
        grp!("matmul", format!("sgl wo_a {n}x{d}x{d}"));
        mm(&mut steps, &mut flop, &ctx, &w_dd, &proj, n, d, d);
        grp!("matmul", format!("sgl wo_b {n}x{mlp}x{d}"));
        mm(&mut steps, &mut flop, &hs, &w_md, &mlpb, n, mlp, d);
        grp!("gate_row", format!("sgl gate {n}x{d}"));
        gate_rows(&mut steps, xa, &proj, xb, 0, n);
        std::mem::swap(&mut xa, &mut xb);
        gate_rows(&mut steps, xa, &mlpb, xb, 0, n);
        std::mem::swap(&mut xa, &mut xb);
    }

    grp!("layernorm", format!("final ln {ni}x{d}"));
    ln_rows(&mut steps, xa, &n1, nt, n);
    grp!("matmul", format!("final {ni}x{d}x{cin}"));
    {
        let xo = (nt as u64 * d as u64, ni as u64 * d as u64);
        let oo = (0u64, ni as u64 * cin as u64);
        steps.push(gpu.step_sliced(kmm, &[&n1, &w_final, &out], &[xo, (0, 0), oo], &[ni, d, cin], ni.div_ceil(128) * cin.div_ceil(128) * 256));
        flop += 2.0 * ni as f64 * d as f64 * cin as f64;
    }
    if let Some((kk, ss, st)) = open.take() {
        groups.push((kk, ss, st, steps.len()));
    }
    // Keep the scratch alive for the caller's timing runs.
    std::mem::forget((x0, x1, n1, q, k, v, qn, kn, qr, kr));
    std::mem::forget((qkv, ctx, proj, h1, h2, hs, mlpb, out, cos, sin));
    std::mem::forget((tok_in, ctx_in, vecd, w_dd, w_dm, w_md, w_txtin, w_imgin, w_final, nscale));
    (Replay { steps, groups }, flop)
}

fn bench_replay(gpu: &Gpu, reps: usize, lh: u32, lw: u32) {
    let cfg = Flux2Config::klein_4b();
    let nt = cfg.txt_len as u32;
    let ni = lh * lw;
    let (r, flop) = build_replay(gpu, &cfg, nt, ni);
    let n = nt + ni;
    println!(
        "\n=== klein-4B DiT graph replay: {n} joint tokens ({nt} txt + {ni} img), {} dispatches ===",
        r.steps.len()
    );
    println!("work: {:.0} GFLOP/forward", flop / 1e9);

    // Whole graph in ONE submit — the production path.
    let whole = time_steps(gpu, &r.steps, reps);
    println!(
        "whole graph, single submit: {:.3} s -> {:.0} GFLOP/s ({:.1}% of P40 peak)",
        whole,
        flop / 1e9 / whole,
        pct(flop / 1e9 / whole)
    );

    // Per kind/shape: drain between groups. Costs one round-trip per group.
    let mut agg: std::collections::BTreeMap<(&str, String), (f64, usize)> = Default::default();
    let mut sum = 0.0;
    for (kind, shape, s0, s1) in &r.groups {
        let t = time_steps(gpu, &r.steps[*s0..*s1], reps.max(2));
        let e = agg.entry((kind, shape.clone())).or_insert((0.0, 0));
        e.0 += t;
        e.1 += s1 - s0;
        sum += t;
    }
    let mut rows: Vec<_> = agg.into_iter().collect();
    rows.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    println!("\nper group (queue drained between groups; {} groups × 1 round-trip overhead):", r.groups.len());
    println!("{:<10} {:<30} {:>5} {:>10} {:>7}", "kind", "shape", "disp", "ms", "%");
    for ((kind, shape), (t, c)) in &rows {
        println!("{kind:<10} {shape:<30} {c:>5} {:>10.1} {:>6.1}%", t * 1e3, 100.0 * t / sum);
    }
    println!("sum of groups: {sum:.3} s (vs {whole:.3} s in one submit)");

    // Roll up by kind.
    let mut by_kind: std::collections::BTreeMap<&str, (f64, usize)> = Default::default();
    for ((kind, _), (t, c)) in &rows {
        let e = by_kind.entry(kind).or_insert((0.0, 0));
        e.0 += t;
        e.1 += c;
    }
    let mut kr: Vec<_> = by_kind.into_iter().collect();
    kr.sort_by(|a, b| b.1 .0.total_cmp(&a.1 .0));
    println!("\nby kind:");
    for (kind, (t, c)) in kr {
        println!("  {kind:<12} {c:>5} disp {:>9.1} ms  {:>5.1}%", t * 1e3, 100.0 * t / sum);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("replay");
    let gpu = Gpu::new_wgpu(KERNELS);
    let c = gpu.caps();
    eprintln!(
        "device: {} class={:?} cu={} max_wg={} wg_mem={}B",
        gpu.kind(),
        c.class,
        c.compute_units.unwrap_or(0),
        c.max_workgroup_size,
        c.workgroup_mem_bytes
    );
    match mode {
        "mm" => bench_mm(&gpu, args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5)),
        "floor" => bench_floor(&gpu, args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500)),
        // `flash n heads hd`: time BOTH bidirectional flash kernels at one
        // shape and check they agree numerically on the same random qkv.
        "flash" => {
            let n: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1536);
            let nh: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(24);
            let hd: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(128);
            let d = nh * hd;
            let qkv = gpu.storage(n as u64 * 3 * d as u64);
            let src: Vec<f32> = (0..(n as usize * 3 * d as usize))
                .map(|i| ((i as f64 * 0.7).sin() * 0.5) as f32)
                .collect();
            gpu.write(&qkv, bytemuck::cast_slice(&src));
            let prm = [1, nh, n, hd, 3 * d, 0, d, 2 * d, d];
            let mut res = Vec::new();
            for (name, kind, ws) in [("flash_attn_bidir", K_FLASH, 64u32), ("flash_attn_bidir_split", K_SPLIT, 256u32)] {
                let o = gpu.storage(n as u64 * d as u64);
                let nwg = nh * n.div_ceil(64);
                let st = vec![gpu.step(kind, &[&qkv, &o], &prm, nwg * ws)];
                let t = time_steps(&gpu, &st, 5);
                let gf = 4.0 * n as f64 * n as f64 * d as f64 / 1e9;
                println!(
                    "{name:<24} n={n} nh={nh} hd={hd}: {:.2} ms  {:.0} GFLOP/s  {:.2}% peak",
                    t * 1e3,
                    gf / t,
                    pct(gf / t)
                );
                res.push(gpu.read(&o, (n * d) as usize));
            }
            let (a, b) = (&res[0], &res[1]);
            let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f32);
            for (&x, &y) in a.iter().zip(b) {
                dot += x as f64 * y as f64;
                na += x as f64 * x as f64;
                nb += y as f64 * y as f64;
                mx = mx.max((x - y).abs());
            }
            println!("agreement: cosine {:.8}  max_abs {mx:.3e}", dot / (na.sqrt() * nb.sqrt()));
        }
        // `mm3 [reps]`: matmul_reg2 vs matmul_reg3 at the DiT's shapes, with a
        // numeric cross-check on the first shape.
        "mm3" => {
            let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
            let cfg = Flux2Config::klein_4b();
            println!("\n{:<22} {:>6} {:>6} {:>6} {:>10} {:>10} {:>8}", "shape", "m", "k", "n", "reg2 ms", "reg3 ms", "speedup");
            let (mut t2tot, mut t3tot, mut gtot) = (0.0, 0.0, 0.0);
            for &(m, k, n, count, label) in &mm_shapes(&cfg, 512, 1024) {
                let x = gpu.storage(m as u64 * k as u64);
                let w = gpu.storage(n as u64 * k as u64);
                let src: Vec<f32> = (0..(m as usize * k as usize)).map(|i| ((i as f64 * 0.37).sin() * 0.3) as f32).collect();
                gpu.write(&x, bytemuck::cast_slice(&src));
                let srw: Vec<f32> = (0..(n as usize * k as usize)).map(|i| ((i as f64 * 0.11).cos() * 0.3) as f32).collect();
                gpu.write(&w, bytemuck::cast_slice(&srw));
                let mut out = Vec::new();
                let mut ts = Vec::new();
                for kind in [K_MM, K_MM3] {
                    let o = gpu.storage(m as u64 * n as u64);
                    let st: Vec<Step> = (0..4)
                        .map(|_| gpu.step(kind, &[&x, &w, &o], &[m, k, n], m.div_ceil(128) * n.div_ceil(128) * 256))
                        .collect();
                    ts.push(time_steps(&gpu, &st, reps) / 4.0);
                    out.push(gpu.read(&o, (m.min(64) * n) as usize));
                }
                let mx = out[0].iter().zip(&out[1]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                let gf = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
                t2tot += ts[0] * count as f64;
                t3tot += ts[1] * count as f64;
                gtot += gf * count as f64;
                println!(
                    "{label:<22} {m:>6} {k:>6} {n:>6} {:>10.3} {:>10.3} {:>7.2}x   reg2 {:.0} GF/s ({:.1}%) reg3 {:.0} GF/s ({:.1}%) maxdiff {mx:.2e}",
                    ts[0] * 1e3, ts[1] * 1e3, ts[0] / ts[1],
                    gf / ts[0], pct(gf / ts[0]), gf / ts[1], pct(gf / ts[1])
                );
            }
            println!(
                "forward matmul total: reg2 {:.0} ms ({:.1}% peak) -> reg3 {:.0} ms ({:.1}% peak) = {:.2}x",
                t2tot * 1e3, pct(gtot / t2tot), t3tot * 1e3, pct(gtot / t3tot), t2tot / t3tot
            );
        }
        // `norm`: the QK-norm shape — one thread per row (rmsnorm_eps, today)
        // vs one workgroup per row (rmsnorm_rows). eps is 1e-6 in both.
        "norm" => {
            let rows: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1536 * 24);
            let d: u32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(128);
            let x = gpu.storage(rows as u64 * d as u64);
            let src: Vec<f32> = (0..(rows as usize * d as usize)).map(|i| ((i as f64 * 0.21).sin()) as f32).collect();
            gpu.write(&x, bytemuck::cast_slice(&src));
            let w = gpu.storage(d as u64);
            gpu.write(&w, bytemuck::cast_slice(&vec![1.25f32; d as usize]));
            let mut outs = Vec::new();
            for (name, kind, prm, threads) in [
                ("rmsnorm_eps", K_RMS, vec![d, rows, f(1e-6)], rows),
                ("rmsnorm_rows", K_RMS_ROWS, vec![d, rows], rows * 64),
            ] {
                let o = gpu.storage(rows as u64 * d as u64);
                let st = vec![gpu.step(kind, &[&x, &w, &o], &prm, threads)];
                let t = time_steps(&gpu, &st, 8);
                let bytes = 2.0 * rows as f64 * d as f64 * 4.0;
                println!("{name:<14} rows={rows} d={d}: {:.3} ms  {:.0} GB/s", t * 1e3, bytes / t / 1e9);
                outs.push(gpu.read(&o, (rows.min(4096) * d) as usize));
            }
            let mx = outs[0].iter().zip(&outs[1]).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            println!("max_abs difference: {mx:.3e}");
        }
        "mmone" => {
            let m: u32 = args[2].parse().unwrap();
            let k: u32 = args[3].parse().unwrap();
            let nn: u32 = args[4].parse().unwrap();
            let x = gpu.storage(m as u64 * k as u64);
            let w = gpu.storage(nn as u64 * k as u64);
            let o = gpu.storage(m as u64 * nn as u64);
            let steps: Vec<Step> = (0..4).map(|_| mm_step(&gpu, &x, &w, &o, m, k, nn)).collect();
            let s = time_steps(&gpu, &steps, 8) / 4.0;
            let gf = 2.0 * m as f64 * k as f64 * nn as f64 / 1e9;
            println!("{m}x{k}x{nn}: {:.3} ms/disp  {:.0} GFLOP/s  {:.1}% peak", s * 1e3, gf / s, pct(gf / s));
        }
        "replay" => bench_replay(
            &gpu,
            args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3),
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(32),
            args.get(4).and_then(|s| s.parse().ok()).unwrap_or(32),
        ),
        other => {
            eprintln!("unknown mode {other} (mm|floor|replay)");
            std::process::exit(1);
        }
    }
}
