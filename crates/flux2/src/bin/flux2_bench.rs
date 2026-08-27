// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! FLUX.2 DiT profiler: where the forward's time actually goes, per matmul
//! shape and per step kind, against the card's own datasheet fp32 peak.
//!
//! The DiT forward is a fixed sequence of dispatches whose *cost depends only
//! on shape*, so this drives correctly-shaped scratch instead of the 15.5 GiB
//! checkpoint (same idea as `zimage_bench train`). That makes the profile
//! runnable in seconds, with no weights, and lets each shape class be timed in
//! isolation. Total analytic FLOP is asserted against the real graph's own
//! FLOP count so the replay cannot silently drift from `model.rs`.
//!
//! Usage:
//!   flux2_bench mm      [reps]        standalone matmul at the DiT's shapes
//!   flux2_bench floor   [n]           per-dispatch floor (tiny kernel × n)
//!   flux2_bench replay  [reps] [h w]  the whole DiT graph, per shape + kind
//!   flux2_bench te      [reps]        the Qwen3-4B 512-token TE prefill, per kind
//!   flux2_bench tei8    [reps]        ...the INT8 (DP4A) shard of the same
//!   flux2_bench vae     [reps] [h w]  the FLUX.2 VAE decode graph, per kind
//!   flux2_bench load    [gguf] [var]  where the one-off weight load spends host time
//!
//! `BRAIN_FLUX2_BENCH_BASELINE=1` profiles the PRE-optimization kernel set
//! (`replay` and `te`/`tei8`), so before/after come from one binary in one run.
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

// -------------------------------------------------------------- load ------

/// Whether `model.rs` uploads this tensor as packed int8 under
/// [`Precision::Int8`]. Mirrors the tier decisions in `Flux2Model::
/// new_batched` - the double-block mlp-down, the three boundary linears, the
/// qk-norm scale vectors and the host-resident conditioning matrices all stay
/// fp32 there, so quantizing them here would overstate the load's real cost.
fn is_int8_tier(name: &str, shape: &[usize]) -> bool {
    if shape.len() != 2 || !shape[1].is_multiple_of(4) {
        return false;
    }
    !(name.ends_with("_mlp.2.weight")
        || name == "img_in.weight"
        || name == "txt_in.weight"
        || name == "final_layer.linear.weight"
        || name.contains("modulation")
        || name.starts_with("time_in.")
        || name.starts_with("guidance_in."))
}

/// Where the one-off per-process weight load actually spends its host time.
///
/// Times the production functions, in the production order, on a real
/// checkpoint: `checkpoint::gguf::read` (slurp + dequantize to fp32),
/// `flux2::import_bfl` (name map + two-way manifest coverage) and
/// `model::int8::quantize_weight` over exactly the tensors the int8 tier
/// packs. Host only - no GPU, so it runs while a card is busy; the upload is
/// the remaining term and is timed on the device by `BRAIN_PROFILE`.
fn bench_load(path: &str, variant: &str) {
    let cfg = match Flux2Config::from_name(variant) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    println!("\n=== weight load, host side: {variant}, {path} ===");
    let t0 = Instant::now();
    let raw = match checkpoint::gguf::read(path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let t_read = t0.elapsed().as_secs_f64();
    let elems: usize = raw.iter().map(|t| t.data.len()).sum();
    let nbytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let t1 = Instant::now();
    let ts = match flux2::import_bfl(raw, &cfg) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let t_import = t1.elapsed().as_secs_f64();

    let t2 = Instant::now();
    let (mut q_elems, mut q_count) = (0usize, 0usize);
    for (name, (shape, data)) in &ts {
        if !is_int8_tier(name, shape) {
            continue;
        }
        let (n, k) = (shape[0], shape[1]);
        let _ = model::int8::quantize_weight(data, n, k);
        q_elems += data.len();
        q_count += 1;
    }
    let t_quant = t2.elapsed().as_secs_f64();
    let total = t_read + t_import + t_quant;

    let gb = |e: usize| e as f64 * 4.0 / 1e9;
    println!("  file {:.2} GB, {} tensors, {:.2} G params ({:.1} GB fp32)", nbytes as f64 / 1e9, ts.len(), elems as f64 / 1e9, gb(elems));
    println!("  {:<28} {:>8.2} s  ({:>4.1}%)   slurp + Q8_0 -> fp32", "gguf::read", t_read, 100.0 * t_read / total);
    println!("  {:<28} {:>8.2} s  ({:>4.1}%)   name map + manifest coverage", "import_bfl", t_import, 100.0 * t_import / total);
    println!(
        "  {:<28} {:>8.2} s  ({:>4.1}%)   {q_count} tensors, {:.1} GB fp32 read",
        "quantize_weight", t_quant, 100.0 * t_quant / total, gb(q_elems)
    );
    println!("  {:<28} {:>8.2} s", "TOTAL host load", total);
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

// ---------------------------------------------------------- profiler ------

/// Per-kernel-kind profile of an arbitrary recorded graph.
///
/// Every `Step` built through the `gpu_core` facade carries a `StepMeta` naming
/// the kernel slot it dispatches, so the table below is derived from the graph
/// itself instead of being hand-annotated (as `bench_replay`'s `grp!` groups
/// are). Each kind is timed by submitting *only* its steps: one queue
/// round-trip per kind instead of one per group, and the isolation is sound
/// because a dispatch's cost depends on its shape, not on what ran before it.
/// The whole graph in one submit is timed too, so the sum-vs-whole gap bounds
/// the instrumentation error.
fn profile_kinds(gpu: &Gpu, steps: &[Step], names: &[(&str, &str)], reps: usize, flop: f64, bytes: f64) {
    let whole = time_steps(gpu, steps, reps);
    println!("\nwhole graph, single submit: {:.3} s ({} dispatches)", whole, steps.len());
    if flop > 0.0 {
        println!(
            "work: {:.0} GFLOP -> {:.0} GFLOP/s ({:.1}% of P40 fp32 peak)",
            flop / 1e9,
            flop / 1e9 / whole,
            pct(flop / 1e9 / whole)
        );
    }
    if bytes > 0.0 {
        println!("minimum traffic: {:.1} GB -> {:.0} GB/s of the P40's 346", bytes / 1e9, bytes / whole / 1e9);
    }

    let mut by_kind: std::collections::BTreeMap<usize, Vec<Step>> = Default::default();
    for s in steps {
        let k = s.meta().expect("step built through the facade").kernel;
        by_kind.entry(k).or_default().push(s.clone());
    }
    let mut rows: Vec<(usize, usize, f64)> = Vec::new();
    let mut sum = 0.0;
    for (k, v) in &by_kind {
        let t = time_steps(gpu, v, reps);
        sum += t;
        rows.push((*k, v.len(), t));
    }
    rows.sort_by(|a, b| b.2.total_cmp(&a.2));
    println!("\n{:<24} {:>6} {:>10} {:>8}", "kernel", "disp", "ms", "% phase");
    for (k, c, t) in &rows {
        println!("{:<24} {c:>6} {:>10.1} {:>7.1}%", names[*k].0, t * 1e3, 100.0 * t / sum);
    }
    println!("sum of kinds: {:.3} s (vs {:.3} s in one submit)", sum, whole);

    // The dominant kind again, split by uniform params (= by shape): which
    // instance of it is the cost, not just which kernel.
    if let Some((top, _, _)) = rows.first() {
        let mut by_shape: std::collections::BTreeMap<Vec<u32>, Vec<Step>> = Default::default();
        for s in &by_kind[top] {
            let m = s.meta().unwrap();
            by_shape.entry(m.params.clone().unwrap_or_default()).or_default().push(s.clone());
        }
        let mut sh: Vec<(String, usize, f64)> = by_shape
            .iter()
            .map(|(p, v)| (format!("{p:?}"), v.len(), time_steps(gpu, v, reps)))
            .collect();
        sh.sort_by(|a, b| b.2.total_cmp(&a.2));
        println!("\n{} by shape (params):", names[*top].0);
        for (p, c, t) in &sh {
            println!("  {p:<48} {c:>4} {:>9.1} ms", t * 1e3);
        }
    }
}

// ------------------------------------------------------------- te ---------

/// Kernel slots for the Qwen3 text-encoder prefill replay. Order is this
/// bench's own; the *names* are what the profile prints.
const TE_KERNELS: &[(&str, &str)] = &[
    ("rmsnorm", kernels::RMSNORM),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("rope_base", kernels::ROPE_BASE),
    ("gqa_scores_kmask", kernels::GQA_SCORES_KMASK),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("softmax_rows", kernels::SOFTMAX_ROWS),
    ("matmul_reg3", kernels::MATMUL_REG3),
];
const T_RMS: usize = 0;
const T_MM: usize = 1;
const T_ROPE: usize = 2;
const T_SCORES: usize = 3;
const T_SOFTMAX: usize = 4;
const T_APPLY: usize = 5;
const T_SILU: usize = 6;
const T_ADD: usize = 7;
const T_MAXABS: usize = 8;
const T_QPACK: usize = 9;
const T_MM8: usize = 10;
const T_RMS_ROWS: usize = 11;
const T_SM_ROWS: usize = 12;
const T_MM3: usize = 13;

/// Replay of `Qwen::forward_steps` for the FLUX.2 text encoder: ONE 512-token
/// prefill of the layer-27-truncated Qwen3-4B, masked-pad (`gqa_scores_kmask`),
/// no head. Shapes only — no weights — exactly as `build_replay` does for the
/// DiT.
///
/// `i8` swaps the 7 per-layer linears for the DP4A path (`max_abs_row` +
/// `quant_pack` + `matmul_i8_dyn`), matching `Qwen::new_shard_i8`.
/// `base` selects the PRE-fix kernel set (`rmsnorm` + `attn_softmax` +
/// `matmul_reg2`) instead of the one qwen now dispatches, so the before and
/// after tables come from one binary on one device in one run — the same
/// `BRAIN_FLUX2_BENCH_BASELINE` convention the DiT replay uses.
fn build_te_replay(gpu: &Gpu, layers: u32, t: u32, i8: bool, base: bool) -> (Vec<Step>, f64) {
    let (d, ff, hd, nh, nkv) = (2560u32, 9728u32, 128u32, 32u32, 8u32);
    let (hq, hkv) = (nh * hd, nkv * hd);
    let group = nh / nkv;
    let n = t;
    let a = |len: u64| gpu.storage(len);
    let (du, ffu, nu) = (d as u64, ff as u64, n as u64);

    let res = a(nu * du);
    let xn = a(nu * du);
    let q_pre = a(nu * hq as u64);
    let k_pre = a(nu * hkv as u64);
    let v = a(nu * hkv as u64);
    let q = a(nu * hq as u64);
    let k = a(nu * hkv as u64);
    let scores = a(nh as u64 * t as u64 * t as u64);
    let probs = a(nh as u64 * t as u64 * t as u64);
    let ctx = a(nu * hq as u64);
    let proj = a(nu * du);
    let xmid = a(nu * du);
    let gate = a(nu * ffu);
    let up = a(nu * ffu);
    let hbuf = a(nu * ffu);
    let mlp_out = a(nu * du);
    let kmask = a(t as u64);
    let gain_d = a(du);
    let gain_hd = a(hd as u64);
    // fp32 weights, one buffer per distinct (n_out, k).
    let w_q = a(hq as u64 * du);
    let w_kv = a(hkv as u64 * du);
    let w_o = a(du * hq as u64);
    let w_ff = a(ffu * du);
    let w_down = a(du * ffu);
    // int8 weights: packed [n, k/4] u32 + per-channel scale [n].
    let p_q = a(hq as u64 * du / 4);
    let p_kv = a(hkv as u64 * du / 4);
    let p_o = a(du * hq as u64 / 4);
    let p_ff = a(ffu * du / 4);
    let p_down = a(du * ffu / 4);
    let s_q = a(hq as u64);
    let s_kv = a(hkv as u64);
    let s_o = a(du);
    let s_ff = a(ffu);
    let s_down = a(du);
    let sx = a(nu);
    let xq = a(nu * ffu / 4);

    let kmm = if base { T_MM } else { T_MM3 };
    let mut s: Vec<Step> = Vec::new();
    let mut flop = 0.0f64;
    let mm = |s: &mut Vec<Step>, flop: &mut f64, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, kk: u32, nn: u32| {
        s.push(gpu.step(kmm, &[x, w, o], &[n, kk, nn], n.div_ceil(128) * nn.div_ceil(128) * 256));
        *flop += 2.0 * n as f64 * kk as f64 * nn as f64;
    };
    let quant = |s: &mut Vec<Step>, x: &DeviceBuffer, kk: u32| {
        s.push(gpu.step(T_MAXABS, &[x, &sx], &[n, kk], n));
        s.push(gpu.step(T_QPACK, &[x, &sx, &xq], &[n, kk], n * kk / 4));
    };
    let mm8 = |s: &mut Vec<Step>, flop: &mut f64, pw: &DeviceBuffer, sw: &DeviceBuffer, o: &DeviceBuffer, kk: u32, nn: u32| {
        s.push(gpu.step(T_MM8, &[&xq, pw, &sx, sw, o], &[n, kk / 4, nn], n.div_ceil(128) * nn.div_ceil(128) * 256));
        *flop += 2.0 * n as f64 * kk as f64 * nn as f64;
    };
    let rms = |s: &mut Vec<Step>, x: &DeviceBuffer, g: &DeviceBuffer, o: &DeviceBuffer, dim: u32, rows: u32| {
        if base {
            s.push(gpu.step(T_RMS, &[x, g, o], &[dim, rows], rows));
        } else {
            s.push(gpu.step(T_RMS_ROWS, &[x, g, o], &[dim, rows, f(1e-6)], rows * 64));
        }
    };

    for _ in 0..layers {
        rms(&mut s, &res, &gain_d, &xn, d, n);
        if i8 {
            quant(&mut s, &xn, d);
            mm8(&mut s, &mut flop, &p_q, &s_q, &q_pre, d, hq);
            mm8(&mut s, &mut flop, &p_kv, &s_kv, &k_pre, d, hkv);
            mm8(&mut s, &mut flop, &p_kv, &s_kv, &v, d, hkv);
        } else {
            mm(&mut s, &mut flop, &xn, &w_q, &q_pre, d, hq);
            mm(&mut s, &mut flop, &xn, &w_kv, &k_pre, d, hkv);
            mm(&mut s, &mut flop, &xn, &w_kv, &v, d, hkv);
        }
        rms(&mut s, &q_pre, &gain_hd, &q, hd, n * nh);
        rms(&mut s, &k_pre, &gain_hd, &k, hd, n * nkv);
        let half = hd / 2;
        s.push(gpu.step(T_ROPE, &[&q], &[n, nh, hd, hq, 0, t, f(1.0e6)], n * nh * half));
        s.push(gpu.step(T_ROPE, &[&k], &[n, nkv, hd, hkv, 0, t, f(1.0e6)], n * nkv * half));
        let ap = [1, nh, nkv, t, hd, group];
        s.push(gpu.step(T_SCORES, &[&q, &k, &kmask, &scores], &ap, nh * t * t));
        if base {
            s.push(gpu.step(T_SOFTMAX, &[&scores, &probs], &[1, nh, t], nh * t));
        } else {
            s.push(gpu.step(T_SM_ROWS, &[&scores, &probs], &[nh * t, t], nh * t * 64));
        }
        s.push(gpu.step(T_APPLY, &[&probs, &v, &ctx], &ap, nh * t * hd));
        // scores+apply FLOP (causal: t(t+1)/2 pairs, 2 FLOP each, both passes).
        flop += 2.0 * 2.0 * nh as f64 * (t as f64 * (t as f64 + 1.0) / 2.0) * hd as f64;
        if i8 {
            quant(&mut s, &ctx, hq);
            mm8(&mut s, &mut flop, &p_o, &s_o, &proj, hq, d);
        } else {
            mm(&mut s, &mut flop, &ctx, &w_o, &proj, hq, d);
        }
        s.push(gpu.step(T_ADD, &[&res, &proj, &xmid], &[n * d], n * d));
        rms(&mut s, &xmid, &gain_d, &xn, d, n);
        if i8 {
            quant(&mut s, &xn, d);
            mm8(&mut s, &mut flop, &p_ff, &s_ff, &gate, d, ff);
            mm8(&mut s, &mut flop, &p_ff, &s_ff, &up, d, ff);
        } else {
            mm(&mut s, &mut flop, &xn, &w_ff, &gate, d, ff);
            mm(&mut s, &mut flop, &xn, &w_ff, &up, d, ff);
        }
        s.push(gpu.step(T_SILU, &[&gate, &up, &hbuf], &[n * ff], n * ff));
        if i8 {
            quant(&mut s, &hbuf, ff);
            mm8(&mut s, &mut flop, &p_down, &s_down, &mlp_out, ff, d);
        } else {
            mm(&mut s, &mut flop, &hbuf, &w_down, &mlp_out, ff, d);
        }
        s.push(gpu.step(T_ADD, &[&xmid, &mlp_out, &res], &[n * d], n * d));
    }
    std::mem::forget((res, xn, q_pre, k_pre, v, q, k, scores, probs, ctx));
    std::mem::forget((proj, xmid, gate, up, hbuf, mlp_out, kmask, gain_d, gain_hd));
    std::mem::forget((w_q, w_kv, w_o, w_ff, w_down));
    std::mem::forget((p_q, p_kv, p_o, p_ff, p_down, s_q, s_kv, s_o, s_ff, s_down, sx, xq));
    (s, flop)
}

fn bench_te(reps: usize, i8: bool, base: bool) {
    let gpu = Gpu::new_wgpu(TE_KERNELS);
    eprintln!("device: {} max_wg={}", gpu.kind(), gpu.caps().max_workgroup_size);
    let (layers, t) = (28u32, 512u32);
    let (steps, flop) = build_te_replay(&gpu, layers, t, i8, base);
    println!(
        "\n=== FLUX.2 text encoder: Qwen3-4B {layers}-layer prefill, {t} tokens{}{} ===",
        if i8 { ", INT8 (DP4A)" } else { ", fp32" },
        if base { ", PRE-FIX kernel set" } else { "" }
    );
    profile_kinds(&gpu, &steps, TE_KERNELS, reps, flop, 0.0);
}

// ------------------------------------------------------------ vae ---------

/// Profile the FLUX.2 VAE decode graph. Needs the real checkpoint (the graph is
/// built from its tensors), pointed at by `BRAIN_FLUX2_VAE`; the *timing* still
/// depends only on shape.
fn bench_vae(reps: usize, lh: u32, lw: u32) {
    let path = std::env::var("BRAIN_FLUX2_VAE").expect("set BRAIN_FLUX2_VAE to the vae dir");
    let vp = std::path::Path::new(&path);
    let (file, json) = if vp.is_dir() {
        (vp.join("diffusion_pytorch_model.safetensors"), std::fs::read_to_string(vp.join("config.json")).ok())
    } else {
        (vp.to_path_buf(), None)
    };
    let cfg = match json {
        Some(j) => vae::VaeConfig::from_json(&serde_json::from_str(&j).unwrap()),
        None => vae::VaeConfig::flux2(),
    };
    let mut map = std::collections::HashMap::new();
    for t in checkpoint::safetensors::read(file.to_str().unwrap()).expect("read vae") {
        map.insert(t.name, (t.shape, t.data));
    }
    let dec = vae::VaeDecoder::from_diffusers(cfg, &map, lh, lw, Some("gpu"));
    drop(map);
    println!("\n=== FLUX.2 VAE decode: [32,{lh},{lw}] latent -> [3,{},{}] image ===", lh * 8, lw * 8);
    profile_kinds(dec.gpu(), dec.steps(), &vae::decoder::KERNELS, reps, 0.0, 0.0);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("replay");
    let arg = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    let base = std::env::var("BRAIN_FLUX2_BENCH_BASELINE").as_deref() == Ok("1");
    match mode {
        "te" => return bench_te(arg(2, 3), false, base),
        "tei8" => return bench_te(arg(2, 3), true, base),
        "vae" => return bench_vae(arg(2, 3), arg(3, 64) as u32, arg(4, 64) as u32),
        "load" => {
            let path = args.get(2).cloned().or_else(|| std::env::var("BRAIN_FLUX2_DIT").ok());
            let Some(path) = path else {
                eprintln!("load: pass a .gguf path or set BRAIN_FLUX2_DIT");
                std::process::exit(1);
            };
            return bench_load(&path, args.get(3).map(|s| s.as_str()).unwrap_or("klein-9b"));
        }
        _ => {}
    }
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
