// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wan2.1 profiler: where a generation's device time actually goes, per kernel
//! kind, against the Tesla P40 fp32 peak (11.76 TFLOP/s).
//!
//! Both graphs a generation submits are a fixed sequence of dispatches whose
//! *cost depends only on shape*, so this drives correctly-shaped scratch and
//! needs **no checkpoint on disk**: the tensor manifests (`import::dit_manifest`,
//! `WanVaeConfig::tensor_manifest`) name every tensor and its shape, and a
//! zero-filled source of exactly those shapes builds the same graph the real
//! weights would. That is deliberately a step further than a hand-written
//! replay: the graph profiled here is the production one, recorded by
//! `WanDitDev::build` / `WanVaeDecoder::build` themselves, so it cannot drift
//! from what a generation runs.
//!
//! Usage:
//!   wan_bench dit  [reps] [frames] [w] [h]   the DiT block stack, per kind
//!   wan_bench vae  [reps] [frames] [w] [h]   the VAE decode graph, per kind
//!   wan_bench host [reps] [frames] [w] [h]   the HOST stages either side of it
//!   wan_bench floor [n]                      per-dispatch floor (tiny kernel x n)
//!   wan_bench flash [reps] [T] [nh] [hd]     A/B every bidirectional flash kernel
//!
//! Defaults are the measured end-to-end point: 33 frames at 832x480, i.e.
//! 14,040 DiT tokens and a 9-latent-frame decode.
//!
//! `BRAIN_GPU_INDEX=0` selects a card. Per-kind timings drain the queue between
//! kinds; that adds one queue round-trip (~the `floor` number) per kind, which
//! is why `floor` exists and why the sum-of-kinds is reported next to the
//! single-submit whole-graph time.

use std::collections::BTreeMap;
use std::time::Instant;

use gpu_core::{Gpu, Step};
use wan::{WanConfig, WanVaeConfig, WanVaeDecoder};

const P40_FP32_TFLOPS: f64 = 11.76;

fn pct(gflops: f64) -> f64 {
    100.0 * gflops / (P40_FP32_TFLOPS * 1e3)
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

/// A weight-free tensor source: every name in `manifest` at its real shape,
/// zero-filled. Timing depends on shape, not on values, and every kernel in
/// both graphs is value-independent (no early exit, no data-dependent branch).
fn zeros(manifest: &[(String, Vec<usize>)]) -> wan::model::Tensors {
    manifest
        .iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            (name.clone(), (shape.clone(), vec![0.0f32; n]))
        })
        .collect()
}

/// Per-kernel-kind profile of an arbitrary recorded graph.
///
/// Every `Step` built through the `gpu_core` facade carries a `StepMeta` naming
/// the kernel slot it dispatches, so this table is derived from the graph
/// itself rather than hand-annotated. Each kind is timed by submitting *only*
/// its steps: one queue round-trip per kind, and the isolation is sound because
/// a dispatch's cost depends on its shape, not on what ran before it. The whole
/// graph in one submit is timed too, so the sum-vs-whole gap bounds the
/// instrumentation error.
fn profile_kinds(gpu: &Gpu, steps: &[Step], names: &[(&str, &str)], reps: usize, flop: f64) {
    let whole = time_steps(gpu, steps, reps);
    println!("\nwhole graph, single submit: {:.3} s ({} dispatches)", whole, steps.len());
    if flop > 0.0 {
        println!(
            "work: {:.0} GFLOP -> {:.0} GFLOP/s ({:.2}% of P40 fp32 peak)",
            flop / 1e9,
            flop / 1e9 / whole,
            pct(flop / 1e9 / whole)
        );
    }

    let mut by_kind: BTreeMap<usize, Vec<Step>> = Default::default();
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
    println!("\n{:<24} {:>7} {:>11} {:>9}", "kernel", "disp", "ms", "% graph");
    for (k, c, t) in &rows {
        println!("{:<24} {c:>7} {:>11.1} {:>8.1}%", names[*k].0, t * 1e3, 100.0 * t / sum);
    }
    println!("sum of kinds: {:.3} s (vs {:.3} s in one submit)", sum, whole);

    // The three dominant kinds again, split by uniform params (= by shape):
    // WHICH instance of a kind is the cost, not just which kernel. Three
    // rather than one because the top kind is rarely a majority on its own,
    // and an optimization pass needs the runner-up's shapes too.
    for (top, _, _) in rows.iter().take(3) {
        let mut by_shape: BTreeMap<Vec<u32>, Vec<Step>> = Default::default();
        for s in &by_kind[top] {
            let m = s.meta().unwrap();
            by_shape.entry(m.params.clone().unwrap_or_default()).or_default().push(s.clone());
        }
        let mut sh: Vec<(String, usize, f64)> =
            by_shape.iter().map(|(p, v)| (format!("{p:?}"), v.len(), time_steps(gpu, v, reps))).collect();
        sh.sort_by(|a, b| b.2.total_cmp(&a.2));
        println!("\n{} by shape (params):", names[*top].0);
        for (p, c, t) in sh.iter().take(12) {
            println!("  {p:<52} {c:>5} {:>10.1} ms", t * 1e3);
        }
    }
}

// -------------------------------------------------------------- dit -------

/// Analytic FLOP of one DiT forward's device graph: the eight `dim x dim`
/// projections and two feed-forward matrices per block, plus both attentions.
/// The host ends (patchify, the timestep MLPs, `text_embedding`, the head) are
/// not in the graph and are not counted here.
fn dit_flop(cfg: &WanConfig, t: f64) -> f64 {
    let (d, ff, te) = (cfg.dim as f64, cfg.ffn_dim as f64, cfg.text_len as f64);
    let per_block =
        // self q,k,v,o
        4.0 * 2.0 * t * d * d
        // self attention: scores + apply
        + 4.0 * t * t * d
        // cross q,o over t rows; cross k,v over the 512 text rows
        + 2.0 * 2.0 * t * d * d + 2.0 * 2.0 * te * d * d
        // cross attention: scores + apply
        + 4.0 * t * te * d
        // ffn
        + 2.0 * 2.0 * t * d * ff;
    per_block * cfg.num_layers as f64
}

fn bench_dit(reps: usize, frames: usize, w: usize, h: usize) {
    let cfg = WanConfig::t2v_1_3b();
    let tokens = cfg.token_count(frames, w, h).expect("frames must be 1 + 4k");
    let (lf, lh, lw) = cfg.latent_shape(frames, w, h).unwrap();
    let manifest = wan::import::dit_manifest(&cfg);
    println!(
        "\n=== Wan {} DiT: {frames} frames at {w}x{h} -> latent [{lf},{lh},{lw}] -> {tokens} tokens, {} blocks ===",
        cfg.name, cfg.num_layers
    );
    let t0 = Instant::now();
    let src = zeros(&manifest);
    // LATENT extent, not pixels: `WanDitDev::build` patchifies what the VAE
    // encoder would have produced, exactly as `pipeline::denoise` calls it.
    let dit = wan::WanDitDev::build(&cfg, &src, lf as u32, lh as u32, lw as u32, Some("gpu"), &[]);
    drop(src);
    eprintln!("built in {:.1} s ({} tensors, weight-free)", t0.elapsed().as_secs_f64(), manifest.len());
    profile_kinds(dit.gpu(), dit.steps(), &wan::block::KERNELS, reps, dit_flop(&cfg, tokens as f64));

    // The WHOLE forward, not just the recorded graph: host pre/post, the
    // uploads, the submit and the readback. `profile_kinds` above times the
    // graph alone, so the difference between these two numbers is every cost a
    // per-kind table structurally cannot show - and a seconds-per-forward
    // figure from a real generation is this number, not that one.
    let latent = vec![0.0f32; cfg.in_channels * lf * lh * lw];
    dit.set_context_embed(&vec![0.0f32; cfg.text_len * cfg.dim]);
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        std::hint::black_box(dit.forward(&latent, 500.0));
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("\nfull forward (host + upload + submit + readback): {best:.3} s");
}

// -------------------------------------------------------------- vae -------

fn bench_vae(reps: usize, frames: usize, w: usize, h: usize) {
    let cfg = WanVaeConfig::wan21();
    let lat_t = cfg.latent_frames(frames as u32).expect("frames must be 1 + 4k");
    let (lh, lw) = (h as u32 / 8, w as u32 / 8);
    let manifest = cfg.tensor_manifest();
    println!("\n=== Wan-VAE decode: latent [{}, {lat_t}, {lh}, {lw}] -> {frames} frames at {w}x{h} ===", cfg.z_dim);
    let t0 = Instant::now();
    let src = zeros(&manifest);
    let dec = WanVaeDecoder::build(&cfg, &src, lat_t, lh, lw, Some("gpu"));
    drop(src);
    eprintln!("built in {:.1} s ({} tensors, weight-free)", t0.elapsed().as_secs_f64(), manifest.len());
    profile_kinds(dec.gpu(), dec.steps(), &vae::blocks3d::KERNELS, reps, 0.0);
}

// ------------------------------------------------------------- host -------

/// The HOST stages a forward pays either side of the one device submit.
///
/// `WanDitDev::forward` is not just the block-stack submit: it patch-embeds the
/// latent, builds the timestep vectors, builds the RoPE tables, and runs the
/// head + unpatchify - all on the CPU, all in `wan::model`. Those do not appear
/// in the `dit` table (which times the recorded graph alone), so the gap between
/// this bench's total and a measured seconds-per-forward is exactly what this
/// mode accounts for. Weight-free like the rest: every stage's cost is set by
/// its shape, and none of them branches on a value.
fn bench_host(reps: usize, frames: usize, w: usize, h: usize) {
    let cfg = WanConfig::t2v_1_3b();
    let (lf, lh, lw) = cfg.latent_shape(frames, w, h).expect("frames must be 1 + 4k");
    let grid = wan::model::patch_grid(&cfg, lf as u32, lh as u32, lw as u32);
    let tokens = (grid.0 * grid.1 * grid.2) as usize;
    let src = zeros(&wan::import::dit_manifest(&cfg));
    let latent = vec![0.0f32; cfg.in_channels * lf * lh * lw];
    let x = vec![0.0f32; tokens * cfg.dim];
    let e = vec![0.0f32; cfg.dim];
    let ctx = vec![0.0f32; cfg.text_len * cfg.text_dim];
    println!("\n=== Wan host stages: {frames} frames at {w}x{h} -> {tokens} tokens ===");

    let best = |f: &dyn Fn()| {
        let mut b = f64::INFINITY;
        for _ in 0..reps.max(1) {
            let t0 = Instant::now();
            f();
            b = b.min(t0.elapsed().as_secs_f64());
        }
        b
    };
    let rows: Vec<(&str, f64)> = vec![
        ("embed_tokens (patchify+proj)", best(&|| {
            std::hint::black_box(wan::model::embed_tokens(&cfg, &src, &latent, lf as u32, lh as u32, lw as u32));
        })),
        ("postprocess (head+unpatch)", best(&|| {
            std::hint::black_box(wan::model::postprocess(&cfg, &src, &x, &e, grid));
        })),
        ("rope tables", best(&|| {
            std::hint::black_box(wan::rope::tables(&cfg, grid.0, grid.1, grid.2));
        })),
        ("text_embed (per prompt)", best(&|| {
            std::hint::black_box(wan::model::text_embed(&cfg, &src, &ctx, cfg.text_len));
        })),
        ("timestep_cond", best(&|| {
            std::hint::black_box(wan::model::timestep_cond(&cfg, &src, 500.0));
        })),
    ];
    println!("\n{:<32} {:>11}", "host stage", "ms");
    let mut per_forward = 0.0;
    for (n, t) in &rows {
        println!("{n:<32} {:>11.1}", t * 1e3);
        // text_embed is once a prompt (see `set_context_embed`), not a forward.
        if !n.starts_with("text_embed") {
            per_forward += t;
        }
    }
    println!("\nper-forward host total: {:.3} s (text_embed excluded: once a prompt)", per_forward);
}

// ------------------------------------------------------------ flash ------

/// A/B the bidirectional flash-attention kernels at ONE shape, for speed and
/// for agreement, in the SAME harness - a faster kernel that disagrees is not
/// a faster kernel, and kernel-vs-kernel timings printed without a numeric
/// cross-check beside them have shipped wrong answers here before.
///
/// Defaults are Wan 1.3B's self-attention at the measured end-to-end point:
/// T = 14040, 12 heads, head_dim 128. Every variant computes the same thing;
/// the first registered one is the reference the others are compared against,
/// and the achieved rate is graded against the device's own MEASURED fp32 roof
/// (`gpu_core::roof`), not a datasheet constant - the P40 clocks down hard
/// under sustained load, so a datasheet percentage flatters a long kernel.
///
/// `BR` differs per variant (the two-query-row kernel owns 128 rows, not 64),
/// so the grid is sized from each kernel's own BR rather than a shared
/// constant.
fn bench_flash(reps: usize, t: u32, nh: u32, hd: u32) {
    const VARIANTS: [(&str, &str, u32, u32); 4] = [
        ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR, 64, 64),
        ("flash_attn_bidir_split", kernels::FLASH_ATTN_BIDIR_SPLIT, 256, 64),
        ("flash_attn_bidir_reg", kernels::FLASH_ATTN_BIDIR_REG, 256, 64),
        ("flash_attn_bidir_reg2", kernels::FLASH_ATTN_BIDIR_REG2, 256, 128),
    ];
    let ks: Vec<(&str, &str)> = VARIANTS.iter().map(|(n, s, _, _)| (*n, *s)).collect();
    let gpu = Gpu::new_wgpu(&ks);
    let c = gpu.caps();
    eprintln!("max_workgroup_size {} workgroup_mem {} B", c.max_workgroup_size, c.workgroup_mem_bytes);
    let roof = gpu_core::roof::ensure(&gpu);
    let peak = roof.map(|r| r.gflops as f64).unwrap_or(P40_FP32_TFLOPS * 1e3);
    println!("\n=== bidirectional flash attention: T={t} heads={nh} head_dim={hd} ===");
    match roof {
        Some(r) => println!("measured roof: {:.0} GFLOP/s fp32, {:.0} GB/s DRAM", r.gflops, r.gbs),
        None => println!("measured roof unavailable - grading against the P40 datasheet peak"),
    }

    let d = nh * hd;
    let qkv = gpu.storage(t as u64 * 3 * d as u64);
    let src: Vec<f32> =
        (0..(t as usize * 3 * d as usize)).map(|i| ((i as f64 * 0.7).sin() * 0.5) as f32).collect();
    gpu.write_f32(&qkv, &src);
    drop(src);
    let prm = [1, nh, t, hd, 3 * d, 0, d, 2 * d, d];

    // The fused trio's FLOP: scores (2·T²·hd) + apply (2·T²·hd), per head.
    let gf = 4.0 * t as f64 * t as f64 * d as f64 / 1e9;
    let mut refout: Option<Vec<f32>> = None;
    println!("\n{:<26} {:>10} {:>12} {:>9} {:>12} {:>11}", "kernel", "ms", "GFLOP/s", "% roof", "cosine", "max_abs");
    for (idx, (name, _, ws, br)) in VARIANTS.iter().enumerate() {
        let o = gpu.storage(t as u64 * d as u64);
        let nwg = nh * t.div_ceil(*br);
        let st = vec![gpu.step(idx, &[&qkv, &o], &prm, nwg * ws)];
        let secs = time_steps(&gpu, &st, reps);
        let got = gpu.read(&o, (t * d) as usize);
        let (cos, mx) = match &refout {
            None => (1.0, 0.0f32),
            Some(r) => {
                let (mut dot, mut na, mut nb, mut mx) = (0f64, 0f64, 0f64, 0f32);
                for (&x, &y) in r.iter().zip(&got) {
                    dot += x as f64 * y as f64;
                    na += x as f64 * x as f64;
                    nb += y as f64 * y as f64;
                    mx = mx.max((x - y).abs());
                }
                (dot / (na.sqrt() * nb.sqrt()), mx)
            }
        };
        println!(
            "{name:<26} {:>10.1} {:>12.0} {:>8.1}% {:>12.9} {mx:>11.2e}",
            secs * 1e3,
            gf / secs,
            100.0 * gf / secs / peak,
            cos
        );
        if refout.is_none() {
            refout = Some(got);
        }
    }
}

// ------------------------------------------------------------ floor -------

/// Per-dispatch floor: `n` one-element matmuls in ONE submit. Isolates
/// pipeline-set + bind-group + dispatch cost from any real work, which is what
/// bounds the per-kind round-trip overhead the tables above carry.
fn bench_floor(n: usize) {
    let gpu = Gpu::new_wgpu(&wan::block::KERNELS);
    let k = wan::block::KERNELS.iter().position(|(name, _)| *name == "matmul").expect("matmul slot");
    let (x, w, o) = (gpu.storage(4), gpu.storage(4), gpu.storage(4));
    let steps: Vec<Step> = (0..n).map(|_| gpu.step(k, &[&x, &w, &o], &[1, 1, 1], 1)).collect();
    let s = time_steps(&gpu, &steps, 20);
    let one = time_steps(&gpu, &steps[..1], 20);
    println!("\n=== dispatch floor (1x1x1 matmul, one workgroup) ===");
    println!("  1 dispatch + queue round-trip: {:.3} ms", one * 1e3);
    println!("  {n} dispatches in one submit:   {:.3} ms total, {:.4} ms/dispatch", s * 1e3, s * 1e3 / n as f64);
}

fn main() {
    // The backend's readback deadlock guard defaults to 30 s, which is sized
    // for a token-at-a-time decoder. Every mode here submits a whole graph at
    // a real generation's shape - the DiT stack alone was 28 s a forward when
    // this bench was written - so the default turned `wan_bench dit` at its
    // OWN documented default shape into "device likely wedged" right after it
    // had printed the table. `brain wan t2v` already raises it for the same
    // reason; a profiler that cannot finish its own headline measurement is
    // worse than one that hangs, because the table above the panic looks fine.
    // Raise it only when the caller has expressed no opinion: the guard is
    // still what turns a genuinely wedged queue into an error instead of a
    // hang.
    if std::env::var_os("BRAIN_GPU_WAIT_S").is_none() {
        std::env::set_var("BRAIN_GPU_WAIT_S", "1200");
    }
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("dit");
    let arg = |i: usize, d: usize| args.get(i).and_then(|s| s.parse().ok()).unwrap_or(d);
    match mode {
        "dit" => bench_dit(arg(2, 3), arg(3, 33), arg(4, 832), arg(5, 480)),
        "vae" => bench_vae(arg(2, 3), arg(3, 33), arg(4, 832), arg(5, 480)),
        "host" => bench_host(arg(2, 3), arg(3, 33), arg(4, 832), arg(5, 480)),
        "floor" => bench_floor(arg(2, 500)),
        "flash" => bench_flash(arg(2, 3), arg(3, 14040) as u32, arg(4, 12) as u32, arg(5, 128) as u32),
        other => {
            eprintln!("unknown mode {other} (dit|vae|host|floor|flash)");
            std::process::exit(1);
        }
    }
}
