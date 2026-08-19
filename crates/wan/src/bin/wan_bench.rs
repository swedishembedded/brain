// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Wan2.1 profiler: where a generation's device time actually goes, per kernel
//! kind, against the device's own MEASURED roofline.
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
//! Method for `dit`/`vae` lives in `gpu_core::profile` - the shared per-kernel
//! profiler this and every other model bench call, rather than each carrying
//! a copy: device-timed where the backend supports it (one timed submit of
//! the WHOLE production pass, per-kernel totals read back from timestamp
//! queries - no group slicing, no per-group drain, no launch+fence floor
//! folded into a kernel's number), and the utilisation columns divide by the
//! device's own *measured* roofline (`gpu_core::roof`), never a hardcoded
//! peak.
//!
//! `host` and `train` are wall-clock, min-of-N host-side timings - there is no
//! device graph to profile there, so `Instant`-bracketing IS the honest method
//! for them. `flash` and `floor` are microbenchmarks against a hand-built
//! kernel list, already using `gpu_core::roof`/`gpu_core::profile::best_of`.
//!
//! Usage:
//!   wan_bench dit   [reps] [frames] [w] [h]   the DiT block stack, per kind
//!   wan_bench vae   [reps] [frames] [w] [h]   the VAE decode graph, per kind
//!   wan_bench host  [reps] [frames] [w] [h]   the HOST stages either side of it
//!   wan_bench train [reps] [t] [te]            the HOST trainer's block fwd/bwd
//!   wan_bench floor [n]                       per-dispatch floor (tiny kernel x n)
//!   wan_bench flash [reps] [T] [nh] [hd]      A/B every bidirectional flash kernel
//!
//! Defaults are the measured end-to-end point: 33 frames at 832x480, i.e.
//! 14,040 DiT tokens and a 9-latent-frame decode.
//!
//! `BRAIN_GPU_INDEX=0` selects a card. `dit`/`vae` also print a full
//! host-to-host round trip (upload, submit, readback, and for `dit` the host
//! patchify/head) next to the device-graph number, so the gap between them -
//! everything a per-kernel table structurally cannot show - is visible too.

use std::time::Instant;

use gpu_core::roof::Roofs;
use gpu_core::{Gpu, Step};
use wan::{WanConfig, WanVaeConfig, WanVaeDecoder};

/// Best-of-`reps` wall seconds for one submitted-and-drained step list.
/// Thin wrapper over the shared implementation - see `gpu_core::profile`'s
/// module doc for why every timed region here is `poll_wait`-bracketed.
fn best_of(gpu: &Gpu, steps: &[Step], reps: usize) -> f64 {
    gpu_core::profile::best_of(gpu, steps, reps)
}

/// One profile pass, printed against the device's MEASURED roofline. Same
/// shape as `vqgan_bench`'s `report()` and `sdxlunet`'s `unet_bench` - the
/// table, the grouping, the drain accounting and the coverage honesty all
/// live in `gpu_core::profile`, so every bench that calls this gets a fix in
/// one place rather than N private copies drifting on their own `PEAK_TFLOPS`
/// literal.
fn report(gpu: &Gpu, label: &str, steps: &[Step], reps: usize, roofs: Option<Roofs>) -> f64 {
    let p = gpu_core::profile::profile(gpu, label, steps, reps);
    p.print_top(roofs, 14);
    if let Some(r) = roofs {
        for (row, bound, pct) in p.defects(r, 5.0) {
            println!(
                "  DEFECT  {:<24} {:>5.1}% of its {} roof (floor {:.0}%) - {:.1}% of this pass",
                row.name,
                pct,
                bound.as_str(),
                bound.defect_pct(),
                100.0 * row.secs / p.summed_secs,
            );
        }
    }
    p.total_secs
}

fn print_roofline(gpu: &Gpu) -> Option<Roofs> {
    let roofs = gpu_core::roof::ensure(gpu);
    match roofs {
        Some(r) => println!(
            "measured roofline: {:.0} GFLOP/s, {:.1} GB/s DRAM, {:.1} GB/s cache, ridge {:.1} FLOP/byte",
            r.gflops,
            r.gbs,
            r.cache_gbs,
            r.ridge()
        ),
        None => println!("roofline unmeasured - utilisation columns print '-' rather than a guess"),
    }
    roofs
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

// -------------------------------------------------------------- dit -------

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

    let roofs = print_roofline(dit.gpu());
    let graph_secs = report(dit.gpu(), "DiT forward, one submit", dit.steps(), reps, roofs);

    // The WHOLE forward, not just the recorded graph: host pre/post, the
    // uploads, the submit and the readback. `report` above times the graph
    // alone, so the gap to this number is every cost a per-kind table
    // structurally cannot show - and a seconds-per-forward figure from a real
    // generation is THIS number, not that one.
    let latent = vec![0.0f32; cfg.in_channels * lf * lh * lw];
    dit.set_context_embed(&vec![0.0f32; cfg.text_len * cfg.dim]);
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        std::hint::black_box(dit.forward(&latent, 500.0));
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("\nfull forward (host + upload + submit + readback): {best:.3} s");
    let gap = (best - graph_secs).max(0.0);
    println!(
        "upload+readback+host overhead: {:.3} s ({:.1}% of the full forward) - full forward minus the \
         single-submit graph device time above. This bundles host patchify/head/upload/readback into one \
         number (indirect method: `wan_bench host` breaks the host-only part out separately; nothing here \
         instruments `write_f32`/`read` in isolation, which would need restructuring `WanDitDev::forward`).",
        gap,
        100.0 * gap / best,
    );
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

    let roofs = print_roofline(dec.gpu());
    let graph_secs = report(dec.gpu(), "VAE decode, one submit", dec.steps(), reps, roofs);

    let latent = vec![0.0f32; (cfg.z_dim as usize) * lat_t as usize * lh as usize * lw as usize];
    let mut best = f64::INFINITY;
    for _ in 0..reps.max(1) {
        let t = Instant::now();
        std::hint::black_box(dec.decode(&latent));
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("\nfull decode (upload + submit + readback): {best:.3} s");
    let gap = (best - graph_secs).max(0.0);
    println!("upload+readback overhead: {:.3} s ({:.1}% of the full decode)", gap, 100.0 * gap / best);
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

// ------------------------------------------------------------ train -------

/// The HOST trainer's per-block forward/backward, stage by stage.
///
/// Wan training is host-only (`crate::grad`'s `f32` instantiation of `Fp`
/// routes through `model::hostmath::matvec_par`, a CPU rayon path) - there is
/// no recorded device `&[Step]` graph for it, so `gpu_core::profile::profile`
/// does not apply here. Wall-clock min-of-N IS the honest method for host
/// code: every stage below is a plain function call with no device
/// involvement to bracket incorrectly.
///
/// The default `t` is a small-shape token count (9 frames at 256x256), not
/// the 14,040-token flagship shape `dit`/`vae`/`host` default to: `attn_fwd`/
/// `attn_bwd` in `grad.rs` are plain nested loops, not routed through
/// `matvec_par` like the linear layers, so their cost is O(t^2) on ONE
/// thread. At 14,040 tokens that loop alone is minutes per call; a caller who
/// wants that shape can still pass it explicitly, but the default here stays
/// in a range `reps` calls actually finish in.
fn bench_train(reps: usize, t: usize, te: usize) {
    use wan::grad::{block_backward, block_forward, layernorm_bwd, rmsnorm_bwd, rope_bwd, Dims};

    let cfg = WanConfig::t2v_1_3b();
    let d = Dims { t, te, dim: cfg.dim, nh: cfg.num_heads, ffn: cfg.ffn_dim, eps: cfg.eps as f64 };
    println!("\n=== Wan host trainer: t={t} te={te} dim={} nh={} ffn={} ===", d.dim, d.nh, d.ffn);

    let mut rng = data::rng::Rng::new(11);
    let mut u = || 0.1 * (2.0 * rng.next_f32() - 1.0);
    let lin = |out: usize, inn: usize, u: &mut dyn FnMut() -> f32| wan::grad::Lin::<f32> {
        w: (0..out * inn).map(|_| u()).collect(),
        b: (0..out).map(|_| u()).collect(),
    };
    let vecf = |n: usize, u: &mut dyn FnMut() -> f32| (0..n).map(|_| u()).collect::<Vec<f32>>();
    let w = wan::grad::BlockW::<f32> {
        modulation: vecf(6 * d.dim, &mut u),
        sq: lin(d.dim, d.dim, &mut u),
        sk: lin(d.dim, d.dim, &mut u),
        sv: lin(d.dim, d.dim, &mut u),
        so: lin(d.dim, d.dim, &mut u),
        snq: vecf(d.dim, &mut || 1.0),
        snk: vecf(d.dim, &mut || 1.0),
        cq: lin(d.dim, d.dim, &mut u),
        ck: lin(d.dim, d.dim, &mut u),
        cv: lin(d.dim, d.dim, &mut u),
        co: lin(d.dim, d.dim, &mut u),
        cnq: vecf(d.dim, &mut || 1.0),
        cnk: vecf(d.dim, &mut || 1.0),
        norm3_w: vecf(d.dim, &mut || 1.0),
        norm3_b: vecf(d.dim, &mut u),
        ff1: lin(d.ffn, d.dim, &mut u),
        ff2: lin(d.dim, d.ffn, &mut u),
    };
    let x = vecf(t * d.dim, &mut u);
    let e0 = vecf(6 * d.dim, &mut u);
    let ctx = vecf(te * d.dim, &mut u);
    let dout = vecf(t * d.dim, &mut u);
    let half = d.hd() / 2;
    let cos = vecf(t * half, &mut || rng.next_f32());
    let sin = vecf(t * half, &mut || rng.next_f32());

    let best = |f: &dyn Fn()| {
        let mut b = f64::INFINITY;
        for _ in 0..reps.max(1) {
            let t0 = Instant::now();
            f();
            b = b.min(t0.elapsed().as_secs_f64());
        }
        b
    };

    let t_fwd = best(&|| {
        std::hint::black_box(block_forward(d, &w, &x, &e0, &ctx, &cos, &sin));
    });
    let (_, cache) = block_forward(d, &w, &x, &e0, &ctx, &cos, &sin);
    let t_bwd = best(&|| {
        std::hint::black_box(block_backward(d, &w, &cache, &dout));
    });

    // A few named primitives inside the block, isolated, since `block_forward`
    // /`block_backward` above are the whole thing at once and do not say which
    // op dominates.
    let inv = vecf(t, &mut || 1.0);
    let t_ln_bwd = best(&|| {
        std::hint::black_box(layernorm_bwd(&x, &inv, t, d.dim, &dout));
    });
    let t_rms_bwd = best(&|| {
        let mut dw_dummy = vec![0.0f32; d.dim];
        std::hint::black_box(rmsnorm_bwd(&x, t, d.dim, &w.snq, &inv, &dout, &mut dw_dummy));
    });
    let t_rope_bwd = best(&|| {
        std::hint::black_box(rope_bwd(&dout, t, d.nh, d.hd(), &cos, &sin));
    });

    println!("\n{:<32} {:>11}", "stage (min-of-{reps}, host wall clock)", "ms");
    println!("{:<32} {:>11.1}", "block_forward (whole block)", t_fwd * 1e3);
    println!("{:<32} {:>11.1}", "block_backward (whole block)", t_bwd * 1e3);
    println!("{:<32} {:>11.1}", "  layernorm_bwd (one call)", t_ln_bwd * 1e3);
    println!("{:<32} {:>11.1}", "  rmsnorm_bwd (one call)", t_rms_bwd * 1e3);
    println!("{:<32} {:>11.1}", "  rope_bwd (one call)", t_rope_bwd * 1e3);
    println!(
        "\nbackward/forward = {:.2}x. This is CPU wall-clock (rayon row-parallel `matvec_par`), not a \
         device roofline - there is no GB/s to report against without a CPU `Roofs`, and \
         `gpu_core::roof::ensure` measures a `Gpu`, not the host, so it is not attempted here.",
        t_bwd / t_fwd
    );
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
    let peak = roof.map(|r| r.gflops as f64).unwrap_or(11.76e3);
    println!("\n=== bidirectional flash attention: T={t} heads={nh} head_dim={hd} ===");
    match roof {
        Some(r) => println!("measured roof: {:.0} GFLOP/s fp32, {:.0} GB/s DRAM", r.gflops, r.gbs),
        None => println!("measured roof unavailable - grading against a 11.76 TFLOP/s P40 fallback"),
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
        let secs = best_of(&gpu, &st, reps);
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
    let s = best_of(&gpu, &steps, 20);
    let one = best_of(&gpu, &steps[..1], 20);
    println!("\n=== dispatch floor (1x1x1 matmul, one workgroup) ===");
    println!("  1 dispatch + queue round-trip: {:.3} ms", one * 1e3);
    println!("  {n} dispatches in one submit:   {:.3} ms total, {:.4} ms/dispatch", s * 1e3, s * 1e3 / n as f64);
}

fn main() {
    // The backend's readback deadlock guard defaults to 30 s, which is sized
    // for a token-at-a-time decoder. Every mode here submits a whole graph at
    // a real generation's shape, far past that default. Raise it only when the
    // caller has expressed no opinion: the guard is still what turns a
    // genuinely wedged queue into an error instead of a hang.
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
        "train" => bench_train(arg(2, 5), arg(3, 768), arg(4, 512)),
        "floor" => bench_floor(arg(2, 500)),
        "flash" => bench_flash(arg(2, 3), arg(3, 14040) as u32, arg(4, 12) as u32, arg(5, 128) as u32),
        other => {
            eprintln!("unknown mode {other} (dit|vae|host|train|floor|flash)");
            std::process::exit(1);
        }
    }
}
