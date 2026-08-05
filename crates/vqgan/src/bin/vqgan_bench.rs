// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training-step profiler: where a FORWARD **and a BACKWARD** actually spend
//! their time, per kernel kind.
//!
//! Every profiler in this tree measured a forward. `docs/kernel-checklist.md`
//! §E asks for a per-kernel-kind table before anyone optimises, and for the
//! backward there was no way to get one — so the training datapath had never
//! been looked at, only assumed to look like the forward. It does not: the
//! reverse of a conv is TWO dispatches with different shapes (`conv2d_dx` reads
//! the weights transposed, `conv2d_dw` reduces over the batch), and the
//! per-channel reductions have no cooperative twin at all
//! (`vae::blocks::BWD_KERNELS`' documented §C.2 gap).
//!
//! The VQ autoencoder is the subject because its backward IS the shared block
//! backward set — `conv2d_dx/dw`, `gn_*`, `silu_bwd`, `upsample2_dx`, the
//! `attn_bwd_*_bidir` quartet — which `crates/vae`, `crates/unet` and
//! `crates/restore` all train through. A finding here is a finding for all of
//! them.
//!
//! Method (identical to `unet_bench`, deliberately):
//!   * every timed region is `poll_wait()`-bracketed — a bare `submit` only
//!     appends to `pending`, so an unbracketed loop times the HOST and reports
//!     it as device throughput (`docs/lessons.md` #6);
//!   * best-of-N, not mean: the minimum is the least contaminated sample;
//!   * groups are CONTIGUOUS runs of one kernel in submit order, so the sum of
//!     the parts is comparable to the whole, and both are printed.
//!
//! Usage:
//!   vqgan_bench [size] [reps]        # default 256, 3

use std::collections::HashMap;
use std::time::Instant;

use gpu_core::{f, Gpu, Step};
use vqgan::config::VqganConfig;
use vqgan::train::VqganTrainer;

/// Tesla P40 fp32 peak, printed as a denominator so a rate above the physical
/// roof is visibly impossible rather than quietly believed.
const PEAK_TFLOPS: f64 = 11.76;

fn best_of(gpu: &Gpu, steps: &[Step], reps: usize) -> f64 {
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

/// Contiguous runs of one kernel, in graph order.
fn groups(steps: &[Step]) -> Vec<(usize, usize, usize)> {
    let mut g: Vec<(usize, usize, usize)> = Vec::new();
    for (i, s) in steps.iter().enumerate() {
        let k = s.meta().map(|m| m.kernel).unwrap_or(usize::MAX);
        match g.last_mut() {
            Some((gk, _, len)) if *gk == k => *len += 1,
            _ => g.push((k, i, 1)),
        }
    }
    g
}

fn report(gpu: &Gpu, label: &str, steps: &[Step], reps: usize) -> f64 {
    let total = best_of(gpu, steps, reps);
    let gs = groups(steps);
    // Per kind: time, call count, and the ANALYTICAL FLOP/byte volume from
    // `gpu_core::cost` — the repo's own accounting, so the rate below is
    // comparable with every other number in docs/performance/.
    let mut per: HashMap<usize, (f64, usize, u64, u64, bool)> = HashMap::new();
    for (k, start, len) in &gs {
        let t = best_of(gpu, &steps[*start..*start + *len], reps);
        let name = gpu.kernel_name(*k).unwrap_or("?");
        let (mut fl, mut by, mut covered) = (0u64, 0u64, true);
        for st in &steps[*start..*start + *len] {
            let m = st.meta();
            let params = m.as_ref().and_then(|m| m.params.as_deref());
            match gpu_core::cost::kernel_cost(name, params, m.as_ref().map(|m| m.threads).unwrap_or(0)) {
                Some(c) => {
                    fl += c.flops;
                    by += c.bytes;
                }
                None => covered = false,
            }
        }
        let e = per.entry(*k).or_insert((0.0, 0, 0, 0, true));
        e.0 += t;
        e.1 += *len;
        e.2 += fl;
        e.3 += by;
        e.4 &= covered;
    }
    let mut rows: Vec<(String, f64, usize, u64, u64, bool)> = per
        .into_iter()
        .map(|(k, (t, n, fl, by, cov))| (gpu.kernel_name(k).unwrap_or("?").to_string(), t, n, fl, by, cov))
        .collect();
    rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let summed: f64 = rows.iter().map(|r| r.1).sum();
    let tot_flops: u64 = rows.iter().map(|r| r.3).sum();

    println!("\n=== {label}: {} dispatches, {:.2} ms ===", steps.len(), total * 1e3);
    println!(
        "{:<24} {:>9} {:>6} {:>6} {:>9} {:>11} {:>7} {:>9}",
        "kernel", "ms", "n", "%", "ms/call", "GFLOP/s", "%peak", "GB/s"
    );
    println!("{}", "-".repeat(88));
    for (name, t, n, fl, by, cov) in rows.iter().take(14) {
        // A compute rate is only meaningful where `cost` has a formula. An
        // UNCOVERED kernel prints "-" rather than a zero that reads as slow.
        let gfs = if *cov && *fl > 0 { format!("{:.1}", *fl as f64 / t / 1e9) } else { "-".into() };
        let pk = if *cov && *fl > 0 {
            format!("{:.1}%", 100.0 * (*fl as f64 / t / 1e9) / (PEAK_TFLOPS * 1e3))
        } else {
            "-".into()
        };
        let gbs = if *cov && *by > 0 { format!("{:.1}", *by as f64 / t / 1e9) } else { "-".into() };
        println!(
            "{:<24} {:>9.2} {:>6} {:>5.1}% {:>9.3} {:>11} {:>7} {:>9}",
            name,
            t * 1e3,
            n,
            100.0 * t / summed,
            t * 1e3 / *n as f64,
            gfs,
            pk,
            gbs
        );
    }
    if tot_flops > 0 {
        println!(
            "{:<24} {:>9.2} {:>6} {:>6} {:>9} {:>11.1} {:>6.1}%",
            "WHOLE PASS",
            total * 1e3,
            steps.len(),
            "",
            "",
            tot_flops as f64 / total / 1e9,
            100.0 * (tot_flops as f64 / total / 1e9) / (PEAK_TFLOPS * 1e3)
        );
    }
    println!("{}", "-".repeat(64));
    println!("{:<26} {:>9.2}  (whole {:.2} ms, {} drains)", "sum of groups", summed * 1e3, total * 1e3, gs.len());
    total
}

/// Shape-correct random weights. The profile depends only on the graph, so this
/// stands in for the 377 MB checkpoint and makes the bench runnable anywhere —
/// the same reason `unet_bench` is weight-free.
fn init_weights(cfg: &VqganConfig, seed: u64) -> vae::blocks::Tensors {
    let mut rng = data::rng::Rng::new(seed);
    let mut t = vae::blocks::Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        let u = |r: &mut data::rng::Rng| 2.0 * r.next_f32() - 1.0;
        let d: Vec<f32> = match shape.len() {
            1 if name.ends_with(".weight") => (0..n).map(|_| 1.0 + 0.1 * u(&mut rng)).collect(),
            1 => (0..n).map(|_| 0.1 * u(&mut rng)).collect(),
            2 => (0..n).map(|_| 0.6 * u(&mut rng)).collect(),
            _ => {
                let s = 1.0 / ((n / shape[0]) as f32).sqrt();
                (0..n).map(|_| s * u(&mut rng)).collect()
            }
        };
        t.insert(name, (shape, d));
    }
    t
}

/// A/B the two GroupNorm statistic reductions at one shape, for CORRECTNESS and
/// speed.
///
/// `vae::blocks` selects `gn_stats` (serial, one lane per group) or
/// `gn_stats_wg` (workgroup-cooperative) on `DeviceCaps::workgroup_reductions`.
/// `crates/wm-diamond` independently built a THIRD path — `gn_part` +
/// `gn_stats2`, a barrier-free two-stage reduction — after measuring the serial
/// one at 77% of its frame time. Nobody has compared them, and the answer
/// decides whether wm-diamond can drop its private Builder (task #25) or
/// whether the shared one has to learn the two-stage path first.
///
/// Barrier-free matters beyond speed: `backend-cpu` reports
/// `workgroup_reductions: false`, so on the CPU JIT the cooperative kernel is
/// not selectable at all and the shared builder falls back to the serial one.
fn gn_ab(reps: usize) {
    const P: u32 = 64; // partials per group, wm-diamond's GN_P
    let kernels: &[(&str, &str)] = &[
        ("gn_stats", kernels::GN_STATS),
        ("gn_stats_wg", kernels::GN_STATS_WG),
        ("gn_part", kernels::GN_PART),
        ("gn_stats2", kernels::GN_STATS2),
    ];
    let gpu = Gpu::new(kernels);
    let (k_ser, k_wg, k_part, k_st2) = (0usize, 1, 2, 3);
    // `gn_stats_wg` uses workgroupBarrier; backend-cpu reports
    // workgroup_reductions: false and its JIT cannot compile it. Dispatching it
    // anyway panics — which is exactly why `vae::blocks` branches on the
    // QUERIED capability rather than assuming (docs/lessons.md #5).
    let coop = gpu.caps().workgroup_reductions;
    if !coop {
        println!("(no workgroup reductions on this device — the cooperative kernel is not selectable)");
    }
    println!("{:<44} {:>10} {:>10} {:>10}  {:>12}", "shape [C,H,W] groups", "serial", "coop", "2-stage", "max|delta|");

    // A small shape first, checked against a HOST reference, so "they disagree"
    // becomes "this one is wrong" instead of a three-way stare.
    {
        let (c, h, w, g) = (8u32, 4u32, 4u32, 2u32);
        let n = (c * h * w) as usize;
        let mut rng = data::rng::Rng::new(3);
        let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let xb = gpu.storage(n as u64);
        gpu.write_f32(&xb, &x);
        let eps = f(1e-6);
        let per = n / g as usize;
        let mut host = vec![0.0f32; 2 * g as usize];
        for k in 0..g as usize {
            let sl = &x[k * per..(k + 1) * per];
            let m = sl.iter().sum::<f32>() / per as f32;
            let v = sl.iter().map(|q| (q - m) * (q - m)).sum::<f32>() / per as f32;
            host[2 * k] = m;
            host[2 * k + 1] = 1.0 / (v + 1e-6).sqrt();
        }
        let s_ser = gpu.storage(2 * g as u64);
        gpu.submit(&[], &[gpu.step(k_ser, &[&xb, &s_ser], &[1, c, h, w, g, eps], g)]);
        let s_wg = gpu.storage(2 * g as u64);
        if coop {
            gpu.submit(&[], &[gpu.step(k_wg, &[&xb, &s_wg], &[1, c, h, w, g, eps], g * 256)]);
        }
        let part = gpu.storage(2 * g as u64 * P as u64);
        let s_2 = gpu.storage(2 * g as u64);
        gpu.submit(&[], &[
            gpu.step(k_part, &[&xb, &part], &[1, c, h, w, g, P], g * P),
            gpu.step(k_st2, &[&part, &s_2], &[1, c, h, w, g, P, eps], g),
        ]);
        gpu.poll_wait();
        let m = |b: &gpu_core::DeviceBuffer| gpu.read(b, 2 * g as usize);
        let (a, b, cc) = (m(&s_ser), m(&s_wg), m(&s_2));
        let dev = |v: &Vec<f32>| v.iter().zip(&host).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        let _ = &b;
        println!("  host check [{c},{h},{w}] g={g}:  serial {:.3e}   coop {:.3e}   2-stage {:.3e}",
                 dev(&a), dev(&b), dev(&cc));
        println!("    host    {host:?}");
        println!("    serial  {a:?}");
        println!("    coop    {b:?}");
        println!("    2-stage {cc:?}");
    }

    // The decoder's real shapes, widest-to-narrowest.
    for &(c, h, w, g) in &[(512u32, 64u32, 64u32, 32u32), (512, 128, 128, 32), (256, 256, 256, 32), (128, 512, 512, 32)] {
        let n = (c * h * w) as u64;
        let mut rng = data::rng::Rng::new(9);
        let x: Vec<f32> = (0..n as usize).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let xb = gpu.storage(n);
        gpu.write_f32(&xb, &x);
        let eps = f(1e-6);

        let s_ser = gpu.storage(2 * g as u64);
        let t_ser = best_of(&gpu, &[gpu.step(k_ser, &[&xb, &s_ser], &[1, c, h, w, g, eps], g)], reps);

        let s_wg = gpu.storage(2 * g as u64);
        let t_wg = if coop {
            best_of(&gpu, &[gpu.step(k_wg, &[&xb, &s_wg], &[1, c, h, w, g, eps], g * 256)], reps)
        } else {
            f64::NAN
        };

        let part = gpu.storage(2 * g as u64 * P as u64);
        let s_2 = gpu.storage(2 * g as u64);
        let two = vec![
            gpu.step(k_part, &[&xb, &part], &[1, c, h, w, g, P], g * P),
            gpu.step(k_st2, &[&part, &s_2], &[1, c, h, w, g, P, eps], g),
        ];
        let t_2 = best_of(&gpu, &two, reps);

        // Correctness: all three must agree. A fast wrong reduction is not fast.
        let (a, b, cc) = (gpu.read(&s_ser, 2 * g as usize), gpu.read(&s_wg, 2 * g as usize), gpu.read(&s_2, 2 * g as usize));
        let d = if coop {
            a.iter().zip(&b).zip(&cc).map(|((x, y), z)| (x - y).abs().max((x - z).abs())).fold(0.0f32, f32::max)
        } else {
            a.iter().zip(&cc).map(|(x, z)| (x - z).abs()).fold(0.0f32, f32::max)
        };
        println!(
            "[{c:4},{h:4},{w:4}] g={g:<3} {:>28.3} {:>10.3} {:>10.3}  {:>12.3e}",
            t_ser * 1e3,
            t_wg * 1e3,
            t_2 * 1e3,
            d
        );
    }
}

/// A/B the conv INPUT gradient: the direct `conv2d_dx` against the GEMM
/// lowering (`nchw_nlc` -> `matmul_dx_reg` -> `col2im`).
///
/// The backward profile says `conv2d_dx` is 41% of a VQGAN training step at
/// 12.9 ms/call, against the forward conv's 4.8. The direct kernel reduces over
/// `Cout*K*K` per input pixel; the lowering moves the `Cout` axis into a
/// register-tiled GEMM and leaves `col2im` summing only `K*K`.
fn convbwd_ab(reps: usize) {
    let kernels: &[(&str, &str)] = &[
        ("conv2d_dx", kernels::CONV2D_DX),
        ("matmul_dx_reg", kernels::MATMUL_DX_REG),
        ("col2im", kernels::COL2IM),
        ("nchw_nlc", kernels::NCHW_NLC),
    ];
    let gpu = Gpu::new(kernels);
    let (k_dx, k_mm, k_c2i, k_t) = (0usize, 1, 2, 3);
    println!("{:<34} {:>10} {:>10} {:>9}", "conv (cin,cout,HxW,k,s)", "direct", "lowered", "speedup");

    // Sweep Cout at a fixed spatial size to FIND the crossover, rather than
    // guessing it — the lowering's cost is dominated by materialising
    // dcol[HW, Cin*K*K], which does not shrink with Cout, while the direct
    // kernel's cost is linear in Cout. So there must be a Cout below which
    // direct wins, and the whole point is to know where.
    for &(cin, cout, h, w, k, st) in &[
        (128u32, 3u32, 256u32, 256u32, 3u32, 1u32),
        (128, 8, 256, 256, 3, 1),
        (128, 16, 256, 256, 3, 1),
        (128, 32, 256, 256, 3, 1),
        (128, 64, 256, 256, 3, 1),
        (128, 128, 256, 256, 3, 1),
    ] {
        let pad = k / 2;
        let (ho, wo) = ((h + 2 * pad - k) / st + 1, (w + 2 * pad - k) / st + 1);
        let hw = (ho * wo) as u64;
        let cinkk = (cin * k * k) as u64;
        let dy = gpu.storage((cout as u64) * hw);
        let wt = gpu.storage((cout as u64) * cinkk);
        let dx = gpu.storage((cin * h * w) as u64);
        let t_direct = best_of(&gpu, &[gpu.step(k_dx, &[&dy, &wt, &dx], &[1, cin, h, w, cout, k, st, pad, ho, wo], cin * h * w)], reps);
        let dy_nlc = gpu.storage((cout as u64) * hw);
        let dcol = gpu.storage(hw * cinkk);
        let t_low = best_of(&gpu, &[
            gpu.step(k_t, &[&dy, &dy_nlc], &[cout * hw as u32, cout, hw as u32], cout * hw as u32),
            gpu.step(k_mm, &[&dy_nlc, &wt, &dcol], &[hw as u32, cinkk as u32, cout, 0],
                     (hw as u32).div_ceil(128) * (cinkk as u32).div_ceil(128) * 256),
            gpu.step(k_c2i, &[&dcol, &dx], &[1, cin, h, w, k, st, pad, ho, wo, cinkk as u32], cin * h * w),
        ], reps);
        println!("  sweep cout={cout:4}          {:>10.3} {:>10.3} {:>8.2}x", t_direct * 1e3, t_low * 1e3, t_direct / t_low);
    }

    // The VQGAN decoder's real convs, widest first.
    for &(cin, cout, h, w, k, st) in &[
        (512u32, 512u32, 64u32, 64u32, 3u32, 1u32),
        (512, 512, 128, 128, 3, 1),
        (256, 256, 256, 256, 3, 1),
        (128, 128, 512, 512, 3, 1),
        (128, 3, 512, 512, 3, 1),
    ] {
        let pad = k / 2;
        let (ho, wo) = ((h + 2 * pad - k) / st + 1, (w + 2 * pad - k) / st + 1);
        let hw = (ho * wo) as u64;
        let cinkk = (cin * k * k) as u64;
        let dy = gpu.storage((cout as u64) * hw);
        let wt = gpu.storage((cout as u64) * cinkk);
        let dx = gpu.storage((cin * h * w) as u64);

        let direct = vec![gpu.step(
            k_dx,
            &[&dy, &wt, &dx],
            &[1, cin, h, w, cout, k, st, pad, ho, wo],
            cin * h * w,
        )];
        let t_direct = best_of(&gpu, &direct, reps);

        let dy_nlc = gpu.storage((cout as u64) * hw);
        let dcol = gpu.storage(hw * cinkk);
        let lowered = vec![
            gpu.step(k_t, &[&dy, &dy_nlc], &[cout * hw as u32, cout, hw as u32], cout * hw as u32),
            gpu.step(
                k_mm,
                &[&dy_nlc, &wt, &dcol],
                &[hw as u32, cinkk as u32, cout, 0],
                (hw as u32).div_ceil(128) * (cinkk as u32).div_ceil(128) * 256,
            ),
            gpu.step(k_c2i, &[&dcol, &dx], &[1, cin, h, w, k, st, pad, ho, wo, cinkk as u32], cin * h * w),
        ];
        let t_low = best_of(&gpu, &lowered, reps);

        println!(
            "({cin:4},{cout:4}) {h:4}x{w:<4} k{k} s{st}       {:>10.3} {:>10.3} {:>8.2}x",
            t_direct * 1e3,
            t_low * 1e3,
            t_direct / t_low
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(|s| s == "convbwd").unwrap_or(false) {
        convbwd_ab(a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3));
        return;
    }
    if a.get(1).map(|s| s == "gn").unwrap_or(false) {
        gn_ab(a.get(2).and_then(|s| s.parse().ok()).unwrap_or(5));
        return;
    }
    let size: u32 = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(256);
    let reps: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    let cfg = VqganConfig::codeformer();
    let scale = cfg.downscale();
    assert!(size.is_multiple_of(scale), "size {size} must be a multiple of the {scale}x downscale");

    // Weight-free: the cost depends only on shape, so random weights profile the
    // same graph as the 377 MB checkpoint and the run takes seconds.
    let gpu = Gpu::new(vqgan::TRAIN_PIPELINES);
    let tensors = init_weights(&cfg, 7);
    eprintln!("vqgan_bench: {size}x{size}, latent {}x{}, {reps} reps", size / scale, size / scale);
    let t0 = Instant::now();
    let tr = VqganTrainer::new(cfg, &tensors, size, size, gpu.share());
    eprintln!("built in {:.1}s\n", t0.elapsed().as_secs_f32());

    let img: Vec<f32> = (0..(3 * size * size) as usize).map(|i| (i % 251) as f32 / 251.0).collect();
    tr.set_batch(&img, &img);
    tr.latch_assignment();

    let fwd = report(&gpu, "FORWARD", tr.fwd_steps(), reps);
    let bwd = report(&gpu, "BACKWARD", tr.bwd_steps(), reps);

    println!("\nforward {:.2} ms + backward {:.2} ms = {:.2} ms/step", fwd * 1e3, bwd * 1e3, (fwd + bwd) * 1e3);
    println!("backward/forward = {:.2}x   ({} clears before each backward)", bwd / fwd, tr.bwd_clears().len());
    println!("P40 fp32 peak {PEAK_TFLOPS} TFLOP/s; a rate above it means the host was timed, not the device.");
}
