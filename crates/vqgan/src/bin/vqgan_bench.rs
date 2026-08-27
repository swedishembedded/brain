// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Training-step profiler: where a FORWARD **and a BACKWARD** actually spend
//! their time, per kernel kind.
//!
//! Every profiler in this tree measured a forward. A per-kernel-kind table
//! is wanted before anyone optimises, and for the
//! backward there was no way to get one — so the training datapath had never
//! been looked at, only assumed to look like the forward. It does not: the
//! reverse of a conv is TWO dispatches with different shapes (`conv2d_dx` reads
//! the weights transposed, `conv2d_dw` reduces over the batch), and the
//! per-channel reductions have no cooperative twin at all
//! (`vae::blocks::BWD_KERNELS`' documented §C.2 gap).
//!
//! The VQ autoencoder is the subject because its backward IS the shared block
//! backward set — `conv2d_dx/dw`, `gn_*`, `silu_bwd`, `upsample2_dx`, the
//! `attn_bwd_*_bidir` quartet - which `crates/vae`, `crates/sdxlunet` and
//! `crates/codeformer` all train through. A finding here is a finding for all of
//! them.
//!
//! Method lives in `gpu_core::profile` — the shared §F.1 profiler this and
//! every other model bench now call, rather than each carrying a copy:
//!   * every timed region is `poll_wait()`-bracketed — a bare `submit` only
//!     appends to `pending`, so an unbracketed loop times the HOST and reports
//!     it as device throughput;
//!   * best-of-N, not mean: the minimum is the least contaminated sample;
//!   * groups are CONTIGUOUS runs of one kernel in submit order, so the sum of
//!     the parts is comparable to the whole, and both are printed;
//!   * utilisation divides by the device's **measured** roofline
//!     (`gpu_core::roof`), not by a hardcoded P40 peak.
//!
//! Usage:
//!   vqgan_bench [size] [reps]        # default 256, 3

use std::time::Instant;

use gpu_core::roof::Roofs;
use gpu_core::{f, Gpu, Step};
use vqgan::config::VqganConfig;
use vqgan::train::VqganTrainer;

fn best_of(gpu: &Gpu, steps: &[Step], reps: usize) -> f64 {
    gpu_core::profile::best_of(gpu, steps, reps)
}

/// One §F.1 pass profile, printed against the device's MEASURED roofline.
///
/// The table, the grouping, the drain accounting and the coverage honesty all
/// live in `gpu_core::profile` — four benches used to carry a copy of them, and
/// each copy divided by its own `PEAK_TFLOPS = 11.76` literal.
fn report(gpu: &Gpu, label: &str, steps: &[Step], reps: usize, roofs: Option<Roofs>) -> f64 {
    let p = gpu_core::profile::profile(gpu, label, steps, reps);
    p.print_top(roofs, 14);
    if let Some(r) = roofs {
        for (row, bound, pct) in p.defects(r, 5.0) {
            println!(
                "  DEFECT  {:<24} {:>5.1}% of its {} roof (floor {:.0}%) — {:.1}% of this pass",
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
/// `crates/diamond` independently built a THIRD path - `gn_part` +
/// `gn_stats2`, a barrier-free two-stage reduction - after profiling put the
/// serial one at the bulk of its frame time. Nobody has compared them, and the answer
/// decides whether diamond can drop its private Builder (task #25) or
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
    // QUERIED capability rather than assuming.
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
/// The backward profile says `conv2d_dx` is a large share of a VQGAN training
/// step, several times the per-call cost of the forward conv it mirrors. The
/// direct kernel reduces over
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

/// `vqgan_bench dwtn [reps]` — A/B `matmul_dw_reg` against `matmul_dw_reg_tn`.
///
/// The dw GEMM is a TN contraction: `dW[n,k] += sum_m dY[m,n]*X[m,k]` sums over
/// the ROW index of both operands, so consecutive lanes read a row apart. The
/// profile shows it at several times the per-call cost of `matmul_dx_reg` at
/// IDENTICAL m/k/n and a fraction of its bandwidth, with the same FLOP count: a
/// coalescing
/// gap, not an arithmetic one. `_tn` takes dY already transposed (which for conv
/// backward is just the raw NCHW dY) so the A-side load coalesces.
///
/// Correctness first, against a HOST oracle: a kernel that disagrees is not a
/// faster kernel.
fn dwtn_ab(reps: usize) {
    let kernels: &[(&str, &str)] =
        &[
        ("matmul_dw_reg", kernels::MATMUL_DW_REG),
        ("matmul_dw_reg_tn", kernels::MATMUL_DW_REG_TN),
        ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
        ("dw_splitk_reduce", kernels::DW_SPLITK_REDUCE),
    ];
    let gpu = Gpu::new(kernels);
    let (k_dw, k_tn, k_sk, k_rd) = (0usize, 1usize, 2usize, 3usize);

    // ---- correctness, small enough to check on the host --------------------
    {
        let (m, k, n) = (37u32, 23u32, 19u32);
        let mut rng = data::rng::Rng::new(7);
        let dy: Vec<f32> = (0..(m * n) as usize).map(|_| rng.next_f32() - 0.5).collect();
        let x: Vec<f32> = (0..(m * k) as usize).map(|_| rng.next_f32() - 0.5).collect();
        // dY^T, the layout `_tn` expects.
        let mut dyt = vec![0.0f32; (m * n) as usize];
        for mm in 0..m as usize {
            for nn in 0..n as usize {
                dyt[nn * m as usize + mm] = dy[mm * n as usize + nn];
            }
        }
        let mut want = vec![0.0f64; (n * k) as usize];
        for nn in 0..n as usize {
            for kk in 0..k as usize {
                let mut acc = 0.0f64;
                for mm in 0..m as usize {
                    acc += dy[mm * n as usize + nn] as f64 * x[mm * k as usize + kk] as f64;
                }
                want[nn * k as usize + kk] = acc;
            }
        }
        for (name, ki, a_host) in [("matmul_dw_reg", k_dw, &dy), ("matmul_dw_reg_tn", k_tn, &dyt)] {
            let ab = gpu.storage((m * n) as u64);
            let xb = gpu.storage((m * k) as u64);
            let ob = gpu.storage((n * k) as u64);
            gpu.write_f32(&ab, a_host);
            gpu.write_f32(&xb, &x);
            gpu.write_f32(&ob, &vec![0.0f32; (n * k) as usize]); // it ACCUMULATES
            gpu.submit(&[], &[gpu.step(ki, &[&ab, &xb, &ob], &[m, k, n], n.div_ceil(128) * k.div_ceil(128) * 256)]);
            gpu.poll_wait();
            let got = gpu.read(&ob, (n * k) as usize);
            let err = got.iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0f64, f64::max);
            println!("  oracle [{m},{k},{n}]  {name:<18} max|delta| {err:.3e}");
            assert!(err < 1e-3, "{name} diverges from the f64 host oracle: max|delta| {err:.3e}");
        }
        // split-K, at several slice counts: the reduction must not change what
        // the GEMM means, at any split.
        for slices in [1u32, 3, 8, 64] {
            let ab = gpu.storage((m * n) as u64);
            let xb = gpu.storage((m * k) as u64);
            let ob = gpu.storage((n * k) as u64);
            let pb = gpu.storage((slices * n * k) as u64);
            gpu.write_f32(&ab, &dy);
            gpu.write_f32(&xb, &x);
            gpu.write_f32(&ob, &vec![0.0f32; (n * k) as usize]);
            let tiles = n.div_ceil(128) * k.div_ceil(128);
            gpu.submit(&[], &[
                gpu.step(k_sk, &[&ab, &xb, &pb], &[m, k, n, slices], slices * tiles * 256),
                gpu.step(k_rd, &[&pb, &ob], &[n * k, slices], (n * k).div_ceil(64) * 64),
            ]);
            gpu.poll_wait();
            let got = gpu.read(&ob, (n * k) as usize);
            let err = got.iter().zip(&want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0f64, f64::max);
            println!("  oracle [{m},{k},{n}]  splitk s={slices:<3}        max|delta| {err:.3e}");
            // A real gate, not a printout: a split that changed what the GEMM
            // means must ABORT the bench, or the speed table below would be
            // benchmarking a wrong kernel (routed audit item F5).
            assert!(err < 1e-3, "split-K (s={slices}) diverges from the f64 host oracle: max|delta| {err:.3e}");
        }
    }

    // ---- speed, at the shapes the VQGAN backward actually dispatches -------
    println!();
    println!("{:<30} {:>10} {:>10} {:>9} {:>10} {:>10}", "conv (cin,cout,HxW)", "dw", "dw_tn", "speedup", "GB/s dw", "GB/s tn");
    for &(cin, cout, h, w, k) in &[
        (512u32, 512u32, 64u32, 64u32, 3u32),
        (512, 512, 128, 128, 3),
        (256, 256, 256, 256, 3),
        (128, 128, 512, 512, 3),
    ] {
        let (hw, cinkk) = ((h * w) as u64, (cin * k * k) as u64);
        let (m, kk, n) = (hw as u32, cinkk as u32, cout);
        let dy = gpu.storage((cout as u64) * hw); // [m,n] and [n,m] are the same size
        let col = gpu.storage(hw * cinkk);
        let dw = gpu.storage((cout as u64) * cinkk);
        let threads = n.div_ceil(128) * kk.div_ceil(128) * 256;
        let t_a = best_of(&gpu, &[gpu.step(k_dw, &[&dy, &col, &dw], &[m, kk, n], threads)], reps);
        let t_b = best_of(&gpu, &[gpu.step(k_tn, &[&dy, &col, &dw], &[m, kk, n], threads)], reps);
        let tiles = n.div_ceil(128) * kk.div_ceil(128);
        // bytes: dY + col read, dW read-modify-write.
        let bytes = 4.0 * ((m as f64 * n as f64) + (m as f64 * kk as f64) + 2.0 * (n as f64 * kk as f64));
        println!(
            "({cin:4},{cout:4}) {h:4}x{w:<4}          {:>10.3} {:>10.3} {:>8.2}x {:>10.1} {:>10.1}",
            t_a * 1e3, t_b * 1e3, t_a / t_b, bytes / t_a / 1e9, bytes / t_b / 1e9
        );
        // Sweep the slice count rather than guess it: more slices buys
        // occupancy but costs a wider partial buffer and a longer reduction.
        let mut best = (1u32, f64::INFINITY);
        for slices in [1u32, 2, 4, 8, 16, 32, 64] {
            if (slices as u64) * (n as u64) * (kk as u64) > 400_000_000 {
                continue;
            }
            let pb = gpu.storage((slices as u64) * (n as u64) * (kk as u64));
            let t = best_of(&gpu, &[
                gpu.step(k_sk, &[&dy, &col, &pb], &[m, kk, n, slices], slices * tiles * 256),
                gpu.step(k_rd, &[&pb, &dw], &[n * kk, slices], (n * kk).div_ceil(64) * 64),
            ], reps);
            print!("  s={slices}:{:.1}", t * 1e3);
            if t < best.1 {
                best = (slices, t);
            }
        }
        println!("   -> best s={} at {:.3} ms ({:.2}x, {} wgs)", best.0, best.1 * 1e3, t_a / best.1, best.0 * tiles);
    }
}

/// `vqgan_bench convfwd [reps]` — re-derive `GEMM_CONV_MIN_COUT`.
///
/// The forward threshold (128) was measured for the ORIGINAL kernel pair and
/// never re-derived after `matmul_reg3` replaced `matmul_reg2` in the lowering.
/// The backward's equivalent re-derivation moved 128 -> 32 and was worth a
/// multiple at the shapes in between, and every kernel
/// pair gets its own swept threshold — so this must be measured, not inherited.
///
/// direct  = conv_bias_reg
/// lowered = im2col_at + matmul_reg3 + nlc_bias_nchw
fn convfwd_ab(reps: usize) {
    let kernels: &[(&str, &str)] = &[
        ("conv_bias_reg", kernels::CONV_BIAS_REG),
        ("im2col_at", kernels::IM2COL_AT),
        ("matmul_reg3", kernels::MATMUL_REG3),
        ("nlc_bias_nchw", kernels::NLC_BIAS_NCHW),
    ];
    let gpu = Gpu::new(kernels);
    let (k_direct, k_im2col, k_mm, k_epi) = (0usize, 1, 2, 3);
    println!("{:<32} {:>10} {:>10} {:>9}", "conv (cin,cout,HxW)", "direct", "lowered", "speedup");
    for &(cin, cout, h, w) in &[
        (128u32, 8u32, 256u32, 256u32),
        (128, 16, 256, 256),
        (128, 32, 256, 256),
        (128, 64, 256, 256),
        (128, 96, 256, 256),
        (128, 128, 256, 256),
        (128, 256, 256, 256),
        (256, 32, 128, 128),
        (256, 64, 128, 128),
        (256, 128, 128, 128),
        (512, 64, 64, 64),
        (512, 128, 64, 64),
        (512, 512, 64, 64),
    ] {
        let (k, st, pad) = (3u32, 1u32, 1u32);
        let (ho, wo) = ((h + 2 * pad - k) / st + 1, (w + 2 * pad - k) / st + 1);
        let (hw, cinkk) = ((ho * wo) as u64, (cin * k * k) as u64);
        let x = gpu.storage((cin * h * w) as u64);
        let wt = gpu.storage((cout as u64) * cinkk);
        let bias = gpu.storage(cout as u64);
        let y = gpu.storage((cout as u64) * hw);

        let t_direct = best_of(&gpu, &[gpu.step(
            k_direct,
            &[&x, &wt, &bias, &y],
            &[1, cin, h, w, cout, k, st, pad, ho, wo],
            cout.div_ceil(8) * (ho * wo).div_ceil(4),
        )], reps);

        let col = gpu.storage(hw * cinkk);
        let nhwc = gpu.storage(hw * cout as u64);
        let t_low = best_of(&gpu, &[
            gpu.step(k_im2col, &[&x, &col],
                     &[cin, h, w, k, st, pad, ho, wo, cinkk as u32, 0, hw as u32], hw as u32 * cinkk as u32),
            gpu.step(k_mm, &[&col, &wt, &nhwc], &[hw as u32, cinkk as u32, cout],
                     (hw as u32).div_ceil(128) * cout.div_ceil(128) * 256),
            gpu.step(k_epi, &[&nhwc, &bias, &y], &[hw as u32 * cout, cout, hw as u32],
                     cout.div_ceil(64) * (hw as u32).div_ceil(64) * 64),
        ], reps);

        let mark = if t_direct / t_low > 1.0 { " <- lowered wins" } else { "" };
        println!(
            "({cin:4},{cout:4}) {h:4}x{w:<4}            {:>10.3} {:>10.3} {:>8.2}x{mark}",
            t_direct * 1e3, t_low * 1e3, t_direct / t_low
        );
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(|s| s == "convfwd").unwrap_or(false) {
        convfwd_ab(a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3));
        return;
    }
    if a.get(1).map(|s| s == "dwtn").unwrap_or(false) {
        dwtn_ab(a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3));
        return;
    }
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

    // Measure this device's own roofline before profiling anything: every
    // utilisation number below divides by it, so a hardcoded peak would make
    // the whole table a statement about one card (`gpu_core::roof`).
    let roofs = gpu_core::roof::ensure(&gpu);
    match roofs {
        Some(r) => println!(
            "measured roofline: {:.0} GFLOP/s, {:.1} GB/s DRAM, {:.1} GB/s cache, ridge {:.1} FLOP/byte",
            r.gflops,
            r.gbs,
            r.cache_gbs,
            r.ridge()
        ),
        None => println!("roofline unmeasured — utilisation columns print '-' rather than a guess"),
    }

    let fwd = report(&gpu, "FORWARD", tr.fwd_steps(), reps, roofs);
    let bwd = report(&gpu, "BACKWARD", tr.bwd_steps(), reps, roofs);

    println!("\nforward {:.2} ms + backward {:.2} ms = {:.2} ms/step", fwd * 1e3, bwd * 1e3, (fwd + bwd) * 1e3);
    println!("backward/forward = {:.2}x   ({} clears before each backward)", bwd / fwd, tr.bwd_clears().len());
}
