// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GEMM parity + throughput across every backend, on the shapes brain's models
//! actually dispatch.
//!
//! ```text
//! DISPLAY= cargo test --release -p brain-gpu-core -- --ignored --nocapture bench_matmul
//! ```
//!
//! Two questions, one table:
//!
//! * **is the GPU equal to the CPU reference** — every variant's output is
//!   diffed against the CPU backend's `matmul` result (max-abs and relative,
//!   fp32 reduction-order differences only), and
//! * **is it faster** — min-of-N GFLOP/s per variant.
//!
//! `out = x·Wᵀ` with `x:[M,K]` and `W:[N,K]`, matching `nn.Linear`.
//!
//! Roofline for the box this was written on (Tesla P40 / GP102): 11.76 TFLOP/s
//! fp32 against 346 GB/s, i.e. a ridge point of ~34 FLOP/byte. `matmul.wgsl`
//! moves 8 bytes per 2 FLOP, so it cannot exceed ~86 GFLOP/s no matter how
//! wide the card is — the reason this benchmark exists.

use gpu_core::Gpu;

/// Relative-difference gate for a cross-backend fp32 GEMM. The dominant term is
/// the K-length accumulation: ~1e-6 at K=384, ~1.2e-5 at K=6144.
const TOL: f32 = 5e-5;

struct Shape {
    label: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

/// Real dispatch shapes: the `[B*T, d] × [d_out, d]` linears brain's decoders
/// run, plus square roofline probes.
const SHAPES: &[Shape] = &[
    Shape { label: "gpt-small qkv   B*T=512  384->1152", m: 512, k: 384, n: 1152 },
    Shape { label: "gpt-small mlp   B*T=512  384->1536", m: 512, k: 384, n: 1536 },
    Shape { label: "qwen0.6b qkv    B*T=256 1024->3072", m: 256, k: 1024, n: 3072 },
    Shape { label: "qwen0.6b down   B*T=256 3072->1024", m: 256, k: 3072, n: 1024 },
    Shape { label: "glm mla-ish     B*T=512 6144->2048", m: 512, k: 6144, n: 2048 },
    Shape { label: "tts talker q_proj 256x1024->2048", m: 256, k: 1024, n: 2048 },
    Shape { label: "tts talker ffn-dn 256x3072->1024", m: 256, k: 3072, n: 1024 },
    Shape { label: "square 1024", m: 1024, k: 1024, n: 1024 },
    Shape { label: "square 2048", m: 2048, k: 2048, n: 2048 },
];

/// Kernel slots, in registration order.
const K_MATMUL: usize = 0;
const K_TILED: usize = 1;
const K_REG: usize = 2;
const K_REG2: usize = 3;

fn kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("matmul", kernels::MATMUL),
        ("matmul_tiled", kernels::MATMUL_TILED),
        ("matmul_reg", kernels::MATMUL_REG),
        ("matmul_reg2", kernels::MATMUL_REG2),
    ]
}

/// GFLOP/s as a percent of the P40's 11.76 TFLOP/s fp32 peak.
const PEAK_GFLOPS: f64 = 11760.0;

/// Deterministic, bounded inputs — small magnitudes keep the fp32 accumulation
/// well-conditioned so a parity failure means a real bug, not cancellation.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 17) % 97) as f32 / 97.0) - 0.5).collect()
}

/// Threads for the 32×32-output-tile GEMM at `@workgroup_size(64)`.
fn tiled_threads(m: usize, n: usize) -> u32 {
    (m.div_ceil(32) * n.div_ceil(32) * 64) as u32
}

/// Threads for the 128×128-output-tile GEMM at `@workgroup_size(256)`.
fn reg_threads(m: usize, n: usize) -> u32 {
    (m.div_ceil(128) * n.div_ceil(128) * 256) as u32
}

/// Run one variant `reps` times, returning (output, best wall-clock seconds).
///
/// `poll_wait` after `submit` is what makes the timing real: `submit` only
/// records, so without it the loop would time command-buffer construction.
fn time_variant(
    gpu: &Gpu,
    kind: usize,
    threads: u32,
    (m, k, n): (usize, usize, usize),
    x: &[f32],
    w: &[f32],
    reps: usize,
) -> (Vec<f32>, f64) {
    let xb = gpu.storage_init("x", x);
    let wb = gpu.storage_init("w", w);
    let ob = gpu.storage((m * n) as u64);
    let params = [m as u32, k as u32, n as u32];

    // one warm-up (pipeline/JIT warm, allocations resident)
    let s = gpu.step(kind, &[&xb, &wb, &ob], &params, threads);
    gpu.submit(&[], &[s]);
    gpu.poll_wait();

    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        let s = gpu.step(kind, &[&xb, &wb, &ob], &params, threads);
        gpu.submit(&[], &[s]);
        gpu.poll_wait();
        best = best.min(t0.elapsed().as_secs_f64());
    }
    (gpu.read(&ob, m * n), best)
}

/// max-abs and relative difference of `got` against `want`.
fn diff(want: &[f32], got: &[f32]) -> (f32, f32) {
    let maxd = want.iter().zip(got).fold(0f32, |acc, (a, b)| acc.max((a - b).abs()));
    let scale = want.iter().fold(1e-6f32, |acc, v| acc.max(v.abs()));
    (maxd, maxd / scale)
}

#[test]
#[ignore]
fn bench_matmul() {
    let reps: usize = std::env::var("BRAIN_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let ks = kernels();

    let cpu = Gpu::new_cpu(&ks);
    let wgpu = Gpu::new_wgpu(&ks);
    let vk = Gpu::try_new_vulkan(&ks).ok();
    if vk.is_none() {
        eprintln!("(no native Vulkan device — reporting cpu + wgpu only)");
    }

    println!(
        "\n{:<30} {:>8} {:>9} {:>9} {:>9} {:>9} {:>6} {:>8}",
        "shape", "GFLOP", "cpu avx2", "gpu naive", "gpu reg", "gpu reg2", "%peak", "reg2/cpu"
    );
    println!("{}", "-".repeat(104));

    for s in SHAPES {
        let (m, k, n) = (s.m, s.k, s.n);
        let x = fill(m * k, 1);
        let w = fill(n * k, 2);
        let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;

        // CPU backend's naive matmul is the correctness oracle (it is also what
        // `--device cpu` actually runs, so the speedup below is the honest
        // like-for-like ratio, not a strawman).
        let (want, t_cpu) = time_variant(&cpu, K_MATMUL, (m * n) as u32, (m, k, n), &x, &w, reps);

        let (g_naive, t_gn) = time_variant(&wgpu, K_MATMUL, (m * n) as u32, (m, k, n), &x, &w, reps);
        let (dn_abs, dn_rel) = diff(&want, &g_naive);

        let (g_tiled, t_gt) =
            time_variant(&wgpu, K_TILED, tiled_threads(m, n), (m, k, n), &x, &w, reps);
        let (dt_abs, dt_rel) = diff(&want, &g_tiled);

        let (g_reg, t_gr) = time_variant(&wgpu, K_REG, reg_threads(m, n), (m, k, n), &x, &w, reps);
        let (dr_abs, dr_rel) = diff(&want, &g_reg);

        let (g_reg2, t_gr2) = time_variant(&wgpu, K_REG2, reg_threads(m, n), (m, k, n), &x, &w, reps);
        let (dr2_abs, dr2_rel) = diff(&want, &g_reg2);


        let gfs = |t: f64| gflop / t;
        println!(
            "{:<30} {:>8.2} {:>9.0} {:>9.0} {:>9.0} {:>9.0} {:>5.1}% {:>7.1}x",
            s.label,
            gflop,
            gfs(t_cpu),
            gfs(t_gn),
            gfs(t_gr),
            gfs(t_gr2),
            100.0 * gfs(t_gr2) / PEAK_GFLOPS,
            t_cpu / t_gr2
        );
        println!(
            "{:<30} parity vs cpu (rel): naive {:.1e}  tiled {:.1e}  reg {:.1e}  reg2 {:.1e}",
            "", dn_rel, dt_rel, dr_rel, dr2_rel
        );
        let _ = (dt_abs, dr_abs, dr2_abs);
        assert!(dr_rel < TOL, "{}: gpu reg diverges from cpu (rel {dr_rel:.3e})", s.label);
        assert!(dr2_rel < TOL, "{}: gpu reg2 diverges from cpu (rel {dr2_rel:.3e})", s.label);

        // fp32 GEMM accumulates K terms in whatever order the backend's loop
        // nest produces, so the gate is relative, not bitwise. K=6144 lands at
        // ~1.2e-5 — that is float addition, not a bug.
        assert!(dn_rel < TOL, "{}: gpu naive diverges from cpu (rel {dn_rel:.3e})", s.label);
        assert!(dt_rel < TOL, "{}: gpu tiled diverges from cpu (rel {dt_rel:.3e})", s.label);

        if let Some(vk) = &vk {
            // Both variants, so a native-Vulkan/wgpu gap can be attributed:
            // a gap on BOTH points at the dispatch/memory path, a gap on the
            // tiled one only points at naga's SPIR-V for workgroup memory.
            let (v_naive, t_vn) = time_variant(vk, K_MATMUL, (m * n) as u32, (m, k, n), &x, &w, reps);
            let (v_tiled, t_vt) =
                time_variant(vk, K_TILED, tiled_threads(m, n), (m, k, n), &x, &w, reps);
            let (_, dvn_rel) = diff(&want, &v_naive);
            let (dv_abs, dv_rel) = diff(&want, &v_tiled);
            println!(
                "{:<34} vulkan: naive {:.1} tiled {:.1} GFLOP/s, tiled max-abs {:.2e} rel {:.2e}",
                "",
                gfs(t_vn),
                gfs(t_vt),
                dv_abs,
                dv_rel
            );
            assert!(dvn_rel < TOL, "{}: vulkan naive diverges from cpu (rel {dvn_rel:.3e})", s.label);
            assert!(dv_rel < TOL, "{}: vulkan tiled diverges from cpu (rel {dv_rel:.3e})", s.label);
        }
    }
}
