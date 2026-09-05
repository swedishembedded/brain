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
//! Roofline for the box this was written on (Tesla P40 / GP102): its datasheet
//! fp32 peak over its datasheet bandwidth puts the ridge point in the tens of
//! FLOP per byte. `matmul.wgsl` moves 8 bytes per 2 FLOP, i.e. an arithmetic
//! intensity two orders of magnitude below that ridge, so it cannot come near
//! peak no matter how wide the card is - the reason this benchmark exists.

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
    // Wide-K / wide-N prefill linears at the scale brain's larger decoders
    // dispatch (d_model 4096, ffn 14336). The square probes above are a
    // roofline instrument, not a workload: they say nothing about how a tile
    // schedule behaves when N is 3.5x M, which is the common case for an
    // up-projection and the case a 128x128 tile can most easily get wrong.
    Shape { label: "big qkv       1024x4096->6144", m: 1024, k: 4096, n: 6144 },
    Shape { label: "big ffn-up    1024x4096->14336", m: 1024, k: 4096, n: 14336 },
    Shape { label: "big ffn-down  1024x14336->4096", m: 1024, k: 14336, n: 4096 },
];

/// Kernel slots, in registration order.
const K_MATMUL: usize = 0;
const K_TILED: usize = 1;
const K_REG: usize = 2;
const K_REG2: usize = 3;
const K_REG3: usize = 4;
const K_REG4: usize = 5;

fn kernels() -> Vec<(&'static str, &'static str)> {
    vec![
        ("matmul", kernels::MATMUL),
        ("matmul_tiled", kernels::MATMUL_TILED),
        ("matmul_reg", kernels::MATMUL_REG),
        ("matmul_reg2", kernels::MATMUL_REG2),
        ("matmul_reg3", kernels::MATMUL_REG3),
        ("matmul_reg4", kernels::MATMUL_REG4),
    ]
}

/// GFLOP/s as a percent of the P40's datasheet fp32 peak, below.
const PEAK_GFLOPS: f64 = 11760.0;

/// Deterministic, bounded inputs - small magnitudes keep the fp32 accumulation
/// well-conditioned so a parity failure means a real bug, not cancellation.
fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 37 + seed * 17) % 97) as f32 / 97.0) - 0.5).collect()
}

/// Threads for the 32x32-output-tile GEMM at `@workgroup_size(64)`.
fn tiled_threads(m: usize, n: usize) -> u32 {
    (m.div_ceil(32) * n.div_ceil(32) * 64) as u32
}

/// Threads for the 128x128-output-tile GEMM at `@workgroup_size(256)`.
fn reg_threads(m: usize, n: usize) -> u32 {
    (m.div_ceil(128) * n.div_ceil(128) * 256) as u32
}

/// One kernel under test at one shape.
struct Arm {
    label: &'static str,
    kind: usize,
    threads: u32,
}

/// Run every arm `reps` times, **round-robin**, returning each arm's output and
/// its best (minimum) wall-clock time.
///
/// Two disciplines are baked in here rather than left to the caller, because
/// this repo has been burned by both:
///
/// * **Every timed region is fenced.** `submit` only *records*; without the
///   `poll_wait` inside the timed span the loop would measure command-buffer
///   construction. An earlier unfenced native-Vulkan run in this tree reported
///   a physically impossible 57 TFLOP/s exactly that way.
/// * **Arms are interleaved, not run back-to-back**, and reduced with min-of-N.
///   Run-to-run variance on this box has been measured as high as 4.6x when
///   another tenant is holding cores, so timing kernel A's whole batch and then
///   kernel B's whole batch attributes that drift to the kernel. Round-robin
///   (A, B, C, A, B, C, ...) spreads any drift across all arms equally, and the
///   minimum is the sample least polluted by it.
fn time_arms(
    gpu: &Gpu,
    arms: &[Arm],
    (m, k, n): (usize, usize, usize),
    x: &[f32],
    w: &[f32],
    reps: usize,
) -> Vec<(Vec<f32>, f64)> {
    let xb = gpu.storage_init("x", x);
    let wb = gpu.storage_init("w", w);
    let params = [m as u32, k as u32, n as u32];
    let obs: Vec<_> = arms.iter().map(|_| gpu.storage((m * n) as u64)).collect();

    // Warm every arm first (pipeline/JIT compiled, allocations resident) so the
    // first round-robin pass is not measuring one arm's shader compile.
    for (a, ob) in arms.iter().zip(&obs) {
        let s = gpu.step(a.kind, &[&xb, &wb, ob], &params, a.threads);
        gpu.submit(&[], &[s]);
        gpu.poll_wait();
    }

    let mut best = vec![f64::INFINITY; arms.len()];
    for _ in 0..reps {
        for (i, (a, ob)) in arms.iter().zip(&obs).enumerate() {
            let t0 = std::time::Instant::now();
            let s = gpu.step(a.kind, &[&xb, &wb, ob], &params, a.threads);
            gpu.submit(&[], &[s]);
            gpu.poll_wait();
            best[i] = best[i].min(t0.elapsed().as_secs_f64());
        }
    }
    arms.iter().zip(&obs).zip(best).map(|((_, ob), t)| (gpu.read(ob, m * n), t)).collect()
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
        eprintln!("(no native Vulkan device - reporting cpu + wgpu only)");
    }

    println!(
        "\n{:<32} {:>8} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6}",
        "shape", "GFLOP", "cpu", "naive", "tiled", "reg2", "reg3", "reg4", "%peak"
    );
    println!("{}", "-".repeat(110));

    for s in SHAPES {
        let (m, k, n) = (s.m, s.k, s.n);
        let x = fill(m * k, 1);
        let w = fill(n * k, 2);
        let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;

        // The CPU backend's matmul is the correctness oracle (it is also what
        // `--device cpu` actually runs, so the speedup below is the honest
        // like-for-like ratio, not a strawman).
        let cpu_arm = [Arm { label: "cpu", kind: K_MATMUL, threads: (m * n) as u32 }];
        let mut r = time_arms(&cpu, &cpu_arm, (m, k, n), &x, &w, reps);
        let (want, t_cpu) = r.pop().expect("one arm");

        let arms = [
            Arm { label: "naive", kind: K_MATMUL, threads: (m * n) as u32 },
            Arm { label: "tiled", kind: K_TILED, threads: tiled_threads(m, n) },
            Arm { label: "reg", kind: K_REG, threads: reg_threads(m, n) },
            Arm { label: "reg2", kind: K_REG2, threads: reg_threads(m, n) },
            Arm { label: "reg3", kind: K_REG3, threads: reg_threads(m, n) },
            Arm { label: "reg4", kind: K_REG4, threads: reg_threads(m, n) },
        ];
        let got = time_arms(&wgpu, &arms, (m, k, n), &x, &w, reps);

        let gfs = |t: f64| gflop / t;
        let t = |i: usize| got[i].1;
        let best = got.iter().map(|(_, t)| *t).fold(f64::INFINITY, f64::min);
        println!(
            "{:<32} {:>8.2} {:>8.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>7.0} {:>5.1}%",
            s.label,
            gflop,
            gfs(t_cpu),
            gfs(t(0)),
            gfs(t(1)),
            gfs(t(3)),
            gfs(t(4)),
            gfs(t(5)),
            100.0 * gfs(best) / PEAK_GFLOPS,
        );

        // Parity against the CPU oracle for EVERY arm - a speed number from a
        // kernel that computes the wrong thing is not a speed number. fp32 GEMM
        // accumulates K terms in whatever order the backend's loop nest
        // produces, so the gate is relative, not bitwise: K=6144 lands at
        // ~1.2e-5, which is float addition, not a bug.
        let mut parity = String::new();
        for (a, (out, _)) in arms.iter().zip(&got) {
            let (dabs, drel) = diff(&want, out);
            parity.push_str(&format!(" {} {:.0e}/{:.0e}", a.label, dabs, drel));
            assert!(drel < TOL, "{}: gpu {} diverges from cpu (rel {drel:.3e})", s.label, a.label);
        }
        println!("{:<32} parity vs cpu (max-abs/rel):{parity}", "");
        println!(
            "{:<32} ms: cpu {:>8.3}  reg2 {:>7.3}  reg3 {:>7.3}  reg4 {:>7.3}  (reg4 vs reg3 {:>5.2}x, vs cpu {:>6.1}x)",
            "",
            t_cpu * 1e3,
            t(3) * 1e3,
            t(4) * 1e3,
            t(5) * 1e3,
            t(4) / t(5),
            t_cpu / t(5),
        );

        if let Some(vk) = &vk {
            // A trivial kernel and the two register-tiled ones, so a
            // native-Vulkan/wgpu gap can be attributed: a gap on ALL of them
            // points at the dispatch/memory path, a gap on the tiled ones only
            // points at naga's SPIR-V for workgroup memory - and reg3-vs-reg4
            // answers that question for the vec4 shared path specifically.
            let varms = [
                Arm { label: "naive", kind: K_MATMUL, threads: (m * n) as u32 },
                Arm { label: "reg3", kind: K_REG3, threads: reg_threads(m, n) },
                Arm { label: "reg4", kind: K_REG4, threads: reg_threads(m, n) },
            ];
            let vg = time_arms(vk, &varms, (m, k, n), &x, &w, reps);
            println!(
                "{:<32} vulkan: naive {:>7.0}  reg3 {:>7.0}  reg4 {:>7.0} GFLOP/s",
                "",
                gfs(vg[0].1),
                gfs(vg[1].1),
                gfs(vg[2].1),
            );
            for (a, (out, _)) in varms.iter().zip(&vg) {
                let (_, drel) = diff(&want, out);
                assert!(drel < TOL, "{}: vulkan {} diverges from cpu (rel {drel:.3e})", s.label, a.label);
            }
        }
    }
}

/// int8 (DP4A) and int4 (W4A8) GEMM throughput, on the same shapes as
/// [`bench_matmul`] above but split into its own test because the quantized
/// kernels don't share one shape across backends the way the fp32 tier does:
/// `matmul_i8_dyn` (the fast, 128x128-tiled DP4A GEMM used at prefill) is
/// `@cpu no` — no CPU-JIT lowering exists for it, so brain has NO CPU int8
/// path at prefill scale — while `matmul_i8_gemv`/`matmul_q4_gemv` (the
/// decode-regime GEMMs, one workgroup per output column) are `@cpu yes` but
/// `REQUIRES m <= 32`. Reporting that gap honestly (rather than picking one
/// shape and hiding the other kernel) is the point of a second table.
///
/// ```text
/// DISPLAY= cargo test --release -p brain-gpu-core -- --ignored --nocapture bench_matmul_quant
/// ```
#[test]
#[ignore]
fn bench_matmul_quant() {
    let reps: usize = std::env::var("BRAIN_BENCH_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);

    // matmul_i8_dyn (DP4A, 128x128 tile, GPU only, any M), matmul_i8_gemv
    // (DP4A, 64-thread/column, CPU+GPU, M<=32), matmul_q4_dyn (W4A8 naive,
    // CPU+GPU, any M), matmul_q4_gemv (W4A8, 64-thread/column, CPU+GPU, M<=32).
    let ks: Vec<(&str, &str)> = vec![
        ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
        ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
        ("matmul_q4_dyn", kernels::MATMUL_Q4_DYN),
        ("matmul_q4_gemv", kernels::MATMUL_Q4_GEMV),
    ];

    let cpu = Gpu::new_cpu(&ks);
    let wgpu = Gpu::new_wgpu(&ks);

    /// Weight-scale group along K, mirroring `model::int8::GROUP`.
    const GROUP: usize = 32;

    /// Per-row symmetric quantization of an ACTIVATION (scale = max|row|/127),
    /// matching what `max_abs_row`/`quant_pack` do on device, duplicated here
    /// (gpu-core has no dependency on `brain-model`) - returns packed
    /// `[rows, k/4]` u32, per-row scale, and the UNPACKED signed values (kept
    /// for the exact host reference below; round-tripping through f32 dequant
    /// would launder a packing bug).
    fn quant_act(x: &[f32], rows: usize, k: usize) -> (Vec<u32>, Vec<f32>, Vec<i8>) {
        let kg = k / 4;
        let mut packed = vec![0u32; rows * kg];
        let mut scale = vec![0f32; rows];
        let mut q = vec![0i8; rows * k];
        for r in 0..rows {
            let row = &x[r * k..r * k + k];
            let s = row.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / 127.0;
            scale[r] = s;
            for c in 0..k {
                q[r * k + c] = (row[c] / s).round().clamp(-127.0, 127.0) as i8;
            }
            for g in 0..kg {
                let mut word = 0u32;
                for b in 0..4 {
                    word |= (q[r * k + g * 4 + b] as u8 as u32) << (8 * b);
                }
                packed[r * kg + g] = word;
            }
        }
        (packed, scale, q)
    }

    /// GROUP-wise symmetric quantization of a WEIGHT (one scale per 32
    /// elements of K), matching `model::int8::quantize_weight` /
    /// `model::int4::quantize_weight_q4` exactly. Returns packed
    /// `[rows, k/per_word]` u32, the `[rows, k/32]` scale, and the unpacked
    /// signed values.
    fn quant_weight(x: &[f32], rows: usize, k: usize, per_word: usize, qmax: f32) -> (Vec<u32>, Vec<f32>, Vec<i8>) {
        assert_eq!(k % GROUP, 0, "quant_weight: k must be a whole number of groups");
        let kg = k / per_word;
        let gs = k / GROUP;
        let bits = 32 / per_word;
        let mut packed = vec![0u32; rows * kg];
        let mut scale = vec![0f32; rows * gs];
        let mut q = vec![0i8; rows * k];
        for r in 0..rows {
            let row = &x[r * k..r * k + k];
            for g in 0..gs {
                let blk = &row[g * GROUP..g * GROUP + GROUP];
                let s = blk.iter().fold(0f32, |m, &v| m.max(v.abs())).max(1e-8) / qmax;
                scale[r * gs + g] = s;
                for c in 0..GROUP {
                    q[r * k + g * GROUP + c] = (blk[c] / s).round().clamp(-qmax, qmax) as i8;
                }
            }
            for g in 0..kg {
                let mut word = 0u32;
                for b in 0..per_word {
                    word |= ((q[r * k + g * per_word + b] as u8 as u32) & ((1 << bits) - 1)) << (bits * b);
                }
                packed[r * kg + g] = word;
            }
        }
        (packed, scale, q)
    }

    /// Reference for the group-wise contract:
    /// `out[m,n] = sx[m] * sum_g (sum_{k in g} xq*wq) * sw[n,g]`. Each group's
    /// sum is exact integer; only the cross-group fold is floating point,
    /// which is what every kernel below does too.
    fn host_group_gemm(xq: &[i8], wq: &[i8], sx: &[f32], sw: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let gs = k / GROUP;
        let mut out = vec![0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for g in 0..gs {
                    let mut ia = 0i32;
                    for ki in g * GROUP..g * GROUP + GROUP {
                        ia += xq[mi * k + ki] as i32 * wq[ni * k + ki] as i32;
                    }
                    acc += ia as f32 * sw[ni * gs + g];
                }
                out[mi * n + ni] = acc * sx[mi];
            }
        }
        out
    }

    fn diff(want: &[f32], got: &[f32]) -> (f32, f32) {
        let maxd = want.iter().zip(got).fold(0f32, |acc, (a, b)| acc.max((a - b).abs()));
        let scale = want.iter().fold(1e-6f32, |acc, v| acc.max(v.abs()));
        (maxd, maxd / scale)
    }

    /// One dispatch, `reps` timed resubmits — the int8/int4 sibling of
    /// `time_variant` above, generalized to N buffers since these kernels
    /// take 5 (x, w, sx, sw, out) instead of `time_variant`'s fixed 3.
    fn time_kernel(
        gpu: &Gpu,
        kind: usize,
        in_bufs: &[&gpu_core::DeviceBuffer],
        params: &[u32],
        threads: u32,
        out: &gpu_core::DeviceBuffer,
        out_len: usize,
        reps: usize,
    ) -> (Vec<f32>, f64) {
        // `out` is itself a bind-group entry (the kernel's last storage
        // binding) — appended here rather than left for the caller to
        // remember, since forgetting it is a silent bind-group-count
        // mismatch, not a compile error.
        let mut bufs: Vec<&gpu_core::DeviceBuffer> = in_bufs.to_vec();
        bufs.push(out);
        let s = gpu.step(kind, &bufs, params, threads);
        gpu.submit(&[], &[s]);
        gpu.poll_wait();
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            let s = gpu.step(kind, &bufs, params, threads);
            gpu.submit(&[], &[s]);
            gpu.poll_wait();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        (gpu.read(out, out_len), best)
    }

    /// Runs int8 (`matmul_i8_dyn`, GPU only — `@cpu no`) at `(m, k, n)`.
    fn run_i8_dyn(wgpu: &Gpu, m: usize, k: usize, n: usize, reps: usize) {
        let x = fill(m * k, 1);
        let w = fill(n * k, 2);
        let (xq, sx, xi) = quant_act(&x, m, k);
        let (wq, sw, wi) = quant_weight(&w, n, k, 4, 127.0);
        let want = host_group_gemm(&xi, &wi, &sx, &sw, m, k, n);

        let xb = wgpu.storage(xq.len() as u64);
        wgpu.write(&xb, &xq);
        let wb = wgpu.storage(wq.len() as u64);
        wgpu.write(&wb, &wq);
        let sxb = wgpu.storage_init("sx", &sx);
        let swb = wgpu.storage_init("sw", &sw);
        let ob = wgpu.storage((m * n) as u64);
        let tiles = (m.div_ceil(128) * n.div_ceil(128) * 256) as u32;
        let ki = wgpu.kernel_index("matmul_i8_dyn").expect("registered above");
        let (got, t) = time_kernel(wgpu, ki, &[&xb, &wb, &sxb, &swb], &[m as u32, (k / 4) as u32, n as u32], tiles, &ob, m * n, reps);
        let (dabs, drel) = diff(&want, &got);
        let gops = 2.0 * m as f64 * k as f64 * n as f64 / t / 1e9;
        println!(
            "int8 dyn (DP4A)  m={m:<5} k={k:<5} n={n:<5}  gpu {:>8.3} ms  {:>8.0} GOP/s  max-abs/rel {:.2e}/{:.2e}  (no CPU-capable kernel at this M)",
            t * 1e3, gops, dabs, drel
        );
        assert!(drel < 1e-5, "matmul_i8_dyn diverges (rel {drel:.3e})");
    }

    /// Runs int4/W4A8 (`matmul_q4_dyn`, CPU+GPU — naive, "correct, then
    /// freeze") at `(m, k, n)` on both backends.
    fn run_q4_dyn(cpu: &Gpu, wgpu: &Gpu, m: usize, k: usize, n: usize, reps: usize) {
        let x = fill(m * k, 3);
        let w = fill(n * k, 4);
        let (xq, sx, xi) = quant_act(&x, m, k); // W4A8: activations stay int8
        let (wq, sw, wi) = quant_weight(&w, n, k, 8, 7.0); // weights are int4
        let want = host_group_gemm(&xi, &wi, &sx, &sw, m, k, n);
        let threads = (m * n) as u32;

        for (label, g) in [("cpu", cpu), ("gpu", wgpu)] {
            let xb = g.storage(xq.len() as u64);
            g.write(&xb, &xq);
            let wb = g.storage(wq.len() as u64);
            g.write(&wb, &wq);
            let sxb = g.storage_init("sx", &sx);
            let swb = g.storage_init("sw", &sw);
            let ob = g.storage((m * n) as u64);
            let ki = g.kernel_index("matmul_q4_dyn").expect("registered above");
            let (got, t) = time_kernel(g, ki, &[&xb, &wb, &sxb, &swb], &[m as u32, k as u32, n as u32], threads, &ob, m * n, reps);
            let (dabs, drel) = diff(&want, &got);
            let gops = 2.0 * m as f64 * k as f64 * n as f64 / t / 1e9;
            println!(
                "int4 dyn (W4A8)  m={m:<5} k={k:<5} n={n:<5}  {label:<3} {:>8.3} ms  {:>8.0} GOP/s  max-abs/rel {:.2e}/{:.2e}",
                t * 1e3, gops, dabs, drel
            );
            assert!(drel < 1e-5, "matmul_q4_dyn diverges on {label} (rel {drel:.3e})");
        }
    }

    /// Runs the decode-regime (`m <= 32`) int8 and int4 GEMV kernels on both
    /// backends — the ONLY shape where brain has a CPU int8 GEMM at all.
    fn run_decode_gemv(cpu: &Gpu, wgpu: &Gpu, m: usize, k: usize, n: usize, reps: usize) {
        let x = fill(m * k, 5);
        let w8 = fill(n * k, 6);
        let w4 = fill(n * k, 7);
        let (xq, sx, xi) = quant_act(&x, m, k);
        let (wq8, sw8, wi8) = quant_weight(&w8, n, k, 4, 127.0);
        let (wq4, sw4, wi4) = quant_weight(&w4, n, k, 8, 7.0);
        let want8 = host_group_gemm(&xi, &wi8, &sx, &sw8, m, k, n);
        let want4 = host_group_gemm(&xi, &wi4, &sx, &sw4, m, k, n);
        let threads = (n * 64) as u32;

        for (label, g) in [("cpu", cpu), ("gpu", wgpu)] {
            let xb = g.storage(xq.len() as u64);
            g.write(&xb, &xq);
            let sxb = g.storage_init("sx", &sx);

            let wb8 = g.storage(wq8.len() as u64);
            g.write(&wb8, &wq8);
            let swb8 = g.storage_init("sw8", &sw8);
            let ob8 = g.storage((m * n) as u64);
            let ki8 = g.kernel_index("matmul_i8_gemv").expect("registered above");
            let (got8, t8) =
                time_kernel(g, ki8, &[&xb, &wb8, &sxb, &swb8], &[m as u32, (k / 4) as u32, n as u32], threads, &ob8, m * n, reps);
            let (d8abs, d8rel) = diff(&want8, &got8);

            let wb4 = g.storage(wq4.len() as u64);
            g.write(&wb4, &wq4);
            let swb4 = g.storage_init("sw4", &sw4);
            let ob4 = g.storage((m * n) as u64);
            let ki4 = g.kernel_index("matmul_q4_gemv").expect("registered above");
            let (got4, t4) = time_kernel(g, ki4, &[&xb, &wb4, &sxb, &swb4], &[m as u32, k as u32, n as u32], threads, &ob4, m * n, reps);
            let (d4abs, d4rel) = diff(&want4, &got4);

            println!(
                "decode gemv      m={m:<5} k={k:<5} n={n:<5}  {label:<3} int8 {:>7.3} ms  int4 {:>7.3} ms  \
                 parity int8 {:.2e}/{:.2e}  int4 {:.2e}/{:.2e}",
                t8 * 1e3, t4 * 1e3, d8abs, d8rel, d4abs, d4rel
            );
            assert!(d8rel < 1e-5, "matmul_i8_gemv diverges on {label} (rel {d8rel:.3e})");
            assert!(d4rel < 1e-5, "matmul_q4_gemv diverges on {label} (rel {d4rel:.3e})");
        }
    }

    println!("\n-- prefill-shape (int8 DP4A is GPU-only; int4 W4A8 is CPU+GPU, naive) --");
    for &(m, k, n) in &[(1024usize, 1024usize, 1024usize), (2048, 2048, 2048)] {
        run_i8_dyn(&wgpu, m, k, n, reps);
        run_q4_dyn(&cpu, &wgpu, m, k, n, reps);
    }

    println!("\n-- decode-shape (m<=32; the only shape with a CPU-capable int8 kernel) --");
    run_decode_gemv(&cpu, &wgpu, 8, 1024, 1024, reps);
}
