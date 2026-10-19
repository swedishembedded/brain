// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for prompt-processing throughput on
// quantised language models for its clients. If your team needs expertise in
// putting int8 weights onto Hopper warpgroup tensor cores without giving up
// their group-wise scales, then you can procure our services by sending an
// email to info@swedishembedded.com.

//! `matmul_i8w_bf16a_wgmma` against a host reference: int8 weights with
//! group-wise scales decoded to bf16, bf16 activations, f32 accumulation on the
//! warpgroup tensor cores. Skipped off Hopper. The shapes include a ragged M
//! (rows past the end of the activations) and the two K depths of the 27B
//! (5120 and 17408).

use backend_cuda::exec::{CompileOptions, Context, Shape};
use backend_cuda::nvrtc::ArchFeatures;

const SRC: &str = include_str!("../../kernels-cuda/cu/matmul_i8w_bf16a_wgmma.cu");
const SMEM: u32 = 219136 + 1024;

fn bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    // round to nearest even
    let r = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    (r >> 16) as u16
}
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// Run the kernel; returns (out, seconds per launch).
fn run(ctx: &Context, m: usize, n: usize, k: usize, reps: u32) -> (Vec<f32>, Vec<f32>, f64) {
    run_variant(ctx, &Variant::SINGLE, m, n, k, reps)
}

/// A kernel source, its entry point and block size, and the compile-time probe defines it is built with.
struct Variant {
    label: &'static str,
    src: &'static str,
    entry: &'static str,
    threads: u32,
    defines: &'static [&'static str],
}

impl Variant {
    const SINGLE: Variant = Variant { label: "single role", src: SRC, entry: "brain_matmul_i8w_bf16a_wgmma", threads: 256, defines: &[] };
}

fn run_variant(ctx: &Context, v: &Variant, m: usize, n: usize, k: usize, reps: u32) -> (Vec<f32>, Vec<f32>, f64) {
    let (src, entry, threads) = (v.src, v.entry, v.threads);
    let defines: Vec<(String, String)> = v.defines.iter().map(|d| (d.to_string(), "1".to_string())).collect();
    let mut rng = Lcg(0x5eed ^ (m * 31 + n * 7 + k) as u64); // seed: any data serves, the reference reads the same bytes
    let a: Vec<u16> = (0..m * k).map(|_| bf16_bits((rng.next() - 0.5) * 4.0)).collect();
    let w: Vec<i8> = (0..n * k).map(|_| ((rng.next() * 255.0) as i32 - 128) as i8).collect();
    let sw: Vec<f32> = (0..n * (k / 32)).map(|_| 0.002 + rng.next() * 0.02).collect();

    // Reference: the decoded weights are bf16(w * s), the products accumulate in f64.
    let mut want = vec![0f32; m * n];
    let wd: Vec<f32> = (0..n * k).map(|i| bf16_to_f32(bf16_bits(w[i] as f32 * sw[(i / k) * (k / 32) + (i % k) / 32]))).collect();
    let af: Vec<f32> = a.iter().map(|&h| bf16_to_f32(h)).collect();
    if m * n * k <= 400_000_000 {
        for r in 0..m {
            for c in 0..n {
                let mut acc = 0f64;
                for i in 0..k {
                    acc += af[r * k + i] as f64 * wd[c * k + i] as f64;
                }
                want[r * n + c] = acc as f32;
            }
        }
    }

    let bytes = |v: &[u8]| v.to_vec();
    let a_mem = ctx.alloc(a.len() * 2).unwrap();
    ctx.upload(&a_mem, &bytes(unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 2) })).unwrap();
    let w_mem = ctx.alloc(w.len()).unwrap();
    ctx.upload(&w_mem, &bytes(unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len()) })).unwrap();
    let s_mem = ctx.alloc(sw.len() * 4).unwrap();
    ctx.upload(&s_mem, &bytes(unsafe { std::slice::from_raw_parts(sw.as_ptr() as *const u8, sw.len() * 4) })).unwrap();
    let p_mem = ctx.alloc(12).unwrap();
    let params = [m as u32, k as u32, n as u32];
    ctx.upload(&p_mem, &bytes(unsafe { std::slice::from_raw_parts(params.as_ptr() as *const u8, 12) })).unwrap();
    let o_mem = ctx.alloc(m * n * 4).unwrap();
    ctx.zero(&o_mem).unwrap();

    let module = ctx.compile_with(src, entry, &CompileOptions { features: ArchFeatures::Required, defines }).expect("compile");
    let f = module.function(entry).expect("entry");
    let raw = f.raw();
    unsafe { ctx.allow_dynamic_shared(raw, SMEM).expect("opt in to the shared memory") };
    let grid = ((m.div_ceil(128) * (n / 128)) as u32, 1, 1);
    let args = [p_mem.device_ptr(), a_mem.device_ptr(), w_mem.device_ptr(), s_mem.device_ptr(), o_mem.device_ptr()];
    let launch = || unsafe { ctx.launch_shaped(raw, grid, (threads, 1, 1), &Shape { shared_bytes: SMEM, cluster: None }, &args).expect("launch") };
    launch();
    ctx.sync().unwrap();
    let t = std::time::Instant::now();
    for _ in 0..reps {
        launch();
    }
    ctx.sync().unwrap();
    let secs = t.elapsed().as_secs_f64() / reps as f64;
    let mut got_bytes = vec![0u8; m * n * 4];
    ctx.download(&o_mem, &mut got_bytes).unwrap();
    let got: Vec<f32> = got_bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (got, want, secs)
}

fn open() -> Option<Context> {
    let ctx = match Context::open(0) {
        Ok(c) => c,
        Err(e) => {
            brain_testutil::skip_unavailable(&format!("no usable CUDA device: {e}"));
            return None;
        }
    };
    if ctx.compute_capability() != (9, 0) || backend_cuda::nvrtc::version().is_err() {
        brain_testutil::skip_unavailable("needs Hopper and NVRTC");
        return None;
    }
    Some(ctx)
}

#[test]
fn the_wgmma_kernel_matches_the_host_reference() {
    let Some(ctx) = open() else { return };
    for (m, n, k) in [(128, 128, 128), (200, 256, 512), (64, 384, 1024)] {
        let (got, want, _) = run(&ctx, m, n, k, 1);
        let rms = (want.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
        let worst = got.iter().zip(&want).fold(0f64, |w, (g, e)| w.max((*g as f64 - *e as f64).abs()));
        assert!(worst < 2e-3 * rms, "m={m} n={n} k={k}: worst error {worst} against an output rms of {rms}");
    }
}

/// Prints the rate at the 27B's prefill shapes (a measurement, not a gate).
#[test]
fn the_wgmma_kernel_rate_at_the_27b_shapes() {
    let Some(ctx) = open() else { return };
    for (m, n, k) in [(512, 17408, 5120), (512, 5120, 17408), (1024, 17408, 5120), (512, 10240, 5120)] {
        let (_, _, secs) = run(&ctx, m, n, k, 20);
        println!("m={m} n={n} k={k}: {:.3} ms, {:.0} TFLOP/s", secs * 1e3, 2.0 * (m * n * k) as f64 / secs / 1e12);
    }
}

const WS_SRC: &str = include_str!("../../kernels-cuda/cu/matmul_i8w_bf16a_wgmma_ws.cu");

/// Where the time of the kernel goes: the same GEMM with parts compiled out (probe defines), and
/// the warp-specialised variant. A measurement (`--ignored --nocapture`), not a gate.
#[test]
#[ignore = "a measurement; needs a Hopper card"]
fn the_variants_rate_at_the_27b_shapes() {
    let Some(ctx) = open() else { return };
    let variants = [
        Variant::SINGLE,
        Variant { label: "no decode", defines: &["BRAIN_PROBE_NO_DECODE"], ..Variant::SINGLE },
        Variant { label: "no loads", defines: &["BRAIN_PROBE_NO_LOAD"], ..Variant::SINGLE },
        Variant { label: "no MMA", defines: &["BRAIN_PROBE_NO_MMA"], ..Variant::SINGLE },
        Variant { label: "MMA only", defines: &["BRAIN_PROBE_NO_DECODE", "BRAIN_PROBE_NO_LOAD"], ..Variant::SINGLE },
        Variant { label: "warp specialised", src: WS_SRC, entry: "brain_matmul_i8w_bf16a_wgmma_ws", threads: 384, defines: &[] },
    ];
    for v in &variants {
        for (m, n, k) in [(512, 17408, 5120), (1024, 17408, 5120)] {
            let (_, _, secs) = run_variant(&ctx, v, m, n, k, 20);
            println!("{:>18}: m={m} n={n} k={k}: {:.3} ms, {:.0} TFLOP/s", v.label, secs * 1e3, 2.0 * (m * n * k) as f64 / secs / 1e12);
        }
    }
}
