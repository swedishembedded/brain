// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device-timed cost of the memory-bound glue kernels of a served prefill round against their byte roofline, at the
//! production shapes (a 4096-row round of the 27B FP8 model).
//!
//! Swedish Embedded AB implements solutions for memory-bound GPU kernels in LLM serving for its clients. If your team needs
//! expertise in bringing norm, quantisation and activation passes to the memory roofline then you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! `#[ignore]`d - a measurement, not a gate (the byte-identity gates are the `*_native.rs` tests of each kernel):
//!
//! ```text
//! BRAIN_DEVICE=gpu0 BRAIN_BACKEND=cuda make test/one P=brain-gpu-core T=glue_roof_bench ARGS="--ignored --nocapture"
//! ```
//!
//! Time is the kernels' own from the backend's device events (`Gpu::kernel_times`), a launch after a warm-up, buffers cold
//! enough that a round's tensors do not sit in L2 (a 4096-row tensor is tens of megabytes). The roof is the card's measured
//! streaming rate, `ROOF_GBS`; a kernel is at its roof when it moves its bytes at that rate.

use gpu_core::{ActPrecision, Fused, Gpu};

/// What a streaming kernel sustains on a GH200 (GB/s), the denominator of every percentage below.
const ROOF_GBS: f64 = 3400.0;
const REPS: usize = 20;
const ROUND_ROWS: u32 = 4096;

fn time_us(gpu: &Gpu, steps: &dyn Fn() -> Vec<gpu_core::Step>) -> f64 {
    let run = || {
        let all: Vec<_> = (0..REPS).flat_map(|_| steps()).collect();
        gpu.reset_kernel_times();
        gpu.submit(&[], &all);
        gpu.poll_wait();
        let rows = gpu.kernel_times().expect("device timing on this backend");
        rows.iter().map(|(_, ms, _)| ms).sum::<f64>() * 1e3 / REPS as f64
    };
    run();
    (0..4).map(|_| run()).fold(f64::MAX, f64::min)
}

fn report(what: &str, us: f64, bytes: f64) {
    let gbs = bytes / us / 1e3;
    println!("{what:<56} {us:>9.1} us {:>8.1} MB {gbs:>8.0} GB/s {:>6.1} % of roof", bytes / 1e6, 100.0 * gbs / ROOF_GBS);
}

fn filled(gpu: &Gpu, n: usize) -> backend_api::DeviceBuffer {
    let v: Vec<f32> = (0..n).map(|i| ((i * 37 % 1001) as f32 - 500.0) * 0.003).collect();
    gpu.storage_init("data", &v)
}

/// `n` bf16 values (two to a word) of the magnitudes an MLP gate/up product carries.
fn filled_bf16(gpu: &Gpu, n: usize) -> backend_api::DeviceBuffer {
    let bf16 = |i: usize| (((i * 37 % 1001) as f32 - 500.0) * 0.003).to_bits() >> 16;
    let words: Vec<f32> = (0..n / 2).map(|i| f32::from_bits(bf16(2 * i) | (bf16(2 * i + 1) << 16))).collect();
    gpu.storage_init("data", &words)
}

#[test]
#[ignore]
fn glue_kernels_against_the_byte_roof_at_a_4096_row_round() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if gpu.kind() != "cuda" {
        brain_testutil::skip_unavailable("device timing here is the CUDA backend's");
        return;
    }
    assert!(gpu.set_kernel_timing(true));
    let rows = ROUND_ROWS;

    // SwiGLU product + E4M3 rows: bf16 gate and up in (2 x 2 bytes), E4M3 byte + a scale per 128 out.
    let k = 17408u32;
    let n = (rows * k) as usize;
    let (g, u) = (filled_bf16(&gpu, n), filled_bf16(&gpu, n));
    let bytes = n as f64 * (4.0 + 1.0) + (n / 128) as f64 * 4.0;
    let us = time_us(&gpu, &|| {
        let mut steps = Vec::new();
        gpu_core::provider::library::silu_mul_quantize_e4m3_rows(&gpu, &mut steps, &g, &u, ActPrecision::Bf16, None, rows, k).expect("offered");
        steps
    });
    report("silu_mul_quant_e4m3_128 bf16 in, no product (4096x17408)", us, bytes);

    // The norm front end: a, b in (f32), sum out (f32), E4M3 rows + scales out; xn is stored only for a GDN layer's dense gates.
    let d = 5120u32;
    let n = (rows * d) as usize;
    let (a, b, w) = (filled(&gpu, n), filled(&gpu, n), filled(&gpu, d as usize));
    let (sum, xn, q, s) = (gpu.storage(n as u64), gpu.storage(n as u64), gpu.storage(n as u64 / 4), gpu.storage(n as u64 / 128));
    for (label, flags, out_f32_rows) in [("add_rms_quant_e4m3 xn not stored (4096x5120)", 1u32 | 16, 1.0), ("add_rms_quant_e4m3 xn stored (4096x5120)", 1, 2.0)] {
        let us = time_us(&gpu, &|| vec![gpu.fused_step(Fused::AddRmsQuantE4m3, &[&a, &b, &w, &sum, &xn, &q, &s], &[d, rows, 1e-6f32.to_bits(), flags]).expect("offered")]);
        let bytes = n as f64 * (4.0 * (2.0 + out_f32_rows) + 1.0) + (n / 128) as f64 * 4.0;
        report(label, us, bytes);
    }

    // The attention layer's front half: the query heads' value halves and the key heads in, the normalised rotated heads out
    // (the gate half is not touched: the gate-mul pass below reads it in place).
    let (nh, nkv, hd, half) = (24u32, 4u32, 256u32, 32u32);
    let (q_full, k) = (filled(&gpu, (rows * 2 * nh * hd) as usize), filled(&gpu, (rows * nkv * hd) as usize));
    let (qw, kw) = (filled(&gpu, hd as usize), filled(&gpu, hd as usize));
    let (cos, sin) = (filled(&gpu, (rows * half) as usize), filled(&gpu, (rows * half) as usize));
    let shape = gpu_core::provider::library::GqaQkShape { rows, n_heads: nh, n_kv_heads: nkv, head_dim: hd, rotary_half: half, rms_eps: 1e-6 };
    let us = time_us(&gpu, &|| {
        let mut steps = Vec::new();
        gpu_core::provider::library::gqa_qk_prepare(&gpu, &mut steps, &shape, &q_full, &k, &qw, &kw, &cos, &sin).expect("offered");
        steps
    });
    report("gqa_qk_prep query+key (4096 rows, 24+4 heads of 256)", us, f64::from(rows * (nh + nkv) * hd) * 8.0);

    // The gate: attention output and gate in (f32), E4M3 bytes + scales out.
    let k = nh * hd;
    let n = (rows * k) as usize;
    let ctx = filled(&gpu, n);
    let us = time_us(&gpu, &|| {
        let mut steps = Vec::new();
        gpu_core::provider::library::gate_mul_quantize_e4m3_rows(&gpu, &mut steps, &ctx, &q_full, None, rows, k, hd).expect("offered");
        steps
    });
    report("gate_mul_quant_e4m3_128 no product (4096x6144)", us, n as f64 * (8.0 + 1.0) + (n / 128) as f64 * 4.0);
}
