// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A LoRA on an INT8 linear must be applied beside the base matmul, never
//! folded into the quantized weight.
//!
//! Swedish Embedded AB implements low-rank adapter serving on quantized
//! transformers for its clients. If your team needs expertise in INT8
//! inference or adapter deployment then you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! ## The two deployments, measured against the same oracle
//!
//! The delta a trained adapter asks for is `Δy = s·B·(A·x)`. There are two
//! ways to deliver it on a quantized linear, and this test runs both against a
//! host-computed `Δy` with the activation quantization held IDENTICAL, so the
//! only thing that varies is where the adapter was applied:
//!
//! * **Folded** - rebuild `W' = W + s·B·A` and requantize. An int8 weight has
//!   256 levels and a trained delta is typically a fraction of one, so most of
//!   it rounds straight back to the base code. What comes out is not a small
//!   error on `Δy`; it is mostly not `Δy` at all.
//! * **Runtime** ([`model::dispatch::lora_rows_off`]) - leave `W` exactly as
//!   the checkpoint quantized it and add `s·B·(A·x)` in fp32. Nothing is
//!   requantized, so the delta arrives whole.
//!
//! Both assertions are load-bearing. The first gates the fix. The second pins
//! the defect it replaces: if folding ever became accurate here, the premise
//! this design rests on would have changed and the right response is to find
//! out why, not to relax the bound.

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block::GemmVariants;
use model::dispatch::{lora_rows_off, mm8_rows_off, I8Scratch, LoraW};
use model::int8::{dequantize_weight, quantize_weight};

const PIPES: &[(&str, &str)] = &[
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("lora_delta", kernels::LORA_DELTA),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn upload_u32(g: &Gpu, data: &[u32]) -> DeviceBuffer {
    let b = g.storage(data.len() as u64);
    g.write(&b, data);
    b
}

fn upload(g: &Gpu, data: &[f32]) -> DeviceBuffer {
    g.storage_init("lora_runtime_delta", data)
}

fn fill(seed: u64, n: usize, amp: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp
        })
        .collect()
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (&a, &b) in got.iter().zip(want) {
        num += ((a - b) as f64).powi(2);
        den += (b as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

/// `[m, n]` = `x [m, k]` against `w [n, k]`, in f64 - the oracle.
fn matmul(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut o = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f64;
            for c in 0..k {
                acc += x[i * k + c] as f64 * w[j * k + c] as f64;
            }
            o[i * n + j] = acc as f32;
        }
    }
    o
}

#[test]
fn a_runtime_lora_delivers_the_whole_delta_a_folded_one_does_not() {
    let gpu = gpu_core::testgpu::dev(PIPES);
    if !gpu.caps().workgroup_reductions {
        brain_testutil::skip_unavailable(&format!("int8 DP4A needs a GPU backend, current is {}", gpu.kind()));
        return;
    }
    let (m, k, n, r) = (128usize, 256usize, 192usize, 8usize);
    let x = fill(0xA1, m * k, 1.0);
    let w = fill(0xB2, n * k, 0.05);
    let a = fill(0xC3, r * k, 0.05);
    // `B` is sized so the folded delta sits at ~0.5% of the weight magnitude -
    // the regime a trained adapter actually occupies, and the one
    // `flux2/tests/lora_requant_int8.rs` measures as 94.8% of weights rounding
    // back to their base code.
    let b0 = fill(0xD4, n * r, 1.0);
    let raw_delta = {
        let mut d = vec![0f32; n * k];
        for o in 0..n {
            for kk in 0..r {
                let bok = b0[o * r + kk];
                for i in 0..k {
                    d[o * k + i] += bok * a[kk * k + i];
                }
            }
        }
        d
    };
    let rms = |v: &[f32]| (v.iter().map(|&z| z as f64 * z as f64).sum::<f64>() / v.len() as f64).sqrt();
    let s = (rms(&w) * 0.005 / rms(&raw_delta)) as f32;
    let delta: Vec<f32> = raw_delta.iter().map(|v| v * s).collect();

    // The base weight, exactly as a checkpoint quantized it.
    let (wq, sw) = quantize_weight(&w, n, k);
    let w_i8 = dequantize_weight(&wq, &sw, n, k);
    // What a FOLDED deployment ships instead: the same weight plus the delta,
    // requantized.
    let folded: Vec<f32> = w_i8.iter().zip(&delta).map(|(a, b)| a + b).collect();
    let (wq_folded, sw_folded) = quantize_weight(&folded, n, k);

    // One activation quantization, shared by all three runs, so nothing but
    // the weight/adapter treatment can move the result.
    let i8tier = GemmVariants::Fast { gemv: None, tiled: idx(&gpu, "matmul_i8_dyn") };
    let f32tier = GemmVariants::Fast { gemv: None, tiled: idx(&gpu, "matmul_reg3") };
    let scr = I8Scratch::new(&gpu, m as u64, m as u64, &[k as u32]);
    let xb = upload(&gpu, &x);
    let mut steps = Vec::new();
    scr.quant_rows(&gpu, [idx(&gpu, "max_abs_row"), idx(&gpu, "quant_pack")], &mut steps, &xb, 0, m as u32, k as u32);

    let run = |extra: &dyn Fn(&DeviceBuffer, &mut Vec<Step>)| -> Vec<f32> {
        let o = gpu.storage((m * n) as u64);
        let mut s2 = steps.clone();
        extra(&o, &mut s2);
        gpu.submit(&[], &s2);
        gpu.read(&o, m * n)
    };

    let (pb, sb) = (upload_u32(&gpu, &wq), upload(&gpu, &sw));
    let base = run(&|o, s2| {
        s2.push(mm8_rows_off(&gpu, i8tier, &scr, &pb, &sb, o, 0, 0, m as u32, k as u32, n as u32));
    });

    // The runtime correction: base weight untouched, `s·B·(A·x)` added in fp32.
    let lo = LoraW {
        a: upload(&gpu, &a),
        bt: {
            let mut bt = vec![0f32; r * n];
            for o in 0..n {
                for kk in 0..r {
                    bt[kk * n + o] = b0[o * r + kk] * s;
                }
            }
            upload(&gpu, &bt)
        },
        r: r as u32,
    };
    let tscr = gpu.storage((m * r) as u64);
    let runtime = run(&|o, s2| {
        s2.push(mm8_rows_off(&gpu, i8tier, &scr, &pb, &sb, o, 0, 0, m as u32, k as u32, n as u32));
        s2.extend(lora_rows_off(&gpu, f32tier, idx(&gpu, "lora_delta"), &lo, &tscr, &xb, o, 0, 0, m as u32, k as u32, n as u32));
    });

    // The folded deployment, for comparison.
    let (pbf, sbf) = (upload_u32(&gpu, &wq_folded), upload(&gpu, &sw_folded));
    let folded_out = run(&|o, s2| {
        s2.push(mm8_rows_off(&gpu, i8tier, &scr, &pbf, &sbf, o, 0, 0, m as u32, k as u32, n as u32));
    });

    // The oracle. `A·x` then `B` against it, in f64, from the same numbers the
    // device was given - never by calling the code under test.
    let want = matmul(&x, &delta, m, k, n);
    let got_runtime: Vec<f32> = runtime.iter().zip(&base).map(|(a, b)| a - b).collect();
    let got_folded: Vec<f32> = folded_out.iter().zip(&base).map(|(a, b)| a - b).collect();
    let e_runtime = rel_l2(&got_runtime, &want);
    let e_folded = rel_l2(&got_folded, &want);
    eprintln!("adapter delta delivered: runtime rel_l2 {e_runtime:.2e}, folded rel_l2 {e_folded:.2e}");

    // fp32 GEMM accumulation order against an f64 oracle over k=256 - the same
    // budget every fp32 matmul test in this workspace carries.
    assert!(e_runtime < 1e-4, "the runtime correction must deliver the adapter's delta exactly: rel_l2 {e_runtime:.2e}");
    // And the deployment it replaces must NOT be mistaken for an equivalent
    // one. This bound is deliberately a floor, not a ceiling: it says folding
    // into an int8 grid loses the adapter, which is the premise the runtime
    // path exists for.
    assert!(e_folded > 0.5, "folding into the int8 grid is supposed to LOSE most of the delta; it delivered rel_l2 {e_folded:.2e}");
}
