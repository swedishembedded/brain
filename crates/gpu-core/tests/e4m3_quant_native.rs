// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The gate for the E4M3 half of every activation quantiser (`cu/e4m3_quant_common.cuh`, used by `quant_e4m3_128`,
//! `silu_mul_quant_e4m3_128`, `gate_mul_quant_e4m3_128`, `add_rms_quant_e4m3` and `gdn_gated_norm`): four elements that share a
//! scale become four bytes, `e4m3(RN(v / scale))`.
//!
//! Swedish Embedded AB implements solutions for FP8 serving of LLMs on GPUs for its clients. If your team needs expertise in
//! cutting the instruction and memory cost of activation quantisation then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The quotient is computed from one reciprocal a scale (three fused operations an element) instead of the compiler's division
//! (a range check and a slow-path call as well), so the claim to hold is that the bytes are exactly the IEEE quotient's: the
//! reference here is the host's own f32 division and a round-to-nearest-even, saturating E4M3 encoder, over blocks whose
//! scales run from subnormal to infinite and whose elements run from zero (of either sign) to the block's maximum.

use gpu_core::Gpu;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

/// The finite E4M3 magnitudes by code 0..=126 (code 127 is the NaN).
fn magnitudes() -> Vec<f64> {
    (0u32..127)
        .map(|c| {
            let (e, m) = (c >> 3, f64::from(c & 7));
            if e == 0 { m / 8.0 * 2f64.powi(-6) } else { (1.0 + m / 8.0) * 2f64.powi(e as i32 - 7) }
        })
        .collect()
}

/// `cvt.rn.satfinite.e4m3.f32`: nearest even, saturating at 448, NaN stays a NaN, the sign of a zero is kept.
fn encode(x: f32, table: &[f64]) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    if x.is_nan() {
        return 0x7f | sign;
    }
    let a = f64::from(x.abs());
    if a >= 448.0 {
        return 0x7e | sign;
    }
    let hi = table.partition_point(|&m| m < a);
    // `hi` is the first magnitude >= a; the answer is it or its predecessor, a tie going to the even code.
    let code = match hi {
        0 => 0,
        _ => {
            let (lo_d, hi_d) = (a - table[hi - 1], table[hi] - a);
            if lo_d < hi_d || (lo_d == hi_d && (hi - 1) % 2 == 0) { hi - 1 } else { hi }
        }
    };
    code as u8 | sign
}

fn block_values(seed: u64, decade: i32, floor: i32) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..128)
        .map(|i| {
            let u = (r.next_u32() % 2_000_001) as f32 / 1_000_000.0 - 1.0;
            // Most elements sit well below the block's maximum, down to `floor` decades under it, some are exact zeros of
            // either sign.
            let under = -((r.next_u32() % (floor as u32 + 1)) as i32);
            match i % 17 {
                0 => 0.0,
                1 => -0.0,
                _ => u * 10f32.powi(decade + under),
            }
        })
        .collect()
}

#[test]
fn the_reciprocal_quotient_makes_the_bytes_the_ieee_quotient_makes() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    let table = magnitudes();
    // Scales across the fast range and past both ends of it, a block of subnormals, and a block holding an infinity.
    let mut x: Vec<f32> = Vec::new();
    for (seed, decade) in (-45..=38).enumerate() {
        for rep in 0..6u64 {
            x.extend(block_values(1000 * seed as u64 + rep, decade, [0, 3, 8, 30][rep as usize % 4]));
        }
    }
    let mut inf_block = block_values(7, 0, 3);
    inf_block[5] = f32::INFINITY;
    x.extend(inf_block);
    let rows = (x.len() / 128) as u32;
    let xb = gpu.storage_init("x", &x);
    let mut steps = Vec::new();
    let x_op = gpu_core::provider::library::Operand { buf: &xb, range: (0, x.len() as u64) };
    let (q, s) = gpu_core::provider::library::quantize_e4m3_rows(&gpu, &mut steps, &x_op, rows, 128).expect("quant_e4m3_128 offered");
    gpu.submit(&[], &steps);
    gpu.poll_wait();
    let (words, scales) = (gpu.read(&q, x.len() / 4), gpu.read(&s, rows as usize));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_bits().to_le_bytes()).collect();

    for (b, block) in x.chunks(128).enumerate() {
        let amax = block.iter().fold(0f32, |m, v| if v.abs() > m { v.abs() } else { m });
        let scale = amax / 448.0;
        assert_eq!(scales[b].to_bits(), scale.to_bits(), "block {b}: scale");
        for (i, v) in block.iter().enumerate() {
            // An all-zero block (scale 0) is written as zero bytes, every other element as the encoded IEEE quotient.
            let want = if scale > 0.0 { encode(v / scale, &table) } else { 0 };
            assert_eq!(bytes[b * 128 + i], want, "block {b} element {i}: {v:e} / {scale:e} = {:e}", v / scale);
        }
    }
}
