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

/// Whether the card has the FP8 conversions the E4M3 kernels are built on
/// (`kernels_cuda::FP8_CVT_MIN_CC`): no earlier card is offered them.
fn has_fp8(gpu: &Gpu) -> bool {
    gpu.caps().arch.compute_capability.is_some_and(|cc| cc >= kernels_cuda::FP8_CVT_MIN_CC)
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
    if !has_fp8(&gpu) {
        brain_testutil::skip_unavailable("this card has no FP8 conversions: the E4M3 kernels are not offered");
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

fn bf16_round(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
}

/// `gdn_gated_norm`: `RMSNorm(x) * w * silu(z)` per 128-wide head row and its E4M3 rows, against the definition in f64, for fp32 and for
/// bf16 inputs, row counts that leave a ragged last block of the kernel, and a first row past zero (rows before it are not touched).
/// The outputs start as NaN so a row the kernel skips cannot pass on stale data.
#[test]
fn the_gated_norm_matches_its_definition_on_both_input_forms() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    if !has_fp8(&gpu) {
        brain_testutil::skip_unavailable("this card has no FP8 conversions: the E4M3 kernels are not offered");
        return;
    }
    let table = magnitudes();
    let mut rng = data::rng::Lcg::new(11);
    let mut uniform = |scale: f32| (rng.next_u32() % 2_000_001) as f32 / 1_000_000.0 * scale - scale;
    for (head_rows, first, bf16) in [(37u32, 0u32, false), (37, 5, false), (131, 0, true), (131, 17, true), (16, 0, true)] {
        let n = head_rows as usize * 128;
        let round = |v: f32| if bf16 { bf16_round(v) } else { v };
        let x: Vec<f32> = (0..n).map(|_| round(uniform(3.0))).collect();
        let z: Vec<f32> = (0..n).map(|_| round(uniform(6.0))).collect();
        let w: Vec<f32> = (0..128).map(|_| uniform(2.0)).collect();
        let words = |v: &[f32]| -> Vec<f32> { v.chunks(2).map(|p| f32::from_bits((p[0].to_bits() >> 16) | (p[1].to_bits() & 0xffff0000))).collect() };
        let (xb, zb) = if bf16 { (gpu.storage_init("x", &words(&x)), gpu.storage_init("z", &words(&z))) } else { (gpu.storage_init("x", &x), gpu.storage_init("z", &z)) };
        let wb = gpu.storage_init("w", &w);
        let nan = f32::from_bits(0xffff_ffff);
        let (out, q, s) = (gpu.storage_init("out", &vec![nan; n]), gpu.storage_init("q", &vec![nan; n / 4]), gpu.storage_init("s", &vec![nan; head_rows as usize]));
        let eps = 1e-6f32;
        // Only the E4M3 form is offered over bf16 inputs; the fp32 form stores the product too.
        let step = gpu_core::provider::cuda::gdn_gated_norm_step(&gpu, &[&xb, &wb, &zb, &out, &q, &s], head_rows, first, eps, true, bf16).expect("gdn_gated_norm offered");
        gpu.submit(&[], &[step]);
        gpu.poll_wait();
        let (product, qw, sc) = (gpu.read(&out, n), gpu.read(&q, n / 4), gpu.read(&s, head_rows as usize));
        let bytes: Vec<u8> = qw.iter().flat_map(|w| w.to_bits().to_le_bytes()).collect();
        for row in 0..head_rows as usize {
            let xs = &x[row * 128..(row + 1) * 128];
            if (row as u32) < first {
                assert!(sc[row].is_nan() && bytes[row * 128..(row + 1) * 128].iter().all(|&b| b == 0xff), "row {row} before the first was written");
                continue;
            }
            let inv = 1.0 / (xs.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / 128.0 + f64::from(eps)).sqrt();
            let want: Vec<f64> = (0..128)
                .map(|i| {
                    let zi = f64::from(z[row * 128 + i]);
                    f64::from(xs[i]) * inv * f64::from(w[i]) * zi / (1.0 + (-zi).exp())
                })
                .collect();
            let amax = want.iter().fold(0f64, |m, v| m.max(v.abs()));
            let scale = f64::from(sc[row]);
            assert!((scale - amax / 448.0).abs() <= amax / 448.0 * 2e-6, "row {row}: scale {scale:e} vs {:e} (bf16 {bf16})", amax / 448.0);
            for i in 0..128 {
                if !bf16 {
                    assert!((f64::from(product[row * 128 + i]) - want[i]).abs() <= 2e-6 * amax, "row {row} col {i}: product {} vs {}", product[row * 128 + i], want[i]);
                }
                // The byte decodes to the value within one E4M3 step (2^-4 relative, 2^-9 absolute below the normals) of the scaled definition.
                let b = bytes[row * 128 + i];
                let mag = table[usize::from(b & 0x7f)] * scale;
                let got = if b & 0x80 != 0 { -mag } else { mag };
                assert!((got - want[i]).abs() <= want[i].abs() * 0.0625 + scale * 2f64.powi(-9), "row {row} col {i}: byte {b:#x} = {got:e} vs {:e}", want[i]);
            }
        }
    }
}

/// `quant_e4m3_row`: one scale a row (`amax / 448` over the whole row), the bytes the IEEE quotient encodes, an all-zero row written as zeros with scale 0;
/// rows of 128 up to the 17408 of the 27B's MLP down projection, the window of rows starting past zero left untouched below it.
#[test]
fn the_row_quantiser_scales_a_whole_row_and_encodes_the_ieee_quotient() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    if !has_fp8(&gpu) {
        brain_testutil::skip_unavailable("this card has no FP8 conversions: the E4M3 kernels are not offered");
        return;
    }
    let table = magnitudes();
    for (rows, k) in [(1u32, 128u32), (5, 640), (7, 5120), (3, 17408)] {
        let mut x: Vec<f32> = Vec::new();
        for r in 0..rows {
            x.extend(block_values(31 * u64::from(r) + u64::from(k), (r as i32) - 2, 3).into_iter().cycle().take(k as usize));
        }
        // A row of zeros, and an outlier an order of magnitude past the rest in the last row.
        for v in &mut x[..k as usize] {
            *v = if rows > 1 { 0.0 } else { *v };
        }
        let last = (rows as usize - 1) * k as usize + 3;
        x[last] *= 64.0;
        let xb = gpu.storage_init("x", &x);
        let mut steps = Vec::new();
        let x_op = gpu_core::provider::library::Operand { buf: &xb, range: (0, x.len() as u64) };
        let (q, s) = gpu_core::provider::library::quantize_e4m3_row_scaled(&gpu, &mut steps, &x_op, rows, k).expect("quant_e4m3_row offered");
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        let (words, scales) = (gpu.read(&q, x.len() / 4), gpu.read(&s, rows as usize));
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_bits().to_le_bytes()).collect();
        for (r, row) in x.chunks(k as usize).enumerate() {
            let scale = row.iter().fold(0f32, |m, v| if v.abs() > m { v.abs() } else { m }) / 448.0;
            assert_eq!(scales[r].to_bits(), scale.to_bits(), "rows={rows} k={k} row {r}: scale");
            for (i, v) in row.iter().enumerate() {
                let want = if scale > 0.0 { encode(v / scale, &table) } else { 0 };
                assert_eq!(bytes[r * k as usize + i], want, "rows={rows} k={k} row {r} element {i}: {v:e} / {scale:e}");
            }
        }
    }
}
