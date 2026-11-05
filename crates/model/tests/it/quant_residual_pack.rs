// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Device gate for `quant_residual_pack.wgsl`, the second level of a
//! two-level int8 activation.
//!
//! Swedish Embedded AB implements solutions for running large quantized
//! models within a fixed GPU memory budget for its clients. If your team needs
//! expertise in low-precision inference kernels then you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The kernel's claim is that it is the unfused composition - first-level
//! rounding error written out in f32, then `max_abs_row` + `quant_pack` over
//! it - without ever storing that error. The residual `x - q·s` may be
//! evaluated with a fused multiply-add on the device, so it can differ from a
//! host-rounded one in the last bit of `x`; the gate therefore holds the
//! scales to the composition's within that rounding, and the two levels
//! together to reconstructing the activation within half a second-level step.

use gpu_core::testgpu::dev;

const PIPES: &[(&str, &str)] = &[
    ("quant_residual_pack", kernels::QUANT_RESIDUAL_PACK),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
];
const RESIDUAL_PACK: usize = 0;
const MAX_ABS_ROW: usize = 1;
const QUANT_PACK: usize = 2;

fn unpack(words: &[f32]) -> Vec<i8> {
    words.iter().flat_map(|w| w.to_bits().to_le_bytes()).map(|b| b as i8).collect()
}

#[test]
fn fused_residual_level_matches_the_stored_residual_composition() {
    let g = dev(PIPES);
    let (m, k) = (7usize, 160usize);
    // Irregular magnitudes per group, including an all-zero group, so scales
    // differ group to group and the 1e-8 floor is exercised.
    let xs: Vec<f32> = (0..m * k).map(|i| if (i / 32) % 9 == 4 { 0.0 } else { ((i as f32 * 0.7311).sin() * (1.0 + (i / 32) as f32 * 0.37)).powi(3) }).collect();
    let (mu, ku) = (m as u32, k as u32);
    let x = g.storage_init("x", &xs);
    let sx = g.storage((m * k / 32) as u64);
    let xq = g.storage((m * k / 4) as u64);
    g.submit(&[&sx, &xq], &[g.step(MAX_ABS_ROW, &[&x, &sx], &[mu, ku], mu), g.step(QUANT_PACK, &[&x, &sx, &xq], &[mu, ku], mu * ku / 4)]);
    let sx_h = g.read(&sx, m * k / 32);

    // The unfused oracle: the residual written out on the host with
    // quant_pack's own rounding, then quantized by the existing kernels.
    let residual: Vec<f32> = xs
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let s = sx_h[i / 32];
            v - (v * (1.0 / s)).round().clamp(-127.0, 127.0) * s
        })
        .collect();
    let r = g.storage_init("r", &residual);
    let want_rs = g.storage((m * k / 32) as u64);
    let want_rq = g.storage((m * k / 4) as u64);
    g.submit(&[&want_rs, &want_rq], &[g.step(MAX_ABS_ROW, &[&r, &want_rs], &[mu, ku], mu), g.step(QUANT_PACK, &[&r, &want_rs, &want_rq], &[mu, ku], mu * ku / 4)]);

    let rq = g.storage((m * k / 4) as u64);
    let rs = g.storage((m * k / 32) as u64);
    g.submit(&[&rq, &rs], &[g.step(RESIDUAL_PACK, &[&x, &sx, &rq, &rs], &[mu, ku], mu * ku / 32)]);

    let (got_rs, got_rq) = (g.read(&rs, m * k / 32), g.read(&rq, m * k / 4));
    for (gi, (&got, want)) in got_rs.iter().zip(g.read(&want_rs, m * k / 32)).enumerate() {
        assert!((got - want).abs() <= want * 1e-3, "group {gi}: scale {got} vs the composition's {want}");
    }
    let q1 = unpack(&g.read(&xq, m * k / 4));
    let q2 = unpack(&got_rq);
    for i in 0..m * k {
        let back = sx_h[i / 32] * q1[i] as f32 + got_rs[i / 32] * q2[i] as f32;
        let bound = got_rs[i / 32] * 0.5 + xs[i].abs() * 4.0 * f32::EPSILON;
        assert!((back - xs[i]).abs() <= bound, "element {i}: two levels give {back}, x is {}", xs[i]);
    }
}
