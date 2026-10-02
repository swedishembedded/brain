// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `gdn_ut_fwd` - the whole Gated DeltaNet UT-transform in ONE dispatch -
//! must reproduce the `gdn_ut_step` row loop plus `gdn_add_identity` it
//! replaces **bit for bit**: the same ascending-`k` reduction per element, so
//! swapping 64 launches for one changes how long the transform takes and
//! nothing it computes.
//!
//! Swedish Embedded AB implements low-latency chunked linear-attention
//! kernels for its clients. If your team needs expertise in collapsing a
//! sequential GPU dependency chain into a single launch, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! The kernel is a barrier kernel the CPU JIT cannot run, so this is a CUDA
//! test and skips on a box without one (the loop it replaces stays the
//! fallback everywhere else).

use gpu_core::Gpu;

const PIPES: &[(&str, &str)] =
    &[("gdn_ut_step", kernels::GDN_UT_STEP), ("gdn_add_identity", kernels::GDN_ADD_IDENTITY), ("gdn_ut_fwd", kernels::GDN_UT_FWD)];

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

/// A strictly-lower `[bhc, c, c]` `attn0` with entries scaled by `amp`.
fn attn0(bhc: usize, c: usize, amp: f32, correlated: bool, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut a = vec![0f32; bhc * c * c];
    for r in 0..bhc {
        for i in 0..c {
            for j in 0..i {
                a[r * c * c + i * c + j] = if correlated { -amp } else { amp * lcg(&mut s) };
            }
        }
    }
    a
}

fn reference(g: &Gpu, a: &[f32], bhc: u32, c: u32) -> Vec<f32> {
    let attn = g.storage_init("attn0", a);
    let t = g.storage((bhc * c * c) as u64);
    let step = |name: &str, bufs: &[&gpu_core::DeviceBuffer], params: &[u32], threads: u32| g.step(g.kernel_index(name).unwrap(), bufs, params, threads);
    let mut steps = Vec::new();
    for i in 1..c {
        steps.push(step("gdn_ut_step", &[&attn, &t], &[bhc, c, i], bhc * i));
    }
    steps.push(step("gdn_add_identity", &[&t], &[bhc, c], bhc * c));
    g.submit(&[&t], &steps);
    g.read(&t, (bhc * c * c) as usize)
}

fn fused(g: &Gpu, a: &[f32], bhc: u32, c: u32) -> Vec<f32> {
    let attn = g.storage_init("attn0", a);
    // Deliberately NOT cleared: the kernel owns the whole output.
    let t = g.storage_init("t_mat", &vec![f32::NAN; (bhc * c * c) as usize]);
    let step = g.dispatch(g.kernel_index("gdn_ut_fwd").unwrap(), &[&attn, &t], &[bhc, c], gpu_core::Dispatch::Workgroups(bhc));
    g.submit(&[], &[step]);
    g.read(&t, (bhc * c * c) as usize)
}

#[test]
fn the_fused_ut_transform_is_bit_identical_to_the_row_loop() {
    let Ok(g) = Gpu::try_new_cuda(PIPES) else {
        eprintln!("gdn_ut_fwd: no CUDA device on this box - skipping");
        return;
    };
    // c = 1 (identity only), small odd sizes, and the real chunk of 64; bhc
    // coprime to every c so a row-index slip cannot cancel out.
    for (c, bhc) in [(1u32, 5u32), (2, 7), (3, 5), (16, 9), (33, 5), (64, 7), (64, 192)] {
        for (amp, correlated) in [(0.5f32, false), (0.99, true)] {
            let a = attn0(bhc as usize, c as usize, amp, correlated, 11 + c as u64);
            let want = reference(&g, &a, bhc, c);
            let got = fused(&g, &a, bhc, c);
            let bad = want.iter().zip(&got).position(|(w, x)| w.to_bits() != x.to_bits());
            assert!(bad.is_none(), "c={c} bhc={bhc} amp={amp}: first mismatch at {:?}: want {:?}, got {:?}", bad, bad.map(|i| want[i]), bad.map(|i| got[i]));
        }
    }
}
