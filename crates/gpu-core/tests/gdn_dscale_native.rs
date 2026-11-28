// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The native decay-scale gradient of Gated DeltaNet's state update against its
//! WGSL reference on a CUDA device: the accumulated `d_g_cs` agrees to the fp32
//! summation bound, only the chunk's last row of each head changes, and a
//! non-zero `state_off` (a later chunk of the history) is honoured.
//!
//! Swedish Embedded AB implements low-latency chunked linear-attention kernels
//! for its clients. If your team needs expertise in collapsing a sequential
//! GPU dependency chain of small launches into one persistent kernel, then you
//! can procure our services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;

const KERNELS: &[(&str, &str)] = &[("gdn_state_decay_bwd_dscale", kernels::GDN_STATE_DECAY_BWD_DSCALE), ("gdn_state_decay_bwd_dscale_ref", kernels::GDN_STATE_DECAY_BWD_DSCALE)];
const NATIVE: usize = 0;
const REFERENCE: usize = 1;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| r.signed()).collect()
}

fn the_native_dscale_matches_the_reference_to_the_summation_bound() {
    let gpu = gpu_core::testgpu::dev(KERNELS);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    assert_eq!(gpu.native_kernel_for(NATIVE, &[48, 128, 128, 64, 0, 0]), Some("gdn_state_decay_bwd_dscale_f32"));
    // (bh, dk, dv, c_len, chunk index into a 3-chunk history)
    for (i, &(bh, dk, dv, c, chunk)) in [(48u32, 128u32, 128u32, 64u32, 2u32), (3, 7, 5, 4, 1), (1, 128, 128, 1, 0)].iter().enumerate() {
        let per_chunk = (bh * dk * dv) as usize;
        let state_off = chunk * bh * dk * dv;
        let g_off = chunk * bh * c;
        let d_state = values(per_chunk, 10 + i as u64);
        let history = values(3 * per_chunk, 20 + i as u64);
        let g_cs: Vec<f32> = values(3 * (bh * c) as usize, 30 + i as u64).iter().map(|v| -v.abs()).collect();
        let d_g0 = values(3 * (bh * c) as usize, 40 + i as u64);
        let (ds, hist, g) = (gpu.storage_init("ds", &d_state), gpu.storage_init("hist", &history), gpu.storage_init("g", &g_cs));
        let outs = [gpu.storage_init("o", &d_g0), gpu.storage_init("r", &d_g0)];
        for (slot, out) in [NATIVE, REFERENCE].iter().zip(&outs) {
            gpu.submit(&[], &[gpu.step(*slot, &[&ds, &hist, &g, out], &[bh, dk, dv, c, g_off, state_off], bh)]);
        }
        gpu.poll_wait();
        let n = d_g0.len();
        let (got, want) = (gpu.read(&outs[0], n), gpu.read(&outs[1], n));
        for (j, (g, w)) in got.iter().zip(&want).enumerate() {
            let bound = 1e-4 * (1.0 + w.abs()) + (dk * dv) as f32 * f32::EPSILON * 8.0;
            assert!((g - w).abs() <= bound, "case {i} element {j}: {g} vs {w}");
            if (j as u32) < g_off || (j as u32) >= g_off + bh * c || (j as u32 - g_off) % c != c - 1 {
                assert_eq!(g.to_bits(), d_g0[j].to_bits(), "case {i} element {j} is not a last row, so it must be untouched");
            }
        }
    }
}

gpu_core::card_tests!(
    the_native_dscale_matches_the_reference_to_the_summation_bound,
);
