// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The native single-launch reverse sweep of Gated DeltaNet's across-chunk
//! recurrence against an f64 transcription of the six per-chunk products the
//! host walk dispatches: every output agrees to the fp32 summation bound for a
//! full 64-row chunk and for a short one, `d_v_new` and `d_g_cs` are
//! accumulated into (not overwritten), and only each chunk's last `d_g_cs` row
//! changes.
//!
//! Swedish Embedded AB implements low-latency chunked linear-attention training
//! kernels for its clients. If your team needs expertise in collapsing a
//! sequential GPU dependency chain of small launches into one persistent
//! kernel, then you can procure our services by sending an email to
//! info@swedishembedded.com.

use gpu_core::Gpu;

fn is_cuda(gpu: &Gpu) -> bool {
    gpu.kind() == "cuda" && gpu.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut r = data::rng::Lcg::new(seed);
    (0..n).map(|_| r.signed()).collect()
}

const D: usize = 128;

/// The sweep in f64: returns (d_decayed_k, d_v_new, d_g_cs, d_initial_state).
#[allow(clippy::too_many_arguments)]
fn reference(bh: usize, c: usize, n: usize, v_new: &[f32], dk_in: &[f32], qs: &[f32], w: &[f32], d_out: &[f32], hist: &[f32], g_cs: &[f32], d_final: &[f32], dvn0: &[f32], dg0: &[f32]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut ddk = vec![0f64; n * bh * c * D];
    let mut dvn: Vec<f64> = dvn0.iter().map(|&v| v as f64).collect();
    let mut dg: Vec<f64> = dg0.iter().map(|&v| v as f64).collect();
    let mut state: Vec<f64> = d_final.iter().map(|&v| v as f64).collect();
    for h in 0..bh {
        let s = &mut state[h * D * D..(h + 1) * D * D];
        for ci in (0..n).rev() {
            let rows = (ci * bh + h) * c;
            let hist_in = &hist[(ci * bh + h) * D * D..(ci * bh + h + 1) * D * D];
            for r in 0..c {
                for p in 0..D {
                    ddk[(rows + r) * D + p] = (0..D).map(|q| v_new[(rows + r) * D + q] as f64 * s[p * D + q]).sum();
                }
                for q in 0..D {
                    dvn[(rows + r) * D + q] += (0..D).map(|p| dk_in[(rows + r) * D + p] as f64 * s[p * D + q]).sum::<f64>();
                }
            }
            let decay = (g_cs[rows + c - 1] as f64).exp();
            dg[rows + c - 1] += decay * s.iter().zip(hist_in).map(|(a, &b)| a * b as f64).sum::<f64>();
            for p in 0..D {
                for q in 0..D {
                    let upd: f64 = (0..c).map(|i| qs[(rows + i) * D + p] as f64 * d_out[(rows + i) * D + q] as f64 - w[(rows + i) * D + p] as f64 * dvn[(rows + i) * D + q]).sum();
                    s[p * D + q] = s[p * D + q] * decay + upd;
                }
            }
        }
    }
    (ddk, dvn, dg, state)
}

fn the_native_sweep_matches_the_f64_reference_to_the_summation_bound() {
    let gpu = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&gpu) {
        brain_testutil::skip_unavailable("not a CUDA device");
        return;
    }
    // (heads, chunk rows, chunks): a full chunk over several heads, and a short chunk.
    for (i, &(bh, c, n)) in [(3usize, 64usize, 3usize), (2, 5, 2)].iter().enumerate() {
        let s = 100 * i as u64;
        let rows = n * bh * c;
        let v_new = values(rows * D, s + 1);
        let dk_in = values(rows * D, s + 2);
        let qs = values(rows * D, s + 3);
        let w: Vec<f32> = values(rows * D, s + 4).iter().map(|v| v * 0.1).collect();
        let d_out = values(rows * D, s + 5);
        let hist = values(n * bh * D * D, s + 6);
        let g_cs: Vec<f32> = values(rows, s + 7).iter().map(|v| -v.abs()).collect();
        let d_final = values(bh * D * D, s + 8);
        let dvn0 = values(rows * D, s + 9);
        let dg0 = values(rows, s + 10);
        let (want_ddk, want_dvn, want_dg, want_state) = reference(bh, c, n, &v_new, &dk_in, &qs, &w, &d_out, &hist, &g_cs, &d_final, &dvn0, &dg0);

        let ins = [&v_new, &dk_in, &qs, &w, &d_out, &hist, &g_cs, &d_final].map(|v| gpu.storage_init("in", v));
        let (ddk, dvn, dg) = (gpu.storage((rows * D) as u64), gpu.storage_init("dvn", &dvn0), gpu.storage_init("dg", &dg0));
        let dinit = gpu.storage((bh * D * D) as u64);
        let bufs = [&ins[0], &ins[1], &ins[2], &ins[3], &ins[4], &ins[5], &ins[6], &ins[7], &ddk, &dvn, &dg, &dinit];
        let step = gpu_core::provider::cuda::gdn_bwd_sweep_step(&gpu, &bufs, bh as u32, c as u32, D as u32, D as u32, n as u32).expect("the native sweep must be offered on a CUDA device");
        gpu.submit(&[], &[step]);
        gpu.poll_wait();

        let check = |name: &str, got: Vec<f32>, want: &[f64]| {
            for (j, (g, w)) in got.iter().zip(want).enumerate() {
                let bound = 2e-3 * (1.0 + w.abs());
                assert!((*g as f64 - w).abs() <= bound, "case {i} {name}[{j}]: {g} vs {w}");
            }
        };
        check("d_decayed_k", gpu.read(&ddk, rows * D), &want_ddk);
        check("d_v_new", gpu.read(&dvn, rows * D), &want_dvn);
        check("d_initial_state", gpu.read(&dinit, bh * D * D), &want_state);
        let got_dg = gpu.read(&dg, rows);
        for (j, (g, w)) in got_dg.iter().zip(&want_dg).enumerate() {
            let bound = 2e-3 * (1.0 + w.abs());
            assert!((*g as f64 - w).abs() <= bound, "case {i} d_g_cs[{j}]: {g} vs {w}");
            if j % c != c - 1 {
                assert_eq!(g.to_bits(), dg0[j].to_bits(), "case {i} d_g_cs[{j}] is not a last row, so it must be untouched");
            }
        }
    }
}

gpu_core::card_tests!(
    the_native_sweep_matches_the_f64_reference_to_the_summation_bound,
);
