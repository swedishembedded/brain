// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in verifying several
// tokens per sequence through a Gated DeltaNet layer without losing the ability
// to undo a rejection, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The native `gdn_verify_pool` / `gdn_commit_pool` kernels
//! (`cu/gdn_decode.cu`) against plain single-step decoding.
//!
//! A verify launch carries several consecutive rows per sequence of a batch,
//! each sequence from its own pool row. The claim, byte for byte: the layer
//! output of every row, the pool state after each sequence's LAST row, and the
//! state and conv window after every earlier row (the snapshots a rejection
//! restores) are exactly what repeated single decode steps produce - and
//! committing a snapshot puts that state back over the pool row, leaving every
//! other row of the pool alone.

use data::rng::{random_seed, Lcg};
use model::gdn::GdnShape;
use model::gdn_mixer::{gdn_mixer_decode_fused, GdnMixerShape, GdnMixerWeights, GdnPoolRows, GdnStream};
use model::gdn_verify::{gdn_commit_pooled, gdn_mixer_verify_pooled_fused, GdnRowSnapshots, VerifyLayout};

fn rnd(r: &mut Lcg, n: usize, s: f32) -> Vec<f32> {
    (0..n).map(|_| r.scaled(s)).collect()
}

fn bits(v: Vec<f32>) -> Vec<u32> {
    v.iter().map(|f| f.to_bits()).collect()
}

fn is_cuda(g: &gpu_core::Gpu) -> bool {
    g.kind() == "cuda" && g.caps().arch.compute_capability.is_some() && gpu_core::native_kernels_enabled()
}

fn u32s(g: &gpu_core::Gpu, v: &[u32]) -> gpu_core::DeviceBuffer {
    let b = g.storage(v.len().max(1) as u64);
    g.write(&b, v);
    b
}

/// One verify case: key heads, group, rows per sequence, pool rows, pool row of each sequence.
type GdnCase<'a> = (u32, u32, &'a [u32], u32, &'a [u32]);

#[test]
fn verify_rows_and_snapshots_equal_repeated_single_steps_and_commit_restores_them() {
    let g = gpu_core::testgpu::dev(&[]);
    if !is_cuda(&g) {
        brain_testutil::skip_unavailable("native fused kernels need a CUDA device");
        return;
    }
    let (dk, dv, kw) = (128u32, 128u32, 4u32);
    // (key heads, group, rows per sequence, pool rows, pool row of each sequence)
    let cases: [GdnCase; 4] =
        [(1, 1, &[1], 2, &[1]), (2, 3, &[3, 1, 5], 5, &[3, 0, 2]), (4, 2, &[8, 2, 1, 4], 6, &[5, 1, 0, 3]), (16, 3, &[2, 2], 3, &[2, 0])];
    for (case, &(nkh, group, counts, pool_rows, slots)) in cases.iter().enumerate() {
        let nvh = nkh * group;
        let b = counts.len() as u32;
        let n: u32 = counts.iter().sum();
        let shape = GdnMixerShape { gdn: GdnShape { b: 1, h: nvh, t: 1, dk, dv, chunk: 1 }, nkh, conv_kernel: kw, rms_eps: 1e-6 };
        let (value_dim, conv_dim) = (shape.value_dim() as usize, shape.conv_dim() as usize);
        let state_len = (nvh * dk * dv) as usize;
        let hist_len = conv_dim * (kw as usize - 1);
        let ctx = format!("case {case}: nkh={nkh} group={group} counts={counts:?}");

        let mut r = Lcg::new(random_seed());
        let conv_w = g.storage_init("conv_w", &rnd(&mut r, conv_dim * kw as usize, 0.5));
        let a_log = g.storage_init("a_log", &rnd(&mut r, nvh as usize, 0.5));
        let dt_bias = g.storage_init("dt_bias", &rnd(&mut r, nvh as usize, 0.5));
        let norm_w = g.storage_init("norm_w", &rnd(&mut r, dv as usize, 0.5));
        let ones = g.storage_init("ones", &vec![1.0f32; dk as usize]);
        let w = GdnMixerWeights { conv1d_weight: &conv_w, a_log: &a_log, dt_bias: &dt_bias, norm_weight: &norm_w, ones_khd: &ones };

        let nn = n as usize;
        let mixed = rnd(&mut r, nn * conv_dim, 1.0);
        let bp = rnd(&mut r, nn * nvh as usize, 1.0);
        let ap = rnd(&mut r, nn * nvh as usize, 1.0);
        let z = rnd(&mut r, nn * value_dim, 1.0);
        let state0 = rnd(&mut r, pool_rows as usize * state_len, 0.3);
        let hist0 = rnd(&mut r, pool_rows as usize * hist_len, 0.3);

        // Reference: every sequence stepped one row at a time on a private copy
        // of its pool row, recording the state after each row.
        let mut ref_gated = vec![0.0f32; nn * value_dim];
        let mut ref_snap_state = vec![0.0f32; nn * state_len];
        let mut ref_snap_hist = vec![0.0f32; nn * hist_len];
        let mut ref_final = (state0.clone(), hist0.clone());
        let mut off = 0usize;
        for (si, &cnt) in counts.iter().enumerate() {
            let slot = slots[si] as usize;
            let st = g.storage_init("state", &state0[slot * state_len..(slot + 1) * state_len]);
            let hi = g.storage_init("hist", &hist0[slot * hist_len..(slot + 1) * hist_len]);
            for row in 0..cnt as usize {
                let t = off + row;
                let slice = |v: &[f32], width: usize, name: &str| g.storage_init(name, &v[t * width..(t + 1) * width]);
                let gated = gdn_mixer_decode_fused(
                    &g,
                    &shape,
                    &w,
                    &slice(&mixed, conv_dim, "mixed"),
                    &slice(&bp, nvh as usize, "bp"),
                    &slice(&ap, nvh as usize, "ap"),
                    &slice(&z, value_dim, "z"),
                    &GdnStream { state: &st, hist: &hi },
                )
                .expect("the fused single-step kernel was declined on a CUDA device");
                ref_gated[t * value_dim..(t + 1) * value_dim].copy_from_slice(&g.read(&gated, value_dim));
                if row + 1 < cnt as usize {
                    ref_snap_state[t * state_len..(t + 1) * state_len].copy_from_slice(&g.read(&st, state_len));
                    ref_snap_hist[t * hist_len..(t + 1) * hist_len].copy_from_slice(&g.read(&hi, hist_len));
                }
            }
            ref_final.0[slot * state_len..(slot + 1) * state_len].copy_from_slice(&g.read(&st, state_len));
            ref_final.1[slot * hist_len..(slot + 1) * hist_len].copy_from_slice(&g.read(&hi, hist_len));
            off += cnt as usize;
        }

        // The batched verify launch.
        let (pst, phi) = (g.storage_init("pool_state", &state0), g.storage_init("pool_hist", &hist0));
        let rows_buf = u32s(&g, slots);
        let offs: Vec<u32> = counts.iter().scan(0u32, |acc, &c| { let o = *acc; *acc += c; Some(o) }).collect();
        let (counts_buf, offs_buf) = (u32s(&g, counts), u32s(&g, &offs));
        let pool = GdnPoolRows { state: &pst, hist: &phi, rows: &rows_buf, gather: 0, scatter: 0, fuse: true, bf16: false };
        let snaps = GdnRowSnapshots::new(&g, &shape, n);
        let layout = VerifyLayout { rows: &rows_buf, counts: &counts_buf, offs: &offs_buf, b };
        let gated = gdn_mixer_verify_pooled_fused(
            &g,
            &shape,
            &w,
            &g.storage_init("mixed", &mixed),
            &g.storage_init("bp", &bp),
            &g.storage_init("ap", &ap),
            &g.storage_init("z", &z),
            &pool,
            &layout,
            n,
            &snaps,
        )
        .expect("the verify kernel was declined on a CUDA device");
        g.poll_wait();

        assert_eq!(bits(g.read(&gated, nn * value_dim)), bits(ref_gated), "layer outputs differ: {ctx}");
        assert_eq!(bits(g.read(&pst, pool_rows as usize * state_len)), bits(ref_final.0.clone()), "pool state after the last row differs: {ctx}");
        assert_eq!(bits(g.read(&phi, pool_rows as usize * hist_len)), bits(ref_final.1.clone()), "pool window after the last row differs: {ctx}");
        let got_snap_state = g.read(&snaps.state, nn * state_len);
        let got_snap_hist = g.read(&snaps.hist, nn * hist_len);
        for (si, &cnt) in counts.iter().enumerate() {
            for row in 0..cnt as usize - 1 {
                let t = offs[si] as usize + row;
                assert_eq!(bits(got_snap_state[t * state_len..(t + 1) * state_len].to_vec()), bits(ref_snap_state[t * state_len..(t + 1) * state_len].to_vec()), "snapshot state of sequence {si} row {row}: {ctx}");
                assert_eq!(bits(got_snap_hist[t * hist_len..(t + 1) * hist_len].to_vec()), bits(ref_snap_hist[t * hist_len..(t + 1) * hist_len].to_vec()), "snapshot window of sequence {si} row {row}: {ctx}");
            }
        }

        // Commit: sequence 0 restores its first snapshot (when it has one), the
        // others are left as the launch left them.
        let mut sel = vec![u32::MAX; counts.len()];
        let mut want_state = ref_final.0.clone();
        let mut want_hist = ref_final.1.clone();
        if counts[0] > 1 {
            sel[0] = offs[0];
            let slot = slots[0] as usize;
            want_state[slot * state_len..(slot + 1) * state_len].copy_from_slice(&ref_snap_state[offs[0] as usize * state_len..(offs[0] as usize + 1) * state_len]);
            want_hist[slot * hist_len..(slot + 1) * hist_len].copy_from_slice(&ref_snap_hist[offs[0] as usize * hist_len..(offs[0] as usize + 1) * hist_len]);
        }
        let sel_buf = u32s(&g, &sel);
        let step = gdn_commit_pooled(&g, &shape, &pst, &phi, &snaps, &rows_buf, &sel_buf, b).expect("the commit kernel was declined on a CUDA device");
        g.submit(&[], &[step]);
        g.poll_wait();
        assert_eq!(bits(g.read(&pst, pool_rows as usize * state_len)), bits(want_state), "pool state after commit differs: {ctx}");
        assert_eq!(bits(g.read(&phi, pool_rows as usize * hist_len)), bits(want_hist), "pool window after commit differs: {ctx}");
    }
}
