// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `gdn_chunk_fwd_train` with the fused tensor-core prefix (`GdnFastIds::tensor_core`)
//! against the portable whole-tensor path: the forward outputs, both histories and
//! every gradient of `gdn_chunk_bwd` agree to the TF32-split bound, the launch
//! replaces thirteen dispatches by one (the census says the fused kernel was
//! selected), and a caller that does not opt in, or a device without tensor cores
//! (the CPU and Vulkan backends here), keeps the portable sequence unchanged. The backward is two launches
//! (the reverse sweep and one kernel per (head, chunk) for everything else) in the tensor-core tier.
//!
//! Swedish Embedded AB implements tensor-core training kernels for linear-
//! attention language models for its clients. If your team needs expertise in
//! fusing the chunked forward of a recurrent layer without changing its
//! gradients beyond a stated bound, you can procure our services by sending an
//! email to info@swedishembedded.com.

use gpu_core::Gpu;
use model::gdn::{gdn_chunk_bwd, gdn_chunk_fwd_train, GdnBwdIds, GdnBwdScratchBufs, GdnFastIds, GdnIds, GdnScratchTrainBufs, GdnShape};

const PIPES: &[(&str, &str)] = &[
    ("bmm", kernels::BMM),
    ("bmm_acc", kernels::BMM_ACC),
    ("gdn_chunk_cumsum_step", kernels::GDN_CHUNK_CUMSUM_STEP),
    ("gdn_decay_mask", kernels::GDN_DECAY_MASK),
    ("gdn_mask_strict_lower", kernels::GDN_MASK_STRICT_LOWER),
    ("gdn_ut_step", kernels::GDN_UT_STEP),
    ("gdn_add_identity", kernels::GDN_ADD_IDENTITY),
    ("scale_row", kernels::SCALE_ROW),
    ("gdn_row_scale_off", kernels::GDN_ROW_SCALE_OFF),
    ("gdn_decay_scale", kernels::GDN_DECAY_SCALE),
    ("gdn_state_decay", kernels::GDN_STATE_DECAY),
    ("exp", kernels::EXP),
    ("sub", kernels::SUB),
    ("mul", kernels::MUL),
    ("region_copy", kernels::REGION_COPY),
    ("gdn_ut_fwd", kernels::GDN_UT_FWD),
    ("bmm_tiled", kernels::BMM_TILED),
    ("splice_add", kernels::SPLICE_ADD),
    ("row_dot", kernels::ROW_DOT),
    ("scale_add", kernels::SCALE_ADD),
    ("gdn_chunk_reverse_cumsum_step", kernels::GDN_CHUNK_REVERSE_CUMSUM_STEP),
    ("gdn_ut_bwd_dattn0", kernels::GDN_UT_BWD_DATTN0),
    ("gdn_ut_bwd_dtmat", kernels::GDN_UT_BWD_DTMAT),
    ("gdn_mask_strict_lower_bwd", kernels::GDN_MASK_STRICT_LOWER_BWD),
    ("gdn_decay_mask_bwd", kernels::GDN_DECAY_MASK_BWD),
    ("gdn_decay_scale_bwd", kernels::GDN_DECAY_SCALE_BWD),
    ("gdn_decay_scale_bwd_last", kernels::GDN_DECAY_SCALE_BWD_LAST),
    ("gdn_state_decay_bwd_dscale", kernels::GDN_STATE_DECAY_BWD_DSCALE),
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

/// `Portable`: no fast kernels. `Fast`: the bit-identical fast kernels. `TensorCore`: those plus the fused prefix.
#[derive(Clone, Copy, PartialEq)]
enum Tier {
    Portable,
    Fast,
    TensorCore,
}

fn ids(g: &Gpu, tier: Tier) -> GdnIds {
    GdnIds {
        bmm: idx(g, "bmm"),
        bmm_acc: idx(g, "bmm_acc"),
        cumsum_step: idx(g, "gdn_chunk_cumsum_step"),
        decay_mask: idx(g, "gdn_decay_mask"),
        mask_strict_lower: idx(g, "gdn_mask_strict_lower"),
        ut_step: idx(g, "gdn_ut_step"),
        add_identity: idx(g, "gdn_add_identity"),
        row_scale: idx(g, "scale_row"),
        row_scale_off: idx(g, "gdn_row_scale_off"),
        decay_scale: idx(g, "gdn_decay_scale"),
        state_decay: idx(g, "gdn_state_decay"),
        exp: idx(g, "exp"),
        sub: idx(g, "sub"),
        mul: idx(g, "mul"),
        region_copy: idx(g, "region_copy"),
        fast: (tier != Tier::Portable).then(|| GdnFastIds { ut_fwd: idx(g, "gdn_ut_fwd"), bmm_tiled: idx(g, "bmm_tiled"), tensor_core: tier == Tier::TensorCore }),
    }
}

fn bwd_ids(g: &Gpu) -> GdnBwdIds {
    GdnBwdIds {
        splice_add: idx(g, "splice_add"),
        row_dot: idx(g, "row_dot"),
        scale_add: idx(g, "scale_add"),
        reverse_cumsum_step: idx(g, "gdn_chunk_reverse_cumsum_step"),
        ut_bwd_dattn0: idx(g, "gdn_ut_bwd_dattn0"),
        ut_bwd_dtmat: idx(g, "gdn_ut_bwd_dtmat"),
        mask_strict_lower_bwd: idx(g, "gdn_mask_strict_lower_bwd"),
        decay_mask_bwd: idx(g, "gdn_decay_mask_bwd"),
        decay_scale_bwd: idx(g, "gdn_decay_scale_bwd"),
        decay_scale_bwd_last: idx(g, "gdn_decay_scale_bwd_last"),
        state_decay_bwd_dscale: idx(g, "gdn_state_decay_bwd_dscale"),
    }
}

fn rand(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((*seed >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// What one training forward and backward produced.
struct Run {
    out: Vec<f32>,
    final_state: Vec<f32>,
    state_history: Vec<f32>,
    decay_hist: Vec<f32>,
    grads: Vec<(&'static str, Vec<f32>)>,
    steps: usize,
    bwd_steps: usize,
}

fn run(g: &Gpu, tier: Tier, shape: &GdnShape) -> Run {
    let (bhc, bh) = (shape.bhc() as usize, shape.bh() as usize);
    let (cn, dk, dv) = (shape.chunk as usize, shape.dk as usize, shape.dv as usize);
    let n_chunks = shape.n_chunks() as usize;
    let mut s = 99u64;
    let mut keys = rand(&mut s, bhc * cn * dk, 1.0);
    for row in keys.chunks_mut(dk) {
        let n = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        row.iter_mut().for_each(|x| *x /= n);
    }
    let query = g.storage_init("q", &rand(&mut s, bhc * cn * dk, 1.0));
    let key = g.storage_init("k", &keys);
    let value = g.storage_init("v", &rand(&mut s, bhc * cn * dv, 1.0));
    let raw_g = g.storage_init("g", &rand(&mut s, bhc * cn, 0.2).iter().map(|x| -x.abs()).collect::<Vec<_>>());
    let beta = g.storage_init("beta", &rand(&mut s, bhc * cn, 0.5).iter().map(|x| x + 0.5).collect::<Vec<_>>());
    let initial_state = g.storage_init("s0", &rand(&mut s, bh * dk * dv, 0.1));
    let d_out = g.storage_init("d_out", &rand(&mut s, bhc * cn * dv, 1.0));
    let d_final = g.storage_init("d_final", &rand(&mut s, bh * dk * dv, 0.1));
    let scratch = GdnScratchTrainBufs::new(g, shape);
    let out = g.storage((bhc * cn * dv) as u64);
    let final_state = g.storage((bh * dk * dv) as u64);
    let (ids, bids) = (ids(g, tier), bwd_ids(g));
    let steps = gdn_chunk_fwd_train(g, &ids, &bids, shape, &query, &key, &value, &raw_g, &beta, &initial_state, &scratch.as_ref(), &out, &final_state);
    let n = steps.len();
    g.submit(&scratch.clears(), &steps);

    let bwd = GdnBwdScratchBufs::new_for(g, &ids, shape);
    let (d_query, d_key) = (g.storage((bhc * cn * dk) as u64), g.storage((bhc * cn * dk) as u64));
    let (d_value, d_raw_g, d_beta) = (g.storage((bhc * cn * dv) as u64), g.storage((bhc * cn) as u64), g.storage((bhc * cn) as u64));
    let d_init = g.storage((bh * dk * dv) as u64);
    let bsteps = gdn_chunk_bwd(g, &ids, &bids, shape, &query, &key, &value, &beta, &scratch.as_ref(), &d_out, &d_final, &bwd.as_ref(), &d_query, &d_key, &d_value, &d_raw_g, &d_beta, &d_init);
    let mut clears = bwd.clears();
    clears.extend([&d_query, &d_key, &d_beta]);
    let bwd_steps = bsteps.len();
    g.submit(&clears, &bsteps);

    let r = scratch.as_ref();
    Run {
        out: g.read(&out, bhc * cn * dv),
        final_state: g.read(&final_state, bh * dk * dv),
        state_history: g.read(r.state_history, (n_chunks + 1) * bh * dk * dv),
        decay_hist: g.read(r.decay_scale_hist, bhc * cn),
        grads: vec![
            ("d_query", g.read(&d_query, bhc * cn * dk)),
            ("d_key", g.read(&d_key, bhc * cn * dk)),
            ("d_value", g.read(&d_value, bhc * cn * dv)),
            ("d_raw_g", g.read(&d_raw_g, bhc * cn)),
            ("d_beta", g.read(&d_beta, bhc * cn)),
            ("d_initial_state", g.read(&d_init, bh * dk * dv)),
        ],
        steps: n,
        bwd_steps,
    }
}

/// Largest absolute difference relative to the larger of 1 and the reference's own magnitude.
fn rel_diff(a: &[f32], reference: &[f32]) -> f64 {
    let scale = reference.iter().fold(1.0f64, |m, v| m.max(v.abs() as f64));
    a.iter().zip(reference).fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs() / scale))
}

#[test]
fn the_tensor_core_prefix_matches_the_portable_path_in_value_and_gradient_and_is_one_launch() {
    let Ok(g) = Gpu::try_new_cuda(PIPES) else {
        eprintln!("gdn_chunk_fwd_train_tc: no CUDA device on this box - nothing to run");
        return;
    };
    if !model::gdn::use_fast_kernels(&g) {
        eprintln!("gdn_chunk_fwd_train_tc: this device does not take the fast kernels - nothing to run");
        return;
    }
    for (h, t, chunk) in [(6u32, 256u32, 64u32), (3, 96, 32), (2, 40, 8), (2, 40, 5), (3, 4, 1)] {
        let shape = GdnShape { b: 1, h, t, dk: 128, dv: 128, chunk };
        let (reference, fast, tc) = (run(&g, Tier::Portable, &shape), run(&g, Tier::Fast, &shape), run(&g, Tier::TensorCore, &shape));
        assert!(reference.out.iter().all(|x| x.is_finite()), "h={h} t={t} chunk={chunk}: the reference produced non-finite values");
        let tol = 2e-5;
        let check = |what: &str, a: &[f32], b: &[f32]| {
            let e = rel_diff(a, b);
            println!("h={h} t={t} chunk={chunk} {what}: {e:.2e}");
            assert!(e <= tol, "h={h} t={t} chunk={chunk}: {what} differs by {e:.3e} (bound {tol:.0e})");
        };
        check("out", &tc.out, &reference.out);
        check("final_state", &tc.final_state, &reference.final_state);
        check("state_history", &tc.state_history, &reference.state_history);
        check("decay_scale_hist", &tc.decay_hist, &reference.decay_hist);
        for ((name, got), (_, want)) in tc.grads.iter().zip(&reference.grads) {
            check(name, got, want);
        }
        // Selected, from the launch census: the prefix (7 row/mask steps, the UT solve, 3 + 2 products and
        // scalings, the intra scores = 14 with the fast UT kernel) is one step.
        assert_eq!(tc.bwd_steps, 2, "h={h} t={t} chunk={chunk}: the backward should be the sweep and one post kernel ({} steps; the portable one is {})", tc.bwd_steps, fast.bwd_steps);
        assert_eq!(fast.steps - tc.steps, 13, "h={h} t={t} chunk={chunk}: the fused prefix was not selected ({} -> {} steps)", fast.steps, tc.steps);
    }
}
