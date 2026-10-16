// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `gdn_chunk_fwd_train` with [`GdnIds::fast`] registered computes **exactly**
//! what it computes without, histories included: the single-launch training
//! recurrence replaces eleven dispatches per chunk (nine of the walk plus the
//! two history commits) and writes the state history and the decay-scale
//! history the backward sweep reads, bit for bit as the walk's `splice_add`s
//! do.
//!
//! Swedish Embedded AB implements low-latency chunked linear-attention
//! training for its clients. If your team needs expertise in cutting the
//! dispatch count of a recurrent layer's backward pass without changing the
//! gradients, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The fast kernels are barrier kernels the CPU JIT cannot run and
//! `model::gdn::use_fast_kernels` only turns them on for a CUDA device, so this
//! reports and returns elsewhere.

use gpu_core::Gpu;
use model::gdn::{gdn_chunk_fwd_train, GdnBwdIds, GdnFastIds, GdnIds, GdnScratchTrainBufs, GdnShape};

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

fn ids(g: &Gpu, fast: bool) -> GdnIds {
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
        fast: fast.then(|| GdnFastIds { ut_fwd: idx(g, "gdn_ut_fwd"), bmm_tiled: idx(g, "bmm_tiled") }),
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

/// What one training forward produced: `out`, `final_state`, the two histories,
/// and how many steps it took.
struct Run {
    out: Vec<f32>,
    final_state: Vec<f32>,
    state_history: Vec<f32>,
    decay_hist: Vec<f32>,
    steps: usize,
}

fn run(g: &Gpu, fast: bool, shape: &GdnShape) -> Run {
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
    let scratch = GdnScratchTrainBufs::new(g, shape);
    let out = g.storage((bhc * cn * dv) as u64);
    let final_state = g.storage((bh * dk * dv) as u64);
    let steps = gdn_chunk_fwd_train(g, &ids(g, fast), &bwd_ids(g), shape, &query, &key, &value, &raw_g, &beta, &initial_state, &scratch.as_ref(), &out, &final_state);
    let n = steps.len();
    g.submit(&scratch.clears(), &steps);
    let r = scratch.as_ref();
    Run {
        out: g.read(&out, bhc * cn * dv),
        final_state: g.read(&final_state, bh * dk * dv),
        state_history: g.read(r.state_history, (n_chunks + 1) * bh * dk * dv),
        decay_hist: g.read(r.decay_scale_hist, bhc * cn),
        steps: n,
    }
}

#[test]
fn the_fast_training_forward_changes_the_step_count_and_nothing_else() {
    let Ok(g) = Gpu::try_new_cuda(PIPES) else {
        eprintln!("gdn_chunk_fwd_train_fast: no CUDA device on this box - nothing to run");
        return;
    };
    if !model::gdn::use_fast_kernels(&g) {
        eprintln!("gdn_chunk_fwd_train_fast: this device does not take the fast kernels - nothing to run");
        return;
    }
    // Real head widths; the real chunk over several chunks and heads, then the
    // shorter chunks a ragged round can land on (down to 1).
    for (h, t, chunk) in [(6u32, 256u32, 64u32), (3, 96, 32), (2, 48, 16), (5, 12, 4), (2, 13, 1)] {
        let shape = GdnShape { b: 1, h, t, dk: 128, dv: 128, chunk };
        let (slow, fast) = (run(&g, false, &shape), run(&g, true, &shape));
        assert!(slow.out.iter().all(|x| x.is_finite()), "h={h} t={t} chunk={chunk}: the reference produced non-finite values");
        let same = |what: &str, a: &[f32], b: &[f32]| {
            assert_eq!(a.iter().zip(b).position(|(x, y)| x != y), None, "h={h} t={t} chunk={chunk}: the fast path changed {what}");
        };
        same("out", &slow.out, &fast.out);
        same("final_state", &slow.final_state, &fast.final_state);
        same("state_history", &slow.state_history, &fast.state_history);
        same("decay_scale_hist", &slow.decay_hist, &fast.decay_hist);
        // The prefix swaps `chunk - 1` UT rows for one dispatch; the loop's state copy,
        // initial snapshot and 11 dispatches per chunk become one launch.
        let n_chunks = (t / chunk) as usize;
        let removed = (chunk as usize - 1) + 2 + 11 * n_chunks - 1;
        assert_eq!(slow.steps - fast.steps, removed, "h={h} t={t} chunk={chunk}: expected {removed} fewer steps ({} -> {})", slow.steps, fast.steps);
    }
}
