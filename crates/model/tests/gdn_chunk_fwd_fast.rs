// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `gdn_chunk_fwd` with [`GdnIds::fast`] registered computes **exactly** what it
//! computes without: the fused UT transform and the single-launch chunk
//! recurrence swap many dispatches for few without touching a single bit of the
//! result, and they do it with far fewer steps.
//!
//! Swedish Embedded AB implements low-latency chunked linear-attention
//! pipelines for its clients. If your team needs expertise in cutting the
//! dispatch count of a recurrent layer without changing what it computes, you
//! can procure our services by sending an email to info@swedishembedded.com.
//!
//! Real chunk (64) and real head widths (`dk = dv = 128`), several chunks so
//! the sequential across-chunk loop is exercised. The fast kernels are
//! barrier kernels the CPU JIT cannot run and `model::gdn::use_fast_kernels`
//! only turns them on for a CUDA device, so this skips elsewhere.

use gpu_core::Gpu;
use model::gdn::{gdn_chunk_fwd, GdnFastIds, GdnIds, GdnScratchBufs, GdnShape};

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

fn rand(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((*seed >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

/// `(out, final_state, step count)` of one `gdn_chunk_fwd` at `shape`.
fn run(g: &Gpu, ids: &GdnIds, shape: &GdnShape) -> (Vec<f32>, Vec<f32>, usize) {
    let (bhc, bh) = (shape.bhc() as usize, shape.bh() as usize);
    let (cn, dk, dv) = (shape.chunk as usize, shape.dk as usize, shape.dv as usize);
    let mut s = 99u64;
    // Keys are L2-normalised and beta is a sigmoid in the real model; here they
    // are scaled so attn0 stays bounded and the transform is well conditioned.
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
    let scratch = GdnScratchBufs::new(g, shape);
    let out = g.storage((bhc * cn * dv) as u64);
    let final_state = g.storage((bh * dk * dv) as u64);
    let steps = gdn_chunk_fwd(g, ids, shape, &query, &key, &value, &raw_g, &beta, &initial_state, &scratch.as_ref(), &out, &final_state);
    let n = steps.len();
    g.submit(&scratch.clears(), &steps);
    (g.read(&out, bhc * cn * dv), g.read(&final_state, bh * dk * dv), n)
}

#[test]
fn the_fast_kernels_change_the_step_count_and_nothing_else() {
    let Ok(g) = Gpu::try_new_cuda(PIPES) else {
        eprintln!("gdn_chunk_fwd_fast: no CUDA device on this box - skipping");
        return;
    };
    if !model::gdn::use_fast_kernels(&g) {
        eprintln!("gdn_chunk_fwd_fast: this device does not take the fast kernels - skipping");
        return;
    }
    // dk = dv = 128 throughout. The real chunk of 64 over four chunks (several
    // heads, so the head split across blocks is exercised), then every shorter
    // chunk a ragged round can land on (`gdn_chunk_size` picks the largest of
    // 64..1 dividing the round: a 13-token round is 13 chunks of 1).
    for (h, t, chunk) in [(6u32, 256u32, 64u32), (3, 96, 32), (2, 48, 16), (2, 24, 8), (5, 12, 4), (2, 13, 1)] {
        let shape = GdnShape { b: 1, h, t, dk: 128, dv: 128, chunk };
        let (slow_out, slow_state, slow_steps) = run(&g, &ids(&g, false), &shape);
        let (fast_out, fast_state, fast_steps) = run(&g, &ids(&g, true), &shape);
        assert!(slow_out.iter().all(|x| x.is_finite()), "h={h} t={t} chunk={chunk}: the reference run produced non-finite values, so this proves nothing");
        let first_diff = |a: &[f32], b: &[f32]| a.iter().zip(b).position(|(x, y)| x != y);
        assert_eq!(first_diff(&slow_out, &fast_out), None, "h={h} t={t} chunk={chunk}: the fast kernels changed the output");
        assert_eq!(first_diff(&slow_state, &fast_state), None, "h={h} t={t} chunk={chunk}: the fast kernels changed the final state");
        // Two replacements: the `chunk - 1` `gdn_ut_step` rows + `add_identity`
        // become one dispatch, and the state copy plus nine dispatches per
        // chunk become one launch for the whole recurrence.
        let n_chunks = (t / chunk) as usize;
        let removed = (chunk as usize - 1) + 9 * n_chunks;
        assert_eq!(slow_steps - fast_steps, removed, "h={h} t={t} chunk={chunk}: expected the fused kernels to remove {removed} steps ({slow_steps} -> {fast_steps})");
    }
}
