// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::gdn::gdn_recurrent_step` (the single-token decode primitive) vs.
//! `model::gdn::gdn_chunk_fwd` run at `chunk=1` (six one-token "chunks") on the
//! SAME tiny random input, at a real, non-degenerate `GdnShape` (`b,h,dk,dv,t`
//! all pairwise distinct — `docs/lessons.md` #4).
//!
//! Why `gdn_chunk_fwd` at `chunk=1` is a valid oracle (not circular): at
//! `chunk=1` the UT-transform collapses to the 1x1 identity (`decay_mask`'s
//! only cell is `exp(0)=1`, `attn0` is always `0` — there is no `j<i` cell in
//! a 1x1 block), so `u = v_beta`, `w = k_beta*exp(g)`, and the sequential
//! per-chunk loop's `v_prime`/`v_new`/state-decay/state-accumulate become
//! EXACTLY `gdn_recurrent_step`'s own `kv_mem`/`delta`/state update — the same
//! algebra this module's own doc (above `gdn_recurrent_step`) works through.
//! `gdn_chunk_fwd` is already gated against an independent host f64 oracle
//! (`crates/model/tests/gdn_chunk_fwd.rs`), so this is a strong, self-contained
//! cross-check that needs no second host reference or PyTorch run — matching
//! the porting task's own reasoning for why this is a legitimate oracle.
//!
//! Run on both backends (`docs/lessons.md` #5):
//!   cargo test -p brain-model --test gdn_recurrent_step
//!   BRAIN_DEVICE=cpu cargo test -p brain-model --test gdn_recurrent_step

use data::rng::Lcg;
use gpu_core::Gpu;
use model::gdn::{gdn_chunk_fwd, gdn_recurrent_step, GdnIds, GdnRecurrentScratch, GdnScratch, GdnShape};

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
];

fn idx(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| panic!("kernel '{name}' not registered"))
}

fn ids(g: &Gpu) -> GdnIds {
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
    }
}

#[test]
fn gdn_recurrent_step_matches_chunk_fwd_at_chunk_1() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids = ids(&g);

    // Pairwise-distinct dims (docs/lessons.md #4): b=2,h=3,dk=5,dv=4,t=6.
    let (bn, hn, tn, dk, dv) = (2usize, 3usize, 6usize, 5usize, 4usize);
    let bh = bn * hn;
    let cn = 1usize; // chunk = 1: six one-token "chunks".
    let n_chunks = tn / cn;
    assert_eq!(n_chunks, tn);
    let bhc = bh * n_chunks;

    // Token-major-with-chunk-index-outermost == this module's chunk-major
    // layout AT chunk=1 (each "chunk" is exactly one token, so the chunk
    // index IS the token index): flat row = t*bh + (b*hn+h).
    let row = |t: usize, b: usize, h: usize| t * bh + (b * hn + h);

    let mut rng = Lcg::new(20260810);
    let mut q_cm = vec![0f32; bhc * dk];
    let mut k_cm = vec![0f32; bhc * dk];
    let mut v_cm = vec![0f32; bhc * dv];
    let mut g_cm = vec![0f32; bhc];
    let mut beta_cm = vec![0f32; bhc];
    for t in 0..tn {
        for b in 0..bn {
            for h in 0..hn {
                let r = row(t, b, h);
                for d in 0..dk {
                    q_cm[r * dk + d] = rng.scaled(1.0);
                    k_cm[r * dk + d] = rng.scaled(1.0);
                }
                for d in 0..dv {
                    v_cm[r * dv + d] = rng.scaled(1.0);
                }
                // Mildly negative on average (a real gate), some positive.
                g_cm[r] = rng.scaled(0.4) - 0.2;
                beta_cm[r] = rng.unit();
            }
        }
    }
    let init_state_h: Vec<f32> = rng.vec_scaled(bh * dk * dv, 0.5);

    // ==================== Oracle: gdn_chunk_fwd at chunk=1 ====================
    let query = g.storage_init("query", &q_cm);
    let key = g.storage_init("key", &k_cm);
    let value = g.storage_init("value", &v_cm);
    let raw_g = g.storage_init("raw_g", &g_cm);
    let beta = g.storage_init("beta", &beta_cm);
    let initial_state = g.storage_init("initial_state", &init_state_h);

    let shape = GdnShape { b: bn as u32, h: hn as u32, t: tn as u32, dk: dk as u32, dv: dv as u32, chunk: cn as u32 };
    assert_eq!(shape.n_chunks() as usize, n_chunks);
    assert_eq!(shape.bh() as usize, bh);
    assert_eq!(shape.bhc() as usize, bhc);

    let g_cs = g.storage((bhc * cn) as u64);
    let exp_g_cs = g.storage((bhc * cn) as u64);
    let k_beta = g.storage((bhc * cn * dk) as u64);
    let v_beta = g.storage((bhc * cn * dv) as u64);
    let k_beta_decay = g.storage((bhc * cn * dk) as u64);
    let decay_mask = g.storage((bhc * cn * cn) as u64);
    let raw_attn0 = g.storage((bhc * cn * cn) as u64);
    let attn0 = g.storage((bhc * cn * cn) as u64);
    let t_mat = g.storage((bhc * cn * cn) as u64);
    let u = g.storage((bhc * cn * dv) as u64);
    let w = g.storage((bhc * cn * dk) as u64);
    let raw_intra = g.storage((bhc * cn * cn) as u64);
    let intra_scores = g.storage((bhc * cn * cn) as u64);
    let q_scaled = g.storage((bh * cn * dk) as u64);
    let decay_scale = g.storage((bh * cn) as u64);
    let decayed_k = g.storage((bh * cn * dk) as u64);
    let v_prime = g.storage((bh * cn * dv) as u64);
    let v_new = g.storage((bh * cn * dv) as u64);

    let scratch = GdnScratch {
        g_cs: &g_cs,
        exp_g_cs: &exp_g_cs,
        k_beta: &k_beta,
        v_beta: &v_beta,
        k_beta_decay: &k_beta_decay,
        decay_mask: &decay_mask,
        raw_attn0: &raw_attn0,
        attn0: &attn0,
        t_mat: &t_mat,
        u: &u,
        w: &w,
        raw_intra: &raw_intra,
        intra_scores: &intra_scores,
        q_scaled: &q_scaled,
        decay_scale: &decay_scale,
        decayed_k: &decayed_k,
        v_prime: &v_prime,
        v_new: &v_new,
    };

    let out_ref = g.storage((bhc * cn * dv) as u64);
    let final_state_ref = g.storage((bh * dk * dv) as u64);

    let steps = gdn_chunk_fwd(&g, &ids, &shape, &query, &key, &value, &raw_g, &beta, &initial_state, &scratch, &out_ref, &final_state_ref);
    g.submit(&[&t_mat], &steps);

    let want_out = g.read(&out_ref, bhc * cn * dv);
    let want_state = g.read(&final_state_ref, bh * dk * dv);

    // ==================== gdn_recurrent_step, six times in a row ====================
    let state_rec = g.storage_init("state_rec", &init_state_h);
    let kv_mem = g.storage((bh * dv) as u64);
    let sub_out = g.storage((bh * dv) as u64);
    let rec_scratch = GdnRecurrentScratch { kv_mem: &kv_mem, sub_out: &sub_out };

    let q_step = g.storage((bh * dk) as u64);
    let k_step = g.storage((bh * dk) as u64);
    let v_step = g.storage((bh * dv) as u64);
    let g_step = g.storage(bh as u64);
    let beta_step = g.storage(bh as u64);
    let out_step = g.storage((bh * dv) as u64);

    let mut got_out = vec![0f32; bhc * dv];
    for t in 0..tn {
        let qs = &q_cm[t * bh * dk..(t + 1) * bh * dk];
        let ks = &k_cm[t * bh * dk..(t + 1) * bh * dk];
        let vs = &v_cm[t * bh * dv..(t + 1) * bh * dv];
        let gs = &g_cm[t * bh..(t + 1) * bh];
        let bs = &beta_cm[t * bh..(t + 1) * bh];

        g.write_f32(&q_step, qs);
        g.write_f32(&k_step, ks);
        g.write_f32(&v_step, vs);
        g.write_f32(&g_step, gs);
        g.write_f32(&beta_step, bs);

        let steps = gdn_recurrent_step(&g, &ids, &shape, &q_step, &k_step, &v_step, &g_step, &beta_step, &state_rec, &rec_scratch, &out_step);
        g.submit(&[], &steps);

        let out_h = g.read(&out_step, bh * dv);
        got_out[t * bh * dv..(t + 1) * bh * dv].copy_from_slice(&out_h);
    }
    let got_state = g.read(&state_rec, bh * dk * dv);

    let tol = 1e-4;
    let mut worst = 0f64;
    for (i, (&got, &want)) in got_out.iter().zip(&want_out).enumerate() {
        let delta = (got as f64 - want as f64).abs();
        worst = worst.max(delta);
        assert!(delta < tol, "out[{i}]: got {got} want {want} (delta {delta})");
    }
    for (i, (&got, &want)) in got_state.iter().zip(&want_state).enumerate() {
        let delta = (got as f64 - want as f64).abs();
        worst = worst.max(delta);
        assert!(delta < tol, "final_state[{i}]: got {got} want {want} (delta {delta})");
    }
    eprintln!("gdn_recurrent_step_matches_chunk_fwd_at_chunk_1: worst |delta| = {worst:e}");
}
