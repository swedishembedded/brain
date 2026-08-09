// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::gdn::gdn_chunk_fwd` vs. a host **f64** oracle, on a tiny synthetic
//! shape with pairwise-distinct `B,H,T,Dk,Dv,C` (`docs/lessons.md` #4: a toy
//! shape where dims coincide can hide a head/width swap that a real port
//! would carry into the checkpoint). `B=1,H=2,T=8,Dk=3,Dv=4,C=4` gives TWO
//! chunks, so the sequential across-chunk state carry (step 10) is actually
//! exercised, not vacuously skipped.
//!
//! The oracle below is a plain nested-loop transcription of
//! `torch_chunk_gated_delta_rule` (`transformers/models/qwen3_5_moe/
//! modeling_qwen3_5_moe.py`), independently re-derived from the reference
//! source rather than mirroring `model::gdn`'s kernel decomposition — this is
//! the finite-difference-gradcheck-oracle exception in `AGENTS.md`, applied
//! to a forward oracle. It matches the reference LITERALLY, including the
//! easy-to-miss reassignment `value = attn @ v_beta` (step 8): every later
//! `value`/`v_i` in the reference (`v_new = v_i - v_prime`) means THIS
//! reassigned tensor, never the function's original `value` parameter.
//!
//! Run on both backends (`docs/lessons.md` #5 — a barrier-crossing bug can
//! return silently-wrong results on exactly one backend):
//!   cargo test -p brain-model --test gdn_chunk_fwd
//!   BRAIN_DEVICE=cpu cargo test -p brain-model --test gdn_chunk_fwd

use data::rng::Lcg;
use gpu_core::Gpu;
use model::gdn::{gdn_chunk_fwd, GdnIds, GdnScratch, GdnShape};

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

/// Host f64 oracle: `torch_chunk_gated_delta_rule`, one `(b,h)` at a time,
/// plain nested loops, natural `[b,h,t,d]` indexing (NOT this module's
/// chunk-major device layout -- the two are reconciled by the test's own
/// index-conversion helpers). Returns `(core_attn_out, final_state)`, both
/// natural-layout.
#[allow(clippy::too_many_arguments)]
fn host_oracle(
    bn: usize,
    hn: usize,
    tn: usize,
    dk: usize,
    dv: usize,
    cn: usize,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    g_raw: &[f64],
    beta: &[f64],
    init_state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let n_chunks = tn / cn;
    let scale = 1.0f64 / (dk as f64).sqrt();
    let iq = |b: usize, h: usize, t: usize, d: usize| ((b * hn + h) * tn + t) * dk + d;
    let iv = |b: usize, h: usize, t: usize, d: usize| ((b * hn + h) * tn + t) * dv + d;
    let ig = |b: usize, h: usize, t: usize| (b * hn + h) * tn + t;
    let ist = |b: usize, h: usize, dki: usize, dvi: usize| ((b * hn + h) * dk + dki) * dv + dvi;
    let iout = |b: usize, h: usize, t: usize, d: usize| ((b * hn + h) * tn + t) * dv + d;

    let mut out = vec![0f64; bn * hn * tn * dv];
    let mut state = init_state.to_vec();

    for b in 0..bn {
        for h in 0..hn {
            for c in 0..n_chunks {
                let mut qs = vec![vec![0f64; dk]; cn]; // query, GLOBALLY scaled (step 1)
                let mut kraw = vec![vec![0f64; dk]; cn];
                let mut kb = vec![vec![0f64; dk]; cn]; // k_beta
                let mut vb = vec![vec![0f64; dv]; cn]; // v_beta
                let mut gg = vec![0f64; cn];
                for i in 0..cn {
                    let t = c * cn + i;
                    let bt = beta[ig(b, h, t)];
                    for d in 0..dk {
                        let qv = q[iq(b, h, t, d)] * scale;
                        qs[i][d] = qv;
                        let kv = k[iq(b, h, t, d)];
                        kraw[i][d] = kv;
                        kb[i][d] = kv * bt;
                    }
                    for d in 0..dv {
                        vb[i][d] = v[iv(b, h, t, d)] * bt;
                    }
                    gg[i] = g_raw[ig(b, h, t)];
                }
                // g_cs: per-chunk cumsum.
                let mut gcs = vec![0f64; cn];
                gcs[0] = gg[0];
                for i in 1..cn {
                    gcs[i] = gcs[i - 1] + gg[i];
                }
                // decay_mask[i,j] = exp(gcs[i]-gcs[j]) for j<=i else 0.
                let mut dm = vec![vec![0f64; cn]; cn];
                for (i, row) in dm.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate().take(i + 1) {
                        *cell = (gcs[i] - gcs[j]).exp();
                    }
                }
                // attn0[i,j] = -(k_beta[i].key[j])*decay_mask[i,j] for j<i else 0.
                let mut attn0 = vec![vec![0f64; cn]; cn];
                for i in 0..cn {
                    for j in 0..i {
                        let dot: f64 = (0..dk).map(|d| kb[i][d] * kraw[j][d]).sum();
                        attn0[i][j] = -dot * dm[i][j];
                    }
                }
                // UT-transform: forward substitution, then += I.
                let mut tmat = vec![vec![0f64; cn]; cn];
                for i in 1..cn {
                    for j in 0..i {
                        let mut acc = attn0[i][j];
                        for kk in (j + 1)..i {
                            acc += attn0[i][kk] * tmat[kk][j];
                        }
                        tmat[i][j] = acc;
                    }
                }
                for (i, row) in tmat.iter_mut().enumerate() {
                    row[i] += 1.0;
                }
                // u = T_mat @ v_beta -- REASSIGNS the reference's "value".
                let mut u = vec![vec![0f64; dv]; cn];
                for i in 0..cn {
                    for d in 0..dv {
                        u[i][d] = (0..cn).map(|j| tmat[i][j] * vb[j][d]).sum();
                    }
                }
                // k_cumdecay = T_mat @ (k_beta * exp(g_cs)).
                let mut kg = vec![vec![0f64; dk]; cn];
                for i in 0..cn {
                    let eg = gcs[i].exp();
                    for d in 0..dk {
                        kg[i][d] = kb[i][d] * eg;
                    }
                }
                let mut w = vec![vec![0f64; dk]; cn];
                for i in 0..cn {
                    for d in 0..dk {
                        w[i][d] = (0..cn).map(|j| tmat[i][j] * kg[j][d]).sum();
                    }
                }
                // intra_scores[i,j] = (q_i.k_j)*decay_mask[i,j] (q already scaled).
                let mut intra = vec![vec![0f64; cn]; cn];
                for (i, row) in intra.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate() {
                        let dot: f64 = (0..dk).map(|d| qs[i][d] * kraw[j][d]).sum();
                        *cell = dot * dm[i][j];
                    }
                }
                // v_prime = k_cumdecay @ state (state BEFORE this chunk's update).
                let mut v_prime = vec![vec![0f64; dv]; cn];
                for i in 0..cn {
                    for d in 0..dv {
                        v_prime[i][d] = (0..dk).map(|dd| w[i][dd] * state[ist(b, h, dd, d)]).sum();
                    }
                }
                // v_new = u - v_prime  (NOT the original raw value -- see module doc).
                let mut v_new = vec![vec![0f64; dv]; cn];
                for i in 0..cn {
                    for d in 0..dv {
                        v_new[i][d] = u[i][d] - v_prime[i][d];
                    }
                }
                // attn_inter = (q * exp(g_cs)) @ state.
                let mut attn_inter = vec![vec![0f64; dv]; cn];
                for i in 0..cn {
                    let eg = gcs[i].exp();
                    for d in 0..dv {
                        attn_inter[i][d] = (0..dk).map(|dd| qs[i][dd] * eg * state[ist(b, h, dd, d)]).sum();
                    }
                }
                // core_out = attn_inter + intra_scores @ v_new.
                for i in 0..cn {
                    let t = c * cn + i;
                    for d in 0..dv {
                        let acc: f64 = (0..cn).map(|j| intra[i][j] * v_new[j][d]).sum();
                        out[iout(b, h, t, d)] = attn_inter[i][d] + acc;
                    }
                }
                // state = state*exp(g_cs[-1]) + (key*exp(g_cs[-1]-g_cs))^T @ v_new.
                let g_last = gcs[cn - 1];
                let decay = g_last.exp();
                let mut new_state = vec![0f64; dk * dv];
                for dd in 0..dk {
                    for d in 0..dv {
                        new_state[dd * dv + d] = state[ist(b, h, dd, d)] * decay;
                    }
                }
                for i in 0..cn {
                    let dscale = (g_last - gcs[i]).exp();
                    for dd in 0..dk {
                        let dkv = kraw[i][dd] * dscale;
                        for d in 0..dv {
                            new_state[dd * dv + d] += dkv * v_new[i][d];
                        }
                    }
                }
                for dd in 0..dk {
                    for d in 0..dv {
                        state[ist(b, h, dd, d)] = new_state[dd * dv + d];
                    }
                }
            }
        }
    }
    (out, state)
}

/// This module's own chunk-major device layout (see `model::gdn`'s doc):
/// `bhc = chunk*(B*H) + b*H + h` is the outermost flat batch axis.
fn to_chunk_major_d(b: usize, h: usize, c: usize, i: usize, d: usize, hn: usize, cn: usize, bh: usize, dd: usize) -> usize {
    let bhc = c * bh + b * hn + h;
    (bhc * cn + i) * dd + d
}
fn to_chunk_major(b: usize, h: usize, c: usize, i: usize, hn: usize, cn: usize, bh: usize) -> usize {
    let bhc = c * bh + b * hn + h;
    bhc * cn + i
}

#[test]
fn gdn_chunk_fwd_matches_host_oracle() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids = ids(&g);

    // Pairwise-distinct dims (docs/lessons.md #4), 2 chunks so the
    // across-chunk state carry is actually exercised.
    let (bn, hn, tn, dk, dv, cn) = (1usize, 2usize, 8usize, 3usize, 4usize, 4usize);
    let n_chunks = tn / cn;
    let bh = bn * hn;
    let bhc = bh * n_chunks;

    let mut rng = Lcg::new(20260809);
    let q_h: Vec<f32> = rng.vec_scaled(bn * hn * tn * dk, 1.0);
    let k_h: Vec<f32> = rng.vec_scaled(bn * hn * tn * dk, 1.0);
    let v_h: Vec<f32> = rng.vec_scaled(bn * hn * tn * dv, 1.0);
    // Log-decay: mildly negative on average (a real gate), some positive.
    let g_h: Vec<f32> = (0..bn * hn * tn).map(|_| rng.scaled(0.4) - 0.2).collect();
    // beta: already a sigmoid gate, i.e. in (0,1).
    let beta_h: Vec<f32> = (0..bn * hn * tn).map(|_| rng.unit()).collect();
    let init_state_h: Vec<f32> = rng.vec_scaled(bh * dk * dv, 0.5);

    // ---- host f64 oracle ----
    let to_f64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
    let (want_out, want_state) = host_oracle(
        bn,
        hn,
        tn,
        dk,
        dv,
        cn,
        &to_f64(&q_h),
        &to_f64(&k_h),
        &to_f64(&v_h),
        &to_f64(&g_h),
        &to_f64(&beta_h),
        &to_f64(&init_state_h),
    );

    // ---- device buffers, laid out chunk-major per model::gdn's contract ----
    let mut q_cm = vec![0f32; bhc * cn * dk];
    let mut k_cm = vec![0f32; bhc * cn * dk];
    let mut v_cm = vec![0f32; bhc * cn * dv];
    let mut g_cm = vec![0f32; bhc * cn];
    let mut beta_cm = vec![0f32; bhc * cn];
    for b in 0..bn {
        for h in 0..hn {
            for c in 0..n_chunks {
                for i in 0..cn {
                    let t = c * cn + i;
                    for d in 0..dk {
                        q_cm[to_chunk_major_d(b, h, c, i, d, hn, cn, bh, dk)] = q_h[((b * hn + h) * tn + t) * dk + d];
                        k_cm[to_chunk_major_d(b, h, c, i, d, hn, cn, bh, dk)] = k_h[((b * hn + h) * tn + t) * dk + d];
                    }
                    for d in 0..dv {
                        v_cm[to_chunk_major_d(b, h, c, i, d, hn, cn, bh, dv)] = v_h[((b * hn + h) * tn + t) * dv + d];
                    }
                    g_cm[to_chunk_major(b, h, c, i, hn, cn, bh)] = g_h[(b * hn + h) * tn + t];
                    beta_cm[to_chunk_major(b, h, c, i, hn, cn, bh)] = beta_h[(b * hn + h) * tn + t];
                }
            }
        }
    }

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

    // Scratch buffer sizes -- documented here at the point of allocation, per
    // `GdnScratch`'s own doc: bhc = B*H*n_chunks, bh = B*H, c = chunk.
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

    let out = g.storage((bhc * cn * dv) as u64);
    let final_state = g.storage((bh * dk * dv) as u64);

    let steps = gdn_chunk_fwd(&g, &ids, &shape, &query, &key, &value, &raw_g, &beta, &initial_state, &scratch, &out, &final_state);
    // t_mat MUST be cleared before the UT-transform loop -- see gdn_chunk_fwd's doc.
    g.submit(&[&t_mat], &steps);

    let got_out_cm = g.read(&out, bhc * cn * dv);
    let got_state = g.read(&final_state, bh * dk * dv);

    // Compare "out" by converting the device's chunk-major layout back to
    // natural [b,h,t,d] to match the oracle's own layout.
    let tol = 1e-4;
    let mut worst = 0f64;
    for b in 0..bn {
        for h in 0..hn {
            for c in 0..n_chunks {
                for i in 0..cn {
                    let t = c * cn + i;
                    for d in 0..dv {
                        let got = got_out_cm[to_chunk_major_d(b, h, c, i, d, hn, cn, bh, dv)] as f64;
                        let want = want_out[((b * hn + h) * tn + t) * dv + d];
                        let delta = (got - want).abs();
                        worst = worst.max(delta);
                        assert!(delta < tol, "out[b={b},h={h},t={t},d={d}]: got {got} want {want} (delta {delta})");
                    }
                }
            }
        }
    }
    for (i, (&got, &want)) in got_state.iter().zip(&want_state).enumerate() {
        let delta = (got as f64 - want).abs();
        worst = worst.max(delta);
        assert!(delta < tol, "final_state[{i}]: got {got} want {want} (delta {delta})");
    }
    eprintln!("gdn_chunk_fwd_matches_host_oracle: worst |delta| = {worst:e}");
}
