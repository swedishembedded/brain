// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `model::gdn::gdn_chunk_bwd` vs. a finite-difference gradient of a host
//! **f64** oracle, on the EXACT SAME tiny synthetic shape
//! `crates/model/tests/gdn_chunk_fwd.rs` uses (`B=1,H=2,T=8,Dk=3,Dv=4,C=4`,
//! two chunks). The host oracle here is `gdn_chunk_fwd.rs`'s own
//! `host_oracle` function, copied verbatim (test binaries are separate
//! crates, so it cannot be imported directly) rather than re-derived — this
//! is the finite-difference-gradcheck-oracle exception in `AGENTS.md`,
//! applied to a backward gradcheck rather than a forward one.
//!
//! `host_oracle` returns `(out, final_state)`, not a scalar, so it is wrapped
//! with `loss = sum(out) + sum(final_state)` — the simplest reduction that
//! makes EVERY output element (not just `out` or just `final_state` alone)
//! influence the scalar being perturbed around, which is what a
//! finite-difference check needs (a loss depending on `out` alone would never
//! exercise `d_final_state`'s seed into the reverse chunk sweep, and vice
//! versa). The DEVICE side seeds this same loss's gradient directly:
//! `d_out = 1` everywhere, `d_final_state = 1` everywhere (the gradient of a
//! plain sum is uniformly 1).
//!
//! Run on both backends (`docs/lessons.md` #5 — a barrier-crossing bug can
//! return silently-wrong results on exactly one backend):
//!   cargo test -p brain-model --test gdn_chunk_bwd
//!   BRAIN_DEVICE=cpu cargo test -p brain-model --test gdn_chunk_bwd

use data::rng::Lcg;
use gpu_core::Gpu;
use model::gdn::{gdn_chunk_bwd, gdn_chunk_fwd_train, GdnBwdIds, GdnBwdScratch, GdnIds, GdnScratchTrain, GdnShape};

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
    // ---- backward-only kernels ----
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

/// `crates/model/tests/gdn_chunk_fwd.rs`'s own host f64 oracle, copied
/// verbatim (see this file's module doc for why). DO NOT hand-edit this
/// function without also updating the forward test's copy — they must stay
/// byte-identical, since the whole point is an INDEPENDENTLY re-derived
/// reference, not a copy that quietly diverges from the one the forward
/// gate already trusts.
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
                let mut qs = vec![vec![0f64; dk]; cn];
                let mut kraw = vec![vec![0f64; dk]; cn];
                let mut kb = vec![vec![0f64; dk]; cn];
                let mut vb = vec![vec![0f64; dv]; cn];
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
                let mut gcs = vec![0f64; cn];
                gcs[0] = gg[0];
                for i in 1..cn {
                    gcs[i] = gcs[i - 1] + gg[i];
                }
                let mut dm = vec![vec![0f64; cn]; cn];
                for (i, row) in dm.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate().take(i + 1) {
                        *cell = (gcs[i] - gcs[j]).exp();
                    }
                }
                let mut attn0 = vec![vec![0f64; cn]; cn];
                for i in 0..cn {
                    for j in 0..i {
                        let dot: f64 = (0..dk).map(|d| kb[i][d] * kraw[j][d]).sum();
                        attn0[i][j] = -dot * dm[i][j];
                    }
                }
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
                let mut u = vec![vec![0f64; dv]; cn];
                for i in 0..cn {
                    for d in 0..dv {
                        u[i][d] = (0..cn).map(|j| tmat[i][j] * vb[j][d]).sum();
                    }
                }
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
                let mut intra = vec![vec![0f64; cn]; cn];
                for (i, row) in intra.iter_mut().enumerate() {
                    for (j, cell) in row.iter_mut().enumerate() {
                        let dot: f64 = (0..dk).map(|d| qs[i][d] * kraw[j][d]).sum();
                        *cell = dot * dm[i][j];
                    }
                }
                let mut v_prime = vec![vec![0f64; dv]; cn];
                for i in 0..cn {
                    for d in 0..dv {
                        v_prime[i][d] = (0..dk).map(|dd| w[i][dd] * state[ist(b, h, dd, d)]).sum();
                    }
                }
                let mut v_new = vec![vec![0f64; dv]; cn];
                for i in 0..cn {
                    for d in 0..dv {
                        v_new[i][d] = u[i][d] - v_prime[i][d];
                    }
                }
                let mut attn_inter = vec![vec![0f64; dv]; cn];
                for i in 0..cn {
                    let eg = gcs[i].exp();
                    for d in 0..dv {
                        attn_inter[i][d] = (0..dk).map(|dd| qs[i][dd] * eg * state[ist(b, h, dd, d)]).sum();
                    }
                }
                for i in 0..cn {
                    let t = c * cn + i;
                    for d in 0..dv {
                        let acc: f64 = (0..cn).map(|j| intra[i][j] * v_new[j][d]).sum();
                        out[iout(b, h, t, d)] = attn_inter[i][d] + acc;
                    }
                }
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

/// `loss = sum(out) + sum(final_state)` — see this file's module doc for why
/// this specific reduction.
#[allow(clippy::too_many_arguments)]
fn loss(bn: usize, hn: usize, tn: usize, dk: usize, dv: usize, cn: usize, q: &[f64], k: &[f64], v: &[f64], g_raw: &[f64], beta: &[f64], init_state: &[f64]) -> f64 {
    let (out, state) = host_oracle(bn, hn, tn, dk, dv, cn, q, k, v, g_raw, beta, init_state);
    out.iter().sum::<f64>() + state.iter().sum::<f64>()
}

/// Central-difference gradient of `loss_of` w.r.t. every element of `x`.
fn fd_grad(x: &mut [f64], eps: f64, loss_of: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    let mut grad = vec![0.0; x.len()];
    for i in 0..x.len() {
        let orig = x[i];
        x[i] = orig + eps;
        let lp = loss_of(x);
        x[i] = orig - eps;
        let lm = loss_of(x);
        x[i] = orig;
        grad[i] = (lp - lm) / (2.0 * eps);
    }
    grad
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
fn gdn_chunk_bwd_gradcheck() {
    let g = gpu_core::testgpu::dev(PIPES);
    let ids_v = ids(&g);
    let bids = bwd_ids(&g);

    // Exact same tiny shape as gdn_chunk_fwd.rs.
    let (bn, hn, tn, dk, dv, cn) = (1usize, 2usize, 8usize, 3usize, 4usize, 4usize);
    let n_chunks = tn / cn;
    let bh = bn * hn;
    let bhc = bh * n_chunks;

    let mut rng = Lcg::new(20260810);
    let q_h: Vec<f32> = rng.vec_scaled(bn * hn * tn * dk, 1.0);
    let k_h: Vec<f32> = rng.vec_scaled(bn * hn * tn * dk, 1.0);
    let v_h: Vec<f32> = rng.vec_scaled(bn * hn * tn * dv, 1.0);
    let g_h: Vec<f32> = (0..bn * hn * tn).map(|_| rng.scaled(0.4) - 0.2).collect();
    let beta_h: Vec<f32> = (0..bn * hn * tn).map(|_| rng.unit()).collect();
    let init_state_h: Vec<f32> = rng.vec_scaled(bh * dk * dv, 0.5);

    let to_f64 = |v: &[f32]| -> Vec<f64> { v.iter().map(|&x| x as f64).collect() };
    let mut q64 = to_f64(&q_h);
    let mut k64 = to_f64(&k_h);
    let mut v64 = to_f64(&v_h);
    let mut g64 = to_f64(&g_h);
    let mut beta64 = to_f64(&beta_h);
    let mut init64 = to_f64(&init_state_h);

    // ---- finite-difference gradients, host f64 oracle ----
    let eps = 1e-4;
    let d_query_fd = fd_grad(&mut q64, eps, |q| loss(bn, hn, tn, dk, dv, cn, q, &k64, &v64, &g64, &beta64, &init64));
    let d_key_fd = fd_grad(&mut k64, eps, |k| loss(bn, hn, tn, dk, dv, cn, &q64, k, &v64, &g64, &beta64, &init64));
    let d_value_fd = fd_grad(&mut v64, eps, |v| loss(bn, hn, tn, dk, dv, cn, &q64, &k64, v, &g64, &beta64, &init64));
    let d_raw_g_fd = fd_grad(&mut g64, eps, |gr| loss(bn, hn, tn, dk, dv, cn, &q64, &k64, &v64, gr, &beta64, &init64));
    let d_beta_fd = fd_grad(&mut beta64, eps, |be| loss(bn, hn, tn, dk, dv, cn, &q64, &k64, &v64, &g64, be, &init64));
    let d_init_fd = fd_grad(&mut init64, eps, |is| loss(bn, hn, tn, dk, dv, cn, &q64, &k64, &v64, &g64, &beta64, is));

    // ---- device buffers, chunk-major layout per model::gdn's contract ----
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

    // ---- GdnScratchTrain buffers ----
    let g_cs = g.storage((bhc * cn) as u64);
    let exp_g_cs = g.storage((bhc * cn) as u64);
    let k_beta = g.storage((bhc * cn * dk) as u64);
    let v_beta = g.storage((bhc * cn * dv) as u64);
    let k_beta_decay = g.storage((bhc * cn * dk) as u64);
    let decay_mask = g.storage((bhc * cn * cn) as u64);
    let raw_attn0 = g.storage((bhc * cn * cn) as u64);
    let attn0 = g.storage((bhc * cn * cn) as u64);
    let t_mat = g.storage((bhc * cn * cn) as u64);
    let u_buf = g.storage((bhc * cn * dv) as u64);
    let w_buf = g.storage((bhc * cn * dk) as u64);
    let raw_intra = g.storage((bhc * cn * cn) as u64);
    let intra_scores = g.storage((bhc * cn * cn) as u64);
    let q_scaled = g.storage((bh * cn * dk) as u64);
    let decay_scale = g.storage((bh * cn) as u64);
    let decayed_k = g.storage((bh * cn * dk) as u64);
    let v_prime = g.storage((bh * cn * dv) as u64);
    let v_new = g.storage((bh * cn * dv) as u64);
    let q_scaled_hist = g.storage((bhc * cn * dk) as u64);
    let decay_scale_hist = g.storage((bhc * cn) as u64);
    let decayed_k_hist = g.storage((bhc * cn * dk) as u64);
    let v_prime_hist = g.storage((bhc * cn * dv) as u64);
    let v_new_hist = g.storage((bhc * cn * dv) as u64);
    let state_history = g.storage(((n_chunks + 1) * bh * dk * dv) as u64);

    let scratch_train = GdnScratchTrain {
        g_cs: &g_cs,
        exp_g_cs: &exp_g_cs,
        k_beta: &k_beta,
        v_beta: &v_beta,
        k_beta_decay: &k_beta_decay,
        decay_mask: &decay_mask,
        raw_attn0: &raw_attn0,
        attn0: &attn0,
        t_mat: &t_mat,
        u: &u_buf,
        w: &w_buf,
        raw_intra: &raw_intra,
        intra_scores: &intra_scores,
        q_scaled: &q_scaled,
        decay_scale: &decay_scale,
        decayed_k: &decayed_k,
        v_prime: &v_prime,
        v_new: &v_new,
        q_scaled_hist: &q_scaled_hist,
        decay_scale_hist: &decay_scale_hist,
        decayed_k_hist: &decayed_k_hist,
        v_prime_hist: &v_prime_hist,
        v_new_hist: &v_new_hist,
        state_history: &state_history,
    };

    let out = g.storage((bhc * cn * dv) as u64);
    let final_state = g.storage((bh * dk * dv) as u64);

    let fwd_steps = gdn_chunk_fwd_train(&g, &ids_v, &bids, &shape, &query, &key, &value, &raw_g, &beta, &initial_state, &scratch_train, &out, &final_state);
    g.submit(&[&t_mat, &q_scaled_hist, &decay_scale_hist, &decayed_k_hist, &v_prime_hist, &v_new_hist, &state_history], &fwd_steps);

    // ---- cross-check: gdn_chunk_fwd_train's own out/final_state should
    // match the f64 host oracle (same tolerance as gdn_chunk_fwd.rs) before
    // trusting the backward gradcheck built on top of it. ----
    let (want_out, want_state) = host_oracle(bn, hn, tn, dk, dv, cn, &q64, &k64, &v64, &g64, &beta64, &init64);
    let got_out_cm = g.read(&out, bhc * cn * dv);
    let got_state = g.read(&final_state, bh * dk * dv);
    let mut worst_fwd = 0f64;
    for b in 0..bn {
        for h in 0..hn {
            for c in 0..n_chunks {
                for i in 0..cn {
                    let t = c * cn + i;
                    for d in 0..dv {
                        let got = got_out_cm[to_chunk_major_d(b, h, c, i, d, hn, cn, bh, dv)] as f64;
                        let want = want_out[((b * hn + h) * tn + t) * dv + d];
                        worst_fwd = worst_fwd.max((got - want).abs());
                    }
                }
            }
        }
    }
    for (&got, &want) in got_state.iter().zip(&want_state) {
        worst_fwd = worst_fwd.max((got as f64 - want).abs());
    }
    assert!(worst_fwd < 1e-4, "gdn_chunk_fwd_train forward cross-check: worst |delta| = {worst_fwd:e}");
    eprintln!("gdn_chunk_fwd_train forward cross-check: worst |delta| = {worst_fwd:e}");

    // ---- GdnBwdScratch buffers ----
    let d_decayed_k = g.storage((bh * cn * dk) as u64);
    let d_q_scaled = g.storage((bh * cn * dk) as u64);
    let d_v_new = g.storage((bh * cn * dv) as u64);
    let d_decay_scale = g.storage((bh * cn) as u64);
    let d_query_chunk = g.storage((bh * cn * dk) as u64);
    let d_key_chunk = g.storage((bh * cn * dk) as u64);
    let state_a = g.storage((bh * dk * dv) as u64);
    let state_b = g.storage((bh * dk * dv) as u64);
    let d_raw_intra = g.storage((bhc * cn * cn) as u64);
    let d_k_beta_decay = g.storage((bhc * cn * dk) as u64);
    let d_v_beta = g.storage((bhc * cn * dv) as u64);
    let d_raw_attn0 = g.storage((bhc * cn * cn) as u64);
    let d_attn0 = g.storage((bhc * cn * cn) as u64);
    let d_g_cs = g.storage((bhc * cn) as u64);
    let d_exp_g_cs = g.storage((bhc * cn) as u64);
    let d_t_mat = g.storage((bhc * cn * cn) as u64);
    let d_u = g.storage((bhc * cn * dv) as u64);
    let d_w = g.storage((bhc * cn * dk) as u64);
    let d_intra_scores = g.storage((bhc * cn * cn) as u64);
    let d_decay_mask = g.storage((bhc * cn * cn) as u64);
    let d_k_beta = g.storage((bhc * cn * dk) as u64);
    let dot_scratch = g.storage((bhc * cn) as u64);
    let mul_scratch = g.storage((bhc * cn) as u64);
    let mul_scratch_cc = g.storage((bhc * cn * cn) as u64);

    let bwd_scratch = GdnBwdScratch {
        d_decayed_k: &d_decayed_k,
        d_q_scaled: &d_q_scaled,
        d_v_new: &d_v_new,
        d_decay_scale: &d_decay_scale,
        d_query_chunk: &d_query_chunk,
        d_key_chunk: &d_key_chunk,
        state_a: &state_a,
        state_b: &state_b,
        d_raw_intra: &d_raw_intra,
        d_k_beta_decay: &d_k_beta_decay,
        d_v_beta: &d_v_beta,
        d_raw_attn0: &d_raw_attn0,
        d_attn0: &d_attn0,
        d_g_cs: &d_g_cs,
        d_exp_g_cs: &d_exp_g_cs,
        d_t_mat: &d_t_mat,
        d_u: &d_u,
        d_w: &d_w,
        d_intra_scores: &d_intra_scores,
        d_decay_mask: &d_decay_mask,
        d_k_beta: &d_k_beta,
        dot_scratch: &dot_scratch,
        mul_scratch: &mul_scratch,
        mul_scratch_cc: &mul_scratch_cc,
    };

    let d_out = g.storage_init("d_out", &vec![1f32; bhc * cn * dv]);
    let d_final_state = g.storage_init("d_final_state", &vec![1f32; bh * dk * dv]);

    let d_query = g.storage((bhc * cn * dk) as u64);
    let d_key = g.storage((bhc * cn * dk) as u64);
    let d_value = g.storage((bhc * cn * dv) as u64);
    let d_raw_g = g.storage((bhc * cn) as u64);
    let d_beta = g.storage((bhc * cn) as u64);
    let d_initial_state = g.storage((bh * dk * dv) as u64);

    let bwd_steps = gdn_chunk_bwd(
        &g,
        &ids_v,
        &bids,
        &shape,
        &query,
        &key,
        &value,
        &beta,
        &scratch_train,
        &d_out,
        &d_final_state,
        &bwd_scratch,
        &d_query,
        &d_key,
        &d_value,
        &d_raw_g,
        &d_beta,
        &d_initial_state,
    );
    g.submit(&[&d_g_cs, &d_exp_g_cs, &d_u, &d_decay_mask, &d_query, &d_key, &d_beta], &bwd_steps);

    let got_d_query = g.read(&d_query, bhc * cn * dk);
    let got_d_key = g.read(&d_key, bhc * cn * dk);
    let got_d_value = g.read(&d_value, bhc * cn * dv);
    let got_d_raw_g = g.read(&d_raw_g, bhc * cn);
    let got_d_beta = g.read(&d_beta, bhc * cn);
    let got_d_init = g.read(&d_initial_state, bh * dk * dv);

    // ---- compare, converting the device's chunk-major layout back to
    // natural [b,h,t,d] to match the FD gradient's own layout ----
    let mut worst_abs = 0f64;
    let mut worst_rel = 0f64;
    let mut check = |name: &str, got: f64, want: f64| {
        let d = (got - want).abs();
        let rel = d / want.abs().max(1e-6);
        worst_abs = worst_abs.max(d);
        worst_rel = worst_rel.max(rel);
        assert!(d < 1e-3 || rel < 1e-3, "{name}: got {got} want {want} (abs {d:e}, rel {rel:e})");
    };

    for b in 0..bn {
        for h in 0..hn {
            for c in 0..n_chunks {
                for i in 0..cn {
                    let t = c * cn + i;
                    for d in 0..dk {
                        let cm = to_chunk_major_d(b, h, c, i, d, hn, cn, bh, dk);
                        let nat = ((b * hn + h) * tn + t) * dk + d;
                        check("d_query", got_d_query[cm] as f64, d_query_fd[nat]);
                        check("d_key", got_d_key[cm] as f64, d_key_fd[nat]);
                    }
                    for d in 0..dv {
                        let cm = to_chunk_major_d(b, h, c, i, d, hn, cn, bh, dv);
                        let nat = ((b * hn + h) * tn + t) * dv + d;
                        check("d_value", got_d_value[cm] as f64, d_value_fd[nat]);
                    }
                    let cm = to_chunk_major(b, h, c, i, hn, cn, bh);
                    let nat = (b * hn + h) * tn + t;
                    check("d_raw_g", got_d_raw_g[cm] as f64, d_raw_g_fd[nat]);
                    check("d_beta", got_d_beta[cm] as f64, d_beta_fd[nat]);
                }
            }
        }
    }
    for (&got, &want) in got_d_init.iter().zip(&d_init_fd) {
        check("d_initial_state", got as f64, want);
    }

    eprintln!("gdn_chunk_bwd_gradcheck: worst abs = {worst_abs:e}, worst rel = {worst_rel:e}");
}
