// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for off-policy reinforcement learning of large language models on one
// pass of the LM head for its clients. If your team needs expertise in score-centred policy-gradient training then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Sparse extra targets on `model::head_ce::HeadCe`: per position up to K `(id, weight)` pairs beside the label.
//!
//! Spec: after one forward the trainer's log-probability at every extra id equals the f64 host reference; one backward
//! leaves the hidden gradient of `sum_n w_n nll(y_n) + sum_nj c_nj nll(v_nj)` (the f64 reference), for chunk sizes and
//! vocabulary windows that divide neither the token count nor the vocabulary, under a logit scale and a soft cap, with
//! duplicate ids, ids equal to the label, unused (IGNORE) slots and ignored rows; extra targets whose ids are all
//! IGNORE give the plain weighted gradient; and a malformed id list is refused naming the position.

use data::rng::Lcg;
use gpu_core::provider::library::GemmPrecision;
use gpu_core::select::Dtype;
use gpu_core::Gpu;
use model::head_ce::{check_sparse_ids, HeadCe, HeadCeConfig, HeadKernels, IGNORE};
use model::head_transform::HeadTransform;
use model::ops::{training_kernel_list, Ops, Weight};

fn ops() -> Ops {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| training_kernel_list().iter().chain(model::head_ce::kernels()).copied().collect());
    let mut ops = Ops::new(Gpu::new(list)).expect("Ops::new");
    ops.set_gemm_precision(GemmPrecision::Fp32Reference);
    ops
}

struct Problem {
    n: usize,
    d: usize,
    v: usize,
    k: usize,
    hidden: Vec<f32>,
    raw: Vec<f32>,
    labels: Vec<u32>,
    weights: Vec<f32>,
    ids: Vec<u32>,
    coefs: Vec<f32>,
}

/// Random rows with a few ignored labels; each row's K slots hold random ids, a duplicate, the label itself and an
/// unused slot, with signed coefficients (the score-centred residuals are of either sign).
fn problem(n: usize, d: usize, v: usize, k: usize) -> Problem {
    let mut rng = Lcg::random();
    let hidden = rng.vec_scaled(n * d, 1.0);
    let raw = rng.vec_scaled(v * d, 0.4);
    let labels: Vec<u32> = (0..n).map(|i| if i % 11 == 3 { IGNORE } else { (rng.next_u64() % v as u64) as u32 }).collect();
    let weights: Vec<f32> = (0..n).map(|i| if i % 7 == 2 { 0.0 } else { rng.unit() * 2.0 - 0.5 }).collect();
    let mut ids = vec![IGNORE; n * k];
    let mut coefs = vec![0f32; n * k];
    for i in (0..n).filter(|&i| labels[i] != IGNORE) {
        for j in 0..k {
            ids[i * k + j] = match j {
                1 => ids[i * k],
                2 => labels[i],
                j if j == k - 1 && i % 2 == 0 => IGNORE,
                _ => (rng.next_u64() % v as u64) as u32,
            };
            coefs[i * k + j] = rng.unit() - 0.5;
        }
    }
    Problem { n, d, v, k, hidden, raw, labels, weights, ids, coefs }
}

/// f64 logprobs at the extra ids and the hidden gradient (times `d_scale`) of the loss with extra targets, under a
/// logit scale and an optional soft cap.
fn reference(p: &Problem, w: &[f32], logit_scale: f32, cap: Option<f32>, d_scale: f32) -> (Vec<f64>, Vec<f64>) {
    let (n, d, v, k) = (p.n, p.d, p.v, p.k);
    let mut lps = vec![0f64; n * k];
    let mut d_hidden = vec![0f64; n * d];
    for i in (0..n).filter(|&i| p.labels[i] != IGNORE) {
        let z: Vec<f64> = (0..v).map(|j| (0..d).map(|c| f64::from(p.hidden[i * d + c]) * f64::from(w[j * d + c])).sum::<f64>()).collect();
        let y: Vec<f64> = z.iter().map(|&z| f64::from(logit_scale) * cap.map_or(z, |c| f64::from(c) * (z / f64::from(c)).tanh())).collect();
        let mx = y.iter().copied().fold(f64::MIN, f64::max);
        let lse = mx + y.iter().map(|&t| (t - mx).exp()).sum::<f64>().ln();
        let mut target = vec![0f64; v];
        target[p.labels[i] as usize] += f64::from(p.weights[i]);
        for j in 0..k {
            let id = p.ids[i * k + j];
            if id != IGNORE {
                lps[i * k + j] = y[id as usize] - lse;
                target[id as usize] += f64::from(p.coefs[i * k + j]);
            }
        }
        let total: f64 = target.iter().sum();
        for j in 0..v {
            let dy = total * (y[j] - lse).exp() - target[j];
            let dz = dy * f64::from(logit_scale) * cap.map_or(1.0, |c| 1.0 - (z[j] / f64::from(c)).tanh().powi(2));
            for c in 0..d {
                d_hidden[i * d + c] += dz * f64::from(d_scale) * f64::from(w[j * d + c]);
            }
        }
    }
    (lps, d_hidden)
}

fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(0f64, |a, &v| a.max(v.abs())).max(1e-6);
    let worst = got.iter().zip(want).map(|(&g, &w)| (f64::from(g) - w).abs()).fold(0f64, f64::max);
    assert!(worst <= tol * scale, "{what}: worst abs error {worst:e} exceeds {tol:e} of the scale {scale:e}");
}

fn build(ops: &Ops, p: &Problem, w: &Weight, rows: u32, windows: bool) -> HeadCe {
    let row_words = if w.dtype() == Dtype::BF16 { p.d / 2 } else { p.d } as u64;
    let cfg = HeadCeConfig { rows_per_chunk: rows, tile_budget_words: windows.then_some(row_words * 250) };
    let hc = HeadCe::new(ops, p.d as u32, p.v as u32, w, cfg, HeadKernels::from_gpu(ops.gpu()).expect("head kernels")).expect("HeadCe::new");
    assert_eq!(hc.window_count() > 1, windows);
    assert!(hc.sparse_targets_supported(), "head_ce::kernels() registers the sparse-target kernels");
    hc
}

/// The read-back log-probabilities and the one-pass gradient equal the f64 reference over chunk boundaries, vocabulary
/// windows, a logit scale and a soft cap.
#[test]
fn extra_target_logprobs_and_gradient_match_the_host_reference() {
    let ops = ops();
    let g = ops.gpu();
    // n = 150 and v = 1000 divide by neither the 64-row chunk nor the 250-row windows; K = 5.
    let p = problem(150, 96, 1000, 5);
    let (logit_scale, d_scale) = (0.7f32, 1.0 / 140.0);
    for tier in [Dtype::F32, Dtype::BF16] {
        let w = Weight::upload(&ops, &p.raw, p.v, p.d, tier);
        let eff: Vec<f32> = if tier == Dtype::BF16 { p.raw.iter().map(|&x| f32::from_bits(u32::from(model::half::f32_to_bf16(x)) << 16)).collect() } else { p.raw.clone() };
        for cap in [None, Some(3.0f32)] {
            let (lps, d_hidden) = reference(&p, &eff, logit_scale, cap, d_scale);
            for (rows, windows) in [(64, false), (64, true), (192, false)] {
                let mut hc = build(&ops, &p, &w, rows, windows);
                if let Some(c) = cap {
                    hc.set_head_transform(g, HeadTransform { softcap: Some(c), valid_vocab: None }).expect("transform kernels");
                }
                let what = format!("{tier:?} cap={cap:?} rows={rows} windows={windows}");
                let hidden = g.storage_init("hidden", &p.hidden);
                let (_, got_lps) = hc.forward_sparse(&ops, &hidden, p.n as u32, &p.labels, &p.ids, p.k as u32, logit_scale).expect("valid ids");
                assert_close(&got_lps, &lps, 1e-4, &format!("{what}: extra-target logprobs"));
                let dh = g.storage((p.n * p.d) as u64);
                hc.backward_sparse(&ops, &hidden, &dh, p.n as u32, &p.labels, &p.weights, &p.ids, &p.coefs, p.k as u32, d_scale, logit_scale, None).expect("valid ids");
                assert_close(&g.read(&dh, p.n * p.d), &d_hidden, 1e-4, &format!("{what}: d_hidden"));
            }
        }
    }
}

/// Extra targets that are all unused are the plain weighted cross-entropy (up to the order of two multiplications).
#[test]
fn unused_extra_targets_give_the_weighted_gradient() {
    let ops = ops();
    let g = ops.gpu();
    let mut p = problem(70, 64, 300, 3);
    p.ids.iter_mut().for_each(|id| *id = IGNORE);
    let w = Weight::upload(&ops, &p.raw, p.v, p.d, Dtype::F32);
    let hc = build(&ops, &p, &w, 64, false);
    let hidden = g.storage_init("hidden", &p.hidden);
    let (plain, sparse) = (g.storage((p.n * p.d) as u64), g.storage((p.n * p.d) as u64));
    hc.backward(&ops, &hidden, &plain, p.n as u32, &p.labels, Some(&p.weights), 0.05, 0.8, None);
    hc.backward_sparse(&ops, &hidden, &sparse, p.n as u32, &p.labels, &p.weights, &p.ids, &p.coefs, p.k as u32, 0.05, 0.8, None).expect("valid ids");
    let want: Vec<f64> = g.read(&plain, p.n * p.d).into_iter().map(f64::from).collect();
    assert_close(&g.read(&sparse, p.n * p.d), &want, 1e-6, "all-IGNORE extra targets");
}

/// An id past the vocabulary, or an extra target on a row whose label is ignored, is refused naming the position.
#[test]
fn malformed_extra_targets_are_refused() {
    let labels = [3u32, IGNORE];
    assert!(check_sparse_ids(&labels, &[1, 2, IGNORE, IGNORE], 2, 10).is_ok());
    let e = check_sparse_ids(&labels, &[1, 10, IGNORE, IGNORE], 2, 10).unwrap_err();
    assert!(e.contains("position 0") && e.contains("10"), "{e}");
    let e = check_sparse_ids(&labels, &[1, 2, 4, IGNORE], 2, 10).unwrap_err();
    assert!(e.contains("position 1") && e.contains("ignored"), "{e}");
    let e = check_sparse_ids(&labels, &[1, 2, 4], 2, 10).unwrap_err();
    assert!(e.contains("3 extra-target ids"), "{e}");
}

/// Spec (support replay): a loss of log-probabilities renormalised over a sampler's support,
/// `sum_n omega_n (-log pi_S(y_n)) + sum_nj omega_nj (-log pi_S(s_nj))` with `pi_S = softmax` over the support's
/// logits, is one forward and one backward of the extra targets: the support's ids in the slots, the read-back
/// whole-row log-probabilities renormalised on the host ([`renormalise_over_support`]) equal an f64 softmax over the
/// support's logits, and the label weight `omega` with the slot coefficients [`support_coefficients`] give the hidden
/// gradient of an f64 oracle that differentiates the renormalised loss directly. Supports of varying width, padded
/// with unused slots, over chunk boundaries and a logit scale.
#[test]
fn a_loss_renormalised_over_a_support_is_one_backward_of_its_extra_targets() {
    use model::head_ce::{renormalise_over_support, support_coefficients};
    let ops = ops();
    let g = ops.gpu();
    let (n, d, v, k) = (90usize, 48usize, 400usize, 6usize);
    let mut p = problem(n, d, v, k);
    let mut rng = Lcg::random();
    // Each supervised row's support: its label first, then distinct ids, 2..=k of them; the rest unused. Some slots
    // carry a weight of their own (score centring's listed ids), the others none.
    let mut slot_w = vec![0f64; n * k];
    for i in 0..n {
        let width = if p.labels[i] == IGNORE { 0 } else { 2 + i % (k - 1) };
        let mut ids = vec![IGNORE; k];
        if width > 0 {
            ids[0] = p.labels[i];
            let mut j = 1;
            while j < width {
                let id = (rng.next_u64() % v as u64) as u32;
                if !ids[..j].contains(&id) {
                    ids[j] = id;
                    slot_w[i * k + j] = if j % 2 == 0 { f64::from(rng.unit()) - 0.5 } else { 0.0 };
                    j += 1;
                }
            }
        }
        p.ids[i * k..(i + 1) * k].copy_from_slice(&ids);
    }
    let (logit_scale, d_scale) = (0.8f32, 1.0 / 70.0);
    let w = Weight::upload(&ops, &p.raw, p.v, p.d, Dtype::F32);
    let hc = build(&ops, &p, &w, 64, false);
    let hidden = g.storage_init("hidden", &p.hidden);
    let (_, lps) = hc.forward_sparse(&ops, &hidden, n as u32, &p.labels, &p.ids, k as u32, logit_scale).expect("valid ids");
    let mut coefs = vec![0f32; n * k];
    let mut want_dh = vec![0f64; n * d];
    let mut worst = 0f64;
    for i in (0..n).filter(|&i| p.labels[i] != IGNORE) {
        let (slots, ids) = (&lps[i * k..(i + 1) * k], &p.ids[i * k..(i + 1) * k]);
        let norm = renormalise_over_support(slots[0], slots, ids).expect("a support");
        // The f64 oracle: logits over the support, its softmax, and the renormalised loss's gradient in the logits.
        let y: Vec<f64> = (0..v).map(|j| f64::from(logit_scale) * (0..d).map(|c| f64::from(p.hidden[i * d + c]) * f64::from(p.raw[j * d + c])).sum::<f64>()).collect();
        let used: Vec<usize> = (0..k).filter(|&j| ids[j] != IGNORE).collect();
        let mx = used.iter().map(|&j| y[ids[j] as usize]).fold(f64::MIN, f64::max);
        let lse = mx + used.iter().map(|&j| (y[ids[j] as usize] - mx).exp()).sum::<f64>().ln();
        worst = worst.max((norm.label - (y[p.labels[i] as usize] - lse)).abs());
        for &j in &used {
            worst = worst.max((norm.pi[j] - (y[ids[j] as usize] - lse).exp()).abs());
        }
        let omega_y = f64::from(p.weights[i]);
        let slot = &slot_w[i * k..(i + 1) * k];
        coefs[i * k..(i + 1) * k].copy_from_slice(&support_coefficients(&norm, omega_y, slot));
        let total = omega_y + slot.iter().sum::<f64>();
        let mut dy = vec![0f64; v];
        dy[p.labels[i] as usize] -= omega_y;
        for &j in &used {
            dy[ids[j] as usize] += total * (y[ids[j] as usize] - lse).exp() - slot[j];
        }
        for (col, g) in dy.iter().enumerate().filter(|(_, g)| **g != 0.0) {
            for c in 0..d {
                want_dh[i * d + c] += g * f64::from(logit_scale) * f64::from(d_scale) * f64::from(p.raw[col * d + c]);
            }
        }
    }
    assert!(worst < 1e-5, "the renormalised log-probabilities are the f64 softmax over the support's logits: worst {worst:e}");
    let dh = g.storage((n * d) as u64);
    hc.backward_sparse(&ops, &hidden, &dh, n as u32, &p.labels, &p.weights, &p.ids, &coefs, k as u32, d_scale, logit_scale, None).expect("valid ids");
    assert_close(&g.read(&dh, n * d), &want_dh, 1e-4, "the hidden gradient of the loss renormalised over the support");
}
