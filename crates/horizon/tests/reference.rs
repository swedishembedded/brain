// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Horizon's survival arithmetic against an independent implementation:
//! `tools/goldens/survival_likelihood_reference.py` (PyTorch) wrote
//! `testdata/survival_reference.json`. The encoder's exposure and event
//! pieces, the loss kernels (on the CPU backend) and their gradient, and the
//! closed-form survival and cumulative incidence in `survival::Curves` are held
//! to the numbers PyTorch derived from the raw subjects by its own rules, its
//! autograd, and a matrix exponential of each piece's competing-risks
//! generator.

use gpu_core::Gpu;
use horizon::encode::encode;
use horizon::survival::Curves;
use horizon::timeline::Subject;
use horizon::vocab::{FitOptions, Vocab};
use horizon::HorizonConfig;
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("../testdata/survival_reference.json"))
        .expect("survival_reference.json")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn setup(r: &Value) -> (Vec<Subject>, Vocab, HorizonConfig) {
    let subjects: Vec<Subject> = r["subjects"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| Subject::from_json_line(&s.to_string()).expect("a legal timeline"))
        .collect();
    let codes: Vec<String> = r["codes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_string())
        .collect();
    let absorbing: Vec<String> = codes
        .iter()
        .zip(r["absorbing"].as_array().unwrap())
        .filter(|(_, a)| a.as_bool().unwrap())
        .map(|(c, _)| c.clone())
        .collect();
    let opts = FitOptions {
        knots: 5,
        min_count: 1,
    };
    let vocab = Vocab::fit(&subjects, &codes, &absorbing, &opts).unwrap();
    let mut cfg = HorizonConfig::tiny(vocab.len(), codes.len() as u32);
    cfg.knots = floats(&r["knots"]).iter().map(|&k| k as f32).collect();
    (subjects, vocab, cfg)
}

#[test]
fn exposure_and_event_pieces_match_the_reference_rules() {
    let r = reference();
    let (subjects, vocab, cfg) = setup(&r);
    for (i, s) in subjects.iter().enumerate() {
        let e = encode(s, &vocab, &cfg);
        for (k, o) in e.outcomes.iter().enumerate() {
            // exposure is stored [subject][piece][code] in the reference.
            for (p, &x) in o.exposure.iter().enumerate() {
                let w = r["exposure"][i][p][k].as_f64().unwrap();
                assert!((x as f64 - w).abs() < 1e-6, "subject {i} code {k} piece {p}: {x} vs {w}");
            }
            let want_piece = r["event_piece"][i][k].as_u64().map(|p| p as u32);
            assert_eq!(o.event_piece, want_piece, "subject {i} code {k} event piece");
        }
    }
}

/// The weighted negative log-likelihood and its gradient, computed by the
/// loss kernels on the CPU backend from the encoder's exposure and events.
#[test]
fn the_loss_and_its_gradient_match_autograd() {
    let r = reference();
    let (subjects, vocab, cfg) = setup(&r);
    let (n, p, k) = (subjects.len(), cfg.knots.len() - 1, vocab.codes.len());
    let (mut loglam, mut event, mut expo) = (vec![], vec![], vec![]);
    for (i, s) in subjects.iter().enumerate() {
        let e = encode(s, &vocab, &cfg);
        for piece in 0..p {
            for code in 0..k {
                loglam.push(r["log_hazards"][i][piece][code].as_f64().unwrap() as f32);
                expo.push(e.outcomes[code].exposure[piece]);
                event.push(f32::from(e.outcomes[code].event_piece == Some(piece as u32)));
            }
        }
    }
    let w: Vec<f32> = subjects.iter().map(|s| s.weight as f32).collect();
    let inv_wsum = 1.0 / w.iter().sum::<f32>();
    let gpu = Gpu::new_cpu(&[
        ("pexp_nll_value", kernels::PEXP_NLL_VALUE),
        ("pexp_nll_grad", kernels::PEXP_NLL_GRAD),
    ]);
    let (lb, eb, xb, wb) = (
        gpu.storage_init("l", &loglam),
        gpu.storage_init("e", &event),
        gpu.storage_init("x", &expo),
        gpu.storage_init("w", &w),
    );
    let len = loglam.len();
    let (ob, gb) = (gpu.storage(len as u64), gpu.storage(len as u64));
    let params = [(n * p) as u32, k as u32, p as u32, gpu_core::f(inv_wsum)];
    let steps = [
        gpu.step(0, &[&lb, &eb, &xb, &wb, &ob], &params, len as u32),
        gpu.step(1, &[&lb, &eb, &xb, &wb, &gb], &params, len as u32),
    ];
    gpu.submit(&[], &steps);
    let (val, grad) = (gpu.read(&ob, len), gpu.read(&gb, len));
    let nll: f64 = val.iter().map(|&v| v as f64).sum();
    let want = r["nll"].as_f64().unwrap();
    assert!(
        (nll - want).abs() <= 1e-5 * (1.0 + want.abs()),
        "negative log-likelihood {nll} vs reference {want}"
    );
    for (e, &g) in grad.iter().enumerate() {
        let (i, rest) = (e / (p * k), e % (p * k));
        let want = r["nll_grad"][i][rest / k][rest % k].as_f64().unwrap();
        assert!(
            (g as f64 - want).abs() <= 1e-5 * (1.0 + want.abs()),
            "gradient element {e}: {g} vs reference {want}"
        );
    }
}

/// Survival and cumulative incidence, including a time past the last knot
/// (held at its value there).
#[test]
fn survival_and_cumulative_incidence_match_the_matrix_exponential() {
    let r = reference();
    let (subjects, _, cfg) = setup(&r);
    let absorbing: Vec<bool> = r["absorbing"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_bool().unwrap())
        .collect();
    let times = floats(&r["times"]);
    for i in 0..subjects.len() {
        let lh: Vec<f32> = r["log_hazards"][i]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| floats(row).into_iter().map(|x| x as f32))
            .collect();
        let curves = Curves::new(&lh, &cfg.knots, &absorbing);
        for (j, &t) in times.iter().enumerate() {
            let s = r["curves"][i]["survival"][j].as_f64().unwrap();
            assert!(
                (curves.survival(t) - s).abs() < 1e-6,
                "subject {i} survival at {t}: {} vs {s}",
                curves.survival(t)
            );
            for code in 0..absorbing.len() {
                let c = r["curves"][i]["cif"][j][code].as_f64().unwrap();
                assert!(
                    (curves.cif(code, t) - c).abs() < 1e-6,
                    "subject {i} code {code} cif at {t}: {} vs {c}",
                    curves.cif(code, t)
                );
            }
        }
    }
}
