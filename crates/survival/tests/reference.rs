// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Every metric against the reference libraries, on the data and values
//! `tools/goldens/survival_metrics_reference.py` wrote to
//! `testdata/reference.json` (scikit-survival and scikit-learn; versions are
//! recorded in the file).

use serde_json::Value;
use survival::auc;
use survival::brier::{brier, integrated_brier};
use survival::calibration::at_horizon;
use survival::concordance::{harrell, uno};
use survival::estimate::{aalen_johansen, censoring, kaplan_meier};
use survival::Obs;

fn reference() -> Value {
    serde_json::from_str(include_str!("../testdata/reference.json")).expect("reference.json")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-9 * (1.0 + want.abs()),
        "{what}: {got} vs reference {want}"
    );
}

#[test]
fn every_metric_matches_the_reference_libraries() {
    let r = reference();
    let time = floats(&r["single"]["time"]);
    let event: Vec<bool> = r["single"]["event"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_bool().unwrap())
        .collect();
    let risk = floats(&r["single"]["risk"]);
    let obs: Vec<Obs> = time
        .iter()
        .zip(&event)
        .map(|(&t, &e)| {
            if e {
                Obs::event(t, 0)
            } else {
                Obs::censored(t)
            }
        })
        .collect();
    let times = floats(&r["times"]);

    let (km, g) = (kaplan_meier(&obs), censoring(&obs));
    for (k, &t) in times.iter().enumerate() {
        close(
            km.at(t),
            floats(&r["km"])[k],
            &format!("kaplan-meier at {t}"),
        );
        close(
            g.at(t),
            floats(&r["censoring"])[k],
            &format!("censoring distribution at {t}"),
        );
    }
    close(
        harrell(&risk, &obs, 0).unwrap(),
        r["harrell"].as_f64().unwrap(),
        "harrell",
    );
    close(
        uno(&risk, &obs, 0, r["uno_tau"].as_f64().unwrap(), &g).unwrap(),
        r["uno"].as_f64().unwrap(),
        "uno",
    );

    for (k, &t) in floats(&r["auc_times"]).iter().enumerate() {
        close(
            auc::at(&risk, &obs, 0, t, &g).unwrap(),
            floats(&r["auc"])[k],
            &format!("time-dependent auc at {t}"),
        );
    }

    // Predicted survival exp(-risk * t / 4); the cumulative incidence is its complement.
    let cif_at = |k: usize| {
        risk.iter()
            .map(|r| 1.0 - (-r * times[k] / 4.0).exp())
            .collect::<Vec<f64>>()
    };
    for (k, &t) in times.iter().enumerate() {
        close(
            brier(&cif_at(k), &obs, 0, t, &g).unwrap(),
            floats(&r["brier"])[k],
            &format!("brier at {t}"),
        );
    }
    close(
        integrated_brier(&times, cif_at, &obs, 0, &g).unwrap(),
        r["ibs"].as_f64().unwrap(),
        "integrated brier",
    );

    let ts = r["recal_horizon"].as_f64().unwrap();
    let f: Vec<f64> = risk.iter().map(|r| 1.0 - (-r * ts / 4.0).exp()).collect();
    let cal = at_horizon(&f, &obs, 0, ts, &g, 4);
    // The logistic fits agree to the optimiser's tolerance, not to rounding.
    assert!(
        (cal.intercept - r["recal_intercept"].as_f64().unwrap()).abs() < 1e-6,
        "intercept {}",
        cal.intercept
    );
    assert!(
        (cal.slope - r["recal_slope"].as_f64().unwrap()).abs() < 1e-5,
        "slope {}",
        cal.slope
    );

    let ctime = floats(&r["competing"]["time"]);
    let cause: Vec<u64> = r["competing"]["cause"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap())
        .collect();
    // The reference numbers causes 1 and 2; this crate numbers them from 0.
    let cobs: Vec<Obs> = ctime
        .iter()
        .zip(&cause)
        .map(|(&t, &c)| {
            if c == 0 {
                Obs::censored(t)
            } else {
                Obs::event(t, c as usize - 1)
            }
        })
        .collect();
    let aj = aalen_johansen(&cobs, 0);
    for (k, &t) in times.iter().enumerate() {
        close(
            aj.at(t),
            floats(&r["aalen_johansen_cause1"])[k],
            &format!("aalen-johansen at {t}"),
        );
    }
}

/// The treatment effect with and without a prognostic score, against
/// statsmodels' OLS with HC3 standard errors
/// (`tools/goldens/trial_effect_reference.py`).
#[test]
fn the_trial_effect_matches_statsmodels() {
    let r: Value = serde_json::from_str(include_str!("../testdata/effect_reference.json"))
        .expect("effect_reference.json");
    let outcome = floats(&r["outcome"]);
    let treated: Vec<bool> = r["treated"]
        .as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_bool().expect("bool"))
        .collect();
    let score = floats(&r["score"]);
    for (name, got) in [
        (
            "unadjusted",
            survival::effect::ancova(&outcome, &treated, None).expect("estimable"),
        ),
        (
            "adjusted",
            survival::effect::ancova(&outcome, &treated, Some(&score)).expect("estimable"),
        ),
    ] {
        let want = &r[name];
        close(
            got.estimate,
            want["estimate"].as_f64().unwrap(),
            &format!("{name} estimate"),
        );
        close(got.se, want["se"].as_f64().unwrap(), &format!("{name} se"));
        close(got.lo, want["lo"].as_f64().unwrap(), &format!("{name} lo"));
        close(got.hi, want["hi"].as_f64().unwrap(), &format!("{name} hi"));
        assert!(
            (got.p_value - want["p_value"].as_f64().unwrap()).abs() < 1e-7,
            "{name} p {}",
            got.p_value
        );
    }
}

/// A NaN risk or value is not measured, never silently ranked or averaged.
#[test]
fn non_finite_inputs_are_not_measured() {
    let obs = [Obs::event(1.0, 0), Obs::censored(2.0), Obs::event(3.0, 0)];
    let g = censoring(&obs);
    assert!(harrell(&[0.9, f64::NAN, 0.1], &obs, 0).is_none());
    assert!(uno(&[0.9, 0.5, f64::NAN], &obs, 0, 5.0, &g).is_none());
    assert!(harrell(&[0.9, 0.5, 0.1], &obs, 0).is_some());
    let none =
        survival::compare::cluster_bootstrap(&[1.0, f64::NAN], &[1.0, 1.0], &[1, 2], 10, 0.95, 1);
    assert!(none.is_none());
}
