// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Tuning advisor** — turn an eval artifact (and, if present, a per-capability
//! scaling artifact) into a *ranked, concrete* set of recommendations for "what to
//! tune to improve in the best capability direction."
//!
//! This is the deliverable the user asked for: the eval output should "include a
//! breakdown of what parameters need to be tuned in order to achieve results that
//! improve in the best capability direction." The advisor reasons **per capability
//! axis**, ranks axes by *expected gain per lever*, and for each weak axis inspects
//! its signals to emit a concrete action with a rationale.
//!
//! ## Inputs
//! - **eval artifact** (`results/<arch>-<seed>.json`): per-axis scores, per-axis
//!   gating status, and per-benchmark `train_ce` / `init_ce` / `chance` / threshold.
//! - **capscale artifact** (`results/scale-<arch>-<seed>.json`, optional): per-axis
//!   size-slope / verdict / predicted-if-doubled. When absent the advisor still
//!   ranks by headroom and the train-vs-eval signal, but cannot say *whether size
//!   will help* — so the recommendations note that and suggest running `scale`.
//!
//! ## Heuristics (each documented at its use site)
//! 1. **Rank lever = headroom × responsiveness.** `headroom = 1 − score` (gated
//!    axes only); `responsiveness = size-slope` from capscale (or a neutral prior
//!    when absent). The product is the *expected gain* from spending capacity on
//!    that axis — highest first.
//! 2. **Per-axis signal → action:**
//!    - low score + steep size-slope ⇒ *increase model size/depth* (capacity-bound).
//!    - low score + FLAT size-slope ⇒ *change the mechanism* (attention /
//!      positional / memory): the axis is **architecture-bound**, more params won't
//!      help.
//!    - low eval score but **low train_ce** (train fits, eval lags) ⇒
//!      *overfitting / undertraining mismatch* → more data / regularization / steps.
//!    - score ≈ ceiling ⇒ *saturated*, deprioritize.
//! 3. **Compute-efficiency.** Each rec carries score-per-million-params so the
//!    advice weighs cost, and the lever ranking can be read as "best gain per
//!    capacity dollar."

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;

use crate::axes::axes;
use crate::capscale::{self, LoadedCapScale};

/// One ranked tuning recommendation for a capability axis.
#[derive(Clone, Debug)]
pub struct Recommendation {
    /// 1-based priority (1 = act on this first).
    pub priority: usize,
    pub axis: String,
    /// Current per-axis eval score.
    pub current: f32,
    /// Predicted score if the knob (size today) is doubled, from capscale; `None`
    /// when no capscale artifact was supplied.
    pub predicted_if_doubled: Option<f64>,
    /// The concrete action to take.
    pub action: String,
    /// Why — the signals that produced this action.
    pub rationale: String,
    /// Expected-gain lever score the ranking used (headroom × responsiveness).
    pub lever: f64,
}

impl Recommendation {
    /// A one-line rendering for the CLI footer / `advise` output.
    pub fn render(&self) -> String {
        let pred = match self.predicted_if_doubled {
            Some(p) => format!("{p:.3}"),
            None => "n/a".to_string(),
        };
        format!(
            "[{}] axis={:<14} current={:.3}  pred-if-size-doubled={:>5}  → {}\n      rationale: {}",
            self.priority, self.axis, self.current, pred, self.action, self.rationale,
        )
    }
}

/// The advisor's output: the ranked recommendations plus a short headline.
#[derive(Clone, Debug)]
pub struct Advice {
    pub arch: String,
    pub recommendations: Vec<Recommendation>,
    /// `true` if a capscale artifact informed the ranking (size-slopes available).
    pub used_capscale: bool,
}

impl Advice {
    /// The top `n` recommendations (already ranked).
    pub fn top(&self, n: usize) -> &[Recommendation] {
        &self.recommendations[..self.recommendations.len().min(n)]
    }
}

/// Per-axis facts gathered from the eval artifact.
struct AxisFacts {
    score: f32,
    /// `true` if any *gating* (non-informational) benchmark maps to this axis.
    gated: bool,
    /// Mean `train_ce` over the axis's benchmarks (NaN if none reported it).
    mean_train_ce: f32,
    /// Mean `init_ce` (the untrained-baseline CE) over the axis's benchmarks.
    mean_init_ce: f32,
}

/// Parse the eval artifact into per-axis facts + the headline params.
fn eval_axis_facts(v: &Value) -> (String, u64, BTreeMap<String, AxisFacts>) {
    let arch = v["arch"].as_str().unwrap_or("?").to_string();
    let params = v["param_count"].as_u64().unwrap_or(0);

    let mut scores: BTreeMap<String, f32> = BTreeMap::new();
    if let Some(obj) = v["axis_scores"].as_object() {
        for (k, val) in obj {
            if let Some(x) = val.as_f64() {
                scores.insert(k.clone(), x as f32);
            }
        }
    }

    // Aggregate per-axis train_ce / init_ce and whether the axis is gated.
    let mut gated: BTreeMap<String, bool> = BTreeMap::new();
    let mut train_ce: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut init_ce: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    if let Some(arr) = v["benchmarks"].as_array() {
        for b in arr {
            let Some(axis) = b["axis"].as_str() else { continue };
            let informational = b["informational"].as_bool().unwrap_or(false);
            if !informational {
                *gated.entry(axis.to_string()).or_insert(false) = true;
            }
            gated.entry(axis.to_string()).or_insert(false);
            if let Some(t) = b["metrics"]["fields"]["train_ce"].as_f64() {
                train_ce.entry(axis.to_string()).or_default().push(t as f32);
            }
            if let Some(t) = b["metrics"]["fields"]["init_ce"].as_f64() {
                init_ce.entry(axis.to_string()).or_default().push(t as f32);
            }
        }
    }

    let mut facts = BTreeMap::new();
    for (axis, score) in scores {
        let mean = |m: &BTreeMap<String, Vec<f32>>| -> f32 {
            match m.get(&axis) {
                Some(v) if !v.is_empty() => v.iter().sum::<f32>() / v.len() as f32,
                _ => f32::NAN,
            }
        };
        facts.insert(
            axis.clone(),
            AxisFacts {
                score,
                gated: gated.get(&axis).copied().unwrap_or(false),
                mean_train_ce: mean(&train_ce),
                mean_init_ce: mean(&init_ce),
            },
        );
    }
    (arch, params, facts)
}

/// Build the ranked tuning advice from an eval artifact and optional capscale.
///
/// Steps: (1) gather per-axis facts; (2) compute the expected-gain lever for each
/// *gated* axis = headroom × responsiveness; (3) sort descending; (4) per axis,
/// pick the concrete action from its signals.
pub fn advise(eval: &Value, capscale: Option<&LoadedCapScale>) -> Advice {
    let (arch, params, facts) = eval_axis_facts(eval);
    let params_m = (params as f64 / 1.0e6).max(1e-9); // millions of params (cost basis)

    // Neutral responsiveness prior when no capscale: assume an axis *might* respond
    // to size (slope ≈ a small positive), so headroom still drives the ranking but
    // we don't over-claim. With capscale we use the measured slope.
    const NEUTRAL_SLOPE: f64 = 0.10;

    struct Scored {
        axis: String,
        facts_score: f32,
        lever: f64,
        slope: f64,
        verdict: String,
        pred_2x: Option<f64>,
        beta: f64,
    }

    let mut scored: Vec<Scored> = Vec::new();
    for ax in axes() {
        let Some(f) = facts.get(ax) else { continue };
        // Heuristic 1: rank only GATED axes (informational-only axes don't gate the
        // suite, so "improving" them is not the user's objective).
        if !f.gated {
            continue;
        }
        let headroom = (1.0 - f.score as f64).max(0.0);
        let cap = capscale.and_then(|c| c.by_axis.get(ax));
        let slope = cap.map(|a| a.slope_per_doubling).unwrap_or(NEUTRAL_SLOPE);
        // Responsiveness floored at 0 (a negative measured slope is noise, not a
        // reason to invest). Lever = expected gain from spending capacity here.
        let lever = headroom * slope.max(0.0);
        scored.push(Scored {
            axis: ax.to_string(),
            facts_score: f.score,
            lever,
            slope,
            verdict: cap.map(|a| a.verdict.clone()).unwrap_or_else(|| "unknown".to_string()),
            pred_2x: cap.map(|a| a.pred_2x),
            beta: cap.map(|a| a.beta).unwrap_or(0.0),
        });
    }

    // Sort by lever desc; tie-break by lowest current score (weakest first).
    scored.sort_by(|a, b| {
        b.lever
            .partial_cmp(&a.lever)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.facts_score.partial_cmp(&b.facts_score).unwrap_or(std::cmp::Ordering::Equal))
    });

    let mut recs = Vec::new();
    for (i, s) in scored.iter().enumerate() {
        let f = &facts[&s.axis];
        let (action, mut rationale) = pick_action(f, s.slope, &s.verdict, capscale.is_some());

        // Heuristic 3: compute-efficiency — score per million params, so the advice
        // weighs cost. Cheap-but-weak axes are better levers than expensive ones.
        let spp = f.score as f64 / params_m;
        rationale.push_str(&format!(
            " [score/Mparam={spp:.3}; lever={:.3}]",
            s.lever
        ));

        recs.push(Recommendation {
            priority: i + 1,
            axis: s.axis.clone(),
            current: f.score,
            predicted_if_doubled: s.pred_2x,
            action,
            rationale,
            lever: s.lever,
        });
        let _ = s.beta; // beta surfaced in capscale report; rationale uses verdict/slope
    }

    Advice { arch, recommendations: recs, used_capscale: capscale.is_some() }
}

/// Choose the concrete action + rationale for one axis from its signals
/// (Heuristic 2). `has_capscale` distinguishes a measured flat slope (architecture-
/// bound) from "we don't know yet, run scale".
fn pick_action(
    f: &AxisFacts,
    slope: f64,
    verdict: &str,
    has_capscale: bool,
) -> (String, String) {
    let score = f.score;
    let train_ce = f.mean_train_ce;
    let init_ce = f.mean_init_ce;

    // Saturated: near the ceiling — no further gain from size. Deprioritize.
    if verdict == "saturating" || score >= 0.95 {
        return (
            "deprioritize: already near the capability ceiling".to_string(),
            format!("score {score:.3} is saturated (verdict={verdict}); spend capacity elsewhere"),
        );
    }

    // Train fits but eval lags: train_ce is low (the model learned the training
    // distribution) yet the eval score is weak ⇒ generalization gap, not capacity.
    let train_fits = train_ce.is_finite() && train_ce < 0.5;
    if train_fits && score < 0.7 {
        return (
            "more data / regularization / steps (close the train→eval gap)".to_string(),
            format!(
                "train_ce={train_ce:.3} is low (model fits train) but eval score {score:.3} lags \
                 → overfitting/undertraining mismatch, not a capacity limit"
            ),
        );
    }

    // With a scaling curve, the capscale verdict is the authority on whether size
    // helps (it already folds slope + ceiling proximity into one call).
    //
    // Capacity-bound: the curve is still climbing with size ⇒ more params/depth
    // helps. (verdict==improving, or a steep measured slope.)
    if has_capscale && (verdict == "improving" || slope >= 0.05) {
        return (
            "increase model size / depth (more capacity)".to_string(),
            format!(
                "score {score:.3} with a rising size-slope ({slope:.3}/doubling, verdict={verdict}) \
                 → capacity-bound; bigger N is predicted to raise this axis"
            ),
        );
    }

    // Architecture-bound: the curve is FLAT with size ⇒ size won't help; change
    // the mechanism (attention / positional / memory).
    if has_capscale {
        let init = if init_ce.is_finite() { format!(", init_ce={init_ce:.3}") } else { String::new() };
        return (
            "change the MECHANISM (attention / positional / memory) — not size".to_string(),
            format!(
                "score {score:.3} with a FLAT size-slope ({slope:.3}/doubling, verdict={verdict}{init}) \
                 → architecture-bound, not capacity-bound: more params won't move it"
            ),
        );
    }

    // No capscale: we know the axis is weak but not whether size helps. Recommend
    // running the scaling sweep, with size as the default first lever.
    (
        "run `brain bench scale` to test if size helps; tentatively increase size".to_string(),
        format!(
            "score {score:.3} is below ceiling but no scaling curve available \
             → run the per-capability sweep to choose size-vs-mechanism"
        ),
    )
}

/// Load an eval artifact JSON from `path`.
pub fn load_eval(path: &Path) -> std::io::Result<Value> {
    let text = std::fs::read_to_string(path)?;
    serde_json::from_str(&text).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Print the ranked tuning recommendations (the `advise` subcommand output).
pub fn print_advice(advice: &Advice) {
    println!("\ntuning advisor — arch: {}{}", advice.arch, if advice.used_capscale {
        "  (size-scaling informed)"
    } else {
        "  (no scaling artifact — run `brain bench scale` for size-vs-mechanism guidance)"
    });
    println!("ranked: what to tune to improve in the best capability direction\n");
    if advice.recommendations.is_empty() {
        println!("  (no gated capability axes below ceiling — nothing to recommend)");
        return;
    }
    for r in &advice.recommendations {
        println!("{}", r.render());
    }
    println!();
}

/// Print a short "top tuning recommendations" footer (used by `bench eval` so the
/// eval output itself carries the tuning breakdown). Shows the top `n`.
pub fn print_footer(advice: &Advice, n: usize) {
    let top = advice.top(n);
    if top.is_empty() {
        return;
    }
    println!("top tuning recommendations (improve in the best capability direction)");
    println!("{}", "-".repeat(68));
    for r in top {
        let pred = match r.predicted_if_doubled {
            Some(p) => format!(" → ~{p:.2} if size 2×", ),
            None => String::new(),
        };
        println!("  [{}] {:<14} (now {:.3}{}): {}", r.priority, r.axis, r.current, pred, r.action);
    }
    if !advice.used_capscale {
        println!("  (run `brain bench scale --arch {}` for size-vs-mechanism predictions)", advice.arch);
    }
    println!();
}

/// Convenience: load the optional capscale artifact for an `(arch, seed)` if it
/// exists on disk, returning `None` (not an error) when absent. Lets `bench eval`
/// enrich its footer when a scaling artifact happens to be present.
pub fn try_load_capscale(arch: &str, seed: u64) -> Option<LoadedCapScale> {
    let path = capscale::default_out_path(arch, seed);
    capscale::load_artifact(&path).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn eval_with(axis_scores: Value, benches: Value) -> Value {
        json!({
            "arch": "gpt2",
            "param_count": 110336u64,
            "axis_scores": axis_scores,
            "benchmarks": benches,
        })
    }

    #[test]
    fn ranks_weak_responsive_axis_first() {
        // recall weak (0.4), state_tracking strong (0.99). With a capscale where
        // recall has a steep slope, recall must rank first.
        let eval = eval_with(
            json!({ "recall": 0.40, "state_tracking": 0.99 }),
            json!([
                { "axis": "recall", "informational": false,
                  "metrics": { "fields": { "train_ce": 1.2, "init_ce": 2.9 } } },
                { "axis": "state_tracking", "informational": false,
                  "metrics": { "fields": { "train_ce": 0.1, "init_ce": 1.4 } } },
            ]),
        );
        let advice = advise(&eval, None);
        assert!(!advice.recommendations.is_empty());
        assert_eq!(advice.recommendations[0].axis, "recall");
    }

    #[test]
    fn train_fits_but_eval_lags_recommends_data_reg() {
        let eval = eval_with(
            json!({ "recall": 0.45 }),
            json!([
                { "axis": "recall", "informational": false,
                  "metrics": { "fields": { "train_ce": 0.1, "init_ce": 2.9 } } },
            ]),
        );
        let advice = advise(&eval, None);
        let r = &advice.recommendations[0];
        assert!(r.action.contains("data") || r.action.contains("regular"), "got: {}", r.action);
    }

    #[test]
    fn saturated_axis_is_deprioritized() {
        let eval = eval_with(
            json!({ "memory": 0.99 }),
            json!([
                { "axis": "memory", "informational": false,
                  "metrics": { "fields": { "train_ce": 1.1, "init_ce": 3.9 } } },
            ]),
        );
        let advice = advise(&eval, None);
        let r = &advice.recommendations[0];
        assert!(r.action.contains("deprioritize"), "got: {}", r.action);
    }

    #[test]
    fn informational_only_axis_is_skipped() {
        // arithmetic is informational-only here → not a recommendation target.
        let eval = eval_with(
            json!({ "arithmetic": 0.30 }),
            json!([
                { "axis": "arithmetic", "informational": true,
                  "metrics": { "fields": { "train_ce": 1.0, "init_ce": 2.9 } } },
            ]),
        );
        let advice = advise(&eval, None);
        assert!(advice.recommendations.is_empty(), "informational axis should not gate advice");
    }
}
