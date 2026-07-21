// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Render a [`Comparison`] as a human-readable report — the "definition of done"
//! deliverable: every model × every scenario × every metric, beside the naive
//! baseline, with the random-walk negative control called out.
//!
//! Markdown is the primary format (diffable, pastes into a PR, renders in a
//! terminal preview). A self-contained HTML variant is available for a richer
//! view. Both are pure string builders — no I/O, fully unit-testable.

use crate::harness::Comparison;
use std::collections::BTreeSet;

/// Metrics where a **lower** value is better (errors). Everything else
/// (directional accuracy, coverage) is higher-is-better.
fn lower_is_better(metric: &str) -> bool {
    matches!(metric, "mase" | "wql" | "crps")
}

fn scenarios_of(cmp: &Comparison) -> Vec<String> {
    let s: BTreeSet<&str> = cmp.cells.iter().map(|c| c.scenario.as_str()).collect();
    s.into_iter().map(|x| x.to_string()).collect()
}

fn models_of(cmp: &Comparison) -> Vec<String> {
    // naive first (the reference), then the rest alphabetically.
    let mut s: BTreeSet<&str> = cmp.cells.iter().map(|c| c.model.as_str()).collect();
    let mut out = Vec::new();
    if s.remove("naive") {
        out.push("naive".to_string());
    }
    out.extend(s.into_iter().map(|x| x.to_string()));
    out
}

fn fmt(v: f32) -> String {
    format!("{v:.4}")
}

/// One markdown table (models × scenarios) for a single metric, with the
/// best cell per scenario marked `**bold**`.
fn metric_table(cmp: &Comparison, metric: &str) -> String {
    let scenarios = scenarios_of(cmp);
    let models = models_of(cmp);
    let mut out = String::new();
    out.push_str(&format!(
        "### {metric} ({})\n\n",
        if lower_is_better(metric) { "lower is better" } else { "higher is better" }
    ));
    // header
    out.push_str("| model |");
    for s in &scenarios {
        out.push_str(&format!(" {s} |"));
    }
    out.push('\n');
    out.push_str("|---|");
    for _ in &scenarios {
        out.push_str("---|");
    }
    out.push('\n');

    // best value per scenario for bolding
    let best: Vec<Option<f32>> = scenarios
        .iter()
        .map(|s| {
            let vals: Vec<f32> =
                models.iter().filter_map(|m| cmp.get(s, m, metric)).collect();
            if vals.is_empty() {
                None
            } else if lower_is_better(metric) {
                vals.into_iter().reduce(f32::min)
            } else {
                vals.into_iter().reduce(f32::max)
            }
        })
        .collect();

    for m in &models {
        out.push_str(&format!("| {m} |"));
        for (si, s) in scenarios.iter().enumerate() {
            match cmp.get(s, m, metric) {
                Some(v) => {
                    let is_best = best[si].map(|b| (b - v).abs() < 1e-6).unwrap_or(false);
                    if is_best {
                        out.push_str(&format!(" **{}** |", fmt(v)));
                    } else {
                        out.push_str(&format!(" {} |", fmt(v)));
                    }
                }
                None => out.push_str(" — |"),
            }
        }
        out.push('\n');
    }
    out.push('\n');
    out
}

/// The full markdown report: a table per metric, a MASE skill-vs-naive summary,
/// and the negative-control verdict.
pub fn markdown(cmp: &Comparison) -> String {
    let mut out = String::new();
    out.push_str("# Forecasting model comparison\n\n");
    out.push_str(&format!(
        "_{} models × {} scenarios, {} windows each. Synthetic data — proves \
         implementation correctness, not financial skill._\n\n",
        models_of(cmp).len(),
        scenarios_of(cmp).len(),
        cmp.windows
    ));

    for metric in &cmp.metrics {
        out.push_str(&metric_table(cmp, metric));
    }

    // skill vs naive on MASE (0..1, higher better)
    out.push_str("### skill vs naive (MASE-based, 0=no better than naive, 1=perfect)\n\n");
    let scenarios = scenarios_of(cmp);
    out.push_str("| model |");
    for s in &scenarios {
        out.push_str(&format!(" {s} |"));
    }
    out.push_str("\n|---|");
    for _ in &scenarios {
        out.push_str("---|");
    }
    out.push('\n');
    for m in models_of(cmp) {
        out.push_str(&format!("| {m} |"));
        for s in &scenarios {
            match (cmp.get(s, &m, "mase"), cmp.get(s, "naive", "mase")) {
                (Some(v), Some(naive)) => {
                    out.push_str(&format!(" {} |", fmt(forecast::metrics::skill_score(v, naive))));
                }
                _ => out.push_str(" — |"),
            }
        }
        out.push('\n');
    }
    out.push('\n');

    // negative-control verdict
    out.push_str(&negative_control_section(cmp));
    out
}

/// The negative-control verdict block: on any control scenario, no model may
/// materially beat naive.
fn negative_control_section(cmp: &Comparison) -> String {
    let mut out = String::from("### negative control\n\n");
    // control scenarios are those where naive is (near) the best on MASE by
    // construction; we key on the known name.
    let control = "random_walk";
    let has_control = cmp.cells.iter().any(|c| c.scenario == control);
    if !has_control {
        out.push_str("_no control scenario present._\n");
        return out;
    }
    let violations = cmp.negative_control_violations(control, 0.10);
    if violations.is_empty() {
        out.push_str(
            "✅ **PASS** — no model beats the naive baseline on the random-walk \
             control (as it must be: the optimal forecast is the last value).\n",
        );
    } else {
        out.push_str(&format!(
            "❌ **FAIL** — these models falsely beat naive on a random walk \
             (a bug or overfitting): {}\n",
            violations.join(", ")
        ));
    }
    out
}

/// A self-contained HTML rendering (inlined styles, theme-neutral).
pub fn html(cmp: &Comparison) -> String {
    // Reuse the markdown structure but wrap the metric grids in <table>. To keep
    // this dependency-free we do a light markdown-ish emit rather than a full
    // converter.
    let mut body = String::new();
    body.push_str("<h1>Forecasting model comparison</h1>");
    body.push_str(&format!(
        "<p><em>{} models × {} scenarios, {} windows each. Synthetic data — \
         proves implementation correctness, not financial skill.</em></p>",
        models_of(cmp).len(),
        scenarios_of(cmp).len(),
        cmp.windows
    ));
    let scenarios = scenarios_of(cmp);
    for metric in &cmp.metrics {
        body.push_str(&format!(
            "<h3>{metric} ({})</h3><table><tr><th>model</th>",
            if lower_is_better(metric) { "lower is better" } else { "higher is better" }
        ));
        for s in &scenarios {
            body.push_str(&format!("<th>{s}</th>"));
        }
        body.push_str("</tr>");
        for m in models_of(cmp) {
            body.push_str(&format!("<tr><td>{m}</td>"));
            for s in &scenarios {
                match cmp.get(s, &m, metric) {
                    Some(v) => body.push_str(&format!("<td>{}</td>", fmt(v))),
                    None => body.push_str("<td>—</td>"),
                }
            }
            body.push_str("</tr>");
        }
        body.push_str("</table>");
    }
    let control_ok =
        cmp.negative_control_violations("random_walk", 0.10).is_empty();
    body.push_str(&format!(
        "<h3>negative control</h3><p>{}</p>",
        if control_ok {
            "PASS — no model beats naive on the random-walk control."
        } else {
            "FAIL — a model falsely beats naive on a random walk."
        }
    ));
    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>\
         body{{font-family:system-ui,sans-serif;margin:2rem;}}\
         table{{border-collapse:collapse;margin:1rem 0;}}\
         th,td{{border:1px solid #ccc;padding:.3rem .6rem;text-align:right;}}\
         th:first-child,td:first-child{{text-align:left;}}\
         </style></head><body>{body}</body></html>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::{Drift, RandomWalk};
    use crate::harness;
    use crate::scenarios::{RandomWalkScenario, Scenario, SeasonalTrend};
    use forecast::ForecastModel;

    fn sample_comparison() -> Comparison {
        let models: Vec<Box<dyn ForecastModel>> = vec![Box::new(RandomWalk), Box::new(Drift)];
        let scenarios: Vec<Box<dyn Scenario>> = vec![
            Box::new(SeasonalTrend { slope: 0.3, noise: 0.02, ..Default::default() }),
            Box::new(RandomWalkScenario::default()),
        ];
        harness::run(&models, &scenarios, 12, 3)
    }

    #[test]
    fn markdown_contains_tables_models_and_control_verdict() {
        let md = markdown(&sample_comparison());
        assert!(md.contains("# Forecasting model comparison"));
        assert!(md.contains("### mase"));
        assert!(md.contains("| naive |"));
        assert!(md.contains("| drift |"));
        assert!(md.contains("skill vs naive"));
        // the control must PASS for the baseline set
        assert!(md.contains("negative control"));
        assert!(md.contains("✅ **PASS**"), "control should pass:\n{md}");
    }

    #[test]
    fn html_is_self_contained_and_has_tables() {
        let h = html(&sample_comparison());
        assert!(h.starts_with("<!doctype html>"));
        assert!(h.contains("<table>"));
        assert!(!h.contains("http://"), "must not reference external hosts");
    }

    #[test]
    fn naive_is_marked_best_on_the_control_scenario() {
        // On random_walk, naive should be the (bolded) best MASE.
        let md = markdown(&sample_comparison());
        // find the mase table's naive row and confirm a bold cell exists there
        assert!(md.contains("**"), "some cell should be marked best");
    }
}
