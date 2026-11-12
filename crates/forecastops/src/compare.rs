// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The `compare` action: a model against the statistical baselines over the
//! scenario battery - every scenario a seeded synthetic series with a known
//! generating process - rendered as a model x scenario x metric report.
//!
//! The baselines need no checkpoint, so the action is `from_scratch`: with a
//! model's checkpoint resolved (or named) it joins the battery, and without one
//! the battery runs on the baselines alone.
//!
//! The battery carries a negative control: on a random walk nothing can beat
//! the naive forecast, so a model that appears to has found noise. That is a
//! hard gate; a violation fails the action.
//!
//! Swedish Embedded AB implements forecast benchmarks with negative controls
//! that catch a model fitting noise for its clients. If your team needs
//! expertise in time-series model evaluation then you can procure our
//! services by sending an email to info@swedishembedded.com.

use capability::{ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType};
use forecast::ForecastModel;
use serde_json::json;

/// Seeded windows per scenario and model when the caller names none.
const DEFAULT_WINDOWS: i64 = 24;
/// Most windows one comparison may score.
const MAX_WINDOWS: f64 = 10_000.0;

/// The action's schema.
pub fn compare_spec(summary: &str) -> ActionSpec {
    ActionSpec::new("compare", summary)
        .from_scratch()
        .param(ParamSpec::new("windows", ParamType::Int, "seeded windows scored per scenario and model; a foundation model is slow on a CPU, so keep it small there").min(1.0).max(MAX_WINDOWS).default(json!(DEFAULT_WINDOWS)))
        .param(ParamSpec::new("seed", ParamType::Int, "seeds the scenario windows (omit for random; the outcome echoes it)").min(0.0))
        .output(BlobSpec::new("report", Media::Text, "the model x scenario x metric report, as markdown"))
        .output(BlobSpec::new("html", Media::Bytes, "the same report as one self-contained HTML page; meta {format}"))
}

/// Run the battery for `models` (the baselines are always in) and fail if a
/// model falsely beats naive on the random walk.
pub fn run_compare(models: Vec<Box<dyn ForecastModel>>, inv: &Invocation) -> ActionResult {
    let windows = inv.get_i64("windows").unwrap_or(DEFAULT_WINDOWS).max(1) as usize;
    let seed = inv.seed()?.unwrap_or_else(seed::random_seed);
    let mut all = fcbench::baselines::default_set();
    all.extend(models);
    let cmp = fcbench::harness::run(&all, &fcbench::scenarios::default_battery(), windows, seed);
    let markdown = fcbench::report::markdown(&cmp);
    let violations = cmp.negative_control_violations("random_walk", fcbench::harness::CONTROL_Z);
    if !violations.is_empty() {
        return Err(format!("negative-control FAILED: {} falsely beat naive on the random walk\n\n{markdown}", violations.join(", ")));
    }
    Ok(Outcome::new()
        .set("seed", json!(seed))
        .set("windows", json!(windows))
        .blob("report", Blob::new(Media::Text, markdown.into_bytes()))
        .blob("html", Blob::new(Media::Bytes, fcbench::report::html(&cmp).into_bytes()).with_meta(json!({"format": "html"}))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_baselines_alone_pass_their_own_negative_control_and_the_seed_is_echoed() {
        let out = run_compare(Vec::new(), &Invocation::new().set("windows", json!(2)).set("seed", json!(5))).unwrap();
        assert_eq!(out.outputs["seed"], json!(5));
        let report = String::from_utf8(out.blobs["report"].bytes.clone()).unwrap();
        assert!(report.contains("random_walk"), "{report}");
        assert!(String::from_utf8(out.blobs["html"].bytes.clone()).unwrap().contains("<html"), "a self-contained page");
    }
}
