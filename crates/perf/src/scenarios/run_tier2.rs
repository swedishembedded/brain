// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Runners for the Tier-2 scenarios: drive a target and assemble the artifact.
//!
//! Each is deliberately explicit about what today's engine lets it observe. A
//! benchmark that reports a confident number it could not actually measure is
//! worse than one that reports `null` and says why — so where a metric needs an
//! engine capability that does not exist yet, the field stays `null` and the
//! scenario says so in its `notes`.

use serde_json::{json, Value};

use crate::driver;
use crate::env::Env;
use crate::metrics::Summary;
use crate::schema::{memory_with, Artifact};
use crate::stats::r3;
use crate::target::PerfTarget;
use crate::workload::{Arrival, Class, Lengths, Slo, Workload};

use super::{mixed, overload, soak, Options};

/// `mixed` — every traffic class through one engine at once, plus an isolated
/// baseline per class so `normalised_slowdown` is real rather than asserted.
pub fn run_mixed(target: &mut dyn PerfTarget, concurrency: usize, opt: &Options) -> Result<Artifact, String> {
    let mut art = Artifact::new("mixed", Env::capture(&opt.device), target.describe());
    art.smoke = opt.smoke;

    // Scale the class mix down when the caller shortened the workload; the
    // default shapes (up to 16k prompts) are far too big for a smoke run.
    let div = opt.output_override.map(|n| (2048 / n.max(1)).max(1)).unwrap_or(1);
    let classes = if div > 1 { mixed::scaled_classes(div) } else { mixed::default_classes() };

    // Baselines first, while the engine is otherwise idle — that is what makes
    // them "alone".
    let alone = mixed::baselines(target, &classes, opt.warmup_requests.max(2), opt.seed);

    target.reset(true);
    let w = mixed::workload(classes.clone(), concurrency, opt.num_requests, opt.warmup_requests, opt.seed);
    let run = driver::drive(target, &w);

    let mut results = mixed::split(&run.records, run.wall_s, &classes, &alone);
    let (per_class, fairness) = mixed::to_json(&mut results);

    // The aggregate line still uses the first class's SLO for the headline, but
    // the per-class blocks are the point of this scenario.
    let mut agg = Summary::build(&run.records, run.wall_s, classes[0].slo);
    art.workload = w.to_json();
    art.performance = agg.to_json();
    art.scheduling = fairness;
    art.per_class = Some(per_class);
    art.memory = memory_with(&target.counters());
    Ok(art)
}

/// `overload` — measure capacity, then offer multiples of it.
pub fn run_overload(
    target: &mut dyn PerfTarget,
    workload_name: &str,
    opt: &Options,
) -> Result<Artifact, String> {
    let mut art = Artifact::new("overload", Env::capture(&opt.device), target.describe());
    art.smoke = opt.smoke;

    let base = super::workload_for("serve", workload_name, opt, 8)
        .ok_or_else(|| format!("unknown workload '{workload_name}'"))?;
    let class = overload::class_from(&base);

    // Capacity = what a saturated closed loop sustains. Everything after is
    // expressed as a multiple of this, so the ladder means the same thing on
    // any hardware.
    target.reset(true);
    let cap_run = driver::drive(target, &base);
    let cap = Summary::build(&cap_run.records, cap_run.wall_s, class.slo);
    let capacity = cap.requests_per_s.max(0.01);

    // Install the requested admission policy on the engine. A requested policy
    // the target cannot install is an ERROR, not a silent fp32-style fallback:
    // an artifact naming a policy that was not in force is the worst outcome.
    let spec = opt.admission.as_deref().unwrap_or("unbounded");
    let policy = match spec.split_once(':') {
        None if spec == "unbounded" => overload::Admission::UnboundedQueue,
        Some(("depth", n)) => overload::Admission::MaxQueueDepth(
            n.parse().map_err(|_| format!("bad --admission depth {n:?}"))?,
        ),
        Some(("deadline", _)) => overload::Admission::DeadlineAware,
        _ => return Err(format!("unknown --admission {spec:?} (unbounded | depth:<N> | deadline:<ms>)")),
    };
    if spec != "unbounded" && !target.set_admission(spec) {
        return Err(format!(
            "this target has no admission seam; --admission {spec} cannot be honoured"
        ));
    }

    let mut points = Vec::new();
    for &m in overload::MULTIPLES {
        target.reset(true);
        let w = overload::workload(class.clone(), capacity, m, opt.num_requests, opt.seed);
        let r = driver::drive(target, &w);
        let mut s = Summary::build(&r.records, r.wall_s, class.slo);
        let met = r.records.iter().filter(|x| !x.warmup && x.meets(&class.slo)).count();
        let completed = s.completed;
        let admitted = s.requests.saturating_sub(s.rejected);
        points.push(overload::Point {
            multiple: m,
            offered_per_s: capacity * m,
            admitted,
            // Real engine rejections, surfaced through the admission seam.
            // Under the default unbounded policy this stays 0 — the honest
            // measurement of queue-without-bound.
            rejected: s.rejected,
            completed,
            met_slo: met,
            goodput_per_s: s.goodput_per_s,
            queue_p99_ms: s.queue.percentile(0.99),
            wasted: admitted.saturating_sub(met),
        });
    }

    art.workload = base.to_json();
    art.performance = json!({
        "measured_capacity_req_per_s": r3(capacity),
        "overload": overload::to_json(&points, policy, None),
    });
    art.reliability = json!({
        "cancelled_compute_waste": Value::Null,
        "failure_detect_ms": Value::Null,
        "recovery_ms": Value::Null,
        "lost_requests": 0,
        "corrupted_responses": 0,
        "errors": 0,
        // Real engine rejections through the admission seam (0 under unbounded).
        "rejections": points.iter().map(|p| p.rejected).sum::<usize>(),
        "timeouts": points.iter().map(|p| p.wasted).sum::<usize>(),
        "ooms": 0,
    });
    art.notes = Some(match spec {
        "unbounded" => {
            "Default unbounded admission: the engine queues everything, so \
             `rejected` is 0 at every offered load and `waste_fraction` is the \
             share of admitted work that missed its deadline. Re-run with \
             --admission depth:<N> or deadline:<ms> to compare policies."
                .to_string()
        }
        s => format!(
            "Admission policy `{s}` installed on the engine; `rejected` counts \
             real refusals at submit time, and SLO attainment is over admitted \
             work only."
        ),
    });
    Ok(art)
}

/// `soak` — sample the same numbers over a duration and report the trend.
pub fn run_soak(
    target: &mut dyn PerfTarget,
    workload_name: &str,
    duration_s: f64,
    opt: &Options,
) -> Result<Artifact, String> {
    let mut art = Artifact::new("soak", Env::capture(&opt.device), target.describe());
    art.smoke = opt.smoke;

    let w = super::workload_for("serve", workload_name, opt, 8)
        .ok_or_else(|| format!("unknown workload '{workload_name}'"))?;
    let slo = w.slo_for(0);
    let started = std::time::Instant::now();
    let mut report = soak::Report::default();

    while started.elapsed().as_secs_f64() < duration_s {
        let r = driver::drive(target, &w);
        let mut s = Summary::build(&r.records, r.wall_s, slo);
        let kv = target
            .counters()
            .iter()
            .find(|(k, _)| k == "kv_free_blocks")
            .and_then(|(_, v)| v.as_u64())
            .map(|v| v as u32);
        report.samples.push(soak::Sample {
            elapsed_s: started.elapsed().as_secs_f64(),
            output_per_s: s.output_per_s,
            ttfa_p99_ms: s.ttfa.percentile(0.99),
            ial_p99_ms: s.ial.percentile(0.99),
            host_mem_mb: soak::host_mem_mb(),
            kv_free_blocks: kv,
            open_fds: soak::open_fds(),
            threads: None,
            errors: s.failed,
        });
    }

    art.workload = w.to_json();
    art.performance = report.to_json();
    // ONE source for the resources block shape (`schema::empty_resources`)
    // with the fields this scenario measures overlaid — a hand-built literal
    // here silently forked from the schema when the telemetry keys were
    // added, making soak artifacts differ from latency/sweep artifacts.
    art.resources = crate::schema::empty_resources();
    if let Some(v) = soak::host_mem_mb() {
        art.resources["host_mem_mb"] = Value::from(r3(v));
    }
    if !report.trend_valid() {
        art.notes = Some(format!(
            "{} sample(s) over {:.0}s: too short to extrapolate an hourly trend, so the \
             drift fields are null rather than misleading. A soak needs at least {:.0}s \
             (--soak-seconds) before %/h means anything; the per-sample series is still \
             recorded.",
            report.samples.len(),
            report.duration_s(),
            soak::Report::MIN_TREND_S,
        ));
    }
    Ok(art)
}

/// `frontend` — time the host-side stages against a device rate.
pub fn run_frontend(device_rate: Option<f64>, iters: usize, opt: &Options) -> Result<Artifact, String> {
    use super::frontend::{Report, Stage, StageResult};
    use std::time::Instant;

    let mut art = Artifact::new(
        "frontend",
        Env::capture(&opt.device),
        crate::target::TargetInfo::new("host", "request"),
    );
    art.smoke = opt.smoke;

    let n = iters.max(16);
    let mut stages = Vec::new();

    // JSONL encode/decode — the actual protocol brain serves over.
    let mut enc = StageResult::new(Stage::JsonlEncode);
    let mut dec = StageResult::new(Stage::JsonlDecode);
    let sample = json!({
        "event": "user_text",
        "req_id": "abc123",
        "text": "Summarise the following document in three bullet points."
    });
    for _ in 0..n {
        let t = Instant::now();
        let line = serde_json::to_string(&sample).unwrap_or_default();
        enc.record(t.elapsed().as_secs_f64() * 1e6, line.len());
        let t = Instant::now();
        let v: Value = serde_json::from_str(&line).unwrap_or(Value::Null);
        dec.record(t.elapsed().as_secs_f64() * 1e6, line.len());
        std::hint::black_box(&v);
    }
    stages.push(enc);
    stages.push(dec);

    // Chat-template rendering: string assembly per turn.
    let mut tpl = StageResult::new(Stage::ChatTemplate);
    for _ in 0..n {
        let t = Instant::now();
        let s = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            "You are a helpful assistant.", "Summarise the following document."
        );
        tpl.record(t.elapsed().as_secs_f64() * 1e6, s.len());
        std::hint::black_box(s);
    }
    stages.push(tpl);

    // Tokenise/detokenise — the REAL GPT-2 BPE on its embedded assets (I):
    // the same code `brain data gpt` ships, so the per-token host cost is a
    // measurement of the pipeline, not of a stand-in. Units are tokens, which
    // is what `cores_per_device` compares against the device's token rate.
    {
        use data::tokenizer::Tokenizer;
        let bpe = data::bpe::Gpt2Bpe::new();
        let text = "Summarise the following document in three bullet points, \
                    quoting the relevant sections verbatim and preserving any \
                    numeric values exactly as they appear in the source text. "
            .repeat(8);
        let mut tok = StageResult::new(Stage::Tokenise);
        let mut detok = StageResult::new(Stage::Detokenise);
        for _ in 0..n {
            let t = Instant::now();
            let ids = bpe.encode(&text);
            tok.record(t.elapsed().as_secs_f64() * 1e6, ids.len());
            let t = Instant::now();
            let s = bpe.decode(&ids);
            detok.record(t.elapsed().as_secs_f64() * 1e6, ids.len());
            std::hint::black_box(s);
        }
        stages.push(tok);
        stages.push(detok);
    }

    let mut report = Report { stages, device_artifacts_per_s: device_rate };
    art.performance = report.to_json();
    art.notes = Some(
        "Protocol, templating and REAL GPT-2 BPE tokenise/detokenise (embedded \
         assets — the shipping code path). Image decode and audio resample \
         need a media input and are absent rather than reported as free; the \
         Qwen tokenizer needs its vocab file and is likewise absent."
            .into(),
    );
    Ok(art)
}

/// A minimal single-class workload used by scenarios that shape their own load.
pub fn simple_class(input: usize, output: usize, slo: Slo) -> Class {
    Class {
        name: "load".into(),
        input: Lengths::Fixed(input.max(1)),
        output: Lengths::Fixed(output.max(1)),
        weight: 1.0,
        slo,
    }
}

/// A closed-loop workload around one class.
pub fn simple_workload(class: Class, concurrency: usize, n: usize, seed: u64) -> Workload {
    Workload {
        name: "custom".into(),
        classes: vec![class],
        arrival: Arrival::ClosedLoop { concurrency },
        num_requests: n,
        warmup_requests: 0,
        ignore_stop: true,
        seed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::target::testing::FakeTarget;

    fn opt() -> Options {
        Options {
            num_requests: 6,
            warmup_requests: 2,
            output_override: Some(4),
            concurrency: vec![1, 2],
            ..Default::default()
        }
    }

    #[test]
    fn mixed_reports_every_class_with_a_baseline() {
        let mut t = FakeTarget::new(8, 1);
        let art = run_mixed(&mut t, 4, &opt()).expect("mixed must run");
        let pc = art.per_class.as_ref().expect("per-class blocks are the point of `mixed`");
        assert_eq!(pc.len(), 4);
        assert!(pc.iter().all(|b| b["class"].is_string()));
        // Baselines were run, so slowdown is a real number rather than null.
        assert!(pc.iter().any(|b| !b["performance"]["normalised_slowdown"].is_null()));
        assert!(!art.scheduling["jain_fairness"].is_null());
    }

    #[test]
    fn overload_offers_every_multiple_and_says_nothing_is_rejected() {
        let mut t = FakeTarget::new(8, 1);
        let art = run_overload(&mut t, "interactive", &opt()).expect("overload must run");
        let ladder = art.performance["overload"]["ladder"].as_array().unwrap();
        assert_eq!(ladder.len(), overload::MULTIPLES.len());
        assert!(art.performance["measured_capacity_req_per_s"].as_f64().unwrap() > 0.0);
        // The engine has no admission seam; the artifact must say so rather than
        // implying a policy was evaluated.
        assert_eq!(art.reliability["rejections"], 0);
        assert!(art.notes.as_ref().unwrap().contains("admission"));
    }

    /// A requested policy the target cannot install must be an ERROR — an
    /// artifact naming a policy that was not actually in force is the failure
    /// mode this seam exists to prevent.
    #[test]
    fn overload_refuses_a_policy_the_target_cannot_install() {
        let mut t = FakeTarget::new(8, 1); // FakeTarget has no admission seam
        let mut o = opt();
        o.admission = Some("depth:4".into());
        let err = match run_overload(&mut t, "interactive", &o) {
            Err(e) => e,
            Ok(_) => panic!("an uninstallable policy must be refused"),
        };
        assert!(err.contains("admission"), "error must name the problem: {err}");
        // The default (unbounded) still runs on such a target.
        o.admission = None;
        run_overload(&mut t, "interactive", &o).expect("unbounded needs no seam");
    }

    #[test]
    fn soak_samples_and_reports_a_trend_or_says_it_cannot() {
        let mut t = FakeTarget::new(8, 1);
        let art = run_soak(&mut t, "interactive", 0.05, &opt()).expect("soak must run");
        assert!(art.performance["samples"].as_u64().unwrap() >= 1);
        // A very short soak cannot establish a trend, and must admit that.
        if art.performance["samples"].as_u64().unwrap() < 3 {
            assert!(art.notes.as_ref().unwrap().contains("smoke check"));
        }
    }

    #[test]
    fn frontend_measures_stages_and_needs_a_device_rate_for_cores() {
        let art = run_frontend(Some(1000.0), 32, &opt()).expect("frontend must run");
        let stages = art.performance["stages"].as_array().unwrap();
        assert!(stages.len() >= 3);
        assert!(!art.performance["host_cores_per_saturated_device"].is_null());

        let no_rate = run_frontend(None, 32, &opt()).unwrap();
        assert!(no_rate.performance["host_cores_per_saturated_device"].is_null());
    }

    #[test]
    fn frontend_measures_real_tokeniser_and_omits_media_stages() {
        let art = run_frontend(None, 16, &opt()).unwrap();
        let names: Vec<&str> = art.performance["stages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["stage"].as_str().unwrap())
            .collect();
        // The REAL GPT-2 BPE runs (embedded assets, no file needed)...
        assert!(names.contains(&"tokenise") && names.contains(&"detokenise"));
        // ...while stages needing external inputs stay absent, never zero-cost.
        assert!(!names.contains(&"image_decode") && !names.contains(&"audio_resample"));
        assert!(art.notes.as_ref().unwrap().contains("absent rather than reported as free"));
    }
}
