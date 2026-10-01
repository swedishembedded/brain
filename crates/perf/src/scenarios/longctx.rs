// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `longctx` - what one GPU sustains on one model at a long context: real cold
//! prefill speed, single-stream decode, batched decode at that context, and the
//! batch size at which the model stops fitting in device memory.
//!
//! Swedish Embedded AB implements long-context inference serving and the
//! measurement that shows what a given GPU can really sustain for its clients.
//! If your team needs expertise in sizing and tuning LLM serving on a specific
//! accelerator then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The scenario is model-agnostic: it drives a [`LongContextEngine`], the
//! narrow seam a model family implements (this crate depends on no model
//! crate; the CLI wires engines to the trait). Everything below the seam is the
//! engine's business - how it sizes its caches, how it batches - and everything
//! above it (ladder, boundary, statistics, artifact) is shared, so two model
//! families are measured by the same code and their artifacts compare.
//!
//! **GPU only, by construction.** [`LongContextEngine::plan`] answers "do the
//! weights plus the caches of `batch` sequences at `context` fit in device
//! memory", and a `false` ends the sweep: that batch size is the out-of-memory
//! boundary. There is no host-memory fallback to spill to, and every artifact
//! records `host_offload: false`.
//!
//! **Context.** An engine states whether the context its decode step attends
//! over is [`ContextKind::Real`] (built by prefilling) or
//! [`ContextKind::Synthetic`] (fresh caches, decode work at a real position).
//! Decode throughput does not depend on what the cached keys and values hold,
//! only on how many there are, so a synthetic context measures the same
//! kernels and memory traffic at a fraction of the setup cost; its logits mean
//! nothing and the artifact says so. Prefill speed is always real.

use serde_json::{json, Value};

use crate::env::Env;
use crate::schema::Artifact;
use crate::stats::r3;
use crate::target::TargetInfo;

/// Scenario name as registered in [`super::SCENARIOS`].
pub const NAME: &str = "longctx";

/// Default batch ladder: doubling until the engine stops fitting.
pub const DEFAULT_LADDER: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128, 256];

/// Decode steps run and discarded before timing starts: kernel compilation and
/// graph capture are not decode time.
const WARMUP_STEPS: u32 = 2;

/// Whether the decode context holds real activations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContextKind {
    /// Built by prefilling real tokens.
    Real,
    /// Fresh caches; decode work is that of a real context of the same length.
    Synthetic,
}

impl ContextKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ContextKind::Real => "real",
            ContextKind::Synthetic => "synthetic",
        }
    }
}

/// What an engine is, for the artifact's fingerprint.
#[derive(Clone, Debug)]
pub struct EngineInfo {
    pub model: String,
    /// Weight storage tier, e.g. `int8`, `fp32`.
    pub weight_tier: String,
    /// KV / recurrent-state precision, e.g. `int8`, `fp32`.
    pub kv_precision: String,
    pub backend: String,
    pub devices: Vec<String>,
    pub context_kind: ContextKind,
}

/// Answer to "does this sizing fit in device memory".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fit {
    pub fits: bool,
    /// Bytes the sizing needs on the device (weights + caches + scratch).
    pub needed_bytes: u64,
    /// Bytes the devices offer for it (after the engine's reserve).
    pub usable_bytes: u64,
}

/// The seam a model family implements. One engine value is reused across the
/// sweep: `load` builds the resident model for one sizing, `unload` frees it.
pub trait LongContextEngine {
    fn describe(&self) -> EngineInfo;

    /// Longest context the model supports, `None` when it has no stated limit.
    fn max_context(&self) -> Option<u32>;

    /// Do the weights plus the caches of `batch` sequences of `context` tokens
    /// fit on the GPU(s)? Pure estimate: allocates nothing. GPU only - an
    /// engine must not count host memory as capacity.
    fn plan(&mut self, batch: u32, context: u32) -> Result<Fit, String>;

    /// Build the resident model for `batch` sequences of up to `context`
    /// tokens. An error here (allocation failure) is also an out-of-memory
    /// boundary.
    fn load(&mut self, batch: u32, context: u32) -> Result<(), String>;

    /// Real, cold prefill of `prompt_tokens` tokens into one sequence, waited
    /// for the device; returns seconds. Never a continuation of earlier work.
    fn prefill(&mut self, prompt_tokens: u32) -> Result<f64, String>;

    /// One batched decode step, one row per entry of `positions` (the
    /// position of the token being decoded), device-synchronised; returns
    /// milliseconds.
    fn decode_step(&mut self, positions: &[u32]) -> Result<f64, String>;

    /// Device memory in use now, as the driver reports it; `None` when the
    /// platform has no way to say.
    fn device_used_bytes(&self) -> Option<u64> {
        None
    }

    /// Free everything `load` built.
    fn unload(&mut self);
}

/// How to run the scenario.
#[derive(Clone, Debug)]
pub struct Options {
    /// Context length the decode sweep is measured at.
    pub context: u32,
    /// Batch sizes to try, ascending.
    pub ladder: Vec<u32>,
    /// Prompt lengths for the prefill table; empty skips it.
    pub prefill: Vec<u32>,
    /// Timed decode steps per batch size.
    pub steps: u32,
    /// The device label recorded in the artifact.
    pub device: String,
    pub smoke: bool,
}

/// Parse a comma-separated list of positive integers into an ascending,
/// de-duplicated ladder. Rejects empty lists, zero and non-numbers loudly: a
/// silently dropped rung would measure a different sweep than was asked for.
pub fn parse_list(spec: &str, what: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let n: u32 = part.parse().map_err(|_| format!("{what}: {part:?} is not a positive integer"))?;
        if n == 0 {
            return Err(format!("{what}: 0 is not a valid value"));
        }
        out.push(n);
    }
    if out.is_empty() {
        return Err(format!("{what}: needs at least one value"));
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// The largest `b` in `[lo_fit, hi_nofit)` for which `fits(b)`, given that
/// `lo_fit` fits and `hi_nofit` does not (fitting is monotone in batch).
/// Planner-only: it refines the boundary between two measured rungs without
/// loading anything.
pub fn bisect_fit(lo_fit: u32, hi_nofit: u32, mut fits: impl FnMut(u32) -> bool) -> u32 {
    let (mut lo, mut hi) = (lo_fit, hi_nofit);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    lo
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// Why a sweep stopped before the end of its ladder.
struct Boundary {
    batch: u32,
    reason: &'static str,
    needed_bytes: Option<u64>,
    usable_bytes: Option<u64>,
    error: Option<String>,
}

fn opt_u64(v: Option<u64>) -> Value {
    v.map(Value::from).unwrap_or(Value::Null)
}

/// Run the scenario on `engine` and return its artifact.
pub fn run(engine: &mut dyn LongContextEngine, opt: &Options) -> Result<Artifact, String> {
    let info = engine.describe();
    validate(engine, opt)?;

    // Batch 1 is the single-stream figure and anchors the sweep, so it is
    // always measured whether or not the caller listed it.
    let mut ladder = opt.ladder.clone();
    if ladder.first() != Some(&1) {
        ladder.insert(0, 1);
    }

    let position = opt.context - (opt.steps + WARMUP_STEPS);

    let prefill = run_prefill(engine, opt);

    let mut rows: Vec<Value> = Vec::new();
    let mut boundary: Option<Boundary> = None;
    let mut last_fit: Option<u32> = None;
    for &batch in &ladder {
        let fit = engine.plan(batch, opt.context)?;
        if !fit.fits {
            boundary = Some(Boundary { batch, reason: "planned_over_capacity", needed_bytes: Some(fit.needed_bytes), usable_bytes: Some(fit.usable_bytes), error: None });
            break;
        }
        if let Err(e) = engine.load(batch, opt.context) {
            engine.unload();
            boundary = Some(Boundary { batch, reason: "load_failed", needed_bytes: Some(fit.needed_bytes), usable_bytes: Some(fit.usable_bytes), error: Some(e) });
            break;
        }
        let measured = measure_batch(engine, batch, position, opt.steps);
        let used = engine.device_used_bytes();
        engine.unload();
        match measured {
            Ok(mut times) => {
                let min = times.iter().copied().fold(f64::INFINITY, f64::min);
                let ms = median(&mut times);
                let total = batch as f64 * 1e3 / ms;
                rows.push(json!({
                    "batch": batch,
                    "planned_bytes": fit.needed_bytes,
                    "usable_bytes": fit.usable_bytes,
                    "device_used_bytes": opt_u64(used),
                    "ms_per_step_median": r3(ms),
                    "ms_per_step_min": r3(min),
                    "tok_per_s_total": r3(total),
                    "tok_per_s_per_stream": r3(total / batch as f64),
                }));
                last_fit = Some(batch);
            }
            Err(e) => {
                boundary = Some(Boundary { batch, reason: "decode_failed", needed_bytes: Some(fit.needed_bytes), usable_bytes: Some(fit.usable_bytes), error: Some(e) });
                break;
            }
        }
    }

    // Between the last rung that ran and the first that did not, the planner
    // can still say how far the real limit is, without loading anything.
    let max_planned = match (&boundary, last_fit) {
        (Some(b), Some(lo)) if b.batch > lo + 1 => {
            Some(bisect_fit(lo, b.batch, |n| engine.plan(n, opt.context).map(|f| f.fits).unwrap_or(false)))
        }
        (Some(_), Some(lo)) => Some(lo),
        _ => None,
    };

    Ok(build_artifact(&info, opt, position, ladder, prefill, rows, boundary, max_planned))
}

fn validate(engine: &dyn LongContextEngine, opt: &Options) -> Result<(), String> {
    if opt.steps == 0 {
        return Err("longctx: --steps must be at least 1".into());
    }
    let need = opt.steps + WARMUP_STEPS;
    if opt.context <= need {
        return Err(format!("longctx: --context {} is too short for {} warm-up + timed decode steps", opt.context, need));
    }
    if let Some(max) = engine.max_context() {
        let longest = opt.prefill.iter().copied().max().unwrap_or(0).max(opt.context);
        if longest > max {
            return Err(format!("longctx: {longest} tokens exceeds this model's maximum context of {max}"));
        }
    }
    Ok(())
}

/// Warm up, then time a cold prefill of each requested length. `Null` when no
/// lengths were asked for; a table that stopped early says why in `error`.
fn run_prefill(engine: &mut dyn LongContextEngine, opt: &Options) -> Value {
    let Some(&longest) = opt.prefill.iter().max() else {
        return Value::Null;
    };
    let cap = opt.context.max(longest);
    let fit = match engine.plan(1, cap) {
        Ok(f) => f,
        Err(e) => return json!({ "rows": [], "error": e }),
    };
    if !fit.fits {
        return json!({ "rows": [], "error": format!("a one-sequence instance at capacity {cap} does not fit in GPU memory") });
    }
    if let Err(e) = engine.load(1, cap) {
        engine.unload();
        return json!({ "rows": [], "error": e });
    }
    let mut rows = Vec::new();
    let mut error = Value::Null;
    // One short pass first: kernel compilation is not prefill time.
    let warm = opt.prefill.iter().copied().min().unwrap_or(1).min(512);
    let result = engine.prefill(warm).map(|_| ()).and_then(|()| {
        for &n in &opt.prefill {
            let secs = engine.prefill(n)?;
            rows.push(json!({ "prompt_tokens": n, "seconds": r3(secs), "tok_per_s": r3(n as f64 / secs) }));
        }
        Ok(())
    });
    if let Err(e) = result {
        error = Value::from(e);
    }
    engine.unload();
    json!({ "rows": rows, "error": error })
}

/// Warm-up then `steps` timed decode steps at consecutive positions starting at
/// `position`, every row of the batch at the same position.
fn measure_batch(engine: &mut dyn LongContextEngine, batch: u32, position: u32, steps: u32) -> Result<Vec<f64>, String> {
    for w in 0..WARMUP_STEPS {
        engine.decode_step(&vec![position + w; batch as usize])?;
    }
    (0..steps).map(|s| engine.decode_step(&vec![position + WARMUP_STEPS + s; batch as usize])).collect()
}

#[allow(clippy::too_many_arguments)]
fn build_artifact(
    info: &EngineInfo,
    opt: &Options,
    position: u32,
    ladder: Vec<u32>,
    prefill: Value,
    rows: Vec<Value>,
    boundary: Option<Boundary>,
    max_planned: Option<u32>,
) -> Artifact {
    let mut target = TargetInfo::new(&info.model, "token")
        .with("weight_tier", info.weight_tier.clone().into())
        .with("kv_precision", info.kv_precision.clone().into())
        .with("backend", info.backend.clone().into())
        .with("devices", info.devices.clone().into())
        .with("host_offload", false.into());
    target.quant = Some(info.weight_tier.clone());

    let mut art = Artifact::new(NAME, Env::capture(&opt.device), target);
    art.smoke = opt.smoke;
    art.workload = json!({
        "name": NAME,
        "context_tokens": opt.context,
        "context": info.context_kind.as_str(),
        "decode_position": position,
        "steps": opt.steps,
        "warmup_steps": WARMUP_STEPS,
        "ladder": ladder,
        "prefill_prompt_tokens": opt.prefill,
        "host_offload": false,
    });

    let by_total = |a: &&Value, b: &&Value| a["tok_per_s_total"].as_f64().unwrap_or(0.0).total_cmp(&b["tok_per_s_total"].as_f64().unwrap_or(0.0));
    let best = rows.iter().max_by(by_total);
    let single = rows.iter().find(|r| r["batch"] == 1);
    art.performance = json!({
        "prefill": prefill,
        "single_stream_decode": single.map(|r| json!({
            "tok_per_s": r["tok_per_s_total"],
            "ms_per_step_median": r["ms_per_step_median"],
        })).unwrap_or(Value::Null),
        "max_sustained_throughput": best.map(|r| json!({
            "batch": r["batch"],
            "tok_per_s_total": r["tok_per_s_total"],
            "tok_per_s_per_stream": r["tok_per_s_per_stream"],
        })).unwrap_or(Value::Null),
        "oom_boundary": boundary.as_ref().map(|b| json!({
            "first_failing_batch": b.batch,
            "reason": b.reason,
            "needed_bytes": opt_u64(b.needed_bytes),
            "usable_bytes": opt_u64(b.usable_bytes),
            "error": b.error.clone().map(Value::from).unwrap_or(Value::Null),
            "max_fitting_batch_by_plan": max_planned.map(Value::from).unwrap_or(Value::Null),
        })).unwrap_or(Value::Null),
    });

    let peak = rows.iter().filter_map(|r| r["device_used_bytes"].as_u64()).max();
    art.memory = crate::schema::memory_with(&[("peak_device_mb".to_string(), peak.map(|b| Value::from(b / (1 << 20))).unwrap_or(Value::Null))]);
    art.curve = Some(rows);
    if info.context_kind == ContextKind::Synthetic {
        art.notes = Some(
            "decode context is SYNTHETIC: fresh caches, so each step does the work of a real context of this length but its logits are not meaningful. Prefill timings are real.".into(),
        );
    }
    art
}

/// The human table.
pub fn render(art: &Artifact) -> String {
    let w = &art.workload;
    let p = &art.performance;
    let mut s = format!(
        "\n{NAME} - {} on {}\n  weights {}, kv {}, backend {}, context {} tokens ({}), host offload: none\n",
        art.target.model,
        art.env.label(),
        art.target.quant.as_deref().unwrap_or("-"),
        art.target.config.iter().find(|(k, _)| k == "kv_precision").and_then(|(_, v)| v.as_str()).unwrap_or("-"),
        art.env.backend,
        w["context_tokens"],
        w["context"].as_str().unwrap_or("-"),
    );
    if !art.valid {
        s.push_str(&format!("  ! INVALID: {}\n", art.invalid_reason.clone().unwrap_or_default()));
    }
    let f = |v: &Value, d: usize| v.as_f64().map(|x| format!("{x:.d$}")).unwrap_or_else(|| "-".into());
    let gib = |v: &Value| v.as_f64().map(|x| format!("{:.1}", x / (1u64 << 30) as f64)).unwrap_or_else(|| "-".into());

    if let Some(rows) = p["prefill"]["rows"].as_array().filter(|r| !r.is_empty()) {
        s.push_str(&format!("\nprefill (real, cold, one sequence)\n{:>10} {:>10} {:>12}\n", "prompt tok", "seconds", "tok/s"));
        for r in rows {
            s.push_str(&format!("{:>10} {:>10} {:>12}\n", r["prompt_tokens"], f(&r["seconds"], 2), f(&r["tok_per_s"], 1)));
        }
    }
    if let Some(e) = p["prefill"]["error"].as_str() {
        s.push_str(&format!("  prefill stopped: {e}\n"));
    }

    s.push_str(&format!(
        "\ndecode at context {} ({} steps per batch size)\n{:>5} {:>10} {:>10} {:>11} {:>11} {:>13}\n",
        w["context_tokens"], w["steps"], "batch", "plan GiB", "used GiB", "ms/step", "tok/s total", "tok/s/stream"
    ));
    for r in art.curve.iter().flatten() {
        s.push_str(&format!(
            "{:>5} {:>10} {:>10} {:>11} {:>11} {:>13}\n",
            r["batch"],
            gib(&r["planned_bytes"]),
            gib(&r["device_used_bytes"]),
            f(&r["ms_per_step_median"], 2),
            f(&r["tok_per_s_total"], 1),
            f(&r["tok_per_s_per_stream"], 2),
        ));
    }
    if let Some(r) = p["single_stream_decode"].as_object() {
        s.push_str(&format!("\nsingle stream: {} tok/s\n", f(&r["tok_per_s"], 1)));
    }
    if let Some(r) = p["max_sustained_throughput"].as_object() {
        s.push_str(&format!("maximum sustained throughput: {} tok/s total at batch {}\n", f(&r["tok_per_s_total"], 1), r["batch"]));
    }
    match p["oom_boundary"].as_object() {
        Some(b) => {
            s.push_str(&format!("out-of-memory boundary: batch {} ({})", b["first_failing_batch"], b["reason"].as_str().unwrap_or("?")));
            if let (Some(n), Some(u)) = (b["needed_bytes"].as_u64(), b["usable_bytes"].as_u64()) {
                s.push_str(&format!(": needs {:.1} GiB, {:.1} GiB usable", n as f64 / (1u64 << 30) as f64, u as f64 / (1u64 << 30) as f64));
            }
            if let Some(e) = b["error"].as_str() {
                s.push_str(&format!(": {e}"));
            }
            s.push('\n');
            if let Some(m) = b["max_fitting_batch_by_plan"].as_u64() {
                s.push_str(&format!("largest batch the planner admits: {m}\n"));
            }
        }
        None => s.push_str("no out-of-memory boundary reached within the ladder\n"),
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake engine with a hard byte capacity: weights plus `per_seq` bytes
    /// per sequence. Decode cost grows with batch, so there is a throughput peak.
    struct Fake {
        weights: u64,
        per_seq: u64,
        usable: u64,
        max_ctx: Option<u32>,
        loaded: Option<u32>,
        loads: Vec<u32>,
        fail_load_at: Option<u32>,
        used: Option<u64>,
        kind: ContextKind,
    }

    impl Fake {
        fn new() -> Fake {
            Fake { weights: 100, per_seq: 10, usable: 175, max_ctx: None, loaded: None, loads: vec![], fail_load_at: None, used: None, kind: ContextKind::Synthetic }
        }
    }

    impl LongContextEngine for Fake {
        fn describe(&self) -> EngineInfo {
            EngineInfo { model: "fake".into(), weight_tier: "int8".into(), kv_precision: "fp32".into(), backend: "test".into(), devices: vec!["gpu0".into()], context_kind: self.kind }
        }
        fn max_context(&self) -> Option<u32> {
            self.max_ctx
        }
        fn plan(&mut self, batch: u32, _context: u32) -> Result<Fit, String> {
            let needed = self.weights + self.per_seq * batch as u64;
            Ok(Fit { fits: needed <= self.usable, needed_bytes: needed, usable_bytes: self.usable })
        }
        fn load(&mut self, batch: u32, _context: u32) -> Result<(), String> {
            if self.fail_load_at == Some(batch) {
                return Err("allocation failed".into());
            }
            self.loads.push(batch);
            self.loaded = Some(batch);
            Ok(())
        }
        fn prefill(&mut self, prompt_tokens: u32) -> Result<f64, String> {
            self.loaded.ok_or("not loaded")?;
            Ok(prompt_tokens as f64 / 1000.0)
        }
        fn decode_step(&mut self, positions: &[u32]) -> Result<f64, String> {
            assert_eq!(Some(positions.len() as u32), self.loaded, "decode batch must match what was loaded");
            Ok(10.0 + positions.len() as f64)
        }
        fn device_used_bytes(&self) -> Option<u64> {
            self.used
        }
        fn unload(&mut self) {
            self.loaded = None;
        }
    }

    fn opts() -> Options {
        Options { context: 1024, ladder: vec![1, 2, 4, 8, 16], prefill: vec![], steps: 4, device: "gpu".into(), smoke: false }
    }

    #[test]
    fn ladder_parsing_sorts_dedups_and_rejects_bad_input() {
        assert_eq!(parse_list("4, 1,2,2", "--ladder").unwrap(), vec![1, 2, 4]);
        assert!(parse_list("", "--ladder").is_err());
        assert!(parse_list("1,0", "--ladder").is_err());
        assert!(parse_list("1,x", "--ladder").is_err());
        assert!(parse_list("-1", "--ladder").is_err());
    }

    #[test]
    fn bisect_finds_the_largest_fitting_batch() {
        for cap in 1..40u32 {
            assert_eq!(bisect_fit(1, 64, |b| b <= cap), cap.max(1), "cap {cap}");
        }
    }

    #[test]
    fn the_sweep_stops_at_the_first_batch_that_does_not_fit_and_refines_it() {
        // 100 + 10*b <= 175 -> batch 7 is the largest that fits; the ladder
        // rungs are 1,2,4,8,16 so 8 is the first that does not.
        let mut e = Fake::new();
        let art = run(&mut e, &opts()).unwrap();
        let curve = art.curve.as_ref().unwrap();
        assert_eq!(curve.iter().map(|r| r["batch"].as_u64().unwrap()).collect::<Vec<_>>(), vec![1, 2, 4]);
        assert_eq!(e.loads, vec![1, 2, 4], "a batch that does not fit is never loaded");
        let b = &art.performance["oom_boundary"];
        assert_eq!(b["first_failing_batch"], 8);
        assert_eq!(b["reason"], "planned_over_capacity");
        assert_eq!(b["needed_bytes"], 180);
        assert_eq!(b["usable_bytes"], 175);
        assert_eq!(b["max_fitting_batch_by_plan"], 7);
    }

    #[test]
    fn a_load_failure_is_a_boundary_and_carries_the_error() {
        let mut e = Fake::new();
        e.fail_load_at = Some(4);
        let art = run(&mut e, &opts()).unwrap();
        let b = &art.performance["oom_boundary"];
        assert_eq!(b["first_failing_batch"], 4);
        assert_eq!(b["reason"], "load_failed");
        assert_eq!(b["error"], "allocation failed");
        assert_eq!(art.curve.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn no_boundary_within_the_ladder_is_null_not_zero() {
        let mut e = Fake::new();
        e.usable = 10_000;
        let art = run(&mut e, &opts()).unwrap();
        assert!(art.performance["oom_boundary"].is_null());
        assert_eq!(art.curve.as_ref().unwrap().len(), 5);
    }

    #[test]
    fn batch_one_is_always_measured_and_gives_the_single_stream_figure() {
        let mut e = Fake::new();
        let mut o = opts();
        o.ladder = vec![2, 4];
        let art = run(&mut e, &o).unwrap();
        assert_eq!(art.curve.as_ref().unwrap()[0]["batch"], 1);
        let s = &art.performance["single_stream_decode"];
        assert_eq!(s["ms_per_step_median"], 11.0);
        assert!((s["tok_per_s"].as_f64().unwrap() - 1e3 / 11.0).abs() < 0.01);
    }

    #[test]
    fn max_sustained_is_the_best_total_not_the_largest_batch() {
        let mut e = Fake::new();
        e.usable = 10_000;
        let art = run(&mut e, &opts()).unwrap();
        // total = b * 1000 / (10 + b): strictly increasing in b, so the best is the last.
        assert_eq!(art.performance["max_sustained_throughput"]["batch"], 16);
        let curve = art.curve.as_ref().unwrap();
        let t = |i: usize| curve[i]["tok_per_s_total"].as_f64().unwrap();
        assert!(t(4) > t(0));
    }

    #[test]
    fn unmeasured_fields_are_null_never_zero() {
        // The engine can not report device use, no prefill table was asked for,
        // and batch 1 does not fit: nothing may read as 0.
        let mut e = Fake::new();
        e.usable = 50;
        let art = run(&mut e, &opts()).unwrap();
        let j = art.to_json();
        assert!(j["performance"]["prefill"].is_null());
        assert!(j["performance"]["single_stream_decode"].is_null());
        assert!(j["performance"]["max_sustained_throughput"].is_null());
        assert_eq!(j["curve"], json!([]));
        assert!(j["memory"]["peak_device_mb"].is_null());
        assert_eq!(j["performance"]["oom_boundary"]["first_failing_batch"], 1);
        assert!(j["performance"]["oom_boundary"]["max_fitting_batch_by_plan"].is_null());

        let mut e = Fake::new();
        let art = run(&mut e, &opts()).unwrap();
        assert!(art.curve.as_ref().unwrap().iter().all(|r| r["device_used_bytes"].is_null()), "no driver reading must stay null");
        assert!(art.to_json()["memory"]["peak_device_mb"].is_null());
    }

    #[test]
    fn measured_device_memory_reaches_the_row_and_the_peak() {
        let mut e = Fake::new();
        e.used = Some(3 << 30);
        let art = run(&mut e, &opts()).unwrap();
        assert_eq!(art.curve.as_ref().unwrap()[0]["device_used_bytes"], 3u64 << 30);
        assert_eq!(art.to_json()["memory"]["peak_device_mb"], 3072);
    }

    #[test]
    fn the_artifact_records_the_fingerprint_and_the_honesty_labels() {
        let mut e = Fake::new();
        let art = run(&mut e, &opts()).unwrap();
        let j = art.to_json();
        assert_eq!(j["schema"], "brain.perf/1");
        assert_eq!(j["scenario"], "longctx");
        assert_eq!(j["workload"]["context"], "synthetic");
        assert_eq!(j["workload"]["host_offload"], false);
        assert_eq!(j["workload"]["context_tokens"], 1024);
        assert_eq!(j["workload"]["steps"], 4);
        assert_eq!(j["target"]["config"]["host_offload"], false);
        assert_eq!(j["target"]["config"]["weight_tier"], "int8");
        assert_eq!(j["target"]["quant"], "int8");
        assert!(j["env"]["backend"].is_string());
        assert!(j["notes"].as_str().unwrap().contains("SYNTHETIC"));

        let mut real = Fake::new();
        real.kind = ContextKind::Real;
        let art = run(&mut real, &opts()).unwrap();
        assert_eq!(art.to_json()["workload"]["context"], "real");
        assert!(art.to_json()["notes"].is_null(), "a real context needs no synthetic caveat");
    }

    #[test]
    fn the_prefill_table_reports_each_length_and_survives_as_rows() {
        let mut e = Fake::new();
        let mut o = opts();
        o.prefill = vec![2048, 512];
        let art = run(&mut e, &o).unwrap();
        let rows = art.performance["prefill"]["rows"].as_array().unwrap();
        assert_eq!(rows.iter().map(|r| r["prompt_tokens"].as_u64().unwrap()).collect::<Vec<_>>(), vec![2048, 512]);
        assert!((rows[0]["tok_per_s"].as_f64().unwrap() - 1000.0).abs() < 0.01);
        assert!(art.performance["prefill"]["error"].is_null());
        assert!(render(&art).contains("prefill (real, cold"));
    }

    #[test]
    fn a_prefill_that_does_not_fit_is_reported_and_does_not_end_the_decode_sweep() {
        let mut e = Fake::new();
        e.usable = 110; // exactly one sequence fits: weights 100 + 10
        let mut o = opts();
        o.prefill = vec![256];
        let art = run(&mut e, &o).unwrap();
        // One sequence fits, so prefill runs; a second does not.
        assert_eq!(art.performance["prefill"]["rows"].as_array().unwrap().len(), 1);
        assert_eq!(art.performance["oom_boundary"]["first_failing_batch"], 2);
    }

    #[test]
    fn a_context_beyond_the_model_or_too_short_to_time_is_refused() {
        let mut e = Fake::new();
        e.max_ctx = Some(512);
        assert!(run(&mut e, &opts()).err().unwrap().contains("maximum context"));
        let mut o = opts();
        o.prefill = vec![4096];
        let mut e = Fake::new();
        e.max_ctx = Some(2048);
        assert!(run(&mut e, &o).is_err(), "a prefill length past the model limit is refused too");
        let mut e = Fake::new();
        let mut o = opts();
        o.context = 6;
        assert!(run(&mut e, &o).is_err());
    }

    #[test]
    fn decode_positions_stay_inside_the_context() {
        struct Spy(Vec<u32>);
        impl LongContextEngine for Spy {
            fn describe(&self) -> EngineInfo {
                Fake::new().describe()
            }
            fn max_context(&self) -> Option<u32> {
                None
            }
            fn plan(&mut self, _: u32, _: u32) -> Result<Fit, String> {
                Ok(Fit { fits: true, needed_bytes: 1, usable_bytes: 2 })
            }
            fn load(&mut self, _: u32, _: u32) -> Result<(), String> {
                Ok(())
            }
            fn prefill(&mut self, _: u32) -> Result<f64, String> {
                Ok(1.0)
            }
            fn decode_step(&mut self, p: &[u32]) -> Result<f64, String> {
                self.0.extend_from_slice(p);
                Ok(1.0)
            }
            fn unload(&mut self) {}
        }
        let mut spy = Spy(vec![]);
        let mut o = opts();
        o.ladder = vec![1];
        run(&mut spy, &o).unwrap();
        assert_eq!(spy.0.len(), 6, "2 warm-up + 4 timed");
        assert!(spy.0.iter().all(|&p| p < 1024), "positions {:?}", spy.0);
        assert_eq!(*spy.0.last().unwrap(), 1023, "the last step decodes at the final context position");
    }

    #[test]
    fn the_table_renders_boundary_and_peak() {
        let mut e = Fake::new();
        let t = render(&run(&mut e, &opts()).unwrap());
        assert!(t.contains("maximum sustained throughput"), "{t}");
        assert!(t.contains("out-of-memory boundary: batch 8"), "{t}");
        assert!(t.contains("largest batch the planner admits: 7"), "{t}");
    }
}
