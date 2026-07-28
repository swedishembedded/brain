// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Concrete [`PerfTarget`] adapters.
//!
//! * [`CapabilityTarget`] — wraps any `capability::Provider`. The strategic
//!   path: a model that implements the seam becomes benchmarkable with **zero**
//!   new benchmark code, because `Action::run` already takes a `Progress`
//!   callback that gives the harness its artifact timeline.
//! * [`PagedLlmTarget`] — wraps `qwen::serve::{Engine, Scheduler}` directly. The
//!   paged continuous-batching engine is the thing most worth measuring today
//!   and does not yet sit behind a `Provider`.
//!
//! As `qwen`, `yolo`, `depth` and `tts` adopt `capability::Provider`, they
//! become measurable through the first adapter and need nothing here.

use std::sync::Arc;
use std::time::Instant;

use capability::{Invocation, Progress, Provider};

use crate::target::{Emission, EmissionKind, PerfRequest, PerfTarget, ReqId, TargetInfo};

// ===================== capability seam =====================

/// Drives any `capability::Provider` action. Each `step` runs one queued
/// invocation to completion, timestamping every `Progress` callback as an
/// artifact — so streaming models yield a real TTFA/IAL timeline and one-shot
/// models collapse to a single artifact with no special-casing.
pub struct CapabilityTarget {
    provider: Arc<dyn Provider>,
    model: String,
    action: String,
    artifact_unit: String,
    /// Builds the invocation for a request (so the caller owns payload shape).
    build: Box<dyn Fn(&PerfRequest) -> Invocation>,
    queue: std::collections::VecDeque<(ReqId, PerfRequest)>,
    next: ReqId,
}

impl CapabilityTarget {
    pub fn new(
        provider: Arc<dyn Provider>,
        action: &str,
        artifact_unit: &str,
        build: Box<dyn Fn(&PerfRequest) -> Invocation>,
    ) -> CapabilityTarget {
        let model = provider.manifest().model.clone();
        CapabilityTarget {
            provider,
            model,
            action: action.to_string(),
            artifact_unit: artifact_unit.to_string(),
            build,
            queue: Default::default(),
            next: 0,
        }
    }
}

impl PerfTarget for CapabilityTarget {
    fn describe(&self) -> TargetInfo {
        TargetInfo::new(&self.model, &self.artifact_unit).with("action", self.action.clone().into())
    }

    fn submit(&mut self, req: PerfRequest) -> ReqId {
        let id = self.next;
        self.next += 1;
        self.queue.push_back((id, req));
        id
    }

    fn step(&mut self, out: &mut Vec<Emission>) -> bool {
        let Some((id, req)) = self.queue.pop_front() else { return false };
        let Some(action) = self.provider.action(&self.action) else {
            out.push(Emission { id, at: Instant::now(), kind: EmissionKind::Failed });
            return self.busy();
        };
        let inv = (self.build)(&req);
        out.push(Emission { id, at: Instant::now(), kind: EmissionKind::Admitted });

        let mut emitted = 0usize;
        let mut progress_times: Vec<Instant> = Vec::new();
        let mut on_progress = |_p: Progress| {
            progress_times.push(Instant::now());
        };
        let res = action.run(&inv, &mut on_progress);
        for at in progress_times {
            out.push(Emission { id, at, kind: EmissionKind::Artifact });
            emitted += 1;
        }
        let now = Instant::now();
        match res {
            Ok(_) => {
                // A one-shot action reports no progress; its single result IS the
                // artifact, so record one rather than reporting zero output.
                if emitted == 0 {
                    out.push(Emission { id, at: now, kind: EmissionKind::Artifact });
                }
                out.push(Emission { id, at: now, kind: EmissionKind::Done });
            }
            Err(_) => out.push(Emission { id, at: now, kind: EmissionKind::Failed }),
        }
        self.busy()
    }

    fn busy(&self) -> bool {
        !self.queue.is_empty()
    }
}

// ===================== paged LLM engine =====================

/// Drives `qwen::serve::Scheduler`. One [`PerfTarget::step`] is one scheduler
/// iteration — admit what fits, one batched decode over the running set, reap
/// completions — which is exactly the granularity TTFA and IAL are defined at.
pub struct PagedLlmTarget {
    sched: qwen::serve::Scheduler,
    info: TargetInfo,
    eos: Option<u32>,
    /// Deterministic synthetic prompt vocabulary bound (avoids tokenizer I/O in
    /// the measurement path; the engine cost is the same for any token id).
    vocab: u32,
    submitted: usize,
}

impl PagedLlmTarget {
    pub fn new(sched: qwen::serve::Scheduler, info: TargetInfo, eos: Option<u32>, vocab: u32) -> PagedLlmTarget {
        PagedLlmTarget { sched, info, eos, vocab, submitted: 0 }
    }

    /// Synthetic prompt of exactly `n` tokens, deterministic in `seed`. Content
    /// is irrelevant to cost; only length and determinism matter.
    fn prompt(&self, n: usize, seed: u64) -> Vec<u32> {
        let mut rng = data::rng::Rng::new(seed);
        (0..n).map(|_| (rng.next_u64() % self.vocab.max(1) as u64) as u32).collect()
    }
}

impl PerfTarget for PagedLlmTarget {
    fn describe(&self) -> TargetInfo {
        self.info.clone()
    }

    fn submit(&mut self, req: PerfRequest) -> ReqId {
        let prompt = self.prompt(req.input_artifacts, req.seed);
        self.submitted += 1;
        // `ignore_stop` is expressed by passing no EOS: a synthetic run must
        // produce the full requested length, or the workload silently shortens
        // and the reported rate is inflated.
        self.sched.submit(qwen::serve::Request { prompt, max_new: req.output_artifacts, eos: self.eos })
    }

    fn step(&mut self, out: &mut Vec<Emission>) -> bool {
        let report = self.sched.step_report();
        let now = Instant::now();
        for id in report.admitted {
            out.push(Emission { id, at: now, kind: EmissionKind::Admitted });
        }
        for (id, n) in report.produced {
            for _ in 0..n {
                out.push(Emission { id, at: now, kind: EmissionKind::Artifact });
            }
        }
        for id in report.finished {
            out.push(Emission { id, at: now, kind: EmissionKind::Done });
        }
        self.busy()
    }

    fn busy(&self) -> bool {
        self.sched.pending()
    }

    fn counters(&self) -> Vec<(String, serde_json::Value)> {
        vec![("kv_free_blocks".into(), serde_json::json!(self.sched.free_blocks()))]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Action, ActionResult, ActionSpec, Manifest, Outcome};

    /// A provider whose action streams `steps` progress callbacks.
    struct Streamer {
        steps: u32,
    }
    struct StreamAction {
        steps: u32,
    }
    impl Action for StreamAction {
        fn spec(&self) -> ActionSpec {
            ActionSpec::new("gen", "streams N artifacts")
        }
        fn run(&self, _inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
            for i in 0..self.steps {
                progress(Progress { step: i, total: self.steps, message: String::new() });
            }
            Ok(Outcome::new())
        }
    }
    impl Provider for Streamer {
        fn manifest(&self) -> Manifest {
            Manifest::new("streamer", "test", vec![ActionSpec::new("gen", "streams N artifacts").streaming()])
        }
        fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
            (name == "gen").then(|| Arc::new(StreamAction { steps: self.steps }) as Arc<dyn Action>)
        }
    }

    /// A provider whose action returns a single result with no progress.
    struct OneShot;
    struct OneShotAction;
    impl Action for OneShotAction {
        fn spec(&self) -> ActionSpec {
            ActionSpec::new("detect", "one-shot")
        }
        fn run(&self, _inv: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            Ok(Outcome::new())
        }
    }
    impl Provider for OneShot {
        fn manifest(&self) -> Manifest {
            Manifest::new("oneshot", "test", vec![ActionSpec::new("detect", "one-shot")])
        }
        fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
            (name == "detect").then(|| Arc::new(OneShotAction) as Arc<dyn Action>)
        }
    }

    fn req(out: usize) -> PerfRequest {
        PerfRequest { input_artifacts: 8, output_artifacts: out, class: 0, seed: 1 }
    }

    #[test]
    fn streaming_provider_yields_one_artifact_per_progress() {
        let mut t = CapabilityTarget::new(
            Arc::new(Streamer { steps: 5 }),
            "gen",
            "token",
            Box::new(|_| Invocation::new()),
        );
        t.submit(req(5));
        let mut out = Vec::new();
        while t.step(&mut out) {}
        let arts = out.iter().filter(|e| e.kind == EmissionKind::Artifact).count();
        assert_eq!(arts, 5, "each Progress callback is one artifact");
        assert_eq!(out.iter().filter(|e| e.kind == EmissionKind::Done).count(), 1);
    }

    #[test]
    fn one_shot_provider_reports_one_artifact_not_zero() {
        let mut t = CapabilityTarget::new(
            Arc::new(OneShot),
            "detect",
            "frame",
            Box::new(|_| Invocation::new()),
        );
        t.submit(req(1));
        let mut out = Vec::new();
        while t.step(&mut out) {}
        let arts = out.iter().filter(|e| e.kind == EmissionKind::Artifact).count();
        assert_eq!(arts, 1, "a non-streaming action still produced one unit of output");
    }

    #[test]
    fn missing_action_fails_the_request_rather_than_hanging() {
        let mut t = CapabilityTarget::new(
            Arc::new(OneShot),
            "no-such-action",
            "frame",
            Box::new(|_| Invocation::new()),
        );
        t.submit(req(1));
        let mut out = Vec::new();
        while t.step(&mut out) {}
        assert_eq!(out.iter().filter(|e| e.kind == EmissionKind::Failed).count(), 1);
    }

    #[test]
    fn capability_target_reports_the_model_and_unit() {
        let t = CapabilityTarget::new(
            Arc::new(Streamer { steps: 1 }),
            "gen",
            "audio_chunk",
            Box::new(|_| Invocation::new()),
        );
        let d = t.describe();
        assert_eq!(d.model, "streamer");
        assert_eq!(d.artifact_unit, "audio_chunk");
    }
}
