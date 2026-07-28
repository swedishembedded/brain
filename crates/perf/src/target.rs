// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The seam every benchmark drives: a [`PerfTarget`] accepts requests and
//! reports **emissions** — timestamped events on the path from submit to done.
//!
//! This is what makes the suite model-agnostic rather than LLM-shaped. brain
//! serves detection, depth, TTS, image generation, forecasting, 3D and world
//! models from one engine, so the harness cannot be written in terms of
//! "tokens". It is written in terms of *artifacts arriving over time*:
//!
//! ```text
//! t_submit ──► t_admit ──► t_first ──► t_artifact[1..n] ──► t_done
//! ```
//!
//! For a decoder LM that timeline yields TTFT / ITL / TPOT / E2E. For a one-shot
//! model (`detect`, `depth`, a single forecast) `n == 1` and time-to-first
//! collapses into end-to-end — no special-casing anywhere in the harness.
//!
//! `capability::Action::run` already takes a `progress: &mut dyn FnMut(Progress)`
//! callback, so [`CapabilityTarget`] turns **any** model implementing
//! `capability::Provider` into a benchmarkable target with no new benchmark code.

use std::time::Instant;

/// Identifies one in-flight request.
pub type ReqId = u64;

/// What a target produced, and when.
#[derive(Clone, Debug)]
pub struct Emission {
    pub id: ReqId,
    pub at: Instant,
    pub kind: EmissionKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmissionKind {
    /// The scheduler admitted the request (queue time ends here).
    Admitted,
    /// One output artifact: a token, an audio chunk, a frame, a denoise step.
    Artifact,
    /// The request completed successfully.
    Done,
    /// The request failed; it counts as an error, never as goodput.
    Failed,
    /// The engine's admission policy refused the request. Distinct from
    /// `Failed`: refusing provably-late work is the *desired* behaviour under
    /// overload, and conflating the two would penalise exactly the policy the
    /// overload scenario exists to reward.
    Rejected,
}

/// One unit of work to submit.
#[derive(Clone, Debug)]
pub struct PerfRequest {
    /// Number of input artifacts (prompt tokens, input frames, …). Recorded for
    /// input-rate accounting; the target decides how to realise it.
    pub input_artifacts: usize,
    /// Number of output artifacts to produce.
    pub output_artifacts: usize,
    /// Which traffic class this belongs to (`mixed`); `0` for single-class runs.
    pub class: usize,
    /// Deterministic per-request seed, so payloads reproduce exactly.
    pub seed: u64,
}

/// What a target is, for the result fingerprint.
#[derive(Clone, Debug)]
pub struct TargetInfo {
    pub model: String,
    /// The name of one output artifact — `token`, `frame`, `audio_chunk`,
    /// `denoise_step`. **Results with different units are never ranked against
    /// each other**; the report refuses to.
    pub artifact_unit: String,
    pub params: Option<u64>,
    pub quant: Option<String>,
    pub weights_sha256: Option<String>,
    /// Free-form target configuration recorded verbatim in the artifact
    /// (block size, KV dtype, batch caps, …).
    pub config: Vec<(String, serde_json::Value)>,
}

impl TargetInfo {
    pub fn new(model: &str, artifact_unit: &str) -> TargetInfo {
        TargetInfo {
            model: model.to_string(),
            artifact_unit: artifact_unit.to_string(),
            params: None,
            quant: None,
            weights_sha256: None,
            config: Vec::new(),
        }
    }
    pub fn with(mut self, k: &str, v: serde_json::Value) -> TargetInfo {
        self.config.push((k.to_string(), v));
        self
    }
    pub fn to_json(&self) -> serde_json::Value {
        let cfg: serde_json::Map<String, serde_json::Value> = self.config.iter().cloned().collect();
        serde_json::json!({
            "model": self.model,
            "artifact_unit": self.artifact_unit,
            "params": self.params.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
            "quant": self.quant.clone().map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
            "weights_sha256": self.weights_sha256.clone().map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
            "config": serde_json::Value::Object(cfg),
        })
    }
}

/// A benchmarkable engine.
///
/// The driver calls [`submit`](PerfTarget::submit) as its arrival process
/// dictates and [`step`](PerfTarget::step) to let the engine make progress,
/// collecting whatever emissions came out. Targets are **not** required to be
/// async or threaded — `step` doing one scheduler iteration is the normal case,
/// which keeps the measurement free of runtime scheduling noise.
pub trait PerfTarget {
    fn describe(&self) -> TargetInfo;

    /// Enqueue a request. Returns its id.
    fn submit(&mut self, req: PerfRequest) -> ReqId;

    /// Advance the engine by one unit of work, appending anything produced to
    /// `out`. Returns `false` when there is nothing left in flight.
    fn step(&mut self, out: &mut Vec<Emission>) -> bool;

    /// True while any request is queued or running.
    fn busy(&self) -> bool;

    /// Optional engine-internal counters merged into the result's `memory`
    /// block (KV hit rate, free blocks, evictions). Empty by default.
    fn counters(&self) -> Vec<(String, serde_json::Value)> {
        Vec::new()
    }

    /// Reset between repetitions of a best-of-N measurement. Targets that hold a
    /// cache decide here whether a repeat is a cold or a warm run.
    fn reset(&mut self, _warm: bool) {}

    /// Install a named admission policy on the underlying engine:
    /// `"unbounded"`, `"depth:<N>"`, or `"deadline:<ms>"`. Returns `false`
    /// when this target has no admission seam — the scenario then reports
    /// that nothing could ever be rejected, rather than pretending a policy
    /// was in force.
    fn set_admission(&mut self, _policy: &str) -> bool {
        false
    }

    /// Self-verify: produce a correctness verdict for THIS configuration, or
    /// `None` when the target has no way to check itself. For a decoder the
    /// meaningful check is batched-vs-sequential greedy equality through the
    /// same engine — the exact numeric paths a batching optimisation can break.
    /// Called after measurement; a failing verdict marks the artifact
    /// `valid: false` and excludes it from comparison.
    fn fidelity(&mut self) -> Option<crate::fidelity::Fidelity> {
        None
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    /// A deterministic fake target: each request takes `prefill_steps` to admit,
    /// then emits one artifact per step. Used to test the driver and metrics
    /// without a model or a device.
    pub struct FakeTarget {
        queue: std::collections::VecDeque<(ReqId, PerfRequest)>,
        running: Vec<(ReqId, usize, usize)>, // id, produced, wanted
        next: ReqId,
        pub max_concurrent: usize,
        pub prefill_steps: usize,
        pending_prefill: Vec<(ReqId, usize, PerfRequest)>,
    }

    impl FakeTarget {
        pub fn new(max_concurrent: usize, prefill_steps: usize) -> FakeTarget {
            FakeTarget {
                queue: Default::default(),
                running: Vec::new(),
                next: 0,
                max_concurrent,
                prefill_steps,
                pending_prefill: Vec::new(),
            }
        }
    }

    impl PerfTarget for FakeTarget {
        fn describe(&self) -> TargetInfo {
            TargetInfo::new("fake", "token")
        }
        fn submit(&mut self, req: PerfRequest) -> ReqId {
            let id = self.next;
            self.next += 1;
            self.queue.push_back((id, req));
            id
        }
        fn step(&mut self, out: &mut Vec<Emission>) -> bool {
            // Admit into prefill.
            while self.running.len() + self.pending_prefill.len() < self.max_concurrent {
                match self.queue.pop_front() {
                    Some((id, req)) => self.pending_prefill.push((id, self.prefill_steps, req)),
                    None => break,
                }
            }
            // Advance prefill; on completion admit + emit the first artifact.
            let mut still = Vec::new();
            for (id, left, req) in std::mem::take(&mut self.pending_prefill) {
                if left <= 1 {
                    out.push(Emission { id, at: Instant::now(), kind: EmissionKind::Admitted });
                    out.push(Emission { id, at: Instant::now(), kind: EmissionKind::Artifact });
                    if req.output_artifacts <= 1 {
                        out.push(Emission { id, at: Instant::now(), kind: EmissionKind::Done });
                    } else {
                        self.running.push((id, 1, req.output_artifacts));
                    }
                } else {
                    still.push((id, left - 1, req));
                }
            }
            self.pending_prefill = still;
            // One decode step for every running request.
            let mut i = 0;
            while i < self.running.len() {
                let (id, ref mut produced, wanted) = self.running[i];
                *produced += 1;
                let done = *produced >= wanted;
                out.push(Emission { id, at: Instant::now(), kind: EmissionKind::Artifact });
                if done {
                    out.push(Emission { id, at: Instant::now(), kind: EmissionKind::Done });
                    self.running.remove(i);
                } else {
                    i += 1;
                }
            }
            self.busy()
        }
        fn busy(&self) -> bool {
            !self.queue.is_empty() || !self.running.is_empty() || !self.pending_prefill.is_empty()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::FakeTarget;
    use super::*;

    #[test]
    fn one_shot_request_collapses_to_a_single_artifact() {
        let mut t = FakeTarget::new(4, 1);
        t.submit(PerfRequest { input_artifacts: 8, output_artifacts: 1, class: 0, seed: 0 });
        let mut out = Vec::new();
        while t.step(&mut out) {}
        let arts = out.iter().filter(|e| e.kind == EmissionKind::Artifact).count();
        let done = out.iter().filter(|e| e.kind == EmissionKind::Done).count();
        assert_eq!(arts, 1, "a one-shot model emits exactly one artifact");
        assert_eq!(done, 1);
    }

    #[test]
    fn artifact_count_matches_the_request() {
        let mut t = FakeTarget::new(2, 2);
        t.submit(PerfRequest { input_artifacts: 4, output_artifacts: 5, class: 0, seed: 0 });
        t.submit(PerfRequest { input_artifacts: 4, output_artifacts: 3, class: 0, seed: 1 });
        let mut out = Vec::new();
        while t.step(&mut out) {}
        for (id, want) in [(0u64, 5usize), (1, 3)] {
            let n = out.iter().filter(|e| e.id == id && e.kind == EmissionKind::Artifact).count();
            assert_eq!(n, want, "request {id} must emit exactly {want} artifacts");
        }
    }

    #[test]
    fn concurrency_cap_is_respected() {
        let mut t = FakeTarget::new(2, 1);
        for i in 0..6 {
            t.submit(PerfRequest { input_artifacts: 4, output_artifacts: 4, class: 0, seed: i });
        }
        let mut out = Vec::new();
        t.step(&mut out);
        let admitted = out.iter().filter(|e| e.kind == EmissionKind::Admitted).count();
        assert!(admitted <= 2, "never admit more than max_concurrent, got {admitted}");
    }

    #[test]
    fn units_are_recorded_for_cross_model_safety() {
        let t = FakeTarget::new(1, 1);
        assert_eq!(t.describe().artifact_unit, "token");
    }
}
