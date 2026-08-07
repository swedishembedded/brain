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
//! * [`ExecutorTarget`] — wraps a [`residency::Executor`] holding a real
//!   `ResidentModel`: the scheduler + budgets + per-device lanes production
//!   serving (D-Bus) uses, so concurrency numbers reflect real batching/placement.
//! * [`HttpTarget`] — wraps the REAL served path: `apiserve::router()`, called
//!   in-process via `tower::Service::oneshot`. The only adapter here that
//!   exercises JSON parsing, the edge concurrency limiter, and the admission
//!   race in `apiserve::bridge` — the layer `.todo/serving-performance-audit.md`'s
//!   600s regression actually lived in, and that `ExecutorTarget`/`PagedLlmTarget`
//!   skip by construction (they drive the executor/engine directly).
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

/// [`PagedLlmTarget::fidelity`]'s fixed correctness-probe prompt length. Any
/// caller sizing this target's KV pool (`crates/cli/src/perf_cli.rs::pool_for`)
/// must reserve at least this much per-sequence capacity regardless of the
/// measured workload's own (possibly much smaller, e.g. under `--smoke`)
/// `--input`, or the probe's own requests get rejected at admission --
/// `compared == 0` positions, which reads as a confusing gate failure rather
/// than the capacity shortfall it actually is.
pub const FIDELITY_PROMPT_TOKENS: u32 = 24;
/// [`PagedLlmTarget::fidelity`]'s fixed correctness-probe generation length --
/// same reasoning as [`FIDELITY_PROMPT_TOKENS`], for `--output`.
pub const FIDELITY_MAX_NEW: u32 = 12;

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
        for (id, _reason) in report.rejected {
            out.push(Emission { id, at: now, kind: EmissionKind::Rejected });
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
        let (hit, looked, cached) = self.sched.prefix_stats();
        // Hit rate only once something was looked up — never a fabricated 0.
        let rate = (looked > 0).then(|| hit as f64 / looked as f64);
        vec![
            ("kv_free_blocks".into(), serde_json::json!(self.sched.free_blocks())),
            (
                "kv_prefix_hit_rate".into(),
                rate.map(serde_json::Value::from).unwrap_or(serde_json::Value::Null),
            ),
            ("kv_prefix_cached_blocks".into(), serde_json::json!(cached)),
            // Device-op accounting (K): submits/dispatches/readbacks since the
            // engine was built. Null where the backend does not count.
            (
                "device_ops".into(),
                self.sched
                    .device_stats()
                    .map(|d| {
                        serde_json::json!({
                            "submits": d.submits,
                            "dispatches": d.dispatches,
                            "readbacks": d.readbacks,
                        })
                    })
                    .unwrap_or(serde_json::Value::Null),
            ),
            // The int8-KV memory claim, measured rather than asserted: an
            // analytic byte count derived from the SAME arithmetic the engine
            // allocated from (qwen::serve::kv_pool_bytes), not a device
            // counter -- see that function's doc comment for why. `kv_dtype`
            // itself is already reported in the artifact's `target.config`
            // (`build_qwen_synth`/`build_qwen`'s `.with("kv_dtype", ...)`).
            ("kv_pool_bytes".into(), serde_json::json!(self.sched.kv_pool_bytes())),
            ("kv_theoretical_artifacts".into(), serde_json::json!(self.sched.kv_pool_capacity_tokens())),
            // A point sample, not a tracked peak (no backend exposes device
            // memory here -- see `kv_pool_bytes` for the honest device-side
            // number). On a box with no discrete GPU, host RSS genuinely IS
            // device memory, so this is the field to fill rather than
            // fabricate `peak_device_mb`.
            (
                "peak_host_mb".into(),
                crate::scenarios::soak::host_mem_mb().map(|v| serde_json::json!(v)).unwrap_or(serde_json::Value::Null),
            ),
        ]
    }

    /// Map the harness policy names onto the engine's admission seam.
    fn set_admission(&mut self, policy: &str) -> bool {
        use qwen::serve::{DeadlineAware, MaxQueueDepth, UnboundedQueue};
        match policy.split_once(':') {
            None if policy == "unbounded" => {
                self.sched.set_admission(Box::new(UnboundedQueue));
                true
            }
            Some(("depth", n)) => match n.parse::<usize>() {
                Ok(n) if n > 0 => {
                    self.sched.set_admission(Box::new(MaxQueueDepth(n)));
                    true
                }
                _ => false,
            },
            Some(("deadline", ms)) => match ms.parse::<f64>() {
                Ok(ms) if ms > 0.0 => {
                    self.sched.set_admission(Box::new(DeadlineAware { deadline_ms: ms }));
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    /// Batched-vs-sequential greedy equality through the SAME engine instance —
    /// the numeric paths a batching/kernel optimisation can break. Greedy is
    /// deterministic, so the gate demands exactness.
    fn fidelity(&mut self) -> Option<crate::fidelity::Fidelity> {
        let prompts: Vec<Vec<u32>> =
            (0..3u64).map(|i| self.prompt(FIDELITY_PROMPT_TOKENS as usize, 0x5EED ^ i)).collect();
        let max_new = FIDELITY_MAX_NEW as usize;

        // Sequential reference: one request at a time, drained to completion.
        let mut seq_out = Vec::new();
        for p in &prompts {
            let id = self.sched.submit(qwen::serve::Request {
                prompt: p.clone(),
                max_new,
                eos: None,
            });
            let done = self.sched.run();
            seq_out.push(done.get(&id).cloned().unwrap_or_default());
        }

        // Batched candidate: all three in flight together.
        let ids: Vec<u64> = prompts
            .iter()
            .map(|p| {
                self.sched.submit(qwen::serve::Request { prompt: p.clone(), max_new, eos: None })
            })
            .collect();
        let done = self.sched.run();
        let bat_out: Vec<Vec<u32>> =
            ids.iter().map(|id| done.get(id).cloned().unwrap_or_default()).collect();

        Some(crate::fidelity::Fidelity::greedy(
            "sequential-greedy-same-engine",
            &bat_out,
            &seq_out,
            crate::fidelity::EXACT,
        ))
    }
}

// ===================== residency executor seam =====================

/// Drives a [`residency::Executor`] — the scheduler + budgets + per-device
/// lanes that production serving (D-Bus) uses — so concurrency numbers reflect
/// brain's real batching/placement, not a synchronous provider mutex.
///
/// `submit` hands the job to the executor immediately (arrivals are genuinely async);
/// emissions come back on a channel from the job callbacks: the first
/// `Progress` marks `Admitted` (the lane started the group this job is in),
/// the reply marks `Artifact` + `Done` (one-shot models) or `Failed`.
/// `step` drains the channel; the executor's own threads make the progress.
///
/// For a **streaming** resident action ([`ExecutorTarget::new_streaming`]) the
/// in-flight `Progress` callbacks themselves are the artifact timeline — the
/// same contract [`CapabilityTarget`] gives a `Provider` — filtered by a
/// predicate so bookkeeping callbacks ("encoding", "decoding") never count as
/// output units.
pub struct ExecutorTarget {
    exec: residency::Executor,
    info: TargetInfo,
    model: String,
    action: String,
    build: Box<dyn Fn(&PerfRequest) -> Invocation>,
    /// `Some` = streaming: a `Progress` this predicate accepts is timestamped
    /// as one `Artifact`. `None` = one-shot: the reply is the single artifact.
    is_artifact: Option<Arc<dyn Fn(&Progress) -> bool + Send + Sync>>,
    rx: std::sync::mpsc::Receiver<Emission>,
    tx: std::sync::mpsc::Sender<Emission>,
    inflight: std::collections::HashSet<ReqId>,
    next: ReqId,
}

impl ExecutorTarget {
    pub fn new(
        exec: residency::Executor,
        model: &str,
        action: &str,
        artifact_unit: &str,
        info_extra: Vec<(String, serde_json::Value)>,
        build: Box<dyn Fn(&PerfRequest) -> Invocation>,
    ) -> ExecutorTarget {
        let mut info = TargetInfo::new(model, artifact_unit);
        info.config = info_extra;
        ExecutorTarget::build(exec, model, action, info, build, None)
    }

    /// Streaming variant: every `Progress` accepted by `is_artifact` becomes an
    /// `Artifact` emission at its callback time, so a multi-step resident model
    /// (diffusion denoise steps, TTS chunks) yields a real TTFA/IAL timeline
    /// through the real scheduler. `job_model` routes to the resident's
    /// manifest id; `info.model` may name the variant more precisely. An action
    /// that streams no accepted progress still records its reply as one
    /// artifact (one-shot collapse, as [`CapabilityTarget`] does).
    pub fn new_streaming(
        exec: residency::Executor,
        job_model: &str,
        action: &str,
        info: TargetInfo,
        build: Box<dyn Fn(&PerfRequest) -> Invocation>,
        is_artifact: Arc<dyn Fn(&Progress) -> bool + Send + Sync>,
    ) -> ExecutorTarget {
        ExecutorTarget::build(exec, job_model, action, info, build, Some(is_artifact))
    }

    fn build(
        exec: residency::Executor,
        job_model: &str,
        action: &str,
        info: TargetInfo,
        build: Box<dyn Fn(&PerfRequest) -> Invocation>,
        is_artifact: Option<Arc<dyn Fn(&Progress) -> bool + Send + Sync>>,
    ) -> ExecutorTarget {
        let (tx, rx) = std::sync::mpsc::channel();
        ExecutorTarget {
            exec,
            info,
            model: job_model.to_string(),
            action: action.to_string(),
            build,
            is_artifact,
            rx,
            tx,
            inflight: std::collections::HashSet::new(),
            next: 1,
        }
    }
}

impl PerfTarget for ExecutorTarget {
    fn describe(&self) -> TargetInfo {
        self.info.clone()
    }

    fn submit(&mut self, req: PerfRequest) -> ReqId {
        let id = self.next;
        self.next += 1;
        self.inflight.insert(id);
        let inv = (self.build)(&req);
        let tx_p = self.tx.clone();
        let tx_r = self.tx.clone();
        let mut admitted = false;
        let is_artifact = self.is_artifact.clone();
        // Shared between the two callbacks: how many artifacts streamed, so the
        // reply knows whether it must stand in as the single artifact.
        let streamed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let streamed_r = streamed.clone();
        self.exec.submit(
            residency::executor::Job::new(self.model.clone(), self.action.clone(), inv)
                .on_progress(move |p| {
                    if !admitted {
                        admitted = true;
                        let _ = tx_p.send(Emission { id, at: Instant::now(), kind: EmissionKind::Admitted });
                    }
                    if let Some(accept) = &is_artifact {
                        if accept(&p) {
                            streamed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let _ = tx_p.send(Emission { id, at: Instant::now(), kind: EmissionKind::Artifact });
                        }
                    }
                })
                .reply(move |r| {
                    let at = Instant::now();
                    match r {
                        Ok(_) => {
                            if streamed_r.load(std::sync::atomic::Ordering::Relaxed) == 0 {
                                let _ = tx_r.send(Emission { id, at, kind: EmissionKind::Artifact });
                            }
                            let _ = tx_r.send(Emission { id, at, kind: EmissionKind::Done });
                        }
                        Err(_) => {
                            let _ = tx_r.send(Emission { id, at, kind: EmissionKind::Failed });
                        }
                    }
                }),
        );
        id
    }

    fn step(&mut self, out: &mut Vec<Emission>) -> bool {
        // The executor's threads do the work; drain what they emitted. Block
        // briefly when something is in flight so the driver doesn't spin.
        if self.inflight.is_empty() {
            while let Ok(e) = self.rx.try_recv() {
                out.push(e);
            }
            return false;
        }
        if let Ok(e) = self.rx.recv_timeout(std::time::Duration::from_millis(2)) {
            if matches!(e.kind, EmissionKind::Done | EmissionKind::Failed) {
                self.inflight.remove(&e.id);
            }
            out.push(e);
        }
        while let Ok(e) = self.rx.try_recv() {
            if matches!(e.kind, EmissionKind::Done | EmissionKind::Failed) {
                self.inflight.remove(&e.id);
            }
            out.push(e);
        }
        !self.inflight.is_empty()
    }

    fn busy(&self) -> bool {
        !self.inflight.is_empty()
    }

    /// The scheduler's own counters — **the proof that batching happened**.
    /// `sched_max_batch > 1` means concurrent same-key jobs really were handed
    /// to one `Instance::run_batch` call; `sched_jobs / sched_batches` is the
    /// mean group size the model actually saw. Without these, a throughput
    /// number cannot distinguish "the model batched" from "the scheduler ran
    /// them one at a time and the GPU happened to be idle".
    fn counters(&self) -> Vec<(String, serde_json::Value)> {
        let s = self.exec.stats();
        vec![
            ("sched_batches".into(), s.batches.into()),
            ("sched_jobs".into(), s.jobs.into()),
            ("sched_max_batch".into(), s.max_batch.into()),
            ("sched_mean_batch".into(), serde_json::json!(if s.batches > 0 { s.jobs as f64 / s.batches as f64 } else { 0.0 })),
            ("sched_queue_peak".into(), s.queue_peak.into()),
            ("sched_max_parallel".into(), s.max_parallel.into()),
            ("sched_builds".into(), s.builds.into()),
            ("sched_evictions".into(), s.evictions.into()),
        ]
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
                progress(Progress::step(i, self.steps, ""));
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

    /// A resident model whose action streams flux2-shaped progress: one
    /// "encoding" callback, N "denoising" steps, one "decoding" callback.
    struct Diffuser;
    struct DiffuserInst;
    impl residency::ResidentModel for Diffuser {
        fn manifest(&self) -> Manifest {
            Manifest::new("diffuser", "test", vec![ActionSpec::new("gen", "denoise").streaming()])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> residency::InstanceKey {
            residency::InstanceKey::new("diffuser", "default")
        }
        fn estimate(&self, _k: &residency::InstanceKey) -> residency::MemCost {
            residency::MemCost::new(0, 1 << 20)
        }
        fn activate(
            &self,
            _k: &residency::InstanceKey,
            _d: residency::Device,
        ) -> Result<Box<dyn residency::Instance>, String> {
            Ok(Box::new(DiffuserInst))
        }
    }
    impl residency::Instance for DiffuserInst {
        fn run(&mut self, _a: &str, _i: &Invocation, p: &mut dyn FnMut(Progress)) -> ActionResult {
            p(Progress::step(0, 5, "encoding prompt"));
            for i in 0..3u32 {
                p(Progress::step(i + 1, 5, "denoising"));
            }
            p(Progress::step(5, 5, "decoding"));
            Ok(Outcome::new())
        }
    }

    /// The streaming executor path: only the accepted `Progress` callbacks are
    /// artifacts — bookkeeping callbacks (encode/decode) never inflate the
    /// output count, and the reply does not double-count a streamed run.
    #[test]
    fn streaming_executor_target_counts_only_accepted_progress_as_artifacts() {
        let mut budgets = residency::budget::Budgets::new();
        budgets.set(residency::Device::Cpu, 1 << 30, 0);
        let exec = residency::Executor::start(
            vec![Arc::new(Diffuser)],
            budgets,
            residency::Policy::default(),
        );
        let mut t = ExecutorTarget::new_streaming(
            exec,
            "diffuser",
            "gen",
            TargetInfo::new("diffuser", "denoise_step"),
            Box::new(|_| Invocation::new()),
            Arc::new(|p: &Progress| p.message == "denoising"),
        );
        t.submit(req(3));
        let mut out = Vec::new();
        while t.step(&mut out) {}
        t.step(&mut out); // drain anything raced past the last busy poll
        let arts = out.iter().filter(|e| e.kind == EmissionKind::Artifact).count();
        assert_eq!(arts, 3, "exactly the denoise-step callbacks are artifacts");
        assert_eq!(out.iter().filter(|e| e.kind == EmissionKind::Admitted).count(), 1);
        assert_eq!(out.iter().filter(|e| e.kind == EmissionKind::Done).count(), 1);
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

// ===================== HTTP surface seam =====================

/// Drives the REAL served path: `apiserve::router()`, called in-process via
/// `tower::Service::oneshot` (no socket) — so the numbers are exactly what a
/// real client sees: JSON parsing, the edge concurrency limiter, the
/// admission race in `apiserve::bridge`, chat-template rendering,
/// tokenization, and generation, all the way down to `residency::Executor`
/// and the resident model. This is the target the serving-performance audit's
/// 600s regression (`.todo/serving-performance-audit.md`) would have shown up
/// in, and the synthetic in-process targets above (measuring the executor or
/// engine directly) do not — they skip exactly the layer the bug lived in.
///
/// Requests are OpenAI-dialect **streaming** chat completions (`stream:
/// true`), so the artifact timeline comes from real SSE `delta.content`
/// chunks as they arrive over the wire, timestamped as each one is read off
/// the response body — not a post-hoc replay after the whole body is
/// buffered. A background multi-thread Tokio runtime drives the actual router
/// calls; `step` only drains what they produced, mirroring
/// [`ExecutorTarget`]'s sync/async seam (the executor's own threads —
/// here, the runtime's worker threads — do the real work; the driver loop
/// stays synchronous and free of runtime-scheduling noise).
///
/// `describe`/`fidelity`/`set_admission`/device-op counters are NOT yet wired
/// through this target: today's `QwenInstance` (`crates/cli/src/
/// resident_llm.rs`) is built on `Qwen::from_reader_decode`, which has no
/// `device_stats()`/admission-policy seam to read. Once the LLM residents are
/// rewired onto `qwen::serve::Engine` (this plan's W5), those numbers become
/// reachable here with no change to this struct's shape — reporting a
/// fabricated number now would violate the "never a fabricated zero" rule
/// (`docs/performance/benchmarking.md` §1), so they are simply absent.
pub struct HttpTarget {
    rt: tokio::runtime::Runtime,
    router: axum::Router,
    model: String,
    key: String,
    info: TargetInfo,
    tx: std::sync::mpsc::Sender<Emission>,
    rx: std::sync::mpsc::Receiver<Emission>,
    inflight: std::collections::HashSet<ReqId>,
    next: ReqId,
}

impl HttpTarget {
    /// `router` is `apiserve::router(state)` built for a `Provider::OpenAI`
    /// `AppState` whose key is `key`; `model` is the id the resident
    /// registered itself under (the `"model"` field every request names).
    pub fn new(router: axum::Router, model: &str, key: &str, info: TargetInfo) -> HttpTarget {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .expect("perf: build the HttpTarget background tokio runtime");
        let (tx, rx) = std::sync::mpsc::channel();
        HttpTarget { rt, router, model: model.to_string(), key: key.to_string(), info, tx, rx, inflight: Default::default(), next: 1 }
    }

    /// A deterministic synthetic system-prompt filler of approximately
    /// `n_tokens` words. Real BPE tokenization will not reproduce `n_tokens`
    /// exactly — `PerfRequest::input_artifacts`'s own contract is "the target
    /// decides how to realise it", and the driver records the REQUESTED
    /// length for input-rate accounting (`crates/perf/src/driver.rs`), not a
    /// count re-derived from the target. The realistic-agentic-prompt shape
    /// (long shared system preamble + a short user turn) is what the serving-
    /// performance audit measured against; this is the synthetic stand-in.
    fn synth_system_prompt(n_tokens: usize, seed: u64) -> String {
        let mut rng = data::rng::Rng::new(seed);
        (0..n_tokens.max(1)).map(|_| format!("tok{}", rng.next_u64() % 4096)).collect::<Vec<_>>().join(" ")
    }
}

impl PerfTarget for HttpTarget {
    fn describe(&self) -> TargetInfo {
        self.info.clone()
    }

    fn submit(&mut self, req: PerfRequest) -> ReqId {
        let id = self.next;
        self.next += 1;
        self.inflight.insert(id);

        let body = serde_json::json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": HttpTarget::synth_system_prompt(req.input_artifacts, req.seed)},
                {"role": "user", "content": "Continue the sequence."},
            ],
            "max_tokens": req.output_artifacts,
            "temperature": 0,
            "stream": true,
        });
        let http_req = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .header("authorization", format!("Bearer {}", self.key))
            .body(axum::body::Body::from(serde_json::to_vec(&body).expect("perf: serialize chat body")))
            .expect("perf: build chat request");

        let router = self.router.clone();
        let tx = self.tx.clone();
        self.rt.spawn(async move { drive_one(router, http_req, id, tx).await });
        id
    }

    fn step(&mut self, out: &mut Vec<Emission>) -> bool {
        // The runtime's worker threads do the real work; drain what they
        // emitted, blocking briefly when something is in flight so the
        // driver doesn't spin (same shape as `ExecutorTarget::step`).
        if self.inflight.is_empty() {
            while let Ok(e) = self.rx.try_recv() {
                out.push(e);
            }
            return false;
        }
        let terminal = |k: EmissionKind| matches!(k, EmissionKind::Done | EmissionKind::Failed | EmissionKind::Rejected);
        if let Ok(e) = self.rx.recv_timeout(std::time::Duration::from_millis(2)) {
            if terminal(e.kind) {
                self.inflight.remove(&e.id);
            }
            out.push(e);
        }
        while let Ok(e) = self.rx.try_recv() {
            if terminal(e.kind) {
                self.inflight.remove(&e.id);
            }
            out.push(e);
        }
        !self.inflight.is_empty()
    }

    fn busy(&self) -> bool {
        !self.inflight.is_empty()
    }
}

/// Fire one chat-completion request through the router and translate the SSE
/// response into a real-time emission timeline. Runs on the target's
/// background runtime, concurrently with every other in-flight request —
/// exactly as real concurrent clients would be served.
async fn drive_one(router: axum::Router, req: axum::http::Request<axum::body::Body>, id: ReqId, tx: std::sync::mpsc::Sender<Emission>) {
    use futures::StreamExt;
    use tower::ServiceExt;

    let resp = match router.oneshot(req).await {
        Ok(r) => r,
        Err(_) => {
            let _ = tx.send(Emission { id, at: Instant::now(), kind: EmissionKind::Failed });
            return;
        }
    };
    let status = resp.status();
    if status == axum::http::StatusCode::TOO_MANY_REQUESTS || status == axum::http::StatusCode::SERVICE_UNAVAILABLE {
        let _ = tx.send(Emission { id, at: Instant::now(), kind: EmissionKind::Rejected });
        return;
    }
    if !status.is_success() {
        let _ = tx.send(Emission { id, at: Instant::now(), kind: EmissionKind::Failed });
        return;
    }
    // By the time headers are back, `apiserve::bridge`'s admission race has
    // already resolved — it runs BEFORE the SSE body is returned (see
    // `bridge::stream_inner`) — so "response received" IS the real admit
    // instant, not an approximation of it.
    let _ = tx.send(Emission { id, at: Instant::now(), kind: EmissionKind::Admitted });

    let mut stream = resp.into_body().into_data_stream();
    let mut buf = String::new();
    let mut saw_done = false;
    while let Some(chunk) = stream.next().await {
        let bytes = match chunk {
            Ok(b) => b,
            Err(_) => {
                let _ = tx.send(Emission { id, at: Instant::now(), kind: EmissionKind::Failed });
                return;
            }
        };
        buf.push_str(&String::from_utf8_lossy(&bytes));
        // SSE events are separated by a blank line; process every COMPLETE one
        // as soon as it arrives so the timestamp reflects real wire timing,
        // not a post-hoc replay after the whole body was buffered.
        while let Some(pos) = buf.find("\n\n") {
            let frame: String = buf.drain(..pos + 2).collect();
            let at = Instant::now();
            for line in frame.lines() {
                let Some(data) = line.strip_prefix("data: ") else { continue };
                if data == "[DONE]" {
                    saw_done = true;
                    continue;
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else { continue };
                // A thinking-enabled model's early tokens stream as
                // `delta.reasoning_content`, never `delta.content`
                // (`openai::event_delta`'s "reasoning" branch) -- both are real
                // generated tokens for TTFA/ITL purposes, so both count.
                let delta = &v["choices"][0]["delta"];
                let has_text = |field: &str| delta[field].as_str().map(|s| !s.is_empty()).unwrap_or(false);
                if has_text("content") || has_text("reasoning_content") {
                    let _ = tx.send(Emission { id, at, kind: EmissionKind::Artifact });
                }
            }
        }
    }
    let kind = if saw_done { EmissionKind::Done } else { EmissionKind::Failed };
    let _ = tx.send(Emission { id, at: Instant::now(), kind });
}
