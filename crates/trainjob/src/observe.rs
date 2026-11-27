// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for seeing where a fine-tune's wall
// time goes on a shared accelerator, for its clients. If your team needs
// expertise in efficient training beside production serving then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A training job's account of its own wall time, published through its invocation's report
//! ([`capability::report`]) so a server lists it with the job (`training_jobs` in its metrics document).
//!
//! A production stage spent most of its wall time outside optimizer steps and nobody could say where (knowledge
//! #617). The split here answers it: every second of the job's wall time is in exactly one phase,
//!
//! * `waiting_for_card`: queued for its device(s), and backing off after the card refused an allocation;
//! * `staging`: datasets and starting adapters fetched and verified into the job's directory;
//! * `model_load`: building the trainer before its first step: the base weights, the tokenized datasets;
//! * `optimizer_steps`: inside optimizer steps (batches drawn, forward, backward, optimizer);
//! * `evaluation`: held-out and monitoring passes, with the snapshots and best parameters they keep;
//! * `checkpoint_save`: checkpoints, the training state and published adapter versions;
//! * `waiting_for_records`: a streamed run waiting for its next records;
//! * `finalize`: after the last step: the final save and publishing the adapter;
//! * `other`: whatever the rest does not name (the residual, so the phases always add up to the wall time).
//!
//! The observer reads only the job's own progress frames: the `phase` a `stage` frame names and the `time` a `step`
//! frame carries (the training loop's own accounting, [`brain::LoopTime`]), so every trainer that streams its steps is
//! covered without code of its own.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use capability::report::JobReport;
use capability::Progress;
use serde_json::{json, Value};

/// The phases of a job's wall time, in the order a metrics document lists them.
pub const PHASES: [&str; 9] = ["waiting_for_card", "staging", "model_load", "optimizer_steps", "evaluation", "checkpoint_save", "waiting_for_records", "finalize", "other"];

/// The `phase` a `stage` frame names for the datasets being staged.
pub const STAGING: &str = "staging";
/// The `phase` a `stage` frame names when a training attempt starts building its trainer.
pub const MODEL_LOAD: &str = "model_load";
/// The `phase` a `stage` frame names for a wait for the card (a back-off after it refused an allocation).
pub const WAITING_FOR_CARD: &str = "waiting_for_card";

/// The training loop's own accounting, as a `step` frame's `time` carries it ([`brain::LoopTime`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Loop {
    elapsed: f64,
    steps: f64,
    evaluation: f64,
    checkpoint: f64,
    waiting_for_records: f64,
    tokens: u64,
}

impl Loop {
    fn of(time: &Value) -> Loop {
        let secs = |name: &str| time.get(name).and_then(Value::as_f64).unwrap_or(0.0).max(0.0);
        Loop {
            elapsed: secs("elapsed_secs"),
            steps: secs("steps_secs"),
            evaluation: secs("evaluation_secs"),
            checkpoint: secs("checkpoint_secs"),
            waiting_for_records: secs("waiting_for_records_secs"),
            tokens: time.get("useful_tokens").and_then(Value::as_u64).unwrap_or(0),
        }
    }
}

/// [`brain::LoopTime`] as a step frame's `time`.
pub fn loop_time_json(time: &brain::LoopTime) -> Value {
    json!({
        "elapsed_secs": time.elapsed_secs,
        "steps_secs": time.steps_secs,
        "evaluation_secs": time.evaluation_secs,
        "checkpoint_secs": time.checkpoint_secs,
        "waiting_for_records_secs": time.waiting_for_records_secs,
        "useful_tokens": time.useful_tokens,
    })
}

/// One attempt at training (a retry after the card refused an allocation is another): when it started, and the loop's
/// accounting as of its last step and when that step was reported.
#[derive(Clone, Copy, Debug)]
struct Attempt {
    started: Instant,
    last: Option<(Instant, Loop)>,
}

/// Seconds per phase, in [`PHASES`] order.
#[derive(Clone, Copy, Debug, Default)]
struct Phases([f64; 9]);

impl Phases {
    fn add(&mut self, phase: &str, secs: f64) {
        if let Some(i) = PHASES.iter().position(|p| *p == phase) {
            self.0[i] += secs.max(0.0);
        }
    }

    /// Fold `attempt` as of `now` into these phases: the time before its first step loaded the model; the loop's own
    /// parts are its phases; a loop that has not reported yet is all model load.
    fn fold(&mut self, attempt: &Attempt, now: Instant) -> u64 {
        match attempt.last {
            None => {
                self.add(MODEL_LOAD, now.duration_since(attempt.started).as_secs_f64());
                0
            }
            Some((at, l)) => {
                self.add(MODEL_LOAD, at.duration_since(attempt.started).as_secs_f64() - l.elapsed);
                self.add("optimizer_steps", l.steps);
                self.add("evaluation", l.evaluation);
                self.add("checkpoint_save", l.checkpoint);
                self.add("waiting_for_records", l.waiting_for_records);
                l.tokens
            }
        }
    }
}

/// Follows a job's progress frames and publishes its account to the job's report. See the module doc.
pub struct RunObserver {
    report: JobReport,
    action: String,
    devices: Vec<String>,
    started: Instant,
    queued: Duration,
    /// Phases closed so far (stages done, attempts that ended).
    closed: Phases,
    /// Tokens the ended attempts trained on.
    closed_tokens: u64,
    /// The stage running now, outside an attempt, and since when.
    stage: Option<(&'static str, Instant)>,
    attempt: Option<Attempt>,
    /// When the last step of the job was reported: what follows it, once the job ends, finalizes it.
    last_step_at: Option<Instant>,
    steps: (u32, u32),
    oom_retries: u32,
    ended: Option<Instant>,
}

/// A frame's `phase`, as one of the known names.
fn phase_named(name: &str) -> Option<&'static str> {
    [STAGING, MODEL_LOAD, WAITING_FOR_CARD].into_iter().find(|p| *p == name)
}

fn unix_ms(at: Instant) -> u64 {
    let ago = Instant::now().saturating_duration_since(at);
    SystemTime::now().checked_sub(ago).and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map_or(0, |d| d.as_millis() as u64)
}

impl RunObserver {
    /// The observer of a job running `action` on `devices` that started at `started`, publishing to `report` (which
    /// says how long the job waited for its devices).
    pub fn new(report: JobReport, action: &str, devices: Vec<String>, started: Instant) -> RunObserver {
        let queued = report.admitted_after().unwrap_or_default();
        let observer = RunObserver { report, action: action.to_string(), devices, started, queued, closed: Phases::default(), closed_tokens: 0, stage: None, attempt: None, last_step_at: None, steps: (0, 0), oom_retries: 0, ended: None };
        observer.publish(started);
        observer
    }

    /// Close whatever runs now (a stage, or an attempt) at `now`.
    fn close(&mut self, now: Instant) {
        if let Some((phase, since)) = self.stage.take() {
            self.closed.add(phase, now.duration_since(since).as_secs_f64());
        }
        if let Some(attempt) = self.attempt.take() {
            self.closed_tokens += self.closed.fold(&attempt, now);
            // What the attempt did after its last step before it ended (a failed allocation's unwind) is not named.
        }
    }

    /// Read one progress frame of the job, seen at `now`.
    pub fn frame(&mut self, progress: &Progress, now: Instant) {
        let Some(event) = &progress.event else { return };
        match event.get("type").and_then(Value::as_str) {
            Some("stage") => {
                let Some(phase) = event.get("phase").and_then(Value::as_str).and_then(phase_named) else { return };
                self.close(now);
                if phase == WAITING_FOR_CARD {
                    self.oom_retries += 1;
                }
                if phase == MODEL_LOAD {
                    self.attempt = Some(Attempt { started: now, last: None });
                } else {
                    self.stage = Some((phase, now));
                }
            }
            Some("step") => {
                let step = |name: &str| event.get(name).and_then(Value::as_u64).and_then(|n| u32::try_from(n).ok());
                self.steps = (step("step").unwrap_or(self.steps.0), step("steps").unwrap_or(self.steps.1));
                self.last_step_at = Some(now);
                if let Some(time) = event.get("time") {
                    // A trainer that streams steps without saying it started one has its attempt start with the job.
                    if self.attempt.is_none() {
                        self.stage = None;
                        self.attempt = Some(Attempt { started: self.started, last: None });
                    }
                    if let Some(attempt) = self.attempt.as_mut() {
                        attempt.last = Some((now, Loop::of(time)));
                    }
                }
            }
            _ => return,
        }
        self.publish(now);
    }

    /// The job ended at `now`: what followed its last step finalized it.
    pub fn end(&mut self, now: Instant) {
        let last_step = self.last_step_at;
        if let Some(attempt) = self.attempt.take() {
            self.closed_tokens += self.closed.fold(&attempt, last_step.unwrap_or(now).max(attempt.started));
            if let Some(at) = last_step.filter(|&at| at >= attempt.started) {
                self.closed.add("finalize", now.duration_since(at).as_secs_f64());
            }
        }
        self.close(now);
        self.ended = Some(now);
        self.publish(now);
    }

    /// The account as of `now`.
    pub fn account(&self, now: Instant) -> Value {
        let now = self.ended.unwrap_or(now);
        let mut phases = self.closed;
        let mut tokens = self.closed_tokens;
        if let Some((phase, since)) = self.stage {
            phases.add(phase, now.duration_since(since).as_secs_f64());
        }
        let current = match (&self.attempt, self.stage) {
            (Some(attempt), _) => {
                tokens += phases.fold(attempt, attempt.last.map_or(now, |(at, _)| at));
                if attempt.last.is_some() {
                    "training"
                } else {
                    MODEL_LOAD
                }
            }
            (None, Some((phase, _))) => phase,
            (None, None) if self.ended.is_some() => "ended",
            (None, None) => "other",
        };
        // Since a live attempt's last step the loop may step, evaluate or save: unnamed (`other`) until its next step
        // says which.
        phases.add(WAITING_FOR_CARD, self.queued.as_secs_f64());
        let wall = self.queued.as_secs_f64() + now.duration_since(self.started).as_secs_f64();
        let named: f64 = phases.0[..PHASES.len() - 1].iter().sum();
        phases.0[PHASES.len() - 1] = (wall - named).max(0.0);
        let steps_secs = phases.0[3];
        let split: serde_json::Map<String, Value> = PHASES.iter().zip(phases.0).map(|(p, s)| (p.to_string(), json!(s))).collect();
        json!({
            "action": self.action,
            "devices": self.devices,
            "as_of_unix_ms": unix_ms(now),
            "current_phase": current,
            "steps": {"done": self.steps.0, "total": self.steps.1},
            "optimizer_tokens": tokens,
            "optimizer_tokens_per_s": if steps_secs > 0.0 { tokens as f64 / steps_secs } else { 0.0 },
            "oom_retries": self.oom_retries,
            "wall_secs": wall,
            "outside_optimizer_steps_share": if wall > 0.0 { (1.0 - steps_secs / wall).clamp(0.0, 1.0) } else { 0.0 },
            "phases_secs": split,
        })
    }

    fn publish(&self, now: Instant) {
        self.report.publish(self.account(now));
    }
}

/// Runs `work` as the job `action` on `devices`, following its progress frames and publishing its account to `report`
/// (see the module doc); every frame still reaches `progress`.
pub fn observed<T>(report: &JobReport, action: &str, devices: Vec<String>, progress: &mut dyn FnMut(Progress), work: impl FnOnce(&mut dyn FnMut(Progress)) -> T) -> T {
    if !report.is_armed() {
        return work(progress);
    }
    let mut observer = RunObserver::new(report.clone(), action, devices, Instant::now());
    let done = {
        let mut seen = |p: Progress| {
            observer.frame(&p, Instant::now());
            progress(p)
        };
        work(&mut seen)
    };
    observer.end(Instant::now());
    done
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(phase: &str) -> Progress {
        Progress::event(0, 10, json!({"type": "stage", "stage": "said to the client", "phase": phase}))
    }

    fn step(n: u32, time: Value) -> Progress {
        Progress::event(n, 10, json!({"type": "step", "step": n, "steps": 10, "loss": 1.0, "time": time}))
    }

    fn secs(n: f64) -> Duration {
        Duration::from_secs_f64(n)
    }

    fn phase(account: &Value, name: &str) -> f64 {
        account["phases_secs"][name].as_f64().unwrap()
    }

    /// The production finding (knowledge #617) as a spec: a stage whose optimizer steps are well under half its wall
    /// time says where the rest went, phase by phase, and the phases add up to the wall time.
    #[test]
    fn every_second_of_a_jobs_wall_time_is_in_exactly_one_phase() {
        let report = JobReport::armed();
        report.set_admitted_after(secs(5.0));
        let t0 = Instant::now();
        let mut run = RunObserver::new(report.clone(), "train", vec!["gpu0".into()], t0);
        run.frame(&stage(STAGING), t0);
        run.frame(&stage(MODEL_LOAD), t0 + secs(3.0));
        // The loop starts a while after the attempt (the model load); at its 10th step its own accounting names its
        // steps, evaluation and checkpoints, and leaves a part unnamed.
        let time = json!({"elapsed_secs": 77.0, "steps_secs": 41.0, "evaluation_secs": 20.0, "checkpoint_secs": 12.0, "waiting_for_records_secs": 0.0, "useful_tokens": 41_000});
        run.frame(&step(10, time), t0 + secs(100.0));
        run.end(t0 + secs(110.0));
        let account = report.latest().expect("published");
        assert_eq!(account["current_phase"], "ended");
        assert_eq!(account["steps"], json!({"done": 10, "total": 10}));
        let expect = [("waiting_for_card", 5.0), ("staging", 3.0), ("model_load", 20.0), ("optimizer_steps", 41.0), ("evaluation", 20.0), ("checkpoint_save", 12.0), ("waiting_for_records", 0.0), ("finalize", 10.0), ("other", 4.0)];
        for (name, want) in expect {
            assert!((phase(&account, name) - want).abs() < 1e-6, "{name}: {} (want {want}): {account}", phase(&account, name));
        }
        let wall = account["wall_secs"].as_f64().unwrap();
        assert!((wall - 115.0).abs() < 1e-6);
        let total: f64 = PHASES.iter().map(|p| phase(&account, p)).sum();
        assert!((total - wall).abs() < 1e-6, "the phases add up to the wall time: {total} of {wall}");
        assert!((account["outside_optimizer_steps_share"].as_f64().unwrap() - (1.0 - 41.0 / 115.0)).abs() < 1e-9);
        assert!((account["optimizer_tokens_per_s"].as_f64().unwrap() - 1000.0).abs() < 1e-9, "tokens per second inside optimizer steps");
    }

    /// A card that refused an allocation: the attempt that failed keeps what it did, the back-off waits for the card,
    /// and the retry loads the model again; the retry count is reported.
    #[test]
    fn a_retry_after_the_card_refused_an_allocation_is_counted_and_its_wait_is_waiting_for_the_card() {
        let report = JobReport::armed();
        let t0 = Instant::now();
        let mut run = RunObserver::new(report.clone(), "train", vec![], t0);
        run.frame(&stage(MODEL_LOAD), t0);
        run.frame(&step(5, json!({"elapsed_secs": 8.0, "steps_secs": 8.0, "useful_tokens": 800})), t0 + secs(10.0));
        run.frame(&stage(WAITING_FOR_CARD), t0 + secs(12.0));
        run.frame(&stage(MODEL_LOAD), t0 + secs(16.0));
        let live = report.latest().unwrap();
        assert_eq!((live["current_phase"].as_str(), live["oom_retries"].as_u64()), (Some(MODEL_LOAD), Some(1)));
        run.frame(&step(10, json!({"elapsed_secs": 6.0, "steps_secs": 6.0, "useful_tokens": 600})), t0 + secs(25.0));
        run.end(t0 + secs(25.0));
        let account = report.latest().unwrap();
        assert!((phase(&account, "waiting_for_card") - 4.0).abs() < 1e-6, "{account}");
        assert!((phase(&account, "model_load") - (2.0 + 3.0)).abs() < 1e-6, "{account}");
        assert!((phase(&account, "optimizer_steps") - 14.0).abs() < 1e-6, "{account}");
        assert!((phase(&account, "other") - 2.0).abs() < 1e-6, "the failed attempt's unwind is not named: {account}");
        assert_eq!(account["optimizer_tokens"], 1400);
    }

    /// Read while it runs, a job says what it does now and its phases add up to the time so far.
    #[test]
    fn a_running_job_says_its_current_phase_and_its_split_so_far() {
        let report = JobReport::armed();
        let t0 = Instant::now();
        let mut run = RunObserver::new(report.clone(), "train", vec![], t0);
        assert_eq!(run.account(t0 + secs(1.0))["current_phase"], "other");
        run.frame(&stage(STAGING), t0 + secs(1.0));
        assert_eq!(run.account(t0 + secs(2.0))["current_phase"], STAGING);
        run.frame(&stage(MODEL_LOAD), t0 + secs(2.0));
        let loading = run.account(t0 + secs(7.0));
        assert_eq!(loading["current_phase"], MODEL_LOAD);
        assert!((phase(&loading, "model_load") - 5.0).abs() < 1e-6);
        run.frame(&step(1, json!({"elapsed_secs": 1.0, "steps_secs": 1.0})), t0 + secs(8.0));
        let training = run.account(t0 + secs(8.5));
        assert_eq!(training["current_phase"], "training");
        let wall = training["wall_secs"].as_f64().unwrap();
        let total: f64 = PHASES.iter().map(|p| phase(&training, p)).sum();
        assert!((wall - 8.5).abs() < 1e-6 && (total - wall).abs() < 1e-6, "{training}");
    }
}
