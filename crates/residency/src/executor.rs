// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The [`Executor`] — brain's general model-execution layer. Every path (the CLI,
//! the event runtime, the D-Bus surface) submits [`Job`]s here instead of calling
//! models directly, so scheduling, residency, and batching are shared and uniform.
//!
//! A background worker owns the [`ResidencyManager`] and a pending queue. Each round
//! it groups queued jobs by instance-key + action, asks the [`crate::scheduler`]
//! policy which group to run next (balancing batch size against queue age), and runs
//! that group on one hot instance — promoting/evicting via the manager and reusing
//! the hot path across the group's jobs. Replies and progress go back through each
//! job's callbacks, so this crate needs no async runtime (the D-Bus side adapts to
//! Tokio, the CLI to a channel).
//!
//! One worker for now (jobs serialize on the single GPU pipeline, which is the
//! bottleneck anyway); the multi-device parallel-lane variant — one worker per device
//! plus a per-GPU serialize — is a drop-in extension noted in the plan.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use capability::{ActionResult, Invocation, Manifest, Progress};

use crate::scheduler::{choose_next, Group, Policy};
use crate::{InstanceKey, ResidencyManager, ResidentModel};

/// One unit of work. `on_progress`/`reply` are callbacks (no async dependency here).
pub struct Job {
    pub model: String,
    pub action: String,
    pub inv: Invocation,
    pub on_progress: Box<dyn FnMut(Progress) + Send>,
    pub reply: Box<dyn FnOnce(ActionResult) + Send>,
}

struct Pending {
    model: String,
    action: String,
    inv: Invocation,
    key: InstanceKey,
    enqueued: Instant,
    on_progress: Box<dyn FnMut(Progress) + Send>,
    reply: Box<dyn FnOnce(ActionResult) + Send>,
}

/// Cheap-to-clone submission handle (many front-ends can submit concurrently).
#[derive(Clone)]
pub struct Executor {
    tx: Sender<Job>,
    manifests: Arc<Vec<Manifest>>,
}

impl Executor {
    /// Build over a set of resident models + a policy, and start the worker thread.
    pub fn start(models: Vec<Arc<dyn ResidentModel>>, budgets: crate::budget::Budgets, policy: Policy) -> Executor {
        let manifests: Vec<Manifest> = models.iter().map(|m| m.manifest()).collect();
        let mut mgr = ResidencyManager::new(budgets);
        for m in models {
            mgr.register(m);
        }
        let (tx, rx) = channel::<Job>();
        std::thread::Builder::new()
            .name("brain-executor".into())
            .spawn(move || worker_loop(rx, mgr, policy))
            .expect("spawn executor worker");
        Executor { tx, manifests: Arc::new(manifests) }
    }

    /// Submit a job; the result arrives via `job.reply`.
    pub fn submit(&self, job: Job) {
        let _ = self.tx.send(job);
    }

    /// Submit and block for the result (the CLI's synchronous path).
    pub fn run_blocking(&self, model: &str, action: &str, inv: Invocation, on_progress: impl FnMut(Progress) + Send + 'static) -> ActionResult {
        let (rtx, rrx) = channel();
        self.submit(Job {
            model: model.into(),
            action: action.into(),
            inv,
            on_progress: Box::new(on_progress),
            reply: Box::new(move |r| {
                let _ = rtx.send(r);
            }),
        });
        rrx.recv().unwrap_or_else(|_| Err("executor worker gone".into()))
    }

    pub fn manifests(&self) -> &[Manifest] {
        &self.manifests
    }
}

/// One group's identity + summary, kept together so the policy's chosen index maps
/// straight back to the queue filter.
struct GroupRow {
    key: InstanceKey,
    model: String,
    action: String,
    summary: Group,
}

fn worker_loop(rx: Receiver<Job>, mut mgr: ResidencyManager, policy: Policy) {
    let mut queue: Vec<Pending> = Vec::new();
    loop {
        if queue.is_empty() {
            match rx.recv() {
                Ok(job) => enqueue(&mgr, &mut queue, job),
                Err(_) => return,
            }
        }
        while let Ok(job) = rx.try_recv() {
            enqueue(&mgr, &mut queue, job);
        }
        if queue.is_empty() {
            continue;
        }

        let now = Instant::now();
        let rows = group_rows(&queue, now);
        let summaries: Vec<Group> = rows.iter().map(|r| r.summary.clone()).collect();
        let (gid, batch) = match choose_next(&summaries, &policy) {
            Some(x) => x,
            None => continue,
        };
        let (key, model, action) = (rows[gid].key.clone(), rows[gid].model.clone(), rows[gid].action.clone());

        // Pull the group's oldest `batch` jobs out of the queue (submission order).
        let mut idxs: Vec<usize> = queue.iter().enumerate().filter(|(_, p)| p.key == key && p.action == action).map(|(i, _)| i).collect();
        idxs.sort_by_key(|&i| queue[i].enqueued);
        idxs.truncate(batch);
        idxs.sort_unstable_by(|a, b| b.cmp(a)); // remove high→low
        let mut jobs: Vec<Pending> = idxs.into_iter().map(|i| queue.remove(i)).collect();
        jobs.reverse();

        run_group(&mut mgr, &model, &action, jobs);
    }
}

fn enqueue(mgr: &ResidencyManager, queue: &mut Vec<Pending>, job: Job) {
    match mgr.instance_key_for(&job.model, &job.action, &job.inv) {
        Some(key) => queue.push(Pending {
            model: job.model,
            action: job.action,
            inv: job.inv,
            key,
            enqueued: Instant::now(),
            on_progress: job.on_progress,
            reply: job.reply,
        }),
        None => (job.reply)(Err(format!("no model '{}'", job.model))),
    }
}

fn group_rows(queue: &[Pending], now: Instant) -> Vec<GroupRow> {
    let mut rows: Vec<GroupRow> = Vec::new();
    for p in queue {
        let age = now.saturating_duration_since(p.enqueued).as_millis() as u64;
        match rows.iter_mut().find(|r| r.key == p.key && r.action == p.action) {
            Some(r) => {
                r.summary.size += 1;
                r.summary.oldest_age_ms = r.summary.oldest_age_ms.max(age);
            }
            None => {
                let id = rows.len();
                rows.push(GroupRow {
                    key: p.key.clone(),
                    model: p.model.clone(),
                    action: p.action.clone(),
                    summary: Group { id, oldest_age_ms: age, size: 1 },
                });
            }
        }
    }
    rows
}

fn run_group(mgr: &mut ResidencyManager, model: &str, action: &str, jobs: Vec<Pending>) {
    let invs: Vec<Invocation> = jobs.iter().map(|p| p.inv.clone()).collect();
    // Separate the callbacks: progress sinks (broadcast) and one-shot replies.
    let mut sinks: Vec<Box<dyn FnMut(Progress) + Send>> = Vec::with_capacity(jobs.len());
    let mut replies: Vec<Box<dyn FnOnce(ActionResult) + Send>> = Vec::with_capacity(jobs.len());
    for p in jobs {
        sinks.push(p.on_progress);
        replies.push(p.reply);
    }
    let results = {
        let mut fanout = |pr: Progress| {
            for s in sinks.iter_mut() {
                s(pr.clone());
            }
        };
        mgr.run_batch(model, action, &invs, &mut fanout)
    };
    match results {
        Ok(mut outs) => {
            // A well-behaved model returns one result per invocation.
            if outs.len() != replies.len() {
                let err = format!("model '{model}' returned {} results for {} jobs", outs.len(), replies.len());
                for r in replies {
                    r(Err(err.clone()));
                }
            } else {
                for (r, o) in replies.into_iter().zip(outs.drain(..)) {
                    r(o);
                }
            }
        }
        Err(e) => {
            for r in replies {
                r(Err(e.clone()));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::Budgets;
    use crate::{Device, MemCost};
    use capability::{ActionResult, ActionSpec, Blob, Media, Outcome};
    use std::sync::atomic::{AtomicU32, Ordering};

    const GB: u64 = 1 << 30;

    /// Model that records how many times each instance was BUILT (activate) — so a
    /// test can prove that a group of jobs reused one hot instance.
    struct Fake {
        name: String,
        builds: Arc<AtomicU32>,
    }
    struct FakeInst {
        n: String,
        batch_runs: Arc<AtomicU32>,
    }
    impl ResidentModel for Fake {
        fn manifest(&self) -> Manifest {
            Manifest::new(&self.name, "fake", vec![ActionSpec::new("run", "run")])
        }
        fn instance_key(&self, _a: &str, _i: &Invocation) -> InstanceKey {
            InstanceKey::new(&self.name, "default")
        }
        fn estimate(&self, _k: &InstanceKey) -> MemCost {
            MemCost::new(GB, 0)
        }
        fn activate(&self, _k: &InstanceKey, _d: Device) -> Result<Box<dyn crate::Instance>, String> {
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(FakeInst { n: self.name.clone(), batch_runs: Arc::new(AtomicU32::new(0)) }))
        }
    }
    impl crate::Instance for FakeInst {
        fn run(&mut self, _a: &str, _i: &Invocation, _p: &mut dyn FnMut(Progress)) -> ActionResult {
            Ok(Outcome::new().set("model", serde_json::json!(self.n)).blob("out", Blob::new(Media::Bytes, vec![1])))
        }
        fn run_batch(&mut self, action: &str, invs: &[Invocation], p: &mut dyn FnMut(Progress)) -> Vec<ActionResult> {
            self.batch_runs.fetch_add(1, Ordering::SeqCst);
            invs.iter().map(|i| self.run(action, i, p)).collect()
        }
    }

    #[test]
    fn jobs_to_one_model_reuse_a_single_hot_build_and_all_reply() {
        let builds = Arc::new(AtomicU32::new(0));
        let mut budgets = Budgets::new();
        budgets.set(Device::Gpu(0), 24 * GB, 0);
        let m: Arc<dyn ResidentModel> = Arc::new(Fake { name: "a".into(), builds: builds.clone() });
        let exec = Executor::start(vec![m], budgets, Policy::default());

        // Fire 16 jobs; collect replies.
        let (tx, rx) = channel();
        for _ in 0..16 {
            let tx = tx.clone();
            exec.submit(Job {
                model: "a".into(),
                action: "run".into(),
                inv: Invocation::new(),
                on_progress: Box::new(|_| {}),
                reply: Box::new(move |r| {
                    let _ = tx.send(r);
                }),
            });
        }
        drop(tx);
        let mut got = 0;
        while let Ok(r) = rx.recv() {
            assert!(r.is_ok());
            got += 1;
            if got == 16 {
                break;
            }
        }
        assert_eq!(got, 16, "every job replied");
        // Hot-path reuse: the instance was built at most a handful of times (not 16).
        assert!(builds.load(Ordering::SeqCst) <= 3, "too many rebuilds: {}", builds.load(Ordering::SeqCst));
    }

    #[test]
    fn unknown_model_replies_error() {
        let exec = Executor::start(vec![], Budgets::new(), Policy::default());
        let r = exec.run_blocking("nope", "x", Invocation::new(), |_| {});
        assert!(r.is_err());
    }
}
