// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The resource-admission and fairness contract: what the dispatcher decides
//! for every job it holds that has not started yet.
//!
//! Swedish Embedded AB implements schedulers that share accelerators between
//! interactive inference and long training runs without starving either, for
//! its clients. If your team needs expertise in admission control and
//! fairness on shared GPUs then you can procure our services by sending an
//! email to info@swedishembedded.com.
//!
//! # The decisions
//!
//! Every queued job gets exactly one [`Decision`] each scheduling round:
//!
//! * [`Decision::Admit`]: claim a device for it now, wherever it fits.
//! * [`Decision::Elsewhere`]: claim a device for it now, but not one of the
//!   named devices, which are held for a workload ([`Holds`]).
//! * [`Decision::Wait`]: it stays queued (no device can take it this round);
//!   it is answered no later than its deadline.
//! * [`Decision::Shrink`]: a running group keeps the jobs it has and takes no
//!   late ones, so its batch shrinks to nothing and its device frees.
//! * [`Decision::Refuse`]: it is answered now with an error that names the
//!   setting that would have changed the outcome ([`Expired`]).
//!
//! # Queue deadlines
//!
//! Every job carries a deadline from the moment it is queued: its caller's
//! ([`crate::Job::deadline`]) or the operator's [`Contract::queue_deadline`],
//! whichever is earlier. A job still queued when its deadline passes is
//! refused with a definitive error ([`capability::failure::FailureClass::Timeout`])
//! naming [`QUEUE_DEADLINE_FLAG`]. "Queued" includes a job waiting in a
//! running group's inbox that the group has not taken yet: it has no device
//! slot either. A queued job holds no memory; what it may hold is a fairness
//! [`Holds`] entry, which leaves with it.
//!
//! # Fairness for workloads
//!
//! A workload ([`crate::ResidentModel::transient`]: a fine-tune) holds a whole
//! device while it runs, so it can only start once a device is idle. Under
//! continuous inference traffic a device need never be idle: every finished
//! request is replaced by a new one. The contract therefore guarantees a
//! workload a slot: once it has waited [`Contract::workload_max_wait`], the
//! devices it could run on are held for it. Serving jobs are then admitted
//! only elsewhere and running serving groups on those devices take no late
//! jobs, so the first held device to drain goes to the workload. The hold is
//! released the moment the workload is claimed, cancelled, refused or past its
//! deadline, so serving is throttled for at most the time it takes one held
//! device to drain, and never after the workload has left the queue.
//!
//! The guarantee: a workload that fits a device is claimed within
//! `workload_max_wait` plus the longest serving request already running on a
//! held device. `crates/residency/tests/it/fairness_model.rs` checks it, the
//! deadlines and the conservation of holds over every interleaving of small
//! bounds, with the scheduler's choice among runnable groups left to an
//! adversary.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::admission::DEFAULT_COLD_BUILD_ADMIT_DEADLINE;
use crate::Device;

/// Default [`Contract::queue_deadline`]: a minute past the longest a
/// transport's front door ever waits for admission (a cold build of the
/// request's own model), so a request a front door is still willing to wait
/// for is never refused inside the executor first, and a front door's own
/// refusal, which names its own setting, answers before this one does.
pub const DEFAULT_QUEUE_DEADLINE: Duration = Duration::from_secs(DEFAULT_COLD_BUILD_ADMIT_DEADLINE.as_secs() + 60);

/// Default [`Contract::workload_max_wait`]: long enough that a workload
/// arriving at a busy server does not interrupt a burst of requests, short
/// enough that it starts well within the front door's admission deadline.
pub const DEFAULT_WORKLOAD_MAX_WAIT: Duration = Duration::from_secs(60);

/// The serve flag that sets [`Contract::queue_deadline`].
pub const QUEUE_DEADLINE_FLAG: &str = "--queue-deadline-ms";
/// The serve flag that sets [`Contract::workload_max_wait`].
pub const WORKLOAD_MAX_WAIT_FLAG: &str = "--train-max-wait-ms";

/// The operator's admission settings. Both are positive: the command line
/// refuses zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contract {
    /// The longest any job waits in the executor to be admitted.
    pub queue_deadline: Duration,
    /// How long a workload waits for a device before serving admission on the
    /// devices it could use is held for it.
    pub workload_max_wait: Duration,
}

impl Default for Contract {
    fn default() -> Contract {
        Contract { queue_deadline: DEFAULT_QUEUE_DEADLINE, workload_max_wait: DEFAULT_WORKLOAD_MAX_WAIT }
    }
}

/// Which promise a job is owed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// A request a resident model answers.
    Serving,
    /// A run that holds a device only while it runs ([`crate::ResidentModel::transient`]).
    Workload,
}

/// Where a job's deadline came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeadlineSource {
    /// The operator's [`Contract::queue_deadline`].
    Operator,
    /// The job's own caller ([`crate::Job::deadline`]), earlier than the operator's.
    Caller,
}

/// A queued job as the contract sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Waiting {
    pub id: u64,
    pub class: Class,
    pub enqueued: Instant,
    pub deadline: Instant,
    pub source: DeadlineSource,
}

/// Why a job was refused: it was not admitted before its deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expired {
    pub waited: Duration,
    pub limit: Duration,
    pub source: DeadlineSource,
}

impl Expired {
    /// The caller's error: the fact, the limit and the setting that changes it,
    /// tagged [`capability::failure::FailureClass::Timeout`].
    pub fn message(&self, model: &str, action: &str) -> String {
        let setting = match self.source {
            DeadlineSource::Operator => format!("raise {QUEUE_DEADLINE_FLAG} if such work should wait longer"),
            DeadlineSource::Caller => "the deadline was the caller's own".to_string(),
        };
        let text = format!(
            "{model}/{action}: not admitted within its queue deadline: it waited {} ms for a device and the limit is {} ms; {setting}",
            self.waited.as_millis(),
            self.limit.as_millis()
        );
        capability::failure::tag(capability::failure::FailureClass::Timeout, text)
    }
}

/// What the dispatcher does with a job (or, for [`Decision::Shrink`], a running group) this round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// Claim a device now, wherever it fits.
    Admit,
    /// Claim a device now, but none of `avoid`: they are held for a workload.
    Elsewhere { avoid: Vec<Device> },
    /// Stay queued; answered no later than `deadline`.
    Wait { deadline: Instant },
    /// Keep running without taking late jobs.
    Shrink,
    /// Answer now with this error.
    Refuse(Expired),
}

impl Contract {
    /// The deadline of a job queued at `enqueued` whose caller asked for
    /// `requested` (`None`: no deadline of its own). The earlier one applies.
    pub fn deadline(&self, enqueued: Instant, requested: Option<Instant>) -> (Instant, DeadlineSource) {
        let operator = enqueued + self.queue_deadline;
        match requested {
            Some(r) if r < operator => (r, DeadlineSource::Caller),
            _ => (operator, DeadlineSource::Operator),
        }
    }

    /// Whether `w` is a workload that has waited its guaranteed time.
    pub fn owed(&self, w: &Waiting, now: Instant) -> bool {
        w.class == Class::Workload && now.saturating_duration_since(w.enqueued) >= self.workload_max_wait
    }

    /// The decision for the queued job `w` at `now`, given the devices held
    /// for workloads. [`Decision::Wait`] is never returned here: it is what an
    /// admitted job becomes when no device can take it this round.
    pub fn decide(&self, w: &Waiting, now: Instant, holds: &Holds) -> Decision {
        if now >= w.deadline {
            return Decision::Refuse(Expired { waited: now.saturating_duration_since(w.enqueued), limit: w.deadline.saturating_duration_since(w.enqueued), source: w.source });
        }
        match w.class {
            Class::Serving if !holds.is_empty() => Decision::Elsewhere { avoid: holds.devices() },
            Class::Serving | Class::Workload => Decision::Admit,
        }
    }

    /// Whether a running group on `devices` may take late jobs: [`Decision::Shrink`]
    /// while any of them is held for a workload, else [`Decision::Admit`].
    pub fn late_joins(&self, devices: &[Device], holds: &Holds) -> Decision {
        if devices.iter().any(|d| holds.holds(*d)) {
            Decision::Shrink
        } else {
            Decision::Admit
        }
    }

    /// The next instant after `now` at which a decision about `waiting` changes
    /// by the passage of time alone: a deadline passing, or a workload becoming
    /// owed. The dispatcher wakes then, so a deadline is honoured to the
    /// moment, not to its next idle tick.
    ///
    /// A deadline that has already passed is returned as it is (the job is
    /// answered at the next round, so it cannot be returned twice); a workload
    /// that is already owed is not.
    pub fn next_change<'a>(&self, waiting: impl IntoIterator<Item = &'a Waiting>, now: Instant) -> Option<Instant> {
        waiting
            .into_iter()
            .flat_map(|w| {
                let owed = (w.class == Class::Workload).then(|| w.enqueued + self.workload_max_wait).filter(|t| *t > now);
                [Some(w.deadline), owed]
            })
            .flatten()
            .min()
    }
}

/// The devices held for workloads that have waited their guaranteed time, by
/// the job each is held for. An entry exists only while its job is queued:
/// [`Holds::reconcile`] adds and removes them against the queue, so a hold
/// can never outlive the job that earned it.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Holds {
    by_job: BTreeMap<u64, Vec<Device>>,
}

/// A hold that [`Holds::reconcile`] took or released, for the log.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HoldChange {
    Taken { job: u64, devices: Vec<Device> },
    Released { job: u64 },
}

impl Holds {
    /// Bring the holds in line with `waiting` at `now`: one for every owed
    /// workload still queued, on the devices `candidates` names for it (the
    /// ones it could run on), and none for a job that is no longer queued. A
    /// workload no device could ever hold gets no hold: throttling serving for
    /// it would starve serving and help nobody.
    pub fn reconcile<'a>(&mut self, contract: &Contract, waiting: impl IntoIterator<Item = &'a Waiting>, now: Instant, mut candidates: impl FnMut(u64) -> Vec<Device>) -> Vec<HoldChange> {
        let waiting: Vec<&Waiting> = waiting.into_iter().collect();
        let mut changes = Vec::new();
        let gone: Vec<u64> = self.by_job.keys().copied().filter(|id| !waiting.iter().any(|w| w.id == *id && w.class == Class::Workload && now < w.deadline)).collect();
        for job in gone {
            self.by_job.remove(&job);
            changes.push(HoldChange::Released { job });
        }
        for w in waiting.iter().filter(|w| contract.owed(w, now) && now < w.deadline) {
            if self.by_job.contains_key(&w.id) {
                continue;
            }
            let mut devices = candidates(w.id);
            devices.dedup();
            if devices.is_empty() {
                continue;
            }
            changes.push(HoldChange::Taken { job: w.id, devices: devices.clone() });
            self.by_job.insert(w.id, devices);
        }
        changes
    }

    /// Release the hold of `job` (it was claimed, cancelled or refused).
    pub fn release(&mut self, job: u64) -> bool {
        self.by_job.remove(&job).is_some()
    }

    pub fn is_empty(&self) -> bool {
        self.by_job.is_empty()
    }

    /// Whether `device` is held for some workload.
    pub fn holds(&self, device: Device) -> bool {
        self.by_job.values().any(|ds| ds.contains(&device))
    }

    /// Every held device, once each, in the order the holds were taken.
    pub fn devices(&self) -> Vec<Device> {
        let mut out: Vec<Device> = Vec::new();
        for d in self.by_job.values().flatten() {
            if !out.contains(d) {
                out.push(*d);
            }
        }
        out
    }

    /// The jobs a hold is kept for.
    pub fn jobs(&self) -> impl Iterator<Item = u64> + '_ {
        self.by_job.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    fn contract() -> Contract {
        Contract { queue_deadline: Duration::from_millis(1000), workload_max_wait: Duration::from_millis(100) }
    }

    fn waiting(c: &Contract, id: u64, class: Class, enqueued: Instant) -> Waiting {
        let (deadline, source) = c.deadline(enqueued, None);
        Waiting { id, class, enqueued, deadline, source }
    }

    #[test]
    fn a_caller_deadline_applies_only_when_it_is_earlier_than_the_operators() {
        let (c, t0) = (contract(), Instant::now());
        assert_eq!(c.deadline(t0, None), (at(t0, 1000), DeadlineSource::Operator));
        assert_eq!(c.deadline(t0, Some(at(t0, 10))), (at(t0, 10), DeadlineSource::Caller));
        assert_eq!(c.deadline(t0, Some(at(t0, 5000))), (at(t0, 1000), DeadlineSource::Operator), "a caller cannot wait past the operator's limit");
    }

    #[test]
    fn a_job_past_its_deadline_is_refused_naming_the_flag() {
        let (c, t0) = (contract(), Instant::now());
        let w = waiting(&c, 1, Class::Serving, t0);
        assert_eq!(c.decide(&w, at(t0, 999), &Holds::default()), Decision::Admit);
        let Decision::Refuse(why) = c.decide(&w, at(t0, 1000), &Holds::default()) else { panic!("must refuse at the deadline") };
        let msg = why.message("m", "generate");
        assert!(msg.contains(QUEUE_DEADLINE_FLAG) && msg.contains("1000 ms"), "{msg}");
        assert_eq!(capability::failure::classify(&msg).map(|(class, _)| class), Some(capability::failure::FailureClass::Timeout));
    }

    #[test]
    fn an_owed_workload_holds_its_devices_until_it_leaves_the_queue() {
        let (c, t0) = (contract(), Instant::now());
        let train = waiting(&c, 7, Class::Workload, t0);
        let serve = waiting(&c, 8, Class::Serving, t0);
        let mut holds = Holds::default();
        holds.reconcile(&c, [&train, &serve], at(t0, 99), |_| vec![Device::Gpu(0)]);
        assert!(holds.is_empty(), "not owed before its guaranteed wait");
        assert_eq!(c.decide(&serve, at(t0, 99), &holds), Decision::Admit);

        let changes = holds.reconcile(&c, [&train, &serve], at(t0, 100), |_| vec![Device::Gpu(0)]);
        assert_eq!(changes, vec![HoldChange::Taken { job: 7, devices: vec![Device::Gpu(0)] }]);
        assert_eq!(c.decide(&serve, at(t0, 100), &holds), Decision::Elsewhere { avoid: vec![Device::Gpu(0)] });
        assert_eq!(c.decide(&train, at(t0, 100), &holds), Decision::Admit);
        assert_eq!(c.late_joins(&[Device::Gpu(0)], &holds), Decision::Shrink);
        assert_eq!(c.late_joins(&[Device::Gpu(1)], &holds), Decision::Admit);

        // The workload left the queue (claimed, cancelled or refused): its hold goes with it.
        let changes = holds.reconcile(&c, [&serve], at(t0, 150), |_| vec![Device::Gpu(0)]);
        assert_eq!(changes, vec![HoldChange::Released { job: 7 }]);
        assert!(holds.is_empty());
    }

    #[test]
    fn a_hold_ends_at_its_workloads_deadline_and_none_is_taken_for_a_workload_no_device_fits() {
        let (c, t0) = (contract(), Instant::now());
        let train = waiting(&c, 1, Class::Workload, t0);
        let mut holds = Holds::default();
        holds.reconcile(&c, [&train], at(t0, 500), |_| vec![Device::Gpu(0), Device::Gpu(1)]);
        assert_eq!(holds.devices(), vec![Device::Gpu(0), Device::Gpu(1)]);
        holds.reconcile(&c, [&train], at(t0, 1000), |_| vec![Device::Gpu(0)]);
        assert!(holds.is_empty(), "an expired workload throttles nothing");

        let other = waiting(&c, 2, Class::Workload, t0);
        holds.reconcile(&c, [&other], at(t0, 500), |_| Vec::new());
        assert!(holds.is_empty(), "no device could ever run it");
    }

    #[test]
    fn the_next_change_is_the_nearest_deadline_or_owed_time() {
        let (c, t0) = (contract(), Instant::now());
        let train = waiting(&c, 1, Class::Workload, t0);
        let serve = waiting(&c, 2, Class::Serving, at(t0, 50));
        assert_eq!(c.next_change([&train, &serve], t0), Some(at(t0, 100)));
        assert_eq!(c.next_change([&train, &serve], at(t0, 100)), Some(at(t0, 1000)));
        assert_eq!(c.next_change([&serve], at(t0, 2000)), Some(at(t0, 1050)), "a passed deadline is due at once");
        assert_eq!(c.next_change(std::iter::empty(), t0), None);
    }
}
