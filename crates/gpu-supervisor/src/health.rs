// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping accelerator fleets
// serving through driver faults for its clients. If your team needs expertise
// in fault isolation and recovery policy then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The shared health registry: one [`Breaker`] per device, and the only way a
//! supervised call is made ([`Health::call`]), so admission, fault recording
//! and reaping cannot be forgotten by a caller.

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::process::Child;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::breaker::{Breaker, BreakerPolicy, BreakerState};
use crate::worker::{FaultKind, Worker, WorkerCommand, WorkerFailure};

/// A device fault: what was being done, how it failed, and what became of
/// the worker. Always definitive: the operation is not retried.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fault {
    pub device: String,
    /// The request's verb (its first word).
    pub op: String,
    pub kind: FaultKind,
    pub pid: u32,
    /// Whether the killed worker was reaped within the bound. When false the
    /// registry keeps polling it and holds the device's probes back.
    pub reaped: bool,
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let fate = if self.reaped { "was killed and reaped" } else { "was killed but is not reaped yet (still inside the driver)" };
        write!(f, "{}: `{}` {}; worker {} {fate}; the device is quarantined until a health probe passes", self.device, self.op, self.kind, self.pid)
    }
}

/// Why [`Health::call`] did not return an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CallError {
    /// The device is quarantined; nothing was sent.
    Refused { device: String, cause: String },
    /// The worker answered with an error. The device is healthy.
    Failed(String),
    /// The request itself was unusable; nothing was sent.
    Invalid(String),
    /// The device faulted; the breaker is open.
    Fault(Fault),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::Refused { device, cause } => write!(f, "{device} is quarantined until a health probe passes (last fault: {cause})"),
            CallError::Failed(m) | CallError::Invalid(m) => f.write_str(m),
            CallError::Fault(fault) => std::fmt::Display::fmt(fault, f),
        }
    }
}

impl std::error::Error for CallError {}

/// A snapshot of one device's health.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceStatus {
    pub state: BreakerState,
    pub cause: Option<String>,
    pub tripped: bool,
    /// Faults recorded since the registry was created. A caller compares two
    /// reads to tell whether a fault happened in between.
    pub faults: u64,
    /// Killed workers the kernel has not released yet.
    pub unreaped: usize,
}

struct Entry {
    breaker: Breaker,
    faults: u64,
    unreaped: Vec<Child>,
}

impl Entry {
    /// Reap what the kernel has let go of since the last look.
    fn reap(&mut self) {
        self.unreaped.retain_mut(|c| matches!(c.try_wait(), Ok(None)));
    }
}

/// One breaker per device, keyed by the caller's device type. Devices are
/// healthy until a fault is recorded against them.
pub struct Health<K> {
    policy: BreakerPolicy,
    reap_within: Duration,
    devices: Mutex<HashMap<K, Entry>>,
}

impl<K: Eq + Hash + Clone + Debug> Health<K> {
    /// `reap_within` bounds how long a killed worker is waited for before it
    /// is kept for later polling.
    pub fn new(policy: BreakerPolicy, reap_within: Duration) -> Health<K> {
        Health { policy, reap_within, devices: Mutex::new(HashMap::new()) }
    }

    pub fn policy(&self) -> &BreakerPolicy {
        &self.policy
    }

    fn lock(&self) -> MutexGuard<'_, HashMap<K, Entry>> {
        self.devices.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn with<T>(&self, device: &K, f: impl FnOnce(&mut Entry) -> T) -> T {
        let mut devices = self.lock();
        let entry = devices.entry(device.clone()).or_insert_with(|| Entry { breaker: Breaker::new(self.policy.clone()), faults: 0, unreaped: Vec::new() });
        f(entry)
    }

    /// Whether work may start on `device`.
    pub fn admits(&self, device: &K) -> bool {
        self.lock().get(device).is_none_or(|e| e.breaker.admits())
    }

    /// The devices that do not admit work now.
    pub fn quarantined(&self) -> Vec<K> {
        self.lock().iter().filter(|(_, e)| !e.breaker.admits()).map(|(k, _)| k.clone()).collect()
    }

    pub fn status(&self, device: &K) -> DeviceStatus {
        self.with(device, |e| {
            e.reap();
            DeviceStatus { state: e.breaker.state(), cause: e.breaker.cause().map(str::to_string), tripped: e.breaker.tripped(), faults: e.faults, unreaped: e.unreaped.len() }
        })
    }

    /// Faults recorded against `device` so far.
    pub fn faults(&self, device: &K) -> u64 {
        self.lock().get(device).map_or(0, |e| e.faults)
    }

    /// Record a fault observed outside a supervised call (an in-process
    /// driver call that timed out, say): the device is quarantined like after
    /// any other fault.
    pub fn record_fault(&self, device: &K, now: Instant, cause: &str) {
        self.with(device, |e| self.note_fault(device, e, now, cause, None));
    }

    fn note_fault(&self, device: &K, e: &mut Entry, now: Instant, cause: &str, unreaped: Option<Child>) {
        e.faults += 1;
        e.unreaped.extend(unreaped);
        let probing = e.breaker.state() == BreakerState::HalfOpen;
        let tripped = if probing { e.breaker.probe_failed(now, cause) } else { e.breaker.record_fault(now, cause) };
        if tripped {
            let p = &self.policy;
            eprintln!(
                "[gpu-supervisor] {device:?} tripped: {} faults within {:?}; probing every {:?}, {} clean probes readmit it",
                p.trip_after, p.window, p.probe_interval, p.probes_to_close
            );
        }
    }

    /// Run one supervised request on `worker` for `device`, at most
    /// `deadline` long. Refused without sending anything while the device is
    /// quarantined. A fault kills and reaps the worker, records the fault
    /// once, and is returned as a definitive [`CallError::Fault`]: the
    /// request is not retried, on this worker or any other.
    pub fn call(&self, device: &K, worker: &mut Worker, request: &str, deadline: Duration) -> Result<String, CallError> {
        if let Some(cause) = self.with(device, |e| (!e.breaker.admits()).then(|| e.breaker.cause().unwrap_or("a probe is running").to_string())) {
            return Err(CallError::Refused { device: format!("{device:?}"), cause });
        }
        // The worker call runs outside the registry lock: a wedged device
        // must never stall another device's admission.
        match worker.request(request, deadline) {
            Ok(Ok(payload)) => Ok(payload),
            Ok(Err(message)) => Err(CallError::Failed(message)),
            Err(WorkerFailure::Invalid(why)) => Err(CallError::Invalid(why)),
            Err(WorkerFailure::Fault(kind)) => Err(CallError::Fault(self.fault(device, worker, request, kind))),
        }
    }

    fn fault(&self, device: &K, worker: &mut Worker, request: &str, kind: FaultKind) -> Fault {
        let unreaped = worker.take_unreaped();
        let fault = Fault { device: format!("{device:?}"), op: verb(request), kind, pid: worker.pid(), reaped: unreaped.is_none() };
        let cause = fault.to_string();
        self.with(device, |e| self.note_fault(device, e, Instant::now(), &cause, unreaped));
        fault
    }

    /// The devices whose probe may start at `now`: quarantined, due, and with
    /// every killed worker reaped (a probe beside a worker still inside the
    /// driver would test nothing).
    pub fn due(&self, now: Instant) -> Vec<K> {
        let mut devices = self.lock();
        devices
            .iter_mut()
            .filter_map(|(k, e)| {
                e.reap();
                (e.unreaped.is_empty() && e.breaker.probe_due(now)).then(|| k.clone())
            })
            .collect()
    }

    /// Claim `device`'s probe (open to half-open). False if none is due or a
    /// killed worker is still unreaped; then the caller does not probe.
    pub fn begin_probe(&self, device: &K, now: Instant) -> bool {
        self.with(device, |e| {
            e.reap();
            e.unreaped.is_empty() && e.breaker.begin_probe(now)
        })
    }

    /// Run the probe [`Self::begin_probe`] claimed: a fresh worker from `cmd`
    /// answers `request` within `deadline`. A pass readmits the device (once
    /// enough have passed); anything else is a fault. The probe's worker is
    /// terminated and reaped before this returns.
    pub fn run_probe(&self, device: &K, cmd: &WorkerCommand, request: &str, deadline: Duration) -> Result<(), String> {
        if self.with(device, |e| e.breaker.state()) != BreakerState::HalfOpen {
            return Err(format!("{device:?}: no health probe was begun"));
        }
        let mut worker = match Worker::spawn_with(cmd, self.reap_within) {
            Ok(w) => w,
            Err(e) => {
                let cause = format!("{device:?}: health probe could not start {} ({e})", cmd.program().display());
                self.with(device, |en| self.note_fault(device, en, Instant::now(), &cause, None));
                return Err(cause);
            }
        };
        let failure = match worker.request(request, deadline) {
            Ok(Ok(_)) => None,
            Ok(Err(message)) => Some(format!("{device:?}: health probe answered an error: {message}")),
            Err(WorkerFailure::Invalid(why)) => Some(format!("{device:?}: health probe request refused: {why}")),
            Err(WorkerFailure::Fault(kind)) => {
                let unreaped = worker.take_unreaped();
                let fault = Fault { device: format!("{device:?}"), op: verb(request), kind, pid: worker.pid(), reaped: unreaped.is_none() };
                let cause = format!("health probe {fault}");
                self.with(device, |e| self.note_fault(device, e, Instant::now(), &cause, unreaped));
                return Err(cause);
            }
        };
        match failure {
            None => self.pass_probe(device),
            Some(cause) => {
                self.with(device, |e| self.note_fault(device, e, Instant::now(), &cause, None));
                Err(cause)
            }
        }
    }

    /// Count the probe [`Self::begin_probe`] claimed as passed without running
    /// one: for a caller with no worker program to probe with, whose breaker
    /// then paces readmission only. Readmits the device once enough have passed.
    pub fn pass_probe(&self, device: &K) -> Result<(), String> {
        let state = self.with(device, |e| {
            if e.breaker.state() != BreakerState::HalfOpen {
                return None;
            }
            e.breaker.probe_passed(Instant::now());
            Some(e.breaker.admits())
        });
        match state {
            None => Err(format!("{device:?}: no health probe was begun")),
            Some(closed) => {
                if closed {
                    eprintln!("[gpu-supervisor] {device:?} passed its health probe; admitting work again");
                }
                Ok(())
            }
        }
    }
}

fn verb(request: &str) -> String {
    request.split_whitespace().next().unwrap_or("").to_string()
}
