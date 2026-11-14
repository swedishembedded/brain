// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that schedulers keep
// their promises through device faults for its clients. If your team needs
// expertise in formal verification of concurrent recovery protocols then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Exhaustive model check of supervised device execution: the executor's lanes
//! and dispatcher, the supervised worker, the per-device circuit breaker, and
//! a resident instance that has to leave a device whose breaker tripped.
//!
//! Each action is one critical section of the real code, and the breaker in
//! the state is the real [`gpu_supervisor::Breaker`] driven by a discrete
//! clock, so its transitions are checked as written, not as re-described. The
//! environment may hang a worker, crash it, keep a killed worker in the kernel
//! past the reap bound, fail a probe, or wedge the teardown of an abandoned
//! instance forever, each within a fault budget.
//!
//! Two kinds of operation run: a plain one, which any device can host, and
//! one that needs the resident instance `R`. `R` starts on device 0, takes the
//! whole of a device's memory, and is built on a free device when it is not
//! resident. When `R`'s device trips while a job needs it and another device
//! is healthy, the dispatcher abandons the instance: it leaves residency at
//! once, its memory stays charged, and its teardown runs on a thread of its
//! own until it finishes (or forever, if it wedges). A faulted `R` job's
//! instance is abandoned the same way.
//!
//! The explorer visits every reachable state for small bounds and checks, in
//! every state:
//!
//! * a device never runs two exclusive operations (two jobs, or a job and a
//!   probe);
//! * no operation is answered twice;
//! * the breaker never admits work while it is open: an operation running on
//!   a device was admitted by a closed breaker that nothing has opened since;
//! * a probe runs only once every worker killed on that device was reaped;
//! * a running `R` job runs on the device `R` is resident on (an instance in
//!   use is never abandoned);
//! * no device holds more than its memory, counting abandoned instances whose
//!   teardown has not finished, and its budget charges exactly what it holds;
//!
//! and in every state with no successor (the system has quiesced):
//!
//! * every operation got exactly one terminal answer, except one that waits
//!   only on time (a device whose next probe is still due) or on memory a
//!   wedged teardown holds;
//! * every killed worker was reaped;
//! * every lane is idle.
//!
//! A deliberate bug in the model must make it fail: see
//! `every_mutation_of_the_protocol_is_caught`.
//!
//! | action | code |
//! |---|---|
//! | `Admit` | `assign` claiming with `exclude = busy ∪ quarantined` (building `R` where it fits); `Health::call`'s admission check |
//! | `Refuse` | `refuse_quarantined` |
//! | `Abandon` | `move_off_quarantined` (`ResidencyManager::abandon_resident`, `drop_elsewhere`) |
//! | `Dropped` / `DropWedges` | the drop thread finishing and its `Msg::Dropped` (`release_abandoned`) / a teardown that never returns |
//! | `Complete` | `Worker::request` answered; `run_group` replies |
//! | `Hang` / `Crash` | the environment: the driver call never returns / the worker dies |
//! | `Kill` | `Worker::request` missing its deadline or reading end of stream, then `Worker::fail` (`SIGKILL` to the group) |
//! | `Reap` / `ReapTimeout` | `Worker::terminate`'s bounded `poll_exit`, then `Health::fault` (fault recorded, unreaped worker parked) and the job's error reply |
//! | `LateReap` | `Entry::reap` (`try_wait` in `Health::due` and `Health::begin_probe`) |
//! | `Release` | the lane's `Msg::Done` / `Msg::Probed` reaching the dispatcher (`busy.remove`; a failed `R` group's `unwind_failed`) |
//! | `StartProbe` | `start_probes` and `Health::begin_probe` |
//! | `ProbePass` / `ProbeFail` | `Health::run_probe` |
//! | `Tick` | the clock |

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use gpu_supervisor::{Breaker, BreakerPolicy, BreakerState};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Queued,
    Running(usize),
    Answered(Answer),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Answer {
    Ok,
    Fault,
    Refused,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Proc {
    Alive,
    Hung,
    Dead,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Lane {
    Idle,
    Running { op: usize, worker: Proc },
    Probing { worker: Proc },
    /// Killed; the bounded reap is in progress. `op` is `None` for a probe.
    Terminating { op: Option<usize> },
    /// Done (or Probed) sent; the dispatcher has not freed the device yet.
    /// `r` says the group ran on `R`, `failed` that it faulted (its claim,
    /// instance included, is unwound when the dispatcher takes the message).
    Releasing { r: bool, failed: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Dispatcher {
    Free,
    /// Inside a teardown on its own thread (only under a mutation).
    Dropping(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    ops: Vec<Op>,
    answers: Vec<u8>,
    lanes: Vec<Lane>,
    breakers: Vec<Breaker>,
    /// Killed workers the kernel has not released yet, per device.
    unreaped: Vec<u8>,
    killed: Vec<u8>,
    reaped: Vec<u8>,
    /// The device `R` is resident on.
    resident: Option<usize>,
    /// Abandoned instances whose teardown is running, per device.
    dropping: Vec<u8>,
    /// Abandoned instances whose teardown wedged for good, per device.
    wedged: Vec<u8>,
    /// What each device's budget charges (instances it holds memory for).
    charged: Vec<u8>,
    dispatcher: Dispatcher,
    faults_left: u8,
    now: u8,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    devices: usize,
    ops: usize,
    /// The first `r_ops` operations need `R`.
    r_ops: usize,
    faults: u8,
    ticks: u8,
}

/// A deliberate bug, to show the model can fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    None,
    /// `assign` ignores the breaker.
    AdmitWhileOpen,
    /// `assign` ignores that the device is busy.
    DoubleAdmit,
    /// A faulted operation is answered and also put back in the queue.
    RetryWedged,
    /// The kill is not followed by any reaping.
    SkipReap,
    /// A probe starts while a killed worker is still unreaped.
    ProbeWithUnreaped,
    /// A fault frees the lane without answering the operation.
    NoAnswerOnFault,
    /// A resident on a tripped device is never moved.
    NeverAbandon,
    /// An abandoned instance's memory is given back before its teardown ends.
    ReleaseOnAbandon,
    /// The dispatcher drops an abandoned instance itself.
    DropOnDispatcher,
}

fn at(now: u8) -> Instant {
    static BASE: OnceLock<Instant> = OnceLock::new();
    *BASE.get_or_init(Instant::now) + Duration::from_secs(u64::from(now))
}

fn policy() -> BreakerPolicy {
    // In clock ticks: two faults within three ticks trip; tripped, probe every
    // tick and close after two clean probes.
    BreakerPolicy { window: Duration::from_secs(3), trip_after: 2, probe_interval: Duration::from_secs(1), probes_to_close: 2 }
}

impl State {
    fn answer(&mut self, op: usize, a: Answer) {
        self.answers[op] += 1;
        self.ops[op] = Op::Answered(a);
    }

    fn quarantined(&self, d: usize) -> bool {
        !self.breakers[d].admits()
    }

    /// Whether `R` is held by a group: a lane runs it or has not reported its end yet.
    fn r_in_use(&self, b: Bounds) -> bool {
        self.lanes.iter().any(|l| match *l {
            Lane::Running { op, .. } | Lane::Terminating { op: Some(op) } => op < b.r_ops,
            Lane::Releasing { r, .. } => r,
            _ => false,
        })
    }

    /// Memory physically held on `d`: the resident instance and every
    /// abandoned one whose teardown has not finished.
    fn held(&self, d: usize) -> u8 {
        u8::from(self.resident == Some(d)) + self.dropping[d] + self.wedged[d]
    }

    /// Abandon `R` from `d`: out of residency, memory still charged, teardown
    /// started on its own thread.
    fn abandon(&mut self, d: usize, m: Mutation) {
        self.resident = None;
        self.dropping[d] += 1;
        if m == Mutation::ReleaseOnAbandon {
            self.charged[d] -= 1;
        }
        if m == Mutation::DropOnDispatcher {
            self.dispatcher = Dispatcher::Dropping(d);
        }
    }
}

fn successors(s: &State, b: Bounds, m: Mutation) -> Vec<(String, State)> {
    let mut out: Vec<(String, State)> = Vec::new();
    let now = at(s.now);
    let dispatcher_free = s.dispatcher == Dispatcher::Free;

    for op in 0..s.ops.len() {
        if s.ops[op] != Op::Queued || !dispatcher_free {
            continue;
        }
        let needs_r = op < b.r_ops;
        for d in 0..b.devices {
            let free = s.lanes[d] == Lane::Idle || (m == Mutation::DoubleAdmit && matches!(s.lanes[d], Lane::Running { .. }));
            let admits = s.breakers[d].admits() || m == Mutation::AdmitWhileOpen;
            // `R` runs where it is resident, or is built on a device with room.
            let hosts = !needs_r || s.resident == Some(d) || (s.resident.is_none() && s.charged[d] == 0);
            if free && admits && hosts && !(needs_r && s.r_in_use(b)) {
                let mut t = s.clone();
                if needs_r && t.resident.is_none() {
                    t.resident = Some(d);
                    t.charged[d] += 1;
                }
                t.ops[op] = Op::Running(d);
                t.lanes[d] = Lane::Running { op, worker: Proc::Alive };
                out.push((format!("Admit(op{op},d{d})"), t));
            }
        }
        // Refused once nothing healthy can ever host it while its breaker is
        // tripped: every device for a plain job or an unplaced `R`; `R`'s own
        // device when `R` is resident and cannot be moved.
        let tripped = |d: usize| s.quarantined(d) && s.breakers[d].tripped();
        let refuse = match (needs_r, s.resident) {
            (true, Some(r)) => tripped(r) && (0..b.devices).all(|e| e == r || s.quarantined(e)),
            _ => (0..b.devices).all(tripped),
        };
        if refuse && !(needs_r && s.r_in_use(b)) {
            let mut t = s.clone();
            t.answer(op, Answer::Refused);
            out.push((format!("Refuse(op{op})"), t));
        }
    }

    // `move_off_quarantined`: `R` resident on a tripped device, a queued job
    // needs it, and another device is healthy.
    if let Some(r) = s.resident {
        let wanted = (0..b.r_ops).any(|op| s.ops[op] == Op::Queued);
        let movable = s.quarantined(r) && s.breakers[r].tripped() && (0..b.devices).any(|e| e != r && !s.quarantined(e));
        if dispatcher_free && wanted && movable && !s.r_in_use(b) && m != Mutation::NeverAbandon {
            let mut t = s.clone();
            t.abandon(r, m);
            out.push((format!("Abandon(d{r})"), t));
        }
    }

    for d in 0..b.devices {
        if s.dropping[d] > 0 {
            let mut t = s.clone();
            t.dropping[d] -= 1;
            if m != Mutation::ReleaseOnAbandon {
                t.charged[d] -= 1;
            }
            if t.dispatcher == Dispatcher::Dropping(d) {
                t.dispatcher = Dispatcher::Free;
            }
            out.push((format!("Dropped(d{d})"), t));
            if s.faults_left > 0 {
                let mut t = s.clone();
                t.faults_left -= 1;
                t.dropping[d] -= 1;
                t.wedged[d] += 1;
                out.push((format!("DropWedges(d{d})"), t));
            }
        }
        match s.lanes[d] {
            Lane::Running { op, worker } => {
                if worker == Proc::Alive {
                    let mut t = s.clone();
                    t.answer(op, Answer::Ok);
                    t.lanes[d] = Lane::Releasing { r: op < b.r_ops, failed: false };
                    out.push((format!("Complete(d{d})"), t));
                }
                if worker == Proc::Alive && s.faults_left > 0 {
                    for (name, w) in [("Hang", Proc::Hung), ("Crash", Proc::Dead)] {
                        let mut t = s.clone();
                        t.faults_left -= 1;
                        t.lanes[d] = Lane::Running { op, worker: w };
                        out.push((format!("{name}(d{d})"), t));
                    }
                }
                if worker != Proc::Alive {
                    let mut t = s.clone();
                    t.killed[d] += 1;
                    if m == Mutation::SkipReap {
                        fault(&mut t, d, Some(op), now, m, b);
                    } else {
                        t.lanes[d] = Lane::Terminating { op: Some(op) };
                    }
                    out.push((format!("Kill(d{d})"), t));
                }
            }
            Lane::Probing { worker } => {
                if worker == Proc::Alive {
                    let mut t = s.clone();
                    t.breakers[d].probe_passed(now);
                    t.lanes[d] = Lane::Releasing { r: false, failed: false };
                    out.push((format!("ProbePass(d{d})"), t));
                }
                if worker == Proc::Alive && s.faults_left > 0 {
                    let mut t = s.clone();
                    t.faults_left -= 1;
                    t.breakers[d].probe_failed(now, "probe answered err");
                    t.lanes[d] = Lane::Releasing { r: false, failed: false };
                    out.push((format!("ProbeFail(d{d})"), t));
                    for (name, w) in [("Hang", Proc::Hung), ("Crash", Proc::Dead)] {
                        let mut t = s.clone();
                        t.faults_left -= 1;
                        t.lanes[d] = Lane::Probing { worker: w };
                        out.push((format!("{name}Probe(d{d})"), t));
                    }
                }
                if worker != Proc::Alive {
                    let mut t = s.clone();
                    t.killed[d] += 1;
                    t.lanes[d] = Lane::Terminating { op: None };
                    out.push((format!("KillProbe(d{d})"), t));
                }
            }
            Lane::Terminating { op } => {
                let mut t = s.clone();
                t.reaped[d] += 1;
                fault(&mut t, d, op, now, m, b);
                out.push((format!("Reap(d{d})"), t));
                if s.faults_left > 0 {
                    let mut t = s.clone();
                    t.faults_left -= 1;
                    t.unreaped[d] += 1;
                    fault(&mut t, d, op, now, m, b);
                    out.push((format!("ReapTimeout(d{d})"), t));
                }
            }
            Lane::Releasing { r, failed } => {
                if dispatcher_free {
                    let mut t = s.clone();
                    t.lanes[d] = Lane::Idle;
                    // A failed `R` group's instance is unwound like an abandoned one.
                    if r && failed && t.resident == Some(d) {
                        t.abandon(d, m);
                    }
                    out.push((format!("Release(d{d})"), t));
                }
            }
            Lane::Idle => {
                let reaped = s.unreaped[d] == 0 || m == Mutation::ProbeWithUnreaped;
                let mut t = s.clone();
                if dispatcher_free && reaped && t.breakers[d].begin_probe(now) {
                    t.lanes[d] = Lane::Probing { worker: Proc::Alive };
                    out.push((format!("StartProbe(d{d})"), t));
                }
            }
        }
        if s.unreaped[d] > 0 {
            let mut t = s.clone();
            t.unreaped[d] -= 1;
            t.reaped[d] += 1;
            out.push((format!("LateReap(d{d})"), t));
        }
    }

    if s.now < b.ticks {
        let mut t = s.clone();
        t.now += 1;
        out.push(("Tick".to_string(), t));
    }
    out
}

/// What `Health::fault` and the failed reply do once the worker is gone: the
/// fault is recorded against the device (a probe's as a failed probe), the
/// operation, if any, gets its one definitive answer, and the lane reports
/// the group's end.
fn fault(t: &mut State, d: usize, op: Option<usize>, now: Instant, m: Mutation, b: Bounds) {
    match op {
        Some(_) => {
            t.breakers[d].record_fault(now, "worker wedged or crashed");
        }
        None => {
            t.breakers[d].probe_failed(now, "probe worker wedged or crashed");
        }
    }
    if let Some(op) = op {
        match m {
            Mutation::NoAnswerOnFault => {}
            Mutation::RetryWedged => {
                t.answer(op, Answer::Fault);
                t.ops[op] = Op::Queued;
            }
            _ => t.answer(op, Answer::Fault),
        }
    }
    t.lanes[d] = Lane::Releasing { r: op.is_some_and(|o| o < b.r_ops), failed: op.is_some() };
}

fn check_invariants(s: &State, b: Bounds) -> Result<(), String> {
    if let Some(op) = s.answers.iter().position(|&n| n > 1) {
        return Err(format!("op{op} answered {} times", s.answers[op]));
    }
    for d in 0..b.devices {
        let running: Vec<usize> = (0..s.ops.len()).filter(|&o| s.ops[o] == Op::Running(d)).collect();
        let probing = matches!(s.lanes[d], Lane::Probing { .. } | Lane::Terminating { op: None });
        if running.len() + usize::from(probing) > 1 {
            return Err(format!("d{d} runs two exclusive operations: ops {running:?}, probing={probing}"));
        }
        for &o in &running {
            let held = matches!(s.lanes[d], Lane::Running { op, .. } | Lane::Terminating { op: Some(op) } if op == o);
            if !held {
                return Err(format!("op{o} is running on d{d} but the lane holds {:?}", s.lanes[d]));
            }
            if s.breakers[d].state() != BreakerState::Closed {
                return Err(format!("op{o} runs on d{d} while its breaker is {:?}", s.breakers[d].state()));
            }
            if o < b.r_ops && s.resident != Some(d) {
                return Err(format!("op{o} runs R on d{d} but R is resident on {:?}: an instance in use was abandoned", s.resident));
            }
        }
        if matches!(s.lanes[d], Lane::Probing { .. }) {
            if s.breakers[d].state() != BreakerState::HalfOpen {
                return Err(format!("d{d} probes with its breaker {:?}", s.breakers[d].state()));
            }
            if s.unreaped[d] > 0 {
                return Err(format!("d{d} probes beside {} unreaped worker(s)", s.unreaped[d]));
            }
        }
        if s.held(d) > 1 {
            return Err(format!("d{d} holds {} instances in memory for one: an abandoned instance was built over", s.held(d)));
        }
        if s.charged[d] != s.held(d) {
            return Err(format!("d{d}'s budget charges {} instance(s) while it holds {}", s.charged[d], s.held(d)));
        }
    }
    Ok(())
}

/// Whether a still-queued `op` waits only on time or on memory a wedged
/// teardown holds: every device that could host it is quarantined with its
/// next probe still in the future, or (for an unplaced `R`) is full with a
/// wedged teardown. `R` resident on such a device waits legitimately only
/// when no other device is healthy (else it should have been moved).
fn waits_legitimately(s: &State, b: Bounds, op: usize) -> bool {
    let now = at(s.now);
    let later = |d: usize| s.quarantined(d) && !s.breakers[d].probe_due(now);
    match (op < b.r_ops, s.resident) {
        (true, Some(r)) => later(r) && (0..b.devices).all(|e| e == r || s.quarantined(e)),
        (true, None) => (0..b.devices).all(|d| later(d) || s.wedged[d] > 0),
        (false, _) => (0..b.devices).all(later),
    }
}

fn check_terminal(s: &State, b: Bounds) -> Result<(), String> {
    for (o, op) in s.ops.iter().enumerate() {
        if *op == Op::Queued && s.answers[o] == 0 && waits_legitimately(s, b, o) {
            continue;
        }
        if !matches!(op, Op::Answered(_)) || s.answers[o] != 1 {
            return Err(format!("op{o} ended {op:?} with {} answers", s.answers[o]));
        }
    }
    for d in 0..b.devices {
        if s.killed[d] != s.reaped[d] {
            return Err(format!("d{d}: {} workers killed, {} reaped", s.killed[d], s.reaped[d]));
        }
        if s.lanes[d] != Lane::Idle {
            return Err(format!("d{d}'s lane ended {:?}", s.lanes[d]));
        }
    }
    Ok(())
}

/// Breadth-first over every reachable state; the number of states, or the
/// shortest trace to the first violation.
fn explore(b: Bounds, m: Mutation) -> Result<usize, String> {
    let resident = (b.r_ops > 0).then_some(0);
    let mut charged = vec![0; b.devices];
    if resident.is_some() {
        charged[0] = 1;
    }
    let init = State {
        ops: vec![Op::Queued; b.ops],
        answers: vec![0; b.ops],
        lanes: vec![Lane::Idle; b.devices],
        breakers: vec![Breaker::new(policy()); b.devices],
        unreaped: vec![0; b.devices],
        killed: vec![0; b.devices],
        reaped: vec![0; b.devices],
        resident,
        dropping: vec![0; b.devices],
        wedged: vec![0; b.devices],
        charged,
        dispatcher: Dispatcher::Free,
        faults_left: b.faults,
        now: 0,
    };
    let mut seen: HashSet<State> = HashSet::new();
    let mut parent: HashMap<State, (State, String)> = HashMap::new();
    let mut frontier = VecDeque::from([init.clone()]);
    seen.insert(init);
    let trace = |parent: &HashMap<State, (State, String)>, mut s: State| {
        let mut steps = Vec::new();
        while let Some((p, a)) = parent.get(&s) {
            steps.push(a.clone());
            s = p.clone();
        }
        steps.reverse();
        steps.join(" -> ")
    };
    while let Some(s) = frontier.pop_front() {
        check_invariants(&s, b).map_err(|e| format!("{e}\n  {b:?} {m:?}\n  trace: {}", trace(&parent, s.clone())))?;
        let next = successors(&s, b, m);
        if next.is_empty() {
            check_terminal(&s, b).map_err(|e| format!("{e}\n  {b:?} {m:?}\n  trace: {}", trace(&parent, s.clone())))?;
        }
        for (action, t) in next {
            if seen.insert(t.clone()) {
                parent.insert(t.clone(), (s.clone(), action));
                frontier.push_back(t);
            }
        }
    }
    Ok(seen.len())
}

fn all_bounds() -> Vec<Bounds> {
    let mut out = Vec::new();
    for devices in 1..=2 {
        for ops in 1..=3 {
            // Three operations with two needing `R` add little the smaller
            // bounds do not already reach, at the cost of most of the run time.
            let most_r = if ops == 3 { 1 } else { ops.min(2) };
            for r_ops in 0..=most_r {
                for faults in 0..=3 {
                    for ticks in 0..=3 {
                        out.push(Bounds { devices, ops, r_ops, faults, ticks });
                    }
                }
            }
        }
    }
    out
}

/// [`explore`] over every bound, the bounds shared out over a few threads
/// (each bound is independent): the total state count, or the first violation.
fn explore_all(m: Mutation) -> Result<usize, String> {
    let bounds = all_bounds();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 8);
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut states = 0;
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&b) = bounds.get(i) else { return Ok(states) };
                        states += explore(b, m)?;
                    }
                })
            })
            .collect();
        workers.into_iter().map(|w| w.join().expect("an explorer thread panicked")).sum()
    })
}

#[test]
fn supervised_execution_keeps_its_promises_through_every_fault() {
    let states = explore_all(Mutation::None).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("{states} states explored");
    assert!(states > 10_000, "the bounds explored too little: {states} states");
}

#[test]
fn every_mutation_of_the_protocol_is_caught() {
    for m in [
        Mutation::AdmitWhileOpen,
        Mutation::DoubleAdmit,
        Mutation::RetryWedged,
        Mutation::SkipReap,
        Mutation::ProbeWithUnreaped,
        Mutation::NoAnswerOnFault,
        Mutation::NeverAbandon,
        Mutation::ReleaseOnAbandon,
        Mutation::DropOnDispatcher,
    ] {
        let why = explore_all(m).err().unwrap_or_else(|| panic!("{m:?} broke the protocol and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
