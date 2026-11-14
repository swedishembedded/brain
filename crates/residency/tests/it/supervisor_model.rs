// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that schedulers keep
// their promises through device faults for its clients. If your team needs
// expertise in formal verification of concurrent recovery protocols then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Exhaustive model check of supervised device execution: the executor's lanes
//! and dispatcher, the supervised worker, and the per-device circuit breaker.
//!
//! Each action is one critical section of the real code, and the breaker in
//! the state is the real [`gpu_supervisor::Breaker`] driven by a discrete
//! clock, so its transitions are checked as written, not as re-described. The
//! environment may hang a worker, crash it, keep a killed worker in the kernel
//! past the reap bound, or fail a probe, each within a fault budget. The
//! explorer visits every reachable state for small bounds and checks, in every
//! state:
//!
//! * a device never runs two exclusive operations (two jobs, or a job and a
//!   probe);
//! * no operation is answered twice;
//! * the breaker never admits work while it is open: an operation running on
//!   a device was admitted by a closed breaker that nothing has opened since;
//! * a probe runs only once every worker killed on that device was reaped;
//!
//! and in every state with no successor (the system has quiesced):
//!
//! * every operation got exactly one terminal answer;
//! * every killed worker was reaped;
//! * every lane is idle.
//!
//! A deliberate bug in the model must make it fail: see
//! `every_mutation_of_the_protocol_is_caught`.
//!
//! | action | code |
//! |---|---|
//! | `Admit` | `assign` claiming with `exclude = busy ∪ quarantined`; `Health::call`'s admission check |
//! | `Refuse` | `refuse_quarantined` |
//! | `Complete` | `Worker::request` answered; `run_group` replies |
//! | `Hang` / `Crash` | the environment: the driver call never returns / the worker dies |
//! | `Kill` | `Worker::request` missing its deadline or reading end of stream, then `Worker::fail` (`SIGKILL` to the group) |
//! | `Reap` / `ReapTimeout` | `Worker::terminate`'s bounded `poll_exit`, then `Health::fault` (fault recorded, unreaped worker parked) and the job's error reply |
//! | `LateReap` | `Entry::reap` (`try_wait` in `Health::due` and `Health::begin_probe`) |
//! | `Release` | the lane's `Msg::Done` / `Msg::Probed` reaching the dispatcher (`busy.remove`) |
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
    Releasing,
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
    faults_left: u8,
    now: u8,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    devices: usize,
    ops: usize,
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
}

fn successors(s: &State, b: Bounds, m: Mutation) -> Vec<(String, State)> {
    let mut out: Vec<(String, State)> = Vec::new();
    let now = at(s.now);

    for op in 0..s.ops.len() {
        if s.ops[op] != Op::Queued {
            continue;
        }
        for d in 0..b.devices {
            let free = s.lanes[d] == Lane::Idle || (m == Mutation::DoubleAdmit && matches!(s.lanes[d], Lane::Running { .. }));
            let admits = s.breakers[d].admits() || m == Mutation::AdmitWhileOpen;
            if free && admits {
                let mut t = s.clone();
                t.ops[op] = Op::Running(d);
                t.lanes[d] = Lane::Running { op, worker: Proc::Alive };
                out.push((format!("Admit(op{op},d{d})"), t));
            }
        }
        // Every operation fits every device, so it is refused exactly when
        // every device is quarantined with a tripped breaker.
        if s.breakers.iter().all(|br| !br.admits() && br.tripped()) {
            let mut t = s.clone();
            t.answer(op, Answer::Refused);
            out.push((format!("Refuse(op{op})"), t));
        }
    }

    for d in 0..b.devices {
        match s.lanes[d] {
            Lane::Running { op, worker } => {
                if worker == Proc::Alive {
                    let mut t = s.clone();
                    t.answer(op, Answer::Ok);
                    t.lanes[d] = Lane::Releasing;
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
                        fault(&mut t, d, Some(op), now, m);
                        t.lanes[d] = Lane::Releasing;
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
                    t.lanes[d] = Lane::Releasing;
                    out.push((format!("ProbePass(d{d})"), t));
                }
                if worker == Proc::Alive && s.faults_left > 0 {
                    let mut t = s.clone();
                    t.faults_left -= 1;
                    t.breakers[d].probe_failed(now, "probe answered err");
                    t.lanes[d] = Lane::Releasing;
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
                fault(&mut t, d, op, now, m);
                t.lanes[d] = Lane::Releasing;
                out.push((format!("Reap(d{d})"), t));
                if s.faults_left > 0 {
                    let mut t = s.clone();
                    t.faults_left -= 1;
                    t.unreaped[d] += 1;
                    fault(&mut t, d, op, now, m);
                    t.lanes[d] = Lane::Releasing;
                    out.push((format!("ReapTimeout(d{d})"), t));
                }
            }
            Lane::Releasing => {
                let mut t = s.clone();
                t.lanes[d] = Lane::Idle;
                out.push((format!("Release(d{d})"), t));
            }
            Lane::Idle => {
                let reaped = s.unreaped[d] == 0 || m == Mutation::ProbeWithUnreaped;
                let mut t = s.clone();
                if reaped && t.breakers[d].begin_probe(now) {
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
/// fault is recorded against the device (a probe's as a failed probe) and the
/// operation, if any, gets its one definitive answer.
fn fault(t: &mut State, d: usize, op: Option<usize>, now: Instant, m: Mutation) {
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
        }
        if matches!(s.lanes[d], Lane::Probing { .. }) {
            if s.breakers[d].state() != BreakerState::HalfOpen {
                return Err(format!("d{d} probes with its breaker {:?}", s.breakers[d].state()));
            }
            if s.unreaped[d] > 0 {
                return Err(format!("d{d} probes beside {} unreaped worker(s)", s.unreaped[d]));
            }
        }
    }
    Ok(())
}

fn check_terminal(s: &State, b: Bounds) -> Result<(), String> {
    for (o, op) in s.ops.iter().enumerate() {
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
    let init = State {
        ops: vec![Op::Queued; b.ops],
        answers: vec![0; b.ops],
        lanes: vec![Lane::Idle; b.devices],
        breakers: vec![Breaker::new(policy()); b.devices],
        unreaped: vec![0; b.devices],
        killed: vec![0; b.devices],
        reaped: vec![0; b.devices],
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
            for faults in 0..=3 {
                for ticks in 0..=3 {
                    out.push(Bounds { devices, ops, faults, ticks });
                }
            }
        }
    }
    out
}

#[test]
fn supervised_execution_keeps_its_promises_through_every_fault() {
    let mut states = 0;
    for b in all_bounds() {
        states += explore(b, Mutation::None).unwrap_or_else(|e| panic!("{e}"));
    }
    assert!(states > 10_000, "the bounds explored too little: {states} states");
}

#[test]
fn every_mutation_of_the_protocol_is_caught() {
    for m in [Mutation::AdmitWhileOpen, Mutation::DoubleAdmit, Mutation::RetryWedged, Mutation::SkipReap, Mutation::ProbeWithUnreaped, Mutation::NoAnswerOnFault] {
        let caught = all_bounds().into_iter().find_map(|b| explore(b, m).err());
        let why = caught.unwrap_or_else(|| panic!("{m:?} broke the protocol and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
