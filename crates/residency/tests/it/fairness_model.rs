// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that shared
// accelerators keep their promises to inference and training alike for its
// clients. If your team needs expertise in formal verification of admission
// and fairness protocols then you can procure our services by sending an
// email to info@swedishembedded.com.

//! Exhaustive model check of the admission contract (`residency::contract`)
//! as the dispatcher applies it: queue deadlines, the devices held for a
//! workload, late joins of running serving groups, and the memory each
//! device is charged.
//!
//! The decisions are the real [`Contract`] and [`Holds`], driven by a
//! discrete clock, so they are checked as written. Everything around them is
//! left to an adversary: serving requests arrive whenever the queue has room,
//! the scheduler may pick any group the contract lets it claim (so no
//! scheduler weight is relied on), a running group may take a late job
//! whenever the contract allows one, and a serving request may finish at any
//! moment up to its longest run. One workload arrives at any moment.
//!
//! Devices hold one instance each. A serving group claims an idle device and
//! keeps its serving instance there; the workload claims an idle device,
//! evicting an idle serving instance, and gives the device back when it ends.
//!
//! In every state the explorer checks:
//!
//! * holds are conserved: every hold belongs to the workload while it is
//!   queued, and none survives it;
//! * every device is charged exactly the instance it holds, at most its
//!   capacity;
//! * no job is still queued at or past its deadline (every round answers it);
//! * the workload, while queued, has waited no longer than its guarantee:
//!   the contract's `workload_max_wait` plus the longest serving run; so it is
//!   never refused for its deadline, which lies beyond.
//!
//! A deliberate bug in the wiring must make it fail: see
//! `every_mutation_of_the_admission_contract_is_caught`.
//!
//! | action | code |
//! |---|---|
//! | `Serve` | a request submitted (`Msg::Submit`), with its caller's deadline |
//! | `Train` | a workload submitted |
//! | round | `dispatch_loop`: `Dispatcher::enforce_contract` (deadlines, `Holds::reconcile`), then `assign` (`Contract::decide`, the claim, `Holds::release`) |
//! | `Join` | `refill_open_groups` under `Contract::late_joins` |
//! | `Finish` / `TrainDone` | a group's job finishing; `Msg::Done` and `ResidencyManager::release` |
//! | `Tick` | the clock; the dispatcher's timed wake (`Dispatcher::next_wake`) |

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use residency::contract::{Class, Contract, Decision, Holds, Waiting};
use residency::Device;

/// The workload's job id; serving jobs are anonymous in the model.
const TRAIN: u64 = u64::MAX;

#[derive(Clone, Copy, Debug)]
struct Bounds {
    devices: usize,
    /// Serving requests the queue holds before the edge sheds.
    queue: usize,
    /// Requests a serving group runs at once.
    slots: usize,
    /// The longest a serving request runs, in ticks.
    run: u8,
    /// A serving request's own deadline, in ticks after it arrives.
    serve_deadline: u8,
    /// The last tick explored.
    horizon: u8,
}

/// Ticks a workload waits before devices are held for it.
const MAX_WAIT: u8 = 2;
/// The operator's queue deadline, in ticks: past the workload's guarantee for every bound.
const QUEUE_DEADLINE: u8 = 7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    None,
    /// The dispatcher never takes holds.
    NoHolds,
    /// A running group takes late jobs on a held device.
    JoinsIgnoreHolds,
    /// A serving group is claimed on a held device.
    ServingIgnoresHolds,
    /// The dispatcher keeps telling the contract the workload waits after it left.
    StaleWorkload,
    /// Expired jobs are left in the queue.
    NoExpiry,
    /// A finished workload's memory is not given back.
    KeepWorkloadCharge,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Lane {
    Idle,
    /// A serving group: the ticks each running request may still take.
    Serving(Vec<u8>),
    Training,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Resident {
    None,
    Serving,
    Training,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Train {
    NotYet,
    Queued { at: u8 },
    Running,
    Done,
    Refused,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    now: u8,
    lanes: Vec<Lane>,
    resident: Vec<Resident>,
    charged: Vec<u8>,
    /// Arrival ticks of the queued serving requests, oldest first.
    queue: Vec<u8>,
    train: Train,
    /// When the workload was queued, as the dispatcher last told the contract.
    told: Option<u8>,
    holds: Holds,
}

fn at(tick: u8) -> Instant {
    static BASE: OnceLock<Instant> = OnceLock::new();
    *BASE.get_or_init(Instant::now) + Duration::from_secs(u64::from(tick))
}

fn contract() -> Contract {
    Contract { queue_deadline: Duration::from_secs(u64::from(QUEUE_DEADLINE)), workload_max_wait: Duration::from_secs(u64::from(MAX_WAIT)) }
}

fn device(d: usize) -> Device {
    Device::Gpu(d as u32)
}

fn serving_waiting(arrived: u8, b: Bounds) -> Waiting {
    let (deadline, source) = contract().deadline(at(arrived), Some(at(arrived + b.serve_deadline)));
    Waiting { id: u64::from(arrived), class: Class::Serving, enqueued: at(arrived), deadline, source }
}

fn train_waiting(queued: u8) -> Waiting {
    let (deadline, source) = contract().deadline(at(queued), None);
    Waiting { id: TRAIN, class: Class::Workload, enqueued: at(queued), deadline, source }
}

impl State {
    /// The dispatcher's round: expire, reconcile holds, then assign until no
    /// idle device can take anything the contract allows. Assignment is the
    /// adversary's choice, so a round has several outcomes.
    fn round(mut self, b: Bounds, m: Mutation) -> Vec<State> {
        let c = contract();
        let now = at(self.now);
        if m != Mutation::NoExpiry {
            self.queue.retain(|&t| !matches!(c.decide(&serving_waiting(t, b), now, &self.holds), Decision::Refuse(_)));
            if let Train::Queued { at: q } = self.train {
                if matches!(c.decide(&train_waiting(q), now, &self.holds), Decision::Refuse(_)) {
                    self.train = Train::Refused;
                }
            }
        }
        self.tell(m);
        if m != Mutation::NoHolds {
            let devices: Vec<Device> = (0..b.devices).map(device).collect();
            let waiting: Vec<Waiting> = self.told.map(train_waiting).into_iter().chain(self.queue.iter().map(|&t| serving_waiting(t, b))).collect();
            self.holds.reconcile(&c, &waiting, now, |_| devices.clone());
        }
        let mut out = Vec::new();
        self.assign(b, m, &mut out);
        out
    }

    /// What the dispatcher tells the contract about the workload.
    fn tell(&mut self, m: Mutation) {
        match self.train {
            Train::Queued { at: q } => self.told = Some(q),
            _ if m == Mutation::StaleWorkload => {}
            _ => self.told = None,
        }
    }

    fn assign(self, b: Bounds, m: Mutation, out: &mut Vec<State>) {
        let c = contract();
        let now = at(self.now);
        let mut choices: Vec<State> = Vec::new();
        for d in (0..b.devices).filter(|&d| self.lanes[d] == Lane::Idle) {
            if let Train::Queued { at: q } = self.train {
                if c.decide(&train_waiting(q), now, &self.holds) == Decision::Admit {
                    let mut t = self.clone();
                    // An idle serving instance is evicted; the workload is charged in its place.
                    t.lanes[d] = Lane::Training;
                    t.resident[d] = Resident::Training;
                    t.charged[d] = 1;
                    t.train = Train::Running;
                    t.holds.release(TRAIN);
                    t.tell(m);
                    choices.push(t);
                }
            }
            if let Some(&oldest) = self.queue.first() {
                let allowed = match c.decide(&serving_waiting(oldest, b), now, &self.holds) {
                    Decision::Admit => true,
                    Decision::Elsewhere { avoid } => m == Mutation::ServingIgnoresHolds || !avoid.contains(&device(d)),
                    _ => false,
                };
                if allowed {
                    let mut t = self.clone();
                    let n = b.slots.min(t.queue.len());
                    t.queue.drain(..n);
                    t.lanes[d] = Lane::Serving(vec![b.run; n]);
                    t.resident[d] = Resident::Serving;
                    t.charged[d] = 1;
                    choices.push(t);
                }
            }
        }
        if choices.is_empty() {
            out.push(self);
            return;
        }
        for t in choices {
            t.assign(b, m, out);
        }
    }

    fn train_queued(&self) -> bool {
        matches!(self.train, Train::Queued { .. })
    }
}

fn successors(s: &State, b: Bounds, m: Mutation) -> Vec<(String, State)> {
    let mut events: Vec<(String, State)> = Vec::new();
    if s.queue.len() < b.queue {
        let mut t = s.clone();
        t.queue.push(s.now);
        events.push(("Serve".into(), t));
    }
    if s.train == Train::NotYet {
        let mut t = s.clone();
        t.train = Train::Queued { at: s.now };
        events.push(("Train".into(), t));
    }
    for d in 0..b.devices {
        match &s.lanes[d] {
            Lane::Serving(live) => {
                let joins = m == Mutation::JoinsIgnoreHolds || contract().late_joins(&[device(d)], &s.holds) == Decision::Admit;
                if joins && live.len() < b.slots && !s.queue.is_empty() {
                    let mut t = s.clone();
                    t.queue.remove(0);
                    if let Lane::Serving(live) = &mut t.lanes[d] {
                        live.push(b.run);
                    }
                    events.push((format!("Join({d})"), t));
                }
                for i in 0..live.len() {
                    let mut t = s.clone();
                    if let Lane::Serving(live) = &mut t.lanes[d] {
                        live.remove(i);
                        if live.is_empty() {
                            t.lanes[d] = Lane::Idle;
                        }
                    }
                    events.push((format!("Finish({d})"), t));
                }
            }
            Lane::Training => {
                let mut t = s.clone();
                t.lanes[d] = Lane::Idle;
                t.resident[d] = Resident::None;
                if m != Mutation::KeepWorkloadCharge {
                    t.charged[d] = 0;
                }
                t.train = Train::Done;
                events.push((format!("TrainDone({d})"), t));
            }
            Lane::Idle => {}
        }
    }
    // Time passes only once every run that is out of time has finished.
    let due = s.lanes.iter().any(|l| matches!(l, Lane::Serving(live) if live.contains(&0)));
    if s.now < b.horizon && !due {
        let mut t = s.clone();
        t.now += 1;
        for l in t.lanes.iter_mut() {
            if let Lane::Serving(live) = l {
                for r in live.iter_mut() {
                    *r -= 1;
                }
            }
        }
        events.push(("Tick".into(), t));
    }
    events.into_iter().flat_map(|(name, t)| t.round(b, m).into_iter().map(move |r| (name.clone(), r))).collect()
}

fn check_invariants(s: &State, b: Bounds) -> Result<(), String> {
    let held: Vec<u64> = s.holds.jobs().collect();
    if held.iter().any(|&j| j != TRAIN) || (!held.is_empty() && !s.train_queued()) {
        return Err(format!("a hold outlives its workload: holds {held:?}, workload {:?}", s.train));
    }
    for d in 0..b.devices {
        let holds = u8::from(s.resident[d] != Resident::None);
        if s.charged[d] != holds || s.charged[d] > 1 {
            return Err(format!("device {d} is charged {} for {:?}", s.charged[d], s.resident[d]));
        }
    }
    let now = at(s.now);
    if let Some(&t) = s.queue.iter().find(|&&t| serving_waiting(t, b).deadline <= now) {
        return Err(format!("a request queued at tick {t} is still queued past its deadline"));
    }
    match s.train {
        Train::Queued { at: q } if train_waiting(q).deadline <= now => return Err("the workload is still queued past its deadline".into()),
        Train::Queued { at: q } if s.now - q > MAX_WAIT + b.run => {
            return Err(format!("the workload queued at tick {q} still waits at tick {}: past its guarantee of {} ticks", s.now, MAX_WAIT + b.run));
        }
        Train::Refused => return Err("the workload was refused for its deadline, past its guarantee".into()),
        _ => {}
    }
    Ok(())
}

/// Breadth-first over every reachable state; the number of states, or the
/// shortest trace to the first violation.
fn explore(b: Bounds, m: Mutation) -> Result<usize, String> {
    let init = State { now: 0, lanes: vec![Lane::Idle; b.devices], resident: vec![Resident::None; b.devices], charged: vec![0; b.devices], queue: Vec::new(), train: Train::NotYet, told: None, holds: Holds::default() };
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
        check_invariants(&s, b).map_err(|e| format!("{e}\n  {b:?} {m:?}\n  state: {s:?}\n  trace: {}", trace(&parent, s.clone())))?;
        for (action, t) in successors(&s, b, m) {
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
        for queue in 1..=3 {
            for slots in 1..=2 {
                for run in 1..=2 {
                    for serve_deadline in [2, 4] {
                        out.push(Bounds { devices, queue, slots, run, serve_deadline, horizon: QUEUE_DEADLINE + 1 });
                    }
                }
            }
        }
    }
    out
}

/// [`explore`] over every bound, shared out over a few threads.
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
fn the_admission_contract_keeps_its_promises_under_any_arrivals() {
    let states = explore_all(Mutation::None).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("{states} states explored");
    assert!(states > 10_000, "the bounds explored too little: {states} states");
}

#[test]
fn every_mutation_of_the_admission_contract_is_caught() {
    for m in [Mutation::NoHolds, Mutation::JoinsIgnoreHolds, Mutation::ServingIgnoresHolds, Mutation::StaleWorkload, Mutation::NoExpiry, Mutation::KeepWorkloadCharge] {
        let why = explore_all(m).err().unwrap_or_else(|| panic!("{m:?} broke the contract and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
