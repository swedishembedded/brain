// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that model servers
// account for every byte their isolated workers hold on an accelerator, for
// its clients. If your team needs expertise in formal verification of memory
// residency across process boundaries then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Exhaustive model check of a hosted instance moving between tiers: the
//! residency manager's charges (`evict_entry`'s demotion, `claim`'s promotion
//! of a Warm instance), the lane's promotion (`SupervisedInstance::promote`)
//! and the worker that holds the instance (`host`'s `open`, `demote`,
//! `promote` and `batch`), with the instance's metrics riding on each answer
//! and a worker that may die in any request.
//!
//! Two cards hold three units each. The hosted instance is three units Hot:
//! two of buffers and one of its worker's device context, which a demotion
//! cannot free while the worker lives. Another model of two units may be
//! built on any card with room, and stays. The instance serves `jobs` jobs;
//! between them the dispatcher may demote it (an eviction) up to `demotes`
//! times, and a job for a Warm instance claims a card for its promotion, the
//! card its worker holds when the instance fits back there, else another one.
//!
//! In every state:
//!
//! * no card holds more than the manager charges it, but for the context of a
//!   worker the lane is about to kill (its promotion onto another card was
//!   claimed and not started: the charge moved with the claim);
//! * a job runs only on a Hot instance in a worker on the card the manager
//!   placed it on;
//! * between requests the parent's copy of the instance's metrics is the
//!   worker's;
//!
//! and in every state with no successor: every job was answered once, every
//! card holds exactly what it is charged, and no worker outlives the
//! instance the manager tracks.
//!
//! A deliberate bug in the model must make it fail: see
//! `every_mutation_of_the_tier_protocol_is_caught`.
//!
//! | action | code |
//! |---|---|
//! | `Build` / `Opened` / `OpenFails` | `claim` charging a cold build; `WorkerStart::open_on` (`hello`, `open`, metrics from the answer); a fault in either |
//! | `Run` / `RunCrash` | `batch` and its answer's metrics; a worker that dies in it (the group fails, the claim is unwound) |
//! | `Demote` / `DemoteRefused` / `DemoteCrash` | `evict_entry`: `demote warm` answered (the Warm entry stays on its card charged the kept context), refused by the model (a full drop) or faulted (a full drop) |
//! | `Other` | another model's claim on a card with room |
//! | `ClaimPromote` | `claim` of a Warm entry: its card first when the instance fits there (`kept_home`), else `pick_device` |
//! | `LanePromote` / `PromoteCrash` | `SupervisedInstance::promote`: `promote <device>` in the worker on its own card, else the old worker killed and a new one opened |

use std::collections::HashMap;

const CARDS: usize = 2;
const CAP: u8 = 3;
const HOT: u8 = 3;
const CONTEXT: u8 = 1;
const OTHER: u8 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Tier {
    Hot,
    Warm,
}

/// A live worker: its card, the tier of the instance in it, and the jobs that instance ran (its metric).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Worker {
    card: usize,
    tier: Tier,
    runs: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Lane {
    Idle,
    /// A cold build or a promotion onto another card: a new worker opening on the card.
    Opening(usize),
    /// A promotion was claimed onto the card; the lane has not taken it yet.
    Promoting(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    worker: Option<Worker>,
    /// A worker the lane has not killed yet while the next one opens (only under `OpenBeforeKill`).
    stale: Option<Worker>,
    /// The manager's residency entry for the instance: its card and tier.
    entry: Option<(usize, Tier)>,
    charged: [u8; CARDS],
    other: Option<usize>,
    lane: Lane,
    /// The parent's copy of the instance's metrics.
    cache: u8,
    jobs_left: u8,
    answered: u8,
    demotes_left: u8,
    faults_left: u8,
    /// A job ran somewhere other than a Hot instance on the card the manager placed it on.
    ran_wrong: bool,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    jobs: u8,
    demotes: u8,
    faults: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    None,
    /// A demoted instance is charged to the host, as an in-process one is: its worker's context goes uncharged.
    ReleaseContextOnDemote,
    /// A claim of a Warm instance hands it over Hot without the lane promoting it.
    RunWhileWarm,
    /// A promotion onto another card is sent to the old worker, which promotes on its own card.
    PromoteInPlace,
    /// The new worker of a promotion onto another card opens before the old one is killed.
    OpenBeforeKill,
    /// The metrics the answer to a new worker's `open` carries are not kept.
    StaleMetrics,
}

fn held(s: &State) -> [u8; CARDS] {
    let mut held = [0; CARDS];
    for w in s.worker.iter().chain(s.stale.iter()) {
        held[w.card] += CONTEXT + if w.tier == Tier::Hot { HOT - CONTEXT } else { 0 };
    }
    if let Lane::Opening(card) = s.lane {
        held[card] += HOT;
    }
    if let Some(card) = s.other {
        held[card] += OTHER;
    }
    held
}

fn fits(s: &State, card: usize, need: u8) -> bool {
    s.charged[card] + need <= CAP
}

/// The instance leaves residency: the manager releases its charge and drops it, which kills its worker.
fn drop_instance(t: &mut State) {
    if let Some((card, tier)) = t.entry.take() {
        t.charged[card] -= if tier == Tier::Hot { HOT } else { CONTEXT };
    }
    t.worker = None;
}

/// The environment spends one fault of its budget: a worker dies or wedges and is killed.
fn fault(t: &mut State) {
    t.faults_left -= 1;
}

fn successors(s: &State, m: Mutation) -> Vec<(String, State)> {
    let mut out = Vec::new();
    if s.other.is_none() {
        for card in 0..CARDS {
            if fits(s, card, OTHER) {
                let mut t = s.clone();
                t.charged[card] += OTHER;
                t.other = Some(card);
                out.push((format!("Other(gpu{card})"), t));
            }
        }
    }
    match s.lane {
        Lane::Idle => idle(s, m, &mut out),
        Lane::Opening(card) => {
            let mut t = s.clone();
            t.lane = Lane::Idle;
            t.stale = None;
            t.worker = Some(Worker { card, tier: Tier::Hot, runs: 0 });
            if m != Mutation::StaleMetrics {
                t.cache = 0;
            }
            out.push((format!("Opened(gpu{card})"), t));
            if s.faults_left > 0 {
                // The job that needed it fails, and the claim is unwound.
                let mut t = s.clone();
                t.lane = Lane::Idle;
                t.stale = None;
                fault(&mut t);
                drop_instance(&mut t);
                t.jobs_left -= 1;
                t.answered += 1;
                out.push((format!("OpenFails(gpu{card})"), t));
            }
        }
        Lane::Promoting(to) => {
            let w = s.worker.expect("a Warm instance has its worker");
            if w.card == to || m == Mutation::PromoteInPlace {
                let mut t = s.clone();
                t.lane = Lane::Idle;
                t.worker = Some(Worker { tier: Tier::Hot, ..w });
                t.cache = w.runs;
                out.push((format!("LanePromote(gpu{to}, in place)"), t));
                if s.faults_left > 0 {
                    let mut t = s.clone();
                    t.lane = Lane::Idle;
                    fault(&mut t);
                    drop_instance(&mut t);
                    t.jobs_left -= 1;
                    t.answered += 1;
                    out.push((format!("PromoteCrash(gpu{to})"), t));
                }
            } else {
                let mut t = s.clone();
                if m == Mutation::OpenBeforeKill {
                    t.stale = t.worker;
                }
                t.worker = None;
                t.lane = Lane::Opening(to);
                out.push((format!("LanePromote(gpu{} -> gpu{to}, new worker)", w.card), t));
            }
        }
    }
    out
}

fn idle(s: &State, m: Mutation, out: &mut Vec<(String, State)>) {
    let Some((card, tier)) = s.entry else {
        if s.jobs_left > 0 {
            for card in 0..CARDS {
                if fits(s, card, HOT) {
                    let mut t = s.clone();
                    t.charged[card] += HOT;
                    t.entry = Some((card, Tier::Hot));
                    t.lane = Lane::Opening(card);
                    out.push((format!("Build(gpu{card})"), t));
                }
            }
        }
        return;
    };
    let worker = s.worker.expect("a resident instance has its worker");
    match tier {
        Tier::Hot if s.jobs_left > 0 => {
            let mut t = s.clone();
            t.jobs_left -= 1;
            t.answered += 1;
            t.ran_wrong |= worker.tier != Tier::Hot || worker.card != card;
            let runs = worker.runs + 1;
            t.worker = Some(Worker { runs, ..worker });
            t.cache = runs;
            out.push(("Run".to_string(), t));
            if s.faults_left > 0 {
                let mut t = s.clone();
                t.jobs_left -= 1;
                t.answered += 1;
                fault(&mut t);
                drop_instance(&mut t);
                out.push(("RunCrash".to_string(), t));
            }
        }
        _ => {}
    }
    if tier == Tier::Hot && s.demotes_left > 0 {
        // The demotion keeps the worker's context on its card (`home`), unless the bug charges it to the host.
        let mut t = s.clone();
        t.demotes_left -= 1;
        t.worker = Some(Worker { tier: Tier::Warm, ..worker });
        t.cache = worker.runs;
        t.charged[card] -= HOT;
        if m != Mutation::ReleaseContextOnDemote {
            t.charged[card] += CONTEXT;
        }
        t.entry = Some((card, Tier::Warm));
        out.push(("Demote".to_string(), t));
        let mut t = s.clone();
        t.demotes_left -= 1;
        drop_instance(&mut t);
        out.push(("DemoteRefused".to_string(), t));
        if s.faults_left > 0 {
            let mut t = s.clone();
            t.demotes_left -= 1;
            fault(&mut t);
            drop_instance(&mut t);
            out.push(("DemoteCrash".to_string(), t));
        }
    }
    if tier == Tier::Warm && s.jobs_left > 0 {
        let kept = if m == Mutation::ReleaseContextOnDemote { 0 } else { CONTEXT };
        let home = s.charged[card] - kept + HOT <= CAP;
        let targets: Vec<usize> = if home { vec![card] } else { (0..CARDS).filter(|&d| d != card && fits(s, d, HOT)).collect() };
        for to in targets {
            let mut t = s.clone();
            t.charged[card] -= kept;
            t.charged[to] += HOT;
            t.entry = Some((to, Tier::Hot));
            if m == Mutation::RunWhileWarm {
                t.lane = Lane::Idle;
            } else {
                t.lane = Lane::Promoting(to);
            }
            out.push((format!("ClaimPromote(gpu{card} -> gpu{to})"), t));
        }
    }
}

fn check(s: &State) -> Result<(), String> {
    for (card, (&held, &charged)) in held(s).iter().zip(&s.charged).enumerate() {
        // The old worker of a promotion onto another card, until the lane kills it.
        let exempt = matches!(s.lane, Lane::Promoting(to) if to != card) && s.worker.is_some_and(|w| w.card == card && w.tier == Tier::Warm);
        if held > charged && !(exempt && held == charged + CONTEXT) {
            return Err(format!("gpu{card} holds {held} units and is charged {charged}"));
        }
        if held > CAP {
            return Err(format!("gpu{card} holds {held} units of {CAP}"));
        }
    }
    if s.ran_wrong {
        return Err("a job ran on a demoted instance or on another card than the one it was placed on".into());
    }
    if let (Lane::Idle, Some(w)) = (s.lane, s.worker) {
        if s.cache != w.runs {
            return Err(format!("the parent reports {} runs and the worker's instance ran {}", s.cache, w.runs));
        }
    }
    Ok(())
}

fn check_terminal(s: &State, b: Bounds) -> Result<(), String> {
    if s.answered != b.jobs {
        return Err(format!("{} of {} jobs answered", s.answered, b.jobs));
    }
    if held(s) != s.charged {
        return Err(format!("the cards hold {:?} and are charged {:?}", held(s), s.charged));
    }
    if s.entry.is_none() && s.worker.is_some() {
        return Err("a worker outlived the instance the manager tracks".into());
    }
    Ok(())
}

fn explore(b: Bounds, m: Mutation) -> Result<usize, String> {
    let init = State {
        worker: None,
        stale: None,
        entry: None,
        charged: [0; CARDS],
        other: None,
        lane: Lane::Idle,
        cache: 0,
        jobs_left: b.jobs,
        answered: 0,
        demotes_left: b.demotes,
        faults_left: b.faults,
        ran_wrong: false,
    };
    let mut index: HashMap<State, usize> = HashMap::from([(init.clone(), 0)]);
    let mut states = vec![init];
    let mut parent: Vec<Option<(usize, String)>> = vec![None];
    let trace = |parent: &[Option<(usize, String)>], mut i: usize| {
        let mut steps = Vec::new();
        while let Some((p, a)) = &parent[i] {
            steps.push(a.clone());
            i = *p;
        }
        steps.reverse();
        steps.join(" -> ")
    };
    let mut at = 0;
    while at < states.len() {
        let s = states[at].clone();
        check(&s).map_err(|e| format!("{e}\n  {b:?} {m:?}\n  trace: {}", trace(&parent, at)))?;
        let next = successors(&s, m);
        if next.is_empty() {
            check_terminal(&s, b).map_err(|e| format!("{e}\n  {b:?} {m:?}\n  trace: {}", trace(&parent, at)))?;
        }
        for (action, t) in next {
            if !index.contains_key(&t) {
                index.insert(t.clone(), states.len());
                states.push(t);
                parent.push(Some((at, action)));
            }
        }
        at += 1;
    }
    Ok(states.len())
}

fn explore_all(m: Mutation) -> Result<usize, String> {
    let mut states = 0;
    for faults in 0..=1 {
        for (jobs, demotes) in [(1, 1), (2, 1), (2, 2), (3, 2)] {
            states += explore(Bounds { jobs, demotes, faults }, m)?;
        }
    }
    Ok(states)
}

#[test]
fn a_hosted_instance_keeps_its_charges_and_metrics_across_tiers() {
    let states = explore_all(Mutation::None).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("{states} states explored");
    assert!(states > 200, "the bounds explored too little: {states} states");
}

#[test]
fn every_mutation_of_the_tier_protocol_is_caught() {
    let mutations = [Mutation::ReleaseContextOnDemote, Mutation::RunWhileWarm, Mutation::PromoteInPlace, Mutation::OpenBeforeKill, Mutation::StaleMetrics];
    for m in mutations {
        let why = explore_all(m).err().unwrap_or_else(|| panic!("{m:?} broke the tier protocol and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
