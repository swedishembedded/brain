// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that a server and its
// isolated workers agree on which fine-tuned variants they serve, for its
// clients. If your team needs expertise in formal verification of hot-swap
// protocols then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Exhaustive model check of a hosted base's adapters reaching its worker:
//! the serving process's registry changing at any time, the lane's
//! `AdapterFeed` (`sync_adapters` before a group, `JobsTalk`'s frames while
//! one runs: before the reply to a `take`, and at any poll), the two sockets,
//! the worker's reader (which applies an adapter frame at once, whether its
//! group still runs or ended, so changes apply in the order they were sent),
//! the request loop, and a worker that may die and be replaced.
//!
//! One adapter is offered and withdrawn up to `changes` times. The lane runs
//! `groups` open groups one after another; while one runs, up to `late` jobs
//! arrive in the parent and join it through `take`.
//!
//! In every state:
//!
//! * a group starts in the worker with its registry at least as new as the
//!   parent's was when the lane last brought the worker in line before
//!   sending it;
//! * a job that joins a group reaches the worker after every change the
//!   parent made before handing it out, so the worker's registry holds them
//!   when the group takes it;
//! * the worker's registry is always one the parent offered, never newer
//!   than the parent's;
//!
//! and in every state with no successor, a live worker holds what the parent
//! believes it told it.
//!
//! A deliberate bug in the model must make it fail: see
//! `every_mutation_of_the_adapter_protocol_is_caught`.
//!
//! | action | code |
//! |---|---|
//! | `Change` | the operator offering or withdrawing an adapter (`AdapterRegistry::attach`, a lease dropped); the generation moves |
//! | `Flush` / `Flushed` | `sync_adapters`: each change as an `attach`/`detach` request, answered by the request loop |
//! | `Send` | `group_in_worker` writing the `group` request |
//! | `Arrive` | a job queued for the running group |
//! | `Poll` | `JobsTalk::poll` sending the changes since the worker was last told |
//! | `Take` | `JobsTalk::frame("take")`: the changes, then `joined` |
//! | `Read` | `host::read_requests`: an adapter frame (about a request already read, its group running or not) applied to the sink at once, anything else to the loop |
//! | `Loop` | the request loop: a standalone change applied and answered, a group started, `joined` taken, the group answered and cleared |
//! | `Crash` | the worker dies; the next group starts a new worker and the feed starts over (`AdapterFeed::reset`) |

use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToWorker {
    /// An `attach`/`detach` request between groups: the offered state and its generation.
    Request(bool, u8),
    /// An adapter frame about the running group.
    Frame(bool, u8),
    Group,
    /// The reply to `take`: the job that joined, if one did.
    Joined(Option<usize>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToLoop {
    Request(bool, u8),
    /// A frame that arrived after its group ended (only under `ForwardLateFrames`).
    Late(bool, u8),
    Group,
    Joined(Option<usize>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToParent {
    /// A standalone change answered.
    Applied,
    Take,
    Answer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Lane {
    Idle,
    /// Waiting for the answer to a standalone change.
    Flushing,
    /// Brought the worker in line; the group is not sent yet.
    Flushed,
    InGroup,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Loop {
    Idle,
    Running,
    Joining,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    /// The parent's registry: offered, and its generation.
    offered: bool,
    generation: u8,
    changes_left: u8,
    /// The feed: the generation the worker was brought in line with, and what it was told.
    seen: Option<u8>,
    told: bool,
    /// The generation the lane brought the worker in line with before sending the group.
    group_from: u8,
    lane: Lane,
    groups_left: u8,
    /// Jobs waiting in the parent for the running group, and the generation each was handed out at.
    waiting: VecDeque<usize>,
    handed_at: Vec<Option<u8>>,
    arrived: Vec<bool>,
    to_worker: VecDeque<ToWorker>,
    to_parent: VecDeque<ToParent>,
    queued: VecDeque<ToLoop>,
    /// The reader registered the running group.
    current: bool,
    run: Loop,
    /// The worker's registry and the generation it reflects.
    sink: bool,
    sink_at: u8,
    alive: bool,
    faults_left: u8,
    /// A violation seen when it happened.
    broken: Option<String>,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    changes: u8,
    groups: u8,
    late: usize,
    faults: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    None,
    /// The lane sends a group without bringing the worker in line first.
    NoFlushBeforeGroup,
    /// The changes are sent after the `joined` reply to a take, not before it.
    ChangesAfterJoined,
    /// The reader drops an adapter frame that arrives after its group ended.
    DropLateFrames,
    /// The reader hands an adapter frame that arrives after its group ended to the request loop, which may apply it
    /// after a newer one the reader applied itself.
    ForwardLateFrames,
    /// A new worker's feed is not started over.
    NoResetOnNewWorker,
}

/// The registry state at generation `g`: the adapter is offered after an odd number of changes.
fn offered_at(g: u8) -> bool {
    g % 2 == 1
}

fn apply(t: &mut State, offered: bool, at: u8) {
    if at < t.sink_at {
        t.broken = Some(format!("the worker went back from generation {} to {at}", t.sink_at));
    }
    if at > t.generation || offered_at(at) != offered {
        t.broken = Some(format!("the worker holds a registry the parent never offered (generation {at})"));
    }
    t.sink = offered;
    t.sink_at = at;
}

/// Whether the worker's registry is at least as new as the parent's at generation `g`: a later change reached it,
/// or it holds what the parent offered at `g` (a change back to what the worker held is not sent).
fn in_line(t: &State, g: u8) -> bool {
    t.sink_at > g || t.sink == offered_at(g)
}

/// What the feed would send now, marking it told.
fn changes(t: &mut State) -> Option<ToWorker> {
    if t.seen == Some(t.generation) {
        return None;
    }
    t.seen = Some(t.generation);
    // A change back to what the worker was told is no change at all.
    let differs = t.told != t.offered;
    t.told = t.offered;
    differs.then_some(ToWorker::Frame(t.offered, t.generation))
}

fn lane(s: &State, b: Bounds, m: Mutation, out: &mut Vec<(String, State)>) {
    match s.lane {
        Lane::Idle if s.groups_left > 0 => {
            let mut t = s.clone();
            if !t.alive {
                t.alive = true;
                t.sink = false;
                t.sink_at = 0;
                t.to_worker.clear();
                t.to_parent.clear();
                t.queued.clear();
                t.current = false;
                t.run = Loop::Idle;
                if m != Mutation::NoResetOnNewWorker {
                    t.seen = None;
                    t.told = false;
                }
            }
            t.group_from = t.generation;
            match changes(&mut t) {
                Some(ToWorker::Frame(v, g)) if m != Mutation::NoFlushBeforeGroup => {
                    t.to_worker.push_back(ToWorker::Request(v, g));
                    t.lane = Lane::Flushing;
                }
                _ => t.lane = Lane::Flushed,
            }
            out.push(("Flush".to_string(), t));
        }
        Lane::Flushing => {
            if let Some(ToParent::Applied) = s.to_parent.front() {
                let mut t = s.clone();
                t.to_parent.pop_front();
                t.lane = Lane::Flushed;
                out.push(("Flushed".to_string(), t));
            } else if !s.alive {
                let mut t = s.clone();
                t.lane = Lane::Idle;
                out.push(("FlushFailed".to_string(), t));
            }
        }
        Lane::Flushed => {
            let mut t = s.clone();
            t.to_worker.push_back(ToWorker::Group);
            t.lane = Lane::InGroup;
            t.groups_left -= 1;
            out.push(("Send".to_string(), t));
        }
        Lane::InGroup => {
            for j in 0..b.late {
                if !s.arrived[j] {
                    let mut t = s.clone();
                    t.arrived[j] = true;
                    t.waiting.push_back(j);
                    out.push((format!("Arrive(j{j})"), t));
                }
            }
            if s.alive {
                let mut t = s.clone();
                if let Some(frame) = changes(&mut t) {
                    t.to_worker.push_back(frame);
                    out.push(("Poll".to_string(), t));
                }
            }
            match s.to_parent.front() {
                Some(ToParent::Take) => {
                    let mut t = s.clone();
                    t.to_parent.pop_front();
                    let frame = changes(&mut t);
                    let joined = t.waiting.pop_front();
                    if let Some(j) = joined {
                        t.handed_at[j] = Some(t.generation);
                    }
                    if m != Mutation::ChangesAfterJoined {
                        t.to_worker.extend(frame);
                    }
                    t.to_worker.push_back(ToWorker::Joined(joined));
                    if m == Mutation::ChangesAfterJoined {
                        t.to_worker.extend(frame);
                    }
                    out.push((format!("Take({joined:?})"), t));
                }
                Some(ToParent::Answer) => {
                    let mut t = s.clone();
                    t.to_parent.pop_front();
                    t.lane = Lane::Idle;
                    t.waiting.clear();
                    out.push(("Answered".to_string(), t));
                }
                Some(ToParent::Applied) => {
                    let mut t = s.clone();
                    t.to_parent.pop_front();
                    out.push(("Discard".to_string(), t));
                }
                None if !s.alive => {
                    let mut t = s.clone();
                    t.lane = Lane::Idle;
                    t.waiting.clear();
                    out.push(("Eof".to_string(), t));
                }
                None => {}
            }
        }
        Lane::Idle => {}
    }
}

fn reader(s: &State, m: Mutation, out: &mut Vec<(String, State)>) {
    let Some(&frame) = s.to_worker.front() else { return };
    let mut t = s.clone();
    t.to_worker.pop_front();
    match frame {
        // A frame is about a request the reader already read: applied at once, whether its group still runs or not.
        ToWorker::Frame(v, g) => match m {
            Mutation::ForwardLateFrames if !t.current => t.queued.push_back(ToLoop::Late(v, g)),
            Mutation::DropLateFrames if !t.current => {}
            _ => apply(&mut t, v, g),
        },
        ToWorker::Request(v, g) => t.queued.push_back(ToLoop::Request(v, g)),
        ToWorker::Group => {
            t.current = true;
            t.queued.push_back(ToLoop::Group);
        }
        ToWorker::Joined(j) => t.queued.push_back(ToLoop::Joined(j)),
    }
    out.push((format!("Read({frame:?})"), t));
}

fn request_loop(s: &State, b: Bounds, out: &mut Vec<(String, State)>) {
    match s.run {
        Loop::Idle => {
            let Some(&next) = s.queued.front() else { return };
            let mut t = s.clone();
            t.queued.pop_front();
            match next {
                ToLoop::Request(v, g) => {
                    apply(&mut t, v, g);
                    t.to_parent.push_back(ToParent::Applied);
                }
                ToLoop::Late(v, g) => apply(&mut t, v, g),
                ToLoop::Group => {
                    if !in_line(&t, t.group_from) {
                        t.broken = Some(format!("a group started with the worker at generation {} after the lane brought it to {}", t.sink_at, t.group_from));
                    }
                    t.run = Loop::Running;
                }
                ToLoop::Joined(_) => t.broken = Some("a joined reply outside a group".into()),
            }
            out.push((format!("Loop({next:?})"), t));
        }
        Loop::Running => {
            if (0..b.late).any(|j| s.handed_at[j].is_none()) {
                let mut t = s.clone();
                t.to_parent.push_back(ToParent::Take);
                t.run = Loop::Joining;
                out.push(("AskTake".to_string(), t));
            }
            let mut t = s.clone();
            t.to_parent.push_back(ToParent::Answer);
            t.current = false;
            t.run = Loop::Idle;
            out.push(("Answer".to_string(), t));
        }
        Loop::Joining => {
            if let Some(&ToLoop::Joined(joined)) = s.queued.front() {
                let mut t = s.clone();
                t.queued.pop_front();
                if let Some(j) = joined {
                    let handed = t.handed_at[j].expect("handed out");
                    if !in_line(&t, handed) {
                        t.broken = Some(format!("job {j} joined with the worker at generation {} though it was handed out at {handed}", t.sink_at));
                    }
                }
                t.run = Loop::Running;
                out.push((format!("Joined({joined:?})"), t));
            }
        }
    }
}

fn successors(s: &State, b: Bounds, m: Mutation) -> Vec<(String, State)> {
    let mut out = Vec::new();
    if s.changes_left > 0 {
        let mut t = s.clone();
        t.changes_left -= 1;
        t.offered = !t.offered;
        t.generation += 1;
        out.push(("Change".to_string(), t));
    }
    lane(s, b, m, &mut out);
    if s.alive {
        reader(s, m, &mut out);
        request_loop(s, b, &mut out);
        if s.faults_left > 0 {
            let mut t = s.clone();
            t.faults_left -= 1;
            t.alive = false;
            t.to_worker.clear();
            t.queued.clear();
            t.current = false;
            t.run = Loop::Idle;
            out.push(("Crash".to_string(), t));
        }
    }
    out
}

fn check_terminal(s: &State) -> Result<(), String> {
    if s.alive && s.seen.is_some() && s.sink != s.told {
        return Err(format!("the worker holds {} where its parent believes it told it {}", s.sink, s.told));
    }
    Ok(())
}

fn explore(b: Bounds, m: Mutation) -> Result<usize, String> {
    let init = State {
        offered: false,
        generation: 0,
        changes_left: b.changes,
        seen: None,
        told: false,
        group_from: 0,
        lane: Lane::Idle,
        groups_left: b.groups,
        waiting: VecDeque::new(),
        handed_at: vec![None; b.late],
        arrived: vec![false; b.late],
        to_worker: VecDeque::new(),
        to_parent: VecDeque::new(),
        queued: VecDeque::new(),
        current: false,
        run: Loop::Idle,
        sink: false,
        sink_at: 0,
        alive: true,
        faults_left: b.faults,
        broken: None,
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
        if let Some(why) = &s.broken {
            return Err(format!("{why}\n  {b:?} {m:?}\n  trace: {}", trace(&parent, at)));
        }
        let next = successors(&s, b, m);
        if next.is_empty() {
            check_terminal(&s).map_err(|e| format!("{e}\n  {b:?} {m:?}\n  trace: {}", trace(&parent, at)))?;
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
        for (changes, groups, late) in [(1, 1, 1), (2, 2, 1), (2, 1, 2), (3, 2, 1)] {
            states += explore(Bounds { changes, groups, late, faults }, m)?;
        }
    }
    Ok(states)
}

#[test]
fn a_hosted_bases_adapters_reach_its_worker_before_the_jobs_that_need_them() {
    let states = explore_all(Mutation::None).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("{states} states explored");
    assert!(states > 1_000, "the bounds explored too little: {states} states");
}

#[test]
fn every_mutation_of_the_adapter_protocol_is_caught() {
    for m in [Mutation::NoFlushBeforeGroup, Mutation::ChangesAfterJoined, Mutation::DropLateFrames, Mutation::ForwardLateFrames, Mutation::NoResetOnNewWorker] {
        let why = explore_all(m).err().unwrap_or_else(|| panic!("{m:?} broke the adapter protocol and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
