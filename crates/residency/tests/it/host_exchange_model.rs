// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that request
// protocols between a server and its worker processes keep their promises,
// for its clients. If your team needs expertise in formal verification of
// cancellation and crash recovery then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Exhaustive model check of one hosted model's jobs crossing the worker
//! boundary: the lane's exchange (`SupervisedInstance::run_in_worker`,
//! `Worker::exchange`), the two sockets' frame queues, and the worker's
//! reader thread and request loop (`supervised::host`), with progress
//! frames, cancellation and a worker that may die at any point.
//!
//! Jobs run one after another on one lane, as `run_batch` runs them. The
//! caller of each job may ask to cancel it at any time; the worker may be
//! killed, or wedge (its request loop stops inside the driver while its
//! reader thread runs on), at any time within a fault budget, after which the
//! next job starts a fresh worker on a fresh socket.
//!
//! Time is ticks of the lane's deadline, which every frame the lane reads
//! about the job in flight renews (`Exchange::renew`): a job may run far
//! longer than one deadline as long as it keeps reporting. The environment
//! is fair to a healthy worker: it reports within one deadline, so time
//! stops one tick short of the deadline until its next frame is read. A
//! wedged worker reports nothing, and time runs out.
//!
//! In every state:
//!
//! * no job is answered twice;
//! * a job is answered `cancelled` only if its own caller asked;
//! * a job answered `cancelled` did not also run to completion in the worker
//!   (a caller told "cancelled" must be able to trust nothing was done);
//! * a cancellation that reached the worker while its job was waiting to
//!   start or running is never dropped;
//! * a deadline kills only a worker that went silent (wedged), never one
//!   that kept reporting;
//!
//! in every state with no successor:
//!
//! * every job got exactly one answer;
//! * no frame is left unread on a live socket;
//!
//! and from every state a state with no successor is reachable: no cycle
//! keeps a job from its answer (a deadline that never fires, say).
//!
//! A deliberate bug in the model must make it fail: see
//! `every_mutation_of_the_exchange_is_caught`.
//!
//! | action | code |
//! |---|---|
//! | `Send` | `run_in_worker` writing `run <action> <envelope>` (`wire::send`); a dead worker's successor is spawned by the next activation |
//! | `CallerCancel` | the caller firing the job's `CancelToken` |
//! | `SendCancel` | `Worker::exchange` seeing `cancelled()` and writing `<id> cancel` once |
//! | `Receive` | `Worker::exchange` reading a frame: progress for the job in flight is forwarded and renews the deadline, its answer ends the exchange |
//! | `SkipCancelled` | `Instance::run_batch` answering a job cancelled before its turn without running it |
//! | `Eof` | `Worker::exchange` reading end of stream: the job's fault answer |
//! | `Tick` | time passing while `Worker::exchange` waits on the socket |
//! | `Timeout` | `Worker::exchange` finding its deadline passed: the worker is killed, the job's fault answer |
//! | `Wedge` | the environment: the worker's request loop stops inside the driver |
//! | `ReaderStep` | `host::read_requests`: a run registers its token as `Current` and is queued; a cancel fires the token of the run it names |
//! | `Start` | `Host::run` taking the queued run |
//! | `Progress` | the model's progress callback writing `<id> progress` |
//! | `Finish` | `Host::run` answering, then `Host::handle` clearing `Current` if it still names this run |
//! | `Crash` | the environment: the worker dies |

use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Answer {
    Ok,
    Cancelled,
    Fault,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToWorker {
    Run(usize),
    Cancel(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToParent {
    Progress(usize),
    Answer(usize, Answer),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Exec {
    Idle,
    /// Running job `j`; `progressed` once it sent its one progress frame.
    Running { job: usize, progressed: bool },
    /// Answered `j`, has not cleared `Current` yet.
    Clearing(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    answers: Vec<u8>,
    answered: Vec<Option<Answer>>,
    /// The job the lane is exchanging, if any.
    in_flight: Option<usize>,
    /// The next job to send.
    next: usize,
    asked: Vec<bool>,
    cancel_sent: Vec<bool>,
    to_worker: VecDeque<ToWorker>,
    to_parent: VecDeque<ToParent>,
    /// The reader has queued these runs for the request loop.
    queued: VecDeque<usize>,
    /// `Current`: the run read last and not finished, and its token.
    current: Option<(usize, bool)>,
    /// Tokens of runs the request loop holds (queued or running), fired or not.
    fired: Vec<bool>,
    exec: Exec,
    /// `None` when the worker is dead (its socket is closed).
    alive: bool,
    /// What the worker produced for each job, if it ran to its end.
    produced: Vec<Option<Answer>>,
    /// A cancellation the reader read for a run it held and did not deliver.
    dropped_cancel: bool,
    /// Ticks since the lane last read a frame about the job in flight (or sent it).
    since_heard: u8,
    /// Ticks the lane's deadline has counted; equal to `since_heard` when
    /// every frame renews it.
    timer: u8,
    /// The request loop is stuck inside the driver; the reader still runs.
    wedged: bool,
    /// A deadline killed a worker that was not wedged.
    killed_healthy: bool,
    faults_left: u8,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    jobs: usize,
    /// Crashes and wedges together.
    faults: u8,
    /// Ticks in one deadline.
    deadline: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    None,
    /// The token is registered when the run starts, not when it is read.
    LateRegistration,
    /// `Current` is cleared unconditionally, after the answer was sent.
    ClearAnyAfterAnswer,
    /// The lane answers `cancelled` as soon as it sends the cancel frame.
    AnswerAtCancel,
    /// End of stream frees the lane without answering the job.
    NoAnswerOnEof,
    /// The deadline counts from the request, as before progress renewed it.
    AbsoluteDeadline,
    /// The deadline starts over at every wake-up of the wait (the poll for a
    /// cancellation), not only at a frame from the worker.
    RenewOnPoll,
}

fn answer(t: &mut State, job: usize, a: Answer) {
    t.answers[job] += 1;
    t.answered[job] = Some(a);
}

fn successors(s: &State, b: Bounds, m: Mutation) -> Vec<(String, State)> {
    let mut out = Vec::new();

    // The lane.
    if s.in_flight.is_none() && s.next < b.jobs {
        let mut t = s.clone();
        let j = t.next;
        t.next += 1;
        if t.asked[j] {
            // `run_batch` answers a job cancelled before its turn without running it.
            answer(&mut t, j, Answer::Cancelled);
            out.push((format!("SkipCancelled(j{j})"), t));
        } else {
            if !t.alive {
                // A fresh worker on a fresh socket.
                t.alive = true;
                t.wedged = false;
                t.to_worker.clear();
                t.to_parent.clear();
                t.queued.clear();
                t.current = None;
                t.exec = Exec::Idle;
            }
            t.to_worker.push_back(ToWorker::Run(j));
            t.in_flight = Some(j);
            t.since_heard = 0;
            t.timer = 0;
            out.push((format!("Send(j{j})"), t));
        }
    }
    for j in 0..b.jobs {
        if !s.asked[j] && s.answers[j] == 0 {
            let mut t = s.clone();
            t.asked[j] = true;
            out.push((format!("CallerCancel(j{j})"), t));
        }
    }
    if let Some(j) = s.in_flight {
        if s.asked[j] && !s.cancel_sent[j] && s.alive {
            let mut t = s.clone();
            t.cancel_sent[j] = true;
            t.to_worker.push_back(ToWorker::Cancel(j));
            if m == Mutation::AnswerAtCancel {
                answer(&mut t, j, Answer::Cancelled);
                t.in_flight = None;
            }
            out.push((format!("SendCancel(j{j})"), t));
        }
        if let Some(&frame) = s.to_parent.front() {
            let mut t = s.clone();
            t.to_parent.pop_front();
            match frame {
                // Forwarded to the job in flight: frames are read in order
                // and a job's answer is its last frame, so it is that job's.
                ToParent::Progress(i) if i == j => {
                    t.since_heard = 0;
                    if m != Mutation::AbsoluteDeadline {
                        t.timer = 0;
                    }
                }
                ToParent::Progress(_) => {}
                ToParent::Answer(i, a) if i == j => {
                    answer(&mut t, j, a);
                    t.in_flight = None;
                }
                ToParent::Answer(..) => {}
            }
            out.push((format!("Receive({frame:?})"), t));
        } else if s.alive {
            // Time passes while the lane waits on an empty socket: up to the
            // deadline for a wedged worker, one tick short of it for a healthy
            // one, whose next frame comes first.
            let room = if s.wedged { b.deadline } else { b.deadline - 1 };
            if s.since_heard < room {
                let mut t = s.clone();
                t.since_heard += 1;
                t.timer = if m == Mutation::RenewOnPoll { 0 } else { (t.timer + 1).min(b.deadline) };
                out.push(("Tick".to_string(), t));
            }
            if s.timer >= b.deadline {
                let mut t = s.clone();
                t.killed_healthy |= !t.wedged;
                kill(&mut t);
                answer(&mut t, j, Answer::Fault);
                t.in_flight = None;
                out.push((format!("Timeout(j{j})"), t));
            }
        } else {
            let mut t = s.clone();
            if m != Mutation::NoAnswerOnEof {
                answer(&mut t, j, Answer::Fault);
            }
            t.in_flight = None;
            out.push((format!("Eof(j{j})"), t));
        }
    } else if let Some(&frame) = s.to_parent.front() {
        // Frames about a job the lane already finished: read and discarded
        // by the next exchange.
        let mut t = s.clone();
        t.to_parent.pop_front();
        out.push((format!("Discard({frame:?})"), t));
    }

    if !s.alive {
        return out;
    }

    // The worker's reader thread.
    if let Some(&frame) = s.to_worker.front() {
        let mut t = s.clone();
        t.to_worker.pop_front();
        match frame {
            ToWorker::Run(j) => {
                t.queued.push_back(j);
                if m != Mutation::LateRegistration {
                    t.current = Some((j, false));
                }
            }
            ToWorker::Cancel(j) => match t.current {
                Some((c, _)) if c == j => {
                    t.current = Some((c, true));
                    t.fired[j] = true;
                }
                _ => {
                    // Dropped: a real loss only if the request loop still holds that run.
                    let held = t.queued.contains(&j) || matches!(t.exec, Exec::Running { job, .. } if job == j);
                    if held {
                        t.dropped_cancel = true;
                    }
                }
            },
        }
        out.push((format!("ReaderStep({frame:?})"), t));
    }

    if s.faults_left > 0 {
        let mut t = s.clone();
        t.faults_left -= 1;
        kill(&mut t);
        out.push(("Crash".to_string(), t));
        if !s.wedged {
            let mut t = s.clone();
            t.faults_left -= 1;
            t.wedged = true;
            out.push(("Wedge".to_string(), t));
        }
    }

    // The worker's request loop, unless it is stuck in the driver.
    if s.wedged {
        return out;
    }
    match s.exec {
        Exec::Idle => {
            if let Some(&j) = s.queued.front() {
                let mut t = s.clone();
                t.queued.pop_front();
                if m == Mutation::LateRegistration {
                    t.current = Some((j, false));
                }
                t.exec = Exec::Running { job: j, progressed: false };
                out.push((format!("Start(j{j})"), t));
            }
        }
        Exec::Running { job, progressed } => {
            if !progressed {
                let mut t = s.clone();
                t.to_parent.push_back(ToParent::Progress(job));
                t.exec = Exec::Running { job, progressed: true };
                out.push((format!("Progress(j{job})"), t));
            }
            // A run observes its token between steps: a fired token may still
            // be seen too late, an unfired one is never seen.
            let outcomes: &[Answer] = if s.fired[job] { &[Answer::Ok, Answer::Cancelled] } else { &[Answer::Ok] };
            for &a in outcomes {
                let mut t = s.clone();
                t.produced[job] = Some(a);
                t.to_parent.push_back(ToParent::Answer(job, a));
                t.exec = Exec::Clearing(job);
                out.push((format!("Finish(j{job},{a:?})"), t));
            }
        }
        Exec::Clearing(job) => {
            let mut t = s.clone();
            let mine = t.current.is_some_and(|(c, _)| c == job);
            if mine || m == Mutation::ClearAnyAfterAnswer {
                t.current = None;
            }
            t.exec = Exec::Idle;
            out.push((format!("Cleared(j{job})"), t));
        }
    }
    out
}

/// The worker dies (or is killed). What it wrote before stays readable; what
/// it had not read yet is gone with it.
fn kill(t: &mut State) {
    t.alive = false;
    t.wedged = false;
    t.to_worker.clear();
    t.queued.clear();
    t.current = None;
    t.exec = Exec::Idle;
}

fn check_invariants(s: &State) -> Result<(), String> {
    if let Some(j) = s.answers.iter().position(|&n| n > 1) {
        return Err(format!("j{j} answered {} times", s.answers[j]));
    }
    for (j, a) in s.answered.iter().enumerate() {
        if *a == Some(Answer::Cancelled) && !s.asked[j] {
            return Err(format!("j{j} was answered cancelled though its caller never asked"));
        }
        if *a == Some(Answer::Cancelled) && s.produced[j] == Some(Answer::Ok) {
            return Err(format!("j{j} was answered cancelled and also ran to completion in the worker"));
        }
    }
    if s.dropped_cancel {
        return Err("a cancellation reached the worker while it held the run and was dropped".into());
    }
    if s.killed_healthy {
        return Err("a deadline killed a worker that kept reporting".into());
    }
    Ok(())
}

fn check_terminal(s: &State, b: Bounds) -> Result<(), String> {
    for (j, &answers) in s.answers.iter().enumerate().take(b.jobs) {
        if answers != 1 {
            return Err(format!("j{j} ended with {answers} answers"));
        }
    }
    if s.alive && !(s.to_worker.is_empty() && s.to_parent.is_empty() && s.queued.is_empty()) {
        return Err(format!("frames left unread: to the worker {:?}, to the lane {:?}, queued {:?}", s.to_worker, s.to_parent, s.queued));
    }
    Ok(())
}

fn explore(b: Bounds, m: Mutation) -> Result<usize, String> {
    let init = State {
        answers: vec![0; b.jobs],
        answered: vec![None; b.jobs],
        in_flight: None,
        next: 0,
        asked: vec![false; b.jobs],
        cancel_sent: vec![false; b.jobs],
        to_worker: VecDeque::new(),
        to_parent: VecDeque::new(),
        queued: VecDeque::new(),
        current: None,
        fired: vec![false; b.jobs],
        exec: Exec::Idle,
        alive: true,
        produced: vec![None; b.jobs],
        dropped_cancel: false,
        since_heard: 0,
        timer: 0,
        wedged: false,
        killed_healthy: false,
        faults_left: b.faults,
    };
    // Every state by index, with the step that first reached it and the
    // states it leads to, so a cycle that never ends can be found afterwards.
    let mut index: HashMap<State, usize> = HashMap::from([(init.clone(), 0)]);
    let mut states = vec![init];
    let mut parent: Vec<Option<(usize, String)>> = vec![None];
    let mut edges: Vec<Vec<usize>> = Vec::new();
    let mut ends: Vec<usize> = Vec::new();
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
        check_invariants(&s).map_err(|e| format!("{e}\n  {b:?} {m:?}\n  trace: {}", trace(&parent, at)))?;
        let next = successors(&s, b, m);
        if next.is_empty() {
            check_terminal(&s, b).map_err(|e| format!("{e}\n  {b:?} {m:?}\n  trace: {}", trace(&parent, at)))?;
            ends.push(at);
        }
        let mut out = Vec::with_capacity(next.len());
        for (action, t) in next {
            let i = *index.entry(t.clone()).or_insert_with(|| {
                states.push(t);
                parent.push(Some((at, action)));
                states.len() - 1
            });
            out.push(i);
        }
        edges.push(out);
        at += 1;
    }
    // Every state must be able to reach an end: walk the edges backwards
    // from the ends and report the first state that was never reached.
    let mut into: Vec<Vec<usize>> = vec![Vec::new(); states.len()];
    for (from, outs) in edges.iter().enumerate() {
        for &to in outs {
            into[to].push(from);
        }
    }
    let mut ends_reachable = vec![false; states.len()];
    for &e in &ends {
        ends_reachable[e] = true;
    }
    let mut stack = ends;
    while let Some(i) = stack.pop() {
        for &p in &into[i] {
            if !ends_reachable[p] {
                ends_reachable[p] = true;
                stack.push(p);
            }
        }
    }
    if let Some(stuck) = ends_reachable.iter().position(|&r| !r) {
        return Err(format!("a cycle that never ends: no end is reachable from here\n  {b:?} {m:?}\n  trace: {}", trace(&parent, stuck)));
    }
    Ok(states.len())
}

fn explore_all(m: Mutation) -> Result<usize, String> {
    let mut states = 0;
    for jobs in 1..=3 {
        for faults in 0..=1 {
            states += explore(Bounds { jobs, faults, deadline: 2 }, m)?;
        }
    }
    Ok(states)
}

#[test]
fn a_hosted_models_jobs_keep_their_promises_across_the_worker_boundary() {
    let states = explore_all(Mutation::None).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("{states} states explored");
    assert!(states > 1_000, "the bounds explored too little: {states} states");
}

#[test]
fn every_mutation_of_the_exchange_is_caught() {
    let mutations = [
        Mutation::LateRegistration,
        Mutation::ClearAnyAfterAnswer,
        Mutation::AnswerAtCancel,
        Mutation::NoAnswerOnEof,
        Mutation::AbsoluteDeadline,
        Mutation::RenewOnPoll,
    ];
    for m in mutations {
        let why = explore_all(m).err().unwrap_or_else(|| panic!("{m:?} broke the exchange and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
