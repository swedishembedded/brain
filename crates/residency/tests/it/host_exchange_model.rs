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
//! killed at any time within a fault budget, after which the next job starts
//! a fresh worker on a fresh socket.
//!
//! In every state:
//!
//! * no job is answered twice;
//! * a job is answered `cancelled` only if its own caller asked;
//! * a job answered `cancelled` did not also run to completion in the worker
//!   (a caller told "cancelled" must be able to trust nothing was done);
//! * a cancellation that reached the worker while its job was waiting to
//!   start or running is never dropped;
//!
//! and in every state with no successor:
//!
//! * every job got exactly one answer;
//! * no frame is left unread on a live socket.
//!
//! A deliberate bug in the model must make it fail: see
//! `every_mutation_of_the_exchange_is_caught`.
//!
//! | action | code |
//! |---|---|
//! | `Send` | `run_in_worker` writing `run <action> <envelope>` (`wire::send`); a dead worker's successor is spawned by the next activation |
//! | `CallerCancel` | the caller firing the job's `CancelToken` |
//! | `SendCancel` | `Worker::exchange` seeing `cancelled()` and writing `<id> cancel` once |
//! | `Receive` | `Worker::exchange` reading a frame: progress for the job in flight is forwarded, its answer ends the exchange |
//! | `SkipCancelled` | `Instance::run_batch` answering a job cancelled before its turn without running it |
//! | `Eof` | `Worker::exchange` reading end of stream: the job's fault answer |
//! | `ReaderStep` | `host::read_requests`: a run registers its token as `Current` and is queued; a cancel fires the token of the run it names |
//! | `Start` | `Host::run` taking the queued run |
//! | `Progress` | the model's progress callback writing `<id> progress` |
//! | `Finish` | `Host::run` answering, then `Host::handle` clearing `Current` if it still names this run |
//! | `Crash` | the environment: the worker dies |

use std::collections::{HashMap, HashSet, VecDeque};

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
    crashes_left: u8,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    jobs: usize,
    crashes: u8,
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
                t.to_worker.clear();
                t.to_parent.clear();
                t.queued.clear();
                t.current = None;
                t.exec = Exec::Idle;
            }
            t.to_worker.push_back(ToWorker::Run(j));
            t.in_flight = Some(j);
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
                ToParent::Progress(_) => {}
                ToParent::Answer(i, a) if i == j => {
                    answer(&mut t, j, a);
                    t.in_flight = None;
                }
                ToParent::Answer(..) => {}
            }
            out.push((format!("Receive({frame:?})"), t));
        } else if !s.alive {
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

    // The worker's request loop.
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

    if s.crashes_left > 0 {
        let mut t = s.clone();
        t.crashes_left -= 1;
        t.alive = false;
        // What the worker wrote before it died stays readable; what it had
        // not read yet is gone with it.
        t.to_worker.clear();
        t.queued.clear();
        t.current = None;
        t.exec = Exec::Idle;
        out.push(("Crash".to_string(), t));
    }
    out
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
        crashes_left: b.crashes,
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
        check_invariants(&s).map_err(|e| format!("{e}\n  {b:?} {m:?}\n  trace: {}", trace(&parent, s.clone())))?;
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

fn explore_all(m: Mutation) -> Result<usize, String> {
    let mut states = 0;
    for jobs in 1..=3 {
        for crashes in 0..=1 {
            states += explore(Bounds { jobs, crashes }, m)?;
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
    for m in [Mutation::LateRegistration, Mutation::ClearAnyAfterAnswer, Mutation::AnswerAtCancel, Mutation::NoAnswerOnEof] {
        let why = explore_all(m).err().unwrap_or_else(|| panic!("{m:?} broke the exchange and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
