// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that request
// protocols between a server and its worker processes keep their promises,
// for its clients. If your team needs expertise in formal verification of
// cancellation and crash recovery then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Exhaustive model check of one hosted model's jobs crossing the worker
//! boundary: the lane's exchange (`SupervisedInstance::batch_in_worker`,
//! `Worker::exchange`), the two sockets' frame queues, and the worker's
//! reader thread and request loop (`supervised::host`), with progress
//! frames, per-job cancellation and a worker that may die at any point.
//!
//! The lane's jobs go to the worker in groups of up to `batch`, one `batch`
//! request per group, as the executor hands a group to `run_batch`; the
//! worker's own `run_batch` works on the group's jobs in any order (one
//! after another, as the default does, or interleaved, as a model that
//! batches does), skipping a job whose token fired before it ran, and
//! answers once with every job's result. The caller of each job may ask to
//! cancel it at any time; the worker may be killed, or wedge (its request
//! loop stops inside the driver while its reader thread runs on), at any
//! time within a fault budget, after which the next group starts a fresh
//! worker on a fresh socket.
//!
//! Time is ticks of the lane's deadline, which every frame the lane reads
//! about the request in flight renews (`Exchange::renew`): a request may run
//! far longer than one deadline as long as it keeps reporting. The
//! environment is fair to a healthy worker: it reports within one deadline,
//! so time stops one tick short of the deadline until its next frame is
//! read. A wedged worker reports nothing, and time runs out.
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
//! | `Send` | `batch_in_worker` writing `batch <action> <k> <envelope>` (`wire::send`); a dead worker's successor is spawned by the next activation |
//! | `CallerCancel` | the caller firing the job's `CancelToken` |
//! | `SendCancel` | `JobsTalk::poll` seeing a job's token fired and writing `<id> cancel <job>` once |
//! | `Receive` | `Worker::exchange` reading a frame: a job's progress is forwarded and renews the deadline, the answer ends the exchange with every job's result |
//! | `Eof` | `Worker::exchange` reading end of stream: every job of the request gets the fault |
//! | `Tick` | time passing while `Worker::exchange` waits on the socket |
//! | `Timeout` | `Worker::exchange` finding its deadline passed: the worker is killed, every job gets the fault |
//! | `Wedge` | the environment: the worker's request loop stops inside the driver |
//! | `ReaderStep` | `host::read_requests`: a request registers one token per job as `Current` and is queued; a cancel fires the token of the job it names in the request it names |
//! | `Start` | `Host::batch` taking the queued request |
//! | `Progress` | the model's progress callback writing `<id> progress {"job": j}` |
//! | `Finish` | the model's `run_batch` finishing one job (cancelled only if its token fired) |
//! | `Answer` | `Host::batch` answering with every result |
//! | `Cleared` | `Host::handle` clearing `Current` if it still names this request |
//! | `Crash` | the environment: the worker dies |

use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Answer {
    Ok,
    Cancelled,
    Fault,
}

/// A request is named by its index; request `r` holds jobs
/// `r * batch .. (r + 1) * batch`, as many as there are.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToWorker {
    Run(usize),
    Cancel(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToParent {
    Progress(usize),
    /// The request's answer: each of its jobs' results, as `produced` holds them.
    Answer(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Exec {
    Idle,
    Running(usize),
    /// Answered the request, has not cleared `Current` yet.
    Clearing(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    answers: Vec<u8>,
    answered: Vec<Option<Answer>>,
    /// The request the lane is exchanging, if any.
    in_flight: Option<usize>,
    /// The next request to send.
    next: usize,
    asked: Vec<bool>,
    cancel_sent: Vec<bool>,
    to_worker: VecDeque<ToWorker>,
    to_parent: VecDeque<ToParent>,
    /// The reader has queued these requests for the request loop.
    queued: VecDeque<usize>,
    /// `Current`: the request read last and not finished; its jobs' tokens
    /// are `registered`.
    current: Option<usize>,
    /// The job's token is in `Current`, so a cancel naming it fires it.
    registered: Vec<bool>,
    /// The job's token fired in the worker.
    fired: Vec<bool>,
    /// The job sent its one progress frame.
    progressed: Vec<bool>,
    exec: Exec,
    alive: bool,
    /// What the worker produced for each job, if it ran to its end.
    produced: Vec<Option<Answer>>,
    /// A cancellation the reader read for a job it held and did not deliver.
    dropped_cancel: bool,
    /// Ticks since the lane last read a frame about the request in flight (or sent it).
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
    /// Jobs per request.
    batch: usize,
    /// Crashes and wedges together.
    faults: u8,
    /// Ticks in one deadline.
    deadline: u8,
}

impl Bounds {
    fn requests(&self) -> usize {
        self.jobs.div_ceil(self.batch)
    }

    fn jobs_of(&self, r: usize) -> std::ops::Range<usize> {
        r * self.batch..((r + 1) * self.batch).min(self.jobs)
    }

    fn request_of(&self, j: usize) -> usize {
        j / self.batch
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    None,
    /// The tokens are registered when the request starts, not when it is read.
    LateRegistration,
    /// `Current` is cleared unconditionally, after the answer was sent.
    ClearAnyAfterAnswer,
    /// The lane answers a job `cancelled` as soon as it sends its cancel frame.
    AnswerAtCancel,
    /// End of stream frees the lane without answering the request's jobs.
    NoAnswerOnEof,
    /// The deadline counts from the request, as before progress renewed it.
    AbsoluteDeadline,
    /// The deadline starts over at every wake-up of the wait (the poll for a
    /// cancellation), not only at a frame from the worker.
    RenewOnPoll,
    /// A cancel fires every token of the request, not only its job's.
    CancelWholeRequest,
    /// Only the request's first job gets a token a cancel can reach.
    FirstTokenOnly,
}

fn answer(t: &mut State, job: usize, a: Answer) {
    t.answers[job] += 1;
    t.answered[job] = Some(a);
}

fn successors(s: &State, b: Bounds, m: Mutation) -> Vec<(String, State)> {
    let mut out = Vec::new();

    // The lane.
    if s.in_flight.is_none() && s.next < b.requests() {
        let mut t = s.clone();
        let r = t.next;
        t.next += 1;
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
        t.to_worker.push_back(ToWorker::Run(r));
        t.in_flight = Some(r);
        t.since_heard = 0;
        t.timer = 0;
        out.push((format!("Send(r{r})"), t));
    }
    for j in 0..b.jobs {
        if !s.asked[j] && s.answers[j] == 0 {
            let mut t = s.clone();
            t.asked[j] = true;
            out.push((format!("CallerCancel(j{j})"), t));
        }
    }
    if let Some(r) = s.in_flight {
        for j in b.jobs_of(r) {
            if s.asked[j] && !s.cancel_sent[j] && s.answers[j] == 0 && s.alive {
                let mut t = s.clone();
                t.cancel_sent[j] = true;
                t.to_worker.push_back(ToWorker::Cancel(j));
                if m == Mutation::AnswerAtCancel {
                    answer(&mut t, j, Answer::Cancelled);
                }
                out.push((format!("SendCancel(j{j})"), t));
            }
        }
        if let Some(&frame) = s.to_parent.front() {
            let mut t = s.clone();
            t.to_parent.pop_front();
            match frame {
                // Frames are read in order and a request's answer is its last
                // frame, so a job's progress is about the request in flight.
                ToParent::Progress(j) if b.request_of(j) == r => {
                    t.since_heard = 0;
                    if m != Mutation::AbsoluteDeadline {
                        t.timer = 0;
                    }
                }
                ToParent::Progress(_) => {}
                ToParent::Answer(i) if i == r => {
                    for j in b.jobs_of(r) {
                        let a = t.produced[j].expect("an answered request produced every result");
                        answer(&mut t, j, a);
                    }
                    t.in_flight = None;
                }
                ToParent::Answer(_) => {}
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
                fail_request(&mut t, b, r);
                out.push((format!("Timeout(r{r})"), t));
            }
        } else {
            let mut t = s.clone();
            if m != Mutation::NoAnswerOnEof {
                fail_request(&mut t, b, r);
            }
            t.in_flight = None;
            out.push((format!("Eof(r{r})"), t));
        }
    } else if let Some(&frame) = s.to_parent.front() {
        // Frames about a request the lane already finished: read and
        // discarded by the next exchange.
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
            ToWorker::Run(r) => {
                t.queued.push_back(r);
                if m != Mutation::LateRegistration {
                    register(&mut t, b, r, m);
                }
            }
            ToWorker::Cancel(j) => {
                let r = b.request_of(j);
                if t.current == Some(r) && t.registered[j] {
                    if m == Mutation::CancelWholeRequest {
                        for k in b.jobs_of(r) {
                            t.fired[k] = true;
                        }
                    } else {
                        t.fired[j] = true;
                    }
                } else {
                    // Dropped: a real loss only if the request loop still holds that job.
                    let held = t.queued.contains(&r) || (t.exec == Exec::Running(r) && t.produced[j].is_none());
                    if held {
                        t.dropped_cancel = true;
                    }
                }
            }
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
            if let Some(&r) = s.queued.front() {
                let mut t = s.clone();
                t.queued.pop_front();
                if m == Mutation::LateRegistration {
                    register(&mut t, b, r, m);
                }
                t.exec = Exec::Running(r);
                out.push((format!("Start(r{r})"), t));
            }
        }
        Exec::Running(r) => {
            let unfinished: Vec<usize> = b.jobs_of(r).filter(|&j| s.produced[j].is_none()).collect();
            for &j in &unfinished {
                if !s.progressed[j] {
                    let mut t = s.clone();
                    t.to_parent.push_back(ToParent::Progress(j));
                    t.progressed[j] = true;
                    out.push((format!("Progress(j{j})"), t));
                }
                // A job observes its token between steps (or before it starts):
                // a fired token may still be seen too late, an unfired one is
                // never seen.
                let outcomes: &[Answer] = if s.fired[j] { &[Answer::Ok, Answer::Cancelled] } else { &[Answer::Ok] };
                for &a in outcomes {
                    let mut t = s.clone();
                    t.produced[j] = Some(a);
                    out.push((format!("Finish(j{j},{a:?})"), t));
                }
            }
            if unfinished.is_empty() {
                let mut t = s.clone();
                t.to_parent.push_back(ToParent::Answer(r));
                t.exec = Exec::Clearing(r);
                out.push((format!("Answer(r{r})"), t));
            }
        }
        Exec::Clearing(r) => {
            let mut t = s.clone();
            if t.current == Some(r) || m == Mutation::ClearAnyAfterAnswer {
                t.current = None;
                t.registered.iter_mut().for_each(|x| *x = false);
            }
            t.exec = Exec::Idle;
            out.push((format!("Cleared(r{r})"), t));
        }
    }
    out
}

/// The reader makes request `r` current, with a token for each of its jobs.
fn register(t: &mut State, b: Bounds, r: usize, m: Mutation) {
    t.current = Some(r);
    t.registered.iter_mut().for_each(|x| *x = false);
    for j in b.jobs_of(r) {
        t.registered[j] = m != Mutation::FirstTokenOnly || j == b.jobs_of(r).start;
    }
}

/// Every job of request `r` not answered yet gets the worker's fault.
fn fail_request(t: &mut State, b: Bounds, r: usize) {
    for j in b.jobs_of(r) {
        if t.answers[j] == 0 {
            answer(t, j, Answer::Fault);
        }
    }
    t.in_flight = None;
}

/// The worker dies (or is killed). What it wrote before stays readable; what
/// it had not read yet is gone with it.
fn kill(t: &mut State) {
    t.alive = false;
    t.wedged = false;
    t.to_worker.clear();
    t.queued.clear();
    t.current = None;
    t.registered.iter_mut().for_each(|x| *x = false);
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
        return Err("a cancellation reached the worker while it held the job and was dropped".into());
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
        registered: vec![false; b.jobs],
        fired: vec![false; b.jobs],
        progressed: vec![false; b.jobs],
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
    for (jobs, batch) in [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2)] {
        for faults in 0..=1 {
            states += explore(Bounds { jobs, batch, faults, deadline: 2 }, m)?;
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
        Mutation::CancelWholeRequest,
        Mutation::FirstTokenOnly,
    ];
    for m in mutations {
        let why = explore_all(m).err().unwrap_or_else(|| panic!("{m:?} broke the exchange and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
