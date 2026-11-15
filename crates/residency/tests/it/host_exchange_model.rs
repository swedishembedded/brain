// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that request
// protocols between a server and its worker processes keep their promises,
// for its clients. If your team needs expertise in formal verification of
// cancellation and crash recovery then you can procure our services by
// sending an email to info@swedishembedded.com.

//! Exhaustive model check of one hosted model's jobs crossing the worker
//! boundary: the lane's exchange (`SupervisedInstance::batch_in_worker` and
//! `run_open`, `Worker::exchange`), the two sockets' frame queues, and the
//! worker's reader thread and request loop (`supervised::host`), with
//! progress frames, per-job cancellation, late joins into an open group and
//! a worker that may die at any point.
//!
//! Two shapes of work are explored:
//!
//! * **Closed groups.** The lane's jobs go to the worker in groups of up to
//!   `batch`, one `batch` request per group, as the executor hands a group
//!   to `run_batch`; the worker's own `run_batch` works on the group's jobs
//!   in any order (one after another, as the default does, or interleaved,
//!   as a model that batches does), skipping a job whose token fired before
//!   it ran, and answers once with every job's result.
//! * **An open group.** The first `batch` jobs go out as one `group`
//!   request, as the executor hands them to `run_open`, and `late` more
//!   arrive in the parent's queue at any time. Between steps the worker's
//!   `run_open` asks for joiners (`take`, answered `joined` with what the
//!   parent's `JobFeed` gave), finishes each job as soon as it is done
//!   (`done`), and once it holds nothing asks to close (`close`, answered
//!   `closed true` only when no job is waiting); a job that arrives after
//!   the close is the next group's.
//!
//! The caller of each job may ask to cancel it at any time; the worker may
//! be killed, or wedge (its request loop stops inside the driver while its
//! reader thread runs on), at any time within a fault budget, after which
//! the next request starts a fresh worker on a fresh socket.
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
//! * a cancellation that reached the worker while it held the job (queued,
//!   joining or running) is never dropped;
//! * a deadline kills only a worker that went silent (wedged), never one
//!   that kept reporting;
//!
//! in every state with no successor:
//!
//! * every job got exactly one answer, except a late job that arrived after
//!   its group closed (the next group's);
//! * no late job is left waiting for a group that ended healthy (the
//!   executor would fail it for arriving while the group ended);
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
//! | `Send` | `batch_in_worker` / `run_open` writing `batch` / `group <action> <k> <jobs>` (`wire::send`); a dead worker's successor is spawned by the next activation |
//! | `CallerCancel` | the caller firing the job's `CancelToken` |
//! | `Arrive` | the dispatcher queueing a late job in the group's inbox |
//! | `SendCancel` | `JobsTalk::poll` seeing a job's token fired and writing `<id> cancel <job>` once |
//! | `Receive` | `Worker::exchange` reading a frame, every one renewing the deadline: a job's progress is forwarded; `take` is answered `joined` with what `JobFeed::take` gave; `done` finishes its job through `JobFeed::finish`; `close` is answered with `JobFeed::close`; the answer ends the exchange |
//! | `Eof` | `Worker::exchange` reading end of stream: every job of the request not finished yet gets the fault |
//! | `Tick` | time passing while `Worker::exchange` waits on the socket |
//! | `Timeout` | `Worker::exchange` finding its deadline passed: the worker is killed, every unfinished job gets the fault |
//! | `Wedge` | the environment: the worker's request loop stops inside the driver |
//! | `ReaderStep` | `host::read_requests`: a request registers one token per job as `Current` and is queued; `joined` registers its jobs' tokens and is queued; a cancel fires the token of the job it names in the request it names |
//! | `Start` | `Host::batch` / `Host::group` taking the queued request |
//! | `Progress` | the model's progress callback writing `<id> progress {"job": j}` |
//! | `Finish` | the model finishing one job (cancelled only if its token fired); in an open group `WorkerFeed::finish` writes `done` |
//! | `Take` / `Joined` | `WorkerFeed::take` writing `take` and taking the `joined` reply off the request loop's channel |
//! | `Close` / `Closed` | `WorkerFeed::close` writing `close` and taking the `closed` reply |
//! | `Answer` | `Host::batch` answering with every result, `Host::group` answering that it ran the group |
//! | `Cleared` | `Host::handle` clearing `Current` if it still names this request |
//! | `Crash` | the environment: the worker dies |

use std::collections::{HashMap, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Answer {
    Ok,
    Cancelled,
    Fault,
    /// The group ended without finishing the job.
    Unfinished,
}

/// A request is named by its index; request `r` holds jobs
/// `r * batch .. (r + 1) * batch`, as many as there are; an open group's
/// late jobs follow its first `batch`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToWorker {
    Run(usize),
    Cancel(usize),
    /// The reply to `take`: the job that joined, if one did.
    Joined(Option<usize>),
    /// The reply to `close`.
    Closed(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToParent {
    Progress(usize),
    /// An open group's job finished.
    Done(usize, Answer),
    Take,
    Close,
    /// The request's answer: for a closed group each of its jobs' results, as
    /// `produced` holds them.
    Answer(usize),
}

/// What the reader hands the request loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ToLoop {
    Run(usize),
    Joined(Option<usize>),
    Closed(bool),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Exec {
    Idle,
    Running(usize),
    /// An open group waiting for the reply to its `take`.
    Joining(usize),
    /// An open group waiting for the reply to its `close`.
    Closing(usize),
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
    /// The jobs of the request in flight, its own and the ones that joined.
    in_group: Vec<bool>,
    asked: Vec<bool>,
    cancel_sent: Vec<bool>,
    /// The late job is in the parent's queue (or was).
    arrived: Vec<bool>,
    /// Late jobs waiting in the group's inbox, in order.
    waiting: VecDeque<usize>,
    /// `JobFeed::close` returned true: later arrivals are the next group's.
    closed: bool,
    /// The request in flight faulted (its worker died or was killed).
    faulted: bool,
    to_worker: VecDeque<ToWorker>,
    to_parent: VecDeque<ToParent>,
    /// The reader has handed these to the request loop.
    queued: VecDeque<ToLoop>,
    /// `Current`: the request read last and not finished; its jobs' tokens
    /// are `registered`.
    current: Option<usize>,
    /// The job's token is in `Current`, so a cancel naming it fires it.
    registered: Vec<bool>,
    /// The job's token fired in the worker.
    fired: Vec<bool>,
    /// The worker's request loop holds the job (it started or joined).
    held: Vec<bool>,
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
    /// Jobs per request; an open group's own jobs.
    batch: usize,
    /// Jobs that arrive while an open group runs; 0 for closed groups.
    late: usize,
    /// Crashes and wedges together.
    faults: u8,
    /// Ticks in one deadline.
    deadline: u8,
}

impl Bounds {
    fn closed(jobs: usize, batch: usize, faults: u8) -> Bounds {
        Bounds { jobs, batch, late: 0, faults, deadline: 2 }
    }

    fn open(own: usize, late: usize, faults: u8) -> Bounds {
        Bounds { jobs: own + late, batch: own, late, faults, deadline: 2 }
    }

    fn is_open(&self) -> bool {
        self.late > 0
    }

    fn requests(&self) -> usize {
        if self.is_open() {
            1
        } else {
            self.jobs.div_ceil(self.batch)
        }
    }

    fn jobs_of(&self, r: usize) -> std::ops::Range<usize> {
        r * self.batch..((r + 1) * self.batch).min(self.jobs)
    }

    fn is_late(&self, j: usize) -> bool {
        self.is_open() && j >= self.batch
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
    /// A joined job's token is registered when `take` returns it, not when
    /// the reader reads `joined`.
    LateJoinRegistration,
    /// A fault answers every job of the group, finished or not.
    FaultAnswersFinished,
    /// A fault answers the group's own jobs and forgets the ones that joined.
    ForgetJoined,
    /// The worker ends the group on `closed false` too.
    IgnoreClosedFalse,
}

fn answer(t: &mut State, job: usize, a: Answer) {
    t.answers[job] += 1;
    t.answered[job] = Some(a);
}

fn lane(s: &State, b: Bounds, m: Mutation, out: &mut Vec<(String, State)>) {
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
        t.in_group = vec![false; b.jobs];
        for j in b.jobs_of(r) {
            t.in_group[j] = true;
        }
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
        if b.is_late(j) && !s.arrived[j] && s.next > 0 {
            let mut t = s.clone();
            t.arrived[j] = true;
            // After the close, or once the group is over, it is the next group's.
            if !t.closed && t.in_flight.is_some() {
                t.waiting.push_back(j);
            }
            out.push((format!("Arrive(j{j})"), t));
        }
    }
    let Some(r) = s.in_flight else {
        if let Some(&frame) = s.to_parent.front() {
            // Frames about a request the lane already finished: read and
            // discarded by the next exchange.
            let mut t = s.clone();
            t.to_parent.pop_front();
            out.push((format!("Discard({frame:?})"), t));
        }
        return;
    };
    for j in 0..b.jobs {
        if s.in_group[j] && s.asked[j] && !s.cancel_sent[j] && s.answers[j] == 0 && s.alive {
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
        // Frames are read in order and a request's answer is its last frame,
        // so every frame read here is about the request in flight.
        t.since_heard = 0;
        if m != Mutation::AbsoluteDeadline && !matches!(frame, ToParent::Answer(_)) {
            t.timer = 0;
        }
        match frame {
            ToParent::Progress(_) => {}
            ToParent::Done(j, a) => answer(&mut t, j, a),
            ToParent::Take => {
                let joined = t.waiting.pop_front();
                if let Some(j) = joined {
                    t.in_group[j] = true;
                }
                t.to_worker.push_back(ToWorker::Joined(joined));
            }
            ToParent::Close => {
                let empty = t.waiting.is_empty();
                t.closed |= empty;
                t.to_worker.push_back(ToWorker::Closed(empty));
            }
            ToParent::Answer(i) => {
                assert_eq!(i, r, "an answer is about the request in flight");
                for j in 0..b.jobs {
                    if !t.in_group[j] || t.answers[j] > 0 {
                        continue;
                    }
                    // A closed group's answer carries every result; an open
                    // group's jobs were each finished by `done`.
                    let a = if b.is_open() { Answer::Unfinished } else { t.produced[j].expect("an answered request produced every result") };
                    answer(&mut t, j, a);
                }
                t.in_flight = None;
            }
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
            fail_request(&mut t, b, m);
            out.push((format!("Timeout(r{r})"), t));
        }
    } else {
        let mut t = s.clone();
        if m == Mutation::NoAnswerOnEof {
            t.in_flight = None;
        } else {
            fail_request(&mut t, b, m);
        }
        out.push((format!("Eof(r{r})"), t));
    }
}

fn reader(s: &State, b: Bounds, m: Mutation, out: &mut Vec<(String, State)>) {
    let Some(&frame) = s.to_worker.front() else { return };
    let mut t = s.clone();
    t.to_worker.pop_front();
    match frame {
        ToWorker::Run(r) => {
            t.queued.push_back(ToLoop::Run(r));
            if m != Mutation::LateRegistration {
                register(&mut t, b, r, m);
            }
        }
        ToWorker::Joined(joined) => {
            t.queued.push_back(ToLoop::Joined(joined));
            if let (Some(j), true) = (joined, m != Mutation::LateJoinRegistration) {
                t.registered[j] = t.current.is_some();
            }
        }
        ToWorker::Closed(c) => t.queued.push_back(ToLoop::Closed(c)),
        ToWorker::Cancel(j) => {
            if t.current.is_some() && t.registered[j] {
                if m == Mutation::CancelWholeRequest {
                    for k in 0..b.jobs {
                        t.fired[k] |= t.registered[k];
                    }
                } else {
                    t.fired[j] = true;
                }
            } else {
                // Dropped: a real loss only if the request loop still holds that job.
                let queued = t.queued.iter().any(|q| match *q {
                    ToLoop::Run(r) => b.jobs_of(r).contains(&j),
                    ToLoop::Joined(x) => x == Some(j),
                    ToLoop::Closed(_) => false,
                });
                let running = t.held[j] && t.produced[j].is_none();
                if queued || running {
                    t.dropped_cancel = true;
                }
            }
        }
    }
    out.push((format!("ReaderStep({frame:?})"), t));
}

fn request_loop(s: &State, b: Bounds, m: Mutation, out: &mut Vec<(String, State)>) {
    match s.exec {
        Exec::Idle => {
            if let Some(&ToLoop::Run(r)) = s.queued.front() {
                let mut t = s.clone();
                t.queued.pop_front();
                if m == Mutation::LateRegistration {
                    register(&mut t, b, r, m);
                }
                for j in b.jobs_of(r) {
                    t.held[j] = true;
                }
                t.exec = Exec::Running(r);
                out.push((format!("Start(r{r})"), t));
            }
        }
        Exec::Running(r) => {
            let unfinished: Vec<usize> = (0..b.jobs).filter(|&j| s.held[j] && s.produced[j].is_none()).collect();
            for &j in &unfinished {
                // An open group's `take`, `done` and `close` frames renew the deadline as
                // progress does, so its jobs send none.
                if !s.progressed[j] && !b.is_open() {
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
                    if b.is_open() {
                        t.to_parent.push_back(ToParent::Done(j, a));
                    }
                    out.push((format!("Finish(j{j},{a:?})"), t));
                }
            }
            if b.is_open() {
                // Between steps the group asks for joiners, while one could still come.
                if (0..b.jobs).any(|j| b.is_late(j) && !s.held[j]) {
                    let mut t = s.clone();
                    t.to_parent.push_back(ToParent::Take);
                    t.exec = Exec::Joining(r);
                    out.push(("Take".to_string(), t));
                }
                if unfinished.is_empty() {
                    let mut t = s.clone();
                    t.to_parent.push_back(ToParent::Close);
                    t.exec = Exec::Closing(r);
                    out.push(("Close".to_string(), t));
                }
            } else if unfinished.is_empty() {
                let mut t = s.clone();
                t.to_parent.push_back(ToParent::Answer(r));
                t.exec = Exec::Clearing(r);
                out.push((format!("Answer(r{r})"), t));
            }
        }
        Exec::Joining(r) => {
            if let Some(&ToLoop::Joined(joined)) = s.queued.front() {
                let mut t = s.clone();
                t.queued.pop_front();
                if let Some(j) = joined {
                    t.held[j] = true;
                    if m == Mutation::LateJoinRegistration {
                        t.registered[j] = t.current.is_some();
                    }
                }
                t.exec = Exec::Running(r);
                out.push((format!("Joined({joined:?})"), t));
            }
        }
        Exec::Closing(r) => {
            if let Some(&ToLoop::Closed(closed)) = s.queued.front() {
                let mut t = s.clone();
                t.queued.pop_front();
                if closed || m == Mutation::IgnoreClosedFalse {
                    t.to_parent.push_back(ToParent::Answer(r));
                    t.exec = Exec::Clearing(r);
                } else {
                    t.exec = Exec::Running(r);
                }
                out.push((format!("Closed({closed})"), t));
            }
        }
        Exec::Clearing(r) => {
            let mut t = s.clone();
            if t.current == Some(r) || m == Mutation::ClearAnyAfterAnswer {
                t.current = None;
                t.registered.iter_mut().for_each(|x| *x = false);
            }
            t.held.iter_mut().for_each(|x| *x = false);
            t.exec = Exec::Idle;
            out.push((format!("Cleared(r{r})"), t));
        }
    }
}

fn successors(s: &State, b: Bounds, m: Mutation) -> Vec<(String, State)> {
    let mut out = Vec::new();
    lane(s, b, m, &mut out);
    if !s.alive {
        return out;
    }
    reader(s, b, m, &mut out);
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
    if !s.wedged {
        request_loop(s, b, m, &mut out);
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

/// Every job of the request in flight not answered yet gets the worker's fault.
fn fail_request(t: &mut State, b: Bounds, m: Mutation) {
    let r = t.in_flight.expect("a request is in flight");
    for j in 0..b.jobs {
        let in_it = if m == Mutation::ForgetJoined { b.jobs_of(r).contains(&j) } else { t.in_group[j] };
        if in_it && (t.answers[j] == 0 || m == Mutation::FaultAnswersFinished) {
            answer(t, j, Answer::Fault);
        }
    }
    // The lane fails the group, and the executor answers the jobs still
    // waiting to join it with why it ended.
    for j in std::mem::take(&mut t.waiting) {
        answer(t, j, Answer::Fault);
    }
    t.faulted = true;
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
    t.held.iter_mut().for_each(|x| *x = false);
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
        if *a == Some(Answer::Unfinished) {
            return Err(format!("j{j} was in a group that ended without finishing it"));
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
        // A late job that arrived after the close, or after the group was
        // over, never joined it: it is the next group's to answer.
        let next_groups = b.is_late(j) && !s.in_group[j] && !s.waiting.contains(&j);
        if answers != 1 && !next_groups {
            return Err(format!("j{j} ended with {answers} answers"));
        }
    }
    if !s.faulted && !s.waiting.is_empty() {
        return Err(format!("late jobs {:?} were left waiting for a group that ended healthy", s.waiting));
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
        in_group: vec![false; b.jobs],
        asked: vec![false; b.jobs],
        cancel_sent: vec![false; b.jobs],
        arrived: vec![false; b.jobs],
        waiting: VecDeque::new(),
        closed: false,
        faulted: false,
        to_worker: VecDeque::new(),
        to_parent: VecDeque::new(),
        queued: VecDeque::new(),
        current: None,
        registered: vec![false; b.jobs],
        fired: vec![false; b.jobs],
        held: vec![false; b.jobs],
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
    for faults in 0..=1 {
        for b in [Bounds::closed(1, 1, faults), Bounds::closed(2, 1, faults), Bounds::closed(3, 1, faults), Bounds::closed(2, 2, faults), Bounds::closed(3, 2, faults), Bounds::open(1, 1, faults), Bounds::open(1, 2, faults), Bounds::open(2, 1, faults)] {
            states += explore(b, m)?;
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
        Mutation::LateJoinRegistration,
        Mutation::FaultAnswersFinished,
        Mutation::ForgetJoined,
        Mutation::IgnoreClosedFalse,
    ];
    for m in mutations {
        let why = explore_all(m).err().unwrap_or_else(|| panic!("{m:?} broke the exchange and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
