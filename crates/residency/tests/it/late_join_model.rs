// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for proving that concurrent
// schedulers never lose or double-answer work for its clients. If your team
// needs expertise in formal verification of concurrent systems then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Exhaustive model check of the late-join protocol between the dispatcher and
//! a running group (`Inbox` in `src/executor.rs`).
//!
//! Each action below is one critical section of the real code, taken under the
//! inbox mutex, so the interleavings of these actions are every behaviour the
//! threads can produce. The explorer visits every reachable state for small
//! bounds and checks, in every state:
//!
//! * no job is answered twice;
//! * `room` plus the jobs the group runs equals the model's batch limit, so a
//!   group never runs more than its limit;
//! * a job is never in two places at once;
//!
//! and in every state with no successor (the protocol has quiesced):
//!
//! * every job has been answered exactly once (none stranded in the inbox or
//!   the queue);
//! * a job answered with an error has a cause: the model refused it, the group
//!   panicked, or its instance failed to build. A healthy group never errors a
//!   job that merely arrived late.
//!
//! The mapping to the code:
//!
//! | action | code |
//! |---|---|
//! | `Refill` | `refill_open_groups` (under the inbox lock) |
//! | `Take` | `GroupFeed::take` |
//! | `Screen` | `GroupFeed::screen` refusing a job |
//! | `Finish` | `GroupFeed::finish` |
//! | `Close` | `GroupFeed::close` in `run_open_scheduled`'s idle branch |
//! | `CloseBatch` | `run_group`'s closed-batch tail |
//! | `Panic` | `lane_loop`'s `catch_unwind` arm |
//! | `BuildFails` | `run_req`'s failed `activate` |
//! | `LaneEnd` | `lane_loop`'s inbox drain and `Msg::Done` |
//! | `Forget` / `NextGroup` | the dispatcher dropping the open group on `Done` and assigning the queue |

use std::collections::{HashSet, VecDeque};

/// Where a job is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum At {
    Queue,
    Inbox,
    /// Admitted to the group: one of its own jobs, or a late one it took.
    Live,
    Answered(Answer),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Answer {
    Ok,
    /// The model turned the request away (it can never run).
    Refused,
    Panicked,
    BuildFailed,
    /// Errored for having arrived while the group ended: legitimate only when
    /// the group failed.
    GroupEnded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Lane {
    /// The instance is being built; late jobs may already arrive.
    Building,
    /// `run_open_scheduled`'s loop.
    Open,
    /// The model returned; `lane_loop` has not yet drained and sent `Done`.
    Returned { failed: bool },
    /// `Done` sent; the dispatcher still lists the group as open.
    DoneSent { failed: bool },
    /// The dispatcher forgot the group.
    Gone { failed: bool },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    jobs: Vec<At>,
    /// Answers per job, to catch a second one.
    answers: Vec<u8>,
    open: bool,
    room: usize,
    /// Jobs in the inbox, in arrival order.
    inbox: Vec<usize>,
    taken: usize,
    lane: Lane,
    /// The model batches only closed groups (`run_open` returned false).
    closed_batch: bool,
}

#[derive(Clone, Copy, Debug)]
struct Bounds {
    jobs: usize,
    limit: usize,
    lookahead: usize,
    /// Jobs the group starts with.
    initial: usize,
}

impl State {
    fn live(&self) -> usize {
        self.jobs.iter().filter(|a| **a == At::Live).count()
    }

    fn answer(&mut self, j: usize, a: Answer) {
        self.answers[j] += 1;
        self.jobs[j] = At::Answered(a);
    }

    fn queued(&self) -> Vec<usize> {
        (0..self.jobs.len()).filter(|&j| self.jobs[j] == At::Queue).collect()
    }

    /// Answer every job the group holds with `a`: its own and taken jobs that
    /// are still live (the parked reply slots) and the inbox's.
    fn fail_group(&mut self, live: Answer, inbox: Answer) {
        for j in 0..self.jobs.len() {
            if self.jobs[j] == At::Live {
                self.answer(j, live);
            }
        }
        for j in std::mem::take(&mut self.inbox) {
            self.answer(j, inbox);
        }
    }
}

fn successors(s: &State, b: Bounds) -> Vec<(&'static str, State)> {
    let mut out = Vec::new();
    let in_open_groups = matches!(s.lane, Lane::Building | Lane::Open | Lane::Returned { .. } | Lane::DoneSent { .. });

    // Dispatcher: move queued jobs into the inbox, oldest first, up to the
    // space it has. It may also not refill at all (another instance waits).
    let space = (s.room + b.lookahead).saturating_sub(s.inbox.len());
    if in_open_groups && s.open && space > 0 && !s.queued().is_empty() {
        let mut t = s.clone();
        for j in s.queued().into_iter().take(space) {
            t.jobs[j] = At::Inbox;
            t.inbox.push(j);
        }
        out.push(("Refill", t));
    }

    match s.lane {
        Lane::Building => {
            let mut t = s.clone();
            t.lane = Lane::Open;
            out.push(("Built", t));
            // A failed build closes the inbox and answers the group's jobs with
            // the build error; `lane_loop` then drains the inbox.
            let mut t = s.clone();
            t.open = false;
            for j in 0..t.jobs.len() {
                if t.jobs[j] == At::Live {
                    t.answer(j, Answer::BuildFailed);
                }
            }
            t.lane = Lane::Returned { failed: true };
            out.push(("BuildFails", t));
        }
        Lane::Open if s.closed_batch => {
            // Close, take whatever slipped in, run the whole batch.
            let mut t = s.clone();
            t.open = false;
            t.taken += t.inbox.len();
            for j in std::mem::take(&mut t.inbox) {
                t.jobs[j] = At::Live;
            }
            for j in 0..t.jobs.len() {
                if t.jobs[j] == At::Live {
                    t.answer(j, Answer::Ok);
                }
            }
            t.lane = Lane::Returned { failed: false };
            out.push(("CloseBatch", t));
        }
        Lane::Open => {
            let free = b.limit - s.live();
            let n = free.min(s.inbox.len());
            if n > 0 {
                let mut t = s.clone();
                for j in t.inbox.drain(..n).collect::<Vec<_>>() {
                    t.jobs[j] = At::Live;
                }
                t.taken += n;
                t.room = t.room.saturating_sub(n);
                out.push(("Take", t));
            }
            if let Some(&j) = s.inbox.first() {
                let mut t = s.clone();
                t.inbox.retain(|&x| x != j);
                t.taken += 1;
                t.answer(j, Answer::Refused);
                out.push(("Screen", t));
            }
            for j in 0..s.jobs.len() {
                if s.jobs[j] == At::Live {
                    let mut t = s.clone();
                    t.answer(j, Answer::Ok);
                    t.room += 1;
                    out.push(("Finish", t));
                }
            }
            if s.live() == 0 && s.inbox.is_empty() {
                let mut t = s.clone();
                t.open = false;
                t.lane = Lane::Returned { failed: false };
                out.push(("Close", t));
            }
        }
        _ => {}
    }

    // A panic can strike whenever model code runs.
    if matches!(s.lane, Lane::Building | Lane::Open) {
        let mut t = s.clone();
        t.open = false;
        t.fail_group(Answer::Panicked, Answer::Panicked);
        t.lane = Lane::Returned { failed: true };
        out.push(("Panic", t));
    }

    if let Lane::Returned { failed } = s.lane {
        // `lane_loop`: close and drain the inbox, then send `Done`.
        let mut t = s.clone();
        t.open = false;
        for j in std::mem::take(&mut t.inbox) {
            t.answer(j, Answer::GroupEnded);
        }
        t.lane = Lane::DoneSent { failed };
        out.push(("LaneEnd", t));
    }
    if let Lane::DoneSent { failed } = s.lane {
        let mut t = s.clone();
        t.lane = Lane::Gone { failed };
        out.push(("Forget", t));
    }
    if let Lane::Gone { .. } = s.lane {
        // The queue is assigned to a fresh group; its own runs are this
        // model's initial state again, so they are answered here as served.
        if let Some(&j) = s.queued().first() {
            let mut t = s.clone();
            t.answer(j, Answer::Ok);
            out.push(("NextGroup", t));
        }
    }
    out
}

fn check_invariants(s: &State, b: Bounds) -> Result<(), String> {
    if let Some(j) = s.answers.iter().position(|&n| n > 1) {
        return Err(format!("job {j} answered {} times", s.answers[j]));
    }
    if matches!(s.lane, Lane::Open) && s.room + s.live() != b.limit {
        return Err(format!("room {} + live {} != limit {}", s.room, s.live(), b.limit));
    }
    for (j, at) in s.jobs.iter().enumerate() {
        let in_inbox = s.inbox.contains(&j);
        if in_inbox != (*at == At::Inbox) {
            return Err(format!("job {j} is {at:?} but in_inbox={in_inbox}"));
        }
    }
    if !s.open && matches!(s.lane, Lane::Open) && !s.inbox.is_empty() && !s.closed_batch {
        return Err("a closed inbox still holds jobs while the group runs".into());
    }
    Ok(())
}

fn check_terminal(s: &State) -> Result<(), String> {
    for (j, at) in s.jobs.iter().enumerate() {
        let At::Answered(a) = at else { return Err(format!("job {j} never answered: {at:?}")) };
        let failed = matches!(s.lane, Lane::Gone { failed: true });
        if *a == Answer::GroupEnded && !failed {
            return Err(format!("job {j} errored as late by a healthy group"));
        }
    }
    if !matches!(s.lane, Lane::Gone { .. }) {
        return Err(format!("quiesced before the group was forgotten: {:?}", s.lane));
    }
    Ok(())
}

/// Breadth-first over every reachable state; returns the number visited, or
/// the trace to the first violation.
fn explore(b: Bounds, closed_batch: bool) -> Result<usize, String> {
    let mut jobs = vec![At::Queue; b.jobs];
    for j in jobs.iter_mut().take(b.initial) {
        *j = At::Live;
    }
    let init = State { jobs, answers: vec![0; b.jobs], open: true, room: b.limit - b.initial, inbox: Vec::new(), taken: 0, lane: Lane::Building, closed_batch };
    let mut seen: HashSet<State> = HashSet::new();
    let mut parent: std::collections::HashMap<State, (State, &'static str)> = std::collections::HashMap::new();
    let mut frontier = VecDeque::from([init.clone()]);
    seen.insert(init);
    let trace = |parent: &std::collections::HashMap<State, (State, &'static str)>, mut s: State| {
        let mut steps = Vec::new();
        while let Some((p, a)) = parent.get(&s) {
            steps.push(*a);
            s = p.clone();
        }
        steps.reverse();
        steps.join(" -> ")
    };
    while let Some(s) = frontier.pop_front() {
        check_invariants(&s, b).map_err(|e| format!("{e}\n  {b:?} closed_batch={closed_batch}\n  trace: {}", trace(&parent, s.clone())))?;
        let next = successors(&s, b);
        if next.is_empty() {
            check_terminal(&s).map_err(|e| format!("{e}\n  {b:?} closed_batch={closed_batch}\n  state: {s:?}\n  trace: {}", trace(&parent, s.clone())))?;
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

#[test]
fn every_job_is_answered_once_and_only_failures_error_late_jobs() {
    let mut states = 0;
    for jobs in 1..=4 {
        for limit in 1..=3 {
            for initial in 1..=limit.min(jobs) {
                for lookahead in 0..=2 {
                    for closed_batch in [false, true] {
                        states += explore(Bounds { jobs, limit, lookahead, initial }, closed_batch).unwrap_or_else(|e| panic!("{e}"));
                    }
                }
            }
        }
    }
    assert!(states > 1000, "the bounds explored too little: {states} states");
}
