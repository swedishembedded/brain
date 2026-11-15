// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for isolating the one failing card
// of a model sharded across several, for its clients. If your team needs
// expertise in formal verification of fault attribution then you can procure
// our services by sending an email to info@swedishembedded.com.

//! Exhaustive model check of fault attribution for a model hosted across two
//! cards by one worker: model code that reports a fault on one card
//! (`report_device_fault`), the worker sending the reports as `fault` frames
//! before its next frame (`Host::send`), the parent collecting them
//! (`Blamed`, `JobsTalk`/`FaultTalk`) and recording faults when the request
//! ends (`Health::exchange_on`, `Blamed::record`), admission of a request
//! only while every card admits work, and a worker that may die or wedge.
//!
//! The instance serves up to `jobs` requests, one at a time; a lost instance
//! is rebuilt in a new worker, and a quarantined card is readmitted when its
//! probe passes. In every state:
//!
//! * no request is sent while either card is quarantined;
//! * a request is sent only to a worker that has not lost its instance;
//!
//! and once a request has ended:
//!
//! * a card the worker sent a fault for is quarantined;
//! * when the worker died without sending one, every card is quarantined;
//! * otherwise no card is quarantined that the worker did not blame;
//!
//! and in every state with no successor no frame is left unread.
//!
//! A deliberate bug in the model must make it fail: see
//! `every_mutation_of_fault_attribution_is_caught`.

use std::collections::{HashMap, VecDeque};

const CARDS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Frame {
    Fault(usize),
    Progress,
    Answer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Work {
    Idle,
    /// The worker runs the request in flight.
    Running,
    /// The worker answered it.
    Answered,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct State {
    jobs_left: u8,
    in_flight: bool,
    work: Work,
    alive: bool,
    wedged: bool,
    /// Faults model code reported and the worker has not sent.
    pending: [bool; CARDS],
    /// Faults the worker sent for the request in flight.
    sent: [bool; CARDS],
    /// The request in flight sent its one progress frame.
    progressed: bool,
    to_parent: VecDeque<Frame>,
    /// Faults the parent read for the request in flight.
    blamed: [bool; CARDS],
    quarantined: [bool; CARDS],
    /// The instance is lost: no request goes to its worker again.
    lost: bool,
    reports_left: u8,
    faults_left: u8,
    broken: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mutation {
    None,
    /// A death nothing was reported for quarantines only the instance's first card.
    BlameFirstOnly,
    /// The parent ignores fault frames.
    IgnoreFaultFrames,
    /// A fault reported in a request that was answered quarantines nothing.
    NoRecordOnAnswer,
    /// Admission asks only the first card.
    AdmitIfFirstAdmits,
    /// The worker sends its reports after the answer, not before.
    ReportsAfterAnswer,
}

/// The worker writes one frame: its pending reports first.
fn write(t: &mut State, frame: Frame, m: Mutation) {
    let reports: Vec<usize> = (0..CARDS).filter(|&c| t.pending[c]).collect();
    if m == Mutation::ReportsAfterAnswer && frame == Frame::Answer {
        t.to_parent.push_back(frame);
    }
    for c in reports {
        t.pending[c] = false;
        t.sent[c] = true;
        t.to_parent.push_back(Frame::Fault(c));
    }
    if !(m == Mutation::ReportsAfterAnswer && frame == Frame::Answer) {
        t.to_parent.push_back(frame);
    }
}

/// The request in flight ends; `died` when the worker is gone without an answer.
fn end(t: &mut State, died: bool, m: Mutation) {
    let named: Vec<usize> = (0..CARDS).filter(|&c| t.blamed[c]).collect();
    if died {
        let cards: Vec<usize> = if !named.is_empty() {
            named.clone()
        } else if m == Mutation::BlameFirstOnly {
            vec![0]
        } else {
            (0..CARDS).collect()
        };
        for c in cards {
            t.quarantined[c] = true;
        }
        t.lost = true;
    } else if !named.is_empty() && m != Mutation::NoRecordOnAnswer {
        for &c in &named {
            t.quarantined[c] = true;
        }
        t.lost = true;
    }
    // What attribution promises once the request is over.
    for c in 0..CARDS {
        if t.sent[c] && !t.quarantined[c] {
            t.broken = Some(format!("gpu{c} was blamed by its worker and is not quarantined"));
        }
    }
    if died && !t.sent.iter().any(|&s| s) && !t.quarantined.iter().all(|&q| q) {
        t.broken = Some("a worker died without blaming a card and not every card it held is quarantined".into());
    }
    if t.sent.iter().any(|&s| s) {
        for c in 0..CARDS {
            if t.quarantined[c] && !t.sent[c] {
                t.broken = Some(format!("gpu{c} is quarantined though its worker blamed only another card"));
            }
        }
    }
    t.in_flight = false;
    t.blamed = [false; CARDS];
    t.sent = [false; CARDS];
}

fn successors(s: &State, m: Mutation) -> Vec<(String, State)> {
    let mut out = Vec::new();
    // The lane: a request, admitted only while the cards admit work.
    if !s.in_flight && s.jobs_left > 0 && !s.lost {
        let admits = if m == Mutation::AdmitIfFirstAdmits { !s.quarantined[0] } else { s.quarantined.iter().all(|&q| !q) };
        if admits {
            let mut t = s.clone();
            if t.quarantined.iter().any(|&q| q) {
                t.broken = Some("a request was sent while a card of its instance was quarantined".into());
            }
            t.jobs_left -= 1;
            t.in_flight = true;
            t.progressed = false;
            t.work = Work::Running;
            out.push(("Send".to_string(), t));
        }
    }
    // A lost instance is rebuilt in a new worker on the same cards, and a quarantined card is readmitted once its
    // probe passes.
    if s.lost && !s.in_flight && s.jobs_left > 0 {
        let mut t = s.clone();
        t.lost = false;
        t.alive = true;
        t.wedged = false;
        t.work = Work::Idle;
        t.pending = [false; CARDS];
        t.to_parent.clear();
        out.push(("Rebuild".to_string(), t));
    }
    for c in 0..CARDS {
        if s.quarantined[c] && !s.in_flight {
            let mut t = s.clone();
            t.quarantined[c] = false;
            out.push((format!("Readmit(gpu{c})"), t));
        }
    }
    // The lane reads what the worker sent.
    if s.in_flight {
        if let Some(&frame) = s.to_parent.front() {
            let mut t = s.clone();
            t.to_parent.pop_front();
            match frame {
                Frame::Fault(c) => {
                    if m != Mutation::IgnoreFaultFrames {
                        t.blamed[c] = true;
                    }
                }
                Frame::Progress => {}
                Frame::Answer => end(&mut t, false, m),
            }
            out.push((format!("Read({frame:?})"), t));
        } else if !s.alive || s.wedged {
            // End of stream, or the deadline of a silent worker: it is killed and the request fails.
            let mut t = s.clone();
            t.alive = false;
            end(&mut t, true, m);
            out.push(("Dead".to_string(), t));
        }
    } else if let Some(&frame) = s.to_parent.front() {
        // A frame about a request already over: discarded.
        let mut t = s.clone();
        t.to_parent.pop_front();
        out.push((format!("Discard({frame:?})"), t));
    }
    if s.alive && !s.wedged && s.work == Work::Running {
        for c in 0..CARDS {
            if s.reports_left > 0 && !s.pending[c] {
                let mut t = s.clone();
                t.reports_left -= 1;
                t.pending[c] = true;
                out.push((format!("Report(gpu{c})"), t));
            }
        }
        if !s.progressed {
            let mut t = s.clone();
            write(&mut t, Frame::Progress, m);
            t.progressed = true;
            out.push(("Progress".to_string(), t));
        }
        let mut t = s.clone();
        write(&mut t, Frame::Answer, m);
        t.work = Work::Answered;
        out.push(("Answer".to_string(), t));
    }
    if s.work == Work::Answered && !s.in_flight {
        let mut t = s.clone();
        t.work = Work::Idle;
        out.push(("Idle".to_string(), t));
    }
    if s.alive && s.faults_left > 0 && s.in_flight {
        let mut t = s.clone();
        t.faults_left -= 1;
        t.alive = false;
        t.work = Work::Idle;
        out.push(("Crash".to_string(), t));
        if !s.wedged {
            let mut t = s.clone();
            t.faults_left -= 1;
            t.wedged = true;
            out.push(("Wedge".to_string(), t));
        }
    }
    out
}

fn explore(jobs: u8, reports: u8, faults: u8, m: Mutation) -> Result<usize, String> {
    let init = State {
        jobs_left: jobs,
        in_flight: false,
        work: Work::Idle,
        alive: true,
        wedged: false,
        pending: [false; CARDS],
        sent: [false; CARDS],
        progressed: false,
        to_parent: VecDeque::new(),
        blamed: [false; CARDS],
        quarantined: [false; CARDS],
        lost: false,
        reports_left: reports,
        faults_left: faults,
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
            return Err(format!("{why}\n  jobs {jobs} reports {reports} faults {faults} {m:?}\n  trace: {}", trace(&parent, at)));
        }
        let next = successors(&s, m);
        if next.is_empty() && s.alive && !s.to_parent.is_empty() {
            return Err(format!("frames left unread: {:?}\n  trace: {}", s.to_parent, trace(&parent, at)));
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
        for (jobs, reports) in [(1, 0), (1, 1), (2, 1), (2, 2)] {
            states += explore(jobs, reports, faults, m)?;
        }
    }
    Ok(states)
}

#[test]
fn a_multi_device_workers_faults_quarantine_the_cards_it_blamed_or_every_card() {
    let states = explore_all(Mutation::None).unwrap_or_else(|e| panic!("{e}"));
    eprintln!("{states} states explored");
    assert!(states > 1_000, "the bounds explored too little: {states} states");
}

#[test]
fn every_mutation_of_fault_attribution_is_caught() {
    for m in [Mutation::BlameFirstOnly, Mutation::IgnoreFaultFrames, Mutation::NoRecordOnAnswer, Mutation::AdmitIfFirstAdmits, Mutation::ReportsAfterAnswer] {
        let why = explore_all(m).err().unwrap_or_else(|| panic!("{m:?} broke fault attribution and the model did not notice"));
        eprintln!("{m:?} caught: {}", why.lines().next().unwrap_or(""));
    }
}
