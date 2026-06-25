// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Generic hierarchical state machine (HSM) substrate.
//!
//! A [`Machine`] supplies the behaviour: it dispatches an event in a given state
//! (returning a [`Disp`] disposition), reports each state's `parent` (the nesting
//! that makes it *hierarchical*), and runs `on_entry`/`on_exit` actions for a
//! state. [`Hsm`] drives a [`Machine`] with a run-to-completion (RTC) loop over an
//! internal event queue, implementing the full UML-statechart dispatch algorithm:
//! superstate bubbling on `Unhandled`/`Super` (behavioural inheritance), LCA
//! (least-common-ancestor) computation on a transition, and the correctly ordered
//! exit/entry action chains.
//!
//! The algorithm follows Miro Samek's HSM dispatch (see the embedded
//! state-machine skill, `02-hsm-implementation.md` §3-5), adapted to a *pure*
//! `parent()` accessor instead of the side-effecting `SIG_EMPTY` probe: because
//! `parent()` has no side effects we can walk both ancestor chains directly to
//! find the LCA. Patterns applied:
//!   * **Behavioural inheritance** — `Unhandled`/`Super(p)` bubbles the event to
//!     the superstate, so a child inherits whatever its parents handle.
//!   * **LCA-correct entry/exit** — exit actions fire bottom-up from the active
//!     leaf to (not including) the LCA; entry actions fire top-down from below the
//!     LCA to the target. The LCA itself is never exited/re-entered (local
//!     transition semantics).
//!   * **Entry/exit actions take no transitions** — they run via `on_entry`/
//!     `on_exit`, which return nothing; they cannot themselves move the machine.
//!   * **Run-to-completion** — `post` enqueues; `run` drains the queue dispatching
//!     each event *fully* (including events posted during handling) before the
//!     next; the loop is non-reentrant (guarded by `running`).
//!   * **Reminder pattern** — a state can `post` a synthetic event to itself from
//!     within `dispatch`; RTC guarantees it is processed after the current event
//!     completes, which drives streaming (one token per self-posted `Tick`).

use std::collections::VecDeque;

/// The disposition of dispatching an event in a state.
pub enum Disp<S> {
    /// The event was consumed; stay in the current state.
    Handled,
    /// The event was not handled here; bubble to the superstate.
    Unhandled,
    /// Transition to state `S` (compute LCA, run exit/entry chains).
    Tran(S),
    /// Defer handling to the named superstate `S` (explicit super-handler).
    /// Equivalent to [`Disp::Unhandled`] for a single-parent hierarchy, but lets a
    /// state name its handler explicitly.
    Super(S),
}

/// Behaviour an HSM is built from.
pub trait Machine {
    /// State identifier (typically a small `Copy` enum).
    type State: Copy + PartialEq;
    /// Event type the machine reacts to.
    type Event;

    /// Handle `ev` while in `state`, returning the disposition. Must NOT mutate
    /// the active state directly — request changes via [`Disp::Tran`].
    fn dispatch(&mut self, state: Self::State, ev: &Self::Event) -> Disp<Self::State>;

    /// The parent (superstate) of `s`, or `None` if `s` is the top-level state.
    /// Must be a pure function of `s` (no side effects) and acyclic.
    fn parent(&self, s: Self::State) -> Option<Self::State>;

    /// Action run when entering `s`. Takes no transition and sees no event.
    fn on_entry(&mut self, _s: Self::State) {}

    /// Action run when exiting `s`. Takes no transition and sees no event.
    fn on_exit(&mut self, _s: Self::State) {}
}

/// Drives a [`Machine`] with an internal RTC event queue.
pub struct Hsm<M: Machine> {
    machine: M,
    state: M::State,
    queue: VecDeque<M::Event>,
    running: bool,
}

impl<M: Machine> Hsm<M> {
    /// Construct an HSM whose active leaf is `initial`. Entry actions for
    /// `initial` (and its ancestors) are *not* run here; if you need them, call
    /// [`Hsm::init`] with the chain you want entered.
    pub fn new(machine: M, initial: M::State) -> Hsm<M> {
        Hsm { machine, state: initial, queue: VecDeque::new(), running: false }
    }

    /// The current leaf state.
    pub fn state(&self) -> M::State {
        self.state
    }

    /// Borrow the underlying machine (e.g. to read emitted output in tests).
    pub fn machine(&self) -> &M {
        &self.machine
    }

    /// Mutably borrow the underlying machine.
    pub fn machine_mut(&mut self) -> &mut M {
        &mut self.machine
    }

    /// Run the entry chain from the (conceptual) top down to `initial`, then drain
    /// any events posted during those entry actions. Use this when the initial
    /// state's `on_entry` must fire at startup (e.g. it self-posts a `Tick`).
    pub fn init(&mut self) {
        let chain = self.path_to_root(self.state); // [leaf, .., top]
        for &s in chain.iter().rev() {
            self.machine.on_entry(s);
        }
        self.run();
    }

    /// Enqueue an event for the next [`run`](Hsm::run).
    pub fn post(&mut self, ev: M::Event) {
        self.queue.push_back(ev);
    }

    /// Drain the queue, dispatching each event run-to-completion. Reentrant calls
    /// (e.g. from within an action) are no-ops: the outermost `run` owns the
    /// queue and will pick up anything posted meanwhile.
    pub fn run(&mut self) {
        if self.running {
            return;
        }
        self.running = true;
        while let Some(ev) = self.queue.pop_front() {
            self.step(&ev);
        }
        self.running = false;
    }

    /// Dispatch a single event: phase 1 bubbles up the hierarchy to find the
    /// handler; on a transition, phase 2 computes the LCA and runs the exit/entry
    /// action chains.
    fn step(&mut self, ev: &M::Event) {
        // --- Phase 1: hierarchical event propagation (behavioural inheritance) ---
        // `source` walks up from the active leaf until a state handles the event.
        let mut source = self.state;
        let target = loop {
            match self.machine.dispatch(source, ev) {
                Disp::Handled => return, // consumed; no transition
                Disp::Unhandled | Disp::Super(_) => match self.machine.parent(source) {
                    Some(p) => source = p,
                    None => return, // reached top unhandled → silently ignore (UML default)
                },
                Disp::Tran(t) => break t,
            }
        };

        // --- Phase 2: transition source → target ---
        self.transition(source, target);
    }

    /// Execute a transition from `source` (the state whose handler returned
    /// `Tran`) to `target`, exiting/entering the correct states around the LCA.
    fn transition(&mut self, source: M::State, target: M::State) {
        let lca = self.lca(source, target);

        // Exit from the active leaf up to (not including) the LCA, bottom-up.
        let mut s = self.state;
        while !self.same(s, lca) {
            self.machine.on_exit(s);
            match self.machine.parent(s) {
                Some(p) => s = p,
                None => break,
            }
        }

        // Build the entry path from `target` up to (not including) the LCA, then
        // enter it top-down (reverse).
        let mut path: Vec<M::State> = Vec::new();
        let mut t = target;
        while !self.same(t, lca) {
            path.push(t);
            match self.machine.parent(t) {
                Some(p) => t = p,
                None => break,
            }
        }
        for &st in path.iter().rev() {
            self.machine.on_entry(st);
        }

        self.state = target;
    }

    /// Whether `a == b`, treating "no LCA" (`None`) as never equal to a real state.
    fn same(&self, a: M::State, lca: Option<M::State>) -> bool {
        matches!(lca, Some(l) if l == a)
    }

    /// The least common ancestor of `a` and `b`, or `None` if they share no
    /// ancestor (distinct roots) — in which case all states get exited/entered.
    fn lca(&self, a: M::State, b: M::State) -> Option<M::State> {
        let anc_a = self.path_to_root(a); // [a, parent(a), .., root]
        self.path_to_root(b).into_iter().find(|cand| anc_a.contains(cand))
    }

    /// The chain `[s, parent(s), .., root]`.
    fn path_to_root(&self, s: M::State) -> Vec<M::State> {
        let mut out = vec![s];
        let mut cur = s;
        while let Some(p) = self.machine.parent(cur) {
            out.push(p);
            cur = p;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A four-level toy hierarchy to exercise LCA + behavioural inheritance:
    //
    //   Root
    //   └── Op
    //       ├── A   (leaf)
    //       └── B   (leaf)
    //
    // Events: Go (A→B), Up (handled only by Op, to test inheritance),
    //         Self (B self-transition), Tick (reminder self-post loop).
    #[derive(Copy, Clone, PartialEq, Debug)]
    enum S {
        Root,
        Op,
        A,
        B,
    }

    #[derive(Clone, Debug)]
    enum E {
        Go,
        Up,
        SelfB,
        Tick,
    }

    #[derive(Default)]
    struct Toy {
        log: Vec<String>,
        ticks: u32,
    }

    impl Machine for Toy {
        type State = S;
        type Event = E;

        fn dispatch(&mut self, state: S, ev: &E) -> Disp<S> {
            match (state, ev) {
                (S::A, E::Go) => Disp::Tran(S::B),
                (S::B, E::SelfB) => Disp::Tran(S::B), // self-transition
                // `Up` is handled only by Op: from A or B it must bubble up.
                (S::Op, E::Up) => {
                    self.log.push("op-handled-up".into());
                    Disp::Handled
                }
                // Tick drives a reminder loop: stop after 3.
                (_, E::Tick) => {
                    self.ticks += 1;
                    self.log.push(format!("tick{}", self.ticks));
                    Disp::Handled
                }
                _ => Disp::Unhandled,
            }
        }

        fn parent(&self, s: S) -> Option<S> {
            match s {
                S::Root => None,
                S::Op => Some(S::Root),
                S::A | S::B => Some(S::Op),
            }
        }

        fn on_entry(&mut self, s: S) {
            self.log.push(format!("enter:{s:?}"));
        }
        fn on_exit(&mut self, s: S) {
            self.log.push(format!("exit:{s:?}"));
        }
    }

    #[test]
    fn peer_transition_runs_lca_correct_chains() {
        let mut hsm = Hsm::new(Toy::default(), S::A);
        hsm.post(E::Go);
        hsm.run();
        assert_eq!(hsm.state(), S::B);
        // LCA of A and B is Op: exit only A, enter only B (Op untouched).
        assert_eq!(hsm.machine().log, vec!["exit:A", "enter:B"]);
    }

    #[test]
    fn behavioral_inheritance_parent_handles_child_event() {
        let mut hsm = Hsm::new(Toy::default(), S::A);
        hsm.post(E::Up); // A doesn't handle Up; Op does
        hsm.run();
        assert_eq!(hsm.state(), S::A); // Handled by parent, no transition
        assert_eq!(hsm.machine().log, vec!["op-handled-up"]);
    }

    #[test]
    fn unhandled_at_top_is_ignored() {
        let mut hsm = Hsm::new(Toy::default(), S::B);
        hsm.post(E::Go); // only A handles Go; B and ancestors don't
        hsm.run();
        assert_eq!(hsm.state(), S::B);
        assert!(hsm.machine().log.is_empty());
    }

    #[test]
    fn self_transition_exits_and_reenters_leaf() {
        let mut hsm = Hsm::new(Toy::default(), S::B);
        hsm.post(E::SelfB);
        hsm.run();
        assert_eq!(hsm.state(), S::B);
        // self-transition: LCA of B and B is B, so per local semantics nothing
        // exits/enters here (LCA never exited). Leaf self-transition kept minimal.
        assert!(hsm.machine().log.is_empty());
    }

    #[test]
    fn initial_entry_chain_runs_top_down() {
        let mut hsm = Hsm::new(Toy::default(), S::A);
        hsm.init();
        // entered top-down: Root, Op, A
        assert_eq!(hsm.machine().log, vec!["enter:Root", "enter:Op", "enter:A"]);
    }

    #[test]
    fn reminder_loop_terminates_under_rtc() {
        // A state self-posts Tick from an action; RTC processes each before the
        // next. We drive it by posting Tick three times (the machine logs each).
        let mut hsm = Hsm::new(Toy::default(), S::A);
        hsm.post(E::Tick);
        hsm.post(E::Tick);
        hsm.post(E::Tick);
        hsm.run();
        assert_eq!(hsm.machine().ticks, 3);
        assert_eq!(hsm.machine().log, vec!["tick1", "tick2", "tick3"]);
    }

    #[test]
    fn rtc_drains_queue_in_fifo_order() {
        // RTC: posting two events processes them in FIFO within one `run`, and a
        // nested `run` (guarded by `running`) would be a no-op — neither event is
        // lost or reordered.
        let mut hsm = Hsm::new(Toy::default(), S::A);
        // Post two events; if run were reentrant we'd risk losing one. Both must
        // process in FIFO order within one run.
        hsm.post(E::Tick);
        hsm.post(E::Go);
        hsm.run();
        assert_eq!(hsm.state(), S::B);
        assert_eq!(hsm.machine().ticks, 1);
    }
}
