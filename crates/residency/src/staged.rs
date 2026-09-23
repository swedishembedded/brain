// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Trying a new version of a model without serving it yet.
//!
//! [`Executor::evict`](crate::executor::Executor::evict) is a one-way swap:
//! bump the version a key names, drop the resident instance, and the next
//! request rebuilds against the new file. It refuses to interrupt a running
//! job, so nothing is ever torn mid-request - but once the swap has happened
//! there is no way back except another swap, and by then the bad version has
//! already answered live traffic.
//!
//! A continual learner needs the other half. Its gate decides on frozen
//! probes, before publication, and that decision can still be wrong: a gate
//! promotes on evidence, and evidence is not proof. So a version gets to be
//! a CANDIDATE first - resident, addressable, answerable - while the version
//! that is actually served carries on untouched.
//!
//! ```text
//!   stage(v)      the candidate becomes resident; served is untouched
//!   validate      run whatever the caller trusts, against the candidate
//!   commit        the candidate becomes served
//!   rollback      the candidate is discarded; served is untouched
//! ```
//!
//! **The property that matters is the one on the failure path.** A rollback
//! must leave the served version answering exactly as it did before the
//! stage - not equivalently, not close enough, identically. A staging
//! mechanism that can perturb what it was supposed to protect has moved the
//! risk rather than removed it. That is why `served` is never written except
//! by [`StagedSlot::commit`], and why commit is the only method that can
//! change it.
//!
//! This module is the state machine alone: which version is served, which is
//! on trial, and what each transition is allowed to do. It holds no model
//! and no device, so the rules can be tested exhaustively and in
//! microseconds. Driving an [`Executor`](crate::executor::Executor) with it
//! is the caller's, and that binding cannot invent a transition this refuses.

use std::fmt;

/// Names one build of a model: an adapter id, a checkpoint version, whatever
/// the caller's `InstanceKey` is derived from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Version(pub String);

impl Version {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Why a transition was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StageError {
    /// Something is already on trial. Two candidates at once would make
    /// "commit" ambiguous and "rollback" destructive.
    AlreadyStaged { staged: Version, offered: Version },
    /// There is nothing on trial to commit or roll back.
    NothingStaged,
    /// The candidate is the version already being served. Staging it would
    /// spend a slot to change nothing, and committing it would report a swap
    /// that did not happen.
    AlreadyServed(Version),
}

impl fmt::Display for StageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StageError::AlreadyStaged { staged, offered } => {
                write!(f, "{staged} is already on trial, so {offered} cannot be staged as well - commit or roll back first")
            }
            StageError::NothingStaged => f.write_str("nothing is on trial"),
            StageError::AlreadyServed(v) => write!(f, "{v} is already what is served"),
        }
    }
}

impl std::error::Error for StageError {}

/// What a completed transition did, for the caller to act on and a ledger to
/// record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Swap {
    /// What was served before.
    pub from: Version,
    /// What is served now. Equal to `from` after a rollback.
    pub to: Version,
    /// The version that was on trial.
    pub candidate: Version,
}

impl Swap {
    /// Whether what is served actually changed. False after every rollback,
    /// and the one thing a caller must check before announcing a swap.
    pub fn changed(&self) -> bool {
        self.from != self.to
    }
}

/// Which version is served, and which is on trial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StagedSlot {
    served: Version,
    staged: Option<Version>,
}

impl StagedSlot {
    pub fn new(served: Version) -> StagedSlot {
        StagedSlot { served, staged: None }
    }

    /// What answers requests right now.
    pub fn served(&self) -> &Version {
        &self.served
    }

    /// What is on trial, if anything.
    pub fn staged(&self) -> Option<&Version> {
        self.staged.as_ref()
    }

    pub fn is_staging(&self) -> bool {
        self.staged.is_some()
    }

    /// Put `candidate` on trial. Does not change what is served.
    pub fn stage(&mut self, candidate: Version) -> Result<(), StageError> {
        if let Some(staged) = &self.staged {
            return Err(StageError::AlreadyStaged { staged: staged.clone(), offered: candidate });
        }
        if candidate == self.served {
            return Err(StageError::AlreadyServed(candidate));
        }
        self.staged = Some(candidate);
        Ok(())
    }

    /// Accept the candidate: it becomes what is served.
    pub fn commit(&mut self) -> Result<Swap, StageError> {
        let candidate = self.staged.take().ok_or(StageError::NothingStaged)?;
        let from = std::mem::replace(&mut self.served, candidate.clone());
        Ok(Swap { from, to: candidate.clone(), candidate })
    }

    /// Reject the candidate. What is served is not touched.
    pub fn rollback(&mut self) -> Result<Swap, StageError> {
        let candidate = self.staged.take().ok_or(StageError::NothingStaged)?;
        Ok(Swap { from: self.served.clone(), to: self.served.clone(), candidate })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str) -> Version {
        Version(s.to_string())
    }

    fn slot() -> StagedSlot {
        StagedSlot::new(v("adapter-1"))
    }

    /// The whole point: putting a candidate on trial must not change what
    /// answers requests.
    #[test]
    fn staging_does_not_change_what_is_served() {
        let mut s = slot();
        s.stage(v("adapter-2")).expect("stage");
        assert_eq!(s.served(), &v("adapter-1"), "the served version moved during a stage");
        assert_eq!(s.staged(), Some(&v("adapter-2")));
        assert!(s.is_staging());
    }

    /// The failure path, and the reason this module exists. A rollback must
    /// leave the served version exactly as it was - a staging mechanism that
    /// can perturb what it was protecting has moved the risk, not removed
    /// it.
    #[test]
    fn a_rollback_leaves_the_served_version_untouched() {
        let mut s = slot();
        let before = s.clone();
        s.stage(v("adapter-2")).expect("stage");
        let swap = s.rollback().expect("rollback");

        assert_eq!(s, before, "the slot must be indistinguishable from before the stage");
        assert_eq!(swap.candidate, v("adapter-2"), "the rollback must still name what it rejected");
        assert!(!swap.changed(), "a rollback must not report a swap");
        assert_eq!(swap.from, swap.to);
        assert!(!s.is_staging());
    }

    /// The success path, as the other half of that pair.
    #[test]
    fn a_commit_makes_the_candidate_what_is_served() {
        let mut s = slot();
        s.stage(v("adapter-2")).expect("stage");
        let swap = s.commit().expect("commit");

        assert_eq!(s.served(), &v("adapter-2"));
        assert_eq!(s.staged(), None, "a committed candidate is no longer on trial");
        assert!(swap.changed(), "a commit must report a swap");
        assert_eq!((swap.from, swap.to), (v("adapter-1"), v("adapter-2")));
    }

    /// Two candidates at once would make `commit` ambiguous and `rollback`
    /// destructive, so the second is refused by name rather than silently
    /// replacing the first.
    #[test]
    fn a_second_candidate_is_refused_while_one_is_on_trial() {
        let mut s = slot();
        s.stage(v("adapter-2")).expect("stage");
        match s.stage(v("adapter-3")) {
            Err(StageError::AlreadyStaged { staged, offered }) => {
                assert_eq!(staged, v("adapter-2"));
                assert_eq!(offered, v("adapter-3"));
            }
            other => panic!("expected AlreadyStaged, got {other:?}"),
        }
        assert_eq!(s.staged(), Some(&v("adapter-2")), "the refused offer must not have displaced the one on trial");
    }

    /// Committing or rolling back nothing is a caller error, not a no-op: a
    /// silent success would let a caller believe a swap happened.
    #[test]
    fn committing_or_rolling_back_nothing_is_refused() {
        let mut s = slot();
        assert_eq!(s.commit(), Err(StageError::NothingStaged));
        assert_eq!(s.rollback(), Err(StageError::NothingStaged));
        assert_eq!(s.served(), &v("adapter-1"), "a refused transition must not have touched the slot");
    }

    /// Staging what is already served would spend a slot to change nothing,
    /// and committing it would report a swap that did not happen.
    #[test]
    fn staging_the_version_already_served_is_refused() {
        let mut s = slot();
        assert_eq!(s.stage(v("adapter-1")), Err(StageError::AlreadyServed(v("adapter-1"))));
        assert!(!s.is_staging());
    }

    /// A reader promotes repeatedly over a long run, so the slot has to come
    /// back to a clean state after every outcome, whichever it was.
    #[test]
    fn the_slot_is_reusable_after_both_outcomes() {
        let mut s = slot();
        for i in 2..6 {
            s.stage(v(&format!("adapter-{i}"))).expect("stage");
            if i % 2 == 0 {
                assert!(s.commit().expect("commit").changed());
            } else {
                assert!(!s.rollback().expect("rollback").changed());
            }
            assert!(!s.is_staging(), "round {i} left the slot occupied");
        }
        // Committed 2 and 4, rolled back 3 and 5.
        assert_eq!(s.served(), &v("adapter-4"));
    }

    /// `served` is written by exactly one method. Anything else able to move
    /// it would make the rollback guarantee a convention rather than a
    /// property.
    #[test]
    fn only_commit_can_change_what_is_served() {
        let mut s = slot();
        let served = s.served().clone();
        let _ = s.stage(v("adapter-2"));
        assert_eq!(s.served(), &served);
        let _ = s.stage(v("adapter-3"));
        assert_eq!(s.served(), &served);
        let _ = s.rollback();
        assert_eq!(s.served(), &served);
        let _ = s.rollback();
        assert_eq!(s.served(), &served);
        s.stage(v("adapter-9")).expect("stage");
        s.commit().expect("commit");
        assert_ne!(s.served(), &served, "commit is the one that must move it");
    }
}

/// Dropping a resident instance so the next request rebuilds.
///
/// A trait rather than a direct [`crate::executor::Executor`] reference so
/// the transitions can be tested for what they EVICT without a device, a
/// model or a dispatcher: the property that matters here is which instance
/// goes, and getting that backwards is the whole hazard.
pub trait Evictor {
    /// Drop the instance named by `key`. `false` if it was not resident, or
    /// if a job is running against it.
    fn evict(&self, key: &InstanceKey) -> bool;
}

impl Evictor for crate::executor::Executor {
    fn evict(&self, key: &InstanceKey) -> bool {
        crate::executor::Executor::evict(self, key.clone())
    }
}

use crate::InstanceKey;

/// A [`StagedSlot`] driven against live residency.
///
/// Two versions are two [`InstanceKey`]s, so both can be resident at once
/// and a request can be addressed to either. That is the part the existing
/// hot-swap could not do: it mutates the version a single key rebuilds from,
/// so the old one is gone the moment the new one exists and there is nothing
/// left to compare against or fall back to.
///
/// **A transition evicts at most one instance, and never the wrong one.** A
/// commit drops what was being served, because the candidate has replaced
/// it. A rollback drops the candidate, because it was refused. A REFUSED
/// transition drops nothing at all - if staging a second candidate could
/// evict the first, the refusal would be more destructive than the action.
pub struct StagedResident<E: Evictor> {
    slot: StagedSlot,
    model: String,
    evictor: E,
}

impl<E: Evictor> StagedResident<E> {
    pub fn new(model: impl Into<String>, served: Version, evictor: E) -> StagedResident<E> {
        StagedResident { slot: StagedSlot::new(served), model: model.into(), evictor }
    }

    /// The key a version is resident under. The version IS the config part,
    /// which is what lets two of them coexist.
    pub fn key_for(&self, v: &Version) -> InstanceKey {
        InstanceKey::new(self.model.as_str(), v.as_str())
    }

    pub fn served(&self) -> &Version {
        self.slot.served()
    }

    pub fn staged(&self) -> Option<&Version> {
        self.slot.staged()
    }

    /// The key requests should currently be addressed to.
    pub fn serving_key(&self) -> InstanceKey {
        self.key_for(self.slot.served())
    }

    /// The key to validate against, once something is on trial.
    pub fn candidate_key(&self) -> Option<InstanceKey> {
        self.slot.staged().map(|v| self.key_for(v))
    }

    /// Put `candidate` on trial. Evicts nothing: the candidate becomes
    /// resident when something is first addressed to its key, and what is
    /// served is untouched either way.
    pub fn stage(&mut self, candidate: Version) -> Result<(), StageError> {
        self.slot.stage(candidate)
    }

    /// Accept the candidate. It becomes what is served, and the version it
    /// replaced is evicted, since nothing will address it again.
    pub fn commit(&mut self) -> Result<Swap, StageError> {
        let swap = self.slot.commit()?;
        self.evictor.evict(&self.key_for(&swap.from));
        Ok(swap)
    }

    /// Reject the candidate. It is evicted; what is served is untouched, and
    /// in particular is NOT evicted - it has been answering throughout and
    /// has no reason to rebuild.
    pub fn rollback(&mut self) -> Result<Swap, StageError> {
        let swap = self.slot.rollback()?;
        self.evictor.evict(&self.key_for(&swap.candidate));
        Ok(swap)
    }
}

#[cfg(test)]
mod binding_tests {
    use super::*;
    use std::sync::Mutex;

    /// Records what it was asked to drop, so a transition can be checked for
    /// which instance it evicted rather than merely that it evicted one.
    #[derive(Default)]
    struct Recorder(Mutex<Vec<String>>);

    impl Evictor for &Recorder {
        fn evict(&self, key: &InstanceKey) -> bool {
            self.0.lock().expect("not poisoned").push(key.config.clone());
            true
        }
    }

    fn v(s: &str) -> Version {
        Version(s.to_string())
    }

    fn evicted(r: &Recorder) -> Vec<String> {
        r.0.lock().expect("not poisoned").clone()
    }

    /// Two versions are two keys, which is the property the existing
    /// hot-swap lacks: it rebuilds one key from a changed file, so the old
    /// version stops existing the moment the new one does.
    #[test]
    fn a_candidate_is_addressable_alongside_what_is_served() {
        let rec = Recorder::default();
        let mut s = StagedResident::new("qwen", v("adapter-1"), &rec);
        assert_eq!(s.candidate_key(), None);

        s.stage(v("adapter-2")).expect("stage");
        let serving = s.serving_key();
        let candidate = s.candidate_key().expect("on trial");
        assert_ne!(serving, candidate, "the two versions must be separately addressable");
        assert_eq!(serving.config, "adapter-1");
        assert_eq!(candidate.config, "adapter-2");
        assert!(evicted(&rec).is_empty(), "staging must not evict anything");
    }

    /// A commit drops what it replaced, and only that.
    #[test]
    fn a_commit_evicts_the_version_it_replaced() {
        let rec = Recorder::default();
        let mut s = StagedResident::new("qwen", v("adapter-1"), &rec);
        s.stage(v("adapter-2")).expect("stage");
        let swap = s.commit().expect("commit");

        assert!(swap.changed());
        assert_eq!(evicted(&rec), vec!["adapter-1".to_string()], "a commit must drop the OLD version, not the new one");
        assert_eq!(s.serving_key().config, "adapter-2");
    }

    /// V17's half that matters: a rollback drops the candidate and leaves
    /// what is served resident and answering. Evicting the served version
    /// here would make a refused candidate cost a rebuild of the one that
    /// was working.
    #[test]
    fn a_rollback_evicts_the_candidate_and_never_what_is_served() {
        let rec = Recorder::default();
        let mut s = StagedResident::new("qwen", v("adapter-1"), &rec);
        let serving_before = s.serving_key();
        s.stage(v("adapter-2")).expect("stage");
        let swap = s.rollback().expect("rollback");

        assert!(!swap.changed());
        assert_eq!(evicted(&rec), vec!["adapter-2".to_string()], "a rollback must drop the CANDIDATE");
        assert_eq!(s.serving_key(), serving_before, "and must leave the served key exactly as it was");
        assert_eq!(s.candidate_key(), None);
    }

    /// A refusal must be the least destructive outcome, not the most: if a
    /// rejected second candidate could evict the first, refusing would cost
    /// more than accepting.
    #[test]
    fn a_refused_transition_evicts_nothing() {
        let rec = Recorder::default();
        let mut s = StagedResident::new("qwen", v("adapter-1"), &rec);
        assert!(s.commit().is_err());
        assert!(s.rollback().is_err());
        assert!(s.stage(v("adapter-1")).is_err());
        s.stage(v("adapter-2")).expect("stage");
        assert!(s.stage(v("adapter-3")).is_err());

        assert!(evicted(&rec).is_empty(), "refusals evicted {:?}", evicted(&rec));
        assert_eq!(s.staged(), Some(&v("adapter-2")), "and left the one on trial in place");
    }
}
