// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A pool of adapters on disk, and which of them is active right now.
//!
//! Capacity that grows has to be capacity that can be put down again. This
//! module owns the bookkeeping for that and nothing else: which adapters
//! exist, which are in the working set, which have gone unaddressed long
//! enough to retire, and where a retired one went. None of that needs a
//! model, so none of it is here mixed with delta arithmetic - the working set
//! is MATERIALISED into a runtime correction elsewhere, by concatenating the
//! selected adapters on the rank axis, which `model::lora` already does.
//!
//! Three rules, each of which exists because the obvious alternative is
//! wrong.
//!
//! **An adapter already in the working set keeps the slot it is in.** Demand
//! comes back sorted, so the ORDER churns constantly while the SET barely
//! moves. Matching by identity rather than by position is what keeps the
//! number of loads equal to how much the set really changed, instead of
//! reloading almost everything every time two scores swap.
//!
//! **Retirement archives, it never deletes.** Rare is not dead. A capability
//! addressed once a quarter is exactly what a personal continual reader is
//! for, and deletion is the one action such a reader cannot undo. An
//! archived adapter keeps its file and can be resurrected byte for byte.
//!
//! **Liveness is staleness, never a contribution score.** An adapter chosen
//! constantly that contributes a little each time reads as dead on any
//! per-use score, while one nothing has wanted for months reads as alive. The
//! question that matters is whether anything still asks for it.
//!
//! ## Known gap this module does not close
//!
//! An adapter's Adam moments do not survive a round trip through its file.
//! `model::lora::Pair` owns its moments, but the save path writes only the
//! `.lora_a`/`.lora_b` tensors and `Pair::from_ab` leaves the moments empty
//! by design, because that path exists for a one-shot finetune that trains,
//! saves and folds. In a pool, eviction is routine: an adapter evicted and
//! re-admitted would restart its optimiser state every time, which is the
//! failure mode of handing one adapter's momentum history to whatever came
//! back in its place. Closing it means the POOL's own file carrying the
//! moments alongside the weights - a superset of the existing format, so
//! existing readers are unaffected - and that is the next milestone, not
//! this one.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Where live adapter files sit under the pool root.
const LIVE: &str = "live";
/// ...and where retired ones are kept. Retirement is a rename into here.
const ARCHIVE: &str = "archive";
const INDEX: &str = "index.json";

/// Stable identity of one adapter in the pool. Supplied by the caller that
/// trained it, so it can carry that adapter's own lineage.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AdapterId(pub String);

impl AdapterId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum State {
    /// In the pool and selectable.
    Live,
    /// Retired. Its file is kept; it is not selectable until resurrected.
    Archived,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoolEntry {
    pub id: AdapterId,
    pub state: State,
    /// Episode index when it joined the pool.
    pub born: u64,
    /// Episode index when it was last IN the working set. What staleness is
    /// measured from.
    pub last_seen: u64,
    /// Episode index when it last ENTERED the working set. What dwell
    /// protection is measured from, and deliberately not the same question as
    /// `last_seen`.
    pub admitted: u64,
    /// File name under the pool root, in whichever directory `state` implies.
    pub file: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PoolConfig {
    /// How many adapters may be active at once.
    pub resident: usize,
    /// How much more a candidate must be wanted before it displaces a
    /// resident. Without it the set churns on noise.
    pub margin: f64,
    /// Episodes a newly admitted adapter is safe from eviction, so it cannot
    /// be judged before it has had a chance to be useful.
    pub dwell: u64,
    /// Episodes unaddressed before retirement. Set wide: the cost of this
    /// trade is that a genuinely rare adapter is archived.
    pub survival: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig { resident: 8, margin: 0.10, dwell: 16, survival: 4096 }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PoolError {
    #[error("pool io at {0}: {1}")]
    Io(PathBuf, String),
    #[error("no adapter {0} in this pool")]
    Unknown(String),
    #[error("adapter {0} is already in this pool - an id identifies one adapter, so re-admitting would silently replace it")]
    AlreadyPresent(String),
    #[error("adapter {0} is not archived, so there is nothing to resurrect")]
    NotArchived(String),
}

type Result<T> = std::result::Result<T, PoolError>;

#[derive(Debug, Serialize, Deserialize)]
struct Index {
    cfg: PoolConfig,
    now: u64,
    entries: Vec<PoolEntry>,
    slots: Vec<Option<AdapterId>>,
}

/// Every adapter this reader has, and which are active.
#[derive(Debug)]
pub struct Pool {
    root: PathBuf,
    cfg: PoolConfig,
    now: u64,
    entries: BTreeMap<AdapterId, PoolEntry>,
    slots: Vec<Option<AdapterId>>,
}

impl Pool {
    /// Start a pool under `root`, creating its directories.
    pub fn create(root: &Path, cfg: PoolConfig) -> Result<Pool> {
        for d in [root.join(LIVE), root.join(ARCHIVE)] {
            std::fs::create_dir_all(&d).map_err(|e| PoolError::Io(d.clone(), e.to_string()))?;
        }
        Ok(Pool { root: root.to_path_buf(), cfg, now: 0, entries: BTreeMap::new(), slots: vec![None; cfg.resident] })
    }

    /// Reopen a pool written by [`Pool::save`].
    pub fn open(root: &Path) -> Result<Pool> {
        let path = root.join(INDEX);
        let text = std::fs::read_to_string(&path).map_err(|e| PoolError::Io(path.clone(), e.to_string()))?;
        let ix: Index = serde_json::from_str(&text).map_err(|e| PoolError::Io(path, e.to_string()))?;
        let entries = ix.entries.into_iter().map(|e| (e.id.clone(), e)).collect();
        Ok(Pool { root: root.to_path_buf(), cfg: ix.cfg, now: ix.now, entries, slots: ix.slots })
    }

    /// Write the index. Atomic: a temporary file then a rename, so an
    /// interrupted save cannot leave a pool that will not open.
    pub fn save(&self) -> Result<()> {
        let ix = Index { cfg: self.cfg, now: self.now, entries: self.entries.values().cloned().collect(), slots: self.slots.clone() };
        let text = serde_json::to_string_pretty(&ix).map_err(|e| PoolError::Io(self.root.clone(), e.to_string()))?;
        let tmp = self.root.join(format!("{INDEX}.tmp"));
        std::fs::write(&tmp, text).map_err(|e| PoolError::Io(tmp.clone(), e.to_string()))?;
        std::fs::rename(&tmp, self.root.join(INDEX)).map_err(|e| PoolError::Io(self.root.join(INDEX), e.to_string()))
    }

    /// Advance to the next episode. Staleness and dwell are both counted in
    /// these, not in wall-clock.
    pub fn tick(&mut self) {
        self.now += 1;
    }

    pub fn now(&self) -> u64 {
        self.now
    }

    pub fn entry(&self, id: &AdapterId) -> Option<&PoolEntry> {
        self.entries.get(id)
    }

    pub fn live(&self) -> Vec<&PoolEntry> {
        self.entries.values().filter(|e| e.state == State::Live).collect()
    }

    /// Slot assignment. `None` is an empty slot.
    pub fn slots(&self) -> &[Option<AdapterId>] {
        &self.slots
    }

    /// Add a newly trained adapter, storing `bytes` as its file.
    pub fn admit(&mut self, id: &AdapterId, bytes: &[u8]) -> Result<()> {
        if self.entries.contains_key(id) {
            return Err(PoolError::AlreadyPresent(id.0.clone()));
        }
        let file = format!("{}.adapter", id.0);
        let path = self.root.join(LIVE).join(&file);
        std::fs::write(&path, bytes).map_err(|e| PoolError::Io(path, e.to_string()))?;
        // `last_seen` starts at birth rather than at zero: an adapter that has
        // just arrived has not "gone unaddressed since episode zero", and
        // starting it at zero would retire a newborn into a long-running pool
        // on its first `retire_stale`.
        self.entries.insert(
            id.clone(),
            PoolEntry { id: id.clone(), state: State::Live, born: self.now, last_seen: self.now, admitted: self.now, file },
        );
        Ok(())
    }

    /// The adapter's file, wherever it currently lives.
    pub fn read(&self, id: &AdapterId) -> Result<Vec<u8>> {
        let e = self.entries.get(id).ok_or_else(|| PoolError::Unknown(id.0.clone()))?;
        let path = self.path_of(e);
        std::fs::read(&path).map_err(|err| PoolError::Io(path, err.to_string()))
    }

    /// Choose the working set for this episode from `demand`, and return the
    /// adapters that had to be LOADED - which is the cost this policy exists
    /// to keep small.
    ///
    /// An adapter absent from `demand` is simply not wanted this episode; it
    /// is not thereby evicted, since a resident is only displaced by a
    /// candidate that beats it.
    pub fn select(&mut self, demand: &BTreeMap<AdapterId, f64>) -> Vec<AdapterId> {
        let want = |id: &AdapterId| demand.get(id).copied().unwrap_or(0.0);

        // Candidates: live, not already resident, in descending demand with a
        // deterministic tiebreak so two equal scores do not depend on map
        // iteration order.
        let resident: Vec<AdapterId> = self.slots.iter().flatten().cloned().collect();
        let mut candidates: Vec<AdapterId> = self
            .entries
            .values()
            .filter(|e| e.state == State::Live && !resident.contains(&e.id))
            .map(|e| e.id.clone())
            .collect();
        candidates.sort_by(|a, b| want(b).partial_cmp(&want(a)).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.cmp(b)));

        let mut loaded = Vec::new();
        let mut next = candidates.into_iter().peekable();

        // Empty slots first: nothing has to be displaced to use them.
        for i in 0..self.slots.len() {
            if self.slots[i].is_none() {
                match next.next() {
                    Some(c) => {
                        self.slots[i] = Some(c.clone());
                        loaded.push(c);
                    }
                    None => break,
                }
            }
        }

        // Then displacement, weakest resident first, and only by a candidate
        // that is clear of the margin. A resident inside its dwell is not a
        // candidate for eviction at all, however wanted the challenger.
        while let Some(c) = next.peek().cloned() {
            let weakest = self
                .slots
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.as_ref().map(|id| (i, id.clone())))
                .filter(|(_, id)| self.entries.get(id).is_some_and(|e| self.now.saturating_sub(e.admitted) >= self.cfg.dwell))
                .min_by(|(_, a), (_, b)| want(a).partial_cmp(&want(b)).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.cmp(b)));
            let Some((slot, victim)) = weakest else { break };
            if want(&c) <= want(&victim) + self.cfg.margin {
                break;
            }
            self.slots[slot] = Some(c.clone());
            loaded.push(c);
            next.next();
        }

        // Being in the working set IS being addressed, which is what
        // staleness is measured from. `admitted` moves only for an adapter
        // that actually entered this episode, so dwell protects a newcomer
        // rather than everything that stayed.
        let now = self.now;
        let entered: Vec<AdapterId> = loaded.clone();
        for id in self.slots.clone().into_iter().flatten() {
            if let Some(e) = self.entries.get_mut(&id) {
                e.last_seen = now;
                if entered.contains(&id) {
                    e.admitted = now;
                }
            }
        }
        loaded
    }

    /// Archive every live adapter nothing has asked for in `survival`
    /// episodes. Never touches one that is currently in the working set.
    pub fn retire_stale(&mut self) -> Result<Vec<AdapterId>> {
        let resident: Vec<AdapterId> = self.slots.iter().flatten().cloned().collect();
        let stale: Vec<AdapterId> = self
            .entries
            .values()
            .filter(|e| e.state == State::Live && !resident.contains(&e.id) && self.now.saturating_sub(e.last_seen) >= self.cfg.survival)
            .map(|e| e.id.clone())
            .collect();
        for id in &stale {
            self.move_to(id, State::Archived)?;
        }
        Ok(stale)
    }

    /// Bring an archived adapter back, file intact.
    pub fn resurrect(&mut self, id: &AdapterId) -> Result<()> {
        match self.entries.get(id) {
            None => Err(PoolError::Unknown(id.0.clone())),
            Some(e) if e.state != State::Archived => Err(PoolError::NotArchived(id.0.clone())),
            Some(_) => {
                self.move_to(id, State::Live)?;
                // Back from the archive with a clean slate on staleness: it
                // was brought back because something wants it, and leaving
                // `last_seen` where it was would retire it again immediately.
                let now = self.now;
                if let Some(e) = self.entries.get_mut(id) {
                    e.last_seen = now;
                }
                Ok(())
            }
        }
    }

    fn path_of(&self, e: &PoolEntry) -> PathBuf {
        let dir = match e.state {
            State::Live => LIVE,
            State::Archived => ARCHIVE,
        };
        self.root.join(dir).join(&e.file)
    }

    /// Move an adapter's file between the live and archive directories and
    /// record the new state. A rename, never a copy and never a delete.
    fn move_to(&mut self, id: &AdapterId, to: State) -> Result<()> {
        let e = self.entries.get(id).ok_or_else(|| PoolError::Unknown(id.0.clone()))?;
        let from_path = self.path_of(e);
        let mut moved = e.clone();
        moved.state = to;
        let to_path = self.path_of(&moved);
        std::fs::rename(&from_path, &to_path).map_err(|err| PoolError::Io(to_path, err.to_string()))?;
        self.entries.insert(id.clone(), moved);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static N: AtomicUsize = AtomicUsize::new(0);

    struct Dir(PathBuf);

    impl Dir {
        fn new(name: &str) -> Dir {
            let d = std::env::temp_dir().join(format!("brain-pool-{name}-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&d);
            Dir(d)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn id(s: &str) -> AdapterId {
        AdapterId(s.to_string())
    }

    fn demand(pairs: &[(&str, f64)]) -> BTreeMap<AdapterId, f64> {
        pairs.iter().map(|(k, v)| (id(k), *v)).collect()
    }

    /// A pool of `n` adapters, each with recognisable bytes, all past their
    /// dwell so eviction rules rather than protection is what is under test.
    fn seeded(dir: &Dir, cfg: PoolConfig, n: usize) -> Pool {
        let mut p = Pool::create(dir.path(), cfg).expect("create");
        for i in 0..n {
            p.admit(&id(&format!("a{i}")), format!("adapter {i} bytes").as_bytes()).expect("admit");
        }
        p
    }

    /// The rule that keeps loads equal to real churn. Demand is returned
    /// sorted, so the ORDER moves constantly; a resident must not be reloaded
    /// into a different slot because two scores swapped.
    #[test]
    fn a_resident_keeps_its_slot_when_demand_reorders() {
        let d = Dir::new("slot");
        let mut p = seeded(&d, PoolConfig { resident: 3, dwell: 0, ..PoolConfig::default() }, 3);

        p.select(&demand(&[("a0", 0.9), ("a1", 0.8), ("a2", 0.7)]));
        let first = p.slots().to_vec();
        assert_eq!(first.iter().flatten().count(), 3, "three adapters must fill three slots");

        let loaded = p.select(&demand(&[("a2", 0.9), ("a0", 0.8), ("a1", 0.7)]));
        assert_eq!(p.slots(), first.as_slice(), "the same SET in a different order must not move anything");
        assert!(loaded.is_empty(), "and must load nothing, got {loaded:?}");
    }

    /// The pair that makes `margin` mean something: a clear win displaces, a
    /// difference inside the margin does not.
    #[test]
    fn a_candidate_displaces_a_resident_only_when_it_beats_it_by_the_margin() {
        let d = Dir::new("margin");
        let cfg = PoolConfig { resident: 2, margin: 0.10, dwell: 0, ..PoolConfig::default() };
        let mut p = seeded(&d, cfg, 3);

        p.select(&demand(&[("a0", 0.9), ("a1", 0.8)]));
        assert_eq!(p.slots().iter().flatten().count(), 2);

        // Inside the margin: 0.85 beats a1's 0.80 by 0.05, less than 0.10.
        let loaded = p.select(&demand(&[("a0", 0.9), ("a1", 0.8), ("a2", 0.85)]));
        assert!(loaded.is_empty(), "a candidate inside the margin must not displace, got {loaded:?}");

        // Clear of it: 0.95 beats 0.80 by 0.15.
        let loaded = p.select(&demand(&[("a0", 0.9), ("a1", 0.8), ("a2", 0.95)]));
        assert_eq!(loaded, vec![id("a2")], "a candidate clear of the margin must displace the weakest resident");
        let resident: Vec<&AdapterId> = p.slots().iter().flatten().collect();
        assert!(resident.contains(&&id("a2")) && !resident.contains(&&id("a1")));
    }

    /// Both halves. A newcomer cannot be judged before it has had a chance to
    /// be useful, and cannot be protected forever either.
    #[test]
    fn a_newcomer_is_protected_for_its_dwell_and_evictable_after_it() {
        let d = Dir::new("dwell");
        let cfg = PoolConfig { resident: 1, margin: 0.0, dwell: 3, ..PoolConfig::default() };
        let mut p = seeded(&d, cfg, 2);

        p.select(&demand(&[("a0", 0.5)]));
        assert_eq!(p.slots()[0], Some(id("a0")));

        // Wanted far more, but a0 has only just arrived.
        for _ in 0..2 {
            p.tick();
            let loaded = p.select(&demand(&[("a0", 0.1), ("a1", 9.9)]));
            assert!(loaded.is_empty(), "a resident inside its dwell must not be evicted, however wanted the candidate");
        }

        p.tick();
        let loaded = p.select(&demand(&[("a0", 0.1), ("a1", 9.9)]));
        assert_eq!(loaded, vec![id("a1")], "past its dwell the same resident must be displaceable");
    }

    /// Rare is not dead, and deletion is the one thing a self-driving reader
    /// cannot undo.
    #[test]
    fn a_stale_adapter_is_archived_not_deleted_and_resurrects_with_its_bytes_intact() {
        let d = Dir::new("archive");
        let cfg = PoolConfig { resident: 1, margin: 0.0, dwell: 0, survival: 5 };
        let mut p = seeded(&d, cfg, 2);
        let before = p.read(&id("a1")).expect("read");

        p.select(&demand(&[("a0", 1.0)]));
        for _ in 0..6 {
            p.tick();
            p.select(&demand(&[("a0", 1.0)]));
        }

        let gone = p.retire_stale().expect("retire");
        assert_eq!(gone, vec![id("a1")], "the adapter nothing asked for must retire");
        assert_eq!(p.entry(&id("a1")).map(|e| e.state), Some(State::Archived), "and be ARCHIVED, not removed from the index");
        assert_eq!(p.read(&id("a1")).expect("an archived file is still readable"), before);

        p.resurrect(&id("a1")).expect("resurrect");
        assert_eq!(p.entry(&id("a1")).map(|e| e.state), Some(State::Live));
        assert_eq!(p.read(&id("a1")).expect("read"), before, "resurrection must return the same bytes");
        assert!(matches!(p.resurrect(&id("a1")), Err(PoolError::NotArchived(_))));
    }

    /// The stay-silent half of retirement: being in the working set is proof
    /// something asked for it this episode.
    #[test]
    fn retirement_never_archives_an_adapter_that_is_currently_resident() {
        let d = Dir::new("resident");
        let cfg = PoolConfig { resident: 2, margin: 0.0, dwell: 0, survival: 1 };
        let mut p = seeded(&d, cfg, 2);
        p.select(&demand(&[("a0", 1.0), ("a1", 0.9)]));
        for _ in 0..10 {
            p.tick();
            p.select(&demand(&[("a0", 1.0), ("a1", 0.9)]));
        }
        assert!(p.retire_stale().expect("retire").is_empty(), "nothing resident may be retired");
    }

    /// A reader is stopped and restarted; the pool it reopens has to be the
    /// pool it closed, slots included.
    #[test]
    fn the_pool_index_round_trips_through_disk() {
        let d = Dir::new("roundtrip");
        let cfg = PoolConfig { resident: 2, margin: 0.25, dwell: 4, survival: 99 };
        let mut p = seeded(&d, cfg, 3);
        p.tick();
        p.select(&demand(&[("a2", 0.9), ("a0", 0.4)]));
        let slots = p.slots().to_vec();
        let now = p.now();
        p.save().expect("save");
        drop(p);

        let re = Pool::open(d.path()).expect("open");
        assert_eq!(re.slots(), slots.as_slice());
        assert_eq!(re.now(), now);
        assert_eq!(re.live().len(), 3);
        assert_eq!(re.read(&id("a2")).expect("read"), b"adapter 2 bytes");
    }

    /// An id names one adapter. Re-admitting under the same id would replace
    /// a trained adapter with a different one and keep its lineage.
    #[test]
    fn re_admitting_a_known_id_is_refused_rather_than_silently_replacing_it() {
        let d = Dir::new("dup");
        let mut p = seeded(&d, PoolConfig::default(), 1);
        assert!(matches!(p.admit(&id("a0"), b"different"), Err(PoolError::AlreadyPresent(_))));
        assert_eq!(p.read(&id("a0")).expect("read"), b"adapter 0 bytes");
    }
}
