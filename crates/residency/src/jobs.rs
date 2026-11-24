// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A generic in-flight job registry: armed [`capability::CancelToken`]s keyed
//! by a per-request id. Previously written identically in two places -
//! `apiserve::state::AppState` (keyed by `Uuid`) and `dbus::service::Manager`
//! (keyed by `u64`) — both wrapping an `Arc<Mutex<HashMap<K, CancelToken>>>`
//! with the same insert-on-submit / remove-on-finish shape. Factored here so
//! both transports share one implementation instead of two copies that can
//! drift. What does NOT move here: minting the id (a fresh `Uuid` for HTTP,
//! an incrementing counter for D-Bus), building the `Job`, or the admission
//! race — those are genuinely transport-specific.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex};

use capability::CancelToken;

/// What a job a client can address is, for the client that owns it: its owner (the transport's caller identity),
/// what it runs and when it was submitted. A job registered with it is listed to its owner and cancelled only by
/// its owner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JobInfo {
    pub owner: String,
    pub model: String,
    pub action: String,
    pub submitted: std::time::SystemTime,
}

struct Registered {
    token: CancelToken,
    info: Option<JobInfo>,
    /// Where the job's input goes, for a job that takes some ([`capability::inbox`]).
    inbox: Option<capability::inbox::InboxSender>,
}

/// Armed cancel tokens for in-flight jobs, keyed by `K`. An entry lives from
/// submission until the reply fires (or an admission timeout sheds it first).
/// Cloning is cheap — an `Arc` around the shared map.
pub struct JobRegistry<K>(Arc<Mutex<HashMap<K, Registered>>>);

impl<K> Clone for JobRegistry<K> {
    fn clone(&self) -> JobRegistry<K> {
        JobRegistry(self.0.clone())
    }
}

impl<K: Eq + Hash> Default for JobRegistry<K> {
    fn default() -> JobRegistry<K> {
        JobRegistry(Arc::new(Mutex::new(HashMap::new())))
    }
}

impl<K: Eq + Hash> JobRegistry<K> {
    pub fn new() -> JobRegistry<K> {
        JobRegistry::default()
    }

    /// Register `token` under `id`. The caller mints `id` and arms `token`
    /// before calling this.
    pub fn insert(&self, id: K, token: CancelToken) {
        if let Ok(mut m) = self.0.lock() {
            m.insert(id, Registered { token, info: None, inbox: None });
        }
    }

    /// [`Self::insert`] for a job a client addresses by its id: listed to `info.owner` by [`Self::owned_by`] and
    /// cancelled only through [`Self::get_owned`] by that owner.
    pub fn insert_described(&self, id: K, token: CancelToken, info: JobInfo, inbox: Option<capability::inbox::InboxSender>) {
        if let Ok(mut m) = self.0.lock() {
            m.insert(id, Registered { token, info: Some(info), inbox });
        }
    }

    /// Where the described job `id` takes input when `owner` owns it; `None` as [`Self::get_owned`] answers `None`, and
    /// for a job registered without an inbox.
    pub fn inbox_owned(&self, id: &K, owner: &str) -> Option<capability::inbox::InboxSender> {
        self.0.lock().ok().and_then(|m| m.get(id).filter(|r| r.info.as_ref().is_some_and(|i| i.owner == owner)).and_then(|r| r.inbox.clone()))
    }

    /// Drop a finished (or admission-timed-out) job's entry.
    pub fn remove(&self, id: &K) {
        if let Ok(mut m) = self.0.lock() {
            m.remove(id);
        }
    }

    /// A clone of the token registered under `id`, if it's still in flight —
    /// e.g. to cancel a job by id (`token.cancel()` on the result).
    pub fn get(&self, id: &K) -> Option<CancelToken> {
        self.0.lock().ok().and_then(|m| m.get(id).map(|r| r.token.clone()))
    }

    /// The token of the described job `id` when `owner` owns it; `None` for an id that is unknown, finished,
    /// undescribed or another owner's (which a caller cannot tell apart).
    pub fn get_owned(&self, id: &K, owner: &str) -> Option<CancelToken> {
        self.0.lock().ok().and_then(|m| m.get(id).filter(|r| r.info.as_ref().is_some_and(|i| i.owner == owner)).map(|r| r.token.clone()))
    }

    /// Every in-flight described job of `owner`, oldest first.
    pub fn owned_by(&self, owner: &str) -> Vec<(K, JobInfo)>
    where
        K: Clone,
    {
        let Ok(m) = self.0.lock() else { return Vec::new() };
        let mut mine: Vec<(K, JobInfo)> = m.iter().filter_map(|(id, r)| r.info.as_ref().filter(|i| i.owner == owner).map(|i| (id.clone(), i.clone()))).collect();
        mine.sort_by_key(|(_, i)| i.submitted);
        mine
    }

    /// Cancel every registered job: what a stopping server does so that
    /// in-flight generations end at their next poll instead of running on.
    /// Entries stay until their owners [`remove`](Self::remove) them.
    pub fn cancel_all(&self) {
        if let Ok(m) = self.0.lock() {
            m.values().for_each(|r| r.token.cancel());
        }
    }

    /// How many jobs are currently registered (in flight).
    pub fn len(&self) -> usize {
        self.0.lock().map(|m| m.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_remove_and_len_roundtrip() {
        let reg: JobRegistry<u64> = JobRegistry::new();
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());

        let t = CancelToken::armed();
        reg.insert(1, t.clone());
        assert_eq!(reg.len(), 1);
        assert!(!t.is_cancelled());

        // `get` returns a clone that can cancel the SAME underlying token.
        let found = reg.get(&1).expect("job 1 must be registered");
        found.cancel();
        assert!(t.is_cancelled(), "cancelling the clone from `get` must cancel the original");

        assert!(reg.get(&2).is_none(), "unregistered id must miss");

        reg.remove(&1);
        assert_eq!(reg.len(), 0);
        assert!(reg.get(&1).is_none(), "removed id must miss");
    }

    #[test]
    fn a_described_job_is_listed_to_and_cancelled_by_its_owner_alone() {
        let reg: JobRegistry<u64> = JobRegistry::new();
        let info = |owner: &str| JobInfo { owner: owner.to_string(), model: "m".into(), action: "a".into(), submitted: std::time::SystemTime::now() };
        let (mine, theirs) = (CancelToken::armed(), CancelToken::armed());
        reg.insert_described(1, mine.clone(), info("uid:1000"), None);
        reg.insert_described(2, theirs.clone(), info("uid:1001"), None);
        reg.insert(3, CancelToken::armed());
        assert_eq!(reg.owned_by("uid:1000").into_iter().map(|(id, _)| id).collect::<Vec<_>>(), vec![1]);
        assert!(reg.get_owned(&2, "uid:1000").is_none(), "another owner's job cannot be reached");
        assert!(reg.get_owned(&3, "uid:1000").is_none(), "nor can an undescribed one");
        reg.get_owned(&1, "uid:1000").expect("the owner reaches its job").cancel();
        assert!(mine.is_cancelled() && !theirs.is_cancelled());
        reg.remove(&1);
        assert!(reg.owned_by("uid:1000").is_empty(), "a finished job is no longer listed");
    }

    #[test]
    fn cancel_all_cancels_every_registered_job_and_keeps_the_entries() {
        let reg: JobRegistry<u64> = JobRegistry::new();
        let (a, b) = (CancelToken::armed(), CancelToken::armed());
        reg.insert(1, a.clone());
        reg.insert(2, b.clone());
        reg.cancel_all();
        assert!(a.is_cancelled() && b.is_cancelled());
        assert_eq!(reg.len(), 2, "owners still remove their own entries");
    }

    #[test]
    fn clone_shares_the_same_underlying_map() {
        let reg: JobRegistry<u64> = JobRegistry::new();
        let reg2 = reg.clone();
        reg.insert(7, CancelToken::armed());
        assert_eq!(reg2.len(), 1, "a clone must see inserts made through the original");
        reg2.remove(&7);
        assert_eq!(reg.len(), 0, "a clone's remove must be visible through the original");
    }
}
