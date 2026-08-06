// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A generic in-flight job registry: armed [`capability::CancelToken`]s keyed
//! by a per-request id. Previously hand-rolled identically in two places —
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

/// Armed cancel tokens for in-flight jobs, keyed by `K`. An entry lives from
/// submission until the reply fires (or an admission timeout sheds it first).
/// Cloning is cheap — an `Arc` around the shared map.
pub struct JobRegistry<K>(Arc<Mutex<HashMap<K, CancelToken>>>);

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
            m.insert(id, token);
        }
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
        self.0.lock().ok().and_then(|m| m.get(id).cloned())
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
    fn clone_shares_the_same_underlying_map() {
        let reg: JobRegistry<u64> = JobRegistry::new();
        let reg2 = reg.clone();
        reg.insert(7, CancelToken::armed());
        assert_eq!(reg2.len(), 1, "a clone must see inserts made through the original");
        reg2.remove(&7);
        assert_eq!(reg.len(), 0, "a clone's remove must be visible through the original");
    }
}
