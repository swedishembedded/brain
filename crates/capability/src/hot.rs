// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A single-slot cache of a built model, keyed by what it was built from.
//!
//! Every capability provider that loads weights lazily held the same cell:
//! `Arc<Mutex<Option<(String, Session)>>>`, a key compared on every call, the
//! previous build dropped before another is attempted, and the build done
//! under the lock so concurrent callers serialize on one load. Restated once
//! per provider, the policy (which build wins, when the old one is freed,
//! what a poisoned lock means) was re-decided each time. This is that cell,
//! written once.
//!
//! [`Hot::get`] keeps the lock for the whole call - the build and the use -
//! because a resident session is not safe to hand out while another caller is
//! replacing it. It therefore takes a closure for the use rather than
//! returning a reference.
//!
//! Swedish Embedded AB implements serving lifecycles where a model's lazy
//! load, residency and replacement follow one policy across every model, for
//! its clients. If your team needs expertise in capability-serving
//! infrastructure, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::sync::Mutex;

/// One built value, rebuilt when the key it was built from changes.
pub struct Hot<K, V> {
    cell: Mutex<Option<(K, V)>>,
}

impl<K, V> Default for Hot<K, V> {
    fn default() -> Self {
        Hot::new()
    }
}

impl<K, V> Hot<K, V> {
    /// An empty cache.
    pub const fn new() -> Hot<K, V> {
        Hot { cell: Mutex::new(None) }
    }
}

impl<K: PartialEq, V> Hot<K, V> {
    /// The slot, with the value for `key` built if it was not already.
    fn ready(&self, key: K, build: impl FnOnce() -> Result<V, String>) -> Result<std::sync::MutexGuard<'_, Option<(K, V)>>, String> {
        let mut guard = self.cell.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if !matches!(&*guard, Some((k, _)) if *k == key) {
            *guard = None;
            let value = build()?;
            *guard = Some((key, value));
        }
        Ok(guard)
    }

    /// Build (or reuse) the value for `key` and run `use_it` on it, all under
    /// one lock.
    ///
    /// On a miss the previous value is DROPPED before `build` runs, so a
    /// replacement of a model that holds most of a card's memory frees it
    /// first instead of holding two at once. A build that fails leaves the
    /// cache empty rather than holding the value it replaced: the next call
    /// retries.
    ///
    /// A panic elsewhere in the process may poison the lock; this recovers the
    /// value rather than failing every later call, because a panic during
    /// `use_it` does not change what the cache holds.
    pub fn get<R>(&self, key: K, build: impl FnOnce() -> Result<V, String>, use_it: impl FnOnce(&V) -> R) -> Result<R, String> {
        let guard = self.ready(key, build)?;
        Ok(use_it(&guard.as_ref().expect("built above").1))
    }

    /// [`Hot::get`] for a use that needs the value mutably.
    pub fn get_mut<R>(&self, key: K, build: impl FnOnce() -> Result<V, String>, use_it: impl FnOnce(&mut V) -> R) -> Result<R, String> {
        let mut guard = self.ready(key, build)?;
        Ok(use_it(&mut guard.as_mut().expect("built above").1))
    }

    /// [`Hot::get`] for a use that can itself fail - the shape every provider's
    /// `Action::run` has, so the result is one `Result` rather than a nested
    /// one.
    pub fn try_get<R>(&self, key: K, build: impl FnOnce() -> Result<V, String>, use_it: impl FnOnce(&V) -> Result<R, String>) -> Result<R, String> {
        self.get(key, build, use_it)?
    }

    /// [`Hot::get_mut`] for a use that can itself fail.
    pub fn try_get_mut<R>(&self, key: K, build: impl FnOnce() -> Result<V, String>, use_it: impl FnOnce(&mut V) -> Result<R, String>) -> Result<R, String> {
        self.get_mut(key, build, use_it)?
    }

    /// Build (or reuse) the value for `key` and hand back a CLONE of it.
    ///
    /// For a value that is itself a shared handle (`Arc<Session>`): the caller
    /// uses it outside the lock, so one long generation does not serialize
    /// every other call on the same provider.
    pub fn get_cloned(&self, key: K, build: impl FnOnce() -> Result<V, String>) -> Result<V, String>
    where
        V: Clone,
    {
        self.get(key, build, |v| v.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[test]
    fn a_hit_does_not_build_again() {
        let hot: Hot<String, u32> = Hot::new();
        let builds = Cell::new(0);
        let build = || {
            builds.set(builds.get() + 1);
            Ok(7)
        };
        assert_eq!(hot.get("a".to_string(), build, |v| *v), Ok(7));
        assert_eq!(hot.get("a".to_string(), build, |v| *v), Ok(7));
        assert_eq!(builds.get(), 1);
    }

    #[test]
    fn a_different_key_replaces_the_value() {
        let hot: Hot<String, u32> = Hot::new();
        assert_eq!(hot.get("a".to_string(), || Ok(1), |v| *v), Ok(1));
        assert_eq!(hot.get("b".to_string(), || Ok(2), |v| *v), Ok(2));
        assert_eq!(hot.get("b".to_string(), || Ok(3), |v| *v), Ok(2), "the key still hits");
    }

    #[test]
    fn the_previous_value_is_dropped_before_the_replacement_is_built() {
        use std::rc::Rc;
        struct Tracked(Rc<Cell<u32>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.set(self.0.get() - 1);
            }
        }
        let hot: Hot<String, Tracked> = Hot::new();
        let live = Rc::new(Cell::new(0));
        hot.get("a".to_string(), || {
            live.set(live.get() + 1);
            Ok(Tracked(live.clone()))
        }, |_| ())
        .expect("built");
        assert_eq!(live.get(), 1);
        hot.get("b".to_string(), || {
            assert_eq!(live.get(), 0, "the old build must be free before the new one allocates");
            live.set(1);
            Ok(Tracked(live.clone()))
        }, |_| ())
        .expect("built");
    }

    #[test]
    fn a_failed_build_leaves_the_cache_empty() {
        let hot: Hot<String, u32> = Hot::new();
        hot.get("a".to_string(), || Ok(1), |v| *v).expect("built");
        assert_eq!(hot.get("b".to_string(), || Err("no weights".to_string()), |v| *v), Err("no weights".to_string()));
        // The replaced value is gone: the next call for the old key builds.
        assert_eq!(hot.get("a".to_string(), || Ok(2), |v| *v), Ok(2));
    }
}
