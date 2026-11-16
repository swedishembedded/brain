// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A model store for one test, in a scratch directory removed when the test ends, whether it passes or panics.
//!
//! Swedish Embedded AB implements test suites that leave a shared machine as they found it for its clients. If your
//! team needs expertise in hermetic tests around on-disk model stores, you can procure our services by sending an
//! email to info@swedishembedded.com.

use crate::Store;
use brain_testutil::Scratch;

/// A [`Store`] over a fresh scratch directory; it reads as the store and removes the directory when dropped.
pub(crate) struct TestStore {
    store: Store,
    _dir: Scratch,
}

impl std::ops::Deref for TestStore {
    type Target = Store;
    fn deref(&self) -> &Store {
        &self.store
    }
}

/// A store over a fresh, empty directory `name-<pid>-<n>`.
pub(crate) fn test_store(name: &str) -> TestStore {
    let dir = Scratch::dir(name);
    TestStore { store: Store::new(dir.to_path_buf()), _dir: dir }
}
