// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for engines that must start fast and stay fast for its clients. If your
// team needs expertise in GPU kernel caching and inference start-up then you can procure our services by sending an
// email to info@swedishembedded.com.

//! The cache directory every compiled kernel, tuning choice and measured roofline persists in is only answered when it
//! can actually hold files. A directory behind a dangling link used to be handed out anyway: every store then failed
//! without a word, and each start recompiled and re-tuned everything, with NVRTC compiles inside the first served
//! rounds.
//!
//! One test function: the directory is process-global state.

use std::path::PathBuf;

fn scratch(name: &str) -> PathBuf {
    // Cargo's per-target scratch directory for integration tests, fixed at compile time.
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let dir = base.join(format!("backend-api-cache-dir-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch directory");
    dir
}

#[test]
fn the_cache_directory_is_answered_only_when_it_can_hold_files() {
    let root = scratch("spec");

    // Missing but creatable: answered, and created so the first store succeeds.
    let fresh = root.join("fresh").join("brain");
    backend_api::set_cache_dir(Some(fresh.clone()));
    assert_eq!(backend_api::cache_dir(), Some(fresh.clone()));
    assert!(fresh.is_dir(), "the answered directory must exist");

    // A link into a wiped data drive: nowhere to persist, so no directory is answered.
    #[cfg(unix)]
    {
        let link = root.join("cache-link");
        std::os::unix::fs::symlink(root.join("wiped-drive").join("brain-cache"), &link).expect("symlink");
        backend_api::set_cache_dir(Some(link));
        assert_eq!(backend_api::cache_dir(), None, "a directory behind a dangling link must not be answered");
    }

    backend_api::set_cache_dir(None);
    let _ = std::fs::remove_dir_all(&root);
}
