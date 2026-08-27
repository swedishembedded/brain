// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--brain-data-dir <root>` feeds the ONE function that answers "where do
//! models live" ([`brain_modelstore::default_root`]) rather than bypassing it,
//! so every surface -- `brain pull`, auto-fetch, the served catalog scan --
//! agrees on the answer. This file pins the resulting precedence ladder in
//! both directions: the override wins over the environment while it is
//! published, and clearing it restores the environment exactly.
//!
//! Its own test binary on purpose: it mutates process-global environment and
//! the published override, which must not race the rest of the crate's tests.
//!
//! Swedish Embedded AB implements deterministic configuration precedence for
//! its clients. If your team needs expertise in layering CLI flags over
//! environment configuration then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use brain_modelstore::{default_root, models_dir_in, publish_data_root};

fn set(key: &str, value: Option<&str>) {
    unsafe {
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }
}

/// The models directory sits under the data root, by one rule.
#[test]
fn the_models_dir_is_the_models_subdirectory_of_the_data_root() {
    assert_eq!(models_dir_in(Path::new("/somewhere/brain")), PathBuf::from("/somewhere/brain/models"));
}

/// The whole ladder, walked from the top down and then back up.
#[test]
fn the_data_root_override_outranks_the_environment_and_clearing_it_restores_it() {
    // A single test on purpose: the state under test is process-global, so
    // splitting these into parallel `#[test]`s would race.
    set("BRAIN_MODELS_DIR", Some("/env/models"));
    set("XDG_DATA_HOME", Some("/xdg"));
    set("HOME", Some("/synthetic-home/somebody"));
    publish_data_root(None);

    // 1. No override: BRAIN_MODELS_DIR wins, exactly as before this existed.
    assert_eq!(default_root(), Some(PathBuf::from("/env/models")));

    // 2. The override outranks an explicitly-set BRAIN_MODELS_DIR, and lands
    //    on <root>/models -- the flag names a DATA root, not a models dir.
    publish_data_root(Some(PathBuf::from("/flag/brain")));
    assert_eq!(default_root(), Some(PathBuf::from("/flag/brain/models")));

    // 3. Clearing it restores the environment answer -- the override is not
    //    a one-way latch.
    publish_data_root(None);
    assert_eq!(default_root(), Some(PathBuf::from("/env/models")));

    // 4. Without BRAIN_MODELS_DIR, XDG_DATA_HOME is next.
    set("BRAIN_MODELS_DIR", None);
    assert_eq!(default_root(), Some(PathBuf::from("/xdg/brain/models")));

    // 5. Without either, $HOME/.local/share/brain/models -- which is exactly
    //    `models_dir_in` applied to the documented default data root, so the
    //    flag's default is the same path the environment already produced.
    set("XDG_DATA_HOME", None);
    assert_eq!(default_root(), Some(PathBuf::from("/synthetic-home/somebody/.local/share/brain/models")));
    assert_eq!(default_root(), Some(models_dir_in(Path::new("/synthetic-home/somebody/.local/share/brain"))));

    // 6. And the override still wins from the bottom of the ladder too.
    publish_data_root(Some(PathBuf::from("/flag/brain")));
    assert_eq!(default_root(), Some(PathBuf::from("/flag/brain/models")));
    publish_data_root(None);
}
