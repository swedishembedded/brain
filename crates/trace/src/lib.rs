// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain-trace` - the workspace's observability front end.
//!
//! Two halves, deliberately separated the way `tracing` intends:
//!
//! * **Library crates emit** through the plain `tracing` facade
//!   (`tracing::{error,warn,info,debug,trace}!`, `#[tracing::instrument]`).
//!   They depend on `tracing` only - never on this crate - so nothing in the
//!   engine can accidentally install a subscriber, and a crate emitting
//!   events costs nothing when no subscriber is installed.
//! * **One binary installs** exactly one subscriber, here, from a
//!   [`Config`] the CLI parsed ([`strip_args`]).
//!
//! What this crate adds on top of `tracing_subscriber` is the *vocabulary*:
//! [`registry::FAMILIES`] maps a short user-facing name (`gpu`, `ltxv`) onto
//! the set of Rust targets it covers, so `--trace-gpu 5` does not require the
//! user to know that "the GPU" spans four crates. Everything else - the
//! per-component labelling, the level filtering, the text and JSON
//! renderers - is `tracing`'s, used as-is.
//!
//! # The 0-5 scale
//!
//! `--trace-<family> <0-5>` maps straight onto `tracing`'s five levels, so
//! the scale is not a second, parallel notion of verbosity:
//!
//! | level | means | `tracing` |
//! |---|---|---|
//! | 0 | off - no subscriber at all for this family | - |
//! | 1 | only things that failed | `ERROR` |
//! | 2 | + degraded/retried/fell-back | `WARN` |
//! | 3 | + coarse lifecycle (a run started, a stage finished) | `INFO` |
//! | 4 | + per-step decisions and timings | `DEBUG` |
//! | 5 | everything, per-iteration | `TRACE` |
//!
//! # Adding a family
//!
//! One [`registry::Family`] entry, plus the matching line in the `brain` CLI
//! help (a test asserts every registry family appears there). `--trace-<name>`
//! parsing and the filter directives both derive from the table, so there is
//! no third place to update. Until a family has a dedicated help line, the
//! generic `--trace <name>=<level>` reaches it already.

use std::path::PathBuf;

pub mod registry;

mod cli;
mod init;

pub use cli::{parse_level, strip_args, BRAIN_TRACE_ENV};
pub use init::{install, install_to, subscriber, Sink};
pub use registry::{family, names, Family, FAMILIES};

/// The highest accepted `--trace-<family>` level.
pub const MAX_LEVEL: u8 = 5;

/// How each event is rendered. `--trace-format text|json`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Format {
    /// `tracing_subscriber`'s human-readable formatter: timestamp, level,
    /// `target::module`, message, then the event's fields.
    #[default]
    Text,
    /// One JSON object per line, with `target`, `level`, `fields` and the
    /// enclosing span list as real JSON members - greppable with `jq`, not a
    /// message string that happens to contain braces.
    Json,
}

/// Where formatted events go. `--trace-output -|<path>`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Output {
    /// `-` (the default): the process's stdout.
    #[default]
    Stdout,
    /// A file, truncated at install time.
    File(PathBuf),
}

/// Everything the CLI collected: which families at which level, how to render,
/// and where to write. Applies globally across whichever families are active.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Config {
    /// `(family name, level)` pairs in the order they were given. A family
    /// named more than once keeps the LAST level - a later flag overrides an
    /// earlier one, matching how every other repeated CLI option behaves.
    pub families: Vec<(String, u8)>,
    pub format: Format,
    pub output: Output,
}

impl Config {
    /// The effective level for `name`, or 0 (off) if it was never requested.
    pub fn level_of(&self, name: &str) -> u8 {
        self.families.iter().rev().find(|(f, _)| f == name).map(|(_, l)| *l).unwrap_or(0)
    }

    /// True when nothing at all would be traced, so no subscriber needs to
    /// exist. `--trace-ltxv 0` is this, and so is passing no trace flag.
    pub fn is_off(&self) -> bool {
        registry::names().all(|n| self.level_of(n) == 0)
    }

    /// The `EnvFilter` directives this config expands to: one
    /// `<target>=<level>` per target of every family at a nonzero level.
    ///
    /// This is the only place a family becomes a filter, and it reads the
    /// registry - which is what makes a new family a one-entry change rather
    /// than an edit to filter-construction logic.
    pub fn directives(&self) -> Vec<String> {
        let mut out = Vec::new();
        for f in FAMILIES {
            let level = self.level_of(f.name);
            let Some(name) = level_name(level) else { continue };
            for t in f.targets {
                out.push(format!("{t}={name}"));
            }
        }
        out
    }
}

/// The `tracing` level a 0-5 verbosity maps to, or `None` for 0 (off).
pub fn level_filter(level: u8) -> Option<tracing::Level> {
    match level {
        1 => Some(tracing::Level::ERROR),
        2 => Some(tracing::Level::WARN),
        3 => Some(tracing::Level::INFO),
        4 => Some(tracing::Level::DEBUG),
        5 => Some(tracing::Level::TRACE),
        _ => None,
    }
}

/// The lowercase `EnvFilter` spelling of [`level_filter`], or `None` for off.
pub fn level_name(level: u8) -> Option<&'static str> {
    match level {
        1 => Some("error"),
        2 => Some("warn"),
        3 => Some("info"),
        4 => Some("debug"),
        5 => Some("trace"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 0-5 scale is exactly `tracing`'s five levels plus off - if these
    /// ever diverge, `--trace-x 4` and a `debug!` call stop meaning the same
    /// thing, which is the whole point of reusing `tracing`'s model.
    #[test]
    fn the_scale_is_tracings_own_five_levels_plus_off() {
        assert_eq!(level_filter(0), None);
        assert_eq!(level_filter(1), Some(tracing::Level::ERROR));
        assert_eq!(level_filter(2), Some(tracing::Level::WARN));
        assert_eq!(level_filter(3), Some(tracing::Level::INFO));
        assert_eq!(level_filter(4), Some(tracing::Level::DEBUG));
        assert_eq!(level_filter(5), Some(tracing::Level::TRACE));
        assert_eq!(level_filter(MAX_LEVEL), Some(tracing::Level::TRACE));
        for l in 0..=MAX_LEVEL {
            assert_eq!(level_filter(l).is_some(), level_name(l).is_some());
        }
    }

    /// A family expands to one directive per registered target, at the
    /// requested level, and an unrequested family contributes nothing - the
    /// property that keeps `--trace-ltxv 5` from also tracing the GPU.
    #[test]
    fn directives_cover_exactly_the_requested_families() {
        let cfg = Config { families: vec![("ltxv".into(), 2)], ..Default::default() };
        assert_eq!(cfg.directives(), vec!["ltxv=warn".to_string()]);

        let cfg = Config { families: vec![("gpu".into(), 5)], ..Default::default() };
        let gpu = family("gpu").expect("the gpu family is registered");
        assert_eq!(cfg.directives().len(), gpu.targets.len());
        for t in gpu.targets {
            assert!(cfg.directives().contains(&format!("{t}=trace")));
        }
    }

    /// Level 0 must produce no directives at all, not a directive at some
    /// floor level: "0 = no tracing" has to mean the subscriber is never
    /// installed, or a level-0 run still pays formatting cost and still
    /// prints errors.
    #[test]
    fn level_zero_is_genuinely_off() {
        let cfg = Config { families: vec![("ltxv".into(), 0), ("gpu".into(), 0)], ..Default::default() };
        assert!(cfg.is_off());
        assert!(cfg.directives().is_empty());
        assert!(Config::default().is_off());
    }

    /// A repeated family takes the last level, so `--trace-ltxv 1
    /// --trace ltxv=5` means 5 rather than silently keeping the first.
    #[test]
    fn a_repeated_family_keeps_the_last_level() {
        let cfg = Config { families: vec![("ltxv".into(), 1), ("ltxv".into(), 5)], ..Default::default() };
        assert_eq!(cfg.level_of("ltxv"), 5);
        assert_eq!(cfg.directives(), vec!["ltxv=trace".to_string()]);
    }
}
