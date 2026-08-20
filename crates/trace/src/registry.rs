// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The trace-family registry - the ONE source of truth that both the CLI's
//! `--trace-<family>` flags and the `tracing_subscriber` filter directives
//! read from.
//!
//! A *family* is a short, user-facing name (`gpu`, `ltxv`) standing for the
//! set of Rust targets a user actually means when they say "trace the GPU".
//! Nobody typing a command knows that "the GPU" is four crates, and nobody
//! should have to: the family is the vocabulary, the target list is the
//! implementation.
//!
//! **Targets are `tracing` targets, which default to a Rust module path**, so
//! an entry here is a crate's LIB name (`gpu_core`, not `brain-gpu-core`) and
//! matching is by prefix - `ltxv` covers `ltxv::pipeline`, `ltxv::dit` and
//! every other module inside that crate, while still printing the precise
//! module on each line so a reader can tell which component a message came
//! from. That is `tracing`'s own component-labelling mechanism; nothing here
//! re-implements it.
//!
//! **Adding a family is one entry in [`FAMILIES`]** plus one line in the
//! `brain` CLI's help text (a test in `crates/cli` asserts the two agree, so
//! they cannot drift). No filter-construction code changes, no new flag
//! parsing: [`crate::strip_args`] derives `--trace-<name>` from this table.

/// One `--trace-<name>` family: a short user-facing name over the set of
/// `tracing` targets it expands to.
pub struct Family {
    /// The user-facing name, as it appears in `--trace-<name>` and in
    /// `--trace <name>=<level>`. Lowercase, no separators.
    pub name: &'static str,
    /// The `tracing` targets this family turns on, matched by prefix. Each is
    /// a crate's lib name (the first segment of its modules' `module_path!`).
    pub targets: &'static [&'static str],
    /// One line describing what a user gets by enabling this family, shown in
    /// the CLI help.
    pub about: &'static str,
}

/// Every registered family, in the order the CLI help lists them.
pub const FAMILIES: &[Family] = &[
    Family {
        name: "gpu",
        // The compute-device facade plus the three backends that can each
        // fail differently at device-open time. Instrumentation calls inside
        // these crates are added separately from the flag itself - the flag,
        // the registry entry and the filter it builds are complete and
        // functional regardless of how many events those crates emit today.
        targets: &["gpu_core", "backend_wgpu", "backend_vulkan", "vulkan"],
        about: "device registry, adapter enumeration, backend open/submit/wait",
    },
    Family {
        name: "ltxv",
        targets: &["ltxv"],
        about: "LTX-2.5 video: pipeline stages, denoise steps, streamed DiT blocks",
    },
];

/// The family registered under `name`, if any.
pub fn family(name: &str) -> Option<&'static Family> {
    FAMILIES.iter().find(|f| f.name == name)
}

/// Every registered family name, in registry order.
pub fn names() -> impl Iterator<Item = &'static str> {
    FAMILIES.iter().map(|f| f.name)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A duplicate name would make `--trace-<name>` ambiguous and silently
    /// resolve to whichever entry `family()` happened to find first.
    #[test]
    fn family_names_are_unique_and_flag_shaped() {
        let mut seen: Vec<&str> = Vec::new();
        for f in FAMILIES {
            assert!(!seen.contains(&f.name), "duplicate trace family {:?}", f.name);
            seen.push(f.name);
            assert!(!f.name.is_empty(), "a trace family may not have an empty name");
            assert!(
                f.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit()),
                "trace family {:?} is not spellable as --trace-<name>",
                f.name
            );
            assert!(!f.about.is_empty(), "trace family {:?} has no description", f.name);
        }
    }

    /// A target here is matched against a `tracing` target (a Rust module
    /// path), so a crate NAME with a hyphen never matches anything: the
    /// resulting filter would be silently empty rather than an error.
    #[test]
    fn targets_are_lib_names_not_package_names() {
        for f in FAMILIES {
            assert!(!f.targets.is_empty(), "trace family {:?} covers no targets", f.name);
            for t in f.targets {
                assert!(
                    t.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'),
                    "trace target {t:?} in family {:?} is not a Rust lib name (hyphen? that is the package name)",
                    f.name
                );
            }
        }
    }
}
