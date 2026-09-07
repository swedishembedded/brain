// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared CLI plumbing for a `--model`/`--<role>`/`--variant`-driven
//! architecture command (`brain flux2 generate`, and every architecture
//! migrated onto the model-store resolver after it) - one place for the
//! "parse role override flags, resolve, print and exit on Ambiguous/
//! Missing" shape every such command needs identically, instead of each
//! architecture's own `<arch>_cli.rs` hand-writing it again.
//!
//! Swedish Embedded AB implements CLI plumbing like this for clients whose
//! own tools need to expose a typed resolver's ambiguity/missing outcomes
//! consistently across many subcommands. If your team needs the same
//! discipline for its own CLI, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::BTreeMap;

use brain_modelstore::resolve::{describe_ambiguity, describe_missing, ArchSpec, Resolution};
use capability::Assembly;

/// Exit code for an unresolvable `Resolution::Ambiguous` - distinct from
/// [`MISSING_EXIT`] so a script can tell "name one more thing" apart from
/// "nothing here at all" without parsing the message.
pub const AMBIGUOUS_EXIT: i32 = 3;
/// Exit code for `Resolution::Missing`.
pub const MISSING_EXIT: i32 = 4;

/// Resolve `arch`'s roles through the model-store resolver, printing and
/// exiting on `Ambiguous`/`Missing` - neither is recoverable within a single
/// command invocation, and resolving is never a place to guess. Scans the
/// models directory itself (`crate::model_dir::resolve`), so a caller only
/// needs its own [`ArchSpec`] and whatever role/variant overrides its own
/// flags parsed.
pub fn resolve_or_exit(arch: &str, spec: &dyn ArchSpec, overrides: &BTreeMap<String, String>) -> Assembly {
    let Some(root) = crate::model_dir::resolve(None) else {
        eprintln!("{arch}: no models directory (set --models-dir, BRAIN_MODELS_DIR, or $HOME)");
        std::process::exit(1);
    };
    let records = brain_modelstore::inventory::scan(&root);
    let specs: [&dyn ArchSpec; 1] = [spec];
    match brain_modelstore::resolve::resolve(arch, &records, &specs, overrides) {
        Resolution::Resolved(assembly) => *assembly,
        Resolution::Ambiguous(a) => {
            eprint!("{}", describe_ambiguity(&a));
            std::process::exit(AMBIGUOUS_EXIT);
        }
        Resolution::Missing(m) => {
            eprint!("{}", describe_missing(&m));
            std::process::exit(MISSING_EXIT);
        }
    }
}

/// Strip every `--<dashed-role>` flag matching one of `spec`'s declared
/// roles out of `args` (e.g. `--text-encoder <path>` for a role named
/// `"text_encoder"`), returning the collected role overrides and the
/// remaining argv untouched - so a command's own flag-parsing loop only
/// has to handle its own flags, not reimplement this per architecture.
/// A role flag with no following value is left in `remaining` (and so
/// surfaces as that command's own "unknown flag"/"needs a value" error),
/// never silently dropped.
pub fn extract_role_overrides(spec: &dyn ArchSpec, args: &[String]) -> (BTreeMap<String, String>, Vec<String>) {
    let mut overrides = BTreeMap::new();
    let mut remaining = Vec::with_capacity(args.len());
    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        let matched_role = spec.roles().iter().find(|r| *flag == format!("--{}", r.replace('_', "-")));
        match (matched_role, args.get(i + 1)) {
            (Some(role), Some(value)) => {
                overrides.insert((*role).to_string(), value.clone());
                i += 2;
            }
            _ => {
                remaining.push(flag.clone());
                i += 1;
            }
        }
    }
    (overrides, remaining)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use brain_modelstore::inventory::ArtifactRecord;
    use brain_modelstore::resolve::{AssembleOutcome, AssembledVariant, Confidence};

    struct TwoRoleSpec;
    impl ArchSpec for TwoRoleSpec {
        fn arch(&self) -> &'static str {
            "two"
        }
        fn roles(&self) -> &'static [&'static str] {
            &["dit", "text_encoder"]
        }
        fn classify(&self, _records: &[ArtifactRecord], _root: &Path) -> Vec<(usize, String, Confidence)> {
            Vec::new()
        }
        fn assemble(&self, _chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
            Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/two".to_string(), variant: None }))
        }
        fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
            Ok(())
        }
    }

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    /// A role flag (its dashed form, per `spec.roles()`) is pulled out with
    /// its value; every other flag (including one this architecture has no
    /// role named after) passes through untouched, in its original order.
    #[test]
    fn extract_role_overrides_pulls_out_only_declared_role_flags() {
        let spec = TwoRoleSpec;
        let args = s(&["--prompt", "a dog", "--text-encoder", "Qwen/Qwen3-8B", "--steps", "4", "--dit", "unsloth/dit.gguf"]);
        let (overrides, remaining) = extract_role_overrides(&spec, &args);
        assert_eq!(overrides.get("text_encoder").map(String::as_str), Some("Qwen/Qwen3-8B"));
        assert_eq!(overrides.get("dit").map(String::as_str), Some("unsloth/dit.gguf"));
        assert_eq!(remaining, s(&["--prompt", "a dog", "--steps", "4"]));
    }

    /// A role flag with nothing after it (a truncated/malformed argv) is
    /// left in `remaining` rather than silently swallowed - the caller's own
    /// "needs a value" handling sees it, exactly as if it were any other
    /// flag it doesn't specifically recognize the shape of.
    #[test]
    fn extract_role_overrides_leaves_a_valueless_role_flag_for_the_caller() {
        let spec = TwoRoleSpec;
        let args = s(&["--dit"]);
        let (overrides, remaining) = extract_role_overrides(&spec, &args);
        assert!(overrides.is_empty());
        assert_eq!(remaining, s(&["--dit"]));
    }

    /// A flag that isn't shaped like any declared role at all (e.g. a
    /// generic `--variant`, which is an architecture-specific override key,
    /// not a role) is untouched - this helper only ever strips role flags.
    #[test]
    fn extract_role_overrides_ignores_non_role_flags_entirely() {
        let spec = TwoRoleSpec;
        let args = s(&["--variant", "klein-9b"]);
        let (overrides, remaining) = extract_role_overrides(&spec, &args);
        assert!(overrides.is_empty());
        assert_eq!(remaining, args);
    }
}
