// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared CLI plumbing for a `--model`/`--<role>`/`--variant`-driven
//! architecture command (`brain flux2 generate`, and every architecture
//! migrated onto the model-store resolver after it) - one place for the
//! "resolve, then print and exit on `Ambiguous`/`Missing`" shape every such
//! command needs identically. Parsing a command's own `--<role>` flags into
//! the override map this expects is still each `<arch>_cli.rs`'s own job -
//! its flags are interleaved with plenty that aren't roles at all.
//!
//! [`resolve_or_exit`]/[`extract_role_overrides`] are the two primitives a
//! dedicated command (`flux2_cli`, `sam2_cli`'s `track`) calls directly.
//! [`run_generic_migrated`] is the third: the single-role counterpart for an
//! architecture with NO dedicated command at all, reached only through
//! `crate::resolve::dispatch_arch`'s generic `ARCH_TO_MODEL` capability
//! dispatch - it wires the same two primitives into
//! `crate::caps_cli::run_do_with_assembly` so a migrated architecture needs
//! one row in its own `with_arch_spec` match, not a hand-written command.
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

/// [`try_resolve`]'s failure shapes, each carrying its own already-rendered
/// message - kept distinct (rather than one `String`) so a caller that must
/// exit the process ([`resolve_or_exit`]) can still pick the right exit code,
/// while a caller that cannot (`crate::catalog::provider`) can collapse it to
/// one message via [`ResolveFailure::message`].
pub enum ResolveFailure {
    /// No models directory is configured at all - not a resolver outcome,
    /// a precondition the resolver was never reached to evaluate.
    NoModelsDir(String),
    Ambiguous(String),
    Missing(String),
}

impl ResolveFailure {
    /// The rendered message, regardless of which shape this is.
    pub fn message(&self) -> &str {
        match self {
            ResolveFailure::NoModelsDir(m) | ResolveFailure::Ambiguous(m) | ResolveFailure::Missing(m) => m,
        }
    }
    /// The process exit code a CLI caller should use for this outcome.
    pub fn exit_code(&self) -> i32 {
        match self {
            ResolveFailure::NoModelsDir(_) => 1,
            ResolveFailure::Ambiguous(_) => AMBIGUOUS_EXIT,
            ResolveFailure::Missing(_) => MISSING_EXIT,
        }
    }
}

impl std::fmt::Display for ResolveFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

/// Resolve `arch`'s roles through the model-store resolver: scans the models
/// directory itself (`crate::model_dir::resolve`) and reports every
/// non-`Resolved` outcome as an [`ResolveFailure`] instead of exiting - for a
/// caller that cannot exit the process on failure, such as
/// `crate::catalog::provider`'s "build me a runnable provider, or say why
/// not" contract.
pub fn try_resolve(arch: &str, spec: &dyn ArchSpec, overrides: &BTreeMap<String, String>) -> Result<Assembly, ResolveFailure> {
    let root = crate::model_dir::resolve(None).ok_or_else(|| ResolveFailure::NoModelsDir(format!("{arch}: no models directory (set --models-dir, BRAIN_MODELS_DIR, or $HOME)")))?;
    let records = brain_modelstore::inventory::scan(&root);
    let specs: [&dyn ArchSpec; 1] = [spec];
    match brain_modelstore::resolve::resolve(arch, &records, &specs, overrides) {
        Resolution::Resolved(assembly) => Ok(*assembly),
        Resolution::Ambiguous(a) => Err(ResolveFailure::Ambiguous(describe_ambiguity(&a))),
        Resolution::Missing(m) => Err(ResolveFailure::Missing(describe_missing(&m))),
    }
}

/// [`try_resolve`], printing and exiting on `Ambiguous`/`Missing`/no-models-
/// directory instead of returning an `Err` - neither is recoverable within a
/// single command invocation, and resolving is never a place to guess. What
/// every dedicated `<arch>_cli.rs` resolver-backed command calls.
pub fn resolve_or_exit(arch: &str, spec: &dyn ArchSpec, overrides: &BTreeMap<String, String>) -> Assembly {
    match try_resolve(arch, spec, overrides) {
        Ok(assembly) => assembly,
        Err(e) => {
            eprint!("{}", e.message());
            std::process::exit(e.exit_code());
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

/// Every architecture reached through `crate::resolve::ARCH_TO_MODEL`'s
/// generic capability dispatch (or a dedicated `_cli.rs` module that still
/// forwards its own non-special verbs to that generic path, e.g. `sam2_cli`)
/// whose weight resolution has moved to the model-store resolver, mapped to
/// its own [`ArchSpec`] - the one place a newly migrated single-role
/// architecture is wired into the generic dispatch path, instead of
/// hand-writing the resolve/extract-overrides call again per architecture.
/// `flux2` has its own dedicated command (`flux2_cli::resolve_flux2`) and is
/// not reached generically, so it has no row here.
fn with_arch_spec<R>(arch: &str, f: impl FnOnce(&dyn ArchSpec) -> R) -> Option<R> {
    match arch {
        "qwen3asr" => Some(f(&qwen3asr::spec::Qwen3AsrSpec)),
        "nemotronasr" => Some(f(&nemotronasr::spec::NemotronAsrSpec)),
        "sam2" => Some(f(&sam2::spec::Sam2Spec)),
        "rrdbnet" => Some(f(&rrdbnet::spec::RrdbnetSpec)),
        "timesfm3" => Some(f(&timesfm3::spec::Timesfm3Spec)),
        _ => None,
    }
}

/// Run a resolver-migrated architecture's generic capability action: extract
/// this architecture's own `--<role>` override flags out of `rest`, resolve
/// its [`Assembly`] (printing and exiting on `Ambiguous`/`Missing`, via
/// [`resolve_or_exit`]), then run the action through
/// `crate::catalog::provider_from_assembly` (`crate::caps_cli::run_do_with_assembly`).
/// `model` is the catalog id the action dispatches under.
fn run_generic_or_exit(arch: &str, spec: &dyn ArchSpec, model: &str, rest: &[String]) -> i32 {
    let (overrides, remaining) = extract_role_overrides(spec, rest);
    let assembly = resolve_or_exit(arch, spec, &overrides);
    let mut do_args = vec![model.to_string()];
    do_args.extend(remaining);
    crate::caps_cli::run_do_with_assembly(&do_args, &assembly)
}

/// [`run_generic_or_exit`], for an `arch` this module knows how to resolve -
/// `None` for any architecture not yet migrated onto the resolver, so its
/// caller falls back to the pre-existing env-based dispatch unchanged.
pub fn run_generic_migrated(arch: &str, model: &str, rest: &[String]) -> Option<i32> {
    with_arch_spec(arch, |spec| run_generic_or_exit(arch, spec, model, rest))
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
