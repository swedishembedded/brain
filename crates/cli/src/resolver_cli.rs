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
//! The non-exiting resolver core ([`try_resolve`]/[`resolve_structured`]/
//! [`ResolveFailure`]) now lives in `loader::resolver` - any embedder
//! wants the same "resolve, or say exactly why not" contract, not just this
//! CLI - and is re-exported here so every existing caller in this crate is
//! unaffected. [`resolve_or_exit`] (calls `std::process::exit`, a
//! process-lifetime decision with no meaning off a CLI) is what stays
//! CLI-only, as a thin wrapper over the moved core.
//!
//! Swedish Embedded AB implements CLI plumbing like this for clients whose
//! own tools need to expose a typed resolver's ambiguity/missing outcomes
//! consistently across many subcommands. If your team needs the same
//! discipline for its own CLI, you can procure our services by emailing
//! info@swedishembedded.com.

use std::collections::BTreeMap;

use brain_modelstore::resolve::ArchSpec;
use capability::Assembly;
// `ResolveFailure`/`AMBIGUOUS_EXIT`/`MISSING_EXIT` are used here only through
// `try_resolve`'s return type - a caller that needs them by name reaches for
// `loader::resolver` (or the `loader` crate root) directly. `try_resolve`
// itself is re-exported (several `catalog.rs`/`label_cli.rs` call sites still
// spell it `crate::resolver_cli::try_resolve`).
pub use loader::resolver::try_resolve;

/// [`try_resolve`], printing and exiting on `Ambiguous`/`Missing`/no-models-
/// directory instead of returning an `Err` - neither is recoverable within a
/// single command invocation, and resolving is never a place to guess. What
/// every dedicated `<arch>_cli.rs` resolver-backed command calls.
///
/// None of those commands takes a `--models-dir` flag, so the store is the
/// resolver's own flagless answer (`--brain-data-dir`, then
/// `BRAIN_MODELS_DIR`, then XDG/HOME - see `loader::model_dir::resolve`).
pub fn resolve_or_exit(arch: &str, spec: &dyn ArchSpec, overrides: &BTreeMap<String, String>) -> Assembly {
    match try_resolve(loader::model_dir::resolve(None).as_deref(), arch, spec, overrides) {
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

/// Whether [`with_arch_spec`] can build an `ArchSpec` for `arch` - the
/// half of "resolver-migrated" that actually replaces the env path, checked
/// against `resolve::RESOLVER_MIGRATED_ARCHS` by that module's own test (its
/// only caller, hence test-only).
#[cfg(test)]
pub fn has_arch_spec(arch: &str) -> bool {
    with_arch_spec(arch, |_| ()).is_some()
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
        "codeformer" => Some(f(&codeformer::spec::CodeFormerSpec)),
        "timesfm3" => Some(f(&timesfm3::spec::Timesfm3Spec)),
        "s3dit" => Some(f(&s3dit::spec::S3ditSpec)),
        "kronos" => Some(f(&kronos::spec::KronosSpec)),
        "qwen3vl" => Some(f(&qwen3vl::spec::Qwen3VlSpec)),
        "fastvlm" => Some(f(&fastvlm::spec::FastvlmSpec)),
        "deepseek2ocr" => Some(f(&deepseek2ocr::spec::Deepseek2ocrSpec)),
        "scrfd" => Some(f(&scrfd::spec::ScrfdSpec)),
        "arcface" => Some(f(&arcface::spec::ArcFaceSpec)),
        "clip" => Some(f(&clip::spec::ClipSpec)),
        "florence2" => Some(f(&florence2::spec::Florence2Spec)),
        "flux1" => Some(f(&flux1::spec::Flux1Spec)),
        "llava" => Some(f(&llava::spec::LlavaSpec)),
        "pulid" => Some(f(&pulid::spec::PulidSpec)),
        "vqgan" => Some(f(&vqgan::spec::VqganSpec)),
        "sdxlunet" => Some(f(&sdxlunet::spec::SdxlunetSpec)),
        "controlnet" => Some(f(&controlnet::spec::ControlnetSpec)),
        "t5encoder" => Some(f(&t5encoder::spec::T5encoderSpec)),
        "cosyvoice" => Some(f(&cosyvoice::spec::CosyVoiceSpec)),
        "minimaxmusic3" => Some(f(&minimaxmusic3::spec::MinimaxMusic3Spec)),
        _ => None,
    }
}

/// Run a resolver-migrated architecture's generic capability action: extract
/// this architecture's own `--<role>` override flags out of `rest`, resolve
/// its [`Assembly`] (printing and exiting on `Ambiguous`/`Missing`, via
/// [`resolve_or_exit`]), then run the action through
/// `catalog::provider_from_assembly` (`crate::caps_cli::run_do_with_assembly`).
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

    /// `s3dit` (Z-Image) has no dedicated `_cli.rs` of its own, so it can
    /// only ever accept a `--dit`/`--vae`/`--text-encoder`/`--tokenizer`
    /// disambiguation flag through THIS generic path - without a row here,
    /// `catalog::provider`'s `resolved_assembly_for` (which every
    /// `brain s3dit <verb>` invocation goes through, since `s3dit` is on
    /// `ARCH_TO_MODEL`) always resolves with an EMPTY override map, so an
    /// `Ambiguous` outcome (e.g. two unrelated VAEs in the models directory)
    /// can never be answered - the same flag the printed refusal message
    /// itself suggests typing is silently unusable.
    #[test]
    fn s3dit_is_reachable_through_the_generic_dispatch_with_its_real_roles() {
        let roles = with_arch_spec("s3dit", |spec| spec.roles().to_vec());
        assert_eq!(roles, Some(vec!["dit", "vae", "text_encoder", "tokenizer"]));
    }

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
