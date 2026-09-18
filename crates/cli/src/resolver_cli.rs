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
use std::path::PathBuf;

use brain_modelstore::resolve::{describe_ambiguity, Ambiguity, ArchSpec, Question, Resolution};
use capability::Assembly;
// `ResolveFailure`/`AMBIGUOUS_EXIT`/`MISSING_EXIT` are used here only through
// `try_resolve`'s return type - a caller that needs them by name reaches for
// `loader::resolver` (or the `loader` crate root) directly. `try_resolve`
// itself is re-exported (several `resident_*.rs`/`catalog.rs`/`label_cli.rs`
// call sites still spell it `crate::resolver_cli::try_resolve`);
// `resolve_structured` is used only inside this module, so a plain `use`.
pub use loader::resolver::try_resolve;
use loader::resolver::resolve_structured;

// ===================== the served (resident) path =====================
//
// `crates/cli/src/resident_*.rs` used to read its weight paths straight out of
// the environment (`std::env::var("BRAIN_FLUX1_DIR").ok()?`) and serve nothing
// at all when a variable was unset - even with exactly one unambiguous
// candidate for that role sitting in the model store the one-shot CLI would
// have found on its own. That was a second, far weaker weight-resolution
// mechanism living beside the real one. [`served_roles`] is the seam that
// removes it: the served path now calls INTO the same resolver
// (`brain_modelstore::resolve`, the same `ArchSpec`s, the same inventory scan)
// instead of carrying its own rules, exactly as `crates/cli/src/placement.rs`
// is the only thing that knows both halves of the GPU-placement split.

/// The DIRECTORY an artifact lives in, as the `String` every resident's own
/// constructor takes.
///
/// The two sources a role's path can come from do not agree on shape, and both
/// are correct: an operator's `BRAIN_ARCFACE_DIR` names the DIRECTORY holding
/// the released graph (that is what the variable has always meant), while the
/// resolver's `weights` role resolves to the graph FILE itself (that is the
/// artifact the inventory recorded and the classifier identified). A resident
/// whose loader takes the directory - `ArcFaceSession::load`, `ScrfdSession::
/// load`, PuLID's per-component roots, all of which join a known release
/// filename onto it - needs the same answer from either, so the conversion
/// lives here once rather than in each caller.
///
/// A path that is already a directory is returned unchanged; anything else
/// yields its parent.
pub fn containing_dir(path: &std::path::Path) -> Option<String> {
    let dir = if path.is_dir() { path } else { path.parent()? };
    Some(dir.to_string_lossy().into_owned())
}

/// One served role's binding to the environment variable that overrides it -
/// what makes an operator's explicit `BRAIN_FLUX1_DIR` win over anything the
/// store holds, unconditionally, for that role alone.
pub struct RoleEnv {
    /// A role name this architecture's [`ArchSpec::roles`] declares.
    pub role: &'static str,
    /// The variable whose value replaces whatever the resolver would pick.
    pub var: &'static str,
}

/// An [`Ambiguity`] rendered for a daemon's startup log: each real candidate
/// as the `VAR=<path>` assignment that would pin it, rather than
/// [`describe_ambiguity`]'s `--flag <path>` (there is no interactive terminal
/// to retype a flag at). Falls back to the shared CLI rendering for a
/// [`Question::Variant`]/[`Question::Unverifiable`] question, which names no
/// role and so has no variable to suggest.
fn describe_served_ambiguity(a: &Ambiguity, bindings: &[RoleEnv]) -> String {
    let Question::Role { role } = &a.question else { return describe_ambiguity(a) };
    let Some(var) = bindings.iter().find(|b| b.role == role).map(|b| b.var) else { return describe_ambiguity(a) };
    let mut s = format!("more than one candidate for '{role}' and nothing named which - set {var} to one of:\n");
    for c in &a.choices {
        for (_, value) in &c.selector {
            s.push_str(&format!("  {var}={value}\n"));
        }
    }
    s
}

/// Every role `spec` declares, resolved for a served model: each role's own
/// environment variable first, then the model store.
///
/// Precedence, per role independently:
/// 1. `bindings`' variable for that role, if set and non-empty - taken
///    verbatim, NEVER re-derived, never checked against the store. An operator
///    who names a path gets exactly that path, including one outside the model
///    store entirely.
/// 2. Otherwise whatever [`resolve_structured`] picks for it, from the same
///    scan/classification the one-shot CLI runs.
///
/// `None` - the model is simply not served, which is never a hard startup
/// failure for the whole daemon - when the store cannot answer:
/// - `Missing`/no models directory: silent, matching the behavior an unset
///   variable already had. A served model whose weights were never fetched
///   must not take the other 30 models down with it.
/// - `Ambiguous`: the store genuinely holds more than one candidate, so
///   picking one would be a guess. Logged with every real candidate and the
///   variable that pins it (see [`describe_served_ambiguity`]) - the daemon's
///   own startup log being the only place an operator can read it.
///
/// Short-circuits the scan entirely when every REQUIRED role already has a
/// variable set: a fully configured operator must pay nothing for a resolver
/// they are not using, and must not be refused because the paths they named
/// happen to live outside the store the scan covers.
///
/// Returns a real [`Assembly`] rather than a bare role map so that every
/// retrofitted resident speaks the SAME currency as the ones already migrated
/// off the environment entirely (`Qwen35Resident::from_assembly`,
/// `Moondream3Resident::from_assembly`, `flux2::Paths::from_assembly`) - a
/// resident should not need one path-extraction shape for the resolver and a
/// different one for the environment.
pub fn served_assembly(arch: &str, spec: &dyn ArchSpec, bindings: &[RoleEnv]) -> Option<Assembly> {
    let named: BTreeMap<String, String> =
        bindings.iter().filter_map(|b| std::env::var(b.var).ok().filter(|v| !v.is_empty()).map(|v| (b.role.to_string(), v))).collect();
    let fully_named = spec.roles().iter().filter(|r| !spec.optional_roles().contains(r)).all(|r| named.contains_key(*r));
    if fully_named {
        let roles: BTreeMap<String, PathBuf> = named.iter().map(|(role, path)| (role.clone(), PathBuf::from(path))).collect();
        let provenance = named.iter().map(|(role, path)| format!("{role}: {path} (named by the environment)")).collect();
        return Some(Assembly { id: format!("local/{arch}"), arch: arch.to_string(), variant: None, roles, provenance });
    }
    let resolution = match resolve_structured(arch, spec, &named) {
        Ok(r) => r,
        // No models directory at all - the resolver was never reached, and an
        // unset variable already meant "not served" here.
        Err(_) => return None,
    };
    let mut assembly = match resolution {
        Resolution::Resolved(assembly) => *assembly,
        Resolution::Ambiguous(a) => {
            eprintln!("brain: {arch} not served over the scheduler - {}", describe_served_ambiguity(&a, bindings));
            return None;
        }
        Resolution::Missing(_) => return None,
    };
    // An explicitly named path outranks the resolver's pick for that role even
    // when the resolver also found one - rule 1 above, applied last so it
    // cannot be undone.
    for (role, path) in named {
        assembly.provenance.push(format!("{role}: {path} (named by the environment, overriding the store)"));
        assembly.roles.insert(role, PathBuf::from(path));
    }
    Some(assembly)
}

/// [`served_assembly`]'s multi-instance counterpart: every real, independent
/// candidate of `spec.instance_role()` (see that method's own doc) becomes
/// its OWN served [`Assembly`], addressed by its real vendor/repo id.
/// `default_id` is used instead whenever there is exactly one instance for a
/// reason OTHER than a real, distinct candidate - every role already named
/// by the environment, or the store holding no real instance-role candidate
/// at all - so a fully-configured or single-checkpoint operator keeps
/// [`served_assembly`]'s exact historical id (the caller's own well-known
/// constant, e.g. `flux2::caps::MODEL`), not a resolver-internal placeholder.
///
/// An `Ambiguous`/`Missing` OTHER role on one candidate (a `dit` this store
/// has no matching `vae` size for, say) is logged and that ONE candidate is
/// dropped - it never takes every other real, independently-servable
/// candidate down with it.
pub fn served_assemblies(arch: &str, spec: &dyn ArchSpec, bindings: &[RoleEnv], default_id: &str) -> Vec<Assembly> {
    let named: BTreeMap<String, String> =
        bindings.iter().filter_map(|b| std::env::var(b.var).ok().filter(|v| !v.is_empty()).map(|v| (b.role.to_string(), v))).collect();
    let fully_named = spec.roles().iter().filter(|r| !spec.optional_roles().contains(r)).all(|r| named.contains_key(*r));
    if fully_named {
        let roles: BTreeMap<String, PathBuf> = named.iter().map(|(role, path)| (role.clone(), PathBuf::from(path))).collect();
        let provenance = named.iter().map(|(role, path)| format!("{role}: {path} (named by the environment)")).collect();
        return vec![Assembly { id: default_id.to_string(), arch: arch.to_string(), variant: None, roles, provenance }];
    }
    let Some(root) = loader::model_dir::resolve(None) else { return Vec::new() };
    let records = brain_modelstore::inventory::scan(&root);
    let specs: [&dyn ArchSpec; 1] = [spec];
    let placeholder = format!("local/{arch}");
    let mut out = Vec::new();
    for (id, resolution) in brain_modelstore::resolve::resolve_all(arch, &records, &specs, &named) {
        let mut assembly = match resolution {
            Resolution::Resolved(assembly) => *assembly,
            Resolution::Ambiguous(a) => {
                eprintln!("brain: {id} ({arch}) not served over the scheduler - {}", describe_served_ambiguity(&a, bindings));
                continue;
            }
            Resolution::Missing(_) => continue,
        };
        assembly.id = if id == placeholder { default_id.to_string() } else { id };
        for (role, path) in &named {
            assembly.provenance.push(format!("{role}: {path} (named by the environment, overriding the store)"));
            assembly.roles.insert(role.clone(), PathBuf::from(path));
        }
        out.push(assembly);
    }
    out
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
/// Whether [`with_arch_spec`] can build an `ArchSpec` for `arch` - the
/// half of "resolver-migrated" that actually replaces the env path, checked
/// against `resolve::RESOLVER_MIGRATED_ARCHS` by that module's own test.
pub fn has_arch_spec(arch: &str) -> bool {
    with_arch_spec(arch, |_| ()).is_some()
}

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
        "cosyvoice" => Some(f(&cosyvoice::spec::CosyVoiceSpec)),
        "minimaxmusic3" => Some(f(&minimaxmusic3::spec::MinimaxMusic3Spec)),
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

    /// `s3dit` (Z-Image) has no dedicated `_cli.rs` of its own, so it can
    /// only ever accept a `--dit`/`--vae`/`--text-encoder`/`--tokenizer`
    /// disambiguation flag through THIS generic path - without a row here,
    /// `crate::catalog::provider`'s `resolved_assembly_for` (which every
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

    // ============ the served (resident) path: `served_assembly` ============
    //
    // A one-role architecture whose real artifact is a `.safetensors` file
    // carrying a marker tensor, so `classify` reads genuine content the way
    // every real spec does rather than matching on a path.

    const MARKER: &str = "served.marker.weight";

    struct ServedSpec;
    impl ArchSpec for ServedSpec {
        fn arch(&self) -> &'static str {
            "servedtest"
        }
        fn roles(&self) -> &'static [&'static str] {
            &["weights"]
        }
        fn classify(&self, records: &[ArtifactRecord], _root: &Path) -> Vec<(usize, String, Confidence)> {
            records
                .iter()
                .enumerate()
                .filter(|(_, r)| {
                    r.usable()
                        && checkpoint::mmap::MmapSafetensors::open(r.path.to_string_lossy().as_ref()).is_ok_and(|m| m.shape(MARKER).is_some())
                })
                .map(|(i, _)| (i, "weights".to_string(), Confidence::Declared))
                .collect()
        }
        fn assemble(&self, chosen: &BTreeMap<String, usize>, _records: &[ArtifactRecord], _overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
            chosen.get("weights").ok_or("no weights")?;
            Ok(AssembleOutcome::Assembled(AssembledVariant { id: "local/servedtest".to_string(), variant: None }))
        }
        fn validate(&self, _assembly: &Assembly) -> Result<(), String> {
            Ok(())
        }
    }

    const SERVED_BINDINGS: &[RoleEnv] = &[RoleEnv { role: "weights", var: "BRAIN_SERVEDTEST_WEIGHTS" }];

    fn store(tag: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("brain-served-assembly-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    /// A real artifact the toy spec above will classify, at `<store>/<vendor>/<name>`.
    fn write_candidate(store: &Path, vendor: &str, name: &str) -> std::path::PathBuf {
        let path = store.join(vendor).join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        checkpoint::st::save_safetensors(path.to_str().unwrap(), &[(MARKER.to_string(), vec![1u64], vec![0.0f32])], &serde_json::json!({}), None).unwrap();
        path
    }

    /// With no variable set, ONE unambiguous candidate in the store is found
    /// on its own - the whole gap this seam closes. Before it, the served path
    /// read a bare `std::env::var(...)` and served nothing at all here.
    #[test]
    fn an_unambiguous_store_candidate_is_found_with_no_env_var_set() {
        let _serial = brain_testutil::env_lock();
        let root = store("unambiguous");
        let real = write_candidate(&root, "vendor", "model.safetensors");
        std::env::set_var("BRAIN_MODELS_DIR", &root);
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");

        let assembly = served_assembly("servedtest", &ServedSpec, SERVED_BINDINGS).expect("one candidate must resolve on its own");
        assert_eq!(assembly.roles["weights"], real);

        std::env::remove_var("BRAIN_MODELS_DIR");
        std::fs::remove_dir_all(&root).ok();
    }

    /// The override is unconditional: an explicitly named path wins even when
    /// the store holds a real, classifiable candidate that the resolver would
    /// otherwise have picked - and even when the named path is somewhere the
    /// scan does not cover at all. An operator who names a path must get
    /// exactly that path.
    #[test]
    fn an_env_var_override_wins_over_a_different_store_candidate() {
        let _serial = brain_testutil::env_lock();
        let root = store("override");
        let in_store = write_candidate(&root, "vendor", "model.safetensors");
        // Deliberately OUTSIDE the store, so this can only come from the
        // variable - a scan-based answer could never produce it.
        let outside = store("override-elsewhere").join("hand-placed.safetensors");
        std::fs::create_dir_all(outside.parent().unwrap()).unwrap();
        std::fs::write(&outside, b"not even a real checkpoint").unwrap();

        std::env::set_var("BRAIN_MODELS_DIR", &root);
        std::env::set_var("BRAIN_SERVEDTEST_WEIGHTS", &outside);
        let assembly = served_assembly("servedtest", &ServedSpec, SERVED_BINDINGS).expect("an explicitly named path must always resolve");
        assert_eq!(assembly.roles["weights"], outside, "the variable must win over the store's own candidate");
        assert_ne!(assembly.roles["weights"], in_store);

        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");
        std::env::remove_var("BRAIN_MODELS_DIR");
        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(outside.parent().unwrap()).ok();
    }

    /// Two equally good candidates and nothing naming one is a genuine
    /// question, so the model is NOT served - never a silent pick of whichever
    /// the directory walk reached first, and never a crash. The refusal names
    /// every real candidate and the variable that pins it, since a daemon's
    /// startup log is the only place its operator can read it.
    #[test]
    fn genuine_ambiguity_refuses_to_serve_and_names_every_candidate() {
        let _serial = brain_testutil::env_lock();
        let root = store("ambiguous");
        let a = write_candidate(&root, "vendor-a", "model.safetensors");
        let b = write_candidate(&root, "vendor-b", "model.safetensors");
        std::env::set_var("BRAIN_MODELS_DIR", &root);
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");

        assert!(served_assembly("servedtest", &ServedSpec, SERVED_BINDINGS).is_none(), "two real candidates must not be silently collapsed to one");

        // The message an operator actually gets: both paths, each as the
        // assignment that would select it.
        let records = brain_modelstore::inventory::scan(&root);
        let specs: [&dyn ArchSpec; 1] = [&ServedSpec];
        let Resolution::Ambiguous(amb) = brain_modelstore::resolve::resolve("servedtest", &records, &specs, &BTreeMap::new()) else {
            panic!("expected the two candidates to be ambiguous");
        };
        let msg = describe_served_ambiguity(&amb, SERVED_BINDINGS);
        assert!(msg.contains("BRAIN_SERVEDTEST_WEIGHTS="), "the message must name the variable to set: {msg}");
        assert!(msg.contains(a.to_str().unwrap()), "{msg}");
        assert!(msg.contains(b.to_str().unwrap()), "{msg}");

        // ...and naming one of them resolves it, which is what makes the
        // message actionable rather than merely informative.
        std::env::set_var("BRAIN_SERVEDTEST_WEIGHTS", &a);
        let assembly = served_assembly("servedtest", &ServedSpec, SERVED_BINDINGS).expect("naming one candidate must resolve the ambiguity");
        assert_eq!(assembly.roles["weights"], a);

        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");
        std::env::remove_var("BRAIN_MODELS_DIR");
        std::fs::remove_dir_all(&root).ok();
    }

    /// Nothing configured and nothing found is "not served", quietly - never a
    /// hard failure. A daemon serves ~30 models; one whose weights were never
    /// fetched must not take the other 29 down with it.
    #[test]
    fn nothing_configured_and_nothing_found_is_silently_not_served() {
        let _serial = brain_testutil::env_lock();
        let root = store("empty");
        std::env::set_var("BRAIN_MODELS_DIR", &root);
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");
        assert!(served_assembly("servedtest", &ServedSpec, SERVED_BINDINGS).is_none());
        std::env::remove_var("BRAIN_MODELS_DIR");
        std::fs::remove_dir_all(&root).ok();
    }

    // ============ the multi-instance served path: `served_assemblies` ============

    /// [`ServedSpec`] with its one role opted into instance-per-candidate
    /// resolution - the FLUX.2 `dit` shape, reusing `ServedSpec`'s own
    /// marker-tensor classify so both fixtures stay real content checks.
    struct MultiServedSpec;
    impl ArchSpec for MultiServedSpec {
        fn arch(&self) -> &'static str {
            "servedtest"
        }
        fn roles(&self) -> &'static [&'static str] {
            &["weights"]
        }
        fn classify(&self, records: &[ArtifactRecord], root: &Path) -> Vec<(usize, String, Confidence)> {
            ServedSpec.classify(records, root)
        }
        fn assemble(&self, chosen: &BTreeMap<String, usize>, records: &[ArtifactRecord], overrides: &BTreeMap<String, String>) -> Result<AssembleOutcome, String> {
            ServedSpec.assemble(chosen, records, overrides)
        }
        fn validate(&self, assembly: &Assembly) -> Result<(), String> {
            ServedSpec.validate(assembly)
        }
        fn instance_role(&self) -> Option<&'static str> {
            Some("weights")
        }
    }

    /// Two genuinely independent candidates, opted into instance-per-
    /// candidate resolution, must each become their OWN served `Assembly` -
    /// the exact case `served_assembly` (single-instance) refuses outright
    /// as `genuine_ambiguity_refuses_to_serve_and_names_every_candidate`
    /// above pins.
    #[test]
    fn served_assemblies_serves_every_real_candidate_under_its_own_id() {
        let _serial = brain_testutil::env_lock();
        let root = store("multi-instance");
        let a = write_candidate(&root, "vendor-a", "model.safetensors");
        let b = write_candidate(&root, "vendor-b", "model.safetensors");
        std::env::set_var("BRAIN_MODELS_DIR", &root);
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");

        let mut got = served_assemblies("servedtest", &MultiServedSpec, SERVED_BINDINGS, "default/servedtest");
        got.sort_by(|x, y| x.id.cmp(&y.id));
        assert_eq!(got.len(), 2, "{:?}", got.iter().map(|a| &a.id).collect::<Vec<_>>());
        assert_eq!(got[0].id, "vendor-a/model");
        assert_eq!(got[0].roles["weights"], a);
        assert_eq!(got[1].id, "vendor-b/model");
        assert_eq!(got[1].roles["weights"], b);

        std::env::remove_var("BRAIN_MODELS_DIR");
        std::fs::remove_dir_all(&root).ok();
    }

    /// Every role already named by the environment still collapses to
    /// exactly ONE instance, under `default_id` - never fanned out into
    /// per-candidate instances just because the spec opted its role in.
    #[test]
    fn served_assemblies_uses_default_id_when_every_role_is_named_by_the_environment() {
        let _serial = brain_testutil::env_lock();
        let root = store("multi-instance-named");
        let named = write_candidate(&root, "vendor", "model.safetensors");
        std::env::set_var("BRAIN_MODELS_DIR", &root);
        std::env::set_var("BRAIN_SERVEDTEST_WEIGHTS", &named);

        let got = served_assemblies("servedtest", &MultiServedSpec, SERVED_BINDINGS, "default/servedtest");
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].id, "default/servedtest");
        assert_eq!(got[0].roles["weights"], named);

        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");
        std::env::remove_var("BRAIN_MODELS_DIR");
        std::fs::remove_dir_all(&root).ok();
    }

    /// An empty store falls back to `resolve_all`'s own single-instance
    /// path, which resolves to `Missing` exactly as `served_assembly`'s own
    /// "nothing configured and nothing found" case does - silently not
    /// served, never a hard failure.
    #[test]
    fn served_assemblies_serves_nothing_when_the_store_holds_nothing() {
        let _serial = brain_testutil::env_lock();
        let root = store("multi-instance-empty");
        std::env::set_var("BRAIN_MODELS_DIR", &root);
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");

        assert!(served_assemblies("servedtest", &MultiServedSpec, SERVED_BINDINGS, "default/servedtest").is_empty(), "nothing on disk must serve nothing");

        std::env::remove_var("BRAIN_MODELS_DIR");
        std::fs::remove_dir_all(&root).ok();
    }

    /// `containing_dir` is what lets a resident take either source: the
    /// variable names the DIRECTORY, the resolver's role names the FILE inside
    /// it, and the loader wants the directory in both cases.
    #[test]
    fn containing_dir_normalizes_a_file_role_and_a_directory_variable_to_the_same_answer() {
        let root = store("containing");
        let repo = root.join("DIAMONIK7777").join("antelopev2");
        std::fs::create_dir_all(&repo).unwrap();
        let graph = repo.join("glintr100.onnx");
        std::fs::write(&graph, b"x").unwrap();
        assert_eq!(containing_dir(&graph), containing_dir(&repo), "a role's file and an operator's directory must agree");
        assert_eq!(containing_dir(&repo).as_deref(), repo.to_str());
        std::fs::remove_dir_all(&root).ok();
    }
}
