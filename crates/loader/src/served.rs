// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The served (resident) path's weight resolution: each served model's roles
//! come from its own environment variables first and from the model store
//! second, through the SAME resolver and `ArchSpec`s the one-shot CLI uses
//! rather than a second, weaker mechanism of their own.
//!
//! Moved out of `crates/cli/src/resolver_cli.rs` so the residency adapters,
//! which have to resolve their weights this way, can live in a library an
//! embedder can link (see `crates/catalog`) instead of in the `brain` binary.
//! A failure to resolve is never fatal here: a model whose weights are absent
//! or ambiguous is simply not served, and the reason goes to the log.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use brain_modelstore::resolve::{describe_ambiguity, Ambiguity, ArchSpec, Question, Resolution};
use capability::Assembly;

use crate::resolver::resolve_structured;

/// The roles an operator has pinned through their environment variables, each
/// to the path the variable holds. An unset or empty variable pins nothing.
fn named_by_environment(bindings: &[RoleEnv]) -> BTreeMap<String, String> {
    bindings.iter().filter_map(|b| std::env::var(b.var).ok().filter(|v| !v.is_empty()).map(|v| (b.role.to_string(), v))).collect()
}

/// The [`Assembly`] for a model whose every required role is already named by
/// the environment, under `id`; `None` when the store has to be consulted for
/// at least one of them.
fn fully_named(named: &BTreeMap<String, String>, arch: &str, spec: &dyn ArchSpec, id: String) -> Option<Assembly> {
    if !spec.roles().iter().filter(|r| !spec.optional_roles().contains(r)).all(|r| named.contains_key(*r)) {
        return None;
    }
    let roles = named.iter().map(|(role, path)| (role.clone(), PathBuf::from(path))).collect();
    let provenance = named.iter().map(|(role, path)| format!("{role}: {path} (named by the environment)")).collect();
    Some(Assembly { id, arch: arch.to_string(), variant: None, roles, provenance })
}

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
/// environment variable first, then the model store at `models_dir` - the
/// directory `brain serve` resolved ONCE from its own `--models-dir` flag
/// (falling back to `BRAIN_MODELS_DIR`, see `loader::model_dir::resolve`),
/// so every served model reads the same store the startup scan does.
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
pub fn served_assembly(models_dir: Option<&Path>, arch: &str, spec: &dyn ArchSpec, bindings: &[RoleEnv]) -> Option<Assembly> {
    let named = named_by_environment(bindings);
    if let Some(assembly) = fully_named(&named, arch, spec, format!("local/{arch}")) {
        return Some(assembly);
    }
    let resolution = match resolve_structured(models_dir, arch, spec, &named) {
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

/// [`served_assembly`]'s multi-instance counterpart (same `models_dir`
/// contract): every real, independent candidate of `spec.instance_role()`
/// (see that method's own doc) becomes its OWN served [`Assembly`],
/// addressed by its real vendor/repo id.
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
pub fn served_assemblies(models_dir: Option<&Path>, arch: &str, spec: &dyn ArchSpec, bindings: &[RoleEnv], default_id: &str) -> Vec<Assembly> {
    let named = named_by_environment(bindings);
    if let Some(assembly) = fully_named(&named, arch, spec, default_id.to_string()) {
        return vec![assembly];
    }
    let Some(root) = models_dir else { return Vec::new() };
    let records = brain_modelstore::inventory::scan(root);
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

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::inventory::ArtifactRecord;
    use brain_modelstore::resolve::{AssembleOutcome, AssembledVariant, Confidence};

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
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");

        let assembly = served_assembly(Some(&root), "servedtest", &ServedSpec, SERVED_BINDINGS).expect("one candidate must resolve on its own");
        assert_eq!(assembly.roles["weights"], real);

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

        std::env::set_var("BRAIN_SERVEDTEST_WEIGHTS", &outside);
        let assembly = served_assembly(Some(&root), "servedtest", &ServedSpec, SERVED_BINDINGS).expect("an explicitly named path must always resolve");
        assert_eq!(assembly.roles["weights"], outside, "the variable must win over the store's own candidate");
        assert_ne!(assembly.roles["weights"], in_store);

        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");
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
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");

        assert!(served_assembly(Some(&root), "servedtest", &ServedSpec, SERVED_BINDINGS).is_none(), "two real candidates must not be silently collapsed to one");

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
        let assembly = served_assembly(Some(&root), "servedtest", &ServedSpec, SERVED_BINDINGS).expect("naming one candidate must resolve the ambiguity");
        assert_eq!(assembly.roles["weights"], a);

        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");
        std::fs::remove_dir_all(&root).ok();
    }

    /// Nothing configured and nothing found is "not served", quietly - never a
    /// hard failure. A daemon serves ~30 models; one whose weights were never
    /// fetched must not take the other 29 down with it.
    #[test]
    fn nothing_configured_and_nothing_found_is_silently_not_served() {
        let _serial = brain_testutil::env_lock();
        let root = store("empty");
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");
        assert!(served_assembly(Some(&root), "servedtest", &ServedSpec, SERVED_BINDINGS).is_none());
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
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");

        let mut got = served_assemblies(Some(&root), "servedtest", &MultiServedSpec, SERVED_BINDINGS, "default/servedtest");
        got.sort_by(|x, y| x.id.cmp(&y.id));
        assert_eq!(got.len(), 2, "{:?}", got.iter().map(|a| &a.id).collect::<Vec<_>>());
        assert_eq!(got[0].id, "vendor-a/model");
        assert_eq!(got[0].roles["weights"], a);
        assert_eq!(got[1].id, "vendor-b/model");
        assert_eq!(got[1].roles["weights"], b);

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
        std::env::set_var("BRAIN_SERVEDTEST_WEIGHTS", &named);

        let got = served_assemblies(Some(&root), "servedtest", &MultiServedSpec, SERVED_BINDINGS, "default/servedtest");
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].id, "default/servedtest");
        assert_eq!(got[0].roles["weights"], named);

        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");
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
        std::env::remove_var("BRAIN_SERVEDTEST_WEIGHTS");

        assert!(served_assemblies(Some(&root), "servedtest", &MultiServedSpec, SERVED_BINDINGS, "default/servedtest").is_empty(), "nothing on disk must serve nothing");

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
