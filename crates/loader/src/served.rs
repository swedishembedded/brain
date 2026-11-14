// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The served (resident) path's weight resolution: each served model's roles
//! come from the paths its operator pinned first and from the model store
//! second, through the SAME resolver and `ArchSpec`s the one-shot CLI uses
//! rather than a second, weaker mechanism of their own.
//!
//! [`ServedStore`] is that input, built once by the serving process
//! (`brain serve --models-dir DIR --pin ARCH.ROLE=PATH`) and handed down to
//! every residency adapter. A failure to resolve is never fatal here: a model
//! whose weights are absent or ambiguous is simply not served, and the reason
//! goes to the log.
//!
//! Swedish Embedded AB implements model serving that finds its weights without
//! configuration for its clients. If your team needs expertise in model
//! distribution and serving then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use brain_modelstore::resolve::{describe_ambiguity, Ambiguity, ArchSpec, Question, Resolution};
use capability::Assembly;

use crate::resolver::{resolve_structured, try_resolve, ResolveFailure};

/// Where a serving process's models come from: the model store it scans and
/// the roles its operator pinned to explicit paths.
#[derive(Clone, Debug, Default)]
pub struct ServedStore {
    models_dir: Option<PathBuf>,
    /// `arch -> role -> path`.
    pins: BTreeMap<String, BTreeMap<String, String>>,
    /// `arch -> roles` some served model read pins under, shared by every
    /// clone so the serving process can check, once its models are built,
    /// that no pin went unread (see [`ServedStore::unread_pins`]).
    read: Arc<Mutex<BTreeMap<String, BTreeSet<String>>>>,
}

/// The override key a spec's `assemble` reads besides its roles: which named
/// variant of the architecture to serve (`--pin flux2.variant=klein-4b`).
const VARIANT: &str = "variant";

impl ServedStore {
    pub fn new(models_dir: Option<PathBuf>) -> ServedStore {
        ServedStore { models_dir, ..ServedStore::default() }
    }

    /// The same pins over the model store at `models_dir`.
    pub fn with_models_dir(self, models_dir: Option<PathBuf>) -> ServedStore {
        ServedStore { models_dir, ..self }
    }

    /// The model store, when the process has one.
    pub fn models_dir(&self) -> Option<&Path> {
        self.models_dir.as_deref()
    }

    /// Pin one role of one architecture to a path: `ARCH.ROLE=PATH`, the
    /// argument of `brain serve --pin`. A later pin of the same role wins.
    pub fn pin(&mut self, spec: &str) -> Result<(), String> {
        let (key, path) = spec.split_once('=').ok_or_else(|| format!("--pin {spec:?}: expected ARCH.ROLE=PATH"))?;
        let (arch, role) = key.split_once('.').ok_or_else(|| format!("--pin {spec:?}: expected ARCH.ROLE=PATH"))?;
        if arch.is_empty() || role.is_empty() || path.is_empty() {
            return Err(format!("--pin {spec:?}: architecture, role and path must all be named"));
        }
        self.pins.entry(arch.to_string()).or_default().insert(role.to_string(), path.to_string());
        Ok(())
    }

    /// The pins of `arch` among `roles`, the roles the caller reads; reading
    /// them is what makes a pin of one of them a pin some model honours.
    pub fn pins(&self, arch: &str, roles: &[&str]) -> BTreeMap<String, String> {
        self.read.lock().unwrap_or_else(|p| p.into_inner()).entry(arch.to_string()).or_default().extend(roles.iter().map(|r| r.to_string()));
        let pinned = self.pins.get(arch);
        roles.iter().filter_map(|r| Some((r.to_string(), pinned?.get(*r)?.clone()))).collect()
    }

    /// The pins `spec` reads: each of its roles and the variant override.
    fn spec_pins(&self, arch: &str, spec: &dyn ArchSpec) -> BTreeMap<String, String> {
        self.pins(arch, &spec.roles().iter().copied().chain([VARIANT]).collect::<Vec<_>>())
    }

    /// Every pin no served model has read, each as the message refusing it:
    /// an architecture nothing resolves names the ones that take pins, and a
    /// role its architecture lacks names the roles it has. The serving
    /// process asks once every model is built, so a mistyped `--pin` is an
    /// error at startup rather than the store's pick served in its place.
    pub fn unread_pins(&self) -> Vec<String> {
        let read = self.read.lock().unwrap_or_else(|p| p.into_inner());
        let mut unread = Vec::new();
        for (arch, roles) in &self.pins {
            for role in roles.keys() {
                match read.get(arch) {
                    Some(known) if known.contains(role) => {}
                    Some(known) => unread.push(format!(
                        "--pin {arch}.{role}: {arch} has no role '{role}' (it reads {})",
                        known.iter().map(String::as_str).collect::<Vec<_>>().join(", ")
                    )),
                    None => unread.push(format!(
                        "--pin {arch}.{role}: no served model reads pins for '{arch}' (pins are read for {})",
                        read.keys().map(String::as_str).collect::<Vec<_>>().join(", ")
                    )),
                }
            }
        }
        unread
    }

    /// `arch`'s [`Assembly`] with its pins applied, or why it has none: the
    /// resolver every caller of the CLI uses, for an adapter that reports its
    /// own failure.
    pub fn resolve(&self, arch: &str, spec: &dyn ArchSpec) -> Result<Assembly, ResolveFailure> {
        try_resolve(self.models_dir(), arch, spec, &self.spec_pins(arch, spec))
    }
}

/// The [`Assembly`] for a model whose every required role is pinned, under
/// `id`; `None` when the store has to be consulted for at least one of them.
fn fully_named(named: &BTreeMap<String, String>, arch: &str, spec: &dyn ArchSpec, id: String) -> Option<Assembly> {
    if !spec.roles().iter().filter(|r| !spec.optional_roles().contains(r)).all(|r| named.contains_key(*r)) {
        return None;
    }
    let pinned_roles = || named.iter().filter(|(role, _)| role.as_str() != VARIANT);
    let roles = pinned_roles().map(|(role, path)| (role.clone(), PathBuf::from(path))).collect();
    let provenance = pinned_roles().map(|(role, path)| format!("{role}: {path} (pinned)")).collect();
    Some(Assembly { id, arch: arch.to_string(), variant: named.get(VARIANT).cloned(), roles, provenance })
}

/// The DIRECTORY an artifact lives in, as the `String` every resident's own
/// constructor takes.
///
/// A pin may name the DIRECTORY holding a released graph, while the
/// resolver's role resolves to the graph FILE itself (the artifact the
/// inventory recorded and the classifier identified). A resident whose loader
/// takes the directory - `ArcFaceSession::load`, `ScrfdSession::load`,
/// PuLID's per-component roots, all of which join a known release filename
/// onto it - needs the same answer from either, so the conversion lives here
/// once rather than in each caller.
///
/// A path that is already a directory is returned unchanged; anything else
/// yields its parent.
pub fn containing_dir(path: &std::path::Path) -> Option<String> {
    let dir = if path.is_dir() { path } else { path.parent()? };
    Some(dir.to_string_lossy().into_owned())
}

/// An [`Ambiguity`] rendered for a daemon's startup log: each real candidate
/// as the `--pin` that would select it. Falls back to the shared CLI
/// rendering for a [`Question::Variant`]/[`Question::Unverifiable`] question,
/// which names no role.
fn describe_served_ambiguity(a: &Ambiguity, arch: &str) -> String {
    let Question::Role { role } = &a.question else { return describe_ambiguity(a) };
    let mut s = format!("more than one candidate for '{role}' and nothing named which - start brain serve with one of:\n");
    for c in &a.choices {
        for (_, value) in &c.selector {
            s.push_str(&format!("  --pin {arch}.{role}={value}\n"));
        }
    }
    s
}

/// Every role `spec` declares, resolved for a served model from `store`.
///
/// Precedence, per role independently:
/// 1. The role's pin, if any - taken verbatim, never re-derived, never
///    checked against the store. An operator who names a path gets exactly
///    that path, including one outside the model store entirely.
/// 2. Otherwise whatever [`resolve_structured`] picks for it, from the same
///    scan/classification the one-shot CLI runs.
///
/// `None` - the model is simply not served, which is never a hard startup
/// failure for the whole daemon - when the store cannot answer:
/// - `Missing`/no models directory: silent. A served model whose weights were
///   never fetched must not take the other models down with it.
/// - `Ambiguous`: the store genuinely holds more than one candidate, so
///   picking one would be a guess. Logged with every real candidate and the
///   pin that selects it (see [`describe_served_ambiguity`]) - the daemon's
///   own startup log being the only place an operator can read it.
///
/// Short-circuits the scan entirely when every REQUIRED role is pinned: a
/// fully configured operator must pay nothing for a resolver they are not
/// using, and must not be refused because the paths they named happen to live
/// outside the store the scan covers.
pub fn served_assembly(store: &ServedStore, arch: &str, spec: &dyn ArchSpec) -> Option<Assembly> {
    let named = store.spec_pins(arch, spec);
    if let Some(assembly) = fully_named(&named, arch, spec, format!("local/{arch}")) {
        return Some(assembly);
    }
    let resolution = match resolve_structured(store.models_dir(), arch, spec, &named) {
        Ok(r) => r,
        // No models directory at all - the resolver was never reached.
        Err(_) => return None,
    };
    let mut assembly = match resolution {
        Resolution::Resolved(assembly) => *assembly,
        Resolution::Ambiguous(a) => {
            residency::log::not_served(&format!("brain: {arch} not served over the scheduler - {}", describe_served_ambiguity(&a, arch)));
            return None;
        }
        Resolution::Missing(_) => return None,
    };
    // A pinned path outranks the resolver's pick for that role even when the
    // resolver also found one - rule 1 above, applied last so it cannot be
    // undone. A pin that names no role (`variant`) is the resolver's own
    // override, already applied.
    for (role, path) in named.into_iter().filter(|(role, _)| spec.roles().contains(&role.as_str())) {
        assembly.provenance.push(format!("{role}: {path} (pinned, overriding the store)"));
        assembly.roles.insert(role, PathBuf::from(path));
    }
    Some(assembly)
}

/// [`served_assembly`]'s multi-instance counterpart: every real, independent
/// candidate of `spec.instance_role()` (see that method's own doc) becomes
/// its OWN served [`Assembly`], addressed by its real vendor/repo id.
/// `default_id` is used instead whenever there is exactly one instance for a
/// reason OTHER than a real, distinct candidate - every role pinned, or the
/// store holding no real instance-role candidate at all - so a
/// fully-configured or single-checkpoint operator keeps the caller's own
/// well-known id (e.g. `flux2::caps::MODEL`), not a resolver-internal
/// placeholder.
///
/// An `Ambiguous`/`Missing` OTHER role on one candidate (a `dit` this store
/// has no matching `vae` size for, say) is logged and that ONE candidate is
/// dropped - it never takes every other real, independently-servable
/// candidate down with it.
pub fn served_assemblies(store: &ServedStore, arch: &str, spec: &dyn ArchSpec, default_id: &str) -> Vec<Assembly> {
    let named = store.spec_pins(arch, spec);
    if let Some(assembly) = fully_named(&named, arch, spec, default_id.to_string()) {
        return vec![assembly];
    }
    let Some(root) = store.models_dir() else { return Vec::new() };
    let records = brain_modelstore::inventory::scan(root);
    let specs: [&dyn ArchSpec; 1] = [spec];
    let placeholder = format!("local/{arch}");
    let mut out = Vec::new();
    for (id, resolution) in brain_modelstore::resolve::resolve_all(arch, &records, &specs, &named) {
        let mut assembly = match resolution {
            Resolution::Resolved(assembly) => *assembly,
            Resolution::Ambiguous(a) => {
                residency::log::not_served(&format!("brain: {id} ({arch}) not served over the scheduler - {}", describe_served_ambiguity(&a, arch)));
                continue;
            }
            Resolution::Missing(_) => continue,
        };
        assembly.id = if id == placeholder { default_id.to_string() } else { id };
        for (role, path) in named.iter().filter(|(role, _)| spec.roles().contains(&role.as_str())) {
            assembly.provenance.push(format!("{role}: {path} (pinned, overriding the store)"));
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

    /// The store at `root` with nothing pinned.
    fn unpinned(root: &Path) -> ServedStore {
        ServedStore::new(Some(root.to_path_buf()))
    }

    /// The store at `root` with its `weights` pinned to `path`.
    fn pinned(root: &Path, path: &Path) -> ServedStore {
        let mut store = unpinned(root);
        store.pin(&format!("servedtest.weights={}", path.display())).unwrap();
        store
    }

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

    /// With nothing pinned, ONE unambiguous candidate in the store is found
    /// on its own.
    #[test]
    fn an_unambiguous_store_candidate_is_found_with_nothing_pinned() {
        let root = store("unambiguous");
        let real = write_candidate(&root, "vendor", "model.safetensors");

        let assembly = served_assembly(&unpinned(&root), "servedtest", &ServedSpec).expect("one candidate must resolve on its own");
        assert_eq!(assembly.roles["weights"], real);

        std::fs::remove_dir_all(&root).ok();
    }

    /// A pin is unconditional: an explicitly named path wins even when
    /// the store holds a real, classifiable candidate that the resolver would
    /// otherwise have picked - and even when the named path is somewhere the
    /// scan does not cover at all. An operator who names a path must get
    /// exactly that path.
    #[test]
    fn a_pin_wins_over_a_different_store_candidate() {
        let root = store("override");
        let in_store = write_candidate(&root, "vendor", "model.safetensors");
        // Deliberately OUTSIDE the store, so this can only come from the
        // pin - a scan-based answer could never produce it.
        let outside = store("override-elsewhere").join("hand-placed.safetensors");
        std::fs::create_dir_all(outside.parent().unwrap()).unwrap();
        std::fs::write(&outside, b"not even a real checkpoint").unwrap();

        let assembly = served_assembly(&pinned(&root, &outside), "servedtest", &ServedSpec).expect("an explicitly named path must always resolve");
        assert_eq!(assembly.roles["weights"], outside, "the pin must win over the store's own candidate");
        assert_ne!(assembly.roles["weights"], in_store);

        std::fs::remove_dir_all(&root).ok();
        std::fs::remove_dir_all(outside.parent().unwrap()).ok();
    }

    /// Two equally good candidates and nothing naming one is a genuine
    /// question, so the model is NOT served - never a silent pick of whichever
    /// the directory walk reached first, and never a crash. The refusal names
    /// every real candidate and the pin that selects it, since a daemon's
    /// startup log is the only place its operator can read it.
    #[test]
    fn genuine_ambiguity_refuses_to_serve_and_names_every_candidate() {
        let root = store("ambiguous");
        let a = write_candidate(&root, "vendor-a", "model.safetensors");
        let b = write_candidate(&root, "vendor-b", "model.safetensors");

        assert!(served_assembly(&unpinned(&root), "servedtest", &ServedSpec).is_none(), "two real candidates must not be silently collapsed to one");

        // The message an operator actually gets: both paths, each as the
        // assignment that would select it.
        let records = brain_modelstore::inventory::scan(&root);
        let specs: [&dyn ArchSpec; 1] = [&ServedSpec];
        let Resolution::Ambiguous(amb) = brain_modelstore::resolve::resolve("servedtest", &records, &specs, &BTreeMap::new()) else {
            panic!("expected the two candidates to be ambiguous");
        };
        let msg = describe_served_ambiguity(&amb, "servedtest");
        assert!(msg.contains("--pin servedtest.weights="), "the message must name the pin to give: {msg}");
        assert!(msg.contains(a.to_str().unwrap()), "{msg}");
        assert!(msg.contains(b.to_str().unwrap()), "{msg}");

        // ...and naming one of them resolves it, which is what makes the
        // message actionable rather than merely informative.
        let assembly = served_assembly(&pinned(&root, &a), "servedtest", &ServedSpec).expect("naming one candidate must resolve the ambiguity");
        assert_eq!(assembly.roles["weights"], a);

        std::fs::remove_dir_all(&root).ok();
    }

    /// Nothing configured and nothing found is "not served", quietly - never a
    /// hard failure. A daemon serves ~30 models; one whose weights were never
    /// fetched must not take the other 29 down with it.
    #[test]
    fn nothing_configured_and_nothing_found_is_silently_not_served() {
        let root = store("empty");
        assert!(served_assembly(&unpinned(&root), "servedtest", &ServedSpec).is_none());
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
        let root = store("multi-instance");
        let a = write_candidate(&root, "vendor-a", "model.safetensors");
        let b = write_candidate(&root, "vendor-b", "model.safetensors");

        let mut got = served_assemblies(&unpinned(&root), "servedtest", &MultiServedSpec, "default/servedtest");
        got.sort_by(|x, y| x.id.cmp(&y.id));
        assert_eq!(got.len(), 2, "{:?}", got.iter().map(|a| &a.id).collect::<Vec<_>>());
        assert_eq!(got[0].id, "vendor-a/model");
        assert_eq!(got[0].roles["weights"], a);
        assert_eq!(got[1].id, "vendor-b/model");
        assert_eq!(got[1].roles["weights"], b);

        std::fs::remove_dir_all(&root).ok();
    }

    /// Every role pinned still collapses to
    /// exactly ONE instance, under `default_id` - never fanned out into
    /// per-candidate instances just because the spec opted its role in.
    #[test]
    fn served_assemblies_uses_default_id_when_every_role_is_pinned() {
        let root = store("multi-instance-named");
        let named = write_candidate(&root, "vendor", "model.safetensors");

        let got = served_assemblies(&pinned(&root, &named), "servedtest", &MultiServedSpec, "default/servedtest");
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].id, "default/servedtest");
        assert_eq!(got[0].roles["weights"], named);

        std::fs::remove_dir_all(&root).ok();
    }

    /// An empty store falls back to `resolve_all`'s own single-instance
    /// path, which resolves to `Missing` exactly as `served_assembly`'s own
    /// "nothing configured and nothing found" case does - silently not
    /// served, never a hard failure.
    #[test]
    fn served_assemblies_serves_nothing_when_the_store_holds_nothing() {
        let root = store("multi-instance-empty");

        assert!(served_assemblies(&unpinned(&root), "servedtest", &MultiServedSpec, "default/servedtest").is_empty(), "nothing on disk must serve nothing");

        std::fs::remove_dir_all(&root).ok();
    }

    /// `containing_dir` is what lets a resident take either source: a pin
    /// may name the DIRECTORY, the resolver's role names the FILE inside it,
    /// and the loader wants the directory in both cases.
    #[test]
    fn containing_dir_normalizes_a_file_role_and_a_directory_pin_to_the_same_answer() {
        let root = store("containing");
        let repo = root.join("DIAMONIK7777").join("antelopev2");
        std::fs::create_dir_all(&repo).unwrap();
        let graph = repo.join("glintr100.onnx");
        std::fs::write(&graph, b"x").unwrap();
        assert_eq!(containing_dir(&graph), containing_dir(&repo), "a role's file and an operator's directory must agree");
        assert_eq!(containing_dir(&repo).as_deref(), repo.to_str());
        std::fs::remove_dir_all(&root).ok();
    }

    /// A pin that names no role of the spec (`variant`) reaches the
    /// resolver as its override and never becomes a role of the assembly.
    #[test]
    fn a_pin_that_names_no_role_is_not_a_role() {
        let root = store("non-role-pin");
        let real = write_candidate(&root, "vendor", "model.safetensors");
        let mut store = unpinned(&root);
        store.pin("servedtest.variant=whatever").unwrap();
        let assembly = served_assembly(&store, "servedtest", &ServedSpec).expect("the one candidate resolves");
        assert_eq!(assembly.roles.keys().collect::<Vec<_>>(), ["weights"]);
        assert_eq!(assembly.roles["weights"], real);
        std::fs::remove_dir_all(&root).ok();
    }

    /// `--pin ARCH.ROLE=PATH` pins one role of one architecture, a later pin
    /// of the same role wins, and a malformed one is refused by name.
    #[test]
    fn a_pin_names_one_role_of_one_architecture() {
        let mut store = ServedStore::new(None);
        store.pin("flux2.dit=/a").unwrap();
        store.pin("flux2.dit=/b").unwrap();
        store.pin("flux2.vae=/c=d").unwrap();
        assert_eq!(store.pins("flux2", &["dit", "vae"]), BTreeMap::from([("dit".to_string(), "/b".to_string()), ("vae".to_string(), "/c=d".to_string())]));
        assert!(store.pins("wan", &["dit"]).is_empty());
        for bad in ["flux2.dit", "flux2=/a", ".dit=/a", "flux2.=/a", "flux2.dit="] {
            assert!(store.pin(bad).unwrap_err().contains(bad), "{bad}");
        }
    }

    /// A pin no served model reads is refused by name instead of silently
    /// serving the store's pick: an unknown architecture names the ones that
    /// take pins, an unknown role names the roles its architecture has, and a
    /// clone of the store shares what was read.
    #[test]
    fn a_pin_nothing_reads_is_reported_with_what_would_be_read() {
        let root = store("unread-pins");
        let mut store = unpinned(&root);
        for spec in ["servedtest.weights=/w", "servedtest.variant=v", "servedtest.wieghts=/typo", "qwen3.weights=/q", "horizon.train=/t"] {
            store.pin(spec).unwrap();
        }
        assert_eq!(store.unread_pins().len(), 5, "nothing has been resolved yet");
        let _ = served_assembly(&store.clone(), "servedtest", &ServedSpec);
        let _ = store.pins("horizon", &["weights", "train"]);
        let unread = store.unread_pins();
        assert_eq!(unread.len(), 2, "{unread:?}");
        assert!(unread[0].contains("--pin qwen3.weights") && unread[0].contains("horizon, servedtest"), "{}", unread[0]);
        assert!(unread[1].contains("--pin servedtest.wieghts") && unread[1].contains("variant, weights"), "{}", unread[1]);
        std::fs::remove_dir_all(&root).ok();
    }
}
