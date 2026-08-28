// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The resolution ladder: given a [`ModelRef`], decide what has to happen
//! before it can be served. Pure with respect to the filesystem beyond
//! [`Store::local`] lookups, and pure with respect to the network beyond
//! [`Hub`] calls -- so it is fully unit-tested against [`FakeHub`], with no
//! server and no real disk layout beyond a scratch dir.

use brain_modelref::{ModelRef, Quant};

use crate::hub::{Hub, HubError};
use crate::recipe::{quant_of_gguf, ArtifactRecipe};
use crate::{LocalModel, Store};

/// One thing [`plan`] decided has to happen. `modelstore` executes
/// [`Step::Download`] itself ([`execute`]); [`Step::Convert`] and
/// [`Step::Quantize`] are returned for the caller to run, since converting a
/// foreign checkpoint needs a model crate's importer and quantizing needs
/// `crates/checkpoint`'s writer -- neither of which this crate depends on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Step {
    /// `reference` already resolves to bytes on disk; nothing to fetch.
    Serve,
    /// Fetch `file` from `vendor/repo@revision` into the store, under
    /// `dest_name` inside that ref's repo directory.
    Download { vendor: String, repo: String, revision: String, file: String, dest_name: String },
    /// Make the just-downloaded upstream checkpoint servable for
    /// `vendor/repo` -- a tensor-format rewrite into `model.brain.safetensors`
    /// for a single-file family, a manifest write for a compound one, etc.
    /// `recipe` is the [`crate::recipe::ArtifactRecipe::id`] that planned
    /// this repo's artifacts, so the finish-side dispatcher
    /// (`crates/cli/src/supply.rs::convert`) knows which family it's
    /// finishing without re-deriving it from disk a second time.
    Convert { vendor: String, repo: String, recipe: &'static str },
    /// Locally quantize `base` (already on disk after its own plan) to `quant`.
    Quantize { base: ModelRef, quant: Quant },
}

/// The ordered list of [`Step`]s that materializes `reference`. Steps execute
/// in order -- a `Quantize` step's `base` was made servable by the steps
/// before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub reference: ModelRef,
    pub steps: Vec<Step>,
}

/// Why a [`Plan`] could not be built.
///
/// Three of the four variants carry a [`ModelRef`], which makes the enum wide
/// enough that returning it unboxed would bloat every `Result<Plan, _>` on the
/// success path too. Planning is a once-per-model, network-bound operation, so
/// the allocation an error path pays is free relative to the HTTP call that
/// produced it: every fallible entry point here returns `Box<PlanError>`, the
/// same way [`recipe::Recipe::artifacts`](crate::recipe) does.
#[derive(Debug)]
pub enum PlanError {
    /// `reference`'s vendor is reserved (`brain/`, `local/`, `test/`) and it
    /// is not on disk -- reserved vendors are never fetched, by design (this
    /// is what keeps discovery endpoints network-free).
    NotFetchable(ModelRef),
    /// No upstream artifact and no viable local-quantize path.
    NoUpstreamArtifact(ModelRef, String),
    /// The base repo's declared architecture is not one brain can load.
    UnsupportedArchitecture(ModelRef, String),
    Hub(HubError),
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanError::NotFetchable(r) => write!(f, "{r}: reserved vendor, not on disk"),
            PlanError::NoUpstreamArtifact(r, why) => write!(f, "{r}: {why}"),
            PlanError::UnsupportedArchitecture(r, arch) => write!(f, "{r}: unsupported architecture {arch:?}"),
            PlanError::Hub(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PlanError {}

pub(crate) const REVISION: &str = "main";

/// Decide how to materialize `reference`: on disk (serve), an existing
/// upstream quantized artifact (download), or the base checkpoint plus a
/// local quantize step (recurse + append).
pub fn plan(reference: &ModelRef, store: &Store, hub: &dyn Hub) -> Result<Plan, Box<PlanError>> {
    plan_at(reference, None, store, hub)
}

/// [`plan`] at a caller-named `revision` (a branch, tag or sha). `None` is
/// the repo's default branch, which is what [`plan`] passes and what every
/// caller that has no revision in hand wants.
///
/// The revision reaches here from a pasted URL's own `/tree/<rev>` or
/// `/blob/<rev>/...` segment (`crate::refurl`). The store keys a repo
/// directory by `vendor/repo` alone, so fetching a non-default revision
/// REPLACES the default one's files in that directory rather than living
/// beside them -- explicit, and the same thing `git checkout` of another
/// branch into one working tree does.
pub fn plan_at(reference: &ModelRef, revision: Option<&str>, store: &Store, hub: &dyn Hub) -> Result<Plan, Box<PlanError>> {
    // Naming the default branch is naming no revision: `/tree/main` and the
    // bare repo URL are the same request, so they must take the same
    // already-on-disk fast path rather than one of them re-listing the repo.
    let revision = revision.filter(|r| *r != REVISION);
    if revision.is_none() && store.local(reference).is_some() {
        return Ok(Plan { reference: reference.clone(), steps: vec![Step::Serve] });
    }
    if reference.is_reserved() {
        return Err(Box::new(PlanError::NotFetchable(reference.clone())));
    }
    let revision = revision.unwrap_or(REVISION);
    match reference.quant() {
        Some(q) => plan_quant(reference, q, revision, store, hub),
        None => plan_base(reference, revision, store, hub),
    }
}

/// The plan for ONE named artifact inside a repo -- what a pasted
/// `/blob/<rev>/<path>` or `/resolve/<rev>/<path>` URL asks for.
///
/// This is the shape that needs no policy at all: the file is named, so
/// nothing is inferred, no quantization ladder runs, and any extension works
/// (a single `.safetensors` component is as valid a target as a `.gguf` --
/// `brain flux2 generate --text-encoder <file>` takes exactly such a file).
/// A GGUF whose name declares a quantization is the one exception, and it is
/// a naming one rather than a policy one: it lands under the store's own
/// `<QUANT>.gguf` destination so that pulling it by URL and pulling it as
/// `<repo>-<QUANT>` are the same artifact in the same place, not two copies.
pub fn plan_file(reference: &ModelRef, file: &str, revision: Option<&str>, hub: &dyn Hub) -> Result<Plan, Box<PlanError>> {
    if reference.is_reserved() {
        return Err(Box::new(PlanError::NotFetchable(reference.clone())));
    }
    let vendor = reference.vendor();
    let repo = reference.repo();
    let revision = revision.unwrap_or(REVISION);
    let listing = hub.list_files(vendor, repo, revision).map_err(|e| Box::new(PlanError::Hub(e)))?;
    if !listing.iter().any(|f| f == file) {
        return Err(Box::new(PlanError::NoUpstreamArtifact(reference.clone(), format!("no file {file:?} at revision {revision:?}; it holds {}", listed(&listing)))));
    }
    // A quantization token in the name is the store's own destination
    // convention; anything else keeps its own repo-relative path, nested
    // directories included (`execute` creates them, the same way the
    // diffusers-pipeline recipe's role directories already rely on).
    let (reference, dest) = match quant_of_gguf(file).filter(|_| !file.contains('/')) {
        Some(q) => (reference.with_quant(q), format!("{}.gguf", q.as_str())),
        None => (reference.clone(), file.to_string()),
    };
    let mut steps = vec![Step::Download {
        vendor: vendor.to_string(),
        repo: repo.to_string(),
        revision: revision.to_string(),
        file: file.to_string(),
        dest_name: dest.clone(),
    }];
    // A GGUF gets the same finish step a whole-repo GGUF pull gets: its
    // header is read back off disk and its `general.architecture` reported.
    // That check cannot happen before the download -- see `Hub`, which has no
    // range request to read a header with.
    if dest.ends_with(".gguf") {
        steps.push(Step::Convert { vendor: vendor.to_string(), repo: repo.to_string(), recipe: "gguf" });
    }
    Ok(Plan { reference, steps })
}

/// A repo's file listing, rendered for an error a human reads. Capped: a
/// listing is occasionally hundreds of entries, and a wall of them buries
/// the sentence above it.
fn listed(listing: &[String]) -> String {
    const MAX: usize = 30;
    let shown: Vec<&str> = listing.iter().take(MAX).map(String::as_str).collect();
    match listing.len().checked_sub(MAX) {
        Some(rest) if rest > 0 => format!("{} and {rest} more", shown.join(", ")),
        _ => shown.join(", "),
    }
}

fn plan_quant(reference: &ModelRef, quant: Quant, revision: &str, store: &Store, hub: &dyn Hub) -> Result<Plan, Box<PlanError>> {
    // The repo the user NAMED gets first refusal. When it is itself a GGUF
    // release, `-<QUANT>` selects a file inside it, and the naming ladder
    // below -- which guesses at SIBLING repos (`<repo>-GGUF`) -- must not run
    // at all: it would miss (upstream file names are not `<repo>-<QUANT>`),
    // then fall through to fetching a base checkpoint to quantize locally,
    // which for a repo that already publishes the exact file asked for is
    // both wrong and enormous.
    if let Ok(listing) = hub.list_files(reference.vendor(), reference.repo(), revision) {
        if crate::recipe::GgufRecipe.matches(reference, &listing) {
            return plan_from_listing(reference, &listing, revision, hub);
        }
    }
    // A revision the caller named applies to the repo it named, not to a
    // DIFFERENT repo guessed at below -- `<repo>-GGUF`'s branches have
    // nothing to do with `<repo>`'s -- so the ladder stays on the default.
    for (vendor, repo, file) in quant_candidates(reference, quant) {
        match hub.list_files(&vendor, &repo, REVISION) {
            Ok(files) if files.iter().any(|f| f == &file) => {
                return Ok(Plan {
                    reference: reference.clone(),
                    steps: vec![Step::Download {
                        vendor,
                        repo,
                        revision: REVISION.to_string(),
                        file,
                        dest_name: format!("{}.gguf", quant.as_str()),
                    }],
                });
            }
            // Repo doesn't exist, or exists but doesn't hold this file: try
            // the next candidate. A hard hub error (e.g. rate limited) is
            // swallowed here too -- the local-quantize fallback below is
            // always a valid answer, so one candidate's transient failure
            // must not fail the whole ladder.
            _ => continue,
        }
    }
    let base = reference.base();
    let mut base_plan = plan_base(&base, revision, store, hub)?;
    base_plan.reference = reference.clone();
    base_plan.steps.push(Step::Quantize { base, quant });
    Ok(base_plan)
}

/// A `(vendor, repo)` publishing its own GGUF quants of some upstream base
/// checkpoint under a *different* vendor/repo name -- the same-vendor
/// conventions in [`quant_candidates`] can't express this, since the
/// quantizer (e.g. `bartowski`, `unsloth`) is not the model's own org. Add a
/// row here when a model's own org doesn't ship GGUF and no same-vendor
/// convention matches; this is the *only* place such overrides live.
struct GgufVendorOverride {
    vendor: &'static str,
    repo: &'static str,
    gguf_vendor: &'static str,
    gguf_repo: &'static str,
}

const GGUF_VENDOR_OVERRIDES: &[GgufVendorOverride] = &[
    // Qwen3.5-35B-A3B ships no GGUF of its own; bartowski's quants are the
    // ones this repo has fetched and validated against.
    GgufVendorOverride {
        vendor: "Qwen",
        repo: "Qwen3.5-35B-A3B",
        gguf_vendor: "bartowski",
        gguf_repo: "Qwen_Qwen3.5-35B-A3B-GGUF",
    },
];

/// Upstream naming conventions for a pre-quantized artifact, tried in order.
/// This list (plus [`GGUF_VENDOR_OVERRIDES`] for cross-vendor quantizers) is
/// the *only* place those conventions live.
fn quant_candidates(reference: &ModelRef, quant: Quant) -> Vec<(String, String, String)> {
    let repo = reference.repo();
    let q = quant.as_str();
    let mut out = Vec::new();
    if let Some(o) = GGUF_VENDOR_OVERRIDES.iter().find(|o| o.vendor == reference.vendor() && o.repo == repo) {
        out.push((o.gguf_vendor.to_string(), o.gguf_repo.to_string(), format!("{repo}-{q}.gguf")));
    }
    out.extend([
        // The common `<repo>-GGUF` sibling repo, one file per quant.
        (reference.vendor().to_string(), format!("{repo}-GGUF"), format!("{repo}-{q}.gguf")),
        // Some repos ship a gguf alongside their own safetensors.
        (reference.vendor().to_string(), repo.to_string(), format!("{repo}-{q}.gguf")),
        // A few publish one repo per quant level.
        (reference.vendor().to_string(), format!("{repo}-{q}"), format!("{repo}-{q}.gguf")),
    ]);
    out
}

fn plan_base(reference: &ModelRef, revision: &str, store: &Store, hub: &dyn Hub) -> Result<Plan, Box<PlanError>> {
    if revision == REVISION {
        if let Some(local) = store.local(reference) {
            return Ok(Plan { reference: reference.clone(), steps: local_serve_steps(&local) });
        }
    }
    let listing = hub.list_files(reference.vendor(), reference.repo(), revision).map_err(|e| Box::new(PlanError::Hub(e)))?;
    plan_from_listing(reference, &listing, revision, hub)
}

/// The recipe half of [`plan_base`], over a listing the caller already has --
/// so the quant path can hand its own listing straight over instead of
/// asking the hub for it a second time.
fn plan_from_listing(reference: &ModelRef, listing: &[String], revision: &str, hub: &dyn Hub) -> Result<Plan, Box<PlanError>> {
    let vendor = reference.vendor();
    let repo = reference.repo();
    let recipe = crate::recipe::recipes()
        .into_iter()
        .find(|r| r.matches(reference, listing))
        .expect("the last recipe in the registry is a catch-all and always matches");
    let artifacts = recipe.artifacts(reference, listing, hub)?;
    // A recipe that CHOSE between interchangeable artifacts says so, and the
    // choice rides on the plan's reference: that is what the front end prints
    // and what `Store::local` resolves the finished download by.
    let resolved = match recipe.resolved_quant(reference, listing) {
        Some(q) => reference.with_quant(q),
        None => reference.clone(),
    };

    let mut steps: Vec<Step> = artifacts.into_iter().map(|a| download_step(vendor, repo, revision, &a.file, &a.dest_name)).collect();
    steps.push(Step::Convert { vendor: vendor.to_string(), repo: repo.to_string(), recipe: recipe.id() });
    Ok(Plan { reference: resolved, steps })
}

fn local_serve_steps(_local: &LocalModel) -> Vec<Step> {
    vec![Step::Serve]
}

fn download_step(vendor: &str, repo: &str, revision: &str, file: &str, dest_name: &str) -> Step {
    Step::Download {
        vendor: vendor.to_string(),
        repo: repo.to_string(),
        revision: revision.to_string(),
        file: file.to_string(),
        dest_name: dest_name.to_string(),
    }
}

/// The HF `config.json` architecture string: `architectures[0]` (the
/// standard `transformers` field) or `model_type` as a fallback. Public so a
/// Convert-step dispatcher (`crates/cli/src/supply.rs`) can re-derive the
/// same architecture string `plan()` gated on, rather than re-parsing
/// `config.json` with different logic.
pub fn declared_architecture(config: &serde_json::Value) -> Option<String> {
    config
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|v| v.as_str())
        .or_else(|| config.get("model_type").and_then(|v| v.as_str()))
        .map(str::to_string)
}

/// The canonical `brain_arch` architecture id an HF `architecture` string
/// maps to, or `None` if unsupported. Exact match against
/// [`brain_arch::Arch::hf`] -- no substring/prefix scan, so a class name that
/// happens to CONTAIN another architecture's id as a substring (Qwen3-Omni's
/// real HF class name is `Qwen3OmniMoeForConditionalGeneration`, which
/// contains `"qwen"`) cannot be mis-routed regardless of table order. Public
/// so the Convert-step dispatcher (`crates/cli/src/supply.rs`) picks the
/// exact same id `plan()` already gated the fetch on -- one implementation of
/// "which architectures brain can fetch today", not two.
///
/// Narrower than the substring scan it replaces in one respect: `Arch::hf`
/// carries `declared_architecture`'s `model_type` fallback spelling only for
/// architectures that document it (`qwen3`'s `"qwen3"` row); a repo lacking
/// `architectures[0]` whose `model_type` isn't registered will not match,
/// where the old substring scan might have. Add the real `model_type` value
/// to that architecture's `Arch::hf` row when a repo needs it.
pub fn family_of_architecture(arch: &str) -> Option<&'static str> {
    brain_arch::by_hf(arch).map(|a| a.id)
}

pub(crate) fn is_supported_architecture(arch: &str) -> bool {
    family_of_architecture(arch).is_some()
}

/// Runs every [`Step::Download`] in `plan.steps` against `hub`, writing into
/// `store`'s `<vendor>/<repo>` directory for `plan.reference`'s base repo.
/// Returns the steps this crate could not execute (`Convert`/`Quantize`), in
/// order, for the caller to run next.
///
/// **Idempotent per file.** A [`Step::Download`] whose `dest` already exists is
/// skipped. This is what makes a killed fetch restartable: [`plan_base`] always
/// returns the FULL artifact list for a family, and the one early-out
/// [`plan`] has (`store.local(reference)`) only fires once EVERY role/file is
/// present -- so without this, a multi-shard repo interrupted partway through
/// re-downloads every shard it already landed. Confirmed live: a killed
/// `qwen3vl` fetch re-fetched an already-complete 4.97 GB shard from scratch.
///
/// Bare existence is the correct test here, not a size or checksum comparison:
/// [`crate::fetch::stream_to_file`] writes to a `.part` sibling and renames
/// into place **only on full success**, so a file at `dest` is by construction
/// a download that completed. A partial transfer leaves a `.part` file, which
/// this never mistakes for the real one. That invariant is load-bearing for
/// the skip -- if `stream_to_file` ever writes `dest` incrementally, this
/// check must become a size/digest comparison in the same change.
///
/// The skip does not re-verify the revision, matching what `store.local` (the
/// existing early-out) already does: a file under `<vendor>/<repo>/` is taken
/// as that reference's copy. Removing the file is how a caller forces a
/// re-fetch.
pub fn execute(store: &Store, hub: &dyn Hub, plan: &Plan, progress: &mut dyn FnMut(&str, u64, Option<u64>)) -> Result<Vec<Step>, HubError> {
    let dir = store.repo_dir(&plan.reference.base());
    let mut deferred = Vec::new();
    for step in &plan.steps {
        match step {
            Step::Serve => {}
            Step::Download { vendor, repo, revision, file, dest_name } => {
                let dest = dir.join(dest_name);
                if dest.exists() {
                    continue;
                }
                hub.download(vendor, repo, revision, file, &dest, &mut |got, total| progress(dest_name, got, total))?;
            }
            Step::Convert { .. } | Step::Quantize { .. } => deferred.push(step.clone()),
        }
    }
    Ok(deferred)
}

/// What a [`Plan`] still has to pull off the network: how many files, how
/// many bytes, and whether the host disclosed a size for every one of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Remaining {
    /// Files [`execute`] will actually download (already-present ones excluded).
    pub files: usize,
    /// Their total size, summed over the files the host reported a size for.
    pub bytes: u64,
    /// False when at least one remaining file had no size from the host, so
    /// `bytes` is a floor rather than the total. A progress budget spread
    /// over `bytes` is only exact when this is true.
    pub sizes_known: bool,
}

/// Size `plan`'s outstanding work WITHOUT downloading it, so a caller can
/// show a whole-pull progress bar rather than a per-file one.
///
/// Applies exactly [`execute`]'s own skip rule -- a `dest` that already
/// exists is a completed download and is excluded from both counts -- so a
/// resumed pull is sized over what is actually left, and the two cannot
/// drift apart. A file the host reports no size for still counts toward
/// `files` but contributes no bytes, and clears `sizes_known`.
///
/// One [`Hub::file_sizes`] call per distinct `(vendor, repo, revision)` in
/// the plan, not one per file.
pub fn remaining_download(store: &Store, hub: &dyn Hub, plan: &Plan) -> Result<Remaining, HubError> {
    let dir = store.repo_dir(&plan.reference.base());
    let mut sizes: std::collections::HashMap<(String, String, String), std::collections::BTreeMap<String, u64>> = std::collections::HashMap::new();
    let mut out = Remaining { files: 0, bytes: 0, sizes_known: true };
    for step in &plan.steps {
        let Step::Download { vendor, repo, revision, file, dest_name } = step else { continue };
        if dir.join(dest_name).exists() {
            continue;
        }
        out.files += 1;
        let key = (vendor.clone(), repo.clone(), revision.clone());
        let table = match sizes.entry(key) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => e.insert(hub.file_sizes(vendor, repo, revision)?),
        };
        match table.get(file) {
            Some(n) => out.bytes += n,
            None => out.sizes_known = false,
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hub::FakeHub;

    fn store(tmp_name: &str) -> Store {
        let dir = std::env::temp_dir().join(tmp_name);
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        Store::new(dir)
    }

    fn write_base_fixture(st: &Store, vendor: &str, repo: &str) {
        let dir = st.repo_dir(&ModelRef::new(vendor, repo, None));
        std::fs::create_dir_all(&dir).unwrap();
        let card = checkpoint::st::ModelCard::new(format!("{vendor}/{repo}"), "qwen");
        checkpoint::st::save_safetensors(
            dir.join("model.brain.safetensors").to_str().unwrap(),
            &[("weight".to_string(), vec![2], vec![1.0, 2.0])],
            &serde_json::json!({"hidden_size": 8}),
            Some(&card),
        )
        .unwrap();
    }

    /// A [`Hub`] that answers everything except sizes -- the shape of a host
    /// that lists a repo but discloses no `size` for its files.
    struct NoSizes(FakeHub);

    impl Hub for NoSizes {
        fn resolve_revision(&self, v: &str, r: &str, rev: &str) -> Result<String, HubError> {
            self.0.resolve_revision(v, r, rev)
        }
        fn list_files(&self, v: &str, r: &str, rev: &str) -> Result<Vec<String>, HubError> {
            self.0.list_files(v, r, rev)
        }
        fn file_sizes(&self, v: &str, r: &str, rev: &str) -> Result<std::collections::BTreeMap<String, u64>, HubError> {
            self.0.file_sizes(v, r, rev)?;
            Ok(Default::default())
        }
        fn read_file(&self, v: &str, r: &str, rev: &str, f: &str) -> Result<Vec<u8>, HubError> {
            self.0.read_file(v, r, rev, f)
        }
        fn download(&self, v: &str, r: &str, rev: &str, f: &str, dest: &std::path::Path, p: &mut dyn FnMut(u64, Option<u64>)) -> Result<(), HubError> {
            self.0.download(v, r, rev, f, dest, p)
        }
    }

    const CONFIG: &[u8] = br#"{"architectures":["Qwen3ForCausalLM"]}"#;

    fn sharded_hub() -> FakeHub {
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", CONFIG.to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors.index.json", vec![0u8; 100]);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model-00001-of-00002.safetensors", vec![1u8; 4000]);
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model-00002-of-00002.safetensors", vec![2u8; 6000]);
        hub
    }

    /// Sizing a plan sums every outstanding file, and applies exactly
    /// `execute`'s own skip rule -- so a resumed pull is measured over what is
    /// LEFT. If these two ever disagree, a progress bar either finishes early
    /// or never reaches the end.
    #[test]
    fn remaining_download_sums_the_plan_and_excludes_what_is_already_on_disk() {
        let st = store("modelstore-remaining-test-skip");
        let hub = sharded_hub();
        let reference = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let p = plan(&reference, &st, &hub).unwrap();

        let total: u64 = CONFIG.len() as u64 + 100 + 4000 + 6000;
        let all = remaining_download(&st, &hub, &p).unwrap();
        assert_eq!(all, Remaining { files: 4, bytes: total, sizes_known: true });

        // One shard already landed: both counts drop by exactly that file.
        let dir = st.repo_dir(&reference.base());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), vec![2u8; 6000]).unwrap();
        let left = remaining_download(&st, &hub, &p).unwrap();
        assert_eq!(left, Remaining { files: 3, bytes: total - 6000, sizes_known: true });

        // And `execute` agrees: it downloads three files, not four. Proven by
        // the shard's bytes staying as written rather than being re-fetched.
        std::fs::write(dir.join("model-00002-of-00002.safetensors"), b"sentinel").unwrap();
        let mut seen: Vec<String> = Vec::new();
        execute(&st, &hub, &p, &mut |name, _, _| {
            if seen.last().map(String::as_str) != Some(name) {
                seen.push(name.to_string());
            }
        })
        .unwrap();
        assert!(!seen.iter().any(|n| n == "model-00002-of-00002.safetensors"), "a completed file must not be re-downloaded: {seen:?}");
        assert_eq!(std::fs::read(dir.join("model-00002-of-00002.safetensors")).unwrap(), b"sentinel");
    }

    /// A host that discloses no size still yields a usable file COUNT, and
    /// says the byte total is a floor rather than reporting a confident zero
    /// -- which a caller would otherwise divide by.
    #[test]
    fn remaining_download_flags_a_host_that_withholds_sizes() {
        let st = store("modelstore-remaining-test-nosize");
        let hub = NoSizes(sharded_hub());
        let reference = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let p = plan(&reference, &st, &hub).unwrap();
        assert_eq!(remaining_download(&st, &hub, &p).unwrap(), Remaining { files: 4, bytes: 0, sizes_known: false });
    }

    #[test]
    fn omni_architecture_does_not_fall_through_to_qwen() {
        // Qwen3-Omni's real HF class name contains "qwen" as a substring
        // ("Qwen3OmniMoeForConditionalGeneration"), so a naive first-match
        // substring scan checking "qwen" before "omni" would silently route
        // it to the dense qwen3 importer. Exact matching against
        // brain_arch::Arch::hf makes that class of bug structurally
        // impossible regardless of table order.
        assert_eq!(family_of_architecture("Qwen3OmniMoeForConditionalGeneration"), Some("qwen3omnimoe"));
        // Plain dense Qwen3 is unaffected.
        assert_eq!(family_of_architecture("Qwen3ForCausalLM"), Some("qwen3"));
        // A near-miss (not an exact registered class name) is correctly
        // unsupported, not fuzzily routed to the nearest substring match --
        // brain's qwen3 crate is dense-only, so "Qwen3MoeForCausalLM" (the
        // sparse variant) has no importer to route to yet.
        assert_eq!(family_of_architecture("Qwen3MoeForCausalLM"), None);
        assert_eq!(family_of_architecture("qwen3_omni_moe"), None);
    }

    #[test]
    fn on_disk_ref_plans_to_serve_with_no_hub_calls() {
        // A hub with nothing registered would error on any call, so a Serve
        // plan proves plan() never touched it.
        let st = store("modelstore-plan-test-serve");
        write_base_fixture(&st, "Qwen", "Qwen3-0.6B");
        let hub = FakeHub::new();
        let r = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let p = plan(&r, &st, &hub).unwrap();
        assert_eq!(p, Plan { reference: r, steps: vec![Step::Serve] });
    }

    #[test]
    fn reserved_vendor_not_on_disk_is_not_fetchable() {
        let st = store("modelstore-plan-test-reserved");
        let hub = FakeHub::new();
        let reserved = ModelRef::new("brain", "mock", None);
        let err = plan(&reserved, &st, &hub).unwrap_err();
        assert!(matches!(*err, PlanError::NotFetchable(_)));
    }

    #[test]
    fn non_reserved_ref_with_no_hub_entry_and_nothing_on_disk_errors() {
        let st = store("modelstore-plan-test-missing");
        let hub = FakeHub::new();
        let r = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let err = plan(&r, &st, &hub).unwrap_err();
        assert!(matches!(*err, PlanError::Hub(HubError::NotFound(_))));
    }

    #[test]
    fn base_ref_with_single_safetensors_file_plans_config_tokenizer_weights_convert() {
        let st = store("modelstore-plan-test-base-single");
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", br#"{"architectures":["Qwen3ForCausalLM"]}"#.to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "tokenizer.json", b"{}".to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", vec![0u8; 8]);

        let r = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let p = plan(&r, &st, &hub).unwrap();
        assert_eq!(p.reference, r);
        assert_eq!(
            p.steps,
            vec![
                download_step("Qwen", "Qwen3-0.6B", REVISION, "config.json", "config.json"),
                download_step("Qwen", "Qwen3-0.6B", REVISION, "tokenizer.json", "tokenizer.json"),
                download_step("Qwen", "Qwen3-0.6B", REVISION, "model.safetensors", "model.safetensors"),
                Step::Convert { vendor: "Qwen".to_string(), repo: "Qwen3-0.6B".to_string(), recipe: "transformers" },
            ]
        );
    }

    #[test]
    fn base_ref_for_a_diffusers_pipeline_repo_routes_to_the_zimage_recipe() {
        // A Z-Image-shaped repo has no root config.json, so it must NOT fall
        // through to TransformersRecipe's "no config.json in repo" error --
        // it needs no extra Hub::read_file call either (unlike transformers'
        // config.json gate), so an empty FakeHub with only list_files
        // registered (via add_file's directory side effect) is enough.
        let st = store("modelstore-plan-test-zimage-base");
        let mut hub = FakeHub::new();
        for f in [
            "model_index.json",
            "transformer/config.json",
            "transformer/diffusion_pytorch_model.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/config.json",
            "text_encoder/model.safetensors",
            "tokenizer/tokenizer.json",
        ] {
            hub.add_file("Tongyi-MAI", "Z-Image-Turbo", "main", f, b"stub".to_vec());
        }

        let r = ModelRef::new("Tongyi-MAI", "Z-Image-Turbo", None);
        let p = plan(&r, &st, &hub).unwrap();
        let last = p.steps.last().unwrap();
        assert_eq!(last, &Step::Convert { vendor: "Tongyi-MAI".to_string(), repo: "Z-Image-Turbo".to_string(), recipe: "zimage" });
        // The manifest itself, plus every role file -- none renamed, so
        // subdirectory structure survives into the destination.
        let dest_names: Vec<&str> = p
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Download { dest_name, .. } => Some(dest_name.as_str()),
                _ => None,
            })
            .collect();
        assert!(dest_names.contains(&"model_index.json"));
        assert!(dest_names.contains(&"transformer/config.json"));
        assert!(dest_names.contains(&"vae/diffusion_pytorch_model.safetensors"));
    }

    #[test]
    fn base_ref_with_sharded_weights_plans_index_then_each_shard_sorted() {
        let st = store("modelstore-plan-test-base-sharded");
        let mut hub = FakeHub::new();
        hub.add_file("nvidia", "big-model", "main", "config.json", br#"{"model_type":"qwen3"}"#.to_vec());
        hub.add_file("nvidia", "big-model", "main", "model.safetensors.index.json", b"{}".to_vec());
        hub.add_file("nvidia", "big-model", "main", "model-00002-of-00002.safetensors", vec![0u8; 4]);
        hub.add_file("nvidia", "big-model", "main", "model-00001-of-00002.safetensors", vec![0u8; 4]);

        let r = ModelRef::new("nvidia", "big-model", None);
        let p = plan(&r, &st, &hub).unwrap();
        let files: Vec<&str> = p
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Download { file, .. } => Some(file.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            files,
            vec!["config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        );
    }

    #[test]
    fn unsupported_architecture_aborts_before_any_weight_step() {
        let st = store("modelstore-plan-test-unsupported-arch");
        let mut hub = FakeHub::new();
        hub.add_file("someone", "exotic-model", "main", "config.json", br#"{"architectures":["MambaForCausalLM"]}"#.to_vec());
        hub.add_file("someone", "exotic-model", "main", "model.safetensors", vec![0u8; 8]);

        let r = ModelRef::new("someone", "exotic-model", None);
        let err = plan(&r, &st, &hub).unwrap_err();
        assert!(matches!(*err, PlanError::UnsupportedArchitecture(_, arch) if arch == "MambaForCausalLM"));
    }

    #[test]
    fn missing_config_json_errors_before_any_weight_step() {
        let st = store("modelstore-plan-test-no-config");
        let mut hub = FakeHub::new();
        hub.add_file("someone", "no-config", "main", "model.safetensors", vec![0u8; 8]);

        let r = ModelRef::new("someone", "no-config", None);
        let err = plan(&r, &st, &hub).unwrap_err();
        assert!(matches!(*err, PlanError::NoUpstreamArtifact(_, _)));
    }

    #[test]
    fn quant_ref_with_upstream_gguf_repo_plans_a_single_download() {
        let st = store("modelstore-plan-test-quant-upstream");
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B-GGUF", "main", "Qwen3-0.6B-Q8_0.gguf", vec![0u8; 16]);

        let r = ModelRef::new("Qwen", "Qwen3-0.6B", Some(Quant::Q8_0));
        let p = plan(&r, &st, &hub).unwrap();
        assert_eq!(
            p.steps,
            vec![Step::Download {
                vendor: "Qwen".to_string(),
                repo: "Qwen3-0.6B-GGUF".to_string(),
                revision: "main".to_string(),
                file: "Qwen3-0.6B-Q8_0.gguf".to_string(),
                dest_name: "Q8_0.gguf".to_string(),
            }]
        );
    }

    #[test]
    fn quant_ref_with_cross_vendor_gguf_override_is_preferred() {
        // Qwen3.5-35B-A3B ships no GGUF under its own vendor; bartowski's
        // sibling repo is the only place a quant exists, so the override
        // table must be tried before the same-vendor conventions (which
        // would otherwise silently fall through to a local-quantize plan
        // that then OOMs materializing the 35B fp32 base).
        let st = store("modelstore-plan-test-quant-cross-vendor");
        let mut hub = FakeHub::new();
        hub.add_file("bartowski", "Qwen_Qwen3.5-35B-A3B-GGUF", "main", "Qwen3.5-35B-A3B-Q8_0.gguf", vec![0u8; 16]);

        let r = ModelRef::new("Qwen", "Qwen3.5-35B-A3B", Some(Quant::Q8_0));
        let p = plan(&r, &st, &hub).unwrap();
        assert_eq!(
            p.steps,
            vec![Step::Download {
                vendor: "bartowski".to_string(),
                repo: "Qwen_Qwen3.5-35B-A3B-GGUF".to_string(),
                revision: "main".to_string(),
                file: "Qwen3.5-35B-A3B-Q8_0.gguf".to_string(),
                dest_name: "Q8_0.gguf".to_string(),
            }]
        );
        // The plan's own reference stays canonical (`Qwen/...`), even
        // though the bytes came from a different vendor/repo.
        assert_eq!(p.reference, r);
    }

    #[test]
    fn quant_ref_with_no_upstream_artifact_falls_back_to_base_plus_quantize() {
        let st = store("modelstore-plan-test-quant-fallback");
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", br#"{"architectures":["Qwen3ForCausalLM"]}"#.to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", vec![0u8; 8]);

        let r = ModelRef::new("Qwen", "Qwen3-0.6B", Some(Quant::Q4KM));
        let p = plan(&r, &st, &hub).unwrap();
        assert_eq!(p.reference, r);
        let last = p.steps.last().unwrap();
        assert_eq!(last, &Step::Quantize { base: ModelRef::new("Qwen", "Qwen3-0.6B", None), quant: Quant::Q4KM });
        // and the base download/convert steps precede it
        assert!(p.steps.iter().any(|s| matches!(s, Step::Convert { .. })));
    }

    #[test]
    fn execute_runs_downloads_and_defers_convert_and_quantize() {
        let st = store("modelstore-plan-test-execute");
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", br#"{"architectures":["Qwen3ForCausalLM"]}"#.to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", vec![7u8; 32]);

        let r = ModelRef::new("Qwen", "Qwen3-0.6B", Some(Quant::Q4KM));
        let p = plan(&r, &st, &hub).unwrap();
        let mut progressed = Vec::new();
        let deferred = execute(&st, &hub, &p, &mut |name, got, total| progressed.push((name.to_string(), got, total))).unwrap();

        assert_eq!(deferred, vec![Step::Convert { vendor: "Qwen".to_string(), repo: "Qwen3-0.6B".to_string(), recipe: "transformers" }, Step::Quantize {
            base: ModelRef::new("Qwen", "Qwen3-0.6B", None),
            quant: Quant::Q4KM,
        }]);
        let dir = st.repo_dir(&r.base());
        assert_eq!(std::fs::read(dir.join("config.json")).unwrap(), br#"{"architectures":["Qwen3ForCausalLM"]}"#);
        assert_eq!(std::fs::read(dir.join("model.safetensors")).unwrap(), vec![7u8; 32]);
        assert!(!progressed.is_empty());
    }

    /// A restarted fetch must not re-download files the previous attempt
    /// already landed.
    ///
    /// `plan_base` always returns the FULL artifact list for a family, and the
    /// one early-out `plan()` has (`store.local(reference)`) only fires once
    /// EVERY role/file is present -- so a multi-shard repo killed partway
    /// through re-plans every shard it already has. This was confirmed live:
    /// a killed `qwen3vl` fetch re-downloaded an already-complete 4.97 GB
    /// shard from scratch.
    ///
    /// The assertion is behavioural rather than a download counter: the local
    /// file is overwritten with different bytes between the two `execute`
    /// calls, so if the second call re-downloads, `stream_to_file`'s rename
    /// puts the hub's bytes back and the sentinel is gone.
    #[test]
    fn execute_does_not_redownload_a_file_already_on_disk() {
        let st = store("modelstore-plan-test-execute-resume");
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", br#"{"architectures":["Qwen3ForCausalLM"]}"#.to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", vec![7u8; 32]);

        let r = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let p = plan(&r, &st, &hub).unwrap();
        execute(&st, &hub, &p, &mut |_, _, _| {}).unwrap();

        let dir = st.repo_dir(&r.base());
        let shard = dir.join("model.safetensors");
        std::fs::write(&shard, b"already-here").unwrap();

        let mut progressed = Vec::new();
        execute(&st, &hub, &p, &mut |name, got, total| progressed.push((name.to_string(), got, total))).unwrap();

        assert_eq!(std::fs::read(&shard).unwrap(), b"already-here", "an already-present file was re-downloaded");
        assert!(
            !progressed.iter().any(|(name, _, _)| name == "model.safetensors"),
            "a skipped download must not report transfer progress: {progressed:?}"
        );
    }

    /// The skip is per-file, not per-plan: a repo missing ONE of its files
    /// still fetches that one. Without this, "don't redo work already done"
    /// could be implemented as an all-or-nothing early-out and still pass the
    /// test above while never completing a partially-fetched repo.
    #[test]
    fn execute_still_fetches_the_files_that_are_missing() {
        let st = store("modelstore-plan-test-execute-partial");
        let mut hub = FakeHub::new();
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", br#"{"architectures":["Qwen3ForCausalLM"]}"#.to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors", vec![7u8; 32]);

        let r = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let p = plan(&r, &st, &hub).unwrap();
        let dir = st.repo_dir(&r.base());
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), b"already-here").unwrap();

        execute(&st, &hub, &p, &mut |_, _, _| {}).unwrap();

        assert_eq!(std::fs::read(dir.join("config.json")).unwrap(), b"already-here");
        assert_eq!(std::fs::read(dir.join("model.safetensors")).unwrap(), vec![7u8; 32], "the missing file was not fetched");
    }

    /// A GGUF-release hub fixture: 15 quantizations of one model, no
    /// `config.json` -- the shape `unsloth/FLUX.2-klein-9B-GGUF` really has.
    fn gguf_release_hub() -> FakeHub {
        let mut hub = FakeHub::new();
        hub.add_file("unsloth", "FLUX.2-klein-9B-GGUF", "main", "README.md", vec![b'#'; 10]);
        for (i, q) in ["BF16", "F16", "Q2_K", "Q3_K_M", "Q3_K_S", "Q4_0", "Q4_1", "Q4_K_M", "Q4_K_S", "Q5_0", "Q5_1", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"].iter().enumerate() {
            hub.add_file("unsloth", "FLUX.2-klein-9B-GGUF", "main", &format!("flux-2-klein-9b-{q}.gguf"), vec![i as u8; 8]);
        }
        hub
    }

    /// The bug: a GGUF-only repo fell through to the transformers catch-all
    /// and failed with "no config.json in repo". It must instead resolve to
    /// exactly ONE quantization, and the plan must SAY which -- the resolved
    /// reference carries it, so the front end can print the choice rather
    /// than making it silently.
    #[test]
    fn a_gguf_only_repo_plans_exactly_one_quantization_and_reports_which() {
        let st = store("modelstore-plan-gguf-default");
        let hub = gguf_release_hub();
        let reference = ModelRef::new("unsloth", "FLUX.2-klein-9B-GGUF", None);
        let p = plan(&reference, &st, &hub).unwrap();
        assert_eq!(p.reference.quant(), Some(Quant::Q8_0), "the plan must name the quantization it chose");
        let downloads: Vec<&Step> = p.steps.iter().filter(|s| matches!(s, Step::Download { .. })).collect();
        assert_eq!(downloads.len(), 1, "never more than one quantization: {:?}", p.steps);
        let Step::Download { file, dest_name, .. } = downloads[0] else { unreachable!() };
        assert_eq!(file, "flux-2-klein-9b-Q8_0.gguf");
        assert_eq!(dest_name, "Q8_0.gguf");
    }

    /// The reference grammar's own `-<QUANT>` suffix selects a quantization
    /// inside a repo the user NAMED, rather than sending the sibling-repo
    /// naming ladder off to guess at `<repo>-GGUF`.
    #[test]
    fn a_quant_suffix_selects_a_file_inside_the_named_gguf_repo() {
        let st = store("modelstore-plan-gguf-named");
        let hub = gguf_release_hub();
        let reference = ModelRef::new("unsloth", "FLUX.2-klein-9B-GGUF", Some(Quant::Q4KM));
        let p = plan(&reference, &st, &hub).unwrap();
        let downloads: Vec<&Step> = p.steps.iter().filter(|s| matches!(s, Step::Download { .. })).collect();
        assert_eq!(downloads.len(), 1);
        let Step::Download { file, dest_name, .. } = downloads[0] else { unreachable!() };
        assert_eq!(file, "flux-2-klein-9b-Q4_K_M.gguf");
        assert_eq!(dest_name, "Q4_K_M.gguf");
        assert!(!p.steps.iter().any(|s| matches!(s, Step::Quantize { .. })), "an upstream artifact exists; nothing may be re-quantized locally");

        // A quantization this repo does NOT offer fails with the list, rather
        // than falling back to downloading a base checkpoint to quantize.
        let absent = ModelRef::new("unsloth", "FLUX.2-klein-9B-GGUF", Some(Quant::Q3KL));
        let err = plan(&absent, &st, &hub).unwrap_err().to_string();
        assert!(err.contains("Q3_K_L") && err.contains("Q6_K"), "{err}");
    }

    /// A file URL names the artifact outright: pull exactly that one file,
    /// whatever its extension, from whatever revision was named.
    #[test]
    fn a_named_file_is_pulled_verbatim_from_the_revision_that_named_it() {
        let mut hub = gguf_release_hub();
        hub.add_file("unsloth", "FLUX.2-klein-9B-GGUF", "refs/pr/1", "text_encoder/model.safetensors", vec![3u8; 16]);
        let reference = ModelRef::new("unsloth", "FLUX.2-klein-9B-GGUF", None);

        // A GGUF whose name declares a quantization lands under the store's
        // own `<QUANT>.gguf` name, so this and `<repo>-Q8_0` are one artifact.
        let p = plan_file(&reference, "flux-2-klein-9b-Q8_0.gguf", None, &hub).unwrap();
        assert_eq!(p.reference.quant(), Some(Quant::Q8_0));
        let Some(Step::Download { file, dest_name, revision, .. }) = p.steps.first() else { panic!("{:?}", p.steps) };
        assert_eq!(file, "flux-2-klein-9b-Q8_0.gguf");
        assert_eq!(dest_name, "Q8_0.gguf");
        assert_eq!(revision, "main");

        // Any other file lands under its own path, nested directories intact.
        let p = plan_file(&reference, "text_encoder/model.safetensors", Some("refs/pr/1"), &hub).unwrap();
        let Some(Step::Download { file, dest_name, revision, .. }) = p.steps.first() else { panic!("{:?}", p.steps) };
        assert_eq!(file, "text_encoder/model.safetensors");
        assert_eq!(dest_name, "text_encoder/model.safetensors");
        assert_eq!(revision, "refs/pr/1");

        // A file the revision does not hold is refused with what it does.
        let err = plan_file(&reference, "flux-2-klein-9b-Q9_9.gguf", None, &hub).unwrap_err().to_string();
        assert!(err.contains("Q9_9") && err.contains("flux-2-klein-9b-Q8_0.gguf"), "{err}");
    }
}
