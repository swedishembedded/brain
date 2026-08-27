// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The on-disk model store.
//!
//! Layout, keyed by `<vendor>/<base-repo>` (quant suffix stripped -- a quant
//! shares its base repo's tokenizer and config, so it lives in the same
//! directory):
//!
//! ```text
//! <models-dir>/Qwen/Qwen3-0.6B/
//!     config.json  tokenizer.json  tokenizer_config.json   # this repo's own tokenizer
//!     model.safetensors                                    # upstream, as downloaded
//!     model.brain.safetensors                               # brain-format conversion
//!     Q8_0.gguf                                             # downloaded OR locally produced
//! ```
//!
//! This crate has three responsibilities, each in its own module:
//! [`Store`] (here) finds what already exists on disk; [`hub`] talks to a
//! remote model host; [`plan`] is the pure resolution ladder that decides
//! what to do when nothing is on disk yet. It deliberately depends on nothing
//! above `checkpoint`/`modelref` -- no `capability`, no `residency`, no model
//! crates -- so it can be used by `crates/cli` (today's `model_dir.rs`) and
//! later by a residency-side supplier without either pulling in the other.

pub mod fetch;
pub mod hub;
pub mod plan;
pub mod recipe;
pub mod refurl;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use checkpoint::st::ModelCard;
use checkpoint::weightio::WeightReader;
use brain_modelref::{AdapterRef, ModelRef, Quant};
use serde::{Deserialize, Serialize};

pub use hub::{FakeHub, HfHub, Hub, HubError};
pub use plan::{declared_architecture, execute, family_of_architecture, plan, remaining_download, Plan, PlanError, Remaining, Step};

/// The on-disk container format backing a [`LocalModel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    Gguf,
    /// A [`CompoundManifest`]-described model: several distinct-role files
    /// (not shards of one tensor set) rather than a single weights file --
    /// see [`LocalModel::roles`].
    Compound,
}

/// One servable model found on disk: a fully-qualified [`ModelRef`], the file
/// that holds its weights, and whatever [`ModelCard`] could be read from it.
pub struct LocalModel {
    pub reference: ModelRef,
    pub dir: PathBuf,
    /// For [`Format::Compound`], the manifest file itself (there is no single
    /// weights file) -- real consumers of a compound model use [`roles`]
    /// instead. Present regardless of format so every `LocalModel` has SOME
    /// on-disk anchor to point at (logging, discovery).
    ///
    /// [`roles`]: LocalModel::roles
    pub weights: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub card: Option<ModelCard>,
    pub format: Format,
    /// A named LoRA adapter's own weight file, when `reference.adapter()` is
    /// `Some` -- `weights` above still points at the BASE model (a resident
    /// folds this into those weights at load; see `qwen3::lora::fold_adapter_into`).
    /// Always `None` for a plain base/quant reference.
    pub adapter: Option<PathBuf>,
    /// For [`Format::Compound`] only: role name (`"dit"`, `"vae"`, ...) to its
    /// absolute path (a file or a directory -- whatever the role's loader
    /// accepts) inside [`dir`](LocalModel::dir). `None` for every other format.
    pub roles: Option<BTreeMap<String, PathBuf>>,
}

/// The on-disk description of a compound (multi-file) model: several
/// distinct-role files that together make one servable model, rather than
/// shards of a single tensor set. Written by a [`recipe::ArtifactRecipe`]'s
/// finish step (`crates/cli/src/supply.rs::convert`) after the recipe's
/// artifacts have downloaded; read back by [`Store::local`]/[`Store::scan`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundManifest {
    /// The fully-qualified id this model registers under (`vendor/repo`).
    pub id: String,
    /// The `resident_for`-style family dispatch key (`"zimage"`, ...).
    pub family: String,
    /// Role name -> path RELATIVE to the repo directory (a file or a
    /// directory, whichever that role's loader accepts).
    pub roles: BTreeMap<String, String>,
}

/// The manifest file name inside a compound model's repo directory.
pub const MANIFEST_FILE: &str = "brain.manifest.json";

/// A models directory on disk, laid out `<vendor>/<base-repo>/...`.
pub struct Store {
    root: PathBuf,
}

/// brain's data root, published by the `--brain-data-dir` flag. `None` (the
/// default) means "nothing published", not "no data root" -- the environment
/// ladder in [`default_root`] then answers, exactly as it did before this
/// existed.
///
/// A published override rather than a second resolver function, so
/// [`default_root`] stays the ONE answer to "where do models live" for every
/// caller -- including the ones with no CLI flag in scope
/// (`brain_testutil`'s fixtures, this crate's own internals). A `RwLock`
/// rather than a `OnceLock` because clearing it has to work: it is written
/// once at process start in production, but tests must be able to put it
/// back.
static DATA_ROOT: std::sync::RwLock<Option<PathBuf>> = std::sync::RwLock::new(None);

/// Publish (or, with `None`, clear) brain's data root -- the top of
/// [`default_root`]'s ladder. Called once from the CLI's global flag parse
/// before any subcommand runs; a lock poisoned by a panicking writer is
/// recovered rather than propagated, since losing the override must not turn
/// an unrelated panic into a second one here.
pub fn publish_data_root(root: Option<PathBuf>) {
    let mut slot = DATA_ROOT.write().unwrap_or_else(|e| e.into_inner());
    *slot = root;
}

/// The models directory inside a data root. brain's data root holds more
/// than models over time, so models live in their own subdirectory; this is
/// the one place that fact is written down.
pub fn models_dir_in(data_root: &Path) -> PathBuf {
    data_root.join("models")
}

/// Resolve the models directory. Precedence, highest first:
///
/// 1. the data root published by `--brain-data-dir <root>`, as
///    [`models_dir_in`] of it;
/// 2. `BRAIN_MODELS_DIR`;
/// 3. `$XDG_DATA_HOME/brain/models`;
/// 4. `$HOME/.local/share/brain/models`.
///
/// `None` only when nothing is published and every one of those is unset (no
/// `$HOME`). The flag deliberately outranks `BRAIN_MODELS_DIR` -- an
/// explicitly typed flag beating an inherited environment variable is the
/// same rule `--models-dir` and `--device` already follow -- and the CLI says
/// so out loud when both are set and disagree, so the environment is never
/// silently overruled.
///
/// This is the env-only tail of `crates/cli/src/model_dir.rs`'s `resolve`
/// (which layers a `--models-dir` flag override on top) -- shared here so
/// anything that needs "the models dir" without a CLI flag in scope (this
/// crate's own callers, `brain_testutil`'s model-backed test fixtures)
/// doesn't duplicate the precedence.
pub fn default_root() -> Option<PathBuf> {
    if let Some(root) = DATA_ROOT.read().unwrap_or_else(|e| e.into_inner()).as_deref() {
        return Some(models_dir_in(root));
    }
    if let Some(p) = std::env::var_os("BRAIN_MODELS_DIR").filter(|s| !s.is_empty()) {
        return Some(PathBuf::from(p));
    }
    if let Some(x) = std::env::var_os("XDG_DATA_HOME").filter(|s| !s.is_empty()) {
        return Some(Path::new(&x).join("brain").join("models"));
    }
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .map(|h| Path::new(&h).join(".local").join("share").join("brain").join("models"))
}

/// The brain-format conversion of a base repo's weights -- what a resident
/// actually loads. The upstream `model.safetensors` (or shard set) that
/// produced it is not itself servable: `model_dir::register` requires a
/// `brain.card`, which only a brain-format file carries.
const BASE_WEIGHTS_FILE: &str = "model.brain.safetensors";
const TOKENIZER_FILE: &str = "tokenizer.json";
/// Named LoRA adapters live inside their base repo's directory (they share
/// its tokenizer/config, same reasoning as quants) under
/// `adapters/<owner>/<name>/<tag>/`, one adapter-only safetensors file each
/// (`qwen3::lora::save_adapter`'s output). Adding this subdirectory is
/// backward compatible with `Store::scan`'s existing base/quant walk, which
/// only ever iterates `is_file()` entries in a repo dir.
const ADAPTERS_DIR: &str = "adapters";
const ADAPTER_WEIGHTS_FILE: &str = "adapter.brain.safetensors";

impl Store {
    pub fn new(root: impl Into<PathBuf>) -> Store {
        Store { root: root.into() }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The directory holding everything for `reference`'s base repo
    /// (quant suffix stripped, since a quant shares its base repo's directory).
    pub fn repo_dir(&self, reference: &ModelRef) -> PathBuf {
        self.root.join(reference.vendor()).join(reference.repo())
    }

    /// Where `reference`'s adapter file belongs on disk, whether or not it
    /// exists yet -- for a trainer that is ABOUT TO WRITE one (unlike
    /// [`Store::local`], which requires the file to already be there).
    /// `None` if `reference` names no adapter.
    pub fn adapter_weights_path(&self, reference: &ModelRef) -> Option<PathBuf> {
        let a = reference.adapter()?;
        Some(self.repo_dir(reference).join(ADAPTERS_DIR).join(a.owner()).join(a.name()).join(a.tag()).join(ADAPTER_WEIGHTS_FILE))
    }

    /// Looks up `reference` on disk. `None` if the expected file is absent or
    /// unreadable as a weight file -- callers fall through to [`plan`] either way.
    pub fn local(&self, reference: &ModelRef) -> Option<LocalModel> {
        let dir = self.repo_dir(reference);
        if reference.adapter().is_some() {
            return self.local_adapter(reference, &dir);
        }
        match reference.quant() {
            Some(q) => self.local_quant(reference, &dir, q),
            None => self.local_base(reference, &dir),
        }
    }

    fn local_base(&self, reference: &ModelRef, dir: &Path) -> Option<LocalModel> {
        if let Some(m) = self.local_compound(reference, dir) {
            return Some(m);
        }
        let weights = dir.join(BASE_WEIGHTS_FILE);
        open_local(reference.clone(), dir.to_path_buf(), weights, Format::Safetensors)
    }

    /// A [`CompoundManifest`]-described model: tried before the single-file
    /// case (same class -- no quant, no adapter), since a repo dir carrying
    /// `brain.manifest.json` has no `model.brain.safetensors` to fall back
    /// to. `None` (not a partial `LocalModel`) if the manifest is missing,
    /// unparseable, or any role's path doesn't exist -- an incomplete
    /// compound model is "not found", the same policy [`open_local`] already
    /// applies to a missing single weights file.
    fn local_compound(&self, reference: &ModelRef, dir: &Path) -> Option<LocalModel> {
        let manifest_path = dir.join(MANIFEST_FILE);
        if !manifest_path.is_file() {
            return None;
        }
        let bytes = std::fs::read(&manifest_path).ok()?;
        let manifest: CompoundManifest = serde_json::from_slice(&bytes).ok()?;
        let mut roles = BTreeMap::new();
        for (role, rel) in &manifest.roles {
            let p = dir.join(rel);
            if !p.exists() {
                return None;
            }
            roles.insert(role.clone(), p);
        }
        let card = ModelCard::for_ref(&manifest.id, reference.vendor(), reference.repo(), None, &manifest.family);
        Some(LocalModel {
            reference: reference.clone(),
            dir: dir.to_path_buf(),
            weights: manifest_path,
            tokenizer: None,
            card: Some(card),
            format: Format::Compound,
            adapter: None,
            roles: Some(roles),
        })
    }

    fn local_quant(&self, reference: &ModelRef, dir: &Path, quant: Quant) -> Option<LocalModel> {
        let weights = dir.join(format!("{}.gguf", quant.as_str()));
        open_local(reference.clone(), dir.to_path_buf(), weights, Format::Gguf)
    }

    /// `reference.adapter()` is `Some`: resolve the BASE weights the normal
    /// way (quant, if any, still respected) and the adapter's own file
    /// alongside it. `None` if either is missing -- a dangling adapter
    /// (base deleted) or a not-yet-trained adapter are both "not found",
    /// not a partial result.
    fn local_adapter(&self, reference: &ModelRef, dir: &Path) -> Option<LocalModel> {
        let base_ref = reference.without_adapter();
        let (base_weights, format) = match base_ref.quant() {
            Some(q) => (dir.join(format!("{}.gguf", q.as_str())), Format::Gguf),
            None => (dir.join(BASE_WEIGHTS_FILE), Format::Safetensors),
        };
        if !base_weights.is_file() {
            return None;
        }
        let adapter_path = self.adapter_weights_path(reference)?;
        if !adapter_path.is_file() {
            return None;
        }
        let reader = WeightReader::open(adapter_path.to_str()?).ok()?;
        let tokenizer = {
            let t = dir.join(TOKENIZER_FILE);
            t.is_file().then_some(t)
        };
        Some(LocalModel {
            reference: reference.clone(),
            dir: dir.to_path_buf(),
            weights: base_weights,
            tokenizer,
            card: reader.card(),
            format,
            adapter: Some(adapter_path),
            roles: None,
        })
    }

    /// Every servable model currently on disk: one entry per base repo with a
    /// `model.brain.safetensors`, plus one entry per `<QUANT>.gguf` file found
    /// in each vendor/repo directory. Unreadable files are skipped, not errors.
    pub fn scan(&self) -> Vec<LocalModel> {
        let mut out = Vec::new();
        let Ok(vendors) = std::fs::read_dir(&self.root) else {
            return out;
        };
        for v in vendors.flatten() {
            if !v.path().is_dir() {
                continue;
            }
            let Some(vendor) = v.file_name().to_str().map(str::to_string) else {
                continue;
            };
            let Ok(repos) = std::fs::read_dir(v.path()) else {
                continue;
            };
            for r in repos.flatten() {
                if !r.path().is_dir() {
                    continue;
                }
                let Some(repo) = r.file_name().to_str().map(str::to_string) else {
                    continue;
                };
                out.extend(self.scan_repo_dir(&vendor, &repo, &r.path()));
            }
        }
        out.sort_by(|a, b| a.reference.cmp(&b.reference));
        out
    }

    fn scan_repo_dir(&self, vendor: &str, repo: &str, dir: &Path) -> Vec<LocalModel> {
        let mut out = Vec::new();
        let base = ModelRef::new(vendor, repo, None);
        if let Some(m) = self.local_base(&base, dir) {
            out.push(m);
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return out;
        };
        for e in entries.flatten() {
            let path = e.path();
            if !path.is_file() {
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(quant) = Quant::parse(stem) else {
                continue;
            };
            let r = ModelRef::new(vendor, repo, Some(quant));
            if let Some(m) = self.local_quant(&r, dir, quant) {
                out.push(m);
            }
        }
        out.extend(self.scan_adapters(vendor, repo, dir));
        out
    }

    /// Walk `<repo-dir>/adapters/<owner>/<name>/<tag>/` for every trained
    /// adapter on this base repo. Each level is skipped (not an error) if
    /// unreadable or contains a non-UTF8/non-directory entry -- a scan finds
    /// what's servable, it doesn't validate the whole tree.
    fn scan_adapters(&self, vendor: &str, repo: &str, base_dir: &Path) -> Vec<LocalModel> {
        let mut out = Vec::new();
        let Ok(owners) = std::fs::read_dir(base_dir.join(ADAPTERS_DIR)) else {
            return out;
        };
        for o in owners.flatten().filter(|e| e.path().is_dir()) {
            let Some(owner) = o.file_name().to_str().map(str::to_string) else { continue };
            let Ok(names) = std::fs::read_dir(o.path()) else { continue };
            for n in names.flatten().filter(|e| e.path().is_dir()) {
                let Some(name) = n.file_name().to_str().map(str::to_string) else { continue };
                let Ok(tags) = std::fs::read_dir(n.path()) else { continue };
                for t in tags.flatten().filter(|e| e.path().is_dir()) {
                    let Some(tag) = t.file_name().to_str().map(str::to_string) else { continue };
                    let adapter = AdapterRef::new(&owner, &name, &tag);
                    let r = ModelRef::new_adapter(vendor, repo, None, adapter);
                    if let Some(m) = self.local_adapter(&r, base_dir) {
                        out.push(m);
                    }
                }
            }
        }
        out
    }
}

fn open_local(reference: ModelRef, dir: PathBuf, weights: PathBuf, format: Format) -> Option<LocalModel> {
    if !weights.is_file() {
        return None;
    }
    let reader = WeightReader::open(weights.to_str()?).ok()?;
    let tokenizer = {
        let t = dir.join(TOKENIZER_FILE);
        t.is_file().then_some(t)
    };
    Some(LocalModel { reference, dir, weights, tokenizer, card: reader.card(), format, adapter: None, roles: None })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scratch_store(name: &str) -> Store {
        let dir = std::env::temp_dir().join(name);
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        Store::new(dir)
    }

    fn write_base_fixture(store: &Store, vendor: &str, repo: &str) {
        let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
        std::fs::create_dir_all(&dir).unwrap();
        let card = ModelCard::new(format!("{vendor}/{repo}"), "qwen");
        checkpoint::st::save_safetensors(
            dir.join(BASE_WEIGHTS_FILE).to_str().unwrap(),
            &[("weight".to_string(), vec![2], vec![1.0, 2.0])],
            &json!({"hidden_size": 8}),
            Some(&card),
        )
        .unwrap();
        std::fs::write(dir.join(TOKENIZER_FILE), b"{}").unwrap();
    }

    #[test]
    fn repo_dir_is_vendor_slash_repo_regardless_of_quant() {
        let store = scratch_store("modelstore-lib-test-repo-dir");
        let base = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let quantized = ModelRef::new("Qwen", "Qwen3-0.6B", Some(Quant::Q4KM));
        assert_eq!(store.repo_dir(&base), store.repo_dir(&quantized));
        assert_eq!(store.repo_dir(&base), store.root().join("Qwen").join("Qwen3-0.6B"));
    }

    #[test]
    fn local_finds_a_real_base_model_and_reads_its_card_and_tokenizer() {
        let store = scratch_store("modelstore-lib-test-local-base");
        write_base_fixture(&store, "Qwen", "Qwen3-0.6B");

        let r = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let local = store.local(&r).expect("fixture should be found");
        assert_eq!(local.format, Format::Safetensors);
        assert!(local.tokenizer.is_some());
        let card = local.card.expect("save_safetensors wrote a card");
        assert_eq!(card.id, "Qwen/Qwen3-0.6B");
    }

    fn write_compound_fixture(store: &Store, vendor: &str, repo: &str) {
        let dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        std::fs::create_dir_all(dir.join("vae")).unwrap();
        std::fs::write(dir.join("transformer").join("config.json"), b"{}").unwrap();
        std::fs::write(dir.join("vae").join("diffusion_pytorch_model.safetensors"), b"stub").unwrap();
        let manifest = CompoundManifest {
            id: format!("{vendor}/{repo}"),
            family: "zimage".to_string(),
            roles: BTreeMap::from([
                ("dit".to_string(), "transformer".to_string()),
                ("vae".to_string(), "vae/diffusion_pytorch_model.safetensors".to_string()),
            ]),
        };
        std::fs::write(dir.join(MANIFEST_FILE), serde_json::to_vec(&manifest).unwrap()).unwrap();
    }

    #[test]
    fn local_finds_a_compound_model_via_its_manifest_and_resolves_every_role() {
        let store = scratch_store("modelstore-lib-test-compound-local");
        write_compound_fixture(&store, "Tongyi-MAI", "Z-Image-Turbo");

        let r = ModelRef::new("Tongyi-MAI", "Z-Image-Turbo", None);
        let local = store.local(&r).expect("compound fixture should be found");
        assert_eq!(local.format, Format::Compound);
        assert!(local.tokenizer.is_none());
        let card = local.card.expect("local_compound synthesizes a card");
        assert_eq!(card.id, "Tongyi-MAI/Z-Image-Turbo");
        assert_eq!(card.family, "zimage");
        let roles = local.roles.expect("compound model must carry roles");
        assert_eq!(roles.len(), 2);
        assert!(roles["dit"].ends_with("transformer"));
        assert!(roles["vae"].ends_with("diffusion_pytorch_model.safetensors"));
    }

    #[test]
    fn local_compound_is_none_when_a_role_path_is_missing_incomplete_fetch() {
        let store = scratch_store("modelstore-lib-test-compound-incomplete");
        write_compound_fixture(&store, "Tongyi-MAI", "Z-Image-Turbo");
        // Simulate an interrupted fetch: the manifest is there, but one of its
        // role paths never landed.
        std::fs::remove_dir_all(store.repo_dir(&ModelRef::new("Tongyi-MAI", "Z-Image-Turbo", None)).join("transformer")).unwrap();

        let r = ModelRef::new("Tongyi-MAI", "Z-Image-Turbo", None);
        assert!(store.local(&r).is_none(), "an incomplete compound model must not resolve as found");
    }

    #[test]
    fn scan_finds_a_compound_model_alongside_a_single_file_one() {
        let store = scratch_store("modelstore-lib-test-compound-scan");
        write_base_fixture(&store, "Qwen", "Qwen3-0.6B");
        write_compound_fixture(&store, "Tongyi-MAI", "Z-Image-Turbo");

        let found = store.scan();
        let refs: Vec<ModelRef> = found.iter().map(|m| m.reference.clone()).collect();
        assert!(refs.contains(&ModelRef::new("Qwen", "Qwen3-0.6B", None)));
        assert!(refs.contains(&ModelRef::new("Tongyi-MAI", "Z-Image-Turbo", None)));
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn default_root_prefers_brain_models_dir_over_xdg_and_home() {
        // No other test in this crate reads these three env vars, so mutating
        // them here (Rust >= 1.82 requires `unsafe` for env mutation, since
        // it is genuinely process-global and unsafe to race against another
        // thread reading it) cannot race a concurrently-running test. Capture
        // the real values up front so they can be restored exactly, not
        // guessed, once this test is done mutating them.
        let orig_home = std::env::var_os("HOME");
        let orig_xdg = std::env::var_os("XDG_DATA_HOME");
        let orig_models = std::env::var_os("BRAIN_MODELS_DIR");

        unsafe {
            std::env::set_var("BRAIN_MODELS_DIR", "/scratch/models-dir");
            std::env::set_var("XDG_DATA_HOME", "/scratch/xdg");
        }
        assert_eq!(default_root(), Some(PathBuf::from("/scratch/models-dir")));

        unsafe {
            std::env::remove_var("BRAIN_MODELS_DIR");
        }
        assert_eq!(default_root(), Some(PathBuf::from("/scratch/xdg/brain/models")));

        unsafe {
            std::env::remove_var("XDG_DATA_HOME");
            std::env::set_var("HOME", "/scratch/home");
        }
        assert_eq!(default_root(), Some(PathBuf::from("/scratch/home/.local/share/brain/models")));

        unsafe {
            std::env::remove_var("HOME");
        }
        assert_eq!(default_root(), None);

        // Restore every var to its real pre-test value (or unset, if it was
        // unset before) so nothing else in this process is left disturbed.
        unsafe {
            match orig_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            match orig_xdg {
                Some(v) => std::env::set_var("XDG_DATA_HOME", v),
                None => std::env::remove_var("XDG_DATA_HOME"),
            }
            match orig_models {
                Some(v) => std::env::set_var("BRAIN_MODELS_DIR", v),
                None => std::env::remove_var("BRAIN_MODELS_DIR"),
            }
        }
    }

    #[test]
    fn local_returns_none_when_nothing_is_on_disk() {
        let store = scratch_store("modelstore-lib-test-local-missing");
        let r = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        assert!(store.local(&r).is_none());
    }

    #[test]
    fn scan_finds_the_base_model_and_ignores_unrelated_files() {
        let store = scratch_store("modelstore-lib-test-scan");
        write_base_fixture(&store, "Qwen", "Qwen3-0.6B");
        // A stray file that must not be picked up as a quant artifact.
        std::fs::write(store.repo_dir(&ModelRef::new("Qwen", "Qwen3-0.6B", None)).join("README.md"), b"hi").unwrap();

        let found = store.scan();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].reference, ModelRef::new("Qwen", "Qwen3-0.6B", None));
    }

    #[test]
    fn scan_across_two_vendors_returns_both_sorted_by_reference() {
        let store = scratch_store("modelstore-lib-test-scan-multi");
        write_base_fixture(&store, "Qwen", "Qwen3-0.6B");
        write_base_fixture(&store, "LiquidAI", "LFM2.5-350M");

        let found = store.scan();
        let refs: Vec<ModelRef> = found.into_iter().map(|m| m.reference).collect();
        let mut sorted = refs.clone();
        sorted.sort();
        assert_eq!(refs, sorted);
        assert_eq!(refs.len(), 2);
    }

    fn write_adapter_fixture(store: &Store, vendor: &str, repo: &str, owner: &str, name: &str, tag: &str) {
        let base_dir = store.repo_dir(&ModelRef::new(vendor, repo, None));
        let adapter_dir = base_dir.join(ADAPTERS_DIR).join(owner).join(name).join(tag);
        std::fs::create_dir_all(&adapter_dir).unwrap();
        let card_id = format!("{vendor}/{repo}:{owner}:{name}:{tag}");
        let mut card = ModelCard::new(&card_id, "qwen");
        card.variant_of = Some(format!("{vendor}/{repo}"));
        card.adapter = Some(checkpoint::st::Adapter {
            kind: "lora".into(),
            rank: Some(8),
            base: Some(format!("{vendor}/{repo}")),
            alpha: Some(16.0),
            targets: Some(vec!["wq".into()]),
            dataset_id: None,
        });
        checkpoint::st::save_safetensors(
            adapter_dir.join(ADAPTER_WEIGHTS_FILE).to_str().unwrap(),
            &[("blocks.0.attn.wq.weight.lora_a".to_string(), vec![2], vec![0.1, 0.2])],
            &json!({"rank": 8, "alpha": 16.0}),
            Some(&card),
        )
        .unwrap();
    }

    #[test]
    fn local_resolves_an_adapter_ref_to_base_weights_plus_the_adapter_file() {
        let store = scratch_store("modelstore-lib-test-adapter-local");
        write_base_fixture(&store, "Qwen", "Qwen3-0.6B");
        write_adapter_fixture(&store, "Qwen", "Qwen3-0.6B", "swedishembedded-com", "generic-sft", "latest");

        let r = ModelRef::parse("Qwen/Qwen3-0.6B:swedishembedded-com:generic-sft:latest").unwrap();
        let found = store.local(&r).expect("adapter should resolve");
        assert!(found.weights.ends_with(BASE_WEIGHTS_FILE), "weights must point at the BASE model: {:?}", found.weights);
        assert!(found.weights.is_file());
        let adapter_path = found.adapter.expect("adapter path must be set");
        assert!(adapter_path.ends_with(ADAPTER_WEIGHTS_FILE));
        assert!(adapter_path.is_file());
        assert_eq!(found.tokenizer, Some(store.repo_dir(&ModelRef::new("Qwen", "Qwen3-0.6B", None)).join(TOKENIZER_FILE)));
        let card = found.card.expect("adapter file's card");
        assert_eq!(card.id, "Qwen/Qwen3-0.6B:swedishembedded-com:generic-sft:latest");
        assert_eq!(card.adapter.unwrap().rank, Some(8));
    }

    #[test]
    fn local_is_none_when_the_base_is_missing_even_if_the_adapter_exists() {
        let store = scratch_store("modelstore-lib-test-adapter-no-base");
        write_adapter_fixture(&store, "Qwen", "Qwen3-0.6B", "owner", "name", "latest");
        let r = ModelRef::parse("Qwen/Qwen3-0.6B:owner:name:latest").unwrap();
        assert!(store.local(&r).is_none());
    }

    #[test]
    fn local_is_none_when_the_adapter_is_missing_even_if_the_base_exists() {
        let store = scratch_store("modelstore-lib-test-adapter-no-adapter");
        write_base_fixture(&store, "Qwen", "Qwen3-0.6B");
        let r = ModelRef::parse("Qwen/Qwen3-0.6B:owner:name:latest").unwrap();
        assert!(store.local(&r).is_none());
    }

    #[test]
    fn scan_finds_a_base_model_and_its_adapter_as_two_distinct_entries() {
        let store = scratch_store("modelstore-lib-test-scan-adapter");
        write_base_fixture(&store, "Qwen", "Qwen3-0.6B");
        write_adapter_fixture(&store, "Qwen", "Qwen3-0.6B", "swedishembedded-com", "generic-sft", "latest");

        let found = store.scan();
        assert_eq!(found.len(), 2, "base and adapter must be two distinct catalog entries, not merged");
        let refs: Vec<String> = found.iter().map(|m| m.reference.to_string()).collect();
        assert!(refs.contains(&"Qwen/Qwen3-0.6B".to_string()));
        assert!(refs.contains(&"Qwen/Qwen3-0.6B:swedishembedded-com:generic-sft:latest".to_string()));
    }
}
