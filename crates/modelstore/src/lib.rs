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

use std::path::{Path, PathBuf};

use checkpoint::st::ModelCard;
use checkpoint::weightio::WeightReader;
use brain_modelref::{ModelRef, Quant};

pub use hub::{FakeHub, HfHub, Hub, HubError};
pub use plan::{execute, plan, Plan, PlanError, Step};

/// The on-disk container format backing a [`LocalModel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    Gguf,
}

/// One servable model found on disk: a fully-qualified [`ModelRef`], the file
/// that holds its weights, and whatever [`ModelCard`] could be read from it.
pub struct LocalModel {
    pub reference: ModelRef,
    pub dir: PathBuf,
    pub weights: PathBuf,
    pub tokenizer: Option<PathBuf>,
    pub card: Option<ModelCard>,
    pub format: Format,
}

/// A models directory on disk, laid out `<vendor>/<base-repo>/...`.
pub struct Store {
    root: PathBuf,
}

/// The brain-format conversion of a base repo's weights -- what a resident
/// actually loads. The upstream `model.safetensors` (or shard set) that
/// produced it is not itself servable: `model_dir::register` requires a
/// `brain.card`, which only a brain-format file carries.
const BASE_WEIGHTS_FILE: &str = "model.brain.safetensors";
const TOKENIZER_FILE: &str = "tokenizer.json";

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

    /// Looks up `reference` on disk. `None` if the expected file is absent or
    /// unreadable as a weight file -- callers fall through to [`plan`] either way.
    pub fn local(&self, reference: &ModelRef) -> Option<LocalModel> {
        let dir = self.repo_dir(reference);
        match reference.quant() {
            Some(q) => self.local_quant(reference, &dir, q),
            None => self.local_base(reference, &dir),
        }
    }

    fn local_base(&self, reference: &ModelRef, dir: &Path) -> Option<LocalModel> {
        let weights = dir.join(BASE_WEIGHTS_FILE);
        open_local(reference.clone(), dir.to_path_buf(), weights, Format::Safetensors)
    }

    fn local_quant(&self, reference: &ModelRef, dir: &Path, quant: Quant) -> Option<LocalModel> {
        let weights = dir.join(format!("{}.gguf", quant.as_str()));
        open_local(reference.clone(), dir.to_path_buf(), weights, Format::Gguf)
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
    Some(LocalModel { reference, dir, weights, tokenizer, card: reader.card(), format })
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
}
