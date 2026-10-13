// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which decision models this host has, and whether each can be served -
//! answered from the filesystem alone, loading nothing.
//!
//! A model's id is its position in the model store, `<vendor>/<repo>`, the
//! same id `brain pull` lays it down under, so two checkpoints are two
//! separately addressable models and an id is stable across restarts. A
//! checkpoint outside the store gets `local/<directory name>`.
//!
//! [`Catalog::records`] is the cheap `ready`/status query: it checks that the
//! files a load needs are present and that the manifest parses, and it reads
//! no weights. Pulling weights is not loading, so a model pulled after the
//! process started becomes `ready` here without a restart (a serving process
//! registers its residents once, so serving it still takes a restart).
//!
//! Swedish Embedded AB implements model inventories that tell an operator,
//! without loading anything, which models are present and fit to serve. If
//! your team needs expertise in model lifecycle and readiness, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::engine::Backend;
use crate::manifest::DecisionManifest;
use crate::select::{select, Candidate, Selection};

/// Whether a model can be served right now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readiness {
    Ready,
    /// Files the load needs are absent.
    NotInstalled { missing: Vec<String> },
    /// The files are there but are not a servable decision model.
    Incompatible { reason: String },
}

impl Readiness {
    pub fn slug(&self) -> &'static str {
        match self {
            Readiness::Ready => "ready",
            Readiness::NotInstalled { .. } => "not-installed",
            Readiness::Incompatible { .. } => "incompatible",
        }
    }
}

/// One decision model this host knows about.
#[derive(Clone, Debug)]
pub struct ModelRecord {
    pub id: String,
    pub dir: PathBuf,
    /// The architecture the directory's structure names, if it names one.
    pub backend: Option<Backend>,
    pub manifest: Option<DecisionManifest>,
    pub readiness: Readiness,
    /// Named by the operator, rather than found by scanning the store.
    pub configured: bool,
}

impl ModelRecord {
    pub fn installed(&self) -> bool {
        self.readiness == Readiness::Ready
    }

    /// What the selector needs. A record with no known architecture is never a
    /// candidate for automatic selection.
    pub fn candidate(&self) -> Option<Candidate> {
        Some(Candidate {
            id: self.id.clone(),
            backend: self.backend?,
            installed: self.installed(),
            validated: self.manifest.as_ref().is_some_and(|m| m.validation.is_validated()),
        })
    }

    /// The status document a client reads.
    pub fn to_json(&self) -> Value {
        let mut v = json!({
            "id": self.id,
            "state": self.readiness.slug(),
            "configured": self.configured,
            "backend": self.backend.map(Backend::slug),
        });
        match &self.readiness {
            Readiness::Ready => {}
            Readiness::NotInstalled { missing } => {
                v["missing"] = json!(missing);
                v["hint"] = json!(format!("run `brain pull {}`", self.id));
            }
            Readiness::Incompatible { reason } => v["reason"] = json!(reason),
        }
        if let Some(m) = &self.manifest {
            v["manifest"] = m.to_json();
            v["validated"] = json!(m.validation.is_validated());
        } else {
            v["validated"] = json!(false);
        }
        v
    }
}

/// The files a Laya load reads, relative to the checkpoint directory.
const LAYA_FILES: [&str; 4] = ["model.safetensors", "tokenizer/tokenizer.json", "encoder/config.json", "rl_agent_config.json"];
/// The files a `decide` load reads.
const DECIDE_FILES: [&str; 3] = ["model.safetensors", "tokenizer.json", "config.json"];

/// The weights file and tokenizer file of an architecture, for identity checks.
pub fn identity_files(backend: Backend) -> (&'static str, &'static str) {
    match backend {
        Backend::Laya => ("model.safetensors", "tokenizer/tokenizer.json"),
        Backend::Decide => ("model.safetensors", "tokenizer.json"),
    }
}

/// Whether a `decide` directory is a SAVED decision model: `save_model` marks
/// its `config.json` with `"head": "option-head"`. A bare encoder checkpoint
/// has no such mark and no trained head, so scanning must not offer it.
fn decide_dir_carries_a_head(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .is_some_and(|c| c.get("head").and_then(Value::as_str) == Some("option-head"))
}

/// Judge one directory without reading its weights.
pub fn inspect(id: &str, dir: &Path, configured: bool) -> ModelRecord {
    let record = |backend, manifest, readiness| ModelRecord { id: id.to_string(), dir: dir.to_path_buf(), backend, manifest, readiness, configured };
    if !dir.is_dir() {
        return record(None, None, Readiness::NotInstalled { missing: vec!["the model directory".into()] });
    }
    let Some(backend) = Backend::sniff(dir) else {
        let empty = std::fs::read_dir(dir).map(|mut d| d.next().is_none()).unwrap_or(true);
        let readiness = if empty {
            Readiness::NotInstalled { missing: vec!["the model files".into()] }
        } else {
            Readiness::Incompatible { reason: "the directory is neither a Laya checkpoint nor a decide checkpoint".into() }
        };
        return record(None, None, readiness);
    };
    let required: &[&str] = if backend == Backend::Laya { &LAYA_FILES } else { &DECIDE_FILES };
    let missing: Vec<String> = required.iter().filter(|f| !dir.join(f).is_file()).map(|f| (*f).to_string()).collect();
    let manifest = DecisionManifest::read(dir, backend);
    let (manifest, manifest_err) = match manifest {
        Ok(m) => (Some(m), None),
        Err(e) => (None, Some(e)),
    };
    let readiness = if !missing.is_empty() {
        Readiness::NotInstalled { missing }
    } else if let Some(e) = manifest_err {
        Readiness::Incompatible { reason: e }
    } else if backend == Backend::Decide && !decide_dir_carries_a_head(dir) {
        Readiness::Incompatible { reason: "a decide encoder with no trained head is not a decision model: it would answer near-uniformly".into() }
    } else {
        Readiness::Ready
    };
    record(Some(backend), manifest, readiness)
}

/// The id and directory a model spec names. A spec starting with `/` or `.` is
/// a directory; anything else is a store reference, `<vendor>/<repo>`, under
/// `models_dir`.
pub fn resolve_spec(models_dir: Option<&Path>, spec: &str) -> (String, PathBuf) {
    if spec.starts_with('/') || spec.starts_with('.') {
        let dir = PathBuf::from(spec);
        if let Some(rel) = models_dir.and_then(|m| dir.strip_prefix(m).ok()) {
            if rel.components().count() == 2 {
                return (rel.to_string_lossy().into_owned(), dir);
            }
        }
        let leaf = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "model".into());
        return (format!("local/{leaf}"), dir);
    }
    (spec.to_string(), models_dir.map(|m| m.join(spec)).unwrap_or_else(|| PathBuf::from(spec)))
}

/// The decision models a host has: the ones its operator named, plus every
/// servable one found in the model store.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    models_dir: Option<PathBuf>,
    configured: Vec<String>,
    default: Option<String>,
}

impl Catalog {
    /// `configured` are model specs the operator named (each always listed,
    /// installed or not, so a missing one reports `not-installed` instead of
    /// vanishing); `default` is the one id `"auto"` resolves to when set.
    pub fn new(models_dir: Option<PathBuf>, configured: Vec<String>, default: Option<String>) -> Catalog {
        Catalog { models_dir, configured, default }
    }

    pub fn models_dir(&self) -> Option<&Path> {
        self.models_dir.as_deref()
    }

    /// Every known model, configured first, then discovered, each judged now.
    pub fn records(&self) -> Vec<ModelRecord> {
        let mut out: Vec<ModelRecord> = Vec::new();
        for spec in &self.configured {
            let (id, dir) = resolve_spec(self.models_dir.as_deref(), spec);
            if !out.iter().any(|r| r.id == id) {
                out.push(inspect(&id, &dir, true));
            }
        }
        if let Some(root) = &self.models_dir {
            for (id, dir) in scan_store(root) {
                if out.iter().any(|r| r.id == id) {
                    continue;
                }
                let rec = inspect(&id, &dir, false);
                // A scan offers only what can be served now: a directory that
                // merely happens to sit in the store, or one still being
                // pulled, is not a model this host has. Naming it makes it
                // visible in whatever state it is in.
                if rec.installed() {
                    out.push(rec);
                }
            }
        }
        out
    }

    /// The model `requested` names. `"auto"` applies [`select`] with the
    /// catalog's default as the explicit choice; any other value is taken as
    /// an id and must be known.
    pub fn select(&self, requested: &str) -> Selection {
        let records = self.records();
        let candidates: Vec<Candidate> = records.iter().filter_map(ModelRecord::candidate).collect();
        if requested == "auto" {
            return select(&candidates, self.default.as_deref());
        }
        select(&candidates, Some(requested))
    }
}

/// `<vendor>/<repo>` directories under a store root.
fn scan_store(root: &Path) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    let Ok(vendors) = std::fs::read_dir(root) else { return found };
    let mut vendors: Vec<_> = vendors.flatten().filter(|e| e.path().is_dir()).collect();
    vendors.sort_by_key(|e| e.file_name());
    for vendor in vendors {
        let Ok(repos) = std::fs::read_dir(vendor.path()) else { continue };
        let mut repos: Vec<_> = repos.flatten().filter(|e| e.path().is_dir()).collect();
        repos.sort_by_key(|e| e.file_name());
        for repo in repos {
            let id = format!("{}/{}", vendor.file_name().to_string_lossy(), repo.file_name().to_string_lossy());
            found.push((id, repo.path()));
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{write_decide, write_laya, Scratch};
    use crate::manifest::FILE;

    fn store() -> Scratch {
        let s = Scratch::new("store");
        write_laya(&s.join("convaiinnovations/laya"), 48);
        s
    }

    #[test]
    fn an_installed_laya_checkpoint_is_ready_and_unvalidated_by_default() {
        let s = store();
        let r = inspect("convaiinnovations/laya", &s.join("convaiinnovations/laya"), false);
        assert_eq!(r.readiness, Readiness::Ready);
        assert_eq!(r.backend, Some(Backend::Laya));
        let c = r.candidate().unwrap();
        assert!(c.installed && !c.validated);
        assert_eq!(r.to_json()["validated"], false);
    }

    #[test]
    fn a_missing_directory_is_not_installed_and_says_what_to_pull() {
        let s = Scratch::new("store-missing");
        let r = inspect("acme/gone", &s.join("acme/gone"), true);
        assert!(matches!(r.readiness, Readiness::NotInstalled { .. }));
        assert_eq!(r.to_json()["hint"], "run `brain pull acme/gone`");
        assert!(r.candidate().is_none(), "no known architecture, no candidate");
    }

    #[test]
    fn a_laya_checkpoint_missing_its_weights_names_the_missing_file() {
        let s = store();
        std::fs::remove_file(s.join("convaiinnovations/laya/model.safetensors")).unwrap();
        let r = inspect("convaiinnovations/laya", &s.join("convaiinnovations/laya"), false);
        assert_eq!(r.readiness, Readiness::NotInstalled { missing: vec!["model.safetensors".into()] });
        assert!(!r.candidate().unwrap().installed);
    }

    #[test]
    fn a_saved_decide_model_is_ready_but_a_bare_encoder_is_not_a_decision_model() {
        let s = Scratch::new("store-decide");
        write_decide(&s.join("acme/triage"), 1);
        assert_eq!(inspect("acme/triage", &s.join("acme/triage"), false).readiness, Readiness::Ready);
        // A published encoder's config.json carries no head mark.
        std::fs::write(s.join("acme/triage/config.json"), r#"{"model_type": "bert"}"#).unwrap();
        let r = inspect("acme/triage", &s.join("acme/triage"), false);
        assert!(matches!(r.readiness, Readiness::Incompatible { .. }), "{:?}", r.readiness);
    }

    #[test]
    fn a_broken_manifest_makes_the_model_incompatible_not_silently_unvalidated() {
        let s = store();
        std::fs::write(s.join("convaiinnovations/laya").join(FILE), "{").unwrap();
        let r = inspect("convaiinnovations/laya", &s.join("convaiinnovations/laya"), false);
        assert!(matches!(r.readiness, Readiness::Incompatible { .. }), "{:?}", r.readiness);
    }

    #[test]
    fn specs_resolve_to_store_positions_or_local_ids() {
        let root = Path::new("/models");
        assert_eq!(resolve_spec(Some(root), "a/b"), ("a/b".into(), PathBuf::from("/models/a/b")));
        assert_eq!(resolve_spec(Some(root), "/models/a/b"), ("a/b".into(), PathBuf::from("/models/a/b")));
        assert_eq!(resolve_spec(Some(root), "/elsewhere/mine"), ("local/mine".into(), PathBuf::from("/elsewhere/mine")));
    }

    #[test]
    fn the_catalog_lists_configured_models_first_even_when_absent_and_scans_the_store() {
        let s = store();
        let c = Catalog::new(Some(s.to_path_buf()), vec!["acme/gone".into()], None);
        let recs = c.records();
        let ids: Vec<&str> = recs.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["acme/gone", "convaiinnovations/laya"]);
        assert!(recs[0].configured && !recs[1].configured);
    }

    #[test]
    fn a_store_directory_that_is_not_a_decision_model_is_not_offered() {
        let s = store();
        std::fs::create_dir_all(s.join("other/llm")).unwrap();
        std::fs::write(s.join("other/llm/config.json"), r#"{"model_type": "llama"}"#).unwrap();
        let ids: Vec<String> = Catalog::new(Some(s.to_path_buf()), vec![], None).records().into_iter().map(|r| r.id).collect();
        assert_eq!(ids, ["convaiinnovations/laya"]);
    }

    #[test]
    fn auto_never_picks_an_unvalidated_model_and_explains_it() {
        let s = store();
        let c = Catalog::new(Some(s.to_path_buf()), vec![], None);
        let sel = c.select("auto");
        assert_eq!(sel.chosen, None);
        assert!(sel.reason.contains("convaiinnovations/laya"), "{}", sel.reason);
    }

    #[test]
    fn auto_picks_a_validated_model_once_its_validation_is_recorded() {
        let s = store();
        let dir = s.join("convaiinnovations/laya");
        let digest = brain_modelstore::fetch::file_digest(&dir.join("model.safetensors")).unwrap();
        std::fs::write(
            dir.join(FILE),
            format!(r#"{{"schema": 1, "weights_sha256": "{digest}", "validation": {{"status": "validated", "profile": "routing"}}}}"#),
        )
        .unwrap();
        let sel = Catalog::new(Some(s.to_path_buf()), vec![], None).select("auto");
        assert_eq!(sel.chosen.as_deref(), Some("convaiinnovations/laya"));
    }

    #[test]
    fn an_explicit_id_is_taken_as_named_and_an_unknown_one_selects_nothing() {
        let s = store();
        let c = Catalog::new(Some(s.to_path_buf()), vec![], None);
        assert_eq!(c.select("convaiinnovations/laya").chosen.as_deref(), Some("convaiinnovations/laya"));
        assert_eq!(c.select("nobody/home").chosen, None);
    }

    #[test]
    fn the_configured_default_is_what_auto_resolves_to() {
        let s = store();
        let c = Catalog::new(Some(s.to_path_buf()), vec![], Some("convaiinnovations/laya".into()));
        let sel = c.select("auto");
        assert_eq!(sel.chosen.as_deref(), Some("convaiinnovations/laya"));
        assert_eq!(sel.rule.slug(), "configured");
    }
}
