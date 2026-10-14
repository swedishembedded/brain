// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A Laya checkpoint that carries only its trained decision head.
//!
//! A stock Laya directory holds a 395M-parameter ModernBERT trunk and a
//! decision head in one `model.safetensors`. Training the head with the trunk
//! frozen (the only mode the Laya arm trains in) changes only the head, so a
//! trained directory that copied the trunk would put 1.6 GB of unchanged bytes
//! on disk for every checkpoint of every experiment. This layout names the
//! trunk instead:
//!
//! ```text
//! <dir>/
//!   decision.manifest.json      lineage, hashes, validation (see `manifest`)
//!   laya.base.json              the backbone, by store id + content digest
//!   head.safetensors            the trained head (`LayaDecision::save_head`)
//!   rl_agent_config.json        copied: context lengths + serving calibration
//!   encoder/config.json         copied: the trunk's architecture
//!   tokenizer/tokenizer.json    copied: what the head was trained to read
//! ```
//!
//! The three copied files are small, and they have to be in the directory:
//! the directory's STRUCTURE is how [`crate::engine::Backend::sniff`] knows it
//! is a Laya checkpoint, and the calibration tables and tokenizer are part of
//! what the head's probabilities mean, so they are covered by the model's
//! identity ([`crate::manifest::weights_identity`]) rather than read from
//! wherever the base happens to be.
//!
//! # The base reference
//!
//! [`BaseRef`] is `{id, sha256, path}`. `sha256` is
//! `modelstore::fetch::base_digest` of the stock directory - the digest of its
//! weight files - and it is the only thing that identifies the base: `id` and
//! `path` are where to look. [`BaseRef::resolve`] tries, in order, the model
//! store the trained directory itself sits in (`<root>/<vendor>/<repo>`), the
//! default model store, and the recorded `path`, and takes the first that
//! exists AND hashes to `sha256`. A base that is absent is "not installed"; a
//! base that is present with other bytes is refused, because a head trained
//! against one trunk answers noise on another.
//!
//! A trained directory is never itself a base: `train` names the stock
//! checkpoint, so a base reference is always one hop to real weights.
//!
//! Swedish Embedded AB implements model artifacts that reference their
//! foundation by content instead of copying it, so a fleet of fine-tunes costs
//! the head's bytes and not the backbone's. If your team needs expertise in
//! checkpoint layout and provenance, you can procure our services by sending
//! an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::manifest::write_atomic;

/// The base reference's file name; its presence is what marks the layout.
pub const BASE_FILE: &str = "laya.base.json";
/// The trained head's file name.
pub const HEAD_FILE: &str = "head.safetensors";
/// The small files copied from the base so the directory is a complete Laya
/// checkpoint structurally.
const COPIED: [&str; 3] = ["rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"];
const SCHEMA: u64 = 1;

/// Whether `dir` is laid out as a trained head that references its base.
pub fn is_head_only(dir: &Path) -> bool {
    dir.join(BASE_FILE).is_file()
}

/// The backbone a head-only directory attaches to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BaseRef {
    /// The store id (`<vendor>/<repo>`) the base is addressed by.
    pub id: String,
    /// `sha256:<hex>` identity of the base's weight files.
    pub sha256: String,
    /// Where the base was when this reference was written; a hint only.
    pub path: Option<String>,
}

impl BaseRef {
    /// The reference to the stock Laya checkpoint at `dir`, addressed as `id`.
    /// Hashes the base's weight files; a caller that already holds the digest
    /// (`modelstore::fetch::base_digest`) uses [`BaseRef::with_digest`].
    pub fn of(id: &str, dir: &Path) -> Result<BaseRef, String> {
        let sha256 = brain_modelstore::fetch::base_digest(dir).map_err(|e| format!("hashing the base {id}: {e}"))?;
        BaseRef::with_digest(id, dir, sha256)
    }

    /// The reference to the stock Laya checkpoint at `dir` whose weights hash
    /// to `sha256`. Refuses anything that is not a stock Laya directory, a
    /// head-only one included.
    pub fn with_digest(id: &str, dir: &Path, sha256: String) -> Result<BaseRef, String> {
        check_id(id)?;
        if !crate::engine::is_laya_dir(dir) {
            return Err(format!("the base {id} is not a Laya checkpoint"));
        }
        if is_head_only(dir) {
            return Err(format!("the base {id} is itself a trained head; name the stock checkpoint it was trained from"));
        }
        let path = dir.canonicalize().ok().and_then(|p| p.to_str().map(str::to_string));
        Ok(BaseRef { id: id.to_string(), sha256, path })
    }

    /// The reference recorded in `dir`: `None` when `dir` is not head-only, an
    /// error when the file is there and is not a valid reference.
    pub fn read(dir: &Path) -> Result<Option<BaseRef>, String> {
        let text = match std::fs::read_to_string(dir.join(BASE_FILE)) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("{BASE_FILE}: {e}")),
        };
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{BASE_FILE} is not valid JSON: {e}"))?;
        if v.get("schema").and_then(Value::as_u64) != Some(SCHEMA) {
            return Err(format!("{BASE_FILE}: `schema` must be {SCHEMA}"));
        }
        let text_of = |k: &str| v.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
        let id = text_of("id").ok_or_else(|| format!("{BASE_FILE}: `id` is required"))?;
        check_id(&id).map_err(|e| format!("{BASE_FILE}: {e}"))?;
        let sha256 = text_of("sha256").filter(|h| crate::manifest::is_digest(h)).ok_or_else(|| format!("{BASE_FILE}: `sha256` must be \"sha256:\" followed by 64 lowercase hex digits"))?;
        Ok(Some(BaseRef { id, sha256, path: text_of("path") }))
    }

    fn write(&self, dir: &Path) -> Result<(), String> {
        let v = json!({"schema": SCHEMA, "id": self.id, "sha256": self.sha256, "path": self.path});
        let text = serde_json::to_string_pretty(&v).map_err(|e| format!("{BASE_FILE}: {e}"))?;
        write_atomic(&dir.join(BASE_FILE), (text + "\n").as_bytes())
    }

    /// Where the base may be, most authoritative first.
    fn candidates(&self, dir: &Path) -> Vec<PathBuf> {
        let mut found: Vec<PathBuf> = Vec::new();
        let sibling_store = dir.canonicalize().ok().and_then(|d| d.parent()?.parent().map(Path::to_path_buf));
        let roots = sibling_store.into_iter().chain(brain_modelstore::default_root());
        for candidate in roots.map(|r| r.join(&self.id)).chain(self.path.iter().map(PathBuf::from)) {
            if !found.contains(&candidate) {
                found.push(candidate);
            }
        }
        found
    }

    /// The first candidate directory that exists, without verifying it.
    pub fn locate(&self, dir: &Path) -> Option<PathBuf> {
        self.candidates(dir).into_iter().find(|c| c.is_dir() && !is_head_only(c))
    }

    /// The directory holding the base's weights: the first candidate that
    /// exists and hashes to [`BaseRef::sha256`]. The error text names the id
    /// and digests but no path, because it reaches remote callers.
    pub fn resolve(&self, dir: &Path) -> Result<PathBuf, String> {
        let mut mismatched: Option<String> = None;
        for candidate in self.candidates(dir).into_iter().filter(|c| c.is_dir() && !is_head_only(c)) {
            let found = brain_modelstore::fetch::base_digest(&candidate).map_err(|e| format!("hashing the base {}: {e}", self.id))?;
            if found == self.sha256 {
                return Ok(candidate);
            }
            mismatched.get_or_insert(found);
        }
        Err(match mismatched {
            Some(found) => format!("the base {} is installed but is not the one this head was trained on: expected {}, found {found}", self.id, self.sha256),
            None => format!("the base {} is not installed (expected {}): run `brain pull {}`", self.id, self.sha256, self.id),
        })
    }
}

/// A base id is a store address, `<vendor>/<repo>`; refusing anything else
/// keeps a hand-edited reference from steering a lookup out of the store.
fn check_id(id: &str) -> Result<(), String> {
    let ok = id.split('/').count() == 2 && id.split('/').all(|s| !s.is_empty() && s != "." && s != ".." && !s.contains('\\'));
    if ok { Ok(()) } else { Err(format!("base id {id:?} is not of the form <vendor>/<repo>")) }
}

/// Lay out `dir` as a head-only checkpoint of `base`: the base reference and
/// the small files copied from `base_dir`. The caller writes [`HEAD_FILE`]
/// (`DecisionPipeline::save_head`) and the manifest; `dir` must exist.
pub fn write_layout(dir: &Path, base_dir: &Path, base: &BaseRef) -> Result<(), String> {
    for file in COPIED {
        let to = dir.join(file);
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        std::fs::copy(base_dir.join(file), &to).map_err(|e| format!("copying {file} from the base: {e}"))?;
    }
    base.write(dir)
}

/// The identity of a head-only directory's weights: its head, the base it
/// names, and the calibration tables that turn the head's logits into
/// probabilities. It plays the part `model.safetensors`' digest plays for a
/// stock checkpoint, so a recorded validation stays attached to exactly this
/// model. The base is covered by its recorded digest; [`BaseRef::resolve`]
/// is what binds that digest to bytes at load.
pub fn identity(dir: &Path, base: &BaseRef) -> Result<String, String> {
    let digest = |file: &str| brain_modelstore::fetch::file_digest(&dir.join(file)).map_err(|e| format!("hashing {file}: {e}"));
    let text = format!("laya-head-only/1\nhead={}\nbase={}\nrl_agent_config={}\n", digest(HEAD_FILE)?, base.sha256, digest("rl_agent_config.json")?);
    Ok(format!("sha256:{}", brain_modelstore::fetch::bytes_digest(text.as_bytes())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Backend, LayaWeights};
    use crate::fixtures::{write_laya, write_laya_head_only, Scratch};
    use crate::manifest::{verify_identity, weights_identity, DecisionManifest};

    fn base_in(root: &Path) -> PathBuf {
        let base = root.join("base");
        write_laya(&base, 48);
        base
    }

    #[test]
    fn the_layout_references_the_backbone_instead_of_copying_it() {
        let s = Scratch::new("head-only-layout");
        let base = base_in(&s);
        let out = s.join("out");
        write_laya_head_only(&out, &base, "acme/laya", 99);

        assert!(!out.join("model.safetensors").exists(), "the trunk must not be duplicated");
        assert!(is_head_only(&out) && Backend::sniff(&out) == Some(Backend::Laya));
        let size = |p: &Path| std::fs::metadata(p).unwrap().len();
        assert!(size(&out.join(HEAD_FILE)) < size(&base.join("model.safetensors")), "the head is a part of the model, not all of it");
        let r = BaseRef::read(&out).unwrap().unwrap();
        assert_eq!((r.id.as_str(), r.sha256.as_str()), ("acme/laya", brain_modelstore::fetch::base_digest(&base).unwrap().as_str()));
    }

    #[test]
    fn a_head_only_directory_reads_as_the_base_with_the_trained_head_in_place() {
        let s = Scratch::new("head-only-read");
        let base = base_in(&s);
        let out = s.join("out");
        write_laya_head_only(&out, &base, "acme/laya", 99);

        let via_reference = LayaWeights::read(out.to_str().unwrap(), None).unwrap();
        let explicit = LayaWeights::read(base.to_str().unwrap(), Some(out.join(HEAD_FILE).to_str().unwrap())).unwrap();
        assert_eq!(via_reference.max_len(), explicit.max_len());
        assert_eq!(via_reference.head_tensors(), explicit.head_tensors(), "the head in the directory is the head that is served");
        let stock = LayaWeights::read(base.to_str().unwrap(), None).unwrap();
        assert_ne!(via_reference.head_tensors(), stock.head_tensors(), "and it is not the base's own");
    }

    #[test]
    fn the_base_is_found_in_the_store_the_directory_sits_in_without_a_recorded_path() {
        let s = Scratch::new("head-only-store");
        write_laya(&s.join("convai/laya"), 48);
        let out = s.join("acme/trained");
        write_laya_head_only(&out, &s.join("convai/laya"), "convai/laya", 99);
        let text = std::fs::read_to_string(out.join(BASE_FILE)).unwrap();
        let mut v: Value = serde_json::from_str(&text).unwrap();
        v["path"] = Value::Null;
        std::fs::write(out.join(BASE_FILE), v.to_string()).unwrap();

        let r = BaseRef::read(&out).unwrap().unwrap();
        assert_eq!(r.path, None);
        assert_eq!(r.resolve(&out).unwrap().canonicalize().unwrap(), s.join("convai/laya").canonicalize().unwrap());
    }

    #[test]
    fn an_absent_base_is_not_installed_and_another_base_is_refused_and_neither_names_a_path() {
        let s = Scratch::new("head-only-base");
        let base = base_in(&s);
        let out = s.join("out");
        write_laya_head_only(&out, &base, "acme/laya", 99);
        let r = BaseRef::read(&out).unwrap().unwrap();

        // Same place, different bytes: a head trained on one trunk is noise on another.
        write_laya(&base, 48);
        let weights = base.join("model.safetensors");
        let mut bytes = std::fs::read(&weights).unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        std::fs::write(&weights, bytes).unwrap();
        let e = r.resolve(&out).unwrap_err();
        assert!(e.contains("not the one this head was trained on") && !e.contains(s.to_str().unwrap()), "{e}");

        std::fs::remove_dir_all(&base).unwrap();
        let e = r.resolve(&out).unwrap_err();
        assert!(e.contains("not installed") && e.contains("brain pull acme/laya") && !e.contains(s.to_str().unwrap()), "{e}");
    }

    #[test]
    fn a_trained_directory_is_not_a_base_and_a_base_id_cannot_leave_the_store() {
        let s = Scratch::new("head-only-chain");
        let base = base_in(&s);
        let out = s.join("out");
        write_laya_head_only(&out, &base, "acme/laya", 99);
        assert!(BaseRef::of("acme/again", &out).unwrap_err().contains("itself a trained head"));
        for bad in ["../x/y", "a/..", "one", "a/b/c", "/abs/x", ""] {
            assert!(BaseRef::of(bad, &base).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn the_weights_identity_covers_the_head_the_base_and_the_calibration() {
        let s = Scratch::new("head-only-identity");
        let base = base_in(&s);
        let a = s.join("a");
        let b = s.join("b");
        write_laya_head_only(&a, &base, "acme/laya", 99);
        write_laya_head_only(&b, &base, "acme/laya", 100);
        let id = |d: &Path| weights_identity(d, Backend::Laya).unwrap();
        assert_ne!(id(&a), id(&b), "another head is another model");
        assert_eq!(id(&a), id(&a));

        let before = id(&a);
        std::fs::write(a.join("rl_agent_config.json"), r#"{"head_layers": 2, "max_len": 48, "head_max_len": 32, "max_prefixes": 6, "temperature": [2.0, 1.0, 1.0], "temperature_by_options": {}}"#).unwrap();
        assert_ne!(id(&a), before, "the calibration changes what the head's probabilities mean");

        let mut m = DecisionManifest::undeclared(Backend::Laya);
        m.weights_sha256 = Some(id(&b));
        assert!(verify_identity(&m, &b, Backend::Laya).is_ok());
        assert!(verify_identity(&m, &a, Backend::Laya).is_err());
    }

    #[test]
    fn a_head_that_leaves_a_tensor_out_is_refused_by_name() {
        let s = Scratch::new("head-only-incomplete");
        let base = base_in(&s);
        let out = s.join("out");
        write_laya_head_only(&out, &base, "acme/laya", 99);
        let mut tensors = checkpoint::safetensors::read(out.join(HEAD_FILE).to_str().unwrap()).unwrap();
        let dropped = tensors.pop().unwrap().name;
        let kept: Vec<(String, Vec<u64>, Vec<f32>)> = tensors.into_iter().map(|t| (t.name, vec![t.data.len() as u64], t.data)).collect();
        checkpoint::save_carded(out.join(HEAD_FILE).to_str().unwrap(), json!({}), &kept, &checkpoint::st::ModelCard::default());
        let e = LayaWeights::read(out.to_str().unwrap(), None).err().expect("must fail");
        assert!(e.contains(&dropped), "{e}");
    }
}
