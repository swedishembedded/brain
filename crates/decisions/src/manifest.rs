// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The decision model manifest: what a checkpoint IS, as opposed to what it
//! can do.
//!
//! A decision model is called with options its caller invents, so a caller
//! must be able to ask, before trusting it, what it was built from and
//! whether anyone has measured it. The manifest is a sidecar file,
//! [`FILE`], beside the weights:
//!
//! ```json
//! {"schema": 1,
//!  "capability": "decide",
//!  "architecture": "laya",
//!  "weights_sha256": "sha256:...", "tokenizer_sha256": "sha256:...",
//!  "lineage": {"base": "convaiinnovations/laya", "trained_on": "...", "recipe": "..."},
//!  "max_input_tokens": 512,
//!  "question_types": ["choice", "score", "noul"],
//!  "validation": {"status": "validated", "profile": "ticket-routing", "evidence": "..."}}
//! ```
//!
//! # Validation is a claim someone recorded, never a default
//!
//! A checkpoint with no manifest, or a manifest with no validation, is
//! [`Validation::Unvalidated`]. A manifest cannot be parsed into
//! [`Validation::Validated`] without the profile it was validated on AND the
//! weights hash the validation applies to: validation measures particular
//! weights, so a claim not bound to them would stay true after the weights
//! changed. At load, a declared hash that does not match the file is a load
//! failure, which is what keeps a recorded validation attached to the bytes it
//! was measured on.
//!
//! Swedish Embedded AB implements model provenance for its clients: knowing
//! which weights answered, what they descend from and what was measured
//! before they were trusted. If your team needs expertise in decision-model
//! governance, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

use serde_json::{json, Value};

use crate::engine::Backend;
use crate::wire::QuestionType;

/// The manifest's file name, inside the checkpoint directory.
pub const FILE: &str = "decision.manifest.json";
/// The only schema this reader understands.
pub const SCHEMA: u64 = 1;
/// The capability every decision model serves.
pub const CAPABILITY: &str = "decide";

/// Whether, and on what, a model was measured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Validation {
    /// Nothing recorded. The only state a model reaches without someone
    /// writing a validation into its manifest.
    Unvalidated,
    Validated {
        /// The task profile it was measured on.
        profile: String,
        /// Where the measurement can be found, if recorded.
        evidence: Option<String>,
    },
}

impl Validation {
    pub fn is_validated(&self) -> bool {
        matches!(self, Validation::Validated { .. })
    }
}

/// Where the weights came from.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Lineage {
    /// The model this one descends from.
    pub base: Option<String>,
    /// What it was trained on.
    pub trained_on: Option<String>,
    /// How it was trained.
    pub recipe: Option<String>,
}

/// A checkpoint's declared identity and provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecisionManifest {
    pub capability: String,
    pub architecture: Backend,
    /// `sha256:<hex>` of `model.safetensors`, when declared.
    pub weights_sha256: Option<String>,
    /// `sha256:<hex>` of the tokenizer file, when declared.
    pub tokenizer_sha256: Option<String>,
    pub lineage: Lineage,
    /// The most input tokens the model holds, when declared; otherwise the
    /// loaded engine's own figure is authoritative.
    pub max_input_tokens: Option<usize>,
    pub question_types: Vec<QuestionType>,
    pub validation: Validation,
}

impl DecisionManifest {
    /// What a checkpoint with no manifest is: unvalidated, with nothing
    /// declared beyond the architecture its structure names.
    pub fn undeclared(architecture: Backend) -> DecisionManifest {
        DecisionManifest {
            capability: CAPABILITY.to_string(),
            architecture,
            weights_sha256: None,
            tokenizer_sha256: None,
            lineage: Lineage::default(),
            max_input_tokens: None,
            question_types: QuestionType::ALL.to_vec(),
            validation: Validation::Unvalidated,
        }
    }

    /// Parse manifest text. `structural` is the architecture the directory's
    /// structure names; a manifest declaring another is refused, since the
    /// loader would otherwise follow the file into the wrong engine.
    pub fn parse(text: &str, structural: Backend) -> Result<DecisionManifest, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| format!("{FILE} is not valid JSON: {e}"))?;
        let obj = v.as_object().ok_or_else(|| format!("{FILE} must be a JSON object"))?;
        match obj.get("schema").and_then(Value::as_u64) {
            Some(SCHEMA) => {}
            other => return Err(format!("{FILE}: `schema` must be {SCHEMA}, got {}", other.map_or("nothing".to_string(), |n| n.to_string()))),
        }
        let capability = obj.get("capability").and_then(Value::as_str).unwrap_or(CAPABILITY);
        if capability != CAPABILITY {
            return Err(format!("{FILE}: capability {capability:?} is not {CAPABILITY:?}, the only capability decision models serve"));
        }
        let architecture = match obj.get("architecture").and_then(Value::as_str) {
            None => structural,
            Some("decide") => Backend::Decide,
            Some("laya") => Backend::Laya,
            Some(other) => return Err(format!("{FILE}: architecture {other:?} is neither \"decide\" nor \"laya\"")),
        };
        if architecture != structural {
            return Err(format!(
                "{FILE} declares architecture {:?} but the directory is laid out as {:?}",
                architecture.slug(),
                structural.slug()
            ));
        }
        let hash = |key: &str| -> Result<Option<String>, String> {
            match obj.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(h)) if is_digest(h) => Ok(Some(h.clone())),
                Some(_) => Err(format!("{FILE}: `{key}` must be \"sha256:\" followed by 64 lowercase hex digits")),
            }
        };
        let weights_sha256 = hash("weights_sha256")?;
        let tokenizer_sha256 = hash("tokenizer_sha256")?;

        let text_of = |o: &serde_json::Map<String, Value>, k: &str| o.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
        let lineage = match obj.get("lineage") {
            Some(Value::Object(l)) => Lineage { base: text_of(l, "base"), trained_on: text_of(l, "trained_on"), recipe: text_of(l, "recipe") },
            None | Some(Value::Null) => Lineage::default(),
            Some(_) => return Err(format!("{FILE}: `lineage` must be an object")),
        };
        let max_input_tokens = match obj.get("max_input_tokens") {
            None | Some(Value::Null) => None,
            Some(n) => match n.as_u64().filter(|&n| n > 0) {
                Some(n) => Some(n as usize),
                None => return Err(format!("{FILE}: `max_input_tokens` must be a positive integer")),
            },
        };
        let question_types = match obj.get("question_types") {
            None | Some(Value::Null) => QuestionType::ALL.to_vec(),
            Some(Value::Array(items)) => {
                let mut types = Vec::new();
                for item in items {
                    let slug = item.as_str().unwrap_or_default();
                    let t = QuestionType::parse(slug).ok_or_else(|| format!("{FILE}: unknown question type {item}"))?;
                    if !types.contains(&t) {
                        types.push(t);
                    }
                }
                if types.is_empty() {
                    return Err(format!("{FILE}: `question_types` must name at least one type"));
                }
                types
            }
            Some(_) => return Err(format!("{FILE}: `question_types` must be an array")),
        };
        let validation = match obj.get("validation") {
            None | Some(Value::Null) => Validation::Unvalidated,
            Some(Value::Object(val)) => match val.get("status").and_then(Value::as_str) {
                Some("unvalidated") => Validation::Unvalidated,
                Some("validated") => {
                    let profile = text_of(val, "profile").ok_or_else(|| format!("{FILE}: a validated model must name the `profile` it was validated on"))?;
                    if weights_sha256.is_none() {
                        return Err(format!("{FILE}: a validated model must declare `weights_sha256`, the weights the validation applies to"));
                    }
                    Validation::Validated { profile, evidence: text_of(val, "evidence") }
                }
                other => return Err(format!("{FILE}: validation status must be \"unvalidated\" or \"validated\", got {other:?}")),
            },
            Some(_) => return Err(format!("{FILE}: `validation` must be an object")),
        };
        Ok(DecisionManifest { capability: CAPABILITY.to_string(), architecture, weights_sha256, tokenizer_sha256, lineage, max_input_tokens, question_types, validation })
    }

    /// The manifest in `dir`, or [`DecisionManifest::undeclared`] when there
    /// is none. A manifest that is present and wrong is an error, never
    /// silently replaced by the unvalidated default: that would hide a typo in
    /// a validation record behind a model that still loads.
    pub fn read(dir: &Path, structural: Backend) -> Result<DecisionManifest, String> {
        let path = dir.join(FILE);
        match std::fs::read_to_string(&path) {
            Ok(text) => DecisionManifest::parse(&text, structural),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(DecisionManifest::undeclared(structural)),
            Err(e) => Err(format!("{FILE}: {e}")),
        }
    }

    /// The manifest as JSON, in the shape [`DecisionManifest::parse`] reads.
    pub fn to_json(&self) -> Value {
        let validation = match &self.validation {
            Validation::Unvalidated => json!({"status": "unvalidated"}),
            Validation::Validated { profile, evidence } => json!({"status": "validated", "profile": profile, "evidence": evidence}),
        };
        json!({
            "schema": SCHEMA,
            "capability": self.capability,
            "architecture": self.architecture.slug(),
            "weights_sha256": self.weights_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "lineage": {"base": self.lineage.base, "trained_on": self.lineage.trained_on, "recipe": self.lineage.recipe},
            "max_input_tokens": self.max_input_tokens,
            "question_types": self.question_types.iter().map(|t| t.slug()).collect::<Vec<_>>(),
            "validation": validation,
        })
    }
}

fn is_digest(h: &str) -> bool {
    h.strip_prefix("sha256:").is_some_and(|hex| hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
}

/// Check the declared hashes against the files in `dir` - the step that keeps
/// a recorded validation attached to the bytes it was measured on. Hashes only
/// what the manifest declares, so an undeclared manifest costs nothing.
///
/// `weights` and `tokenizer` are the file names relative to `dir`.
pub fn verify_identity(m: &DecisionManifest, dir: &Path, weights: &str, tokenizer: &str) -> Result<(), String> {
    for (declared, file) in [(&m.weights_sha256, weights), (&m.tokenizer_sha256, tokenizer)] {
        let Some(declared) = declared else { continue };
        let actual = brain_modelstore::fetch::file_digest(&dir.join(file)).map_err(|e| format!("hashing {file}: {e}"))?;
        if &actual != declared {
            return Err(format!("{file} does not match the manifest: it declares {declared} but the file is {actual}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: &str = "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn validated(extra: &str) -> String {
        format!(r#"{{"schema": 1, "architecture": "laya", "weights_sha256": "{H}", "validation": {{"status": "validated", "profile": "ticket-routing"{extra}}}}}"#)
    }

    #[test]
    fn no_manifest_means_unvalidated_and_never_validated() {
        let d = crate::fixtures::Scratch::new("manifest-absent");
        let m = DecisionManifest::read(&d, Backend::Laya).unwrap();
        assert_eq!(m.validation, Validation::Unvalidated);
        assert!(!m.validation.is_validated());
        assert_eq!(m.question_types, QuestionType::ALL.to_vec());
    }

    #[test]
    fn a_manifest_without_a_validation_block_is_unvalidated() {
        let m = DecisionManifest::parse(r#"{"schema": 1}"#, Backend::Decide).unwrap();
        assert_eq!(m.validation, Validation::Unvalidated);
        assert_eq!(m.architecture, Backend::Decide);
    }

    #[test]
    fn a_validated_model_names_its_profile_and_the_weights_it_applies_to() {
        let m = DecisionManifest::parse(&validated(r#", "evidence": "run 7""#), Backend::Laya).unwrap();
        assert_eq!(m.validation, Validation::Validated { profile: "ticket-routing".into(), evidence: Some("run 7".into()) });
        assert_eq!(m.weights_sha256.as_deref(), Some(H));
    }

    #[test]
    fn a_validated_claim_without_a_profile_or_without_weights_identity_is_refused() {
        let no_profile = format!(r#"{{"schema": 1, "weights_sha256": "{H}", "validation": {{"status": "validated"}}}}"#);
        assert!(DecisionManifest::parse(&no_profile, Backend::Laya).unwrap_err().contains("profile"));
        let no_hash = r#"{"schema": 1, "validation": {"status": "validated", "profile": "p"}}"#;
        assert!(DecisionManifest::parse(no_hash, Backend::Laya).unwrap_err().contains("weights_sha256"));
        let unknown = r#"{"schema": 1, "validation": {"status": "probably"}}"#;
        assert!(DecisionManifest::parse(unknown, Backend::Laya).unwrap_err().contains("probably"));
    }

    #[test]
    fn a_manifest_that_disagrees_with_the_directory_layout_is_refused() {
        let err = DecisionManifest::parse(r#"{"schema": 1, "architecture": "decide"}"#, Backend::Laya).unwrap_err();
        assert!(err.contains("decide") && err.contains("laya"), "{err}");
    }

    #[test]
    fn malformed_manifests_are_refused_by_field() {
        let cases = [
            ("nope", "valid JSON"),
            ("[]", "object"),
            (r#"{"schema": 2}"#, "schema"),
            (r#"{"schema": 1, "capability": "generate"}"#, "capability"),
            (r#"{"schema": 1, "weights_sha256": "abc"}"#, "weights_sha256"),
            (r#"{"schema": 1, "max_input_tokens": 0}"#, "max_input_tokens"),
            (r#"{"schema": 1, "question_types": []}"#, "question_types"),
            (r#"{"schema": 1, "question_types": ["vibes"]}"#, "vibes"),
            (r#"{"schema": 1, "lineage": 3}"#, "lineage"),
        ];
        for (text, needle) in cases {
            let err = DecisionManifest::parse(text, Backend::Laya).unwrap_err();
            assert!(err.contains(needle), "{text}: expected {needle:?} in {err}");
        }
    }

    #[test]
    fn a_present_but_wrong_manifest_is_an_error_not_the_default() {
        let d = crate::fixtures::Scratch::new("manifest-wrong");
        std::fs::write(d.join(FILE), "{").unwrap();
        assert!(DecisionManifest::read(&d, Backend::Laya).is_err());
    }

    #[test]
    fn the_manifest_round_trips_through_json() {
        let m = DecisionManifest::parse(&validated(""), Backend::Laya).unwrap();
        let back = DecisionManifest::parse(&m.to_json().to_string(), Backend::Laya).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn a_declared_hash_must_match_the_file() {
        let d = crate::fixtures::Scratch::new("manifest-identity");
        std::fs::write(d.join("w.bin"), b"abc").unwrap();
        std::fs::write(d.join("t.json"), b"{}").unwrap();
        let right = brain_modelstore::fetch::file_digest(&d.join("w.bin")).unwrap();
        let mut m = DecisionManifest::undeclared(Backend::Laya);
        assert!(verify_identity(&m, &d, "w.bin", "t.json").is_ok(), "nothing declared, nothing to check");
        m.weights_sha256 = Some(right);
        assert!(verify_identity(&m, &d, "w.bin", "t.json").is_ok());
        m.tokenizer_sha256 = Some(H.to_string());
        let err = verify_identity(&m, &d, "w.bin", "t.json").unwrap_err();
        assert!(err.contains("t.json") && err.contains("does not match"), "{err}");
    }
}
