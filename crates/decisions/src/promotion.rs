// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Promotion: the only way a trained checkpoint becomes one the `auto`
//! selector may choose.
//!
//! A [`Policy`] declares what held-out evidence is enough; [`Policy::judge`]
//! compares an evaluation [`Report`] with it, gate by gate; [`commit`] records
//! the outcome in the checkpoint's manifest. Nothing here trusts a number
//! stored with a checkpoint: the caller re-evaluates the weights on the test
//! split and passes the fresh report in.
//!
//! # The policy file
//!
//! ```json
//! {"schema": 1,
//!  "profile": "tool-routing",
//!  "min_test_examples": 100,
//!  "min_accuracy_lift": 0.10,
//!  "max_ece": 0.10,
//!  "max_order_instability": 0.05,
//!  "max_regret": 1.0,
//!  "costs": {"false_exclusion": 5.0, "wasted_tokens": 1.0},
//!  "required_families": ["tool_selection", "noisy_output"],
//!  "order_seed": 0}
//! ```
//!
//! `profile` and `min_test_examples` are required, and so is at least one of
//! the four quality gates (a policy that only counts examples would promote
//! anything). `max_regret` needs `costs`. Unknown keys are refused, so a
//! misspelt gate cannot silently disappear. Gates:
//!
//! * `min_test_examples` - the test split holds at least this many examples;
//! * `min_accuracy_lift` - accuracy minus the baseline accuracy;
//! * `max_ece` - expected calibration error of the top probability;
//! * `max_order_instability` - the share of winners that change when the
//!   options are reordered;
//! * `max_regret` - mean cost per example of the model's own trust-or-defer
//!   policy, in the units of the cost file;
//! * `required_families` - every named family has test examples, and, when
//!   `min_accuracy_lift` is set, meets it within that family. A model trained
//!   on one family cannot be promoted for another.
//!
//! # What a promotion changes
//!
//! At most one checkpoint per profile is validated. [`commit`] writes the
//! evaluation record beside the manifest, un-validates the checkpoint it
//! supersedes, writes the validation (profile, evidence and `weights_sha256`)
//! into the new manifest, and then points the pointer file at it, keeping the
//! superseded one as the last-known-good. [`rollback`] swaps them back. The
//! order of those writes means a crash can leave a profile with no validated
//! checkpoint, which `auto` answers by choosing nothing, and never one with an
//! unmeasured checkpoint validated; re-running the command completes it.
//!
//! The pointer is `decision.promoted.<profile>.json` in the directory that
//! holds the checkpoints, so the checkpoints of a profile are siblings.
//!
//! Swedish Embedded AB implements release gates for learned components: a
//! policy that says what evidence is enough, a check that does not trust the
//! candidate's own numbers, and a rollback that works. If your team needs
//! expertise in governing model promotion, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::engine::Backend;
use crate::evaluation::{CostSpec, EvalRequest, Report};
use crate::examples::Split;
use crate::manifest::{write_atomic, DecisionManifest, Validation};
use crate::store::identity_files;

/// The evaluation record's file name, inside the checkpoint directory.
pub const EVALUATION_FILE: &str = "decision.evaluation.json";

/// What held-out evidence is enough to promote a checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    /// The task profile written into the validation.
    pub profile: String,
    pub min_test_examples: usize,
    pub min_accuracy_lift: Option<f64>,
    pub max_ece: Option<f64>,
    pub max_order_instability: Option<f64>,
    pub max_regret: Option<f64>,
    pub costs: Option<CostSpec>,
    pub required_families: Vec<String>,
    pub order_seed: u64,
}

impl Policy {
    pub fn parse(text: &str) -> Result<Policy, String> {
        const KEYS: [&str; 10] = [
            "schema", "profile", "min_test_examples", "min_accuracy_lift", "max_ece", "max_order_instability", "max_regret", "costs", "required_families", "order_seed",
        ];
        let v: Value = serde_json::from_str(text).map_err(|e| format!("the policy is not valid JSON: {e}"))?;
        let obj = v.as_object().ok_or("the policy must be a JSON object")?;
        if let Some(k) = obj.keys().find(|k| !KEYS.contains(&k.as_str())) {
            return Err(format!("the policy has an unknown key {k:?}"));
        }
        if obj.get("schema").and_then(Value::as_u64) != Some(1) {
            return Err("the policy's `schema` must be 1".to_string());
        }
        let profile = obj.get("profile").and_then(Value::as_str).unwrap_or_default().to_string();
        if profile.is_empty() || !profile.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-')) {
            return Err("the policy's `profile` is required and may use only letters, digits, '.', '_' and '-'".to_string());
        }
        let number = |key: &str, at_most_one: bool| -> Result<Option<f64>, String> {
            match obj.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(n) => {
                    let x = n.as_f64().filter(|x| x.is_finite() && *x >= 0.0 && (!at_most_one || *x <= 1.0));
                    x.map(Some).ok_or_else(|| format!("`{key}` must be a number from 0{}", if at_most_one { " to 1" } else { " up" }))
                }
            }
        };
        let min_test_examples = obj
            .get("min_test_examples")
            .and_then(Value::as_u64)
            .filter(|n| *n >= 1)
            .ok_or("`min_test_examples` is required and must be at least 1")? as usize;
        let min_accuracy_lift = match obj.get("min_accuracy_lift") {
            None | Some(Value::Null) => None,
            Some(n) => Some(n.as_f64().filter(|x| x.is_finite() && (-1.0..=1.0).contains(x)).ok_or("`min_accuracy_lift` must be a number from -1 to 1")?),
        };
        let max_ece = number("max_ece", true)?;
        let max_order_instability = number("max_order_instability", true)?;
        let max_regret = number("max_regret", false)?;
        let costs = match obj.get("costs") {
            None | Some(Value::Null) => None,
            Some(c) => Some(CostSpec::parse(&c.to_string())?),
        };
        if max_regret.is_some() && costs.is_none() {
            return Err("`max_regret` needs `costs`, the cost of a false exclusion and of wasted tokens".to_string());
        }
        if min_accuracy_lift.is_none() && max_ece.is_none() && max_order_instability.is_none() && max_regret.is_none() {
            return Err("the policy needs at least one of min_accuracy_lift, max_ece, max_order_instability, max_regret".to_string());
        }
        let required_families = match obj.get("required_families") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(items)) => {
                let mut names = Vec::new();
                for item in items {
                    let name = item.as_str().filter(|s| !s.is_empty()).ok_or("`required_families` must list family names")?;
                    if !names.iter().any(|n| n == name) {
                        names.push(name.to_string());
                    }
                }
                names
            }
            Some(_) => return Err("`required_families` must be an array of family names".to_string()),
        };
        let order_seed = match obj.get("order_seed") {
            None | Some(Value::Null) => 0,
            Some(n) => n.as_u64().ok_or("`order_seed` must be a non-negative integer")?,
        };
        Ok(Policy { profile, min_test_examples, min_accuracy_lift, max_ece, max_order_instability, max_regret, costs, required_families, order_seed })
    }

    pub fn read(path: &Path) -> Result<Policy, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Policy::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// How the checkpoint is to be measured: on the test split, with the
    /// policy's cost file and order seed.
    pub fn eval_request(&self) -> EvalRequest {
        EvalRequest { split: Split::Test, order_seed: self.order_seed, costs: self.costs }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "schema": 1, "profile": self.profile, "min_test_examples": self.min_test_examples,
            "min_accuracy_lift": self.min_accuracy_lift, "max_ece": self.max_ece,
            "max_order_instability": self.max_order_instability, "max_regret": self.max_regret,
            "costs": self.costs.map(|c| c.to_json()), "required_families": self.required_families, "order_seed": self.order_seed,
        })
    }

    /// Compare a fresh test-split report with every gate of the policy.
    pub fn judge(&self, report: &Report) -> Verdict {
        let mut gates = vec![Gate::at_least("min_test_examples", report.examples as f64, self.min_test_examples as f64)];
        if let Some(t) = self.min_accuracy_lift {
            gates.push(Gate::at_least("min_accuracy_lift", report.lift, t));
        }
        if let Some(t) = self.max_ece {
            gates.push(Gate::at_most("max_ece", report.ece, t));
        }
        if let Some(t) = self.max_order_instability {
            gates.push(Gate::at_most("max_order_instability", report.order_instability, t));
        }
        if let Some(t) = self.max_regret {
            match &report.regret {
                Some(r) => gates.push(Gate::at_most("max_regret", r.model, t)),
                None => gates.push(Gate::failed("max_regret", "the evaluation measured no regret")),
            }
        }
        for family in &self.required_families {
            match report.families.get(family) {
                None => gates.push(Gate::failed(&format!("required_family:{family}"), "the test split has no examples of this family")),
                Some(f) => match self.min_accuracy_lift {
                    Some(t) => gates.push(Gate::at_least(&format!("required_family:{family}:min_accuracy_lift"), f.accuracy - f.baseline_accuracy, t)),
                    None => gates.push(Gate::at_least(&format!("required_family:{family}:examples"), f.examples as f64, 1.0)),
                },
            }
        }
        Verdict { gates }
    }
}

/// One gate's outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub name: String,
    pub passed: bool,
    pub observed: Option<f64>,
    pub threshold: Option<f64>,
    /// What the gate asks and by how much it was missed, in words.
    pub message: String,
}

impl Gate {
    fn at_least(name: &str, observed: f64, threshold: f64) -> Gate {
        let passed = observed >= threshold;
        let message = if passed {
            format!("{name}: {observed:.4} >= {threshold:.4}")
        } else {
            format!("{name}: {observed:.4} is below the required {threshold:.4} by {:.4}", threshold - observed)
        };
        Gate { name: name.to_string(), passed, observed: Some(observed), threshold: Some(threshold), message }
    }

    fn at_most(name: &str, observed: f64, threshold: f64) -> Gate {
        let passed = observed <= threshold;
        let message = if passed {
            format!("{name}: {observed:.4} <= {threshold:.4}")
        } else {
            format!("{name}: {observed:.4} is above the allowed {threshold:.4} by {:.4}", observed - threshold)
        };
        Gate { name: name.to_string(), passed, observed: Some(observed), threshold: Some(threshold), message }
    }

    fn failed(name: &str, why: &str) -> Gate {
        Gate { name: name.to_string(), passed: false, observed: None, threshold: None, message: format!("{name}: {why}") }
    }
}

/// Every gate's outcome, in policy order.
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub gates: Vec<Gate>,
}

impl Verdict {
    pub fn passed(&self) -> bool {
        self.gates.iter().all(|g| g.passed)
    }

    pub fn failures(&self) -> impl Iterator<Item = &Gate> {
        self.gates.iter().filter(|g| !g.passed)
    }

    pub fn to_json(&self) -> Value {
        let gates: Vec<Value> = self.gates.iter().map(|g| json!({"gate": g.name, "passed": g.passed, "observed": g.observed, "threshold": g.threshold, "message": g.message})).collect();
        json!({"passed": self.passed(), "gates": gates})
    }
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for g in &self.gates {
            writeln!(f, "{} {}", if g.passed { "pass" } else { "FAIL" }, g.message)?;
        }
        Ok(())
    }
}

/// A checkpoint the pointer names.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// The checkpoint's directory name, beside the pointer file.
    pub dir: String,
    pub weights_sha256: String,
    pub evidence: Option<String>,
}

impl Entry {
    fn to_json(&self) -> Value {
        json!({"dir": self.dir, "weights_sha256": self.weights_sha256, "evidence": self.evidence})
    }

    fn from_json(v: &Value) -> Result<Entry, String> {
        let text = |k: &str| v.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
        Ok(Entry {
            dir: text("dir").ok_or("a pointer entry needs `dir`")?,
            weights_sha256: text("weights_sha256").ok_or("a pointer entry needs `weights_sha256`")?,
            evidence: text("evidence"),
        })
    }
}

/// The promoted checkpoint of one profile and the one before it.
#[derive(Clone, Debug, PartialEq)]
pub struct Pointer {
    pub profile: String,
    pub current: Entry,
    pub last_known_good: Option<Entry>,
}

impl Pointer {
    pub fn file_name(profile: &str) -> String {
        format!("decision.promoted.{profile}.json")
    }

    /// The pointer of `profile` in `parent`, if one was ever written.
    pub fn read(parent: &Path, profile: &str) -> Result<Option<Pointer>, String> {
        let path = parent.join(Pointer::file_name(profile));
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let entry = |k: &str| -> Result<Option<Entry>, String> {
            match v.get(k) {
                None | Some(Value::Null) => Ok(None),
                Some(e) => Entry::from_json(e).map(Some).map_err(|m| format!("{}: {m}", path.display())),
            }
        };
        let current = entry("current")?.ok_or_else(|| format!("{}: no `current`", path.display()))?;
        Ok(Some(Pointer { profile: profile.to_string(), current, last_known_good: entry("last_known_good")? }))
    }

    fn write(&self, parent: &Path) -> Result<(), String> {
        let v = json!({
            "schema": 1, "profile": self.profile,
            "current": self.current.to_json(), "last_known_good": self.last_known_good.as_ref().map(Entry::to_json),
        });
        let text = serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?;
        write_atomic(&parent.join(Pointer::file_name(&self.profile)), (text + "\n").as_bytes())
    }
}

/// An exclusive claim on one profile's pointer, so two promotions cannot
/// interleave. It is a file created exclusively and removed on drop; a process
/// that dies holding it leaves the file, which the error names.
struct ProfileLock(PathBuf);

impl ProfileLock {
    fn take(parent: &Path, profile: &str) -> Result<ProfileLock, String> {
        let path = parent.join(format!("decision.promoted.{profile}.lock"));
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map(|_| ProfileLock(path.clone()))
            .map_err(|e| format!("another promotion of this profile holds {} ({e}); if no promotion is running, delete that file", path.display()))
    }
}

impl Drop for ProfileLock {
    fn drop(&mut self) {
        std::fs::remove_file(&self.0).ok();
    }
}

/// What [`commit`] needs to record a passed promotion.
pub struct Promotion<'a> {
    /// The checkpoint directory being promoted.
    pub dir: &'a Path,
    pub policy: &'a Policy,
    /// The weights digest the report was measured on.
    pub weights_sha256: &'a str,
    /// The evaluation record to store beside the manifest.
    pub record: &'a Value,
}

fn backend_of(dir: &Path) -> Result<Backend, String> {
    Backend::sniff(dir).ok_or_else(|| format!("{} is not a decision checkpoint", dir.display()))
}

fn dir_name(dir: &Path) -> Result<String, String> {
    dir.file_name().map(|n| n.to_string_lossy().into_owned()).ok_or_else(|| format!("{} has no directory name", dir.display()))
}

fn parent_of(dir: &Path) -> Result<&Path, String> {
    dir.parent().filter(|p| !p.as_os_str().is_empty()).ok_or_else(|| format!("{} has no parent directory to hold the promotion pointer", dir.display()))
}

/// Un-validate the checkpoint at `dir`, if it is there and validated.
fn withdraw(dir: &Path) -> Result<(), String> {
    let Ok(backend) = backend_of(dir) else { return Ok(()) };
    let mut manifest = DecisionManifest::read(dir, backend)?;
    if manifest.validation.is_validated() {
        manifest.validation = Validation::Unvalidated;
        manifest.write(dir)?;
    }
    Ok(())
}

/// Record a promotion that passed its gates. Refuses, changing nothing, when
/// the weights on disk are not the ones that were measured.
pub fn commit(p: &Promotion) -> Result<Pointer, String> {
    let backend = backend_of(p.dir)?;
    let (weights_file, tokenizer_file) = identity_files(backend);
    let on_disk = brain_modelstore::fetch::file_digest(&p.dir.join(weights_file)).map_err(|e| format!("hashing {weights_file}: {e}"))?;
    if on_disk != p.weights_sha256 {
        return Err(format!("the weights changed during the evaluation: measured {} but the file is {on_disk}", p.weights_sha256));
    }
    let parent = parent_of(p.dir)?;
    let profile = &p.policy.profile;
    let _lock = ProfileLock::take(parent, profile)?;
    let mut manifest = DecisionManifest::read(p.dir, backend)?;
    let this = Entry { dir: dir_name(p.dir)?, weights_sha256: on_disk.clone(), evidence: Some(EVALUATION_FILE.to_string()) };

    let previous = Pointer::read(parent, profile)?;
    let superseded = previous.as_ref().map(|ptr| ptr.current.clone()).filter(|cur| cur.dir != this.dir);
    let last_known_good = match (&superseded, &previous) {
        (Some(cur), _) => Some(cur.clone()),
        (None, Some(ptr)) => ptr.last_known_good.clone().filter(|e| e.dir != this.dir),
        (None, None) => None,
    };

    let text = serde_json::to_string_pretty(p.record).map_err(|e| e.to_string())?;
    write_atomic(&p.dir.join(EVALUATION_FILE), (text + "\n").as_bytes())?;
    if let Some(old) = &superseded {
        withdraw(&parent.join(&old.dir))?;
    }
    manifest.weights_sha256 = Some(on_disk);
    if manifest.tokenizer_sha256.is_none() {
        manifest.tokenizer_sha256 = Some(brain_modelstore::fetch::file_digest(&p.dir.join(tokenizer_file)).map_err(|e| format!("hashing {tokenizer_file}: {e}"))?);
    }
    manifest.validation = Validation::Validated { profile: profile.clone(), evidence: this.evidence.clone() };
    manifest.write(p.dir)?;

    let pointer = Pointer { profile: profile.clone(), current: this, last_known_good };
    pointer.write(parent)?;
    Ok(pointer)
}

/// Make the last-known-good checkpoint of `dir`'s profile the validated one
/// again, and un-validate `dir`. Refuses, changing nothing, when `dir` is not
/// the profile's current checkpoint, when there is no last-known-good, or when
/// the last-known-good's weights are no longer the ones that were validated.
pub fn rollback(dir: &Path) -> Result<Pointer, String> {
    let backend = backend_of(dir)?;
    let manifest = DecisionManifest::read(dir, backend)?;
    let Validation::Validated { profile, .. } = &manifest.validation else {
        return Err(format!("{} is not validated, so there is nothing to roll back", dir.display()));
    };
    let parent = parent_of(dir)?;
    let _lock = ProfileLock::take(parent, profile)?;
    let pointer = Pointer::read(parent, profile)?.ok_or_else(|| format!("no promotion pointer for profile {profile:?} beside {}", dir.display()))?;
    if pointer.current.dir != dir_name(dir)? {
        return Err(format!("{} is not the current checkpoint of profile {profile:?} (that is {})", dir.display(), pointer.current.dir));
    }
    let good = pointer.last_known_good.clone().ok_or_else(|| format!("profile {profile:?} has no last-known-good checkpoint"))?;
    let good_dir = parent.join(&good.dir);
    let good_backend = backend_of(&good_dir)?;
    let (weights_file, _) = identity_files(good_backend);
    let on_disk = brain_modelstore::fetch::file_digest(&good_dir.join(weights_file)).map_err(|e| format!("hashing {}: {e}", good_dir.display()))?;
    if on_disk != good.weights_sha256 {
        return Err(format!("the last-known-good {} no longer has the weights that were validated: recorded {}, file {on_disk}", good.dir, good.weights_sha256));
    }
    let mut good_manifest = DecisionManifest::read(&good_dir, good_backend)?;

    withdraw(dir)?;
    good_manifest.weights_sha256 = Some(on_disk);
    good_manifest.validation = Validation::Validated { profile: profile.clone(), evidence: good.evidence.clone() };
    good_manifest.write(&good_dir)?;
    let restored = Pointer { profile: profile.clone(), current: good, last_known_good: None };
    restored.write(parent)?;
    Ok(restored)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluation::{Baseline, FamilyReport, Regret};
    use crate::fixtures::{write_decide, Scratch};
        use std::collections::BTreeMap;

    fn policy_json(extra: &str) -> String {
        format!(r#"{{"schema": 1, "profile": "routing", "min_test_examples": 40, "min_accuracy_lift": 0.1, "max_ece": 0.1, "max_order_instability": 0.05{extra}}}"#)
    }

    /// A report that passes `policy_json("")`.
    fn good_report() -> Report {
        Report {
            split: Split::Test,
            examples: 50,
            accuracy: 0.9,
            baseline: Baseline { majority_label: Some("a".into()), majority_accuracy: 0.3, chance_accuracy: 0.25 },
            baseline_from_train: true,
            lift: 0.6,
            nll: 0.3,
            brier: 0.2,
            ece: 0.05,
            ada_ece: 0.05,
            reliability: vec![],
            coverage_accuracy: vec![],
            order_seed: 0,
            order_instability: 0.0,
            regret: Some(Regret { costs: CostSpec { false_exclusion: 5.0, wasted_tokens: 1.0 }, model: 0.4, always_trust: 0.9, always_defer: 0.8, trusted_share: 0.9 }),
            families: BTreeMap::from([("alpha".to_string(), FamilyReport { examples: 30, accuracy: 0.9, baseline_accuracy: 0.3, nll: 0.3 })]),
        }
    }

    #[test]
    fn a_policy_is_parsed_and_every_malformed_one_is_refused_by_field() {
        let p = Policy::parse(&policy_json(r#", "max_regret": 1.5, "costs": {"false_exclusion": 5, "wasted_tokens": 1}, "required_families": ["alpha", "alpha", "beta"], "order_seed": 4"#)).unwrap();
        assert_eq!((p.profile.as_str(), p.min_test_examples, p.max_regret, p.order_seed), ("routing", 40, Some(1.5), 4));
        assert_eq!(p.required_families, ["alpha", "beta"]);
        assert_eq!(p.eval_request().split, Split::Test);
        for (bad, needle) in [
            ("nope".to_string(), "valid JSON"),
            ("[]".to_string(), "object"),
            (policy_json(r#", "max_ecce": 0.1"#), "max_ecce"),
            (policy_json("").replace(r#""schema": 1"#, r#""schema": 2"#), "schema"),
            (policy_json("").replace("routing", "has space"), "profile"),
            (policy_json("").replace(r#""min_test_examples": 40, "#, ""), "min_test_examples"),
            (r#"{"schema": 1, "profile": "p", "min_test_examples": 5}"#.to_string(), "at least one of"),
            (policy_json(r#", "max_regret": 1"#), "needs `costs`"),
            (policy_json("").replace(r#""max_ece": 0.1"#, r#""max_ece": 2"#), "max_ece"),
            (policy_json(r#", "required_families": [3]"#), "required_families"),
        ] {
            let e = Policy::parse(&bad).unwrap_err();
            assert!(e.contains(needle), "{bad}: {e}");
        }
    }

    #[test]
    fn a_report_that_meets_every_gate_passes() {
        let v = Policy::parse(&policy_json(r#", "required_families": ["alpha"]"#)).unwrap().judge(&good_report());
        assert!(v.passed(), "{v}");
        assert_eq!(v.gates.len(), 5, "examples, lift, ece, order, family lift");
    }

    #[test]
    fn each_failed_gate_is_named_with_how_far_it_was_missed() {
        let p = Policy::parse(&policy_json(r#", "max_regret": 0.2, "costs": {"false_exclusion": 5, "wasted_tokens": 1}, "required_families": ["alpha", "beta"]"#)).unwrap();
        let mut r = good_report();
        r.examples = 30;
        r.lift = 0.04;
        r.ece = 0.25;
        r.order_instability = 0.2;
        let v = p.judge(&r);
        assert!(!v.passed());
        let failed: Vec<&str> = v.failures().map(|g| g.name.as_str()).collect();
        assert_eq!(failed, ["min_test_examples", "min_accuracy_lift", "max_ece", "max_order_instability", "max_regret", "required_family:beta"]);
        let text = v.to_string();
        assert!(text.contains("max_ece: 0.2500 is above the allowed 0.1000 by 0.1500"), "{text}");
        assert!(text.contains("min_accuracy_lift: 0.0400 is below the required 0.1000 by 0.0600"), "{text}");
        assert!(text.contains("required_family:beta") && text.contains("no examples"), "{text}");
    }

    #[test]
    fn a_family_the_model_does_not_beat_its_baseline_on_fails_even_if_overall_accuracy_passes() {
        let p = Policy::parse(&policy_json(r#", "required_families": ["alpha"]"#)).unwrap();
        let mut r = good_report();
        r.families.insert("alpha".into(), FamilyReport { examples: 30, accuracy: 0.31, baseline_accuracy: 0.3, nll: 1.3 });
        let v = p.judge(&r);
        assert_eq!(v.failures().map(|g| g.name.as_str()).collect::<Vec<_>>(), ["required_family:alpha:min_accuracy_lift"]);
    }

    /// A store directory holding checkpoints `names`, each a complete decide model.
    fn series(names: &[&str]) -> Scratch {
        let root = Scratch::new("promotion");
        for (i, name) in names.iter().enumerate() {
            write_decide(&root.join(name), 100 + i as u64);
        }
        root
    }

    fn promote(root: &Path, name: &str) -> Result<Pointer, String> {
        let policy = Policy::parse(&policy_json("")).unwrap();
        let dir = root.join(name);
        let weights = brain_modelstore::fetch::file_digest(&dir.join("model.safetensors")).unwrap();
        commit(&Promotion { dir: &dir, policy: &policy, weights_sha256: &weights, record: &json!({"measured": name}) })
    }

    fn validated(root: &Path, name: &str) -> bool {
        DecisionManifest::read(&root.join(name), Backend::Decide).unwrap().validation.is_validated()
    }

    #[test]
    fn a_promotion_validates_the_checkpoint_on_the_weights_it_measured_and_stores_the_evidence() {
        let root = series(&["v1"]);
        assert!(!validated(&root, "v1"));
        let ptr = promote(&root, "v1").unwrap();
        let m = DecisionManifest::read(&root.join("v1"), Backend::Decide).unwrap();
        let weights = brain_modelstore::fetch::file_digest(&root.join("v1/model.safetensors")).unwrap();
        assert_eq!(m.validation, Validation::Validated { profile: "routing".into(), evidence: Some(EVALUATION_FILE.into()) });
        assert_eq!(m.weights_sha256, Some(weights.clone()));
        assert_eq!(ptr, Pointer { profile: "routing".into(), current: Entry { dir: "v1".into(), weights_sha256: weights, evidence: Some(EVALUATION_FILE.into()) }, last_known_good: None });
        assert_eq!(Pointer::read(&root, "routing").unwrap(), Some(ptr));
        let record: Value = serde_json::from_str(&std::fs::read_to_string(root.join("v1").join(EVALUATION_FILE)).unwrap()).unwrap();
        assert_eq!(record["measured"], "v1");
        let leftovers: Vec<_> = std::fs::read_dir(&*root).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).filter(|n| n.contains(".lock") || n.contains(".tmp")).collect();
        assert!(leftovers.is_empty(), "{leftovers:?}");
    }

    #[test]
    fn promoting_a_second_checkpoint_withdraws_the_first_and_keeps_it_as_last_known_good() {
        let root = series(&["v1", "v2"]);
        promote(&root, "v1").unwrap();
        let ptr = promote(&root, "v2").unwrap();
        assert!(!validated(&root, "v1") && validated(&root, "v2"), "one validated checkpoint per profile");
        assert_eq!(ptr.current.dir, "v2");
        assert_eq!(ptr.last_known_good.as_ref().map(|e| e.dir.as_str()), Some("v1"));
        // Promoting the current checkpoint again changes nothing about the last-known-good.
        let again = promote(&root, "v2").unwrap();
        assert_eq!(again.last_known_good, ptr.last_known_good);
    }

    #[test]
    fn rollback_makes_the_last_known_good_the_validated_one_again_and_only_once() {
        let root = series(&["v1", "v2"]);
        promote(&root, "v1").unwrap();
        promote(&root, "v2").unwrap();
        let ptr = rollback(&root.join("v2")).unwrap();
        assert!(validated(&root, "v1") && !validated(&root, "v2"));
        assert_eq!((ptr.current.dir.as_str(), ptr.last_known_good.is_none()), ("v1", true));
        assert!(rollback(&root.join("v2")).unwrap_err().contains("not validated"));
        assert!(rollback(&root.join("v1")).unwrap_err().contains("no last-known-good"));
    }

    #[test]
    fn rollback_refuses_a_last_known_good_whose_weights_changed_and_changes_nothing() {
        let root = series(&["v1", "v2"]);
        promote(&root, "v1").unwrap();
        promote(&root, "v2").unwrap();
        write_decide(&root.join("v1"), 999);
        let e = rollback(&root.join("v2")).unwrap_err();
        assert!(e.contains("no longer has the weights"), "{e}");
        assert!(validated(&root, "v2") && !validated(&root, "v1"));
        assert_eq!(Pointer::read(&root, "routing").unwrap().unwrap().current.dir, "v2");
    }

    #[test]
    fn a_promotion_of_weights_other_than_the_measured_ones_is_refused_and_changes_nothing() {
        let root = series(&["v1"]);
        let policy = Policy::parse(&policy_json("")).unwrap();
        let wrong = format!("sha256:{}", "0".repeat(64));
        let e = commit(&Promotion { dir: &root.join("v1"), policy: &policy, weights_sha256: &wrong, record: &json!({}) }).unwrap_err();
        assert!(e.contains("changed during the evaluation"), "{e}");
        assert!(!validated(&root, "v1"));
        assert!(!root.join("v1").join(EVALUATION_FILE).exists());
        assert!(Pointer::read(&root, "routing").unwrap().is_none());
    }

    #[test]
    fn two_promotions_of_one_profile_cannot_interleave() {
        let root = series(&["v1"]);
        let _held = ProfileLock::take(&root, "routing").unwrap();
        let e = promote(&root, "v1").unwrap_err();
        assert!(e.contains("another promotion"), "{e}");
        assert!(!validated(&root, "v1"));
    }
}
