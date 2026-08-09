// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The resolution ladder: given a [`ModelRef`], decide what has to happen
//! before it can be served. Pure with respect to the filesystem beyond
//! [`Store::local`] lookups, and pure with respect to the network beyond
//! [`Hub`] calls -- so it is fully unit-tested against [`FakeHub`], with no
//! server and no real disk layout beyond a scratch dir.

use brain_modelref::{ModelRef, Quant};

use crate::hub::{Hub, HubError};
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
pub fn plan(reference: &ModelRef, store: &Store, hub: &dyn Hub) -> Result<Plan, PlanError> {
    if store.local(reference).is_some() {
        return Ok(Plan { reference: reference.clone(), steps: vec![Step::Serve] });
    }
    if reference.is_reserved() {
        return Err(PlanError::NotFetchable(reference.clone()));
    }
    match reference.quant() {
        Some(q) => plan_quant(reference, q, store, hub),
        None => plan_base(reference, store, hub),
    }
}

fn plan_quant(reference: &ModelRef, quant: Quant, store: &Store, hub: &dyn Hub) -> Result<Plan, PlanError> {
    for (repo, file) in quant_candidates(reference, quant) {
        match hub.list_files(reference.vendor(), &repo, REVISION) {
            Ok(files) if files.iter().any(|f| f == &file) => {
                return Ok(Plan {
                    reference: reference.clone(),
                    steps: vec![Step::Download {
                        vendor: reference.vendor().to_string(),
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
    let mut base_plan = plan_base(&base, store, hub)?;
    base_plan.reference = reference.clone();
    base_plan.steps.push(Step::Quantize { base, quant });
    Ok(base_plan)
}

/// Upstream naming conventions for a pre-quantized artifact, tried in order.
/// This list is the *only* place those conventions live.
fn quant_candidates(reference: &ModelRef, quant: Quant) -> Vec<(String, String)> {
    let repo = reference.repo();
    let q = quant.as_str();
    vec![
        // The common `<repo>-GGUF` sibling repo, one file per quant.
        (format!("{repo}-GGUF"), format!("{repo}-{q}.gguf")),
        // Some repos ship a gguf alongside their own safetensors.
        (repo.to_string(), format!("{repo}-{q}.gguf")),
        // A few publish one repo per quant level.
        (format!("{repo}-{q}"), format!("{repo}-{q}.gguf")),
    ]
}

fn plan_base(reference: &ModelRef, store: &Store, hub: &dyn Hub) -> Result<Plan, PlanError> {
    if let Some(local) = store.local(reference) {
        return Ok(Plan { reference: reference.clone(), steps: local_serve_steps(&local) });
    }
    let vendor = reference.vendor();
    let repo = reference.repo();
    let listing = hub.list_files(vendor, repo, REVISION).map_err(PlanError::Hub)?;

    let recipe = crate::recipe::recipes()
        .into_iter()
        .find(|r| r.matches_listing(&listing))
        .expect("the last recipe in the registry is a catch-all and always matches");
    let artifacts = recipe.artifacts(reference, &listing, hub).map_err(|e| *e)?;

    let mut steps: Vec<Step> = artifacts.into_iter().map(|a| download_step(vendor, repo, &a.file, &a.dest_name)).collect();
    steps.push(Step::Convert { vendor: vendor.to_string(), repo: repo.to_string(), recipe: recipe.id() });
    Ok(Plan { reference: reference.clone(), steps })
}

fn local_serve_steps(_local: &LocalModel) -> Vec<Step> {
    vec![Step::Serve]
}

fn download_step(vendor: &str, repo: &str, file: &str, dest_name: &str) -> Step {
    Step::Download {
        vendor: vendor.to_string(),
        repo: repo.to_string(),
        revision: REVISION.to_string(),
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

/// The brain family (`crates/cli/src/model_dir.rs`'s `resident_for` dispatch
/// key) an HF `architecture` string maps to, or `None` if unsupported. An
/// approximation (substring match on the family name), documented as such --
/// tightening it to an exact HF class-name table is future work once more
/// families are wired. Public so the Convert-step dispatcher picks the exact
/// same family `plan()` already gated the fetch on -- one implementation of
/// "which families brain can serve today", not two.
pub fn family_of_architecture(arch: &str) -> Option<&'static str> {
    let lower = arch.to_ascii_lowercase();
    // "omni" MUST be checked before "qwen": Qwen3-Omni's HF class name is
    // `Qwen3OmniMoeForConditionalGeneration`, which contains "qwen" as a
    // substring too — a plain first-match-wins scan in the other order would
    // silently route it to the dense qwen importer, which would download the
    // full 70.5 GB checkpoint and then fail (or worse, partially import) on a
    // family it cannot represent. See docs/models/omni/status.md M3.
    ["omni", "gpt", "glm", "qwen", "lfm"].into_iter().find(|fam| lower.contains(fam))
}

pub(crate) fn is_supported_architecture(arch: &str) -> bool {
    family_of_architecture(arch).is_some()
}

/// Runs every [`Step::Download`] in `plan.steps` against `hub`, writing into
/// `store`'s `<vendor>/<repo>` directory for `plan.reference`'s base repo.
/// Returns the steps this crate could not execute (`Convert`/`Quantize`), in
/// order, for the caller to run next.
pub fn execute(store: &Store, hub: &dyn Hub, plan: &Plan, progress: &mut dyn FnMut(&str, u64, Option<u64>)) -> Result<Vec<Step>, HubError> {
    let dir = store.repo_dir(&plan.reference.base());
    let mut deferred = Vec::new();
    for step in &plan.steps {
        match step {
            Step::Serve => {}
            Step::Download { vendor, repo, revision, file, dest_name } => {
                let dest = dir.join(dest_name);
                hub.download(vendor, repo, revision, file, &dest, &mut |got, total| progress(dest_name, got, total))?;
            }
            Step::Convert { .. } | Step::Quantize { .. } => deferred.push(step.clone()),
        }
    }
    Ok(deferred)
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

    #[test]
    fn omni_architecture_does_not_fall_through_to_qwen() {
        // Qwen3-Omni's real HF class name contains "qwen" as a substring
        // ("Qwen3OmniMoeForConditionalGeneration"), so a naive first-match
        // scan checking "qwen" before "omni" would silently route it to the
        // dense qwen importer. "omni" must win.
        assert_eq!(family_of_architecture("Qwen3OmniMoeForConditionalGeneration"), Some("omni"));
        assert_eq!(family_of_architecture("qwen3_omni_moe"), Some("omni"));
        // Plain Qwen3 architectures are unaffected.
        assert_eq!(family_of_architecture("Qwen3ForCausalLM"), Some("qwen"));
        assert_eq!(family_of_architecture("Qwen3MoeForCausalLM"), Some("qwen"));
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
        assert!(matches!(err, PlanError::NotFetchable(_)));
    }

    #[test]
    fn non_reserved_ref_with_no_hub_entry_and_nothing_on_disk_errors() {
        let st = store("modelstore-plan-test-missing");
        let hub = FakeHub::new();
        let r = ModelRef::new("Qwen", "Qwen3-0.6B", None);
        let err = plan(&r, &st, &hub).unwrap_err();
        assert!(matches!(err, PlanError::Hub(HubError::NotFound(_))));
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
                download_step("Qwen", "Qwen3-0.6B", "config.json", "config.json"),
                download_step("Qwen", "Qwen3-0.6B", "tokenizer.json", "tokenizer.json"),
                download_step("Qwen", "Qwen3-0.6B", "model.safetensors", "model.safetensors"),
                Step::Convert { vendor: "Qwen".to_string(), repo: "Qwen3-0.6B".to_string(), recipe: "transformers" },
            ]
        );
    }

    #[test]
    fn base_ref_with_sharded_weights_plans_index_then_each_shard_sorted() {
        let st = store("modelstore-plan-test-base-sharded");
        let mut hub = FakeHub::new();
        hub.add_file("nvidia", "big-model", "main", "config.json", br#"{"model_type":"glm"}"#.to_vec());
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
        assert!(matches!(err, PlanError::UnsupportedArchitecture(_, arch) if arch == "MambaForCausalLM"));
    }

    #[test]
    fn missing_config_json_errors_before_any_weight_step() {
        let st = store("modelstore-plan-test-no-config");
        let mut hub = FakeHub::new();
        hub.add_file("someone", "no-config", "main", "model.safetensors", vec![0u8; 8]);

        let r = ModelRef::new("someone", "no-config", None);
        let err = plan(&r, &st, &hub).unwrap_err();
        assert!(matches!(err, PlanError::NoUpstreamArtifact(_, _)));
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
}
