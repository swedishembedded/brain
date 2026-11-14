// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training services for risk-over-time models on irregular records, run beside
// the models being served and published by content, for its clients. If your team needs expertise in training
// and serving timeline (survival) models as a service, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The `timeline_train` action: a timeline model (or an ensemble) trained on `timeline-v1` subjects, published like any
//! other checkpoint of this service.
//!
//! The training itself is `horizon::lifecycle::train`, the one implementation the local action shares: this module
//! stages the two datasets (blobs of the call, or digests of the artifact store), points it at a private output
//! directory, writes the model-store manifest of family `horizon` beside what it wrote, and publishes the directory as
//! [`crate::consolidate`] publishes a full fine-tune: its files by digest and a manifest of them (`checkpoint_digest`),
//! optionally installed as `local/<install_as>`, where the model-dir scan serves it as a model of its own with
//! `predict`, `eval` and `calibrate`.

use capability::{ActionResult, ActionSpec, Blob, Invocation, Media, ParamSpec, ParamType, Progress};
use serde_json::json;

use crate::consolidate::{check_install_name, checkpoint_outcome};
use crate::job::{bring, JobEnv, Workdir};
use crate::request::Source;
use crate::store::Digest;

/// The action's name.
pub const ACTION: &str = "timeline_train";
/// What a run is budgeted for on the device: the models are small, so a flat allowance bounds the batches and the
/// optimiser state.
pub const DEVICE_BUDGET: u64 = 1 << 30;
/// The id a checkpoint carries in its manifest when it is not installed under a name.
const DEFAULT_ID: &str = "local/timeline";

/// What `timeline_train` is: the local `train` action's contract with the host's `out` and `force` taken away, the
/// datasets by blob or digest, and the model-store name to install under.
pub fn spec() -> ActionSpec {
    let mut spec = horizon::lifecycle::train_spec();
    spec.name = ACTION.to_string();
    spec.summary = "train a timeline model (or an ensemble) on timeline-v1 subjects, early-stopping on held-out subjects: the model leaves by manifest digest in the artifact store (and optionally into the model store, where it is served as a model of its own)".to_string();
    spec.params.retain(|p| p.name != "out" && p.name != "force");
    for input in &mut spec.inputs {
        input.required = false;
    }
    spec.param(ParamSpec::new("dataset_digest", ParamType::Str, "the training subjects, as a blake3 digest of the server's artifact store (or attach them as the input blob `dataset`; exactly one of the two)"))
        .param(ParamSpec::new("held_out_digest", ParamType::Str, "the early-stopping subjects, as a digest (or the input blob `held_out`; exactly one of the two)"))
        .param(ParamSpec::new("install_as", ParamType::Str, "also install the model in the server's model store as `local/<this name>` (letters, digits, `.`, `_`, `-`; it must not exist yet), where it is served with predict, eval and calibrate"))
}

/// Where the two datasets of a call come from.
#[derive(Debug)]
pub struct Sources {
    pub dataset: Source,
    pub held_out: Source,
    pub install_as: Option<String>,
}

fn source(inv: &Invocation, blob: &str, digest_param: &str) -> Result<Source, String> {
    match (inv.get_str(digest_param), inv.blobs.contains_key(blob)) {
        (Some(digest), false) => Digest::parse(&digest).map(Source::Digest).map_err(|e| format!("timeline_train: {digest_param}: {e}")),
        (None, true) => Ok(Source::Blob(blob.to_string())),
        _ => Err(format!("timeline_train: name the {blob} exactly once: as the input blob `{blob}` or as `{digest_param}`")),
    }
}

impl Sources {
    /// Reads the validated invocation `inv`: each dataset is named exactly once, and no other blob is attached.
    pub fn from_invocation(inv: &Invocation) -> Result<Sources, String> {
        let sources = Sources { dataset: source(inv, "dataset", "dataset_digest")?, held_out: source(inv, "held_out", "held_out_digest")?, install_as: inv.get_str("install_as") };
        if let Some(extra) = inv.blobs.keys().find(|k| !["dataset", "held_out"].contains(&k.as_str())) {
            return Err(format!("timeline_train: the input blob \"{extra}\" is not an input of this action"));
        }
        if let Some(name) = &sources.install_as {
            check_install_name(name)?;
        }
        Ok(sources)
    }

    /// Every digest the call names.
    pub fn digests(&self) -> Vec<Digest> {
        [&self.dataset, &self.held_out].into_iter().filter_map(|s| if let Source::Digest(d) = s { Some(*d) } else { None }).collect()
    }
}

/// Runs the `timeline_train` job `inv` on the device this thread is scoped to.
pub fn run(env: &JobEnv, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let sources = Sources::from_invocation(inv)?;
    let work = Workdir::create(env)?;
    let mut call = Invocation::new();
    call.cancel = inv.cancel.clone();
    call.params = inv.params.clone();
    if let Some(params) = call.params.as_object_mut() {
        for host_side in ["dataset_digest", "held_out_digest", "install_as"] {
            params.remove(host_side);
        }
    }
    for (name, source) in [("dataset", &sources.dataset), ("held_out", &sources.held_out)] {
        let staged = work.path().join(name);
        bring(source, &inv.blobs, env.store.as_ref(), work.staging(), &staged)?;
        let bytes = std::fs::read(&staged).map_err(|e| format!("timeline_train: reading {name}: {e}"))?;
        std::fs::remove_file(&staged).ok();
        call.blobs.insert(name.to_string(), Blob::new(Media::Text, bytes));
    }
    if inv.cancel.is_cancelled() {
        return Err("cancelled".into());
    }
    let out = work.path().join("model");
    call = call.set("out", json!(out.to_string_lossy()));
    let trained = horizon::lifecycle::train(&call, progress)?;
    // The manifest the model-dir scan serves the directory by: the id it will have in the model store.
    let id = sources.install_as.as_ref().map_or_else(|| DEFAULT_ID.to_string(), |name| format!("local/{name}"));
    let manifest = json!({"id": id, "family": "horizon", "roles": {"weights": "."}});
    std::fs::write(out.join(brain_modelstore::MANIFEST_FILE), manifest.to_string()).map_err(|e| format!("timeline_train: writing the manifest: {e}"))?;
    let mut report = trained.outputs.clone();
    report["status"] = json!("completed");
    report["objective"] = json!("timeline");
    work.ensure_within_quota()?;
    checkpoint_outcome(env, &out, sources.install_as.as_deref(), report)
}
