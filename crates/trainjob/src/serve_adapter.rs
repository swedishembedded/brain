// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for evaluating the policies an online training run publishes on serving
// capacity of their own, for its clients. If your team needs expertise in serving and comparing fine-tuned model
// versions, you can procure our services by sending an email to info@swedishembedded.com.

//! The `serve_adapter` and `withdraw_adapter` actions: serve a stored adapter, named by its artifact digest, under an
//! id the caller chooses, until the caller withdraws it.
//!
//! A version a training run published lives on the server that trained it. Any other server that serves the same base
//! and reaches the same artifact store can serve that version too: `serve_adapter` fetches the stored file, verifies it
//! against the bases this server serves, offers it as a model of its own under `id` (an adapter reference), with the
//! policy version the caller names, and returns; the adapter stays served, beside any others, until
//! `withdraw_adapter` takes it off (requests already accepted finish). The adapter bank's capacity bounds how many are
//! served at once; an offer it cannot hold is refused.

use capability::{ActionResult, ActionSpec, Invocation, Outcome, ParamSpec, ParamType, Progress};
use serde::Deserialize;
use serde_json::json;

use crate::job::JobEnv;
use crate::store::Digest;

/// The action's name.
pub const ACTION: &str = "serve_adapter";

/// The action that withdraws what [`ACTION`] serves.
pub const WITHDRAW_ACTION: &str = "withdraw_adapter";

/// Where the served files are kept, under the work root, while they are served.
const SERVED_DIR: &str = "served-adapters";

/// The largest adapter file the job fetches from the store.
const MAX_ADAPTER_BYTES: u64 = 64 << 30;

/// What `serve_adapter` is.
pub fn spec() -> ActionSpec {
    ActionSpec::new(ACTION, "serve a stored LoRA adapter, named by its artifact digest, under an id the caller chooses (an adapter reference), until withdraw_adapter takes it off; the adapter rides on the served base its card names")
        .training()
        .cancellable()
        .param(ParamSpec::new("adapter_digest", ParamType::Str, "the adapter file in this server's artifact store (BLAKE3 hex), as a training run's published version names it (`artifact_digest`)").required())
        .param(ParamSpec::new("id", ParamType::Str, "the model id to serve it under: an adapter reference `<vendor>/<repo>:<owner>:<name>:<tag>` no model or adapter of this server uses").required())
        .param(ParamSpec::new("version", ParamType::Int, "the policy version the run published it as, which replies name (`brain.policy_version`); the stored file does not carry it"))
}

/// The action's parameters.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Params {
    pub adapter_digest: String,
    pub id: String,
    #[serde(default)]
    pub version: Option<u64>,
}

impl Params {
    /// Reads the invocation's parameters and the digest.
    ///
    /// # Errors
    /// A parameter is missing, unknown or not a digest.
    pub fn from_invocation(inv: &Invocation) -> Result<(Params, Digest), String> {
        let params: Params = serde_json::from_value(inv.params.clone()).map_err(|e| format!("{ACTION}: {e}"))?;
        let digest = Digest::parse(&params.adapter_digest).map_err(|e| format!("{ACTION}: adapter_digest: {e}"))?;
        if params.id.trim().is_empty() {
            return Err(format!("{ACTION}: id is empty"));
        }
        Ok((params, digest))
    }
}

/// What `withdraw_adapter` is.
pub fn withdraw_spec() -> ActionSpec {
    ActionSpec::new(WITHDRAW_ACTION, "withdraw an adapter serve_adapter serves: it leaves the model listing at once, requests already accepted finish")
        .training()
        .cancellable()
        .param(ParamSpec::new("id", ParamType::Str, "the id serve_adapter served it under").required())
}

/// The file `id` is served from while it is served.
fn served_file(env: &JobEnv, id: &str) -> std::path::PathBuf {
    env.work_root.join(SERVED_DIR).join(format!("{}.safetensors", blake3::hash(id.as_bytes()).to_hex()))
}

/// Offers the adapter and returns what it is served as.
pub fn run(env: &JobEnv, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    inv.cancel.check()?;
    let (params, digest) = Params::from_invocation(inv)?;
    let store = env.store.as_ref().ok_or_else(|| format!("{ACTION}: this server has no artifact store (--artifact-store)"))?;
    let bank = env.bank.get().ok_or_else(|| format!("{ACTION}: this server serves no base adapters can ride on (--training with a served base)"))?;
    let file = served_file(env, &params.id);
    std::fs::create_dir_all(file.parent().expect("a file under a directory")).map_err(|e| format!("{ACTION}: {e}"))?;
    store.fetch(&digest, &file, MAX_ADAPTER_BYTES).map_err(|e| format!("{ACTION}: {}", e.client_message()))?;
    let identity = bank.serve_as(&params.id, &file, params.version).inspect_err(|_| {
        let _ = std::fs::remove_file(&file);
    })?;
    eprintln!("brain train: serving adapter {} as {} (version {})", digest.hex(), params.id, identity.version);
    progress(Progress::event(1, 1, json!({"type": "serving", "id": params.id, "version": identity.version})));
    Ok(Outcome::new().set("id", json!(params.id)).set("adapter_digest", json!(digest.hex())).set("version", json!(identity.version)))
}

/// Withdraws what `serve_adapter` serves under `id`.
pub fn withdraw(env: &JobEnv, inv: &Invocation) -> ActionResult {
    inv.cancel.check()?;
    let id = inv.params.get("id").and_then(serde_json::Value::as_str).filter(|id| !id.trim().is_empty()).ok_or_else(|| format!("{WITHDRAW_ACTION}: id is required"))?;
    let bank = env.bank.get().ok_or_else(|| format!("{WITHDRAW_ACTION}: this server serves no stored adapters"))?;
    bank.withdraw(id);
    let _ = std::fs::remove_file(served_file(env, id));
    eprintln!("brain train: withdrew {id}");
    Ok(Outcome::new().set("id", json!(id)))
}
