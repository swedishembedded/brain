// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What the served actions of `brain/qwen3` share: the checkpoint and
//! tokenizer the model store resolved for them, the base model's identity on
//! an adapter's card, and the adapters a store files under it.
//!
//! Swedish Embedded AB implements model-serving back ends whose every action
//! resolves its weights the same way for its clients. If your team needs
//! expertise in model stores and adapter registries then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Invocation, Progress};

/// The roles the model store resolved for this provider; a request that names
/// its own `weights` or `tokenizer` overrides them.
#[derive(Clone, Default)]
pub(crate) struct Resolved {
    pub weights: Option<String>,
    pub tokenizer: Option<String>,
}

impl Resolved {
    /// The checkpoint a request names, else the resolved one; it must exist.
    pub fn weights(&self, inv: &Invocation, action: &str) -> Result<String, String> {
        let weights = inv
            .get_str("weights")
            .filter(|s| !s.is_empty())
            .or_else(|| self.weights.clone())
            .ok_or_else(|| format!("qwen {action}: no checkpoint (pass 'weights', or pull one into the models directory)"))?;
        if !Path::new(&weights).exists() {
            return Err(format!("qwen {action}: weights not found at '{weights}'"));
        }
        Ok(weights)
    }

    /// The tokenizer a request names, else the resolved one, else the
    /// `tokenizer.json` beside `weights` (inside it, for a checkpoint directory).
    pub fn tokenizer(&self, inv: &Invocation, weights: &str) -> PathBuf {
        if let Some(path) = inv.get_str("tokenizer").filter(|p| !p.is_empty()).or_else(|| self.tokenizer.clone()) {
            return PathBuf::from(path);
        }
        let w = Path::new(weights);
        let dir = if w.is_dir() { w } else { w.parent().unwrap_or(Path::new(".")) };
        dir.join("tokenizer.json")
    }
}

/// An action that holds no resident state: its body is a function of the
/// store-resolved roles and the invocation.
pub(crate) struct Stateless {
    pub spec: ActionSpec,
    pub resolved: Resolved,
    pub run: fn(&Resolved, &Invocation, &mut dyn FnMut(Progress)) -> ActionResult,
}

impl Action for Stateless {
    fn spec(&self) -> ActionSpec {
        self.spec.clone()
    }
    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        inv.cancel.check()?;
        (self.run)(&self.resolved, inv, progress)
    }
}

/// `spec`'s action over `resolved`, as a provider hands it out.
pub(crate) fn stateless(spec: ActionSpec, resolved: &Resolved, run: fn(&Resolved, &Invocation, &mut dyn FnMut(Progress)) -> ActionResult) -> Arc<dyn Action> {
    Arc::new(Stateless { spec, resolved: resolved.clone(), run })
}

/// A required host path param.
pub(crate) fn path_param(inv: &Invocation, name: &str, action: &str) -> Result<PathBuf, String> {
    inv.get_str(name).filter(|p| !p.is_empty()).map(PathBuf::from).ok_or_else(|| format!("qwen {action}: missing required param '{name}'"))
}

/// A non-negative whole-number param, `default` when the caller names none.
pub(crate) fn count(inv: &Invocation, name: &str, default: u32) -> u32 {
    inv.get_i64(name).map_or(default, |v| v.clamp(0, i64::from(u32::MAX)) as u32)
}

/// Run `f`, turning a panic in the scoring and conversion code (which asserts
/// on a malformed checkpoint or dataset) into this request's error.
pub(crate) fn guarded<T>(what: &str, f: impl FnOnce() -> T) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).map_err(|e| {
        let msg = e.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| e.downcast_ref::<String>().cloned()).unwrap_or_else(|| "unknown failure".to_string());
        format!("qwen {what}: {msg}")
    })
}

/// The identity a base checkpoint goes by on an adapter's card, as a model
/// reference (`vendor/repo`): the store directory it lives in, else its own
/// card's id when that is a plain reference, else `local/<file stem>`.
pub(crate) fn base_id(weights: &str) -> String {
    let path = Path::new(weights);
    let dir = if path.is_dir() { Some(path) } else { path.parent() };
    let in_store = brain_modelstore::default_root().zip(dir).and_then(|(root, dir)| {
        let parts: Vec<_> = dir.strip_prefix(&root).ok()?.components().collect();
        (parts.len() == 2).then(|| brain_modelstore::repo_id(dir))
    });
    if let Some(id) = in_store {
        return id;
    }
    let carded = checkpoint::st::read_card(weights).ok().flatten().map(|c| c.id).filter(|id| brain_modelref::ModelRef::parse(id).is_ok_and(|r| r.adapter().is_none()));
    carded.unwrap_or_else(|| format!("local/{}", path.file_stem().and_then(|s| s.to_str()).unwrap_or("base")))
}

/// `OWNER/NAME[:TAG]` as `(owner, name, tag)`, the tag defaulting to `latest`
/// (mutable, Docker-style: training under the same name again continues or
/// replaces that tag).
pub(crate) fn parse_adapter_spec(spec: &str) -> Result<(String, String, String), String> {
    let (owner_name, tag) = match spec.split_once(':') {
        Some((a, b)) => (a, b.to_string()),
        None => (spec, "latest".to_string()),
    };
    let (owner, name) = owner_name.split_once('/').ok_or_else(|| "expected \"OWNER/NAME[:TAG]\"".to_string())?;
    if owner.is_empty() || name.is_empty() || tag.is_empty() {
        return Err("OWNER, NAME, and TAG must all be non-empty".to_string());
    }
    Ok((owner.to_string(), name.to_string(), tag))
}

/// A named adapter of the base `weights` as the model store files it.
pub struct StoredAdapter {
    /// `<base>:<owner>:<name>:<tag>`, the id on the adapter's card.
    pub reference: String,
    /// Where the store keeps (or will keep) the adapter's weights.
    pub path: PathBuf,
    /// The base model's id on that card.
    pub base: String,
}

impl StoredAdapter {
    pub fn locate(weights: &str, spec: &str, action: &str) -> Result<StoredAdapter, String> {
        let (owner, name, tag) = parse_adapter_spec(spec).map_err(|e| format!("qwen {action}: adapter {spec:?}: {e}"))?;
        let base = base_id(weights);
        let reference = format!("{base}:{owner}:{name}:{tag}");
        let parsed = brain_modelref::ModelRef::parse(&reference).map_err(|e| format!("qwen {action}: {reference}: {e}"))?;
        let root = brain_modelstore::default_root().ok_or_else(|| format!("qwen {action}: no models directory resolved (name one with --brain-data-dir)"))?;
        let path = brain_modelstore::Store::new(root).adapter_weights_path(&parsed).ok_or_else(|| format!("qwen {action}: {reference} is not an adapter reference"))?;
        Ok(StoredAdapter { reference, path, base })
    }
}
