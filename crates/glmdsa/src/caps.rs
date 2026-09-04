// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM-5.2 capabilities behind the generalized [`capability`] interface - what
//! makes `brain caps` list this model and `brain glmdsa generate ...` run it,
//! with no GLM-specific plumbing in the CLI.
//!
//! Swedish Embedded AB implements discoverable, schedulable on-device model
//! serving for teams who need one uniform interface across a fleet of very
//! different architectures. If your team needs expertise in capability-driven
//! model serving then you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! # Why this module exists
//!
//! GLM was reachable over D-Bus/HTTP before this: `cli::resident_llm::
//! GlmResident` implements [`capability::Manifest`] through its
//! `ResidentModel`, which the serving contract accepts. What it did NOT have is
//! a **weight-free** manifest, and that is a real gap rather than a cosmetic
//! one. `GlmResident::from_env` returns `None` unless `BRAIN_GLMDSA_WEIGHTS` is
//! set, so with no weights on the box `brain caps` did not list GLM **at all** -
//! while every other model in the catalog advertises itself with no checkpoint
//! present and takes its weights as a request parameter. Discovery that depends
//! on deployment state is discovery a client cannot rely on.
//!
//! So [`manifest`] is static and safe to build with nothing on disk, and
//! [`manifest_resident`] is the same manifest with `weights` dropped - the
//! resident adapter gets its checkpoint from service-side configuration, so
//! advertising a `weights` parameter a caller cannot usefully set would be a
//! lie. `GlmResident::manifest` calls [`manifest_resident`] rather than
//! building its own [`capability::ActionSpec`], so the two surfaces cannot
//! drift apart: they are one definition.
//!
//! The dropping itself is no longer GLM's own code: `weights` declares
//! [`capability::ParamSpec::host_env`] (`BRAIN_GLMDSA_WEIGHTS`), which both
//! projects it out of every off-machine surface
//! ([`capability::Manifest::for_serving`]) and fills it from that variable at
//! validate time. Every model with a checkpoint path says it the same way.
//!
//! # Scope
//!
//! One action, `generate`. GLM in this repo is **char-level** (the checkpoint
//! carries its own `itos` vocabulary), so there is no tokenizer parameter and
//! no chat template - unlike `qwen3::caps`, whose `generate` is a chat surface.
//! Decoding is the KV-cached [`crate::sample::generate_kv`] path, the same one
//! `brain glmdsa infer` uses, so the served path and the CLI path cannot
//! disagree about how this model samples.

use std::sync::{Arc, Mutex};

use capability::{
    Action, ActionResult, ActionSpec, Blob, BlobSpec, Invocation, Manifest, Media, Outcome, ParamSpec, ParamType,
    Progress, Provider,
};
use data::rng::Rng;
use data::tokenizer::{CharTokenizer, Tokenizer};
use serde_json::json;

use crate::config::GlmConfig;
use crate::model::Glm;

/// The model id used by `brain caps`, `brain glmdsa <verb>` and the event API.
///
/// `brain/glm`, not `brain/glmdsa`: this is the MODEL ref, and it is already
/// the canonical id everywhere else - `modelref::alias::ROWS` maps the legacy
/// bare `glm` onto it, `cli::perf_cli` names it, and the checkpoint's own
/// `ModelCard` carries it. `glmdsa` is the *architecture* id (`brain_arch`,
/// from llama.cpp's `LLM_ARCH_GLM_DSA`), a different namespace. Minting a
/// second model ref here would give one model two names, which is precisely
/// what the naming invariant exists to prevent.
pub const MODEL: &str = "brain/glm";

/// The full, static capability manifest - safe to build with no weights loaded.
pub fn manifest() -> Manifest {
    let generate = ActionSpec::new("generate", "generate text continuing a prompt (GLM MLA + MoE decoder)")
        .param(ParamSpec::new("weights", ParamType::Str, "path to a brain-format GLM checkpoint (.safetensors)").required().host_env("BRAIN_GLMDSA_WEIGHTS"))
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate").default(json!(128)))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").default(json!(0.8)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (40 = standard; 1 = greedy; 0 or negative = disabled)").default(json!(40)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed").default(json!(0)))
        .output(BlobSpec::new("text", Media::Text, "the generated text"));
    Manifest::new(MODEL, "GLM-5.2 decoder (MLA + sigmoid noaux_tc MoE + DSA indexer): char-level text generation.", vec![generate])
}

/// The manifest for the RESIDENT/scheduled service (D-Bus, executor, HTTP): the
/// checkpoint is service-side configuration (`BRAIN_GLMDSA_WEIGHTS`), so the
/// action carries only request parameters.
///
/// One line, because "drop the host's own paths" is not a GLM fact: `weights`
/// carries [`capability::ParamSpec::host_env`], and
/// [`capability::Manifest::for_serving`] projects every such param out for
/// every model at once. This function stays as GLM's own name for that
/// projection.
pub fn manifest_resident() -> Manifest {
    manifest().for_serving()
}

/// Sampling parameters, read once so the served and CLI paths agree on
/// defaults. Kept next to the [`ParamSpec`] defaults above deliberately: a
/// default that lives in only one of the two places is how they drift.
pub fn sampling(inv: &Invocation) -> (usize, f32, usize, u64) {
    (
        inv.get_i64("max_new").unwrap_or(128).max(0) as usize,
        inv.get_f64("temp").unwrap_or(0.8) as f32,
        inv.get_i64("top_k").unwrap_or(40).max(0) as usize,
        inv.get_i64("seed").unwrap_or(0).max(0) as u64,
    )
}

/// The resident (hot) model + its char vocabulary, and the checkpoint path that
/// fixes them.
struct Hot {
    weights: String,
    tok: CharTokenizer,
    model: Glm,
}

/// The executable GLM model behind the manifest. Construction is free - the
/// checkpoint loads lazily on the first run and stays resident, so `brain caps`
/// costs nothing and a repeated `brain glmdsa generate` pays the load once.
#[derive(Default)]
pub struct GlmProvider {
    hot: Arc<Mutex<Option<Hot>>>,
}

impl GlmProvider {
    pub fn new() -> GlmProvider {
        GlmProvider::default()
    }
}

impl Provider for GlmProvider {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "generate" => Some(Arc::new(GenerateAction { hot: self.hot.clone() }) as Arc<dyn Action>),
            _ => None,
        }
    }
}

struct GenerateAction {
    hot: Arc<Mutex<Option<Hot>>>,
}

impl Action for GenerateAction {
    fn spec(&self) -> ActionSpec {
        manifest().actions.into_iter().next().expect("the manifest declares exactly one action")
    }

    fn run(&self, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let weights = inv.get_str("weights").unwrap_or_default();
        if weights.is_empty() {
            return Err("glmdsa: 'weights' is required (path to a brain-format GLM checkpoint)".into());
        }
        let (max_new, temp, top_k, seed) = sampling(inv);

        let mut guard = self.hot.lock().map_err(|_| "glmdsa: model lock poisoned")?;
        if guard.as_ref().map(|h| h.weights != weights).unwrap_or(true) {
            // A char-level checkpoint that carries no embedded vocabulary
            // cannot be decoded at all, and there is no request parameter that
            // could supply one - so this is a named error, never a fallback to
            // a guessed alphabet.
            let itos = Glm::load_itos(&weights)
                .ok_or_else(|| format!("glmdsa: checkpoint has no embedded char vocab: {weights}"))?;
            let block = GlmConfig::from_json(&checkpoint::load(&weights).header["config"]).block_size;
            let model = Glm::load_inference(&weights, 1, block);
            *guard = Some(Hot { weights: weights.clone(), tok: CharTokenizer::from_itos(itos), model });
        }
        let hot = guard.as_ref().expect("just populated");

        // An empty prompt seeds with a newline, matching `brain glmdsa infer`.
        let prompt = inv.get_str("prompt").unwrap_or_default();
        let prompt_text = if prompt.is_empty() { "\n".to_string() } else { prompt };
        let ids = hot.tok.encode(&prompt_text);

        progress(Progress::step(0, max_new as u32, "generating"));
        let mut rng = Rng::new(seed);
        let gen = crate::sample::generate_kv(&hot.model, &ids, max_new, temp, top_k, None, &mut rng);
        progress(Progress::step(max_new as u32, max_new as u32, "done"));

        let text = hot.tok.decode(&gen);
        Ok(Outcome::new().blob("text", Blob::new(Media::Text, text.into_bytes())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_manifest_is_weight_free_and_names_one_action() {
        // The whole point of this module: buildable with nothing on disk, so
        // `brain caps` lists GLM on a box that has no checkpoint.
        let m = manifest();
        assert_eq!(m.model, MODEL);
        assert_eq!(m.actions.len(), 1);
        assert_eq!(m.actions[0].name, "generate");
    }

    /// The resident surface advertises no `weights`: the executor supplies the
    /// checkpoint from `BRAIN_GLMDSA_WEIGHTS`, so a caller cannot set it and
    /// must not be told otherwise. Every OTHER parameter has to survive, or the
    /// resident silently loses a knob the direct path has.
    #[test]
    fn the_resident_manifest_drops_weights_and_keeps_the_rest() {
        let direct = &manifest().actions[0];
        let resident = &manifest_resident().actions[0];
        assert!(direct.params.iter().any(|p| p.name == "weights"));
        assert!(!resident.params.iter().any(|p| p.name == "weights"));

        let kept: Vec<&str> = resident.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(kept, vec!["prompt", "max_new", "temp", "top_k", "seed"]);
        assert_eq!(resident.params.len() + 1, direct.params.len());
    }

    /// `sampling` reads the same defaults the `ParamSpec`s declare. They live in
    /// two places (a spec default is JSON, a read is Rust), so pin them equal -
    /// a drifting default changes generation silently, with nothing failing.
    ///
    /// Each is compared in the type the code actually reads it as, not as raw
    /// JSON: `temp` is declared `0.8` (an f64) and read as f32, so a JSON-value
    /// comparison fails on the widening round-trip (0.800000011920929) while
    /// the values agree perfectly at the precision the sampler uses.
    #[test]
    fn the_read_defaults_match_the_declared_defaults() {
        let (max_new, temp, top_k, seed) = sampling(&Invocation::default());
        let spec = &manifest().actions[0];
        let declared = |name: &str| {
            spec.params.iter().find(|p| p.name == name).and_then(|p| p.default.clone()).unwrap_or_else(|| panic!("'{name}' declares no default"))
        };

        assert_eq!(declared("max_new").as_i64(), Some(max_new as i64));
        assert_eq!(declared("temp").as_f64().map(|v| v as f32), Some(temp));
        assert_eq!(declared("top_k").as_i64(), Some(top_k as i64));
        assert_eq!(declared("seed").as_i64(), Some(seed as i64));
    }

    #[test]
    fn an_unknown_action_is_none_and_generate_resolves() {
        let p = GlmProvider::new();
        assert!(p.action("generate").is_some());
        assert!(p.action("summarise").is_none());
    }

    /// Construction must not touch the filesystem - `brain caps` builds every
    /// provider in the catalog, so a provider that loads weights eagerly makes
    /// listing capabilities cost a checkpoint read per model.
    #[test]
    fn a_missing_checkpoint_is_a_named_error_not_a_panic() {
        let p = GlmProvider::new();
        let action = p.action("generate").expect("generate exists");
        let err = action.run(&Invocation::default(), &mut |_| {}).unwrap_err();
        assert!(err.contains("weights"), "{err}");
    }
}
