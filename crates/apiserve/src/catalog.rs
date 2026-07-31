// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deriving an API capability set from a model's `capability::Manifest`, and the
//! per-provider exposure filter.
//!
//! `api_caps` reads the *shape* of a model's actions — nothing is hard-coded per
//! model — so a new model shows up in the right `/models` list automatically:
//! - **Chat**: a streaming action taking a `prompt`/`messages`/`text` param and
//!   emitting a `Text` output.
//! - **Embeddings**: an `embed` action, or an output blob that is embedding bytes.
//! - **ImageGen**: any action emitting an `Image` output.
//!
//! OpenAI/OpenRouter expose Chat ∪ Embeddings ∪ ImageGen; Anthropic exposes Chat.

use capability::{Manifest, Media};
use residency::Executor;

use crate::surface::Provider;

/// The set of API capabilities a model advertises.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CapSet {
    pub chat: bool,
    pub embeddings: bool,
    pub image: bool,
}

impl CapSet {
    /// Does this model advertise any exposable capability?
    pub fn any(&self) -> bool {
        self.chat || self.embeddings || self.image
    }
}

/// Derive the [`CapSet`] from a manifest's action shapes.
pub fn api_caps(m: &Manifest) -> CapSet {
    let chat = m.actions.iter().any(|a| {
        a.streaming
            && a.params.iter().any(|p| matches!(p.name.as_str(), "prompt" | "messages" | "text"))
            && a.outputs.iter().any(|o| o.media == Media::Text)
    });
    let embeddings = m.actions.iter().any(|a| {
        a.name == "embed" || a.outputs.iter().any(|o| o.media == Media::Bytes && o.name.contains("embed"))
    });
    let image = m.actions.iter().any(|a| a.outputs.iter().any(|o| o.media == Media::Image));
    CapSet { chat, embeddings, image }
}

/// Does `provider` expose a model with capabilities `caps`?
pub fn exposes(provider: Provider, caps: CapSet) -> bool {
    match provider {
        // The Anthropic Messages API is chat-only.
        Provider::Anthropic => caps.chat,
        // OpenAI/OpenRouter carry chat, embeddings, and image generation.
        Provider::OpenAI | Provider::OpenRouter => caps.any(),
    }
}

/// The models `provider` exposes, as `(model_name, caps)`, sorted by name — the
/// input to every `/models` response.
pub fn exposed(exec: &Executor, provider: Provider) -> Vec<(String, CapSet)> {
    let mut out: Vec<(String, CapSet)> = exec
        .manifests()
        .iter()
        .map(|m| (m.model.clone(), api_caps(m)))
        .filter(|(_, caps)| exposes(provider, *caps))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Is `model` exposed by `provider`? (drives `GET /models/{id}` 404s)
pub fn is_exposed(exec: &Executor, provider: Provider, model: &str) -> bool {
    exec.manifests().iter().any(|m| m.model == model && exposes(provider, api_caps(m)))
}
