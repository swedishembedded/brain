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
use serde_json::Value;

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

/// Resolve a chat request's `model` string: is there a manifest whose `model == id`
/// AND that advertises the chat capability? Chat dispatch (the `generate` action)
/// gates on this — an unknown or non-chat model is a `model_not_found`.
pub fn resolve_chat(exec: &Executor, model: &str) -> bool {
    exec.manifests().iter().any(|m| m.model == model && api_caps(m).chat)
}

/// Resolve an embeddings request's `model` string: is there a manifest whose
/// `model == id` AND that advertises the embeddings capability? The `/embeddings`
/// dispatch (the `embed` action) gates on this — an unknown or non-embeddings model
/// is a `model_not_found`.
pub fn resolve_embed(exec: &Executor, model: &str) -> bool {
    exec.manifests().iter().any(|m| m.model == model && api_caps(m).embeddings)
}

/// Resolve an image request's `model` string to the **action name** to invoke: is
/// there a manifest whose `model == id` that advertises the image capability, and if
/// so, which action realises a text-to-image (the one to dispatch for
/// `/images/generations`)? `None` on an unknown or non-image model (a
/// `model_not_found`). The action is picked by shape — nothing is hard-coded per
/// model — so any model exposing a text→image action serves here (z-image/flux2 name
/// it `text2image`).
pub fn resolve_image(exec: &Executor, model: &str) -> Option<String> {
    exec.manifests().iter().find(|m| m.model == model && api_caps(m).image).and_then(|m| text2image_action(&m))
}

/// Strip a leading `"<segment>/"` provider namespace from an OpenRouter-style model
/// id: `anything/qwen3-4b` -> `Some("qwen3-4b")`, `openai/gpt-4o` -> `Some("gpt-4o")`.
/// Only the FIRST segment is removed (`a/b/c` -> `b/c`). Returns `None` when there is
/// no `'/'` or the remainder would be empty (so `qwen3-4b` and `foo/` both yield None).
pub fn strip_provider_prefix(model: &str) -> Option<&str> {
    model.split_once('/').map(|(_, rest)| rest).filter(|s| !s.is_empty())
}

/// The ordered list of candidate local model ids to try for `provider` + request
/// `body`. On OpenAI/Anthropic this is exactly the primary `model` (exact match only).
/// On OpenRouter it is, in order:
/// - the primary `model`, then its prefix-stripped form;
/// - then, if a `models` fallback array is present, each entry followed by its
///   prefix-stripped form.
/// Exact match always precedes the stripped form of the same id, and the primary
/// `model` (both forms) is always tried before any `models` fallback entry.
fn candidates(provider: Provider, body: &Value) -> Vec<String> {
    let openrouter = provider == Provider::OpenRouter;
    let mut out: Vec<String> = Vec::new();
    let mut push = |id: &str| {
        if id.is_empty() {
            return;
        }
        out.push(id.to_string());
        if openrouter {
            if let Some(stripped) = strip_provider_prefix(id) {
                out.push(stripped.to_string());
            }
        }
    };
    if let Some(m) = body.get("model").and_then(|v| v.as_str()) {
        push(m);
    }
    if openrouter {
        if let Some(arr) = body.get("models").and_then(|v| v.as_array()) {
            for m in arr.iter().filter_map(|v| v.as_str()) {
                push(m);
            }
        }
    }
    out
}

/// Resolve a request's `model` to a concrete local catalog id using `resolve` (the
/// per-capability lookup, returning `Some(extra)` on a hit — `()` for chat/embeddings,
/// the text-to-image action name for images). Walks [`candidates`] in order and
/// returns the first `(resolved_id, extra)` that resolves, or `None` (a
/// `model_not_found`). This is where OpenRouter's prefix-strip + `models` fallback
/// live; OpenAI/Anthropic reduce to a single exact-match lookup of the primary `model`.
pub fn resolve_model<T, F: Fn(&str) -> Option<T>>(provider: Provider, body: &Value, resolve: F) -> Option<(String, T)> {
    candidates(provider, body).into_iter().find_map(|id| resolve(&id).map(|extra| (id, extra)))
}

/// The pure text-to-image action of an image manifest: emits an `Image` output,
/// takes a `prompt` param, and requires NO input blob (so image-editing actions like
/// `image2image`/`inpaint`/`outpaint`, which need a source image, are skipped).
fn text2image_action(m: &Manifest) -> Option<String> {
    m.actions
        .iter()
        .find(|a| {
            a.outputs.iter().any(|o| o.media == Media::Image)
                && a.params.iter().any(|p| p.name == "prompt")
                && !a.inputs.iter().any(|b| b.required)
        })
        .map(|a| a.name.clone())
}
