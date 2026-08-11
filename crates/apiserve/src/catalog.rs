// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Deriving an API capability set from a model's `capability::Manifest`, and the
//! per-provider exposure filter.
//!
//! `api_caps` reads the *shape* of a model's actions — nothing is hard-coded per
//! model — so a new model shows up in the right `/models` list automatically:
//! - **Chat**: a streaming action **named `generate`** taking a `prompt`/`messages`/
//!   `text` param and emitting a `Text` output. Named, not just shaped, because the
//!   chat handlers always dispatch the literal action `"generate"`
//!   (`openai.rs`/`anthropic.rs`) — a differently-shaped-but-similar action (e.g.
//!   FastVLM's streaming `caption`, which also takes a `prompt` and emits `Text`)
//!   would be listed but then fail `ActionSpec::validate` on the chat params it
//!   doesn't declare. See `.agents/rules/serving-contract.md`.
//! - **Embeddings**: an action **named `embed`** that also takes a `text` param.
//!   Named AND shaped, for two independent reasons `/v1/embeddings`
//!   (`openai.rs::handle_embeddings`) breaks on otherwise-plausible matches: it
//!   always dispatches the literal action `"embed"` with a `text` param — never a
//!   blob — and always reads the result back from an `outputs.mean` scalar. That
//!   rules out two real models that used to slip through a looser "any embedding-
//!   shaped output" check: CLIP's `embed_text`/`embed_image` (right shape, wrong
//!   name — dispatch would 500 with "no action 'embed'") and FaceNet's `embed`
//!   (right name, but it takes a required `image` blob and no `text` param at all
//!   — dispatch would 400 on the missing input). LFM's `embed` (a `text` param,
//!   falling back to a `text` blob for long documents, and it actually populates
//!   `outputs.mean`) is the shape this rule is built to admit.
//! - **ImageGen**: a pure text-to-image action — see [`text2image_action`]. Not just
//!   "emits an `Image`", because `/images/generations` only ever dispatches that one
//!   action; image-*editing* models (restore/upscale/vqgan) emit `Image` too but have
//!   no such action and would otherwise be advertised on a route that 404s them.
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
        a.name == "generate"
            && a.streaming
            && a.params.iter().any(|p| matches!(p.name.as_str(), "prompt" | "messages" | "text"))
            && a.outputs.iter().any(|o| o.media == Media::Text)
    });
    let embeddings = m.actions.iter().any(|a| a.name == "embed" && a.params.iter().any(|p| p.name == "text"));
    let image = text2image_action(m).is_some();
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

/// The models `provider` exposes, as `(model_name, caps, max_context_tokens)`,
/// sorted by name — the input to every `/models` response.
pub fn exposed(exec: &Executor, provider: Provider) -> Vec<(String, CapSet, Option<u64>)> {
    let mut out: Vec<(String, CapSet, Option<u64>)> = exec
        .manifests()
        .iter()
        .map(|m| (m.model.clone(), api_caps(m), m.max_context_tokens))
        .filter(|(_, caps, _)| exposes(provider, *caps))
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
///
/// Takes the manifest SNAPSHOT, not the `Executor`: `Executor::manifests()` deep-
/// clones the whole catalog (every `ActionSpec`, param and help string), and the
/// OpenRouter candidate walk calls a resolver once per candidate id — so each
/// handler fetches the snapshot ONCE per request and the walk borrows it, instead
/// of paying a full catalog clone per candidate on the concurrent serving path.
pub fn resolve_chat(manifests: &[Manifest], model: &str) -> bool {
    manifests.iter().any(|m| m.model == model && api_caps(m).chat)
}

/// Resolve an embeddings request's `model` string: is there a manifest whose
/// `model == id` AND that advertises the embeddings capability? The `/embeddings`
/// dispatch (the `embed` action) gates on this — an unknown or non-embeddings model
/// is a `model_not_found`. Snapshot-taking — see [`resolve_chat`].
pub fn resolve_embed(manifests: &[Manifest], model: &str) -> bool {
    manifests.iter().any(|m| m.model == model && api_caps(m).embeddings)
}

/// Resolve an image request's `model` string to the **action name** to invoke: is
/// there a manifest whose `model == id` that advertises the image capability, and if
/// so, which action realises a text-to-image (the one to dispatch for
/// `/images/generations`)? `None` on an unknown or non-image model (a
/// `model_not_found`). The action is picked by shape — nothing is hard-coded per
/// model — so any model exposing a text→image action serves here (z-image/flux2 name
/// it `text2image`). Snapshot-taking — see [`resolve_chat`].
pub fn resolve_image(manifests: &[Manifest], model: &str) -> Option<String> {
    manifests.iter().find(|m| m.model == model && api_caps(m).image).and_then(text2image_action)
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
///
/// Exact match always precedes the stripped form of the same id, and the primary
/// `model` (both forms) is always tried before any `models` fallback entry.
///
/// A legacy short name (e.g. `"mock"`) is a deprecation, not a second id: every
/// pushed id also tries its canonical `brain/<name>` form immediately after the
/// literal, so an old client keeps working with no change to `Manifest.model`
/// itself (see `modelref::alias`'s module docs).
fn candidates(provider: Provider, body: &Value) -> Vec<String> {
    let openrouter = provider == Provider::OpenRouter;
    let mut out: Vec<String> = Vec::new();
    let mut push = |id: &str| {
        if id.is_empty() {
            return;
        }
        out.push(id.to_string());
        if let Some(canon) = brain_modelref::alias::canonical(id) {
            out.push(canon.to_string());
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn legacy_short_name_resolves_via_its_canonical_alias() {
        // A catalog holding only the canonical id -- a client still sending the
        // old bare "mock" must still resolve, without Manifest.model ever being
        // anything but "brain/mock".
        let body = json!({"model": "mock"});
        let resolve = |id: &str| (id == "brain/mock").then_some(());
        let got = resolve_model(Provider::Anthropic, &body, resolve);
        assert_eq!(got, Some(("brain/mock".to_string(), ())));
    }

    #[test]
    fn canonical_name_resolves_directly_with_no_alias_hop() {
        let body = json!({"model": "brain/mock"});
        let resolve = |id: &str| (id == "brain/mock").then_some(());
        let got = resolve_model(Provider::Anthropic, &body, resolve);
        assert_eq!(got, Some(("brain/mock".to_string(), ())));
    }

    #[test]
    fn unknown_name_with_no_alias_does_not_resolve() {
        let body = json!({"model": "totally-unknown"});
        let resolve = |id: &str| (id == "brain/mock").then_some(());
        assert_eq!(resolve_model(Provider::Anthropic, &body, resolve), None);
    }
}
