// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Proves `qwen3omnimoe::caps::manifest()` is actually classified chat-capable by
//! the real `apiserve::catalog::api_caps` logic — not a re-derivation of
//! that logic (which could silently drift from the real check), the ACTUAL
//! function `/v1/chat/completions`/`/v1/messages` gate exposure on. The
//! D-Bus surface is fully generic (any registered resident is servable with
//! zero new code) but the OpenAI/Anthropic surfaces additionally require
//! `streaming` + a
//! `messages`/`prompt`/`text` param + a `Media::Text` output — this test is
//! the regression guard for that shape, since `crates/omni/src/caps.rs`'s
//! own module doc explaining WHY it has that shape is easy to silently
//! break with an innocuous-looking edit.
//!
//! No real weights needed — `Manifest`/`ActionSpec` construction is pure.

use apiserve::catalog::api_caps;
use capability::{Invocation, Media};

#[test]
fn omni_manifest_is_chat_exposed() {
    let caps = api_caps(&qwen3omnimoe::caps::manifest());
    assert!(caps.chat, "qwen3omnimoe::caps::manifest() is not classified chat-capable by apiserve::catalog::api_caps -- \
        /v1/chat/completions and /v1/messages would 404 it, same as if it were never registered");
}

/// Spec: the GPU-resident int8 Thinker is reachable over the SAME chat
/// surfaces as `brain/qwen3omnimoe`. It used to declare raw blob actions only, so
/// `/v1/chat/completions` 404'd the one model on this box fast enough to be
/// worth calling (more than an order of magnitude on the streaming path's
/// tokens per second, on two P40s) - the
/// speed was unreachable through the interface anyone actually uses.
#[test]
fn int8_thinker_manifest_is_chat_exposed() {
    use residency::{Device, ResidentModel};
    // Manifest construction is pure: no checkpoint is opened, no GPU touched.
    let r = qwen3omnimoe::int8_thinker_resident::Int8ThinkerResident::new(
        "/nonexistent.safetensors".to_string(),
        qwen3omnimoe::config::ThinkerConfig::defaults(),
        vec![(Device::Gpu(0), 1 << 34)],
    );
    let caps = api_caps(&r.manifest());
    assert!(
        caps.chat,
        "brain/Qwen3-Omni-30B-A3B-Instruct-W8A16 is not classified chat-capable by apiserve::catalog::api_caps -- \
         /v1/chat/completions and /v1/messages would 404 it, leaving the fast path reachable only over D-Bus"
    );
}

/// Spec: both Thinker-backed models answer the same request contract, because
/// they build their `generate` spec from the same `chat_generate_spec` - a
/// param the chat path sets that only one of them declares is a request that
/// silently means different things depending on which model you address.
#[test]
fn both_thinker_models_declare_the_same_chat_params() {
    use residency::{Device, ResidentModel};
    let r = qwen3omnimoe::int8_thinker_resident::Int8ThinkerResident::new(
        "/nonexistent.safetensors".to_string(),
        qwen3omnimoe::config::ThinkerConfig::defaults(),
        vec![(Device::Gpu(0), 1 << 34)],
    );
    let m = r.manifest();
    let int8 = m.actions.iter().find(|a| a.name == "generate").expect("int8 must declare generate");
    for p in qwen3omnimoe::caps::generate_spec().params {
        assert!(
            int8.params.iter().any(|q| q.name == p.name),
            "brain/qwen3omnimoe declares chat param '{}' but brain/Qwen3-Omni-30B-A3B-Instruct-W8A16 does not",
            p.name
        );
    }
}

/// Spec: both Thinker-backed models declare the SAME media inputs on
/// `generate` - `brain/Qwen3-Omni-30B-A3B-Instruct-W8A16` used to declare none at all
/// (multimodal input was silently dropped even though `apiserve::openai`
/// attaches `image`/`audio` blobs to every request regardless of which model
/// the caller addressed), which meant a client got a text-only answer with no
/// error telling it why. A mismatch here (a name only one model declares, or
/// a different `Media` kind for the same name) is exactly the kind of drift
/// `qwen3omnimoe::caps::with_multimodal_inputs` exists to make impossible by
/// construction; this test is the mechanical guard in case a future edit
/// still manages it (e.g. by not routing through that helper).
#[test]
fn both_thinker_models_declare_the_same_media_inputs() {
    use residency::{Device, ResidentModel};
    let r = qwen3omnimoe::int8_thinker_resident::Int8ThinkerResident::new(
        "/nonexistent.safetensors".to_string(),
        qwen3omnimoe::config::ThinkerConfig::defaults(),
        vec![(Device::Gpu(0), 1 << 34)],
    );
    let m = r.manifest();
    let int8 = m.actions.iter().find(|a| a.name == "generate").expect("int8 must declare generate");
    let bf16_inputs = qwen3omnimoe::caps::generate_spec().inputs;
    assert!(!bf16_inputs.is_empty(), "brain/qwen3omnimoe's generate_spec must declare at least one media input (audio/image/video)");
    for want in &bf16_inputs {
        let got = int8.inputs.iter().find(|i| i.name == want.name);
        assert!(
            got.is_some(),
            "brain/qwen3omnimoe declares media input '{}' but brain/Qwen3-Omni-30B-A3B-Instruct-W8A16 does not -- \
             an attached blob would be silently dropped on the fast path",
            want.name
        );
        assert_eq!(
            got.unwrap().media,
            want.media,
            "brain/qwen3omnimoe's '{}' input is {:?} but brain/Qwen3-Omni-30B-A3B-Instruct-W8A16's is {:?}",
            want.name,
            want.media,
            got.unwrap().media
        );
    }
}

#[test]
fn last_user_text_prefers_messages_last_user_turn() {
    let inv = Invocation::new().set(
        "messages",
        serde_json::json!([
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second"},
        ])
        .to_string()
        .into(),
    );
    assert_eq!(qwen3omnimoe::caps::last_user_text(&inv), "second");
}

#[test]
fn last_user_text_falls_back_to_prompt() {
    let inv = Invocation::new().set("prompt", serde_json::json!("direct prompt"));
    assert_eq!(qwen3omnimoe::caps::last_user_text(&inv), "direct prompt");
}

#[test]
fn last_user_text_falls_back_to_last_message_when_no_user_role_matches() {
    // A messages array with no "user"-role entry (e.g. all assistant/tool) still
    // yields the array's last entry's content, matching crate::resident_mock's
    // identical fallback -- never an empty prompt when messages is non-empty.
    let inv = Invocation::new().set("messages", serde_json::json!([{"role": "assistant", "content": "only turn"}]).to_string().into());
    assert_eq!(qwen3omnimoe::caps::last_user_text(&inv), "only turn");
}

#[test]
fn generate_spec_declares_every_param_the_http_handlers_set() {
    let spec = qwen3omnimoe::caps::generate_spec();
    assert!(spec.streaming, "generate must be .streaming() -- api_caps requires it for chat exposure");
    for name in ["messages", "prompt", "max_new", "temp", "top_p", "top_k", "seed", "stop"] {
        assert!(spec.params.iter().any(|p| p.name == name), "generate_spec is missing param '{name}'");
    }
    assert!(spec.outputs.iter().any(|o| o.media == Media::Text), "generate_spec must output a Media::Text blob");
}

/// `speak` IS `.streaming()`: `OmniInner::speak` now vocodes via
/// `Codec::decode_omni_chunked`, emitting real audio chunks mid-run via
/// `Progress::chunk` for a `Subscribe`-based caller (the terminal `Outcome`
/// still carries the full reassembled waveform too, unchanged, for a plain
/// `Run` caller). `speak` is reached only by its literal action name --
/// `api_caps` only classifies `generate` as chat, so `speak` alone changing
/// shape would never silently start (or stop) being chat-exposed. This pins
/// the shape `caps.rs`'s own doc claims (`speak_spec`'s own doc: "text
/// response + spoken waveform out").
#[test]
fn speak_spec_declares_the_documented_shape() {
    let spec = qwen3omnimoe::caps::speak_spec();
    assert!(spec.streaming, "speak now streams real audio chunks mid-run via Progress::chunk");
    for name in ["messages", "prompt", "max_new", "speaker"] {
        assert!(spec.params.iter().any(|p| p.name == name), "speak_spec is missing param '{name}'");
    }
    assert!(spec.outputs.iter().any(|o| o.media == Media::Text), "speak_spec must output a Media::Text blob");
    assert!(spec.outputs.iter().any(|o| o.media == Media::Audio), "speak_spec must output a Media::Audio blob");
}

/// `converse` must NOT be chat-exposed: `api_caps` gates chat classification
/// on the literal action name `generate` (`crates/apiserve/src/catalog.rs`),
/// so an action named `converse` is invisible to `/v1/chat/completions`/
/// `/v1/messages` regardless of its own shape -- this is the real function,
/// not a re-derivation, proving `caps.rs`'s own "D-Bus/CLI only" doc claim
/// for `converse` rather than merely asserting it in prose.
#[test]
fn converse_is_not_chat_exposed() {
    let solo_manifest = capability::Manifest::new(qwen3omnimoe::caps::MODEL, "converse-only probe", vec![qwen3omnimoe::caps::converse_spec()]);
    let caps = api_caps(&solo_manifest);
    assert!(!caps.chat, "converse alone must not be classified chat-capable -- it would then wrongly appear on /v1/chat/completions");
}

