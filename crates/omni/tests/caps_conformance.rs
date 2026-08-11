// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Proves `omni::caps::manifest()` is actually classified chat-capable by
//! the real `apiserve::catalog::api_caps` logic — not a re-derivation of
//! that logic (which could silently drift from the real check), the ACTUAL
//! function `/v1/chat/completions`/`/v1/messages` gate exposure on. Written
//! after M10/M11/M12's investigation found the D-Bus surface is fully
//! generic (any registered resident is servable with zero new code) but the
//! OpenAI/Anthropic surfaces additionally require `streaming` + a
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
    let caps = api_caps(&omni::caps::manifest());
    assert!(caps.chat, "omni::caps::manifest() is not classified chat-capable by apiserve::catalog::api_caps -- \
        /v1/chat/completions and /v1/messages would 404 it, same as if it were never registered");
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
    assert_eq!(omni::caps::last_user_text(&inv), "second");
}

#[test]
fn last_user_text_falls_back_to_prompt() {
    let inv = Invocation::new().set("prompt", serde_json::json!("direct prompt"));
    assert_eq!(omni::caps::last_user_text(&inv), "direct prompt");
}

#[test]
fn last_user_text_falls_back_to_last_message_when_no_user_role_matches() {
    // A messages array with no "user"-role entry (e.g. all assistant/tool) still
    // yields the array's last entry's content, matching crate::resident_mock's
    // identical fallback -- never an empty prompt when messages is non-empty.
    let inv = Invocation::new().set("messages", serde_json::json!([{"role": "assistant", "content": "only turn"}]).to_string().into());
    assert_eq!(omni::caps::last_user_text(&inv), "only turn");
}

#[test]
fn generate_spec_declares_every_param_the_http_handlers_set() {
    let spec = omni::caps::generate_spec();
    assert!(spec.streaming, "generate must be .streaming() -- api_caps requires it for chat exposure");
    for name in ["messages", "prompt", "max_new", "temp", "top_p", "top_k", "seed", "stop"] {
        assert!(spec.params.iter().any(|p| p.name == name), "generate_spec is missing param '{name}'");
    }
    assert!(spec.outputs.iter().any(|o| o.media == Media::Text), "generate_spec must output a Media::Text blob");
}

/// `speak` is deliberately NOT `.streaming()` (one-shot: the whole waveform
/// is the single artifact today, per its own module doc) and is reached only
/// by its literal action name -- `api_caps` only classifies `generate` as
/// chat, so `speak` alone changing shape would never silently start (or
/// stop) being chat-exposed. This pins the shape `caps.rs`'s own doc claims
/// (`speak_spec`'s own doc: "text response + spoken waveform out").
#[test]
fn speak_spec_declares_the_documented_shape() {
    let spec = omni::caps::speak_spec();
    assert!(!spec.streaming, "speak is one-shot today, not .streaming()");
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
    let solo_manifest = capability::Manifest::new(omni::caps::MODEL, "converse-only probe", vec![omni::caps::converse_spec()]);
    let caps = api_caps(&solo_manifest);
    assert!(!caps.chat, "converse alone must not be classified chat-capable -- it would then wrongly appear on /v1/chat/completions");
}

