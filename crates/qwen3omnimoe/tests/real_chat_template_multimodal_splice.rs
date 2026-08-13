// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-only, no-GPU proof that a real typed multimodal content array,
//! rendered through the REAL Qwen3-Omni-30B-A3B-Instruct chat template on
//! disk, places each medium's placeholder run exactly where its content part
//! appeared -- inside the user turn, right after its own caption -- not
//! before the system prompt (the bug this session's first fix closed) and
//! not merely somewhere after it (the whole-turn heuristic that fix used
//! before this session's second, real-architecture fix moved media splicing
//! to an inline, per-content-part expansion).
//!
//! Real-weight-adjacent (needs the real tokenizer_config.json/
//! chat_template.json on disk), so it follows the engine's standard
//! opt-in-env-var pattern: skips (never panics) when `BRAIN_OMNI_HF_DIR` is
//! unset or the checkpoint dir is absent.
//!
//! usage: BRAIN_OMNI_HF_DIR=/tmp/.X11-unix/brain/omni/Qwen3-Omni-30B-A3B-Instruct \
//!        cargo test --release -p brain-omni --test real_chat_template_multimodal_splice -- --ignored

use capability::Invocation;
use serde_json::json;
use std::path::PathBuf;

fn hf_dir() -> Option<PathBuf> {
    let d = PathBuf::from(std::env::var("BRAIN_OMNI_HF_DIR").ok()?);
    d.join("chat_template.json").exists().then_some(d)
}

/// The exact shape `crates/apiserve/src/openai.rs`'s `message_content`
/// (this session's Gap A/B fix) now builds for a real sven-shaped request:
/// a plain-string system turn, and a user turn whose content is a typed
/// array with `image_url`/`input_audio` parts interleaved with their own
/// text captions -- `input_audio` carrying the `"audio"`/`"audio_url"` keys
/// `message_content` adds so the template's own audio detection
/// (`content.type == 'audio' or 'audio' in content or 'audio_url' in
/// content`) actually matches it (Gap B).
fn sven_shaped_messages() -> serde_json::Value {
    json!([
        {"role": "system", "content": "You are Sven, a specialized AI coding agent built for professional software engineering."},
        {"role": "user", "content": [
            {"type": "text", "text": "Follow the spoken instruction."},
            {"type": "text", "text": "Attached image: three-objects.png"},
            {"type": "image_url", "image_url": {}},
            {"type": "text", "text": "Attached audio: explain-this-image.wav"},
            {"type": "input_audio", "input_audio": {"format": "wav"}, "audio": true, "audio_url": true},
        ]},
    ])
}

#[test]
#[ignore]
fn real_template_places_image_and_audio_placeholders_inline_after_their_own_captions() {
    let Some(dir) = hf_dir() else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset or chat_template.json missing");
        return;
    };
    let tmpl = data::chat_template::ChatTemplate::from_model_dir(&dir).expect("load real chat template");

    let messages = sven_shaped_messages();
    let inv = Invocation::new().set("messages", json!(serde_json::to_string(&messages).unwrap())).set("enable_thinking", json!(false));
    let prompt = qwen3omnimoe::caps::render_chat_prompt(&tmpl, &inv).expect("render");

    let img_pos = prompt.find("<|vision_start|><|image_pad|><|vision_end|>").expect("image placeholder must be present in the rendered text");
    let audio_pos = prompt.find("<|audio_start|><|audio_pad|><|audio_end|>").expect("audio placeholder must be present in the rendered text");
    let system_end = prompt.find("<|im_end|>").expect("system turn must close");
    let user_open = prompt.find("<|im_start|>user").expect("user turn must open");

    assert!(system_end < user_open, "system turn must close before the user turn opens: {prompt:?}");
    // The real bug this session's FIRST fix closed: media used to be spliced
    // before index 0 of the whole rendered text, i.e. before the system
    // prompt entirely. That can't be checked at the STRING level here (this
    // test is about the template's own rendering, not mm.rs's splice), but
    // this next assertion catches the SAME class of failure at this layer:
    // a placeholder appearing before the user turn even opens would mean the
    // template put media outside where it belongs.
    assert!(user_open < img_pos, "image placeholder must be INSIDE the user turn, not before it opens: {prompt:?}");
    assert!(user_open < audio_pos, "audio placeholder must be INSIDE the user turn, not before it opens: {prompt:?}");

    // The real, precise architecture this session's SECOND fix ported: each
    // placeholder sits exactly where its own content part appeared, not
    // merely "somewhere after the system turn" (the first fix's heuristic).
    assert!(img_pos < audio_pos, "the image content part appears before the audio one in the real request, so its placeholder must too: {prompt:?}");
    let before_img = &prompt[..img_pos];
    assert!(before_img.ends_with("three-objects.png"), "image placeholder must sit right after ITS OWN caption, not the audio one or anywhere else: ...{:?}", &before_img[before_img.len().saturating_sub(60)..]);
    let before_audio = &prompt[..audio_pos];
    assert!(before_audio.ends_with("explain-this-image.wav"), "audio placeholder must sit right after ITS OWN caption: ...{:?}", &before_audio[before_audio.len().saturating_sub(60)..]);

    // And the generation prompt still closes the whole thing correctly
    // (unaffected by any of this -- a real regression guard, since an
    // earlier probe this session bypassed openai.rs's own flattening and
    // could not have caught a regression here).
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "generation prompt suffix: {:?}", &prompt[prompt.len().saturating_sub(60)..]);

    println!("real chat template places image/audio placeholders inline, after their own captions, inside the user turn.");
}

/// Gap C's regression test: `inv.get_str("tools")` (already set by
/// `crates/apiserve/src/openai.rs`'s `to_invocation` whenever a real request
/// supplies one) must now actually reach the template's own `{%- if tools %}`
/// preamble -- previously `render_chat_prompt` always rendered with
/// `tools: None`, so this branch NEVER fired for any real request, no matter
/// what the client sent.
#[test]
#[ignore]
fn real_template_renders_the_tools_preamble_when_tools_are_forwarded() {
    let Some(dir) = hf_dir() else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset or chat_template.json missing");
        return;
    };
    let tmpl = data::chat_template::ChatTemplate::from_model_dir(&dir).expect("load real chat template");

    let messages = json!([{"role": "system", "content": "be helpful"}, {"role": "user", "content": "what's the weather?"}]);
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}}}]);
    let inv = Invocation::new()
        .set("messages", json!(serde_json::to_string(&messages).unwrap()))
        .set("tools", json!(serde_json::to_string(&tools).unwrap()))
        .set("enable_thinking", json!(false));
    let prompt = qwen3omnimoe::caps::render_chat_prompt(&tmpl, &inv).expect("render");

    assert!(prompt.contains("# Tools"), "tools preamble must render when tools are supplied: {prompt:?}");
    assert!(prompt.contains("<tools>"), "tool schema block must render: {prompt:?}");
    assert!(prompt.contains("\"name\": \"get_weather\""), "the real tool schema (via the tojson filter) must appear verbatim: {prompt:?}");

    // Without tools, the preamble must NOT appear (the template's own
    // `{%- if tools %}` branch is genuinely conditional, not a bug pinned to
    // "always on" by this fix).
    let inv_no_tools = Invocation::new().set("messages", json!(serde_json::to_string(&messages).unwrap())).set("enable_thinking", json!(false));
    let prompt_no_tools = qwen3omnimoe::caps::render_chat_prompt(&tmpl, &inv_no_tools).expect("render");
    assert!(!prompt_no_tools.contains("# Tools"), "no tools supplied -> no tools preamble: {prompt_no_tools:?}");

    println!("real chat template renders the tools preamble exactly when tools are forwarded.");
}

/// Real regression test for `crate::mm::strip_media_placeholder_text`'s
/// entire reason to exist: the typed-content-array render path (Gap A/B,
/// this session) plus a string-level strip of the resulting inline
/// placeholder literal must produce a prompt BYTE-IDENTICAL to the old,
/// always-flatten path (`content_text`-style, no typed array, no
/// placeholder ever rendered) - proving the two paths are interchangeable
/// at the text level, which is what makes real embedding placement
/// (a whole-block splice, unconditionally - see `crate::mm`'s module doc)
/// safe to keep using regardless of which path built the prompt.
#[test]
#[ignore]
fn typed_array_plus_strip_matches_the_old_always_flatten_prompt_byte_for_byte() {
    let Some(dir) = hf_dir() else {
        eprintln!("skip: BRAIN_OMNI_HF_DIR unset or chat_template.json missing");
        return;
    };
    let tmpl = data::chat_template::ChatTemplate::from_model_dir(&dir).expect("load real chat template");

    let system = "You are Sven, a specialized AI coding agent built for professional software engineering.";
    let instruction = "Follow the spoken instruction.";
    let img_caption = "Attached image: three-objects.png (697x503, image/png)";
    let audio_caption = "Attached audio: explain-this-image.wav (5.8s, 44100 Hz)";

    let old_user_content = format!("{instruction}{img_caption}{audio_caption}");
    let old_messages = json!([
        {"role": "system", "content": system},
        {"role": "user", "content": old_user_content},
    ]);
    let old_inv = Invocation::new().set("messages", json!(serde_json::to_string(&old_messages).unwrap())).set("enable_thinking", json!(false));
    let old_prompt = qwen3omnimoe::caps::render_chat_prompt(&tmpl, &old_inv).expect("render old");

    let new_messages = json!([
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "text", "text": img_caption},
            {"type": "image_url", "image_url": {}},
            {"type": "text", "text": audio_caption},
            {"type": "input_audio", "input_audio": {"format": "wav"}, "audio": true, "audio_url": true},
        ]},
    ]);
    let new_inv = Invocation::new().set("messages", json!(serde_json::to_string(&new_messages).unwrap())).set("enable_thinking", json!(false));
    let new_prompt_raw = qwen3omnimoe::caps::render_chat_prompt(&tmpl, &new_inv).expect("render new");
    let new_prompt = qwen3omnimoe::mm::strip_media_placeholder_text(&new_prompt_raw);

    println!("old_prompt len={}", old_prompt.len());
    println!("new_prompt len={}", new_prompt.len());
    println!("EQUAL: {}", old_prompt == new_prompt);
    if old_prompt != new_prompt {
        let old_b = old_prompt.as_bytes();
        let new_b = new_prompt.as_bytes();
        let n = old_b.len().min(new_b.len());
        let mut diff_at = None;
        for i in 0..n {
            if old_b[i] != new_b[i] {
                diff_at = Some(i);
                break;
            }
        }
        if let Some(i) = diff_at {
            let lo = i.saturating_sub(40);
            println!("FIRST DIFF at byte {i}:");
            println!("  old: ...{:?}", &old_prompt[lo..(i + 40).min(old_prompt.len())]);
            println!("  new: ...{:?}", &new_prompt[lo..(i + 40).min(new_prompt.len())]);
        } else {
            println!("common prefix matches for {n} bytes; length differs: old={} new={}", old_b.len(), new_b.len());
            if old_b.len() > n {
                println!("  old tail: {:?}", &old_prompt[n..]);
            }
            if new_b.len() > n {
                println!("  new tail: {:?}", &new_prompt[n..]);
            }
        }
    }
    assert_eq!(old_prompt, new_prompt, "the two rendering paths must produce byte-identical prompts");
}
