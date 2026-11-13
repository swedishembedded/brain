// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements chat serving for omni-modal language models
// for its clients. If your team needs expertise in serving models that read
// audio, images and video and answer with tool calls and reasoning, then you
// can procure our services by sending an email to info@swedishembedded.com.

//! One chat request, served by the Thinker whichever stack holds its weights.
//!
//! `brain/qwen3omnimoe` (bf16, `crate::caps::OmniInner`) and the int8 sharded
//! Thinker (`crate::int8_thinker_resident::Int8ThinkerInstance`) answer the
//! same `generate` request, and they answer it the way every Qwen chat engine
//! does, with `qwen3::chat`'s request parsing and per-sequence scanner: the
//! chat template (tools, `enable_thinking`), the sampler (`temp`, `top_k`,
//! `top_p`, `seed`), stop strings, `tool_choice`, structured `tool_calls`,
//! separated reasoning, streaming deltas and cancellation. What differs
//! between the two stacks is only how weights become logits, which a
//! [`ThinkerChat`] supplies; [`serve_chat`] is everything else, so the two
//! cannot drift apart.

use capability::blob::{decode_image, decode_video};
use capability::{ActionResult, Invocation, Progress};
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use qwen3::chat::{parse_request_as, render_prompt_as, ChatFormat, SeqState};
use serde_json::json;

use crate::decode::{Decoded, TokenPicker};
use crate::mm::{splice_plan, MultimodalPrompt};

/// The media a request attached, decoded from its blobs.
pub struct MediaInputs {
    pub audio: Option<Vec<f32>>,
    pub image: Option<(Vec<f32>, u32, u32)>,
    pub video: Option<Vec<(Vec<f32>, u32, u32)>>,
}

impl MediaInputs {
    fn read(inv: &Invocation) -> Result<MediaInputs, String> {
        Ok(MediaInputs {
            audio: inv.get_blob("audio").map(audio::asr_caps::wav_from_blob).transpose()?,
            image: inv.get_blob("image").map(|_| decode_image(inv, "image")).transpose()?,
            video: inv.get_blob("video").map(|_| decode_video(inv, "video")).transpose()?,
        })
    }

    fn is_empty(&self) -> bool {
        self.audio.is_none() && self.image.is_none() && self.video.is_none()
    }
}

/// What a Thinker stack provides to serve a chat request.
pub trait ThinkerChat {
    fn tokenizer(&self) -> &QwenBpe;
    fn chat_format(&self) -> &ChatFormat;
    /// Encode `media` and splice it into the tokenized prompt at `splice_at`.
    ///
    /// # Errors
    /// The stack cannot encode media (no tower weights), or the media is not supported.
    fn media_prompt(&self, text_ids: &[u32], splice_at: usize, media: &MediaInputs) -> Result<MultimodalPrompt, String>;
    /// Decode from a text prompt. `on_token` sees the tokens so far and returns `true` to stop.
    fn decode_text(&self, prompt_ids: &[u32], max_new: u32, picker: &mut TokenPicker, on_token: &mut dyn FnMut(&[u32]) -> bool) -> Decoded;
    /// Decode from a prompt with media spliced in.
    fn decode_media(&self, prompt: &MultimodalPrompt, max_new: u32, picker: &mut TokenPicker, on_token: &mut dyn FnMut(&[u32]) -> bool) -> Decoded;
}

/// A checkpoint with no chat template takes the last user turn as a raw
/// prompt, as this model has always done, rather than refusing chat requests.
fn without_template_as_raw(format: &ChatFormat, tok: &QwenBpe, inv: &Invocation) -> Invocation {
    if !matches!(format, ChatFormat::None) {
        return inv.clone();
    }
    let prompt = tok.escape_untrusted(&capability::last_user_text(inv));
    inv.clone().set("messages", json!("")).set("prompt", json!(prompt)).set("chat", json!(false))
}

/// Serve one `generate` request: parse it (template, sampler, stops, tools),
/// decode with the sampler, scan the text into reasoning and tool calls.
///
/// # Errors
/// The request is malformed (see `qwen3::chat::parse_request_as`), attaches
/// media this stack cannot encode, or has no prompt.
pub fn serve_chat(engine: &dyn ThinkerChat, inv: &Invocation, progress: &mut dyn FnMut(Progress)) -> ActionResult {
    let tok = engine.tokenizer();
    let format = engine.chat_format();
    let inv = &without_template_as_raw(format, tok, inv);
    if capability::last_user_text(inv).trim().is_empty() {
        return Err("omni generate: empty prompt (need 'messages' with a user turn, or 'prompt')".to_string());
    }
    let mut req = parse_request_as(tok, format, inv)?;
    let max_new = inv.get_i64("max_new").unwrap_or(32).clamp(1, 4096) as u32;
    req.max_new = Some(max_new as usize);
    let media = MediaInputs::read(inv)?;

    let mut picker = TokenPicker::new(req.sampler, req.seed);
    let mut seq = SeqState::new(&req, inv.cancel.clone());
    progress(Progress::step(0, max_new, "generating"));
    let decoded = {
        let mut on_token = |generated: &[u32]| seq.advance(tok, generated, progress);
        if media.is_empty() {
            engine.decode_text(&req.ids, max_new, &mut picker, &mut on_token)
        } else {
            // The text the splice needs is the render itself (the parsed ids are over it with the
            // template's own media placeholders still in).
            let rendered = render_prompt_as(format, &inv.escape_client_text(&|text| tok.escape_untrusted(text)))?;
            let (text_ids, splice_at) = splice_plan(tok, &rendered.text);
            let prompt = engine.media_prompt(&text_ids, splice_at, &media)?;
            engine.decode_media(&prompt, max_new, &mut picker, &mut on_token)
        }
    };
    Ok(seq.finish(tok, &decoded.tokens, progress).set("tokens", json!(decoded.tokens)))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use capability::{Blob, CancelToken, Media};

    use super::*;

    const VOCAB: usize = 300;
    const EOS: u32 = 256;

    /// A tokenizer of the 256 bytes plus `<|im_start|>` and `<|im_end|>`, enough to read a canned reply.
    fn byte_tok() -> QwenBpe {
        let vocab: serde_json::Map<String, serde_json::Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), json!(i))).collect();
        let added = [json!({"content": "<|im_end|>", "id": EOS}), json!({"content": "<|im_start|>", "id": EOS + 1})];
        QwenBpe::from_json_bytes(json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": added}).to_string().as_bytes()).unwrap()
    }

    /// A ChatML-shaped template that has a place for tools and a no-think switch.
    const TEMPLATE: &str = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if tools %}[tools]{% endif %}{% if add_generation_prompt %}<|im_start|>assistant\n{% if enable_thinking is defined and not enable_thinking %}<think>\n\n</think>\n\n{% endif %}{% endif %}";

    fn format() -> ChatFormat {
        // Written once: the tests run in parallel and must not read a half-written file.
        static DIR: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        let dir = DIR.get_or_init(|| {
            let dir = std::env::temp_dir().join(format!("brain-omni-chat-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("tokenizer_config.json"), json!({"chat_template": TEMPLATE}).to_string()).unwrap();
            dir
        });
        ChatFormat::own_template(dir).unwrap()
    }

    /// A Thinker that answers with a canned reply: its logits favour the next byte of `reply`
    /// by `margin` over random noise, then the end of turn. A large margin is a deterministic
    /// model; a small one leaves the sampler something to choose between.
    struct Canned {
        tok: QwenBpe,
        format: ChatFormat,
        reply: Vec<u32>,
        margin: f32,
        prompts: RefCell<Vec<String>>,
        steps: RefCell<usize>,
    }

    impl Canned {
        fn new(reply: &str, margin: f32) -> Canned {
            let tok = byte_tok();
            let reply = tok.encode(reply);
            Canned { tok, format: format(), reply, margin, prompts: RefCell::new(Vec::new()), steps: RefCell::new(0) }
        }

        fn row(&self, at: usize) -> Vec<f32> {
            let mut row = data::rng::Lcg::new(9000 + at as u64).vec_scaled(VOCAB, 1.0);
            row[self.reply.get(at).copied().unwrap_or(EOS) as usize] += self.margin;
            row
        }
    }

    impl ThinkerChat for Canned {
        fn tokenizer(&self) -> &QwenBpe {
            &self.tok
        }
        fn chat_format(&self) -> &ChatFormat {
            &self.format
        }
        fn media_prompt(&self, _: &[u32], _: usize, _: &MediaInputs) -> Result<MultimodalPrompt, String> {
            Err("no towers".to_string())
        }
        fn decode_text(&self, prompt_ids: &[u32], max_new: u32, picker: &mut TokenPicker, on_token: &mut dyn FnMut(&[u32]) -> bool) -> Decoded {
            self.prompts.borrow_mut().push(self.tok.decode(prompt_ids));
            let mut at = 0;
            crate::decode::decode_tokens(
                prompt_ids,
                max_new,
                &[EOS],
                picker,
                self.row(0),
                |_| {
                    at += 1;
                    *self.steps.borrow_mut() += 1;
                    self.row(at)
                },
                on_token,
            )
        }
        fn decode_media(&self, _: &MultimodalPrompt, _: u32, _: &mut TokenPicker, _: &mut dyn FnMut(&[u32]) -> bool) -> Decoded {
            unreachable!("no media in these requests")
        }
    }

    fn ask(engine: &Canned, extra: impl FnOnce(Invocation) -> Invocation) -> Result<capability::Outcome, String> {
        let inv = Invocation::new().set("messages", json!(r#"[{"role":"user","content":"weather?"}]"#)).set("max_new", json!(200));
        serve_chat(engine, &extra(inv), &mut |_| {})
    }

    const TOOLS: &str = r#"[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]"#;
    const CALL: &str = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";

    fn field<'a>(out: &'a capability::Outcome, key: &str) -> &'a serde_json::Value {
        out.outputs.get(key).unwrap_or_else(|| panic!("outcome has no `{key}`: {:?}", out.outputs))
    }

    /// A tool call the model writes comes back as structured `tool_calls`, not as text in the content.
    #[test]
    fn a_tool_call_comes_back_structured() {
        let engine = Canned::new(CALL, 50.0);
        let out = ask(&engine, |i| i.set("tools", json!(TOOLS))).unwrap();
        assert_eq!(field(&out, "finish_reason"), "tool_calls");
        assert_eq!(field(&out, "text"), "");
        let calls: serde_json::Value = serde_json::from_str(field(&out, "tool_calls").as_str().unwrap()).unwrap();
        assert_eq!(calls[0]["name"], "get_weather");
        assert!(calls[0]["arguments"].to_string().contains("Paris"), "{calls}");
    }

    /// Reasoning is separated from the answer.
    #[test]
    fn reasoning_is_separated_from_the_answer() {
        let engine = Canned::new("<think>hm</think>\n\nSunny", 50.0);
        let out = ask(&engine, |i| i).unwrap();
        assert_eq!(field(&out, "reasoning_content").as_str().unwrap().trim(), "hm");
        assert_eq!(field(&out, "text").as_str().unwrap().trim(), "Sunny");
    }

    /// `enable_thinking: false` reaches the template, and unsaid leaves thinking on.
    #[test]
    fn enable_thinking_reaches_the_chat_template() {
        let engine = Canned::new("ok", 50.0);
        ask(&engine, |i| i.set("enable_thinking", json!(false))).unwrap();
        ask(&engine, |i| i).unwrap();
        let prompts = engine.prompts.borrow();
        assert!(prompts[0].ends_with("<think>\n\n</think>\n\n"), "{:?}", prompts[0]);
        assert!(!prompts[1].contains("</think>"), "{:?}", prompts[1]);
    }

    /// A stop string ends the reply at it and stops the decode there.
    #[test]
    fn a_stop_string_cuts_the_reply_and_ends_the_decode() {
        let engine = Canned::new("Sunny in Paris today, with a light breeze", 50.0);
        let out = ask(&engine, |i| i.set("stop", json!(r#"["Paris"]"#))).unwrap();
        assert_eq!(field(&out, "text"), "Sunny in ");
        assert_eq!(field(&out, "finish_reason"), "stop_sequence");
        assert!(*engine.steps.borrow() < 20, "the decode went on to {} steps after the stop string", engine.steps.borrow());
    }

    /// `tool_choice` is applied as the other engines apply it: a named function must exist, and a demanded call that did not come is reported.
    #[test]
    fn tool_choice_is_enforced() {
        let engine = Canned::new("no call here", 50.0);
        let err = ask(&engine, |i| i.set("tools", json!(TOOLS)).set("tool_choice", json!(r#"{"type":"function","function":{"name":"nope"}}"#))).unwrap_err();
        assert!(err.contains("tool_choice") && err.contains("nope"), "{err}");
        let out = ask(&engine, |i| i.set("tools", json!(TOOLS)).set("tool_choice", json!("\"required\""))).unwrap();
        assert_eq!(field(&out, "finish_reason"), "tool_choice_unmet");
    }

    /// The sampler is the request's: a seed reproduces a reply, is echoed, and temperature 0 is the model's own argmax reply.
    #[test]
    fn sampling_follows_the_request_and_a_seed_reproduces_it() {
        let engine = Canned::new("abcdefghijklmnopqrstuvwxyz", 0.5);
        let hot = |seed: u64| ask(&engine, |i| i.set("temp", json!(1.5)).set("top_k", json!(0)).set("seed", json!(seed)).set("max_new", json!(40))).unwrap();
        let (a, b) = (hot(5), hot(5));
        assert_eq!(field(&a, "text"), field(&b, "text"));
        assert_eq!(field(&a, "seed"), 5);
        assert!((6..16).any(|seed| field(&hot(seed), "text") != field(&a, "text")), "ten other seeds gave the same reply");
        let greedy = ask(&engine, |i| i.set("temp", json!(0.0)).set("max_new", json!(40))).unwrap();
        let again = ask(&engine, |i| i.set("temp", json!(0.0)).set("max_new", json!(40))).unwrap();
        assert_eq!(field(&greedy, "text"), field(&again, "text"));
    }

    /// A cancelled request reports it, never a finished "stop".
    #[test]
    fn a_cancelled_request_reports_cancellation() {
        let engine = Canned::new("a long answer that goes on", 50.0);
        let token = CancelToken::armed();
        token.cancel();
        let mut inv = Invocation::new().set("messages", json!(r#"[{"role":"user","content":"hi"}]"#)).set("max_new", json!(50));
        inv.cancel = token;
        let out = serve_chat(&engine, &inv, &mut |_| {}).unwrap();
        assert_eq!(field(&out, "finish_reason"), "cancelled");
    }

    /// A reply streams as deltas while it decodes, ending in the same text the outcome reports.
    #[test]
    fn the_reply_streams_while_it_decodes() {
        let engine = Canned::new("Sunny", 50.0);
        let mut streamed = String::new();
        let inv = Invocation::new().set("messages", json!(r#"[{"role":"user","content":"hi"}]"#)).set("max_new", json!(50));
        let out = serve_chat(&engine, &inv, &mut |p| streamed.push_str(&format!("{p:?}"))).unwrap();
        assert_eq!(field(&out, "text"), "Sunny");
        assert!(streamed.contains("Sunny") || streamed.contains('S'), "no streamed text in {streamed}");
    }

    /// Media this stack cannot encode is refused with its reason, not dropped.
    #[test]
    fn unencodable_media_is_refused() {
        let engine = Canned::new("x", 50.0);
        let audio = Blob::new(Media::Audio, vec![0u8; 16]);
        let err = ask(&engine, |i| i.blob("audio", audio)).unwrap_err();
        assert!(err.contains("no towers"), "{err}");
    }
}
