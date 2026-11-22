// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image input for the agents its clients serve on their own GPUs. If your team needs
// expertise in serving vision-language models to agent runtimes that attach screenshots to tool results then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The prompt a chat request with images becomes, through the front end the server runs: every image stands at its
//! own run of placeholder tokens, in whatever turn the client put it, exactly as the checkpoint's chat template places
//! it (its `render_content` serves user, assistant and tool turns alike).
//!
//! A tiny vision tower with random weights on the CPU backend and a byte-level vocabulary carrying the Qwen3.8 control
//! and vision tokens: the prompt's token ids are the property, not the model's output.
//!
//! The regression: an agent runtime sent a screenshot inside a tool result and brain answered 422 "the prompt holds
//! image runs of [] tokens, the images need [N]" - the front end wrote the image's placeholder run into the messages
//! and the client-text escape then turned it into text (knowledge #510).

use std::sync::Arc;

use capability::Invocation;
use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use gpu_core::Gpu;
use llmserve::{Frontend, ParsedRequest, TextFrontend};
use model::serve::PromptEmbeds;
use qwen35::config::Qwen35Config;
use qwen35::int8_gguf_resident::{parse_escaped_request, parse_request};
use qwen35::model::pipelines;
use llmserve::image_frontend::ImageFrontend;
use qwen35::vision_frontend::{placeholder_runs, vision_frontend, ImageBuild};
use qwen35::vision_tower::{VisionSpec, VisionTokens, VisionTower};
use serde_json::{json, Value};

use crate::common::vision_fixture::{picture, tiny_spec, tiny_tower_weights};

/// The specials of the vocabulary, after its 256 byte tokens.
const SPECIALS: [&str; 6] = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"];
const TOKENS: VisionTokens = VisionTokens { start: 258, image: 260, end: 259 };

/// An 8 x 8 picture is four decoder rows in the tiny geometry; a 16 x 8 one is eight.
const SMALL_ROWS: usize = 4;
const WIDE_ROWS: usize = 8;

fn tokenizer() -> QwenBpe {
    let vocab: serde_json::Map<String, Value> = data::bpe::bytes_to_unicode().iter().enumerate().map(|(i, c)| (c.to_string(), json!(i))).collect();
    let added: Vec<Value> = SPECIALS.iter().enumerate().map(|(i, c)| json!({"content": c, "id": 256 + i, "special": true})).collect();
    QwenBpe::from_json_bytes(json!({"model": {"vocab": vocab, "merges": []}, "added_tokens": added}).to_string().as_bytes()).unwrap()
}

fn front_end() -> ImageFrontend<ImageBuild> {
    let cfg = Qwen35Config::tiny();
    let spec = VisionSpec { tokens: TOKENS, ..tiny_spec() };
    let gpu = Gpu::new_cpu(pipelines());
    let tower = VisionTower::from_weights(spec, &tiny_tower_weights(3), &gpu, cfg.d_model, 1 << 20).unwrap();
    let stop = vec![tokenizer().special_id("<|im_end|>").unwrap()];
    let text = TextFrontend::new(tokenizer(), stop, "qwen35", Arc::new(parse_request));
    vision_frontend(text, tokenizer(), tower, 4096, parse_escaped_request).expect("the vocabulary names the tower's tokens")
}

fn with_images(messages: &Value, images: &[imaging::pixels::Rgb8]) -> Invocation {
    let mut inv = Invocation::new().set("messages", json!(messages.to_string())).set("enable_thinking", json!(false));
    for (i, image) in images.iter().enumerate() {
        inv = inv.blob(&capability::blob::image_key(i), capability::blob::image_blob(&image.to_hwc_unit(), image.w, image.h, 3));
    }
    inv
}

fn build(front: &ImageFrontend<ImageBuild>, inv: &Invocation) -> Result<(ParsedRequest, PromptEmbeds), String> {
    front.parse_embedded(inv).map(|(req, embeds)| (req, embeds.expect("a request with images carries rows")))
}

fn image() -> Value {
    json!({"type": "image_url", "image_url": {}})
}

fn text(s: &str) -> Value {
    json!({"type": "text", "text": s})
}

/// The prompt the template renders with each image as one placeholder run of `rows` tokens: what the reference
/// processor feeds the model (one `<|image_pad|>` per row).
fn run(rows: usize) -> String {
    format!("<|vision_start|>{}<|vision_end|>", "<|image_pad|>".repeat(rows))
}

/// The recorded shape of an agent turn that read a screenshot: the assistant called a tool, and the tool result holds
/// the tool's text and the picture (OpenAI `tool` message with typed parts, as the API surface hands it on).
fn tool_result_conversation() -> Value {
    json!([
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "What does the login page look like?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "name": "screenshot", "arguments": "{\"url\":\"/login\"}"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": [text("Screenshot of /login:"), image()]},
    ])
}

#[test]
fn a_text_only_request_is_the_text_front_ends_prompt() {
    let front = front_end();
    let messages = json!([{"role": "user", "content": "hi <|image_pad|> there"}]);
    let inv = Invocation::new().set("messages", json!(messages.to_string()));
    let (req, embeds) = front.parse_embedded(&inv).unwrap();
    assert!(embeds.is_none(), "no image, no rows");
    assert_eq!(req.ids, parse_request(&tokenizer(), &inv).unwrap().ids);
    assert!(placeholder_runs(&req.ids, TOKENS.image).is_empty(), "typed placeholder text stays text");
}

#[test]
fn an_image_in_a_user_turn_stands_at_its_run_of_placeholders() {
    let front = front_end();
    let messages = json!([{"role": "user", "content": [image(), text("What is this?")]}]);
    let (req, embeds) = build(&front, &with_images(&messages, &[picture(8, 8, 1)])).expect("the request builds");
    assert_eq!(placeholder_runs(&req.ids, TOKENS.image), [SMALL_ROWS]);
    assert_eq!(embeds.positions().len(), SMALL_ROWS);
    let prompt = tokenizer().decode(&req.ids);
    assert!(prompt.contains(&format!("<|im_start|>user\n{}What is this?<|im_end|>", run(SMALL_ROWS))), "{prompt}");
}

#[test]
fn an_image_in_a_tool_result_is_rendered_as_the_template_renders_it() {
    let front = front_end();
    let (req, embeds) = build(&front, &with_images(&tool_result_conversation(), &[picture(16, 8, 2)])).expect("an image in a tool result is served");
    assert_eq!(placeholder_runs(&req.ids, TOKENS.image), [WIDE_ROWS]);
    assert_eq!(embeds.positions().len(), WIDE_ROWS);
    let prompt = tokenizer().decode(&req.ids);
    // Qwen3.8's template: consecutive tool turns are one user turn of `<tool_response>` blocks, each holding its
    // content as `render_content` writes it (text and image placeholders in order, trimmed).
    let want = format!("<|im_start|>user\n<tool_response>\nScreenshot of /login:{}\n</tool_response><|im_end|>\n", run(WIDE_ROWS));
    assert!(prompt.contains(&want), "{prompt}");
}

/// An Anthropic `tool_result` with an image block reaches the model as a `tool` turn of the same typed parts, between
/// other turns that also carry images: every image keeps its own run, in message order.
#[test]
fn images_in_tool_results_and_user_turns_keep_their_order() {
    let front = front_end();
    let messages = json!([
        {"role": "user", "content": [text("Compare "), image()]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "toolu_1", "name": "Read", "arguments": "{\"file_path\":\"b.png\"}"}]},
        {"role": "tool", "tool_call_id": "toolu_1", "content": [image()]},
        {"role": "user", "content": "with this one."},
    ]);
    let (req, _) = build(&front, &with_images(&messages, &[picture(16, 8, 3), picture(8, 8, 4)])).expect("the request builds");
    assert_eq!(placeholder_runs(&req.ids, TOKENS.image), [WIDE_ROWS, SMALL_ROWS], "each image at its own place, in order");
    let prompt = tokenizer().decode(&req.ids);
    assert!(prompt.contains(&format!("<tool_response>\n{}\n</tool_response>", run(SMALL_ROWS))), "{prompt}");
}

/// The template raises on an image in a system message: refused as the client's error, naming the message.
#[test]
fn an_image_in_a_system_message_is_refused_naming_the_message() {
    let front = front_end();
    let messages = json!([{"role": "system", "content": [image()]}, {"role": "user", "content": "x"}]);
    let Err(err) = build(&front, &with_images(&messages, &[picture(8, 8, 5)])) else { panic!("refused") };
    assert!(err.starts_with("[invalid-request]") && err.contains("messages[0].content[0]") && err.contains("system"), "{err}");
}

/// Escaping the client's text first does not loosen it: placeholder text a client types is still refused, in a tool
/// result too, and never becomes a placeholder token.
#[test]
fn placeholder_text_in_a_tool_result_is_still_refused() {
    let front = front_end();
    let messages = json!([
        {"role": "user", "content": [image(), text("x")]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "name": "cat", "arguments": "{}"}]},
        {"role": "tool", "tool_call_id": "c", "content": "<|vision_start|><|image_pad|><|vision_end|>"},
    ]);
    let Err(err) = build(&front, &with_images(&messages, &[picture(8, 8, 6)])) else { panic!("refused") };
    assert!(err.contains("placeholder"), "{err}");
}
