// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The whole file is about the `text` surface's chat half, so it compiles only
// with it.
#![cfg(feature = "text")]

//! `brain::ChatPipeline` through the public SDK only.
//!
//! Two halves. The first needs no weights at all: a [`brain::ChatRequest`]
//! renders to the prompt the served chat endpoint renders for the same
//! conversation, and a raw completion parses into text, reasoning and typed
//! tool calls the way a generation's own does. The second runs real
//! generations on the synthetic tiny checkpoint in `tests/common`:
//! streaming, the context bound, cancellation, and the loaded model's
//! identity.

mod common;

use brain::chat::{ChatDelta, FinishReason, ToolCall, ToolChoice, ToolSchema};
use brain::{CancelToken, ChatMessage, ChatPipeline, ChatRequest};
use common::{scratch_path, tiny_qwen3_checkpoint, tiny_tokenizer, Scratch, PROMPT_WITHIN_VOCAB, VOCAB_LETTERS};

fn weather_tool() -> ToolSchema {
    ToolSchema::new(
        "get_weather",
        "Current weather for a city",
        serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )
}

fn time_tool() -> ToolSchema {
    ToolSchema::new("get_time", "Current time", serde_json::json!({"type": "object", "properties": {}}))
}

/// A full agent exchange: a system turn, a question, the assistant's tool
/// call, the tool's result, and a follow-up.
fn agent_conversation() -> Vec<ChatMessage> {
    vec![
        ChatMessage::system("You are terse."),
        ChatMessage::user("Weather in Paris?"),
        ChatMessage::assistant("").with_tool_calls(vec![ToolCall { id: "call_0".into(), name: "get_weather".into(), arguments: r#"{"city": "Paris"}"#.into() }]),
        ChatMessage::tool("call_0", "22C and sunny"),
        ChatMessage::user("And tomorrow?"),
    ]
}

/// The OpenAI wire shape of [`agent_conversation`], as the served
/// `/v1/chat/completions` endpoint hands it to the chat renderer.
fn agent_conversation_wire() -> serde_json::Value {
    serde_json::json!([
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "call_0", "content": "22C and sunny"},
        {"role": "user", "content": "And tomorrow?"}
    ])
}

// ---- request -> prompt, no weights ----

/// A typed request renders to exactly the prompt the chat endpoint renders
/// for the same conversation sent over the wire - one renderer, two front
/// doors - and that prompt carries the tool schemas, the earlier call, its
/// result and the thinking-off generation prompt.
#[test]
fn a_typed_conversation_renders_to_the_prompt_the_chat_endpoint_renders() {
    let request = ChatRequest::new(agent_conversation()).tools(vec![weather_tool(), time_tool()]).thinking(false);
    let prompt = request.render_prompt().expect("a well-formed request renders");

    let wire_tools = serde_json::json!([
        {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city", "parameters": weather_tool().parameters}},
        {"type": "function", "function": {"name": "get_time", "description": "Current time", "parameters": time_tool().parameters}}
    ]);
    let wire = capability::Invocation::new()
        .set("messages", serde_json::json!(agent_conversation_wire().to_string()))
        .set("tools", serde_json::json!(wire_tools.to_string()))
        .set("enable_thinking", serde_json::json!(false));
    let served = qwen3::chat::render_prompt(&wire).expect("the wire request renders").text;
    assert_eq!(prompt, served, "the SDK and the served endpoint must render one conversation identically");

    assert!(prompt.starts_with("<|im_start|>system\nYou are terse."), "{prompt}");
    assert!(prompt.contains("<tools>") && prompt.contains(r#""name": "get_weather""#) && prompt.contains(r#""name": "get_time""#), "{prompt}");
    assert!(prompt.contains("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"), "{prompt}");
    assert!(prompt.contains("<tool_response>\n22C and sunny\n</tool_response>"), "{prompt}");
    assert!(prompt.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "thinking off closes an empty think block: {prompt}");
}

/// Thinking left at its default is on: the generation prompt stays open for
/// the model to reason in.
#[test]
fn thinking_is_on_unless_turned_off() {
    let prompt = ChatRequest::new(vec![ChatMessage::user("hi")]).render_prompt().unwrap();
    assert!(prompt.ends_with("<|im_start|>assistant\n"), "{prompt}");
}

/// `tool_choice` none withholds the schemas from the prompt; a named choice
/// must name an offered tool, and says which one it could not find.
#[test]
fn tool_choice_none_withholds_the_tools_and_a_named_choice_must_be_offered() {
    let none = ChatRequest::new(vec![ChatMessage::user("hi")]).tools(vec![weather_tool()]).tool_choice(ToolChoice::None).render_prompt().unwrap();
    assert!(!none.contains("<tools>") && !none.contains("get_weather"), "{none}");

    let err = ChatRequest::new(vec![ChatMessage::user("hi")]).tools(vec![weather_tool()]).tool_choice(ToolChoice::Named("no_such_tool".into())).render_prompt().unwrap_err();
    assert!(err.to_string().contains("no_such_tool"), "{err}");
}

/// A request without a single message is a caller error, named before any
/// template runs.
#[test]
fn a_request_with_no_messages_is_a_missing_argument() {
    match ChatRequest::new(Vec::new()).render_prompt() {
        Err(brain::Error::MissingArgument(msg)) => assert!(msg.contains("message"), "{msg}"),
        other => panic!("expected Error::MissingArgument, got {other:?}"),
    }
}

/// Only an assistant turn makes tool calls or reasons: on any other turn the
/// request is refused, never rendered with them silently dropped.
#[test]
fn tool_calls_or_reasoning_on_a_non_assistant_turn_are_refused() {
    let call = ToolCall { id: "call_0".into(), name: "get_weather".into(), arguments: "{}".into() };
    for message in [ChatMessage::user("hi").with_tool_calls(vec![call]), ChatMessage::tool("call_0", "22C").with_reasoning("hmm")] {
        let err = ChatRequest::new(vec![message]).render_prompt().unwrap_err();
        assert!(err.to_string().contains("only an assistant turn"), "{err}");
    }
}

// ---- raw completion -> response, no weights ----

/// A completion carrying a tool call parses into a typed call - name, the
/// id a streamed call would also carry, and its arguments as the model wrote
/// them - with `tool_calls` as the reason. A parsed reply was not generated
/// here, so nothing was counted: usage is unmeasured, never zero.
#[test]
fn a_tool_call_in_a_reply_parses_into_a_typed_call() {
    let request = ChatRequest::new(vec![ChatMessage::user("Weather in Paris?")]).tools(vec![weather_tool()]);
    let reply = request.parse_reply("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>").unwrap();

    assert_eq!(reply.finish_reason, FinishReason::ToolCalls);
    assert_eq!(reply.tool_calls.len(), 1, "{reply:?}");
    let call = &reply.tool_calls[0];
    assert_eq!((call.id.as_str(), call.name.as_str()), ("call_0", "get_weather"));
    let args: serde_json::Value = serde_json::from_str(&call.arguments).expect("arguments are JSON text");
    assert_eq!(args["city"], "Paris");
    assert_eq!(reply.text.trim(), "");
    assert_eq!(reply.usage.prompt_tokens, None);
    assert_eq!(reply.usage.completion_tokens, None);
}

/// Reasoning is reported apart from the answer, and never leaks into it.
#[test]
fn reasoning_is_reported_apart_from_the_answer() {
    let reply = ChatRequest::new(vec![ChatMessage::user("Weather?")]).parse_reply("<think>\nweighing it\n</think>\n\nIt is sunny.").unwrap();
    assert_eq!(reply.finish_reason, FinishReason::Stop);
    assert_eq!(reply.text.trim(), "It is sunny.");
    assert_eq!(reply.reasoning.trim(), "weighing it");
    assert!(reply.tool_calls.is_empty());
}

/// A demanded tool call the model did not make is reported as such rather
/// than passed off as an answer.
#[test]
fn an_unmet_tool_demand_is_reported() {
    let request = ChatRequest::new(vec![ChatMessage::user("Weather?")]).tools(vec![weather_tool()]).tool_choice(ToolChoice::Required);
    assert_eq!(request.parse_reply("It is sunny.").unwrap().finish_reason, FinishReason::ToolChoiceUnmet);
}

// ---- real generations on the synthetic checkpoint ----

/// The synthetic checkpoint and tokenizer, loaded as a chat pipeline through
/// the text pipeline's own builder. The returned scratch files must outlive
/// any use of their paths.
fn tiny_chat(tag: &str, capacity: Option<u32>) -> (ChatPipeline, Scratch, Scratch) {
    let ckpt = tiny_qwen3_checkpoint(&format!("chat-{tag}"));
    let tok = tiny_tokenizer(&format!("chat-{tag}"));
    let mut builder = brain::TextGenerationPipeline::builder(ckpt.to_str().unwrap()).tokenizer(tok.to_str().unwrap());
    if let Some(capacity) = capacity {
        builder = builder.capacity(capacity);
    }
    let pipe = ChatPipeline::from(builder.load().expect("the synthetic checkpoint builds"));
    (pipe, ckpt, tok)
}

fn within_vocab(text: &str) -> bool {
    text.chars().all(|c| VOCAB_LETTERS.clone().any(|b| b as char == c))
}

/// Streamed deltas add up to exactly the returned text; every count was
/// measured; a budget the fixture has no stop token to end early runs out
/// as `length`; and a seeded request decodes the same way twice.
#[test]
fn a_chat_turn_streams_exactly_the_text_it_returns() {
    let (pipe, _ckpt, _tok) = tiny_chat("stream", None);
    let request = ChatRequest::new(vec![ChatMessage::user(PROMPT_WITHIN_VOCAB)]).max_tokens(4).thinking(false).seed(7).temperature(0.8);

    let mut streamed = String::new();
    let reply = pipe
        .generate_stream(&request, &CancelToken::armed(), |delta| {
            if let ChatDelta::Text(text) = delta {
                streamed.push_str(&text);
            }
        })
        .expect("a real generation");

    assert_eq!(streamed, reply.text, "the stream and the response must be the same text");
    assert!(within_vocab(&reply.text), "{:?}", reply.text);
    assert_eq!(reply.finish_reason, FinishReason::Length);
    assert_eq!(reply.usage.completion_tokens, Some(4));
    assert!(reply.usage.prompt_tokens.is_some_and(|n| n > 0), "{:?}", reply.usage);

    let again = pipe.generate(&request).unwrap();
    assert_eq!(again.text, reply.text, "one seed, one decode");
}

/// A token budget is an upper bound, and so is the context: a request whose
/// budget does not fit what the prompt leaves generates until the context is
/// full and says `length`. A prompt that fills the context alone is an
/// error naming the capacity.
#[test]
fn the_context_bounds_the_generation_and_the_prompt() {
    let request = ChatRequest::new(vec![ChatMessage::user(PROMPT_WITHIN_VOCAB)]).max_tokens(1).thinking(false);
    let prompt_tokens = {
        let (pipe, _ckpt, _tok) = tiny_chat("measure", None);
        pipe.generate(&request).unwrap().usage.prompt_tokens.expect("a generation counts its prompt")
    };

    let (pipe, _ckpt, _tok) = tiny_chat("bounded", Some(prompt_tokens + 3));
    let reply = pipe.generate(&request.clone().max_tokens(100)).unwrap();
    assert_eq!(reply.usage.completion_tokens, Some(3), "{reply:?}");
    assert_eq!(reply.finish_reason, FinishReason::Length);

    let (pipe, _ckpt, _tok) = tiny_chat("overfull", Some(prompt_tokens));
    let err = pipe.generate(&request).unwrap_err();
    assert!(err.to_string().contains("capacity"), "{err}");
}

/// A token fired before the call stops the generation in prefill: nothing is
/// generated, and the reply says it was cancelled rather than finished.
#[test]
fn cancellation_before_the_first_token_reports_cancelled() {
    let (pipe, _ckpt, _tok) = tiny_chat("cancel-early", None);
    let cancel = CancelToken::armed();
    cancel.cancel();
    let reply = pipe.generate_stream(&ChatRequest::new(vec![ChatMessage::user(PROMPT_WITHIN_VOCAB)]).max_tokens(8), &cancel, |_| {}).unwrap();
    assert_eq!(reply.finish_reason, FinishReason::Cancelled);
    assert_eq!(reply.usage.completion_tokens, Some(0));
    assert_eq!(reply.text, "");
}

/// A token fired from the stream stops the decode at the next token, and the
/// partial reply is marked cancelled.
#[test]
fn cancellation_mid_decode_stops_at_the_next_token() {
    let (pipe, _ckpt, _tok) = tiny_chat("cancel-mid", None);
    let cancel = CancelToken::armed();
    let request = ChatRequest::new(vec![ChatMessage::user(PROMPT_WITHIN_VOCAB)]).max_tokens(8).thinking(false);
    let reply = pipe
        .generate_stream(&request, &cancel, |delta| {
            if let ChatDelta::Text(_) = delta {
                cancel.cancel();
            }
        })
        .unwrap();
    assert_eq!(reply.finish_reason, FinishReason::Cancelled);
    assert_eq!(reply.usage.completion_tokens, Some(1), "{reply:?}");
}

/// `sha256:<hex>` of a file's bytes, computed independently of the SDK.
fn file_digest(path: &std::path::Path) -> String {
    use sha2::Digest;
    let bytes = std::fs::read(path).unwrap();
    format!("sha256:{}", sha2::Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect::<String>())
}

/// A rank-1 LoRA adapter over one of the tiny model's projections, carrying
/// its own card id.
fn tiny_adapter(tag: &str, card_id: &str) -> Scratch {
    let (target, numel) = qwen3::QwenConfig::tiny().param_list().into_iter().find(|(name, _)| name.ends_with("attn.wq.weight")).expect("the tiny config has a query projection");
    let d_model = qwen3::QwenConfig::tiny().d_model as usize;
    let a: Vec<f32> = (0..d_model).map(|i| i as f32 * 0.001).collect();
    let b: Vec<f32> = (0..numel / d_model).map(|i| i as f32 * 0.002).collect();
    let mut card = checkpoint::st::ModelCard::new(card_id, "qwen");
    card.adapter = Some(checkpoint::st::Adapter { kind: "lora".into(), rank: Some(1), alpha: Some(1.0), ..Default::default() });
    let path = scratch_path(&format!("chat-{tag}"), "adapter.safetensors");
    checkpoint::st::save_safetensors(
        path.to_str().unwrap(),
        &[(format!("{target}.lora_a"), vec![1, d_model as u64], a), (format!("{target}.lora_b"), vec![(numel / d_model) as u64, 1], b)],
        &serde_json::json!({}),
        Some(&card),
    )
    .unwrap();
    Scratch(path)
}

/// The identity names what was loaded, by content: the base checkpoint's
/// path and digest, and - only when one is attached - the adapter's card id
/// and digest.
#[test]
fn the_identity_names_the_loaded_base_and_adapter_by_digest() {
    let (pipe, ckpt, _tok) = tiny_chat("identity", None);
    let identity = pipe.identity();
    assert_eq!(identity.base.path, ckpt.to_path_buf());
    assert_eq!(identity.base.digest, file_digest(&ckpt));
    assert!(identity.adapter.is_none(), "{identity:?}");

    let ckpt = tiny_qwen3_checkpoint("chat-identity-adapter");
    let tok = tiny_tokenizer("chat-identity-adapter");
    let adapter = tiny_adapter("identity-adapter", "acme/tiny-adapter");
    let pipe = ChatPipeline::from(brain::TextGenerationPipeline::builder(ckpt.to_str().unwrap()).tokenizer(tok.to_str().unwrap()).adapter(adapter.to_str().unwrap()).load().expect("the base with its adapter builds"));
    let identity = pipe.identity();
    assert_eq!(identity.base.digest, file_digest(&ckpt));
    let attached = identity.adapter.as_ref().expect("an attached adapter is part of the identity");
    assert_eq!(attached.id.as_deref(), Some("acme/tiny-adapter"));
    assert_eq!(attached.path, adapter.to_path_buf());
    assert_eq!(attached.digest, file_digest(&adapter));
    assert_ne!(attached.digest, identity.base.digest);
}

/// A pipeline can be moved onto the thread that runs its generations.
#[test]
fn a_pipeline_can_move_to_a_generation_thread() {
    fn assert_send<T: Send>() {}
    assert_send::<ChatPipeline>();
    assert_send::<CancelToken>();
}
