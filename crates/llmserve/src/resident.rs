// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many language models
// from one process under a memory budget for its clients. If your team needs
// expertise in LLM inference servers then you can procure our services by
// sending an email to info@swedishembedded.com.

//! What every text resident shares outside the drive loop: the `generate`
//! action it declares ([`generate_spec`]). A family's resident, in the catalog
//! or in the model's own crate, calls it rather than keeping its own copy; the
//! device scope its engine is built in is `residency::on_device`.

use capability::{ActionSpec, BlobSpec, Media, ParamSpec, ParamType};
use serde_json::json;

/// The `"generate"` action spec every text resident serves. `chat` adds
/// Qwen's chat contract: the chat-template toggle plus
/// `messages`/`system`/`top_p`/`stop` and per-token streaming (one
/// `Progress::token` delta each accepted token).
pub fn generate_spec(summary: &str, chat: bool) -> ActionSpec {
    let mut s = ActionSpec::new("generate", summary)
        .cancellable()
        .param(ParamSpec::new("prompt", ParamType::Str, "the prompt to continue (or chat message)"))
        .param(ParamSpec::new("max_new", ParamType::Int, "number of new tokens to generate; omitted, the reply runs to its end of turn or to the end of the context").min(0.0).max(capability::bounds::TOKENS))
        .param(ParamSpec::new("temp", ParamType::Float, "sampling temperature (<= 0 = greedy)").max(capability::bounds::TEMPERATURE).default(json!(0.8)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k filter (40 = standard; 1 = greedy; 0 or negative = disabled)").min(0.0).max(capability::bounds::TOP_K).default(json!(40)))
        .param(ParamSpec::new("seed", ParamType::Int, "RNG seed (omit for random)").min(0.0).max(capability::bounds::SEED));
    if chat {
        s = s
            .streaming()
            .param(ParamSpec::new("chat", ParamType::Bool, "apply the chat template to the prompt").default(json!(true)))
            .param(ParamSpec::new(
                "messages",
                ParamType::Str,
                "JSON array of {role,content,reasoning_content?,tool_calls?,tool_call_id?} chat turns (overrides prompt)",
            ))
            .param(ParamSpec::new("system", ParamType::Str, "optional system prompt prepended to the chat"))
            .param(ParamSpec::new(
                "suffix",
                ParamType::Str,
                "fill-in-the-middle: the text after the insertion point, with the raw prompt (chat false) the text before it (needs FIM tokens in the vocabulary)",
            ))
            .param(ParamSpec::new("top_p", ParamType::Float, "nucleus sampling threshold (1 = disabled)").min(0.0).max(capability::bounds::UNIT).default(json!(1.0)))
            .param(ParamSpec::new("stop", ParamType::Str, "JSON array of stop strings"))
            .param(ParamSpec::new("tools", ParamType::Str, "JSON array of tool definitions (OpenAI function-calling schema)"))
            .param(ParamSpec::new("tool_choice", ParamType::Str, "tool_choice directive, raw JSON text (\"auto\"|\"none\"|\"required\"|{...})"))
            .param(ParamSpec::new("enable_thinking", ParamType::Bool, "allow the model to emit a <think> reasoning block").default(json!(true)))
            .params(frontend_params());
    }
    crate::with_token_capture(s.output(BlobSpec::new("text", Media::Text, "the generated text")))
}

/// The params a [`crate::TextFrontend`] reads beyond the prompt, the basic sampler and the chat template: `ignore_eos`
/// (its own), the thinking budget and the extended sampler (the shared `qwen3::chat` request parser's). A resident
/// whose requests go through the front end declares them ([`generate_spec`] does), so a serving surface, which refuses
/// a param the action does not declare, passes them on.
pub fn frontend_params() -> [ParamSpec; 10] {
    [
        ParamSpec::new("ignore_eos", ParamType::Bool, "generate to max_new past the model's end-of-sequence tokens (stop strings still apply)").default(json!(false)),
        ParamSpec::new("thinking_budget", ParamType::Int, "tokens the reply may spend inside think blocks, over every block it opens; when spent the closing tag is forced and the model answers, and after a tool call written inside the block the reply ends instead (default: unbounded)").min(0.0).max(capability::bounds::TOKENS),
        ParamSpec::new("min_p", ParamType::Float, "drop tokens below this fraction of the best token's probability (0 = off)").min(0.0).max(capability::bounds::UNIT),
        ParamSpec::new("presence_penalty", ParamType::Float, "subtracted from the logit of every token already generated (0 = off)").min(-capability::bounds::LOGIT_PENALTY).max(capability::bounds::LOGIT_PENALTY),
        ParamSpec::new("frequency_penalty", ParamType::Float, "subtracted from a token's logit once per time it was generated (0 = off)").min(-capability::bounds::LOGIT_PENALTY).max(capability::bounds::LOGIT_PENALTY),
        ParamSpec::new("repetition_penalty", ParamType::Float, "divides the positive logit of every token in the context, prompt included (1 = off)").min(0.01).max(capability::bounds::REPETITION_PENALTY),
        ParamSpec::new("dry_multiplier", ParamType::Float, "DRY repetition penalty strength over the whole context (0 = off)").min(0.0).max(capability::bounds::DRY_MULTIPLIER),
        ParamSpec::new("dry_base", ParamType::Float, "DRY penalty growth per extra repeated token (default 1.75)").min(1.0).max(capability::bounds::DRY_BASE),
        ParamSpec::new("dry_allowed_length", ParamType::Int, "DRY: repeats up to this many tokens are not penalised (default 2)").min(0.0).max(1024.0),
        ParamSpec::new("dry_sequence_breakers", ParamType::Str, "JSON array of strings that end a DRY sequence (default newline, colon, quote, asterisk)"),
    ]
}
