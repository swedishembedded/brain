// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving language models behind
// the OpenAI and Anthropic APIs without silently changing what a client asked
// for. If your team needs expertise in API compatibility layers for inference
// servers then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The top-level fields a chat request may carry, per dialect.
//!
//! A chat route reads some fields (each parser lists them: [`Dialect::reads`]),
//! accepts a few more and ignores them ([`IGNORED`], each with the reason the
//! dialect's own documentation gives for ignoring it being correct), and
//! refuses every other field with an `unsupported_parameter` naming the field
//! and the model. A field brain dropped would answer a different request than
//! the one sent, and the client would never learn it.
//!
//! Which params the model itself takes is a second, separate check: the
//! invocation a route builds is validated against the model's served action
//! (`crate::bridge::admit`).

use serde_json::Value;

use crate::error::ApiError;
use crate::surface::Provider;

/// A chat request shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dialect {
    /// OpenAI `POST /chat/completions`.
    OpenAiChat,
    /// OpenRouter `POST /chat/completions`: OpenAI's grammar plus OpenRouter's routing fields.
    OpenRouterChat,
    /// OpenAI's legacy `POST /completions`.
    OpenAiCompletion,
    /// Anthropic `POST /v1/messages`.
    AnthropicMessages,
}

/// The fields every chat dialect brain serves reads the same way: the sampler and the thinking budget
/// ([`crate::sampling::apply`]).
fn sampler_fields() -> impl Iterator<Item = &'static str> {
    crate::sampling::request_fields()
}

/// The fields OpenAI's chat grammar reads, OpenRouter's included.
const OPENAI_CHAT_READS: [&str; 20] = [
    "model",
    "messages",
    "stream",
    "stream_options",
    "n",
    "max_tokens",
    "max_completion_tokens",
    "stop",
    "tools",
    "tool_choice",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    // The vLLM/SGLang chat-template extension point, and its members tolerated at the top level.
    "chat_template_kwargs",
    "enable_thinking",
    "reasoning_effort",
    "max_tiles",
    "preserve_thinking",
    "template_flavor",
];

/// The fields the legacy completions grammar reads.
const OPENAI_COMPLETION_READS: [&str; 14] =
    ["model", "prompt", "suffix", "stream", "stream_options", "n", "best_of", "echo", "max_tokens", "max_completion_tokens", "stop", "logit_bias", "logprobs", "top_logprobs"];

/// The fields Anthropic's Messages grammar reads.
const ANTHROPIC_READS: [&str; 9] = ["model", "messages", "max_tokens", "stream", "system", "stop_sequences", "tools", "tool_choice", "thinking"];

impl Dialect {
    /// The dialect of a chat request to `provider`'s chat route.
    pub fn chat(provider: Provider) -> Dialect {
        match provider {
            Provider::OpenAI => Dialect::OpenAiChat,
            Provider::OpenRouter => Dialect::OpenRouterChat,
            Provider::Anthropic => Dialect::AnthropicMessages,
        }
    }

    /// Whether this dialect's parser reads `field`.
    fn reads(self, field: &str) -> bool {
        let own: &[&str] = match self {
            Dialect::OpenAiChat => &OPENAI_CHAT_READS,
            // `models`: the fallback list the model is resolved from (`crate::catalog::resolve_model`).
            Dialect::OpenRouterChat => return field == "models" || Dialect::OpenAiChat.reads(field),
            Dialect::OpenAiCompletion => &OPENAI_COMPLETION_READS,
            Dialect::AnthropicMessages => &ANTHROPIC_READS,
        };
        let token_capture = matches!(self, Dialect::OpenAiChat) && crate::sampling::TOKEN_CAPTURE.contains(&field);
        own.contains(&field) || token_capture || sampler_fields().any(|f| f == field)
    }
}

/// Which values of an ignored field are accepted.
#[derive(Clone, Copy)]
enum Accepts {
    /// Any value: the field never changes the reply.
    Any,
    /// Only these values (as JSON text): the field's default, or values that ask for what brain does anyway. Any
    /// other value asks for something brain does not do, and is refused.
    Only(&'static [&'static str]),
}

/// A field a dialect defines that brain accepts and does not act on.
pub struct Ignored {
    pub name: &'static str,
    dialects: &'static [Dialect],
    accepts: Accepts,
    /// Why ignoring it answers the request the client sent.
    pub reason: &'static str,
}

const OPENAI: &[Dialect] = &[Dialect::OpenAiChat, Dialect::OpenRouterChat];
const OPENAI_ALL: &[Dialect] = &[Dialect::OpenAiChat, Dialect::OpenRouterChat, Dialect::OpenAiCompletion];
const OPENROUTER: &[Dialect] = &[Dialect::OpenRouterChat];
const ANTHROPIC: &[Dialect] = &[Dialect::AnthropicMessages];

/// Every field a chat route accepts and ignores; the HTTP API reference carries the same table.
pub const IGNORED: [Ignored; 13] = [
    Ignored {
        name: "user",
        dialects: OPENAI_ALL,
        accepts: Accepts::Any,
        reason: "an end-user id for the provider's abuse monitoring; it does not change the reply",
    },
    Ignored {
        name: "safety_identifier",
        dialects: OPENAI,
        accepts: Accepts::Any,
        reason: "the successor of `user`: an end-user id for abuse monitoring; it does not change the reply",
    },
    Ignored {
        name: "prompt_cache_key",
        dialects: OPENAI,
        accepts: Accepts::Any,
        reason: "a hint for routing requests to a warm prompt cache; brain's prefix cache matches prompts by their tokens",
    },
    Ignored {
        name: "metadata",
        dialects: &[Dialect::OpenAiChat, Dialect::OpenRouterChat, Dialect::AnthropicMessages],
        accepts: Accepts::Any,
        reason: "tags for the provider's stored-completion dashboards (OpenAI) or an end-user id (Anthropic's `user_id`); brain stores nothing and the reply does not depend on them",
    },
    Ignored {
        name: "store",
        dialects: OPENAI,
        accepts: Accepts::Only(&["false"]),
        reason: "`false` (keep no copy of the completion) is what brain does; `true` asks the server to store it, which brain does not, and is refused",
    },
    Ignored {
        name: "service_tier",
        dialects: OPENAI,
        accepts: Accepts::Only(&["\"auto\"", "\"default\""]),
        reason: "the provider's processing tier; brain has one, which `auto` and `default` name; a request for another tier (`flex`, `priority`) is refused",
    },
    Ignored {
        name: "parallel_tool_calls",
        dialects: OPENAI,
        accepts: Accepts::Only(&["true"]),
        reason: "`true` (the default) lets the model call several tools in one turn, as brain's models may; `false` asks the server to allow one at most, which brain does not enforce, and is refused",
    },
    Ignored {
        name: "modalities",
        dialects: OPENAI,
        accepts: Accepts::Only(&["[\"text\"]"]),
        reason: "the output modalities; `[\"text\"]` is what a chat reply is; audio output is refused",
    },
    Ignored {
        name: "provider",
        dialects: OPENROUTER,
        accepts: Accepts::Any,
        reason: "OpenRouter's preferences for which upstream provider serves the request; brain is the provider",
    },
    Ignored {
        name: "route",
        dialects: OPENROUTER,
        accepts: Accepts::Any,
        reason: "OpenRouter's fallback routing; brain resolves `model`, then `models`, in order, whatever it says",
    },
    Ignored {
        name: "transforms",
        dialects: OPENROUTER,
        accepts: Accepts::Any,
        reason: "OpenRouter's prompt transforms (`middle-out` trims a prompt that does not fit); a prompt that fits is not transformed, and brain refuses one that does not with `context_length_exceeded` rather than trimming it",
    },
    Ignored {
        name: "usage",
        dialects: OPENROUTER,
        accepts: Accepts::Any,
        reason: "OpenRouter's switch for usage accounting in the reply; brain always reports usage",
    },
    Ignored {
        name: "service_tier",
        dialects: ANTHROPIC,
        accepts: Accepts::Only(&["\"auto\"", "\"standard_only\""]),
        reason: "the provider's capacity tier; brain has one, which both values Anthropic defines accept",
    },
];

/// Refuse a field of `body` the `dialect` route neither reads nor accepts-and-ignores ([`IGNORED`]), or an ignored
/// one whose value asks for something brain does not do. An explicit `null` is absent. The refusal names the field and
/// the requested model.
pub fn refuse_unknown(provider: Provider, dialect: Dialect, body: &Value) -> Result<(), ApiError> {
    let Some(fields) = body.as_object() else { return Ok(()) };
    let model = body.get("model").and_then(Value::as_str).unwrap_or_default();
    for (name, value) in fields.iter().filter(|(_, v)| !v.is_null()) {
        if dialect.reads(name) {
            continue;
        }
        match IGNORED.iter().find(|i| i.name == name && i.dialects.contains(&dialect)) {
            None => return Err(ApiError::unsupported_parameter(provider, model, name, "brain does not read this field")),
            Some(Ignored { accepts: Accepts::Any, .. }) => {}
            Some(Ignored { accepts: Accepts::Only(values), .. }) => {
                if !values.iter().any(|v| serde_json::from_str::<Value>(v).is_ok_and(|v| &v == value)) {
                    return Err(ApiError::unsupported_parameter(provider, model, name, &format!("brain accepts only {} here, and ignores it", values.join(" or "))));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const DIALECTS: [Dialect; 4] = [Dialect::OpenAiChat, Dialect::OpenRouterChat, Dialect::OpenAiCompletion, Dialect::AnthropicMessages];

    fn refused_field(dialect: Dialect, body: Value) -> Option<String> {
        refuse_unknown(Provider::OpenAI, dialect, &body).err().map(|e| e.param.unwrap_or_default())
    }

    /// A field no dialect defines is refused by name; one the dialect reads or ignores passes.
    #[test]
    fn an_unknown_field_is_refused_by_name_and_a_read_one_passes() {
        for dialect in DIALECTS {
            assert_eq!(refused_field(dialect, json!({"model": "m", "frobnicate": 1})).as_deref(), Some("frobnicate"), "{dialect:?}");
            assert_eq!(refused_field(dialect, json!({"model": "m", "temperature": 0.5, "stream": true, "frobnicate": null})), None, "{dialect:?}");
        }
        assert_eq!(refused_field(Dialect::AnthropicMessages, json!({"model": "m", "stop": ["x"]})).as_deref(), Some("stop"), "Anthropic's is stop_sequences");
        assert_eq!(refused_field(Dialect::OpenAiChat, json!({"model": "m", "models": ["a"]})).as_deref(), Some("models"), "an OpenRouter field");
        assert_eq!(refused_field(Dialect::OpenRouterChat, json!({"model": "m", "models": ["a"]})), None);
        assert_eq!(refused_field(Dialect::OpenAiChat, json!({"model": "m", "functions": []})).as_deref(), Some("functions"), "the legacy function-calling fields");
    }

    /// Every ignored field passes with each value it accepts in each dialect that defines it, and an ignored field
    /// restricted to some values refuses another.
    #[test]
    fn an_ignored_field_passes_only_with_a_value_it_accepts() {
        for ignored in &IGNORED {
            for &dialect in ignored.dialects {
                let accepted: Vec<Value> = match ignored.accepts {
                    Accepts::Any => vec![json!("anything"), json!({"k": 1})],
                    Accepts::Only(values) => values.iter().map(|v| serde_json::from_str(v).unwrap()).collect(),
                };
                for v in accepted {
                    assert_eq!(refused_field(dialect, json!({"model": "m", ignored.name: v})), None, "{} {v} in {dialect:?}", ignored.name);
                }
                if let Accepts::Only(_) = ignored.accepts {
                    assert_eq!(refused_field(dialect, json!({"model": "m", ignored.name: "something-else"})).as_deref(), Some(ignored.name));
                }
            }
        }
        assert_eq!(refused_field(Dialect::OpenAiCompletion, json!({"model": "m", "store": false})).as_deref(), Some("store"), "a chat-only field");
    }
}
