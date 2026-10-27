// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The sampling parameters every chat surface forwards to `generate`.
//!
//! A parameter the client did not send is not sent on: the action applies its
//! own default, and for `seed` that default is a fresh random seed. Filling in
//! a constant here would make every unseeded request decode the same sequence.
//! A parameter brain cannot honour is refused by name rather than ignored.

use capability::Invocation;
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::surface::Provider;

/// Upper bound on `top_k`, the range the `generate` actions declare.
const MAX_TOP_K: i64 = 1000;

/// Upper bound on `temperature`, the range the `generate` actions declare
/// (and OpenAI's).
const MAX_TEMPERATURE: f64 = 2.0;

/// Set `temp`, `top_p`, `top_k` and `seed` on `inv` from `body`, each only when
/// the client sent it: the action's default stands otherwise, which for a model
/// that declares its own sampling defaults (its `generation_config.json` or GGUF
/// metadata) is the setting its authors recommend, else the chat fallback.
pub fn apply(provider: Provider, body: &Value, mut inv: Invocation) -> Result<Invocation, ApiError> {
    let in_range = |name: &str, max: f64| -> Result<Option<f64>, ApiError> {
        present(body, name)
            .map(|v| v.as_f64().filter(|x| (0.0..=max).contains(x)).ok_or_else(|| ApiError::invalid_request(provider, format!("'{name}' must be a number in 0..={max}"))))
            .transpose()
    };
    if let Some(t) = in_range("temperature", MAX_TEMPERATURE)? {
        inv = inv.set("temp", json!(t));
    }
    if let Some(p) = in_range("top_p", 1.0)? {
        inv = inv.set("top_p", json!(p));
    }
    if let Some(v) = present(body, "top_k") {
        let k = v.as_i64().filter(|k| (0..=MAX_TOP_K).contains(k));
        let k = k.ok_or_else(|| ApiError::invalid_request(provider, format!("'top_k' must be an integer in 0..={MAX_TOP_K}")))?;
        inv = inv.set("top_k", json!(k));
    }
    inv = apply_extended(provider, body, inv)?;
    // vLLM's `ignore_eos`: generate to the token budget past the model's end-of-sequence tokens (a throughput
    // benchmark's protocol). Forwarded only when on, so a model that does not honour it refuses it by name.
    if let Some(v) = present(body, "ignore_eos") {
        let on = v.as_bool().ok_or_else(|| ApiError::invalid_request(provider, "'ignore_eos' must be a boolean"))?;
        if on {
            inv = inv.set("ignore_eos", json!(true));
        }
    }
    if let Some(v) = present(body, "seed") {
        // A seed the model drew itself spans all of u64 and is echoed back as
        // such; sending it back must reproduce the decode.
        let seed = v.as_u64().ok_or_else(|| ApiError::invalid_request(provider, "'seed' must be a non-negative integer"))?;
        inv = inv.set("seed", json!(seed));
    }
    Ok(inv)
}

/// The sampler fields beyond temperature, top-k and top-p, forwarded only when
/// the client sent them (and, for the penalties, only when not neutral): the
/// OpenAI `presence_penalty` and `frequency_penalty`, vLLM's `min_p` and
/// `repetition_penalty`, llama.cpp's DRY (`dry_multiplier`, `dry_base`,
/// `dry_allowed_length`, `dry_sequence_breakers`), and the thinking budget under
/// the names servers use for it (`thinking_budget`, `thinking_token_budget`,
/// `max_thinking_tokens`). A model that has no such sampler refuses the field by
/// name, as it does any parameter it does not declare.
fn apply_extended(provider: Provider, body: &Value, mut inv: Invocation) -> Result<Invocation, ApiError> {
    let bad = |name: &str, what: &str| ApiError::invalid_request(provider, format!("'{name}' must be {what}"));
    let number = |name: &str, lo: f64, hi: f64| -> Result<Option<f64>, ApiError> {
        match present(body, name) {
            None => Ok(None),
            Some(v) => v.as_f64().filter(|x| (lo..=hi).contains(x)).map(Some).ok_or_else(|| bad(name, &format!("a number in {lo}..={hi}"))),
        }
    };
    for (name, lo, hi, neutral) in [("presence_penalty", -2.0, 2.0, 0.0), ("frequency_penalty", -2.0, 2.0, 0.0), ("min_p", 0.0, 1.0, 0.0), ("repetition_penalty", 0.01, 4.0, 1.0), ("dry_multiplier", 0.0, 100.0, 0.0)] {
        if let Some(x) = number(name, lo, hi)? {
            if x != neutral {
                inv = inv.set(name, json!(x));
            }
        }
    }
    if let Some(x) = number("dry_base", 1.0, 16.0)? {
        inv = inv.set("dry_base", json!(x));
    }
    if let Some(v) = present(body, "dry_allowed_length") {
        let n = v.as_i64().filter(|n| (0..=4096).contains(n)).ok_or_else(|| bad("dry_allowed_length", "an integer in 0..=4096"))?;
        inv = inv.set("dry_allowed_length", json!(n));
    }
    if let Some(v) = present(body, "dry_sequence_breakers") {
        let ok = v.as_array().filter(|a| a.iter().all(Value::is_string));
        let breakers = ok.ok_or_else(|| bad("dry_sequence_breakers", "an array of strings"))?;
        inv = inv.set("dry_sequence_breakers", json!(serde_json::to_string(breakers).unwrap_or_default()));
    }
    for name in ["thinking_budget", "thinking_token_budget", "max_thinking_tokens"] {
        if let Some(v) = present(body, name) {
            let n = v.as_i64().filter(|n| *n >= 0).ok_or_else(|| bad(name, "a non-negative integer"))?;
            inv = inv.set("thinking_budget", json!(n));
            break;
        }
    }
    Ok(inv)
}

/// Ask the model for the sampled tokens' ids and log-probabilities when the
/// client sent `logprobs: true` (the log-probability of each sampled token) or
/// `return_token_ids: true` (the prompt's and the completion's token ids; the
/// extension vLLM names the same way). Neither is forwarded unless sent, so a
/// request that does not read them pays nothing.
pub fn apply_token_capture(provider: Provider, body: &Value, mut inv: Invocation) -> Result<Invocation, ApiError> {
    for name in ["logprobs", "return_token_ids"] {
        match present(body, name) {
            None => {}
            Some(v) => {
                let on = v.as_bool().ok_or_else(|| ApiError::invalid_request(provider, format!("'{name}' must be a boolean")))?;
                if on {
                    inv = inv.set(name, json!(true));
                }
            }
        }
    }
    Ok(inv)
}

/// The completion budget a client sent as `name`: a positive integer. There
/// is no upper bound here - whether the prompt plus this many tokens fits is
/// the serving model's context to decide, and it answers with
/// `context_length_exceeded`.
pub fn max_tokens(provider: Provider, name: &str, v: Option<&Value>) -> Result<i64, ApiError> {
    let v = v.filter(|v| !v.is_null()).ok_or_else(|| ApiError::invalid_request(provider, format!("'{name}' is required")))?;
    v.as_i64().filter(|&n| n >= 1).ok_or_else(|| ApiError::invalid_request(provider, format!("'{name}' must be a positive integer")))
}

/// Refuse an OpenAI parameter brain cannot honour when it asks for anything
/// but its neutral value. Clients send the neutral values by default
/// (`presence_penalty: 0`, `logprobs: false`), and those are accepted.
///
/// `token_capture` says the endpoint can return the sampled tokens' ids and
/// log-probabilities (see [`apply_token_capture`]): there `logprobs: true` is
/// honoured instead of refused. `top_logprobs` stays refused either way: only
/// the sampled token's own log-probability is available, not the alternatives.
pub fn refuse_unsupported_openai(provider: Provider, body: &Value, token_capture: bool) -> Result<(), ApiError> {
    let refuse = |name: &str| Err(ApiError::invalid_request(provider, format!("'{name}' is not supported")));
    if present(body, "logit_bias").is_some_and(|v| v.as_object().is_none_or(|m| !m.is_empty())) {
        return refuse("logit_bias");
    }
    if present(body, "logprobs").is_some_and(|v| v.as_bool() != Some(false) && !(token_capture && v.as_bool() == Some(true))) {
        return refuse("logprobs");
    }
    if present(body, "top_logprobs").is_some_and(|v| v.as_i64() != Some(0)) {
        return refuse("top_logprobs");
    }
    if present(body, "response_format").is_some_and(|v| v.get("type").and_then(Value::as_str) != Some("text")) {
        return refuse("response_format");
    }
    Ok(())
}

/// `body[key]`, treating an explicit `null` as absent.
fn present<'a>(body: &'a Value, key: &str) -> Option<&'a Value> {
    body.get(key).filter(|v| !v.is_null())
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: Provider = Provider::OpenAI;

    #[test]
    fn ignore_eos_is_forwarded_only_when_on_and_must_be_a_boolean() {
        assert_eq!(apply(P, &json!({}), Invocation::new()).unwrap().get_bool("ignore_eos"), None);
        assert_eq!(apply(P, &json!({"ignore_eos": false}), Invocation::new()).unwrap().get_bool("ignore_eos"), None);
        assert_eq!(apply(P, &json!({"ignore_eos": true}), Invocation::new()).unwrap().get_bool("ignore_eos"), Some(true));
        assert!(apply(P, &json!({"ignore_eos": "yes"}), Invocation::new()).is_err());
    }

    #[test]
    fn an_omitted_sampling_setting_is_left_to_the_action() {
        let inv = apply(P, &json!({}), Invocation::new()).unwrap();
        assert_eq!(inv.get_i64("seed"), None);
        assert_eq!(inv.get_i64("top_k"), None);
        assert_eq!((inv.get_f64("temp"), inv.get_f64("top_p")), (None, None), "the checkpoint's declared defaults apply, not a constant");

        let inv = apply(P, &json!({"seed": 7, "top_k": 3, "temperature": 0.2}), Invocation::new()).unwrap();
        assert_eq!((inv.get_i64("seed"), inv.get_i64("top_k"), inv.get_f64("temp")), (Some(7), Some(3), Some(0.2)));
    }

    /// A seed is any 64-bit unsigned integer, the range the actions draw their
    /// own seeds from, so a seed brain echoed can always be sent back.
    #[test]
    fn a_seed_is_any_unsigned_64_bit_integer_and_a_negative_one_is_refused_by_name() {
        let inv = apply(P, &json!({"seed": u64::MAX}), Invocation::new()).unwrap();
        assert_eq!(inv.seed(), Ok(Some(u64::MAX)));
        for bad in [json!(-1), json!(1.5)] {
            let e = apply(P, &json!({"seed": bad}), Invocation::new()).unwrap_err();
            assert!(format!("{e:?}").contains("'seed' must be a non-negative integer"), "{bad}: {e:?}");
        }
    }

    #[test]
    fn a_malformed_seed_or_top_k_is_refused() {
        for body in [json!({"seed": "x"}), json!({"top_k": -1}), json!({"top_k": 1001}), json!({"top_k": 2.5})] {
            assert!(apply(P, &body, Invocation::new()).is_err(), "{body}");
        }
    }

    /// `generate` refuses these outside its declared ranges; the surface
    /// refuses them first, naming the client's own field.
    #[test]
    fn an_out_of_range_temperature_top_p_or_max_tokens_is_refused_by_name() {
        for (body, name) in [
            (json!({"temperature": -0.1}), "temperature"),
            (json!({"temperature": 2.5}), "temperature"),
            (json!({"temperature": "hot"}), "temperature"),
            (json!({"top_p": 1.5}), "top_p"),
            (json!({"top_p": -0.5}), "top_p"),
        ] {
            let e = apply(P, &body, Invocation::new()).unwrap_err();
            assert!(format!("{e:?}").contains(name), "{body}: {e:?}");
        }
        let ok = apply(P, &json!({"temperature": 2.0, "top_p": 0.0}), Invocation::new()).unwrap();
        assert_eq!((ok.get_f64("temp"), ok.get_f64("top_p")), (Some(2.0), Some(0.0)));

        for bad in [json!(0), json!(-5), json!(1.5), json!("many")] {
            let e = max_tokens(P, "max_tokens", Some(&bad)).unwrap_err();
            assert!(format!("{e:?}").contains("max_tokens"), "{bad}: {e:?}");
        }
        // No upper bound: how much the model can produce is its context's to say.
        assert_eq!(max_tokens(P, "max_tokens", Some(&json!(1_000_000))).unwrap(), 1_000_000);
    }

    #[test]
    fn token_capture_is_forwarded_only_when_asked_and_logprobs_are_honoured_where_it_is() {
        let body = json!({"logprobs": true, "return_token_ids": false});
        refuse_unsupported_openai(P, &body, true).unwrap();
        let inv = apply_token_capture(P, &body, Invocation::new()).unwrap();
        assert_eq!((inv.get_bool("logprobs"), inv.get_bool("return_token_ids")), (Some(true), None));
        assert!(refuse_unsupported_openai(P, &body, false).is_err());
        assert!(refuse_unsupported_openai(P, &json!({"top_logprobs": 3}), true).is_err());
        assert!(apply_token_capture(P, &json!({"return_token_ids": "yes"}), Invocation::new()).is_err());
    }

    /// The extended sampler fields reach the action only when the client asked for
    /// something: the neutral values every client sends by default are dropped, so
    /// a request that asks for nothing is the request it always was.
    #[test]
    fn extended_sampler_fields_are_forwarded_only_when_asked_for() {
        let neutral = json!({"presence_penalty": 0, "frequency_penalty": 0.0, "min_p": 0, "repetition_penalty": 1.0, "dry_multiplier": 0});
        let inv = apply(P, &neutral, Invocation::new()).unwrap();
        for k in ["presence_penalty", "frequency_penalty", "min_p", "repetition_penalty", "dry_multiplier", "thinking_budget"] {
            assert_eq!(inv.get_f64(k), None, "{k}");
        }
        let asked = json!({"presence_penalty": 0.5, "frequency_penalty": -0.25, "min_p": 0.05, "repetition_penalty": 1.1,
            "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2, "dry_sequence_breakers": ["\n", ":"], "max_thinking_tokens": 512});
        let inv = apply(P, &asked, Invocation::new()).unwrap();
        assert_eq!((inv.get_f64("presence_penalty"), inv.get_f64("frequency_penalty"), inv.get_f64("min_p"), inv.get_f64("repetition_penalty")), (Some(0.5), Some(-0.25), Some(0.05), Some(1.1)));
        assert_eq!((inv.get_f64("dry_multiplier"), inv.get_f64("dry_base"), inv.get_i64("dry_allowed_length")), (Some(0.8), Some(1.75), Some(2)));
        assert_eq!(inv.get_str("dry_sequence_breakers").as_deref(), Some("[\"\\n\",\":\"]"));
        assert_eq!(inv.get_i64("thinking_budget"), Some(512), "every name servers use for it lands on one field");
        for name in ["thinking_budget", "thinking_token_budget"] {
            assert_eq!(apply(P, &json!({ name: 7 }), Invocation::new()).unwrap().get_i64("thinking_budget"), Some(7));
        }
    }

    #[test]
    fn out_of_range_extended_sampler_fields_are_refused_by_name() {
        for (body, name) in [
            (json!({"presence_penalty": 3}), "presence_penalty"),
            (json!({"min_p": 1.5}), "min_p"),
            (json!({"repetition_penalty": 0}), "repetition_penalty"),
            (json!({"dry_base": 0.5}), "dry_base"),
            (json!({"dry_allowed_length": -1}), "dry_allowed_length"),
            (json!({"dry_sequence_breakers": [1]}), "dry_sequence_breakers"),
            (json!({"thinking_budget": -1}), "thinking_budget"),
            (json!({"max_thinking_tokens": "lots"}), "max_thinking_tokens"),
        ] {
            let e = apply(P, &body, Invocation::new()).unwrap_err();
            assert!(format!("{e:?}").contains(name), "{body}: {e:?}");
        }
    }

    #[test]
    fn unsupported_openai_parameters_are_refused_unless_neutral() {
        let neutral = json!({"presence_penalty": 0, "frequency_penalty": 0.0, "logit_bias": {}, "logprobs": false,
            "top_logprobs": 0, "response_format": {"type": "text"}});
        refuse_unsupported_openai(P, &neutral, false).unwrap();
        for (body, name) in [
            (json!({"logit_bias": {"50256": -100}}), "logit_bias"),
            (json!({"logprobs": true}), "logprobs"),
            (json!({"top_logprobs": 5}), "top_logprobs"),
            (json!({"response_format": {"type": "json_object"}}), "response_format"),
        ] {
            let e = refuse_unsupported_openai(P, &body, false).unwrap_err();
            assert!(format!("{e:?}").contains(name), "{body}: {e:?}");
        }
    }
}
