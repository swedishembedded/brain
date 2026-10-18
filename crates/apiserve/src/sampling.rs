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

/// Set `temp`, `top_p`, `top_k` and `seed` on `inv` from `body`. `temperature`
/// and `top_p` default to 1.0, the default of both the OpenAI and the
/// Anthropic API; `top_k` and `seed` are forwarded only when present.
pub fn apply(provider: Provider, body: &Value, mut inv: Invocation) -> Result<Invocation, ApiError> {
    let in_range = |name: &str, max: f64| -> Result<f64, ApiError> {
        match present(body, name) {
            None => Ok(1.0),
            Some(v) => v.as_f64().filter(|x| (0.0..=max).contains(x)).ok_or_else(|| ApiError::invalid_request(provider, format!("'{name}' must be a number in 0..={max}"))),
        }
    };
    inv = inv.set("temp", json!(in_range("temperature", MAX_TEMPERATURE)?)).set("top_p", json!(in_range("top_p", 1.0)?));
    if let Some(v) = present(body, "top_k") {
        let k = v.as_i64().filter(|k| (0..=MAX_TOP_K).contains(k));
        let k = k.ok_or_else(|| ApiError::invalid_request(provider, format!("'top_k' must be an integer in 0..={MAX_TOP_K}")))?;
        inv = inv.set("top_k", json!(k));
    }
    if let Some(v) = present(body, "seed") {
        // A seed the model drew itself spans all of u64 and is echoed back as
        // such; sending it back must reproduce the decode.
        let seed = v.as_u64().ok_or_else(|| ApiError::invalid_request(provider, "'seed' must be a non-negative integer"))?;
        inv = inv.set("seed", json!(seed));
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
    for name in ["presence_penalty", "frequency_penalty"] {
        if present(body, name).is_some_and(|v| v.as_f64() != Some(0.0)) {
            return refuse(name);
        }
    }
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
    fn an_omitted_seed_or_top_k_is_left_to_the_action() {
        let inv = apply(P, &json!({}), Invocation::new()).unwrap();
        assert_eq!(inv.get_i64("seed"), None);
        assert_eq!(inv.get_i64("top_k"), None);
        assert_eq!(inv.get_f64("temp"), Some(1.0));

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

    #[test]
    fn unsupported_openai_parameters_are_refused_unless_neutral() {
        let neutral = json!({"presence_penalty": 0, "frequency_penalty": 0.0, "logit_bias": {}, "logprobs": false,
            "top_logprobs": 0, "response_format": {"type": "text"}});
        refuse_unsupported_openai(P, &neutral, false).unwrap();
        for (body, name) in [
            (json!({"presence_penalty": 0.5}), "presence_penalty"),
            (json!({"frequency_penalty": -1}), "frequency_penalty"),
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
