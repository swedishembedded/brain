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
        let seed = v.as_i64().ok_or_else(|| ApiError::invalid_request(provider, "'seed' must be an integer"))?;
        inv = inv.set("seed", json!(seed));
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
pub fn refuse_unsupported_openai(provider: Provider, body: &Value) -> Result<(), ApiError> {
    let refuse = |name: &str| Err(ApiError::invalid_request(provider, format!("'{name}' is not supported")));
    for name in ["presence_penalty", "frequency_penalty"] {
        if present(body, name).is_some_and(|v| v.as_f64() != Some(0.0)) {
            return refuse(name);
        }
    }
    if present(body, "logit_bias").is_some_and(|v| v.as_object().is_none_or(|m| !m.is_empty())) {
        return refuse("logit_bias");
    }
    if present(body, "logprobs").is_some_and(|v| v.as_bool() != Some(false)) {
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
    fn unsupported_openai_parameters_are_refused_unless_neutral() {
        let neutral = json!({"presence_penalty": 0, "frequency_penalty": 0.0, "logit_bias": {}, "logprobs": false,
            "top_logprobs": 0, "response_format": {"type": "text"}});
        refuse_unsupported_openai(P, &neutral).unwrap();
        for (body, name) in [
            (json!({"presence_penalty": 0.5}), "presence_penalty"),
            (json!({"frequency_penalty": -1}), "frequency_penalty"),
            (json!({"logit_bias": {"50256": -100}}), "logit_bias"),
            (json!({"logprobs": true}), "logprobs"),
            (json!({"top_logprobs": 5}), "top_logprobs"),
            (json!({"response_format": {"type": "json_object"}}), "response_format"),
        ] {
            let e = refuse_unsupported_openai(P, &body).unwrap_err();
            assert!(format!("{e:?}").contains(name), "{body}: {e:?}");
        }
    }
}
