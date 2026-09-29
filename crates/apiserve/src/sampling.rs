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

/// Set `temp`, `top_p`, `top_k` and `seed` on `inv` from `body`. `temperature`
/// and `top_p` default to 1.0, the default of both the OpenAI and the
/// Anthropic API; `top_k` and `seed` are forwarded only when present.
pub fn apply(provider: Provider, body: &Value, mut inv: Invocation) -> Result<Invocation, ApiError> {
    inv = inv
        .set("temp", json!(body.get("temperature").and_then(Value::as_f64).unwrap_or(1.0)))
        .set("top_p", json!(body.get("top_p").and_then(Value::as_f64).unwrap_or(1.0)));
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
