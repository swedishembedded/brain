// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The sampling parameters every chat surface forwards to `generate`.
//!
//! A parameter the client did not send takes the default the model's served
//! `generate` action declares (the validation every queued call passes,
//! `crate::bridge::admit`: for a checkpoint that declares its sampling, the
//! declared setting), and `seed`, which declares none, a fresh random seed.
//! Filling in a constant here would make every unseeded request decode the
//! same sequence. A parameter brain cannot honour is refused by name rather
//! than ignored.

use capability::{bounds, ActionSpec, Invocation};
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::surface::Provider;

/// How a sampling field's value is read and forwarded.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// A number in `lo..=hi`. A `neutral` value, which clients send by default, asks for nothing: [`settle`] drops it
    /// for a model without that sampler, so the request is the one it always was, and keeps it for a model with one,
    /// where it overrides a default the model declares.
    Number { lo: f64, hi: f64, neutral: Option<f64> },
    /// An integer in `lo..=hi`.
    Integer { lo: i64, hi: i64 },
    /// Any 64-bit unsigned integer: a seed the model drew itself spans all of u64 and is echoed back as such, and
    /// sending it back must reproduce the decode.
    Seed,
    /// A boolean, forwarded only when on.
    Switch,
    /// An array of strings, forwarded as its JSON text.
    Strings,
}

/// One request field the sampler reads, and the `generate` param it sets.
struct Field {
    name: &'static str,
    param: &'static str,
    kind: Kind,
}

impl Field {
    const fn new(name: &'static str, param: &'static str, kind: Kind) -> Field {
        Field { name, param, kind }
    }

    /// The param value `v` asks for (`None`: nothing to forward), or what the field must be.
    fn read(&self, v: &Value) -> Result<Option<Value>, String> {
        match self.kind {
            Kind::Number { lo, hi, .. } => {
                let x = v.as_f64().filter(|x| (lo..=hi).contains(x)).ok_or_else(|| format!("a number in {lo}..={hi}"))?;
                Ok(Some(json!(x)))
            }
            Kind::Integer { lo, hi } => {
                let what = if (lo, hi) == (0, i64::MAX) { "a non-negative integer".to_string() } else { format!("an integer in {lo}..={hi}") };
                v.as_i64().filter(|n| (lo..=hi).contains(n)).map(|n| Some(json!(n))).ok_or(what)
            }
            Kind::Seed => v.as_u64().map(|n| Some(json!(n))).ok_or_else(|| "a non-negative integer".to_string()),
            Kind::Switch => v.as_bool().map(|on| on.then_some(json!(true))).ok_or_else(|| "a boolean".to_string()),
            Kind::Strings => {
                let strings = v.as_array().filter(|a| a.iter().all(Value::is_string)).ok_or_else(|| "an array of strings".to_string())?;
                Ok(Some(json!(serde_json::to_string(strings).unwrap_or_default())))
            }
        }
    }
}

/// Upper bound on `top_k`, the range the `generate` actions declare.
const MAX_TOP_K: i64 = bounds::TOP_K as i64;

/// Every sampling field a chat surface forwards, in the order [`honoured`] lists them: the OpenAI `temperature`,
/// `top_p`, `seed`, `presence_penalty` and `frequency_penalty`, `top_k`, vLLM's `min_p`, `repetition_penalty` and
/// `ignore_eos` (generate to the token budget past the end-of-sequence tokens, a throughput benchmark's protocol),
/// llama.cpp's DRY (`dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`), and the thinking
/// budget under each name servers use for it (the first one sent counts). The one table [`apply`] forwards from and
/// [`honoured`] reads.
const FIELDS: [Field; 16] = [
    Field::new("temperature", "temp", Kind::Number { lo: 0.0, hi: bounds::TEMPERATURE, neutral: None }),
    Field::new("top_p", "top_p", Kind::Number { lo: 0.0, hi: bounds::UNIT, neutral: None }),
    Field::new("top_k", "top_k", Kind::Integer { lo: 0, hi: MAX_TOP_K }),
    Field::new("seed", "seed", Kind::Seed),
    Field::new("presence_penalty", "presence_penalty", Kind::Number { lo: -bounds::LOGIT_PENALTY, hi: bounds::LOGIT_PENALTY, neutral: Some(0.0) }),
    Field::new("frequency_penalty", "frequency_penalty", Kind::Number { lo: -bounds::LOGIT_PENALTY, hi: bounds::LOGIT_PENALTY, neutral: Some(0.0) }),
    Field::new("min_p", "min_p", Kind::Number { lo: 0.0, hi: bounds::UNIT, neutral: Some(0.0) }),
    Field::new("repetition_penalty", "repetition_penalty", Kind::Number { lo: 0.01, hi: bounds::REPETITION_PENALTY, neutral: Some(1.0) }),
    Field::new("dry_multiplier", "dry_multiplier", Kind::Number { lo: 0.0, hi: bounds::DRY_MULTIPLIER, neutral: Some(0.0) }),
    Field::new("dry_base", "dry_base", Kind::Number { lo: 1.0, hi: bounds::DRY_BASE, neutral: None }),
    Field::new("dry_allowed_length", "dry_allowed_length", Kind::Integer { lo: 0, hi: 4096 }),
    Field::new("dry_sequence_breakers", "dry_sequence_breakers", Kind::Strings),
    Field::new("thinking_budget", "thinking_budget", Kind::Integer { lo: 0, hi: i64::MAX }),
    Field::new("thinking_token_budget", "thinking_budget", Kind::Integer { lo: 0, hi: i64::MAX }),
    Field::new("max_thinking_tokens", "thinking_budget", Kind::Integer { lo: 0, hi: i64::MAX }),
    Field::new("ignore_eos", "ignore_eos", Kind::Switch),
];

/// The `generate` param a request's completion budget (`max_tokens`, or OpenAI's `max_completion_tokens`) sets.
pub const MAX_TOKENS_PARAM: &str = "max_new";

/// The token-capture fields [`apply_token_capture`] forwards, each to the param of its own name.
pub const TOKEN_CAPTURE: [&str; 3] = ["logprobs", "return_token_ids", "policy_trace"];

/// The `policy_trace` object's options and the `generate` params they set: how many of the most probable tokens of
/// each generated position's row the trace returns, and the most survivors of a cut draw it returns as each
/// position's support. Their ranges are the action's to declare (and refuse by name).
const TRACE_TOP_FIELD: &str = "top_log_probs";
const TRACE_TOP_PARAM: &str = "policy_trace_top_log_probs";
const TRACE_SUPPORT_FIELD: &str = "support";
const TRACE_SUPPORT_PARAM: &str = "policy_trace_support";
const TRACE_OPTIONS: [(&str, &str); 2] = [(TRACE_TOP_FIELD, TRACE_TOP_PARAM), (TRACE_SUPPORT_FIELD, TRACE_SUPPORT_PARAM)];

/// The request fields [`apply`] reads ([`FIELDS`]).
pub fn request_fields() -> impl Iterator<Item = &'static str> {
    FIELDS.iter().map(|f| f.name)
}

/// Set the sampling params of `inv` from `body` ([`FIELDS`]), each only when the client sent it: the validation fills the
/// action's default otherwise, which for a model that declares its own sampling defaults (its `generation_config.json`
/// or GGUF metadata) is the setting its authors recommend, else the chat fallback. A field outside its range is
/// refused by name; a model that has no such sampler refuses the param by name, as it does any parameter it does not
/// declare.
pub fn apply(provider: Provider, body: &Value, mut inv: Invocation) -> Result<Invocation, ApiError> {
    let mut sent: Vec<&str> = Vec::new();
    for field in &FIELDS {
        let Some(v) = present(body, field.name) else { continue };
        // A setting with several names (the thinking budget) is read from the first one sent.
        if sent.contains(&field.param) {
            continue;
        }
        sent.push(field.param);
        let value = field.read(v).map_err(|what| ApiError::invalid_request(provider, format!("'{}' must be {what}", field.name)))?;
        if let Some(value) = value {
            inv = inv.set(field.param, value);
        }
    }
    Ok(inv)
}

/// `inv` (a chat surface's request, [`apply`]) settled against the served action `spec` of the model it is for, before
/// it is validated (`crate::bridge::admit`): a neutral value the client sent for a sampler `spec` does not declare is
/// dropped, so a request that asks for nothing reaches a model without that sampler as the request it always was
/// instead of being refused for a field it did not mean. A neutral value for a sampler `spec` declares is kept: it
/// overrides the default the model declares. Every other default is the validation's to fill.
pub fn settle(spec: &ActionSpec, mut inv: Invocation) -> Invocation {
    let declared = |param: &str| spec.params.iter().any(|p| p.name == param);
    let Some(params) = inv.params.as_object_mut() else { return inv };
    for field in &FIELDS {
        if let (Kind::Number { neutral: Some(n), .. }, Some(v)) = (field.kind, params.get(field.param)) {
            if v.as_f64() == Some(n) && !declared(field.param) {
                params.remove(field.param);
            }
        }
    }
    inv
}

/// The chat request field that sets the `generate` param `param` (the first of [`FIELDS`] that sets it, `max_tokens`
/// for the completion budget), or `param` itself for one a chat request sends by its own name (`messages`, `tools`).
pub fn request_field(param: &str) -> &str {
    if param == MAX_TOKENS_PARAM {
        return "max_tokens";
    }
    if param == TRACE_TOP_PARAM {
        return "policy_trace.top_log_probs";
    }
    if param == TRACE_SUPPORT_PARAM {
        return "policy_trace.support";
    }
    FIELDS.iter().find(|f| f.param == param).map_or(param, |f| f.name)
}

/// The request fields a model whose `generate` action declares the params `declares` answers yes for honours: those
/// [`apply`] and [`apply_token_capture`] forward to a declared param, and `max_tokens` when the budget is one. Derived
/// from the tables the requests are read with, so a model card states exactly what a request may send it (a field
/// whose param the model does not declare is refused by name).
pub fn honoured(declares: impl Fn(&str) -> bool) -> Vec<&'static str> {
    let budget = declares(MAX_TOKENS_PARAM).then_some("max_tokens");
    let sampling = FIELDS.iter().filter(|f| declares(f.param)).map(|f| f.name);
    let capture = TOKEN_CAPTURE.into_iter().filter(|&name| declares(name));
    budget.into_iter().chain(sampling).chain(capture).collect()
}

/// The request fields a reply's sampler echo names: the settings that cut the distribution a token is drawn from.
/// Penalties, DRY and the thinking budget are not echoed: a request that sets one is answered as asked or refused.
const ECHOED: [&str; 4] = ["temp", "top_k", "top_p", "min_p"];

/// A model's report of the sampler settings a reply was drawn with (its `sampler` output, keyed by `generate` param),
/// renamed to the request fields a client sends them as ([`FIELDS`]: `temp` is `temperature`): what the reply's `brain`
/// object carries as `sampler`. `None` when the model reported none of them.
pub fn echo(settings: &Value) -> Option<Value> {
    let echoed: serde_json::Map<String, Value> = ECHOED
        .iter()
        .filter_map(|&param| {
            let field = FIELDS.iter().find(|f| f.param == param)?;
            settings.get(param).filter(|v| v.is_number()).map(|v| (field.name.to_string(), v.clone()))
        })
        .collect();
    (!echoed.is_empty()).then_some(Value::Object(echoed))
}

/// Ask the model for the sampled tokens' ids and log-probabilities when the
/// client sent `logprobs: true` (the log-probability of each sampled token
/// under the truncated, temperature-scaled distribution the sampler drew from,
/// and `0.0` for greedy), `return_token_ids: true` (the prompt's and the
/// completion's token ids; the extension vLLM names the same way) or
/// `policy_trace: true` (the training-grade trace: the ids and their
/// log-probabilities under the model's full, untruncated distribution, with
/// the policy that served them). None is forwarded unless sent, so a request
/// that does not read them pays nothing.
pub fn apply_token_capture(provider: Provider, body: &Value, mut inv: Invocation) -> Result<Invocation, ApiError> {
    for name in TOKEN_CAPTURE {
        match present(body, name) {
            None => {}
            // `policy_trace: {"top_log_probs": K, "support": N}` is the trace plus the sampler's top K and its support (at
            // most N ids) per generated position; either option alone, at least one.
            Some(Value::Object(options)) if name == "policy_trace" => {
                if let Some(unknown) = options.keys().find(|k| !TRACE_OPTIONS.iter().any(|(field, _)| field == k)) {
                    return Err(ApiError::invalid_request(provider, format!("'policy_trace.{unknown}' is not a policy trace option; the options are '{TRACE_TOP_FIELD}' and '{TRACE_SUPPORT_FIELD}'")));
                }
                if options.is_empty() {
                    return Err(ApiError::invalid_request(provider, format!("'policy_trace' as an object takes '{TRACE_TOP_FIELD}', '{TRACE_SUPPORT_FIELD}' or both; send true for the plain trace")));
                }
                inv = inv.set(name, json!(true));
                for (field, param) in TRACE_OPTIONS {
                    let Some(value) = options.get(field) else { continue };
                    let n = value.as_i64().ok_or_else(|| ApiError::invalid_request(provider, format!("'policy_trace.{field}' must be an integer")))?;
                    inv = inv.set(param, json!(n));
                }
            }
            Some(v) => {
                let on = v.as_bool().ok_or_else(|| ApiError::invalid_request(provider, if name == "policy_trace" { format!("'{name}' must be a boolean or an object {{\"{TRACE_TOP_FIELD}\": K, \"{TRACE_SUPPORT_FIELD}\": N}}") } else { format!("'{name}' must be a boolean") }))?;
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

    /// What the model receives for a setting the request left out is the default its served action declares (the
    /// checkpoint's own, where it declares one), the value its manifest shows, once the call is validated; what the
    /// request set stays the request's, a neutral value the client sent is kept for a sampler the model has, and the
    /// seed is never filled in.
    #[test]
    fn an_omitted_setting_takes_the_default_the_served_action_declares() {
        let spec = generate(&["temp", "top_k", "top_p", "min_p", "seed"]);
        let defaults = [("temp", json!(1.0)), ("top_k", json!(20)), ("top_p", json!(0.95)), ("min_p", json!(0.05))];
        let spec = capability::ActionSpec { params: spec.params.into_iter().map(|p| match defaults.iter().find(|(n, _)| *n == p.name) {
            Some((_, d)) => p.default(d.clone()),
            None => p,
        }).collect(), ..spec };
        let inv = spec.validate(settle(&spec, apply(P, &json!({"top_k": 5, "min_p": 0}), Invocation::new()).unwrap())).unwrap();
        assert_eq!((inv.get_f64("temp"), inv.get_i64("top_k"), inv.get_f64("top_p"), inv.get_f64("min_p")), (Some(1.0), Some(5), Some(0.95), Some(0.0)));
        assert_eq!(inv.get_i64("seed"), None, "an unseeded request draws a fresh seed");
    }

    /// A refusal names the field the client sent, not the param it set.
    #[test]
    fn a_param_is_named_by_the_request_field_that_sets_it() {
        assert_eq!(request_field("temp"), "temperature");
        assert_eq!(request_field(MAX_TOKENS_PARAM), "max_tokens");
        assert_eq!(request_field("thinking_budget"), "thinking_budget");
        assert_eq!(request_field("tools"), "tools");
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

    /// The extended sampler fields reach a model without that sampler only when the client asked for something: the
    /// neutral values every client sends by default are dropped for it, so a request that asks for nothing is the
    /// request it always was; a model with the sampler gets the value the client sent, over any default it declares.
    #[test]
    fn extended_sampler_fields_are_forwarded_only_when_asked_for() {
        let neutral = json!({"presence_penalty": 0, "frequency_penalty": 0.0, "min_p": 0, "repetition_penalty": 1.0, "dry_multiplier": 0});
        let inv = settle(&generate(&[]), apply(P, &neutral, Invocation::new()).unwrap());
        for k in ["presence_penalty", "frequency_penalty", "min_p", "repetition_penalty", "dry_multiplier", "thinking_budget"] {
            assert_eq!(inv.get_f64(k), None, "{k}");
        }
        let spec = generate(&["min_p", "repetition_penalty"]);
        let inv = settle(&spec, apply(P, &neutral, Invocation::new()).unwrap());
        assert_eq!((inv.get_f64("min_p"), inv.get_f64("repetition_penalty"), inv.get_f64("presence_penalty")), (Some(0.0), Some(1.0), None), "kept where the model has the sampler");
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

    /// A value of `field` a client may send that asks for something (not the neutral one).
    fn asking(field: &Field) -> Value {
        match field.kind {
            Kind::Number { hi, .. } => json!(hi),
            Kind::Integer { lo, .. } => json!(lo + 1),
            Kind::Seed => json!(7),
            Kind::Switch => json!(true),
            Kind::Strings => json!(["\n"]),
        }
    }

    /// A `generate` action declaring `params` (and the prompt it always takes).
    fn generate(params: &[&str]) -> capability::ActionSpec {
        params.iter().fold(capability::ActionSpec::new("generate", "test").param(capability::ParamSpec::new("prompt", capability::ParamType::Str, "")), |spec, &name| {
            let ty = match name {
                "seed" | "top_k" | "max_new" | "dry_allowed_length" | "thinking_budget" => capability::ParamType::Int,
                "ignore_eos" | "logprobs" | "return_token_ids" | "policy_trace" => capability::ParamType::Bool,
                "dry_sequence_breakers" => capability::ParamType::Str,
                _ => capability::ParamType::Float,
            };
            spec.param(capability::ParamSpec::new(name, ty, ""))
        })
    }

    /// The listing is what the sampler accepts: for every field a client may send, a model lists it exactly when a
    /// request carrying it passes the model's own `generate` validation, whatever subset of params the model declares.
    #[test]
    fn a_model_lists_exactly_the_sampling_fields_a_request_may_send_it() {
        let every: Vec<&str> = FIELDS.iter().map(|f| f.param).chain(TOKEN_CAPTURE).chain([MAX_TOKENS_PARAM]).collect();
        let models = [
            vec!["temp", "top_p", "max_new"],
            vec!["temp", "top_k", "top_p", "seed", "max_new", "logprobs"],
            vec!["temp", "top_k", "top_p", "seed", "min_p", "repetition_penalty", "thinking_budget", "ignore_eos"],
            every,
        ];
        for declared in models {
            let spec = generate(&declared);
            let listed = honoured(|p| spec.params.iter().any(|q| q.name == p));
            for field in &FIELDS {
                let body = json!({ field.name: asking(field) });
                let accepted = spec.validate(apply(P, &body, Invocation::new().set("prompt", json!("x"))).unwrap()).is_ok();
                assert_eq!(listed.contains(&field.name), accepted, "{} for a model declaring {declared:?}", field.name);
            }
            for name in TOKEN_CAPTURE {
                let inv = apply_token_capture(P, &json!({ name: true }), Invocation::new().set("prompt", json!("x"))).unwrap();
                assert_eq!(listed.contains(&name), spec.validate(inv).is_ok(), "{name} for a model declaring {declared:?}");
            }
            let budget = spec.validate(Invocation::new().set("prompt", json!("x")).set(MAX_TOKENS_PARAM, json!(8))).is_ok();
            assert_eq!(listed.contains(&"max_tokens"), budget, "max_tokens for a model declaring {declared:?}");
        }
        let listed = honoured(|p| ["temp", "top_k", "top_p", "seed", "max_new"].contains(&p));
        assert_eq!(listed, ["max_tokens", "temperature", "top_p", "top_k", "seed"], "a chat model's own sampler, seed and top_k included");
    }

    /// The echo names each setting by the request field that sets it: a value a client sends reaches the model as its
    /// param ([`apply`]), and the model's report of that param comes back under the client's own field, whatever the
    /// table calls it. A setting the echo leaves out (a penalty) is not reported.
    #[test]
    fn the_sampler_echo_names_each_setting_by_the_field_that_sets_it() {
        for param in ECHOED {
            let field = FIELDS.iter().find(|f| f.param == param).expect("an echoed setting is a request field");
            let sent = asking(field);
            let model_saw = apply(P, &json!({ field.name: sent }), Invocation::new()).unwrap().params;
            assert_eq!(echo(&model_saw), Some(json!({ field.name: sent })), "{param}");
        }
        assert_eq!(echo(&json!({"temp": 0.7, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 1.0})), Some(json!({"temperature": 0.7, "top_k": 20, "top_p": 0.95, "min_p": 0.0})));
        assert_eq!(echo(&json!({"presence_penalty": 1.0})), None, "nothing echoed, no object");
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
