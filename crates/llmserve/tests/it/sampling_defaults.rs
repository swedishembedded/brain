// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving language models with the
// sampling their authors validated, for its clients. If your team needs
// expertise in making served models behave as their checkpoints intend, then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Spec of a checkpoint's declared sampling defaults reaching the requests that
//! omit the settings: through the action's param defaults, which both the
//! manifest a client reads and the validation of a request apply.

use capability::{ActionSpec, Invocation, ParamSpec, ParamType};
use data::generation::SamplingDefaults;
use llmserve::with_sampling_defaults;
use serde_json::json;

fn generate() -> ActionSpec {
    ActionSpec::new("generate", "generate")
        .param(ParamSpec::new("temp", ParamType::Float, "temperature").default(json!(0.0)))
        .param(ParamSpec::new("top_k", ParamType::Int, "top-k").default(json!(40)))
        .param(ParamSpec::new("top_p", ParamType::Float, "top-p").default(json!(1.0)))
}

fn params_of(spec: &ActionSpec, request: Invocation) -> serde_json::Value {
    spec.validate(request).expect("a valid request").params
}

#[test]
fn a_request_that_omits_the_settings_gets_what_the_checkpoint_declares() {
    let declared = SamplingDefaults { temp: Some(1.0), top_k: Some(20), top_p: Some(0.95), ..Default::default() };
    let spec = with_sampling_defaults(generate(), "test", &declared);
    let got = params_of(&spec, Invocation::new());
    assert_eq!((got["temp"].as_f64(), got["top_k"].as_u64(), got["top_p"].as_f64()), (Some(1.0), Some(20), Some(0.95)), "the value a client reads is the decimal the checkpoint declared");
    // A setting the request names stays the request's.
    let got = params_of(&spec, Invocation::new().set("temp", json!(0.6)));
    assert_eq!(got["temp"].as_f64().map(|t| (t * 10.0).round()), Some(6.0));
    assert_eq!(got["top_k"].as_u64(), Some(20));
}

fn chat() -> ActionSpec {
    generate().param(ParamSpec::new("messages", ParamType::Str, "chat turns"))
}

#[test]
fn a_chat_action_whose_checkpoint_declares_nothing_gets_the_chat_fallback() {
    let spec = with_sampling_defaults(chat(), "test", &SamplingDefaults::default());
    let got = params_of(&spec, Invocation::new());
    assert_eq!((got["temp"].as_f64(), got["top_k"].as_u64(), got["top_p"].as_f64()), (Some(0.7), Some(20), Some(0.95)));
    // A partly declared checkpoint keeps what it declares and takes the fallback for the rest.
    let spec = with_sampling_defaults(chat(), "test", &SamplingDefaults { temp: Some(1.0), ..Default::default() });
    let got = params_of(&spec, Invocation::new());
    assert_eq!((got["temp"].as_f64(), got["top_k"].as_u64(), got["top_p"].as_f64()), (Some(1.0), Some(20), Some(0.95)));
}

#[test]
fn an_action_that_is_not_chat_keeps_its_constants_when_nothing_is_declared() {
    let spec = with_sampling_defaults(generate(), "test", &SamplingDefaults::default());
    let got = params_of(&spec, Invocation::new());
    assert_eq!((got["temp"].as_f64(), got["top_k"].as_u64(), got["top_p"].as_f64()), (Some(0.0), Some(40), Some(1.0)));
}

#[test]
fn a_declared_setting_the_action_cannot_carry_does_not_invent_a_parameter() {
    let spec = with_sampling_defaults(generate(), "test", &SamplingDefaults { min_p: Some(0.05), ..Default::default() });
    assert!(spec.params.iter().all(|p| p.name != "min_p"));
}
