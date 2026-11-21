// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `capabilities` keeps a binary of its own.

mod api;
mod audio;
mod backpressure;
mod callers;
mod connections;
mod edge;
mod route;
mod seams;
mod shutdown;
mod speech;
mod tls;
mod training_gate;
mod video;

use capability::{ActionSpec, ParamSpec, ParamType};

/// `spec` declaring the prompt half of the chat contract a served chat model's `generate` declares
/// (`llmserve::generate_spec`): the conversation, the raw prompt, the token budget and the stop strings, none with a default, each added
/// only where `spec` does not declare it itself. A chat route's invocation is validated against the served spec, so a
/// fake chat model must take what the routes send it; the sampler and tools a fixture declares itself, since a model
/// card lists them.
pub fn chat_contract(spec: ActionSpec) -> ActionSpec {
    const CONTRACT: [(&str, ParamType); 6] = [
        ("prompt", ParamType::Str),
        ("max_new", ParamType::Int),
        ("messages", ParamType::Str),
        ("system", ParamType::Str),
        ("chat", ParamType::Bool),
        ("stop", ParamType::Str),
    ];
    CONTRACT.into_iter().fold(spec, |spec, (name, ty)| if spec.params.iter().any(|p| p.name == name) { spec } else { spec.param(ParamSpec::new(name, ty, "the chat contract")) })
}
