// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching LLM serving for clients
// whose agents resend long conversation histories. If your team needs
// expertise in LLM serving throughput then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The qwen35 paged engine through the generic [`model::serve::conformance`]
//! suite on a tiny random-weight model. This engine keeps no prefix cache, and
//! says so (`prefix_reuse: false`); the suite then checks it reports no hits.

use model::serve::conformance::{self, Conformance, MIN_SEQ_TOKENS};
use qwen35::config::Qwen35Config;
use qwen35::model::pipelines;
use qwen35::serve::Engine;

fn engine() -> Engine {
    let cfg = Qwen35Config::tiny();
    let weights = qwen35::init::init_weights(&cfg, 11);
    Engine::from_map_on(&gpu_core::Gpu::new_cpu(pipelines()), cfg, &weights, MIN_SEQ_TOKENS as u32, 4)
}

#[test]
fn qwen35_engine_conforms() {
    conformance::run_all(&Conformance { make: engine, prefix_reuse: false, prefix_prompt_tokens: 24, load_adapter: None });
}
