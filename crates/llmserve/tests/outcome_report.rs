// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for LLM serving for agent
// workloads for its clients. If your team needs expertise in serving that lets
// a client see what the model really wrote and what the server reused, then you
// can procure our services by sending an email to info@swedishembedded.com.

//! Spec of what a served generation reports beside its text: how many prompt
//! tokens the prefix cache served (`cached_tokens`), the seed it sampled with,
//! and - when asked with `return_raw` - the exact text the model wrote, ending
//! with the stop token that ended it.

use capability::Invocation;
use data::qwen_chat::TemplateFlavor;
use data::qwen_tokenizer::QwenBpe;
use llmserve::{run_batch_scheduled, Frontend, ParsedRequest};
use model::paged::BlockAllocator;
use model::serve::{SampleParams, Scheduler};
use qwen3::chat::ToolChoice;
use serde_json::json;

mod common;
use common::{tokenizer, Toy, VOCAB};

/// Ends on token 7; the toy decoder counts up from the prompt's last token.
struct Front {
    tok: QwenBpe,
}

impl Frontend for Front {
    fn tok(&self) -> &QwenBpe {
        &self.tok
    }
    fn stop(&self) -> &[u32] {
        &[7]
    }
    fn label(&self) -> &str {
        "toy"
    }
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String> {
        Ok(ParsedRequest {
            ids: vec![1, 2, 3, 4],
            max_new: inv.get_i64("max_new").unwrap_or(16) as usize,
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            sampler: SampleParams::greedy(),
            seed: inv.seed()?.unwrap_or(99),
            stops: Vec::new(),
            tool_choice: ToolChoice::Auto,
            flavor: TemplateFlavor::default(),
            thinking_open: false,
            raw: true,
        })
    }
}

fn serve(cached: u32, inv: Invocation) -> capability::Outcome {
    let front = Front { tok: tokenizer() };
    let mut sched = Scheduler::new(Toy::new(BlockAllocator::new(64, 8), cached), 4);
    let mut results = run_batch_scheduled(&mut sched, &front, &[inv], &mut |_, _| {});
    results.remove(0).expect("served")
}

#[test]
fn an_outcome_reports_the_prompt_tokens_the_prefix_cache_served() {
    assert_eq!(serve(0, Invocation::new()).outputs["cached_tokens"], json!(0));
    assert_eq!(serve(3, Invocation::new()).outputs["cached_tokens"], json!(3));
}

#[test]
fn an_outcome_echoes_the_seed_it_sampled_with() {
    assert_eq!(serve(0, Invocation::new().set("seed", json!(1234))).outputs["seed"], json!(1234));
    assert_eq!(serve(0, Invocation::new()).outputs["seed"], json!(99), "the seed the request was parsed with, drawn or given");
}

#[test]
fn raw_output_is_the_generated_text_ending_with_the_stop_token() {
    let out = serve(0, Invocation::new().set("return_raw", json!(true)));
    // 5 and 6 are generated; 7 stops the sequence and is not among the completion tokens.
    assert_eq!(out.outputs["completion_tokens"], json!(2));
    assert_eq!(out.outputs["stop_token"], json!(7));
    assert_eq!(out.outputs["raw_output"], json!("t5t6t7"));
    assert!(serve(0, Invocation::new()).outputs.get("raw_output").is_none(), "off unless asked for");
}

#[test]
fn a_sequence_that_runs_out_of_tokens_has_no_stop_token() {
    let out = serve(0, Invocation::new().set("return_raw", json!(true)).set("max_new", json!(1)));
    assert!(out.outputs.get("stop_token").is_none());
    assert_eq!(out.outputs["raw_output"], json!("t5"));
}

const _: () = assert!(VOCAB > 8);
