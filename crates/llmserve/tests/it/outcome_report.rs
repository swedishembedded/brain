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

use crate::common;
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
            max_new: inv.get_i64("max_new").map(|n| n as usize),
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
    // 128 blocks of 8: the KV for the toy's whole 1024-token context.
    let mut sched = Scheduler::new(Toy::new(BlockAllocator::new(128, 8), cached), 4);
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

#[test]
fn a_request_without_a_budget_runs_to_the_end_of_the_context() {
    // Without its stop token the toy never ends its turn, so only the
    // context (1024 tokens, 4 of them the prompt) ends the reply.
    let out = serve(0, Invocation::new().set("ignore_eos", json!(true)));
    assert_eq!(out.outputs["completion_tokens"], json!(1020));
    assert_eq!(out.outputs["finish_reason"], json!("length"));
}

const _: () = assert!(VOCAB > 8);

/// A policy trace is the prompt, the generated ids with the stop token that ended them, and a log-probability for
/// each of those generated ids, scored by the decoder; the plain outputs are unchanged by asking.
#[test]
fn a_policy_trace_scores_the_generation_with_its_stop_token() {
    let front = Front { tok: tokenizer() };
    let mut sched = Scheduler::new(Toy::new(BlockAllocator::new(128, 8), 0).scoring(), 4);
    let asked = Invocation::new().set("policy_trace", json!(true));
    let results = run_batch_scheduled(&mut sched, &front, &[asked, Invocation::new()], &mut |_, _| {});
    let asked = results[0].clone().expect("served").outputs;
    let plain = results[1].clone().expect("served").outputs;
    assert!(plain.get("policy_trace").is_none());
    assert_eq!(asked["completion_tokens"], plain["completion_tokens"]);
    let trace = &asked["policy_trace"];
    // The toy counts up from the prompt's last token 4: 5, 6, then the stop token 7.
    assert_eq!(trace["prompt_token_ids"], json!([1, 2, 3, 4]));
    assert_eq!(trace["generation_token_ids"], json!([5, 6, 7]));
    let scored: Vec<f64> = serde_json::from_value(trace["generation_log_probs"].clone()).unwrap();
    assert!(scored.iter().zip([-0.05, -0.06, -0.07]).all(|(got, want)| (got - want).abs() < 1e-6), "{scored:?}");
    assert_eq!(trace["log_prob_temperature"], json!(1.0), "a greedy request is scored at temperature 1");
    assert!(trace["chat_template_digest"].as_str().is_some_and(|d| d.starts_with("sha256:") && d.len() == 71), "the probe conversation renders: {trace}");
}

/// A decoder that cannot score its own generation refuses the request before decoding, by class and name.
#[test]
fn a_policy_trace_is_refused_by_a_decoder_that_cannot_score() {
    let front = Front { tok: tokenizer() };
    let mut sched = Scheduler::new(Toy::new(BlockAllocator::new(128, 8), 0), 4);
    let results = run_batch_scheduled(&mut sched, &front, &[Invocation::new().set("policy_trace", json!(true))], &mut |_, _| {});
    let error = results[0].clone().expect_err("no scoring pass");
    assert!(error.starts_with("[unsupported-capability]") && error.contains("policy_trace"), "{error}");
}
