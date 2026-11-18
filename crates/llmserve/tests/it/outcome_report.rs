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
use common::{row, tokenizer, Toy, VOCAB};

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

/// A decoder that is busy when a request finishes (a prompt is being read in) scores its trace once it is ready: the
/// request is answered then, with its whole trace, and a request without a trace is not held up by it.
#[test]
fn a_policy_trace_waits_for_a_decoder_that_is_not_ready_to_score() {
    let front = Front { tok: tokenizer() };
    let mut sched = Scheduler::new(Toy::new(BlockAllocator::new(128, 8), 0).scoring_after(5), 4);
    let results = run_batch_scheduled(&mut sched, &front, &[Invocation::new().set("policy_trace", json!(true)), Invocation::new()], &mut |_, _| {});
    let traced = results[0].clone().expect("served after the wait").outputs;
    assert_eq!(traced["policy_trace"]["generation_token_ids"], json!([5, 6, 7]));
    assert!(results[1].clone().expect("served").outputs.get("policy_trace").is_none());
}

/// SPEC: a policy trace of a sampled reply is the decode steps' own log-probabilities under the whole row at the request
/// temperature, `log_softmax(logits / T)[token]` of the row that drew each token, with the stop token's, whatever cut
/// (top-k, top-p, min-p) the draw made. No scoring pass runs for it: the trace says it came from the decode, and the
/// decoder is never asked to score. A cut draw's own figure (what `completion_logprobs` reports) is the token's under
/// the survivors, which the whole-row figure never exceeds.
#[test]
fn a_policy_trace_of_a_sampled_draw_is_the_whole_row_log_probability_of_the_decode_that_drew_it() {
    let spec = chat_spec();
    let request = |seed: u64, top_k: i64, top_p: f64, min_p: f64| {
        let inv = Invocation::new().set("prompt", json!("t1t2t3t4")).set("chat", json!(false)).set("max_new", json!(12)).set("logprobs", json!(true)).set("policy_trace", json!(true)).set("seed", json!(seed));
        spec.validate(inv.set("temp", json!(1.0)).set("top_k", json!(top_k)).set("top_p", json!(top_p)).set("min_p", json!(min_p))).expect("a valid request")
    };
    let front = Parsed { tok: tokenizer() };
    // The toy's scores are `-token / 100` whatever precedes: a trace that came from a scoring pass would read so.
    let mut sched = Scheduler::new(Toy::new(BlockAllocator::new(128, 8), 0).sampling().scoring(), 4);
    let cuts = [(0, 1.0, 0.0), (0, 1.0, 0.0), (5, 1.0, 0.0), (0, 0.9, 0.0), (0, 1.0, 0.2)];
    let requests: Vec<_> = cuts.iter().enumerate().map(|(i, &(k, p, m))| request(3 + i as u64, k, p, m)).collect();
    let results = run_batch_scheduled(&mut sched, &front, &requests, &mut |_, _| {});
    assert_eq!(sched.decoder_mut().scores_run, 0, "no reply is re-read by a scoring prefill: the decode already reported its figures");
    for (i, result) in results.into_iter().enumerate() {
        let out = result.unwrap_or_else(|e| panic!("request {i} {:?}: {e:?}", cuts[i])).outputs;
        let trace = &out["policy_trace"];
        assert_eq!(trace["log_prob_source"], json!("decode"), "{:?}: {trace}", cuts[i]);
        let generated: Vec<u32> = serde_json::from_value(trace["generation_token_ids"].clone()).unwrap();
        let traced: Vec<f32> = serde_json::from_value(trace["generation_log_probs"].clone()).unwrap();
        let sampler: Vec<f32> = serde_json::from_value(out["completion_logprobs"].clone()).unwrap();
        let prompt: Vec<u32> = serde_json::from_value(trace["prompt_token_ids"].clone()).unwrap();
        assert_eq!(traced.len(), generated.len());
        let whole = SampleParams { temp: 1.0, ..SampleParams::greedy() };
        let mut previous = *prompt.last().expect("a prompt");
        for (j, (&token, &lp)) in generated.iter().zip(&traced).enumerate() {
            let want = model::serve::candidate_logprob(&row(previous), whole, token);
            assert!((lp - want).abs() < 1e-4, "{:?} token {j}: traced {lp} against the row's {want}", cuts[i]);
            if let Some(&own) = sampler.get(j) {
                assert!(lp <= own + 1e-4, "{:?} token {j}: the whole row's figure {lp} exceeds the survivors' {own}", cuts[i]);
            }
            previous = token;
        }
        if cuts[i].0 == 5 {
            assert!(traced.iter().zip(&sampler).any(|(w, own)| w + 1e-3 < *own), "a top-k draw's whole-row figures sit below the survivors': {traced:?} against {sampler:?}");
        }
    }
}

/// The real request path's front end: a `generate` request read by the Qwen parser, with the defaults its action spec
/// applies already in the invocation (the API validates every request against the spec before the model sees it).
struct Parsed {
    tok: QwenBpe,
}

impl Frontend for Parsed {
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
        qwen3::chat::parse_request(&self.tok, inv)
    }
}

/// A chat action whose checkpoint declares its own sampling defaults, the shape of every served chat model's.
fn chat_spec() -> capability::ActionSpec {
    let declared = data::generation::SamplingDefaults { temp: Some(0.6), top_k: Some(20), top_p: Some(0.95), ..Default::default() };
    llmserve::with_sampling_defaults(llmserve::with_token_capture(llmserve::generate_spec("toy", true)), "toy", &declared)
}

/// SPEC: a reply names the sampler settings its tokens were drawn with (`sampler`: `temp`, `top_k`, `top_p`, `min_p`),
/// after the checkpoint's declared defaults and the parse, and they are the ones the sampler used: the log-probability
/// the sampler reported for every drawn token is that token's under the distribution those settings cut from the row.
#[test]
fn a_reply_names_the_sampler_settings_its_tokens_were_drawn_with() {
    let spec = chat_spec();
    let request = |extra: &[(&str, serde_json::Value)]| {
        let inv = extra.iter().fold(Invocation::new().set("prompt", json!("t1t2t3t4")).set("chat", json!(false)).set("max_new", json!(12)).set("ignore_eos", json!(true)).set("logprobs", json!(true)).set("return_token_ids", json!(true)), |inv, (k, v)| inv.set(k, v.clone()));
        spec.validate(inv).expect("a valid request")
    };
    let cases = [
        (request(&[("seed", json!(3))]), json!({"temp": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.0})),
        (request(&[("seed", json!(4)), ("temp", json!(1.3)), ("top_k", json!(0)), ("top_p", json!(1.0)), ("min_p", json!(0.05))]), json!({"temp": 1.3, "top_k": 0, "top_p": 1.0, "min_p": 0.05})),
        (request(&[("seed", json!(5)), ("temp", json!(0.9)), ("top_k", json!(5)), ("top_p", json!(0.5))]), json!({"temp": 0.9, "top_k": 5, "top_p": 0.5, "min_p": 0.0})),
        (request(&[("temp", json!(0.0))]), json!({"temp": 0.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0})),
    ];
    let front = Parsed { tok: tokenizer() };
    let mut sched = Scheduler::new(Toy::new(BlockAllocator::new(128, 8), 0).sampling(), 4);
    let invs: Vec<Invocation> = cases.iter().map(|(inv, _)| inv.clone()).collect();
    let results = run_batch_scheduled(&mut sched, &front, &invs, &mut |_, _| {});
    for ((inv, want), result) in cases.iter().zip(results) {
        let out = result.expect("served").outputs;
        assert_eq!(out["sampler"], *want, "{inv:?}");
        let echoed = |key: &str| out["sampler"][key].as_f64().expect("a number") as f32;
        let params = SampleParams { temp: echoed("temp"), top_k: echoed("top_k") as usize, top_p: echoed("top_p"), min_p: echoed("min_p"), ..SampleParams::greedy() };
        let prompt: Vec<u32> = serde_json::from_value(out["prompt_token_ids"].clone()).expect("prompt ids");
        let tokens: Vec<u32> = serde_json::from_value(out["completion_token_ids"].clone()).expect("ids");
        let logprobs: Vec<f32> = serde_json::from_value(out["completion_logprobs"].clone()).expect("logprobs");
        assert_eq!(tokens.len(), 12);
        let mut previous = *prompt.last().expect("a prompt");
        for (i, (&token, &reported)) in tokens.iter().zip(&logprobs).enumerate() {
            let recomputed = model::serve::candidate_logprob(&row(previous), params, token);
            assert!((recomputed - reported).abs() < 1e-5, "{want}: token {i} ({token}) was drawn at {reported}, the echoed settings give {recomputed}");
            previous = token;
        }
    }
}
