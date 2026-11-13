// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring what a served language model assigns to given text,
// under any adapter, without a training process, for its clients. If your team needs expertise in held-out
// evaluation of fine-tuned models then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The `score` action: teacher-forced log-likelihood of the supervised answers of given chat records under the served
//! model (and the adapter its id names), through the serving scheduler.
//!
//! A record is a `generic-messages-v2` line. Each supervised assistant message (`"train": true`) is one answer, scored
//! as the model is asked for it: the prompt is the request path's own render of the messages before it (the same
//! [`Frontend::parse`] a `generate` call uses, so the template, the tokenizer's escaping of client text and the
//! generation prompt are the sampler's), the answer's tokens are its content, tokenized with client text escaped, and
//! the end-of-turn token that closes it (`include_stop`, default on). The log-probabilities are the model's full,
//! untruncated distribution at temperature 1 ([`Scheduler::score_continuation`]): one teacher-forced pass per answer
//! under the adapter, with the same weights, quantisation and KV precision that serve generation.
//!
//! Scoring is scheduled work: a `score` group runs on the instance like any other group, one prefill per answer
//! between the instance's decode steps, so it is concurrency-safe by the same lock the scheduler owns. A decoder that
//! cannot score a continuation refuses the request by capability, before any work.

use capability::{ActionResult, ActionSpec, BlobSpec, Invocation, Media, Outcome, ParamSpec, ParamType};
use data::chat::{ChatSample, TextOnly};
use data::tokenizer::Tokenizer;
use model::serve::{PagedDecoder, Scheduler};
use serde_json::{json, Value};

use crate::Frontend;

/// The action's name.
pub const SCORE_ACTION: &str = "score";
/// The most records one call scores.
pub const MAX_SCORE_RECORDS: usize = 4096;
/// The most answers (prefill passes) one call scores, over all its records: a call holds the instance for as long as
/// it runs, so its work is bounded as a request's is.
pub const MAX_ANSWERS_PER_CALL: usize = 8192;
/// The most supervised answers one record may hold.
pub const MAX_ANSWERS_PER_RECORD: usize = 64;

/// What `score` is, for every scheduler-backed chat manifest.
pub fn score_spec() -> ActionSpec {
    ActionSpec::new(SCORE_ACTION, "teacher-forced log-likelihood of the supervised answers of chat records under this model (and the adapter its id names): per answer, per record and in total, optionally per token; nothing is generated")
        .param(ParamSpec::new("per_token", ParamType::Bool, "also return each answer's token ids and their log-probabilities").default(json!(false)))
        .param(ParamSpec::new("include_stop", ParamType::Bool, "score the end-of-turn token that closes each answer too, as a trainer that trains the stop does").default(json!(true)))
        .param(ParamSpec::new("enable_thinking", ParamType::Bool, "render the prompts as a request that allows reasoning would").default(json!(true)))
        .input(BlobSpec::new("records", Media::Text, "`generic-messages-v2` JSONL: the messages marked `\"train\": true` are scored; plain text answers only (no tool calls, reasoning content or tool schemas)").required())
}

/// One answer to score: where it sits and the ids around it.
struct Answer {
    message: usize,
    prompt: Vec<u32>,
    answer: Vec<u32>,
}

fn invalid(message: impl AsRef<str>) -> String {
    capability::failure::tag(capability::failure::FailureClass::InvalidRequest, message)
}

/// The answers of record `index` (1-based in messages), each with its prompt rendered by `front`.
fn answers_of<F: Frontend + ?Sized>(front: &F, index: usize, line: &str, thinking: bool, include_stop: bool) -> Result<Vec<Answer>, String> {
    let at = |e: String| invalid(format!("record {index}: {e}"));
    let sample = ChatSample::from_json_line_with(line, &TextOnly::TRAINER).map_err(at)?;
    if !sample.tools.is_empty() || sample.messages.iter().any(|m| !m.tool_calls.is_empty() || m.tool_call_id.is_some() || m.role == "tool") {
        return Err(at("tool schemas, tool calls and tool results cannot be scored: records hold plain text turns".to_string()));
    }
    let supervised: Vec<usize> = sample.messages.iter().enumerate().filter(|(_, m)| m.train).map(|(i, _)| i).collect();
    if supervised.is_empty() {
        return Err(at("holds no supervised message (`\"train\": true`)".to_string()));
    }
    if supervised.len() > MAX_ANSWERS_PER_RECORD {
        return Err(at(format!("holds {} supervised messages; at most {MAX_ANSWERS_PER_RECORD} are scored", supervised.len())));
    }
    let tok = front.tok();
    let mut answers = Vec::with_capacity(supervised.len());
    for message in supervised {
        let m = &sample.messages[message];
        if m.role != "assistant" || m.reasoning_content.is_some() {
            return Err(at(format!("message {} is supervised but is not a plain assistant answer (a reasoning part cannot be scored)", message + 1)));
        }
        if message == 0 {
            return Err(at("the first message is supervised, so there is no prompt to score it after".to_string()));
        }
        let context: Vec<Value> = sample.messages[..message].iter().map(|c| json!({"role": c.role, "content": c.content})).collect();
        let ask = Invocation::new().set("messages", json!(Value::Array(context).to_string())).set("enable_thinking", json!(thinking));
        let prompt = front.parse(&ask).map_err(|e| at(format!("message {}: {e}", message + 1)))?.ids;
        // Client text cannot spell a control token: the same escaping the request path applies.
        let mut answer = tok.encode(&tok.escape_untrusted(&m.content));
        if include_stop {
            answer.extend(front.stop().first().copied());
        }
        if answer.is_empty() {
            return Err(at(format!("message {} encodes to no token", message + 1)));
        }
        answers.push(Answer { message: message + 1, prompt, answer });
    }
    Ok(answers)
}

/// Scores one `score` call: every answer of every record of its `records` blob.
fn score_one<D: PagedDecoder, F: Frontend + ?Sized>(sched: &mut Scheduler<D>, front: &F, inv: &Invocation) -> ActionResult {
    if !sched.scores_continuations() {
        return Err(capability::failure::tag(capability::failure::FailureClass::UnsupportedCapability, format!("{}: score: this model has no teacher-forced scoring pass", front.label())));
    }
    let blob = inv.get_blob("records").ok_or_else(|| invalid("score: the input blob `records` is required"))?;
    let text = std::str::from_utf8(&blob.bytes).map_err(|e| invalid(format!("score: records is not UTF-8: {e}")))?;
    let (per_token, include_stop, thinking) = (inv.get_bool("per_token").unwrap_or(false), inv.get_bool("include_stop").unwrap_or(true), inv.get_bool("enable_thinking").unwrap_or(true));
    let (adapter, identity) = match front.route(inv)? {
        Some((adapter, identity)) => (Some(adapter), Some(identity)),
        None => (None, None),
    };
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.is_empty() {
        return Err(invalid("score: records holds no record"));
    }
    if lines.len() > MAX_SCORE_RECORDS {
        return Err(invalid(format!("score: {} records; at most {MAX_SCORE_RECORDS} are scored in one call", lines.len())));
    }
    // Every record is read before any is scored, so a malformed one fails the call without costing a pass.
    let records: Vec<Vec<Answer>> = lines.iter().enumerate().map(|(i, line)| answers_of(front, i + 1, line, thinking, include_stop)).collect::<Result<_, _>>()?;
    let answers_in_call: usize = records.iter().map(Vec::len).sum();
    if answers_in_call > MAX_ANSWERS_PER_CALL {
        return Err(invalid(format!("score: {answers_in_call} supervised answers; at most {MAX_ANSWERS_PER_CALL} are scored in one call")));
    }
    let capacity = sched.sequence_capacity();
    let (mut total_tokens, mut total_logprob) = (0usize, 0f64);
    let mut scored_records = Vec::with_capacity(records.len());
    for (index, answers) in records.into_iter().enumerate() {
        let mut scored_answers = Vec::with_capacity(answers.len());
        let (mut record_tokens, mut record_logprob) = (0usize, 0f64);
        for a in answers {
            if inv.cancel.is_cancelled() {
                return Err("cancelled".to_string());
            }
            let mut all = a.prompt.clone();
            all.extend(&a.answer);
            if all.len() > capacity {
                return Err(invalid(format!("record {}: message {} is {} tokens with its prompt; this model holds at most {capacity}", index + 1, a.message, all.len())));
            }
            let logprobs = sched.score_continuation(adapter, &all, a.prompt.len(), 1.0).map_err(|e| capability::failure::tag(capability::failure::FailureClass::Unavailable, format!("{}: score: {e}", front.label())))?;
            let sum: f64 = logprobs.iter().map(|&l| f64::from(l)).sum();
            record_tokens += logprobs.len();
            record_logprob += sum;
            let mut entry = json!({"message": a.message, "tokens": logprobs.len(), "logprob": sum, "mean_logprob": sum / logprobs.len() as f64});
            if per_token {
                entry["token_ids"] = json!(a.answer);
                entry["token_logprobs"] = json!(logprobs);
            }
            scored_answers.push(entry);
        }
        total_tokens += record_tokens;
        total_logprob += record_logprob;
        scored_records.push(json!({"record": index + 1, "tokens": record_tokens, "logprob": record_logprob, "mean_logprob": record_logprob / record_tokens as f64, "answers": scored_answers}));
    }
    let mut outcome = Outcome::new()
        .set("records", json!(scored_records))
        .set("tokens", json!(total_tokens))
        .set("logprob", json!(total_logprob))
        .set("mean_logprob", json!(total_logprob / total_tokens as f64))
        .set("log_prob_temperature", json!(1.0))
        .set("backend", json!(gpu_core::backend_name()))
        .set("engine", json!(concat!("brain ", env!("CARGO_PKG_VERSION"))));
    if let Some(digest) = front.tokenizer_digest() {
        outcome = outcome.set("tokenizer_digest", json!(digest));
    }
    if let Some(digest) = front.chat_template_digest() {
        outcome = outcome.set("chat_template_digest", json!(digest));
    }
    if let Some(digest) = front.base_digest() {
        outcome = outcome.set("base_digest", json!(digest));
    }
    if let Some(identity) = identity {
        outcome = outcome.set("adapter", json!({"id": identity.id, "digest": identity.digest, "version": identity.version}));
    }
    Ok(outcome)
}

/// Scores every call of `invs` on `sched`, one result per call, in order. The calls of a group share the scheduler
/// and its prefix cache is neither read nor written by a scoring pass, so what they cost is one prefill per answer.
pub fn score_batch_scheduled<D: PagedDecoder, F: Frontend + ?Sized>(sched: &mut Scheduler<D>, front: &F, invs: &[Invocation]) -> Vec<ActionResult> {
    invs.iter().map(|inv| score_one(sched, front, inv)).collect()
}
