// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training agents on whole sessions of their own sampled tokens, for its
// clients. If your team needs expertise in turning multi-call agent episodes into exact training samples then you can
// procure our services by sending an email to info@swedishembedded.com.

//! A session-merged rollout record: one record of a whole agent session whose every sampled answer is supervised,
//! each with the ids the sampler was given (`prompt_token_ids`) and generated (`token_ids`) and their behaviour
//! log-probabilities, trained as one sequence ([`ChatSample::answers_packed`]). It is exact only when the session is
//! append-only: every call's request is the previous call's request, its sampled answer and the turns between, token for
//! token. Then the merged sample is its per-call records laid over one another, and the shared context is encoded once.
//! A session whose requests were re-rendered between calls (a template that drops an earlier answer's reasoning, a
//! compaction) is refused by rule (`session_not_append_only`), never trained in a context the sampler did not see.

use data::chat::{ChatSample, EncodedSample, RenderOpts, TextOnly};
use data::chat_rollout::RecordRule;
use data::chat_template::{parse_json_ordered, ChatTemplate};
use data::tokenizer::{CharTokenizer, Tokenizer};

/// A template that writes every turn as it was given: a session through it is append-only.
const KEEPS_HISTORY: &str = "{% for m in messages %}{% if m.role == 'assistant' %}<|im_start|>assistant\n{% if m.reasoning_content %}<think>\n{{ m.reasoning_content }}\n</think>\n\n{% endif %}{{ m.content }}<|im_end|>\n\
{% elif m.role == 'tool' %}<|im_start|>user\n<tool_response>\n{{ m.content }}\n</tool_response><|im_end|>\n\
{% else %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// Qwen3's habit: an assistant turn's reasoning is written only when it is the conversation's last message, so the
/// next call's request holds the earlier answer without the reasoning the model sampled.
const DROPS_REASONING: &str = "{% for m in messages %}{% if m.role == 'assistant' %}<|im_start|>assistant\n{% if m.reasoning_content and loop.last %}<think>\n{{ m.reasoning_content }}\n</think>\n\n{% endif %}{{ m.content }}<|im_end|>\n\
{% elif m.role == 'tool' %}<|im_start|>user\n<tool_response>\n{{ m.content }}\n</tool_response><|im_end|>\n\
{% else %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

/// One call of the session: its answer's message index, the ids its request held, the ids it generated and their
/// behaviour log-probabilities.
type Call = (usize, Vec<u32>, Vec<u32>, Vec<f32>);

fn tokenizer() -> CharTokenizer {
    CharTokenizer::from_corpus(&(32u8..127).map(char::from).chain(['\n']).collect::<String>())
}

/// The session: a user question, then three answers (each with its reasoning) with a tool's result after the first two.
fn session() -> Vec<serde_json::Value> {
    use serde_json::json;
    vec![
        json!({"role": "user", "content": "list the files"}),
        json!({"role": "assistant", "content": "ls", "reasoning_content": "look"}),
        json!({"role": "user", "content": "a.txt b.txt"}),
        json!({"role": "assistant", "content": "cat a.txt", "reasoning_content": "read a"}),
        json!({"role": "user", "content": "hello"}),
        json!({"role": "assistant", "content": "done", "reasoning_content": "enough"}),
    ]
}

/// What the sampler was given and generated at each call of the session under `template`: the request's ids (the
/// conversation so far with the generation prompt) and the answer's ids (as the model wrote it, reasoning included,
/// with its end-of-turn marker), and a behaviour log-probability per generated id.
fn calls(template: &str) -> Vec<Call> {
    let (tok, tmpl) = (tokenizer(), ChatTemplate::compile(template).unwrap());
    let messages = session();
    (0..messages.len())
        .filter(|&at| messages[at]["role"] == "assistant")
        .map(|at| {
            let so_far = serde_json::to_string(&messages[..at]).unwrap();
            let prompt = tok.encode(&tmpl.render(parse_json_ordered(&so_far).unwrap(), None, true, &Default::default()).unwrap());
            let m = &messages[at];
            let answer = tok.encode(&format!("<think>\n{}\n</think>\n\n{}<|im_end|>", m["reasoning_content"].as_str().unwrap(), m["content"].as_str().unwrap()));
            let behavior = (0..answer.len()).map(|i| -0.01 * (at * 100 + i + 1) as f32).collect();
            (at, prompt, answer, behavior)
        })
        .collect()
}

/// A rollout record of the session's first `len` messages that supervises the answers at `supervised`, each with the
/// call's ids and log-probabilities; the answers at `prompted` also name the ids their request held.
fn record(len: usize, supervised: &[usize], calls: &[Call], prompted: &[usize]) -> ChatSample {
    let mut messages = session();
    messages.truncate(len);
    for (at, prompt, answer, behavior) in calls.iter().filter(|c| supervised.contains(&c.0)) {
        let m = messages[*at].as_object_mut().unwrap();
        m.insert("train".into(), true.into());
        m.insert("token_ids".into(), serde_json::json!(answer));
        m.insert("behavior_logprobs".into(), serde_json::json!(behavior));
        if prompted.contains(at) {
            m.insert("prompt_token_ids".into(), serde_json::json!(prompt));
        }
    }
    for m in &mut messages {
        m.as_object_mut().unwrap().entry("train").or_insert(false.into());
    }
    let line = serde_json::json!({"reward": 1.0, "group": "g", "rollout": "r", "messages": messages}).to_string();
    ChatSample::from_json_line_with(&line, &TextOnly::TRAINER).unwrap_or_else(|e| panic!("{line}: {e}"))
}

fn merged(template: &str, calls: &[Call], prompted: &[usize]) -> Result<EncodedSample, data::chat_template::TemplateError> {
    let tmpl = ChatTemplate::compile(template).unwrap();
    let answers: Vec<usize> = calls.iter().map(|c| c.0).collect();
    let packed = record(session().len(), &answers, calls, prompted).answers_packed(&tmpl, true).unwrap().expect("assistant answers");
    packed.encode_rollout(&tokenizer(), &tmpl, RenderOpts::default())
}

/// Through an append-only template, the merged record is the session's last request and answer, each call's answer
/// supervised where it was sampled with its own log-probabilities, and every call record is a prefix of it: the
/// merged sample holds exactly the per-call records' supervised tokens, each in the context its call was sampled in.
#[test]
fn an_append_only_session_merges_into_one_sample_that_lays_its_call_records_over_one_another() {
    let (tok, tmpl) = (tokenizer(), ChatTemplate::compile(KEEPS_HISTORY).unwrap());
    let calls = calls(KEEPS_HISTORY);
    let merged = merged(KEEPS_HISTORY, &calls, &[1, 3, 5]).expect("an append-only session merges");
    let mut union = vec![false; merged.ids.len()];
    let mut forwarded = 0;
    for (at, prompt, answer, behavior) in &calls {
        let one = record(at + 1, &[*at], &calls, &[*at]).answers_as_asked(&tmpl, true).unwrap().unwrap().remove(0).encode_rollout(&tok, &tmpl, RenderOpts::default()).expect("each call record is valid");
        forwarded += one.ids.len();
        assert_eq!(one.ids[..], merged.ids[..one.ids.len()], "call {at}'s record is a prefix of the merged sample");
        assert_eq!(one.ids[..prompt.len()], prompt[..], "call {at}'s context is its request");
        let from = prompt.len();
        assert!(merged.mask[from..from + answer.len()].iter().all(|&m| m) && one.mask[from..from + answer.len()].iter().all(|&m| m));
        assert_eq!(merged.behavior[from..from + answer.len()], behavior[..], "call {at}'s log-probabilities sit on its answer");
        union[from..from + answer.len()].iter_mut().for_each(|u| *u = true);
    }
    assert_eq!(merged.mask, union, "the merged sample supervises exactly the calls' answers");
    assert!(merged.ids.len() * 2 < forwarded, "the shared context is encoded once: {} tokens merged against {forwarded} over the call records", merged.ids.len());
}

/// A session whose next request holds an earlier answer otherwise than it was sampled (here without its reasoning) is
/// not append-only: the merged record is refused by rule, naming the first answer whose request does not extend the
/// session, so its producer sends it as one record per call (each of which is valid).
#[test]
fn a_session_re_rendered_between_calls_is_refused_as_not_append_only() {
    let (tok, tmpl) = (tokenizer(), ChatTemplate::compile(DROPS_REASONING).unwrap());
    let calls = calls(DROPS_REASONING);
    for (at, ..) in &calls {
        record(at + 1, &[*at], &calls, &[*at]).answers_as_asked(&tmpl, true).unwrap().unwrap().remove(0).encode_rollout(&tok, &tmpl, RenderOpts::default()).expect("each call record is valid on its own");
    }
    let why = merged(DROPS_REASONING, &calls, &[1, 3, 5]).expect_err("refused");
    let fault = why.fault().expect("a contract rule");
    assert_eq!((fault.rule, fault.field.as_str()), (RecordRule::SessionNotAppendOnly, "messages[3].prompt_token_ids"), "{why}");
    assert_eq!(RecordRule::SessionNotAppendOnly.name(), "session_not_append_only");
    assert!(why.to_string().contains("one record per call"), "{why}");
}

/// A merged record names the ids its requests held, and then every later answer that carries behaviour
/// log-probabilities must name them: without them nothing shows that the context the trainer renders is the one the
/// sampler saw. (A packed record that names none is the older packed form, whose history is the template's.)
#[test]
fn a_merged_answer_without_the_ids_its_request_held_is_refused() {
    let why = merged(KEEPS_HISTORY, &calls(KEEPS_HISTORY), &[1]).expect_err("refused");
    let fault = why.fault().expect("a contract rule");
    assert_eq!((fault.rule, fault.field.as_str()), (RecordRule::SessionNotAppendOnly, "messages[3].prompt_token_ids"), "{why}");
}
