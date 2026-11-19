// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for exact off-policy reinforcement learning from the rollouts an agent
// collected, for its clients. If your team needs expertise in importance-weighted training on sampled agent
// trajectories then you can procure our services by sending an email to info@swedishembedded.com.

//! A supervised answer of a rollout trains the ids the sampler generated (`"token_ids"`), never the ids its text renders
//! to again: a model writes a tool call's arguments its own way (`{"a": 1}`) and the parsed message renders them another
//! (`{"a":1}`), and a behaviour log-probability belongs to the sampled token. The ids are admitted when they read back,
//! by the template's own reply format, as the message the record holds; the context the trainer renders before the
//! answer must be the ids the sampler was given (`"prompt_token_ids"`) when the record names them.

use data::chat::{ChatSample, RenderOpts, TextOnly};
use data::chat_rollout::RecordRule;
use data::chat_template::{parse_json_ordered, ChatTemplate};
use data::tokenizer::{CharTokenizer, Tokenizer};

/// A Qwen3-like template: the reasoning in a think block, a call as a JSON `<tool_call>` whose text arguments are
/// written verbatim, `<|im_end|>` closing each turn.
const TEMPLATE: &str = "{% for m in messages %}{% if m.role == 'assistant' %}<|im_start|>assistant\n<think>\n{{ m.reasoning_content or '' }}\n</think>\n\n{{ m.content }}\
{% for tc in m.tool_calls %}{% if loop.first and m.content %}\n{% endif %}<tool_call>\n{\"name\": \"{{ tc.function.name }}\", \"arguments\": \
{% if tc.function.arguments is string %}{{ tc.function.arguments }}{% else %}{{ tc.function.arguments | tojson }}{% endif %}}\n</tool_call>{% endfor %}<|im_end|>\n\
{% elif m.role == 'tool' %}<|im_start|>user\n<tool_response>\n{{ m.content }}\n</tool_response><|im_end|>\n\
{% else %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

const PROMPT: &str = "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n";

/// What the model generated: the arguments with a space after the colon.
const GENERATED: &str = "<think>\nr\n</think>\n\n<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call><|im_end|>";

fn tokenizer() -> CharTokenizer {
    CharTokenizer::from_corpus(&(32u8..127).map(char::from).chain(['\n']).collect::<String>())
}

fn template() -> ChatTemplate {
    ChatTemplate::compile(TEMPLATE).unwrap()
}

/// A rollout record whose one answer (reasoning `r`, a call of `f` with `arguments` as the parsed message holds them)
/// carries `generated` as its sampled ids, one log-probability per id, and `prompt` as its prompt ids when given.
fn record(arguments: &str, generated: &str, prompt: Option<&str>) -> ChatSample {
    let tok = tokenizer();
    let ids = tok.encode(generated);
    let lps: Vec<f32> = (0..ids.len()).map(|i| -0.01 * (i + 1) as f32).collect();
    let prompt = prompt.map(|p| format!(r#","prompt_token_ids":{}"#, serde_json::to_string(&tok.encode(p)).unwrap())).unwrap_or_default();
    let line = format!(
        r#"{{"reward":1.0,"group":"g","messages":[{{"role":"user","content":"q","train":false}},{{"role":"assistant","content":"","reasoning_content":"r","train":true,"tool_calls":[{{"id":"c","type":"function","function":{{"name":"f","arguments":{}}}}}],"token_ids":{},"behavior_logprobs":{}{prompt}}}]}}"#,
        serde_json::to_string(arguments).unwrap(),
        serde_json::to_string(&ids).unwrap(),
        serde_json::to_string(&lps).unwrap(),
    );
    ChatSample::from_json_line_with(&line, &TextOnly::TRAINER).unwrap_or_else(|e| panic!("{line}: {e}"))
}

fn encode(sample: &ChatSample) -> Result<data::chat::EncodedSample, data::chat_template::TemplateError> {
    let tmpl = template();
    let mut asked = sample.answers_as_asked(&tmpl, true).unwrap().expect("an assistant answer");
    asked.remove(0).encode_rollout(&tokenizer(), &tmpl, RenderOpts::default())
}

#[test]
fn a_sampled_answer_trains_its_own_ids_where_the_template_writes_its_text_differently() {
    let tok = tokenizer();
    let sample = record(r#"{"a":1}"#, GENERATED, Some(PROMPT));
    let messages = r#"[{"role":"user","content":"q"},{"role":"assistant","content":"","reasoning_content":"r","tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}]"#;
    let rendered = template().render(parse_json_ordered(messages).unwrap(), None, false, &Default::default()).unwrap();
    assert!(!rendered.contains(GENERATED), "the template writes the parsed answer otherwise than it was generated: {rendered:?}");

    let encoded = encode(&sample).expect("ids that read back as the answer are admitted");
    let sampled = tok.encode(GENERATED);
    let supervised: Vec<u32> = encoded.ids.iter().zip(&encoded.mask).filter(|(_, m)| **m).map(|(i, _)| *i).collect();
    assert_eq!(supervised, sampled, "the trained answer is the sampled ids, not the template's rendering of its text");
    let first = encoded.mask.iter().position(|m| *m).unwrap();
    assert_eq!(encoded.ids[..first], tok.encode(PROMPT)[..], "the context is the prompt the sampler was given");
    let placed: Vec<f32> = encoded.behavior.iter().zip(&encoded.mask).filter(|(_, m)| **m).map(|(b, _)| *b).collect();
    let recorded: Vec<f32> = (0..sampled.len()).map(|i| -0.01 * (i + 1) as f32).collect();
    assert_eq!(placed, recorded, "one behaviour log-probability per sampled id, in order");
    assert!(encoded.behavior.iter().zip(&encoded.mask).all(|(b, m)| *m != b.is_nan()));
}

#[test]
fn sampled_ids_that_read_back_as_another_answer_are_refused() {
    let other = GENERATED.replace("{\"a\": 1}", "{\"a\": 2}");
    let why = encode(&record(r#"{"a":1}"#, &other, None)).expect_err("refused");
    let fault = why.fault().expect("a contract rule");
    assert_eq!((fault.rule, fault.field.as_str()), (RecordRule::TokenIdsMismatch, "messages[1].token_ids"), "{why}");

    let other_reasoning = GENERATED.replace("<think>\nr\n", "<think>\ns\n");
    let why = encode(&record(r#"{"a":1}"#, &other_reasoning, None)).expect_err("refused");
    assert_eq!(why.fault().map(|f| f.rule), Some(RecordRule::TokenIdsMismatch), "{why}");
}

#[test]
fn a_context_other_than_the_prompt_the_sampler_was_given_is_refused_at_the_first_difference() {
    let given = format!("<|im_start|>system\nbe brief<|im_end|>\n{PROMPT}");
    let why = encode(&record(r#"{"a":1}"#, GENERATED, Some(&given))).expect_err("refused");
    let fault = why.fault().expect("a contract rule");
    assert_eq!((fault.rule, fault.field.as_str()), (RecordRule::PromptTokenIdsMismatch, "messages[1].prompt_token_ids"), "{why}");
    let at = "<|im_start|>".len();
    assert!(why.to_string().contains(&format!("position {at}")), "{why}");
}

#[test]
fn prompt_token_ids_come_with_the_answers_token_ids() {
    let line = r#"{"reward":1.0,"group":"g","messages":[{"role":"user","content":"q","train":false},{"role":"assistant","content":"a","train":true,"prompt_token_ids":[1,2]}]}"#;
    let why = ChatSample::from_json_line_with(line, &TextOnly::TRAINER).expect_err("refused").to_string();
    assert!(why.contains("messages[1]") && why.contains("prompt_token_ids") && why.contains("token_ids"), "{why}");
}
