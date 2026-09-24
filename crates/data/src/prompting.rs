// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! How a particular model expects to be asked something.
//!
//! A checkpoint that ships a chat template is an instruction-tuned model, and
//! handing one a bare question is not asking it a question. It is asking it to
//! CONTINUE a document that happens to end in a question mark, which is a
//! different task with a different answer distribution. Measured on
//! Qwen3-0.6B, the same prompt:
//!
//! ```text
//! raw:       "Q: <question>\nA: <command>\nAnswering the question in the
//!             format above." x18
//! templated: a correct, grounded answer
//! ```
//!
//! Nothing about the raw answer looks like an error. It is fluent text of the
//! right shape, so a scorer reports it as a wrong answer rather than as a
//! question that was never put - and every number taken through that path is
//! about the wrong task.
//!
//! So this is not a flag a caller opts into. [`Prompting::for_model_dir`]
//! reads what the model requires and applies it, and a caller that wants raw
//! continuation has to say so by name ([`Prompting::none`]).
//!
//! ## Thinking is off by default here
//!
//! A hybrid reasoning model (Qwen3 and later) spends its token budget in
//! `<think>` before answering, which for a 160-token evaluation means it
//! never reaches the answer at all. Evaluation and data generation want the
//! answer, deterministically and within a bounded budget, so
//! [`Prompting::for_model_dir`] disables it. A caller wanting the reasoning
//! trace turns it back on with [`Prompting::thinking`].
//!
//! Swedish Embedded AB builds evaluation and data-generation pipelines whose
//! numbers are about the question that was actually asked. If your team needs
//! expertise in making a model's measured behaviour match its deployed
//! behaviour, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::BTreeMap;
use std::path::Path;

use minijinja::Value;

use crate::chat_template::ChatTemplate;

/// What a model needs a question wrapped in before it will answer it.
pub struct Prompting {
    template: Option<ChatTemplate>,
    extra: BTreeMap<String, Value>,
    /// The token the model ends a turn with, from its own config. See
    /// [`Prompting::stop`].
    stop: Option<String>,
}

impl std::fmt::Debug for Prompting {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prompting").field("templated", &self.template.is_some()).finish_non_exhaustive()
    }
}

impl Prompting {
    /// Read it off the model's own directory.
    ///
    /// A directory carrying a chat template yields one that applies it. A
    /// directory without one yields [`Prompting::none`], because a base model
    /// genuinely is a continuation model - the absence is information, not a
    /// failure, and is not worth an error.
    pub fn for_model_dir(dir: &Path) -> Prompting {
        let stop = read_eos_token(dir);
        match ChatTemplate::from_model_dir(dir) {
            Ok(template) => Prompting { template: Some(template), extra: no_thinking(), stop },
            Err(_) => Prompting { stop, ..Prompting::none() },
        }
    }

    /// The string the model ends a turn with, from its own
    /// `tokenizer_config.json`.
    ///
    /// Part of the same requirement as the template, and missed for the same
    /// reason: generation that does not stop at it runs to its token budget
    /// and returns an answer with `<|im_end|>` glued to the end, which every
    /// exact-match scorer then marks wrong. The model DID answer; the caller
    /// did not stop listening.
    pub fn stop(&self) -> Option<&str> {
        self.stop.as_deref()
    }

    /// Raw continuation: the text is the prompt, unwrapped.
    ///
    /// Named rather than defaulted, so that a caller doing this has decided
    /// to rather than inherited it.
    pub fn none() -> Prompting {
        Prompting { template: None, extra: BTreeMap::new(), stop: None }
    }

    /// Turn the model's reasoning trace back on, for a caller that wants it
    /// and has the token budget for it.
    pub fn thinking(mut self, on: bool) -> Self {
        self.extra.insert("enable_thinking".to_string(), Value::from(on));
        self
    }

    /// Whether this model asked for a template. Reported so a run can record
    /// which of the two things it measured.
    pub fn is_templated(&self) -> bool {
        self.template.is_some()
    }

    /// Present `text` as a question, ready for the model to answer.
    ///
    /// Falls back to the bare text if the template refuses to render, rather
    /// than failing the call: a template that cannot render is a defect in
    /// the checkpoint, and the caller's alternative is no answer at all.
    /// [`Prompting::is_templated`] still reports true, so the run records
    /// that a template was expected.
    pub fn question(&self, text: &str) -> String {
        let Some(t) = &self.template else { return text.to_string() };
        let messages = Value::from_serialize(vec![BTreeMap::from([("role", "user"), ("content", text)])]);
        t.render(messages, None, true, &self.extra).unwrap_or_else(|_| text.to_string())
    }
}

/// `eos_token` as the model's own config states it, as a plain string or as
/// the `{"content": ...}` object the HuggingFace `AddedToken` form uses.
fn read_eos_token(dir: &Path) -> Option<String> {
    let raw = std::fs::read_to_string(dir.join("tokenizer_config.json")).ok()?;
    let cfg: serde_json::Value = serde_json::from_str(&raw).ok()?;
    let eos = cfg.get("eos_token")?;
    eos.as_str().map(str::to_string).or_else(|| eos.get("content")?.as_str().map(str::to_string))
}

/// The evaluation/generation default: answer, do not deliberate. See this
/// module's doc.
fn no_thinking() -> BTreeMap<String, Value> {
    BTreeMap::from([("enable_thinking".to_string(), Value::from(false))])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_with(template: Option<&str>) -> std::path::PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let d = std::env::temp_dir().join(format!(
            "brain-prompting-{}-{}",
            std::process::id(),
            N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("temp dir");
        let cfg = match template {
            Some(t) => serde_json::json!({ "chat_template": t }),
            None => serde_json::json!({}),
        };
        std::fs::write(d.join("tokenizer_config.json"), cfg.to_string()).expect("write");
        d
    }

    /// The whole point: a model that ships a template gets it, without the
    /// caller having asked.
    #[test]
    fn a_model_that_ships_a_template_has_it_applied_without_being_asked() {
        let d = dir_with(Some(
            "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        ));
        let p = Prompting::for_model_dir(&d);
        assert!(p.is_templated());
        let rendered = p.question("how do I graft?");
        assert_eq!(rendered, "<|im_start|>user\nhow do I graft?<|im_end|>\n<|im_start|>assistant\n");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// And a base model is left alone: the absence of a template is a fact
    /// about the model, not a failure to configure one.
    #[test]
    fn a_model_with_no_template_is_prompted_raw() {
        let d = dir_with(None);
        let p = Prompting::for_model_dir(&d);
        assert!(!p.is_templated());
        assert_eq!(p.question("how do I graft?"), "how do I graft?");
        assert_eq!(Prompting::none().question("x"), "x");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A hybrid reasoning model answers rather than deliberates, unless the
    /// caller says otherwise. Both halves, because a default that cannot be
    /// overridden is a different kind of problem.
    #[test]
    fn thinking_is_off_by_default_and_can_be_turned_back_on() {
        let tmpl = "{% if enable_thinking %}THINK{% else %}ANSWER{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}";
        let d = dir_with(Some(tmpl));
        assert!(Prompting::for_model_dir(&d).question("q").starts_with("ANSWER"));
        assert!(Prompting::for_model_dir(&d).thinking(true).question("q").starts_with("THINK"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The stop token is part of what the model requires, and missing it
    /// costs every exact match: the answer comes back correct with the end
    /// marker glued to it.
    #[test]
    fn the_models_own_end_of_turn_token_is_reported() {
        let d = dir_with(Some("{% for m in messages %}{{ m.content }}{% endfor %}"));
        std::fs::write(
            d.join("tokenizer_config.json"),
            serde_json::json!({ "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}", "eos_token": "<|im_end|>" }).to_string(),
        )
        .expect("write");
        assert_eq!(Prompting::for_model_dir(&d).stop(), Some("<|im_end|>"));

        // The AddedToken object form the HuggingFace configs also use.
        std::fs::write(
            d.join("tokenizer_config.json"),
            serde_json::json!({ "eos_token": { "content": "</s>", "lstrip": false } }).to_string(),
        )
        .expect("write");
        assert_eq!(Prompting::for_model_dir(&d).stop(), Some("</s>"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The mechanism is the HuggingFace convention, not one model's quirk:
    /// a Jinja `chat_template` in `tokenizer_config.json`, rendered with
    /// `messages` and `add_generation_prompt`. Every instruction-tuned model
    /// in that ecosystem ships one, and they look nothing like each other.
    ///
    /// Structurally real templates from four different families, each
    /// asserted to produce that family's actual wire format.
    #[test]
    fn any_family_s_template_is_applied_not_only_qwen_s() {
        let cases: [(&str, &str, &str); 4] = [
            (
                "qwen/chatml",
                "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
                "<|im_start|>user\nhow?<|im_end|>\n<|im_start|>assistant\n",
            ),
            (
                "llama-3",
                "{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}",
                "<|start_header_id|>user<|end_header_id|>\n\nhow?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            ),
            (
                "mistral",
                "{% for m in messages %}{% if m.role == 'user' %}[INST] {{ m.content }} [/INST]{% endif %}{% endfor %}",
                "[INST] how? [/INST]",
            ),
            (
                "gemma",
                "{% for m in messages %}<start_of_turn>{{ m.role }}\n{{ m.content }}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}",
                "<start_of_turn>user\nhow?<end_of_turn>\n<start_of_turn>model\n",
            ),
        ];
        for (family, template, expected) in cases {
            let d = dir_with(Some(template));
            let p = Prompting::for_model_dir(&d);
            assert!(p.is_templated(), "{family}: a model shipping a template must be detected as needing one");
            assert_eq!(p.question("how?"), expected, "{family}");
            let _ = std::fs::remove_dir_all(&d);
        }
    }

    /// `enable_thinking` is one family's template kwarg, and it is passed to
    /// every render. A template that has never heard of it must be
    /// unaffected rather than fail - otherwise turning thinking off for
    /// Qwen would break every other model.
    #[test]
    fn a_template_that_does_not_know_the_thinking_kwarg_is_unaffected_by_it() {
        let d = dir_with(Some("{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}"));
        assert_eq!(Prompting::for_model_dir(&d).question("how?"), "[INST] how? [/INST]");
        assert_eq!(Prompting::for_model_dir(&d).thinking(true).question("how?"), "[INST] how? [/INST]");
        let _ = std::fs::remove_dir_all(&d);
    }

    /// The template may also arrive as a standalone file, which is the other
    /// shape the ecosystem ships it in.
    #[test]
    fn a_template_in_its_own_file_is_found_too() {
        let d = dir_with(None);
        std::fs::write(d.join("chat_template.jinja"), "{% for m in messages %}<s>{{ m.content }}</s>{% endfor %}").expect("write");
        let p = Prompting::for_model_dir(&d);
        assert!(p.is_templated(), "a chat_template.jinja beside the config is still a chat template");
        assert_eq!(p.question("how?"), "<s>how?</s>");
        let _ = std::fs::remove_dir_all(&d);
    }
}
