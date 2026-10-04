// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements tooling that lets a training loop see the
// exact tokens a deployed chat model sees, for its clients. If your team needs
// expertise in training against verifiers on the same prompts that serving
// renders, you can procure our services by sending an email to
// info@swedishembedded.com.

//! Token ids of a chat prompt, and back.
//!
//! A loop that trains or scores on token ids (a reinforcement-learning
//! environment hands a policy its prompt as ids and reads a verdict off the ids
//! it generates) must feed the model the tokens serving would. They are the
//! rendering [`ChatRequest::render_prompt`] already makes, tokenized by the
//! checkpoint's own tokenizer: the same text, the same ids, with no second
//! template to drift from the first.

use std::path::Path;

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;

use crate::chat::ChatRequest;
use crate::{Error, Result};

/// A checkpoint's tokenizer, for chat prompts.
pub struct ChatTokenizer {
    tok: QwenBpe,
}

impl ChatTokenizer {
    /// The tokenizer of the checkpoint directory `dir`.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<ChatTokenizer> {
        let dir = dir.as_ref();
        let utf8 = dir
            .to_str()
            .ok_or_else(|| Error::Backend(format!("{}: not a UTF-8 path", dir.display())))?;
        let tok = QwenBpe::from_dir(utf8)
            .map_err(|e| Error::Backend(format!("{}: {e}", dir.display())))?;
        Ok(ChatTokenizer { tok })
    }

    /// The tokenizer described by the bytes of a `tokenizer.json`.
    pub fn from_json(bytes: &[u8]) -> Result<ChatTokenizer> {
        QwenBpe::from_json_bytes(bytes)
            .map(|tok| ChatTokenizer { tok })
            .map_err(|e| Error::Backend(format!("tokenizer.json: {e}")))
    }

    /// The ids of the prompt `request` renders to: what the model is fed.
    pub fn prompt_ids(&self, request: &ChatRequest) -> Result<Vec<u32>> {
        Ok(self.tok.encode(&request.render_prompt()?))
    }

    /// The text of `ids`.
    pub fn decode(&self, ids: &[u32]) -> String {
        self.tok.decode(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::ChatMessage;

    fn minimal() -> ChatTokenizer {
        let json = serde_json::json!({
            "model": { "vocab": {"a": 0u32, "b": 1u32}, "merges": [] },
            "added_tokens": [{"content": "<pad>", "id": 2u32}],
        });
        ChatTokenizer::from_json(json.to_string().as_bytes()).unwrap()
    }

    #[test]
    fn a_tokenizer_is_built_from_the_bytes_of_its_json_and_a_bad_one_is_refused() {
        let _ = minimal();
        assert!(ChatTokenizer::from_json(b"not json").is_err());
    }

    #[test]
    fn ids_decode_to_their_text() {
        let t = minimal();
        assert_eq!(t.decode(&[0, 1, 0]), "aba");
        assert_eq!(t.decode(&[]), "");
    }

    #[test]
    fn a_missing_checkpoint_directory_is_an_error_naming_it() {
        let err = ChatTokenizer::from_model_dir("/nonexistent/checkpoint")
            .err()
            .unwrap()
            .to_string();
        assert!(err.contains("/nonexistent/checkpoint"), "{err}");
    }

    /// The tokenizer of a real checkpoint, from `QWEN_TOKENIZER` (the path of a
    /// `tokenizer.json`); the spec is skipped where there is none.
    fn real() -> Option<ChatTokenizer> {
        let path = std::env::var("QWEN_TOKENIZER").ok()?;
        ChatTokenizer::from_json(&std::fs::read(path).ok()?).ok()
    }

    #[test]
    fn a_chat_prompt_is_the_rendered_prompt_tokenized_and_decodes_back_to_it() {
        let Some(t) = real() else {
            brain_testutil::skip("QWEN_TOKENIZER unset");
            return;
        };
        let request = ChatRequest::new(vec![
            ChatMessage::system("Answer briefly."),
            ChatMessage::user("What is the capital of France?"),
        ])
        .thinking(false);
        let ids = t.prompt_ids(&request).unwrap();
        let text = request.render_prompt().unwrap();
        assert!(!ids.is_empty());
        assert_eq!(
            t.decode(&ids),
            text,
            "the ids are exactly the rendered prompt"
        );
        assert!(
            text.contains("What is the capital of France?") && text.contains("Answer briefly.")
        );
    }

    #[test]
    fn the_same_request_gives_the_same_ids_and_a_different_one_does_not() {
        let Some(t) = real() else {
            brain_testutil::skip("QWEN_TOKENIZER unset");
            return;
        };
        let ask = |q: &str| {
            t.prompt_ids(&ChatRequest::new(vec![ChatMessage::user(q)]).thinking(false))
                .unwrap()
        };
        assert_eq!(ask("one"), ask("one"));
        assert_ne!(ask("one"), ask("two"));
    }
}
