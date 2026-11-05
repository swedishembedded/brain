// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training data pipelines that size every
// record against the row the trainer will use, for its clients. If your team
// needs expertise in fitting conversations to a fine-tuning block, you can
// procure our services by sending an email to info@swedishembedded.com.

//! How long a training record is, as the chat fine-tune measures it.
//!
//! The trainer refuses a record longer than its row (`max_block`) after the
//! checkpoint is loaded. A producer that wants to build records which fit, or a
//! caller that wants to refuse an overlong one before claiming a device, needs
//! the trainer's own length, not an estimate from the record's text: the chat
//! template's markup, the shape each record takes (one example per answer, or
//! packed), and the rows an image adds all count. [`ExampleMeter`] reads a
//! record with the code [`crate::ChatFineTune`] reads its dataset with and
//! returns the length of its longest example.

use std::path::Path;

use data::chat::{ChatSample, ImageReader, RenderOpts};
use data::chat_template::ChatTemplate;
use data::tokenizer::Tokenizer;

use crate::finetune::Asked;
use crate::{Error, Result};

/// Measures `generic-messages-v2` records for one base, as its fine-tune would
/// train them.
pub struct ExampleMeter {
    tok: Box<dyn Tokenizer>,
    tmpl: ChatTemplate,
    no_think: Option<String>,
    render: RenderOpts,
    pack: bool,
    images: Box<dyn ImageReader>,
}

impl ExampleMeter {
    /// The meter of the checkpoint in `model_dir`, with the settings of the
    /// fine-tune it measures for: `thinking` and `pack_answers` are
    /// [`crate::ChatFineTune::thinking`] and [`crate::ChatFineTune::pack_answers`]
    /// (the defaults of a fine-tune are `true` and `false`).
    pub fn for_model(model_dir: impl AsRef<Path>, thinking: bool, pack_answers: bool) -> Result<Self> {
        let model_dir = model_dir.as_ref();
        let (tok, tmpl) = crate::finetune::tokenizer_and_template(model_dir)?;
        let arch = crate::study::arch_of_checkpoint(model_dir).map_or("qwen3", |a| a.id);
        let images = crate::finetune::image_reader(arch, model_dir)?;
        Ok(Self::from_parts(Box::new(tok), tmpl, images, thinking, pack_answers))
    }

    pub(crate) fn from_parts(tok: Box<dyn Tokenizer>, tmpl: ChatTemplate, images: Box<dyn ImageReader>, thinking: bool, pack: bool) -> Self {
        let no_think = if thinking { None } else { tmpl.no_think_block() };
        let render = RenderOpts { keep_reasoning: no_think.is_some() };
        Self { tok, tmpl, no_think, render, pack, images }
    }

    /// The tokens of the longest example the dataset line `record` is trained
    /// as, which is what the training row has to hold. A picture counts the
    /// rows the base's vision tower gives it.
    ///
    /// # Errors
    /// The line is not a record the trainer reads, or it cannot be encoded.
    pub fn tokens(&self, record: &str) -> Result<usize> {
        let invalid = |e: String| Error::Backend(format!("record cannot be measured: {e}"));
        let sample = ChatSample::from_json_line_with(record, &*self.images).map_err(invalid)?;
        let asked = Asked { tok: &*self.tok, tmpl: &self.tmpl, no_think: self.no_think.as_deref(), render: self.render, pack: self.pack, images: &*self.images };
        let mut longest = 0;
        for example in asked.examples_of(&sample).map_err(invalid)? {
            let encoded = example.encode_rollout(&*self.tok, &self.tmpl, self.render).map_err(|e| invalid(e.to_string()))?;
            // A teacher's rendering is scored in a row of its own, so the row has to hold it too.
            longest = longest.max(encoded.ids.len()).max(encoded.teacher.map_or(0, |(ids, _)| ids.len()));
        }
        Ok(longest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIALOGUE: &str = r#"{"messages":[{"role":"user","content":"q1","train":false},{"role":"assistant","content":"a1","train":true},{"role":"user","content":"q2 is longer","train":false},{"role":"assistant","content":"a2","train":true}]}"#;
    const WITH_PICTURE: &str = r#"{"messages":[{"role":"user","content":[{"type":"text","text":"see"},{"type":"image_url","image_url":{"url":"fake:one"}}],"train":false},{"role":"assistant","content":"a","train":true}]}"#;

    /// Reads `fake:<name>` URLs; an image stands for three `~` rows.
    struct Pictures;

    impl ImageReader for Pictures {
        fn image(&self, url: &str) -> std::result::Result<(data::chat::ImageBytes, String), String> {
            let name = url.strip_prefix("fake:").ok_or_else(|| format!("{url} is not a picture"))?;
            Ok((name.as_bytes().into(), "~~~".to_string()))
        }
        fn check_text(&self, _text: &str) -> std::result::Result<(), String> {
            Ok(())
        }
        fn placeholder_token(&self) -> Option<u32> {
            None
        }
    }

    fn meter(thinking: bool, pack: bool) -> ExampleMeter {
        let tmpl = ChatTemplate::compile(&std::fs::read_to_string(brain_testutil::testdata("qwen3/chat_template.jinja")).unwrap()).unwrap();
        let chars: String = (32u8..127).map(char::from).chain(['\n']).collect();
        let tok = data::tokenizer::CharTokenizer::from_corpus(&chars);
        ExampleMeter::from_parts(Box::new(tok), tmpl, Box::new(Pictures), thinking, pack)
    }

    /// A record with a teacher context is measured by the longer of its two renderings: the row holds the teacher's too.
    #[test]
    fn a_teacher_context_counts_toward_the_length_of_a_record() {
        let plain = r#"{"messages":[{"role":"user","content":"q","train":false},{"role":"assistant","content":"a","train":true}]}"#;
        let taught = r#"{"teacher_context":{"at":1,"messages":[{"role":"user","content":"the hidden check wants a"}]},"messages":[{"role":"user","content":"q","train":false},{"role":"assistant","content":"a","train":true}]}"#;
        let m = meter(true, false);
        let (alone, with_teacher) = (m.tokens(plain).unwrap(), m.tokens(taught).unwrap());
        assert!(with_teacher > alone + "the hidden check wants a".len(), "{alone} -> {with_teacher}");
    }

    /// The meter's length is the longest example `Asked` yields, encoded: the
    /// template's markup is in it, so it exceeds the record's own text.
    #[test]
    fn the_length_is_the_longest_encoded_example_markup_included() {
        let m = meter(false, false);
        let n = m.tokens(DIALOGUE).unwrap();
        let text_chars = "q1a1q2 is longera2".len();
        assert!(n > text_chars + 10, "{n} must include the template markup");
        let sample = ChatSample::from_json_line_with(DIALOGUE, &Pictures).unwrap();
        let asked = Asked { tok: &*m.tok, tmpl: &m.tmpl, no_think: m.no_think.as_deref(), render: m.render, pack: false, images: &Pictures };
        let longest = asked.examples_of(&sample).unwrap().iter().map(|e| e.encode_with(&*m.tok, &m.tmpl, m.render).unwrap().0.len()).max().unwrap();
        assert_eq!(n, longest);
    }

    /// A dialogue is one example per answer unless packed, so packing it makes
    /// the example (and the row it needs) longer.
    #[test]
    fn packing_a_dialogue_lengthens_its_example() {
        assert!(meter(false, true).tokens(DIALOGUE).unwrap() > meter(false, false).tokens(DIALOGUE).unwrap());
    }

    /// A picture adds the rows its reader gives it.
    #[test]
    fn a_picture_adds_its_rows() {
        let with = meter(true, false).tokens(WITH_PICTURE).unwrap();
        let without = meter(true, false).tokens(&WITH_PICTURE.replace(r#",{"type":"image_url","image_url":{"url":"fake:one"}}"#, "")).unwrap();
        assert_eq!(with - without, 3);
    }

    /// A line the trainer would not read is an error naming why, not a length.
    #[test]
    fn a_line_the_trainer_would_refuse_is_an_error() {
        let err = meter(true, false).tokens(r#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap_err().to_string();
        assert!(err.contains("train"), "{err}");
    }
}
