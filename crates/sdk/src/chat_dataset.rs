// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Check a packed chat dataset before paying for a training run.
//!
//! `qwen3 finetune --lora` consumes a directory of `generic-messages-v2`
//! JSONL: one packed conversation per line, `train` per message deciding
//! which spans are supervised. The parser behind it is strict on purpose - a
//! missing, mistyped or unexpected field is a hard failure naming the line and
//! the field rather than a default that trains silently on the wrong thing -
//! but until now the only way to find out was to start the run.
//!
//! That is the wrong moment. A dataset is usually produced by something other
//! than the thing that trains on it, and the producer's mistakes are cheapest
//! to find before a checkpoint is loaded and a device is claimed. This is the
//! same seam [`crate::DocumentStudy::validate_dataset`] provides for fact
//! batches, for the other dataset shape, and it runs the SAME parser the
//! trainer runs so the two cannot come to disagree about what is valid.
//!
//! ```no_run
//! # fn demo() -> Result<(), String> {
//! let summary = brain::validate_chat_dataset("dataset/train.jsonl")?;
//! println!("{} conversation(s), {} supervised turn(s)", summary.records, summary.trained_messages);
//! # Ok(()) }
//! ```
//!
//! Swedish Embedded AB implements training data pipelines whose failures
//! surface at the producer rather than three hours into a GPU run, for its
//! clients. If your team needs expertise in dataset validation or supervised
//! fine-tuning pipelines, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

use data::chat::ChatSample;

/// What a dataset file contains, once it has been shown to parse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChatDatasetSummary {
    /// Packed conversations - one per line.
    pub records: usize,
    /// Messages across every conversation.
    pub messages: usize,
    /// Messages marked `train: true`. **Zero is reported, not an error**: it
    /// is a well-formed dataset that would train on nothing, which is a
    /// producer bug rather than a parse failure, and the caller is better
    /// placed to decide how loudly to complain.
    pub trained_messages: usize,
    /// Conversations carrying a tool schema.
    pub records_with_tools: usize,
}

/// Parse `path` exactly as the trainer would, and report what is in it.
///
/// Errors carry the file, the line number and the offending field, because a
/// dataset is usually machine-produced and "invalid" is not an actionable
/// thing to tell whoever produced it.
pub fn validate_chat_dataset(path: impl AsRef<Path>) -> Result<ChatDatasetSummary, String> {
    let path = path.as_ref();
    let samples = ChatSample::from_jsonl(path).map_err(|e| e.to_string())?;

    let mut summary = ChatDatasetSummary { records: samples.len(), ..Default::default() };
    for sample in &samples {
        summary.messages += sample.messages.len();
        summary.trained_messages += sample.messages.iter().filter(|m| m.train).count();
        if !sample.tools.is_empty() {
            summary.records_with_tools += 1;
        }
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-chat-dataset-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(name);
        std::fs::write(&path, body).expect("write");
        path
    }

    const ONE: &str = r#"{"messages":[{"role":"user","content":"hi","train":false},{"role":"assistant","content":"hello","train":true}],"tools":[]}"#;

    #[test]
    fn a_well_formed_file_reports_what_is_in_it() {
        let path = write("ok.jsonl", &format!("{ONE}\n{ONE}\n"));
        let summary = validate_chat_dataset(&path).expect("valid");
        assert_eq!(summary.records, 2);
        assert_eq!(summary.messages, 4);
        assert_eq!(summary.trained_messages, 2);
    }

    #[test]
    fn a_message_with_no_supervision_boundary_is_a_parse_error_naming_the_line() {
        // `train` is required on every message: a record without it is either
        // a silent no-op or a silent prompt leak into the loss, and both are
        // worse than a refusal.
        let path = write("no-train.jsonl", r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        let err = validate_chat_dataset(&path).expect_err("must not parse");
        assert!(err.contains("train"), "the error must name the missing field: {err}");
    }

    #[test]
    fn a_dataset_that_would_train_on_nothing_is_reported_rather_than_refused() {
        // Well-formed and useless. The producer needs to know; the parser is
        // not the place to decide how badly.
        let path = write(
            "untrained.jsonl",
            r#"{"messages":[{"role":"user","content":"hi","train":false}],"tools":[]}"#,
        );
        let summary = validate_chat_dataset(&path).expect("it parses");
        assert_eq!(summary.records, 1);
        assert_eq!(summary.trained_messages, 0);
    }
}
