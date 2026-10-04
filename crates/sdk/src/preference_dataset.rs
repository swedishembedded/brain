// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Check a `generic-preference-v1` dataset before paying for a training run.
//!
//! The preference counterpart of [`crate::validate_chat_dataset`]: the same
//! parser [`crate::PreferenceFineTune`] trains through (`data::preference`),
//! run without a device, with errors naming the file, the record and the
//! field.
//!
//! ```no_run
//! # fn demo() -> Result<(), String> {
//! let summary = brain::validate_preference_dataset_for("pairs.jsonl", "/models/qwen3-0.6b", Some(512))?;
//! println!("{} pair(s), longest candidate {:?} tokens", summary.pairs, summary.longest_tokens);
//! # Ok(()) }
//! ```
//!
//! Swedish Embedded AB implements training data pipelines whose failures
//! surface at the producer rather than into a GPU run, for its clients. If
//! your team needs expertise in preference datasets or preference
//! fine-tuning, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::path::Path;

use data::chat_template::ChatTemplate;
use data::chat::RenderOpts;
use data::preference::{EncodedTurn, PreferenceSample};
use data::qwen_tokenizer::QwenBpe;

/// What a preference dataset contains, once it has been shown to parse.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreferenceDatasetSummary {
    /// Preference pairs - one per line.
    pub pairs: usize,
    /// Prompt messages across every pair.
    pub prompt_messages: usize,
    /// Pairs carrying a tool schema.
    pub pairs_with_tools: usize,
    /// The longest rendered candidate (prompt plus answer), in tokens;
    /// `None` when the pairs were only parsed, not rendered.
    pub longest_tokens: Option<usize>,
}

/// Parse `path` exactly as the trainer would, and report what is in it.
pub fn validate_preference_dataset(path: impl AsRef<Path>) -> Result<PreferenceDatasetSummary, String> {
    let pairs = PreferenceSample::from_jsonl(path.as_ref()).map_err(|e| e.to_string())?;
    Ok(summarize(&pairs, None))
}

/// Parse `path` and render both candidates of every pair through the
/// checkpoint in `model_dir`'s own tokenizer and chat template, the way
/// training will. Beyond parsing, this refuses a pair whose candidate does
/// not render under the template, whose candidate has no supervised token,
/// whose two candidates render to the same supervised tokens (it would state
/// no preference to this model), or - with `max_block` - whose candidate is
/// longer than that many tokens.
pub fn validate_preference_dataset_for(path: impl AsRef<Path>, model_dir: impl AsRef<Path>, max_block: Option<u32>) -> Result<PreferenceDatasetSummary, String> {
    let model_dir = model_dir.as_ref();
    let tmpl = ChatTemplate::from_model_dir(model_dir).map_err(|e| format!("{}: could not load the chat template: {e}", model_dir.display()))?;
    let tok_path = model_dir.join("tokenizer.json");
    let tok = QwenBpe::from_file(tok_path.to_str().unwrap_or_default()).map_err(|e| format!("{}: could not load the tokenizer: {e}", tok_path.display()))?;
    check_pairs(path.as_ref(), &tok, &tmpl, max_block, RenderOpts::default()).map(|(summary, _)| summary)
}

/// [`validate_preference_dataset_for`] with the tokenizer and template
/// already loaded, handing back the rendered pairs so a caller does not
/// render them twice.
pub(crate) fn check_pairs(path: &Path, tok: &QwenBpe, tmpl: &ChatTemplate, max_block: Option<u32>, render: RenderOpts) -> Result<(PreferenceDatasetSummary, Vec<(EncodedTurn, EncodedTurn)>), String> {
    let pairs = PreferenceSample::from_jsonl(path).map_err(|e| e.to_string())?;
    let mut encoded = Vec::with_capacity(pairs.len());
    let mut longest = 0usize;
    for (index, pair) in pairs.iter().enumerate() {
        let record = || format!("{}: record {}", path.display(), index + 1);
        let (chosen, rejected) = pair.encode_with(tok, tmpl, render).map_err(|e| format!("{} cannot be rendered for training: {e}", record()))?;
        for (name, turn) in [("chosen", &chosen), ("rejected", &rejected)] {
            if !turn.mask.iter().any(|m| *m) {
                return Err(format!("{}: \"{name}\" renders to no supervised token", record()));
            }
            if let Some(max) = max_block.filter(|&max| turn.ids.len() > max as usize) {
                return Err(format!("{}: \"{name}\" is {} tokens, past max_block {max}; raise max_block or shorten the record", record(), turn.ids.len()));
            }
            longest = longest.max(turn.ids.len());
        }
        if chosen.supervised() == rejected.supervised() {
            return Err(format!("{}: \"chosen\" and \"rejected\" render to the same tokens under this model's tokenizer, so the pair states no preference", record()));
        }
        encoded.push((chosen, rejected));
    }
    Ok((summarize(&pairs, Some(longest)), encoded))
}

fn summarize(pairs: &[PreferenceSample], longest_tokens: Option<usize>) -> PreferenceDatasetSummary {
    PreferenceDatasetSummary {
        pairs: pairs.len(),
        prompt_messages: pairs.iter().map(PreferenceSample::prompt_len).sum(),
        pairs_with_tools: pairs.iter().filter(|p| !p.chosen.tools.is_empty()).count(),
        longest_tokens,
    }
}
