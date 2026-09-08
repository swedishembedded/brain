// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The prompt: `BOS ++ before ++ <image>*n_rows ++ after`, off the LM
//! checkpoint's own tokenizer.
//!
//! Deliberately smaller than v1's `crates/deepseek2ocr/src/prompt.rs`: v2's
//! image block has no per-row newline structure to reproduce in the token
//! stream, so EVERY row of the spliced block - every local tile's rows, the
//! global view's rows, and the one separator row - shares the same `<image>`
//! placeholder id. The splice (`model::vlm::splice_fwd`) overwrites all of
//! their embeddings regardless of which checkpoint tensor produced the row,
//! so the placeholder id itself carries no information the decoder ever
//! reads.
//!
//! Swedish Embedded AB builds from-scratch GPU inference stacks for
//! vision-language models. If your team needs a document-understanding model
//! ported without a PyTorch dependency in the loop, you can procure our
//! services by emailing info@swedishembedded.com.

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;

/// The reserved marker every image row (tile, global view, and the
/// separator alike) is filled with in the token stream.
pub const IMAGE: &str = "<image>";

/// The LM checkpoint's own tokenizer - the exact same file `deepseek2::import`
/// reads its decoder from, so a served path and this module never disagree
/// about which vocabulary is in play.
pub fn tokenizer_from_gguf(lm_path: &str) -> Result<QwenBpe, String> {
    let mg = checkpoint::gguf::MmapGguf::open(lm_path)?;
    let gt = mg.tokenizer().ok_or_else(|| format!("{lm_path}: no tokenizer KV block"))?;
    QwenBpe::from_gguf(&gt)
}

/// One assembled prompt: the full id sequence, and where the image block
/// (every row a caller's [`crate::rows::RowPlan`] describes) sits in it.
pub struct Prompt {
    pub ids: Vec<u32>,
    pub row0: u32,
    pub n_rows: u32,
}

impl Prompt {
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn image_run(&self) -> (u32, u32) {
        (self.row0, self.n_rows)
    }
}

/// `BOS ++ before ++ <image>*n_rows ++ after`. Fails if the tokenizer's own
/// vocabulary has no `<image>` marker - a checkpoint whose reserved-token
/// inventory moved is a real error, not a silent fallback.
pub fn build_prompt(tok: &QwenBpe, before: &str, after: &str, n_rows: u32) -> Result<Prompt, String> {
    let image = tok.special_id(IMAGE).ok_or_else(|| format!("tokenizer has no {IMAGE:?} marker"))?;
    let bos = tok.special_id(BOS).ok_or_else(|| "tokenizer has no BOS marker".to_string())?;

    let mut ids = Vec::with_capacity(1 + before.len() + n_rows as usize + after.len());
    ids.push(bos);
    ids.extend(tok.encode(before));
    let row0 = ids.len() as u32;
    ids.extend(std::iter::repeat_n(image, n_rows as usize));
    ids.extend(tok.encode(after));
    Ok(Prompt { ids, row0, n_rows })
}

/// The reserved BOS marker's own text, so [`build_prompt`] can look its id up
/// through the same `special_id` path as every other marker rather than
/// hardcoding `0`. The shape (length, image-run placement against
/// `before`/`after`) is exercised end to end by `tests/prompt_real.rs`
/// against the real checkpoint's own vocabulary - a hand-built stand-in
/// tokenizer here would only duplicate that with a fixture of its own.
pub const BOS: &str = "<｜begin▁of▁sentence｜>";
