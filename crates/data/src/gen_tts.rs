// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Synthetic **text -> codes** dataset for the Qwen3-TTS Talker (codebook-0).
//!
//! The Talker predicts codebook-0 acoustic tokens conditioned on text. Real
//! speech codes need the codec encoder; for a from-scratch / fine-tune *smoke*
//! we instead emit a tiny, fully-deterministic `text -> codes` mapping so a small
//! decoder can overfit it (loss must drop) without any audio.
//!
//! ## Layout (brain's standard large-vocab token stream)
//! One flat `u32` token stream written as `train.u32.bin` / `val.u32.bin`, plus
//! `meta.json` (`{"vocab_size":V,"token_width":32}`). The single shared vocab is
//! `[0, V)` with two contiguous regions: `[0, n_text)` are text tokens and
//! `[n_text, n_text+n_code)` are codebook-0 codes (offset by `n_text`).
//!
//! Each example is `BOS, text_id, code_0, code_1, …, code_{L-1}` where
//! `code_j = n_text + ((text_id * 7 + j * 3) % n_code)` is a deterministic
//! function of the text id — so an autoregressive next-token model can learn
//! `text -> codes` exactly. `BOS = 0` separates examples (and is a no-op text
//! token). This mirrors the real interleaved Talker stream (a text prefix then
//! its codec frames) at toy scale.
//!
//! The same generator also exposes [`codes_frame`] used by the multi-codebook
//! alignment tests in `crates/tts` (see `tts::sft`).

use std::io;
use std::path::Path;

use crate::binio;
use crate::rng::Rng;

/// Dataset shape knobs (kept tiny — this is a learnability smoke, not real TTS).
#[derive(Clone, Copy, Debug)]
pub struct TtsGenConfig {
    pub n_text: u32,   // text-token vocab
    pub n_code: u32,   // codebook-0 vocab
    pub frames: u32,   // codes per example (sequence length L)
    pub examples: u32, // number of (text -> codes) examples
}

impl Default for TtsGenConfig {
    fn default() -> Self {
        TtsGenConfig { n_text: 16, n_code: 48, frames: 12, examples: 4000 }
    }
}

impl TtsGenConfig {
    /// Total shared vocabulary: `BOS`-inclusive text region + the code region.
    pub fn vocab(&self) -> u32 {
        self.n_text + self.n_code
    }
}

/// Deterministic codebook-0 code (already vocab-offset by `n_text`) for text id
/// `text_id` at frame `j`. The single source of truth for the mapping, shared by
/// the writer and the tests.
pub fn code_token(cfg: &TtsGenConfig, text_id: u32, j: u32) -> u32 {
    cfg.n_text + ((text_id.wrapping_mul(7).wrapping_add(j.wrapping_mul(3))) % cfg.n_code)
}

/// Raw (un-offset) codebook-0 code for `(text_id, frame)` in `[0, n_code)` — the
/// value a Talker codec head would predict. Used by the alignment tests.
pub fn codes_frame(cfg: &TtsGenConfig, text_id: u32, j: u32) -> u32 {
    code_token(cfg, text_id, j) - cfg.n_text
}

/// Build the full token stream for `cfg.examples` examples with the given `seed`.
pub fn build_stream(cfg: &TtsGenConfig, seed: u64) -> Vec<u32> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity((cfg.examples * (cfg.frames + 2)) as usize);
    for _ in 0..cfg.examples {
        let text_id = 1 + ((rng.next_u64() as u32) % (cfg.n_text - 1)); // [1, n_text), reserve 0 for BOS
        out.push(0); // BOS / separator
        out.push(text_id);
        for j in 0..cfg.frames {
            out.push(code_token(cfg, text_id, j));
        }
    }
    out
}

/// Generate and write `train.u32.bin`, `val.u32.bin`, and `meta.json` into `dir`.
/// `examples` overrides `cfg.examples` when non-zero. Returns the vocab size.
pub fn write(dir: &Path, cfg: TtsGenConfig, seed: u64) -> io::Result<u32> {
    std::fs::create_dir_all(dir)?;
    let train = build_stream(&cfg, seed);
    // Validation reuses the same deterministic mapping (different sampling seed):
    // the model must generalise the text->codes rule, not memorise an order.
    let mut val_cfg = cfg;
    val_cfg.examples = (cfg.examples / 8).max(64);
    let val = build_stream(&val_cfg, seed ^ 0x9E37_79B9);

    binio::write_u32_bin(&dir.join("train.u32.bin"), &train)?;
    binio::write_u32_bin(&dir.join("val.u32.bin"), &val)?;
    std::fs::write(dir.join("meta.json"), binio::Meta::vocab_only(cfg.vocab() as usize))?;
    Ok(cfg.vocab())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_is_deterministic_and_in_vocab() {
        let cfg = TtsGenConfig { n_text: 8, n_code: 16, frames: 5, examples: 50 };
        let a = build_stream(&cfg, 7);
        let b = build_stream(&cfg, 7);
        assert_eq!(a, b, "same seed -> same stream");
        assert!(a.iter().all(|&t| t < cfg.vocab()), "token out of vocab");
        // Example layout: BOS(0), text_id in [1,n_text), then `frames` codes.
        assert_eq!(a[0], 0);
        let text_id = a[1];
        assert!((1..cfg.n_text).contains(&text_id));
        for j in 0..cfg.frames {
            assert_eq!(a[2 + j as usize], code_token(&cfg, text_id, j));
            assert!(a[2 + j as usize] >= cfg.n_text, "code not in code region");
        }
    }
}
