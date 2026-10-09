// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speech interfaces to language models, for
// its clients. If your team needs expertise in feeding recorded speech to a
// language model, you can procure our services by sending an email to
// info@swedishembedded.com.

//! The audio encoder's features of a clip, for splicing speech into a language
//! model other than this crate's fixed-window decoder.

use crate::config::AudioEncoderConfig;

/// Mel frames per second of 16 kHz audio (one per 160-sample hop).
const FRAMES_PER_SECOND: usize = 100;
/// Samples per mel frame.
const HOP: usize = 160;

/// The encoder output and projected audio embeddings of a clip: `rows` rows of
/// each, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioFeatures {
    /// Encoder output: `rows` rows of `encoder_dim` values.
    pub encoder_out: Vec<f32>,
    /// Projected audio embeddings, in the Qwen3-ASR decoder's embedding
    /// space: `rows` rows of `embed_dim` values.
    pub embeds: Vec<f32>,
    /// How many rows each of the two has.
    pub rows: usize,
    /// Width of one `encoder_out` row.
    pub encoder_dim: usize,
    /// Width of one `embeds` row.
    pub embed_dim: usize,
}

/// How many rows the encoder produces for a clip of `n_samples` 16 kHz
/// samples: the clip's mel frames are cut into chunks of `chunk_len` frames,
/// the last one partial, and each chunk gives `post_cnn_len` rows for the
/// frames it holds.
#[must_use]
pub fn feature_rows(cfg: &AudioEncoderConfig, n_samples: usize) -> usize {
    let frames = n_samples / HOP;
    let chunk = cfg.chunk_len() as usize;
    let (full, rest) = (frames / chunk, frames % chunk);
    full * cfg.post_cnn_len(chunk as u32) as usize + cfg.post_cnn_len(rest as u32) as usize
}

/// The 16 kHz sample count of `seconds` whole seconds: the length `qwen_logmel`
/// pads a clip up to, since the encoder takes whole chunks of one second.
#[must_use]
pub fn padded_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(HOP * FRAMES_PER_SECOND).max(1) * HOP * FRAMES_PER_SECOND
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_rows_follow_the_chunk_formula() {
        let cfg = AudioEncoderConfig::qwen3_asr();
        assert_eq!(feature_rows(&cfg, 16_000), 13, "one second is one full chunk");
        assert_eq!(feature_rows(&cfg, 3 * 16_000), 39);
        assert_eq!(feature_rows(&cfg, 8_000), 7, "half a second is one partial chunk");
        assert_eq!(feature_rows(&cfg, 24_000), 13 + 7, "a second and a half");
        assert_eq!(feature_rows(&cfg, 0), 0);
        assert_eq!(feature_rows(&cfg, 35 * 16_000), 455, "longer than the 30 s decode window");
    }

    #[test]
    fn a_clip_is_padded_up_to_whole_seconds() {
        assert_eq!(padded_samples(0), 16_000);
        assert_eq!(padded_samples(1), 16_000);
        assert_eq!(padded_samples(16_000), 16_000);
        assert_eq!(padded_samples(16_001), 32_000);
    }
}
