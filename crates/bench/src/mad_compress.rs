// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MAD **compression** — *unsupported placeholder*.
//!
//! The MAD compression task asks a model to squeeze an input sequence into a
//! **single** fixed representation (one vector / one token slot) from which a
//! separate MLP head must **reconstruct the whole sequence**. It is an
//! autoencoder objective: encode `x_1..x_T → z` (one bottleneck), decode
//! `z → x_1..x_T`, loss = reconstruction error over *all* positions.
//!
//! ## Why it does not fit today's engine
//! Every other benchmark here trains through [`gpt::train`], a **causal
//! next-token decoder**: the loss at position `t` predicts `x_{t+1}` from the
//! prefix `x_1..x_t`, and the only loss-shaping lever is *masking* a position out
//! (the `mask_before` / `mask_per_line` recipe). Compression needs something this
//! API cannot express:
//!
//! 1. **A bottleneck.** Reconstruction must flow *only* through a single
//!    representation `z`, not through the full causal context. There is no way to
//!    tell the decoder "you may look at `z` but not at the original tokens" — a
//!    causal LM at position `t` always sees `x_1..x_t` directly, so it can copy
//!    rather than compress. Masking removes a *target*, not the model's *access*
//!    to inputs.
//! 2. **A non-next-token objective.** The loss is reconstruct-all-from-`z`, not
//!    predict-the-next-token. `gpt::train` only ever computes next-token CE.
//! 3. **A separate decoder/MLP head** distinct from the encoder, with its own
//!    forward/backward — the engine wires exactly one GPT graph end to end.
//!
//! Forcing this into the next-token frame (e.g. "append the sequence again after
//! a separator and let the model copy it") would measure *copying*, not
//! *compression*, because the decoder still sees the original tokens. The task
//! prompt explicitly says **do not force it** — so this module is a deliberate,
//! documented placeholder.
//!
//! ## What it needs
//! The existing model seam — [`DecoderLm`](crate::model::DecoderLm) — abstracts a
//! *causal next-token decoder* (train on a token dataset, read per-position
//! logits), which is exactly the shape every other benchmark needs and now
//! shares. Compression is a different shape: an **encoder → bottleneck → decoder
//! MLP** trained on a *reconstruct-all-from-`z`* objective, not next-token CE.
//! That needs a **second model trait** (e.g. an `Autoencoder` trait with its own
//! train/reconstruct methods) which `DecoderLm` deliberately does not cover. Once
//! that lands, compression becomes a normal [`Benchmark`] — `prepare` writes
//! random sequences, `evaluate` trains the autoencoder and scores per-token
//! reconstruction accuracy. Until then this benchmark reports a clear
//! "unsupported" status rather than a misleading score.
//!
//! It is intentionally **not** registered in [`registry`](crate::registry) (it
//! would always fail its threshold), but is kept here, compiled and tested, so
//! the gap is documented in code and the module is ready to flesh out the moment
//! the autoencoder model trait lands.

use std::path::Path;

use crate::metrics::Metrics;
use crate::Benchmark;

/// Placeholder for the MAD compression (autoencoder) task. See the module docs
/// for why it cannot be expressed via the current next-token `gpt::train` API and
/// what it needs (the `Model` trait). Not registered in [`registry`](crate::registry).
#[derive(Clone, Debug, Default)]
pub struct MadCompress;

impl MadCompress {
    /// Human-readable reason this task is currently unsupported, surfaced by
    /// [`evaluate`](Benchmark::evaluate) and usable by callers/tests.
    pub const UNSUPPORTED: &'static str =
        "mad_compress: autoencoder (sequence -> single representation -> MLP reconstruction) \
         requires a bottleneck + non-next-token objective + separate decoder head; \
         the DecoderLm model trait only covers next-token decoders, so this needs a \
         separate autoencoder model trait before it can be supported";
}

impl Benchmark for MadCompress {
    fn name(&self) -> &str {
        "mad_compress"
    }

    fn description(&self) -> &str {
        "MAD compression (autoencoder) — UNSUPPORTED until an autoencoder model trait lands"
    }

    /// No dataset is written: the task cannot be trained with the current engine.
    fn prepare(&self, _dir: &Path, _seed: u64) -> std::io::Result<()> {
        Ok(())
    }

    /// Returns a clear error rather than a misleading score. The headline `score`
    /// of any returned metrics would be meaningless, so we surface the gap as an
    /// `Unsupported` error containing [`MadCompress::UNSUPPORTED`].
    fn evaluate(&self, _dir: &Path, _seed: u64) -> std::io::Result<Metrics> {
        Err(std::io::Error::new(std::io::ErrorKind::Unsupported, Self::UNSUPPORTED))
    }

    /// Unreachable in practice (evaluate errors first); a score of 0 can never
    /// clear a 1.0 bar, encoding "not passing until implemented".
    fn threshold(&self) -> f32 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluate_reports_unsupported() {
        let b = MadCompress;
        let dir = std::env::temp_dir();
        let err = b.evaluate(&dir, 0).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::Unsupported);
        assert!(err.to_string().contains("model trait"));
    }

    #[test]
    fn metadata_is_present() {
        let b = MadCompress;
        assert_eq!(b.name(), "mad_compress");
        assert!(b.description().contains("UNSUPPORTED"));
    }
}
