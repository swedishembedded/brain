// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving and training language models whose output head is padded or
// soft-capped for its clients. If your team needs expertise in LM-head numerics then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The declared transform between a language-model head's raw logits and the logits every consumer reads: an optional
//! soft cap `c * tanh(x / c)` and an optional bound on the valid vocabulary (the unembedding table may be padded with
//! rows no token maps to).
//!
//! One declaration, applied identically wherever logits are consumed:
//!
//! - training: [`crate::head_ce::HeadCe::set_head_transform`] (chunked cross-entropy, forward and backward);
//! - serving, on the device: [`HeadTransform::step`] over a `[rows, stride]` logits block, before the greedy pick, the
//!   top-k extraction or the keyed draw of [`crate::readout`] reads it;
//! - serving, on the host: [`HeadTransform::served`] over a raw row, which is what a [`crate::serve::PagedDecoder`] that
//!   declares a transform ([`crate::serve::PagedDecoder::head_transform`]) returns from `logits`, from its full rows and
//!   from the rows it scores a continuation over.
//!
//! Padded columns are excluded, not merely unlikely: on the device they are set to a large negative finite value that no
//! argmax, top-k or draw can prefer to a valid column and whose softmax weight is exactly zero; on the host the row is
//! cut to the valid vocabulary, so a served row has `valid_vocab` entries and `vocab()` reports that count.

use gpu_core::{DeviceBuffer, Gpu, Step};

/// What a padded column's logit is set to on the device ([`HeadTransform::step`]).
pub const MASKED_LOGIT: f32 = -1.0e30;

/// The kernels the device transform dispatches, for a model's kernel list.
pub const KERNELS: [(&str, &str); 2] = [("logit_cap_mask", kernels::LOGIT_CAP_MASK), ("logit_cap_mask_bwd", kernels::LOGIT_CAP_MASK_BWD)];

/// A declared head transform. `HeadTransform::NONE` (the default) is the identity.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HeadTransform {
    /// `Some(c)`: logits become `c * tanh(x / c)`. Must be positive and finite.
    pub softcap: Option<f32>,
    /// `Some(n)`: only the first `n` columns are real; the rest are padding.
    pub valid_vocab: Option<u32>,
}

impl HeadTransform {
    pub const NONE: HeadTransform = HeadTransform { softcap: None, valid_vocab: None };

    /// Whether the transform changes nothing (and a device step is not needed).
    pub fn is_identity(&self) -> bool {
        self.softcap.is_none() && self.valid_vocab.is_none()
    }

    /// Check the declaration against a head of `vocab` rows.
    pub fn validate(&self, vocab: u32) -> Result<(), String> {
        if let Some(c) = self.softcap {
            if !(c.is_finite() && c > 0.0) {
                return Err(format!("head transform: softcap must be positive and finite (got {c})"));
            }
        }
        match self.valid_vocab {
            Some(0) => Err("head transform: valid_vocab must be positive".into()),
            Some(n) if n > vocab => Err(format!("head transform: valid_vocab {n} exceeds the head's {vocab} rows")),
            _ => Ok(()),
        }
    }

    /// The vocabulary a consumer sees: `valid_vocab` where declared, else the head's own `vocab`.
    pub fn served_vocab(&self, vocab: usize) -> usize {
        self.valid_vocab.map_or(vocab, |n| n as usize)
    }

    /// The transform of one raw host row: soft cap on every valid entry, and the row cut to the valid vocabulary.
    pub fn served(&self, mut raw: Vec<f32>) -> Vec<f32> {
        raw.truncate(self.served_vocab(raw.len()));
        if let Some(c) = self.softcap {
            raw.iter_mut().for_each(|v| *v = crate::hostmath::softcap(*v, c));
        }
        raw
    }

    /// A row the device already transformed ([`Self::step`], whose cap is applied and whose padding columns hold
    /// [`MASKED_LOGIT`]) cut to the valid vocabulary: what a decoder whose logits live on the device returns as a full
    /// row. Applying [`Self::served`] to it would cap twice.
    pub fn trim(&self, mut row: Vec<f32>) -> Vec<f32> {
        row.truncate(self.served_vocab(row.len()));
        row
    }

    /// `log(softmax(served(raw) / temperature))[token]`: the log-probability of `token` under the full distribution over
    /// the valid vocabulary, what a decoder scoring a continuation reports. `token` must be a valid column.
    pub fn logprob_at(&self, raw: Vec<f32>, token: usize, temperature: f32) -> f32 {
        let row = self.served(raw);
        assert!(token < row.len(), "head transform: token {token} is outside the valid vocabulary of {}", row.len());
        crate::logprobs::row_logprob_at(&row, token, temperature)
    }

    /// [`Self::served`] for each row of a batch.
    pub fn served_rows(&self, rows: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        rows.into_iter().map(|r| self.served(r)).collect()
    }

    /// The device words `[rows, vocab, valid, cap]` of the two kernels (cap 0 means none).
    fn words(&self, rows: u32, vocab: u32) -> [u32; 4] {
        [rows, vocab, self.valid_vocab.unwrap_or(vocab), gpu_core::f(self.softcap.unwrap_or(0.0))]
    }

    /// The in-place step over a `[rows, vocab]` logits block, or `None` for the identity. `kernel` is
    /// `logit_cap_mask` on `g`'s kernel list ([`KERNELS`]).
    pub fn step(&self, g: &Gpu, kernel: usize, logits: &DeviceBuffer, rows: u32, vocab: u32) -> Option<Step> {
        (!self.is_identity()).then(|| g.step(kernel, &[logits], &self.words(rows, vocab), rows * vocab))
    }

    /// The backward of [`Self::step`] in place on `d` (the gradient with respect to the transformed logits), reading the
    /// transformed logits `y`; `kernel` is `logit_cap_mask_bwd`. `None` for the identity.
    pub fn bwd_step(&self, g: &Gpu, kernel: usize, d: &DeviceBuffer, y: &DeviceBuffer, rows: u32, vocab: u32) -> Option<Step> {
        (!self.is_identity()).then(|| g.step(kernel, &[d, y], &self.words(rows, vocab), rows * vocab))
    }
}
