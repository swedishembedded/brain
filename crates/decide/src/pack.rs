// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Laying a request out as the packed, span-described batch the encoder takes.
//!
//! A request is a state of arbitrary length plus some number of short slots
//! (one per option, carrying that question's instructions). The encoder sees
//! one flat token stream and a `(row0, len)` per sequence. This module is what
//! turns the former into the latter, and it owns the two rules that make the
//! layout legal:
//!
//! **Windowing.** The state is cut into windows of at most `max_span` tokens,
//! because the learned position table has that many rows and because attention
//! within a window is quadratic. Windows OVERLAP by `overlap` tokens so a fact
//! straddling a cut is still seen whole by at least one window. Every window
//! is its own span, so cost grows linearly with the state rather than
//! quadratically - which is what makes a 32k state affordable at all.
//!
//! Cross-window integration is the head's job, not the encoder's: the head
//! attends over every window's tokens at once. What a window cannot do is
//! contextualize against another window's tokens INSIDE the encoder.
//!
//! **Alignment.** The attention binds a view of the fused qkv and of the
//! context buffer starting at a span's first row, and a bound offset must be a
//! multiple of 256 bytes. So span starts are aligned BY CONSTRUCTION here,
//! padding the gap between spans when a width needs it, rather than left for
//! the encoder to reject. The pad rows belong to no span, so they are never
//! attended to and never pooled; they cost only their own GEMM rows.
//!
//! For every real checkpoint of this family (`d_model` a multiple of 64) the
//! required alignment is one row and no padding is ever emitted.

use crate::config::EncoderConfig;

/// WebGPU's `min_storage_buffer_offset_alignment`.
const BIND_ALIGN: u32 = 256;

/// The segment id carried by state tokens.
pub const SEG_STATE: u32 = 0;
/// The segment id carried by slot tokens (a question's instructions plus one
/// option). Distinguishing the two roles through BERT's own pretrained
/// `token_type` embedding is what avoids a second tower.
pub const SEG_SLOT: u32 = 1;

/// A packed request: the flat streams the encoder reads, plus where each
/// sequence lives in them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Packed {
    pub ids: Vec<u32>,
    pub types: Vec<u32>,
    /// `(row0, len)` per sequence, in the order they were added.
    pub spans: Vec<(u32, u32)>,
    /// How many leading spans are state windows; the rest are slots.
    pub windows: usize,
}

impl Packed {
    /// The state windows' spans.
    pub fn window_spans(&self) -> &[(u32, u32)] {
        &self.spans[..self.windows]
    }

    /// The slots' spans, in the order they were supplied.
    pub fn slot_spans(&self) -> &[(u32, u32)] {
        &self.spans[self.windows..]
    }

    /// Rows that belong to no span - pure alignment padding. Zero for every
    /// real checkpoint of this family.
    pub fn pad_rows(&self) -> u32 {
        self.ids.len() as u32 - self.spans.iter().map(|&(_, l)| l).sum::<u32>()
    }
}

/// How many rows a span start must be a multiple of, for this width.
///
/// Both the fused qkv row (`3H` floats) and the context row (`H` floats) are
/// bound at a span's first row, so both offsets must clear [`BIND_ALIGN`]; the
/// narrower row is the binding one.
pub fn align_rows(cfg: &EncoderConfig) -> u32 {
    let mut a = 1;
    for bytes in [3 * cfg.d_model * 4, cfg.d_model * 4] {
        let need = BIND_ALIGN / gcd(BIND_ALIGN, bytes);
        a = lcm(a, need.max(1));
    }
    a
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn lcm(a: u32, b: u32) -> u32 {
    a / gcd(a, b) * b
}

/// Builds a [`Packed`] one sequence at a time, aligning as it goes.
pub struct Packer {
    align: u32,
    pad_id: u32,
    ids: Vec<u32>,
    types: Vec<u32>,
    spans: Vec<(u32, u32)>,
    windows: usize,
}

impl Packer {
    /// `pad_id` fills alignment gaps. It is never attended to, so its value
    /// only has to be a legal index into the vocabulary - `[PAD]` by
    /// convention, so a dump of the stream reads correctly.
    pub fn new(cfg: &EncoderConfig, pad_id: u32) -> Packer {
        assert!(pad_id < cfg.vocab, "pad_id {pad_id} is outside the vocabulary");
        Packer { align: align_rows(cfg), pad_id, ids: Vec::new(), types: Vec::new(), spans: Vec::new(), windows: 0 }
    }

    /// Pad up to the next legal span start.
    fn align_to_boundary(&mut self) {
        while self.ids.len() as u32 % self.align != 0 {
            self.ids.push(self.pad_id);
            self.types.push(SEG_STATE);
        }
    }

    /// Append one sequence as its own span.
    pub fn push(&mut self, tokens: &[u32], segment: u32) -> (u32, u32) {
        assert!(!tokens.is_empty(), "a span must have at least one token");
        self.align_to_boundary();
        let row0 = self.ids.len() as u32;
        self.ids.extend_from_slice(tokens);
        self.types.extend(std::iter::repeat_n(segment, tokens.len()));
        let span = (row0, tokens.len() as u32);
        self.spans.push(span);
        span
    }

    /// Append a state, cut into overlapping windows of at most `max_span`.
    ///
    /// `overlap` is clamped below `max_span`: an overlap that met or exceeded
    /// the window would advance zero rows per window and never terminate.
    /// Must be called before any slot, so the window spans lead.
    pub fn push_state(&mut self, tokens: &[u32], max_span: u32, overlap: u32) {
        assert_eq!(self.windows, self.spans.len(), "the state must be packed before any slot");
        assert!(max_span > 0, "max_span must be positive");
        let overlap = overlap.min(max_span - 1);
        let stride = (max_span - overlap) as usize;
        let mut at = 0usize;
        while at < tokens.len() {
            let end = (at + max_span as usize).min(tokens.len());
            self.push(&tokens[at..end], SEG_STATE);
            self.windows += 1;
            if end == tokens.len() {
                break;
            }
            at += stride;
        }
    }

    /// Append one slot (a question's instructions plus one option).
    pub fn push_slot(&mut self, tokens: &[u32]) -> (u32, u32) {
        self.push(tokens, SEG_SLOT)
    }

    pub fn finish(self) -> Packed {
        Packed { ids: self.ids, types: self.types, spans: self.spans, windows: self.windows }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> EncoderConfig {
        EncoderConfig::mini_lm_l6()
    }

    /// Every real checkpoint of this family is 64-aligned, so the packer emits
    /// no padding at all and `pad_rows` is the proof.
    #[test]
    fn a_real_width_needs_no_alignment_padding() {
        assert_eq!(align_rows(&cfg()), 1);
        let mut p = Packer::new(&cfg(), 0);
        p.push_state(&[1, 2, 3, 4, 5], 512, 0);
        p.push_slot(&[6, 7]);
        p.push_slot(&[8]);
        let packed = p.finish();
        assert_eq!(packed.pad_rows(), 0);
        assert_eq!(packed.ids, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(packed.spans, vec![(0, 5), (5, 2), (7, 1)]);
    }

    /// ...and a width that is NOT gets padding rather than an illegal offset.
    /// 16 floats is 64 bytes per context row, so a span may only start every
    /// 4th row; the fused qkv row (192 bytes) needs every 4th too.
    #[test]
    fn a_narrow_width_is_padded_to_a_legal_boundary() {
        let narrow = EncoderConfig { d_model: 16, n_heads: 4, ..cfg() };
        assert_eq!(align_rows(&narrow), 4);
        let mut p = Packer::new(&narrow, 0);
        p.push_state(&[1, 2, 3], 512, 0);
        p.push_slot(&[4, 5]);
        let packed = p.finish();
        // The slot cannot start at row 3, so one pad row lands between them.
        assert_eq!(packed.spans, vec![(0, 3), (4, 2)]);
        assert_eq!(packed.ids, vec![1, 2, 3, 0, 4, 5]);
        assert_eq!(packed.pad_rows(), 1);
        for &(row0, _) in &packed.spans {
            assert_eq!(row0 % 4, 0, "span start {row0} is not a legal binding offset");
        }
    }

    /// A state shorter than one window is ONE span, so the windowed path and
    /// an unwindowed one are the same layout - which is what lets the encoder
    /// test assert they are bit-identical.
    #[test]
    fn a_short_state_is_a_single_window() {
        let mut p = Packer::new(&cfg(), 0);
        p.push_state(&[1, 2, 3], 512, 64);
        let packed = p.finish();
        assert_eq!(packed.windows, 1);
        assert_eq!(packed.spans, vec![(0, 3)]);
    }

    /// Windows advance by `max_span - overlap` and the last one is short
    /// rather than padded, so every token appears in at least one window.
    #[test]
    fn windows_overlap_and_cover_every_token() {
        let tokens: Vec<u32> = (1..=25).collect();
        let mut p = Packer::new(&cfg(), 0);
        p.push_state(&tokens, 10, 3);
        let packed = p.finish();
        // stride 7: rows 0..10, 7..17, 14..24, 21..25.
        assert_eq!(packed.window_spans(), &[(0, 10), (10, 10), (20, 10), (30, 4)]);
        let mut seen = vec![false; 26];
        let mut at = 0usize;
        for (w, &(_, len)) in packed.window_spans().iter().enumerate() {
            let start = w * 7;
            for i in 0..len as usize {
                seen[tokens[start + i] as usize] = true;
            }
            at += len as usize;
        }
        assert_eq!(at, packed.ids.len());
        assert!(seen[1..].iter().all(|&s| s), "some token appears in no window");
    }

    /// An overlap at or above the window size would advance zero rows per
    /// window and never terminate. It is clamped to the largest value that
    /// still makes progress, so the invariant asserted here is TERMINATION and
    /// coverage, not a particular span list - the list is a consequence of the
    /// clamp and would make this test brittle for no gain.
    ///
    /// The cost of asking for a silly overlap is real (a window per token at
    /// the limit), and it is the caller's to avoid; what the packer owes is
    /// that it finishes.
    #[test]
    fn an_overlap_that_would_not_advance_is_clamped() {
        let mut p = Packer::new(&cfg(), 0);
        p.push_state(&[1, 2, 3, 4, 5], 2, 99);
        let packed = p.finish();
        assert!(!packed.window_spans().is_empty());
        assert!(
            packed.window_spans().iter().all(|&(_, l)| l <= 2),
            "a window outran max_span: {:?}",
            packed.window_spans()
        );
        // Every token is in some window: the first window holds tokens 0..2
        // and each later one advances by the clamped stride of 1.
        assert_eq!(packed.window_spans().first().map(|&(_, l)| l), Some(2));
        assert!(packed.ids.contains(&5), "the last token was dropped");
    }

    /// State windows lead and slots follow, so a caller can tell which rows
    /// are keys and which are queries from the span list alone.
    #[test]
    fn windows_lead_and_slots_follow() {
        let mut p = Packer::new(&cfg(), 0);
        p.push_state(&[1, 2, 3, 4], 2, 0);
        p.push_slot(&[9, 9]);
        let packed = p.finish();
        assert_eq!(packed.windows, 2);
        assert_eq!(packed.window_spans(), &[(0, 2), (2, 2)]);
        assert_eq!(packed.slot_spans(), &[(4, 2)]);
        assert_eq!(packed.types, vec![SEG_STATE; 4].into_iter().chain([SEG_SLOT; 2]).collect::<Vec<_>>());
    }
}
