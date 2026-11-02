// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements multimodal LLM serving for clients whose
// pipelines feed images, audio or documents into a decoder. If your team needs
// expertise in batching multimodal prompts through one shared scheduler then
// you can procure our services by sending an email to info@swedishembedded.com.

//! Embedding rows inside a prompt.
//!
//! A multimodal request is a token prompt in which some positions are not
//! tokens at all: an encoder (a vision tower, an audio encoder) produced one
//! embedding row for each of them. [`PromptEmbeds`] carries those rows next to
//! the prompt, so the scheduler can admit, chunk, preempt and resume such a
//! request exactly as it does a text one, and a decoder that declares support
//! ([`super::PagedDecoder::embed_width`]) reads a position's row instead of
//! looking its token up.
//!
//! The prompt's token at an embedding position is a placeholder: it must be a
//! valid vocabulary id (the scheduler checks every prompt token), and the
//! decoder never reads its embedding. Its value matters in one place only: the
//! decoder's prefix reuse. Two prompts with the same tokens and different rows
//! are different prompts; [`PromptEmbeds::content_key`] is what a decoder keys
//! reuse by (or it keeps embedding spans out of reuse): it answers, for every
//! prefix length, a digest of the rows and rotary positions inside that prefix,
//! so an image wholly inside a reused prefix is part of the match and an image
//! beyond it is not.
//!
//! A decoder whose rotary position is not a row's index (multi-axis M-RoPE: an
//! image occupies fewer positions than it has rows) reads the position of every
//! prompt row from [`PromptEmbeds::rotary_positions`].

use std::sync::Arc;

/// What the embedding rows and rotary positions make of each prefix of a prompt, as one digest per prefix length.
///
/// A prefix of `len` tokens is identified by the rows standing before `len` and by the rotary positions that are not
/// the default `previous + 1` (an image of `h x w` merged rows spans `max(h, w)` positions; the text after it
/// continues from there): both are events at a prompt position, folded in order into a running digest. A prompt with
/// no rows and default positions has no events and digest 0 at every length, so a text prefix is the same prefix
/// whether or not an image follows it. The key holds one event per row and per rotary break, never one per token.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ContentKey {
    /// `(position, digest after folding in every event up to and including this one)`, ascending by position.
    events: Vec<(u32, u64)>,
}

impl ContentKey {
    /// The key of a prompt without embedding rows.
    pub const fn none() -> ContentKey {
        ContentKey { events: Vec::new() }
    }

    /// Digest of everything this prompt's rows and rotary positions say about its first `len` tokens.
    pub fn digest_at(&self, len: usize) -> u64 {
        let before = self.events.partition_point(|&(pos, _)| (pos as usize) < len);
        before.checked_sub(1).map_or(0, |i| self.events[i].1)
    }

    fn build(row_hashes: &[u64], positions: &[u32], rotary: Option<&[[u32; 3]]>) -> ContentKey {
        let breaks: Vec<(u32, [u32; 3])> = rotary.map_or_else(Vec::new, |r| {
            let mut expected = [0u32; 3];
            let mut out = Vec::new();
            for (i, &axes) in r.iter().enumerate() {
                if axes != expected {
                    out.push((i as u32, axes));
                }
                expected = axes.map(|a| a.wrapping_add(1));
            }
            out
        });
        let (mut rows, mut brk) = (positions.iter().zip(row_hashes).peekable(), breaks.iter().peekable());
        let mut events = Vec::with_capacity(positions.len() + breaks.len());
        let mut digest = 0u64;
        loop {
            let next_row = rows.peek().map(|(&p, _)| p);
            let next_break = brk.peek().map(|(p, _)| *p);
            let take_row = match (next_row, next_break) {
                (None, None) => break,
                (Some(r), Some(b)) => r <= b,
                (Some(_), None) => true,
                (None, Some(_)) => false,
            };
            let (pos, event) = if take_row {
                let (&p, &h) = rows.next().expect("peeked");
                (p, mix(mix(1, u64::from(p)), h))
            } else {
                let &(p, axes) = brk.next().expect("peeked");
                (p, axes.iter().fold(mix(2, u64::from(p)), |h, &a| mix(h, u64::from(a))))
            };
            digest = mix(digest, event);
            events.push((pos, digest));
        }
        ContentKey { events }
    }
}

/// The embedding rows of a prompt, at ascending prompt positions.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptEmbeds {
    width: usize,
    positions: Vec<u32>,
    rows: Vec<f32>,
    /// A hash of each row's values, so the key can be rebuilt when rotary positions arrive without touching the rows.
    row_hashes: Vec<u64>,
    /// The rotary position of every prompt row, on three axes, when it is not the row's index.
    rotary: Option<Vec<[u32; 3]>>,
    content: Arc<ContentKey>,
}

impl PromptEmbeds {
    /// `rows` holds `positions.len()` rows of `width` floats, row `i` standing
    /// at prompt position `positions[i]`. Positions must be strictly ascending.
    pub fn new(width: usize, positions: Vec<u32>, rows: Vec<f32>) -> Result<PromptEmbeds, String> {
        if width == 0 {
            return Err("embedding rows have width 0".to_string());
        }
        if rows.len() != positions.len() * width {
            return Err(format!("{} positions of width {width} need {} floats, got {}", positions.len(), positions.len() * width, rows.len()));
        }
        if let Some(w) = positions.windows(2).find(|w| w[0] >= w[1]) {
            return Err(format!("embedding positions must be strictly ascending, found {} then {}", w[0], w[1]));
        }
        if let Some(bad) = rows.iter().position(|v| !v.is_finite()) {
            return Err(format!("embedding value {bad} is not finite"));
        }
        let row_hashes: Vec<u64> = rows.chunks_exact(width).map(|row| hash_row(width, row)).collect();
        let content = Arc::new(ContentKey::build(&row_hashes, &positions, None));
        Ok(PromptEmbeds { width, positions, rows, row_hashes, rotary: None, content })
    }

    /// One contiguous run of `rows` (`rows.len() / width` of them) starting at prompt position `start`.
    pub fn run(width: usize, start: u32, rows: Vec<f32>) -> Result<PromptEmbeds, String> {
        if width == 0 || !rows.len().is_multiple_of(width) {
            return Err(format!("{} floats are not whole rows of width {width}", rows.len()));
        }
        let n = (rows.len() / width) as u32;
        PromptEmbeds::new(width, (start..start + n).collect(), rows)
    }

    /// These rows with the rotary position of each of the prompt's rows (all of them, text included) on the three
    /// axes of a multi-axis rotary embedding. The positions are part of the prompt's identity ([`Self::content_key`]).
    pub fn with_rotary_positions(mut self, rotary: Vec<[u32; 3]>) -> PromptEmbeds {
        self.content = Arc::new(ContentKey::build(&self.row_hashes, &self.positions, Some(&rotary)));
        self.rotary = Some(rotary);
        self
    }

    /// The rotary position of every prompt row, when the prompt carries them.
    pub fn rotary_positions(&self) -> Option<&[[u32; 3]]> {
        self.rotary.as_deref()
    }

    /// The rows, `len() * width()` floats: row `i` stands at [`Self::positions`]`[i]`.
    pub fn rows(&self) -> &[f32] {
        &self.rows
    }

    /// Floats per row.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of embedding rows.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// The prompt positions the rows stand at, ascending.
    pub fn positions(&self) -> &[u32] {
        &self.positions
    }

    /// A 64-bit hash of the positions, row values and rotary positions of the whole prompt: equal for equal content.
    pub fn digest(&self) -> u64 {
        self.content.digest_at(usize::MAX)
    }

    /// What a decoder keys prefix reuse by: a digest of the content inside each prefix length (see [`ContentKey`]).
    pub fn content_key(&self) -> Arc<ContentKey> {
        self.content.clone()
    }

    /// The row standing at prompt position `pos`, if any.
    pub fn row_at(&self, pos: u32) -> Option<&[f32]> {
        let i = self.positions.binary_search(&pos).ok()?;
        Some(&self.rows[i * self.width..(i + 1) * self.width])
    }

    /// The rows standing in prompt positions `from..to`, as `(position, row)`.
    pub fn rows_in(&self, from: u32, to: u32) -> impl Iterator<Item = (u32, &[f32])> + '_ {
        let lo = self.positions.partition_point(|&p| p < from);
        let hi = self.positions.partition_point(|&p| p < to);
        (lo..hi).map(move |i| (self.positions[i], &self.rows[i * self.width..(i + 1) * self.width]))
    }

    /// The first prompt position at or after `from` that carries a row.
    pub fn next_at_or_after(&self, from: u32) -> Option<u32> {
        let i = self.positions.partition_point(|&p| p < from);
        self.positions.get(i).copied()
    }

    /// Whether any position in `from..to` carries a row.
    pub fn touches(&self, from: u32, to: u32) -> bool {
        self.next_at_or_after(from).is_some_and(|p| p < to)
    }

    /// The reason these rows cannot belong to a prompt of `prompt_len` tokens
    /// for a decoder taking rows of `width`, if any.
    pub fn refusal(&self, prompt_len: usize, width: Option<usize>) -> Option<String> {
        let Some(width) = width else {
            return Some("the decoder does not take embedding rows in a prompt".to_string());
        };
        if self.width != width {
            return Some(format!("rows are {} floats wide, the decoder takes {width}", self.width));
        }
        match self.positions.last() {
            Some(&last) if last as usize >= prompt_len => Some(format!("a row stands at position {last}, the prompt has {prompt_len} tokens")),
            _ => match &self.rotary {
                Some(r) if r.len() != prompt_len => Some(format!("{} rotary positions for a prompt of {prompt_len} tokens", r.len())),
                _ => None,
            },
        }
    }
}

const PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold `v` into `h`. Not cryptographic: it separates prompts inside one process's cache, where an attacker cannot
/// choose both sides (the rows come out of a vision tower, not from the client).
fn mix(h: u64, v: u64) -> u64 {
    (h.rotate_left(23) ^ v).wrapping_mul(PRIME).wrapping_add(0x9e37_79b9_7f4a_7c15)
}

/// A hash of one row's width and values, a word at a time.
fn hash_row(width: usize, row: &[f32]) -> u64 {
    row.iter().fold(mix(0xcbf2_9ce4_8422_2325, width as u64), |h, v| mix(h, u64::from(v.to_bits())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_validated_at_construction() {
        assert!(PromptEmbeds::new(2, vec![3, 3], vec![0.0; 4]).is_err(), "positions must ascend strictly");
        assert!(PromptEmbeds::new(2, vec![1], vec![0.0; 3]).is_err(), "rows must match positions x width");
        assert!(PromptEmbeds::new(2, vec![1], vec![0.0, f32::NAN]).is_err(), "a non-finite value is refused");
        assert!(PromptEmbeds::new(0, vec![], vec![]).is_err());
    }

    #[test]
    fn rows_are_found_by_position() {
        let e = PromptEmbeds::new(2, vec![2, 3, 7], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(e.row_at(3), Some(&[3.0, 4.0][..]));
        assert_eq!(e.row_at(4), None);
        assert_eq!(e.rows_in(3, 8).map(|(p, _)| p).collect::<Vec<_>>(), vec![3, 7]);
        assert!(e.touches(4, 8) && !e.touches(4, 7) && !e.touches(8, 20));
        assert_eq!(e.next_at_or_after(4), Some(7));
    }

    #[test]
    fn the_digest_follows_the_content() {
        let a = PromptEmbeds::run(2, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(a.digest(), PromptEmbeds::run(2, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap().digest());
        assert_ne!(a.digest(), PromptEmbeds::run(2, 4, vec![1.0, 2.0, 3.0, 4.5]).unwrap().digest(), "a different value");
        assert_ne!(a.digest(), PromptEmbeds::run(2, 5, vec![1.0, 2.0, 3.0, 4.0]).unwrap().digest(), "a different position");
    }

    #[test]
    fn rotary_positions_are_part_of_the_prompts_identity_and_must_cover_it() {
        let plain = PromptEmbeds::run(2, 1, vec![0.0; 4]).unwrap();
        let rotary = vec![[0, 0, 0], [1, 1, 1], [1, 1, 2], [3, 3, 3]];
        let with = plain.clone().with_rotary_positions(rotary.clone());
        assert_eq!(with.rotary_positions(), Some(&rotary[..]));
        assert_ne!(plain.digest(), with.digest(), "the same rows at other rotary positions are another prompt");
        let moved = plain.clone().with_rotary_positions(vec![[0, 0, 0], [1, 1, 1], [1, 2, 1], [3, 3, 3]]);
        assert_ne!(with.digest(), moved.digest());
        assert_eq!(with.refusal(4, Some(2)), None);
        assert!(with.refusal(5, Some(2)).unwrap().contains("rotary"), "one position per prompt row");
    }

    /// A prefix is identified by the rows inside it: the text before an image is the same prefix whatever the image,
    /// a prefix holding the image is another prefix for another image, and one cut inside the image is too.
    #[test]
    fn the_content_key_digests_the_rows_inside_each_prefix() {
        let image = |v: f32| PromptEmbeds::run(2, 4, vec![v, 1.0, 2.0, 3.0]).unwrap().content_key();
        let (a, b) = (image(0.5), image(0.75));
        assert_eq!(a.digest_at(4), 0, "before the image there is nothing to tell prompts apart");
        assert_eq!(a.digest_at(4), b.digest_at(4));
        assert_eq!(a.digest_at(4), ContentKey::none().digest_at(4), "a text prefix is a text prefix");
        assert_ne!(a.digest_at(5), b.digest_at(5), "cut inside the image");
        assert_ne!(a.digest_at(100), b.digest_at(100), "past the image");
        assert_eq!(a.digest_at(100), image(0.5).digest_at(100), "equal rows, equal key");
        assert_ne!(a.digest_at(100), 0);
    }

    /// Rotary positions that are the default (each row one past the last) say nothing; a break from it does, and
    /// the text after a break is identified by where it continues from, however many rows follow.
    #[test]
    fn default_rotary_positions_are_no_content_and_a_break_is() {
        let plain = PromptEmbeds::run(2, 1, vec![0.0; 4]).unwrap();
        let default: Vec<[u32; 3]> = (0..6).map(|i| [i; 3]).collect();
        assert_eq!(plain.clone().with_rotary_positions(default).content_key().digest_at(6), plain.content_key().digest_at(6));
        let narrow: Vec<[u32; 3]> = vec![[0; 3], [1; 3], [1, 1, 2], [3; 3], [4; 3], [5; 3]];
        let wide: Vec<[u32; 3]> = vec![[0; 3], [1; 3], [1, 1, 2], [3; 3], [4; 3], [9; 3]];
        let (n, w) = (plain.clone().with_rotary_positions(narrow).content_key(), plain.with_rotary_positions(wide).content_key());
        assert_eq!(n.digest_at(5), w.digest_at(5));
        assert_ne!(n.digest_at(6), w.digest_at(6), "the last row continues from another position");
    }

    #[test]
    fn refusal_names_the_mismatch() {
        let e = PromptEmbeds::run(2, 1, vec![0.0; 4]).unwrap();
        assert!(e.refusal(8, None).unwrap().contains("does not take"));
        assert!(e.refusal(8, Some(3)).unwrap().contains("wide"));
        assert!(e.refusal(2, Some(2)).unwrap().contains("position 2"));
        assert_eq!(e.refusal(3, Some(2)), None);
    }
}
