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
//! are different prompts; [`PromptEmbeds::digest`] is the content hash a
//! decoder keys reuse by (or it keeps embedding spans out of reuse).

/// The embedding rows of a prompt, at ascending prompt positions.
#[derive(Clone, Debug, PartialEq)]
pub struct PromptEmbeds {
    width: usize,
    positions: Vec<u32>,
    rows: Vec<f32>,
    digest: u64,
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
        let digest = fingerprint(width, &positions, &rows);
        Ok(PromptEmbeds { width, positions, rows, digest })
    }

    /// One contiguous run of `rows` (`rows.len() / width` of them) starting at prompt position `start`.
    pub fn run(width: usize, start: u32, rows: Vec<f32>) -> Result<PromptEmbeds, String> {
        if width == 0 || !rows.len().is_multiple_of(width) {
            return Err(format!("{} floats are not whole rows of width {width}", rows.len()));
        }
        let n = (rows.len() / width) as u32;
        PromptEmbeds::new(width, (start..start + n).collect(), rows)
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

    /// A 64-bit hash of the width, positions and row values: equal for equal
    /// content, so a decoder can key prefix reuse by it.
    pub fn digest(&self) -> u64 {
        self.digest
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
            _ => None,
        }
    }
}

/// FNV-1a over the content, 64 bit. Not cryptographic: it separates prompts
/// inside one process's cache, where an attacker cannot choose both sides.
fn fingerprint(width: usize, positions: &[u32], rows: &[f32]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(PRIME);
        }
    };
    mix(&(width as u64).to_le_bytes());
    mix(&(positions.len() as u64).to_le_bytes());
    for p in positions {
        mix(&p.to_le_bytes());
    }
    for v in rows {
        mix(&v.to_bits().to_le_bytes());
    }
    h
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
    fn refusal_names_the_mismatch() {
        let e = PromptEmbeds::run(2, 1, vec![0.0; 4]).unwrap();
        assert!(e.refusal(8, None).unwrap().contains("does not take"));
        assert!(e.refusal(8, Some(3)).unwrap().contains("wide"));
        assert!(e.refusal(2, Some(2)).unwrap().contains("position 2"));
        assert_eq!(e.refusal(3, Some(2)), None);
    }
}
