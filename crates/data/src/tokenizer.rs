// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tokenizers. `u16` token ids throughout (GPT-2's 50257-entry vocab fits).
//!
//! - [`CharTokenizer`] — character-level, vocabulary built from a corpus
//!   (sorted unique chars), as in nanogpt's `create_char_level_encoding`.
//! - The GPT-2 byte-pair tokenizer lives in [`crate::bpe`] and also implements
//!   [`Tokenizer`].

/// A reversible text tokenizer.
pub trait Tokenizer {
    /// Encode text to token ids.
    fn encode(&self, text: &str) -> Vec<u16>;
    /// Decode token ids back to text.
    fn decode(&self, ids: &[u16]) -> String;
    /// Number of distinct tokens.
    fn vocab_size(&self) -> usize;
}

/// Character-level tokenizer: each distinct `char` is one token.
#[derive(Clone, Debug)]
pub struct CharTokenizer {
    itos: Vec<char>,
    stoi: std::collections::HashMap<char, u16>,
}

impl CharTokenizer {
    /// Build a tokenizer from a corpus: the vocabulary is the sorted set of
    /// distinct characters (deterministic, matching nanogpt).
    pub fn from_corpus(text: &str) -> Self {
        let mut chars: Vec<char> = text.chars().collect::<std::collections::BTreeSet<_>>().into_iter().collect();
        chars.sort_unstable();
        Self::from_itos(chars)
    }

    /// Build from an explicit id->char table (e.g. loaded from `meta.json`).
    pub fn from_itos(itos: Vec<char>) -> Self {
        let stoi = itos
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u16))
            .collect();
        CharTokenizer { itos, stoi }
    }

    /// id -> char table.
    pub fn itos(&self) -> &[char] {
        &self.itos
    }

    /// Token id for a single character, if in-vocabulary.
    pub fn token_of(&self, c: char) -> Option<u16> {
        self.stoi.get(&c).copied()
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<u16> {
        text.chars().filter_map(|c| self.stoi.get(&c).copied()).collect()
    }

    fn decode(&self, ids: &[u16]) -> String {
        ids.iter()
            .filter_map(|&i| self.itos.get(i as usize).copied())
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.itos.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_roundtrip_and_sorted_vocab() {
        let tok = CharTokenizer::from_corpus("banana=cab\n");
        // sorted unique: '\n','=','a','b','c','n'
        assert_eq!(tok.vocab_size(), 6);
        assert_eq!(tok.itos()[0], '\n');
        let s = "cabana\n";
        assert_eq!(tok.decode(&tok.encode(s)), s);
    }
}
