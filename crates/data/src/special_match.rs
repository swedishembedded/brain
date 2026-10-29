// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Literal special-token matching in one pass over the text.
//!
//! A tokenizer cuts added/special tokens (`<|im_start|>`, `[MASK]`, ...) out of
//! the raw text before BPE. Searching the text once per special
//! (`text.find(content)` in a loop) costs O(specials x text) per cut; this
//! matcher is a byte trie that answers "which special starts here, leftmost
//! then longest" in O(length of the match), and skips every position whose
//! byte starts no special with one table lookup.
//!
//! Swedish Embedded AB implements solutions for fast, memory-lean text
//! tokenization for its clients. If your team needs expertise in inference
//! engines then you can procure our services by sending an email to
//! info@swedishembedded.com.

/// A byte trie over the special-token contents, answering leftmost-longest
/// matches. Duplicate contents keep the first id given.
#[derive(Debug, Clone)]
pub struct SpecialMatcher {
    /// Trie nodes; node 0 is the root. Each node's children are `(byte, node)`
    /// pairs, few per node (a special set shares a handful of leading bytes).
    nodes: Vec<Node>,
    /// Bytes that begin at least one special: the scan's O(1) skip test.
    first: [bool; 256],
}

#[derive(Debug, Clone, Default)]
struct Node {
    children: Vec<(u8, u32)>,
    id: Option<u32>,
}

impl SpecialMatcher {
    /// Build over `(content, id)` pairs; empty contents are ignored (they
    /// would match everywhere). Construction is O(total content bytes).
    pub fn new<'a>(specials: impl IntoIterator<Item = (&'a str, u32)>) -> SpecialMatcher {
        let mut m = SpecialMatcher { nodes: vec![Node::default()], first: [false; 256] };
        for (content, id) in specials {
            m.insert(content, id);
        }
        m
    }

    /// Register one special; an existing identical content keeps its id and an
    /// empty content is ignored. O(content length).
    pub fn insert(&mut self, content: &str, id: u32) {
        let Some(&lead) = content.as_bytes().first() else { return };
        self.first[lead as usize] = true;
        let mut at = 0usize;
        for &b in content.as_bytes() {
            at = match self.nodes[at].children.iter().find(|(c, _)| *c == b) {
                Some(&(_, next)) => next as usize,
                None => {
                    self.nodes.push(Node::default());
                    let next = self.nodes.len() - 1;
                    self.nodes[at].children.push((b, next as u32));
                    next
                }
            };
        }
        self.nodes[at].id.get_or_insert(id);
    }

    /// True when no special is registered.
    pub fn is_empty(&self) -> bool {
        self.nodes.len() == 1
    }

    /// The id of the special spelled exactly `content`. O(len).
    pub fn id_of(&self, content: &str) -> Option<u32> {
        self.walk(content.as_bytes()).and_then(|(len, id)| (len == content.len()).then_some(id))
    }

    /// The longest special that is a prefix of `text`, as `(byte length, id)`.
    /// O(length of the match); at most the longest special's length.
    pub fn longest_at(&self, text: &str) -> Option<(usize, u32)> {
        self.walk(text.as_bytes())
    }

    fn walk(&self, bytes: &[u8]) -> Option<(usize, u32)> {
        let mut at = 0usize;
        let mut best = None;
        for (i, &b) in bytes.iter().enumerate() {
            let Some(&(_, next)) = self.nodes[at].children.iter().find(|(c, _)| *c == b) else { break };
            at = next as usize;
            if let Some(id) = self.nodes[at].id {
                best = Some((i + 1, id));
            }
        }
        best
    }

    /// The leftmost special in `text`, longest among those starting there:
    /// `(start, byte length, id)`. One pass: O(text.len() x longest special)
    /// worst case, O(text.len()) when no position repeatedly extends a long
    /// partial match, which holds for natural text.
    pub fn find(&self, text: &str) -> Option<(usize, usize, u32)> {
        let bytes = text.as_bytes();
        // A special starts on a char boundary (its first byte is a UTF-8 lead
        // byte), and a continuation byte never equals a lead byte, so the
        // `first` table alone rules out every mid-char position.
        (0..bytes.len()).filter(|&i| self.first[bytes[i] as usize]).find_map(|i| self.walk(&bytes[i..]).map(|(len, id)| (i, len, id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The quadratic reference this replaces: earliest position of any
    /// special, longest first on a tie.
    fn naive(text: &str, sorted_longest_first: &[(String, u32)]) -> Option<(usize, usize, u32)> {
        let mut best: Option<(usize, usize, u32)> = None;
        for (c, id) in sorted_longest_first {
            if let Some(p) = text.find(c.as_str()) {
                if best.map(|(bp, _, _)| p < bp).unwrap_or(true) {
                    best = Some((p, c.len(), *id));
                }
            }
        }
        best
    }

    #[test]
    fn matches_the_per_special_search_on_random_text() {
        let mut specials: Vec<(String, u32)> =
            ["<|im_start|>", "<|im_end|>", "<|im|>", "<", "<|", "é", "[MASK]", "[M]", "ab", "abc"].iter().enumerate().map(|(i, s)| (s.to_string(), i as u32)).collect();
        specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        let m = SpecialMatcher::new(specials.iter().map(|(c, i)| (c.as_str(), *i)));
        let alphabet = ["<", "|", "im", "_", "start", "end", ">", "é", "[", "M", "ASK", "]", "a", "b", "c", " ", "ü"];
        // seed: a fixed stream so a failing text reproduces
        let mut rng = crate::rng::Lcg::new(7);
        for _ in 0..2000 {
            let n = (rng.unit() * 20.0) as usize;
            let text: String = (0..n).map(|_| alphabet[(rng.unit() * alphabet.len() as f32) as usize % alphabet.len()]).collect();
            assert_eq!(m.find(&text), naive(&text, &specials), "text {text:?}");
        }
    }

    #[test]
    fn exact_lookup_and_duplicates_keep_the_first_id() {
        let m = SpecialMatcher::new([("<a>", 1), ("<a>", 2), ("<ab>", 3), ("", 9)]);
        assert_eq!(m.id_of("<a>"), Some(1));
        assert_eq!(m.id_of("<ab>"), Some(3));
        assert_eq!(m.id_of("<"), None);
        assert_eq!(m.id_of("<a>x"), None);
        assert_eq!(m.longest_at("<ab>z"), Some((4, 3)));
        assert!(SpecialMatcher::new([]).is_empty());
    }

    #[test]
    fn a_long_text_with_many_specials_is_scanned_in_one_pass() {
        let specials: Vec<(String, u32)> = (0..2000).map(|i| (format!("<|tok{i}|>"), i)).collect();
        let m = SpecialMatcher::new(specials.iter().map(|(c, i)| (c.as_str(), *i)));
        let text = "plain words ".repeat(50_000) + "<|tok1999|>";
        let t0 = std::time::Instant::now();
        assert_eq!(m.find(&text), Some((text.len() - 11, 11, 1999)));
        // A per-special search costs 2000 passes over 600 KB (~1e9 byte
        // compares); the single pass is milliseconds. The bound is ~100x slack.
        assert!(t0.elapsed() < std::time::Duration::from_secs(2));
    }
}
