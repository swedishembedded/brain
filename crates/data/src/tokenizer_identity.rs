// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for distilling one language model into another across checkpoints that may
// not share a vocabulary, for its clients. If your team needs expertise in teacher-student training of language
// models then you can procure our services by sending an email to info@swedishembedded.com.

//! Whether two checkpoints share one tokenizer: [`TokenizerIdentity::check`].
//!
//! A teacher's log-probability of a token id is the student's signal for that id only when both models mean the same
//! text by it and would cut the same text into the same ids. That holds when the two `tokenizer.json` files agree on
//! everything that decides an encoding: the model kind and its settings, the vocabulary (every piece and its id), the
//! merges in order, the normalizer, the pre-tokenizer, and the added (special) tokens with their ids. The decoder and
//! the post-processor are not compared: they decide how ids read back or what a pipeline wraps around them, not which
//! ids a text is.
//!
//! The answer decides which on-policy distillation estimator a run may use (`rl::objective::onpolicy_distill`):
//! token-level reverse KL only for identical tokenizers, the sequence-level estimator for any pair. The value can only
//! come from the check, so a caller cannot assert an identity the files do not have.
//!
//! The tokenizer files are untrusted input: each is read through a size bound and parsed into typed values; a file
//! that is missing, too large or not a byte-pair `tokenizer.json` is an error naming it, never a guess.

use std::fmt;
use std::io;
use std::path::Path;

/// The largest `tokenizer.json` read, in bytes. The largest real ones are around 13 MB.
pub const MAX_TOKENIZER_BYTES: u64 = 256 << 20;

/// Which parts of two tokenizers differ. Every `false` field is a part they agree on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenizerDifferences {
    /// The model kind (`BPE`, `WordPiece`, ...) or one of its settings (unknown token, byte fallback, ...).
    pub model: bool,
    /// The vocabulary: some piece is missing from one of them or has another id.
    pub vocab: bool,
    /// The merges, in order.
    pub merges: bool,
    /// The normalizer.
    pub normalizer: bool,
    /// The pre-tokenizer (the split pattern a text is cut by before merging).
    pub pre_tokenizer: bool,
    /// The added tokens: a special token is missing, has another id or other matching flags.
    pub added_tokens: bool,
}

impl TokenizerDifferences {
    /// Whether no part differs.
    pub fn is_empty(&self) -> bool {
        *self == TokenizerDifferences::default()
    }

    /// The names of the parts that differ, in a fixed order.
    pub fn names(&self) -> Vec<&'static str> {
        [(self.model, "model"), (self.vocab, "vocab"), (self.merges, "merges"), (self.normalizer, "normalizer"), (self.pre_tokenizer, "pre_tokenizer"), (self.added_tokens, "added_tokens")].into_iter().filter(|(differs, _)| *differs).map(|(_, name)| name).collect()
    }
}

/// How large one tokenizer is: what a refusal reports beside the parts that differ.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenizerSize {
    /// Pieces in the model's vocabulary (added tokens not counted).
    pub vocab: usize,
    pub merges: usize,
    pub added_tokens: usize,
}

/// The result of comparing a student's tokenizer with a teacher's. It is made only by [`TokenizerIdentity::check`]
/// (or [`TokenizerIdentity::of_json`]), so holding one means the files were compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenizerIdentity {
    differences: TokenizerDifferences,
    student: TokenizerSize,
    teacher: TokenizerSize,
}

impl TokenizerIdentity {
    /// Compare the `tokenizer.json` of the student checkpoint directory with the teacher's.
    ///
    /// # Errors
    /// Either file is missing, larger than [`MAX_TOKENIZER_BYTES`], or not a tokenizer this check reads; the error
    /// names the file.
    pub fn check(student_dir: &Path, teacher_dir: &Path) -> io::Result<TokenizerIdentity> {
        let read = |dir: &Path| -> io::Result<Vec<u8>> {
            let path = dir.join("tokenizer.json");
            let len = std::fs::metadata(&path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?.len();
            if len > MAX_TOKENIZER_BYTES {
                return Err(io::Error::new(io::ErrorKind::InvalidData, format!("{}: {len} bytes, more than the {MAX_TOKENIZER_BYTES} a tokenizer is read up to", path.display())));
            }
            std::fs::read(&path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))
        };
        let (student, teacher) = (read(student_dir)?, read(teacher_dir)?);
        TokenizerIdentity::of_json(&student, &teacher).map_err(|why| io::Error::new(io::ErrorKind::InvalidData, format!("comparing {} with {}: {why}", student_dir.join("tokenizer.json").display(), teacher_dir.join("tokenizer.json").display())))
    }

    /// Compare two `tokenizer.json` documents.
    ///
    /// # Errors
    /// Either is not JSON, or lacks the `model` object with its `vocab` this check reads; the error says which side.
    pub fn of_json(student: &[u8], teacher: &[u8]) -> Result<TokenizerIdentity, String> {
        let s = Parts::parse(student).map_err(|why| format!("the student's tokenizer: {why}"))?;
        let t = Parts::parse(teacher).map_err(|why| format!("the teacher's tokenizer: {why}"))?;
        let differences = TokenizerDifferences {
            model: s.settings != t.settings,
            vocab: s.vocab != t.vocab,
            merges: s.merges != t.merges,
            normalizer: s.normalizer != t.normalizer,
            pre_tokenizer: s.pre_tokenizer != t.pre_tokenizer,
            added_tokens: s.added_tokens != t.added_tokens,
        };
        Ok(TokenizerIdentity { differences, student: s.size(), teacher: t.size() })
    }

    /// Whether the two tokenizers encode every text to the same ids.
    pub fn is_identical(&self) -> bool {
        self.differences.is_empty()
    }

    /// The parts that differ (none for identical tokenizers).
    pub fn differences(&self) -> TokenizerDifferences {
        self.differences
    }

    /// The student's and the teacher's sizes.
    pub fn sizes(&self) -> (TokenizerSize, TokenizerSize) {
        (self.student, self.teacher)
    }
}

/// What one `tokenizer.json` says about encoding, in a form two can be compared in.
struct Parts {
    /// The model object without its vocabulary and merges: the kind and its settings.
    settings: serde_json::Map<String, serde_json::Value>,
    /// The vocabulary as the file gives it (a piece-to-id map for byte-pair and word-piece models, a list for unigram).
    vocab: serde_json::Value,
    vocab_len: usize,
    /// The merges as pairs, whichever of the two spellings (`"a b"` or `["a", "b"]`) the file uses.
    merges: Vec<(String, String)>,
    normalizer: serde_json::Value,
    pre_tokenizer: serde_json::Value,
    /// The added tokens ordered by id.
    added_tokens: Vec<serde_json::Value>,
}

impl Parts {
    fn parse(bytes: &[u8]) -> Result<Parts, String> {
        #[derive(serde::Deserialize)]
        struct Document {
            model: serde_json::Map<String, serde_json::Value>,
            #[serde(default)]
            normalizer: serde_json::Value,
            #[serde(default)]
            pre_tokenizer: serde_json::Value,
            #[serde(default)]
            added_tokens: Vec<serde_json::Value>,
        }
        let Document { mut model, normalizer, pre_tokenizer, mut added_tokens } = serde_json::from_slice(bytes).map_err(|e| format!("not a tokenizer.json: {e}"))?;
        let vocab = model.remove("vocab").ok_or("the model has no \"vocab\"")?;
        let vocab_len = match &vocab {
            serde_json::Value::Object(map) => map.len(),
            serde_json::Value::Array(list) => list.len(),
            other => return Err(format!("the model's \"vocab\" is {}, neither a map nor a list", kind(other))),
        };
        let merges = match model.remove("merges") {
            None => Vec::new(),
            Some(serde_json::Value::Array(list)) => list.iter().enumerate().map(|(i, m)| merge(m).ok_or_else(|| format!("merge {i} is {m}, neither \"a b\" nor [\"a\", \"b\"]"))).collect::<Result<_, _>>()?,
            Some(other) => return Err(format!("the model's \"merges\" is {}, not a list", kind(&other))),
        };
        if let Some(i) = added_tokens.iter().position(|t| t.get("id").and_then(serde_json::Value::as_u64).is_none()) {
            return Err(format!("added token {i} has no numeric \"id\""));
        }
        added_tokens.sort_by_key(|t| t["id"].as_u64());
        Ok(Parts { settings: model, vocab, vocab_len, merges, normalizer, pre_tokenizer, added_tokens })
    }

    fn size(&self) -> TokenizerSize {
        TokenizerSize { vocab: self.vocab_len, merges: self.merges.len(), added_tokens: self.added_tokens.len() }
    }
}

/// One merge in either spelling: `"a b"` (split at the one space) or `["a", "b"]`.
fn merge(m: &serde_json::Value) -> Option<(String, String)> {
    match m {
        serde_json::Value::String(s) => s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())),
        serde_json::Value::Array(pair) => match pair.as_slice() {
            [serde_json::Value::String(a), serde_json::Value::String(b)] => Some((a.clone(), b.clone())),
            _ => None,
        },
        _ => None,
    }
}

fn kind(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "a list",
        serde_json::Value::Object(_) => "a map",
    }
}

impl fmt::Display for TokenizerIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_identical() {
            return write!(f, "identical tokenizers ({} pieces, {} merges, {} added tokens)", self.student.vocab, self.student.merges, self.student.added_tokens);
        }
        let (s, t) = (self.student, self.teacher);
        let parts: Vec<String> = self
            .differences
            .names()
            .into_iter()
            .map(|name| match name {
                "vocab" => format!("vocab ({} vs {} pieces)", s.vocab, t.vocab),
                "merges" => format!("merges ({} vs {})", s.merges, t.merges),
                "added_tokens" => format!("added_tokens ({} vs {})", s.added_tokens, t.added_tokens),
                other => other.to_string(),
            })
            .collect();
        write!(f, "different tokenizers (student vs teacher): {}", parts.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A small byte-pair tokenizer in the `tokenizer.json` layout, with Qwen's shape (a split pattern then byte-level).
    fn tokenizer() -> serde_json::Value {
        json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [
                {"id": 6, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
                {"id": 7, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
            ],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": "\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+"}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
            ]},
            "post_processor": {"type": "ByteLevel"},
            "decoder": {"type": "ByteLevel"},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "abc": 5},
                "merges": [["a", "b"], ["b", "c"], ["ab", "c"]]}
        })
    }

    fn compare(student: &serde_json::Value, teacher: &serde_json::Value) -> TokenizerIdentity {
        TokenizerIdentity::of_json(student.to_string().as_bytes(), teacher.to_string().as_bytes()).expect("both parse")
    }

    #[test]
    fn a_tokenizer_is_identical_to_a_copy_of_itself_whatever_its_merge_spelling_and_decoder() {
        let t = tokenizer();
        let same = compare(&t, &t);
        assert!(same.is_identical(), "{same}");
        assert_eq!(same.sizes().0, TokenizerSize { vocab: 6, merges: 3, added_tokens: 2 });
        // The older merge spelling ("a b") is the same merge; the decoder and post-processor decide no ids.
        let mut respelled = t.clone();
        respelled["model"]["merges"] = json!(["a b", "b c", "ab c"]);
        respelled["decoder"] = json!({"type": "Fuse"});
        respelled["post_processor"] = serde_json::Value::Null;
        assert!(compare(&t, &respelled).is_identical());
    }

    #[test]
    fn every_part_that_decides_an_encoding_is_compared() {
        let base = tokenizer();
        let perturb = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut t = base.clone();
            edit(&mut t);
            compare(&base, &t)
        };
        type Edit<'a> = &'a dyn Fn(&mut serde_json::Value);
        let cases: [(&str, Edit); 7] = [
            ("vocab", &|t| t["model"]["vocab"]["abc"] = json!(9)),
            ("vocab", &|t| t["model"]["vocab"]["cab"] = json!(6)),
            ("merges", &|t| t["model"]["merges"] = json!([["b", "c"], ["a", "b"], ["ab", "c"]])),
            ("pre_tokenizer", &|t| t["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] = json!("[\\p{L}\\p{M}]+|\\p{N}")),
            ("normalizer", &|t| t["normalizer"] = json!({"type": "NFKC"})),
            ("added_tokens", &|t| t["added_tokens"][1]["id"] = json!(8)),
            ("model", &|t| t["model"]["byte_fallback"] = json!(true)),
        ];
        for (part, edit) in cases {
            let identity = perturb(edit);
            assert!(!identity.is_identical(), "{part}: {identity}");
            assert_eq!(identity.differences().names(), vec![part], "{identity}");
            assert!(identity.to_string().contains(part), "{identity}");
        }
    }

    #[test]
    fn a_document_that_is_not_a_tokenizer_is_refused_naming_the_side() {
        let t = tokenizer().to_string();
        let err = TokenizerIdentity::of_json(t.as_bytes(), b"{\"model\": {}}").expect_err("no vocab");
        assert!(err.contains("teacher"), "{err}");
        let err = TokenizerIdentity::of_json(b"not json", t.as_bytes()).expect_err("not json");
        assert!(err.contains("student"), "{err}");
    }
}
