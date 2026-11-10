// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The SentencePiece-BPE tokenizer packaged in the `.cact` file.
//!
//! The blob is a 24-byte header (`n`, pad, eos, bos and unk ids as `u32`, a
//! dummy-prefix flag, a byte-fallback flag and two padding bytes) followed by
//! `n` records of an `f32` score, a type byte and a `u16`-length UTF-8 piece.
//!
//! Encoding replaces spaces with `▁`, optionally prepends one, cuts the text at
//! user-defined markers (longest first) and merges each stretch between
//! markers byte-pair style: repeatedly join the adjacent pair whose
//! concatenation is a vocabulary piece with the best score, leftmost on ties,
//! until no pair joins. A merge is always against the whole vocabulary, so a
//! control or marker spelling typed as text merges into its piece too. A
//! symbol left over that is not a piece becomes its UTF-8 bytes through the
//! `<0xXX>` pieces, or `<unk>` without byte fallback. The merge uses a
//! priority queue over a linked list of symbols, so long inputs cost
//! `O(n log n)`.
//!
//! Swedish Embedded AB implements solutions for multilingual text front ends on
//! small models for its clients. If your team needs expertise in tokenizer
//! design and verification then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;

/// Longest text [`Tokenizer::encode`] accepts, in bytes.
pub const MAX_TEXT_BYTES: usize = 16 << 20;

/// Most pieces a blob may declare.
const MAX_PIECES: usize = 1 << 20;

/// Header bytes: five `u32`, two flags, two padding bytes.
const HEADER_BYTES: usize = 24;

/// Bytes of one piece record before its text: `f32` score, `u8` type, `u16` length.
const RECORD_BYTES: usize = 7;

/// The word-boundary mark that stands for a space inside pieces.
const SPACE_MARK: char = '\u{2581}';

/// What kind of piece a vocabulary entry is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    /// An ordinary piece.
    Normal,
    /// The unknown-token piece.
    Unknown,
    /// A control piece (padding, sequence boundaries); never produced from text.
    Control,
    /// A marker matched atomically in the text (chat and language tags).
    UserDefined,
    /// A `<0xXX>` byte piece.
    Byte,
}

impl PieceType {
    fn from_byte(b: u8) -> Option<PieceType> {
        Some(match b {
            0 => PieceType::Normal,
            1 => PieceType::Unknown,
            2 => PieceType::Control,
            3 => PieceType::UserDefined,
            4 => PieceType::Byte,
            _ => return None,
        })
    }
}

/// Why a blob was refused or a text could not be processed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerError {
    /// The tokenizer blob is malformed.
    Blob(String),
    /// A text exceeds [`MAX_TEXT_BYTES`].
    TextTooLong { bytes: usize },
    /// Byte fallback needs a `<0xXX>` piece the vocabulary lacks.
    MissingBytePiece(u8),
    /// An id outside the vocabulary was decoded.
    UnknownId(u32),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Blob(m) => write!(f, "mirage tokenizer: {m}"),
            TokenizerError::TextTooLong { bytes } => write!(f, "mirage tokenizer: a {bytes}-byte text exceeds the {MAX_TEXT_BYTES}-byte limit"),
            TokenizerError::MissingBytePiece(b) => write!(f, "mirage tokenizer: byte fallback needs piece <0x{b:02X}>, which the vocabulary lacks"),
            TokenizerError::UnknownId(id) => write!(f, "mirage tokenizer: id {id} is outside the vocabulary"),
        }
    }
}

impl std::error::Error for TokenizerError {}

fn blob_error(reason: impl Into<String>) -> TokenizerError {
    TokenizerError::Blob(reason.into())
}

/// A parsed tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    pieces: Vec<String>,
    scores: Vec<f32>,
    types: Vec<PieceType>,
    ids: HashMap<String, u32>,
    byte_ids: [Option<u32>; 256],
    /// User-defined pieces, longest first (vocabulary order among equals).
    markers: Vec<u32>,
    pad: u32,
    eos: u32,
    bos: u32,
    unk: u32,
    dummy_prefix: bool,
    byte_fallback: bool,
}

fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// The byte a `<0xXX>` piece stands for.
fn byte_of_piece(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2).then(|| u8::from_str_radix(hex, 16).ok()).flatten()
}

impl Tokenizer {
    /// Parse and validate a tokenizer blob.
    pub fn from_blob(blob: &[u8]) -> Result<Tokenizer, TokenizerError> {
        if blob.len() < HEADER_BYTES {
            return Err(blob_error(format!("{} bytes is shorter than the {HEADER_BYTES}-byte header", blob.len())));
        }
        let n = le_u32(blob, 0) as usize;
        let [pad, eos, bos, unk] = [le_u32(blob, 4), le_u32(blob, 8), le_u32(blob, 12), le_u32(blob, 16)];
        let (dummy, fallback) = (blob[20], blob[21]);
        if n == 0 || n > MAX_PIECES {
            return Err(blob_error(format!("{n} pieces is outside 1..={MAX_PIECES}")));
        }
        if n.checked_mul(RECORD_BYTES).is_none_or(|need| need > blob.len() - HEADER_BYTES) {
            return Err(blob_error(format!("{n} pieces do not fit in {} bytes", blob.len())));
        }
        for (name, id) in [("pad", pad), ("eos", eos), ("bos", bos), ("unk", unk)] {
            if id as usize >= n {
                return Err(blob_error(format!("{name} id {id} is outside the {n} pieces")));
            }
        }
        if dummy > 1 || fallback > 1 {
            return Err(blob_error(format!("flags {dummy} and {fallback} are not 0 or 1")));
        }

        let mut pieces = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);
        let mut types = Vec::with_capacity(n);
        let mut ids = HashMap::with_capacity(n);
        let mut byte_ids = [None; 256];
        let mut at = HEADER_BYTES;
        for id in 0..n {
            let record = blob.get(at..at + RECORD_BYTES).ok_or_else(|| blob_error(format!("piece {id} is cut off")))?;
            let score = f32::from_bits(le_u32(record, 0));
            let kind = PieceType::from_byte(record[4]).ok_or_else(|| blob_error(format!("piece {id} has type {}, expected 0..=4", record[4])))?;
            let len = u16::from_le_bytes([record[5], record[6]]) as usize;
            let text = blob.get(at + RECORD_BYTES..at + RECORD_BYTES + len).ok_or_else(|| blob_error(format!("piece {id} is cut off")))?;
            at += RECORD_BYTES + len;
            if !score.is_finite() {
                return Err(blob_error(format!("piece {id} has a non-finite score")));
            }
            let piece = std::str::from_utf8(text).map_err(|e| blob_error(format!("piece {id} is not UTF-8: {e}")))?.to_string();
            if piece.is_empty() {
                return Err(blob_error(format!("piece {id} is empty")));
            }
            if kind == PieceType::Byte {
                let byte = byte_of_piece(&piece).ok_or_else(|| blob_error(format!("byte piece {id} `{piece}` is not <0xXX>")))?;
                if byte_ids[byte as usize].replace(id as u32).is_some() {
                    return Err(blob_error(format!("byte piece {id} `{piece}` duplicates byte {byte:#04x}")));
                }
            }
            if ids.insert(piece.clone(), id as u32).is_some() {
                return Err(blob_error(format!("duplicate piece `{piece}` at {id}")));
            }
            pieces.push(piece);
            scores.push(score);
            types.push(kind);
        }
        if at != blob.len() {
            return Err(blob_error(format!("{} trailing bytes after the last piece", blob.len() - at)));
        }
        let mut markers: Vec<u32> = (0..n as u32).filter(|&i| types[i as usize] == PieceType::UserDefined).collect();
        markers.sort_by_key(|&i| std::cmp::Reverse(pieces[i as usize].chars().count()));
        Ok(Tokenizer { pieces, scores, types, ids, byte_ids, markers, pad, eos, bos, unk, dummy_prefix: dummy == 1, byte_fallback: fallback == 1 })
    }

    /// Pieces in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
    /// Padding id.
    pub fn pad_id(&self) -> u32 {
        self.pad
    }
    /// End-of-sequence id.
    pub fn eos_id(&self) -> u32 {
        self.eos
    }
    /// Beginning-of-sequence id.
    pub fn bos_id(&self) -> u32 {
        self.bos
    }
    /// Unknown-token id.
    pub fn unk_id(&self) -> u32 {
        self.unk
    }
    /// Whether encoding prepends a word-boundary mark.
    pub fn adds_dummy_prefix(&self) -> bool {
        self.dummy_prefix
    }
    /// Whether unknown characters fall back to bytes.
    pub fn has_byte_fallback(&self) -> bool {
        self.byte_fallback
    }
    /// The text of piece `id`.
    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(String::as_str)
    }
    /// The id of a piece spelled exactly `piece`.
    pub fn id_of(&self, piece: &str) -> Option<u32> {
        self.ids.get(piece).copied()
    }
    /// The id of the `<|code|>` tag for a language code (`"en"`, `"de"`, ...).
    pub fn language_tag(&self, code: &str) -> Option<u32> {
        self.id_of(&format!("<|{code}|>"))
    }

    /// Token ids for `text`.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        if text.len() > MAX_TEXT_BYTES {
            return Err(TokenizerError::TextTooLong { bytes: text.len() });
        }
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut marked = String::with_capacity(text.len() + 3);
        if self.dummy_prefix {
            marked.push(SPACE_MARK);
        }
        marked.extend(text.chars().map(|c| if c == ' ' { SPACE_MARK } else { c }));

        let mut out = Vec::new();
        let mut segment_start = 0;
        let mut i = 0;
        while i < marked.len() {
            let rest = &marked[i..];
            match self.markers.iter().copied().find(|&m| rest.starts_with(self.pieces[m as usize].as_str())) {
                Some(marker) => {
                    self.merge_segment(&marked[segment_start..i], &mut out)?;
                    out.push(marker);
                    i += self.pieces[marker as usize].len();
                    segment_start = i;
                }
                None => i += rest.chars().next().map_or(1, char::len_utf8),
            }
        }
        self.merge_segment(&marked[segment_start..], &mut out)?;
        Ok(out)
    }

    /// Byte-pair merge one stretch of marker-free text and append its ids.
    fn merge_segment(&self, segment: &str, out: &mut Vec<u32>) -> Result<(), TokenizerError> {
        if segment.is_empty() {
            return Ok(());
        }
        const NONE: usize = usize::MAX;
        struct Symbol {
            start: usize,
            end: usize,
            prev: usize,
            next: usize,
            alive: bool,
        }
        struct Candidate {
            score: f32,
            left: usize,
            right: usize,
            left_end: usize,
            right_end: usize,
        }
        impl Ord for Candidate {
            // Best score first; among equals the leftmost pair.
            fn cmp(&self, other: &Self) -> Ordering {
                self.score.total_cmp(&other.score).then(other.left.cmp(&self.left))
            }
        }
        impl PartialOrd for Candidate {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl PartialEq for Candidate {
            fn eq(&self, other: &Self) -> bool {
                self.cmp(other) == Ordering::Equal
            }
        }
        impl Eq for Candidate {}

        let count = segment.chars().count();
        let mut symbols: Vec<Symbol> = segment
            .char_indices()
            .enumerate()
            .map(|(i, (start, c))| Symbol {
                start,
                end: start + c.len_utf8(),
                prev: if i == 0 { NONE } else { i - 1 },
                next: if i + 1 == count { NONE } else { i + 1 },
                alive: true,
            })
            .collect();
        let candidate = |symbols: &[Symbol], left: usize, right: usize| {
            let joined = &segment[symbols[left].start..symbols[right].end];
            self.ids.get(joined).map(|&id| Candidate {
                score: self.scores[id as usize],
                left,
                right,
                left_end: symbols[left].end,
                right_end: symbols[right].end,
            })
        };
        let mut heap: BinaryHeap<Candidate> = (0..count.saturating_sub(1)).filter_map(|i| candidate(&symbols, i, i + 1)).collect();
        while let Some(c) = heap.pop() {
            let live = symbols[c.left].alive
                && symbols[c.right].alive
                && symbols[c.left].next == c.right
                && symbols[c.left].end == c.left_end
                && symbols[c.right].end == c.right_end;
            if !live {
                continue;
            }
            symbols[c.left].end = symbols[c.right].end;
            symbols[c.right].alive = false;
            let next = symbols[c.right].next;
            symbols[c.left].next = next;
            if next != NONE {
                symbols[next].prev = c.left;
            }
            let prev = symbols[c.left].prev;
            heap.extend(
                [(prev != NONE).then_some((prev, c.left)), (next != NONE).then_some((c.left, next))]
                    .into_iter()
                    .flatten()
                    .filter_map(|(l, r)| candidate(&symbols, l, r)),
            );
        }

        let mut i = 0;
        while i != NONE {
            let piece = &segment[symbols[i].start..symbols[i].end];
            match self.ids.get(piece) {
                Some(&id) => out.push(id),
                None if self.byte_fallback => {
                    for b in piece.bytes() {
                        out.push(self.byte_ids[b as usize].ok_or(TokenizerError::MissingBytePiece(b))?);
                    }
                }
                None => out.push(self.unk),
            }
            i = symbols[i].next;
        }
        Ok(())
    }

    /// The text for `ids`: byte pieces are assembled, control and unknown
    /// pieces are dropped, `▁` becomes a space and invalid UTF-8 is replaced.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let mut bytes = Vec::new();
        for &id in ids {
            let i = id as usize;
            let kind = *self.types.get(i).ok_or(TokenizerError::UnknownId(id))?;
            match kind {
                PieceType::Byte => bytes.push(byte_of_piece(&self.pieces[i]).expect("byte pieces were validated at parse")),
                PieceType::Control | PieceType::Unknown => {}
                PieceType::Normal | PieceType::UserDefined => bytes.extend_from_slice(self.pieces[i].as_bytes()),
            }
        }
        let text = String::from_utf8_lossy(&bytes).replace(SPACE_MARK, " ");
        Ok(if self.dummy_prefix { text.strip_prefix(' ').unwrap_or(&text).to_string() } else { text })
    }
}
