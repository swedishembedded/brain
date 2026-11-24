// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The training shard: recorded speech as the Talker and code predictor read it.
//!
//! One shard is one file (an artifact a training job takes by digest), never
//! one file per utterance. Each record is one utterance:
//!
//! * the tokenised assistant turn (`<|im_start|>assistant\n{text}<|im_end|>\n
//!   <|im_start|>assistant\n`, the ids generation tokenises the request into),
//! * its acoustic codes, `[frames, codebooks]` row-major,
//! * the frozen speaker encoder's x-vector of the recording,
//! * the voice it trains (an index into [`ShardMeta::voices`], or none for a
//!   speaker that only conditions through its x-vector),
//! * the language it is spoken in (an index into [`ShardMeta::languages`]),
//! * whether the recording ends where the utterance ends (only then is the
//!   end of speech a training target).
//!
//! The file is the 8 bytes [`MAGIC`], a little-endian `u64` header length, the
//! JSON header `{"meta": ShardMeta, "columns": {name: {"dtype", "offset",
//! "len"}}}` and the columns, each starting at a multiple of 8 bytes counted
//! from the end of the header; `len` counts elements, all little-endian:
//!
//! | column | dtype | contents |
//! |---|---|---|
//! | `text`, `text_off` | `u32`, `u64` | concatenated token ids; `records + 1` offsets into them |
//! | `codes`, `codes_off` | `u16`, `u64` | concatenated frames; `records + 1` frame offsets |
//! | `xvec` | `f32` | `records * xvector_dim` |
//! | `voice`, `language`, `complete` | `u32` | one per record (`voice` is [`NO_VOICE`] for none) |
//! | `ids` | `u8` | one JSON [`RecordId`] per record, one per line |
//!
//! Everything a shard holds is checked when it is read ([`Shard::from_bytes`]):
//! column ranges lie inside the file, offsets start at zero, never decrease and
//! end at the column's length, codes lie inside the codebook, voices and
//! languages resolve, and the per-record columns agree on the record count. A
//! shard that fails is refused naming the column and the record, never trained
//! on.

use std::fs;
use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// The `format` string of the current shard layout.
pub const FORMAT: &str = "brain-tts-shard-v1";

/// A record that trains no named voice.
pub const NO_VOICE: u32 = u32::MAX;

/// What a shard holds, and the vocabulary its per-record indices refer to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShardMeta {
    /// Always [`FORMAT`].
    pub format: String,
    /// Codebooks per frame.
    pub codebooks: u32,
    /// The codebook size: every code is below it.
    pub codebook_size: u32,
    /// The x-vector width.
    pub xvector_dim: u32,
    /// The acoustic frame rate, frames per second.
    pub frame_rate: f32,
    /// Language names, as a checkpoint's `codec_language_id` keys them.
    pub languages: Vec<String>,
    /// Named voices, as a checkpoint's `spk_id` keys them.
    pub voices: Vec<String>,
    /// How many records the shard holds.
    pub records: u64,
    /// Content digests of the models that produced it (codec, speaker encoder,
    /// tokenizer), so a shard is never mixed with codes from another codec.
    pub producers: std::collections::BTreeMap<String, String>,
}

/// Where a record came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordId {
    /// Unique within the corpus.
    pub id: String,
    /// The source speaker, as the source dataset names them.
    pub speaker: String,
    /// The source dataset.
    pub source: String,
    /// The transcript the token ids were made from.
    pub text: String,
}

/// One record to write.
#[derive(Clone, Debug)]
pub struct Record<'a> {
    pub id: RecordId,
    pub text_ids: &'a [u32],
    /// `[frames, codebooks]` row-major.
    pub codes: &'a [u16],
    pub xvector: &'a [f32],
    pub voice: u32,
    pub language: u32,
    pub complete: bool,
}

/// A shard read into memory (the whole corpus fits: 100 hours is about 4.5M
/// frames, 144 MB of codes).
#[derive(Clone, Debug)]
pub struct Shard {
    pub meta: ShardMeta,
    text: Vec<u32>,
    text_off: Vec<u64>,
    codes: Vec<u16>,
    codes_off: Vec<u64>,
    xvec: Vec<f32>,
    voice: Vec<u32>,
    language: Vec<u32>,
    complete: Vec<u32>,
    ids: Vec<RecordId>,
}

/// A borrowed view of one record.
#[derive(Clone, Copy, Debug)]
pub struct RecordView<'a> {
    pub id: &'a RecordId,
    pub text_ids: &'a [u32],
    pub codes: &'a [u16],
    pub xvector: &'a [f32],
    /// `None` for a record that trains no named voice.
    pub voice: Option<u32>,
    pub language: u32,
    pub complete: bool,
}

impl RecordView<'_> {
    /// Frames in the record.
    #[must_use]
    pub fn frames(&self, codebooks: u32) -> usize {
        self.codes.len() / codebooks as usize
    }
}

/// The first 8 bytes of a shard file.
pub const MAGIC: &[u8; 8] = b"BTTSHRD1";

/// Where one column lies in the file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Column {
    pub dtype: String,
    /// Bytes from the end of the header; a multiple of 8.
    pub offset: u64,
    /// Elements.
    pub len: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    meta: ShardMeta,
    columns: std::collections::BTreeMap<String, Column>,
}

/// The columns, their element type and width, in the order they are written.
const COLUMNS: &[(&str, &str, usize)] = &[
    ("text", "u32", 4),
    ("text_off", "u64", 8),
    ("codes", "u16", 2),
    ("codes_off", "u64", 8),
    ("xvec", "f32", 4),
    ("voice", "u32", 4),
    ("language", "u32", 4),
    ("complete", "u32", 4),
    ("ids", "u8", 1),
];

fn invalid(what: impl std::fmt::Display) -> String {
    format!("{FORMAT}: {what}")
}

/// Offsets must start at 0, never decrease, end at `len` and number `records + 1`.
fn check_offsets(column: &str, off: &[u64], records: u64, len: usize) -> Result<(), String> {
    if off.len() as u64 != records + 1 {
        return Err(invalid(format!("{column}: {} offsets for {records} records, expected {}", off.len(), records + 1)));
    }
    if off[0] != 0 {
        return Err(invalid(format!("{column}: the first offset is {}, expected 0", off[0])));
    }
    if let Some(r) = off.windows(2).position(|w| w[1] < w[0]) {
        return Err(invalid(format!("{column}: record {r}: offset {} is below the one before it, {}", off[r + 1], off[r])));
    }
    if off[off.len() - 1] != len as u64 {
        return Err(invalid(format!("{column}: the last offset is {}, the column holds {len}", off[off.len() - 1])));
    }
    Ok(())
}

/// The bytes of column `name`, checked against its declared type and the file.
fn column<'a>(columns: &std::collections::BTreeMap<String, Column>, body: &'a [u8], name: &str, dtype: &str, width: usize) -> Result<&'a [u8], String> {
    let c = columns.get(name).ok_or_else(|| invalid(format!("no column {name:?}")))?;
    if c.dtype != dtype {
        return Err(invalid(format!("{name}: dtype {:?}, expected {dtype:?}", c.dtype)));
    }
    if !c.offset.is_multiple_of(8) {
        return Err(invalid(format!("{name}: offset {} is not 8-byte aligned", c.offset)));
    }
    let bytes = c.len.checked_mul(width as u64).ok_or_else(|| invalid(format!("{name}: length overflows")))?;
    let end = c.offset.checked_add(bytes).filter(|&e| e <= body.len() as u64).ok_or_else(|| invalid(format!("{name}: {} elements at {} run past the end of the file", c.len, c.offset)))?;
    Ok(&body[c.offset as usize..end as usize])
}

fn le<const N: usize, T>(bytes: &[u8], f: fn([u8; N]) -> T) -> Vec<T> {
    bytes.as_chunks::<N>().0.iter().map(|c| f(*c)).collect()
}

impl Shard {
    /// Read and validate the shard file at `path`.
    pub fn open(path: &Path) -> io::Result<Shard> {
        let bytes = fs::read(path)?;
        Shard::from_bytes(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", path.display())))
    }

    /// Read and validate a shard from its bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Shard, String> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err(invalid("not a shard file (bad magic)"));
        }
        let hlen = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes"));
        let hend = 16u64.checked_add(hlen).filter(|&e| e <= bytes.len() as u64).ok_or_else(|| invalid(format!("a header of {hlen} bytes runs past the end of the file")))? as usize;
        let header: Header = serde_json::from_slice(&bytes[16..hend]).map_err(|e| invalid(format!("header: {e}")))?;
        let meta = header.meta;
        if meta.format != FORMAT {
            return Err(invalid(format!("meta: format {:?}, expected {FORMAT:?}", meta.format)));
        }
        if meta.codebooks == 0 || meta.codebook_size == 0 || meta.codebook_size > u32::from(u16::MAX) + 1 || meta.xvector_dim == 0 {
            return Err(invalid("meta: codebooks, codebook_size (at most 65536) and xvector_dim must be positive"));
        }
        if meta.languages.is_empty() {
            return Err(invalid("meta: a shard names at least one language"));
        }
        if let Some(extra) = header.columns.keys().find(|k| !COLUMNS.iter().any(|(n, _, _)| n == k)) {
            return Err(invalid(format!("unknown column {extra:?}")));
        }
        let body = &bytes[hend..];
        let col = |name: &str| -> Result<&[u8], String> {
            let (_, dtype, width) = COLUMNS.iter().find(|(n, _, _)| *n == name).expect("a known column");
            column(&header.columns, body, name, dtype, *width)
        };
        let n = meta.records;
        let q = meta.codebooks as usize;
        let text = le(col("text")?, u32::from_le_bytes);
        let text_off = le(col("text_off")?, u64::from_le_bytes);
        check_offsets("text_off", &text_off, n, text.len())?;
        let codes = le(col("codes")?, u16::from_le_bytes);
        if !codes.len().is_multiple_of(q) {
            return Err(invalid(format!("codes: {} codes are not whole frames of {q}", codes.len())));
        }
        let codes_off = le(col("codes_off")?, u64::from_le_bytes);
        check_offsets("codes_off", &codes_off, n, codes.len() / q)?;
        if let Some(i) = codes.iter().position(|&c| u32::from(c) >= meta.codebook_size) {
            let record = codes_off.partition_point(|&o| o as usize * q <= i) - 1;
            return Err(invalid(format!("codes: record {record}: code {} is outside the codebook of {}", codes[i], meta.codebook_size)));
        }
        let xvec = le(col("xvec")?, f32::from_le_bytes);
        if xvec.len() as u64 != n * u64::from(meta.xvector_dim) {
            return Err(invalid(format!("xvec: {} values, expected {n} x {}", xvec.len(), meta.xvector_dim)));
        }
        if let Some(i) = xvec.iter().position(|x| !x.is_finite()) {
            return Err(invalid(format!("xvec: record {}: a non-finite x-vector value", i / meta.xvector_dim as usize)));
        }
        let per_record = |name: &str| -> Result<Vec<u32>, String> {
            let v = le(col(name)?, u32::from_le_bytes);
            if v.len() as u64 != n {
                return Err(invalid(format!("{name}: {} values for {n} records", v.len())));
            }
            Ok(v)
        };
        let voice = per_record("voice")?;
        if let Some(r) = voice.iter().position(|&v| v != NO_VOICE && v as usize >= meta.voices.len()) {
            return Err(invalid(format!("voice: record {r}: voice {} is not one of the {} named", voice[r], meta.voices.len())));
        }
        let language = per_record("language")?;
        if let Some(r) = language.iter().position(|&l| l as usize >= meta.languages.len()) {
            return Err(invalid(format!("language: record {r}: language {} is not one of the {} named", language[r], meta.languages.len())));
        }
        let complete = per_record("complete")?;
        if let Some(r) = complete.iter().position(|&c| c > 1) {
            return Err(invalid(format!("complete: record {r}: complete is {}, expected 0 or 1", complete[r])));
        }
        let ids_text = std::str::from_utf8(col("ids")?).map_err(|e| invalid(format!("ids: {e}")))?;
        let ids: Vec<RecordId> = ids_text
            .lines()
            .enumerate()
            .map(|(r, l)| serde_json::from_str(l).map_err(|e| invalid(format!("ids: record {r}: {e}"))))
            .collect::<Result<_, _>>()?;
        if ids.len() as u64 != n {
            return Err(invalid(format!("ids: {} ids for {n} records", ids.len())));
        }
        if let Some(r) = (0..n as usize).find(|&r| codes_off[r + 1] == codes_off[r] || text_off[r + 1] == text_off[r]) {
            return Err(invalid(format!("record {r} ({}): no frames or no text", ids[r].id)));
        }
        Ok(Shard { meta, text, text_off, codes, codes_off, xvec, voice, language, complete, ids })
    }

    /// How many records the shard holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.voice.len()
    }

    /// Whether the shard holds no records.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.voice.is_empty()
    }

    /// Record `r`.
    #[must_use]
    pub fn record(&self, r: usize) -> RecordView<'_> {
        let q = self.meta.codebooks as usize;
        let d = self.meta.xvector_dim as usize;
        RecordView {
            id: &self.ids[r],
            text_ids: &self.text[self.text_off[r] as usize..self.text_off[r + 1] as usize],
            codes: &self.codes[self.codes_off[r] as usize * q..self.codes_off[r + 1] as usize * q],
            xvector: &self.xvec[r * d..(r + 1) * d],
            voice: (self.voice[r] != NO_VOICE).then_some(self.voice[r]),
            language: self.language[r],
            complete: self.complete[r] == 1,
        }
    }

    /// Frames in record `r`.
    #[must_use]
    pub fn frames(&self, r: usize) -> usize {
        (self.codes_off[r + 1] - self.codes_off[r]) as usize
    }
}

/// Builds a shard in memory.
pub struct ShardWriter {
    meta: ShardMeta,
    text: Vec<u32>,
    text_off: Vec<u64>,
    codes: Vec<u16>,
    codes_off: Vec<u64>,
    xvec: Vec<f32>,
    voice: Vec<u32>,
    language: Vec<u32>,
    complete: Vec<u32>,
    ids: String,
}

impl ShardWriter {
    /// A writer for records described by `meta` (its `records` is ignored and
    /// set when the shard is written).
    #[must_use]
    pub fn new(meta: ShardMeta) -> ShardWriter {
        ShardWriter { meta, text: Vec::new(), text_off: vec![0], codes: Vec::new(), codes_off: vec![0], xvec: Vec::new(), voice: Vec::new(), language: Vec::new(), complete: Vec::new(), ids: String::new() }
    }

    /// Append one record; a record that does not fit the meta is refused.
    pub fn push(&mut self, r: &Record<'_>) -> Result<(), String> {
        let q = self.meta.codebooks as usize;
        let who = &r.id.id;
        if r.codes.is_empty() || !r.codes.len().is_multiple_of(q) {
            return Err(format!("{who}: {} codes are not one or more whole frames of {q}", r.codes.len()));
        }
        if let Some(c) = r.codes.iter().find(|&&c| u32::from(c) >= self.meta.codebook_size) {
            return Err(format!("{who}: code {c} is outside the codebook of {}", self.meta.codebook_size));
        }
        if r.text_ids.is_empty() {
            return Err(format!("{who}: no text"));
        }
        if r.xvector.len() != self.meta.xvector_dim as usize || r.xvector.iter().any(|x| !x.is_finite()) {
            return Err(format!("{who}: the x-vector has {} values (expected {}) or a non-finite one", r.xvector.len(), self.meta.xvector_dim));
        }
        if r.voice != NO_VOICE && r.voice as usize >= self.meta.voices.len() {
            return Err(format!("{who}: voice {} is not one of the {} named", r.voice, self.meta.voices.len()));
        }
        if r.language as usize >= self.meta.languages.len() {
            return Err(format!("{who}: language {} is not one of the {} named", r.language, self.meta.languages.len()));
        }
        self.text.extend_from_slice(r.text_ids);
        self.text_off.push(self.text.len() as u64);
        self.codes.extend_from_slice(r.codes);
        self.codes_off.push((self.codes.len() / q) as u64);
        self.xvec.extend_from_slice(r.xvector);
        self.voice.push(r.voice);
        self.language.push(r.language);
        self.complete.push(u32::from(r.complete));
        self.ids.push_str(&serde_json::to_string(&r.id).map_err(|e| e.to_string())?);
        self.ids.push('\n');
        Ok(())
    }

    /// The shard's bytes.
    #[must_use]
    pub fn to_bytes(mut self) -> Vec<u8> {
        self.meta.records = self.voice.len() as u64;
        let raw: Vec<Vec<u8>> = vec![
            self.text.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.text_off.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.codes.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.codes_off.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.xvec.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.voice.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.language.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.complete.iter().flat_map(|v| v.to_le_bytes()).collect(),
            self.ids.into_bytes(),
        ];
        let mut columns = std::collections::BTreeMap::new();
        let mut body = Vec::new();
        for ((name, dtype, width), bytes) in COLUMNS.iter().zip(&raw) {
            columns.insert((*name).to_string(), Column { dtype: (*dtype).to_string(), offset: body.len() as u64, len: (bytes.len() / width) as u64 });
            body.extend_from_slice(bytes);
            body.resize(body.len().next_multiple_of(8), 0);
        }
        let header = serde_json::to_vec(&Header { meta: self.meta, columns }).expect("a shard header serialises");
        let mut out = Vec::with_capacity(16 + header.len() + body.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&body);
        out
    }

    /// Write the shard to `path`; an existing file there is refused.
    pub fn finish(self, path: &Path) -> io::Result<()> {
        use std::io::Write;
        let mut f = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
        f.write_all(&self.to_bytes())?;
        f.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> ShardMeta {
        ShardMeta {
            format: FORMAT.into(),
            codebooks: 2,
            codebook_size: 8,
            xvector_dim: 3,
            frame_rate: 12.5,
            languages: vec!["swedish".into()],
            voices: vec!["astrid".into()],
            records: 0,
            producers: Default::default(),
        }
    }

    fn id(n: &str) -> RecordId {
        RecordId { id: n.into(), speaker: "s1".into(), source: "test".into(), text: "hej".into() }
    }

    fn writer() -> ShardWriter {
        let mut w = ShardWriter::new(meta());
        w.push(&Record { id: id("a"), text_ids: &[1, 2, 3], codes: &[0, 1, 2, 3], xvector: &[0.1, 0.2, 0.3], voice: 0, language: 0, complete: true }).unwrap();
        w.push(&Record { id: id("b"), text_ids: &[4], codes: &[7, 7], xvector: &[1.0, 1.0, 1.0], voice: NO_VOICE, language: 0, complete: false }).unwrap();
        w
    }

    /// The header and body of a written shard, for tests that corrupt one.
    fn parts() -> (serde_json::Value, Vec<u8>) {
        let bytes = writer().to_bytes();
        let hlen = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        (serde_json::from_slice(&bytes[16..16 + hlen]).unwrap(), bytes[16 + hlen..].to_vec())
    }

    fn assemble(header: &serde_json::Value, body: &[u8]) -> Vec<u8> {
        let h = serde_json::to_vec(header).unwrap();
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&(h.len() as u64).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn a_written_shard_reads_back_record_for_record() {
        let s = Shard::from_bytes(&writer().to_bytes()).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.meta.records, 2);
        let a = s.record(0);
        assert_eq!((a.text_ids, a.codes, a.xvector, a.voice, a.complete), (&[1u32, 2, 3][..], &[0u16, 1, 2, 3][..], &[0.1f32, 0.2, 0.3][..], Some(0), true));
        assert_eq!(a.frames(2), 2);
        let b = s.record(1);
        assert_eq!((b.text_ids, b.codes, b.voice, b.complete, b.id.id.as_str()), (&[4u32][..], &[7u16, 7][..], None, false, "b"));
        assert_eq!(s.frames(1), 1);
    }

    #[test]
    fn records_that_do_not_fit_the_meta_are_refused_when_written() {
        let mut w = ShardWriter::new(meta());
        let ok = Record { id: id("x"), text_ids: &[1], codes: &[0, 1], xvector: &[0.0; 3], voice: 0, language: 0, complete: true };
        for (bad, why) in [
            (Record { codes: &[0, 1, 2], ..ok.clone() }, "whole frames"),
            (Record { codes: &[0, 8], ..ok.clone() }, "outside the codebook"),
            (Record { text_ids: &[], ..ok.clone() }, "no text"),
            (Record { xvector: &[0.0; 2], ..ok.clone() }, "x-vector"),
            (Record { xvector: &[0.0, f32::NAN, 0.0], ..ok.clone() }, "non-finite"),
            (Record { voice: 1, ..ok.clone() }, "voice 1"),
            (Record { language: 1, ..ok.clone() }, "language 1"),
        ] {
            let e = w.push(&bad).unwrap_err();
            assert!(e.contains(why), "{e}");
        }
        w.push(&ok).unwrap();
    }

    #[test]
    fn a_corrupted_shard_is_refused_naming_the_column_and_record() {
        let refused = |bytes: Vec<u8>| Shard::from_bytes(&bytes).unwrap_err();

        let (h, mut body) = parts();
        let codes_at = h["columns"]["codes"]["offset"].as_u64().unwrap() as usize;
        body[codes_at + 8..codes_at + 10].copy_from_slice(&9u16.to_le_bytes());
        let e = refused(assemble(&h, &body));
        assert!(e.contains("codes: record 1") && e.contains("outside the codebook"), "{e}");

        let (mut h, body) = parts();
        h["columns"]["text_off"]["len"] = serde_json::json!(2);
        let e = refused(assemble(&h, &body));
        assert!(e.contains("text_off: 2 offsets for 2 records"), "{e}");

        let (h, mut body) = parts();
        let voice_at = h["columns"]["voice"]["offset"].as_u64().unwrap() as usize;
        body[voice_at + 4..voice_at + 8].copy_from_slice(&5u32.to_le_bytes());
        let e = refused(assemble(&h, &body));
        assert!(e.contains("voice: record 1"), "{e}");

        let (mut h, body) = parts();
        h["meta"]["extra"] = serde_json::json!(1);
        let e = refused(assemble(&h, &body));
        assert!(e.contains("header") && e.contains("unknown field"), "{e}");

        let (mut h, body) = parts();
        h["columns"]["ids"]["len"] = serde_json::json!(1u64 << 40);
        let e = refused(assemble(&h, &body));
        assert!(e.contains("ids") && e.contains("past the end"), "{e}");

        let mut truncated = writer().to_bytes();
        truncated.truncate(20);
        assert!(refused(truncated).contains("past the end"));
        assert!(refused(b"not a shard at all".to_vec()).contains("bad magic"));
    }

    #[test]
    fn a_shard_is_never_written_over_another() {
        let path = std::env::temp_dir().join(format!("qwen3tts-shard-{}.bttshrd", std::process::id()));
        let _ = fs::remove_file(&path);
        writer().finish(&path).unwrap();
        assert_eq!(Shard::open(&path).unwrap().len(), 2);
        let e = writer().finish(&path).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::AlreadyExists);
        fs::remove_file(&path).unwrap();
    }
}
