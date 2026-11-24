// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training pipelines that teach speech
// synthesizers new languages and voices for its clients. If your team needs
// expertise in speech-model fine-tuning, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The training shard: recorded speech as the Talker and code predictor read it.
//!
//! One shard is a directory of a few column files, never one file per
//! utterance. Each record is one utterance:
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
//! | file | contents |
//! |---|---|
//! | `meta.json` | [`ShardMeta`] |
//! | `text.u32`, `text_off.u64` | concatenated token ids; `records + 1` offsets into them |
//! | `codes.u16`, `codes_off.u64` | concatenated frames; `records + 1` frame offsets |
//! | `xvec.f32` | `records * xvector_dim` |
//! | `voice.u32`, `language.u32`, `complete.u32` | one per record (`voice` is [`NO_VOICE`] for none) |
//! | `ids.jsonl` | one [`RecordId`] per record, for provenance |
//!
//! Everything a shard holds is checked when it is opened ([`Shard::open`]):
//! offsets start at zero, never decrease and end at the column's length, codes
//! lie inside the codebook, voices and languages resolve, and the per-record
//! columns agree on the record count. A shard that fails is refused naming the
//! file and the record, never trained on.

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

fn invalid(path: &Path, what: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("{}: {what}", path.display()))
}

/// Offsets must start at 0, never decrease, end at `len` and number `records + 1`.
fn check_offsets(path: &Path, off: &[u64], records: u64, len: usize) -> io::Result<()> {
    if off.len() as u64 != records + 1 {
        return Err(invalid(path, format!("{} offsets for {records} records, expected {}", off.len(), records + 1)));
    }
    if off[0] != 0 {
        return Err(invalid(path, format!("the first offset is {}, expected 0", off[0])));
    }
    if let Some(r) = off.windows(2).position(|w| w[1] < w[0]) {
        return Err(invalid(path, format!("record {r}: offset {} is below the one before it, {}", off[r + 1], off[r])));
    }
    if off[off.len() - 1] != len as u64 {
        return Err(invalid(path, format!("the last offset is {}, the column holds {len}", off[off.len() - 1])));
    }
    Ok(())
}

impl Shard {
    /// Read and validate the shard at `dir`.
    pub fn open(dir: &Path) -> io::Result<Shard> {
        use data::binio::{read_f32_bin, read_u16_bin, read_u32_bin, read_u64_bin};
        let meta_path = dir.join("meta.json");
        let meta: ShardMeta = serde_json::from_slice(&fs::read(&meta_path)?).map_err(|e| invalid(&meta_path, e))?;
        if meta.format != FORMAT {
            return Err(invalid(&meta_path, format!("format {:?}, expected {FORMAT:?}", meta.format)));
        }
        if meta.codebooks == 0 || meta.codebook_size == 0 || meta.codebook_size > u32::from(u16::MAX) + 1 || meta.xvector_dim == 0 {
            return Err(invalid(&meta_path, "codebooks, codebook_size (at most 65536) and xvector_dim must be positive"));
        }
        if meta.languages.is_empty() {
            return Err(invalid(&meta_path, "a shard names at least one language"));
        }
        let n = meta.records;
        let text = read_u32_bin(&dir.join("text.u32"))?;
        let text_off = read_u64_bin(&dir.join("text_off.u64"))?;
        check_offsets(&dir.join("text_off.u64"), &text_off, n, text.len())?;
        let codes = read_u16_bin(&dir.join("codes.u16"))?;
        let codes_off = read_u64_bin(&dir.join("codes_off.u64"))?;
        let q = meta.codebooks as usize;
        if !codes.len().is_multiple_of(q) {
            return Err(invalid(&dir.join("codes.u16"), format!("{} codes are not whole frames of {q}", codes.len())));
        }
        check_offsets(&dir.join("codes_off.u64"), &codes_off, n, codes.len() / q)?;
        if let Some(i) = codes.iter().position(|&c| u32::from(c) >= meta.codebook_size) {
            let record = codes_off.partition_point(|&o| o as usize * q <= i) - 1;
            return Err(invalid(&dir.join("codes.u16"), format!("record {record}: code {} is outside the codebook of {}", codes[i], meta.codebook_size)));
        }
        let xvec = read_f32_bin(&dir.join("xvec.f32"))?;
        if xvec.len() as u64 != n * u64::from(meta.xvector_dim) {
            return Err(invalid(&dir.join("xvec.f32"), format!("{} values, expected {n} x {}", xvec.len(), meta.xvector_dim)));
        }
        if let Some(i) = xvec.iter().position(|x| !x.is_finite()) {
            return Err(invalid(&dir.join("xvec.f32"), format!("record {}: a non-finite x-vector value", i / meta.xvector_dim as usize)));
        }
        let per_record = |name: &str| -> io::Result<Vec<u32>> {
            let v = read_u32_bin(&dir.join(name))?;
            if v.len() as u64 != n {
                return Err(invalid(&dir.join(name), format!("{} values for {n} records", v.len())));
            }
            Ok(v)
        };
        let voice = per_record("voice.u32")?;
        if let Some(r) = voice.iter().position(|&v| v != NO_VOICE && v as usize >= meta.voices.len()) {
            return Err(invalid(&dir.join("voice.u32"), format!("record {r}: voice {} is not one of the {} named", voice[r], meta.voices.len())));
        }
        let language = per_record("language.u32")?;
        if let Some(r) = language.iter().position(|&l| l as usize >= meta.languages.len()) {
            return Err(invalid(&dir.join("language.u32"), format!("record {r}: language {} is not one of the {} named", language[r], meta.languages.len())));
        }
        let complete = per_record("complete.u32")?;
        if let Some(r) = complete.iter().position(|&c| c > 1) {
            return Err(invalid(&dir.join("complete.u32"), format!("record {r}: complete is {}, expected 0 or 1", complete[r])));
        }
        let ids_path = dir.join("ids.jsonl");
        let ids: Vec<RecordId> = fs::read_to_string(&ids_path)?
            .lines()
            .enumerate()
            .map(|(r, l)| serde_json::from_str(l).map_err(|e| invalid(&ids_path, format!("record {r}: {e}"))))
            .collect::<io::Result<_>>()?;
        if ids.len() as u64 != n {
            return Err(invalid(&ids_path, format!("{} ids for {n} records", ids.len())));
        }
        if let Some(r) = (0..n as usize).find(|&r| codes_off[r + 1] == codes_off[r] || text_off[r + 1] == text_off[r]) {
            return Err(invalid(dir, format!("record {r} ({}): no frames or no text", ids[r].id)));
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

/// Builds a shard in memory and writes it once complete.
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
    ids: Vec<String>,
}

impl ShardWriter {
    /// A writer for records described by `meta` (its `records` is ignored and
    /// set by [`ShardWriter::finish`]).
    #[must_use]
    pub fn new(meta: ShardMeta) -> ShardWriter {
        ShardWriter { meta, text: Vec::new(), text_off: vec![0], codes: Vec::new(), codes_off: vec![0], xvec: Vec::new(), voice: Vec::new(), language: Vec::new(), complete: Vec::new(), ids: Vec::new() }
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
        self.ids.push(serde_json::to_string(&r.id).map_err(|e| e.to_string())?);
        Ok(())
    }

    /// Write the shard into `dir` (created; an existing shard there is refused).
    pub fn finish(mut self, dir: &Path) -> io::Result<()> {
        use data::binio::{write_f32_bin, write_u16_bin, write_u32_bin, write_u64_bin};
        if dir.join("meta.json").exists() {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{}: a shard is already there", dir.display())));
        }
        fs::create_dir_all(dir)?;
        self.meta.records = self.voice.len() as u64;
        write_u32_bin(&dir.join("text.u32"), &self.text)?;
        write_u64_bin(&dir.join("text_off.u64"), &self.text_off)?;
        write_u16_bin(&dir.join("codes.u16"), &self.codes)?;
        write_u64_bin(&dir.join("codes_off.u64"), &self.codes_off)?;
        write_f32_bin(&dir.join("xvec.f32"), &self.xvec)?;
        write_u32_bin(&dir.join("voice.u32"), &self.voice)?;
        write_u32_bin(&dir.join("language.u32"), &self.language)?;
        write_u32_bin(&dir.join("complete.u32"), &self.complete)?;
        let mut ids = self.ids.join("\n");
        ids.push('\n');
        fs::write(dir.join("ids.jsonl"), ids)?;
        // meta.json last: a shard without it is an unfinished write, never opened.
        fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&self.meta).map_err(io::Error::other)?)
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

    fn written(dir: &Path) -> Shard {
        let mut w = ShardWriter::new(meta());
        w.push(&Record { id: id("a"), text_ids: &[1, 2, 3], codes: &[0, 1, 2, 3], xvector: &[0.1, 0.2, 0.3], voice: 0, language: 0, complete: true }).unwrap();
        w.push(&Record { id: id("b"), text_ids: &[4], codes: &[7, 7], xvector: &[1.0, 1.0, 1.0], voice: NO_VOICE, language: 0, complete: false }).unwrap();
        w.finish(dir).unwrap();
        Shard::open(dir).unwrap()
    }

    #[test]
    fn a_written_shard_reads_back_record_for_record() {
        let dir = tempdir("roundtrip");
        let s = written(&dir);
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
    fn a_corrupted_shard_is_refused_naming_the_file_and_record() {
        let dir = tempdir("corrupt");
        written(&dir);
        let mut codes = data::binio::read_u16_bin(&dir.join("codes.u16")).unwrap();
        codes[4] = 9;
        data::binio::write_u16_bin(&dir.join("codes.u16"), &codes).unwrap();
        let e = Shard::open(&dir).unwrap_err().to_string();
        assert!(e.contains("codes.u16") && e.contains("record 1") && e.contains("outside the codebook"), "{e}");

        let dir = tempdir("offsets");
        written(&dir);
        data::binio::write_u64_bin(&dir.join("text_off.u64"), &[0, 3]).unwrap();
        let e = Shard::open(&dir).unwrap_err().to_string();
        assert!(e.contains("text_off.u64") && e.contains("2 offsets for 2 records"), "{e}");

        let dir = tempdir("voice");
        written(&dir);
        data::binio::write_u32_bin(&dir.join("voice.u32"), &[0, 5]).unwrap();
        let e = Shard::open(&dir).unwrap_err().to_string();
        assert!(e.contains("voice.u32") && e.contains("record 1"), "{e}");

        let dir = tempdir("meta");
        written(&dir);
        let mut m: serde_json::Value = serde_json::from_slice(&fs::read(dir.join("meta.json")).unwrap()).unwrap();
        m["extra"] = serde_json::json!(1);
        fs::write(dir.join("meta.json"), m.to_string()).unwrap();
        let e = Shard::open(&dir).unwrap_err().to_string();
        assert!(e.contains("meta.json") && e.contains("unknown field"), "{e}");
    }

    #[test]
    fn a_shard_is_never_written_over_another() {
        let dir = tempdir("twice");
        written(&dir);
        let e = ShardWriter::new(meta()).finish(&dir).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::AlreadyExists);
    }

    fn tempdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("qwen3tts-shard-{tag}-{}-{:?}", std::process::id(), std::thread::current().id()));
        let _ = fs::remove_dir_all(&d);
        d
    }
}
