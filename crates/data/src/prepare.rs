// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Dataset preparation — turn a source (synthetic generator, downloaded text,
//! or synthetic signal) into brain's on-disk dataset layout. Mirrors nanogpt's
//! `data_generators/*.py` `main()` flows.

use std::fs;
use std::io;
use std::path::Path;

use crate::binio::{self, Meta};
use crate::rng::Rng;
use crate::tokenizer::{CharTokenizer, Tokenizer};

/// The datasets brain can prepare (one per nanogpt generator).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dataset {
    /// Char-level tiny-shakespeare (needs `input.txt` in the output dir).
    ShakespeareChar,
    /// Synthetic `expr=result` arithmetic, char-level.
    Calculator,
    /// Synthetic `string=reverse`, char-level.
    Reverser,
    /// Synthetic `words=math`, char-level.
    Wordcalc,
    /// Tiny-shakespeare under GPT-2 BPE (needs `input.txt`); no `meta.json`.
    Gpt,
    /// Synthetic 3-phase signal, float-valued.
    Timeseries,
    /// TinyStories under a byte-level BPE trained on it (needs the raw
    /// `train.txt`/`valid.txt` in the output dir); see [`write_tinystories`].
    TinyStories,
    /// Synthetic Qwen3-TTS `text -> codebook-0 codes` stream (large-vocab `u32`).
    Tts,
    /// Synthetic object-detection scenes (RGB shapes + exact boxes). Carries the
    /// generator preset + image geometry + class count.
    Detect {
        preset: crate::gen_detect::Preset,
        h: u32,
        w: u32,
        nc: u32,
    },
}

impl Dataset {
    /// Parse the canonical dataset name used on the CLI / in directories.
    pub fn from_name(name: &str) -> Option<Dataset> {
        Some(match name {
            "shakespeare_char" | "shakespeare" => Dataset::ShakespeareChar,
            "calculator" => Dataset::Calculator,
            "reverser" => Dataset::Reverser,
            "wordcalc" => Dataset::Wordcalc,
            "gpt" => Dataset::Gpt,
            "timeseries" => Dataset::Timeseries,
            "tinystories" => Dataset::TinyStories,
            "tts" => Dataset::Tts,
            // `detect` (and per-preset names) map to a default tiny-config scene:
            // 128px, 3 classes, multi-object — the geometry the tiny YOLO uses.
            "detect" => Dataset::Detect { preset: crate::gen_detect::Preset::MultiObject, h: 128, w: 128, nc: 3 },
            other => {
                let preset = crate::gen_detect::Preset::from_name(other)?;
                Dataset::Detect { preset, h: 128, w: 128, nc: 3 }
            }
        })
    }

    /// Canonical directory/name string.
    pub fn name(self) -> &'static str {
        match self {
            Dataset::ShakespeareChar => "shakespeare_char",
            Dataset::Calculator => "calculator",
            Dataset::Reverser => "reverser",
            Dataset::Wordcalc => "wordcalc",
            Dataset::Gpt => "gpt",
            Dataset::Timeseries => "timeseries",
            Dataset::TinyStories => "tinystories",
            Dataset::Tts => "tts",
            Dataset::Detect { preset, .. } => preset.name(),
        }
    }
}

const TRAIN_SPLIT: f64 = 0.9;

/// Prepare `ds` into `dir`. `n_examples` controls the size of synthetic text
/// corpora (calculator/reverser/wordcalc) and the number of timeseries steps;
/// it is ignored for the shakespeare/gpt datasets (which consume `input.txt`).
pub fn prepare(ds: Dataset, dir: &Path, n_examples: usize, seed: u64) -> io::Result<()> {
    fs::create_dir_all(dir)?;
    match ds {
        Dataset::Calculator => {
            let mut rng = Rng::new(seed);
            let text = crate::gen_calculator::generate(n_examples, &mut rng);
            write_char_dataset(&text, dir)
        }
        Dataset::Reverser => {
            let mut rng = Rng::new(seed);
            let text = crate::gen_reverser::generate(n_examples, &mut rng);
            write_char_dataset(&text, dir)
        }
        Dataset::Wordcalc => {
            let mut rng = Rng::new(seed);
            let text = crate::gen_wordcalc::generate(n_examples, &mut rng);
            write_char_dataset(&text, dir)
        }
        Dataset::ShakespeareChar => {
            let text = read_input_txt(dir)?;
            write_char_dataset(&text, dir)
        }
        Dataset::Gpt => {
            let text = read_input_txt(dir)?;
            write_bpe_dataset(&text, dir)
        }
        Dataset::Timeseries => write_timeseries_dataset(dir, n_examples.max(1), seed),
        Dataset::TinyStories => write_tinystories(dir, n_examples),
        Dataset::Tts => {
            let mut cfg = crate::gen_tts::TtsGenConfig::default();
            if n_examples > 0 {
                cfg.examples = n_examples as u32;
            }
            crate::gen_tts::write(dir, cfg, seed).map(|_| ())
        }
        Dataset::Detect { preset, h, w, nc } => {
            crate::gen_detect::write_dataset(dir, preset, n_examples.max(1), w, h, nc, seed)
        }
    }
}

fn read_input_txt(dir: &Path) -> io::Result<String> {
    let p = dir.join("input.txt");
    fs::read_to_string(&p).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!(
                "{}: missing source text. Download tiny-shakespeare first, e.g.\n  \
                 curl -sSL -o {} https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
                p.display(),
                p.display()
            ),
        )
    })
}

/// Char-level: build vocab from the full corpus, write `meta.json`, `input.txt`,
/// and the `u16` `train.bin`/`val.bin` (split by character, as in nanogpt).
fn write_char_dataset(text: &str, dir: &Path) -> io::Result<()> {
    let tok = CharTokenizer::from_corpus(text);
    let meta = Meta {
        vocab_size: tok.vocab_size(),
        itos: tok.itos().to_vec(),
    };
    fs::write(dir.join("meta.json"), meta.to_json())?;
    fs::write(dir.join("input.txt"), text)?;

    let (train_text, val_text) = char_split(text, TRAIN_SPLIT);
    binio::write_u16_bin(&dir.join("train.bin"), &to_u16(tok.encode(&train_text)))?;
    binio::write_u16_bin(&dir.join("val.bin"), &to_u16(tok.encode(&val_text)))?;
    Ok(())
}

/// BPE: GPT-2 tokenized `train.bin`/`val.bin`, plus `input.txt`. No `meta.json`.
fn write_bpe_dataset(text: &str, dir: &Path) -> io::Result<()> {
    let tok = crate::bpe::Gpt2Bpe::new();
    fs::write(dir.join("input.txt"), text)?;
    let (train_text, val_text) = char_split(text, TRAIN_SPLIT);
    binio::write_u16_bin(&dir.join("train.bin"), &to_u16(tok.encode(&train_text)))?;
    binio::write_u16_bin(&dir.join("val.bin"), &to_u16(tok.encode(&val_text)))?;
    Ok(())
}

/// Vocabulary of the TinyStories byte-level BPE.
pub const TINYSTORIES_VOCAB: usize = 4096;
/// Training stories the TinyStories tokenizer is learned from (the first ones
/// of the training split).
pub const TINYSTORIES_TOKENIZER_STORIES: usize = 30_000;

/// The documents of a TinyStories text file: the segments between
/// [`crate::bytebpe::END_OF_TEXT`] separators, trimmed, empty ones dropped.
/// Every tokenizer that is scored on the corpus splits it with this rule, so
/// they all see the same documents and the same bytes.
pub fn split_documents(text: &str) -> impl Iterator<Item = &str> {
    text.split(crate::bytebpe::END_OF_TEXT).map(str::trim).filter(|d| !d.is_empty())
}

/// TinyStories: learn a [`TINYSTORIES_VOCAB`]-token byte-level BPE on the first
/// [`TINYSTORIES_TOKENIZER_STORIES`] training stories, then encode the first
/// `n_stories` training stories (`0`: all) and every validation story as one
/// stream each, every document preceded by the end-of-text id. Writes
/// `train.bin`/`val.bin` (`u16`), `meta.json`, `tokenizer.json` and
/// `stats.json` (document, token and byte counts).
fn write_tinystories(dir: &Path, n_stories: usize) -> io::Result<()> {
    let read = |name: &str| {
        fs::read_to_string(dir.join(name))
            .map_err(|e| io::Error::new(e.kind(), format!("{}: {e} (fetch the raw TinyStories text first)", dir.join(name).display())))
    };
    let (train, valid) = (read("train.txt")?, read("valid.txt")?);
    let sample: Vec<&str> = split_documents(&train).take(TINYSTORIES_TOKENIZER_STORIES).collect();
    let bpe = crate::bytebpe::ByteBpe::train(&sample.join("\n"), TINYSTORIES_VOCAB);
    bpe.save(&dir.join("tokenizer.json"))?;
    let encode = |docs: &mut dyn Iterator<Item = &str>| {
        let (mut ids, mut n, mut bytes) = (Vec::new(), 0usize, 0usize);
        for doc in docs {
            ids.push(bpe.end_of_text() as u16);
            ids.extend(bpe.encode(doc).into_iter().map(|t| t as u16));
            n += 1;
            bytes += doc.len();
        }
        (ids, n, bytes)
    };
    let limit = if n_stories == 0 { usize::MAX } else { n_stories };
    let (train_ids, train_docs, train_bytes) = encode(&mut split_documents(&train).take(limit));
    let (val_ids, val_docs, val_bytes) = encode(&mut split_documents(&valid));
    binio::write_u16_bin(&dir.join("train.bin"), &train_ids)?;
    binio::write_u16_bin(&dir.join("val.bin"), &val_ids)?;
    fs::write(dir.join("meta.json"), Meta::vocab_only(TINYSTORIES_VOCAB))?;
    let stats = serde_json::json!({
        "vocab_size": TINYSTORIES_VOCAB,
        "tokenizer_stories": sample.len(),
        "train": { "documents": train_docs, "tokens": train_ids.len(), "bytes": train_bytes },
        "val": { "documents": val_docs, "tokens": val_ids.len(), "bytes": val_bytes },
    });
    fs::write(dir.join("stats.json"), serde_json::to_string_pretty(&stats)?)?;
    Ok(())
}

/// Narrow small-vocab token ids (char / GPT-2 BPE, both < 65536) from the `u32`
/// [`Tokenizer`] interface to the `u16` on-disk format used by those datasets.
fn to_u16(ids: Vec<u32>) -> Vec<u16> {
    ids.into_iter().map(|t| t as u16).collect()
}

/// Time series: generate the 3-phase signal, chronological split with a gap,
/// write raw `f32` blobs + a shape `meta.json`.
fn write_timeseries_dataset(dir: &Path, n_steps: usize, seed: u64) -> io::Result<()> {
    const CONTEXT_LENGTH: usize = 60;
    let data = crate::gen_timeseries::generate(n_steps, seed);
    let (train, val) = crate::gen_timeseries::split(&data, TRAIN_SPLIT, CONTEXT_LENGTH);
    binio::write_f32_bin(&dir.join("train.f32"), &train)?;
    binio::write_f32_bin(&dir.join("val.f32"), &val)?;
    let nf = crate::gen_timeseries::N_FEATURES;
    let meta = serde_json::json!({
        "n_features": nf,
        "train_rows": train.len() / nf,
        "val_rows": val.len() / nf,
        "context_length": CONTEXT_LENGTH,
    });
    fs::write(dir.join("meta.json"), meta.to_string())?;
    Ok(())
}

/// Split text by character index (Python str slicing is by code point).
fn char_split(text: &str, train_split: f64) -> (String, String) {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let split = (n as f64 * train_split) as usize;
    let train: String = chars[..split].iter().collect();
    let val: String = chars[split..].iter().collect();
    (train, val)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("brain_data_test_{name}_{}", std::process::id()));
        p
    }

    #[test]
    fn prepares_calculator_char_dataset() {
        let dir = tmp("calc");
        let _ = fs::remove_dir_all(&dir);
        prepare(Dataset::Calculator, &dir, 200, 123).unwrap();

        // meta.json round-trips and the '=' char is in-vocab.
        let meta = Meta::from_json(&fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        assert!(meta.itos.contains(&'='));
        // train/val are non-empty u16 token arrays.
        let train = binio::read_u16_bin(&dir.join("train.bin")).unwrap();
        let val = binio::read_u16_bin(&dir.join("val.bin")).unwrap();
        assert!(!train.is_empty() && !val.is_empty());
        // all ids are within vocab.
        assert!(train.iter().all(|&t| (t as usize) < meta.vocab_size));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn prepares_timeseries_dataset() {
        let dir = tmp("ts");
        let _ = fs::remove_dir_all(&dir);
        prepare(Dataset::Timeseries, &dir, 2000, 42).unwrap();
        let train = binio::read_f32_bin(&dir.join("train.f32")).unwrap();
        assert_eq!(train.len() % crate::gen_timeseries::N_FEATURES, 0);
        assert!(!train.is_empty());
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn dataset_name_roundtrip() {
        for n in [
            "shakespeare_char",
            "calculator",
            "reverser",
            "wordcalc",
            "gpt",
            "timeseries",
        ] {
            assert_eq!(Dataset::from_name(n).unwrap().name(), if n == "shakespeare" { "shakespeare_char" } else { n });
        }
    }
}
