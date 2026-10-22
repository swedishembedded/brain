// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for measuring whether an adapter as
// trained behaves as the adapter as served, for its clients. If your team needs
// expertise in train-serve agreement of large language models then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Teacher-forced per-token scoring: the readout both of this model's scoring
//! sides produce, and the record format they both consume.
//!
//! One record is a token stream and a supervision mask - exactly what
//! `data::chat::ChatSample::encode` yields for a chat record, and exactly the
//! `(ids, mask)` pair a prepared dataset holds. Position `p` (always `>= 1`)
//! is the model predicting `ids[p]` from `ids[..p]`, and it is scored when
//! `mask[p]` is set - the same convention `data::loader::TokenDataset`
//! applies when it lays a batch out for training (`y[t] = data[a + 1 + t]`
//! where `mask[a + 1 + t]`), so a score and a training loss are the same
//! quantity over the same positions.
//!
//! A record file is JSONL, one object per line, in either form:
//!
//! ```text
//! {"ids": [151644, 872, 198, ...], "mask": [false, false, true, ...]}
//! {"messages": [{"role": "user", "content": "..."}, ...]}
//! ```
//!
//! The token form is scored as written, which is the form a caller that
//! prepared its training data should emit: the trainer may have rendered a
//! record differently from its JSON (`answers_as_asked`, a no-think block),
//! and only the ids it actually trained on score the model that trained on
//! them. The chat form is encoded with the model's own tokenizer and chat
//! template through [`data::chat::ChatSample::encode`] - the one
//! implementation of "render a record the way this checkpoint does".

use std::path::Path;

use data::chat::ChatSample;
use data::chat_template::ChatTemplate;
use data::tokenizer::Tokenizer;
use model::training::TokenObjectives;

/// One predicted position's teacher-forced readout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenScore {
    /// Index into the record's token ids of the token being scored, so the
    /// prediction was made from `ids[..position]`. Never `0`: nothing predicts
    /// a record's first token.
    pub position: u32,
    /// `log p(ids[position])` under the model at `position - 1`.
    pub logprob: f32,
    /// The token the model itself picks at `position - 1` (ties broken toward
    /// the lowest token id). Equal to `ids[position]` when the model would have
    /// produced the recorded stream itself.
    pub argmax: u32,
}

/// One record to score: the encoded token ids and the mask that marks which of
/// them are supervised (see the module documentation).
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredRecord {
    pub ids: Vec<u32>,
    pub mask: Vec<bool>,
}

impl ScoredRecord {
    /// The positions this record is scored at: every `p >= 1` whose token the
    /// mask supervises. Position `0` is never among them - no logits precede
    /// the first token.
    pub fn positions(&self) -> impl Iterator<Item = u32> + '_ {
        self.mask.iter().enumerate().skip(1).filter(|(_, &supervised)| supervised).map(|(p, _)| p as u32)
    }
}

/// Read a record file (see the module documentation for the two forms), in the
/// form its first record is written in: token records as written, chat records
/// through `tok` and `tmpl`.
///
/// A record that encodes to fewer than two tokens, or to no supervised
/// position at all, is refused by record number: scoring it would silently
/// contribute nothing to a comparison.
pub fn read_records(path: &Path, tok: &dyn Tokenizer, tmpl: &ChatTemplate) -> Result<Vec<ScoredRecord>, String> {
    let raw = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let lines: Vec<&str> = raw.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let first: serde_json::Value = serde_json::from_str(lines.first().copied().unwrap_or("null")).map_err(|e| format!("{}: record 1: not JSON: {e}", path.display()))?;
    let out: Vec<ScoredRecord> = if first.get("ids").is_some() {
        lines.iter().enumerate().map(|(at, line)| token_record(line).map_err(|e| format!("{}: record {}: {e}", path.display(), at + 1))).collect::<Result<_, _>>()?
    } else {
        let samples = ChatSample::from_jsonl(path).map_err(|e| format!("{}: {e}", path.display()))?;
        samples
            .iter()
            .enumerate()
            .map(|(at, sample)| encode_record(sample, tok, tmpl).map_err(|e| format!("{}: record {}: {e}", path.display(), at + 1)))
            .collect::<Result<_, _>>()?
    };
    for (at, record) in out.iter().enumerate() {
        if record.ids.len() < 2 {
            return Err(format!("{}: record {} encodes to {} token(s): there is no position to score", path.display(), at + 1, record.ids.len()));
        }
        if record.positions().next().is_none() {
            return Err(format!("{}: record {} ({} tokens) supervises no position", path.display(), at + 1, record.ids.len()));
        }
    }
    if out.is_empty() {
        return Err(format!("{}: holds no records", path.display()));
    }
    Ok(out)
}

/// One line of the token form.
fn token_record(line: &str) -> Result<ScoredRecord, String> {
    let value: serde_json::Value = serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    let ints = |key: &str| -> Result<Vec<&serde_json::Value>, String> {
        value.get(key).and_then(|v| v.as_array()).map(|a| a.iter().collect()).ok_or_else(|| format!("no \"{key}\" array"))
    };
    let mask: Vec<bool> = ints("mask")?.into_iter().map(|v| v.as_bool().ok_or_else(|| "a mask entry is not a boolean".to_string())).collect::<Result<_, _>>()?;
    let ids: Vec<u32> = ints("ids")?
        .into_iter()
        .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or_else(|| "a token id is not a u32".to_string()))
        .collect::<Result<_, _>>()?;
    if ids.len() != mask.len() {
        return Err(format!("{} ids but {} mask entries", ids.len(), mask.len()));
    }
    Ok(ScoredRecord { ids, mask })
}

fn encode_record(sample: &ChatSample, tok: &dyn Tokenizer, tmpl: &ChatTemplate) -> Result<ScoredRecord, String> {
    let (ids, mask) = sample.encode(tok, tmpl).map_err(|e| e.to_string())?;
    Ok(ScoredRecord { ids, mask })
}

/// The readout of one logits row against the token that was recorded there:
/// its log-probability under the row's distribution, and the row's own pick.
///
/// Both come from the SAME row, so a caller that reads a served or an
/// in-process forward through this function is comparing like with like.
pub fn readout(position: u32, row: &[f32], recorded: u32) -> TokenScore {
    TokenScore { position, logprob: model::logprobs::row_logprob(row, recorded as usize), argmax: model::hostmath::argmax_row(row) }
}

/// One scoring pass over a record set, from the model's own training forward.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenScores {
    /// One entry per record, in input order: its supervised positions'
    /// readouts, ascending by position.
    pub records: Vec<Vec<TokenScore>>,
    /// Mean cross-entropy over every scored position: the same quantity the
    /// trainer reports as its loss for the same records, aggregated over the
    /// pass by each batch's own denominator.
    pub mean_loss: Option<f32>,
    /// Scored positions in all.
    pub positions: usize,
}

/// Teacher-forced per-token readouts of `records` from `m`'s own training
/// forward, with no optimizer step.
///
/// The records are laid out by [`data::loader::TokenDataset::new_examples`] and
/// walked by `example_rows` - the batching a training step and a held-out
/// evaluation already use - so a position scored here is a position trained
/// there, with the same shifted target and the same mask. `m` must be built for
/// `batch` sequences of `block` tokens and have the argmax readout on
/// ([`crate::model::Qwen35::enable_token_argmax`]); a record longer than
/// `block` is refused rather than truncated.
pub fn score_records(m: &crate::model::Qwen35, records: &[ScoredRecord], batch: u32, block: u32) -> Result<TokenScores, String> {
    if records.is_empty() || batch == 0 || block == 0 {
        return Err("qwen35 score: at least one record, a batch of one row and a block of one token are needed".to_string());
    }
    let (mut data, mut mask, mut starts) = (Vec::new(), Vec::new(), Vec::new());
    for (at, record) in records.iter().enumerate() {
        if record.ids.len() != record.mask.len() {
            return Err(format!("qwen35 score: record {} has {} ids and {} mask entries", at + 1, record.ids.len(), record.mask.len()));
        }
        if record.ids.len() > block as usize {
            return Err(format!("qwen35 score: record {} is {} tokens, longer than the {block}-token scoring row", at + 1, record.ids.len()));
        }
        starts.push(data.len());
        data.extend_from_slice(&record.ids);
        mask.extend_from_slice(&record.mask);
    }
    let cfg = data::loader::BatchConfig { batch_size: batch as usize, block_size: block as usize, ..Default::default() };
    let set = data::loader::TokenDataset::new_examples(data, mask, &starts, 0, &cfg).map_err(|e| format!("qwen35 score: {e}"))?;
    let (rows, width) = (batch as usize, block as usize);
    let mut out = TokenScores { records: vec![Vec::new(); records.len()], ..Default::default() };
    let mut weighted = 0f64;
    let mut first = 0usize;
    while first < records.len() {
        let (x, y) = set.example_rows(&cfg, first).ok_or("qwen35 score: the record set holds no examples")?;
        let y: Vec<u32> = y.into_iter().map(|v| if v < 0 { model::IGNORE } else { v as u32 }).collect();
        m.set_batch(&x, &y);
        let loss = m.forward();
        let logprobs = m.batch_token_logprobs();
        let argmax = m.batch_token_argmax().ok_or("qwen35 score: the per-token argmax readout is off on this model")?;
        let mut scored = 0usize;
        for row in 0..rows {
            let record = first + row;
            if record >= records.len() {
                break;
            }
            for t in (0..width).filter(|&t| y[row * width + t] != model::IGNORE) {
                out.records[record].push(TokenScore { position: (t + 1) as u32, logprob: logprobs[row * width + t], argmax: argmax[row * width + t] });
                scored += 1;
            }
        }
        out.positions += scored;
        // The trainer's loss is `sum(nll) / count` for this batch, so the pass's
        // mean is each batch's loss weighted by the positions it covered.
        weighted += loss as f64 * scored as f64;
        first += rows;
    }
    out.mean_loss = (out.positions > 0).then(|| (weighted / out.positions as f64) as f32);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SPEC: a record's scored positions are the mask's set positions past the
    /// first token - the same predicate a training batch applies when it lays
    /// `y` out (`data::loader::TokenDataset`).
    #[test]
    fn scored_positions_are_the_supervised_tokens_after_the_first() {
        let record = ScoredRecord { ids: vec![7, 8, 9, 10, 11], mask: vec![true, false, true, false, true] };
        assert_eq!(record.positions().collect::<Vec<u32>>(), vec![2, 4], "position 0 is never scored, and a masked token is not either");
    }

    /// SPEC: the readout is the row's log-softmax at the recorded token plus the
    /// row's own argmax, from that one row.
    #[test]
    fn a_readout_is_the_rows_own_log_probability_and_pick() {
        let row = [0.5f32, 2.0, -1.0, 1.0];
        let s = readout(3, &row, 1);
        assert_eq!((s.position, s.argmax), (3, 1));
        assert!((s.logprob - model::logprobs::row_logprob(&row, 1)).abs() < 1e-6);
    }

    /// SPEC: a token record is read as written, and a record that would score
    /// nothing - no supervised position, or too short to have one - is refused
    /// instead of silently contributing nothing to a comparison.
    #[test]
    fn token_records_are_read_as_written_and_a_record_that_scores_nothing_is_refused() {
        let tok = data::tokenizer::CharTokenizer::from_corpus("ab");
        let tmpl = ChatTemplate::compile("{{ messages[0].content }}").expect("a one-line template compiles");
        let path = std::env::temp_dir().join(format!("brain-score-records-{}.jsonl", std::process::id()));
        let read = |lines: &str| {
            std::fs::write(&path, lines).unwrap();
            read_records(&path, &tok, &tmpl)
        };
        let got = read(r#"{"ids":[4,5,6],"mask":[true,false,true]}"#).expect("a token record reads as written");
        assert_eq!(got, vec![ScoredRecord { ids: vec![4, 5, 6], mask: vec![true, false, true] }]);
        assert!(read(r#"{"ids":[4,5,6],"mask":[true,false,"yes"]}"#).is_err(), "a non-boolean mask entry is refused");
        assert!(read(r#"{"ids":[4,5],"mask":[true]}"#).is_err(), "ids and mask of different lengths are refused");
        assert!(read(r#"{"ids":[4],"mask":[true]}"#).is_err(), "a one-token record has no position to score");
        assert!(read(r#"{"ids":[4,5,6],"mask":[false,false,false]}"#).is_err(), "a record that supervises nothing is refused");
        std::fs::remove_file(&path).ok();
    }
}
