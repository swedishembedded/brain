// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements training-efficiency work that provably leaves every trained result where it was for
// its clients. If your team needs expertise in optimising training pipelines without regressing them then you can
// procure our services by sending an email to info@swedishembedded.com.

//! Tier-1 step parity: an optimisation of a training path is held to the path it replaces on a fixed set of real
//! training records, one optimiser step from a fixed adapter state, comparing the loss, the per-token
//! log-probabilities, every adapter gradient and the weights after the step.
//!
//! The tolerance is not chosen. It is the band of the old path against itself: the old path is run twice and the
//! difference of those two runs is measured, field by field. A deterministic backend (the CPU one) gives a band of
//! zero, and a replacement that only reorders a reduction (a trimmed batch sums its weight gradient over fewer rows)
//! cannot be bit-identical; the band is then floored at the binary32 agreement bound of the step's longest reduction
//! ([`crate::rounding::agreement_bound`], which [`crate::training::step_bound`] states per step). The new path fails
//! when any field's difference from the old path exceeds that band.
//!
//! The records are real: eight supervised agent records of a proven SFT dataset (`fixtures/a2-sft-records.jsonl`,
//! the smallest of its 420 by bytes, copied verbatim but for the em dashes, written as the JSON escape `\u2014`: the
//! same records, encoded to the same ids), each record's first answer as the trainer asks for it encoded by the
//! Qwen3-4B tokenizer and chat template into `fixtures/a2-sft-records.encoded.json`. Both files are pinned by digest, so a changed fixture fails loudly instead
//! of moving the comparison. The model that trains on them is small and Qwen3-shaped; the records' ids are mapped onto
//! a compact vocabulary of the ids they use ([`compact_vocab`]), which keeps every sequence's token identity and lets
//! the step run in seconds on the CPU.

use std::path::PathBuf;

/// The raw records, as the dataset holds them.
pub const A2_RECORDS: &str = "fixtures/a2-sft-records.jsonl";
/// Their encoding: per record the token ids and the supervision mask the Qwen3-4B template gives.
pub const A2_ENCODED: &str = "fixtures/a2-sft-records.encoded.json";
/// The blake3 digest of [`A2_RECORDS`].
pub const A2_RECORDS_DIGEST: &str = "blake3:4f20c36087cc2cf748bf2a94b37036ff111b3e99b5901b67ccab60b33ba4c207";
/// The blake3 digest of [`A2_ENCODED`].
pub const A2_ENCODED_DIGEST: &str = "blake3:bc5a6ea81da2ae2a6c947e3652f5d9c5dc94372e52a483a2716b509732e4a753";
/// The dataset the records were copied from.
pub const A2_DATASET: &str = "blake3:e2eb1efb27e7c3d8e144448a4dad3b7c9806e8c680c2a838aae2eca2620d5388";

/// The path of a fixture of this crate.
pub fn fixture_path(rel: &str) -> PathBuf {
    PathBuf::from(crate::repo_path("crates/testutil")).join(rel)
}

/// `blake3:<hex>` of `bytes`.
pub fn digest(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

/// The bytes of a fixture, refused unless they have the pinned digest.
pub fn pinned(rel: &str, want: &str) -> Vec<u8> {
    let path = fixture_path(rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("step parity fixture {}: {e}", path.display()));
    let got = digest(&bytes);
    assert_eq!(got, want, "step parity fixture {} changed: digest {got}, pinned {want}", path.display());
    bytes
}

/// One record encoded: its token ids and which of them are supervised.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Encoded {
    /// `blake3:` of the record's line in [`A2_RECORDS`].
    pub source: String,
    pub ids: Vec<u32>,
    /// One character per id, `1` supervised, `0` not.
    pub mask: String,
}

impl Encoded {
    pub fn mask(&self) -> Vec<bool> {
        self.mask.bytes().map(|c| c == b'1').collect()
    }

    /// The record up to the end of its first supervised turn: `(prompt, answer)`, the ids before the first supervised
    /// id and the first run of supervised ids. A training row cut here supervises its last positions only up to the
    /// answer's end, the shape of a prompt-plus-completion example.
    pub fn first_answer(&self) -> (Vec<u32>, Vec<u32>) {
        let mask = self.mask();
        let start = mask.iter().position(|&m| m).expect("a fixture record supervises a turn");
        let end = start + mask[start..].iter().position(|&m| !m).unwrap_or(mask.len() - start);
        (self.ids[..start].to_vec(), self.ids[start..end].to_vec())
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct EncodedFile {
    dataset: String,
    template: String,
    records: Vec<Encoded>,
}

/// The eight encoded records, after checking both fixtures' digests and that each encoding names its raw line.
pub fn a2_records() -> Vec<Encoded> {
    let raw = pinned(A2_RECORDS, A2_RECORDS_DIGEST);
    let file: EncodedFile = serde_json::from_slice(&pinned(A2_ENCODED, A2_ENCODED_DIGEST)).expect("the encoded fixture parses");
    assert_eq!(file.dataset, A2_DATASET, "the encoded fixture names its dataset");
    assert!(!file.template.is_empty(), "the encoded fixture names its template");
    let lines: Vec<String> = raw.split(|&b| b == b'\n').filter(|l| !l.is_empty()).map(digest).collect();
    assert_eq!(lines, file.records.iter().map(|r| r.source.clone()).collect::<Vec<_>>(), "one encoding per raw record, in order");
    for r in &file.records {
        assert_eq!(r.ids.len(), r.mask.len(), "{}: one mask entry per id", r.source);
    }
    file.records
}

/// Map every id in `rows` onto `0..n` by its rank among the distinct ids they use, and return `n`: the sequences keep
/// their token identity (equal ids stay equal, distinct ones distinct) in a vocabulary a small model trains in
/// seconds.
pub fn compact_vocab(rows: &mut [Vec<u32>]) -> u32 {
    let mut used: Vec<u32> = rows.iter().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    for row in rows.iter_mut() {
        for id in row.iter_mut() {
            *id = used.binary_search(id).expect("collected above") as u32;
        }
    }
    used.len() as u32
}

/// What one optimiser step through a path observed.
#[derive(Clone, Debug, Default)]
pub struct Observation {
    /// The loss the step's micro-batches returned, summed.
    pub loss: f64,
    /// Per-token log-probabilities of the supervised positions, in a fixed order.
    pub logprobs: Vec<f32>,
    /// Every trained tensor's gradient before the update, by name.
    pub grads: Vec<(String, Vec<f32>)>,
    /// Every trained tensor after the update, by name.
    pub weights: Vec<(String, Vec<f32>)>,
}

/// The largest difference of two observations, per field: the loss relative to its magnitude (at least one), the
/// log-probabilities absolute, each tensor in relative L2 (the largest over tensors).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Band {
    pub loss: f64,
    pub logprobs: f64,
    pub grads: f64,
    pub weights: f64,
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let diff: f64 = a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2)).sum::<f64>().sqrt();
    let norm: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum::<f64>().sqrt();
    if norm == 0.0 {
        diff
    } else {
        diff / norm
    }
}

fn tensors(a: &[(String, Vec<f32>)], b: &[(String, Vec<f32>)], what: &str) -> f64 {
    assert_eq!(a.iter().map(|t| &t.0).collect::<Vec<_>>(), b.iter().map(|t| &t.0).collect::<Vec<_>>(), "{what}: the same tensors");
    a.iter().zip(b).map(|((name, x), (_, y))| {
        assert_eq!(x.len(), y.len(), "{what}: {name} has one size");
        rel_l2(x, y)
    }).fold(0.0, f64::max)
}

impl Band {
    pub fn between(a: &Observation, b: &Observation) -> Band {
        assert_eq!(a.logprobs.len(), b.logprobs.len(), "the same supervised positions");
        Band {
            loss: (a.loss - b.loss).abs() / b.loss.abs().max(1.0),
            logprobs: a.logprobs.iter().zip(&b.logprobs).map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs()).fold(0.0, f64::max),
            grads: tensors(&a.grads, &b.grads, "gradients"),
            weights: tensors(&a.weights, &b.weights, "weights"),
        }
    }

    /// Each field at least `floor`.
    pub fn floored(self, floor: f64) -> Band {
        Band { loss: self.loss.max(floor), logprobs: self.logprobs.max(floor), grads: self.grads.max(floor), weights: self.weights.max(floor) }
    }

    fn within(&self, band: &Band) -> bool {
        self.loss <= band.loss && self.logprobs <= band.logprobs && self.grads <= band.grads && self.weights <= band.weights
    }
}

/// The parity verdict of a replacement: the old path's band against itself and the new path's difference from it.
#[derive(Clone, Copy, Debug)]
pub struct Verdict {
    /// The old path's run-to-run difference, before flooring.
    pub run_to_run: Band,
    /// The band the new path is held to.
    pub band: Band,
    /// The new path's difference from the old path's first run.
    pub new: Band,
}

/// Hold `new` to the band of `old` run twice (`old`, `old_again`), floored at `floor` (the binary32 agreement bound of
/// the step's longest reduction). Prints the three bands, so a test log carries the numbers; panics when a field of the
/// new path's difference exceeds the band.
pub fn assert_parity(what: &str, old: &Observation, old_again: &Observation, new: &Observation, floor: f64) -> Verdict {
    assert!(!old.grads.is_empty() && !old.weights.is_empty() && !old.logprobs.is_empty(), "{what}: the old path observed a step");
    assert!(old.grads.iter().any(|(_, g)| g.iter().any(|&x| x != 0.0)), "{what}: the step has a gradient to compare");
    let run_to_run = Band::between(old_again, old);
    let band = run_to_run.floored(floor);
    let diff = Band::between(new, old);
    eprintln!("step parity {what}: run-to-run {run_to_run:?}, band {band:?}, new {diff:?}");
    assert!(diff.within(&band), "{what}: the new path is off the old one by {diff:?}, past the band {band:?}");
    Verdict { run_to_run, band, new: diff }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_is_eight_pinned_records_with_supervised_answers() {
        let records = a2_records();
        assert_eq!(records.len(), 8);
        for r in &records {
            let (prompt, answer) = r.first_answer();
            assert!(!prompt.is_empty() && !answer.is_empty(), "{}: a prompt and an answer", r.source);
        }
        let mut rows: Vec<Vec<u32>> = records.iter().map(|r| r.ids.clone()).collect();
        let before = rows.clone();
        let n = compact_vocab(&mut rows);
        assert!(rows.iter().flatten().all(|&id| id < n));
        for (a, b) in rows.iter().zip(&before) {
            for i in 0..a.len() {
                for j in 0..a.len().min(i + 8) {
                    assert_eq!(a[i] == a[j], b[i] == b[j], "token identity is kept");
                }
            }
        }
    }

    #[test]
    fn a_difference_past_the_band_fails_and_one_inside_it_passes() {
        let obs = |g: f32| Observation { loss: 1.0, logprobs: vec![-1.0], grads: vec![("a".into(), vec![g, 1.0])], weights: vec![("a".into(), vec![1.0, 1.0])] };
        let v = assert_parity("inside", &obs(1.0), &obs(1.0), &obs(1.0 + 1e-7), 1e-6);
        assert_eq!(v.run_to_run, Band::default());
        let past = std::panic::catch_unwind(|| assert_parity("past", &obs(1.0), &obs(1.0), &obs(1.01), 1e-6));
        assert!(past.is_err());
    }
}
