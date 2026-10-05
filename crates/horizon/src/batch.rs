// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Encoded subjects flattened into the arrays one forward pass reads.
//!
//! Row `b * max_tokens` is subject `b`'s summary token; its inputs follow and
//! the rest of its rows are padding (masked out of attention, out of every
//! loss). Weights are normalised here, so both loss kernels' host sums are the
//! weighted MEAN: the event loss over subjects, the value loss over the
//! values that were hidden for it.

use data::rng::Rng;

use crate::config::{HorizonConfig, BIN_MASK, BIN_PRESENT};
use crate::encode::Encoded;
use crate::vocab::{CLS, PAD};

/// One batch, host side.
#[derive(Clone, Debug, Default)]
pub struct HostBatch {
    /// Subjects in the batch (`<= b`; the remaining rows are padding).
    pub n_subjects: usize,
    /// `[B*N]` token ids.
    pub token_ids: Vec<u32>,
    /// `[B*N]` 1 for a real token, 0 for padding.
    pub keep: Vec<u32>,
    /// `[B*N, value_table]` value-bin weights.
    pub value_bins: Vec<f32>,
    /// `[B*N, time_table]` time-ago bin weights.
    pub time_bins: Vec<f32>,
    /// `[B]` the summary token's row.
    pub summary_rows: Vec<u32>,
    /// `[B*N]` value targets (normal scores).
    pub value_target: Vec<f32>,
    /// `[B*N]` 0 not scored, 1 exact, 2 below, 3 above.
    pub value_state: Vec<u32>,
    /// `[B*N]` normalised value-loss weight (includes the configured value weight).
    pub value_weight: Vec<f32>,
    /// `[B*P]` the subject each (subject, piece) row belongs to.
    pub piece_subject: Vec<u32>,
    /// `[B*P, time_features]` per-piece time features.
    pub time_features: Vec<f32>,
    /// `[B*P, K]` 1 where the event happened.
    pub event: Vec<f32>,
    /// `[B*P, K]` time at risk.
    pub exposure: Vec<f32>,
    /// `[B]` normalised subject weights (sum to 1 over real subjects).
    pub subject_weight: Vec<f32>,
}

/// Per-(subject, piece) time features, shared by training and prediction:
/// age at the piece midpoint (centred and scaled, and its square), calendar
/// time at the midpoint, and a one-hot of the piece.
pub fn piece_features(cfg: &HorizonConfig, entry: f64, calendar: f64) -> Vec<f32> {
    let p = cfg.pieces() as usize;
    let mut f = vec![0.0f32; p * cfg.time_features() as usize];
    for (i, k) in cfg.knots.windows(2).enumerate() {
        let mid = 0.5 * (k[0] + k[1]) as f64;
        let age = ((entry + mid - 60.0) / 20.0) as f32;
        let row = &mut f[i * cfg.time_features() as usize..][..cfg.time_features() as usize];
        row[0] = age;
        row[1] = age * age;
        row[2] = ((calendar + mid - 2010.0) / 10.0) as f32;
        row[3 + i] = 1.0;
    }
    f
}

/// Assemble a batch of exactly `b` subject slots. `mask_rate` hides that
/// fraction of each subject's numeric values (at least one when it has any)
/// for the masked-value objective; `0` hides none, and then no value is
/// scored. `rng` decides which, so a seed reproduces the batch.
pub fn assemble(
    cfg: &HorizonConfig,
    subjects: &[&Encoded],
    b: usize,
    mask_rate: f64,
    rng: &mut Rng,
) -> HostBatch {
    assert!(
        subjects.len() <= b,
        "batch of {} subjects into {b} slots",
        subjects.len()
    );
    let (n, vt, tt) = (
        cfg.max_tokens as usize,
        cfg.value_table() as usize,
        cfg.time_table() as usize,
    );
    let (p, k, nf) = (
        cfg.pieces() as usize,
        cfg.n_codes as usize,
        cfg.time_features() as usize,
    );
    let mut hb = HostBatch {
        n_subjects: subjects.len(),
        token_ids: vec![PAD; b * n],
        keep: vec![0; b * n],
        value_bins: vec![0.0; b * n * vt],
        time_bins: vec![0.0; b * n * tt],
        summary_rows: (0..b).map(|i| (i * n) as u32).collect(),
        value_target: vec![0.0; b * n],
        value_state: vec![0; b * n],
        value_weight: vec![0.0; b * n],
        piece_subject: (0..b * p).map(|r| (r / p) as u32).collect(),
        time_features: vec![0.0; b * p * nf],
        event: vec![0.0; b * p * k],
        exposure: vec![0.0; b * p * k],
        subject_weight: vec![0.0; b],
    };
    let wsum: f32 = subjects.iter().map(|s| s.weight).sum();
    let mut masked_weight = 0.0f32;
    for (i, s) in subjects.iter().enumerate() {
        let row0 = i * n;
        hb.token_ids[row0] = CLS;
        hb.keep[row0] = 1;
        // The summary token carries the "present" bin like every value-less
        // token: an input of beta alone has near-zero variance, which the
        // first LayerNorm then amplifies into an ill-conditioned row.
        hb.value_bins[row0 * vt + (cfg.value_bins + BIN_PRESENT) as usize] = 1.0;
        let numeric: Vec<usize> = s
            .tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| t.target.is_some())
            .map(|(j, _)| j)
            .collect();
        let hide = if mask_rate > 0.0 && !numeric.is_empty() {
            ((numeric.len() as f64 * mask_rate).round() as usize).max(1)
        } else {
            0
        };
        let mut chosen = numeric;
        for j in 0..hide {
            // Partial Fisher-Yates: the first `hide` entries become the sample.
            let r = j + (rng.next_u64() as usize) % (chosen.len() - j);
            chosen.swap(j, r);
        }
        let hidden = &chosen[..hide];
        for (j, t) in s.tokens.iter().enumerate() {
            let row = row0 + 1 + j;
            hb.token_ids[row] = t.id;
            hb.keep[row] = 1;
            if hidden.contains(&j) {
                hb.value_bins[row * vt + (cfg.value_bins + BIN_MASK) as usize] = 1.0;
                let (y, state) = t.target.expect("only numeric tokens are hidden");
                hb.value_target[row] = y;
                hb.value_state[row] = state as u32;
                hb.value_weight[row] = s.weight; // normalised below
                masked_weight += s.weight;
            } else {
                for &(bin, w) in &t.value_bins {
                    hb.value_bins[row * vt + bin as usize] = w;
                }
            }
            for &(bin, w) in &t.time_bins {
                hb.time_bins[row * tt + bin as usize] = w;
            }
        }
        hb.subject_weight[i] = if wsum > 0.0 { s.weight / wsum } else { 0.0 };
        let feats = piece_features(cfg, s.entry, s.calendar_at_entry);
        hb.time_features[i * p * nf..][..p * nf].copy_from_slice(&feats);
        for (code, o) in s.outcomes.iter().enumerate() {
            for (piece, &x) in o.exposure.iter().enumerate() {
                hb.exposure[(i * p + piece) * k + code] = x;
            }
            if let Some(piece) = o.event_piece {
                hb.event[(i * p + piece as usize) * k + code] = 1.0;
            }
        }
    }
    if masked_weight > 0.0 {
        let scale = cfg.value_weight / masked_weight;
        hb.value_weight.iter_mut().for_each(|w| *w *= scale);
    }
    hb
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::encode;
    use crate::timeline::Subject;
    use crate::vocab::{FitOptions, Vocab};

    fn fixture() -> (Vec<Encoded>, HorizonConfig) {
        let lines = [
            r#"{"subject_id":"a","weight":1,"source":"s","entry":50,"calendar_at_entry":2000,
               "observations":[{"t":50,"var":"x","value":1},{"t":50,"var":"y","value":2},{"t":50,"var":"z","value":3}],
               "events":[{"t":52,"code":"death"}],"at_risk":[{"code":"*","from":50,"to":52}]}"#,
            r#"{"subject_id":"b","weight":3,"source":"s","entry":40,"calendar_at_entry":2010,
               "observations":[{"t":40,"var":"x","value":5}],"at_risk":[{"code":"*","from":40,"to":45}]}"#,
        ];
        let subjects: Vec<Subject> = lines
            .iter()
            .map(|l| Subject::from_json_line(l).unwrap())
            .collect();
        let codes = vec!["death".to_string()];
        let v = Vocab::fit(
            &subjects,
            &codes,
            &codes,
            &FitOptions {
                knots: 5,
                min_count: 1,
            },
        )
        .unwrap();
        let cfg = HorizonConfig::tiny(v.len(), 1);
        (subjects.iter().map(|s| encode(s, &v, &cfg)).collect(), cfg)
    }

    #[test]
    fn layout_padding_and_weights() {
        let (enc, cfg) = fixture();
        let refs: Vec<&Encoded> = enc.iter().collect();
        let hb = assemble(&cfg, &refs, 3, 0.0, &mut Rng::new(1));
        let n = cfg.max_tokens as usize;
        assert_eq!(hb.summary_rows, vec![0, n as u32, 2 * n as u32]);
        assert_eq!(hb.keep[..5], [1, 1, 1, 1, 0]);
        assert_eq!(hb.keep[2 * n], 0, "an empty slot has no summary token");
        let present = (cfg.value_bins + BIN_PRESENT) as usize;
        assert_eq!(
            hb.value_bins[present], 1.0,
            "the summary token is present, not empty"
        );
        assert_eq!(hb.subject_weight, vec![0.25, 0.75, 0.0]);
        assert!(
            hb.value_state.iter().all(|&s| s == 0),
            "nothing scored without masking"
        );
        // a: died at +2, piece (1, 2.5]; exposure [1, 1, 0].
        let p = cfg.pieces() as usize;
        assert_eq!(hb.exposure[..p], [1.0, 1.0, 0.0]);
        assert_eq!(hb.event[..p], [0.0, 1.0, 0.0]);
        // b: censored at +5, beyond the last knot (4).
        assert_eq!(hb.exposure[p..2 * p], [1.0, 1.5, 1.5]);
    }

    #[test]
    fn masking_hides_values_and_normalises_their_weight() {
        let (enc, cfg) = fixture();
        let refs: Vec<&Encoded> = enc.iter().collect();
        let hb = assemble(&cfg, &refs, 2, 0.34, &mut Rng::new(7));
        let vt = cfg.value_table() as usize;
        let mask_bin = (cfg.value_bins + BIN_MASK) as usize;
        let hidden: Vec<usize> = (0..hb.value_state.len())
            .filter(|&r| hb.value_state[r] != 0)
            .collect();
        assert_eq!(hidden.len(), 2, "one of a's three values, b's only value");
        for &r in &hidden {
            assert_eq!(hb.value_bins[r * vt + mask_bin], 1.0);
            let others: f32 = (0..vt)
                .filter(|&c| c != mask_bin)
                .map(|c| hb.value_bins[r * vt + c])
                .sum();
            assert_eq!(others, 0.0, "a hidden value leaves no trace of itself");
        }
        let total: f32 = hb.value_weight.iter().sum();
        assert!((total - cfg.value_weight).abs() < 1e-6);
        let again = assemble(&cfg, &refs, 2, 0.34, &mut Rng::new(7));
        assert_eq!(
            again.value_state, hb.value_state,
            "the seed reproduces the mask"
        );
    }
}
