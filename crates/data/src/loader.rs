// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Token-dataset batching with optional masking and line alignment - a faithful
//! port of nanogpt's `DataLoader.get_batch` / `_apply_masking` /
//! `_precompute_line_starts`.
//!
//! Batches are flattened row-major `[batch_size * block_size]`:
//! - `x`: input token ids (`u32`).
//! - `y`: next-token targets (`i32`); masked positions are [`IGNORE`] (`-1`),
//!   the cross-entropy ignore index.

use crate::rng::Rng;

/// Target value for positions excluded from the loss.
pub const IGNORE: i32 = -1;

/// Batching / masking configuration.
#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub block_size: usize,
    /// Mask loss for tokens up to & including this token id, per the nanogpt
    /// calculator/reverser/wordcalc recipe (the `=` token).
    pub mask_before_token: Option<u32>,
    /// Reset masking at each newline (only meaningful with `mask_before_token`).
    pub mask_per_line: bool,
    /// Sample windows aligned to line starts (requires `newline_token`).
    pub align_to_lines: bool,
    /// Newline token id, needed for `mask_per_line` and `align_to_lines`.
    pub newline_token: Option<u32>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        BatchConfig {
            batch_size: 32,
            block_size: 64,
            mask_before_token: None,
            mask_per_line: false,
            align_to_lines: false,
            newline_token: None,
        }
    }
}

/// A loaded token split plus precomputed line starts for aligned sampling.
pub struct TokenDataset {
    data: Vec<u32>,
    line_starts: Option<Vec<usize>>,
    /// Optional per-token supervision mask (parallel to `data`): `mask[i] == true`
    /// means "token `i` is a trainable target". Used for chat / tool-call
    /// fine-tuning, where only the assistant/response span is supervised and the
    /// prompt is masked - token-level, unlike the char-boundary `mask_before_token`
    /// which cannot express a multi-token prompt prefix.
    mask: Option<Vec<bool>>,
    /// Optional per-token reward/advantage weight (parallel to `data`), for
    /// continuous/reward-driven training (`crates/rl`) - see
    /// [`TokenDataset::get_batch_weighted`]. `None` (the default) means every
    /// token implicitly weights `1.0`, matching `model::Batch::Lm`'s
    /// semantics on a weighted-loss-enabled model.
    weights: Option<Vec<f32>>,
}

impl TokenDataset {
    /// Wrap a token array; precomputes line starts when `align_to_lines` is set.
    pub fn new(data: Vec<u32>, cfg: &BatchConfig) -> Self {
        let line_starts = if cfg.align_to_lines {
            cfg.newline_token
                .map(|nl| Self::precompute_line_starts(&data, nl, cfg.block_size))
        } else {
            None
        };
        TokenDataset { data, line_starts, mask: None, weights: None }
    }

    /// Wrap a token array with an explicit per-token supervision mask (see
    /// [`TokenDataset::mask`]). `mask.len()` must equal `data.len()`.
    pub fn new_with_mask(data: Vec<u32>, mask: Vec<bool>, cfg: &BatchConfig) -> Self {
        assert_eq!(data.len(), mask.len(), "mask length must match data length");
        let mut d = Self::new(data, cfg);
        d.mask = Some(mask);
        d
    }

    /// Wrap a token array with an explicit per-token reward/advantage weight
    /// (see [`TokenDataset::weights`]). `weights.len()` must equal
    /// `data.len()`. Composable with [`TokenDataset::new_with_mask`]'s
    /// supervision mask via [`TokenDataset::with_mask`] - a token can be both
    /// unsupervised (IGNORE) and, were it supervised, carry a weight; the
    /// mask still wins (IGNORE positions never enter the loss regardless of
    /// weight).
    pub fn new_with_weights(data: Vec<u32>, weights: Vec<f32>, cfg: &BatchConfig) -> Self {
        assert_eq!(data.len(), weights.len(), "weights length must match data length");
        let mut d = Self::new(data, cfg);
        d.weights = Some(weights);
        d
    }

    /// Attach a supervision mask to a dataset already built with
    /// [`TokenDataset::new_with_weights`] (or vice versa via
    /// [`TokenDataset::with_weights`]) - the two are independent optional
    /// fields; either constructor alone only sets its own.
    pub fn with_mask(mut self, mask: Vec<bool>) -> Self {
        assert_eq!(self.data.len(), mask.len(), "mask length must match data length");
        self.mask = Some(mask);
        self
    }

    /// Attach a reward/advantage weight to a dataset already built with
    /// [`TokenDataset::new_with_mask`] - see [`TokenDataset::with_mask`].
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        assert_eq!(self.data.len(), weights.len(), "weights length must match data length");
        self.weights = Some(weights);
        self
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Line starts whose next newline fits within `block_size` (mirrors
    /// `_precompute_line_starts`).
    fn precompute_line_starts(data: &[u32], newline: u32, block_size: usize) -> Vec<usize> {
        let nl_pos: Vec<usize> = data
            .iter()
            .enumerate()
            .filter(|&(_, &t)| t == newline)
            .map(|(i, _)| i)
            .collect();
        let mut starts = Vec::new();
        // A start needs `block_size` input tokens AND one more for the shifted
        // target, so it is valid only when `start + block_size + 1 <= len`.
        let fits = |start: usize| start + block_size < data.len();

        if fits(0) && (nl_pos.first().is_some_and(|&p| p < block_size) || nl_pos.is_empty()) {
            starts.push(0);
        }
        for w in nl_pos.windows(2) {
            let line_start = w[0] + 1;
            let next_nl = w[1];
            if next_nl - line_start < block_size && fits(line_start) {
                starts.push(line_start);
            }
        }
        if let Some(&last) = nl_pos.last() {
            let line_start = last + 1;
            if fits(line_start) {
                starts.push(line_start);
            }
        }
        starts
    }

    /// Draw a `(x, y)` batch. `x[b*block + t]` is the input token, `y[..]` the
    /// next-token target (`IGNORE` where masked).
    pub fn get_batch(&self, cfg: &BatchConfig, rng: &mut Rng) -> (Vec<u32>, Vec<i32>) {
        let (x, y, _starts) = self.sample_windows(cfg, rng);
        (x, y)
    }

    /// [`TokenDataset::get_batch`], plus the per-position reward/advantage
    /// weight (`w[b*block + t]`, matching `x`/`y`'s layout) for `crates/rl`'s
    /// weighted training driver - `1.0` everywhere when this dataset carries
    /// no [`TokenDataset::weights`] (`new`/`new_with_mask`), matching
    /// `model::Batch::Lm`'s implicit-weight-1.0 semantics on a
    /// weighted-loss-enabled model, so an unweighted dataset run through this
    /// method reproduces `get_batch`'s gradient exactly.
    pub fn get_batch_weighted(&self, cfg: &BatchConfig, rng: &mut Rng) -> (Vec<u32>, Vec<i32>, Vec<f32>) {
        let (x, y, starts) = self.sample_windows(cfg, rng);
        let bl = cfg.block_size;
        let mut w = vec![1.0f32; x.len()];
        if let Some(weights) = &self.weights {
            for (b, &start) in starts.iter().enumerate() {
                for t in 0..bl {
                    w[b * bl + t] = weights[start + 1 + t];
                }
            }
        }
        (x, y, w)
    }

    /// The shared core of [`TokenDataset::get_batch`]/
    /// [`TokenDataset::get_batch_weighted`]: sample `batch_size` windows,
    /// apply both masking schemes, and additionally return each row's
    /// absolute start offset into `data` - needed only by the weighted path
    /// to gather the matching weight window, so `get_batch` itself just
    /// drops it.
    fn sample_windows(&self, cfg: &BatchConfig, rng: &mut Rng) -> (Vec<u32>, Vec<i32>, Vec<usize>) {
        let bs = cfg.batch_size;
        let bl = cfg.block_size;
        let mut x = vec![0u32; bs * bl];
        let mut y = vec![0i32; bs * bl];

        let mut starts = vec![0usize; bs];
        for b in 0..bs {
            let start = self.sample_start(cfg, rng);
            starts[b] = start;
            for t in 0..bl {
                x[b * bl + t] = self.data[start + t];
                y[b * bl + t] = self.data[start + 1 + t] as i32;
            }
        }

        if let Some(mask_tok) = cfg.mask_before_token {
            self.apply_masking(&mut y, cfg, mask_tok);
        }
        // Token-level supervision mask: target y[b,t] predicts data[start+1+t];
        // supervise it only where that target token is flagged trainable.
        if let Some(mask) = &self.mask {
            for b in 0..bs {
                let start = starts[b];
                for t in 0..bl {
                    if !mask[start + 1 + t] {
                        y[b * bl + t] = IGNORE;
                    }
                }
            }
        }
        (x, y, starts)
    }

    fn sample_start(&self, cfg: &BatchConfig, rng: &mut Rng) -> usize {
        match &self.line_starts {
            Some(ls) if !ls.is_empty() => {
                let idx = rng.gen_range_inclusive(0, ls.len() as i64 - 1) as usize;
                ls[idx]
            }
            _ => {
                // checked_sub, never bare `-`: a dataset shorter than one
                // block (+1 for the shifted target) used to underflow here
                // and panic with a bare subtract-overflow. Say what is
                // actually wrong instead.
                let hi = self.data.len().checked_sub(cfg.block_size + 1).unwrap_or_else(|| {
                    panic!(
                        "dataset has {} tokens but block_size {} needs at least {} - \
                         use a longer dataset or a smaller --block-size",
                        self.data.len(),
                        cfg.block_size,
                        cfg.block_size + 1
                    )
                });
                rng.gen_range_inclusive(0, hi as i64) as usize
            }
        }
    }

    /// Port of `_apply_masking`: per-line resets masking at newlines; global
    /// masks up to & including the first occurrence in each row.
    fn apply_masking(&self, y: &mut [i32], cfg: &BatchConfig, mask_tok: u32) {
        let bl = cfg.block_size;
        let mask_tok = mask_tok as i32;
        let nl = cfg.newline_token.map(|n| n as i32);
        let bs = y.len() / bl;

        if cfg.mask_per_line {
            for b in 0..bs {
                let row = &mut y[b * bl..(b + 1) * bl];
                let mut line_start = 0usize;
                for pos in 0..bl {
                    if row[pos] == mask_tok {
                        for v in row.iter_mut().take(pos + 1).skip(line_start) {
                            *v = IGNORE;
                        }
                    }
                    if let Some(nl) = nl {
                        if row[pos] == nl {
                            line_start = pos + 1;
                        }
                    }
                }
            }
        } else {
            for b in 0..bs {
                let row = &mut y[b * bl..(b + 1) * bl];
                if let Some(first) = row.iter().position(|&v| v == mask_tok) {
                    for v in row.iter_mut().take(first + 1) {
                        *v = IGNORE;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // "ab=cd\nef=gh\n" with ids: a0 b1 =2 c3 d4 \n5 ... build a tiny vocab.
    fn toks(s: &str, stoi: &dyn Fn(char) -> u32) -> Vec<u32> {
        s.chars().map(stoi).collect()
    }

    #[test]
    fn masks_up_to_equals_per_line() {
        // vocab: a=0 b=1 ==2 c=3 d=4 \n=5
        let stoi = |c: char| match c {
            'a' => 0,
            'b' => 1,
            '=' => 2,
            'c' => 3,
            'd' => 4,
            '\n' => 5,
            _ => unreachable!(),
        };
        let data = toks("ab=cd\nab=cd\nab=cd\n", &stoi);
        let cfg = BatchConfig {
            batch_size: 1,
            block_size: 6,
            mask_before_token: Some(2),
            mask_per_line: true,
            align_to_lines: false,
            newline_token: Some(5),
        };
        let ds = TokenDataset::new(data, &cfg);
        let mut rng = Rng::new(0);
        let (_x, y) = ds.get_batch(&cfg, &mut rng);
        // Wherever a '=' (2) appears in y, it and everything before it on the
        // line is IGNORE; tokens after '=' are kept.
        // Just assert at least one IGNORE and at least one kept target.
        assert!(y.contains(&IGNORE));
        assert!(y.iter().any(|&v| v >= 0));
    }

    #[test]
    fn aligned_sampling_starts_on_line_boundaries() {
        let stoi = |c: char| match c {
            'a' => 0,
            'b' => 1,
            '=' => 2,
            'c' => 3,
            'd' => 4,
            '\n' => 5,
            _ => unreachable!(),
        };
        // lines of length 6 ("ab=cd\n"); block_size 6.
        let data = toks("ab=cd\nab=cd\nab=cd\nab=cd\n", &stoi);
        let cfg = BatchConfig {
            batch_size: 4,
            block_size: 6,
            mask_before_token: None,
            mask_per_line: false,
            align_to_lines: true,
            newline_token: Some(5),
        };
        let ds = TokenDataset::new(data, &cfg);
        let mut rng = Rng::new(3);
        let (x, _y) = ds.get_batch(&cfg, &mut rng);
        // Each row should begin with 'a' (id 0) since starts are line-aligned.
        for b in 0..cfg.batch_size {
            assert_eq!(x[b * cfg.block_size], 0);
        }
    }

    #[test]
    fn get_batch_weighted_defaults_every_position_to_1_when_no_weights_attached() {
        let data: Vec<u32> = (0..40).collect();
        let cfg = BatchConfig { batch_size: 3, block_size: 5, ..Default::default() };
        let ds = TokenDataset::new(data, &cfg);
        let mut rng = Rng::new(1);
        let (x, y, w) = ds.get_batch_weighted(&cfg, &mut rng);
        assert_eq!(w, vec![1.0f32; x.len()]);
        assert_eq!(x.len(), y.len());
        assert_eq!(x.len(), w.len());
    }

    #[test]
    fn get_batch_weighted_gathers_the_window_matching_targets_not_inputs() {
        // weights[i] is deliberately `i` as f32 so the test can assert
        // exactly which absolute offsets got gathered, not just "some
        // window". `get_batch_weighted` must align weights to the TARGET
        // token y[t] (== data[start+1+t]), the same offset the supervision
        // mask in `sample_windows` uses - not the input token x[t].
        let data: Vec<u32> = (0..40).collect();
        let weights: Vec<f32> = (0..40).map(|i| i as f32).collect();
        let cfg = BatchConfig { batch_size: 2, block_size: 5, ..Default::default() };
        let ds = TokenDataset::new_with_weights(data, weights, &cfg);
        let mut rng = Rng::new(7);
        let (x, _y, w) = ds.get_batch_weighted(&cfg, &mut rng);
        for b in 0..cfg.batch_size {
            for t in 0..cfg.block_size {
                // x[t] == start+t, so the matching weight is at start+1+t == x[t]+1.
                assert_eq!(w[b * cfg.block_size + t], x[b * cfg.block_size + t] as f32 + 1.0);
            }
        }
    }

    #[test]
    fn with_weights_and_with_mask_compose_independently() {
        let data: Vec<u32> = (0..40).collect();
        let mask: Vec<bool> = (0..40).map(|i| i % 2 == 0).collect();
        let weights: Vec<f32> = vec![2.0; 40];
        let cfg = BatchConfig { batch_size: 2, block_size: 5, ..Default::default() };
        let ds = TokenDataset::new(data, &cfg).with_mask(mask).with_weights(weights);
        let mut rng = Rng::new(2);
        let (_x, y, w) = ds.get_batch_weighted(&cfg, &mut rng);
        assert!(w.iter().all(|&wi| wi == 2.0), "weights must come through regardless of mask");
        assert!(y.contains(&IGNORE), "the mask attached via with_mask must still apply");
    }
}
