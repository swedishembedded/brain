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
    /// Example boundaries `(start, end)` into `data`, `end` exclusive and
    /// pointing one past the example's separator. `Some` puts the dataset in
    /// one-example-per-row mode (see [`TokenDataset::new_examples`]), where a
    /// row is a single example rather than an arbitrary window of the stream.
    examples: Option<Examples>,
}

/// Example boundaries plus the token a short row is padded with - the
/// separator itself, so a padded row reads as "example, then nothing".
struct Examples {
    bounds: Vec<(usize, usize)>,
    pad: u32,
}

/// An example that cannot be a row on its own, because it is longer than the
/// row. Truncating it would silently drop the end of a training answer, so
/// [`TokenDataset::new_examples`] refuses instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExampleTooLong {
    pub index: usize,
    pub tokens: usize,
    pub block_size: usize,
    /// The buffers the rejected call took ownership of, handed back so a
    /// caller that has another way to train on this data does not have to
    /// re-read or re-encode it.
    returned: (Vec<u32>, Vec<bool>),
}

impl ExampleTooLong {
    /// The `(data, mask)` passed to the rejected [`TokenDataset::new_examples`].
    pub fn returned(self) -> (Vec<u32>, Vec<bool>) {
        self.returned
    }
}

impl std::fmt::Display for ExampleTooLong {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "example {} is {} tokens, longer than block_size {} - raise --block to at least {} \
             (truncating it would train on a cut-off answer)",
            self.index, self.tokens, self.block_size, self.tokens
        )
    }
}

impl std::error::Error for ExampleTooLong {}

impl TokenDataset {
    /// Wrap a token array; precomputes line starts when `align_to_lines` is set.
    pub fn new(data: Vec<u32>, cfg: &BatchConfig) -> Self {
        let line_starts = if cfg.align_to_lines {
            cfg.newline_token
                .map(|nl| Self::precompute_line_starts(&data, nl, cfg.block_size))
        } else {
            None
        };
        TokenDataset { data, line_starts, mask: None, weights: None, examples: None }
    }

    /// Wrap a token array with an explicit per-token supervision mask (see
    /// [`TokenDataset::mask`]). `mask.len()` must equal `data.len()`.
    pub fn new_with_mask(data: Vec<u32>, mask: Vec<bool>, cfg: &BatchConfig) -> Self {
        assert_eq!(data.len(), mask.len(), "mask length must match data length");
        let mut d = Self::new(data, cfg);
        d.mask = Some(mask);
        d
    }

    /// Wrap a token array whose examples are delimited by `separator`, so that
    /// **one row is one example**.
    ///
    /// The alternative - drawing an arbitrary `block_size` window from the
    /// concatenated stream - packs however many examples happen to fit into
    /// one row and lets every one of them attend to all the others. For short
    /// instruction-tuning examples that is not a rounding error: at 42 tokens
    /// an example and a 1024-token row, 24 question/answer pairs share a row,
    /// and a model can drive its loss down by copying a sibling's answer
    /// instead of learning the mapping. At serving time there is one question
    /// and nothing to copy, so the training regime is one that never occurs in
    /// use, and held-out loss measured that way scores copying rather than
    /// generalisation.
    ///
    /// Each row here holds exactly one example, left-aligned and padded with
    /// `separator`. Padding carries [`IGNORE`] targets, so it contributes no
    /// gradient; causal attention inside the row only ever reaches the
    /// example's own earlier tokens.
    ///
    /// Errors when an example does not fit in `cfg.block_size`, rather than
    /// truncating a training answer to fit.
    pub fn new_examples(
        data: Vec<u32>,
        mask: Vec<bool>,
        separator: u32,
        cfg: &BatchConfig,
    ) -> Result<Self, ExampleTooLong> {
        assert_eq!(data.len(), mask.len(), "mask length must match data length");
        let mut examples = Vec::new();
        let mut start = 0usize;
        for (i, &t) in data.iter().enumerate() {
            if t == separator {
                examples.push((start, i + 1));
                start = i + 1;
            }
        }
        // A trailing example the writer did not terminate is still an example.
        if start < data.len() {
            examples.push((start, data.len()));
        }
        if let Some((index, &(a, b))) = examples.iter().enumerate().find(|(_, &(a, b))| b - a > cfg.block_size) {
            return Err(ExampleTooLong {
                index,
                tokens: b - a,
                block_size: cfg.block_size,
                returned: (data, mask),
            });
        }
        Ok(TokenDataset {
            data,
            line_starts: None,
            mask: Some(mask),
            weights: None,
            examples: Some(Examples { bounds: examples, pad: separator }),
        })
    }

    /// How many examples this dataset holds, when built by
    /// [`TokenDataset::new_examples`]. `None` for a plain token stream, which
    /// has no example boundaries to count.
    pub fn example_count(&self) -> Option<usize> {
        self.examples.as_ref().map(|e| e.bounds.len())
    }

    /// The longest example in tokens, when built by
    /// [`TokenDataset::new_examples`] - what `block_size` actually has to
    /// cover, so a caller can size a row to the data instead of guessing.
    pub fn longest_example(&self) -> Option<usize> {
        self.examples.as_ref().and_then(|e| e.bounds.iter().map(|&(a, b)| b - a).max())
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
        if let Some(ex) = &self.examples {
            // One row, one example: `end` bounds both the tokens copied in and
            // the targets, so nothing from the next example is ever visible or
            // supervised. The rest of the row is padding with IGNORE targets.
            for b in 0..bs {
                let pick = rng.gen_range_inclusive(0, ex.bounds.len() as i64 - 1) as usize;
                let (a, e) = ex.bounds[pick];
                starts[b] = a;
                for t in 0..bl {
                    let target = a + 1 + t;
                    let supervised = target < e && self.mask.as_ref().is_none_or(|m| m[target]);
                    x[b * bl + t] = if a + t < e { self.data[a + t] } else { ex.pad };
                    y[b * bl + t] = if supervised { self.data[target] as i32 } else { IGNORE };
                }
            }
            return (x, y, starts);
        }
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

    /// Three 4-token examples behind a separator, in a row twice that long.
    const SEP: u32 = 9;

    fn three_examples() -> (Vec<u32>, Vec<bool>) {
        // [0 1 2 SEP][3 4 5 SEP][6 7 8 SEP]; the separator is never a target,
        // matching what `data::chat` writes.
        let data = vec![0, 1, 2, SEP, 3, 4, 5, SEP, 6, 7, 8, SEP];
        let mask = data.iter().map(|&t| t != SEP).collect();
        (data, mask)
    }

    /// THE spec: a row is one example. A row that runs on into the next one
    /// lets the model answer by copying a sibling that will not be there at
    /// serving time, and makes held-out loss score that copying.
    #[test]
    fn an_example_row_never_reaches_into_the_next_example() {
        let (data, mask) = three_examples();
        let cfg = BatchConfig { batch_size: 6, block_size: 8, ..Default::default() };
        let ds = TokenDataset::new_examples(data, mask, SEP, &cfg).expect("each example fits");
        let mut rng = Rng::new(1);
        let whole = [vec![0, 1, 2, SEP], vec![3, 4, 5, SEP], vec![6, 7, 8, SEP]];

        for _ in 0..40 {
            let (x, y) = ds.get_batch(&cfg, &mut rng);
            for b in 0..cfg.batch_size {
                let row = &x[b * 8..(b + 1) * 8];
                let end = row.iter().position(|&t| t == SEP).expect("the example's own separator") + 1;
                assert!(whole.iter().any(|e| e == &row[..end]), "row {row:?} is not one whole example");
                // Everything from the separator on is padding or another
                // example's business: it must carry no gradient.
                for t in (end - 1)..8 {
                    assert_eq!(y[b * 8 + t], IGNORE, "row {row:?} supervises position {t}");
                }
            }
        }
    }

    /// Sampling by example must still reach every example. The stream-window
    /// sampler it replaces silently dropped any example whose start left less
    /// than `block_size` behind it.
    #[test]
    fn every_example_can_be_drawn() {
        let (data, mask) = three_examples();
        let cfg = BatchConfig { batch_size: 4, block_size: 8, ..Default::default() };
        let ds = TokenDataset::new_examples(data, mask, SEP, &cfg).expect("each example fits");
        assert_eq!(ds.example_count(), Some(3));
        assert_eq!(ds.longest_example(), Some(4));
        let mut rng = Rng::new(5);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..40 {
            let (x, _) = ds.get_batch(&cfg, &mut rng);
            for b in 0..cfg.batch_size {
                seen.insert(x[b * 8]);
            }
        }
        assert_eq!(seen, [0, 3, 6].into_iter().collect(), "some example is unreachable");
    }

    /// Refused, not truncated: a row too short for an example would cut the
    /// end off a training answer and train on the stump.
    #[test]
    fn an_example_longer_than_the_row_is_refused() {
        let (data, mask) = three_examples();
        let cfg = BatchConfig { batch_size: 2, block_size: 3, ..Default::default() };
        let err = TokenDataset::new_examples(data.clone(), mask, SEP, &cfg).map(|_| ()).expect_err("must refuse");
        assert_eq!((err.index, err.tokens, err.block_size), (0, 4, 3));
        assert!(err.to_string().contains("--block"), "the message must say how to fix it: {err}");
        // The caller gets its buffers back rather than having to re-encode.
        assert_eq!(err.returned().0, data);
    }
}
