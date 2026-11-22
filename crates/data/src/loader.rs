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

use std::path::{Path, PathBuf};

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
    /// Example boundaries `(start, end)` into `data`, `end` exclusive. `Some`
    /// puts the dataset in one-example-per-row mode (see
    /// [`TokenDataset::new_examples`]), where a row is a single example rather
    /// than an arbitrary window of the stream.
    examples: Option<Examples>,
}

/// Example boundaries plus the token a short row is padded with.
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

/// The per-token loss-weight file of `split` in a prepared dataset directory: `<split>.weight.bin`, one `f32` per
/// token of `<split>.u32.bin` ([`crate::binio::write_f32_bin`]). Absent, every position weighs 1.0 and the split
/// trains the plain loss.
pub fn split_weights_path(dir: &Path, split: &str) -> PathBuf {
    dir.join(format!("{split}.weight.bin"))
}

/// Write (`Some`) or remove (`None`) the weight file of `split`, so a directory prepared again never keeps the
/// weights of an earlier preparation.
pub fn write_split_weights(dir: &Path, split: &str, weights: Option<&[f32]>) -> std::io::Result<()> {
    let path = split_weights_path(dir, split);
    match weights {
        Some(weights) => crate::binio::write_f32_bin(&path, weights),
        None => match std::fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        },
    }
}

/// The weights of `split` (see [`split_weights_path`]): `None` when the split has no weight file. A file that cannot
/// be read, holds a weight that is not a finite number, or does not hold one weight per token of the split's
/// `tokens` is refused, naming the file: training would otherwise weigh positions it was not given weights for.
pub fn read_split_weights(dir: &Path, split: &str, tokens: usize) -> std::io::Result<Option<Vec<f32>>> {
    let path = split_weights_path(dir, split);
    if !path.exists() {
        return Ok(None);
    }
    let refuse = |why: String| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}: {why}", path.display()));
    let weights = crate::binio::read_f32_bin(&path).map_err(|e| refuse(format!("unreadable loss weights: {e}")))?;
    if weights.len() != tokens {
        return Err(refuse(format!("{} loss weights for the {tokens} tokens of the {split} split; the file holds one f32 per token", weights.len())));
    }
    if let Some(at) = weights.iter().position(|w| !w.is_finite()) {
        return Err(refuse(format!("loss weight {at} is {}, not a finite number", weights[at])));
    }
    Ok(Some(weights))
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

    /// Wrap a token array whose examples start at `starts` (ascending offsets
    /// into `data`, the first 0, each running to the next or to the end), so
    /// that **one row is one example**.
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
    /// `pad`, any id the model's vocabulary holds. Padding carries [`IGNORE`]
    /// targets, so it contributes no gradient; causal attention inside the row
    /// only ever reaches the example's own earlier tokens.
    ///
    /// Errors when an example does not fit in `cfg.block_size`, rather than
    /// truncating a training answer to fit.
    ///
    /// # Panics
    /// If `starts` is not strictly ascending from 0 within `data`.
    pub fn new_examples(
        data: Vec<u32>,
        mask: Vec<bool>,
        starts: &[usize],
        pad: u32,
        cfg: &BatchConfig,
    ) -> Result<Self, ExampleTooLong> {
        assert_eq!(data.len(), mask.len(), "mask length must match data length");
        assert!(starts.first().is_none_or(|&s| s == 0), "the first example starts at 0");
        assert!(starts.windows(2).all(|w| w[0] < w[1]) && starts.last().is_none_or(|&s| s < data.len()), "example starts must ascend within the data");
        let examples: Vec<(usize, usize)> = starts.iter().enumerate().map(|(i, &a)| (a, starts.get(i + 1).copied().unwrap_or(data.len()))).collect();
        if let Some((index, &(a, b))) = examples.iter().enumerate().find(|(_, &(a, b))| b - a > cfg.block_size) {
            return Err(ExampleTooLong {
                index,
                tokens: b - a,
                block_size: cfg.block_size,
            });
        }
        Ok(TokenDataset {
            data,
            line_starts: None,
            mask: Some(mask),
            weights: None,
            examples: Some(Examples { bounds: examples, pad }),
        })
    }

    /// The example starts of a stream whose examples each END with
    /// `separator` (the separator belongs to the example it closes); a
    /// trailing unterminated run is an example too. For datasets written
    /// before examples were indexed out of band.
    pub fn separator_starts(data: &[u32], separator: u32) -> Vec<usize> {
        let mut starts = Vec::new();
        let mut start = 0usize;
        for (i, &t) in data.iter().enumerate() {
            if t == separator {
                starts.push(start);
                start = i + 1;
            }
        }
        if start < data.len() {
            starts.push(start);
        }
        starts
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

    /// The batch whose rows are the examples `first..first + batch_size`, in
    /// order, laid out exactly as [`TokenDataset::get_batch`] lays a drawn
    /// example out; a row past the last example is padding with [`IGNORE`]
    /// targets throughout. `None` for a plain token stream, which has no
    /// examples to walk. Walking a split this way, `batch_size` examples at a
    /// time, visits every example exactly once: the deterministic pass a
    /// held-out evaluation needs, where a drawn sample would score a
    /// different subset every time.
    pub fn example_rows(&self, cfg: &BatchConfig, first: usize) -> Option<(Vec<u32>, Vec<i32>)> {
        let picks: Vec<Option<usize>> = (first..first + cfg.batch_size).map(Some).collect();
        self.example_batch(cfg, &picks)
    }

    /// The record lengths of the examples `picks` names, in order: `0` for a
    /// row of padding (`None`) and for an example past the last. `None` for a
    /// plain token stream, whose rows carry no record boundary. This is what a
    /// row's padding is the difference from: [`Self::example_batch`] lays each
    /// example out in a `block_size` row, and its own length is the token count
    /// that row actually holds.
    pub fn example_lengths(&self, picks: &[Option<usize>]) -> Option<Vec<usize>> {
        let ex = self.examples.as_ref()?;
        Some(picks.iter().map(|pick| pick.and_then(|p| ex.bounds.get(p)).map_or(0, |&(a, e)| e - a)).collect())
    }

    /// The batch whose row `b` is example `picks[b]` - `None` is a row of
    /// padding with [`IGNORE`] targets throughout, as is any example past the
    /// last - laid out as [`TokenDataset::get_batch`] lays a drawn example
    /// out. `None` for a plain token stream, which has no examples.
    pub fn example_batch(&self, cfg: &BatchConfig, picks: &[Option<usize>]) -> Option<(Vec<u32>, Vec<i32>)> {
        let ex = self.examples.as_ref()?;
        let (bs, bl) = (cfg.batch_size, cfg.block_size);
        assert_eq!(picks.len(), bs, "one pick per row");
        let mut x = vec![ex.pad; bs * bl];
        let mut y = vec![IGNORE; bs * bl];
        for (b, pick) in picks.iter().enumerate() {
            let Some(&(a, e)) = pick.and_then(|p| ex.bounds.get(p)) else { continue };
            for t in 0..bl {
                let target = a + 1 + t;
                let supervised = target < e && self.mask.as_ref().is_none_or(|m| m[target]);
                if a + t < e {
                    x[b * bl + t] = self.data[a + t];
                }
                if supervised {
                    y[b * bl + t] = self.data[target] as i32;
                }
            }
        }
        Some((x, y))
    }

    /// The per-position loss weights of the batch [`Self::example_batch`] lays out for `picks`, aligned with its
    /// targets (`w[b*block + t]` is the weight of target `t + 1` of row `b`'s example, 1.0 past the example and on
    /// padding rows). `None` when the dataset carries no weights or is a plain token stream.
    pub fn example_weights(&self, cfg: &BatchConfig, picks: &[Option<usize>]) -> Option<Vec<f32>> {
        let (ex, weights) = (self.examples.as_ref()?, self.weights.as_ref()?);
        let bl = cfg.block_size;
        let mut w = vec![1.0f32; picks.len() * bl];
        for (b, pick) in picks.iter().enumerate() {
            let Some(&(a, e)) = pick.and_then(|p| ex.bounds.get(p)) else { continue };
            for t in 0..bl.min((e - a).saturating_sub(1)) {
                w[b * bl + t] = weights[a + 1 + t];
            }
        }
        Some(w)
    }

    /// The dataset without its per-token weights: every position weighs 1.0 again.
    pub fn without_weights(mut self) -> Self {
        self.weights = None;
        self
    }

    /// Whether the dataset carries per-token weights ([`Self::with_weights`]).
    pub fn has_weights(&self) -> bool {
        self.weights.is_some()
    }

    /// The length in tokens of each example (the document a packed row lays out whole), `None` for a plain token
    /// stream.
    pub fn example_len(&self, example: usize) -> Option<usize> {
        self.examples.as_ref()?.bounds.get(example).map(|&(a, e)| e - a)
    }

    /// The packed batch whose row `r` lays out the examples `rows[r]` one after another (several documents per row,
    /// [`crate::packing`]), each laid out as [`TokenDataset::example_batch`] lays out one: its tokens, its targets
    /// under the supervision mask, and IGNORE on its last token, so no target crosses into the next document. The rest
    /// of a row is padding with IGNORE targets. Refused for a plain token stream, an unknown example, a row whose
    /// examples overflow it, or a number of rows other than `cfg.batch_size`.
    pub fn packed_batch(&self, cfg: &BatchConfig, rows: &[Vec<usize>]) -> Result<crate::packing::PackedBatch, crate::packing::PackingError> {
        use crate::packing::{PackedBatch, PackingError};
        let ex = self.examples.as_ref().ok_or(PackingError::NoExamples)?;
        let (bs, bl) = (cfg.batch_size, cfg.block_size);
        if rows.len() != bs {
            return Err(PackingError::RowCount { rows: rows.len(), batch_size: bs });
        }
        let mut x = vec![ex.pad; bs * bl];
        let mut y = vec![IGNORE; bs * bl];
        let mut segments = vec![0u32; bs * bl];
        let (mut document, mut useful) = (0u32, 0u64);
        for (r, picks) in rows.iter().enumerate() {
            let bounds: Vec<(usize, usize)> = picks.iter().map(|&pick| ex.bounds.get(pick).copied().ok_or(PackingError::UnknownExample { row: r, example: pick, examples: ex.bounds.len() })).collect::<Result<_, _>>()?;
            let tokens: usize = bounds.iter().map(|&(a, e)| e - a).sum();
            if tokens > bl {
                return Err(PackingError::RowOverflow { row: r, tokens, row_len: bl });
            }
            let mut at = 0usize;
            for (a, e) in bounds {
                for t in 0..e - a {
                    let i = r * bl + at + t;
                    let target = a + 1 + t;
                    x[i] = self.data[a + t];
                    segments[i] = document;
                    if target < e && self.mask.as_ref().is_none_or(|m| m[target]) {
                        y[i] = self.data[target] as i32;
                    }
                }
                at += e - a;
                document += 1;
            }
            useful += tokens as u64;
            // The padding that ends the row is a document of its own.
            if at < bl {
                segments[r * bl + at..(r + 1) * bl].fill(document);
                document += 1;
            }
        }
        Ok(PackedBatch { tokens: x, targets: y, segments, rows: rows.to_vec(), useful_tokens: useful })
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

    /// [`TokenDataset::get_batch`], plus the example each row holds (`None` for every row of a plain token stream,
    /// whose rows are windows of no example).
    pub fn get_batch_examples(&self, cfg: &BatchConfig, rng: &mut Rng) -> (Vec<u32>, Vec<i32>, Vec<Option<usize>>) {
        let (x, y, picks, _) = self.get_batch_examples_weighted(cfg, rng);
        (x, y, picks)
    }

    /// [`TokenDataset::get_batch_examples`], plus each position's weight laid out as
    /// [`TokenDataset::get_batch_weighted`] lays it out when the dataset carries weights (`None` when it does not).
    /// Draws exactly what `get_batch_examples` draws from `rng`.
    #[allow(clippy::type_complexity)]
    pub fn get_batch_examples_weighted(&self, cfg: &BatchConfig, rng: &mut Rng) -> (Vec<u32>, Vec<i32>, Vec<Option<usize>>, Option<Vec<f32>>) {
        let (x, y, starts) = self.sample_windows(cfg, rng);
        let picks = match &self.examples {
            Some(ex) => starts.iter().map(|start| ex.bounds.binary_search_by_key(start, |&(a, _)| a).ok()).collect(),
            None => vec![None; cfg.batch_size],
        };
        let weights = self.weights.as_ref().map(|_| self.window_weights(cfg, &starts));
        (x, y, picks, weights)
    }

    /// The weight of every position of the windows starting at `starts`, aligned with their targets; 1.0 everywhere
    /// when the dataset carries no weights.
    fn window_weights(&self, cfg: &BatchConfig, starts: &[usize]) -> Vec<f32> {
        let bl = cfg.block_size;
        let mut w = vec![1.0f32; starts.len() * bl];
        if let Some(weights) = &self.weights {
            for (b, &start) in starts.iter().enumerate() {
                for t in 0..bl {
                    if let Some(&weight) = weights.get(start + 1 + t) {
                        w[b * bl + t] = weight;
                    }
                }
            }
        }
        w
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
        let w = self.window_weights(cfg, &starts);
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
    use crate::rng::random_seed;

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
        let mut rng = Rng::new(random_seed());
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
        let mut rng = Rng::new(random_seed());
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
        let mut rng = Rng::new(random_seed());
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
        let mut rng = Rng::new(random_seed());
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
        let mut rng = Rng::new(random_seed());
        let (_x, y, w) = ds.get_batch_weighted(&cfg, &mut rng);
        assert!(w.iter().all(|&wi| wi == 2.0), "weights must come through regardless of mask");
        assert!(y.contains(&IGNORE), "the mask attached via with_mask must still apply");
    }

    /// Three 4-token examples, indexed by their starts; rows pad with `PAD`.
    const PAD: u32 = 99;
    const STARTS: [usize; 3] = [0, 4, 8];

    fn three_examples() -> (Vec<u32>, Vec<bool>) {
        let data = vec![0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23];
        let mask = vec![true; data.len()];
        (data, mask)
    }

    /// THE spec: a row is one example. A row that runs on into the next one
    /// lets the model answer by copying a sibling that will not be there at
    /// serving time, and makes held-out loss score that copying.
    #[test]
    fn an_example_row_never_reaches_into_the_next_example() {
        let (data, mask) = three_examples();
        let cfg = BatchConfig { batch_size: 6, block_size: 8, ..Default::default() };
        let ds = TokenDataset::new_examples(data, mask, &STARTS, PAD, &cfg).expect("each example fits");
        let mut rng = Rng::new(random_seed());
        let whole = [vec![0, 1, 2, 3], vec![10, 11, 12, 13], vec![20, 21, 22, 23]];

        for _ in 0..40 {
            let (x, y) = ds.get_batch(&cfg, &mut rng);
            for b in 0..cfg.batch_size {
                let row = &x[b * 8..(b + 1) * 8];
                assert!(whole.iter().any(|e| e == &row[..4]), "row {row:?} is not one whole example");
                assert!(row[4..].iter().all(|&t| t == PAD), "row {row:?} runs past its example");
                // The example's last token predicts nothing, and padding is
                // no one's business: neither carries gradient.
                for t in 3..8 {
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
        let ds = TokenDataset::new_examples(data, mask, &STARTS, PAD, &cfg).expect("each example fits");
        assert_eq!(ds.example_count(), Some(3));
        assert_eq!(ds.longest_example(), Some(4));
        let mut rng = Rng::new(random_seed());
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..40 {
            let (x, _) = ds.get_batch(&cfg, &mut rng);
            for b in 0..cfg.batch_size {
                seen.insert(x[b * 8]);
            }
        }
        assert_eq!(seen, [0, 10, 20].into_iter().collect(), "some example is unreachable");
    }

    /// Walked in order, `batch_size` at a time, every example is a row
    /// exactly once, laid out as a drawn one is, and the rows past the last
    /// example supervise nothing - the deterministic pass an evaluation makes.
    #[test]
    fn examples_walked_in_order_are_each_one_row_once() {
        let (data, mut mask) = three_examples();
        // The second example's first target is not supervised.
        mask[5] = false;
        let cfg = BatchConfig { batch_size: 2, block_size: 6, ..Default::default() };
        let ds = TokenDataset::new_examples(data, mask, &STARTS, PAD, &cfg).expect("each example fits");
        let (x, y) = ds.example_rows(&cfg, 0).expect("an example-indexed split walks");
        assert_eq!(x, [0, 1, 2, 3, PAD, PAD, 10, 11, 12, 13, PAD, PAD]);
        assert_eq!(y, [1, 2, 3, IGNORE, IGNORE, IGNORE, IGNORE, 12, 13, IGNORE, IGNORE, IGNORE]);
        let (x, y) = ds.example_rows(&cfg, 2).expect("the last batch");
        assert_eq!(&x[..6], [20, 21, 22, 23, PAD, PAD]);
        assert!(x[6..].iter().all(|&t| t == PAD) && y[6..].iter().all(|&t| t == IGNORE), "past the last example is padding");
        assert_eq!(&y[..6], [21, 22, 23, IGNORE, IGNORE, IGNORE]);
        let stream = TokenDataset::new((0..40).collect(), &cfg);
        assert!(stream.example_rows(&cfg, 0).is_none(), "a token stream has no examples to walk");
    }

    /// Refused, not truncated: a row too short for an example would cut the
    /// end off a training answer and train on the stump.
    #[test]
    fn an_example_longer_than_the_row_is_refused() {
        let (data, mask) = three_examples();
        let cfg = BatchConfig { batch_size: 2, block_size: 3, ..Default::default() };
        let err = TokenDataset::new_examples(data, mask, &STARTS, PAD, &cfg).map(|_| ()).expect_err("must refuse");
        assert_eq!((err.index, err.tokens, err.block_size), (0, 4, 3));
        assert!(err.to_string().contains("--block"), "the message must say how to fix it: {err}");
    }

    /// A stream written with each example closed by a separator indexes to
    /// the same starts, the separator staying with the example it closes.
    #[test]
    fn separator_terminated_examples_index_to_their_starts() {
        assert_eq!(TokenDataset::separator_starts(&[5, 6, 9, 7, 9, 8], 9), vec![0, 3, 5]);
        assert_eq!(TokenDataset::separator_starts(&[5, 9], 9), vec![0]);
        assert!(TokenDataset::separator_starts(&[], 9).is_empty());
    }
}
