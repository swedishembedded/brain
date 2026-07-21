// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end weekly fine-tuning: turn a universe of raw OHLCV series into a
//! gated, promotable Kronos-decoder checkpoint. Ties the leak-safe windowing
//! ([`forecast::train_data`]) to the frozen tokenizer ([`KronosModel::tokenize`])
//! and the differentiable decoder + promotion gate ([`crate::train`]).
//!
//! The tokenizer is **frozen** (encode under no-grad, matching the reference
//! recipe); only the decoder is fine-tuned, and only promoted if it beats the base
//! on a held-out (embargoed) split.

use crate::generate::KronosModel;
use crate::train::{finetune, FinetuneOpts, FinetuneReport, TokenBatch};
use forecast::train_data::{self, Series, SplitConfig, Window, WindowRef};
use std::collections::HashMap;

/// Tokenize one leak-safe window into a next-token training example: 6-feature
/// bars (amount = volume·mean(OHLC)) → frozen-tokenizer `(s1,s2)` → shift by one so
/// position `i` predicts token `i+1`; the calendar is the input positions' stamps;
/// the dep sibling is teacher-forced with the ground-truth next-s1 (a valid,
/// simpler alternative to the exposure-bias sampling for the batch builder).
pub fn tokenize_window(model: &KronosModel, w: &Window) -> Option<TokenBatch> {
    let feat = model.feat();
    let ctxlen = w.ctx.len() / 5;
    if ctxlen < 2 {
        return None;
    }
    let mut bars = vec![0.0f32; ctxlen * feat];
    for r in 0..ctxlen {
        for c in 0..5 {
            bars[r * feat + c] = w.ctx[r * 5 + c];
        }
        if feat == 6 {
            let m = (w.ctx[r * 5] + w.ctx[r * 5 + 1] + w.ctx[r * 5 + 2] + w.ctx[r * 5 + 3]) * 0.25;
            bars[r * feat + 5] = w.ctx[r * 5 + 4] * m;
        }
    }
    let (s1, s2) = model.tokenize(&bars, ctxlen);
    let l = ctxlen - 1;
    let stamps = w.ctx_stamps(); // [ctxlen, 5]
    let cal: [Vec<u32>; 5] = std::array::from_fn(|c| (0..l).map(|i| stamps[i * 5 + c]).collect());
    Some(TokenBatch {
        s1: s1[..l].to_vec(),
        s2: s2[..l].to_vec(),
        cal,
        sampled_s1: s1[1..].to_vec(),
        s1_targets: s1[1..].to_vec(),
        s2_targets: s2[1..].to_vec(),
    })
}

/// Fine-tune the decoder over a whole universe: enumerate leak-safe windows,
/// embargo-split, tokenize with the frozen `model`, and run the gated
/// [`finetune`]. `base_init` is the base decoder's reference-named weights (e.g.
/// from `import::load_decoder`); on promotion the returned map is the fine-tuned
/// decoder, ready for `KronosTrain::save` / `checkpoint::save`.
#[allow(clippy::too_many_arguments)]
pub fn finetune_universe(
    model: &KronosModel,
    base_init: &HashMap<String, Vec<f32>>,
    series: &[Series],
    context: usize,
    horizon: usize,
    split: SplitConfig,
    opts: &FinetuneOpts,
) -> (FinetuneReport, Option<HashMap<String, Vec<f32>>>) {
    let windows = train_data::enumerate_windows(series, context, horizon);
    let sp = train_data::temporal_split(series, &windows, horizon, split);
    let tok = |ws: &[WindowRef]| -> Vec<TokenBatch> {
        ws.iter()
            .filter_map(|&wr| tokenize_window(model, &train_data::extract(series, wr, context, horizon)))
            .collect()
    };
    let train = tok(&sp.train);
    let val = tok(&sp.val);
    let cfg = model.decoder_config().clone();
    let t = (context - 1) as u32;
    finetune(cfg, t, base_init, &train, &val, opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{KronosConfig, KronosTokenizerConfig};
    use std::collections::HashMap;

    fn tiny_model() -> (KronosModel, HashMap<String, Vec<f32>>) {
        let tc = KronosTokenizerConfig::tiny();
        let dc = KronosConfig::tiny();
        let mut seed = 1u64;
        let mut rnd = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
                })
                .collect()
        };
        let tw: HashMap<String, Vec<f32>> = tc.param_list().into_iter().map(|(k, s)| (k, rnd(s.iter().product()))).collect();
        let dw: HashMap<String, Vec<f32>> = dc.param_list().into_iter().map(|(k, s)| (k, rnd(s.iter().product()))).collect();
        (KronosModel::from_weights(tc, &tw, dc, &dw).unwrap(), dw)
    }

    fn synth_series(ticker: &str, n: usize) -> Series {
        let dates: Vec<(i32, u32, u32)> = (0..n).map(|i| (2025, 1 + (i / 28) as u32, 1 + (i % 28) as u32)).collect();
        let ohlcv: Vec<[f32; 5]> = (0..n)
            .map(|i| {
                let x = 100.0 + (i as f32 * 0.1).sin() * 5.0;
                [x, x + 1.0, x - 1.0, x + 0.3, 1000.0 + i as f32]
            })
            .collect();
        Series { ticker: ticker.into(), dates, ohlcv }
    }

    #[test]
    fn tokenize_window_has_shifted_shapes() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (model, _) = tiny_model();
        let s = vec![synth_series("A", 40)];
        let w = train_data::extract(&s, WindowRef { series_idx: 0, origin: 20 }, 10, 3);
        let tb = tokenize_window(&model, &w).unwrap();
        assert_eq!(tb.s1.len(), 9); // context 10 -> 9 next-token positions
        assert_eq!(tb.s2_targets.len(), 9);
        assert_eq!(tb.cal[2].len(), 9);
        assert!(tb.s1.iter().all(|&x| x < model.decoder_config().s1_vocab() as u32));
    }

    #[test]
    fn finetune_universe_runs_end_to_end_and_gates() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let (model, base) = tiny_model();
        let series: Vec<Series> = (0..3).map(|i| synth_series(&format!("S{i}"), 120)).collect();
        let split = SplitConfig { train_frac: 0.6, val_frac: 0.25, embargo: 5 };
        let opts = FinetuneOpts { epochs: 2, lr: 1e-3, wd: 0.0, clip: 3.0, lora: None };
        let (rep, w) = finetune_universe(&model, &base, &series, 10, 3, split, &opts);
        // plumbing must run and produce a well-formed decision.
        assert!(rep.base_val.is_finite() && rep.ft_val.is_finite());
        assert!(rep.steps > 0, "no training steps ran (no train windows tokenized)");
        assert_eq!(rep.promoted, w.is_some());
        eprintln!("universe finetune: base {:.3} ft {:.3} promoted {} steps {}", rep.base_val, rep.ft_val, rep.promoted, rep.steps);
    }
}
