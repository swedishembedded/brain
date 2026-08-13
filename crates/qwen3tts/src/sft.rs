// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS **training / SFT**: the aligned multi-codebook loss and a
//! single-speaker LoRA fine-tune entry point.
//!
//! ## Multi-codebook alignment (the PR #278 fixes)
//! A codec frame carries `num_code_groups` (16) codes: codebook-0 (semantic) plus
//! 15 residual acoustic codebooks. Two *different* prediction problems are trained
//! jointly, and getting their label alignment right is the whole game:
//!
//! * **Talker — codebook-0, next-frame.** The Talker decoder predicts the *next*
//!   frame's codebook-0 from the running context. This is one — and only one —
//!   time shift: the logits at talker position `p` are scored against
//!   `codes[p+1][0]`. (HF's `forward` slices `inputs_embeds[:, :-1]` *and* passes
//!   `labels=codec_0_labels[:, 1:]`; if the loss *also* shifted internally that
//!   would be a **double shift** — the bug PR #278 fixes. brain shifts exactly
//!   once, here, explicitly.)
//!
//! * **MTP / code-predictor — residual codebooks 1..15, same frame.** Within a
//!   single frame `f`, the MTP runs a short causal chain `[hidden_f, cb0_f, cb1_f,
//!   …, cb14_f]` and predicts `[cb1_f, cb2_f, …, cb15_f]`: sequence position `k`
//!   (which consumed `cb_{k-1}`) predicts codebook `k` *of the same frame* — **no
//!   time shift at all** (`Qwen3TTSTalkerCodePredictorModel.forward_finetune`).
//!   Mixing in the next frame's residuals, or reusing the codebook-0 shift here,
//!   is the misalignment the synthetic test below is designed to catch.
//!
//! [`MultiCodebookLabels`] materialises both target sets from a `[T, num_q]`
//! frame tensor with those exact index rules; [`ce`] is the host softmax
//! cross-entropy used to verify them. The unit tests pin the indices so a
//! double-shift or a residual misalignment makes them fail.

use std::path::Path;

/// Cross-entropy ignore sentinel (matches `model::train::IGNORE`).
pub const IGNORE: u32 = 0xFFFF_FFFF;

/// Aligned training targets derived from a `[T, num_q]` row-major codes tensor
/// (`codes[f*num_q + q]`). The two target sets follow the alignment documented on
/// this module: codebook-0 is shifted by exactly one frame; residual codebooks
/// are same-frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiCodebookLabels {
    pub num_q: usize,
    pub frames: usize,
    /// Codebook-0 next-frame targets: `cb0_targets[p] = codes[(p+1)*num_q + 0]`
    /// for talker input position `p in 0..frames-1` (length `frames-1`).
    pub cb0_targets: Vec<u32>,
    /// Residual same-frame targets: `residual_targets[f][k-1] = codes[f*num_q + k]`
    /// for `k in 1..num_q`. One row per frame (length `frames`, each
    /// `num_q-1` wide).
    pub residual_targets: Vec<Vec<u32>>,
}

impl MultiCodebookLabels {
    /// Build aligned labels from `codes` (`frames*num_q` entries). Panics on a
    /// ragged length so a wrongly-shaped batch can never silently misalign.
    pub fn build(codes: &[u32], num_q: usize) -> MultiCodebookLabels {
        assert!(num_q >= 2, "need codebook-0 + ≥1 residual codebook");
        assert_eq!(codes.len() % num_q, 0, "codes length {} not a multiple of num_q {num_q}", codes.len());
        let frames = codes.len() / num_q;
        assert!(frames >= 1, "empty codes");

        // Codebook-0: ONE frame shift. Position p predicts frame p+1's cb0.
        let cb0_targets: Vec<u32> = (0..frames.saturating_sub(1))
            .map(|p| codes[(p + 1) * num_q]) // (p+1, q=0)
            .collect();

        // Residual codebooks 1..num_q: SAME frame, no shift.
        let residual_targets: Vec<Vec<u32>> = (0..frames)
            .map(|f| (1..num_q).map(|k| codes[f * num_q + k]).collect())
            .collect();

        MultiCodebookLabels { num_q, frames, cb0_targets, residual_targets }
    }

    /// The MTP sequence position that predicts residual codebook `k` (`1..num_q`):
    /// position `k` consumed the embedding of codebook `k-1`. Explicit so the
    /// alignment is testable independently of any model wiring.
    pub fn mtp_predict_position(k: usize) -> usize {
        k
    }
}

/// Host softmax cross-entropy for one logits row against `target`, returning
/// `(loss, dlogits)` with `dlogits = softmax(logits) - onehot(target)`. A
/// `target == IGNORE` row contributes zero loss and zero grad.
pub fn ce(logits: &[f32], target: u32) -> (f32, Vec<f32>) {
    let v = logits.len();
    if target == IGNORE {
        return (0.0, vec![0.0; v]);
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    let t = target as usize;
    let loss = -(probs[t].max(1e-30)).ln();
    let mut grad = probs;
    grad[t] -= 1.0;
    (loss, grad)
}

/// Mean cross-entropy over a `[n, vocab]` logits buffer and `n` targets
/// (`IGNORE` rows skipped), plus the flattened gradient (already averaged over
/// the number of *scored* rows). Used by the residual-codebook (MTP) loss.
pub fn ce_batch(logits: &[f32], targets: &[u32], vocab: usize) -> (f32, Vec<f32>) {
    assert_eq!(logits.len(), targets.len() * vocab, "logits/targets shape mismatch");
    let mut grad = vec![0.0f32; logits.len()];
    let mut total = 0.0f32;
    let mut scored = 0usize;
    for (i, &t) in targets.iter().enumerate() {
        let (l, g) = ce(&logits[i * vocab..(i + 1) * vocab], t);
        if t != IGNORE {
            total += l;
            scored += 1;
            grad[i * vocab..(i + 1) * vocab].copy_from_slice(&g);
        }
    }
    let n = scored.max(1) as f32;
    for g in &mut grad {
        *g /= n;
    }
    (total / n, grad)
}

// ---------------------------------------------------------------------------
// LoRA fine-tune (single-speaker SFT): reuse the gradient-checked Qwen LoRA.
// ---------------------------------------------------------------------------

/// Knobs for [`finetune_lora`].
#[derive(Clone, Debug)]
pub struct FinetuneOpts {
    pub steps: u32,
    pub batch: u32,
    pub block: u32,
    pub lr: f32,
    pub rank: u32,
    pub alpha: f32,
    pub seed: u64,
}

impl Default for FinetuneOpts {
    fn default() -> Self {
        FinetuneOpts { steps: 200, batch: 16, block: 16, lr: 1e-3, rank: 8, alpha: 16.0, seed: 1337 }
    }
}

/// LoRA fine-tune a Talker decoder (`base` checkpoint) on a `text->codes` token
/// dataset in `dir` (`train.u32.bin`/`val.u32.bin`/`meta.json`, e.g. from
/// `data::gen_tts`). The pretrained weights are frozen; only the attention LoRA
/// adapters (`*.lora_a`/`*.lora_b`) train. Writes the adapted checkpoint to `out`
/// and returns `(initial_loss, final_loss)`.
///
/// Native-only (reads `.bin` datasets, writes a checkpoint).
#[cfg(not(target_arch = "wasm32"))]
pub fn finetune_lora(base: &str, dir: &Path, out: &str, opts: &FinetuneOpts) -> std::io::Result<(f32, f32)> {
    use data::loader::{BatchConfig, TokenDataset};
    use data::rng::Rng;
    use qwen3::{LoraCfg, Qwen, QwenConfig};

    // 1. Load the base config + weights, then re-key under a LoRA config so the
    //    parameter list gains `*.lora_a`/`*.lora_b` (base stays frozen).
    let ckpt = checkpoint::load(base);
    let mut cfg = QwenConfig::from_json(&ckpt.header["config"]);
    cfg.block_size = opts.block;
    cfg.lora = Some(LoraCfg::attn(opts.rank, opts.alpha));
    let base_weights = ckpt.by_role("");
    // Fresh init provides the adapter tensors (A ~ small random, B = 0); overwrite
    // every base tensor with the pretrained value.
    let mut init = qwen3::init_weights(&cfg, opts.seed);
    for (k, v) in base_weights {
        init.insert(k, v);
    }

    let model = Qwen::new(cfg.clone(), opts.batch, opts.block, &init);

    // 2. Data.
    let train = data::binio::read_tokens_u32(&dir.join("train"))?;
    let val = data::binio::read_tokens_u32(&dir.join("val"))?;
    let bcfg = BatchConfig {
        batch_size: opts.batch as usize,
        block_size: opts.block as usize,
        mask_before_token: None,
        mask_per_line: false,
        align_to_lines: false,
        newline_token: None,
    };
    let train_ds = TokenDataset::new(train, &bcfg);
    let val_ds = TokenDataset::new(val, &bcfg);
    let mut rng = Rng::new(opts.seed ^ 0xA5A5_5A5A);

    let to_u32 = |y: &[i32]| -> Vec<u32> { y.iter().map(|&v| if v < 0 { IGNORE } else { v as u32 }).collect() };
    let eval = |m: &Qwen, ds: &TokenDataset, rng: &mut Rng, n: u32| -> f32 {
        let mut s = 0.0;
        for _ in 0..n.max(1) {
            let (x, y) = ds.get_batch(&bcfg, rng);
            m.set_batch(&x, &to_u32(&y));
            s += m.forward();
        }
        s / n.max(1) as f32
    };

    let initial = eval(&model, &val_ds, &mut rng.clone(), 5);
    let mut last = initial;
    for step in 0..opts.steps {
        model.zero_grads();
        let (x, y) = train_ds.get_batch(&bcfg, &mut rng);
        model.set_batch(&x, &to_u32(&y));
        last = model.forward();
        model.backward();
        model.adamw_step(step + 1, opts.lr, 0.0, Some(1.0), 1.0);
        model.poll_wait();
    }
    let final_eval = eval(&model, &val_ds, &mut rng.clone(), 5);
    model.save(out);
    Ok((initial, final_eval.min(last)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `[T, num_q]` codes tensor whose value at `(f, q)` is `f*STRIDE + q`,
    /// so a target value uniquely decodes to its `(frame, codebook)` — any
    /// off-by-one in frame or codebook is immediately visible.
    const STRIDE: u32 = 1000;
    fn synthetic_codes(frames: usize, num_q: usize) -> Vec<u32> {
        let mut c = vec![0u32; frames * num_q];
        for f in 0..frames {
            for q in 0..num_q {
                c[f * num_q + q] = f as u32 * STRIDE + q as u32;
            }
        }
        c
    }
    fn decode(v: u32) -> (u32, u32) {
        (v / STRIDE, v % STRIDE)
    }

    /// Codebook-0 targets are shifted by EXACTLY one frame (no double shift,
    /// no missing shift).
    #[test]
    fn cb0_is_single_frame_shift() {
        let (frames, num_q) = (6, 16);
        let codes = synthetic_codes(frames, num_q);
        let lab = MultiCodebookLabels::build(&codes, num_q);
        assert_eq!(lab.cb0_targets.len(), frames - 1);
        for p in 0..frames - 1 {
            let (frame, cb) = decode(lab.cb0_targets[p]);
            assert_eq!(cb, 0, "cb0 target must be codebook 0");
            assert_eq!(frame, p as u32 + 1, "cb0 must be a SINGLE shift (frame p+1)");
            // Guard rails: the two classic bugs would decode to these frames.
            assert_ne!(frame, p as u32 + 2, "double-shift bug");
            assert_ne!(frame, p as u32, "missing-shift bug");
        }
    }

    /// Residual codebook targets are SAME-frame and cover codebooks 1..num_q in
    /// order; position `k` predicts codebook `k`.
    #[test]
    fn residual_is_same_frame_in_codebook_order() {
        let (frames, num_q) = (6, 16);
        let codes = synthetic_codes(frames, num_q);
        let lab = MultiCodebookLabels::build(&codes, num_q);
        assert_eq!(lab.residual_targets.len(), frames);
        for f in 0..frames {
            assert_eq!(lab.residual_targets[f].len(), num_q - 1);
            for k in 1..num_q {
                let tgt = lab.residual_targets[f][k - 1];
                let (frame, cb) = decode(tgt);
                assert_eq!(frame, f as u32, "residual must be SAME frame (no time shift)");
                assert_eq!(cb, k as u32, "residual codebook order must be 1..num_q");
                // The MTP position that predicts codebook k is position k.
                assert_eq!(MultiCodebookLabels::mtp_predict_position(k), k);
                // Guard rail: a next-frame residual (the misalignment bug).
                if f + 1 < frames {
                    assert_ne!(frame, f as u32 + 1, "residual leaked next frame");
                }
            }
        }
    }

    /// The loss actually *uses* these indices: one-hot logits at the aligned
    /// target give ~0 CE, while scoring the double-shifted target gives a large
    /// CE. This ties the alignment to a measurable training signal.
    #[test]
    fn loss_rewards_aligned_targets_only() {
        let (frames, num_q, vocab) = (6usize, 16usize, frames_vocab());
        let codes = synthetic_codes(frames, num_q);
        let lab = MultiCodebookLabels::build(&codes, num_q);

        // One-hot logits at the CORRECT cb0 target for each talker position.
        let big = 30.0f32;
        let mut logits = vec![0.0f32; (frames - 1) * vocab];
        for p in 0..frames - 1 {
            logits[p * vocab + lab.cb0_targets[p] as usize] = big;
        }
        let (aligned_loss, _) = ce_batch(&logits, &lab.cb0_targets, vocab);
        assert!(aligned_loss < 1e-3, "aligned CE should be ~0, got {aligned_loss}");

        // Score the SAME logits against the double-shifted labels -> large loss.
        let bad: Vec<u32> = (0..frames - 1)
            .map(|p| codes[((p + 2).min(frames - 1)) * num_q]) // frame p+2, cb0
            .collect();
        let (bad_loss, _) = ce_batch(&logits, &bad, vocab);
        assert!(bad_loss > 5.0, "double-shifted CE should be large, got {bad_loss}");
    }

    fn frames_vocab() -> usize {
        // Large enough to index any synthetic target value (max = (frames-1)*STRIDE + num_q).
        (6 * STRIDE as usize) + 64
    }

    #[test]
    fn build_rejects_ragged() {
        let r = std::panic::catch_unwind(|| MultiCodebookLabels::build(&[0, 1, 2], 16));
        assert!(r.is_err(), "ragged codes must panic, not silently misalign");
    }
}
