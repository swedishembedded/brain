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

/// Knobs for [`finetune_lora`]/[`finetune_full`]. Field-for-field a subset of
/// [`model::FitOpts`] (see [`FinetuneOpts::to_fit_opts`]) plus the two
/// LoRA-specific knobs (`rank`/`alpha`) - kept as its own type rather than
/// `model::FitOpts` directly so this crate's CLI surface (`brain qwen3tts
/// finetune`) stays independent of the generic engine's full option set.
#[derive(Clone, Debug)]
pub struct FinetuneOpts {
    pub steps: u32,
    pub batch: u32,
    pub block: u32,
    pub lr: f32,
    /// Floor the cosine decay holds after `decay_iters` (`model::FitOpts::min_lr`).
    pub min_lr: f32,
    /// Steps of linear warmup before `lr` is reached (`model::FitOpts::warmup`).
    pub warmup: u32,
    /// Step by which the decay has reached `min_lr` (`model::FitOpts::decay_iters`).
    pub decay_iters: u32,
    pub weight_decay: f32,
    pub grad_clip: f32,
    /// Micro-batches averaged into one optimizer step (`model::FitOpts::grad_accum`).
    pub grad_accum: u32,
    /// Wall-clock checkpoint cadence in seconds, `0` disables periodic saves
    /// (only the final checkpoint is written) - `model::FitOpts::checkpoint_secs`.
    pub checkpoint_secs: u64,
    pub rank: u32,
    pub alpha: f32,
    pub seed: u64,
}

impl Default for FinetuneOpts {
    fn default() -> Self {
        let steps = 200;
        FinetuneOpts {
            steps,
            batch: 16,
            block: 16,
            lr: 1e-3,
            min_lr: 1e-4,
            warmup: steps / 10,
            decay_iters: steps,
            weight_decay: 0.0,
            grad_clip: 1.0,
            grad_accum: 1,
            checkpoint_secs: 600,
            rank: 8,
            alpha: 16.0,
            seed: 1337,
        }
    }
}

impl FinetuneOpts {
    /// Lift these opts into the generic engine's [`model::FitOpts`] -
    /// [`run_finetune`] hands the result straight to [`model::fit_with`].
    /// Every field not exposed on `FinetuneOpts` (masking, eval cadence) is a
    /// no-op default: this crate's dataset carries no mask/vocab metadata
    /// (see [`run_finetune`]'s doc) and prints no periodic eval line, matching
    /// this loop's behaviour before it was collapsed onto `fit_with`.
    fn to_fit_opts(&self) -> model::FitOpts {
        model::FitOpts {
            steps: self.steps,
            batch_size: self.batch,
            block_size: self.block,
            lr: self.lr,
            min_lr: self.min_lr,
            warmup: self.warmup,
            decay_iters: self.decay_iters,
            weight_decay: self.weight_decay,
            grad_clip: self.grad_clip,
            grad_accum: self.grad_accum,
            eval_interval: 0,
            eval_batches: 0,
            seed: self.seed,
            checkpoint_secs: self.checkpoint_secs,
            mask_before: None,
            mask_per_line: false,
            align_to_lines: false,
        }
    }
}

/// LoRA fine-tune a Talker decoder (`base` checkpoint) on a `text->codes` token
/// dataset in `dir` (`train.u32.bin`/`val.u32.bin`, e.g. from `data::gen_tts` -
/// no `meta.json`/mask needed, every position is scored). The pretrained
/// weights are frozen; only the attention LoRA
/// adapters (`*.lora_a`/`*.lora_b`) train. Writes the adapted checkpoint to `out`
/// and returns `(initial_loss, final_loss)`.
///
/// Native-only (reads `.bin` datasets, writes a checkpoint).
#[cfg(not(target_arch = "wasm32"))]
pub fn finetune_lora(base: &str, dir: &Path, out: &str, opts: &FinetuneOpts) -> std::io::Result<(f32, f32)> {
    use qwen3::{LoraCfg, QwenConfig};

    // Load the base config + weights, then re-key under a LoRA config so the
    // parameter list gains `*.lora_a`/`*.lora_b` (base stays frozen).
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
    run_finetune(cfg, init, dir, out, opts)
}

/// Full (non-LoRA) single-speaker SFT: every Talker weight trains, matching
/// Qwen's own documented single-speaker fine-tuning workflow (encode target
/// speech to codes, fine-tune the base model on them, serve the result
/// through the CustomVoice interface) rather than the lighter-weight adapter
/// path [`finetune_lora`] offers alongside it. AdamW moments offload to
/// system RAM for the duration of the call (`BRAIN_OFFLOAD_ADAM`, the same
/// convention `qwen3::finetune::Mode::FullOffload` uses) - full fine-tuning
/// triples the optimizer state over the frozen-base LoRA path, and this is
/// the one knob that keeps that affordable. Writes the fine-tuned checkpoint
/// to `out` and returns `(initial_loss, final_loss)`.
///
/// Native-only (reads `.bin` datasets, writes a checkpoint).
#[cfg(not(target_arch = "wasm32"))]
pub fn finetune_full(base: &str, dir: &Path, out: &str, opts: &FinetuneOpts) -> std::io::Result<(f32, f32)> {
    use qwen3::QwenConfig;

    let ckpt = checkpoint::load(base);
    let mut cfg = QwenConfig::from_json(&ckpt.header["config"]);
    cfg.block_size = opts.block;
    cfg.lora = None;
    let base_weights = ckpt.by_role("");
    // Matches finetune_lora's own pattern (and qwen3::finetune::Mode::FullOffload's):
    // a fresh init first, then overwritten by the checkpoint's real values, rather
    // than trusting the checkpoint alone to carry every key `Qwen::new` expects.
    let mut init = qwen3::init_weights(&cfg, opts.seed);
    for (k, v) in base_weights {
        init.insert(k, v);
    }

    let prev_off = std::env::var("BRAIN_OFFLOAD_ADAM").ok();
    std::env::set_var("BRAIN_OFFLOAD_ADAM", "1");
    let result = run_finetune(cfg, init, dir, out, opts);
    match prev_off {
        Some(v) => std::env::set_var("BRAIN_OFFLOAD_ADAM", v),
        None => std::env::remove_var("BRAIN_OFFLOAD_ADAM"),
    }
    result
}

/// The training loop shared by [`finetune_lora`]/[`finetune_full`]: the only
/// difference between the two modes is how `cfg`/`init` are built above (a
/// LoRA-extended config with the base frozen vs. the base config with every
/// tensor trainable) - the dataset/optimizer loop is identical either way, so
/// it is [`model::fit_with`] over the shared causal-LM [`model::causal_lm`]
/// objective rather than a fourth hand-rolled copy of it.
///
/// Deliberately does **not** go through [`model::load_dataset`]: this crate's
/// `text->codes` datasets (`data::gen_tts`) carry no `meta.json`/`.mask.bin` -
/// every position is scored, unmasked - so the [`data::loader::BatchConfig`]
/// is built by hand here with masking off, exactly as this loop already did
/// before the collapse onto `fit_with`.
///
/// Returns `model::fit_with`'s own `(initial_train, last_train)` - the
/// 5-micro-step train-split estimate and the final step's train loss, NOT
/// the previous `(initial_val_eval, min(final_val_eval, last_train))` this
/// loop used to compute by hand. Callers reading the returned pair (the CLI's
/// printed summary) were updated for this at the same time.
#[cfg(not(target_arch = "wasm32"))]
fn run_finetune(
    cfg: qwen3::QwenConfig,
    init: std::collections::HashMap<String, Vec<f32>>,
    dir: &Path,
    out: &str,
    opts: &FinetuneOpts,
) -> std::io::Result<(f32, f32)> {
    use data::loader::{BatchConfig, TokenDataset};
    use qwen3::Qwen;

    let model = Qwen::new(cfg, opts.batch, opts.block, &init);

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
    let obj = model::causal_lm::<Qwen>(train_ds, val_ds, bcfg, None);
    model::fit_with(model, obj, &opts.to_fit_opts(), Some(Path::new(out)))
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

    fn write_u32_bin(path: &std::path::Path, toks: &[u32]) {
        let bytes: Vec<u8> = toks.iter().flat_map(|t| t.to_le_bytes()).collect();
        std::fs::write(path, bytes).unwrap();
    }

    /// Build a tiny base checkpoint + a `text->codes`-shaped synthetic dataset
    /// (`train.u32.bin`/`val.u32.bin`, no `meta.json`) under a fresh tmp dir -
    /// the fixture every `run_finetune`-driving test below shares. `tag`
    /// disambiguates the tmp dir between tests running in the same process.
    fn setup(tag: &str) -> (std::path::PathBuf, String) {
        use qwen3::{Qwen, QwenConfig};

        let cfg = QwenConfig::tiny();
        let init = qwen3::init_weights(&cfg, 7);
        let base_model = Qwen::new(cfg.clone(), 1, cfg.block_size, &init);

        let dir = std::env::temp_dir().join(format!("qwen3tts-sft-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let base_path = dir.join("base.safetensors").to_str().unwrap().to_string();
        base_model.save(&base_path);

        // Enough tokens for a couple of batches at batch=2/block=4 (needs
        // >= batch*block+1), values kept inside the tiny 23-word vocab.
        let toks: Vec<u32> = (0..64).map(|i| (i * 7 + 3) % cfg.vocab).collect();
        write_u32_bin(&dir.join("train.u32.bin"), &toks);
        write_u32_bin(&dir.join("val.u32.bin"), &toks);

        (dir, base_path)
    }

    /// The whole point of having two fine-tune modes: `finetune_full` must
    /// actually move the base decoder weights (that's what "full" means),
    /// while `finetune_lora` must leave them bit-for-bit untouched (the base
    /// stays frozen; only the adapters train). A tiny synthetic checkpoint +
    /// dataset, no real Qwen3-TTS weights needed - this is a contract test on
    /// the two training modes, not a quality test on real speech.
    #[test]
    fn full_finetune_moves_base_weights_lora_does_not() {
        let (dir, base_path) = setup("full-vs-lora");
        let init = qwen3::init_weights(&qwen3::QwenConfig::tiny(), 7);

        let opts = FinetuneOpts { steps: 2, batch: 2, block: 4, lr: 1e-2, rank: 2, alpha: 4.0, seed: 11, ..Default::default() };
        let key = "blocks.0.attn.wq.weight";
        let original = init[key].clone();

        let full_out = dir.join("full.safetensors").to_str().unwrap().to_string();
        finetune_full(&base_path, &dir, &full_out, &opts).expect("full finetune");
        let full_w = checkpoint::load(&full_out).by_role("");
        let full_diff: f32 = full_w[key].iter().zip(&original).map(|(a, b)| (a - b).abs()).sum();
        assert!(full_diff > 1e-6, "full finetune must move base weights (diff={full_diff})");

        let lora_out = dir.join("lora.safetensors").to_str().unwrap().to_string();
        finetune_lora(&base_path, &dir, &lora_out, &opts).expect("lora finetune");
        let lora_w = checkpoint::load(&lora_out).by_role("");
        let lora_diff: f32 = lora_w[key].iter().zip(&original).map(|(a, b)| (a - b).abs()).sum();
        assert_eq!(lora_diff, 0.0, "LoRA finetune must leave base weights untouched (diff={lora_diff})");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The defect this collapse onto `fit_with` fixes (a): the hand-rolled
    /// loop applied `opts.lr` as a flat constant rate at every step, ignoring
    /// `warmup`/`min_lr`/`decay_iters` entirely. Two single-step runs, same
    /// seed/data/model, differing only in `warmup`: at step 0 a `warmup: 0`
    /// run trains at the full peak rate, a `warmup: 1000` run trains at
    /// `peak/1000` (`LrSchedule::at`). AdamW's first-step update magnitude is
    /// ~`lr` per parameter (bias-corrected `v_hat` ≈ `grad^2`, so
    /// `m_hat/(sqrt(v_hat)+eps)` ≈ `sign(grad)`), so the warmed-down run's
    /// adapter movement must come out roughly 1000x smaller. Before the
    /// collapse this held identically regardless of `warmup` - RED.
    #[test]
    fn lora_finetune_lr_follows_the_warmup_schedule_not_a_constant_rate() {
        let (dir, base_path) = setup("lr-schedule");

        let no_warmup = FinetuneOpts {
            steps: 1,
            batch: 2,
            block: 4,
            lr: 1e-1,
            min_lr: 1e-1,
            warmup: 0,
            decay_iters: 1,
            rank: 2,
            alpha: 4.0,
            seed: 11,
            ..Default::default()
        };
        let out_a = dir.join("no_warmup.safetensors").to_str().unwrap().to_string();
        finetune_lora(&base_path, &dir, &out_a, &no_warmup).expect("finetune (no warmup)");

        let long_warmup = FinetuneOpts { warmup: 1000, ..no_warmup };
        let out_b = dir.join("long_warmup.safetensors").to_str().unwrap().to_string();
        finetune_lora(&base_path, &dir, &out_b, &long_warmup).expect("finetune (warmup=1000)");

        let key = "blocks.0.attn.wq.weight.lora_b"; // zero-init - any nonzero value is training-induced movement
        let move_a: f32 = checkpoint::load(&out_a).by_role("")[key].iter().map(|v| v.abs()).sum();
        let move_b: f32 = checkpoint::load(&out_b).by_role("")[key].iter().map(|v| v.abs()).sum();

        assert!(move_a > 0.0, "the no-warmup run must move the adapter at all: {move_a}");
        assert!(
            move_b < move_a * 0.1,
            "a step-0 rate of peak/1000 (warmup=1000) must move the adapter far less than the \
             full peak rate (warmup=0); got no-warmup={move_a}, warmup=1000={move_b} - the LR \
             schedule is not being applied"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The defect this collapse fixes (b): the hand-rolled loop drew exactly
    /// one micro-batch per step no matter what `grad_accum` said. Two
    /// single-step runs, same seed/data/model, differing only in
    /// `grad_accum`: `fit_with` returns the step's average forward loss
    /// (`last_train`) over however many micro-batches it actually drew.
    /// `grad_accum: 1` scores one (fixed, seed-determined) batch; `grad_accum:
    /// 4` must score FOUR different batches drawn off the same rng stream and
    /// average their loss - a different number, almost certainly, from the
    /// one-batch loss. Before the collapse both returned the identical
    /// single-batch loss regardless of `grad_accum` - RED.
    #[test]
    fn lora_finetune_grad_accum_actually_accumulates_multiple_micro_batches() {
        let (dir, base_path) = setup("grad-accum");

        let single = FinetuneOpts {
            steps: 1,
            batch: 2,
            block: 4,
            lr: 1e-2,
            min_lr: 1e-2,
            warmup: 0,
            decay_iters: 1,
            grad_accum: 1,
            rank: 2,
            alpha: 4.0,
            seed: 23,
            ..Default::default()
        };
        let out_1 = dir.join("accum1.safetensors").to_str().unwrap().to_string();
        let (_, last_1) = finetune_lora(&base_path, &dir, &out_1, &single).expect("finetune (grad_accum=1)");

        let accumulated = FinetuneOpts { grad_accum: 4, ..single };
        let out_4 = dir.join("accum4.safetensors").to_str().unwrap().to_string();
        let (_, last_4) = finetune_lora(&base_path, &dir, &out_4, &accumulated).expect("finetune (grad_accum=4)");

        assert!(
            (last_1 - last_4).abs() > 1e-4,
            "grad_accum=4 must average the forward loss over 4 micro-batches drawn from the \
             SAME rng stream grad_accum=1 draws only its first batch from, not silently behave \
             like grad_accum=1 (accum=1 loss {last_1}, accum=4 loss {last_4} are suspiciously \
             identical) - grad_accum is not being read"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The defect this collapse fixes (c): the hand-rolled loop never saved a
    /// checkpoint until the run finished, no matter how long it took - a slow
    /// or interrupted run had nothing to resume from. A background thread
    /// polls `out`'s mtime while a long-enough (`checkpoint_secs: 1`,
    /// thousands of tiny steps) run is still in flight; it must observe the
    /// file written more than once - a periodic save plus the always-present
    /// final save - proving a save happened BEFORE the run completed. Before
    /// the collapse this always saw exactly one write (the final one) - RED.
    #[test]
    fn full_finetune_writes_a_periodic_checkpoint_before_the_run_completes() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::time::{Duration, SystemTime};

        let (dir, base_path) = setup("periodic-checkpoint");
        let out = dir.join("periodic.safetensors");
        let out_watch = out.clone();

        let writes = Arc::new(AtomicUsize::new(0));
        let writes2 = writes.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop2 = stop.clone();
        let watcher = std::thread::spawn(move || {
            let mut last: Option<SystemTime> = None;
            while !stop2.load(Ordering::Relaxed) {
                if let Ok(meta) = std::fs::metadata(&out_watch) {
                    if let Ok(mtime) = meta.modified() {
                        if last != Some(mtime) {
                            last = Some(mtime);
                            writes2.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });

        let opts = FinetuneOpts {
            steps: 20_000,
            batch: 2,
            block: 4,
            lr: 1e-2,
            checkpoint_secs: 1,
            rank: 2,
            alpha: 4.0,
            seed: 41,
            ..Default::default()
        };
        finetune_lora(&base_path, &dir, out.to_str().unwrap(), &opts).expect("finetune");

        stop.store(true, Ordering::Relaxed);
        watcher.join().unwrap();

        assert!(
            writes.load(Ordering::Relaxed) >= 2,
            "expected at least one periodic checkpoint save plus the final save before the run \
             completed (checkpoint_secs=1, a multi-second run); saw {} distinct write(s)",
            writes.load(Ordering::Relaxed)
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
