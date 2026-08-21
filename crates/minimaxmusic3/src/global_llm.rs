// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Global LLM: a real Qwen3-8B architecture (`hidden=4096, layers=36,
//! heads=32, kv_heads=8, head_dim=128, vocab=200000, rope_theta=1e6` -
//! confirmed against the checkpoint's own `language_model/config.json`,
//! NOT the smaller published Qwen3-8B's `vocab=151936` preset), reused
//! VERBATIM from `crates/qwen3` rather than reimplemented - see this
//! crate's own module doc. [`import`]'s own doc explains why
//! `language_model/`, specifically, and not the repository's OTHER
//! same-shaped language-model directory (`qwen_7B/qwen_7B/`, MiniMax's
//! own native, non-Qwen3, architecture). This module owns only two
//! things: streamed import (so the ~18 GB checkpoint never needs to be
//! resident in host RAM at once) and the audio-code-restricted training
//! objective this port adds on top of `crates/qwen3`'s already-
//! gradchecked forward/backward - not a second copy of Qwen3 itself.
//!
//! # Prompt / audio-code token contract
//!
//! The checkpoint's own special-token convention (confirmed against the
//! reference `diffusers` PR's own `MiniMaxMusic3TextEncoderStep`/
//! `MiniMaxMusic3SemanticGenerationStep` - even whitespace-level changes to
//! the assembled prompt change the generated audio, per that reference's
//! own comment): the conditional prompt is assembled as
//! `<|im_start|><|caption_start|>{caption}<|caption_end|><|lyrics_start|>
//! {lyrics}<|lyrics_end|><|im_end|><|audio_start|>`, then one token per
//! 25 Hz audio frame - a semantic RVQ code as vocab id
//! `AUDIO_CODE_OFFSET + code` (`code` in `[0, SEMANTIC_VOCAB_SIZE)`), fed
//! back into the SAME `embed_tokens`/`lm_head` any ordinary text token
//! uses (audio codes are plain extra vocab ids, not a separate embedding
//! space - `Qwen`'s own weights need no change to read or write them),
//! until `AUDIO_END_TOKEN_ID` or a frame cap. Classifier-free guidance is
//! NOT a `Qwen3`-level feature: the reference runs the SAME ordinary
//! `Qwen3ForCausalLM` forward on a 2-row `[conditional, unconditional]`
//! batch (the unconditional row replaces every prompt token except the
//! first and the two trailing structure tokens with `AUDIO_CFG_TOKEN_ID`)
//! and blends the two logit rows on the host - pure orchestration on top
//! of `crates/qwen3::Qwen`'s ordinary forward, not new model-level API.
//! That orchestration (prompt assembly text, the AR sampling loop, the
//! depth-decoder feedback) is `crate` M7 scope (pipeline glue); this
//! module owns only the constants the contract is built from, so both
//! this milestone's training objective and the next milestone's sampling
//! loop read them from one place.

use qwen3::{Qwen, QwenConfig};

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const CAPTION_START: &str = "<|caption_start|>";
pub const CAPTION_END: &str = "<|caption_end|>";
pub const LYRICS_START: &str = "<|lyrics_start|>";
pub const LYRICS_END: &str = "<|lyrics_end|>";
pub const AUDIO_START: &str = "<|audio_start|>";
pub const AUDIO_END_TOKEN_ID: u32 = 151670;
pub const AUDIO_CFG_TOKEN_ID: u32 = 151654;
pub const AUDIO_CODE_OFFSET: u32 = 151675;
pub const SEMANTIC_VOCAB_SIZE: u32 = 16384;
/// The reference inference recipe's own fixed sampling parameters (M7's
/// AR sampling loop uses these; recorded here alongside the token
/// contract they parameterize).
pub const AR_CFG_SCALE: f32 = 1.5;
pub const AR_CFG_TOP_K: usize = 50;
pub const AR_SAMPLING_TOP_K: usize = 50;

/// The vocab id one semantic RVQ code occupies in the Global LLM's own
/// `vocab=200000` space.
pub fn audio_code_token_id(code: u32) -> u32 {
    assert!(code < SEMANTIC_VOCAB_SIZE, "audio_code_token_id: code {code} out of range [0, {SEMANTIC_VOCAB_SIZE})");
    AUDIO_CODE_OFFSET + code
}

/// Streamed import of the real Global LLM checkpoint: `dir` is the
/// `language_model/` subfolder (`config.json` + a 4-shard
/// `model.safetensors.index.json` set) - a genuine `Qwen3ForCausalLM`/
/// `model_type: "qwen3"` re-export, standard fields throughout
/// (`attention_bias: false`, `rope_theta: 1e6`, no residual-scaling
/// extras). The checkpoint's OTHER language-model directory,
/// `qwen_7B/qwen_7B/`, is NOT this - its own `config.json` reads
/// `"architectures": ["AbabForCausalLM"]`, `"model_type": "mixtral"`,
/// with per-layer LayerNorm alpha/beta residual-scaling constants no
/// plain Qwen3 decoder layer has - MiniMax's native training-checkpoint
/// format, a materially different architecture despite matching
/// `hidden=4096, layers=36, heads=32, kv_heads=8, head_dim=128,
/// vocab=200000` on the surface. Only `language_model/` is safe to load
/// through `crates/qwen3::Qwen` verbatim; `qwen_7B/qwen_7B/`'s weights
/// would either fail `qwen3::import::hf_to_brain`'s name mapping outright
/// or, worse, load silently wrong (the alpha/beta scaling `qwen3`'s own
/// forward has no code path for). The tokenizer lives at a THIRD
/// location, `qwen_7B/qwen3-8B-tokenizer-music/` (not under
/// `language_model/`, which ships no tokenizer files of its own).
///
/// `checkpoint::weightio::WeightReader::open_hf_dir` mmaps only shard
/// HEADERS up front; `qwen3::import::hf_source` then resolves brain
/// parameter names against those headers with zero tensor bytes read;
/// `Qwen::new_shard_i8` pulls one tensor at a time straight to the
/// device (CPU-JIT-backend host buffers on this machine, since there is
/// no discrete GPU), quantizing to int8 (DP4A) as it goes and dropping
/// each tensor's transient f32 expansion before the next - peak host RAM
/// stays at "one tensor", never the whole ~18 GB bf16 checkpoint (which
/// would expand past this machine's ~21 GB usable RAM at fp32). Int8 is
/// not merely smaller-and-nice-to-have here: it is what makes an 8B
/// model resident on this machine's CPU backend possible at all - the
/// same load-bearing role this crate's plan recorded for it going in.
/// Inference-only (matches [`qwen3::Qwen::new_shard_i8`]'s own scope);
/// the audio-code training objective below runs at
/// [`qwen3::QwenConfig::tiny`] scale instead, where fp32 + a real
/// backward pass both fit comfortably.
pub fn import(dir: &str, b: u32, t: u32) -> Result<(QwenConfig, Qwen), String> {
    let config_path = std::path::Path::new(dir).join("config.json");
    let config_json = std::fs::read_to_string(&config_path).map_err(|e| format!("global_llm::import: reading {}: {e}", config_path.display()))?;
    let cfg = qwen3::import::config_from_hf(&config_json)?;
    let reader = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(dir)).map_err(|e| format!("global_llm::import: {e}"))?;
    let src = qwen3::import::hf_source(&reader, &cfg)?;
    let qwen = Qwen::new_shard_i8(cfg.clone(), b, t, &src, model::Shard::whole(cfg.n_layers as usize));
    Ok((cfg, qwen))
}

/// Build a `Batch::LmWeighted` triple (`tokens`, `targets`, `weights`)
/// training the Global LLM to predict ONLY the audio-code targets that
/// follow `prompt_ids` - the training objective this milestone adds:
/// ordinary next-token cross-entropy (`crates/qwen3`'s own, already
/// gradchecked), restricted by POSITION via `model::Batch::LmWeighted`'s
/// existing per-position gradient weight (`0.0` on every position whose
/// TARGET still falls inside the prompt, `1.0` once the target is the
/// first audio-code token and thereafter) rather than a new loss kernel
/// or a vocab-subset mask - the same "reuse the existing weighted-CE
/// seam" choice `model::Batch::LmWeighted`'s own doc anticipates for
/// exactly this kind of prompt-masked training.
///
/// `prompt_ids`/`audio_code_ids` are caller-assembled token id sequences
/// (this function does no prompt-text assembly or offset arithmetic
/// itself - see [`audio_code_token_id`] for the offset, and this crate's
/// M7 milestone for prompt-text assembly). Returns `(tokens, targets,
/// weights)`, each `prompt_ids.len() + audio_code_ids.len() - 1` long (a
/// `Batch::Lm`-style shifted pair over the whole concatenated sequence).
pub fn audio_code_batch(prompt_ids: &[u32], audio_code_ids: &[u32]) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut seq = prompt_ids.to_vec();
    seq.extend_from_slice(audio_code_ids);
    let n = seq.len();
    assert!(n >= 2, "audio_code_batch: prompt_ids + audio_code_ids must have at least 2 tokens total");
    let tokens = seq[..n - 1].to_vec();
    let targets = seq[1..].to_vec();
    let prompt_len = prompt_ids.len();
    // Position i's target is seq[i+1]; that target is an audio code iff
    // i+1 >= prompt_len (the first audio-code token sits at seq[prompt_len]).
    let weights: Vec<f32> = (0..n - 1).map(|i| if i + 1 >= prompt_len { 1.0 } else { 0.0 }).collect();
    (tokens, targets, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;
    use model::{Batch, Model};

    #[test]
    fn audio_code_token_id_offsets_into_the_real_vocab_range() {
        assert_eq!(audio_code_token_id(0), AUDIO_CODE_OFFSET);
        assert_eq!(audio_code_token_id(SEMANTIC_VOCAB_SIZE - 1), AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE - 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn audio_code_token_id_rejects_an_out_of_range_code() {
        audio_code_token_id(SEMANTIC_VOCAB_SIZE);
    }

    #[test]
    fn audio_code_batch_masks_out_every_prompt_position() {
        let prompt = vec![1u32, 2, 3, 4, 5];
        let audio = vec![100u32, 101, 102];
        let (tokens, targets, weights) = audio_code_batch(&prompt, &audio);
        assert_eq!(tokens, vec![1, 2, 3, 4, 5, 100, 101]);
        assert_eq!(targets, vec![2, 3, 4, 5, 100, 101, 102]);
        // targets[0..4] are still prompt tokens (2,3,4,5) -> weight 0;
        // targets[4..] are the 3 audio codes (100,101,102) -> weight 1.
        assert_eq!(weights, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    /// The wiring this milestone actually adds: `Batch::LmWeighted`
    /// masking restricted to audio-code target positions must be
    /// trainable end to end, not just structurally well-formed - plain
    /// AdamW at `QwenConfig::tiny()` scale (where a real backward pass is
    /// cheap) must collapse the loss on a single fixed batch. `crates/
    /// qwen3` itself already gradchecks `LmWeighted`'s own gradient
    /// (weight=1 reproduces `Batch::Lm`'s gradient exactly, weight=0
    /// contributes exactly zero); this test proves only that THIS
    /// module's own position-masking construction is correct, not a
    /// second gradcheck of `qwen3`'s backward.
    #[test]
    fn audio_code_ce_training_overfits_a_single_batch() {
        let cfg = QwenConfig::tiny();
        let init = qwen3::init_weights(&cfg, 41);
        let mut r = Lcg::new(42);
        let prompt: Vec<u32> = (0..6).map(|_| r.next_u32() % cfg.vocab).collect();
        let audio: Vec<u32> = (0..4).map(|_| r.next_u32() % cfg.vocab).collect();
        let (tokens, targets, weights) = audio_code_batch(&prompt, &audio);

        let mut qwen = Qwen::new(cfg.clone(), 1, tokens.len() as u32, &init);
        qwen.enable_weighted_loss();
        Model::set_batch(&qwen, Batch::LmWeighted { tokens: &tokens, targets: &targets, weights: &weights });

        let loss0 = Model::forward(&qwen);
        let mut loss = loss0;
        for step in 1..=300u32 {
            Model::zero_grads(&qwen);
            Model::set_batch(&qwen, Batch::LmWeighted { tokens: &tokens, targets: &targets, weights: &weights });
            loss = Model::forward(&qwen);
            Model::backward(&qwen);
            Model::adamw_step(&qwen, step, 5e-2, 0.0, Some(1.0), 1.0);
            Model::poll_wait(&qwen);
        }
        assert!(loss < loss0 * 0.1, "audio-code CE training did not collapse the loss: start={loss0} end={loss} (300 steps)");
    }
}
