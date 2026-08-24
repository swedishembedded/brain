// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pipeline glue: the autoregressive semantic+depth-code generation loop
//! that ties `global_llm` (two CFG-branch `qwen3::Qwen` instances) and
//! `depth_decoder` together into the per-frame hidden states the
//! flow-matching DiT conditions on. Chunk windowing, DiT denoising, and
//! vocoder stitching are `crate::denoise`/`crate::stitch`-scoped (this
//! module owns only the AR half); see this crate's own module doc for the
//! full five-component chain.
//!
//! # CFG runs through the depth decoder too, not just the Global LLM
//!
//! The reference's `_generate_depth_codes` takes BOTH the LM's conditional
//! AND unconditional hidden states (not just the conditional one) and
//! builds two SEPARATE depth-decoder sequences, one per branch - the
//! residual-code logits are CFG-blended the exact same way the semantic
//! code's are, and the SAMPLED code (one value, not a per-branch pair) is
//! fed back into BOTH branches' growing sequences. This crate's depth
//! decoder operates on one sequence at a time (no batch dimension), so
//! "batched over 2 rows" here just means driving two independent
//! `depth_decoder::KvCache`s, one per branch, in lockstep - no change to
//! `depth_decoder` itself.
//!
//! # The depth decoder is KV-cached per frame
//!
//! The reference's `_generate_depth_codes` calls the whole depth decoder on
//! the entire growing sequence at every depth step, so a frame costs
//! sequence lengths 2,3,..,`num_codebooks` per CFG branch. Since the
//! architecture is a plain causal transformer with no RoPE and no cross-row
//! state beyond attention, each position's contribution is fixed once it is
//! computed: `depth_decoder::step` appends one position to a per-frame
//! `KvCache` and returns the identical hidden state (bit-identical, gated by
//! `depth_decoder`'s own test), turning 35 position-forwards per branch per
//! frame into 8.
//!
//! # Two `Qwen` instances, not one batched one
//!
//! `crates/qwen3::Qwen`'s incremental KV-cache decode path (`step`/
//! `step_embed`/`prefill`) is fundamentally single-sequence: `b` is only
//! real in the batched *training* forward, and the decode-path buffers
//! (`kcache`/`vcache`) are sized `t·kv_dim` with no `b` factor at all. So
//! the conditional and unconditional branches are two independent `Qwen`
//! instances (own KV cache, own `dec_pos`), stepped in lockstep and
//! combined on the host - matching the reference's own `[conditional,
//! unconditional]` 2-row batch semantics exactly, just realized as two
//! `b=1` instances instead of one `b=2` one (a capability this crate does
//! not have and a real one-off addition to `crates/qwen3` was judged not
//! worth making for a single caller - see `qwen3::Qwen::embed_row`'s own
//! doc for the one addition that WAS worth making, a genuinely reusable
//! single-row lookup).

use data::rng::Lcg;
use model::hostmath::matvec_par;
use qwen3::model::PrefillInput;
use qwen3::Qwen;

use crate::config::DepthDecoderConfig;
use crate::depth_decoder::{self, DepthDecoderWeights};
use crate::global_llm::{self, AR_CFG_SCALE, AR_CFG_TOP_K, AR_SAMPLING_TOP_K, AUDIO_CODE_OFFSET, AUDIO_END_TOKEN_ID, SEMANTIC_VOCAB_SIZE};

/// `true` for a Global-LLM vocab id the AR sampling loop may ever emit: a
/// semantic RVQ code (`AUDIO_CODE_OFFSET..AUDIO_CODE_OFFSET+SEMANTIC_VOCAB_SIZE`)
/// or the end-of-audio token - every other id (ordinary text tokens,
/// `AUDIO_CFG_TOKEN_ID`, unused vocab) is illegal once generation has
/// passed `<|audio_start|>`.
fn is_legal_audio_id(id: u32) -> bool {
    (AUDIO_CODE_OFFSET..AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE).contains(&id) || id == AUDIO_END_TOKEN_ID
}

/// `hidden @ headᵀ`, masked to `-inf` on every vocab id [`is_legal_audio_id`]
/// rejects - the reference's own `logits.masked_fill(vocab_mask, -inf)`,
/// applied once per branch before any CFG blending.
fn masked_logits(hidden: &[f32], head: &[f32], vocab: usize, d_model: usize) -> Vec<f32> {
    let mut logits = matvec_par(head, hidden, vocab, d_model);
    for (i, l) in logits.iter_mut().enumerate() {
        if !is_legal_audio_id(i as u32) {
            *l = f32::NEG_INFINITY;
        }
    }
    logits
}

/// `unconditional + (conditional - unconditional) * scale`, element-wise -
/// the reference's own CFG combine, used identically for the semantic-code
/// logits and the depth decoder's residual-code logits.
fn cfg_blend(conditional: &[f32], unconditional: &[f32], scale: f32) -> Vec<f32> {
    unconditional.iter().zip(conditional).map(|(&u, &c)| u + (c - u) * scale).collect()
}

/// The `k`-th largest FINITE value in `values` (1-indexed: `k=1` is the
/// max), or `-inf` if fewer than `k` finite values exist - `torch.topk`'s
/// own `.values[..., -1]` (the smallest of the top-`k`), used as a
/// keep-if-at-least-this threshold.
fn kth_largest(values: &[f32], k: usize) -> f32 {
    let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() || k == 0 {
        return f32::NEG_INFINITY;
    }
    finite.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    finite[k.min(finite.len()) - 1]
}

/// The reference's own `_sample_top_k`: restrict `logits` to its own
/// top-`k` (any lower value or non-finite entry excluded), softmax the
/// rest, and draw one index from that distribution via `rng`.
fn sample_top_k(logits: &[f32], top_k: usize, rng: &mut Lcg) -> u32 {
    let threshold = kth_largest(logits, top_k);
    let mut probs: Vec<f32> = logits.iter().map(|&v| if v.is_finite() && v >= threshold { v } else { f32::NEG_INFINITY }).collect();
    let max = probs.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        *p = if p.is_finite() { (*p - max).exp() } else { 0.0 };
        sum += *p;
    }
    if sum > 1e-12 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
    let draw = rng.unit();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if draw < cum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Autoregressively sample the residual codes `c1..c{num_codebooks-1}` for
/// one frame, CFG-blended through the depth decoder the same way the
/// semantic code was through the Global LLM - see this module's own doc.
/// `hidden_cond`/`hidden_uncond` are the Global LLM's two branches' hidden
/// state for the position that just produced `semantic_code`.
/// `embed_code_row` looks up ONE Global-LLM vocab embedding (either
/// branch's `Qwen::embed_row` - both give the identical row, same
/// weights) without touching either instance's own decode state. Returns
/// `(codes, depth_hidden)`: `codes[0]` is `semantic_code` and `codes[1..]`
/// the `num_codebooks-1` residual codes; `depth_hidden` is the
/// CONDITIONAL branch's own per-step hidden states concatenated,
/// `[(num_codebooks-1) * hidden_size]` - the reference's own `hidden[:1]`
/// slice (only the conditional branch's hidden state feeds `frame_hiddens`).
fn generate_depth_codes(
    dd_w: &DepthDecoderWeights,
    dd_cfg: &DepthDecoderConfig,
    hidden_cond: &[f32],
    hidden_uncond: &[f32],
    semantic_code: u32,
    embed_code_row: impl Fn(u32) -> Vec<f32>,
    rng: &mut Lcg,
) -> (Vec<u32>, Vec<f32>) {
    let d = dd_cfg.hidden_size as usize;
    let num_codebooks = dd_cfg.num_codebooks as usize;
    let audio_vocab = dd_cfg.audio_vocab_size as usize;

    // One KV cache per CFG branch, fresh for this frame. The two branches
    // differ only in their FIRST row (the LM hidden state); every later row
    // is the same sampled code's projection fed to both. See
    // `depth_decoder::step` for why the cached path is bit-identical to
    // re-running `depth_decoder::forward` over the whole growing sequence -
    // which is what the reference recipe does, at ~4.4x the arithmetic.
    let mut cache_cond = depth_decoder::KvCache::new(dd_cfg);
    let mut cache_uncond = depth_decoder::KvCache::new(dd_cfg);
    depth_decoder::step(dd_w, dd_cfg, &mut cache_cond, &depth_decoder::projection(dd_w, dd_cfg, hidden_cond));
    depth_decoder::step(dd_w, dd_cfg, &mut cache_uncond, &depth_decoder::projection(dd_w, dd_cfg, hidden_uncond));

    let code_embed = embed_code_row(global_llm::audio_code_token_id(semantic_code));
    let mut row = depth_decoder::projection(dd_w, dd_cfg, &code_embed);

    let mut codes = vec![semantic_code];
    let mut depth_hidden = Vec::with_capacity((num_codebooks - 1) * d);

    for index in 1..num_codebooks {
        let h_cond = depth_decoder::step(dd_w, dd_cfg, &mut cache_cond, &row);
        let h_uncond = depth_decoder::step(dd_w, dd_cfg, &mut cache_uncond, &row);
        depth_hidden.extend_from_slice(&h_cond);

        let logits_cond = depth_decoder::audio_head(dd_w, dd_cfg, index - 1, &h_cond);
        let logits_uncond = depth_decoder::audio_head(dd_w, dd_cfg, index - 1, &h_uncond);
        let guided = cfg_blend(&logits_cond, &logits_uncond, AR_CFG_SCALE);
        let code = sample_top_k(&guided, AR_SAMPLING_TOP_K, rng);
        codes.push(code);

        if index < num_codebooks - 1 {
            let embed = depth_decoder::audio_embedding_row(dd_w, dd_cfg, code as usize + (index - 1) * audio_vocab);
            row = depth_decoder::projection(dd_w, dd_cfg, &embed);
        }
    }
    (codes, depth_hidden)
}

/// The reference's own `_embed_audio_frame`: sum the semantic code's
/// Global-LLM embedding with the `num_codebooks-1` residual codes'
/// depth-decoder embeddings, scaled by `num_codebooks^-0.5`. `codes` is
/// shared across both CFG branches (the reference samples ONE code and
/// broadcasts it via `.repeat(2)`), so this embedding is too - the same
/// vector is fed back into both `Qwen::step_embed` calls.
fn embed_audio_frame(dd_w: &DepthDecoderWeights, dd_cfg: &DepthDecoderConfig, codes: &[u32], embed_code_row: impl Fn(u32) -> Vec<f32>) -> Vec<f32> {
    let audio_vocab = dd_cfg.audio_vocab_size as usize;
    let mut embeds = embed_code_row(global_llm::audio_code_token_id(codes[0]));
    for (k, &code) in codes[1..].iter().enumerate() {
        let row = depth_decoder::audio_embedding_row(dd_w, dd_cfg, code as usize + k * audio_vocab);
        for (e, r) in embeds.iter_mut().zip(&row) {
            *e += r;
        }
    }
    let scale = (dd_cfg.num_codebooks as f32).powf(-0.5);
    for e in embeds.iter_mut() {
        *e *= scale;
    }
    embeds
}

/// The full AR generation loop: frame by frame, sample a CFG-guided
/// semantic code from the Global LLM, then the depth decoder's residual
/// codes, until `AUDIO_END_TOKEN_ID` or `max_frames` frames have been
/// emitted. Returns the concatenated per-frame hidden states,
/// `[frames_emitted, num_codebooks * hidden_size]` row-major (the
/// condition encoder's own expected input shape) - `frames_emitted <=
/// max_frames`, and is `0` only if generation ended immediately (the
/// reference treats that as an error condition; this function leaves that
/// judgment to its caller).
///
/// `lm_cond`/`lm_uncond` must already be constructed at the SAME
/// `t` (decode capacity) covering `conditional_ids.len() + max_frames + 1`
/// positions, with a fresh (never-stepped) KV cache. `head` is
/// `lm_cond.read_weight(lm_cond.cfg.head_weight())` (read ONCE by the
/// caller and reused across every frame - a `[vocab, d_model]` table,
/// multiple GB at real dims, far too large to re-read per frame).
#[allow(clippy::too_many_arguments)]
pub fn generate_frames(
    lm_cond: &Qwen,
    lm_uncond: &Qwen,
    dd_w: &DepthDecoderWeights,
    dd_cfg: &DepthDecoderConfig,
    head: &[f32],
    vocab: usize,
    d_model: usize,
    conditional_ids: &[u32],
    unconditional_ids: &[u32],
    max_frames: usize,
    seed: u64,
    progress: crate::ProgressSink<'_>,
) -> Vec<f32> {
    assert_eq!(conditional_ids.len(), unconditional_ids.len(), "generate_frames: conditional/unconditional prompt length mismatch");
    let mut rng = Lcg::new(seed);

    let prefill_cond: Vec<PrefillInput> = conditional_ids.iter().map(|&t| PrefillInput::Token(t)).collect();
    let prefill_uncond: Vec<PrefillInput> = unconditional_ids.iter().map(|&t| PrefillInput::Token(t)).collect();
    let mut hidden_cond = lm_cond.prefill(&prefill_cond);
    let mut hidden_uncond = lm_uncond.prefill(&prefill_uncond);

    let mut frame_hiddens: Vec<f32> = Vec::new();
    let mut frames = 0usize;

    for frame_index in 0..=max_frames {
        let logits_cond = masked_logits(&hidden_cond, head, vocab, d_model);
        let logits_uncond = masked_logits(&hidden_uncond, head, vocab, d_model);
        let mut guided = cfg_blend(&logits_cond, &logits_uncond, AR_CFG_SCALE);
        // Restrict the guided distribution to the CONDITIONAL branch's own
        // top-k legal candidates before its own top-k re-sample (the
        // reference's two-stage restriction: `threshold = topk(conditional,
        // top_k)`, then `_sample_top_k`'s own top-k on what's left).
        let cond_threshold = kth_largest(&logits_cond, AR_CFG_TOP_K);
        for (i, g) in guided.iter_mut().enumerate() {
            if logits_cond[i] < cond_threshold {
                *g = f32::NEG_INFINITY;
            }
        }
        let sampled = sample_top_k(&guided, AR_SAMPLING_TOP_K, &mut rng);
        if sampled == AUDIO_END_TOKEN_ID {
            break;
        }
        let semantic_code = sampled - AUDIO_CODE_OFFSET;

        let embed_code_row = |id: u32| lm_cond.embed_row(id);
        let (codes, depth_hidden) = generate_depth_codes(dd_w, dd_cfg, &hidden_cond, &hidden_uncond, semantic_code, embed_code_row, &mut rng);

        if frame_index > 0 {
            frame_hiddens.extend_from_slice(&hidden_cond);
            frame_hiddens.extend_from_slice(&depth_hidden);
            frames += 1;
            progress(frames as u32, max_frames as u32, "ar");
            if frames >= max_frames {
                break;
            }
        }

        let feedback = embed_audio_frame(dd_w, dd_cfg, &codes, embed_code_row);
        hidden_cond = lm_cond.step_embed(&feedback);
        hidden_uncond = lm_uncond.step_embed(&feedback);
    }
    frame_hiddens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DepthDecoderConfig;
    use crate::depth_decoder::random_weights as random_dd_weights;
    use qwen3::QwenConfig;

    #[test]
    fn is_legal_audio_id_covers_exactly_the_semantic_range_and_the_end_token() {
        assert!(is_legal_audio_id(AUDIO_CODE_OFFSET));
        assert!(is_legal_audio_id(AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE - 1));
        assert!(!is_legal_audio_id(AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE));
        assert!(!is_legal_audio_id(AUDIO_CODE_OFFSET - 1));
        assert!(is_legal_audio_id(AUDIO_END_TOKEN_ID));
        assert!(!is_legal_audio_id(0));
    }

    #[test]
    fn kth_largest_matches_a_sorted_reference() {
        let v = [3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut sorted = v.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for k in 1..=v.len() {
            assert_eq!(kth_largest(&v, k), sorted[k - 1], "k={k}");
        }
        assert_eq!(kth_largest(&v, v.len() + 5), *sorted.last().unwrap(), "k beyond length clamps to the smallest");
        assert_eq!(kth_largest(&[], 1), f32::NEG_INFINITY);
    }

    #[test]
    fn cfg_blend_at_scale_one_is_the_conditional_branch() {
        let cond = [1.0f32, -2.0, 3.0];
        let uncond = [0.5f32, 0.5, 0.5];
        assert_eq!(cfg_blend(&cond, &uncond, 1.0), cond);
        assert_eq!(cfg_blend(&cond, &uncond, 0.0), uncond);
    }

    /// `sample_top_k` must never return an index whose logit was masked to
    /// `-inf` or excluded by the top-k threshold, and (statistically, over
    /// many draws) should never pick the one deliberately low outlier
    /// included only to make sure it's excludable.
    #[test]
    fn sample_top_k_only_returns_legal_high_probability_indices() {
        let mut logits = vec![f32::NEG_INFINITY; 20];
        logits[3] = 5.0;
        logits[7] = 4.0;
        logits[11] = -100.0; // legal (finite) but should never survive top_k=2
        let mut rng = Lcg::new(1);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let picked = sample_top_k(&logits, 2, &mut rng);
            assert!(picked == 3 || picked == 7, "picked {picked} outside the top-2 candidates");
            seen.insert(picked);
        }
        assert_eq!(seen.len(), 2, "expected both top-2 candidates to be sampled at least once over 200 draws");
    }

    /// `generate_depth_codes`/`embed_audio_frame` wiring, at
    /// `DepthDecoderConfig::tiny()` scale with a synthetic (not a real
    /// Global LLM) `embed_code_row` closure - proves the shapes and the
    /// CFG-blend/feedback plumbing are correct without needing a real
    /// `Qwen` instance.
    #[test]
    fn generate_depth_codes_and_embed_audio_frame_produce_the_expected_shapes() {
        let cfg = DepthDecoderConfig::tiny();
        let w = random_dd_weights(&cfg, 1);
        let d = cfg.hidden_size as usize;
        let mut r = Lcg::new(2);
        let hidden_cond = r.vec_scaled(d, 0.3);
        let hidden_uncond = r.vec_scaled(d, 0.3);
        let semantic_code = 1u32;

        let embed_code_row = |id: u32| {
            // A deterministic stand-in for `Qwen::embed_row`: any function
            // of the (real, out-of-tiny-range) Global-LLM vocab id works,
            // since this test only checks the depth-decoder side of the
            // wiring, not real embedding values.
            let mut rr = Lcg::new(u64::from(id));
            rr.vec_scaled(d, 0.1)
        };

        let (codes, depth_hidden) = generate_depth_codes(&w, &cfg, &hidden_cond, &hidden_uncond, semantic_code, embed_code_row, &mut r);
        assert_eq!(codes.len(), cfg.num_codebooks as usize);
        assert_eq!(codes[0], semantic_code);
        assert_eq!(depth_hidden.len(), (cfg.num_codebooks as usize - 1) * d);

        let feedback = embed_audio_frame(&w, &cfg, &codes, embed_code_row);
        assert_eq!(feedback.len(), d);
        assert!(feedback.iter().all(|v| v.is_finite()), "feedback embedding must be finite");
    }

    /// The reference recipe [`generate_depth_codes`] replaced: re-run
    /// `depth_decoder::forward` over the WHOLE growing sequence at every
    /// depth step, on both CFG branches. Test-only, as the oracle the
    /// KV-cached production path is gated against.
    fn generate_depth_codes_uncached(
        dd_w: &DepthDecoderWeights,
        dd_cfg: &DepthDecoderConfig,
        hidden_cond: &[f32],
        hidden_uncond: &[f32],
        semantic_code: u32,
        embed_code_row: impl Fn(u32) -> Vec<f32>,
        rng: &mut Lcg,
    ) -> (Vec<u32>, Vec<f32>) {
        let d = dd_cfg.hidden_size as usize;
        let num_codebooks = dd_cfg.num_codebooks as usize;
        let audio_vocab = dd_cfg.audio_vocab_size as usize;

        let mut seq_cond = depth_decoder::projection(dd_w, dd_cfg, hidden_cond);
        let mut seq_uncond = depth_decoder::projection(dd_w, dd_cfg, hidden_uncond);
        let code_embed = embed_code_row(global_llm::audio_code_token_id(semantic_code));
        seq_cond.extend(depth_decoder::projection(dd_w, dd_cfg, &code_embed));
        seq_uncond.extend(depth_decoder::projection(dd_w, dd_cfg, &code_embed));

        let mut codes = vec![semantic_code];
        let mut depth_hidden = Vec::with_capacity((num_codebooks - 1) * d);
        for index in 1..num_codebooks {
            let s = seq_cond.len() / d;
            let (out_cond, _) = depth_decoder::forward(dd_w, dd_cfg, &seq_cond, s);
            let (out_uncond, _) = depth_decoder::forward(dd_w, dd_cfg, &seq_uncond, s);
            let h_cond = &out_cond[(s - 1) * d..s * d];
            let h_uncond = &out_uncond[(s - 1) * d..s * d];
            depth_hidden.extend_from_slice(h_cond);

            let logits_cond = depth_decoder::audio_head(dd_w, dd_cfg, index - 1, h_cond);
            let logits_uncond = depth_decoder::audio_head(dd_w, dd_cfg, index - 1, h_uncond);
            let guided = cfg_blend(&logits_cond, &logits_uncond, AR_CFG_SCALE);
            codes.push(sample_top_k(&guided, AR_SAMPLING_TOP_K, rng));

            if index < num_codebooks - 1 {
                let embed = depth_decoder::audio_embedding_row(dd_w, dd_cfg, codes[index] as usize + (index - 1) * audio_vocab);
                let proj = depth_decoder::projection(dd_w, dd_cfg, &embed);
                seq_cond.extend_from_slice(&proj);
                seq_uncond.extend_from_slice(&proj);
            }
        }
        (codes, depth_hidden)
    }

    /// The KV-cached depth loop must sample the SAME codes and return the
    /// SAME `depth_hidden` as the full-recompute reference recipe, exactly -
    /// `assert_eq!`, not an epsilon compare. Any drift here would change the
    /// generated audio.
    #[test]
    fn kv_cached_depth_loop_is_bit_identical_to_the_full_recompute_recipe() {
        let cfg = DepthDecoderConfig::tiny();
        let w = random_dd_weights(&cfg, 5);
        let d = cfg.hidden_size as usize;
        let mut r = Lcg::new(6);
        let hidden_cond = r.vec_scaled(d, 0.3);
        let hidden_uncond = r.vec_scaled(d, 0.3);
        let embed_code_row = |id: u32| {
            let mut rr = Lcg::new(u64::from(id));
            rr.vec_scaled(d, 0.1)
        };

        for semantic_code in 0..cfg.audio_vocab_size {
            let (codes_a, hidden_a) = generate_depth_codes(&w, &cfg, &hidden_cond, &hidden_uncond, semantic_code, embed_code_row, &mut Lcg::new(77));
            let (codes_b, hidden_b) = generate_depth_codes_uncached(&w, &cfg, &hidden_cond, &hidden_uncond, semantic_code, embed_code_row, &mut Lcg::new(77));
            assert_eq!(codes_a, codes_b, "semantic_code {semantic_code}: sampled codes diverged");
            assert_eq!(hidden_a, hidden_b, "semantic_code {semantic_code}: depth_hidden diverged");
        }
    }

    /// A `QwenConfig` with the REAL vocab size (so the audio-code/end-token
    /// id range this module's constants hardcode is actually addressable)
    /// but otherwise tiny dims - random weights, no real checkpoint needed.
    /// `d_model` is pinned to [`DepthDecoderConfig::tiny`]'s own
    /// `hidden_size` (the Global LLM's hidden state feeds straight into
    /// `depth_decoder::projection`, which assumes the two widths agree -
    /// exactly as the real checkpoint's own `hidden_size=4096` matches
    /// both components).
    fn tiny_real_vocab_qwen_config(block_size: u32) -> QwenConfig {
        QwenConfig { vocab: 200_000, block_size, max_position_embeddings: block_size, d_model: DepthDecoderConfig::tiny().hidden_size, ..QwenConfig::tiny() }
    }

    /// The full AR loop, end to end, at toy dims with random weights for
    /// both the Global LLM (real vocab, tiny everything else) and the
    /// depth decoder: proves the two-`Qwen`-instance CFG wiring, the
    /// masked-logit/top-k sampling, and the depth-decoder feedback loop
    /// all fit together and terminate with a well-shaped result - not a
    /// numerical parity check (no real reference exists for this exact
    /// composition), a structural/wiring one.
    #[test]
    fn generate_frames_produces_well_shaped_output_and_terminates() {
        let max_frames = 3usize;
        let prompt_len = 5usize;
        let qcfg = tiny_real_vocab_qwen_config((prompt_len + max_frames + 2) as u32);
        let qinit = qwen3::init_weights(&qcfg, 7);
        let lm_cond = Qwen::new(qcfg.clone(), 1, qcfg.block_size, &qinit);
        let lm_uncond = Qwen::new(qcfg.clone(), 1, qcfg.block_size, &qinit);
        let head = lm_cond.read_weight(qcfg.head_weight());

        let dd_cfg = DepthDecoderConfig::tiny();
        let dd_w = random_dd_weights(&dd_cfg, 8);

        let mut r = Lcg::new(3);
        let conditional_ids: Vec<u32> = (0..prompt_len).map(|_| r.next_u32() % qcfg.vocab).collect();
        let unconditional_ids = conditional_ids.clone();

        let frame_hiddens = generate_frames(
            &lm_cond,
            &lm_uncond,
            &dd_w,
            &dd_cfg,
            &head,
            qcfg.vocab as usize,
            qcfg.d_model as usize,
            &conditional_ids,
            &unconditional_ids,
            max_frames,
            99,
            &mut crate::ignore_progress(),
        );

        let frame_width = dd_cfg.num_codebooks as usize * dd_cfg.hidden_size as usize;
        assert_eq!(frame_hiddens.len() % frame_width, 0, "frame_hiddens must be a whole number of frames");
        let frames = frame_hiddens.len() / frame_width;
        assert!(frames <= max_frames, "generated {frames} frames, expected at most {max_frames}");
        assert!(frame_hiddens.iter().all(|v| v.is_finite()), "every frame hidden value must be finite");
    }
}
