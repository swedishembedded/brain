// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Qwen2LM` (CosyVoice 2) speech-token LM: prompt assembly + autoregressive
//! generation, hosted on `qwen3::Qwen`'s Qwen2.5-0.5B decoder.
//!
//! ## Prompt assembly (`Qwen2LM.inference`, reproduced verbatim)
//! ```text
//! text          = concat([prompt_text, text])                  // Qwen BPE token ids
//! text_emb      = qwen_backbone.embed_tokens(text)              // via the SAME tok.weight the backbone forward uses
//! sos_emb       = llm_embedding.weight[0]                       // [1, d]
//! task_id_emb   = llm_embedding.weight[1]                       // [1, d]
//! prompt_speech_emb = speech_embedding(prompt_speech_token)     // [m, d], or empty
//! lm_input      = concat([sos_emb, text_emb, task_id_emb, prompt_speech_emb], dim=1)
//! ```
//! Note the reference's `Qwen2LM.inference` accepts a speaker `embedding`
//! parameter but never references it in the concat above (unlike the base
//! `TransformerLM.inference`, which does) - verified by reading
//! `resources/cosyvoice/source/cosyvoice/llm/llm.py` line-for-line, not
//! assumed from the docstring plan. [`CosyVoiceLm::prefill`] therefore takes
//! no speaker embedding either.
//!
//! ## Hidden-state readout
//! The reference runs the WHOLE prefix through one batched
//! `Qwen2Encoder.forward_one_step` call (`cache=None`) and reads
//! `hidden_states[-1]` (the final-RMSNorm hidden state) for EVERY position -
//! this is what `llm_real_prefill_hidden.f32` captures. This crate reproduces
//! it by walking the SAME prefix one row at a time through
//! `qwen3::Qwen::step_embed`'s incremental KV-cache decode instead of a
//! batched forward: causal attention makes the two algebraically identical
//! per position (proven generally by `qwen3tts::gen::kv_tests::
//! kv_step_matches_full_recompute` for the same block math), and brain's
//! `qwen3::Qwen` decode-only build exposes no batched "forward over a raw
//! embedding sequence" entry point to call instead.
//!
//! ## Generation loop
//! At each step: `hidden = qwen_backbone.step_embed(lm_input_row)`, `logits =
//! llm_decoder(hidden)` (CosyVoice's OWN head - 896 -> 6564 - never the Qwen
//! backbone's own discarded 151936-wide `lm_head`), sample via
//! [`crate::sampling::ras_sampling`], feed `speech_embedding.weight[token_id]`
//! as the next row. Stops on `cfg.stop_token_ids` or a step cap.

use crate::config::CosyVoiceLmConfig;
use crate::llm_import::LmWeights;
use crate::sampling::{log_softmax, ras_sampling, RasParams};
use data::rng::Rng;

fn row(table: &[f32], d: usize, i: u32) -> &[f32] {
    let s = i as usize * d;
    &table[s..s + d]
}

/// The `Qwen2LM` speech-token LM: a `qwen3::Qwen` decode-only backbone plus
/// CosyVoice's own bolted-on `llm_embedding`/`speech_embedding`/`llm_decoder`
/// tables.
pub struct CosyVoiceLm {
    pub cfg: CosyVoiceLmConfig,
    qwen: qwen3::Qwen,
    llm_embedding: Vec<f32>,
    speech_embedding: Vec<f32>,
    llm_decoder_w: Vec<f32>,
    llm_decoder_b: Vec<f32>,
}

impl CosyVoiceLm {
    /// Build from already-imported weights (see [`crate::llm_import::import_llm_pt`]).
    /// `ctx` sizes the KV cache: it must cover the assembled prefix length plus
    /// the AR decode budget the caller intends to run.
    pub fn from_weights(cfg: CosyVoiceLmConfig, w: LmWeights, ctx: u32) -> CosyVoiceLm {
        let qcfg = cfg.to_qwen_config(ctx);
        let qwen = qwen3::Qwen::from_tensors_decode(qcfg, &w.backbone, ctx);
        CosyVoiceLm {
            cfg,
            qwen,
            llm_embedding: w.llm_embedding,
            speech_embedding: w.speech_embedding,
            llm_decoder_w: w.llm_decoder_w,
            llm_decoder_b: w.llm_decoder_b,
        }
    }

    /// Import `llm.pt` and build in one step.
    pub fn load(llm_pt_path: &str, ctx: u32) -> Result<CosyVoiceLm, String> {
        let cfg = CosyVoiceLmConfig::cosyvoice2();
        let w = crate::llm_import::import_llm_pt(llm_pt_path, &cfg)?;
        Ok(CosyVoiceLm::from_weights(cfg, w, ctx))
    }

    fn d(&self) -> usize {
        self.cfg.llm_input_size as usize
    }

    /// `speech_token_size + 3` (6564 for the real model).
    pub fn speech_vocab(&self) -> usize {
        self.cfg.speech_vocab() as usize
    }

    /// `qwen_backbone.embed_tokens(id)` - the Qwen backbone's OWN token
    /// embedding table (`tok.weight`), the same one `text_emb` in the module
    /// doc's prompt assembly comes from.
    fn text_embed(&self, id: u32) -> Vec<f32> {
        self.qwen.embed_row(id)
    }

    fn llm_embed(&self, i: u32) -> &[f32] {
        row(&self.llm_embedding, self.d(), i)
    }

    fn speech_embed(&self, id: u32) -> &[f32] {
        row(&self.speech_embedding, self.d(), id)
    }

    /// `llm_decoder(hidden)` - CosyVoice's own `Linear(896, speech_vocab)`
    /// head, applied to one hidden row.
    pub fn decoder_logits(&self, hidden_row: &[f32]) -> Vec<f32> {
        let d = self.d();
        let v = self.speech_vocab();
        assert_eq!(hidden_row.len(), d);
        let mut out = vec![0.0f32; v];
        for (o, dst) in out.iter_mut().enumerate() {
            let wrow = &self.llm_decoder_w[o * d..(o + 1) * d];
            let mut acc = self.llm_decoder_b[o];
            for k in 0..d {
                acc += wrow[k] * hidden_row[k];
            }
            *dst = acc;
        }
        out
    }

    /// `llm_decoder(hidden)` over every row of a flattened `[n, d]` hidden
    /// buffer, returning `[n, speech_vocab]` flattened.
    pub fn decoder_logits_all(&self, hidden: &[f32]) -> Vec<f32> {
        let d = self.d();
        assert_eq!(hidden.len() % d, 0, "hidden buffer is not a whole number of {d}-dim rows");
        let n = hidden.len() / d;
        let mut out = Vec::with_capacity(n * self.speech_vocab());
        for i in 0..n {
            out.extend(self.decoder_logits(&hidden[i * d..(i + 1) * d]));
        }
        out
    }

    /// Assemble `sos ++ text_emb ++ task_id ++ prompt_speech_emb` and run the
    /// prefill (see the module doc's "hidden-state readout" note), returning
    /// every position's final-norm hidden state, `[n, d]` flattened where
    /// `n = 1 + text_ids.len() + 1 + prompt_speech_tokens.len()`.
    /// `text_ids` is `concat([prompt_text, text])`'s token ids - the caller's
    /// job, matching the reference's own `torch.concat([prompt_text, text])`.
    pub fn prefill(&self, text_ids: &[u32], prompt_speech_tokens: &[u32]) -> Vec<f32> {
        self.qwen.reset_cache();
        let n = 1 + text_ids.len() + 1 + prompt_speech_tokens.len();
        let mut hidden = Vec::with_capacity(n * self.d());

        hidden.extend(self.qwen.step_embed(self.llm_embed(self.cfg.sos)));
        for &id in text_ids {
            let e = self.text_embed(id);
            hidden.extend(self.qwen.step_embed(&e));
        }
        hidden.extend(self.qwen.step_embed(self.llm_embed(self.cfg.task_id)));
        for &tok in prompt_speech_tokens {
            let e = self.speech_embed(tok).to_vec();
            hidden.extend(self.qwen.step_embed(&e));
        }
        hidden
    }

    /// Autoregressive decode continuing from the KV-cache position [`Self::prefill`]
    /// left the model at, using `last_hidden` (the prefill's own last row) as
    /// the first step's hidden state. Stops at `cfg.stop_token_ids` or
    /// `max_tokens`, whichever comes first. `min_len` mirrors the reference's
    /// `min_token_text_ratio` gate: `eos` is masked out of the distribution for
    /// steps `< min_len`.
    ///
    /// **Not bit-exact with the reference sampler** - see `crate::sampling`'s
    /// module doc. `seed` selects brain's OWN reproducible RNG stream, so the
    /// SAME seed always yields the SAME token sequence on this implementation
    /// (verified in `crates/cosyvoice/tests/llm_parity.rs`), but that sequence
    /// will not match `torch.manual_seed`'s.
    pub fn generate(&self, last_hidden: &[f32], max_tokens: usize, min_len: usize, seed: u64) -> Vec<u32> {
        assert_eq!(last_hidden.len(), self.d());
        let mut rng = Rng::new(seed);
        let params = RasParams::default();
        let mut tokens: Vec<u32> = Vec::new();
        let mut hidden = last_hidden.to_vec();
        for i in 0..max_tokens {
            let logits = self.decoder_logits(&hidden);
            let mut logp = log_softmax(&logits);
            if i < min_len {
                logp[self.cfg.eos_token as usize] = f32::NEG_INFINITY;
            }
            let tok = ras_sampling(&mut rng, &mut logp, &tokens, &params);
            if self.cfg.stop_token_ids.contains(&tok) {
                break;
            }
            tokens.push(tok);
            let next = self.speech_embed(tok).to_vec();
            hidden = self.qwen.step_embed(&next);
        }
        tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_weights(cfg: &CosyVoiceLmConfig) -> LmWeights {
        let mut backbone = HashMap::new();
        for (name, numel) in cfg.qwen.param_list() {
            let v: Vec<f32> = (0..numel).map(|i| ((i % 7) as f32 - 3.0) * 0.05).collect();
            backbone.insert(name, v);
        }
        let d = cfg.llm_input_size as usize;
        let v = cfg.speech_vocab() as usize;
        LmWeights {
            backbone,
            llm_embedding: (0..2 * d).map(|i| (i % 5) as f32 * 0.01).collect(),
            speech_embedding: (0..v * d).map(|i| (i % 11) as f32 * 0.01 - 0.05).collect(),
            llm_decoder_w: (0..v * d).map(|i| (i % 13) as f32 * 0.01 - 0.06).collect(),
            llm_decoder_b: vec![0.0f32; v],
        }
    }

    fn tiny_cfg() -> CosyVoiceLmConfig {
        let mut cfg = CosyVoiceLmConfig::cosyvoice2();
        cfg.qwen = qwen3::QwenConfig::qwen2(29, 2, 16, 4, 2, 32, true);
        cfg.llm_input_size = 16;
        cfg.llm_output_size = 16;
        cfg.speech_token_size = 20;
        cfg.eos_token = 20;
        cfg.fill_token = 22;
        cfg.stop_token_ids = [20, 21, 22];
        cfg
    }

    #[test]
    fn prefill_produces_one_hidden_row_per_prompt_position() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg);
        let d = cfg.llm_input_size as usize;
        let lm = CosyVoiceLm::from_weights(cfg.clone(), w, 64);

        let text_ids = [3u32, 5, 7, 2];
        let prompt_speech = [1u32, 4, 9];
        let hidden = lm.prefill(&text_ids, &prompt_speech);
        let n = 1 + text_ids.len() + 1 + prompt_speech.len();
        assert_eq!(hidden.len(), n * d);
        assert!(hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn generate_stops_within_the_token_cap_and_never_emits_a_stop_id() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg);
        let lm = CosyVoiceLm::from_weights(cfg.clone(), w, 64);

        let hidden = lm.prefill(&[3, 5, 7], &[1, 4]);
        let d = cfg.llm_input_size as usize;
        let last = &hidden[hidden.len() - d..];
        let tokens = lm.generate(last, 16, 0, 1234);
        assert!(tokens.len() <= 16);
        for &t in &tokens {
            assert!(!cfg.stop_token_ids.contains(&t), "generate() must not push a stop id into its own output");
            assert!((t as usize) < lm.speech_vocab());
        }
    }

    #[test]
    fn generate_is_deterministic_given_the_same_seed() {
        let cfg = tiny_cfg();
        let w = tiny_weights(&cfg);
        let lm = CosyVoiceLm::from_weights(cfg.clone(), w, 64);
        let d = cfg.llm_input_size as usize;

        let h1 = lm.prefill(&[3, 5, 7], &[1, 4]);
        let a = lm.generate(&h1[h1.len() - d..], 12, 0, 99);
        let h2 = lm.prefill(&[3, 5, 7], &[1, 4]);
        let b = lm.generate(&h2[h2.len() - d..], 12, 0, 99);
        assert_eq!(a, b, "same seed + same prefix must reproduce the same generated sequence");
    }
}
