// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `Florence2Lm`: the top-level BART-style encoder-decoder orchestration -
//! `Florence2ForConditionalGeneration.forward`'s `language_model` call plus
//! its own `_merge_input_ids_with_image_features`/`lm_head`/
//! `final_logits_bias` tail. `scale_embedding=false` in the real checkpoint
//! (verified against `config.json`, not assumed - see
//! [`super::config::BartConfig::florence2_base`]'s doc), so text token
//! embeddings need no `sqrt(d_model)` scale - a plain `embed` gather from
//! `language_model.model.shared.weight` (tied across encoder input, decoder
//! input, AND the LM head - one tensor, three uses; HF's own weight tying
//! silently fails to alias this checkpoint's `encoder.embed_tokens`/
//! `decoder.embed_tokens`/`lm_head` onto it, confirmed via a direct
//! `torch.equal` check, so reading `shared.weight` directly for all three
//! uses here is not just simpler but the only way to get the trained
//! weights at all).
//!
//! No padding/attention mask anywhere: this crate's only consumer is the
//! android-ui-test initiative's `ground` capability, always a single
//! unbatched image + text query, so the reference's `attention_mask`
//! plumbing (built only to mask padding in a batch) is simply never needed
//! and never built.

use gpu_core::{DeviceBuffer, Gpu};
use model::vit::row_index_buffer;

use super::config::BartConfig;
use super::decoder::{Decoder, DecoderKernelIds};
use super::encoder::{Encoder, EncoderKernelIds};

pub struct Florence2LmKernelIds {
    pub encoder: EncoderKernelIds,
    pub decoder: DecoderKernelIds,
    pub embed: usize,
    pub row_scatter: usize,
    pub matmul_rows: usize,
    pub bias_add: usize,
}

impl Florence2LmKernelIds {
    pub fn resolve(pipelines: &[(&str, &str)]) -> Florence2LmKernelIds {
        let k = |name: &str| pipelines.iter().position(|(n, _)| *n == name).expect(name);
        Florence2LmKernelIds {
            encoder: EncoderKernelIds::resolve(pipelines),
            decoder: DecoderKernelIds::resolve(pipelines),
            embed: k("embed"),
            row_scatter: k("row_scatter"),
            matmul_rows: k("matmul_rows"),
            bias_add: k("bias_add"),
        }
    }
}

pub struct Florence2Lm {
    cfg: BartConfig,
    t_vision: u32,
    t_prompt: u32,
    t_total: u32,
    max_decode_len: u32,
    vision_idx: DeviceBuffer,
    encoder_input: DeviceBuffer,
    encoder: Encoder,
    decoder: Decoder,
    dec_embedded: DeviceBuffer,
    logits: DeviceBuffer,
}

impl Florence2Lm {
    /// `t_vision`: the vision-token count `ImageProject` produced for this
    /// query (Florence-2-base: 577). `t_prompt`: the tokenized text-prompt
    /// length (task marker + query phrase, e.g.
    /// `<CAPTION_TO_PHRASE_GROUNDING>the Login button`). `max_decode_len`:
    /// the decode-buffer ceiling (see `text::attn`'s module doc) - for
    /// [`Florence2Lm::generate`] this must cover `max_new_tokens + 1` (the
    /// leading `decoder_start_token_id` counts against the ceiling too).
    pub fn new(gpu: &Gpu, cfg: BartConfig, t_vision: u32, t_prompt: u32, max_decode_len: u32) -> Florence2Lm {
        let t_total = t_vision + t_prompt;
        let d = cfg.d_model;
        let vision_rows: Vec<u32> = (0..t_vision).collect();
        Florence2Lm {
            cfg,
            t_vision,
            t_prompt,
            t_total,
            max_decode_len,
            vision_idx: row_index_buffer(gpu, "florence2_lm_vision_rows", &vision_rows),
            encoder_input: gpu.storage((t_total * d) as u64),
            encoder: Encoder::new(gpu, cfg.encoder_attention_heads, d, cfg.encoder_ffn_dim, cfg.encoder_layers, t_total, cfg.layer_norm_eps),
            decoder: Decoder::new(gpu, cfg.decoder_attention_heads, d, cfg.decoder_ffn_dim, cfg.decoder_layers, max_decode_len, t_total, cfg.layer_norm_eps),
            dec_embedded: gpu.storage((max_decode_len * d) as u64),
            logits: gpu.storage((max_decode_len * cfg.vocab_size) as u64),
        }
    }

    /// Concats `[image_features; text_prompt_embeds]` (`image_features`:
    /// `[t_vision,d_model]`, `crate::vision::ImageProject`'s output;
    /// `prompt_ids`: this query's tokenized text prompt, length `t_prompt`)
    /// and runs the encoder. Returns the `[t_total,d_model]` encoder memory
    /// every decode step cross-attends over.
    pub fn encode<'a>(&'a self, gpu: &Gpu, k: &Florence2LmKernelIds, ps: &paramstore::ParamStore, image_features: &DeviceBuffer, prompt_ids: &[u32]) -> &'a DeviceBuffer {
        assert_eq!(prompt_ids.len() as u32, self.t_prompt, "florence2 lm: prompt_ids length must match t_prompt");
        let d = self.cfg.d_model;
        let shared = ps.w("language_model.model.shared.weight");
        let prompt_idx = row_index_buffer(gpu, "florence2_lm_prompt_ids", prompt_ids);

        let vision_words = (self.t_vision * d) as u64;
        let s0 = vec![
            gpu.step(k.row_scatter, &[&self.vision_idx, image_features, &self.encoder_input], &[self.t_vision, d, self.t_total], self.t_vision * d),
            gpu.step_sliced(k.embed, &[&prompt_idx, shared, &self.encoder_input], &[(0, 0), (0, 0), (vision_words, (self.t_prompt * d) as u64)], &[d, self.t_prompt], self.t_prompt * d),
        ];
        gpu.submit(&[], &s0);

        self.encoder.forward(gpu, &k.encoder, ps, &self.encoder_input)
    }

    /// `decoder_ids`: the full decoder-input-id prefix so far (teacher-forced
    /// target ids, or the greedily-sampled prefix during generation), always
    /// starting with `decoder_start_token_id`. `enc`: [`Florence2Lm::encode`]'s
    /// output. Returns logits for every position, `[decoder_ids.len(),
    /// vocab_size]` - row `t-1` is the distribution for the NEXT token.
    pub fn decode<'a>(&'a self, gpu: &Gpu, k: &Florence2LmKernelIds, ps: &paramstore::ParamStore, enc: &DeviceBuffer, decoder_ids: &[u32]) -> &'a DeviceBuffer {
        let t = decoder_ids.len() as u32;
        assert!(t <= self.max_decode_len, "florence2 lm: decoder_ids length {t} exceeds max_decode_len {}", self.max_decode_len);
        let d = self.cfg.d_model;
        let shared = ps.w("language_model.model.shared.weight");
        let bias = ps.w("language_model.final_logits_bias");
        let dec_idx = row_index_buffer(gpu, "florence2_lm_dec_ids", decoder_ids);

        gpu.submit(&[], &[gpu.step(k.embed, &[&dec_idx, shared, &self.dec_embedded], &[d, t], t * d)]);

        let hidden = self.decoder.forward(gpu, &k.decoder, ps, &self.dec_embedded, enc, t);

        let s = vec![
            gpu.step(k.matmul_rows, &[hidden, shared, &self.logits], &[t, d, self.cfg.vocab_size], t.div_ceil(8) * self.cfg.vocab_size),
            gpu.step(k.bias_add, &[&self.logits, bias], &[t, self.cfg.vocab_size], t * self.cfg.vocab_size),
        ];
        gpu.submit(&[], &s);
        &self.logits
    }

    /// Greedy (argmax, no sampling) autoregressive generation: starts from
    /// `decoder_start_token_id`, decodes one token at a time (each step a
    /// full [`Florence2Lm::decode`] call - see `text::attn`'s module doc for
    /// why full recompute, no KV cache, is the right tradeoff here), and
    /// stops at `eos_token_id` or `max_new_tokens`. The argmax itself runs
    /// on the HOST: reading back one `[vocab_size]` row per step is cheap
    /// next to the decoder forward it follows, and keeps sampling strategy
    /// (this method's greedy policy today, top-k/nucleus later if grounding
    /// quality ever needs it) out of the GPU dispatch path entirely.
    ///
    /// Returns the generated ids, NOT including `decoder_start_token_id`
    /// (matching the reference's own `generate()` convention of returning
    /// the sequence a caller then strips the seed token from) - EOS IS
    /// included if generation stopped by producing it.
    pub fn generate(&self, gpu: &Gpu, k: &Florence2LmKernelIds, ps: &paramstore::ParamStore, enc: &DeviceBuffer, max_new_tokens: u32) -> Vec<u32> {
        let vocab = self.cfg.vocab_size as usize;
        let mut ids = vec![self.cfg.decoder_start_token_id];
        let mut generated = Vec::with_capacity(max_new_tokens as usize);
        for _ in 0..max_new_tokens {
            let logits = self.decode(gpu, k, ps, enc, &ids);
            gpu.poll_wait();
            let t = ids.len();
            let last_row = gpu.read(logits, t * vocab);
            let last_row = &last_row[(t - 1) * vocab..t * vocab];
            let next = last_row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).expect("vocab is non-empty");
            ids.push(next);
            generated.push(next);
            if next == self.cfg.eos_token_id {
                break;
            }
        }
        generated
    }
}
