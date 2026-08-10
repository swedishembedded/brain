// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Talker generation: the codec-id sampling loop chaining Talker's own
//! KV-cache decode (`crate::talker`) with the MTP code predictor
//! (`tts::mtp::MtpModel::generate_residuals`, already built and validated
//! against real Omni weights — M7b — reused unchanged). Mirrors
//! `tts::pipeline::generate_codes`'s already-working loop shape exactly
//! (same `sample_cb0` suppressed-token-set logic, same feedback-sum-then-
//! decode-step pattern), adapted for Talker's MoE decoder instead of
//! `tts::talker`'s dense one and for this crate's per-token weight-streaming
//! pattern (`crate::generate`'s module doc — the same tradeoff applies here:
//! correct, not fast, every layer's weights re-read from `reader` on every
//! prefill call and every decode step).

use checkpoint::weightio::WeightReader;
use data::rng::Rng;
use gpu_core::{DeviceBuffer, Gpu};
use qwenvl::mrope::{get_rope_index, mrope_tables};
use tts::mtp::MtpModel;

use crate::config::MoeTextConfig;
use crate::talker::{layer_decode_step, layer_fwd, TalkerLayerCache, TalkerLayerWeights};
use crate::talker_prompt::TalkerPrompt;
use crate::thinker::{final_norm, lm_head_fwd};

/// One Talker decoder layer's weights, freshly uploaded from `reader` —
/// same shape/lifetime contract as `crate::generate::OwnedLayer`.
struct OwnedTalkerLayer {
    ln1: DeviceBuffer,
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    wo: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    ln2: DeviceBuffer,
    router: DeviceBuffer,
    experts: Vec<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,
    shared_gate: DeviceBuffer,
    shared_up: DeviceBuffer,
    shared_down: DeviceBuffer,
    shared_expert_gate: DeviceBuffer,
}

impl OwnedTalkerLayer {
    fn as_weights(&self) -> TalkerLayerWeights<'_> {
        TalkerLayerWeights {
            ln1: &self.ln1,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            wo: &self.wo,
            q_norm: &self.q_norm,
            k_norm: &self.k_norm,
            ln2: &self.ln2,
            router: &self.router,
            experts: &self.experts,
            shared_expert: (&self.shared_gate, &self.shared_up, &self.shared_down),
            shared_expert_gate: &self.shared_expert_gate,
        }
    }
}

/// Reads RAW HF tensor names (`talker.model.layers.{l}.*`) straight from
/// `reader` — same convention `crate::generate::load_thinker_layer` uses:
/// `reader` is opened on the raw HF checkpoint directory, not the (still
/// separately loader-naming-gapped, per M7b/M8) unified/imported one.
fn load_talker_layer(reader: &WeightReader, gpu: &Gpu, l: u32, n_experts: u32) -> Result<OwnedTalkerLayer, String> {
    // Errors, not panics: streamed per generated codec frame inside the
    // serving daemon -- a missing tensor fails the request, not the process.
    let p = |leaf: &str| format!("talker.model.layers.{l}.{leaf}");
    let get = |name: &str| Ok::<_, String>(gpu.storage_init("w", &reader.tensor(name).ok_or_else(|| format!("omni: missing tensor {name}"))?));
    Ok(OwnedTalkerLayer {
        ln1: get(&p("input_layernorm.weight"))?,
        wq: get(&p("self_attn.q_proj.weight"))?,
        wk: get(&p("self_attn.k_proj.weight"))?,
        wv: get(&p("self_attn.v_proj.weight"))?,
        wo: get(&p("self_attn.o_proj.weight"))?,
        q_norm: get(&p("self_attn.q_norm.weight"))?,
        k_norm: get(&p("self_attn.k_norm.weight"))?,
        ln2: get(&p("post_attention_layernorm.weight"))?,
        router: get(&p("mlp.gate.weight"))?,
        experts: (0..n_experts)
            .map(|e| Ok((get(&p(&format!("mlp.experts.{e}.gate_proj.weight")))?, get(&p(&format!("mlp.experts.{e}.up_proj.weight")))?, get(&p(&format!("mlp.experts.{e}.down_proj.weight")))?)))
            .collect::<Result<Vec<_>, String>>()?,
        shared_gate: get(&p("mlp.shared_expert.gate_proj.weight"))?,
        shared_up: get(&p("mlp.shared_expert.up_proj.weight"))?,
        shared_down: get(&p("mlp.shared_expert.down_proj.weight"))?,
        shared_expert_gate: get(&p("mlp.shared_expert_gate.weight"))?,
    })
}

struct TalkerKvCache {
    layers: Vec<(DeviceBuffer, DeviceBuffer)>,
    cap: u32,
}

impl TalkerKvCache {
    fn new(gpu: &Gpu, cfg: &MoeTextConfig, cap: u32) -> Self {
        let hkv = (cfg.n_kv_heads * cfg.head_dim) as u64;
        let layers = (0..cfg.n_layers).map(|_| (gpu.storage(cap as u64 * hkv), gpu.storage(cap as u64 * hkv))).collect();
        Self { layers, cap }
    }
    fn layer(&self, l: usize) -> TalkerLayerCache<'_> {
        TalkerLayerCache { kcache: &self.layers[l].0, vcache: &self.layers[l].1 }
    }
}

/// Prefill: `x_host [n, hidden]` through every Talker layer once (batched
/// causal attention), bulk-filling `cache` — plain-sequential positions
/// (the text-only prefill this crate builds today, `crate::talker_prompt`'s
/// scope note). Returns the final-normed hidden state `[n, hidden]`.
fn prefill(reader: &WeightReader, gpu: &Gpu, cfg: &MoeTextConfig, x_host: &[f32], n: u32, cache: &TalkerKvCache) -> Result<DeviceBuffer, String> {
    let tokens: Vec<u32> = (0..n).collect();
    let positions = get_rope_index(&tokens, u32::MAX, &[]);
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(&positions, section, cfg.head_dim, cfg.rope_theta);
    let cos = gpu.storage_init("cos", &cos_tab);
    let sin = gpu.storage_init("sin", &sin_tab);

    let mut h = gpu.storage_init("x", x_host);
    for l in 0..cfg.n_layers {
        let layer = load_talker_layer(reader, gpu, l, cfg.n_experts)?;
        let lc = cache.layer(l as usize);
        let (out, ..) = layer_fwd(gpu, cfg, &layer.as_weights(), &h, &cos, &sin, n, Some(&lc));
        h = out;
    }
    let norm_w = gpu.storage_init("w", &reader.tensor("talker.model.norm.weight").ok_or("omni: missing tensor talker.model.norm.weight")?);
    Ok(final_norm(gpu, cfg, &norm_w, &h, n))
}

/// One incremental decode step: a single new "frame" embedding row (the
/// summed codec feedback, see [`generate_codes`]) through every layer,
/// attending against `cache` at cache row `cache_row` / M-RoPE position
/// `pos` (plain-sequential — same scope note as [`prefill`]).
fn decode_step(reader: &WeightReader, gpu: &Gpu, cfg: &MoeTextConfig, x_host: &[f32], pos: u32, cache: &TalkerKvCache) -> Result<DeviceBuffer, String> {
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(&[[pos, pos, pos]], section, cfg.head_dim, cfg.rope_theta);
    let cos = gpu.storage_init("cos", &cos_tab);
    let sin = gpu.storage_init("sin", &sin_tab);

    let mut h = gpu.storage_init("x", x_host);
    for l in 0..cfg.n_layers {
        let layer = load_talker_layer(reader, gpu, l, cfg.n_experts)?;
        let lc = cache.layer(l as usize);
        h = layer_decode_step(gpu, cfg, &layer.as_weights(), &lc, &h, &cos, &sin, pos, cache.cap);
    }
    let norm_w = gpu.storage_init("w", &reader.tensor("talker.model.norm.weight").ok_or("omni: missing tensor talker.model.norm.weight")?);
    Ok(final_norm(gpu, cfg, &norm_w, &h, 1))
}

/// Sampling / length controls — identical defaults to `tts::pipeline::
/// GenOpts` (the reference `Qwen3TTSModel._merge_generate_kwargs`'s
/// `do_sample=True, top_k=50, temperature=0.9`; greedy collapses codebook-0
/// into a repeating token after a few frames, per that module's own doc).
#[derive(Clone, Debug)]
pub struct GenOpts {
    pub max_frames: u32,
    pub temperature: f32,
    pub top_k: usize,
    pub seed: u64,
    pub min_new: u32,
}

impl Default for GenOpts {
    fn default() -> GenOpts {
        GenOpts { max_frames: 256, temperature: 0.9, top_k: 50, seed: 0, min_new: 2 }
    }
}

/// Sample codebook-0 from `logits` with the reference's `suppress_tokens`:
/// the top-1024 vocab entries are masked except the codec EOS, itself masked
/// unless `allow_eos` — identical logic to `tts::pipeline::sample_cb0`
/// (kept as a second, small copy rather than a shared crate dependency
/// since `tts::pipeline::sample_cb0` is `pub(crate)` there; if a THIRD
/// caller ever needs this, hoist it to `model::` or `data::` instead of
/// adding a fourth).
fn sample_cb0(mut logits: Vec<f32>, eos: u32, allow_eos: bool, temperature: f32, top_k: usize, rng: &mut Rng) -> u32 {
    let v = logits.len();
    let lo = v - 1024;
    let eos_logit = logits[eos as usize];
    for x in logits[lo..].iter_mut() {
        *x = f32::NEG_INFINITY;
    }
    if allow_eos {
        logits[eos as usize] = eos_logit;
    }
    if temperature <= 0.0 {
        return logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).expect("non-empty vocab");
    }
    let mut scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();
    if top_k > 0 && top_k < scaled.len() {
        let mut idx: Vec<usize> = (0..scaled.len()).collect();
        idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
        let threshold = scaled[idx[top_k - 1]];
        for x in scaled.iter_mut() {
            if *x < threshold {
                *x = f32::NEG_INFINITY;
            }
        }
    }
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for x in scaled.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    let r = rng.next_f32() * sum;
    let mut acc = 0.0f32;
    for (i, &p) in scaled.iter().enumerate() {
        acc += p;
        if acc >= r {
            return i as u32;
        }
    }
    (scaled.len() - 1) as u32
}

fn add_into(a: &mut [f32], b: &[f32]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Autoregressively generate codec codes `[n_frames*16]` (row-major,
/// codebooks 0..15 per frame) for an assembled [`TalkerPrompt`]. Stops at
/// `codec_eos_id` (once `min_new` frames have been produced) or
/// `opts.max_frames`. `codec_head_w` is `talker.codec_head.weight`
/// (`[vocab, hidden]` — tied to `codec_embedding` in the reference, but
/// materialized as its own tensor in the real checkpoint, loaded like any
/// other weight here).
#[allow(clippy::too_many_arguments)]
pub fn generate_codes(reader: &WeightReader, gpu: &Gpu, cfg: &MoeTextConfig, codec_head_w: &[f32], codec_eos_id: u32, mtp: &MtpModel, codec_embed: impl Fn(u32) -> Vec<f32>, prompt: &TalkerPrompt, opts: &GenOpts) -> Result<Vec<u32>, String> {
    let d = cfg.hidden as usize;
    let n_prefix = (prompt.embeds.len() / d) as u32;
    let n_trailing = prompt.trailing.len() / d;
    let mut rng = Rng::new(opts.seed);

    let cap = n_prefix + opts.max_frames;
    let cache = TalkerKvCache::new(gpu, cfg, cap);
    let codec_head = gpu.storage_init("codec_head", codec_head_w);

    let hidden = prefill(reader, gpu, cfg, &prompt.embeds, n_prefix, &cache)?;
    let logits_all = lm_head_fwd(gpu, &codec_head, &hidden, n_prefix, cfg.hidden, cfg.vocab);
    let last_logits = gpu.read(&logits_all, (n_prefix * cfg.vocab) as usize)[((n_prefix - 1) * cfg.vocab) as usize..].to_vec();
    let mut past_hidden = gpu.read(&hidden, (n_prefix * cfg.hidden) as usize)[((n_prefix - 1) * cfg.hidden) as usize..].to_vec();
    let mut cb0 = sample_cb0(last_logits, codec_eos_id, opts.min_new == 0, opts.temperature, opts.top_k, &mut rng);

    let mut frames: Vec<u32> = Vec::new();
    let mut s = 0u32;
    let mut cache_row = n_prefix;
    loop {
        if (cb0 == codec_eos_id && s >= opts.min_new) || s >= opts.max_frames {
            break;
        }
        let cb0_embed = codec_embed(cb0);
        let (residuals, res_sum) = mtp.generate_residuals(&past_hidden, &cb0_embed);
        frames.push(cb0);
        frames.extend_from_slice(&residuals);

        let mut feed = cb0_embed;
        add_into(&mut feed, &res_sum);
        if (s as usize) < n_trailing {
            add_into(&mut feed, &prompt.trailing[s as usize * d..(s as usize + 1) * d]);
        } else {
            add_into(&mut feed, &prompt.tts_pad_embed);
        }
        s += 1;
        if cache_row >= cap {
            break;
        }

        let hidden = decode_step(reader, gpu, cfg, &feed, cache_row, &cache)?;
        let logits = lm_head_fwd(gpu, &codec_head, &hidden, 1, cfg.hidden, cfg.vocab);
        past_hidden = gpu.read(&hidden, cfg.hidden as usize);
        cb0 = sample_cb0(gpu.read(&logits, cfg.vocab as usize), codec_eos_id, s >= opts.min_new, opts.temperature, opts.top_k, &mut rng);
        cache_row += 1;
    }
    Ok(frames)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The suppressed-token-set (top-1024 ids masked except EOS) and the
    /// EOS-gate (`allow_eos`) are the two easy-to-get-subtly-wrong pieces of
    /// `sample_cb0` -- greedy (temperature<=0) makes the outcome
    /// deterministic and checkable exactly, without needing to reason about
    /// the sampling distribution.
    #[test]
    fn greedy_never_picks_a_suppressed_token_unless_it_is_the_allowed_eos() {
        let vocab = 2048usize;
        let eos = (vocab - 1024 + 5) as u32; // inside the suppressed range
        let mut logits = vec![0.0f32; vocab];
        logits[vocab - 1] = 100.0; // the single highest logit is suppressed
        logits[eos as usize] = 50.0; // EOS is the next-highest
        logits[10] = 1.0; // a normal, unsuppressed candidate

        let mut rng = Rng::new(0);
        // EOS not allowed yet: the whole suppressed range (incl. EOS) is
        // masked, so greedy must fall through to the unsuppressed candidate.
        let picked = sample_cb0(logits.clone(), eos, false, 0.0, 0, &mut rng);
        assert_eq!(picked, 10, "must not pick a suppressed token when EOS isn't allowed either");

        // EOS allowed: EOS is un-masked and is now the highest surviving logit.
        let picked = sample_cb0(logits.clone(), eos, true, 0.0, 0, &mut rng);
        assert_eq!(picked, eos, "EOS must be selectable once allow_eos is true");

        // The single highest-logit token (outside EOS) stays suppressed
        // regardless of allow_eos.
        assert_ne!(picked, (vocab - 1) as u32);
    }

    #[test]
    fn sampling_path_also_respects_the_suppressed_range() {
        // temperature > 0 (the real default path): run many draws and
        // confirm none ever lands in the suppressed range (other than EOS,
        // which is excluded here so every draw must avoid the whole range).
        let vocab = 2048usize;
        let eos = (vocab - 1) as u32;
        let mut logits = vec![0.0f32; vocab];
        for x in logits[vocab - 1024..].iter_mut() {
            *x = 1000.0; // would dominate sampling if not masked
        }
        let mut rng = Rng::new(7);
        for _ in 0..200 {
            let picked = sample_cb0(logits.clone(), eos, false, 0.9, 50, &mut rng);
            assert!((picked as usize) < vocab - 1024, "picked {picked} is in the suppressed range");
        }
    }
}
