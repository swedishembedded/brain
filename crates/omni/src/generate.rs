// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Greedy text generation over the Thinker decoder, streaming each layer's
//! weights from a checkpoint one at a time instead of holding all 48 layers
//! (128 experts each) GPU-resident — the real model does not fit that way on
//! this box without int8 quantization + cross-GPU sharding (`docs/models/
//! omni/status.md` M1), which is a separate, not-yet-built piece of
//! production residency (`crates/qwen/src/shard.rs`'s int8 path is the
//! precedent). Weight I/O is still the validation-tier: every layer's weights
//! are re-read from the mmap for every generated token (no resident weights),
//! so a real 48-layer decode is still minutes, not milliseconds, per token —
//! `docs/models/omni/status.md`'s M9 entry records this half of the scope
//! boundary and it remains true.
//!
//! **KV-cache**: the ATTENTION math is no longer the validation-tier's O(T²)
//! full recompute. [`generate_greedy`] now [`prefill`]s the prompt once
//! (one pass through each layer, same as before, but ALSO bulk-fills that
//! layer's persistent KV cache — `model::block::kv_cache_fill`, wired through
//! `thinker::layer_fwd`'s `cache` param) and then [`decode_step`]s one new
//! token at a time, attending only against the growing cache
//! (`model::block::gqa_decode_step`, `thinker::layer_decode_step`) — O(cached
//! length) per step, not O(cached length)². Every layer's weights are still
//! reloaded from `reader` on every step (prefill: once per layer; decode:
//! once per layer per generated token) — the cache changes the ATTENTION
//! complexity, not the weight-I/O pattern; full GPU-resident weights across
//! steps is the same not-yet-built M1 work referenced above.
//!
//! `layer_fwd`/`layer_decode_step`/`decode`/`final_norm`/`lm_head_fwd`'s own
//! per-call math is already validated exactly (cosine 1.000000, M6a/M6b; the
//! KV-cache primitive itself by `model::block`'s own
//! `decode_step_matches_causal_batched_attention_at_every_position` test) —
//! this module's own risk surface is purely the loop control (prefill/decode
//! sequencing, cache sizing, sampling), not the per-layer forward.

use checkpoint::weightio::WeightReader;
use gpu_core::{DeviceBuffer, Gpu};
use qwenvl::mrope::{get_rope_index, mrope_tables};

use crate::config::MoeTextConfig;
use crate::thinker::{final_norm, layer_decode_step, layer_fwd, lm_head_fwd, ThinkerLayerCache, ThinkerLayerWeights};

/// One decoder layer's weights, freshly uploaded from `reader` — dropped
/// (freeing the GPU buffers) when the caller's borrow of [`OwnedLayer::as_weights`]
/// goes out of scope, i.e. after that one [`layer_fwd`] call.
struct OwnedLayer {
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
}

impl OwnedLayer {
    fn as_weights(&self) -> ThinkerLayerWeights<'_> {
        ThinkerLayerWeights {
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
        }
    }
}

fn load_thinker_layer(reader: &WeightReader, gpu: &Gpu, l: u32, n_experts: u32) -> OwnedLayer {
    let p = |leaf: &str| format!("thinker.model.layers.{l}.{leaf}");
    let get = |name: &str| gpu.storage_init("w", &reader.tensor(name).unwrap_or_else(|| panic!("missing tensor {name}")));
    OwnedLayer {
        ln1: get(&p("input_layernorm.weight")),
        wq: get(&p("self_attn.q_proj.weight")),
        wk: get(&p("self_attn.k_proj.weight")),
        wv: get(&p("self_attn.v_proj.weight")),
        wo: get(&p("self_attn.o_proj.weight")),
        q_norm: get(&p("self_attn.q_norm.weight")),
        k_norm: get(&p("self_attn.k_norm.weight")),
        ln2: get(&p("post_attention_layernorm.weight")),
        router: get(&p("mlp.gate.weight")),
        experts: (0..n_experts)
            .map(|e| {
                (
                    get(&p(&format!("mlp.experts.{e}.gate_proj.weight"))),
                    get(&p(&format!("mlp.experts.{e}.up_proj.weight"))),
                    get(&p(&format!("mlp.experts.{e}.down_proj.weight"))),
                )
            })
            .collect(),
    }
}

/// One layer's persistent incremental-decode KV cache buffers, sized once for
/// the whole generation (`cap = prompt length + max_new_tokens`) — [`prefill`]
/// fills rows `0..prompt_len`, [`decode_step`] extends one row per call.
struct ThinkerKvCache {
    layers: Vec<(DeviceBuffer, DeviceBuffer)>,
    cap: u32,
}

impl ThinkerKvCache {
    fn new(gpu: &Gpu, cfg: &MoeTextConfig, cap: u32) -> Self {
        let hkv = (cfg.n_kv_heads * cfg.head_dim) as u64;
        let layers = (0..cfg.n_layers).map(|_| (gpu.storage(cap as u64 * hkv), gpu.storage(cap as u64 * hkv))).collect();
        Self { layers, cap }
    }
    fn layer(&self, l: usize) -> ThinkerLayerCache<'_> {
        ThinkerLayerCache { kcache: &self.layers[l].0, vcache: &self.layers[l].1 }
    }
}

/// Prefill: the whole prompt `x_host [n, hidden]` through every layer ONCE
/// (batched causal attention, same math `thinker::decode` uses), bulk-filling
/// `cache`'s rows `0..n` as a side effect of each layer's own forward
/// (`thinker::layer_fwd`'s `cache` param) — after this, [`decode_step`]
/// continues from `pos = n` without recomputing anything this pass already
/// did. Returns the final-normed hidden state `[n, hidden]`. `positions` is
/// the real per-token 3-axis M-RoPE position (`qwenvl::mrope::get_rope_index`/
/// `get_rope_index_multi`) — plain-sequential for pure text, real per-axis
/// values wherever a caller (`crate::mm`) spliced in media.
fn prefill(reader: &WeightReader, gpu: &Gpu, cfg: &MoeTextConfig, x_host: &[f32], positions: &[[u32; 3]], n: u32, cache: &ThinkerKvCache) -> DeviceBuffer {
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(positions, section, cfg.head_dim, cfg.rope_theta);
    let cos = gpu.storage_init("cos", &cos_tab);
    let sin = gpu.storage_init("sin", &sin_tab);

    let mut h = gpu.storage_init("x", x_host);
    for l in 0..cfg.n_layers {
        let layer = load_thinker_layer(reader, gpu, l, cfg.n_experts);
        let lc = cache.layer(l as usize);
        let (out, ..) = layer_fwd(gpu, cfg, &layer.as_weights(), &h, &cos, &sin, n, Some(&lc));
        h = out;
    }
    let norm_w = gpu.storage_init("w", &reader.tensor("thinker.model.norm.weight").expect("missing thinker.model.norm.weight"));
    final_norm(gpu, cfg, &norm_w, &h, n)
}

/// One incremental decode step: a single new token's embedding row `x_host`
/// (`[hidden]`) through every layer, attending against `cache` at cache row
/// `cache_row` (`thinker::layer_decode_step` — O(cache_row), not
/// O(cache_row)²) and RoPE'd at the real 3-axis position `mrope_pos` — these
/// two are DELIBERATELY separate: `cache_row` is always the plain append
/// index (0,1,2,…, one per generated token), while `mrope_pos` can be
/// non-monotonic per axis if a media block appeared earlier in the prompt
/// (`qwenvl::mrope::get_rope_index_multi`'s doc). Layer weights are still
/// reloaded fresh from `reader` every call (see the module doc). Returns the
/// final-normed hidden row `[1, hidden]`.
fn decode_step(reader: &WeightReader, gpu: &Gpu, cfg: &MoeTextConfig, x_host: &[f32], mrope_pos: [u32; 3], cache_row: u32, cache: &ThinkerKvCache) -> DeviceBuffer {
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(&[mrope_pos], section, cfg.head_dim, cfg.rope_theta);
    let cos = gpu.storage_init("cos", &cos_tab);
    let sin = gpu.storage_init("sin", &sin_tab);

    let mut h = gpu.storage_init("x", x_host);
    for l in 0..cfg.n_layers {
        let layer = load_thinker_layer(reader, gpu, l, cfg.n_experts);
        let lc = cache.layer(l as usize);
        h = layer_decode_step(gpu, cfg, &layer.as_weights(), &lc, &h, &cos, &sin, cache_row, cache.cap);
    }
    let norm_w = gpu.storage_init("w", &reader.tensor("thinker.model.norm.weight").expect("missing thinker.model.norm.weight"));
    final_norm(gpu, cfg, &norm_w, &h, 1)
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).expect("non-empty vocab")
}

/// Opt-in (`BRAIN_OMNI_DEBUG_LOGITS=1`) top-3 logit dump for one decode step
/// — diagnoses exactly the failure mode `crates/omni/tests/generate_e2e.rs`
/// found on the real checkpoint: a reference comparison (HF's bf16 compute
/// vs. this engine's fp32) diverging at a token whose top candidates are
/// closely spaced, distinguishing "a near-tied logit flipped by accumulated
/// rounding" (small margin between the top few candidates, the wanted token
/// still nearby) from "an actual bug" (a wildly wrong, confidently-argmaxed
/// token). Costs nothing when unset.
fn debug_log_top_candidates(cache_row: u32, logits: &[f32]) {
    if std::env::var("BRAIN_OMNI_DEBUG_LOGITS").is_err() {
        return;
    }
    let mut sorted: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!("decode step cache_row={cache_row}: top3 (token_id, logit) = {:?}", &sorted[..3.min(sorted.len())]);
}

/// Greedy (argmax) text generation: [`prefill`]s `prompt_ids` once (populating
/// a KV cache sized `prompt_ids.len() + max_new_tokens`), samples the first
/// new token from the prefill's last logit row, then [`decode_step`]s one
/// token at a time — appending each highest-logit next token and repeating
/// until `max_new_tokens` or an id in `eos_ids` is produced. Returns the FULL
/// sequence (prompt + generated). Positions are plain-sequential throughout —
/// the pure-text M-RoPE-collapse case (see `crate::thinker`'s module doc); a
/// caller with an image/audio/video span needs the real per-axis positions
/// wired in separately, not done here.
pub fn generate_greedy(reader: &WeightReader, gpu: &Gpu, cfg: &MoeTextConfig, embed_table: &[f32], lm_head_w: &[f32], prompt_ids: &[u32], max_new_tokens: u32, eos_ids: &[u32]) -> Vec<u32> {
    if prompt_ids.is_empty() {
        return prompt_ids.to_vec();
    }
    let positions = get_rope_index(prompt_ids, u32::MAX, &[]); // plain-sequential: no placeholder token in prompt_ids ever matches u32::MAX
    generate_greedy_positioned(reader, gpu, cfg, embed_table, lm_head_w, prompt_ids, &positions, max_new_tokens, eos_ids)
}

/// [`generate_greedy`], generalized to a caller-supplied embedding buffer and
/// real per-token M-RoPE positions — the entry `crate::mm::build_multimodal_prompt`
/// feeds once a prompt has media spliced into it. `prompt_ids`/`positions` must
/// be the same length; `x_host` is `[prompt_ids.len(), hidden]` and, for a
/// multimodal prompt, already has the media rows overwritten
/// (`crate::mm::splice_host`) — this function does no splicing itself, only
/// the prefill/decode/sample loop. New tokens generated beyond the prompt are
/// always plain text (no further media insertion mid-generation), so their
/// positions continue the diagonal from the prompt's last position + 1 on
/// every axis, exactly like a trailing text run in `get_rope_index`'s own
/// algorithm.
#[allow(clippy::too_many_arguments)]
pub fn generate_greedy_positioned(
    reader: &WeightReader,
    gpu: &Gpu,
    cfg: &MoeTextConfig,
    embed_table: &[f32],
    lm_head_w: &[f32],
    prompt_ids: &[u32],
    positions: &[[u32; 3]],
    max_new_tokens: u32,
    eos_ids: &[u32],
) -> Vec<u32> {
    generate_greedy_with_embeds(reader, gpu, cfg, embed_table, lm_head_w, prompt_ids, None, positions, max_new_tokens, eos_ids)
}

/// The shared implementation behind [`generate_greedy`]/[`generate_greedy_positioned`]
/// and `crate::mm`'s multimodal entry: `x_host_override`, when `Some`, is used
/// as the prompt's embedding buffer verbatim (already spliced with media);
/// when `None`, it's built by a plain per-token gather from `embed_table` (the
/// pure-text case both public wrappers above use).
#[allow(clippy::too_many_arguments)]
fn generate_greedy_with_embeds(
    reader: &WeightReader,
    gpu: &Gpu,
    cfg: &MoeTextConfig,
    embed_table: &[f32],
    lm_head_w: &[f32],
    prompt_ids: &[u32],
    x_host_override: Option<Vec<f32>>,
    positions: &[[u32; 3]],
    max_new_tokens: u32,
    eos_ids: &[u32],
) -> Vec<u32> {
    let d = cfg.hidden as usize;
    let mut ids: Vec<u32> = prompt_ids.to_vec();
    assert_eq!(positions.len(), prompt_ids.len(), "generate_greedy: positions/prompt_ids length mismatch");
    if max_new_tokens == 0 || prompt_ids.is_empty() {
        return ids;
    }
    let lm_head = gpu.storage_init("lm_head", lm_head_w);
    let embed_row = |t: u32| embed_table[t as usize * d..(t as usize + 1) * d].to_vec();

    let n0 = prompt_ids.len() as u32;
    let cap = n0 + max_new_tokens;
    let cache = ThinkerKvCache::new(gpu, cfg, cap);

    let x_host = x_host_override.unwrap_or_else(|| {
        let mut x = Vec::with_capacity(prompt_ids.len() * d);
        for &t in prompt_ids {
            x.extend_from_slice(&embed_row(t));
        }
        x
    });
    let hidden = prefill(reader, gpu, cfg, &x_host, positions, n0, &cache);
    let logits = lm_head_fwd(gpu, &lm_head, &hidden, n0, cfg.hidden, cfg.vocab);
    let last_row = gpu.read(&logits, (n0 * cfg.vocab) as usize);
    let mut next = argmax(&last_row[((n0 - 1) * cfg.vocab) as usize..(n0 * cfg.vocab) as usize]);
    ids.push(next);
    let mut cache_row = n0;
    // New tokens are always plain text: continue the diagonal from the
    // prompt's last position + 1 on every axis (see this function's doc).
    let mut mrope_pos = positions[positions.len() - 1].map(|p| p + 1);

    if !eos_ids.contains(&next) {
        for _ in 1..max_new_tokens {
            let x_row = embed_row(next);
            let hidden = decode_step(reader, gpu, cfg, &x_row, mrope_pos, cache_row, &cache);
            let logits = lm_head_fwd(gpu, &lm_head, &hidden, 1, cfg.hidden, cfg.vocab);
            let row = gpu.read(&logits, cfg.vocab as usize);
            next = argmax(&row);
            debug_log_top_candidates(cache_row, &row);
            ids.push(next);
            cache_row += 1;
            mrope_pos = mrope_pos.map(|p| p + 1);
            if eos_ids.contains(&next) {
                break;
            }
        }
    }
    ids
}

/// The multimodal entry: `prompt.x_host` already has media spliced in
/// (`crate::mm::build_multimodal_prompt`), so no `embed_table` gather is
/// needed for the prompt itself — only for tokens generated AFTER it (always
/// plain text, per [`generate_greedy_positioned`]'s doc).
pub fn generate_greedy_multimodal(reader: &WeightReader, gpu: &Gpu, cfg: &MoeTextConfig, embed_table: &[f32], lm_head_w: &[f32], prompt: &crate::mm::MultimodalPrompt, max_new_tokens: u32, eos_ids: &[u32]) -> Vec<u32> {
    generate_greedy_with_embeds(reader, gpu, cfg, embed_table, lm_head_w, &prompt.token_ids, Some(prompt.x_host.clone()), &prompt.positions, max_new_tokens, eos_ids)
}
