// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Greedy text generation over the Thinker decoder, streaming each layer's
//! weights from a checkpoint one at a time instead of holding all 48 layers
//! (128 experts each) GPU-resident — the real model does not fit that way on
//! this box without int8 quantization + cross-GPU sharding (`docs/models/
//! omni/status.md` M1), which is a separate, not-yet-built piece of
//! production residency (`crates/qwen/src/shard.rs`'s int8 path is the
//! precedent). This module is deliberately the validation-tier: correct,
//! not fast — every layer's weights are re-read from the mmap and re-run for
//! every generated token (no KV-cache), so a real 48-layer decode is minutes,
//! not milliseconds, per token. `docs/models/omni/status.md`'s M9 entry
//! records this as the explicit scope boundary: this proves the generation
//! LOOP (tokenizer round-trip, sampling, EOS, layer chaining) is correct
//! against real weights; production serving speed is later work.
//!
//! `layer_fwd`/`decode`/`final_norm`/`lm_head_fwd`'s own per-call math is
//! already validated exactly (cosine 1.000000, M6a/M6b) — this module's own
//! risk surface is purely the loop control this docstring's first paragraph
//! describes, not the per-layer forward.

use checkpoint::weightio::WeightReader;
use gpu_core::{DeviceBuffer, Gpu};
use qwenvl::mrope::{get_rope_index, mrope_tables};

use crate::config::MoeTextConfig;
use crate::thinker::{final_norm, layer_fwd, lm_head_fwd, ThinkerLayerWeights};

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

/// One full Thinker forward pass (all `cfg.n_layers` layers + final norm),
/// streaming each layer's weights from `reader` and dropping them before the
/// next layer loads. `x_host` is an already-embedded `[n, hidden]` sequence
/// (host-side gather from `thinker.model.embed_tokens.weight`, the same
/// pattern every real-weight test in this crate uses).
pub fn thinker_forward_streaming(reader: &WeightReader, gpu: &Gpu, cfg: &MoeTextConfig, x_host: &[f32], n: u32) -> DeviceBuffer {
    let tokens: Vec<u32> = (0..n).collect(); // caller's position ids are always plain-sequential here (see generate_greedy's doc)
    let positions = get_rope_index(&tokens, u32::MAX, &[]);
    let section: [u32; 3] = [cfg.mrope_section[0], cfg.mrope_section[1], cfg.mrope_section[2]];
    let (cos_tab, sin_tab) = mrope_tables(&positions, section, cfg.head_dim, cfg.rope_theta);
    let cos = gpu.storage_init("cos", &cos_tab);
    let sin = gpu.storage_init("sin", &sin_tab);

    let mut h = gpu.storage_init("x", x_host);
    for l in 0..cfg.n_layers {
        let layer = load_thinker_layer(reader, gpu, l, cfg.n_experts);
        let (out, ..) = layer_fwd(gpu, cfg, &layer.as_weights(), &h, &cos, &sin, n);
        h = out;
    }
    let norm_w = gpu.storage_init("w", &reader.tensor("thinker.model.norm.weight").expect("missing thinker.model.norm.weight"));
    final_norm(gpu, cfg, &norm_w, &h, n)
}

/// Greedy (argmax) text generation: embeds `prompt_ids`, runs the streaming
/// forward, appends the highest-logit next token, and repeats until
/// `max_new_tokens` or an id in `eos_ids` is produced. Returns the FULL
/// sequence (prompt + generated). Positions are plain-sequential `0..n`
/// throughout — the pure-text M-RoPE-collapse case (see `crate::thinker`'s
/// module doc); a caller with an image/audio/video span needs the real
/// per-axis positions wired in separately, not done here.
///
/// This is the validation-tier, no-KV-cache loop this module's doc describes:
/// step `k` re-runs the FULL `k`-token forward from scratch, re-streaming
/// every layer's weights from `reader`. Correct; not the shape a production
/// decode loop should take.
pub fn generate_greedy(reader: &WeightReader, gpu: &Gpu, cfg: &MoeTextConfig, embed_table: &[f32], lm_head_w: &[f32], prompt_ids: &[u32], max_new_tokens: u32, eos_ids: &[u32]) -> Vec<u32> {
    let d = cfg.hidden as usize;
    let mut ids: Vec<u32> = prompt_ids.to_vec();
    let lm_head = gpu.storage_init("lm_head", lm_head_w);

    for _ in 0..max_new_tokens {
        let n = ids.len() as u32;
        let mut x_host = Vec::with_capacity(ids.len() * d);
        for &t in &ids {
            let row = &embed_table[t as usize * d..(t as usize + 1) * d];
            x_host.extend_from_slice(row);
        }
        let hidden = thinker_forward_streaming(reader, gpu, cfg, &x_host, n);
        let logits = lm_head_fwd(gpu, &lm_head, &hidden, n, cfg.hidden, cfg.vocab);
        let last_row = gpu.read(&logits, (n * cfg.vocab) as usize);
        let last_row = &last_row[((n - 1) * cfg.vocab) as usize..(n * cfg.vocab) as usize];
        let next = last_row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).expect("non-empty vocab");
        ids.push(next);
        if eos_ids.contains(&next) {
            break;
        }
    }
    ids
}
