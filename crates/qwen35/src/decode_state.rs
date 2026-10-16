// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded LoRA/QLoRA fine-tuning and
// low-latency serving of large language models on a single GPU for its
// clients. If your team needs expertise in sizing model state against a fixed
// memory budget then you can procure our services by sending an email to
// info@swedishembedded.com.

//! The single-sequence decode caches of a [`crate::model::Qwen35`]: what
//! `step` and the chunked prefill read and update between calls. Built with an
//! inference instance, and on first use by a training instance (see
//! `Qwen35::decode`).

use gpu_core::{DeviceBuffer, Gpu};
use model::kv_tier::{KvLayer, KvTier};
use model::Shard;

use crate::config::{LayerType, Qwen35Config};

/// Per-layer decode state, each vector indexed by absolute layer index.
pub(crate) struct DecodeState {
    /// Per-layer plain (non-paged) `f32` KV cache for GQA layers, `[dec_cap,
    /// kv_dim]`; a size-1 placeholder at GDN layer indices.
    pub gqa_kv: Vec<KvLayer>,
    /// Per-layer persistent Gated DeltaNet recurrent state, `[bh, dk, dv]`,
    /// for GDN layers; a size-1 dummy at GQA layer indices. Threaded across
    /// `step` calls; zeroed by [`crate::model::Qwen35::reset_decode_cache`].
    pub gdn_state: Vec<DeviceBuffer>,
    /// Per-layer persistent causal-conv history ring buffer, `[1, conv_dim,
    /// K-1]`, for GDN layers; a size-1 dummy at GQA layer indices.
    pub gdn_hist: Vec<DeviceBuffer>,
}

impl DecodeState {
    /// A layer this shard does not own never runs its decode step
    /// (`run_decode_step` loops only `shard.start..shard.end`), so its
    /// cache/state/history buffers are dead weight - a full-size GQA KV cache
    /// or GDN state per un-owned layer, on every non-owning shard.
    pub(crate) fn new(gpu: &Gpu, cfg: &Qwen35Config, shard: &Shard, t: u32) -> DecodeState {
        let kv_dim = cfg.kv_dim() as u64;
        let mut st = DecodeState { gqa_kv: Vec::new(), gdn_state: Vec::new(), gdn_hist: Vec::new() };
        for (l, ty) in cfg.layer_types().into_iter().enumerate() {
            let (kv, state, hist) = match (shard.owns(l), ty) {
                (true, LayerType::Full) => (KvLayer::new(gpu, KvTier::F32, t as u64, kv_dim, cfg.head_dim as u64), gpu.storage(1), gpu.storage(1)),
                (true, LayerType::Linear) => (KvLayer::placeholder(gpu), gpu.storage(cfg.gdn_state_len()), gpu.storage(cfg.gdn_hist_len())),
                (false, _) => (KvLayer::placeholder(gpu), gpu.storage(1), gpu.storage(1)),
            };
            st.gqa_kv.push(kv);
            st.gdn_state.push(state);
            st.gdn_hist.push(hist);
        }
        st
    }
}
