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
use model::anchor_memory::{AnchorBankPool, AnchorMemoryShape};
use model::kv_tier::{KvLayer, KvTier};
use model::Shard;

use crate::config::{LayerType, Qwen35Config};

/// Per-layer decode state, each vector indexed by absolute layer index.
pub(crate) struct DecodeState {
    /// Per-layer plain (non-paged) `f32` KV cache for GQA layers, `[dec_cap,
    /// kv_dim]`; a size-1 placeholder at GDN layer indices. A sliding-window
    /// config holds the ring instead: `[window, kv_dim]`, whatever `dec_cap` is.
    pub gqa_kv: Vec<KvLayer>,
    /// Per-layer persistent Gated DeltaNet recurrent state, `[bh, dk, dv]`,
    /// for GDN layers; a size-1 dummy at GQA layer indices. Threaded across
    /// `step` calls; zeroed by [`crate::model::Qwen35::reset_decode_cache`].
    pub gdn_state: Vec<DeviceBuffer>,
    /// Per-layer persistent causal-conv history ring buffer, `[1, conv_dim,
    /// K-1]`, for GDN layers; a size-1 dummy at GQA layer indices.
    pub gdn_hist: Vec<DeviceBuffer>,
    /// Per-layer anchor-memory bank (one sequence slot) at the layers that carry
    /// one; empty for a config without an anchor memory.
    pub memory: Vec<Option<AnchorBankPool>>,
    /// Every persistent buffer with its size in words, for [`Self::bytes`].
    ledger: Vec<(DeviceBuffer, u64)>,
}

impl DecodeState {
    /// A layer this shard does not own never runs its decode step
    /// (`run_decode_step` loops only `shard.start..shard.end`), so its
    /// cache/state/history buffers are dead weight - a full-size GQA KV cache
    /// or GDN state per un-owned layer, on every non-owning shard.
    pub(crate) fn new(gpu: &Gpu, cfg: &Qwen35Config, shard: &Shard, t: u32) -> DecodeState {
        let kv_dim = cfg.kv_dim() as u64;
        let rows = u64::from(cfg.swa_window.unwrap_or(t));
        let mut st = DecodeState { gqa_kv: Vec::new(), gdn_state: Vec::new(), gdn_hist: Vec::new(), memory: Vec::new(), ledger: Vec::new() };
        for (l, ty) in cfg.layer_types().into_iter().enumerate() {
            let owned = shard.owns(l);
            let (kv, state, hist) = match (owned, ty) {
                (true, LayerType::Full) => (KvLayer::new(gpu, KvTier::F32, rows, kv_dim, cfg.head_dim as u64), gpu.storage(1), gpu.storage(1)),
                (true, LayerType::Linear) => (KvLayer::placeholder(gpu), gpu.storage(cfg.gdn_state_len()), gpu.storage(cfg.gdn_hist_len())),
                (false, _) => (KvLayer::placeholder(gpu), gpu.storage(1), gpu.storage(1)),
            };
            if owned {
                match ty {
                    LayerType::Full => {
                        st.ledger.push((kv.k.data().clone(), rows * kv_dim));
                        st.ledger.push((kv.v.data().clone(), rows * kv_dim));
                    }
                    LayerType::Linear => {
                        st.ledger.push((state.clone(), cfg.gdn_state_len()));
                        st.ledger.push((hist.clone(), cfg.gdn_hist_len()));
                    }
                }
            }
            let pool = (owned && cfg.layer_has_memory(l)).then(|| {
                let pool = AnchorBankPool::new(gpu, &memory_geometry(cfg), 1);
                st.ledger.extend(pool.arrays().iter().map(|(buf, per_slot)| ((*buf).clone(), *per_slot)));
                pool
            });
            st.memory.push(pool);
            st.gqa_kv.push(kv);
            st.gdn_state.push(state);
            st.gdn_hist.push(hist);
        }
        st
    }

    /// Bytes this state holds, summed over the buffers it allocated: the per-sequence state of the model, however
    /// many tokens the sequence has seen.
    pub(crate) fn bytes(&self) -> u64 {
        self.ledger.iter().map(|(_, words)| words * 4).sum()
    }

    /// Identity and size of every persistent buffer: a run that grows nothing leaves this list unchanged.
    pub(crate) fn allocations(&self) -> Vec<(usize, u64)> {
        self.ledger.iter().map(|(buf, words)| (buf.alloc_id() as usize, *words)).collect()
    }
}

/// The geometry an anchor-memory pool of `cfg` is built from.
pub(crate) fn memory_geometry(cfg: &Qwen35Config) -> AnchorMemoryShape {
    let mem = cfg.anchor_memory.as_ref().expect("a memory pool is built only for a config with an anchor memory");
    AnchorMemoryShape {
        n_seq: 1,
        t: 1,
        d_model: cfg.d_model,
        n_heads: cfg.n_heads,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        rotary_half: cfg.rotary_dim() / 2,
        rms_eps: cfg.rms_eps,
        capacity: mem.capacity,
        commit_block: cfg.memory_commit_block().expect("a memory has a commit block"),
        admission_rank: mem.admission_rank,
        writer_intermediate: mem.writer_intermediate,
    }
}

/// The fixed-size state of `slots` sequences of a sliding-window config, one pool row per sequence slot: the ring KV
/// of every full-attention layer (`slots * window` rows, sequence `i`'s ring is rows `i * window ..`), each
/// sequence's Gated-DeltaNet state and conv history, and the anchor-memory bank pools. Nothing here grows with a
/// sequence's length, and every operation a serving engine needs per sequence is addressed by slot row:
/// [`crate::model::Qwen35::slot_prefill`], [`crate::model::Qwen35::slot_decode`] (any mix of slots in one batch),
/// [`crate::model::Qwen35::slot_reset`] and [`crate::model::Qwen35::slot_copy`] (what a snapshot prefix cache copies).
pub struct SlotPool {
    pub(crate) slots: u32,
    pub(crate) window: u32,
    pub(crate) gqa_kv: Vec<KvLayer>,
    /// `[slot][layer]`
    pub(crate) gdn_state: Vec<Vec<DeviceBuffer>>,
    pub(crate) gdn_hist: Vec<Vec<DeviceBuffer>>,
    pub(crate) memory: Vec<Option<AnchorBankPool>>,
    /// Tokens each slot has absorbed.
    pub(crate) pos: std::cell::RefCell<Vec<u32>>,
    ledger: Vec<(DeviceBuffer, u64)>,
}

impl SlotPool {
    pub(crate) fn new(gpu: &Gpu, cfg: &Qwen35Config, shard: &Shard, slots: u32) -> SlotPool {
        let window = cfg.swa_window.expect("a slot pool holds the ring state of a sliding-window config");
        let kv_dim = cfg.kv_dim() as u64;
        let rows = u64::from(window) * u64::from(slots);
        let mut pool = SlotPool { slots, window, gqa_kv: Vec::new(), gdn_state: vec![Vec::new(); slots as usize], gdn_hist: vec![Vec::new(); slots as usize], memory: Vec::new(), pos: std::cell::RefCell::new(vec![0; slots as usize]), ledger: Vec::new() };
        for (l, ty) in cfg.layer_types().into_iter().enumerate() {
            let owned = shard.owns(l);
            let kv = if owned && ty == LayerType::Full {
                let kv = KvLayer::new(gpu, KvTier::F32, rows, kv_dim, cfg.head_dim as u64);
                pool.ledger.push((kv.k.data().clone(), rows * kv_dim));
                pool.ledger.push((kv.v.data().clone(), rows * kv_dim));
                kv
            } else {
                KvLayer::placeholder(gpu)
            };
            pool.gqa_kv.push(kv);
            for slot in 0..slots as usize {
                let (state, hist) = if owned && ty == LayerType::Linear {
                    let (state, hist) = (gpu.storage(cfg.gdn_state_len()), gpu.storage(cfg.gdn_hist_len()));
                    pool.ledger.push((state.clone(), cfg.gdn_state_len()));
                    pool.ledger.push((hist.clone(), cfg.gdn_hist_len()));
                    (state, hist)
                } else {
                    (gpu.storage(1), gpu.storage(1))
                };
                pool.gdn_state[slot].push(state);
                pool.gdn_hist[slot].push(hist);
            }
            let bank = (owned && cfg.layer_has_memory(l)).then(|| {
                let bank = AnchorBankPool::new(gpu, &memory_geometry(cfg), slots);
                pool.ledger.extend(bank.arrays().iter().map(|(buf, per_slot)| ((*buf).clone(), *per_slot * u64::from(slots))));
                bank
            });
            pool.memory.push(bank);
        }
        pool
    }

    /// Number of sequence slots.
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Tokens slot `slot` has absorbed.
    pub fn position(&self, slot: u32) -> u32 {
        self.pos.borrow()[slot as usize]
    }

    /// Bytes the pool holds, summed over the buffers it allocated: independent of every sequence's length.
    pub fn bytes(&self) -> u64 {
        self.ledger.iter().map(|(_, words)| words * 4).sum()
    }

    /// Identity and size of every buffer: unchanged by any amount of decoding.
    pub fn allocations(&self) -> Vec<(usize, u64)> {
        self.ledger.iter().map(|(buf, words)| (buf.alloc_id() as usize, *words)).collect()
    }
}
