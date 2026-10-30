// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speculative decoding - drafters, verifiers
// and the state rollback a recurrent stack needs - for clients serving large
// models on modest hardware. If your team needs expertise in verifying several
// tokens per sequence through a hybrid Gated DeltaNet decoder, then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The speculative **verify** pass of a Qwen3.8 stage: the batched decode
//! forward ([`Qwen35::run_decode_batch`]) over several consecutive tokens per
//! sequence instead of one.
//!
//! The batch is *packed*: sequence `i` owns `counts[i]` consecutive rows, laid
//! out sequence after sequence, and row `r` of sequence `i` sits at absolute
//! position `positions[i] + r`. Full-attention layers need nothing special - a
//! row attends to its own prefix and the rows are written before any is read,
//! the same way a chunked prefill works. The Gated DeltaNet layers do: a
//! sequence's rows must be taken in order from its pooled state, and the state
//! between them kept so a rejected tail can be undone ([`VerifyBatch`]).

use gpu_core::{DeviceBuffer, Gpu};
use model::gdn_verify::{GdnRowSnapshots, MAX_VERIFY_ROWS};

use super::{BatchDecodeCaches, Qwen35};
use crate::stage_rows::StageRows;

/// The scratch arena a verify step's layer stack draws from. Its own: a verify
/// step asks for a different sequence of buffers than a decode step does. One
/// for every row count, deliberately: an arena keeps the buffers it handed out,
/// so an arena per row count held the intermediates of every row count the
/// windows had ever produced (out of memory at 128 rows), against a few percent
/// of step time for replaying a recorded pass at a repeated row count.
pub(crate) const VERIFY_ARENA: &str = "qwen35.verify";

/// Whether the verify kernels serve this model's Gated DeltaNet layer shape -
/// the half of [`Qwen35::verify_supported`] a plan can ask before a device exists.
pub(crate) fn verify_shape_supported(c: &crate::Qwen35Config) -> bool {
    c.linear_key_head_dim == 128
        && c.linear_value_head_dim == 128
        && c.linear_conv_kernel_dim == 4
        && c.linear_num_value_heads.is_multiple_of(c.linear_num_key_heads)
        && (1..=3).contains(&(c.linear_num_value_heads / c.linear_num_key_heads))
}

/// What a verify pass needs beyond a decode pass: the rows each sequence
/// carries and, for every Gated DeltaNet layer, somewhere to keep the state
/// between them.
pub(crate) struct VerifyBatch<'a> {
    /// Pool row (slot) of each sequence.
    slots: &'a [u32],
    /// Rows of each sequence, `1..=MAX_VERIFY_ROWS`.
    counts: &'a [u32],
    /// `[layer]`: row snapshots of a Gated DeltaNet layer this stage owns (a
    /// placeholder everywhere else), with room for every packed row.
    snaps: &'a [GdnRowSnapshots],
}

/// [`VerifyBatch`]'s per-sequence layout on the device.
pub(crate) struct VerifyLayout {
    pub rows: DeviceBuffer,
    pub counts: DeviceBuffer,
    pub offs: DeviceBuffer,
}

impl<'a> VerifyBatch<'a> {
    pub(crate) fn new(slots: &'a [u32], counts: &'a [u32], snaps: &'a [GdnRowSnapshots]) -> VerifyBatch<'a> {
        assert_eq!(slots.len(), counts.len(), "qwen35 verify: one slot per sequence");
        assert!(!slots.is_empty(), "qwen35 verify: an empty batch");
        assert!(counts.iter().all(|c| (1..=MAX_VERIFY_ROWS).contains(c)), "qwen35 verify: a sequence carries 1..={MAX_VERIFY_ROWS} rows, got {counts:?}");
        VerifyBatch { slots, counts, snaps }
    }

    pub(crate) fn sequences(&self) -> u32 {
        self.slots.len() as u32
    }

    /// Rows of all the sequences together.
    pub(crate) fn total_rows(&self) -> u32 {
        self.counts.iter().sum()
    }

    pub(crate) fn snapshots(&self, layer: usize) -> &'a GdnRowSnapshots {
        &self.snaps[layer]
    }

    /// First packed row of each sequence.
    pub(crate) fn offsets(&self) -> Vec<u32> {
        self.counts
            .iter()
            .scan(0u32, |acc, &c| {
                let o = *acc;
                *acc += c;
                Some(o)
            })
            .collect()
    }

    /// Rewrite the buffers of a layout uploaded earlier ([`Self::upload`]) for this batch, which has the same row counts
    /// as the batch the layout was made for: a recorded round is replayed with the sequences it is handed.
    pub(crate) fn write_layout(&self, g: &Gpu, layout: &VerifyLayout) {
        g.write(&layout.rows, self.slots);
        g.write(&layout.counts, self.counts);
        g.write(&layout.offs, &self.offsets());
    }

    /// Upload the layout to `g`. Called before the pass's first dispatch.
    pub(crate) fn upload(&self, g: &Gpu) -> VerifyLayout {
        let put = |v: &[u32]| {
            let b = g.storage(v.len() as u64);
            g.write(&b, v);
            b
        };
        VerifyLayout { rows: put(self.slots), counts: put(self.counts), offs: put(&self.offsets()) }
    }
}

impl Qwen35 {
    /// [`Self::run_decode_batch`] as a speculative verify pass: the packed `[rows, d_model]` hidden block,
    /// left on the device for the next stage.
    pub(crate) fn verify_batch(&self, tokens: &[u32], caches: &BatchDecodeCaches, verify: &VerifyBatch, input_override: Option<StageRows<'_>>) -> gpu_core::DeviceBuffer {
        assert_eq!(tokens.len() as u32, verify.counts.iter().sum::<u32>(), "qwen35 verify: {} tokens for {:?} rows", tokens.len(), verify.counts);
        self.run_decode_batch_with(tokens, caches, input_override, Some(verify))
    }

    /// Copy, for every sequence that rejected some of its rows, the snapshot of
    /// its last accepted row back over its pooled recurrent state, in every
    /// Gated DeltaNet layer this stage owns. `keep[i]` is how many of
    /// sequence `i`'s rows stand; the state after a sequence's last row is the
    /// pool's already.
    pub(crate) fn commit_verify_stage(&self, pool: &super::GdnPoolStore, verify: &VerifyBatch, keep: &[usize]) {
        assert_eq!(keep.len(), verify.counts.len());
        let offs = verify.offsets();
        let sel: Vec<u32> = keep
            .iter()
            .zip(verify.counts)
            .zip(&offs)
            .map(|((&k, &c), &o)| {
                assert!(k >= 1 && k as u32 <= c, "qwen35 verify: keeping {k} of {c} rows");
                if (k as u32) < c {
                    o + k as u32 - 1
                } else {
                    u32::MAX
                }
            })
            .collect();
        if sel.iter().all(|&s| s == u32::MAX) {
            return;
        }
        let g = &self.gpu;
        let put = |v: &[u32]| {
            let b = g.storage(v.len() as u64);
            g.write(&b, v);
            b
        };
        let (rows, sel) = (put(verify.slots), put(&sel));
        let shape = self.gdn_verify_shape();
        let steps: Vec<_> = (0..self.cfg.n_layers as usize)
            .filter(|&l| self.shard.owns(l) && self.cfg.layer_types()[l] == crate::config::LayerType::Linear)
            .map(|l| {
                model::gdn_verify::gdn_commit_pooled(g, &shape, &pool.state[l], &pool.hist[l], verify.snapshots(l), &rows, &sel, verify.sequences())
                    .expect("the commit kernel was offered at engine construction (Qwen35GgufInstance::speculation_supported)")
            })
            .collect();
        g.submit(&[], &steps);
    }

    /// The Gated DeltaNet mixer shape the verify kernels are called with.
    fn gdn_verify_shape(&self) -> model::gdn_mixer::GdnMixerShape {
        let c = &self.cfg;
        let gdn = model::gdn::GdnShape { b: 1, h: c.linear_num_value_heads, t: 1, dk: c.linear_key_head_dim, dv: c.linear_value_head_dim, chunk: 1 };
        model::gdn_mixer::GdnMixerShape { gdn, nkh: c.linear_num_key_heads, conv_kernel: c.linear_conv_kernel_dim, rms_eps: c.rms_eps }
    }

    /// Whether this stage's device offers every kernel a verify pass runs and
    /// the layer shapes are ones they serve.
    pub(crate) fn verify_supported(&self) -> bool {
        self.gpu.has_fused(gpu_core::Fused::GdnVerifyPool) && self.gpu.has_fused(gpu_core::Fused::GdnCommitPool) && verify_shape_supported(&self.cfg)
    }

    /// Row snapshots for one Gated DeltaNet layer of this stage, with room for
    /// `rows` packed verify rows.
    pub(crate) fn new_row_snapshots(&self, rows: u32) -> GdnRowSnapshots {
        GdnRowSnapshots::new(&self.gpu, &self.gdn_verify_shape(), rows)
    }
}
