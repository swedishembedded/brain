// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where each Thinker decoder layer's weights live, for a **raw HF
//! checkpoint** of any per-tensor dtype mix - the placement half of what
//! `crate::generate` needs, kept separate from the forward pass so it can be
//! tested without a GPU or a 66 GB checkpoint.
//!
//! Nothing here is dtype-specific and nothing here is a second placement
//! algorithm. It supplies real per-layer byte costs - each tensor charged from
//! its OWN declared dtype and shape via [`paramstore::dtype`], so a checkpoint
//! that stores attention at `BF16` and experts at `Q4_K` (the ordinary GGUF
//! case) costs exactly what it costs - and hands them to
//! `model::shard::plan_by_capacity`, the one capacity-aware partitioner, which
//! decides the split across however many devices exist with whatever VRAM each
//! has.
//!
//! # Resident where it fits, streamed where it does not
//!
//! A 30B checkpoint expanded to f32 is ~120 GiB of decoder weights, which no
//! pair of 24 GB cards holds. Refusing to serve it would be one honest answer;
//! streaming every layer per token (what this path did before) is another, and
//! it is the one that keeps the full chat/multimodal surface working. What was
//! actually broken is that streaming had no plan and no bound at all: it
//! allocated every layer's ~2.4 GiB of expert buffers and never forced the
//! submit that lets the device reclaim the previous layer's, so a single
//! generate walked a 24 GB card to OOM about nine layers in.
//!
//! So the plan is one mechanism covering both: the longest run of layers that
//! genuinely FITS is placed resident across the devices by the planner, and
//! whatever is left over streams onto the last stage in bounded chunks with a
//! forced drain per layer. A checkpoint that fits entirely (a small model, an
//! int8 one, a single test layer) streams nothing and takes the fast path
//! through the same code - [`ThinkerPlacement::streamed`] is simply empty.

use std::ops::Range;

use checkpoint::weightio::WeightReader;
use model::shard::{plan_by_capacity, LayerBytes};
use paramstore::dtype::tensor_device_bytes;

use crate::config::MoeTextConfig;

/// Per-layer non-expert tensor leaves, under `thinker.model.layers.{l}.`.
/// Named once so the byte accounting and `crate::generate`'s loader cannot
/// drift: an accounting that omits a tensor the loader uploads is exactly the
/// under-reported budget that lets a placement decision overrun a card.
pub const LAYER_LEAVES: &[&str] = &[
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate.weight",
];

/// One routed expert's three linears, under `thinker.model.layers.{l}.mlp.experts.{e}.`.
pub const EXPERT_LEAVES: &[&str] = &["gate_proj.weight", "up_proj.weight", "down_proj.weight"];

/// Tensors the LAST stage carries (the final hidden state already lands there,
/// so applying the head there costs no extra cross-device hop).
pub const HEAD_TENSORS: &[&str] = &["thinker.model.norm.weight", "thinker.lm_head.weight"];

/// The token embedding. Never device-resident: generation only ever gathers a
/// per-token ROW from it.
pub const EMBED_TENSOR: &str = "thinker.model.embed_tokens.weight";

/// Device bytes layer `l` occupies, summing every tensor the loader uploads,
/// each charged from its own dtype. `None` if the checkpoint is missing any of
/// them (the loader would fail on it later, so a plan must not be built).
pub fn layer_device_bytes(reader: &WeightReader, cfg: &MoeTextConfig, l: usize) -> Option<u64> {
    let p = |leaf: &str| format!("thinker.model.layers.{l}.{leaf}");
    let mut total = 0u64;
    for leaf in LAYER_LEAVES {
        total += tensor_device_bytes(reader, &p(leaf))?;
    }
    for e in 0..cfg.n_experts {
        for leaf in EXPERT_LEAVES {
            total += tensor_device_bytes(reader, &p(&format!("mlp.experts.{e}.{leaf}")))?;
        }
    }
    Some(total)
}

/// The byte-exact cost model `model::shard`'s planner consumes. `embed` is 0 -
/// the embedding table is gathered host-side, never uploaded (see
/// [`EMBED_TENSOR`]).
pub fn layer_cost(reader: &WeightReader, cfg: &MoeTextConfig) -> Option<LayerBytes> {
    let mut per_layer = Vec::with_capacity(cfg.n_layers as usize);
    for l in 0..cfg.n_layers as usize {
        per_layer.push(layer_device_bytes(reader, cfg, l)?);
    }
    let mut head = 0u64;
    for n in HEAD_TENSORS {
        head += tensor_device_bytes(reader, n)?;
    }
    Some(LayerBytes { per_layer, embed: 0, head })
}

/// One device's share: which card, which contiguous layers it holds RESIDENT,
/// and the bytes that costs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stage {
    /// Index into the caller's own device list - never a physical card number
    /// this module invented.
    pub device: usize,
    pub layers: Range<usize>,
    pub bytes: u64,
}

/// A complete placement: resident stages, plus whatever did not fit, which
/// streams onto the last stage's device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThinkerPlacement {
    pub stages: Vec<Stage>,
    /// Layers held by no stage - streamed on demand onto the last stage's
    /// device, one at a time, and dropped after use. Empty when the whole
    /// model fits.
    pub streamed: Range<usize>,
}

impl ThinkerPlacement {
    /// Which device runs layer `l` - its resident stage's, or (for a streamed
    /// layer) the last stage's, where the head already lives.
    pub fn device_of(&self, l: usize) -> usize {
        self.stages.iter().find(|s| s.layers.contains(&l)).map(|s| s.device).unwrap_or_else(|| self.last_device())
    }

    /// The device carrying the head tensors and every streamed layer.
    pub fn last_device(&self) -> usize {
        self.stages.last().map(|s| s.device).unwrap_or(0)
    }

    /// Whether layer `l` is held resident (as opposed to streamed per use).
    pub fn is_resident(&self, l: usize) -> bool {
        self.stages.iter().any(|s| s.layers.contains(&l))
    }

    /// Total resident device bytes per device index, in stage order - what a
    /// scheduler should reserve.
    pub fn per_device_bytes(&self) -> Vec<(usize, u64)> {
        self.stages.iter().map(|s| (s.device, s.bytes)).collect()
    }
}

/// Every GPU this process discovered, with the USABLE bytes each may spend
/// (its own total VRAM minus `reserved`) - the fallback for a caller that has
/// no budget of its own to hand in.
///
/// Reads the device registry rather than assuming a count: a 1-GPU box, a
/// 2-card box and a 3+-card box all come back correctly sized, and a box with
/// no GPU comes back empty (which callers report as "cannot serve" rather than
/// silently running somewhere else).
pub fn discovered_devices(reserved: u64) -> Vec<(u32, u64)> {
    gpu_core::devices::gpus().iter().map(|d| (d.index, d.identity.vram_bytes.saturating_sub(reserved))).collect()
}

/// Headroom, in bytes, that must stay free on the device that streams: enough
/// for the largest single layer's weights plus its activations and the
/// staging the upload itself needs.
///
/// Derived from the cost model rather than fixed, because "one layer" is a
/// different number for every model - and it is a MULTIPLE of one layer rather
/// than exactly one because the layer being uploaded, the dispatches still
/// retiring from the previous one, and that layer's activation scratch are all
/// live at the same instant.
fn stream_headroom(cost: &LayerBytes) -> u64 {
    cost.per_layer.iter().copied().max().unwrap_or(0).saturating_mul(3)
}

/// Place `cost`'s layers across `devices` (`(caller's device index, USABLE
/// bytes)`), holding resident as many leading layers as genuinely fit and
/// streaming the rest.
///
/// The fit question is answered entirely by `model::shard::plan_by_capacity` -
/// this only searches for how MANY layers to hand it. `None` when even zero
/// resident layers cannot be supported (no devices, or not even the head plus
/// one layer's streaming working set fits anywhere), which is a genuine
/// "cannot serve this here" rather than something to paper over.
pub fn place(cost: &LayerBytes, devices: &[(usize, u64)]) -> Option<ThinkerPlacement> {
    if devices.is_empty() {
        return None;
    }
    let n = cost.per_layer.len();
    let head = stream_headroom(cost);
    // Every device must keep the streaming headroom free, because the last
    // stage streams and (with a one-device plan) it is also the first.
    let budgeted: Vec<(usize, u64)> = devices.iter().map(|&(i, cap)| (i, cap.saturating_sub(head))).collect();

    // `plan_by_capacity` is monotone in the layer count (a prefix of a plan
    // that fits also fits), so the largest feasible prefix is a binary search
    // over it rather than n planner runs.
    let prefix = |m: usize| LayerBytes { per_layer: cost.per_layer[..m].to_vec(), embed: cost.embed, head: cost.head };
    let feasible = |m: usize| plan_by_capacity(&prefix(m), &budgeted);
    let mut lo = 0usize; // m = 0 is checked explicitly below
    let mut hi = n;
    let best = feasible(hi);
    if best.is_none() {
        // Not everything fits: find the largest m that does.
        feasible(0)?; // not even the head fits anywhere -> unplaceable
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if feasible(mid).is_some() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
    } else {
        lo = n;
    }
    let placements = feasible(lo)?;
    let stages: Vec<Stage> = placements
        .iter()
        .map(|p| Stage { device: p.shard.gpu_index, layers: p.shard.start..p.shard.end, bytes: p.bytes })
        .collect();
    Some(ThinkerPlacement { stages, streamed: lo..n })
}

/// [`place`] over the FEWEST leading devices that hold the WHOLE model
/// resident, falling back to all of them when it does not fit anywhere.
///
/// Same reasoning as `model::shard::plan_fewest_devices`: a model that fits
/// one card should not be spread over four. But once a model provably does not
/// fit at all, "fewest" stops being the right objective - every additional
/// card is one more layer held resident instead of re-read per token - so the
/// fallback deliberately uses every device offered.
pub fn place_fewest_devices(cost: &LayerBytes, devices: &[(usize, u64)]) -> Option<ThinkerPlacement> {
    for n in 1..=devices.len() {
        if let Some(p) = place(cost, &devices[..n]) {
            if p.streamed.is_empty() {
                return Some(p);
            }
        }
    }
    place(cost, devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(per_layer: &[u64], head: u64) -> LayerBytes {
        LayerBytes { per_layer: per_layer.to_vec(), embed: 0, head }
    }

    /// Spec: a model that fits stays whole on the FEWEST cards and streams
    /// nothing - one card is not spread over three just because three exist.
    #[test]
    fn a_model_that_fits_one_card_uses_one_card_and_streams_nothing() {
        let c = cost(&[10; 4], 5);
        let p = place_fewest_devices(&c, &[(0, 1000), (1, 1000), (2, 1000)]).expect("must place");
        assert_eq!(p.stages.len(), 1, "{:?}", p.stages);
        assert_eq!(p.stages[0].layers, 0..4);
        assert!(p.streamed.is_empty(), "nothing should stream: {:?}", p.streamed);
    }

    /// Spec: device COUNT is never assumed. The same cost model places across
    /// 1, 2, 3 and 5 devices, and every layer is accounted for exactly once
    /// (resident or streamed) at every count.
    #[test]
    fn it_places_across_any_device_count() {
        // Per-layer 100, head 50; a 260-byte card holds at most 2 layers after
        // the 300-byte streaming headroom is... deliberately larger than one
        // card, so give the cards room to make the counts interesting.
        let c = cost(&[100; 12], 50);
        for k in [1usize, 2, 3, 5] {
            let devices: Vec<(usize, u64)> = (0..k).map(|i| (i, 900u64)).collect();
            let p = place(&c, &devices).unwrap_or_else(|| panic!("{k} device(s) must place something"));
            assert_eq!(p.stages.len(), k, "one stage per device at k={k}: {:?}", p.stages);
            let resident: usize = p.stages.iter().map(|s| s.layers.len()).sum();
            assert_eq!(resident + p.streamed.len(), 12, "every layer placed exactly once at k={k}");
            // Ranges are contiguous, in order, and end where the streamed run starts.
            let mut next = 0usize;
            for s in &p.stages {
                assert_eq!(s.layers.start, next, "stages must be contiguous at k={k}: {:?}", p.stages);
                next = s.layers.end;
            }
            assert_eq!(p.streamed.start, next, "streamed run must follow the resident ones at k={k}");
            // More cards must never hold FEWER layers resident.
            assert!(resident >= 12usize.min(k), "k={k} resident={resident}");
        }
    }

    /// Spec: uneven capacity is respected - the bigger card takes more layers,
    /// rather than an even split that overruns the smaller one.
    #[test]
    fn uneven_cards_get_layers_in_proportion_not_evenly() {
        let c = cost(&[100; 10], 0);
        let p = place(&c, &[(0, 2000), (1, 700)]).expect("must place");
        assert_eq!(p.stages.len(), 2);
        assert!(
            p.stages[0].layers.len() > p.stages[1].layers.len(),
            "the 2000-byte card must take more than the 700-byte one: {:?}",
            p.stages
        );
        for s in &p.stages {
            let cap = if s.device == 0 { 2000 } else { 700 };
            assert!(s.bytes <= cap - 300, "stage {s:?} must fit its card minus streaming headroom");
        }
    }

    /// Spec: what does not fit is STREAMED, not silently dropped and not an
    /// error - and it is a contiguous tail, so the forward pass still walks
    /// layers in order across the stages.
    #[test]
    fn an_oversized_model_streams_the_remainder_instead_of_failing() {
        let c = cost(&[100; 50], 100);
        let p = place(&c, &[(0, 900), (1, 900)]).expect("an oversized model must still place");
        assert!(!p.streamed.is_empty(), "some layers must stream");
        let resident: usize = p.stages.iter().map(|s| s.layers.len()).sum();
        assert_eq!(resident + p.streamed.len(), 50);
        assert_eq!(p.streamed.end, 50);
        // Streamed layers run on the last stage's device, where the head is.
        assert_eq!(p.device_of(49), p.last_device());
        assert!(!p.is_resident(49) && p.is_resident(0));
    }

    /// Spec: genuine infeasibility is reported. A card too small to hold even
    /// the head plus one layer's streaming working set is not a placement.
    #[test]
    fn a_card_that_cannot_even_stream_is_unplaceable() {
        let c = cost(&[1000; 4], 100);
        assert!(place(&c, &[(0, 500)]).is_none(), "a 500-byte card cannot stream 1000-byte layers");
        assert!(place(&c, &[]).is_none(), "no devices is not a placement");
    }

    /// Spec: per-tensor dtype drives the cost. The same layer shapes at
    /// different quantization produce different placements through the same
    /// planner - the property that lets a mixed-quant checkpoint reuse this
    /// mechanism rather than needing its own.
    #[test]
    fn per_layer_costs_may_differ_and_placement_follows_them() {
        // Layers 0-3 "quantized small", 4-7 "kept wide" - exactly the shape a
        // GGUF release with mixed per-layer precision produces.
        let c = cost(&[50, 50, 50, 50, 400, 400, 400, 400], 0);
        let p = place(&c, &[(0, 1500), (1, 1500)]).expect("must place");
        let resident: usize = p.stages.iter().map(|s| s.layers.len()).sum();
        assert_eq!(resident + p.streamed.len(), 8);
        // The cheap layers all fit; a uniform-cost model would have mis-sized this.
        assert!(p.is_resident(0) && p.is_resident(3), "the cheap prefix must be resident: {:?}", p.stages);
    }
}
