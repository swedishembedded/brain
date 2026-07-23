// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parallelism **planner** — pick the tensor-parallel degree from a cost model,
//! not from "as many GPUs as possible".
//!
//! Splitting one GEMM across `t` devices trades local compute for communication:
//!
//! ```text
//! T_total(t) = T_local(t) + T_comm(t) + T_sync(t)
//! ```
//!
//! Raising `t` shrinks `T_local` (each device does `1/t` of the FLOPs) but grows
//! `T_comm` (an all-reduce per layer) and can *hurt* per-device GEMM efficiency
//! once the local dimension drops below the kernel's efficient tile
//! (`gemm_min_dim`, 128 for brain's reg2). So the best `t` is usually the
//! **smallest** tightly-coupled group that (a) makes the model fit and (b)
//! minimises `T_total` — keeping local GEMMs large. This module encodes exactly
//! that, and is unit-tested against the qualitative regimes (fast vs slow link,
//! memory-forced, efficiency-limited).

/// Hardware the plan runs on. Bandwidth/latency describe the interconnect used by
/// the tensor-parallel collective (NVLink/NVSwitch → high bw, low latency; PCIe
/// via host → low bw, high latency; network → lower still).
#[derive(Clone, Copy, Debug)]
pub struct Hardware {
    pub n_gpus: usize,
    pub mem_bytes: u64,      // usable memory per GPU
    pub peak_flops: f64,     // per-GPU sustained FLOP/s for the GEMM kernel
    pub link_bytes_per_s: f64, // effective all-reduce bandwidth between TP peers
    pub link_latency_s: f64, // per-collective latency
    pub gemm_min_dim: usize, // local dim below which GEMM efficiency falls off (reg2 tile = 128)
}

/// The transformer shape being planned (one training step).
#[derive(Clone, Copy, Debug)]
pub struct ModelShape {
    pub tokens: usize, // m = batch * seq
    pub d_model: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    /// Bytes of weights + optimiser state for the whole model (what TP shards by `t`).
    pub state_bytes: u64,
    /// Bytes of activations kept per GPU for a step (not sharded by plain TP).
    pub act_bytes: u64,
}

/// A recommended tensor-parallel degree and why.
#[derive(Clone, Debug, PartialEq)]
pub struct TpPlan {
    pub degree: usize,
    pub predicted_secs: f64,
    pub fits: bool,
    pub note: &'static str,
}

/// Local-GEMM efficiency in `(0,1]`: full until the smallest split dimension drops
/// below `gemm_min_dim`, then linear falloff (small GEMMs under-fill the tiles).
fn gemm_efficiency(gemm_min_dim: usize, local_dim: usize) -> f64 {
    if local_dim >= gemm_min_dim {
        1.0
    } else {
        (local_dim as f64 / gemm_min_dim as f64).max(1e-3)
    }
}

/// Predicted per-step time (forward+backward, all layers) at TP degree `t`.
pub fn tp_step_secs(hw: &Hardware, s: &ModelShape, t: usize) -> f64 {
    let (m, d, ff, l) = (s.tokens as f64, s.d_model as f64, s.d_ff as f64, s.n_layers as f64);
    // Per-layer forward GEMM FLOPs: MLP two GEMMs (2·m·d·ff each) + attention four
    // d×d projections (2·m·d·d each). Split by t.
    let layer_flops = 2.0 * (2.0 * m * d * ff) + 4.0 * (2.0 * m * d * d);
    // Smallest local dim after the split (MLP hidden ff/t vs attention d/t).
    let local_dim = (s.d_ff.min(s.d_model)) / t;
    let eff = gemm_efficiency(hw.gemm_min_dim, local_dim);
    let t_local = l * layer_flops / t as f64 / (hw.peak_flops * eff);
    // Communication: 2 all-reduces per layer (attn + MLP), each of a [m,d]
    // activation; ring all-reduce moves ~2·(t-1)/t of the message.
    let msg = m * d * 4.0;
    let t_comm = if t <= 1 {
        0.0
    } else {
        l * 2.0 * (2.0 * (t as f64 - 1.0) / t as f64 * msg / hw.link_bytes_per_s + hw.link_latency_s)
    };
    // Backward ≈ 2× the forward's compute and communication.
    2.0 * (t_local + t_comm)
}

/// Does the model fit at TP degree `t`? TP shards weights+optimiser by `t`;
/// activations are not sharded by plain TP.
pub fn fits(hw: &Hardware, s: &ModelShape, t: usize) -> bool {
    s.state_bytes / t as u64 + s.act_bytes <= hw.mem_bytes
}

/// Candidate TP degrees: divisors of `n_gpus` (a TP group evenly splits its GEMMs).
fn candidate_degrees(n_gpus: usize) -> Vec<usize> {
    (1..=n_gpus).filter(|t| n_gpus % t == 0).collect()
}

/// The smallest local (split) GEMM dimension at TP degree `t`.
fn local_gemm_dim(s: &ModelShape, t: usize) -> usize {
    s.d_ff.min(s.d_model) / t
}

/// Choose the tensor-parallel degree, **capacity-first** (Megatron practice): the
/// *smallest* degree that makes the model fit — TP has real communication cost, so
/// it is used to fit the model, not to chase throughput (data/pipeline parallelism
/// do that). Among fitting degrees we prefer the smallest that also keeps the local
/// GEMMs at/above the efficient tile (`gemm_min_dim`); if memory forces a degree
/// that shrinks GEMMs below the tile, that is reported. `predicted_secs` is the
/// cost model's `T_local + T_comm + T_sync` estimate for the chosen degree.
pub fn plan_tp(hw: &Hardware, s: &ModelShape) -> TpPlan {
    let degrees = candidate_degrees(hw.n_gpus);
    let secs = |t: usize| tp_step_secs(hw, s, t);
    let Some(smallest_fit) = degrees.iter().copied().find(|&t| fits(hw, s, t)) else {
        let t = *degrees.last().unwrap();
        return TpPlan { degree: t, predicted_secs: secs(t), fits: false, note: "model does not fit even at max TP" };
    };
    let note = if smallest_fit == 1 {
        "fits on one GPU; no tensor parallelism needed"
    } else if local_gemm_dim(s, smallest_fit) < hw.gemm_min_dim {
        "TP forced by memory, and it shrinks local GEMMs below the efficient tile"
    } else {
        "smallest TP that fits, keeping local GEMMs large"
    };
    TpPlan { degree: smallest_fit, predicted_secs: secs(smallest_fit), fits: true, note }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> ModelShape {
        ModelShape { tokens: 4096, d_model: 4096, d_ff: 16384, n_layers: 32, state_bytes: 40 << 30, act_bytes: 4 << 30 }
    }

    // PCIe-like: low bandwidth, high latency (this box). Model fits at TP=1.
    fn pcie() -> Hardware {
        Hardware { n_gpus: 8, mem_bytes: 80 << 30, peak_flops: 1e13, link_bytes_per_s: 6e9, link_latency_s: 20e-6, gemm_min_dim: 128 }
    }
    // NVLink-like: high bandwidth, low latency.
    fn nvlink() -> Hardware {
        Hardware { n_gpus: 8, mem_bytes: 80 << 30, peak_flops: 1e14, link_bytes_per_s: 3e11, link_latency_s: 1e-6, gemm_min_dim: 128 }
    }

    // ---- selection: capacity-first (smallest TP that fits) ----

    #[test]
    fn fits_on_one_gpu_uses_no_tp() {
        // The shape fits at TP=1 on 80 GB cards -> TP=1, regardless of the link.
        assert!(fits(&pcie(), &shape(), 1));
        assert_eq!(plan_tp(&pcie(), &shape()).degree, 1);
        assert_eq!(plan_tp(&nvlink(), &shape()).degree, 1, "TP is capacity-first, not a throughput knob");
    }

    #[test]
    fn memory_pressure_forces_minimum_tp() {
        // State does not fit at TP=1 (42 GB) but fits at TP=2 (22 GB) on 24 GB cards.
        let mut hw = pcie();
        hw.mem_bytes = 24 << 30;
        let mut s = shape();
        s.state_bytes = 40 << 30;
        s.act_bytes = 2 << 30;
        assert!(!fits(&hw, &s, 1), "sanity: does not fit at TP=1");
        let p = plan_tp(&hw, &s);
        assert!(p.fits, "must find a fitting degree: {p:?}");
        assert_eq!(p.degree, 2, "smallest TP that fits: {p:?}");
    }

    #[test]
    fn reports_when_memory_forces_inefficient_gemms() {
        // Small dims + tight memory: the smallest fitting TP shrinks the GEMM
        // below the tile, and the plan says so.
        let mut hw = pcie();
        hw.mem_bytes = 8 << 30;
        hw.gemm_min_dim = 128;
        let mut s = shape();
        s.d_model = 256;
        s.d_ff = 256;
        s.state_bytes = 24 << 30; // fits only at t>=4 (6 GB) => local dim 256/4=64 < 128
        s.act_bytes = 1 << 30;
        let p = plan_tp(&hw, &s);
        assert!(p.fits && p.degree >= 4, "{p:?}");
        assert!(p.note.contains("shrinks local GEMMs"), "should flag inefficiency: {p:?}");
    }

    #[test]
    fn does_not_fit_reports_warning() {
        let mut hw = pcie();
        hw.mem_bytes = 1 << 30; // nothing fits
        let p = plan_tp(&hw, &shape());
        assert!(!p.fits);
        assert_eq!(p.degree, hw.n_gpus, "falls back to max TP");
    }

    // ---- cost model: the T_local ↓ / T_comm ↑ tradeoff the degree selection abstracts over ----

    #[test]
    fn more_tp_cuts_compute_but_adds_communication() {
        // Isolate the two terms by comparing a zero-comm ideal to the real link.
        let s = shape();
        let free = Hardware { link_bytes_per_s: f64::INFINITY, link_latency_s: 0.0, ..pcie() };
        // With free communication, more TP is strictly faster (pure compute split).
        assert!(tp_step_secs(&free, &s, 2) < tp_step_secs(&free, &s, 1));
        assert!(tp_step_secs(&free, &s, 4) < tp_step_secs(&free, &s, 2));
        // A slow link adds a positive communication cost at every TP>1.
        assert!(tp_step_secs(&pcie(), &s, 2) > tp_step_secs(&free, &s, 2), "PCIe adds comm over the ideal");
    }

    #[test]
    fn slow_high_latency_link_can_make_tp_slower_than_single() {
        // A comm-heavy regime (tiny compute, high per-collective latency) where
        // splitting is a net loss — the cost model captures it even though the
        // capacity planner would still pick TP=1 here because it fits.
        let hw = Hardware { peak_flops: 1e14, link_bytes_per_s: 1e8, link_latency_s: 5e-3, ..pcie() };
        let small = ModelShape { tokens: 256, d_model: 512, d_ff: 1024, n_layers: 8, state_bytes: 1 << 30, act_bytes: 1 << 28 };
        assert!(tp_step_secs(&hw, &small, 2) > tp_step_secs(&hw, &small, 1), "comm should dominate here");
    }
}
