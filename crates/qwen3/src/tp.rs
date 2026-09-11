// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tensor-parallel SwiGLU MLP - M7.3: the first real consumer of
//! `model::plan::{Hardware, ModelShape, plan_tp}` and of the M7.2-redesigned
//! `model::collective::Collective` trait, wired through one real piece of a
//! served model end to end (plan a degree -> shard real weights -> run N
//! simulated ranks -> combine through a collective -> reproduce the
//! unsharded model's own output).
//!
//! # Scope
//!
//! Qwen3's SwiGLU MLP (`mlp.{gate,up,down}.weight`, see `model.rs`'s
//! `SwiGLU MLP` dispatch block) is the sharded slice, using the standard
//! Megatron-LM tensor-parallel MLP layout (Shoeybi, Patwary, Puri, LeGresley,
//! Casper, Catanzaro - "Megatron-LM: Training Multi-Billion Parameter
//! Language Models Using Model Parallelism", arXiv:1909.08053, section 3):
//!
//! * `gate`/`up` are **column-parallel** - each rank owns a contiguous
//!   `d_ff / world_size` slice of OUTPUT rows. `SiLU` and the elementwise
//!   `gate * up` are both pointwise, so no communication is needed between
//!   the two column-parallel linears and the activation.
//! * `down` is **row-parallel** on the matching contracting dimension - each
//!   rank owns the `d_ff / world_size` INPUT columns that line up with its
//!   own `h` shard, so `down`'s local product is only a partial sum over
//!   `d_ff`.
//! * Exactly ONE all-reduce combines the `world_size` partial sums into the
//!   true `[rows, d_model]` MLP output - the same output the unsharded
//!   model produces on the same weights and input.
//!
//! This is not the full model: attention's QKV/O projections are not
//! sharded here (see the M7.3 report / roadmap entry for why the MLP was
//! chosen and what generalising to the rest of the model would take).
//!
//! # What this validates, and what it does NOT
//!
//! There is no real multi-GPU hardware in this environment. Every "rank"
//! below is an OS thread in ONE process on the ONE real device/backend this
//! box has, communicating through `model::collective::HostCollective`
//! (staged through host RAM, `Barrier`-synchronised). This is a legitimate
//! way to prove the SHARDING MATH and the COLLECTIVE WIRING are correct -
//! the equivalence test below fails if either is wrong - but it does **not**
//! validate real multi-GPU behaviour: no network transport, no real
//! device-to-device transfer, no multi-process concurrency. Do not read a
//! green test here as "TP works on real multi-GPU deployment".
//!
//! Host math (`model::hostmath::linear_rows`/`silu_slice`) is used
//! deliberately, per that module's own rule: this is exactly the
//! reference-vs-sharded correctness check that rule carves out, not a
//! device-invisible hot path.

use model::collective::{Collective, CollectiveResult, Payload};
use model::hostmath::{linear_rows, silu_slice};
pub use model::plan::{plan_tp, Hardware, ModelShape, TpPlan};

/// One rank's shard of a SwiGLU MLP's three weight matrices.
///
/// `gate`/`up` are `[d_ff_local, d_model]` row-major (this rank's contiguous
/// slice of output rows); `down` is `[d_model, d_ff_local]` row-major (this
/// rank's contiguous slice of input COLUMNS, gathered out of the full
/// `[d_model, d_ff]` matrix since a column slice of a row-major matrix is
/// not contiguous).
#[derive(Clone, Debug)]
pub struct MlpShard {
    pub d_model: usize,
    pub d_ff_local: usize,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

/// Split full `gate`/`up: [d_ff, d_model]` and `down: [d_model, d_ff]`
/// (`qwen3::model`'s own layout - see `model.rs`'s
/// `"mlp.gate.weight" => (ffd, dd)` / `"mlp.down.weight" => (dd, ffd)`) into
/// `world_size` column-parallel/row-parallel shards.
pub fn shard_mlp(d_model: usize, d_ff: usize, gate: &[f32], up: &[f32], down: &[f32], world_size: usize) -> Vec<MlpShard> {
    assert!(world_size >= 1, "world_size must be >= 1");
    assert_eq!(gate.len(), d_ff * d_model, "gate: {} != d_ff*d_model {}", gate.len(), d_ff * d_model);
    assert_eq!(up.len(), d_ff * d_model, "up: {} != d_ff*d_model {}", up.len(), d_ff * d_model);
    assert_eq!(down.len(), d_model * d_ff, "down: {} != d_model*d_ff {}", down.len(), d_model * d_ff);
    assert_eq!(d_ff % world_size, 0, "d_ff {d_ff} must divide evenly across world_size {world_size} (column-parallel split)");
    let chunk = d_ff / world_size;
    (0..world_size)
        .map(|r| {
            let lo = r * chunk;
            let gate_r = gate[lo * d_model..(lo + chunk) * d_model].to_vec();
            let up_r = up[lo * d_model..(lo + chunk) * d_model].to_vec();
            // `down` is [d_model, d_ff] row-major; this rank owns COLUMNS
            // [lo, lo+chunk) of every row, which is not a contiguous slice.
            let mut down_r = vec![0f32; d_model * chunk];
            for row in 0..d_model {
                down_r[row * chunk..(row + 1) * chunk].copy_from_slice(&down[row * d_ff + lo..row * d_ff + lo + chunk]);
            }
            MlpShard { d_model, d_ff_local: chunk, gate: gate_r, up: up_r, down: down_r }
        })
        .collect()
}

/// One rank's local forward: `x [rows, d_model] -> partial_out [rows,
/// d_model]`. This is a genuine PARTIAL sum over `d_model` - it must be
/// all-reduced across ranks (see [`mlp_forward_tp`]) before it is the true
/// MLP output. Mirrors `qwen3::model`'s own SwiGLU dispatch order
/// (`gate_pre`/`up` projections -> `SiLU(gate) * up` -> `down` projection,
/// see `model.rs`'s `block::swiglu_fwd` call site) at this rank's shard.
pub fn mlp_shard_forward(shard: &MlpShard, x: &[f32], rows: usize) -> Vec<f32> {
    let gate_pre = linear_rows(x, &shard.gate, rows, shard.d_model, shard.d_ff_local);
    let up = linear_rows(x, &shard.up, rows, shard.d_model, shard.d_ff_local);
    let silu_gate = silu_slice(&gate_pre);
    let h: Vec<f32> = silu_gate.iter().zip(&up).map(|(g, u)| g * u).collect();
    linear_rows(&h, &shard.down, rows, shard.d_ff_local, shard.d_model)
}

/// Rank `rank`'s full sharded-MLP step: local forward, then all-reduce the
/// row-parallel partial sums through `collective` so every rank ends up with
/// the SAME `[rows, d_model]` output the unsharded MLP would produce.
pub async fn mlp_forward_tp(collective: &dyn Collective, rank: usize, shard: &MlpShard, x: &[f32], rows: usize) -> CollectiveResult<Vec<f32>> {
    let partial = mlp_shard_forward(shard, x, rows);
    let summed = collective.all_reduce(rank, Payload::f32(partial)).await?;
    Ok(summed.data)
}

/// The unsharded reference MLP forward - the SAME math as
/// [`mlp_shard_forward`] at `world_size = 1`, computed directly rather than
/// through a shard, so the equivalence test below has an independent target
/// to compare the sharded/collective path against.
pub fn mlp_forward_reference(d_model: usize, d_ff: usize, gate: &[f32], up: &[f32], down: &[f32], x: &[f32], rows: usize) -> Vec<f32> {
    let gate_pre = linear_rows(x, gate, rows, d_model, d_ff);
    let up_pre = linear_rows(x, up, rows, d_model, d_ff);
    let silu_gate = silu_slice(&gate_pre);
    let h: Vec<f32> = silu_gate.iter().zip(&up_pre).map(|(g, u)| g * u).collect();
    linear_rows(&h, down, rows, d_ff, d_model)
}

/// End-to-end result of planning and running one TP step: the [`TpPlan`]
/// that decided `world_size`, and every simulated rank's own output (equal
/// across ranks by the collective's own contract - `all_reduce` delivers the
/// same total to everyone).
pub struct TpMlpRun {
    pub plan: TpPlan,
    pub outputs: Vec<Vec<f32>>,
}

/// Plan the TP degree for this MLP shape via `model::plan::plan_tp`, shard
/// the real weights at that degree, and execute `plan.degree` simulated
/// ranks (real OS threads on this one process/device) through
/// `model::collective::HostCollective`. THIS is the end-to-end wiring M7.3
/// asks for: the degree used to shard and execute is not a hardcoded test
/// parameter, it is `plan_tp`'s own prediction.
///
/// `hw`/`state_bytes`/`act_bytes` are the caller's cost-model inputs (see
/// `model::plan::Hardware`/`ModelShape`) - this function does not invent
/// them, it feeds the real weight/activation byte counts for THIS MLP shape
/// into the SAME planner every other model would use.
pub fn run_tp_mlp(hw: &Hardware, d_model: usize, d_ff: usize, gate: &[f32], up: &[f32], down: &[f32], x: &[f32], rows: usize) -> TpMlpRun {
    let state_bytes = ((2 * d_ff * d_model + d_model * d_ff) * std::mem::size_of::<f32>()) as u64;
    let act_bytes = (rows * d_ff * std::mem::size_of::<f32>()) as u64;
    let shape = ModelShape { tokens: rows, d_model, d_ff, n_layers: 1, state_bytes, act_bytes };
    let plan = plan_tp(hw, &shape);
    let world = plan.degree;
    let shards = shard_mlp(d_model, d_ff, gate, up, down, world);
    let collective = model::collective::HostCollective::new(world);
    let outputs = std::thread::scope(|s| {
        let handles: Vec<_> = (0..world)
            .map(|r| {
                let collective = collective.as_ref();
                let shard = &shards[r];
                s.spawn(move || pollster::block_on(mlp_forward_tp(collective, r, shard, x, rows)).expect("TP MLP collective step failed"))
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("TP rank thread panicked")).collect()
    });
    TpMlpRun { plan, outputs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use model::hostmath::randn;

    struct Case {
        d_model: usize,
        d_ff: usize,
        rows: usize,
        gate: Vec<f32>,
        up: Vec<f32>,
        down: Vec<f32>,
        x: Vec<f32>,
    }

    /// A real (if small) SwiGLU-shaped MLP with seeded random weights -
    /// distinct seeds per tensor so a swapped operand would not go unnoticed
    /// by coincidence, `d_model`/`d_ff`/`rows` mutually distinct so a
    /// transposed dimension would not silently type-check either.
    fn case() -> Case {
        let d_model = 64;
        let d_ff = 256;
        let rows = 6;
        Case {
            d_model,
            d_ff,
            rows,
            gate: randn(d_ff * d_model, 1),
            up: randn(d_ff * d_model, 2),
            down: randn(d_model * d_ff, 3),
            x: randn(rows * d_model, 4),
        }
    }

    /// Hardware whose memory only fits the MLP's weights once split across
    /// `want_degree` GPUs - forces `plan_tp` to actually pick `want_degree`
    /// rather than defaulting to TP=1, so the equivalence test below
    /// exercises real sharding (N>1), not a degenerate single-rank no-op.
    /// `fits` (see `model::plan`) checks `state_bytes/t + act_bytes <=
    /// mem_bytes`, so the budget below must clear both terms, not just the
    /// sharded weight bytes.
    fn hw_forcing_degree(want_degree: usize, state_bytes: u64, act_bytes: u64) -> Hardware {
        Hardware {
            n_gpus: want_degree,
            mem_bytes: state_bytes / want_degree as u64 + act_bytes + 1,
            peak_flops: 1e13,
            link_bytes_per_s: 6e9,
            link_latency_s: 20e-6,
            gemm_min_dim: 1, // tiny test shapes - don't let the efficiency floor mask the memory forcing
        }
    }

    fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol * w.abs().max(1.0), "index {i}: got {g}, want {w}");
        }
    }

    fn run_and_check(want_degree: usize) -> TpMlpRun {
        let c = case();
        let state_bytes = ((2 * c.d_ff * c.d_model + c.d_model * c.d_ff) * 4) as u64;
        let act_bytes = (c.rows * c.d_ff * 4) as u64;
        let hw = hw_forcing_degree(want_degree, state_bytes, act_bytes);
        let run = run_tp_mlp(&hw, c.d_model, c.d_ff, &c.gate, &c.up, &c.down, &c.x, c.rows);
        assert!(run.plan.fits, "plan should find a fitting degree: {:?}", run.plan);
        assert_eq!(run.plan.degree, want_degree, "plan_tp should have picked the forced degree: {:?}", run.plan);

        let want = mlp_forward_reference(c.d_model, c.d_ff, &c.gate, &c.up, &c.down, &c.x, c.rows);
        assert_eq!(run.outputs.len(), want_degree);
        for (r, got) in run.outputs.iter().enumerate() {
            assert_close(got, &want, 1e-4);
            let _ = r;
        }
        run
    }

    /// The correctness gate M7.3 asks for: TP-sharded output (N=2 simulated
    /// ranks) matches the unsharded reference within tolerance, on the same
    /// real weights/input, with the degree itself coming from `plan_tp`.
    #[test]
    fn tp_sharded_mlp_matches_unsharded_reference_n2() {
        run_and_check(2);
    }

    /// Same gate at N=4, to show the sharding/collective wiring generalises
    /// past the smallest nontrivial split.
    #[test]
    fn tp_sharded_mlp_matches_unsharded_reference_n4() {
        run_and_check(4);
    }

    /// `plan_tp`'s own no-TP-needed case still runs correctly through this
    /// path (world_size = 1 degenerates `all_reduce` to a no-op sum).
    #[test]
    fn tp_plan_degree_one_still_matches_reference() {
        let c = case();
        let hw = Hardware { n_gpus: 1, mem_bytes: 1 << 30, peak_flops: 1e13, link_bytes_per_s: 6e9, link_latency_s: 20e-6, gemm_min_dim: 128 };
        let run = run_tp_mlp(&hw, c.d_model, c.d_ff, &c.gate, &c.up, &c.down, &c.x, c.rows);
        assert_eq!(run.plan.degree, 1);
        let want = mlp_forward_reference(c.d_model, c.d_ff, &c.gate, &c.up, &c.down, &c.x, c.rows);
        assert_close(&run.outputs[0], &want, 1e-4);
    }

    /// `shard_mlp`'s down-projection column gather must reproduce the exact
    /// bytes of the corresponding slice of the unsharded `down` matrix - a
    /// targeted check on the one non-contiguous split in this module (the
    /// gate/up split is a contiguous row slice; a bug there would already
    /// fail the end-to-end tests above less legibly).
    #[test]
    fn shard_mlp_down_projection_columns_are_exact() {
        let c = case();
        let world = 4;
        let shards = shard_mlp(c.d_model, c.d_ff, &c.gate, &c.up, &c.down, world);
        let chunk = c.d_ff / world;
        for (r, shard) in shards.iter().enumerate() {
            for row in 0..c.d_model {
                let want = &c.down[row * c.d_ff + r * chunk..row * c.d_ff + (r + 1) * chunk];
                let got = &shard.down[row * chunk..(row + 1) * chunk];
                assert_eq!(got, want, "rank {r}, row {row}");
            }
        }
    }
}
