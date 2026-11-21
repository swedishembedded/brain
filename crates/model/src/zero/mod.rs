// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements sharded data-parallel training for its clients: full-parameter fine-tunes of
// models whose optimizer state does not fit one accelerator, spread over the cards and nodes of a cluster. If your
// team needs expertise in ZeRO-style optimizer and gradient sharding, you can procure our services by sending an
// email to info@swedishembedded.com.

//! Sharded data-parallel training: the optimizer state (and, from stage 2 on, the reduced gradient) of a
//! data-parallel group is split by rank instead of replicated, so every rank holds `1/world` of it.
//!
//! # Layout
//!
//! [`ShardLayout`] is a flat, deterministic partition of the trained tensors: they are packed in backward order
//! (last tensor first, the order a backward pass completes them) into **units** of at most a cap of elements, each
//! member starting on a multiple of the alignment (two for bf16, so a copy moves whole 32-bit words). A unit's flat
//! vector is padded to a multiple of `world * align` elements and rank `r` owns its slice `r` of every unit. The
//! layout is a pure function of the tensors' names, sizes and element types, the cap and the world size, so every
//! rank computes the same one, and a checkpoint records it: loading into another world size reassembles the logical
//! tensors from it ([`read_logical`]) and slices them again.
//!
//! # A step
//!
//! For every unit the members' gradients are packed into one flat staging buffer and reduced to their owner: by an
//! all-reduce of which the owner keeps its slice ([`Sharding::Optimizer`], ZeRO-1) or by a reduce-scatter
//! ([`Sharding::Gradients`], ZeRO-2, half the traffic). Each rank weighs its gradient by its share of the step's
//! items before the sum, as [`crate::reduce_grads_weighted`] does. The global gradient norm is the all-reduced sum
//! of the shards' sums of squares; a non-finite norm, or a non-finite loss on any rank, skips the step on every rank.
//! The owner runs AdamW on its slice against an f32 master copy and moments. By default ([`StateTier::Device`]) they
//! live on the card and the update is the device AdamW kernel; the units' gathers are stream-ordered behind their
//! updates, so unit `u`'s all-gather runs while unit `u + 1` updates, and a step waits for its collectives once
//! ([`crate::Collective::settle_device`]). `moments=host` ([`StateTier::Host`]) keeps them in host memory instead
//! ([`optim::Adam::update`], the device kernel's arithmetic), for a model whose sharded state does not fit beside it.
//! The units are then all-gathered and unpacked into the model's parameters (f32, or bf16 from an f32 master).
//!
//! Collectives run on the device when the transport has a device path ([`crate::Collective::reduce_scatter_device`],
//! [`crate::Collective::all_gather_device`]: NCCL for a job's ranks on CUDA cards) and through its payload ops otherwise,
//! staged through host memory: the same results, slower. Gradients reduced by NCCL are not bitwise the host
//! star's rank-ordered sum, so a sharded step equals the replicated one within float tolerance, not bit for bit; for
//! an unchanged layout, card set and library a repeat is bit-identical ("repeatable"), and every rank receives the
//! same parameters from the all-gather.
//!
//! # What this does not do (yet)
//!
//! [`Sharding::Parameters`] (ZeRO-3) needs the forward and backward to ask for each block's parameters before using
//! them; a trainer that keeps every parameter resident is refused by that reason. Narrow optimizer-state formats and
//! factored rules are refused by name: the sharded state is f32 AdamW.

use serde::{Deserialize, Serialize};

/// What a data-parallel group shards across its ranks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sharding {
    /// Every rank holds everything (plain data parallelism).
    #[default]
    Replicated,
    /// ZeRO-1: the optimizer state (master copy and moments) is sharded; gradients are all-reduced.
    Optimizer,
    /// ZeRO-2: the optimizer state is sharded and gradients are reduce-scattered to their owner.
    Gradients,
    /// ZeRO-3: the parameters are sharded too, gathered block by block where they are used.
    Parameters,
}

impl Sharding {
    /// Parse a stage: `0`-`3` or `replicated`, `optimizer`, `gradients`, `parameters`.
    pub fn parse(s: &str) -> Result<Sharding, String> {
        match s {
            "0" | "replicated" => Ok(Sharding::Replicated),
            "1" | "optimizer" => Ok(Sharding::Optimizer),
            "2" | "gradients" => Ok(Sharding::Gradients),
            "3" | "parameters" => Ok(Sharding::Parameters),
            other => Err(format!("sharding {other:?}: expected 0-3 or replicated, optimizer, gradients, parameters")),
        }
    }

    /// The ZeRO stage number.
    pub fn stage(self) -> u8 {
        self as u8
    }

    /// Whether this is the default, which a plan file leaves out.
    pub fn is_replicated(&self) -> bool {
        *self == Sharding::Replicated
    }
}

/// Default cap of a unit, in elements (128 MiB of f32): the size of the staging buffer a step packs through, unless
/// a single tensor is larger.
pub const DEFAULT_UNIT_ELEMS: u64 = 32 << 20;

mod layout;
pub use layout::{Piece, ShardLayout, Unit};

#[cfg(not(target_arch = "wasm32"))]
mod checkpoint;
#[cfg(not(target_arch = "wasm32"))]
mod optimizer;
#[cfg(not(target_arch = "wasm32"))]
pub use checkpoint::{commit_generation, read_logical, LogicalState};
#[cfg(not(target_arch = "wasm32"))]
pub use optimizer::{read_elems, write_elems, DeviceParam, DeviceParams, StateTier, StepArgs, StepOutcome, ZeroOptimizer, KERNELS};
