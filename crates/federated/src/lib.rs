// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Federated / sharded Mixture-of-Experts.
//!
//! - [`shard`] — vertical expert split/assemble + hash-verified manifests, the
//!   "train experts separately → assemble into one model" artifact mechanic.
//! - [`sha256`] — dependency-free content hashing for the manifests.
//!
//! Train-scope control (freeze backbone + train one expert) and the
//! router-integration phase live in the MoE engine; this crate owns the
//! checkpoint-level federation that ties independent expert training together.

pub mod sha256;
pub mod shard;

pub use shard::{assemble, expert_id, merge_to_full, split, split_filtered, verify, Manifest};
