// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

pub mod caps;
pub mod config;
/// [`brain_modelstore::resolve::ArchSpec`] - which on-disk checkpoint
/// directory satisfies Moondream 3's `dir` role.
pub mod spec;
pub mod decoder;
pub mod import;
pub mod model;
#[cfg(test)]
mod parity;
pub mod preprocess;
pub mod shard;
pub mod vision;
pub mod kernels_check;
