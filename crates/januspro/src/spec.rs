// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Model-store resolution for Janus-Pro: `brain-deepseekvl`'s family spec,
//! claiming the checkpoints whose config carries the generation heads.

pub use deepseekvl::spec::MultiModalitySpec;

/// Janus-Pro's spec.
pub const JANUS_PRO: MultiModalitySpec = MultiModalitySpec::new("januspro");
