// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sampling-based training objectives built on P10's [`model::rollout`] and
//! P11's [`crate::env`] seam - GRPO, DPO and top-K distillation as siblings
//! of this module, all reducing to [`model::Batch::LmWeighted`].

pub mod distill;
pub mod dpo;
pub mod grpo;
