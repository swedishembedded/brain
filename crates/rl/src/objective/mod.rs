// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Sampling-based training objectives built on P10's [`model::rollout`] and
//! P11's [`crate::env`] seam - GRPO first, DPO/top-K distillation follow in
//! later phases as siblings of this module.

pub mod grpo;
