// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it; `conformance`, `continual_study`, `distill_full_kl`, `distill_gradcheck`, `dpo_gradcheck`, `dpo_objective`, `grpo_fixes`, `grpo_gradcheck`, `grpo_objective`, `grpo_qwen35`, `improve_cycle`, `improve_cycle_qwen35`, `mixture_anchor_regression`, `qwen3_fit_weighted` keep binaries of their own.

#[cfg(feature = "qwen3")]
#[path = "../arch_builds/mod.rs"]
mod arch_builds;
#[cfg(feature = "qwen3")]
#[path = "../grpo_common/mod.rs"]
mod grpo_common;

#[cfg(feature = "qwen3")]
mod chat_trainer_bases;
#[cfg(feature = "qwen3")]
mod document_study;
mod external_policy_gradient;
mod external_policy_gradient_gradcheck;
mod external_policy_gradient_gradcheck_qwen3;
mod lazy_scoring;
mod self_distill;
#[cfg(feature = "qwen3")]
mod lora_acceptance;
#[cfg(feature = "qwen3")]
mod objective_fits;
#[cfg(feature = "qwen3")]
mod reader_learner;
#[cfg(feature = "qwen3")]
mod served_conformance;
