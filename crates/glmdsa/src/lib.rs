// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GLM-5.2 (`glm_moe_dsa`) decoder for brain: pure Rust + WGSL, fp32, on the
//! shared `gpu_core` engine (wgpu or the native CPU JIT).
//!
//! Architecture (see `resources/glm/`):
//!   * **MLA** (Multi-head Latent Attention) — low-rank q/kv with a decoupled
//!     nope/rope head split and interleaved RoPE on the rope slice.
//!   * **MoE** — sigmoid `noaux_tc` router (per-expert selection bias), a shared
//!     always-on expert, and a `first_k_dense_replace` dense→MoE layer schedule.
//!   * **DSA** sparse indexer + IndexShare and **MTP** are added in later phases;
//!     with `index_topk >= block_size` the indexer is a no-op and attention is
//!     exact dense MLA (the regime tiny models / tests run in).
//!
//! `model.rs` holds the forward/backprop dispatch graph, `config.rs` the
//! architecture + parameter layout, `import.rs` the HuggingFace weight import.

pub mod caps;
pub mod config;
pub mod distill;
pub mod import;
pub mod init;
pub mod model;
pub mod sample;

pub use config::GlmConfig;
pub use init::init_weights;
pub use model::{Glm, IGNORE};
