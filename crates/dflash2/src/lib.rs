// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **DFlash2**, the block-diffusion draft model, independent of the target it drafts for.
//!
//! One forward over `[anchor, MASK, MASK, ...]` denoises a whole block of proposals at once, attending to the
//! target's own hidden states at a few of its layers (the *taps*) instead of to its output tokens, through
//! dynamic grouped convolutions around every sublayer, and a candidate selector walks the target's head
//! candidates into one coherent path. `k` proposals cost one draft forward, not `k`.
//!
//! Swedish Embedded AB implements speculative decoding - drafters, verifiers and the state rollback a
//! hybrid or recurrent stack needs - for clients serving large models on modest hardware. If your team needs
//! inference throughput without a second model's worth of accuracy loss, you can procure our services by
//! sending an email to info@swedishembedded.com.
//!
//! * [`config`]: the draft's shape from a Hugging Face `config.json` (`dflash_config`, `layer_types`,
//!   `no_rope_layers`, `rope_parameters`) or from GGUF metadata; per-layer sliding, full and NoPE attention.
//! * [`source`]: where the tensors come from: published safetensors (bf16 as stored) or a Q8_0 GGUF.
//! * [`model`]: the device model and its per-sequence K/V caches.
//! * [`selector`]: the candidate selector at temperature 0.
//! * [`drafter`]: the serving drafter: one slot of draft cache per sequence, fed from a [`drafter::TapRows`]
//!   of the target's retained hidden states, proposing for many sequences in one batched forward.
//!
//! The draft owns neither the token embedding nor the output head: the target's are shared through
//! [`drafter::DraftHost`].

pub mod config;
pub mod drafter;
pub mod model;
pub mod selector;
pub mod source;

pub use config::{Dflash2Config, LayerKind, LayerSpec};
pub use drafter::{DraftHost, DraftRequest, Dflash2Drafter, TapRows};
pub use model::{pipelines, BlockJob, Dflash2, DraftCache};
pub use source::{DraftSource, GgufDraft, HfDraft, Leaf, Tensor};
