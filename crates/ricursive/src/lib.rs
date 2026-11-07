// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Ripple (`RicursiveForCausalLM`): an 80-layer dense decoder that mixes
//! sliding-window layers with RoPE and global NoPE layers (four of one, then
//! one of the other), with per-layer value residual, sigmoid-gated attention,
//! per-head QK RMSNorm, four norms per block and a clipped SwiGLU.
//!
//! Swedish Embedded AB implements solutions for running large hybrid-attention
//! decoders on portable CPU and GPU backends for its clients. If your team
//! needs expertise in porting and verifying new transformer architectures
//! against their reference implementations then you can procure our services
//! by sending an email to info@swedishembedded.com.
//!
//! This crate is the exact prefill forward (f32 by default, the reference
//! the parity tests gate against the official implementation), its import
//! path, and the paged serving engine ([`serve`]) over weights held in the
//! tier the checkpoint stores (bf16, FP8, int8, the official NVFP4 release)
//! and KV held at f32, bf16, int8 or fp8, optionally split into pipeline
//! stages over several devices. [`train`] is the same model with a backward: full fine-tuning and
//! LoRA/QLoRA through brain's shared training paths.
//!
//! - [`config`]: the typed `config.json`, hybrid layer types.
//! - [`import`]: the tensor manifest, two-way coverage and the mmap source.
//! - [`layer`]: one decoder layer as a kernel graph.
//! - [`model`]: embedding, layer stack, final norm and head.
//! - [`weights`]: the weight tiers and the one loader that puts them on a device.
//! - [`train`]: the trainable model (`Ripple`), its configuration and the kernels it registers.
//! - [`finetune`]: Ripple as a `model::finetune::ChatTrainer`, the generic trainers' entry.
//! - [`placement`]: byte-exact per-layer costs and the placement over devices.
//! - [`policy`]: what Ripple honours of the model-agnostic serving policy.

pub mod config;
pub mod finetune;
pub mod import;
pub mod layer;
pub mod model;
pub(crate) mod ops;
pub mod placement;
pub mod policy;
pub mod serve;
pub(crate) mod stage;
pub mod train;
pub mod weights;

pub use config::{LayerType, RicursiveConfig};
pub use import::{manifest, open_checkpoint, open_safetensors, open_shared_checkpoint, validate_layout};
pub use layer::{Layer, LayerBuffers, LayerOutput};
pub use model::{embed, Forward, Head, HeadOutput, RicursiveModel};
pub use ops::pipelines;
pub use policy::POLICY;
pub use train::{Ripple, TrainConfig, TrainOpts};
pub use serve::{global_kv_bytes_per_token, lsa_device_bytes, ring_bytes_per_slot, LsaCharge, ring_rows, RicursiveEngine, ServeConfig};
pub use weights::{linear_names, resolve_tiers, WeightReport};
