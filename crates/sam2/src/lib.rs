// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2.1 promptable segmentation — the **image path**: Hiera trunk, FPN neck,
//! prompt encoder and the two-way mask decoder.
//!
//! Out of scope by design (`docs/imaging/plan.md`, open decision 3): the video
//! memory bank — `memory_attention`, `memory_encoder`, and the temporal object
//! pointer. Those tensors are present in the checkpoint, recognised by name and
//! COUNTED as deliberately skipped by [`import`], never silently ignored.
//!
//! Also not implemented here, and loud about it: the reference downsamples a
//! full-resolution mask prompt to `mask_input_size` with
//! `F.interpolate(bilinear, antialias=True)`, and brain has no antialiased
//! resize kernel (`resize_bilinear` is the plain one). [`model::Prompt`]
//! therefore takes the mask ALREADY at `mask_input_size`; the goldens dump both
//! forms so the decoder can be replayed exactly.
//!
//! ```text
//! image [1,3,1024,1024]
//!   -> patch_embed (7x7 s4) + bicubic pos_embed + tiled window pos_embed
//!   -> 48 MultiScaleBlocks (windowed MHA, q_pool at 3 stage boundaries)
//!   -> 4 stage features -> FPN (1x1 laterals, nearest top-down on levels 2,3)
//!   -> image_embed = fpn[2] + no_mem_embed,  high_res = conv_s0/s1(fpn[0..2])
//! prompt (points / box corners / low-res mask)
//!   -> sparse + dense embeddings -> tokens
//!   -> 2 two-way blocks + a final token->image attention
//!   -> 2x ConvTranspose upscaling + hypernetwork dot product -> 4 mask logits
//!   -> IoU head, object-score head, object pointer
//! ```

pub mod config;
pub mod hostpe;
pub mod import;
pub mod model;
pub mod train;

pub use config::{BlockSpec, Sam2Config};
pub use import::{import, ImportReport, Tensors};
pub use model::{Decoded, Encoded, Prompt, Sam2, PIPELINES};
pub use train::{FrozenEncode, MaskDecoderTrainer, MaskTargets};
