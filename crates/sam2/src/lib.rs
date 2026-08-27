// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2.1 promptable segmentation: the **image path** (Hiera trunk, FPN neck,
//! prompt encoder, two-way mask decoder) and the **video path** (the temporal
//! memory bank that makes a mask follow the object through a clip).
//!
//! The video half - `memory_attention`, `memory_encoder`, the object-pointer
//! temporal encoding and the propagation loop - lives in [`video`]. Import runs
//! at one of two [`import::Scope`]s: at `Scope::Image` the video tensors are
//! recognised by name and COUNTED as deliberately skipped, never silently
//! ignored; at `Scope::Video` nothing is skipped and an unmatched key on
//! EITHER side is an error naming it.
//!
//! Also not implemented here, and loud about it: the reference downsamples a
//! full-resolution mask prompt to `mask_input_size` with
//! `F.interpolate(bilinear, antialias=True)`, and brain has no antialiased
//! resize kernel (`resize_bilinear` is the plain one). [`model::Prompt`]
//! therefore takes the mask ALREADY at `mask_input_size`; the goldens dump both
//! forms so the decoder can be replayed exactly.
//!
//! The serving contract is met by [`caps`] (the `segment`
//! `capability::Provider`), `crates/cli/src/resident_sam2.rs` (the residency
//! adapter, `BRAIN_SAM2_WEIGHTS`, with a genuine per-image `run_batch`) and
//! `examples/vision/`.
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
//!
//! video: frame t, given the memory of frames < t
//!   fpn[2] (NLC) + 0.1*possine
//!     -> 4 memory-attention layers: RoPE self-attn, RoPE cross-attn into
//!        [maskmem slabs | object pointers], ReLU MLP
//!     -> pix_feat_with_mem  (replaces image_embed in the mask decoder)
//!   best mask -> sigmoid*20-10 -> memory encoder (stride-16 mask conv +
//!        1x1 pix_feat proj + 2 ConvNeXt blocks + 1x1 -> mem_dim)
//!     -> this frame's memory entry
//! ```

pub mod caps;
pub mod config;
pub mod hostpe;
pub mod import;
pub mod maskseq;
pub mod model;
pub mod train;
pub mod video;

pub use config::{BlockSpec, Sam2Config};
pub use import::{import, import_scoped, ImportReport, Scope, Tensors};
pub use maskseq::{MaskSeq, Polarity};
pub use model::{Decoded, Encoded, Prompt, Sam2, PIPELINES};
pub use train::{FrozenEncode, MaskDecoderTrainer, MaskTargets};
pub use video::{MemoryEntry, TrackStep, Tracker, VideoConsts};
