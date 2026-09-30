// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **SAM-1 / ViTDet ViT-B tower** -- the front half of DeepSeek-OCR's
//! DeepEncoder, and the first consumer of the decomposed relative-position
//! bias kernels.
//!
//! Image in, `[1, compress_out, gh', gw']` out, `(gh', gw')` being the
//! compressor grid (`grid/4`, or `resize/4` with a neck resize):
//!
//! ```text
//! Conv2d(3, C, k=patch, s=patch) -> [C, gh, gw] -> row-major [gh*gw, C]
//!   + pos_embed
//! N x pre-LN ViT block, windowed (zero-padded partition) or global:
//!     x += proj(attn(qkv(norm1(x))) + decomposed_rel_pos)
//!     x += fc2(gelu(fc1(norm2(x))))
//! neck:      Conv2d(C, n, 1) -> LayerNorm2d -> Conv2d(n, n, 3, p=1) -> LayerNorm2d
//! resize:    bilinear, align_corners = false (optional, `neck_resize`)
//! compress:  Conv2d(n, mid, 3, s=2, p=1) -> Conv2d(mid, out, 3, s=2, p=1)
//! HD branch (optional, `hd_branch`; DeepSeek-VL's `sam_b_downsample`):
//!     out += hd_alpha * compress(resize(neck_hd(first global block's output)))
//!     -- the SAME compressor weights, so their gradient sums both uses
//! ```
//!
//! Everything is composed from `model::vit`, `model::block` and
//! `vision::blocks`; **this crate adds no kernel and no shared block.** See
//! [`model`]'s module header for what each piece contributes and for the
//! window-padding contract.
//!
//! ## What is and is not claimed
//!
//! * The forward AND the full analytic backward exist and are gradient-checked
//!   end to end (`tests/gradcheck.rs`), on the GPU and on the CPU JIT.
//! * The upstream PyTorch tensor names of every release that ships this tower
//!   (DeepSeek-OCR, DeepSeek-VL's high-resolution tower) are mapped in one
//!   place, [`hf`]; DeepSeek-VL's tower is parity-gated against the pinned
//!   reference module on the real checkpoint (`tests/deepseek_vl_parity.rs`)
//!   and its HD branch gradient-checked (`gradcheck::sam_hd`).
//! * The tensor manifest is the `vision.sam.*` half of what `brain-gguf`'s
//!   mmproj loader emits, and [`import`] reuses that loader's own classifier --
//!   no second name table. A header-only coverage gate runs against the real
//!   shipped checkpoint when it is present.
//! * **Not** done: `pos_embed` is used at its native grid only. Real
//!   multi-resolution SAM bicubically interpolates it to the input's grid.
//!   `TODO(deepseek-ocr): pos-embed interpolation for grids other than the
//!   checkpoint's.`
//! * **Not** done: batch > 1. Every span offset would have to stay 256 B
//!   aligned across the batch stride as well, which `grid_h*grid_w*d_model`
//!   generally is not.
//! * **Not** done: no serving surface of its own, no CLI, no INT8.

pub mod config;
pub mod hf;
pub mod import;
pub mod init;
pub mod model;

pub use config::SamViTConfig;
pub use init::{init_dense, init_weights};
pub use model::{SamEncoder, PIPELINES};
