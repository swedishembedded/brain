// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 3D Gaussian Splatting for brain: scene IO (Inria `.ply`), the generic
//! device scan / radix-sort primitives (brain's first — reusable beyond
//! splatting), and the tiled fp32 rasterizer built from atomic-free,
//! barrier-free WGSL kernels so the same source runs on wgpu and the CPU JIT.
//!
//! The kernel list is exposed as [`PIPELINES`] plus the positional
//! [`Kernels`] resolver so a host binary can compose it after its own model
//! pipelines in a single `Gpu` (kernel `kind` indices are per-`Gpu`
//! positional).

pub mod opt;
pub mod ply;
pub mod reference;
pub mod renderer;
pub mod sort;
pub mod types;

/// WGSL kernels this crate dispatches, in [`Kernels`] order. Pass to
/// `Gpu::new(..)` either alone or appended to a model's own pipeline list
/// (then resolve indices with [`Kernels::at`] using the base offset).
pub const PIPELINES: &[(&str, &str)] = &[
    ("scan_block", kernels::SCAN_BLOCK),
    ("scan_add", kernels::SCAN_ADD),
    ("sort_hist", kernels::SORT_HIST),
    ("sort_scatter", kernels::SORT_SCATTER),
    ("splat_project", kernels::SPLAT_PROJECT),
    ("splat_naive", kernels::SPLAT_NAIVE),
    ("splat_tile_count", kernels::SPLAT_TILE_COUNT),
    ("splat_emit", kernels::SPLAT_EMIT),
    ("splat_tile_ranges", kernels::SPLAT_TILE_RANGES),
    ("splat_rasterize", kernels::SPLAT_RASTERIZE),
    ("splat_pack_rgba8", kernels::SPLAT_PACK_RGBA8),
    ("splat_bwd_count", kernels::SPLAT_BWD_COUNT),
    ("splat_bwd_emit", kernels::SPLAT_BWD_EMIT),
    ("splat_bwd_keys", kernels::SPLAT_BWD_KEYS),
    ("splat_grad_reduce", kernels::SPLAT_GRAD_REDUCE),
    ("splat_project_bwd", kernels::SPLAT_PROJECT_BWD),
    ("splat_unpack", kernels::SPLAT_UNPACK),
    ("adamw", kernels::ADAMW),
];

/// Positional kernel indices into a `Gpu` whose pipeline list contains
/// [`PIPELINES`] starting at `base`.
#[derive(Clone, Copy)]
pub struct Kernels {
    pub scan_block: usize,
    pub scan_add: usize,
    pub sort_hist: usize,
    pub sort_scatter: usize,
    pub splat_project: usize,
    pub splat_naive: usize,
    pub splat_tile_count: usize,
    pub splat_emit: usize,
    pub splat_tile_ranges: usize,
    pub splat_rasterize: usize,
    pub splat_pack_rgba8: usize,
    pub splat_bwd_count: usize,
    pub splat_bwd_emit: usize,
    pub splat_bwd_keys: usize,
    pub splat_grad_reduce: usize,
    pub splat_project_bwd: usize,
    pub splat_unpack: usize,
    pub adamw: usize,
}

impl Kernels {
    /// Resolve indices for a `Gpu` built with [`PIPELINES`] at offset `base`
    /// (0 when the splat pipelines are the whole list).
    pub fn at(base: usize) -> Kernels {
        Kernels {
            scan_block: base,
            scan_add: base + 1,
            sort_hist: base + 2,
            sort_scatter: base + 3,
            splat_project: base + 4,
            splat_naive: base + 5,
            splat_tile_count: base + 6,
            splat_emit: base + 7,
            splat_tile_ranges: base + 8,
            splat_rasterize: base + 9,
            splat_pack_rgba8: base + 10,
            splat_bwd_count: base + 11,
            splat_bwd_emit: base + 12,
            splat_bwd_keys: base + 13,
            splat_grad_reduce: base + 14,
            splat_project_bwd: base + 15,
            splat_unpack: base + 16,
            adamw: base + 17,
        }
    }
}
