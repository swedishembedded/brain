// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! 3D Gaussian Splatting for brain: scene IO (Inria `.ply`), the generic
//! device scan / radix-sort primitives (brain's first — reusable beyond
//! splatting), and the tiled fp32 rasterizer built from atomic-free WGSL
//! kernels so the same source runs on wgpu and the CPU JIT - in two
//! evaluations: the EWA splat of the original 3DGS, and exact per-ray
//! evaluation through any lens ([`types::RenderOpts::ray`]).
//!
//! The kernel list is exposed as [`PIPELINES`] plus the positional
//! [`Kernels`] resolver so a host binary can compose it after its own model
//! pipelines in a single `Gpu` (kernel `kind` indices are per-`Gpu`
//! positional).

pub mod align;
pub mod caps;
pub mod density;
pub mod env;
pub mod geometry;
pub mod init;
pub mod isp;
pub mod loss;
pub mod mcmc;
pub mod mip;
pub mod opt;
pub mod orient;
pub mod ply;
pub mod prune;
pub mod quality;
pub mod reference;
pub mod renderer;
pub mod sh;
pub mod sort;
pub(crate) mod train;
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
    ("splat_bwd_slots", kernels::SPLAT_BWD_SLOTS),
    ("splat_bwd_tile_reduce", kernels::SPLAT_BWD_TILE_REDUCE),
    ("splat_grad_reduce", kernels::SPLAT_GRAD_REDUCE),
    ("splat_project_bwd", kernels::SPLAT_PROJECT_BWD),
    ("splat_sh", kernels::SPLAT_SH),
    ("adamw", kernels::ADAMW),
    ("l1ssim_moments_h", kernels::L1SSIM_MOMENTS_H),
    ("l1ssim_map_v", kernels::L1SSIM_MAP_V),
    ("l1ssim_partials_h", kernels::L1SSIM_PARTIALS_H),
    ("l1ssim_grad_v", kernels::L1SSIM_GRAD_V),
    ("splat_ray_project", kernels::SPLAT_RAY_PROJECT),
    ("splat_ray_rasterize", kernels::SPLAT_RAY_RASTERIZE),
    ("splat_ray_bwd_slots", kernels::SPLAT_RAY_BWD_SLOTS),
    ("splat_ray_project_bwd", kernels::SPLAT_RAY_PROJECT_BWD),
    ("splat_ray_camera_grad", kernels::SPLAT_RAY_CAMERA_GRAD),
    ("splat_adam", kernels::SPLAT_ADAM),
    ("splat_activate", kernels::SPLAT_ACTIVATE),
    ("splat_geom_loss", kernels::SPLAT_GEOM_LOSS),
    ("splat_gather_ids", kernels::SPLAT_GATHER_IDS),
    ("splat_ray_bwd_tile", kernels::SPLAT_RAY_BWD_TILE),
    ("splat_env", kernels::SPLAT_ENV),
    ("splat_env_grad", kernels::SPLAT_ENV_GRAD),
    ("isp_pixel", kernels::ISP_PIXEL),
    ("isp_grid_grad", kernels::ISP_GRID_GRAD),
    ("dw_splitk_reduce", kernels::DW_SPLITK_REDUCE),
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
    pub splat_bwd_slots: usize,
    pub splat_bwd_tile_reduce: usize,
    pub splat_grad_reduce: usize,
    pub splat_project_bwd: usize,
    pub splat_sh: usize,
    pub adamw: usize,
    pub l1ssim_moments_h: usize,
    pub l1ssim_map_v: usize,
    pub l1ssim_partials_h: usize,
    pub l1ssim_grad_v: usize,
    pub splat_ray_project: usize,
    pub splat_ray_rasterize: usize,
    pub splat_ray_bwd_slots: usize,
    pub splat_ray_project_bwd: usize,
    pub splat_ray_camera_grad: usize,
    pub splat_adam: usize,
    pub splat_activate: usize,
    pub splat_geom_loss: usize,
    pub splat_gather_ids: usize,
    pub splat_ray_bwd_tile: usize,
    pub splat_env: usize,
    pub splat_env_grad: usize,
    pub isp_pixel: usize,
    pub isp_grid_grad: usize,
    pub dw_splitk_reduce: usize,
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
            splat_bwd_slots: base + 11,
            splat_bwd_tile_reduce: base + 12,
            splat_grad_reduce: base + 13,
            splat_project_bwd: base + 14,
            splat_sh: base + 15,
            adamw: base + 16,
            l1ssim_moments_h: base + 17,
            l1ssim_map_v: base + 18,
            l1ssim_partials_h: base + 19,
            l1ssim_grad_v: base + 20,
            splat_ray_project: base + 21,
            splat_ray_rasterize: base + 22,
            splat_ray_bwd_slots: base + 23,
            splat_ray_project_bwd: base + 24,
            splat_ray_camera_grad: base + 25,
            splat_adam: base + 26,
            splat_activate: base + 27,
            splat_geom_loss: base + 28,
            splat_gather_ids: base + 29,
            splat_ray_bwd_tile: base + 30,
            splat_env: base + 31,
            splat_env_grad: base + 32,
            isp_pixel: base + 33,
            isp_grid_grad: base + 34,
            dw_splitk_reduce: base + 35,
        }
    }
}
