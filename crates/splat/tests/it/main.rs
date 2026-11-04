// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
//! The crate's integration tests, linked as one binary rather than one per
//! file: each links the whole dependency tree, so a binary per file
//! multiplied build time and disk for nothing. A test here must not change
//! process-wide state (the current directory, environment variables): the
//! others run beside it.

mod s10_align;
mod s11_pose;
mod s12_scale_invariance;
mod s13_depth_supervision;
mod s14_mcmc;
mod s15_geometry_stability;
mod s16_photometric_loss;
mod s17_camera_model;
mod s18_geometry_regularizers;
mod s19_hybrid_density;
mod s1_project_naive;
mod s20_sh_schedule;
mod s21_coarse_to_fine;
mod s22_sparse_start;
mod s23_ray_renderer;
mod s24_environment;
mod s25_transients;
mod s26_device_memory;
mod s27_diagnostics;
mod s28_free_space;
mod s29_large_frame;
mod s2_tiled_parity;
mod s5_bwd_fit;
mod s6_prune;
mod s6_reconstruction_quality;
mod s7_rasterizer_fidelity;
mod s8_densification;
mod s9_orient;
