// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Library crate for the tiny sparse-MoE / PID transformer.
//!
//! Its sole purpose is the **browser (wasm32 + WebGPU) inference build**. The
//! native CLI lives entirely in the `moe` binary (`src/main.rs`) and does NOT
//! depend on this lib, so the native build and its parity gates are unaffected.
//!
//! To avoid re-compiling (and re-running, under `cargo test`) the binary's
//! modules a second time through this lib on native, the whole module surface is
//! gated to `wasm32`. On native this lib compiles to nothing.
//!
//! Only the modules needed for single-inference are pulled in. Training,
//! evaluation, the MoE model, and the native CLI are intentionally excluded from
//! the wasm surface (they use threads / `std::fs` / blocking GPU readback).
//!
//! Build for the browser with:
//!   cargo build --release --target wasm32-unknown-unknown --features webgpu
//! or via `wasm-pack build --target web -- --features webgpu`.

// The inference subset, shared with the native binary's own `mod` declarations.
// Pure-compute / GPU-plumbing modules; their non-test code has no fs/thread deps.
#[cfg(target_arch = "wasm32")]
pub mod checkpoint;
#[cfg(target_arch = "wasm32")]
pub mod gpu;
#[cfg(target_arch = "wasm32")]
pub mod optim;
#[cfg(target_arch = "wasm32")]
pub mod paramstore;
#[cfg(target_arch = "wasm32")]
pub mod pid;
#[cfg(target_arch = "wasm32")]
pub mod pid_data;

// The wasm-bindgen entry point.
#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
pub mod web;
