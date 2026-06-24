// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Browser (wasm32 + WebGPU) inference build for the PID transformer.
//!
//! The native CLI lives in the `brain` binary (`brain-cli`) and does NOT depend
//! on this crate, so the native build and its parity gates are unaffected. The
//! shared engine (`gpu_core`, `checkpoint`, `optim`, `paramstore`, `pid`) is
//! pulled in as ordinary workspace crates; only the `#[wasm_bindgen]` entry
//! point ([`web`]) is defined here, gated to `wasm32 + webgpu`. On every other
//! target this crate compiles to nothing.
//!
//! Build for the browser with:
//!   cargo build --release --target wasm32-unknown-unknown -p brain-web --features webgpu

#[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
mod web;
