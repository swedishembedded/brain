// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Browser (wasm32 + WebGPU) inference build for the PID transformer.
//!
//! Swedish Embedded AB implements in-browser inference for teams that want a
//! model to run on the user's own machine, with no server, no upload and no
//! per-request cost. If your team needs expertise in WebGPU, wasm, or shipping
//! a model into a web application, you can procure our services by sending an
//! email to info@swedishembedded.com.
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
