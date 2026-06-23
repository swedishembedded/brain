// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Two transformers, one crate, wgpu/Vulkan acceleration.
//!
//!   * `moe`  — sparse-MoE Transformer (RMSNorm/RoPE/top-k experts). See `moe`
//!     (inference) and `train` (training + PyTorch validation).
//!   * `pid`  — event/effect control transformer (LayerNorm, learned position
//!     embeddings, dense SwiGLU, biased linears). See `pid` / `pid_data` / `cli`.
//!
//! Shared infrastructure: `gpu` (device + dispatch), `checkpoint` (weights I/O),
//! `paramstore` (weight/grad/Adam buffers), `optim` (AdamW + grad clip).
//!
//! Subcommands:
//!   moe [--weights F --prompt ... --max-new N]   # MoE inference (default)
//!   moe train|eval|validate [...]                # MoE training / eval / parity
//!   moe pid validate <ref.bin>                   # PID single-step parity gate
//!   moe pid stream   <stream.bin>                # PID fixed-data multi-step check
//!   moe pid train [--steps --effective-batch --mem-budget --seq-len ...]
//!   moe pid rollout --weights F                  # closed-loop generalization report

// The native CLI. None of these modules (training, MoE, the CLI, fs/thread code)
// are part of the browser build; the wasm inference path lives in the lib crate
// (`src/lib.rs` -> `web`). Gating the binary to non-wasm lets a
// `--target wasm32` check of the whole crate succeed without compiling the
// native-only binary, while leaving the native build byte-for-byte unchanged.
#[cfg(not(target_arch = "wasm32"))]
mod checkpoint;
#[cfg(not(target_arch = "wasm32"))]
mod cli;
#[cfg(not(target_arch = "wasm32"))]
mod gpu;
#[cfg(not(target_arch = "wasm32"))]
mod moe;
#[cfg(not(target_arch = "wasm32"))]
mod optim;
#[cfg(not(target_arch = "wasm32"))]
mod paramstore;
#[cfg(not(target_arch = "wasm32"))]
mod pid;
#[cfg(not(target_arch = "wasm32"))]
mod pid_data;
#[cfg(not(target_arch = "wasm32"))]
mod train;
#[cfg(feature = "vulkan-coopmat")]
mod vulkan;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let argv: Vec<String> = std::env::args().collect();
    match argv.get(1).map(|s| s.as_str()) {
        Some("pid") => cli::run_pid(&argv[2..]),
        Some("validate") => {
            let path = argv.get(2).map(|s| s.as_str()).unwrap_or("../train_ref.bin");
            train::validate(path);
        }
        Some("train") => moe::run_train(&argv[2..]),
        Some("eval") => moe::run_eval(&argv[2..]),
        _ => moe::run_generate(),
    }
}

// Stub entry for wasm so the `moe` bin target still has a `main`; never run.
#[cfg(target_arch = "wasm32")]
fn main() {}
