// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Weight initialization, deterministic for a fixed seed.
//!
//! Everything a SwiGLU model shares with Qwen3 is initialized as
//! `qwen3::init_weights` does: norm gains 1, linears and the embedding
//! Normal(0, 0.02), the residual output projections Normal(0, 0.02/sqrt(2L)).
//!
//! The spectral filters are scaled so the mixer starts in the same regime as
//! a SwiGLU of hidden width `C * D` (the signal-domain units it has). A row
//! of `S` has squared norm `m/D` on average, so `S(G * h)` with `G ~
//! Normal(0, s)` and unit-variance `h` has per-sample variance `(m/D) s^2`;
//! a dense gate with fan-in `m` and std 0.02 has `m * 0.02^2`. Equal at
//! `s = 0.02 sqrt(D)`. On the output side `S^T` preserves variance, and
//! `C` channels weighted by `W ~ Normal(0, w)` give `C w^2`, against `C D
//! sigma_p^2` for the dense down projection, so `w = sigma_p sqrt(D)`.

use std::collections::HashMap;

use data::rng::Rng;

use crate::config::{Mixer, SpectralConfig};

fn is_norm_gain(name: &str) -> bool {
    name == "norm.weight" || name.ends_with("ln1.weight") || name.ends_with("ln2.weight") || name.ends_with("q_norm.weight") || name.ends_with("k_norm.weight")
}

pub fn init_weights(cfg: &SpectralConfig, seed: u64) -> HashMap<String, Vec<f32>> {
    let mut rng = Rng::new(seed);
    let std = 0.02f32;
    let proj_std = std / (2.0 * cfg.n_layers as f32).sqrt();
    let signal = match cfg.mixer {
        Mixer::Spectral { signal_len, .. } => (signal_len as f32).sqrt(),
        Mixer::SwiGlu { .. } => 1.0,
    };
    let mut w = HashMap::new();
    for (name, numel) in cfg.param_list() {
        let s = if is_norm_gain(&name) {
            w.insert(name, vec![1.0; numel]);
            continue;
        } else if name.ends_with("attn.wo.weight") || name.ends_with("mlp.down.weight") {
            proj_std
        } else if name.ends_with("mlp.gate_filter") || name.ends_with("mlp.up_filter") {
            std * signal
        } else if name.ends_with("mlp.out_filter") {
            proj_std * signal
        } else {
            std
        };
        w.insert(name, (0..numel).map(|_| rng.next_gaussian() as f32 * s).collect());
    }
    w
}
