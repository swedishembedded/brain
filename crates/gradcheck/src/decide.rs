// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference check for the decision model's encoder backward.
//!
//! Its own module rather than a `check_*` beside the decoder LMs, for the same
//! reason [`crate::florence2`] has one: the encoder has no vocabulary and no
//! scalar loss of its own, so the checker needs a small objective wrapper. The
//! objective is a fixed random linear readout of the final hidden states,
//! `L = sum(hidden * w)`, whose gradient is exactly `w`.
//!
//! That isolates the encoder. A real decision loss arrives with the head and is
//! checked against that; a failure here therefore cannot be blamed on the
//! objective, only on the adjoint.

use decide::config::EncoderConfig;
use decide::model::{Encoder, PIPELINES};

use crate::CheckModel;

/// The encoder plus the objective's fixed seed.
pub struct Probe {
    enc: Encoder,
    /// `dL/d(hidden)`, which for this objective IS the readout vector.
    w: Vec<f32>,
}

impl CheckModel for Probe {
    fn param_names(&self) -> Vec<String> {
        self.enc.cfg.tensor_manifest().into_iter().map(|(n, _)| n).collect()
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.enc.read_weight(name)
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        self.enc.set_weight(name, data);
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.enc.read_grad(name)
    }

    fn loss(&self) -> f32 {
        self.enc.forward();
        // f64 accumulation: the sum runs over `rows*H` terms and the checker
        // DIFFERENCES two of these, so summing in f32 would put rounding noise
        // straight into the numeric derivative it is compared against.
        self.enc.hidden().iter().zip(&self.w).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() as f32
    }

    fn zero_grads(&self) {
        self.enc.zero_grads();
    }

    fn backward(&self) {
        self.enc.backward(&self.w);
    }
}

/// Spans of DIFFERENT lengths, so the packing is part of what is checked: a
/// backward that quietly assumed one length per row would pass on a uniform
/// batch and fail on every real request.
const SPANS: &[(u32, u32)] = &[(0, 7), (7, 5), (12, 6)];

/// Build the probe on a tiny config with a fixed batch already set.
pub fn probe(seed: u64) -> Probe {
    let cfg = EncoderConfig::tiny();
    let init = decide::init::init_weights(&cfg, seed);
    let rows: u32 = SPANS.iter().map(|&(_, l)| l).sum();
    let max_span = SPANS.iter().map(|&(_, l)| l).max().expect("SPANS is not empty");
    let gpu = gpu_core::testgpu::dev(PIPELINES);
    let mut enc = Encoder::new_train_on(gpu, cfg.clone(), rows, max_span, &init);

    let ids: Vec<u32> = (0..rows).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    // Both segment ids are live: the decision model's slot role is segment 1,
    // and a backward that never scatters into row 1 of that table would
    // otherwise look correct.
    let types: Vec<u32> = SPANS
        .iter()
        .enumerate()
        .flat_map(|(i, &(_, l))| std::iter::repeat_n((i % 2) as u32, l as usize))
        .collect();
    enc.set_batch(&ids, &types, SPANS);

    let mut rng = data::rng::Lcg::new(seed ^ 0xA5A5_1234);
    let w = (0..(rows * cfg.d_model)).map(|_| rng.signed()).collect();
    Probe { enc, w }
}

/// Directional finite-difference check over every encoder parameter.
///
/// `eps = 5e-3` on f32 weights: smaller drowns the difference in rounding,
/// larger lets the function's curvature into the secant - the same value the
/// other encoder checks in this workspace use.
pub fn check_decide(seed: u64) -> crate::Report {
    let p = probe(seed);
    crate::directional_check(&p, 5e-3, 4, seed ^ 0x1234)
}
