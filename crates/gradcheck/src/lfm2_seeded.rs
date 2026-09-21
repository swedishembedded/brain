// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Finite-difference check for LFM2's SEEDED backward
//! ([`lfm2::model::Lfm::seed_buf`]/`backward_seeded`) - the entry point an
//! external objective (contrastive fine-tuning, or anything else) uses
//! instead of the checkpoint's own masked-LM CE loss.
//!
//! Same isolation technique [`crate::decide`] uses for its own headless
//! encoder: a fixed random linear readout of the final hidden states,
//! `L = sum(xn_final * w)`, whose gradient is exactly `w` - broadcast to
//! every row here (mirroring what a real caller pooling over rows before
//! computing its own loss would do: mean pooling's own adjoint is "divide by
//! n and scatter to every row", and a UNIFORM per-row `w` exercises exactly
//! that scatter without hand-deriving mean-pool's own adjoint into the
//! fixture).
//!
//! A failure here means the trunk-backward refactor
//! ([`lfm2::model::Lfm`]'s `trunk_backward_steps`, shared with the
//! CE-seeded path [`crate::check_lfm`] already gates) is wrong on the
//! SEEDED entry specifically - since the CE path's own numbers are already
//! gated separately, a pass on both together is the real end-to-end proof
//! this milestone needs: seeding from outside the model reaches every
//! parameter the CE path does, unchanged.

use lfm2::model::Lfm;
use lfm2::LfmConfig;

use crate::CheckModel;

pub struct Probe {
    model: Lfm,
    rows: u32,
    d_model: u32,
    /// `dL/d(xn_final)` for every row - identical across rows, the mean-pool
    /// broadcast a real pooling caller would produce.
    w: Vec<f32>,
}

impl CheckModel for Probe {
    fn param_names(&self) -> Vec<String> {
        self.model.cfg.param_list().into_iter().map(|(n, _)| n).collect()
    }

    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.model.read_weight(name)
    }

    fn write_weight(&self, name: &str, data: &[f32]) {
        self.model.write_weight(name, data);
    }

    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.model.read_grad(name)
    }

    fn loss(&self) -> f32 {
        self.model.forward();
        let hidden = self.model.read_hidden();
        debug_assert_eq!(hidden.len(), (self.rows * self.d_model) as usize);
        // f64 accumulation: summed over rows*d_model terms and DIFFERENCED
        // against a numeric perturbation - see `decide::Probe::loss`'s own
        // identical rationale.
        hidden.iter().zip(&self.w).map(|(a, b)| (*a as f64) * (*b as f64)).sum::<f64>() as f32
    }

    fn zero_grads(&self) {
        self.model.zero_grads();
    }

    fn backward(&self) {
        self.model.gpu.write_f32(self.model.seed_buf(), &self.w);
        self.model.backward_seeded();
    }
}

/// Build the probe on a tiny trainable config with a fixed batch already set.
pub fn probe(seed: u64) -> Probe {
    let cfg = LfmConfig::tiny();
    let (rows, d_model) = (12u32, cfg.d_model);
    let init = lfm2::init::init_weights(&cfg, seed);
    let model = Lfm::new_train(cfg.clone(), 2, 6, &init);

    let x: Vec<u32> = (0..rows).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    // Targets are irrelevant on the seeded path (no CE runs at all), but
    // `set_batch` still wants a same-shaped array.
    let y: Vec<u32> = vec![lfm2::model::IGNORE; rows as usize];
    model.set_batch(&x, &y);

    let mut rng = data::rng::Lcg::new(seed ^ 0xA5A5_5EED);
    let per_row: Vec<f32> = (0..d_model).map(|_| rng.signed()).collect();
    let w: Vec<f32> = (0..rows).flat_map(|_| per_row.iter().copied()).collect();
    Probe { model, rows, d_model, w }
}

/// Directional finite-difference check over every LFM2 parameter, through
/// the SEEDED entry point.
///
/// `eps = 5e-3` on f32 weights - the same value every other encoder check in
/// this workspace uses (see `crate::decide::check_decide`'s own doc for why).
pub fn check_lfm_seeded(seed: u64) -> crate::Report {
    let p = probe(seed);
    crate::directional_check(&p, 5e-3, 4, seed ^ 0x5EED)
}
