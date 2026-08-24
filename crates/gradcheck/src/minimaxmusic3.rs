// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `check_vocoder` - the finite-difference gate on MiniMax Music 3's
//! vocoder backward (`minimaxmusic3::train::Trainer`).
//!
//! Device style (like `check_sam2`), not host-f64: the vocoder's backward
//! is composed from shipped WGSL (`audio::conv`'s existing conv/conv-transpose
//! kernels plus the new `snake1d`/`bias_grad_ncl`/`tanh_act_bwd`), so the
//! only thing worth checking is those kernels running for real, not a
//! second host implementation.

use minimaxmusic3::config::{DitConfig, VocoderConfig};
use minimaxmusic3::train::{random_weights, Trainer};

use crate::CheckModel;

/// Orphan-rule wrapper (`CheckModel` has a blanket impl over `model::Model`,
/// so a foreign type cannot implement it directly - same workaround
/// `Sam2DecoderCheck` uses).
pub struct VocoderCheck(pub Trainer);

impl CheckModel for VocoderCheck {
    fn param_names(&self) -> Vec<String> {
        self.0.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.0.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.0.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.0.read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.0.loss()
    }
    fn zero_grads(&self) {
        self.0.zero_grads();
    }
    fn backward(&self) {
        self.0.backward();
    }
}

/// Same orphan-rule wrapper as [`VocoderCheck`], over the DiT's own
/// `dit_train::Trainer`.
pub struct DitCheck(pub minimaxmusic3::dit_train::Trainer);

impl CheckModel for DitCheck {
    fn param_names(&self) -> Vec<String> {
        self.0.param_names()
    }
    fn read_weight(&self, name: &str) -> Vec<f32> {
        self.0.read_weight(name)
    }
    fn write_weight(&self, name: &str, data: &[f32]) {
        self.0.write_weight(name, data);
    }
    fn read_grad(&self, name: &str) -> Vec<f32> {
        self.0.read_grad(name)
    }
    fn loss(&self) -> f32 {
        self.0.loss()
    }
    fn zero_grads(&self) {
        self.0.zero_grads();
    }
    fn backward(&self) {
        self.0.backward();
    }
}

/// A tiny DiT: `num_layers=2, num_attention_heads=2, attention_head_dim=4`
/// (`inner_dim=8`), `rotary_dim=2`, matching
/// `crates/minimaxmusic3::config::DitConfig::tiny`. Every structural
/// feature the real DiT has is present (partial RoPE, bidirectional
/// attention, the fused gated FFN, the prepended timestep token) - only the
/// sizes are small.
pub fn tiny_dit_config() -> DitConfig {
    DitConfig::tiny()
}

pub fn check_dit(seed: u64) -> crate::Report {
    let cfg = tiny_dit_config();
    let w = minimaxmusic3::dit_train::random_weights(&cfg, seed);
    let length = 3usize;
    let mut r = data::rng::Lcg::new(seed ^ 0x1DE5_1234);
    let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
    let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
    let timestep = 0.4f32;
    let target = r.vec_scaled(cfg.in_channels as usize * length, 0.3);

    let trainer = minimaxmusic3::dit_train::Trainer::new(cfg, &w, latents, condition, timestep, length, target);
    let model = DitCheck(trainer);
    crate::directional_check(&model, 5e-3, 4, seed)
}

/// [`check_dit`] on the shared POOLED TEST DEVICE, at dims chosen to clear
/// `model::block::pick_gemm`'s tile-fill crossover (`m >= 8` output rows,
/// `n >= 128` output cols) - so the backward really dispatches
/// `matmul_dx_reg`/`matmul_dw_reg`, and the forward `matmul_reg3`/
/// `matmul_gemv`, instead of the naive `@opt 2` trio.
///
/// The same reason `check_t5_tiled` exists beside `check_t5`, and it is doing
/// two jobs at once. `check_dit` runs on `Gpu::new_cpu` by construction, so it
/// covers neither the register-tiled kernels nor a real card; this one covers
/// both, and on `backend-cpu` (`BRAIN_DEVICE=cpu`) it covers a THIRD
/// implementation - that backend intercepts those kernel names and routes them
/// to its AVX2 GEMMs rather than JIT-compiling them, which is a different code
/// path again from the WGSL the GPU runs.
///
/// `num_layers = 1`: this gate is about which GEMM kernel runs, and the block
/// stack's depth is already covered by `check_dit`.
pub fn check_dit_tiled(seed: u64) -> crate::Report {
    let cfg = DitConfig {
        in_channels: 4,
        condition_dim: 6,
        num_layers: 1,
        num_attention_heads: 2,
        attention_head_dim: 64, // inner_dim = 128
        ff_inner_dim: 128,
        rotary_dim: 32,
        fourier_embedding_dim: 8,
    };
    let w = minimaxmusic3::dit_train::random_weights(&cfg, seed);
    let length = 8usize; // rows = length + 1 = 9
    let mut r = data::rng::Lcg::new(seed ^ 0x71_1ED0);
    let latents = r.vec_scaled(cfg.in_channels as usize * length, 0.3);
    let condition = r.vec_scaled(length * cfg.condition_dim as usize, 0.3);
    let target = r.vec_scaled(cfg.in_channels as usize * length, 0.3);

    let gpu = gpu_core::testgpu::dev(minimaxmusic3::dit_train::PIPELINES);
    let trainer = minimaxmusic3::dit_train::Trainer::new_on(gpu, cfg, &w, latents, condition, 0.4, length, target);
    crate::directional_check(&DitCheck(trainer), 5e-3, 4, seed)
}

/// A tiny vocoder: `latent_channels=4` (2 folded stereo streams of 2),
/// `decoder_hidden_dim=16` halving across 2 upsample stages to `4`,
/// matching `crates/minimaxmusic3::config::VocoderConfig::tiny`. Every
/// structural feature the real vocoder has is present (weight-normed-shape
/// convs already folded at random-weight construction time, the
/// ConvTranspose1d upsample, all 3 dilations of the residual unit, the
/// final tanh) - only the sizes are small.
pub fn tiny_config() -> VocoderConfig {
    VocoderConfig::tiny()
}

pub fn check_vocoder(seed: u64) -> crate::Report {
    let cfg = tiny_config();
    let w = random_weights(&cfg, seed);
    let (batch, length) = (1, 4);
    let mut r = data::rng::Lcg::new(seed ^ 0x5A5A_1234);
    let latents = r.vec_scaled(batch * cfg.latent_channels as usize * length, 0.5);
    let out_len = length * cfg.upsampling_ratios.iter().product::<u32>() as usize;
    let target = r.vec_scaled(batch * 2 * out_len, 0.5);

    let trainer = Trainer::new(cfg, &w, latents, batch, length, target);
    let model = VocoderCheck(trainer);
    crate::directional_check(&model, 5e-3, 4, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocoder_backward_matches_finite_differences() {
        let report = check_vocoder(1);
        report.print();
        assert!(report.all_within(4e-3, 8e-2), "vocoder gradcheck failed: {:?}", report.failures(4e-3, 8e-2));
        assert!(report.dead_gradients().is_empty(), "dead gradients: {:?}", report.dead_gradients());
    }

    #[test]
    fn dit_backward_matches_finite_differences() {
        let report = check_dit(1);
        report.print();
        assert!(report.all_within(4e-3, 8e-2), "DiT gradcheck failed: {:?}", report.failures(4e-3, 8e-2));
        assert!(report.dead_gradients().is_empty(), "dead gradients: {:?}", report.dead_gradients());
    }
}
