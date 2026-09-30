// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Moondream's vision side: the SigLIP tower and the `proj_mlp` connector.
//!
//! The tower is NOT implemented here. It is `clip::model::ClipVision` with the
//! SigLIP stem ([`crate::config::VisionConfig::tower`]): pre-LN bidirectional
//! blocks, no class token, a biased patch conv, per-patch learned positions,
//! tanh-GELU (`gelu_approx`) and a final post-LN - the one SigLIP
//! implementation DeepSeek-VL and Janus-Pro run too. Moondream drives it
//! through `ClipVision::encode`, one attention span per crop, because its crop
//! count varies per image.
//!
//! What stays Moondream's own is the connector and the device kernel list the
//! tower, the connector and the crop-stitch pooling share.

use std::collections::HashMap;
use std::sync::OnceLock;

use clip::model::{ClipVision, PatchSource, CLIP_VISION_PIPELINES};
use gpu_core::{DeviceBuffer, Gpu};

use crate::config::VisionConfig;

/// Every kernel the vision device runs: the tower's whole list, plus
/// `adaptive_avgpool2d` for the multi-crop stitch
/// ([`crate::preprocess::build_connector_input`]). The connector's
/// matmul/bias/tanh-GELU are already in the tower's list. Everything on this
/// device is resolved by name ([`kernel`]), so the order is free.
pub fn vision_pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: OnceLock<Vec<(&'static str, &'static str)>> = OnceLock::new();
    LIST.get_or_init(|| {
        CLIP_VISION_PIPELINES.iter().copied().chain([("adaptive_avgpool2d", kernels::ADAPTIVE_AVGPOOL2D)]).collect()
    })
}

/// `name`'s pipeline index on the vision device.
pub fn kernel(g: &Gpu, name: &str) -> usize {
    g.kernel_index(name).unwrap_or_else(|| {
        panic!("moondream3: vision kernel `{name}` is not registered on this handle - build it from moondream3::vision::vision_pipelines()")
    })
}

/// The SigLIP tower over Moondream's vision weights (`ClipVision` manifest
/// names, as [`crate::import::load`] produces them), uploaded once. Weights
/// only: [`ClipVision::encode`] records a graph per crop batch.
pub fn encoder(gpu: Gpu, cfg: &VisionConfig, weights: &HashMap<String, Vec<f32>>) -> ClipVision {
    ClipVision::new_encoder_on(gpu, cfg.tower(), PatchSource::Pixels, weights)
}

/// Moondream connector: a 2-layer MLP `Linear(in→inner)` → tanh-GELU →
/// `Linear(inner→out)` mapping the `[729, 2·dim]` global‖local concat to `[729,
/// dim_text]` image tokens. Reuses matmul/bias/gelu (no new kernels). Weight keys:
/// `fc1.weight` `[inner,in]`/`fc1.bias`, `fc2.weight` `[out,inner]`/`fc2.bias`.
pub struct Connector {
    w: HashMap<String, DeviceBuffer>,
    in_dim: u32,
    inner: u32,
    out_dim: u32,
}

impl Connector {
    pub fn new(gpu: &Gpu, weights: &HashMap<String, Vec<f32>>, in_dim: u32, inner: u32, out_dim: u32) -> Connector {
        let w = weights.iter().map(|(k, v)| (k.clone(), gpu.storage_init(k, v))).collect();
        Connector { w, in_dim, inner, out_dim }
    }
    fn wb(&self, n: &str) -> &DeviceBuffer {
        self.w.get(n).unwrap_or_else(|| panic!("connector weight missing: {n}"))
    }
    /// Project `rows × in_dim` → `rows × out_dim`.
    pub fn forward(&self, g: &Gpu, rows: u32, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), (rows * self.in_dim) as usize);
        let (matmul, bias_add, gelu) = (kernel(g, "matmul"), kernel(g, "bias_add"), kernel(g, "gelu"));
        let xb = g.storage_init("cin", x);
        let h = g.storage((rows * self.inner) as u64);
        let h2 = g.storage((rows * self.inner) as u64);
        let out = g.storage((rows * self.out_dim) as u64);
        g.submit(
            &[],
            &[
                g.step(matmul, &[&xb, self.wb("fc1.weight"), &h], &[rows, self.in_dim, self.inner], rows * self.inner),
                g.step(bias_add, &[&h, self.wb("fc1.bias")], &[rows, self.inner], rows * self.inner),
                g.step(gelu, &[&h, &h2], &[rows * self.inner], rows * self.inner),
                g.step(matmul, &[&h2, self.wb("fc2.weight"), &out], &[rows, self.inner, self.out_dim], rows * self.out_dim),
                g.step(bias_add, &[&out, self.wb("fc2.bias")], &[rows, self.out_dim], rows * self.out_dim),
            ],
        );
        g.read(&out, (rows * self.out_dim) as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Rng;

    /// A tiny Moondream vision config: 4×4 grid (16 patches), dim 32, 2 heads,
    /// 2 layers.
    fn tiny() -> VisionConfig {
        VisionConfig { dim: 32, patch: 2, n_layers: 2, ff_dim: 64, n_heads: 2, crop_size: 8, max_crops: 4, overlap_margin: 1 }
    }

    /// The tower Moondream builds from its own config encodes a batch of crops
    /// into one post-LN feature row per patch - no class-token row - on a
    /// device carrying Moondream's kernel list rather than the tower's own.
    #[test]
    fn the_shared_siglip_tower_encodes_moondream_crops() {
        let cfg = tiny();
        let gpu = gpu_core::testgpu::dev(vision_pipelines());
        let w = clip::init::init_vision_weights(&cfg.tower(), 3);
        let enc = encoder(gpu, &cfg, &w);
        let n_crops = 3u32;
        let mut rng = Rng::new(3);
        let crops: Vec<f32> = (0..(n_crops * 3 * cfg.crop_size * cfg.crop_size) as usize).map(|_| rng.next_f32() - 0.5).collect();
        let out = enc.encode(n_crops, &crops);
        assert_eq!(out.len(), (n_crops * cfg.patches_per_crop() * cfg.dim) as usize);
        assert!(out.iter().all(|v| v.is_finite()) && out.iter().any(|&v| v.abs() > 1e-6));
    }

    #[test]
    fn connector_projects() {
        let gpu = gpu_core::testgpu::dev(vision_pipelines());
        let (in_dim, inner, out_dim, rows) = (48u32, 96u32, 32u32, 9u32);
        let mut rng = Rng::new(4);
        let mut r = |n: usize| (0..n).map(|_| (rng.next_f32() - 0.5) * 0.2).collect::<Vec<f32>>();
        let mut w = HashMap::new();
        w.insert("fc1.weight".into(), r((inner * in_dim) as usize));
        w.insert("fc1.bias".into(), r(inner as usize));
        w.insert("fc2.weight".into(), r((out_dim * inner) as usize));
        w.insert("fc2.bias".into(), r(out_dim as usize));
        let conn = Connector::new(&gpu, &w, in_dim, inner, out_dim);
        let x: Vec<f32> = (0..(rows * in_dim) as usize).map(|_| rng.next_f32() - 0.5).collect();
        let out = conn.forward(&gpu, rows, &x);
        assert_eq!(out.len(), (rows * out_dim) as usize);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
