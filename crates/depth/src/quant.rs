// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-layer activation statistics for the INT8 quantization decision.
//!
//! This is the "measure, then decide" part, and it runs with NO NPU and NO
//! OpenVINO — entirely inside brain's engine on `--device cpu`. QuartDepth reports
//! that depth models are quant-sensitive in the DECODER, not the attention (the
//! opposite of the usual ViT-PTQ intuition), so before building any quantization
//! machinery we measure whether that holds for ZipDepth: the per-layer
//! `outlier_ratio = absmax / p99.99`. A layer whose activations have a long tail
//! (ratio >> 1) loses precision under per-tensor symmetric INT8, because the scale
//! is set by the outlier and the bulk of the distribution collapses into a few
//! codes. If ZipDepth's decoder ratios dwarf its encoder's, INT8 there is risky and
//! an FP decoder is worth it; if not, the whole ablation is unnecessary.
//!
//! Coverage: every conv that flows through `vision::Conv` is tapped at its input,
//! including the Norm::None raw convs (the fusion projections, the head, GCB's raw
//! convs). ChannelAttention (SE) is the one exception — it dispatches raw `conv2d`
//! over a `[N,C,1,1]` global descriptor rather than composing `vision::Conv`, so it
//! is outside this surface. That is deliberate and immaterial to the decision: SE
//! quantizes C values per image, negligible next to the spatial feature maps.

use std::cell::RefCell;
use std::collections::HashMap;

use gpu_core::Gpu;
use paramstore::ParamStore;
use vision::{ActTap, Ctx};

use crate::config::ZipConfig;
use crate::model::ZipDepth;

/// The cap on samples kept per layer. ~50k gives a p99.99 resolved to ~5 samples,
/// which is plenty for a ratio, and bounds memory regardless of image count/size.
const SAMPLE_CAP: usize = 50_000;

#[derive(Default)]
struct LayerAcc {
    absmax: f32,
    /// Strided subsample of `|activation|` values; percentiles are computed from it.
    samples: Vec<f32>,
    /// Total values seen, to keep the subsample stride roughly uniform.
    seen: u64,
}

/// An [`ActTap`] that records, per conv-input tensor, the absolute max and a bounded
/// subsample of magnitudes. It observes only — it never rewrites `x`.
pub struct ActStatsCollector {
    layers: RefCell<HashMap<String, LayerAcc>>,
}

impl Default for ActStatsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl ActStatsCollector {
    pub fn new() -> ActStatsCollector {
        ActStatsCollector { layers: RefCell::new(HashMap::new()) }
    }

    /// Per-layer `(absmax, p99.99, outlier_ratio)`, sorted by ratio descending — the
    /// most quant-hostile layer first.
    pub fn report(&self) -> Vec<LayerReport> {
        let layers = self.layers.borrow();
        let mut out: Vec<LayerReport> = layers
            .iter()
            .map(|(name, a)| {
                let mut s = a.samples.clone();
                s.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let p = |q: f32| -> f32 {
                    if s.is_empty() {
                        return 0.0;
                    }
                    let k = ((q * (s.len() - 1) as f32).round() as usize).min(s.len() - 1);
                    s[k]
                };
                let p9999 = p(0.9999).max(1e-9);
                LayerReport {
                    name: name.clone(),
                    absmax: a.absmax,
                    p9999,
                    outlier_ratio: a.absmax / p9999,
                    p99: p(0.99),
                }
            })
            .collect();
        out.sort_by(|a, b| b.outlier_ratio.partial_cmp(&a.outlier_ratio).unwrap());
        out
    }
}

impl ActTap for ActStatsCollector {
    fn tap(&self, name: &str, x: &mut [f32]) {
        let mut layers = self.layers.borrow_mut();
        let acc = layers.entry(name.to_string()).or_default();
        // Keep the subsample bounded: once full, take roughly every k-th value so the
        // sample stays representative of the whole run, not just the first image.
        let stride = ((acc.seen as usize / SAMPLE_CAP) + 1).max(1);
        for (i, v) in x.iter().enumerate() {
            let a = v.abs();
            if a > acc.absmax {
                acc.absmax = a;
            }
            if acc.samples.len() < SAMPLE_CAP && (acc.seen as usize + i).is_multiple_of(stride) {
                acc.samples.push(a);
            }
        }
        acc.seen += x.len() as u64;
    }
}

/// One layer's activation-range summary.
#[derive(Clone, Debug)]
pub struct LayerReport {
    pub name: String,
    pub absmax: f32,
    /// The 99.99th percentile of `|activation|` — the range INT8's scale should
    /// really target.
    pub p9999: f32,
    pub p99: f32,
    /// `absmax / p99.99`. ~1 means a tight distribution (INT8-friendly); a large
    /// value means a heavy tail that per-tensor symmetric INT8 handles poorly.
    pub outlier_ratio: f32,
}

impl LayerReport {
    /// Is this an encoder tensor (vs decoder)? The names carry the module path.
    pub fn is_encoder(&self) -> bool {
        self.name.starts_with("encoder")
    }
}

/// One calibration image at the size the predictor would feed it.
pub struct CalibImage {
    /// CHW `[3, h, w]` in `[0,1]`.
    pub chw: Vec<f32>,
    pub h: u32,
    pub w: u32,
}

/// Run `images` through an eval-mode ZipDepth built on `ps`, collecting
/// per-layer activation stats. Pure CPU/GPU forward — no NPU.
///
/// Each image carries its OWN `(h, w)`, because the reference's preprocessing is
/// an aspect-preserving resize to a multiple of 32 — so a landscape and a
/// portrait photo produce different input shapes. ZipDepth is fully
/// convolutional and runs at any such size; the model is rebuilt when the shape
/// changes, so images are processed grouped by shape and each distinct shape
/// costs one build.
///
/// This matters for correctness, not tidiness: calibration used to letterbox
/// every image to a padded square, which is a different resampler, a different
/// geometry and a grey fill the model never sees at inference — so the INT8
/// scales were fitted to a distribution that does not occur.
pub fn collect_activation_stats_sized(
    gpu: &Gpu,
    cfg: &ZipConfig,
    ps: &ParamStore,
    images: &[CalibImage],
) -> ActStatsCollector {
    let collector = ActStatsCollector::new();
    let ids = crate::net::ids();
    let ctx = Ctx::with_tap(gpu, ids, &collector);

    // Group by shape so each distinct (h, w) costs one build, not one per image.
    let mut order: Vec<usize> = (0..images.len()).collect();
    order.sort_by_key(|&i| (images[i].h, images[i].w));

    let mut cur: Option<(u32, u32, ZipDepth, gpu_core::DeviceBuffer)> = None;
    for i in order {
        let im = &images[i];
        assert_eq!(im.chw.len(), (3 * im.h * im.w) as usize, "calib image must be [3,{},{}] CHW", im.h, im.w);
        let need = cur.as_ref().map(|(h, w, _, _)| (*h, *w) != (im.h, im.w)).unwrap_or(true);
        if need {
            let model = ZipDepth::build_hw(&ctx, cfg.clone(), 1, im.h, im.w, false);
            model.set_eval(true);
            let input = gpu.storage((3 * im.h * im.w) as u64);
            cur = Some((im.h, im.w, model, input));
        }
        let (_, _, model, input) = cur.as_ref().expect("built above");
        gpu.write(input, bytemuck::cast_slice(&im.chw));
        model.forward(&ctx, ps, input);
    }
    collector
}

/// Square-input convenience: every image is `[3, input, input]`.
///
/// Kept for the unit tests, which build synthetic square tensors and care about
/// the collector's bookkeeping rather than the preprocessing. Production
/// calibration goes through [`collect_activation_stats_sized`] so it sees the
/// shapes inference produces.
pub fn collect_activation_stats(gpu: &Gpu, cfg: &ZipConfig, ps: &ParamStore, images: &[Vec<f32>]) -> ActStatsCollector {
    let sized: Vec<CalibImage> =
        images.iter().map(|c| CalibImage { chw: c.clone(), h: cfg.input, w: cfg.input }).collect();
    collect_activation_stats_sized(gpu, cfg, ps, &sized)
}
