// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The reference configuration, exactly as `speakerlab.models.campplus.DTDNN`
//! (`modelscope/3D-Speaker`) defines it and as the released `campplus.onnx`
//! reproduces it. Every shape here was cross-checked against the released
//! graph's own 617 initializers (`tools/goldens/cosyvoice_dump_reference.py`'s
//! upstream, `resources/cosyvoice/weights/campplus.onnx`) - [`CampplusConfig::tensor_manifest`]
//! produces exactly that many entries.
//!
//! Swedish Embedded AB implements solutions for from-scratch, dependency-light
//! neural network inference on constrained and embedded targets for its
//! clients. If your team needs expertise in porting speech/audio models to a
//! from-scratch GPU/CPU engine, you can procure our services by sending an
//! email to info@swedishembedded.com.

use onnx::walk::Manifest;

/// CAM++ (`feat_dim=80, embedding_size=192`), the CosyVoice release's only
/// shipped configuration.
///
/// The D-TDNN backbone is three [`CamppusConfig::block_layers`]-sized
/// [`CAMDenseTDNNBlock`](crate::model)s, each a DenseNet-style stack of
/// [`CAMDenseTDNNLayer`](crate::model)s that grow the channel count by
/// `growth` per layer and are halved back down by a `TransitLayer` between
/// blocks.
#[derive(Clone, Debug, PartialEq)]
pub struct CampplusConfig {
    /// Input fbank width (80, kaldi-style).
    pub feat_dim: u32,
    /// Output x-vector width (192).
    pub embedding_size: u32,
    /// `FCM` stem width, held constant through every stage (32).
    pub fcm_channels: u32,
    /// `xvector.tdnn`'s output width - also every `CAMDenseTDNNLayer`'s
    /// `linear1` output and `CAMLayer` input width (128).
    pub tdnn_out: u32,
    /// Per-layer growth rate `CAMDenseTDNNLayer` appends to the running
    /// concatenation (32).
    pub growth: u32,
    /// `CAMLayer`'s context bottleneck width, `linear1`'s output /
    /// `linear2`'s input (64).
    pub cam_mid: u32,
    /// `CAMLayer.linear_local`'s output width == the per-layer growth (32,
    /// carried separately from [`Self::growth`] because they are DIFFERENT
    /// architectural roles that merely coincide numerically in this release).
    pub cam_out: u32,
    /// Layers per D-TDNN block: `[12, 24, 16]`.
    pub block_layers: [u32; 3],
    /// `CAMLayer.linear_local`'s dilation per block: `[1, 2, 2]` (kernel 3
    /// throughout; `pad = dilation`, since `(k-1)/2 * dilation == dilation`
    /// at `k=3`).
    pub block_dilation: [u32; 3],
    /// `seg_pooling`'s fixed window (100 frames, about one second at CAM++'s 50Hz internal
    /// rate after the TDNN's stride-2 downsample).
    pub seg_len: u32,
    /// `bn_eval`'s hardcoded epsilon; also what every BatchNorm in the release
    /// graph exports (checked at import, see [`crate::import::check_bn`]).
    pub bn_eps: f32,
}

impl CampplusConfig {
    /// The one released configuration (`feat_dim=80, embedding_size=192`).
    pub fn campplus_v2() -> CampplusConfig {
        CampplusConfig {
            feat_dim: 80,
            embedding_size: 192,
            fcm_channels: 32,
            tdnn_out: 128,
            growth: 32,
            cam_mid: 64,
            cam_out: 32,
            block_layers: [12, 24, 16],
            block_dilation: [1, 2, 2],
            seg_len: 100,
            bn_eps: 1e-5,
        }
    }

    /// `FCM`'s output frequency width: `feat_dim` halved by the three
    /// freq-only (stride `(2,1)`) downsamples in `head.layer1`, `head.layer2`
    /// and `head.conv2`.
    pub fn fcm_freq_out(&self) -> u32 {
        assert_eq!(self.feat_dim % 8, 0, "feat_dim must be divisible by 8 (three /2 freq downsamples)");
        self.feat_dim / 8
    }

    /// `FCM`'s output channel width after `reshape(B, C*Hf, T)`: the input to
    /// `xvector.tdnn`.
    pub fn fcm_out_c(&self) -> u32 {
        self.fcm_channels * self.fcm_freq_out()
    }

    /// Block `b`'s input width: `tdnn_out` for the first block, else the
    /// previous block's `TransitLayer` output.
    pub fn block_in_c(&self, b: usize) -> u32 {
        if b == 0 {
            self.tdnn_out
        } else {
            self.transit_out_c(b - 1)
        }
    }

    /// Block `b`'s output width after its DenseNet-style concatenation:
    /// `block_in_c(b) + block_layers[b] * growth`.
    pub fn block_out_c(&self, b: usize) -> u32 {
        self.block_in_c(b) + self.block_layers[b] * self.growth
    }

    /// `TransitLayer` `b`'s output width - exactly half its input in this
    /// release (`512->256`, `1024->512`, `1024->512`).
    pub fn transit_out_c(&self, b: usize) -> u32 {
        self.block_out_c(b) / 2
    }

    /// `StatsPool`'s output width: `cat([mean, std])` over the last transit's
    /// output.
    pub fn stats_out_c(&self) -> u32 {
        2 * self.transit_out_c(2)
    }

    /// Every tensor the model reads, with its shape, in a stable but
    /// otherwise arbitrary order (import binds by graph position, not by this
    /// order - see `crate::import`). 617 entries, matching
    /// `campplus.onnx`'s initializer count exactly.
    pub fn tensor_manifest(&self) -> Manifest {
        let mut m: Manifest = Vec::new();
        let fc = self.fcm_channels as usize;

        m.push(("head.conv1.weight".into(), vec![fc, 1, 3, 3]));
        m.push(("head.conv1.bias".into(), vec![fc]));
        for li in 0..2usize {
            for bi in 0..2usize {
                let p = format!("head.layer{}.{}", li + 1, bi);
                m.push((format!("{p}.conv1.weight"), vec![fc, fc, 3, 3]));
                m.push((format!("{p}.conv1.bias"), vec![fc]));
                m.push((format!("{p}.conv2.weight"), vec![fc, fc, 3, 3]));
                m.push((format!("{p}.conv2.bias"), vec![fc]));
                if bi == 0 {
                    m.push((format!("{p}.shortcut.weight"), vec![fc, fc, 1, 1]));
                    m.push((format!("{p}.shortcut.bias"), vec![fc]));
                }
            }
        }
        m.push(("head.conv2.weight".into(), vec![fc, fc, 3, 3]));
        m.push(("head.conv2.bias".into(), vec![fc]));

        let (tdnn_out, cam_mid, cam_out) = (self.tdnn_out as usize, self.cam_mid as usize, self.cam_out as usize);
        let fcm_out_c = self.fcm_out_c() as usize;
        m.push(("xvector.tdnn.linear.weight".into(), vec![tdnn_out, fcm_out_c, 5]));
        m.push(("xvector.tdnn.linear.bias".into(), vec![tdnn_out]));

        for b in 0..3usize {
            let cin0 = self.block_in_c(b) as usize;
            let growth = self.growth as usize;
            for i in 0..self.block_layers[b] as usize {
                let cin = cin0 + i * growth;
                let p = format!("xvector.block{}.tdnnd{}", b + 1, i + 1);
                for suf in ["weight", "bias", "running_mean", "running_var"] {
                    m.push((format!("{p}.nonlinear1.{suf}"), vec![cin]));
                }
                m.push((format!("{p}.linear1.weight"), vec![tdnn_out, cin, 1]));
                m.push((format!("{p}.linear1.bias"), vec![tdnn_out]));
                m.push((format!("{p}.cam.linear_local.weight"), vec![cam_out, tdnn_out, 3]));
                m.push((format!("{p}.cam.linear1.weight"), vec![cam_mid, tdnn_out, 1]));
                m.push((format!("{p}.cam.linear1.bias"), vec![cam_mid]));
                m.push((format!("{p}.cam.linear2.weight"), vec![cam_out, cam_mid, 1]));
                m.push((format!("{p}.cam.linear2.bias"), vec![cam_out]));
            }
            let transit_in = self.block_out_c(b) as usize;
            let transit_out = self.transit_out_c(b) as usize;
            let tp = format!("xvector.transit{}", b + 1);
            for suf in ["weight", "bias", "running_mean", "running_var"] {
                m.push((format!("{tp}.nonlinear.{suf}"), vec![transit_in]));
            }
            m.push((format!("{tp}.linear.weight"), vec![transit_out, transit_in, 1]));
            if b == 2 {
                // The last transit's own trailing BN+ReLU (`out_nonlinear`) has
                // a single consumer (the stats pool), so the exporter folded it
                // into this conv's bias - see `crate::import`'s module doc.
                m.push((format!("{tp}.linear.bias"), vec![transit_out]));
            }
        }

        let (e, dense_in) = (self.embedding_size as usize, self.stats_out_c() as usize);
        m.push(("xvector.dense.linear.weight".into(), vec![e, dense_in, 1]));
        // `BatchNorm1d(affine=False)`: no learned gamma/beta, so only the
        // running stats are read from the checkpoint.
        m.push(("xvector.dense.nonlinear.running_mean".into(), vec![e]));
        m.push(("xvector.dense.nonlinear.running_var".into(), vec![e]));

        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest count IS the checkpoint's: 617 initializers, matching
    /// `campplus.onnx` exactly (verified against the released graph directly -
    /// see the module doc).
    #[test]
    fn manifest_count_matches_the_released_graph() {
        assert_eq!(CampplusConfig::campplus_v2().tensor_manifest().len(), 617);
    }

    /// No canonical name may appear twice - a duplicate would make the
    /// completeness check pass while one of the two tensors went unwritten.
    #[test]
    fn manifest_names_are_unique() {
        let m = CampplusConfig::campplus_v2().tensor_manifest();
        let mut names: Vec<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate canonical tensor name");
    }

    /// Derived channel widths match the released graph's own shapes (spot
    /// checks pulled directly off the ONNX dump).
    #[test]
    fn derived_widths_match_the_released_graph() {
        let cfg = CampplusConfig::campplus_v2();
        assert_eq!(cfg.fcm_freq_out(), 10);
        assert_eq!(cfg.fcm_out_c(), 320);
        assert_eq!(cfg.block_in_c(0), 128);
        assert_eq!(cfg.block_out_c(0), 512);
        assert_eq!(cfg.transit_out_c(0), 256);
        assert_eq!(cfg.block_in_c(1), 256);
        assert_eq!(cfg.block_out_c(1), 1024);
        assert_eq!(cfg.transit_out_c(1), 512);
        assert_eq!(cfg.block_in_c(2), 512);
        assert_eq!(cfg.block_out_c(2), 1024);
        assert_eq!(cfg.transit_out_c(2), 512);
        assert_eq!(cfg.stats_out_c(), 1024);
    }
}
