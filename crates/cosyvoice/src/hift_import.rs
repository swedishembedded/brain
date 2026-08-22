// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import `hift.pt` (a `torch.save`'d `HiFTGenerator.state_dict()`) into
//! [`HiftWeights`], folding every `nn.utils.parametrizations.weight_norm`
//! pair at import time via `audio::conv::fold_weight_norm` (already hoisted;
//! not reimplemented here).
//!
//! **Real gotcha, verified against the actual checkpoint (not assumed):**
//! this build of torch (2.13.0) serializes `weight_norm` via the NEWER
//! `torch.nn.utils.parametrizations.weight_norm` API, whose state_dict keys
//! are `<prefix>.parametrizations.weight.original0`/`.original1` - NOT the
//! older `<prefix>.weight_g`/`<prefix>.weight_v` pair
//! `minimaxmusic3::vocoder`'s importer reads. Shapes/semantics are identical
//! (`original0` is `g`, `original1` is `v`), so `fold_weight_norm` applies
//! unchanged; only the key names differ. `source_downs[i]` is a genuinely
//! PLAIN `Conv1d` in the reference (never wrapped in `weight_norm`), so it
//! reads straight `.weight`/`.bias` - confirmed by listing the real
//! checkpoint's keys, not assumed from the source.
//!
//! Every tensor the checkpoint carries is either consumed here or the import
//! fails loudly (two-way coverage) -
//! there is no zero-fill and no silent skip.

use std::collections::HashMap;

use audio::conv::fold_weight_norm;
use checkpoint::torchpt::NamedTensor;

use crate::hift_config::HiftConfig;

/// A conv's folded weight + bias, ready for `audio::conv::conv1d_ref` /
/// `convtr1d_ref`.
#[derive(Clone)]
pub struct ConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

/// One `ResBlock`: 3 (kernel, dilation) branches, each `Snake -> conv1
/// (dilated) -> Snake -> conv2 (dilation=1)`, `+x` residual.
#[derive(Clone)]
pub struct ResBlockW {
    pub convs1: [ConvW; 3],
    pub convs2: [ConvW; 3],
    pub alpha1: [Vec<f32>; 3],
    pub alpha2: [Vec<f32>; 3],
}

#[derive(Clone)]
pub struct F0PredictorW {
    /// 5 conv layers: `in->512`, then `512->512` x4, each `ELU`-activated.
    pub condnet: [ConvW; 5],
    /// `Linear(512, 1)`: weight `[512]`, bias scalar.
    pub classifier_w: Vec<f32>,
    pub classifier_b: f32,
}

pub struct HiftWeights {
    pub f0_predictor: F0PredictorW,
    /// `m_source.l_linear`: `Linear(9, 1)` merging the harmonic sines - weight
    /// `[9]`, bias scalar.
    pub m_source_linear_w: Vec<f32>,
    pub m_source_linear_b: f32,
    pub conv_pre: ConvW,
    /// 3 `ConvTranspose1d` upsample stages; weight layout `[Cin, Cout/G, K]`.
    pub ups: [ConvW; 3],
    /// 3 plain (non-weight-normed) strided `Conv1d`s over the excitation STFT.
    pub source_downs: [ConvW; 3],
    pub source_resblocks: [ResBlockW; 3],
    /// 9 = 3 upsample stages x 3 kernel sizes, `resblocks[i*3+j]`.
    pub resblocks: [ResBlockW; 9],
    pub conv_post: ConvW,
}

struct TensorMap(HashMap<String, NamedTensor>);

impl TensorMap {
    fn take(&mut self, name: &str) -> Result<NamedTensor, String> {
        self.0.remove(name).ok_or_else(|| format!("hift_import: missing tensor {name:?}"))
    }

    /// A plain (non-weight-normed) conv: `<prefix>.weight`/`<prefix>.bias`.
    fn plain_conv(&mut self, prefix: &str) -> Result<ConvW, String> {
        let weight = self.take(&format!("{prefix}.weight"))?.data;
        let bias = self.take(&format!("{prefix}.bias"))?.data;
        Ok(ConvW { weight, bias })
    }

    /// A weight-normed conv (the new `parametrizations.weight.{original0,
    /// original1}` key pair - see this module's doc), folded at import time.
    /// `d0` is `weight_v`'s (== `original1`'s) leading dim: `Cout` for a
    /// plain `Conv1d`, `Cin` for a `ConvTranspose1d` (`audio::conv::
    /// fold_weight_norm`'s own contract).
    fn wn_conv(&mut self, prefix: &str, d0: usize) -> Result<ConvW, String> {
        let g = self.take(&format!("{prefix}.parametrizations.weight.original0"))?.data;
        let v = self.take(&format!("{prefix}.parametrizations.weight.original1"))?.data;
        let bias = self.take(&format!("{prefix}.bias"))?.data;
        Ok(ConvW { weight: fold_weight_norm(&g, &v, d0), bias })
    }

    fn resblock(&mut self, prefix: &str, channels: usize) -> Result<ResBlockW, String> {
        let mut convs1: Vec<ConvW> = Vec::with_capacity(3);
        let mut convs2: Vec<ConvW> = Vec::with_capacity(3);
        let mut alpha1: Vec<Vec<f32>> = Vec::with_capacity(3);
        let mut alpha2: Vec<Vec<f32>> = Vec::with_capacity(3);
        for j in 0..3 {
            convs1.push(self.wn_conv(&format!("{prefix}.convs1.{j}"), channels)?);
            convs2.push(self.wn_conv(&format!("{prefix}.convs2.{j}"), channels)?);
            alpha1.push(self.take(&format!("{prefix}.activations1.{j}.alpha"))?.data);
            alpha2.push(self.take(&format!("{prefix}.activations2.{j}.alpha"))?.data);
        }
        Ok(ResBlockW {
            convs1: convs1.try_into().unwrap_or_else(|_| unreachable!()),
            convs2: convs2.try_into().unwrap_or_else(|_| unreachable!()),
            alpha1: alpha1.try_into().unwrap_or_else(|_| unreachable!()),
            alpha2: alpha2.try_into().unwrap_or_else(|_| unreachable!()),
        })
    }
}

/// Import `hift.pt` into [`HiftWeights`], validated with the same two-way
/// coverage discipline `llm_import::import_llm_pt` uses: every tensor the
/// checkpoint carries is consumed exactly once, and any tensor left over
/// after every named field is read is an error rather than a silent skip.
pub fn import_hift_pt(path: &str, cfg: &HiftConfig) -> Result<HiftWeights, String> {
    let tensors = checkpoint::torchpt::read(path)?;
    let mut map = TensorMap(tensors.into_iter().map(|t| (t.name.clone(), t)).collect());

    let base = cfg.base_channels as usize;
    let f0_predictor = F0PredictorW {
        condnet: [
            map.wn_conv("f0_predictor.condnet.0", base)?,
            map.wn_conv("f0_predictor.condnet.2", base)?,
            map.wn_conv("f0_predictor.condnet.4", base)?,
            map.wn_conv("f0_predictor.condnet.6", base)?,
            map.wn_conv("f0_predictor.condnet.8", base)?,
        ],
        classifier_w: map.take("f0_predictor.classifier.weight")?.data,
        classifier_b: map.take("f0_predictor.classifier.bias")?.data[0],
    };

    let m_source_linear = map.take("m_source.l_linear.weight")?;
    let m_source_linear_w = m_source_linear.data;
    let m_source_linear_b = map.take("m_source.l_linear.bias")?.data[0];

    let conv_pre = map.wn_conv("conv_pre", base)?;

    let mut ups: Vec<ConvW> = Vec::with_capacity(3);
    for (i, _) in cfg.upsample_rates.iter().enumerate() {
        // ConvTranspose1d weight_norm dim=0 is Cin (audio::conv's documented
        // contract): Cin = base_channels / 2^i for stage i.
        let cin = base / (1 << i);
        ups.push(map.wn_conv(&format!("ups.{i}"), cin)?);
    }

    let mut source_downs: Vec<ConvW> = Vec::with_capacity(3);
    let mut source_resblocks: Vec<ResBlockW> = Vec::with_capacity(3);
    for i in 0..3usize {
        source_downs.push(map.plain_conv(&format!("source_downs.{i}"))?);
        let ch = base / (1 << (i + 1));
        source_resblocks.push(map.resblock(&format!("source_resblocks.{i}"), ch)?);
    }

    let mut resblocks: Vec<ResBlockW> = Vec::with_capacity(9);
    for i in 0..3usize {
        let ch = base / (1 << (i + 1));
        for j in 0..3usize {
            resblocks.push(map.resblock(&format!("resblocks.{}", i * 3 + j), ch)?);
        }
    }

    let final_ch = base / (1 << cfg.upsample_rates.len());
    let conv_post = map.wn_conv("conv_post", cfg.source_stft_channels() as usize)?;
    // conv_post's own weight_norm d0 is its Cout (n_fft+2 = 18), the
    // ordinary Conv1d convention - `final_ch` (64) is its Cin, verified via
    // the checkpoint's own conv_post.parametrizations.weight.original1 shape
    // (18, 64, 7), unused directly here.
    let _ = final_ch;

    if !map.0.is_empty() {
        let mut extra: Vec<&String> = map.0.keys().collect();
        extra.sort();
        return Err(format!("hift_import: {} tensor(s) unused: {extra:?}", extra.len()));
    }

    Ok(HiftWeights {
        f0_predictor,
        m_source_linear_w,
        m_source_linear_b,
        conv_pre,
        ups: ups.try_into().unwrap_or_else(|_| unreachable!()),
        source_downs: source_downs.try_into().unwrap_or_else(|_| unreachable!()),
        source_resblocks: source_resblocks.try_into().unwrap_or_else(|_| unreachable!()),
        resblocks: resblocks.try_into().unwrap_or_else(|_| unreachable!()),
        conv_post,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `import_hift_pt` against a real `hift.pt`, when present - the two-way
    /// coverage check IS this test (a bad tensor-name mapping fails loudly
    /// before any forward runs). Skips cleanly when the checkpoint is absent
    /// (mirrors `llm_import`'s own convention, but this crate has no
    /// `brain_testutil` dependency in a `[dependencies]`-only unit test, so
    /// this one just returns rather than calling `skip()`).
    #[test]
    fn import_hift_pt_from_the_real_checkpoint_covers_every_tensor() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights/hift.pt");
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let cfg = HiftConfig::cosyvoice2();
        let w = import_hift_pt(path, &cfg).unwrap_or_else(|e| panic!("import_hift_pt: {e}"));
        assert_eq!(w.conv_pre.weight.len(), 512 * 80 * 7);
        assert_eq!(w.conv_pre.bias.len(), 512);
        assert_eq!(w.ups[0].weight.len(), 512 * 256 * 16);
        assert_eq!(w.ups[1].weight.len(), 256 * 128 * 11);
        assert_eq!(w.ups[2].weight.len(), 128 * 64 * 7);
        assert_eq!(w.source_downs[0].weight.len(), 256 * 18 * 30);
        assert_eq!(w.source_downs[1].weight.len(), 128 * 18 * 6);
        assert_eq!(w.source_downs[2].weight.len(), (64 * 18));
        assert_eq!(w.conv_post.weight.len(), 18 * 64 * 7);
        assert_eq!(w.f0_predictor.classifier_w.len(), 512);
        assert_eq!(w.m_source_linear_w.len(), 9);
    }
}
