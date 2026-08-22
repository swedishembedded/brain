// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import `hift.pt` (the real `FunAudioLLM/Fun-CosyVoice3-0.5B-2512`
//! checkpoint - a `torch.save`'d `CausalHiFTGenerator.state_dict()`) into
//! [`crate::hift_import::HiftWeights`] - the SAME storage type CosyVoice 2's
//! `HiftGenerator` uses (`f0_predictor`/`m_source_linear_*`/`conv_pre`/`ups`/
//! `source_downs`/`source_resblocks`/`resblocks`/`conv_post`, all plain
//! `Vec<f32>` weight/bias holders with no causality baked into the type) -
//! only the SHAPES and one weight-norm folding convention differ, verified
//! directly against the real checkpoint's own 328 tensor keys (grouped by
//! name pattern - see below), not assumed to match CosyVoice 2's `hift.pt`
//! layout:
//!
//! - **Identical key structure** to CosyVoice 2's `hift.pt`
//!   (`conv_pre`/`conv_post`/`f0_predictor.condnet.{0,2,4,6,8}`/`resblocks.
//!   {i}`/`source_resblocks.{i}` all `parametrizations.weight.{original0,
//!   original1}` weight-normed; `source_downs.{i}` plain `.weight`/`.bias`) -
//!   `CausalConv1d`/`CausalConv1dUpsample`/`CausalConv1dDownSample` all
//!   subclass `nn.Conv1d` directly, so `weight_norm` wraps them exactly the
//!   same way it wraps a plain `Conv1d`.
//! - **`ups[i]`'s weight layout is a plain Conv1d's `[Cout,Cin,K]`, NOT
//!   `ConvTranspose1d`'s `[Cin,Cout,K]`** - confirmed against the real
//!   checkpoint: `ups.0.parametrizations.weight.original1` is `(256,512,16)`
//!   (`Cout=256,Cin=512,K=16`), matching `CausalConv1dUpsample`'s underlying
//!   `nn.Conv1d(512,256,16)`, not a `ConvTranspose1d(512,256,16)`'s
//!   `(512,256,16)`. `weight_norm`'s default `dim=0` is therefore `Cout` here
//!   (256), the SAME convention `crate::hift_import::wn_conv`'s `d0`
//!   parameter already uses for every OTHER plain-Conv1d case in this crate
//!   (`f0_predictor`/`resblocks`/`source_resblocks`) - `d0=Cin` is only ever
//!   needed for CosyVoice 2's genuinely-`ConvTranspose1d` `ups[i]`, so this
//!   importer passes `Cout`, not `Cin`, unlike `hift_import::import_hift_pt`'s
//!   own `ups` loop.
//! - `conv_pre`'s real kernel size is `conv_pre_look_right + 1 = 5` (verified
//!   `(512,80,5)`), not CosyVoice 2's `7`.
//! - `f0_predictor.condnet.0`'s real kernel size is `4` (verified
//!   `(512,80,4)`), not CosyVoice 2's uniform `3` - the rest of `condnet`
//!   (indices 2/4/6/8) stay kernel `3`, byte-identical shapes to CosyVoice 2.
//!
//! Two-way coverage, same discipline as [`crate::hift_import::import_hift_pt`]:
//! every tensor the checkpoint carries is consumed exactly once, and any
//! tensor left over after every named field is read is an error rather than
//! a silent skip.

use std::collections::HashMap;

use audio::conv::fold_weight_norm;
use checkpoint::torchpt::NamedTensor;

use crate::cv3_hift_config::Cv3HiftConfig;
use crate::hift_import::{ConvW, F0PredictorW, HiftWeights, ResBlockW};

struct TensorMap(HashMap<String, NamedTensor>);

impl TensorMap {
    fn take(&mut self, name: &str) -> Result<NamedTensor, String> {
        self.0.remove(name).ok_or_else(|| format!("cv3_hift_import: missing tensor {name:?}"))
    }

    fn plain_conv(&mut self, prefix: &str) -> Result<ConvW, String> {
        let weight = self.take(&format!("{prefix}.weight"))?.data;
        let bias = self.take(&format!("{prefix}.bias"))?.data;
        Ok(ConvW { weight, bias })
    }

    /// A weight-normed conv - `d0` is `weight_v`'s leading dim, `Cout` for
    /// EVERY conv in this causal generator (all subclass plain `nn.Conv1d`,
    /// including `ups[i]` - see this module's doc for why that is a real
    /// divergence from `crate::hift_import::wn_conv`'s CosyVoice-2-only
    /// `Cin` case).
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

/// Import `hift.pt` into [`HiftWeights`] for `CausalHiFTGenerator` (CosyVoice 3).
pub fn import_cv3_hift_pt(path: &str, cfg: &Cv3HiftConfig) -> Result<HiftWeights, String> {
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
    for i in 0..cfg.upsample_rates.len() {
        // Plain Conv1d (see this module's doc): weight_norm dim=0 is Cout =
        // base_channels / 2^(i+1) for stage i.
        let cout = base / (1 << (i + 1));
        ups.push(map.wn_conv(&format!("ups.{i}"), cout)?);
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

    let conv_post = map.wn_conv("conv_post", cfg.source_stft_channels() as usize)?;

    if !map.0.is_empty() {
        let mut extra: Vec<&String> = map.0.keys().collect();
        extra.sort();
        return Err(format!("cv3_hift_import: {} tensor(s) unused: {extra:?}", extra.len()));
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

    #[test]
    fn import_cv3_hift_pt_from_the_real_checkpoint_covers_every_tensor() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights3/hift.pt");
        if !std::path::Path::new(path).is_file() {
            return;
        }
        let cfg = Cv3HiftConfig::cosyvoice3();
        let w = import_cv3_hift_pt(path, &cfg).unwrap_or_else(|e| panic!("import_cv3_hift_pt: {e}"));
        assert_eq!(w.conv_pre.weight.len(), 512 * 80 * 5, "conv_pre kernel must be conv_pre_look_right+1=5");
        assert_eq!(w.conv_pre.bias.len(), 512);
        assert_eq!(w.ups[0].weight.len(), 256 * 512 * 16, "ups[0] must be a plain Conv1d [Cout,Cin,K], not ConvTranspose1d");
        assert_eq!(w.ups[1].weight.len(), 128 * 256 * 11);
        assert_eq!(w.ups[2].weight.len(), 64 * 128 * 7);
        assert_eq!(w.source_downs[0].weight.len(), 256 * 18 * 30);
        assert_eq!(w.source_downs[1].weight.len(), 128 * 18 * 6);
        assert_eq!(w.source_downs[2].weight.len(), 64 * 18);
        assert_eq!(w.conv_post.weight.len(), 18 * 64 * 7);
        assert_eq!(w.f0_predictor.condnet[0].weight.len(), 512 * 80 * 4, "condnet[0] kernel must be 4");
        assert_eq!(w.f0_predictor.condnet[1].weight.len(), 512 * 512 * 3);
        assert_eq!(w.f0_predictor.classifier_w.len(), 512);
        assert_eq!(w.m_source_linear_w.len(), 9);
    }
}
