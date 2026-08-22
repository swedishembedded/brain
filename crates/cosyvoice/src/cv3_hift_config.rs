// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `CausalHiFTGenerator` (CosyVoice 3) hyperparameters, verified against the
//! real `resources/cosyvoice/weights3/cosyvoice3.yaml`'s `hift:` block and
//! `cosyvoice/hifigan/generator.py`'s `CausalHiFTGenerator.__init__`.
//!
//! Reuses the exact same `ConvRNNF0Predictor` -> NSF -> conv-trunk -> ISTFT
//! topology `crate::hift_config`/`crate::hift` already implement for
//! CosyVoice 2, but every conv is causal: `conv_pre` is a RIGHT-looking
//! causal conv (`conv_pre_look_right=4`, kernel size 5, right-only padding -
//! NOT CosyVoice 2's symmetric `Conv1d(k=7,p=3)`), `ups[i]` are
//! nearest-upsample-then-LEFT-causal-`Conv1d` (`CausalConv1dUpsample`), NOT
//! `ConvTranspose1d` - so `ups[i]`'s real checkpoint weight layout is a plain
//! Conv1d's `[Cout,Cin,K]`, not `ConvTranspose1d`'s `[Cin,Cout/G,K]` (verified
//! against the real `hift.pt`'s own tensor shapes:
//! `ups.0.parametrizations.weight.original1` is `(256,512,16)` = `[Cout,
//! Cin,K]`, matching a plain `nn.Conv1d(512,256,16)`, not the `[512,256,16]`
//! a `ConvTranspose1d(512,256,16)` would carry). `CausalConvRNNF0Predictor`'s
//! first conv is right-causal `k=4` (3 frames lookahead), the rest are
//! left-causal `k=3` - everything else (channel counts, dilations, Snake
//! activations, ISTFT params) is BYTE-IDENTICAL to CosyVoice 2's
//! `HiftConfig::cosyvoice2()`, verified against the real `cosyvoice3.yaml`.

pub const RESBLOCK_DILATIONS: [u32; 3] = [1, 3, 5];

#[derive(Clone, Debug)]
pub struct Cv3HiftConfig {
    pub in_channels: u32,
    pub base_channels: u32,
    pub nb_harmonics: u32,
    pub sampling_rate: u32,
    pub nsf_alpha: f32,
    pub nsf_sigma: f32,
    pub nsf_voiced_threshold: f32,
    pub upsample_rates: [u32; 3],
    pub upsample_kernel_sizes: [u32; 3],
    pub n_fft: u32,
    pub hop_len: u32,
    pub resblock_kernel_sizes: [u32; 3],
    pub source_resblock_kernel_sizes: [u32; 3],
    pub lrelu_slope: f32,
    pub audio_limit: f32,
    pub f0_cond_channels: u32,
    /// `conv_pre`'s right-lookahead frame count (`4` - kernel size is this
    /// plus one).
    pub conv_pre_look_right: u32,
}

impl Cv3HiftConfig {
    /// The real `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` `CausalHiFTGenerator`
    /// configuration.
    pub fn cosyvoice3() -> Cv3HiftConfig {
        Cv3HiftConfig {
            in_channels: 80,
            base_channels: 512,
            nb_harmonics: 8,
            sampling_rate: 24000,
            nsf_alpha: 0.1,
            nsf_sigma: 0.003,
            nsf_voiced_threshold: 10.0,
            upsample_rates: [8, 5, 3],
            upsample_kernel_sizes: [16, 11, 7],
            n_fft: 16,
            hop_len: 4,
            resblock_kernel_sizes: [3, 7, 11],
            source_resblock_kernel_sizes: [7, 7, 11],
            lrelu_slope: 0.1,
            audio_limit: 0.99,
            f0_cond_channels: 512,
            conv_pre_look_right: 4,
        }
    }

    pub fn harmonics(&self) -> u32 {
        self.nb_harmonics + 1
    }

    pub fn nsf_upsample_scale(&self) -> u32 {
        self.upsample_rates.iter().product::<u32>() * self.hop_len
    }

    pub fn stft_bins(&self) -> u32 {
        self.n_fft / 2 + 1
    }

    pub fn source_stft_channels(&self) -> u32 {
        self.n_fft + 2
    }

    /// Identical formula to `HiftConfig::source_downsample_strides` (the
    /// `upsample_rates` are byte-identical between generations, so the
    /// derived strides are too - `[15, 3, 1]`).
    pub fn source_downsample_strides(&self) -> [u32; 3] {
        let rev = [self.upsample_rates[2], self.upsample_rates[1]];
        let cum = [1u32, rev[0], rev[0] * rev[1]];
        [cum[2], cum[1], cum[0]]
    }

    /// `conv_pre`'s real kernel size: `conv_pre_look_right + 1`.
    pub fn conv_pre_kernel(&self) -> u32 {
        self.conv_pre_look_right + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosyvoice3_matches_the_real_yaml_and_generator_defaults() {
        let cfg = Cv3HiftConfig::cosyvoice3();
        assert_eq!(cfg.upsample_rates, [8, 5, 3]);
        assert_eq!(cfg.upsample_kernel_sizes, [16, 11, 7]);
        assert_eq!(cfg.source_resblock_kernel_sizes, [7, 7, 11]);
        assert_eq!(cfg.harmonics(), 9);
        assert_eq!(cfg.nsf_upsample_scale(), 480);
        assert_eq!(cfg.stft_bins(), 9);
        assert_eq!(cfg.source_stft_channels(), 18);
        assert_eq!(cfg.source_downsample_strides(), [15, 3, 1]);
        assert_eq!(cfg.conv_pre_look_right, 4);
        assert_eq!(cfg.conv_pre_kernel(), 5);
    }
}
