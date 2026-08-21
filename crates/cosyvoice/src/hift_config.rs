// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `HiFTGenerator` (CosyVoice 2, NON-causal) hyperparameters, verified
//! against the real `resources/cosyvoice/weights/cosyvoice2.yaml`'s `hift:`
//! block and `cosyvoice/hifigan/generator.py`'s `HiFTGenerator.__init__`
//! defaults (both read directly, not assumed from the paper).
//!
//! CosyVoice 3's `CausalHiFTGenerator` (causal convs throughout, no
//! `cache_source` state, `SineGen2(causal=True)`'s own frozen-noise-buffer
//! branch) is a deliberate follow-up - see `resources/cosyvoice/source/
//! cosyvoice/hifigan/generator.py`'s `CausalHiFTGenerator` for the delta.
//! Nothing here models it.

/// One (kernel, dilation) `ResBlock` shape, reused for both `resblocks` and
/// `source_resblocks`.
pub const RESBLOCK_DILATIONS: [u32; 3] = [1, 3, 5];

#[derive(Clone, Debug)]
pub struct HiftConfig {
    pub in_channels: u32,
    pub base_channels: u32,
    /// Harmonic overtone count; `nb_harmonics + 1` sinusoids total (the
    /// fundamental plus this many overtones).
    pub nb_harmonics: u32,
    pub sampling_rate: u32,
    pub nsf_alpha: f32,
    pub nsf_sigma: f32,
    pub nsf_voiced_threshold: f32,
    /// 3 stages, applied in order.
    pub upsample_rates: [u32; 3],
    pub upsample_kernel_sizes: [u32; 3],
    pub n_fft: u32,
    pub hop_len: u32,
    pub resblock_kernel_sizes: [u32; 3],
    pub source_resblock_kernel_sizes: [u32; 3],
    /// `F.leaky_relu(x, lrelu_slope)` between `conv_pre`/each upsample stage
    /// and its `ups[i]` - NOT the same slope `decode`'s final activation
    /// uses (that one is `F.leaky_relu(x)`, PyTorch's default 0.01).
    pub lrelu_slope: f32,
    pub audio_limit: f32,
    /// `ConvRNNF0Predictor`'s hidden width (its 5 conv layers + the
    /// `Linear(cond_channels, 1)` classifier).
    pub f0_cond_channels: u32,
}

impl HiftConfig {
    /// The real `FunAudioLLM/CosyVoice2-0.5B` `HiFTGenerator` configuration.
    pub fn cosyvoice2() -> HiftConfig {
        HiftConfig {
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
        }
    }

    /// Sinusoid count (fundamental + overtones) `SineGen2` generates per
    /// sample: `nb_harmonics + 1`.
    pub fn harmonics(&self) -> u32 {
        self.nb_harmonics + 1
    }

    /// `nn.Upsample(scale_factor=...)` HiFT applies to `f0` before the NSF
    /// source branch: `product(upsample_rates) * hop_len` (480 for the real
    /// config) - samples-per-mel-frame at the final waveform rate.
    pub fn nsf_upsample_scale(&self) -> u32 {
        self.upsample_rates.iter().product::<u32>() * self.hop_len
    }

    /// `istft_params["n_fft"] // 2 + 1` - one-sided spectrum bin count (9 for
    /// the real config); `conv_post`'s output is `2 * stft_bins` channels
    /// (magnitude half + phase half).
    pub fn stft_bins(&self) -> u32 {
        self.n_fft / 2 + 1
    }

    /// The excitation `_stft`'s channel count fed to every `source_downs[i]`:
    /// `n_fft + 2` (real + imag concatenated, 18 for the real config).
    pub fn source_stft_channels(&self) -> u32 {
        self.n_fft + 2
    }

    /// `downsample_cum_rates[::-1]` from the reference
    /// (`[1] + upsample_rates[::-1][:-1]`, cumulative product, reversed) -
    /// the stride each `source_downs[i]` uses against the fixed-length
    /// excitation STFT (`[15, 3, 1]` for the real config).
    pub fn source_downsample_strides(&self) -> [u32; 3] {
        let rev = [self.upsample_rates[2], self.upsample_rates[1]]; // upsample_rates[::-1][:-1]
        let cum = [1u32, rev[0], rev[0] * rev[1]];
        [cum[2], cum[1], cum[0]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosyvoice2_matches_the_real_yaml_and_generator_defaults() {
        let cfg = HiftConfig::cosyvoice2();
        assert_eq!(cfg.upsample_rates, [8, 5, 3]);
        assert_eq!(cfg.upsample_kernel_sizes, [16, 11, 7]);
        assert_eq!(cfg.source_resblock_kernel_sizes, [7, 7, 11]);
        assert_eq!(cfg.harmonics(), 9);
        assert_eq!(cfg.nsf_upsample_scale(), 480);
        assert_eq!(cfg.stft_bins(), 9);
        assert_eq!(cfg.source_stft_channels(), 18);
        assert_eq!(cfg.source_downsample_strides(), [15, 3, 1]);
    }
}
