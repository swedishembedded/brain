// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's audio VAE decoder: a BigVGAN-topology vocoder (`dec_in_proj`
//! -> `conv_pre` -> 7 upsample stages (`ConvTranspose1d` + 3 parallel
//! AMPBlock1 resblocks averaged) -> `activation_post` -> `conv_post` ->
//! clamp), reached through [`decode`], matching the checkpoint's own
//! `DacAudioVAE.decode`: `z = dec_in_proj(z); return decoder(z)` - a plain
//! (non-weight-normalized) 1x1 conv from the 32-channel VAE latent to the
//! decoder's 2048-channel working width, then the BigVGAN stack below.
//!
//! Ported from the checkpoint's own shipped `dac_audio_vae.py`/
//! `dac_bigvgan.py` (Apache-2.0/MIT), reusing an existing in-workspace
//! BigVGAN v2 / AMP1 / anti-aliased-SnakeBeta vocoder's structure near-
//! verbatim rather than rederiving it - the same topology already runs in
//! this workspace at different config numbers. Real, checked differences
//! from that precedent (never assumed from the shared topology name):
//!
//! * **7 upsample stages, not 6** (`upsample_rates=[5,5,2,2,2,2,2]`,
//!   `upsample_kernel_sizes=[9,9,4,4,4,4,4]` - the 32kHz BigVGAN config in
//!   `dac_audio_vae.py`), `upsample_initial_channel=1024`.
//! * **Mono output** (`out_channels=1`) - [`decode`] takes a plain `[C,T]`
//!   latent directly, no stereo/mel-bin reshape (H3's decoder input is the
//!   VAE latent itself, not a channel-split mel spectrogram).
//! * **`ups.{i}.0.*`, not `ups.{i}.*`** - `BigVGAN.__init__` wraps each
//!   stage's single `ConvTranspose1d` in its own one-element `nn.ModuleList`
//!   (`self.ups.append(nn.ModuleList([weight_norm(ConvTranspose1d(...))]))`),
//!   confirmed against the real checkpoint header.
//! * **`resblocks.{idx}.activations.{0..5}`, not split `.acts1.{d}`/
//!   `.acts2.{d}` prefixes** - `AMPBlock1` stores ONE flat 6-element
//!   `activations` list and SLICES it in its own forward (`acts1, acts2 =
//!   self.activations[::2], self.activations[1::2]`), confirmed against the
//!   real header (no `acts1`/`acts2` keys anywhere). [`amp_block`] indexes
//!   `activations.{2*d}`/`activations.{2*d+1}` directly rather than renaming
//!   at import, since the checkpoint's own layout already carries everything
//!   it needs.
//! * **`activation_post`, not `act_post`.**
//! * **`dec_in_proj` is a top-level, PLAIN (non-weight-normalized) 1x1
//!   `Conv1d(32, 2048)`** - no `weight_g`/`weight_v` pair in the real header,
//!   unlike every other conv in this file.
//!
//! Everything else matches the shared topology exactly: `use_tanh_at_final:
//! false` in this checkpoint's own BigVGAN config -> the final activation is
//! `torch.clamp(x, -1, 1)`, NOT `tanh`; `use_bias_at_final: false` -> `conv_post`
//! has no bias tensor at all; the anti-aliased `Activation1d` (2x replicate-
//! padded depthwise `ConvTranspose1d` against a checkpoint-loaded Kaiser-sinc
//! filter -> SnakeBeta -> 2x replicate-padded depthwise `Conv1d` lowpass-
//! decimate) wraps every SnakeBeta call; weight_norm (`weight = g*v/||v||`,
//! dim 0 of the STORED tensor - `Cout` for a plain `Conv1d`, `Cin` for
//! `ConvTranspose1d`'s native `[Cin,Cout/G,K]` layout) is folded once at
//! import, never at forward time.
//!
//! **Encode is out of scope for this milestone.** The checkpoint's own
//! `DacAudioVAE.decode` never touches `encoder`/`mean_proj`/`logs_proj`/
//! `pre_block` at all - present in the checkpoint for training-time
//! compatibility, dead in the shipped inference-only forward - and neither
//! does this port. Only a task needing an ENCODED audio reference (a later
//! reference-conditioning milestone) will need it, and how that is actually
//! done has to be settled from the reference implementation before porting
//! it, not guessed from the unused submodules' shapes.

use std::collections::HashMap;

use audio::conv::{ConvGemmKernels, ConvKernels, ConvScratch};
use gpu_core::{DeviceBuffer, Gpu};
use vae::blocks::Tensors;

const K_CONV1D: usize = 0;
const K_CONVTR1D: usize = 1;
const K_SNAKE_BETA: usize = 2;
const K_AXPY: usize = 3;
const K_ADD2: usize = 4;
const K_PAD1D_EDGE: usize = 5;
const K_ADD_CHAN_INPLACE: usize = 6;
const K_IM2COL1D_AT: usize = 7;
const K_MATMUL_REG3: usize = 8;
const K_MATMUL_DX_REG: usize = 9;
const K_MATMUL_DW_REG_SPLITK: usize = 10;
const K_NLC_BIAS_NCHW: usize = 11;
const K_COL2IM1D_BIAS: usize = 12;

const KERNELS: [(&str, &str); 13] = [
    ("conv1d", kernels::CONV1D),
    ("convtr1d", kernels::CONVTR1D),
    ("snake_beta", kernels::SNAKE_BETA),
    ("axpy", kernels::AXPY),
    ("add2", kernels::ADD2),
    ("pad1d_edge", kernels::PAD1D_EDGE),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("im2col1d_at", kernels::IM2COL1D_AT),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
    ("nlc_bias_nchw", kernels::NLC_BIAS_NCHW),
    ("col2im1d_bias", kernels::COL2IM1D_BIAS),
];

/// `SnakeBeta`'s fixed `eps` (`no_div_by_zero`, never a config field).
const SNAKE_EPS: f32 = 1e-9;
/// The antialiasing filters' fixed shape and ratio (`up_kernel_size`/
/// `down_kernel_size` both default to 12 in `Activation1d.__init__`,
/// confirmed against the real header's `[1,1,12]` filter tensors;
/// `up_ratio`/`down_ratio` both 2).
const AA_K: u32 = 12;
const AA_RATIO: u32 = 2;

/// Real `MiniMaxAI/MiniMax-H3` `audio_vae/model.safetensors` config
/// (`decoder_dim=1024`, `sample_rate=32000` -> the 32kHz `bigvgan_conf` branch
/// in `dac_audio_vae.py`), `vae_latent_channels=32` (`config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct VocoderConfig {
    /// The VAE bottleneck width `decode`'s input `z` arrives at (`config.json`'s
    /// `latent_channels`).
    pub vae_latent_channels: u32,
    pub upsample_initial_channel: u32,
    pub upsample_rates: [u32; 7],
    pub upsample_kernel_sizes: [u32; 7],
    pub resblock_kernel_sizes: [u32; 3],
    pub resblock_dilations: [[u32; 3]; 3],
    /// `dec_in_proj`'s OUTPUT width = `conv_pre`'s input width (BigVGAN's own
    /// `num_mels`, here fed a projected VAE latent rather than a real mel).
    pub mel_channels: u32,
    pub out_channels: u32,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self::h3_32khz()
    }
}

impl VocoderConfig {
    pub fn h3_32khz() -> VocoderConfig {
        VocoderConfig {
            vae_latent_channels: 32,
            upsample_initial_channel: 1024,
            upsample_rates: [5, 5, 2, 2, 2, 2, 2],
            upsample_kernel_sizes: [9, 9, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: [3, 7, 11],
            resblock_dilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            mel_channels: 2048,
            out_channels: 1,
        }
    }

    /// The number of upsample stages (7, `len(upsample_rates)`).
    pub fn num_upsamples(&self) -> usize {
        self.upsample_rates.len()
    }

    /// Channel width entering upsample stage `i` (`upsample_initial_channel >> i`).
    pub fn stage_cin(&self, i: usize) -> u32 {
        self.upsample_initial_channel >> i
    }

    /// Channel width leaving upsample stage `i` and feeding its resblocks.
    pub fn stage_cout(&self, i: usize) -> u32 {
        self.upsample_initial_channel >> (i + 1)
    }

    /// `activation_post`'s width - the channel count after every upsample
    /// stage (`upsample_initial_channel >> num_upsamples`, 8 at the real
    /// config).
    pub fn final_channels(&self) -> u32 {
        self.upsample_initial_channel >> self.num_upsamples()
    }

    /// The upsample rate product - `t` decoder-frame steps become `t*prod`
    /// samples.
    pub fn hop_length(&self) -> u32 {
        self.upsample_rates.iter().product()
    }

    /// Every tensor [`decode`] reads, POST weight-norm fold, in the
    /// checkpoint's own name space (`dec_in_proj.*` top-level, everything
    /// else under `decoder.`) - cross-checked leaf-by-leaf against the real
    /// header (914 raw tensors incl. `weight_g`/`weight_v` pairs, 779 after
    /// folding each pair to one `.weight`).
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m: Vec<(String, Vec<usize>)> = Vec::new();
        m.push(("dec_in_proj.weight".into(), vec![self.mel_channels as usize, self.vae_latent_channels as usize, 1]));
        m.push(("dec_in_proj.bias".into(), vec![self.mel_channels as usize]));

        m.push(("decoder.conv_pre.weight".into(), vec![self.upsample_initial_channel as usize, self.mel_channels as usize, 7]));
        m.push(("decoder.conv_pre.bias".into(), vec![self.upsample_initial_channel as usize]));

        let act = |m: &mut Vec<(String, Vec<usize>)>, prefix: &str, ch: usize| {
            m.push((format!("{prefix}.act.alpha"), vec![ch]));
            m.push((format!("{prefix}.act.beta"), vec![ch]));
            m.push((format!("{prefix}.upsample.filter"), vec![1, 1, AA_K as usize]));
            m.push((format!("{prefix}.downsample.lowpass.filter"), vec![1, 1, AA_K as usize]));
        };

        for i in 0..self.num_upsamples() {
            let (cin, cout) = (self.stage_cin(i), self.stage_cout(i));
            let k = self.upsample_kernel_sizes[i];
            m.push((format!("decoder.ups.{i}.0.weight"), vec![cin as usize, cout as usize, k as usize]));
            m.push((format!("decoder.ups.{i}.0.bias"), vec![cout as usize]));
            for r in 0..self.resblock_kernel_sizes.len() {
                let idx = i * self.resblock_kernel_sizes.len() + r;
                let k = self.resblock_kernel_sizes[r];
                let p = format!("decoder.resblocks.{idx}");
                for d in 0..3usize {
                    m.push((format!("{p}.convs1.{d}.weight"), vec![cout as usize, cout as usize, k as usize]));
                    m.push((format!("{p}.convs1.{d}.bias"), vec![cout as usize]));
                    m.push((format!("{p}.convs2.{d}.weight"), vec![cout as usize, cout as usize, k as usize]));
                    m.push((format!("{p}.convs2.{d}.bias"), vec![cout as usize]));
                    act(&mut m, &format!("{p}.activations.{}", 2 * d), cout as usize);
                    act(&mut m, &format!("{p}.activations.{}", 2 * d + 1), cout as usize);
                }
            }
        }
        act(&mut m, "decoder.activation_post", self.final_channels() as usize);
        m.push(("decoder.conv_post.weight".into(), vec![self.out_channels as usize, self.final_channels() as usize, 7]));
        m
    }
}

/// The direct kernels, used for the depthwise antialias convolutions and as
/// the selector's own structural fallback inside [`gemm_kernels`].
fn kernels_id() -> ConvKernels {
    ConvKernels { fwd: K_CONV1D, dx: 0, dw: 0 }
}
fn kernels_id_tr() -> ConvKernels {
    ConvKernels { fwd: K_CONVTR1D, dx: 0, dw: 0 }
}

/// The selected-lowering pipeline set for the `groups == 1` convolutions.
fn gemm_kernels(direct: ConvKernels) -> ConvGemmKernels {
    ConvGemmKernels {
        direct,
        bias: K_ADD_CHAN_INPLACE,
        im2col: K_IM2COL1D_AT,
        matmul: K_MATMUL_REG3,
        matmul_nn: K_MATMUL_DX_REG,
        matmul_tn: K_MATMUL_DW_REG_SPLITK,
        nlc_bias: K_NLC_BIAS_NCHW,
        col2im: K_COL2IM1D_BIAS,
    }
}

fn weight<'a>(t: &'a Tensors, name: &str) -> &'a [f32] {
    &t.get(name).unwrap_or_else(|| panic!("minimaxh3 vocoder: missing tensor {name}")).1
}

/// One synthesis run's device context: the handle, the weights, the shared
/// GEMM scratch and the antialias-filter cache. Same shape as the existing
/// in-workspace BigVGAN precedent this file reparameterizes.
struct Ctx<'a> {
    gpu: Gpu,
    t: &'a Tensors,
    scratch: ConvScratch,
    /// `(channels, the filter's 12 taps as bits) -> the depthwise weight`.
    filters: HashMap<(u32, Vec<u32>), DeviceBuffer>,
    /// A zeroed `[cout]` bias for the one convolution the checkpoint gives no
    /// bias tensor at all (`decoder.conv_post`, `use_bias_at_final: false`).
    zero_bias: HashMap<u32, DeviceBuffer>,
}

impl<'a> Ctx<'a> {
    fn new(device: Option<&str>, t: &'a Tensors) -> Ctx<'a> {
        Ctx { gpu: Gpu::open(device, &KERNELS), t, scratch: ConvScratch::new(), filters: HashMap::new(), zero_bias: HashMap::new() }
    }

    fn w(&self, name: &str) -> &'a [f32] {
        weight(self.t, name)
    }

    fn upload(&self, name: &str) -> DeviceBuffer {
        self.gpu.storage_init(name, self.w(name))
    }

    fn zero_bias_for(&mut self, cout: u32) -> DeviceBuffer {
        let gpu = &self.gpu;
        self.zero_bias.entry(cout).or_insert_with(|| gpu.storage_init("vocoder.zero_bias", &vec![0.0f32; cout as usize])).clone()
    }

    /// The checkpoint's shared `[1,1,12]` filter as a depthwise `[c,1,12]`
    /// weight, with `scale` folded into every tap (exact for the power-of-two
    /// `AA_RATIO` this vocoder uses).
    fn filter(&mut self, name: &str, c: u32, scale: f32) -> DeviceBuffer {
        let f = self.w(name);
        assert_eq!(f.len(), AA_K as usize, "{name}: {} values, expected {AA_K}", f.len());
        let key: Vec<u32> = f.iter().map(|v| (v * scale).to_bits()).collect();
        if let Some(b) = self.filters.get(&(c, key.clone())) {
            return b.clone();
        }
        let mut data = Vec::with_capacity((c * AA_K) as usize);
        for _ in 0..c {
            data.extend(f.iter().map(|v| v * scale));
        }
        let b = self.gpu.storage_init(name, &data);
        self.filters.insert((c, key), b.clone());
        b
    }

    /// Replicate-pad an NCL `[c,l]` DEVICE buffer by `(left,right)` samples per channel.
    fn pad_edge(&self, x: &DeviceBuffer, c: u32, l: u32, left: u32, right: u32) -> (DeviceBuffer, u32) {
        let lp = l + left + right;
        let total = c * lp;
        let y = self.gpu.storage(total as u64);
        self.gpu.submit(&[], &[self.gpu.step(K_PAD1D_EDGE, &[x, &y], &[total, l, left, right], total)]);
        (y, lp)
    }
}

/// Symmetric "same" plain `conv1d` (stride 1, `Lo == L`, `pad = dilation*(k-1)/2`).
fn conv1d_same(cx: &mut Ctx, prefix: &str, c: u32, k: u32, dilation: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let pad = dilation * (k - 1) / 2;
    let cfg = audio::conv::Conv1d { n: 1, cin: c, l, cout: c, k, stride: 1, pad, dilation, groups: 1, lo: l };
    let wgt = cx.upload(&format!("{prefix}.weight"));
    let bias = cx.upload(&format!("{prefix}.bias"));
    let y = cx.gpu.storage((c * l) as u64);
    let steps = audio::conv::conv1d_bias_fwd(&cx.gpu, &gemm_kernels(kernels_id()), &cfg, x, &wgt, &bias, &y, &mut cx.scratch);
    cx.gpu.submit(&[], &steps);
    y
}

/// A "same" plain `conv1d` at stride 1 that also changes channel count
/// (`dec_in_proj`, `conv_pre`, `conv_post`) - `pad = k/2`, exact for the odd
/// kernel sizes this file uses (1, 7). `has_bias=false` is `conv_post`'s
/// `use_bias_at_final: false` - it takes the same entry point against a
/// zeroed bias rather than a second code path, so the selector's decision is
/// shared.
fn conv1d_kx(cx: &mut Ctx, prefix: &str, cin: u32, cout: u32, k: u32, l: u32, has_bias: bool, x: &DeviceBuffer) -> DeviceBuffer {
    let cfg = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride: 1, pad: k / 2, dilation: 1, groups: 1, lo: l };
    let wgt = cx.upload(&format!("{prefix}.weight"));
    let bias = if has_bias { cx.upload(&format!("{prefix}.bias")) } else { cx.zero_bias_for(cout) };
    let y = cx.gpu.storage((cout * l) as u64);
    let steps = audio::conv::conv1d_bias_fwd(&cx.gpu, &gemm_kernels(kernels_id()), &cfg, x, &wgt, &bias, &y, &mut cx.scratch);
    cx.gpu.submit(&[], &steps);
    y
}

fn convtr1d(cx: &mut Ctx, prefix: &str, cin: u32, cout: u32, k: u32, stride: u32, l: u32, x: &DeviceBuffer) -> (DeviceBuffer, u32) {
    let pad = (k - stride) / 2;
    let lo = audio::conv::Conv1d::out_len_transposed(l, k, stride, pad, 0, 1);
    let cfg = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride, pad, dilation: 1, groups: 1, lo };
    let wgt = cx.upload(&format!("{prefix}.weight"));
    let bias = cx.upload(&format!("{prefix}.bias"));
    let y = cx.gpu.storage((cout * lo) as u64);
    let steps = audio::conv::convtr1d_bias_fwd(&cx.gpu, &gemm_kernels(kernels_id_tr()), &cfg, x, &wgt, &bias, &y, &mut cx.scratch);
    cx.gpu.submit(&[], &steps);
    (y, lo)
}

fn add(gpu: &Gpu, n: u32, a: &DeviceBuffer, b: &DeviceBuffer) -> DeviceBuffer {
    let y = gpu.storage(n as u64);
    gpu.submit(&[], &[gpu.step(K_ADD2, &[a, b, &y], &[n], n)]);
    y
}

/// `SnakeBeta` over NCL `[c,l]`.
fn snake_beta(cx: &Ctx, prefix: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let a = cx.upload(&format!("{prefix}.alpha"));
    let b = cx.upload(&format!("{prefix}.beta"));
    let total = c * l;
    let y = cx.gpu.storage(total as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(K_SNAKE_BETA, &[x, &a, &b, &y], &[total, c, l, SNAKE_EPS.to_bits()], total)]);
    y
}

/// `UpSample1d(ratio=2, kernel_size=12, window_type="kaiser")`: replicate-pad
/// `pad=5` each side, depthwise `ConvTranspose1d(k=12, stride=2)` against the
/// checkpoint filter scaled by `ratio=2`, cropping 15 off each side (folded
/// into the transposed conv's own `pad` parameter). Output length `2*l`.
fn antialias_upsample(cx: &mut Ctx, filter_name: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let pad = AA_K / AA_RATIO - 1; // 5
    let (pbuf, lp) = cx.pad_edge(x, c, l, pad, pad);
    let fbuf = cx.filter(filter_name, c, AA_RATIO as f32);
    let crop = pad * AA_RATIO + (AA_K - AA_RATIO) / 2; // 15
    let lo = audio::conv::Conv1d::out_len_transposed(lp, AA_K, AA_RATIO, crop, 0, 1);
    debug_assert_eq!(lo, 2 * l, "antialias_upsample: {lo}, expected {}", 2 * l);
    let cfg = audio::conv::Conv1d { n: 1, cin: c, l: lp, cout: c, k: AA_K, stride: AA_RATIO, pad: crop, dilation: 1, groups: c, lo };
    let y = cx.gpu.storage((c * lo) as u64);
    cx.gpu.submit(&[], &[audio::conv::convtr1d_fwd(&cx.gpu, &kernels_id_tr(), &cfg, &pbuf, &fbuf, &y)]);
    y
}

/// `DownSample1d(ratio=2)` == `LowPassFilter1d(cutoff=0.25, kernel_size=12)`:
/// replicate-pad `(left=5,right=6)`, depthwise `Conv1d(k=12,stride=2,pad=0)`
/// against the checkpoint's lowpass filter. Output length `l/2`.
fn antialias_downsample(cx: &mut Ctx, filter_name: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let (pl, pr) = (5u32, 6u32);
    let (pbuf, lp) = cx.pad_edge(x, c, l, pl, pr);
    let fbuf = cx.filter(filter_name, c, 1.0);
    let lo = (lp - AA_K) / AA_RATIO + 1;
    debug_assert_eq!(lo, l / 2, "antialias_downsample: {lo}, expected {}", l / 2);
    let cfg = audio::conv::Conv1d { n: 1, cin: c, l: lp, cout: c, k: AA_K, stride: AA_RATIO, pad: 0, dilation: 1, groups: c, lo };
    let y = cx.gpu.storage((c * lo) as u64);
    cx.gpu.submit(&[], &[audio::conv::conv1d_fwd(&cx.gpu, &kernels_id(), &cfg, &pbuf, &fbuf, &y)]);
    y
}

/// `Activation1d(SnakeBeta(c))`: antialiased upsample -> SnakeBeta ->
/// antialiased downsample. Length-preserving (`l` in, `l` out).
fn activation1d(cx: &mut Ctx, prefix: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let up = antialias_upsample(cx, &format!("{prefix}.upsample.filter"), c, l, x);
    let sn = snake_beta(cx, &format!("{prefix}.act"), c, 2 * l, &up);
    antialias_downsample(cx, &format!("{prefix}.downsample.lowpass.filter"), c, 2 * l, &sn)
}

/// One `AMPBlock1`: 3 `(act1 -> conv1 -> act2 -> conv2 -> residual-add)`
/// steps at dilations `(1,3,5)` (`convs2` always dilation 1), kernel `k`.
/// `act1`/`act2` read `activations.{2*d}`/`activations.{2*d+1}` - the
/// checkpoint's own flat 6-element list, sliced the same way
/// `AMPBlock1.forward`'s `self.activations[::2]`/`[1::2]` does.
fn amp_block(cx: &mut Ctx, prefix: &str, c: u32, l: u32, k: u32, dilations: [u32; 3], x: &DeviceBuffer) -> DeviceBuffer {
    let mut cur = x.clone();
    for (d, &dilation) in dilations.iter().enumerate() {
        let a1 = activation1d(cx, &format!("{prefix}.activations.{}", 2 * d), c, l, &cur);
        let c1 = conv1d_same(cx, &format!("{prefix}.convs1.{d}"), c, k, dilation, l, &a1);
        let a2 = activation1d(cx, &format!("{prefix}.activations.{}", 2 * d + 1), c, l, &c1);
        let c2 = conv1d_same(cx, &format!("{prefix}.convs2.{d}"), c, k, 1, l, &a2);
        cur = add(&cx.gpu, c * l, &cur, &c2);
    }
    cur
}

/// Zero an accumulator then `AXPY` `s*y` into it - the device-side mean of
/// the 3 parallel resblocks (`torch.stack(...).mean(dim=0)`).
fn axpy_into(gpu: &Gpu, acc: &DeviceBuffer, s: f32, y: &DeviceBuffer, n: u32) {
    gpu.submit(&[], &[gpu.step(K_AXPY, &[acc, y], &[n, s.to_bits()], n)]);
}

/// Decode a `[vae_latent_channels, t]` row-major VAE latent into a mono
/// waveform, matching the checkpoint's own `DacAudioVAE.decode`: `z =
/// dec_in_proj(z); return decoder(z)`.
///
/// Returns `[t * hop_length()]` samples, clamped to `[-1, 1]`
/// (`use_tanh_at_final=false` in this checkpoint's own BigVGAN config - the
/// final activation is `clamp`, NOT `tanh`).
pub fn decode(cfg: &VocoderConfig, tensors: &Tensors, z: &[f32], t: u32, device: Option<&str>) -> Vec<f32> {
    assert_eq!(z.len(), (cfg.vae_latent_channels * t) as usize, "decode: {} values, expected {}", z.len(), cfg.vae_latent_channels * t);

    let mut cx = Ctx::new(device, tensors);
    let x_in = cx.gpu.storage_init("vocoder.z_in", z);
    let proj = conv1d_kx(&mut cx, "dec_in_proj", cfg.vae_latent_channels, cfg.mel_channels, 1, t, true, &x_in);
    let mut h = conv1d_kx(&mut cx, "decoder.conv_pre", cfg.mel_channels, cfg.upsample_initial_channel, 7, t, true, &proj);
    let mut l = t;

    for i in 0..cfg.num_upsamples() {
        let (cin, cout) = (cfg.stage_cin(i), cfg.stage_cout(i));
        let (up, lo) = convtr1d(&mut cx, &format!("decoder.ups.{i}.0"), cin, cout, cfg.upsample_kernel_sizes[i], cfg.upsample_rates[i], l, &h);
        l = lo;
        let acc = cx.gpu.storage((cout * l) as u64);
        cx.gpu.write_f32(&acc, &vec![0.0f32; (cout * l) as usize]);
        for r in 0..cfg.resblock_kernel_sizes.len() {
            let idx = i * cfg.resblock_kernel_sizes.len() + r;
            let y = amp_block(&mut cx, &format!("decoder.resblocks.{idx}"), cout, l, cfg.resblock_kernel_sizes[r], cfg.resblock_dilations[r], &up);
            axpy_into(&cx.gpu, &acc, 1.0 / 3.0, &y, cout * l);
        }
        h = acc;
    }

    let fin = cfg.final_channels();
    h = activation1d(&mut cx, "decoder.activation_post", fin, l, &h);
    h = conv1d_kx(&mut cx, "decoder.conv_post", fin, cfg.out_channels, 7, l, false, &h);

    let mut wave = cx.gpu.read(&h, (cfg.out_channels * l) as usize);
    for v in &mut wave {
        *v = v.clamp(-1.0, 1.0);
    }
    wave
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_counts_the_shipped_checkpoint() {
        let m = VocoderConfig::h3_32khz().tensor_manifest();
        // 914 raw tensors under decoder.*+dec_in_proj.* in the real header
        // (incl. weight_g/weight_v pairs), 779 after folding each pair to
        // one .weight - counted directly against the real safetensors header.
        assert_eq!(m.len(), 779, "manifest has {} tensors", m.len());
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name in the manifest");
        assert!(names.contains("dec_in_proj.weight"));
        assert!(names.contains("decoder.conv_pre.weight"));
        assert!(names.contains("decoder.ups.0.0.weight"));
        assert!(names.contains("decoder.resblocks.20.activations.5.act.alpha"));
        assert!(names.contains("decoder.activation_post.act.alpha"));
        assert!(!names.contains("decoder.conv_post.bias"), "use_bias_at_final=false: conv_post must have no bias");
        assert!(!names.contains("decoder.resblocks.21.convs1.0.weight"), "only 21 resblocks (0..=20)");

        let get = |n: &str| m.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(get("dec_in_proj.weight"), vec![2048, 32, 1]);
        assert_eq!(get("decoder.conv_pre.weight"), vec![1024, 2048, 7]);
        assert_eq!(get("decoder.ups.0.0.weight"), vec![1024, 512, 9]);
        assert_eq!(get("decoder.ups.6.0.weight"), vec![16, 8, 4]);
        assert_eq!(get("decoder.resblocks.0.convs1.0.weight"), vec![512, 512, 3]);
        assert_eq!(get("decoder.resblocks.20.convs1.0.weight"), vec![8, 8, 11]);
        assert_eq!(get("decoder.conv_post.weight"), vec![1, 8, 7]);
    }

    /// The upsample rate product (800) is this decoder's hop length - `t`
    /// latent steps become `t*800` samples at 32kHz.
    #[test]
    fn hop_length_and_final_channels_match_the_real_config() {
        let cfg = VocoderConfig::h3_32khz();
        assert_eq!(cfg.hop_length(), 800);
        assert_eq!(cfg.final_channels(), 8);
        assert_eq!(cfg.num_upsamples(), 7);
    }

    /// Weight-free tiny-config smoke test (porting.md's own ordering: this
    /// rung runs BEFORE any real checkpoint is available) at a small but
    /// still-real 7-stage/3-resblock-per-stage shape, seeded random weights
    /// standing in for the real ones. Exercises the entire kernel-dispatch
    /// graph this file wires - `conv1d`/`convtr1d` (direct AND selector-
    /// lowered widths), `snake_beta`, the antialias replicate-pad + depthwise
    /// transposed/plain conv pair, `axpy`-based resblock averaging, and the
    /// final clamp - none of which the manifest test above touches at all.
    fn tiny_cfg() -> VocoderConfig {
        VocoderConfig {
            vae_latent_channels: 4,
            // Must be >= 2^7 = 128: `stage_cout` halves the channel count
            // once per upsample stage (7 of them), and `final_channels`
            // would silently truncate to 0 below that via integer
            // right-shift - caught by this test's own
            // "must not be trivially all-zero" assertion the first time
            // this constant was too small.
            upsample_initial_channel: 128,
            upsample_rates: [2, 2, 2, 2, 2, 2, 2],
            upsample_kernel_sizes: [4, 4, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: [3, 5, 7],
            resblock_dilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            mel_channels: 12,
            out_channels: 1,
        }
    }

    fn rand_tensors(cfg: &VocoderConfig, seed: u64) -> Tensors {
        let mut rng = data::rng::Lcg::new(seed);
        cfg.tensor_manifest()
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                let vals = rng.vec_scaled(n, 0.3);
                (name, (shape, vals))
            })
            .collect()
    }

    #[test]
    fn tiny_config_decode_is_finite_and_the_right_shape() {
        let cfg = tiny_cfg();
        let tensors = rand_tensors(&cfg, 7);
        let z: Vec<f32> = (0..cfg.vae_latent_channels * 3).map(|i| (i as f32 * 0.037).sin() * 0.3).collect();
        let wave = decode(&cfg, &tensors, &z, 3, Some("cpu"));
        assert_eq!(wave.len(), (cfg.out_channels * 3 * cfg.hop_length()) as usize);
        assert!(wave.iter().all(|v| v.is_finite()), "decode output must be finite");
        assert!(wave.iter().all(|&v| (-1.0..=1.0).contains(&v)), "decode output must be clamped to [-1,1]");
        assert!(wave.iter().any(|&v| v != 0.0), "decode output must not be trivially all-zero");
    }
}
