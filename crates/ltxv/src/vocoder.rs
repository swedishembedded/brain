// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The LTX-2.5 BASE vocoder (BigVGAN v2 topology, AMP1/snakebeta resblocks):
//! `conv_pre -> 6 upsample stages (ConvTranspose1d + 3 parallel AMPBlock1
//! resblocks averaged) -> act_post -> conv_post -> clamp`.
//!
//! Ported from `ltx_core.model.audio_vae.vocoder.{Vocoder,AMPBlock1,
//! Activation1d,UpSample1d,DownSample1d,LowPassFilter1d,SnakeBeta}`, real
//! weights (`ltx-2.5-audio-vae-bf16.safetensors`'s `vocoder.vocoder.*` keys -
//! same file as [`crate::audio_vae`], a DIFFERENT tensor subset). The
//! bandwidth-extension stage (`VocoderWithBWE`/`vocoder.bwe_generator.*`/
//! `vocoder.mel_stft.*`) is explicitly OUT OF SCOPE for this milestone - see
//! `tools/goldens/ltxv_audio_dump_reference.py`'s own module doc for why (no
//! ISTFT anywhere in scope, deliberately).
//!
//! Conventions pinned by reading `vocoder.py` directly rather than guessed
//! from the topology name:
//!
//! * **`use_tanh_at_final: false` in the real checkpoint config
//!   (`config.vocoder.vocoder`)**, so the final activation is `torch.clamp(x,
//!   -1, 1)`, NOT `tanh` - `Vocoder.forward`'s own branch
//!   (`torch.tanh(x) if self.use_tanh_at_final else torch.clamp(x, -1, 1)`),
//!   easy to get backwards guessing from the class name alone.
//! * **`use_bias_at_final: false`**: `conv_post` has no bias tensor at all
//!   (confirmed against the real header - no `conv_post.bias` key), unlike
//!   every other conv in this model.
//! * **The anti-aliased `Activation1d` (BigVGAN v2's own "AMP" contribution)
//!   wraps every SnakeBeta call**: 2x nearest-free upsample (a REPLICATE-
//!   padded depthwise `ConvTranspose1d` against a fixed Kaiser-sinc filter,
//!   scaled by the ratio, then cropped) -> SnakeBeta -> 2x depthwise
//!   `Conv1d` lowpass-and-decimate (also replicate-padded) back down - a
//!   length-preserving resample-filter-resample trick for antialiasing, NOT
//!   an actual up/downsample of the signal's rate. The two Kaiser-sinc
//!   filters (`*.upsample.filter`/`*.downsample.lowpass.filter`, both
//!   `[1,1,12]`) are checkpoint-loaded PERSISTENT BUFFERS, not learned and
//!   not derived here - `kaiser_sinc_filter1d`'s own formula is never ported
//!   to Rust; the buffer's real values are imported and used directly as a
//!   depthwise conv weight (replicated across the channel dim on the host,
//!   since `nn.Module.expand` is a broadcast view of one shared filter, not
//!   per-channel data).
//! * **`conv1d`/`convtr1d`'s existing `pad` param already covers every
//!   padding shape this vocoder needs except the antialiasing filters'
//!   REPLICATE edges**: symmetric "same" padding (`conv_pre`, `conv_post`,
//!   every `AMPBlock1` conv) reduces to a single LOW-side `pad` with
//!   `Lo == L` (`conv1d.wgsl`'s own contract: taps past `L` are implicitly
//!   zero, which is exactly what a symmetric-pad "same" conv needs at
//!   `stride=1`); the 6 `ups.*` `ConvTranspose1d`s use `convtr1d.wgsl`'s
//!   native symmetric `pad` directly (same convention `crates/mimi`'s
//!   `causal_convtr_sym` already established for Qwen3-Omni's SEANet).
//!   REPLICATE edge padding has no existing kernel, so it is done as plain
//!   host arithmetic (read back, extend edges, reupload) - the same
//!   "cheap non-hot-path op on the host" precedent `crates/mimi`'s own
//!   `enc_downsample` sets for an identical replicate-pad.
//! * **Resblock averaging is a device accumulate**, not a host reduction:
//!   the 3 resblocks per upsample stage run over the SAME input and their
//!   outputs are meaned (`torch.stack(...).mean(dim=0)`) - implemented with
//!   [`kernels::AXPY`] (`out += s*in`, already used by `crates/mimi` for an
//!   identical mean-of-groups accumulation) at `s = 1/3`, zeroing the
//!   accumulator first.
//!
//! Zero new kernels: `conv1d`/`convtr1d` (`crates/audio/src/conv.rs`),
//! `snake_beta` (already wired up in `crates/mimi/src/model.rs`), `axpy`,
//! `add2`. Eager dispatch, same style as [`crate::audio_vae`] and
//! `crates/mimi/src/model.rs`.

use gpu_core::{DeviceBuffer, Gpu};
use vae::blocks::Tensors;

const K_CONV1D: usize = 0;
const K_CONVTR1D: usize = 1;
const K_SNAKE_BETA: usize = 2;
const K_AXPY: usize = 3;
const K_ADD2: usize = 4;

const KERNELS: [(&str, &str); 5] = [
    ("conv1d", kernels::CONV1D),
    ("convtr1d", kernels::CONVTR1D),
    ("snake_beta", kernels::SNAKE_BETA),
    ("axpy", kernels::AXPY),
    ("add2", kernels::ADD2),
];

/// `SnakeBeta`'s fixed `eps` (`no_div_by_zero`, never a config field).
const SNAKE_EPS: f32 = 1e-9;
/// The antialiasing filters' fixed shape (`up_kernel_size`/`down_kernel_size`
/// both default to 12 in `Activation1d.__init__`, confirmed against the real
/// header's `[1,1,12]` filter tensors) and ratio (`up_ratio`/`down_ratio`
/// both 2).
const AA_K: u32 = 12;
const AA_RATIO: u32 = 2;

/// Real `ltx-2.5-audio-vae-bf16.safetensors` config (`config.vocoder.vocoder`,
/// the nested ltx-2.3+ layout - `bwe` sibling ignored, out of scope):
/// `upsample_initial_channel=1536`, `resblock="AMP1"`,
/// `upsample_rates=[5,2,2,2,2,2]` (product 160 = the mel hop length),
/// `upsample_kernel_sizes=[11,4,4,4,4,4]`, `resblock_kernel_sizes=[3,7,11]`,
/// `resblock_dilation_sizes=[[1,3,5]]x3`, stereo (`in_channels=128 = 2 audio
/// channels * 64 mel bins`, `out_channels=2`), `use_tanh_at_final=false`,
/// `use_bias_at_final=false`, `activation="snakebeta"`.
#[derive(Clone, Debug, PartialEq)]
pub struct VocoderConfig {
    pub upsample_initial_channel: u32,
    pub upsample_rates: [u32; 6],
    pub upsample_kernel_sizes: [u32; 6],
    pub resblock_kernel_sizes: [u32; 3],
    pub resblock_dilations: [[u32; 3]; 3],
    pub mel_channels: u32,
    pub out_channels: u32,
}

impl Default for VocoderConfig {
    fn default() -> Self {
        Self::ltx25()
    }
}

impl VocoderConfig {
    pub fn ltx25() -> VocoderConfig {
        VocoderConfig {
            upsample_initial_channel: 1536,
            upsample_rates: [5, 2, 2, 2, 2, 2],
            upsample_kernel_sizes: [11, 4, 4, 4, 4, 4],
            resblock_kernel_sizes: [3, 7, 11],
            resblock_dilations: [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            mel_channels: 128,
            out_channels: 2,
        }
    }

    /// The number of upsample stages (6, `len(upsample_rates)`).
    pub fn num_upsamples(&self) -> usize {
        self.upsample_rates.len()
    }

    /// Channel width entering upsample stage `i` (`upsample_initial_channel
    /// >> i`).
    pub fn stage_cin(&self, i: usize) -> u32 {
        self.upsample_initial_channel >> i
    }

    /// Channel width leaving upsample stage `i` and feeding its resblocks.
    pub fn stage_cout(&self, i: usize) -> u32 {
        self.upsample_initial_channel >> (i + 1)
    }

    /// `act_post`'s width - the channel count after every upsample stage
    /// (`upsample_initial_channel >> num_upsamples`, 24 at the real config).
    pub fn final_channels(&self) -> u32 {
        self.upsample_initial_channel >> self.num_upsamples()
    }

    /// Every tensor this model reads, in the checkpoint's OWN bare name space
    /// AFTER stripping the doubled `vocoder.vocoder.` prefix (see
    /// [`crate::import`]) - cross-checked leaf-by-leaf against the real
    /// header (667 tensors).
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m: Vec<(String, Vec<usize>)> = Vec::new();
        m.push(("conv_pre.weight".into(), vec![self.upsample_initial_channel as usize, self.mel_channels as usize, 7]));
        m.push(("conv_pre.bias".into(), vec![self.upsample_initial_channel as usize]));

        let act = |m: &mut Vec<(String, Vec<usize>)>, prefix: &str, ch: usize| {
            m.push((format!("{prefix}.act.alpha"), vec![ch]));
            m.push((format!("{prefix}.act.beta"), vec![ch]));
            m.push((format!("{prefix}.upsample.filter"), vec![1, 1, AA_K as usize]));
            m.push((format!("{prefix}.downsample.lowpass.filter"), vec![1, 1, AA_K as usize]));
        };

        for i in 0..self.num_upsamples() {
            let (cin, cout) = (self.stage_cin(i), self.stage_cout(i));
            let k = self.upsample_kernel_sizes[i];
            m.push((format!("ups.{i}.weight"), vec![cin as usize, cout as usize, k as usize]));
            m.push((format!("ups.{i}.bias"), vec![cout as usize]));
            for r in 0..self.resblock_kernel_sizes.len() {
                let idx = i * self.resblock_kernel_sizes.len() + r;
                let k = self.resblock_kernel_sizes[r];
                let p = format!("resblocks.{idx}");
                for d in 0..3usize {
                    m.push((format!("{p}.convs1.{d}.weight"), vec![cout as usize, cout as usize, k as usize]));
                    m.push((format!("{p}.convs1.{d}.bias"), vec![cout as usize]));
                    m.push((format!("{p}.convs2.{d}.weight"), vec![cout as usize, cout as usize, k as usize]));
                    m.push((format!("{p}.convs2.{d}.bias"), vec![cout as usize]));
                    act(&mut m, &format!("{p}.acts1.{d}"), cout as usize);
                    act(&mut m, &format!("{p}.acts2.{d}"), cout as usize);
                }
            }
        }
        act(&mut m, "act_post", self.final_channels() as usize);
        m.push(("conv_post.weight".into(), vec![self.out_channels as usize, self.final_channels() as usize, 7]));
        m
    }
}

fn new_gpu(device: Option<&str>) -> Gpu {
    match device {
        Some("cpu") => Gpu::new_cpu(&KERNELS),
        Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
        _ => Gpu::new(&KERNELS),
    }
}

fn kernels_id() -> audio::conv::ConvKernels {
    audio::conv::ConvKernels { fwd: K_CONV1D, dx: 0, dw: 0 }
}
fn kernels_id_tr() -> audio::conv::ConvKernels {
    audio::conv::ConvKernels { fwd: K_CONVTR1D, dx: 0, dw: 0 }
}

fn weight<'a>(t: &'a Tensors, name: &str) -> &'a [f32] {
    &t.get(name).unwrap_or_else(|| panic!("ltxv vocoder: missing tensor {name}")).1
}

/// Per-channel bias broadcast over an NCL `[c,l]` buffer, then `add2` - the
/// same host-broadcast-then-device-add shape `crates/mimi::model::
/// add_ncl_bias` uses.
fn add_ncl_bias(gpu: &Gpu, t: &Tensors, x: &DeviceBuffer, bias_name: &str, c: u32, l: u32) -> DeviceBuffer {
    let bias = weight(t, bias_name);
    let mut bcast = vec![0.0f32; (c * l) as usize];
    for (ch, row) in bcast.chunks_mut(l as usize).enumerate() {
        row.fill(bias[ch]);
    }
    let bbuf = gpu.storage_init(bias_name, &bcast);
    let y = gpu.storage((c * l) as u64);
    gpu.submit(&[], &[gpu.step(K_ADD2, &[x, &bbuf, &y], &[c * l], c * l)]);
    y
}

/// Symmetric "same" `conv1d` (stride 1, `Lo == L`): `pad = dilation*(k-1)/2`.
fn conv1d_same(gpu: &Gpu, t: &Tensors, prefix: &str, c: u32, k: u32, dilation: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let pad = dilation * (k - 1) / 2;
    let cfg = audio::conv::Conv1d { n: 1, cin: c, l, cout: c, k, stride: 1, pad, dilation, groups: 1, lo: l };
    let wgt = gpu.storage_init(&format!("{prefix}.weight"), weight(t, &format!("{prefix}.weight")));
    let y = gpu.storage((c * l) as u64);
    gpu.submit(&[], &[audio::conv::conv1d_fwd(gpu, &kernels_id(), &cfg, x, &wgt, &y)]);
    add_ncl_bias(gpu, t, &y, &format!("{prefix}.bias"), c, l)
}

/// `conv_pre`/`conv_post`: `128 -> upsample_initial_channel` / `final_channels
/// -> 2`, kernel 7, native `padding=3` (also a symmetric "same" conv at
/// stride 1). `has_bias=false` skips [`add_ncl_bias`] (`conv_post`'s
/// `use_bias_at_final: false`).
fn conv1d_k7(gpu: &Gpu, t: &Tensors, prefix: &str, cin: u32, cout: u32, l: u32, has_bias: bool, x: &DeviceBuffer) -> DeviceBuffer {
    let cfg = audio::conv::Conv1d { n: 1, cin, l, cout, k: 7, stride: 1, pad: 3, dilation: 1, groups: 1, lo: l };
    let wgt = gpu.storage_init(&format!("{prefix}.weight"), weight(t, &format!("{prefix}.weight")));
    let y = gpu.storage((cout * l) as u64);
    gpu.submit(&[], &[audio::conv::conv1d_fwd(gpu, &kernels_id(), &cfg, x, &wgt, &y)]);
    if has_bias {
        add_ncl_bias(gpu, t, &y, &format!("{prefix}.bias"), cout, l)
    } else {
        y
    }
}

fn convtr1d(gpu: &Gpu, t: &Tensors, prefix: &str, cin: u32, cout: u32, k: u32, stride: u32, l: u32, x: &DeviceBuffer) -> (DeviceBuffer, u32) {
    let pad = (k - stride) / 2;
    let lo = audio::conv::Conv1d::out_len_transposed(l, k, stride, pad, 0, 1);
    let cfg = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride, pad, dilation: 1, groups: 1, lo };
    let wgt = gpu.storage_init(&format!("{prefix}.weight"), weight(t, &format!("{prefix}.weight")));
    let y = gpu.storage((cout * lo) as u64);
    gpu.submit(&[], &[audio::conv::convtr1d_fwd(gpu, &kernels_id_tr(), &cfg, x, &wgt, &y)]);
    (add_ncl_bias(gpu, t, &y, &format!("{prefix}.bias"), cout, lo), lo)
}

fn add(gpu: &Gpu, n: u32, a: &DeviceBuffer, b: &DeviceBuffer) -> DeviceBuffer {
    let y = gpu.storage(n as u64);
    gpu.submit(&[], &[gpu.step(K_ADD2, &[a, b, &y], &[n], n)]);
    y
}

/// `SnakeBeta` over NCL `[c,l]`.
fn snake_beta(gpu: &Gpu, t: &Tensors, prefix: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let a = gpu.storage_init(&format!("{prefix}.alpha"), weight(t, &format!("{prefix}.alpha")));
    let b = gpu.storage_init(&format!("{prefix}.beta"), weight(t, &format!("{prefix}.beta")));
    let total = c * l;
    let y = gpu.storage(total as u64);
    gpu.submit(&[], &[gpu.step(K_SNAKE_BETA, &[x, &a, &b, &y], &[total, c, l, SNAKE_EPS.to_bits()], total)]);
    y
}

/// Replicate-pad an NCL `[c,l]` HOST array by `(left,right)` samples per
/// channel (edge value repeated) - the one op in this vocoder with no device
/// kernel (see this module's header).
fn replicate_pad_host(x: &[f32], c: usize, l: usize, left: usize, right: usize) -> Vec<f32> {
    let lp = left + l + right;
    let mut out = vec![0.0f32; c * lp];
    for ch in 0..c {
        let src = &x[ch * l..(ch + 1) * l];
        let dst = &mut out[ch * lp..(ch + 1) * lp];
        dst[..left].fill(src[0]);
        dst[left..left + l].copy_from_slice(src);
        dst[left + l..].fill(src[l - 1]);
    }
    out
}

/// The checkpoint's shared `[1,1,12]` filter, replicated to `[c,1,12]` (one
/// depthwise-group weight per channel, all identical - `nn.Module.expand` is
/// a broadcast view of the SAME 12 values, not per-channel data).
fn replicate_filter(t: &Tensors, name: &str, c: u32) -> Vec<f32> {
    let f = weight(t, name);
    assert_eq!(f.len(), AA_K as usize, "{name}: {} values, expected {AA_K}", f.len());
    let mut out = Vec::with_capacity((c * AA_K) as usize);
    for _ in 0..c {
        out.extend_from_slice(f);
    }
    out
}

/// `UpSample1d(ratio=2, kernel_size=12, window_type="kaiser")`: replicate-pad
/// `pad=kernel_size/ratio-1=5` each side, depthwise `ConvTranspose1d(k=12,
/// stride=2,pad=0)` against the checkpoint filter, scale by `ratio=2`, crop
/// `pad_left=pad_right=15` off each side. Output length `2*l`.
fn antialias_upsample(gpu: &Gpu, t: &Tensors, filter_name: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let pad = AA_K / AA_RATIO - 1; // 5
    let host = gpu.read(x, (c * l) as usize);
    let padded = replicate_pad_host(&host, c as usize, l as usize, pad as usize, pad as usize);
    let lp = l + 2 * pad;
    let pbuf = gpu.storage_init("aa_up.padded", &padded);
    let filt = replicate_filter(t, filter_name, c);
    let fbuf = gpu.storage_init(filter_name, &filt);
    let lo_native = audio::conv::Conv1d::out_len_transposed(lp, AA_K, AA_RATIO, 0, 0, 1);
    let cfg = audio::conv::Conv1d { n: 1, cin: c, l: lp, cout: c, k: AA_K, stride: AA_RATIO, pad: 0, dilation: 1, groups: c, lo: lo_native };
    let y = gpu.storage((c * lo_native) as u64);
    gpu.submit(&[], &[audio::conv::convtr1d_fwd(gpu, &kernels_id_tr(), &cfg, &pbuf, &fbuf, &y)]);
    let native = gpu.read(&y, (c * lo_native) as usize);
    // `UpSample1d.__init__`: pad_left = pad*stride + (k-stride)//2, pad_right =
    // pad*stride + (k-stride+1)//2 - both equal 15 at (pad=5,stride=2,k=12)
    // since (k-stride) is even, so ONE `crop` covers both sides.
    let crop = (pad * AA_RATIO + (AA_K - AA_RATIO) / 2) as usize;
    let out_l = (lo_native as usize) - 2 * crop;
    let mut cropped = vec![0.0f32; c as usize * out_l];
    for ch in 0..c as usize {
        let src = &native[ch * lo_native as usize + crop..ch * lo_native as usize + crop + out_l];
        cropped[ch * out_l..(ch + 1) * out_l].copy_from_slice(src);
        for v in &mut cropped[ch * out_l..(ch + 1) * out_l] {
            *v *= 2.0; // `x = self.ratio * conv_transpose1d(...)` - the *ratio* scale
        }
    }
    debug_assert_eq!(out_l as u32, 2 * l, "antialias_upsample: {out_l}, expected {}", 2 * l);
    gpu.storage_init("aa_up.out", &cropped)
}

/// `DownSample1d(ratio=2)` == `LowPassFilter1d(cutoff=0.25,kernel_size=12)`:
/// replicate-pad `(left=5,right=6)`, depthwise `Conv1d(k=12,stride=2,pad=0)`
/// against the checkpoint's lowpass filter. Output length `l/2` (`l` is the
/// ALREADY-upsampled length, `2x` the block's nominal width).
fn antialias_downsample(gpu: &Gpu, t: &Tensors, filter_name: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let (pl, pr) = (5u32, 6u32); // LowPassFilter1d: pad_left = k/2 - 1, pad_right = k/2 (k=12, even)
    let host = gpu.read(x, (c * l) as usize);
    let padded = replicate_pad_host(&host, c as usize, l as usize, pl as usize, pr as usize);
    let lp = l + pl + pr;
    let pbuf = gpu.storage_init("aa_down.padded", &padded);
    let filt = replicate_filter(t, filter_name, c);
    let fbuf = gpu.storage_init(filter_name, &filt);
    let lo = (lp - AA_K) / AA_RATIO + 1;
    let cfg = audio::conv::Conv1d { n: 1, cin: c, l: lp, cout: c, k: AA_K, stride: AA_RATIO, pad: 0, dilation: 1, groups: c, lo };
    let y = gpu.storage((c * lo) as u64);
    gpu.submit(&[], &[audio::conv::conv1d_fwd(gpu, &kernels_id(), &cfg, &pbuf, &fbuf, &y)]);
    debug_assert_eq!(lo, l / 2, "antialias_downsample: {lo}, expected {}", l / 2);
    y
}

/// `Activation1d(SnakeBeta(c))`: antialiased upsample -> SnakeBeta ->
/// antialiased downsample. Length-preserving (`l` in, `l` out).
fn activation1d(gpu: &Gpu, t: &Tensors, prefix: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let up = antialias_upsample(gpu, t, &format!("{prefix}.upsample.filter"), c, l, x);
    let sn = snake_beta(gpu, t, &format!("{prefix}.act"), c, 2 * l, &up);
    antialias_downsample(gpu, t, &format!("{prefix}.downsample.lowpass.filter"), c, 2 * l, &sn)
}

/// One `AMPBlock1`: 3 (act1 -> conv1 -> act2 -> conv2 -> residual-add) steps
/// at dilations `(1,3,5)` (`convs2` always dilation 1), kernel `k`.
fn amp_block(gpu: &Gpu, t: &Tensors, prefix: &str, c: u32, l: u32, k: u32, dilations: [u32; 3], x: &DeviceBuffer) -> DeviceBuffer {
    let mut cur = x.clone();
    for (d, &dilation) in dilations.iter().enumerate() {
        let a1 = activation1d(gpu, t, &format!("{prefix}.acts1.{d}"), c, l, &cur);
        let c1 = conv1d_same(gpu, t, &format!("{prefix}.convs1.{d}"), c, k, dilation, l, &a1);
        let a2 = activation1d(gpu, t, &format!("{prefix}.acts2.{d}"), c, l, &c1);
        let c2 = conv1d_same(gpu, t, &format!("{prefix}.convs2.{d}"), c, k, 1, l, &a2);
        cur = add(gpu, c * l, &cur, &c2);
    }
    cur
}

/// Zero an accumulator then `AXPY` `s*y` into it - the device-side mean of
/// the 3 parallel resblocks (`torch.stack(...).mean(dim=0)`).
fn axpy_into(gpu: &Gpu, acc: &DeviceBuffer, s: f32, y: &DeviceBuffer, n: u32) {
    gpu.submit(&[], &[gpu.step(K_AXPY, &[acc, y], &[n, s.to_bits()], n)]);
}

/// Synthesize a stereo waveform from a reconstructed log-mel spectrogram.
///
/// `mel` is `[channels=2, t, mel_bins=64]` row-major (the audio VAE decoder's
/// own `recon_mel` layout) - `Vocoder.forward`'s own
/// `x.transpose(2,3)` + `einops.rearrange("b s c t -> b (s c) t")` is exactly
/// a host reshape into the `[128, t]` NCL input (`ch = stereo*mel_bins +
/// freq_bin`), done here rather than as a device permute since it runs once
/// per call on a tiny buffer.
///
/// Returns `[out_channels, t*upsample_product]` row-major (`t*160` samples at
/// the real config, matching `mel_hop_length`).
pub fn synthesize(cfg: &VocoderConfig, tensors: &Tensors, mel: &[f32], channels: u32, t: u32, mel_bins: u32, device: Option<&str>) -> Vec<f32> {
    assert_eq!(mel.len(), (channels * t * mel_bins) as usize, "synthesize: {} values, expected {}", mel.len(), channels * t * mel_bins);
    assert_eq!(channels * mel_bins, cfg.mel_channels, "channels*mel_bins {} != mel_channels {}", channels * mel_bins, cfg.mel_channels);

    // (channels, t, mel_bins) -> (channels*mel_bins, t), stereo outer / freq inner.
    let mut nc = vec![0.0f32; (cfg.mel_channels * t) as usize];
    for s in 0..channels as usize {
        for ti in 0..t as usize {
            for fb in 0..mel_bins as usize {
                let ch = s * mel_bins as usize + fb;
                nc[ch * t as usize + ti] = mel[(s * t as usize + ti) * mel_bins as usize + fb];
            }
        }
    }

    let gpu = new_gpu(device);
    let x_in = gpu.storage_init("vocoder.mel_in", &nc);
    let mut h = conv1d_k7(&gpu, tensors, "conv_pre", cfg.mel_channels, cfg.upsample_initial_channel, t, true, &x_in);
    let mut l = t;

    for i in 0..cfg.num_upsamples() {
        let (cin, cout) = (cfg.stage_cin(i), cfg.stage_cout(i));
        let (up, lo) = convtr1d(&gpu, tensors, &format!("ups.{i}"), cin, cout, cfg.upsample_kernel_sizes[i], cfg.upsample_rates[i], l, &h);
        l = lo;
        let acc = gpu.storage((cout * l) as u64);
        gpu.write_f32(&acc, &vec![0.0f32; (cout * l) as usize]);
        for r in 0..cfg.resblock_kernel_sizes.len() {
            let idx = i * cfg.resblock_kernel_sizes.len() + r;
            let y = amp_block(&gpu, tensors, &format!("resblocks.{idx}"), cout, l, cfg.resblock_kernel_sizes[r], cfg.resblock_dilations[r], &up);
            axpy_into(&gpu, &acc, 1.0 / 3.0, &y, cout * l);
        }
        h = acc;
    }

    let fin = cfg.final_channels();
    h = activation1d(&gpu, tensors, "act_post", fin, l, &h);
    h = conv1d_k7(&gpu, tensors, "conv_post", fin, cfg.out_channels, l, false, &h);

    let mut wave = gpu.read(&h, (cfg.out_channels * l) as usize);
    // `apply_final_activation=True, use_tanh_at_final=False` -> clamp, not tanh.
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
        let m = VocoderConfig::ltx25().tensor_manifest();
        assert_eq!(m.len(), 667, "manifest has {} tensors", m.len());
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate tensor name in the manifest");
        assert!(names.contains("conv_pre.weight"));
        assert!(names.contains("ups.0.weight"));
        assert!(names.contains("resblocks.17.acts2.2.downsample.lowpass.filter"));
        assert!(names.contains("act_post.act.alpha"));
        assert!(!names.contains("conv_post.bias"), "use_bias_at_final=false: conv_post must have no bias");

        let get = |n: &str| m.iter().find(|(k, _)| k == n).unwrap().1.clone();
        assert_eq!(get("conv_pre.weight"), vec![1536, 128, 7]);
        assert_eq!(get("ups.0.weight"), vec![1536, 768, 11]);
        assert_eq!(get("ups.5.weight"), vec![48, 24, 4]);
        assert_eq!(get("resblocks.0.convs1.0.weight"), vec![768, 768, 3]);
        assert_eq!(get("resblocks.17.convs1.0.weight"), vec![24, 24, 11]);
        assert_eq!(get("conv_post.weight"), vec![2, 24, 7]);
    }

    /// The upsample rate product (160) is the real vocoder's mel hop length -
    /// `t` mel frames become `t*160` samples.
    #[test]
    fn upsample_product_is_the_mel_hop_length() {
        let cfg = VocoderConfig::ltx25();
        let prod: u32 = cfg.upsample_rates.iter().product();
        assert_eq!(prod, 160);
        assert_eq!(cfg.final_channels(), 24);
    }
}
