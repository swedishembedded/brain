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
//! ## Where the convolutions are dispatched, and why it is not one answer
//!
//! Every `groups == 1` convolution here goes through the SELECTED lowering
//! (`audio::conv::conv1d_bias_fwd` / `convtr1d_bias_fwd`), which asks
//! `gpu_core::select` per shape whether the im2col+GEMM form or the direct
//! kernel wins and folds the per-channel bias into whichever it picks. Every
//! width in this vocoder's `groups == 1` sites clears that selector's
//! measured crossovers except `conv_post`, whose two output channels put it
//! below the `conv1d` threshold - and the selector, not this file, is what
//! decides that.
//!
//! The antialiasing convolutions stay on the direct kernels, and NOT because
//! of a threshold: they are depthwise (`groups == channels`), and a grouped
//! convolution is a block-diagonal GEMM - a different kernel, not a different
//! shape. `Conv1d::lowerable` refuses them structurally. They also carry no
//! bias at all, so the bias-folding entry point would only add a zero-add
//! dispatch to each of them.
//!
//! ## The boundary handling is on the device, and that is the whole cost
//!
//! `Activation1d` runs twice per resblock branch and once per `act_post`, so
//! it is by far the most-executed thing in this file. Both of its resamplers
//! need REPLICATE-padded edges, which is not what `conv1d`'s implicit zero
//! padding gives: the pad is done with [`kernels::PAD1D_EDGE`] as an ordinary
//! recorded step, and the upsampler's post-crop is expressed as
//! `convtr1d`'s own `pad` parameter (the kernel indexes its output as
//! `lo + pad`, so a symmetric crop IS that parameter) rather than as a host
//! slice. The `ratio` scale folds into the filter, exactly - multiplying
//! every tap by a power of two rounds nothing, so each product and therefore
//! each partial sum is exactly doubled.
//!
//! That matters because the alternative is a blocking `Gpu::read` per stage:
//! host padding and host cropping mean the pipeline drains to the host and
//! back three times per activation call, which on a card with no resizable
//! BAR is the dominant cost of the whole audio stage. Nothing in
//! [`synthesize`] reads back until the final waveform.
//!
//! The two Kaiser-sinc filters are also **cached per channel width** rather
//! than re-broadcast and re-uploaded per call site ([`FilterCache`]): the
//! checkpoint stores one shared `[1,1,12]` buffer per site, every site's copy
//! holds the same taps, and the depthwise weight this file needs from it is a
//! pure function of `(taps, channels)`. The cache is keyed on both, so two
//! sites that genuinely differed would still get their own weight.
//!
//! One new kernel ([`kernels::PAD1D_EDGE`]); everything else is
//! `conv1d`/`convtr1d` (`crates/audio/src/conv.rs`, direct and lowered),
//! `snake_beta` (already wired up in `crates/mimi/src/model.rs`), `axpy`,
//! `add2`. Eager dispatch, same style as [`crate::audio_vae`] and
//! `crates/mimi/src/model.rs`.

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

/// The direct kernels, used for the depthwise antialias convolutions and as
/// the selector's own structural fallback inside [`gemm_kernels`].
fn kernels_id() -> ConvKernels {
    ConvKernels { fwd: K_CONV1D, dx: 0, dw: 0 }
}
fn kernels_id_tr() -> ConvKernels {
    ConvKernels { fwd: K_CONVTR1D, dx: 0, dw: 0 }
}

/// The selected-lowering pipeline set for the `groups == 1` convolutions.
/// `direct` names whichever of the two forward kernels this call site is,
/// since `audio::conv` reaches for it whenever the lowering does not apply.
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
    &t.get(name).unwrap_or_else(|| panic!("ltxv vocoder: missing tensor {name}")).1
}

/// One synthesis run's device context: the handle, the weights, the shared
/// GEMM scratch and the antialias-filter cache.
///
/// The scratch is the one `audio::conv::ConvScratch` every lowered conv in
/// this pass shares - safe because a recorded pass runs its steps in submit
/// order, which is exactly the contract that type documents.
struct Ctx<'a> {
    gpu: Gpu,
    t: &'a Tensors,
    scratch: ConvScratch,
    /// `(channels, the filter's 12 taps as bits) -> the depthwise weight`.
    /// Keyed on the taps themselves, not the tensor name, because the
    /// checkpoint stores one physically separate `[1,1,12]` buffer per site
    /// holding the same values - so two sites at the same width share a
    /// buffer, and a site that genuinely differed would still get its own.
    filters: HashMap<(u32, Vec<u32>), DeviceBuffer>,
    /// A zeroed `[cout]` bias for the one convolution the checkpoint gives no
    /// bias tensor at all (`conv_post`, `use_bias_at_final: false`), so it can
    /// take the same bias-folding entry point as every other conv here.
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
    /// weight (`nn.Module.expand` is a broadcast VIEW of the same 12 values,
    /// not per-channel data), with `scale` folded into every tap.
    ///
    /// Folding `ratio` into the weight rather than scaling the output is
    /// EXACT for the power-of-two ratio this vocoder uses: doubling an f32
    /// rounds nothing, so every product and hence every partial sum of the
    /// convolution is exactly doubled.
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

    /// Replicate-pad an NCL `[c,l]` DEVICE buffer by `(left,right)` samples per
    /// channel. Returns the padded buffer and its length.
    fn pad_edge(&self, x: &DeviceBuffer, c: u32, l: u32, left: u32, right: u32) -> (DeviceBuffer, u32) {
        let lp = l + left + right;
        let total = c * lp;
        let y = self.gpu.storage(total as u64);
        self.gpu.submit(&[], &[self.gpu.step(K_PAD1D_EDGE, &[x, &y], &[total, l, left, right], total)]);
        (y, lp)
    }
}

/// Symmetric "same" `conv1d` (stride 1, `Lo == L`): `pad = dilation*(k-1)/2`.
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

/// `conv_pre`/`conv_post`: `128 -> upsample_initial_channel` / `final_channels
/// -> 2`, kernel 7, native `padding=3` (also a symmetric "same" conv at
/// stride 1). `has_bias=false` is `conv_post`'s `use_bias_at_final: false` -
/// it takes the same entry point against a zeroed bias rather than a second
/// code path, so the selector's decision is shared.
fn conv1d_k7(cx: &mut Ctx, prefix: &str, cin: u32, cout: u32, l: u32, has_bias: bool, x: &DeviceBuffer) -> DeviceBuffer {
    let cfg = audio::conv::Conv1d { n: 1, cin, l, cout, k: 7, stride: 1, pad: 3, dilation: 1, groups: 1, lo: l };
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
/// `pad=kernel_size/ratio-1=5` each side, depthwise `ConvTranspose1d(k=12,
/// stride=2)` against the checkpoint filter scaled by `ratio=2`, cropping
/// `pad_left=pad_right=15` off each side. Output length `2*l`.
///
/// The crop IS the transposed convolution's own `pad` parameter:
/// `convtr1d.wgsl` indexes its input as `lo + pad`, so a `pad` of 15 skips
/// the first 15 native outputs, and the shortened `Lo` drops the last 15.
/// `UpSample1d.__init__` computes `pad_left = pad*stride + (k-stride)//2` and
/// `pad_right = pad*stride + (k-stride+1)//2`, equal at `(pad=5,stride=2,
/// k=12)` because `k-stride` is even - which is what lets one symmetric
/// parameter express both sides.
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

/// `DownSample1d(ratio=2)` == `LowPassFilter1d(cutoff=0.25,kernel_size=12)`:
/// replicate-pad `(left=5,right=6)`, depthwise `Conv1d(k=12,stride=2,pad=0)`
/// against the checkpoint's lowpass filter. Output length `l/2` (`l` is the
/// ALREADY-upsampled length, `2x` the block's nominal width).
fn antialias_downsample(cx: &mut Ctx, filter_name: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let (pl, pr) = (5u32, 6u32); // LowPassFilter1d: pad_left = k/2 - 1, pad_right = k/2 (k=12, even)
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
/// antialiased downsample. Length-preserving (`l` in, `l` out), and nothing
/// in it touches the host.
fn activation1d(cx: &mut Ctx, prefix: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let up = antialias_upsample(cx, &format!("{prefix}.upsample.filter"), c, l, x);
    let sn = snake_beta(cx, &format!("{prefix}.act"), c, 2 * l, &up);
    antialias_downsample(cx, &format!("{prefix}.downsample.lowpass.filter"), c, 2 * l, &sn)
}

/// One `AMPBlock1`: 3 (act1 -> conv1 -> act2 -> conv2 -> residual-add) steps
/// at dilations `(1,3,5)` (`convs2` always dilation 1), kernel `k`.
fn amp_block(cx: &mut Ctx, prefix: &str, c: u32, l: u32, k: u32, dilations: [u32; 3], x: &DeviceBuffer) -> DeviceBuffer {
    let mut cur = x.clone();
    for (d, &dilation) in dilations.iter().enumerate() {
        let a1 = activation1d(cx, &format!("{prefix}.acts1.{d}"), c, l, &cur);
        let c1 = conv1d_same(cx, &format!("{prefix}.convs1.{d}"), c, k, dilation, l, &a1);
        let a2 = activation1d(cx, &format!("{prefix}.acts2.{d}"), c, l, &c1);
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

    let mut cx = Ctx::new(device, tensors);
    let x_in = cx.gpu.storage_init("vocoder.mel_in", &nc);
    let mut h = conv1d_k7(&mut cx, "conv_pre", cfg.mel_channels, cfg.upsample_initial_channel, t, true, &x_in);
    let mut l = t;

    for i in 0..cfg.num_upsamples() {
        let (cin, cout) = (cfg.stage_cin(i), cfg.stage_cout(i));
        let (up, lo) = convtr1d(&mut cx, &format!("ups.{i}"), cin, cout, cfg.upsample_kernel_sizes[i], cfg.upsample_rates[i], l, &h);
        l = lo;
        let acc = cx.gpu.storage((cout * l) as u64);
        cx.gpu.write_f32(&acc, &vec![0.0f32; (cout * l) as usize]);
        for r in 0..cfg.resblock_kernel_sizes.len() {
            let idx = i * cfg.resblock_kernel_sizes.len() + r;
            let y = amp_block(&mut cx, &format!("resblocks.{idx}"), cout, l, cfg.resblock_kernel_sizes[r], cfg.resblock_dilations[r], &up);
            axpy_into(&cx.gpu, &acc, 1.0 / 3.0, &y, cout * l);
        }
        h = acc;
    }

    let fin = cfg.final_channels();
    h = activation1d(&mut cx, "act_post", fin, l, &h);
    h = conv1d_k7(&mut cx, "conv_post", fin, cfg.out_channels, l, false, &h);

    let mut wave = cx.gpu.read(&h, (cfg.out_channels * l) as usize);
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
