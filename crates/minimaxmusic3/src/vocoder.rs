// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The Flow-VAE vocoder: a DAC-style (Descript Audio Codec) decoder.
//! `dec_in_proj (1x1 conv) -> conv_in (k=7) -> 4 VocoderBlocks (Snake ->
//! ConvTranspose1d upsample -> 3 parallel-dilation VocoderResidualUnit) ->
//! snake_out -> conv_out (k=7) -> tanh`. Folds `latent_channels` into 2
//! (stereo) by treating each channel as its own `latent_channels/2`-wide
//! stream (`batch*2` in the NCL layout).
//!
//! Every conv/conv-transpose reuses `audio::conv`'s device kernels (forward
//! AND backward already exist there); the Snake activation is
//! `audio::snake` (this port's own single-parameter form - the checkpoint's
//! `Snake1d`, not `kernels::SNAKE_BETA`'s two-parameter log-space BigVGAN
//! form). Device forward (not host math like `condition_encoder`): this
//! component upsamples up to 512-fold per call and is genuinely compute-heavy,
//! so it belongs on the same tape-based device engine every other serving
//! path uses.

use audio::conv::{conv1d_bias_fwd, convtr1d_bias_fwd, fold_weight_norm, Conv1d, ConvGemmKernels, ConvKernels, ConvScratch};
use audio::snake::{snake1d_fwd, Snake1d, SnakeKernels};
use checkpoint::safetensors::{self, StTensor};
use gpu_core::{DeviceBuffer, Gpu, Step};
use std::collections::HashMap;
use std::path::Path;

use crate::config::VocoderConfig;

/// This crate's own kernel list for the vocoder's device forward. Index
/// order fixes the constants below - keep them in lockstep with any edit here.
///
/// Slots 5..10 are `audio::conv`'s GEMM lowering (`im2col1d_at`,
/// `matmul_reg3`, `matmul_dx_reg`, `matmul_dw_reg_splitk`, `nlc_bias_nchw`,
/// `col2im1d_bias`). They are APPENDED, so nothing that indexes this list by
/// the first five moves, and `conv1d`/`convtr1d` stay registered because the
/// selector still routes narrow convs (this decoder's `Cout = 1` output conv)
/// to them - and because they are the only path on a backend without
/// workgroup reductions.
pub const PIPELINES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("convtr1d", kernels::CONVTR1D),
    ("snake1d", kernels::SNAKE1D),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("add2", kernels::ADD2),
    ("im2col1d_at", kernels::IM2COL1D_AT),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("matmul_dx_reg", kernels::MATMUL_DX_REG),
    ("matmul_dw_reg_splitk", kernels::MATMUL_DW_REG_SPLITK),
    ("nlc_bias_nchw", kernels::NLC_BIAS_NCHW),
    ("col2im1d_bias", kernels::COL2IM1D_BIAS),
];
const CONV1D: usize = 0;
const CONVTR1D: usize = 1;
const SNAKE1D: usize = 2;
const BIAS_ADD: usize = 3;
const ADD2: usize = 4;
const IM2COL1D_AT: usize = 5;
const MATMUL_REG3: usize = 6;
const MATMUL_DX_REG: usize = 7;
const MATMUL_DW_REG_SPLITK: usize = 8;
const NLC_BIAS_NCHW: usize = 9;
const COL2IM1D_BIAS: usize = 10;

const SNAKE_EPS: f32 = 1e-9;

/// The recorded forward: the dispatch list plus the ONE conv scratch every
/// lowered convolution in it shares.
///
/// The scratch has to live at this scope rather than inside a conv builder:
/// the whole decoder is recorded and submitted as a single pass, so a
/// per-conv scratch would keep all 28 of them resident at once (~4 GB here)
/// instead of one buffer sized for the largest. Reuse is safe because
/// dispatches execute in submit order with the backend's inter-dispatch
/// barriers - see `audio::conv::ConvScratch`.
#[derive(Default)]
struct Tape {
    steps: Vec<Step>,
    scratch: ConvScratch,
}

fn conv_gemm_kernels(fwd: usize) -> ConvGemmKernels {
    ConvGemmKernels {
        direct: ConvKernels { fwd, dx: 0, dw: 0 },
        bias: BIAS_ADD,
        im2col: IM2COL1D_AT,
        matmul: MATMUL_REG3,
        matmul_nn: MATMUL_DX_REG,
        matmul_tn: MATMUL_DW_REG_SPLITK,
        nlc_bias: NLC_BIAS_NCHW,
        col2im: COL2IM1D_BIAS,
    }
}
fn snake_kernels() -> SnakeKernels {
    SnakeKernels { fwd: SNAKE1D, bwd_dx: 0, bwd_dalpha: 0 }
}

/// A conv (or conv-transpose)'s folded weight + bias, ready to upload.
#[derive(Clone)]
pub struct ConvW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Clone)]
pub struct ResidualUnitW {
    pub snake1_alpha: Vec<f32>,
    pub conv1: ConvW,
    pub snake2_alpha: Vec<f32>,
    pub conv2: ConvW,
}

#[derive(Clone)]
pub struct VocoderBlockW {
    pub snake1_alpha: Vec<f32>,
    pub conv_t1: ConvW,
    pub res_units: Vec<ResidualUnitW>,
}

#[derive(Clone)]
pub struct VocoderWeights {
    pub dec_in_proj: ConvW,
    pub conv_in: ConvW,
    pub blocks: Vec<VocoderBlockW>,
    pub snake_out_alpha: Vec<f32>,
    pub conv_out: ConvW,
}

impl VocoderWeights {
    /// Mutable access to one conv's weight tensor by its `train::flatten`
    /// name (e.g. `"dec_in_proj.weight"`, `"blocks.1.res_unit2.conv1.weight"`),
    /// LoRA's seam for substituting `base + delta` without a full
    /// unflatten/rebuild. `None` for a name that doesn't parse or doesn't
    /// name a `.weight` leaf (LoRA never adapts a bias or a Snake alpha).
    pub fn conv_weight_mut(&mut self, name: &str) -> Option<&mut Vec<f32>> {
        let name = name.strip_suffix(".weight")?;
        match name {
            "dec_in_proj" => return Some(&mut self.dec_in_proj.weight),
            "conv_in" => return Some(&mut self.conv_in.weight),
            "conv_out" => return Some(&mut self.conv_out.weight),
            _ => {}
        }
        let rest = name.strip_prefix("blocks.")?;
        let (i, rest) = rest.split_once('.')?;
        let block = self.blocks.get_mut(i.parse::<usize>().ok()?)?;
        if rest == "conv_t1" {
            return Some(&mut block.conv_t1.weight);
        }
        let rest = rest.strip_prefix("res_unit")?;
        let (j, conv) = rest.split_once('.')?;
        let ru = block.res_units.get_mut(j.parse::<usize>().ok()?.checked_sub(1)?)?;
        match conv {
            "conv1" => Some(&mut ru.conv1.weight),
            "conv2" => Some(&mut ru.conv2.weight),
            _ => None,
        }
    }
}

struct TensorMap(HashMap<String, StTensor>);

impl TensorMap {
    fn get(&self, name: &str) -> Result<&[f32], String> {
        self.0.get(name).map(|t| t.data.as_slice()).ok_or_else(|| format!("vocoder: missing tensor {name:?}"))
    }
    /// A conv weight, folding `{prefix}.weight_g`/`{prefix}.weight_v` if
    /// present (every conv here except `dec_in_proj`), else reading a plain
    /// `{prefix}.weight`.
    fn conv_weight(&self, prefix: &str) -> Result<Vec<f32>, String> {
        let gname = format!("{prefix}.weight_g");
        let vname = format!("{prefix}.weight_v");
        if let (Some(g), Some(v)) = (self.0.get(&gname), self.0.get(&vname)) {
            Ok(fold_weight_norm(&g.data, &v.data, g.data.len()))
        } else {
            self.get(&format!("{prefix}.weight")).map(<[f32]>::to_vec)
        }
    }
    fn conv(&self, prefix: &str) -> Result<ConvW, String> {
        Ok(ConvW { weight: self.conv_weight(prefix)?, bias: self.get(&format!("{prefix}.bias"))?.to_vec() })
    }
    fn alpha(&self, prefix: &str) -> Result<Vec<f32>, String> {
        self.get(&format!("{prefix}.alpha")).map(<[f32]>::to_vec)
    }
}

/// Read every vocoder tensor from a `vocoder/` checkpoint directory (a
/// single `diffusion_pytorch_model.safetensors`), folding every
/// `weight_g`/`weight_v` pair at import time - a one-time host op, not a
/// hot-path kernel.
pub fn import(dir: &str, cfg: &VocoderConfig) -> Result<VocoderWeights, String> {
    let tensors = safetensors::read_model_dir(Path::new(dir))?;
    from_tensors(tensors, cfg)
}

/// [`import`], from tensors already read (e.g. a golden fixture's own
/// `state_dict.safetensors`).
pub fn from_tensors(tensors: Vec<StTensor>, cfg: &VocoderConfig) -> Result<VocoderWeights, String> {
    let map = TensorMap(tensors.into_iter().map(|t| (t.name.clone(), t)).collect());
    let num_stages = cfg.upsampling_ratios.len();
    let mut blocks = Vec::with_capacity(num_stages);
    for i in 0..num_stages {
        let p = format!("blocks.{i}");
        let mut res_units = Vec::with_capacity(3);
        for j in 0..3 {
            let ru = format!("{p}.res_unit{}", j + 1);
            res_units.push(ResidualUnitW {
                snake1_alpha: map.alpha(&format!("{ru}.snake1"))?,
                conv1: map.conv(&format!("{ru}.conv1"))?,
                snake2_alpha: map.alpha(&format!("{ru}.snake2"))?,
                conv2: map.conv(&format!("{ru}.conv2"))?,
            });
        }
        blocks.push(VocoderBlockW { snake1_alpha: map.alpha(&format!("{p}.snake1"))?, conv_t1: map.conv(&format!("{p}.conv_t1"))?, res_units });
    }
    Ok(VocoderWeights {
        dec_in_proj: map.conv("dec_in_proj")?,
        conv_in: map.conv("conv_in")?,
        blocks,
        snake_out_alpha: map.alpha("snake_out")?,
        conv_out: map.conv("conv_out")?,
    })
}

/// The device forward. `latents` is `[batch, latent_channels, length]`
/// row-major; returns `[batch, 2, length * product(upsampling_ratios)]`
/// row-major stereo samples in `[-1, 1]`.
pub fn forward(gpu: &Gpu, cfg: &VocoderConfig, w: &VocoderWeights, latents: &[f32], batch: usize, length: usize) -> Vec<f32> {
    let half = cfg.latent_channels as usize / 2;
    assert_eq!(latents.len(), batch * cfg.latent_channels as usize * length, "vocoder::forward: latents length mismatch");

    // "hidden_states.reshape(batch_size * 2, latent_channels // 2, length)":
    // fold the stereo split into the batch axis. The row-major NCL layout
    // makes this a pure reinterpretation, no data movement.
    let rows = batch * 2;
    let mut t = Tape::default();
    let x = gpu.storage_init("vocoder.in", latents);

    let cur = conv1d_bias_step(gpu, &mut t, &w.dec_in_proj, rows as u32, half as u32, length as u32, cfg.decoder_input_dim, 1, 0, 1, &x);
    let mut cur_dim = cfg.decoder_input_dim as usize;
    let mut cur_len = length;
    let mut cur = conv1d_bias_step(gpu, &mut t, &w.conv_in, rows as u32, cur_dim as u32, cur_len as u32, cfg.decoder_hidden_dim, 7, 3, 1, &cur);
    cur_dim = cfg.decoder_hidden_dim as usize;

    for (i, block) in w.blocks.iter().enumerate() {
        let stride = cfg.upsampling_ratios[i] as usize;
        let out_dim = cur_dim / 2;
        let a1 = snake_step(gpu, &mut t, &block.snake1_alpha, rows as u32, cur_dim as u32, cur_len as u32, &cur);
        let out_len = cur_len * stride; // exact for the even strides this checkpoint uses (see module test).
        let pad = stride.div_ceil(2);
        cur = convtr1d_bias_step(gpu, &mut t, &block.conv_t1, rows as u32, cur_dim as u32, cur_len as u32, out_dim as u32, 2 * stride as u32, stride as u32, pad as u32, out_len as u32, &a1);
        cur_dim = out_dim;
        cur_len = out_len;
        for (dilation, ru) in [1u32, 3, 9].into_iter().zip(&block.res_units) {
            cur = residual_unit_step(gpu, &mut t, ru, rows as u32, cur_dim as u32, cur_len as u32, dilation, &cur);
        }
    }

    let sn = snake_step(gpu, &mut t, &w.snake_out_alpha, rows as u32, cur_dim as u32, cur_len as u32, &cur);
    let out = conv1d_bias_step(gpu, &mut t, &w.conv_out, rows as u32, cur_dim as u32, cur_len as u32, 1, 7, 3, 1, &sn);
    // tanh has its own elementwise kernel elsewhere in the workspace, but at
    // one 1-channel row this is a single host pass over the final samples -
    // cheaper than a device round trip for a buffer this small relative to
    // everything already computed.
    gpu.submit(&[], &t.steps);
    let raw = gpu.read(&out, rows * cur_len);
    let waveform: Vec<f32> = raw.iter().map(|&v| v.tanh()).collect();

    // "waveform.reshape(batch_size, 2, -1)": undo the stereo-fold, again a
    // pure reinterpretation since `rows = batch*2` was already contiguous.
    debug_assert_eq!(waveform.len(), batch * 2 * cur_len);
    waveform
}

#[allow(clippy::too_many_arguments)]
fn conv1d_bias_step(gpu: &Gpu, t: &mut Tape, w: &ConvW, n: u32, cin: u32, l: u32, cout: u32, k: u32, pad: u32, dilation: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let lo = Conv1d::out_len(l, k, 1, pad, pad, dilation);
    let c = Conv1d { n, cin, l, cout, k, stride: 1, pad, dilation, groups: 1, lo };
    let wb = gpu.storage_init("w", &w.weight);
    let bb = gpu.storage_init("b", &w.bias);
    let y = gpu.storage(u64::from(n) * u64::from(cout) * u64::from(lo));
    t.steps.extend(conv1d_bias_fwd(gpu, &conv_gemm_kernels(CONV1D), &c, x, &wb, &bb, &y, &mut t.scratch));
    y
}

#[allow(clippy::too_many_arguments)]
fn convtr1d_bias_step(gpu: &Gpu, t: &mut Tape, w: &ConvW, n: u32, cin: u32, l: u32, cout: u32, k: u32, stride: u32, pad: u32, lo: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let c = Conv1d { n, cin, l, cout, k, stride, pad, dilation: 1, groups: 1, lo };
    let wb = gpu.storage_init("w", &w.weight);
    let bb = gpu.storage_init("b", &w.bias);
    let y = gpu.storage(u64::from(n) * u64::from(cout) * u64::from(lo));
    t.steps.extend(convtr1d_bias_fwd(gpu, &conv_gemm_kernels(CONVTR1D), &c, x, &wb, &bb, &y, &mut t.scratch));
    y
}

fn snake_step(gpu: &Gpu, t: &mut Tape, alpha: &[f32], rows: u32, c: u32, inner: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let sc = Snake1d { rows, c, inner, eps: SNAKE_EPS };
    let ab = gpu.storage_init("alpha", alpha);
    let y = gpu.storage(u64::from(sc.total()));
    t.steps.push(snake1d_fwd(gpu, &snake_kernels(), &sc, x, &ab, &y));
    y
}

fn residual_unit_step(gpu: &Gpu, t: &mut Tape, ru: &ResidualUnitW, n: u32, dim: u32, l: u32, dilation: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let pad = 3 * dilation; // (k-1)*dilation/2 at k=7.
    let s1 = snake_step(gpu, t, &ru.snake1_alpha, n, dim, l, x);
    let c1 = conv1d_bias_step(gpu, t, &ru.conv1, n, dim, l, dim, 7, pad, dilation, &s1);
    let s2 = snake_step(gpu, t, &ru.snake2_alpha, n, dim, l, &c1);
    let c2 = conv1d_bias_step(gpu, t, &ru.conv2, n, dim, l, dim, 1, 0, 1, &s2);
    let out = gpu.storage(u64::from(n) * u64::from(dim) * u64::from(l));
    t.steps.push(gpu.step(ADD2, &[x, &c2, &out], &[n * dim * l], n * dim * l));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // `fold_weight_norm`'s own correctness gate now lives with the hoisted
    // implementation: `crates/audio/tests/weight_norm.rs`.

    #[test]
    fn even_stride_transposed_conv_output_length_is_exactly_l_times_stride() {
        // Confirms the `out_len = cur_len * stride` shortcut used in `forward`
        // for this checkpoint's always-even strides (8,8,4,2): out_len_transposed
        // with k=2*stride, pad=stride/2, out_pad=0 collapses to l*stride.
        for &stride in &[8u32, 4, 2] {
            let l = 6u32;
            let k = 2 * stride;
            let pad = stride.div_ceil(2);
            let want = Conv1d::out_len_transposed(l, k, stride, pad, 0, 1);
            assert_eq!(want, l * stride, "stride={stride}");
        }
    }
}
