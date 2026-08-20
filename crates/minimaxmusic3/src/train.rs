// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Vocoder training: device forward + backward + an MSE reconstruction loss,
//! wired to `crates/gradcheck`'s `CheckModel` contract (`param_names`,
//! `read_weight`/`write_weight`, `read_grad`, `loss`, `zero_grads`,
//! `backward`).
//!
//! Separate from `vocoder::forward` (the served path) because training needs
//! persistent DEVICE-resident weight/gradient buffers reused across many
//! steps, and every intermediate activation kept around for backward -
//! neither of which the serving forward should pay for on its one-shot,
//! re-upload-per-call path. The `forward_matches_serving_forward` test
//! guards the two implementations against drifting apart, the same
//! discipline the host-f64 training paths elsewhere in this workspace use
//! against their own served device forward.
//!
//! This lands backward/gradcheck for the vocoder's own conv/snake stack.
//! The multi-scale STFT/mel discriminator + adversarial + feature-matching
//! loss this model's training scope also requires is a further, larger
//! step, tracked separately in the roadmap - an MSE reconstruction loss is
//! enough to prove every gradient here is analytically correct, which is
//! what gradcheck exists to do; it is not the loss a real training run
//! would use.

use audio::conv::{conv1d_bwd, conv1d_fwd, convtr1d_bwd, convtr1d_fwd, Conv1d, ConvKernels};
use audio::snake::{snake1d_bwd_dalpha, snake1d_bwd_dx, snake1d_fwd, Snake1d, SnakeKernels};
use data::rng::Lcg;
use gpu_core::{DeviceBuffer, Gpu, Step};
use std::cell::RefCell;
use std::collections::HashMap;

use crate::config::VocoderConfig;
use crate::vocoder::{ConvW, ResidualUnitW, VocoderBlockW, VocoderWeights};

pub const PIPELINES: &[(&str, &str)] = &[
    ("conv1d", kernels::CONV1D),
    ("conv1d_dx", kernels::CONV1D_DX),
    ("conv1d_dw", kernels::CONV1D_DW),
    ("convtr1d", kernels::CONVTR1D),
    ("convtr1d_dx", kernels::CONVTR1D_DX),
    ("convtr1d_dw", kernels::CONVTR1D_DW),
    ("snake1d", kernels::SNAKE1D),
    ("snake1d_bwd_dx", kernels::SNAKE1D_BWD_DX),
    ("snake1d_bwd_dalpha", kernels::SNAKE1D_BWD_DALPHA),
    ("add_chan_inplace", kernels::ADD_CHAN_INPLACE),
    ("bias_grad_ncl", kernels::BIAS_GRAD_NCL),
    ("add2", kernels::ADD2),
    ("tanh_act", kernels::TANH_ACT),
    ("tanh_act_bwd", kernels::TANH_ACT_BWD),
];
const CONV1D: usize = 0;
const CONV1D_DX: usize = 1;
const CONV1D_DW: usize = 2;
const CONVTR1D: usize = 3;
const CONVTR1D_DX: usize = 4;
const CONVTR1D_DW: usize = 5;
const SNAKE1D: usize = 6;
const SNAKE1D_BWD_DX: usize = 7;
const SNAKE1D_BWD_DALPHA: usize = 8;
const BIAS_ADD: usize = 9;
const BIAS_GRAD_NCL: usize = 10;
const ADD2: usize = 11;
const TANH_ACT: usize = 12;
const TANH_ACT_BWD: usize = 13;

const SNAKE_EPS: f32 = 1e-9;

fn conv_kernels() -> ConvKernels {
    ConvKernels { fwd: CONV1D, dx: CONV1D_DX, dw: CONV1D_DW }
}
fn convtr_kernels() -> ConvKernels {
    ConvKernels { fwd: CONVTR1D, dx: CONVTR1D_DX, dw: CONVTR1D_DW }
}
fn snake_kernels() -> SnakeKernels {
    SnakeKernels { fwd: SNAKE1D, bwd_dx: SNAKE1D_BWD_DX, bwd_dalpha: SNAKE1D_BWD_DALPHA }
}

/// `{ prefix }.weight`/`.bias` (or `.alpha`) names, in the exact order
/// [`flatten`]/[`Trainer::new`] walk [`VocoderWeights`] - one enumeration
/// shared by every named-parameter access so the tree can't drift out of
/// sync with itself.
fn conv_names(prefix: &str) -> [String; 2] {
    [format!("{prefix}.weight"), format!("{prefix}.bias")]
}

/// Depth-first `(name, values)` pairs over every leaf tensor in a
/// [`VocoderWeights`], in a fixed, deterministic order.
pub fn flatten(w: &VocoderWeights) -> Vec<(String, Vec<f32>)> {
    let mut out = Vec::new();
    let conv = |prefix: &str, c: &ConvW, out: &mut Vec<(String, Vec<f32>)>| {
        let [wn, bn] = conv_names(prefix);
        out.push((wn, c.weight.clone()));
        out.push((bn, c.bias.clone()));
    };
    conv("dec_in_proj", &w.dec_in_proj, &mut out);
    conv("conv_in", &w.conv_in, &mut out);
    for (i, block) in w.blocks.iter().enumerate() {
        out.push((format!("blocks.{i}.snake1.alpha"), block.snake1_alpha.clone()));
        conv(&format!("blocks.{i}.conv_t1"), &block.conv_t1, &mut out);
        for (j, ru) in block.res_units.iter().enumerate() {
            let p = format!("blocks.{i}.res_unit{}", j + 1);
            out.push((format!("{p}.snake1.alpha"), ru.snake1_alpha.clone()));
            conv(&format!("{p}.conv1"), &ru.conv1, &mut out);
            out.push((format!("{p}.snake2.alpha"), ru.snake2_alpha.clone()));
            conv(&format!("{p}.conv2"), &ru.conv2, &mut out);
        }
    }
    out.push(("snake_out.alpha".to_string(), w.snake_out_alpha.clone()));
    conv("conv_out", &w.conv_out, &mut out);
    out
}

/// Device-resident mirror of one [`ConvW`]: persistent weight/bias buffers
/// plus their gradient buffers, reused across every training step.
struct ConvD {
    w: DeviceBuffer,
    b: DeviceBuffer,
    dw: DeviceBuffer,
    db: DeviceBuffer,
    w_name: String,
    b_name: String,
}

impl ConvD {
    fn upload(gpu: &Gpu, prefix: &str, c: &ConvW) -> ConvD {
        let [wn, bn] = conv_names(prefix);
        ConvD {
            w: gpu.storage_init("w", &c.weight),
            b: gpu.storage_init("b", &c.bias),
            dw: gpu.storage(c.weight.len() as u64 * 4),
            db: gpu.storage(c.bias.len() as u64 * 4),
            w_name: wn,
            b_name: bn,
        }
    }
}

struct ResidualUnitD {
    snake1_alpha: DeviceBuffer,
    d_snake1_alpha: DeviceBuffer,
    conv1: ConvD,
    snake2_alpha: DeviceBuffer,
    d_snake2_alpha: DeviceBuffer,
    conv2: ConvD,
    alpha1_name: String,
    alpha2_name: String,
}

struct VocoderBlockD {
    snake1_alpha: DeviceBuffer,
    d_snake1_alpha: DeviceBuffer,
    alpha_name: String,
    conv_t1: ConvD,
    res_units: Vec<ResidualUnitD>,
}

struct VocoderDeviceWeights {
    dec_in_proj: ConvD,
    conv_in: ConvD,
    blocks: Vec<VocoderBlockD>,
    snake_out_alpha: DeviceBuffer,
    d_snake_out_alpha: DeviceBuffer,
    conv_out: ConvD,
}

impl VocoderDeviceWeights {
    fn upload(gpu: &Gpu, w: &VocoderWeights) -> VocoderDeviceWeights {
        let alpha = |v: &[f32]| (gpu.storage_init("a", v), gpu.storage(v.len() as u64 * 4));
        let blocks = w
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b): (usize, &VocoderBlockW)| {
                let (s1, ds1) = alpha(&b.snake1_alpha);
                let res_units = b
                    .res_units
                    .iter()
                    .enumerate()
                    .map(|(j, ru): (usize, &ResidualUnitW)| {
                        let p = format!("blocks.{i}.res_unit{}", j + 1);
                        let (a1, da1) = alpha(&ru.snake1_alpha);
                        let (a2, da2) = alpha(&ru.snake2_alpha);
                        ResidualUnitD {
                            snake1_alpha: a1,
                            d_snake1_alpha: da1,
                            conv1: ConvD::upload(gpu, &format!("{p}.conv1"), &ru.conv1),
                            snake2_alpha: a2,
                            d_snake2_alpha: da2,
                            conv2: ConvD::upload(gpu, &format!("{p}.conv2"), &ru.conv2),
                            alpha1_name: format!("{p}.snake1.alpha"),
                            alpha2_name: format!("{p}.snake2.alpha"),
                        }
                    })
                    .collect();
                VocoderBlockD {
                    snake1_alpha: s1,
                    d_snake1_alpha: ds1,
                    alpha_name: format!("blocks.{i}.snake1.alpha"),
                    conv_t1: ConvD::upload(gpu, &format!("blocks.{i}.conv_t1"), &b.conv_t1),
                    res_units,
                }
            })
            .collect();
        let (so, dso) = alpha(&w.snake_out_alpha);
        VocoderDeviceWeights {
            dec_in_proj: ConvD::upload(gpu, "dec_in_proj", &w.dec_in_proj),
            conv_in: ConvD::upload(gpu, "conv_in", &w.conv_in),
            blocks,
            snake_out_alpha: so,
            d_snake_out_alpha: dso,
            conv_out: ConvD::upload(gpu, "conv_out", &w.conv_out),
        }
    }

    /// Every `(weight_name, buffer)` pair, for [`Trainer::read_weight`]/
    /// [`Trainer::write_weight`].
    fn weight_bufs(&self) -> Vec<(&str, &DeviceBuffer)> {
        let mut out = vec![(self.dec_in_proj.w_name.as_str(), &self.dec_in_proj.w), (self.dec_in_proj.b_name.as_str(), &self.dec_in_proj.b)];
        out.push((self.conv_in.w_name.as_str(), &self.conv_in.w));
        out.push((self.conv_in.b_name.as_str(), &self.conv_in.b));
        for b in &self.blocks {
            out.push((b.alpha_name.as_str(), &b.snake1_alpha));
            out.push((b.conv_t1.w_name.as_str(), &b.conv_t1.w));
            out.push((b.conv_t1.b_name.as_str(), &b.conv_t1.b));
            for ru in &b.res_units {
                out.push((ru.alpha1_name.as_str(), &ru.snake1_alpha));
                out.push((ru.conv1.w_name.as_str(), &ru.conv1.w));
                out.push((ru.conv1.b_name.as_str(), &ru.conv1.b));
                out.push((ru.alpha2_name.as_str(), &ru.snake2_alpha));
                out.push((ru.conv2.w_name.as_str(), &ru.conv2.w));
                out.push((ru.conv2.b_name.as_str(), &ru.conv2.b));
            }
        }
        out.push(("snake_out.alpha", &self.snake_out_alpha));
        out.push((self.conv_out.w_name.as_str(), &self.conv_out.w));
        out.push((self.conv_out.b_name.as_str(), &self.conv_out.b));
        out
    }

    /// Every `(grad_name, buffer)` pair - same names as [`Self::weight_bufs`].
    fn grad_bufs(&self) -> Vec<(&str, &DeviceBuffer)> {
        let mut out = vec![(self.dec_in_proj.w_name.as_str(), &self.dec_in_proj.dw), (self.dec_in_proj.b_name.as_str(), &self.dec_in_proj.db)];
        out.push((self.conv_in.w_name.as_str(), &self.conv_in.dw));
        out.push((self.conv_in.b_name.as_str(), &self.conv_in.db));
        for b in &self.blocks {
            out.push((b.alpha_name.as_str(), &b.d_snake1_alpha));
            out.push((b.conv_t1.w_name.as_str(), &b.conv_t1.dw));
            out.push((b.conv_t1.b_name.as_str(), &b.conv_t1.db));
            for ru in &b.res_units {
                out.push((ru.alpha1_name.as_str(), &ru.d_snake1_alpha));
                out.push((ru.conv1.w_name.as_str(), &ru.conv1.dw));
                out.push((ru.conv1.b_name.as_str(), &ru.conv1.db));
                out.push((ru.alpha2_name.as_str(), &ru.d_snake2_alpha));
                out.push((ru.conv2.w_name.as_str(), &ru.conv2.dw));
                out.push((ru.conv2.b_name.as_str(), &ru.conv2.db));
            }
        }
        out.push(("snake_out.alpha", &self.d_snake_out_alpha));
        out.push((self.conv_out.w_name.as_str(), &self.conv_out.dw));
        out.push((self.conv_out.b_name.as_str(), &self.conv_out.db));
        out
    }
}

/// Cached forward activations a residual unit's backward needs: the unit's
/// own input, and the input to each of its two convs (their outputs are
/// each other's next input, so only 3 buffers pin the whole chain).
struct ResidualUnitCache {
    x: DeviceBuffer,
    s1: DeviceBuffer,
    c1: DeviceBuffer,
    s2: DeviceBuffer,
    dim: u32,
    l: u32,
    dilation: u32,
}

struct BlockCache {
    x: DeviceBuffer,
    s1: DeviceBuffer,
    in_dim: u32,
    in_len: u32,
    out_dim: u32,
    out_len: u32,
    stride: u32,
    res: Vec<ResidualUnitCache>,
}

struct ForwardCache {
    latents: DeviceBuffer, // dec_in_proj input
    rows: u32,
    x1: DeviceBuffer, // conv_in input (= dec_in_proj output)
    blocks: Vec<BlockCache>,
    pre_snake_out: DeviceBuffer, // last block's output
    sn: DeviceBuffer,            // snake_out output = conv_out input
    pre_tanh: DeviceBuffer,      // conv_out output
    out_len: u32,
    output: DeviceBuffer,
}

/// Vocoder trainer: persistent device weights/gradients, a fixed
/// (latents, target) pair, and an MSE reconstruction loss - enough to
/// gradient-check every backward pass here, not a real training loss.
pub struct Trainer {
    gpu: Gpu,
    cfg: VocoderConfig,
    w: VocoderDeviceWeights,
    /// Element count per named parameter (`DeviceBuffer` is opaque and does
    /// not expose its own byte size), keyed the same as [`flatten`].
    sizes: HashMap<String, usize>,
    latents: Vec<f32>,
    batch: usize,
    length: usize,
    target: Vec<f32>,
    cache: RefCell<Option<ForwardCache>>,
    last_loss: RefCell<f32>,
}

impl Trainer {
    pub fn new(cfg: VocoderConfig, w: &VocoderWeights, latents: Vec<f32>, batch: usize, length: usize, target: Vec<f32>) -> Trainer {
        let gpu = Gpu::new_cpu(PIPELINES);
        let dw = VocoderDeviceWeights::upload(&gpu, w);
        let sizes = flatten(w).into_iter().map(|(name, data)| (name, data.len())).collect();
        let out_len = length * cfg.upsampling_ratios.iter().product::<u32>() as usize;
        assert_eq!(target.len(), batch * 2 * out_len, "Trainer::new: target length mismatch");
        Trainer { gpu, cfg, w: dw, sizes, latents, batch, length, target, cache: RefCell::new(None), last_loss: RefCell::new(0.0) }
    }

    fn run_forward(&self) -> f32 {
        let gpu = &self.gpu;
        let cfg = &self.cfg;
        let half = cfg.latent_channels as usize / 2;
        let rows = (self.batch * 2) as u32;
        let mut steps: Vec<Step> = Vec::new();

        let x0 = gpu.storage_init("latents", &self.latents);
        let x1 = conv1d_bias_fwd(gpu, &mut steps, &self.w.dec_in_proj, rows, half as u32, self.length as u32, cfg.decoder_input_dim, 1, 0, 1, &x0);
        let mut cur = conv1d_bias_fwd(gpu, &mut steps, &self.w.conv_in, rows, cfg.decoder_input_dim, self.length as u32, cfg.decoder_hidden_dim, 7, 3, 1, &x1);
        let mut cur_dim = cfg.decoder_hidden_dim;
        let mut cur_len = self.length as u32;

        let mut block_caches = Vec::new();
        for (i, block) in self.w.blocks.iter().enumerate() {
            let in_dim = cur_dim;
            let in_len = cur_len;
            let block_x = cur.clone();
            let stride = cfg.upsampling_ratios[i];
            let out_dim = cur_dim / 2;
            let s1 = snake_fwd(gpu, &mut steps, &block.snake1_alpha, rows, cur_dim, cur_len, &cur);
            let out_len = cur_len * stride;
            let pad = stride.div_ceil(2);
            cur = convtr1d_bias_fwd(gpu, &mut steps, &block.conv_t1, rows, cur_dim, cur_len, out_dim, 2 * stride, stride, pad, out_len, &s1);
            cur_dim = out_dim;
            cur_len = out_len;

            let mut res_caches = Vec::new();
            for (dilation, ru) in [1u32, 3, 9].into_iter().zip(&block.res_units) {
                let ru_x = cur.clone();
                let ru_pad = 3 * dilation;
                let rs1 = snake_fwd(gpu, &mut steps, &ru.snake1_alpha, rows, cur_dim, cur_len, &cur);
                let rc1 = conv1d_bias_fwd(gpu, &mut steps, &ru.conv1, rows, cur_dim, cur_len, cur_dim, 7, ru_pad, dilation, &rs1);
                let rs2 = snake_fwd(gpu, &mut steps, &ru.snake2_alpha, rows, cur_dim, cur_len, &rc1);
                let rc2 = conv1d_bias_fwd(gpu, &mut steps, &ru.conv2, rows, cur_dim, cur_len, cur_dim, 1, 0, 1, &rs2);
                let out = gpu.storage(u64::from(rows) * u64::from(cur_dim) * u64::from(cur_len) * 4);
                steps.push(gpu.step(ADD2, &[&ru_x, &rc2, &out], &[rows * cur_dim * cur_len], rows * cur_dim * cur_len));
                res_caches.push(ResidualUnitCache { x: ru_x, s1: rs1, c1: rc1, s2: rs2, dim: cur_dim, l: cur_len, dilation });
                cur = out;
            }
            block_caches.push(BlockCache { x: block_x, s1, in_dim, in_len, out_dim, out_len, stride, res: res_caches });
        }

        let pre_snake_out = cur.clone();
        let sn = snake_fwd(gpu, &mut steps, &self.w.snake_out_alpha, rows, cur_dim, cur_len, &cur);
        let pre_tanh = conv1d_bias_fwd(gpu, &mut steps, &self.w.conv_out, rows, cur_dim, cur_len, 1, 7, 3, 1, &sn);
        let output = gpu.storage(u64::from(rows) * u64::from(cur_len) * 4);
        steps.push(gpu.step(TANH_ACT, &[&pre_tanh, &output], &[rows * cur_len], rows * cur_len));

        gpu.submit(&[], &steps);

        let got = gpu.read(&output, (rows * cur_len) as usize);
        let n = got.len() as f32;
        let loss: f32 = got.iter().zip(&self.target).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / (2.0 * n);

        *self.cache.borrow_mut() =
            Some(ForwardCache { latents: x0, rows, x1, blocks: block_caches, pre_snake_out, sn, pre_tanh, out_len: cur_len, output });
        loss
    }

    fn run_backward(&self) {
        let cache_ref = self.cache.borrow();
        let cache = cache_ref.as_ref().expect("Trainer::backward called before a forward (loss()) ran");
        let gpu = &self.gpu;
        let cfg = &self.cfg;
        let rows = cache.rows;
        let n = (rows * cache.out_len) as usize;

        // d(MSE)/d(output) = (output - target) / n.
        let got = gpu.read(&cache.output, n);
        let want = &self.target;
        let d_loss: Vec<f32> = got.iter().zip(want).map(|(a, b)| (a - b) / n as f32).collect();
        let d_output = gpu.storage_init("d_output", &d_loss);

        let mut steps: Vec<Step> = Vec::new();
        let d_pre_tanh = gpu.storage(n as u64 * 4);
        steps.push(gpu.step(TANH_ACT_BWD, &[&d_output, &cache.pre_tanh, &d_pre_tanh], &[rows * cache.out_len], rows * cache.out_len));

        let mut cur_dim = self.cfg.decoder_hidden_dim / 2u32.pow(cfg.upsampling_ratios.len() as u32);
        let cur_len = cache.out_len;
        let mut d_cur = conv1d_bias_bwd(gpu, &mut steps, &self.w.conv_out, rows, cur_dim, cur_len, 1, 7, 3, 1, &cache.sn, &d_pre_tanh);
        d_cur = snake_bwd(gpu, &mut steps, &self.w.snake_out_alpha, rows, cur_dim, cur_len, &cache.pre_snake_out, &d_cur, &self.w.d_snake_out_alpha);

        for block in self.w.blocks.iter().zip(cache.blocks.iter()).collect::<Vec<_>>().into_iter().rev() {
            let (bw, bc) = block;
            for (ru_w, ru_c) in bw.res_units.iter().zip(bc.res.iter()).rev() {
                let dy_out = d_cur;
                // add2 backward: both branches get dy_out unchanged.
                let d_c2 = dy_out.clone();
                let d_c1_branch = conv1d_bias_bwd(gpu, &mut steps, &ru_w.conv2, rows, ru_c.dim, ru_c.l, ru_c.dim, 1, 0, 1, &ru_c.s2, &d_c2);
                let d_s1_out = snake_bwd(gpu, &mut steps, &ru_w.snake2_alpha, rows, ru_c.dim, ru_c.l, &ru_c.c1, &d_c1_branch, &ru_w.d_snake2_alpha);
                let ru_pad = 3 * ru_c.dilation;
                let d_x_branch = conv1d_bias_bwd(gpu, &mut steps, &ru_w.conv1, rows, ru_c.dim, ru_c.l, ru_c.dim, 7, ru_pad, ru_c.dilation, &ru_c.s1, &d_s1_out);
                let d_x_from_branch = snake_bwd(gpu, &mut steps, &ru_w.snake1_alpha, rows, ru_c.dim, ru_c.l, &ru_c.x, &d_x_branch, &ru_w.d_snake1_alpha);
                let dx = gpu.storage(u64::from(rows) * u64::from(ru_c.dim) * u64::from(ru_c.l) * 4);
                steps.push(gpu.step(ADD2, &[&dy_out, &d_x_from_branch, &dx], &[rows * ru_c.dim * ru_c.l], rows * ru_c.dim * ru_c.l));
                d_cur = dx;
            }
            let d_after_convt = d_cur;
            let d_s1 = convtr1d_bias_bwd(gpu, &mut steps, &bw.conv_t1, rows, bc.in_dim, bc.in_len, bc.out_dim, 2 * bc.stride, bc.stride, bc.stride.div_ceil(2), bc.out_len, &bc.s1, &d_after_convt);
            d_cur = snake_bwd(gpu, &mut steps, &bw.snake1_alpha, rows, bc.in_dim, bc.in_len, &bc.x, &d_s1, &bw.d_snake1_alpha);
            cur_dim = bc.in_dim;
        }

        let d_x1 = conv1d_bias_bwd(gpu, &mut steps, &self.w.conv_in, rows, self.cfg.decoder_input_dim, self.length as u32, cur_dim, 7, 3, 1, &cache.x1, &d_cur);
        let half = self.cfg.latent_channels / 2;
        let _d_x0 = conv1d_bias_bwd(gpu, &mut steps, &self.w.dec_in_proj, rows, half, self.length as u32, self.cfg.decoder_input_dim, 1, 0, 1, &cache.latents, &d_x1);

        gpu.submit(&[], &steps);
    }

    pub fn param_names(&self) -> Vec<String> {
        self.w.weight_bufs().into_iter().map(|(n, _)| n.to_string()).collect()
    }
    fn size_of(&self, name: &str) -> usize {
        *self.sizes.get(name).unwrap_or_else(|| panic!("no such parameter {name:?}"))
    }
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        let (_, buf) = self.w.weight_bufs().into_iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("no such weight {name:?}"));
        self.gpu.read(buf, self.size_of(name))
    }
    pub fn write_weight(&self, name: &str, data: &[f32]) {
        let (_, buf) = self.w.weight_bufs().into_iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("no such weight {name:?}"));
        self.gpu.write_f32(buf, data);
    }
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        let (_, buf) = self.w.grad_bufs().into_iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("no such grad {name:?}"));
        self.gpu.read(buf, self.size_of(name))
    }
    pub fn zero_grads(&self) {
        for (name, buf) in self.w.grad_bufs() {
            let n = self.size_of(name);
            self.gpu.write(buf, &vec![0u32; n]);
        }
    }
    pub fn loss(&self) -> f32 {
        let l = self.run_forward();
        *self.last_loss.borrow_mut() = l;
        l
    }
    pub fn backward(&self) {
        self.run_backward();
    }

    /// The last forward's output waveform - test-only, to check this
    /// module's own forward against `vocoder::forward`'s served path.
    pub fn output(&self) -> Vec<f32> {
        let cache_ref = self.cache.borrow();
        let cache = cache_ref.as_ref().expect("Trainer::output called before a forward (loss()) ran");
        self.gpu.read(&cache.output, (cache.rows * cache.out_len) as usize)
    }
}

#[allow(clippy::too_many_arguments)]
fn conv1d_bias_fwd(gpu: &Gpu, steps: &mut Vec<Step>, w: &ConvD, n: u32, cin: u32, l: u32, cout: u32, k: u32, pad: u32, dilation: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let lo = Conv1d::out_len(l, k, 1, pad, pad, dilation);
    let c = Conv1d { n, cin, l, cout, k, stride: 1, pad, dilation, groups: 1, lo };
    let y = gpu.storage(u64::from(n) * u64::from(cout) * u64::from(lo) * 4);
    steps.push(conv1d_fwd(gpu, &conv_kernels(), &c, x, &w.w, &y));
    steps.push(gpu.step(BIAS_ADD, &[&y, &w.b], &[n * cout * lo, cout, lo], n * cout * lo));
    y
}

#[allow(clippy::too_many_arguments)]
fn convtr1d_bias_fwd(gpu: &Gpu, steps: &mut Vec<Step>, w: &ConvD, n: u32, cin: u32, l: u32, cout: u32, k: u32, stride: u32, pad: u32, lo: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let c = Conv1d { n, cin, l, cout, k, stride, pad, dilation: 1, groups: 1, lo };
    let y = gpu.storage(u64::from(n) * u64::from(cout) * u64::from(lo) * 4);
    steps.push(convtr1d_fwd(gpu, &convtr_kernels(), &c, x, &w.w, &y));
    steps.push(gpu.step(BIAS_ADD, &[&y, &w.b], &[n * cout * lo, cout, lo], n * cout * lo));
    y
}

fn snake_fwd(gpu: &Gpu, steps: &mut Vec<Step>, alpha: &DeviceBuffer, rows: u32, c: u32, inner: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let sc = Snake1d { rows, c, inner, eps: SNAKE_EPS };
    let y = gpu.storage(u64::from(sc.total()) * 4);
    steps.push(snake1d_fwd(gpu, &snake_kernels(), &sc, x, alpha, &y));
    y
}

/// `dx` of a conv-with-bias, ALSO writing `dw`/`db` into `w`'s (pre-zeroed
/// by [`Trainer::zero_grads`]) gradient buffers.
#[allow(clippy::too_many_arguments)]
fn conv1d_bias_bwd(gpu: &Gpu, steps: &mut Vec<Step>, w: &ConvD, n: u32, cin: u32, l: u32, cout: u32, k: u32, pad: u32, dilation: u32, x: &DeviceBuffer, dy: &DeviceBuffer) -> DeviceBuffer {
    let lo = Conv1d::out_len(l, k, 1, pad, pad, dilation);
    let c = Conv1d { n, cin, l, cout, k, stride: 1, pad, dilation, groups: 1, lo };
    let dx = gpu.storage(u64::from(n) * u64::from(cin) * u64::from(l) * 4);
    steps.extend(conv1d_bwd(gpu, &conv_kernels(), &c, dy, x, &w.w, Some(&dx), Some(&w.dw)));
    steps.push(gpu.step(BIAS_GRAD_NCL, &[dy, &w.db], &[n, cout, lo], cout));
    dx
}

#[allow(clippy::too_many_arguments)]
fn convtr1d_bias_bwd(gpu: &Gpu, steps: &mut Vec<Step>, w: &ConvD, n: u32, cin: u32, l: u32, cout: u32, k: u32, stride: u32, pad: u32, lo: u32, x: &DeviceBuffer, dy: &DeviceBuffer) -> DeviceBuffer {
    let c = Conv1d { n, cin, l, cout, k, stride, pad, dilation: 1, groups: 1, lo };
    let dx = gpu.storage(u64::from(n) * u64::from(cin) * u64::from(l) * 4);
    steps.extend(convtr1d_bwd(gpu, &convtr_kernels(), &c, dy, x, &w.w, Some(&dx), Some(&w.dw)));
    steps.push(gpu.step(BIAS_GRAD_NCL, &[dy, &w.db], &[n, cout, lo], cout));
    dx
}

/// `dx` of a Snake activation, ALSO writing `dalpha` into `dalpha_out`.
#[allow(clippy::too_many_arguments)]
fn snake_bwd(gpu: &Gpu, steps: &mut Vec<Step>, alpha: &DeviceBuffer, rows: u32, c: u32, inner: u32, x: &DeviceBuffer, dy: &DeviceBuffer, dalpha_out: &DeviceBuffer) -> DeviceBuffer {
    let sc = Snake1d { rows, c, inner, eps: SNAKE_EPS };
    let dx = gpu.storage(u64::from(sc.total()) * 4);
    steps.push(snake1d_bwd_dx(gpu, &snake_kernels(), &sc, dy, x, alpha, &dx));
    steps.push(snake1d_bwd_dalpha(gpu, &snake_kernels(), &sc, dy, x, alpha, dalpha_out));
    dx
}

/// Random weights at `cfg`'s dims, deterministic from `seed` - shared by
/// this crate's own tests and `crates/gradcheck::minimaxmusic3::check_vocoder`
/// (real weights are far too large for a directional-derivative sweep; a
/// gradcheck fixture always needs a `::tiny()`-scale random one). Snake
/// alphas are drawn positive and away from 0, matching every real trained
/// checkpoint value and keeping `1/(alpha+eps)` finite-difference-stable.
pub fn random_weights(cfg: &VocoderConfig, seed: u64) -> VocoderWeights {
    let mut r = Lcg::new(seed);
    let conv = |cout: usize, cin: usize, k: usize, r: &mut Lcg| ConvW { weight: r.vec_scaled(cout * cin * k, 0.3), bias: r.vec_scaled(cout, 0.1) };
    let alpha = |n: usize, r: &mut Lcg| -> Vec<f32> { r.vec_scaled(n, 0.05).iter().map(|v: &f32| v.abs() + 0.3).collect() };
    let half = cfg.latent_channels as usize / 2;
    let mut dim = cfg.decoder_hidden_dim as usize;
    let mut blocks = Vec::new();
    for &stride in &cfg.upsampling_ratios {
        let out_dim = dim / 2;
        let res_units = (0..3)
            .map(|_| ResidualUnitW {
                snake1_alpha: alpha(out_dim, &mut r),
                conv1: conv(out_dim, out_dim, 7, &mut r),
                snake2_alpha: alpha(out_dim, &mut r),
                conv2: conv(out_dim, out_dim, 1, &mut r),
            })
            .collect();
        blocks.push(VocoderBlockW { snake1_alpha: alpha(dim, &mut r), conv_t1: conv(out_dim, dim, 2 * stride as usize, &mut r), res_units });
        dim = out_dim;
    }
    VocoderWeights {
        dec_in_proj: conv(cfg.decoder_input_dim as usize, half, 1, &mut r),
        conv_in: conv(cfg.decoder_hidden_dim as usize, cfg.decoder_input_dim as usize, 7, &mut r),
        blocks,
        snake_out_alpha: alpha(dim, &mut r),
        conv_out: conv(1, dim, 7, &mut r),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocoder;

    #[test]
    fn forward_matches_serving_forward() {
        let cfg = VocoderConfig::tiny();
        let w = random_weights(&cfg, 11);
        let (batch, length) = (1, 5);
        let mut r = Lcg::new(12);
        let latents = r.vec_scaled(batch * cfg.latent_channels as usize * length, 0.5);

        let gpu = Gpu::new_cpu(vocoder::PIPELINES);
        let served = vocoder::forward(&gpu, &cfg, &w, &latents, batch, length);

        let target = vec![0.0f32; served.len()];
        let trainer = Trainer::new(cfg, &w, latents, batch, length, target);
        let _ = trainer.loss();
        let trained_fwd = trainer.output();

        assert_eq!(served.len(), trained_fwd.len());
        let max_abs = served.iter().zip(&trained_fwd).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs < 1e-4, "train.rs's own forward drifted from vocoder::forward: max_abs={max_abs}");
    }

    #[test]
    fn backward_matches_finite_differences() {
        let cfg = VocoderConfig::tiny();
        let w = random_weights(&cfg, 21);
        let (batch, length) = (1, 4);
        let mut r = Lcg::new(22);
        let latents = r.vec_scaled(batch * cfg.latent_channels as usize * length, 0.5);
        let out_len = length * cfg.upsampling_ratios.iter().product::<u32>() as usize;
        let target = r.vec_scaled(batch * 2 * out_len, 0.5);

        let trainer = Trainer::new(cfg, &w, latents, batch, length, target);
        trainer.zero_grads();
        let _ = trainer.loss();
        trainer.backward();

        let eps = 5e-3f32;
        let mut checked = 0;
        for name in trainer.param_names() {
            let base = trainer.read_weight(&name);
            let ana = trainer.read_grad(&name);
            // One representative index per parameter keeps this test fast
            // (the vocoder has ~40 named tensors); every kernel family
            // (conv1d/convtr1d fwd+bwd, snake1d fwd+bwd, bias_grad_ncl,
            // tanh_act_bwd, the residual add) is exercised by SOME parameter.
            let i = 0usize;
            let mut p = base.clone();
            p[i] = base[i] + eps;
            trainer.write_weight(&name, &p);
            let lp = trainer.loss();
            p[i] = base[i] - eps;
            trainer.write_weight(&name, &p);
            let lm = trainer.loss();
            trainer.write_weight(&name, &base);
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (num - ana[i]).abs() < 2e-2 + 2e-2 * num.abs().max(ana[i].abs()),
                "{name}[{i}]: numeric={num} analytic={} (loss+={lp} loss-={lm})",
                ana[i]
            );
            checked += 1;
        }
        assert!(checked > 30, "expected the full ~40-parameter vocoder to be checked, got {checked}");
    }
}
