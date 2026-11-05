// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! MiniMax-H3's audio VAE **encoder** - the half `ref2va` needs to turn a
//! reference clip's soundtrack into the 32-channel, 40 Hz latent the DiT
//! conditions on. Ported from `diffusers`' Apache-2.0
//! `AutoencoderKLMiniMaxH3Audio.encode`:
//!
//! 1. Right-pad the mono waveform to a multiple of the hop (`prod(rates)` =
//!    800 samples at 32 kHz).
//! 2. The DAC waveform encoder: `Conv1d(1, 64, 7)`, then per rate `s` three
//!    residual units (`Snake -> Conv1d(k=7, dilation 1/3/9) -> Snake ->
//!    Conv1d(k=1)`, plus the input) at the incoming width, `Snake` and a
//!    strided channel-doubling `Conv1d(k=2s, stride=s, pad=ceil(s/2))`; then
//!    `Snake -> Conv1d(2048, 2048, 3)`. Every convolution is weight-normed in
//!    the checkpoint (`weight_g`/`weight_v`), folded once at import
//!    ([`import_audio_vae_encoder`]).
//! 3. `pre_block`, a residual causal-attention + GeGLU block that narrows
//!    2048 features to 32: `proj(norm3(x)) + attn(norm1(x))`, then
//!    `+ mlp(norm2(.))`. The attention is unusual and taken exactly as the
//!    reference writes it: a bias-free fused `qkv` with separate query and
//!    value biases and a frozen zero key bias, causal, 8 heads of 256 - and
//!    the heads are MEAN-pooled away rather than concatenated, the remaining
//!    256 features adaptive-average-pooled to 32 (contiguous groups of 8),
//!    then projected.
//! 4. The posterior MEAN (`mean_proj`, a 1x1 conv): the pipeline always
//!    consumes `latent_dist.mode()`, so `logs_proj` is never evaluated.
//!
//! The waveform convolutions - a 32 kHz stream at up to 2048 channels, the
//! cost of this module - run on the device through `audio::conv`'s selected
//! lowering and the shared `snake1d` kernel, on the vocoder's own device
//! context ([`crate::vocoder`]). `pre_block` runs over the 40 Hz latent rows,
//! at most a few hundred per reference clip, on the host.
//!
//! Stereo is two mono encodes: H3 carries a stereo reference as two batch
//! items through this encoder, exactly as its decoder handles the two
//! channels of a generated soundtrack.
//!
//! Swedish Embedded AB implements audio codecs and their conditioning paths
//! for generative models for its clients. If your team needs expertise in
//! audio representation learning, you can procure our services by sending an
//! email to info@swedishembedded.com.

use gpu_core::DeviceBuffer;
use model::hostmath::{gelu_tanh, layernorm_rows, linear_rows};
use vae::blocks::Tensors;

use crate::vocoder::{gemm_kernels, kernels_id, Ctx, K_SNAKE1D};

/// `nn.LayerNorm`'s default epsilon, which every `pre_block` norm uses.
const LN_EPS: f32 = 1e-5;
/// `Snake1d`'s fixed `1e-9` (`(alpha + 1e-9).reciprocal()`).
const SNAKE_EPS: f32 = 1e-9;
/// The residual units' dilations, in order.
const DILATIONS: [u32; 3] = [1, 3, 9];

/// The real checkpoint's `audio_vae/config.json` encoder half.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioEncoderConfig {
    pub encoder_dim: u32,
    pub rates: Vec<u32>,
    pub latent_dim: u32,
    pub latent_channels: u32,
    pub num_attention_heads: u32,
}

impl AudioEncoderConfig {
    pub fn h3() -> AudioEncoderConfig {
        AudioEncoderConfig { encoder_dim: 64, rates: vec![2, 4, 4, 5, 5], latent_dim: 2048, latent_channels: 32, num_attention_heads: 8 }
    }

    /// Samples per latent frame (800 at the real config: 32 kHz / 40 Hz).
    pub fn hop(&self) -> u32 {
        self.rates.iter().product()
    }

    /// Width entering encoder stage `i` (`encoder_dim * 2^i`).
    fn stage_in(&self, i: usize) -> u32 {
        self.encoder_dim << i
    }

    /// Every tensor [`encode`] reads, at its stored shape, in the
    /// checkpoint's own names (weight-norm pairs already folded to `.weight`).
    pub fn tensor_manifest(&self) -> Vec<(String, Vec<usize>)> {
        let mut m: Vec<(String, Vec<usize>)> = Vec::new();
        let conv = |m: &mut Vec<(String, Vec<usize>)>, p: &str, cout: u32, cin: u32, k: u32| {
            m.push((format!("{p}.weight"), vec![cout as usize, cin as usize, k as usize]));
            m.push((format!("{p}.bias"), vec![cout as usize]));
        };
        let snake = |m: &mut Vec<(String, Vec<usize>)>, p: &str, c: u32| m.push((format!("{p}.alpha"), vec![1, c as usize, 1]));
        conv(&mut m, "encoder.block.0", self.encoder_dim, 1, 7);
        for (i, &stride) in self.rates.iter().enumerate() {
            let (half, d) = (self.stage_in(i), self.stage_in(i + 1));
            let p = format!("encoder.block.{}.block", i + 1);
            for j in 0..DILATIONS.len() {
                let u = format!("{p}.{j}.block");
                snake(&mut m, &format!("{u}.0"), half);
                conv(&mut m, &format!("{u}.1"), half, half, 7);
                snake(&mut m, &format!("{u}.2"), half);
                conv(&mut m, &format!("{u}.3"), half, half, 1);
            }
            snake(&mut m, &format!("{p}.3"), half);
            conv(&mut m, &format!("{p}.4"), d, half, 2 * stride);
        }
        let n = self.rates.len();
        let top = self.stage_in(n);
        snake(&mut m, &format!("encoder.block.{}", n + 1), top);
        conv(&mut m, &format!("encoder.block.{}", n + 2), self.latent_dim, top, 3);

        let (ld, lc) = (self.latent_dim as usize, self.latent_channels as usize);
        for norm in ["norm1", "norm3"] {
            m.push((format!("pre_block.{norm}.weight"), vec![ld]));
            m.push((format!("pre_block.{norm}.bias"), vec![ld]));
        }
        for norm in ["norm2", "mlp.norm"] {
            m.push((format!("pre_block.{norm}.weight"), vec![lc]));
            m.push((format!("pre_block.{norm}.bias"), vec![lc]));
        }
        m.push(("pre_block.attn.qkv.weight".into(), vec![3 * ld, ld]));
        m.push(("pre_block.attn.q_bias".into(), vec![ld]));
        m.push(("pre_block.attn.v_bias".into(), vec![ld]));
        m.push(("pre_block.attn.proj.weight".into(), vec![lc, lc]));
        m.push(("pre_block.attn.proj.bias".into(), vec![lc]));
        m.push(("pre_block.proj.weight".into(), vec![lc, ld]));
        m.push(("pre_block.proj.bias".into(), vec![lc]));
        for (w, out, inn) in [("w0", 2 * lc, lc), ("w1", 2 * lc, lc), ("w2", lc, 2 * lc)] {
            m.push((format!("pre_block.mlp.{w}.weight"), vec![out, inn]));
            m.push((format!("pre_block.mlp.{w}.bias"), vec![out]));
        }
        m.push(("mean_proj.weight".into(), vec![lc, lc, 1]));
        m.push(("mean_proj.bias".into(), vec![lc]));
        m
    }
}

/// Import the encoder half of the `audio_vae/` weights: every manifest entry
/// at its manifest shape, every `encoder.*` convolution's weight-norm pair
/// folded. The decoder half and `logs_proj` (never evaluated) are left for
/// [`crate::import::import_audio_vae_decoder`] / unread.
pub fn import_audio_vae_encoder(dir: &str, cfg: &AudioEncoderConfig) -> Result<Tensors, String> {
    let src = checkpoint::weightio::WeightReader::open_hf_dir(std::path::Path::new(dir)).map_err(|e| format!("minimaxh3 audio encoder import: {dir}: {e}"))?;
    let read = |name: &str, len: usize| -> Result<Vec<f32>, String> {
        let mut v = Vec::with_capacity(len);
        if !checkpoint::TensorSource::with_tensor_chunks(&src, name, 1 << 20, &mut |_, c| v.extend_from_slice(c)) {
            return Err(format!("minimaxh3 audio encoder import: missing tensor {name:?}"));
        }
        if v.len() != len {
            return Err(format!("minimaxh3 audio encoder import: {name} has {} values, expected {len}", v.len()));
        }
        Ok(v)
    };
    let mut out = Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let numel: usize = shape.iter().product();
        let data = if name.starts_with("encoder.") && name.ends_with(".weight") {
            let cout = shape[0];
            let g = read(&format!("{name}_g"), cout)?;
            let v = read(&format!("{name}_v"), numel)?;
            audio::conv::fold_weight_norm(&g, &v, cout)
        } else {
            read(&name, numel)?
        };
        out.insert(name, (shape, data));
    }
    Ok(out)
}

/// Encode one mono waveform (`samples` at the codec's own 32 kHz) to its
/// posterior mean, `[latent_channels, ceil(samples / hop)]` channel-major.
pub fn encode(cfg: &AudioEncoderConfig, tensors: &Tensors, wave: &[f32], device: Option<&str>) -> Vec<f32> {
    assert!(!wave.is_empty(), "minimaxh3 audio encoder: an empty waveform has no latent");
    let hop = cfg.hop() as usize;
    let t = wave.len().div_ceil(hop);
    let mut padded = wave.to_vec();
    padded.resize(t * hop, 0.0);

    let mut cx = Ctx::new(device, tensors);
    let x = cx.gpu.storage_init("audio_encoder.wave", &padded);
    let (mut h, mut l) = conv(&mut cx, "encoder.block.0", 1, cfg.encoder_dim, 7, 1, 3, 1, (t * hop) as u32, &x);
    for (i, &stride) in cfg.rates.iter().enumerate() {
        let (half, d) = (cfg.stage_in(i), cfg.stage_in(i + 1));
        let p = format!("encoder.block.{}.block", i + 1);
        for (j, &dilation) in DILATIONS.iter().enumerate() {
            h = residual_unit(&mut cx, &format!("{p}.{j}.block"), half, dilation, l, &h);
        }
        h = snake(&cx, &format!("{p}.3"), half, l, &h);
        (h, l) = conv(&mut cx, &format!("{p}.4"), half, d, 2 * stride, stride, stride.div_ceil(2), 1, l, &h);
    }
    let n = cfg.rates.len();
    h = snake(&cx, &format!("encoder.block.{}", n + 1), cfg.stage_in(n), l, &h);
    (h, l) = conv(&mut cx, &format!("encoder.block.{}", n + 2), cfg.stage_in(n), cfg.latent_dim, 3, 1, 1, 1, l, &h);
    assert_eq!(l as usize, t, "minimaxh3 audio encoder: the stack produced {l} frames, the hop implies {t}");
    let features = cx.gpu.read(&h, cfg.latent_dim as usize * t);

    let narrowed = pre_block(cfg, tensors, &features, t);
    mean_proj(cfg, tensors, &narrowed, t)
}

/// One DAC residual unit at width `c`: `x + conv1(snake(conv7_dil(snake(x))))`.
fn residual_unit(cx: &mut Ctx, p: &str, c: u32, dilation: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let a = snake(cx, &format!("{p}.0"), c, l, x);
    let (b, _) = conv(cx, &format!("{p}.1"), c, c, 7, 1, 3 * dilation, dilation, l, &a);
    let b = snake(cx, &format!("{p}.2"), c, l, &b);
    let (r, _) = conv(cx, &format!("{p}.3"), c, c, 1, 1, 0, 1, l, &b);
    let y = cx.gpu.storage((c * l) as u64);
    cx.gpu.submit(&[], &[cx.gpu.step(crate::vocoder::K_ADD2, &[x, &r, &y], &[c * l], c * l)]);
    y
}

/// A biased `Conv1d` over NCL `[cin, l]`, zero padding `pad` both sides.
#[allow(clippy::too_many_arguments)]
fn conv(cx: &mut Ctx, p: &str, cin: u32, cout: u32, k: u32, stride: u32, pad: u32, dilation: u32, l: u32, x: &DeviceBuffer) -> (DeviceBuffer, u32) {
    let lo = audio::conv::Conv1d::out_len(l, k, stride, pad, pad, dilation);
    let cfg = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride, pad, dilation, groups: 1, lo };
    let w = cx.upload(&format!("{p}.weight"));
    let b = cx.upload(&format!("{p}.bias"));
    let y = cx.gpu.storage((cout * lo) as u64);
    let steps = audio::conv::conv1d_bias_fwd(&cx.gpu, &gemm_kernels(kernels_id()), &cfg, x, &w, &b, &y, &mut cx.scratch);
    cx.gpu.submit(&[], &steps);
    (y, lo)
}

/// `Snake1d` over NCL `[c, l]` with the checkpoint's `[1, c, 1]` alpha.
fn snake(cx: &Ctx, p: &str, c: u32, l: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let alpha = cx.upload(&format!("{p}.alpha"));
    let shape = audio::snake::Snake1d { rows: 1, c, inner: l, eps: SNAKE_EPS };
    let y = cx.gpu.storage(shape.total() as u64);
    let k = audio::snake::SnakeKernels { fwd: K_SNAKE1D, bwd_dx: K_SNAKE1D, bwd_dalpha: K_SNAKE1D };
    cx.gpu.submit(&[], &[audio::snake::snake1d_fwd(&cx.gpu, &k, &shape, x, &alpha, &y)]);
    y
}

fn w<'a>(t: &'a Tensors, name: &str) -> &'a [f32] {
    &t.get(name).unwrap_or_else(|| panic!("minimaxh3 audio encoder: missing tensor {name}")).1
}

/// `y = x @ W^T + b` over `rows` rows.
fn linear(x: &[f32], t: &Tensors, p: &str, rows: usize, inn: usize, out: usize) -> Vec<f32> {
    let mut y = linear_rows(x, w(t, &format!("{p}.weight")), rows, inn, out);
    let b = w(t, &format!("{p}.bias"));
    for row in y.chunks_exact_mut(out) {
        row.iter_mut().zip(b).for_each(|(v, b)| *v += b);
    }
    y
}

fn layernorm(x: &[f32], t: &Tensors, p: &str, rows: usize, c: usize) -> Vec<f32> {
    layernorm_rows(x, w(t, &format!("{p}.weight")), w(t, &format!("{p}.bias")), rows, c, LN_EPS)
}

/// `pre_block` over the encoder's `[latent_dim, t]` features: `[t,
/// latent_channels]` row-major out.
fn pre_block(cfg: &AudioEncoderConfig, tw: &Tensors, features: &[f32], t: usize) -> Vec<f32> {
    let (ld, lc, heads) = (cfg.latent_dim as usize, cfg.latent_channels as usize, cfg.num_attention_heads as usize);
    let hd = ld / heads;
    let mut x = vec![0f32; t * ld];
    for c in 0..ld {
        for r in 0..t {
            x[r * ld + c] = features[c * t + r];
        }
    }

    // attn(norm1(x)): causal, per-head, heads mean-pooled, then the head
    // dimension adaptive-average-pooled to `lc` and projected.
    let a = layernorm(&x, tw, "pre_block.norm1", t, ld);
    let mut qkv = linear_rows(&a, w(tw, "pre_block.attn.qkv.weight"), t, ld, 3 * ld);
    let (qb, vb) = (w(tw, "pre_block.attn.q_bias"), w(tw, "pre_block.attn.v_bias"));
    for row in qkv.chunks_exact_mut(3 * ld) {
        row[..ld].iter_mut().zip(qb).for_each(|(v, b)| *v += b);
        row[2 * ld..].iter_mut().zip(vb).for_each(|(v, b)| *v += b);
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let mut pooled = vec![0f32; t * hd];
    let mut scores = vec![0f32; t];
    for h in 0..heads {
        for i in 0..t {
            let q = &qkv[i * 3 * ld + h * hd..i * 3 * ld + (h + 1) * hd];
            for (j, s) in scores[..=i].iter_mut().enumerate() {
                let k = &qkv[j * 3 * ld + ld + h * hd..j * 3 * ld + ld + (h + 1) * hd];
                *s = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
            }
            model::hostmath::softmax(&mut scores[..=i]);
            let dst = &mut pooled[i * hd..(i + 1) * hd];
            for (j, &p) in scores[..=i].iter().enumerate() {
                let v = &qkv[j * 3 * ld + 2 * ld + h * hd..j * 3 * ld + 2 * ld + (h + 1) * hd];
                dst.iter_mut().zip(v).for_each(|(d, v)| *d += p * v / heads as f32);
            }
        }
    }
    assert_eq!(hd % lc, 0, "adaptive_avg_pool1d({hd} -> {lc}) is only a plain block mean when {lc} divides {hd}");
    let group = hd / lc;
    let narrowed: Vec<f32> = pooled.chunks_exact(group).map(|g| g.iter().sum::<f32>() / group as f32).collect();
    let attn = linear(&narrowed, tw, "pre_block.attn.proj", t, lc, lc);

    let skip = linear(&layernorm(&x, tw, "pre_block.norm3", t, ld), tw, "pre_block.proj", t, ld, lc);
    let mut h: Vec<f32> = skip.iter().zip(&attn).map(|(a, b)| a + b).collect();

    // h + mlp(norm2(h)), the GeGLU MLP applying its own norm first.
    let m = layernorm(&layernorm(&h, tw, "pre_block.norm2", t, lc), tw, "pre_block.mlp.norm", t, lc);
    let gate = linear(&m, tw, "pre_block.mlp.w0", t, lc, 2 * lc);
    let value = linear(&m, tw, "pre_block.mlp.w1", t, lc, 2 * lc);
    let prod: Vec<f32> = gate.iter().zip(&value).map(|(g, v)| gelu_tanh(*g) * v).collect();
    let mlp = linear(&prod, tw, "pre_block.mlp.w2", t, 2 * lc, lc);
    h.iter_mut().zip(&mlp).for_each(|(a, b)| *a += b);
    h
}

/// `mean_proj` (a 1x1 conv) over `[t, lc]` rows, channel-major `[lc, t]` out.
fn mean_proj(cfg: &AudioEncoderConfig, tw: &Tensors, h: &[f32], t: usize) -> Vec<f32> {
    let lc = cfg.latent_channels as usize;
    let rows = linear(h, tw, "mean_proj", t, lc, lc);
    let mut out = vec![0f32; lc * t];
    for r in 0..t {
        for c in 0..lc {
            out[c * t + r] = rows[r * lc + c];
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden() -> Option<(Tensors, Vec<checkpoint::safetensors::StTensor>)> {
        let fx = brain_testutil::testdata("golden/minimaxh3/audio_encoder/minimaxh3_audio_encoder.safetensors");
        if !std::path::Path::new(&fx).exists() {
            brain_testutil::skip(&format!("{fx} absent - run tools/goldens/minimaxh3_audio_encoder_dump_reference.py"));
            return None;
        }
        let Some(dir) = brain_testutil::model_dir("MiniMaxAI/MiniMax-H3").map(|d| format!("{d}/audio_vae")).filter(|d| std::path::Path::new(d).exists()) else {
            brain_testutil::skip("no MiniMaxAI/MiniMax-H3/audio_vae in the model store");
            return None;
        };
        let weights = import_audio_vae_encoder(&dir, &AudioEncoderConfig::h3()).expect("import the real encoder");
        Some((weights, checkpoint::safetensors::read(&fx).expect("read golden")))
    }

    /// The real-weight encoder against `diffusers`' own `encode()`, gated on
    /// cosine AND rel_l2 (a scale error is invisible to cosine), with the DAC
    /// stack's output checked on its own so a `pre_block` bug and a
    /// convolution bug cannot hide behind each other.
    #[test]
    fn the_real_encoder_matches_the_reference() {
        let Some((weights, fx)) = golden() else { return };
        let get = |n: &str| fx.iter().find(|t| t.name == n).unwrap_or_else(|| panic!("no golden {n}"));
        let cfg = AudioEncoderConfig::h3();
        let latent = encode(&cfg, &weights, &get("waveform").data, Some("cpu"));
        let mut r = brain_testutil::parity::Report::new(0.99999);
        r.check("latent (posterior mean)", &latent, &get("latent").data);
        r.finish("minimaxh3 audio encoder vs diffusers encode()");
        let want = &get("latent").data;
        let rel = (latent.iter().zip(want).map(|(a, b)| ((a - b) as f64).powi(2)).sum::<f64>() / want.iter().map(|b| (*b as f64).powi(2)).sum::<f64>()).sqrt();
        assert!(rel < 1e-4, "latent rel_l2 {rel}");
    }

    #[test]
    fn the_manifest_covers_the_real_encoder_and_nothing_else() {
        let cfg = AudioEncoderConfig::h3();
        let m = cfg.tensor_manifest();
        let names: std::collections::HashSet<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), m.len(), "duplicate manifest name");
        // 1 stem conv + per stage (3 units x (2 convs + 2 snakes) + snake + conv) + snake + conv.
        let convs = 1 + 5 * (3 * 2 + 1) + 1;
        let snakes = 5 * (3 * 2 + 1) + 1;
        assert_eq!(m.iter().filter(|(n, _)| n.starts_with("encoder.") && n.ends_with(".weight")).count(), convs);
        assert_eq!(m.iter().filter(|(n, _)| n.ends_with(".alpha")).count(), snakes);
        assert_eq!(cfg.hop(), 800, "32 kHz / 40 latent frames per second");
    }
}
