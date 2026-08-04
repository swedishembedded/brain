// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Pure-CPU **stateful streaming** codec decoder. Mirrors
//! [`crate::model::Codec::decode`] / [`onnx`-graph `codec_topology`] op-for-op,
//! but decodes in frame-chunks carrying per-module state, so each chunk emits
//! only its new audio (no warmup re-decode).
//!
//! Split:
//!  * **front** (RVQ dequant -> `pre_conv` -> 8-layer causal transformer ->
//!    `pre_transformer.output_proj`) is *causal*, so its output for frames
//!    `[a,b)` is independent of future frames — computed once over all codes and
//!    sliced. Cheap (runs over `T` frames, not the upsampled length).
//!  * **back** (upsample + SEANet over the ~`T*1920` upsampled sequence) is the
//!    expensive part and is *streamed*: every causal (transposed-)conv carries
//!    its state via [`crate::streaming`], so the per-chunk cost scales with the
//!    chunk size, not a window. `decode_full == decode_streaming(chunk = T)`.
//!
//! Pointwise ops (SnakeBeta, RMSNorm/LayerNorm, GELU, matmuls, RoPE) act per
//! position, so chunking is exact for them; the only cross-chunk state is in the
//! conv primitives (independently proven exact). The tests confirm chunked ==
//! one-shot for the whole back.

use std::collections::HashMap;
// Single implementation of the elementwise/normalisation math.
use model::hostmath;

use crate::config::CodecConfig;
use crate::streaming::{StreamConv1d, StreamConvTr1d};

type W = HashMap<String, Vec<f32>>;

// ---- small host ops (flat slices; NCL = [C,L] index c*L+l, TM = [L,C] l*C+c) ----

fn erf(x: f32) -> f32 {
    // Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7).
    let s = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152) * t) + 1.421_413_7) * t - 0.284_496_74) * t + 0.254_829_6)
            * t
            * (-x * x).exp();
    s * y
}

/// `y[t,o] = sum_i x[t,i] * w[o,i]` for `x:[rows,inp]`, `w:[out,inp]`.
fn matmul(x: &[f32], w: &[f32], rows: usize, inp: usize, out: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    for t in 0..rows {
        let xr = &x[t * inp..t * inp + inp];
        for o in 0..out {
            let wr = &w[o * inp..o * inp + inp];
            y[t * out + o] = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    }
    y
}

fn add_bias(y: &mut [f32], b: &[f32], rows: usize, out: usize) {
    for t in 0..rows {
        for o in 0..out {
            y[t * out + o] += b[o];
        }
    }
}

fn transpose(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut o = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            o[c * rows + r] = x[r * cols + c];
        }
    }
    o
}

/// RMSNorm over the last axis (`c`) of token-major `[rows,c]`.

/// LayerNorm over the last axis (`c`) of token-major `[rows,c]`, affine.

fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 0.5 * *v * (1.0 + erf(*v * std::f32::consts::FRAC_1_SQRT_2));
    }
}

/// Per-last-channel scale (`[rows,c]` *= `s[c]`).
fn scale_last(x: &mut [f32], s: &[f32], rows: usize, c: usize) {
    for t in 0..rows {
        for j in 0..c {
            x[t * c + j] *= s[j];
        }
    }
}

/// SnakeBeta over NCL `[c,l]`: `x + (1/(exp(b)+eps))*sin(exp(a)*x)^2`.
fn snake(x: &[f32], alpha: &[f32], beta: &[f32], c: usize, l: usize) -> Vec<f32> {
    let eps = 1e-9f32;
    let mut y = vec![0.0f32; c * l];
    for ch in 0..c {
        let ea = alpha[ch].exp();
        let inv = 1.0 / (beta[ch].exp() + eps);
        for i in 0..l {
            let v = x[ch * l + i];
            let s = (ea * v).sin();
            y[ch * l + i] = v + inv * s * s;
        }
    }
    y
}

// ---- streaming back (upsample + SEANet) ----

/// Holds the per-conv streaming state for the codec back, created lazily by name.
struct Back<'a> {
    w: &'a W,
    cfg: &'a CodecConfig,
    convs: HashMap<String, StreamConv1d>,
    convtrs: HashMap<String, StreamConvTr1d>,
}

impl<'a> Back<'a> {
    fn new(w: &'a W, cfg: &'a CodecConfig) -> Back<'a> {
        Back { w, cfg, convs: HashMap::new(), convtrs: HashMap::new() }
    }

    /// Streaming causal conv on NCL `[cin,l]` -> `[cout,l]`.
    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, x: &[f32], prefix: &str, cin: usize, cout: usize, k: usize, dil: usize, groups: usize, l: usize) -> Vec<f32> {
        if !self.convs.contains_key(prefix) {
            let wn = format!("{prefix}.conv.weight");
            let bn = format!("{prefix}.conv.bias");
            let sc = StreamConv1d::new(cin, cout, k, dil, groups, self.w[&wn].clone(), self.w[&bn].clone());
            self.convs.insert(prefix.to_string(), sc);
        }
        self.convs.get_mut(prefix).unwrap().step(x, l)
    }

    /// Streaming causal transposed conv on NCL `[cin,l]` -> `[cout,l*stride]`.
    fn convtr_step(&mut self, x: &[f32], prefix: &str, cin: usize, cout: usize, k: usize, stride: usize, l: usize) -> Vec<f32> {
        if !self.convtrs.contains_key(prefix) {
            let wn = format!("{prefix}.conv.weight");
            let bn = format!("{prefix}.conv.bias");
            let sc = StreamConvTr1d::new(cin, cout, k, stride, self.w[&wn].clone(), self.w[&bn].clone());
            self.convtrs.insert(prefix.to_string(), sc);
        }
        self.convtrs.get_mut(prefix).unwrap().step(x, l)
    }

    fn snake_w(&self, prefix: &str) -> (Vec<f32>, Vec<f32>) {
        (self.w[&format!("{prefix}.alpha")].clone(), self.w[&format!("{prefix}.beta")].clone())
    }

    /// ConvNeXt block (NCL in/out), `l` = current length.
    fn convnext(&mut self, x: &[f32], prefix: &str, c: usize, l: usize) -> Vec<f32> {
        let dw = self.conv(x, &format!("{prefix}.dwconv"), c, c, 7, 1, c, l); // [c,l]
        let tm = transpose(&dw, c, l); // [l,c]
        let gn = self.w[&format!("{prefix}.norm.weight")].clone();
        let gb = self.w[&format!("{prefix}.norm.bias")].clone();
        let normed = hostmath::layernorm_rows(&tm, &gn, &gb, l, c, 1e-6);
        let hid = c * 4;
        let p1w = self.w[&format!("{prefix}.pwconv1.weight")].clone();
        let p1b = self.w[&format!("{prefix}.pwconv1.bias")].clone();
        let mut g1 = matmul(&normed, &p1w, l, c, hid);
        add_bias(&mut g1, &p1b, l, hid);
        gelu_inplace(&mut g1);
        let p2w = self.w[&format!("{prefix}.pwconv2.weight")].clone();
        let p2b = self.w[&format!("{prefix}.pwconv2.bias")].clone();
        let mut o = matmul(&g1, &p2w, l, hid, c);
        add_bias(&mut o, &p2b, l, c);
        let gamma = self.w[&format!("{prefix}.gamma")].clone();
        scale_last(&mut o, &gamma, l, c);
        let o_ncl = transpose(&o, l, c); // [c,l]
        let mut out = x.to_vec();
        for i in 0..c * l {
            out[i] += o_ncl[i];
        }
        out
    }

    /// SEANet residual unit (NCL): `x + conv2(snake(conv1(snake(x))))`.
    fn residual_unit(&mut self, x: &[f32], prefix: &str, c: usize, dil: usize, l: usize) -> Vec<f32> {
        let (a1, b1) = self.snake_w(&format!("{prefix}.act1"));
        let y = snake(x, &a1, &b1, c, l);
        let y = self.conv(&y, &format!("{prefix}.conv1"), c, c, 7, dil, 1, l);
        let (a2, b2) = self.snake_w(&format!("{prefix}.act2"));
        let y = snake(&y, &a2, &b2, c, l);
        let y = self.conv(&y, &format!("{prefix}.conv2"), c, c, 1, 1, 1, l);
        let mut out = x.to_vec();
        for i in 0..c * l {
            out[i] += y[i];
        }
        out
    }

    /// One chunk: latent NCL `[latent, l_new]` -> waveform `[l_new * 1920]`.
    fn step(&mut self, latent: &[f32], l_new: usize) -> Vec<f32> {
        let latent_dim = self.cfg.latent_dim as usize;
        let dec_dim = self.cfg.decoder_dim as usize;
        let mut l = l_new;
        let mut h = latent.to_vec(); // [latent, l]

        // upsample stages: convtr (k=stride=factor, no overlap) + convnext.
        let ratios = self.cfg.upsampling_ratios.clone();
        for (u, &factor) in ratios.iter().enumerate() {
            let f = factor as usize;
            h = self.convtr_step(&h, &format!("upsample.{u}.0"), latent_dim, latent_dim, f, f, l);
            l *= f;
            h = self.convnext(&h, &format!("upsample.{u}.1"), latent_dim, l);
        }

        // SEANet decoder.
        h = self.conv(&h, "decoder.0", latent_dim, dec_dim, 7, 1, 1, l);
        let rates = self.cfg.upsample_rates.clone();
        for (i, &rate) in rates.iter().enumerate() {
            let in_dim = dec_dim >> i;
            let out_dim = dec_dim >> (i + 1);
            let bp = format!("decoder.{}", i + 1);
            let (a, b) = self.snake_w(&format!("{bp}.block.0"));
            h = snake(&h, &a, &b, in_dim, l);
            h = self.convtr_step(&h, &format!("{bp}.block.1"), in_dim, out_dim, 2 * rate as usize, rate as usize, l);
            l *= rate as usize;
            for (j, dil) in [(2usize, 1usize), (3, 3), (4, 9)] {
                h = self.residual_unit(&h, &format!("{bp}.block.{j}"), out_dim, dil, l);
            }
        }
        let out_dim = dec_dim >> rates.len();
        let (a, b) = self.snake_w("decoder.5");
        h = snake(&h, &a, &b, out_dim, l);
        h = self.conv(&h, "decoder.6", out_dim, 1, 7, 1, 1, l); // [1, l]
        for v in h.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        h
    }
}

// ---- front (RVQ + pre_conv + transformer) -> latent NCL [latent, T] ----

fn gather_codebook(w: &W, table: &str, codes: &[i64], col: usize, nq: usize, t: usize, dim: usize) -> Vec<f32> {
    let tab = &w[table];
    let mut out = vec![0.0f32; t * dim];
    for f in 0..t {
        let idx = codes[col * t + f] as usize;
        out[f * dim..f * dim + dim].copy_from_slice(&tab[idx * dim..idx * dim + dim]);
    }
    let _ = nq;
    out
}

fn full_causal_conv(w: &W, prefix: &str, x: &[f32], cin: usize, cout: usize, k: usize, dil: usize, groups: usize, l: usize) -> Vec<f32> {
    use audio::conv::{conv1d_ref, Conv1d};
    let ctx = dil * (k - 1);
    let lin = ctx + l;
    let mut xin = vec![0.0f32; cin * lin];
    for c in 0..cin {
        xin[c * lin + ctx..c * lin + lin].copy_from_slice(&x[c * l..c * l + l]);
    }
    let conv = Conv1d {
        n: 1,
        cin: cin as u32,
        l: lin as u32,
        cout: cout as u32,
        k: k as u32,
        stride: 1,
        pad: 0,
        dilation: dil as u32,
        groups: groups as u32,
        lo: l as u32,
    };
    let mut y = conv1d_ref(&conv, &xin, &w[&format!("{prefix}.conv.weight")]);
    let b = &w[&format!("{prefix}.conv.bias")];
    for co in 0..cout {
        for i in 0..l {
            y[co * l + i] += b[co];
        }
    }
    y
}


/// Front: codes `[nq,T]` (codebook-major i64) -> latent NCL `[latent, T]`.
fn front(w: &W, cfg: &CodecConfig, codes: &[i64], t: usize) -> Vec<f32> {
    let nq = cfg.num_quantizers as usize;
    let dim = (cfg.codebook_dim / 2) as usize;
    let lat = cfg.codebook_dim as usize;
    let hidden = cfg.hidden_size as usize;
    let latent = cfg.latent_dim as usize;

    // RVQ dequant.
    let sem = gather_codebook(w, "quantizer.rvq_first.vq.layers.0.table", codes, 0, nq, t, dim);
    let first = matmul(&sem, &w["quantizer.rvq_first.output_proj.weight"], t, dim, lat);
    let mut acc = gather_codebook(w, "quantizer.rvq_rest.vq.layers.0.table", codes, 1, nq, t, dim);
    for i in 1..(nq - 1) {
        let gi = gather_codebook(w, &format!("quantizer.rvq_rest.vq.layers.{i}.table"), codes, i + 1, nq, t, dim);
        for j in 0..t * dim {
            acc[j] += gi[j];
        }
    }
    let rest = matmul(&acc, &w["quantizer.rvq_rest.output_proj.weight"], t, dim, lat);
    let mut quant_tm = first; // [t,lat]
    for j in 0..t * lat {
        quant_tm[j] += rest[j];
    }
    let h_ncl = transpose(&quant_tm, t, lat); // [lat,t]
    let pre = full_causal_conv(w, "pre_conv", &h_ncl, lat, latent, 3, 1, 1, t); // [latent,t]

    // transformer.
    let mut x = transpose(&pre, latent, t); // [t,latent]
    let ipw = &w["pre_transformer.input_proj.weight"];
    let ipb = &w["pre_transformer.input_proj.bias"];
    let mut xx = matmul(&x, ipw, t, latent, hidden);
    add_bias(&mut xx, ipb, t, hidden);
    x = xx; // [t,hidden]

    let hd = cfg.head_dim as usize;
    let nh = cfg.num_attention_heads as usize;
    let nkv = cfg.num_key_value_heads as usize;
    let ff = cfg.intermediate_size as usize;
    let hq = nh * hd;
    let hkv = nkv * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let theta = cfg.rope_theta;
    let eps = cfg.rms_norm_eps;
    assert_eq!(nh, nkv, "decode_stream assumes no GQA in codec transformer");

    for layer in 0..cfg.num_hidden_layers as usize {
        let p = |leaf: &str| format!("pre_transformer.layers.{layer}.{leaf}");
        let xn = hostmath::rmsnorm_rows(&x, &w[&p("input_layernorm.weight")], t, hidden, eps);
        let mut q = matmul(&xn, &w[&p("self_attn.q_proj.weight")], t, hidden, hq);
        let mut k = matmul(&xn, &w[&p("self_attn.k_proj.weight")], t, hidden, hkv);
        let v = matmul(&xn, &w[&p("self_attn.v_proj.weight")], t, hidden, hkv);
        hostmath::rope_neox(&mut q, t, nh, hd, 0, theta);
        hostmath::rope_neox(&mut k, t, nkv, hd, 0, theta);
        // attention per head (full causal).
        let mut ctx = vec![0.0f32; t * hq];
        for h in 0..nh {
            for i in 0..t {
                let qv = &q[(i * nh + h) * hd..(i * nh + h) * hd + hd];
                let mut scores = vec![f32::NEG_INFINITY; i + 1];
                for (j, sc) in scores.iter_mut().enumerate() {
                    let kv = &k[(j * nkv + h) * hd..(j * nkv + h) * hd + hd];
                    *sc = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - mx).exp();
                    sum += *s;
                }
                let dst = &mut ctx[(i * nh + h) * hd..(i * nh + h) * hd + hd];
                for (j, &pj) in scores.iter().enumerate() {
                    let p = pj / sum;
                    let vv = &v[(j * nkv + h) * hd..(j * nkv + h) * hd + hd];
                    for d in 0..hd {
                        dst[d] += p * vv[d];
                    }
                }
            }
        }
        let mut attn = matmul(&ctx, &w[&p("self_attn.o_proj.weight")], t, hq, hidden);
        scale_last(&mut attn, &w[&p("self_attn_layer_scale.scale")], t, hidden);
        for j in 0..t * hidden {
            x[j] += attn[j];
        }
        // MLP (SwiGLU).
        let xn = hostmath::rmsnorm_rows(&x, &w[&p("post_attention_layernorm.weight")], t, hidden, eps);
        let gate = matmul(&xn, &w[&p("mlp.gate_proj.weight")], t, hidden, ff);
        let up = matmul(&xn, &w[&p("mlp.up_proj.weight")], t, hidden, ff);
        let mut hmul = vec![0.0f32; t * ff];
        for j in 0..t * ff {
            let g = gate[j];
            hmul[j] = (g / (1.0 + (-g).exp())) * up[j];
        }
        let mut down = matmul(&hmul, &w[&p("mlp.down_proj.weight")], t, ff, hidden);
        scale_last(&mut down, &w[&p("mlp_layer_scale.scale")], t, hidden);
        for j in 0..t * hidden {
            x[j] += down[j];
        }
    }
    let x = hostmath::rmsnorm_rows(&x, &w["pre_transformer.norm.weight"], t, hidden, eps);
    let opw = &w["pre_transformer.output_proj.weight"];
    let opb = &w["pre_transformer.output_proj.bias"];
    let mut out = matmul(&x, opw, t, hidden, latent);
    add_bias(&mut out, opb, t, latent);
    transpose(&out, t, latent) // [latent, t] NCL
}

/// Stateful streaming codec decoder (pure CPU).
pub struct StreamingCodecDecoder {
    w: W,
    cfg: CodecConfig,
}

impl StreamingCodecDecoder {
    /// Load from a brain codec checkpoint (role "" decoder tensors).
    pub fn load(weights_path: &str) -> StreamingCodecDecoder {
        let c = checkpoint::load(weights_path);
        let cfg = CodecConfig::from_json(&c.header["config"]);
        let w: W = c.by_role("");
        StreamingCodecDecoder { w, cfg }
    }

    pub fn from_parts(w: W, cfg: CodecConfig) -> StreamingCodecDecoder {
        StreamingCodecDecoder { w, cfg }
    }

    pub fn cfg(&self) -> &CodecConfig {
        &self.cfg
    }

    /// Like [`Self::decode_streaming`] but emits each chunk's new audio via
    /// `on_audio(samples, seq)` as it is decoded — the streaming entry point for
    /// the server (front computed once; back carries conv state, so each chunk
    /// decodes only its new frames — no warmup re-decode).
    pub fn decode_streaming_cb(&self, codes_rowmajor: &[u32], chunk: usize, on_audio: &mut dyn FnMut(&[f32], u32)) {
        let nq = self.cfg.num_quantizers as usize;
        let t = codes_rowmajor.len() / nq;
        if t == 0 {
            return;
        }
        let mut codes = vec![0i64; nq * t];
        for f in 0..t {
            for q in 0..nq {
                codes[q * t + f] = codes_rowmajor[f * nq + q] as i64;
            }
        }
        let latent_dim = self.cfg.latent_dim as usize;
        let latent = front(&self.w, &self.cfg, &codes, t);
        let chunk = if chunk == 0 { t } else { chunk.max(1) };
        let mut back = Back::new(&self.w, &self.cfg);
        let mut seq = 0u32;
        let mut a = 0usize;
        while a < t {
            let b = (a + chunk).min(t);
            let l_new = b - a;
            let mut slab = vec![0.0f32; latent_dim * l_new];
            for c in 0..latent_dim {
                slab[c * l_new..c * l_new + l_new].copy_from_slice(&latent[c * t + a..c * t + b]);
            }
            let audio = back.step(&slab, l_new);
            on_audio(&audio, seq);
            seq += 1;
            a = b;
        }
    }

    /// Decode `[T,16]` row-major codes to a 24 kHz waveform, processing `chunk`
    /// frames at a time (carried state). `chunk = 0` => whole clip at once.
    pub fn decode_streaming(&self, codes_rowmajor: &[u32], chunk: usize) -> Vec<f32> {
        let nq = self.cfg.num_quantizers as usize;
        let t = codes_rowmajor.len() / nq;
        // row-major [T,nq] -> codebook-major [nq,T] i64.
        let mut codes = vec![0i64; nq * t];
        for f in 0..t {
            for q in 0..nq {
                codes[q * t + f] = codes_rowmajor[f * nq + q] as i64;
            }
        }
        // Front once (causal): latent NCL [latent, T].
        let latent_dim = self.cfg.latent_dim as usize;
        let latent = front(&self.w, &self.cfg, &codes, t);

        let chunk = if chunk == 0 { t } else { chunk.max(1) };
        let mut back = Back::new(&self.w, &self.cfg);
        let mut wav = Vec::new();
        let mut a = 0usize;
        while a < t {
            let b = (a + chunk).min(t);
            let l_new = b - a;
            // slice latent columns [a,b) per channel -> [latent, l_new] NCL.
            let mut slab = vec![0.0f32; latent_dim * l_new];
            for c in 0..latent_dim {
                slab[c * l_new..c * l_new + l_new].copy_from_slice(&latent[c * t + a..c * t + b]);
            }
            wav.extend_from_slice(&back.step(&slab, l_new));
            a = b;
        }
        wav
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use data::rng::Lcg;

    /// Build a tiny but structurally-complete codec config + random weights, and
    /// check the streaming back: chunked == one-shot, sample-exact.
    #[test]
    fn back_chunked_equals_full() {
        let mut seed = Lcg::new(7);
        let mut cfg = CodecConfig::default();
        cfg.latent_dim = 4;
        cfg.decoder_dim = 8;
        cfg.upsampling_ratios = vec![2];
        cfg.upsample_rates = vec![2, 2];

        let mut w: W = HashMap::new();
        let fill = |w: &mut W, name: &str, n: usize, seed: &mut Lcg| {
            w.insert(name.to_string(), seed.vec(n));
        };
        let latent = 4usize;
        let dec = 8usize;
        // upsample.0.0 convtr [latent,latent,2]; .1 convnext.
        fill(&mut w, "upsample.0.0.conv.weight", latent * latent * 2, &mut seed);
        fill(&mut w, "upsample.0.0.conv.bias", latent, &mut seed);
        fill(&mut w, "upsample.0.1.dwconv.conv.weight", latent * 7, &mut seed);
        fill(&mut w, "upsample.0.1.dwconv.conv.bias", latent, &mut seed);
        fill(&mut w, "upsample.0.1.norm.weight", latent, &mut seed);
        fill(&mut w, "upsample.0.1.norm.bias", latent, &mut seed);
        fill(&mut w, "upsample.0.1.pwconv1.weight", 4 * latent * latent, &mut seed);
        fill(&mut w, "upsample.0.1.pwconv1.bias", 4 * latent, &mut seed);
        fill(&mut w, "upsample.0.1.pwconv2.weight", latent * 4 * latent, &mut seed);
        fill(&mut w, "upsample.0.1.pwconv2.bias", latent, &mut seed);
        fill(&mut w, "upsample.0.1.gamma", latent, &mut seed);
        // decoder.0 conv [dec,latent,7].
        fill(&mut w, "decoder.0.conv.weight", dec * latent * 7, &mut seed);
        fill(&mut w, "decoder.0.conv.bias", dec, &mut seed);
        for (i, &rate) in [2u32, 2].iter().enumerate() {
            let in_dim = dec >> i;
            let out_dim = dec >> (i + 1);
            let bp = format!("decoder.{}", i + 1);
            fill(&mut w, &format!("{bp}.block.0.alpha"), in_dim, &mut seed);
            fill(&mut w, &format!("{bp}.block.0.beta"), in_dim, &mut seed);
            fill(&mut w, &format!("{bp}.block.1.conv.weight"), in_dim * out_dim * (2 * rate as usize), &mut seed);
            fill(&mut w, &format!("{bp}.block.1.conv.bias"), out_dim, &mut seed);
            for j in [2usize, 3, 4] {
                fill(&mut w, &format!("{bp}.block.{j}.act1.alpha"), out_dim, &mut seed);
                fill(&mut w, &format!("{bp}.block.{j}.act1.beta"), out_dim, &mut seed);
                fill(&mut w, &format!("{bp}.block.{j}.conv1.conv.weight"), out_dim * out_dim * 7, &mut seed);
                fill(&mut w, &format!("{bp}.block.{j}.conv1.conv.bias"), out_dim, &mut seed);
                fill(&mut w, &format!("{bp}.block.{j}.act2.alpha"), out_dim, &mut seed);
                fill(&mut w, &format!("{bp}.block.{j}.act2.beta"), out_dim, &mut seed);
                fill(&mut w, &format!("{bp}.block.{j}.conv2.conv.weight"), out_dim * out_dim, &mut seed);
                fill(&mut w, &format!("{bp}.block.{j}.conv2.conv.bias"), out_dim, &mut seed);
            }
        }
        let out_dim = dec >> 2;
        fill(&mut w, "decoder.5.alpha", out_dim, &mut seed);
        fill(&mut w, "decoder.5.beta", out_dim, &mut seed);
        fill(&mut w, "decoder.6.conv.weight", out_dim * 7, &mut seed);
        fill(&mut w, "decoder.6.conv.bias", 1, &mut seed);

        let t = 12usize;
        let lat: Vec<f32> = seed.vec(latent * t);

        let mut full = Back::new(&w, &cfg);
        let y_full = full.step(&lat, t);

        // chunked 5 + 4 + 3.
        let mut bk = Back::new(&w, &cfg);
        let sizes = [5usize, 4, 3];
        let mut y_chunk = Vec::new();
        let mut a = 0;
        for &li in &sizes {
            let mut slab = vec![0.0f32; latent * li];
            for c in 0..latent {
                slab[c * li..c * li + li].copy_from_slice(&lat[c * t + a..c * t + a + li]);
            }
            y_chunk.extend_from_slice(&bk.step(&slab, li));
            a += li;
        }
        assert_eq!(y_full.len(), y_chunk.len(), "length mismatch");
        let maxd = y_full.iter().zip(&y_chunk).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(maxd < 1e-6, "streaming back chunked != full: {maxd}");
    }

    /// End-to-end parity: the pure-CPU streaming decoder vs the real gpu_core
    /// `Codec::decode`, on real weights. Run explicitly:
    ///   BRAIN_CODEC_WEIGHTS=out/tts-1b7/codec.safetensors \
    ///   cargo test -p brain-codec --lib e2e_parity_vs_codec -- --ignored --nocapture
    #[test]
    #[ignore]
    fn e2e_parity_vs_codec() {
        let path = std::env::var("BRAIN_CODEC_WEIGHTS").expect("set BRAIN_CODEC_WEIGHTS");
        let dec = StreamingCodecDecoder::load(&path);
        let nq = dec.cfg.num_quantizers as usize;
        let t = 16usize; // small T (< sliding_window) so attention is plain causal
        let mut seed = Lcg::new(3);
        let codes: Vec<u32> = (0..t * nq).map(|_| seed.next_u32() % 256).collect();

        let full = dec.decode_streaming(&codes, 0); // whole clip
        let stream = dec.decode_streaming(&codes, 4); // 4-frame chunks
        assert_eq!(full.len(), stream.len());
        let d_stream = full.iter().zip(&stream).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("streaming vs full max-abs = {d_stream:.3e} ({} samples)", full.len());
        assert!(d_stream < 1e-6, "streaming != full: {d_stream}");

        let codec = crate::model::Codec::load_inference(&path);
        let reference = codec.decode(&codes);
        let n = full.len().min(reference.len());
        let d_ref = full[..n].iter().zip(&reference[..n]).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        eprintln!("CPU-stream vs gpu_core Codec::decode max-abs = {d_ref:.3e} (len {} vs {})", full.len(), reference.len());
        assert_eq!(full.len(), reference.len(), "length mismatch vs Codec::decode");
        // ~2.4e-3 max-abs on [-1,1] audio: fp reduction-order + erf-approx
        // differences vs the gpu_core path, not an algorithmic mismatch.
        assert!(d_ref < 5e-3, "parity vs Codec::decode too large: {d_ref}");
    }

    /// Wall-clock of the rayon decoder. Run in RELEASE:
    ///   BRAIN_CODEC_WEIGHTS=.../codec.safetensors \
    ///   cargo test --release -p brain-codec --lib bench_decode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_decode() {
        let path = std::env::var("BRAIN_CODEC_WEIGHTS").expect("set BRAIN_CODEC_WEIGHTS");
        let dec = StreamingCodecDecoder::load(&path);
        let nq = dec.cfg().num_quantizers as usize;
        let t = 48usize;
        let mut seed = Lcg::new(5);
        let codes: Vec<u32> = (0..t * nq).map(|_| seed.next_u32() % 256).collect();
        let t0 = std::time::Instant::now();
        let wav = dec.decode_streaming(&codes, 16);
        let dt = t0.elapsed().as_secs_f64();
        let audio_s = wav.len() as f64 / 24000.0;
        eprintln!(
            "decode {t} frames -> {} samples ({audio_s:.2}s audio) in {dt:.2}s  =>  {:.2}x realtime",
            wav.len(),
            audio_s / dt
        );
    }
}
