// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build the Qwen3-TTS 12 Hz **codec decoder** as an ONNX graph (fixed code
//! length `T`), mirroring `codec::Codec::decode` op-for-op with standard ONNX
//! ops plus the new `ConvTranspose` (see [`onnx::conv`]). Pure Rust — no NPU /
//! runtime needed to *produce* the graph.
//!
//! Pipeline (same as the brain decoder):
//!   1. RVQ dequant: `Gather` each codebook's embedding, sum the acoustic group,
//!      `MatMul` each group's `output_proj`, `Add` -> latent `[1,512,T]`.
//!   2. `pre_conv` causal `Conv` 512->1024 (k3).
//!   3. 8-layer causal transformer (RMSNorm, MHA, half-split RoPE θ=1e4, SwiGLU,
//!      per-channel LayerScale) — reusing the same op vocabulary as the Qwen
//!      topology.
//!   4. `upsample.{0,1}`: causal `ConvTranspose` (×ratio) + ConvNeXt block
//!      (depthwise `Conv`, `LayerNorm` from primitives, exact-erf `GELU` via
//!      `Erf`, γ scale).
//!   5. SEANet decoder: head `Conv`, 4 blocks (`SnakeBeta` + causal
//!      `ConvTranspose` + 3 dilated residual units), tail `SnakeBeta` + `Conv`.
//!   6. `Clip` to [-1,1].
//!
//! **SnakeBeta** is `x + (1/(exp(β)+ε))·sin(exp(α)·x)²`, assembled from
//! `Exp/Mul/Sin/Add/Reciprocal`. Conv weights use brain's `[Cout,Cin/g,K]` layout
//! (== ONNX `Conv`); transposed-conv weights use `[Cin,Cout/g,K]` (== ONNX
//! `ConvTranspose`); `nn.Linear` weights are `[out,in]`, transposed once to
//! `[in,out]` for `MatMul`. Layouts: conv stages NCL `[1,C,L]`, transformer /
//! pointwise token-major `[1,L,C]`, flipped with `Transpose`.

use std::collections::HashMap;

use codec::CodecConfig;
use onnx::builder::GraphBuilder;
use onnx::conv::ConvTranspose1d;
use onnx::graph::Node;

type W = HashMap<String, Vec<f32>>;

/// Assemble the codec decoder graph into `g`. `w` holds the brain checkpoint's
/// decoder tensors (role ""); `t` is the fixed number of code frames. Input
/// `codes:[nq,T]` (int64, codebook-major), output `waveform:[1,1,L]` (f32).
pub fn build_codec_graph(cfg: &CodecConfig, w: &W, t: usize, g: &mut GraphBuilder) {
    let mut tp = CodecTopo { b: crate::topo::TopoBase::new(g), w, cfg, stream: false, bufs: Vec::new() };
    tp.consts();
    tp.g.input_i64("codes", &[cfg.num_quantizers as i64, t as i64]);
    let h = tp.front(t);
    let (wav, l) = tp.back(&h, t);
    tp.node("Identity", &[&wav], "waveform");
    tp.g.output_f32("waveform", &[1, 1, l as i64]);
}

/// Front-only graph: `codes:[nq,T]` -> `latent:[1,latent,T]` (RVQ + pre_conv +
/// causal transformer). The front is causal, so in streaming decode it is run
/// once over all frames and the back consumes slices of its output.
pub fn build_codec_front_graph(cfg: &CodecConfig, w: &W, t: usize, g: &mut GraphBuilder) {
    let mut tp = CodecTopo { b: crate::topo::TopoBase::new(g), w, cfg, stream: false, bufs: Vec::new() };
    tp.consts();
    tp.g.input_i64("codes", &[cfg.num_quantizers as i64, t as i64]);
    let h = tp.front(t);
    tp.node("Identity", &[&h], "latent");
    tp.g.output_f32("latent", &[1, cfg.latent_dim as i64, t as i64]);
}

/// Streaming-back graph: `latent:[1,latent,chunk]` + per-conv `bufin.{prefix}`
/// -> `waveform:[1,1,chunk*R]` + per-conv `bufout.{prefix}`. Each causal
/// (transposed-)conv carries its left-context / overlap as graph I/O, so a chunk
/// decodes only its new frames. Returns the buffer specs `(prefix, C, width)`.
pub fn build_codec_back_stream_graph(cfg: &CodecConfig, w: &W, chunk: usize, g: &mut GraphBuilder) -> Vec<(String, i64, i64)> {
    let mut tp = CodecTopo { b: crate::topo::TopoBase::new(g), w, cfg, stream: true, bufs: Vec::new() };
    tp.consts();
    tp.g.input_f32("latent", &[1, cfg.latent_dim as i64, chunk as i64]);
    let (wav, l) = tp.back("latent", chunk);
    tp.node("Identity", &[&wav], "waveform");
    tp.g.output_f32("waveform", &[1, 1, l as i64]);
    tp.bufs
}

struct CodecTopo<'a> {
    b: crate::topo::TopoBase<'a>,
    w: &'a W,
    cfg: &'a CodecConfig,
    /// When true, the back's causal (transposed-)convs carry per-module state via
    /// `bufin.{prefix}` / `bufout.{prefix}` graph I/O (streaming decode).
    stream: bool,
    /// Streaming buffer specs `(prefix, channels, width)` for the host.
    bufs: Vec<(String, i64, i64)>,
}

// Identical DSL helpers live on `TopoBase` (crate::topo); this file keeps only
// its dialect-specific helpers (tagged unary/binary, weight-registering matmul)
// and codec-specific emission.
impl<'a> std::ops::Deref for CodecTopo<'a> {
    type Target = crate::topo::TopoBase<'a>;
    fn deref(&self) -> &Self::Target { &self.b }
}
impl<'a> std::ops::DerefMut for CodecTopo<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.b }
}

impl<'a> CodecTopo<'a> {
    fn unary(&mut self, op: &str, x: &str, tag: &str) -> String {
        let o = self.tmp(tag);
        self.node(op, &[x], &o);
        o
    }
    fn binary(&mut self, op: &str, a: &str, b: &str, tag: &str) -> String {
        let o = self.tmp(tag);
        self.node(op, &[a, b], &o);
        o
    }
    fn reshape_to(&mut self, x: &str, shape: &[i64]) -> String {
        let sname = format!("shape_{}", self.n + 1);
        self.g.init_i64(&sname, &[shape.len() as i64], shape.to_vec());
        self.binary("Reshape", x, &sname, "rs")
    }

    /// Shared scalar constants used across the front and back.
    fn consts(&mut self) {
        let cfg = self.cfg;
        self.f32("rms_eps", &[1], vec![cfg.rms_norm_eps]);
        self.f32("ln_eps", &[1], vec![1e-6]);
        self.f32("snake_eps", &[1], vec![1e-9]);
        self.f32("c_half", &[1], vec![0.5]);
        self.f32("c_one", &[1], vec![1.0]);
        self.f32("c_isqrt2", &[1], vec![std::f32::consts::FRAC_1_SQRT_2]);
        self.f32("c_attn_scale", &[1], vec![1.0 / (cfg.head_dim as f32).sqrt()]);
        self.f32("clip_lo", &[1], vec![-1.0]);
        self.f32("clip_hi", &[1], vec![1.0]);
    }

    /// Front (causal): `codes:[nq,T]` -> latent NCL `[1,latent,T]`.
    fn front(&mut self, t: usize) -> String {
        let nq = self.cfg.num_quantizers as usize;
        let dim = (self.cfg.codebook_dim / 2) as usize;
        let lat = self.cfg.codebook_dim as usize;
        let hidden = self.cfg.hidden_size as usize;
        let latent = self.cfg.latent_dim as usize;
        let ti = t as i64;

        let sem = self.gather_codebook("quantizer.rvq_first.vq.layers.0.table", 0, t, dim);
        let first = self.matmul(&sem, "quantizer.rvq_first.output_proj.weight", lat, dim);
        let mut acc = self.gather_codebook("quantizer.rvq_rest.vq.layers.0.table", 1, t, dim);
        for i in 1..(nq - 1) {
            let gi = self.gather_codebook(&format!("quantizer.rvq_rest.vq.layers.{i}.table"), i + 1, t, dim);
            acc = self.add(&acc, &gi);
        }
        let rest = self.matmul(&acc, "quantizer.rvq_rest.output_proj.weight", lat, dim);
        let quant_tm = self.add(&first, &rest);
        let quant_tm = self.reshape_to(&quant_tm, &[1, ti, lat as i64]);
        let h_ncl = self.transpose(&quant_tm, &[0, 2, 1]); // [1,lat,T]
        let h_ncl = self.causal_conv(&h_ncl, "pre_conv", lat, latent, 3, 1, 1, t);
        let mut x = self.transpose(&h_ncl, &[0, 2, 1]); // [1,T,latent]
        x = self.linear_bias(&x, "pre_transformer.input_proj", latent, hidden);
        x = self.transformer(&x, t);
        x = self.linear_bias(&x, "pre_transformer.output_proj", hidden, latent);
        self.transpose(&x, &[0, 2, 1]) // [1,latent,T]
    }

    /// Back: latent NCL `[1,latent,l0]` -> `(waveform, L)`. In `self.stream` mode
    /// every causal (transposed-)conv carries its state via graph buffers.
    fn back(&mut self, h_in: &str, l0: usize) -> (String, usize) {
        let latent = self.cfg.latent_dim as usize;
        let dec_dim = self.cfg.decoder_dim as usize;
        let mut l = l0;
        let mut h = h_in.to_string();

        let ratios = self.cfg.upsampling_ratios.clone();
        for (u, &factor) in ratios.iter().enumerate() {
            let f = factor as usize;
            h = self.causal_convtr(&h, &format!("upsample.{u}.0"), latent, latent, f, f, l);
            l *= f;
            h = self.convnext(&h, &format!("upsample.{u}.1"), latent, l);
        }

        h = self.causal_conv(&h, "decoder.0", latent, dec_dim, 7, 1, 1, l);
        let rates = self.cfg.upsample_rates.clone();
        for (i, &rate) in rates.iter().enumerate() {
            let in_dim = dec_dim >> i;
            let out_dim = dec_dim >> (i + 1);
            let bp = format!("decoder.{}", i + 1);
            h = self.snake(&h, &format!("{bp}.block.0"), in_dim);
            h = self.causal_convtr(&h, &format!("{bp}.block.1"), in_dim, out_dim, 2 * rate as usize, rate as usize, l);
            l *= rate as usize;
            for (j, dil) in [(2usize, 1usize), (3, 3), (4, 9)] {
                h = self.residual_unit(&h, &format!("{bp}.block.{j}"), out_dim, dil, l);
            }
        }
        let out_dim = dec_dim >> rates.len();
        h = self.snake(&h, "decoder.5", out_dim);
        h = self.causal_conv(&h, "decoder.6", out_dim, 1, 7, 1, 1, l);
        let wav = self.clip(&h, "clip_lo", "clip_hi");
        (wav, l)
    }

    /// Slice NCL `[1,C,L]` along the length axis -> `[1,C,end-start]`.
    fn slice_ncl(&mut self, x: &str, start: i64, end: i64) -> String {
        let s = format!("sl_s_{}", self.n + 1);
        let e = format!("sl_e_{}", self.n + 1);
        let a = "axis2_const".to_string();
        if !self.has(&a) {
            self.g.init_i64(&a, &[1], vec![2]);
        }
        self.g.init_i64(&s, &[1], vec![start]);
        self.g.init_i64(&e, &[1], vec![end]);
        let o = self.tmp("slc");
        self.g.add(Node::new("Slice", &[x, &s, &e, &a], &[&o]));
        o
    }

    /// Concat two NCL `[1,C,*]` tensors along the length axis.
    fn concat_ncl(&mut self, a: &str, b: &str) -> String {
        let o = self.tmp("cat");
        self.g.add(Node::new("Concat", &[a, b], &[&o]).attr_int("axis", 2));
        o
    }

    /// NCL bias add: `[1,C,L] + bias[C]` broadcast as `[1,C,1]`.
    fn add_ncl_bias(&mut self, x: &str, bias_name: &str, c: usize) -> String {
        let bn = format!("{bias_name}.ncl");
        if !self.has(&bn) {
            let wv = self.w[bias_name].clone();
            self.f32(&bn, &[1, c as i64, 1], wv);
        }
        self.add(x, &bn)
    }

    /// Gather codebook `col`'s rows: `idx = codes[col,:]` -> `table[idx]` -> `[T,dim]`.
    fn gather_codebook(&mut self, table: &str, col: usize, t: usize, dim: usize) -> String {
        // table init [bins, dim].
        if !self.has(table) {
            let data = self.w[table].clone();
            let bins = data.len() / dim;
            self.f32(table, &[bins as i64, dim as i64], data);
        }
        // scalar index `col` -> Gather(codes, col, axis=0) drops axis -> [T].
        let cidx = format!("codecol_{col}");
        if !self.has(&cidx) {
            self.g.init_i64(&cidx, &[], vec![col as i64]); // 0-D scalar
        }
        let idx = self.tmp("cidx");
        self.g.add(Node::new("Gather", &["codes", &cidx], &[&idx]).attr_int("axis", 0));
        let emb = self.tmp("emb");
        self.g.add(Node::new("Gather", &[table, &idx], &[&emb]).attr_int("axis", 0));
        let _ = t;
        emb // [T, dim]
    }

    /// `y = x · Wᵀ` (no bias). brain weight `name` is `[out,in]` -> `[in,out]`.
    fn matmul(&mut self, x: &str, name: &str, out: usize, inp: usize) -> String {
        let wt = format!("{name}.wt");
        if !self.has(&wt) {
            let t = transpose(&self.w[name], out, inp);
            self.f32(&wt, &[inp as i64, out as i64], t);
        }
        self.binary("MatMul", x, &wt, "mm")
    }

    /// `nn.Linear` with bias: `MatMul` then broadcast-`Add` of `{prefix}.bias`.
    fn linear_bias(&mut self, x: &str, prefix: &str, inp: usize, out: usize) -> String {
        let y = self.matmul(x, &format!("{prefix}.weight"), out, inp);
        let bname = format!("{prefix}.bias");
        if !self.has(&bname) {
            let wv = self.w[&bname].clone();
            self.f32(&bname, &[out as i64], wv);
        }
        self.add(&y, &bname)
    }

    /// RMSNorm over the last axis (`dim`) with gain `name`.
    fn rmsnorm(&mut self, x: &str, name: &str, dim: usize) -> String {
        let gain = format!("{name}.g");
        let gw = self.w[name].clone();
        self.b.rmsnorm(x, &gain, gw, dim, "rms_eps")
    }

    /// Per-channel (last-axis) scale by initializer `name` (LayerScale / γ).
    fn scale_last(&mut self, x: &str, name: &str, c: usize) -> String {
        if !self.has(name) {
            let wv = self.w[name].clone();
            self.f32(name, &[c as i64], wv);
        }
        self.mul(x, name)
    }

    /// Causal `Conv1d` in NCL: left pad `dilation·(K-1)`, stride 1. brain weight
    /// `[Cout,Cin/g,K]` == ONNX `Conv`. Returns `[1,Cout,L]`.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv(&mut self, x: &str, prefix: &str, cin: usize, cout: usize, k: usize, dil: usize, groups: usize, l: usize) -> String {
        let wname = format!("{prefix}.conv.weight");
        if !self.has(&wname) {
            let wv = self.w[&wname].clone();
            self.f32(&wname, &[cout as i64, (cin / groups) as i64, k as i64], wv);
        }
        let bname = format!("{prefix}.conv.bias");
        if !self.has(&bname) {
            let wv = self.w[&bname].clone();
            self.f32(&bname, &[cout as i64], wv);
        }
        let ctx = dil * (k - 1);
        if self.stream && ctx > 0 {
            // Prepend the carried left-context buffer, conv with no left pad, then
            // emit the last `ctx` input columns as the next chunk's buffer.
            let bin = format!("bufin.{prefix}");
            self.g.input_f32(&bin, &[1, cin as i64, ctx as i64]);
            self.bufs.push((prefix.to_string(), cin as i64, ctx as i64));
            let xin = self.concat_ncl(&bin, x); // [1,cin,ctx+l]
            let o = self.tmp("conv");
            self.g.add(
                Node::new("Conv", &[&xin, &wname, &bname], &[&o])
                    .name(prefix)
                    .attr_ints("kernel_shape", &[k as i64])
                    .attr_ints("strides", &[1])
                    .attr_ints("pads", &[0, 0])
                    .attr_ints("dilations", &[dil as i64])
                    .attr_int("group", groups as i64),
            );
            let bo = self.slice_ncl(&xin, l as i64, (ctx + l) as i64);
            let bout = format!("bufout.{prefix}");
            self.node("Identity", &[&bo], &bout);
            self.g.output_f32(&bout, &[1, cin as i64, ctx as i64]);
            return o;
        }
        let o = self.tmp("conv");
        self.g.add(
            Node::new("Conv", &[x, &wname, &bname], &[&o])
                .name(prefix)
                .attr_ints("kernel_shape", &[k as i64])
                .attr_ints("strides", &[1])
                .attr_ints("pads", &[ctx as i64, 0])
                .attr_ints("dilations", &[dil as i64])
                .attr_int("group", groups as i64),
        );
        let _ = l;
        o
    }

    /// Causal `ConvTranspose1d` in NCL (upsample by `stride`). brain weight
    /// `[Cin,Cout,K]` == ONNX `ConvTranspose`; `pads=[0,K-stride]` keeps the first
    /// `L·stride` samples. Returns `[1,Cout,L·stride]`.
    fn causal_convtr(&mut self, x: &str, prefix: &str, cin: usize, cout: usize, k: usize, stride: usize, l: usize) -> String {
        let ov = k - stride;
        if self.stream && ov > 0 {
            // Full (untrimmed, unbiased) transposed conv, then add the carried
            // overlap to the head, emit `l*stride` finalized samples (+ bias), and
            // carry the trailing `ov` raw samples.
            let c = ConvTranspose1d { cin, cout, l, k, stride, pad_begin: 0, pad_end: 0, dilation: 1, groups: 1, output_padding: 0 };
            let raw = self.tmp("convtr_raw");
            let weight = self.w[&format!("{prefix}.conv.weight")].clone();
            self.g.conv_transpose1d(prefix, x, &raw, weight, None, &c);
            let raw_len = ((l - 1) * stride + k) as i64;
            let fin = (l * stride) as i64;
            let ovi = ov as i64;
            let bin = format!("bufin.{prefix}");
            self.g.input_f32(&bin, &[1, cout as i64, ovi]);
            self.bufs.push((prefix.to_string(), cout as i64, ovi));
            let first = self.slice_ncl(&raw, 0, ovi);
            let summed = self.add(&first, &bin);
            let rest = self.slice_ncl(&raw, ovi, raw_len);
            let raw2 = self.concat_ncl(&summed, &rest);
            let finsl = self.slice_ncl(&raw2, 0, fin);
            let finb = self.add_ncl_bias(&finsl, &format!("{prefix}.conv.bias"), cout);
            let co = self.slice_ncl(&raw2, fin, raw_len);
            let bout = format!("bufout.{prefix}");
            self.node("Identity", &[&co], &bout);
            self.g.output_f32(&bout, &[1, cout as i64, ovi]);
            return finb;
        }
        let c = ConvTranspose1d { cin, cout, l, k, stride, pad_begin: 0, pad_end: k - stride, dilation: 1, groups: 1, output_padding: 0 };
        let out = self.tmp("convtr");
        let weight = self.w[&format!("{prefix}.conv.weight")].clone();
        let bias = self.w[&format!("{prefix}.conv.bias")].clone();
        self.g.conv_transpose1d(prefix, x, &out, weight, Some(bias), &c);
        out
    }

    /// SnakeBeta over NCL `[1,C,L]`: `x + (1/(exp(β)+ε))·sin(exp(α)·x)²`.
    fn snake(&mut self, x: &str, prefix: &str, c: usize) -> String {
        let aname = format!("{prefix}.alpha");
        let bname = format!("{prefix}.beta");
        if !self.has(&aname) {
            let wv = self.w[&aname].clone();
            self.f32(&aname, &[1, c as i64, 1], wv);
        }
        if !self.has(&bname) {
            let wv = self.w[&bname].clone();
            self.f32(&bname, &[1, c as i64, 1], wv);
        }
        let ea = self.unary("Exp", &aname, "snexp_a");
        let ax = self.mul(x, &ea);
        let sn = self.unary("Sin", &ax, "snsin");
        let sn2 = self.mul(&sn, &sn);
        let eb = self.unary("Exp", &bname, "snexp_b");
        let denom = self.add(&eb, "snake_eps");
        let recip = self.unary("Reciprocal", &denom, "snrecip");
        let term = self.mul(&sn2, &recip);
        self.add(x, &term)
    }

    /// SEANet residual unit: `x + conv2(snake(conv1(snake(x))))`.
    fn residual_unit(&mut self, x: &str, prefix: &str, c: usize, dil: usize, l: usize) -> String {
        let y = self.snake(x, &format!("{prefix}.act1"), c);
        let y = self.causal_conv(&y, &format!("{prefix}.conv1"), c, c, 7, dil, 1, l);
        let y = self.snake(&y, &format!("{prefix}.act2"), c);
        let y = self.causal_conv(&y, &format!("{prefix}.conv2"), c, c, 1, 1, 1, l);
        self.add(x, &y)
    }

    /// ConvNeXt block (NCL in/out): depthwise causal conv -> LayerNorm -> pwconv1
    /// -> exact GELU -> pwconv2 -> γ scale -> residual.
    fn convnext(&mut self, x: &str, prefix: &str, c: usize, l: usize) -> String {
        let dw = self.causal_conv(x, &format!("{prefix}.dwconv"), c, c, 7, 1, c, l); // [1,C,L]
        let tm = self.transpose(&dw, &[0, 2, 1]); // [1,L,C]
        let normed = self.layernorm(&tm, &format!("{prefix}.norm"), c);
        let hid = c * 4;
        let g1 = self.linear_bias(&normed, &format!("{prefix}.pwconv1"), c, hid); // [1,L,4C]
        let g1 = self.gelu(&g1);
        let o = self.linear_bias(&g1, &format!("{prefix}.pwconv2"), hid, c); // [1,L,C]
        let o = self.scale_last(&o, &format!("{prefix}.gamma"), c);
        let o_ncl = self.transpose(&o, &[0, 2, 1]); // [1,C,L]
        self.add(x, &o_ncl)
    }

    /// `nn.LayerNorm` over the last axis (`c`), eps 1e-6, affine — opset-13
    /// primitives (no `LayerNormalization`).
    fn layernorm(&mut self, x: &str, prefix: &str, c: usize) -> String {
        let gn = format!("{prefix}.weight");
        let bn = format!("{prefix}.bias");
        if !self.has(&gn) {
            let wv = self.w[&gn].clone();
            self.f32(&gn, &[c as i64], wv);
        }
        if !self.has(&bn) {
            let wv = self.w[&bn].clone();
            self.f32(&bn, &[c as i64], wv);
        }
        let mean = {
            let o = self.tmp("ln_mean");
            self.g.add(Node::new("ReduceMean", &[x], &[&o]).attr_ints("axes", &[-1]).attr_int("keepdims", 1));
            o
        };
        let xc = self.binary("Sub", x, &mean, "ln_xc");
        let sq = self.mul(&xc, &xc);
        let var = {
            let o = self.tmp("ln_var");
            self.g.add(Node::new("ReduceMean", &[&sq], &[&o]).attr_ints("axes", &[-1]).attr_int("keepdims", 1));
            o
        };
        let vae = self.add(&var, "ln_eps");
        let std = self.unary("Sqrt", &vae, "ln_std");
        let norm = self.binary("Div", &xc, &std, "ln_div");
        let scaled = self.mul(&norm, &gn);
        self.add(&scaled, &bn)
    }

    /// Exact (erf) GELU: `0.5·x·(1 + erf(x/√2))`.
    fn gelu(&mut self, x: &str) -> String {
        let half = self.mul(x, "c_half");
        let inner = self.mul(x, "c_isqrt2");
        let e = self.unary("Erf", &inner, "erf");
        let ep1 = self.add(&e, "c_one");
        self.mul(&half, &ep1)
    }

    /// `Clip(x, lo, hi)`.
    fn clip(&mut self, x: &str, lo: &str, hi: &str) -> String {
        let o = self.tmp("clip");
        self.node("Clip", &[x, lo, hi], &o);
        o
    }

    /// The 8-layer causal transformer over token-major `[1,T,hidden]`.
    fn transformer(&mut self, x0: &str, t: usize) -> String {
        // RoPE cos/sin tables + causal mask, sized for T.
        let hd = self.cfg.head_dim as usize;
        let d = self.cfg.hidden_size as usize;
        let nh = self.cfg.num_attention_heads as usize;
        let nkv = self.cfg.num_key_value_heads as usize;
        let ff = self.cfg.intermediate_size as usize;
        let theta = self.cfg.rope_theta;
        let hq = nh * hd;
        let hkv = nkv * hd;
        let half = hd / 2;
        let ti = t as i64;
        // cos/sin [1,T,1,hd]
        let (mut cos, mut sin) = (vec![0f32; t * hd], vec![0f32; t * hd]);
        for p in 0..t {
            for j in 0..hd {
                let m = (j % half) as f32;
                let ang = p as f32 * theta.powf(-2.0 * m / hd as f32);
                cos[p * hd + j] = ang.cos();
                sin[p * hd + j] = ang.sin();
            }
        }
        self.f32("tf_cos", &[1, ti, 1, hd as i64], cos);
        self.f32("tf_sin", &[1, ti, 1, hd as i64], sin);
        let mut mask = vec![0f32; t * t];
        for i in 0..t {
            for j in 0..t {
                if j > i {
                    mask[i * t + j] = -1.0e9;
                }
            }
        }
        self.f32("tf_mask", &[1, 1, ti, ti], mask);
        // rotate_half slice bounds.
        self.g.init_i64("tf_rh_ax", &[1], vec![3]);
        self.g.init_i64("tf_lo0", &[1], vec![0]);
        self.g.init_i64("tf_hi0", &[1], vec![half as i64]);
        self.g.init_i64("tf_lo1", &[1], vec![half as i64]);
        self.g.init_i64("tf_hi1", &[1], vec![hd as i64]);

        let nlayers = self.cfg.num_hidden_layers as usize;
        let mut x = x0.to_string();
        for layer in 0..nlayers {
            let p = |leaf: &str| format!("pre_transformer.layers.{layer}.{leaf}");
            // --- attention ---
            let xn = self.rmsnorm(&x, &p("input_layernorm.weight"), d);
            let q = self.matmul(&xn, &p("self_attn.q_proj.weight"), hq, d);
            let k = self.matmul(&xn, &p("self_attn.k_proj.weight"), hkv, d);
            let v = self.matmul(&xn, &p("self_attn.v_proj.weight"), hkv, d);
            let q = self.reshape_to(&q, &[1, ti, nh as i64, hd as i64]);
            let k = self.reshape_to(&k, &[1, ti, nkv as i64, hd as i64]);
            let v = self.reshape_to(&v, &[1, ti, nkv as i64, hd as i64]);
            let q = self.rope(&q, half);
            let k = self.rope(&k, half);
            let q = self.transpose(&q, &[0, 2, 1, 3]); // [1,nh,T,hd]
            let k = self.transpose(&k, &[0, 2, 1, 3]);
            let v = self.transpose(&v, &[0, 2, 1, 3]);
            let kt = self.transpose(&k, &[0, 1, 3, 2]); // [1,nh,hd,T]
            let scores = self.binary("MatMul", &q, &kt, "mm");
            let scores = self.mul(&scores, "c_attn_scale");
            let scores = self.add(&scores, "tf_mask");
            let probs = {
                let o = self.tmp("sm");
                self.g.add(Node::new("Softmax", &[&scores], &[&o]).attr_int("axis", -1));
                o
            };
            let ctx = self.binary("MatMul", &probs, &v, "mm"); // [1,nh,T,hd]
            let ctx = self.transpose(&ctx, &[0, 2, 1, 3]); // [1,T,nh,hd]
            let ctx = self.reshape_to(&ctx, &[1, ti, hq as i64]);
            let attn = self.matmul(&ctx, &p("self_attn.o_proj.weight"), d, hq);
            let attn = self.scale_last(&attn, &p("self_attn_layer_scale.scale"), d);
            x = self.add(&x, &attn);
            // --- MLP (SwiGLU) ---
            let xn = self.rmsnorm(&x, &p("post_attention_layernorm.weight"), d);
            let gate = self.matmul(&xn, &p("mlp.gate_proj.weight"), ff, d);
            let up = self.matmul(&xn, &p("mlp.up_proj.weight"), ff, d);
            let sig = self.unary("Sigmoid", &gate, "sig");
            let silu = self.mul(&gate, &sig);
            let hmul = self.mul(&silu, &up);
            let down = self.matmul(&hmul, &p("mlp.down_proj.weight"), d, ff);
            let down = self.scale_last(&down, &p("mlp_layer_scale.scale"), d);
            x = self.add(&x, &down);
        }
        self.rmsnorm(&x, "pre_transformer.norm.weight", d)
    }

    /// Half-split RoPE on `[1,T,heads,hd]`: `x·cos + rotate_half(x)·sin`.
    fn rope(&mut self, x: &str, _half: usize) -> String {
        let x2 = {
            let o = self.tmp("rh_x2");
            self.g.add(Node::new("Slice", &[x, "tf_lo1", "tf_hi1", "tf_rh_ax"], &[&o]));
            o
        };
        let x1 = {
            let o = self.tmp("rh_x1");
            self.g.add(Node::new("Slice", &[x, "tf_lo0", "tf_hi0", "tf_rh_ax"], &[&o]));
            o
        };
        let nx2 = self.unary("Neg", &x2, "neg");
        let rot = {
            let o = self.tmp("rh");
            self.g.add(Node::new("Concat", &[&nx2, &x1], &[&o]).attr_int("axis", 3));
            o
        };
        let a = self.mul(x, "tf_cos");
        let b = self.mul(&rot, "tf_sin");
        self.add(&a, &b)
    }
}

/// Transpose a row-major `[rows,cols]` to `[cols,rows]`.
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}
