// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS 12 Hz codec — decode path (codes `[T,16]` -> 24 kHz waveform).
//!
//! Forward only; pure fp32 over the shared WGSL engine. The graph follows
//! `Qwen3TTSTokenizerV2Decoder.forward`:
//!
//! 1. `quantizer.decode` — SplitResidualVectorQuantizer: gather the semantic
//!    codebook (q0) and the 15 acoustic codebooks (q1..15) via `embed`, sum each
//!    group, run each group's `output_proj` (1×1 conv == matmul), and add the two
//!    -> latent `[512, T]`.
//! 2. `pre_conv` — causal `conv1d` 512->1024, k3.
//! 3. `pre_transformer` — 8-layer causal MHA (head_dim 64, 16 heads, half-split
//!    RoPE θ=1e4, RMSNorm, SiLU MLP, per-channel LayerScale on each residual),
//!    with `input_proj`/`output_proj` around it.
//! 4. `upsample.{0,1}` — causal transposed conv (×2) + ConvNeXt block.
//! 5. `decoder.{0..6}` — SEANet: head conv, 4 `DecoderBlock`s (SnakeBeta + causal
//!    transposed conv + 3 dilated residual units), tail SnakeBeta + conv to 1ch.
//! 6. clamp to [-1, 1].
//!
//! Layout convention: conv stages run in channel-major NCL `[C, L]`; the
//! transformer and the pointwise linears in ConvNeXt run token-major `[L, C]`.
//! The few layout flips are explicit host transposes (no transpose kernel
//! exists). ConvNeXt's `LayerNorm` (eps 1e-6) and exact-erf `GELU` are computed
//! on the host for bit-faithfulness (the device kernels use eps 1e-5 / the tanh
//! GELU approximation); everything else is on-device.

use std::collections::HashMap;

use bytemuck::cast_slice;
use gpu_core::{f, BufUsage, DeviceBuffer, Gpu};
use model::block::{self, Gqa, KernelIds};
use paramstore::{ParamStore, Role};

use crate::config::CodecConfig;

// ---- kernel indices (order matches PIPELINES) ----
const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const ROPE: usize = 3;
const GQA_SCORES: usize = 4;
const ATTN_SOFTMAX: usize = 5;
const GQA_APPLY: usize = 6;
const SILU_MUL: usize = 7;
const ADD2: usize = 8;
const BIAS_ADD: usize = 9;
const CONV1D: usize = 10;
const CONVTR1D: usize = 11;
const SNAKE: usize = 12;
const SCALE_CHAN: usize = 13;

const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rope_base", kernels::ROPE_BASE),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("bias_add", kernels::BIAS_ADD),
    ("conv1d", kernels::CONV1D),
    ("convtr1d", kernels::CONVTR1D),
    ("snake_beta", kernels::SNAKE_BETA),
    ("scale_chan", kernels::SCALE_CHAN),
];

/// SnakeBeta's `no_div_by_zero` (the reference's fixed epsilon).
const SNAKE_EPS: f32 = 1e-9;
/// ConvNeXt LayerNorm epsilon.
const LN_EPS: f32 = 1e-6;

/// The decode model: frozen weights on device + small (≤1-D) tensors mirrored on
/// the host for per-channel NCL bias broadcasts and the host LayerNorm/GELU.
pub struct Codec {
    pub gpu: Gpu,
    pub cfg: CodecConfig,
    ps: ParamStore,
    host: HashMap<String, Vec<f32>>,
}

impl Codec {
    /// Load an inference-only decoder from a brain checkpoint produced by
    /// [`crate::import::import`]. Every parameter is frozen (weights only).
    pub fn load_inference(weights_path: &str) -> Codec {
        let c = checkpoint::load(weights_path);
        let cfg = CodecConfig::from_json(&c.header["config"]);
        let init = c.by_role("");
        Codec::from_weights(cfg, init)
    }

    /// Build from an in-memory weight map (used by tests and [`load_inference`]).
    pub fn from_weights(cfg: CodecConfig, init: HashMap<String, Vec<f32>>) -> Codec {
        let gpu = Gpu::new_cpu(PIPELINES);
        // Mirror small tensors (biases, norms, scales, alphas/betas/gammas) on the
        // host: NCL conv-bias broadcasts and ConvNeXt's host LayerNorm read these.
        let host: HashMap<String, Vec<f32>> = init
            .iter()
            .filter(|(_, v)| v.len() <= 8192)
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let roles: Vec<(String, usize, Role)> =
            init.iter().map(|(n, v)| (n.clone(), v.len(), Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &init);
        Codec { gpu, cfg, ps, host }
    }

    // -- tiny eager helpers (each submits immediately on the CPU backend) --

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }
    fn st(&self, n: usize) -> DeviceBuffer {
        self.gpu.storage(n as u64)
    }
    fn run(&self, step: gpu_core::Step) {
        self.gpu.submit(&[], &[step]);
    }
    fn upload(&self, data: &[f32]) -> DeviceBuffer {
        let b = self.gpu.buffer("act", (data.len() * 4) as u64, BufUsage::STORAGE | BufUsage::COPY_DST);
        self.gpu.write(&b, cast_slice(data));
        b
    }

    /// `out = x[M,K] @ W[N,K]ᵀ` (PyTorch `nn.Linear`, no bias).
    fn matmul(&self, x: &DeviceBuffer, wname: &str, m: u32, k: u32, n: u32) -> DeviceBuffer {
        let out = self.st((m * n) as usize);
        self.run(self.gpu.step(MATMUL, &[x, self.w(wname), &out], &[m, k, n], m * n));
        out
    }
    /// In-place per-output-feature bias for a token-major `[M,N]` buffer.
    fn bias_add(&self, buf: &DeviceBuffer, bias_name: &str, m: u32, n: u32) {
        self.run(self.gpu.step(BIAS_ADD, &[buf, self.w(bias_name)], &[m, n], m * n));
    }
    fn add2(&self, a: &DeviceBuffer, b: &DeviceBuffer, total: u32) -> DeviceBuffer {
        let out = self.st(total as usize);
        self.run(self.gpu.step(ADD2, &[a, b, &out], &[total], total));
        out
    }
    /// SnakeBeta over NCL `[C, L]`: `y = x + (1/(exp(β)+ε))·sin(exp(α)·x)²`.
    fn snake(&self, x: &DeviceBuffer, prefix: &str, c: u32, l: u32) -> DeviceBuffer {
        let out = self.st((c * l) as usize);
        let (a, b) = (format!("{prefix}.alpha"), format!("{prefix}.beta"));
        let total = c * l;
        self.run(self.gpu.step(SNAKE, &[x, self.w(&a), self.w(&b), &out], &[total, c, l, f(SNAKE_EPS)], total));
        out
    }
    /// Per-channel scale over a `[rows, c, inner]` layout: `y = x·scale[c]` where
    /// channel `= (idx / inner) % c` (LayerScale / ConvNeXt γ). `total` is the
    /// full element count (`rows·c·inner`).
    fn scale_chan(&self, x: &DeviceBuffer, scale_name: &str, total: u32, c: u32, inner: u32) -> DeviceBuffer {
        let out = self.st(total as usize);
        self.run(self.gpu.step(SCALE_CHAN, &[x, self.w(scale_name), &out], &[total, c, inner], total));
        out
    }
    /// Transpose a row-major `[a, b]` buffer to `[b, a]` on the host.
    fn transpose(&self, x: &DeviceBuffer, a: u32, b: u32) -> DeviceBuffer {
        let v = self.gpu.read(x, (a * b) as usize);
        let mut o = vec![0.0f32; (a * b) as usize];
        for i in 0..a as usize {
            for j in 0..b as usize {
                o[j * a as usize + i] = v[i * b as usize + j];
            }
        }
        self.upload(&o)
    }

    /// Causal `conv1d` in NCL: left pad `dilation*(K-1)`, stride 1, `Lo == L`,
    /// then add the per-channel bias (`{prefix}.conv.bias`). Returns `[Cout, L]`.
    fn causal_conv(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, dilation: u32, groups: u32) -> DeviceBuffer {
        let pad = dilation * (k - 1);
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride: 1, pad, dilation, groups, lo: l };
        let out = self.st((cout * l) as usize);
        self.run(audio::conv::conv1d_fwd(
            &self.gpu,
            &audio::conv::ConvKernels { fwd: CONV1D, dx: 0, dw: 0 },
            &c,
            x,
            self.w(&format!("{prefix}.conv.weight")),
            &out,
        ));
        self.add_ncl_bias(&out, &format!("{prefix}.conv.bias"), cout, l)
    }

    /// Causal transposed conv in NCL (upsampling by `stride`). The reference crops
    /// `K-stride` samples off the right; cropping the right == keeping the first
    /// `L·stride` outputs, so we just request `Lo = L·stride` (pad 0). Returns
    /// `[Cout, L·stride]`.
    fn causal_convtr(&self, x: &DeviceBuffer, prefix: &str, cin: u32, cout: u32, l: u32, k: u32, stride: u32) -> DeviceBuffer {
        let lo = l * stride;
        let c = audio::conv::Conv1d { n: 1, cin, l, cout, k, stride, pad: 0, dilation: 1, groups: 1, lo };
        let out = self.st((cout * lo) as usize);
        self.run(audio::conv::convtr1d_fwd(
            &self.gpu,
            &audio::conv::ConvKernels { fwd: CONVTR1D, dx: 0, dw: 0 },
            &c,
            x,
            self.w(&format!("{prefix}.conv.weight")),
            &out,
        ));
        self.add_ncl_bias(&out, &format!("{prefix}.conv.bias"), cout, lo)
    }

    /// Add a per-channel bias to an NCL `[C, L]` buffer (channel = idx / L). The
    /// engine's `bias_add` indexes the *inner* axis, so we broadcast the host bias
    /// to `[C, L]` and use `add2`.
    fn add_ncl_bias(&self, x: &DeviceBuffer, bias_name: &str, c: u32, l: u32) -> DeviceBuffer {
        let bias = &self.host[bias_name];
        let mut bcast = vec![0.0f32; (c * l) as usize];
        for ch in 0..c as usize {
            let v = bias[ch];
            for t in 0..l as usize {
                bcast[ch * l as usize + t] = v;
            }
        }
        let bbuf = self.upload(&bcast);
        self.add2(x, &bbuf, c * l)
    }

    // ------------------------------------------------------------------
    // decode
    // ------------------------------------------------------------------

    /// Decode `codes` (`[T,16]` row-major, q0 semantic + q1..15 acoustic) into a
    /// mono 24 kHz waveform (`≈ T·1920` samples).
    pub fn decode(&self, codes: &[u32]) -> Vec<f32> {
        let nq = self.cfg.num_quantizers as usize;
        assert_eq!(codes.len() % nq, 0, "codes length not a multiple of {nq}");
        let t = (codes.len() / nq) as u32;
        assert!(t > 0, "empty codes");
        let dim = self.cfg.codebook_dim / 2; // per-codebook dim (256)
        let lat = self.cfg.codebook_dim; // 512 (quantizer output)
        let hidden = self.cfg.hidden_size; // 512
        let latent = self.cfg.latent_dim; // 1024

        // --- 1. quantizer.decode ---
        let gather = |table: &str, col: usize| -> DeviceBuffer {
            let col_codes: Vec<u32> = (0..t as usize).map(|ti| codes[ti * nq + col]).collect();
            let codes_buf = {
                let b = self.gpu.buffer("codes", (t as u64) * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
                self.gpu.write(&b, &col_codes);
                b
            };
            let out = self.st((t * dim) as usize);
            self.run(self.gpu.step(EMBED, &[&codes_buf, self.w(table), &out], &[dim, t], t * dim));
            out
        };
        // semantic group (rvq_first): 1 codebook.
        let sem = gather("quantizer.rvq_first.vq.layers.0.table", 0);
        let first = self.matmul(&sem, "quantizer.rvq_first.output_proj.weight", t, dim, lat);
        // acoustic group (rvq_rest): codebooks q1..15 summed, then projected.
        let mut acc = gather("quantizer.rvq_rest.vq.layers.0.table", 1);
        for i in 1..(nq - 1) {
            let g = gather(&format!("quantizer.rvq_rest.vq.layers.{i}.table"), i + 1);
            acc = self.add2(&acc, &g, t * dim);
        }
        let rest = self.matmul(&acc, "quantizer.rvq_rest.output_proj.weight", t, dim, lat);
        let quant_tm = self.add2(&first, &rest, t * lat); // [T, 512] token-major
        let quant = self.transpose(&quant_tm, t, lat); // [512, T] NCL

        // --- 2. pre_conv (512 -> 1024, k3 causal) ---
        let pre = self.causal_conv(&quant, "pre_conv", lat, latent, t, 3, 1, 1); // [1024, T]
        let mut x = self.transpose(&pre, latent, t); // [T, 1024] token-major

        // --- 3. pre_transformer ---
        x = self.matmul(&x, "pre_transformer.input_proj.weight", t, latent, hidden);
        self.bias_add(&x, "pre_transformer.input_proj.bias", t, hidden);
        x = self.transformer(&x, t);
        x = self.matmul(&x, "pre_transformer.output_proj.weight", t, hidden, latent);
        self.bias_add(&x, "pre_transformer.output_proj.bias", t, latent);
        let mut h = self.transpose(&x, t, latent); // [1024, L] NCL
        let mut l = t;

        // --- 4. upsample stages ---
        for (u, &factor) in self.cfg.upsampling_ratios.clone().iter().enumerate() {
            h = self.causal_convtr(&h, &format!("upsample.{u}.0"), latent, latent, l, factor, factor);
            l *= factor;
            h = self.convnext(&h, &format!("upsample.{u}.1"), latent, l);
        }

        // --- 5. SEANet decoder ---
        let dec_dim = self.cfg.decoder_dim; // 1536
        h = self.causal_conv(&h, "decoder.0", latent, dec_dim, l, 7, 1, 1); // [1536, L]
        let rates = self.cfg.upsample_rates.clone();
        for (i, &rate) in rates.iter().enumerate() {
            let in_dim = dec_dim >> i;
            let out_dim = dec_dim >> (i + 1);
            let bp = format!("decoder.{}", i + 1);
            // block.0 SnakeBeta(in_dim)
            h = self.snake(&h, &format!("{bp}.block.0"), in_dim, l);
            // block.1 causal transposed conv (in->out, k=2·rate, stride=rate)
            h = self.causal_convtr(&h, &format!("{bp}.block.1"), in_dim, out_dim, l, 2 * rate, rate);
            l *= rate;
            // block.2/3/4 dilated residual units (dilation 1, 3, 9)
            for (j, dil) in [(2u32, 1u32), (3, 3), (4, 9)] {
                h = self.residual_unit(&h, &format!("{bp}.block.{j}"), out_dim, l, dil);
            }
        }
        let out_dim = dec_dim >> rates.len(); // 96
        h = self.snake(&h, "decoder.5", out_dim, l); // tail SnakeBeta
        h = self.causal_conv(&h, "decoder.6", out_dim, 1, l, 7, 1, 1); // -> [1, L]

        // --- 6. clamp ---
        let mut wav = self.gpu.read(&h, l as usize);
        for s in &mut wav {
            *s = s.clamp(-1.0, 1.0);
        }
        wav
    }

    /// One SEANet residual unit: `x + conv2(snake(conv1(snake(x))))`, conv1 is a
    /// dilated causal k7 conv, conv2 a causal k1 conv.
    fn residual_unit(&self, x: &DeviceBuffer, prefix: &str, c: u32, l: u32, dil: u32) -> DeviceBuffer {
        let y = self.snake(x, &format!("{prefix}.act1"), c, l);
        let y = self.causal_conv(&y, &format!("{prefix}.conv1"), c, c, l, 7, dil, 1);
        let y = self.snake(&y, &format!("{prefix}.act2"), c, l);
        let y = self.causal_conv(&y, &format!("{prefix}.conv2"), c, c, l, 1, 1, 1);
        self.add2(x, &y, c * l)
    }

    /// ConvNeXt block (NCL in/out): depthwise causal conv -> LayerNorm -> pwconv1
    /// -> GELU -> pwconv2 -> γ scale -> residual. LayerNorm (eps 1e-6) and GELU
    /// (exact erf) are host-computed to match `nn.LayerNorm`/`nn.GELU`.
    fn convnext(&self, x: &DeviceBuffer, prefix: &str, c: u32, l: u32) -> DeviceBuffer {
        // depthwise causal conv (groups = C, k7)
        let dw = self.causal_conv(x, &format!("{prefix}.dwconv"), c, c, l, 7, 1, c); // [C, L]
        let mut tm = self.gpu.read(&self.transpose(&dw, c, l), (l * c) as usize); // host [L, C]
        // host LayerNorm over C
        host_layernorm(&mut tm, l as usize, c as usize, &self.host[&format!("{prefix}.norm.weight")], &self.host[&format!("{prefix}.norm.bias")]);
        let tmb = self.upload(&tm);
        // pwconv1 (C -> 4C) + bias, exact GELU on host
        let hid = c * 4;
        let g = self.matmul(&tmb, &format!("{prefix}.pwconv1.weight"), l, c, hid);
        self.bias_add(&g, &format!("{prefix}.pwconv1.bias"), l, hid);
        let mut gv = self.gpu.read(&g, (l * hid) as usize);
        for v in &mut gv {
            *v = gelu_exact(*v);
        }
        let gbuf = self.upload(&gv);
        // pwconv2 (4C -> C) + bias
        let o = self.matmul(&gbuf, &format!("{prefix}.pwconv2.weight"), l, hid, c);
        self.bias_add(&o, &format!("{prefix}.pwconv2.bias"), l, c);
        // γ per-channel scale over token-major [L,C]: channel = idx % C (inner=1).
        let o = self.scale_chan(&o, &format!("{prefix}.gamma"), l * c, c, 1);
        let o_ncl = self.transpose(&o, l, c); // [C, L]
        self.add2(x, &o_ncl, c * l)
    }

    /// The 8-layer causal transformer over a token-major `[T, hidden]` buffer.
    fn transformer(&self, x0: &DeviceBuffer, t: u32) -> DeviceBuffer {
        let c = &self.cfg;
        let d = c.hidden_size;
        let ff = c.intermediate_size;
        let hd = c.head_dim;
        let nh = c.num_attention_heads;
        let nkv = c.num_key_value_heads;
        let hq = nh * hd;
        let hkv = nkv * hd;
        let theta = c.rope_theta;
        let ids = ids();
        let ga = Gqa { b: 1, t, n_heads: nh, n_kv_heads: nkv, head_dim: hd };

        let mut x = clone_buf(self, x0, t * d);
        for layer in 0..c.num_hidden_layers as usize {
            let p = |leaf: &str| format!("pre_transformer.layers.{layer}.{leaf}");
            // --- attention ---
            let xn = self.st((t * d) as usize);
            self.run(block::rmsnorm_fwd(&self.gpu, &ids, &x, self.w(&p("input_layernorm.weight")), &xn, d, t));
            let q = self.matmul(&xn, &p("self_attn.q_proj.weight"), t, d, hq);
            let k = self.matmul(&xn, &p("self_attn.k_proj.weight"), t, d, hkv);
            let v = self.matmul(&xn, &p("self_attn.v_proj.weight"), t, d, hkv);
            self.run(block::rope_fwd(&self.gpu, &ids, &q, t, nh, hd, hq, t, theta));
            self.run(block::rope_fwd(&self.gpu, &ids, &k, t, nkv, hd, hkv, t, theta));
            let scores = self.st((nh * t * t) as usize);
            let probs = self.st((nh * t * t) as usize);
            let ctx = self.st((t * hq) as usize);
            self.gpu.submit(&[], &block::gqa_fwd(&self.gpu, &ids, &ga, &q, &k, &v, &scores, &probs, &ctx));
            let attn = self.matmul(&ctx, &p("self_attn.o_proj.weight"), t, hq, d);
            let attn = self.scale_chan(&attn, &p("self_attn_layer_scale.scale"), t * d, d, 1);
            x = self.add2(&x, &attn, t * d);
            // --- MLP ---
            let xn = self.st((t * d) as usize);
            self.run(block::rmsnorm_fwd(&self.gpu, &ids, &x, self.w(&p("post_attention_layernorm.weight")), &xn, d, t));
            let gate = self.matmul(&xn, &p("mlp.gate_proj.weight"), t, d, ff);
            let up = self.matmul(&xn, &p("mlp.up_proj.weight"), t, d, ff);
            let hmid = self.st((t * ff) as usize);
            self.run(block::swiglu_fwd(&self.gpu, &ids, &gate, &up, &hmid, t * ff));
            let mlp = self.matmul(&hmid, &p("mlp.down_proj.weight"), t, ff, d);
            let mlp = self.scale_chan(&mlp, &p("mlp_layer_scale.scale"), t * d, d, 1);
            x = self.add2(&x, &mlp, t * d);
        }
        let out = self.st((t * d) as usize);
        self.run(block::rmsnorm_fwd(&self.gpu, &ids, &x, self.w("pre_transformer.norm.weight"), &out, d, t));
        out
    }
}

fn clone_buf(c: &Codec, x: &DeviceBuffer, n: u32) -> DeviceBuffer {
    let v = c.gpu.read(x, n as usize);
    c.upload(&v)
}

/// Kernel-id map for the shared `model::block` builders (only forward ids used).
fn ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: 0,
        rmsnorm_dx: 0,
        rmsnorm_dw: 0,
        rope: ROPE,
        rope_bwd: 0,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: 0,
        gqa_dv: 0,
        gqa_dq: 0,
        gqa_dk: 0,
        silu_mul: SILU_MUL,
        silu_da: 0,
        silu_db: 0,
    }
}

/// `nn.LayerNorm` over the last axis (C) of a row-major `[rows, C]` buffer,
/// population variance, eps 1e-6, with affine γ/β. In place.
fn host_layernorm(x: &mut [f32], rows: usize, c: usize, gamma: &[f32], beta: &[f32]) {
    for r in 0..rows {
        let row = &mut x[r * c..(r + 1) * c];
        let mean = row.iter().sum::<f32>() / c as f32;
        let var = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / c as f32;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for j in 0..c {
            row[j] = (row[j] - mean) * inv * gamma[j] + beta[j];
        }
    }
}

/// Exact (erf) GELU, matching `nn.GELU()` (default `approximate='none'`).
fn gelu_exact(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// erf via the Numerical-Recipes `erfc` rational approximation (|err| < 1.2e-7).
fn erf(x: f32) -> f32 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let erfc = t
        * (-z * z - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    let e = 1.0 - erfc;
    if x >= 0.0 {
        e
    } else {
        -e
    }
}
