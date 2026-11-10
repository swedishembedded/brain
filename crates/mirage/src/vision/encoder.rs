// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The vision encoder: convolutional stem, gated-mixing HConvGLU body,
//! register-augmented attention top with level memory.
//!
//! Padding cells (outside the resized image) are zero after the stem and are
//! never updated; the attention top therefore runs over the valid cells plus
//! the registers only, which leaves every padding row exactly as it would
//! stay if it were computed and then discarded.

use backend_cpu::par;
use model::hostmath;

use super::ops::{self, Linear, Map};
use super::weights::Weights;
use crate::config::MirageConfig;

/// Rotary base of the encoder is `enc_rope`; the head width is split into a
/// row half and a column half.
const ROPE_HALVES: usize = 2;

/// The HConvGLU channel mixer: Walsh transform, three banded convolutions
/// (gate with bias, value, output) and the transform back.
pub struct HConv {
    pub(crate) norm: Vec<f32>,
    gate: Vec<f32>,
    pub(crate) gate_bias: Vec<f32>,
    value: Vec<f32>,
    out: Vec<f32>,
    width: usize,
    kernel: usize,
}

/// An HConvGLU mixer with the Walsh transforms and the banded convolutions
/// multiplied out into two dense matrices, in the `[out, in]` layout GEMM
/// kernels consume: `forward = gelu_glu(x W_in + bias) W_out` on the normalised
/// input. The transforms are orthonormal and the bands sparse, so the products
/// are exact (accumulated in `f64`) and the device runs two GEMMs where the
/// host runs transforms and band loops.
pub struct FoldedHConv {
    /// `ZC` scale plus one, the weight of the RMS norm in front.
    pub norm_weight: Vec<f32>,
    /// `[2n, n]`: gate rows then value rows.
    pub in_wt: Vec<f32>,
    /// Gate bias, `[n]`.
    pub gate_bias: Vec<f32>,
    /// `[n, n]`.
    pub out_wt: Vec<f32>,
    /// Channel width.
    pub width: usize,
}

impl HConv {
    /// Multiply the transforms and bands out (see [`FoldedHConv`]).
    pub fn fold(&self) -> FoldedHConv {
        let (n, k) = (self.width, self.kernel);
        let r = k / 2;
        // The orthonormal sequency Walsh matrix: transform the identity rows.
        let mut walsh = vec![0f32; n * n];
        for i in 0..n {
            walsh[i * n + i] = 1.0;
        }
        ops::walsh_sequency_rows(&mut walsh, n);
        // C[i][j] = sum_k W[i][k] * M[k][j], M[k][j] = w[j][k - j + r] (|k - j| <= r).
        let walsh_band = |w: &[f32]| -> Vec<f32> {
            let mut c = vec![0f32; n * n];
            par::rows_mut(&mut c, n, |i, row| {
                for (j, out) in row.iter_mut().enumerate() {
                    let (lo, hi) = (j.saturating_sub(r), (j + r + 1).min(n));
                    *out = (lo..hi).map(|kk| f64::from(walsh[i * n + kk]) * f64::from(w[j * k + kk + r - j])).sum::<f64>() as f32;
                }
            });
            c
        };
        let gate = walsh_band(&self.gate);
        let value = walsh_band(&self.value);
        // D[i][j] = sum_k Mo[i][k] * W[k][j], Mo[i][k] = wo[k][i - k + r].
        let mut band_walsh = vec![0f32; n * n];
        par::rows_mut(&mut band_walsh, n, |i, row| {
            for (j, out) in row.iter_mut().enumerate() {
                let (lo, hi) = (i.saturating_sub(r), (i + r + 1).min(n));
                *out = (lo..hi).map(|kk| f64::from(self.out[kk * k + i + r - kk]) * f64::from(walsh[kk * n + j])).sum::<f64>() as f32;
            }
        });
        let mut in_wt = vec![0f32; 2 * n * n];
        let mut out_wt = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                in_wt[j * n + i] = gate[i * n + j];
                in_wt[(n + j) * n + i] = value[i * n + j];
                out_wt[j * n + i] = band_walsh[i * n + j];
            }
        }
        FoldedHConv { norm_weight: self.norm.iter().map(|s| 1.0 + s).collect(), in_wt, gate_bias: self.gate_bias.clone(), out_wt, width: n }
    }

    /// Read the mixer `prefix` (`body_3`, `top_0`, ...) from the checkpoint tensors.
    pub fn new(w: &Weights, prefix: &str, width: usize, kernel: usize) -> HConv {
        HConv {
            norm: w.vec(&format!("{prefix}/hconv_norm/scale")),
            gate: w.vec(&format!("{prefix}/hconv/gate_conv")),
            gate_bias: w.vec(&format!("{prefix}/hconv/gate_bias")),
            value: w.vec(&format!("{prefix}/hconv/value_conv")),
            out: w.vec(&format!("{prefix}/hconv/out_conv")),
            width,
            kernel,
        }
    }

    /// `HConvGLU(ZC(x, hconv_norm))` for `rows` rows of `x`.
    pub fn forward(&self, x: &[f32], rows: usize) -> Vec<f32> {
        let n = self.width;
        let mut u = ops::zc_norm(x, n, &self.norm);
        debug_assert_eq!(u.len(), rows * n);
        ops::walsh_sequency_rows(&mut u, n);
        let mut a = ops::band_rows(&u, n, &self.gate, self.kernel);
        let v = ops::band_rows(&u, n, &self.value, self.kernel);
        let bias = &self.gate_bias;
        par::rows_mut(&mut a, n, |r, row| {
            let vr = &v[r * n..(r + 1) * n];
            for ((a, b), v) in row.iter_mut().zip(bias).zip(vr) {
                *a = hostmath::gelu_tanh(*a + b) * v;
            }
        });
        let mut y = ops::band_rows(&a, n, &self.out, self.kernel);
        ops::walsh_sequency_rows(&mut y, n);
        y
    }
}

pub(crate) struct Stem {
    pub(crate) w0: Vec<f32>,
    pub(crate) norms: Vec<Vec<f32>>,
    pub(crate) dw: Vec<Vec<f32>>,
    pub(crate) pw: Vec<Linear>,
    pub(crate) widths: Vec<usize>,}

pub(crate) struct Body {
    pub(crate) mix_norm: Vec<f32>,
    pub(crate) dw_value: Vec<f32>,
    pub(crate) b_value: Vec<f32>,
    pub(crate) dw_gate: Vec<f32>,
    pub(crate) b_gate: Vec<f32>,
    pub(crate) dw_down: Vec<f32>,
    pub(crate) hconv: HConv,
    pub(crate) gamma: Vec<f32>,
    pub(crate) beta: Vec<f32>,}

pub(crate) struct Top {
    pub(crate) attn_norm: Vec<f32>,
    pub(crate) q: Linear,
    pub(crate) k: Linear,
    pub(crate) v: Linear,
    pub(crate) gate: Linear,
    pub(crate) out: Linear,
    pub(crate) q_norm: Vec<f32>,
    pub(crate) k_norm: Vec<f32>,
    pub(crate) post_attn_norm: Vec<f32>,
    pub(crate) attn_gate: f32,
    pub(crate) hconv: HConv,}

/// Everything the encoder produces that later stages consume.
pub struct EncoderOutput {
    /// Stem stage outputs by stride: `[256², 128², 64², 32², 16²]` maps
    /// (post-activation, padding zeroed).
    pub stages: Vec<(Map, Vec<f32>)>,
    /// Body output, `[256, 512]`.
    pub body: Vec<f32>,
    /// Top output with the registers appended, `[260, 512]`.
    pub top: Vec<f32>,
    /// The encoder memory `E`, `[256, 512]`, padding rows zero.
    pub encoder: Vec<f32>,
}

/// The encoder's weights, prepared for execution.
pub struct Encoder {
    pub(crate) cfg: MirageConfig,
    pub(crate) stem: Stem,
    pub(crate) body: Vec<Body>,
    pub(crate) top: Vec<Top>,
    pub(crate) registers: Vec<f32>,
    pub(crate) enc_norm: Vec<f32>,
    pub(crate) level_norm: Vec<Vec<f32>>,
    pub(crate) level_gates: Vec<f32>,
}

impl Encoder {
    /// Prepare the encoder from the checkpoint tensors.
    pub fn new(cfg: &MirageConfig, w: &Weights) -> Encoder {
        let stem = Stem {
            w0: w.vec("stem/w0"),
            norms: (0..cfg.stem.len()).map(|k| w.vec(&format!("stem/norm_{k}/scale"))).collect(),
            dw: (1..cfg.stem.len()).map(|k| w.vec(&format!("stem/dw_{k}"))).collect(),
            pw: (1..cfg.stem.len()).map(|k| w.linear(&format!("stem/pw_{k}/kernel"))).collect(),
            widths: cfg.stem.clone(),
        };
        let body = (0..cfg.body_layers)
            .map(|i| {
                let p = format!("body_{i}");
                Body {
                    mix_norm: w.vec(&format!("{p}/mix_norm/scale")),
                    dw_value: w.vec(&format!("{p}/mix/dw_value")),
                    b_value: w.vec(&format!("{p}/mix/b_value")),
                    dw_gate: w.vec(&format!("{p}/mix/dw_gate")),
                    b_gate: w.vec(&format!("{p}/mix/b_gate")),
                    dw_down: w.vec(&format!("{p}/mix/dw_down")),
                    hconv: HConv::new(w, &p, cfg.dim, cfg.hconv_kernel),
                    gamma: w.vec(&format!("{p}/grn/gamma")),
                    beta: w.vec(&format!("{p}/grn/beta")),
                }
            })
            .collect();
        let top = (0..cfg.attn_layers)
            .map(|i| {
                let p = format!("top_{i}");
                Top {
                    attn_norm: w.vec(&format!("{p}/attn_norm/scale")),
                    q: w.linear(&format!("{p}/self_attn/q_proj/kernel")),
                    k: w.linear(&format!("{p}/self_attn/k_proj/kernel")),
                    v: w.linear(&format!("{p}/self_attn/v_proj/kernel")),
                    gate: w.linear(&format!("{p}/self_attn/gate_proj/kernel")),
                    out: w.linear(&format!("{p}/self_attn/out_proj/kernel")),
                    q_norm: w.vec(&format!("{p}/self_attn/q_norm/scale")),
                    k_norm: w.vec(&format!("{p}/self_attn/k_norm/scale")),
                    post_attn_norm: w.vec(&format!("{p}/post_attn_norm/scale")),
                    attn_gate: w.get(&format!("{p}/attn_gate"))[0],
                    hconv: HConv::new(w, &p, cfg.dim, cfg.hconv_kernel),
                }
            })
            .collect();
        Encoder {
            cfg: cfg.clone(),
            stem,
            body,
            top,
            registers: w.vec("registers"),
            enc_norm: w.vec("enc_norm/scale"),
            level_norm: (0..cfg.attn_layers).map(|i| w.vec(&format!("level_norm_{i}/scale"))).collect(),
            level_gates: w.vec("level_gates"),
        }
    }

    /// Run the encoder on a canvas (`[512, 512, 3]`, see `preprocess::canvas`)
    /// whose image occupies the top-left `h x w` pixels.
    pub fn forward(&self, canvas: &[f32], h: usize, w: usize) -> EncoderOutput {
        let size = self.cfg.canvas;
        let mut stages = Vec::with_capacity(self.stem.widths.len());
        let (mut x, mut m) = ops::conv_dense(canvas, Map { h: size, w: size, c: 3 }, &self.stem.w0, 3, 3, self.stem.widths[0], 2, 2);
        for k in 0..self.stem.widths.len() {
            if k > 0 {
                let (d, dm) = ops::depthwise(&x, m, &self.stem.dw[k - 1], 3, 3, 2, 2);
                x = self.stem.pw[k - 1].apply(&d, dm.h * dm.w);
                m = Map { h: dm.h, w: dm.w, c: self.stem.widths[k] };
            }
            x = ops::zc_norm(&x, m.c, &self.stem.norms[k]);
            ops::silu_inplace(&mut x);
            ops::mask_rows(&mut x, m.c, &ops::valid_cells(h, w, m.h));
            stages.push((m, x.clone()));
        }

        let grid = self.cfg.grid();
        let d = self.cfg.dim;
        let valid = ops::valid_cells(h, w, grid);
        let gm = Map { h: grid, w: grid, c: d };
        let cells = grid * grid;
        for b in &self.body {
            self.body_block(b, &mut x, gm, &valid);
        }
        let body = x.clone();

        let (top_x, encoder) = self.attention_top(&x, &valid);
        debug_assert_eq!(top_x.len(), (cells + self.cfg.registers) * d);
        EncoderOutput { stages, body, top: top_x, encoder }
    }

    fn body_block(&self, b: &Body, x: &mut [f32], gm: Map, valid: &[bool]) {
        let d = gm.c;
        let mut hh = ops::zc_norm(x, d, &b.mix_norm);
        ops::mask_rows(&mut hh, d, valid);
        let (mut g, _) = ops::depthwise(&hh, gm, &b.dw_gate, self.cfg.mix_kernel, self.cfg.mix_kernel, 1, 1);
        let (v, _) = ops::depthwise(&hh, gm, &b.dw_value, self.cfg.mix_kernel, self.cfg.mix_kernel, 1, 1);
        par::rows_mut(&mut g, d, |r, row| {
            let vr = &v[r * d..(r + 1) * d];
            for (c, o) in row.iter_mut().enumerate() {
                *o = hostmath::silu(*o + b.b_gate[c]) * (vr[c] + b.b_value[c]);
            }
        });
        let (down, _) = ops::depthwise(&g, gm, &b.dw_down, self.cfg.mix_kernel, self.cfg.mix_kernel, 1, 1);
        par::rows_mut(x, d, |r, row| {
            if valid[r] {
                row.iter_mut().zip(&down[r * d..(r + 1) * d]).for_each(|(a, b)| *a += b);
            }
        });

        let y = b.hconv.forward(x, gm.h * gm.w);
        // Global response normalisation over the valid cells.
        let mut energy = vec![0f32; d];
        for (_, row) in y.chunks_exact(d).enumerate().filter(|(r, _)| valid[*r]) {
            energy.iter_mut().zip(row).for_each(|(e, v)| *e += v * v);
        }
        energy.iter_mut().for_each(|e| *e = (*e + ops::NORM_EPS).sqrt());
        let mean = energy.iter().sum::<f32>() / d as f32;
        let scale: Vec<f32> = (0..d).map(|c| b.gamma[c] * energy[c] / (mean + ops::NORM_EPS) + 1.0).collect();
        par::rows_mut(x, d, |r, row| {
            if valid[r] {
                for c in 0..d {
                    row[c] += y[r * d + c] * scale[c] + b.beta[c];
                }
            }
        });
    }

    /// The attention top over the valid cells plus registers; returns the
    /// `[cells + registers, dim]` token matrix and the encoder memory.
    fn attention_top(&self, body: &[f32], valid: &[bool]) -> (Vec<f32>, Vec<f32>) {
        let c = &self.cfg;
        let (d, grid) = (c.dim, c.grid());
        let cells = grid * grid;
        let live: Vec<usize> = (0..cells).filter(|&i| valid[i]).collect();
        let (nv, regs) = (live.len(), c.registers);
        let tokens = nv + regs;
        let mut x = Vec::with_capacity(tokens * d);
        for &i in &live {
            x.extend_from_slice(&body[i * d..(i + 1) * d]);
        }
        x.extend_from_slice(&self.registers);

        let mut acc = vec![0f32; nv * d];
        for (l, t) in self.top.iter().enumerate() {
            let level = ops::zc_norm(&x[..nv * d], d, &self.level_norm[l]);
            let gates = &self.level_gates[l * d..(l + 1) * d];
            par::rows_mut(&mut acc, d, |r, row| {
                for ch in 0..d {
                    row[ch] += gates[ch] * level[r * d + ch];
                }
            });

            let hh = ops::zc_norm(&x, d, &t.attn_norm);
            let (kd, vd) = (c.attn_kv_heads * c.attn_qk_dim, c.attn_kv_heads * c.attn_v_dim);
            let mut q = ops::zc_norm(&t.q.apply(&hh, tokens), c.attn_qk_dim, &t.q_norm);
            let mut k = ops::zc_norm(&t.k.apply(&hh, tokens), c.attn_qk_dim, &t.k_norm);
            let v = t.v.apply(&hh, tokens);
            self.rope_cells(&mut q, c.attn_heads, &live);
            self.rope_cells(&mut k, c.attn_kv_heads, &live);

            let att = self.attend(&q, &k, &v, tokens, (kd, vd));
            let gate = t.gate.apply(&hh, tokens);
            let mut gated = att;
            par::chunks_mut(&mut gated, 4096, |i, ch| ch.iter_mut().zip(&gate[i * 4096..]).for_each(|(a, g)| *a *= ops::sigmoid(*g)));
            let o = ops::zc_norm(&t.out.apply(&gated, tokens), d, &t.post_attn_norm);
            let a_gate = ops::sigmoid(t.attn_gate);
            let mut x2 = x.clone();
            par::chunks_mut(&mut x2, 4096, |i, ch| ch.iter_mut().zip(&o[i * 4096..]).for_each(|(a, o)| *a += a_gate * o));
            let mut y = t.hconv.forward(&x2, tokens);
            ops::add_inplace(&mut y, &x2);
            x = y;
        }

        let mut encoded = ops::zc_norm(&x[..nv * d], d, &self.enc_norm);
        ops::add_inplace(&mut encoded, &acc);

        let mut top = vec![0f32; (cells + regs) * d];
        let mut e = vec![0f32; cells * d];
        for (r, &i) in live.iter().enumerate() {
            top[i * d..(i + 1) * d].copy_from_slice(&x[r * d..(r + 1) * d]);
            e[i * d..(i + 1) * d].copy_from_slice(&encoded[r * d..(r + 1) * d]);
        }
        top[cells * d..].copy_from_slice(&x[nv * d..]);
        (top, e)
    }

    /// 2D rotary embedding on the cell rows: the head width is split into a
    /// row half and a column half; rotate-half pairs `(j, j + width/2)`.
    fn rope_cells(&self, t: &mut [f32], heads: usize, live: &[usize]) {
        let hd = self.cfg.attn_qk_dim;
        let grid = self.cfg.grid();
        let pairs = hd / 2;
        let per_axis = pairs / ROPE_HALVES;
        let freq: Vec<f32> = (0..per_axis).map(|j| self.cfg.enc_rope.powf(-(j as f32) / per_axis as f32)).collect();
        par::rows_mut(&mut t[..live.len() * heads * hd], heads * hd, |r, row| {
            let cell = live[r];
            let (cy, cx) = ((cell / grid) as f32, (cell % grid) as f32);
            for head in row.chunks_exact_mut(hd) {
                for j in 0..pairs {
                    let pos = if j < per_axis { cy } else { cx };
                    let angle = pos * freq[j % per_axis];
                    let (s, co) = angle.sin_cos();
                    let (a, b) = (head[j], head[j + pairs]);
                    head[j] = a * co - b * s;
                    head[j + pairs] = b * co + a * s;
                }
            }
        });
    }

    /// Grouped-query attention over all tokens (every token is a valid key).
    fn attend(&self, q: &[f32], k: &[f32], v: &[f32], tokens: usize, widths: (usize, usize)) -> Vec<f32> {
        let c = &self.cfg;
        let (kd, vd) = widths;
        let (hq, hk, qk, vh) = (c.attn_heads, c.attn_kv_heads, c.attn_qk_dim, c.attn_v_dim);
        let group = hq / hk;
        let scale = 1.0 / (qk as f32).sqrt();
        let mut out = vec![0f32; tokens * hq * vh];
        par::rows_mut(&mut out, hq * vh, |t, row| {
            let mut scores = vec![0f32; tokens];
            for h in 0..hq {
                let kvh = h / group;
                let qv = &q[(t * hq + h) * qk..][..qk];
                for (s, sc) in scores.iter_mut().enumerate() {
                    let kv = &k[s * kd + kvh * qk..][..qk];
                    *sc = qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale;
                }
                hostmath::softmax(&mut scores);
                let o = &mut row[h * vh..(h + 1) * vh];
                for (s, &p) in scores.iter().enumerate() {
                    let vv = &v[s * vd + kvh * vh..][..vh];
                    o.iter_mut().zip(vv).for_each(|(a, b)| *a += p * b);
                }
            }
        });
        out
    }
}
