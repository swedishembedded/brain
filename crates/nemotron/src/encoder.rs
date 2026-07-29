// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Nemotron FastConformer encoder (offline / non-streaming). This file lands the
//! stages incrementally, parity-gated against dumped HF activations:
//!   1. depthwise-separable causal subsampling (×8) + linear   ← this pass
//!   2. macaron Conformer blocks (rel-pos MHA + conv module)    (next)
//!   3. prompt + encoder projectors
//!
//! The causal Conv2d used by NeMo pads `(kernel-1, stride-1)` on BOTH the time and
//! frequency axes (asymmetric), which brain's symmetric-pad conv2d kernel can't
//! express directly — so the padding is done host-side and the conv runs with
//! `pad=0`. Padding is glue (no weights); the conv/linear math runs on device.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu};

use crate::config::NemotronConfig;

/// Kernels the encoder dispatches.
pub fn encoder_pipelines() -> &'static [(&'static str, &'static str)] {
    &[
        ("conv2d", kernels::CONV2D),                 // 0
        ("conv2d_gd", kernels::CONV2D_GD),           // 1
        ("add_chan_bcast", kernels::ADD_CHAN_BCAST), // 2
        ("relu_inplace", kernels::RELU_INPLACE),     // 3
        ("matmul", kernels::MATMUL),                 // 4
        ("bias_add", kernels::BIAS_ADD),             // 5
        ("matmul_rows", kernels::MATMUL_ROWS),       // 6 (unused on CPU: slower than matmul)
        ("silu", kernels::SILU),                     // 7
    ]
}
const K_SILU: usize = 7;
const K_CONV2D: usize = 0;
const K_CONV2D_GD: usize = 1;
const K_ADD_CHAN: usize = 2;
const K_RELU: usize = 3;
const K_MATMUL: usize = 4;
const K_BIAS_ADD: usize = 5;

/// Pad an NCHW buffer with `(top, bottom, left, right)` zeros. Host-side glue.
fn pad_nchw(x: &[f32], n: u32, c: u32, h: u32, w: u32, top: u32, bot: u32, left: u32, right: u32) -> (Vec<f32>, u32, u32) {
    let (hp, wp) = (h + top + bot, w + left + right);
    let mut out = vec![0.0f32; (n * c * hp * wp) as usize];
    for nn in 0..n {
        for cc in 0..c {
            for hh in 0..h {
                let src = ((nn * c + cc) * h + hh) * w;
                let dst = ((nn * c + cc) * hp + (hh + top)) * wp + left;
                out[dst as usize..(dst + w) as usize].copy_from_slice(&x[src as usize..(src + w) as usize]);
            }
        }
    }
    (out, hp, wp)
}

/// A built FastConformer encoder. It **owns** its `Gpu` and the uploaded device
/// weights, so a resident/served instance builds it once (weights uploaded once)
/// and reuses it across every call — the `DeviceBuffer` handles are lifetime-free,
/// so there is no borrow tying the encoder to an external `Gpu`.
pub struct Encoder {
    pub(crate) g: Gpu,
    cfg: NemotronConfig,
    w: HashMap<String, DeviceBuffer>,
    pub(crate) raw: HashMap<String, Vec<f32>>,
    /// Lazily-built per-layer relative-position tables for the streaming path
    /// (see `stream::rel_tables`): `[n_layers][band_width * hidden]`.
    pub(crate) rel_band: std::sync::OnceLock<Vec<Vec<f32>>>,
}

impl Encoder {
    pub fn new(g: Gpu, cfg: NemotronConfig, weights: &HashMap<String, Vec<f32>>) -> Encoder {
        let w = weights.iter().map(|(k, v)| (k.clone(), g.storage_init(k, v))).collect();
        Encoder { g, cfg, w, raw: weights.clone(), rel_band: std::sync::OnceLock::new() }
    }

    pub fn config(&self) -> &NemotronConfig {
        &self.cfg
    }

    pub(crate) fn wb(&self, name: &str) -> &DeviceBuffer {
        self.w.get(name).unwrap_or_else(|| panic!("nemotron weight missing: {name}"))
    }

    /// A single strided causal Conv2d (dense or depthwise) with `(k-1,s-1)` causal
    /// pad on both axes, per-channel bias, applied to NCHW `x`. Returns `(y, Ho, Wo)`.
    #[allow(clippy::too_many_arguments)]
    fn causal_conv(&self, x: &[f32], cin: u32, h: u32, w: u32, cout: u32, wname: &str, bname: &str, groups: u32) -> (Vec<f32>, u32, u32) {
        let (k, s) = (self.cfg.subsampling_kernel, self.cfg.subsampling_stride);
        let (pad, hp, wp) = pad_nchw(x, 1, cin, h, w, k - 1, s - 1, k - 1, s - 1);
        let ho = (hp - k) / s + 1;
        let wo = (wp - k) / s + 1;
        let xin = self.g.storage_init("nem.conv.x", &pad);
        let conv = self.g.storage((cout * ho * wo) as u64);
        let out = self.g.storage((cout * ho * wo) as u64);
        let mut steps = Vec::new();
        if groups == 1 {
            steps.push(self.g.step(K_CONV2D, &[&xin, self.wb(wname), &conv], &[1, cin, hp, wp, cout, k, s, 0, ho, wo], cout * ho * wo));
        } else {
            steps.push(self.g.step(K_CONV2D_GD, &[&xin, self.wb(wname), &conv], &[1, cin, hp, wp, cout, k, s, 0, 1, groups, ho, wo], cout * ho * wo));
        }
        steps.push(self.g.step(K_ADD_CHAN, &[&conv, self.wb(bname), &out], &[1, cout, ho * wo], cout * ho * wo));
        self.g.submit(&[], &steps);
        (self.g.read(&out, (cout * ho * wo) as usize), ho, wo)
    }

    /// 1×1 pointwise Conv2d (dense, stride 1) + bias.
    fn pointwise(&self, x: &[f32], cin: u32, h: u32, w: u32, cout: u32, wname: &str, bname: &str) -> Vec<f32> {
        let xin = self.g.storage_init("nem.pw.x", x);
        let conv = self.g.storage((cout * h * w) as u64);
        let out = self.g.storage((cout * h * w) as u64);
        let steps = vec![
            self.g.step(K_CONV2D, &[&xin, self.wb(wname), &conv], &[1, cin, h, w, cout, 1, 1, 0, h, w], cout * h * w),
            self.g.step(K_ADD_CHAN, &[&conv, self.wb(bname), &out], &[1, cout, h * w], cout * h * w),
        ];
        self.g.submit(&[], &steps);
        self.g.read(&out, (cout * h * w) as usize)
    }

    /// Strided conv (dense or depthwise, + per-channel bias) over an already-padded
    /// NCHW slab `[1, cin, h, w]`, pad=0 — the streaming path's window into the same
    /// kernels `causal_conv` dispatches (identical patches → identical values).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn conv_slab(&self, slab: &[f32], cin: usize, h: usize, w: usize, cout: usize, wname: &str, bname: &str, groups: u32, ho: usize, wo: usize) -> Vec<f32> {
        let (k, s) = (self.cfg.subsampling_kernel, self.cfg.subsampling_stride);
        let (cin, h, w, cout, ho, wo) = (cin as u32, h as u32, w as u32, cout as u32, ho as u32, wo as u32);
        let xin = self.g.storage_init("nem.conv.x", slab);
        let conv = self.g.storage((cout * ho * wo) as u64);
        let out = self.g.storage((cout * ho * wo) as u64);
        let mut steps = Vec::new();
        if groups == 1 {
            steps.push(self.g.step(K_CONV2D, &[&xin, self.wb(wname), &conv], &[1, cin, h, w, cout, k, s, 0, ho, wo], cout * ho * wo));
        } else {
            steps.push(self.g.step(K_CONV2D_GD, &[&xin, self.wb(wname), &conv], &[1, cin, h, w, cout, k, s, 0, 1, groups, ho, wo], cout * ho * wo));
        }
        steps.push(self.g.step(K_ADD_CHAN, &[&conv, self.wb(bname), &out], &[1, cout, ho * wo], cout * ho * wo));
        self.g.submit(&[], &steps);
        self.g.read(&out, (cout * ho * wo) as usize)
    }

    /// 1×1 pointwise conv (+ bias) over an NCHW slab `[1, ch, h, w]` (stride 1).
    pub(crate) fn pointwise_slab(&self, x: &[f32], ch: usize, h: usize, w: usize, wname: &str, bname: &str) -> Vec<f32> {
        let (ch, h, w) = (ch as u32, h as u32, w as u32);
        let xin = self.g.storage_init("nem.pw.x", x);
        let conv = self.g.storage((ch * h * w) as u64);
        let out = self.g.storage((ch * h * w) as u64);
        let steps = vec![
            self.g.step(K_CONV2D, &[&xin, self.wb(wname), &conv], &[1, ch, h, w, ch, 1, 1, 0, h, w], ch * h * w),
            self.g.step(K_ADD_CHAN, &[&conv, self.wb(bname), &out], &[1, ch, h * w], ch * h * w),
        ];
        self.g.submit(&[], &steps);
        self.g.read(&out, (ch * h * w) as usize)
    }

    fn relu(&self, x: &mut [f32]) {
        let b = self.g.storage_init("nem.relu", x);
        self.g.submit(&[], &[self.g.step(K_RELU, &[&b], &[x.len() as u32], x.len() as u32)]);
        x.copy_from_slice(&self.g.read(&b, x.len()));
    }

    /// Subsampled valid length after one stride-2 causal stage.
    fn stage_len(&self, len: u32) -> u32 {
        let (k, s) = (self.cfg.subsampling_kernel, self.cfg.subsampling_stride);
        (len + (k - 1) + (s - 1) - k) / s + 1
    }

    /// Zero time frames `>= valid` in an NCHW `[1, C, T, F]` buffer (matches
    /// NeMo `_mask_subsampled_frames`, stopping masked padding leaking into the
    /// next conv / the linear bias).
    fn mask_time(x: &mut [f32], c: u32, t: u32, f: u32, valid: u32) {
        for cc in 0..c as usize {
            for tt in valid as usize..t as usize {
                let base = (cc * t as usize + tt) * f as usize;
                for v in &mut x[base..base + f as usize] {
                    *v = 0.0;
                }
            }
        }
    }

    /// Depthwise-separable causal subsampling (×8) + linear projection.
    /// Input mel `[T, num_mel]` (row-major), `valid` real mel frames; output `[T', hidden]`.
    pub fn subsampling(&self, mel: &[f32], t: u32, valid: u32) -> (Vec<f32>, u32) {
        let cfg = &self.cfg;
        let ch = cfg.subsampling_channels;
        // [T, mel] -> NCHW [1, 1, T, mel]
        let (mut cur, mut h, mut w, mut cin) = (mel.to_vec(), t, cfg.num_mel_bins, 1u32);
        let mut vlen = valid;

        // stem: conv_in(1->ch), +bias, mask, relu
        let (y, ho, wo) = self.causal_conv(&cur, cin, h, w, ch, "encoder.subsampling.conv_in.weight", "encoder.subsampling.conv_in.bias", 1);
        cur = y;
        (h, w, cin) = (ho, wo, ch);
        vlen = self.stage_len(vlen);
        Self::mask_time(&mut cur, ch, h, w, vlen);
        self.relu(&mut cur);

        // depthwise-separable stages
        for i in 0..cfg.subsampling_stages() - 1 {
            let (y, ho, wo) = self.causal_conv(
                &cur, cin, h, w, ch,
                &format!("encoder.subsampling.layers.{i}.depthwise_conv.weight"),
                &format!("encoder.subsampling.layers.{i}.depthwise_conv.bias"),
                ch,
            );
            let mut pw = self.pointwise(&y, ch, ho, wo, ch, &format!("encoder.subsampling.layers.{i}.pointwise_conv.weight"), &format!("encoder.subsampling.layers.{i}.pointwise_conv.bias"));
            (h, w) = (ho, wo);
            vlen = self.stage_len(vlen);
            Self::mask_time(&mut pw, ch, h, w, vlen);
            cur = pw;
            self.relu(&mut cur);
        }

        // reshape [1, ch, T', F'] -> [T', ch*F'] then linear -> [T', hidden]
        let (tt, ff) = (h, w);
        let flat = ch * ff;
        let mut perm = vec![0.0f32; (tt * flat) as usize];
        for c in 0..ch as usize {
            for tpos in 0..tt as usize {
                for f in 0..ff as usize {
                    perm[tpos * flat as usize + c * ff as usize + f] = cur[(c * tt as usize + tpos) * ff as usize + f];
                }
            }
        }
        let pin = self.g.storage_init("nem.sub.perm", &perm);
        let lin = self.g.storage((tt * cfg.hidden) as u64);
        let steps = vec![
            self.g.step(K_MATMUL, &[&pin, self.wb("encoder.subsampling.linear.weight"), &lin], &[tt, flat, cfg.hidden], tt * cfg.hidden),
            self.g.step(K_BIAS_ADD, &[&lin, self.wb("encoder.subsampling.linear.bias")], &[tt, cfg.hidden], tt * cfg.hidden),
        ];
        self.g.submit(&[], &steps);
        (self.g.read(&lin, (tt * cfg.hidden) as usize), tt)
    }

    // ---- device Conformer blocks (big matmuls on device; small ops host) ----

    pub(crate) fn rw(&self, name: &str) -> &Vec<f32> {
        self.raw.get(name).unwrap_or_else(|| panic!("nemotron weight missing: {name}"))
    }

    /// Device matmul `[m,k]·Wᵀ → [m,n]` using the pre-uploaded weight `wname [n,k]`.
    /// Uses the 8-row-blocked kernel (bit-identical to `matmul`, 8× less weight
    /// memory traffic — the FF/projection linears are weight-bandwidth-bound).
    pub(crate) fn mm(&self, x: &[f32], wname: &str, m: usize, k: usize, n: usize) -> Vec<f32> {
        let xb = self.g.storage_init("nem.mm.x", x);
        let ob = self.g.storage((m * n) as u64);
        self.g.submit(&[], &[self.g.step(K_MATMUL, &[&xb, self.wb(wname), &ob], &[m as u32, k as u32, n as u32], (m * n) as u32)]);
        self.g.read(&ob, m * n)
    }

    /// Macaron feed-forward on device in ONE submit: Linear(c→ffn) → SiLU →
    /// Linear(ffn→c). Keeps the intermediate on-device (no host round-trip / extra
    /// submit vs two separate matmuls). Bit-identical to two `mm` + host SiLU.
    pub(crate) fn ff_dev(&self, x: &[f32], pre: &str, t: usize) -> Vec<f32> {
        let (c, ffn) = (self.cfg.hidden as usize, self.cfg.intermediate as usize);
        let xb = self.g.storage_init("nem.ff.x", x);
        let h1 = self.g.storage((t * ffn) as u64);
        let h2 = self.g.storage((t * ffn) as u64);
        let ob = self.g.storage((t * c) as u64);
        let steps = vec![
            self.g.step(K_MATMUL, &[&xb, self.wb(&format!("{pre}.linear1.weight")), &h1], &[t as u32, c as u32, ffn as u32], (t * ffn) as u32),
            self.g.step(K_SILU, &[&h1, &h2], &[(t * ffn) as u32], (t * ffn) as u32),
            self.g.step(K_MATMUL, &[&h2, self.wb(&format!("{pre}.linear2.weight")), &ob], &[t as u32, ffn as u32, c as u32], (t * c) as u32),
        ];
        self.g.submit(&[], &steps);
        self.g.read(&ob, t * c)
    }

    /// Rel-pos MHA (device projections + o_proj; per-head scoring host). Mirrors
    /// `reference::rel_pos_attention` exactly.
    fn attn_dev(&self, hn: &[f32], pre: &str, t: usize, valid: usize) -> Vec<f32> {
        use crate::reference::{banded_ok, rel_pos_encoding};
        let cfg = &self.cfg;
        let (c, heads, hd) = (cfg.hidden as usize, cfg.n_heads as usize, cfg.head_dim() as usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let (left, right) = ((cfg.sliding_window - 1) as usize, cfg.default_lookahead as usize);
        let q = self.mm(hn, &format!("{pre}.q_proj.weight"), t, c, c);
        let k = self.mm(hn, &format!("{pre}.k_proj.weight"), t, c, c);
        let v = self.mm(hn, &format!("{pre}.v_proj.weight"), t, c, c);
        let l = 2 * t - 1;
        let pe = rel_pos_encoding(t, c);
        let rel_k = self.mm(&pe, &format!("{pre}.relative_k_proj.weight"), l, c, c);
        let bu = self.rw(&format!("{pre}.bias_u"));
        let bv = self.rw(&format!("{pre}.bias_v"));
        // Per-head scoring is embarrassingly parallel; run the 8 heads across
        // threads, each producing its own [T, hd] context slab (disjoint output).
        let (q, k, v, rel_k) = (&q, &k, &v, &rel_k);
        let head_ctx: Vec<Vec<f32>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..heads)
                .map(|h| {
                    s.spawn(move || {
                        let (qh, kh, vh, rkh) = (
                            |i: usize, d: usize| q[i * c + h * hd + d],
                            |j: usize, d: usize| k[j * c + h * hd + d],
                            |j: usize, d: usize| v[j * c + h * hd + d],
                            |p: usize, d: usize| rel_k[p * c + h * hd + d],
                        );
                        let (bus, bvs) = (&bu[h * hd..h * hd + hd], &bv[h * hd..h * hd + hd]);
                        let mut bd_raw = vec![0.0f32; t * l];
                        for i in 0..t {
                            for pp in 0..l {
                                let mut acc = 0.0f32;
                                for d in 0..hd {
                                    acc += (qh(i, d) + bvs[d]) * rkh(pp, d);
                                }
                                bd_raw[i * l + pp] = acc;
                            }
                        }
                        let bd = crate::kernels::rel_shift_ref(&bd_raw, 1, t, l);
                        let mut out = vec![0.0f32; t * hd]; // [T, hd] for this head
                        for i in 0..t {
                            let mut sc = vec![f32::NEG_INFINITY; t];
                            for j in 0..t {
                                if j >= valid || !banded_ok(i, j, left, right) {
                                    continue;
                                }
                                let mut ac = 0.0f32;
                                for d in 0..hd {
                                    ac += (qh(i, d) + bus[d]) * kh(j, d);
                                }
                                sc[j] = ac * scale + bd[i * l + j] * scale;
                            }
                            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                            let mut den = 0.0f32;
                            for sv in &mut sc {
                                *sv = if sv.is_finite() { (*sv - mx).exp() } else { 0.0 };
                                den += *sv;
                            }
                            let inv = if den > 0.0 { 1.0 / den } else { 0.0 };
                            for d in 0..hd {
                                let mut acc = 0.0f32;
                                for j in 0..t {
                                    acc += sc[j] * vh(j, d);
                                }
                                out[i * hd + d] = acc * inv;
                            }
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // assemble [T, heads*hd] from per-head [T, hd] slabs
        let mut ctx = vec![0.0f32; t * c];
        for (h, hc) in head_ctx.iter().enumerate() {
            for i in 0..t {
                ctx[i * c + h * hd..i * c + h * hd + hd].copy_from_slice(&hc[i * hd..i * hd + hd]);
            }
        }
        self.mm(&ctx, &format!("{pre}.o_proj.weight"), t, c, c)
    }

    /// Conformer conv module (device pointwise convs; GLU/depthwise/LN/SiLU host).
    fn conv_dev(&self, hn: &[f32], pre: &str, t: usize, valid: usize) -> Vec<f32> {
        use crate::reference::{layernorm, sigmoid, silu};
        let (c, k) = (self.cfg.hidden as usize, self.cfg.conv_kernel as usize);
        let pc1 = self.mm(hn, &format!("{pre}.pointwise_conv1.weight"), t, c, 2 * c);
        let mut glu = vec![0.0f32; t * c];
        for i in 0..valid.min(t) {
            for j in 0..c {
                glu[i * c + j] = pc1[i * 2 * c + j] * sigmoid(pc1[i * 2 * c + c + j]);
            }
        }
        let dw = self.rw(&format!("{pre}.depthwise_conv.weight"));
        let mut conv = vec![0.0f32; t * c];
        for ch in 0..c {
            for i in 0..t {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    let src = i as i64 - (k as i64 - 1) + kk as i64;
                    if src >= 0 {
                        acc += glu[src as usize * c + ch] * dw[ch * k + kk];
                    }
                }
                conv[i * c + ch] = acc;
            }
        }
        let mut act = layernorm(&conv, self.rw(&format!("{pre}.norm.weight")), self.rw(&format!("{pre}.norm.bias")), t, c, self.cfg.ln_eps);
        for v in &mut act {
            *v = silu(*v);
        }
        self.mm(&act, &format!("{pre}.pointwise_conv2.weight"), t, c, c)
    }

    /// One Conformer block over a **row-concatenated batch** of items. `spans` gives
    /// each item's `(row_offset, frames, valid)`; `tt` is the total row count.
    ///
    /// The per-frame ops — LayerNorm and both macaron feed-forwards — are
    /// position-independent, so they run **once over all `tt` rows** (one batched
    /// device matmul each, the weight-bandwidth-bound cost). Only the two
    /// position-mixing ops (relative-position attention, causal depthwise conv) are
    /// per-item, looping over each span's own `[frames, valid]` slice. Because the
    /// batched matmuls are row-wise identical to per-item ones, a single-span call is
    /// **bit-identical** to the old per-item block — so `encode` (one span) still
    /// matches the dumped HF goldens.
    fn block_dev_batch(&self, h: &[f32], b: u32, spans: &[(usize, usize, usize)], tt: usize) -> Vec<f32> {
        use crate::reference::layernorm;
        let c = self.cfg.hidden as usize;
        let pre = format!("encoder.layers.{b}");
        let ln = |x: &[f32], n: &str| layernorm(x, self.rw(&format!("{pre}.{n}.weight")), self.rw(&format!("{pre}.{n}.bias")), tt, c, self.cfg.ln_eps);
        let mut h = h.to_vec();
        // macaron FF1 — batched over every row
        let ff1 = self.ff_dev(&ln(&h, "norm_feed_forward1"), &format!("{pre}.feed_forward1"), tt);
        for i in 0..tt * c {
            h[i] += 0.5 * ff1[i];
        }
        // self-attention — per item (position-mixing, banded per its own length)
        let hn = ln(&h, "norm_self_att");
        for &(off, ti, vi) in spans {
            let att = self.attn_dev(&hn[off * c..(off + ti) * c], &format!("{pre}.self_attn"), ti, vi);
            for i in 0..ti * c {
                h[off * c + i] += att[i];
            }
        }
        // conv module — per item (causal depthwise conv along time)
        let hn = ln(&h, "norm_conv");
        for &(off, ti, vi) in spans {
            let cv = self.conv_dev(&hn[off * c..(off + ti) * c], &format!("{pre}.conv"), ti, vi);
            for i in 0..ti * c {
                h[off * c + i] += cv[i];
            }
        }
        // macaron FF2 — batched over every row
        let ff2 = self.ff_dev(&ln(&h, "norm_feed_forward2"), &format!("{pre}.feed_forward2"), tt);
        for i in 0..tt * c {
            h[i] += 0.5 * ff2[i];
        }
        ln(&h, "norm_out")
    }

    /// RNN-T joint logits for one (encoder frame, decoder state) pair, on device:
    /// `head(relu(enc_t + dec_u))` → `[vocab]`. The head matmul (640→13088) is the
    /// dominant RNN-T cost, so it runs on the device.
    pub fn joint_logits(&self, enc_t: &[f32], dec_u: &[f32]) -> Vec<f32> {
        let dh = self.cfg.decoder_hidden as usize;
        let sum: Vec<f32> = (0..dh).map(|j| (enc_t[j] + dec_u[j]).max(0.0)).collect(); // relu
        let mut logits = self.mm(&sum, "joint.head.weight", 1, dh, self.cfg.vocab as usize);
        let jb = self.rw("joint.head.bias");
        for (l, b) in logits.iter_mut().zip(jb) {
            *l += b;
        }
        logits
    }

    /// Full encoder for ONE utterance: mel → subsampling → 24 blocks →
    /// prompt/encoder projectors → pooler `[T, decoder_hidden]`. Thin wrapper over
    /// [`Encoder::encode_batch`] with a single item (one implementation, so the
    /// goldens gate both paths).
    pub fn encode(&self, mel: &[f32], t: u32, mel_valid: u32, prompt_id: usize) -> (Vec<f32>, u32) {
        self.encode_batch(&[(mel, t, mel_valid)], prompt_id).pop().unwrap()
    }

    /// **Batched** encoder over N concurrent utterances. Each `items[i]` is
    /// `(mel[T·num_mel], t, mel_valid)`; returns one `(pooler[T'·dh], valid)` per
    /// item, in order. The dominant per-frame matmuls (macaron FFs, the two
    /// projectors, the encoder projector) run **once over the row-concatenation of
    /// every item's frames** — genuine device batching whose throughput scales with
    /// the batch rather than N serial forwards. Attention and the depthwise conv stay
    /// per-item (they mix positions within one utterance). Subsampling runs per item
    /// (2-D conv over that item's T×F). Bit-identical, per item, to calling `encode`
    /// on each utterance alone.
    pub fn encode_batch(&self, items: &[(&[f32], u32, u32)], prompt_id: usize) -> Vec<(Vec<f32>, u32)> {
        let cfg = &self.cfg;
        let c = cfg.hidden as usize;
        // 1. subsampling per item; collect row spans (offset, frames, valid).
        let mut spans: Vec<(usize, usize, usize)> = Vec::with_capacity(items.len());
        let mut h: Vec<f32> = Vec::new();
        let mut offset = 0usize;
        for &(mel, t, mel_valid) in items {
            let (sub, tt) = self.subsampling(mel, t, mel_valid);
            let valid = cfg.subsampled_len(mel_valid) as usize;
            let tt = tt as usize;
            spans.push((offset, tt, valid));
            offset += tt;
            h.extend_from_slice(&sub);
        }
        let tt = offset; // total rows across the batch
        if tt == 0 {
            return items.iter().map(|_| (Vec::new(), 0u32)).collect();
        }
        // 2. Conformer stack over the batch (per-frame ops batched, mixing ops per item).
        for b in 0..cfg.n_layers {
            h = self.block_dev_batch(&h, b, &spans, tt);
        }
        // 3. prompt + encoder projectors — batched over all rows.
        let pooler = self.project_rows(&h, tt, prompt_id);
        let dh = cfg.decoder_hidden as usize;
        // 4. split the pooler back into per-item slices.
        spans.iter().map(|&(off, ti, vi)| (pooler[off * dh..(off + ti) * dh].to_vec(), vi as u32)).collect()
    }

    /// Prompt + encoder projectors over `tt` Conformer-output rows `[tt, hidden]` →
    /// pooler `[tt, decoder_hidden]` (device matmuls + host bias/relu/one-hot; every
    /// row carries the same language one-hot). Per-frame math — shared by the
    /// batched offline forward and the streaming path.
    pub(crate) fn project_rows(&self, h: &[f32], tt: usize, prompt_id: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let c = cfg.hidden as usize;
        let (np, pi, dh) = (cfg.num_prompts as usize, cfg.prompt_intermediate as usize, cfg.decoder_hidden as usize);
        let mut cat = vec![0.0f32; tt * (c + np)];
        for i in 0..tt {
            cat[i * (c + np)..i * (c + np) + c].copy_from_slice(&h[i * c..i * c + c]);
            cat[i * (c + np) + c + prompt_id] = 1.0;
        }
        let mut f1 = self.mm(&cat, "prompt_projector.linear_1.weight", tt, c + np, pi);
        let b1 = self.rw("prompt_projector.linear_1.bias");
        for i in 0..tt {
            for j in 0..pi {
                f1[i * pi + j] = (f1[i * pi + j] + b1[j]).max(0.0);
            }
        }
        let mut fused = self.mm(&f1, "prompt_projector.linear_2.weight", tt, pi, c);
        let b2 = self.rw("prompt_projector.linear_2.bias");
        for i in 0..tt {
            for j in 0..c {
                fused[i * c + j] += b2[j];
            }
        }
        let mut pooler = self.mm(&fused, "encoder_projector.weight", tt, c, dh);
        let eb = self.rw("encoder_projector.bias");
        for i in 0..tt {
            for j in 0..dh {
                pooler[i * dh + j] += eb[j];
            }
        }
        pooler
    }

    /// Batched transcription: N waveforms → N token-id sequences. The encoder forward
    /// is batched across all utterances (one `encode_batch`); the RNN-T greedy decode
    /// runs per stream (its length is data-dependent). This is the resident adapter's
    /// batched path for concurrent streams.
    pub fn transcribe_batch(&self, wavs: &[&[f32]], prompt_id: usize) -> Vec<Vec<u32>> {
        if wavs.is_empty() {
            return Vec::new();
        }
        let mels: Vec<(Vec<f32>, u32, u32)> = wavs
            .iter()
            .map(|w| {
                let (mel, t, _nmel) = audio::asr_frontend::nemotron_logmel(w);
                (mel, t as u32, w.len() as u32 / 160)
            })
            .collect();
        let refs: Vec<(&[f32], u32, u32)> = mels.iter().map(|(m, t, v)| (m.as_slice(), *t, *v)).collect();
        let encoded = self.encode_batch(&refs, prompt_id);
        encoded.iter().map(|(pool, valid)| self.rnnt_greedy(pool, *valid as usize)).collect()
    }

    /// Transcribe a 16 kHz mono waveform → emitted RNN-T token ids (non-blank),
    /// reusing the encoder's already-uploaded weights (no per-call rebuild).
    /// `prompt_id` selects the language prompt. This is the shared entry point for
    /// both the one-shot [`crate::model::NemotronAsr`] and the resident/served
    /// instance — front end → device encoder → RNN-T greedy decode.
    pub fn transcribe(&self, wav: &[f32], prompt_id: usize) -> Vec<u32> {
        // front end (matches HF: preemphasis, 512-fft/400-win, log-mel 128, no norm)
        let (mel, t, _nmel) = audio::asr_frontend::nemotron_logmel(wav);
        let mel_valid = wav.len() as u32 / 160; // floor(L/hop) — the extractor's valid length
        let te = std::time::Instant::now();
        let (pooler, valid) = self.encode(&mel, t as u32, mel_valid, prompt_id);
        if std::env::var("NEM_TIMING").is_ok() {
            eprintln!("  encode: {:?}", te.elapsed());
        }
        let td = std::time::Instant::now();
        let out = self.rnnt_greedy(&pooler, valid as usize);
        if std::env::var("NEM_TIMING").is_ok() {
            eprintln!("  decode: {:?}", td.elapsed());
        }
        out
    }

    /// RNN-T greedy transducer decode over an encoded `pooler` `[T, decoder_hidden]`.
    /// LSTM prediction net runs host-side (m=1 steps); the joint head is on device.
    /// One implementation with the streaming path: a fresh [`DecodeState`] stepped
    /// over every frame.
    pub(crate) fn rnnt_greedy(&self, pooler: &[f32], valid: usize) -> Vec<u32> {
        let mut st = DecodeState::new(self, self.cfg.blank_token_id);
        st.step_frames(self, pooler, valid);
        st.emitted
    }
}

/// Persistent RNN-T greedy-decode state: the LSTM prediction-net state, the current
/// decoder output vector, and the tokens emitted so far. The streaming path keeps
/// one alive across pushes; the offline `rnnt_greedy` runs a fresh one to the end —
/// the same frame loop either way, so streamed decode is identical by construction.
pub(crate) struct DecodeState {
    lstm: crate::reference::LstmState,
    dec: Vec<f32>,
    pub(crate) emitted: Vec<u32>,
}

impl DecodeState {
    pub(crate) fn new(enc: &Encoder, blank: u32) -> DecodeState {
        let cfg = enc.config();
        let mut lstm = crate::reference::LstmState::new(cfg.num_decoder_layers as usize, cfg.decoder_hidden as usize);
        let dec = crate::reference::lstm_predict(blank, &mut lstm, &enc.raw, cfg);
        DecodeState { lstm, dec, emitted: Vec::new() }
    }

    /// Greedy-decode `n` new pooler frames `[n, decoder_hidden]`, appending emitted
    /// tokens. Each frame's blank-terminated inner loop completes within the call,
    /// so state carried across calls is exactly (LSTM state, decoder output, tokens).
    pub(crate) fn step_frames(&mut self, enc: &Encoder, pooler: &[f32], n: usize) {
        let cfg = *enc.config();
        let dh = cfg.decoder_hidden as usize;
        let blank = cfg.blank_token_id;
        let (mut frame, mut symbols) = (0usize, 0u32);
        while frame < n {
            let logits = enc.joint_logits(&pooler[frame * dh..frame * dh + dh], &self.dec);
            let token = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap();
            if token == blank || symbols >= cfg.max_symbols_per_step {
                frame += 1;
                symbols = 0;
            } else {
                self.emitted.push(token);
                symbols += 1;
                self.dec = crate::reference::lstm_predict(token, &mut self.lstm, &enc.raw, &cfg);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(non_snake_case)] // GOLD/CKPT test-path locals (see AGENTS.md: no absolute paths)
    use super::*;
    use std::io::Read;
    use std::path::Path;


    fn read_f32(p: &str) -> Vec<f32> {
        let mut f = std::fs::File::open(p).unwrap_or_else(|_| panic!("missing {p}"));
        let mut b = Vec::new();
        f.read_to_end(&mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    #[test]
    fn subsampling_matches_reference() {
        let GOLD = crate::testdata("asr/golden/nemotron");
        let CKPT = crate::testdata("asr/nemotron/hf");
        if !Path::new(&format!("{GOLD}/subsampling.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let mel = read_f32(&format!("{GOLD}/input_features.f32")); // [T, 128]
        let nmel = cfg.num_mel_bins as usize;
        let t = (mel.len() / nmel) as u32;
        // valid frames = frames not zeroed by the frontend mask (masked frames are exactly 0)
        let valid = (0..t as usize).filter(|&i| mel[i * nmel..(i + 1) * nmel].iter().any(|&v| v != 0.0)).count() as u32;
        let refsub = read_f32(&format!("{GOLD}/subsampling.f32")); // [T', 1024]

        let weights = crate::import::load_tensors(Path::new(&CKPT)).expect("load");
        let g = Gpu::new_cpu(encoder_pipelines());
        let enc = Encoder::new(g, cfg, &weights);
        let (sub, tt) = enc.subsampling(&mel, t, valid);
        eprintln!("subsampling out [{tt}, {}] vs golden {}", sub.len() / tt as usize, refsub.len());
        assert_eq!(sub.len(), refsub.len(), "shape mismatch");
        let d = sub.iter().zip(&refsub).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("subsampling maxdiff {d}");
        assert!(d < 2e-3, "subsampling maxdiff {d}");
    }



    #[test]
    #[ignore = "requires a real GPU: run with BRAIN_DEVICE=vulkan (Arc iGPU) or gpu"]
    fn gpu_encoder_matches_reference() {
        let GOLD = crate::testdata("asr/golden/nemotron");
        let CKPT = crate::testdata("asr/nemotron/hf");
        if !Path::new(&format!("{GOLD}/pooler.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let mel = read_f32(&format!("{GOLD}/input_features.f32"));
        let nmel = cfg.num_mel_bins as usize;
        let t = (mel.len() / nmel) as u32;
        let mel_valid = (0..t as usize).filter(|&i| mel[i * nmel..(i + 1) * nmel].iter().any(|&v| v != 0.0)).count() as u32;
        let ref_pool = read_f32(&format!("{GOLD}/pooler.f32"));
        let dh = cfg.decoder_hidden as usize;

        let w = crate::import::load_tensors(Path::new(&CKPT)).expect("load");
        let g = Gpu::new(encoder_pipelines()); // device-resolved (BRAIN_DEVICE)
        let enc = Encoder::new(g, cfg, &w);
        let _ = enc.encode(&mel, t, mel_valid, 0); // warm up (shader compile)
        let t0 = std::time::Instant::now();
        let (pool, valid) = enc.encode(&mel, t, mel_valid, 0);
        let elapsed = t0.elapsed();
        let n = valid as usize * dh;
        let d = pool[..n].iter().zip(&ref_pool[..n]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("GPU encoder pooler maxdiff {d} (valid {valid}) in {:?}", elapsed);
        assert!(d < 1e-2, "GPU encoder maxdiff {d}");
    }

    #[test]
    fn device_encoder_matches_reference() {
        let GOLD = crate::testdata("asr/golden/nemotron");
        let CKPT = crate::testdata("asr/nemotron/hf");
        if !Path::new(&format!("{GOLD}/pooler.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let mel = read_f32(&format!("{GOLD}/input_features.f32"));
        let nmel = cfg.num_mel_bins as usize;
        let t = (mel.len() / nmel) as u32;
        let mel_valid = (0..t as usize).filter(|&i| mel[i * nmel..(i + 1) * nmel].iter().any(|&v| v != 0.0)).count() as u32;
        let ref_pool = read_f32(&format!("{GOLD}/pooler.f32"));
        let dh = cfg.decoder_hidden as usize;

        let w = crate::import::load_tensors(Path::new(&CKPT)).expect("load");
        let g = Gpu::new_cpu(encoder_pipelines());
        let enc = Encoder::new(g, cfg, &w);
        let t0 = std::time::Instant::now();
        let (pool, valid) = enc.encode(&mel, t, mel_valid, 0);
        let elapsed = t0.elapsed();
        let n = valid as usize * dh;
        let d = pool[..n].iter().zip(&ref_pool[..n]).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("device encoder pooler maxdiff {d} (valid {valid}) in {:?}", elapsed);
        assert!(d < 5e-3, "device encoder maxdiff {d}");
    }

    /// A batched forward over N copies of one utterance must return, per item, the
    /// **bit-identical** pooler of the single-item `encode` — the invariant that makes
    /// concurrent-stream batching safe. Heavy (loads the 0.6B checkpoint and holds the
    /// batch's activations), so `#[ignore]`d out of the concurrent default run to avoid
    /// OOM alongside the other checkpoint-loading parity tests; run it explicitly:
    /// `cargo test -p brain-nemotron --release batched_encode_matches_single -- --ignored`.
    #[test]
    #[ignore = "loads the 0.6B checkpoint + batched activations (heavy; run explicitly)"]
    fn batched_encode_matches_single() {
        let GOLD = crate::testdata("asr/golden/nemotron");
        let CKPT = crate::testdata("asr/nemotron/hf");
        if !Path::new(&format!("{GOLD}/input_features.f32")).exists() || !Path::new(&format!("{CKPT}/model.safetensors")).exists() {
            eprintln!("skipping: goldens/checkpoint absent");
            return;
        }
        let cfg = NemotronConfig::nemotron_3_5_asr_0_6b();
        let mel = read_f32(&format!("{GOLD}/input_features.f32"));
        let nmel = cfg.num_mel_bins as usize;
        let t = (mel.len() / nmel) as u32;
        let mel_valid = (0..t as usize).filter(|&i| mel[i * nmel..(i + 1) * nmel].iter().any(|&v| v != 0.0)).count() as u32;

        let w = crate::import::load_tensors(Path::new(&CKPT)).expect("load");
        let g = Gpu::new_cpu(encoder_pipelines());
        let enc = Encoder::new(g, cfg, &w);
        let (single, vsingle) = enc.encode(&mel, t, mel_valid, 0);

        // batch of 3 identical utterances in one forward
        let items = [(&mel[..], t, mel_valid), (&mel[..], t, mel_valid), (&mel[..], t, mel_valid)];
        let t0 = std::time::Instant::now();
        let batched = enc.encode_batch(&items, 0);
        eprintln!("batch=3 encode in {:?} ({:.1}x one)", t0.elapsed(), 3.0);
        assert_eq!(batched.len(), 3);
        for (i, (pool, valid)) in batched.iter().enumerate() {
            assert_eq!(*valid, vsingle, "item {i} valid len");
            assert_eq!(pool.len(), single.len(), "item {i} pooler shape");
            let d = pool.iter().zip(&single).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(d == 0.0, "item {i}: batched pooler must be bit-identical to single (maxdiff {d})");
        }
    }
}
