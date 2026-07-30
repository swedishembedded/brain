// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The FLUX.2 transformer forward: double-stream blocks (separate img/txt
//! weights, joint attention) then single-stream parallel blocks, with the three
//! **global** modulation linears folded into LayerNorm affine params.
//!
//! Klein's standard path modulates every token identically (the modulation
//! vector depends only on timestep — and guidance on dev), so
//! `(1 + scale)·LN(x) + shift` is exactly `LayerNorm` with `gamma = 1+scale`,
//! `beta = shift`. The whole model therefore carries six modulated-LN
//! (gamma, beta) pairs and five gate vectors, recomputed on the host and
//! re-uploaded once per forward — no per-block modulation work at all. (The
//! Klein-9B-KV blended per-token modulation path would break this fold; it is
//! deliberately out of scope here.)
//!
//! Layout: one joint residual slab `[n, D]`, text rows first (`0..nt`), image
//! (+ reference) rows after — the reference's `[txt, img, refs]` order. Stream-
//! specific ops run on row ranges via `step_sliced`; joint attention reads the
//! whole slab. Fused checkpoint weights (`qkv`, `mlp.0`, `linear1`, `linear2`)
//! are split at build time into per-projection device buffers so every matmul
//! is a plain full-buffer dispatch.

use gpu_core::{DeviceBuffer, Gpu, Step};

use crate::config::Flux2Config;
use crate::import::Tensors;

pub const KERNELS: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("matmul", kernels::MATMUL),
    ("matmul_reg2", kernels::MATMUL_REG2),
    ("rmsnorm_eps", kernels::RMSNORM_EPS),
    ("rope_interleave_table", kernels::ROPE_INTERLEAVE_TABLE),
    ("pack_qkv", kernels::PACK_QKV),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("flash_attn_bidir", kernels::FLASH_ATTN_BIDIR),
    ("silu_mul", kernels::SILU_MUL),
    ("gate_row", kernels::GATE_ROW),
];
const K_LN: usize = 0;
const K_MATMUL: usize = 1;
const K_MATMUL_REG2: usize = 2;
const K_RMSNORM: usize = 3;
const K_ROPE: usize = 4;
const K_PACK: usize = 5;
const K_SCORES: usize = 6;
const K_SOFTMAX: usize = 7;
const K_APPLY: usize = 8;
const K_FLASH: usize = 9;
const K_SILU_MUL: usize = 10;
const K_GATE: usize = 11;

const EPS: f32 = 1e-6;

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// Split rows `[r0, r1)` out of a fused `[rows, cols]` host tensor.
fn rows(data: &[f32], cols: usize, r0: usize, r1: usize) -> &[f32] {
    &data[r0 * cols..r1 * cols]
}

/// One attention/MLP weight set (a double block holds two: img and txt).
struct StreamW {
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    nq: DeviceBuffer,
    nk: DeviceBuffer,
    wo: DeviceBuffer,
    w1: DeviceBuffer,
    w3: DeviceBuffer,
    w2: DeviceBuffer,
}

struct SingleW {
    wq: DeviceBuffer,
    wk: DeviceBuffer,
    wv: DeviceBuffer,
    nq: DeviceBuffer,
    nk: DeviceBuffer,
    w1: DeviceBuffer,
    w3: DeviceBuffer,
    /// linear2 column-split: `out = wo_a @ attn_ctx + wo_b @ mlp_act`.
    wo_a: DeviceBuffer,
    wo_b: DeviceBuffer,
}

/// The six modulated-LN sites and five gates, in upload order.
struct ModBufs {
    gamma: Vec<DeviceBuffer>, // [img1, img2, txt1, txt2, single, final]
    beta: Vec<DeviceBuffer>,
    gate: Vec<DeviceBuffer>, // [img1, img2, txt1, txt2, single]
}

struct Scratch {
    x0: DeviceBuffer,
    x1: DeviceBuffer,
    n1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    qn: DeviceBuffer,
    kn: DeviceBuffer,
    qr: DeviceBuffer,
    kr: DeviceBuffer,
    qkv: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    h1: DeviceBuffer,
    h2: DeviceBuffer,
    hs: DeviceBuffer,
    mlp: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    out: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    tok_in: DeviceBuffer,
    ctx_in: DeviceBuffer,
}

pub struct Flux2Model {
    pub cfg: Flux2Config,
    gpu: Gpu,
    /// max joint rows (txt + img + refs) the scratch is sized for
    n_max: u32,
    fast: bool,
    dbl: Vec<(StreamW, StreamW)>,
    sgl: Vec<SingleW>,
    img_in: DeviceBuffer,
    txt_in: DeviceBuffer,
    final_w: DeviceBuffer,
    modb: ModBufs,
    scr: Scratch,
    // host-side conditioning weights
    time_in_a: Vec<f32>,  // [D,256]
    time_in_b: Vec<f32>,  // [D,D]
    mod_img: Vec<f32>,    // [6D,D]
    mod_txt: Vec<f32>,    // [6D,D]
    mod_single: Vec<f32>, // [3D,D]
    final_adaln: Vec<f32>, // [2D,D]
}

impl Flux2Model {
    /// Build device state from imported (BFL-named) tensors, sized for at most
    /// `n_max` joint tokens (txt_len + image + reference tokens).
    pub fn new(cfg: &Flux2Config, ts: &Tensors, gpu: Gpu, n_max: u32) -> Flux2Model {
        assert!(!cfg.guidance_embed, "guidance-embedded variants not supported");
        let d = cfg.hidden;
        let mlp = cfg.mlp_hidden();
        let hd = cfg.head_dim();
        let nh = cfg.n_heads as u32;
        let get = |name: &str| -> &(Vec<usize>, Vec<f32>) {
            ts.get(name).unwrap_or_else(|| panic!("flux2: missing tensor {name}"))
        };
        // Periodic poll_wait during the multi-GB weight upload: wgpu holds a
        // staging copy per `write` until a blocking poll reclaims them; on a
        // non-ReBAR card the un-reclaimed staging OOMs the device (observed
        // 22 GiB for 15.5 GiB of weights on a P40 — zimage's dev.rs documents
        // the same). Flush roughly every GiB.
        let uploaded = std::cell::Cell::new(0u64);
        let upv = |w: &[f32]| -> DeviceBuffer {
            let b = gpu.storage(w.len() as u64);
            gpu.write(&b, bytemuck::cast_slice(w));
            uploaded.set(uploaded.get() + 4 * w.len() as u64);
            if uploaded.get() > (1 << 30) {
                // force a real flush: a readback drains the queue (an empty
                // submit records nothing) and the poll reclaims the staging
                // wgpu holds per write — without this a non-ReBAR card OOMs
                // at ~22 GiB for 15.5 GiB of weights
                let _ = gpu.read(&b, 1);
                uploaded.set(0);
            }
            b
        };
        let up = |name: &str| -> DeviceBuffer { upv(&get(name).1) };

        let stream = |p: &str| -> StreamW {
            let (_, qkv) = get(&format!("{p}_attn.qkv.weight"));
            let (_, m0) = get(&format!("{p}_mlp.0.weight"));
            StreamW {
                wq: upv(rows(qkv, d, 0, d)),
                wk: upv(rows(qkv, d, d, 2 * d)),
                wv: upv(rows(qkv, d, 2 * d, 3 * d)),
                nq: up(&format!("{p}_attn.norm.query_norm.scale")),
                nk: up(&format!("{p}_attn.norm.key_norm.scale")),
                wo: up(&format!("{p}_attn.proj.weight")),
                // SwiGLU chunk order: x1 (silu-gated) is the FIRST half
                w1: upv(rows(m0, d, 0, mlp)),
                w3: upv(rows(m0, d, mlp, 2 * mlp)),
                w2: up(&format!("{p}_mlp.2.weight")),
            }
        };
        let dbl: Vec<(StreamW, StreamW)> = (0..cfg.depth_double)
            .map(|b| {
                (stream(&format!("double_blocks.{b}.img")), stream(&format!("double_blocks.{b}.txt")))
            })
            .collect();

        let sgl: Vec<SingleW> = (0..cfg.depth_single)
            .map(|b| {
                let p = format!("single_blocks.{b}");
                let (_, l1) = get(&format!("{p}.linear1.weight"));
                let (_, l2) = get(&format!("{p}.linear2.weight"));
                // linear2 is [D, D+mlp]; split its input (column) dim
                let mut wo_a = Vec::with_capacity(d * d);
                let mut wo_b = Vec::with_capacity(d * mlp);
                for r in 0..d {
                    wo_a.extend_from_slice(&l2[r * (d + mlp)..r * (d + mlp) + d]);
                    wo_b.extend_from_slice(&l2[r * (d + mlp) + d..(r + 1) * (d + mlp)]);
                }
                SingleW {
                    wq: upv(rows(l1, d, 0, d)),
                    wk: upv(rows(l1, d, d, 2 * d)),
                    wv: upv(rows(l1, d, 2 * d, 3 * d)),
                    nq: up(&format!("{p}.norm.query_norm.scale")),
                    nk: up(&format!("{p}.norm.key_norm.scale")),
                    w1: upv(rows(l1, d, 3 * d, 3 * d + mlp)),
                    w3: upv(rows(l1, d, 3 * d + mlp, 3 * d + 2 * mlp)),
                    wo_a: upv(&wo_a),
                    wo_b: upv(&wo_b),
                }
            })
            .collect();

        let n = n_max as u64;
        let du = d as u64;
        let mlpu = mlp as u64;
        let fast = gpu.caps().workgroup_reductions;
        let attn_mat = if fast { 1 } else { (nh as u64) * n * n };
        let a = |len: u64| gpu.storage(len);
        let scr = Scratch {
            x0: a(n * du),
            x1: a(n * du),
            n1: a(n * du),
            q: a(n * du),
            k: a(n * du),
            v: a(n * du),
            qn: a(n * du),
            kn: a(n * du),
            qr: a(n * du),
            kr: a(n * du),
            qkv: a(n * 3 * du),
            ctx: a(n * du),
            proj: a(n * du),
            h1: a(n * mlpu),
            h2: a(n * mlpu),
            hs: a(n * mlpu),
            mlp: a(n * du),
            scores: a(attn_mat),
            probs: a(attn_mat),
            out: a(n * cfg.in_channels as u64),
            cos: a(n * (hd as u64 / 2)),
            sin: a(n * (hd as u64 / 2)),
            tok_in: a(n * cfg.in_channels as u64),
            ctx_in: a(cfg.txt_len as u64 * cfg.context_in_dim as u64),
        };
        let modb = ModBufs {
            gamma: (0..6).map(|_| a(du)).collect(),
            beta: (0..6).map(|_| a(du)).collect(),
            gate: (0..5).map(|_| a(du)).collect(),
        };

        Flux2Model {
            cfg: cfg.clone(),
            n_max,
            fast,
            dbl,
            sgl,
            img_in: up("img_in.weight"),
            txt_in: up("txt_in.weight"),
            final_w: up("final_layer.linear.weight"),
            modb,
            scr,
            time_in_a: get("time_in.in_layer.weight").1.clone(),
            time_in_b: get("time_in.out_layer.weight").1.clone(),
            mod_img: get("double_stream_modulation_img.lin.weight").1.clone(),
            mod_txt: get("double_stream_modulation_txt.lin.weight").1.clone(),
            mod_single: get("single_stream_modulation.lin.weight").1.clone(),
            final_adaln: get("final_layer.adaLN_modulation.1.weight").1.clone(),
            gpu,
        }
    }

    /// `timestep_embedding(t·1000, 256)`: 128 freqs, **cos first**, then sin.
    fn timestep_embedding(t: f32) -> Vec<f32> {
        let half = 128usize;
        let x = t * 1000.0;
        let mut emb = vec![0.0f32; 256];
        for i in 0..half {
            let freq = (-(10000.0f64.ln()) * i as f64 / half as f64).exp();
            let arg = x as f64 * freq;
            emb[i] = arg.cos() as f32;
            emb[half + i] = arg.sin() as f32;
        }
        emb
    }

    /// Host conditioning: timestep MLP + the three global modulation linears,
    /// folded into (gamma, beta, gate) vectors and uploaded.
    fn upload_modulation(&self, t: f32) {
        let d = self.cfg.hidden;
        use model::hostmath::{matvec_par, silu_slice};
        let emb = Self::timestep_embedding(t);
        let h = silu_slice(&matvec_par(&self.time_in_a, &emb, d, 256));
        let vec_ = matvec_par(&self.time_in_b, &h, d, d);
        let sv = silu_slice(&vec_);

        let m_img = matvec_par(&self.mod_img, &sv, 6 * d, d);
        let m_txt = matvec_par(&self.mod_txt, &sv, 6 * d, d);
        let m_sgl = matvec_par(&self.mod_single, &sv, 3 * d, d);
        let m_fin = matvec_par(&self.final_adaln, &sv, 2 * d, d);

        // chunk order per triple: (shift, scale, gate); final layer: (shift, scale)
        let wf = |buf: &DeviceBuffer, v: &[f32]| self.gpu.write(buf, bytemuck::cast_slice(v));
        let gamma = |m: &[f32], c: usize| -> Vec<f32> {
            m[(3 * c + 1) * d..(3 * c + 2) * d].iter().map(|s| 1.0 + s).collect()
        };
        let beta = |m: &[f32], c: usize| m[3 * c * d..(3 * c + 1) * d].to_vec();
        let gate = |m: &[f32], c: usize| m[(3 * c + 2) * d..(3 * c + 3) * d].to_vec();

        for (i, m) in [&m_img, &m_txt].iter().enumerate() {
            wf(&self.modb.gamma[2 * i], &gamma(m, 0));
            wf(&self.modb.beta[2 * i], &beta(m, 0));
            wf(&self.modb.gate[2 * i], &gate(m, 0));
            wf(&self.modb.gamma[2 * i + 1], &gamma(m, 1));
            wf(&self.modb.beta[2 * i + 1], &beta(m, 1));
            wf(&self.modb.gate[2 * i + 1], &gate(m, 1));
        }
        wf(&self.modb.gamma[4], &gamma(&m_sgl, 0));
        wf(&self.modb.beta[4], &beta(&m_sgl, 0));
        wf(&self.modb.gate[4], &gate(&m_sgl, 0));
        let fin_gamma: Vec<f32> = m_fin[d..2 * d].iter().map(|s| 1.0 + s).collect();
        wf(&self.modb.gamma[5], &fin_gamma);
        wf(&self.modb.beta[5], &m_fin[..d]);
    }

    fn mm(&self, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
        if self.fast {
            self.gpu.step(K_MATMUL_REG2, &[x, w, o], &[m, k, n], m.div_ceil(128) * n.div_ceil(128) * 256)
        } else {
            self.gpu.step(K_MATMUL, &[x, w, o], &[m, k, n], m * n)
        }
    }

    /// Sliced matmul: read rows `r0..r1` of `x`, write rows `r0..r1` of `o`
    /// (both `[.., k]` / `[.., n]` row-major).
    #[allow(clippy::too_many_arguments)]
    fn mm_rows(&self, x: &DeviceBuffer, w: &DeviceBuffer, o: &DeviceBuffer, r0: u32, r1: u32, k: u32, n: u32) -> Step {
        let m = r1 - r0;
        let xo = (r0 as u64 * k as u64, m as u64 * k as u64);
        let oo = (r0 as u64 * n as u64, m as u64 * n as u64);
        if self.fast {
            self.gpu.step_sliced(K_MATMUL_REG2, &[x, w, o], &[xo, (0, 0), oo], &[m, k, n], m.div_ceil(128) * n.div_ceil(128) * 256)
        } else {
            self.gpu.step_sliced(K_MATMUL, &[x, w, o], &[xo, (0, 0), oo], &[m, k, n], m * n)
        }
    }

    /// Modulated LayerNorm over rows `r0..r1`: `LN_noaffine·gamma + beta`.
    fn ln_rows(&self, x: &DeviceBuffer, site: usize, o: &DeviceBuffer, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        self.gpu.step_sliced(
            K_LN,
            &[x, &self.modb.gamma[site], &self.modb.beta[site], o],
            &[off, (0, 0), (0, 0), off],
            &[d, m, f(EPS)],
            m,
        )
    }

    /// Gated residual over rows `r0..r1`: `y = x + gate ⊙ h` (whole-range cond).
    fn gate_rows(&self, x: &DeviceBuffer, gi: usize, h: &DeviceBuffer, y: &DeviceBuffer, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        self.gpu.step_sliced(
            K_GATE,
            &[x, &self.modb.gate[gi], h, y],
            &[off, (0, 0), off, off],
            &[m, d, m],
            m * d,
        )
    }

    /// QK-RMSNorm over rows `r0..r1` (per-head rows of length `head_dim`).
    fn qknorm_rows(&self, x: &DeviceBuffer, scale: &DeviceBuffer, o: &DeviceBuffer, r0: u32, r1: u32) -> Step {
        let d = self.cfg.hidden as u32;
        let hd = self.cfg.head_dim() as u32;
        let nh = self.cfg.n_heads as u32;
        let m = r1 - r0;
        let off = (r0 as u64 * d as u64, m as u64 * d as u64);
        self.gpu.step_sliced(
            K_RMSNORM,
            &[x, scale, o],
            &[off, (0, 0), off],
            &[hd, m * nh, f(EPS)],
            m * nh,
        )
    }

    fn push_attention(&self, s: &mut Vec<Step>, n: u32) {
        let nh = self.cfg.n_heads as u32;
        let hd = self.cfg.head_dim() as u32;
        let dim = self.cfg.hidden as u32;
        let scr = &self.scr;
        if self.fast {
            let br = 64u32; // must match BR in flash_attn_bidir.wgsl
            let nwg = nh * n.div_ceil(br);
            s.push(self.gpu.step(K_FLASH, &[&scr.qkv, &scr.ctx], &[1, nh, n, hd, 3 * dim, 0, dim, 2 * dim, dim], nwg * br));
        } else {
            s.push(self.gpu.step(K_SCORES, &[&scr.qkv, &scr.scores], &[1, nh, n, hd, 3 * dim, 0, dim], nh * n * n));
            s.push(self.gpu.step(K_SOFTMAX, &[&scr.scores, &scr.probs], &[1, nh, n], nh * n));
            s.push(self.gpu.step(K_APPLY, &[&scr.probs, &scr.qkv, &scr.ctx], &[1, nh, n, hd, 3 * dim, 2 * dim, dim], nh * n * hd));
        }
    }

    /// Attention core shared by both block kinds: qkv is already in
    /// `scr.q/k/v` (rope'd + packed here), result lands in `scr.ctx`.
    fn push_attn_core(&self, s: &mut Vec<Step>, n: u32) {
        let d = self.cfg.hidden as u32;
        let hd = self.cfg.head_dim() as u32;
        let nh = self.cfg.n_heads as u32;
        let half = hd / 2;
        let scr = &self.scr;
        s.push(self.gpu.step(K_ROPE, &[&scr.qn, &scr.cos, &scr.sin, &scr.qr], &[n, nh, hd, half], n * nh * half));
        s.push(self.gpu.step(K_ROPE, &[&scr.kn, &scr.cos, &scr.sin, &scr.kr], &[n, nh, hd, half], n * nh * half));
        s.push(self.gpu.step(K_PACK, &[&scr.qr, &scr.kr, &scr.v, &scr.qkv], &[n, d], n * 3 * d));
        self.push_attention(s, n);
    }

    /// Forward one denoising evaluation.
    ///
    /// `img_tokens`: packed latent tokens `[n_img, in_channels]` (noise image
    /// first, then any reference tokens). `ctx`: text conditioning
    /// `[txt_len, context_in_dim]`. `ids`: joint 4-axis position ids, **text
    /// rows first** then image/ref rows (`(txt_len + n_img) * 4`). Returns the
    /// prediction for the first `n_pred` image tokens `[n_pred, in_channels]`.
    pub fn forward(&self, img_tokens: &[f32], ctx: &[f32], t: f32, ids: &[u32], n_pred: usize) -> Vec<f32> {
        let cfg = &self.cfg;
        let d = cfg.hidden as u32;
        let mlp = cfg.mlp_hidden() as u32;
        let cin = cfg.in_channels as u32;
        let nt = cfg.txt_len as u32;
        let ni = (img_tokens.len() / cfg.in_channels) as u32;
        let n = nt + ni;
        assert!(n <= self.n_max, "sized for {} joint tokens, got {n}", self.n_max);
        assert_eq!(ctx.len(), cfg.txt_len * cfg.context_in_dim);
        assert_eq!(ids.len() as u32, n * 4);
        assert!(n_pred as u32 <= ni);

        self.upload_modulation(t);

        // RoPE tables from the joint ids (t, h, w, l), interleaved pairs.
        let rc = dit::rope::RopeConfig {
            axes_dims: cfg.axes_dim.iter().map(|&a| a as u32).collect(),
            axes_lens: vec![4096, 4096, 4096, 4096],
            theta: cfg.rope_theta,
        };
        let tables = dit::rope::tables_for_ids(&rc, ids, 4);
        self.gpu.write(&self.scr.cos, bytemuck::cast_slice(&tables.cos));
        self.gpu.write(&self.scr.sin, bytemuck::cast_slice(&tables.sin));

        self.gpu.write(&self.scr.tok_in, bytemuck::cast_slice(img_tokens));
        self.gpu.write(&self.scr.ctx_in, bytemuck::cast_slice(ctx));

        let scr = &self.scr;
        let mut s: Vec<Step> = Vec::new();
        // embed both streams into the joint residual slab x0 = [txt | img]
        s.push(self.mm_rows(&scr.ctx_in, &self.txt_in, &scr.x0, 0, nt, cfg.context_in_dim as u32, d));
        // img rows: input read starts at tok_in row 0, output lands at row nt
        {
            let xo = (0u64, ni as u64 * cin as u64);
            let oo = (nt as u64 * d as u64, ni as u64 * d as u64);
            let st = if self.fast {
                self.gpu.step_sliced(K_MATMUL_REG2, &[&scr.tok_in, &self.img_in, &scr.x0], &[xo, (0, 0), oo], &[ni, cin, d], ni.div_ceil(128) * d.div_ceil(128) * 256)
            } else {
                self.gpu.step_sliced(K_MATMUL, &[&scr.tok_in, &self.img_in, &scr.x0], &[xo, (0, 0), oo], &[ni, cin, d], ni * d)
            };
            s.push(st);
        }

        let (mut xa, mut xb) = (&scr.x0, &scr.x1);
        // sites: 0=img1 1=img2 2=txt1 3=txt2; gates likewise
        for (img_w, txt_w) in &self.dbl {
            // attention halves of both streams into the joint q/k/v
            s.push(self.ln_rows(xa, 2, &scr.n1, 0, nt)); // txt norm1
            s.push(self.mm_rows(&scr.n1, &txt_w.wq, &scr.q, 0, nt, d, d));
            s.push(self.mm_rows(&scr.n1, &txt_w.wk, &scr.k, 0, nt, d, d));
            s.push(self.mm_rows(&scr.n1, &txt_w.wv, &scr.v, 0, nt, d, d));
            s.push(self.ln_rows(xa, 0, &scr.n1, nt, n)); // img norm1
            s.push(self.mm_rows(&scr.n1, &img_w.wq, &scr.q, nt, n, d, d));
            s.push(self.mm_rows(&scr.n1, &img_w.wk, &scr.k, nt, n, d, d));
            s.push(self.mm_rows(&scr.n1, &img_w.wv, &scr.v, nt, n, d, d));
            s.push(self.qknorm_rows(&scr.q, &txt_w.nq, &scr.qn, 0, nt));
            s.push(self.qknorm_rows(&scr.k, &txt_w.nk, &scr.kn, 0, nt));
            s.push(self.qknorm_rows(&scr.q, &img_w.nq, &scr.qn, nt, n));
            s.push(self.qknorm_rows(&scr.k, &img_w.nk, &scr.kn, nt, n));
            self.push_attn_core(&mut s, n);
            // per-stream projection + gated residual
            s.push(self.mm_rows(&scr.ctx, &txt_w.wo, &scr.proj, 0, nt, d, d));
            s.push(self.mm_rows(&scr.ctx, &img_w.wo, &scr.proj, nt, n, d, d));
            s.push(self.gate_rows(xa, 2, &scr.proj, xb, 0, nt));
            s.push(self.gate_rows(xa, 0, &scr.proj, xb, nt, n));
            std::mem::swap(&mut xa, &mut xb);
            // MLP halves
            s.push(self.ln_rows(xa, 3, &scr.n1, 0, nt)); // txt norm2
            s.push(self.mm_rows(&scr.n1, &txt_w.w1, &scr.h1, 0, nt, d, mlp));
            s.push(self.mm_rows(&scr.n1, &txt_w.w3, &scr.h2, 0, nt, d, mlp));
            s.push(self.ln_rows(xa, 1, &scr.n1, nt, n)); // img norm2
            s.push(self.mm_rows(&scr.n1, &img_w.w1, &scr.h1, nt, n, d, mlp));
            s.push(self.mm_rows(&scr.n1, &img_w.w3, &scr.h2, nt, n, d, mlp));
            s.push(self.gpu.step(K_SILU_MUL, &[&scr.h1, &scr.h2, &scr.hs], &[n * mlp], n * mlp));
            s.push(self.mm_rows(&scr.hs, &txt_w.w2, &scr.mlp, 0, nt, mlp, d));
            s.push(self.mm_rows(&scr.hs, &img_w.w2, &scr.mlp, nt, n, mlp, d));
            s.push(self.gate_rows(xa, 3, &scr.mlp, xb, 0, nt));
            s.push(self.gate_rows(xa, 1, &scr.mlp, xb, nt, n));
            std::mem::swap(&mut xa, &mut xb);
        }

        for w in &self.sgl {
            // parallel attn ‖ MLP over one shared modulated norm
            s.push(self.ln_rows(xa, 4, &scr.n1, 0, n));
            s.push(self.mm(&scr.n1, &w.wq, &scr.q, n, d, d));
            s.push(self.mm(&scr.n1, &w.wk, &scr.k, n, d, d));
            s.push(self.mm(&scr.n1, &w.wv, &scr.v, n, d, d));
            s.push(self.qknorm_rows(&scr.q, &w.nq, &scr.qn, 0, n));
            s.push(self.qknorm_rows(&scr.k, &w.nk, &scr.kn, 0, n));
            self.push_attn_core(&mut s, n);
            s.push(self.mm(&scr.n1, &w.w1, &scr.h1, n, d, mlp));
            s.push(self.mm(&scr.n1, &w.w3, &scr.h2, n, d, mlp));
            s.push(self.gpu.step(K_SILU_MUL, &[&scr.h1, &scr.h2, &scr.hs], &[n * mlp], n * mlp));
            // linear2 over cat(attn, mlp): two column-split matmuls, summed
            s.push(self.mm(&scr.ctx, &w.wo_a, &scr.proj, n, d, d));
            s.push(self.mm(&scr.hs, &w.wo_b, &scr.mlp, n, mlp, d));
            // y = x + gate ⊙ proj ; then y += gate ⊙ mlp (two gated adds)
            s.push(self.gate_rows(xa, 4, &scr.proj, xb, 0, n));
            std::mem::swap(&mut xa, &mut xb);
            s.push(self.gate_rows(xa, 4, &scr.mlp, xb, 0, n));
            std::mem::swap(&mut xa, &mut xb);
        }

        // final layer on the predicted image rows only
        let p0 = nt;
        let p1 = nt + n_pred as u32;
        s.push(self.ln_rows(xa, 5, &scr.n1, p0, p1));
        {
            let xo = (p0 as u64 * d as u64, n_pred as u64 * d as u64);
            let oo = (0u64, n_pred as u64 * cin as u64);
            let m = n_pred as u32;
            let st = if self.fast {
                self.gpu.step_sliced(K_MATMUL_REG2, &[&scr.n1, &self.final_w, &scr.out], &[xo, (0, 0), oo], &[m, d, cin], m.div_ceil(128) * cin.div_ceil(128) * 256)
            } else {
                self.gpu.step_sliced(K_MATMUL, &[&scr.n1, &self.final_w, &scr.out], &[xo, (0, 0), oo], &[m, d, cin], m * cin)
            };
            s.push(st);
        }

        // debug aid: SMOKE_STEPS=k submits only the first k steps
        let take = std::env::var("SMOKE_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(s.len());
        self.gpu.submit(&[], &s[..take.min(s.len())]);
        self.gpu.read(&self.scr.out, n_pred * cfg.in_channels)
    }
}

/// Joint 4-axis position ids in the reference layout, text rows first.
///
/// Text tokens: `(0,0,0,l)`; generated image: `(0,h,w,0)` raster-major; each
/// reference image i: `(10·(i+1), h, w, 0)`. `refs` are (height, width) in
/// latent-token units.
pub fn position_ids(txt_len: usize, lh: usize, lw: usize, refs: &[(usize, usize)]) -> Vec<u32> {
    let mut ids = Vec::with_capacity((txt_len + lh * lw) * 4);
    for l in 0..txt_len {
        ids.extend([0, 0, 0, l as u32]);
    }
    for h in 0..lh {
        for w in 0..lw {
            ids.extend([0, h as u32, w as u32, 0]);
        }
    }
    for (i, &(rh, rw)) in refs.iter().enumerate() {
        let t = 10 * (i as u32 + 1);
        for h in 0..rh {
            for w in 0..rw {
                ids.extend([t, h as u32, w as u32, 0]);
            }
        }
    }
    ids
}
