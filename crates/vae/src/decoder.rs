// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `AutoencoderKL` decoder as a pre-recorded brain kernel graph.
//!
//! Mirrors the diffusers `Decoder`: [1×1 `post_quant_conv` when the config
//! enables it — FLUX.2] → `conv_in` → mid block (resnet, self-attn, resnet) →
//! up blocks (`layers_per_block+1` resnets each, nearest-2× upsample + conv on
//! all but the last block) → `conv_norm_out` → SiLU → `conv_out`.
//! The graph is built once for a fixed input `[latent_ch, h, w]`; `decode`
//! uploads the latent, submits, and reads the image `[out_ch, H, W]`.
//!
//! Reuse: identical conv/GroupNorm/SiLU/add/upsample/attention kernels as the
//! DIAMOND UNet (`crates/wm-diamond/src/model.rs`), which validates them. VAE
//! specifics vs DIAMOND: static (non-conditioned) affine GroupNorm with **32
//! groups** and **eps 1e-6**; the mid-block attention is **single-head**
//! (`head_dim = C`, scale `1/√C`) with the residual added to the **pre-norm**
//! input; and `to_q/to_k/to_v` are fused into one 1×1 qkv conv at build time so
//! the exact DIAMOND attention path (`nchw_nlc` → bidir scores/softmax/apply →
//! `nlc_nchw`) applies unchanged.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use std::collections::HashMap;

use crate::config::VaeConfig;

// Kernel-table indices (order matches KERNELS).
const K_CONV: usize = 0;
const K_GN_STATS: usize = 1;
const K_GN_APPLY: usize = 2;
const K_SILU: usize = 3;
const K_ADD2: usize = 4;
const K_UPSAMPLE2: usize = 5;
const K_NCHW_NLC: usize = 6;
const K_NLC_NCHW: usize = 7;
const K_ATTN_SCORES: usize = 8;
const K_ATTN_SOFTMAX: usize = 9;
const K_ATTN_APPLY: usize = 10;
const K_GN_STATS_WG: usize = 11;
const K_MATMUL: usize = 12;
const K_IM2COL_AT: usize = 13;
const K_NLC_BIAS_NCHW: usize = 14;

/// The decoder/encoder graph's kernel set, in slot order. Public so a profiler
/// can name the kernel behind each recorded [`Step`] (`flux2_bench vae`).
pub const KERNELS: [(&str, &str); 15] = [
    ("conv_bias_reg", kernels::CONV_BIAS_REG),
    ("gn_stats", kernels::GN_STATS),
    ("gn_apply", kernels::GN_APPLY),
    ("silu", kernels::SILU),
    ("add2", kernels::ADD2),
    ("upsample2", kernels::UPSAMPLE2),
    ("nchw_nlc", kernels::NCHW_NLC),
    ("nlc_nchw", kernels::NLC_NCHW),
    ("attn_scores_bidir", kernels::ATTN_SCORES_BIDIR),
    ("attn_softmax_bidir", kernels::ATTN_SOFTMAX_BIDIR),
    ("attn_apply_bidir", kernels::ATTN_APPLY_BIDIR),
    ("gn_stats_wg", kernels::GN_STATS_WG),
    ("matmul_reg3", kernels::MATMUL_REG3),
    ("im2col_at", kernels::IM2COL_AT),
    ("nlc_bias_nchw", kernels::NLC_BIAS_NCHW),
];

/// Host tensors by name (diffusers key, e.g. `decoder.conv_in.weight`) →
/// `(shape, row-major f32 data)`.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// A decode graph for a fixed input latent size, with weights resident.
pub struct VaeDecoder {
    gpu: Gpu,
    cfg: VaeConfig,
    steps: Vec<Step>,
    z_in: DeviceBuffer,
    out: DeviceBuffer,
    out_len: usize,
    taps: Vec<(String, DeviceBuffer, usize)>,
}

/// Graph-construction state (borrows the device + host tensors).
struct Builder<'a> {
    gpu: &'a Gpu,
    t: &'a Tensors,
    eps: f32,
    groups: u32,
    steps: Vec<Step>,
    taps: Vec<(String, DeviceBuffer, usize)>,
    /// Free-list of activation buffers keyed by exact length (words). An `act(len)`
    /// reuses a buffer of the same length whose last read is already recorded, so
    /// the resident peak is the max *concurrently-live* activation set instead of
    /// the sum of every activation — the difference between decoding 640² and 1536²
    /// on a 24 GB card. Reuse is bit-exact: the graph runs its steps in order with
    /// barriers (as the qwen/zimage scratch reuse relies on), and a buffer is only
    /// freed after its last consumer step is emitted, so the reusing write always
    /// follows the last read. Disabled when `taps_on` (taps pin buffers).
    pool: std::collections::HashMap<u64, Vec<DeviceBuffer>>,
    /// Record intermediate taps (for parity debugging via `read_tap`). Off by
    /// default — pins buffers and defeats pooling. Enable with `BRAIN_VAE_TAPS=1`.
    taps_on: bool,
    /// The device executes workgroup-cooperative reductions (barriers): pick the
    /// workgroup-per-group GroupNorm statistics kernel and the conv/attention
    /// GEMM lowerings. False on the CPU JIT, which keeps the reference kernels
    /// (whose native AVX2 fast paths are the fast CPU route anyway).
    coop: bool,
    /// The single im2col scratch (`length, buffer`) shared by every lowered
    /// conv, grown on demand. Bounded by [`COL_BUDGET_FLOATS`] — a whole-image
    /// im2col operand exceeds the P40's 2047 MiB binding limit, so the GEMM is
    /// chunked over spatial positions instead (see `im2col_at.wgsl`).
    col: Option<(u64, DeviceBuffer)>,
}

/// Ceiling on the im2col scratch, in f32 words (512 MiB). The lowered conv
/// processes `floor(budget / CinKK)` output positions per GEMM, so this trades
/// scratch for the number of chunks; at 512² the largest operand would be
/// 2.4 GB unchunked, which is both unbindable and hostile to a card shared with
/// a resident DiT. Override with `BRAIN_VAE_COL_MIB`.
fn col_budget_floats() -> u64 {
    let mib: u64 = std::env::var("BRAIN_VAE_COL_MIB").ok().and_then(|v| v.parse().ok()).unwrap_or(512);
    mib * 1024 * 1024 / 4
}

/// Minimum output channels for the lowered conv: `matmul_reg3` computes a
/// 128-wide column tile, so a conv with fewer output channels pays for a full
/// tile and wins nothing (the FLUX.2 `conv_out`, Cout = 3, is 42x wasted). It
/// stays on the direct register-tiled conv.
const GEMM_CONV_MIN_COUT: u32 = 128;

impl<'a> Builder<'a> {
    fn get(&self, name: &str) -> &(Vec<usize>, Vec<f32>) {
        self.t.get(name).unwrap_or_else(|| panic!("vae: missing tensor {name}"))
    }
    fn dev(&self, name: &str) -> DeviceBuffer {
        self.gpu.storage_init(name, &self.get(name).1)
    }
    /// Allocate an activation buffer of `len` words, reusing a same-length freed
    /// buffer from the pool when one is available (see [`Builder::pool`]).
    fn act(&mut self, len: u64) -> DeviceBuffer {
        if let Some(b) = self.pool.get_mut(&len).and_then(Vec::pop) {
            return b;
        }
        self.gpu.storage(len)
    }
    /// Return an activation buffer to the pool for reuse. MUST be called only after
    /// the buffer's last read step has been pushed (else a later reuse would clobber
    /// data a pending step still needs). No-op when pooling is disabled.
    fn free(&mut self, len: u64, buf: DeviceBuffer) {
        if !self.taps_on {
            self.pool.entry(len).or_default().push(buf);
        }
    }
    fn tap(&mut self, name: String, buf: &DeviceBuffer, len: u32) {
        if self.taps_on {
            self.taps.push((name, buf.clone(), len as usize));
        }
    }

    /// Conv (+bias) `prefix.{weight,bias}`: `x[cin,h,w] → y[cout,ho,wo]`.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        k: u32,
        pad: u32,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let ho = (h + 2 * pad - k) + 1;
        let wo = (w + 2 * pad - k) + 1;
        self.conv_s(prefix, cin, cout, k, 1, pad, h, w, ho, wo, x)
    }

    /// diffusers `Downsample2D` (`use_conv`, `padding=0`): F.pad(x,(0,1,0,1)) then
    /// a stride-2, k=3, pad=0 conv → `[c, h/2, w/2]`. The right/bottom zero-pad is
    /// reproduced by forcing `ho=wo=h/2` with `pad=0`: the kernels bounds-check
    /// their reads, so the extra bottom/right taps read 0 — exactly the
    /// asymmetric pad.
    fn conv_down(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        self.conv_s(prefix, c, c, 3, 2, 0, h, w, h / 2, w / 2, x)
    }

    /// The im2col scratch, grown on demand and shared by every lowered conv.
    /// An outgrown buffer goes back to the activation pool (its last read is
    /// already recorded, which is exactly the pool's reuse contract).
    fn col_buf(&mut self, need: u64) -> DeviceBuffer {
        if let Some((len, b)) = &self.col {
            if *len >= need {
                return b.clone();
            }
        }
        if let Some((len, b)) = self.col.take() {
            self.free(len, b);
        }
        let b = self.act(need);
        self.col = Some((need, b.clone()));
        b
    }

    /// Conv with an explicit stride and output size. Two lowerings:
    ///
    /// * **direct** — `conv_bias_reg`, the 8x4 register-tiled kernel. Measured
    ///   on a P40 across every FLUX.2 VAE decode shape: a flat **~700 GFLOP/s,
    ///   6% of the card's fp32 peak**, and it was 3535 ms of a 3600 ms decode.
    ///   Its ceiling is structural — 12 global loads per 32 FMAs is
    ///   0.75 byte/FLOP, a 461 GFLOP/s roofline that caching stretches to ~700.
    /// * **lowered** (`self.coop`, `cout >= GEMM_CONV_MIN_COUT`) — `im2col_at` +
    ///   `matmul_reg3` + `nlc_bias_nchw`, i.e. `y[HW, Cout] = col[HW, CinKK] ·
    ///   Wᵀ`, which runs at the GEMM's ~34% of peak. This is the trade
    ///   `docs/performance/overview.md` scoped to "a compute-bound discrete
    ///   GPU" and `docs/performance/p40.md` already took for YOLO's convs; the
    ///   P40 is that GPU. The transposed orientation (positions as GEMM ROWS)
    ///   is what makes it chunkable: a spatial chunk is a contiguous row range
    ///   of both `col` and the output, so the 2.4 GB whole-image operand
    ///   becomes a bounded scratch (see `im2col_at.wgsl`).
    #[allow(clippy::too_many_arguments)]
    fn conv_s(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        k: u32,
        stride: u32,
        pad: u32,
        h: u32,
        w: u32,
        ho: u32,
        wo: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let wgt = self.dev(&format!("{prefix}.weight"));
        let bias = self.dev(&format!("{prefix}.bias"));
        let y = self.act((cout * ho * wo) as u64);
        let hw = ho * wo;
        let cinkk = cin * k * k;
        if self.coop && cout >= GEMM_CONV_MIN_COUT && hw >= 128 {
            // Positions per GEMM: a multiple of the 128-row tile, inside the
            // scratch budget, at least one tile.
            let budget = col_budget_floats();
            let chunk = (((budget / cinkk as u64) / 128) * 128).clamp(128, hw as u64) as u32;
            let col = self.col_buf(chunk as u64 * cinkk as u64);
            let nhwc = self.act((hw * cout) as u64);
            let mut pos = 0u32;
            while pos < hw {
                let cnt = chunk.min(hw - pos);
                self.steps.push(self.gpu.step(
                    K_IM2COL_AT,
                    &[x, &col],
                    &[cin, h, w, k, stride, pad, ho, wo, cinkk, pos, cnt],
                    cnt * cinkk,
                ));
                self.steps.push(self.gpu.step_sliced(
                    K_MATMUL,
                    &[&col, &wgt, &nhwc],
                    &[(0, 0), (0, 0), (pos as u64 * cout as u64, cnt as u64 * cout as u64)],
                    &[cnt, cinkk, cout],
                    cnt.div_ceil(128) * cout.div_ceil(128) * 256,
                ));
                pos += cnt;
            }
            self.steps.push(self.gpu.step(
                K_NLC_BIAS_NCHW,
                &[&nhwc, &bias, &y],
                &[hw * cout, cout, hw],
                cout.div_ceil(64) * hw.div_ceil(64) * 64,
            ));
            self.free((hw * cout) as u64, nhwc);
        } else {
            let threads = cout.div_ceil(8) * hw.div_ceil(4);
            self.steps.push(self.gpu.step(
                K_CONV,
                &[x, &wgt, &bias, &y],
                &[1, cin, h, w, cout, k, stride, pad, ho, wo],
                threads,
            ));
        }
        y
    }

    /// Static affine GroupNorm from `prefix.{weight,bias}` (32 groups, eps
    /// 1e-6): `y = gamma·(x-μ)/σ + beta` per group. `gb = [gamma‖beta]`.
    fn gn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let (_, gamma) = self.get(&format!("{prefix}.weight"));
        let (_, beta) = self.get(&format!("{prefix}.bias"));
        let mut gbv = gamma.clone();
        gbv.extend_from_slice(beta);
        let gb = self.gpu.storage_init(&format!("{prefix}.gb"), &gbv);
        let g = self.groups;
        let stats = self.act(2 * g as u64);
        let y = self.act((c * h * w) as u64);
        // Statistics: one WORKGROUP per group where the device can run a
        // workgroup reduction (`gn_stats_wg`), else the per-group reference
        // kernel. `gn_stats` dispatches `g` = 32 *invocations* for up to 33 M
        // elements — measured at 35% of a 512² FLUX.2 VAE decode; the
        // cooperative kernel is the same two-pass math, coalesced and 32-way
        // parallel (see `gn_stats_wg.wgsl`).
        if self.coop {
            self.steps.push(self.gpu.step(
                K_GN_STATS_WG,
                &[x, &stats],
                &[1, c, h, w, g, f(self.eps)],
                g * 256,
            ));
        } else {
            self.steps.push(self.gpu.step(
                K_GN_STATS,
                &[x, &stats],
                &[1, c, h, w, g, f(self.eps)],
                g,
            ));
        }
        self.steps.push(self.gpu.step(
            K_GN_APPLY,
            &[x, &stats, &gb, &y],
            &[1, c, h, w, g],
            c * h * w,
        ));
        self.free(2 * g as u64, stats); // last read was GN_APPLY above
        y
    }

    fn silu(&mut self, n: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        self.steps.push(self.gpu.step(K_SILU, &[x, &y], &[n], n));
        y
    }

    fn add(&mut self, n: u32, a: &DeviceBuffer, b: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act(n as u64);
        self.steps.push(self.gpu.step(K_ADD2, &[a, b, &y], &[n], n));
        y
    }

    /// Nearest-neighbour 2× upsample: `[c,h,w] → [c,2h,2w]`.
    fn upsample(&mut self, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let y = self.act((c * 2 * h * 2 * w) as u64);
        self.steps.push(self.gpu.step(K_UPSAMPLE2, &[x, &y], &[1, c, h, w], c * 4 * h * w));
        y
    }

    /// One diffusers `ResnetBlock2D` (no temb): `x → conv2(silu(norm2(conv1(
    /// silu(norm1(x)))))) + shortcut(x)`, shortcut = 1×1 conv when `cin≠cout`.
    fn resnet(
        &mut self,
        prefix: &str,
        cin: u32,
        cout: u32,
        h: u32,
        w: u32,
        x: &DeviceBuffer,
    ) -> DeviceBuffer {
        let (nin, nout) = ((cin * h * w) as u64, (cout * h * w) as u64);
        // `r` aliases the input `x` when cin==cout (a residual we must NOT free — the
        // caller owns `x`); when cin!=cout it is a fresh shortcut-conv buffer we own.
        let (r, r_owned) = if cin != cout {
            (self.conv(&format!("{prefix}.conv_shortcut"), cin, cout, 1, 0, h, w, x), true)
        } else {
            (x.clone(), false)
        };
        let n1 = self.gn(&format!("{prefix}.norm1"), cin, h, w, x);
        let s1 = self.silu(cin * h * w, &n1);
        self.free(nin, n1);
        let c1 = self.conv(&format!("{prefix}.conv1"), cin, cout, 3, 1, h, w, &s1);
        self.free(nin, s1);
        let n2 = self.gn(&format!("{prefix}.norm2"), cout, h, w, &c1);
        self.free(nout, c1);
        let s2 = self.silu(cout * h * w, &n2);
        self.free(nout, n2);
        let c2 = self.conv(&format!("{prefix}.conv2"), cout, cout, 3, 1, h, w, &s2);
        self.free(nout, s2);
        let out = self.add(cout * h * w, &c2, &r); // last read of c2 and r
        self.free(nout, c2);
        if r_owned {
            self.free(nout, r);
        }
        self.tap(prefix.to_string(), &out, cout * h * w);
        out
    }

    /// Mid-block single-head self-attention (diffusers `Attention`,
    /// `residual_connection=True`): `x + to_out(attn(to_qkv(group_norm(x))))`.
    /// `to_q/k/v` are fused into a single 1×1 qkv conv so the bidir attention
    /// trio applies unchanged; residual is added to the **pre-norm** input.
    fn attn(&mut self, prefix: &str, c: u32, h: u32, w: u32, x: &DeviceBuffer) -> DeviceBuffer {
        let t = h * w;
        let normed = self.gn(&format!("{prefix}.group_norm"), c, h, w, x);

        // Fuse to_q/to_k/to_v (each [C,C] linear = [C,C,1,1] conv) into one
        // [3C,C,1,1] qkv conv weight + [3C] bias.
        let (_, qw) = self.get(&format!("{prefix}.to_q.weight"));
        let (_, kw) = self.get(&format!("{prefix}.to_k.weight"));
        let (_, vw) = self.get(&format!("{prefix}.to_v.weight"));
        let mut qkv_w = Vec::with_capacity(qw.len() * 3);
        qkv_w.extend_from_slice(qw);
        qkv_w.extend_from_slice(kw);
        qkv_w.extend_from_slice(vw);
        let (_, qb) = self.get(&format!("{prefix}.to_q.bias"));
        let (_, kb) = self.get(&format!("{prefix}.to_k.bias"));
        let (_, vb) = self.get(&format!("{prefix}.to_v.bias"));
        let mut qkv_b = Vec::with_capacity(qb.len() * 3);
        qkv_b.extend_from_slice(qb);
        qkv_b.extend_from_slice(kb);
        qkv_b.extend_from_slice(vb);
        // GEMM path (below): the 1/√C attention scale lives in
        // `attn_scores_bidir`'s epilogue, and a plain GEMM has no epilogue — so
        // fold it into `to_q` instead. `q = (Wx+b)/√C = (W/√C)x + b/√C`
        // exactly; only the fp32 rounding of the two orders differs (≈1 ulp on
        // a score of O(1), invisible through the softmax).
        if self.coop {
            let sc = 1.0f32 / (c as f32).sqrt();
            for v in qkv_w[..qw.len()].iter_mut() {
                *v *= sc;
            }
            for v in qkv_b[..qb.len()].iter_mut() {
                *v *= sc;
            }
        }
        let qkv_wd = self.gpu.storage_init(&format!("{prefix}.qkv.w"), &qkv_w);
        let qkv_bd = self.gpu.storage_init(&format!("{prefix}.qkv.b"), &qkv_b);

        // qkv 1×1 conv: [C,h,w] → [3C,h,w].
        let qkv_chw = self.act((3 * c * t) as u64);
        self.steps.push(self.gpu.step(
            K_CONV,
            &[&normed, &qkv_wd, &qkv_bd, &qkv_chw],
            &[1, c, h, w, 3 * c, 1, 1, 0, h, w],
            (3 * c).div_ceil(8) * t.div_ceil(4),
        ));
        self.free((c * t) as u64, normed); // last read was the qkv conv

        let attn_rows = if self.coop {
            // ---- attention as two GEMMs -----------------------------------
            // The per-element trio gives one thread per (i,j) score, each
            // looping head_dim with its `k` reads a whole row apart: at the
            // FLUX.2 mid block (T = 4096, C = 512) `attn_scores_bidir`
            // measured **562 ms = 13% of a 512² decode** for 17 GFLOP =
            // 30 GFLOP/s, 0.26% of the P40's peak. Both contractions are plain
            // matmuls at shapes `matmul_reg3` runs at ~34% of peak, so express
            // them as such. The qkv conv already emits **channel-major**
            // [3C, T], which is qᵀ/kᵀ/vᵀ — so `v` needs no transpose at all
            // (it is directly the `[n, k]` operand of the apply GEMM), and
            // q/k need one cheap `nchw_nlc` each.
            let q_nlc = self.act((c * t) as u64);
            let k_nlc = self.act((c * t) as u64);
            for (i, dst) in [&q_nlc, &k_nlc].into_iter().enumerate() {
                let off = (i as u64) * (c * t) as u64;
                self.steps.push(self.gpu.step_sliced(
                    K_NCHW_NLC,
                    &[&qkv_chw, dst],
                    &[(off, (c * t) as u64), (0, 0)],
                    &[c * t, c, t],
                    c * t,
                ));
            }
            // scores[T,T] = q[T,C] · k[T,C]ᵀ  (the 1/√C is folded into to_q)
            let scores = self.act((t * t) as u64);
            self.steps.push(self.gpu.step(
                K_MATMUL,
                &[&q_nlc, &k_nlc, &scores],
                &[t, c, t],
                t.div_ceil(128) * t.div_ceil(128) * 256,
            ));
            self.free((c * t) as u64, q_nlc);
            self.free((c * t) as u64, k_nlc);
            let probs = self.act((t * t) as u64);
            self.steps.push(self.gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, 1, t], t));
            self.free((t * t) as u64, scores);
            // ctx[T,C] = probs[T,T] · v[T,C], with vᵀ = the third channel block
            // of the conv output, read in place as the [n=C, k=T] operand.
            let rows = self.act((t * c) as u64);
            self.steps.push(self.gpu.step_sliced(
                K_MATMUL,
                &[&probs, &qkv_chw, &rows],
                &[(0, 0), (2 * (c * t) as u64, (c * t) as u64), (0, 0)],
                &[t, t, c],
                t.div_ceil(128) * c.div_ceil(128) * 256,
            ));
            self.free((t * t) as u64, probs);
            self.free((3 * c * t) as u64, qkv_chw);
            rows
        } else {
            // NCHW [3C,h,w] → NLC rows [T, 3C].
            let qkv = self.act((3 * c * t) as u64);
            self.steps.push(self.gpu.step(K_NCHW_NLC, &[&qkv_chw, &qkv], &[3 * c * t, 3 * c, t], 3 * c * t));
            self.free((3 * c * t) as u64, qkv_chw);

            // Single head, head_dim = C, scale 1/√C (applied in the kernel).
            let scores = self.act((t * t) as u64);
            self.steps.push(self.gpu.step(
                K_ATTN_SCORES,
                &[&qkv, &scores],
                &[1, 1, t, c, 3 * c, 0, c],
                t * t,
            ));
            let probs = self.act((t * t) as u64);
            self.steps.push(self.gpu.step(K_ATTN_SOFTMAX, &[&scores, &probs], &[1, 1, t], t));
            self.free((t * t) as u64, scores);
            let rows = self.act((t * c) as u64);
            self.steps.push(self.gpu.step(
                K_ATTN_APPLY,
                &[&probs, &qkv, &rows], // last read of both probs and qkv
                &[1, 1, t, c, 3 * c, 2 * c, c],
                t * c,
            ));
            self.free((t * t) as u64, probs);
            self.free((3 * c * t) as u64, qkv);
            rows
        };
        // NLC rows [T, C] → NCHW [C,h,w].
        let attn_chw = self.act((c * t) as u64);
        self.steps.push(self.gpu.step(K_NLC_NCHW, &[&attn_rows, &attn_chw], &[c * t, c, t], c * t));
        self.free((t * c) as u64, attn_rows);

        let proj = self.conv(&format!("{prefix}.to_out.0"), c, c, 1, 0, h, w, &attn_chw);
        self.free((c * t) as u64, attn_chw);
        let out = self.add(c * h * w, x, &proj); // x is the residual input (caller-owned)
        self.free((c * h * w) as u64, proj);
        out
    }
}

impl VaeDecoder {
    /// Build the decode graph for an input latent `[latent_ch, h, w]` and upload
    /// all decoder weights. `device`: `Some("cpu")` | `Some("gpu")` | `None`.
    pub fn from_diffusers(cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32, device: Option<&str>) -> VaeDecoder {
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
        let mut b = Builder {
            gpu: &gpu,
            t: tensors,
            eps: cfg.norm_eps,
            groups: cfg.norm_num_groups,
            steps: vec![],
            taps: vec![],
            pool: std::collections::HashMap::new(),
            taps_on: std::env::var("BRAIN_VAE_TAPS").is_ok(),
            coop: gpu.caps().workgroup_reductions,
            col: None,
        };

        let z_in = gpu.storage((cfg.latent_channels * h * w) as u64);
        let rc = cfg.reversed_channels();
        let mid_c = *cfg.block_out_channels.last().unwrap();

        // FLUX.2 post_quant_conv: 1×1 latent→latent ahead of conv_in. The shipped
        // diffusers checkpoint stores it top-level (`post_quant_conv.weight`); the
        // BFL-native layout nests it (`decoder.post_quant_conv.*`). Accept both;
        // a checkpoint with neither fails loudly in `Builder::get`.
        let zc = cfg.latent_channels;
        let pq = cfg.use_post_quant_conv.then(|| {
            let p = if tensors.contains_key("post_quant_conv.weight") {
                "post_quant_conv"
            } else {
                "decoder.post_quant_conv"
            };
            b.conv(p, zc, zc, 1, 0, h, w, &z_in)
        });

        // conv_in: latent → highest block channel. `z_in` is the persistent input,
        // never freed; `xlen` tracks the current `x` length so the previous `x` can
        // be returned to the pool once the next op has consumed it.
        let mut x = b.conv("decoder.conv_in", zc, mid_c, 3, 1, h, w, pq.as_ref().unwrap_or(&z_in));
        if let Some(p) = pq {
            b.free((zc * h * w) as u64, p); // last read was conv_in above
        }
        let mut xlen = (mid_c * h * w) as u64;
        b.tap("conv_in".into(), &x, mid_c * h * w);

        // Mid block: resnet, (optional) attention, resnet.
        let nx = b.resnet("decoder.mid_block.resnets.0", mid_c, mid_c, h, w, &x);
        b.free(xlen, x);
        x = nx;
        if cfg.mid_block_add_attention {
            let nx = b.attn("decoder.mid_block.attentions.0", mid_c, h, w, &x);
            b.free(xlen, x);
            x = nx;
            b.tap("mid_attn".into(), &x, mid_c * h * w);
        }
        let nx = b.resnet("decoder.mid_block.resnets.1", mid_c, mid_c, h, w, &x);
        b.free(xlen, x);
        x = nx;
        b.tap("mid".into(), &x, mid_c * h * w);

        // Up blocks: reversed channel schedule; upsample on all but the last.
        let n_blocks = rc.len();
        let n_res = cfg.layers_per_block + 1;
        let (mut cur_h, mut cur_w) = (h, w);
        let mut prev = mid_c;
        for (i, &out_c) in rc.iter().enumerate() {
            for r in 0..n_res {
                let cin = if r == 0 { prev } else { out_c };
                let nx = b.resnet(&format!("decoder.up_blocks.{i}.resnets.{r}"), cin, out_c, cur_h, cur_w, &x);
                b.free(xlen, x);
                x = nx;
                xlen = (out_c * cur_h * cur_w) as u64;
            }
            if i < n_blocks - 1 {
                let nx = b.upsample(out_c, cur_h, cur_w, &x);
                b.free(xlen, x);
                x = nx;
                cur_h *= 2;
                cur_w *= 2;
                xlen = (out_c * cur_h * cur_w) as u64;
                let nx = b.conv(&format!("decoder.up_blocks.{i}.upsamplers.0.conv"), out_c, out_c, 3, 1, cur_h, cur_w, &x);
                b.free(xlen, x);
                x = nx;
            }
            prev = out_c;
            b.tap(format!("up_block.{i}"), &x, out_c * cur_h * cur_w);
        }

        // Head: GroupNorm → SiLU → conv_out.
        let hn = b.gn("decoder.conv_norm_out", prev, cur_h, cur_w, &x);
        b.free(xlen, x);
        let hlen = (prev * cur_h * cur_w) as u64;
        let hs = b.silu(prev * cur_h * cur_w, &hn);
        b.free(hlen, hn);
        let out = b.conv("decoder.conv_out", prev, cfg.out_channels, 3, 1, cur_h, cur_w, &hs);
        b.free(hlen, hs);
        let out_len = (cfg.out_channels * cur_h * cur_w) as usize;

        let Builder { steps, taps, .. } = b;
        VaeDecoder { gpu, cfg, steps, z_in, out, out_len, taps }
    }

    /// Decode a latent `[latent_ch·h·w]` (row-major NCHW, batch 1) into the
    /// image `[out_ch·H·W]`. Raw decode — no scaling/shift is applied here.
    pub fn decode(&self, latent: &[f32]) -> Vec<f32> {
        let bits: Vec<u32> = latent.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.z_in, &bits);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, self.out_len)
    }

    /// Read a named intermediate tap after a `decode` (parity debugging).
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        self.taps.iter().find(|(n, _, _)| n == name).map(|(_, buf, len)| self.gpu.read(buf, *len))
    }

    pub fn config(&self) -> &VaeConfig {
        &self.cfg
    }

    /// The device the graph was built on (profiling / benches).
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The pre-recorded decode dispatch sequence (profiling / benches). Each
    /// [`Step`]'s `meta()` names its slot in [`KERNELS`].
    pub fn steps(&self) -> &[Step] {
        &self.steps
    }
}

// ===================== encoder =====================

/// An `AutoencoderKL` **encoder** graph for a fixed image size, weights resident.
/// Mirrors [`VaeDecoder`]: `image[3,H,W] → moments[2·latent, H/8, W/8]` (mean ‖
/// logvar). `conv_out` yields the moments directly unless the config enables the
/// FLUX.2 1×1 `quant_conv`, which then maps them in place. Reuses the shared
/// [`Builder`] blocks (conv/resnet/gn/silu/attn) and the strided
/// [`Builder::conv_down`] for the diffusers `Downsample2D`.
pub struct VaeEncoder {
    gpu: Gpu,
    cfg: VaeConfig,
    steps: Vec<Step>,
    img_in: DeviceBuffer,
    out: DeviceBuffer,
    out_len: usize,
    taps: Vec<(String, DeviceBuffer, usize)>,
}

impl VaeEncoder {
    /// Build the encode graph for an input image `[in_channels, h, w]` (full-res,
    /// NOT latent size) and upload all encoder weights. `device`: `Some("cpu")` |
    /// `Some("gpu")` | `None`.
    pub fn from_diffusers(cfg: VaeConfig, tensors: &Tensors, h: u32, w: u32, device: Option<&str>) -> VaeEncoder {
        let gpu = match device {
            Some("cpu") => Gpu::new_cpu(&KERNELS),
            Some("gpu") | Some("wgpu") => Gpu::new_wgpu(&KERNELS),
            _ => Gpu::new(&KERNELS),
        };
        let mut b = Builder { gpu: &gpu, t: tensors, eps: cfg.norm_eps, groups: cfg.norm_num_groups, steps: vec![], taps: vec![], pool: std::collections::HashMap::new(), taps_on: std::env::var("BRAIN_VAE_TAPS").is_ok(), coop: gpu.caps().workgroup_reductions, col: None };

        let img_in = gpu.storage((cfg.in_channels * h * w) as u64);
        let ch = &cfg.block_out_channels;

        // conv_in: image → block_out[0].
        let mut x = b.conv("encoder.conv_in", cfg.in_channels, ch[0], 3, 1, h, w, &img_in);
        b.tap("conv_in".into(), &x, ch[0] * h * w);

        // Down blocks: `layers_per_block` resnets each; downsample on all but last.
        let n_blocks = ch.len();
        let n_res = cfg.layers_per_block;
        let (mut cur_h, mut cur_w) = (h, w);
        let mut prev = ch[0];
        for (i, &out_c) in ch.iter().enumerate() {
            for r in 0..n_res {
                let cin = if r == 0 { prev } else { out_c };
                x = b.resnet(&format!("encoder.down_blocks.{i}.resnets.{r}"), cin, out_c, cur_h, cur_w, &x);
            }
            prev = out_c;
            if i < n_blocks - 1 {
                x = b.conv_down(&format!("encoder.down_blocks.{i}.downsamplers.0.conv"), out_c, cur_h, cur_w, &x);
                cur_h /= 2;
                cur_w /= 2;
            }
            b.tap(format!("down_block.{i}"), &x, out_c * cur_h * cur_w);
        }

        // Mid block: resnet, (optional) attention, resnet.
        let mid_c = *ch.last().unwrap();
        x = b.resnet("encoder.mid_block.resnets.0", mid_c, mid_c, cur_h, cur_w, &x);
        if cfg.mid_block_add_attention {
            x = b.attn("encoder.mid_block.attentions.0", mid_c, cur_h, cur_w, &x);
        }
        x = b.resnet("encoder.mid_block.resnets.1", mid_c, mid_c, cur_h, cur_w, &x);
        b.tap("mid".into(), &x, mid_c * cur_h * cur_w);

        // Head: GroupNorm → SiLU → conv_out → moments (2·latent channels).
        let hn = b.gn("encoder.conv_norm_out", mid_c, cur_h, cur_w, &x);
        let hs = b.silu(mid_c * cur_h * cur_w, &hn);
        let moments = 2 * cfg.latent_channels;
        let out = b.conv("encoder.conv_out", mid_c, moments, 3, 1, cur_h, cur_w, &hs);
        // FLUX.2 quant_conv: 1×1 moments→moments after conv_out. Top-level
        // (`quant_conv.weight`, diffusers export) or nested (`encoder.quant_conv.*`,
        // BFL-native); a checkpoint with neither fails loudly in `Builder::get`.
        let out = if cfg.use_quant_conv {
            let p = if tensors.contains_key("quant_conv.weight") {
                "quant_conv"
            } else {
                "encoder.quant_conv"
            };
            b.conv(p, moments, moments, 1, 0, cur_h, cur_w, &out)
        } else {
            out
        };
        let out_len = (moments * cur_h * cur_w) as usize;

        let Builder { steps, taps, .. } = b;
        VaeEncoder { gpu, cfg, steps, img_in, out, out_len, taps }
    }

    /// Encode an image `[in_channels·H·W]` (row-major NCHW, batch 1) into the
    /// moments `[2·latent·h·w]` — the first `latent` channels are the posterior
    /// mean, the next `latent` the log-variance. Raw: no scaling/shift applied.
    pub fn encode(&self, image: &[f32]) -> Vec<f32> {
        let bits: Vec<u32> = image.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.img_in, &bits);
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.out, self.out_len)
    }

    /// Encode and return only the posterior **mean** `[latent·h·w]` (the deterministic
    /// latent used for image-conditioned generation).
    pub fn encode_mean(&self, image: &[f32], lh: u32, lw: u32) -> Vec<f32> {
        let m = self.encode(image);
        let plane = (lh * lw) as usize;
        m[..self.cfg.latent_channels as usize * plane].to_vec()
    }

    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        self.taps.iter().find(|(n, _, _)| n == name).map(|(_, buf, len)| self.gpu.read(buf, *len))
    }
}
