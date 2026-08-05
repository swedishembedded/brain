// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The CodeFormer forward graph: encoder → code-prediction Transformer →
//! codebook gather → generator with the controllable feature transformation.
//!
//! Composition, not re-implementation.
//!
//! * The whole convolutional half is [`vqgan::model::run_blocks`] over
//!   [`vae::blocks::Builder`] — the same encoder/generator `crates/vqgan` is
//!   parity-gated on, walked in **segments** so the encoder taps and the CFT
//!   can be spliced between blocks without a second copy of the loop.
//! * The codebook gather is [`vqgan::model::record_lookup`] (the existing
//!   `embed` kernel), the reference's `get_codebook_feat`. The nearest-neighbour
//!   `vq_argmin` search is **not** on this path — predicting the indices instead
//!   is the whole point of CodeFormer — but its kernels stay registered so a
//!   caller can build a plain [`vqgan::Vqgan`] on the same device handle.
//! * The Transformer is assembled from `model::block`'s shared Step builders
//!   (`layernorm_fwd`, `pick_gemm`) and the existing bidirectional attention
//!   trio `vae::blocks` already registers (`attn_scores_bidir` /
//!   `attn_softmax_bidir` / `attn_apply_bidir`), resolved BY NAME.
//! * The CFT is `concat2` → [`vae::blocks::Builder::resnet`] → two
//!   `conv → leaky_relu → conv` towers → `mul`/`add2`/`scale_add`.
//!
//! **This crate adds no kernel and no block.**
//!
//! ## The graph
//!
//! ```text
//! submit A (encode + predict)
//!   img[3,512,512]
//!     -> encoder.blocks[0..=5]   -> enc['256'] [128,256,256]   pinned
//!     -> encoder.blocks[6..=8]   -> enc['128'] [128,128,128]   pinned
//!     -> encoder.blocks[9..=11]  -> enc['64']  [256, 64, 64]   pinned
//!     -> encoder.blocks[12..=14] -> enc['32']  [256, 32, 32]   pinned
//!     -> encoder.blocks[15..=24] -> lq_feat    [256, 16, 16]
//!   lq_feat -> nchw_to_rows -> [T=256, 256] -> feat_emb -> x0[T, 512]
//!   9 x TransformerSALayer   (pre-LN; q = k = LN(x) + position_emb, v = LN(x))
//!   idx_pred_layer(LN + biasless linear) -> logits[T, 1024] -> argmax_row
//! (host) f32 indices -> u32
//! submit B (gather + generate)
//!   embed(indices, codebook) -> rows[T,256] -> rows_to_nchw -> quant_feat
//!     -> generator.blocks[0..=9]   -> fuse('32',  enc['32'],  w)
//!     -> generator.blocks[10..=12] -> fuse('64',  enc['64'],  w)
//!     -> generator.blocks[13..=15] -> fuse('128', enc['128'], w)
//!     -> generator.blocks[16..=18] -> fuse('256', enc['256'], w)
//!     -> generator.blocks[19..=24] -> out[3,512,512]
//! ```
//!
//! ## The fidelity dial `w`
//!
//! `Fuse_sft_block.forward` is
//! `out = dec + w · (dec · scale(e) + shift(e))`, and the reference **skips the
//! block entirely when `w == 0`** (`codeformer_arch.py:275`). So:
//!
//! * **w = 0** — no encoder feature reaches the generator; the output is the
//!   reconstruction of the *predicted codes* alone. Maximum **quality**.
//! * **w = 1** — the full CFT residual. Maximum **fidelity** to the input face.
//!
//! `w` lives in a one-element device buffer read by `scale_add`, not baked into
//! a recorded step, so changing it is a buffer write and not a graph rebuild.
//! Evaluating the block and scaling by 0 is bit-identical to skipping it
//! (`0·finite = 0`, `x + 0 = x`), which is what makes the `w = 0` golden a real
//! gate on the fuse wiring rather than a test of the branch.

use std::collections::HashMap;

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block;
use vae::blocks::{BlockNames, Builder, Tensors};

use crate::config::{CodeFormerConfig, FuseTap};

/// Kernel slots appended after [`vqgan::KERNELS`] (which itself starts with the
/// [`vae::blocks::KERNELS`] block set, so slots `0..vae::blocks::NEXT_SLOT` stay
/// exactly where the shared `Builder` addresses them).
const N_VQGAN: usize = vqgan::KERNELS.len();
const K_LAYERNORM: usize = N_VQGAN;
// `layernorm_rows` is N_VQGAN + 1 — registered so `block::LayerNormIds::resolve_fwd`
// can pick it up BY NAME on a device with workgroup reductions; never indexed.
const K_MATMUL: usize = N_VQGAN + 2;
const K_MATMUL_REG3: usize = N_VQGAN + 3;
const K_BIAS_ADD: usize = N_VQGAN + 4;
const K_GELU_ERF: usize = N_VQGAN + 5;
const K_ARGMAX_ROW: usize = N_VQGAN + 6;
const K_CONCAT2: usize = N_VQGAN + 7;
const K_MUL: usize = N_VQGAN + 8;
const K_LEAKY_RELU: usize = N_VQGAN + 9;
const K_SCALE_ADD: usize = N_VQGAN + 10;
const K_REGION_COPY: usize = N_VQGAN + 11;

/// This model's kernel set: [`vqgan::KERNELS`] verbatim (never restated — a
/// restated list that drifts by one entry is silently wrong, not a crash) plus
/// the twelve the Transformer and the CFT need.
pub const KERNELS: [(&str, &str); N_VQGAN + 12] = kernel_set();

const fn kernel_set() -> [(&'static str, &'static str); N_VQGAN + 12] {
    let mut k = [("", ""); N_VQGAN + 12];
    let mut i = 0;
    while i < N_VQGAN {
        k[i] = vqgan::KERNELS[i];
        i += 1;
    }
    k[K_LAYERNORM] = ("layernorm", kernels::LAYERNORM);
    k[N_VQGAN + 1] = ("layernorm_rows", kernels::LAYERNORM_ROWS);
    k[K_MATMUL] = ("matmul", kernels::MATMUL);
    k[K_MATMUL_REG3] = ("matmul_reg3", kernels::MATMUL_REG3);
    k[K_BIAS_ADD] = ("bias_add", kernels::BIAS_ADD);
    k[K_GELU_ERF] = ("gelu_erf", kernels::GELU_ERF);
    k[K_ARGMAX_ROW] = ("argmax_row", kernels::ARGMAX_ROW);
    k[K_CONCAT2] = ("concat2", kernels::CONCAT2);
    k[K_MUL] = ("mul", kernels::MUL);
    k[K_LEAKY_RELU] = ("leaky_relu", kernels::LEAKY_RELU);
    k[K_SCALE_ADD] = ("scale_add", kernels::SCALE_ADD);
    k[K_REGION_COPY] = ("region_copy", kernels::REGION_COPY);
    k
}

/// Slot indices resolved from the device by NAME, so this crate never restates
/// where `vae::blocks` put its attention trio.
#[derive(Clone, Copy)]
pub(crate) struct Ids {
    scores: usize,
    softmax: usize,
    apply: usize,
    ln: block::LayerNormIds,
}

impl Ids {
    pub(crate) fn resolve(g: &Gpu) -> Ids {
        let at = |n: &str| {
            g.kernel_index(n)
                .unwrap_or_else(|| panic!("restore: kernel {n} not in the device's set"))
        };
        Ids {
            scores: at("attn_scores_bidir"),
            softmax: at("attn_softmax_bidir"),
            apply: at("attn_apply_bidir"),
            ln: block::LayerNormIds::resolve_fwd(g, K_LAYERNORM),
        }
    }
}

/// One `image → predicted codes → restored image` pass.
pub struct Restoration {
    /// Codebook index predicted for each latent position, row-major over the
    /// 16×16 grid.
    pub indices: Vec<u32>,
    /// The fidelity weight the generator ran at.
    pub w: f32,
    /// Restored image `[3, H, W]`, row-major, in the reference's `[-1, 1]`
    /// range (the reference does not clamp; neither does this).
    pub image: Vec<f32>,
}

/// A CodeFormer restoration graph for the config's fixed input size, weights
/// resident.
pub struct CodeFormer {
    gpu: Gpu,
    cfg: CodeFormerConfig,
    hw: (u32, u32),
    lhw: (u32, u32),
    encode_steps: Vec<Step>,
    decode_steps: Vec<Step>,
    img_in: DeviceBuffer,
    logits: DeviceBuffer,
    idx_f32: DeviceBuffer,
    idx_in: DeviceBuffer,
    w_buf: DeviceBuffer,
    out: DeviceBuffer,
    taps: Vec<(String, DeviceBuffer, usize)>,
    /// Whether [`CodeFormer::predict_codes`] has run on this instance.
    ///
    /// The four encoder features and the logits live in device buffers written
    /// by submit A and read by submit B, so `generate`/`code_logits` are only
    /// meaningful after it. Without this flag a caller that skips it gets a
    /// plausible image computed from never-written buffers — silently wrong,
    /// not a crash, which is the failure class this repo pays for most.
    /// `AtomicBool` rather than `Cell` so the type stays `Sync` for the
    /// deferred serving contract.
    encoded: std::sync::atomic::AtomicBool,
}

impl CodeFormer {
    /// Build both graphs on `gpu` (which MUST have been created with
    /// [`KERNELS`]) and upload the weights from a validated
    /// [`crate::import::Import`]'s tensor map.
    ///
    /// `taps` records every stage for parity replay; it pins every activation
    /// (the shared builder's buffer pool is disabled), so leave it off outside
    /// tests.
    pub fn new(cfg: CodeFormerConfig, tensors: &Tensors, gpu: Gpu, taps: bool) -> CodeFormer {
        let img = cfg.img_size();
        let scale = cfg.vqgan.downscale();
        let (lh, lw) = (img / scale, img / scale);
        let t = lh * lw;
        assert_eq!(
            t, cfg.latent_size,
            "restore: {img}² at the {scale}× downscale gives {t} latent positions, but \
             position_emb covers {}",
            cfg.latent_size
        );
        let emb = cfg.vqgan.emb_dim;
        let ids = Ids::resolve(&gpu);

        let img_in = gpu.storage((cfg.vqgan.in_channels * img * img) as u64);
        let idx_in = gpu.storage(t as u64);
        let w_buf = gpu.storage(1);

        // ---- submit A: encoder segments + transformer + argmax --------------
        let (encode_steps, logits, idx_f32, enc_feat, mut all_taps) = {
            let mut b = Builder::new(
                &gpu,
                tensors,
                cfg.vqgan.norm_eps,
                cfg.vqgan.norm_groups,
                BlockNames::vqgan(),
                taps,
            );
            let blocks = cfg.vqgan.encoder_blocks();
            // Encoder taps, ascending by block index (the walk order); the
            // config lists them in generator order.
            let mut enc_taps = cfg.taps();
            enc_taps.sort_by_key(|t| t.enc_block);

            let mut enc_feat: HashMap<u32, DeviceBuffer> = HashMap::new();
            let mut x = img_in.clone();
            let (mut h, mut w) = (img, img);
            let mut start = 0usize;
            for tp in &enc_taps {
                let (y, (hh, ww)) = vqgan::model::run_blocks(
                    &mut b,
                    "encoder",
                    &blocks[start..=tp.enc_block],
                    start,
                    h,
                    w,
                    &x,
                );
                b.tap(format!("enc.{}", tp.size), &y, tp.channels * hh * ww);
                enc_feat.insert(tp.size, y.clone());
                x = y;
                h = hh;
                w = ww;
                start = tp.enc_block + 1;
            }
            let (lq, (qh, qw)) =
                vqgan::model::run_blocks(&mut b, "encoder", &blocks[start..], start, h, w, &x);
            assert_eq!((qh, qw), (lh, lw), "restore: encoder produced {qh}x{qw}, expected {lh}x{lw}");
            b.tap("lq_feat".into(), &lq, emb * t);

            // `lq_feat.flatten(2).permute(2,0,1)` == the NCHW→NLC permutation.
            let rows = b.nchw_to_rows(emb, t, &lq);
            // `lq` is dead once its rows exist — it is NOT one of the four
            // encoder features the generator needs, so unlike the segment
            // outputs above it goes back to the pool. (`free` is a no-op in the
            // tapped build, which is what keeps the `lq_feat` tap readable.)
            b.free((emb * t) as u64, lq);
            let (logits, idx_f32) = record_transformer(&mut b, &cfg, &ids, t, &rows);
            b.free((emb * t) as u64, rows);
            let (steps, tp) = b.finish();
            (steps, logits, idx_f32, enc_feat, tp)
        };

        // ---- submit B: codebook gather + generator with the CFT --------------
        let (decode_steps, out, dec_taps) = {
            let mut b = Builder::new(
                &gpu,
                tensors,
                cfg.vqgan.norm_eps,
                cfg.vqgan.norm_groups,
                BlockNames::vqgan(),
                taps,
            );
            let cb = b.dev("quantize.embedding.weight");
            let rows = vqgan::model::record_lookup(&mut b, &cb, t, emb, &idx_in);
            let quant = b.rows_to_nchw(emb, t, &rows);
            b.free((t * emb) as u64, rows);
            b.tap("quant_feat".into(), &quant, emb * t);

            let blocks = cfg.vqgan.generator_blocks();
            let mut x = quant;
            // `run_blocks` documents that the input buffer stays the CALLER's,
            // so each segment input must be returned here or it is pinned for
            // the life of the graph. In the encoder that is exactly what we
            // want (the four features are read by the CFT much later); in the
            // generator nothing reads a segment input again, so holding them
            // was ~47 MB of dead VRAM on the production (`taps = false`) path.
            let mut xlen = (emb * t) as u64;
            let (mut h, mut w) = (lh, lw);
            let mut start = 0usize;
            for tp in cfg.taps() {
                // Non-empty by construction: `FUSE_TAPS`' generator blocks are
                // 3 apart, so no subset of `connect` can produce an empty
                // segment (which would return the input itself and make the
                // free below a use-after-free).
                let seg = &blocks[start..=tp.gen_block];
                assert!(!seg.is_empty(), "restore: empty generator segment at tap {}", tp.size);
                let (y, (hh, ww)) =
                    vqgan::model::run_blocks(&mut b, "generator", seg, start, h, w, &x);
                b.free(xlen, x);
                let enc = &enc_feat[&tp.size];
                let fused = record_fuse(&mut b, &cfg, &tp, enc, &y, hh, ww, &w_buf);
                b.free((tp.channels * hh * ww) as u64, y);
                x = fused;
                xlen = (tp.channels * hh * ww) as u64;
                h = hh;
                w = ww;
                start = tp.gen_block + 1;
            }
            let (out, (oh, ow)) =
                vqgan::model::run_blocks(&mut b, "generator", &blocks[start..], start, h, w, &x);
            b.free(xlen, x);
            assert_eq!((oh, ow), (img, img), "restore: generator produced {oh}x{ow}");
            let (steps, tp) = b.finish();
            (steps, out, tp)
        };
        all_taps.extend(dec_taps);

        CodeFormer {
            gpu,
            cfg,
            hw: (img, img),
            lhw: (lh, lw),
            encode_steps,
            decode_steps,
            img_in,
            logits,
            idx_f32,
            idx_in,
            w_buf,
            out,
            taps: all_taps,
            encoded: std::sync::atomic::AtomicBool::new(false),
        }
    }

    pub fn config(&self) -> &CodeFormerConfig {
        &self.cfg
    }

    /// `(lh, lw)` of the latent code grid.
    pub fn latent_size(&self) -> (u32, u32) {
        self.lhw
    }

    /// Run the encoder and the code-prediction Transformer over an image
    /// `[3·H·W]` (row-major NCHW, batch 1, values in `[-1, 1]`) and return the
    /// predicted codebook index per latent position.
    ///
    /// This is the reference's `code_only` path: the argmax of the 1024-way
    /// logits, which equals `topk(softmax(logits), 1)` because softmax is
    /// monotone. Ties go to the lowest index in both.
    pub fn predict_codes(&self, image: &[f32]) -> Vec<u32> {
        let n = (self.cfg.vqgan.in_channels * self.hw.0 * self.hw.1) as usize;
        assert_eq!(image.len(), n, "restore: image has {} values, expected {n}", image.len());
        let bits: Vec<u32> = image.iter().map(|v| v.to_bits()).collect();
        self.gpu.write(&self.img_in, &bits);
        self.gpu.submit(&[], &self.encode_steps);
        self.encoded.store(true, std::sync::atomic::Ordering::Release);
        let t = (self.lhw.0 * self.lhw.1) as usize;
        // `argmax_row` returns the winning column as f32 (exact below 2^24).
        self.gpu.read(&self.idx_f32, t).into_iter().map(|v| v as u32).collect()
    }

    /// Gather `indices` from the codebook and run the generator with the
    /// controllable feature transformation at fidelity weight `w`.
    ///
    /// `w = 0` is maximum quality (no encoder feature injected), `w = 1`
    /// maximum fidelity to the input. The encoder features come from the LAST
    /// [`CodeFormer::predict_codes`] call — they live on the device between the
    /// two submits, which is the whole reason the graph is split there.
    pub fn generate(&self, indices: &[u32], w: f32) -> Vec<f32> {
        self.assert_encoded("generate");
        let t = (self.lhw.0 * self.lhw.1) as usize;
        assert_eq!(indices.len(), t, "restore: {} indices, expected {t}", indices.len());
        // The `embed` gather indexes the codebook with NO bounds check (brain
        // compiles its shaders with runtime checks off), so an out-of-range code
        // would read past the buffer instead of trapping. Validate here exactly
        // as `vqgan::Vqgan::decode` does — this entry point takes caller-supplied
        // indices.
        let k = self.cfg.vqgan.codebook_size;
        if let Some(&bad) = indices.iter().find(|&&i| i >= k) {
            panic!("restore: code index {bad} out of range for {k} codes");
        }
        self.gpu.write(&self.idx_in, indices);
        self.gpu.write(&self.w_buf, &[w.to_bits()]);
        self.gpu.submit(&[], &self.decode_steps);
        self.gpu.read(&self.out, self.out_len())
    }

    /// Full `image → predicted codes → restored image` pass at fidelity `w`.
    pub fn restore(&self, image: &[f32], w: f32) -> Restoration {
        let indices = self.predict_codes(image);
        let image = self.generate(&indices, w);
        Restoration { indices, w, image }
    }

    /// The 1024-way code logits `[T, codebook_size]` from the last
    /// [`CodeFormer::predict_codes`].
    pub fn code_logits(&self) -> Vec<f32> {
        self.assert_encoded("code_logits");
        let t = (self.lhw.0 * self.lhw.1) as usize;
        self.gpu.read(&self.logits, t * self.cfg.vqgan.codebook_size as usize)
    }

    /// Panic if submit A has never run: every buffer `what` reads is written by
    /// [`CodeFormer::predict_codes`], and reading them before it produces a
    /// finite, plausible, wrong answer rather than an error.
    fn assert_encoded(&self, what: &str) {
        assert!(
            self.encoded.load(std::sync::atomic::Ordering::Acquire),
            "restore: {what}() before predict_codes() — the encoder features and logits it \
             reads have never been written on this CodeFormer"
        );
    }

    fn out_len(&self) -> usize {
        (self.cfg.vqgan.out_channels * self.hw.0 * self.hw.1) as usize
    }

    /// Read a recorded intermediate by name (`enc.256`, `lq_feat`, `ft.03`,
    /// `logits`, `quant_feat`, `generator.blocks.9`, `fuse.32.out`, …). `None`
    /// unless the model was built with `taps = true`.
    pub fn read_tap(&self, name: &str) -> Option<Vec<f32>> {
        self.taps.iter().find(|(n, _, _)| n == name).map(|(_, b, len)| self.gpu.read(b, *len))
    }

    /// Every recorded tap name (parity-test diagnostics).
    pub fn tap_names(&self) -> Vec<&str> {
        self.taps.iter().map(|(n, _, _)| n.as_str()).collect()
    }

    /// The device the graphs were built on (profiling / benches).
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The recorded encode and decode dispatch sequences (profiling / benches).
    pub fn steps(&self) -> (&[Step], &[Step]) {
        (&self.encode_steps, &self.decode_steps)
    }
}

/// `y[m, n] = x[m, k] · wgt[n, k]ᵀ` — the register-tiled kernel once both output
/// dims fill a 128×128 tile, else the naive one. Same math either way.
fn matmul(
    b: &mut Builder,
    m: u32,
    k: u32,
    n: u32,
    x: &DeviceBuffer,
    wgt: &DeviceBuffer,
) -> DeviceBuffer {
    let y = b.act((m * n) as u64);
    let (kind, threads) =
        block::pick_gemm(m as usize, n as usize, K_MATMUL, K_MATMUL_REG3, false);
    let step = b.gpu().step(kind, &[x, wgt, &y], &[m, k, n], threads);
    b.push_step(step);
    y
}

/// A biased linear from `{prefix}.{weight,bias}`: `matmul` then `bias_add`
/// (Params `[m, n]`, in place over the output).
fn linear(b: &mut Builder, prefix: &str, m: u32, k: u32, n: u32, x: &DeviceBuffer) -> DeviceBuffer {
    let wgt = b.dev(&format!("{prefix}.weight"));
    let bias = b.dev(&format!("{prefix}.bias"));
    let y = matmul(b, m, k, n, x, &wgt);
    let step = b.gpu().step(K_BIAS_ADD, &[&y, &bias], &[m, n], m * n);
    b.push_step(step);
    y
}

/// LayerNorm from `{prefix}.{weight,bias}` over `rows` rows of `d`.
fn layernorm(
    b: &mut Builder,
    ids: &Ids,
    prefix: &str,
    d: u32,
    rows: u32,
    eps: f32,
    x: &DeviceBuffer,
) -> DeviceBuffer {
    let gamma = b.dev(&format!("{prefix}.weight"));
    let beta = b.dev(&format!("{prefix}.bias"));
    let y = b.act((rows * d) as u64);
    let step = block::layernorm_fwd(b.gpu(), &ids.ln, x, &gamma, &beta, &y, d, rows, eps);
    b.push_step(step);
    y
}

/// The code-prediction Transformer: `feat_emb` → `n_layers` ×
/// `TransformerSALayer` → `idx_pred_layer` → `argmax_row`.
///
/// Returns `(logits[T, codebook_size], argmax[T] as f32)`.
///
/// The reference layer (`codeformer_arch.py:120`) is **pre-norm**, and the
/// position embedding goes to q and k but NOT to v:
///
/// ```text
/// tgt2 = norm1(tgt);  q = k = tgt2 + query_pos;  v = tgt2
/// tgt  = tgt + self_attn(q, k, v)
/// tgt2 = norm2(tgt)
/// tgt  = tgt + linear2(gelu(linear1(tgt2)))
/// ```
///
/// That asymmetry is why the checkpoint's fused `in_proj_weight` is split at
/// import: q/k read `tgt2 + pos` and v reads `tgt2`, so one fused GEMM cannot
/// serve both. The split lands exactly where the attention kernels want it —
/// `attn_scores_bidir` reads q and k out of ONE buffer at `qkv_stride = 2E`
/// (`q_off = 0`, `k_off = E`) and `attn_apply_bidir` reads v out of its own at
/// stride `E`.
pub(crate) fn record_transformer(
    b: &mut Builder,
    cfg: &CodeFormerConfig,
    ids: &Ids,
    t: u32,
    lq_rows: &DeviceBuffer,
) -> (DeviceBuffer, DeviceBuffer) {
    let e = cfg.dim_embd;
    let mlp = cfg.dim_mlp;
    let heads = cfg.n_head;
    let hd = cfg.head_dim();
    let eps = cfg.ln_eps;
    let (ne, nmlp) = ((t * e) as u64, (t * mlp) as u64);

    let pos = b.dev("position_emb");
    let mut x = linear(b, "feat_emb", t, cfg.vqgan.emb_dim, e, lq_rows);
    b.tap("feat_emb".into(), &x, t * e);

    for l in 0..cfg.n_layers as usize {
        let p = CodeFormerConfig::layer_prefix(l);
        let n1 = layernorm(b, ids, &format!("{p}.norm1"), e, t, eps, &x);
        b.tap(format!("ft.{l:02}.norm1"), &n1, t * e);

        // q = k = norm1(x) + position_emb; v = norm1(x).
        let qk_in = b.add(t * e, &n1, &pos);
        let qk = linear(b, &format!("{p}.self_attn.qk"), t, e, 2 * e, &qk_in);
        b.free(ne, qk_in);
        let v = linear(b, &format!("{p}.self_attn.v"), t, e, e, &n1);
        b.free(ne, n1);

        // `attn_scores_bidir`  Params: [bsz, n_heads, tcols, head_dim, qkv_stride, q_off, k_off]
        // `attn_softmax_bidir` Params: [bsz, n_heads, tcols]
        // `attn_apply_bidir`   Params: [bsz, n_heads, tcols, head_dim, qkv_stride, v_off, d_model]
        let scores = b.act((heads * t * t) as u64);
        let step = b.gpu().step(
            ids.scores,
            &[&qk, &scores],
            &[1, heads, t, hd, 2 * e, 0, e],
            heads * t * t,
        );
        b.push_step(step);
        b.free((2 * t * e) as u64, qk);
        let probs = b.act((heads * t * t) as u64);
        let step = b.gpu().step(ids.softmax, &[&scores, &probs], &[1, heads, t], heads * t);
        b.push_step(step);
        b.free((heads * t * t) as u64, scores);
        let ctx = b.act(ne);
        let step =
            b.gpu().step(ids.apply, &[&probs, &v, &ctx], &[1, heads, t, hd, e, 0, e], heads * t * hd);
        b.push_step(step);
        b.free((heads * t * t) as u64, probs);
        b.free(ne, v);
        b.tap(format!("ft.{l:02}.ctx"), &ctx, t * e);

        let ao = linear(b, &format!("{p}.self_attn.out_proj"), t, e, e, &ctx);
        b.free(ne, ctx);
        b.tap(format!("ft.{l:02}.attn_out"), &ao, t * e);
        let res = b.add(t * e, &x, &ao);
        b.free(ne, ao);
        b.free(ne, x);

        let n2 = layernorm(b, ids, &format!("{p}.norm2"), e, t, eps, &res);
        b.tap(format!("ft.{l:02}.norm2"), &n2, t * e);
        let h1 = linear(b, &format!("{p}.linear1"), t, e, mlp, &n2);
        b.free(ne, n2);
        b.tap(format!("ft.{l:02}.linear1"), &h1, t * mlp);
        // `F.gelu`'s default is the ERF form, not the tanh approximation.
        let hact = b.act(nmlp);
        let step = b.gpu().step(K_GELU_ERF, &[&h1, &hact], &[t * mlp], t * mlp);
        b.push_step(step);
        b.free(nmlp, h1);
        let mo = linear(b, &format!("{p}.linear2"), t, mlp, e, &hact);
        b.free(nmlp, hact);
        b.tap(format!("ft.{l:02}.linear2"), &mo, t * e);
        x = b.add(t * e, &res, &mo);
        b.free(ne, mo);
        b.free(ne, res);
        b.tap(format!("ft.{l:02}"), &x, t * e);
    }

    // idx_pred_layer = Sequential(LayerNorm(E), Linear(E, K, bias=False)).
    let ln = layernorm(b, ids, "idx_pred_layer.0", e, t, eps, &x);
    b.tap("logits_norm".into(), &ln, t * e);
    b.free(ne, x);
    let k = cfg.vqgan.codebook_size;
    let head = b.dev("idx_pred_layer.1.weight");
    let logits = matmul(b, t, e, k, &ln, &head);
    b.free(ne, ln);
    b.tap("logits".into(), &logits, t * k);

    // `argmax_row` Params: [m, n]; one invocation per ROW, ties → lowest index,
    // returned as f32 (exact below 2^24, and the codebook is 1024).
    let idx = b.act(t as u64);
    let step = b.gpu().step(K_ARGMAX_ROW, &[&logits, &idx], &[t, k], t);
    b.push_step(step);
    (logits, idx)
}

/// One `Fuse_sft_block` — the controllable feature transformation.
///
/// ```text
/// e     = encode_enc(cat([enc_feat, dec_feat], dim=1))     ResBlock(2C -> C)
/// scale = conv3x3 -> LeakyReLU(0.2) -> conv3x3   (e)
/// shift = conv3x3 -> LeakyReLU(0.2) -> conv3x3   (e)
/// out   = dec_feat + w * (dec_feat * scale + shift)
/// ```
///
/// `w` is read from a one-element device buffer by `scale_add`, whose Params
/// `[seq_len, d_model, n_experts, e_idx, accumulate]` degenerate to
/// `[1, N, 1, 0, 1]`: `acc[i] += gate[0] * src[i]` over the whole tensor. That
/// is the ONE place the dial enters the graph.
#[allow(clippy::too_many_arguments)]
fn record_fuse(
    b: &mut Builder,
    cfg: &CodeFormerConfig,
    tap: &FuseTap,
    enc: &DeviceBuffer,
    dec: &DeviceBuffer,
    h: u32,
    w: u32,
    w_buf: &DeviceBuffer,
) -> DeviceBuffer {
    let p = CodeFormerConfig::fuse_prefix(tap);
    let c = tap.channels;
    let n = c * h * w;
    let nn = n as u64;

    // `concat2` Params: [N, Ca, Cb, H, W]; one invocation per OUTPUT element.
    let cat = b.act(2 * nn);
    let step = b.gpu().step(K_CONCAT2, &[enc, dec, &cat], &[1, c, c, h, w], 2 * n);
    b.push_step(step);
    let e = b.resnet(&format!("{p}.encode_enc"), 2 * c, c, h, w, &cat);
    b.free(2 * nn, cat);
    b.tap(format!("fuse.{}.encode_enc", tap.size), &e, n);

    let scale = record_tower(b, cfg, &format!("{p}.scale"), c, h, w, &e);
    b.tap(format!("fuse.{}.scale", tap.size), &scale, n);
    let shift = record_tower(b, cfg, &format!("{p}.shift"), c, h, w, &e);
    b.tap(format!("fuse.{}.shift", tap.size), &shift, n);
    b.free(nn, e);

    // residual = dec * scale + shift
    let prod = b.act(nn);
    let step = b.gpu().step(K_MUL, &[dec, &scale, &prod], &[n], n);
    b.push_step(step);
    b.free(nn, scale);
    let resid = b.add(n, &prod, &shift);
    b.free(nn, prod);
    b.free(nn, shift);

    // out = dec; out += w * residual
    let out = b.act(nn);
    // `region_copy` Params: [rows, width, row_stride, off] — one whole row.
    let step = b.gpu().step(K_REGION_COPY, &[dec, &out], &[1, n, n, 0], n);
    b.push_step(step);
    // `scale_add` Params: [seq_len, d_model, n_experts, e_idx, accumulate].
    let step = b.gpu().step(K_SCALE_ADD, &[w_buf, &resid, &out], &[1, n, 1, 0, 1], n);
    b.push_step(step);
    b.free(nn, resid);
    b.tap(format!("fuse.{}.out", tap.size), &out, n);
    out
}

/// `Conv3×3 → LeakyReLU(0.2) → Conv3×3`, the `scale` / `shift` towers. The
/// reference builds them as `nn.Sequential`, so the convs are indices 0 and 2.
fn record_tower(
    b: &mut Builder,
    cfg: &CodeFormerConfig,
    prefix: &str,
    c: u32,
    h: u32,
    w: u32,
    x: &DeviceBuffer,
) -> DeviceBuffer {
    let n = c * h * w;
    let a = b.conv(&format!("{prefix}.0"), c, c, 3, 1, h, w, x);
    // `leaky_relu` Params: [total, slope] (slope bit-cast into the uniform).
    let l = b.act(n as u64);
    let step = b.gpu().step(K_LEAKY_RELU, &[&a, &l], &[n, f(cfg.leaky_slope)], n);
    b.push_step(step);
    b.free(n as u64, a);
    let y = b.conv(&format!("{prefix}.2"), c, c, 3, 1, h, w, &l);
    b.free(n as u64, l);
    y
}

#[cfg(test)]
mod tests {
    /// The shared block builder and `vqgan` address their slots by position, so
    /// this crate's set must start with theirs verbatim.
    #[test]
    fn shared_slots_are_copied_verbatim() {
        assert_eq!(super::KERNELS[..super::N_VQGAN], vqgan::KERNELS[..]);
        assert_eq!(super::KERNELS[..vae::blocks::NEXT_SLOT], vae::blocks::KERNELS[..]);
    }

    /// Every appended slot must hold the kernel its constant names — an
    /// off-by-one here is silently wrong arithmetic, not a crash.
    #[test]
    fn appended_slots_hold_their_named_kernels() {
        for (slot, name) in [
            (super::K_LAYERNORM, "layernorm"),
            (super::N_VQGAN + 1, "layernorm_rows"),
            (super::K_MATMUL, "matmul"),
            (super::K_MATMUL_REG3, "matmul_reg3"),
            (super::K_BIAS_ADD, "bias_add"),
            (super::K_GELU_ERF, "gelu_erf"),
            (super::K_ARGMAX_ROW, "argmax_row"),
            (super::K_CONCAT2, "concat2"),
            (super::K_MUL, "mul"),
            (super::K_LEAKY_RELU, "leaky_relu"),
            (super::K_SCALE_ADD, "scale_add"),
            (super::K_REGION_COPY, "region_copy"),
        ] {
            assert_eq!(super::KERNELS[slot].0, name, "slot {slot}");
        }
        // No empty slot left over.
        assert!(super::KERNELS.iter().all(|(n, s)| !n.is_empty() && !s.is_empty()));
    }

    /// The attention trio this crate dispatches lives in `vae::blocks`' set and
    /// is resolved by name; assert the names are actually there.
    #[test]
    fn the_bidir_attention_trio_is_registered() {
        for n in ["attn_scores_bidir", "attn_softmax_bidir", "attn_apply_bidir"] {
            assert!(super::KERNELS.iter().any(|(k, _)| *k == n), "{n} missing from KERNELS");
        }
    }
}
