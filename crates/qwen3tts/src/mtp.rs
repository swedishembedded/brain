// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS **MTP code-predictor**: a small (5-layer) Qwen3 decoder that fills
//! the residual codebooks 1..15 of one acoustic frame, conditioned on the
//! Talker hidden state and codebook-0.
//!
//! Per `modeling_qwen3_tts.py` (`Qwen3TTSTalkerCodePredictorModel` +
//! `forward_finetune` / `forward_sub_talker_finetune`), one frame is processed as
//! a length-`num_code_groups` sequence of *input embeddings* under full (causal)
//! attention:
//!   pos 0 : the Talker hidden state (`small_to_mtp_projection` is `Identity`
//!           here, since the MTP and Talker share `hidden_size = 1024`),
//!   pos 1 : `talker.codec_embedding(codebook0)`  (the Talker's table),
//!   pos i (2..15) : `code_predictor.codec_embedding[i-2](codebook_{i-1})`.
//! The per-position output head `lm_head[i-1]` reads `hidden[:, i]` to predict
//! codebook `i` (positions 1..15 → 15 residual codebooks).
//!
//! The decoder block is the same Qwen3 block as the Talker (RMSNorm, GQA with
//! per-head QK-norm, half-split RoPE base 1e6, SwiGLU), so it is built from the
//! shared `model::block` step-builders. The decoder runs on the GPU engine; the
//! (tiny) input-embedding gather and the per-position output heads run on the
//! CPU. This is an inference forward (no backward) — the Talker decoder carries
//! the gradient-checked block coverage.

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block::{self, Gqa, KernelIds};
use paramstore::ParamStore;

use crate::config::MtpConfig;

// ---- kernel pipeline table (forward subset; indices match block::KernelIds) ----
const MATMUL: usize = 0;
const RMSNORM: usize = 1;
const RMS_INV: usize = 2;
const ROPE: usize = 3;
const GQA_SCORES: usize = 4;
const ATTN_SOFTMAX: usize = 5;
const GQA_APPLY: usize = 6;
const SILU_MUL: usize = 7;
const ADD2: usize = 8;
// Coalesced RMSNorm - the throughput twin of `RMSNORM`, selected by
// `block::rms_variant` inside `block::rmsnorm_fwd`.
const RMSNORM_ROWS: usize = 9;

pub const PIPELINES: &[(&str, &str)] = &[
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("rope_base", kernels::ROPE_BASE),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
];

struct Layer {
    xn1: DeviceBuffer,
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
}

pub struct MtpModel {
    pub cfg: MtpConfig,
    gpu: Gpu,
    ps: ParamStore,
    t: u32,
    // GPU forward scratch
    res: Vec<DeviceBuffer>,
    layers: Vec<Layer>,
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    scores: DeviceBuffer,
    xn_final: DeviceBuffer,
    fwd_steps: Vec<Step>,
    // CPU input-embedding tables (residual codebooks) and output heads.
    codec_embedding: Vec<Vec<f32>>, // [n_residual][vocab*d]
    lm_head: Vec<Vec<f32>>,         // [n_residual][vocab*d]
}

impl MtpModel {
    fn only_fwd_ids() -> KernelIds {
        // Forward needs rmsnorm/rms_inv, rope, gqa scores/apply/softmax, silu_mul.
        // No backward graph is built, so every backward slot is UNREGISTERED -
        // out of range of PIPELINES, so reaching one is a panic rather than a
        // silent dispatch of whichever kernel the stand-in index named.
        KernelIds {
            rmsnorm: RMSNORM,
            rms_inv: RMS_INV,
            rmsnorm_dx: block::UNREGISTERED,
            rmsnorm_dx_rows: block::UNREGISTERED,
            rmsnorm_dw: block::UNREGISTERED,
            rope: ROPE,
            rope_bwd: block::UNREGISTERED,
            gqa_scores: GQA_SCORES,
            gqa_apply: GQA_APPLY,
            attn_softmax: ATTN_SOFTMAX,
            gqa_dscores: block::UNREGISTERED,
            gqa_dv: block::UNREGISTERED,
            gqa_dq: block::UNREGISTERED,
            gqa_dk: block::UNREGISTERED,
            silu_mul: SILU_MUL,
            silu_da: block::UNREGISTERED,
            silu_db: block::UNREGISTERED,
            rmsnorm_rows: RMSNORM_ROWS,
        }
    }

    /// Decoder block parameter list (blocks + final norm); the codec embeddings
    /// and heads live on the CPU.
    pub(crate) fn decoder_param_list(cfg: &MtpConfig) -> Vec<(String, usize)> {
        let d = cfg.d_model as usize;
        let ff = cfg.d_ff as usize;
        let hq = cfg.q_dim() as usize;
        let hkv = cfg.kv_dim() as usize;
        let hd = cfg.head_dim as usize;
        let mut out = Vec::new();
        for l in 0..cfg.n_layers {
            let p = |s: &str| format!("blocks.{l}.{s}");
            out.push((p("ln1.weight"), d));
            out.push((p("attn.wq.weight"), hq * d));
            out.push((p("attn.wk.weight"), hkv * d));
            out.push((p("attn.wv.weight"), hkv * d));
            out.push((p("attn.q_norm.weight"), hd));
            out.push((p("attn.k_norm.weight"), hd));
            out.push((p("attn.wo.weight"), d * hq));
            out.push((p("ln2.weight"), d));
            out.push((p("mlp.gate.weight"), ff * d));
            out.push((p("mlp.up.weight"), ff * d));
            out.push((p("mlp.down.weight"), d * ff));
        }
        out.push(("norm.weight".to_string(), d));
        out
    }

    /// Build on an existing device handle (see `gpu_core::Gpu::share`) so a
    /// process holds ONE device however many components it loads. `pub`
    /// (not just `pub(crate)`) so a caller with weights already in hand --
    /// e.g. a real-weight parity test reading straight from an HF mmap,
    /// bypassing `ParamStore`/file I/O entirely, the same pattern
    /// `crates/omni`'s other real-weight tests use -- doesn't need a round
    /// trip through a brain checkpoint file first.
    pub fn build_on(
        gpu: Gpu,
        cfg: MtpConfig,
        decoder: std::collections::HashMap<String, Vec<f32>>,
        codec_embedding: Vec<Vec<f32>>,
        lm_head: Vec<Vec<f32>>,
    ) -> MtpModel {
        let t = cfg.num_code_groups;
        let roles = Self::decoder_param_list(&cfg)
            .into_iter()
            .map(|(n, c)| (n, c, paramstore::Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, &decoder);

        let n = t as u64;
        let d = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let hq = cfg.q_dim() as u64;
        let hkv = cfg.kv_dim() as u64;
        let bht2 = (cfg.n_heads * t * t) as u64;
        let st = |x: u64| gpu.storage(x);

        let mut res = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(n * d));
        }
        let mut layers = Vec::new();
        for _ in 0..cfg.n_layers {
            layers.push(Layer {
                xn1: st(n * d),
                q_pre: st(n * hq),
                q: st(n * hq),
                k_pre: st(n * hkv),
                k: st(n * hkv),
                v: st(n * hkv),
                probs: st(bht2),
                ctx: st(n * hq),
                xmid: st(n * d),
                xn2: st(n * d),
                gate_pre: st(n * ff),
                up: st(n * ff),
                h: st(n * ff),
            });
        }
        let mut m = MtpModel {
            cfg,
            t,
            ps,
            res,
            layers,
            proj: st(n * d),
            mlp_out: st(n * d),
            scores: st(bht2),
            xn_final: st(n * d),
            fwd_steps: Vec::new(),
            codec_embedding,
            lm_head,
            gpu,
        };
        m.fwd_steps = m.forward_steps();
        m
    }

    fn forward_steps(&self) -> Vec<Step> {
        let c = &self.cfg;
        let n = self.t;
        let d = c.d_model;
        let ff = c.d_ff;
        let hd = c.head_dim;
        let hq = c.q_dim();
        let hkv = c.kv_dim();
        let nh = c.n_heads;
        let nkv = c.n_kv_heads;
        let ids = Self::only_fwd_ids();
        let ga = Gqa {
            b: 1,
            t: n,
            n_heads: nh,
            n_kv_heads: nkv,
            head_dim: hd,
        };
        let theta = c.rope_theta;
        let g = &self.gpu;
        let w = |name: &str| self.ps.w(name);
        let mut s: Vec<Step> = Vec::new();

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = |name: &str| format!("blocks.{l}.{name}");
            s.push(block::rmsnorm_fwd(
                g,
                &ids,
                &self.res[l],
                w(&p("ln1.weight")),
                &lb.xn1,
                d,
                n,
            ));
            s.push(g.step(
                MATMUL,
                &[&lb.xn1, w(&p("attn.wq.weight")), &lb.q_pre],
                &[n, d, hq],
                n * hq,
            ));
            s.push(g.step(
                MATMUL,
                &[&lb.xn1, w(&p("attn.wk.weight")), &lb.k_pre],
                &[n, d, hkv],
                n * hkv,
            ));
            s.push(g.step(
                MATMUL,
                &[&lb.xn1, w(&p("attn.wv.weight")), &lb.v],
                &[n, d, hkv],
                n * hkv,
            ));
            s.push(block::rmsnorm_fwd(
                g,
                &ids,
                &lb.q_pre,
                w(&p("attn.q_norm.weight")),
                &lb.q,
                hd,
                n * nh,
            ));
            s.push(block::rmsnorm_fwd(
                g,
                &ids,
                &lb.k_pre,
                w(&p("attn.k_norm.weight")),
                &lb.k,
                hd,
                n * nkv,
            ));
            s.push(block::rope_fwd(g, &ids, &lb.q, n, nh, hd, hq, n, theta));
            s.push(block::rope_fwd(g, &ids, &lb.k, n, nkv, hd, hkv, n, theta));
            s.extend(block::gqa_fwd(
                g,
                &ids,
                &ga,
                &lb.q,
                &lb.k,
                &lb.v,
                &self.scores,
                &lb.probs,
                &lb.ctx,
            ));
            s.push(g.step(
                MATMUL,
                &[&lb.ctx, w(&p("attn.wo.weight")), &self.proj],
                &[n, hq, d],
                n * d,
            ));
            s.push(g.step(ADD2, &[&self.res[l], &self.proj, &lb.xmid], &[n * d], n * d));
            s.push(block::rmsnorm_fwd(
                g,
                &ids,
                &lb.xmid,
                w(&p("ln2.weight")),
                &lb.xn2,
                d,
                n,
            ));
            s.push(g.step(
                MATMUL,
                &[&lb.xn2, w(&p("mlp.gate.weight")), &lb.gate_pre],
                &[n, d, ff],
                n * ff,
            ));
            s.push(g.step(
                MATMUL,
                &[&lb.xn2, w(&p("mlp.up.weight")), &lb.up],
                &[n, d, ff],
                n * ff,
            ));
            s.push(block::swiglu_fwd(
                g,
                &ids,
                &lb.gate_pre,
                &lb.up,
                &lb.h,
                n * ff,
            ));
            s.push(g.step(
                MATMUL,
                &[&lb.h, w(&p("mlp.down.weight")), &self.mlp_out],
                &[n, ff, d],
                n * d,
            ));
            s.push(g.step(
                ADD2,
                &[&lb.xmid, &self.mlp_out, &self.res[l + 1]],
                &[n * d],
                n * d,
            ));
        }
        let last = c.n_layers as usize;
        s.push(block::rmsnorm_fwd(
            g,
            &ids,
            &self.res[last],
            w("norm.weight"),
            &self.xn_final,
            d,
            n,
        ));
        s
    }

    /// Run the decoder over an assembled `[num_code_groups, d_model]` input
    /// embedding sequence and return the residual-codebook logits, shape
    /// `[(num_code_groups - 1) * vocab]` (row `i` = logits for codebook `i+1`,
    /// produced by `lm_head[i]` from decoder position `i+1`).
    pub fn logits(&self, inputs_embeds: &[f32]) -> Vec<f32> {
        let d = self.cfg.d_model as usize;
        let v = self.cfg.vocab as usize;
        let t = self.t as usize;
        assert_eq!(
            inputs_embeds.len(),
            t * d,
            "inputs_embeds must be [num_code_groups, d_model]"
        );
        self.gpu
            .write(&self.res[0], bytemuck::cast_slice(inputs_embeds));
        self.gpu.submit(&[], &self.fwd_steps);
        let hidden = self.gpu.read(&self.xn_final, t * d);
        let mut out = vec![0.0f32; (t - 1) * v];
        for i in 1..t {
            let h = &hidden[i * d..(i + 1) * d];
            let head = &self.lm_head[i - 1];
            let dst = &mut out[(i - 1) * v..i * v];
            for (o, dv) in dst.iter_mut().enumerate() {
                let wrow = &head[o * d..(o + 1) * d];
                let mut acc = 0.0f32;
                for k in 0..d {
                    acc += wrow[k] * h[k];
                }
                *dv = acc;
            }
        }
        out
    }

    /// Assemble the input-embedding sequence for one frame. `talker_hidden` is the
    /// Talker hidden state (`[d_model]`); `cb0_embed` is the Talker codec-0
    /// embedding (`[d_model]`, supplied by the Talker since the MTP does not own
    /// that table); `residual_codes` are codebooks `1..=(num_code_groups-2)`
    /// (length `num_code_groups - 2`), embedded by the MTP's own tables. Returns
    /// `[num_code_groups, d_model]`.
    pub fn assemble(
        &self,
        talker_hidden: &[f32],
        cb0_embed: &[f32],
        residual_codes: &[u32],
    ) -> Vec<f32> {
        let d = self.cfg.d_model as usize;
        let t = self.t as usize;
        assert_eq!(talker_hidden.len(), d);
        assert_eq!(cb0_embed.len(), d);
        assert_eq!(residual_codes.len(), t.saturating_sub(2));
        let mut out = vec![0.0f32; t * d];
        out[0..d].copy_from_slice(talker_hidden);
        out[d..2 * d].copy_from_slice(cb0_embed);
        for (i, &code) in residual_codes.iter().enumerate() {
            // position 2+i embeds codebook (i+1) via codec_embedding[i].
            let tbl = &self.codec_embedding[i];
            let src = code as usize * d;
            out[(2 + i) * d..(3 + i) * d].copy_from_slice(&tbl[src..src + d]);
        }
        out
    }

    /// Per-frame residual codebook generation (greedy). Given the Talker final
    /// hidden state at this frame (`talker_hidden`, `[d_model]`) and the Talker
    /// codebook-0 embedding (`cb0_embed`, `[d_model]`, from the Talker's own
    /// table), autoregressively predict residual codebooks `1..=15` and return
    /// `(codes, residual_embed_sum)`:
    ///   * `codes` — the 15 residual codebook ids (codebooks 1..15),
    ///   * `residual_embed_sum` — `Σ_{i=1}^{15} codec_embedding[i-1][code_i]`
    ///     (`[d_model]`), the residual part of the frame's feedback embedding.
    ///
    /// Mirrors `code_predictor.generate` in `modeling_qwen3_tts.py`: position 0 is
    /// the Talker hidden, position 1 the cb0 embed, position `i+1` (`i>=1`) the
    /// embedding of codebook `i`; `lm_head[i-1]` reads hidden position `i+1` to
    /// predict codebook `i+1`. Because attention is causal, predicting codebook
    /// `k` only needs positions `0..=k` filled, so we grow the sequence in place
    /// (future positions stay zero and never influence the read position).
    pub fn generate_residuals(
        &self,
        talker_hidden: &[f32],
        cb0_embed: &[f32],
    ) -> (Vec<u32>, Vec<f32>) {
        let d = self.cfg.d_model as usize;
        let v = self.cfg.vocab as usize;
        let t = self.t as usize; // num_code_groups (16)
        let nres = t - 1; // 15
        assert_eq!(talker_hidden.len(), d);
        assert_eq!(cb0_embed.len(), d);

        let mut emb = vec![0.0f32; t * d];
        emb[0..d].copy_from_slice(talker_hidden);
        emb[d..2 * d].copy_from_slice(cb0_embed);

        let mut codes = vec![0u32; nres];
        let mut res_sum = vec![0.0f32; d];
        // k = codebook index being predicted (1..=15); head index = k-1; read pos = k.
        for k in 1..=nres {
            let logits = self.logits(&emb); // [(t-1)*v]
            let row = &logits[(k - 1) * v..k * v];
            let mut best = 0usize;
            for j in 1..v {
                if row[j] > row[best] {
                    best = j;
                }
            }
            codes[k - 1] = best as u32;
            // codec_embedding[k-1] embeds codebook k.
            let tbl = &self.codec_embedding[k - 1];
            let r = &tbl[best * d..(best + 1) * d];
            for j in 0..d {
                res_sum[j] += r[j];
            }
            if k < nres {
                // position k+1 carries the embedding of codebook k for the next step.
                emb[(k + 1) * d..(k + 2) * d].copy_from_slice(r);
            }
        }
        (codes, res_sum)
    }

    /// Residual codebook embedding row: `codec_embedding[residual_idx][code]`
    /// (`[d_model]`). `residual_idx` is `0..=14` (codebook `residual_idx + 1`).
    /// Used to build the reference-audio codec embedding in ICL voice-clone
    /// prompts.
    pub fn codec_embed(&self, residual_idx: usize, code: u32) -> &[f32] {
        let d = self.cfg.d_model as usize;
        let s = code as usize * d;
        &self.codec_embedding[residual_idx][s..s + d]
    }

    /// Load an inference-only MTP from a brain checkpoint written by
    /// [`crate::import::import_mtp`].
    pub fn load_inference(path: &str) -> MtpModel {
        Self::load_inference_on(Gpu::new(PIPELINES), path)
    }

    /// Build on an existing device handle (see `gpu_core::Gpu::share`).
    pub fn load_inference_on(gpu: Gpu, path: &str) -> MtpModel {
        let c = checkpoint::load(path);
        let cfg = MtpConfig::from_brain_json(&c.header["config"]);
        let take = |name: &str| {
            c.find(name, "")
                .cloned()
                .unwrap_or_else(|| panic!("missing {name}"))
        };
        let mut decoder = std::collections::HashMap::new();
        for (n, _) in Self::decoder_param_list(&cfg) {
            let d = take(&n);
            decoder.insert(n, d);
        }
        let nres = cfg.n_residual() as usize;
        let codec_embedding = (0..nres)
            .map(|i| take(&format!("codec_embedding.{i}.weight")))
            .collect();
        let lm_head = (0..nres)
            .map(|i| take(&format!("lm_head.{i}.weight")))
            .collect();
        MtpModel::build_on(gpu, cfg, decoder, codec_embedding, lm_head)
    }

    /// Build a randomly-initialised MTP for tests.
    pub fn new_synthetic(cfg: MtpConfig, seed: u64) -> MtpModel {
        Self::new_synthetic_on(Gpu::new(PIPELINES), cfg, seed)
    }

    pub(crate) fn new_synthetic_on(gpu: Gpu, cfg: MtpConfig, seed: u64) -> MtpModel {
        use data::rng::Rng;
        let mut rng = Rng::new(seed);
        let mut normal = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|_| (rng.next_gaussian() as f32) * s).collect()
        };
        let proj_std = 0.02f32 / ((2.0 * cfg.n_layers as f32).sqrt());
        let mut decoder = std::collections::HashMap::new();
        for (n, numel) in Self::decoder_param_list(&cfg) {
            let v = if n.ends_with("norm.weight")
                || n.ends_with("ln1.weight")
                || n.ends_with("ln2.weight")
            {
                vec![1.0; numel]
            } else if n.ends_with("attn.wo.weight") || n.ends_with("mlp.down.weight") {
                normal(numel, proj_std)
            } else {
                normal(numel, 0.02)
            };
            decoder.insert(n, v);
        }
        let nres = cfg.n_residual() as usize;
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let codec_embedding = (0..nres).map(|_| normal(v * d, 0.02)).collect();
        let lm_head = (0..nres).map(|_| normal(v * d, 0.02)).collect();
        MtpModel::build_on(gpu, cfg, decoder, codec_embedding, lm_head)
    }
}

impl crate::prompt::MtpHost for MtpModel {
    fn codec_embed(&self, residual_idx: usize, code: u32) -> &[f32] {
        MtpModel::codec_embed(self, residual_idx, code)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_disabled() -> bool {
        std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
    }

    #[test]
    fn forward_shape_and_finite() {
        if gpu_disabled() {
            return;
        }
        let cfg = MtpConfig::tiny();
        let t = cfg.num_code_groups as usize;
        let d = cfg.d_model as usize;
        let v = cfg.vocab as usize;
        let m = MtpModel::new_synthetic_on(gpu_core::testgpu::dev(PIPELINES), cfg, 5);
        let embeds: Vec<f32> = (0..t * d).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
        let logits = m.logits(&embeds);
        assert_eq!(logits.len(), (t - 1) * v);
        assert!(logits.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn assemble_layout() {
        if gpu_disabled() {
            return;
        }
        let cfg = MtpConfig::tiny(); // num_code_groups = 4 -> residual_codes len 2
        let d = cfg.d_model as usize;
        let m = MtpModel::new_synthetic_on(gpu_core::testgpu::dev(PIPELINES), cfg, 1);
        let th = vec![0.5f32; d];
        let cb0 = vec![-0.5f32; d];
        let embeds = m.assemble(&th, &cb0, &[1, 2]);
        assert_eq!(embeds.len(), 4 * d);
        assert_eq!(&embeds[0..d], &th[..]);
        assert_eq!(&embeds[d..2 * d], &cb0[..]);
        let logits = m.logits(&embeds);
        assert!(logits.iter().all(|x| x.is_finite()));
    }
}

/// The coalesced RMSNorm this model now selects (`rmsnorm_rows`, via
/// `block::rms_variant` inside `block::rmsnorm_fwd`) is NOT bit-identical to
/// the per-element `rmsnorm` it replaced: 64 partial sums fold in a different
/// order. It was adopted for throughput, so what it computes is gated here,
/// against a HOST reference, at the shapes THIS model's decode tape really
/// dispatches - narrow rows are where the two reduction orders differ most,
/// and they are also the whole reason the swap is worth making.
#[cfg(test)]
mod rmsnorm_variant_agreement {
    use super::*;

    #[test]
    fn the_registered_slot_names_the_coalesced_kernel() {
        assert_eq!(PIPELINES[MtpModel::only_fwd_ids().rmsnorm_rows].0, "rmsnorm_rows");
    }

    #[test]
    fn the_tape_norms_match_the_host_reference() {
        // The MTP tape is the one adopting tape here that is NOT
        // decode-shaped: its rows are the 16 code groups, never 1, and it is
        // replayed 15 times per audio frame. Real MTP: d_model 1024, 16/8
        // heads of 128, num_code_groups 16.
        let shapes = [(16, 1024, "ln1/ln2/final norm"), (256, 128, "q_norm (t*n_heads)"), (128, 128, "k_norm (t*n_kv_heads)")];
        let gpu = gpu_core::testgpu::dev(PIPELINES);
        model::block::assert_rmsnorm_variant_agrees(&gpu, &MtpModel::only_fwd_ids(), &shapes);
    }
}
