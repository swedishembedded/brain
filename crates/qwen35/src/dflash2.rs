// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **DFlash2** - a real block-diffusion DRAFT model for
//! [`crate::int8_gguf_resident::Qwen35GgufInstance::generate_speculative`].
//!
//! Everything else that has driven that loop so far was a stand-in: an oracle
//! fed the answer (the ceiling, not a drafter) and an n-gram repeat detector
//! (a real drafter, and a measured LOSS on free-form text). This is the first
//! one that runs a model.
//!
//! **What makes it different from a small autoregressive drafter.** It is not
//! autoregressive at all. One forward pass over `[anchor, MASK, MASK, ...]`
//! denoises the whole block at once - `block_size` rows, `block_size - 1`
//! proposals - so `k` draft tokens cost ONE draft forward, not `k` of them.
//! That is the entire economic argument for it on this stack: a target decode
//! step measures 152 ms here and a whole draft round measures 48 ms (44 ms of
//! device work, 4 ms of selector), so a per-token drafter would have to fit
//! seven of its own passes inside one target step and this one does not have
//! to.
//!
//! Measured (2x Tesla P40, both checkpoints Q8_0 served INT8, greedy): 2.30
//! accepted draft tokens per round on a free-form prompt and 6.00 on a
//! repetition workload, against the model-free n-gram drafter's 0.57 and 5.12,
//! and on free-form text the first drafter on this stack that is a net
//! win (1.08x at `k = 3`) rather than a loss. `tests/dflash2_real.rs` carries
//! the full table and the cost model that explains why the wall-clock win is
//! much smaller than the acceptance rate alone implies.
//!
//! Four things carry the quality that a plain masked denoiser would not have,
//! and all four are re-derived from the checkpoint in this file rather than
//! assumed:
//!
//! * **It reads the target's INTERNALS, not its output.** Each draft layer's
//!   keys and values are computed over the concatenation of (a) the target's
//!   own residual at layers `target_layers`, projected by `fc` and normalized
//!   by `enc.output_norm`, one row per context token, and (b) the block's own
//!   rows. So the drafter's "prompt" is 25600 numbers per token that only the
//!   target can produce - which is why
//!   [`Qwen35GgufInstance::enable_hidden_taps`] exists.
//! * **Attention is NON-causal** (`dflash.attention.causal = false`), within a
//!   2048 sliding window in BOTH directions. The block's mask rows see each
//!   other, which is what a diffusion denoiser needs and what a causal kernel
//!   cannot express.
//! * **Two-tap dynamic convolutions**, before and after each of the attention
//!   and MLP sublayers. The taps are not weights: half of a `[5120 -> 1280]`
//!   projection of the activations IS the tap set, per token, per group of 16
//!   channels, added to a static per-channel base. This is what the model card
//!   credits with keeping the draft from decaying toward the end of a block,
//!   and it needed a new kernel (`kernels::DYN_GROUP_CONV1D`) - nothing in the
//!   tree had a data-dependent conv weight.
//! * **A selector**, not an argmax. The head's top-16 candidates at every
//!   position are scored against the token actually chosen at the previous
//!   position through a rank-256 three-way Hadamard product, and the walk is
//!   greedy left to right. Picking each position's argmax independently, which
//!   is what a non-diffusion drafter does, produces a block whose tokens are
//!   individually likely and jointly incoherent.
//!
//! **What this module does NOT own.** The token embedding and the `lm_head`
//! are the TARGET's - the draft checkpoint ships neither, by design - so the
//! noise embedding is gathered from the target's `token_embd.weight` and the
//! draft's hidden states are projected by the target's own resident INT8 head
//! (with the DRAFT's `output_norm` standing in for the target's final norm,
//! which is exactly the composition the reference implementation performs).
//! That also means no second 1.27 GB head copy on the cards.
//!
//! Gated against a dependency-free host reference over the same bytes,
//! `tools/goldens/dflash2_reference_forward.py`, which re-implements the
//! published `dflash/model.py` and is what every claim above was checked
//! against - see `crates/qwen35/tests/dflash2_real.rs`.
//!
//! Swedish Embedded AB implements speculative decoding - drafters, verifiers
//! and the state rollback a hybrid recurrent stack needs - for clients serving
//! large models on modest hardware. If your team needs inference throughput
//! without a second model's worth of accuracy loss, you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::cell::RefCell;

use checkpoint::gguf::{GgufValue, MmapGguf};
use gpu_core::select::Dtype;
use gpu_core::{DeviceBuffer, Gpu};
use model::ops::{Ops, Weight};

use crate::int8_gguf_resident::Qwen35GgufInstance;

const MODEL: &str = "dflash2";

/// The GGUF architecture string this module loads.
pub const GGUF_ARCHITECTURE: &str = "dflash";

/// How many context rows one [`Dflash2::append_context`] round projects at a
/// time. The `fc` leaf is `[5120, 25600]`, so a round's input block is
/// `rows * 25600` f32 - 26 MB at 256 rows, and a 250k-token prompt would be
/// 25 GB in one allocation if this were unbounded.
const CTX_CHUNK: u32 = 256;

// ------------------------------------------------------------------- config

/// The draft model's shape, read from the checkpoint's own `dflash.*` metadata.
///
/// Every field here was cross-checked against the published `config.json`
/// (`dflash_config`) and against the real tensor shapes; where the two
/// conventions differ, the difference is recorded on the field.
#[derive(Clone, Debug)]
pub struct Dflash2Config {
    pub n_layers: usize,
    pub d_model: u32,
    pub d_ff: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub eps: f32,
    pub rope_theta: f32,
    /// Sliding window, applied in BOTH directions because the attention is
    /// non-causal.
    pub window: u32,
    /// Rows per denoising block: one anchor plus `block_size - 1` masks, so
    /// this is the largest `k` the drafter can answer.
    pub block_size: u32,
    pub conv_kernel: u32,
    pub conv_group: u32,
    pub selector_rank: u32,
    pub selector_top_k: u32,
    pub mask_token_id: u32,
    /// `false` in every published DFlash2 checkpoint; kept as a field rather
    /// than a constant because it is a metadata key and metadata is what this
    /// loader is supposed to believe.
    pub causal: bool,
    /// **0-based TARGET decoder-layer indices** whose residual feeds `fc`, in
    /// concatenation order.
    ///
    /// The GGUF stores these one higher than the HF config does
    /// (`[6, 20, 34, 48, 62]` against `target_layer_ids: [5, 19, 33, 47, 61]`)
    /// because the reference indexes the `hidden_states` TUPLE, whose entry 0
    /// is the embedding output and whose entry `i+1` is decoder layer `i`'s
    /// output. This field carries the decoder-layer form - what a tap actually
    /// needs - so the conversion happens once, here.
    pub target_layers: Vec<usize>,
    /// The TARGET's vocabulary, which is also the draft's (it reuses the
    /// target's head).
    pub vocab: u32,
}

impl Dflash2Config {
    /// Read the shape from a DFlash2 GGUF, checking the metadata against the
    /// file's own tensor shapes rather than trusting either alone.
    pub fn from_gguf(mg: &MmapGguf) -> Result<Dflash2Config, String> {
        let kv = gguf::kv::ArchKv::expect_architecture(mg, GGUF_ARCHITECTURE)?;
        let d_model = kv.req_u32("embedding_length")?;
        let n_layers = kv.req_u32("block_count")? as usize;
        let head_dim = kv.req_u32("attention.key_length")?;
        let target_layers = match mg.kv().get("dflash.target_layers") {
            Some(GgufValue::Array(items)) => items
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|i| i as usize - 1)
                        .ok_or_else(|| format!("{MODEL}: dflash.target_layers holds a non-integer"))
                })
                .collect::<Result<Vec<usize>, String>>()?,
            _ => return Err(format!("{MODEL}: dflash.target_layers missing or not an array")),
        };
        let mask_token_id = mg
            .kv()
            .get("tokenizer.ggml.mask_token_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| format!("{MODEL}: tokenizer.ggml.mask_token_id missing - a masked denoiser cannot draft without it"))?
            as u32;
        let cfg = Dflash2Config {
            n_layers,
            d_model,
            d_ff: kv.req_u32("feed_forward_length")?,
            n_heads: kv.req_u32("attention.head_count")?,
            n_kv_heads: kv.req_u32("attention.head_count_kv")?,
            head_dim,
            eps: kv.f32_or("attention.layer_norm_rms_epsilon", 1e-6),
            rope_theta: kv.req_f32("rope.freq_base")?,
            window: kv.u32_or("attention.sliding_window", u32::MAX),
            block_size: kv.req_u32("block_size")?,
            conv_kernel: kv.req_u32("conv_kernel_size")?,
            conv_group: kv.req_u32("conv_group_size")?,
            selector_rank: kv.req_u32("selector_rank")?,
            selector_top_k: kv.req_u32("selector_top_k")?,
            mask_token_id,
            causal: kv.bool("attention.causal").unwrap_or(true),
            target_layers,
            // The draft ships no head, so the vocabulary is whatever the
            // selector codebooks are indexed by - which is the target's.
            vocab: mg.shape("selector_predecessor.weight").map(|s| s[0] as u32).ok_or_else(|| format!("{MODEL}: selector_predecessor.weight missing"))?,
        };

        // Shapes are ground truth; metadata is a claim about them. Check the
        // three that a wrong value would make silently wrong rather than
        // loudly wrong.
        let want = |name: &str, rows: u32, k: u32| -> Result<(), String> {
            let got = mg.shape(name).ok_or_else(|| format!("{MODEL}: {name} missing"))?;
            if got != vec![rows as usize, k as usize] {
                return Err(format!("{MODEL}: {name} is {got:?}, but the metadata implies [{rows}, {k}]"));
            }
            Ok(())
        };
        want("blk.0.attn_q.weight", cfg.n_heads * head_dim, d_model)?;
        want("blk.0.attn_k.weight", cfg.n_kv_heads * head_dim, d_model)?;
        want("fc.weight", d_model, cfg.target_layers.len() as u32 * d_model)?;
        want("blk.0.attn_conv_proj.weight", 2 * cfg.conv_kernel * (d_model / cfg.conv_group), d_model)?;
        Ok(cfg)
    }

    /// One dynamic-conv projection row: `2 * taps * groups`, where the leading
    /// 2 is "one set of taps for the convolution entering the sublayer, one
    /// for the convolution leaving it".
    fn dyn_width(&self) -> u32 {
        2 * self.conv_kernel * self.groups()
    }

    fn groups(&self) -> u32 {
        self.d_model / self.conv_group
    }

    fn kv_dim(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    fn q_dim(&self) -> u32 {
        self.n_heads * self.head_dim
    }
}

// ------------------------------------------------------------------ weights

struct Layer {
    attn_norm: DeviceBuffer,
    ffn_norm: DeviceBuffer,
    q_norm: DeviceBuffer,
    k_norm: DeviceBuffer,
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
    /// `[2, taps, d_model]`: `[0]` is the base kernel of the convolution
    /// ENTERING the sublayer, `[1]` of the one leaving it.
    attn_conv_base: DeviceBuffer,
    attn_conv_proj: Weight,
    ffn_conv_base: DeviceBuffer,
    ffn_conv_proj: Weight,
}

/// Kernel pipeline indices, resolved by name once at load.
#[derive(Clone, Copy)]
struct Kids {
    rmsnorm: usize,
    add2: usize,
    silu_mul: usize,
    splice: usize,
    rope2d: usize,
    dyn_conv: usize,
    scores: usize,
    softmax: usize,
    apply: usize,
}

/// This model's kernel set: the target decoder's own list (which already
/// carries every `model::ops::Ops` requirement, the paged-attention triad this
/// module attends through, and `splice`), plus the two it adds - FULL
/// table-driven RoPE (the target rotates only 64 of its 256 head channels and
/// so registers `rope2d_partial` instead) and the dynamic conv.
pub fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut v = crate::model::pipelines().to_vec();
        v.push(("rope2d", kernels::ROPE2D));
        v.push(("dyn_group_conv1d", kernels::DYN_GROUP_CONV1D));
        v
    })
}

/// A loaded DFlash2 draft model, resident on one card.
pub struct Dflash2 {
    gpu: Gpu,
    ops: Ops,
    k: Kids,
    /// The weight tier this instance was loaded at - also decides whether an
    /// activation needs packing (`Ops::act`) or not (`Ops::act_f32`), which
    /// `Ops` refuses to guess on a caller's behalf.
    dt: Dtype,
    pub cfg: Dflash2Config,
    /// Kept open for the selector codebooks, which are read by ROW (17 of
    /// 248320 per position) and so are never materialized.
    mg: MmapGguf,
    fc: Weight,
    /// `enc.output_norm`: applied AFTER `fc`, not before it.
    hidden_norm: DeviceBuffer,
    /// `output_norm`, on the HOST - it is applied by the target's head path,
    /// on the target's card, standing in for the target's own final norm.
    out_norm: Vec<f32>,
    /// `selector_hidden.weight`, `[rank, d_model]`, on the host: 7 rows per
    /// round against a 1.3M-element matrix is not worth a dispatch.
    sel_hidden: Vec<f32>,
    layers: Vec<Layer>,
    /// Per-layer K/V over `[cap + block_size, kv_dim]`. Row index IS absolute
    /// token position, which is what makes a speculative rollback free: the
    /// re-commit pass rewrites exactly the rows it invalidated.
    kcache: Vec<DeviceBuffer>,
    vcache: Vec<DeviceBuffer>,
    cache_cap: u32,
}

impl Dflash2 {
    /// [`Self::load_dt`] at the serving tier, INT8.
    pub fn load(gpu: Gpu, gguf_path: &str, cap: u32) -> Result<Dflash2, String> {
        Self::load_dt(gpu, gguf_path, cap, Dtype::I8)
    }

    /// Load the draft checkpoint onto `gpu`, with room for `cap` context
    /// positions, at weight tier `dt`.
    ///
    /// `Dtype::I8` is the serving tier: weights land as group-wise INT8
    /// straight out of the file's Q8_0 blocks
    /// (`model::int8::upload_quantized`) - no fp32 intermediate, the same
    /// route the target's own resident takes, and the reason a 1.9B draft is
    /// ~2 GB beside a 27 GB target rather than 7.6.
    ///
    /// `Dtype::F32` exists for the host-reference gate and is 7.4 GB. It is
    /// not an academic option: this model amplifies its input by five orders
    /// of magnitude (the embedding it starts from has RMS 0.014, the hidden
    /// state it ends with has RMS ~1000), so "the INT8 and fp32 answers differ
    /// by 10%" and "the port is wrong by 10%" are indistinguishable without
    /// running the tier the reference actually implements. Measured: see
    /// `tests/dflash2_real.rs`.
    pub fn load_dt(gpu: Gpu, gguf_path: &str, cap: u32, dt: Dtype) -> Result<Dflash2, String> {
        let mg = MmapGguf::open(gguf_path).map_err(|e| format!("{MODEL}: {gguf_path}: {e}"))?;
        let cfg = Dflash2Config::from_gguf(&mg)?;
        let ops = Ops::new(gpu.share())?;
        let kid = |n: &str| gpu.kernel_index(n).ok_or_else(|| format!("{MODEL}: kernel {n:?} is not registered - use dflash2::pipelines()"));
        let k = Kids {
            rmsnorm: kid("rmsnorm")?,
            add2: kid("add2")?,
            silu_mul: kid("silu_mul")?,
            splice: kid("splice")?,
            rope2d: kid("rope2d")?,
            dyn_conv: kid("dyn_group_conv1d")?,
            scores: kid("paged_decode_scores_batched")?,
            softmax: kid("decode_softmax_batched")?,
            apply: kid("paged_decode_apply_batched")?,
        };

        let mut up = paramstore::upload::Uploader::new(&gpu);
        let f32_buf = |name: &str| -> Result<DeviceBuffer, String> {
            let v = mg.tensor(name).ok_or_else(|| format!("{MODEL}: {name} missing"))??;
            Ok(gpu.storage_init(name, &v))
        };
        let host = |name: &str| -> Result<Vec<f32>, String> { mg.tensor(name).ok_or_else(|| format!("{MODEL}: {name} missing"))? };
        let quant = |up: &mut paramstore::upload::Uploader, name: &str| -> Result<Weight, String> {
            let shape = mg.shape(name).ok_or_else(|| format!("{MODEL}: {name} missing"))?;
            let (n, kk) = (shape[0], shape[1]);
            if dt == Dtype::I8 {
                let (w, s) = model::int8::upload_quantized(up, &mg, name, n, kk)?;
                return Ok(Weight::I8 { w, s, n: n as u32, k: kk as u32 });
            }
            let raw = mg.tensor(name).ok_or_else(|| format!("{MODEL}: {name} missing"))??;
            Ok(Weight::upload(&ops, &raw, n, kk, dt))
        };

        let fc = quant(&mut up, "fc.weight")?;
        let hidden_norm = f32_buf("enc.output_norm.weight")?;
        let out_norm = host("output_norm.weight")?;
        let sel_hidden = host("selector_hidden.weight")?;

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for l in 0..cfg.n_layers {
            let p = |leaf: &str| format!("blk.{l}.{leaf}");
            layers.push(Layer {
                attn_norm: f32_buf(&p("attn_norm.weight"))?,
                ffn_norm: f32_buf(&p("ffn_norm.weight"))?,
                q_norm: f32_buf(&p("attn_q_norm.weight"))?,
                k_norm: f32_buf(&p("attn_k_norm.weight"))?,
                wq: quant(&mut up, &p("attn_q.weight"))?,
                wk: quant(&mut up, &p("attn_k.weight"))?,
                wv: quant(&mut up, &p("attn_v.weight"))?,
                wo: quant(&mut up, &p("attn_output.weight"))?,
                gate: quant(&mut up, &p("ffn_gate.weight"))?,
                up: quant(&mut up, &p("ffn_up.weight"))?,
                down: quant(&mut up, &p("ffn_down.weight"))?,
                attn_conv_base: f32_buf(&p("attn_conv_base"))?,
                attn_conv_proj: quant(&mut up, &p("attn_conv_proj.weight"))?,
                ffn_conv_base: f32_buf(&p("ffn_conv_base"))?,
                ffn_conv_proj: quant(&mut up, &p("ffn_conv_proj.weight"))?,
            });
        }

        let cache_cap = cap + cfg.block_size;
        let rows = (cache_cap * cfg.kv_dim()) as u64;
        let kcache: Vec<DeviceBuffer> = (0..cfg.n_layers).map(|_| gpu.storage(rows)).collect();
        let vcache: Vec<DeviceBuffer> = (0..cfg.n_layers).map(|_| gpu.storage(rows)).collect();
        Ok(Dflash2 { gpu, ops, k, dt, cfg, mg, fc, hidden_norm, out_norm, sel_hidden, layers, kcache, vcache, cache_cap })
    }

    /// Host-side cos/sin tables for `positions`, `[n, head_dim/2]` each, in
    /// the half-split pairing `rope2d.wgsl` reads.
    ///
    /// FULL rotation over every head channel. This is the one architectural
    /// detail most likely to be carried over wrong from the target, which
    /// rotates `rope.dimension_count = 64` of its 256: the draft's own config
    /// says `rope_type: "default"` with no partial factor, and its GGUF says
    /// `rope.dimension_sections = [64, 0, 0, 0]`, which sums to `head_dim / 2`,
    /// the mrope spelling of a full rotation with every section on the text
    /// axis, not a 64-channel partial one.
    fn rope_tables(&self, positions: &[u32]) -> (DeviceBuffer, DeviceBuffer) {
        let half = (self.cfg.head_dim / 2) as usize;
        let mut cos = Vec::with_capacity(positions.len() * half);
        let mut sin = Vec::with_capacity(positions.len() * half);
        for &p in positions {
            for d in 0..half {
                let inv = (self.cfg.rope_theta as f64).powf(-2.0 * d as f64 / self.cfg.head_dim as f64);
                let a = p as f64 * inv;
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
        (self.gpu.storage_init("dflash2.rope.cos", &cos), self.gpu.storage_init("dflash2.rope.sin", &sin))
    }

    /// One activation, packed for an INT8 weight or handed over raw for an
    /// fp32 one. `Ops` deliberately makes this the caller's declaration
    /// rather than inferring it - see `Ops::act_f32`'s own doc.
    fn act(&self, s: &mut Vec<gpu_core::Step>, x: &DeviceBuffer, xr0: u32, rows: u32, k: u32) -> model::ops::Act {
        if self.dt == Dtype::F32 { self.ops.act_f32(x, xr0, rows, k) } else { self.ops.act(s, x, xr0, rows, k) }
    }

    fn rms(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, dim: u32, rows: u32) -> gpu_core::Step {
        model::block::rmsnorm_eps_fwd(&self.gpu, self.k.rmsnorm, x, w, out, dim, rows, self.cfg.eps)
    }

    /// One grouped dynamic causal convolution over the whole block.
    ///
    /// `half` selects which of the projection's two tap sets and which of the
    /// base kernel's two halves to spend - `0` entering the sublayer, `1`
    /// leaving it.
    #[allow(clippy::too_many_arguments)]
    fn dyn_conv(&self, x: &DeviceBuffer, dynb: &DeviceBuffer, base: &DeviceBuffer, out: &DeviceBuffer, n: u32, half: u32) -> gpu_core::Step {
        let c = self.cfg.d_model;
        let taps = self.cfg.conv_kernel;
        let groups = self.cfg.groups();
        self.gpu.step(
            self.k.dyn_conv,
            &[x, dynb, base, out],
            &[n, c, taps, self.cfg.conv_group, groups, self.cfg.dyn_width(), half * taps * groups, half * taps * c],
            n * c,
        )
    }

    /// The non-causal attention triad, over cache rows `0 .. pos + n`.
    ///
    /// Dispatched here rather than through `model::block::gqa_chunk_step`
    /// because that helper's contract is `seq_lens[i] == start + i + 1`, i.e.
    /// CAUSAL - and the whole point of a block-diffusion denoiser is that row
    /// `i` sees rows after it. `seq_lens` is uniform instead: every query row
    /// sees every cached row, the ctx rows and all `n` block rows alike, which
    /// is what `dflash.attention.causal = false` means. The kernels themselves
    /// need no change; `seq_lens` is a per-row live-key COUNT to them and
    /// nothing more.
    #[allow(clippy::too_many_arguments)]
    fn attend(&self, s: &mut Vec<gpu_core::Step>, l: usize, q: &DeviceBuffer, ctx: &DeviceBuffer, pos: u32, n: u32) {
        let (nh, nkv, hd) = (self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim);
        let t_max = pos + n;
        assert!(
            t_max <= self.cache_cap,
            "{MODEL}: a block ending at {t_max} exceeds this instance's {} cached positions",
            self.cache_cap
        );
        assert!(
            self.cfg.window >= t_max,
            "{MODEL}: context {t_max} has outgrown the {}-token sliding window, which this path does not yet apply \
             (every query row would have to drop a different prefix, and the triad below takes one count per row from key 0)",
            self.cfg.window
        );
        let group = nh / nkv;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let block_ids = self.gpu.storage(n as u64);
        self.gpu.write(&block_ids, &vec![0u32; n as usize]);
        let seq_lens = self.gpu.storage(n as u64);
        self.gpu.write(&seq_lens, &vec![t_max; n as usize]);
        let scores = self.gpu.storage((n * nh * t_max) as u64);
        let probs = self.gpu.storage((n * nh * t_max) as u64);
        let kv_stride = self.cfg.kv_dim();
        s.push(self.gpu.step(
            self.k.scores,
            &[q, &self.kcache[l], &block_ids, &seq_lens, &scores],
            &[n, nh, group, hd, self.cache_cap, kv_stride, t_max, 1, scale.to_bits()],
            n * nh * t_max,
        ));
        s.push(self.gpu.step(self.k.softmax, &[&scores, &seq_lens, &probs], &[n, nh, t_max], n * nh));
        s.push(self.gpu.step(
            self.k.apply,
            &[&probs, &self.vcache[l], &block_ids, &seq_lens, ctx],
            &[n, nh, group, hd, self.cache_cap, kv_stride, t_max, 1],
            n * nh * hd,
        ));
    }

    /// **Project and cache the target's hidden states for `m` context tokens**
    /// starting at absolute position `pos_start`.
    ///
    /// `hidden` is `[m, target_layers.len() * d_model]` - the target's residual
    /// at each tapped layer, concatenated in `target_layers` order. It goes
    /// through `fc`, then `enc.output_norm` (that order, which is the one place
    /// this architecture departs from the usual "norm then project"), and then
    /// each layer's own `k_proj`/`v_proj` + per-head norm + RoPE, landing in
    /// the per-layer cache at rows `pos_start .. pos_start + m`.
    ///
    /// Incremental by design: a round pays only for the tokens the target
    /// just committed, never for the whole context, which is what keeps the
    /// drafter's cost flat in context length.
    pub fn append_context(&self, hidden: &[f32], pos_start: u32) -> Result<(), String> {
        let d = self.cfg.d_model;
        let wide = self.cfg.target_layers.len() as u32 * d;
        if !(hidden.len() as u32).is_multiple_of(wide) {
            return Err(format!("{MODEL}: context block of {} floats is not a multiple of {wide}", hidden.len()));
        }
        let total = hidden.len() as u32 / wide;
        for off in (0..total).step_by(CTX_CHUNK as usize) {
            let m = CTX_CHUNK.min(total - off);
            let rows = &hidden[(off * wide) as usize..((off + m) * wide) as usize];
            let base = pos_start + off;
            let _scope = self.gpu.scratch_scope();
            let x = self.gpu.storage_init("dflash2.ctx.in", rows);
            let mut s = Vec::new();
            let a = self.act(&mut s, &x, 0, m, wide);
            let proj = self.gpu.storage((m * d) as u64);
            self.ops.matmul(&mut s, &self.fc, &a, &proj, 0);
            let normed = self.gpu.storage((m * d) as u64);
            s.push(self.rms(&proj, &self.hidden_norm, &normed, d, m));
            let a2 = self.act(&mut s, &normed, 0, m, d);
            let positions: Vec<u32> = (base..base + m).collect();
            let (cos, sin) = self.rope_tables(&positions);
            let kv = self.cfg.kv_dim();
            for l in 0..self.cfg.n_layers {
                let kb = self.gpu.storage((m * kv) as u64);
                self.ops.matmul(&mut s, &self.layers[l].wk, &a2, &kb, 0);
                let vb = self.gpu.storage((m * kv) as u64);
                self.ops.matmul(&mut s, &self.layers[l].wv, &a2, &vb, 0);
                let kn = self.gpu.storage((m * kv) as u64);
                s.push(self.rms(&kb, &self.layers[l].k_norm, &kn, self.cfg.head_dim, m * self.cfg.n_kv_heads));
                s.push(model::block::rope2d_fwd(&self.gpu, self.k.rope2d, &kn, &cos, &sin, m, self.cfg.n_kv_heads, self.cfg.head_dim, kv));
                s.push(model::block::kv_cache_fill_at(&self.gpu, self.k.splice, &kn, &self.kcache[l], base, m, self.cfg.n_kv_heads, self.cfg.head_dim));
                s.push(model::block::kv_cache_fill_at(&self.gpu, self.k.splice, &vb, &self.vcache[l], base, m, self.cfg.n_kv_heads, self.cfg.head_dim));
            }
            self.gpu.submit(&[], &s);
            self.gpu.poll_wait();
            self.gpu.flush();
        }
        Ok(())
    }

    /// **The denoising pass**: one forward over `[anchor, MASK, ...]` at
    /// absolute positions `pos .. pos + n`, returning the PRE-final-norm
    /// hidden state of every row, `[n, d_model]`.
    ///
    /// `noise` is the target's raw embedding of those `n` token ids. The final
    /// norm is deliberately left off: it is applied on the target's card by
    /// the target's own head epilogue, with this model's `output_norm` weights
    /// substituted, so no second copy of a 1.27 GB `lm_head` is needed.
    pub fn denoise_block(&self, noise: &[f32], pos: u32) -> Result<Vec<f32>, String> {
        let d = self.cfg.d_model;
        let n = noise.len() as u32 / d;
        if n == 0 || !(noise.len() as u32).is_multiple_of(d) {
            return Err(format!("{MODEL}: noise embedding of {} floats is not whole rows of {d}", noise.len()));
        }
        let (nh, nkv, hd, ff) = (self.cfg.n_heads, self.cfg.n_kv_heads, self.cfg.head_dim, self.cfg.d_ff);
        let (hq, hkv) = (self.cfg.q_dim(), self.cfg.kv_dim());
        let positions: Vec<u32> = (pos..pos + n).collect();
        let (cos, sin) = self.rope_tables(&positions);
        let mut x = self.gpu.storage_init("dflash2.block.in", noise);

        for l in 0..self.cfg.n_layers {
            let lb = &self.layers[l];
            let _scope = self.gpu.scratch_scope();
            let mut s = Vec::new();

            // ---- attention sublayer, wrapped in its own two convolutions ----
            let xn1 = self.gpu.storage((n * d) as u64);
            s.push(self.rms(&x, &lb.attn_norm, &xn1, d, n));
            let a1 = self.act(&mut s, &xn1, 0, n, d);
            let dyn1 = self.gpu.storage((n * self.cfg.dyn_width()) as u64);
            self.ops.matmul(&mut s, &lb.attn_conv_proj, &a1, &dyn1, 0);
            let h = self.gpu.storage((n * d) as u64);
            s.push(self.dyn_conv(&xn1, &dyn1, &lb.attn_conv_base, &h, n, 0));

            let ah = self.act(&mut s, &h, 0, n, d);
            let qp = self.gpu.storage((n * hq) as u64);
            self.ops.matmul(&mut s, &lb.wq, &ah, &qp, 0);
            let kp = self.gpu.storage((n * hkv) as u64);
            self.ops.matmul(&mut s, &lb.wk, &ah, &kp, 0);
            let vp = self.gpu.storage((n * hkv) as u64);
            self.ops.matmul(&mut s, &lb.wv, &ah, &vp, 0);
            let qn = self.gpu.storage((n * hq) as u64);
            s.push(self.rms(&qp, &lb.q_norm, &qn, hd, n * nh));
            let kn = self.gpu.storage((n * hkv) as u64);
            s.push(self.rms(&kp, &lb.k_norm, &kn, hd, n * nkv));
            s.push(model::block::rope2d_fwd(&self.gpu, self.k.rope2d, &qn, &cos, &sin, n, nh, hd, hq));
            s.push(model::block::rope2d_fwd(&self.gpu, self.k.rope2d, &kn, &cos, &sin, n, nkv, hd, hkv));
            // The block's own rows join the cache at their absolute positions,
            // right after the context rows. Next round's `append_context`
            // overwrites whichever of them the target actually committed with
            // the real thing - which is exactly the reference's "crop the
            // draft cache back to `start`".
            s.push(model::block::kv_cache_fill_at(&self.gpu, self.k.splice, &kn, &self.kcache[l], pos, n, nkv, hd));
            s.push(model::block::kv_cache_fill_at(&self.gpu, self.k.splice, &vp, &self.vcache[l], pos, n, nkv, hd));
            let attn = self.gpu.storage((n * hq) as u64);
            self.attend(&mut s, l, &qn, &attn, pos, n);

            let ao = self.act(&mut s, &attn, 0, n, hq);
            let proj = self.gpu.storage((n * d) as u64);
            self.ops.matmul(&mut s, &lb.wo, &ao, &proj, 0);
            let post = self.gpu.storage((n * d) as u64);
            s.push(self.dyn_conv(&proj, &dyn1, &lb.attn_conv_base, &post, n, 1));
            let xmid = self.gpu.storage((n * d) as u64);
            s.push(self.gpu.step(self.k.add2, &[&x, &post, &xmid], &[n * d], n * d));

            // ---- MLP sublayer, same convolution sandwich ----
            let xn2 = self.gpu.storage((n * d) as u64);
            s.push(self.rms(&xmid, &lb.ffn_norm, &xn2, d, n));
            let a2 = self.act(&mut s, &xn2, 0, n, d);
            let dyn2 = self.gpu.storage((n * self.cfg.dyn_width()) as u64);
            self.ops.matmul(&mut s, &lb.ffn_conv_proj, &a2, &dyn2, 0);
            let hm = self.gpu.storage((n * d) as u64);
            s.push(self.dyn_conv(&xn2, &dyn2, &lb.ffn_conv_base, &hm, n, 0));

            let am = self.act(&mut s, &hm, 0, n, d);
            let gate = self.gpu.storage((n * ff) as u64);
            self.ops.matmul(&mut s, &lb.gate, &am, &gate, 0);
            let upb = self.gpu.storage((n * ff) as u64);
            self.ops.matmul(&mut s, &lb.up, &am, &upb, 0);
            let act = self.gpu.storage((n * ff) as u64);
            s.push(self.gpu.step(self.k.silu_mul, &[&gate, &upb, &act], &[n * ff], n * ff));
            let ad = self.act(&mut s, &act, 0, n, ff);
            let dn = self.gpu.storage((n * d) as u64);
            self.ops.matmul(&mut s, &lb.down, &ad, &dn, 0);
            let postm = self.gpu.storage((n * d) as u64);
            s.push(self.dyn_conv(&dn, &dyn2, &lb.ffn_conv_base, &postm, n, 1));
            let xnext = self.gpu.storage((n * d) as u64);
            s.push(self.gpu.step(self.k.add2, &[&xmid, &postm, &xnext], &[n * d], n * d));

            self.gpu.submit(&[], &s);
            self.gpu.poll_wait();
            self.gpu.flush();
            x = xnext;
        }
        let out = self.gpu.read(&x, (n * d) as usize);
        self.gpu.scratch_release();
        Ok(out)
    }

    /// **The candidate selector**, at temperature 0.
    ///
    /// Per position: the head's top-`selector_top_k` candidates, then a
    /// rank-`selector_rank` three-way Hadamard score against the token
    /// actually chosen at the PREVIOUS position,
    ///
    /// ```text
    /// score[c] = logit[c] + < predecessor[prev] * (hidden @ W_sel), successor[c] >
    /// ```
    ///
    /// and the argmax of that. The walk is GREEDY left to right - the chosen
    /// token becomes the next position's predecessor and an earlier position
    /// is never revisited. That is the reference's own search, not a
    /// simplification of it: a Viterbi pass over the same lattice would be a
    /// DIFFERENT (and unvalidated) drafter, so the cheap one is also the
    /// correct one here.
    ///
    /// The two codebooks are `[vocab, rank]` each - 127 MB of Q8_0 - and this
    /// reads 17 rows of them per position, straight from the mapping.
    ///
    /// `hidden` is the PRE-final-norm block, the same thing
    /// [`Self::denoise_block`] returns and the same thing the head is fed;
    /// the final norm is applied here, on the host, because
    /// `hidden_projection` consumes the POST-norm hidden and nothing else.
    /// That is not a detail: the pre-norm hidden has an RMS around 1000 on
    /// this checkpoint, so projecting it instead scales the pairwise term by
    /// ~1000 against logits of order ten, and the selector stops being a
    /// tie-breaker and becomes the whole decision - it then overrides the
    /// head at EVERY position, including the ones the head had right. That is
    /// what this looked like before it was fixed, and it is a failure mode a
    /// "does it produce fluent text" check cannot see, because every proposal
    /// is verified by the target anyway; it shows up only as an acceptance
    /// rate that is mysteriously near zero.
    pub fn select(&self, hidden: &[f32], logits: &[f32], anchor: u32) -> Result<Vec<u32>, String> {
        let d = self.cfg.d_model as usize;
        let rank = self.cfg.selector_rank as usize;
        let top_k = self.cfg.selector_top_k as usize;
        let vocab = self.cfg.vocab as usize;
        let n = hidden.len() / d;
        assert_eq!(logits.len(), n * vocab, "{MODEL}: select got {} logits for {n} rows of {vocab}", logits.len());

        let proj: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let row = &hidden[i * d..(i + 1) * d];
                let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / d as f32 + self.cfg.eps).sqrt();
                let normed: Vec<f32> = row.iter().zip(&self.out_norm).map(|(x, w)| x * inv * w).collect();
                model::hostmath::matvec_par(&self.sel_hidden, &normed, rank, d)
            })
            .collect();
        let code = |name: &str, row: u32| -> Result<Vec<f32>, String> {
            self.mg
                .tensor_range(name, row as usize * rank, rank)
                .ok_or_else(|| format!("{MODEL}: {name} has no row {row}"))?
        };

        let mut path = Vec::with_capacity(n);
        let mut prev = anchor;
        for i in 0..n {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let cand = top_k_ids(row, top_k);
            let pred = code("selector_predecessor.weight", prev)?;
            let left: Vec<f32> = (0..rank).map(|r| pred[r] * proj[i][r]).collect();
            let mut best = (f32::NEG_INFINITY, cand[0]);
            for &c in &cand {
                let succ = code("selector_successor.weight", c)?;
                let pair: f32 = (0..rank).map(|r| left[r] * succ[r]).sum();
                let score = row[c as usize] + pair;
                if score > best.0 {
                    best = (score, c);
                }
            }
            prev = best.1;
            path.push(prev);
        }
        Ok(path)
    }

    /// This model's final RMSNorm weights - handed to the TARGET's head
    /// epilogue in place of the target's own, which is exactly the composition
    /// `DFlashDraftModel.forward` + `compute_logits` performs (the draft's
    /// `norm`, then the target's `lm_head`).
    pub fn output_norm(&self) -> &[f32] {
        &self.out_norm
    }
}

/// The `k` largest entries' indices, unordered.
///
/// One pass with a `k`-element insertion buffer rather than
/// `select_nth_unstable` over a materialized `0..vocab`, because the caller
/// runs this once per block POSITION on a 248320-wide row and the allocation
/// alone was the larger half of the selector's cost. Unordered is enough: the
/// caller takes an argmax over the whole set anyway.
fn top_k_ids(row: &[f32], k: usize) -> Vec<u32> {
    assert!(k > 0 && row.len() >= k, "top_k_ids: asked for {k} of {} values", row.len());
    let mut best: Vec<(f32, u32)> = row.iter().take(k).copied().zip(0u32..).collect();
    best.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut floor = best[0].0;
    for (i, &v) in row.iter().enumerate().skip(k) {
        if v <= floor {
            continue;
        }
        // The buffer is sorted ascending and `k` is 16, so this is a shift of
        // at most 15 pairs - cheaper than a heap at this size, and it keeps
        // `floor` correct without a second scan.
        let at = best.partition_point(|e| e.0 < v);
        best.copy_within(1..at, 0);
        best[at - 1] = (v, i as u32);
        floor = best[0].0;
    }
    best.into_iter().map(|(_, i)| i).collect()
}

// --------------------------------------------------------------- the drafter

/// **The drafter**, in the shape `generate_speculative` takes: given the
/// context so far, return up to `want` proposals.
///
/// Holds the target instance because half the work is the target's - the
/// tapped hidden states it cross-attends to, the embedding of `[anchor,
/// MASK...]`, and the head that turns draft hidden states into logits.
///
/// The state it carries is `fed`: how many context positions are already in
/// the draft's K/V cache. Everything else about a speculative round - which
/// proposals were accepted, whether the recurrent state was rolled back -
/// shows up here as nothing more than "the context is now this long", because
/// the draft cache is indexed by absolute position and a re-committed prefix
/// overwrites itself.
///
/// The one thing `fed` alone cannot survive is a NEW generation, which is why
/// `seen` exists beside it: a second request with a longer prompt looks
/// exactly like the first one continuing, and carrying `fed` across would
/// leave the previous request's context in the draft's cache under the new
/// request's positions - a drafter conditioned on someone else's prompt,
/// producing fluent proposals that are simply never accepted. The target's
/// [`Qwen35GgufInstance::hidden_tap_generation`] is what distinguishes them,
/// and it is checked every round rather than left to a caller to remember.
pub struct Dflash2Drafter<'a> {
    inst: &'a Qwen35GgufInstance,
    model: Dflash2,
    fed: RefCell<u32>,
    /// The target's tap generation this drafter's cache belongs to.
    seen: RefCell<u64>,
    stats: RefCell<DraftStats>,
}

/// What the drafter itself cost, as opposed to what it bought - the two are
/// separately actionable and a single tok/s figure hides both.
#[derive(Clone, Copy, Debug, Default)]
pub struct DraftStats {
    pub calls: u64,
    /// Seconds in the draft forward (context projection + denoise + head).
    pub draft_s: f64,
    /// Seconds in the selector walk (host).
    pub select_s: f64,
    /// Proposals returned.
    pub proposed: u64,
}

impl<'a> Dflash2Drafter<'a> {
    /// Attach a loaded draft model to a target instance, turning on the hidden
    /// taps the draft needs.
    pub fn new(inst: &'a Qwen35GgufInstance, model: Dflash2) -> Dflash2Drafter<'a> {
        inst.enable_hidden_taps(&model.cfg.target_layers);
        let seen = RefCell::new(inst.hidden_tap_generation());
        Dflash2Drafter { inst, model, fed: RefCell::new(0), seen, stats: RefCell::new(DraftStats::default()) }
    }

    /// The largest useful `k`: one block is `block_size` rows, of which the
    /// first is the anchor.
    pub fn max_draft(&self) -> u32 {
        self.model.cfg.block_size - 1
    }

    pub fn stats(&self) -> DraftStats {
        *self.stats.borrow()
    }

    /// Forget the context fed so far. Rarely needed by hand: [`Self::propose`]
    /// notices a new generation on its own (see this struct's own doc), and
    /// this is here for a caller that wants to force it.
    pub fn reset(&self) {
        *self.fed.borrow_mut() = 0;
        *self.seen.borrow_mut() = self.inst.hidden_tap_generation();
    }

    /// **One draft**, the whole contract `generate_speculative` asks for.
    ///
    /// The anchor is `ctx`'s LAST token; the context rows are everything
    /// before it. That split is the reference's: a block's first row carries
    /// the already-verified token as an EMBEDDING, and the target's hidden
    /// state for it does not exist yet (it is produced by the very verify pass
    /// this draft is feeding).
    ///
    /// Returns an empty proposal rather than an error when anything is not
    /// ready - a drafter that fails is a drafter that proposes nothing, and
    /// the verify loop is already correct for that case.
    pub fn propose(&self, ctx: &[u32], want: u32) -> Vec<u32> {
        match self.try_propose(ctx, want) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{MODEL}: drafting skipped: {e}");
                Vec::new()
            }
        }
    }

    fn try_propose(&self, ctx: &[u32], want: u32) -> Result<Vec<u32>, String> {
        if want == 0 || ctx.len() < 2 {
            return Ok(Vec::new());
        }
        let want = want.min(self.max_draft());
        let block = want + 1;
        let anchor = *ctx.last().expect("non-empty (checked)");
        let ctx_len = ctx.len() as u32 - 1;

        let t0 = std::time::Instant::now();
        // A new sequence invalidates every cached context row, whatever the
        // context length says - see this struct's own doc.
        let gen = self.inst.hidden_tap_generation();
        if gen != *self.seen.borrow() {
            *self.seen.borrow_mut() = gen;
            *self.fed.borrow_mut() = 0;
        }
        // Catch the draft's K/V cache up with whatever the target committed
        // since the last round.
        let fed = *self.fed.borrow();
        if ctx_len > fed {
            let rows = self.inst.target_hidden(fed, ctx_len - fed)?;
            self.model.append_context(&rows, fed)?;
            *self.fed.borrow_mut() = ctx_len;
        }

        let mut ids = Vec::with_capacity(block as usize);
        ids.push(anchor);
        ids.extend(std::iter::repeat_n(self.model.cfg.mask_token_id, want as usize));
        let noise = self.inst.embed_rows_of(&ids)?;
        let hidden = self.model.denoise_block(&noise, ctx_len)?;

        // Row 0 is the anchor's own denoised state and predicts nothing; the
        // MASK rows are the proposals.
        let d = self.model.cfg.d_model as usize;
        let mask_rows = &hidden[d..];
        let logits = self.inst.logits_with_norm(mask_rows, want, self.model.output_norm())?;
        let draft_s = t0.elapsed().as_secs_f64();

        let t1 = std::time::Instant::now();
        let path = self.model.select(mask_rows, &logits, anchor)?;
        let mut st = self.stats.borrow_mut();
        st.calls += 1;
        st.draft_s += draft_s;
        st.select_s += t1.elapsed().as_secs_f64();
        st.proposed += path.len() as u64;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::top_k_ids;

    /// [`top_k_ids`] is hand-rolled index arithmetic over a 248320-wide row,
    /// and it decides which candidates the selector ever sees - a wrong one
    /// shows up only as an acceptance rate slightly lower than it should be.
    /// So it is checked against the obvious (and far slower) sort, on a
    /// pseudo-random row, on ties, and on the boundary where every element
    /// qualifies.
    #[test]
    fn top_k_ids_agrees_with_a_full_sort() {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 1024.0 - 8.0
        };
        for len in [16usize, 17, 64, 1000] {
            for k in [1usize, 3, 16] {
                if len < k {
                    continue;
                }
                let row: Vec<f32> = (0..len).map(|_| next()).collect();
                let mut want: Vec<u32> = (0..len as u32).collect();
                want.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
                want.truncate(k);
                want.sort();
                let mut got = top_k_ids(&row, k);
                got.sort();
                assert_eq!(got, want, "top {k} of {len}");
            }
        }
        // All equal: any k of them is a correct answer, but there must be
        // exactly k of them and they must be distinct.
        let flat = vec![1.0f32; 50];
        let mut got = top_k_ids(&flat, 16);
        got.sort();
        got.dedup();
        assert_eq!(got.len(), 16, "ties must still yield k distinct indices");
    }
}
