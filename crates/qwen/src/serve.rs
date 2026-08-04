// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Concurrent serving engine for the Qwen3 decoder: a **paged** KV cache shared by
//! many sequences + **batched** decode that advances every active sequence by one
//! token per iteration. Each sequence's KV grows a block at a time from a shared
//! pool (no per-sequence worst-case reservation), and one batched forward serves
//! the whole running set — so more sequences stay resident and decode together.
//!
//! Self-contained: it owns its `Gpu` (with the batched paged kernels), a
//! `ParamStore` of the decoder weights, per-layer block pools, and the block
//! allocator. The forward math is the shared [`model::block`] Qwen3 block; only
//! the attention is paged + ragged-batched.

use std::collections::HashMap;

use gpu_core::select::{
    AutoTuner, CachedSelector, DefaultSelector, Dtype, KernelSelector, KernelVariant, Op, OpShape,
    DECODE_REGIME_MAX_ROWS,
};
use gpu_core::{DeviceBuffer, DeviceCaps, Gpu, Step};
use model::block::{self, KernelIds};
use model::paged::{BlockAllocator, BlockTable, PrefixCache};
use paramstore::ParamStore;

use crate::config::QwenConfig;

const EMBED: usize = 0;
const MATMUL: usize = 1;
const RMSNORM: usize = 2;
const RMS_INV: usize = 3;
const SILU_MUL: usize = 4;
const ADD2: usize = 5;
const ROPE_PAGED: usize = 6;
const KV_APPEND_B: usize = 7;
const SCORES_B: usize = 8;
const SOFTMAX_B: usize = 9;
const APPLY_B: usize = 10;
// Causal attention over a whole prompt (batched prefill).
const GQA_SCORES: usize = 11;
const ATTN_SOFTMAX: usize = 12;
const GQA_APPLY: usize = 13;
// int8 paged KV (dequant on read).
const APPEND_I8: usize = 14;
const SCORES_I8: usize = 15;
const APPLY_I8: usize = 16;
// Device-side greedy head: matmul -> row argmax, so decode never ships a
// [batch, vocab] logit block to the host.
const ARGMAX_ROW: usize = 17;
const ARGMAX_PART: usize = 18;
const ARGMAX_FINAL: usize = 19;
// Decode-regime kernels: selected per dispatch by row count.
const RMSNORM_ROWS: usize = 20;
const MATMUL_GEMV: usize = 21;
// Int8 weight path (A0): per-token activation quant + DP4A GEMMs with
// per-token x per-channel dequant scales — the tile GEMM for prefill shapes,
// the packed GEMV for decode row counts.
const MAX_ABS_ROW: usize = 22;
const QUANT_PACK: usize = 23;
const MATMUL_I8_DYN: usize = 24;
const MATMUL_I8_GEMV: usize = 25;
// On-device decode window (A4): feed the argmax back as the next input and
// advance the paged metadata without a host round-trip.
const DECODE_FEED: usize = 26;
const DECODE_ADVANCE: usize = 27;

const PIPELINES: &[(&str, &str)] = &[
    ("embed", kernels::EMBED),
    ("matmul", kernels::MATMUL),
    ("rmsnorm", kernels::RMSNORM),
    ("rms_inv", kernels::RMS_INV),
    ("silu_mul", kernels::SILU_MUL),
    ("add2", kernels::ADD2),
    ("rope_paged", kernels::ROPE_PAGED),
    ("paged_kv_append_batched", kernels::PAGED_KV_APPEND_BATCHED),
    ("paged_decode_scores_batched", kernels::PAGED_DECODE_SCORES_BATCHED),
    ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
    ("paged_decode_apply_batched", kernels::PAGED_DECODE_APPLY_BATCHED),
    ("gqa_scores", kernels::GQA_SCORES),
    ("attn_softmax", kernels::ATTN_SOFTMAX),
    ("gqa_apply", kernels::GQA_APPLY),
    ("paged_kv_append_i8_batched", kernels::PAGED_KV_APPEND_I8_BATCHED),
    ("paged_decode_scores_i8_batched", kernels::PAGED_DECODE_SCORES_I8_BATCHED),
    ("paged_decode_apply_i8_batched", kernels::PAGED_DECODE_APPLY_I8_BATCHED),
    ("argmax_row", kernels::ARGMAX_ROW),
    ("argmax_part", kernels::ARGMAX_PART),
    ("argmax_final", kernels::ARGMAX_FINAL),
    ("rmsnorm_rows", kernels::RMSNORM_ROWS),
    ("matmul_gemv", kernels::MATMUL_GEMV),
    ("max_abs_row", kernels::MAX_ABS_ROW),
    ("quant_pack", kernels::QUANT_PACK),
    ("matmul_i8_dyn", kernels::MATMUL_I8_DYN),
    ("matmul_i8_gemv", kernels::MATMUL_I8_GEMV),
    ("decode_feed", kernels::DECODE_FEED),
    ("decode_advance", kernels::DECODE_ADVANCE),
];

/// Longest on-device decode window (tokens per host round-trip). The scheduler
/// picks `min(this, tokens remaining)`; the window trades one readback per
/// token for one per window, at the cost of up to `window - 1` wasted decode
/// steps when a sequence hits EOS mid-window.
pub const DECODE_WINDOW: usize = 4;

fn ids() -> KernelIds {
    KernelIds {
        rmsnorm: RMSNORM,
        rms_inv: RMS_INV,
        silu_mul: SILU_MUL,
        // unused on the forward decode path:
        rmsnorm_dx: RMSNORM,
        rmsnorm_dw: RMSNORM,
        rope: ROPE_PAGED,
        rope_bwd: ROPE_PAGED,
        gqa_scores: GQA_SCORES,
        gqa_apply: GQA_APPLY,
        attn_softmax: ATTN_SOFTMAX,
        gqa_dscores: 0,
        gqa_dv: 0,
        gqa_dq: 0,
        gqa_dk: 0,
        silu_da: SILU_MUL,
        silu_db: SILU_MUL,
    }
}

// The decode-regime boundaries (max rows, argmax split vocab) live in the
// shared selection policy — `gpu_core::select` — not here: which kernel runs
// for a shape on a device is the selector's single job.

/// Chunks per row for the two-stage argmax; 256 threads per row saturates the
/// reduction without a large partial buffer (256*2 f32 per row).
const ARGMAX_CHUNKS: u32 = 256;

fn fb(x: f32) -> u32 {
    x.to_bits()
}

/// Fault injection (G): `brain perf faults` arms a fault, the next pass
/// through its check point fires it. Feature-gated — a build without
/// `fault-injection` compiles the sink to nothing, so there is no release
/// cost and nothing to accidentally arm in production.
#[cfg(feature = "fault-injection")]
pub mod fault {
    use std::sync::atomic::{AtomicBool, Ordering};
    static KERNEL_FAILURE: AtomicBool = AtomicBool::new(false);

    /// Arm a one-shot kernel-dispatch failure: the next batched decode
    /// panics at its dispatch point, as a real device fault would surface.
    pub fn arm_kernel_failure() {
        KERNEL_FAILURE.store(true, Ordering::SeqCst);
    }
    pub(crate) fn take_kernel_failure() -> bool {
        KERNEL_FAILURE.swap(false, Ordering::SeqCst)
    }
}

/// A batched-forward input: token ids (embedded via `tok.weight`) or ready-made
/// per-row embeddings written straight into the residual stream (the tts Talker
/// feeds codec/text-conditioned embeddings rather than ids).
pub enum Input<'a> {
    Tokens(&'a [u32]),
    Embeds(&'a [f32]),
    /// Token ids already resident in the engine's `tok_buf` — the on-device
    /// decode window (A4): `decode_feed` wrote them from the previous step's
    /// argmax, and `decode_advance` already advanced the paged metadata, so
    /// the forward performs NO host writes at all.
    Resident,
}

/// One decoder-param leaf name → element count (mirrors the decode weight set).
fn decoder_param_list(cfg: &QwenConfig) -> Vec<(String, usize)> {
    let (d, ff) = (cfg.d_model as usize, cfg.d_ff as usize);
    let (hq, hkv, hd) = (cfg.q_dim() as usize, cfg.kv_dim() as usize, cfg.head_dim as usize);
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
    out.push(("tok.weight".to_string(), cfg.vocab as usize * d)); // embedding gather
    out
}

/// A running sequence: its block table, generated tokens, and completion flag.
struct Seq {
    table: BlockTable,
    generated: Vec<u32>,
    done: bool,
}

/// Batched scratch (sized for `max_batch` rows), reused every iteration.
struct Scratch {
    res: Vec<DeviceBuffer>, // n_layers+1, each [B*d]
    xn1: DeviceBuffer,
    q_pre: DeviceBuffer,
    q: DeviceBuffer,
    k_pre: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    h: DeviceBuffer,
    proj: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn_final: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    // per-step metadata (uploaded each iteration)
    tok_buf: DeviceBuffer,
    pos_buf: DeviceBuffer,
    seqlen_buf: DeviceBuffer,
    blk_buf: DeviceBuffer,
    off_buf: DeviceBuffer,
    bt_buf: DeviceBuffer,
    /// `[DECODE_WINDOW-1, max_batch, 3]` device (block, offset, bt_index)
    /// schedule for the on-device decode window (A4).
    sched_buf: DeviceBuffer,
    /// `[DECODE_WINDOW, max_batch]` window token history (greedy indices).
    hist_buf: DeviceBuffer,
}

/// Paged, batched Qwen3 serving engine.
pub struct Engine {
    cfg: QwenConfig,
    gpu: Gpu,
    /// The device's capabilities, read once at build — the selector's input.
    caps: DeviceCaps,
    /// Which kernel variant runs for each (op, shape) — the shared decode-regime
    /// policy, memoised per distinct shape.
    selector: CachedSelector<DefaultSelector>,
    ps: ParamStore,
    block_size: u32,
    max_batch: u32,
    max_blocks_per_seq: u32,
    max_prefill: u32,
    cap: u32,
    alloc: BlockAllocator,
    /// Prompt-prefix cache (D): full prompt blocks are indexed after prefill
    /// and adopted (shared, refcounted) by later prompts with the same prefix,
    /// so prefill computes only the unmatched tail. Purely a prefill
    /// optimisation — decode never touches it.
    prefix: PrefixCache,
    /// Prefix-reuse counters: tokens looked up / tokens served from the cache.
    prefix_lookup_tokens: u64,
    prefix_hit_tokens: u64,
    pool_k: Vec<DeviceBuffer>,
    pool_v: Vec<DeviceBuffer>,
    // int8 KV: pools hold packed int8 (4/u32, ~4x smaller) + per-(token,kv-head)
    // dequant scales. Empty when kv_int8 is false (fp32 pools).
    kv_int8: bool,
    scales_k: Vec<DeviceBuffer>,
    scales_v: Vec<DeviceBuffer>,
    /// Int8 WEIGHT path (A0): the 7 per-layer linears quantized per-channel and
    /// packed 4/u32 (~4x fewer weight bytes in the bandwidth-bound decode
    /// regime), with per-token dynamic activation quant. `None` = fp32 weights
    /// — always the case on a device whose caps report no packed-int8 path.
    w8: Option<crate::q8::Q8>,
    /// The int8-packed LM head (present iff `w8` is).
    head8: Option<crate::q8::Lin8>,
    /// Measured GEMV/tile choices for the int8 linears (S5), keyed by
    /// `(row bucket, n, k)` — tuned once at build on THIS device (persisted
    /// per adapter), so the hot path never measures. Empty on fp32 engines.
    tuned_i8: HashMap<(u32, u32, u32), KernelVariant>,
    sc: Scratch,
    /// `[vocab, d]` tied/untied head, kept on the host for the prefill path
    /// (applied once per request) and for callers that need full logits.
    head: Vec<f32>,
    /// The same head resident on the device (fp32), for the batched decode
    /// path. `None` when the head lives on the device as int8 (`head8`) —
    /// keeping both resident would forfeit the memory the quantisation buys.
    head_dev: Option<DeviceBuffer>,
    /// `[max_batch, vocab]` decode logits, and `[max_batch]` argmax indices.
    logits_dev: DeviceBuffer,
    argmax_dev: DeviceBuffer,
    /// `[max_batch, ARGMAX_CHUNKS, 2]` partial (value, index) pairs for the
    /// two-stage argmax reduction.
    argmax_part_dev: DeviceBuffer,
}

impl Engine {
    /// Build from an in-memory decoder weight map (tests / embedded weights).
    /// `num_blocks` physical blocks of `block_size` tokens, up to `max_batch`
    /// concurrent sequences of at most `max_blocks_per_seq * block_size` tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn from_map(cfg: QwenConfig, weights: &HashMap<String, Vec<f32>>, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool, weights_int8: bool) -> Engine {
        Self::from_map_with_gpu(Gpu::new(PIPELINES), cfg, weights, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, weights_int8)
    }

    /// [`Engine::from_map`] on an EXISTING device (F1 warm start): the caller's
    /// `Gpu` parents this engine via [`Gpu::new_like`], so building another
    /// engine costs pipeline compilation only — never a second full device
    /// init. This is what a serving process or the residency executor should
    /// use: one device per process, many engines on it (many concurrent
    /// devices on one card is both slow and hostile to the driver).
    #[allow(clippy::too_many_arguments)]
    pub fn from_map_on(parent: &Gpu, cfg: QwenConfig, weights: &HashMap<String, Vec<f32>>, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool, weights_int8: bool) -> Engine {
        Self::from_map_with_gpu(parent.new_like(PIPELINES), cfg, weights, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, weights_int8)
    }

    #[allow(clippy::too_many_arguments)]
    fn from_map_with_gpu(gpu: Gpu, cfg: QwenConfig, weights: &HashMap<String, Vec<f32>>, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool, weights_int8: bool) -> Engine {
        // Int8 weights are capability-driven, never assumed: the request only
        // takes effect where the packed-dot GEMM executes (the selector's
        // PackedInt8 gate). Elsewhere — the CPU JIT — fp32 weights stay, and
        // the fallback is said out loud rather than silently absorbed.
        let caps = gpu.caps();
        // The gate is the CAPABILITY, not the selector's head: which int8
        // variant is best at some shape is a tuning question, but whether the
        // packed-dot kernels execute at all is numeric.int8_dot.
        let w8_on = weights_int8 && caps.numeric.int8_dot;
        if weights_int8 && !w8_on {
            eprintln!("serve: int8 weights requested but this device has no packed-int8 path; using fp32 weights");
        }
        // The 7 per-layer linears live in the int8 bank when it is on — loading
        // them into the fp32 ParamStore as well would keep both copies resident
        // and forfeit the memory the quantisation buys.
        let roles = decoder_param_list(&cfg)
            .into_iter()
            .filter(|(n, _)| !(w8_on && crate::q8::Q8::is_i8_linear(n)))
            .map(|(n, c)| (n, c, paramstore::Role::Frozen))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, weights);
        let head = weights.get(cfg.head_weight()).cloned().unwrap_or_else(|| weights.get("tok.weight").cloned().expect("head weight"));

        let (d, ff) = (cfg.d_model as u64, cfg.d_ff as u64);
        let (hq, hkv) = (cfg.q_dim() as u64, cfg.kv_dim() as u64);
        // Scratch rows serve both decode (max_batch sequences) and prefill (a whole
        // prompt of up to max_prefill tokens processed in one forward).
        let b = max_batch.max(max_prefill) as u64;
        let cap = max_blocks_per_seq * block_size;
        let nh = cfg.n_heads as u64;
        // scores/probs hold decode [rows,nh,cap] OR prefill causal [nh,N,N].
        let bcap = (b * nh * cap as u64).max(max_prefill as u64 * max_prefill as u64 * nh);
        let st = |x: u64| gpu.storage(x);

        let mut res = Vec::new();
        for _ in 0..=cfg.n_layers {
            res.push(st(b * d));
        }
        let n_kv = cfg.n_kv_heads as u64;
        let slots = num_blocks as u64 * block_size as u64;
        // int8 pools pack 4 values/u32 (÷4 words) + a scale per (token slot, kv-head).
        let pool_words = if kv_int8 { slots * hkv / 4 } else { slots * hkv };
        let mut pool_k = Vec::new();
        let mut pool_v = Vec::new();
        let mut scales_k = Vec::new();
        let mut scales_v = Vec::new();
        for _ in 0..cfg.n_layers {
            pool_k.push(st(pool_words));
            pool_v.push(st(pool_words));
            if kv_int8 {
                scales_k.push(st(slots * n_kv));
                scales_v.push(st(slots * n_kv));
            }
        }
        let sc = Scratch {
            res,
            xn1: st(b * d),
            q_pre: st(b * hq),
            q: st(b * hq),
            k_pre: st(b * hkv),
            k: st(b * hkv),
            v: st(b * hkv),
            ctx: st(b * hq),
            xmid: st(b * d),
            xn2: st(b * d),
            gate_pre: st(b * ff),
            up: st(b * ff),
            h: st(b * ff),
            proj: st(b * d),
            mlp_out: st(b * d),
            xn_final: st(b * d),
            scores: st(bcap),
            probs: st(bcap),
            tok_buf: st(b),
            pos_buf: st(b),
            seqlen_buf: st(b),
            blk_buf: st(b),
            off_buf: st(b),
            bt_buf: st(b * max_blocks_per_seq as u64),
            sched_buf: st((DECODE_WINDOW as u64 - 1) * max_batch as u64 * 3),
            hist_buf: st(DECODE_WINDOW as u64 * max_batch as u64),
        };
        // Int8 weight bank: the 7 per-layer linears + the head, quantized once
        // at build (per-channel scales) and packed 4/u32. Activation-quant
        // scratch is sized for the widest input rows the forward ever sees —
        // prefill chunks run through the same path as decode.
        let (w8, head8) = if w8_on {
            let (dm, ffm) = (cfg.d_model as usize, cfg.d_ff as usize);
            let (hqm, hkvm) = (cfg.q_dim() as usize, cfg.kv_dim() as usize);
            let dims = move |leaf: &str| -> (usize, usize) {
                match leaf {
                    "attn.wq.weight" => (hqm, dm),
                    "attn.wk.weight" | "attn.wv.weight" => (hkvm, dm),
                    "attn.wo.weight" => (dm, hqm),
                    "mlp.gate.weight" | "mlp.up.weight" => (ffm, dm),
                    "mlp.down.weight" => (dm, ffm),
                    other => panic!("q8 dims: unknown linear {other}"),
                }
            };
            let q8 = crate::q8::Q8::build(
                &gpu,
                weights,
                0..cfg.n_layers as usize,
                dims,
                b as u32,
                ff.max(hq).max(d) as u32,
                MAX_ABS_ROW,
                QUANT_PACK,
                MATMUL_I8_DYN,
            );
            let (packed, sw) = crate::q8::quantize_weight(&head, cfg.vocab as usize, cfg.d_model as usize);
            let pb = gpu.storage(packed.len() as u64);
            gpu.write(&pb, &packed);
            gpu.poll_wait();
            let sb = gpu.storage(sw.len() as u64);
            gpu.write(&sb, &sw.iter().map(|v| v.to_bits()).collect::<Vec<u32>>());
            gpu.poll_wait();
            let h8 = crate::q8::Lin8 { packed: pb, scale: sb, k: cfg.d_model, n: cfg.vocab };
            (Some(q8), Some(h8))
        } else {
            (None, None)
        };
        // S5: measure the GEMV/tile crossover for THIS device's int8 shapes at
        // build time (a few ms; persisted per adapter + kernel sources), so
        // the hot path only ever looks the choice up. Row counts vary freely
        // at runtime, so choices are keyed by power-of-two bucket.
        let tuned_i8 = match (&w8, &head8) {
            (Some(q8), Some(h8)) => Self::tune_i8(&gpu, &caps, q8, h8, b as u32),
            _ => HashMap::new(),
        };
        // Decode-side head. Sized by max_batch (NOT the prefill row count): only
        // decode rows need logits, and [max_prefill, vocab] would be gigabytes.
        // fp32 head only when the int8 head is absent — never both resident.
        let vocab = cfg.vocab as u64;
        let head_dev = if w8_on { None } else { Some(gpu.storage_init("lm_head", &head)) };
        let logits_dev = st(max_batch as u64 * vocab);
        let argmax_dev = st(max_batch as u64);
        let argmax_part_dev = st(max_batch as u64 * ARGMAX_CHUNKS as u64 * 2);
        Engine {
            cfg,
            caps,
            gpu,
            selector: CachedSelector::new(DefaultSelector),
            ps,
            block_size,
            max_batch,
            max_blocks_per_seq,
            max_prefill,
            cap,
            alloc: BlockAllocator::new(num_blocks, block_size),
            prefix: PrefixCache::new(),
            prefix_lookup_tokens: 0,
            prefix_hit_tokens: 0,
            pool_k,
            pool_v,
            kv_int8,
            scales_k,
            scales_v,
            w8,
            head8,
            tuned_i8,
            sc,
            head,
            head_dev,
            logits_dev,
            argmax_dev,
            argmax_part_dev,
        }
    }

    /// Load a serving engine from a brain Qwen checkpoint (fp32 tensors;
    /// `weights_int8` quantizes the linears + head at load where the device
    /// has a packed-int8 path).
    #[allow(clippy::too_many_arguments)]
    pub fn load(path: &str, block_size: u32, num_blocks: u32, max_batch: u32, max_blocks_per_seq: u32, max_prefill: u32, kv_int8: bool, weights_int8: bool) -> Engine {
        let c = checkpoint::load(path);
        let cfg = QwenConfig::from_json(&c.header["config"]);
        let mut map = HashMap::new();
        for (name, _) in decoder_param_list(&cfg) {
            let t = c.find(&name, "").cloned().unwrap_or_else(|| panic!("serve: checkpoint missing tensor {name}"));
            map.insert(name, t);
        }
        let hw = cfg.head_weight();
        if !map.contains_key(hw) {
            let h = c.find(hw, "").cloned().unwrap_or_else(|| panic!("serve: checkpoint missing head {hw}"));
            map.insert(hw.to_string(), h);
        }
        Engine::from_map(cfg, &map, block_size, num_blocks, max_batch, max_blocks_per_seq, max_prefill, kv_int8, weights_int8)
    }

    /// True when the decode path runs on int8 weights (the request survived the
    /// device capability gate). What a caller should report, rather than what
    /// was asked for.
    pub fn weights_int8(&self) -> bool {
        self.w8.is_some()
    }

    /// The device this engine runs on — the parent handle for building more
    /// engines on the same device ([`Engine::from_map_on`]).
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The model's vocabulary size (for a caller doing its own sampling).
    pub fn vocab(&self) -> usize {
        self.cfg.vocab as usize
    }

    /// Append one slot per sequence and gather the batched-forward metadata.
    fn append_meta(&mut self, tables: &mut [&mut BlockTable]) -> (u32, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>) {
        let mbt = self.max_blocks_per_seq as usize;
        let bsz = tables.len() as u32;
        assert!(bsz <= self.max_batch);
        let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut bt = vec![0u32; bsz as usize * mbt];
        for (i, table) in tables.iter_mut().enumerate() {
            let pos = table.len();
            let (block, offset) = table.append(&mut self.alloc).expect("KV pool exhausted");
            positions.push(pos);
            seqlens.push(pos + 1);
            blocks.push(block);
            offsets.push(offset);
            for (lb, &phys) in table.blocks().iter().enumerate() {
                bt[i * mbt + lb] = phys;
            }
        }
        (bsz, positions, seqlens, blocks, offsets, bt)
    }

    /// Advance every sequence by one token (decode).
    pub(crate) fn forward_batched(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<f32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        self.run_batched(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt)
    }

    /// Advance every sequence by one token and return the **greedy next token**
    /// per row, with the LM head applied on the device (see
    /// [`Engine::run_batched_greedy`]).
    pub(crate) fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        self.run_batched_greedy(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt)
    }

    /// Advance every sequence by one token from a ready-made embedding per sequence
    /// (`[bsz, d_model]`) — the tts Talker multi-stream path: concurrent voice
    /// streams decode together on the shared paged pool.
    pub fn forward_batched_embed(&mut self, tables: &mut [&mut BlockTable], embeds: &[f32]) -> Vec<f32> {
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        assert_eq!(embeds.len(), bsz as usize * self.cfg.d_model as usize);
        self.run_batched(bsz, Input::Embeds(embeds), &positions, &seqlens, &blocks, &offsets, &bt)
    }

    /// Run one batched forward over `bsz` rows given fully-computed metadata:
    /// `positions[i]` RoPE position, `seqlens[i]` the cached length row i attends
    /// (row i's query attends `j < seqlens[i]` — set to start+i+1 for causal
    /// prefill), `(blocks[i], offsets[i])` the pool slot to write row i's K/V, and
    /// `bt` the per-row block tables (`bsz * max_blocks_per_seq`). Serves decode
    /// (one new token per sequence) and prefill chunks alike.
    #[allow(clippy::too_many_arguments)]
    fn run_batched(&self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) -> Vec<f32> {
        let b = self.run_batched_submit(bsz, input, positions, seqlens, blocks, offsets, bt);
        self.gpu.read(&self.sc.xn_final, (b * self.cfg.d_model) as usize)
    }

    /// The transformer body only: records and submits every stage, leaving the
    /// final norm in `sc.xn_final` **without reading it back**. Submits are
    /// accumulated lazily and flushed on the next read, so a caller that appends
    /// more device work (the greedy head) still pays one flush per step rather
    /// than two. Returns the row count.
    /// `out = x @ W^T`, choosing the decode-regime GEMV (one workgroup per
    /// output column, W streamed once across all rows) when the selector says
    /// the shape is in that regime. Same contract, same result.
    fn mm(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, m: u32, k: u32, n: u32) -> Step {
        let g = &self.gpu;
        let shape = OpShape { m, n, k, dtype: Dtype::F32 };
        match self.selector.select(Op::MatMul, shape, &self.caps) {
            KernelVariant::WorkgroupPerOutput => g.step(MATMUL_GEMV, &[x, w, out], &[m, k, n], n * 64),
            _ => g.step(MATMUL, &[x, w, out], &[m, k, n], m * n),
        }
    }

    /// One int8 linear: the MEASURED choice for this device where one exists
    /// (S5, tuned at build, keyed by row bucket), else the static policy. The
    /// packed GEMV owns few rows, the 128x128 tile owns prefill shapes; the
    /// crossover is per-device. Must be preceded by a matching `Q8::quant`.
    fn mm8(&self, q8: &crate::q8::Q8, s: &mut Vec<Step>, w: &crate::q8::Lin8, out: &DeviceBuffer, rows: u32) {
        let shape = OpShape { m: rows, n: w.n, k: w.k, dtype: Dtype::I8 };
        let variant = if rows <= DECODE_REGIME_MAX_ROWS {
            let bucket = rows.next_power_of_two().min(DECODE_REGIME_MAX_ROWS);
            self.tuned_i8
                .get(&(bucket, w.n, w.k))
                .copied()
                .unwrap_or_else(|| self.selector.select(Op::MatMul, shape, &self.caps))
        } else {
            self.selector.select(Op::MatMul, shape, &self.caps)
        };
        match variant {
            KernelVariant::WorkgroupPerOutput => s.push(self.gpu.step(
                MATMUL_I8_GEMV,
                &[&q8.xq, &w.packed, &q8.sx, &w.scale, out],
                &[rows, w.k / 4, w.n],
                w.n * 64,
            )),
            _ => q8.mm8(&self.gpu, s, w, out, rows),
        }
    }

    /// Measure the GEMV/tile crossover for every distinct int8 linear shape
    /// and row bucket on THIS device (S5). Both candidates are dispatched on
    /// the engine's real buffers — REPS dispatches per timing so submit/poll
    /// overhead amortises — and the winner persists per adapter + kernel
    /// sources. `BRAIN_NO_AUTOTUNE=1` skips every measurement (static policy).
    fn tune_i8(
        gpu: &Gpu,
        caps: &DeviceCaps,
        q8: &crate::q8::Q8,
        head: &crate::q8::Lin8,
        max_rows: u32,
    ) -> HashMap<(u32, u32, u32), KernelVariant> {
        let fp = gpu_core::tune::source_fingerprint(&[kernels::MATMUL_I8_GEMV, kernels::MATMUL_I8_DYN]);
        let store = gpu_core::tune::FileTuneStore::for_adapter(fp)
            .map(|s| Box::new(s) as Box<dyn gpu_core::select::TuneStore>);
        let tuner = AutoTuner::new(store);
        // Distinct (n, k) shapes: every layer shares them, so layer 0 + head
        // covers the whole model.
        let mut shapes: Vec<(u32, u32, &crate::q8::Lin8)> = Vec::new();
        if let Some(lay) = q8.layers.values().next() {
            for lin in [&lay.wq, &lay.wk, &lay.wv, &lay.wo, &lay.gate, &lay.up, &lay.down] {
                if !shapes.iter().any(|&(n, k, _)| n == lin.n && k == lin.k) {
                    shapes.push((lin.n, lin.k, lin));
                }
            }
        }
        shapes.push((head.n, head.k, head));
        let mut out = HashMap::new();
        let cap_bucket = max_rows.next_power_of_two().min(DECODE_REGIME_MAX_ROWS);
        for &m in &[1u32, 2, 4, 8, 16, 32] {
            if m > cap_bucket {
                break;
            }
            for &(n, k, lin) in &shapes {
                let shape = OpShape { m, n, k, dtype: Dtype::I8 };
                let mut measure =
                    |v: KernelVariant| Self::measure_i8(gpu, q8, lin, m, v);
                let choice = tuner.resolve(Op::MatMul, shape, caps, &mut measure);
                out.insert((m, n, k), choice);
            }
        }
        out
    }

    /// Time one int8 GEMM variant on real buffers: REPS dispatches in one
    /// submission, mean milliseconds per dispatch. `None` = not measurable.
    fn measure_i8(
        gpu: &Gpu,
        q8: &crate::q8::Q8,
        lin: &crate::q8::Lin8,
        m: u32,
        variant: KernelVariant,
    ) -> Option<f64> {
        const REPS: usize = 8;
        let out = gpu.storage(m as u64 * lin.n as u64);
        let step = |_: usize| match variant {
            KernelVariant::WorkgroupPerOutput => gpu.step(
                MATMUL_I8_GEMV,
                &[&q8.xq, &lin.packed, &q8.sx, &lin.scale, &out],
                &[m, lin.k / 4, lin.n],
                lin.n * 64,
            ),
            KernelVariant::PackedInt8 => gpu.step(
                MATMUL_I8_DYN,
                &[&q8.xq, &lin.packed, &q8.sx, &lin.scale, &out],
                &[m, lin.k / 4, lin.n],
                m.div_ceil(128) * lin.n.div_ceil(128) * 256,
            ),
            other => unreachable!("int8 candidates are GEMV or tile, got {other:?}"),
        };
        // Warm-up (pipeline residency, first-touch allocations), then timed.
        gpu.submit(&[], &[step(0)]);
        gpu.poll_wait();
        let steps: Vec<Step> = (0..REPS).map(step).collect();
        let t0 = std::time::Instant::now();
        gpu.submit(&[], &steps);
        gpu.poll_wait();
        Some(t0.elapsed().as_secs_f64() * 1e3 / REPS as f64)
    }

    /// RMSNorm, choosing the workgroup-per-row kernel at decode row counts
    /// (the per-element kernel runs `rows` threads — 8 threads on a 3840-core
    /// card at batch 8, measured at 16.6% of decode time).
    fn rms(&self, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, d: u32, rows: u32) -> Step {
        let g = &self.gpu;
        let shape = OpShape { m: rows, n: d, k: 0, dtype: Dtype::F32 };
        match self.selector.select(Op::RmsNorm, shape, &self.caps) {
            KernelVariant::WorkgroupPerOutput => g.step(RMSNORM_ROWS, &[x, w, out], &[d, rows, gpu_core::f(1e-6)], rows * 64),
            _ => g.step(RMSNORM, &[x, w, out], &[d, rows], rows),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_batched_submit(&self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) -> u32 {
        #[cfg(feature = "fault-injection")]
        if fault::take_kernel_failure() {
            panic!("injected fault: kernel dispatch failure");
        }
        let c = &self.cfg;
        let (d, ff, hd) = (c.d_model, c.d_ff, c.head_dim);
        let (hq, hkv) = (c.q_dim(), c.kv_dim());
        let (nh, nkv) = (c.n_heads, c.n_kv_heads);
        let group = nh / nkv;
        let half = hd / 2;
        let bs = self.block_size;
        let cap = self.cap;
        let mbt = self.max_blocks_per_seq;
        let scale = 1.0f32 / (hd as f32).sqrt();
        let theta = c.rope_theta;
        let g = &self.gpu;
        // Resident mode (A4): every input — token ids AND paged metadata — was
        // produced on the device by `decode_feed`/`decode_advance`, so writing
        // host copies here would both be wrong (stale) and force a flush.
        if !matches!(input, Input::Resident) {
            g.write(&self.sc.pos_buf, positions);
            g.write(&self.sc.seqlen_buf, seqlens);
            g.write(&self.sc.blk_buf, blocks);
            g.write(&self.sc.off_buf, offsets);
            g.write(&self.sc.bt_buf, bt);
        }
        let kids = ids();
        let sc = &self.sc;
        let w = |name: &str| self.ps.w(name);
        let b = bsz;
        let mut s: Vec<Step> = Vec::new();
        match input {
            Input::Tokens(t) => {
                g.write(&sc.tok_buf, t);
                s.push(g.step(EMBED, &[&sc.tok_buf, w("tok.weight"), &sc.res[0]], &[d, b], d * b));
            }
            Input::Resident => {
                s.push(g.step(EMBED, &[&sc.tok_buf, w("tok.weight"), &sc.res[0]], &[d, b], d * b));
            }
            Input::Embeds(e) => {
                g.write(&sc.res[0], bytemuck::cast_slice(e));
            }
        }
        for l in 0..c.n_layers as usize {
            let p = |name: &str| format!("blocks.{l}.{name}");
            let l8 = self.w8.as_ref().map(|q8| (q8, &q8.layers[&l]));
            s.push(self.rms(&sc.res[l], w(&p("ln1.weight")), &sc.xn1, d, b));
            if let Some((q8, lay)) = l8 {
                // One activation quant per distinct input, shared by every
                // linear reading it (xn1 -> q/k/v).
                q8.quant(g, &mut s, &sc.xn1, d, b);
                self.mm8(q8, &mut s, &lay.wq, &sc.q_pre, b);
                self.mm8(q8, &mut s, &lay.wk, &sc.k_pre, b);
                self.mm8(q8, &mut s, &lay.wv, &sc.v, b);
            } else {
                s.push(self.mm(&sc.xn1, w(&p("attn.wq.weight")), &sc.q_pre, b, d, hq));
                s.push(self.mm(&sc.xn1, w(&p("attn.wk.weight")), &sc.k_pre, b, d, hkv));
                s.push(self.mm(&sc.xn1, w(&p("attn.wv.weight")), &sc.v, b, d, hkv));
            }
            s.push(block::rmsnorm_fwd(g, &kids, &sc.q_pre, w(&p("attn.q_norm.weight")), &sc.q, hd, b * nh));
            s.push(block::rmsnorm_fwd(g, &kids, &sc.k_pre, w(&p("attn.k_norm.weight")), &sc.k, hd, b * nkv));
            s.push(g.step(ROPE_PAGED, &[&sc.q, &sc.pos_buf], &[b, nh, hd, hq, fb(theta)], b * nh * half));
            s.push(g.step(ROPE_PAGED, &[&sc.k, &sc.pos_buf], &[b, nkv, hd, hkv, fb(theta)], b * nkv * half));
            if self.kv_int8 {
                s.push(g.step(APPEND_I8, &[&sc.k, &sc.blk_buf, &sc.off_buf, &self.pool_k[l], &self.scales_k[l]], &[b, hkv, bs, hd], b * nkv));
                s.push(g.step(APPEND_I8, &[&sc.v, &sc.blk_buf, &sc.off_buf, &self.pool_v[l], &self.scales_v[l]], &[b, hkv, bs, hd], b * nkv));
                s.push(g.step(SCORES_I8, &[&sc.q, &self.pool_k[l], &sc.bt_buf, &sc.seqlen_buf, &self.scales_k[l], &sc.scores], &[b, nh, group, hd, bs, hkv, cap, mbt, fb(scale)], b * nh * cap));
                s.push(g.step(SOFTMAX_B, &[&sc.scores, &sc.seqlen_buf, &sc.probs], &[b, nh, cap], b * nh));
                s.push(g.step(APPLY_I8, &[&sc.probs, &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &self.scales_v[l], &sc.ctx], &[b, nh, group, hd, bs, hkv, cap, mbt], b * nh * hd));
            } else {
                s.push(g.step(KV_APPEND_B, &[&sc.k, &sc.blk_buf, &sc.off_buf, &self.pool_k[l]], &[b, hkv, bs], b * hkv));
                s.push(g.step(KV_APPEND_B, &[&sc.v, &sc.blk_buf, &sc.off_buf, &self.pool_v[l]], &[b, hkv, bs], b * hkv));
                s.push(g.step(SCORES_B, &[&sc.q, &self.pool_k[l], &sc.bt_buf, &sc.seqlen_buf, &sc.scores], &[b, nh, group, hd, bs, hkv, cap, mbt, fb(scale)], b * nh * cap));
                s.push(g.step(SOFTMAX_B, &[&sc.scores, &sc.seqlen_buf, &sc.probs], &[b, nh, cap], b * nh));
                s.push(g.step(APPLY_B, &[&sc.probs, &self.pool_v[l], &sc.bt_buf, &sc.seqlen_buf, &sc.ctx], &[b, nh, group, hd, bs, hkv, cap, mbt], b * nh * hd));
            }
            if let Some((q8, lay)) = l8 {
                q8.quant(g, &mut s, &sc.ctx, hq, b);
                self.mm8(q8, &mut s, &lay.wo, &sc.proj, b);
            } else {
                s.push(self.mm(&sc.ctx, w(&p("attn.wo.weight")), &sc.proj, b, hq, d));
            }
            s.push(g.step(ADD2, &[&sc.res[l], &sc.proj, &sc.xmid], &[b * d], b * d));
            s.push(self.rms(&sc.xmid, w(&p("ln2.weight")), &sc.xn2, d, b));
            if let Some((q8, lay)) = l8 {
                q8.quant(g, &mut s, &sc.xn2, d, b);
                self.mm8(q8, &mut s, &lay.gate, &sc.gate_pre, b);
                self.mm8(q8, &mut s, &lay.up, &sc.up, b);
                s.push(block::swiglu_fwd(g, &kids, &sc.gate_pre, &sc.up, &sc.h, b * ff));
                q8.quant(g, &mut s, &sc.h, ff, b);
                self.mm8(q8, &mut s, &lay.down, &sc.mlp_out, b);
            } else {
                s.push(self.mm(&sc.xn2, w(&p("mlp.gate.weight")), &sc.gate_pre, b, d, ff));
                s.push(self.mm(&sc.xn2, w(&p("mlp.up.weight")), &sc.up, b, d, ff));
                s.push(block::swiglu_fwd(g, &kids, &sc.gate_pre, &sc.up, &sc.h, b * ff));
                s.push(self.mm(&sc.h, w(&p("mlp.down.weight")), &sc.mlp_out, b, ff, d));
            }
            s.push(g.step(ADD2, &[&sc.xmid, &sc.mlp_out, &sc.res[l + 1]], &[b * d], b * d));
        }
        let last = c.n_layers as usize;
        s.push(self.rms(&sc.res[last], w("norm.weight"), &sc.xn_final, d, b));
        g.submit(&[], &s);
        b
    }

    /// One batched decode step that returns the **greedy next token per row**,
    /// with the LM head evaluated on the device.
    ///
    /// The head is the largest single matmul in a small model (`vocab x d_model`
    /// = 16.4M MACs per row at vocab 32k). Applying it on the host, once per
    /// sequence per token, made decode host-bound: cost grew linearly with batch
    /// size while the GPU idled, so continuous batching stopped paying — measured
    /// at ~85% of each decode step, and throughput that gained only 1.4x from
    /// concurrency 1->16 before regressing.
    ///
    /// Here the hidden state never leaves the device: `matmul` produces
    /// `[bsz, vocab]` logits (parallel over every output element) and
    /// `argmax_row` reduces each row, so only `bsz` indices are read back
    /// instead of a `[bsz, vocab]` block.
    #[allow(clippy::too_many_arguments)]
    fn run_batched_greedy(&self, bsz: u32, input: Input, positions: &[u32], seqlens: &[u32], blocks: &[u32], offsets: &[u32], bt: &[u32]) -> Vec<u32> {
        assert!(
            bsz <= self.max_batch,
            "greedy decode is sized for max_batch={} rows, got {bsz}",
            self.max_batch
        );
        self.run_batched_submit(bsz, input, positions, seqlens, blocks, offsets, bt);
        self.greedy_from_hidden(bsz)
    }

    /// `argmax(xn_final @ head^T)` per row, entirely on the device.
    ///
    /// Two-stage reduction: `argmax_part` splits each row into
    /// [`ARGMAX_CHUNKS`] chunks reduced by independent threads, `argmax_final`
    /// folds the partials — `bsz * chunks` threads instead of `bsz`. The
    /// original single-thread-per-row `argmax_row` scanned 32k logits alone
    /// and was 10.3% of decode time; it remains registered as the small-vocab
    /// path and the reference the tests compare against.
    fn greedy_from_hidden(&self, bsz: u32) -> Vec<u32> {
        self.submit_greedy_head(bsz);
        // Indices come back as f32 (exact below 2^24, far above any vocabulary).
        self.gpu.read(&self.argmax_dev, bsz as usize).into_iter().map(|x| x as u32).collect()
    }

    /// Record + submit the greedy head (logits + row argmax into
    /// `argmax_dev`) WITHOUT reading back — the on-device decode window feeds
    /// the result straight into the next step.
    fn submit_greedy_head(&self, bsz: u32) {
        let g = &self.gpu;
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        let mut steps: Vec<Step> = Vec::new();
        match (&self.w8, &self.head8, &self.head_dev) {
            // Int8 head: quantize the final hidden rows, DP4A GEMM into logits.
            (Some(q8), Some(h8), _) => {
                q8.quant(g, &mut steps, &self.sc.xn_final, d, bsz);
                self.mm8(q8, &mut steps, h8, &self.logits_dev, bsz);
            }
            (_, _, Some(head_dev)) => {
                steps.push(self.mm(&self.sc.xn_final, head_dev, &self.logits_dev, bsz, d, v));
            }
            _ => unreachable!("engine holds either an fp32 or an int8 head"),
        }
        let argmax_shape = OpShape { m: bsz, n: v, k: 0, dtype: Dtype::F32 };
        if self.selector.select(Op::ArgMaxRow, argmax_shape, &self.caps)
            == KernelVariant::SplitReduction
        {
            let chunk = v.div_ceil(ARGMAX_CHUNKS);
            steps.push(g.step(
                ARGMAX_PART,
                &[&self.logits_dev, &self.argmax_part_dev],
                &[bsz, v, ARGMAX_CHUNKS, chunk],
                bsz * ARGMAX_CHUNKS,
            ));
            steps.push(g.step(ARGMAX_FINAL, &[&self.argmax_part_dev, &self.argmax_dev], &[bsz, ARGMAX_CHUNKS], bsz));
        } else {
            steps.push(g.step(ARGMAX_ROW, &[&self.logits_dev, &self.argmax_dev], &[bsz, v], bsz));
        }
        g.submit(&[], &steps);
    }

    /// Advance every sequence by `k` tokens with ONE host round-trip (A4).
    ///
    /// The host allocates the window's K/V slots up front and uploads them as
    /// a device schedule; between sub-steps `decode_feed` turns the argmax
    /// into the next input and `decode_advance` walks the metadata, so the
    /// whole window records into one submission and the only readback is the
    /// `[k, bsz]` token history at the end. Callers own EOS handling: a
    /// sequence that finishes mid-window has its surplus tokens trimmed and
    /// its surplus K/V slots rolled back (`BlockTable::truncate`).
    ///
    /// Requires `k <= DECODE_WINDOW` and enough free blocks for every append;
    /// the scheduler falls back to the single-step path when either fails.
    pub(crate) fn forward_batched_greedy_window(
        &mut self,
        tables: &mut [&mut BlockTable],
        inputs: &[u32],
        k: usize,
    ) -> Vec<Vec<u32>> {
        assert!((1..=DECODE_WINDOW).contains(&k), "window {k} out of range");
        let (bsz, positions, seqlens, blocks, offsets, bt) = self.append_meta(tables);
        let mbt = self.max_blocks_per_seq;
        // Pre-allocate sub-steps 1..k and build the device schedule:
        // (block, offset, bt_index | NO_BT) per row per sub-step.
        const NO_BT: u32 = u32::MAX;
        let n = bsz as usize;
        let mut sched = vec![0u32; (k - 1) * n * 3];
        for s in 1..k {
            for (i, table) in tables.iter_mut().enumerate() {
                let before = table.blocks().len();
                let (block, offset) =
                    table.append(&mut self.alloc).expect("KV pool exhausted mid-window");
                let bti = if table.blocks().len() > before {
                    (table.blocks().len() - 1) as u32
                } else {
                    NO_BT
                };
                let base = ((s - 1) * n + i) * 3;
                sched[base] = block;
                sched[base + 1] = offset;
                sched[base + 2] = bti;
            }
        }
        let g = &self.gpu;
        if k > 1 {
            g.write(&self.sc.sched_buf, &sched);
        }
        // Sub-step 0: host-fed, as today — but the argmax stays on the device.
        self.run_batched_submit(bsz, Input::Tokens(inputs), &positions, &seqlens, &blocks, &offsets, &bt);
        self.submit_greedy_head(bsz);
        for s in 1..k {
            let g = &self.gpu;
            let feed = g.step(
                DECODE_FEED,
                &[&self.argmax_dev, &self.sc.tok_buf, &self.sc.hist_buf],
                &[bsz, (s - 1) as u32],
                bsz,
            );
            let adv = g.step(
                DECODE_ADVANCE,
                &[
                    &self.sc.sched_buf,
                    &self.sc.pos_buf,
                    &self.sc.seqlen_buf,
                    &self.sc.blk_buf,
                    &self.sc.off_buf,
                    &self.sc.bt_buf,
                ],
                &[bsz, (s - 1) as u32, mbt, NO_BT],
                bsz,
            );
            g.submit(&[], &[feed, adv]);
            self.run_batched_submit(bsz, Input::Resident, &[], &[], &[], &[], &[]);
            self.submit_greedy_head(bsz);
        }
        // Record the final sub-step's tokens, then read the whole window once.
        let g = &self.gpu;
        let last = g.step(
            DECODE_FEED,
            &[&self.argmax_dev, &self.sc.tok_buf, &self.sc.hist_buf],
            &[bsz, (k - 1) as u32],
            bsz,
        );
        g.submit(&[], &[last]);
        let flat = g.read(&self.sc.hist_buf, k * n);
        (0..n).map(|i| (0..k).map(|s| flat[s * n + i] as u32).collect()).collect()
    }

    /// **Chunked prefill**: process the prompt in chunks of up to `max_prefill`
    /// tokens. Each chunk is a batched forward whose C queries attend the paged
    /// prefix + the causal chunk (seqlens[i] = start+i+1), scattering K/V into the
    /// pool for the decode phase. One chunk == whole-prompt prefill; larger prompts
    /// stream through without a giant single forward. Returns the last token's
    /// final-norm hidden `[d_model]`.
    pub(crate) fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        assert!(table.is_empty(), "prefill expects a fresh sequence");
        // A prompt longer than the per-sequence capacity would write past its
        // row of the block table (`bt` is sized cc * max_blocks_per_seq), which
        // silently corrupts the next row's mapping. Callers must check
        // `max_seq_len()`; the scheduler rejects such requests at admission.
        assert!(
            prompt.len() <= self.max_seq_len(),
            "prompt of {} tokens exceeds the engine's per-sequence capacity of {} \
             (max_blocks_per_seq {} x block_size {})",
            prompt.len(),
            self.max_seq_len(),
            self.max_blocks_per_seq,
            self.block_size,
        );
        // An out-of-vocab id would make the embedding gather read out of
        // bounds — the kernels are trusted (no per-access clamps on either
        // backend), so the failure is silent garbage, not a clean error. The
        // scheduler rejects such requests at admission; this backstop catches
        // callers that bypass it.
        if let Some(&bad) = prompt.iter().find(|&&t| t >= self.cfg.vocab) {
            panic!("prompt token {bad} is outside the model vocabulary ({})", self.cfg.vocab);
        }
        let d = self.cfg.d_model as usize;
        let bs = self.block_size;
        let mbt = self.max_blocks_per_seq as usize;
        let n = prompt.len() as u32;
        let chunk = self.max_prefill.max(1);
        // Prefix reuse (D): adopt the longest cached chain of full prompt
        // blocks and compute only the tail. Always leave at least one token to
        // compute — the caller needs the LAST token's hidden state, which only
        // a real forward produces.
        let max_reuse = prompt.len().saturating_sub(1);
        let hits = self.prefix.lookup(prompt, bs, max_reuse);
        let matched = hits.len();
        if matched > 0 {
            table.adopt_prefix(&hits, &mut self.alloc);
        }
        self.prefix_lookup_tokens += prompt.len() as u64;
        self.prefix_hit_tokens += (matched as u64) * bs as u64;
        let mut last = Vec::new();
        let mut start = matched as u32 * bs;
        while start < n {
            let cc = (n - start).min(chunk);
            table.reserve(cc, &mut self.alloc).expect("KV pool exhausted");
            let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut bt = vec![0u32; cc as usize * mbt];
            for i in 0..cc {
                let pos = start + i;
                let (bl, off) = table.locate(pos, bs);
                positions.push(pos);
                seqlens.push(pos + 1); // causal: query i attends positions 0..=pos
                blocks.push(bl);
                offsets.push(off);
                for (lb, &phys) in table.blocks().iter().enumerate() {
                    bt[i as usize * mbt + lb] = phys;
                }
            }
            let hidden = self.run_batched(cc, Input::Tokens(&prompt[start as usize..(start + cc) as usize]), &positions, &seqlens, &blocks, &offsets, &bt);
            let cu = cc as usize;
            last = hidden[(cu - 1) * d..cu * d].to_vec();
            start += cc;
        }
        // Index this prompt's freshly-computed full blocks for later prompts.
        self.prefix.insert_chain(prompt, table.blocks(), matched, &mut self.alloc);
        last
    }

    /// Release up to `want` least-recently-used cache-only prefix blocks back
    /// to the pool — the admission path calls this when the pool is short.
    pub(crate) fn reclaim_prefix(&mut self, want: u32) -> u32 {
        self.prefix.evict(want, &mut self.alloc)
    }

    /// Prefix-cache effectiveness: `(tokens served from cache, tokens looked
    /// up, full blocks currently cached)`.
    pub fn prefix_stats(&self) -> (u64, u64, usize) {
        (self.prefix_hit_tokens, self.prefix_lookup_tokens, self.prefix.len())
    }

    /// Device-op accounting for this engine's handle (K) — what a benchmark
    /// records so submit/dispatch/readback cost is machine-readable. `None`
    /// where the backend does not count.
    pub fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.gpu.stats()
    }

    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        let (d, v) = (self.cfg.d_model as usize, self.cfg.vocab as usize);
        (0..v).map(|o| self.head[o * d..o * d + d].iter().zip(hidden).map(|(a, b)| a * b).sum()).collect()
    }

    /// Blocks free in the pool — the capacity figure `brain perf kvcache` sizes
    /// its overcommitted session mix against.
    pub fn free_blocks_for_perf(&self) -> u32 {
        self.alloc.free_blocks()
    }

    /// Prefill a prompt and return the final hidden state. Public so the
    /// `startup` benchmark can time a first real forward without going through
    /// the scheduler.
    pub fn prefill_for_perf(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        self.prefill(table, prompt)
    }

    /// Physical KV blocks currently free in the pool.
    pub fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    /// The prefill chunk size this engine was built with — the unit the
    /// scheduler's per-iteration prefill budget is expressed against.
    pub fn max_prefill_tokens(&self) -> u32 {
        self.max_prefill
    }

    /// The longest sequence (prompt + generated) this engine can hold for one
    /// request: `max_blocks_per_seq * block_size`.
    pub fn max_seq_len(&self) -> usize {
        (self.max_blocks_per_seq * self.block_size) as usize
    }

    /// Blocks a sequence of `tokens` length occupies (for admission checks).
    pub fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_size)
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }

    fn argmax(s: &[f32]) -> u32 {
        let mut bi = 0;
        for i in 1..s.len() {
            if s[i] > s[bi] {
                bi = i;
            }
        }
        bi as u32
    }

    /// Greedy generation of `max_new` tokens for each prompt, run with a **paged
    /// KV cache** and **batched decode** across all prompts concurrently. Prompts
    /// are prefilled per-sequence (one token per step), then every still-running
    /// sequence advances together each decode iteration. Returns the generated
    /// tokens per prompt. `eos` (when set) stops a sequence early.
    pub fn generate_greedy(&mut self, prompts: &[Vec<u32>], max_new: usize, eos: Option<u32>) -> Vec<Vec<u32>> {
        let mut seqs: Vec<Seq> = prompts.iter().map(|_| Seq { table: BlockTable::new(), generated: Vec::new(), done: false }).collect();

        // Prefill each sequence and sample its first token.
        for (i, prompt) in prompts.iter().enumerate() {
            assert!(!prompt.is_empty(), "empty prompt");
            let hidden = self.prefill(&mut seqs[i].table, prompt);
            let first = Self::argmax(&self.logits(&hidden));
            seqs[i].generated.push(first);
            if Some(first) == eos {
                seqs[i].done = true;
            }
        }

        // Batched decode: feed each running sequence its last token together.
        for _ in 1..max_new {
            let active: Vec<usize> = (0..seqs.len()).filter(|&i| !seqs[i].done).collect();
            if active.is_empty() {
                break;
            }
            let inputs: Vec<u32> = active.iter().map(|&i| *seqs[i].generated.last().unwrap()).collect();
            // Borrow the active sequences' block tables mutably for the batched step.
            let hidden = {
                let mut refs: Vec<&mut BlockTable> = Vec::new();
                for (idx, seq) in seqs.iter_mut().enumerate() {
                    if active.contains(&idx) {
                        refs.push(&mut seq.table);
                    }
                }
                self.forward_batched(&mut refs, &inputs)
            };
            let d = self.cfg.d_model as usize;
            for (bi, &si) in active.iter().enumerate() {
                let next = Self::argmax(&self.logits(&hidden[bi * d..(bi + 1) * d]));
                seqs[si].generated.push(next);
                if Some(next) == eos {
                    seqs[si].done = true;
                }
            }
        }
        for s in seqs.iter_mut() {
            self.release_table(&mut s.table);
        }
        seqs.into_iter().map(|s| s.generated).collect()
    }

    /// **Speculative decoding** (greedy): a `draft` proposes up to `k` tokens from
    /// the running context; the target verifies them in ONE batched forward,
    /// accepting the longest correct prefix plus a bonus/correction token, and
    /// rolling the paged cache back over any rejected tokens. The output is
    /// identical to plain greedy target decoding — the win is fewer (expensive)
    /// target forwards when the draft guesses well. `draft(ctx, want) -> tokens`.
    /// Returns `(generated_tokens, target_forward_count)`.
    pub fn spec_decode<D: FnMut(&[u32], u32) -> Vec<u32>>(&mut self, prompt: &[u32], max_new: usize, k: u32, mut draft: D) -> (Vec<u32>, usize) {
        assert!(!prompt.is_empty() && k >= 1);
        let d = self.cfg.d_model as usize;
        let bs = self.block_size;
        let mbt = self.max_blocks_per_seq as usize;
        let mut table = BlockTable::new();
        // Prefill all but the last prompt token; the last is the first `pending`.
        if prompt.len() > 1 {
            self.prefill(&mut table, &prompt[..prompt.len() - 1]);
        }
        let mut pending = *prompt.last().unwrap();
        let mut ctx: Vec<u32> = prompt.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut forwards = 0usize;

        while generated.len() < max_new {
            let want = ((max_new - generated.len()) as u32).min(k);
            let mut props = draft(&ctx, want);
            props.truncate(want as usize);
            let kk = props.len() as u32;

            // Verify forward over [pending, props...] at positions base..=base+kk.
            let base = table.len();
            let inputs: Vec<u32> = std::iter::once(pending).chain(props.iter().copied()).collect();
            let rows = kk + 1;
            table.reserve(rows, &mut self.alloc).expect("KV pool exhausted");
            let (mut positions, mut seqlens, mut blocks, mut offsets) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut bt = vec![0u32; rows as usize * mbt];
            for i in 0..rows {
                let pos = base + i;
                let (bl, off) = table.locate(pos, bs);
                positions.push(pos);
                seqlens.push(pos + 1);
                blocks.push(bl);
                offsets.push(off);
                for (lb, &phys) in table.blocks().iter().enumerate() {
                    bt[i as usize * mbt + lb] = phys;
                }
            }
            let hidden = self.run_batched(rows, Input::Tokens(&inputs), &positions, &seqlens, &blocks, &offsets, &bt);
            forwards += 1;

            // hidden[j] gives the target distribution that should have produced
            // props[j]; accept while it matches, else take the target's own token.
            let mut accepted = 0usize;
            let correction;
            loop {
                if accepted < kk as usize {
                    let pred = Self::argmax(&self.logits(&hidden[accepted * d..(accepted + 1) * d]));
                    if pred == props[accepted] {
                        accepted += 1;
                        continue;
                    }
                    correction = pred;
                    break;
                }
                // All drafts accepted → the bonus token from the last position.
                correction = Self::argmax(&self.logits(&hidden[kk as usize * d..(kk as usize + 1) * d]));
                break;
            }
            for prop in props.iter().take(accepted) {
                generated.push(*prop);
                ctx.push(*prop);
            }
            generated.push(correction);
            ctx.push(correction);
            // Commit pending + accepted drafts; the correction is the next pending.
            table.truncate(base + accepted as u32 + 1, &mut self.alloc);
            pending = correction;
        }
        generated.truncate(max_new);
        table.release(&mut self.alloc);
        (generated, forwards)
    }
}

/// A submitted generation request.
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub eos: Option<u32>,
}

/// Why a request was refused at admission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RejectReason {
    /// Can never fit the engine's per-sequence capacity.
    ExceedsCapacity { need: u32, capacity: u32 },
    /// Refused by the installed [`AdmissionPolicy`].
    PolicyRejected { policy: &'static str },
    /// A prompt token outside the model's vocabulary. Admitting it would make
    /// the embedding gather read out of bounds — the kernels are trusted (no
    /// per-access clamps), so the failure would be silent garbage, not an
    /// error.
    InvalidToken { token: u32, vocab: u32 },
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RejectReason::ExceedsCapacity { need, capacity } => {
                write!(f, "needs {need} tokens, engine capacity is {capacity}")
            }
            RejectReason::PolicyRejected { policy } => write!(f, "rejected by {policy}"),
            RejectReason::InvalidToken { token, vocab } => {
                write!(f, "token {token} is outside the vocabulary ({vocab})")
            }
        }
    }
}

/// What the queue looks like when an admission decision is made.
#[derive(Clone, Copy, Debug)]
pub struct QueueState {
    /// Requests waiting behind this one (its position in the queue).
    pub queued_ahead: usize,
    /// Sequences currently decoding.
    pub running: usize,
    /// KV blocks free in the pool.
    pub free_blocks: u32,
    /// Observed mean milliseconds to serve one request, when known.
    pub mean_service_ms: Option<f64>,
}

/// Decide what to do with work that arrives beyond capacity.
///
/// `perf overload` measured the default (queue without bound) collapsing at 2x
/// offered load: goodput fell below half its peak because compute was spent on
/// answers past their deadline. An engine is rewarded for refusing work it
/// provably cannot finish in time; policies are pure functions of
/// [`QueueState`], unit-testable with no engine at all.
pub trait AdmissionPolicy: Send + Sync {
    fn name(&self) -> &'static str;
    /// May this request enter the queue / stay admissible?
    fn admit(&self, req: &Request, state: &QueueState) -> bool;
}

/// Queue without bound — the historical behaviour and the default.
pub struct UnboundedQueue;
impl AdmissionPolicy for UnboundedQueue {
    fn name(&self) -> &'static str {
        "unbounded_queue"
    }
    fn admit(&self, _req: &Request, _state: &QueueState) -> bool {
        true
    }
}

/// Refuse once more than `max` requests are already waiting.
pub struct MaxQueueDepth(pub usize);
impl AdmissionPolicy for MaxQueueDepth {
    fn name(&self) -> &'static str {
        "max_queue_depth"
    }
    fn admit(&self, _req: &Request, state: &QueueState) -> bool {
        state.queued_ahead < self.0
    }
}

/// Refuse work that provably cannot start inside its deadline: everything
/// ahead must clear first, and if that alone exceeds the budget the compute
/// would be spent on an answer nobody can use.
pub struct DeadlineAware {
    /// Per-request start deadline, ms.
    pub deadline_ms: f64,
}
impl AdmissionPolicy for DeadlineAware {
    fn name(&self) -> &'static str {
        "deadline_aware"
    }
    fn admit(&self, _req: &Request, state: &QueueState) -> bool {
        match state.mean_service_ms {
            Some(svc) => (state.queued_ahead as f64) * svc <= self.deadline_ms,
            None => true, // nothing measured yet — cannot prove lateness
        }
    }
}

/// What one [`Scheduler::step_report`] iteration did. Latency metrics
/// (time-to-first-token, inter-token latency) are computed from this: [`Scheduler::step`]
/// alone reports only *completions*, which is too coarse to see when a sequence
/// was admitted or when each token landed, so no caller can derive TTFT/ITL from it.
#[derive(Debug, Default)]
pub struct StepReport {
    /// Requests admitted (prefilled + first token sampled) this iteration.
    pub admitted: Vec<u64>,
    /// `(id, tokens produced this iteration)` — the first token counts on the
    /// iteration the request was admitted.
    pub produced: Vec<(u64, usize)>,
    /// Requests that finished this iteration.
    pub finished: Vec<u64>,
    /// Requests refused at admission — impossible sizes and policy rejections
    /// alike. Refusing beats both crashing and queueing forever.
    pub rejected: Vec<(u64, RejectReason)>,
    /// The same `(id, tokens)` pairs [`Scheduler::step`] returns.
    pub completed: Vec<(u64, Vec<u32>)>,
}

/// A sequence the scheduler is actively decoding.
struct Running {
    id: u64,
    table: BlockTable,
    generated: Vec<u32>,
    max_new: usize,
    eos: Option<u32>,
    next_input: u32,
    done: bool,
}

/// **Continuous-batching scheduler.** Requests are submitted at any time, admitted
/// when the KV pool + batch have room (prefilled + first token sampled), then every
/// running sequence advances together in one batched decode step per iteration.
/// Finished sequences return their blocks immediately, so newly submitted requests
/// can be admitted mid-flight — the batch composition changes each iteration to keep
/// as much useful work resident as possible.
pub struct Scheduler {
    eng: Engine,
    waiting: std::collections::VecDeque<(u64, Request)>,
    running: Vec<Running>,
    next_id: u64,
    max_running: usize,
    /// Admission policy — what to do with work arriving beyond capacity.
    admission: Box<dyn AdmissionPolicy>,
    /// EWMA of ms per completed request, feeding DeadlineAware decisions.
    mean_service_ms: Option<f64>,
    started: std::collections::HashMap<u64, std::time::Instant>,
    /// Policy rejections made at submit time, surfaced in the next report.
    pending_rejects: Vec<(u64, RejectReason)>,
    /// Max prompt tokens prefilled per iteration before yielding to decode.
    ///
    /// Admission runs a FULL prefill per accepted request, so without a budget
    /// a burst of N arrivals performs N whole prompt forwards back-to-back
    /// while every running sequence stalls — measured as TTFA p99 growing
    /// 230 ms → 3413 ms (15×) and inter-token p99 10× from concurrency 1→32,
    /// with the interactive SLO met at no concurrency level. Bounding the
    /// prefill work per iteration lets decode run every iteration and spreads
    /// a burst across several; the budget always admits at least one waiting
    /// request per iteration, so nothing can starve.
    prefill_budget: u32,
}

impl Scheduler {
    pub fn new(eng: Engine, max_running: usize) -> Scheduler {
        // Default budget: two full prefill chunks per iteration. Enough to keep
        // admission moving under load, small enough that running sequences see
        // a decode step between arrivals.
        let prefill_budget = eng.max_prefill_tokens().saturating_mul(2).max(1);
        Scheduler {
            eng,
            waiting: std::collections::VecDeque::new(),
            running: Vec::new(),
            next_id: 0,
            max_running,
            admission: Box::new(UnboundedQueue),
            mean_service_ms: None,
            started: std::collections::HashMap::new(),
            pending_rejects: Vec::new(),
            prefill_budget,
        }
    }

    /// Install an admission policy (default: [`UnboundedQueue`], the historical
    /// behaviour). Applied at submit time; a refused request is reported in the
    /// next iteration's [`StepReport::rejected`].
    pub fn set_admission(&mut self, p: Box<dyn AdmissionPolicy>) {
        self.admission = p;
    }

    /// Override the per-iteration prefill budget (tokens). `u32::MAX` restores
    /// the old admit-everything behaviour; recorded by `brain perf` in the
    /// artifact's target config so a run states the policy it used.
    pub fn set_prefill_budget(&mut self, tokens: u32) {
        self.prefill_budget = tokens.max(1);
    }

    /// Enqueue a request; returns its id (results come back keyed by it).
    /// Submit a request. The admission policy is consulted HERE — a refusal
    /// returns the id with the request never queued, and the rejection appears
    /// in the next [`Scheduler::step_report`].
    pub fn submit(&mut self, req: Request) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let state = QueueState {
            queued_ahead: self.waiting.len(),
            running: self.running.len(),
            free_blocks: self.eng.free_blocks(),
            mean_service_ms: self.mean_service_ms,
        };
        if !self.admission.admit(&req, &state) {
            self.pending_rejects.push((id, RejectReason::PolicyRejected { policy: self.admission.name() }));
            return id;
        }
        self.started.insert(id, std::time::Instant::now());
        self.waiting.push_back((id, req));
        id
    }

    /// Cancel a request: drop it whether queued or mid-decode and return its KV
    /// blocks to the pool immediately. Returns the tokens produced so far, or
    /// `None` if the id is unknown (already finished, or never submitted).
    ///
    /// Without this, an abandoned request keeps decoding to `max_new` — spending
    /// device time on output nobody will read, and holding KV blocks that
    /// requests still being waited on need. Reclaiming on cancel is what stops a
    /// server under normal churn from losing its cache to dead sequences.
    pub fn cancel(&mut self, id: u64) -> Option<Vec<u32>> {
        if let Some(pos) = self.waiting.iter().position(|(qid, _)| *qid == id) {
            self.waiting.remove(pos);
            return Some(Vec::new()); // never admitted, so nothing was produced
        }
        let pos = self.running.iter().position(|r| r.id == id)?;
        let mut r = self.running.remove(pos);
        self.eng.release_table(&mut r.table);
        Some(r.generated)
    }

    /// Requests currently admitted and decoding.
    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Requests admitted but not yet started.
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// True while any request is waiting or running.
    pub fn pending(&self) -> bool {
        !self.waiting.is_empty() || !self.running.is_empty()
    }

    fn finish_check(r: &mut Running) {
        if Some(*r.generated.last().unwrap()) == r.eos || r.generated.len() >= r.max_new {
            r.done = true;
        }
    }

    /// One scheduler iteration, reporting **everything that happened** — not just
    /// what finished.
    ///
    /// [`Scheduler::step`] returns only completed requests, which is all a caller
    /// collecting outputs needs but leaves per-request latency unobservable: with
    /// completions alone you cannot tell when a sequence was admitted or when
    /// each token appeared, so time-to-first-token and inter-token latency cannot
    /// be computed at all. This variant additionally reports admissions and
    /// per-sequence token counts, which is what `brain perf` measures.
    pub fn step_report(&mut self) -> StepReport {
        let mut report = StepReport::default();
        report.rejected.append(&mut self.pending_rejects);
        let produced_before: HashMap<u64, usize> =
            self.running.iter().map(|r| (r.id, r.generated.len())).collect();

        let completed = self.step_inner(&mut report);

        // Tokens produced this iteration by sequences that are still running...
        for r in &self.running {
            let prev = produced_before.get(&r.id).copied().unwrap_or(0);
            if r.generated.len() > prev {
                report.produced.push((r.id, r.generated.len() - prev));
            }
        }
        // ...and by those that finished in this iteration.
        for (id, toks) in &completed {
            let prev = produced_before.get(id).copied().unwrap_or(0);
            if toks.len() > prev {
                report.produced.push((*id, toks.len() - prev));
            }
            report.finished.push(*id);
            if let Some(t0) = self.started.remove(id) {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                self.mean_service_ms =
                    Some(self.mean_service_ms.map_or(ms, |m| 0.8 * m + 0.2 * ms));
            }
        }
        report.completed = completed;
        report
    }

    /// The number of KV blocks still free in the pool — the memory-pressure
    /// signal a benchmark records alongside its latencies.
    pub fn free_blocks(&self) -> u32 {
        self.eng.free_blocks()
    }

    /// Prefix-cache effectiveness — see [`Engine::prefix_stats`].
    pub fn prefix_stats(&self) -> (u64, u64, usize) {
        self.eng.prefix_stats()
    }

    /// Device-op accounting — see [`Engine::device_stats`].
    pub fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.eng.device_stats()
    }

    /// One scheduler iteration: admit waiting requests that fit (prefill + sample
    /// first token), run one batched decode step over all running sequences, then
    /// reap completed ones. Returns the `(id, tokens)` of requests finished here.
    pub fn step(&mut self) -> Vec<(u64, Vec<u32>)> {
        let mut sink = StepReport::default();
        self.step_inner(&mut sink)
    }

    fn step_inner(&mut self, report: &mut StepReport) -> Vec<(u64, Vec<u32>)> {

        // 1. Admit while there's batch room, enough free blocks for the prompt,
        //    and prefill budget left this iteration (head-of-line guard: decode
        //    must run between bursts of admissions).
        let mut budget_left = self.prefill_budget;
        let mut admitted_this_iter = 0u32;
        while self.running.len() < self.max_running {
            // Drop anything that can never fit, whatever the pool does — it
            // would otherwise block the queue forever (or, before the capacity
            // check, corrupt the block table).
            let cap = self.eng.max_seq_len();
            let vocab = self.eng.vocab() as u32;
            while let Some((id, req)) = self.waiting.front() {
                let need = req.prompt.len() + req.max_new;
                let bad = req.prompt.iter().find(|&&t| t >= vocab).copied();
                if need <= cap && bad.is_none() {
                    break;
                }
                let (id, need) = (*id, need as u32);
                self.waiting.pop_front();
                self.started.remove(&id);
                let reason = match bad {
                    Some(token) => RejectReason::InvalidToken { token, vocab },
                    None => RejectReason::ExceedsCapacity { need, capacity: cap as u32 },
                };
                report.rejected.push((id, reason));
            }
            let fits = match self.waiting.front() {
                Some((_, req)) => {
                    let need = req.prompt.len() as u32;
                    // Always admit at least one request per iteration (no
                    // starvation); after that, stop once the budget is spent.
                    if admitted_this_iter > 0 && need > budget_left {
                        break;
                    }
                    let want = self.eng.blocks_for(need + 1);
                    let free = self.eng.free_blocks();
                    if free < want {
                        // Cached prefix blocks are reclaimable capacity: live
                        // sequences always outrank the cache.
                        self.eng.reclaim_prefix(want - free);
                    }
                    self.eng.free_blocks() >= want
                }
                None => false,
            };
            if !fits {
                break;
            }
            let (id, req) = self.waiting.pop_front().unwrap();
            budget_left = budget_left.saturating_sub(req.prompt.len() as u32);
            admitted_this_iter += 1;
            let mut table = BlockTable::new();
            let hidden = self.eng.prefill(&mut table, &req.prompt);
            let first = Engine::argmax(&self.eng.logits(&hidden));
            let mut r = Running { id, table, generated: vec![first], max_new: req.max_new, eos: req.eos, next_input: first, done: false };
            Self::finish_check(&mut r);
            report.admitted.push(id);
            self.running.push(r);
        }

        // 2. Batched decode over every running (not-done) sequence. When
        //    nothing is waiting to be admitted, decode a WINDOW of tokens per
        //    host round-trip (A4): the readback-per-token becomes a readback
        //    per window, at the cost of up to window-1 wasted decode steps for
        //    a sequence that hits EOS mid-window (its surplus K/V is rolled
        //    back below). With work waiting, the window stays 1 so admission
        //    latency is never traded away silently.
        let active: Vec<usize> = (0..self.running.len()).filter(|&i| !self.running[i].done).collect();
        if !active.is_empty() {
            let inputs: Vec<u32> = active.iter().map(|&i| self.running[i].next_input).collect();
            let remaining_min = active
                .iter()
                .map(|&i| {
                    let r = &self.running[i];
                    r.max_new.saturating_sub(r.generated.len()).max(1)
                })
                .min()
                .unwrap_or(1);
            let mut k = if self.waiting.is_empty() { remaining_min.min(DECODE_WINDOW) } else { 1 };
            // Every append must succeed mid-window (no host decisions there):
            // require a comfortable block reserve, else fall back to one step.
            if k > 1 && (self.eng.free_blocks() as usize) < active.len() * k {
                k = 1;
            }
            let window = {
                let mut refs: Vec<&mut BlockTable> = Vec::new();
                for (idx, r) in self.running.iter_mut().enumerate() {
                    if active.contains(&idx) {
                        refs.push(&mut r.table);
                    }
                }
                if k > 1 {
                    self.eng.forward_batched_greedy_window(&mut refs, &inputs, k)
                } else {
                    self.eng
                        .forward_batched_greedy(&mut refs, &inputs)
                        .into_iter()
                        .map(|t| vec![t])
                        .collect()
                }
            };
            for (bi, &si) in active.iter().enumerate() {
                let r = &mut self.running[si];
                let mut used = 0usize;
                for &next in &window[bi] {
                    r.generated.push(next);
                    r.next_input = next;
                    used += 1;
                    Self::finish_check(r);
                    if r.done {
                        break;
                    }
                }
                // A sequence that finished mid-window consumed only `used`
                // inputs; the remaining pre-allocated slots hold garbage K/V
                // and are rolled back so the pool never leaks waste.
                let surplus = (k - used) as u32;
                if surplus > 0 {
                    let len = r.table.len();
                    r.table.truncate(len - surplus, &mut self.eng.alloc);
                }
            }
        }

        // 3. Reap completed sequences, returning their blocks to the pool.
        let mut completed = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            if self.running[i].done {
                let mut r = self.running.remove(i);
                self.eng.release_table(&mut r.table);
                completed.push((r.id, r.generated));
            } else {
                i += 1;
            }
        }
        completed
    }

    /// Drive to completion, returning every request's tokens keyed by id.
    pub fn run(&mut self) -> HashMap<u64, Vec<u32>> {
        let mut out = HashMap::new();
        while self.pending() {
            for (id, toks) in self.step() {
                out.insert(id, toks);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Qwen;
    use data::rng::Rng;

    fn tiny_weights(cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
        let mut rng = Rng::new(1);
        let mut map = HashMap::new();
        for (name, count) in cfg.param_list() {
            let v = if name.contains("norm") { vec![1.0f32; count] } else { (0..count).map(|_| rng.next_gaussian() as f32 * 0.05).collect() };
            map.insert(name, v);
        }
        map
    }

    /// Single-sequence paged/batched serving must match the reference contiguous
    /// KV generation (`Qwen::generate_kv`) token-for-token, and a two-sequence
    /// batch must equal running each prompt on its own — proving batched paged
    /// decode is exact.
    #[test]
    fn batched_serving_matches_reference() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let bs = 4;
        let (num_blocks, max_batch, mbt) = (64u32, 4u32, 8u32);

        // Reference: the committed single-sequence KV generation.
        let model = Qwen::new(cfg.clone(), 1, 64, &map);
        let p0 = vec![1u32, 5, 3, 9];
        let p1 = vec![7u32, 2, 4];
        let mut r0 = Rng::new(0);
        let mut r1 = Rng::new(0);
        let ref0 = crate::sample::generate_kv(&model, &p0, 12, 0.0, 0, 1.0, None, &mut r0);
        let ref1 = crate::sample::generate_kv(&model, &p1, 12, 0.0, 0, 1.0, None, &mut r1);

        // Engine: run both prompts concurrently (batched paged).
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, bs, num_blocks, max_batch, mbt, 32, false, false);
        let out = eng.generate_greedy(&[p0.clone(), p1.clone()], 12, None);

        assert_eq!(out[0], ref0, "seq0 batched paged != reference");
        assert_eq!(out[1], ref1, "seq1 batched paged != reference");
    }

    /// THE prefix-cache invariant: a warm prefill (served from cached blocks)
    /// must produce output IDENTICAL to the cold one — a cache hit that
    /// changes a single token is corruption, not a cache. Also pins that the
    /// cache actually engaged (a test that silently measured two cold runs
    /// would prove nothing).
    #[test]
    fn warm_prefill_is_identical_to_cold() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 2, 12, 16, false, false);
        let prompt: Vec<u32> = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8];
        let cold = eng.generate_greedy(&[prompt.clone()], 10, None);
        let (hit0, _, cached) = eng.prefix_stats();
        assert_eq!(hit0, 0, "first prefill must be cold");
        assert!(cached > 0, "full prompt blocks must be indexed after prefill");
        let warm = eng.generate_greedy(&[prompt.clone()], 10, None);
        let (hit1, _, _) = eng.prefix_stats();
        assert!(hit1 > 0, "the second prefill must actually reuse the prefix");
        assert_eq!(warm, cold, "a cache hit must be byte-identical to computing the prefix");
    }

    /// Random shared prefixes: a prompt sharing a random-length prefix with
    /// earlier traffic must prefill (through adopted cached blocks) to the
    /// same final hidden state a fresh engine computes — within rounding.
    ///
    /// Deliberately NOT a token-equality test: reused KV is bit-identical to
    /// its original computation, but the CPU backend's blocked GEMMs are not
    /// row-count-invariant in final-bit rounding, so a tail-only prefill can
    /// differ from a full one by an ulp — which flips argmax on a degenerate
    /// random model while meaning nothing. Structural corruption (a wrongly
    /// adopted block) produces O(1) relative error; rounding produces ~1e-6.
    /// The 1e-3 gate separates them cleanly. Token-level identity is pinned by
    /// `warm_prefill_is_identical_to_cold` where chunking is identical.
    #[test]
    fn random_shared_prefixes_stay_exact() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut cached_eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 128, 2, 12, 16, false, false);
        let mut rng = Rng::new(42);
        let vocab = cfg.vocab as u64;
        let base: Vec<u32> = (0..14).map(|_| (rng.next_u64() % vocab) as u32).collect();
        for trial in 0..6 {
            let keep = (rng.next_u64() as usize) % base.len();
            let mut prompt = base[..keep].to_vec();
            let extra = 3 + (rng.next_u64() as usize) % 6;
            prompt.extend((0..extra).map(|_| (rng.next_u64() % vocab) as u32));
            // Reference: a fresh engine has an empty cache by construction.
            let mut fresh = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 128, 2, 12, 16, false, false);
            let mut tf = BlockTable::new();
            let cold = fresh.prefill(&mut tf, &prompt);
            let mut tc = BlockTable::new();
            let warm = cached_eng.prefill(&mut tc, &prompt);
            let err: f32 = warm.iter().zip(&cold).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            let norm: f32 = cold.iter().map(|v| v * v).sum::<f32>().sqrt();
            let rel = err / norm.max(1e-12);
            assert!(
                rel < 1e-3,
                "trial {trial}: warm prefill diverged structurally (rel L2 {rel:.6}) on prompt {prompt:?}"
            );
            cached_eng.release_table(&mut tc);
        }
        let (hit, looked, _) = cached_eng.prefix_stats();
        assert!(hit > 0, "at least one trial must have actually reused a prefix ({hit}/{looked})");
    }

    /// Int8 weights (A0) must stay numerically faithful to the fp32 engine:
    /// same weights, same prompt, and the final-norm hidden state after a
    /// quantized prefill must sit within a few percent of the fp32 one. A
    /// scale-handling bug (the realistic failure mode) produces ~100% error,
    /// so the 10% gate separates cleanly while tolerating honest quant noise.
    /// The stream-level greedy-agreement threshold lives in `brain perf`'s
    /// fidelity gate, which measures it on real checkpoints.
    #[test]
    fn int8_weights_track_fp32() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng8 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 2, 8, 32, false, true);
        if !eng8.weights_int8() {
            // Capability-gated fallback (CPU JIT): the engine must run fp32
            // and say so. A device whose caps DO report the packed-int8 path
            // must never land here — a silent fallback on capable hardware is
            // exactly the regression this branch once masked.
            assert!(
                !eng8.gpu().caps().numeric.int8_dot,
                "device reports int8_dot but the engine fell back to fp32"
            );
            eprintln!("skipping int8 comparison: device has no packed-int8 path");
            return;
        }
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 2, 8, 32, false, false);
        let prompt = vec![1u32, 5, 3, 9, 2, 7];
        let mut t8 = BlockTable::new();
        let h8 = eng8.prefill(&mut t8, &prompt);
        let mut tf = BlockTable::new();
        let hf = eng.prefill(&mut tf, &prompt);
        let dot_err: f32 = h8.iter().zip(&hf).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        let norm: f32 = hf.iter().map(|v| v * v).sum::<f32>().sqrt();
        let rel = dot_err / norm.max(1e-12);
        assert!(rel < 0.10, "int8 hidden state diverged from fp32: relative L2 {rel:.4}");
        // And the engine still decodes end-to-end on the int8 path.
        let out = eng8.generate_greedy(&[prompt], 8, None);
        assert_eq!(out[0].len(), 8, "int8 engine must produce the requested tokens");
    }

    /// A request too long for the engine must be REJECTED, not crash the process
    /// and not sit in the queue forever. Before the capacity check, a prompt
    /// longer than `max_blocks_per_seq * block_size` wrote past its row of the
    /// block table.
    #[test]
    fn oversized_requests_are_rejected_not_fatal() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        // capacity = max_blocks_per_seq(4) * block_size(4) = 16 tokens.
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 2, 4, 8, false, false);
        assert_eq!(eng.max_seq_len(), 16);
        let mut sched = Scheduler::new(eng, 2);

        let huge = sched.submit(Request { prompt: vec![1u32; 64], max_new: 4, eos: None });
        let ok = sched.submit(Request { prompt: vec![2u32, 3, 4], max_new: 4, eos: None });

        let rep = sched.step_report();
        assert_eq!(rep.rejected.len(), 1, "the oversized request must be refused");
        assert_eq!(rep.rejected[0].0, huge);
        assert!(matches!(rep.rejected[0].1, RejectReason::ExceedsCapacity { .. }));

        // The queue keeps moving and the viable request completes. With the
        // decode window a short request can finish inside the SAME iteration
        // that admitted it, so its tokens arrive in that report's `completed`.
        let mut out = sched.run();
        out.extend(rep.completed);
        assert!(out.contains_key(&ok));
        assert!(!out.contains_key(&huge));
    }

    /// An out-of-vocab token must be REJECTED at admission with a typed
    /// reason. Admitting it would make the embedding gather read out of
    /// bounds — the kernels are trusted, so the failure would be silent
    /// garbage in the hidden states (found the hard way: NaN on CPU, wrong
    /// finite values on GPU), not an error anyone can see.
    #[test]
    fn out_of_vocab_tokens_are_rejected_not_gathered() {
        let cfg = QwenConfig::tiny();
        let vocab = cfg.vocab;
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 2, 8, 8, false, false);
        let mut sched = Scheduler::new(eng, 2);
        let bad = sched.submit(Request { prompt: vec![1, vocab + 7, 2], max_new: 4, eos: None });
        let ok = sched.submit(Request { prompt: vec![1, 2, 3], max_new: 4, eos: None });
        let rep = sched.step_report();
        assert_eq!(rep.rejected.len(), 1);
        assert_eq!(rep.rejected[0].0, bad);
        assert!(matches!(rep.rejected[0].1, RejectReason::InvalidToken { token, .. } if token == vocab + 7));
        let mut out = sched.run();
        out.extend(rep.completed);
        assert!(out.contains_key(&ok), "valid work behind the bad request still completes");
        assert!(!out.contains_key(&bad));
    }

    /// Admission policies must refuse at submit time, report the refusal, and
    /// leave already-queued work untouched.
    #[test]
    fn admission_policy_rejects_and_reports() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 2, 12, 8, false, false);
        let mut sched = Scheduler::new(eng, 2);
        sched.set_admission(Box::new(MaxQueueDepth(1)));

        let a = sched.submit(Request { prompt: vec![1, 5, 3], max_new: 4, eos: None }); // queued (0 ahead)
        let b = sched.submit(Request { prompt: vec![2, 6, 4], max_new: 4, eos: None }); // queued (1 ahead? depth=1 => 1 not < 1 => REJECTED)
        let rep = sched.step_report();
        assert_eq!(rep.rejected.len(), 1, "the over-depth submit must be refused");
        assert_eq!(rep.rejected[0].0, b);
        assert!(matches!(rep.rejected[0].1, RejectReason::PolicyRejected { policy: "max_queue_depth" }));

        let mut out = sched.run();
        out.extend(rep.completed);
        assert!(out.contains_key(&a), "admitted work completes normally");
        assert!(!out.contains_key(&b));
    }

    /// DeadlineAware admits until a service time is measured, then refuses work
    /// that provably cannot start in time.
    #[test]
    fn deadline_aware_uses_measured_service_time() {
        let p = DeadlineAware { deadline_ms: 100.0 };
        let mk = |queued, svc| QueueState {
            queued_ahead: queued,
            running: 0,
            free_blocks: 99,
            mean_service_ms: svc,
        };
        let r = Request { prompt: vec![1], max_new: 1, eos: None };
        assert!(p.admit(&r, &mk(50, None)), "no measurement -> cannot prove lateness");
        assert!(p.admit(&r, &mk(4, Some(20.0))), "4 x 20ms fits a 100ms deadline");
        assert!(!p.admit(&r, &mk(6, Some(20.0))), "6 x 20ms provably misses it");
    }

    /// Cancelling must free the sequence's KV blocks immediately and stop its
    /// decoding, without disturbing the requests running alongside it.
    #[test]
    fn cancel_reclaims_blocks_and_spares_neighbours() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 3, 12, 8, false, false);
        let mut sched = Scheduler::new(eng, 3);

        // Everything long enough to still be decoding after two (windowed)
        // iterations, and inside the per-sequence capacity (12 blocks x 4 = 48).
        let keep_a = sched.submit(Request { prompt: vec![1u32, 5, 3], max_new: 20, eos: None });
        let doomed = sched.submit(Request { prompt: vec![7u32, 2, 9], max_new: 30, eos: None });
        let keep_b = sched.submit(Request { prompt: vec![4u32, 4, 1], max_new: 20, eos: None });

        // Admit everything and decode a couple of steps.
        sched.step();
        sched.step();
        assert_eq!(sched.running_len(), 3);
        let free_before = sched.free_blocks();

        let produced = sched.cancel(doomed).expect("cancelling a running request must succeed");
        assert!(!produced.is_empty(), "it had already produced tokens");
        assert_eq!(sched.running_len(), 2, "only the cancelled request is removed");
        assert!(sched.free_blocks() > free_before, "its KV blocks must return to the pool at once");

        // The survivors still finish normally.
        let out = sched.run();
        assert_eq!(out.len(), 2);
        assert_eq!(out[&keep_a].len(), 20);
        assert_eq!(out[&keep_b].len(), 20);
        assert!(!out.contains_key(&doomed), "a cancelled request must not complete");

        // Cancelling twice, or an unknown id, is a no-op rather than a panic.
        assert!(sched.cancel(doomed).is_none());
        assert!(sched.cancel(9999).is_none());
    }

    /// Cancelling a request that was never admitted just drops it from the queue.
    #[test]
    fn cancel_before_admission_produces_nothing() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        // One slot, so the second request cannot be admitted.
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 32, 1, 12, 8, false, false);
        let mut sched = Scheduler::new(eng, 1);
        let _a = sched.submit(Request { prompt: vec![1u32, 5, 3], max_new: 4, eos: None });
        let queued = sched.submit(Request { prompt: vec![2u32, 6, 4], max_new: 4, eos: None });
        sched.step();
        assert_eq!(sched.waiting_len(), 1);
        assert_eq!(sched.cancel(queued), Some(Vec::new()));
        assert_eq!(sched.waiting_len(), 0);
    }

    /// The device-side greedy head must select exactly the token the host head
    /// would. This is the invariant that lets decode skip shipping a
    /// `[batch, vocab]` logit block back to the host — if it ever drifted, the
    /// engine would silently generate different text at speed.
    #[test]
    fn device_head_argmax_matches_the_host_head() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 4, 12, 8, false, false);

        // Drive a few real decode steps so the hidden states are genuine.
        let mut tables: Vec<BlockTable> = (0..3).map(|_| BlockTable::new()).collect();
        let prompts = [vec![1u32, 5, 3], vec![7u32, 2, 9], vec![4u32, 4, 1]];
        let mut inputs = Vec::new();
        for (t, p) in tables.iter_mut().zip(prompts.iter()) {
            let h = eng.prefill(t, p);
            inputs.push(Engine::argmax(&eng.logits(&h)));
        }
        for _ in 0..4 {
            // Host reference: hidden -> host matmul -> argmax, per row.
            let hidden = {
                let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                eng.forward_batched(&mut refs, &inputs)
            };
            let d = eng.cfg.d_model as usize;
            let host: Vec<u32> =
                (0..inputs.len()).map(|i| Engine::argmax(&eng.logits(&hidden[i * d..(i + 1) * d]))).collect();
            // Device: same hidden already in sc.xn_final, head applied on device.
            let dev = eng.greedy_from_hidden(inputs.len() as u32);
            assert_eq!(dev, host, "device head picked a different token than the host head");
            inputs = host;
        }
    }

    /// The prefill budget must spread a burst of admissions across iterations
    /// (decode runs between them) without changing ANY output, and must never
    /// starve: at least one waiting request is admitted per iteration.
    #[test]
    fn prefill_budget_spreads_admissions_without_changing_outputs() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let reqs = || {
            (0..4u32)
                .map(|i| Request { prompt: vec![1 + i, 5, 3, 7, 2], max_new: 5, eos: None })
                .collect::<Vec<_>>()
        };

        // Reference: unlimited budget (the old behaviour).
        let mut a = Scheduler::new(Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 96, 4, 12, 8, false, false), 4);
        a.set_prefill_budget(u32::MAX);
        for r in reqs() {
            a.submit(r);
        }
        let want = a.run();

        // Tight budget: one 5-token prompt exhausts it, so the 4 arrivals must
        // be admitted over MULTIPLE iterations — with decode in between — and
        // still produce token-identical outputs.
        let mut b = Scheduler::new(Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 4, 12, 8, false, false), 4);
        b.set_prefill_budget(5);
        for r in reqs() {
            b.submit(r);
        }
        let mut admit_iters = 0;
        let mut got = std::collections::HashMap::new();
        while b.pending() {
            let rep = b.step_report();
            if !rep.admitted.is_empty() {
                admit_iters += 1;
                assert!(
                    rep.admitted.len() <= 2,
                    "a 5-token budget must not admit a whole burst at once, got {}",
                    rep.admitted.len()
                );
            }
            for (id, toks) in rep.completed {
                got.insert(id, toks);
            }
        }
        assert!(admit_iters >= 2, "admissions must be spread across iterations");
        assert_eq!(got, want, "the budget changes WHEN tokens appear, never WHICH");
    }

    /// The two-stage argmax (vocab >= ARGMAX_SPLIT_MIN_VOCAB) must pick exactly
    /// the token the host head picks — including the lowest-index tie-break.
    #[test]
    fn split_argmax_matches_the_host_head_at_large_vocab() {
        let mut cfg = QwenConfig::tiny();
        cfg.vocab = 8192; // forces the argmax_part/argmax_final path
        let map = tiny_weights(&cfg);
        let mut eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 3, 12, 8, false, false);

        let mut tables: Vec<BlockTable> = (0..3).map(|_| BlockTable::new()).collect();
        let prompts = [vec![11u32, 55, 33], vec![77u32, 22, 99], vec![44u32, 45, 46]];
        let mut inputs = Vec::new();
        for (t, p) in tables.iter_mut().zip(prompts.iter()) {
            let h = eng.prefill(t, p);
            inputs.push(Engine::argmax(&eng.logits(&h)));
        }
        for _ in 0..3 {
            let hidden = {
                let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
                eng.forward_batched(&mut refs, &inputs)
            };
            let d = eng.cfg.d_model as usize;
            let host: Vec<u32> =
                (0..inputs.len()).map(|i| Engine::argmax(&eng.logits(&hidden[i * d..(i + 1) * d]))).collect();
            let dev = eng.greedy_from_hidden(inputs.len() as u32);
            assert_eq!(dev, host, "split argmax diverged from the host head");
            inputs = host;
        }
    }

    /// `step_report` must account for **every** token exactly once and admit each
    /// request exactly once. `brain perf` derives time-to-first-token and
    /// inter-token latency purely from these counts, so a double-count or a
    /// dropped token silently corrupts every latency number computed from it.
    #[test]
    fn step_report_accounts_for_every_token_exactly_once() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 3, 12, 8, false, false);
        let mut sched = Scheduler::new(eng, 3);

        let wants = [6usize, 4, 9];
        let ids: Vec<u64> = wants
            .iter()
            .enumerate()
            .map(|(i, &n)| sched.submit(Request { prompt: vec![1u32 + i as u32, 5, 3], max_new: n, eos: None }))
            .collect();

        let mut produced: HashMap<u64, usize> = HashMap::new();
        let mut admitted: Vec<u64> = Vec::new();
        let mut finished: Vec<u64> = Vec::new();
        let mut outputs: HashMap<u64, Vec<u32>> = HashMap::new();
        while sched.pending() {
            let rep = sched.step_report();
            admitted.extend(rep.admitted.iter().copied());
            for (id, n) in &rep.produced {
                *produced.entry(*id).or_default() += n;
            }
            finished.extend(rep.finished.iter().copied());
            for (id, toks) in rep.completed {
                outputs.insert(id, toks);
            }
        }

        admitted.sort_unstable();
        finished.sort_unstable();
        let mut expect = ids.clone();
        expect.sort_unstable();
        assert_eq!(admitted, expect, "every request must be admitted exactly once");
        assert_eq!(finished, expect, "every request must finish exactly once");

        for (i, id) in ids.iter().enumerate() {
            let out = outputs.get(id).expect("request must complete");
            assert_eq!(out.len(), wants[i], "request {id} produced the wrong length");
            assert_eq!(
                produced.get(id).copied().unwrap_or(0),
                out.len(),
                "incremental token report for {id} must sum to the tokens returned"
            );
        }
    }

    /// The reporting path must not change what the scheduler computes.
    #[test]
    fn step_report_does_not_perturb_outputs() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let reqs = || {
            vec![
                Request { prompt: vec![1u32, 5, 3], max_new: 7, eos: None },
                Request { prompt: vec![9u32, 2], max_new: 5, eos: None },
            ]
        };

        let mut a = Scheduler::new(Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 96, 2, 12, 8, false, false), 2);
        for r in reqs() {
            a.submit(r);
        }
        let via_step = a.run();

        let mut b = Scheduler::new(Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 96, 2, 12, 8, false, false), 2);
        for r in reqs() {
            b.submit(r);
        }
        let mut via_report: HashMap<u64, Vec<u32>> = HashMap::new();
        while b.pending() {
            for (id, toks) in b.step_report().completed {
                via_report.insert(id, toks);
            }
        }
        assert_eq!(via_step, via_report, "the reporting path must be observationally identical");
    }

    /// Continuous batching: requests submitted at DIFFERENT times (one mid-flight)
    /// must each produce the same tokens as run alone — the scheduler admits,
    /// batches, completes, and frees dynamically without changing any output.
    #[test]
    fn scheduler_dynamic_admission_matches_reference() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let model = Qwen::new(cfg.clone(), 1, 64, &map);

        let prompts = [vec![1u32, 5, 3, 9], vec![7u32, 2, 4], vec![3u32, 3, 8, 1, 6]];
        let maxn = [10usize, 6, 8];
        let refs: Vec<Vec<u32>> = prompts
            .iter()
            .zip(maxn)
            .map(|(p, n)| {
                let mut r = Rng::new(0);
                crate::sample::generate_kv(&model, p, n, 0.0, 0, 1.0, None, &mut r)
            })
            .collect();

        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 4, 8, 32, false, false);
        let mut sched = Scheduler::new(eng, 4);
        let mut out: HashMap<u64, Vec<u32>> = HashMap::new();

        let id0 = sched.submit(Request { prompt: prompts[0].clone(), max_new: maxn[0], eos: None });
        let id1 = sched.submit(Request { prompt: prompts[1].clone(), max_new: maxn[1], eos: None });
        // Run two iterations with only the first two requests active...
        for _ in 0..2 {
            for (id, t) in sched.step() {
                out.insert(id, t);
            }
        }
        // ...then submit a third mid-flight; it must batch in and still be correct.
        let id2 = sched.submit(Request { prompt: prompts[2].clone(), max_new: maxn[2], eos: None });
        while sched.pending() {
            for (id, t) in sched.step() {
                out.insert(id, t);
            }
        }

        assert_eq!(out[&id0], refs[0], "req0 under continuous batching != reference");
        assert_eq!(out[&id1], refs[1], "req1 under continuous batching != reference");
        assert_eq!(out[&id2], refs[2], "mid-flight req2 != reference");
    }

    /// int8 paged KV stays close to fp32 through prefill + decode (both read the
    /// quantised cache) — a ~4× smaller KV pool for a small, bounded error.
    #[test]
    fn int8_kv_close_to_fp32() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9, 2];
        let run = |int8: bool| -> Vec<f32> {
            let mut e = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, int8, false);
            let mut t = BlockTable::new();
            let mut hidden = e.prefill(&mut t, &prompt);
            for _ in 0..6 {
                let next = Engine::argmax(&e.logits(&hidden));
                let mut one = [&mut t];
                hidden = e.forward_batched(&mut one, &[next]);
            }
            hidden
        };
        let h32 = run(false);
        let h8 = run(true);
        let err = h32.iter().zip(&h8).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let mag = h32.iter().fold(0f32, |m, &x| m.max(x.abs()));
        println!("int8 KV vs fp32 (prefill + 6 decode) maxabs={err:e} (mag {mag:e})");
        assert!(err < 0.2 * mag + 1e-3, "int8 diverges too far: {err} vs mag {mag}");
    }

    /// Chunked prefill (small chunk) must produce the same hidden as whole-prompt
    /// prefill — the prompt streams through in pieces attending the paged prefix.
    #[test]
    fn chunked_prefill_matches_whole() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9, 2, 7, 4, 8];
        let prefill_last = |max_prefill: u32| -> Vec<f32> {
            let mut e = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, max_prefill, false, false);
            let mut t = BlockTable::new();
            e.prefill(&mut t, &prompt)
        };
        let whole = prefill_last(16); // one chunk
        let chunked = prefill_last(2); // 4 chunks of 2
        let err = whole.iter().zip(&chunked).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        println!("chunked (2) vs whole prefill: maxabs={err:e}");
        assert!(err < 1e-4, "chunked prefill != whole prefill: {err}");
    }

    /// Speculative decoding output equals plain greedy — with a good (oracle)
    /// draft it takes far fewer target forwards; with a bad draft it falls back to
    /// ~one token per forward. Either way the tokens are identical.
    #[test]
    fn spec_decode_matches_greedy() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let prompt = vec![1u32, 5, 3, 9];
        let max_new = 20usize;

        let mut e_ref = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, false, false);
        let greedy = e_ref.generate_greedy(&[prompt.clone()], max_new, None)[0].clone();
        let full: Vec<u32> = prompt.iter().copied().chain(greedy.iter().copied()).collect();

        // Oracle draft: proposes the true continuation → all accepted.
        let mut e1 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 32, false, false);
        let (out_oracle, fwd_oracle) = e1.spec_decode(&prompt, max_new, 4, |ctx, want| {
            (0..want as usize).map(|i| full.get(ctx.len() + i).copied().unwrap_or(0)).collect()
        });
        // Bad draft: always proposes token 0 → mostly rejected.
        let mut e2 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 4, 64, 1, 8, 32, false, false);
        let (out_bad, fwd_bad) = e2.spec_decode(&prompt, max_new, 4, |_ctx, want| vec![0u32; want as usize]);

        println!("spec decode: greedy={max_new} tokens | oracle-draft {fwd_oracle} target-forwards | bad-draft {fwd_bad} forwards");
        assert_eq!(out_oracle, greedy, "spec (oracle draft) != greedy");
        assert_eq!(out_bad, greedy, "spec (bad draft) != greedy");
        assert!(fwd_oracle < max_new, "oracle draft should cut target forwards ({fwd_oracle} vs {max_new})");
        assert!(fwd_bad >= fwd_oracle, "bad draft should need more forwards");
    }

    /// tts multi-stream: N Talker streams (embedding inputs) decoded together on
    /// the shared paged pool must match each stream decoded alone — bit-for-bit.
    /// (The Talker is the same Qwen3 block, so the tiny config stands in for it.)
    #[test]
    fn tts_multistream_embed_matches_per_stream() {
        let cfg = QwenConfig::tiny();
        let map = tiny_weights(&cfg);
        let d = cfg.d_model as usize;
        let (n_streams, steps) = (3usize, 5usize);
        let mut rng = Rng::new(42);
        let embs: Vec<Vec<Vec<f32>>> = (0..n_streams)
            .map(|_| (0..steps).map(|_| (0..d).map(|_| rng.next_gaussian() as f32).collect()).collect())
            .collect();

        // Batched: all streams advance together each step.
        let mut e = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, n_streams as u32, 8, 4, false, false);
        let mut tables: Vec<BlockTable> = (0..n_streams).map(|_| BlockTable::new()).collect();
        let mut batched: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
        for s in 0..steps {
            let flat: Vec<f32> = (0..n_streams).flat_map(|i| embs[i][s].clone()).collect();
            let mut refs: Vec<&mut BlockTable> = tables.iter_mut().collect();
            let h = e.forward_batched_embed(&mut refs, &flat);
            for (i, b) in batched.iter_mut().enumerate() {
                b.push(h[i * d..(i + 1) * d].to_vec());
            }
        }

        // Per-stream reference.
        let mut worst = 0f32;
        for (i, se) in embs.iter().enumerate() {
            let mut e1 = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 4, 64, 1, 8, 4, false, false);
            let mut t = BlockTable::new();
            for (s, emb) in se.iter().enumerate() {
                let mut refs = [&mut t];
                let h = e1.forward_batched_embed(&mut refs, emb);
                worst = worst.max(h.iter().zip(&batched[i][s]).fold(0f32, |m, (a, b)| m.max((a - b).abs())));
            }
        }
        println!("tts multi-stream (embed) vs per-stream: worst maxabs = {worst:e}");
        assert!(worst < 1e-6, "batched embed decode != per-stream: {worst}");
    }

    fn medium_cfg() -> QwenConfig {
        let mut c = QwenConfig::tiny();
        c.n_layers = 8;
        c.d_model = 256;
        c.head_dim = 64;
        c.n_heads = 8;
        c.n_kv_heads = 4;
        c.d_ff = 1024;
        c.vocab = 256;
        c
    }

    /// Throughput: N concurrent requests served with continuous batching vs run one
    /// at a time. Batched decode should give higher aggregate tokens/sec.
    ///   cargo test -p brain-qwen --lib serve_throughput -- --ignored --nocapture
    #[test]
    #[ignore]
    fn serve_throughput() {
        let cfg = medium_cfg();
        let (dm, nl) = (cfg.d_model, cfg.n_layers);
        let map = tiny_weights(&cfg);
        let n_req = 8usize;
        let max_new = 48usize;
        let prompts: Vec<Vec<u32>> = (0..n_req).map(|i| vec![(i as u32 % 200) + 1, 5, 3, 9, 2]).collect();

        // Sequential: one request at a time (fresh reuse of one engine's pool).
        let mut eng_seq = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg.clone(), &map, 16, 512, n_req as u32, 16, 32, false, false);
        let t0 = std::time::Instant::now();
        for p in &prompts {
            eng_seq.generate_greedy(&[p.clone()], max_new, None);
        }
        let seq_s = t0.elapsed().as_secs_f64();

        // Continuous batching: all requests admitted + decoded together.
        let eng = Engine::from_map_with_gpu(gpu_core::testgpu::dev(PIPELINES), cfg, &map, 16, 512, n_req as u32, 16, 32, false, false);
        let mut sched = Scheduler::new(eng, n_req);
        for p in &prompts {
            sched.submit(Request { prompt: p.clone(), max_new, eos: None });
        }
        let t1 = std::time::Instant::now();
        let out = sched.run();
        let batch_s = t1.elapsed().as_secs_f64();

        let total_tokens = (n_req * max_new) as f64;
        assert_eq!(out.len(), n_req);
        println!(
            "serve throughput ({n_req} reqs x {max_new} tok, d{dm} L{nl}): sequential {:.1} tok/s ({seq_s:.2}s) | continuous-batched {:.1} tok/s ({batch_s:.2}s) | {:.1}x",
            total_tokens / seq_s,
            total_tokens / batch_s,
            seq_s / batch_s,
        );
    }
}
