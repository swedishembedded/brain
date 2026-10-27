// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in continuous
// batching of mixture-of-experts decoders then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The paged, batched serving engine of the DeepSeek-V2 MHA decoder: a
//! [`model::serve::PagedDecoder`], so `model::serve::Scheduler` gives it
//! admission, continuous batching, chunked prefill, cancellation and the
//! step timeline without any scheduling code here.
//!
//! # One weight pass per step for every running sequence
//!
//! A decode step carries one row per running sequence. The projections are
//! `[rows, d] x [d, *]` GEMMs, the attention is one paged batched dispatch set
//! (`model::block`'s batched decode over a block-table-addressed pool), the
//! router and the routed experts run over all rows at once, and the head is one
//! `[rows, d] x [d, vocab]` GEMM followed by a device argmax or top-k. A
//! sequence joins or leaves between steps; nothing is padded to the longest
//! sequence.
//!
//! # Precision
//!
//! [`Precision::Reference`] is fp32 everywhere on the WGSL kernels: the oracle
//! the fast tier is gated against, and the arithmetic of the single-sequence
//! decoder (`DeepseekV2`) token for token. [`Precision::Fast`] runs the
//! GEMM-heavy work on tensor cores: the routed experts - 90% of the weights -
//! as int8 banks through `model::moe_i8` (gather GEMV at decode, grouped int8
//! tensor-core GEMM at prefill), and every dense linear (attention, dense MLP,
//! shared expert, head) as bf16 through `model::ops`, which takes the vendor
//! library's tensor-core GEMM where the device has it. The router stays fp32:
//! it makes a hard top-k decision.
//!
//! # Prefill
//!
//! A prompt is read in chunks of `chunk` rows ([`EngineConfig::chunk`]): one
//! forward per chunk with the chunk's K/V appended through the block table
//! and its queries attending the whole cache so far. Positions of the prompt
//! may be embedding rows from an encoder ([`model::serve::PromptEmbeds`], the
//! vision-language splice): the chunk's token embedding is gathered, then each
//! contiguous run of rows overwrites its positions, exactly as the
//! single-sequence path splices an image block. Only the last chunk of a
//! prompt runs the head, and only on its last row.
//!
//! # Prefix reuse
//!
//! None, and declared so (`prefix_stats` is `(0, 0, 0)`): a document-image
//! prompt starts with the image rows, which differ for every page, so there is
//! no shared prefix to reuse; and with none there is nothing that could serve
//! a cached prefix across differing embedding rows.
//!
//! # What is not here
//!
//! Adapters, speculative decoding, KV offload to host memory: the defaults of
//! [`PagedDecoder`] (none of them applies to the OCR workload).

use checkpoint::TensorSource;
use gpu_core::select::Dtype;
use gpu_core::tape::Tape;
use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use model::block::{self, ChunkPaging};
use model::kv_tier::{self, FlashDecodeShape, KvAppend, KvKernels, KvLayer, KvTier};
use model::moe::{self, GroupedExpertFwdIds, GroupedExpertScratch, MoeIds, MoeShape, RouterKind};
use model::moe_i8::{self, I8Bank, I8Experts, SlotKernels, SlotRouter, SlotScratch};
use model::ops::{Ops, PagedDecodeShape, Weight};
use paramstore::upload::Uploader;
use model::paged::{BlockAllocator, BlockTable};
use model::readout::{self, ReadoutKernels};
use model::serve::{PagedDecoder, PrefillProgress, PromptEmbeds};

use crate::config::DeepseekV2Config;

fn f(x: f32) -> u32 {
    x.to_bits()
}

/// Which arithmetic the engine runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// [`Precision::Fast`] where the device runs the kernels it needs, else [`Precision::Reference`].
    Auto,
    /// fp32 everywhere on the WGSL kernels: the parity oracle.
    Reference,
    /// Tensor cores: int8 routed experts, bf16 dense linears (see the module doc).
    Fast,
}

/// The shape of one engine: how long a sequence may be, how many tokens of KV
/// the pool holds across all of them, and the widths a forward is sized for.
#[derive(Clone, Copy, Debug)]
pub struct EngineConfig {
    /// The most tokens (prompt plus generated) one sequence may hold.
    pub max_seq: u32,
    /// Tokens of K/V the pool holds across every sequence; rounded down to whole blocks.
    pub pool_tokens: u32,
    /// Rows a decode step carries: the most sequences decoded together.
    pub max_batch: u32,
    /// Rows a prefill forward carries.
    pub chunk: u32,
    /// Tokens per KV block.
    pub block_size: u32,
    /// How the pool stores K and V.
    pub kv: KvTier,
    /// Candidates a top-k step can return (the no-repeat filter and sampling read them).
    pub topk: u32,
    /// The arithmetic of the linears and the experts.
    pub precision: Precision,
}

impl EngineConfig {
    /// The shape the OCR decoder is served at: 8192-token sequences (the
    /// checkpoint's own context), 64-token blocks, 512-row prefill chunks.
    pub fn serving(max_batch: u32, pool_tokens: u32) -> EngineConfig {
        EngineConfig { max_seq: 8192, pool_tokens, max_batch, chunk: 512, block_size: 64, kv: KvTier::F32, topk: 64, precision: Precision::Auto }
    }
}

/// The pipelines an engine's [`Gpu`] must be built with: the `model::ops` façade set (every dtype
/// tier of the linears), the attention and MoE kernels both precisions dispatch, and the KV tiers.
pub fn pipelines() -> Vec<(&'static str, &'static str)> {
    let mut v: Vec<(&'static str, &'static str)> = model::ops::kernel_list().to_vec();
    let extras: Vec<(&'static str, &'static str)> = [
        vec![
            ("rmsnorm", kernels::RMSNORM),
            ("rmsnorm_rows", kernels::RMSNORM_ROWS),
            ("rope_at", kernels::ROPE_AT),
            ("rope_paged", kernels::ROPE_PAGED),
            ("region_copy", kernels::REGION_COPY),
            ("add2", kernels::ADD2),
            ("silu_mul", kernels::SILU_MUL),
            ("decode_softmax_batched", kernels::DECODE_SOFTMAX_BATCHED),
            ("concat_split", kernels::CONCAT_SPLIT),
            ("argmax_part", kernels::ARGMAX_PART),
            ("argmax_final", kernels::ARGMAX_FINAL),
            ("topk_extract_step", kernels::TOPK_EXTRACT_STEP),
            // The fp32 grouped experts of the reference tier.
            ("scale_add", kernels::SCALE_ADD),
            ("router_gate", kernels::ROUTER_GATE),
            ("router_topk_compact", kernels::ROUTER_TOPK_COMPACT),
            ("moe_group_counts", kernels::MOE_GROUP_COUNTS),
            ("scan_block", kernels::SCAN_BLOCK),
            ("scan_add", kernels::SCAN_ADD),
            ("moe_group_perm_emit", kernels::MOE_GROUP_PERM_EMIT),
            ("matmul_reg3_grouped", kernels::MATMUL_REG3_GROUPED),
            ("moe_group_combine", kernels::MOE_GROUP_COMBINE),
        ],
        SlotKernels::pipelines(),
    ]
    .concat();
    for e in extras {
        if !v.iter().any(|(n, _)| *n == e.0) {
            v.push(e);
        }
    }
    let tiers = kv_tier::kernel_list(&v);
    v.extend(tiers);
    v
}

/// The routed experts of one MoE layer, in the tier's storage.
enum Experts {
    /// fp32 `[E, n, k]` banks, run by `model::moe::expert_fwd_grouped`.
    F32 { gate: DeviceBuffer, up: DeviceBuffer, down: DeviceBuffer },
    /// Group-scaled int8 banks, run by `model::moe_i8`.
    I8(I8Experts),
}

/// A linear `out = x . w^T`: the weight in the tier's storage and its contraction width.
struct Lin {
    w: Weight,
    k: u32,
}

struct MoeLayer {
    /// `[E, d]` fp32: the router's hard top-k decision is never quantised (see `Engine::exact`).
    router: Lin,
    experts: Experts,
    sh_gate: Lin,
    sh_up: Lin,
    sh_down: Lin,
}

struct DenseMlp {
    gate: Lin,
    up: Lin,
    down: Lin,
}

enum Mlp {
    Dense(Box<DenseMlp>),
    Moe(Box<MoeLayer>),
}

struct Layer {
    ln1: DeviceBuffer,
    ln2: DeviceBuffer,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    mlp: Mlp,
}

/// The kernel indices the engine dispatches directly (everything else goes through `Ops`).
#[derive(Clone, Copy)]
struct Ids {
    embed: usize,
    rmsnorm: usize,
    rmsnorm_rows: usize,
    rope_at: usize,
    rope_paged: usize,
    region_copy: usize,
    add2: usize,
    silu_mul: usize,
    softmax: usize,
    concat_split: usize,
}

impl Ids {
    fn resolve(g: &Gpu) -> Result<Ids, String> {
        let find = |name: &str| g.kernel_index(name).ok_or_else(|| format!("deepseek2 engine: kernel '{name}' is not registered (build the Gpu from deepseek2::serve::pipelines)"));
        Ok(Ids {
            embed: find("embed")?,
            rmsnorm: find("rmsnorm")?,
            rmsnorm_rows: find("rmsnorm_rows")?,
            rope_at: find("rope_at")?,
            rope_paged: find("rope_paged")?,
            region_copy: find("region_copy")?,
            add2: find("add2")?,
            silu_mul: find("silu_mul")?,
            softmax: find("decode_softmax_batched")?,
            concat_split: find("concat_split")?,
        })
    }
}

/// Device buffers one forward writes: the per-row indices, the activations and
/// the attention score slabs. Sized once for the widest forward.
struct Scratch {
    tok_ids: DeviceBuffer,
    positions: DeviceBuffer,
    blocks: DeviceBuffer,
    offsets: DeviceBuffer,
    seq_lens: DeviceBuffer,
    tables: DeviceBuffer,
    /// The residual stream ping-pongs between two buffers: layers run one after another.
    res: [DeviceBuffer; 2],
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    xmid: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    // The MLP sublayer.
    xn2: DeviceBuffer,
    dense_gate: DeviceBuffer,
    dense_up: DeviceBuffer,
    dense_h: DeviceBuffer,
    router_logits: DeviceBuffer,
    gate: DeviceBuffer,
    grouped: Option<GroupedExpertScratch>,
    slots: Option<SlotScratch>,
    sh_gate: DeviceBuffer,
    sh_up: DeviceBuffer,
    sh_h: DeviceBuffer,
    sh_out: DeviceBuffer,
    moe_acc: DeviceBuffer,
    mlp_out: DeviceBuffer,
    last_row: DeviceBuffer,
    xn_final: DeviceBuffer,
    logits: DeviceBuffer,
    /// The prompt chunk's embedding rows, laid out like the chunk's residual.
    emb_stage: DeviceBuffer,
}

/// The paged batched serving engine. See the module doc.
pub struct Engine {
    /// Recorded decode steps by row count, least recently used first. Declared before `gpu`: a recording
    /// pins buffers that must be released before the device.
    tapes: Vec<(u32, Tape)>,
    decode_tapes: bool,
    tape_stats: TapeStats,
    gpu: Gpu,
    /// The linears' tier: the vendor library's tensor-core GEMM for a bf16 weight in the fast tier, fp32 in the reference one.
    ops: Ops,
    /// Always fp32 arithmetic (never TF32): the router, whose top-k is a hard decision a rounded logit can flip.
    exact: Ops,
    cfg: DeepseekV2Config,
    ec: EngineConfig,
    fast: bool,
    ids: Ids,
    tok: DeviceBuffer,
    norm: DeviceBuffer,
    head: Lin,
    layers: Vec<Layer>,
    kvk: KvKernels,
    kv: Vec<KvLayer>,
    alloc: BlockAllocator,
    max_bt: u32,
    sc: Scratch,
    ro: ReadoutKernels,
    slot_kernels: Option<SlotKernels>,
    grouped_ids: GroupedExpertFwdIds,
    moe_ids: MoeIds,
}

/// Recorded decode steps kept: one per row count up to this many, least recently used dropped first.
/// A recording pins the activation temporaries of a whole step, so the cache is bounded.
const MAX_DECODE_TAPES: usize = 128;

/// How often a decode step was replayed from a recording or had to be recorded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TapeStats {
    pub replays: u64,
    pub recordings: u64,
}

fn u32_buf(g: &Gpu, label: &str, n: u64) -> DeviceBuffer {
    g.buffer(label, n * 4, BufUsage::STORAGE | BufUsage::COPY_DST)
}

impl Engine {
    /// Build the engine on `gpu` (made from [`pipelines`]) from a checkpoint's decoder tensors.
    pub fn new(gpu: Gpu, cfg: DeepseekV2Config, src: &dyn TensorSource, ec: EngineConfig) -> Result<Engine, String> {
        let c = cfg.clone();
        let (d, nh, hd, vocab) = (c.d_model(), c.n_heads(), c.head_dim(), c.vocab());
        if ec.max_seq == 0 || ec.max_batch == 0 || ec.chunk == 0 || ec.block_size == 0 || ec.topk == 0 {
            return Err(format!("deepseek2 engine: every size must be positive: {ec:?}"));
        }
        let num_blocks = ec.pool_tokens / ec.block_size;
        if num_blocks == 0 {
            return Err(format!("deepseek2 engine: a pool of {} tokens holds no {}-token block", ec.pool_tokens, ec.block_size));
        }
        if !KvTier::fits_addressing(num_blocks as u64 * ec.block_size as u64, d as u64) {
            return Err(format!("deepseek2 engine: a pool of {} tokens is beyond what one K/V plane addresses", ec.pool_tokens));
        }
        let g = &gpu;
        let ids = Ids::resolve(g)?;
        let mut ops = Ops::new(g.share_or_new(&pipelines()))?;
        let mut exact = Ops::new(g.share_or_new(&pipelines()))?;
        exact.set_gemm_precision(gpu_core::provider::library::GemmPrecision::Fp32Reference);
        // A step allocates its activation temporaries (the bf16 operand casts, the uniform slots) and frees
        // them again; on CUDA a driver free waits for the whole device, so each step would stall on it. Held
        // freed blocks make a free a bookkeeping entry and the next step reuses them.
        for handle in [g, ops.gpu(), exact.gpu()] {
            handle.hold_freed_blocks(true);
        }
        let int8_shaped = d.is_multiple_of(32) && c.moe_ff().is_multiple_of(32);
        let fast = match ec.precision {
            Precision::Reference => false,
            Precision::Fast => true,
            Precision::Auto => g.caps().workgroup_reductions && int8_shaped,
        };
        if fast && !int8_shaped {
            return Err(format!("deepseek2 engine: the fast tier needs d_model {d} and moe_ff {} to be multiples of 32", c.moe_ff()));
        }
        if !fast {
            // The oracle's arithmetic is fp32, not TF32.
            ops.set_gemm_precision(gpu_core::provider::library::GemmPrecision::Fp32Reference);
        }
        let dense = if fast { Dtype::BF16 } else { Dtype::F32 };
        let linear = |name: &str, n: u32, k: u32| -> Result<Lin, String> { Ok(Lin { w: Weight::upload_from(&ops, src, name, n as usize, k as usize, dense)?, k }) };
        let mut up = Uploader::new(g);
        let mut upload = |name: &str, numel: usize| -> Result<DeviceBuffer, String> {
            let buf = up.tensor(src, name, numel)?;
            up.maybe_drain(&buf);
            Ok(buf)
        };
        let tok = upload("tok.weight", vocab as usize * d as usize)?;
        let norm = upload("norm.weight", d as usize)?;
        let head = linear(c.head_weight(), vocab, d)?;
        let (e, ff, top_k) = (c.n_experts(), c.moe_ff(), c.top_k());
        let mut layers = Vec::with_capacity(c.n_layers() as usize);
        for l in 0..c.n_layers() {
            let p = |name: &str| format!("blocks.{l}.{name}");
            let mlp = if c.is_moe_layer(l) {
                let bank = |proj: &str, n: u32, k: u32| -> Result<I8Bank, String> {
                    let name = p(&format!("mlp.experts.{proj}.weight"));
                    I8Bank::from_source(g, src, &name, e as usize, n as usize, k as usize).ok_or_else(|| format!("deepseek2 engine: the checkpoint has no tensor '{name}'"))
                };
                let experts = if fast {
                    Experts::I8(I8Experts { gate: bank("gate", ff, d)?, up: bank("up", ff, d)?, down: bank("down", d, ff)? })
                } else {
                    Experts::F32 {
                        gate: upload(&p("mlp.experts.gate.weight"), (e * ff * d) as usize)?,
                        up: upload(&p("mlp.experts.up.weight"), (e * ff * d) as usize)?,
                        down: upload(&p("mlp.experts.down.weight"), (e * d * ff) as usize)?,
                    }
                };
                let sf = c.shared_ff();
                Mlp::Moe(Box::new(MoeLayer {
                    router: Lin { w: Weight::upload_from(&exact, src, &p("mlp.router.weight"), e as usize, d as usize, Dtype::F32)?, k: d },
                    experts,
                    sh_gate: linear(&p("mlp.shared.gate.weight"), sf, d)?,
                    sh_up: linear(&p("mlp.shared.up.weight"), sf, d)?,
                    sh_down: linear(&p("mlp.shared.down.weight"), d, sf)?,
                }))
            } else {
                let df = c.ffn_hidden();
                Mlp::Dense(Box::new(DenseMlp { gate: linear(&p("mlp.gate.weight"), df, d)?, up: linear(&p("mlp.up.weight"), df, d)?, down: linear(&p("mlp.down.weight"), d, df)? }))
            };
            layers.push(Layer {
                ln1: upload(&p("ln1.weight"), d as usize)?,
                ln2: upload(&p("ln2.weight"), d as usize)?,
                q: linear(&p("self_attn.q_proj.weight"), d, d)?,
                k: linear(&p("self_attn.k_proj.weight"), d, d)?,
                v: linear(&p("self_attn.v_proj.weight"), d, d)?,
                o: linear(&p("self_attn.o_proj.weight"), d, d)?,
                mlp,
            });
        }

        let max_bt = ec.max_seq.div_ceil(ec.block_size);
        // The reference tier is fp32 throughout: its prefill attention is the portable triad, not the fp16 tensor-core kernel.
        let kvk = KvKernels::resolve(g, ec.kv)?;
        let kvk = if fast { kvk } else { kvk.portable_prefill() };
        let rows = ec.max_batch.max(ec.chunk);
        let st = |n: u64| g.storage(n);
        let (r, du) = (rows as u64, d as u64);
        let score_elems = r * nh as u64 * ec.max_seq as u64;
        let (df, sf) = (c.ffn_hidden() as u64, c.shared_ff() as u64);
        let mshape = MoeShape { rows, d_model: d, moe_ff: ff, n_experts: e, top_k };
        let sc = Scratch {
            tok_ids: u32_buf(g, "ds2_tok_ids", r),
            positions: u32_buf(g, "ds2_positions", r),
            blocks: u32_buf(g, "ds2_blocks", r),
            offsets: u32_buf(g, "ds2_offsets", r),
            seq_lens: u32_buf(g, "ds2_seq_lens", r),
            tables: u32_buf(g, "ds2_tables", r * max_bt as u64),
            res: [st(r * du), st(r * du)],
            xn1: st(r * du),
            q: st(r * du),
            k: st(r * du),
            v: st(r * du),
            ctx: st(r * du),
            proj: st(r * du),
            xmid: st(r * du),
            scores: st(score_elems),
            probs: st(score_elems),
            xn2: st(r * du),
            dense_gate: st(r * df),
            dense_up: st(r * df),
            dense_h: st(r * df),
            router_logits: st(r * e as u64),
            gate: st(if fast { 1 } else { r * e as u64 }),
            grouped: (!fast).then(|| GroupedExpertScratch::new(g, &mshape)),
            slots: fast.then(|| SlotScratch::new(g, rows, d, ff, e, top_k)),
            sh_gate: st(r * sf),
            sh_up: st(r * sf),
            sh_h: st(r * sf),
            sh_out: st(r * du),
            moe_acc: st(r * du),
            mlp_out: st(r * du),
            last_row: st(du),
            xn_final: st(ec.max_batch as u64 * du),
            logits: st(ec.max_batch as u64 * vocab as u64),
            emb_stage: st(ec.chunk as u64 * du),
        };
        let pool_rows = num_blocks as u64 * ec.block_size as u64;
        let kv = (0..c.n_layers()).map(|_| KvLayer::new(g, ec.kv, pool_rows, d as u64, hd as u64)).collect();
        let find = |name: &str| g.kernel_index(name).ok_or_else(|| format!("kernel '{name}' is not registered"));
        let grouped_ids = GroupedExpertFwdIds {
            router_topk_compact: find("router_topk_compact")?,
            group_counts: find("moe_group_counts")?,
            scan_block: find("scan_block")?,
            scan_add: find("scan_add")?,
            perm_emit: find("moe_group_perm_emit")?,
            gather: ids.embed,
            gemm_grouped: find("matmul_reg3_grouped")?,
            silu_mul: ids.silu_mul,
            combine: find("moe_group_combine")?,
        };
        let moe_ids = MoeIds { router_gate: find("router_gate")?, linear_gated: find("moe_linear_gated")?, silu_mul: ids.silu_mul, scale_add: find("scale_add")? };
        let ro = ReadoutKernels::resolve(g)?;
        let slot_kernels = fast.then(|| SlotKernels::resolve(g)).transpose()?;
        let alloc = BlockAllocator::new(num_blocks, ec.block_size);
        Ok(Engine { tapes: Vec::new(), decode_tapes: true, tape_stats: TapeStats::default(), gpu, ops, exact, cfg, ec, fast, ids, tok, norm, head, layers, kvk, kv, alloc, max_bt, sc, ro, slot_kernels, grouped_ids, moe_ids })
    }

    /// Whether this engine runs the tensor-core tier.
    pub fn is_fast(&self) -> bool {
        self.fast
    }

    /// Record decode steps and replay them, or build every step: the same answer at a very different host
    /// cost. Drops what was recorded.
    pub fn set_decode_tapes(&mut self, on: bool) {
        self.tapes.clear();
        self.decode_tapes = on;
    }

    /// Compile what a request runs and record the decode step of every row count up to the batch ceiling, with throwaway sequences that give their blocks back.
    /// Recording a step (and the graph the device freezes it into) costs more than running it, once per distinct row count: a burst
    /// of requests ramps the batch through every size, and without this its first pages wait for each recording in turn. A no-op
    /// where no step is recorded.
    pub fn warm_up(&mut self) {
        // A full prefill chunk first: it compiles the kernels a prompt runs (the first request pays for them otherwise).
        let mut scratch = BlockTable::new();
        let _ = self.prefill_slice(&mut scratch, &vec![0; self.ec.chunk as usize], None, usize::MAX);
        scratch.release(&mut self.alloc);
        if !(self.decode_tapes && self.fused_decode()) {
            return;
        }
        let rows = self.ec.max_batch as usize;
        let mut tables: Vec<BlockTable> = (0..rows).map(|_| BlockTable::new()).collect();
        for b in 1..=rows {
            let mut refs: Vec<&mut BlockTable> = tables[..b].iter_mut().collect();
            self.decode_batch(&mut refs, &vec![0; b]);
        }
        self.gpu.poll_wait();
        for t in tables.iter_mut() {
            t.release(&mut self.alloc);
        }
    }

    /// Replays and recordings of decode steps so far.
    pub fn tape_stats(&self) -> TapeStats {
        self.tape_stats
    }

    /// The device the engine runs on, for its profile and op counters.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// Tokens of K/V the pool holds.
    pub fn pool_tokens(&self) -> u32 {
        self.alloc.num_blocks() * self.ec.block_size
    }

    // ---- building blocks -------------------------------------------------------------------

    /// `out = x . w^T` for `rows` rows through `Ops`, which picks the kernel for the weight's tier and the row
    /// count: the GEMV at decode widths, the register-tiled GEMM above, and for a bf16 weight the vendor
    /// library's tensor cores where the device has them.
    fn linear(&self, s: &mut Vec<Step>, lin: &Lin, x: &DeviceBuffer, out: &DeviceBuffer, rows: u32) {
        self.ops.matmul(s, &lin.w, &self.ops.act_f32(x, 0, rows, lin.k), out, 0);
    }

    fn rms(&self, s: &mut Vec<Step>, x: &DeviceBuffer, w: &DeviceBuffer, out: &DeviceBuffer, rows: u32) {
        let d = self.cfg.d_model();
        let (kind, grid) = block::rms_variant(&self.gpu, self.ids.rmsnorm, Some(self.ids.rmsnorm_rows), rows, d);
        s.push(self.gpu.dispatch(kind, &[x, w, out], &[d, rows, f(self.cfg.rms_eps())], grid));
    }

    /// Block `l`'s MLP sublayer over `n` rows: `res_next = xmid + MLP(RMSNorm(xmid))`, the dense
    /// SwiGLU for the leading blocks and the routed-plus-shared MoE for the rest. The MLP is
    /// row-wise, so it does not care whose rows they are.
    fn mlp(&self, s: &mut Vec<Step>, l: usize, n: u32, res_next: &DeviceBuffer) {
        let (c, g, sc, ids) = (&self.cfg, &self.gpu, &self.sc, &self.ids);
        let (d, e) = (c.d_model(), c.n_experts());
        let layer = &self.layers[l];
        self.rms(s, &sc.xmid, &layer.ln2, &sc.xn2, n);
        match &layer.mlp {
            Mlp::Dense(dense) => {
                let DenseMlp { gate, up, down } = &**dense;
                let df = c.ffn_hidden();
                self.linear(s, gate, &sc.xn2, &sc.dense_gate, n);
                self.linear(s, up, &sc.xn2, &sc.dense_up, n);
                s.push(g.step(ids.silu_mul, &[&sc.dense_gate, &sc.dense_up, &sc.dense_h], &[n * df], n * df));
                self.linear(s, down, &sc.dense_h, &sc.mlp_out, n);
            }
            Mlp::Moe(m) => {
                // The router is fp32 in both tiers: a quantisation-noised logit can flip which experts are chosen.
                self.exact.matmul(s, &m.router.w, &self.exact.act_f32(&sc.xn2, 0, n, d), &sc.router_logits, 0);
                let shape = MoeShape { rows: n, d_model: d, moe_ff: c.moe_ff(), n_experts: e, top_k: c.top_k() };
                match &m.experts {
                    Experts::F32 { gate, up, down } => {
                        let kind = RouterKind::Softmax { aux_coef: 0.0, z_coef: 0.0, norm_topk_prob: c.norm_topk_prob, routed_scaling: c.routed_scaling };
                        s.push(moe::router_fwd_kind(g, &self.moe_ids, kind, &shape, &sc.router_logits, None, &sc.gate, None));
                        let grouped = sc.grouped.as_ref().expect("the reference tier has grouped scratch");
                        s.extend(moe::expert_fwd_grouped(g, &self.grouped_ids, &shape, &sc.xn2, &sc.gate, gate, up, down, grouped, &sc.moe_acc));
                    }
                    Experts::I8(banks) => {
                        let k = self.slot_kernels.as_ref().expect("the fast tier has slot kernels");
                        let slots = sc.slots.as_ref().expect("the fast tier has slot scratch");
                        let router = SlotRouter { norm: c.norm_topk_prob, scale: c.routed_scaling };
                        let ss = moe_i8::SlotShape { rows: n, d_model: d, moe_ff: c.moe_ff(), n_experts: e, top_k: c.top_k() };
                        s.extend(moe_i8::expert_fwd_i8_slots(g, k, &ss, router, &sc.xn2, &sc.router_logits, banks, slots, moe_i8::GROUPED_MIN_ROWS, &sc.moe_acc));
                    }
                }
                // The shared experts, fused into one SwiGLU, are summed unweighted.
                let sf = c.shared_ff();
                self.linear(s, &m.sh_gate, &sc.xn2, &sc.sh_gate, n);
                self.linear(s, &m.sh_up, &sc.xn2, &sc.sh_up, n);
                s.push(g.step(ids.silu_mul, &[&sc.sh_gate, &sc.sh_up, &sc.sh_h], &[n * sf], n * sf));
                self.linear(s, &m.sh_down, &sc.sh_h, &sc.sh_out, n);
                s.push(g.step(ids.add2, &[&sc.moe_acc, &sc.sh_out, &sc.mlp_out], &[n * d], n * d));
            }
        }
        s.push(g.step(ids.add2, &[&sc.xmid, &sc.mlp_out, res_next], &[n * d], n * d));
    }

    /// Dispatch the layer stack over the `n` rows already staged in `sc.res[0]`
    /// and return which residual buffer holds the output.
    ///
    /// `rope` rotates a q or k buffer, `attend` records one layer's attention over `q`/`k`/`v` (K/V
    /// append included) into `ctx`; everything else is the same for a decode step (one row per
    /// sequence) and a prefill chunk (one sequence's rows).
    fn layers(&self, s: &mut Vec<Step>, n: u32, rope: &dyn Fn(&Engine, &mut Vec<Step>, &DeviceBuffer), attend: &dyn Fn(&Engine, &mut Vec<Step>, usize)) -> usize {
        let (g, sc) = (&self.gpu, &self.sc);
        let d = self.cfg.d_model();
        let mut cur = 0usize;
        for (l, layer) in self.layers.iter().enumerate() {
            self.rms(s, &sc.res[cur], &layer.ln1, &sc.xn1, n);
            self.linear(s, &layer.q, &sc.xn1, &sc.q, n);
            self.linear(s, &layer.k, &sc.xn1, &sc.k, n);
            self.linear(s, &layer.v, &sc.xn1, &sc.v, n);
            rope(self, s, &sc.q);
            rope(self, s, &sc.k);
            attend(self, s, l);
            self.linear(s, &layer.o, &sc.ctx, &sc.proj, n);
            s.push(g.step(self.ids.add2, &[&sc.res[cur], &sc.proj, &sc.xmid], &[n * d], n * d));
            self.mlp(s, l, n, &sc.res[1 - cur]);
            cur = 1 - cur;
        }
        cur
    }

    /// Final norm and head over `rows` rows of `from` into `logits` (`[rows, vocab]`).
    fn head(&self, s: &mut Vec<Step>, from: &DeviceBuffer, rows: u32) {
        self.rms(s, from, &self.norm, &self.sc.xn_final, rows);
        self.linear(s, &self.head, &self.sc.xn_final, &self.sc.logits, rows);
    }

    /// One decode step: append each sequence's next token (`inputs[i]`) and leave every
    /// row's logits in `sc.logits` (`[rows, vocab]`).
    fn decode_batch(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) {
        let b = tables.len() as u32;
        assert!(b >= 1 && b <= self.ec.max_batch, "deepseek2 decode: {b} rows exceeds the {}-row batch", self.ec.max_batch);
        assert_eq!(tables.len(), inputs.len(), "deepseek2 decode: tables/inputs length mismatch");
        let bt = self.max_bt as usize;
        let (mut blocks, mut offsets, mut positions, mut seq_lens) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut host_tables = vec![0u32; b as usize * bt];
        for (i, t) in tables.iter_mut().enumerate() {
            let (block, offset) = t.append(&mut self.alloc).expect("the scheduler let a decode step run past the pool");
            blocks.push(block);
            offsets.push(offset);
            positions.push(t.len() - 1);
            seq_lens.push(t.len());
            host_tables[i * bt..i * bt + t.blocks().len()].copy_from_slice(t.blocks());
        }
        let cap = *seq_lens.iter().max().expect("a non-empty batch");
        let (g, sc) = (&self.gpu, &self.sc);
        g.write(&sc.tok_ids, inputs);
        g.write(&sc.positions, &positions);
        g.write(&sc.blocks, &blocks);
        g.write(&sc.offsets, &offsets);
        g.write(&sc.seq_lens, &seq_lens);
        g.write(&sc.tables, &host_tables);
        let fused = self.fused_decode();
        // The fused step's every parameter is fixed by the row count: which sequences the rows are, their
        // positions and block tables are data it reads from the metadata buffers written above. Such a step is
        // recorded once per row count and replayed, so the host does not rebuild its ~500 dispatches per token.
        if !(self.decode_tapes && fused) {
            let s = self.decode_steps(b, cap);
            self.gpu.submit(&[], &s);
            return;
        }
        let at = match self.tapes.iter().position(|(rows, _)| *rows == b) {
            Some(at) => at,
            None => {
                self.gpu.begin_tape();
                let s = self.decode_steps(b, cap);
                self.gpu.submit(&[], &s);
                let tape = self.gpu.end_tape();
                if self.tapes.len() >= MAX_DECODE_TAPES {
                    self.tapes.remove(0);
                }
                self.tapes.push((b, tape));
                self.tape_stats.recordings += 1;
                self.tapes.len() - 1
            }
        };
        let hit = self.tapes.remove(at);
        self.gpu.replay_tape(&hit.1);
        self.tapes.push(hit);
        self.tape_stats.replays += 1;
    }

    /// Whether decode attention takes the fused split-key kernel (head_dim 128, fast tier).
    fn fused_decode(&self) -> bool {
        self.fast && self.kvk.flash_decode_available(&self.gpu, self.cfg.head_dim(), 1)
    }

    /// The dispatches of one decode step over `b` rows whose longest sequence is `cap` keys.
    fn decode_steps(&self, b: u32, cap: u32) -> Vec<Step> {
        let (g, sc, c) = (&self.gpu, &self.sc, &self.cfg);
        let (d, nh, hd) = (c.d_model(), c.n_heads(), c.head_dim());
        let theta = c.rope_theta();
        let mut s: Vec<Step> = Vec::new();
        s.push(g.step(self.ids.embed, &[&sc.tok_ids, &self.tok, &sc.res[0]], &[d, b], b * d));
        let rope = |e: &Engine, s: &mut Vec<Step>, buf: &DeviceBuffer| {
            s.push(e.gpu.step(e.ids.rope_paged, &[buf, &e.sc.positions], &[b, nh, hd, d, f(theta)], b * nh * (hd / 2)));
        };
        let shape = PagedDecodeShape { batch: b, n_heads: nh, group: 1, head_dim: hd, block_size: self.ec.block_size, kv_stride: d, cap, max_bt: self.max_bt, scale: 1.0 / (hd as f32).sqrt() };
        // The fast tier reads the keys with the split-key fused decode, where the triad's softmax is one
        // thread per (sequence, head) walking every key: its step time grows with the context several times
        // faster than the bytes it reads. The splits are pinned to the configured context, so a row's
        // attention does not depend on which other sequences share its step.
        let fused = self.fused_decode();
        let flash = FlashDecodeShape { batch: b, n_heads: nh, n_kv_heads: nh, head_dim: hd, block_size: self.ec.block_size, max_bt: self.max_bt, cap: self.ec.max_seq, pin_splits: true };
        let attend = |e: &Engine, s: &mut Vec<Step>, l: usize| {
            let (g, sc) = (&e.gpu, &e.sc);
            let append = KvAppend { batch: b, kv_stride: d, block_size: e.ec.block_size, head_dim: hd };
            s.push(e.kvk.append(g, &e.kv[l].k, &sc.k, &sc.blocks, &sc.offsets, append));
            s.push(e.kvk.append(g, &e.kv[l].v, &sc.v, &sc.blocks, &sc.offsets, append));
            if fused {
                s.extend(e.kvk.flash_decode(g, &sc.q, &e.kv[l].k, &e.kv[l].v, &sc.tables, &sc.seq_lens, &sc.ctx, flash));
            } else {
                s.push(e.kvk.scores(g, &sc.q, &e.kv[l].k, &sc.tables, &sc.seq_lens, &sc.scores, shape));
                s.push(g.step(e.ids.softmax, &[&sc.scores, &sc.seq_lens, &sc.probs], &[b, nh, cap], b * nh));
                s.push(e.kvk.apply(g, &sc.probs, &e.kv[l].v, &sc.tables, &sc.seq_lens, &sc.ctx, shape));
            }
        };
        let out = self.layers(&mut s, b, &rope, &attend);
        self.head(&mut s, &self.sc.res[out], b);
        s
    }

    /// Read the next slice of `prompt` (at most `budget` rows) into `table`; when it is
    /// the last, return the final-normed hidden state of the prompt's last row.
    fn prefill_slice(&mut self, table: &mut BlockTable, prompt: &[u32], embeds: Option<&PromptEmbeds>, budget: usize) -> PrefillProgress {
        let start = table.len();
        let total = prompt.len() as u32;
        assert!(start < total, "deepseek2 prefill: the prompt is already in");
        let n = (budget.min(self.ec.chunk as usize) as u32).min(total - start).max(1);
        table.reserve(n, &mut self.alloc).expect("the scheduler admitted a prompt the pool cannot hold");
        let (bs, bt) = (self.ec.block_size, self.max_bt as usize);
        let (mut blocks, mut offsets, mut seq_lens) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            let pos = start + i;
            blocks.push(table.blocks()[(pos / bs) as usize]);
            offsets.push(pos % bs);
            seq_lens.push(pos + 1);
        }
        let mut host_tables = vec![0u32; n as usize * bt];
        for i in 0..n as usize {
            host_tables[i * bt..i * bt + table.blocks().len()].copy_from_slice(table.blocks());
        }
        let (g, sc, c) = (&self.gpu, &self.sc, &self.cfg);
        g.write(&sc.tok_ids, &prompt[start as usize..(start + n) as usize]);
        g.write(&sc.blocks, &blocks);
        g.write(&sc.offsets, &offsets);
        g.write(&sc.seq_lens, &seq_lens);
        g.write(&sc.tables, &host_tables);
        let (d, nh, hd) = (c.d_model(), c.n_heads(), c.head_dim());
        let theta = c.rope_theta();
        let mut s: Vec<Step> = Vec::new();
        s.push(g.step(self.ids.embed, &[&sc.tok_ids, &self.tok, &sc.res[0]], &[d, n], n * d));
        // Each contiguous run of embedding rows overwrites its positions of the token embedding.
        if let Some(e) = embeds {
            let mut run: Option<(u32, u32)> = None; // (first row in chunk, rows)
            let mut runs: Vec<(u32, u32)> = Vec::new();
            for (pos, row) in e.rows_in(start, start + n) {
                g.write_f32_at(&sc.emb_stage, ((pos - start) * d) as u64, row);
                run = match run {
                    Some((r0, m)) if r0 + m == pos - start => Some((r0, m + 1)),
                    other => {
                        runs.extend(other);
                        Some((pos - start, 1))
                    }
                };
            }
            runs.extend(run);
            for (r0, m) in runs {
                s.push(g.step(self.ids.region_copy, &[&sc.emb_stage, &sc.res[0]], &[1, m * d, m * d, r0 * d], m * d));
            }
        }
        let rope = |e: &Engine, s: &mut Vec<Step>, buf: &DeviceBuffer| {
            s.push(e.gpu.step(e.ids.rope_at, &[buf], &[n, nh, hd, d, 0, start, f(theta)], n * nh * (hd / 2)));
        };
        let attend = |e: &Engine, s: &mut Vec<Step>, l: usize| {
            let (g, sc) = (&e.gpu, &e.sc);
            let paging = ChunkPaging { block_size: e.ec.block_size, max_bt: e.max_bt, blocks: &sc.blocks, offsets: &sc.offsets, tables: &sc.tables };
            s.extend(block::gqa_chunk_step_kv_paged(g, &e.kvk, e.ids.softmax, nh, nh, hd, start, n, &sc.q, &sc.k, &sc.v, &e.kv[l], &paging, &sc.seq_lens, &sc.ctx));
        };
        let out = self.layers(&mut s, n, &rope, &attend);
        if start + n < total {
            self.gpu.submit(&[], &s);
            return PrefillProgress::Partial { consumed: n as usize };
        }
        // The prompt's last row, through the final norm; the head runs when the first token is picked.
        s.push(g.step(self.ids.concat_split, &[&self.sc.res[out], &self.sc.last_row], &[1, n * d, d, (n - 1) * d, 1, 1], d));
        self.rms(&mut s, &self.sc.last_row, &self.norm, &self.sc.xn_final, 1);
        self.gpu.submit(&[], &s);
        PrefillProgress::Done(self.gpu.read(&self.sc.xn_final, d as usize))
    }

    /// Head over one already-normed hidden row into `sc.logits` row 0.
    fn head_of_hidden(&self, hidden: &[f32]) {
        let c = &self.cfg;
        assert_eq!(hidden.len(), c.d_model() as usize, "deepseek2: a hidden row is d_model floats");
        self.gpu.write_f32(&self.sc.xn_final, hidden);
        let mut s: Vec<Step> = Vec::new();
        self.linear(&mut s, &self.head, &self.sc.xn_final, &self.sc.logits, 1);
        self.gpu.submit(&[], &s);
    }
}

impl PagedDecoder for Engine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.alloc
    }
    fn max_prefill_tokens(&self) -> u32 {
        self.ec.chunk
    }
    fn max_batch(&self) -> Option<usize> {
        Some(self.ec.max_batch as usize)
    }
    fn free_blocks(&self) -> u32 {
        self.alloc.free_blocks()
    }
    fn max_seq_len(&self) -> usize {
        self.ec.max_seq as usize
    }
    fn vocab(&self) -> usize {
        self.cfg.vocab() as usize
    }
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.ec.block_size)
    }
    /// There is no prefix cache (see the module doc), so nothing to reclaim.
    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }
    fn release_table(&mut self, t: &mut BlockTable) {
        t.release(&mut self.alloc);
    }
    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        loop {
            if let PrefillProgress::Done(h) = self.prefill_slice(table, prompt, None, usize::MAX) {
                return h;
            }
        }
    }
    fn prefill_interleave_tokens(&self) -> Option<usize> {
        Some(self.ec.chunk as usize)
    }
    fn prefill_chunk(&mut self, table: &mut BlockTable, prompt: &[u32], budget: usize) -> PrefillProgress {
        self.prefill_slice(table, prompt, None, budget)
    }
    fn embed_width(&self) -> Option<usize> {
        Some(self.cfg.d_model() as usize)
    }
    fn prefill_chunk_embeds(&mut self, table: &mut BlockTable, prompt: &[u32], embeds: &PromptEmbeds, budget: usize) -> PrefillProgress {
        self.prefill_slice(table, prompt, Some(embeds), budget)
    }
    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.head_of_hidden(hidden);
        self.gpu.read(&self.sc.logits, self.cfg.vocab() as usize)
    }
    fn admit_greedy(&self, hidden: &[f32]) -> u32 {
        self.head_of_hidden(hidden);
        readout::argmax_rows(&self.gpu, self.ro, &self.sc.logits, 1, self.cfg.vocab())[0]
    }
    fn admit_topk(&self, hidden: &[f32], k: usize) -> Vec<(u32, f32)> {
        self.head_of_hidden(hidden);
        let k = k.clamp(1, self.ec.topk as usize) as u32;
        readout::topk_rows(&self.gpu, self.ro, &self.sc.logits, 1, self.cfg.vocab(), k).remove(0)
    }
    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        if tables.is_empty() {
            return Vec::new();
        }
        self.decode_batch(tables, inputs);
        readout::argmax_rows(&self.gpu, self.ro, &self.sc.logits, tables.len() as u32, self.cfg.vocab())
    }
    /// No on-device multi-step schedule: the window is [`Self::decode_window_capacity`] = 1.
    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        let mut next = inputs.to_vec();
        let mut out = vec![Vec::new(); tables.len()];
        for _ in 0..k.max(1) {
            next = self.forward_batched_greedy(tables, &next);
            for (o, &t) in out.iter_mut().zip(&next) {
                o.push(t);
            }
        }
        out
    }
    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        if tables.is_empty() {
            return Vec::new();
        }
        self.decode_batch(tables, inputs);
        let k = k.clamp(1, self.ec.topk as usize) as u32;
        readout::topk_rows(&self.gpu, self.ro, &self.sc.logits, tables.len() as u32, self.cfg.vocab(), k)
    }
    fn topk_capacity(&self) -> usize {
        self.ec.topk as usize
    }
    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }
    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.gpu.stats()
    }
    fn kv_pool_bytes(&self) -> u64 {
        let rows = self.pool_tokens() as u64;
        let d = self.cfg.d_model() as u64;
        2 * self.cfg.n_layers() as u64 * self.ec.kv.plane_bytes(rows, d, self.cfg.head_dim() as u64)
    }
    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.pool_tokens() as u64
    }
    fn decode_window_capacity(&self) -> usize {
        1
    }
}
