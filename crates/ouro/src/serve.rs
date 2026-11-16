// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching serving for recurrent-depth
// language models for its clients. If your team needs expertise in paged KV
// pools where every recurrent pass owns its own history, or in serving a looped
// model behind one continuous-batching engine, then you can procure our services
// by sending an email to info@swedishembedded.com.

//! The Ouro serving engine.
//!
//! One shared parameter set, `T` attention histories per sequence. The pool is
//! paged exactly like a conventional decoder's, except there are `T` pools per
//! layer (one per recurrent pass) and a sequence owns `T` block tables. The
//! scheduler only sees the first table; its length is the sequence length and
//! the engine allocates the other `T-1` tables alongside it from the same
//! allocator, so admission's block accounting (`passes × blocks_for`) matches
//! what is actually consumed.
//!
//! Prefill runs the whole prompt through every pass at once (matmuls over all its rows, the causal flash kernel)
//! and scatters each pass's K/V into that pass's pool. A decode step carries every running sequence in one batch:
//! each pass's linears run over all of its rows, each row's RoPE takes its own position and its attention reads its
//! own block table, and each row's exit pass is chosen from the gates of all passes before the head runs on the
//! chosen rows only. The step graph of each batch size is recorded once over one set of buffers sized for the
//! largest batch.
//!
//! A sequence's addressable context is the model's positions (or the pool, if smaller), not the pool: the pool is
//! shared by every running sequence, so it can be far larger than any one of them.
//!
//! Not enabled for the looped path yet, by design rather than omission:
//! prefix-cache sharing, host-KV swap and runtime adapters. Each needs a
//! pass-aware identity the shared bookkeeping does not have.

use std::collections::HashMap;

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use ::model::ops::{Ops, Weight};
use ::model::paged::{BlockAllocator, BlockTable};
use ::model::serve::PagedDecoder;
use paramstore::ParamStore;

use crate::config::OuroConfig;
use crate::model;

/// Per-row activations of one layer, `rows` rows wide.
struct Scratch {
    xn1: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    xn1_2: DeviceBuffer,
    xmid: DeviceBuffer,
    xn2: DeviceBuffer,
    gate_pre: DeviceBuffer,
    up: DeviceBuffer,
    mlp_h: DeviceBuffer,
    mlp_out: DeviceBuffer,
    xn2_2: DeviceBuffer,
}

impl Scratch {
    fn new(gpu: &Gpu, cfg: &OuroConfig, rows: u64) -> Scratch {
        let st = |x: u64| gpu.storage(rows * x);
        let (d, ff, hq, hkv) = (cfg.d_model as u64, cfg.d_ff as u64, cfg.q_dim() as u64, cfg.kv_dim() as u64);
        Scratch {
            xn1: st(d),
            q: st(hq),
            k: st(hkv),
            v: st(hkv),
            ctx: st(hq),
            proj: st(d),
            xn1_2: st(d),
            xmid: st(d),
            xn2: st(d),
            gate_pre: st(ff),
            up: st(ff),
            mlp_h: st(ff),
            mlp_out: st(d),
            xn2_2: st(d),
        }
    }
}

struct Row {
    /// Block tables for passes `1..T`; pass 0 is the scheduler's table.
    extra: Vec<BlockTable>,
}

/// One decode step's device buffers, sized for `rows` sequences. The step graph of every batch size up to `rows`
/// reads and writes their first rows.
struct DecodeBufs {
    /// Capacity in sequences; `rows * d_model` is a multiple of 64 words, so each pass's slice of `hs` binds at an
    /// aligned offset.
    rows: u32,
    tokens: DeviceBuffer,
    /// Each row's absolute position (the same in every pass: each pass appends one row per token).
    positions: DeviceBuffer,
    h: [DeviceBuffer; 2],
    /// `[T * rows, d]`: pass `p`'s normed output at rows `p * rows ..`, what the next pass reads and the head
    /// gathers each sequence's exit row from.
    hs: DeviceBuffer,
    /// `[T * rows]` exit gates, the same row order as `hs`.
    gates: DeviceBuffer,
    /// `[rows]` u32: the row of `hs` each sequence's exit pass reads.
    sel: DeviceBuffer,
    hsel: DeviceBuffer,
    logits: DeviceBuffer,
    s: Scratch,
    /// Per pass: each row's new K/V slot (block, offset), block table and cached length.
    blocks: Vec<DeviceBuffer>,
    offsets: Vec<DeviceBuffer>,
    tables: Vec<DeviceBuffer>,
    lens: Vec<DeviceBuffer>,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
}

impl DecodeBufs {
    fn new(gpu: &Gpu, cfg: &OuroConfig, rows: u32, max_bt: u32, cap: u32) -> DecodeBufs {
        let t = cfg.total_ut_steps as u64;
        let (r, d) = (rows as u64, cfg.d_model as u64);
        let index = |name: &str, words: u64| gpu.buffer(name, words * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let per_pass = |name: &str, words: u64| (0..t).map(|_| index(name, words)).collect::<Vec<_>>();
        DecodeBufs {
            rows,
            tokens: index("ouro_tokens", r),
            positions: index("ouro_positions", r),
            h: [gpu.storage(r * d), gpu.storage(r * d)],
            hs: gpu.storage(t * r * d),
            gates: gpu.storage(t * r),
            sel: index("ouro_exit_rows", r),
            hsel: gpu.storage(r * d),
            logits: gpu.storage(r * cfg.vocab as u64),
            s: Scratch::new(gpu, cfg, r),
            blocks: per_pass("ouro_blocks", r),
            offsets: per_pass("ouro_offsets", r),
            tables: per_pass("ouro_tables", r * max_bt as u64),
            lens: per_pass("ouro_lens", r),
            scores: gpu.storage(r * cfg.n_heads as u64 * cap as u64),
            probs: gpu.storage(r * cfg.n_heads as u64 * cap as u64),
        }
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// A prompt's device buffers, sized for the longest prompt a sequence holds and allocated once: a prefill or a
/// scoring pass writes and reads their first rows instead of allocating its own.
struct PromptBufs {
    tokens: DeviceBuffer,
    /// Per pass: each prompt token's K/V slot (block, offset).
    blocks: Vec<DeviceBuffer>,
    offsets: Vec<DeviceBuffer>,
    h: [DeviceBuffer; 2],
    hs: Vec<DeviceBuffer>,
    gates: Vec<DeviceBuffer>,
    s: Scratch,
    lse: DeviceBuffer,
}

impl PromptBufs {
    fn new(gpu: &Gpu, cfg: &OuroConfig, rows: u32) -> PromptBufs {
        let (t, r, d) = (cfg.total_ut_steps as u64, rows as u64, cfg.d_model as u64);
        let index = |name: &str| gpu.buffer(name, r * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        PromptBufs {
            tokens: index("ouro_prompt_tokens"),
            blocks: (0..t).map(|_| index("ouro_prompt_blocks")).collect(),
            offsets: (0..t).map(|_| index("ouro_prompt_offsets")).collect(),
            h: [gpu.storage(r * d), gpu.storage(r * d)],
            hs: (0..t).map(|_| gpu.storage(r * d)).collect(),
            gates: (0..t).map(|_| gpu.storage(r)).collect(),
            s: Scratch::new(gpu, cfg, r),
            lse: gpu.storage(cfg.n_heads as u64 * r),
        }
    }
}

/// The recorded graphs of one batch size: every pass's trunk with the gates, then the gather of the exit rows and
/// the head over them.
struct StepGraph {
    trunk: Vec<Step>,
    gather: Step,
    head: Vec<Step>,
}

pub struct Engine {
    gpu: Gpu,
    ops: Ops,
    cfg: OuroConfig,
    ps: ParamStore,
    /// One allocator per recurrent pass: each pass has its own paged plane,
    /// so a sequence's `T` block tables draw from `T` independent pools of
    /// the same capacity. Sharing one allocator across passes would charge a
    /// sequence `T x` its context against a single pool's block space.
    allocs: Vec<BlockAllocator>,
    block_size: u32,
    /// Tokens of the shared pool, per pass.
    pool_tokens: u32,
    /// Blocks one sequence's table may hold.
    max_bt: u32,
    /// Per-sequence logit-addressable capacity (`max_bt * block_size`).
    cap: u32,
    /// `[pass][layer]` K/V planes over the shared block pool.
    k_pool: Vec<Vec<DeviceBuffer>>,
    v_pool: Vec<Vec<DeviceBuffer>>,
    rows: HashMap<u32, Row>,
    next_row: u32,
    rope_paged: usize,
    flash: Option<::model::block::FlashGqaIds>,
    /// One hidden row and its logits: the head of an admission ([`PagedDecoder::logits`]).
    hrow: DeviceBuffer,
    hrow_logits: DeviceBuffer,
    dec: DecodeBufs,
    prompt: PromptBufs,
    graphs: HashMap<u32, StepGraph>,
}

/// The kernels a serving build compiles: the model's own (its GEMM facade's included) and the per-row RoPE of a
/// batched decode step.
fn serve_pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        let mut v = model::pipelines().to_vec();
        v.push(("rope_paged", kernels::ROPE_PAGED));
        v
    })
}

impl Engine {
    /// Build an engine over a checkpoint directory (HF layout) or a brain
    /// checkpoint file, with `max_tokens` of shared KV capacity per pass,
    /// `block_size` tokens per page and decode steps of up to `max_batch` sequences.
    pub fn open(path: &str, max_tokens: u32, block_size: u32, max_batch: u32) -> Result<Engine, String> {
        let gpu = Gpu::new(serve_pipelines());
        let (cfg, ps) = if std::path::Path::new(path).is_dir() {
            let (cfg, src) = crate::import::open_dir(std::path::Path::new(path))?;
            let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
            let ps = ParamStore::new_with_roles_src(&gpu, roles, &src);
            (cfg, ps)
        } else {
            let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("{path}: {e}"))?;
            let cfg = OuroConfig::from_json(&reader.config())?;
            let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
            let ps = ParamStore::new_with_roles_src(&gpu, roles, &reader);
            (cfg, ps)
        };
        Engine::build(gpu, cfg, ps, max_tokens, block_size, max_batch)
    }

    /// Build from an in-memory weight source (tests use `init::init_weights`).
    pub fn from_source(cfg: OuroConfig, src: &dyn checkpoint::TensorSource, max_tokens: u32, block_size: u32, max_batch: u32) -> Engine {
        let gpu = Gpu::new(serve_pipelines());
        let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        Engine::build(gpu, cfg, ps, max_tokens, block_size, max_batch).expect("ouro: engine build")
    }

    /// The KV pool's bytes for `tokens` of shared capacity: every pass owns a K and a V plane per layer.
    pub fn kv_bytes_per_token(cfg: &OuroConfig) -> u64 {
        cfg.kv_dim() as u64 * 4 * 2 * cfg.n_layers as u64 * cfg.total_ut_steps as u64
    }

    fn build(gpu: Gpu, cfg: OuroConfig, ps: ParamStore, max_tokens: u32, block_size: u32, max_batch: u32) -> Result<Engine, String> {
        if block_size == 0 || max_tokens == 0 || !max_tokens.is_multiple_of(block_size) {
            return Err(format!("ouro: a pool of {max_tokens} tokens in pages of {block_size}: the pool must be a positive multiple of the page"));
        }
        if max_batch == 0 {
            return Err("ouro: a decode step of 0 sequences".to_string());
        }
        let num_blocks = max_tokens / block_size;
        // One sequence addresses at most the model's positions (or the whole pool, when that is smaller).
        let max_bt = cfg.block_size.div_ceil(block_size).clamp(1, num_blocks);
        let cap = max_bt * block_size;
        let allocs = (0..cfg.total_ut_steps).map(|_| BlockAllocator::new(num_blocks, block_size)).collect();
        let hkv = cfg.kv_dim() as u64;
        let planes = || (0..cfg.n_layers).map(|_| gpu.storage(max_tokens as u64 * hkv)).collect::<Vec<_>>();
        let k_pool = (0..cfg.total_ut_steps).map(|_| planes()).collect();
        let v_pool = (0..cfg.total_ut_steps).map(|_| planes()).collect();
        let mut ops = Ops::new(gpu.share())?;
        ops.set_gemm_precision(gpu_core::provider::library::GemmPrecision::Auto);
        let rope_paged = gpu.kernel_index("rope_paged").ok_or("ouro: rope_paged is not registered")?;
        let flash_bwd = |name: &str| gpu.kernel_index(name).ok_or_else(|| format!("ouro: {name} is not registered"));
        let flash = ::model::block::FlashGqaIds { fwd: model::FLASH_GQA_ID, bwd_dq: flash_bwd("flash_attn_causal_gqa_bwd_dq")?, bwd_dkv: flash_bwd("flash_attn_causal_gqa_bwd_dkv")?, native: !ops.tensor_gemm().is_reference_pinned() };
        let flash = ::model::block::FlashGqaIds::supported(&gpu, cfg.head_dim).then_some(flash);
        // Rows rounded up so that `rows * d_model` words is a multiple of 64 (256-byte binding offsets).
        let d = cfg.d_model;
        let rows = max_batch.next_multiple_of(64 / gcd(d, 64));
        let dec = DecodeBufs::new(&gpu, &cfg, rows, max_bt, cap);
        let prompt = PromptBufs::new(&gpu, &cfg, cap);
        let hrow = gpu.storage(d as u64);
        let hrow_logits = gpu.storage(cfg.vocab as u64);
        Ok(Engine { gpu, ops, cfg, ps, allocs, block_size, pool_tokens: max_tokens, max_bt, cap, k_pool, v_pool, rows: HashMap::new(), next_row: 0, rope_paged, flash, hrow, hrow_logits, dec, prompt, graphs: HashMap::new() })
    }

    fn ensure_row(&mut self, table: &mut BlockTable) -> Result<u32, String> {
        if let Some(id) = table.state_row() {
            return Ok(id);
        }
        let id = self.next_row;
        self.next_row += 1;
        table.set_state_row(id);
        let extra = (1..self.cfg.total_ut_steps).map(|_| BlockTable::new()).collect();
        self.rows.insert(id, Row { extra });
        Ok(id)
    }

    /// `y = x Wᵀ` over `m` rows through the GEMM facade, which selects the kernel (the library's tensor-core
    /// product, a GEMV for a few rows, the tiled kernels) for the shape and the device.
    #[allow(clippy::too_many_arguments)]
    fn linear(&self, steps: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
        let weight = Weight::dense_f32(self.ps.w(wname).clone(), n, k);
        self.ops.matmul(steps, &weight, &self.ops.act_f32(x, 0, m, k), out, 0);
    }

    /// The decode graph of `b` rows, recorded once: one new token per row through every pass, appending each row's
    /// K/V at its own position, then the gates of every pass; the gather of the exit rows and the head over them.
    fn record(&self, b: u32) -> StepGraph {
        let cfg = &self.cfg;
        let gpu = &self.gpu;
        let ids = model::kernel_ids();
        let dec_ids = model::decode_batched_ids();
        let z = &self.dec;
        let s = &z.s;
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let (hq, hkv, nh, nkv, hd) = (cfg.q_dim(), cfg.kv_dim(), cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let theta = gpu_core::f(cfg.rope_theta);
        let t = cfg.total_ut_steps;
        // Pass `p`'s rows of `hs`, as a binding range.
        let slice = |p: u32| ((p * z.rows * d) as u64, (b * d) as u64);
        let whole = (0u64, 0u64);

        let mut steps = vec![gpu.step(model::EMBED_ID, &[&z.tokens, self.ps.w("tok.weight"), &z.h[0]], &[d, b], b * d)];
        for p in 0..t {
            // What this pass reads: the embedding for the first, the previous pass's normed output after it.
            let (mut prev, mut prev_range): (&DeviceBuffer, (u64, u64)) = if p == 0 { (&z.h[0], whole) } else { (&z.hs, slice(p - 1)) };
            for l in 0..cfg.n_layers as usize {
                let out: &DeviceBuffer = if p == 0 { &z.h[(l + 1) % 2] } else { &z.h[l % 2] };
                let pfx = |name: &str| format!("blocks.{l}.{name}");
                steps.push(::model::block::rmsnorm_fwd_sliced(gpu, &ids, (prev, prev_range), self.ps.w(&pfx("ln1.weight")), (&s.xn1, whole), d, b, cfg.norm_eps));
                self.linear(&mut steps, &s.xn1, &pfx("attn.wq.weight"), &s.q, b, d, hq);
                self.linear(&mut steps, &s.xn1, &pfx("attn.wk.weight"), &s.k, b, d, hkv);
                self.linear(&mut steps, &s.xn1, &pfx("attn.wv.weight"), &s.v, b, d, hkv);
                steps.push(gpu.step(self.rope_paged, &[&s.q, &z.positions], &[b, nh, hd, hq, theta], b * nh * hd / 2));
                steps.push(gpu.step(self.rope_paged, &[&s.k, &z.positions], &[b, nkv, hd, hkv, theta], b * nkv * hd / 2));
                let pu = p as usize;
                steps.extend(::model::block::gqa_decode_batched_step(
                    gpu,
                    &dec_ids,
                    nh,
                    nkv,
                    hd,
                    b,
                    self.block_size,
                    self.max_bt,
                    self.cap,
                    &s.q,
                    &s.k,
                    &s.v,
                    &self.k_pool[pu][l],
                    &self.v_pool[pu][l],
                    &z.blocks[pu],
                    &z.offsets[pu],
                    &z.tables[pu],
                    &z.lens[pu],
                    &z.scores,
                    &z.probs,
                    &s.ctx,
                ));
                self.linear(&mut steps, &s.ctx, &pfx("attn.wo.weight"), &s.proj, b, hq, d);
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, &s.proj, self.ps.w(&pfx("ln1_2.weight")), &s.xn1_2, d, b, cfg.norm_eps));
                steps.push(gpu.step_sliced(model::ADD2_ID, &[prev, &s.xn1_2, &s.xmid], &[prev_range, whole, whole], &[b * d], b * d));
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, &s.xmid, self.ps.w(&pfx("ln2.weight")), &s.xn2, d, b, cfg.norm_eps));
                self.linear(&mut steps, &s.xn2, &pfx("mlp.gate.weight"), &s.gate_pre, b, d, ff);
                self.linear(&mut steps, &s.xn2, &pfx("mlp.up.weight"), &s.up, b, d, ff);
                steps.push(::model::block::swiglu_fwd(gpu, &ids, &s.gate_pre, &s.up, &s.mlp_h, b * ff));
                self.linear(&mut steps, &s.mlp_h, &pfx("mlp.down.weight"), &s.mlp_out, b, ff, d);
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, &s.mlp_out, self.ps.w(&pfx("ln2_2.weight")), &s.xn2_2, d, b, cfg.norm_eps));
                steps.push(gpu.step(model::ADD2_ID, &[&s.xmid, &s.xn2_2, out], &[b * d], b * d));
                (prev, prev_range) = (out, whole);
            }
            steps.push(::model::block::rmsnorm_fwd_sliced(gpu, &ids, (prev, whole), self.ps.w("norm.weight"), (&z.hs, slice(p)), d, b, cfg.norm_eps));
        }
        // Every pass's gate in one product over all of `hs` (rows past `b` are stale and never read).
        let gate_rows = t * z.rows;
        self.linear(&mut steps, &z.hs, "gate.weight", &z.gates, gate_rows, d, 1);
        steps.push(gpu.step(model::BIAS_ADD_ID, &[&z.gates, self.ps.w("gate.bias")], &[gate_rows, 1], gate_rows));
        let gather = gpu.step(model::EMBED_ID, &[&z.sel, &z.hs, &z.hsel], &[d, b], b * d);
        let mut head = Vec::new();
        self.linear(&mut head, &z.hsel, "lm_head.weight", &z.logits, b, d, cfg.vocab);
        StepGraph { trunk: steps, gather, head }
    }

    /// One token for each of `tables` (`inputs[i]` the token row `i` appends) through every pass; each row's K/V
    /// is appended to each pass's table at its own position, and each row's exit pass is chosen from its gates.
    /// Leaves the exit rows' hidden states in `hsel`; `head` also runs the head over them into `logits`.
    fn step(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], head: bool) {
        let b = tables.len() as u32;
        assert!(b > 0 && b <= self.dec.rows, "ouro: a decode step of {b} rows on buffers of {}", self.dec.rows);
        let t = self.cfg.total_ut_steps as usize;
        let max_bt = self.max_bt as usize;
        let mut positions = Vec::with_capacity(b as usize);
        let mut blocks = vec![vec![0u32; b as usize]; t];
        let mut offsets = vec![vec![0u32; b as usize]; t];
        let mut lens = vec![vec![0u32; b as usize]; t];
        let mut bt = vec![vec![0u32; b as usize * max_bt]; t];
        for (r, table) in tables.iter_mut().enumerate() {
            let table: &mut BlockTable = table;
            let row = self.ensure_row(table).expect("row");
            let pos = table.len();
            positions.push(pos);
            for p in 0..t {
                // Pass 0 appends into the scheduler's table; every other pass into its own, from that pass's own
                // allocator, so each paged plane is charged exactly the sequence's page count.
                let appended = if p == 0 { table.append(&mut self.allocs[0]) } else { self.rows.get_mut(&row).expect("row").extra[p - 1].append(&mut self.allocs[p]) };
                let (blk, off) = appended.expect("out of KV blocks");
                blocks[p][r] = blk;
                offsets[p][r] = off;
                lens[p][r] = pos + 1;
                let owned = if p == 0 { table.blocks() } else { self.rows[&row].extra[p - 1].blocks() };
                bt[p][r * max_bt..r * max_bt + owned.len()].copy_from_slice(owned);
            }
        }
        let z = &self.dec;
        self.gpu.write(&z.tokens, inputs);
        self.gpu.write(&z.positions, &positions);
        for p in 0..t {
            self.gpu.write(&z.blocks[p], &blocks[p]);
            self.gpu.write(&z.offsets[p], &offsets[p]);
            self.gpu.write(&z.lens[p], &lens[p]);
            self.gpu.write(&z.tables[p], &bt[p]);
        }
        if !self.graphs.contains_key(&b) {
            let graph = self.record(b);
            self.graphs.insert(b, graph);
        }
        let graph = &self.graphs[&b];
        self.gpu.submit(&[], &graph.trunk);
        let z = &self.dec;
        let gates = self.gpu.read(&z.gates, t * z.rows as usize);
        let sel: Vec<u32> = (0..b)
            .map(|r| {
                let row_gates: Vec<f64> = (0..t).map(|p| gates[p * z.rows as usize + r as usize] as f64).collect();
                model::exit_pass(&row_gates, self.cfg.early_exit_threshold as f64) as u32 * z.rows + r
            })
            .collect();
        self.gpu.write(&z.sel, &sel);
        let mut tail = vec![graph.gather.clone()];
        if head {
            tail.extend(graph.head.iter().cloned());
        }
        self.gpu.submit(&[], &tail);
    }

    /// Each row's logits after [`Self::step`] with the head.
    fn logit_rows(&self, b: usize) -> Vec<Vec<f32>> {
        let v = self.cfg.vocab as usize;
        let all = self.gpu.read(&self.dec.logits, b * v);
        all.chunks(v).map(<[f32]>::to_vec).collect()
    }

    /// Prefill the whole prompt in one pass per recurrent step ([`Self::trunk`], scattering each pass's K/V into
    /// that pass's paged pool). Returns the last token's selected-pass hidden state (what admission samples from).
    fn prefill_batch(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        let n = prompt.len() as u32;
        assert_eq!(table.len(), 0, "prefill_batch expects a fresh sequence");
        let row = self.ensure_row(table).expect("row");
        let bs = self.block_size;

        // Reserve the chunk in every pass's table, then record each token's
        // (block, offset) slot for the batched scatter.
        table.reserve(n, &mut self.allocs[0]).expect("out of KV blocks");
        let slots = |t: &BlockTable| -> (Vec<u32>, Vec<u32>) { (0..n).map(|i| t.locate(i, bs)).unzip() };
        let mut pass_slots: Vec<(Vec<u32>, Vec<u32>)> = vec![slots(table)];
        for p in 1..self.cfg.total_ut_steps {
            let ex = &mut self.rows.get_mut(&row).expect("row").extra[p as usize - 1];
            ex.reserve(n, &mut self.allocs[p as usize]).expect("out of KV blocks");
            pass_slots.push(slots(ex));
        }
        let (hs, gates) = self.trunk(prompt, Some(&pass_slots));
        let last = n as usize - 1;
        let pass = self.exit_of(&gates, last);
        let d = self.cfg.d_model as usize;
        self.gpu.read(&hs[pass], n as usize * d)[last * d..].to_vec()
    }

    /// The exit pass of row `i` of a [`Self::trunk`] by the CDF rule decode uses.
    fn exit_of(&self, gates: &[Vec<f32>], i: usize) -> usize {
        let row: Vec<f64> = gates.iter().map(|g| g[i] as f64).collect();
        model::exit_pass(&row, self.cfg.early_exit_threshold as f64)
    }

    /// `tokens` as one causal sequence through every pass, in the prompt buffers: matmuls over all its rows, the
    /// causal flash kernel for attention (the materialized chain where the device runs no workgroup reductions).
    /// With `scatter`, each pass's K/V rows go to that pass's `(block, offset)` slots of its paged pool. Returns
    /// every pass's normed output (the first `n` rows of each of `prompt.hs`) and its gates, read back.
    fn trunk(&self, prompt: &[u32], scatter: Option<&[(Vec<u32>, Vec<u32>)]>) -> (&[DeviceBuffer], Vec<Vec<f32>>) {
        let cfg = &self.cfg;
        let n = prompt.len() as u32;
        assert!(n <= self.cap, "ouro: a prompt of {n} tokens past a sequence's {}", self.cap);
        let bs = self.block_size;
        let gpu = &self.gpu;
        let nn = n as u64;
        let z = &self.prompt;
        gpu.write(&z.tokens, prompt);
        if let Some(slots) = scatter {
            for (p, (blocks, offsets)) in slots.iter().enumerate() {
                gpu.write(&z.blocks[p], blocks);
                gpu.write(&z.offsets[p], offsets);
            }
        }
        let (tokens, h, hs, gates, sc, lse) = (&z.tokens, &z.h, &z.hs, &z.gates, &z.s, &z.lse);
        // The flash kernel needs no score slab; the materialized fallback does, sized for this prompt.
        let slab = if self.flash.is_some() { 1 } else { cfg.n_heads as u64 * nn * nn };
        let (scores, probs) = (gpu.storage(slab), gpu.storage(slab));

        let ids = model::kernel_ids();
        let dec = model::decode_batched_ids();
        let (nh, nkv, hd) = (cfg.n_heads, cfg.n_kv_heads, cfg.head_dim);
        let ga = ::model::block::Gqa { b: 1, t: n, n_heads: nh, n_kv_heads: nkv, head_dim: hd };
        let t = cfg.total_ut_steps as usize;
        let mut steps = vec![gpu.step(model::EMBED_ID, &[tokens, self.ps.w("tok.weight"), &h[0]], &[cfg.d_model, n], n * cfg.d_model)];
        for p in 0..t {
            let mut prev: &DeviceBuffer = if p == 0 { &h[0] } else { &hs[p - 1] };
            for l in 0..cfg.n_layers as usize {
                let out: &DeviceBuffer = if p == 0 { &h[(l + 1) % 2] } else { &h[l % 2] };
                let pfx = |name: &str| format!("blocks.{l}.{name}");
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, prev, self.ps.w(&pfx("ln1.weight")), &sc.xn1, cfg.d_model, n, cfg.norm_eps));
                self.linear(&mut steps, &sc.xn1, &pfx("attn.wq.weight"), &sc.q, n, cfg.d_model, cfg.q_dim());
                self.linear(&mut steps, &sc.xn1, &pfx("attn.wk.weight"), &sc.k, n, cfg.d_model, cfg.kv_dim());
                self.linear(&mut steps, &sc.xn1, &pfx("attn.wv.weight"), &sc.v, n, cfg.d_model, cfg.kv_dim());
                steps.push(::model::block::rope_fwd(gpu, &ids, &sc.q, n, nh, hd, cfg.q_dim(), n, cfg.rope_theta));
                steps.push(::model::block::rope_fwd(gpu, &ids, &sc.k, n, nkv, hd, cfg.kv_dim(), n, cfg.rope_theta));
                // Scatter this chunk's K/V into the pass's paged plane.
                if scatter.is_some() {
                    let (blocks, offsets) = (&z.blocks[p], &z.offsets[p]);
                    steps.push(gpu.step(dec.kv_append_batched, &[&sc.k, blocks, offsets, &self.k_pool[p][l]], &[n, cfg.kv_dim(), bs], n * cfg.kv_dim()));
                    steps.push(gpu.step(dec.kv_append_batched, &[&sc.v, blocks, offsets, &self.v_pool[p][l]], &[n, cfg.kv_dim(), bs], n * cfg.kv_dim()));
                }
                match &self.flash {
                    Some(flash) => steps.push(::model::block::flash_gqa_causal_fwd(gpu, flash, &ga, &sc.q, &sc.k, &sc.v, &sc.ctx, lse)),
                    None => steps.extend(::model::block::gqa_fwd(gpu, &ids, &ga, &sc.q, &sc.k, &sc.v, &scores, &probs, &sc.ctx)),
                }
                self.linear(&mut steps, &sc.ctx, &pfx("attn.wo.weight"), &sc.proj, n, cfg.q_dim(), cfg.d_model);
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, &sc.proj, self.ps.w(&pfx("ln1_2.weight")), &sc.xn1_2, cfg.d_model, n, cfg.norm_eps));
                steps.push(gpu.step(model::ADD2_ID, &[prev, &sc.xn1_2, &sc.xmid], &[n * cfg.d_model], n * cfg.d_model));
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, &sc.xmid, self.ps.w(&pfx("ln2.weight")), &sc.xn2, cfg.d_model, n, cfg.norm_eps));
                self.linear(&mut steps, &sc.xn2, &pfx("mlp.gate.weight"), &sc.gate_pre, n, cfg.d_model, cfg.d_ff);
                self.linear(&mut steps, &sc.xn2, &pfx("mlp.up.weight"), &sc.up, n, cfg.d_model, cfg.d_ff);
                steps.push(::model::block::swiglu_fwd(gpu, &ids, &sc.gate_pre, &sc.up, &sc.mlp_h, n * cfg.d_ff));
                self.linear(&mut steps, &sc.mlp_h, &pfx("mlp.down.weight"), &sc.mlp_out, n, cfg.d_ff, cfg.d_model);
                steps.push(::model::block::rmsnorm_fwd(gpu, &ids, &sc.mlp_out, self.ps.w(&pfx("ln2_2.weight")), &sc.xn2_2, cfg.d_model, n, cfg.norm_eps));
                steps.push(gpu.step(model::ADD2_ID, &[&sc.xmid, &sc.xn2_2, out], &[n * cfg.d_model], n * cfg.d_model));
                prev = out;
            }
            steps.push(::model::block::rmsnorm_fwd(gpu, &ids, prev, self.ps.w("norm.weight"), &hs[p], cfg.d_model, n, cfg.norm_eps));
            self.linear(&mut steps, &hs[p], "gate.weight", &gates[p], n, cfg.d_model, 1);
            steps.push(gpu.step(model::BIAS_ADD_ID, &[&gates[p], self.ps.w("gate.bias")], &[n, 1], n));
        }
        gpu.submit(&[], &steps);
        let gates = gates.iter().map(|g| gpu.read(g, n as usize)).collect();
        (hs, gates)
    }

    /// [`PagedDecoder::score_continuation`]: one [`Self::trunk`] over `tokens` (no KV written), each position's
    /// exit row gathered on the host as decode gathers it, the head over the rows that predict a scored token.
    fn score(&self, tokens: &[u32], from: usize, temperature: f32) -> Result<Vec<f32>, String> {
        if from == 0 || from > tokens.len() {
            return Err(format!("ouro: scoring from {from} of {} tokens: the first scored position is 1 to the length", tokens.len()));
        }
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(format!("ouro: scoring at temperature {temperature}: it must be finite and positive"));
        }
        if tokens.len() > self.cap as usize {
            return Err(format!("ouro: scoring {} tokens, more than a sequence holds ({})", tokens.len(), self.cap));
        }
        if from == tokens.len() {
            return Ok(Vec::new());
        }
        let (hs, gates) = self.trunk(&tokens[..tokens.len() - 1], None);
        let (d, v) = (self.cfg.d_model as usize, self.cfg.vocab as usize);
        let n = tokens.len() - 1;
        let per_pass: Vec<Vec<f32>> = hs.iter().map(|h| self.gpu.read(h, n * d)).collect();
        // Position `p - 1`'s exit row predicts token `p`.
        let rows: Vec<f32> = (from - 1..n).flat_map(|i| per_pass[self.exit_of(&gates, i)][i * d..(i + 1) * d].to_vec()).collect();
        let m = (n + 1 - from) as u32;
        let (x, logits) = (self.gpu.storage_init("ouro_score_rows", &rows), self.gpu.storage(m as u64 * v as u64));
        let mut steps = Vec::new();
        self.linear(&mut steps, &x, "lm_head.weight", &logits, m, d as u32, v as u32);
        self.gpu.submit(&[], &steps);
        let logits = self.gpu.read(&logits, m as usize * v);
        Ok(logits
            .chunks(v)
            .zip(&tokens[from..])
            .map(|(row, &t)| {
                let scaled: Vec<f64> = row.iter().map(|&z| z as f64 / temperature as f64).collect();
                let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lse = max + scaled.iter().map(|z| (z - max).exp()).sum::<f64>().ln();
                (scaled[t as usize] - lse) as f32
            })
            .collect())
    }

    fn release_extras(&mut self, table: &mut BlockTable) {
        if let Some(id) = table.take_state_row() {
            if let Some(mut row) = self.rows.remove(&id) {
                for (p, t) in row.extra.iter_mut().enumerate() {
                    t.release(&mut self.allocs[p + 1]);
                }
            }
        }
    }
}

impl PagedDecoder for Engine {
    fn alloc_mut(&mut self) -> &mut BlockAllocator {
        &mut self.allocs[0]
    }

    fn max_prefill_tokens(&self) -> u32 {
        self.cap
    }

    fn free_blocks(&self) -> u32 {
        // The binding constraint: every pass's plane must have room for the
        // token the batch is about to append.
        self.allocs.iter().map(|a| a.free_blocks()).min().unwrap_or(0)
    }

    fn max_seq_len(&self) -> usize {
        self.cap as usize
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab as usize
    }

    /// Per pass: each recurrent pass owns a plane of the same capacity, so a
    /// sequence's charge against any one of them is its own page count.
    fn blocks_for(&self, tokens: u32) -> u32 {
        tokens.div_ceil(self.block_size)
    }

    fn reclaim_prefix(&mut self, _want: u32) -> u32 {
        0
    }

    fn release_table(&mut self, t: &mut BlockTable) {
        self.release_extras(t);
        t.release(&mut self.allocs[0]);
    }

    fn max_batch(&self) -> Option<usize> {
        Some(self.dec.rows as usize)
    }

    fn scores_continuations(&self) -> bool {
        true
    }

    /// The looped path serves no adapters, so the base namespace is the only one.
    fn score_continuation(&mut self, namespace: ::model::paged::KvNamespace, tokens: &[u32], from: usize, temperature: f32) -> Result<Vec<f32>, String> {
        if namespace != ::model::paged::BASE_KV_NAMESPACE {
            return Err(format!("ouro: scoring under adapter namespace {namespace}: the looped path serves no adapters"));
        }
        if let Some(&bad) = tokens.iter().find(|&&t| t >= self.cfg.vocab) {
            return Err(format!("ouro: token {bad} is outside the model vocabulary ({})", self.cfg.vocab));
        }
        self.score(tokens, from, temperature)
    }

    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        match prompt.len() {
            0 => Vec::new(),
            1 => {
                // One token is one decode step of one row, without the head.
                self.step(&mut [table], prompt, false);
                self.gpu.read(&self.dec.hsel, self.cfg.d_model as usize)
            }
            _ => self.prefill_batch(table, prompt),
        }
    }

    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.gpu.write_f32(&self.hrow, hidden);
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        let mut steps = Vec::new();
        self.linear(&mut steps, &self.hrow, "lm_head.weight", &self.hrow_logits, 1, d, v);
        self.gpu.submit(&[], &steps);
        self.gpu.read(&self.hrow_logits, v as usize)
    }

    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        self.forward_batched_logit_rows(tables, inputs).iter().map(|row| argmax(row)).collect()
    }

    fn forward_batched_greedy_window(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); tables.len()];
        let mut next = inputs.to_vec();
        for _ in 0..k.max(1) {
            let chosen = self.forward_batched_greedy(tables, &next);
            for (row, tok) in out.iter_mut().zip(&chosen) {
                row.push(*tok);
            }
            next = chosen;
        }
        out
    }

    fn decode_window_capacity(&self) -> usize {
        1
    }

    fn forward_batched_topk(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32], k: usize) -> Vec<Vec<(u32, f32)>> {
        self.forward_batched_logit_rows(tables, inputs).iter().map(|row| ::model::serve::best_candidates(row, k)).collect()
    }

    fn full_row_readback(&self) -> bool {
        true
    }

    fn forward_batched_logit_rows(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<Vec<f32>> {
        // Batches past the step's buffers run as consecutive steps (the scheduler never asks: `max_batch` bounds it).
        let rows = self.dec.rows as usize;
        let mut out = Vec::with_capacity(tables.len());
        for (chunk, input) in tables.chunks_mut(rows).zip(inputs.chunks(rows)) {
            self.step(chunk, input, true);
            out.extend(self.logit_rows(chunk.len()));
        }
        out
    }

    fn topk_capacity(&self) -> usize {
        8
    }

    fn prefix_stats(&self) -> (u64, u64, usize) {
        (0, 0, 0)
    }

    fn device_stats(&self) -> Option<gpu_core::DeviceStats> {
        self.gpu.stats()
    }

    fn kv_pool_bytes(&self) -> u64 {
        self.pool_tokens as u64 * Engine::kv_bytes_per_token(&self.cfg)
    }

    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.pool_tokens as u64
    }
}

fn argmax(s: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in s.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i;
        }
    }
    bi as u32
}

