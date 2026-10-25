// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements continuous-batching serving for
// recurrent-depth decoders for its clients. If your team needs expertise in
// paged KV pools where every prelude block, every recurrent step's block and
// every coda block owns its own history, or in serving a prelude/core/coda
// model behind one continuous-batching engine, then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The Huginn serving engine.
//!
//! One shared parameter set and one paged history per *stage*. The stage list
//! is `prelude.0..`, then `core.0..` once per recurrent iteration, then
//! `coda.0..`; a core block repeats its weights on every step but each
//! repetition read a different residual stream, so its keys are not the same
//! keys and it owns a separate `[cap, d]` plane. Each stage has its own
//! allocator and every sequence owns one block table per stage; the scheduler
//! only sees the first one.
//!
//! Prefill reads the whole prompt through one pass per stage (`m = n`
//! matmuls and causal chunk attention), because token-at-a-time prefill
//! through the full stage list would replay every weight once per prompt
//! token. Generation then runs the recorded one-token graph.
//!
//! The recurrence starts every position from the same initial latent state,
//! exactly as the materialized forward does: a decode step does not carry the
//! previous position's latent. The engine draws that state once at
//! construction (the reference's `like-init` draw: `Normal(0, init_std) *
//! emb_scale`) and reuses it for every request, so a served model's greedy
//! output is reproducible for a given checkpoint; [`Engine::set_state`]
//! replaces it with a caller's own draw (a parity gate hands it the state the
//! reference used).
//!
//! Not enabled for this model, by design rather than omission: prefix-cache
//! sharing and host-KV swap. A core block's history depends on the step it ran
//! in, which the shared prefix bookkeeping does not key on.

use std::collections::HashMap;

use gpu_core::{BufUsage, DeviceBuffer, Gpu, Step};
use ::model::block::{self, Gqa};
use ::model::paged::{BlockAllocator, BlockTable};
use ::model::serve::PagedDecoder;
use paramstore::ParamStore;

use crate::config::HuginnConfig;
use crate::model;

/// The per-block scratch every stage body writes through, sized for `m` rows.
struct BlockBufs {
    xn1: DeviceBuffer,
    xn2: DeviceBuffer,
    xn3: DeviceBuffer,
    qkv: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    ctx: DeviceBuffer,
    proj: DeviceBuffer,
    tmp: DeviceBuffer,
    fc: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    mh: DeviceBuffer,
    mout: DeviceBuffer,
}

impl BlockBufs {
    fn new(gpu: &Gpu, cfg: &HuginnConfig, m: u64) -> BlockBufs {
        let (d, ff) = (cfg.d_model as u64, cfg.d_ff as u64);
        let st = |x: u64| gpu.storage(x);
        BlockBufs {
            xn1: st(m * d),
            xn2: st(m * d),
            xn3: st(m * d),
            qkv: st(m * 3 * d),
            q: st(m * d),
            k: st(m * d),
            v: st(m * d),
            ctx: st(m * d),
            proj: st(m * d),
            tmp: st(m * d),
            fc: st(m * 2 * ff),
            gate: st(m * ff),
            up: st(m * ff),
            mh: st(m * ff),
            mout: st(m * d),
        }
    }
}

/// One sequence's block tables for stages `1..`; stage 0 is the scheduler's.
struct Row {
    extra: Vec<BlockTable>,
}

/// Scratch for a whole-prompt prefill at `n` rows: every matmul runs with
/// `m = n`, and each stage scatters its K/V into its own paged plane with one
/// batched append over the chunk's `(block, offset)` pairs.
struct PrefillBufs {
    tokens: DeviceBuffer,
    scale: DeviceBuffer,
    h: [DeviceBuffer; 2],
    /// The prelude output every recurrent step re-injects.
    inject: DeviceBuffer,
    latent: DeviceBuffer,
    final_hidden: DeviceBuffer,
    concat: DeviceBuffer,
    block: BlockBufs,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    lse: DeviceBuffer,
    /// Per stage: this chunk's physical block and in-block offset per token.
    blocks: Vec<DeviceBuffer>,
    offsets: Vec<DeviceBuffer>,
}

/// Dispatches one stage's RoPE and attention into `ctx` (the prefill variant
/// also scatters the chunk's K/V into that stage's paged plane).
type Attention<'a> = dyn FnMut(&mut Vec<Step>, usize, &DeviceBuffer, &DeviceBuffer, &DeviceBuffer, &DeviceBuffer) + 'a;

pub struct Engine {
    gpu: Gpu,
    cfg: HuginnConfig,
    ps: ParamStore,
    /// Recurrent iterations the core runs per position.
    steps: u32,
    /// One allocator per KV plane: each plane is its own paged pool, so a
    /// sequence's block tables draw from independent pools of the same
    /// capacity.
    allocs: Vec<BlockAllocator>,
    block_size: u32,
    max_bt: u32,
    /// Per-sequence logit-addressable capacity (`max_bt * block_size`).
    cap: u32,
    /// Weight prefix of each stage, in execution order.
    stages: Vec<String>,
    /// The KV plane each stage reads and writes. Without sharing this is the
    /// stage's own index; under a ring, several recurrent stages map to the
    /// same plane and the later one overwrites the earlier one's history.
    stage_plane: Vec<usize>,
    /// `[max_tokens, d]` K/V planes, one per plane index.
    k_pool: Vec<DeviceBuffer>,
    v_pool: Vec<DeviceBuffer>,
    rows: HashMap<u32, Row>,
    next_row: u32,

    token: DeviceBuffer,
    /// The embedding scale, one value.
    scale: DeviceBuffer,
    /// Residual-stream pair for the prelude and the recurrence.
    h: [DeviceBuffer; 2],
    /// The starting latent state, kept apart from the residual pair because
    /// the first core block overwrites the buffer it is copied into.
    state: DeviceBuffer,
    /// The prelude output, held apart from the residual pair for the same
    /// reason (both are needed by every recurrent step).
    inject: DeviceBuffer,
    /// Normed state after the last recurrent step, before the coda.
    latent: DeviceBuffer,
    /// The final normed hidden state the head turns into logits.
    hrow: DeviceBuffer,
    logits: DeviceBuffer,
    sb: BlockBufs,
    concat: DeviceBuffer,
    /// Per stage: where this step's new row lives in that stage's plane, that
    /// stage's block table and cached length. They must be per stage - a
    /// shared set would send every stage's attention to the last stage's
    /// blocks.
    idx_blocks: Vec<DeviceBuffer>,
    idx_offsets: Vec<DeviceBuffer>,
    idx_tables: Vec<DeviceBuffer>,
    idx_lens: Vec<DeviceBuffer>,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    /// One token's whole graph, recorded once: only the position (one rope
    /// uniform per stage), the new K/V slot and the block table change per
    /// step.
    decode: Vec<Step>,
    rope_uni: Vec<DeviceBuffer>,
}

impl Engine {
    /// Build an engine over a checkpoint directory (HF layout) or a brain
    /// checkpoint file, with `max_tokens` of shared KV capacity and
    /// `block_size` tokens per page. The core runs the checkpoint's own
    /// `mean_recurrence` iterations.
    pub fn open(path: &str, max_tokens: u32, block_size: u32) -> Result<Engine, String> {
        Engine::open_with_steps(path, max_tokens, block_size, None)
    }

    /// [`Engine::open`] with the recurrent depth chosen by the caller (`None`
    /// keeps the checkpoint's `mean_recurrence`). Depth is a serving setting:
    /// every extra iteration is another core stage with its own attention
    /// history, so a long context and the released depth compete for the same
    /// card.
    pub fn open_with_steps(path: &str, max_tokens: u32, block_size: u32, depth: Option<u32>) -> Result<Engine, String> {
        Engine::open_with_sharing(path, max_tokens, block_size, depth, None)
    }

    /// [`Engine::open`] with recurrent depth and KV sharing chosen by the
    /// caller.
    ///
    /// `share` is the length, in recurrent stages, of the ring of attention
    /// histories: stage `k` of the flattened core sequence (iteration `k /
    /// n_recurrent`, block `k % n_recurrent`) uses plane `k % share`, so a
    /// later iteration overwrites the history of the one `share /
    /// n_recurrent` earlier. This is the released model's own `compress-sN`
    /// KV strategy: it cuts the pool from `steps` iterations' histories to
    /// `share / n_recurrent` of them at a small divergence from the full
    /// history, which is what lets the trained depth (32 iterations) run
    /// beside fp32 weights on one card. `None` keeps every stage's own
    /// history, and a `share` at least as large as the core's stage count is
    /// the same thing.
    pub fn open_with_sharing(path: &str, max_tokens: u32, block_size: u32, depth: Option<u32>, share: Option<u32>) -> Result<Engine, String> {
        let gpu = Gpu::new(model::PIPELINES);
        let (cfg, ps) = if std::path::Path::new(path).is_dir() {
            let (cfg, src) = crate::import::open_dir(std::path::Path::new(path))?;
            let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
            let ps = ParamStore::new_with_roles_src(&gpu, roles, &src);
            (cfg, ps)
        } else {
            let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("{path}: {e}"))?;
            let cfg = HuginnConfig::from_json(&reader.config())?;
            let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
            let ps = ParamStore::new_with_roles_src(&gpu, roles, &reader);
            (cfg, ps)
        };
        let steps = depth.filter(|d| *d > 0).unwrap_or(cfg.mean_recurrence);
        Ok(Engine::build(gpu, cfg, ps, steps, max_tokens, block_size, share))
    }

    /// Build from an in-memory weight source for a chosen number of recurrent
    /// iterations (tests use `init::init_weights`).
    pub fn from_source(cfg: HuginnConfig, steps: u32, src: &dyn checkpoint::TensorSource, max_tokens: u32, block_size: u32) -> Engine {
        Engine::from_source_sharing(cfg, steps, None, src, max_tokens, block_size)
    }

    /// [`Engine::from_source`] with a KV-sharing ring (`None` = one history
    /// per stage).
    pub fn from_source_sharing(
        cfg: HuginnConfig,
        steps: u32,
        share: Option<u32>,
        src: &dyn checkpoint::TensorSource,
        max_tokens: u32,
        block_size: u32,
    ) -> Engine {
        let gpu = Gpu::new(model::PIPELINES);
        let roles = cfg.param_list().into_iter().map(|(n, c)| (n, c, paramstore::Role::Frozen)).collect();
        let ps = ParamStore::new_with_roles_src(&gpu, roles, src);
        Engine::build(gpu, cfg, ps, steps, max_tokens, block_size, share)
    }

    fn build(gpu: Gpu, cfg: HuginnConfig, ps: ParamStore, steps: u32, max_tokens: u32, block_size: u32, share: Option<u32>) -> Engine {
        cfg.validate().expect("huginn config");
        assert!(steps > 0, "the recurrent core must run at least once");
        let share = share.filter(|r| *r > 0);
        assert!(block_size > 0 && max_tokens > 0 && max_tokens.is_multiple_of(block_size));
        let num_blocks = max_tokens / block_size;
        let max_bt = num_blocks;
        let cap = num_blocks * block_size;
        let d = cfg.d_model;

        // The stage list: prelude, every recurrent iteration's core blocks in
        // order, coda. A core stage repeats its weights on every iteration but
        // owns its own history.
        let mut stages = Vec::with_capacity((cfg.n_prelude + steps * cfg.n_recurrent + cfg.n_coda) as usize);
        for i in 0..cfg.n_prelude {
            stages.push(format!("prelude.{i}"));
        }
        for _ in 0..steps {
            for i in 0..cfg.n_recurrent {
                stages.push(format!("core.{i}"));
            }
        }
        for i in 0..cfg.n_coda {
            stages.push(format!("coda.{i}"));
        }
        let stage_count = stages.len();

        // The plane each stage reads and writes. Prelude and coda stages always
        // own theirs; the recurrent stages wrap around a ring of `share` when
        // one is asked for.
        let core_stages = (steps * cfg.n_recurrent) as usize;
        let core_planes = share.map_or(core_stages, |r| (r as usize).min(core_stages));
        let mut stage_plane = Vec::with_capacity(stage_count);
        for i in 0..cfg.n_prelude as usize {
            stage_plane.push(i);
        }
        for k in 0..core_stages {
            stage_plane.push(cfg.n_prelude as usize + k % core_planes.max(1));
        }
        for i in 0..cfg.n_coda as usize {
            stage_plane.push(cfg.n_prelude as usize + core_planes + i);
        }
        let planes = cfg.n_prelude as usize + core_planes + cfg.n_coda as usize;
        debug_assert_eq!(stage_plane.len(), stage_count);

        let allocs = (0..planes).map(|_| BlockAllocator::new(num_blocks, block_size)).collect();
        let k_pool = (0..planes).map(|_| gpu.storage(max_tokens as u64 * d as u64)).collect();
        let v_pool = (0..planes).map(|_| gpu.storage(max_tokens as u64 * d as u64)).collect();

        let token = gpu.buffer("huginn_serve_token", 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        let scale = gpu.buffer("huginn_serve_scale", 4, BufUsage::STORAGE | BufUsage::COPY_DST);
        gpu.write_f32(&scale, &[cfg.emb_scale]);
        let h = [gpu.storage(d as u64), gpu.storage(d as u64)];
        let state = gpu.storage(d as u64);
        // The reference's `like-init` state: a truncated normal at `init_std`,
        // then multiplied by `emb_scale`. A plain gaussian at that scale is
        // the same draw without the truncation.
        let mut rng = data::rng::Rng::random();
        let init: Vec<f32> = (0..d).map(|_| (rng.next_gaussian() as f32) * cfg.init_std * cfg.emb_scale).collect();
        gpu.write_f32(&state, &init);
        let inject = gpu.storage(d as u64);
        let latent = gpu.storage(d as u64);
        let hrow = gpu.storage(d as u64);
        let logits = gpu.storage(cfg.vocab as u64);
        let sb = BlockBufs::new(&gpu, &cfg, 1);
        let concat = gpu.storage(2 * d as u64);
        let idx_blocks = (0..stage_count)
            .map(|_| gpu.buffer("huginn_serve_blocks", 4, BufUsage::STORAGE | BufUsage::COPY_DST))
            .collect();
        let idx_offsets = (0..stage_count)
            .map(|_| gpu.buffer("huginn_serve_offsets", 4, BufUsage::STORAGE | BufUsage::COPY_DST))
            .collect();
        let idx_tables = (0..stage_count)
            .map(|_| gpu.buffer("huginn_serve_tables", max_bt as u64 * 4, BufUsage::STORAGE | BufUsage::COPY_DST))
            .collect();
        let idx_lens = (0..stage_count)
            .map(|_| gpu.buffer("huginn_serve_lens", 4, BufUsage::STORAGE | BufUsage::COPY_DST))
            .collect();
        let scores = gpu.storage(cfg.n_heads as u64 * cap as u64);
        let probs = gpu.storage(cfg.n_heads as u64 * cap as u64);
        let rope_uni = (0..stage_count).map(|_| gpu.uniform_dynamic(7)).collect();

        let mut engine = Engine {
            gpu,
            cfg,
            ps,
            steps,
            allocs,
            block_size,
            max_bt,
            cap,
            stages,
            stage_plane,
            k_pool,
            v_pool,
            rows: HashMap::new(),
            next_row: 0,
            token,
            scale,
            h,
            state,
            inject,
            latent,
            hrow,
            logits,
            sb,
            concat,
            idx_blocks,
            idx_offsets,
            idx_tables,
            idx_lens,
            scores,
            probs,
            decode: Vec::new(),
            rope_uni,
        };
        engine.decode = engine.decode_steps();
        engine
    }

    /// Recurrent iterations this engine runs per token.
    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// The latent state every position's recurrence starts from (`d` values).
    /// One state per engine, reused for every request, so a served model's
    /// greedy output is reproducible for the engine's lifetime.
    pub fn set_state(&self, state: &[f32]) {
        assert_eq!(state.len(), self.cfg.d_model as usize, "state must hold d_model values");
        self.gpu.write_f32(&self.state, state);
    }

    /// Free pooled blocks in every KV plane's allocator, in plane order.
    pub fn plane_free_blocks(&self) -> Vec<u32> {
        self.allocs.iter().map(|a| a.free_blocks()).collect()
    }

    /// KV planes this engine owns (`prelude + shared core planes + coda`).
    pub fn planes(&self) -> usize {
        self.allocs.len()
    }

    /// The plane every stage reads and writes, in execution order.
    pub fn stage_planes(&self) -> &[usize] {
        &self.stage_plane
    }

    fn ensure_row(&mut self, table: &mut BlockTable) -> u32 {
        if let Some(id) = table.state_row() {
            return id;
        }
        let id = self.next_row;
        self.next_row += 1;
        table.set_state_row(id);
        let extra = (1..self.planes()).map(|_| BlockTable::new()).collect();
        self.rows.insert(id, Row { extra });
        id
    }

    /// A linear over `m` rows: one workgroup per output column for a decode
    /// row (the kernel reads each weight row once for all m rows), the
    /// register-tiled kernel for a prefill-sized slab on a device with its
    /// 256-thread workgroup, the naive per-element kernel elsewhere.
    fn linear(&self, steps: &mut Vec<Step>, x: &DeviceBuffer, wname: &str, out: &DeviceBuffer, m: u32, k: u32, n: u32) {
        let w = self.ps.w(wname);
        if m <= 32 {
            steps.push(self.gpu.dispatch(model::MATMUL_GEMV_ID, &[x, w, out], &[m, k, n], gpu_core::Dispatch::Workgroups(n)));
        } else if m >= 128 && self.gpu.caps().max_workgroup_size >= 256 {
            let (kernel, grid) = ::model::block::pick_gemm(m as usize, n as usize, model::MATMUL_ID, model::MATMUL_REG2_ID, false);
            steps.push(self.gpu.dispatch(kernel, &[x, w, out], &[m, k, n], grid));
        } else {
            steps.push(self.gpu.step(model::MATMUL_ID, &[x, w, out], &[m, k, n], m * n));
        }
    }

    /// One sandwich block over `m` rows. `attention` dispatches the stage's
    /// RoPE and attention (paged decode for one row, causal chunk for a
    /// prefill slab) into `ctx`.
    #[allow(clippy::too_many_arguments)]
    fn block(
        &self,
        steps: &mut Vec<Step>,
        sb: &BlockBufs,
        stage: usize,
        x_in: &DeviceBuffer,
        x_out: &DeviceBuffer,
        m: u32,
        attention: &mut Attention<'_>,
    ) {
        let cfg = &self.cfg;
        let gpu = &self.gpu;
        let ids = model::kernel_ids();
        let (d, ff) = (cfg.d_model, cfg.d_ff);
        let w = |leaf: &str| format!("{}.{leaf}", self.stages[stage]);

        steps.push(block::rmsnorm_fwd(gpu, &ids, x_in, self.ps.w(&w("ln1.weight")), &sb.xn1, d, m, cfg.norm_eps));
        self.linear(steps, &sb.xn1, &w("attn.wqkv.weight"), &sb.qkv, m, d, 3 * d);
        self.copy_cols(steps, &sb.qkv, m, d, 3 * d, 0, &sb.q, d, 0);
        self.copy_cols(steps, &sb.qkv, m, d, 3 * d, d, &sb.k, d, 0);
        self.copy_cols(steps, &sb.qkv, m, d, 3 * d, 2 * d, &sb.v, d, 0);
        steps.push(gpu.step(model::BIAS_ADD_ID, &[&sb.q, self.ps.w(&w("attn.q_bias"))], &[m, d], m * d));
        steps.push(gpu.step(model::BIAS_ADD_ID, &[&sb.k, self.ps.w(&w("attn.k_bias"))], &[m, d], m * d));
        attention(steps, stage, &sb.q, &sb.k, &sb.v, &sb.ctx);
        self.linear(steps, &sb.ctx, &w("attn.wo.weight"), &sb.proj, m, d, d);
        steps.push(gpu.step(model::ADD2_ID, &[&sb.proj, x_in, &sb.tmp], &[m * d], m * d));
        steps.push(block::rmsnorm_fwd(gpu, &ids, &sb.tmp, self.ps.w(&w("ln2.weight")), &sb.xn2, d, m, cfg.norm_eps));
        steps.push(block::rmsnorm_fwd(gpu, &ids, &sb.xn2, self.ps.w(&w("ln3.weight")), &sb.xn3, d, m, cfg.norm_eps));
        self.linear(steps, &sb.xn3, &w("mlp.fc.weight"), &sb.fc, m, d, 2 * ff);
        self.copy_cols(steps, &sb.fc, m, ff, 2 * ff, 0, &sb.gate, ff, 0);
        self.copy_cols(steps, &sb.fc, m, ff, 2 * ff, ff, &sb.up, ff, 0);
        steps.push(block::swiglu_fwd(gpu, &ids, &sb.gate, &sb.up, &sb.mh, m * ff));
        self.linear(steps, &sb.mh, &w("mlp.proj.weight"), &sb.mout, m, ff, d);
        steps.push(gpu.step(model::ADD2_ID, &[&sb.mout, &sb.xn2, &sb.tmp], &[m * d], m * d));
        steps.push(block::rmsnorm_fwd(gpu, &ids, &sb.tmp, self.ps.w(&w("ln4.weight")), x_out, d, m, cfg.norm_eps));
    }

    /// Copy `cnt` columns starting at `src_off` of each of `rows` rows from a
    /// `src_stride`-wide matrix into a `dst_stride`-wide one.
    fn copy_cols(
        &self,
        steps: &mut Vec<Step>,
        src: &DeviceBuffer,
        rows: u32,
        cnt: u32,
        src_stride: u32,
        src_off: u32,
        dst: &DeviceBuffer,
        dst_stride: u32,
        dst_off: u32,
    ) {
        steps.push(self.gpu.step(model::COPY_COLS_ID, &[src, dst], &[rows, cnt, src_stride, src_off, dst_stride, dst_off], rows * cnt));
    }

    /// The decode graph, recorded once: one new token through every stage,
    /// appending its K/V at the absolute position. The position reaches the
    /// per-stage RoPE dispatches through a uniform buffer written per step, so
    /// the per-token cost is buffer writes and one submit.
    fn decode_steps(&self) -> Vec<Step> {
        let cfg = &self.cfg;
        let gpu = &self.gpu;
        let ids = model::kernel_ids();
        let dec = model::decode_batched_ids();
        let sb = &self.sb;
        let d = cfg.d_model;
        let (nh, hd) = (cfg.n_heads, cfg.head_dim);
        let mut stage = 0usize;

        let mut attention = |steps: &mut Vec<Step>, stage: usize, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, ctx: &DeviceBuffer| {
            let plane = self.stage_plane[stage];
            steps.push(gpu.step_buf(model::ROPE_AT_ID, &self.rope_uni[stage], &[q], nh * hd / 2));
            steps.push(gpu.step_buf(model::ROPE_AT_ID, &self.rope_uni[stage], &[k], nh * hd / 2));
            steps.extend(block::gqa_decode_batched_step(
                gpu,
                &dec,
                nh,
                nh,
                hd,
                1,
                self.block_size,
                self.max_bt,
                self.cap,
                q,
                k,
                v,
                &self.k_pool[plane],
                &self.v_pool[plane],
                &self.idx_blocks[stage],
                &self.idx_offsets[stage],
                &self.idx_tables[stage],
                &self.idx_lens[stage],
                &self.scores,
                &self.probs,
                ctx,
            ));
        };

        let mut steps = vec![gpu.step(model::EMBED_ID, &[&self.token, self.ps.w("tok.weight"), &self.h[0]], &[d, 1], d)];
        // Embeddings enter the prelude scaled, as in the materialized forward.
        steps.push(gpu.step(model::SCALE_ROW_ID, &[&self.h[0], &self.scale, &self.h[1]], &[d, d], d));
        // The prelude rewrites the embedding; its output for this position is
        // what every recurrent step re-injects. Both it and the starting state
        // need their own buffers: the residual pair is overwritten by the
        // first core block.
        let mut cur = 1usize;
        for _ in 0..cfg.n_prelude {
            self.block(&mut steps, sb, stage, &self.h[cur], &self.h[1 - cur], 1, &mut attention);
            stage += 1;
            cur = 1 - cur;
        }
        self.copy_cols(&mut steps, &self.h[cur], 1, d, d, 0, &self.inject, d, 0);
        self.copy_cols(&mut steps, &self.state, 1, d, d, 0, &self.h[0], d, 0);

        // Recurrence. `state` holds the running latent; block 0 consumes the
        // adapter output and writes the other residual buffer, later blocks
        // ping-pong, so the last block's output is the running state.
        let mut state = 0usize;
        for _ in 0..self.steps {
            let mid = 1 - state;
            self.copy_cols(&mut steps, &self.h[state], 1, d, 2 * d, 0, &self.concat, 2 * d, 0);
            self.copy_cols(&mut steps, &self.inject, 1, d, d, 0, &self.concat, 2 * d, d);
            self.linear(&mut steps, &self.concat, "adapter.weight", &self.h[mid], 1, 2 * d, d);
            self.block(&mut steps, sb, stage, &self.h[mid], &self.h[state], 1, &mut attention);
            stage += 1;
            for _ in 1..cfg.n_recurrent {
                self.block(&mut steps, sb, stage, &self.h[state], &self.h[1 - state], 1, &mut attention);
                stage += 1;
                state = 1 - state;
            }
        }

        // Final norm, then the coda from the normed state, then the norm
        // again into the hidden state the head reads.
        steps.push(block::rmsnorm_fwd(gpu, &ids, &self.h[state], self.ps.w("ln_f.weight"), &self.latent, d, 1, cfg.norm_eps));
        let mut last: &DeviceBuffer = &self.latent;
        let mut ccur = 0usize;
        for _ in 0..cfg.n_coda {
            let x_out = &self.h[ccur];
            self.block(&mut steps, sb, stage, last, x_out, 1, &mut attention);
            stage += 1;
            last = x_out;
            ccur = 1 - ccur;
        }
        steps.push(block::rmsnorm_fwd(gpu, &ids, last, self.ps.w("ln_f.weight"), &self.hrow, d, 1, cfg.norm_eps));
        debug_assert_eq!(stage, self.stages.len());
        steps
    }

    /// One token through every stage, appending its K/V to each stage's
    /// table. Returns the final normed hidden state.
    fn advance(&mut self, table: &mut BlockTable, row: u32, token: u32) -> Vec<f32> {
        let d = self.cfg.d_model;
        let (nh, hd) = (self.cfg.n_heads, self.cfg.head_dim);
        let theta = gpu_core::f(self.cfg.rope_theta);
        self.gpu.write(&self.token, &[token]);
        for stage in 0..self.stages.len() {
            // Plane 0 appends into the scheduler's table; every other plane
            // appends into its own table, from its own allocator, so each
            // paged plane is charged exactly the sequence's page count. A
            // stage reads and writes its plane, so under a ring the stage that
            // reuses a plane appends the new row at that plane's length, which
            // is the same position for every plane.
            let plane = self.stage_plane[stage];
            let pos = if plane == 0 { table.len() } else { self.rows[&row].extra[plane - 1].len() };
            let appended = if plane == 0 {
                table.append(&mut self.allocs[0])
            } else {
                let ex = &mut self.rows.get_mut(&row).expect("row").extra[plane - 1];
                ex.append(&mut self.allocs[plane])
            };
            let (blk, off) = appended.expect("paged KV pool exhausted");
            self.gpu.write(&self.idx_blocks[stage], &[blk]);
            self.gpu.write(&self.idx_offsets[stage], &[off]);
            self.gpu.write(&self.idx_lens[stage], &[pos + 1]);
            let mut bt = vec![0u32; self.max_bt as usize];
            let blocks = if plane == 0 {
                table.blocks()
            } else {
                self.rows[&row].extra[plane - 1].blocks()
            };
            bt[..blocks.len()].copy_from_slice(blocks);
            self.gpu.write(&self.idx_tables[stage], &bt);
            self.gpu.write(&self.rope_uni[stage], &[1, nh, hd, d, 0, pos, theta]);
        }
        self.gpu.submit(&[], &self.decode);
        self.gpu.read(&self.hrow, d as usize)
    }

    /// Prefill the whole prompt in one pass per stage: matmuls run with
    /// `m = n`, attention is the causal full-chunk form, and each stage
    /// scatters its K/V into its own paged plane. Returns the last token's
    /// final normed hidden state (what admission samples from).
    fn prefill_batch(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        let cfg = self.cfg.clone();
        let n = prompt.len() as u32;
        assert!(n > 0, "prefill needs at least one token");
        assert_eq!(table.len(), 0, "prefill_batch expects a fresh sequence");
        assert!(n <= self.cap, "prompt of {n} tokens exceeds the per-sequence capacity {}", self.cap);
        let row = self.ensure_row(table);
        let bs = self.block_size;

        // Reserve the chunk in every stage's table, then record each token's
        // (block, offset) slot for the batched scatter.
        table.reserve(n, &mut self.allocs[0]).expect("paged KV pool exhausted");
        let mut reserved = vec![false; self.planes()];
        reserved[0] = true;
        // Slots are recorded per stage (each stage scatters its own chunk),
        // but a plane is reserved once however many stages share it.
        let mut slots: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(self.stages.len());
        slots.push(locate_all(table, n, bs));
        for stage in 1..self.stages.len() {
            let plane = self.stage_plane[stage];
            if !reserved[plane] {
                let ex = &mut self.rows.get_mut(&row).expect("row").extra[plane - 1];
                ex.reserve(n, &mut self.allocs[plane]).expect("paged KV pool exhausted");
                reserved[plane] = true;
            }
            let ex = &self.rows[&row].extra[plane - 1];
            slots.push(locate_all(ex, n, bs));
        }

        let gpu = &self.gpu;
        let flash = gpu.caps().workgroup_reductions && gpu.caps().max_workgroup_size >= 256;
        let d = cfg.d_model as u64;
        let nn = n as u64;
        let nh = cfg.n_heads as u64;
        let st = |x: u64| gpu.storage(x);
        let b = PrefillBufs {
            tokens: gpu.buffer("huginn_prefill_tokens", nn * 4, BufUsage::STORAGE | BufUsage::COPY_DST),
            scale: gpu.buffer("huginn_prefill_scale", nn * 4, BufUsage::STORAGE | BufUsage::COPY_DST),
            h: [st(nn * d), st(nn * d)],
            inject: st(nn * d),
            latent: st(nn * d),
            final_hidden: st(nn * d),
            concat: st(nn * 2 * d),
            block: BlockBufs::new(gpu, &cfg, nn),
            // The flash kernel needs no score slab; the materialized fallback does.
            scores: if flash { st(1) } else { st(nh * nn * nn) },
            probs: if flash { st(1) } else { st(nh * nn * nn) },
            lse: st(nh * nn),
            blocks: slots
                .iter()
                .map(|(x, _)| {
                    let buf = gpu.buffer("huginn_pf_blocks", nn * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
                    gpu.write(&buf, x);
                    buf
                })
                .collect(),
            offsets: slots
                .iter()
                .map(|(_, y)| {
                    let buf = gpu.buffer("huginn_pf_offsets", nn * 4, BufUsage::STORAGE | BufUsage::COPY_DST);
                    gpu.write(&buf, y);
                    buf
                })
                .collect(),
        };
        gpu.write_f32(&b.scale, &vec![cfg.emb_scale; n as usize]);
        gpu.write(&b.tokens, prompt);

        let ids = model::kernel_ids();
        let dec = model::decode_batched_ids();
        let (nh, hd) = (cfg.n_heads, cfg.head_dim);
        let theta = cfg.rope_theta;
        let ga = Gqa { b: 1, t: n, n_heads: nh, n_kv_heads: nh, head_dim: hd };
        let mut attention = |steps: &mut Vec<Step>, stage: usize, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, ctx: &DeviceBuffer| {
            // Huginn's rotary pairs consecutive channels (`rope_train`), the
            // kernel its materialized forward dispatches.
            steps.push(block::rope_fwd(gpu, &ids, q, n, nh, hd, d as u32, n, theta));
            steps.push(block::rope_fwd(gpu, &ids, k, n, nh, hd, d as u32, n, theta));
            let plane = self.stage_plane[stage];
            steps.push(gpu.step(dec.kv_append_batched, &[k, &b.blocks[stage], &b.offsets[stage], &self.k_pool[plane]], &[n, d as u32, bs], n * d as u32));
            steps.push(gpu.step(dec.kv_append_batched, &[v, &b.blocks[stage], &b.offsets[stage], &self.v_pool[plane]], &[n, d as u32, bs], n * d as u32));
            if flash {
                steps.push(block::flash_gqa_causal_fwd_portable(gpu, model::FLASH_GQA_ID, &ga, q, k, v, ctx, &b.lse));
            } else {
                steps.extend(block::gqa_fwd(gpu, &ids, &ga, q, k, v, &b.scores, &b.probs, ctx));
            }
        };

        let mut stage = 0usize;
        let mut steps = vec![gpu.step(model::EMBED_ID, &[&b.tokens, self.ps.w("tok.weight"), &b.h[0]], &[cfg.d_model, n], n * cfg.d_model)];
        steps.push(gpu.step(model::SCALE_ROW_ID, &[&b.h[0], &b.scale, &b.h[1]], &[n * cfg.d_model, cfg.d_model], n * cfg.d_model));
        let mut cur = 1usize;
        for _ in 0..cfg.n_prelude {
            self.block(&mut steps, &b.block, stage, &b.h[cur], &b.h[1 - cur], n, &mut attention);
            stage += 1;
            cur = 1 - cur;
        }
        // The prelude's output, not the raw embedding, is what every recurrent
        // step re-injects.
        self.copy_cols(&mut steps, &b.h[cur], n, cfg.d_model, cfg.d_model, 0, &b.inject, cfg.d_model, 0);

        // Every row starts the recurrence from the same state: a source stride
        // of zero broadcasts the one stored row to all `n` rows.
        self.copy_cols(&mut steps, &self.state, n, cfg.d_model, 0, 0, &b.h[0], cfg.d_model, 0);
        let mut state = 0usize;
        for _ in 0..self.steps {
            let mid = 1 - state;
            self.copy_cols(&mut steps, &b.h[state], n, cfg.d_model, cfg.d_model, 0, &b.concat, 2 * cfg.d_model, 0);
            self.copy_cols(&mut steps, &b.inject, n, cfg.d_model, cfg.d_model, 0, &b.concat, 2 * cfg.d_model, cfg.d_model);
            self.linear(&mut steps, &b.concat, "adapter.weight", &b.h[mid], n, 2 * cfg.d_model, cfg.d_model);
            self.block(&mut steps, &b.block, stage, &b.h[mid], &b.h[state], n, &mut attention);
            stage += 1;
            for _ in 1..cfg.n_recurrent {
                self.block(&mut steps, &b.block, stage, &b.h[state], &b.h[1 - state], n, &mut attention);
                stage += 1;
                state = 1 - state;
            }
        }

        steps.push(block::rmsnorm_fwd(gpu, &ids, &b.h[state], self.ps.w("ln_f.weight"), &b.latent, cfg.d_model, n, cfg.norm_eps));
        let mut last: &DeviceBuffer = &b.latent;
        let mut ccur = 0usize;
        for _ in 0..cfg.n_coda {
            let x_out = &b.h[ccur];
            self.block(&mut steps, &b.block, stage, last, x_out, n, &mut attention);
            stage += 1;
            last = x_out;
            ccur = 1 - ccur;
        }
        steps.push(block::rmsnorm_fwd(gpu, &ids, last, self.ps.w("ln_f.weight"), &b.final_hidden, cfg.d_model, n, cfg.norm_eps));
        debug_assert_eq!(stage, self.stages.len());
        gpu.submit(&[], &steps);

        let all = gpu.read(&b.final_hidden, (nn * d) as usize);
        all[(n as usize - 1) * d as usize..].to_vec()
    }

    fn release_extras(&mut self, table: &mut BlockTable) {
        if let Some(id) = table.take_state_row() {
            if let Some(mut row) = self.rows.remove(&id) {
                for (i, t) in row.extra.iter_mut().enumerate() {
                    t.release(&mut self.allocs[i + 1]);
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
        // The binding constraint: every stage's plane must have room for the
        // token the batch is about to append.
        self.allocs.iter().map(|a| a.free_blocks()).min().unwrap_or(0)
    }

    fn max_seq_len(&self) -> usize {
        self.cap as usize
    }

    fn vocab(&self) -> usize {
        self.cfg.vocab as usize
    }

    /// Per stage: each stage owns a plane of the same capacity, so a
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

    fn prefill(&mut self, table: &mut BlockTable, prompt: &[u32]) -> Vec<f32> {
        if prompt.len() > 1 {
            return self.prefill_batch(table, prompt);
        }
        let row = self.ensure_row(table);
        let mut hidden = Vec::new();
        for tok in prompt {
            hidden = self.advance(table, row, *tok);
        }
        hidden
    }

    fn logits(&self, hidden: &[f32]) -> Vec<f32> {
        self.gpu.write_f32(&self.hrow, hidden);
        let (d, v) = (self.cfg.d_model, self.cfg.vocab);
        let mut steps = Vec::new();
        self.linear(&mut steps, &self.hrow, "tok.weight", &self.logits, 1, d, v);
        self.gpu.submit(&[], &steps);
        self.gpu.read(&self.logits, v as usize)
    }

    fn forward_batched_greedy(&mut self, tables: &mut [&mut BlockTable], inputs: &[u32]) -> Vec<u32> {
        let mut out = Vec::with_capacity(tables.len());
        for (t, input) in tables.iter_mut().zip(inputs) {
            let t: &mut BlockTable = t;
            let row = self.ensure_row(t);
            let hidden = self.advance(t, row, *input);
            out.push(argmax(&self.logits(&hidden)));
        }
        out
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
        let mut out = Vec::with_capacity(tables.len());
        for (t, input) in tables.iter_mut().zip(inputs) {
            let t: &mut BlockTable = t;
            let row = self.ensure_row(t);
            let hidden = self.advance(t, row, *input);
            let mut candidates: Vec<(u32, f32)> = self.logits(&hidden).into_iter().enumerate().map(|(i, v)| (i as u32, v)).collect();
            candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            candidates.truncate(k.max(1));
            out.push(candidates);
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
        let per_plane = self.cap as u64 * self.cfg.d_model as u64 * 4;
        per_plane * 2 * self.planes() as u64
    }

    fn kv_pool_capacity_tokens(&self) -> u64 {
        self.cap as u64
    }
}

/// The `(block, offset)` slot of every token in a chunk, for the batched
/// scatter.
fn locate_all(t: &BlockTable, n: u32, block_size: u32) -> (Vec<u32>, Vec<u32>) {
    let mut blocks = vec![0u32; n as usize];
    let mut offsets = vec![0u32; n as usize];
    for i in 0..n as usize {
        let (b, o) = t.locate(i as u32, block_size);
        blocks[i] = b;
        offsets[i] = o;
    }
    (blocks, offsets)
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
