// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A stack of residual blocks over a subject's visits (`Backbone::Stack`).
//!
//! The sequence per subject is the visit slots in time order, then a query
//! token at the prediction time, then padding up to a whole number of chunks
//! (`HorizonConfig::stack_layout`):
//!
//! ```text
//! x0  = [u of each visit slot | query | 0 ...]                    [B*T, D]
//! per block l (pre-norm, as the set encoder's layers are):
//!   x   = x + Mixer_l(LN1(x))
//!   x   = x + MLP(LN2(x))
//! z   = LN_f(x)[query row]                                         [B, D]
//! ```
//!
//! The mixer of a block is one of two sequence mixers (see
//! `HorizonConfig::mixer` for how a stack picks one per block):
//!
//! **Attention**: bidirectional multi-head attention over the visits, with
//! rotary angles taken from each row's real time relative to the prediction
//! time (the same rotation as `Backbone::Attention`). Every visit precedes
//! the prediction time, so letting visits see each other leaks nothing; unused
//! visit slots and padding are never keys.
//!
//! **Gated DeltaNet**: the matrix-state delta rule of `model::gdn`, read in
//! time order, per head with `k`, `q` L2-normalised:
//!
//! ```text
//! S_t = exp(g_t) * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
//! o_t = S_t q_t / sqrt(d_head)
//! g_t = -softplus(rate_h) * dt_t        beta_t = sigmoid(w_beta . x_t + b)
//! ```
//!
//! `dt_t` is the physical time since the previous visit (for the query, since
//! the last visit): the decay is the exponential of the time that actually
//! passed, as in the diagonal state of `Backbone::State`, so a gap longer than
//! any in training is extrapolated by the same exponential. The raw rates
//! start log-spaced from days to a century (`init::raw_rates`). An unused
//! visit slot or a padding row has `dt < 0`: it neither decays nor writes, the
//! recurrence passes the state through it. The query token is an ordinary
//! token at the prediction time, so it reads (and adds to) the state decayed
//! over the gap since the last visit, and `o` at that row is the block's
//! output there. A new visit costs one more recurrent step.
//!
//! The recurrence runs on the chunked kernels of `model::gdn` (the same ones
//! the Qwen3.5 mixer trains with); the only new kernels are the gap-aware
//! gates, `gdn_gap_gate` and its backward.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block;
use model::gdn::{
    gdn_chunk_bwd, gdn_chunk_fwd_train, GdnBwdIds, GdnBwdScratchBufs, GdnFastIds, GdnIds,
    GdnScratchTrainBufs, GdnShape,
};

use super::*;
use crate::config::BlockMixer;

/// The epsilon of the query/key L2 normalisation (the Qwen3.5 mixer's).
const L2_EPS: f32 = 1e-6;

/// Activations of an attention block.
struct AttnBlock {
    qkv: DeviceBuffer,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    d_ctx: DeviceBuffer,
}

/// Activations and gradients of a Gated DeltaNet block, token-major (`[R, D]`
/// with the heads along the columns) unless named `_cm` (chunk-major).
struct GdnBlock {
    q_pre: DeviceBuffer,
    k_pre: DeviceBuffer,
    v: DeviceBuffer,
    q_n: DeviceBuffer,
    k_n: DeviceBuffer,
    /// `[R, H]` write-strength pre-activation, then its gate outputs.
    b_pre: DeviceBuffer,
    g: DeviceBuffer,
    beta: DeviceBuffer,
    q_cm: DeviceBuffer,
    k_cm: DeviceBuffer,
    v_cm: DeviceBuffer,
    g_cm: DeviceBuffer,
    beta_cm: DeviceBuffer,
    out_cm: DeviceBuffer,
    out_tok: DeviceBuffer,
    /// `[B*H, d_head, d_head]`: the state every sequence starts from (zero)
    /// and ends in.
    initial_state: DeviceBuffer,
    final_state: DeviceBuffer,
    saved: GdnScratchTrainBufs,
    // backward
    bwd: GdnBwdScratchBufs,
    d_final_state: DeviceBuffer,
    d_initial_state: DeviceBuffer,
    d_out_tok: DeviceBuffer,
    d_out_cm: DeviceBuffer,
    d_q_cm: DeviceBuffer,
    d_k_cm: DeviceBuffer,
    d_v_cm: DeviceBuffer,
    d_g_cm: DeviceBuffer,
    d_beta_cm: DeviceBuffer,
    d_q_n: DeviceBuffer,
    d_k_n: DeviceBuffer,
    d_v: DeviceBuffer,
    d_g: DeviceBuffer,
    d_beta: DeviceBuffer,
    d_b_pre: DeviceBuffer,
    d_rate: DeviceBuffer,
    d_q_pre: DeviceBuffer,
    d_k_pre: DeviceBuffer,
}

enum MixerBufs {
    Attention(AttnBlock),
    Gdn(Box<GdnBlock>),
}

/// One block's buffers: the mixer's own and the saved activations of the
/// pre-norm residual structure around it.
struct Block {
    ln1_out: DeviceBuffer,
    xmid: DeviceBuffer,
    ln2_out: DeviceBuffer,
    up_pre: DeviceBuffer,
    up: DeviceBuffer,
    mixer: MixerBufs,
}

/// Device buffers of the stack backbone.
pub(super) struct StackBufs {
    /// Subject slots, visit slots per subject and rows per subject.
    subjects: u32,
    visits: u32,
    rows: u32,
    /// `[S]` each visit set's row in the sequences (constant).
    seq_rows: DeviceBuffer,
    /// `[B]` each subject's query row (constant).
    query_rows: DeviceBuffer,
    /// `[B]` zeros: every subject reads the one query embedding (constant).
    query_index: DeviceBuffer,
    /// `[d_head]` ones: the L2 normalisation has no learnable gain (constant).
    ones: DeviceBuffer,
    /// `[R]` each row's time relative to entry (rotary angle), whether it is
    /// a key, and its elapsed time since the previous token (`< 0`: padding).
    pos: DeviceBuffer,
    keep: DeviceBuffer,
    dt: DeviceBuffer,
    u: DeviceBuffer,
    qe: DeviceBuffer,
    /// The residual stream, `blocks + 1` states.
    x: Vec<DeviceBuffer>,
    blocks: Vec<Block>,
    mix: DeviceBuffer,
    ffn_out: DeviceBuffer,
    xf: DeviceBuffer,
    // backward
    d_x: Vec<DeviceBuffer>,
    d_xf: DeviceBuffer,
    d_xmid: DeviceBuffer,
    d_branch: DeviceBuffer,
    d_tmp: DeviceBuffer,
    d_up: DeviceBuffer,
    d_up_pre: DeviceBuffer,
    d_scores: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_u: DeviceBuffer,
    d_qe: DeviceBuffer,
}

impl StackBufs {
    pub(super) fn new(gpu: &Gpu, cfg: &HorizonConfig, b: u32) -> StackBufs {
        let vs = cfg.sets_per_subject();
        let (chunk, t) = cfg.stack_layout();
        let (d, ff, h) = (cfg.d_model as u64, cfg.d_ff as u64, cfg.n_heads as u64);
        let hd = d / h;
        let (s, r, bb) = ((b * vs) as u64, (b * t) as u64, b as u64);
        let st = |x: u64| gpu.storage(x);
        let input = |label: &str, words: u64| {
            gpu.buffer(label, words * 4, BufUsage::STORAGE | BufUsage::COPY_DST)
        };
        let (seq_rows, query_rows, query_index, ones) = (
            input("stack_seq_rows", s),
            input("stack_query_rows", bb),
            input("stack_query_index", bb),
            input("stack_ones", hd),
        );
        let set_rows: Vec<u32> = (0..b * vs).map(|x| (x / vs) * t + x % vs).collect();
        let q_rows: Vec<u32> = (0..b).map(|i| i * t + vs).collect();
        gpu.write(&seq_rows, &set_rows);
        gpu.write(&query_rows, &q_rows);
        gpu.write(&query_index, &vec![0u32; b as usize]);
        gpu.write_f32(&ones, &vec![1.0; hd as usize]);
        let shape = GdnShape {
            b,
            h: cfg.n_heads,
            t,
            dk: hd as u32,
            dv: hd as u32,
            chunk,
        };
        let blocks = cfg
            .block_mixers()
            .into_iter()
            .map(|mixer| Block {
                ln1_out: st(r * d),
                xmid: st(r * d),
                ln2_out: st(r * d),
                up_pre: st(r * ff),
                up: st(r * ff),
                mixer: match mixer {
                    BlockMixer::Attention => {
                        let att = bb * h * (t as u64 * t as u64);
                        MixerBufs::Attention(AttnBlock {
                            qkv: st(r * 3 * d),
                            scores: st(att),
                            probs: st(att),
                            ctx: st(r * d),
                            d_ctx: st(r * d),
                        })
                    }
                    BlockMixer::GatedDeltaNet => {
                        let state = bb * h * hd * hd;
                        MixerBufs::Gdn(Box::new(GdnBlock {
                            q_pre: st(r * d),
                            k_pre: st(r * d),
                            v: st(r * d),
                            q_n: st(r * d),
                            k_n: st(r * d),
                            b_pre: st(r * h),
                            g: st(r * h),
                            beta: st(r * h),
                            q_cm: st(r * d),
                            k_cm: st(r * d),
                            v_cm: st(r * d),
                            g_cm: st(r * h),
                            beta_cm: st(r * h),
                            out_cm: st(r * d),
                            out_tok: st(r * d),
                            initial_state: st(state),
                            final_state: st(state),
                            saved: GdnScratchTrainBufs::new(gpu, &shape),
                            bwd: GdnBwdScratchBufs::new(gpu, &shape),
                            d_final_state: st(state),
                            d_initial_state: st(state),
                            d_out_tok: st(r * d),
                            d_out_cm: st(r * d),
                            d_q_cm: st(r * d),
                            d_k_cm: st(r * d),
                            d_v_cm: st(r * d),
                            d_g_cm: st(r * h),
                            d_beta_cm: st(r * h),
                            d_q_n: st(r * d),
                            d_k_n: st(r * d),
                            d_v: st(r * d),
                            d_g: st(r * h),
                            d_beta: st(r * h),
                            d_b_pre: st(r * h),
                            d_rate: st(r * h),
                            d_q_pre: st(r * d),
                            d_k_pre: st(r * d),
                        }))
                    }
                },
            })
            .collect::<Vec<_>>();
        let n = blocks.len();
        StackBufs {
            subjects: b,
            visits: vs,
            rows: t,
            seq_rows,
            query_rows,
            query_index,
            ones,
            pos: input("stack_pos", r),
            keep: input("stack_keep", r),
            dt: input("stack_dt", r),
            u: st(s * d),
            qe: st(bb * d),
            x: (0..=n).map(|_| st(r * d)).collect(),
            blocks,
            mix: st(r * d),
            ffn_out: st(r * d),
            xf: st(r * d),
            d_x: (0..=n).map(|_| st(r * d)).collect(),
            d_xf: st(r * d),
            d_xmid: st(r * d),
            d_branch: st(r * d),
            d_tmp: st(r * d),
            d_up: st(r * ff),
            d_up_pre: st(r * ff),
            d_scores: st(bb * h * (t as u64 * t as u64)),
            d_qkv: st(r * 3 * d),
            d_u: st(s * d),
            d_qe: st(bb * d),
        }
    }

    /// Upload the batch's per-row inputs: the sequences' rotary times, key
    /// mask and elapsed times, laid out in the stack's rows.
    pub(super) fn write(&self, gpu: &Gpu, hb: &HostBatch) {
        let (b, vs, t) = (self.subjects as usize, self.visits as usize, self.rows as usize);
        let (mut pos, mut keep, mut dt) = (vec![0.0f32; b * t], vec![0u32; b * t], vec![-1.0f32; b * t]);
        for i in 0..b {
            for j in 0..=vs {
                let (row, src) = (i * t + j, i * (vs + 1) + j);
                pos[row] = hb.seq_pos[src];
                keep[row] = hb.seq_keep[src];
                // The query's gap is the one from the last visit to entry.
                dt[row] = if j < vs {
                    hb.visit_dt[i * vs + j]
                } else {
                    hb.visit_dt[b * vs + i]
                };
            }
        }
        gpu.write_f32(&self.pos, &pos);
        gpu.write(&self.keep, &keep);
        gpu.write_f32(&self.dt, &dt);
    }

    /// Buffers the forward pass accumulates into: every Gated DeltaNet block's
    /// saved history (written by `splice_add`) and its triangular scratch.
    pub(super) fn forward_cleared(&self) -> Vec<&DeviceBuffer> {
        self.blocks
            .iter()
            .filter_map(|b| match &b.mixer {
                MixerBufs::Gdn(g) => Some(g.saved.clears()),
                MixerBufs::Attention(_) => None,
            })
            .flatten()
            .collect()
    }

    /// Buffers the backward pass accumulates into.
    pub(super) fn cleared(&self) -> Vec<&DeviceBuffer> {
        let mut v = Vec::new();
        for b in &self.blocks {
            if let MixerBufs::Gdn(g) = &b.mixer {
                v.extend(g.bwd.clears());
                v.extend([&g.d_q_cm, &g.d_k_cm, &g.d_v_cm, &g.d_beta_cm]);
            }
        }
        v
    }
}

/// The kernel indices of `model::gdn`'s forward and backward in this model's
/// pipeline list.
fn gdn_ids() -> (GdnIds, GdnBwdIds) {
    (
        GdnIds {
            bmm: BMM,
            bmm_acc: BMM_ACC,
            cumsum_step: GDN_CHUNK_CUMSUM_STEP,
            decay_mask: GDN_DECAY_MASK,
            mask_strict_lower: GDN_MASK_STRICT_LOWER,
            ut_step: GDN_UT_STEP,
            add_identity: GDN_ADD_IDENTITY,
            row_scale: SCALE_ROW,
            row_scale_off: GDN_ROW_SCALE_OFF,
            decay_scale: GDN_DECAY_SCALE,
            state_decay: GDN_STATE_DECAY,
            exp: EXP,
            sub: SUB,
            mul: MUL,
            region_copy: REGION_COPY,
            fast: Some(GdnFastIds {
                ut_fwd: GDN_UT_FWD,
                bmm_tiled: BMM_TILED,
            }),
        },
        GdnBwdIds {
            splice_add: SPLICE_ADD,
            row_dot: ROW_DOT,
            scale_add: SCALE_ADD,
            reverse_cumsum_step: GDN_CHUNK_REVERSE_CUMSUM_STEP,
            ut_bwd_dattn0: GDN_UT_BWD_DATTN0,
            ut_bwd_dtmat: GDN_UT_BWD_DTMAT,
            mask_strict_lower_bwd: GDN_MASK_STRICT_LOWER_BWD,
            decay_mask_bwd: GDN_DECAY_MASK_BWD,
            decay_scale_bwd: GDN_DECAY_SCALE_BWD,
            decay_scale_bwd_last: GDN_DECAY_SCALE_BWD_LAST,
            state_decay_bwd_dscale: GDN_STATE_DECAY_BWD_DSCALE,
        },
    )
}

impl Horizon {
    fn stack_gdn_shape(&self, k: &StackBufs) -> GdnShape {
        let (chunk, t) = self.cfg.stack_layout();
        let hd = self.cfg.d_model / self.cfg.n_heads;
        GdnShape {
            b: k.subjects,
            h: self.cfg.n_heads,
            t,
            dk: hd,
            dv: hd,
            chunk,
        }
    }

    /// Token-major `[B, T, H, dim]` into chunk-major `[chunks, B, H, C, dim]`
    /// (`to_chunk_major` 1) or back (0).
    fn gdn_permute(
        &self,
        shape: &GdnShape,
        src: &DeviceBuffer,
        dst: &DeviceBuffer,
        dim: u32,
        to_chunk_major: u32,
    ) -> Step {
        let n = shape.b * shape.h * shape.t * dim;
        self.gpu.step(
            GDN_LAYOUT_PERMUTE,
            &[src, dst],
            &[
                shape.b,
                shape.h,
                shape.n_chunks(),
                shape.chunk,
                dim,
                to_chunk_major,
            ],
            n,
        )
    }

    /// The forward of the stack: the query token and the visits through the
    /// blocks, and the query row's state into `z`.
    pub(super) fn stack_forward_steps(&self, k: &StackBufs) -> Vec<Step> {
        let g = &self.gpu;
        let i = &self.inp;
        let (d, b, s) = (self.cfg.d_model, self.b, self.sets);
        let rows = b * self.cfg.stack_layout().1;
        let mut steps = vec![
            g.step(EMBED, &[&i.summary_rows, &self.xf, &k.u], &[d, s], s * d),
            g.step(
                ROW_SCATTER,
                &[&k.seq_rows, &k.u, &k.x[0]],
                &[s, d, rows],
                s * d,
            ),
            g.step(
                EMBED,
                &[&k.query_index, self.w("visit.query"), &k.qe],
                &[d, b],
                b * d,
            ),
            g.step(
                ROW_SCATTER,
                &[&k.query_rows, &k.qe, &k.x[0]],
                &[b, d, rows],
                b * d,
            ),
        ];
        for (l, blk) in k.blocks.iter().enumerate() {
            let pn = |name: &str| format!("visit.blocks.{l}.{name}");
            steps.push(block::layernorm_fwd(
                g,
                &LN_IDS,
                &k.x[l],
                self.w(&pn("ln1.weight")),
                self.w(&pn("ln1.bias")),
                &blk.ln1_out,
                d,
                rows,
                LN_EPS,
            ));
            match &blk.mixer {
                MixerBufs::Attention(a) => steps.extend(self.stack_attention_forward(k, blk, a, &pn)),
                MixerBufs::Gdn(gb) => steps.extend(self.stack_gdn_forward(k, blk, gb, &pn)),
            }
            steps.push(g.step(
                ADD2,
                &[&k.x[l], &k.mix, &blk.xmid],
                &[rows * d],
                rows * d,
            ));
            steps.extend(self.mlp_forward_steps(
                &format!("visit.blocks.{l}."),
                rows,
                &blk.xmid,
                (&blk.ln2_out, &blk.up_pre, &blk.up, &k.ffn_out),
                &k.x[l + 1],
            ));
        }
        steps.extend([
            block::layernorm_fwd(
                g,
                &LN_IDS,
                k.x.last().expect("a stack has a state"),
                self.w("visit.ln_f.weight"),
                self.w("visit.ln_f.bias"),
                &k.xf,
                d,
                rows,
                LN_EPS,
            ),
            g.step(EMBED, &[&k.query_rows, &k.xf, &self.z], &[d, b], b * d),
        ]);
        steps
    }

    fn stack_attention_forward(
        &self,
        k: &StackBufs,
        blk: &Block,
        a: &AttnBlock,
        pn: &dyn Fn(&str) -> String,
    ) -> Vec<Step> {
        let g = &self.gpu;
        let (d, b, h) = (self.cfg.d_model, self.b, self.cfg.n_heads);
        let t = self.cfg.stack_layout().1;
        let rows = b * t;
        let mut steps = vec![
            self.mm(
                &blk.ln1_out,
                self.w(&pn("attn.qkv.weight")),
                &a.qkv,
                rows,
                d,
                3 * d,
            ),
            g.step(
                BIAS_ADD,
                &[&a.qkv, self.w(&pn("attn.qkv.bias"))],
                &[rows, 3 * d],
                rows * 3 * d,
            ),
        ];
        steps.extend(self.rope(&k.pos, &a.qkv, rows, 1.0));
        let mut attn = block::bidir_fwd(g, &BIDIR, &self.stack_bidir(), &a.qkv, &a.scores, &a.probs, &a.ctx);
        // Unused visit slots and padding are never keys.
        attn.insert(
            1,
            g.step(
                KEYPAD,
                &[&k.keep, &a.scores],
                &[b, h, t],
                b * h * t * t,
            ),
        );
        steps.extend(attn);
        steps.extend([
            self.mm(&a.ctx, self.w(&pn("attn.out.weight")), &k.mix, rows, d, d),
            g.step(
                BIAS_ADD,
                &[&k.mix, self.w(&pn("attn.out.bias"))],
                &[rows, d],
                rows * d,
            ),
        ]);
        steps
    }

    fn stack_bidir(&self) -> Bidir {
        let d = self.cfg.d_model;
        Bidir {
            b: self.b,
            t: self.cfg.stack_layout().1,
            n_heads: self.cfg.n_heads,
            head_dim: d / self.cfg.n_heads,
            stride: 3 * d,
            q_off: 0,
            k_off: d,
            v_off: 2 * d,
        }
    }

    fn stack_gdn_forward(
        &self,
        k: &StackBufs,
        blk: &Block,
        gb: &GdnBlock,
        pn: &dyn Fn(&str) -> String,
    ) -> Vec<Step> {
        let g = &self.gpu;
        let (d, h) = (self.cfg.d_model, self.cfg.n_heads);
        let shape = self.stack_gdn_shape(k);
        let rows = shape.b * shape.t;
        let hd = d / h;
        let (ids, bwd_ids) = gdn_ids();
        let mut steps = vec![
            self.mm(&blk.ln1_out, self.w(&pn("gdn.q.weight")), &gb.q_pre, rows, d, d),
            self.mm(&blk.ln1_out, self.w(&pn("gdn.k.weight")), &gb.k_pre, rows, d, d),
            self.mm(&blk.ln1_out, self.w(&pn("gdn.v.weight")), &gb.v, rows, d, d),
            self.mm(&blk.ln1_out, self.w(&pn("gdn.beta.weight")), &gb.b_pre, rows, d, h),
            g.step(
                BIAS_ADD,
                &[&gb.b_pre, self.w(&pn("gdn.beta.bias"))],
                &[rows, h],
                rows * h,
            ),
            // q and k are unit vectors per head: the delta rule's erase step
            // is a projection only for |k| = 1.
            g.step(
                L2NORM_SCALE,
                &[&gb.q_pre, &k.ones, &gb.q_n],
                &[rows * h, hd, f(L2_EPS)],
                rows * d,
            ),
            g.step(
                L2NORM_SCALE,
                &[&gb.k_pre, &k.ones, &gb.k_n],
                &[rows * h, hd, f(L2_EPS)],
                rows * d,
            ),
            g.step(
                GDN_GAP_GATE,
                &[&gb.b_pre, &k.dt, self.w(&pn("gdn.rate")), &gb.g, &gb.beta],
                &[rows, h],
                rows * h,
            ),
            self.gdn_permute(&shape, &gb.q_n, &gb.q_cm, hd, 1),
            self.gdn_permute(&shape, &gb.k_n, &gb.k_cm, hd, 1),
            self.gdn_permute(&shape, &gb.v, &gb.v_cm, hd, 1),
            self.gdn_permute(&shape, &gb.g, &gb.g_cm, 1, 1),
            self.gdn_permute(&shape, &gb.beta, &gb.beta_cm, 1, 1),
        ];
        steps.extend(gdn_chunk_fwd_train(
            g,
            &ids,
            &bwd_ids,
            &shape,
            &gb.q_cm,
            &gb.k_cm,
            &gb.v_cm,
            &gb.g_cm,
            &gb.beta_cm,
            &gb.initial_state,
            &gb.saved.as_ref(),
            &gb.out_cm,
            &gb.final_state,
        ));
        steps.extend([
            self.gdn_permute(&shape, &gb.out_cm, &gb.out_tok, hd, 0),
            self.mm(&gb.out_tok, self.w(&pn("gdn.out.weight")), &k.mix, rows, d, d),
            g.step(
                BIAS_ADD,
                &[&k.mix, self.w(&pn("gdn.out.bias"))],
                &[rows, d],
                rows * d,
            ),
        ]);
        steps
    }

    /// The adjoint of [`Horizon::stack_forward_steps`]: `d_z` (complete) into
    /// the summary rows of `d_xf`, the query embedding's and every block's
    /// gradients.
    pub(super) fn stack_backward_steps(&self, k: &StackBufs) -> Vec<Step> {
        let g = &self.gpu;
        let i = &self.inp;
        let gr = |name: &str| self.ps.g(name);
        let (d, b, s) = (self.cfg.d_model, self.b, self.sets);
        let rows = b * self.cfg.stack_layout().1;
        let bn = s * self.cfg.max_tokens;
        let last = k.blocks.len();
        // Only the query rows of the final norm's output are read.
        let mut steps = vec![g.step(
            ROW_SCATTER,
            &[&k.query_rows, &self.d_z, &k.d_xf],
            &[b, d, rows],
            b * d,
        )];
        steps.extend(self.ln_backward_steps(
            &k.x[last],
            ("visit.ln_f.weight", "visit.ln_f.bias"),
            &k.d_xf,
            &k.d_x[last],
            rows,
        ));
        for (l, blk) in k.blocks.iter().enumerate().rev() {
            let pn = |name: &str| format!("visit.blocks.{l}.{name}");
            steps.extend(self.mlp_backward_steps(
                &format!("visit.blocks.{l}."),
                rows,
                &blk.xmid,
                (&blk.ln2_out, &blk.up_pre, &blk.up),
                &k.d_x[l + 1],
                (&k.d_up, &k.d_up_pre, &k.d_branch, &k.d_tmp),
                &k.d_xmid,
            ));
            // From the mixer's output gradient (d_xmid) to the gradient of
            // its input, the first LayerNorm's output (d_branch).
            match &blk.mixer {
                MixerBufs::Attention(a) => steps.extend(self.stack_attention_backward(k, blk, a, &pn)),
                MixerBufs::Gdn(gb) => steps.extend(self.stack_gdn_backward(k, blk, gb, &pn)),
            }
            steps.extend(self.ln_backward_steps(
                &k.x[l],
                (&pn("ln1.weight"), &pn("ln1.bias")),
                &k.d_branch,
                &k.d_tmp,
                rows,
            ));
            steps.push(g.step(
                ADD2,
                &[&k.d_xmid, &k.d_tmp, &k.d_x[l]],
                &[rows * d],
                rows * d,
            ));
        }
        steps.extend([
            g.step(EMBED, &[&k.seq_rows, &k.d_x[0], &k.d_u], &[d, s], s * d),
            g.step(EMBED, &[&k.query_rows, &k.d_x[0], &k.d_qe], &[d, b], b * d),
            g.step(
                EMB_BWD,
                &[&k.query_index, &k.d_qe, gr("visit.query")],
                &[b, d, 1],
                d,
            ),
            g.step(
                ROW_SCATTER,
                &[&i.summary_rows, &k.d_u, &self.d_xf],
                &[s, d, bn],
                s * d,
            ),
        ]);
        steps
    }

    fn stack_attention_backward(
        &self,
        k: &StackBufs,
        blk: &Block,
        a: &AttnBlock,
        pn: &dyn Fn(&str) -> String,
    ) -> Vec<Step> {
        let gr = |name: &str| self.ps.g(&pn(name));
        let (d, rows) = (self.cfg.d_model, self.b * self.cfg.stack_layout().1);
        let mut steps: Vec<Step> = self.bias_grad(&k.d_xmid, gr("attn.out.bias"), rows, d).into();
        steps.extend(self.mm_dw(&k.d_xmid, &a.ctx, gr("attn.out.weight"), rows, d, d));
        steps.push(self.mm_dx(
            &k.d_xmid,
            self.w(&pn("attn.out.weight")),
            &a.d_ctx,
            rows,
            d,
            d,
            0,
        ));
        steps.extend(block::bidir_bwd(
            &self.gpu,
            &BIDIR,
            &self.stack_bidir(),
            &a.qkv,
            &a.probs,
            &a.d_ctx,
            &k.d_scores,
            &k.d_qkv,
        ));
        steps.extend(self.rope(&k.pos, &k.d_qkv, rows, -1.0));
        steps.extend(self.bias_grad(&k.d_qkv, gr("attn.qkv.bias"), rows, 3 * d));
        steps.extend(self.mm_dw(&k.d_qkv, &blk.ln1_out, gr("attn.qkv.weight"), rows, d, 3 * d));
        steps.push(self.mm_dx(
            &k.d_qkv,
            self.w(&pn("attn.qkv.weight")),
            &k.d_branch,
            rows,
            d,
            3 * d,
            0,
        ));
        steps
    }

    fn stack_gdn_backward(
        &self,
        k: &StackBufs,
        blk: &Block,
        gb: &GdnBlock,
        pn: &dyn Fn(&str) -> String,
    ) -> Vec<Step> {
        let g = &self.gpu;
        let gr = |name: &str| self.ps.g(&pn(name));
        let w = |name: &str| self.w(&pn(name));
        let (d, h) = (self.cfg.d_model, self.cfg.n_heads);
        let shape = self.stack_gdn_shape(k);
        let rows = shape.b * shape.t;
        let hd = d / h;
        let (ids, bwd_ids) = gdn_ids();
        let x = &blk.ln1_out;
        // The output projection, then the recurrence's adjoint in chunk-major.
        let mut steps: Vec<Step> = self.bias_grad(&k.d_xmid, gr("gdn.out.bias"), rows, d).into();
        steps.extend(self.mm_dw(&k.d_xmid, &gb.out_tok, gr("gdn.out.weight"), rows, d, d));
        steps.extend([
            self.mm_dx(&k.d_xmid, w("gdn.out.weight"), &gb.d_out_tok, rows, d, d, 0),
            self.gdn_permute(&shape, &gb.d_out_tok, &gb.d_out_cm, hd, 1),
        ]);
        steps.extend(gdn_chunk_bwd(
            g,
            &ids,
            &bwd_ids,
            &shape,
            &gb.q_cm,
            &gb.k_cm,
            &gb.v_cm,
            &gb.beta_cm,
            &gb.saved.as_ref(),
            &gb.d_out_cm,
            &gb.d_final_state,
            &gb.bwd.as_ref(),
            &gb.d_q_cm,
            &gb.d_k_cm,
            &gb.d_v_cm,
            &gb.d_g_cm,
            &gb.d_beta_cm,
            &gb.d_initial_state,
        ));
        steps.extend([
            self.gdn_permute(&shape, &gb.d_q_cm, &gb.d_q_n, hd, 0),
            self.gdn_permute(&shape, &gb.d_k_cm, &gb.d_k_n, hd, 0),
            self.gdn_permute(&shape, &gb.d_v_cm, &gb.d_v, hd, 0),
            self.gdn_permute(&shape, &gb.d_g_cm, &gb.d_g, 1, 0),
            self.gdn_permute(&shape, &gb.d_beta_cm, &gb.d_beta, 1, 0),
            g.step(
                GDN_GAP_GATE_BWD,
                &[
                    &gb.b_pre,
                    &k.dt,
                    w("gdn.rate"),
                    &gb.d_g,
                    &gb.d_beta,
                    &gb.d_b_pre,
                    &gb.d_rate,
                ],
                &[rows, h],
                rows * h,
            ),
        ]);
        // The rate is one per head: every row's share summed.
        steps.extend(self.bias_grad(&gb.d_rate, gr("gdn.rate"), rows, h));
        steps.extend([
            g.step(
                L2NORM_SCALE_DX,
                &[&gb.q_pre, &k.ones, &gb.d_q_n, &gb.d_q_pre],
                &[rows * h, hd, f(L2_EPS)],
                rows * d,
            ),
            g.step(
                L2NORM_SCALE_DX,
                &[&gb.k_pre, &k.ones, &gb.d_k_n, &gb.d_k_pre],
                &[rows * h, hd, f(L2_EPS)],
                rows * d,
            ),
        ]);
        steps.extend(self.mm_dw(&gb.d_q_pre, x, gr("gdn.q.weight"), rows, d, d));
        steps.extend(self.mm_dw(&gb.d_k_pre, x, gr("gdn.k.weight"), rows, d, d));
        steps.extend(self.mm_dw(&gb.d_v, x, gr("gdn.v.weight"), rows, d, d));
        steps.extend(self.bias_grad(&gb.d_b_pre, gr("gdn.beta.bias"), rows, h));
        steps.extend(self.mm_dw(&gb.d_b_pre, x, gr("gdn.beta.weight"), rows, d, h));
        // The four projections all read the normalised input: their input
        // gradients add.
        steps.extend([
            self.mm_dx(&gb.d_q_pre, w("gdn.q.weight"), &k.d_branch, rows, d, d, 0),
            self.mm_dx(&gb.d_k_pre, w("gdn.k.weight"), &k.d_branch, rows, d, d, 1),
            self.mm_dx(&gb.d_v, w("gdn.v.weight"), &k.d_branch, rows, d, d, 1),
            self.mm_dx(&gb.d_b_pre, w("gdn.beta.weight"), &k.d_branch, rows, d, h, 1),
        ]);
        steps
    }
}
