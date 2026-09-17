// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The candidate-to-state head: one score per option, from the encoder's
//! hidden states.
//!
//! ```text
//! q        = cls(slot rows) @ Wq^T + bq                 [S, H]
//! kv       = state rows     @ Wkv^T + bkv               [R, 2H]  (k at 0, v at H)
//! ctx      = crossattn(q -> kv)                         [S, H]
//! out      = LN(ctx + q)                                [S, H]
//! score_i  = out_i . w + b                              [S]
//! ```
//!
//! **The state is encoded once; the options query it.** Every option's slot
//! attends over every state token, including tokens from other windows, which
//! is what puts cross-window integration here rather than in the encoder.
//! Nothing about option `i`'s score depends on which other options were
//! supplied except through the softmax that normalizes them, so a caller may
//! change the option set between requests without re-encoding the state.
//!
//! **The softmax is not here.** `score` is the last device-side value; the
//! grouping of options into questions, the softmax within each, the loss and
//! its gradient are host arithmetic over at most a few hundred numbers. That
//! keeps the option space genuinely runtime-defined - no kernel has an opinion
//! about how many options a question has - at a cost that does not register
//! next to a six-layer encoder.
//!
//! **Slots are read at their `[CLS]` row, not mean-pooled.** A gather has an
//! existing adjoint (`emb_bwd`); a mean over variable-length spans would need
//! a new kernel pair. `[CLS]` is what BERT trains that position for, and the
//! slot template starts with it. Mean pooling is what the released
//! sentence-transformer head uses and is the natural thing to ablate against
//! once a pooling kernel exists.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};

use crate::config::EncoderConfig;


/// Parameter shapes, in one place for the store, the importer and the init.
pub fn tensor_manifest(cfg: &EncoderConfig) -> Vec<(String, Vec<usize>)> {
    let h = cfg.d_model as usize;
    vec![
        ("head.wq.weight".into(), vec![h, h]),
        ("head.wq.bias".into(), vec![h]),
        // K and V in ONE matrix, so the cross-attention's fused `[R, 2H]`
        // memory comes out of a single GEMM in the layout its kernels read.
        ("head.wkv.weight".into(), vec![2 * h, h]),
        ("head.wkv.bias".into(), vec![2 * h]),
        ("head.ln.weight".into(), vec![h]),
        ("head.ln.bias".into(), vec![h]),
        ("head.score.weight".into(), vec![1, h]),
        ("head.score.bias".into(), vec![1]),
    ]
}

struct Bwd {
    d_score: DeviceBuffer,
    d_out: DeviceBuffer,
    d_sum: DeviceBuffer,
    d_q: DeviceBuffer,
    /// `d_q` plus the residual's share. A separate buffer because `add2`
    /// refuses an output that aliases an input.
    d_q_total: DeviceBuffer,
    d_kv: DeviceBuffer,
    d_scores_attn: DeviceBuffer,
    d_cls: DeviceBuffer,
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    steps: Vec<Step>,
}

pub struct Head {
    gpu: Gpu,
    k: crate::kern::Ids,
    cfg: EncoderConfig,
    pub ps: ParamStore,
    cap_rows: u32,
    cap_slots: u32,
    /// Rows of encoder state this call attends over: `[0, state_rows)`.
    state_rows: u32,
    /// The slots' `[CLS]` row indices, one per option.
    n_slots: u32,
    cls_rows: DeviceBuffer,
    cls: DeviceBuffer,
    q: DeviceBuffer,
    kv: DeviceBuffer,
    scores_attn: DeviceBuffer,
    probs: DeviceBuffer,
    ctx: DeviceBuffer,
    sum: DeviceBuffer,
    out: DeviceBuffer,
    score: DeviceBuffer,
    steps: Vec<Step>,
    bwd: Option<Bwd>,
}

impl Head {
    pub fn new_on(
        gpu: Gpu,
        cfg: EncoderConfig,
        cap_rows: u32,
        cap_slots: u32,
        init: &HashMap<String, Vec<f32>>,
        train: bool,
    ) -> Head {
        let role = if train { Role::Trainable } else { Role::Frozen };
        let roles: Vec<(String, usize, Role)> = tensor_manifest(&cfg)
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), role))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);
        let h = cfg.d_model as u64;
        let (r, s) = (cap_rows as u64, cap_slots as u64);
        let slab = cfg.n_heads as u64 * s * r;
        let st = |w: u64| gpu.storage(w);
        let k = crate::kern::Ids::resolve(&gpu);
        let mut head = Head {
            k,
            cap_rows,
            cap_slots,
            state_rows: cap_rows,
            n_slots: cap_slots,
            cls_rows: gpu.buffer("cls_rows", s * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST),
            cls: st(s * h),
            q: st(s * h),
            kv: st(r * 2 * h),
            scores_attn: st(slab),
            probs: st(slab),
            ctx: st(s * h),
            sum: st(s * h),
            out: st(s * h),
            score: st(s),
            steps: Vec::new(),
            bwd: None,
            gpu,
            cfg,
            ps,
        };
        if train {
            head.bwd = Some(Bwd {
                d_score: head.gpu.buffer(
                    "d_score",
                    s * 4,
                    gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST,
                ),
                d_out: head.gpu.storage(s * h),
                d_sum: head.gpu.storage(s * h),
                d_q: head.gpu.storage(s * h),
                d_q_total: head.gpu.storage(s * h),
                d_kv: head.gpu.storage(r * 2 * h),
                d_scores_attn: head.gpu.storage(slab),
                d_cls: head.gpu.storage(s * h),
                mean: head.gpu.storage(s),
                inv: head.gpu.storage(s),
                steps: Vec::new(),
            });
        }
        head
    }

    fn w(&self, n: &str) -> &DeviceBuffer {
        self.ps.w(n)
    }

    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        block::pick_gemm(m as usize, n as usize, self.k.matmul, self.k.matmul_reg3, false)
    }

    /// Point the head at one call: how many encoder rows are state, and which
    /// row each slot's `[CLS]` sits on.
    pub fn set_call(
        &mut self,
        hidden: &DeviceBuffer,
        d_hidden_out: Option<&DeviceBuffer>,
        state_rows: u32,
        cls_rows: &[u32],
    ) {
        assert!(state_rows <= self.cap_rows, "{state_rows} state rows > capacity {}", self.cap_rows);
        assert!(cls_rows.len() <= self.cap_slots as usize, "{} slots > capacity {}", cls_rows.len(), self.cap_slots);
        assert!(!cls_rows.is_empty(), "a call needs at least one option");
        self.gpu.write(&self.cls_rows, cls_rows);
        self.state_rows = state_rows;
        self.n_slots = cls_rows.len() as u32;
        // Rebuilt unconditionally: the step list holds the encoder's hidden
        // buffer, so a caller that swapped encoders would otherwise keep
        // dispatching against the old one.
        self.steps = self.build_steps(hidden);
        self.rebuild_bwd(hidden, d_hidden_out);
    }

    /// Run the head over `hidden` (the encoder's `[rows, H]` output) and
    /// return one raw score per option, in the order the slots were packed.
    pub fn forward(&self) -> Vec<f32> {
        self.gpu.submit(&[], &self.steps);
        self.gpu.read(&self.score, self.n_slots as usize)
    }

    /// Block until this device has finished what it was given.
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    /// L2 norm of each forward stage, for localizing a dead path.
    ///
    /// A stage that reads zero when the one before it does not is where the
    /// wiring broke; every stage reading zero means the encoder handed over
    /// nothing. Cheap enough to call from a test, never on a hot path.
    pub fn stage_norms(&self) -> Vec<(&'static str, f32)> {
        let h = self.cfg.d_model as usize;
        let s = self.n_slots as usize;
        let r = self.state_rows as usize;
        let n = |b: &DeviceBuffer, len: usize| -> f32 {
            self.gpu.read(b, len).iter().map(|v| v * v).sum::<f32>().sqrt()
        };
        vec![
            ("cls", n(&self.cls, s * h)),
            ("q", n(&self.q, s * h)),
            ("kv", n(&self.kv, r * 2 * h)),
            ("probs", n(&self.probs, self.cfg.n_heads as usize * s * r)),
            ("ctx", n(&self.ctx, s * h)),
            ("sum", n(&self.sum, s * h)),
            ("out", n(&self.out, s * h)),
            ("score", n(&self.score, s)),
        ]
    }

    fn build_steps(&self, hidden: &DeviceBuffer) -> Vec<Step> {
        let g = &self.gpu;
        let (h, hd) = (self.cfg.d_model, self.cfg.head_dim());
        let (s, r, heads) = (self.n_slots, self.state_rows, self.cfg.n_heads);
        let ln = block::LayerNormIds::resolve(g, self.k.layernorm, self.k.ln_stats, self.k.layernorm_dx);
        let mut st = vec![
            // Slot representations: gather each slot's [CLS] row out of the
            // encoder's hidden states. `embed` Params: [width, rows].
            g.step(self.k.embed, &[&self.cls_rows, hidden, &self.cls], &[h, s], s * h),
        ];
        let (mk, mt) = self.gemm(s, h);
        st.push(g.step(mk, &[&self.cls, self.w("head.wq.weight"), &self.q], &[s, h, h], mt));
        st.push(g.step(self.k.bias_add, &[&self.q, self.w("head.wq.bias")], &[s, h], s * h));
        // The state's keys and values, from row 0 - the packer puts every
        // window first and contiguously.
        let (mk, mt) = self.gemm(r, 2 * h);
        st.push(g.step(mk, &[hidden, self.w("head.wkv.weight"), &self.kv], &[r, h, 2 * h], mt));
        st.push(g.step(self.k.bias_add, &[&self.kv, self.w("head.wkv.bias")], &[r, 2 * h], r * 2 * h));

        // Cross-attention: every option queries every state token.
        let p_qk = [1, heads, s, r, hd, h, 2 * h, 0, 0];
        let p_v = [1, heads, s, r, hd, 2 * h, h, h];
        st.push(g.step(self.k.scores_cross, &[&self.q, &self.kv, &self.scores_attn], &p_qk, heads * s * r));
        st.push(g.step(self.k.softmax_cross, &[&self.scores_attn, &self.probs], &[1, heads, s, r], heads * s));
        st.push(g.step(self.k.apply_cross, &[&self.probs, &self.kv, &self.ctx], &p_v, heads * s * hd));

        // Residual onto the option's own query, then normalize, then score.
        st.push(g.step(self.k.add2, &[&self.ctx, &self.q, &self.sum], &[s * h], s * h));
        st.push(block::layernorm_fwd(
            g,
            &ln,
            &self.sum,
            self.w("head.ln.weight"),
            self.w("head.ln.bias"),
            &self.out,
            h,
            s,
            self.cfg.eps,
        ));
        let (mk, mt) = self.gemm(s, 1);
        st.push(g.step(mk, &[&self.out, self.w("head.score.weight"), &self.score], &[s, h, 1], mt));
        st.push(g.step(self.k.bias_add, &[&self.score, self.w("head.score.bias")], &[s, 1], s));
        st
    }

    fn rebuild_bwd(&mut self, hidden: &DeviceBuffer, d_hidden_out: Option<&DeviceBuffer>) {
        if self.bwd.is_none() {
            return;
        }
        // A trainable head with nowhere to put its hidden-state gradient would
        // silently train the head alone, so the absence is refused rather than
        // defaulted.
        let out = d_hidden_out.expect("a trainable head needs the encoder's seed buffer");
        let steps = self.build_bwd_steps(hidden, out);
        if let Some(b) = &mut self.bwd {
            b.steps = steps;
        }
    }

    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Seed with `dL/d(score)` (one per option, host-computed by the loss) and
    /// run the reverse pass, which writes the encoder's seed buffer in place.
    pub fn backward(&self, d_score: &[f32]) {
        let b = self.bwd.as_ref().expect("backward on an inference head");
        assert_eq!(d_score.len(), self.n_slots as usize, "one score gradient per option");
        self.gpu.write_f32(&b.d_score, d_score);
        self.gpu.submit(&[], &b.steps);
    }

    fn build_bwd_steps(&self, hidden: &DeviceBuffer, d_hidden_out: &DeviceBuffer) -> Vec<Step> {
        let g = &self.gpu;
        let b = self.bwd.as_ref().expect("training mode only");
        let (h, hd) = (self.cfg.d_model, self.cfg.head_dim());
        let (s, r, heads) = (self.n_slots, self.state_rows, self.cfg.n_heads);
        let ln = block::LayerNormIds::resolve(g, self.k.layernorm, self.k.ln_stats, self.k.layernorm_dx);
        let gr = |n: &str| self.ps.g(n);
        let dw = |m: u32, k: u32| block::pick_gemm(m as usize, k as usize, self.k.matmul_dw, self.k.matmul_dw_reg, false);
        let dx = |m: u32, k: u32| block::pick_gemm(m as usize, k as usize, self.k.matmul_dx, self.k.matmul_dx_reg, false);
        let mut st = Vec::new();

        // ---- scorer ----
        st.push(g.step(self.k.bias_grad, &[&b.d_score, gr("head.score.bias")], &[s, 1], 1));
        let (k, t) = dw(1, h);
        st.push(g.step(k, &[&b.d_score, &self.out, gr("head.score.weight")], &[s, h, 1], t));
        let (k, t) = dx(s, h);
        st.push(g.step(k, &[&b.d_score, self.w("head.score.weight"), &b.d_out], &[s, h, 1, 0], t));

        // ---- LayerNorm ----
        st.push(block::ln_stats_fwd(g, &ln, &self.sum, &b.mean, &b.inv, h, s, self.cfg.eps));
        st.push(g.step(self.k.ln_dgamma, &[&b.d_out, &self.sum, &b.mean, &b.inv, gr("head.ln.weight")], &[h, s], h));
        st.push(g.step(self.k.ln_dbeta, &[&b.d_out, gr("head.ln.bias")], &[h, s], h));
        st.push(block::layernorm_dx_bwd(g, &ln, &self.sum, self.w("head.ln.weight"), &b.d_out, &b.d_sum, h, s, self.cfg.eps));

        // ---- cross-attention ----
        // `sum = ctx + q` passes the gradient through unchanged, so `d_ctx` IS
        // `d_sum` and the attention backward reads it directly rather than
        // through a copy.
        let p_v = [1, heads, s, r, hd, 2 * h, h, h];
        let p_qk = [1, heads, s, r, hd, h, 2 * h, 0, 0];
        // `acc_flag = 0`: one chunk covers every query row here, so dk and dv
        // ASSIGN and `d_kv` needs no clear.
        let mut p_qk_acc = [0u32; 10];
        p_qk_acc[..9].copy_from_slice(&p_qk);
        let mut p_v_acc = [0u32; 9];
        p_v_acc[..8].copy_from_slice(&p_v);
        st.push(g.step(self.k.dscores_cross, &[&b.d_sum, &self.kv, &self.probs, &b.d_scores_attn], &p_v, heads * s));
        st.push(g.step(self.k.dq_cross, &[&b.d_scores_attn, &self.kv, &b.d_q], &p_qk, heads * s * hd));
        st.push(g.step(self.k.dk_cross_acc, &[&b.d_scores_attn, &self.q, &b.d_kv], &p_qk_acc, heads * r * hd));
        st.push(g.step(self.k.dv_cross_acc, &[&self.probs, &b.d_sum, &b.d_kv], &p_v_acc, heads * r * hd));
        // The query is used twice - as the attention's query and as the
        // residual - so its gradient is the sum of both paths.
        st.push(g.step(self.k.add2, &[&b.d_q, &b.d_sum, &b.d_q_total], &[s * h], s * h));

        // ---- projections ----
        st.push(g.step(self.k.bias_grad, &[&b.d_q_total, gr("head.wq.bias")], &[s, h], h));
        let (k, t) = dw(h, h);
        st.push(g.step(k, &[&b.d_q_total, &self.cls, gr("head.wq.weight")], &[s, h, h], t));
        let (k, t) = dx(s, h);
        st.push(g.step(k, &[&b.d_q_total, self.w("head.wq.weight"), &b.d_cls], &[s, h, h, 0], t));

        st.push(g.step(self.k.bias_grad, &[&b.d_kv, gr("head.wkv.bias")], &[r, 2 * h], 2 * h));
        let (k, t) = dw(2 * h, h);
        st.push(g.step(k, &[&b.d_kv, hidden, gr("head.wkv.weight")], &[r, h, 2 * h], t));
        let (k, t) = dx(r, h);
        st.push(g.step(k, &[&b.d_kv, self.w("head.wkv.weight"), d_hidden_out], &[r, h, 2 * h, 0], t));

        // The [CLS] gather's adjoint scatters each slot's grad back onto its
        // own row of `d_hidden`, on top of what the key/value path assigned.
        // `emb_bwd` ACCUMULATES, which is why that assign comes first.
        st.push(g.step(self.k.emb_bwd, &[&self.cls_rows, &b.d_cls, d_hidden_out], &[s, h, self.cap_rows], self.cap_rows * h));
        st
    }
}
