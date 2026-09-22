// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The ModernBERT encoder's forward pass.
//!
//! ```text
//! x[0] = LN_nobias(embed(ids, "tok.weight"), "emb_norm.weight")
//!
//! per layer l:
//!   xn   = has_attn_norm(l) ? LN_nobias(x[l], "blocks.l.attn_norm.weight") : x[l]
//!   qkv  = xn @ Wqkv^T                                    [rows, 3H], no bias
//!   RoPE applied in place to q and k, PER SPAN (see below), theta chosen by
//!     layer_types[l] (Full -> rope_theta_full, Local -> rope_theta_local)
//!   ctx  = bidirectional self-attention, per span:
//!            Full  -> block::chunked_bidir_fwd (unwindowed)
//!            Local -> block::chunked_bidir_fwd_win(window: cfg.window)
//!   attn_out = ctx @ Wo^T                                 [rows, H], no bias
//!   res1 = x[l] + attn_out                                PRE-LN residual
//!   xn2  = LN_nobias(res1, "blocks.l.mlp_norm.weight")
//!   wi_u, wi_v = xn2 @ Wi_u^T, xn2 @ Wi_v^T                each [rows, d_ff]
//!     (Wi_u/Wi_v are the two halves of the ONE fused "mlp.wi.weight"
//!     [2*d_ff, H] tensor, split by OUTPUT ROW - see `mlp_step`'s own note on
//!     why that split needs no new kernel)
//!   h    = gelu_erf(wi_u) * wi_v                          GeGLU, see below
//!   mlp_out = h @ Wo^T                                    [rows, H], no bias
//!   x[l+1] = res1 + mlp_out
//!
//! hidden = LN_nobias(x[n_layers], "final_norm.weight")
//! ```
//!
//! **Packed spans, not padding** - the same convention `crates/decide`
//! documents and for the same reason (a padded bidirectional encoder is
//! subtly wrong: an unmasked pad position leaks into every real position's
//! attention unless something remembers to mask it every time; packing
//! removes the pad rows instead).
//!
//! **GeGLU's chunk order is load-bearing and easy to get backwards.**
//! Verified against the installed `transformers` 5.15 source
//! (`ModernBertMLP.forward`): `input, gate = Wi(x).chunk(2, dim=-1)`, then
//! `Wo(act(input) * gate)` - activation lands on the FIRST half, the SECOND
//! half is the raw multiplier. This module names the two halves `wi_u`
//! (activated) and `wi_v` (raw) to keep that straight; getting them backwards
//! produces a plausible-looking but wrong forward, exactly what
//! `tests/parity.rs` exists to catch.
//!
//! **RoPE must be dispatched per span, not once over the whole buffer.**
//! `block::rope_fwd` (the convenience wrapper) hardcodes `base_off: 0` and
//! assumes row 0 of the bound buffer is every sequence's start - true for
//! `crates/lfm2`'s fixed-length batching, false here, where packed spans of
//! different lengths share one buffer. `rope_base.wgsl`'s own Params compute
//! `pos = row % p.tcols`, so a dispatch must bind a VIEW sliced to one span's
//! rows with `tcols` set to THAT span's length - see `rope_span` below, which
//! hand-builds the dispatch with `Gpu::step_sliced` rather than calling
//! `block::rope_fwd`, mirroring the same slicing convention
//! `block::chunked_bidir_fwd` already uses for `qkv` (`(row0 * stride, 0)`
//! word offsets).
//!
//! **Attention rung choice, deliberately simple for this milestone.** Full
//! layers use the plain materialized `block::chunked_bidir_fwd`; local layers
//! MUST use `block::chunked_bidir_fwd_win` - the fused `flash_attn_bidir_spans`
//! kernel has no window support yet (a separate follow-up to Laya M1), so
//! selecting it for a local layer would silently attend the whole span. Both
//! rungs run with no key-minor transpose optimisation (`km: None`) - a
//! deliberate simplicity choice for a first correct forward, not a fused-path
//! selection ladder like `crates/decide`'s; see `crates/decide/src/model.rs`
//! for that shape if a later milestone wants the throughput.
//!
//! # Backward (Laya M5)
//!
//! [`ModernBert::new_train_on`] builds the exact adjoint of [`ModernBert::
//! build_steps`], mirroring `crates/decide/src/model.rs`'s `prepare_reverse`/
//! `seed_buf`/`backward_seeded` split byte-for-byte - but the reverse walk is
//! ONE STEP LONGER than `decide::Encoder`'s own: `hidden_buf()` (what a head
//! reads) is `final_norm(x[n_layers])`, not `x[n_layers]` itself, so the very
//! first backward step undoes `final_norm` (writing `dx[n_layers]`) BEFORE
//! entering the same per-layer loop `decide::Encoder::build_bwd_steps` walks.
//! [`ModernBert::seed_buf`] therefore returns the PRE-`final_norm`-undo
//! buffer (`hidden`'s own gradient), not `dx[n_layers]`.
//!
//! **No new kernel needed for the no-bias LayerNorm's backward**, contrary to
//! this crate's own M2-era plan note (which predicted a `layernorm_nobias_dx`
//! kernel would be needed and turned out to be wrong once actually checked):
//! `ln_stats.wgsl`/`layernorm_dx.wgsl`/`layernorm_dgamma.wgsl` never bind a
//! `beta` buffer at all - LayerNorm's gradient w.r.t. `x` and w.r.t. `gamma`
//! do not depend on whether a bias exists (a bias is a constant additive
//! offset on `y`; its own gradient is a separate, distinct kernel,
//! `layernorm_dbeta`, which the no-bias trunk simply never calls). The
//! existing biased-LN backward primitives are therefore ALREADY the correct
//! no-bias ones, unmodified - see `crate::kern::PIPELINES`'s own note.
//!
//! **GeGLU's backward composes from two existing kernels**, exactly as the
//! plan predicted: `h = gelu_erf(wi_u) * wi_v`, so `d_wi_u_act = mul(d_h,
//! wi_v)`, `d_wi_v = mul(d_h, h_act)` (the SAME `mul` kernel, swapped
//! operand), then `d_wi_u = gelu_erf_bwd(wi_u, d_wi_u_act)`.
//!
//! **RoPE's backward is dispatched per span**, mirroring `rope_span`'s own
//! forward dispatch exactly (same `Gpu::step_sliced` slicing, same `tcols ==
//! len` convention) but against `k.rope_base_bwd` instead of `k.rope_base` -
//! see [`ModernBert::rope_span_bwd`].
//!
//! **Windowed attention's backward needs ONE new Rust function
//! (`block::chunked_bidir_bwd_win`) and ZERO new kernels** - confirmed by a
//! dedicated finite-difference test
//! (`crates/model/tests/chunked_bidir_bwd_win.rs`), not just the plan's own
//! reasoning: it recomputes scores/probs through the WINDOWED forward kernel
//! and feeds them into the SAME unwindowed `CrossBwdIds` gradient kernels
//! `chunked_bidir_bwd` uses, which is sound because softmax's adjoint is
//! already exactly zero wherever the probability is zero.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};
use std::collections::HashMap;

use crate::config::{LayerAttn, ModernBertConfig};

/// Attention-slab budget for the chunked path - same reasoning as
/// `crates/decide`'s own `SLAB_BUDGET`: the `[heads, chunk, len]` score and
/// probability slabs are sized against it, so `chunk` falls as the longest
/// span grows and the allocation stays bounded whatever a caller asks for.
const SLAB_BUDGET: u64 = 256 << 20;

/// WebGPU's `min_storage_buffer_offset_alignment`. Both the per-span RoPE
/// dispatch and the attention dispatch bind a VIEW of a packed buffer
/// starting at a span's first row, and a bound offset must be a multiple of
/// this - a hardware binding rule, not a tunable.
const BIND_ALIGN: u64 = 256;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

struct LayerBufs {
    /// Fused `[rows, 3H]` - q at 0, k at H, v at 2H. RoPE rotates q/k IN
    /// PLACE in this same buffer, per span.
    qkv: DeviceBuffer,
    /// `has_attn_norm(l)` output, unused (and unallocated cost avoided by
    /// reuse - see `build_steps`) on layer 0, whose `qkv` reads `x[l]`
    /// directly.
    xn: DeviceBuffer,
    ctx: DeviceBuffer,
    attn_out: DeviceBuffer,
    /// `x[l] + attn_out`, the PRE-LN residual the MLP branch normalizes.
    res1: DeviceBuffer,
    xn2: DeviceBuffer,
    /// The two halves of `xn2 @ Wi^T`, `[rows, d_ff]` each - see the module
    /// doc's GeGLU note for why each is its own GEMM against a sliced half
    /// of the one fused `mlp.wi.weight` rather than one `[rows, 2*d_ff]`
    /// dispatch plus a split kernel.
    wi_u: DeviceBuffer,
    wi_v: DeviceBuffer,
    h_act: DeviceBuffer,
    h: DeviceBuffer,
    mlp_out: DeviceBuffer,
}

/// Reverse-pass buffers and the recorded backward step list - allocated only
/// by [`ModernBert::new_train_on`], mirroring `crates/decide/src/model.rs`'s
/// own `Bwd`. Every entry is a GRADIENT; the activations the backward reads
/// are the forward's own cached buffers (`LayerBufs`), nothing recomputed on
/// the host. The per-layer scratch is shared across layers (the reverse walk
/// holds one layer's intermediates live at a time); only `dx` is per layer.
struct Bwd {
    /// `d_hidden` is the objective's own seed - the gradient of
    /// [`ModernBert::hidden_buf`], i.e. of `final_norm(x[n_layers])`. COPY_DST
    /// because a caller (`laya::LayaHead::backward`) writes it directly.
    /// **Not** `dx[n_layers]` - see the module doc's "Backward" section for
    /// why this crate's reverse walk is one step longer than `decide`'s own.
    d_hidden: DeviceBuffer,
    /// `dx[i]` is the grad of `x[i]`.
    dx: Vec<DeviceBuffer>,
    d_res1: DeviceBuffer,
    d_h: DeviceBuffer,
    d_h_act: DeviceBuffer,
    d_wi_u: DeviceBuffer,
    d_wi_v: DeviceBuffer,
    d_xn2: DeviceBuffer,
    /// Scratch for a pre-norm's `dx` output before it re-joins the residual -
    /// reused for BOTH `mlp_norm`'s and `attn_norm`'s backward within one
    /// layer's steps (the two uses never overlap: each is consumed by an
    /// `add2` immediately after it is written).
    d_tmp: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_qkv: DeviceBuffer,
    /// `[heads, chunk, max_span]` scratch for the attention backward's
    /// softmax-jacobian output - a SEPARATE buffer from the forward's own
    /// `scores`/`probs` slabs (which the backward's per-chunk recompute
    /// reads AND writes within the same dispatch sequence; aliasing the
    /// gradient onto one of them would race that recompute).
    d_scores: DeviceBuffer,
    /// Grad flowing out of the qkv GEMM's `dx` - either the attn_norm's
    /// output gradient (most layers) or directly `x[0]`'s gradient (layer 0,
    /// which has no attn_norm to undo).
    d_attn_in: DeviceBuffer,
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    steps: Vec<Step>,
}

/// The ModernBERT encoder. `Role::Frozen` by [`ModernBert::new_on`]
/// (inference only); [`ModernBert::new_train_on`] builds the SAME forward
/// (`Role::Trainable` weights) plus the seeded backward - see the module
/// doc's "Backward" section.
pub struct ModernBert {
    pub gpu: Gpu,
    k: crate::kern::Ids,
    pub cfg: ModernBertConfig,
    pub ps: ParamStore,
    /// Capacity in rows; a call may use fewer.
    cap_rows: u32,
    rows: u32,
    spans: Vec<(u32, u32)>,
    chunk: u32,
    ids: DeviceBuffer,
    e_tok: DeviceBuffer,
    /// `x[0]` = embedding output after its LayerNorm, `x[i+1]` = layer `i`'s
    /// output. `x[n_layers]` is the PRE-`final_norm` hidden state; `hidden`
    /// is the post-`final_norm` one the head actually reads.
    x: Vec<DeviceBuffer>,
    hidden: DeviceBuffer,
    layers: Vec<LayerBufs>,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    steps: Vec<Step>,
    bwd: Option<Bwd>,
    /// Set when a batch arrives that the recorded reverse pass no longer
    /// describes - see `crates/decide/src/model.rs::Encoder`'s own field of
    /// the same name for the full reasoning (a decision loop cannot afford
    /// to record a backward it never runs).
    reverse_stale: bool,
}

impl ModernBert {
    /// Build on an existing device, sized for at most `cap_rows` packed
    /// tokens and a longest span of `max_span`. Every parameter is `Frozen`.
    pub fn new_on(
        gpu: Gpu,
        cfg: ModernBertConfig,
        cap_rows: u32,
        max_span: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> ModernBert {
        ModernBert::build(gpu, cfg, cap_rows, max_span, init, false)
    }

    /// A **trainable** encoder: every parameter `Role::Trainable` (gradient +
    /// AdamW moments, via `ParamStore`) plus the reverse step list. The
    /// forward is the SAME `build_steps` an inference build records - see
    /// `crates/decide/src/model.rs::Encoder::new_train_on`'s own doc for why
    /// that property matters (no cached-vs-uncached split to get wrong, no
    /// way for the training path's existence to move a parity number).
    pub fn new_train_on(
        gpu: Gpu,
        cfg: ModernBertConfig,
        cap_rows: u32,
        max_span: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> ModernBert {
        ModernBert::build(gpu, cfg, cap_rows, max_span, init, true)
    }

    fn build(
        gpu: Gpu,
        cfg: ModernBertConfig,
        cap_rows: u32,
        max_span: u32,
        init: &HashMap<String, Vec<f32>>,
        train: bool,
    ) -> ModernBert {
        assert!(
            max_span <= cfg.max_positions,
            "span {max_span} > max_positions {} - a window may not outrun the RoPE table",
            cfg.max_positions
        );
        assert!(max_span <= cap_rows, "max_span {max_span} > cap_rows {cap_rows}");
        let role = if train { Role::Trainable } else { Role::Frozen };
        let roles: Vec<(String, usize, Role)> =
            cfg.tensor_manifest().into_iter().map(|(n, s)| (n, s.iter().product::<usize>(), role)).collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let n = cap_rows as u64;
        let h = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        let per_row = cfg.n_heads as u64 * max_span as u64 * 4;
        let chunk = ((SLAB_BUDGET / per_row.max(1)).max(1) as u32).min(max_span.max(1));
        let slab = cfg.n_heads as u64 * chunk as u64 * max_span as u64;

        let idbuf = |name: &str| {
            gpu.buffer(name, n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST)
        };
        let layers: Vec<LayerBufs> = (0..cfg.n_layers)
            .map(|_| LayerBufs {
                qkv: gpu.storage(n * 3 * h),
                xn: gpu.storage(n * h),
                ctx: gpu.storage(n * h),
                attn_out: gpu.storage(n * h),
                res1: gpu.storage(n * h),
                xn2: gpu.storage(n * h),
                wi_u: gpu.storage(n * ff),
                wi_v: gpu.storage(n * ff),
                h_act: gpu.storage(n * ff),
                h: gpu.storage(n * ff),
                mlp_out: gpu.storage(n * h),
            })
            .collect();
        let cfg_layers = cfg.n_layers;
        let k = crate::kern::Ids::resolve(&gpu);
        let mut e = ModernBert {
            k,
            cap_rows,
            rows: 0,
            spans: Vec::new(),
            chunk,
            ids: idbuf("ids"),
            e_tok: gpu.storage(n * h),
            x: (0..=cfg.n_layers).map(|_| gpu.storage(n * h)).collect(),
            hidden: gpu.storage(n * h),
            layers,
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
            steps: Vec::new(),
            bwd: None,
            reverse_stale: false,
            gpu,
            cfg,
            ps,
        };
        e.rows = cap_rows;
        e.spans = vec![(0, cap_rows.min(max_span))];
        e.steps = e.build_steps();
        if train {
            let st = |w: u64| e.gpu.storage(w);
            e.bwd = Some(Bwd {
                d_hidden: e.gpu.buffer("d_hidden", n * h * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST),
                dx: (0..=cfg_layers).map(|_| st(n * h)).collect(),
                d_res1: st(n * h),
                d_h: st(n * ff),
                d_h_act: st(n * ff),
                d_wi_u: st(n * ff),
                d_wi_v: st(n * ff),
                d_xn2: st(n * h),
                d_tmp: st(n * h),
                d_ctx: st(n * h),
                d_qkv: st(n * 3 * h),
                d_scores: st(slab),
                d_attn_in: st(n * h),
                mean: st(n),
                inv: st(n),
                steps: Vec::new(),
            });
            e.rebuild_bwd();
        }
        e
    }

    fn w(&self, name: &str) -> &DeviceBuffer {
        self.ps.w(name)
    }

    /// Load one packed call: `ids` is the flat token stream, `spans` the
    /// `(row0, len)` of each sequence within it. No `type_ids`/`pos_ids`
    /// buffers - ModernBERT has no token-type table, and RoPE reads position
    /// straight from a span-sliced dispatch rather than an uploaded index.
    pub fn set_batch(&mut self, ids: &[u32], spans: &[(u32, u32)]) {
        assert!(ids.len() <= self.cap_rows as usize, "{} rows > capacity {}", ids.len(), self.cap_rows);
        let covered: u32 = spans.iter().map(|&(_, l)| l).sum();
        assert_eq!(covered as usize, ids.len(), "spans cover {covered} rows but {} were supplied", ids.len());
        let h = self.cfg.d_model as u64;
        for &(row0, len) in spans {
            assert!(len <= self.cfg.max_positions, "span of {len} rows > max_positions {}", self.cfg.max_positions);
            // The attention and RoPE dispatches bind a VIEW of the fused qkv
            // (row = 3H floats) starting at a span's first row, and the bound
            // offset must land on a 256-byte boundary - see `crates/decide
            // ::model::Encoder::set_batch`'s own note on the identical
            // constraint.
            let off = row0 as u64 * 3 * h * 4;
            assert_eq!(
                off % BIND_ALIGN,
                0,
                "span starting at row {row0} binds qkv at byte {off}, not a multiple of {BIND_ALIGN}; \
                 with d_model {h} a span may only start on a row that is a multiple of {}",
                (BIND_ALIGN / gcd(BIND_ALIGN, 3 * h * 4)).max(1)
            );
        }
        self.gpu.write(&self.ids, ids);
        let reshaped = self.rows != ids.len() as u32 || self.spans != spans;
        self.rows = ids.len() as u32;
        if reshaped {
            self.spans = spans.to_vec();
            self.steps = self.build_steps();
            self.reverse_stale = true;
        }
    }

    /// Record the reverse pass for the batch that is set, if it is not
    /// already - see `crates/decide/src/model.rs::Encoder::prepare_reverse`'s
    /// own doc for why this is deferred rather than done inside
    /// [`ModernBert::set_batch`].
    pub fn prepare_reverse(&mut self) {
        if self.reverse_stale {
            self.rebuild_bwd();
        }
    }

    /// Whether this encoder was built trainable.
    pub fn is_trainable(&self) -> bool {
        self.bwd.is_some()
    }

    /// Zero every parameter gradient. Call once per step BEFORE a backward,
    /// which accumulates into them.
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// The buffer the reverse pass is seeded from - the gradient of
    /// [`ModernBert::hidden_buf`] (post-`final_norm`), NOT `dx[n_layers]` -
    /// see the module doc's "Backward" section. A head writes its
    /// hidden-state gradient straight into this, so a decision step costs no
    /// copy between the trunk and the head.
    pub fn seed_buf(&self) -> &DeviceBuffer {
        &self.bwd.as_ref().expect("seed_buf on an inference build").d_hidden
    }

    /// Run the reverse pass against whatever already sits in
    /// [`ModernBert::seed_buf`].
    pub fn backward_seeded(&self) {
        assert!(!self.reverse_stale, "backward before prepare_reverse: the recorded reverse pass describes a different batch");
        let bw = self.bwd.as_ref().expect("backward on an inference build");
        self.gpu.submit(&[], &bw.steps);
    }

    /// Read one parameter's current value.
    pub fn read_weight(&self, name: &str) -> Vec<f32> {
        self.gpu.read(self.w(name), self.numel(name))
    }

    /// Overwrite one parameter - the finite-difference checker's perturbation.
    pub fn set_weight(&self, name: &str, data: &[f32]) {
        assert_eq!(data.len(), self.numel(name), "{name}");
        self.gpu.write_f32(self.w(name), data);
    }

    fn numel(&self, name: &str) -> usize {
        self.cfg
            .tensor_manifest()
            .into_iter()
            .find(|(n, _)| n == name)
            .map(|(_, s)| s.iter().product::<usize>())
            .unwrap_or_else(|| panic!("no parameter {name:?}"))
    }

    /// Read one parameter's accumulated gradient.
    pub fn read_grad(&self, name: &str) -> Vec<f32> {
        self.gpu.read(self.ps.g(name), self.numel(name))
    }

    fn rebuild_bwd(&mut self) {
        self.reverse_stale = false;
        if self.bwd.is_none() {
            return;
        }
        let steps = self.build_bwd_steps();
        if let Some(bw) = &mut self.bwd {
            bw.steps = steps;
        }
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    /// Block until this device has finished what it was given.
    ///
    /// MUST be called between this encoder's own `forward`/`backward_seeded`
    /// and any call into `laya::LayaHead` that reads this encoder's buffers
    /// (`hidden_buf`/`seed_buf`) through a DIFFERENT `Gpu` handle
    /// (`gpu.share()`'d) - a submit on one handle is not ordered against a
    /// submit on another. See `crates/decide/src/decide.rs::Decide::
    /// run_packed`'s own identical, load-bearing note (with the cost of
    /// skipping it spelled out: the head reads a hidden buffer the encoder
    /// has not finished writing, gets plausible-looking garbage, and the
    /// whole model still trains and still answers - always wrongly).
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    /// One span's RoPE dispatch on the q or k region of `qkv`, hand-built with
    /// `Gpu::step_sliced` rather than `block::rope_fwd` - see the module doc's
    /// "RoPE must be dispatched per span" note for why the convenience wrapper
    /// cannot be used as-is here. `off` is the region's word offset within a
    /// row (`0` for q, `d_model` for k); `stride` is the qkv row's full width
    /// (`3*d_model`).
    #[allow(clippy::too_many_arguments)]
    fn rope_span(&self, row0: u32, len: u32, stride: u32, off: u32, theta: f32, qkv: &DeviceBuffer, steps: &mut Vec<Step>) {
        let head_dim = self.cfg.head_dim();
        let half = head_dim / 2;
        let word_off = row0 as u64 * stride as u64 + off as u64;
        steps.push(self.gpu.step_sliced(
            self.k.rope_base,
            &[qkv],
            &[(word_off, 0)],
            // Params: n_rows, n_heads, head_dim, row_stride, base_off, tcols, rope_base.
            // `n_rows == tcols == len`: the sliced view's row 0 IS this
            // span's first token, so `pos = row % tcols == row` for every
            // row the dispatch covers.
            &[len, self.cfg.n_heads, head_dim, stride, 0, len, f(theta)],
            len * self.cfg.n_heads * half,
        ));
    }

    /// [`ModernBert::rope_span`]'s adjoint: the SAME per-span slicing
    /// convention, against `k.rope_base_bwd` instead of `k.rope_base`. RoPE's
    /// backward is the inverse rotation, applied in place to the gradient
    /// buffer, so this takes `d_qkv` where the forward took `qkv`.
    #[allow(clippy::too_many_arguments)]
    fn rope_span_bwd(&self, row0: u32, len: u32, stride: u32, off: u32, theta: f32, d_qkv: &DeviceBuffer, steps: &mut Vec<Step>) {
        let head_dim = self.cfg.head_dim();
        let half = head_dim / 2;
        let word_off = row0 as u64 * stride as u64 + off as u64;
        steps.push(self.gpu.step_sliced(
            self.k.rope_base_bwd,
            &[d_qkv],
            &[(word_off, 0)],
            &[len, self.cfg.n_heads, head_dim, stride, 0, len, f(theta)],
            len * self.cfg.n_heads * half,
        ));
    }

    /// `xn2 @ W^T` against one OUTPUT-ROW half of a `[2*n_out, k]` weight -
    /// legal as a plain sliced GEMM because the weight is row-major
    /// `[out_features, in_features]` (`matmul.wgsl`'s own convention: `W[n,
    /// k]` is output row `n`), so output rows `[half*n_out, (half+1)*n_out)`
    /// are one contiguous byte range with no interleaving to unpack - no new
    /// kernel needed for the GeGLU split, only a sliced dispatch against the
    /// existing `matmul`.
    #[allow(clippy::too_many_arguments)]
    fn half_matmul(&self, x: &DeviceBuffer, w: &DeviceBuffer, half: u32, m: u32, k_dim: u32, n_out: u32, out: &DeviceBuffer, steps: &mut Vec<Step>) {
        let w_off = half as u64 * n_out as u64 * k_dim as u64;
        steps.push(self.gpu.step_sliced(
            self.k.matmul,
            &[x, w, out],
            &[(0, 0), (w_off, 0), (0, 0)],
            &[m, k_dim, n_out],
            m * n_out,
        ));
    }

    /// [`ModernBert::half_matmul`]'s `dw` adjoint: `matmul_dw` ALWAYS
    /// accumulates (the grad buffer is zeroed once per step), so writing the
    /// two GeGLU halves' gradients into disjoint `[half*n_out*k_dim, ...)`
    /// slices of the SAME `[2*n_out, k_dim]` gradient tensor needs no add and
    /// no clear beyond the caller's own `zero_grads`.
    #[allow(clippy::too_many_arguments)]
    fn half_matmul_dw(&self, dy: &DeviceBuffer, x: &DeviceBuffer, half: u32, m: u32, k_dim: u32, n_out: u32, dw: &DeviceBuffer, steps: &mut Vec<Step>) {
        let w_off = half as u64 * n_out as u64 * k_dim as u64;
        steps.push(self.gpu.step_sliced(self.k.matmul_dw, &[dy, x, dw], &[(0, 0), (0, 0), (w_off, 0)], &[m, k_dim, n_out], n_out * k_dim));
    }

    /// [`ModernBert::half_matmul`]'s `dx` adjoint against one weight HALF.
    /// `accumulate` lets the caller fan the two GeGLU branches' `dx` into one
    /// shared `[m, k_dim]` buffer (`xn2`'s gradient is the SUM of both
    /// branches' contributions, since `xn2` feeds both `wi_u` and `wi_v`) -
    /// `false` for the first branch (assign), `true` for the second
    /// (accumulate onto what the first just wrote).
    #[allow(clippy::too_many_arguments)]
    fn half_matmul_dx(&self, dy: &DeviceBuffer, w: &DeviceBuffer, half: u32, m: u32, k_dim: u32, n_out: u32, accumulate: bool, dx: &DeviceBuffer, steps: &mut Vec<Step>) {
        let w_off = half as u64 * n_out as u64 * k_dim as u64;
        steps.push(self.gpu.step_sliced(
            self.k.matmul_dx,
            &[dy, w, dx],
            &[(0, 0), (w_off, 0), (0, 0)],
            &[m, k_dim, n_out, u32::from(accumulate)],
            m * k_dim,
        ));
    }

    fn build_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let n = self.rows;
        let h = c.d_model;
        let ff = c.d_ff;
        let hd = c.head_dim();
        let cross = block::CrossIds { scores: self.k.scores_cross, softmax: self.k.softmax_cross, apply: self.k.apply_cross };
        let cross_win = block::CrossWinIds { scores: self.k.scores_cross_win };

        // ---- embeddings ----
        let mut s = vec![g.step(self.k.embed, &[&self.ids, self.w("tok.weight"), &self.e_tok], &[h, n], n * h)];
        s.push(block::layernorm_nobias_fwd(g, self.k.layernorm_nobias, &self.e_tok, self.w("emb_norm.weight"), &self.x[0], h, n, c.eps));

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = format!("blocks.{l}");

            // ---- pre-LN attention ----
            // Layer 0 has no attn_norm tensor at all (`nn.Identity()` on the
            // real model - see `ModernBertConfig::has_attn_norm`), so its
            // qkv GEMM reads `x[l]` straight, no norm dispatch recorded.
            let attn_in: &DeviceBuffer = if c.has_attn_norm(l) {
                s.push(block::layernorm_nobias_fwd(g, self.k.layernorm_nobias, &self.x[l], self.w(&format!("{p}.attn_norm.weight")), &lb.xn, h, n, c.eps));
                &lb.xn
            } else {
                &self.x[l]
            };
            s.push(g.step(self.k.matmul, &[attn_in, self.w(&format!("{p}.qkv.weight")), &lb.qkv], &[n, h, 3 * h], n * 3 * h));

            let attn = c.layer_types[l];
            let theta = if attn == LayerAttn::Full { c.rope_theta_full } else { c.rope_theta_local };
            for &(row0, len) in &self.spans {
                self.rope_span(row0, len, 3 * h, 0, theta, &lb.qkv, &mut s); // q
                self.rope_span(row0, len, 3 * h, h, theta, &lb.qkv, &mut s); // k
            }

            match attn {
                LayerAttn::Full => block::chunked_bidir_fwd(
                    g, &cross, None, c.n_heads, hd, h, &lb.qkv, 3 * h, 0, h, 2 * h, &lb.ctx, &self.scores, &self.probs, &self.spans, self.chunk, None, &mut s,
                ),
                // Local layers MUST take the materialized windowed rung - see
                // the module doc's "attention rung choice" note.
                LayerAttn::Local => block::chunked_bidir_fwd_win(
                    g, &cross_win, self.k.softmax_cross, self.k.apply_cross, None, c.window, c.n_heads, hd, h, &lb.qkv, 3 * h, 0, h, 2 * h, &lb.ctx, &self.scores, &self.probs, &self.spans, self.chunk, &mut s,
                ),
            }

            s.push(g.step(self.k.matmul, &[&lb.ctx, self.w(&format!("{p}.proj.weight")), &lb.attn_out], &[n, h, h], n * h));
            s.push(g.step(self.k.add2, &[&self.x[l], &lb.attn_out, &lb.res1], &[n * h], n * h));

            // ---- pre-LN GeGLU MLP ----
            s.push(block::layernorm_nobias_fwd(g, self.k.layernorm_nobias, &lb.res1, self.w(&format!("{p}.mlp_norm.weight")), &lb.xn2, h, n, c.eps));
            let wi = self.w(&format!("{p}.mlp.wi.weight"));
            self.half_matmul(&lb.xn2, wi, 0, n, h, ff, &lb.wi_u, &mut s); // "input" half - gets the activation
            self.half_matmul(&lb.xn2, wi, 1, n, h, ff, &lb.wi_v, &mut s); // "gate" half - raw multiplier
            s.push(g.step(self.k.gelu_erf, &[&lb.wi_u, &lb.h_act], &[n * ff], n * ff));
            s.push(g.step(self.k.mul, &[&lb.h_act, &lb.wi_v, &lb.h], &[n * ff], n * ff));
            s.push(g.step(self.k.matmul, &[&lb.h, self.w(&format!("{p}.mlp.wo.weight")), &lb.mlp_out], &[n, ff, h], n * h));
            s.push(g.step(self.k.add2, &[&lb.res1, &lb.mlp_out, &self.x[l + 1]], &[n * h], n * h));
        }

        s.push(block::layernorm_nobias_fwd(g, self.k.layernorm_nobias, &self.x[c.n_layers as usize], self.w("final_norm.weight"), &self.hidden, h, n, c.eps));
        s
    }

    /// The exact adjoint of [`ModernBert::build_steps`], walked bottom up -
    /// see the module doc's "Backward" section for the shape (final_norm
    /// undone first, GeGLU/RoPE/windowed-attention backward per piece).
    fn build_bwd_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let bw = self.bwd.as_ref().expect("build_bwd_steps in training mode only");
        let n = self.rows;
        let h = c.d_model;
        let ff = c.d_ff;
        let hd = c.head_dim();
        // `layernorm` is never dispatched through this struct (`ln_stats_fwd`
        // and `layernorm_dx_bwd` only ever read `.ln_stats`/`.layernorm_dx`)
        // - see `crate::kern::PIPELINES`'s own note on why the reference
        // no-bias-LN backward needs no dedicated kernel.
        let ln = block::LayerNormIds {
            layernorm: self.k.layernorm_dx,
            layernorm_rows: None,
            ln_stats: self.k.ln_stats,
            ln_stats_rows: None,
            layernorm_dx: self.k.layernorm_dx,
            layernorm_dx_rows: None,
        };
        let cross = block::CrossIds { scores: self.k.scores_cross, softmax: self.k.softmax_cross, apply: self.k.apply_cross };
        let cross_win = block::CrossWinIds { scores: self.k.scores_cross_win };
        let cross_bwd = block::CrossBwdIds::resolve(g, self.k.dscores_cross, self.k.dq_cross, self.k.dk_cross_acc, self.k.dv_cross_acc);
        let gr = |name: &str| self.ps.g(name);
        let mut s: Vec<Step> = Vec::new();

        // ---- undo final_norm FIRST - see the module doc's "Backward"
        // section for why this crate's reverse walk is one step longer than
        // `decide::Encoder`'s own. After this, `dx[n_layers]` is what the
        // per-layer loop below expects as its own entry point. ----
        let x_last = &self.x[c.n_layers as usize];
        s.push(block::ln_stats_fwd(g, &ln, x_last, &bw.mean, &bw.inv, h, n, c.eps));
        s.push(g.step(self.k.ln_dgamma, &[&bw.d_hidden, x_last, &bw.mean, &bw.inv, gr("final_norm.weight")], &[h, n], h));
        s.push(block::layernorm_dx_bwd(g, &ln, x_last, self.w("final_norm.weight"), &bw.d_hidden, &bw.dx[c.n_layers as usize], h, n, c.eps));

        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            let p = format!("blocks.{l}");
            let d_out = &bw.dx[l + 1];

            // ---- GeGLU MLP branch: `x[l+1] = res1 + mlp_out`, so `d_out`
            // is BOTH the residual's own share and the mlp branch's
            // incoming gradient, unchanged (addition fans out). ----
            s.push(g.step(self.k.matmul_dx, &[d_out, self.w(&format!("{p}.mlp.wo.weight")), &bw.d_h], &[n, ff, h, 0], n * ff));
            s.push(g.step(self.k.matmul_dw, &[d_out, &lb.h, gr(&format!("{p}.mlp.wo.weight"))], &[n, ff, h], h * ff));
            // GeGLU backward: h = gelu_erf(wi_u) * wi_v. d(a*b)/da = b, so
            // reuse `mul` (forward-shaped) for both partials, then
            // `gelu_erf_bwd` differentiates the activated half only.
            s.push(g.step(self.k.mul, &[&bw.d_h, &lb.wi_v, &bw.d_h_act], &[n * ff], n * ff));
            s.push(g.step(self.k.mul, &[&bw.d_h, &lb.h_act, &bw.d_wi_v], &[n * ff], n * ff));
            s.push(g.step(self.k.gelu_erf_bwd, &[&lb.wi_u, &bw.d_h_act, &bw.d_wi_u], &[n * ff], n * ff));

            let wi = self.w(&format!("{p}.mlp.wi.weight"));
            let wi_grad = gr(&format!("{p}.mlp.wi.weight"));
            self.half_matmul_dw(&bw.d_wi_u, &lb.xn2, 0, n, h, ff, wi_grad, &mut s);
            self.half_matmul_dw(&bw.d_wi_v, &lb.xn2, 1, n, h, ff, wi_grad, &mut s);
            // `xn2` feeds BOTH halves, so its gradient is their SUM - the
            // second dx call accumulates onto what the first assigned.
            self.half_matmul_dx(&bw.d_wi_u, wi, 0, n, h, ff, false, &bw.d_xn2, &mut s);
            self.half_matmul_dx(&bw.d_wi_v, wi, 1, n, h, ff, true, &bw.d_xn2, &mut s);

            // ---- mlp_norm backward (no-bias) ----
            s.push(block::ln_stats_fwd(g, &ln, &lb.res1, &bw.mean, &bw.inv, h, n, c.eps));
            s.push(g.step(self.k.ln_dgamma, &[&bw.d_xn2, &lb.res1, &bw.mean, &bw.inv, gr(&format!("{p}.mlp_norm.weight"))], &[h, n], h));
            s.push(block::layernorm_dx_bwd(g, &ln, &lb.res1, self.w(&format!("{p}.mlp_norm.weight")), &bw.d_xn2, &bw.d_tmp, h, n, c.eps));
            s.push(g.step(self.k.add2, &[d_out, &bw.d_tmp, &bw.d_res1], &[n * h], n * h));

            // ---- attention branch: `res1 = x[l] + attn_out` ----
            s.push(g.step(self.k.matmul_dx, &[&bw.d_res1, self.w(&format!("{p}.proj.weight")), &bw.d_ctx], &[n, h, h, 0], n * h));
            s.push(g.step(self.k.matmul_dw, &[&bw.d_res1, &lb.ctx, gr(&format!("{p}.proj.weight"))], &[n, h, h], h * h));

            // Per-span attention backward, recomputing scores/probs from the
            // cached (POST-RoPE) qkv - see `block::chunked_bidir_bwd[_win]`'s
            // own doc. `d_qkv` needs no clear: the first chunk of every span
            // ASSIGNS its k/v region and later chunks accumulate onto it.
            let attn = c.layer_types[l];
            match attn {
                LayerAttn::Full => block::chunked_bidir_bwd(
                    g, &cross, None, &cross_bwd, c.n_heads, hd, h, &lb.qkv, 3 * h, 0, h, 2 * h, &bw.d_ctx, &bw.d_qkv, &self.scores, &self.probs, &bw.d_scores, &self.spans, self.chunk, None, &mut s,
                ),
                LayerAttn::Local => block::chunked_bidir_bwd_win(
                    g, &cross_win, self.k.softmax_cross, None, &cross_bwd, c.window, c.n_heads, hd, h, &lb.qkv, 3 * h, 0, h, 2 * h, &bw.d_ctx, &bw.d_qkv, &self.scores, &self.probs, &bw.d_scores, &self.spans, self.chunk, &mut s,
                ),
            }

            // Undo RoPE on the q/k regions of `d_qkv`, per span - the exact
            // adjoint of the matching forward dispatch, same theta.
            let theta = if attn == LayerAttn::Full { c.rope_theta_full } else { c.rope_theta_local };
            for &(row0, len) in &self.spans {
                self.rope_span_bwd(row0, len, 3 * h, 0, theta, &bw.d_qkv, &mut s); // q
                self.rope_span_bwd(row0, len, 3 * h, h, theta, &bw.d_qkv, &mut s); // k
            }

            let attn_in: &DeviceBuffer = if c.has_attn_norm(l) { &lb.xn } else { &self.x[l] };
            s.push(g.step(self.k.matmul_dx, &[&bw.d_qkv, self.w(&format!("{p}.qkv.weight")), &bw.d_attn_in], &[n, h, 3 * h, 0], n * h));
            s.push(g.step(self.k.matmul_dw, &[&bw.d_qkv, attn_in, gr(&format!("{p}.qkv.weight"))], &[n, h, 3 * h], 3 * h * h));

            if c.has_attn_norm(l) {
                s.push(block::ln_stats_fwd(g, &ln, &self.x[l], &bw.mean, &bw.inv, h, n, c.eps));
                s.push(g.step(self.k.ln_dgamma, &[&bw.d_attn_in, &self.x[l], &bw.mean, &bw.inv, gr(&format!("{p}.attn_norm.weight"))], &[h, n], h));
                s.push(block::layernorm_dx_bwd(g, &ln, &self.x[l], self.w(&format!("{p}.attn_norm.weight")), &bw.d_attn_in, &bw.d_tmp, h, n, c.eps));
                s.push(g.step(self.k.add2, &[&bw.d_res1, &bw.d_tmp, &bw.dx[l]], &[n * h], n * h));
            } else {
                // Layer 0's qkv reads `x[0]` directly (no attn_norm to
                // undo), so `d_attn_in` IS the attention branch's share of
                // `x[0]`'s gradient already.
                s.push(g.step(self.k.add2, &[&bw.d_res1, &bw.d_attn_in, &bw.dx[l]], &[n * h], n * h));
            }
        }

        // ---- embeddings ----
        s.push(block::ln_stats_fwd(g, &ln, &self.e_tok, &bw.mean, &bw.inv, h, n, c.eps));
        s.push(g.step(self.k.ln_dgamma, &[&bw.dx[0], &self.e_tok, &bw.mean, &bw.inv, gr("emb_norm.weight")], &[h, n], h));
        // `d_e_tok` reuses `d_tmp` - nothing past this point needs the
        // norm-branch scratch again.
        s.push(block::layernorm_dx_bwd(g, &ln, &self.e_tok, self.w("emb_norm.weight"), &bw.dx[0], &bw.d_tmp, h, n, c.eps));
        let eb = block::EmbBwdIds::resolve(g, self.k.emb_bwd);
        s.push(block::emb_bwd_step(g, &eb, &self.ids, None, &bw.d_tmp, gr("tok.weight"), n, h, c.vocab));
        s
    }

    // ---- parity / inference taps ----

    /// The final hidden states (post-`final_norm`), `[rows, H]` row-major
    /// over the PACKED rows.
    pub fn hidden(&self) -> Vec<f32> {
        self.read(&self.hidden)
    }

    /// The final hidden states' DEVICE buffer - what `laya::LayaHead` reads,
    /// mirroring `decide::model::Encoder::hidden_buf`. A head built on top of
    /// this encoder dispatches straight against this buffer rather than
    /// paying a host round trip through [`ModernBert::hidden`] first.
    pub fn hidden_buf(&self) -> &DeviceBuffer {
        &self.hidden
    }

    /// Mean of the final hidden states over each span. `[spans, H]`. No pad
    /// rows to exclude - see the module doc's packing note.
    pub fn pooled_mean(&self) -> Vec<f32> {
        let h = self.cfg.d_model as usize;
        let hid = self.hidden();
        let mut out = Vec::with_capacity(self.spans.len() * h);
        for &(row0, len) in &self.spans {
            let mut acc = vec![0.0f32; h];
            for r in 0..len as usize {
                let base = (row0 as usize + r) * h;
                for (a, v) in acc.iter_mut().zip(&hid[base..base + h]) {
                    *a += v;
                }
            }
            let inv = 1.0 / len.max(1) as f32;
            out.extend(acc.into_iter().map(|v| v * inv));
        }
        out
    }

    /// The `(row0, len)` of each sequence in the current batch.
    pub fn spans(&self) -> &[(u32, u32)] {
        &self.spans
    }

    fn read(&self, b: &DeviceBuffer) -> Vec<f32> {
        self.gpu.read(b, (self.rows * self.cfg.d_model) as usize)
    }
}
