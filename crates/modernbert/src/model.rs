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

/// The ModernBERT encoder. Inference-only (`Role::Frozen` weights, no
/// backward) - the seeded backward through the trunk is Laya M5, a separate
/// milestone.
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
        assert!(
            max_span <= cfg.max_positions,
            "span {max_span} > max_positions {} - a window may not outrun the RoPE table",
            cfg.max_positions
        );
        assert!(max_span <= cap_rows, "max_span {max_span} > cap_rows {cap_rows}");
        let roles: Vec<(String, usize, Role)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), Role::Frozen))
            .collect();
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
            gpu,
            cfg,
            ps,
        };
        e.rows = cap_rows;
        e.spans = vec![(0, cap_rows.min(max_span))];
        e.steps = e.build_steps();
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
        }
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
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
