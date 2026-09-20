// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The bidirectional state encoder's forward.
//!
//! ```text
//! x0     = LN_emb(embed(ids) + embed(pos_ids) + embed(type_ids))
//! per layer (POST-LayerNorm, which is where BERT differs from CLIP's tower):
//!   qkv  = x @ Wqkv^T + b                                   [rows, 3H]
//!   ctx  = bidirectional self-attention, per span           [rows, H]
//!   res  = LN1(x + (ctx @ Wo^T + bo))
//!   x'   = LN2(res + (gelu(res @ W1^T + b1) @ W2^T + b2))
//! ```
//!
//! **Sequences are packed, not padded.** Every sequence in a call is laid end
//! to end in one flat `[rows, H]` buffer and described by a span `(row0, len)`;
//! `block::chunked_bidir_fwd` self-attends within each span independently. That
//! is what lets one forward carry a batch of different lengths - state windows
//! and option slots in the same call - with no mask buffer, no padded rows to
//! compute, and no wasted attention area.
//!
//! It also sidesteps the trap that makes a padded bidirectional encoder subtly
//! wrong: with no causal structure, an unmasked pad position contributes to
//! every real position's attention. Packing removes the pad positions rather
//! than remembering to mask them.
//!
//! Position ids are supplied per row rather than derived from a fixed stride,
//! for the same reason - `pos_add` assumes every sequence has the same length.
//!
//! The step list is rebuilt when the spans change, because `chunked_bidir_fwd`
//! bakes them in. That is host-side work proportional to the span count, not a
//! device cost, and a caller whose spans are stable (the realtime path, whose
//! window size and option set repeat) rebuilds nothing.

use std::collections::HashMap;

use gpu_core::{DeviceBuffer, Gpu, Step};
use model::block;
use paramstore::{ParamStore, Role};

use crate::config::EncoderConfig;

/// Attention-slab budget for the chunked path: the `[heads, chunk, len]` score
/// and probability slabs are sized against it, so `chunk` falls as the longest
/// span grows and the allocation stays bounded whatever a caller asks for.
const SLAB_BUDGET: u64 = 256 << 20;

/// WebGPU's `min_storage_buffer_offset_alignment`. The span attention binds a
/// VIEW of the fused qkv and of the context buffer starting at a span's first
/// row, and a bound offset must be a multiple of this. It is a hardware
/// binding rule, not a tunable: the driver rejects the bind group outright.
const BIND_ALIGN: u64 = 256;

/// Greatest common divisor, for reporting the span-start granularity a given
/// width imposes.
fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

struct LayerBufs {
    /// Fused `[rows, 3H]` - q at 0, k at H, v at 2H.
    qkv: DeviceBuffer,
    ctx: DeviceBuffer,
    attn_out: DeviceBuffer,
    /// `x + attn_out`, before LN1.
    res_pre: DeviceBuffer,
    res: DeviceBuffer,
    h: DeviceBuffer,
    h_act: DeviceBuffer,
    mlp_out: DeviceBuffer,
    /// `res + mlp_out`, before LN2.
    ffn_pre: DeviceBuffer,
}

/// Reverse-pass buffers and the recorded backward step list. Allocated only by
/// [`Encoder::new_train_on`]; an inference build carries `None` and runs the
/// byte-for-byte graph the parity ladder gates.
///
/// Every entry is a GRADIENT. The activations the backward reads are the
/// forward's own buffers - nothing is recomputed on the host and nothing is
/// aliased. The per-layer scratch is shared across layers because the reverse
/// walk holds one layer's intermediates live at a time; only `dx` is per layer.
struct Bwd {
    /// `dx[i]` is the grad of `x[i]`, so `dx[n_layers]` is the objective's own
    /// seed: the final hidden states ARE `x[n_layers]`, and writing the seed
    /// straight in saves a copy.
    dx: Vec<DeviceBuffer>,
    d_ffn_pre: DeviceBuffer,
    d_res: DeviceBuffer,
    d_res_pre: DeviceBuffer,
    /// Grad arriving from one branch, before it re-joins the residual.
    d_tmp: DeviceBuffer,
    /// Grad of the POST-activation FFN hidden.
    d_h_act: DeviceBuffer,
    /// Grad of the PRE-activation FFN hidden - what the activation backward
    /// differentiates, which is why `h` is kept and not just `h_act`.
    d_h: DeviceBuffer,
    d_ctx: DeviceBuffer,
    d_qkv: DeviceBuffer,
    d_scores: DeviceBuffer,
    /// Grad of the summed embedding, before its LayerNorm. All three tables
    /// read this same buffer: addition fans out, so their adjoints are equal.
    d_sum: DeviceBuffer,
    /// Per-row LayerNorm mean / inverse-std, recomputed per use (they are
    /// `[rows]`, cheaper to recompute than to cache).
    mean: DeviceBuffer,
    inv: DeviceBuffer,
    steps: Vec<Step>,
}




pub struct Encoder {
    pub gpu: Gpu,
    k: crate::kern::Ids,
    pub cfg: EncoderConfig,
    pub ps: ParamStore,
    /// Capacity in rows; a call may use fewer.
    cap_rows: u32,
    rows: u32,
    spans: Vec<(u32, u32)>,
    chunk: u32,
    ids: DeviceBuffer,
    pos_ids: DeviceBuffer,
    type_ids: DeviceBuffer,
    /// The DISTINCT ids of the three index streams above, and how many each
    /// holds. The embedding scatter only has to visit the table rows a call
    /// actually looked up, and a call looks up at most `rows` of a 30522-row
    /// vocabulary - see `block::emb_bwd_step`.
    uniq: [DeviceBuffer; 3],
    uniq_n: [u32; 3],
    e_tok: DeviceBuffer,
    e_pos: DeviceBuffer,
    e_type: DeviceBuffer,
    sum1: DeviceBuffer,
    sum2: DeviceBuffer,
    /// `x[0]` = embedding output, `x[i+1]` = layer `i`'s output.
    x: Vec<DeviceBuffer>,
    layers: Vec<LayerBufs>,
    scores: DeviceBuffer,
    probs: DeviceBuffer,
    /// K for one span, transposed to key-minor. See [`block::KeyMinor`]: the
    /// scores kernel reads K with the key index as its fastest thread index,
    /// so it wants K laid out that way and this is where the transposed copy
    /// lives. `[d_model, longest span]`, rewritten once per span.
    kt: DeviceBuffer,
    steps: Vec<Step>,
    bwd: Option<Bwd>,
}

impl Encoder {
    /// Build on an existing device, sized for at most `cap_rows` packed tokens
    /// and a longest span of `max_span`. Every parameter is `Frozen`: this is
    /// the inference graph the parity ladder gates.
    pub fn new_on(
        gpu: Gpu,
        cfg: EncoderConfig,
        cap_rows: u32,
        max_span: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> Encoder {
        Encoder::build(gpu, cfg, cap_rows, max_span, init, false)
    }

    /// A **trainable** encoder on an existing device: every parameter
    /// `Role::Trainable` (gradient + AdamW moments) plus the reverse step list.
    ///
    /// The forward is the SAME `build_steps` an inference build records - the
    /// graph is already SSA, so there is no cached-vs-uncached split to get
    /// wrong and no way for the training path's existence to move a parity
    /// number.
    pub fn new_train_on(
        gpu: Gpu,
        cfg: EncoderConfig,
        cap_rows: u32,
        max_span: u32,
        init: &HashMap<String, Vec<f32>>,
    ) -> Encoder {
        Encoder::build(gpu, cfg, cap_rows, max_span, init, true)
    }

    fn build(
        gpu: Gpu,
        cfg: EncoderConfig,
        cap_rows: u32,
        max_span: u32,
        init: &HashMap<String, Vec<f32>>,
        train: bool,
    ) -> Encoder {
        assert!(
            max_span <= cfg.max_positions,
            "span {max_span} > max_positions {} - a window may not outrun the learned position table",
            cfg.max_positions
        );
        assert!(max_span <= cap_rows, "max_span {max_span} > cap_rows {cap_rows}");
        let role = if train { Role::Trainable } else { Role::Frozen };
        let roles: Vec<(String, usize, Role)> = cfg
            .tensor_manifest()
            .into_iter()
            .map(|(n, s)| (n, s.iter().product::<usize>(), role))
            .collect();
        let ps = ParamStore::new_with_roles(&gpu, roles, init);

        let n = cap_rows as u64;
        let h = cfg.d_model as u64;
        let ff = cfg.d_ff as u64;
        // `chunk` query rows at a time against a whole span's keys.
        let per_row = cfg.n_heads as u64 * max_span as u64 * 4;
        let chunk = ((SLAB_BUDGET / per_row.max(1)).max(1) as u32).min(max_span.max(1));
        let slab = cfg.n_heads as u64 * chunk as u64 * max_span as u64;

        let idbuf = |name: &str| {
            gpu.buffer(name, n * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST)
        };
        let layers: Vec<LayerBufs> = (0..cfg.n_layers)
            .map(|_| LayerBufs {
                qkv: gpu.storage(n * 3 * h),
                ctx: gpu.storage(n * h),
                attn_out: gpu.storage(n * h),
                res_pre: gpu.storage(n * h),
                res: gpu.storage(n * h),
                h: gpu.storage(n * ff),
                h_act: gpu.storage(n * ff),
                mlp_out: gpu.storage(n * h),
                ffn_pre: gpu.storage(n * h),
            })
            .collect();
        let cfg_layers = cfg.n_layers;
        let k = crate::kern::Ids::resolve(&gpu);
        let mut e = Encoder {
            k,
            cap_rows,
            rows: 0,
            spans: Vec::new(),
            chunk,
            ids: idbuf("ids"),
            pos_ids: idbuf("pos_ids"),
            type_ids: idbuf("type_ids"),
            uniq: [idbuf("uniq_tok"), idbuf("uniq_pos"), idbuf("uniq_type")],
            uniq_n: [0; 3],
            e_tok: gpu.storage(n * h),
            e_pos: gpu.storage(n * h),
            e_type: gpu.storage(n * h),
            sum1: gpu.storage(n * h),
            sum2: gpu.storage(n * h),
            x: (0..=cfg.n_layers).map(|_| gpu.storage(n * h)).collect(),
            layers,
            scores: gpu.storage(slab),
            probs: gpu.storage(slab),
            kt: gpu.storage(h * cfg.max_positions as u64),
            steps: Vec::new(),
            bwd: None,
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
                // COPY_DST because the objective's seed is uploaded into
                // `dx[n_layers]` directly rather than copied in on device.
                dx: (0..=cfg_layers)
                    .map(|_| {
                        e.gpu.buffer("dx", n * h * 4, gpu_core::BufUsage::STORAGE | gpu_core::BufUsage::COPY_DST)
                    })
                    .collect(),
                d_ffn_pre: st(n * h),
                d_res: st(n * h),
                d_res_pre: st(n * h),
                d_tmp: st(n * h),
                d_h_act: st(n * ff),
                d_h: st(n * ff),
                d_ctx: st(n * h),
                d_qkv: st(n * 3 * h),
                d_scores: st(slab),
                d_sum: st(n * h),
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

    /// Which GEMM kernel, and how many invocations it wants.
    ///
    /// The tiling comes from [`block::gemm_tile`], which reads THIS device's
    /// caps - so the rule follows the hardware rather than a number measured
    /// on one card, and a device that cannot run the narrow kernel is never
    /// offered it.
    ///
    /// The shared `pick_gemm` answers a TRAINING-shaped question - is this
    /// output big enough to be worth a tile at all - and then hands every
    /// tiled shape to the 128x128 kernel. That is the wrong last step for an
    /// encoder, because an encoder's linears are not large: at a few hundred
    /// packed rows this model's `proj` and `fc2` produce a 541x384 output,
    /// and a 128x128 tile covers that in FIFTEEN workgroups. The card has
    /// thirty SMs, so half of it is idle before the inner loop executes an
    /// instruction.
    ///
    /// So the tile is chosen by whether it leaves the machine full. Measured
    /// on this repository's own hardware at 541 packed rows:
    ///
    /// ```text
    ///   shape                128x128 tile      64x64 tile     workgroups
    ///   qkv  541x384x1152      0.282 ms         0.330 ms       45 vs 162
    ///   proj 541x384x384       0.209 ms         0.174 ms       15 vs  54
    ///   fc1  541x384x1536      0.268 ms         0.349 ms       60 vs 216
    ///   fc2  541x1536x384      0.545 ms         0.416 ms       15 vs  54
    /// ```
    ///
    /// The wider tile wins wherever it covers the thirty SMs at least once
    /// and loses wherever it does not, which is the rule rather than a table
    /// to memorise: below that point its better arithmetic intensity has
    /// nowhere to be spent.
    fn gemm(&self, m: u32, n: u32) -> (usize, u32) {
        let (kind, threads) =
            block::pick_gemm(m as usize, n as usize, self.k.matmul, self.k.matmul_reg3, false);
        // `pick_gemm` answers "is this worth tiling at all", against a
        // hardcoded discrete-GPU baseline. If it said no, it said no.
        if kind != self.k.matmul_reg3 {
            return (kind, threads);
        }
        match block::gemm_tile(m, n, &self.gpu.caps()) {
            block::GemmTile::Wide => (self.k.matmul_reg3, threads),
            block::GemmTile::Narrow => {
                (self.k.matmul_reg3_64, m.div_ceil(64) * n.div_ceil(64) * 256)
            }
        }
    }

    /// Load one packed call: `ids`/`type_ids` are the flat token and segment
    /// streams, `spans` the `(row0, len)` of each sequence within them.
    ///
    /// Position ids are derived here - `0..len` within each span - so a caller
    /// never has to know that a packed row's position is not its row index.
    pub fn set_batch(&mut self, ids: &[u32], type_ids: &[u32], spans: &[(u32, u32)]) {
        assert_eq!(ids.len(), type_ids.len(), "ids and type_ids must be the same length");
        assert!(ids.len() <= self.cap_rows as usize, "{} rows > capacity {}", ids.len(), self.cap_rows);
        let covered: u32 = spans.iter().map(|&(_, l)| l).sum();
        assert_eq!(covered as usize, ids.len(), "spans cover {covered} rows but {} were supplied", ids.len());
        let mut pos = vec![0u32; ids.len()];
        let h = self.cfg.d_model as u64;
        for &(row0, len) in spans {
            assert!(
                len <= self.cfg.max_positions,
                "span of {len} rows > max_positions {}",
                self.cfg.max_positions
            );
            // The attention binds each span as a view starting at `row0`, in
            // the fused qkv (row = 3H floats) and in the context (row = H
            // floats). Both offsets must land on a 256-byte boundary.
            //
            // Every real checkpoint of this family has `d_model` a multiple of
            // 64, which makes both row strides multiples of 256 and every
            // `row0` legal. A width that does not gets a named failure here
            // rather than a driver-level bind-group rejection several layers
            // deeper, where the offset is all the message contains.
            for (bytes, what) in [(3 * h * 4, "qkv"), (h * 4, "context")] {
                let off = row0 as u64 * bytes;
                assert_eq!(
                    off % BIND_ALIGN,
                    0,
                    "span starting at row {row0} binds the {what} buffer at byte {off}, which is not a                      multiple of {BIND_ALIGN}; with d_model {h} a span may only start on a row that is                      a multiple of {}",
                    (BIND_ALIGN / gcd(BIND_ALIGN, bytes)).max(1)
                );
            }
            for i in 0..len {
                pos[(row0 + i) as usize] = i;
            }
        }
        self.gpu.write(&self.ids, ids);
        self.gpu.write(&self.pos_ids, &pos);
        self.gpu.write(&self.type_ids, type_ids);
        // Built from the SAME slices that were just uploaded, never from the
        // capacity-sized buffers: a `uniq` list that misses an id silently
        // drops that table row's gradient.
        let mut uniq_n = [0u32; 3];
        for (i, src) in [ids, &pos, type_ids].into_iter().enumerate() {
            let u = block::uniq_u32(src);
            self.gpu.write(&self.uniq[i], &u);
            uniq_n[i] = u.len() as u32;
        }
        // The distinct count is both a kernel parameter and a thread count, so
        // a call that repacks the same span layout with different tokens still
        // needs its reverse pass re-recorded.
        let changed = self.rows != ids.len() as u32 || self.spans != spans || self.uniq_n != uniq_n;
        self.rows = ids.len() as u32;
        self.uniq_n = uniq_n;
        if changed {
            self.spans = spans.to_vec();
            self.steps = self.build_steps();
            self.rebuild_bwd();
        }
    }


    /// The recorded forward dispatches, for a profiler that wants to time
    /// this pass without running the model around it.
    pub fn steps(&self) -> &[gpu_core::Step] {
        &self.steps
    }

    pub fn forward(&self) {
        self.gpu.submit(&[], &self.steps);
    }

    fn build_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let n = self.rows;
        let h = c.d_model;
        let ff = c.d_ff;
        let hd = c.head_dim();
        let ln = block::LayerNormIds::resolve(g, self.k.layernorm, self.k.ln_stats, self.k.layernorm_dx);
        let cross = block::CrossIds { scores: self.k.scores_cross, softmax: self.k.softmax_cross, apply: self.k.apply_cross };
        // K read key-minor. The transpose is one dispatch per span, hoisted
        // out of the query-chunk loop by `chunked_bidir_fwd`, and it is what
        // turns the scores kernel's per-lane loads from one transaction each
        // into coalesced ones.
        let km = block::KeyMinor { transpose: self.k.kv_k_headt, scores: self.k.scores_cross_kt, kt: &self.kt };
        // Fused attention, on any device that can run it.
        //
        // Gated on `workgroup_reductions` alone, which is a CORRECTNESS gate:
        // the kernel needs two top-level barriers the Cranelift CPU JIT does
        // not provide. NOT gated on trainability, even though the fused
        // kernel writes no score slab and a backward might have wanted to
        // read one - `chunked_bidir_bwd` recomputes scores and probs from the
        // cached qkv for itself and reads nothing the forward left behind, so
        // the training and inference graphs stay the same forward, which is
        // the property this constructor's documentation promises and the
        // reason a parity number cannot move with how an encoder was built.
        let flash = g.caps().workgroup_reductions.then_some(block::FlashIds {
            bidir: self.k.flash_bidir,
            split: Some(self.k.flash_bidir_split),
            reg: Some(self.k.flash_bidir_reg),
            reg2: Some(self.k.flash_bidir_reg2),
        });
        // ---- embeddings ----
        // `embed` Params: [width, rows]; bufs [index(u32), table, out]. The
        // position and segment tables are gathered the same way as the token
        // table rather than added by stride, because packed spans do not share
        // one sequence length (see the module docs).
        let mut s = vec![
            g.step(self.k.embed, &[&self.ids, self.w("tok.weight"), &self.e_tok], &[h, n], n * h),
            g.step(self.k.embed, &[&self.pos_ids, self.w("pos.weight"), &self.e_pos], &[h, n], n * h),
            g.step(self.k.embed, &[&self.type_ids, self.w("type.weight"), &self.e_type], &[h, n], n * h),
        ];
        s.push(g.step(self.k.add2, &[&self.e_tok, &self.e_pos, &self.sum1], &[n * h], n * h));
        s.push(g.step(self.k.add2, &[&self.sum1, &self.e_type, &self.sum2], &[n * h], n * h));
        s.push(block::layernorm_fwd(
            g,
            &ln,
            &self.sum2,
            self.w("emb_ln.weight"),
            self.w("emb_ln.bias"),
            &self.x[0],
            h,
            n,
            c.eps,
        ));

        for l in 0..c.n_layers as usize {
            let lb = &self.layers[l];
            let p = format!("blocks.{l}");

            let (mk, mt) = self.gemm(n, 3 * h);
            s.push(g.step(mk, &[&self.x[l], self.w(&format!("{p}.qkv.weight")), &lb.qkv], &[n, h, 3 * h], mt));
            s.push(g.step(self.k.bias_add, &[&lb.qkv, self.w(&format!("{p}.qkv.bias"))], &[n, 3 * h], n * 3 * h));

            // Self-attention within each span, independently. q/k/v live at
            // 0/H/2H of the fused row.
            match flash {
                Some(ids) => block::flash_bidir_fwd(
                    g,
                    ids,
                    c.n_heads,
                    hd,
                    h,
                    &lb.qkv,
                    3 * h,
                    0,
                    h,
                    2 * h,
                    &lb.ctx,
                    &self.spans,
                    &mut s,
                ),
                None => block::chunked_bidir_fwd(
                    g,
                    &cross,
                    Some(&km),
                    c.n_heads,
                    hd,
                    h,
                    &lb.qkv,
                    3 * h,
                    0,
                    h,
                    2 * h,
                    &lb.ctx,
                    &self.scores,
                    &self.probs,
                    &self.spans,
                    self.chunk,
                    None,
                    &mut s,
                ),
            }

            let (mk, mt) = self.gemm(n, h);
            s.push(g.step(mk, &[&lb.ctx, self.w(&format!("{p}.proj.weight")), &lb.attn_out], &[n, h, h], mt));
            s.push(g.step(self.k.bias_add, &[&lb.attn_out, self.w(&format!("{p}.proj.bias"))], &[n, h], n * h));
            // POST-LayerNorm: the residual is added first and normalized after.
            s.push(g.step(self.k.add2, &[&self.x[l], &lb.attn_out, &lb.res_pre], &[n * h], n * h));
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &lb.res_pre,
                self.w(&format!("{p}.ln1.weight")),
                self.w(&format!("{p}.ln1.bias")),
                &lb.res,
                h,
                n,
                c.eps,
            ));

            let (mk, mt) = self.gemm(n, ff);
            s.push(g.step(mk, &[&lb.res, self.w(&format!("{p}.fc1.weight")), &lb.h], &[n, h, ff], mt));
            s.push(g.step(self.k.bias_add, &[&lb.h, self.w(&format!("{p}.fc1.bias"))], &[n, ff], n * ff));
            s.push(g.step(self.k.gelu_erf, &[&lb.h, &lb.h_act], &[n * ff], n * ff));
            let (mk, mt) = self.gemm(n, h);
            s.push(g.step(mk, &[&lb.h_act, self.w(&format!("{p}.fc2.weight")), &lb.mlp_out], &[n, ff, h], mt));
            s.push(g.step(self.k.bias_add, &[&lb.mlp_out, self.w(&format!("{p}.fc2.bias"))], &[n, h], n * h));
            s.push(g.step(self.k.add2, &[&lb.res, &lb.mlp_out, &lb.ffn_pre], &[n * h], n * h));
            s.push(block::layernorm_fwd(
                g,
                &ln,
                &lb.ffn_pre,
                self.w(&format!("{p}.ln2.weight")),
                self.w(&format!("{p}.ln2.bias")),
                &self.x[l + 1],
                h,
                n,
                c.eps,
            ));
        }
        s
    }

    /// Re-record the reverse pass. Called whenever the spans change, for the
    /// same reason the forward is: `chunked_bidir_bwd` bakes them in.
    fn rebuild_bwd(&mut self) {
        if self.bwd.is_none() {
            return;
        }
        // Built BEFORE the mutable borrow: `build_bwd_steps` reads the
        // forward's own buffers through `&self`.
        let steps = self.build_bwd_steps();
        if let Some(bw) = &mut self.bwd {
            bw.steps = steps;
        }
    }

    /// Whether this encoder was built trainable.
    pub fn is_trainable(&self) -> bool {
        self.bwd.is_some()
    }

    /// Zero every parameter gradient. Call once per step BEFORE
    /// [`Encoder::backward`], which accumulates into them.
    pub fn zero_grads(&self) {
        self.ps.zero_grads(&self.gpu);
    }

    /// Seed the reverse pass with the objective's gradient on the final hidden
    /// states, `[rows, H]` row-major, and run it.
    pub fn backward(&self, d_hidden: &[f32]) {
        let bw = self.bwd.as_ref().expect("backward on an inference build");
        let want = (self.rows * self.cfg.d_model) as usize;
        assert_eq!(d_hidden.len(), want, "d_hidden must be [rows, H] = {want}");
        self.gpu.write_f32(&bw.dx[self.cfg.n_layers as usize], d_hidden);
        self.gpu.submit(&[], &bw.steps);
    }

    /// Apply one AdamW update to this half's parameters, on this half's own
    /// handle.
    ///
    /// Which handle is not a detail: a submit on one handle is not ordered
    /// against a submit on another, so stepping these weights from the other
    /// half's handle would race this half's next forward and it would read
    /// weights from before the update.
    pub fn adamw_step(&self, opt: &optim::Optim, t: u32, lr: f32, wd: f32, clip: Option<f32>) {
        self.adamw_step_scaled(opt, t, lr, wd, clip, 1.0)
    }

    /// [`Self::adamw_step`] with the accumulated gradient scaled - `1/n` for a
    /// minibatch of `n`, so one learning rate survives a change of batch size.
    pub fn adamw_step_scaled(
        &self,
        opt: &optim::Optim,
        t: u32,
        lr: f32,
        wd: f32,
        clip: Option<f32>,
        scale: f32,
    ) {
        opt.step(&self.gpu, &self.ps, t, lr, wd, 0.9, 0.999, 1e-8, clip, scale);
    }

    /// Block until this device has finished what it was given.
    pub fn poll_wait(&self) {
        self.gpu.poll_wait();
    }

    /// The final hidden states' device buffer - what the head reads.
    pub fn hidden_buf(&self) -> &DeviceBuffer {
        &self.x[self.cfg.n_layers as usize]
    }

    /// The buffer the reverse pass is seeded from. The head writes its
    /// hidden-state gradient straight into this, so a decision step costs no
    /// copy between the two halves.
    pub fn seed_buf(&self) -> &DeviceBuffer {
        &self.bwd.as_ref().expect("seed_buf on an inference build").dx[self.cfg.n_layers as usize]
    }

    /// The `(row0, len)` of each sequence in the current batch.
    pub fn spans(&self) -> &[(u32, u32)] {
        &self.spans
    }

    /// This half's device handle - what a profiler times its steps on.
    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    /// The recorded forward dispatches, for a profiler. Borrowed rather than
    /// run, so the caller decides how to time them.
    pub fn fwd_steps(&self) -> &[Step] {
        &self.steps
    }

    /// The recorded reverse dispatches, empty on an inference build.
    pub fn bwd_steps(&self) -> &[Step] {
        self.bwd.as_ref().map(|b| b.steps.as_slice()).unwrap_or(&[])
    }

    /// Run the reverse pass against whatever already sits in [`Encoder::seed_buf`].
    pub fn backward_seeded(&self) {
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

    /// The exact adjoint of [`Encoder::build_steps`], walked bottom up.
    ///
    /// Post-LayerNorm is what makes this differ structurally from a pre-LN
    /// tower: each block normalizes AFTER its residual add, so the LayerNorm
    /// adjoint sits between the block output and the residual fork rather than
    /// inside the branch. Getting that order wrong still produces finite,
    /// plausible gradients - which is why it is finite-difference checked on
    /// both backends rather than reasoned about.
    fn build_bwd_steps(&self) -> Vec<Step> {
        let g = &self.gpu;
        let c = &self.cfg;
        let bw = self.bwd.as_ref().expect("build_bwd_steps in training mode only");
        let n = self.rows;
        let h = c.d_model;
        let ff = c.d_ff;
        let hd = c.head_dim();
        let ln = block::LayerNormIds::resolve(g, self.k.layernorm, self.k.ln_stats, self.k.layernorm_dx);
        let cross = block::CrossIds { scores: self.k.scores_cross, softmax: self.k.softmax_cross, apply: self.k.apply_cross };
        let cross_bwd = block::CrossBwdIds::resolve(
            g,
            self.k.dscores_cross,
            self.k.dq_cross,
            self.k.dk_cross_acc,
            self.k.dv_cross_acc,
        );
        let gr = |name: &str| self.ps.g(name);
        let dw_gemm = |m: u32, k: u32| block::pick_gemm(m as usize, k as usize, self.k.matmul_dw, self.k.matmul_dw_reg, false);
        let dx_gemm = |m: u32, k: u32| block::pick_gemm(m as usize, k as usize, self.k.matmul_dx, self.k.matmul_dx_reg, false);
        let mut s: Vec<Step> = Vec::new();

        for l in (0..c.n_layers as usize).rev() {
            let lb = &self.layers[l];
            let p = format!("blocks.{l}");
            let d_out = &bw.dx[l + 1];

            // ---- LN2, then the FFN branch ----
            // `layernorm_dgamma` Params: [d_model, n_rows]; bufs [dy, x, mean, inv, dgamma].
            // `layernorm_dbeta`  Params: [d_model, n_rows]; bufs [dy, dbeta].
            s.push(block::ln_stats_fwd(g, &ln, &lb.ffn_pre, &bw.mean, &bw.inv, h, n, c.eps));
            s.push(g.step(self.k.ln_dgamma, &[d_out, &lb.ffn_pre, &bw.mean, &bw.inv, gr(&format!("{p}.ln2.weight"))], &[h, n], h));
            s.push(g.step(self.k.ln_dbeta, &[d_out, gr(&format!("{p}.ln2.bias"))], &[h, n], h));
            s.push(block::layernorm_dx_bwd(g, &ln, &lb.ffn_pre, self.w(&format!("{p}.ln2.weight")), d_out, &bw.d_ffn_pre, h, n, c.eps));

            // `ffn_pre = res + mlp_out`, so the MLP branch's incoming grad IS
            // `d_ffn_pre` and `res` also receives it directly.
            // `bias_grad`  Params: [m, n]; bufs [dy, dbias] - one thread per feature.
            // `matmul_dw`  Params: [m, k, n]; bufs [dy, x, dw] - ACCUMULATES.
            // `matmul_dx`  Params: [m, k, n, accumulate]; bufs [dy, w, dx].
            s.push(g.step(self.k.bias_grad, &[&bw.d_ffn_pre, gr(&format!("{p}.fc2.bias"))], &[n, h], h));
            let (dw, dwt) = dw_gemm(h, ff);
            s.push(g.step(dw, &[&bw.d_ffn_pre, &lb.h_act, gr(&format!("{p}.fc2.weight"))], &[n, ff, h], dwt));
            let (dx, dxt) = dx_gemm(n, ff);
            s.push(g.step(dx, &[&bw.d_ffn_pre, self.w(&format!("{p}.fc2.weight")), &bw.d_h_act], &[n, ff, h, 0], dxt));
            // The activation backward reads the PRE-activation hidden, never
            // the activated one.
            s.push(g.step(self.k.gelu_erf_bwd, &[&lb.h, &bw.d_h_act, &bw.d_h], &[n * ff], n * ff));
            s.push(g.step(self.k.bias_grad, &[&bw.d_h, gr(&format!("{p}.fc1.bias"))], &[n, ff], ff));
            let (dw, dwt) = dw_gemm(ff, h);
            s.push(g.step(dw, &[&bw.d_h, &lb.res, gr(&format!("{p}.fc1.weight"))], &[n, h, ff], dwt));
            let (dx, dxt) = dx_gemm(n, h);
            s.push(g.step(dx, &[&bw.d_h, self.w(&format!("{p}.fc1.weight")), &bw.d_tmp], &[n, h, ff, 0], dxt));
            s.push(g.step(self.k.add2, &[&bw.d_ffn_pre, &bw.d_tmp, &bw.d_res], &[n * h], n * h));

            // ---- LN1, then the attention branch ----
            s.push(block::ln_stats_fwd(g, &ln, &lb.res_pre, &bw.mean, &bw.inv, h, n, c.eps));
            s.push(g.step(self.k.ln_dgamma, &[&bw.d_res, &lb.res_pre, &bw.mean, &bw.inv, gr(&format!("{p}.ln1.weight"))], &[h, n], h));
            s.push(g.step(self.k.ln_dbeta, &[&bw.d_res, gr(&format!("{p}.ln1.bias"))], &[h, n], h));
            s.push(block::layernorm_dx_bwd(g, &ln, &lb.res_pre, self.w(&format!("{p}.ln1.weight")), &bw.d_res, &bw.d_res_pre, h, n, c.eps));

            s.push(g.step(self.k.bias_grad, &[&bw.d_res_pre, gr(&format!("{p}.proj.bias"))], &[n, h], h));
            let (dw, dwt) = dw_gemm(h, h);
            s.push(g.step(dw, &[&bw.d_res_pre, &lb.ctx, gr(&format!("{p}.proj.weight"))], &[n, h, h], dwt));
            let (dx, dxt) = dx_gemm(n, h);
            s.push(g.step(dx, &[&bw.d_res_pre, self.w(&format!("{p}.proj.weight")), &bw.d_ctx], &[n, h, h, 0], dxt));

            // Per-span attention backward, recomputing each chunk's scores and
            // probabilities from the cached qkv. `d_qkv` needs no clear: the
            // first chunk of every span ASSIGNS its region and later chunks
            // accumulate onto it.
            block::chunked_bidir_bwd(
                g,
                &cross,
                None,
                &cross_bwd,
                c.n_heads,
                hd,
                h,
                &lb.qkv,
                3 * h,
                0,
                h,
                2 * h,
                &bw.d_ctx,
                &bw.d_qkv,
                &self.scores,
                &self.probs,
                &bw.d_scores,
                &self.spans,
                self.chunk,
                None,
                &mut s,
            );

            s.push(g.step(self.k.bias_grad, &[&bw.d_qkv, gr(&format!("{p}.qkv.bias"))], &[n, 3 * h], 3 * h));
            let (dw, dwt) = dw_gemm(3 * h, h);
            s.push(g.step(dw, &[&bw.d_qkv, &self.x[l], gr(&format!("{p}.qkv.weight"))], &[n, h, 3 * h], dwt));
            let (dx, dxt) = dx_gemm(n, h);
            s.push(g.step(dx, &[&bw.d_qkv, self.w(&format!("{p}.qkv.weight")), &bw.d_tmp], &[n, h, 3 * h, 0], dxt));
            // `res_pre = x + attn_out`: the block input receives the residual
            // pass-through AND the attention branch.
            s.push(g.step(self.k.add2, &[&bw.d_res_pre, &bw.d_tmp, &bw.dx[l]], &[n * h], n * h));
        }

        // ---- embeddings ----
        s.push(block::ln_stats_fwd(g, &ln, &self.sum2, &bw.mean, &bw.inv, h, n, c.eps));
        s.push(g.step(self.k.ln_dgamma, &[&bw.dx[0], &self.sum2, &bw.mean, &bw.inv, gr("emb_ln.weight")], &[h, n], h));
        s.push(g.step(self.k.ln_dbeta, &[&bw.dx[0], gr("emb_ln.bias")], &[h, n], h));
        s.push(block::layernorm_dx_bwd(g, &ln, &self.sum2, self.w("emb_ln.weight"), &bw.dx[0], &bw.d_sum, h, n, c.eps));
        // Three gathers summed: addition fans the gradient out unchanged, so
        // every table scatters the SAME `d_sum` through its own index buffer.
        let eb = block::EmbBwdIds::resolve(g, self.k.emb_bwd);
        for (i, (index, table, table_rows)) in [
            (&self.ids, "tok.weight", c.vocab),
            (&self.pos_ids, "pos.weight", c.max_positions),
            (&self.type_ids, "type.weight", c.type_vocab),
        ]
        .into_iter()
        .enumerate()
        {
            s.push(block::emb_bwd_step(
                g,
                &eb,
                index,
                Some((&self.uniq[i], self.uniq_n[i])),
                &bw.d_sum,
                gr(table),
                n,
                h,
                table_rows,
            ));
        }
        s
    }

    // ---- parity / inference taps ----

    /// The final hidden states, `[rows, H]` row-major over the PACKED rows.
    pub fn hidden(&self) -> Vec<f32> {
        self.read(&self.x[self.cfg.n_layers as usize])
    }

    /// Layer `l`'s output (`l == 0` is the first block's output; use
    /// [`Encoder::embeddings`] for the pre-block residual).
    pub fn layer_out(&self, l: usize) -> Vec<f32> {
        self.read(&self.x[l + 1])
    }

    /// The post-embedding residual, after its LayerNorm.
    pub fn embeddings(&self) -> Vec<f32> {
        self.read(&self.x[0])
    }

    /// Layer `l`'s attention context, before the output projection.
    pub fn attn_ctx(&self, l: usize) -> Vec<f32> {
        self.read(&self.layers[l].ctx)
    }

    /// Layer `l`'s post-attention residual, after LN1.
    pub fn attn_out(&self, l: usize) -> Vec<f32> {
        self.read(&self.layers[l].res)
    }

    /// Layer `l`'s FFN hidden after its GELU, `[rows, d_ff]`.
    pub fn ffn_act(&self, l: usize) -> Vec<f32> {
        self.gpu.read(&self.layers[l].h_act, (self.rows * self.cfg.d_ff) as usize)
    }

    /// Mean of the final hidden states over each span - the sentence-transformer
    /// pooling head. `[spans, H]`.
    ///
    /// Packing is what makes this exact: there are no pad rows in the mean, so
    /// nothing has to be excluded from it.
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

    /// Read the live prefix of a `[cap_rows, H]` buffer - the rows this call
    /// actually packed, not the capacity.
    fn read(&self, b: &DeviceBuffer) -> Vec<f32> {
        self.gpu.read(b, (self.rows * self.cfg.d_model) as usize)
    }
}
