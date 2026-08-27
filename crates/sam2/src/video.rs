// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! SAM 2's VIDEO path: the temporal memory bank that makes a mask placed on one
//! frame follow the object through a clip.
//!
//! Three pieces, ported from `sam2.modeling.{memory_attention, memory_encoder,
//! sam2_base}` and `sam2.sam2_video_predictor`:
//!
//! 1. [`Sam2::memory_attention`] - 4 pre-norm layers over the current frame's
//!    `[H*W, d_model]` features. Self-attention with axial 2D RoPE, then
//!    cross-attention into the memory (`kv_in_dim = mem_dim`, so ONLY its k/v
//!    projections are narrow), then a ReLU MLP. The cross-attention's RoPE is
//!    the subtle part: the memory is `r` spatial slabs of the SAME `H*W` grid
//!    concatenated, so the query's frequency table is REPEATED `r` times down
//!    the keys (`rope_k_repeat`), and the object-pointer tokens on the end get
//!    NO rotation at all (`num_k_exclude_rope`).
//! 2. [`Sam2::memory_encoder`] - the predicted mask at image resolution is
//!    squeezed to the feature grid by a 4-stage stride-2 conv stack, added to a
//!    1x1 projection of the frame's top-level FPN feature, fused by two
//!    ConvNeXt blocks and projected to `mem_dim`.
//! 3. [`Tracker`] - the propagation loop: which memories frame *t* attends to,
//!    with which temporal position, and which object pointers ride along.
//!
//! ## What conditions frame *t*
//!
//! ```text
//! memory tokens  = [ cond frame slab | up to num_maskmem-1 recent slabs | object pointers ]
//!                    t_pos = 0         t_pos = 1 .. num_maskmem-1          (no spatial pos)
//! memory pos     = maskmem_pos_enc + maskmem_tpos_enc[num_maskmem - t_pos - 1]
//!                                   | sine(Δt / (max_ptrs-1)) -> obj_ptr_tpos_proj
//! ```
//!
//! One `d_model` object pointer becomes `d_model / mem_dim` memory tokens
//! (`Sam2Config::obj_ptr_tokens`), because the memory's channel width is
//! narrower than the model's.
//!
//! ## Deliberately not ported
//!
//! * **Multi-object tracking.** [`Tracker`] follows ONE object. The reference's
//!   `_apply_non_overlapping_constraints` only does anything for a batch of
//!   objects, and per-object slices are independent up to that constraint, so N
//!   objects is N trackers sharing one [`Encoded`] per frame - no new math, and
//!   the mask-sequence format ([`MaskSeq`]) already carries an `object_id`.
//! * **The mask-PROMPT entry point** (`add_new_mask` →
//!   `_use_mask_as_output`), whose only weight is `mask_downsample`. That
//!   tensor is imported and named in `Sam2Config::video_tensor_manifest`
//!   rather than swept under a skip list, so coverage closes at 154/154.
//! * **Reverse tracking** and **correction clicks on an already-tracked
//!   frame**. Both are bookkeeping over the same forward; the prompt frame is
//!   the only conditioning frame here.

use std::collections::BTreeMap;

use gpu_core::{f, DeviceBuffer, Step};
use model::block;
use vision::{Act, Conv, ConvNames, ConvSpec, LayerNorm2d, Ln2dNames, Norm, Shape};

use crate::hostpe;
use crate::import::Scope;
use crate::model::{idx, Decoded, Encoded, Prompt, Sam2};

/// Query rows per memory-attention dispatch. The cross-attention's key length
/// is `num_maskmem * H*W + obj_ptr_tokens` - 28 736 for hiera at 1024 - so an
/// unchunked `[H*W, keys]` score slab would be 470 MiB, twice (scores and
/// probabilities). Chunking the QUERY caps it at a few hundred MiB per
/// dispatch without changing a single value: softmax is per query row.
///
/// A multiple of 64 is REQUIRED, not preferred: the chunk slices `q` and the
/// context buffer at `chunk * d_model` floats and wgpu rejects a storage
/// binding whose offset is not 256-byte aligned.
const MEM_ATTN_CHUNK: u32 = 1024;

/// `a // b` with Python's flooring semantics, for a positive `b`.
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    if a % b != 0 && (a < 0) != (b < 0) {
        q - 1
    } else {
        q
    }
}

/// One frame's encoded memory, as the memory attention consumes it.
pub struct MemoryEntry {
    /// `[mem_dim, h, w]` NCHW - the golden's `maskmem_features`.
    pub features: DeviceBuffer,
    /// `[h*w, mem_dim]` NLC - the same buffer as memory tokens.
    pub features_nlc: DeviceBuffer,
    /// `[mem_dim, h, w]` NCHW - `PositionEmbeddingSine` over the memory grid.
    pub pos_enc: DeviceBuffer,
    /// `[h*w, mem_dim]` NLC.
    pub pos_enc_nlc: DeviceBuffer,
    /// `[d_model]` on the HOST: at most `max_obj_ptrs_in_encoder` of these are
    /// assembled into one uploaded token block per frame, so keeping them host
    /// side costs one 1 KiB copy and saves a gather.
    pub obj_ptr: Vec<f32>,
    /// `pred_obj_score_head` logit - `<= 0` means the object is absent or fully
    /// occluded on this frame.
    pub object_score: f32,
}

/// Everything one propagation step produced for one frame.
pub struct TrackStep {
    pub frame: usize,
    /// `[1, d_model, h, w]` - the memory-conditioned backbone feature the SAM
    /// heads ran on (`pix_feat_with_mem`). The image path's `enc.image_embed`
    /// in the same slot.
    pub pix_feat_with_mem: DeviceBuffer,
    /// The mask decoder's full output for this frame.
    pub decoded: Decoded,
    /// `[1, 1, 4*h, 4*w]` - the best mask's logits at decoder resolution.
    pub low_res_mask: DeviceBuffer,
    /// `[1, 1, image_size, image_size]` - the same mask at image resolution.
    pub high_res_mask: DeviceBuffer,
    pub object_score: f32,
    pub iou: f32,
    /// True for the frame the point prompt was placed on.
    pub is_cond: bool,
}

/// The small video constants, built ONCE per tracker and passed in.
///
/// Not rebuilt per call on purpose: the axial RoPE tables are
/// `[H*W, d_model/2]` twice over (4 MiB at 1024) and cost a host trig sweep to
/// derive, so re-deriving them inside every memory-attention call would pay
/// that on every frame for a value that never changes.
pub struct VideoConsts {
    /// `[d_model]`.
    no_mem_pos_enc: Vec<f32>,
    /// `[mem_dim]`.
    no_obj_embed_spatial: Vec<f32>,
    /// `obj_ptr_tpos_proj` as `(W [mem_dim, d_model], b [mem_dim])`.
    tpos_proj: (Vec<f32>, Vec<f32>),
    /// `(cos, sin)` axial RoPE tables for the memory grid, already on device.
    rope: (DeviceBuffer, DeviceBuffer),
    /// Per fuser layer, the `film_chan` `[2*d_model]` pair that applies
    /// `CXBlock.gamma`: `scale = gamma - 1` then a zero shift, because
    /// `film_chan` computes `x * (1 + s) + b`.
    fuser_gamma: Vec<DeviceBuffer>,
}

impl VideoConsts {
    /// Read them off `m`'s [`ParamStore`](paramstore::ParamStore).
    pub fn read(m: &Sam2) -> VideoConsts {
        let cfg = &m.cfg;
        let g = &m.gpu;
        let d = cfg.d_model;
        let grid = cfg.image_embedding_size();
        let (cos_t, sin_t) = hostpe::axial_rope_tables(d, grid, grid, cfg.memory_rope_theta);
        let fuser_gamma = (0..cfg.memory_fuser_layers)
            .map(|l| {
                let gamma = m.ps.read_weight(g, &format!("memory_encoder.fuser.layers.{l}.gamma"));
                let mut sb = vec![0.0f32; 2 * d as usize];
                for (i, v) in gamma.iter().enumerate() {
                    sb[i] = *v - 1.0;
                }
                g.storage_init("sam2_fuser_gamma", &sb)
            })
            .collect();
        VideoConsts {
            no_mem_pos_enc: m.ps.read_weight(g, "no_mem_pos_enc"),
            no_obj_embed_spatial: m.ps.read_weight(g, "no_obj_embed_spatial"),
            tpos_proj: (
                m.ps.read_weight(g, "obj_ptr_tpos_proj.weight"),
                m.ps.read_weight(g, "obj_ptr_tpos_proj.bias"),
            ),
            rope: (g.storage_init("sam2_rope_cos", &cos_t), g.storage_init("sam2_rope_sin", &sin_t)),
            fuser_gamma,
        }
    }
}

impl Sam2 {
    fn rope_id(&self) -> usize {
        idx(&self.gpu, "rope_interleave_table")
    }

    /// The top-level FPN level the memory bank reads: `backbone_fpn[-1]`, which
    /// is the level `neck()` adds `no_mem_embed` to for the image path. NOT
    /// `enc.image_embed` - that one already has the no-memory embedding in it.
    pub(crate) fn mem_level(&self) -> usize {
        self.cfg.backbone_channel_list.len() - 1 - self.cfg.scalp as usize
    }

    /// `apply_rotary_enc` on `x` `[rows, dim]` (one head), out of place.
    ///
    /// `table_rows` is how many rows the RoPE table has; `rows` may be an exact
    /// multiple of it, which is `repeat_freqs_k`: each block of `table_rows`
    /// keys re-uses the same frequencies, because each is the same spatial grid
    /// from a different frame.
    fn rope_rows(&self, s: &mut Vec<Step>, x: &DeviceBuffer, y: &DeviceBuffer, rows: u32, dim: u32, table_rows: u32, rope: &(DeviceBuffer, DeviceBuffer)) {
        assert_eq!(rows % table_rows, 0, "rope: {rows} rows is not a multiple of the {table_rows}-row table");
        let half = dim / 2;
        let blk = table_rows as u64 * dim as u64;
        for b in 0..(rows / table_rows) as u64 {
            s.push(self.gpu.step_sliced(
                self.rope_id(),
                &[x, &rope.0, &rope.1, y],
                &[(b * blk, 0), (0, 0), (0, 0), (b * blk, 0)],
                &[table_rows, 1, dim, half],
                table_rows * half,
            ));
        }
    }

    /// One `RoPEAttention` with a single head, query-chunked.
    ///
    /// `q_in` is `[tq, d_model]`; `kv_in` is `[tk, kv_dim]` for the keys and
    /// `v_in` `[tk, kv_dim]` for the values (they differ in the cross-attention,
    /// where the keys carry the memory's positional encoding and the values do
    /// not). `n_k_rope` keys are rotated, the rest are not.
    #[allow(clippy::too_many_arguments)]
    fn rope_attention(
        &self,
        prefix: &str,
        q_in: &DeviceBuffer,
        k_in: &DeviceBuffer,
        v_in: &DeviceBuffer,
        tq: u32,
        tk: u32,
        kv_dim: u32,
        n_k_rope: u32,
        rope: &(DeviceBuffer, DeviceBuffer),
        out: &DeviceBuffer,
    ) {
        let g = &self.gpu;
        let d = self.cfg.d_model;
        let heads = self.cfg.memory_attention_heads;
        let hd = d / heads;
        assert_eq!(heads, 1, "sam2 video: the released memory attention is single-head");

        // ---- projections ----
        let q0 = g.storage(tq as u64 * d as u64);
        let k0 = g.storage(tk as u64 * d as u64);
        let v = g.storage(tk as u64 * d as u64);
        let mut steps = Vec::new();
        self.linear(&mut steps, q_in, &q0, tq, d, d, &format!("{prefix}.q_proj.weight"), &format!("{prefix}.q_proj.bias"));
        self.linear(&mut steps, k_in, &k0, tk, kv_dim, d, &format!("{prefix}.k_proj.weight"), &format!("{prefix}.k_proj.bias"));
        self.linear(&mut steps, v_in, &v, tk, kv_dim, d, &format!("{prefix}.v_proj.weight"), &format!("{prefix}.v_proj.bias"));
        g.submit(&[], &steps);

        // ---- rotary, out of place; the un-rotated key tail is copied through ----
        let q = g.storage(tq as u64 * d as u64);
        let k = g.storage(tk as u64 * d as u64);
        let mut steps = Vec::new();
        self.rope_rows(&mut steps, &q0, &q, tq, d, tq, rope);
        // `axpy` is read-modify-write, so `k` goes in the CLEAR list; the rope
        // steps then OVERWRITE the rotated prefix in the same submit.
        steps.push(g.step(self.ids.axpy, &[&k, &k0], &[tk * d, f(1.0)], tk * d));
        if n_k_rope > 0 {
            self.rope_rows(&mut steps, &k0, &k, n_k_rope, d, tq, rope);
        }
        g.submit(&[&k], &steps);

        // ---- scores / softmax / apply, chunked over the query ----
        let kt = g.storage(d as u64 * tk as u64);
        let ctxb = g.storage(tq as u64 * d as u64);
        g.submit(&[], &[g.step(self.ids.key_minor.0, &[&k, &kt], &[tk, d, d, 0], d * tk)]);
        let mut q0r = 0u32;
        while q0r < tq {
            let rows = MEM_ATTN_CHUNK.min(tq - q0r);
            let off = q0r as u64 * d as u64;
            let scores = g.storage(heads as u64 * rows as u64 * tk as u64);
            let probs = g.storage(heads as u64 * rows as u64 * tk as u64);
            g.submit(
                &[],
                &[
                    g.step_sliced(
                        self.ids.key_minor.1,
                        &[&q, &kt, &scores],
                        &[(off, 0), (0, 0), (0, 0)],
                        &[1, heads, rows, tk, hd, d, 0],
                        heads * rows * tk,
                    ),
                    g.step(self.ids.cross.softmax, &[&scores, &probs], &[1, heads, rows, tk], heads * rows),
                    g.step_sliced(
                        self.ids.cross.apply,
                        &[&probs, &v, &ctxb],
                        &[(0, 0), (0, 0), (off, 0)],
                        &[1, heads, rows, tk, hd, d, 0, d],
                        heads * rows * hd,
                    ),
                ],
            );
            q0r += rows;
        }
        let mut steps = Vec::new();
        self.linear(&mut steps, &ctxb, out, tq, d, d, &format!("{prefix}.out_proj.weight"), &format!("{prefix}.out_proj.bias"));
        g.submit(&[], &steps);
    }

    /// `MemoryAttention.forward`.
    ///
    /// `curr` / `curr_pos` are `[tq, d_model]` NLC; `memory` / `memory_pos` are
    /// `[tk, mem_dim]`. The last `n_obj_ptr_tokens` memory rows are object
    /// pointers and are excluded from the key rotation.
    ///
    /// Returns `[tq, d_model]` NLC.
    #[allow(clippy::too_many_arguments)]
    pub fn memory_attention(
        &self,
        curr: &DeviceBuffer,
        curr_pos: &DeviceBuffer,
        memory: &DeviceBuffer,
        memory_pos: &DeviceBuffer,
        tq: u32,
        tk: u32,
        n_obj_ptr_tokens: u32,
        consts: &VideoConsts,
        taps: &mut Vec<DeviceBuffer>,
    ) -> DeviceBuffer {
        self.assert_video();
        let g = &self.gpu;
        let cfg = &self.cfg;
        let d = cfg.d_model;
        let m = cfg.mem_dim;
        let ff = cfg.memory_attention_ff;
        let n_k_rope = tk - n_obj_ptr_tokens;

        // `pos_enc_at_input`: `curr + 0.1 * curr_pos`, NOT a plain add.
        let mut x = g.storage(tq as u64 * d as u64);
        {
            let mut steps = vec![g.step(self.ids.axpy, &[&x, curr], &[tq * d, f(1.0)], tq * d)];
            if cfg.memory_pos_enc_at_input {
                steps.push(g.step(self.ids.axpy, &[&x, curr_pos], &[tq * d, f(0.1)], tq * d));
            }
            g.submit(&[&x], &steps);
        }

        // The cross-attention keys read `memory + pos`; the values read `memory`.
        let k_in = self.add(memory, memory_pos, tk * m);

        for l in 0..cfg.memory_attention_layers {
            let p = format!("memory_attention.layers.{l}");

            // ---- self-attention (pos_enc_at_attn = false: q = k = norm1(x)) ----
            let n1 = self.layernorm(&x, &format!("{p}.norm1"), tq, d);
            let sa = g.storage(tq as u64 * d as u64);
            self.rope_attention(&format!("{p}.self_attn"), &n1, &n1, &n1, tq, tq, d, tq, &consts.rope, &sa);
            let x1 = self.add(&x, &sa, tq * d);

            // ---- cross-attention into the memory ----
            // `pos_enc_at_cross_attn_queries = false`, so the query is the bare
            // norm2 output; `..._keys = true`, hence `k_in` above.
            let n2 = self.layernorm(&x1, &format!("{p}.norm2"), tq, d);
            let ca = g.storage(tq as u64 * d as u64);
            self.rope_attention(&format!("{p}.cross_attn_image"), &n2, &k_in, memory, tq, tk, m, n_k_rope, &consts.rope, &ca);
            let x2 = self.add(&x1, &ca, tq * d);

            // ---- MLP (activation = relu) ----
            let n3 = self.layernorm(&x2, &format!("{p}.norm3"), tq, d);
            let h = g.storage(tq as u64 * ff as u64);
            let a = g.storage(tq as u64 * ff as u64);
            let o = g.storage(tq as u64 * d as u64);
            let mut steps = Vec::new();
            self.linear(&mut steps, &n3, &h, tq, d, ff, &format!("{p}.linear1.weight"), &format!("{p}.linear1.bias"));
            steps.push(self.act_step(&h, &a, tq * ff, Act::Relu));
            self.linear(&mut steps, &a, &o, tq, ff, d, &format!("{p}.linear2.weight"), &format!("{p}.linear2.bias"));
            g.submit(&[], &steps);
            x = self.add(&x2, &o, tq * d);
            taps.push(self.copy_of(&x, tq * d));
        }
        // `MemoryAttention.norm` is applied to the LAST layer's output, and the
        // per-layer taps above are pre-norm - which is what the reference's own
        // forward hooks see.
        self.layernorm(&x, "memory_attention.norm", tq, d)
    }

    /// `MemoryEncoder.forward(pix_feat, mask_for_mem, skip_mask_sigmoid=True)`.
    ///
    /// `pix_feat` is `[1, d_model, h, w]` NCHW (the top FPN level, WITHOUT
    /// `no_mem_embed`); `mask_for_mem` is `[1, 1, image_size, image_size]`,
    /// already through `sigmoid * scale + bias`. Returns the `[1, mem_dim, h, w]`
    /// memory features; the positional encoding is a constant per grid and is
    /// built by [`Tracker`].
    pub fn memory_encoder(
        &self,
        pix_feat: &DeviceBuffer,
        mask_for_mem: &DeviceBuffer,
        consts: &VideoConsts,
        taps: &mut Vec<DeviceBuffer>,
    ) -> DeviceBuffer {
        self.assert_video();
        let cfg = &self.cfg;
        let g = &self.gpu;
        let ctx = self.ctx();
        let d = cfg.d_model;
        let grid = cfg.image_embedding_size();

        // ---- MaskDownSampler: (conv s2 -> LayerNorm2d -> GELU) x N, then 1x1 ----
        let chans = cfg.mem_mask_chans();
        let mut side = cfg.image_size;
        let mut cur: Option<DeviceBuffer> = None;
        for i in 0..cfg.mem_mask_layers() as usize {
            let (cin, cout) = (chans[i], chans[i + 1]);
            let pfx = format!("memory_encoder.mask_downsampler.encoder.{}", 3 * i);
            let cv = Conv::with_names(
                &ctx,
                &pfx,
                ConvNames::torch_flat(&pfx),
                Shape::new(1, cin, side, side),
                ConvSpec::relu(cout, cfg.mem_mask_kernel, cfg.mem_mask_stride, cfg.mem_mask_pad)
                    .with_norm(Norm::None)
                    .with_act(Act::None)
                    .with_bias(),
                false,
            );
            cv.forward(&ctx, &self.ps, cur.as_ref().unwrap_or(mask_for_mem));
            side /= cfg.mem_mask_stride;
            let n = cout * side * side;
            let ln = LayerNorm2d::new(
                &ctx,
                Ln2dNames::torch(&format!("memory_encoder.mask_downsampler.encoder.{}", 3 * i + 1)),
                Shape::new(1, cout, side, side),
                cfg.ln2d_eps,
            );
            ln.forward(&ctx, &self.ps, cv.out());
            let act = g.storage(n as u64);
            g.submit(&[], &[self.act_step(ln.out(), &act, n, Act::GeluErf)]);
            cur = Some(act);
        }
        assert_eq!(side, grid, "mask downsampler landed at {side}, not the {grid} feature grid");
        let cend = *chans.last().unwrap();
        let last = 3 * cfg.mem_mask_layers();
        let pfx = format!("memory_encoder.mask_downsampler.encoder.{last}");
        let proj = Conv::with_names(
            &ctx,
            &pfx,
            ConvNames::torch_flat(&pfx),
            Shape::new(1, cend, grid, grid),
            ConvSpec::relu(d, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
            false,
        );
        proj.forward(&ctx, &self.ps, cur.as_ref().expect("mask downsampler has at least one layer"));
        let n = d * grid * grid;
        let mask_down = self.copy_of(proj.out(), n);
        taps.push(self.copy_of(&mask_down, n));

        // ---- pix_feat_proj + the downsampled mask ----
        let pfx = "memory_encoder.pix_feat_proj";
        let pp = Conv::with_names(
            &ctx,
            pfx,
            ConvNames::torch_flat(pfx),
            Shape::new(1, d, grid, grid),
            ConvSpec::relu(d, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
            false,
        );
        pp.forward(&ctx, &self.ps, pix_feat);
        taps.push(self.copy_of(pp.out(), n));
        let mut x = self.add(pp.out(), &mask_down, n);

        // ---- Fuser: ConvNeXt blocks ----
        for l in 0..cfg.memory_fuser_layers {
            x = self.cx_block(&x, l, &consts.fuser_gamma[l as usize]);
            taps.push(self.copy_of(&x, n));
        }

        // ---- out_proj to mem_dim ----
        let pfx = "memory_encoder.out_proj";
        let op = Conv::with_names(
            &ctx,
            pfx,
            ConvNames::torch_flat(pfx),
            Shape::new(1, d, grid, grid),
            ConvSpec::relu(cfg.mem_dim, 1, 1, 0).with_norm(Norm::None).with_act(Act::None).with_bias(),
            false,
        );
        op.forward(&ctx, &self.ps, &x);
        self.copy_of(op.out(), cfg.mem_dim * grid * grid)
    }

    /// One `memory_encoder.fuser.layers.{l}` ConvNeXt block, NCHW in and out.
    ///
    /// `pwconv1`/`pwconv2` are `nn.Linear` over channels-last, which is a matmul
    /// over the NLC rows - and the channels-first `LayerNorm2d` before them IS a
    /// layernorm over those same rows, so the block permutes exactly twice.
    fn cx_block(&self, x: &DeviceBuffer, layer: u32, gamma_sb: &DeviceBuffer) -> DeviceBuffer {
        let cfg = &self.cfg;
        let g = &self.gpu;
        let ctx = self.ctx();
        let d = cfg.d_model;
        let grid = cfg.image_embedding_size();
        let hw = grid * grid;
        let n = d * hw;
        let p = format!("memory_encoder.fuser.layers.{layer}");

        let pfx = format!("{p}.dwconv");
        let dw = Conv::with_names(
            &ctx,
            &pfx,
            ConvNames::torch_flat(&pfx),
            Shape::new(1, d, grid, grid),
            ConvSpec::relu(d, cfg.memory_fuser_kernel, 1, cfg.memory_fuser_pad)
                .with_groups(d)
                .with_norm(Norm::None)
                .with_act(Act::None)
                .with_bias(),
            false,
        );
        dw.forward(&ctx, &self.ps, x);

        let nlc = g.storage(n as u64);
        let mut steps = Vec::new();
        self.to_nlc(&mut steps, dw.out(), &nlc, d, hw);
        g.submit(&[], &steps);
        let normed = {
            let out = g.storage(n as u64);
            g.submit(
                &[],
                &[block::layernorm_fwd(
                    g,
                    &self.ids.ln,
                    &nlc,
                    self.ps.w(&format!("{p}.norm.weight")),
                    self.ps.w(&format!("{p}.norm.bias")),
                    &out,
                    d,
                    hw,
                    cfg.ln2d_eps,
                )],
            );
            out
        };
        let h = g.storage(hw as u64 * 4 * d as u64);
        let a = g.storage(hw as u64 * 4 * d as u64);
        let o = g.storage(n as u64);
        let mut steps = Vec::new();
        self.linear(&mut steps, &normed, &h, hw, d, 4 * d, &format!("{p}.pwconv1.weight"), &format!("{p}.pwconv1.bias"));
        steps.push(self.act_step(&h, &a, hw * 4 * d, Act::GeluErf));
        self.linear(&mut steps, &a, &o, hw, 4 * d, d, &format!("{p}.pwconv2.weight"), &format!("{p}.pwconv2.bias"));
        let nchw = g.storage(n as u64);
        self.to_nchw(&mut steps, &o, &nchw, d, hw);
        // `gamma * y`: `film_chan` computes `x * (1 + s) + b`, so the uploaded
        // pair is `(gamma - 1, 0)` - see `VideoConsts::fuser_gamma`.
        let scaled = g.storage(n as u64);
        steps.push(g.step(idx(g, "film_chan"), &[&nchw, gamma_sb, &scaled], &[1, d, grid, grid], n));
        g.submit(&[], &steps);
        self.add(x, &scaled, n)
    }

    fn assert_video(&self) {
        assert_eq!(
            self.scope,
            Scope::Video,
            "sam2: the video memory bank needs a model built with Sam2::new_video \
             (import::import_scoped(.., Scope::Video)); this one holds the image path only"
        );
    }
}

// ===========================================================================
// the propagation loop
// ===========================================================================

/// Tracks ONE object through a clip: a point prompt on one frame, then a mask
/// on every frame, each conditioned on the memory of the frames before it.
///
/// The caller drives the encoder, because the Hiera trunk is ~99 % of the work
/// and a caller may already have the [`Encoded`] (a second object on the same
/// frame, a re-prompt, a cached clip):
///
/// ```text
/// let mut t = Tracker::new(&model, n_frames, 0);
/// t.prompt(&enc_of_frame_k, k, &prompt);         // the conditioning frame
/// for f in k+1..n_frames { t.track(&enc_of[f], f); }
/// ```
pub struct Tracker<'a> {
    m: &'a Sam2,
    consts: VideoConsts,
    num_frames: usize,
    object_id: u32,
    cond: BTreeMap<usize, MemoryEntry>,
    non_cond: BTreeMap<usize, MemoryEntry>,
    /// `PositionEmbeddingSine(mem_pos_sine_num_pos_feats)` over the memory
    /// grid - one constant table, shared by every entry.
    mem_pos: DeviceBuffer,
    mem_pos_nlc: DeviceBuffer,
}

impl<'a> Tracker<'a> {
    pub fn new(m: &'a Sam2, num_frames: usize, object_id: u32) -> Tracker<'a> {
        m.assert_video();
        let cfg = &m.cfg;
        let grid = cfg.image_embedding_size();
        let table = hostpe::sine(cfg.mem_pos_sine_num_pos_feats, cfg.pos_sine_temperature, grid, grid);
        let mem_pos = m.gpu.storage_init("sam2_mem_possine", &table);
        let mem_pos_nlc = m.gpu.storage(table.len() as u64);
        let mut steps = Vec::new();
        m.to_nlc(&mut steps, &mem_pos, &mem_pos_nlc, cfg.mem_dim, grid * grid);
        m.gpu.submit(&[], &steps);
        Tracker {
            m,
            consts: VideoConsts::read(m),
            num_frames,
            object_id,
            cond: BTreeMap::new(),
            non_cond: BTreeMap::new(),
            mem_pos,
            mem_pos_nlc,
        }
    }

    pub fn object_id(&self) -> u32 {
        self.object_id
    }

    /// The `Prompt` a TRACKING frame runs with: no real points, so the
    /// reference feeds `point_coords = zeros(1, 1, 2)` with label `-1` and the
    /// prompt encoder appends its own padding point on top - TWO padding tokens,
    /// not one. `multimask_output` is on because
    /// `multimask_output_for_tracking` is, and `0 <= num_pts <= 1` holds.
    pub fn tracking_prompt() -> Prompt {
        Prompt { coords: vec![(0.0, 0.0)], labels: vec![-1.0], mask_lowres: None, multimask_output: true }
    }

    /// Place the point prompt. This is the conditioning frame: it runs the SAM
    /// heads on `enc.image_embed` (`directly_add_no_mem_embed`) with no memory
    /// at all, then encodes its own memory for everything that follows.
    pub fn prompt(&mut self, enc: &Encoded, frame: usize, prompt: &Prompt) -> TrackStep {
        assert!(frame < self.num_frames, "prompt frame {frame} is past the {}-frame clip", self.num_frames);
        assert!(self.cond.is_empty(), "sam2 video: this tracker already has a conditioning frame");
        let decoded = self.m.decode_with(enc, &enc.image_embed, prompt);
        self.finish(enc, frame, decoded, true, self.m.copy_of(&enc.image_embed, self.m.cfg.d_model * self.m.cfg.image_embedding_size().pow(2)))
    }

    /// Propagate onto `frame`, conditioning on every memory recorded so far.
    pub fn track(&mut self, enc: &Encoded, frame: usize) -> TrackStep {
        assert!(frame < self.num_frames, "frame {frame} is past the {}-frame clip", self.num_frames);
        assert!(!self.cond.is_empty(), "sam2 video: track() before prompt() has no memory to condition on");
        let pix = self.memory_conditioned(enc, frame);
        let decoded = self.m.decode_with(enc, &pix, &Tracker::tracking_prompt());
        self.finish(enc, frame, decoded, false, pix)
    }

    /// `_prepare_memory_conditioned_features` for a non-conditioning frame.
    fn memory_conditioned(&self, enc: &Encoded, frame: usize) -> DeviceBuffer {
        let m = self.m;
        let cfg = &m.cfg;
        let g = &m.gpu;
        let d = cfg.d_model;
        let md = cfg.mem_dim;
        let grid = cfg.image_embedding_size();
        let hw = grid * grid;

        // ---- which memories, at which temporal position ----
        let mut slabs: Vec<(u32, &MemoryEntry)> = Vec::new();
        for e in self.cond.values() {
            slabs.push((0, e));
        }
        let stride = cfg.memory_temporal_stride_for_eval as i64;
        for t_pos in 1..cfg.num_maskmem {
            let t_rel = (cfg.num_maskmem - t_pos) as i64;
            let prev = if t_rel == 1 {
                // `t_rel == 1` takes the immediately preceding frame whatever
                // the stride is.
                frame as i64 - 1
            } else {
                // Python's `//` FLOORS; Rust's `/` truncates toward zero. They
                // agree for every non-negative numerator, and the released
                // configs all use stride 1 - but at stride > 1 on an early
                // frame the numerator IS negative, and truncation would name a
                // real frame where the reference names one before the clip.
                floor_div(frame as i64 - 2, stride) * stride - (t_rel - 2) * stride
            };
            if prev < 0 {
                continue;
            }
            if let Some(e) = self.non_cond.get(&(prev as usize)) {
                slabs.push((t_pos, e));
            }
        }

        // ---- object pointers, newest-relevant first, exactly as the reference
        // orders them: conditioning frames (in frame order) then the walk back
        // from `frame` one step at a time. ----
        let max_ptrs = cfg.max_obj_ptrs_in_encoder.min(self.num_frames as u32);
        let mut ptrs: Vec<(f32, &Vec<f32>)> = Vec::new();
        for (t, e) in self.cond.iter() {
            // `only_obj_ptrs_in_the_past_for_eval` with forward tracking.
            if *t <= frame {
                ptrs.push(((frame as i64 - *t as i64) as f32, &e.obj_ptr));
            }
        }
        for t_diff in 1..max_ptrs as i64 {
            let t = frame as i64 - t_diff;
            if t < 0 || t >= self.num_frames as i64 {
                break;
            }
            if let Some(e) = self.non_cond.get(&(t as usize)) {
                ptrs.push((t_diff as f32, &e.obj_ptr));
            }
        }

        let per_ptr = cfg.obj_ptr_tokens();
        let n_ptr_tokens = ptrs.len() as u32 * per_ptr;
        let n_spatial = slabs.len() as u32 * hw;
        let tk = n_spatial + n_ptr_tokens;
        assert!(tk > 0, "sam2 video: frame {frame} has no memory to attend to");

        // ---- the memory token block ----
        let memory = g.storage(tk as u64 * md as u64);
        let memory_pos = g.storage(tk as u64 * md as u64);
        let slab = hw as u64 * md as u64;
        let mut steps = Vec::new();
        for (i, (t_pos, e)) in slabs.iter().enumerate() {
            let off = i as u64 * slab;
            steps.push(g.step_sliced(m.ids.axpy, &[&memory, &e.features_nlc], &[(off, 0), (0, 0)], &[hw * md, f(1.0)], hw * md));
            steps.push(g.step_sliced(m.ids.axpy, &[&memory_pos, &e.pos_enc_nlc], &[(off, 0), (0, 0)], &[hw * md, f(1.0)], hw * md));
            // `maskmem_tpos_enc[num_maskmem - t_pos - 1]`, broadcast down the
            // slab's rows - `bias_add` IS that broadcast.
            let row = (cfg.num_maskmem - t_pos - 1) as u64 * md as u64;
            steps.push(g.step_sliced(
                m.ids.bias_add,
                &[&memory_pos, m.ps.w("maskmem_tpos_enc")],
                &[(off, 0), (row, 0)],
                &[hw, md],
                hw * md,
            ));
        }
        if n_ptr_tokens > 0 {
            // One pointer becomes `d_model / mem_dim` consecutive tokens
            // (`reshape(-1, B, C//mem_dim, mem_dim).permute(0,2,1,3).flatten(0,1)`
            // with B = 1 is exactly a row-major reshape), and its temporal
            // encoding is repeated over them (`repeat_interleave`).
            let mut ptr_block: Vec<f32> = Vec::with_capacity(n_ptr_tokens as usize * md as usize);
            let mut pos_rows: Vec<f32> = Vec::with_capacity(ptrs.len() * d as usize);
            let t_diff_max = (max_ptrs - 1) as f32;
            for (dt, p) in ptrs.iter() {
                assert_eq!(p.len(), d as usize, "object pointer is [d_model]");
                ptr_block.extend_from_slice(p);
                pos_rows.extend_from_slice(&hostpe::sine_1d(dt / t_diff_max, d, cfg.pos_sine_temperature));
            }
            let projected = hostpe::linear_rows(&pos_rows, &self.consts.tpos_proj.0, &self.consts.tpos_proj.1, d as usize, md as usize);
            let mut pos_block: Vec<f32> = Vec::with_capacity(n_ptr_tokens as usize * md as usize);
            for row in projected.chunks(md as usize) {
                for _ in 0..per_ptr {
                    pos_block.extend_from_slice(row);
                }
            }
            let pb = g.storage_init("sam2_objptr_tokens", &ptr_block);
            let pp = g.storage_init("sam2_objptr_pos", &pos_block);
            let off = n_spatial as u64 * md as u64;
            let n = n_ptr_tokens * md;
            steps.push(g.step_sliced(m.ids.axpy, &[&memory, &pb], &[(off, 0), (0, 0)], &[n, f(1.0)], n));
            steps.push(g.step_sliced(m.ids.axpy, &[&memory_pos, &pp], &[(off, 0), (0, 0)], &[n, f(1.0)], n));
            g.submit(&[&memory, &memory_pos], &steps);
        } else {
            g.submit(&[&memory, &memory_pos], &steps);
        }

        // ---- the current frame, as NLC tokens plus its sine encoding ----
        let lvl = m.mem_level();
        let curr = g.storage(hw as u64 * d as u64);
        let curr_pos = g.storage(hw as u64 * d as u64);
        let mut steps = Vec::new();
        m.to_nlc(&mut steps, &enc.fpn[lvl], &curr, d, hw);
        m.to_nlc(&mut steps, &enc.pos_sine[lvl], &curr_pos, d, hw);
        g.submit(&[], &steps);

        let mut taps = Vec::new();
        let out = m.memory_attention(&curr, &curr_pos, &memory, &memory_pos, hw, tk, n_ptr_tokens, &self.consts, &mut taps);
        let nchw = g.storage(d as u64 * hw as u64);
        let mut steps = Vec::new();
        m.to_nchw(&mut steps, &out, &nchw, d, hw);
        g.submit(&[], &steps);
        nchw
    }

    /// `_encode_memory_in_output` plus the bookkeeping: pick the best mask, run
    /// the memory encoder on it, and record the entry.
    fn finish(&mut self, enc: &Encoded, frame: usize, decoded: Decoded, is_cond: bool, pix_feat_with_mem: DeviceBuffer) -> TrackStep {
        let m = self.m;
        let cfg = &m.cfg;
        let g = &m.gpu;
        let grid = cfg.image_embedding_size();
        let side = cfg.image_size;
        let low_per = (4 * grid) * (4 * grid);
        let hi_per = side * side;
        let best = decoded.best_iou_index as u32;
        let iou = decoded.ious[decoded.best_iou_index];
        let object_score = g.read(&decoded.object_score_logits, 1)[0];

        let low_res_mask = g.storage(low_per as u64);
        let high_res_mask = g.storage(hi_per as u64);
        g.submit(
            &[&low_res_mask, &high_res_mask],
            &[
                g.step_sliced(m.ids.axpy, &[&low_res_mask, &decoded.low_res_multimasks], &[(0, 0), (best as u64 * low_per as u64, 0)], &[low_per, f(1.0)], low_per),
                g.step_sliced(m.ids.axpy, &[&high_res_mask, &decoded.high_res_multimasks], &[(0, 0), (best as u64 * hi_per as u64, 0)], &[hi_per, f(1.0)], hi_per),
            ],
        );

        // `mask_for_mem = sigmoid(logits) * scale + bias` - the memory encoder is
        // then told the sigmoid is already applied.
        let sig = g.storage(hi_per as u64);
        let scaled = g.storage(hi_per as u64);
        // `film_chan` with one channel: `x * (1 + s) + b`.
        let sb = g.storage_init("sam2_memenc_sb", &[cfg.sigmoid_scale_for_mem_enc - 1.0, cfg.sigmoid_bias_for_mem_enc]);
        g.submit(
            &[],
            &[
                m.act_step(&high_res_mask, &sig, hi_per, Act::Sigmoid),
                g.step(idx(g, "film_chan"), &[&sig, &sb, &scaled], &[1, 1, side, side], hi_per),
            ],
        );

        let lvl = m.mem_level();
        let mut taps = Vec::new();
        let features = m.memory_encoder(&enc.fpn[lvl], &scaled, &self.consts, &mut taps);
        let n = cfg.mem_dim * grid * grid;
        if object_score <= 0.0 {
            // `no_obj_embed_spatial`: the frame is predicted occluded, so the
            // memory says so rather than carrying an empty mask silently.
            let v = g.storage_init("sam2_no_obj_spatial", &self.consts.no_obj_embed_spatial);
            g.submit(&[], &[g.step(m.ids.add_chan_inplace, &[&features, &v], &[n, cfg.mem_dim, grid * grid], n)]);
        }
        let features_nlc = g.storage(n as u64);
        let mut steps = Vec::new();
        m.to_nlc(&mut steps, &features, &features_nlc, cfg.mem_dim, grid * grid);
        g.submit(&[], &steps);

        let entry = MemoryEntry {
            features,
            features_nlc,
            pos_enc: m.copy_of(&self.mem_pos, n),
            pos_enc_nlc: m.copy_of(&self.mem_pos_nlc, n),
            obj_ptr: g.read(&decoded.obj_ptr, cfg.d_model as usize),
            object_score,
        };
        if is_cond {
            self.cond.insert(frame, entry);
        } else {
            self.non_cond.insert(frame, entry);
        }
        TrackStep { frame, pix_feat_with_mem, decoded, low_res_mask, high_res_mask, object_score, iou, is_cond }
    }

    /// The memory recorded for `frame`, for tests and for a caller that wants to
    /// inspect the bank.
    pub fn memory(&self, frame: usize) -> Option<&MemoryEntry> {
        self.cond.get(&frame).or_else(|| self.non_cond.get(&frame))
    }

    /// `no_mem_pos_enc` - the dummy memory position the reference uses when a
    /// conditioning frame goes through the transformer instead of taking the
    /// `directly_add_no_mem_embed` shortcut. Exposed so the constant is not
    /// silently unread: this port always takes the shortcut (the released
    /// configs all set `directly_add_no_mem_embed: true`).
    pub fn no_mem_pos_enc(&self) -> &[f32] {
        &self.consts.no_mem_pos_enc
    }
}
