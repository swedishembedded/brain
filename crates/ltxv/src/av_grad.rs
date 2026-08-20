// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host reference forward + analytic backward for ONE `LtxAvBlock`
//! (`crate::block::LtxAvBlock` - video stream + audio stream + bidirectional
//! audio<->video cross-attention). The AV counterpart of [`crate::grad`],
//! reusing every primitive that module already defines rather than
//! re-deriving them: [`crate::grad::self_attn_and_text_ca_fwd`]/`_bwd` and
//! [`crate::grad::mlp_fwd`]/`_bwd` run UNCHANGED, once per stream, for the
//! video self-attention/text-cross-attention/FFN and the audio ditto
//! (`crate::block::LtxAvBlock`'s doc, steps 1-2 and step 4) - this module
//! adds only what is genuinely new: the audio<->video cross-attention step
//! (step 3) and the shapes that make the two streams' weights differ (the
//! audio FFN carries bias, the video one does not - `dit::push_ff`'s doc).
//!
//! ## What is genuinely new here, and why
//!
//! 1. **Non-square attention.** `crate::grad::AttnW<T>` assumes
//!    `q_dim == kv_dim == inner_dim` (true for self-/text-cross-attention).
//!    The AV cross-attention projects between streams of DIFFERENT width in
//!    both directions (A2V: query=video width, kv=audio width, working width
//!    = audio's; V2A: query=audio width, kv=video width, working width =
//!    audio's - `crate::block::attention`'s doc). [`CrossAttnW`] carries
//!    independently-shaped `q`/`k`/`v`/`o` for exactly this reason.
//! 2. **A single-row, cross-modality-driven gate.** The self-attn/text-CA/MLP
//!    residuals are all gated PER TOKEN ([`crate::grad::gate_elemwise`]); the
//!    AV cross-attention residual is gated by ONE row shared by every token
//!    of the stream, driven by the OTHER modality's scalar timestep
//!    (`crate::block::gate_row`'s `rows_per_cond = rows` case,
//!    `crate::block::LtxAvBlock`'s doc step 3) - [`crate::grad::gate_bcast`]
//!    is that third point on the per-forward/per-token spectrum.
//! 3. **Two directions sharing one pre-AV snapshot.** Both A2V and V2A read
//!    the SAME post-text-CA state from each stream (`vx1`/`ax1` below,
//!    `crate::block::LtxAvBlock`'s doc: "both directions read a snapshot of
//!    each stream's state taken AFTER step 1/2"), so each stream's adjoint
//!    into that snapshot is a SUM of two contributions - one per direction -
//!    computed independently by [`cross_dir_bwd`] and added by the caller,
//!    the same fan-out discipline `crate::grad`'s own `dadaln_shared` doc
//!    explains for a table read by every block.
//! 4. **Two adaLN tables read twice each, at different row pairs.** Each
//!    stream's own per-token AV scale/shift table (`av_video_ss`/
//!    `av_audio_ss`, model-level, shared by every block) is combined with
//!    THIS block's own `[5,dim]` static table's first four rows ONCE per
//!    stream, then rows (0,1) are read for this stream's role in A2V and
//!    rows (2,3) for its role in V2A (`crate::block::av_scale_shift`'s doc) -
//!    so [`av_scale_shift_fwd`] is called twice per stream on the SAME
//!    operand pair, and its backward ([`av_scale_shift_bwd`]) returns the
//!    FULL `[t,4*dim]` site gradient with only its own two rows populated;
//!    the caller sums the two calls before splitting into `d(table rows
//!    0..4)` (row-sum, this block's own param) and `d(av_*_ss)` (unreduced,
//!    this block's contribution to the model-shared table) - exactly the
//!    `scale_shift_table`/`adaln_shared` fold split [`crate::grad`]'s module
//!    doc explains, at `k=4` instead of `k=9`.
//!
//! `to_gate_logits` (gated attention) is NOT covered, same scope line
//! `crate::modelgrad`'s own tests draw for the video-only path: gated
//! attention's BACKWARD is not implemented here either, so this milestone
//! trains [`crate::config::LtxAvDitConfig::tiny`] (`apply_gated_attention:
//! false`), not `tiny_gated`.
//!
//! One implementation, two instantiations, same discipline as
//! [`crate::grad`]: `f64` is the finite-difference gradcheck oracle
//! (`crates/ltxv/tests/av_block_grad.rs`), `f32` is the host trainer
//! [`crate::av_modelgrad`] drives.

use crate::grad::{
    add_table, attn_bwd, attn_fwd, dgelu, gate_bcast, gate_bcast_bwd, gate_elemwise, gate_elemwise_bwd, gelu, linear, linear_bwd, mlp_bwd, mlp_fwd, mod_affine, mod_affine_bwd, one_plus_plane,
    plane, rmsnorm, rmsnorm_bwd, rope_ltx, rope_ltx_bwd, self_attn_and_text_ca_bwd, self_attn_and_text_ca_fwd, write_plane, AttnGrads, AttnW, Dims, Fp, Lin, LinNB, MlpCache, SattCaCache,
};

/// Shape of the AV block: each stream's own [`Dims`] (`t`/`te`/`dim`/`nh`
/// differ per stream; `eps` is shared - `BasicAVTransformerBlock`'s one
/// `norm_eps` constructor argument covers both streams, `crate::block::
/// LtxAvBlock::forward`'s doc). The cross-attention's own working geometry
/// is always the audio stream's (`a.nh` heads of `a.hd()` width -
/// `crate::block::LtxAvBlock`'s doc), so no separate field is needed.
#[derive(Clone, Copy, Debug)]
pub struct AvDims {
    pub v: Dims,
    pub a: Dims,
}

/// One AV cross-attention module's weights - [`AttnW`]'s non-square twin:
/// `q: [inner,q_dim]`, `k`/`v: [inner,kv_dim]`, `o: [q_dim,inner]`,
/// `qn`/`kn: [inner]`. `inner` is always the audio stream's working width
/// for both `a2v`/`v2a` (`crate::block::LtxAvBlock`'s doc).
#[derive(Clone, Debug, PartialEq)]
pub struct CrossAttnW<T> {
    pub q: Lin<T>,
    pub k: Lin<T>,
    pub v: Lin<T>,
    pub o: Lin<T>,
    pub qn: Vec<T>,
    pub kn: Vec<T>,
}

impl<T: Fp> CrossAttnW<T> {
    pub fn zeros(q_dim: usize, kv_dim: usize, inner: usize) -> CrossAttnW<T> {
        CrossAttnW { q: Lin::zeros(inner, q_dim), k: Lin::zeros(inner, kv_dim), v: Lin::zeros(inner, kv_dim), o: Lin::zeros(q_dim, inner), qn: vec![T::ZERO; inner], kn: vec![T::ZERO; inner] }
    }
}

/// Gradients mirroring [`CrossAttnW`].
#[derive(Clone, Debug)]
pub struct CrossAttnGrads<T> {
    pub q: Lin<T>,
    pub k: Lin<T>,
    pub v: Lin<T>,
    pub o: Lin<T>,
    pub qn: Vec<T>,
    pub kn: Vec<T>,
}

impl<T: Fp> CrossAttnGrads<T> {
    fn zeros(q_dim: usize, kv_dim: usize, inner: usize) -> CrossAttnGrads<T> {
        CrossAttnGrads { q: Lin::zeros(inner, q_dim), k: Lin::zeros(inner, kv_dim), v: Lin::zeros(inner, kv_dim), o: Lin::zeros(q_dim, inner), qn: vec![T::ZERO; inner], kn: vec![T::ZERO; inner] }
    }
}

/// One block's audio<->video cross-attention state - `crate::block::
/// AvCrossWeights`'s generic-`T` twin: both directions' attention modules
/// plus this block's own `[5,dim]` per-stream static tables (rows 0-1 A2V
/// scale/shift, 2-3 V2A scale/shift, 4 this table's own gate -
/// `crate::config`'s doc has the row layout).
#[derive(Clone, Debug, PartialEq)]
pub struct AvCrossW<T> {
    pub a2v: CrossAttnW<T>,
    pub v2a: CrossAttnW<T>,
    /// `[5*vdim]`.
    pub table_video: Vec<T>,
    /// `[5*adim]`.
    pub table_audio: Vec<T>,
}

/// Gradients mirroring [`AvCrossW`].
#[derive(Clone, Debug)]
pub struct AvCrossGrads<T> {
    pub a2v: CrossAttnGrads<T>,
    pub v2a: CrossAttnGrads<T>,
    pub table_video: Vec<T>,
    pub table_audio: Vec<T>,
}

/// One AV block's trainable tensors, named as `crate::dit::
/// av_dit_tensor_manifest` names them (minus the `transformer_blocks.{l}.`
/// prefix). The audio FFN is BIASED (`Lin`, not video's bias-free `LinNB`) -
/// a real asymmetry the real header carries regardless of the shared
/// `ff_bias` config key (`dit::push_ff`'s doc).
#[derive(Clone, Debug, PartialEq)]
pub struct AvBlockW<T> {
    pub v_scale_shift_table: Vec<T>,
    pub v_prompt_scale_shift_table: Vec<T>,
    pub v_attn1: AttnW<T>,
    pub v_attn2: AttnW<T>,
    pub v_ff1: LinNB<T>,
    pub v_ff2: LinNB<T>,
    pub a_scale_shift_table: Vec<T>,
    pub a_prompt_scale_shift_table: Vec<T>,
    pub a_attn1: AttnW<T>,
    pub a_attn2: AttnW<T>,
    pub a_ff1: Lin<T>,
    pub a_ff2: Lin<T>,
    pub av: AvCrossW<T>,
}

/// Gradients mirroring [`AvBlockW`], plus every upstream adjoint: `dvx`/`dax`
/// (into the previous block, one per stream), `dv_ctx`/`da_ctx`, the two
/// UNREDUCED per-token adjoints into the model-shared `adaln_shared` tables
/// (`dv_adaln_shared`/`da_adaln_shared`, `[t*9*dim]` each - same duality
/// `crate::grad`'s own doc explains), the two UNREDUCED per-token adjoints
/// into the model-shared AV scale/shift tables (`dav_video_ss`/
/// `dav_audio_ss`, `[t*4*dim]` each - see this module's doc, point 4), and
/// the two UNREDUCED single-row adjoints into the model-shared AV gate
/// tables (`dav_a2v_gate`/`dav_v2a_gate`, `[dim]` each).
#[derive(Clone, Debug)]
pub struct AvBlockGrads<T> {
    pub v_scale_shift_table: Vec<T>,
    pub v_prompt_scale_shift_table: Vec<T>,
    pub v_attn1: AttnGrads<T>,
    pub v_attn2: AttnGrads<T>,
    pub v_ff1: LinNB<T>,
    pub v_ff2: LinNB<T>,
    pub a_scale_shift_table: Vec<T>,
    pub a_prompt_scale_shift_table: Vec<T>,
    pub a_attn1: AttnGrads<T>,
    pub a_attn2: AttnGrads<T>,
    pub a_ff1: Lin<T>,
    pub a_ff2: Lin<T>,
    pub av: AvCrossGrads<T>,
    pub dvx: Vec<T>,
    pub dax: Vec<T>,
    pub dv_ctx: Vec<T>,
    pub da_ctx: Vec<T>,
    pub dv_adaln_shared: Vec<T>,
    pub da_adaln_shared: Vec<T>,
    pub dav_video_ss: Vec<T>,
    pub dav_audio_ss: Vec<T>,
    pub dav_a2v_gate: Vec<T>,
    pub dav_v2a_gate: Vec<T>,
}

// ---- AV scale/shift + gate combine (crate::block::av_scale_shift/av_gate's generic-T twins) ----

/// [`crate::block::av_scale_shift`]'s generic-`T` twin - see this module's
/// doc, point 4. `mlp_out`: `[t,4*dim]` model-level per-token AV scale/shift
/// table. `table5`: this block's own `[5*dim]` static table (only rows 0-3
/// are read here; row 4 is [`av_gate_fwd`]'s own operand). `row0`: 0 for the
/// A2V role, 2 for the V2A role. Returns `(1+scale, shift)`.
fn av_scale_shift_fwd<T: Fp>(mlp_out: &[T], table5: &[T], t: usize, dim: usize, row0: usize) -> (Vec<T>, Vec<T>) {
    let combined = add_table(mlp_out, &table5[0..4 * dim], t, 4 * dim);
    let scale: Vec<T> = plane(&combined, t, dim, 4, row0).into_iter().map(|v| T::ONE + v).collect();
    let shift = plane(&combined, t, dim, 4, row0 + 1);
    (scale, shift)
}

/// [`av_scale_shift_fwd`] backward for ONE call (one direction/row0): the
/// FULL `[t,4*dim]` site gradient with only rows `(row0,row0+1)` populated.
/// `av_scale_shift_fwd` is called TWICE per stream on the SAME
/// `(mlp_out, table5)` operand pair (A2V's row0=0, V2A's row0=2) - the
/// caller SUMS the two calls' outputs before splitting into `d(table5 rows
/// 0..4)` (row-sum) and `d(mlp_out)` (unreduced), see this module's doc.
fn av_scale_shift_bwd<T: Fp>(t: usize, dim: usize, row0: usize, dscale: &[T], dshift: &[T]) -> Vec<T> {
    let mut dcombined = vec![T::ZERO; t * 4 * dim];
    write_plane(&mut dcombined, t, dim, 4, row0, dscale); // d(1+scale)/d(scale) == 1
    write_plane(&mut dcombined, t, dim, 4, row0 + 1, dshift);
    dcombined
}

/// [`crate::block::av_gate`]'s generic-`T` twin: this block's own row-4 gate
/// combined with the model-level `[dim]` single-row raw gate-MLP output
/// (`rows=1` - `add_table` at `rows=1` is its own exact backward: `dtable =
/// dy`, `d(gate_mlp_out) = dy`, no helper needed).
fn av_gate_fwd<T: Fp>(gate_mlp_out: &[T], table5: &[T], dim: usize) -> Vec<T> {
    add_table(gate_mlp_out, &table5[4 * dim..5 * dim], 1, dim)
}

// ---- audio<->video cross-attention, one direction ----

/// `crate::grad::rmsnorm` at an implicit all-ones gain, folded with a
/// PER-TOKEN `(1+scale,shift)` pair - `crate::block::ada_zero`'s generic-`T`
/// twin, reused here (self-attn/text-CA's own `rmsnorm`+`mod_affine` pair
/// composed the same way, just not factored out under this name there).
fn ada_zero_fwd<T: Fp>(x: &[T], one_plus_scale: &[T], shift: &[T], rows: usize, dim: usize, eps: f64) -> (Vec<T>, Vec<T>, Vec<T>) {
    let ones = vec![T::ONE; dim];
    let (xhat, inv) = rmsnorm(x, rows, dim, &ones, eps);
    (mod_affine(&xhat, one_plus_scale, shift, rows * dim), xhat, inv)
}

/// [`ada_zero_fwd`] backward: `dy` -> `(dx, dscale, dshift)` (`dscale` is
/// already `d(1+scale)/d(scale) == 1`, matching `mod_affine_bwd`'s own
/// convention).
fn ada_zero_bwd<T: Fp>(x: &[T], one_plus_scale: &[T], xhat: &[T], inv: &[T], rows: usize, dim: usize, dy: &[T]) -> (Vec<T>, Vec<T>, Vec<T>) {
    let ones = vec![T::ONE; dim];
    let (dxhat, dscale, dshift) = mod_affine_bwd(xhat, one_plus_scale, dy);
    let mut dw_scratch = vec![T::ZERO; dim];
    let dx = rmsnorm_bwd(x, rows, dim, &ones, inv, &dxhat, &mut dw_scratch);
    (dx, dscale, dshift)
}

/// Everything [`cross_dir_bwd`] needs from [`cross_dir_fwd`] - one AV
/// cross-attention direction (A2V or V2A).
struct CrossDirCache<T> {
    q_scale: Vec<T>,
    q_scaled: Vec<T>,
    q_xhat: Vec<T>,
    q_inv: Vec<T>,
    kv_scale: Vec<T>,
    kv_scaled: Vec<T>,
    kv_xhat: Vec<T>,
    kv_inv: Vec<T>,
    q_pre: Vec<T>,
    k_pre: Vec<T>,
    v_raw: Vec<T>,
    inv_q: Vec<T>,
    inv_k: Vec<T>,
    qr: Vec<T>,
    kr: Vec<T>,
    probs: Vec<T>,
    actx: Vec<T>,
    out: Vec<T>,
    /// This direction's own cross-modal RoPE tables, cached the same way
    /// [`SattCaCache`] caches its self-attention `cos`/`sin` - so
    /// [`cross_dir_bwd`] does not need the (external, non-trainable) tables
    /// supplied a second time.
    q_cos: Vec<T>,
    q_sin: Vec<T>,
    kv_cos: Vec<T>,
    kv_sin: Vec<T>,
}

/// Grads mirroring [`CrossDirCache`]'s owner: the weight grads, this
/// direction's `dscale`/`dshift` for BOTH operands (query-side and kv-side -
/// each destined for a different stream's AV scale/shift site), `dgate`, and
/// the adjoints into each operand's PRE-AV snapshot (`dq_op`/`dkv_op` -
/// summed with the OTHER direction's own contribution by the caller, see
/// this module's doc point 3).
struct CrossDirGrads<T> {
    w: CrossAttnGrads<T>,
    dq_op: Vec<T>,
    dq_scale: Vec<T>,
    dq_shift: Vec<T>,
    dkv_op: Vec<T>,
    dkv_scale: Vec<T>,
    dkv_shift: Vec<T>,
    dgate: Vec<T>,
    dbase: Vec<T>,
}

/// One AV cross-attention direction: scale both operands by their own
/// per-token AV `(1+scale,shift)` pair, project Q/K/V, QK-RMSNorm, per-head
/// RoPE in the shared cross-modal space (Q and K rotate in DIFFERENT
/// position spaces - `crate::block::LtxAvBlock`'s doc), attend, project back
/// to `out_dim` (= the query operand's own dim, so the residual add lines
/// up), gate with a SINGLE broadcast row, add onto `base` (this stream's
/// pre-AV snapshot). `out_dim` is `q_op_dim` (A2V: vdim, V2A: adim).
#[allow(clippy::too_many_arguments)]
fn cross_dir_fwd<T: Fp>(
    q_op: &[T],
    q_rows: usize,
    q_op_dim: usize,
    q_scale: &[T],
    q_shift: &[T],
    kv_op: &[T],
    kv_rows: usize,
    kv_op_dim: usize,
    kv_scale: &[T],
    kv_shift: &[T],
    w: &CrossAttnW<T>,
    inner: usize,
    aheads: usize,
    ahd: usize,
    q_cos: &[T],
    q_sin: &[T],
    kv_cos: &[T],
    kv_sin: &[T],
    gate: &[T],
    base: &[T],
    eps: f64,
) -> (Vec<T>, CrossDirCache<T>) {
    let (q_scaled, q_xhat, q_inv) = ada_zero_fwd(q_op, q_scale, q_shift, q_rows, q_op_dim, eps);
    let (kv_scaled, kv_xhat, kv_inv) = ada_zero_fwd(kv_op, kv_scale, kv_shift, kv_rows, kv_op_dim, eps);

    let q_pre = linear(&q_scaled, q_rows, q_op_dim, &w.q.w, &w.q.b, inner);
    let k_pre = linear(&kv_scaled, kv_rows, kv_op_dim, &w.k.w, &w.k.b, inner);
    let v_raw = linear(&kv_scaled, kv_rows, kv_op_dim, &w.v.w, &w.v.b, inner);
    let (qn, inv_q) = rmsnorm(&q_pre, q_rows, inner, &w.qn, eps);
    let (kn, inv_k) = rmsnorm(&k_pre, kv_rows, inner, &w.kn, eps);
    let qr = rope_ltx(&qn, q_rows, aheads, ahd, q_cos, q_sin);
    let kr = rope_ltx(&kn, kv_rows, aheads, ahd, kv_cos, kv_sin);
    let (probs, actx) = attn_fwd(&qr, q_rows, &kr, &v_raw, kv_rows, aheads, ahd);
    let out = linear(&actx, q_rows, inner, &w.o.w, &w.o.b, q_op_dim);
    let updated = gate_bcast(base, gate, &out, q_rows, q_op_dim);

    (
        updated,
        CrossDirCache {
            q_scale: q_scale.to_vec(), q_scaled, q_xhat, q_inv, kv_scale: kv_scale.to_vec(), kv_scaled, kv_xhat, kv_inv, q_pre, k_pre, v_raw, inv_q, inv_k, qr, kr, probs, actx, out,
            q_cos: q_cos.to_vec(), q_sin: q_sin.to_vec(), kv_cos: kv_cos.to_vec(), kv_sin: kv_sin.to_vec(),
        },
    )
}

/// [`cross_dir_fwd`] backward. `q_op`/`kv_op` are the pre-AV snapshots again
/// (needed by [`ada_zero_bwd`]'s `rmsnorm_bwd`, which reads the raw
/// pre-norm input, not just its cache) - the RoPE tables and the per-token
/// `(1+scale,shift)` pairs are NOT re-passed, [`cross_dir_fwd`] already
/// cached them (see [`CrossDirCache`]'s doc).
#[allow(clippy::too_many_arguments)]
fn cross_dir_bwd<T: Fp>(
    q_op: &[T],
    q_rows: usize,
    q_op_dim: usize,
    kv_op: &[T],
    kv_rows: usize,
    kv_op_dim: usize,
    w: &CrossAttnW<T>,
    inner: usize,
    aheads: usize,
    ahd: usize,
    gate: &[T],
    c: &CrossDirCache<T>,
    dupdated: &[T],
) -> CrossDirGrads<T> {
    let mut wg = CrossAttnGrads::<T>::zeros(q_op_dim, kv_op_dim, inner);

    let (dout, dgate) = gate_bcast_bwd(gate, &c.out, q_rows, q_op_dim, dupdated);
    let dbase = dupdated.to_vec();

    let (dactx, go) = linear_bwd(&c.actx, q_rows, inner, &w.o.w, q_op_dim, &dout);
    wg.o = go;
    let (dqr, dkr, dv_raw) = attn_bwd(&c.probs, &c.qr, &c.kr, &c.v_raw, q_rows, kv_rows, aheads, ahd, &dactx);
    let dqn = rope_ltx_bwd(&dqr, q_rows, aheads, ahd, &c.q_cos, &c.q_sin);
    let dkn = rope_ltx_bwd(&dkr, kv_rows, aheads, ahd, &c.kv_cos, &c.kv_sin);
    let dq_pre = rmsnorm_bwd(&c.q_pre, q_rows, inner, &w.qn, &c.inv_q, &dqn, &mut wg.qn);
    let dk_pre = rmsnorm_bwd(&c.k_pre, kv_rows, inner, &w.kn, &c.inv_k, &dkn, &mut wg.kn);

    let (dq_scaled, gq) = linear_bwd(&c.q_scaled, q_rows, q_op_dim, &w.q.w, inner, &dq_pre);
    wg.q = gq;
    let (dkv_scaled_k, gk) = linear_bwd(&c.kv_scaled, kv_rows, kv_op_dim, &w.k.w, inner, &dk_pre);
    let (dkv_scaled_v, gv) = linear_bwd(&c.kv_scaled, kv_rows, kv_op_dim, &w.v.w, inner, &dv_raw);
    wg.k = gk;
    wg.v = gv;
    let mut dkv_scaled = vec![T::ZERO; kv_rows * kv_op_dim];
    for i in 0..dkv_scaled.len() {
        dkv_scaled[i] = dkv_scaled_k[i] + dkv_scaled_v[i];
    }

    let (dq_op, dq_scale, dq_shift) = ada_zero_bwd(q_op, &c.q_scale, &c.q_xhat, &c.q_inv, q_rows, q_op_dim, &dq_scaled);
    let (dkv_op, dkv_scale, dkv_shift) = ada_zero_bwd(kv_op, &c.kv_scale, &c.kv_xhat, &c.kv_inv, kv_rows, kv_op_dim, &dkv_scaled);

    CrossDirGrads { w: wg, dq_op, dq_scale, dq_shift, dkv_op, dkv_scale, dkv_shift, dgate, dbase }
}

// ---- the audio stream's BIASED MLP sublayer ----
//
// `crate::grad::mlp_fwd`/`mlp_bwd` assume a bias-free FFN (the video-only
// stream's own `ff.net.{0.proj,2}` convention, `ff_bias=false`). The AV
// audio stream's `audio_ff` carries bias regardless of that flag
// (`dit::push_ff`'s doc), so it needs its own, otherwise IDENTICAL, biased
// twin - same op sequence, `Lin`/`linear`/`linear_bwd` instead of
// `LinNB`/`linear_nb`/`linear_nb_bwd`.

/// Everything [`mlp_b_bwd`] needs from [`mlp_b_fwd`].
struct MlpBCache<T> {
    x2: Vec<T>,
    scale_mlp: Vec<T>,
    xhat2: Vec<T>,
    inv2: Vec<T>,
    n2: Vec<T>,
    h1: Vec<T>,
    hg: Vec<T>,
    ff_out: Vec<T>,
    gate_mlp: Vec<T>,
}

/// Grads mirroring [`MlpBCache`]'s owner - [`crate::grad::MlpGrads`]'s
/// biased twin.
struct MlpBGrads<T> {
    ff1: Lin<T>,
    ff2: Lin<T>,
    dshift_mlp: Vec<T>,
    dscale_mlp: Vec<T>,
    dgate_mlp: Vec<T>,
    dx2: Vec<T>,
}

/// [`crate::grad::mlp_fwd`]'s biased twin.
fn mlp_b_fwd<T: Fp>(d: Dims, ff1: &Lin<T>, ff2: &Lin<T>, x2: &[T], combined: &[T]) -> (Vec<T>, MlpBCache<T>) {
    let (t, dim) = (d.t, d.dim);
    let td = t * dim;
    let shift_mlp = plane(combined, t, dim, 9, 3);
    let scale_mlp = one_plus_plane(combined, t, dim, 9, 4);
    let gate_mlp = plane(combined, t, dim, 9, 5);
    let ones = vec![T::ONE; dim];

    let (xhat2, inv2) = rmsnorm(x2, t, dim, &ones, d.eps);
    let n2 = mod_affine(&xhat2, &scale_mlp, &shift_mlp, td);
    let h1 = linear(&n2, t, dim, &ff1.w, &ff1.b, 4 * dim);
    let hg: Vec<T> = h1.iter().map(|&v| gelu(v)).collect();
    let ff_out = linear(&hg, t, 4 * dim, &ff2.w, &ff2.b, dim);
    let out = gate_elemwise(x2, &gate_mlp, &ff_out, td);

    (out, MlpBCache { x2: x2.to_vec(), scale_mlp, xhat2, inv2, n2, h1, hg, ff_out, gate_mlp })
}

/// [`mlp_b_fwd`] backward.
fn mlp_b_bwd<T: Fp>(d: Dims, ff1: &Lin<T>, ff2: &Lin<T>, c: &MlpBCache<T>, dout: &[T]) -> MlpBGrads<T> {
    let (t, dim) = (d.t, d.dim);
    let td = t * dim;
    let ones = vec![T::ONE; dim];

    let (dff_out, dgate_mlp) = gate_elemwise_bwd(&c.gate_mlp, &c.ff_out, dout);
    let mut dx2 = dout.to_vec();

    let (dhg, ff2g) = linear_bwd(&c.hg, t, 4 * dim, &ff2.w, dim, &dff_out);
    let dh1: Vec<T> = dhg.iter().zip(&c.h1).map(|(&gr, &v)| gr * dgelu(v)).collect();
    let (dn2, ff1g) = linear_bwd(&c.n2, t, dim, &ff1.w, 4 * dim, &dh1);
    let (dxhat2, dscale_mlp, dshift_mlp) = mod_affine_bwd(&c.xhat2, &c.scale_mlp, &dn2);
    let mut dw_scratch = vec![T::ZERO; dim];
    let dxhat2_full = rmsnorm_bwd(&c.x2, t, dim, &ones, &c.inv2, &dxhat2, &mut dw_scratch);
    for i in 0..td {
        dx2[i] += dxhat2_full[i];
    }

    MlpBGrads { ff1: ff1g, ff2: ff2g, dshift_mlp, dscale_mlp, dgate_mlp, dx2 }
}

// ---- the AV block ----

/// Everything [`av_block_backward`] needs from [`av_block_forward`].
pub struct AvBlockCache<T> {
    v_satt: SattCaCache<T>,
    a_satt: SattCaCache<T>,
    vx1: Vec<T>,
    ax1: Vec<T>,
    a2v: CrossDirCache<T>,
    v2a: CrossDirCache<T>,
    gate_a2v: Vec<T>,
    gate_v2a: Vec<T>,
    v_mlp: MlpCache<T>,
    a_mlp: MlpBCache<T>,
}

/// One AV block's forward - `crate::block::LtxAvBlock::forward`'s
/// generic-`T` twin, see this module's doc for the exact op order and what
/// is reused vs. new. `vx`/`ax`: `[tv*vdim]`/`[ta*adim]` each stream's
/// current hidden state. `v_adaln_shared`/`a_adaln_shared`: `[t*9*dim]` each
/// stream's model-shared per-token adaLN-single table (BEFORE this block's
/// own `{v,a}_scale_shift_table` is added - the fold happens inside, same
/// convention as `crate::grad::block_forward`). `v_ctx`/`a_ctx`: each
/// stream's raw text context. `{v,a}_cos`/`{v,a}_sin`: each stream's own
/// self-attention RoPE tables. `{v,a}_cross_cos`/`{v,a}_cross_sin`: each
/// stream's own cross-modal RoPE table, at the AUDIO stream's head geometry
/// (`d.a.nh`/`d.a.hd()` - `crate::rope`'s doc). `av_video_ss`/`av_audio_ss`:
/// `[t*4*dim]` model-level per-token AV scale/shift MLP output, one per
/// stream. `av_a2v_gate`/`av_v2a_gate`: `[dim]` model-level SINGLE-row AV
/// gate MLP output, driven by the CROSS modality's scalar sigma.
#[allow(clippy::too_many_arguments)]
pub fn av_block_forward<T: Fp>(
    d: AvDims,
    w: &AvBlockW<T>,
    vx: &[T],
    ax: &[T],
    v_adaln_shared: &[T],
    a_adaln_shared: &[T],
    v_ctx: &[T],
    a_ctx: &[T],
    v_cos: &[T],
    v_sin: &[T],
    a_cos: &[T],
    a_sin: &[T],
    v_cross_cos: &[T],
    v_cross_sin: &[T],
    a_cross_cos: &[T],
    a_cross_sin: &[T],
    av_video_ss: &[T],
    av_audio_ss: &[T],
    av_a2v_gate: &[T],
    av_v2a_gate: &[T],
) -> (Vec<T>, Vec<T>, AvBlockCache<T>) {
    let (vdim, adim) = (d.v.dim, d.a.dim);
    let (tv, ta) = (d.v.t, d.a.t);
    let (aheads, ahd) = (d.a.nh, d.a.hd());
    let eps = d.v.eps;
    assert_eq!(av_video_ss.len(), tv * 4 * vdim, "av_video_ss must be [tv,4*vdim]");
    assert_eq!(av_audio_ss.len(), ta * 4 * adim, "av_audio_ss must be [ta,4*adim]");
    assert_eq!(av_a2v_gate.len(), vdim, "av_a2v_gate must be [vdim]");
    assert_eq!(av_v2a_gate.len(), adim, "av_v2a_gate must be [adim]");
    assert_eq!(w.av.table_video.len(), 5 * vdim, "table_video must be [5*vdim]");
    assert_eq!(w.av.table_audio.len(), 5 * adim, "table_audio must be [5*adim]");

    // ---- 1-2: video, then audio - self-attn + text-CA, each stream run to
    // completion (LtxAvBlock's doc, steps 1-2), reusing crate::grad UNCHANGED.
    let v_combined = add_table(v_adaln_shared, &w.v_scale_shift_table, tv, 9 * vdim);
    let (vx1, v_satt) = self_attn_and_text_ca_fwd(d.v, &w.v_attn1, &w.v_attn2, &w.v_prompt_scale_shift_table, vx, &v_combined, v_ctx, v_cos, v_sin);
    let a_combined = add_table(a_adaln_shared, &w.a_scale_shift_table, ta, 9 * adim);
    let (ax1, a_satt) = self_attn_and_text_ca_fwd(d.a, &w.a_attn1, &w.a_attn2, &w.a_prompt_scale_shift_table, ax, &a_combined, a_ctx, a_cos, a_sin);

    // ---- 3: audio<->video cross-attention - vx1/ax1 are the pre-AV
    // snapshot BOTH directions read (LtxAvBlock's doc, step 3).
    let (scale_a2v_v, shift_a2v_v) = av_scale_shift_fwd(av_video_ss, &w.av.table_video, tv, vdim, 0);
    let (scale_a2v_a, shift_a2v_a) = av_scale_shift_fwd(av_audio_ss, &w.av.table_audio, ta, adim, 0);
    let gate_a2v = av_gate_fwd(av_a2v_gate, &w.av.table_video, vdim);

    let (scale_v2a_a, shift_v2a_a) = av_scale_shift_fwd(av_audio_ss, &w.av.table_audio, ta, adim, 2);
    let (scale_v2a_v, shift_v2a_v) = av_scale_shift_fwd(av_video_ss, &w.av.table_video, tv, vdim, 2);
    let gate_v2a = av_gate_fwd(av_v2a_gate, &w.av.table_audio, adim);

    let (vx2, a2v) = cross_dir_fwd(
        &vx1, tv, vdim, &scale_a2v_v, &shift_a2v_v, &ax1, ta, adim, &scale_a2v_a, &shift_a2v_a, &w.av.a2v, adim, aheads, ahd, v_cross_cos, v_cross_sin, a_cross_cos, a_cross_sin,
        &gate_a2v, &vx1, eps,
    );
    let (ax2, v2a) = cross_dir_fwd(
        &ax1, ta, adim, &scale_v2a_a, &shift_v2a_a, &vx1, tv, vdim, &scale_v2a_v, &shift_v2a_v, &w.av.v2a, adim, aheads, ahd, a_cross_cos, a_cross_sin, v_cross_cos, v_cross_sin,
        &gate_v2a, &ax1, eps,
    );

    // ---- 4: MLPs, video then audio (LtxAvBlock's doc, step 4). Video reuses
    // crate::grad UNCHANGED (bias-free FFN); audio uses this module's own
    // biased twin ([`mlp_b_fwd`], see the section above for why).
    let (vx3, v_mlp) = mlp_fwd(d.v, &w.v_ff1, &w.v_ff2, &vx2, &v_combined);
    let (ax3, a_mlp) = mlp_b_fwd(d.a, &w.a_ff1, &w.a_ff2, &ax2, &a_combined);

    let cache = AvBlockCache { v_satt, a_satt, vx1, ax1, a2v, v2a, gate_a2v, gate_v2a, v_mlp, a_mlp };
    (vx3, ax3, cache)
}

/// One AV block's backward: `dvx_out[tv*vdim]`/`dax_out[ta*adim]` -> every
/// weight grad, `dvx`/`dax`, `dv_ctx`/`da_ctx`, and the four model-shared
/// per-token/single-row adjoints (`dv_adaln_shared`/`da_adaln_shared`,
/// `dav_video_ss`/`dav_audio_ss`, `dav_a2v_gate`/`dav_v2a_gate` - all
/// UNREDUCED, this block's own contribution, see this module's doc).
pub fn av_block_backward<T: Fp>(d: AvDims, w: &AvBlockW<T>, c: &AvBlockCache<T>, dvx_out: &[T], dax_out: &[T]) -> AvBlockGrads<T> {
    let (vdim, adim) = (d.v.dim, d.a.dim);
    let (tv, ta) = (d.v.t, d.a.t);
    let (aheads, ahd) = (d.a.nh, d.a.hd());

    // ---- 4 (reverse): MLPs ----
    let v_mg = mlp_bwd(d.v, &w.v_ff1, &w.v_ff2, &c.v_mlp, dvx_out);
    let a_mg = mlp_b_bwd(d.a, &w.a_ff1, &w.a_ff2, &c.a_mlp, dax_out);
    // v_mg.dx2 / a_mg.dx2 are the COMPLETE gradients into vx2/ax2 - the AV
    // cross-attention residual's own output.

    // ---- 3 (reverse): audio<->video cross-attention, both directions -----
    // A2V's `dupdated` is `v_mg.dx2` (the complete gradient into vx2, which
    // A2V produced); V2A's is `a_mg.dx2`. Everything else A2V/V2A need
    // (RoPE tables, the per-token scale/shift pairs, `vx1`/`ax1`) is either
    // cached in `c.a2v`/`c.v2a` or in `c` directly (this module's doc,
    // point 3 - `vx1`/`ax1` are the pre-AV snapshot BOTH directions read).
    let a2v = cross_dir_bwd(&c.vx1, tv, vdim, &c.ax1, ta, adim, &w.av.a2v, adim, aheads, ahd, &c.gate_a2v, &c.a2v, &v_mg.dx2);
    let v2a = cross_dir_bwd(&c.ax1, ta, adim, &c.vx1, tv, vdim, &w.av.v2a, adim, aheads, ahd, &c.gate_v2a, &c.v2a, &a_mg.dx2);

    // vx1 is read three ways in step 3 (A2V's query operand, A2V's residual
    // base, V2A's kv operand); ax1 is read the mirror three ways - sum every
    // contribution before entering step 1-2's own backward, the same
    // fan-out this module's doc, point 3, names.
    let mut dvx1 = a2v.dbase;
    for ((d, dq), dkv) in dvx1.iter_mut().zip(a2v.dq_op.iter()).zip(v2a.dkv_op.iter()) {
        *d += *dq + *dkv;
    }
    let mut dax1 = v2a.dbase;
    for ((d, dq), dkv) in dax1.iter_mut().zip(v2a.dq_op.iter()).zip(a2v.dkv_op.iter()) {
        *d += *dq + *dkv;
    }

    // Each stream's AV scale/shift table is read TWICE (rows 0-1 for its
    // role in A2V, rows 2-3 for V2A - this module's doc, point 4): sum the
    // two calls' site gradients before splitting into the block's own
    // static-table row-sum and the model-shared table's unreduced
    // contribution.
    let mut dcombined_v_ss = av_scale_shift_bwd(tv, vdim, 0, &a2v.dq_scale, &a2v.dq_shift);
    let dcombined_v_ss_v2a = av_scale_shift_bwd(tv, vdim, 2, &v2a.dkv_scale, &v2a.dkv_shift);
    for i in 0..dcombined_v_ss.len() {
        dcombined_v_ss[i] += dcombined_v_ss_v2a[i];
    }
    let mut dcombined_a_ss = av_scale_shift_bwd(ta, adim, 0, &a2v.dkv_scale, &a2v.dkv_shift);
    let dcombined_a_ss_v2a = av_scale_shift_bwd(ta, adim, 2, &v2a.dq_scale, &v2a.dq_shift);
    for i in 0..dcombined_a_ss.len() {
        dcombined_a_ss[i] += dcombined_a_ss_v2a[i];
    }
    let mut table_video_grad = vec![T::ZERO; 5 * vdim];
    for r in 0..tv {
        for i in 0..4 * vdim {
            table_video_grad[i] += dcombined_v_ss[r * 4 * vdim + i];
        }
    }
    table_video_grad[4 * vdim..5 * vdim].copy_from_slice(&a2v.dgate);
    let mut table_audio_grad = vec![T::ZERO; 5 * adim];
    for r in 0..ta {
        for i in 0..4 * adim {
            table_audio_grad[i] += dcombined_a_ss[r * 4 * adim + i];
        }
    }
    table_audio_grad[4 * adim..5 * adim].copy_from_slice(&v2a.dgate);

    // ---- 1-2 (reverse): self-attn + text-CA, each stream, reusing
    // crate::grad UNCHANGED - dvx1/dax1 above are exactly the `dx2` these
    // calls expect (the complete gradient into the state text-CA produced).
    let v_sg = self_attn_and_text_ca_bwd(d.v, &w.v_attn1, &w.v_attn2, &w.v_prompt_scale_shift_table, &c.v_satt, &dvx1);
    let a_sg = self_attn_and_text_ca_bwd(d.a, &w.a_attn1, &w.a_attn2, &w.a_prompt_scale_shift_table, &c.a_satt, &dax1);

    let mut v_dcombined = vec![T::ZERO; tv * 9 * vdim];
    write_plane(&mut v_dcombined, tv, vdim, 9, 0, &v_sg.dshift_msa);
    write_plane(&mut v_dcombined, tv, vdim, 9, 1, &v_sg.dscale_msa);
    write_plane(&mut v_dcombined, tv, vdim, 9, 2, &v_sg.dgate_msa);
    write_plane(&mut v_dcombined, tv, vdim, 9, 3, &v_mg.dshift_mlp);
    write_plane(&mut v_dcombined, tv, vdim, 9, 4, &v_mg.dscale_mlp);
    write_plane(&mut v_dcombined, tv, vdim, 9, 5, &v_mg.dgate_mlp);
    write_plane(&mut v_dcombined, tv, vdim, 9, 6, &v_sg.dshift_q);
    write_plane(&mut v_dcombined, tv, vdim, 9, 7, &v_sg.dscale_q);
    write_plane(&mut v_dcombined, tv, vdim, 9, 8, &v_sg.dgate_q);
    let mut v_scale_shift_table = vec![T::ZERO; 9 * vdim];
    for r in 0..tv {
        for i in 0..9 * vdim {
            v_scale_shift_table[i] += v_dcombined[r * 9 * vdim + i];
        }
    }

    let mut a_dcombined = vec![T::ZERO; ta * 9 * adim];
    write_plane(&mut a_dcombined, ta, adim, 9, 0, &a_sg.dshift_msa);
    write_plane(&mut a_dcombined, ta, adim, 9, 1, &a_sg.dscale_msa);
    write_plane(&mut a_dcombined, ta, adim, 9, 2, &a_sg.dgate_msa);
    write_plane(&mut a_dcombined, ta, adim, 9, 3, &a_mg.dshift_mlp);
    write_plane(&mut a_dcombined, ta, adim, 9, 4, &a_mg.dscale_mlp);
    write_plane(&mut a_dcombined, ta, adim, 9, 5, &a_mg.dgate_mlp);
    write_plane(&mut a_dcombined, ta, adim, 9, 6, &a_sg.dshift_q);
    write_plane(&mut a_dcombined, ta, adim, 9, 7, &a_sg.dscale_q);
    write_plane(&mut a_dcombined, ta, adim, 9, 8, &a_sg.dgate_q);
    let mut a_scale_shift_table = vec![T::ZERO; 9 * adim];
    for r in 0..ta {
        for i in 0..9 * adim {
            a_scale_shift_table[i] += a_dcombined[r * 9 * adim + i];
        }
    }

    AvBlockGrads {
        v_scale_shift_table,
        v_prompt_scale_shift_table: v_sg.dprompt_scale_shift_table,
        v_attn1: v_sg.attn1,
        v_attn2: v_sg.attn2,
        v_ff1: v_mg.ff1,
        v_ff2: v_mg.ff2,
        a_scale_shift_table,
        a_prompt_scale_shift_table: a_sg.dprompt_scale_shift_table,
        a_attn1: a_sg.attn1,
        a_attn2: a_sg.attn2,
        a_ff1: a_mg.ff1,
        a_ff2: a_mg.ff2,
        av: AvCrossGrads { a2v: a2v.w, v2a: v2a.w, table_video: table_video_grad, table_audio: table_audio_grad },
        dvx: v_sg.dx,
        dax: a_sg.dx,
        dv_ctx: v_sg.dctx,
        da_ctx: a_sg.dctx,
        dv_adaln_shared: v_dcombined,
        da_adaln_shared: a_dcombined,
        dav_video_ss: dcombined_v_ss,
        dav_audio_ss: dcombined_a_ss,
        dav_a2v_gate: a2v.dgate,
        dav_v2a_gate: v2a.dgate,
    }
}
