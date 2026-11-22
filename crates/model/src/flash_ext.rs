// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded attention for long-context
// models for its clients. If your team needs expertise in attention kernels
// with sliding windows, sinks and learned position biases, for training and
// paged serving, then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Causal GQA attention with OPTIONAL, declared inputs - each off by default - computed online (never a
//! `[B,H,T,T]` slab), forward and backward:
//!
//! * a causal sliding window: query at absolute position `P` reads keys `p` with `P - window < p <= P`;
//! * per-(row, head) sink logits that join the softmax denominator with a zero value
//!   (`softmax([logits; sink])[..., :-1]`), with `d_sink`;
//! * a distance-bank additive bias `bias[h,i,j] = sum_r rel[i,h,r] * bank[r, P_i - p_j]` for distances below
//!   `extent` and ZERO beyond (not masked), computed per key in the kernel, with `d_rel` and `d_bank`;
//! * a per-row query scale from the absolute position, `tau(P) = 1 + alpha * ln(max((P + 1) / n_floor, 1))`
//!   ([`log_position_scale`]), applied to the query and, when [`LogPositionScale::scales_bias`], to the bias
//!   (the same factor a caller would otherwise apply to the relative states).
//!
//! * packed documents (training rows that hold several documents, [`AttnOptions::documents`]): a query sees only
//!   the keys of its own document, and the window, the distances and `tau` use its position INSIDE the document, so
//!   each packed document computes exactly what it computes alone, forward and backward.
//!
//! The logit of query row `i` against key `j` is `tq_i * scale * <q_i, k_j> + tb_i * bias_ij`.
//!
//! Distances and `tau` are taken from ABSOLUTE positions, so a chunked prefill against a paged or ring cache
//! ([`flash_ext_paged_fwd`]), a decode step and one whole-sequence pass ([`flash_ext_fwd`]) compute the same
//! logits. A sliding-window layer's KV can be a ring of `window` rows per sequence ([`PagedKv::ring`]): position
//! `p` lives in row `p % ring` (the [`crate::gqa_mixer::ring_row`] contract), and a decode step reads it in place.
//!
//! The kernels are the reference tier (`flash_attn_gqa_ext*.wgsl`, one invocation per row and head, no
//! barrier): they run on every backend including the CPU JIT. The fast attention paths
//! ([`crate::block::flash_gqa_causal_fwd`], the tensor-core and native CUDA kernels, the paged flash decode) take
//! none of these options: a caller with any option set calls this module, never those, so no option is ever
//! dropped by a selection.
//!
//! The optional per-row inputs travel to the kernels in one packed `aux` buffer (sink, rel, bank at word
//! offsets) because a kernel binds at most 8 storage buffers; the packing is internal to the functions here.

use gpu_core::{f, DeviceBuffer, Gpu, Step};

use crate::kv_tier::{KvLayer, KvTier};

/// Flag bits of the kernels' `flags` parameter (see `flash_attn_gqa_ext.wgsl`).
const SINK: u32 = 1;
const BIAS: u32 = 2;
const TAU: u32 = 4;
const TAU_BIAS: u32 = 8;
const PAGED: u32 = 16;
const DOCS: u32 = 32;

/// Widest head the kernels hold (their per-thread accumulators are `array<f32, 256>`).
pub const MAX_HEAD_DIM: u32 = 256;
/// Widest relative state the kernels hold (`array<f32, 64>`).
pub const MAX_D_REL: u32 = 64;

/// A sliding-window layer's or a full layer's distance-bank bias: `bank` is `[d_rel, extent]`, `rel` is
/// `[rows, n_heads, d_rel]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DistanceBias {
    /// Distances `0..extent` carry a bias; the bias is zero beyond.
    pub extent: u32,
    /// Width of a relative state.
    pub d_rel: u32,
}

/// The per-row query factor `tau(P) = 1 + alpha * ln(max((P + 1) / n_floor, 1))` at absolute position `P`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogPositionScale {
    pub alpha: f32,
    pub n_floor: f32,
    /// Scale the distance-bank bias by the same factor (a reference that scales its relative states by `tau`).
    pub scales_bias: bool,
}

/// The host value of [`LogPositionScale`]'s factor at absolute position `pos`, in the kernels' own f32 order.
#[must_use]
pub fn log_position_scale(pos: u32, alpha: f32, n_floor: f32) -> f32 {
    1.0 + alpha * ((pos as f32 + 1.0) / n_floor).max(1.0).ln()
}

/// What the attention computes beyond plain causal GQA. `AttnOptions::plain(scale)` is plain causal attention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnOptions {
    /// Multiplies `<q, k>` (usually `head_dim^-0.5`).
    pub scale: f32,
    /// Keys each query may look back over, itself included (`>= 1`); `None` is plain causal.
    pub window: Option<u32>,
    /// A per-(row, head) sink logit joins the softmax with a zero value ([`AttnInputs::sink`]).
    pub sink: bool,
    /// The distance-bank bias ([`AttnInputs::rel`], [`AttnInputs::bank`]).
    pub bias: Option<DistanceBias>,
    /// The log-position query scale.
    pub tau: Option<LogPositionScale>,
    /// The rows pack several documents ([`AttnInputs::doc_pos`]): attention never crosses a document and positions
    /// restart at each one. Contiguous layout at `pos0 == 0` only (serving has no packed rows).
    pub documents: bool,
}

impl AttnOptions {
    /// Plain causal attention at `scale`: every option off.
    #[must_use]
    pub fn plain(scale: f32) -> AttnOptions {
        AttnOptions { scale, window: None, sink: false, bias: None, tau: None, documents: false }
    }

    fn flags(&self) -> u32 {
        let mut fl = 0;
        if self.sink {
            fl |= SINK;
        }
        if self.bias.is_some() {
            fl |= BIAS;
        }
        if let Some(t) = self.tau {
            fl |= TAU;
            if t.scales_bias {
                fl |= TAU_BIAS;
            }
        }
        if self.documents {
            fl |= DOCS;
        }
        fl
    }

    fn check(&self, heads: &HeadShape) {
        assert!(heads.head_dim >= 1 && heads.head_dim <= MAX_HEAD_DIM, "flash_ext: head_dim {} outside 1..={MAX_HEAD_DIM}", heads.head_dim);
        assert!(heads.n_kv_heads >= 1 && heads.n_heads.is_multiple_of(heads.n_kv_heads), "flash_ext: {} query heads do not group over {} kv heads", heads.n_heads, heads.n_kv_heads);
        assert!(self.scale.is_finite(), "flash_ext: attention scale {} is not finite", self.scale);
        if let Some(w) = self.window {
            assert!(w >= 1, "flash_ext: a window of 0 lets no key through");
        }
        if let Some(b) = self.bias {
            assert!(b.d_rel >= 1 && b.d_rel <= MAX_D_REL, "flash_ext: d_rel {} outside 1..={MAX_D_REL}", b.d_rel);
            assert!(b.extent >= 1, "flash_ext: a distance bank of extent 0 carries no bias; drop the option instead");
        }
        if let Some(t) = self.tau {
            assert!(t.n_floor > 0.0 && t.n_floor.is_finite() && t.alpha.is_finite(), "flash_ext: log-position scale needs a positive n_floor and a finite alpha, got {t:?}");
            assert!(!t.scales_bias || self.bias.is_some(), "flash_ext: tau scales_bias without a distance bias");
        }
    }
}

/// The attention's head layout: q/out rows are `n_heads * head_dim` wide, k/v rows `n_kv_heads * head_dim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeadShape {
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

/// The optional inputs [`AttnOptions`] declares: each must be present exactly when its option is set (a missing
/// buffer or an undeclared one is refused, never ignored).
#[derive(Clone, Copy, Default)]
pub struct AttnInputs<'a> {
    /// `[rows, n_heads]` sink logits.
    pub sink: Option<&'a DeviceBuffer>,
    /// `[rows, n_heads, d_rel]` relative states.
    pub rel: Option<&'a DeviceBuffer>,
    /// `[d_rel, extent]` distance bank.
    pub bank: Option<&'a DeviceBuffer>,
    /// `[rows]` u32: each row's position inside its packed document - 0 at a document's first row, one more on each
    /// following row of the same document ([`data::packing::document_positions`] builds it from per-token document ids). A row's
    /// position is clamped to its index in the sequence, so a sequence may open mid-document.
    pub doc_pos: Option<&'a DeviceBuffer>,
}

/// Gradients of [`AttnInputs`], overwritten by [`flash_ext_bwd`]; each present exactly when its input is
/// (pass `None` for a frozen one only together with a `None` input).
#[derive(Clone, Copy, Default)]
pub struct AttnInputGrads<'a> {
    pub d_sink: Option<&'a DeviceBuffer>,
    pub d_rel: Option<&'a DeviceBuffer>,
    pub d_bank: Option<&'a DeviceBuffer>,
}

/// The contiguous (training) layout: `b` sequences of `t` rows; q/out `[b*t, n_heads*head_dim]`, k/v
/// `[b*t, n_kv_heads*head_dim]`, row `i` of every sequence at absolute position `pos0 + i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Contiguous {
    pub b: u32,
    pub t: u32,
    pub pos0: u32,
}

/// A forward kernel slot and its dispatch geometry: the barrier-free reference (one thread per query row and head)
/// or its cooperative twin (one workgroup per pair, `flash_attn_gqa_ext_rows`, appended by `gpu_core`'s twin table on
/// a device with workgroup reductions), as `select::Op::FlashAttentionExt` chose for the handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FwdKernel {
    pub kernel: usize,
    pub per_pair_workgroup: bool,
}

impl FwdKernel {
    fn dispatch(self, pairs: u32) -> gpu_core::Dispatch {
        if self.per_pair_workgroup {
            gpu_core::Dispatch::Workgroups(pairs)
        } else {
            gpu_core::Dispatch::Threads(pairs)
        }
    }
}

/// Pipeline indices of the kernels this module dispatches, resolved by name ([`FlashExtIds::resolve`]) from a
/// kernel list that includes [`pipelines`].
#[derive(Clone, Copy, Debug)]
pub struct FlashExtIds {
    pub fwd: FwdKernel,
    /// The forward over a bf16 KV pool (`pool_k`/`pool_v` read as packed bf16 pairs).
    pub fwd_bf16: FwdKernel,
    pub bwd_prep: usize,
    pub bwd_dq: usize,
    pub bwd_dkv: usize,
    pub bwd_dbank: usize,
    /// `splice_at`: the word copies that pack [`AttnInputs`] into one buffer.
    pub splice_at: usize,
}

/// The bf16-pool forward: the reference forward with its two pool bindings read as packed bf16 pairs.
fn fwd_bf16_variant() -> (&'static str, &'static str) {
    use gpu_core::select::Dtype;
    let (name, src) = kernels::template::dtype_variant("flash_attn_gqa_ext", kernels::FLASH_ATTN_GQA_EXT, "pool_k", Dtype::BF16).expect("flash_attn_gqa_ext reads pool_k by bare index");
    kernels::template::dtype_variant(name, src, "pool_v", Dtype::BF16).expect("flash_attn_gqa_ext reads pool_v by bare index")
}

/// The kernels a caller registers for this module (next to its own list).
#[must_use]
pub fn pipelines() -> Vec<(&'static str, &'static str)> {
    vec![
        ("flash_attn_gqa_ext", kernels::FLASH_ATTN_GQA_EXT),
        fwd_bf16_variant(),
        ("flash_attn_gqa_ext_bwd_prep", kernels::FLASH_ATTN_GQA_EXT_BWD_PREP),
        ("flash_attn_gqa_ext_bwd_dq", kernels::FLASH_ATTN_GQA_EXT_BWD_DQ),
        ("flash_attn_gqa_ext_bwd_dkv", kernels::FLASH_ATTN_GQA_EXT_BWD_DKV),
        ("flash_attn_gqa_ext_bwd_dbank", kernels::FLASH_ATTN_GQA_EXT_BWD_DBANK),
        ("splice_at", kernels::SPLICE_AT),
    ]
}

impl FlashExtIds {
    /// The indices of [`pipelines`] on `g`, each forward its cooperative twin where the handle carries one and
    /// `select::Op::FlashAttentionExt` picks it on this device (no shape gate, so one choice per handle). Panics when
    /// a kernel is not registered.
    pub fn resolve(g: &Gpu) -> FlashExtIds {
        use gpu_core::select::{Dtype, KernelSelector, KernelVariant, Op, OpShape};
        let coop = gpu_core::select::DefaultSelector.select(Op::FlashAttentionExt, OpShape { m: 1, n: 1, k: 0, dtype: Dtype::F32 }, &g.caps()) == KernelVariant::WorkgroupPerOutput;
        let fwd = |reference: usize| match g.twin_of(reference).filter(|_| coop) {
            Some(twin) => FwdKernel { kernel: twin, per_pair_workgroup: true },
            None => FwdKernel { kernel: reference, per_pair_workgroup: false },
        };
        let mut ids = Self::reference(g);
        ids.fwd = fwd(ids.fwd.kernel);
        ids.fwd_bf16 = fwd(ids.fwd_bf16.kernel);
        ids
    }

    /// [`Self::resolve`] held to the barrier-free reference forwards whatever the device: the arm the twins are gated
    /// against.
    pub fn reference(g: &Gpu) -> FlashExtIds {
        let at = |name: &str| g.kernel_index(name).unwrap_or_else(|| panic!("{name} is not registered on this Gpu (see model::flash_ext::pipelines)"));
        let reference = |name: &str| FwdKernel { kernel: at(name), per_pair_workgroup: false };
        FlashExtIds {
            fwd: reference("flash_attn_gqa_ext"),
            fwd_bf16: reference(fwd_bf16_variant().0),
            bwd_prep: at("flash_attn_gqa_ext_bwd_prep"),
            bwd_dq: at("flash_attn_gqa_ext_bwd_dq"),
            bwd_dkv: at("flash_attn_gqa_ext_bwd_dkv"),
            bwd_dbank: at("flash_attn_gqa_ext_bwd_dbank"),
            splice_at: at("splice_at"),
        }
    }
}

/// A count as the kernels' `u32` element index.
fn word(x: u64, what: &str) -> u32 {
    u32::try_from(x).unwrap_or_else(|_| panic!("flash_ext: {what} = {x} exceeds the kernels' u32 element index"))
}

/// The packed optional inputs and their word offsets.
struct Aux {
    buf: DeviceBuffer,
    sink_off: u32,
    rel_off: u32,
    bank_off: u32,
}

/// Refuse an input that does not match its declared option, then pack the present ones into one buffer.
fn pack_aux(g: &Gpu, ids: &FlashExtIds, opts: &AttnOptions, heads: &HeadShape, inputs: &AttnInputs, rows: u64, steps: &mut Vec<Step>) -> Aux {
    assert_eq!(opts.sink, inputs.sink.is_some(), "flash_ext: the sink option and the sink buffer must come together (option {}, buffer {})", opts.sink, inputs.sink.is_some());
    assert_eq!(opts.bias.is_some(), inputs.rel.is_some(), "flash_ext: the bias option and the rel buffer must come together");
    assert_eq!(opts.bias.is_some(), inputs.bank.is_some(), "flash_ext: the bias option and the bank buffer must come together");
    assert_eq!(opts.documents, inputs.doc_pos.is_some(), "flash_ext: the documents option and the doc_pos buffer must come together (option {}, buffer {})", opts.documents, inputs.doc_pos.is_some());
    let nh = u64::from(heads.n_heads);
    let sink_len = if opts.sink { rows * nh } else { 0 };
    let (rel_len, bank_len) = opts.bias.map_or((0, 0), |b| (rows * nh * u64::from(b.d_rel), u64::from(b.d_rel) * u64::from(b.extent)));
    let (sink_off, rel_off, bank_off) = (0, sink_len, sink_len + rel_len);
    let total = sink_len + rel_len + bank_len;
    let buf = g.storage_uninit(total.max(1));
    for (src, len, off) in [(inputs.sink, sink_len, sink_off), (inputs.rel, rel_len, rel_off), (inputs.bank, bank_len, bank_off)] {
        if let (Some(src), true) = (src, len > 0) {
            let n = word(len, "packed input words");
            steps.push(g.step(ids.splice_at, &[src, &buf], &[n, 0, word(off, "packed input offset")], n));
        }
    }
    Aux { buf, sink_off: word(sink_off, "sink offset"), rel_off: word(rel_off, "rel offset"), bank_off: word(bank_off, "bank offset") }
}

/// The kernels' shared 20-word uniform.
#[allow(clippy::too_many_arguments)]
fn params(opts: &AttnOptions, heads: &HeadShape, n_seqs: u32, max_q: u32, block_size: u32, max_bt: u32, ring: u32, extra_flags: u32, aux: &Aux, pos0: u32) -> [u32; 20] {
    let (extent, d_rel) = opts.bias.map_or((0, 0), |b| (b.extent, b.d_rel));
    let (alpha, floor) = opts.tau.map_or((0.0, 1.0), |t| (t.alpha, t.n_floor));
    [
        n_seqs,
        max_q,
        heads.n_heads,
        heads.n_kv_heads,
        heads.head_dim,
        block_size,
        max_bt,
        ring,
        opts.window.unwrap_or(0),
        opts.flags() | extra_flags,
        extent,
        d_rel,
        aux.sink_off,
        aux.rel_off,
        aux.bank_off,
        pos0,
        f(opts.scale),
        f(alpha),
        f(floor),
        0,
    ]
}

/// Refuse a contiguous layout the documents option cannot run on: packed positions start at 0, and a row-local
/// document start must be exact as the f32 the backward carries it in.
fn check_contiguous(opts: &AttnOptions, layout: &Contiguous) {
    if opts.documents {
        assert_eq!(layout.pos0, 0, "flash_ext: packed documents carry their own positions; pos0 must be 0, got {}", layout.pos0);
        assert!(layout.t <= 1 << 24, "flash_ext: packed rows of {} tokens exceed the 2^24 the backward's f32 document starts hold exactly", layout.t);
    }
}

/// Forward over the contiguous (training) layout: `out` `[b*t, n_heads*head_dim]` and `lse` `[b*t, n_heads]`
/// (each row's log-sum-exp of its logits and its sink: all [`flash_ext_bwd`] needs) are overwritten.
#[allow(clippy::too_many_arguments)]
pub fn flash_ext_fwd(g: &Gpu, ids: &FlashExtIds, heads: &HeadShape, opts: &AttnOptions, inputs: &AttnInputs, layout: Contiguous, q: &DeviceBuffer, k: &DeviceBuffer, v: &DeviceBuffer, out: &DeviceBuffer, lse: &DeviceBuffer) -> Vec<Step> {
    opts.check(heads);
    check_contiguous(opts, &layout);
    let rows = u64::from(layout.b) * u64::from(layout.t);
    word(rows * u64::from(heads.n_heads) * u64::from(heads.head_dim), "q elements");
    let mut steps = Vec::new();
    let aux = pack_aux(g, ids, opts, heads, inputs, rows, &mut steps);
    // The contiguous layout reads no paged index; with documents the binding carries the rows' document positions.
    let spare = g.storage(1);
    let index = inputs.doc_pos.unwrap_or(&spare);
    let p = params(opts, heads, layout.b, layout.t, 0, 0, 0, 0, &aux, layout.pos0);
    let threads = word(rows * u64::from(heads.n_heads), "query rows * heads");
    steps.push(g.dispatch(ids.fwd.kernel, &[q, k, v, index, &aux.buf, out, lse], &p, ids.fwd.dispatch(threads)));
    steps
}

/// Backward of [`flash_ext_fwd`] from `d_out`, its `q`/`k`/`v`/`out`/`lse`: overwrites `d_q`, `d_k`, `d_v` and
/// each gradient of `grads` (which must name a buffer for every input the options declare).
#[allow(clippy::too_many_arguments)]
pub fn flash_ext_bwd(
    g: &Gpu,
    ids: &FlashExtIds,
    heads: &HeadShape,
    opts: &AttnOptions,
    inputs: &AttnInputs,
    layout: Contiguous,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    out: &DeviceBuffer,
    lse: &DeviceBuffer,
    d_out: &DeviceBuffer,
    d_q: &DeviceBuffer,
    d_k: &DeviceBuffer,
    d_v: &DeviceBuffer,
    grads: &AttnInputGrads,
) -> Vec<Step> {
    opts.check(heads);
    check_contiguous(opts, &layout);
    assert_eq!(opts.sink, grads.d_sink.is_some(), "flash_ext_bwd: d_sink must be given exactly when the sink option is set");
    assert_eq!(opts.bias.is_some(), grads.d_rel.is_some(), "flash_ext_bwd: d_rel must be given exactly when the bias option is set");
    assert_eq!(opts.bias.is_some(), grads.d_bank.is_some(), "flash_ext_bwd: d_bank must be given exactly when the bias option is set");
    let rows = u64::from(layout.b) * u64::from(layout.t);
    let nh = u64::from(heads.n_heads);
    let mut steps = Vec::new();
    let aux = pack_aux(g, ids, opts, heads, inputs, rows, &mut steps);
    let p = params(opts, heads, layout.b, layout.t, 0, 0, 0, 0, &aux, layout.pos0);
    // `lse` and `D` per (row, head), then with documents each row's document start (see the prep kernel).
    let stats = g.storage_uninit(2 * rows * nh + if opts.documents { rows } else { 0 });
    let (spare_sink, spare_rel, spare_doc) = (g.storage(1), g.storage(1), g.storage(1));
    let row_threads = word(rows * nh, "query rows * heads");
    steps.push(g.dispatch(ids.bwd_prep, &[out, d_out, lse, &aux.buf, &stats, grads.d_sink.unwrap_or(&spare_sink), inputs.doc_pos.unwrap_or(&spare_doc)], &p, gpu_core::Dispatch::Threads(row_threads)));
    steps.push(g.dispatch(ids.bwd_dq, &[q, k, v, d_out, &stats, &aux.buf, d_q, grads.d_rel.unwrap_or(&spare_rel)], &p, gpu_core::Dispatch::Threads(row_threads)));
    let key_threads = word(rows * u64::from(heads.n_kv_heads), "key rows * kv heads");
    steps.push(g.dispatch(ids.bwd_dkv, &[q, k, v, d_out, &stats, &aux.buf, d_k, d_v], &p, gpu_core::Dispatch::Threads(key_threads)));
    if let (Some(bias), Some(d_bank)) = (opts.bias, grads.d_bank) {
        steps.push(g.dispatch(ids.bwd_dbank, &[q, k, v, d_out, &stats, &aux.buf, d_bank], &p, gpu_core::Dispatch::Threads(bias.extent)));
    }
    steps
}

/// Where one sequence's queries and keys sit for [`flash_ext_paged_fwd`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedSeq {
    /// First row of this sequence's queries in `q`/`out`/`lse`.
    pub q_row0: u32,
    /// Query rows (consecutive positions).
    pub n_q: u32,
    /// Absolute position of the first query.
    pub q_pos0: u32,
    /// Absolute position held by logical KV slot 0: every key this sequence reads sits at a position
    /// `>= kv_pos0`. 0 for a cache that starts at the sequence's first token (and for a ring).
    pub kv_pos0: u32,
}

/// The `index` words [`PagedKv::index`] holds: the `seqs` records, then each sequence's block table
/// (`tables[s]`, padded to `max_bt` entries).
#[must_use]
pub fn paged_index(seqs: &[PagedSeq], tables: &[&[u32]], max_bt: u32) -> Vec<u32> {
    assert_eq!(seqs.len(), tables.len(), "flash_ext::paged_index: one block table per sequence");
    let mut words = Vec::with_capacity(seqs.len() * (4 + max_bt as usize));
    for s in seqs {
        words.extend([s.q_row0, s.n_q, s.q_pos0, s.kv_pos0]);
    }
    for t in tables {
        assert!(t.len() <= max_bt as usize, "flash_ext::paged_index: a block table of {} entries exceeds max_bt {max_bt}", t.len());
        words.extend_from_slice(t);
        words.extend(std::iter::repeat_n(0, max_bt as usize - t.len()));
    }
    words
}

/// The paged (or ring) view [`flash_ext_paged_fwd`] attends through.
#[derive(Clone, Copy)]
pub struct PagedKv<'a> {
    /// [`paged_index`] of the sequences.
    pub index: &'a DeviceBuffer,
    pub n_seqs: u32,
    /// Most query rows of any one sequence.
    pub max_q: u32,
    /// Query rows in `q` (the sum of every sequence's `n_q`, or more).
    pub n_rows: u32,
    pub block_size: u32,
    pub max_bt: u32,
    /// 0: a growing cache, logical slot `p - kv_pos0`. Otherwise a ring of `ring` rows: logical slot
    /// `(p - kv_pos0) % ring`, which needs a window no wider than the ring (a query cannot see more keys than it holds).
    pub ring: u32,
}

/// The K and V planes a paged forward reads: rows of `n_kv_heads * head_dim` f32 words, or the same rows as
/// packed bf16 pairs (low half first).
#[derive(Clone, Copy)]
pub enum KvPlanes<'a> {
    F32 { k: &'a DeviceBuffer, v: &'a DeviceBuffer },
    Bf16 { k: &'a DeviceBuffer, v: &'a DeviceBuffer },
}

impl<'a> KvPlanes<'a> {
    /// The planes of `layer`. Panics on a tier the kernels do not read (int8, FP8): refused, never reinterpreted.
    pub fn of(layer: &'a KvLayer) -> KvPlanes<'a> {
        match (layer.k.tier(), layer.v.tier()) {
            (KvTier::F32, KvTier::F32) => KvPlanes::F32 { k: layer.k.data(), v: layer.v.data() },
            (KvTier::BF16, KvTier::BF16) => KvPlanes::Bf16 { k: layer.k.data(), v: layer.v.data() },
            (tk, tv) => panic!("flash_ext: K/V planes {tk}/{tv}: the extended attention reads f32 or bf16 planes"),
        }
    }
}

/// Forward of `paged.n_rows` query rows against paged or ring K/V planes that already hold every key the
/// queries read (append first). Overwrites `out` and `lse` (`[rows, n_heads]`) for the rows the sequences name.
#[allow(clippy::too_many_arguments)]
pub fn flash_ext_paged_fwd(g: &Gpu, ids: &FlashExtIds, heads: &HeadShape, opts: &AttnOptions, inputs: &AttnInputs, paged: &PagedKv, q: &DeviceBuffer, kv: KvPlanes, out: &DeviceBuffer, lse: &DeviceBuffer) -> Vec<Step> {
    opts.check(heads);
    assert!(!opts.documents, "flash_ext_paged_fwd: packed documents are a training layout; a paged cache holds one sequence per table");
    assert!(paged.block_size >= 1 && paged.max_bt >= 1, "flash_ext_paged_fwd: empty block geometry");
    if paged.ring > 0 {
        let w = opts.window.unwrap_or(u32::MAX);
        assert!(w <= paged.ring, "flash_ext_paged_fwd: a ring of {} rows cannot serve a window of {w} keys", paged.ring);
        assert!(paged.ring <= paged.block_size * paged.max_bt, "flash_ext_paged_fwd: a ring of {} rows exceeds the {} rows its block table addresses", paged.ring, paged.block_size * paged.max_bt);
    }
    let (kernel, k, v) = match kv {
        KvPlanes::F32 { k, v } => (ids.fwd, k, v),
        KvPlanes::Bf16 { k, v } => (ids.fwd_bf16, k, v),
    };
    let mut steps = Vec::new();
    let aux = pack_aux(g, ids, opts, heads, inputs, u64::from(paged.n_rows), &mut steps);
    let p = params(opts, heads, paged.n_seqs, paged.max_q, paged.block_size, paged.max_bt, paged.ring, PAGED, &aux, 0);
    let threads = word(u64::from(paged.n_seqs) * u64::from(paged.max_q) * u64::from(heads.n_heads), "sequences * query rows * heads");
    steps.push(g.dispatch(kernel.kernel, &[q, k, v, paged.index, &aux.buf, out, lse], &p, kernel.dispatch(threads)));
    steps
}
