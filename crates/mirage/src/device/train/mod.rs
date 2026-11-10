// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements GPU training for compact perception models for its clients. If your team needs
// expertise in moving a gradient-checked host trainer onto accelerators then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The device implementation of the training graph ([`Graph`]).
//!
//! [`DevTape`] records the same primitives the host [`crate::train::tape::Tape`] does, but every value is a
//! [`DeviceBuffer`] and every forward and backward is a kernel [`Step`]. Steps are collected and submitted lazily:
//! on [`DevTape::scalar`] / [`DevTape::value`] (a readback) and on [`DevTape::backward`].
//!
//! Design:
//! - Parameters live in a [`ParamStore`] (weights `w(name)`, gradients `g(name)`). A decoder layer stacked on a
//!   leading axis in the checkpoint is registered as one entry per layer, `"{name}#{layer}"`, so a layer is a plain
//!   buffer and needs no slice copy. Dense kernels keep the checkpoint's `[in, out]` layout; nothing is transposed.
//! - Every value carries `needs_grad`: a trainable parameter is true, a constant or a frozen parameter false, an
//!   operation's output the OR of its inputs. A node whose output needs no gradient records no backward, and a
//!   node accumulates only into inputs that need one, so a frozen encoder costs no backward at all.
//! - Backward kernels accumulate (`dst += ...`) with one owning thread per output element: no atomics, and the
//!   result is deterministic. Parameter gradients accumulate straight into the store's gradient buffers (the caller
//!   zeroes them with `ParamStore::zero_grads`); other gradients are created on first touch and zero-cleared in the
//!   submission that first uses them.
//! - Primitives are implemented in families: [`elementwise`], [`linear`], [`norm`], [`movement`], [`rows`].

mod elementwise;
mod launch;
mod linear;
mod losses;
pub mod params;
mod movement;
mod norm;
mod rows;
mod seq;
mod spatial;
pub mod trainer;

use std::collections::{BTreeMap, HashMap, HashSet};

use gpu_core::DeviceBuffer;
use paramstore::ParamStore;

use self::launch::Rec;
use super::Device;
use crate::train::graph::Graph;
use crate::train::ops::Seg;
use crate::train::seq::AttnShape;
use crate::train::tape::{Id, Mark};
use crate::vision::ops::Map;

/// Kernels the training graph dispatches by name, beyond what the inference path lists.
pub(super) const KERNELS: &[(&str, &str)] = &[
    ("gelu", ::kernels::GELU),
    ("grad_scale", ::kernels::GRAD_SCALE),
    ("gradnorm_sq", ::kernels::GRADNORM_SQ),
    ("gradnorm_part", ::kernels::GRADNORM_PART),
    ("adamw", ::kernels::ADAMW),
    ("clip_coef", ::kernels::CLIP_COEF),
    ("clip_coef_wg", ::kernels::CLIP_COEF_WG),
    ("grad_scale_buf", ::kernels::GRAD_SCALE_BUF),
    ("mirage_t_ew", ::kernels::MIRAGE_T_EW),
    ("mirage_t_mul_acc", ::kernels::MIRAGE_T_MUL_ACC),
    ("mirage_t_col_red", ::kernels::MIRAGE_T_COL_RED),
    ("mirage_t_row_red", ::kernels::MIRAGE_T_ROW_RED),
    ("mirage_t_sum", ::kernels::MIRAGE_T_SUM),
    ("mirage_t_bcast_acc", ::kernels::MIRAGE_T_BCAST_ACC),
    ("mirage_t_act_bwd", ::kernels::MIRAGE_T_ACT_BWD),
    ("mirage_t_span", ::kernels::MIRAGE_T_SPAN),
    ("mirage_t_plan", ::kernels::MIRAGE_T_PLAN),
    ("mirage_t_rms_stats", ::kernels::MIRAGE_T_RMS_STATS),
    ("mirage_t_rms_dx", ::kernels::MIRAGE_T_RMS_DX),
    ("mirage_t_rms_dscale", ::kernels::MIRAGE_T_RMS_DSCALE),
    ("mirage_t_softmax", ::kernels::MIRAGE_T_SOFTMAX),
    ("mirage_t_softmax_bwd", ::kernels::MIRAGE_T_SOFTMAX_BWD),
    ("mirage_t_dwconv_dx", ::kernels::MIRAGE_T_DWCONV_DX),
    ("mirage_t_dwconv_dk", ::kernels::MIRAGE_T_DWCONV_DK),
    ("mirage_t_conv_dx", ::kernels::MIRAGE_T_CONV_DX),
    ("mirage_t_conv_dk", ::kernels::MIRAGE_T_CONV_DK),
    ("mirage_t_band", ::kernels::MIRAGE_T_BAND),
    ("mirage_t_band_dx", ::kernels::MIRAGE_T_BAND_DX),
    ("mirage_t_band_dw", ::kernels::MIRAGE_T_BAND_DW),
    ("mirage_t_grn_de", ::kernels::MIRAGE_T_GRN_DE),
    ("mirage_t_grn_dy", ::kernels::MIRAGE_T_GRN_DY),
    ("mirage_t_rope", ::kernels::MIRAGE_T_ROPE),
    ("mirage_t_attn_fwd", ::kernels::MIRAGE_T_ATTN_FWD),
    ("mirage_t_attn_delta", ::kernels::MIRAGE_T_ATTN_DELTA),
    ("mirage_t_attn_dq", ::kernels::MIRAGE_T_ATTN_DQ),
    ("mirage_t_attn_dkv", ::kernels::MIRAGE_T_ATTN_DKV),
    ("mirage_t_taps", ::kernels::MIRAGE_T_TAPS),
    ("mirage_t_taps_dx", ::kernels::MIRAGE_T_TAPS_DX),
    ("mirage_t_taps_dt", ::kernels::MIRAGE_T_TAPS_DT),
    ("mirage_t_sinkhorn", ::kernels::MIRAGE_T_SINKHORN),
    ("mirage_t_sinkhorn_bwd", ::kernels::MIRAGE_T_SINKHORN_BWD),
    ("mirage_t_mix", ::kernels::MIRAGE_T_MIX),
    ("mirage_t_mix_dh", ::kernels::MIRAGE_T_MIX_DH),
    ("mirage_t_mix_dx", ::kernels::MIRAGE_T_MIX_DX),
    ("mirage_t_kron_right", ::kernels::MIRAGE_T_KRON_RIGHT),
    ("mirage_t_kron_left", ::kernels::MIRAGE_T_KRON_LEFT),
    ("mirage_t_kron_da", ::kernels::MIRAGE_T_KRON_DA),
    ("mirage_t_kron_db", ::kernels::MIRAGE_T_KRON_DB),
    ("mirage_t_gather_cols", ::kernels::MIRAGE_T_GATHER_COLS),
    ("mirage_t_ce_rows", ::kernels::MIRAGE_T_CE_ROWS),
    ("mirage_t_ce_bwd", ::kernels::MIRAGE_T_CE_BWD),
    ("mirage_t_focal", ::kernels::MIRAGE_T_FOCAL),
    ("mirage_t_l1", ::kernels::MIRAGE_T_L1),
    ("mirage_t_bce_dice", ::kernels::MIRAGE_T_BCE_DICE),
    ("mirage_t_bce_dice_bwd", ::kernels::MIRAGE_T_BCE_DICE_BWD),
    ("mirage_t_ctc_lp", ::kernels::MIRAGE_T_CTC_LP),
    ("mirage_t_ctc_dp", ::kernels::MIRAGE_T_CTC_DP),
    ("mirage_t_ctc_grad", ::kernels::MIRAGE_T_CTC_GRAD),
];

/// One recorded value.
struct Val {
    buf: DeviceBuffer,
    len: usize,
    needs_grad: bool,
    /// The parameter behind this value: its gradient is the store's gradient buffer.
    param: Option<String>,
}

/// The backward of one node: distributes the output gradient `gy` over the inputs that need one.
type Backward = Box<dyn Fn(&mut BwdCtx<'_, '_>, &DeviceBuffer)>;

struct Node {
    out: Id,
    bwd: Backward,
    /// `Some(i)` marks the barrier a sealed recompute block left in place of its nodes (see [`Graph::seal`]).
    block: Option<usize>,
}

/// What a backward closure sees: forward values, gradient accumulators, and the step recorder.
pub(crate) struct BwdCtx<'t, 'd> {
    pub(crate) rec: &'t mut Rec<'d>,
    vals: &'t [Val],
    grads: &'t mut Vec<Option<DeviceBuffer>>,
    store: &'t ParamStore,
}

impl BwdCtx<'_, '_> {
    /// The forward value of `id`.
    pub(crate) fn val(&self, id: Id) -> DeviceBuffer {
        self.vals[id.0].buf.clone()
    }

    /// The accumulator the gradient of `id` is added into, or `None` when `id` needs no gradient. A parameter's
    /// accumulator is the store's gradient buffer; any other is created on first touch and cleared before use.
    pub(crate) fn target(&mut self, id: Id) -> Option<DeviceBuffer> {
        let v = &self.vals[id.0];
        if !v.needs_grad {
            return None;
        }
        if let Some(name) = &v.param {
            return Some(self.store.g(name).clone());
        }
        if self.grads[id.0].is_none() {
            let g = self.rec.alloc_zeroed(v.len);
            self.grads[id.0] = Some(g);
        }
        self.grads[id.0].clone()
    }
}

/// The adapter registered for one projection key.
struct LoraEntry {
    a: String,
    b: String,
    rank: usize,
    scale: f64,
}

/// The device training tape.
pub struct DevTape<'a> {
    dev: &'a Device,
    store: &'a ParamStore,
    trainable: HashSet<String>,
    lora: BTreeMap<String, LoraEntry>,
    leaves_need_grad: bool,
    vals: Vec<Val>,
    nodes: Vec<Node>,
    rec: Rec<'a>,
    grads: Vec<Option<DeviceBuffer>>,
    /// The next node a backward in progress runs (counting down).
    cursor: usize,
    param_ids: HashMap<String, Id>,
    ones: HashMap<usize, DeviceBuffer>,
    walsh: HashMap<usize, DeviceBuffer>,
}

impl<'a> DevTape<'a> {
    /// A tape over the parameters of `store`. A parameter is trained when the store holds a gradient buffer for it
    /// and it was not withdrawn by `freeze_where`.
    pub fn new(dev: &'a Device, store: &'a ParamStore) -> DevTape<'a> {
        let trainable = store.trained_params().map(|(n, _)| n.clone()).collect();
        DevTape {
            dev,
            store,
            trainable,
            lora: BTreeMap::new(),
            leaves_need_grad: false,
            vals: Vec::new(),
            nodes: Vec::new(),
            rec: Rec::new(dev),
            grads: Vec::new(),
            cursor: 0,
            param_ids: HashMap::new(),
            ones: HashMap::new(),
            walsh: HashMap::new(),
        }
    }

    /// Register the low-rank pair of the adapter for projection `key`: `a` is `[in, rank]`, `b` is `[rank, out]`,
    /// both entries of the store, applied as `scale * (x A) B`.
    pub fn with_lora(mut self, key: &str, a: &str, b: &str, rank: usize, scale: f64) -> DevTape<'a> {
        self.lora.insert(key.to_string(), LoraEntry { a: a.to_string(), b: b.to_string(), rank, scale });
        self
    }

    /// An evaluation tape: no parameter or input receives a gradient, so no backward is recorded.
    pub fn without_grads(mut self) -> DevTape<'a> {
        self.trainable.clear();
        self.leaves_need_grad = false;
        self
    }

    /// Make every [`Graph::leaf`] a value that receives a gradient, readable with [`DevTape::grad`] (the gradient
    /// checks compare input gradients with the host tape). Without it leaves are constants.
    pub fn with_leaf_grads(mut self) -> DevTape<'a> {
        self.leaves_need_grad = true;
        self
    }

    pub(crate) fn push_val(&mut self, buf: DeviceBuffer, len: usize, needs_grad: bool, param: Option<String>) -> Id {
        self.vals.push(Val { buf, len, needs_grad, param });
        Id(self.vals.len() - 1)
    }

    pub(crate) fn len_of(&self, id: Id) -> usize {
        self.vals[id.0].len
    }

    /// A constant buffer holding `v`.
    pub(crate) fn upload(&self, v: &[f32]) -> DeviceBuffer {
        if v.is_empty() {
            self.dev.gpu().storage(1)
        } else {
            self.dev.gpu().storage_init("constant", v)
        }
    }

    /// A constant buffer holding the words `v`.
    pub(crate) fn upload_words(&self, v: &[u32]) -> DeviceBuffer {
        let buf = self.dev.gpu().storage(v.len().max(1) as u64);
        if !v.is_empty() {
            self.dev.gpu().write(&buf, v);
        }
        buf
    }

    /// A constant buffer holding `v` converted to f32.
    pub(crate) fn upload_f64(&self, v: &[f64]) -> DeviceBuffer {
        self.upload(&v.iter().map(|&x| x as f32).collect::<Vec<_>>())
    }

    pub(crate) fn buf(&self, id: Id) -> DeviceBuffer {
        self.vals[id.0].buf.clone()
    }

    pub(crate) fn needs(&self, id: Id) -> bool {
        self.vals[id.0].needs_grad
    }

    /// Record an operation: `out` has `len` elements in `buf`; `bwd` runs only when something reached its gradient
    /// and some input needs one.
    pub(crate) fn op(&mut self, buf: DeviceBuffer, len: usize, inputs: &[Id], bwd: impl Fn(&mut BwdCtx<'_, '_>, &DeviceBuffer) + 'static) -> Id {
        let needs_grad = inputs.iter().any(|&i| self.needs(i));
        let out = self.push_val(buf, len, needs_grad, None);
        if needs_grad {
            self.nodes.push(Node { out, bwd: Box::new(bwd), block: None });
        }
        out
    }

    /// A cached buffer of `d` ones (the weight of a unit RMS norm).
    pub(crate) fn ones(&mut self, d: usize) -> DeviceBuffer {
        let dev = self.dev;
        self.ones.entry(d).or_insert_with(|| dev.gpu().storage_init("ones", &vec![1.0f32; d])).clone()
    }

    fn flush(&mut self) {
        self.rec.submit();
    }

    /// The values of `id`, after running everything recorded so far.
    pub fn value(&mut self, id: Id) -> Vec<f32> {
        self.flush();
        self.dev.read_at(&self.vals[id.0].buf, 0, self.vals[id.0].len)
    }

    /// The gradient of `id` after [`DevTape::backward`]: a parameter's gradient buffer in the store (everything
    /// accumulated there since its last zeroing), or the accumulator of any other value; `None` when no gradient
    /// reached it.
    pub fn grad(&self, id: Id) -> Option<Vec<f32>> {
        let v = &self.vals[id.0];
        match &v.param {
            Some(name) if v.needs_grad => Some(self.store.read_grad(self.dev.gpu(), name)),
            Some(_) => None,
            None => self.grads.get(id.0)?.as_ref().map(|g| self.dev.read_at(g, 0, v.len)),
        }
    }

    /// Number of recorded nodes with a backward (operations whose output needs a gradient).
    pub fn backward_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Run the forward steps still pending, then the backward of `loss` (a scalar), accumulating into the store's
    /// gradient buffers. A tape holding sealed recompute blocks cannot rebuild them itself: differentiate those
    /// through `Net::backward`.
    pub fn backward(&mut self, loss: Id) {
        self.backward_begin_impl(loss);
        assert!(self.backward_to_block_impl().is_none(), "this tape holds sealed recompute blocks: run the backward through Net::backward");
        self.flush();
    }

    fn backward_begin_impl(&mut self, loss: Id) {
        assert_eq!(self.vals[loss.0].len, 1, "backward starts from a scalar");
        self.grads.resize(self.vals.len(), None);
        if self.vals[loss.0].needs_grad {
            self.grads[loss.0] = Some(self.dev.gpu().storage_init("loss grad", &[1.0]));
        }
        self.cursor = self.nodes.len();
    }

    fn run_nodes(&mut self, from: usize, to: usize) {
        for i in (from..to).rev() {
            let node = &self.nodes[i];
            let Some(gy) = self.grads[node.out.0].clone() else { continue };
            let mut cx = BwdCtx { rec: &mut self.rec, vals: &self.vals, grads: &mut self.grads, store: self.store };
            (node.bwd)(&mut cx, &gy);
        }
    }

    fn backward_to_block_impl(&mut self) -> Option<usize> {
        while self.cursor > 0 {
            let i = self.cursor - 1;
            if let Some(b) = self.nodes[i].block {
                self.cursor = i;
                return Some(b);
            }
            self.run_nodes(i, i + 1);
            self.cursor = i;
        }
        None
    }
}

impl DevTape<'_> {
    fn resident(&mut self, key: String) -> Option<Id> {
        if let Some(&id) = self.param_ids.get(&key) {
            return Some(id);
        }
        let buf = self.store.weight.get(&key)?.clone();
        let len = self.store.numel(&key);
        let needs = self.trainable.contains(&key);
        let id = self.push_val(buf, len, needs, Some(key.clone()));
        self.param_ids.insert(key, id);
        Some(id)
    }
}

impl Graph<f32> for DevTape<'_> {
    fn leaf(&mut self, v: Vec<f32>) -> Id {
        let buf = self.upload(&v);
        self.push_val(buf, v.len(), self.leaves_need_grad, None)
    }
    fn len(&self, id: Id) -> usize {
        self.vals[id.0].len
    }
    fn scalar(&mut self, id: Id) -> f64 {
        f64::from(self.value(id)[0])
    }

    fn mark(&self) -> Mark {
        Mark { nodes: self.nodes.len(), vals: self.vals.len() }
    }
    fn seal(&mut self, mark: Mark, keep: &[Id], block: usize) {
        // The pending forward steps still hold the buffers about to be released: run them first.
        self.flush();
        self.nodes.truncate(mark.nodes);
        let keep: HashSet<usize> = keep.iter().map(|id| id.0).collect();
        let released = self.upload(&[0.0]);
        for (i, v) in self.vals.iter_mut().enumerate().skip(mark.vals) {
            // A parameter's buffer belongs to the store, not to the block.
            if v.param.is_none() && !keep.contains(&i) {
                v.buf = released.clone();
            }
        }
        self.nodes.push(Node { out: Id(usize::MAX), bwd: Box::new(|_, _| {}), block: Some(block) });
    }
    fn backward_begin(&mut self, loss: Id) {
        self.backward_begin_impl(loss);
    }
    fn backward_to_block(&mut self) -> Option<usize> {
        self.backward_to_block_impl()
    }
    fn replay_backward(&mut self, mark: Mark, pairs: &[(Id, Id)]) {
        self.grads.resize(self.vals.len(), None);
        for &(new, old) in pairs {
            if new != old {
                if let Some(g) = self.grads[old.0].take() {
                    self.grads[new.0] = Some(g);
                }
            }
        }
        let end = self.nodes.len();
        self.run_nodes(mark.nodes, end);
        self.flush();
        self.nodes.truncate(mark.nodes);
        self.vals.truncate(mark.vals);
        self.grads.truncate(mark.vals);
    }
    fn backward_end(&mut self) {
        self.flush();
    }

    fn resident_param(&mut self, name: &str) -> Option<Id> {
        self.resident(name.to_string())
    }
    fn resident_layer(&mut self, name: &str, layer: usize) -> Option<Id> {
        self.resident(format!("{name}#{layer}"))
    }
    fn resident_lora(&mut self, key: &str) -> Option<(Id, Id, usize, f64)> {
        let e = self.lora.get(key)?;
        let (a, b, rank, scale) = (e.a.clone(), e.b.clone(), e.rank, e.scale);
        Some((self.resident(a)?, self.resident(b)?, rank, scale))
    }

    fn add(&mut self, a: Id, b: Id) -> Id {
        self.op_add(a, b)
    }
    fn mul(&mut self, a: Id, b: Id) -> Id {
        self.op_mul(a, b)
    }
    fn scale(&mut self, a: Id, c: f64) -> Id {
        self.op_scale(a, c)
    }
    fn bias_add(&mut self, x: Id, b: Id) -> Id {
        self.op_bias_add(x, b)
    }
    fn chan_mul(&mut self, x: Id, s: Id) -> Id {
        self.op_chan_mul(x, s)
    }
    fn scalar_mul(&mut self, x: Id, s: Id) -> Id {
        self.op_scalar_mul(x, s)
    }
    fn sigmoid(&mut self, x: Id) -> Id {
        self.op_activation("sigmoid", 1, x)
    }
    fn silu(&mut self, x: Id) -> Id {
        self.op_activation("silu", 0, x)
    }
    fn gelu_tanh(&mut self, x: Id) -> Id {
        self.op_activation("gelu", 2, x)
    }
    fn grad_scale(&mut self, x: Id, factor: f64) -> Id {
        self.op_grad_scale(x, factor)
    }
    fn linear(&mut self, x: Id, w: Id, rows_n: usize, inn: usize, out: usize) -> Id {
        self.op_linear(x, w, rows_n, inn, out)
    }
    fn linear_nt(&mut self, x: Id, w: Id, rows_n: usize, inn: usize, out: usize) -> Id {
        self.op_linear_nt(x, w, rows_n, inn, out)
    }
    fn zc_norm(&mut self, x: Id, d: usize, scale: Id) -> Id {
        self.op_zc_norm(x, d, Some(scale))
    }
    fn rms_unit(&mut self, x: Id, d: usize) -> Id {
        self.op_zc_norm(x, d, None)
    }
    fn mask_rows(&mut self, x: Id, d: usize, valid: &[bool]) -> Id {
        self.op_mask_rows(x, d, valid)
    }
    fn plan(&mut self, src: Id, out_len: usize, segs: Vec<Seg<f32>>) -> Id {
        self.op_plan(src, out_len, segs)
    }
    fn gather_rows(&mut self, x: Id, d: usize, idx: &[usize]) -> Id {
        let segs = idx.iter().enumerate().map(|(o, &i)| Seg::copy(o * d, i * d, d)).collect();
        self.op_plan(x, idx.len() * d, segs)
    }
    fn concat(&mut self, a: Id, b: Id) -> Id {
        self.op_concat(a, b)
    }
    fn slice(&mut self, x: Id, start: usize, len: usize) -> Id {
        self.op_slice(x, start, len)
    }
    fn softmax_rows(&mut self, x: Id, d: usize) -> Id {
        self.op_softmax_rows(x, d)
    }
    fn row_dot(&mut self, a: Id, b: Id, d: usize) -> Id {
        self.op_row_dot(a, b, d)
    }
    fn row_mul(&mut self, s: Id, x: Id, d: usize) -> Id {
        self.op_row_mul(s, x, d)
    }
    fn sum_weighted(&mut self, x: Id, weight: f64) -> Id {
        self.op_sum_weighted(x, weight)
    }

    fn depthwise(&mut self, x: Id, m: Map, kernel: Id, kh: usize, kw: usize, sy: usize, sx: usize) -> (Id, Map) {
        self.op_conv(x, m, kernel, (kh, kw), None, (sy, sx))
    }
    fn conv_dense(&mut self, x: Id, m: Map, kernel: Id, kh: usize, kw: usize, cout: usize, sy: usize, sx: usize) -> (Id, Map) {
        self.op_conv(x, m, kernel, (kh, kw), Some(cout), (sy, sx))
    }
    fn upsample2(&mut self, x: Id, m: Map) -> (Id, Map) {
        self.op_upsample2(x, m)
    }
    fn walsh(&mut self, x: Id, n: usize) -> Id {
        self.op_walsh(x, n)
    }
    fn band(&mut self, u: Id, n: usize, w: Id, k: usize) -> Id {
        self.op_band(u, n, w, k)
    }
    fn grn_scale(&mut self, y: Id, d: usize, valid: &[bool], gamma: Id) -> Id {
        self.op_grn_scale(y, d, valid, gamma)
    }
    fn rope(&mut self, x: Id, heads: usize, hd: usize, angles: &[f64]) -> Id {
        self.op_rope(x, heads, hd, angles)
    }
    fn attention(&mut self, q: Id, k: Id, v: Id, s: AttnShape) -> Id {
        self.op_attention(q, k, v, s)
    }
    fn causal_taps(&mut self, x: Id, taps: Id, w: usize, dilation: usize) -> Id {
        self.op_causal_taps(x, taps, w, dilation)
    }
    fn sinkhorn(&mut self, logits: Id, n: usize, iters: usize) -> Id {
        self.op_sinkhorn(logits, n, iters)
    }
    fn row_mix(&mut self, h: Id, x: Id, a: usize, b: usize, d: usize) -> Id {
        self.op_row_mix(h, x, a, b, d)
    }
    fn kron(&mut self, z: Id, a: Id, b: Id, fa: usize, fb: usize) -> Id {
        self.op_kron(z, a, b, fa, fb)
    }
    fn permute_cols(&mut self, z: Id, n: usize, perm: &[u32]) -> Id {
        self.op_permute_cols(z, n, perm)
    }
    fn softmax_ce(&mut self, logits: Id, v: usize, targets: &[u32], weights: &[f64]) -> Id {
        self.op_softmax_ce(logits, v, targets, weights)
    }
    fn focal_heat(&mut self, logits: Id, target: &[f64], weight: &[f64], alpha: f64, norm: f64) -> Id {
        self.op_focal_heat(logits, target, weight, alpha, norm)
    }
    fn l1(&mut self, x: Id, target: &[f64], weight: &[f64], norm: f64) -> Id {
        self.op_l1(x, target, weight, norm)
    }
    fn bce_dice(&mut self, logits: Id, target: &[f64], bce_weight: f64, dice_weight: f64) -> Id {
        self.op_bce_dice(logits, target, bce_weight, dice_weight)
    }
    fn ctc(&mut self, logits: Id, classes: usize, labels: &[u32]) -> Id {
        self.op_ctc(logits, classes, labels)
    }
}

