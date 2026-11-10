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
mod movement;
mod norm;
mod rows;

use std::collections::{BTreeMap, HashMap, HashSet};

use gpu_core::DeviceBuffer;
use paramstore::ParamStore;

use self::launch::Rec;
use super::Device;
use crate::train::graph::Graph;
use crate::train::ops::Seg;
use crate::train::seq::AttnShape;
use crate::train::tape::Id;
use crate::vision::ops::Map;

/// Kernels the training graph dispatches by name, beyond what the inference path lists.
pub(super) const KERNELS: &[(&str, &str)] = &[
    ("gelu", ::kernels::GELU),
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
    param_ids: HashMap<String, Id>,
    ones: HashMap<usize, DeviceBuffer>,
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
            param_ids: HashMap::new(),
            ones: HashMap::new(),
        }
    }

    /// Register the low-rank pair of the adapter for projection `key`: `a` is `[in, rank]`, `b` is `[rank, out]`,
    /// both entries of the store, applied as `scale * (x A) B`.
    pub fn with_lora(mut self, key: &str, a: &str, b: &str, rank: usize, scale: f64) -> DevTape<'a> {
        self.lora.insert(key.to_string(), LoraEntry { a: a.to_string(), b: b.to_string(), rank, scale });
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
            self.nodes.push(Node { out, bwd: Box::new(bwd) });
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
    /// gradient buffers.
    pub fn backward(&mut self, loss: Id) {
        assert_eq!(self.vals[loss.0].len, 1, "backward starts from a scalar");
        self.grads.resize(self.vals.len(), None);
        if self.vals[loss.0].needs_grad {
            self.grads[loss.0] = Some(self.dev.gpu().storage_init("loss grad", &[1.0]));
        }
        let nodes = std::mem::take(&mut self.nodes);
        for node in nodes.iter().rev() {
            let Some(gy) = self.grads[node.out.0].clone() else { continue };
            let mut cx = BwdCtx { rec: &mut self.rec, vals: &self.vals, grads: &mut self.grads, store: self.store };
            (node.bwd)(&mut cx, &gy);
        }
        self.nodes = nodes;
        self.flush();
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

    fn depthwise(&mut self, _x: Id, _m: Map, _kernel: Id, _kh: usize, _kw: usize, _sy: usize, _sx: usize) -> (Id, Map) {
        unimplemented!("package B/C/D")
    }
    fn conv_dense(&mut self, _x: Id, _m: Map, _kernel: Id, _kh: usize, _kw: usize, _cout: usize, _sy: usize, _sx: usize) -> (Id, Map) {
        unimplemented!("package B/C/D")
    }
    fn upsample2(&mut self, _x: Id, _m: Map) -> (Id, Map) {
        unimplemented!("package B/C/D")
    }
    fn walsh(&mut self, _x: Id, _n: usize) -> Id {
        unimplemented!("package B/C/D")
    }
    fn band(&mut self, _u: Id, _n: usize, _w: Id, _k: usize) -> Id {
        unimplemented!("package B/C/D")
    }
    fn grn_scale(&mut self, _y: Id, _d: usize, _valid: &[bool], _gamma: Id) -> Id {
        unimplemented!("package B/C/D")
    }
    fn rope(&mut self, _x: Id, _heads: usize, _hd: usize, _angles: &[f64]) -> Id {
        unimplemented!("package B/C/D")
    }
    fn attention(&mut self, _q: Id, _k: Id, _v: Id, _s: AttnShape) -> Id {
        unimplemented!("package B/C/D")
    }
    fn causal_taps(&mut self, _x: Id, _taps: Id, _w: usize, _dilation: usize) -> Id {
        unimplemented!("package B/C/D")
    }
    fn sinkhorn(&mut self, _logits: Id, _n: usize, _iters: usize) -> Id {
        unimplemented!("package B/C/D")
    }
    fn row_mix(&mut self, _h: Id, _x: Id, _a: usize, _b: usize, _d: usize) -> Id {
        unimplemented!("package B/C/D")
    }
    fn kron(&mut self, _z: Id, _a: Id, _b: Id, _fa: usize, _fb: usize) -> Id {
        unimplemented!("package B/C/D")
    }
    fn permute_cols(&mut self, _z: Id, _n: usize, _perm: &[u32]) -> Id {
        unimplemented!("package B/C/D")
    }
    fn softmax_ce(&mut self, _logits: Id, _v: usize, _targets: &[u32], _weights: &[f64]) -> Id {
        unimplemented!("package B/C/D")
    }
    fn focal_heat(&mut self, _logits: Id, _target: &[f64], _weight: &[f64], _alpha: f64, _norm: f64) -> Id {
        unimplemented!("package B/C/D")
    }
    fn l1(&mut self, _x: Id, _target: &[f64], _weight: &[f64], _norm: f64) -> Id {
        unimplemented!("package B/C/D")
    }
    fn bce_dice(&mut self, _logits: Id, _target: &[f64], _bce_weight: f64, _dice_weight: f64) -> Id {
        unimplemented!("package B/C/D")
    }
    fn ctc(&mut self, _logits: Id, _classes: usize, _labels: &[u32]) -> Id {
        unimplemented!("package B/C/D")
    }
}

