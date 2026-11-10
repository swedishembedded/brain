// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements gradient-checked training stacks for compact perception models for its clients.
// If your team needs expertise in running one training graph on a host oracle and on accelerators then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The primitive set a Mirage training graph is built from.
//!
//! [`Graph`] is the one interface between the model's training graph (`Net`, the vision, decoder and objective
//! builders) and the machinery that evaluates it. The host [`Tape`] implements it with closures over `Vec<R>` (and is
//! the finite-difference oracle at `f64`); the device tape (`crate::device::train`) implements it with GPU kernels.
//! The graph code is written once against this trait, so the two evaluations cannot drift structurally.

use super::ops::Seg;
use super::real::Real;
use super::seq::AttnShape;
use super::tape::{Id, Tape};
use crate::vision::ops::Map;

/// Reverse-mode primitives. Every method records its output and the backward that distributes the output gradient.
#[allow(clippy::too_many_arguments)]
pub trait Graph<R: Real> {
    /// A value with no producing operation (an input or a constant).
    fn leaf(&mut self, v: Vec<R>) -> Id;
    /// Element count of `id`.
    fn len(&self, id: Id) -> usize;
    /// The single element of a scalar value (a loss term), read back to the host.
    fn scalar(&mut self, id: Id) -> f64;

    /// A trained parameter that lives with the evaluator (device-resident); `None` when the evaluator holds none
    /// and the caller supplies the values.
    fn resident_param(&mut self, _name: &str) -> Option<Id> {
        None
    }
    /// Layer `layer` of a stacked parameter, resident with the evaluator.
    fn resident_layer(&mut self, _name: &str, _layer: usize) -> Option<Id> {
        None
    }
    /// The low-rank pair `(A, B)` of the adapter for `key` and its `(rank, alpha / rank)`, resident with the evaluator.
    fn resident_lora(&mut self, _key: &str) -> Option<(Id, Id, usize, f64)> {
        None
    }

    fn add(&mut self, a: Id, b: Id) -> Id;
    fn mul(&mut self, a: Id, b: Id) -> Id;
    fn scale(&mut self, a: Id, c: f64) -> Id;
    fn bias_add(&mut self, x: Id, b: Id) -> Id;
    fn chan_mul(&mut self, x: Id, s: Id) -> Id;
    fn scalar_mul(&mut self, x: Id, s: Id) -> Id;
    fn sigmoid(&mut self, x: Id) -> Id;
    fn silu(&mut self, x: Id) -> Id;
    fn gelu_tanh(&mut self, x: Id) -> Id;
    fn grad_scale(&mut self, x: Id, factor: f64) -> Id;
    fn linear(&mut self, x: Id, w: Id, rows_n: usize, inn: usize, out: usize) -> Id;
    fn linear_nt(&mut self, x: Id, w: Id, rows_n: usize, inn: usize, out: usize) -> Id;
    fn zc_norm(&mut self, x: Id, d: usize, scale: Id) -> Id;
    fn rms_unit(&mut self, x: Id, d: usize) -> Id;
    fn mask_rows(&mut self, x: Id, d: usize, valid: &[bool]) -> Id;
    fn plan(&mut self, src: Id, out_len: usize, segs: Vec<Seg<R>>) -> Id;
    fn gather_rows(&mut self, x: Id, d: usize, idx: &[usize]) -> Id;
    fn concat(&mut self, a: Id, b: Id) -> Id;
    fn slice(&mut self, x: Id, start: usize, len: usize) -> Id;
    fn softmax_rows(&mut self, x: Id, d: usize) -> Id;
    fn row_dot(&mut self, a: Id, b: Id, d: usize) -> Id;
    fn row_mul(&mut self, s: Id, x: Id, d: usize) -> Id;
    fn sum_weighted(&mut self, x: Id, weight: f64) -> Id;

    fn depthwise(&mut self, x: Id, m: Map, kernel: Id, kh: usize, kw: usize, sy: usize, sx: usize) -> (Id, Map);
    fn conv_dense(&mut self, x: Id, m: Map, kernel: Id, kh: usize, kw: usize, cout: usize, sy: usize, sx: usize) -> (Id, Map);
    fn upsample2(&mut self, x: Id, m: Map) -> (Id, Map);
    fn walsh(&mut self, x: Id, n: usize) -> Id;
    fn band(&mut self, u: Id, n: usize, w: Id, k: usize) -> Id;
    fn grn_scale(&mut self, y: Id, d: usize, valid: &[bool], gamma: Id) -> Id;

    fn rope(&mut self, x: Id, heads: usize, hd: usize, angles: &[f64]) -> Id;
    fn attention(&mut self, q: Id, k: Id, v: Id, s: AttnShape) -> Id;
    fn causal_taps(&mut self, x: Id, taps: Id, w: usize, dilation: usize) -> Id;
    fn sinkhorn(&mut self, logits: Id, n: usize, iters: usize) -> Id;
    fn row_mix(&mut self, h: Id, x: Id, a: usize, b: usize, d: usize) -> Id;
    fn kron(&mut self, z: Id, a: Id, b: Id, fa: usize, fb: usize) -> Id;
    fn permute_cols(&mut self, z: Id, n: usize, perm: &[u32]) -> Id;

    fn softmax_ce(&mut self, logits: Id, v: usize, targets: &[u32], weights: &[f64]) -> Id;
    fn focal_heat(&mut self, logits: Id, target: &[f64], weight: &[f64], alpha: f64, norm: f64) -> Id;
    fn l1(&mut self, x: Id, target: &[f64], weight: &[f64], norm: f64) -> Id;
    fn bce_dice(&mut self, logits: Id, target: &[f64], bce_weight: f64, dice_weight: f64) -> Id;
    fn ctc(&mut self, logits: Id, classes: usize, labels: &[u32]) -> Id;
}

/// Forward every [`Graph`] method of the host tape to its inherent implementation.
macro_rules! host_graph {
    ($( fn $name:ident(&mut self $(, $arg:ident : $ty:ty)*) -> $ret:ty; )*) => {
        $( fn $name(&mut self $(, $arg: $ty)*) -> $ret { Tape::<R>::$name(self $(, $arg)*) } )*
    };
}

impl<R: Real> Graph<R> for Tape<R> {
    fn leaf(&mut self, v: Vec<R>) -> Id {
        Tape::leaf(self, v)
    }
    fn len(&self, id: Id) -> usize {
        Tape::len(self, id)
    }
    fn scalar(&mut self, id: Id) -> f64 {
        self.val(id)[0].to_f64()
    }

    host_graph! {
        fn add(&mut self, a: Id, b: Id) -> Id;
        fn mul(&mut self, a: Id, b: Id) -> Id;
        fn scale(&mut self, a: Id, c: f64) -> Id;
        fn bias_add(&mut self, x: Id, b: Id) -> Id;
        fn chan_mul(&mut self, x: Id, s: Id) -> Id;
        fn scalar_mul(&mut self, x: Id, s: Id) -> Id;
        fn sigmoid(&mut self, x: Id) -> Id;
        fn silu(&mut self, x: Id) -> Id;
        fn gelu_tanh(&mut self, x: Id) -> Id;
        fn grad_scale(&mut self, x: Id, factor: f64) -> Id;
        fn linear(&mut self, x: Id, w: Id, rows_n: usize, inn: usize, out: usize) -> Id;
        fn linear_nt(&mut self, x: Id, w: Id, rows_n: usize, inn: usize, out: usize) -> Id;
        fn zc_norm(&mut self, x: Id, d: usize, scale: Id) -> Id;
        fn rms_unit(&mut self, x: Id, d: usize) -> Id;
        fn mask_rows(&mut self, x: Id, d: usize, valid: &[bool]) -> Id;
        fn plan(&mut self, src: Id, out_len: usize, segs: Vec<Seg<R>>) -> Id;
        fn gather_rows(&mut self, x: Id, d: usize, idx: &[usize]) -> Id;
        fn concat(&mut self, a: Id, b: Id) -> Id;
        fn slice(&mut self, x: Id, start: usize, len: usize) -> Id;
        fn softmax_rows(&mut self, x: Id, d: usize) -> Id;
        fn row_dot(&mut self, a: Id, b: Id, d: usize) -> Id;
        fn row_mul(&mut self, s: Id, x: Id, d: usize) -> Id;
        fn sum_weighted(&mut self, x: Id, weight: f64) -> Id;
        fn depthwise(&mut self, x: Id, m: Map, kernel: Id, kh: usize, kw: usize, sy: usize, sx: usize) -> (Id, Map);
        fn conv_dense(&mut self, x: Id, m: Map, kernel: Id, kh: usize, kw: usize, cout: usize, sy: usize, sx: usize) -> (Id, Map);
        fn upsample2(&mut self, x: Id, m: Map) -> (Id, Map);
        fn walsh(&mut self, x: Id, n: usize) -> Id;
        fn band(&mut self, u: Id, n: usize, w: Id, k: usize) -> Id;
        fn grn_scale(&mut self, y: Id, d: usize, valid: &[bool], gamma: Id) -> Id;
        fn rope(&mut self, x: Id, heads: usize, hd: usize, angles: &[f64]) -> Id;
        fn attention(&mut self, q: Id, k: Id, v: Id, s: AttnShape) -> Id;
        fn causal_taps(&mut self, x: Id, taps: Id, w: usize, dilation: usize) -> Id;
        fn sinkhorn(&mut self, logits: Id, n: usize, iters: usize) -> Id;
        fn row_mix(&mut self, h: Id, x: Id, a: usize, b: usize, d: usize) -> Id;
        fn kron(&mut self, z: Id, a: Id, b: Id, fa: usize, fb: usize) -> Id;
        fn permute_cols(&mut self, z: Id, n: usize, perm: &[u32]) -> Id;
        fn softmax_ce(&mut self, logits: Id, v: usize, targets: &[u32], weights: &[f64]) -> Id;
        fn focal_heat(&mut self, logits: Id, target: &[f64], weight: &[f64], alpha: f64, norm: f64) -> Id;
        fn l1(&mut self, x: Id, target: &[f64], weight: &[f64], norm: f64) -> Id;
        fn bce_dice(&mut self, logits: Id, target: &[f64], bce_weight: f64, dice_weight: f64) -> Id;
        fn ctc(&mut self, logits: Id, classes: usize, labels: &[u32]) -> Id;
    }
}
