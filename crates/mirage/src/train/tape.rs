// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A reverse-mode tape whose every operation carries a hand-written backward.
//!
//! The graph is built eagerly: each operation computes its value, stores it, and records a closure that turns the
//! output gradient into input-gradient contributions. Nothing here differentiates automatically; the backward of
//! each primitive is written out in `ops` and checked against central finite differences of the same forward in
//! `f64`. Models are compositions of primitives, so a model's backward is the chain rule over checked pieces.

use super::real::Real;

/// A value on the tape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(pub(crate) usize);

/// Gradient accumulators, one per tape value, created on first use.
pub struct Grads<R: Real> {
    g: Vec<Option<Vec<R>>>,
}

impl<R: Real> Grads<R> {
    /// The accumulator of `id`, zeroed to `n` elements on first touch.
    pub fn slot(&mut self, id: Id, n: usize) -> &mut [R] {
        let s = self.g[id.0].get_or_insert_with(|| vec![R::ZERO; n]);
        debug_assert_eq!(s.len(), n, "gradient of a tape value has a fixed length");
        s
    }

    /// Add `delta` into the gradient of `id`.
    pub fn add(&mut self, id: Id, delta: &[R]) {
        self.slot(id, delta.len()).iter_mut().zip(delta).for_each(|(a, d)| *a += *d);
    }

    /// The gradient of `id`, when anything reached it.
    pub fn get(&self, id: Id) -> Option<&[R]> {
        self.g[id.0].as_deref()
    }

    /// Move the gradient of `id` out (zeros when nothing reached it).
    pub fn take(&mut self, id: Id, n: usize) -> Vec<R> {
        self.g[id.0].take().unwrap_or_else(|| vec![R::ZERO; n])
    }
}

type Backward<R> = Box<dyn Fn(&[Vec<R>], &[R], &mut Grads<R>)>;

struct Node<R: Real> {
    out: Id,
    bwd: Backward<R>,
}

/// The recorded computation.
pub struct Tape<R: Real> {
    vals: Vec<Vec<R>>,
    nodes: Vec<Node<R>>,
}

impl<R: Real> Default for Tape<R> {
    fn default() -> Self {
        Tape { vals: Vec::new(), nodes: Vec::new() }
    }
}

impl<R: Real> Tape<R> {
    /// An empty tape.
    pub fn new() -> Tape<R> {
        Tape::default()
    }

    /// A value with no producing operation (a parameter or an input).
    pub fn leaf(&mut self, v: Vec<R>) -> Id {
        self.vals.push(v);
        Id(self.vals.len() - 1)
    }

    /// The values of `id`.
    pub fn val(&self, id: Id) -> &[R] {
        &self.vals[id.0]
    }

    /// Number of values recorded so far (the id the next value takes).
    pub(crate) fn value_count(&self) -> usize {
        self.vals.len()
    }

    /// Element count of `id`.
    pub fn len(&self, id: Id) -> usize {
        self.vals[id.0].len()
    }

    /// Whether `id` holds no elements.
    pub fn is_empty(&self, id: Id) -> bool {
        self.vals[id.0].is_empty()
    }

    /// Record an operation: its output `v` and the backward that distributes `d out` over its inputs.
    pub fn op(&mut self, v: Vec<R>, bwd: impl Fn(&[Vec<R>], &[R], &mut Grads<R>) + 'static) -> Id {
        let out = self.leaf(v);
        self.nodes.push(Node { out, bwd: Box::new(bwd) });
        out
    }

    /// Gradients of the scalar `loss` with respect to everything it depends on.
    pub fn backward(&self, loss: Id) -> Grads<R> {
        assert_eq!(self.vals[loss.0].len(), 1, "backward starts from a scalar");
        let mut grads = Grads { g: (0..self.vals.len()).map(|_| None).collect() };
        grads.g[loss.0] = Some(vec![R::ONE]);
        for node in self.nodes.iter().rev() {
            if let Some(gy) = grads.g[node.out.0].take() {
                (node.bwd)(&self.vals, &gy, &mut grads);
                // Keep the output gradient readable for callers that probe intermediates.
                grads.g[node.out.0] = Some(gy);
            }
        }
        grads
    }
}
