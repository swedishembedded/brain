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

/// A position on a tape: how many nodes and values exist. A recompute block spans the range from its mark to
/// the tape's end when it is sealed, and a replay appends past the end and is truncated back to its own mark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mark {
    pub(crate) nodes: usize,
    pub(crate) vals: usize,
}

impl Mark {
    /// Number of values that existed at the mark; ids at or above it were created after it.
    pub fn values(self) -> usize {
        self.vals
    }
}

struct Node<R: Real> {
    out: Id,
    bwd: Backward<R>,
    /// `Some(i)` marks the barrier a sealed recompute block left in place of its nodes (see [`Tape::seal`]).
    block: Option<usize>,
}

/// A backward in progress: the gradients so far and the next node to run (counting down).
struct Reverse<R: Real> {
    grads: Grads<R>,
    cursor: usize,
}

/// The recorded computation.
pub struct Tape<R: Real> {
    vals: Vec<Vec<R>>,
    nodes: Vec<Node<R>>,
    reverse: Option<Reverse<R>>,
    finished: Option<Grads<R>>,
}

impl<R: Real> Default for Tape<R> {
    fn default() -> Self {
        Tape { vals: Vec::new(), nodes: Vec::new(), reverse: None, finished: None }
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
        self.nodes.push(Node { out, bwd: Box::new(bwd), block: None });
        out
    }

    /// Elements held by the recorded values right now (a sealed block's freed values count zero).
    pub fn live_elements(&self) -> usize {
        self.vals.iter().map(Vec::len).sum()
    }

    /// The current position (see [`Mark`]).
    pub fn mark(&self) -> Mark {
        Mark { nodes: self.nodes.len(), vals: self.vals.len() }
    }

    /// Close the recompute block that began at `mark`: its nodes (and everything their backward closures saved)
    /// are dropped, every value it created is freed except the ones in `keep` (its outputs, and the parameters it
    /// touched first), and one barrier node, tagged `block`, stands in for it. The backward stops at the barrier
    /// and hands control to the caller, who rebuilds the block and calls [`Tape::replay_backward`].
    pub fn seal(&mut self, mark: Mark, keep: &[Id], block: usize) {
        assert!(self.reverse.is_none(), "a block cannot be sealed during a backward");
        self.nodes.truncate(mark.nodes);
        let keep: std::collections::HashSet<usize> = keep.iter().map(|id| id.0).collect();
        for (i, v) in self.vals.iter_mut().enumerate().skip(mark.vals) {
            if !keep.contains(&i) {
                *v = Vec::new();
            }
        }
        self.nodes.push(Node { out: Id(usize::MAX), bwd: Box::new(|_, _, _| {}), block: Some(block) });
    }

    /// Gradients of the scalar `loss` with respect to everything it depends on. A tape holding sealed blocks
    /// cannot rebuild them itself: differentiate those through `Net::backward`.
    pub fn backward(&self, loss: Id) -> Grads<R> {
        assert_eq!(self.vals[loss.0].len(), 1, "backward starts from a scalar");
        let mut grads = Grads { g: (0..self.vals.len()).map(|_| None).collect() };
        grads.g[loss.0] = Some(vec![R::ONE]);
        for node in self.nodes.iter().rev() {
            assert!(node.block.is_none(), "this tape holds sealed recompute blocks: run the backward through Net::backward");
            self.run(node, &mut grads);
        }
        grads
    }

    fn run(&self, node: &Node<R>, grads: &mut Grads<R>) {
        if let Some(gy) = grads.g[node.out.0].take() {
            (node.bwd)(&self.vals, &gy, grads);
            // Keep the output gradient readable for callers that probe intermediates.
            grads.g[node.out.0] = Some(gy);
        }
    }

    /// Start a backward from the scalar `loss` that can stop at sealed blocks ([`Tape::backward_to_block`]).
    pub fn backward_begin(&mut self, loss: Id) {
        assert_eq!(self.vals[loss.0].len(), 1, "backward starts from a scalar");
        let mut grads = Grads { g: (0..self.vals.len()).map(|_| None).collect() };
        grads.g[loss.0] = Some(vec![R::ONE]);
        self.reverse = Some(Reverse { grads, cursor: self.nodes.len() });
    }

    /// Run the backward down to the next sealed block and return its index, or `None` when the tape is done.
    pub fn backward_to_block(&mut self) -> Option<usize> {
        let mut st = self.reverse.take().expect("backward_begin starts a backward");
        let mut hit = None;
        while st.cursor > 0 {
            st.cursor -= 1;
            let node = &self.nodes[st.cursor];
            if let Some(b) = node.block {
                hit = Some(b);
                break;
            }
            self.run(node, &mut st.grads);
        }
        self.reverse = Some(st);
        hit
    }

    /// Differentiate the block that was just rebuilt past `mark`: the gradient that reached each sealed output
    /// `old` seeds its rebuilt twin `new`, the rebuilt nodes run in reverse, and the rebuilt nodes and values are
    /// dropped again. Gradients that reached values below `mark` (block inputs and parameters) stay.
    pub fn replay_backward(&mut self, mark: Mark, pairs: &[(Id, Id)]) {
        let mut st = self.reverse.take().expect("a replay needs a backward in progress");
        st.grads.g.resize_with(self.vals.len(), || None);
        for &(new, old) in pairs {
            if new != old {
                if let Some(gy) = st.grads.g[old.0].take() {
                    st.grads.add(new, &gy);
                }
            }
        }
        for node in self.nodes[mark.nodes..].iter().rev() {
            self.run(node, &mut st.grads);
        }
        self.nodes.truncate(mark.nodes);
        self.vals.truncate(mark.vals);
        st.grads.g.truncate(mark.vals);
        self.reverse = Some(st);
    }

    /// Finish a backward: its gradients become readable with [`Tape::take_grads`].
    pub fn backward_end(&mut self) {
        let st = self.reverse.take().expect("a backward is in progress");
        self.finished = Some(st.grads);
    }

    /// The gradients of the last finished backward.
    pub fn take_grads(&mut self) -> Grads<R> {
        self.finished.take().expect("no finished backward")
    }
}
