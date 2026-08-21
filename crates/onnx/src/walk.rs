// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A **topological import walk** over a decoded graph: bind initializers to
//! canonical brain names by their position in the node sequence, and keep the
//! two coverage ledgers that make the binding checkable.
//!
//! # Why a walk rather than a name remap
//!
//! Most brain importers map source names to brain names. That is impossible for
//! a release whose exporter **folded BatchNorm into the convolutions**: the
//! folded tensors lose their names and come back as bare SSA value numbers
//! (`1335`, `1336`, `1643`…). The only identity such a tensor has left is
//! *where in the graph it is consumed* - the n-th `Conv` node in graph order is
//! a known convolution of a known architecture.
//!
//! Binding positionally is a stronger check than a name match, not a weaker
//! one: a name map cannot notice that a graph has 48 residual adds where the
//! architecture has 49, while a walk that asserts its op sequence fails on the
//! first node that does not line up.
//!
//! # Two-way coverage
//!
//! [`Walk::finish`] passes only when
//!
//!   * every tensor the caller's canonical [`Manifest`] expects was produced
//!     exactly once, with the shape the manifest states, and
//!   * every initializer in the source graph was consumed at least once, and
//!   * every node was visited.
//!
//! A mismatch is an error naming the tensor. Nothing is ever zero-filled or
//! silently skipped.
//!
//! One source tensor may legitimately feed two consumers - exporters
//! deduplicate equal tensors, and at least one released graph shares a conv
//! bias between two convolutions - so coverage counts a source tensor as
//! covered when it is used **one or more** times.
//!
//! This lives here, next to the [`crate::read`] decoder, because it is the same
//! shape of work for every ONNX-only release: a per-model crate that grew its
//! own copy would be a second implementation of the ledger that nothing
//! compares against the first. What stays in the model crate is the
//! architecture: which op sequence to expect and which canonical name each
//! binding gets.

use std::collections::HashMap;

use crate::onnx::{GraphProto, NodeProto};
use crate::read;

/// Imported weights: canonical brain name → (shape, fp32 data).
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// A canonical tensor manifest entry: brain name + expected shape.
pub type Manifest = Vec<(String, Vec<usize>)>;

/// A cursor over a graph's nodes plus the coverage ledgers.
pub struct Walk<'g> {
    src: HashMap<String, read::OnnxTensor>,
    /// How many times each source initializer was consumed (as a weight or as
    /// graph structure). Zero at the end = an unaccounted source tensor.
    used: HashMap<String, u32>,
    out: Tensors,
    nodes: &'g [NodeProto],
    /// Cursor over `nodes`, so the op-sequence assertions read linearly.
    at: usize,
}

impl<'g> Walk<'g> {
    pub fn new(g: &'g GraphProto) -> Result<Walk<'g>, String> {
        let src = read::initializers(g)?;
        let used = src.keys().map(|k| (k.clone(), 0u32)).collect();
        Ok(Walk { src, used, out: Tensors::new(), nodes: &g.node, at: 0 })
    }

    /// The next node, asserting its op type.
    pub fn next(&mut self, op: &str) -> Result<&'g NodeProto, String> {
        let n = self
            .nodes
            .get(self.at)
            .ok_or_else(|| format!("import: graph ended early, expected a `{op}` node at index {}", self.at))?;
        if n.op_type != op {
            return Err(format!(
                "import: node {} is `{}`, expected `{op}` (name {:?})",
                self.at, n.op_type, n.name
            ));
        }
        self.at += 1;
        Ok(n)
    }

    /// The next node whatever its op type, or `None` at the end - for walks
    /// driven by a `match` on [`Walk::peek`] rather than a fixed sequence.
    pub fn next_any(&mut self) -> Option<&'g NodeProto> {
        let n = self.nodes.get(self.at)?;
        self.at += 1;
        Some(n)
    }

    /// Peek the op type of the next node without consuming it.
    pub fn peek(&self) -> Option<&'g str> {
        self.nodes.get(self.at).map(|n| n.op_type.as_str())
    }

    /// The cursor's current position - the node index to quote in an error.
    pub fn at(&self) -> usize {
        self.at
    }

    /// Bind source initializer `src_name` to canonical `dst`, optionally
    /// reshaping (an ONNX PReLU slope is `[C,1,1]`; brain wants `[C]`).
    pub fn bind(&mut self, dst: &str, src_name: &str, shape: Vec<usize>) -> Result<(), String> {
        let t = self
            .src
            .get(src_name)
            .ok_or_else(|| format!("import: {dst}: source initializer `{src_name}` not found"))?;
        let want: usize = shape.iter().product();
        if t.data.len() != want {
            return Err(format!(
                "import: {dst} (source `{src_name}`) has {} values, expected {want} for shape {shape:?}",
                t.data.len()
            ));
        }
        let data = t.data.clone();
        *self.used.get_mut(src_name).expect("src key exists") += 1;
        if self.out.insert(dst.to_string(), (shape, data)).is_some() {
            return Err(format!("import: duplicate mapping onto {dst}"));
        }
        Ok(())
    }

    /// Mark a node's initializer inputs as consumed by graph STRUCTURE rather
    /// than as weights (a `Reshape` shape, a `Resize` scales tensor, a `Slice`
    /// bound, a `Gather` index). They are not parameters, but they must still be
    /// accounted for, or the coverage check would flag them as unused.
    pub fn ack_structural(&mut self, n: &NodeProto) {
        for i in n.input.iter().skip(1) {
            if let Some(c) = self.used.get_mut(i) {
                *c += 1;
            }
        }
    }

    /// Two-way coverage: manifest completeness + no unused source tensor + no
    /// unvisited node. `what` names the model in every error.
    pub fn finish(self, manifest: &Manifest, what: &str) -> Result<Tensors, String> {
        for (name, shape) in manifest {
            match self.out.get(name) {
                None => return Err(format!("import({what}): missing tensor {name}")),
                Some((s, d)) => {
                    if s != shape {
                        return Err(format!("import({what}): {name} shape {s:?}, expected {shape:?}"));
                    }
                    let n: usize = shape.iter().product();
                    if d.len() != n {
                        return Err(format!("import({what}): {name} has {} values, expected {n}", d.len()));
                    }
                }
            }
        }
        if self.out.len() != manifest.len() {
            let expected: std::collections::HashSet<&str> =
                manifest.iter().map(|(n, _)| n.as_str()).collect();
            let extra: Vec<&String> = self.out.keys().filter(|k| !expected.contains(k.as_str())).collect();
            return Err(format!("import({what}): produced tensors not in the manifest: {extra:?}"));
        }
        let unused: Vec<&String> =
            self.used.iter().filter(|(_, &c)| c == 0).map(|(k, _)| k).collect();
        if !unused.is_empty() {
            return Err(format!("import({what}): unused source initializers: {unused:?}"));
        }
        if self.at != self.nodes.len() {
            return Err(format!(
                "import({what}): {} of {} graph nodes were never visited",
                self.nodes.len() - self.at,
                self.nodes.len()
            ));
        }
        Ok(self.out)
    }
}

/// Assert an ONNX `Conv` node's hyperparameters against what the importing model
/// will dispatch.
///
/// A positional walk binds weights by position, so a shape check alone is not
/// enough: a release with the same tensor shapes but a different stride, pad or
/// dilation imports cleanly and produces a whole wrong network. The model's
/// geometry comes from its own config, not from the file, which is exactly why
/// the file has to be checked against it.
pub fn check_conv(n: &NodeProto, at: usize, k: i64, stride: i64, pad: i64) -> Result<(), String> {
    let want = |name: &str, got: Vec<i64>, want: Vec<i64>| -> Result<(), String> {
        if got != want {
            return Err(format!(
                "import: Conv at node {at} has {name} {got:?}, expected {want:?} for this architecture"
            ));
        }
        Ok(())
    };
    want("kernel_shape", read::attr_ints(n, "kernel_shape", &[k, k]), vec![k, k])?;
    want("strides", read::attr_ints(n, "strides", &[stride, stride]), vec![stride, stride])?;
    want("pads", read::attr_ints(n, "pads", &[pad; 4]), vec![pad; 4])?;
    want("dilations", read::attr_ints(n, "dilations", &[1, 1]), vec![1, 1])?;
    let g = read::attr_int(n, "group", 1);
    if g != 1 {
        return Err(format!("import: Conv at node {at} has group {g}; this walk imports dense convs (group 1)"));
    }
    Ok(())
}

/// [`check_conv`]'s general form: independent per-axis kernel/stride/pad, for
/// a release whose 2D convs are NOT square-symmetric (CAM++'s `FCM` stem
/// downsamples frequency only, stride `(2,1)`. `check_conv` cannot express
/// that - it asserts `strides == [s, s]`).
pub fn check_conv2d(
    n: &NodeProto,
    at: usize,
    kh: i64,
    kw: i64,
    sh: i64,
    sw: i64,
    ph: i64,
    pw: i64,
) -> Result<(), String> {
    let want = |name: &str, got: Vec<i64>, want: Vec<i64>| -> Result<(), String> {
        if got != want {
            return Err(format!(
                "import: Conv at node {at} has {name} {got:?}, expected {want:?} for this architecture"
            ));
        }
        Ok(())
    };
    want("kernel_shape", read::attr_ints(n, "kernel_shape", &[kh, kw]), vec![kh, kw])?;
    want("strides", read::attr_ints(n, "strides", &[sh, sw]), vec![sh, sw])?;
    want("pads", read::attr_ints(n, "pads", &[ph, pw, ph, pw]), vec![ph, pw, ph, pw])?;
    want("dilations", read::attr_ints(n, "dilations", &[1, 1]), vec![1, 1])?;
    let g = read::attr_int(n, "group", 1);
    if g != 1 {
        return Err(format!("import: Conv at node {at} has group {g}; this walk imports dense convs (group 1)"));
    }
    Ok(())
}

/// [`check_conv`] for a 1D `Conv` (kernel/stride/pad/dilation are each a
/// single value, not a 2-vector) - CAM++'s D-TDNN is entirely `Conv1d`, some
/// of it dilated (`check_conv` hardcodes `dilations == [1, 1]` and would
/// reject every dilated node here as a mismatch, not just report the wrong
/// shape).
pub fn check_conv1d(n: &NodeProto, at: usize, k: i64, stride: i64, pad: i64, dilation: i64) -> Result<(), String> {
    let want = |name: &str, got: Vec<i64>, want: Vec<i64>| -> Result<(), String> {
        if got != want {
            return Err(format!(
                "import: Conv at node {at} has {name} {got:?}, expected {want:?} for this architecture"
            ));
        }
        Ok(())
    };
    want("kernel_shape", read::attr_ints(n, "kernel_shape", &[k]), vec![k])?;
    want("strides", read::attr_ints(n, "strides", &[stride]), vec![stride])?;
    want("pads", read::attr_ints(n, "pads", &[pad, pad]), vec![pad, pad])?;
    want("dilations", read::attr_ints(n, "dilations", &[dilation]), vec![dilation])?;
    let g = read::attr_int(n, "group", 1);
    if g != 1 {
        return Err(format!("import: Conv at node {at} has group {g}; this walk imports dense convs (group 1)"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GraphBuilder, Node};

    /// A two-node graph whose second node carries a structural initializer.
    fn graph() -> crate::onnx::ModelProto {
        let mut g = GraphBuilder::new("walk");
        g.input_f32("x", &[1, 1, 4, 4]);
        g.output_f32("y", &[1, 1, 4, 4]);
        g.init_f32("w", &[1, 1, 3, 3], vec![1.0; 9]);
        g.init_f32("b", &[1], vec![0.5]);
        g.init_f32("shape", &[2], vec![1.0, 16.0]);
        g.add(
            Node::new("Conv", &["x", "w", "b"], &["c"])
                .name("conv")
                .attr_ints("kernel_shape", &[3, 3])
                .attr_ints("strides", &[1, 1])
                .attr_ints("pads", &[1, 1, 1, 1]),
        );
        g.add(Node::new("Reshape", &["c", "shape"], &["r"]).name("reshape"));
        // A trailing node with NO initializer input, so "every tensor consumed"
        // and "every node visited" can fail independently below.
        g.add(Node::new("Relu", &["r"], &["y"]).name("relu"));
        crate::decode_model(&g.finish()).unwrap()
    }

    /// The happy path: a conv bound positionally, a structural initializer
    /// acknowledged, and both coverage ledgers satisfied.
    #[test]
    fn a_complete_walk_binds_weights_and_accounts_for_structure() {
        let m = graph();
        let g = read::graph(&m).unwrap();
        let mut w = Walk::new(g).unwrap();
        let n = w.next("Conv").unwrap();
        check_conv(n, w.at() - 1, 3, 1, 1).unwrap();
        let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
        w.bind("conv.weight", &cw, vec![1, 1, 3, 3]).unwrap();
        w.bind("conv.bias", &cb, vec![1]).unwrap();
        let n = w.next_any().expect("a second node");
        assert_eq!(n.op_type, "Reshape");
        w.ack_structural(n);
        let n = w.next_any().expect("a third node");
        assert_eq!(n.op_type, "Relu");

        let manifest: Manifest =
            vec![("conv.weight".into(), vec![1, 1, 3, 3]), ("conv.bias".into(), vec![1])];
        let t = w.finish(&manifest, "unit").unwrap();
        assert_eq!(t["conv.weight"].1.len(), 9);
    }

    /// Each failure mode must name what went wrong: an unconsumed source
    /// tensor, an unvisited node, and a manifest tensor that was never bound.
    #[test]
    fn coverage_failures_name_the_cause() {
        let m = graph();
        let g = read::graph(&m).unwrap();
        let manifest: Manifest =
            vec![("conv.weight".into(), vec![1, 1, 3, 3]), ("conv.bias".into(), vec![1])];
        // Bind the conv; every case below starts from there.
        let bind_conv = |w: &mut Walk| {
            let n = w.next("Conv").unwrap();
            let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
            w.bind("conv.weight", &cw, vec![1, 1, 3, 3]).unwrap();
            w.bind("conv.bias", &cb, vec![1]).unwrap();
        };

        // walk past the Reshape without acknowledging its shape tensor
        let mut w = Walk::new(g).unwrap();
        bind_conv(&mut w);
        w.next_any().unwrap();
        w.next_any().unwrap();
        let err = w.finish(&manifest, "unit").unwrap_err();
        assert!(err.contains("unused source initializers") && err.contains("shape"), "{err}");

        // acknowledge everything but stop before the last node
        let mut w = Walk::new(g).unwrap();
        bind_conv(&mut w);
        let n = w.next_any().unwrap();
        w.ack_structural(n);
        let err = w.finish(&manifest, "unit").unwrap_err();
        assert!(err.contains("never visited"), "{err}");

        // an empty walk is missing the first manifest tensor
        let w = Walk::new(g).unwrap();
        let err = w.finish(&manifest, "unit").unwrap_err();
        assert!(err.contains("missing tensor conv.weight"), "{err}");
    }

    /// The op-sequence assertion is the whole point of walking: a wrong op at
    /// the cursor is an error naming both what was found and what was expected.
    #[test]
    fn an_unexpected_op_fails_by_name() {
        let m = graph();
        let g = read::graph(&m).unwrap();
        let mut w = Walk::new(g).unwrap();
        let err = w.next("Gemm").unwrap_err();
        assert!(err.contains("is `Conv`") && err.contains("expected `Gemm`"), "{err}");
    }

    /// Conv geometry is checked against the ARCHITECTURE, not read from the
    /// file: a graph with the right shapes and the wrong stride must fail.
    #[test]
    fn conv_geometry_is_asserted_not_adopted() {
        let m = graph();
        let g = read::graph(&m).unwrap();
        let n = &g.node[0];
        assert!(check_conv(n, 0, 3, 1, 1).is_ok());
        let err = check_conv(n, 0, 3, 2, 1).unwrap_err();
        assert!(err.contains("strides"), "{err}");
    }

    /// `check_conv2d` is `check_conv`'s general form: it must accept the exact
    /// geometry AND reject an asymmetric stride swap that `check_conv` cannot
    /// even express (a `(2,1)` graph checked as `(1,2)`).
    #[test]
    fn check_conv2d_accepts_asymmetric_geometry_and_rejects_a_swap() {
        let m = graph();
        let g = read::graph(&m).unwrap();
        let n = &g.node[0];
        assert!(check_conv2d(n, 0, 3, 3, 1, 1, 1, 1).is_ok());
        let err = check_conv2d(n, 0, 3, 3, 1, 2, 1, 1).unwrap_err();
        assert!(err.contains("strides"), "{err}");
    }

    /// A 1D dilated `Conv` (CAM++'s `CAMLayer.linear_local`, `k=3, dilation=2,
    /// pad=2`): `check_conv1d` must accept the exact geometry and reject a
    /// dilation mismatch by name, not just any shape mismatch.
    #[test]
    fn check_conv1d_accepts_dilated_geometry_and_rejects_a_dilation_mismatch() {
        let mut g = GraphBuilder::new("walk1d");
        g.input_f32("x", &[1, 4, 10]);
        g.output_f32("y", &[1, 4, 10]);
        g.init_f32("w", &[4, 4, 3], vec![0.0; 48]);
        g.add(
            Node::new("Conv", &["x", "w"], &["y"])
                .name("conv1d")
                .attr_ints("kernel_shape", &[3])
                .attr_ints("strides", &[1])
                .attr_ints("pads", &[2, 2])
                .attr_ints("dilations", &[2]),
        );
        let m = crate::decode_model(&g.finish()).unwrap();
        let gr = read::graph(&m).unwrap();
        let n = &gr.node[0];
        assert!(check_conv1d(n, 0, 3, 1, 2, 2).is_ok());
        let err = check_conv1d(n, 0, 3, 1, 2, 1).unwrap_err();
        assert!(err.contains("dilations"), "{err}");
    }
}
