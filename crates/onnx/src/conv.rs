// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! ONNX `ConvTranspose` (1-D / 2-D) node construction + a pure-Rust reference
//! evaluator for the 1-D case.
//!
//! `ConvTranspose` is the transposed (a.k.a. "deconv" / fractionally-strided)
//! convolution used by the codec's SEANet upsampling decoder. The vendored ONNX
//! proto already expresses it (it is just an `op_type` string plus attributes —
//! see [`crate::graph::Node`]); this module adds an ergonomic constructor and a
//! reference implementation so exporters can self-check numerics on CPU without
//! an external runtime.
//!
//! **Weight layout.** ONNX `ConvTranspose` weights are `[C_in, C_out/group, k…]`
//! — *identical* to brain's `audio::conv` transposed-conv weight
//! (`w[(ci·Cout_g + co_local)·K + kw]`), so no transpose is needed on export.
//!
//! **Output length (1-D).** With input length `L`, the spatial output is
//! `L_out = stride·(L−1) + dilation·(K−1) + 1 − pad_begin − pad_end +
//! output_padding`. The codec's *causal* transposed conv keeps the first
//! `L·stride` samples, i.e. `pads = [0, K − stride]`, `output_padding = 0`.

use crate::graph::Node;

/// Parameters for a 1-D `ConvTranspose` (the codec's case). `groups`, `dilation`
/// and `output_padding` default-friendly for the common stride-upsample.
#[derive(Clone, Copy, Debug)]
pub struct ConvTranspose1d {
    pub cin: usize,
    pub cout: usize,
    pub l: usize,
    pub k: usize,
    pub stride: usize,
    pub pad_begin: usize,
    pub pad_end: usize,
    pub dilation: usize,
    pub groups: usize,
    pub output_padding: usize,
}

impl ConvTranspose1d {
    /// Spatial length of the produced output.
    pub fn l_out(&self) -> usize {
        self.stride * (self.l - 1) + self.dilation * (self.k - 1) + 1 + self.output_padding
            - self.pad_begin
            - self.pad_end
    }
}

/// Build a `ConvTranspose` [`Node`]. `inputs` is `[x, w]` or `[x, w, bias]`;
/// `kernel_shape`/`strides`/`pads`/`dilations`/`output_padding` follow ONNX
/// conventions (1-D → length-1 spatial vectors and `pads = [begin, end]`; 2-D →
/// length-2 and `pads = [t, l, b, r]`). Works for both 1-D and 2-D — the helper
/// is rank-agnostic, validating nothing beyond what ONNX runtimes do.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose_node(
    name: &str,
    inputs: &[&str],
    output: &str,
    kernel_shape: &[i64],
    strides: &[i64],
    pads: &[i64],
    dilations: &[i64],
    output_padding: &[i64],
    group: i64,
) -> Node {
    let mut n = Node::new("ConvTranspose", inputs, &[output])
        .name(name)
        .attr_ints("kernel_shape", kernel_shape)
        .attr_ints("strides", strides)
        .attr_ints("pads", pads)
        .attr_ints("dilations", dilations)
        .attr_int("group", group);
    // `output_padding` defaults to 0; only emit when non-trivial to keep the
    // graph minimal (some runtimes are picky about all-zero vectors).
    if output_padding.iter().any(|&v| v != 0) {
        n = n.attr_ints("output_padding", output_padding);
    }
    n
}

/// Reference forward for a 1-D `ConvTranspose` implementing exact ONNX
/// semantics (`opset-13`). `x` is `[C_in, L]` row-major, `w` is
/// `[C_in, C_out/group, K]`, optional `bias` is `[C_out]`. Returns
/// `[C_out, L_out]`.
///
/// `out[co, lo] = Σ_{ci∈group(co), kw}  x[ci, li] · w[ci, co_local, kw]`
/// where `lo = li·stride + kw·dilation − pad_begin` selects the contributing
/// input position `li` (those landing outside `[0, L_out)` are dropped — the
/// `pads`/`output_padding` crop).
pub fn conv_transpose1d_ref(c: &ConvTranspose1d, x: &[f32], w: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let lo_n = c.l_out();
    let cin_g = c.cin / c.groups;
    let cout_g = c.cout / c.groups;
    assert_eq!(x.len(), c.cin * c.l, "x must be [cin, l]");
    assert_eq!(w.len(), c.cin * cout_g * c.k, "w must be [cin, cout/group, k]");
    let mut y = vec![0.0f32; c.cout * lo_n];
    for co in 0..c.cout {
        let g = co / cout_g;
        let co_local = co - g * cout_g;
        let b = bias.map(|bb| bb[co]).unwrap_or(0.0);
        for lo in 0..lo_n {
            let mut acc = b;
            // num = lo + pad_begin must equal li*stride + kw*dilation.
            let num = lo + c.pad_begin;
            for kw in 0..c.k {
                let sub = kw * c.dilation;
                if num >= sub && (num - sub).is_multiple_of(c.stride) {
                    let li = (num - sub) / c.stride;
                    if li < c.l {
                        for cl in 0..cin_g {
                            let ci = g * cin_g + cl;
                            let xi = ci * c.l + li;
                            let wi = (ci * cout_g + co_local) * c.k + kw;
                            acc += x[xi] * w[wi];
                        }
                    }
                }
            }
            y[co * lo_n + lo] = acc;
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{decode_model, GraphBuilder};

    /// Hand-computed tiny example: Cin=1, Cout=1, K=2, stride=2, no pad/bias.
    /// x = [1, 2, 3] (L=3), w = [10, 20] (k0,k1).
    /// Full transposed conv (L_out = 2*(3-1)+1*(2-1)+1 = 6):
    ///   contributions: for each input li, places w[0] at lo=2li, w[1] at lo=2li+1
    ///   li0: y[0]+=1*10, y[1]+=1*20
    ///   li1: y[2]+=2*10, y[3]+=2*20
    ///   li2: y[4]+=3*10, y[5]+=3*20
    ///   => [10,20,20,40,30,60]
    #[test]
    fn convtr1d_reference_matches_hand_computation() {
        let c = ConvTranspose1d {
            cin: 1,
            cout: 1,
            l: 3,
            k: 2,
            stride: 2,
            pad_begin: 0,
            pad_end: 0,
            dilation: 1,
            groups: 1,
            output_padding: 0,
        };
        assert_eq!(c.l_out(), 6);
        let y = conv_transpose1d_ref(&c, &[1.0, 2.0, 3.0], &[10.0, 20.0], None);
        assert_eq!(y, vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    /// Causal crop (pads = [0, K-stride]) keeps the first L*stride samples — the
    /// codec's upsample convention. K=4, stride=2 -> crop 2 off the right.
    #[test]
    fn convtr1d_causal_crop_keeps_first_l_stride() {
        let c = ConvTranspose1d {
            cin: 1,
            cout: 1,
            l: 3,
            k: 4,
            stride: 2,
            pad_begin: 0,
            pad_end: 2, // K - stride
            dilation: 1,
            groups: 1,
            output_padding: 0,
        };
        assert_eq!(c.l_out(), 2 * 3); // L*stride
        // full (uncropped) would be stride*(L-1)+K = 8; we keep first 6.
        let full = ConvTranspose1d { pad_end: 0, ..c };
        let x = [1.0, 2.0, 3.0];
        let w = [1.0, 2.0, 3.0, 4.0];
        let y_full = conv_transpose1d_ref(&full, &x, &w, None);
        let y_crop = conv_transpose1d_ref(&c, &x, &w, None);
        assert_eq!(y_crop.len(), 6);
        assert_eq!(&y_crop[..], &y_full[..6]);
    }

    /// Grouped + bias smoke (groups=2): two independent channels.
    #[test]
    fn convtr1d_grouped_with_bias() {
        let c = ConvTranspose1d {
            cin: 2,
            cout: 2,
            l: 2,
            k: 2,
            stride: 1,
            pad_begin: 0,
            pad_end: 0,
            dilation: 1,
            groups: 2,
            output_padding: 0,
        };
        // L_out = 1*(2-1)+1*(2-1)+1 = 3
        assert_eq!(c.l_out(), 3);
        let x = [1.0, 1.0, /*ch0*/ 2.0, 2.0 /*ch1*/];
        // w[cin, cout/g=1, k=2]: ch0 kernel [1,0], ch1 kernel [0,1]
        let w = [1.0, 0.0, 0.0, 1.0];
        let y = conv_transpose1d_ref(&c, &x, &w, Some(&[10.0, 20.0]));
        // ch0: place w[0]=1 at lo, w[1]=0 -> [1,1,0]+bias10 = [11,11,10]
        assert_eq!(&y[0..3], &[11.0, 11.0, 10.0]);
        // ch1: w[0]=0,w[1]=1 -> [0,2,2]+bias20 = [20,22,22]
        assert_eq!(&y[3..6], &[20.0, 22.0, 22.0]);
    }

    /// The emitted node is structurally a valid ConvTranspose with the expected
    /// attributes, and a graph containing it round-trips through the serializer.
    #[test]
    fn convtr_node_structure_and_graph_roundtrip() {
        let node = conv_transpose_node(
            "up0",
            &["x", "w", "b"],
            "y",
            &[4],       // kernel_shape
            &[2],       // strides
            &[0, 2],    // pads (begin, end)
            &[1],       // dilations
            &[0],       // output_padding (omitted, all-zero)
            1,          // group
        );
        assert_eq!(node.op_type, "ConvTranspose");
        assert_eq!(node.inputs, vec!["x", "w", "b"]);
        let names: Vec<&str> = node.attrs.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"kernel_shape"));
        assert!(names.contains(&"strides"));
        assert!(names.contains(&"pads"));
        assert!(names.contains(&"group"));
        assert!(!names.contains(&"output_padding"), "all-zero output_padding omitted");

        let mut g = GraphBuilder::new("convtr");
        g.input_f32("x", &[1, 1, 3]);
        g.init_f32("w", &[1, 1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        g.init_f32("b", &[1], vec![0.0]);
        g.output_f32("y", &[1, 1, 6]);
        g.add(node);
        let bytes = g.finish();
        let model = decode_model(&bytes).expect("serialized ConvTranspose graph must decode");
        let gp = model.graph.unwrap();
        assert_eq!(gp.node.len(), 1);
        assert_eq!(gp.node[0].op_type, "ConvTranspose");
    }
}
