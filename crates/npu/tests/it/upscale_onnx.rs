// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-ESRGAN → ONNX: what can be checked WITHOUT an NPU.
//!
//! There is no Intel NPU on the machine this was written on, so this gates the
//! GRAPH, and says so rather than implying more: the op set, the node counts
//! implied by the config, the residual scalings, and that the exported bytes
//! re-read as a well-formed model with the declared shapes. Numerical parity
//! against `crates/rrdbnet` belongs in `npu_live.rs`, on hardware.
//!
//! That is still worth having. Every defect this catches — a missing residual
//! scale, a dropped activation, an op OpenVINO cannot compile, a shape that
//! does not survive the round trip — would otherwise be found on the far side
//! of a hardware boundary, by someone without the source in front of them.

use std::collections::HashMap;

use npu::topology::WeightSource;
use npu::upscale_topology::{build_rrdb_graph, RrdbTopo};

/// A weight source that answers with correctly-shaped zeros, so the graph's
/// STRUCTURE is testable with no checkpoint.
struct Zeros(HashMap<String, usize>);

impl Zeros {
    fn for_topo(t: &RrdbTopo) -> Zeros {
        let (f, g) = (t.num_feat as usize, t.num_grow_ch as usize);
        let mut m = HashMap::new();
        let conv = |name: String, cout: usize, cin: usize, m: &mut HashMap<String, usize>| {
            m.insert(format!("{name}.weight"), cout * cin * 9);
            m.insert(format!("{name}.bias"), cout);
        };
        conv("conv_first".into(), f, t.in_channels as usize, &mut m);
        for b in 0..t.num_block as usize {
            for r in 1..=3 {
                for c in 1..=5 {
                    let cin = f + (c - 1) * g;
                    let cout = if c == 5 { f } else { g };
                    conv(format!("body.{b}.rdb{r}.conv{c}"), cout, cin, &mut m);
                }
            }
        }
        conv("conv_body".into(), f, f, &mut m);
        for i in 1..=t.scale.trailing_zeros() as usize {
            conv(format!("conv_up{i}"), f, f, &mut m);
        }
        conv("conv_hr".into(), f, f, &mut m);
        conv("conv_last".into(), t.out_channels as usize, f, &mut m);
        Zeros(m)
    }
}

impl WeightSource for Zeros {
    fn get(&self, name: &str) -> Vec<f32> {
        let n = *self.0.get(name).unwrap_or_else(|| panic!("graph asked for an unknown tensor `{name}`"));
        vec![0.0; n]
    }
}

fn tiny() -> RrdbTopo {
    // Dims that all DIFFER, so a width-for-width swap cannot pass:
    // feat 16, grow 8, blocks 2, image 32.
    RrdbTopo {
        in_channels: 3,
        out_channels: 3,
        num_feat: 16,
        num_grow_ch: 8,
        num_block: 2,
        scale: 4,
        h: 32,
        w: 32,
    }
}

fn build(t: &RrdbTopo) -> onnx::GraphBuilder {
    let mut g = onnx::GraphBuilder::new("rrdb");
    build_rrdb_graph(t, &Zeros::for_topo(t), &mut g);
    g
}

/// Static dims of a value-info entry.
fn dims_of(v: &onnx::onnx::ValueInfoProto) -> Vec<i64> {
    v.r#type
        .as_ref()
        .and_then(|t| t.tensor_type.as_ref())
        .and_then(|tt| tt.shape.as_ref())
        .map(|s| s.dim.iter().map(|d| d.dim_value).collect())
        .unwrap_or_default()
}

fn op_counts(g: &onnx::GraphBuilder) -> HashMap<String, usize> {
    let mut c: HashMap<String, usize> = HashMap::new();
    for n in &g.graph().nodes {
        *c.entry(n.op_type.clone()).or_default() += 1;
    }
    c
}

/// Every op must be one OpenVINO compiles for the NPU. A single exotic op is
/// the difference between a graph that runs on the accelerator and one that
/// silently falls back to CPU.
#[test]
fn the_graph_uses_only_npu_supported_ops() {
    let g = build(&tiny());
    let allowed = ["Conv", "LeakyRelu", "Concat", "Resize", "Add", "Mul"];
    let counts = op_counts(&g);
    for op in counts.keys() {
        assert!(allowed.contains(&op.as_str()), "unexpected op `{op}` (allowed: {allowed:?})");
    }
    eprintln!("rrdb ops: {counts:?}");
}

/// The counts follow from the config, so a dropped block or a lost activation
/// is arithmetic rather than opinion.
#[test]
fn node_counts_match_the_architecture() {
    let t = tiny();
    let counts = op_counts(&build(&t));
    let ups = t.scale.trailing_zeros() as usize;
    let blocks = t.num_block as usize;

    // 15 convs per RRDB (3 dense blocks x 5), plus conv_first, conv_body,
    // one per upsample stage, conv_hr and conv_last.
    let want_conv = blocks * 15 + 2 + ups + 2;
    assert_eq!(counts["Conv"], want_conv, "Conv count");

    // 4 activations per dense block (conv5 has none) + one per upsample + conv_hr.
    let want_lrelu = blocks * 3 * 4 + ups + 1;
    assert_eq!(counts["LeakyRelu"], want_lrelu, "LeakyRelu count");

    // 4 growing concats per dense block.
    assert_eq!(counts["Concat"], blocks * 3 * 4, "Concat count");
    assert_eq!(counts["Resize"], ups, "one nearest-2x per doubling");
}

/// THE ONE THAT MATTERS. Each dense block and each RRDB scales its residual by
/// 0.2; emitting a bare `Add` gives a graph that runs, has the right shape, and
/// is wrong. There must be exactly one `Mul` per residual, and the trunk join
/// (`fea + conv_body`) must NOT have one.
#[test]
fn residual_scalings_are_present_and_not_over_applied() {
    let t = tiny();
    let counts = op_counts(&build(&t));
    let blocks = t.num_block as usize;
    // 3 dense residuals + 1 RRDB residual per block.
    let want = blocks * 4;
    assert_eq!(counts["Mul"], want, "one scaled residual per dense block and per RRDB");
    // Every scaled residual is an Add too, plus the unscaled trunk join.
    assert_eq!(counts["Add"], want + 1, "Adds = scaled residuals + the trunk join");
}

/// The exported bytes must re-read as a model with the shapes the caller was
/// promised — the round trip is where a builder bug shows up as an unreadable
/// or mis-shaped graph.
#[test]
fn the_export_round_trips_with_the_declared_shapes() {
    let t = tiny();
    let bytes = build(&t).finish();
    assert!(bytes.len() > 1024, "suspiciously small export: {} bytes", bytes.len());
    let m = onnx::decode_model(&bytes).expect("valid ONNX ModelProto");
    let g = m.graph.expect("model has a graph");
    assert!(g.input.iter().any(|v| v.name == "image"), "missing image input");
    assert!(g.output.iter().any(|v| v.name == "out"), "missing out output");
    assert_eq!(dims_of(&g.output[0]), vec![1, 3, 128, 128], "x4 on both spatial dims");
    assert_eq!(dims_of(&g.input[0]), vec![1, 3, 32, 32]);
}

/// The x2 variant differs only in how many upsample stages it carries, and must
/// build from the same code path — the config is derived from a checkpoint, so
/// a second variant is a real case, not a hypothetical.
#[test]
fn the_x2_variant_builds_with_one_upsample_stage() {
    let t = RrdbTopo { scale: 2, ..tiny() };
    let counts = op_counts(&build(&t));
    assert_eq!(counts["Resize"], 1);
    let bytes = build(&t).finish();
    let g = onnx::decode_model(&bytes).expect("valid ONNX").graph.expect("graph");
    assert_eq!(dims_of(&g.output[0]), vec![1, 3, 64, 64], "x2 on both spatial dims");
}
