// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-ESRGAN (`RRDBNet`) → Intel NPU, as an ONNX graph.
//!
//! The whole net is `Conv` + `LeakyRelu` + `Concat` + nearest `Resize` + `Add`
//! — five ops, every one of them in the OpenVINO NPU's supported set — so this
//! is the most portable model in the imaging stack and a good first NPU
//! topology for it.
//!
//! **The two `* 0.2` residual scalings are the architecture.** Each dense block
//! and each RRDB returns `x + 0.2*f(x)`. Emitting them as a bare `Add` produces
//! a graph that runs, is the right shape, and is WRONG — the same failure mode
//! `crates/rrdbnet`'s parity gate exists to catch. They are emitted as an explicit `Mul` by a scalar
//! initializer, and `residual_scalings_are_present` counts them.
//!
//! Validation, honestly: there is no NPU on the machine this was written on, so
//! what is gated here is the GRAPH — node counts, op set, the residual scalars,
//! and that the exported bytes re-read as a well-formed ONNX model with the
//! declared input/output shapes. Numerical parity against `crates/rrdbnet`
//! needs `npu_live.rs` on a box with the hardware.

use onnx::GraphBuilder;

use crate::topo::TopoBase;
use crate::topology::WeightSource;

/// The shape numbers the graph needs, mirroring `rrdbnet::RrdbConfig` without
/// depending on that crate (the `npu` crate stays free of model crates, the
/// same way every other `*_topology` here does).
#[derive(Clone, Copy, Debug)]
pub struct RrdbTopo {
    pub in_channels: u32,
    pub out_channels: u32,
    pub num_feat: u32,
    pub num_grow_ch: u32,
    pub num_block: u32,
    /// 4 for x4plus, 2 for x2plus — one nearest-2x `Resize` per doubling.
    pub scale: u32,
    pub h: u32,
    pub w: u32,
}

/// `LeakyRelu(negative_slope=0.2)` and the residual weight — both fixed by the
/// architecture, both named once.
const LRELU_SLOPE: f32 = 0.2;
const RESIDUAL_SCALE: f32 = 0.2;

struct Rrdb<'a> {
    b: TopoBase<'a>,
}

impl<'a> Rrdb<'a> {
    fn new(g: &'a mut GraphBuilder) -> Rrdb<'a> {
        Rrdb { b: TopoBase::new(g) }
    }

    /// A 3x3 same-padded conv with bias, from `{prefix}.weight` / `{prefix}.bias`.
    fn conv(&mut self, prefix: &str, cout: u32, cin: u32, w: &dyn WeightSource, x: &str) -> String {
        let (wn, bn) = (format!("{prefix}.weight"), format!("{prefix}.bias"));
        self.b.f32(&wn, &[cout as i64, cin as i64, 3, 3], w.get(&wn));
        self.b.f32(&bn, &[cout as i64], w.get(&bn));
        let out = self.b.tmp("conv");
        let mut n = onnx::Node::new("Conv", &[x, &wn, &bn], &[&out]);
        n = n.attr_ints("kernel_shape", &[3, 3]).attr_ints("pads", &[1, 1, 1, 1]).attr_ints("strides", &[1, 1]);
        self.b.g.add(n);
        out
    }

    fn lrelu(&mut self, x: &str) -> String {
        let out = self.b.tmp("lrelu");
        self.b.g.add(onnx::Node::new("LeakyRelu", &[x], &[&out]).attr_float("alpha", LRELU_SLOPE));
        out
    }

    /// `x + RESIDUAL_SCALE * fx`, with the scale an explicit initializer so the
    /// graph carries the architecture rather than implying it.
    fn residual(&mut self, x: &str, fx: &str) -> String {
        let k = self.b.tmp("res_scale");
        self.b.f32(&k, &[1], vec![RESIDUAL_SCALE]);
        let scaled = self.b.mul_t(fx, &k);
        self.b.add_t(x, &scaled)
    }

    /// `ResidualDenseBlock_5C`: five convs over a growing concat, then the
    /// scaled residual. conv5 has NO activation — it feeds the residual.
    fn dense_block(&mut self, prefix: &str, t: &RrdbTopo, w: &dyn WeightSource, x: &str) -> String {
        let (f, g) = (t.num_feat, t.num_grow_ch);
        let mut acc = x.to_string();
        let mut acc_c = f;
        let mut last = x.to_string();
        for c in 1..=5u32 {
            let cout = if c == 5 { f } else { g };
            let y = self.conv(&format!("{prefix}.conv{c}"), cout, acc_c, w, &acc);
            last = if c == 5 { y } else { self.lrelu(&y) };
            if c < 5 {
                acc = self.b.concat2(&acc, &last, 1);
                acc_c += cout;
            }
        }
        self.residual(x, &last)
    }

    fn rrdb(&mut self, prefix: &str, t: &RrdbTopo, w: &dyn WeightSource, x: &str) -> String {
        let mut cur = x.to_string();
        for r in 1..=3u32 {
            cur = self.dense_block(&format!("{prefix}.rdb{r}"), t, w, &cur);
        }
        self.residual(x, &cur)
    }

    /// Nearest 2x upsample. `Resize` with an explicit `scales` initializer and
    /// empty `roi`/`sizes`, which is the form OpenVINO accepts.
    fn upsample2(&mut self, x: &str) -> String {
        let roi = self.b.tmp("roi");
        self.b.f32(&roi, &[0], vec![]);
        let scales = self.b.tmp("scales");
        self.b.f32(&scales, &[4], vec![1.0, 1.0, 2.0, 2.0]);
        let out = self.b.tmp("resize");
        self.b.g.add(
            onnx::Node::new("Resize", &[x, &roi, &scales], &[&out])
                .attr_str("mode", "nearest")
                .attr_str("nearest_mode", "floor")
                .attr_str("coordinate_transformation_mode", "asymmetric"),
        );
        out
    }
}

/// Build the whole RRDBNet forward into `g`.
pub fn build_rrdb_graph(t: &RrdbTopo, w: &dyn WeightSource, g: &mut GraphBuilder) {
    g.input_f32("image", &[1, t.in_channels as i64, t.h as i64, t.w as i64]);
    let s = t.scale as i64;
    g.output_f32("out", &[1, t.out_channels as i64, t.h as i64 * s, t.w as i64 * s]);

    let mut m = Rrdb::new(g);
    let fea = m.conv("conv_first", t.num_feat, t.in_channels, w, "image");

    let mut trunk = fea.clone();
    for i in 0..t.num_block {
        trunk = m.rrdb(&format!("body.{i}"), t, w, &trunk);
    }
    let body = m.conv("conv_body", t.num_feat, t.num_feat, w, &trunk);
    let mut cur = m.b.add_t(&fea, &body);

    for i in 1..=t.scale.trailing_zeros() {
        let up = m.upsample2(&cur);
        let c = m.conv(&format!("conv_up{i}"), t.num_feat, t.num_feat, w, &up);
        cur = m.lrelu(&c);
    }
    let hr = m.conv("conv_hr", t.num_feat, t.num_feat, w, &cur);
    let hr = m.lrelu(&hr);

    // The final conv writes the graph output directly.
    let (wn, bn) = ("conv_last.weight".to_string(), "conv_last.bias".to_string());
    m.b.f32(&wn, &[t.out_channels as i64, t.num_feat as i64, 3, 3], w.get(&wn));
    m.b.f32(&bn, &[t.out_channels as i64], w.get(&bn));
    m.b.g.add(
        onnx::Node::new("Conv", &[&hr, &wn, &bn], &["out"])
            .attr_ints("kernel_shape", &[3, 3])
            .attr_ints("pads", &[1, 1, 1, 1])
            .attr_ints("strides", &[1, 1]),
    );
}
