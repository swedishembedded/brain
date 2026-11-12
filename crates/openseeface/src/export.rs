// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The network as a static-shape ONNX graph, for a whole-graph compiler
//! (OpenVINO for the Intel NPU and GPU).
//!
//! Built from brain's imported tensors and the [`NetConfig`], in exactly the
//! op order [`crate::model::Net`] runs - not a copy of the released file.
//! The release cannot be handed to the NPU compiler as shipped: its batch
//! dimension is symbolic and its decoder computes each resize target at run
//! time (`Shape` -> `Slice` -> `Concat`), which the NPU compiler rejects.
//! Here every shape is fixed - batch 1, the config's input size - and every
//! resize names its output size as a constant.
//!
//! Op choice is the standard opset-14 vocabulary every runtime implements:
//! `Conv` (with `group`), `Relu`, `Clip` (ReLU6), `HardSwish`,
//! `HardSigmoid` (`alpha = 1/6, beta = 0.5`: torch's `relu6(x + 3) / 6`, not
//! ONNX's default slope), `GlobalAveragePool`, `Mul`, `Add`, `Resize`
//! (linear, `align_corners`), `Concat`, `MaxPool`.

use onnx::walk::Tensors;
use onnx::{GraphBuilder, Node};
use vision::Act;

use crate::config::{Kind, NetConfig};

/// The opset the graph is written against (`HardSwish` needs 14).
pub const OPSET: i64 = 14;
const IR_VERSION: i64 = 8;

/// Output names: the map, and the detector's peak pool.
pub const OUTPUT: &str = "output";
pub const POOLED: &str = "pooled";

struct Emit<'t> {
    g: GraphBuilder,
    t: &'t Tensors,
    n: usize,
}

#[derive(Clone)]
struct Feat {
    name: String,
    c: u32,
    h: u32,
    w: u32,
}

impl Emit<'_> {
    fn fresh(&mut self, what: &str) -> String {
        self.n += 1;
        format!("{what}_{}", self.n)
    }

    fn weight(&mut self, name: &str) -> Result<(), String> {
        let (shape, data) = self.t.get(name).ok_or_else(|| format!("export: tensor {name} is not in the imported weights"))?;
        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        self.g.init_f32(name, &dims, data.clone());
        Ok(())
    }

    fn scalar(&mut self, v: f32) -> String {
        let name = self.fresh("const");
        self.g.init_f32(&name, &[], vec![v]);
        name
    }

    fn act(&mut self, x: Feat, act: Act) -> Result<Feat, String> {
        let out = self.fresh("act");
        self.act_into(x, act, out)
    }

    /// [`Self::act`] writing a tensor named `out`.
    fn act_into(&mut self, x: Feat, act: Act, out: String) -> Result<Feat, String> {
        let node = match act {
            Act::None => return Ok(x),
            Act::Relu => Node::new("Relu", &[&x.name], &[&out]),
            Act::Relu6 => {
                let (lo, hi) = (self.scalar(0.0), self.scalar(6.0));
                Node::new("Clip", &[&x.name, &lo, &hi], &[&out])
            }
            Act::HardSwish => Node::new("HardSwish", &[&x.name], &[&out]),
            Act::HardSigmoid => Node::new("HardSigmoid", &[&x.name], &[&out]).attr_float("alpha", 1.0 / 6.0).attr_float("beta", 0.5),
            other => return Err(format!("export: no ONNX spelling for {other:?} here")),
        };
        self.g.add(node);
        Ok(Feat { name: out, ..x })
    }

    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, prefix: &str, x: &Feat, cout: u32, k: u32, stride: u32, groups: u32, act: Act) -> Result<Feat, String> {
        let (w, b) = (format!("{prefix}.conv.weight"), format!("{prefix}.conv.bias"));
        self.weight(&w)?;
        self.weight(&b)?;
        let out = self.fresh("conv");
        let p = (k / 2) as i64;
        self.g.add(
            Node::new("Conv", &[&x.name, &w, &b], &[&out])
                .attr_ints("kernel_shape", &[k as i64, k as i64])
                .attr_ints("strides", &[stride as i64, stride as i64])
                .attr_ints("pads", &[p, p, p, p])
                .attr_ints("dilations", &[1, 1])
                .attr_int("group", groups as i64),
        );
        let h = (x.h + 2 * (k / 2) - k) / stride + 1;
        let wd = (x.w + 2 * (k / 2) - k) / stride + 1;
        self.act(Feat { name: out, c: cout, h, w: wd }, act)
    }

    fn binary(&mut self, op: &str, a: &Feat, b: &str) -> Feat {
        let out = self.fresh(&op.to_lowercase());
        self.g.add(Node::new(op, &[&a.name, b], &[&out]));
        Feat { name: out, ..a.clone() }
    }

    fn squeeze_excite(&mut self, prefix: &str, x: &Feat, reduced: u32) -> Result<Feat, String> {
        let pooled = self.fresh("gap");
        self.g.add(Node::new("GlobalAveragePool", &[&x.name], &[&pooled]));
        let p = Feat { name: pooled, c: x.c, h: 1, w: 1 };
        let r = self.conv(&format!("{prefix}.reduce"), &p, reduced, 1, 1, 1, Act::Relu)?;
        let gate = self.conv(&format!("{prefix}.expand"), &r, x.c, 1, 1, 1, Act::HardSigmoid)?;
        Ok(self.binary("Mul", x, &gate.name))
    }
}

/// The ONNX bytes of `cfg`'s network over imported tensors `t`.
pub fn export(t: &Tensors, cfg: &NetConfig) -> Result<Vec<u8>, String> {
    let n = cfg.input;
    let mut e = Emit { g: GraphBuilder::new(cfg.file), t, n: 0 };
    e.g.input_f32("input", &[1, 3, n as i64, n as i64]);
    let x = Feat { name: "input".into(), c: 3, h: n, w: n };
    let mut x = e.conv("stem", &x, cfg.stem, 3, 2, 1, cfg.stem_act)?;
    let mut stage_out = Vec::with_capacity(cfg.stages.len());
    for (s, stage) in cfg.stages.iter().enumerate() {
        for (b, blk) in stage.iter().enumerate() {
            let p = format!("blocks.{s}.{b}");
            let input = x.clone();
            let y = match blk.kind {
                Kind::Ds => {
                    let mut y = e.conv(&format!("{p}.dw"), &x, x.c, blk.k, blk.stride, x.c, blk.act)?;
                    if blk.se > 0 {
                        y = e.squeeze_excite(&format!("{p}.se"), &y, blk.se)?;
                    }
                    e.conv(&format!("{p}.pw"), &y, blk.out, 1, 1, 1, Act::None)?
                }
                Kind::Ir => {
                    let y = e.conv(&format!("{p}.pw"), &x, blk.mid, 1, 1, 1, blk.act)?;
                    let mut y = e.conv(&format!("{p}.dw"), &y, blk.mid, blk.k, blk.stride, blk.mid, blk.act)?;
                    if blk.se > 0 {
                        y = e.squeeze_excite(&format!("{p}.se"), &y, blk.se)?;
                    }
                    e.conv(&format!("{p}.pwl"), &y, blk.out, 1, 1, 1, Act::None)?
                }
                Kind::Cn => e.conv(&format!("{p}.conv"), &x, blk.out, 1, 1, 1, blk.act)?,
            };
            x = if crate::model::residual(blk, input.c) { e.binary("Add", &y, &input.name) } else { y };
        }
        stage_out.push(x.clone());
    }
    for (k, up) in cfg.ups.iter().enumerate() {
        let skip = stage_out.get(up.skip).ok_or_else(|| format!("export: up{} skips stage {} of {}", k + 1, up.skip, stage_out.len()))?.clone();
        let sizes = e.fresh("sizes");
        e.g.init_i64(&sizes, &[4], vec![1, x.c as i64, skip.h as i64, skip.w as i64]);
        let resized = e.fresh("resize");
        e.g.add(
            Node::new("Resize", &[&x.name, "", "", &sizes], &[&resized])
                .attr_str("mode", "linear")
                .attr_str("coordinate_transformation_mode", "align_corners"),
        );
        let cat = e.fresh("concat");
        e.g.add(Node::new("Concat", &[&skip.name, &resized], &[&cat]).attr_int("axis", 1));
        let c = Feat { name: cat, c: skip.c + x.c, h: skip.h, w: skip.w };
        let y = e.conv(&format!("up{}.dw", k + 1), &c, c.c, 3, 1, c.c, up.dw_act)?;
        x = e.conv(&format!("up{}.pw", k + 1), &y, up.out, 1, 1, 1, Act::Relu6)?;
    }
    let h = cfg.head;
    let mid = e.conv("head.mid", &x, h.mid, 3, 1, h.groups, h.mid_act)?;
    // the head's last node writes `output` itself: a runtime may fold an
    // Identity away and report the producer's name instead (OpenVINO does)
    let pre = e.conv("head.out", &mid, h.out, 1, 1, h.out_groups, Act::None)?;
    if h.out_act == Act::None {
        return Err("export: the head ends in an activation in every released graph".into());
    }
    let out = e.act_into(pre, h.out_act, OUTPUT.to_string())?;
    let dims = [1, out.c as i64, out.h as i64, out.w as i64];
    e.g.output_f32(OUTPUT, &dims);
    if cfg.peak_pool {
        e.g.add(
            Node::new("MaxPool", &[&out.name], &[POOLED])
                .attr_ints("kernel_shape", &[3, 3])
                .attr_ints("strides", &[1, 1])
                .attr_ints("pads", &[1, 1, 1, 1]),
        );
        e.g.output_f32(POOLED, &dims);
    }
    Ok(e.g.finish_with(OPSET, IR_VERSION))
}
