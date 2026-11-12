// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Import from the released ONNX graphs.
//!
//! The release was exported with its BatchNorms folded into the convolutions
//! (`ConvBnFusion_*` initializers) and then run through onnxruntime's graph
//! optimizer, which fused ReLU and clip activations into `FusedConv` nodes
//! and left hard swish and hard sigmoid as `Add`/`Clip`/`Div`(/`Mul`) chains.
//! The walk follows that exact op sequence through the [`NetConfig`]: every
//! conv's kernel, stride, padding and groups, every fused activation, every
//! constant of every hard swish / hard sigmoid / clip (3, 0, 6, 6), every
//! residual add and every decoder resize's mode are checked against the
//! config, and every weight is bound to its brain name. `onnx::walk`'s
//! two-way coverage then refuses a missing tensor, an unused initializer or
//! an unvisited node.

use std::collections::HashMap;
use std::path::Path;

use onnx::onnx::{GraphProto, NodeProto};
use onnx::read::{self, OnnxTensor};
use onnx::walk::{check_conv_grouped, Manifest, Tensors, Walk};
use vision::Act;

use crate::config::{Kind, NetConfig};

/// Read and import one released graph.
pub fn import(path: &Path, cfg: &NetConfig) -> Result<Tensors, String> {
    let m = read::read_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let g = read::graph(&m)?;
    walk(g, cfg).map_err(|e| format!("{}: {e}", path.display()))
}

struct Importer<'g> {
    w: Walk<'g>,
    consts: HashMap<String, OnnxTensor>,
    manifest: Manifest,
}

impl<'g> Importer<'g> {
    fn constant(&self, n: &NodeProto, input: usize, want: f32) -> Result<(), String> {
        let name = n.input.get(input).ok_or_else(|| format!("node {:?} ({}) has no input {input}", n.name, n.op_type))?;
        let t = self.consts.get(name).ok_or_else(|| format!("node {:?} ({}) input {input} is not a constant", n.name, n.op_type))?;
        if t.data.len() != 1 || t.data[0] != want {
            return Err(format!("node {:?} ({}) input {input} is {:?}, expected {want}", n.name, n.op_type, t.data));
        }
        Ok(())
    }

    fn structural(&mut self, op: &str) -> Result<&'g NodeProto, String> {
        let n = self.w.next(op)?;
        self.w.ack_structural(n);
        Ok(n)
    }

    /// `x + 3 -> clip [0, 6] -> / 6`, the hard sigmoid the release spells out.
    fn hard_sigmoid_chain(&mut self) -> Result<(), String> {
        let add = self.structural("Add")?;
        self.constant(add, 1, 3.0)?;
        let clip = self.structural("Clip")?;
        self.constant(clip, 1, 0.0)?;
        self.constant(clip, 2, 6.0)?;
        let div = self.structural("Div")?;
        self.constant(div, 1, 6.0)
    }

    /// One conv node and the activation it carries.
    fn conv(&mut self, prefix: &str, cin: u32, cout: u32, k: u32, stride: u32, groups: u32, act: Act) -> Result<(), String> {
        let at = self.w.at();
        let n = match self.w.peek() {
            Some("Conv") => self.w.next("Conv")?,
            Some("FusedConv") => self.w.next("FusedConv")?,
            other => return Err(format!("node {at} is {other:?}, expected the conv `{prefix}`")),
        };
        check_conv_grouped(n, at, k as i64, stride as i64, (k / 2) as i64, groups as i64)?;
        let fused = read::attr_str(n, "activation", "");
        let carried = match (n.op_type.as_str(), fused.as_str()) {
            ("FusedConv", "Relu") => Act::Relu,
            ("FusedConv", "Clip") => {
                let p = read::attr(n, "activation_params").map(|a| a.floats.clone()).unwrap_or_default();
                if p != [0.0, 6.0] {
                    return Err(format!("conv `{prefix}` (node {at}) clips to {p:?}, expected [0, 6]"));
                }
                Act::Relu6
            }
            ("Conv", "") => Act::None,
            (op, f) => return Err(format!("conv `{prefix}` (node {at}) is a {op} with activation {f:?}")),
        };
        let shape = vec![cout as usize, (cin / groups) as usize, k as usize, k as usize];
        let (wname, bname) = (format!("{prefix}.conv.weight"), format!("{prefix}.conv.bias"));
        let src_w = n.input.get(1).ok_or_else(|| format!("conv `{prefix}` has no weight"))?;
        let src_b = n.input.get(2).ok_or_else(|| format!("conv `{prefix}` has no bias"))?;
        self.w.bind(&wname, src_w, shape.clone())?;
        self.w.bind(&bname, src_b, vec![cout as usize])?;
        self.manifest.push((wname, shape));
        self.manifest.push((bname, vec![cout as usize]));
        match (carried, act) {
            (a, b) if a == b => Ok(()),
            (Act::None, Act::HardSwish) => {
                self.hard_sigmoid_chain()?;
                self.structural("Mul").map(|_| ())
            }
            (a, b) => Err(format!("conv `{prefix}` (node {at}) carries {a:?}, the config says {b:?}")),
        }
    }

    fn squeeze_excite(&mut self, prefix: &str, c: u32, reduced: u32) -> Result<(), String> {
        self.structural("GlobalAveragePool")?;
        self.conv(&format!("{prefix}.reduce"), c, reduced, 1, 1, 1, Act::Relu)?;
        self.conv(&format!("{prefix}.expand"), reduced, c, 1, 1, 1, Act::None)?;
        self.hard_sigmoid_chain()?;
        self.structural("Mul").map(|_| ())
    }

    fn resize_to_skip(&mut self) -> Result<(), String> {
        self.structural("Shape")?;
        self.structural("Slice")?;
        self.structural("Concat")?;
        let at = self.w.at();
        let r = self.structural("Resize")?;
        let (mode, ct) = (read::attr_str(r, "mode", "nearest"), read::attr_str(r, "coordinate_transformation_mode", "half_pixel"));
        if mode != "linear" || ct != "align_corners" {
            return Err(format!("Resize at node {at} is {mode}/{ct}; the decoder upsamples linear/align_corners"));
        }
        let cat = self.structural("Concat")?;
        if read::attr_int(cat, "axis", 0) != 1 {
            return Err(format!("the decoder concat after node {at} is not over channels"));
        }
        Ok(())
    }
}

fn walk(g: &GraphProto, cfg: &NetConfig) -> Result<Tensors, String> {
    let mut im = Importer { w: Walk::new(g)?, consts: read::initializers(g)?, manifest: Vec::new() };
    im.conv("stem", 3, cfg.stem, 3, 2, 1, cfg.stem_act)?;
    let mut c = cfg.stem;
    let mut stage_out = Vec::with_capacity(cfg.stages.len());
    for (s, stage) in cfg.stages.iter().enumerate() {
        for (b, blk) in stage.iter().enumerate() {
            let p = format!("blocks.{s}.{b}");
            match blk.kind {
                Kind::Ds => {
                    im.conv(&format!("{p}.dw"), c, c, blk.k, blk.stride, c, blk.act)?;
                    if blk.se > 0 {
                        im.squeeze_excite(&format!("{p}.se"), c, blk.se)?;
                    }
                    im.conv(&format!("{p}.pw"), c, blk.out, 1, 1, 1, Act::None)?;
                }
                Kind::Ir => {
                    im.conv(&format!("{p}.pw"), c, blk.mid, 1, 1, 1, blk.act)?;
                    im.conv(&format!("{p}.dw"), blk.mid, blk.mid, blk.k, blk.stride, blk.mid, blk.act)?;
                    if blk.se > 0 {
                        im.squeeze_excite(&format!("{p}.se"), blk.mid, blk.se)?;
                    }
                    im.conv(&format!("{p}.pwl"), blk.mid, blk.out, 1, 1, 1, Act::None)?;
                }
                Kind::Cn => im.conv(&format!("{p}.conv"), c, blk.out, 1, 1, 1, blk.act)?,
            }
            if crate::model::residual(blk, c) {
                im.structural("Add")?;
            }
            c = blk.out;
        }
        stage_out.push(c);
    }
    for (k, up) in cfg.ups.iter().enumerate() {
        im.resize_to_skip()?;
        let cin = stage_out.get(up.skip).ok_or_else(|| format!("up{} skips stage {} of {}", k + 1, up.skip, stage_out.len()))? + c;
        im.conv(&format!("up{}.dw", k + 1), cin, cin, 3, 1, cin, up.dw_act)?;
        im.conv(&format!("up{}.pw", k + 1), cin, up.out, 1, 1, 1, Act::Relu6)?;
        c = up.out;
    }
    let h = cfg.head;
    im.conv("head.mid", c, h.mid, 3, 1, h.groups, h.mid_act)?;
    im.conv("head.out", h.mid, h.out, 1, 1, h.out_groups, h.out_act)?;
    if cfg.peak_pool {
        let at = im.w.at();
        let mp = im.structural("MaxPool")?;
        let (k, st) = (read::attr_ints(mp, "kernel_shape", &[]), read::attr_ints(mp, "strides", &[1, 1]));
        let pads = read::attr_ints(mp, "pads", &[0; 4]);
        if k != [3, 3] || st != [1, 1] || pads != [1; 4] {
            return Err(format!("MaxPool at node {at} is k{k:?} s{st:?} p{pads:?}, expected the 3x3/1 peak pool"));
        }
    }
    let manifest = std::mem::take(&mut im.manifest);
    im.w.finish(&manifest, cfg.file)
}
