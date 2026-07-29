// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Nemotron 3.5 ASR **FastConformer encoder** as an OpenVINO-compilable ONNX graph
//! (mel → pooler `[T', 640]`), built for a FIXED sequence length so it compiles to a
//! static NPU graph. The RNN-T decode stays on host (m=1 steps).
//!
//! This is brain's hardest NPU export: it lands incrementally, each stage
//! parity-gated against the dumped HF activations (`testdata/asr/golden/nemotron/`):
//!   1. depthwise-separable causal subsampling (×8) + linear     ← this pass
//!   2. macaron Conformer blocks (rel-pos attention + GLU conv)  (next)
//!   3. prompt + encoder projectors
//!
//! Design mirrors the host reference (`nemotron::reference`) and the device encoder
//! (`nemotron::encoder`) op-for-op so the same goldens gate all three. Weights arrive
//! through the shared `crate::topology::WeightSource` (a name→f32 map).

use onnx::{GraphBuilder, Node};

use crate::topology::WeightSource;

/// Config subset the encoder graph needs (mirrors `nemotron::NemotronConfig`).
#[derive(Clone, Copy, Debug)]
pub struct NemotronTopo {
    pub num_mel_bins: u32,
    pub hidden: u32,
    pub subsampling_channels: u32,
    pub subsampling_kernel: u32,
    pub subsampling_stride: u32,
    pub subsampling_stages: u32,
}

impl Default for NemotronTopo {
    fn default() -> Self {
        NemotronTopo { num_mel_bins: 128, hidden: 1024, subsampling_channels: 256, subsampling_kernel: 3, subsampling_stride: 2, subsampling_stages: 3 }
    }
}

impl NemotronTopo {
    /// Length after one causal stride-2 stage: `(len + (k-1)+(s-1) - k)/s + 1`.
    fn stage_len(&self, len: u32) -> u32 {
        let (k, s) = (self.subsampling_kernel, self.subsampling_stride);
        (len + (k - 1) + (s - 1) - k) / s + 1
    }
    /// Subsampled time length after the full stack.
    pub fn subsampled_len(&self, mel_valid: u32) -> u32 {
        let mut l = mel_valid;
        for _ in 0..self.subsampling_stages {
            l = self.stage_len(l);
        }
        l
    }
    /// Output frequency bins after the stack.
    pub fn out_freq(&self) -> u32 {
        let mut f = self.num_mel_bins;
        for _ in 0..self.subsampling_stages {
            f = self.stage_len(f);
        }
        f
    }
}

/// A named tensor with its current spatial dims, threaded through the conv stack.
struct Feat {
    name: String,
    c: u32,
    t: u32,
    f: u32,
}

/// Emit a causal Conv2d (`(k-1,s-1)` asymmetric pad on both axes, matching NeMo) with
/// a per-channel bias. When `mask_relu` is set, also apply the time-mask (zero frames
/// `>= stage_len(valid)`) and a ReLU — the depthwise-separable stage masks+ReLUs only
/// AFTER the pointwise conv, so its depthwise call passes `mask_relu=false` (a raw
/// conv). Weight is `[cout, cin/groups, k, k]`, bias `[cout]`, verbatim.
#[allow(clippy::too_many_arguments)]
fn causal_conv(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, x: &Feat, cout: u32, wname: &str, bname: &str, groups: u32, valid: u32, mask_relu: bool, tag: &str) -> Feat {
    let (k, s) = (topo.subsampling_kernel, topo.subsampling_stride);
    let kw = w.get(wname);
    let bw = w.get(bname);
    let wn = format!("{tag}.w");
    let bn = format!("{tag}.b");
    g.init_f32(&wn, &[cout as i64, (x.c / groups) as i64, k as i64, k as i64], kw);
    g.init_f32(&bn, &[cout as i64], bw);
    let out = format!("{tag}.conv");
    // ONNX pads order for 2-D: [t_begin, f_begin, t_end, f_end] = causal (k-1) front, (s-1) back.
    let (pb, pe) = ((k - 1) as i64, (s - 1) as i64);
    g.add(
        Node::new("Conv", &[&x.name, &wn, &bn], &[&out])
            .name(&format!("{tag}.conv"))
            .attr_ints("kernel_shape", &[k as i64, k as i64])
            .attr_ints("strides", &[s as i64, s as i64])
            .attr_ints("pads", &[pb, pb, pe, pe])
            .attr_int("group", groups as i64),
    );
    let (to, fo) = (topo.stage_len(x.t), topo.stage_len(x.f));
    if !mask_relu {
        return Feat { name: out, c: cout, t: to, f: fo };
    }
    let vout = topo.stage_len(valid);
    let masked = mask_time(g, &out, cout, to, fo, vout, tag);
    let relu = format!("{tag}.relu");
    g.add(Node::new("Relu", &[&masked], &[&relu]).name(&format!("{tag}.relu")));
    Feat { name: relu, c: cout, t: to, f: fo }
}

/// Zero time frames `>= valid` in an NCHW `[1,C,T,F]` tensor (NeMo
/// `_mask_subsampled_frames`) by multiplying with a constant `[1,1,T,1]` 0/1 mask.
/// A no-op (all-ones) when `valid >= t`, so a full window costs nothing.
fn mask_time(g: &mut GraphBuilder, x: &str, _c: u32, t: u32, _f: u32, valid: u32, tag: &str) -> String {
    if valid >= t {
        return x.to_string();
    }
    let mask: Vec<f32> = (0..t).map(|i| if i < valid { 1.0 } else { 0.0 }).collect();
    let mn = format!("{tag}.tmask");
    g.init_f32(&mn, &[1, 1, t as i64, 1], mask);
    let out = format!("{tag}.masked");
    g.add(Node::new("Mul", &[x, &mn], &[&out]).name(&format!("{tag}.mask")));
    out
}

/// Build the **subsampling** stage: mel `[1,1,T,num_mel]` → `[T', hidden]` (named
/// `out_name`). `mel_t` is the input time length, `mel_valid` the real frames.
pub fn build_subsampling(g: &mut GraphBuilder, topo: &NemotronTopo, w: &dyn WeightSource, mel_t: u32, mel_valid: u32, input_name: &str, out_name: &str) {
    let ch = topo.subsampling_channels;
    // stem: conv_in (1 -> ch)
    let x = Feat { name: input_name.to_string(), c: 1, t: mel_t, f: topo.num_mel_bins };
    let mut cur = causal_conv(g, topo, w, &x, ch, "encoder.subsampling.conv_in.weight", "encoder.subsampling.conv_in.bias", 1, mel_valid, true, "sub.stem");
    let mut vlen = topo.stage_len(mel_valid);

    // depthwise-separable stages
    for i in 0..topo.subsampling_stages - 1 {
        let dw = causal_conv(
            g,
            topo,
            w,
            &cur,
            ch,
            &format!("encoder.subsampling.layers.{i}.depthwise_conv.weight"),
            &format!("encoder.subsampling.layers.{i}.depthwise_conv.bias"),
            ch,
            vlen,
            false, // depthwise: no mask/relu — they come after the pointwise conv
            &format!("sub.dw{i}"),
        );
        // pointwise 1x1 (stride 1, no pad); the reference masks+relus after pointwise.
        let pw = pointwise(g, w, &dw, ch, &format!("encoder.subsampling.layers.{i}.pointwise_conv.weight"), &format!("encoder.subsampling.layers.{i}.pointwise_conv.bias"), topo.stage_len(vlen), &format!("sub.pw{i}"));
        vlen = topo.stage_len(vlen);
        cur = pw;
    }

    // reshape [1,C,T',F'] -> [T', C*F'] : transpose to [1,T',C,F'] then reshape.
    let (tt, ff) = (cur.t, cur.f);
    let flat = ch * ff;
    let tp = format!("sub.perm");
    g.add(Node::new("Transpose", &[&cur.name], &[&tp]).name("sub.perm").attr_ints("perm", &[0, 2, 1, 3]));
    let shp = "sub.flatshape";
    g.init_i64(shp, &[2], vec![tt as i64, flat as i64]);
    let flatn = "sub.flat";
    g.add(Node::new("Reshape", &[&tp, shp], &[flatn]).name("sub.reshape"));

    // linear [T', flat] @ W^T [flat, hidden] + bias -> [T', hidden]
    linear(g, w, flatn, "encoder.subsampling.linear.weight", "encoder.subsampling.linear.bias", topo.hidden, flat, out_name, "sub.lin");
}

/// 1×1 pointwise Conv2d (dense, stride 1) + bias, then mask+relu.
fn pointwise(g: &mut GraphBuilder, w: &dyn WeightSource, x: &Feat, cout: u32, wname: &str, bname: &str, valid: u32, tag: &str) -> Feat {
    let wn = format!("{tag}.w");
    let bn = format!("{tag}.b");
    g.init_f32(&wn, &[cout as i64, x.c as i64, 1, 1], w.get(wname));
    g.init_f32(&bn, &[cout as i64], w.get(bname));
    let out = format!("{tag}.conv");
    g.add(
        Node::new("Conv", &[&x.name, &wn, &bn], &[&out])
            .name(&format!("{tag}.conv"))
            .attr_ints("kernel_shape", &[1, 1])
            .attr_ints("strides", &[1, 1])
            .attr_ints("pads", &[0, 0, 0, 0])
            .attr_int("group", 1),
    );
    let masked = mask_time(g, &out, cout, x.t, x.f, valid, tag);
    let relu = format!("{tag}.relu");
    g.add(Node::new("Relu", &[&masked], &[&relu]).name(&format!("{tag}.relu")));
    Feat { name: relu, c: cout, t: x.t, f: x.f }
}

/// A linear `x[m, in] @ W^T[in, out] + b -> [m, out]` (weight stored `[out, in]`,
/// transposed into the graph as `[in, out]`).
fn linear(g: &mut GraphBuilder, w: &dyn WeightSource, x: &str, wname: &str, bname: &str, out: u32, inn: u32, out_name: &str, tag: &str) {
    let wt = transpose_2d(&w.get(wname), out as usize, inn as usize);
    let wn = format!("{tag}.wT");
    g.init_f32(&wn, &[inn as i64, out as i64], wt);
    let mm = format!("{tag}.mm");
    g.add(Node::new("MatMul", &[x, &wn], &[&mm]).name(&format!("{tag}.mm")));
    let bn = format!("{tag}.b");
    g.init_f32(&bn, &[out as i64], w.get(bname));
    g.add(Node::new("Add", &[&mm, &bn], &[out_name]).name(&format!("{tag}.bias")));
}

/// Transpose a row-major `[rows, cols]` matrix to `[cols, rows]`.
fn transpose_2d(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}
