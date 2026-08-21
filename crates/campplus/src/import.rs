// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import from the released `campplus.onnx` (CosyVoice's speaker
//! encoder, byte-identical across CosyVoice 2 and 3).
//!
//! The graph walk, the two-way coverage ledger and the `Conv`/`BatchNormalization`
//! hyperparameter assertions are `onnx::walk` - shared with every other
//! ONNX-only release brain imports (`crates/scrfd`, `crates/arcface`). What
//! lives here is the ARCHITECTURE: which op sequence this graph has and which
//! canonical name each binding gets.
//!
//! # A graph that is BOTH BN-folded and BN-standalone
//!
//! Unlike `scrfd`/`arcface` (every BatchNorm folded) this release is a mix,
//! and which applies is determined by how many consumers a conv's raw output
//! has, not by any per-module rule:
//!
//! * The entire `FCM` 2D stem (`head.*`) has every BatchNorm folded into its
//!   preceding conv (single consumer throughout: `conv -> bn -> relu`, next
//!   op reads only the ReLU), so every `head.*` conv is biased and no
//!   `BatchNormalization` node survives there.
//! * `xvector.tdnn.linear` and every `CAMDenseTDNNLayer.linear1` are ALSO
//!   folded (their own trailing `BN+ReLU` has one consumer), so they too are
//!   biased convs.
//! * Each `CAMDenseTDNNLayer`'s LEADING `nonlinear1` BatchNorm - the
//!   pre-activation before `linear1` - survives standalone: its input is a
//!   DenseNet `Concat`, and fold only ever collapses a **trailing** BN into
//!   the conv immediately before it, never a leading one into the conv after.
//! * Each `TransitLayer`'s leading `nonlinear` BatchNorm survives standalone
//!   for the same reason. `transit1`/`transit2`'s conv stays unbiased (their
//!   raw output feeds BOTH the next block's leading BN AND the DenseNet
//!   `Concat` that starts that block - two consumers, so nothing folds).
//!   `transit3`'s conv is the one exception: its single consumer is
//!   `out_nonlinear`'s `BN+ReLU`, whose BN then folds into it - `transit3`'s
//!   conv is biased, `out_nonlinear` has no `BatchNormalization` node of its
//!   own, only the `Relu` that follows.
//! * `xvector.dense.nonlinear.batchnorm` (the final `affine=False` BN) is
//!   ALSO standalone (its own output IS the graph output, single consumer of
//!   nothing) but its `weight`/`bias` inputs are exporter `Constant`s (`1`s
//!   and `0`s), not checkpoint initializers - `affine=False` has no learned
//!   gamma/beta to load. Only `running_mean`/`running_var` are bound; the
//!   forward applies `(x - mean) / sqrt(var + eps)` directly.
//!
//! Every one of these facts was read off `campplus.onnx` directly (`onnx`
//! Python package, node-by-node), not assumed from the reference `torch`
//! module - see `crate::config`'s module doc for the cross-check that the
//! derived manifest's tensor count (617) matches the graph's own initializer
//! count exactly.
//!
//! # `FCM`'s asymmetric stride needs `conv3d`, not `conv2d`
//!
//! `head.layer{1,2}` and `head.conv2` downsample FREQUENCY only (stride
//! `(2,1)`, halving the 80-bin fbank axis while time stays unchanged). No 2D
//! conv kernel in `crates/kernels` supports independent per-axis stride
//! (`conv2d`/`conv2d_gd`/`conv2d_tiled` all take one scalar `stride`) - only
//! `conv3d` does (`st,sh,sw` independently, and it happens to carry a fused
//! bias, which every `head.*` conv needs). So the `FCM` stem runs as
//! `conv3d` with a dummy singleton `T` axis (`KT=1, st=1, pt=0`), not a new
//! kernel: `conv3d`'s NCTHW layout with `T=1` is bit-identical to NCHW at
//! every `(h,w)` a real `conv2d` would visit, so an existing kernel already
//! covers this shape and no new one is needed.

use onnx::onnx::{GraphProto, NodeProto};
use onnx::read;
use onnx::walk::{check_conv1d, check_conv2d, Tensors, Walk};

use crate::config::CampplusConfig;

/// The released file this encoder reads (identical across CosyVoice 2 and 3).
pub const RELEASE_FILE: &str = "campplus.onnx";

/// Assert a `BatchNormalization` node's epsilon against `cfg.bn_eps`.
///
/// `bn_eval.wgsl` hardcodes `eps = 1e-5`, so an imported BatchNorm whose graph
/// says otherwise would run with the wrong epsilon and no error.
pub fn check_bn(n: &NodeProto, at: usize, eps: f32) -> Result<(), String> {
    let got = read::attr_f32(n, "epsilon", eps);
    if (got - eps).abs() > 1e-9 {
        return Err(format!("import(campplus): BatchNormalization at node {at} has epsilon {got:e}, but bn_eval runs at {eps:e}"));
    }
    Ok(())
}

/// The four tensors a standalone `nn.BatchNorm1d` at `prefix` owns, in the
/// order the ONNX `BatchNormalization` node takes them (`scale, B, mean,
/// var`).
fn bn_names(prefix: &str) -> [String; 4] {
    [
        format!("{prefix}.weight"),
        format!("{prefix}.bias"),
        format!("{prefix}.running_mean"),
        format!("{prefix}.running_var"),
    ]
}

/// One `FCM` 2D conv, in graph order: canonical prefix, weight shape
/// `[cout, cin, kh, kw]`, and its `(sh, sw)` stride (kernel/pad follow from
/// the shape: 3x3 convs are pad 1, 1x1 shortcuts are pad 0).
struct FcmConv {
    prefix: String,
    shape: [usize; 4],
    stride: (i64, i64),
}

fn fcm_schedule(cfg: &CampplusConfig) -> Vec<FcmConv> {
    let fc = cfg.fcm_channels as usize;
    let mut v = Vec::new();
    v.push(FcmConv { prefix: "head.conv1".into(), shape: [fc, 1, 3, 3], stride: (1, 1) });
    for li in 0..2usize {
        for bi in 0..2usize {
            let p = format!("head.layer{}.{}", li + 1, bi);
            let s = if bi == 0 { 2 } else { 1 };
            v.push(FcmConv { prefix: format!("{p}.conv1"), shape: [fc, fc, 3, 3], stride: (s, 1) });
            v.push(FcmConv { prefix: format!("{p}.conv2"), shape: [fc, fc, 3, 3], stride: (1, 1) });
            if bi == 0 {
                v.push(FcmConv { prefix: format!("{p}.shortcut"), shape: [fc, fc, 1, 1], stride: (2, 1) });
            }
        }
    }
    v.push(FcmConv { prefix: "head.conv2".into(), shape: [fc, fc, 3, 3], stride: (2, 1) });
    v
}

/// One `xvector`-side 1D conv, in graph order.
struct TdnnConv {
    prefix: String,
    shape: [usize; 3],
    stride: i64,
    dilation: i64,
    bias: bool,
}

fn tdnn_schedule(cfg: &CampplusConfig) -> Vec<TdnnConv> {
    let (tdnn_out, cam_mid, cam_out) = (cfg.tdnn_out as usize, cfg.cam_mid as usize, cfg.cam_out as usize);
    let mut v = Vec::new();
    v.push(TdnnConv {
        prefix: "xvector.tdnn.linear".into(),
        shape: [tdnn_out, cfg.fcm_out_c() as usize, 5],
        stride: 2,
        dilation: 1,
        bias: true,
    });
    for b in 0..3usize {
        let cin0 = cfg.block_in_c(b) as usize;
        let growth = cfg.growth as usize;
        let dilation = cfg.block_dilation[b] as i64;
        for i in 0..cfg.block_layers[b] as usize {
            let cin = cin0 + i * growth;
            let p = format!("xvector.block{}.tdnnd{}", b + 1, i + 1);
            v.push(TdnnConv { prefix: format!("{p}.linear1"), shape: [tdnn_out, cin, 1], stride: 1, dilation: 1, bias: true });
            v.push(TdnnConv {
                prefix: format!("{p}.cam.linear_local"),
                shape: [cam_out, tdnn_out, 3],
                stride: 1,
                dilation,
                bias: false,
            });
            v.push(TdnnConv { prefix: format!("{p}.cam.linear1"), shape: [cam_mid, tdnn_out, 1], stride: 1, dilation: 1, bias: true });
            v.push(TdnnConv { prefix: format!("{p}.cam.linear2"), shape: [cam_out, cam_mid, 1], stride: 1, dilation: 1, bias: true });
        }
        let transit_in = cfg.block_out_c(b) as usize;
        let transit_out = cfg.transit_out_c(b) as usize;
        v.push(TdnnConv {
            prefix: format!("xvector.transit{}.linear", b + 1),
            shape: [transit_out, transit_in, 1],
            stride: 1,
            dilation: 1,
            // Only the LAST transit absorbed a folded trailing BN - see the
            // module doc.
            bias: b == 2,
        });
    }
    let (e, dense_in) = (cfg.embedding_size as usize, cfg.stats_out_c() as usize);
    v.push(TdnnConv { prefix: "xvector.dense.linear".into(), shape: [e, dense_in, 1], stride: 1, dilation: 1, bias: false });
    v
}

/// One standalone `BatchNormalization`, in graph order: canonical prefix and
/// channel width. `dense`'s final BN is handled separately (only 2 of its 4
/// tensors are checkpoint-sourced - see the module doc).
///
/// `pub(crate)`: `crate::model` packs each of these into the interleaved
/// `mv`/`gb` buffers `bn_eval` wants, once at construction time, and needs
/// the identical (prefix, width) list this walk binds against.
pub(crate) struct BnPlan {
    pub(crate) prefix: String,
    pub(crate) c: usize,
}

pub(crate) fn bn_schedule(cfg: &CampplusConfig) -> Vec<BnPlan> {
    let mut v = Vec::new();
    for b in 0..3usize {
        let cin0 = cfg.block_in_c(b) as usize;
        let growth = cfg.growth as usize;
        for i in 0..cfg.block_layers[b] as usize {
            v.push(BnPlan {
                prefix: format!("xvector.block{}.tdnnd{}.nonlinear1", b + 1, i + 1),
                c: cin0 + i * growth,
            });
        }
        v.push(BnPlan { prefix: format!("xvector.transit{}.nonlinear", b + 1), c: cfg.block_out_c(b) as usize });
    }
    v
}

/// Import `campplus.onnx`.
///
/// Walks the graph node-by-node, binding every `Conv`/`BatchNormalization`
/// positionally against [`fcm_schedule`] + [`tdnn_schedule`] and
/// [`bn_schedule`], and acknowledging every other node (`Relu`, `Sigmoid`,
/// `ReduceMean`, `Pad`, `AveragePool`, `Shape`, `Constant`, `Gather`,
/// `Unsqueeze`, `Concat`, `Reshape`, `ConstantOfShape`, `Mul`, `Equal`,
/// `Where`, `Expand`, `Slice`, `Add`, `Transpose`) as structure - the
/// `CAMLayer` context computation (`x.mean(-1) + seg_pooling(x)`) compiles
/// to ~50 of those per layer, none carrying a checkpoint weight.
pub fn import_campplus(g: &GraphProto, cfg: &CampplusConfig) -> Result<Tensors, String> {
    let mut w = Walk::new(g)?;
    let manifest = cfg.tensor_manifest();

    let fcm = fcm_schedule(cfg);
    let tdnn = tdnn_schedule(cfg);
    let bns = bn_schedule(cfg);

    let n_conv = g.node.iter().filter(|n| n.op_type == "Conv").count();
    if n_conv != fcm.len() + tdnn.len() {
        return Err(format!("import(campplus): graph has {n_conv} Conv nodes, the config schedule expects {}", fcm.len() + tdnn.len()));
    }
    let n_bn = g.node.iter().filter(|n| n.op_type == "BatchNormalization").count();
    // +1: `xvector.dense.nonlinear.batchnorm`, handled outside `bns` (its
    // gamma/beta are exporter constants, not checkpoint tensors).
    if n_bn != bns.len() + 1 {
        return Err(format!("import(campplus): graph has {n_bn} BatchNormalization nodes, expected {}", bns.len() + 1));
    }

    let mut conv_i = 0usize;
    let mut bn_i = 0usize;
    let n_fcm_conv = fcm.len();
    loop {
        match w.peek() {
            Some("Conv") => {
                let n = w.next("Conv")?;
                let at = w.at() - 1;
                if conv_i < n_fcm_conv {
                    let plan = &fcm[conv_i];
                    let (kh, kw) = (plan.shape[2] as i64, plan.shape[3] as i64);
                    let pad = if kh == 1 { 0 } else { 1 };
                    check_conv2d(n, at, kh, kw, plan.stride.0, plan.stride.1, pad, pad)?;
                    let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
                    let cout = plan.shape[0];
                    w.bind(&format!("{}.weight", plan.prefix), &cw, plan.shape.to_vec())?;
                    w.bind(&format!("{}.bias", plan.prefix), &cb, vec![cout])?;
                } else {
                    let plan = &tdnn[conv_i - n_fcm_conv];
                    let k = plan.shape[2] as i64;
                    let pad = (k / 2) * plan.dilation;
                    check_conv1d(n, at, k, plan.stride, pad, plan.dilation)?;
                    let cw = n.input[1].clone();
                    w.bind(&format!("{}.weight", plan.prefix), &cw, plan.shape.to_vec())?;
                    if plan.bias {
                        let cb = n.input[2].clone();
                        w.bind(&format!("{}.bias", plan.prefix), &cb, vec![plan.shape[0]])?;
                    }
                }
                conv_i += 1;
            }
            Some("BatchNormalization") => {
                let n = w.next("BatchNormalization")?;
                check_bn(n, w.at() - 1, cfg.bn_eps)?;
                if bn_i < bns.len() {
                    let plan = &bns[bn_i];
                    let ins: Vec<String> = n.input[1..5].to_vec();
                    for (dst, src) in bn_names(&plan.prefix).iter().zip(&ins) {
                        w.bind(dst, src, vec![plan.c])?;
                    }
                } else {
                    // `xvector.dense.nonlinear.batchnorm`: gamma/beta (inputs
                    // 1,2) are exporter Constants, not initializers - only
                    // mean/var are real checkpoint tensors.
                    let e = cfg.embedding_size as usize;
                    w.bind("xvector.dense.nonlinear.running_mean", &n.input[3], vec![e])?;
                    w.bind("xvector.dense.nonlinear.running_var", &n.input[4], vec![e])?;
                }
                bn_i += 1;
            }
            Some(_) => {
                let n = w.next_any().expect("peeked a node");
                w.ack_structural(n);
            }
            None => break,
        }
    }
    w.finish(&manifest, "campplus")
}

/// Read the encoder from `dir` and return its imported tensors.
///
/// `dir` comes from a CLI flag, `BRAIN_CAMPPLUS_DIR` or `$BRAIN_TESTDATA` -
/// never a baked-in path.
pub fn import_dir(dir: &std::path::Path) -> Result<Tensors, String> {
    let m = onnx::read_file(dir.join(RELEASE_FILE))?;
    import_campplus(read::graph(&m)?, &CampplusConfig::campplus_v2())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two independently-derived Conv/BN schedules must have the counts
    /// [`import_campplus`] asserts against the real graph at runtime.
    #[test]
    fn schedules_have_the_expected_lengths() {
        let cfg = CampplusConfig::campplus_v2();
        assert_eq!(fcm_schedule(&cfg).len(), 12);
        assert_eq!(tdnn_schedule(&cfg).len(), 213);
        assert_eq!(bn_schedule(&cfg).len(), 55);
    }
}
