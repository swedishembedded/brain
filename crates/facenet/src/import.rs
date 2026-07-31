// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import from the released insightface `antelopev2` ONNX files, with
//! **two-way coverage validation** (the `flux2::import` discipline):
//!
//!   * every tensor the config's canonical manifest expects is produced exactly
//!     once, with the shape the manifest states, and
//!   * every initializer in the source graph is consumed at least once.
//!
//! A mismatch is an error naming the tensor. Nothing is ever zero-filled or
//! skipped.
//!
//! # Why the walk is topological, not a name remap
//!
//! Every other brain importer maps source names to brain names. That is
//! impossible here: **the ONNX exporter folded BatchNorm into the convolutions
//! and the folded tensors lost their names.** In `glintr100.onnx`, 256 of the 462
//! initializers are called `1335`, `1336`, `1643`… — bare SSA value numbers. Only
//! the 206 tensors that survived folding (the `bn1`/`bn2`/`features`
//! BatchNorms and `fc`) still carry `layer2.3.bn1.weight`-style names.
//!
//! So the source of truth for "which tensor is this" is the graph's **topology**:
//! the n-th `Conv` node in graph order is a known convolution of a known
//! architecture. Both walks therefore assert the op sequence / op counts they
//! expect and bind weights positionally, then check every shape. That is a
//! stronger check than a name match, not a weaker one — a name map cannot notice
//! that the graph has 48 residual adds instead of 49.
//!
//! # One source tensor may feed two convolutions
//!
//! `scrfd_10g_bnkps.onnx` genuinely shares two bias initializers between
//! different convolutions (`neck.fpn_convs.1` reads
//! `neck.downsample_convs.0.conv.bias`). Whether that is exporter deduplication
//! of equal tensors or a quirk of the release, the goldens were dumped by running
//! *this* file, so the import reproduces the sharing exactly: coverage counts a
//! source tensor as covered when it is used **one or more** times.

use std::collections::HashMap;

use onnx::onnx::{GraphProto, NodeProto};
use onnx::read;

use crate::config::{ArcFaceConfig, ScrfdConfig};

/// name → (shape, fp32 data), keyed by brain's canonical names.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// A canonical tensor manifest entry: brain name + expected shape.
pub type Manifest = Vec<(String, Vec<usize>)>;

// ===========================================================================
// The walk helper — binds source initializers to canonical names and keeps the
// two coverage ledgers.
// ===========================================================================

struct Walk<'g> {
    src: HashMap<String, read::OnnxTensor>,
    /// How many times each source initializer was consumed (weights or
    /// structure). Zero at the end = an unaccounted source tensor.
    used: HashMap<String, u32>,
    out: Tensors,
    nodes: &'g [NodeProto],
    /// Cursor over `nodes`, so the op-sequence assertions read linearly.
    at: usize,
}

impl<'g> Walk<'g> {
    fn new(g: &'g GraphProto) -> Result<Walk<'g>, String> {
        let src = read::initializers(g)?;
        let used = src.keys().map(|k| (k.clone(), 0u32)).collect();
        Ok(Walk { src, used, out: Tensors::new(), nodes: &g.node, at: 0 })
    }

    /// The next node, asserting its op type.
    fn next(&mut self, op: &str) -> Result<&'g NodeProto, String> {
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

    /// Peek the op type of the next node without consuming it.
    fn peek(&self) -> Option<&'g str> {
        self.nodes.get(self.at).map(|n| n.op_type.as_str())
    }

    /// Bind source initializer `src_name` to canonical `dst`, optionally
    /// reshaping (an ONNX PReLU slope is `[C,1,1]`; brain wants `[C]`).
    fn bind(&mut self, dst: &str, src_name: &str, shape: Vec<usize>) -> Result<(), String> {
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

    /// Mark a source initializer as consumed by graph STRUCTURE rather than as a
    /// weight (a `Reshape` shape, a `Resize` scales tensor, a `Slice` bound, a
    /// `Gather` index). It is not a parameter, but it must still be accounted
    /// for, or the coverage check would flag it as unused.
    fn ack_structural(&mut self, n: &NodeProto) {
        for i in n.input.iter().skip(1) {
            if let Some(c) = self.used.get_mut(i) {
                *c += 1;
            }
        }
    }

    /// Two-way coverage: manifest completeness + no unused source tensor.
    fn finish(self, manifest: &Manifest, what: &str) -> Result<Tensors, String> {
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

/// `bn_eval.wgsl` hardcodes `eps = 1e-5`, so an imported BatchNorm whose graph
/// says otherwise would run with the wrong epsilon and no error. Both antelopev2
/// graphs export `epsilon = 1e-5`; this is the assertion that keeps that true.
const BN_EPS: f32 = 1e-5;

/// Assert an ONNX `Conv` node's hyperparameters against what the model will
/// dispatch.
///
/// Both walks bind weights POSITIONALLY, so a shape check alone is not enough:
/// a release with the same tensor shapes but a different stride, pad or
/// dilation imports cleanly and produces a whole wrong network. The model's
/// geometry comes from `config`, not from the file, which is exactly why the
/// file has to be checked against it.
fn check_conv(n: &NodeProto, at: usize, k: i64, stride: i64, pad: i64) -> Result<(), String> {
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
        return Err(format!("import: Conv at node {at} has group {g}; both graphs are dense (group 1)"));
    }
    Ok(())
}

/// Assert the geometry of the SCRFD nodes the model reproduces with a kernel but
/// binds no weight from: the stem `MaxPool`, the ResNet-D shortcut
/// `AveragePool`, and the FPN `Resize`.
///
/// These carry no initializer, so the two-way coverage ledger never looks at
/// them — yet `vision::MaxPool` runs at `PoolSpec::half()`, `AvgPool` at an
/// exact 2x2 box, and `resize_nearest` at ONNX's `asymmetric` + `floor` rule,
/// all hardcoded. A graph that says something else must fail loudly here.
fn check_structural(n: &NodeProto, at: usize) -> Result<(), String> {
    let pool2x2 = |what: &str| -> Result<(), String> {
        let k = read::attr_ints(n, "kernel_shape", &[]);
        let s = read::attr_ints(n, "strides", &[]);
        let p = read::attr_ints(n, "pads", &[0; 4]);
        if k != vec![2, 2] || s != vec![2, 2] || p != vec![0; 4] {
            return Err(format!(
                "import(scrfd): {what} at node {at} is kernel {k:?} stride {s:?} pad {p:?}, expected a 2x2/2 unpadded pool"
            ));
        }
        Ok(())
    };
    match n.op_type.as_str() {
        "MaxPool" => pool2x2("MaxPool"),
        "AveragePool" => pool2x2("AveragePool"),
        "Resize" => {
            let m = read::attr_str(n, "mode", "nearest");
            let ct = read::attr_str(n, "coordinate_transformation_mode", "half_pixel");
            let nm = read::attr_str(n, "nearest_mode", "round_prefer_floor");
            if m != "nearest" || ct != "asymmetric" || nm != "floor" {
                return Err(format!(
                    "import(scrfd): Resize at node {at} is mode {m}/{ct}/{nm}; `resize_nearest` implements nearest/asymmetric/floor"
                ));
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Assert a `BatchNormalization` node's epsilon against [`BN_EPS`].
fn check_bn(n: &NodeProto, at: usize) -> Result<(), String> {
    let eps = read::attr_f32(n, "epsilon", BN_EPS);
    if (eps - BN_EPS).abs() > 1e-9 {
        return Err(format!(
            "import: BatchNormalization at node {at} has epsilon {eps:e}, but `bn_eval` runs at {BN_EPS:e}"
        ));
    }
    Ok(())
}

/// The four tensors a torch `nn.BatchNorm2d` at `prefix` owns, in the order the
/// ONNX `BatchNormalization` node takes them (`scale, B, mean, var`).
fn bn_names(prefix: &str) -> [String; 4] {
    [
        format!("{prefix}.weight"),
        format!("{prefix}.bias"),
        format!("{prefix}.running_mean"),
        format!("{prefix}.running_var"),
    ]
}

// ===========================================================================
// ArcFace / IResNet-100
// ===========================================================================

impl ArcFaceConfig {
    /// Every tensor the model reads, with its shape. Counted against the
    /// checkpoint by [`import_arcface`]: 462 for IResNet-100, which is exactly
    /// `glintr100.onnx`'s initializer count.
    pub fn tensor_manifest(&self) -> Manifest {
        let c = |x: u32| x as usize;
        let mut m: Manifest = Vec::new();
        let st = c(self.stem_channels);
        m.push(("stem.conv.weight".into(), vec![st, 3, 3, 3]));
        m.push(("stem.conv.bias".into(), vec![st]));
        m.push(("stem.prelu.weight".into(), vec![st]));
        for s in 0..4usize {
            let cin = c(self.stage_in_c(s));
            let cout = c(self.channels[s]);
            for b in 0..self.layers[s] as usize {
                let p = format!("layer{}.{}", s + 1, b);
                // bn1 sits at the block ENTRY, so it is `cin` wide for the first
                // block of a stage and `cout` wide thereafter.
                let bn_c = if b == 0 { cin } else { cout };
                for n in bn_names(&format!("{p}.bn1")) {
                    m.push((n, vec![bn_c]));
                }
                let conv1_cin = bn_c;
                m.push((format!("{p}.conv1.weight"), vec![cout, conv1_cin, 3, 3]));
                m.push((format!("{p}.conv1.bias"), vec![cout]));
                m.push((format!("{p}.prelu.weight"), vec![cout]));
                m.push((format!("{p}.conv2.weight"), vec![cout, cout, 3, 3]));
                m.push((format!("{p}.conv2.bias"), vec![cout]));
                if b == 0 {
                    m.push((format!("{p}.downsample.weight"), vec![cout, cin, 1, 1]));
                    m.push((format!("{p}.downsample.bias"), vec![cout]));
                }
            }
        }
        let e = c(self.embedding);
        for n in bn_names("bn2") {
            m.push((n, vec![c(self.channels[3])]));
        }
        m.push(("fc.weight".into(), vec![e, c(self.flatten())]));
        m.push(("fc.bias".into(), vec![e]));
        for n in bn_names("features") {
            m.push((n, vec![e]));
        }
        m
    }
}

/// Import `glintr100.onnx` (ArcFace IResNet-100).
///
/// The expected op sequence, asserted node by node:
/// `Conv PRelu`, then per residual block
/// `BatchNormalization Conv PRelu Conv [Conv] Add`, then
/// `BatchNormalization Flatten Gemm BatchNormalization`.
pub fn import_arcface(g: &GraphProto, cfg: &ArcFaceConfig) -> Result<Tensors, String> {
    let mut w = Walk::new(g)?;
    let manifest = cfg.tensor_manifest();

    // ---- stem: Conv(3 -> 64, 3x3 s1 p1, biased) -> PReLU ----
    let st = cfg.stem_channels as usize;
    let n = w.next("Conv")?;
    check_conv(n, w.at - 1, 3, 1, 1)?;
    let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
    w.bind("stem.conv.weight", &cw, vec![st, 3, 3, 3])?;
    w.bind("stem.conv.bias", &cb, vec![st])?;
    let n = w.next("PRelu")?;
    let slope = n.input[1].clone();
    w.bind("stem.prelu.weight", &slope, vec![st])?;

    // ---- 4 stages of residual blocks ----
    for s in 0..4usize {
        let cin = cfg.stage_in_c(s) as usize;
        let cout = cfg.channels[s] as usize;
        for b in 0..cfg.layers[s] as usize {
            let p = format!("layer{}.{}", s + 1, b);
            let bn_c = if b == 0 { cin } else { cout };

            // Stage-first blocks stride by 2 in conv2 and in the 1x1 shortcut.
            let stride = if b == 0 { 2 } else { 1 };

            let n = w.next("BatchNormalization")?;
            check_bn(n, w.at - 1)?;
            let ins: Vec<String> = n.input[1..5].to_vec();
            for (dst, src) in bn_names(&format!("{p}.bn1")).iter().zip(&ins) {
                w.bind(dst, src, vec![bn_c])?;
            }

            let n = w.next("Conv")?;
            check_conv(n, w.at - 1, 3, 1, 1)?;
            let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
            w.bind(&format!("{p}.conv1.weight"), &cw, vec![cout, bn_c, 3, 3])?;
            w.bind(&format!("{p}.conv1.bias"), &cb, vec![cout])?;

            let n = w.next("PRelu")?;
            let slope = n.input[1].clone();
            w.bind(&format!("{p}.prelu.weight"), &slope, vec![cout])?;

            let n = w.next("Conv")?;
            check_conv(n, w.at - 1, 3, stride, 1)?;
            let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
            w.bind(&format!("{p}.conv2.weight"), &cw, vec![cout, cout, 3, 3])?;
            w.bind(&format!("{p}.conv2.bias"), &cb, vec![cout])?;

            if b == 0 {
                let n = w.next("Conv")?;
                check_conv(n, w.at - 1, 1, stride, 0)?;
                let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
                w.bind(&format!("{p}.downsample.weight"), &cw, vec![cout, cin, 1, 1])?;
                w.bind(&format!("{p}.downsample.bias"), &cb, vec![cout])?;
            }
            w.next("Add")?;
        }
    }

    // ---- tail: bn2 -> flatten -> fc -> features BN ----
    let cf = cfg.channels[3] as usize;
    let n = w.next("BatchNormalization")?;
    check_bn(n, w.at - 1)?;
    let ins: Vec<String> = n.input[1..5].to_vec();
    for (dst, src) in bn_names("bn2").iter().zip(&ins) {
        w.bind(dst, src, vec![cf])?;
    }
    // `axis = 1` is what makes the flatten a no-op over brain's NCHW buffer; any
    // other axis reorders the 25088 values `fc` consumes.
    let n = w.next("Flatten")?;
    if read::attr_int(n, "axis", 1) != 1 {
        return Err("import(arcface): Flatten axis != 1; the [1, 25088] row would be reordered".into());
    }
    let n = w.next("Gemm")?;
    // `transB=1`: the weight is stored [out, in], which is brain's `matmul`
    // layout (`out = x · Wᵀ`) already. A transB=0 export would need a transpose,
    // so the attribute is checked rather than assumed.
    if read::attr_int(n, "transB", 0) != 1 {
        return Err("import(arcface): fc Gemm has transB != 1; the weight layout is not [out, in]".into());
    }
    let (gw, gb) = (n.input[1].clone(), n.input[2].clone());
    let (e, fl) = (cfg.embedding as usize, cfg.flatten() as usize);
    w.bind("fc.weight", &gw, vec![e, fl])?;
    w.bind("fc.bias", &gb, vec![e])?;
    let n = w.next("BatchNormalization")?;
    check_bn(n, w.at - 1)?;
    let ins: Vec<String> = n.input[1..5].to_vec();
    for (dst, src) in bn_names("features").iter().zip(&ins) {
        w.bind(dst, src, vec![e])?;
    }

    w.finish(&manifest, "arcface")
}

// ===========================================================================
// SCRFD
// ===========================================================================

/// One entry of the SCRFD convolution schedule: what the n-th `Conv` node in
/// graph order is, in both weight shape AND geometry.
///
/// The geometry is here because the model builds every conv from `ScrfdConfig`,
/// never from the file — so the file has to be checked against it. A release
/// with the same shapes and a different stride imports cleanly otherwise.
struct ConvPlan {
    prefix: String,
    shape: Vec<usize>,
    stride: i64,
}

impl ConvPlan {
    fn new(prefix: &str, shape: Vec<usize>, stride: i64) -> ConvPlan {
        ConvPlan { prefix: prefix.to_string(), shape, stride }
    }
    /// `pads` is fully determined by the kernel: 3x3 convs are `same`-padded,
    /// 1x1 projections are unpadded. Both graphs hold to that without exception.
    fn k(&self) -> i64 {
        self.shape[2] as i64
    }
    fn pad(&self) -> i64 {
        self.k() / 2
    }
}

/// The 58 convolutions of `scrfd_10g_bnkps.onnx`, in graph order.
///
/// Graph order is the contract: the exporter numbered the folded weights, so
/// position is the only identity a conv has. Deriving the list from the config
/// (rather than hardcoding 58 entries) means a config typo shows up as a shape
/// error naming the conv, not as a silently different network.
fn scrfd_conv_schedule(cfg: &ScrfdConfig) -> Vec<ConvPlan> {
    let c = |x: u32| x as usize;
    let mut v: Vec<ConvPlan> = Vec::new();
    // stem: 3x3 s2, 3x3, 3x3 then a max-pool
    let sc = cfg.stem_channels;
    v.push(ConvPlan::new("backbone.stem.0", vec![c(sc[0]), 3, 3, 3], 2));
    v.push(ConvPlan::new("backbone.stem.1", vec![c(sc[1]), c(sc[0]), 3, 3], 1));
    v.push(ConvPlan::new("backbone.stem.2", vec![c(sc[2]), c(sc[1]), 3, 3], 1));
    for s in 0..4usize {
        let cin = c(cfg.stage_in_c(s));
        let cout = c(cfg.channels[s]);
        for b in 0..cfg.layers[s] as usize {
            let p = format!("backbone.layer{}.{}", s + 1, b);
            let bcin = if b == 0 { cin } else { cout };
            let stride = if b == 0 && cfg.stage_stride2[s] { 2 } else { 1 };
            v.push(ConvPlan::new(&format!("{p}.conv1"), vec![cout, bcin, 3, 3], stride));
            v.push(ConvPlan::new(&format!("{p}.conv2"), vec![cout, cout, 3, 3], 1));
            // A shortcut conv exists exactly where the block changes shape:
            // strided stages, and any stage whose width changes. It is NEVER
            // strided — the ResNet-D downsample puts the stride in an
            // `AveragePool(2, 2)` ahead of an unstrided 1x1.
            if b == 0 && (cfg.stage_stride2[s] || bcin != cout) {
                v.push(ConvPlan::new(&format!("{p}.downsample"), vec![cout, bcin, 1, 1], 1));
            }
        }
    }
    let nk = c(cfg.neck_channels);
    for (i, s) in [1usize, 2, 3].iter().enumerate() {
        let name = format!("neck.lateral_convs.{i}.conv");
        v.push(ConvPlan::new(&name, vec![nk, c(cfg.channels[*s]), 1, 1], 1));
    }
    for i in 0..3 {
        v.push(ConvPlan::new(&format!("neck.fpn_convs.{i}.conv"), vec![nk, nk, 3, 3], 1));
    }
    for i in 0..2 {
        v.push(ConvPlan::new(&format!("neck.downsample_convs.{i}.conv"), vec![nk, nk, 3, 3], 2));
    }
    for i in 0..2 {
        v.push(ConvPlan::new(&format!("neck.pafpn_convs.{i}.conv"), vec![nk, nk, 3, 3], 1));
    }
    let hc = c(cfg.head_channels);
    let na = c(cfg.num_anchors);
    for st in cfg.strides {
        let p = format!("head.stride{st}");
        for d in 0..cfg.head_depth as usize {
            let cin = if d == 0 { nk } else { hc };
            v.push(ConvPlan::new(&format!("{p}.stem.{d}"), vec![hc, cin, 3, 3], 1));
        }
        v.push(ConvPlan::new(&format!("{p}.cls"), vec![na, hc, 3, 3], 1));
        v.push(ConvPlan::new(&format!("{p}.reg"), vec![4 * na, hc, 3, 3], 1));
        v.push(ConvPlan::new(&format!("{p}.kps"), vec![10 * na, hc, 3, 3], 1));
    }
    v
}

impl ScrfdConfig {
    /// Every tensor the detector reads: a `.weight`/`.bias` pair per conv in
    /// [`scrfd_conv_schedule`] order, plus the three learned per-stride bbox
    /// scales. 119 canonical tensors from 125 source initializers — the other
    /// six are `Reshape`/`Resize`/`Slice`/`Gather` constants (graph structure,
    /// not parameters), and two bias tensors are shared by two convs each.
    pub fn tensor_manifest(&self) -> Manifest {
        let mut m: Manifest = Vec::new();
        for c in scrfd_conv_schedule(self) {
            let cout = c.shape[0];
            m.push((format!("{}.weight", c.prefix), c.shape));
            m.push((format!("{}.bias", c.prefix), vec![cout]));
        }
        for i in 0..3 {
            // A rank-0 ONNX scalar. brain has no rank-0 tensor, so it is stored
            // as `[1]`; the model reads element 0.
            m.push((format!("head.scales.{i}"), vec![1]));
        }
        m
    }
}

/// Import `scrfd_10g_bnkps.onnx`.
///
/// Binds `Conv` nodes positionally against [`scrfd_conv_schedule`] and the three
/// `Mul` nodes to the learned per-stride bbox scales; every other node
/// (`Relu`/`Add`/`MaxPool`/`AveragePool`/`Resize`/`Shape`/`Gather`/`Unsqueeze`/
/// `Slice`/`Concat`/`Transpose`/`Reshape`/`Sigmoid`) is structure, and its
/// initializer inputs are accounted as such.
pub fn import_scrfd(g: &GraphProto, cfg: &ScrfdConfig) -> Result<Tensors, String> {
    let mut w = Walk::new(g)?;
    let manifest = cfg.tensor_manifest();
    let schedule = scrfd_conv_schedule(cfg);

    let n_conv = g.node.iter().filter(|n| n.op_type == "Conv").count();
    if n_conv != schedule.len() {
        return Err(format!(
            "import(scrfd): graph has {n_conv} Conv nodes, the config schedule expects {}",
            schedule.len()
        ));
    }
    let n_mul = g.node.iter().filter(|n| n.op_type == "Mul").count();
    if n_mul != 3 {
        return Err(format!("import(scrfd): graph has {n_mul} Mul nodes, expected 3 (one bbox scale per stride)"));
    }

    let mut conv_i = 0usize;
    let mut mul_i = 0usize;
    while w.at < w.nodes.len() {
        match w.peek() {
            Some("Conv") => {
                let n = w.next("Conv")?;
                let plan = &schedule[conv_i];
                check_conv(n, w.at - 1, plan.k(), plan.stride, plan.pad())?;
                let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
                let (p, shape) = (plan.prefix.clone(), plan.shape.clone());
                let cout = shape[0];
                w.bind(&format!("{p}.weight"), &cw, shape)?;
                w.bind(&format!("{p}.bias"), &cb, vec![cout])?;
                conv_i += 1;
            }
            Some("Mul") => {
                let n = w.next("Mul")?;
                let s = n.input[1].clone();
                w.bind(&format!("head.scales.{mul_i}"), &s, vec![1])?;
                mul_i += 1;
            }
            Some(_) => {
                let n = &w.nodes[w.at];
                check_structural(n, w.at)?;
                w.at += 1;
                let n2 = n.clone();
                w.ack_structural(&n2);
            }
            None => break,
        }
    }
    w.finish(&manifest, "scrfd")
}

/// Read both antelopev2 models from `dir` and return their imported tensors.
///
/// `dir` comes from a CLI flag or `$BRAIN_TESTDATA` — never a baked-in path.
pub fn import_dir(dir: &std::path::Path) -> Result<(Tensors, Tensors), String> {
    let rec = onnx::read_file(dir.join("glintr100.onnx"))?;
    let det = onnx::read_file(dir.join("scrfd_10g_bnkps.onnx"))?;
    let arc = import_arcface(read::graph(&rec)?, &ArcFaceConfig::iresnet100())?;
    let scr = import_scrfd(read::graph(&det)?, &ScrfdConfig::scrfd_10g_bnkps())?;
    Ok((arc, scr))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest counts ARE the checkpoint's initializer counts — 462 for
    /// glintr100, and 119 canonical tensors from scrfd's 125 initializers (six
    /// structural constants, two shared biases).
    #[test]
    fn manifest_counts_match_the_released_graphs() {
        assert_eq!(ArcFaceConfig::iresnet100().tensor_manifest().len(), 462);
        assert_eq!(ScrfdConfig::scrfd_10g_bnkps().tensor_manifest().len(), 119);
        assert_eq!(scrfd_conv_schedule(&ScrfdConfig::scrfd_10g_bnkps()).len(), 58);
    }

    /// No canonical name may appear twice — a duplicate would make the
    /// completeness check pass while one of the two tensors went unwritten.
    #[test]
    fn manifest_names_are_unique() {
        for m in [ArcFaceConfig::iresnet100().tensor_manifest(), ScrfdConfig::scrfd_10g_bnkps().tensor_manifest()] {
            let mut names: Vec<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
            let n = names.len();
            names.sort_unstable();
            names.dedup();
            assert_eq!(names.len(), n, "duplicate canonical tensor name");
        }
    }
}
