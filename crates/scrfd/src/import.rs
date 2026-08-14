// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import from the released insightface `scrfd_10g_bnkps.onnx`.
//!
//! The graph walk, the two-way coverage ledger (every manifest tensor produced
//! exactly once; every source initializer consumed at least once) and the
//! `Conv` hyper-parameter assertion are `onnx::walk` - shared with every other
//! ONNX-only release brain imports. What lives here is the ARCHITECTURE: which
//! op sequence this graph has and which canonical name each binding gets.
//!
//! # Why the walk is topological, not a name remap
//!
//! The ONNX exporter folded BatchNorm into the convolutions and the folded
//! tensors lost their names, so the source of truth for "which tensor is this"
//! is the graph's **topology**: the n-th `Conv` node in graph order is a known
//! convolution of a known architecture. The walk binds weights positionally and
//! then checks every shape and every conv's geometry - a stronger check than a
//! name match, not a weaker one.
//!
//! # One source tensor may feed two convolutions
//!
//! `scrfd_10g_bnkps.onnx` genuinely shares two bias initializers between
//! different convolutions (`neck.fpn_convs.1` reads
//! `neck.downsample_convs.0.conv.bias`). Whether that is exporter deduplication
//! of equal tensors or a quirk of the release, the goldens were dumped by
//! running *this* file, so the import reproduces the sharing exactly: coverage
//! counts a source tensor as covered when it is used **one or more** times.

use onnx::onnx::{GraphProto, NodeProto};
use onnx::read;
use onnx::walk::{check_conv, Manifest, Tensors, Walk};

use crate::config::ScrfdConfig;

/// The released file this detector reads, by its antelopev2 name.
pub const RELEASE_FILE: &str = "scrfd_10g_bnkps.onnx";

/// Assert the geometry of the nodes the model reproduces with a kernel but binds
/// no weight from: the stem `MaxPool`, the ResNet-D shortcut `AveragePool`, and
/// the FPN `Resize`.
///
/// These carry no initializer, so the two-way coverage ledger never looks at
/// them - yet `vision::MaxPool` runs at `PoolSpec::half()`, `AvgPool` at an
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

/// One entry of the SCRFD convolution schedule: what the n-th `Conv` node in
/// graph order is, in both weight shape AND geometry.
///
/// The geometry is here because the model builds every conv from `ScrfdConfig`,
/// never from the file - so the file has to be checked against it. A release
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
    /// 1x1 projections are unpadded. The graph holds to that without exception.
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
            // strided - the ResNet-D downsample puts the stride in an
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
    /// scales. 119 canonical tensors from 125 source initializers - the other
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
    loop {
        match w.peek() {
            Some("Conv") => {
                let n = w.next("Conv")?;
                let plan = &schedule[conv_i];
                check_conv(n, w.at() - 1, plan.k(), plan.stride, plan.pad())?;
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
                let at = w.at();
                let n = w.next_any().expect("peeked a node");
                check_structural(n, at)?;
                w.ack_structural(n);
            }
            None => break,
        }
    }
    w.finish(&manifest, "scrfd")
}

/// Read the detector from `dir` and return its imported tensors.
///
/// `dir` comes from a CLI flag, `BRAIN_SCRFD_DIR` or `$BRAIN_TESTDATA` - never a
/// baked-in path.
pub fn import_dir(dir: &std::path::Path) -> Result<Tensors, String> {
    let det = onnx::read_file(dir.join(RELEASE_FILE))?;
    import_scrfd(read::graph(&det)?, &ScrfdConfig::scrfd_10g_bnkps())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest count IS the checkpoint's: 119 canonical tensors from
    /// scrfd's 125 initializers (six structural constants, two shared biases).
    #[test]
    fn manifest_counts_match_the_released_graph() {
        assert_eq!(ScrfdConfig::scrfd_10g_bnkps().tensor_manifest().len(), 119);
        assert_eq!(scrfd_conv_schedule(&ScrfdConfig::scrfd_10g_bnkps()).len(), 58);
    }

    /// No canonical name may appear twice - a duplicate would make the
    /// completeness check pass while one of the two tensors went unwritten.
    #[test]
    fn manifest_names_are_unique() {
        let m = ScrfdConfig::scrfd_10g_bnkps().tensor_manifest();
        let mut names: Vec<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate canonical tensor name");
    }
}
