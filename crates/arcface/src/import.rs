// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import from the released insightface `glintr100.onnx`.
//!
//! The graph walk, the two-way coverage ledger (every manifest tensor produced
//! exactly once; every source initializer consumed at least once) and the
//! `Conv` hyper-parameter assertion are `onnx::walk` - shared with every other
//! ONNX-only release brain imports. What lives here is the ARCHITECTURE: which
//! op sequence this graph has and which canonical name each binding gets.
//!
//! # Why the walk is topological, not a name remap
//!
//! Every other brain importer maps source names to brain names. That is
//! impossible here: **the ONNX exporter folded BatchNorm into the convolutions
//! and the folded tensors lost their names.** In `glintr100.onnx`, 256 of the 462
//! initializers are called `1335`, `1336`, `1643`… - bare SSA value numbers. Only
//! the 206 tensors that survived folding (the `bn1`/`bn2`/`features`
//! BatchNorms and `fc`) still carry `layer2.3.bn1.weight`-style names.
//!
//! So the source of truth for "which tensor is this" is the graph's **topology**:
//! the n-th `Conv` node in graph order is a known convolution of a known
//! architecture. The walk asserts the op sequence it expects and binds weights
//! positionally, then checks every shape. That is a stronger check than a name
//! match, not a weaker one - a name map cannot notice that the graph has 48
//! residual adds instead of 49.

use onnx::onnx::{GraphProto, NodeProto};
use onnx::read;
use onnx::walk::{check_conv, Manifest, Tensors, Walk};

use crate::config::ArcFaceConfig;

/// The released file this embedder reads, by its antelopev2 name.
pub const RELEASE_FILE: &str = "glintr100.onnx";

/// `bn_eval.wgsl` hardcodes `eps = 1e-5`, so an imported BatchNorm whose graph
/// says otherwise would run with the wrong epsilon and no error. The released
/// graph exports `epsilon = 1e-5`; this is the assertion that keeps that true.
const BN_EPS: f32 = 1e-5;

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
    check_conv(n, w.at() - 1, 3, 1, 1)?;
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
            check_bn(n, w.at() - 1)?;
            let ins: Vec<String> = n.input[1..5].to_vec();
            for (dst, src) in bn_names(&format!("{p}.bn1")).iter().zip(&ins) {
                w.bind(dst, src, vec![bn_c])?;
            }

            let n = w.next("Conv")?;
            check_conv(n, w.at() - 1, 3, 1, 1)?;
            let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
            w.bind(&format!("{p}.conv1.weight"), &cw, vec![cout, bn_c, 3, 3])?;
            w.bind(&format!("{p}.conv1.bias"), &cb, vec![cout])?;

            let n = w.next("PRelu")?;
            let slope = n.input[1].clone();
            w.bind(&format!("{p}.prelu.weight"), &slope, vec![cout])?;

            let n = w.next("Conv")?;
            check_conv(n, w.at() - 1, 3, stride, 1)?;
            let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
            w.bind(&format!("{p}.conv2.weight"), &cw, vec![cout, cout, 3, 3])?;
            w.bind(&format!("{p}.conv2.bias"), &cb, vec![cout])?;

            if b == 0 {
                let n = w.next("Conv")?;
                check_conv(n, w.at() - 1, 1, stride, 0)?;
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
    check_bn(n, w.at() - 1)?;
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
    check_bn(n, w.at() - 1)?;
    let ins: Vec<String> = n.input[1..5].to_vec();
    for (dst, src) in bn_names("features").iter().zip(&ins) {
        w.bind(dst, src, vec![e])?;
    }

    w.finish(&manifest, "arcface")
}

/// Read the embedder from `dir` and return its imported tensors.
///
/// `dir` comes from a CLI flag, `BRAIN_ARCFACE_DIR` or `$BRAIN_TESTDATA` - never
/// a baked-in path.
pub fn import_dir(dir: &std::path::Path) -> Result<Tensors, String> {
    let rec = onnx::read_file(dir.join(RELEASE_FILE))?;
    import_arcface(read::graph(&rec)?, &ArcFaceConfig::iresnet100())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest count IS the checkpoint's initializer count: 462.
    #[test]
    fn manifest_counts_match_the_released_graph() {
        assert_eq!(ArcFaceConfig::iresnet100().tensor_manifest().len(), 462);
    }

    /// No canonical name may appear twice - a duplicate would make the
    /// completeness check pass while one of the two tensors went unwritten.
    #[test]
    fn manifest_names_are_unique() {
        let m = ArcFaceConfig::iresnet100().tensor_manifest();
        let mut names: Vec<&str> = m.iter().map(|(n, _)| n.as_str()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "duplicate canonical tensor name");
    }
}
