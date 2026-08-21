// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Checkpoint import from the released `speech_tokenizer_v2.onnx`.
//!
//! The graph walk, the two-way coverage ledger and `Walk::finish` are
//! `onnx::walk` - shared with every other ONNX-only release brain imports
//! (see `crates/scrfd/src/import.rs`, `crates/arcface/src/import.rs`). What
//! lives here is the ARCHITECTURE.
//!
//! # Names survive here - the walk is still positional
//!
//! Unlike `scrfd`/`arcface` (whose exporter folded BatchNorm and lost every
//! initializer name), this exporter kept the module hierarchy on every NODE
//! name (`/blocks.0/attn/query/MatMul`) - but not on most INITIALIZER names
//! (`onnx::MatMul_2228`). So the walk still binds by graph position, using
//! the node name (not the initializer name) to pick the canonical
//! destination. This also means the two-way coverage ledger is doing real
//! work here in a way it could not with a plain name map: `speech_tokenizer_
//! v2.onnx` has 1358 nodes total (`Constant`/`Shape`/`Gather`/`Cast`/`Where`/…
//! building the padding mask and the RoPE table dynamically) and only 102
//! actually carry a weight. The walk visits every one of the 1358 - most
//! through the catch-all branch, which acknowledges any initializer input
//! (there are none on the structural nodes; confirmed by the coverage check
//! itself, which fails loudly the moment that stops being true) - so a
//! future export that folds a new constant into a real weight cannot slip
//! through unnoticed.
//!
//! # Why masking is not implemented
//!
//! `AudioEncoderV2.forward` multiplies by a non-pad mask derived from
//! `feats_length` before both convs, and adds a mask BIAS in attention. The
//! reference golden's `feats_length` equals its mel length exactly (no
//! padding, batch size 1) - confirmed structurally: the traced graph's
//! `conv1`/`conv2` read straight from `feats` with no preceding `Mul`, i.e.
//! `torch.onnx.export`'s constant folding erased the `x * mask` multiply
//! because the traced mask was provably all-ones for that example. brain's
//! forward ([`crate::model::forward`]) makes the same assumption explicit
//! rather than implicit: single utterance, no padding. A batched/padded
//! import is unimplemented, not silently wrong.
//!
//! # Linear weight layout
//!
//! See [`crate::config::S3TokenizerConfig::tensor_manifest`]'s doc comment:
//! every `Linear` here traces to a bare ONNX `MatMul(x, W)`, so `W` is
//! `[in, out]` in the file. This module binds it AS STORED (matching the
//! manifest); [`crate::model::S3TokenizerWeights::from_tensors`] transposes
//! to `[out, in]` once, at load time.

use onnx::onnx::{GraphProto, NodeProto};
use onnx::read;
use onnx::walk::{Manifest, Tensors, Walk};

use crate::config::S3TokenizerConfig;

/// The released file this tokenizer reads.
pub const RELEASE_FILE: &str = "speech_tokenizer_v2.onnx";

const LN_EPS: f32 = 1e-5;

/// Assert a 1D `Conv`'s hyperparameters against what [`crate::model`] will
/// compute. `onnx::walk::check_conv` is 2D-shaped (vision convs); this is its
/// 1D twin - every attribute here is a single-element list, not a pair.
fn check_conv1d(n: &NodeProto, at: usize, k: i64, stride: i64, pad: i64, group: i64) -> Result<(), String> {
    let want = |name: &str, got: Vec<i64>, want: Vec<i64>| -> Result<(), String> {
        if got != want {
            return Err(format!(
                "import(s3tokenizer): Conv at node {at} has {name} {got:?}, expected {want:?}"
            ));
        }
        Ok(())
    };
    want("kernel_shape", read::attr_ints(n, "kernel_shape", &[k]), vec![k])?;
    want("strides", read::attr_ints(n, "strides", &[stride]), vec![stride])?;
    want("pads", read::attr_ints(n, "pads", &[pad, pad]), vec![pad, pad])?;
    want("dilations", read::attr_ints(n, "dilations", &[1]), vec![1])?;
    let g = read::attr_int(n, "group", 1);
    if g != group {
        return Err(format!("import(s3tokenizer): Conv at node {at} has group {g}, expected {group}"));
    }
    Ok(())
}

/// `bn_eval.wgsl`-style epsilon assertion, for `LayerNormalization`:
/// `model::hostmath::layernorm_rows` is called at a hardcoded `1e-5`
/// ([`crate::model`]), so an export at a different epsilon must fail here
/// rather than run silently wrong.
fn check_ln(n: &NodeProto, at: usize) -> Result<(), String> {
    let eps = read::attr_f32(n, "epsilon", LN_EPS);
    if (eps - LN_EPS).abs() > 1e-9 {
        return Err(format!(
            "import(s3tokenizer): LayerNormalization at node {at} has epsilon {eps:e}, expected {LN_EPS:e}"
        ));
    }
    Ok(())
}

/// The block index in a node name like `/blocks.3/attn/query/MatMul`, or
/// `None` for a node outside the `blocks.*` hierarchy (`/conv1/Conv`,
/// `/quantizer/...`).
fn block_of(name: &str) -> Option<usize> {
    name.strip_prefix("/blocks.")?.split('/').next()?.parse().ok()
}

/// The canonical destination of a weight-bearing `MatMul` node, or `None` for
/// the attention score/weighted-sum `MatMul`s (`q @ k`, `softmax @ v`), which
/// carry no initializer at all.
fn matmul_dst(name: &str) -> Option<String> {
    if name == "/quantizer/project_in/MatMul" {
        return Some("quantizer.project_down.weight".into());
    }
    let b = block_of(name)?;
    let suffix = if name.ends_with("/attn/query/MatMul") {
        "attn.query.weight"
    } else if name.ends_with("/attn/key/MatMul") {
        "attn.key.weight"
    } else if name.ends_with("/attn/value/MatMul") {
        "attn.value.weight"
    } else if name.ends_with("/attn/out/MatMul") {
        "attn.out.weight"
    } else if name.ends_with("/mlp/mlp.0/MatMul") {
        "mlp.fc1.weight"
    } else if name.ends_with("/mlp/mlp.2/MatMul") {
        "mlp.fc2.weight"
    } else {
        return None;
    };
    Some(format!("blocks.{b}.{suffix}"))
}

/// The canonical destination of a bias-carrying `Add` node (its first input
/// is the initializer - `in=["onnx::Add_2227", "…MatMul_output_0"]`), or
/// `None` for every other `Add` in the graph (residuals, the RoPE
/// `q*cos + rotate_half(q)*sin` combination, the erf-GELU's internal `1 +
/// erf(...)` - all activation-only). `key` has no bias (Whisper convention,
/// confirmed both in `s3tokenizer/model.py` and by the ONNX graph itself:
/// `/attn/key/MatMul` has no matching `/attn/key/Add`), so it is deliberately
/// absent from this match.
fn add_dst(name: &str) -> Option<String> {
    if name == "/quantizer/project_in/Add" {
        return Some("quantizer.project_down.bias".into());
    }
    let b = block_of(name)?;
    let suffix = if name.ends_with("/attn/query/Add") {
        "attn.query.bias"
    } else if name.ends_with("/attn/value/Add") {
        "attn.value.bias"
    } else if name.ends_with("/attn/out/Add") {
        "attn.out.bias"
    } else if name.ends_with("/mlp/mlp.0/Add") {
        "mlp.fc1.bias"
    } else if name.ends_with("/mlp/mlp.2/Add") {
        "mlp.fc2.bias"
    } else {
        return None;
    };
    Some(format!("blocks.{b}.{suffix}"))
}

fn manifest_shape<'m>(manifest: &'m Manifest, name: &str) -> &'m [usize] {
    &manifest
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("import(s3tokenizer): {name} is not in the manifest"))
        .1
}

/// Import `speech_tokenizer_v2.onnx`.
pub fn import_s3tokenizer(g: &GraphProto, cfg: &S3TokenizerConfig) -> Result<Tensors, String> {
    let mut w = Walk::new(g)?;
    let manifest = cfg.tensor_manifest();
    let d = cfg.n_audio_state as i64;

    loop {
        match w.peek() {
            Some("Conv") => {
                let n = w.next("Conv")?;
                let at = w.at() - 1;
                match n.name.as_str() {
                    "/conv1/Conv" => {
                        check_conv1d(n, at, 3, 2, 1, 1)?;
                        let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
                        w.bind("conv1.weight", &cw, manifest_shape(&manifest, "conv1.weight").to_vec())?;
                        w.bind("conv1.bias", &cb, manifest_shape(&manifest, "conv1.bias").to_vec())?;
                    }
                    "/conv2/Conv" => {
                        check_conv1d(n, at, 3, 2, 1, 1)?;
                        let (cw, cb) = (n.input[1].clone(), n.input[2].clone());
                        w.bind("conv2.weight", &cw, manifest_shape(&manifest, "conv2.weight").to_vec())?;
                        w.bind("conv2.bias", &cb, manifest_shape(&manifest, "conv2.bias").to_vec())?;
                    }
                    other if other.ends_with("/attn/fsmn_block/Conv") => {
                        let b = block_of(other)
                            .ok_or_else(|| format!("import(s3tokenizer): fsmn Conv {other:?} has no block index"))?;
                        // Depthwise: groups == cin == cout == n_audio_state. Padding
                        // is applied by the graph's own `Pad` node (left=right=15),
                        // not by the Conv itself.
                        check_conv1d(n, at, 31, 1, 0, d)?;
                        let cw = n.input[1].clone();
                        let dst = format!("blocks.{b}.attn.fsmn_block.weight");
                        let shape = manifest_shape(&manifest, &dst).to_vec();
                        w.bind(&dst, &cw, shape)?;
                    }
                    other => {
                        return Err(format!("import(s3tokenizer): unexpected Conv node {other:?} at {at}"));
                    }
                }
            }
            Some("MatMul") => {
                let n = w.next("MatMul")?;
                match matmul_dst(&n.name) {
                    Some(dst) => {
                        let src = n.input[1].clone();
                        let shape = manifest_shape(&manifest, &dst).to_vec();
                        w.bind(&dst, &src, shape)?;
                    }
                    None => w.ack_structural(n),
                }
            }
            Some("LayerNormalization") => {
                let n = w.next("LayerNormalization")?;
                check_ln(n, w.at() - 1)?;
                let which = if n.name.ends_with("/attn_ln/LayerNormalization") {
                    "attn_ln"
                } else if n.name.ends_with("/mlp_ln/LayerNormalization") {
                    "mlp_ln"
                } else {
                    return Err(format!("import(s3tokenizer): unexpected LayerNormalization node {:?}", n.name));
                };
                let b = block_of(&n.name)
                    .ok_or_else(|| format!("import(s3tokenizer): LayerNormalization {:?} has no block index", n.name))?;
                let (gw, gb) = (n.input[1].clone(), n.input[2].clone());
                let (dw, db) = (format!("blocks.{b}.{which}.weight"), format!("blocks.{b}.{which}.bias"));
                let (sw, sb) = (manifest_shape(&manifest, &dw).to_vec(), manifest_shape(&manifest, &db).to_vec());
                w.bind(&dw, &gw, sw)?;
                w.bind(&db, &gb, sb)?;
            }
            Some("Add") => {
                let n = w.next("Add")?;
                if let Some(dst) = add_dst(&n.name) {
                    let src = n.input[0].clone();
                    let shape = manifest_shape(&manifest, &dst).to_vec();
                    w.bind(&dst, &src, shape)?;
                } else {
                    w.ack_structural(n);
                }
            }
            Some(_) => {
                let n = w.next_any().expect("peeked a node");
                w.ack_structural(n);
            }
            None => break,
        }
    }
    w.finish(&manifest, "s3tokenizer")
}

/// Read the tokenizer from `dir` and return its imported tensors.
///
/// `dir` comes from a CLI flag, `BRAIN_S3TOKENIZER_V2` or `$BRAIN_TESTDATA` -
/// never a baked-in path.
pub fn import_dir(dir: &std::path::Path) -> Result<Tensors, String> {
    let m = onnx::read_file(dir.join(RELEASE_FILE))?;
    import_s3tokenizer(read::graph(&m)?, &S3TokenizerConfig::v2())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_of_reads_the_index_out_of_a_node_name() {
        assert_eq!(block_of("/blocks.0/attn/query/MatMul"), Some(0));
        assert_eq!(block_of("/blocks.5/mlp/mlp.2/Add"), Some(5));
        assert_eq!(block_of("/conv1/Conv"), None);
        assert_eq!(block_of("/quantizer/project_in/MatMul"), None);
    }

    #[test]
    fn matmul_dst_ignores_the_attention_score_matmuls() {
        assert_eq!(matmul_dst("/blocks.0/attn/query/MatMul"), Some("blocks.0.attn.query.weight".into()));
        assert_eq!(matmul_dst("/blocks.2/attn/key/MatMul"), Some("blocks.2.attn.key.weight".into()));
        assert_eq!(matmul_dst("/blocks.0/attn/MatMul"), None);
        assert_eq!(matmul_dst("/blocks.0/attn/MatMul_1"), None);
        assert_eq!(matmul_dst("/quantizer/project_in/MatMul"), Some("quantizer.project_down.weight".into()));
    }

    #[test]
    fn add_dst_excludes_the_key_projection_and_residual_adds() {
        assert_eq!(add_dst("/blocks.1/attn/query/Add"), Some("blocks.1.attn.query.bias".into()));
        assert_eq!(add_dst("/blocks.1/attn/key/Add"), None); // key has no bias
        assert_eq!(add_dst("/blocks.0/attn/Add"), None); // fsmn residual
        assert_eq!(add_dst("/blocks.0/attn/Add_3"), None); // out + fsm_memory
        assert_eq!(add_dst("/blocks.0/Add"), None); // block residual
        assert_eq!(add_dst("/blocks.0/mlp/mlp.1/Add"), None); // erf-GELU internal
    }
}
