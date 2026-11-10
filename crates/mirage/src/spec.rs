// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The canonical tensor manifest: every tensor a Mirage checkpoint must hold,
//! with its Flax name and shape, derived from a [`MirageConfig`].
//!
//! The names are the safetensors file's own (`body_3/hconv/gate_conv`,
//! `stack/layers/block/self_attn/q_proj/kernel`), because brain reads the
//! released files directly. Dense kernels are `[in, out]`, convolutions HWIO,
//! and the decoder's per-layer tensors are stacked along a leading layer axis.
//! Both the strict importer and the `.cact` mapping validate against this one
//! manifest.
//!
//! Swedish Embedded AB implements solutions for verifying model checkpoints
//! against an exact tensor contract for its clients. If your team needs
//! expertise in checkpoint integrity and import pipelines then you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::collections::BTreeMap;

use crate::config::{
    MirageConfig, BOX_ENCODING_DIM, DENSE_BLOCKS, DENSE_OUT_CHANNELS, ENGRAM_TAPS, HADAMARD_COND_RANK, MASK_BRANCH_BLOCKS, MASK_RANK,
    READER_CLASSES, READER_MIX_KERNEL, READER_STAGES,
};

/// One tensor the checkpoint must contain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec {
    /// Flax name as stored in the safetensors file.
    pub name: String,
    /// Required shape.
    pub shape: Vec<usize>,
    /// Present only for training; absent from the deployment `.cact`.
    pub training_only: bool,
}

impl TensorSpec {
    /// Element count.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Whether a tensor exists only for training: the multi-token-prediction block,
/// its combiner and norms, and the name head.
pub fn is_training_only(name: &str) -> bool {
    name.starts_with("mtp_block/") || matches!(name, "mtp_combine/kernel" | "mtp_emb_norm/scale" | "mtp_final_norm/scale" | "name_head/kernel")
}

/// The full set of tensors for one configuration, ordered by name.
#[derive(Debug, Clone)]
pub struct Manifest {
    tensors: BTreeMap<String, TensorSpec>,
}

impl Manifest {
    /// Derive the manifest from a validated configuration.
    pub fn new(c: &MirageConfig) -> Manifest {
        let mut b = Builder::default();
        stem(&mut b, c);
        body(&mut b, c);
        attention_top(&mut b, c);
        encoder_tail(&mut b, c);
        dense_head(&mut b, c);
        slots(&mut b, c);
        reader(&mut b, c);
        decoder(&mut b, c);
        training_heads(&mut b, c);
        Manifest { tensors: b.tensors }
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the manifest is empty (never true for a valid configuration).
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Total parameters over all tensors.
    pub fn param_count(&self) -> usize {
        self.tensors.values().map(TensorSpec::numel).sum()
    }

    /// The tensor named `name`.
    pub fn get(&self, name: &str) -> Option<&TensorSpec> {
        self.tensors.get(name)
    }

    /// All tensors, ordered by name.
    pub fn iter(&self) -> impl Iterator<Item = &TensorSpec> {
        self.tensors.values()
    }

    /// The tensors that also exist in the deployment `.cact`.
    pub fn deployment(&self) -> impl Iterator<Item = &TensorSpec> {
        self.iter().filter(|t| !t.training_only)
    }
}

#[derive(Default)]
struct Builder {
    tensors: BTreeMap<String, TensorSpec>,
}

impl Builder {
    fn add(&mut self, name: impl Into<String>, shape: &[usize]) {
        let name = name.into();
        let spec = TensorSpec { training_only: is_training_only(&name), shape: shape.to_vec(), name: name.clone() };
        let previous = self.tensors.insert(name.clone(), spec);
        assert!(previous.is_none(), "manifest derives `{name}` twice");
    }
}

/// `[3,3,1,c]`: a 3x3 depthwise kernel in HWIO layout.
fn depthwise(k: usize, c: usize) -> [usize; 4] {
    [k, k, 1, c]
}

fn stem(b: &mut Builder, c: &MirageConfig) {
    b.add("stem/w0", &[3, 3, 3, c.stem[0]]);
    b.add("stem/norm_0/scale", &[c.stem[0]]);
    for k in 1..c.stem.len() {
        let (cin, cout) = (c.stem[k - 1], c.stem[k]);
        b.add(format!("stem/dw_{k}"), &depthwise(3, cin));
        b.add(format!("stem/pw_{k}/kernel"), &[cin, cout]);
        b.add(format!("stem/norm_{k}/scale"), &[cout]);
    }
}

/// The three HConvGLU band convolutions and their input norm.
fn hconv(b: &mut Builder, prefix: &str, width: usize, kernel: usize) {
    b.add(format!("{prefix}/hconv_norm/scale"), &[width]);
    b.add(format!("{prefix}/hconv/gate_conv"), &[width, kernel]);
    b.add(format!("{prefix}/hconv/gate_bias"), &[width]);
    b.add(format!("{prefix}/hconv/value_conv"), &[width, kernel]);
    b.add(format!("{prefix}/hconv/out_conv"), &[width, kernel]);
}

fn body(b: &mut Builder, c: &MirageConfig) {
    for i in 0..c.body_layers {
        let p = format!("body_{i}");
        b.add(format!("{p}/mix_norm/scale"), &[c.dim]);
        b.add(format!("{p}/mix/dw_value"), &depthwise(c.mix_kernel, c.dim));
        b.add(format!("{p}/mix/b_value"), &[c.dim]);
        b.add(format!("{p}/mix/dw_gate"), &depthwise(c.mix_kernel, c.dim));
        b.add(format!("{p}/mix/b_gate"), &[c.dim]);
        b.add(format!("{p}/mix/dw_down"), &depthwise(c.mix_kernel, c.dim));
        hconv(b, &p, c.dim, c.hconv_kernel);
        b.add(format!("{p}/grn/gamma"), &[c.dim]);
        b.add(format!("{p}/grn/beta"), &[c.dim]);
    }
}

fn attention_top(b: &mut Builder, c: &MirageConfig) {
    let (q, kv, v) = (c.attn_heads * c.attn_qk_dim, c.attn_kv_heads * c.attn_qk_dim, c.attn_kv_heads * c.attn_v_dim);
    let gate = c.attn_heads * c.attn_v_dim;
    for i in 0..c.attn_layers {
        let p = format!("top_{i}");
        b.add(format!("{p}/attn_norm/scale"), &[c.dim]);
        b.add(format!("{p}/self_attn/q_proj/kernel"), &[c.dim, q]);
        b.add(format!("{p}/self_attn/k_proj/kernel"), &[c.dim, kv]);
        b.add(format!("{p}/self_attn/v_proj/kernel"), &[c.dim, v]);
        b.add(format!("{p}/self_attn/gate_proj/kernel"), &[c.dim, gate]);
        b.add(format!("{p}/self_attn/out_proj/kernel"), &[gate, c.dim]);
        b.add(format!("{p}/self_attn/q_norm/scale"), &[c.attn_qk_dim]);
        b.add(format!("{p}/self_attn/k_norm/scale"), &[c.attn_qk_dim]);
        b.add(format!("{p}/post_attn_norm/scale"), &[c.dim]);
        b.add(format!("{p}/attn_gate"), &[1]);
        hconv(b, &p, c.dim, c.hconv_kernel);
    }
}

fn encoder_tail(b: &mut Builder, c: &MirageConfig) {
    b.add("enc_norm/scale", &[c.dim]);
    b.add("registers", &[c.registers, c.dim]);
    for i in 0..c.attn_layers {
        b.add(format!("level_norm_{i}/scale"), &[c.dim]);
    }
    b.add("level_gates", &[c.attn_layers, c.dim]);
}

/// A trunk of `n` blocks, each a channel norm, a depthwise 3x3 and a pointwise mix.
fn dense_trunk(b: &mut Builder, prefix: &str, norm: &str, dw: &str, pw: &str, n: usize, d: usize) {
    for i in 0..n {
        b.add(format!("{prefix}/{norm}_{i}/scale"), &[d]);
        b.add(format!("{prefix}/{dw}_{i}"), &depthwise(3, d));
        b.add(format!("{prefix}/{pw}_{i}/kernel"), &[d, d]);
    }
}

fn dense_head(b: &mut Builder, c: &MirageConfig) {
    let d = c.dense_dim;
    let (edge, fine, mid) = c.dense_levels();
    b.add("dense/coarse/kernel", &[c.dim, d]);
    b.add("dense/fine/kernel", &[c.stem[fine], d]);
    b.add("dense/mid/kernel", &[c.stem[mid], d]);
    b.add("dense/neck_norm/scale", &[d]);
    b.add("dense/neck_dw", &depthwise(3, d));
    b.add("dense/neck_pw/kernel", &[d, d]);
    dense_trunk(b, "dense", "block_norm", "dw", "pw", DENSE_BLOCKS, d);
    b.add("dense/final_norm/scale", &[d]);
    b.add("dense/out/kernel", &[d, DENSE_OUT_CHANNELS]);
    b.add("dense/bias", &[DENSE_OUT_CHANNELS]);
    b.add("dense/edge/kernel", &[c.stem[edge], d]);
    dense_trunk(b, "dense", "proto_norm", "proto_dw", "proto_pw", MASK_BRANCH_BLOCKS, d);
    b.add("dense/proto_final_norm/scale", &[d]);
    b.add("dense/protos/kernel", &[d, MASK_RANK]);
    dense_trunk(b, "dense", "coef_norm", "coef_dw", "coef_pw", MASK_BRANCH_BLOCKS, d);
    b.add("dense/coef_final_norm/scale", &[d]);
    b.add("dense/coef/kernel", &[d, MASK_RANK]);
}

fn slots(b: &mut Builder, c: &MirageConfig) {
    b.add("slot_proj/kernel", &[c.dense_dim + BOX_ENCODING_DIM, c.dim]);
    b.add("slot_norm/scale", &[c.dim]);
    b.add("slot_pool/kernel", &[c.dim, c.dim]);
}

fn reader(b: &mut Builder, c: &MirageConfig) {
    let widths = [READER_STAGES[0], READER_STAGES[1], READER_STAGES[2], c.reader_dim];
    b.add("reader/w0", &[3, 3, 3, widths[0]]);
    for (k, &w) in widths.iter().enumerate() {
        b.add(format!("reader/stage_norm_{k}/scale"), &[w]);
    }
    for k in 0..READER_STAGES.len() {
        b.add(format!("reader/dw_{k}"), &depthwise(3, widths[k]));
        b.add(format!("reader/pw_{k}/kernel"), &[widths[k], widths[k + 1]]);
    }
    let r = c.reader_dim;
    for i in 0..c.reader_blocks {
        b.add(format!("reader/norm_{i}/scale"), &[r]);
        b.add(format!("reader/mix_{i}"), &[1, READER_MIX_KERNEL, 1, r]);
        b.add(format!("reader/up_{i}/kernel"), &[r, 2 * r]);
        b.add(format!("reader/down_{i}/kernel"), &[r, r]);
    }
    b.add("reader/final_norm/scale", &[r]);
    b.add("reader/out/kernel", &[r, READER_CLASSES]);
    b.add("reader/out/bias", &[READER_CLASSES]);
}

/// One decoder block's tensors. `lead` is the stacked layer count, or `None`
/// for the single unstacked block of the multi-token-prediction head.
fn decoder_block(b: &mut Builder, prefix: &str, lead: Option<usize>, c: &MirageConfig) {
    let heads = c.heads();
    let (q, kv, v) = (heads * c.qk_dim, c.kv_heads * c.qk_dim, c.kv_heads * c.v_dim);
    let width = heads * c.v_dim;
    let n = c.hadamard_width();
    let (fa, fb) = c.hadamard_factors();
    let mut add = |name: String, shape: &[usize]| {
        let mut full = Vec::with_capacity(shape.len() + 1);
        full.extend(lead);
        full.extend_from_slice(shape);
        b.add(format!("{prefix}/{name}"), &full);
    };
    add("ZCRMSNorm_0/scale".into(), &[c.dim]);
    add("self_attn/q_proj/kernel".into(), &[c.dim, q]);
    add("self_attn/k_proj/kernel".into(), &[c.dim, kv]);
    add("self_attn/v_proj/kernel".into(), &[c.dim, v]);
    add("self_attn/q_taps".into(), &[c.taps, q]);
    add("self_attn/k_taps".into(), &[c.taps, kv]);
    add("self_attn/v_taps".into(), &[c.taps, v]);
    add("self_attn/q_norm/scale".into(), &[c.qk_dim]);
    add("self_attn/k_norm/scale".into(), &[c.qk_dim]);
    add("self_attn/gate_proj/kernel".into(), &[c.dim, width]);
    add("self_attn/out_proj/kernel".into(), &[width, c.dim]);
    add("post_attn_norm/scale".into(), &[c.dim]);
    // A per-layer scalar gate is a `[1]` tensor unstacked and `[L]` stacked.
    if lead.is_some() {
        add("attn_gate".into(), &[]);
        add("cross_gate".into(), &[]);
    } else {
        add("attn_gate".into(), &[1]);
        add("cross_gate".into(), &[1]);
    }
    add("pre_hada_norm/scale".into(), &[c.dim]);
    for name in ["d1", "d2", "b2", "d3", "d4"] {
        add(format!("hadamard_mlp/{name}"), &[n]);
    }
    for stage in 1..=3 {
        add(format!("hadamard_mlp/w{stage}a"), &[fa, fa]);
        add(format!("hadamard_mlp/w{stage}b"), &[fb, fb]);
    }
    add("hadamard_mlp/cond_v".into(), &[c.dim, HADAMARD_COND_RANK]);
    add("hadamard_mlp/cond_u".into(), &[HADAMARD_COND_RANK, n]);
    add("cross_norm/scale".into(), &[c.dim]);
    add("cross_attn/q_proj/kernel".into(), &[c.dim, q]);
    add("cross_attn/k_proj/kernel".into(), &[c.dim, q]);
    add("cross_attn/v_proj/kernel".into(), &[c.dim, width]);
    add("cross_attn/gate_proj/kernel".into(), &[c.dim, width]);
    add("cross_attn/out_proj/kernel".into(), &[width, c.dim]);
    add("cross_attn/q_norm/scale".into(), &[c.qk_dim]);
    add("cross_attn/k_norm/scale".into(), &[c.qk_dim]);
    add("post_cross_norm/scale".into(), &[c.dim]);
}

fn decoder(b: &mut Builder, c: &MirageConfig) {
    b.add("embedding/embedding", &[c.embedding_rows(), c.dim]);
    b.add("pe_gate", &[1]);
    decoder_block(b, "stack/layers/block", Some(c.dec_layers), c);
    b.add("stack/final_norm/scale", &[c.dim]);

    let (l, n) = (c.dec_layers, c.lanes);
    let lane_width = n * c.dim;
    for name in ["a_pre", "a_post", "a_res"] {
        b.add(format!("stack/mhc_{name}"), &[l]);
    }
    b.add("stack/mhc_b_pre", &[l, n]);
    b.add("stack/mhc_b_post", &[l, n]);
    b.add("stack/mhc_b_res", &[l, n, n]);
    b.add("stack/mhc_phi_pre", &[l, lane_width, n]);
    b.add("stack/mhc_phi_post", &[l, lane_width, n]);
    b.add("stack/mhc_phi_res", &[l, lane_width, n * n]);

    for i in 0..c.sites.len() {
        b.add(format!("engrams_{i}/embedding"), &[c.engram_tables(), c.slots, c.engram_sub_dim()]);
        b.add(format!("engrams_{i}/key_proj/kernel"), &[c.dim, c.dim]);
        b.add(format!("engrams_{i}/value_proj/kernel"), &[c.dim, c.dim]);
        b.add(format!("engrams_{i}/taps"), &[ENGRAM_TAPS, c.dim]);
    }
}

fn training_heads(b: &mut Builder, c: &MirageConfig) {
    decoder_block(b, "mtp_block", None, c);
    b.add("mtp_combine/kernel", &[2 * c.dim, c.dim]);
    b.add("mtp_emb_norm/scale", &[c.dim]);
    b.add("mtp_final_norm/scale", &[c.dim]);
    b.add("name_head/kernel", &[c.dim, c.dim]);
}
