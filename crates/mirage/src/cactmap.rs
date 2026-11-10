// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! What each positional `.cact` entry is.
//!
//! The container carries no names: entry `i` is whatever the exporter wrote
//! `i`-th. [`layout`] derives that order from the configuration, layer-major:
//! the embedding, then each decoder layer's self-attention and Hadamard-MLP
//! tensors, the hyper-connection blocks, the Hadamard permutations, the engram
//! sites, the final norm, the vision manifest, each layer's cross-attention,
//! and finally the vision encoder, the heads, the reader, the OCR charset and
//! the tokenizer blob.
//!
//! Each entry also records how its stored layout relates to the canonical
//! (Flax) tensor of [`crate::spec`]: quantised matrices are stored `[out, in]`
//! and are transposed back to `[in, out]`; the hyper-connection routing
//! matrices are stored as `[layers * lanes, width]`; convolution kernels and
//! tables are plain reshapes; stacked decoder tensors arrive one layer at a
//! time. [`CactModel`] validates the header and every entry against the
//! configuration before handing anything out.
//!
//! Swedish Embedded AB implements solutions for deploying quantised models from
//! positional binary containers for its clients. If your team needs expertise
//! in model container formats then you can procure our services by sending an
//! email to info@swedishembedded.com.

use std::collections::BTreeMap;

use crate::cact::{CactError, CactFile, EntryDtype};
use crate::config::{MirageConfig, DENSE_BLOCKS, MASK_BRANCH_BLOCKS, READER_CLASSES, READER_STAGES};
use crate::import::Tensor;
use crate::spec::Manifest;

/// Values in the vision manifest entry.
pub const VISION_MANIFEST_LEN: usize = 36;

/// How a stored entry relates to its canonical tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Same element order; only the shape may differ (a depthwise kernel
    /// stored as `[k*k, channels]`, a table stored with its tables flattened).
    Same,
    /// Stored `[out, in]`, canonical `[in, out]`.
    Transpose,
    /// Stored `[layers * lanes, width]`, canonical `[layers, width, lanes]`.
    MhcPhi,
}

/// What an entry holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotKind {
    /// A canonical tensor, or one layer's slice of a stacked one.
    Tensor {
        /// Canonical Flax name.
        name: String,
        /// The layer this entry supplies when the canonical tensor is stacked.
        layer: Option<usize>,
        /// Stored layout.
        layout: Layout,
    },
    /// Hadamard-MLP permutation 1 or 2, as `f32` indices.
    HadamardPermutation(u8),
    /// The 36 vision constants.
    VisionManifest,
    /// The OCR charset as `f32` code points (label `i` is `charset[i - 1]`).
    Charset,
    /// The tokenizer blob.
    Tokenizer,
}

/// One directory position and its meaning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Slot {
    /// Directory index.
    pub entry: usize,
    /// What it holds.
    pub kind: SlotKind,
}

struct Planner {
    slots: Vec<Slot>,
}

impl Planner {
    fn push(&mut self, kind: SlotKind) {
        let entry = self.slots.len();
        self.slots.push(Slot { entry, kind });
    }

    fn tensor(&mut self, name: impl Into<String>, layer: Option<usize>, layout: Layout) {
        self.push(SlotKind::Tensor { name: name.into(), layer, layout });
    }

    /// An entry stored in the canonical element order.
    fn same(&mut self, name: impl Into<String>) {
        self.tensor(name, None, Layout::Same);
    }

    /// A quantised-style matrix stored `[out, in]`.
    fn matrix(&mut self, name: impl Into<String>) {
        self.tensor(name, None, Layout::Transpose);
    }
}

const BLOCK: &str = "stack/layers/block";

/// One decoder layer's self-attention and Hadamard-MLP entries, in file order.
fn self_block_entries(p: &mut Planner, layer: usize) {
    let mut at = |suffix: &str, layout: Layout| p.tensor(format!("{BLOCK}/{suffix}"), Some(layer), layout);
    at("ZCRMSNorm_0/scale", Layout::Same);
    for proj in ["q", "k", "v"] {
        at(&format!("self_attn/{proj}_proj/kernel"), Layout::Transpose);
    }
    for taps in ["q_taps", "k_taps", "v_taps"] {
        at(&format!("self_attn/{taps}"), Layout::Same);
    }
    at("self_attn/q_norm/scale", Layout::Same);
    at("self_attn/k_norm/scale", Layout::Same);
    at("self_attn/gate_proj/kernel", Layout::Transpose);
    at("self_attn/out_proj/kernel", Layout::Transpose);
    at("post_attn_norm/scale", Layout::Same);
    at("attn_gate", Layout::Same);
    at("pre_hada_norm/scale", Layout::Same);
    for name in ["d1", "d2", "b2", "d3", "d4", "w1a", "w1b", "w2a", "w2b", "w3a", "w3b", "cond_v", "cond_u"] {
        at(&format!("hadamard_mlp/{name}"), Layout::Same);
    }
}

fn cross_block_entries(p: &mut Planner, layer: usize) {
    let mut at = |suffix: &str, layout: Layout| p.tensor(format!("{BLOCK}/{suffix}"), Some(layer), layout);
    at("cross_norm/scale", Layout::Same);
    for proj in ["q", "k", "v", "gate", "out"] {
        at(&format!("cross_attn/{proj}_proj/kernel"), Layout::Transpose);
    }
    at("cross_attn/q_norm/scale", Layout::Same);
    at("cross_attn/k_norm/scale", Layout::Same);
    at("post_cross_norm/scale", Layout::Same);
    at("cross_gate", Layout::Same);
}

fn hconv_entries(p: &mut Planner, prefix: &str) {
    for suffix in ["hconv_norm/scale", "hconv/gate_conv", "hconv/gate_bias", "hconv/value_conv", "hconv/out_conv"] {
        p.same(format!("{prefix}/{suffix}"));
    }
}

fn dense_trunk_entries(p: &mut Planner, norm: &str, dw: &str, pw: &str, n: usize) {
    for i in 0..n {
        p.same(format!("dense/{norm}_{i}/scale"));
        p.same(format!("dense/{dw}_{i}"));
        p.same(format!("dense/{pw}_{i}/kernel"));
    }
}

/// The positional meaning of every directory entry for `c`.
pub fn layout(c: &MirageConfig) -> Vec<Slot> {
    let mut p = Planner { slots: Vec::new() };
    p.same("embedding/embedding");
    for layer in 0..c.dec_layers {
        self_block_entries(&mut p, layer);
    }
    for name in ["a_pre", "a_post", "a_res", "b_pre", "b_post", "b_res"] {
        p.same(format!("stack/mhc_{name}"));
    }
    for name in ["phi_pre", "phi_post", "phi_res"] {
        p.tensor(format!("stack/mhc_{name}"), None, Layout::MhcPhi);
    }
    p.push(SlotKind::HadamardPermutation(1));
    p.push(SlotKind::HadamardPermutation(2));
    for site in 0..c.sites.len() {
        p.same(format!("engrams_{site}/embedding"));
        p.matrix(format!("engrams_{site}/key_proj/kernel"));
        p.matrix(format!("engrams_{site}/value_proj/kernel"));
        p.same(format!("engrams_{site}/taps"));
    }
    p.same("stack/final_norm/scale");
    p.push(SlotKind::VisionManifest);
    for layer in 0..c.dec_layers {
        cross_block_entries(&mut p, layer);
    }
    p.same("pe_gate");

    // Vision encoder.
    p.same("stem/w0");
    p.same("stem/norm_0/scale");
    for k in 1..c.stem.len() {
        p.same(format!("stem/dw_{k}"));
        p.same(format!("stem/pw_{k}/kernel"));
        p.same(format!("stem/norm_{k}/scale"));
    }
    for i in 0..c.body_layers {
        let b = format!("body_{i}");
        p.same(format!("{b}/mix_norm/scale"));
        for name in ["dw_value", "b_value", "dw_gate", "b_gate", "dw_down"] {
            p.same(format!("{b}/mix/{name}"));
        }
        hconv_entries(&mut p, &b);
        p.same(format!("{b}/grn/gamma"));
        p.same(format!("{b}/grn/beta"));
    }
    for i in 0..c.attn_layers {
        let t = format!("top_{i}");
        p.same(format!("{t}/attn_norm/scale"));
        for proj in ["q", "k", "v", "gate", "out"] {
            p.matrix(format!("{t}/self_attn/{proj}_proj/kernel"));
        }
        p.same(format!("{t}/self_attn/q_norm/scale"));
        p.same(format!("{t}/self_attn/k_norm/scale"));
        p.same(format!("{t}/post_attn_norm/scale"));
        p.same(format!("{t}/attn_gate"));
        hconv_entries(&mut p, &t);
    }
    p.same("enc_norm/scale");
    p.same("registers");
    for i in 0..c.attn_layers {
        p.same(format!("level_norm_{i}/scale"));
    }
    p.same("level_gates");

    // Heads.
    for name in ["coarse", "fine", "mid"] {
        p.same(format!("dense/{name}/kernel"));
    }
    p.same("dense/neck_norm/scale");
    p.same("dense/neck_dw");
    p.same("dense/neck_pw/kernel");
    dense_trunk_entries(&mut p, "block_norm", "dw", "pw", DENSE_BLOCKS);
    p.same("dense/final_norm/scale");
    p.same("dense/out/kernel");
    p.same("dense/bias");
    p.same("dense/edge/kernel");
    dense_trunk_entries(&mut p, "proto_norm", "proto_dw", "proto_pw", MASK_BRANCH_BLOCKS);
    p.same("dense/proto_final_norm/scale");
    p.same("dense/protos/kernel");
    dense_trunk_entries(&mut p, "coef_norm", "coef_dw", "coef_pw", MASK_BRANCH_BLOCKS);
    p.same("dense/coef_final_norm/scale");
    p.same("dense/coef/kernel");
    p.same("slot_proj/kernel");
    p.same("slot_norm/scale");
    p.matrix("slot_pool/kernel");

    // Reader.
    p.same("reader/w0");
    p.same("reader/stage_norm_0/scale");
    for k in 0..READER_STAGES.len() {
        p.same(format!("reader/dw_{k}"));
        p.matrix(format!("reader/pw_{k}/kernel"));
        p.same(format!("reader/stage_norm_{}/scale", k + 1));
    }
    for i in 0..c.reader_blocks {
        p.same(format!("reader/norm_{i}/scale"));
        p.same(format!("reader/mix_{i}"));
        p.matrix(format!("reader/up_{i}/kernel"));
        p.matrix(format!("reader/down_{i}/kernel"));
    }
    p.same("reader/final_norm/scale");
    p.matrix("reader/out/kernel");
    p.same("reader/out/bias");
    p.push(SlotKind::Charset);
    p.push(SlotKind::Tokenizer);
    p.slots
}

/// A `.cact` file validated against a configuration.
#[derive(Debug)]
pub struct CactModel {
    file: CactFile,
    config: MirageConfig,
    manifest: Manifest,
    slots: Vec<Slot>,
}

fn header_mismatch(field: &str, found: impl std::fmt::Display, expected: impl std::fmt::Display) -> CactError {
    CactError::Header(format!("{field} is {found}, the configuration implies {expected}"))
}

impl CactModel {
    /// Validate `file` against `config`: the header geometry, the entry count
    /// and every entry's dtype and size.
    pub fn new(file: CactFile, config: MirageConfig) -> Result<CactModel, CactError> {
        config.validate().map_err(|e| CactError::Header(e.to_string()))?;
        check_header(&file, &config)?;
        let slots = layout(&config);
        if file.entries().len() != slots.len() {
            return Err(CactError::Header(format!("the file has {} entries, the configuration implies {}", file.entries().len(), slots.len())));
        }
        let manifest = Manifest::new(&config);
        let model = CactModel { file, config, manifest, slots };
        for slot in &model.slots {
            model.check_slot(slot)?;
        }
        Ok(model)
    }

    /// The underlying container.
    pub fn file(&self) -> &CactFile {
        &self.file
    }

    /// The configuration the file was validated against.
    pub fn config(&self) -> &MirageConfig {
        &self.config
    }

    /// The positional map.
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// The canonical shape of the unit one entry supplies (a stacked tensor's
    /// per-layer slice, without the layer axis).
    fn unit_shape(&self, name: &str, layer: Option<usize>) -> Vec<usize> {
        let spec = self.manifest.get(name).expect("the layout only names manifest tensors");
        if layer.is_some() {
            spec.shape[1..].to_vec()
        } else {
            spec.shape.clone()
        }
    }

    fn check_slot(&self, slot: &Slot) -> Result<(), CactError> {
        let e = &self.file.entries()[slot.entry];
        let bad = |reason: String| Err(CactError::Entry { index: slot.entry, reason });
        match &slot.kind {
            SlotKind::Tensor { name, layer, layout } => {
                if e.dtype == EntryDtype::Raw {
                    return bad(format!("`{name}` is stored as a raw blob"));
                }
                let unit = self.unit_shape(name, *layer);
                let numel: usize = unit.iter().product();
                let expected_stored: Vec<usize> = match layout {
                    Layout::Same => return if e.numel() == numel { Ok(()) } else { bad(format!("`{name}` has {} elements ({:?}), expected {numel} for {unit:?}", e.numel(), e.shape)) },
                    Layout::Transpose => unit.iter().rev().copied().collect(),
                    Layout::MhcPhi => vec![unit[0] * unit[2], unit[1]],
                };
                if e.shape != expected_stored {
                    return bad(format!("`{name}` is stored as {:?}, expected {expected_stored:?} for canonical {unit:?}", e.shape));
                }
                Ok(())
            }
            SlotKind::HadamardPermutation(k) => {
                let n = self.config.hadamard_width();
                if e.dtype != EntryDtype::Fp32 || e.shape != [n] {
                    return bad(format!("Hadamard permutation {k} must be {n} fp32 values, found {:?} {:?}", e.dtype, e.shape));
                }
                Ok(())
            }
            SlotKind::VisionManifest => {
                if e.dtype != EntryDtype::Fp32 || e.shape != [VISION_MANIFEST_LEN] {
                    return bad(format!("the vision manifest must be {VISION_MANIFEST_LEN} fp32 values, found {:?} {:?}", e.dtype, e.shape));
                }
                Ok(())
            }
            SlotKind::Charset => {
                if e.dtype != EntryDtype::Fp32 || e.shape != [READER_CLASSES - 1] {
                    return bad(format!("the charset must be {} fp32 code points, found {:?} {:?}", READER_CLASSES - 1, e.dtype, e.shape));
                }
                Ok(())
            }
            SlotKind::Tokenizer => {
                if e.dtype != EntryDtype::Raw {
                    return bad(format!("the tokenizer must be a raw blob, found {:?}", e.dtype));
                }
                Ok(())
            }
        }
    }

    /// One entry decoded and put into canonical element order (the unit's
    /// canonical shape; for a stacked tensor, one layer's slice).
    pub fn slot_tensor(&self, slot: &Slot) -> Result<Vec<f32>, CactError> {
        let SlotKind::Tensor { name, layer, layout } = &slot.kind else {
            return Err(CactError::Entry { index: slot.entry, reason: "is not a canonical tensor".to_string() });
        };
        let stored = self.file.f32_tensor(slot.entry)?;
        let unit = self.unit_shape(name, *layer);
        Ok(match layout {
            Layout::Same => stored,
            Layout::Transpose => transpose(&stored, unit[0], unit[1]),
            Layout::MhcPhi => mhc_phi_to_canonical(&stored, unit[0], unit[1], unit[2]),
        })
    }

    /// Every deployment tensor in canonical shape and element order, stacked
    /// decoder tensors assembled from their per-layer entries.
    pub fn canonical_tensors(&self) -> Result<BTreeMap<String, Tensor>, CactError> {
        let mut out: BTreeMap<String, Tensor> = BTreeMap::new();
        for slot in &self.slots {
            let SlotKind::Tensor { name, layer, .. } = &slot.kind else { continue };
            let unit = self.slot_tensor(slot)?;
            let spec = self.manifest.get(name).expect("the layout only names manifest tensors");
            let tensor = out.entry(name.clone()).or_insert_with(|| Tensor { shape: spec.shape.clone(), data: vec![0.0; spec.numel()] });
            let start = layer.map_or(0, |l| l * unit.len());
            tensor.data[start..start + unit.len()].copy_from_slice(&unit);
        }
        Ok(out)
    }

    fn find(&self, wanted: impl Fn(&SlotKind) -> bool) -> usize {
        self.slots.iter().position(|s| wanted(&s.kind)).expect("the layout always contains the derived entries")
    }

    /// The two Hadamard-MLP permutations, each a permutation of `0..hadamard_width`.
    pub fn hadamard_permutations(&self) -> Result<(Vec<u32>, Vec<u32>), CactError> {
        let read = |k: u8| -> Result<Vec<u32>, CactError> {
            let index = self.find(|s| *s == SlotKind::HadamardPermutation(k));
            let n = self.config.hadamard_width();
            let mut seen = vec![false; n];
            let mut out = Vec::with_capacity(n);
            for v in self.file.f32_tensor(index)? {
                let valid = v.is_finite() && v >= 0.0 && v.fract() == 0.0 && (v as usize) < n;
                if !valid || std::mem::replace(&mut seen[v as usize], true) {
                    return Err(CactError::Entry { index, reason: format!("permutation {k} has an out-of-range or repeated index ({v})") });
                }
                out.push(v as u32);
            }
            Ok(out)
        };
        Ok((read(1)?, read(2)?))
    }

    /// The 36 vision constants (their meaning belongs to the encoder).
    pub fn vision_manifest(&self) -> Result<Vec<f32>, CactError> {
        let index = self.find(|s| *s == SlotKind::VisionManifest);
        let values = self.file.f32_tensor(index)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(CactError::Entry { index, reason: "the vision manifest holds a non-finite value".to_string() });
        }
        Ok(values)
    }

    /// The OCR charset; reader label `i` (non-zero) is `charset[i - 1]`.
    pub fn charset(&self) -> Result<Vec<char>, CactError> {
        let index = self.find(|s| *s == SlotKind::Charset);
        self.file
            .f32_tensor(index)?
            .into_iter()
            .map(|v| {
                let code = if v.is_finite() && v > 0.0 && v.fract() == 0.0 && v <= 0x10FFFF as f32 { char::from_u32(v as u32) } else { None };
                code.ok_or_else(|| CactError::Entry { index, reason: format!("charset value {v} is not a Unicode scalar value") })
            })
            .collect()
    }

    /// The tokenizer blob, for [`crate::tokenizer::Tokenizer::from_blob`].
    pub fn tokenizer_blob(&self) -> Result<&[u8], CactError> {
        self.file.raw(self.find(|s| *s == SlotKind::Tokenizer))
    }
}

/// `[rows, cols]` row-major stored `[out, in]` -> `[in, out]` row-major.
fn transpose(stored: &[f32], inn: usize, out: usize) -> Vec<f32> {
    let mut t = vec![0f32; stored.len()];
    for o in 0..out {
        for i in 0..inn {
            t[i * out + o] = stored[o * inn + i];
        }
    }
    t
}

/// `[layers * lanes, width]` -> `[layers, width, lanes]`.
fn mhc_phi_to_canonical(stored: &[f32], layers: usize, width: usize, lanes: usize) -> Vec<f32> {
    let mut t = vec![0f32; stored.len()];
    for l in 0..layers {
        for j in 0..lanes {
            for w in 0..width {
                t[(l * width + w) * lanes + j] = stored[(l * lanes + j) * width + w];
            }
        }
    }
    t
}

fn check_header(file: &CactFile, c: &MirageConfig) -> Result<(), CactError> {
    let h = file.header();
    let checks: [(&str, usize, usize); 14] = [
        ("vocab", h.vocab(), c.embedding_rows()),
        ("d_model", h.d_model(), c.dim),
        ("num_heads", h.num_heads(), c.heads()),
        ("num_kv_heads", h.num_kv_heads(), c.kv_heads),
        ("num_layers", h.num_layers(), c.dec_layers),
        ("qk_head_dim", h.qk_head_dim(), c.qk_dim),
        ("v_head_dim", h.v_head_dim(), c.v_dim),
        ("max_seq_len", h.max_seq_len(), c.max_tokens),
        ("hada_n", h.hada_n(), c.hadamard_width()),
        ("mhc_lanes", h.mhc_lanes(), c.lanes),
        ("qkv_conv_taps", h.qkv_conv_taps(), c.taps),
        ("engram_slots", h.engram_slots(), c.slots),
        ("engram_sub_dim", h.engram_sub_dim(), c.engram_sub_dim()),
        ("num_engram_tables", h.num_engram_tables(), c.engram_tables()),
    ];
    for (field, found, expected) in checks {
        if found != expected {
            return Err(header_mismatch(field, found, expected));
        }
    }
    if h.engram_orders() != c.orders {
        return Err(header_mismatch("engram_orders", format!("{:?}", h.engram_orders()), format!("{:?}", c.orders)));
    }
    if h.engram_sites() != c.sites {
        return Err(header_mismatch("engram_sites", format!("{:?}", h.engram_sites()), format!("{:?}", c.sites)));
    }
    if h.rope_theta != c.rope {
        return Err(header_mismatch("rope_theta", h.rope_theta, c.rope));
    }
    Ok(())
}
