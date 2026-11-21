// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements checkpoint import boundaries that refuse a wrong file by naming the tensor, for its
// clients. If your team needs expertise in mapping a published release onto an engine's parameters exactly then you
// can procure our services by sending an email to info@swedishembedded.com.

//! The published tensor names of an Inkling 2 checkpoint, derived from the config, with two-way coverage.
//!
//! [`manifest`] lists every parameter the published file holds for a config (the decoder and, when configured, the
//! vision tower), with its logical shape and the encodings its role admits. [`check`] holds a reader to it in both
//! directions: every listed tensor is present with exactly that shape and an admitted encoding, and nothing else is
//! present. The quantization sidecars (`_scale`, `_amax`, `_outer_scale_denominator`) are not parameters:
//! `checkpoint::packed::PackedReader` folds them into the expert banks' encodings and hides them.
//!
//! Three facts of the published layout live here and nowhere else:
//! - `attn_res.attn_queries.{i}` belongs to layer `i + 1` (layer 0 reads the embedding and has no attention input);
//!   a full layer's is `[3, hidden]` (q, k, v inputs);
//! - the message queries are two `[2 * core_layers, hidden]` tensors: `middle_cycle_cross_pass_queries` holds each core
//!   layer's (q, mlp) pair and `middle_cycle_cross_pass_kv_queries` its (k, v) pair ([`message_query_rows`]);
//! - the message gates of a core layer are `controllers.{attn,k,v,mlp}`, where `attn` gates the query input.

use checkpoint::packed::{PackedTensorReader, QuantEncoding};

use crate::config::{Inkling2Config, LayerKind};

/// The decoder's prefix in the published names.
pub const TEXT_PREFIX: &str = "model.llm.";
/// The vision tower's prefix in the published names.
pub const VISION_PREFIX: &str = "model.visual.";

/// The encodings a parameter may be stored in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admits {
    /// A float tensor: f32 or bf16.
    Float,
    /// A stacked expert bank: f32, bf16, NVFP4 with a per-expert second level, or MXFP8.
    ExpertBank,
}

impl Admits {
    pub fn allows(self, e: &QuantEncoding) -> bool {
        let float = *e == QuantEncoding::f32() || *e == QuantEncoding::bf16();
        match self {
            Admits::Float => float,
            Admits::ExpertBank => float || *e == QuantEncoding::nvfp4_per_slab() || *e == QuantEncoding::mxfp8(),
        }
    }
}

/// One published parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub name: String,
    pub shape: Vec<u64>,
    pub admits: Admits,
}

/// The message gate of a core layer: `controllers.<site>`. `Attn` gates the query input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateSite {
    Attn,
    K,
    V,
    Mlp,
}

impl GateSite {
    /// In the order the forward applies them (q, k, v inputs, then the MoE input).
    pub const ALL: [GateSite; 4] = [GateSite::Attn, GateSite::K, GateSite::V, GateSite::Mlp];

    fn word(self) -> &'static str {
        match self {
            GateSite::Attn => "attn",
            GateSite::K => "k",
            GateSite::V => "v",
            GateSite::Mlp => "mlp",
        }
    }
}

/// The published names of layer `i`'s tensors.
#[derive(Clone, Debug)]
pub struct LayerNames {
    p: String,
}

impl LayerNames {
    pub fn new(layer: u32) -> LayerNames {
        LayerNames { p: format!("{TEXT_PREFIX}layers.{layer}.") }
    }
    fn n(&self, s: &str) -> String {
        format!("{}{s}", self.p)
    }
    pub fn wq(&self) -> String {
        self.n("attn.wq_du.weight")
    }
    pub fn wk(&self) -> String {
        self.n("attn.wk_dv.weight")
    }
    pub fn wv(&self) -> String {
        self.n("attn.wv_dv.weight")
    }
    pub fn wo(&self) -> String {
        self.n("attn.wo_ud.weight")
    }
    pub fn gate_attn(&self) -> String {
        self.n("attn.gate_attn.weight")
    }
    pub fn sink(&self) -> String {
        self.n("attn.attn_sink_proj.weight")
    }
    pub fn q_scale(&self) -> String {
        self.n("attn.q_l2_clip.scale")
    }
    pub fn wr(&self) -> String {
        self.n("attn.wr_du.weight")
    }
    pub fn rel_scale(&self) -> String {
        self.n("attn.rel_l2_clip.scale")
    }
    pub fn rel_bank(&self) -> String {
        self.n("attn.rel_logits_proj.proj")
    }
    /// The static K or V conv of a sliding layer (`[channels, 1, taps]`, last tap the current token).
    pub fn kv_conv(&self, v: bool) -> String {
        self.n(if v { "attn.v_sconv.weight" } else { "attn.k_sconv.weight" })
    }
    /// The static taps of a full layer's dynamic K or V conv (`[taps, channels]`, tap 0 the current token).
    pub fn kv_conv_static(&self, v: bool) -> String {
        self.n(if v { "attn.v_sconv.static_w" } else { "attn.k_sconv.static_w" })
    }
    /// The per-token tap projection of a full layer's dynamic K or V conv (`[kv_heads * taps, hidden]`).
    pub fn kv_conv_proj(&self, v: bool) -> String {
        self.n(if v { "attn.v_sconv.weight_proj" } else { "attn.k_sconv.weight_proj" })
    }
    pub fn attn_norm(&self) -> String {
        self.n("attn_norm.weight")
    }
    /// A full layer's input norm of the q (0), k (1) or v (2) path.
    pub fn qkv_norm(&self, which: usize) -> String {
        self.n(["q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"][which])
    }
    pub fn attn_conv(&self) -> String {
        self.n("attn_sconv.weight")
    }
    pub fn mlp_norm(&self) -> String {
        self.n("mlp_norm.weight")
    }
    pub fn mlp_conv(&self) -> String {
        self.n("mlp_sconv.weight")
    }
    pub fn router(&self) -> String {
        self.n("mlp.gate.weight")
    }
    pub fn router_bias(&self) -> String {
        self.n("mlp.gate.bias")
    }
    /// The fused gate/up expert bank `[experts, 2 * ff, hidden]`, rows interleaved `g0, u0, g1, u1, ...`.
    pub fn w13(&self) -> String {
        self.n("mlp.experts.w13_weight")
    }
    /// The down expert bank `[experts, hidden, ff]`.
    pub fn w2(&self) -> String {
        self.n("mlp.experts.w2_weight")
    }
    /// A core layer's message gate: `down` `[rank, 2 hidden]`, `forget` and `gain` `[hidden, rank]`.
    pub fn gate(&self, site: GateSite) -> [String; 3] {
        let s = site.word();
        [self.n(&format!("controllers.{s}.down_weight")), self.n(&format!("controllers.{s}.forget_weight")), self.n(&format!("controllers.{s}.gain_weight"))]
    }
}

pub fn embed() -> String {
    format!("{TEXT_PREFIX}embed.weight")
}
pub fn embed_norm() -> String {
    format!("{TEXT_PREFIX}embed_norm.weight")
}
pub fn final_norm() -> String {
    format!("{TEXT_PREFIX}norm.weight")
}
pub fn unembed() -> String {
    format!("{TEXT_PREFIX}unembed.weight")
}
/// Layer `layer`'s attention-input depth queries (`layer >= 1`): published as `attn_queries.{layer - 1}`.
pub fn attn_queries(layer: u32) -> String {
    assert!(layer >= 1, "layer 0 has no attention-input queries");
    format!("{TEXT_PREFIX}attn_res.attn_queries.{}", layer - 1)
}
pub fn mlp_queries(layer: u32) -> String {
    format!("{TEXT_PREFIX}attn_res.mlp_queries.{layer}")
}
pub fn output_query() -> String {
    format!("{TEXT_PREFIX}attn_res.output_query")
}
/// The (q, mlp) message queries of every core layer.
pub fn message_queries() -> String {
    format!("{TEXT_PREFIX}middle_cycle_cross_pass_queries")
}
/// The (k, v) message queries of every core layer.
pub fn message_kv_queries() -> String {
    format!("{TEXT_PREFIX}middle_cycle_cross_pass_kv_queries")
}

/// Where core layer `c`'s message queries for the q, k, v and MLP inputs sit: `(tensor, row)` with `false` for
/// [`message_queries`] and `true` for [`message_kv_queries`].
pub fn message_query_rows(c: u32) -> [(bool, u32); 4] {
    [(false, 2 * c), (true, 2 * c), (true, 2 * c + 1), (false, 2 * c + 1)]
}

fn entry(name: String, shape: &[u32], admits: Admits) -> Entry {
    Entry { name, shape: shape.iter().map(|&d| d as u64).collect(), admits }
}

/// Every decoder parameter of `cfg` in the published names.
pub fn text_manifest(cfg: &Inkling2Config) -> Vec<Entry> {
    use Admits::{ExpertBank, Float};
    let d = cfg.hidden;
    let mut out = vec![
        entry(embed(), &[cfg.vocab, d], Float),
        entry(embed_norm(), &[d], Float),
        entry(final_norm(), &[d], Float),
        entry(unembed(), &[cfg.vocab, d], Float),
        entry(output_query(), &[d], Float),
    ];
    if let Some(l) = cfg.looped {
        out.push(entry(message_queries(), &[2 * l.layers(), d], Float));
        out.push(entry(message_kv_queries(), &[2 * l.layers(), d], Float));
    }
    for i in 0..cfg.n_layers {
        let n = LayerNames::new(i);
        let a = cfg.attn(i);
        let (q, kv, taps) = (a.q_dim(), a.kv_dim(), cfg.conv_taps);
        match cfg.attn_queries(i) {
            0 => {}
            1 => out.push(entry(attn_queries(i), &[d], Float)),
            nq => out.push(entry(attn_queries(i), &[nq, d], Float)),
        }
        out.push(entry(mlp_queries(i), &[d], Float));
        out.extend([
            entry(n.wq(), &[q, d], Float),
            entry(n.wk(), &[kv, d], Float),
            entry(n.wv(), &[kv, d], Float),
            entry(n.wo(), &[d, q], Float),
            entry(n.gate_attn(), &[q, d], Float),
            entry(n.sink(), &[a.heads, d], Float),
            entry(n.q_scale(), &[a.heads, a.head_dim], Float),
            entry(n.wr(), &[a.heads * cfg.d_rel, d], Float),
            entry(n.rel_scale(), &[a.heads, cfg.d_rel], Float),
            entry(n.rel_bank(), &[cfg.d_rel, cfg.bias_extent(i)], Float),
            entry(n.attn_conv(), &[d, 1, taps], Float),
            entry(n.mlp_norm(), &[d], Float),
            entry(n.mlp_conv(), &[d, 1, taps], Float),
            entry(n.router(), &[cfg.experts, d], Float),
            entry(n.router_bias(), &[cfg.experts], Float),
            entry(n.w13(), &[cfg.experts, 2 * cfg.moe_ff, d], ExpertBank),
            entry(n.w2(), &[cfg.experts, d, cfg.moe_ff], ExpertBank),
        ]);
        match cfg.kind(i) {
            LayerKind::Sliding => {
                out.push(entry(n.attn_norm(), &[d], Float));
                for v in [false, true] {
                    out.push(entry(n.kv_conv(v), &[kv, 1, taps], Float));
                }
            }
            LayerKind::Full => {
                for w in 0..3 {
                    out.push(entry(n.qkv_norm(w), &[d], Float));
                }
                for v in [false, true] {
                    out.push(entry(n.kv_conv_static(v), &[taps, kv], Float));
                    out.push(entry(n.kv_conv_proj(v), &[a.kv_heads * taps, d], Float));
                }
            }
        }
        if cfg.is_core(i) {
            let rank = cfg.looped.expect("a core layer belongs to a loop").gate_rank;
            for site in GateSite::ALL {
                let [down, forget, gain] = n.gate(site);
                out.extend([entry(down, &[rank, 2 * d], Float), entry(forget, &[d, rank], Float), entry(gain, &[d, rank], Float)]);
            }
        }
    }
    out
}

/// Every vision-tower parameter of `cfg` in the published names (empty without a `vision_config`).
pub fn vision_manifest(cfg: &Inkling2Config) -> Vec<Entry> {
    use Admits::Float;
    let Some(v) = &cfg.vision else { return Vec::new() };
    let (w, ff, p, m) = (v.hidden_size, v.mlp_width(), v.native_patch_size, v.adapter_merge_size);
    let b = format!("{VISION_PREFIX}backbone.");
    let mut out = vec![
        entry(format!("{b}embeddings.class_embedding"), &[w], Float),
        entry(format!("{b}embeddings.cls_position_embedding"), &[w], Float),
        entry(format!("{b}embeddings.patch_embedding.weight"), &[w, 3 * p * p], Float),
        entry(format!("{b}embeddings.position_embedding"), &[v.posemb_grid_size * v.posemb_grid_size, w], Float),
        entry(format!("{b}ln_pre.weight"), &[w], Float),
        entry(format!("{b}ln_pre.bias"), &[w], Float),
        entry(format!("{VISION_PREFIX}proj_in.weight"), &[v.projection_hidden_dim, w * m * m], Float),
        entry(format!("{VISION_PREFIX}proj_out.weight"), &[v.text_hidden_size, v.projection_hidden_dim], Float),
    ];
    for l in 0..v.num_hidden_layers {
        let n = |s: &str| format!("{b}layers.{l}.{s}");
        for ln in ["ln_1", "ln_2"] {
            out.push(entry(n(&format!("{ln}.weight")), &[w], Float));
            out.push(entry(n(&format!("{ln}.bias")), &[w], Float));
        }
        out.push(entry(n("ls_1.gamma"), &[w], Float));
        out.push(entry(n("ls_2.gamma"), &[w], Float));
        out.extend([
            entry(n("mlp.c_fc.weight"), &[ff, w], Float),
            entry(n("mlp.c_fc.bias"), &[ff], Float),
            entry(n("mlp.c_proj.weight"), &[w, ff], Float),
            entry(n("mlp.c_proj.bias"), &[w], Float),
        ]);
        for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            out.push(entry(n(&format!("self_attn.{proj}.weight")), &[w, w], Float));
            out.push(entry(n(&format!("self_attn.{proj}.bias")), &[w], Float));
        }
    }
    out
}

/// Every parameter of the published file for `cfg`: the decoder, then the vision tower.
pub fn manifest(cfg: &Inkling2Config) -> Vec<Entry> {
    let mut m = text_manifest(cfg);
    m.extend(vision_manifest(cfg));
    m
}

/// Hold `reader` to [`manifest`] in both directions: every listed parameter present once with its shape and an
/// admitted encoding, nothing unlisted present. Errors name the tensor.
pub fn check(cfg: &Inkling2Config, reader: &dyn PackedTensorReader) -> Result<(), String> {
    let m = manifest(cfg);
    let present = reader.names();
    let mut seen = std::collections::HashSet::with_capacity(m.len());
    for e in &m {
        if !seen.insert(e.name.as_str()) {
            return Err(format!("inkling2 name map: {} is listed twice", e.name));
        }
        let desc = reader.descriptor(&e.name).map_err(|err| format!("inkling2 checkpoint: missing or unreadable tensor {}: {err}", e.name))?;
        if desc.shape != e.shape {
            return Err(format!("inkling2 checkpoint: {} has shape {:?}, the config makes it {:?}", e.name, desc.shape, e.shape));
        }
        if !e.admits.allows(&desc.encoding) {
            return Err(format!("inkling2 checkpoint: {} is stored as {}, which its role ({:?}) does not admit", e.name, desc.encoding, e.admits));
        }
    }
    let mut extra: Vec<&String> = present.iter().filter(|n| !seen.contains(n.as_str())).collect();
    if !extra.is_empty() {
        extra.sort();
        return Err(format!("inkling2 checkpoint: {} tensors the config does not account for: {extra:?}", extra.len()));
    }
    Ok(())
}
