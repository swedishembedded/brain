// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GGUF import for the dense Qwen3 decoder - the second source format for the
//! same model [`crate::import`] reads from HuggingFace safetensors.
//!
//! Two things make this worth having rather than telling users to fetch the
//! bf16 checkpoint. A quantized GGUF is a substantially smaller download for
//! the same model, so it is a bytes win as well as a host-memory one; and
//! FLUX.2's text encoder **is** a Qwen3, so `BRAIN_FLUX2_TE` gets to point at
//! a GGUF for free (see [`crate::import::shard_source`], which sniffs the
//! naming convention rather than being told which one it is looking at).
//!
//! ## Where the name map comes from
//!
//! Transcribed from **llama.cpp** at revision
//! `d7a2074112d27649303fa107eb8c94db1ee435f3`, from the two files that are
//! authoritative for it:
//!
//! - `gguf-py/gguf/constants.py` - `MODEL_ARCH_NAMES[MODEL_ARCH.QWEN3] =
//!   "qwen3"`, the `MODEL_TENSORS[MODEL_ARCH.QWEN3]` list (exactly the 15
//!   entries below), and `TENSOR_NAMES`, which spells each entry's GGUF name.
//! - `gguf-py/gguf/tensor_mapping.py` - the HF-name → `MODEL_TENSOR` table.
//!
//! plus `conversion/qwen.py`, where `Qwen3Model(Qwen2Model)` is registered for
//! `Qwen3ForCausalLM` with `model_arch = MODEL_ARCH.QWEN3` and inherits the
//! base `modify_tensors` - i.e. for a plain (non-rerank, non-MoE) Qwen3 the
//! conversion is a pure 1:1 rename with no reshaping, splitting or permuting
//! anywhere in it.
//!
//! It is transcribed rather than inferred from tensor shapes on purpose. At
//! Qwen3-8B's shape `q_proj` is `[4096, 4096]` and, on any GQA layer, `k_proj`
//! and `v_proj` are the same `[1024, 4096]` as each other - so a map guessed
//! from shapes would swap `k` and `v` silently, and on a square-`q` model
//! could swap `q` with `o`. The `qk_swap_is_caught_by_the_parity_gate` test
//! below is that failure mode, made to fail on purpose.
//!
//! | GGUF (llama.cpp) | HF (`Qwen3ForCausalLM`) | brain |
//! |---|---|---|
//! | `token_embd.weight` | `model.embed_tokens.weight` | `tok.weight` |
//! | `output_norm.weight` | `model.norm.weight` | `norm.weight` |
//! | `output.weight` | `lm_head.weight` | `lm_head.weight` (absent when tied) |
//! | `blk.N.attn_norm.weight` | `model.layers.N.input_layernorm.weight` | `blocks.N.ln1.weight` |
//! | `blk.N.attn_q.weight` | `…self_attn.q_proj.weight` | `blocks.N.attn.wq.weight` |
//! | `blk.N.attn_k.weight` | `…self_attn.k_proj.weight` | `blocks.N.attn.wk.weight` |
//! | `blk.N.attn_v.weight` | `…self_attn.v_proj.weight` | `blocks.N.attn.wv.weight` |
//! | `blk.N.attn_output.weight` | `…self_attn.o_proj.weight` | `blocks.N.attn.wo.weight` |
//! | `blk.N.attn_q_norm.weight` | `…self_attn.q_norm.weight` | `blocks.N.attn.q_norm.weight` |
//! | `blk.N.attn_k_norm.weight` | `…self_attn.k_norm.weight` | `blocks.N.attn.k_norm.weight` |
//! | `blk.N.ffn_norm.weight` | `…post_attention_layernorm.weight` | `blocks.N.ln2.weight` |
//! | `blk.N.ffn_gate.weight` | `…mlp.gate_proj.weight` | `blocks.N.mlp.gate.weight` |
//! | `blk.N.ffn_up.weight` | `…mlp.up_proj.weight` | `blocks.N.mlp.up.weight` |
//! | `blk.N.ffn_down.weight` | `…mlp.down_proj.weight` | `blocks.N.mlp.down.weight` |
//! | `rope_freqs.weight` | (rope-scaling factors) | the config's `RopeScaling::Factors` |
//!
//! `blk.N.attn_{q,k,v}.bias` are accepted too. Qwen3 has no attention bias, so
//! a Qwen3 GGUF never carries them - but [`crate::QwenConfig`] describes Qwen2
//! as well (`attn_bias: true`), the GGUF spelling is the same, and a `qwen2`
//! file reaching this map should map or be refused, never be dropped quietly.
//!
//! No tensor is transposed and none is split: brain's `matmul` is
//! `out = x @ Wᵀ` with `W:[out,in]` row-major, which is what both HF
//! `nn.Linear.weight` and llama.cpp's dequantized row-major output already
//! are. The conversion is a rename plus a dequant, exactly as the HF route is
//! a rename plus a bf16→f32 widen.

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;
use checkpoint::gguf_src::GgufSource;
use checkpoint::remap::{Fetch, RemapSource};
use model::rope_scaling::RopeScaling;
use checkpoint::st::ModelCard;
use gguf::import::{self, ImportStats, Leaf, Mapped};
use gguf::leaf::Role;
use gguf::ArchKv;

use crate::config::QwenConfig;

/// llama.cpp's `general.architecture` value for the dense Qwen3 family
/// (`MODEL_ARCH_NAMES[MODEL_ARCH.QWEN3]`).
pub const GGUF_ARCHITECTURE: &str = "qwen3";

/// Every `general.architecture` this decoder reads: Qwen3 and its two
/// config variants. `qwen2` differs only in its q/k/v bias and missing
/// QK-norm, both declared by which tensors the file carries. `llama` also
/// stores q/k with llama.cpp's per-head row interleave
/// (`conversion/llama.py`'s `LlamaModel.permute`, `undo_permute = True`),
/// which [`llama_unpermute_order`] undoes; a Qwen2 conversion never permutes.
pub const GGUF_ARCHITECTURES: [&str; 3] = [GGUF_ARCHITECTURE, "qwen2", "llama"];

/// Not imported, by reason - [`Mapped::Dropped`]'s payload, counted and
/// printed by the shared driver so a drop is always on the record.
const DROP_TIED_HEAD: &str = "output.weight on a tied-embedding checkpoint (the head reuses tok.weight)";
const DROP_ROPE_FREQS: &str = "rope_freqs.weight (read into the config as its RoPE scaling)";

/// Map one GGUF tensor name to its brain parameter name.
///
/// `None` means "deliberately not a brain parameter": the two named drops
/// above, and anything this map does not recognize. Callers that need the
/// distinction - the importer, which must refuse an unrecognized leaf rather
/// than lose a projection - use [`classify`] instead.
///
/// Shared with [`crate::import::hf_shard_source`]'s GGUF arm, so the streaming
/// text-encoder path and the whole-checkpoint importer cannot disagree about
/// which tensor is which.
pub fn gguf_to_brain(name: &str, tie: bool) -> Option<String> {
    // n_layers is deliberately u32::MAX here: this is the *name* map, and it
    // has no opinion about depth. A block index beyond the config's depth
    // still maps, and is then rejected by the caller's coverage check as a
    // brain parameter outside `param_list()` - which is how a 36-layer
    // checkpoint against a 28-layer config is caught.
    match import::split_name(name, u32::MAX) {
        Leaf::TokenEmbd => Some("tok.weight".to_string()),
        Leaf::OutputNorm => Some("norm.weight".to_string()),
        Leaf::Output => (!tie).then(|| "lm_head.weight".to_string()),
        Leaf::Block { layer, leaf } => {
            // The leaf VOCABULARY (which spellings exist and what they mean)
            // is `gguf::leaf`'s, shared with `qwen35moe`/`qwen35`; only this
            // model's own brain-parameter suffix is qwen3-specific.
            let leaf = match gguf::leaf::role(leaf)? {
                Role::AttnNorm => "ln1.weight",
                Role::FfnNorm => "ln2.weight",
                Role::AttnQ => "attn.wq.weight",
                Role::AttnK => "attn.wk.weight",
                Role::AttnV => "attn.wv.weight",
                Role::AttnOutput => "attn.wo.weight",
                Role::AttnQNorm => "attn.q_norm.weight",
                Role::AttnKNorm => "attn.k_norm.weight",
                Role::AttnQBias => "attn.wq.bias",
                Role::AttnKBias => "attn.wk.bias",
                Role::AttnVBias => "attn.wv.bias",
                Role::FfnGate => "mlp.gate.weight",
                Role::FfnUp => "mlp.up.weight",
                Role::FfnDown => "mlp.down.weight",
                _ => return None,
            };
            Some(format!("blocks.{layer}.{leaf}"))
        }
        Leaf::PastDepth { .. } | Leaf::Other => None,
    }
}

/// [`gguf_to_brain`] with the importer's strictness: an unrecognized tensor is
/// an **error**, not a silent skip, and each drop states its reason.
///
/// That asymmetry is the point. A converter that renames a leaf must break the
/// import loudly rather than quietly produce a checkpoint missing a
/// projection, and "we didn't recognize it" is exactly the case where a
/// missing projection would otherwise look like a clean run.
fn classify(name: &str, cfg: &QwenConfig, llama: bool) -> Result<Mapped, String> {
    let tie = cfg.tie_embeddings;
    if let Some(brain) = gguf_to_brain(name, tie) {
        let heads = if brain.ends_with("attn.wq.weight") {
            Some(cfg.n_heads)
        } else if brain.ends_with("attn.wk.weight") {
            Some(cfg.n_kv_heads)
        } else {
            None
        };
        return Ok(match heads {
            Some(n) if llama => Mapped::Permuted { into: brain, order: llama_unpermute_order(n as usize, cfg.head_dim as usize) },
            _ => Mapped::Simple(brain),
        });
    }
    match import::split_name(name, u32::MAX) {
        Leaf::Output => Ok(Mapped::Dropped(DROP_TIED_HEAD)),
        _ if name == "rope_freqs.weight" => Ok(Mapped::Dropped(DROP_ROPE_FREQS)),
        _ => Err(format!("unrecognized tensor {name:?} - the qwen3 name map has no entry for it")),
    }
}

/// The source row order that undoes llama.cpp's q/k permute for `n_head`
/// heads of `head_dim` rows: llama.cpp stores head `h`'s row `half * hd/2 + i`
/// (HF order) at `h * hd + 2i + half`, so brain's row `h*hd + half*hd/2 + i`
/// reads source row `h*hd + 2i + half`.
pub fn llama_unpermute_order(n_head: usize, head_dim: usize) -> Vec<u32> {
    let half = head_dim / 2;
    (0..n_head * head_dim)
        .map(|r| {
            let (h, within) = (r / head_dim, r % head_dim);
            let (two, i) = (within / half, within % half);
            (h * head_dim + 2 * i + two) as u32
        })
        .collect()
}

/// Every brain parameter of `cfg` as a fetch over the GGUF's own tensors:
/// a rename, or - llama's q/k - a row permutation, which moves quantized
/// rows whole. The one name map both [`open_source`] and
/// [`crate::import::source`] read through, as strict as the importer's: an
/// unrecognized tensor is an error.
pub fn gguf_plan(mg: &MmapGguf, cfg: &QwenConfig) -> Result<HashMap<String, Fetch>, String> {
    let llama = gguf::kv::architecture(mg) == Some("llama");
    let mut plan: HashMap<String, Fetch> = HashMap::new();
    let mut embed_source: Option<String> = None;
    for g in mg.names() {
        if matches!(import::split_name(g, u32::MAX), Leaf::TokenEmbd) {
            embed_source = Some(g.clone());
        }
        let (brain, fetch) = match classify(g, cfg, llama)? {
            Mapped::Simple(brain) => (brain, Fetch::Whole(g.clone())),
            Mapped::Permuted { into, order } => (into, Fetch::RowPermute { name: g.clone(), order }),
            Mapped::Dropped(_) => continue,
            other => return Err(format!("qwen3: {g} maps to {other:?}, which a decoder source cannot read")),
        };
        if plan.insert(brain.clone(), fetch).is_some() {
            return Err(format!("qwen3: two GGUF tensors map to {brain}"));
        }
    }
    // A tied checkpoint carries no `output.weight`: the head IS the embedding
    // table. The model still asks for `lm_head.weight`, and it must resolve.
    if cfg.tie_embeddings {
        if let Some(embed) = embed_source {
            plan.insert("lm_head.weight".to_string(), Fetch::Whole(embed));
        }
    }
    Ok(plan)
}

/// Open a GGUF and present it under **brain's own** qwen3 parameter names,
/// ready to build from directly - no ahead-of-time conversion, no fp32
/// intermediate on disk, no whole-model host copy. The quantized bytes are
/// already what a reduced-precision build wants, so the builder streams
/// straight off the mapping, one leaf at a time, through [`gguf_plan`].
///
/// # Errors
///
/// A human-readable message if the file cannot be mapped, its KV metadata
/// does not describe one of [`GGUF_ARCHITECTURES`], or a tensor is not one
/// the decoder has.
pub fn open_source(path: &str) -> Result<(QwenConfig, RemapSource<'static>), String> {
    let mg = MmapGguf::open(path).map_err(|e| format!("cannot open gguf {path:?}: {e}"))?;
    let cfg = config_from_gguf(&mg)?;
    let plan = gguf_plan(&mg, &cfg)?;
    Ok((cfg, RemapSource::owning(Box::new(GgufSource::identity(mg)), plan)))
}

/// Derive a [`QwenConfig`] from a GGUF's KV metadata.
///
/// Every field comes from llama.cpp's standardized `{arch}.…` keys except
/// `vocab`, which is read from `token_embd.weight`'s own shape: the tensor is
/// ground truth, and a `vocab_size` KV that disagrees with the embedding table
/// would produce a config that cannot load its own checkpoint.
///
/// `block_size` is 2048, matching [`crate::hf::decoder_config`] - it sizes
/// buffers, and the trained RoPE extent (`context_length`, carried through as
/// `max_position_embeddings`) would size them absurdly.
pub fn config_from_gguf(mg: &MmapGguf) -> Result<QwenConfig, String> {
    let got = gguf::kv::architecture(mg).unwrap_or("");
    let arch = GGUF_ARCHITECTURES
        .iter()
        .find(|a| **a == got)
        .ok_or_else(|| format!("gguf: general.architecture {got:?} is not one of {GGUF_ARCHITECTURES:?}"))?;
    config_from_kv(&ArchKv::new(mg, arch), mg)
}

/// [`config_from_gguf`]'s core, against an already-scoped KV view.
///
/// Split out because a Qwen3 decoder is not always the whole checkpoint: a
/// Qwen3-VL GGUF carries the same dense decoder under its OWN architecture
/// prefix (`qwen3vl.*`), and reading it there must not mean a second
/// transcription of llama.cpp's key names that can drift from this one.
pub fn config_from_kv(kv: &ArchKv, mg: &MmapGguf) -> Result<QwenConfig, String> {
    let block_size = 2048;
    let vocab = mg
        .shape("token_embd.weight")
        .and_then(|s| s.first().copied())
        .ok_or("qwen3: missing token_embd.weight (cannot determine vocab)")? as u32;
    let d_model = kv.req_u32("embedding_length")?;
    let n_heads = kv.req_u32("attention.head_count")?;
    // A llama conversion writes no key_length; its rope.dimension_count is the
    // HF head_dim (`conversion/llama.py`), and d_model / n_heads failing both.
    let head_dim = kv.u32("attention.key_length").or_else(|| kv.u32("rope.dimension_count")).unwrap_or(d_model / n_heads.max(1));
    let value_len = kv.u32_or("attention.value_length", head_dim);
    if value_len != head_dim {
        return Err(format!("qwen3: attention.key_length {head_dim} != value_length {value_len} (asymmetric head_dim is unsupported)"));
    }
    let rope_dim = kv.u32_or("rope.dimension_count", head_dim);
    if rope_dim != head_dim {
        return Err(format!("qwen3: rope.dimension_count {rope_dim} != head_dim {head_dim} (partial rotary embedding is unsupported)"));
    }
    let has = |suffix: &str| mg.names().iter().any(|n| n.ends_with(suffix));
    // Qwen3's own default; a llama or qwen2 conversion always writes one.
    let theta_default = if matches!(kv.prefix(), "qwen3" | "qwen3vl") { 1.0e6 } else { 1.0e4 };
    Ok(QwenConfig {
        vocab,
        block_size,
        n_layers: kv.req_u32("block_count")?,
        d_model,
        n_heads,
        n_kv_heads: kv.req_u32("attention.head_count_kv")?,
        head_dim,
        d_ff: kv.req_u32("feed_forward_length")?,
        rope_theta: kv.f32_or("rope.freq_base", theta_default),
        rms_eps: kv.f32_or("attention.layer_norm_rms_epsilon", 1e-6),
        max_position_embeddings: kv.u32_or("context_length", block_size),
        // A GGUF states tying by OMITTING `output.weight` - there is no
        // `tie_word_embeddings` key. That is also what makes the tied drop in
        // `classify` unreachable for a real file: a tied checkpoint has no
        // head tensor to drop. It stays because a converter is free to ship
        // one anyway (HF Qwen3 checkpoints sometimes do), and dropping it with
        // a stated reason beats failing on it.
        tie_embeddings: !mg.names().iter().any(|n| n == "output.weight"),
        // Declared by the tensors the file carries: Qwen3 has QK-norm, Qwen2
        // has q/k/v biases, llama neither.
        qk_norm: has("attn_q_norm.weight"),
        attn_bias: has("attn_q.bias"),
        lora: None,
        rope_scaling: rope_scaling(kv, mg)?,
    }
    .with_defaults())
}

/// The RoPE scaling a GGUF declares: `rope_freqs.weight`, the per-frequency
/// divisors llama.cpp writes for a llama3 scaling, or the
/// `{arch}.rope.scaling.*` keys it writes for linear and YaRN. Any other
/// declared type is refused rather than run unscaled.
fn rope_scaling(kv: &ArchKv, mg: &MmapGguf) -> Result<Option<RopeScaling>, String> {
    use checkpoint::TensorSource;
    let declared = kv.str("rope.scaling.type").filter(|t| *t != "none");
    if mg.names().iter().any(|n| n == "rope_freqs.weight") {
        if let Some(t) = declared {
            return Err(format!("qwen3: both rope_freqs.weight and rope.scaling.type {t:?} - which scaling applies is ambiguous"));
        }
        let mut factors = Vec::new();
        mg.with_tensor("rope_freqs.weight", &mut |d| factors = d.to_vec());
        return Ok(Some(RopeScaling::Factors(factors)));
    }
    Ok(match declared {
        None => None,
        Some("linear") => Some(RopeScaling::Linear { factor: kv.req_f32("rope.scaling.factor")? }),
        Some("yarn") => Some(RopeScaling::Yarn(model::yarn::YarnConfig {
            factor: kv.req_f32("rope.scaling.factor")?,
            original_max_position_embeddings: kv.req_u32("rope.scaling.original_context_length")?,
            beta_fast: kv.f32_or("rope.scaling.yarn_beta_fast", 32.0),
            beta_slow: kv.f32_or("rope.scaling.yarn_beta_slow", 1.0),
            attention_factor: kv.f32("rope.scaling.yarn_attn_factor"),
        })),
        Some(other) => return Err(format!("qwen3: rope.scaling.type {other:?} is not implemented")),
    })
}

/// Import a Qwen3 GGUF into a brain-native safetensors checkpoint.
///
/// The streaming loop, the one-tensor-at-a-time dequant and the two-way
/// coverage check are `gguf::import`'s, shared with every other GGUF-sourced
/// model in the tree. What is Qwen3-specific, and stays here, is
/// [`config_from_gguf`]'s manifest and [`classify`]'s name map.
pub fn import_gguf(gguf_path: &str, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let mg = MmapGguf::open(gguf_path)?;
    import_mmap(&mg, out_path, id_override)
}

/// [`import_gguf`] over an ALREADY-OPEN checkpoint - the shape the generic
/// architecture-dispatch registry needs, since it must read
/// `general.architecture` before it can know which importer to call.
pub fn import_mmap(mg: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let cfg = config_from_gguf(mg)?;
    let params = cfg.param_list();

    let mut card = ModelCard::new(id_override.unwrap_or("qwen3"), "qwen");
    card.context_length = Some(cfg.block_size as u64);
    card.param_count = Some(params.iter().map(|(_, n)| *n as u64).sum());

    let llama = gguf::kv::architecture(mg) == Some("llama");
    import::to_st(mg, &params, &|n| classify(n, &cfg, llama), out_path, &cfg.to_json(), Some(&card), "qwen3")
}

/// Test fixtures for this importer, shared across crates.
///
/// `pub` (not `#[cfg(test)]`) so `brain-cli`'s GGUF-import-registry tests can
/// drive a REAL conversion through the generic architecture dispatch without a
/// second, drifting copy of this checkpoint builder. Not part of the model's
/// runtime surface.
#[doc(hidden)]
pub mod testing {
    use super::*;
    use checkpoint::gguf::GgufValue;
    use checkpoint::gguf_write::{write, TensorOut};

    /// The tiny shape both synthetic checkpoints below are built at. GQA (2 q
    /// heads over 1 kv head) and a decoupled `head_dim` are deliberate: with
    /// `n_heads == n_kv_heads` a q/k swap is a shape-compatible no-op and no
    /// coverage check could see it.
    pub const N_LAYERS: usize = 2;
    pub const VOCAB: usize = 5;
    pub const D_MODEL: usize = 6;
    pub const N_HEADS: usize = 2;
    pub const N_KV_HEADS: usize = 1;
    pub const HEAD_DIM: usize = 4;
    pub const D_FF: usize = 8;

    /// A distinct, exactly-representable value per (tensor, element), so any
    /// two tensors that got swapped show up as a value mismatch rather than
    /// passing a shape check.
    fn seq(base: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| base + i as f32).collect()
    }

    /// The one description of the fixture checkpoint's contents, in HF names.
    /// Both writers below render THIS list, so the safetensors and the GGUF
    /// are the same logical checkpoint by construction rather than by two
    /// hand-kept copies that could drift.
    fn contents(tied: bool) -> Vec<(&'static str, String, usize, f32)> {
        let (hq, hkv) = (N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM);
        let mut out: Vec<(&'static str, String, usize, f32)> = vec![
            ("token_embd.weight", "model.embed_tokens.weight".into(), VOCAB * D_MODEL, 1_000_000.0),
            ("output_norm.weight", "model.norm.weight".into(), D_MODEL, 2_000_000.0),
        ];
        if !tied {
            out.push(("output.weight", "lm_head.weight".into(), VOCAB * D_MODEL, 3_000_000.0));
        }
        for l in 0..N_LAYERS {
            let b = 100_000.0 * (l + 1) as f32;
            let hf = |s: &str| format!("model.layers.{l}.{s}");
            // The GGUF name is `blk.{l}.{leaf}`; `leak` gives the &'static str
            // the table wants, and this runs once per test.
            let g = |leaf: &str| -> &'static str { format!("blk.{l}.{leaf}").leak() };
            out.extend([
                (g("attn_norm.weight"), hf("input_layernorm.weight"), D_MODEL, b + 10.0),
                (g("attn_q.weight"), hf("self_attn.q_proj.weight"), hq * D_MODEL, b + 1000.0),
                (g("attn_k.weight"), hf("self_attn.k_proj.weight"), hkv * D_MODEL, b + 2000.0),
                (g("attn_v.weight"), hf("self_attn.v_proj.weight"), hkv * D_MODEL, b + 3000.0),
                (g("attn_q_norm.weight"), hf("self_attn.q_norm.weight"), HEAD_DIM, b + 4000.0),
                (g("attn_k_norm.weight"), hf("self_attn.k_norm.weight"), HEAD_DIM, b + 5000.0),
                (g("attn_output.weight"), hf("self_attn.o_proj.weight"), D_MODEL * hq, b + 6000.0),
                (g("ffn_norm.weight"), hf("post_attention_layernorm.weight"), D_MODEL, b + 7000.0),
                (g("ffn_gate.weight"), hf("mlp.gate_proj.weight"), D_FF * D_MODEL, b + 8000.0),
                (g("ffn_up.weight"), hf("mlp.up_proj.weight"), D_FF * D_MODEL, b + 9000.0),
                (g("ffn_down.weight"), hf("mlp.down_proj.weight"), D_MODEL * D_FF, b + 10000.0),
            ]);
        }
        out
    }

    /// Write the fixture as a **GGUF**, every tensor F32 (ggml type 0) so the
    /// comparison against the safetensors route is about the NAME MAP and
    /// nothing else - a quantized fixture would fold a lossy dequant into the
    /// same assertion and make bit-identity unavailable for no gain.
    ///
    pub fn write_synthetic_gguf(path: &str, tied: bool) {
        write_gguf(path, tied, &seq, None);
    }

    /// [`write_synthetic_gguf`] (untied) plus a `rope_freqs.weight` of
    /// `factors` - how llama.cpp stores a llama3 RoPE scaling.
    pub fn write_synthetic_gguf_with_rope_freqs(path: &str, factors: &[f32]) {
        write_gguf(path, false, &seq, Some(factors));
    }

    /// [`write_synthetic_gguf`] at small, well-conditioned values, for a test
    /// that runs the model: the identity-tagged values above overflow
    /// `mean(x^2)` in the first RMSNorm and every output is zero.
    pub fn write_conditioned_gguf(path: &str, tied: bool) {
        write_gguf(path, tied, &|base, n| (0..n).map(|i| 0.3 * (0.37 * base + 0.61 * i as f32).sin()).collect(), None);
    }

    fn write_gguf(path: &str, tied: bool, values: &dyn Fn(f32, usize) -> Vec<f32>, rope_freqs: Option<&[f32]>) {
        let tensors: Vec<TensorOut> = contents(tied)
            .into_iter()
            .map(|(gname, _, numel, base)| TensorOut {
                name: gname.to_string(),
                shape: vec![numel], // flat: only the element count is load-bearing here
                ty: 0,
                data: values(base, numel).iter().flat_map(|v| v.to_le_bytes()).collect(),
            })
            .chain(rope_freqs.map(|f| TensorOut {
                name: "rope_freqs.weight".to_string(),
                shape: vec![f.len()],
                ty: 0,
                data: f.iter().flat_map(|v| v.to_le_bytes()).collect(),
            }))
            .collect();

        // `token_embd.weight` is written flat above, but `config_from_gguf`
        // reads `vocab` off its leading dim - so give that one its real 2-D
        // shape. Every other tensor's rank is irrelevant to this importer.
        let tensors: Vec<TensorOut> = tensors
            .into_iter()
            .map(|mut t| {
                if t.name == "token_embd.weight" || t.name == "output.weight" {
                    t.shape = vec![VOCAB, D_MODEL];
                }
                t
            })
            .collect();

        let kv = |k: &str, v: GgufValue| (k.to_string(), v);
        let kvs = vec![
            kv("general.architecture", GgufValue::String(GGUF_ARCHITECTURE.to_string())),
            kv("qwen3.block_count", GgufValue::U32(N_LAYERS as u32)),
            kv("qwen3.embedding_length", GgufValue::U32(D_MODEL as u32)),
            kv("qwen3.feed_forward_length", GgufValue::U32(D_FF as u32)),
            kv("qwen3.attention.head_count", GgufValue::U32(N_HEADS as u32)),
            kv("qwen3.attention.head_count_kv", GgufValue::U32(N_KV_HEADS as u32)),
            kv("qwen3.attention.key_length", GgufValue::U32(HEAD_DIM as u32)),
            kv("qwen3.attention.value_length", GgufValue::U32(HEAD_DIM as u32)),
            kv("qwen3.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-6)),
            kv("qwen3.rope.freq_base", GgufValue::F32(1_000_000.0)),
            kv("qwen3.context_length", GgufValue::U32(40960)),
        ];
        write(path, &kvs, &tensors, 32).unwrap();
    }

    /// Write the SAME fixture as an HF checkpoint directory (`config.json` +
    /// `model.safetensors`), so the two import routes can be compared on
    /// identical logical content.
    pub fn write_synthetic_hf_dir(dir: &std::path::Path, tied: bool) {
        std::fs::create_dir_all(dir).unwrap();
        let json = format!(
            r#"{{"architectures":["Qwen3ForCausalLM"],"vocab_size":{VOCAB},"hidden_size":{D_MODEL},"num_hidden_layers":{N_LAYERS},
            "num_attention_heads":{N_HEADS},"num_key_value_heads":{N_KV_HEADS},"head_dim":{HEAD_DIM},
            "intermediate_size":{D_FF},"rope_theta":1000000,"rms_norm_eps":1e-6,
            "max_position_embeddings":40960,"tie_word_embeddings":{tied}}}"#
        );
        std::fs::write(dir.join("config.json"), json).unwrap();
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = contents(tied)
            .into_iter()
            .map(|(_, hname, numel, base)| (hname, vec![numel as u64], seq(base, numel)))
            .collect();
        checkpoint::st::save_safetensors(dir.join("model.safetensors").to_str().unwrap(), &tensors, &serde_json::Value::Null, None).unwrap();
    }
}

#[cfg(test)]
mod tests {

    /// The point of `open_source`: a quantized GGUF is readable under brain's
    /// own parameter names with NO ahead-of-time conversion. Before this
    /// existed, serving a GGUF meant `brain import` first, which wrote an
    /// fp32 checkpoint roughly 4x the GGUF's size (15 GiB from a 4 GiB Q8_0
    /// Qwen3-4B) purely to rename tensors.
    #[test]
    fn every_brain_parameter_reads_straight_out_of_an_untied_gguf() {
        use checkpoint::TensorSource;
        let dir = scratch("open-source-untied");
        let path = dir.join("m.gguf");
        let path = path.to_str().unwrap();
        testing::write_synthetic_gguf(path, false);

        let (cfg, src) = open_source(path).expect("a synthetic gguf opens");
        let missing: Vec<String> = cfg
            .param_list()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| !src.with_tensor(n, &mut |_| {}))
            .collect();
        assert!(missing.is_empty(), "unreadable brain parameters: {missing:?}");
    }

    /// A tied checkpoint carries no `output.weight` at all - the head IS the
    /// embedding table. The plan has to say so, or the head silently has no
    /// source and the model builds with an uninitialised `lm_head`.
    #[test]
    fn a_tied_gguf_sources_the_head_from_the_embedding_table() {
        use checkpoint::TensorSource;
        let dir = scratch("open-source-tied");
        let path = dir.join("tied.gguf");
        let path = path.to_str().unwrap();
        testing::write_synthetic_gguf(path, true);

        let (cfg, src) = open_source(path).expect("a synthetic tied gguf opens");
        assert!(cfg.tie_embeddings, "fixture is the tied one");
        let missing: Vec<String> = cfg
            .param_list()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| !src.with_tensor(n, &mut |_| {}))
            .collect();
        assert!(missing.is_empty(), "unreadable brain parameters: {missing:?}");
    }
    use super::testing::{write_conditioned_gguf, write_synthetic_gguf, write_synthetic_gguf_with_rope_freqs, write_synthetic_hf_dir, D_FF, D_MODEL, HEAD_DIM, N_HEADS, N_KV_HEADS, N_LAYERS, VOCAB};
    use checkpoint::gguf::GgufValue;
    use checkpoint::gguf_write::{write, TensorOut};

    fn seq(base: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| base + i as f32).collect()
    }

    /// What a resident serving a GGUF builds: `open_checkpoint` reads it under
    /// brain's names at the shape its KV metadata declares, and it decodes
    /// exactly as the brain checkpoint imported from it.
    #[test]
    fn a_gguf_decodes_exactly_as_its_import() {
        if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
            return;
        }
        let dir = scratch("open-decode");
        let gguf = dir.join("m.gguf");
        write_conditioned_gguf(gguf.to_str().unwrap(), false);
        let st = dir.join("m.safetensors");
        import_gguf(gguf.to_str().unwrap(), st.to_str().unwrap(), None).unwrap();
        let decode = |p: &std::path::Path| -> Vec<u32> {
            let (cfg, src) = crate::open_checkpoint(p.to_str().unwrap()).unwrap();
            assert_eq!(cfg.vocab, testing::VOCAB as u32);
            let shard = crate::Shard::whole(cfg.n_layers as usize);
            let m = crate::Qwen::new_shard_dt_decode(cfg, 8, &*src, shard, crate::Dtype::F32);
            [1u32, 3, 2].iter().flat_map(|&t| m.step(t)).map(f32::to_bits).collect()
        };
        let from_gguf = decode(&gguf);
        assert!(from_gguf.iter().any(|&b| f32::from_bits(b) != 0.0), "a decode of all zeros compares nothing");
        assert_eq!(from_gguf, decode(&st));
    }

    /// Every loader reads a checkpoint's config through `QwenConfig::from_reader`,
    /// whatever the file is: a GGUF's shape comes from its KV metadata (read
    /// as a brain config it is all defaults - the `tiny()` shape), a brain
    /// checkpoint's from its header, and a bare HF directory has no config
    /// the reader could see.
    #[test]
    fn from_reader_reads_the_shape_every_format_declares() {
        use crate::config::QwenConfig;
        use checkpoint::weightio::WeightReader;
        let dir = scratch("from-reader");
        let gguf = dir.join("m.gguf");
        write_synthetic_gguf(gguf.to_str().unwrap(), false);
        let from_gguf = QwenConfig::from_reader(&WeightReader::open(gguf.to_str().unwrap()).unwrap()).unwrap();
        assert_eq!(from_gguf, config_from_gguf(&MmapGguf::open(gguf.to_str().unwrap()).unwrap()).unwrap());
        assert_eq!(from_gguf.vocab, testing::VOCAB as u32);

        let st = dir.join("m.safetensors");
        import_gguf(gguf.to_str().unwrap(), st.to_str().unwrap(), None).unwrap();
        assert_eq!(QwenConfig::from_reader(&WeightReader::open(st.to_str().unwrap()).unwrap()).unwrap(), from_gguf);

        let hf = dir.join("hf");
        write_synthetic_hf_dir(&hf, false);
        let e = QwenConfig::from_reader(&WeightReader::open_hf_dir(&hf).unwrap()).unwrap_err();
        assert!(e.contains("config.json"), "{e}");
    }
    use super::*;
    use std::collections::HashMap;

    fn scratch(tag: &str) -> std::path::PathBuf {
        static COUNTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("brain-qwen3-gguf-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Every brain parameter of a checkpoint, read back from an imported
    /// safetensors file.
    fn read_back(path: &str, cfg: &QwenConfig) -> HashMap<String, Vec<f32>> {
        let r = checkpoint::weightio::WeightReader::open(path).unwrap();
        cfg.param_list().into_iter().map(|(n, _)| {
            let v = r.tensor(&n).unwrap_or_else(|| panic!("missing {n}"));
            (n, v)
        }).collect()
    }

    /// **The headline gate.** The GGUF route and the safetensors route must
    /// produce the *same brain parameters, bit for bit*, from the same logical
    /// checkpoint - not merely both "work".
    ///
    /// Bit-identity is available here because the fixture GGUF is F32: both
    /// routes copy the same f32 values under different source names, so
    /// anything but equality means the two disagree about which tensor is
    /// which. `assert_eq!` on the values, never a tolerance - a tolerance
    /// would pass a `k`/`v` swap on any layer whose two projections happened
    /// to be similar.
    #[test]
    fn the_gguf_route_and_the_safetensors_route_agree_bit_for_bit() {
        for tied in [false, true] {
            let dir = scratch(if tied { "parity-tied" } else { "parity-untied" });
            let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
            let hf = dir.join("hf");
            write_synthetic_gguf(&gguf, tied);
            write_synthetic_hf_dir(&hf, tied);

            // Both importers derive their own config from their own source;
            // they must agree about the model before they can agree about its
            // weights.
            let g_cfg = config_from_gguf(&MmapGguf::open(&gguf).unwrap()).unwrap();
            let h_cfg = crate::hf::decoder_config(&std::fs::read_to_string(hf.join("config.json")).unwrap()).unwrap();
            assert_eq!(g_cfg.to_json(), h_cfg.to_json(), "the two routes derive different configs (tied={tied})");

            let g_out = dir.join("from-gguf.safetensors").to_string_lossy().into_owned();
            let h_out = dir.join("from-hf.safetensors").to_string_lossy().into_owned();
            import_gguf(&gguf, &g_out, Some("test/qwen3-tiny")).expect("gguf import");
            crate::import::import(hf.to_str().unwrap(), &h_out).expect("hf import");

            let from_gguf = read_back(&g_out, &g_cfg);
            let from_hf = read_back(&h_out, &h_cfg);
            assert_eq!(from_gguf.len(), h_cfg.param_list().len());
            for (name, want) in &from_hf {
                assert_eq!(&from_gguf[name], want, "{name}: the two routes disagree (tied={tied})");
            }
            std::fs::remove_dir_all(&dir).ok();
        }
    }

    /// The failure mode the map is transcribed rather than inferred to avoid,
    /// demonstrated: perturb the name map and the bit-for-bit parity gate
    /// above must go red. This runs the same comparison that gate does against
    /// deliberately swapped maps and asserts the result DIFFERS from the
    /// safetensors route.
    ///
    /// Both swaps matter, for different reasons:
    ///
    /// - **q ↔ k** is caught by the element count alone, because Qwen3 is GQA
    ///   (`q_dim != kv_dim`). Real, but the easy case.
    /// - **k ↔ v** is the one that motivates this whole approach: `k_proj` and
    ///   `v_proj` have *identical* shapes on every layer of every GQA model,
    ///   so no coverage check, no shape check and no dry run can see it. The
    ///   import succeeds, the model runs, and the output is quietly wrong.
    ///   Only comparing VALUES against the other route catches it - which is
    ///   why the gate asserts on bytes rather than on a tolerance.
    ///
    /// A gate nobody has watched fail is a gate nobody knows is connected.
    #[test]
    fn a_swapped_name_map_is_caught_by_the_parity_gate() {
        let dir = scratch("swap");
        let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
        let hf = dir.join("hf");
        write_synthetic_gguf(&gguf, false);
        write_synthetic_hf_dir(&hf, false);

        let cfg = config_from_gguf(&MmapGguf::open(&gguf).unwrap()).unwrap();
        let h_out = dir.join("from-hf.safetensors").to_string_lossy().into_owned();
        crate::import::import(hf.to_str().unwrap(), &h_out).unwrap();
        let from_hf = read_back(&h_out, &cfg);
        let mg = MmapGguf::open(&gguf).unwrap();

        let swap = |a: &'static str, b: &'static str| {
            move |n: &str| -> Result<Mapped, String> {
                let n = if n.ends_with(a) {
                    n.replace(a, b)
                } else if n.ends_with(b) {
                    n.replace(b, a)
                } else {
                    n.to_string()
                };
                classify(&n, &tie_cfg(false), false)
            }
        };

        // q ↔ k: shapes disagree under GQA, so the import itself refuses.
        let err = gguf::import::to_map(&mg, &cfg.param_list(), &swap("attn_q.weight", "attn_k.weight"), "qwen3-mutant")
            .expect_err("a q/k swap must not import cleanly at a GQA shape");
        assert!(err.contains("element count"), "{err}");

        // k ↔ v: shapes AGREE, so it imports cleanly - and only the value
        // comparison can tell. This is the gate earning its keep.
        let mutant = gguf::import::to_map(&mg, &cfg.param_list(), &swap("attn_k.weight", "attn_v.weight"), "qwen3-mutant")
            .expect("a k/v swap is shape-compatible and WILL import cleanly - that is the whole problem");
        assert_eq!(mutant.len(), cfg.param_list().len());
        let differing: Vec<String> = cfg
            .param_list()
            .into_iter()
            .filter(|(n, _)| mutant[n] != from_hf[n])
            .map(|(n, _)| n)
            .collect();
        assert!(
            !differing.is_empty(),
            "a k/v swap must change the imported weights - if it does not, the parity gate proves nothing"
        );
        // ...and it must be exactly the k and v projections that moved.
        assert!(differing.iter().all(|n| n.ends_with("attn.wk.weight") || n.ends_with("attn.wv.weight")), "{differing:?}");
        assert_eq!(differing.len(), 2 * super::testing::N_LAYERS, "every layer's k and v must have moved: {differing:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two-way coverage, first direction: an unrecognized source tensor is an
    /// error, not a silent skip. A converter that renames a leaf must break
    /// the import loudly rather than write a checkpoint missing a projection.
    #[test]
    fn an_unrecognized_tensor_is_refused_by_name() {
        let err = classify("blk.0.attn_wibble.weight", &tie_cfg(false), false).unwrap_err();
        assert!(err.contains("attn_wibble"), "{err}");
        // ...while the two real drops are decisions with stated reasons.
        assert!(matches!(classify("rope_freqs.weight", &tie_cfg(false), false), Ok(Mapped::Dropped(_))));
        assert!(matches!(classify("output.weight", &tie_cfg(true), false), Ok(Mapped::Dropped(_))));
        assert!(matches!(classify("output.weight", &tie_cfg(false), false), Ok(Mapped::Simple(_))));
    }

    /// Two-way coverage, second direction, and the drop accounting: every
    /// planned brain parameter is written exactly once, and the one dropped
    /// source tensor is COUNTED rather than lost.
    #[test]
    fn import_covers_every_planned_tensor_and_counts_the_rope_freqs_drop() {
        let dir = scratch("coverage");
        let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
        write_synthetic_gguf_with_rope_freqs(&gguf, &[1.0, 2.0]);
        let out = dir.join("out.safetensors").to_string_lossy().into_owned();

        let stats = import_gguf(&gguf, &out, None).unwrap();
        let cfg = config_from_gguf(&MmapGguf::open(&gguf).unwrap()).unwrap();
        assert_eq!(stats.written, cfg.param_list().len());
        assert_eq!(stats.dropped.get(DROP_ROPE_FREQS), Some(&1), "the dropped tensor must be on the record: {stats}");
        let written = QwenConfig::from_json_checked(&checkpoint::read_config(&out)).unwrap();
        assert_eq!(written.rope_scaling, cfg.rope_scaling, "the factors travel in the written config");
        assert_eq!(stats.source_tensors, stats.written + 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A checkpoint deeper than the config claims is the classic
    /// wrong-checkpoint mistake, and must be refused rather than silently
    /// truncated. The name map has no depth opinion (`u32::MAX`), so this is
    /// the coverage check doing its job.
    #[test]
    fn a_checkpoint_deeper_than_the_config_is_refused() {
        let dir = scratch("deeper");
        let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
        write_synthetic_gguf(&gguf, false);
        let mg = MmapGguf::open(&gguf).unwrap();
        let mut cfg = config_from_gguf(&mg).unwrap();
        cfg.n_layers = 1; // the file has 2

        let err = gguf::import::dry_run(&mg, &cfg.param_list(), &|n| classify(n, &tie_cfg(false), false), "qwen3").unwrap_err();
        assert!(err.contains("blocks.1."), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_from_gguf_matches_the_synthetic_header() {
        let dir = scratch("cfg");
        let gguf = dir.join("m.gguf").to_string_lossy().into_owned();
        write_synthetic_gguf(&gguf, false);
        let cfg = config_from_gguf(&MmapGguf::open(&gguf).unwrap()).unwrap();
        assert_eq!(cfg.vocab, 5);
        assert_eq!(cfg.n_layers, 2);
        assert_eq!(cfg.d_model, 6);
        assert_eq!(cfg.n_heads, 2);
        assert_eq!(cfg.n_kv_heads, 1);
        assert_eq!(cfg.head_dim, 4);
        assert_eq!(cfg.d_ff, 8);
        assert_eq!(cfg.max_position_embeddings, 40960);
        assert_eq!(cfg.block_size, 2048, "buffers are sized by block_size, never by the trained RoPE extent");
        assert!(!cfg.tie_embeddings, "output.weight is present -> untied");
        assert!(cfg.qk_norm);
        assert!(!cfg.attn_bias);

        // ...and tying is stated by OMITTING output.weight.
        let tied_path = dir.join("tied.gguf").to_string_lossy().into_owned();
        write_synthetic_gguf(&tied_path, true);
        assert!(config_from_gguf(&MmapGguf::open(&tied_path).unwrap()).unwrap().tie_embeddings);

        // A GGUF that is not a qwen3 is refused by name, not silently defaulted.
        std::fs::remove_dir_all(&dir).ok();
    }

    /// llama.cpp's `LlamaModel.permute` (`conversion/llama.py`), transcribed:
    /// `w.reshape(n_head, 2, rows / n_head / 2, cols).swapaxes(1, 2).reshape(rows, cols)`.
    /// A config whose only relevant field here is its tying.
    fn tie_cfg(tie: bool) -> QwenConfig {
        QwenConfig { tie_embeddings: tie, ..QwenConfig::tiny() }
    }

    fn llamacpp_permute(w: &[f32], rows: usize, cols: usize, n_head: usize) -> Vec<f32> {
        let half = rows / n_head / 2;
        let mut out = vec![0.0; w.len()];
        for h in 0..n_head {
            for two in 0..2 {
                for i in 0..half {
                    let src = (h * 2 + two) * half + i; // [h][two][i]
                    let dst = (h * half + i) * 2 + two; // [h][i][two]
                    out[dst * cols..(dst + 1) * cols].copy_from_slice(&w[src * cols..(src + 1) * cols]);
                }
            }
        }
        out
    }

    #[test]
    fn unpermute_inverts_llamacpp_permute() {
        for (n_head, head_dim, cols) in [(2usize, 4usize, 3usize), (4, 8, 5), (1, 2, 1)] {
            let rows = n_head * head_dim;
            let hf: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
            let gguf = llamacpp_permute(&hf, rows, cols, n_head);
            let order = llama_unpermute_order(n_head, head_dim);
            let back: Vec<f32> = order.iter().flat_map(|&o| gguf[o as usize * cols..(o as usize + 1) * cols].to_vec()).collect();
            assert_eq!(back, hf, "n_head {n_head} head_dim {head_dim}");
        }
    }

    /// A llama-architecture GGUF (q/k stored permuted, no QK-norm, linear
    /// RoPE scaling in its KV) reads back as the checkpoint it was converted
    /// from, under brain's names and a Llama config.
    #[test]
    fn a_llama_gguf_reads_as_the_checkpoint_it_was_converted_from() {
        use checkpoint::TensorSource;
        let dir = scratch("llama-gguf");
        let path = dir.join("m.gguf");
        let path = path.to_str().unwrap();
        let (hq, hkv) = (N_HEADS * HEAD_DIM, N_KV_HEADS * HEAD_DIM);
        let mut hf: Vec<(String, String, Vec<usize>, Vec<f32>)> = vec![
            ("token_embd.weight".into(), "tok.weight".into(), vec![VOCAB, D_MODEL], seq(1_000.0, VOCAB * D_MODEL)),
            ("output_norm.weight".into(), "norm.weight".into(), vec![D_MODEL], seq(2_000.0, D_MODEL)),
            ("output.weight".into(), "lm_head.weight".into(), vec![VOCAB, D_MODEL], seq(3_000.0, VOCAB * D_MODEL)),
        ];
        for l in 0..N_LAYERS {
            let b = 10_000.0 * (l + 1) as f32;
            let t = |g: &str, br: &str, shape: Vec<usize>, base: f32| (format!("blk.{l}.{g}"), format!("blocks.{l}.{br}"), shape.clone(), seq(base, shape.iter().product()));
            hf.extend([
                t("attn_norm.weight", "ln1.weight", vec![D_MODEL], b),
                t("attn_q.weight", "attn.wq.weight", vec![hq, D_MODEL], b + 100.0),
                t("attn_k.weight", "attn.wk.weight", vec![hkv, D_MODEL], b + 200.0),
                t("attn_v.weight", "attn.wv.weight", vec![hkv, D_MODEL], b + 300.0),
                t("attn_output.weight", "attn.wo.weight", vec![D_MODEL, hq], b + 400.0),
                t("ffn_norm.weight", "ln2.weight", vec![D_MODEL], b + 500.0),
                t("ffn_gate.weight", "mlp.gate.weight", vec![D_FF, D_MODEL], b + 600.0),
                t("ffn_up.weight", "mlp.up.weight", vec![D_FF, D_MODEL], b + 700.0),
                t("ffn_down.weight", "mlp.down.weight", vec![D_MODEL, D_FF], b + 800.0),
            ]);
        }
        let tensors: Vec<TensorOut> = hf
            .iter()
            .map(|(g, _, shape, v)| {
                let stored = if g.ends_with("attn_q.weight") {
                    llamacpp_permute(v, hq, D_MODEL, N_HEADS)
                } else if g.ends_with("attn_k.weight") {
                    llamacpp_permute(v, hkv, D_MODEL, N_KV_HEADS)
                } else {
                    v.clone()
                };
                TensorOut { name: g.clone(), shape: shape.clone(), ty: 0, data: stored.iter().flat_map(|x| x.to_le_bytes()).collect() }
            })
            .collect();
        let kv = |k: &str, v: GgufValue| (k.to_string(), v);
        let kvs = vec![
            kv("general.architecture", GgufValue::String("llama".into())),
            kv("llama.block_count", GgufValue::U32(N_LAYERS as u32)),
            kv("llama.embedding_length", GgufValue::U32(D_MODEL as u32)),
            kv("llama.feed_forward_length", GgufValue::U32(D_FF as u32)),
            kv("llama.attention.head_count", GgufValue::U32(N_HEADS as u32)),
            kv("llama.attention.head_count_kv", GgufValue::U32(N_KV_HEADS as u32)),
            kv("llama.rope.dimension_count", GgufValue::U32(HEAD_DIM as u32)),
            kv("llama.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-5)),
            kv("llama.rope.freq_base", GgufValue::F32(100_000.0)),
            kv("llama.rope.scaling.type", GgufValue::String("linear".into())),
            kv("llama.rope.scaling.factor", GgufValue::F32(4.0)),
            kv("llama.context_length", GgufValue::U32(16384)),
        ];
        write(path, &kvs, &tensors, 32).unwrap();

        let (cfg, src) = open_source(path).expect("a llama gguf opens");
        assert!(!cfg.qk_norm && !cfg.attn_bias && !cfg.tie_embeddings);
        assert_eq!((cfg.head_dim, cfg.rms_eps, cfg.rope_theta), (HEAD_DIM as u32, 1e-5, 100_000.0));
        assert_eq!(cfg.rope_scaling, Some(model::rope_scaling::RopeScaling::Linear { factor: 4.0 }));
        for (_, brain, _, want) in &hf {
            let mut got = Vec::new();
            assert!(src.with_tensor(brain, &mut |d| got = d.to_vec()), "{brain} missing");
            assert_eq!(&got, want, "{brain}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// llama.cpp stores a llama3 RoPE scaling as `rope_freqs.weight`, a
    /// divisor per frequency; it is the config's scaling, never dropped.
    #[test]
    fn rope_freqs_is_the_configs_rope_scaling() {
        let dir = scratch("rope-freqs");
        let path = dir.join("m.gguf");
        write_synthetic_gguf_with_rope_freqs(path.to_str().unwrap(), &[1.0, 2.5]);
        let cfg = config_from_gguf(&MmapGguf::open(path.to_str().unwrap()).unwrap()).unwrap();
        assert_eq!(cfg.rope_scaling, Some(model::rope_scaling::RopeScaling::Factors(vec![1.0, 2.5])));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
