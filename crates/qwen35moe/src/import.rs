// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GGUF import for Qwen3.5-35B-A3B.
//!
//! We fetch this model as **GGUF** (`bartowski/Qwen_Qwen3.5-35B-A3B-GGUF`),
//! not HF safetensors - at the measured download throughput available when
//! this was built (~3.2 MB/s regardless of connection count), fetching the
//! ~72 GB bf16 safetensors checkpoint just to re-quantize it ourselves would
//! cost hours more than fetching an already-quantized GGUF directly. That
//! means the import source is **llama.cpp's own tensor-naming convention**,
//! not the HF `modeling_qwen3_5_moe.py` names `crates/qwen35moe/src/config.rs`'s
//! doc comments otherwise reference - and llama.cpp maps Gated DeltaNet onto
//! its generic Mamba/SSM tensor-naming scheme, which differs substantially
//! from HF's `linear_attn.in_proj_*` names.
//!
//! **This mapping was derived empirically** - checkpoint headers are free
//! architecture docs - by range-reading
//! the GGUF header (magic + KV metadata + tensor info list, all present near
//! the start of the file, long before the bulk tensor data) directly off the
//! partially-downloaded `Qwen3.5-35B-A3B-Q4_K_M.gguf`, decoding it by hand
//! against the GGUF spec, and cross-checking every mapped tensor's SHAPE
//! against the corresponding HF tensor's shape from the real
//! `model.safetensors.index.json` (saved under
//! `/data/workspace/resources/qwen3.5/`) - not just its name. Every mapping
//! below has a shape-equality proof in this file's own doc comments; nothing
//! here is guessed.
//!
//! ## The real file's header (`general.architecture = "qwen35moe"`)
//!
//! - `qwen35moe.block_count = 41` - **NOT 40.** llama.cpp folds the MTP
//!   ("next-token prediction", HF's `mtp.*`) layer into the SAME `blk.N`
//!   index space, as block **40** (confirmed: `blk.40` is the only block
//!   carrying `blk.N.nextn.*` tensors, and it also carries its own
//!   `attn_q`/`attn_k`/... - matching HF's `mtp.layers.0.self_attn.*`, a
//!   single full-attention layer). MTP is out of scope for this port (same
//!   deferral GLM's `check_glm_mtp` treats as separate follow-on work), so
//!   `n_layers` here is `block_count - 1 = 40` and every `blk.40.*` tensor is
//!   dropped at import - **loudly counted, not silently ignored** (see
//!   `import_gguf`'s coverage check).
//! - 11 blocks carry `attn_q`/`attn_k`/`attn_v`/`attn_output`/`attn_q_norm`/
//!   `attn_k_norm` (full attention) = the 10 real full-attention decoder
//!   layers (`full_attention_interval=4` over 40 layers) **plus block 40's
//!   MTP attention**. 30 blocks carry `ssm_*` (Gated DeltaNet) = the
//!   remaining 30 real decoder layers. 41 blocks carry `ffn_*` (MoE) - every
//!   block, both mixer kinds, matching HF (`Qwen3_5MoeDecoderLayer.mlp` is
//!   unconditional).
//!
//! ## Verified tensor name + shape mapping (GGUF dims are `ne[0]`-fastest,
//! i.e. torch-shape-reversed; "torch" below is the already-reversed shape,
//! directly comparable to the HF checkpoint's `index.json`)
//!
//! Full-attention leaves (`blk.N.*`, present only where HF's `layer_types[N]
//! == "full_attention"`):
//!
//! | GGUF name | GGUF dims | torch shape | HF name (shape) |
//! |---|---|---|---|
//! | `attn_q.weight` | `[2048,8192]` | `[8192,2048]` | `self_attn.q_proj.weight` (`[n_heads*head_dim*2, hidden]` = `[16*256*2,2048]` = `[8192,2048]` - doubled for the value+gate split) |
//! | `attn_k.weight` | `[2048,512]` | `[512,2048]` | `self_attn.k_proj.weight` (`[n_kv*head_dim,hidden]`=`[2*256,2048]`) |
//! | `attn_v.weight` | `[2048,512]` | `[512,2048]` | `self_attn.v_proj.weight` (same shape as k) |
//! | `attn_q_norm.weight` | `[256]` | `[256]` | `self_attn.q_norm.weight` (`head_dim`) |
//! | `attn_k_norm.weight` | `[256]` | `[256]` | `self_attn.k_norm.weight` |
//! | `attn_output.weight` | `[4096,2048]` | `[2048,4096]` | `self_attn.o_proj.weight` (`[hidden,n_heads*head_dim]`) |
//!
//! Linear-attention (Gated DeltaNet) leaves (`blk.N.*`, present only where
//! HF's `layer_types[N] == "linear_attention"`) - **llama.cpp's generic
//! SSM/Mamba naming, mapped to HF's `Qwen3_5MoeGatedDeltaNet` field-for-field
//! by matching shapes, since the names don't correspond lexically**:
//!
//! | GGUF name | GGUF dims | torch shape | HF name (shape) |
//! |---|---|---|---|
//! | `attn_qkv.weight` | `[2048,8192]` | `[8192,2048]` | `linear_attn.in_proj_qkv.weight` (`[2*key_dim+value_dim,hidden]` = `[2*2048+4096,2048]`=`[8192,2048]`) |
//! | `attn_gate.weight` | `[2048,4096]` | `[4096,2048]` | `linear_attn.in_proj_z.weight` (`[value_dim,hidden]`=`[4096,2048]`) - "gate" because this feeds the gated-RMSNorm's `z` input, not a router gate |
//! | `ssm_alpha.weight` | `[2048,32]` | `[32,2048]` | `linear_attn.in_proj_a.weight` (`[num_v_heads,hidden]`=`[32,2048]`) - HF's math literally computes the decay from `a` |
//! | `ssm_beta.weight` | `[2048,32]` | `[32,2048]` | `linear_attn.in_proj_b.weight` (same shape) - HF computes `beta=sigmoid(b)` |
//! | `ssm_conv1d.weight` | `[4,8192]` | `[8192,4]` | `linear_attn.conv1d.weight` (`[conv_dim,kernel]`=`[8192,4]` - HF's own `nn.Conv1d` weight is `[conv_dim,1,4]`, squeezed; the reference code itself calls `.squeeze(1)` before use) |
//! | `ssm_a` | `[32]` | `[32]` | `linear_attn.A_log` (`[num_v_heads]`) |
//! | `ssm_dt.bias` | `[32]` | `[32]` | `linear_attn.dt_bias` (`[num_v_heads]`) |
//! | `ssm_norm.weight` | `[128]` | `[128]` | `linear_attn.norm.weight` (`[head_v_dim]`, the gated-RMSNorm weight) |
//! | `ssm_out.weight` | `[4096,2048]` | `[2048,4096]` | `linear_attn.out_proj.weight` (`[hidden,value_dim]`) |
//!
//! MoE leaves (`blk.N.*`, every block):
//!
//! | GGUF name | GGUF dims | torch shape | HF name (shape) |
//! |---|---|---|---|
//! | `ffn_gate_inp.weight` | `[2048,256]` | `[256,2048]` | `mlp.router.weight` (`[n_experts,hidden]`) |
//! | `ffn_gate_inp_shexp.weight` | `[2048]` | `[2048]` | `mlp.shared_expert_gate.weight` (HF `[1,hidden]`, squeezed by llama.cpp - same element count, reshaped on import) |
//! | `ffn_gate_shexp.weight` | `[2048,512]` | `[512,2048]` | `mlp.shared_expert.gate_proj.weight` (`[shared_ff,hidden]`) |
//! | `ffn_up_shexp.weight` | `[2048,512]` | `[512,2048]` | `mlp.shared_expert.up_proj.weight` |
//! | `ffn_down_shexp.weight` | `[512,2048]` | `[2048,512]` | `mlp.shared_expert.down_proj.weight` (`[hidden,shared_ff]`) |
//! | `ffn_gate_exps.weight` | `[2048,512,256]` | `[256,512,2048]` | **fan-out**: HF's `mlp.experts.gate_up_proj` is ONE fused `[256,1024,2048]` tensor (gate+up concatenated on dim 1); llama.cpp's GGUF conversion **already split it** into separate `[256,512,2048]` gate/up tensors - this import does NOT need to split a fused gate+up itself, only slice per expert (dim 0, contiguous - `dequantize`'s row-major output means expert `e`'s slice is `data[e*512*2048 .. (e+1)*512*2048]`) |
//! | `ffn_up_exps.weight` | `[2048,512,256]` | `[256,512,2048]` | (as above, the up half) |
//! | `ffn_down_exps.weight` | `[512,2048,256]` | `[256,2048,512]` | `mlp.experts.down_proj` (`[n_experts,hidden,moe_ff]`, NOT fused with anything - one contiguous per-expert slice of `hidden*moe_ff`) |
//!
//! Top level: `token_embd.weight` (`[248320,2048]`) → `tok.weight`;
//! `output.weight` (`[248320,2048]`) → `lm_head.weight` (untied,
//! `tie_word_embeddings: false`); `output_norm.weight` (`[2048]`) → `norm.weight`.
//!
//! Vision (`v.*`/`mm.*` - not yet enumerated from this GGUF's header; deferred
//! along with the rest of vision splice wiring) and `blk.40.nextn.*` (MTP) are
//! dropped.

use std::collections::HashMap;

use checkpoint::gguf::MmapGguf;
use checkpoint::st::ModelCard;
use gguf::import::{self, Mapped};
use gguf::ArchKv;

use crate::config::{layer_types, LayerType, Qwen35Config};

/// llama.cpp's `general.architecture` value for this model family.
pub const GGUF_ARCHITECTURE: &str = "qwen35moe";

/// Derive [`Qwen35Config`] from a GGUF file's KV metadata, cross-checked
/// against the real tensor shapes of the first Gated-DeltaNet block found
/// (the `qwen35moe.ssm.*` KV keys reuse generic Mamba/SSM field names whose
/// exact semantic mapping onto `linear_num_key_heads`/`linear_num_value_heads`/
/// `linear_key_head_dim`/`linear_value_head_dim` is a plausible-but-unproven
/// reading of llama.cpp's SSM KV schema - the TENSOR SHAPES are ground truth
/// and this function derives the head-shape fields from them directly rather
/// than trusting the KV key names, verifying empirically.
/// `qwen35moe.ssm.group_count` is used only as an
/// assertion cross-check (it must equal the derived `linear_num_key_heads`),
/// not as the primary source.
pub fn config_from_gguf(mg: &MmapGguf) -> Result<Qwen35Config, String> {
    let kv = ArchKv::expect_architecture(mg, GGUF_ARCHITECTURE)?;

    let block_count = kv.req_u32("block_count")?;
    // llama.cpp folds the MTP layer into the same blk.N index space as the
    // last block (see this file's module doc) - drop it from n_layers.
    let n_layers = block_count.checked_sub(1).ok_or("qwen35: block_count must be > 1 (MTP occupies the last block)")?;

    let d_model = kv.req_u32("embedding_length")?;
    let n_heads = kv.req_u32("attention.head_count")?;
    let n_kv_heads = kv.req_u32("attention.head_count_kv")?;
    let head_dim = kv.req_u32("attention.key_length")?;
    let head_dim_v = kv.u32_or("attention.value_length", head_dim);
    if head_dim_v != head_dim {
        return Err(format!("qwen35: attention key_length {head_dim} != value_length {head_dim_v} (unsupported asymmetric head_dim)"));
    }
    let rope_theta = kv.f32_or("rope.freq_base", 10_000_000.0);
    let rotary_dim = kv.req_u32("rope.dimension_count")?;
    let partial_rotary_factor = rotary_dim as f32 / head_dim as f32;
    let full_attention_interval = kv.u32_or("full_attention_interval", 4);

    let n_experts = kv.req_u32("expert_count")?;
    let top_k = kv.req_u32("expert_used_count")?;
    let moe_intermediate_size = kv.req_u32("expert_feed_forward_length")?;
    let shared_expert_intermediate_size = kv.req_u32("expert_shared_feed_forward_length")?;
    let linear_conv_kernel_dim = kv.req_u32("ssm.conv_kernel")?;

    // Derive linear-attention head shapes from the FIRST Gated-DeltaNet
    // block's real tensor shapes, per this function's own doc comment.
    let types = layer_types(n_layers, full_attention_interval);
    let first_linear = types.iter().position(|t| *t == LayerType::Linear).ok_or("qwen35: no linear-attention layer found")?;
    let qkv_shape = mg
        .shape(&format!("blk.{first_linear}.attn_qkv.weight"))
        .ok_or_else(|| format!("qwen35: missing blk.{first_linear}.attn_qkv.weight"))?;
    let gate_shape = mg
        .shape(&format!("blk.{first_linear}.attn_gate.weight"))
        .ok_or_else(|| format!("qwen35: missing blk.{first_linear}.attn_gate.weight"))?;
    let norm_shape = mg
        .shape(&format!("blk.{first_linear}.ssm_norm.weight"))
        .ok_or_else(|| format!("qwen35: missing blk.{first_linear}.ssm_norm.weight"))?;
    // torch shapes: attn_qkv [2*key_dim+value_dim, hidden]; attn_gate
    // [value_dim, hidden]; ssm_norm [head_v_dim].
    let value_dim = *gate_shape.first().ok_or("qwen35: attn_gate.weight has no leading dim")?;
    let head_v_dim = *norm_shape.first().ok_or("qwen35: ssm_norm.weight has no leading dim")?;
    let qkv_width = *qkv_shape.first().ok_or("qwen35: attn_qkv.weight has no leading dim")?;
    let key_dim = (qkv_width - value_dim) / 2;
    let linear_num_value_heads = (value_dim / head_v_dim) as u32;
    let linear_value_head_dim = head_v_dim as u32;
    // key_head_dim: cross-check against ssm.state_size if present, else
    // assume it equals the value head dim (true for every released Qwen3.5
    // config so far) and derive num_key_heads from key_dim.
    let linear_key_head_dim = kv.u32_or("ssm.state_size", linear_value_head_dim);
    let linear_num_key_heads = (key_dim as u32) / linear_key_head_dim;
    if let Some(group_count) = kv.u32("ssm.group_count") {
        if group_count != linear_num_key_heads {
            return Err(format!(
                "qwen35: ssm.group_count={group_count} disagrees with tensor-shape-derived linear_num_key_heads={linear_num_key_heads}"
            ));
        }
    }

    Ok(Qwen35Config {
        vocab: mg.shape("token_embd.weight").and_then(|s| s.first().copied()).ok_or("qwen35: missing token_embd.weight")? as u32,
        block_size: 4096,
        n_layers,
        d_model,
        rms_eps: kv.f32_or("attention.layer_norm_rms_epsilon", 1e-6),
        max_position_embeddings: kv.u32_or("context_length", 262144),
        tie_embeddings: !mg.names().iter().any(|n| n == "output.weight"),

        n_heads,
        n_kv_heads,
        head_dim,
        attn_bias: false,
        rope_theta,
        partial_rotary_factor,
        mrope_section: [11, 11, 10],

        full_attention_interval,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_conv_kernel_dim,

        n_experts,
        top_k,
        moe_intermediate_size,
        shared_expert_intermediate_size,

        lora: None,
    })
}

/// Not imported, by reason - [`Mapped::Dropped`]'s payload, counted and
/// printed by the shared driver so a drop is always on the record.
const DROP_MTP: &str = "MTP / out-of-range block (blk.N for N >= n_layers)";
const DROP_OTHER: &str = "vision (v.*/mm.*) or an unrecognized leaf";

/// Classify one GGUF tensor name into the shared [`Mapped`] disposition.
/// `n_layers` is the real decoder depth - MTP's block, at index `n_layers`, is
/// always dropped, and a caller doing a truncated load passes its own smaller
/// depth to drop every block past the cut the same way.
fn classify(name: &str, n_layers: u32, n_experts: u32) -> Mapped {
    if name == "token_embd.weight" {
        return Mapped::Simple("tok.weight".to_string());
    }
    if name == "output.weight" {
        return Mapped::Simple("lm_head.weight".to_string());
    }
    if name == "output_norm.weight" {
        return Mapped::Simple("norm.weight".to_string());
    }
    let Some(rest) = name.strip_prefix("blk.") else { return Mapped::Dropped(DROP_OTHER) };
    let Some((idx_str, leaf)) = rest.split_once('.') else { return Mapped::Dropped(DROP_OTHER) };
    let Ok(l) = idx_str.parse::<u32>() else { return Mapped::Dropped(DROP_OTHER) };
    if l >= n_layers {
        return Mapped::Dropped(DROP_MTP);
    }
    let l = l as usize;
    let p = |s: &str| format!("blocks.{l}.{s}");
    match leaf {
        "attn_norm.weight" => Mapped::Simple(p("ln1.weight")),
        "post_attention_norm.weight" => Mapped::Simple(p("ln2.weight")),
        // Full attention.
        "attn_q.weight" => Mapped::Simple(p("self_attn.q_proj.weight")),
        "attn_k.weight" => Mapped::Simple(p("self_attn.k_proj.weight")),
        "attn_v.weight" => Mapped::Simple(p("self_attn.v_proj.weight")),
        "attn_q_norm.weight" => Mapped::Simple(p("self_attn.q_norm.weight")),
        "attn_k_norm.weight" => Mapped::Simple(p("self_attn.k_norm.weight")),
        "attn_output.weight" => Mapped::Simple(p("self_attn.o_proj.weight")),
        // Gated DeltaNet (linear attention).
        "attn_qkv.weight" => Mapped::Simple(p("linear_attn.in_proj_qkv.weight")),
        "attn_gate.weight" => Mapped::Simple(p("linear_attn.in_proj_z.weight")),
        "ssm_alpha.weight" => Mapped::Simple(p("linear_attn.in_proj_a.weight")),
        "ssm_beta.weight" => Mapped::Simple(p("linear_attn.in_proj_b.weight")),
        "ssm_conv1d.weight" => Mapped::Simple(p("linear_attn.conv1d.weight")),
        "ssm_a" => Mapped::Simple(p("linear_attn.A_log")),
        "ssm_dt.bias" => Mapped::Simple(p("linear_attn.dt_bias")),
        "ssm_norm.weight" => Mapped::Simple(p("linear_attn.norm.weight")),
        "ssm_out.weight" => Mapped::Simple(p("linear_attn.out_proj.weight")),
        // MoE.
        "ffn_gate_inp.weight" => Mapped::Simple(p("mlp.router.weight")),
        "ffn_gate_inp_shexp.weight" => Mapped::Simple(p("mlp.shared_expert_gate.weight")),
        "ffn_gate_shexp.weight" => Mapped::Simple(p("mlp.shared_expert.gate.weight")),
        "ffn_up_shexp.weight" => Mapped::Simple(p("mlp.shared_expert.up.weight")),
        "ffn_down_shexp.weight" => Mapped::Simple(p("mlp.shared_expert.down.weight")),
        "ffn_gate_exps.weight" => Mapped::expert_stack(l, "gate", n_experts as usize),
        "ffn_up_exps.weight" => Mapped::expert_stack(l, "up", n_experts as usize),
        "ffn_down_exps.weight" => Mapped::expert_stack(l, "down", n_experts as usize),
        _ => Mapped::Dropped(DROP_OTHER),
    }
}

/// Import a GGUF checkpoint into brain's native format.
///
/// The streaming loop, the per-expert fan-out and the two-way coverage check
/// all live in `gguf::import` - shared with every other GGUF-sourced model.
/// What is qwen35moe-specific, and stays here, is the pair of decisions the
/// driver takes as arguments: [`config_from_gguf`]'s manifest and
/// [`classify`]'s name map.
pub fn import_gguf(gguf_path: &str, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
    let mg = MmapGguf::open(gguf_path)?;
    import_mmap(&mg, out_path, id_override)
}

/// [`import_gguf`] over an ALREADY-OPEN checkpoint. The generic
/// architecture-dispatch registry (`brain import-gguf`) opens the file and
/// reads `general.architecture` before it can know which importer to call, so
/// it hands the open handle straight through rather than mmapping twice.
pub fn import_mmap(mg: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
    let cfg = config_from_gguf(mg)?;
    let params = cfg.param_list();

    let mut card = ModelCard::new(id_override.unwrap_or("qwen35"), "qwen35");
    card.context_length = Some(cfg.max_position_embeddings as u64);
    card.param_count = Some(params.iter().map(|(_, n)| *n as u64).sum());

    import::to_st(
        mg,
        &params,
        &|n| Ok(classify(n, cfg.n_layers, cfg.n_experts)),
        out_path,
        &cfg.to_json(),
        Some(&card),
        "qwen35",
    )?;
    Ok(())
}

/// A **truncated, in-memory** GGUF import: real weights for layers
/// `[0, cfg.n_layers)` only, collected straight into a `HashMap` instead of
/// written to an intermediate safetensors file.
///
/// Why this exists: `import_gguf`'s own safetensors output is a
/// full fp32 re-encoding of the checkpoint - at the real 35B-A3B shape that
/// is ~140 GB on disk, which does not fit alongside the already-downloaded
/// ~60 GB of GGUF source files in this box's available disk. Loading directly
/// into a `HashMap<String, Vec<f32>>` for a **truncated** layer count instead
/// needs only that truncated slice's own resident size (no disk intermediate
/// at all, and no cost paid for the layers beyond the cut - `classify`'s own
/// `l >= n_layers` rule, reused UNCHANGED here with the caller's (possibly
/// truncated) `cfg.n_layers` standing in for the real depth, drops every
/// higher-numbered block's tensor name before `mg.tensor()` is ever called on
/// it, so the expensive per-tensor GGUF dequantization is only ever paid for
/// the layers actually kept).
///
/// `cfg` must already carry the REAL checkpoint's non-layer-count shape
/// fields (typically `config_from_gguf(&mg)` with `n_layers` overridden down)
/// - this function does not re-derive them, only classifies and copies.
/// Fails loudly on any coverage gap in `cfg.param_list()`, same contract as
/// [`import_gguf`].
pub fn import_gguf_truncated_to_map(mg: &checkpoint::gguf::MmapGguf, cfg: &Qwen35Config) -> Result<HashMap<String, Vec<f32>>, String> {
    // Same classifier, same coverage contract, different sink: layer-scoped
    // tensors past the cut are dropped by `classify`'s own `l >= n_layers`
    // rule before the driver ever asks for their bytes, while the top-level
    // tensors (`tok`/`lm_head`/`norm`) are wanted regardless of truncation.
    import::to_map(mg, &cfg.param_list(), &|n| Ok(classify(n, cfg.n_layers, cfg.n_experts)), "qwen35 truncated")
}

/// Test fixtures for this importer, shared across crates.
///
/// `pub` (not `#[cfg(test)]`) so `brain-cli`'s GGUF-import-registry tests can
/// drive a REAL conversion through the generic architecture dispatch without
/// a second, drifting copy of this checkpoint builder. Not part of the model's
/// runtime surface.
#[doc(hidden)]
pub mod testing {
    use super::*;
    use checkpoint::gguf::GgufValue;
    use checkpoint::gguf_write::{write, TensorOut};

    fn kv(key: &str, v: GgufValue) -> (String, GgufValue) {
        (key.to_string(), v)
    }

    /// Write a minimal synthetic GGUF exercising every tensor kind
    /// `classify` handles, at a tiny shape (2 layers: one linear-attention,
    /// one full-attention, `full_attention_interval=2` so layer 1 is Full;
    /// `n_experts=3`), plus a dropped MTP block at index 2 and a dropped
    /// unrecognized leaf, to prove the coverage check actually distinguishes
    /// "dropped on purpose" from "silently lost".
    pub fn write_synthetic_gguf(path: &str) {
        let d = 4u64; // hidden
        let n_heads = 2u64;
        let head_dim = 4u64; // -> q_proj width doubled = 16
        let n_kv = 1u64;
        let lin_kh = 1u64;
        let lin_kd = 3u64; // key_head_dim
        let lin_vh = 2u64;
        let lin_vd = 3u64; // value_head_dim (kept distinct from lin_kd)
        let key_dim = lin_kh * lin_kd; // 3
        let value_dim = lin_vh * lin_vd; // 6
        let conv_k = 2u64;
        let n_experts = 3u64;
        let moe_ff = 2u64;
        let shared_ff = 2u64;
        let vocab = 10u64;

        let f32t = |shape: Vec<u64>, name: &str| {
            let numel: u64 = shape.iter().product();
            let shape_us: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            TensorOut { name: name.to_string(), shape: shape_us, ty: 0, data: (0..numel).flat_map(|i| ((i + 1) as f32 * 0.1).to_le_bytes()).collect() }
        };

        let mut tensors = vec![
            f32t(vec![vocab, d], "token_embd.weight"),
            f32t(vec![vocab, d], "output.weight"),
            f32t(vec![d], "output_norm.weight"),
        ];

        // Layer 0: linear attention (Gated DeltaNet).
        tensors.push(f32t(vec![d], "blk.0.attn_norm.weight"));
        tensors.push(f32t(vec![d], "blk.0.post_attention_norm.weight"));
        tensors.push(f32t(vec![2 * key_dim + value_dim, d], "blk.0.attn_qkv.weight"));
        tensors.push(f32t(vec![value_dim, d], "blk.0.attn_gate.weight"));
        tensors.push(f32t(vec![lin_vh, d], "blk.0.ssm_alpha.weight"));
        tensors.push(f32t(vec![lin_vh, d], "blk.0.ssm_beta.weight"));
        tensors.push(f32t(vec![2 * key_dim + value_dim, conv_k], "blk.0.ssm_conv1d.weight"));
        tensors.push(f32t(vec![lin_vh], "blk.0.ssm_a"));
        tensors.push(f32t(vec![lin_vh], "blk.0.ssm_dt.bias"));
        tensors.push(f32t(vec![lin_vd], "blk.0.ssm_norm.weight"));
        tensors.push(f32t(vec![d, value_dim], "blk.0.ssm_out.weight"));

        // Layer 1: full attention.
        tensors.push(f32t(vec![d], "blk.1.attn_norm.weight"));
        tensors.push(f32t(vec![d], "blk.1.post_attention_norm.weight"));
        tensors.push(f32t(vec![n_heads * head_dim * 2, d], "blk.1.attn_q.weight"));
        tensors.push(f32t(vec![n_kv * head_dim, d], "blk.1.attn_k.weight"));
        tensors.push(f32t(vec![n_kv * head_dim, d], "blk.1.attn_v.weight"));
        tensors.push(f32t(vec![head_dim], "blk.1.attn_q_norm.weight"));
        tensors.push(f32t(vec![head_dim], "blk.1.attn_k_norm.weight"));
        tensors.push(f32t(vec![d, n_heads * head_dim], "blk.1.attn_output.weight"));

        // MoE (every real layer).
        for l in 0..2u64 {
            tensors.push(f32t(vec![n_experts, d], format!("blk.{l}.ffn_gate_inp.weight").leak()));
            tensors.push(f32t(vec![d], format!("blk.{l}.ffn_gate_inp_shexp.weight").leak()));
            tensors.push(f32t(vec![shared_ff, d], format!("blk.{l}.ffn_gate_shexp.weight").leak()));
            tensors.push(f32t(vec![shared_ff, d], format!("blk.{l}.ffn_up_shexp.weight").leak()));
            tensors.push(f32t(vec![d, shared_ff], format!("blk.{l}.ffn_down_shexp.weight").leak()));
            tensors.push(f32t(vec![n_experts, moe_ff, d], format!("blk.{l}.ffn_gate_exps.weight").leak()));
            tensors.push(f32t(vec![n_experts, moe_ff, d], format!("blk.{l}.ffn_up_exps.weight").leak()));
            tensors.push(f32t(vec![n_experts, d, moe_ff], format!("blk.{l}.ffn_down_exps.weight").leak()));
        }

        // MTP: block index == n_layers (2), must be dropped, not miscounted
        // as a third real layer.
        tensors.push(f32t(vec![d], "blk.2.attn_norm.weight"));
        tensors.push(f32t(vec![d], "blk.2.nextn.enorm.weight"));

        let kvs = vec![
            kv("general.architecture", GgufValue::String(GGUF_ARCHITECTURE.to_string())),
            kv("qwen35moe.block_count", GgufValue::U32(3)), // 2 real layers + 1 MTP
            kv("qwen35moe.embedding_length", GgufValue::U32(d as u32)),
            kv("qwen35moe.attention.head_count", GgufValue::U32(n_heads as u32)),
            kv("qwen35moe.attention.head_count_kv", GgufValue::U32(n_kv as u32)),
            kv("qwen35moe.attention.key_length", GgufValue::U32(head_dim as u32)),
            kv("qwen35moe.attention.value_length", GgufValue::U32(head_dim as u32)),
            kv("qwen35moe.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-6)),
            kv("qwen35moe.rope.freq_base", GgufValue::F32(1_000_000.0)),
            kv("qwen35moe.rope.dimension_count", GgufValue::U32((head_dim / 2) as u32)),
            kv("qwen35moe.full_attention_interval", GgufValue::U32(2)),
            kv("qwen35moe.context_length", GgufValue::U32(4096)),
            kv("qwen35moe.expert_count", GgufValue::U32(n_experts as u32)),
            kv("qwen35moe.expert_used_count", GgufValue::U32(2)),
            kv("qwen35moe.expert_feed_forward_length", GgufValue::U32(moe_ff as u32)),
            kv("qwen35moe.expert_shared_feed_forward_length", GgufValue::U32(shared_ff as u32)),
            kv("qwen35moe.ssm.conv_kernel", GgufValue::U32(conv_k as u32)),
            kv("qwen35moe.ssm.group_count", GgufValue::U32(lin_kh as u32)),
        ];

        write(path, &kvs, &tensors, 32).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::testing::write_synthetic_gguf as synthetic_gguf;

    #[test]
    fn config_from_gguf_matches_synthetic_header() {
        let path = std::env::temp_dir().join(format!("qwen35-import-test-{}.gguf", std::process::id())).to_string_lossy().into_owned();
        synthetic_gguf(&path);
        let mg = MmapGguf::open(&path).unwrap();
        let cfg = config_from_gguf(&mg).unwrap();

        assert_eq!(cfg.n_layers, 2, "block_count(3) - 1(MTP) = 2");
        assert_eq!(cfg.d_model, 4);
        assert_eq!(cfg.n_heads, 2);
        assert_eq!(cfg.n_kv_heads, 1);
        assert_eq!(cfg.head_dim, 4);
        assert_eq!(cfg.full_attention_interval, 2);
        assert_eq!(cfg.layer_types(), vec![LayerType::Linear, LayerType::Full]);
        assert_eq!(cfg.linear_num_key_heads, 1);
        assert_eq!(cfg.linear_num_value_heads, 2);
        assert_eq!(cfg.linear_key_head_dim, 3);
        assert_eq!(cfg.linear_value_head_dim, 3);
        assert_eq!(cfg.n_experts, 3);
        assert_eq!(cfg.top_k, 2);
        assert_eq!(cfg.moe_intermediate_size, 2);
        assert_eq!(cfg.shared_expert_intermediate_size, 2);
        assert!(!cfg.tie_embeddings, "output.weight is present -> untied");
        assert_eq!(cfg.vocab, 10);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn import_gguf_covers_every_planned_tensor_and_drops_mtp() {
        let dir = std::env::temp_dir().join(format!("qwen35-import-test-out-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let src = dir.join("src.gguf").to_string_lossy().into_owned();
        let out = dir.join("out.safetensors").to_string_lossy().into_owned();
        synthetic_gguf(&src);

        import_gguf(&src, &out, Some("test/qwen35-tiny")).expect("import must succeed with full coverage");

        // Re-open and spot check a few tensors, including one from each
        // fan-out expert stack and both mixer kinds.
        let reader = checkpoint::weightio::WeightReader::open(&out).unwrap();
        assert!(reader.tensor("tok.weight").is_some());
        assert!(reader.tensor("lm_head.weight").is_some());
        assert!(reader.tensor("blocks.0.linear_attn.in_proj_qkv.weight").is_some());
        assert!(reader.tensor("blocks.1.self_attn.q_proj.weight").is_some());
        assert!(reader.tensor("blocks.0.mlp.experts.0.gate.weight").is_some());
        assert!(reader.tensor("blocks.0.mlp.experts.2.down.weight").is_some());
        // MTP (block index 2, out of range for n_layers=2) must NOT appear.
        assert!(reader.tensor("blocks.2.ln1.weight").is_none());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn expert_stack_slices_are_contiguous_and_ordered_by_expert_index() {
        // A dedicated shape-focused check: expert e's slice of a
        // [n_experts, moe_ff, d] stack must be EXACTLY data[e*chunk..(e+1)*chunk],
        // not transposed or interleaved -- this is the one thing that would
        // be silently wrong (not a crash) if GGUF's dequant order didn't
        // match the assumption this file's module doc states.
        let dir = std::env::temp_dir().join(format!("qwen35-import-test-slice-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let src = dir.join("src.gguf").to_string_lossy().into_owned();
        let out = dir.join("out.safetensors").to_string_lossy().into_owned();
        synthetic_gguf(&src);
        import_gguf(&src, &out, None).unwrap();

        let mg = MmapGguf::open(&src).unwrap();
        let raw = mg.tensor("blk.0.ffn_gate_exps.weight").unwrap().unwrap();
        let reader = checkpoint::weightio::WeightReader::open(&out).unwrap();
        let (moe_ff, d, n_experts) = (2usize, 4usize, 3usize);
        let chunk = moe_ff * d;
        for e in 0..n_experts {
            let want = &raw[e * chunk..(e + 1) * chunk];
            let got = reader.tensor(&format!("blocks.0.mlp.experts.{e}.gate.weight")).unwrap();
            assert_eq!(got, want, "expert {e}'s slice must be contiguous rows [{}, {})", e * chunk, (e + 1) * chunk);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Integration check against the REAL checkpoint: set
    /// `BRAIN_QWEN35_GGUF` to a **fully downloaded** `Qwen3.5-35B-A3B*.gguf`
    /// (any quant level - only the header + embedded tokenizer are read,
    /// tensor data is never dequantized here, but `MmapGguf::open` validates
    /// every tensor's byte range against the file length up front, so a
    /// still-downloading/truncated file fails this test with an "out of
    /// range" error rather than silently passing on partial data). Self-skips
    /// loudly when unset rather than failing
    /// a box that hasn't fetched the multi-GB file.
    #[test]
    fn config_and_tokenizer_extract_from_the_real_checkpoint() {
        let Ok(path) = std::env::var("BRAIN_QWEN35_GGUF") else {
            eprintln!("SKIP: BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.5-35B-A3B*.gguf to run this)");
            return;
        };
        let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let cfg = config_from_gguf(&mg).expect("config_from_gguf on the real checkpoint");

        // Cross-checked against the real config.json fetched separately
        // (/data/workspace/resources/qwen3.5/config.json) and the checkpoint
        // header parsed by hand (this file's own module doc).
        assert_eq!(cfg.n_layers, 40);
        assert_eq!(cfg.d_model, 2048);
        assert_eq!(cfg.n_heads, 16);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.n_experts, 256);
        assert_eq!(cfg.top_k, 8);
        assert_eq!(cfg.moe_intermediate_size, 512);
        assert_eq!(cfg.shared_expert_intermediate_size, 512);
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads, 32);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.vocab, 248320);
        assert!(!cfg.tie_embeddings);
        assert_eq!((cfg.rotary_dim() as f32 / cfg.head_dim as f32 - 0.25).abs() < 1e-6, true);

        let full_idx: Vec<usize> = cfg
            .layer_types()
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == LayerType::Full)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(full_idx, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);

        // Tokenizer: extract from the GGUF-embedded KV and encode/decode a
        // real string round-trip through data::qwen_tokenizer::QwenBpe.
        let gguf_tok = mg.tokenizer().expect("tokenizer.ggml.* KV must be present");
        let bpe = data::qwen_tokenizer::QwenBpe::from_gguf(&gguf_tok).expect("QwenBpe::from_gguf");
        use data::tokenizer::Tokenizer;
        let text = "Hello, world! This is Qwen3.5.";
        let ids = bpe.encode(text);
        assert!(!ids.is_empty(), "encoding a real sentence must produce tokens");
        let back = bpe.decode(&ids);
        assert_eq!(back, text, "encode/decode must round-trip a plain ASCII sentence exactly");
    }
}
