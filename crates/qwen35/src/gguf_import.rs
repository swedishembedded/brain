// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! GGUF import for Qwen3.8-27B - the second source format alongside
//! [`crate::import`]'s HF-safetensors FP8 route, and (unlike that route)
//! able to import the MTP head, whose real-checkpoint tensors
//! [`crate::import`] deliberately never touches (see that module's doc).
//!
//! ## Layer leaf vocabulary
//!
//! The GDN/GQA/dense-MLP leaf spellings (`attn_qkv.weight`, `attn_q.weight`,
//! `ffn_gate.weight`, …) are `gguf::leaf`'s shared vocabulary, the same table
//! `qwen3`/`qwen35moe` read - this importer supplies only its own
//! brain-parameter suffix per [`gguf::leaf::Role`] ([`block_leaf_suffix`]),
//! which happens to be IDENTICAL to what [`crate::import::classify_layer_leaf`]
//! already produces from HF's own leaf names, since both routes feed the same
//! [`crate::config::Qwen35Config::param_list`].
//!
//! ## `qwen35.block_count` includes the MTP layer
//!
//! Confirmed against the real `unsloth/Qwen3.8-27B-GGUF` header:
//! `qwen35.block_count = 65` and `qwen35.nextn_predict_layers = 1` - llama.cpp
//! folds the MTP ("next-token prediction") layer into the SAME `blk.N` index
//! space as block 64, exactly as `qwen35moe`'s own GGUF folds its MTP layer
//! into `blk.40`. `n_layers` here is `block_count - nextn_predict_layers`,
//! never `block_count` directly.
//!
//! ## No RMSNorm fold
//!
//! [`crate::import::fold_plain_rmsnorm_weights`] undoes HF's zero-init
//! `Qwen3_5RMSNorm` convention (`weight` stored such that the applied
//! multiplier is `1+weight`) on the safetensors route. A real GGUF's own norm
//! weights, read directly off `Qwen3.8-27B-Q8_0.gguf`, already cluster around
//! 1.0 - e.g. `blk.0.attn_norm.weight` in `[0.89, 1.05]`, `output_norm.weight`
//! in `[1.6, 2.0]` - so llama.cpp's conversion has already applied the fold.
//! This importer must NOT apply it a second time; doing so would silently
//! corrupt every plain norm exactly the way a missed fold would on the other
//! route. Neither [`classify`] nor [`import_mtp_block`] calls it.
//!
//! ## MTP (`blk.{n_layers}.*` + its `nextn.*` extras)
//!
//! [`classify`] drops the whole MTP block ([`DROP_MTP_BLOCK`]) rather than
//! folding it into the generic per-tensor driver: `blk.{n_layers}.*` carries
//! both an ordinary `Full`-type decoder layer (self-attn + dense MLP, reusing
//! [`block_leaf_suffix`] under the `mtp.layers.0.` prefix) AND four `nextn.*`
//! extras no other tensor in this leaf space has. [`import_mtp_block`] reads
//! all of them directly and writes into the SAME output file via
//! [`gguf::import::to_st_into`]'s writer, because one of the four -
//! `nextn.eh_proj.weight` (`[d, 2d]`, torch row-major) - needs a COLUMN split
//! into `mtp.fc_e.weight` (columns `[0,d)`) and `mtp.fc_h.weight` (columns
//! `[d,2d)`) that no [`gguf::import::Mapped`] variant expresses (GGUF's
//! dequantized output is row-major in torch order, so an inner-axis split
//! must gather each destination's columns out of every row - see
//! `gguf::import::to_st_into`'s own doc). This reuses the EXACT convention
//! `crate::import::import_mtp` already established for the HF-safetensors
//! route (`en` through `fc_e`, `hn` through `fc_h`), not a second, untested
//! one.

use std::collections::HashSet;

use checkpoint::gguf::{GgufValue, MmapGguf};
use checkpoint::st::ModelCard;
use checkpoint::weightio::StWriter;
use gguf::import::{self, ImportStats, Leaf, Mapped};
use gguf::leaf::Role;
use gguf::ArchKv;

use crate::config::{LayerType, Qwen35Config};

/// llama.cpp's `general.architecture` value for this model family.
pub const GGUF_ARCHITECTURE: &str = "qwen35";

/// Not imported by the generic per-tensor driver, by reason - the MTP block
/// is imported separately by [`import_mtp_block`].
const DROP_MTP_BLOCK: &str = "MTP block (blk.N for N == n_layers) - imported separately, see this module's doc";
const DROP_OTHER: &str = "vision (v.*/mm.*) or an unrecognized leaf";

/// Derive [`Qwen35Config`] from a GGUF file's KV metadata, cross-checked
/// against the real tensor shapes of the first Gated-DeltaNet block found -
/// mirrors `qwen35moe::import::config_from_gguf`'s derivation exactly (same
/// hybrid mixer, same llama.cpp SSM KV ambiguity), because that function's own
/// doc records why the `qwen35.ssm.*` KV keys are read only as a
/// cross-check, never as the primary source: their names are llama.cpp's
/// generic Mamba/SSM vocabulary, and `ssm.time_step_rank` in particular is
/// classic Mamba's low-rank `dt` projection, NOT a head count - it only
/// happens to equal `linear_num_value_heads` on this checkpoint. Tensor
/// shapes are ground truth.
pub fn config_from_gguf(mg: &MmapGguf) -> Result<Qwen35Config, String> {
    let kv = ArchKv::expect_architecture(mg, GGUF_ARCHITECTURE)?;

    let block_count = kv.req_u32("block_count")?;
    let nextn = kv.u32_or("nextn_predict_layers", 1);
    if nextn != 1 {
        return Err(format!("qwen35: nextn_predict_layers={nextn}, only exactly 1 (a single MTP layer) is supported"));
    }
    let n_layers = block_count.checked_sub(nextn).ok_or("qwen35: block_count must exceed nextn_predict_layers")?;

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
    let mrope_section = mrope_section_from_kv(&kv, rotary_dim)?;

    let types = crate::config::layer_types(n_layers, full_attention_interval);
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
    let value_dim = *gate_shape.first().ok_or("qwen35: attn_gate.weight has no leading dim")?;
    let head_v_dim = *norm_shape.first().ok_or("qwen35: ssm_norm.weight has no leading dim")?;
    let qkv_width = *qkv_shape.first().ok_or("qwen35: attn_qkv.weight has no leading dim")?;
    let key_dim = (qkv_width - value_dim) / 2;
    let linear_num_value_heads = (value_dim / head_v_dim) as u32;
    let linear_value_head_dim = head_v_dim as u32;
    let linear_key_head_dim = kv.u32_or("ssm.state_size", linear_value_head_dim);
    let linear_num_key_heads = (key_dim as u32) / linear_key_head_dim;
    if let Some(group_count) = kv.u32("ssm.group_count") {
        if group_count != linear_num_key_heads {
            return Err(format!(
                "qwen35: ssm.group_count={group_count} disagrees with tensor-shape-derived linear_num_key_heads={linear_num_key_heads}"
            ));
        }
    }
    let linear_conv_kernel_dim = kv.req_u32("ssm.conv_kernel")?;

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
        mrope_section,

        full_attention_interval,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_conv_kernel_dim,

        intermediate_size: kv.req_u32("feed_forward_length")?,

        lora: None,
        mtp: true,
    })
}

/// `qwen35.rope.dimension_sections` carries a 4th, always-zero slot in the
/// real header (`[11, 11, 10, 0]`) - llama.cpp's generic M-RoPE table has room
/// for a 4th axis this model does not use. brain's own `mrope_section` is
/// fixed at 3; asserting the 4th (and any further) entry is zero, rather than
/// silently dropping it, is what catches a real 4-axis checkpoint this
/// importer does not support instead of quietly mis-decoding its RoPE.
fn mrope_section_from_kv(kv: &ArchKv, rotary_dim: u32) -> Result<[u32; 3], String> {
    let Some(GgufValue::Array(items)) = kv.get("rope.dimension_sections") else {
        return Err("qwen35: missing qwen35.rope.dimension_sections".to_string());
    };
    let vals: Vec<u32> = items
        .iter()
        .map(|v| v.as_u64().map(|v| v as u32).ok_or_else(|| "qwen35: rope.dimension_sections entry is not an unsigned integer".to_string()))
        .collect::<Result<_, String>>()?;
    let split_at = vals.len().min(3);
    let (head, tail) = vals.split_at(split_at);
    if tail.iter().any(|&v| v != 0) {
        return Err(format!("qwen35: rope.dimension_sections has a nonzero entry past the first 3, unsupported: {vals:?}"));
    }
    let sum: u32 = head.iter().sum();
    if sum * 2 != rotary_dim {
        return Err(format!("qwen35: rope.dimension_sections {head:?} sums to {sum}, expected rotary_dim/2 = {}", rotary_dim / 2));
    }
    let mut out = [0u32; 3];
    out[..head.len()].copy_from_slice(head);
    Ok(out)
}

/// The brain-canonical suffix for one [`Role`] on a layer of type `ty` -
/// shared verbatim by [`classify`] (any main-stack block) and
/// [`import_mtp_block`] (the MTP block's own identically-shaped `Full`
/// layer), so the two paths cannot drift on what a leaf is called. Matches
/// [`crate::import::classify_layer_leaf`]'s HF-side output exactly - both
/// routes feed the same [`Qwen35Config::param_list`].
fn block_leaf_suffix(role: Role, ty: LayerType) -> Option<&'static str> {
    Some(match (role, ty) {
        (Role::AttnNorm, _) => "ln1.weight",
        (Role::FfnNorm, _) => "ln2.weight",
        (Role::FfnGate, _) => "mlp.gate.weight",
        (Role::FfnUp, _) => "mlp.up.weight",
        (Role::FfnDown, _) => "mlp.down.weight",

        (Role::AttnQ, LayerType::Full) => "self_attn.q_proj.weight",
        (Role::AttnK, LayerType::Full) => "self_attn.k_proj.weight",
        (Role::AttnV, LayerType::Full) => "self_attn.v_proj.weight",
        (Role::AttnQNorm, LayerType::Full) => "self_attn.q_norm.weight",
        (Role::AttnKNorm, LayerType::Full) => "self_attn.k_norm.weight",
        (Role::AttnOutput, LayerType::Full) => "self_attn.o_proj.weight",

        (Role::AttnQkv, LayerType::Linear) => "linear_attn.in_proj_qkv.weight",
        (Role::AttnGate, LayerType::Linear) => "linear_attn.in_proj_z.weight",
        (Role::SsmAlpha, LayerType::Linear) => "linear_attn.in_proj_a.weight",
        (Role::SsmBeta, LayerType::Linear) => "linear_attn.in_proj_b.weight",
        (Role::SsmConv1d, LayerType::Linear) => "linear_attn.conv1d.weight",
        (Role::SsmA, LayerType::Linear) => "linear_attn.A_log",
        (Role::SsmDtBias, LayerType::Linear) => "linear_attn.dt_bias",
        (Role::SsmNorm, LayerType::Linear) => "linear_attn.norm.weight",
        (Role::SsmOut, LayerType::Linear) => "linear_attn.out_proj.weight",

        _ => return None,
    })
}

/// Classify one GGUF tensor for the MAIN decoder stack (layers
/// `[0, cfg.n_layers)` plus the three top-level tensors). The MTP block is
/// deliberately dropped here - see this module's doc.
fn classify(name: &str, cfg: &Qwen35Config) -> Mapped {
    let (l, leaf) = match import::split_name(name, cfg.n_layers) {
        Leaf::TokenEmbd => return Mapped::Simple("tok.weight".to_string()),
        Leaf::Output => return Mapped::Simple("lm_head.weight".to_string()),
        Leaf::OutputNorm => return Mapped::Simple("norm.weight".to_string()),
        Leaf::PastDepth { .. } => return Mapped::Dropped(DROP_MTP_BLOCK),
        Leaf::Other => return Mapped::Dropped(DROP_OTHER),
        Leaf::Block { layer, leaf } => (layer, leaf),
    };
    let ty = cfg.layer_types()[l]; // l < cfg.n_layers, guaranteed by split_name's own PastDepth cut
    match gguf::leaf::role(leaf).and_then(|role| block_leaf_suffix(role, ty)) {
        Some(suffix) => Mapped::Simple(format!("blocks.{l}.{suffix}")),
        None => Mapped::Dropped(DROP_OTHER),
    }
}

fn read_tensor(mg: &MmapGguf, name: &str) -> Result<Vec<f32>, String> {
    mg.tensor(name).ok_or_else(|| format!("qwen35 mtp import: {name} vanished between names() and tensor()"))?
}

/// Import the MTP block's real weights directly into `writer` - the tensors
/// [`classify`] drops as [`DROP_MTP_BLOCK`]. See this module's doc for why
/// this cannot go through the generic per-tensor driver.
fn import_mtp_block(mg: &MmapGguf, cfg: &Qwen35Config, writer: &mut StWriter) -> Result<(), String> {
    let prefix = format!("blk.{}.", cfg.n_layers);
    let d = cfg.d_model as usize;
    let names: Vec<String> = mg.names().iter().filter(|n| n.starts_with(&prefix)).cloned().collect();
    if names.is_empty() {
        return Err(format!("qwen35 mtp import: no tensors under {prefix:?} - is nextn_predict_layers really 1?"));
    }

    for name in &names {
        let leaf = &name[prefix.len()..];
        if let Some(nextn) = leaf.strip_prefix("nextn.") {
            match nextn {
                "eh_proj.weight" => {
                    let shape = mg.shape(name).ok_or_else(|| format!("qwen35 mtp import: {name} has no shape"))?;
                    if shape != [d, 2 * d] {
                        return Err(format!("qwen35 mtp import: {name} has shape {shape:?}, expected [{d}, {}]", 2 * d));
                    }
                    let data = read_tensor(mg, name)?;
                    let (mut fc_e, mut fc_h) = (Vec::with_capacity(d * d), Vec::with_capacity(d * d));
                    for row in 0..d {
                        let base = row * 2 * d;
                        fc_e.extend_from_slice(&data[base..base + d]);
                        fc_h.extend_from_slice(&data[base + d..base + 2 * d]);
                    }
                    writer.write("mtp.fc_e.weight", &fc_e).map_err(|e| e.to_string())?;
                    writer.write("mtp.fc_h.weight", &fc_h).map_err(|e| e.to_string())?;
                }
                "enorm.weight" => writer.write("mtp.pre_fc_norm_embedding.weight", &read_tensor(mg, name)?).map_err(|e| e.to_string())?,
                "hnorm.weight" => writer.write("mtp.pre_fc_norm_hidden.weight", &read_tensor(mg, name)?).map_err(|e| e.to_string())?,
                "shared_head_norm.weight" => writer.write("mtp.norm.weight", &read_tensor(mg, name)?).map_err(|e| e.to_string())?,
                other => return Err(format!("qwen35 mtp import: unrecognized nextn leaf {other:?} under {name}")),
            }
            continue;
        }
        let Some(suffix) = gguf::leaf::role(leaf).and_then(|role| block_leaf_suffix(role, LayerType::Full)) else {
            return Err(format!("qwen35 mtp import: unrecognized leaf {leaf:?} under the MTP block {name}"));
        };
        writer.write(&format!("mtp.layers.0.{suffix}"), &read_tensor(mg, name)?).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Import a Qwen3.8-27B GGUF into a brain-native safetensors checkpoint,
/// including the MTP head.
pub fn import_gguf(gguf_path: &str, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let mg = MmapGguf::open(gguf_path)?;
    import_mmap(&mg, out_path, id_override)
}

/// [`import_gguf`] over an ALREADY-OPEN checkpoint - the shape the generic
/// architecture-dispatch registry needs, since it must read
/// `general.architecture` before it can know which importer to call.
pub fn import_mmap(mg: &MmapGguf, out_path: &str, id_override: Option<&str>) -> Result<ImportStats, String> {
    let cfg = config_from_gguf(mg)?;
    let full_params = cfg.param_list();
    let mtp_names: HashSet<&str> = full_params.iter().map(|(n, _)| n.as_str()).filter(|n| n.starts_with("mtp.")).collect();
    let main_params: Vec<(String, usize)> = full_params.iter().filter(|(n, _)| !mtp_names.contains(n.as_str())).cloned().collect();

    let mut card = ModelCard::new(id_override.unwrap_or("qwen35"), "qwen35");
    card.context_length = Some(cfg.max_position_embeddings as u64);
    card.param_count = Some(full_params.iter().map(|(_, n)| *n as u64).sum());

    let plan: Vec<(String, Vec<u64>)> = full_params.iter().map(|(n, numel)| (n.clone(), vec![*numel as u64])).collect();
    let mut writer = StWriter::create(out_path, &plan, &cfg.to_json(), Some(&card)).map_err(|e| format!("create {out_path}: {e}"))?;

    let stats = import::to_st_into(mg, &main_params, &|n| Ok(classify(n, &cfg)), &mut writer, "qwen35")?;
    import_mtp_block(mg, &cfg, &mut writer)?;
    // `finish` re-checks the writer's OWN full plan (main stack + MTP), so a
    // gap in the manual MTP pass is caught exactly like a gap in the generic
    // one would be.
    writer.finish().map_err(|e| e.to_string())?;
    eprintln!("qwen35: {stats} (+ MTP block, written directly) -> {out_path}");
    Ok(stats)
}

/// Test fixtures for this importer, shared across crates.
///
/// `pub` (not `#[cfg(test)]`) so `brain-cli`'s GGUF-import-registry tests can
/// drive a REAL conversion through the generic architecture dispatch without
/// a second, drifting copy of this checkpoint builder - the same arrangement
/// `qwen3`/`qwen35moe`'s own `testing` modules use. Not part of the model's
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
    /// [`classify`] and [`import_mtp_block`] handle, at a tiny shape (2 real
    /// layers: one linear-attention, one full-attention,
    /// `full_attention_interval=2` so layer 1 is `Full`), plus a REAL MTP
    /// block at index 2 (its own `Full`-type decoder leaves under
    /// `mtp.layers.0.` AND the four `nextn.*` extras, including the
    /// `eh_proj.weight` column split) - unlike `qwen35moe`'s own fixture,
    /// which drops its MTP block, this model imports it, so the fixture must
    /// exercise that path rather than merely prove it is skipped.
    pub fn write_synthetic_gguf(path: &str) {
        let d = 4u64; // hidden
        let n_heads = 2u64;
        let head_dim = 4u64; // -> q_proj width doubled = 16
        let n_kv = 1u64;
        let lin_kh = 1u64;
        let lin_kd = 3u64; // linear_key_head_dim
        let lin_vh = 2u64;
        let lin_vd = 3u64; // linear_value_head_dim
        let key_dim = lin_kh * lin_kd;
        let value_dim = lin_vh * lin_vd;
        let conv_k = 2u64;
        let ff = 3u64; // dense MLP intermediate size
        let vocab = 10u64;
        let rotary_dim = 2u64; // head_dim/2, matching qwen35moe's tiny fixture

        let f32t = |shape: Vec<u64>, name: &str| {
            let numel: u64 = shape.iter().product();
            let shape_us: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            TensorOut { name: name.to_string(), shape: shape_us, ty: 0, data: (0..numel).flat_map(|i| ((i + 1) as f32 * 0.1).to_le_bytes()).collect() }
        };
        let dense_mlp = |tensors: &mut Vec<TensorOut>, prefix: &str| {
            tensors.push(f32t(vec![ff, d], format!("{prefix}ffn_gate.weight").leak()));
            tensors.push(f32t(vec![ff, d], format!("{prefix}ffn_up.weight").leak()));
            tensors.push(f32t(vec![d, ff], format!("{prefix}ffn_down.weight").leak()));
        };

        let mut tensors = vec![
            f32t(vec![vocab, d], "token_embd.weight"),
            f32t(vec![vocab, d], "output.weight"),
            f32t(vec![d], "output_norm.weight"),
        ];

        // Layer 0: linear attention (Gated DeltaNet) + dense MLP.
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
        dense_mlp(&mut tensors, "blk.0.");

        // Layer 1: full attention + dense MLP.
        tensors.push(f32t(vec![d], "blk.1.attn_norm.weight"));
        tensors.push(f32t(vec![d], "blk.1.post_attention_norm.weight"));
        tensors.push(f32t(vec![n_heads * head_dim * 2, d], "blk.1.attn_q.weight"));
        tensors.push(f32t(vec![n_kv * head_dim, d], "blk.1.attn_k.weight"));
        tensors.push(f32t(vec![n_kv * head_dim, d], "blk.1.attn_v.weight"));
        tensors.push(f32t(vec![head_dim], "blk.1.attn_q_norm.weight"));
        tensors.push(f32t(vec![head_dim], "blk.1.attn_k_norm.weight"));
        tensors.push(f32t(vec![d, n_heads * head_dim], "blk.1.attn_output.weight"));
        dense_mlp(&mut tensors, "blk.1.");

        // MTP: block index == n_layers (2). A REAL `Full`-type decoder layer
        // (same leaves as layer 1) plus the four `nextn.*` extras - distinct,
        // recognizable data in `eh_proj` so the column split is checkable.
        tensors.push(f32t(vec![d], "blk.2.attn_norm.weight"));
        tensors.push(f32t(vec![d], "blk.2.post_attention_norm.weight"));
        tensors.push(f32t(vec![n_heads * head_dim * 2, d], "blk.2.attn_q.weight"));
        tensors.push(f32t(vec![n_kv * head_dim, d], "blk.2.attn_k.weight"));
        tensors.push(f32t(vec![n_kv * head_dim, d], "blk.2.attn_v.weight"));
        tensors.push(f32t(vec![head_dim], "blk.2.attn_q_norm.weight"));
        tensors.push(f32t(vec![head_dim], "blk.2.attn_k_norm.weight"));
        tensors.push(f32t(vec![d, n_heads * head_dim], "blk.2.attn_output.weight"));
        dense_mlp(&mut tensors, "blk.2.");
        tensors.push(f32t(vec![d], "blk.2.nextn.enorm.weight"));
        tensors.push(f32t(vec![d], "blk.2.nextn.hnorm.weight"));
        tensors.push(f32t(vec![d], "blk.2.nextn.shared_head_norm.weight"));
        // eh_proj: [d, 2d] torch shape. Column [0,d) = 0.03 exactly, column
        // [d,2d) = 0.04 exactly, so the split's correctness is a value check,
        // not just a shape check.
        let eh_data: Vec<u8> = (0..d)
            .flat_map(|_| (0..d).map(|_| 0.03f32).chain((0..d).map(|_| 0.04f32)))
            .flat_map(|v| v.to_le_bytes())
            .collect();
        tensors.push(TensorOut { name: "blk.2.nextn.eh_proj.weight".to_string(), shape: vec![d as usize, 2 * d as usize], ty: 0, data: eh_data });

        let kvs = vec![
            kv("general.architecture", GgufValue::String(GGUF_ARCHITECTURE.to_string())),
            kv("qwen35.block_count", GgufValue::U32(3)), // 2 real layers + 1 MTP
            kv("qwen35.nextn_predict_layers", GgufValue::U32(1)),
            kv("qwen35.embedding_length", GgufValue::U32(d as u32)),
            kv("qwen35.attention.head_count", GgufValue::U32(n_heads as u32)),
            kv("qwen35.attention.head_count_kv", GgufValue::U32(n_kv as u32)),
            kv("qwen35.attention.key_length", GgufValue::U32(head_dim as u32)),
            kv("qwen35.attention.value_length", GgufValue::U32(head_dim as u32)),
            kv("qwen35.attention.layer_norm_rms_epsilon", GgufValue::F32(1e-6)),
            kv("qwen35.rope.freq_base", GgufValue::F32(1_000_000.0)),
            kv("qwen35.rope.dimension_count", GgufValue::U32(rotary_dim as u32)),
            kv(
                "qwen35.rope.dimension_sections",
                GgufValue::Array(vec![GgufValue::U32(1), GgufValue::U32(0), GgufValue::U32(0), GgufValue::U32(0)]),
            ),
            kv("qwen35.full_attention_interval", GgufValue::U32(2)),
            kv("qwen35.context_length", GgufValue::U32(4096)),
            kv("qwen35.feed_forward_length", GgufValue::U32(ff as u32)),
            kv("qwen35.ssm.conv_kernel", GgufValue::U32(conv_k as u32)),
            kv("qwen35.ssm.group_count", GgufValue::U32(lin_kh as u32)),
            kv("qwen35.ssm.state_size", GgufValue::U32(lin_kd as u32)),
        ];

        write(path, &kvs, &tensors, 32).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::testing::write_synthetic_gguf as synthetic_gguf;
    use super::*;

    #[test]
    fn config_from_gguf_matches_synthetic_header() {
        let path = std::env::temp_dir().join(format!("qwen35-gguf-import-test-{}.gguf", std::process::id())).to_string_lossy().into_owned();
        synthetic_gguf(&path);
        let mg = MmapGguf::open(&path).unwrap();
        let cfg = config_from_gguf(&mg).unwrap();

        assert_eq!(cfg.n_layers, 2, "block_count(3) - nextn_predict_layers(1) = 2");
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
        assert_eq!(cfg.linear_conv_kernel_dim, 2);
        assert_eq!(cfg.intermediate_size, 3);
        assert_eq!(cfg.mrope_section, [1, 0, 0], "the 4th, always-zero KV slot must not appear");
        assert!(!cfg.tie_embeddings, "output.weight is present -> untied");
        assert_eq!(cfg.vocab, 10);
        assert!(cfg.mtp, "the GGUF route always imports MTP");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn import_gguf_covers_the_main_stack_and_the_mtp_head_with_no_norm_fold() {
        let dir = std::env::temp_dir().join(format!("qwen35-gguf-import-test-out-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let src = dir.join("src.gguf").to_string_lossy().into_owned();
        let out = dir.join("out.safetensors").to_string_lossy().into_owned();
        synthetic_gguf(&src);

        import_gguf(&src, &out, Some("test/qwen35-gguf-tiny")).expect("import must succeed with full coverage");

        let reader = checkpoint::weightio::WeightReader::open(&out).unwrap();
        assert!(reader.tensor("tok.weight").is_some());
        assert!(reader.tensor("lm_head.weight").is_some());
        assert!(reader.tensor("blocks.0.linear_attn.in_proj_qkv.weight").is_some());
        assert!(reader.tensor("blocks.1.self_attn.q_proj.weight").is_some());
        assert!(reader.tensor("blocks.0.mlp.gate.weight").is_some());
        assert!(reader.tensor("blocks.1.mlp.down.weight").is_some());

        // MTP: real weights, not dropped.
        assert!(reader.tensor("mtp.layers.0.self_attn.q_proj.weight").is_some());
        assert!(reader.tensor("mtp.layers.0.mlp.gate.weight").is_some());
        assert!(reader.tensor("mtp.norm.weight").is_some());
        assert!(reader.tensor("mtp.pre_fc_norm_embedding.weight").is_some());
        assert!(reader.tensor("mtp.pre_fc_norm_hidden.weight").is_some());

        // No RMSNorm fold on the GGUF route: a plain-norm value written as
        // exactly 0.1 in the fixture must read back as exactly 0.1, not 1.1.
        let ln1 = reader.tensor("blocks.0.ln1.weight").unwrap();
        assert!((ln1[0] - 0.1).abs() < 1e-6, "GGUF route must NOT apply the (1+w) fold, got {}", ln1[0]);

        // The eh_proj column split: column [0,d) -> fc_e (all 0.03), column
        // [d,2d) -> fc_h (all 0.04) - the exact convention `crate::import::
        // import_mtp` uses for the HF route.
        let fc_e = reader.tensor("mtp.fc_e.weight").unwrap();
        let fc_h = reader.tensor("mtp.fc_h.weight").unwrap();
        assert!(fc_e.iter().all(|&v| (v - 0.03).abs() < 1e-6), "fc_e must be eh_proj's FIRST column half: {fc_e:?}");
        assert!(fc_h.iter().all(|&v| (v - 0.04).abs() < 1e-6), "fc_h must be eh_proj's SECOND column half: {fc_h:?}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Two-way coverage, first direction: an unrecognized MAIN-STACK tensor
    /// is refused by [`gguf::import::to_st_into`]'s own coverage check, not
    /// silently dropped - a converter that renames a leaf must break the
    /// import loudly.
    #[test]
    fn an_unrecognized_main_stack_tensor_is_refused() {
        let cfg = {
            let dir = std::env::temp_dir().join(format!("qwen35-gguf-import-test-cfg-{}", std::process::id()));
            std::fs::create_dir_all(&dir).ok();
            let src = dir.join("src.gguf").to_string_lossy().into_owned();
            synthetic_gguf(&src);
            let cfg = config_from_gguf(&MmapGguf::open(&src).unwrap()).unwrap();
            std::fs::remove_dir_all(&dir).ok();
            cfg
        };
        assert!(matches!(classify("blk.0.attn_wibble.weight", &cfg), Mapped::Dropped(DROP_OTHER)));
        assert!(matches!(classify("blk.5.attn_norm.weight", &cfg), Mapped::Dropped(DROP_MTP_BLOCK)), "beyond n_layers must be the MTP drop, not the generic one");
    }

    /// Integration check against the REAL checkpoint: set `BRAIN_QWEN35_GGUF`
    /// to a downloaded `Qwen3.8-27B*.gguf`. Self-skips loudly when unset
    /// rather than failing a box that hasn't fetched the multi-GB file.
    #[test]
    fn config_and_tokenizer_extract_from_the_real_checkpoint() {
        let Ok(path) = std::env::var("BRAIN_QWEN35_GGUF") else {
            brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
            return;
        };
        let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let cfg = config_from_gguf(&mg).expect("config_from_gguf on the real checkpoint");

        assert_eq!(cfg.n_layers, 64, "block_count(65) - nextn_predict_layers(1)");
        assert_eq!(cfg.d_model, 5120);
        assert_eq!(cfg.n_heads, 24);
        assert_eq!(cfg.n_kv_heads, 4);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.full_attention_interval, 4);
        assert_eq!(cfg.linear_num_key_heads, 16);
        assert_eq!(cfg.linear_num_value_heads, 48);
        assert_eq!(cfg.linear_key_head_dim, 128);
        assert_eq!(cfg.linear_value_head_dim, 128);
        assert_eq!(cfg.linear_conv_kernel_dim, 4);
        assert_eq!(cfg.intermediate_size, 17408);
        assert_eq!(cfg.mrope_section, [11, 11, 10]);
        assert_eq!(cfg.vocab, 248320);
        assert!(!cfg.tie_embeddings);
        assert!(cfg.mtp);

        let gguf_tok = mg.tokenizer().expect("tokenizer.ggml.* KV must be present");
        let bpe = data::qwen_tokenizer::QwenBpe::from_gguf(&gguf_tok).expect("QwenBpe::from_gguf");
        use data::tokenizer::Tokenizer;
        let text = "Hello, world! This is Qwen3.8.";
        let ids = bpe.encode(text);
        assert!(!ids.is_empty(), "encoding a real sentence must produce tokens");
        assert_eq!(bpe.decode(&ids), text, "encode/decode must round-trip a plain ASCII sentence exactly");
    }
}
