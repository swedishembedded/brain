// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Build a `ParamStore`-ready tensor source from a real
//! `microsoft/Florence-2-base`-layout HF checkpoint (`model.safetensors`).
//!
//! Unlike most models this repo ports, brain's own parameter names here are
//! the checkpoint's OWN tensor names, unchanged - every `vision::*`/`text::*`
//! module was written to read `ps.w("language_model.model.encoder.layers.0.
//! self_attn.q_proj.weight")` etc. directly (see each module's own tests),
//! since Florence-2's real names already carry all the structure brain needs
//! and a rename table would add a large, purely-cosmetic indirection layer
//! for no benefit. So "import" here is NOT a name-remapping pass like
//! `lfm2::import`'s - it is the few real host-side TRANSFORMS every module's
//! own doc already documents as the caller's obligation, done ONCE here
//! instead of duplicated per call site:
//!
//! - `vision::channel_attn::ChannelAttn::new`'s doc: fold each DaViT stage's
//!   `1/sqrt(N)` channel-attention scale into that stage's `channel_attn.fn.
//!   qkv.weight`/`.bias` Q-rows (N is a compile-time-known per-stage
//!   constant, not a runtime scale kernel).
//! - `vision::project`'s module doc: synthesize the static `[24,24,1024]`
//!   position-embedding table (`build_pos_embed_table`) and the transposed
//!   `image_projection` (`transpose_2d`), under new synthetic names this
//!   module owns (`vision_projector.*` - see [`VISION_PROJECTOR_PREFIX`]).
//!
//! Every other tensor passes through byte-for-byte under its checkpoint name.

use std::collections::HashMap;
use std::path::Path;

use crate::text::BartConfig;
use crate::vision::project::{build_pos_embed_table, transpose_2d};
use crate::vision::DavitConfig;

/// The prefix [`vision::project::ImageProject`] reads its two synthesized
/// tensors and the (pass-through) temporal-embed/proj-norm tensors under -
/// matches every call site in `crates/florence2/tests/davit_image_project_
/// parity.rs`.
pub const VISION_PROJECTOR_PREFIX: &str = "vision_projector";

pub type TensorSource = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Read `<hf_dir>/model.safetensors` and build the full tensor source
/// [`crate::vision::Davit`], [`crate::vision::ImageProject`] and
/// [`crate::text::Florence2Lm`] read their weights from - real checkpoint
/// tensors pass through unchanged (channel-attention ones pre-scaled in
/// place), plus the two synthesized `vision_projector.*` tensors.
pub fn build_param_source(hf_dir: &Path) -> Result<TensorSource, String> {
    let path = hf_dir.join("model.safetensors");
    let tensors = checkpoint::safetensors::read(path.to_str().ok_or("hf_dir is not valid UTF-8")?)
        .map_err(|e| format!("florence2 import: read {}: {e}", path.display()))?;

    let mut source: TensorSource = HashMap::new();
    for t in tensors {
        source.insert(t.name, (t.shape, t.data));
    }

    prescale_channel_attn_qkv(&mut source, &DavitConfig::florence2_base())?;
    synthesize_vision_projector(&mut source)?;

    Ok(source)
}

/// Fold every DaViT stage's `1/sqrt(N)` channel-attention scale into that
/// stage's `channel_attn.fn.qkv.{weight,bias}` Q-rows, in place - see this
/// module's own doc for why this happens here rather than in
/// `ChannelAttn::forward` itself.
fn prescale_channel_attn_qkv(source: &mut TensorSource, cfg: &DavitConfig) -> Result<(), String> {
    for (i, stage) in cfg.stages.iter().enumerate() {
        let n = stage.out_hw.0 * stage.out_hw.1;
        let scale = 1.0 / (n as f32).sqrt();
        for pair in 0..stage.depth {
            let prefix = format!("vision_tower.blocks.{i}.{pair}.channel_block.channel_attn.fn");
            let w_name = format!("{prefix}.qkv.weight");
            let (shape, data) = source.get_mut(&w_name).ok_or_else(|| format!("florence2 import: missing {w_name}"))?;
            let in_dim = shape[1];
            for v in data[..stage.dim_out as usize * in_dim].iter_mut() {
                *v *= scale;
            }
            let b_name = format!("{prefix}.qkv.bias");
            let (_, bdata) = source.get_mut(&b_name).ok_or_else(|| format!("florence2 import: missing {b_name}"))?;
            for v in bdata[..stage.dim_out as usize].iter_mut() {
                *v *= scale;
            }
        }
    }
    Ok(())
}

/// Synthesize [`VISION_PROJECTOR_PREFIX`]'s two derived tensors from the
/// checkpoint's real `image_pos_embed.*`/`image_projection` tensors, and
/// copy the pass-through ones (`visual_temporal_embed.pos_idx_to_embed`,
/// `image_proj_norm.{weight,bias}`) under the same prefix - see
/// `vision::project`'s module doc for why each transform is needed.
fn synthesize_vision_projector(source: &mut TensorSource) -> Result<(), String> {
    let (davit_dim, d_model, grid) = (1024u32, 768u32, 24u32);
    let get = |source: &TensorSource, name: &str| -> Result<(Vec<usize>, Vec<f32>), String> {
        source.get(name).cloned().ok_or_else(|| format!("florence2 import: missing {name}"))
    };

    let (_, column) = get(source, "image_pos_embed.column_embeddings.weight")?;
    let (_, row) = get(source, "image_pos_embed.row_embeddings.weight")?;
    let pos_table = build_pos_embed_table(&column, &row, grid, grid, davit_dim / 2);
    source.insert(format!("{VISION_PROJECTOR_PREFIX}.pos_embed_table"), (vec![(grid * grid) as usize, davit_dim as usize], pos_table));

    let (proj_shape, proj_data) = get(source, "image_projection")?;
    if proj_shape != vec![davit_dim as usize, d_model as usize] {
        return Err(format!("florence2 import: image_projection shape {proj_shape:?}, expected [{davit_dim},{d_model}]"));
    }
    let proj_t = transpose_2d(&proj_data, davit_dim as usize, d_model as usize);
    source.insert(format!("{VISION_PROJECTOR_PREFIX}.image_projection_t"), (vec![d_model as usize, davit_dim as usize], proj_t));

    for name in ["visual_temporal_embed.pos_idx_to_embed", "image_proj_norm.weight", "image_proj_norm.bias"] {
        let (shape, data) = get(source, name)?;
        source.insert(format!("{VISION_PROJECTOR_PREFIX}.{name}"), (shape, data));
    }
    Ok(())
}

/// `param_list`-style role table for `ParamStore::new_with_roles_src`: every
/// tensor name [`crate::vision::Davit`]/[`crate::vision::ImageProject`]/
/// [`crate::text::Florence2Lm`] read, generated from the two config
/// schedules rather than hand-listed (matches the generator pattern
/// `davit_full_forward_parity.rs`/`text_lm_parity.rs` already use, now
/// shared with real serving instead of duplicated per test).
pub fn all_tensor_names(davit_cfg: &DavitConfig, bart_cfg: &BartConfig) -> Vec<String> {
    let mut names = Vec::new();

    for (i, stage) in davit_cfg.stages.iter().enumerate() {
        let p = format!("vision_tower.convs.{i}");
        for suffix in ["proj.weight", "proj.bias", "norm.weight", "norm.bias"] {
            names.push(format!("{p}.{suffix}"));
        }
        for pair in 0..stage.depth {
            for block in ["spatial_block", "channel_block"] {
                let bp = format!("vision_tower.blocks.{i}.{pair}.{block}");
                let attn = if block == "spatial_block" { "window_attn" } else { "channel_attn" };
                for suffix in [
                    "conv1.fn.dw.weight",
                    "conv1.fn.dw.bias",
                    "conv2.fn.dw.weight",
                    "conv2.fn.dw.bias",
                    "ffn.norm.weight",
                    "ffn.norm.bias",
                    "ffn.fn.net.fc1.weight",
                    "ffn.fn.net.fc1.bias",
                    "ffn.fn.net.fc2.weight",
                    "ffn.fn.net.fc2.bias",
                ] {
                    names.push(format!("{bp}.{suffix}"));
                }
                for suffix in ["norm.weight", "norm.bias", "fn.qkv.weight", "fn.qkv.bias", "fn.proj.weight", "fn.proj.bias"] {
                    names.push(format!("{bp}.{attn}.{suffix}"));
                }
            }
        }
    }

    for suffix in ["pos_embed_table", "image_projection_t", "visual_temporal_embed.pos_idx_to_embed", "image_proj_norm.weight", "image_proj_norm.bias"] {
        names.push(format!("{VISION_PROJECTOR_PREFIX}.{suffix}"));
    }

    names.push("language_model.model.shared.weight".to_string());
    names.push("language_model.final_logits_bias".to_string());
    for side in ["encoder", "decoder"] {
        let base = format!("language_model.model.{side}");
        names.push(format!("{base}.embed_positions.weight"));
        names.push(format!("{base}.layernorm_embedding.weight"));
        names.push(format!("{base}.layernorm_embedding.bias"));
        let layers = if side == "encoder" { bart_cfg.encoder_layers } else { bart_cfg.decoder_layers };
        for i in 0..layers {
            let lp = format!("{base}.layers.{i}");
            for attn in if side == "encoder" { vec!["self_attn"] } else { vec!["self_attn", "encoder_attn"] } {
                for p in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                    names.push(format!("{lp}.{attn}.{p}.weight"));
                    names.push(format!("{lp}.{attn}.{p}.bias"));
                }
                names.push(format!("{lp}.{attn}_layer_norm.weight"));
                names.push(format!("{lp}.{attn}_layer_norm.bias"));
            }
            for fc in ["fc1", "fc2"] {
                names.push(format!("{lp}.{fc}.weight"));
                names.push(format!("{lp}.{fc}.bias"));
            }
            names.push(format!("{lp}.final_layer_norm.weight"));
            names.push(format!("{lp}.final_layer_norm.bias"));
        }
    }
    names
}
