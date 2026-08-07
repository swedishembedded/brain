// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! HF → brain name remap for Qwen3-Omni-30B-A3B-Instruct, streamed via
//! `checkpoint::weightio` (never the whole 70.5 GB checkpoint in memory at
//! once — one tensor at a time, same as `qwen`/`glm`/`lfm`'s importers).
//!
//! **On-disk size**: `checkpoint::weightio::StWriter` writes f32 only, so a
//! full import of this checkpoint needs ~141 GB on disk (bf16 doubled) — see
//! `docs/models/omni/status.md` M3 for the measurement and why streaming
//! harder cannot fix it (the OUTPUT file itself needs one filesystem that
//! size). [`import_as`] is written and tested against real (partial) data,
//! but a full end-to-end run needs more disk than this development box has;
//! that is an environment limit, recorded, not silently worked around.
//!
//! Every mapping function here is pure (`&str -> Option<String>`), so it is
//! unit-tested against real tensor names from the released checkpoint's
//! `model.safetensors.index.json` without touching any tensor bytes — the
//! same shape `qwenvl::import::map_vision`/`map_decoder` and
//! `qwen_asr::import::map_audio_encoder` already use.
//!
//! Naming targets deliberately match those two crates' existing conventions
//! (`blocks.N.attn.wq/wk/wv/wo`, `blocks.N.qkv` fused, `multi_modal_projector.
//! linear_{1,2}`) rather than inventing new ones — M4/M5 hoist the shared
//! encoder implementations onto Omni's scale, and matching names now is what
//! makes that hoist "config bump", not "second copy".

/// One Qwen3-style decoder attention+norm block, HF leaf -> brain leaf.
/// Shared by the Thinker/Talker MoE decoders' non-expert tensors and by
/// Code2Wav's pre-transformer (dense, no MoE).
fn dense_attn_leaf(leaf: &str) -> Option<&'static str> {
    Some(match leaf {
        "input_layernorm.weight" => "ln1.weight",
        "post_attention_layernorm.weight" => "ln2.weight",
        "self_attn.q_proj.weight" => "attn.wq.weight",
        "self_attn.k_proj.weight" => "attn.wk.weight",
        "self_attn.v_proj.weight" => "attn.wv.weight",
        "self_attn.o_proj.weight" => "attn.wo.weight",
        "self_attn.q_norm.weight" => "attn.q_norm.weight",
        "self_attn.k_norm.weight" => "attn.k_norm.weight",
        _ => return None,
    })
}

// --------------------------------------------------------------------- audio
/// `thinker.audio_tower.*` -> `audio.*`. Fuses `self_attn.{q,k,v}_proj` into
/// one `blocks.{b}.qkv` (matching `qwen_asr::import::map_audio_encoder`,
/// which the same shape reuses at a smaller scale) — brain's audio-encoder
/// block builder (`model::vit`) expects the fused layout.
pub fn map_audio(hf: &str) -> Option<String> {
    let s = hf.strip_prefix("thinker.audio_tower.")?;
    if let Some(rest) = s.strip_prefix("layers.") {
        let (n, leaf) = rest.split_once('.')?;
        let mapped = match leaf {
            "self_attn.out_proj.weight" => "proj.weight".to_string(),
            "self_attn.out_proj.bias" => "proj.bias".to_string(),
            "self_attn_layer_norm.weight" => "norm1.weight".to_string(),
            "self_attn_layer_norm.bias" => "norm1.bias".to_string(),
            "final_layer_norm.weight" => "norm2.weight".to_string(),
            "final_layer_norm.bias" => "norm2.bias".to_string(),
            "fc1.weight" => "fc1.weight".to_string(),
            "fc1.bias" => "fc1.bias".to_string(),
            "fc2.weight" => "fc2.weight".to_string(),
            "fc2.bias" => "fc2.bias".to_string(),
            // q/k/v handled by the fuse step in the caller (needs sibling
            // tensors, not expressible as a pure 1:1 leaf map).
            "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight"
            | "self_attn.q_proj.bias" | "self_attn.k_proj.bias" | "self_attn.v_proj.bias" => return None,
            _ => return None,
        };
        return Some(format!("audio.blocks.{n}.{mapped}"));
    }
    match s {
        "conv2d1.weight" => Some("audio.conv2d1.weight".into()),
        "conv2d1.bias" => Some("audio.conv2d1.bias".into()),
        "conv2d2.weight" => Some("audio.conv2d2.weight".into()),
        "conv2d2.bias" => Some("audio.conv2d2.bias".into()),
        "conv2d3.weight" => Some("audio.conv2d3.weight".into()),
        "conv2d3.bias" => Some("audio.conv2d3.bias".into()),
        "conv_out.weight" => Some("audio.conv_out.weight".into()),
        "ln_post.weight" => Some("audio.ln_post.weight".into()),
        "ln_post.bias" => Some("audio.ln_post.bias".into()),
        "proj1.weight" => Some("audio.multi_modal_projector.linear_1.weight".into()),
        "proj1.bias" => Some("audio.multi_modal_projector.linear_1.bias".into()),
        "proj2.weight" => Some("audio.multi_modal_projector.linear_2.weight".into()),
        "proj2.bias" => Some("audio.multi_modal_projector.linear_2.bias".into()),
        _ => None,
    }
}

/// Fuse `thinker.audio_tower.layers.{b}.self_attn.{q,k,v}_proj.{weight,bias}`
/// into `audio.blocks.{b}.qkv.{weight,bias}` — the one transform
/// [`map_audio`] cannot express as a pure 1:1 leaf map, since it consumes
/// three source tensors per output. `src` must already hold every q/k/v
/// tensor for `b` (the caller buffers a layer's tensors until all six of
/// these six sibling names have arrived, then calls this once).
pub fn fuse_audio_qkv(b: u32, q_w: Vec<f32>, k_w: Vec<f32>, v_w: Vec<f32>, q_b: Vec<f32>, k_b: Vec<f32>, v_b: Vec<f32>) -> [(String, Vec<f32>); 2] {
    let mut w = q_w;
    w.extend(k_w);
    w.extend(v_w);
    let mut bias = q_b;
    bias.extend(k_b);
    bias.extend(v_b);
    [(format!("audio.blocks.{b}.qkv.weight"), w), (format!("audio.blocks.{b}.qkv.bias"), bias)]
}

// -------------------------------------------------------------------- vision
/// `thinker.visual.*` -> `vision.*`. DeepStack's per-tap mergers
/// (`merger_list.{i}.*`) keep their own index; the primary merger (used when
/// DeepStack is off, or as the final-stage merger) has no index.
pub fn map_vision(hf: &str) -> Option<String> {
    let s = hf.strip_prefix("thinker.visual.")?;
    if let Some(rest) = s.strip_prefix("blocks.") {
        let (n, leaf) = rest.split_once('.')?;
        let mapped = match leaf {
            "norm1.weight" => "norm1.weight".to_string(),
            "norm1.bias" => "norm1.bias".to_string(),
            "norm2.weight" => "norm2.weight".to_string(),
            "norm2.bias" => "norm2.bias".to_string(),
            "attn.qkv.weight" => "qkv.weight".to_string(),
            "attn.qkv.bias" => "qkv.bias".to_string(),
            "attn.proj.weight" => "proj.weight".to_string(),
            "attn.proj.bias" => "proj.bias".to_string(),
            "mlp.linear_fc1.weight" => "fc1.weight".to_string(),
            "mlp.linear_fc1.bias" => "fc1.bias".to_string(),
            "mlp.linear_fc2.weight" => "fc2.weight".to_string(),
            "mlp.linear_fc2.bias" => "fc2.bias".to_string(),
            _ => return None,
        };
        return Some(format!("vision.blocks.{n}.{mapped}"));
    }
    if let Some(rest) = s.strip_prefix("merger_list.") {
        let (i, leaf) = rest.split_once('.')?;
        return Some(format!("vision.deepstack_merger.{i}.{}", leaf_or_none(leaf)?));
    }
    if let Some(leaf) = s.strip_prefix("merger.") {
        return Some(format!("vision.merger.{}", leaf_or_none(leaf)?));
    }
    match s {
        "patch_embed.proj.weight" => Some("vision.patch_embed.proj.weight".into()),
        "patch_embed.proj.bias" => Some("vision.patch_embed.proj.bias".into()),
        "pos_embed.weight" => Some("vision.pos_embed.weight".into()),
        _ => None,
    }
}

/// `ln_q.{weight,bias}` and `mlp.{i}.{weight,bias}` pass through unchanged —
/// both merger variants already use names brain's own convention is fine
/// with (no HF `nn.Sequential`-index awkwardness to rename away).
fn leaf_or_none(leaf: &str) -> Option<String> {
    Some(leaf.to_string())
}

// ------------------------------------------------------------- MoE decoders
/// One MoE decoder's non-expert, non-router per-layer tensors (attention +
/// norms) — shared shape between the Thinker and Talker text decoders.
/// `prefix` is `"thinker"` or `"talker"`.
fn map_moe_attn(hf: &str, hf_prefix: &str, brain_prefix: &str) -> Option<String> {
    let s = hf.strip_prefix(hf_prefix)?;
    let rest = s.strip_prefix("layers.")?;
    let (n, leaf) = rest.split_once('.')?;
    let mapped = dense_attn_leaf(leaf)?;
    Some(format!("{brain_prefix}.blocks.{n}.{mapped}"))
}

/// One MoE decoder's router + expert tensors. `thinker.model.layers.{n}.mlp.*`
/// or `talker.model.layers.{n}.mlp.*` -> `{prefix}.blocks.{n}.mlp.*`. Every
/// expert keeps its own index (`experts.{e}.*`) — brain's sparse MoE core
/// (`model::moe`) reads one expert's weight at a time, never concatenated.
fn map_moe_mlp(hf: &str, hf_prefix: &str, brain_prefix: &str) -> Option<String> {
    let s = hf.strip_prefix(hf_prefix)?;
    let rest = s.strip_prefix("layers.")?;
    let (n, leaf) = rest.split_once('.')?;
    let mlp = leaf.strip_prefix("mlp.")?;
    let mapped = if mlp == "gate.weight" {
        "mlp.router.weight".to_string()
    } else if let Some(rest) = mlp.strip_prefix("experts.") {
        let (e, expert_leaf) = rest.split_once('.')?;
        let leaf = match expert_leaf {
            "gate_proj.weight" => "gate.weight",
            "up_proj.weight" => "up.weight",
            "down_proj.weight" => "down.weight",
            _ => return None,
        };
        format!("mlp.experts.{e}.{leaf}")
    } else if let Some(leaf) = mlp.strip_prefix("shared_expert.") {
        let leaf = match leaf {
            "gate_proj.weight" => "gate.weight",
            "up_proj.weight" => "up.weight",
            "down_proj.weight" => "down.weight",
            _ => return None,
        };
        format!("mlp.shared_expert.{leaf}")
    } else if mlp == "shared_expert_gate.weight" {
        "mlp.shared_expert_gate.weight".to_string()
    } else {
        return None;
    };
    Some(format!("{brain_prefix}.blocks.{n}.{mapped}"))
}

/// `thinker.model.*` -> `thinker.*` (embed/norm/head + every decoder layer).
pub fn map_thinker(hf: &str) -> Option<String> {
    match hf {
        "thinker.model.embed_tokens.weight" => return Some("thinker.embed_tokens.weight".into()),
        "thinker.model.norm.weight" => return Some("thinker.norm.weight".into()),
        "thinker.lm_head.weight" => return Some("thinker.lm_head.weight".into()),
        _ => {}
    }
    map_moe_attn(hf, "thinker.model.", "thinker").or_else(|| map_moe_mlp(hf, "thinker.model.", "thinker"))
}

/// `talker.model.*` + `talker.codec_head`/`hidden_projection`/
/// `text_projection` -> `talker.*`.
pub fn map_talker(hf: &str) -> Option<String> {
    match hf {
        "talker.model.codec_embedding.weight" => return Some("talker.codec_embedding.weight".into()),
        "talker.model.norm.weight" => return Some("talker.norm.weight".into()),
        "talker.codec_head.weight" => return Some("talker.codec_head.weight".into()),
        _ => {}
    }
    if let Some(rest) = hf.strip_prefix("talker.hidden_projection.") {
        return Some(format!("talker.hidden_projection.{rest}"));
    }
    if let Some(rest) = hf.strip_prefix("talker.text_projection.") {
        return Some(format!("talker.text_projection.{rest}"));
    }
    map_moe_attn(hf, "talker.model.", "talker").or_else(|| map_moe_mlp(hf, "talker.model.", "talker"))
}

// -------------------------------------------------------------- talker.code_predictor
/// `talker.code_predictor.*` -> `talker.code_predictor.*`. Deliberately
/// IDENTICAL to the HF path (no rename): `tts::mtp`'s loader already reads
/// this exact structure (5-layer Qwen3 block, per-codebook
/// `codec_embedding.{i}`/`lm_head.{i}`) for the standalone Qwen3-TTS MTP, and
/// Omni's `code_predictor_config` is the same shape at the same JSON path
/// (`omni::config::TalkerConfig::from_json` already reuses
/// `tts::config::MtpConfig::from_json` unchanged for this reason) — matching
/// names too means M7 can load this with `tts::mtp` directly, not a fork.
pub fn map_code_predictor(hf: &str) -> Option<String> {
    hf.starts_with("talker.code_predictor.").then(|| hf.to_string())
}

// ----------------------------------------------------------------- code2wav
/// `code2wav.*` -> `code2wav.*`, mostly identity (the SEANet decoder's own
/// names — `alpha`/`beta`/`conv.weight`/`conv.bias` — need no brain-side
/// rename; only the pre-transformer's attention block follows the shared
/// dense-attention convention).
pub fn map_code2wav(hf: &str) -> Option<String> {
    let s = hf.strip_prefix("code2wav.")?;
    if let Some(rest) = s.strip_prefix("pre_transformer.layers.") {
        let (n, leaf) = rest.split_once('.')?;
        if let Some(mapped) = dense_attn_leaf(leaf) {
            return Some(format!("code2wav.pre_transformer.blocks.{n}.{mapped}"));
        }
        let mapped = match leaf {
            "mlp.gate_proj.weight" => "mlp.gate.weight",
            "mlp.up_proj.weight" => "mlp.up.weight",
            "mlp.down_proj.weight" => "mlp.down.weight",
            "mlp_layer_scale.scale" => "mlp_layer_scale.scale",
            "self_attn_layer_scale.scale" => "self_attn_layer_scale.scale",
            _ => return None,
        };
        return Some(format!("code2wav.pre_transformer.blocks.{n}.{mapped}"));
    }
    if s == "pre_transformer.norm.weight" {
        return Some("code2wav.pre_transformer.norm.weight".into());
    }
    // decoder.*, upsample.*, code_embedding.weight: identity (see doc above).
    Some(format!("code2wav.{s}"))
}

/// The single dispatch every top-level tensor in the checkpoint goes through.
/// Returns `None` for a tensor this workstream's mapping does not (yet)
/// recognize — the caller treats that as a hard error (never a silent drop),
/// per the porting playbook's two-way-coverage rule.
pub fn hf_to_brain(hf: &str) -> Option<String> {
    map_audio(hf)
        .or_else(|| map_vision(hf))
        .or_else(|| map_thinker(hf))
        .or_else(|| map_code_predictor(hf)) // before map_talker: shares the "talker." prefix
        .or_else(|| map_talker(hf))
        .or_else(|| map_code2wav(hf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // Every assertion below is a REAL tensor name from the released
    // checkpoint's model.safetensors.index.json (dumped 2026-08-07), not an
    // invented example — see docs/models/omni/status.md.

    #[test]
    fn audio_tower_names() {
        assert_eq!(map_audio("thinker.audio_tower.conv2d1.weight").unwrap(), "audio.conv2d1.weight");
        assert_eq!(map_audio("thinker.audio_tower.conv_out.weight").unwrap(), "audio.conv_out.weight");
        assert_eq!(map_audio("thinker.audio_tower.layers.5.fc1.weight").unwrap(), "audio.blocks.5.fc1.weight");
        assert_eq!(map_audio("thinker.audio_tower.layers.0.self_attn.out_proj.bias").unwrap(), "audio.blocks.0.proj.bias");
        assert_eq!(
            map_audio("thinker.audio_tower.layers.31.self_attn_layer_norm.weight").unwrap(),
            "audio.blocks.31.norm1.weight"
        );
        assert_eq!(map_audio("thinker.audio_tower.proj1.weight").unwrap(), "audio.multi_modal_projector.linear_1.weight");
        assert_eq!(map_audio("thinker.audio_tower.proj2.bias").unwrap(), "audio.multi_modal_projector.linear_2.bias");
        // q/k/v are fused by fuse_audio_qkv, not a 1:1 leaf -> not mapped here.
        assert_eq!(map_audio("thinker.audio_tower.layers.0.self_attn.q_proj.weight"), None);
    }

    #[test]
    fn qkv_fuse_concatenates_in_qkv_order() {
        let [w, b] = fuse_audio_qkv(3, vec![1.0], vec![2.0], vec![3.0], vec![10.0], vec![20.0], vec![30.0]);
        assert_eq!(w, ("audio.blocks.3.qkv.weight".to_string(), vec![1.0, 2.0, 3.0]));
        assert_eq!(b, ("audio.blocks.3.qkv.bias".to_string(), vec![10.0, 20.0, 30.0]));
    }

    #[test]
    fn vision_tower_names() {
        assert_eq!(map_vision("thinker.visual.patch_embed.proj.weight").unwrap(), "vision.patch_embed.proj.weight");
        assert_eq!(map_vision("thinker.visual.pos_embed.weight").unwrap(), "vision.pos_embed.weight");
        assert_eq!(map_vision("thinker.visual.blocks.8.attn.qkv.weight").unwrap(), "vision.blocks.8.qkv.weight");
        assert_eq!(map_vision("thinker.visual.blocks.26.mlp.linear_fc2.bias").unwrap(), "vision.blocks.26.fc2.bias");
        assert_eq!(map_vision("thinker.visual.merger.ln_q.weight").unwrap(), "vision.merger.ln_q.weight");
        assert_eq!(map_vision("thinker.visual.merger.mlp.0.weight").unwrap(), "vision.merger.mlp.0.weight");
        assert_eq!(map_vision("thinker.visual.merger_list.2.ln_q.bias").unwrap(), "vision.deepstack_merger.2.ln_q.bias");
    }

    #[test]
    fn thinker_moe_names() {
        assert_eq!(map_thinker("thinker.model.embed_tokens.weight").unwrap(), "thinker.embed_tokens.weight");
        assert_eq!(map_thinker("thinker.model.norm.weight").unwrap(), "thinker.norm.weight");
        assert_eq!(map_thinker("thinker.lm_head.weight").unwrap(), "thinker.lm_head.weight");
        assert_eq!(
            map_thinker("thinker.model.layers.0.self_attn.q_proj.weight").unwrap(),
            "thinker.blocks.0.attn.wq.weight"
        );
        assert_eq!(
            map_thinker("thinker.model.layers.47.self_attn.k_norm.weight").unwrap(),
            "thinker.blocks.47.attn.k_norm.weight"
        );
        assert_eq!(map_thinker("thinker.model.layers.0.mlp.gate.weight").unwrap(), "thinker.blocks.0.mlp.router.weight");
        assert_eq!(
            map_thinker("thinker.model.layers.0.mlp.experts.127.down_proj.weight").unwrap(),
            "thinker.blocks.0.mlp.experts.127.down.weight"
        );
        // map_moe_mlp is a pure syntactic leaf map, not family-aware about
        // which MoE has a shared expert -- it would map a
        // "thinker....shared_expert_gate.weight" leaf too, but the real
        // checkpoint never emits one (Thinker's shared_expert_intermediate_size
        // is 0, confirmed in docs/models/omni/status.md), so this never runs
        // in practice. Asserted here so a future change to the shared-expert
        // arm doesn't silently start dropping it for the family that DOES
        // have one (talker_moe_names, below, is the real coverage case).
        assert!(map_thinker("thinker.model.layers.0.mlp.shared_expert_gate.weight").is_some());
    }

    #[test]
    fn talker_moe_names() {
        assert_eq!(map_talker("talker.model.codec_embedding.weight").unwrap(), "talker.codec_embedding.weight");
        assert_eq!(map_talker("talker.codec_head.weight").unwrap(), "talker.codec_head.weight");
        assert_eq!(
            map_talker("talker.hidden_projection.linear_fc1.weight").unwrap(),
            "talker.hidden_projection.linear_fc1.weight"
        );
        assert_eq!(
            map_talker("talker.model.layers.19.mlp.experts.5.up_proj.weight").unwrap(),
            "talker.blocks.19.mlp.experts.5.up.weight"
        );
        assert_eq!(
            map_talker("talker.model.layers.0.mlp.shared_expert.down_proj.weight").unwrap(),
            "talker.blocks.0.mlp.shared_expert.down.weight"
        );
        assert_eq!(
            map_talker("talker.model.layers.0.mlp.shared_expert_gate.weight").unwrap(),
            "talker.blocks.0.mlp.shared_expert_gate.weight"
        );
        // code_predictor lives under the same "talker." prefix but is a
        // different sub-model -- map_talker must not claim it.
        assert_eq!(map_talker("talker.code_predictor.model.norm.weight"), None);
    }

    #[test]
    fn code_predictor_is_identity() {
        assert_eq!(
            map_code_predictor("talker.code_predictor.model.layers.0.self_attn.q_proj.weight").unwrap(),
            "talker.code_predictor.model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            map_code_predictor("talker.code_predictor.model.codec_embedding.3.weight").unwrap(),
            "talker.code_predictor.model.codec_embedding.3.weight"
        );
        assert_eq!(map_code_predictor("talker.model.norm.weight"), None);
    }

    #[test]
    fn code2wav_names() {
        assert_eq!(map_code2wav("code2wav.code_embedding.weight").unwrap(), "code2wav.code_embedding.weight");
        assert_eq!(map_code2wav("code2wav.decoder.2.block.1.conv1.conv.weight").unwrap(), "code2wav.decoder.2.block.1.conv1.conv.weight");
        assert_eq!(map_code2wav("code2wav.upsample.0.1.gamma").unwrap(), "code2wav.upsample.0.1.gamma");
        assert_eq!(
            map_code2wav("code2wav.pre_transformer.layers.3.self_attn.q_proj.weight").unwrap(),
            "code2wav.pre_transformer.blocks.3.attn.wq.weight"
        );
        assert_eq!(
            map_code2wav("code2wav.pre_transformer.layers.0.mlp_layer_scale.scale").unwrap(),
            "code2wav.pre_transformer.blocks.0.mlp_layer_scale.scale"
        );
        assert_eq!(map_code2wav("code2wav.pre_transformer.norm.weight").unwrap(), "code2wav.pre_transformer.norm.weight");
    }

    #[test]
    fn full_dispatch_covers_every_family_and_nothing_else() {
        assert!(hf_to_brain("thinker.audio_tower.conv2d1.weight").is_some());
        assert!(hf_to_brain("thinker.visual.pos_embed.weight").is_some());
        assert!(hf_to_brain("thinker.model.norm.weight").is_some());
        assert!(hf_to_brain("talker.model.norm.weight").is_some());
        assert!(hf_to_brain("talker.code_predictor.model.norm.weight").is_some());
        assert!(hf_to_brain("code2wav.code_embedding.weight").is_some());
        assert_eq!(hf_to_brain("something.unrecognized.weight"), None);
    }

    /// Every distinct tensor-name SHAPE actually present in the released
    /// checkpoint (index dumped 2026-08-07, `\d+` positions normalized to
    /// `N`) must be recognized. This is the two-way-coverage check the
    /// porting playbook asks for, run against the real name list rather than
    /// a hand-picked sample — new HF tensor families the mapper doesn't know
    /// about fail loudly here instead of silently vanishing during a real
    /// import.
    #[test]
    fn covers_every_tensor_name_shape_in_the_real_checkpoint() {
        let samples = [
            "code2wav.code_embedding.weight",
            "code2wav.decoder.N.alpha",
            "code2wav.decoder.N.beta",
            "code2wav.decoder.N.block.N.act1.alpha",
            "code2wav.decoder.N.block.N.act1.beta",
            "code2wav.decoder.N.block.N.act2.alpha",
            "code2wav.decoder.N.block.N.act2.beta",
            "code2wav.decoder.N.block.N.alpha",
            "code2wav.decoder.N.block.N.beta",
            "code2wav.decoder.N.block.N.conv.bias",
            "code2wav.decoder.N.block.N.conv.weight",
            "code2wav.decoder.N.block.N.conv1.conv.bias",
            "code2wav.decoder.N.block.N.conv1.conv.weight",
            "code2wav.decoder.N.block.N.conv2.conv.bias",
            "code2wav.decoder.N.block.N.conv2.conv.weight",
            "code2wav.decoder.N.conv.bias",
            "code2wav.decoder.N.conv.weight",
            "code2wav.pre_transformer.layers.N.input_layernorm.weight",
            "code2wav.pre_transformer.layers.N.mlp.down_proj.weight",
            "code2wav.pre_transformer.layers.N.mlp.gate_proj.weight",
            "code2wav.pre_transformer.layers.N.mlp.up_proj.weight",
            "code2wav.pre_transformer.layers.N.mlp_layer_scale.scale",
            "code2wav.pre_transformer.layers.N.post_attention_layernorm.weight",
            "code2wav.pre_transformer.layers.N.self_attn.k_proj.weight",
            "code2wav.pre_transformer.layers.N.self_attn.o_proj.weight",
            "code2wav.pre_transformer.layers.N.self_attn.q_proj.weight",
            "code2wav.pre_transformer.layers.N.self_attn.v_proj.weight",
            "code2wav.pre_transformer.layers.N.self_attn_layer_scale.scale",
            "code2wav.pre_transformer.norm.weight",
            "code2wav.upsample.N.0.conv.bias",
            "code2wav.upsample.N.0.conv.weight",
            "code2wav.upsample.N.1.dwconv.conv.bias",
            "code2wav.upsample.N.1.dwconv.conv.weight",
            "code2wav.upsample.N.1.gamma",
            "code2wav.upsample.N.1.norm.bias",
            "code2wav.upsample.N.1.norm.weight",
            "code2wav.upsample.N.1.pwconv1.bias",
            "code2wav.upsample.N.1.pwconv1.weight",
            "code2wav.upsample.N.1.pwconv2.bias",
            "code2wav.upsample.N.1.pwconv2.weight",
            "talker.code_predictor.lm_head.N.weight",
            "talker.code_predictor.model.codec_embedding.N.weight",
            "talker.code_predictor.model.layers.N.input_layernorm.weight",
            "talker.code_predictor.model.layers.N.mlp.down_proj.weight",
            "talker.code_predictor.model.layers.N.mlp.gate_proj.weight",
            "talker.code_predictor.model.layers.N.mlp.up_proj.weight",
            "talker.code_predictor.model.layers.N.post_attention_layernorm.weight",
            "talker.code_predictor.model.layers.N.self_attn.k_norm.weight",
            "talker.code_predictor.model.layers.N.self_attn.k_proj.weight",
            "talker.code_predictor.model.layers.N.self_attn.o_proj.weight",
            "talker.code_predictor.model.layers.N.self_attn.q_norm.weight",
            "talker.code_predictor.model.layers.N.self_attn.q_proj.weight",
            "talker.code_predictor.model.layers.N.self_attn.v_proj.weight",
            "talker.code_predictor.model.norm.weight",
            "talker.codec_head.weight",
            "talker.hidden_projection.linear_fc1.bias",
            "talker.hidden_projection.linear_fc1.weight",
            "talker.hidden_projection.linear_fc2.bias",
            "talker.hidden_projection.linear_fc2.weight",
            "talker.model.codec_embedding.weight",
            "talker.model.layers.N.input_layernorm.weight",
            "talker.model.layers.N.mlp.experts.N.down_proj.weight",
            "talker.model.layers.N.mlp.experts.N.gate_proj.weight",
            "talker.model.layers.N.mlp.experts.N.up_proj.weight",
            "talker.model.layers.N.mlp.gate.weight",
            "talker.model.layers.N.mlp.shared_expert.down_proj.weight",
            "talker.model.layers.N.mlp.shared_expert.gate_proj.weight",
            "talker.model.layers.N.mlp.shared_expert.up_proj.weight",
            "talker.model.layers.N.mlp.shared_expert_gate.weight",
            "talker.model.layers.N.post_attention_layernorm.weight",
            "talker.model.layers.N.self_attn.k_norm.weight",
            "talker.model.layers.N.self_attn.k_proj.weight",
            "talker.model.layers.N.self_attn.o_proj.weight",
            "talker.model.layers.N.self_attn.q_norm.weight",
            "talker.model.layers.N.self_attn.q_proj.weight",
            "talker.model.layers.N.self_attn.v_proj.weight",
            "talker.model.norm.weight",
            "talker.text_projection.linear_fc1.bias",
            "talker.text_projection.linear_fc1.weight",
            "talker.text_projection.linear_fc2.bias",
            "talker.text_projection.linear_fc2.weight",
            "thinker.audio_tower.conv2d1.bias",
            "thinker.audio_tower.conv2d1.weight",
            "thinker.audio_tower.conv2d2.bias",
            "thinker.audio_tower.conv2d2.weight",
            "thinker.audio_tower.conv2d3.bias",
            "thinker.audio_tower.conv2d3.weight",
            "thinker.audio_tower.conv_out.weight",
            "thinker.audio_tower.layers.N.fc1.bias",
            "thinker.audio_tower.layers.N.fc1.weight",
            "thinker.audio_tower.layers.N.fc2.bias",
            "thinker.audio_tower.layers.N.fc2.weight",
            "thinker.audio_tower.layers.N.final_layer_norm.bias",
            "thinker.audio_tower.layers.N.final_layer_norm.weight",
            "thinker.audio_tower.layers.N.self_attn.k_proj.bias",
            "thinker.audio_tower.layers.N.self_attn.k_proj.weight",
            "thinker.audio_tower.layers.N.self_attn.out_proj.bias",
            "thinker.audio_tower.layers.N.self_attn.out_proj.weight",
            "thinker.audio_tower.layers.N.self_attn.q_proj.bias",
            "thinker.audio_tower.layers.N.self_attn.q_proj.weight",
            "thinker.audio_tower.layers.N.self_attn.v_proj.bias",
            "thinker.audio_tower.layers.N.self_attn.v_proj.weight",
            "thinker.audio_tower.layers.N.self_attn_layer_norm.bias",
            "thinker.audio_tower.layers.N.self_attn_layer_norm.weight",
            "thinker.audio_tower.ln_post.bias",
            "thinker.audio_tower.ln_post.weight",
            "thinker.audio_tower.proj1.bias",
            "thinker.audio_tower.proj1.weight",
            "thinker.audio_tower.proj2.bias",
            "thinker.audio_tower.proj2.weight",
            "thinker.lm_head.weight",
            "thinker.model.embed_tokens.weight",
            "thinker.model.layers.N.input_layernorm.weight",
            "thinker.model.layers.N.mlp.experts.N.down_proj.weight",
            "thinker.model.layers.N.mlp.experts.N.gate_proj.weight",
            "thinker.model.layers.N.mlp.experts.N.up_proj.weight",
            "thinker.model.layers.N.mlp.gate.weight",
            "thinker.model.layers.N.post_attention_layernorm.weight",
            "thinker.model.layers.N.self_attn.k_norm.weight",
            "thinker.model.layers.N.self_attn.k_proj.weight",
            "thinker.model.layers.N.self_attn.o_proj.weight",
            "thinker.model.layers.N.self_attn.q_norm.weight",
            "thinker.model.layers.N.self_attn.q_proj.weight",
            "thinker.model.layers.N.self_attn.v_proj.weight",
            "thinker.model.norm.weight",
            "thinker.visual.blocks.N.attn.proj.bias",
            "thinker.visual.blocks.N.attn.proj.weight",
            "thinker.visual.blocks.N.attn.qkv.bias",
            "thinker.visual.blocks.N.attn.qkv.weight",
            "thinker.visual.blocks.N.mlp.linear_fc1.bias",
            "thinker.visual.blocks.N.mlp.linear_fc1.weight",
            "thinker.visual.blocks.N.mlp.linear_fc2.bias",
            "thinker.visual.blocks.N.mlp.linear_fc2.weight",
            "thinker.visual.blocks.N.norm1.bias",
            "thinker.visual.blocks.N.norm1.weight",
            "thinker.visual.blocks.N.norm2.bias",
            "thinker.visual.blocks.N.norm2.weight",
            "thinker.visual.merger.ln_q.bias",
            "thinker.visual.merger.ln_q.weight",
            "thinker.visual.merger.mlp.N.bias",
            "thinker.visual.merger.mlp.N.weight",
            "thinker.visual.merger_list.N.ln_q.bias",
            "thinker.visual.merger_list.N.ln_q.weight",
            "thinker.visual.merger_list.N.mlp.N.bias",
            "thinker.visual.merger_list.N.mlp.N.weight",
            "thinker.visual.patch_embed.proj.bias",
            "thinker.visual.patch_embed.proj.weight",
            "thinker.visual.pos_embed.weight",
        ];
        // "N" placeholders substituted with 0 (or 0/0 for doubly-indexed
        // names) to get one concrete, dispatchable example per shape. The
        // audio/vision q/k/v leaves are handled by the qkv-fuse step, not
        // hf_to_brain directly, so they are the one deliberate exemption.
        let qkv_fuse_leaves = [
            "thinker.audio_tower.layers.N.self_attn.q_proj.weight",
            "thinker.audio_tower.layers.N.self_attn.k_proj.weight",
            "thinker.audio_tower.layers.N.self_attn.v_proj.weight",
            "thinker.audio_tower.layers.N.self_attn.q_proj.bias",
            "thinker.audio_tower.layers.N.self_attn.k_proj.bias",
            "thinker.audio_tower.layers.N.self_attn.v_proj.bias",
        ];
        let mut unmapped = Vec::new();
        for &shape in &samples {
            if qkv_fuse_leaves.contains(&shape) {
                continue;
            }
            let concrete = shape.replacen('N', "0", 2);
            if hf_to_brain(&concrete).is_none() {
                unmapped.push(shape);
            }
        }
        assert!(unmapped.is_empty(), "unmapped tensor name shapes from the real checkpoint: {unmapped:?}");
    }

    /// The inverse check: every tensor `hf_to_brain` accepts must land on a
    /// name under the right top-level component, so two components can never
    /// silently collide on the same brain-side key.
    #[test]
    fn every_mapped_name_is_prefixed_by_its_own_component() {
        let cases: &[(&str, &str)] = &[
            ("thinker.audio_tower.conv_out.weight", "audio."),
            ("thinker.visual.pos_embed.weight", "vision."),
            ("thinker.model.norm.weight", "thinker."),
            ("talker.model.norm.weight", "talker."),
            ("talker.code_predictor.model.norm.weight", "talker.code_predictor."),
            ("code2wav.code_embedding.weight", "code2wav."),
        ];
        for (hf, want_prefix) in cases {
            let got = hf_to_brain(hf).unwrap();
            assert!(got.starts_with(want_prefix), "{hf} -> {got}, expected prefix {want_prefix}");
        }
    }

    #[test]
    fn brain_init_from_hf_streams_and_fuses_qkv() {
        // A tiny synthetic HF tensor set covering one audio block (to
        // exercise the qkv-fuse path) plus one plain tensor.
        let mut src: HashMap<String, Vec<f32>> = HashMap::new();
        src.insert("thinker.audio_tower.conv2d1.weight".into(), vec![9.0]);
        src.insert("thinker.audio_tower.layers.0.self_attn.q_proj.weight".into(), vec![1.0, 2.0]);
        src.insert("thinker.audio_tower.layers.0.self_attn.k_proj.weight".into(), vec![3.0, 4.0]);
        src.insert("thinker.audio_tower.layers.0.self_attn.v_proj.weight".into(), vec![5.0, 6.0]);
        src.insert("thinker.audio_tower.layers.0.self_attn.q_proj.bias".into(), vec![0.1]);
        src.insert("thinker.audio_tower.layers.0.self_attn.k_proj.bias".into(), vec![0.2]);
        src.insert("thinker.audio_tower.layers.0.self_attn.v_proj.bias".into(), vec![0.3]);

        let mut out: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, data) in &src {
            if let Some(bn) = hf_to_brain(name) {
                out.insert(bn, data.clone());
            }
        }
        let [(wn, w), (bn, b)] = fuse_audio_qkv(
            0,
            src["thinker.audio_tower.layers.0.self_attn.q_proj.weight"].clone(),
            src["thinker.audio_tower.layers.0.self_attn.k_proj.weight"].clone(),
            src["thinker.audio_tower.layers.0.self_attn.v_proj.weight"].clone(),
            src["thinker.audio_tower.layers.0.self_attn.q_proj.bias"].clone(),
            src["thinker.audio_tower.layers.0.self_attn.k_proj.bias"].clone(),
            src["thinker.audio_tower.layers.0.self_attn.v_proj.bias"].clone(),
        );
        out.insert(wn, w);
        out.insert(bn, b);

        assert_eq!(out["audio.conv2d1.weight"], vec![9.0]);
        assert_eq!(out["audio.blocks.0.qkv.weight"], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(out["audio.blocks.0.qkv.bias"], vec![0.1, 0.2, 0.3]);
    }
}
