// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! HF → brain name remap + streaming import for Qwen3-Omni-30B-A3B-Instruct,
//! via `checkpoint::weightio` (never the whole 70.5 GB checkpoint in memory
//! at once — one tensor at a time, same as `qwen`/`glm`/`lfm`'s importers).
//!
//! **On-disk size**: a plain f32 checkpoint of this model needs ~141 GB
//! (bf16 doubled) — more than either filesystem on the box this was
//! developed on, and wasteful even where it fits (141 GB on disk to produce
//! a ~35 GB int8-resident model on every load). [`import_as`] instead
//! quantizes every large 2-D weight matrix to int8 AT IMPORT TIME via
//! `checkpoint::weightio::StWriter::create_mixed`/`write_u32` (a genuine,
//! additive `checkpoint::weightio` extension — existing f32 models are
//! untouched) — the on-disk checkpoint is ~35 GB, small enough to fit this
//! box's 93 GB tmpfs. This is a deliberate departure from every OTHER
//! model's convention (f32 on disk, quantize transiently at residency-load
//! time - `crates/qwen3/src/q8.rs`, `crates/s3dit/src/block.rs`): Omni is
//! large enough, specifically because of its MoE expert count, that the
//! disk cost of the old convention is prohibitive rather than merely
//! wasteful.
//! Norms, biases, layer-scales, and embeddings/heads stay f32 (any tensor
//! that is not rank-2, or whose last dimension is not a multiple of 4 —
//! `model::int8::quantize_weight`'s hard requirement — falls back to f32
//! automatically; [`should_quantize`] is the single decision point).
//!
//! Every mapping function here is pure (`&str -> Option<String>`), so it is
//! unit-tested against real tensor names from the released checkpoint's
//! `model.safetensors.index.json` without touching any tensor bytes — the
//! same shape `qwen3vl::import::map_vision`/`map_decoder` and
//! `qwen3asr::import::map_audio_encoder` already use.
//!
//! Naming targets deliberately match those two crates' existing conventions
//! (`blocks.N.attn.wq/wk/wv/wo`, `blocks.N.qkv` fused, `multi_modal_projector.
//! linear_{1,2}`) rather than inventing new ones - the shared encoder
//! implementations are hoisted onto Omni's scale, and matching names is what
//! keeps that hoist a "config bump", not a "second copy".

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

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

/// The six HF leaves [`map_audio`] deliberately does not map (they need
/// sibling tensors to fuse into one `qkv`, so [`import_as`] buffers and
/// concatenates them itself instead). Shared with the real-checkpoint
/// coverage test (`tests/real_import_coverage.rs`), which needs the same
/// exemption list to know which real tensor names `hf_to_brain` legitimately
/// rejects.
pub fn is_qkv_fuse_leaf(name: &str) -> bool {
    name.starts_with("thinker.audio_tower.layers.")
        && (name.ends_with("self_attn.q_proj.weight")
            || name.ends_with("self_attn.k_proj.weight")
            || name.ends_with("self_attn.v_proj.weight")
            || name.ends_with("self_attn.q_proj.bias")
            || name.ends_with("self_attn.k_proj.bias")
            || name.ends_with("self_attn.v_proj.bias"))
}

// --------------------------------------------------------------------- audio
/// `thinker.audio_tower.*` -> `audio.*`. Fuses `self_attn.{q,k,v}_proj` into
/// one `blocks.{b}.qkv` (matching `qwen3asr::import::map_audio_encoder`,
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
        return Some(format!("vision.deepstack_merger.{i}.{}", merger_leaf(leaf)?));
    }
    if let Some(leaf) = s.strip_prefix("merger.") {
        return Some(format!("vision.merger.{}", merger_leaf(leaf)?));
    }
    // Both names drop a segment to match qwen3vl::encoder::VisionEncoder's
    // exact expected keys (qwen3vl::import::map_vision does the identical
    // strip for Qwen3-VL-4B's own patch_embed.proj.*/pos_embed.weight): the
    // encoder isn't being changed, so its keys are the contract, not HF's.
    match s {
        "patch_embed.proj.weight" => Some("vision.patch_embed.weight".into()),
        "patch_embed.proj.bias" => Some("vision.patch_embed.bias".into()),
        "pos_embed.weight" => Some("vision.pos_embed".into()),
        _ => None,
    }
}

/// HF Omni merger leaf -> `qwen3vl::encoder::PatchMerger`'s expected key.
/// Omni's own HF naming (`ln_q`, `mlp.0`/`mlp.2` — an `nn.Sequential(Linear,
/// GELU, Linear)`, so index 1 is the weightless activation) differs from
/// Qwen3-VL-4B's (`norm`, `linear_fc1`/`linear_fc2`,
/// `qwen3vl::import::merger_leaf`), but both mergers are the SAME
/// LayerNorm->Linear->GELU->Linear shape, so both map onto the one target
/// key set `PatchMerger` actually reads: `ln`/`fc1`/`fc2`.
fn merger_leaf(leaf: &str) -> Option<&'static str> {
    Some(match leaf {
        "ln_q.weight" => "ln.weight",
        "ln_q.bias" => "ln.bias",
        "mlp.0.weight" => "fc1.weight",
        "mlp.0.bias" => "fc1.bias",
        "mlp.2.weight" => "fc2.weight",
        "mlp.2.bias" => "fc2.bias",
        _ => return None,
    })
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
/// `talker.code_predictor.*` -> `qwen3tts::import::mtp_hf_to_brain`'s UNPREFIXED
/// `blocks.N.*`/`norm.weight`/`codec_embedding.N.weight`/`lm_head.N.weight`
/// convention -- exactly what `qwen3tts::mtp::MtpModel::load_inference`'s
/// `ParamStore` lookups expect, so a real served speech-output action can
/// load the code predictor straight out of this unified checkpoint, the
/// same way it loads everything else.
///
/// This does NOT keep the `talker.code_predictor.` prefix as a pure identity
/// mapping (an earlier version did, reasoning it needed to stay "namespaced
/// away from Talker's own `talker.blocks.*`") -- but Talker's
/// own attention/MLP tensors map to `talker.blocks.N.*` (a `talker.`
/// prefix, via `map_moe_attn`/`map_moe_mlp`'s `brain_prefix` argument), so
/// `mtp_hf_to_brain`'s bare `blocks.N.*` never actually collides with
/// anything else in this flat unified namespace; the "namespace collision"
/// concern that earlier reasoning raised doesn't hold. Reusing `mtp_hf_to_brain`
/// (rather than re-deriving the same rename here) is the "one
/// implementation" answer -- `qwen3tts::import::import_mtp` already validated it
/// for the standalone Qwen3-TTS MTP, and `crates/omni/tests/
/// code_predictor_parity.rs` already validated `MtpModel`'s forward pass
/// against real Omni weights renamed this exact way.
pub fn map_code_predictor(hf: &str) -> Option<String> {
    qwen3tts::import::mtp_hf_to_brain(hf)
}

// ----------------------------------------------------------------- code2wav
/// `code2wav.*` -> a plain PREFIX STRIP, identity otherwise — exactly what
/// `mimi::Codec::transformer`/`decode_omni`'s `self.w(name)`/`self.host[name]`
/// lookups expect: raw HF leaf names (`pre_transformer.layers.{l}.self_attn.
/// q_proj.weight`, `input_layernorm.weight`, `mlp.gate_proj.weight`,
/// `self_attn_layer_scale.scale`, …), `layers` not `blocks`, no dense-attn
/// leaf rename at all.
///
/// This does NOT rename `pre_transformer.layers.N.*` onto the shared
/// dense-attention convention (`blocks.N.attn.wq.weight` etc., via
/// `dense_attn_leaf`) the way Thinker/Talker's own tensors are renamed (an
/// earlier version did, to match their style) - but `mimi::Codec` (read
/// directly from `crates/mimi/src/model.rs`'s `transformer()`, lines
/// 507-552) was never given that convention; it reads the untouched HF leaf
/// names straight off its `ParamStore`. That rename made this unified
/// checkpoint's code2wav tensors unloadable by their own consumer.
/// `crates/qwen3omnimoe/tests/code2wav_parity.rs`
/// already validated `Codec`'s forward pass against real Omni weights read
/// this exact (prefix-stripped, otherwise untouched) way.
pub fn map_code2wav(hf: &str) -> Option<String> {
    hf.strip_prefix("code2wav.").map(str::to_string)
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

/// Whether a tensor of this shape should be int8-quantized: rank-2 (a plain
/// `[n, k]` weight matrix — attention/expert/shared-expert/router
/// projections, embeddings, heads) with `k` (the last, contraction,
/// dimension) a multiple of 4 (`model::int8::quantize_weight`'s hard
/// requirement). Every real 2-D weight in this checkpoint meets that (2048,
/// 1024, 1152, 768, 384, 4304, 5120, 3072, 1280, 480, 152064, … are all
/// multiples of 4); convolutions
/// (rank 3: `[out, in, kernel]`) and every 1-D tensor (norms, biases,
/// layer-scales) fall through to f32 automatically, not by a special case.
pub fn should_quantize(shape: &[u64]) -> bool {
    shape.len() == 2 && shape[1].is_multiple_of(4)
}

/// Buffers one audio-tower layer's q/k/v weight+bias until all six sibling
/// tensors have streamed past (they are not necessarily adjacent in the
/// source's iteration order), then hands the fused pair to the caller.
#[derive(Default)]
struct QkvBuf {
    q_w: Option<Vec<f32>>,
    k_w: Option<Vec<f32>>,
    v_w: Option<Vec<f32>>,
    q_b: Option<Vec<f32>>,
    k_b: Option<Vec<f32>>,
    v_b: Option<Vec<f32>>,
}

/// Stream `<hf_dir>/config.json` + sharded safetensors into an int8-native
/// brain checkpoint at `out_path`. Peak host memory is one HF tensor's f32
/// expansion at a time (the audio q/k/v fuse buffers at most 32 layers' worth
/// of small attention-projection tensors concurrently — a few MB, not a
/// concern next to a single MoE expert's own ~5 MB).
///
/// Fails loudly and writes nothing on error (`StWriter::finish` refuses a
/// plan with holes) — this streams the SOURCE in the source's own natural
/// order, not the output plan's order, which is exactly what
/// `StWriter::write`/`write_u32` support (write any planned name once, in any
/// order).
pub fn import_as(hf_dir: &str, out_path: &str, id_override: Option<&str>) -> Result<(), String> {
    let dir = std::path::Path::new(hf_dir);
    let cfg_json = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("read config.json: {e}"))?;
    // Parsed only as a validity gate -- a malformed config.json must fail
    // import before any tensor byte streams, not partway through. The
    // fields aren't otherwise consumed here: name mapping and the
    // quantize/keep-f32 decision are shape-driven (from the source tensors
    // themselves), not config-driven.
    let _cfg = crate::config::OmniConfig::parse(&cfg_json).map_err(|e| format!("parse config.json: {e}"))?;

    let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("open {hf_dir}: {e}"))?;

    // Pass 1: decide the brain-side name, shape and dtype for every source
    // tensor (header-only — `shape()` never touches tensor bytes), fusing
    // the audio qkv triples into one planned entry. This is what
    // `StWriter::create_mixed` needs up front; pass 2 (below) streams data.
    enum PlanItem {
        Direct { brain_name: String, quant: bool },
        /// One of the six audio q/k/v HF leaves — folded into a single fused
        /// `audio.blocks.{b}.qkv.{weight,bias}` plan entry the first time any
        /// sibling of that (layer, weight-or-bias) pair is seen.
        QkvLeaf,
    }
    let mut items: HashMap<String, PlanItem> = HashMap::new();
    let mut plan: Vec<(String, Vec<u64>, checkpoint::weightio::Dtype)> = Vec::new();
    let mut fused_qkv_planned: std::collections::HashSet<(u32, bool)> = std::collections::HashSet::new(); // (layer, is_weight)
    for name in reader.names() {
        let shape = reader.shape(name).unwrap_or_else(|| panic!("no shape for {name}")).to_vec();
        if is_qkv_fuse_leaf(name) {
            let b: u32 = name.strip_prefix("thinker.audio_tower.layers.").unwrap().split_once('.').unwrap().0.parse().unwrap();
            let is_weight = name.ends_with(".weight");
            if fused_qkv_planned.insert((b, is_weight)) {
                let leaf = if is_weight { "qkv.weight" } else { "qkv.bias" };
                let fused_name = format!("audio.blocks.{b}.{leaf}");
                // qkv fuse concatenates 3 equal-sized rows/elements along dim 0.
                let mut fused_shape = shape.clone();
                fused_shape[0] *= 3;
                plan.push((fused_name, fused_shape, checkpoint::weightio::Dtype::F32));
            }
            items.insert(name.to_string(), PlanItem::QkvLeaf);
            continue;
        }
        let brain_name = hf_to_brain(name).ok_or_else(|| format!("import: no mapping for HF tensor {name:?}"))?;
        let quant = should_quantize(&shape);
        if quant {
            let (n, k) = (shape[0], shape[1]);
            plan.push((brain_name.clone(), vec![n, k / 4], checkpoint::weightio::Dtype::U32));
            plan.push((format!("{brain_name}.scale"), vec![n], checkpoint::weightio::Dtype::F32));
        } else {
            plan.push((brain_name.clone(), shape, checkpoint::weightio::Dtype::F32));
        }
        items.insert(name.to_string(), PlanItem::Direct { brain_name, quant });
    }

    let param_count: u64 = plan.iter().map(|(_, s, _)| s.iter().product::<u64>()).sum();
    let id = id_override.unwrap_or_else(|| Path::new(out_path).file_stem().and_then(|s| s.to_str()).unwrap_or("omni"));
    let mut card = checkpoint::st::ModelCard::new(id, "omni");
    card.param_count = Some(param_count);
    let mut writer = checkpoint::weightio::StWriter::create_mixed(out_path, &plan, &serde_json::to_value(&cfg_json).unwrap_or(Value::Null), Some(&card))
        .map_err(|e| format!("create {out_path}: {e}"))?;

    // Pass 2: stream tensor DATA through in the source's own order, writing
    // each planned entry as it completes. qkv triples accumulate in `qkv_buf`
    // until all three weights (or biases) for a layer have arrived.
    let mut qkv_buf: HashMap<u32, QkvBuf> = HashMap::new();
    let mut err: Option<String> = None;
    reader.for_each(|name, _shape, data| {
        if err.is_some() {
            return;
        }
        match items.get(name) {
            Some(PlanItem::QkvLeaf) => {
                let b: u32 = name.strip_prefix("thinker.audio_tower.layers.").unwrap().split_once('.').unwrap().0.parse().unwrap();
                let buf = qkv_buf.entry(b).or_default();
                let is_weight = name.ends_with(".weight");
                let slot = if name.contains(".q_proj.") {
                    if is_weight { &mut buf.q_w } else { &mut buf.q_b }
                } else if name.contains(".k_proj.") {
                    if is_weight { &mut buf.k_w } else { &mut buf.k_b }
                } else {
                    if is_weight { &mut buf.v_w } else { &mut buf.v_b }
                };
                *slot = Some(data);
                // `.take()` unconditionally clears the field it's called on,
                // even inside a tuple whose `if let` pattern ends up not
                // matching — checking `is_some()` on all three FIRST (never
                // touching the buffer) is what makes taking them afterward
                // safe: a partial arrival (e.g. only q_w so far) leaves every
                // field untouched for the next call to find.
                if buf.q_w.is_some() && buf.k_w.is_some() && buf.v_w.is_some() {
                    let mut w = buf.q_w.take().unwrap();
                    w.extend(buf.k_w.take().unwrap());
                    w.extend(buf.v_w.take().unwrap());
                    if let Err(e) = writer.write(&format!("audio.blocks.{b}.qkv.weight"), &w) {
                        err = Some(e.to_string());
                    }
                }
                if buf.q_b.is_some() && buf.k_b.is_some() && buf.v_b.is_some() {
                    let mut bias = buf.q_b.take().unwrap();
                    bias.extend(buf.k_b.take().unwrap());
                    bias.extend(buf.v_b.take().unwrap());
                    if let Err(e) = writer.write(&format!("audio.blocks.{b}.qkv.bias"), &bias) {
                        err = Some(e.to_string());
                    }
                }
            }
            Some(PlanItem::Direct { brain_name, quant }) => {
                if *quant {
                    let shape = reader.shape(name).unwrap();
                    let (n, k) = (shape[0] as usize, shape[1] as usize);
                    let (packed, scale) = model::int8::quantize_weight(&data, n, k);
                    if let Err(e) = writer.write_u32(brain_name, &packed) {
                        err = Some(e.to_string());
                    } else if let Err(e) = writer.write(&format!("{brain_name}.scale"), &scale) {
                        err = Some(e.to_string());
                    }
                } else if let Err(e) = writer.write(brain_name, &data) {
                    err = Some(e.to_string());
                }
            }
            None => err = Some(format!("import: streamed tensor {name:?} was not in the plan (bug: pass 1/pass 2 drifted)")),
        }
    });
    if let Some(e) = err {
        return Err(e);
    }
    writer.finish().map_err(|e| e.to_string())?;
    eprintln!("omni: imported {} planned tensors -> {out_path}", plan.len());
    Ok(())
}

#[cfg(test)]
mod import_as_tests {
    //! `import_as` end to end against a synthetic HF checkpoint (matching
    //! `qwen3tts::import`'s own precedent - a small but structurally real fixture,
    //! not the full 70.5 GB checkpoint) — proves the streaming mechanism
    //! (qkv fuse, the quantize-or-keep-f32 decision, int8-native writing)
    //! rather than re-proving names (`import.rs`'s other tests already cover
    //! those against the real checkpoint's actual tensor list).
    use super::*;

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("omni-import-test-{name}-{}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn streams_qkv_fuse_and_quantizes_2d_weights() {
        let dir = scratch_dir("e2e");
        // d=8 (multiple of 4, so every 2-D weight below is a real
        // should_quantize candidate), 2 experts, 1 layer per decoder.
        let config = serde_json::json!({
            "thinker_config": {
                "audio_config": {"num_mel_bins": 8, "d_model": 8, "encoder_attention_heads": 2,
                    "encoder_ffn_dim": 8, "encoder_layers": 1, "downsample_hidden_size": 8, "output_dim": 8},
                "vision_config": {"depth": 1, "hidden_size": 8, "num_heads": 2, "intermediate_size": 8,
                    "patch_size": 4, "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 8},
                "text_config": {"num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 2,
                    "num_key_value_heads": 2, "head_dim": 4, "moe_intermediate_size": 8,
                    "shared_expert_intermediate_size": 0, "num_experts": 2, "num_experts_per_tok": 1,
                    "vocab_size": 8},
            },
            "talker_config": {
                "text_config": {"num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 2,
                    "num_key_value_heads": 2, "head_dim": 4, "moe_intermediate_size": 8,
                    "shared_expert_intermediate_size": 8, "num_experts": 2, "num_experts_per_tok": 1,
                    "vocab_size": 8},
                "code_predictor_config": {"num_hidden_layers": 1, "hidden_size": 8, "head_dim": 4,
                    "num_attention_heads": 2, "num_key_value_heads": 2, "intermediate_size": 8,
                    "vocab_size": 8, "num_code_groups": 2},
            },
            "code2wav_config": {"hidden_size": 8, "intermediate_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2, "sliding_window": 4,
                "upsample_rates": [2], "upsampling_ratios": [2]},
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();

        // (hf name, shape) -- deterministic fixed values, never random, per
        // the engine's test-PRNG convention (a plain counter is enough here:
        // int8 tolerance is checked, not exact reproduction of a real model).
        let src_plan: Vec<(&str, Vec<u64>)> = vec![
            ("thinker.audio_tower.layers.0.self_attn.q_proj.weight", vec![8, 8]),
            ("thinker.audio_tower.layers.0.self_attn.k_proj.weight", vec![8, 8]),
            ("thinker.audio_tower.layers.0.self_attn.v_proj.weight", vec![8, 8]),
            ("thinker.audio_tower.layers.0.self_attn.q_proj.bias", vec![8]),
            ("thinker.audio_tower.layers.0.self_attn.k_proj.bias", vec![8]),
            ("thinker.audio_tower.layers.0.self_attn.v_proj.bias", vec![8]),
            ("thinker.audio_tower.layers.0.self_attn.out_proj.weight", vec![8, 8]), // quantized (2D)
            ("thinker.audio_tower.layers.0.self_attn_layer_norm.weight", vec![8]), // f32 (1D)
            ("thinker.model.embed_tokens.weight", vec![8, 8]),
            ("thinker.model.norm.weight", vec![8]),
            ("thinker.model.layers.0.input_layernorm.weight", vec![8]),
            ("thinker.model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
            ("thinker.model.layers.0.mlp.gate.weight", vec![2, 8]), // router
            ("thinker.model.layers.0.mlp.experts.0.gate_proj.weight", vec![8, 8]),
            ("thinker.model.layers.0.mlp.experts.0.up_proj.weight", vec![8, 8]),
            ("thinker.model.layers.0.mlp.experts.0.down_proj.weight", vec![8, 8]),
            ("thinker.model.layers.0.mlp.experts.1.gate_proj.weight", vec![8, 8]),
            ("thinker.model.layers.0.mlp.experts.1.up_proj.weight", vec![8, 8]),
            ("thinker.model.layers.0.mlp.experts.1.down_proj.weight", vec![8, 8]),
            ("talker.model.layers.0.mlp.shared_expert.gate_proj.weight", vec![8, 8]),
            ("talker.code_predictor.model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
            ("code2wav.pre_transformer.norm.weight", vec![8]),
        ];
        let src_data: HashMap<&str, Vec<f32>> =
            src_plan.iter().map(|(n, s)| (*n, (0..s.iter().product::<u64>()).map(|i| (i as f32 - 4.0) * 0.37).collect())).collect();

        let src_plan_owned: Vec<(String, Vec<u64>)> = src_plan.iter().map(|(n, s)| (n.to_string(), s.clone())).collect();
        let mut w = checkpoint::weightio::StWriter::create(
            dir.join("model.safetensors").to_str().unwrap(),
            &src_plan_owned,
            &Value::Null,
            None,
        )
        .unwrap();
        for (name, _) in &src_plan {
            w.write(name, &src_data[name]).unwrap();
        }
        w.finish().unwrap();

        let out_path = dir.join("model.brain.safetensors");
        import_as(dir.to_str().unwrap(), out_path.to_str().unwrap(), Some("test/omni-tiny")).unwrap();

        // Independent verification via the raw safetensors crate.
        let out = std::fs::read(&out_path).unwrap();
        let sts = safetensors::SafeTensors::deserialize(&out).unwrap();

        // qkv fused correctly: concatenated q|k|v, in that order.
        let qkv = sts.tensor("audio.blocks.0.qkv.weight").unwrap();
        assert_eq!(qkv.shape(), &[24, 8]); // 3*8 rows
        let qkv_f32: Vec<f32> = qkv.data().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let mut want = src_data["thinker.audio_tower.layers.0.self_attn.q_proj.weight"].clone();
        want.extend(&src_data["thinker.audio_tower.layers.0.self_attn.k_proj.weight"]);
        want.extend(&src_data["thinker.audio_tower.layers.0.self_attn.v_proj.weight"]);
        assert_eq!(qkv_f32, want);
        let qkv_bias = sts.tensor("audio.blocks.0.qkv.bias").unwrap();
        assert_eq!(qkv_bias.shape(), &[24]);

        // A 2-D weight was quantized: packed U32 + sibling F32 scale exist,
        // and dequantizing round-trips within the same tolerance
        // model::int8::quantize_weight's own test asserts.
        let (_, meta) = safetensors::SafeTensors::read_metadata(&out).unwrap();
        let tensor_infos = meta.tensors();
        assert_eq!(tensor_infos.get("thinker.embed_tokens.weight").unwrap().dtype, safetensors::Dtype::U32);
        assert_eq!(tensor_infos.get("thinker.embed_tokens.weight.scale").unwrap().dtype, safetensors::Dtype::F32);

        let packed_view = sts.tensor("thinker.embed_tokens.weight").unwrap();
        let scale_view = sts.tensor("thinker.embed_tokens.weight.scale").unwrap();
        // The tensor-level byte claim (not a whole-file comparison, which at
        // this toy 8x8 scale is swamped by per-tensor JSON header overhead --
        // real savings come from the WEIGHT bytes shrinking to a quarter, which only
        // dominates the header at real model row widths): packed u32 +
        // per-row f32 scale must still be fewer bytes than the f32-equivalent
        // for this same tensor (8*8*4 = 256 B) even at this tiny size, since
        // the scale is only 1 f32 per ROW (8), not per element (64).
        let f32_equivalent_bytes = 8 * 8 * 4;
        let quantized_bytes = packed_view.data().len() + scale_view.data().len();
        assert!(quantized_bytes < f32_equivalent_bytes, "quantized {quantized_bytes} B should be less than f32-equivalent {f32_equivalent_bytes} B");
        let packed_u32: Vec<u32> = packed_view.data().chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
        let scale: Vec<f32> = scale_view.data().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let original = &src_data["thinker.model.embed_tokens.weight"];
        for r in 0..8usize {
            for c in 0..8usize {
                let word = packed_u32[r * 2 + c / 4];
                let q = ((word >> (8 * (c % 4))) & 0xff) as u8 as i8;
                let deq = q as f32 * scale[r];
                assert!((deq - original[r * 8 + c]).abs() <= scale[r] * 0.5 + 1e-6, "row {r} col {c}: deq={deq} orig={}", original[r * 8 + c]);
            }
        }

        // A 1-D tensor was kept exact f32 (no quantization, no scale sibling).
        let norm = sts.tensor("thinker.norm.weight").unwrap();
        let norm_f32: Vec<f32> = norm.data().chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        assert_eq!(norm_f32, src_data["thinker.model.norm.weight"]);
        assert!(meta.tensors().iter().find(|(n, _)| *n == "thinker.norm.weight.scale").is_none(), "a 1-D tensor must not get a scale sibling");

        // code_predictor's rename (qwen3tts::import::mtp_hf_to_brain) reached the
        // output under its unprefixed MtpModel-loader-compatible name.
        assert!(meta.tensors().iter().any(|(n, _)| n == "blocks.0.attn.wq.weight"));

        std::fs::remove_dir_all(&dir).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every assertion below is a REAL tensor name from the released
    // checkpoint's model.safetensors.index.json (dumped 2026-08-07), not an
    // invented example.

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
        // Both strip a segment to match VisionEncoder's own expected keys.
        assert_eq!(map_vision("thinker.visual.patch_embed.proj.weight").unwrap(), "vision.patch_embed.weight");
        assert_eq!(map_vision("thinker.visual.pos_embed.weight").unwrap(), "vision.pos_embed");
        assert_eq!(map_vision("thinker.visual.blocks.8.attn.qkv.weight").unwrap(), "vision.blocks.8.qkv.weight");
        assert_eq!(map_vision("thinker.visual.blocks.26.mlp.linear_fc2.bias").unwrap(), "vision.blocks.26.fc2.bias");
        // ln_q/mlp.{0,2} rename to PatchMerger's ln/fc1/fc2 -- see merger_leaf's doc.
        assert_eq!(map_vision("thinker.visual.merger.ln_q.weight").unwrap(), "vision.merger.ln.weight");
        assert_eq!(map_vision("thinker.visual.merger.mlp.0.weight").unwrap(), "vision.merger.fc1.weight");
        assert_eq!(map_vision("thinker.visual.merger.mlp.2.bias").unwrap(), "vision.merger.fc2.bias");
        assert_eq!(map_vision("thinker.visual.merger_list.2.ln_q.bias").unwrap(), "vision.deepstack_merger.2.ln.bias");
        assert_eq!(map_vision("thinker.visual.merger_list.1.mlp.2.weight").unwrap(), "vision.deepstack_merger.1.fc2.weight");
        // mlp.1 is the weightless GELU in the nn.Sequential -- never mapped.
        assert_eq!(map_vision("thinker.visual.merger.mlp.1.weight"), None);
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
        // is 0), so this never runs
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
    fn code_predictor_matches_tts_mtp_hf_to_brain() {
        // Delegates to qwen3tts::import::mtp_hf_to_brain -- MtpModel::load_inference's
        // ParamStore-compatible, unprefixed naming, not an identity mapping.
        assert_eq!(
            map_code_predictor("talker.code_predictor.model.layers.0.self_attn.q_proj.weight").unwrap(),
            "blocks.0.attn.wq.weight"
        );
        assert_eq!(map_code_predictor("talker.code_predictor.model.codec_embedding.3.weight").unwrap(), "codec_embedding.3.weight");
        assert_eq!(map_code_predictor("talker.code_predictor.model.norm.weight").unwrap(), "norm.weight");
        assert_eq!(map_code_predictor("talker.code_predictor.lm_head.2.weight").unwrap(), "lm_head.2.weight");
        assert_eq!(map_code_predictor("talker.model.norm.weight"), None);
    }

    #[test]
    fn code2wav_names() {
        // A plain prefix strip -- mimi::Codec's own ParamStore lookups (see
        // map_code2wav's doc) read raw HF leaf names, "layers" not "blocks".
        assert_eq!(map_code2wav("code2wav.code_embedding.weight").unwrap(), "code_embedding.weight");
        assert_eq!(map_code2wav("code2wav.decoder.2.block.1.conv1.conv.weight").unwrap(), "decoder.2.block.1.conv1.conv.weight");
        assert_eq!(map_code2wav("code2wav.upsample.0.1.gamma").unwrap(), "upsample.0.1.gamma");
        assert_eq!(map_code2wav("code2wav.pre_transformer.layers.3.self_attn.q_proj.weight").unwrap(), "pre_transformer.layers.3.self_attn.q_proj.weight");
        assert_eq!(map_code2wav("code2wav.pre_transformer.layers.0.mlp_layer_scale.scale").unwrap(), "pre_transformer.layers.0.mlp_layer_scale.scale");
        assert_eq!(map_code2wav("code2wav.pre_transformer.norm.weight").unwrap(), "pre_transformer.norm.weight");
        assert_eq!(map_code2wav("thinker.model.norm.weight"), None);
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
    /// silently collide on the same brain-side key. `talker.code_predictor.*`
    /// and `code2wav.*` are the two deliberate exceptions (see
    /// `map_code_predictor`/`map_code2wav`'s docs): both map to their
    /// consumer's own unprefixed `ParamStore` convention
    /// (`qwen3tts::mtp::MtpModel`, `mimi::Codec`), verified not to collide with
    /// anything else in this flat namespace by
    /// `unprefixed_components_dont_collide_with_anything` below.
    #[test]
    fn every_mapped_name_is_prefixed_by_its_own_component() {
        let cases: &[(&str, &str)] = &[
            ("thinker.audio_tower.conv_out.weight", "audio."),
            ("thinker.visual.pos_embed.weight", "vision."),
            ("thinker.model.norm.weight", "thinker."),
            ("talker.model.norm.weight", "talker."),
        ];
        for (hf, want_prefix) in cases {
            let got = hf_to_brain(hf).unwrap();
            assert!(got.starts_with(want_prefix), "{hf} -> {got}, expected prefix {want_prefix}");
        }
    }

    #[test]
    fn unprefixed_components_dont_collide_with_anything() {
        // code_predictor's and code2wav's unprefixed names must not equal
        // any OTHER component's own mapped name for an analogous tensor, nor
        // each other's.
        let cp = map_code_predictor("talker.code_predictor.model.layers.0.self_attn.q_proj.weight").unwrap();
        let c2w = map_code2wav("code2wav.pre_transformer.layers.0.self_attn.q_proj.weight").unwrap();
        let talker = map_talker("talker.model.layers.0.self_attn.q_proj.weight").unwrap();
        let thinker = map_thinker("thinker.model.layers.0.self_attn.q_proj.weight").unwrap();
        let names = [&cp, &c2w, &talker, &thinker];
        for (i, a) in names.iter().enumerate() {
            for b in &names[i + 1..] {
                assert_ne!(a, b, "unexpected name collision between mapped components");
            }
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
