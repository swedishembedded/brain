// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! HF safetensors import for `Qwen/Qwen3.8-27B-FP8` - unlike `qwen35moe`
//! (imported from a pre-quantized GGUF release), this port reads the real
//! checkpoint's own HF safetensors directly, blockwise-FP8 weights and all.
//!
//! ## Real tensor names (confirmed vs. best-effort)
//!
//! Confirmed directly from the real checkpoint's `config.json`
//! `quantization_config.modules_to_not_convert` list (which enumerates every
//! tensor by its REAL name, whether or not this port's own scope reaches it
//! yet): every per-layer tensor lives under `model.language_model.layers.
//! {i}.`, with leaf names `input_layernorm`/`post_attention_layernorm`
//! (plain RMSNorm), `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,
//! out_proj}` + `linear_attn.{conv1d,A_log,dt_bias,norm}`, `self_attn.
//! {q_proj,k_proj,v_proj,o_proj,q_norm,k_norm}`, `mlp.{gate_proj,up_proj,
//! down_proj}`; the embedding is `model.embed_tokens.weight` (NOT nested
//! under `language_model`) and the head is the top-level `lm_head.weight`.
//! MTP (`mtp.*`) and vision (`model.visual.*`) tensors are real and present
//! but out of THIS import's scope - MTP's own import lands with the M7
//! model code, vision's with M9.
//!
//! **Best-effort, not yet checkpoint-verified:** the final norm's exact path
//! (`model.language_model.norm.weight`, inferred from the per-layer prefix
//! convention - a plain RMSNorm vector is never an FP8 quantization
//! candidate, so unlike every tensor above it never appears in
//! `modules_to_not_convert` to confirm it by name). If this guess is wrong,
//! [`import_dir`]'s two-way coverage fails LOUDLY, by name, the moment a
//! real checkpoint is available (M10) - never a silent placement of the
//! wrong tensor.
//!
//! ## FP8 handling
//!
//! Every `.weight` tensor may carry a sibling `<name>.weight_scale_inv`
//! (BF16, one scale per `128x128` block - `quantization_config.
//! weight_block_size: [128, 128]`). [`dequantize_fp8_pairs`] finds every such
//! pair, applies `model::fp8::dequant_block128`, and removes the `_scale_inv`
//! tensor from the map (metadata, not a real weight) - BEFORE name
//! classification, so [`classify`] only ever sees final, already-scaled f32
//! values. A `.weight` with no paired scale (every non-quantized tensor:
//! norms, `A_log`, `dt_bias`, embeddings, vision, `conv1d`) passes through
//! unchanged.
//!
//! ## The `(1+w)` RMSNorm fold
//!
//! HF's plain `Qwen3_5RMSNorm` stores `weight` such that the applied
//! multiplier is `1+weight` (zero-inits it for exactly that reason - see
//! `tools/goldens/qwen35_dump_reference.py`'s module doc, point 2, which
//! measures this directly against the real reference module). brain's own
//! `rmsnorm.wgsl`/fresh-weight convention (`crate::init`) assumes the STORED
//! value already IS the final multiplier. [`fold_plain_rmsnorm_weights`]
//! applies `+1.0` to every plain-RMSNorm weight at import time - `ln1`/`ln2`/
//! `self_attn.{q,k}_norm`/the final norm - and explicitly NOT to
//! `linear_attn.norm` (the GATED norm, `Qwen3_5RMSNormGated`, which has no
//! such reparameterization: its default `ones()` init is already the real
//! multiplier).
//!
//! ## Two-way coverage
//!
//! [`import_dir`] validates against [`Qwen35Config::param_list`] exactly like
//! `qwen35moe`/`flux1`'s own importers: every expected name present with the
//! right element count, and no source tensor left unclassified - a mismatch
//! is an error naming the tensor, never a silent zero-fill.

use std::collections::HashMap;
use std::path::Path;

use checkpoint::safetensors::StTensor;

use crate::config::{LayerType, Qwen35Config};

/// name -> (shape, fp32 data). Same shape as `flux1::import::Tensors`.
pub type Tensors = HashMap<String, (Vec<usize>, Vec<f32>)>;

/// Find every `<name>.weight_scale_inv` pair and fold it into `<name>` via
/// [`model::fp8::dequant_block128`], removing the scale tensor from the map.
/// Operates on REAL (pre-classification) HF names, so it is entirely
/// independent of the name-mapping question below.
pub fn dequantize_fp8_pairs(tensors: &mut Tensors, block: usize) -> Result<(), String> {
    let scale_names: Vec<String> =
        tensors.keys().filter(|k| k.ends_with(".weight_scale_inv")).cloned().collect();
    for scale_name in scale_names {
        let base_name = scale_name.strip_suffix("_scale_inv").expect("filtered by ends_with above").to_string();
        let (scale_shape, scale_data) =
            tensors.remove(&scale_name).ok_or_else(|| format!("import: {scale_name} vanished"))?;
        let (raw_shape, raw_data) = tensors
            .get(&base_name)
            .ok_or_else(|| format!("import: {scale_name} has no matching base tensor {base_name}"))?;
        if raw_shape.len() != 2 {
            return Err(format!("import: {base_name} has {} dims, expected 2 for FP8 blockwise", raw_shape.len()));
        }
        let (rows, cols) = (raw_shape[0], raw_shape[1]);
        let (rb, cb) = model::fp8::scale_shape(rows, cols, block);
        if scale_shape != [rb, cb] {
            return Err(format!(
                "import: {scale_name} shape {scale_shape:?}, expected [{rb}, {cb}] for {base_name} [{rows}, {cols}]"
            ));
        }
        let dequant = model::fp8::dequant_block128(raw_data, &scale_data, rows, cols, block);
        tensors.insert(base_name, (raw_shape.clone(), dequant));
    }
    Ok(())
}

/// Add `1.0` to every plain-RMSNorm weight (`ln1`/`ln2`/`self_attn.{q,k}_norm`/
/// the final `norm`) - see this module's doc, "The `(1+w)` RMSNorm fold".
/// Takes brain-canonical names (post-[`classify`]), so it cannot accidentally
/// touch `linear_attn.norm` (a different leaf name, never matched here).
fn fold_plain_rmsnorm_weights(tensors: &mut HashMap<String, Vec<f32>>) {
    for (name, data) in tensors.iter_mut() {
        let is_plain_norm = name == "norm.weight"
            || name.ends_with(".ln1.weight")
            || name.ends_with(".ln2.weight")
            || name.ends_with(".self_attn.q_norm.weight")
            || name.ends_with(".self_attn.k_norm.weight");
        if is_plain_norm {
            for v in data.iter_mut() {
                *v += 1.0;
            }
        }
    }
}

/// Real HF tensor name to brain canonical name. Three outcomes, not two:
/// `Ok(Some(name))` is a real tensor classified onto a brain canonical name.
/// `Ok(None)` is a DELIBERATE out-of-scope drop, limited to `mtp.*` and
/// `model.visual.*` (see this module's doc). `Err` covers a real tensor that
/// looked like it should be in scope (a known prefix,
/// `model.language_model.layers.`, or an unrecognized top-level name) but
/// didn't match anything this import understands. The distinction matters:
/// silently dropping the `Err` case into the same bucket as the deliberate
/// `mtp.`/`model.visual.` skip would hide a real bug, such as an
/// out-of-range layer index meaning `cfg.n_layers` disagrees with the
/// checkpoint, as a no-op instead of a loud failure.
fn classify(name: &str, cfg: &Qwen35Config) -> Result<Option<String>, String> {
    if name.starts_with("mtp.") || name.starts_with("model.visual.") || name == "model.visual" {
        return Ok(None);
    }
    if name == "model.embed_tokens.weight" {
        return Ok(Some("tok.weight".to_string()));
    }
    if name == "lm_head.weight" {
        return Ok(Some("lm_head.weight".to_string()));
    }
    if name == "model.language_model.norm.weight" {
        return Ok(Some("norm.weight".to_string()));
    }
    let Some(rest) = name.strip_prefix("model.language_model.layers.") else {
        return Err(format!("import: unrecognized tensor name {name}"));
    };
    let (idx_str, leaf) = rest
        .split_once('.')
        .ok_or_else(|| format!("import: cannot split a layer index from tensor name {name}"))?;
    let l: u32 = idx_str.parse().map_err(|_| format!("import: non-numeric layer index in tensor name {name}"))?;
    let types = cfg.layer_types();
    let ty = types
        .get(l as usize)
        .ok_or_else(|| format!("import: {name} references layer {l}, but cfg.n_layers is {}", types.len()))?;
    let p = |s: &str| format!("blocks.{l}.{s}");
    let mapped = match leaf {
        "input_layernorm.weight" => Some(p("ln1.weight")),
        "post_attention_layernorm.weight" => Some(p("ln2.weight")),
        "mlp.gate_proj.weight" => Some(p("mlp.gate.weight")),
        "mlp.up_proj.weight" => Some(p("mlp.up.weight")),
        "mlp.down_proj.weight" => Some(p("mlp.down.weight")),
        _ if *ty == LayerType::Linear && leaf.starts_with("linear_attn.") => {
            let sub = &leaf["linear_attn.".len()..];
            match sub {
                "in_proj_qkv.weight" | "in_proj_z.weight" | "in_proj_b.weight" | "in_proj_a.weight"
                | "out_proj.weight" | "A_log" | "dt_bias" | "norm.weight" | "conv1d.weight" => {
                    Some(p(&format!("linear_attn.{sub}")))
                }
                _ => None,
            }
        }
        _ if *ty == LayerType::Full && leaf.starts_with("self_attn.") => {
            let sub = &leaf["self_attn.".len()..];
            match sub {
                "q_proj.weight" | "k_proj.weight" | "v_proj.weight" | "o_proj.weight" | "q_norm.weight"
                | "k_norm.weight" => Some(p(&format!("self_attn.{sub}"))),
                _ => None,
            }
        }
        _ => None,
    };
    match mapped {
        Some(n) => Ok(Some(n)),
        None => Err(format!("import: unrecognized leaf {leaf:?} under a known-scope tensor {name}")),
    }
}

/// `conv1d.weight` ships as HF's raw `nn.Conv1d` shape `[conv_dim, 1,
/// kernel]`; squeeze the dead middle dim, matching `qwen35moe::import`'s
/// identical handling of the same tensor shape.
fn squeeze_conv1d(tensors: &mut Tensors) {
    for (name, (shape, _)) in tensors.iter_mut() {
        if name.ends_with(".linear_attn.conv1d.weight") && shape.len() == 3 && shape[1] == 1 {
            *shape = vec![shape[0], shape[2]];
        }
    }
}

/// Classify, rename, and validate two-way coverage against
/// [`Qwen35Config::param_list`]. `raw` is post-[`dequantize_fp8_pairs`]
/// (already-scaled f32 values) and pre-classification (real HF names).
pub fn import_from_tensors(raw: Tensors, cfg: &Qwen35Config) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut raw = raw;
    squeeze_conv1d(&mut raw);

    let mut renamed: HashMap<String, Vec<f32>> = HashMap::with_capacity(raw.len());
    for (name, (_, data)) in raw {
        if let Some(canon) = classify(&name, cfg)? {
            if renamed.insert(canon.clone(), data).is_some() {
                return Err(format!("import: two source tensors mapped onto {canon}"));
            }
        }
    }

    fold_plain_rmsnorm_weights(&mut renamed);

    let expected = cfg.param_list();
    let mut missing = Vec::new();
    for (name, numel) in &expected {
        match renamed.get(name) {
            None => missing.push(name.clone()),
            Some(d) if d.len() != *numel => {
                return Err(format!("import: {name} has {} values, expected {numel}", d.len()))
            }
            _ => {}
        }
    }
    if !missing.is_empty() {
        missing.truncate(16);
        return Err(format!("import: missing tensors: {missing:?}"));
    }
    let expected_names: std::collections::HashSet<&str> = expected.iter().map(|(n, _)| n.as_str()).collect();
    let extra: Vec<&String> = renamed.keys().filter(|k| !expected_names.contains(k.as_str())).collect();
    if !extra.is_empty() {
        return Err(format!(
            "import: {} tensors classified but not in param_list(), e.g. {:?} (a naming-convention bug in classify() itself, not a checkpoint problem)",
            extra.len(),
            extra.iter().take(8).collect::<Vec<_>>(),
        ));
    }
    Ok(renamed)
}

/// Read a checkpoint directory, dequantize every FP8 pair, classify, fold,
/// and validate two-way coverage - the whole pipeline this module's doc
/// describes. `block` is the checkpoint's `weight_block_size` (128 for the
/// real release).
pub fn import_dir(dir: &Path, cfg: &Qwen35Config, block: usize) -> Result<HashMap<String, Vec<f32>>, String> {
    let tensors: Vec<StTensor> = checkpoint::safetensors::read_model_dir(dir)?;
    let mut raw: Tensors = tensors.into_iter().map(|t| (t.name, (t.shape, t.data))).collect();
    dequantize_fp8_pairs(&mut raw, block)?;
    import_from_tensors(raw, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_f32(shape: Vec<usize>, v: f32) -> (Vec<usize>, Vec<f32>) {
        let n: usize = shape.iter().product();
        (shape, vec![v; n])
    }

    /// A synthetic checkpoint using the REAL naming convention this module
    /// documents, at `Qwen35Config::tiny()`'s shapes - the classify/validate
    /// pipeline is fully testable without a real checkpoint (deferred to
    /// M10), since it only depends on the NAMING convention being right.
    fn synthetic(cfg: &Qwen35Config) -> Tensors {
        let d = cfg.d_model as usize;
        let mut t: Tensors = HashMap::new();
        t.insert("model.embed_tokens.weight".into(), one_f32(vec![cfg.vocab as usize, d], 0.01));
        t.insert("lm_head.weight".into(), one_f32(vec![cfg.vocab as usize, d], 0.02));
        t.insert("model.language_model.norm.weight".into(), one_f32(vec![d], 0.0));
        for (l, ty) in cfg.layer_types().iter().enumerate() {
            let p = |s: &str| format!("model.language_model.layers.{l}.{s}");
            t.insert(p("input_layernorm.weight"), one_f32(vec![d], 0.0));
            t.insert(p("post_attention_layernorm.weight"), one_f32(vec![d], 0.0));
            match ty {
                LayerType::Linear => {
                    let kdim = cfg.linear_key_dim() as usize;
                    let vdim = cfg.linear_value_dim() as usize;
                    let conv_dim = cfg.linear_conv_dim() as usize;
                    let k = cfg.linear_conv_kernel_dim as usize;
                    let nvh = cfg.linear_num_value_heads as usize;
                    let hvd = cfg.linear_value_head_dim as usize;
                    t.insert(p("linear_attn.in_proj_qkv.weight"), one_f32(vec![conv_dim, d], 0.01));
                    t.insert(p("linear_attn.in_proj_z.weight"), one_f32(vec![vdim, d], 0.01));
                    t.insert(p("linear_attn.in_proj_b.weight"), one_f32(vec![nvh, d], 0.01));
                    t.insert(p("linear_attn.in_proj_a.weight"), one_f32(vec![nvh, d], 0.01));
                    // Real HF Conv1d shape [conv_dim, 1, kernel] - proves squeeze_conv1d.
                    t.insert(p("linear_attn.conv1d.weight"), one_f32(vec![conv_dim, 1, k], 0.01));
                    t.insert(p("linear_attn.A_log"), one_f32(vec![nvh], -1.0));
                    t.insert(p("linear_attn.dt_bias"), one_f32(vec![nvh], 1.0));
                    t.insert(p("linear_attn.norm.weight"), one_f32(vec![hvd], 1.0));
                    t.insert(p("linear_attn.out_proj.weight"), one_f32(vec![d, vdim], 0.01));
                    let _ = kdim;
                }
                LayerType::Full => {
                    let hq = cfg.q_dim() as usize;
                    let hqp = cfg.q_proj_dim() as usize;
                    let hkv = cfg.kv_dim() as usize;
                    let hd = cfg.head_dim as usize;
                    t.insert(p("self_attn.q_proj.weight"), one_f32(vec![hqp, d], 0.01));
                    t.insert(p("self_attn.k_proj.weight"), one_f32(vec![hkv, d], 0.01));
                    t.insert(p("self_attn.v_proj.weight"), one_f32(vec![hkv, d], 0.01));
                    t.insert(p("self_attn.q_norm.weight"), one_f32(vec![hd], 0.0));
                    t.insert(p("self_attn.k_norm.weight"), one_f32(vec![hd], 0.0));
                    t.insert(p("self_attn.o_proj.weight"), one_f32(vec![d, hq], 0.01));
                }
            }
            t.insert(p("mlp.gate_proj.weight"), one_f32(vec![cfg.intermediate_size as usize, d], 0.01));
            t.insert(p("mlp.up_proj.weight"), one_f32(vec![cfg.intermediate_size as usize, d], 0.01));
            t.insert(p("mlp.down_proj.weight"), one_f32(vec![d, cfg.intermediate_size as usize], 0.01));
        }
        // Real, present, out-of-scope tensors - must be silently dropped, not
        // treated as "unclassified" errors.
        t.insert("mtp.fc.weight".into(), one_f32(vec![d, 2 * d], 0.01));
        t.insert("model.visual.patch_embed.proj.weight".into(), one_f32(vec![8], 0.01));
        t
    }

    #[test]
    fn synthetic_checkpoint_imports_with_full_two_way_coverage() {
        let cfg = Qwen35Config::tiny();
        let raw = synthetic(&cfg);
        let out = import_from_tensors(raw, &cfg).expect("import must succeed");
        for (name, numel) in cfg.param_list() {
            let v = out.get(&name).unwrap_or_else(|| panic!("missing {name} after import"));
            assert_eq!(v.len(), numel, "{name}: wrong length");
        }
        assert_eq!(out.len(), cfg.param_list().len());
    }

    #[test]
    fn plain_rmsnorm_gets_the_1_plus_w_fold_gated_norm_does_not() {
        let cfg = Qwen35Config::tiny();
        let raw = synthetic(&cfg);
        let out = import_from_tensors(raw, &cfg).unwrap();
        // Every synthetic plain-norm weight was seeded at 0.0 -> must read 1.0 after the fold.
        assert!(out["blocks.0.ln1.weight"].iter().all(|&v| v == 1.0));
        assert!(out["norm.weight"].iter().all(|&v| v == 1.0));
        let full_layer = cfg.layer_types().iter().position(|t| *t == LayerType::Full).unwrap();
        assert!(out[&format!("blocks.{full_layer}.self_attn.q_norm.weight")].iter().all(|&v| v == 1.0));
        // The GATED norm was seeded at 1.0 and must be UNCHANGED (no fold).
        let linear_layer = cfg.layer_types().iter().position(|t| *t == LayerType::Linear).unwrap();
        assert!(out[&format!("blocks.{linear_layer}.linear_attn.norm.weight")].iter().all(|&v| v == 1.0));
    }

    #[test]
    fn conv1d_squeeze_drops_the_dead_middle_dim() {
        let cfg = Qwen35Config::tiny();
        let raw = synthetic(&cfg);
        let linear_layer = cfg.layer_types().iter().position(|t| *t == LayerType::Linear).unwrap();
        let expect_len = cfg.linear_conv_dim() as usize * cfg.linear_conv_kernel_dim as usize;
        let out = import_from_tensors(raw, &cfg).unwrap();
        assert_eq!(out[&format!("blocks.{linear_layer}.linear_attn.conv1d.weight")].len(), expect_len);
    }

    #[test]
    fn missing_tensor_errors_by_name() {
        let cfg = Qwen35Config::tiny();
        let mut raw = synthetic(&cfg);
        raw.remove("model.embed_tokens.weight");
        let err = import_from_tensors(raw, &cfg).unwrap_err();
        assert!(err.contains("tok.weight"), "error must name the missing tensor: {err}");
    }

    #[test]
    fn out_of_range_layer_index_errors_loudly_instead_of_silently_dropping() {
        // A real per-layer leaf name this import DOES classify, but at an
        // out-of-range layer index (cfg.n_layers=4 for tiny()) -- must error
        // by name, never fall into the same silent-drop bucket as a
        // deliberately out-of-scope tensor (mtp./model.visual.).
        let cfg = Qwen35Config::tiny();
        let mut raw = synthetic(&cfg);
        raw.insert("model.language_model.layers.99.input_layernorm.weight".into(), one_f32(vec![cfg.d_model as usize], 0.0));
        let err = import_from_tensors(raw, &cfg).unwrap_err();
        assert!(err.contains("layer 99"), "error must name the offending layer index: {err}");
    }

    #[test]
    fn unrecognized_top_level_tensor_errors_loudly() {
        let cfg = Qwen35Config::tiny();
        let mut raw = synthetic(&cfg);
        raw.insert("some_tensor_this_import_has_never_heard_of".into(), one_f32(vec![4], 0.0));
        let err = import_from_tensors(raw, &cfg).unwrap_err();
        assert!(err.contains("some_tensor_this_import_has_never_heard_of"), "error must name the tensor: {err}");
    }

    #[test]
    fn unrecognized_leaf_under_a_known_layer_prefix_errors_loudly() {
        let cfg = Qwen35Config::tiny();
        let mut raw = synthetic(&cfg);
        raw.insert("model.language_model.layers.0.some_new_field_this_import_predates.weight".into(), one_f32(vec![4], 0.0));
        let err = import_from_tensors(raw, &cfg).unwrap_err();
        assert!(err.contains("some_new_field_this_import_predates"), "error must name the tensor: {err}");
    }

    #[test]
    fn fp8_pair_dequantizes_and_the_scale_tensor_disappears() {
        let mut raw: Tensors = HashMap::new();
        // 2x2 raw block-quantized weight (block=2 -> one block, one scale).
        raw.insert("w.weight".into(), (vec![2, 2], vec![1.0, 1.0, 1.0, 1.0]));
        raw.insert("w.weight_scale_inv".into(), (vec![1, 1], vec![3.0]));
        dequantize_fp8_pairs(&mut raw, 2).unwrap();
        assert_eq!(raw["w.weight"].1, vec![3.0, 3.0, 3.0, 3.0]);
        assert!(!raw.contains_key("w.weight_scale_inv"), "scale tensor must be consumed, not left behind");
    }

    #[test]
    fn fp8_pair_wrong_scale_shape_errors_loudly() {
        let mut raw: Tensors = HashMap::new();
        raw.insert("w.weight".into(), (vec![4, 4], vec![1.0; 16]));
        // Should be [2,2] at block=2; this is [1,1] -- a wrong scale-tensor
        // shape must fail, not silently broadcast/truncate.
        raw.insert("w.weight_scale_inv".into(), (vec![1, 1], vec![3.0]));
        let err = dequantize_fp8_pairs(&mut raw, 2).unwrap_err();
        assert!(err.contains("weight_scale_inv"), "error must name the offending tensor: {err}");
    }
}
