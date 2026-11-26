// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Open a checkpoint for building: its config and its tensors under brain's
//! parameter names, whatever its format.

use crate::config::QwenConfig;
use checkpoint::TensorSource;

/// The config and a tensor source of the checkpoint at `path`, read as it
/// is on disk: nothing is converted ahead of time or written back.
///
/// - A Hugging Face checkpoint directory (`config.json` with safetensors or
///   `pytorch_model*.bin` weights) is configured by [`crate::hf::decoder_config`]
///   and read under brain's names through [`crate::import::owned_source`],
///   each tensor decoded to f32 as it is uploaded.
/// - A GGUF is read under brain's names straight off its mapping
///   ([`crate::gguf_import::open_source`]), with the shape its KV metadata
///   declares.
/// - A model-store directory holding only `model.brain.safetensors` (and the
///   repo's `config.json`) opens that brain checkpoint.
/// - Any other file is a brain checkpoint, opened with
///   [`checkpoint::weightio::WeightReader`] (through its packed view when it
///   stores bf16, FP8 or packed-int8 tensors, [`checkpoint::packed::stored_source`])
///   and configured by [`QwenConfig::from_reader`], which refuses a header
///   missing a shape key.
pub fn open_checkpoint(path: &str) -> Result<(QwenConfig, Box<dyn TensorSource>), String> {
    let brain = brain_checkpoint_in(path);
    let path = brain.as_deref().unwrap_or(path);
    let dir = std::path::Path::new(path);
    if dir.is_dir() {
        let cfg = hf_dir_config(dir)?;
        let reader = checkpoint::weightio::WeightReader::open_hf_dir(dir).map_err(|e| format!("cannot open {path}: {e}"))?;
        let src = crate::import::owned_source(reader, &cfg).map_err(|e| format!("{path}: {e}"))?;
        return Ok((cfg, Box::new(src)));
    }
    let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    if reader.gguf().is_some() {
        drop(reader);
        let (cfg, src) = crate::gguf_import::open_source(path)?;
        return Ok((cfg, Box::new(src)));
    }
    let cfg = QwenConfig::from_reader(&reader).map_err(|e| format!("{path}: {e}"))?;
    // A file with reduced-precision, block-scaled FP8 or packed-int8 tensors is read through its packed view: the
    // FP8 scales are sidecar tensors only that view applies, and the others upload in their stored encoding.
    Ok((cfg, checkpoint::packed::stored_source(reader)))
}

/// The config of the checkpoint at `path`, as [`open_checkpoint`] resolves it,
/// without opening a tensor source.
pub fn checkpoint_config(path: &str) -> Result<QwenConfig, String> {
    let brain = brain_checkpoint_in(path);
    let path = brain.as_deref().unwrap_or(path);
    let dir = std::path::Path::new(path);
    if dir.is_dir() {
        return hf_dir_config(dir);
    }
    let reader = checkpoint::weightio::WeightReader::open(path).map_err(|e| format!("cannot open {path}: {e}"))?;
    QwenConfig::from_reader(&reader).map_err(|e| format!("{path}: {e}"))
}

/// The floating tier the checkpoint at `path` stores its linears and head in: `BF16` or `F16` when every one of
/// them is stored at that half tier, else `F32` (an fp32 file, a mixed one, or one stored in a quantized encoding
/// the engine decodes). Reads tensor headers only. What [`crate::footprint::default_weight_tier`] keeps when the
/// caller names no tier.
pub fn stored_weight_tier(path: &str) -> Result<crate::Dtype, String> {
    let (cfg, src) = open_checkpoint(path)?;
    Ok(source_weight_tier(&cfg, &*src))
}

/// [`stored_weight_tier`] of an already opened source.
pub fn source_weight_tier(cfg: &QwenConfig, src: &dyn TensorSource) -> crate::Dtype {
    use crate::Dtype;
    let head = cfg.head_weight().to_string();
    let mut held = cfg.param_list().into_iter().map(|(name, _)| name).filter(|name| crate::q8::Q8::is_i8_linear(name) || *name == head).map(|name| model::ops::stored_tier(src, &name));
    match held.next().flatten() {
        Some(first @ (Dtype::BF16 | Dtype::F16)) if held.all(|t| t == Some(first)) => first,
        _ => Dtype::F32,
    }
}

/// The tiers the engine holds `src`'s linears and head at when the checkpoint stores them block-quantized (a GGUF's
/// Q4_K, Q5_K, Q6_K, Q8_0, ...), leaf by leaf: what [`gguf::held_tier`] keeps of each stored format - Q4_K as `Q4K`,
/// Q5_K as `Q8K`, any other block format at int8 - and fp32 for a leaf stored as floats. The parts of one fused GEMM
/// take one tier ([`crate::serve::Engine::check_tiers`]): their common one; int8 when they are all block-quantized
/// in different formats (a Q4_K_M's Q6_K `wv` beside Q4_K `wq`/`wk`: the K-quant parts are re-quantized to int8,
/// error within half an int8 step of each 32-element group); fp32 when one part is stored as floats. `None` when no
/// linear is block-quantized: [`source_weight_tier`] describes that checkpoint. Reads tensor headers only.
pub fn stored_weight_policy(cfg: &QwenConfig, src: &dyn TensorSource) -> Option<model::ops::TierPolicy> {
    use crate::Dtype;
    use checkpoint::gguf::GgmlType;
    let stored = |name: &str| src.raw_block_layout(name).map(|layout| layout.ty).filter(|ty| !matches!(ty, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 | GgmlType::F64)).map(gguf::held_tier);
    let one_tier = |tiers: &[Option<Dtype>]| match tiers {
        [first, rest @ ..] if rest.iter().all(|t| t == first) => first.unwrap_or(Dtype::F32),
        _ if tiers.iter().all(Option::is_some) => Dtype::I8,
        _ => Dtype::F32,
    };
    let mut leaves: Vec<(String, Dtype, u64)> = Vec::new();
    let mut any_quantized = false;
    for l in 0..cfg.n_layers {
        for group in [&["attn.wq.weight", "attn.wk.weight", "attn.wv.weight"][..], &["attn.wo.weight"], &["mlp.gate.weight", "mlp.up.weight"], &["mlp.down.weight"]] {
            let names: Vec<String> = group.iter().map(|leaf| format!("blocks.{l}.{leaf}")).collect();
            let tiers: Vec<Option<Dtype>> = names.iter().map(|name| stored(name)).collect();
            any_quantized |= tiers.iter().any(Option::is_some);
            let tier = one_tier(&tiers);
            for (name, leaf) in names.into_iter().zip(group) {
                let (n, k) = crate::model::linear_dims(cfg, leaf);
                leaves.push((name, tier, u64::from(n) * u64::from(k)));
            }
        }
    }
    let head = cfg.head_weight();
    // An untied checkpoint without its own head ties it anyway: the engine uploads the head from the embedding.
    let head_src = if src.numel(head).is_some() { head } else { "tok.weight" };
    leaves.push((head.to_string(), stored(head_src).unwrap_or(Dtype::F32), u64::from(cfg.vocab) * u64::from(cfg.d_model)));
    any_quantized.then(|| model::ops::TierPolicy::per_leaf(leaves))
}

/// The brain checkpoint a model-store directory holds in place of upstream
/// weights - `model.brain.safetensors` beside the repo's `config.json`, which
/// is what a converted pull leaves - or `None` when `path` is not such a
/// directory. A directory that still holds upstream safetensors or
/// `pytorch_model*.bin` weights is opened as the Hugging Face checkpoint.
fn brain_checkpoint_in(path: &str) -> Option<String> {
    let dir = std::path::Path::new(path);
    let brain = dir.join("model.brain.safetensors");
    if !dir.is_dir() || !brain.is_file() {
        return None;
    }
    let upstream = std::fs::read_dir(dir).ok()?.filter_map(|e| e.ok()).any(|e| {
        let name = e.file_name().to_string_lossy().into_owned();
        (name.ends_with(".safetensors") && !name.ends_with(".brain.safetensors")) || name.starts_with("pytorch_model")
    });
    (!upstream).then(|| brain.to_string_lossy().into_owned())
}

fn hf_dir_config(dir: &std::path::Path) -> Result<QwenConfig, String> {
    let json = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("{}: {e}", dir.join("config.json").display()))?;
    crate::hf::decoder_config(&json).map_err(|e| format!("{}: {e}", dir.display()))
}

/// The path [`open_checkpoint`] should open for a base the model store
/// resolved to `weights` inside `dir`: a `transformers` directory is answered
/// by its `config.json` and is opened as the directory, anything else as the
/// file.
pub fn store_checkpoint_path(weights: &std::path::Path, dir: &std::path::Path) -> std::path::PathBuf {
    if weights.file_name().is_some_and(|n| n == "config.json") {
        dir.to_path_buf()
    } else {
        weights.to_path_buf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model-store directory `brain pull` converted holds the upstream
    /// `config.json` beside `model.brain.safetensors` and no upstream weights.
    /// Named as a directory it opens that brain checkpoint, as the file would.
    #[test]
    fn a_store_directory_holding_only_the_brain_checkpoint_opens_it() {
        let dir = std::env::temp_dir().join(format!("qwen3-open-brain-dir-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = QwenConfig::tiny();
        let weights = crate::init_weights(&cfg, 5);
        let tensors: Vec<(String, Vec<u64>, Vec<f32>)> = cfg
            .param_list()
            .into_iter()
            .map(|(name, numel)| (name.clone(), vec![numel as u64], weights[&name].clone()))
            .collect();
        checkpoint::st::save_safetensors(dir.join("model.brain.safetensors").to_str().unwrap(), &tensors, &cfg.to_json(), None).unwrap();
        std::fs::write(dir.join("config.json"), r#"{"architectures": ["Qwen3ForCausalLM"]}"#).unwrap();

        let path = dir.to_str().unwrap();
        let (got, _) = open_checkpoint(path).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!((got.n_layers, got.d_model, got.vocab), (cfg.n_layers, cfg.d_model, cfg.vocab));
        assert_eq!(checkpoint_config(path).unwrap().d_model, cfg.d_model);
        std::fs::remove_dir_all(&dir).ok();
    }
}
