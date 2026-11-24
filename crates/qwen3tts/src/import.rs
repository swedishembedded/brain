// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Read a HuggingFace `Qwen3-TTS` checkpoint (`config.json` +
//! `model.safetensors`) as brain's Talker decoder and MTP code-predictor, as
//! it is downloaded ([`talker_view`], [`mtp_view`]), or write either out as a
//! brain `.safetensors` container ([`import_talker`], [`import_mtp`]).
//!
//! Convention match (identical to `crate::qwen3::import`): brain's `matmul.wgsl`
//! is `out = x @ Wᵀ` with `W:[out,in]` row-major — exactly HF `nn.Linear.weight`;
//! the embedding tables are `[vocab, hidden]` row-major in both. So **no tensor
//! is transposed**; each view is a pure 1:1 name remap + bf16→f32 dequant.
//!
//! The Talker decoder is loaded by [`crate::qwen3::Qwen`] with `tie_embeddings =
//! false`: `talker.model.codec_embedding → tok.weight`, `talker.codec_head →
//! lm_head.weight`. The text-conditioning tensors (`talker.model.text_embedding`,
//! `talker.text_projection.*`) ride along in the same container under
//! `text_embedding.weight` / `text_projection.*` names (ignored by the Qwen
//! loader, picked up by [`crate::talker::TalkerModel`]).

use std::collections::HashMap;
use std::path::Path;

use checkpoint::weightio::{Renamed, WeightReader};
use serde_json::Value;

use crate::config::{MtpConfig, TalkerConfig};

/// The Talker tensors outside its decoder layers: `(upstream name, brain name)`.
/// One table read both ways ([`talker_hf_to_brain`], [`talker_brain_to_hf`]).
const TALKER_NAMES: &[(&str, &str)] = &[
    ("talker.model.codec_embedding.weight", "tok.weight"),
    ("talker.model.norm.weight", "norm.weight"),
    ("talker.model.text_embedding.weight", "text_embedding.weight"),
    ("talker.codec_head.weight", "lm_head.weight"),
    ("talker.text_projection.linear_fc1.weight", "text_projection.fc1.weight"),
    ("talker.text_projection.linear_fc1.bias", "text_projection.fc1.bias"),
    ("talker.text_projection.linear_fc2.weight", "text_projection.fc2.weight"),
    ("talker.text_projection.linear_fc2.bias", "text_projection.fc2.bias"),
];

/// The code predictor's tensors outside its decoder layers and per-codebook tables.
const MTP_NAMES: &[(&str, &str)] = &[
    ("talker.code_predictor.model.norm.weight", "norm.weight"),
    ("talker.code_predictor.small_to_mtp_projection.weight", "small_to_mtp_projection.weight"),
    ("talker.code_predictor.small_to_mtp_projection.bias", "small_to_mtp_projection.bias"),
];

/// The code predictor's per-codebook tables: `(upstream prefix, brain prefix)`,
/// each followed by `{i}.weight` on both sides.
const MTP_TABLES: &[(&str, &str)] = &[("talker.code_predictor.model.codec_embedding.", "codec_embedding."), ("talker.code_predictor.lm_head.", "lm_head.")];

/// Where each component's decoder layers live: `(upstream prefix, brain prefix)`,
/// each followed by `{n}.{leaf}` with the Qwen3 leaf names mapped by `qwen3::hf`.
const TALKER_LAYERS: (&str, &str) = ("talker.model.layers.", "blocks.");
const MTP_LAYERS: (&str, &str) = ("talker.code_predictor.model.layers.", "blocks.");

/// Translate `name` through `fixed`, `tables` and `layers`, from the upstream
/// side to brain's (`to_brain`) or back.
type Names = &'static [(&'static str, &'static str)];

fn translate(name: &str, fixed: Names, tables: Names, layers: (&'static str, &'static str), to_brain: bool) -> Option<String> {
    let pick = |&(hf, brain): &(&'static str, &'static str)| if to_brain { (hf, brain) } else { (brain, hf) };
    if let Some((_, to)) = fixed.iter().map(pick).find(|(from, _)| *from == name) {
        return Some(to.to_string());
    }
    for (from, to) in tables.iter().map(pick) {
        if let Some(i) = name.strip_prefix(from).and_then(|r| r.strip_suffix(".weight")).filter(|i| i.parse::<u32>().is_ok()) {
            return Some(format!("{to}{i}.weight"));
        }
    }
    let (from, to) = pick(&layers);
    let (n, leaf) = name.strip_prefix(from)?.split_once('.')?;
    n.parse::<u32>().ok()?;
    let leaf = if to_brain { qwen3::hf::layer_leaf(leaf)? } else { qwen3::hf::hf_layer_leaf(leaf)? };
    Some(format!("{to}{n}.{leaf}"))
}

/// Map an HF `talker.*` tensor name to its brain Talker-container name, or `None`
/// to drop it (a non-Talker tensor: `code_predictor.*`, `speaker_encoder.*`).
pub fn talker_hf_to_brain(name: &str) -> Option<String> {
    translate(name, TALKER_NAMES, &[], TALKER_LAYERS, true)
}

/// The upstream name of a brain Talker tensor: the inverse of [`talker_hf_to_brain`].
pub fn talker_brain_to_hf(name: &str) -> Option<String> {
    translate(name, TALKER_NAMES, &[], TALKER_LAYERS, false)
}

/// Map an HF `talker.code_predictor.*` tensor name to its brain MTP-container
/// name, or `None` to drop it.
pub fn mtp_hf_to_brain(name: &str) -> Option<String> {
    translate(name, MTP_NAMES, MTP_TABLES, MTP_LAYERS, true)
}

/// The upstream name of a brain MTP tensor: the inverse of [`mtp_hf_to_brain`].
pub fn mtp_brain_to_hf(name: &str) -> Option<String> {
    translate(name, MTP_NAMES, MTP_TABLES, MTP_LAYERS, false)
}

fn read_config(dir: &Path) -> Result<Value, String> {
    let s = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    serde_json::from_str(&s).map_err(|e| format!("parse config.json: {e}"))
}

fn open_source(dir: &Path) -> Result<WeightReader, String> {
    let st = dir.join("model.safetensors");
    if !st.exists() {
        return Err(format!(
            "missing {}: sharded checkpoints are not supported",
            st.display()
        ));
    }
    WeightReader::open(st.to_str().unwrap())
        .map_err(|e| format!("import: opening checkpoint: {e}"))
}

/// The Talker decoder parameter list (Qwen3, untied) plus the text-conditioning
/// extras, with the element counts the import must match.
fn talker_param_specs(cfg: &TalkerConfig) -> Vec<(String, usize)> {
    let mut out = cfg.to_qwen(0).param_list(); // tok/blocks/norm/lm_head (untied)
    let th = cfg.text_hidden_size as usize;
    let d = cfg.d_model as usize;
    out.push((
        "text_embedding.weight".to_string(),
        cfg.text_vocab_size as usize * th,
    ));
    out.push(("text_projection.fc1.weight".to_string(), th * th));
    out.push(("text_projection.fc1.bias".to_string(), th));
    out.push(("text_projection.fc2.weight".to_string(), d * th));
    out.push(("text_projection.fc2.bias".to_string(), d));
    out
}

/// MTP parameter list: the 5-layer decoder (blocks + norm) plus the 15 input
/// codec-embedding tables and 15 output lm_head tables.
fn mtp_param_specs(cfg: &MtpConfig) -> Vec<(String, usize)> {
    let d = cfg.d_model as usize;
    let ff = cfg.d_ff as usize;
    let hq = cfg.q_dim() as usize;
    let hkv = cfg.kv_dim() as usize;
    let hd = cfg.head_dim as usize;
    let v = cfg.vocab as usize;
    let mut out = Vec::new();
    for l in 0..cfg.n_layers {
        let p = |s: &str| format!("blocks.{l}.{s}");
        out.push((p("ln1.weight"), d));
        out.push((p("attn.wq.weight"), hq * d));
        out.push((p("attn.wk.weight"), hkv * d));
        out.push((p("attn.wv.weight"), hkv * d));
        out.push((p("attn.q_norm.weight"), hd));
        out.push((p("attn.k_norm.weight"), hd));
        out.push((p("attn.wo.weight"), d * hq));
        out.push((p("ln2.weight"), d));
        out.push((p("mlp.gate.weight"), ff * d));
        out.push((p("mlp.up.weight"), ff * d));
        out.push((p("mlp.down.weight"), d * ff));
    }
    out.push(("norm.weight".to_string(), d));
    let emb = cfg.embedding_dim as usize;
    for i in 0..cfg.n_residual() {
        // codec_embedding rows are in the Talker hidden width (`embedding_dim`);
        // lm_head reads the MTP decoder hidden width (`d_model`).
        out.push((format!("codec_embedding.{i}.weight"), v * emb));
        out.push((format!("lm_head.{i}.weight"), v * d));
    }
    // small_to_mtp_projection (embedding_dim -> d_model) exists only when the two
    // widths differ (the 1.7B); the 0.6B has no such tensor (Identity).
    if emb != d {
        out.push(("small_to_mtp_projection.weight".to_string(), d * emb));
        out.push(("small_to_mtp_projection.bias".to_string(), d));
    }
    out
}

/// One component of the checkpoint at `dir` under its brain names: every
/// tensor `map` keeps, checked against the component's parameter list (a
/// missing tensor, a wrong element count, or a kept tensor the list does not
/// name is an error), with `config` as the view's config.
fn component(dir: &Path, specs: &[(String, usize)], map: impl Fn(&str) -> Option<String>, config: Value) -> Result<WeightReader, String> {
    let src = open_source(dir)?;
    let mut from: HashMap<String, String> = HashMap::new();
    for name in src.names() {
        if let Some(brain) = map(name) {
            if from.insert(brain.clone(), name.to_string()).is_some() {
                return Err(format!("import: two tensors map to {brain}"));
            }
        }
    }
    let mut pairs = Vec::with_capacity(specs.len());
    for (brain, numel) in specs {
        let hf = from.remove(brain).ok_or_else(|| format!("import: checkpoint has no tensor for {brain}"))?;
        let got: u64 = src.shape(&hf).unwrap_or_default().iter().product();
        if got != *numel as u64 {
            return Err(format!("import: {hf} has {got} elements, {brain} needs {numel}"));
        }
        pairs.push((brain.clone(), hf));
    }
    if let Some(extra) = from.keys().next() {
        return Err(format!("import: {} maps to {extra}, which the config does not declare", from[extra]));
    }
    Ok(WeightReader::derived(Box::new(Renamed::new(src, pairs, config)?)))
}

/// The Talker decoder (+ text-conditioning tensors) of the HF checkpoint at
/// `dir`, read under brain's names as it is downloaded. Its config is the
/// Qwen3 decoder config (untied) the shared loader parses; the talker's
/// M-RoPE/code-group metadata is not needed for the decoder forward.
pub fn talker_view(dir: &Path) -> Result<WeightReader, String> {
    let cfg = TalkerConfig::from_json(&read_config(dir)?);
    component(dir, &talker_param_specs(&cfg), talker_hf_to_brain, cfg.to_qwen(2048).to_json())
}

/// The MTP code-predictor of the HF checkpoint at `dir`, under brain's names.
pub fn mtp_view(dir: &Path) -> Result<WeightReader, String> {
    let cfg = MtpConfig::from_json(&read_config(dir)?);
    component(dir, &mtp_param_specs(&cfg), mtp_hf_to_brain, cfg.to_json())
}

/// A Talker checkpoint as the loaders take it: a brain file (from
/// [`import_talker`]) as it is, or an HF checkpoint dir through [`talker_view`].
pub fn open_talker(path: &str) -> Result<WeightReader, String> {
    open(path, talker_view)
}

/// [`open_talker`] for the MTP.
pub fn open_mtp(path: &str) -> Result<WeightReader, String> {
    open(path, mtp_view)
}

fn open(path: &str, view: fn(&Path) -> Result<WeightReader, String>) -> Result<WeightReader, String> {
    if Path::new(path).is_dir() {
        view(Path::new(path)).map_err(|e| format!("{path}: {e}"))
    } else {
        WeightReader::open(path).map_err(|e| format!("{path}: {e}"))
    }
}

/// [`open_talker`], every tensor read as f32, for the eager loaders.
/// Panics naming `path` when it cannot be read, as `checkpoint::load` does.
pub fn load_talker(path: &str) -> checkpoint::Container {
    open_talker(path).and_then(|r| checkpoint::load_reader(&r)).unwrap_or_else(|e| panic!("{e}"))
}

/// [`load_talker`] for the MTP.
pub fn load_mtp(path: &str) -> checkpoint::Container {
    open_mtp(path).and_then(|r| checkpoint::load_reader(&r)).unwrap_or_else(|e| panic!("{e}"))
}

/// A [`checkpoint::weightio::WeightReader::save_while`] outcome as an import's:
/// a stopped save answers the contract's cancelled error.
fn saved(result: std::io::Result<bool>) -> Result<(), String> {
    match result {
        Ok(true) => Ok(()),
        Ok(false) => Err(capability::failure::CANCELLED.to_string()),
        Err(e) => Err(format!("import: {e}")),
    }
}

/// Write the Talker decoder (+ text-conditioning tensors) of `<hf_dir>` to a
/// brain checkpoint at `out_path` - [`talker_view`], one tensor at a time,
/// asking `keep_going` before each (`Err(CANCELLED)` once it says no).
pub fn import_talker(hf_dir: &str, out_path: &str, keep_going: &mut dyn FnMut() -> bool) -> Result<(), String> {
    let view = talker_view(Path::new(hf_dir))?;
    saved(view.save_while(out_path, None, keep_going))?;
    eprintln!("imported Talker: {} tensors -> {out_path}", view.names().count());
    Ok(())
}

/// Write the MTP code-predictor of `<hf_dir>` to a brain checkpoint at
/// `out_path` - [`mtp_view`], one tensor at a time.
pub fn import_mtp(hf_dir: &str, out_path: &str, keep_going: &mut dyn FnMut() -> bool) -> Result<(), String> {
    let view = mtp_view(Path::new(hf_dir))?;
    saved(view.save_while(out_path, None, keep_going))?;
    eprintln!("imported MTP: {} tensors -> {out_path}", view.names().count());
    Ok(())
}

/// The four brain files [`import_all`] writes into its destination directory.
pub const TALKER_FILE: &str = "talker.safetensors";
pub const MTP_FILE: &str = "mtp.safetensors";
pub const CODEC_FILE: &str = "codec.safetensors";
pub const SPEAKER_FILE: &str = "speaker.safetensors";

/// Where the components of one Qwen3-TTS release are read from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sources {
    /// The HF checkpoint dir: the Talker, the MTP and (unless named apart) the
    /// speaker encoder.
    pub release: String,
    /// The speech tokenizer; its own directory when it ships separately.
    pub codec: String,
    /// The speaker encoder; absent from CustomVoice and VoiceDesign releases.
    pub speaker: String,
}

impl Sources {
    /// A release read from `release`, with the codec in its `speech_tokenizer`
    /// subdirectory and the speaker encoder inside the release itself unless
    /// `codec`/`speaker` name them apart.
    pub fn new(release: &str, codec: Option<&str>, speaker: Option<&str>) -> Sources {
        Sources {
            release: release.to_string(),
            codec: codec.map_or_else(|| format!("{release}/speech_tokenizer"), str::to_string),
            speaker: speaker.map_or_else(|| release.to_string(), str::to_string),
        }
    }
}

/// What [`import_all`] wrote.
#[derive(Debug)]
pub struct Imported {
    /// Why no speaker encoder was written (a CustomVoice/VoiceDesign release
    /// has none and cannot clone from reference audio); `None` when it was.
    pub speaker_skipped: Option<String>,
}

/// Import every component of `sources` as brain checkpoints into `dest`
/// ([`TALKER_FILE`], [`MTP_FILE`], [`CODEC_FILE`], [`SPEAKER_FILE`]).
///
/// `keep_going` is asked before each tensor of the Talker and the MTP and
/// before each remaining component; once it says no the import answers
/// `Err(CANCELLED)`.
pub fn import_all(sources: &Sources, dest: &str, keep_going: &mut dyn FnMut() -> bool) -> Result<Imported, String> {
    std::fs::create_dir_all(dest).map_err(|e| format!("create {dest}: {e}"))?;
    let at = |file: &str| format!("{dest}/{file}");
    let component = |e: String, what: &str| if e == capability::failure::CANCELLED { e } else { format!("import {what}: {e}") };
    import_talker(&sources.release, &at(TALKER_FILE), keep_going).map_err(|e| component(e, "talker"))?;
    import_mtp(&sources.release, &at(MTP_FILE), keep_going).map_err(|e| component(e, "mtp"))?;
    if !keep_going() {
        return Err(capability::failure::CANCELLED.to_string());
    }
    mimi::import::import(&sources.codec, &at(CODEC_FILE)).map_err(|e| format!("import codec: {e}"))?;
    if !keep_going() {
        return Err(capability::failure::CANCELLED.to_string());
    }
    let speaker_skipped = ecapatdnn::import::import(&sources.speaker, &at(SPEAKER_FILE)).err();
    Ok(Imported { speaker_skipped })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn talker_name_mapping() {
        assert_eq!(
            talker_hf_to_brain("talker.model.codec_embedding.weight").unwrap(),
            "tok.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.codec_head.weight").unwrap(),
            "lm_head.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.model.norm.weight").unwrap(),
            "norm.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.model.text_embedding.weight").unwrap(),
            "text_embedding.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.text_projection.linear_fc1.weight").unwrap(),
            "text_projection.fc1.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.model.layers.5.self_attn.q_proj.weight").unwrap(),
            "blocks.5.attn.wq.weight"
        );
        assert_eq!(
            talker_hf_to_brain("talker.model.layers.27.mlp.down_proj.weight").unwrap(),
            "blocks.27.mlp.down.weight"
        );
        // Non-talker tensors are dropped.
        assert_eq!(
            talker_hf_to_brain("talker.code_predictor.model.norm.weight"),
            None
        );
        assert_eq!(talker_hf_to_brain("speaker_encoder.foo"), None);
    }

    #[test]
    fn mtp_name_mapping() {
        assert_eq!(
            mtp_hf_to_brain("talker.code_predictor.model.norm.weight").unwrap(),
            "norm.weight"
        );
        assert_eq!(
            mtp_hf_to_brain("talker.code_predictor.model.codec_embedding.0.weight").unwrap(),
            "codec_embedding.0.weight"
        );
        assert_eq!(
            mtp_hf_to_brain("talker.code_predictor.lm_head.14.weight").unwrap(),
            "lm_head.14.weight"
        );
        assert_eq!(
            mtp_hf_to_brain("talker.code_predictor.model.layers.3.self_attn.k_norm.weight")
                .unwrap(),
            "blocks.3.attn.k_norm.weight"
        );
        assert_eq!(mtp_hf_to_brain("talker.model.norm.weight"), None);
    }

    /// Every brain tensor of both components maps to an upstream name and back
    /// to itself, so a trained tensor is exported under the name a reader of the
    /// release looks up.
    #[test]
    fn the_name_table_reads_both_ways() {
        let tc = TalkerConfig::from_json(&serde_json::json!({"talker_config": {}}));
        let mc = MtpConfig { embedding_dim: 2048, ..MtpConfig::from_json(&serde_json::json!({"talker_config": {}})) };
        for (brain, _) in talker_param_specs(&tc) {
            let hf = talker_brain_to_hf(&brain).unwrap_or_else(|| panic!("{brain} has no upstream name"));
            assert!(hf.starts_with("talker.") && !hf.starts_with("talker.code_predictor."), "{hf}");
            assert_eq!(talker_hf_to_brain(&hf).as_deref(), Some(brain.as_str()), "{hf}");
        }
        for (brain, _) in mtp_param_specs(&mc) {
            let hf = mtp_brain_to_hf(&brain).unwrap_or_else(|| panic!("{brain} has no upstream name"));
            assert!(hf.starts_with("talker.code_predictor."), "{hf}");
            assert_eq!(mtp_hf_to_brain(&hf).as_deref(), Some(brain.as_str()), "{hf}");
        }
        assert_eq!(talker_hf_to_brain("talker.model.layers.x.mlp.up_proj.weight"), None);
        assert_eq!(mtp_hf_to_brain("talker.code_predictor.lm_head.x.weight"), None);
    }

    #[test]
    fn param_spec_counts() {
        // Real Talker: 28 layers × 11 + tok + norm + lm_head + text_embedding + 4
        // text_projection.
        let tc = TalkerConfig::from_json(&serde_json::json!({"talker_config": {}}));
        let ts = talker_param_specs(&tc);
        assert_eq!(ts.len(), 28 * 11 + 2 + 1 + 1 + 4);
        // Real MTP: 5 layers × 11 + norm + 15 codec_embedding + 15 lm_head.
        let mc = MtpConfig {
            n_layers: 5,
            num_code_groups: 16,
            ..MtpConfig::tiny()
        };
        let ms = mtp_param_specs(&mc);
        assert_eq!(ms.len(), 5 * 11 + 1 + 15 + 15);
    }

    #[test]
    fn streaming_import_talker_and_mtp_from_one_shared_checkpoint() {
        // A synthetic HF checkpoint dir holding BOTH talker.* and
        // talker.code_predictor.* tensors in one model.safetensors (mirrors the
        // real Qwen3-TTS layout) — each import must pick out only its own
        // tensors, streaming, without cross-contamination.
        let pid = std::process::id();
        let dir = std::env::temp_dir().join(format!("tts-import-src-{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        let config = serde_json::json!({
            "talker_config": {
                "num_hidden_layers": 2, "hidden_size": 16, "head_dim": 8,
                "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 32,
                "vocab_size": 23, "text_hidden_size": 20, "text_vocab_size": 29,
                "code_predictor_config": {
                    "num_hidden_layers": 2, "hidden_size": 16, "head_dim": 8,
                    "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 32,
                    "vocab_size": 23, "num_code_groups": 4,
                },
            },
        });
        std::fs::write(dir.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();

        let root = &config;
        let tc = TalkerConfig::from_json(root);
        let mc = MtpConfig::from_json(root);
        let talker_specs = talker_param_specs(&tc);
        let mtp_specs = mtp_param_specs(&mc);

        // Every tensor both imports need, keyed by its brain name (namespaced so
        // the two never collide, matching how the real checkpoint co-resides).
        let mut plan: Vec<(String, Vec<u64>)> = Vec::new();
        let mut expect_talker: HashMap<String, Vec<f32>> = HashMap::new();
        let mut expect_mtp: HashMap<String, Vec<f32>> = HashMap::new();
        for (i, (name, numel)) in talker_specs.iter().enumerate() {
            let hf = talker_brain_to_hf(name).unwrap();
            let data: Vec<f32> = (0..*numel).map(|j| (i * 10_000 + j) as f32 * 0.001).collect();
            plan.push((hf.clone(), vec![*numel as u64]));
            expect_talker.insert(name.clone(), data);
        }
        for (i, (name, numel)) in mtp_specs.iter().enumerate() {
            let hf = mtp_brain_to_hf(name).unwrap();
            let data: Vec<f32> = (0..*numel).map(|j| (i * 20_000 + j) as f32 * 0.001 + 5.0).collect();
            plan.push((hf.clone(), vec![*numel as u64]));
            expect_mtp.insert(name.clone(), data);
        }

        let mut w = checkpoint::weightio::StWriter::create(
            dir.join("model.safetensors").to_str().unwrap(),
            &plan,
            &serde_json::Value::Null,
            None,
        )
        .unwrap();
        for (i, (name, numel)) in talker_specs.iter().enumerate() {
            let hf = talker_brain_to_hf(name).unwrap();
            let data: Vec<f32> = (0..*numel).map(|j| (i * 10_000 + j) as f32 * 0.001).collect();
            w.write(&hf, &data).unwrap();
        }
        for (i, (name, numel)) in mtp_specs.iter().enumerate() {
            let hf = mtp_brain_to_hf(name).unwrap();
            let data: Vec<f32> = (0..*numel).map(|j| (i * 20_000 + j) as f32 * 0.001 + 5.0).collect();
            w.write(&hf, &data).unwrap();
        }
        w.finish().unwrap();

        let talker_out = std::env::temp_dir().join(format!("tts-import-talker-{pid}.safetensors"));
        let mtp_out = std::env::temp_dir().join(format!("tts-import-mtp-{pid}.safetensors"));
        import_talker(dir.to_str().unwrap(), talker_out.to_str().unwrap(), &mut || true).unwrap();
        import_mtp(dir.to_str().unwrap(), mtp_out.to_str().unwrap(), &mut || true).unwrap();

        // A token that fires mid-conversion stops each import at the next tensor.
        {
            use capability::{Invocation, Outcome, Progress};
            use capability_mock::cancellation::{certify, Bound};
            let stopped = std::env::temp_dir().join(format!("tts-import-stopped-{pid}.safetensors"));
            let run = |inv: &Invocation, progress: &mut dyn FnMut(Progress)| {
                inv.cancel.check()?;
                import_talker(dir.to_str().unwrap(), stopped.to_str().unwrap(), &mut inv.keep_going(progress, 0, "tensor"))?;
                import_mtp(dir.to_str().unwrap(), stopped.to_str().unwrap(), &mut inv.keep_going(progress, 0, "tensor"))?;
                Ok(Outcome::new())
            };
            certify("qwen3tts import", &Invocation::new, &run, Bound::ONE_STEP);
            let mut asks = 0;
            let err = import_mtp(dir.to_str().unwrap(), stopped.to_str().unwrap(), &mut || {
                asks += 1;
                asks < 3
            });
            assert_eq!(err.unwrap_err(), capability::failure::CANCELLED);
            let leftovers = std::fs::read_dir(std::env::temp_dir()).unwrap().filter_map(|e| e.ok()).filter(|e| e.file_name().to_string_lossy().starts_with(&format!("tts-import-stopped-{pid}.safetensors."))).count();
            assert_eq!(leftovers, 0, "a stopped import leaves no temporary file");
            std::fs::remove_file(&stopped).ok();
        }

        let tr = checkpoint::weightio::WeightReader::open(talker_out.to_str().unwrap()).unwrap();
        assert_eq!(tr.names().count(), expect_talker.len());
        for (name, data) in &expect_talker {
            assert_eq!(tr.tensor(name).unwrap(), *data, "talker {name}");
        }

        let mr = checkpoint::weightio::WeightReader::open(mtp_out.to_str().unwrap()).unwrap();
        assert_eq!(mr.names().count(), expect_mtp.len());
        for (name, data) in &expect_mtp {
            assert_eq!(mr.tensor(name).unwrap(), *data, "mtp {name}");
        }

        // Served as downloaded: the views read what the imports wrote, and
        // opening them writes nothing beside the checkpoint.
        for (view, file) in [(load_talker(dir.to_str().unwrap()), &talker_out), (load_mtp(dir.to_str().unwrap()), &mtp_out)] {
            let written = checkpoint::load(file.to_str().unwrap());
            assert_eq!(view.header, written.header);
            assert_eq!(view.by_role(""), written.by_role(""));
        }
        let mut listed: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name()).collect();
        listed.sort();
        assert_eq!(listed, ["config.json", "model.safetensors"]);

        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_file(&talker_out).ok();
        std::fs::remove_file(&mtp_out).ok();
    }

    #[test]
    fn sources_default_the_codec_and_speaker_inside_the_release() {
        let s = Sources::new("/r", None, None);
        assert_eq!((s.codec.as_str(), s.speaker.as_str()), ("/r/speech_tokenizer", "/r"));
        let s = Sources::new("/r", Some("/c"), Some("/s"));
        assert_eq!((s.codec.as_str(), s.speaker.as_str()), ("/c", "/s"));
    }

    #[test]
    fn importing_a_missing_release_names_the_failing_component() {
        let dest = std::env::temp_dir().join(format!("tts-import-missing-{}", std::process::id()));
        let err = import_all(&Sources::new("/nonexistent/release", None, None), dest.to_str().unwrap(), &mut || true).unwrap_err();
        assert!(err.starts_with("import talker"), "got: {err}");
        std::fs::remove_dir_all(&dest).ok();
    }
}
