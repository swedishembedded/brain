// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-checkpoint import: the upstream `deepseek-ai/DeepSeek-OCR-2`
//! safetensors release (read in place, see [`crate::hf`]) or a community
//! Q8_0 pair (`mmproj-deepseek-ocr-2-q8_0.gguf` + `deepseek-ocr-2-q8_0.gguf`),
//! turned into the two [`checkpoint::TensorSource`]s [`crate::model::DeepseekOcr2::new_on`]
//! takes, plus the derived [`crate::config::DeepseekOcr2VisionConfig`].
//!
//! ## Two files in, two derived files out
//!
//! Neither shipped file is read into a host map directly. `expand_vision`
//! runs `gguf::deepseekocr2_vision::import` once and caches the result beside
//! the mmproj; `expand_lm` does the same for the decoder via
//! `deepseek2::import::import_file` - the exact function [`crates/deepseek2ocr`]'s
//! own decoder expansion already calls, since M0 established the decoder is
//! untouched between the two models. Both derived files are opened with
//! [`checkpoint::weightio::WeightReader`], which streams one tensor at a time
//! rather than materialising either tower as a host `HashMap`, so building the
//! composite never needs the sum of both towers' bytes resident twice over.
//!
//! ## The vision config is derived, then checked against the documented shape
//!
//! [`vision_config`] reads every tensor-shape-backed field off the mmproj and
//! rejects anything that does not match [`Qwen2EncoderConfig::deepseek_ocr2`].
//! Deriving alone would silently serve a re-quantized checkpoint of a
//! different shape, and a hardcoded config alone would not notice one at all.
//! `rope_theta` is the one field the file cannot state (no RoPE KV key exists
//! for this projector type - M0's ledger), so it is excluded from the
//! comparison and always taken from the documented preset.

use std::path::{Path, PathBuf};

use checkpoint::gguf::MmapGguf;
use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::weightio::WeightReader;
use deepseek2::DeepseekV2Config;
use sam1::SamViTConfig;

use crate::config::{DeepseekOcr2VisionConfig, Qwen2EncoderConfig};

/// The vision half: SAM tower, the Qwen2 resampler, the projector and the
/// separator.
pub const MMPROJ: &str = "mmproj-deepseek-ocr-2-q8_0.gguf";
/// The language half: the unchanged DeepSeek-V2-family decoder, and its
/// tokenizer.
pub const LM: &str = "deepseek-ocr-2-q8_0.gguf";
/// The vision half's fp32 expansion, cached beside it. Small (~1 GB) and
/// cheap to rebuild, but kept on the same derived-file convention as
/// [`LM_EXPANDED`] rather than a special-cased in-memory path, so the
/// composite's constructor sees one shape (`&dyn TensorSource`) for both
/// halves.
pub const VISION_EXPANDED: &str = "mmproj-deepseek-ocr-2-brain-fp32.safetensors";
/// The decoder's fp32 expansion, cached beside it. Multi-gigabyte and slow to
/// rebuild - see `deepseek2::import::import_file`'s own doc for why this is
/// streamed rather than held as a host map.
pub const LM_EXPANDED: &str = "deepseek-ocr-2-brain-fp32.safetensors";

/// The upstream `transformers` release, the one served by default. No vendor
/// publishes a GGUF; community conversions of it also load.
pub const HF_STORE: &str = "deepseek-ai/DeepSeek-OCR-2";

/// Which of the two checkpoint shapes a directory holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// A community-converted Q8_0 pair, plus the fp32 expansions derived
    /// beside it on first use.
    GgufPair {
        mmproj: PathBuf,
        lm: PathBuf,
        /// Where [`expand_vision`] caches its output. May not exist yet.
        vision_expanded: PathBuf,
        /// Where [`expand_lm`] caches its output. May not exist yet.
        lm_expanded: PathBuf,
    },
    /// The upstream `transformers` release, read **in place**: `config.json`
    /// for the numbers no tensor shape carries, `tokenizer.json` for the
    /// vocabulary, and the safetensors streamed through [`crate::hf`]'s rename.
    /// No conversion step and no expansion on disk.
    Hf,
}

/// The checkpoint directory and the shape it turned out to have.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Files {
    /// The directory the checkpoint lives in.
    pub dir: PathBuf,
    pub layout: Layout,
}

impl Files {
    /// Resolve the checkpoint layout under `dir`, recognizing either release.
    ///
    /// The GGUF pair is the more specific shape (two exactly-named files); an
    /// HF checkpoint is a `config.json` plus model safetensors. Only SHIPPED
    /// files are required: both GGUF expansions are derived on demand.
    pub fn locate(dir: impl AsRef<Path>) -> Result<Files, String> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(format!("{}: not a directory (expected the one holding {MMPROJ} and {LM}, or an HF {HF_STORE} checkpoint)", dir.display()));
        }
        let (mmproj, lm) = (dir.join(MMPROJ), dir.join(LM));
        if mmproj.exists() || lm.exists() {
            for p in [&mmproj, &lm] {
                if !p.exists() {
                    return Err(format!("{}: missing (the checkpoint is two files: {MMPROJ} and {LM})", p.display()));
                }
            }
            return Ok(Files { layout: Layout::GgufPair { mmproj, lm, vision_expanded: dir.join(VISION_EXPANDED), lm_expanded: dir.join(LM_EXPANDED) }, dir });
        }
        if dir.join("config.json").is_file() && checkpoint::safetensors::has_model_weights(&dir) {
            return Ok(Files { dir, layout: Layout::Hf });
        }
        Err(format!("{}: holds neither the {MMPROJ} + {LM} pair nor an HF {HF_STORE} checkpoint (config.json + model safetensors)", dir.display()))
    }
}

/// UTF-8 path or a message naming the offender - `MmapGguf`/`WeightReader`
/// both take `&str`, and a lossy conversion would open the wrong file.
fn utf8(p: &Path) -> Result<&str, String> {
    p.to_str().ok_or_else(|| format!("{}: path is not valid UTF-8", p.display()))
}

/// The vision tower's fp32 expansion, building it on first use.
pub fn expand_vision(mmproj: &Path, out: &Path) -> Result<String, String> {
    let out_s = utf8(out)?.to_string();
    if out.exists() {
        return Ok(out_s);
    }
    let mg = MmapGguf::open(utf8(mmproj)?)?;
    let stats = gguf::deepseekocr2_vision::import(&mg, &out_s, None)?;
    eprintln!("brain: deepseek-ocr-2 vision import: {stats}");
    Ok(out_s)
}

/// The decoder's fp32 expansion, building it on first use. Streams one
/// tensor at a time (`deepseek2::import::import_file`), so it costs disk and
/// minutes, not host RAM - the reason the result is cached beside the
/// checkpoint rather than rebuilt per run.
pub fn expand_lm(lm: &Path, out: &Path) -> Result<String, String> {
    let out_s = utf8(out)?.to_string();
    if out.exists() {
        return Ok(out_s);
    }
    eprintln!("brain: expanding {} -> {} (once)", lm.display(), out.display());
    let stats = deepseek2::import::import_file(utf8(lm)?, &out_s, None)?;
    eprintln!("brain: deepseek-ocr-2 decoder import: {stats}");
    Ok(out_s)
}

fn hf_reader(dir: &Path) -> Result<WeightReader, String> {
    WeightReader::open_hf_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))
}

/// An identity plan over a reader that already carries brain's names, so both
/// layouts hand out the same source type and the same every-parameter check.
fn identity_source(reader: WeightReader, want: &[(String, usize)]) -> Result<RemapSource<'static>, String> {
    let plan = want.iter().map(|(n, _)| (n.clone(), Fetch::Whole(n.clone()))).collect();
    let src = RemapSource::owning(Box::new(reader), plan);
    src.validate(want)?;
    Ok(src)
}

/// A streaming source for the vision tower's weights under brain's names.
pub fn vision_reader(files: &Files) -> Result<RemapSource<'static>, String> {
    let cfg = vision_config(files, DeepseekV2Config::deepseek_ocr(1).shape.d_model)?;
    let want: Vec<(String, usize)> = cfg.sam.param_list().into_iter().chain(cfg.param_list()).collect();
    match &files.layout {
        Layout::GgufPair { mmproj, vision_expanded, .. } => {
            let path = expand_vision(mmproj, vision_expanded)?;
            identity_source(WeightReader::open(&path).map_err(|e| format!("{path}: {e}"))?, &want)
        }
        Layout::Hf => {
            let reader = hf_reader(&files.dir)?;
            let names: Vec<String> = reader.names().map(str::to_string).collect();
            let plan = crate::hf::encoder_plan(&names, &cfg)?;
            let src = RemapSource::owning(Box::new(reader), plan);
            src.validate(&want).map_err(|e| format!("deepseek-ocr-2 hf import: {e}"))?;
            Ok(src)
        }
    }
}

/// A streaming source for the decoder's weights under brain's names.
pub fn decoder_reader(files: &Files) -> Result<RemapSource<'static>, String> {
    let dec = DeepseekV2Config::deepseek_ocr(1);
    match &files.layout {
        Layout::GgufPair { lm, lm_expanded, .. } => {
            let path = expand_lm(lm, lm_expanded)?;
            identity_source(WeightReader::open(&path).map_err(|e| format!("{path}: {e}"))?, &dec.param_list())
        }
        Layout::Hf => {
            let reader = hf_reader(&files.dir)?;
            let names: Vec<String> = reader.names().map(str::to_string).collect();
            let plan = deepseek2::hf::plan(&names, &dec).map_err(|e| format!("deepseek-ocr-2 hf import: {e}"))?;
            let src = RemapSource::owning(Box::new(reader), plan);
            src.validate(&dec.param_list()).map_err(|e| format!("deepseek-ocr-2 hf import: {e}"))?;
            Ok(src)
        }
    }
}

/// The tokenizer: the HF release carries `tokenizer.json` (vocabulary, merges
/// and specials, pre-tokenizer included); the GGUF layout only the LM's
/// `tokenizer.ggml.*` KV.
pub fn tokenizer(files: &Files) -> Result<data::qwen_tokenizer::QwenBpe, String> {
    match &files.layout {
        Layout::GgufPair { lm, .. } => crate::prompt::tokenizer_from_gguf(utf8(lm)?),
        Layout::Hf => data::qwen_tokenizer::QwenBpe::from_dir(utf8(&files.dir)?),
    }
}

/// The vision config: **derived** from the checkpoint's own shapes, then
/// checked against the documented preset (`rope_theta` excepted for the GGUF
/// pair - see this module's header).
pub fn vision_config(files: &Files, decoder_hidden: u32) -> Result<DeepseekOcr2VisionConfig, String> {
    match &files.layout {
        Layout::GgufPair { mmproj, .. } => gguf_vision_config(mmproj, decoder_hidden),
        Layout::Hf => configs(files, 1).map(|(v, _)| v),
    }
}

/// Both configs of the HF layout, derived and compared with the preset; for the
/// GGUF pair the vision config is derived from the mmproj and the decoder is the
/// preset (its GGUF import has its own shape check).
pub fn configs(files: &Files, block_size: u32) -> Result<(DeepseekOcr2VisionConfig, DeepseekV2Config), String> {
    match &files.layout {
        Layout::GgufPair { mmproj, .. } => {
            let dec = DeepseekV2Config::deepseek_ocr(block_size);
            Ok((gguf_vision_config(mmproj, dec.shape.d_model)?, dec))
        }
        Layout::Hf => {
            let reader = hf_reader(&files.dir)?;
            let path = files.dir.join("config.json");
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let json = serde_json::from_slice(&bytes).map_err(|e| format!("{}: unparseable config.json: {e}", path.display()))?;
            crate::hf::config_from_shapes(&deepseek2::hf::shapes(&reader), &json, block_size)
        }
    }
}

fn gguf_vision_config(mmproj: &Path, decoder_hidden: u32) -> Result<DeepseekOcr2VisionConfig, String> {
    let mg = MmapGguf::open(utf8(mmproj)?)?;
    let full = gguf::deepseekocr2_vision::config_from_gguf(&mg)?;
    drop(mg);

    let want = Qwen2EncoderConfig::deepseek_ocr2();
    let encoder = Qwen2EncoderConfig {
        d_model: full.encoder.d_model,
        n_layers: full.encoder.n_layers,
        n_heads: full.encoder.n_heads,
        n_kv_heads: full.encoder.n_kv_heads,
        ffn_hidden: full.encoder.ffn_hidden,
        rms_eps: full.encoder.layer_norm_eps,
        rope_theta: want.rope_theta,
        n_query_local: full.encoder.n_query_local,
        n_query_global: full.encoder.n_query_global,
    };
    if encoder != want {
        return Err(format!(
            "{}: the shipped checkpoint's encoder shape is not the documented preset (derived {encoder:?}, want {want:?})",
            mmproj.display()
        ));
    }
    if full.projection_dim != decoder_hidden {
        return Err(format!(
            "{}: the projector's output width ({}) does not equal the decoder's d_model ({decoder_hidden})",
            mmproj.display(),
            full.projection_dim
        ));
    }

    let cfg = DeepseekOcr2VisionConfig { sam: SamViTConfig::from(&full.sam), encoder, decoder_hidden, lora: None };
    cfg.check();
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `locate` names the file that is missing, rather than failing later
    /// inside an mmap with an errno.
    #[test]
    fn locate_reports_the_missing_half_by_name() {
        let e = Files::locate("/definitely/not/a/deepseek-ocr2/dir").unwrap_err();
        assert!(e.contains("not a directory"), "{e}");

        let tmp = std::env::temp_dir().join(format!("brain-deepseekocr2-import-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("tmp dir");
        let e = Files::locate(&tmp).unwrap_err();
        assert!(e.contains(MMPROJ), "the missing file must be named: {e}");
        std::fs::write(tmp.join(MMPROJ), b"").expect("touch mmproj");
        let e = Files::locate(&tmp).unwrap_err();
        assert!(e.contains(LM) && !e.contains("not a directory"), "{e}");
        std::fs::write(tmp.join(LM), b"").expect("touch lm");
        let f = Files::locate(&tmp).expect("both shipped files present");
        let Layout::GgufPair { vision_expanded, lm_expanded, .. } = &f.layout else { panic!("two GGUFs are the pair layout") };
        assert_eq!(vision_expanded, &tmp.join(VISION_EXPANDED));
        assert_eq!(lm_expanded, &tmp.join(LM_EXPANDED));
        assert!(!vision_expanded.exists() && !lm_expanded.exists(), "both expansions are built on demand, not required");
        std::fs::remove_dir_all(&tmp).ok();
    }
}
