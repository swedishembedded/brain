// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Real-checkpoint import** - the shipped `ggml-org/DeepSeek-OCR-GGUF` pair
//! turned into the two [`checkpoint::TensorSource`]s
//! [`crate::DeepseekOcr::new_with_prompt`] takes, plus the config and the
//! tokenizer that come off the same two files.
//!
//! This is production code, not test glue. It was promoted out of
//! `tests/common/real_vision.rs` (which is now a thin wrapper over it) the
//! moment a served path needed the same three steps, because two copies of
//! "which tensors, under which names, from which file" is exactly how a served
//! model and its own parity test end up disagreeing about what they ran.
//!
//! ## The checkpoint is two files, and the decoder needs a third
//!
//! * `mmproj-DeepSeek-OCR-Q8_0.gguf` (448 MB) - the SAM tower, the CLIP tower,
//!   the projector and the two learned image-block rows. [`encoder_weights`]
//!   dequantizes it in ONE `gguf::import::to_map` pass, so the two-way coverage
//!   check still runs over all 476 source tensors.
//! * `DeepSeek-OCR-Q8_0.gguf` (3.1 GB) - the 2.9 B-parameter decoder, and the
//!   tokenizer KV [`tokenizer`] reads.
//! * `DeepSeek-OCR-brain-fp32.safetensors` (11.7 GB) - the decoder's fp32
//!   expansion, **derived**, cached beside the pair, and built on first use by
//!   [`expand_lm`]. It is not a convenience: a `WeightReader` over it streams
//!   one tensor at a time into the device buffers, whereas
//!   `deepseekv2::import::import_map` would materialise the whole 11.7 GB as a
//!   host `HashMap` *in addition to* the same bytes in the parameter store.
//!   On a 30 GiB box that is the difference between building and being killed.
//!
//! ## The config is derived from the files, then checked against the preset
//!
//! [`config`] reads the SAM/CLIP shapes off the mmproj's own KV + tensor shapes
//! and the decoder's off the LM header, then refuses anything that is not
//! [`DeepseekOcrConfig::deepseek_ocr`]. Deriving *and* comparing is the point:
//! deriving alone would silently serve a re-quantized checkpoint of a different
//! shape, and hardcoding alone would not notice one at all.
//!
//! Every entry point returns `Result` - a missing or wrong checkpoint is an
//! error message, never a panic.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use checkpoint::gguf::MmapGguf;
use checkpoint::weightio::WeightReader;
use clip::config::ClipVisionConfig;
use data::qwen_tokenizer::QwenBpe;

use crate::config::{DeepseekOcrConfig, PROJECTOR_B, PROJECTOR_W};

/// The model-store repo the two GGUFs ship in.
pub const STORE: &str = "ggml-org/DeepSeek-OCR-GGUF";
/// The vision half: SAM tower + CLIP tower + projector + the two learned rows.
pub const MMPROJ: &str = "mmproj-DeepSeek-OCR-Q8_0.gguf";
/// The language half: the 2.9 B-parameter decoder, and the tokenizer KV.
pub const LM: &str = "DeepSeek-OCR-Q8_0.gguf";
/// The LM's fp32 expansion, cached beside it. Derived, never shipped.
pub const EXPANDED: &str = "DeepSeek-OCR-brain-fp32.safetensors";

/// The three paths, resolved and existence-checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Files {
    /// The directory holding both GGUFs (and, once built, the expansion).
    pub dir: PathBuf,
    pub mmproj: PathBuf,
    pub lm: PathBuf,
    /// Where [`expand_lm`] caches the fp32 expansion. May not exist yet.
    pub expanded: PathBuf,
}

impl Files {
    /// Resolve the checkpoint layout under `dir`.
    ///
    /// Only the two SHIPPED files are required to exist; the expansion is
    /// derived on demand, so its absence is not an error here.
    pub fn locate(dir: impl AsRef<Path>) -> Result<Files, String> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(format!("{}: not a directory (expected the one holding {MMPROJ} and {LM})", dir.display()));
        }
        let (mmproj, lm) = (dir.join(MMPROJ), dir.join(LM));
        for p in [&mmproj, &lm] {
            if !p.exists() {
                return Err(format!("{}: missing (the {STORE} checkpoint is two files: {MMPROJ} and {LM})", p.display()));
            }
        }
        let expanded = dir.join(EXPANDED);
        Ok(Files { dir, mmproj, lm, expanded })
    }
}

/// UTF-8 path or a message naming the offender - `MmapGguf`/`WeightReader` both
/// take `&str`, and a lossy conversion would open the wrong file.
fn utf8(p: &Path) -> Result<&str, String> {
    p.to_str().ok_or_else(|| format!("{}: path is not valid UTF-8", p.display()))
}

/// The encoder's whole init map, straight from an open mmproj.
///
/// The SAM tower keeps its `vision.sam.*` names, the CLIP tower's
/// `vision.clip.` prefix is stripped to the bare leaves `ClipVision` wants, and
/// the projector is renamed from the loader's `vision.projector.fc.*` to the
/// composite's [`PROJECTOR_W`]/[`PROJECTOR_B`]. `vision.image_newline` and
/// `vision.view_separator` already carry the names
/// [`DeepseekOcrConfig::glue_param_list`] declares, so they pass through
/// untouched.
///
/// ONE `to_map` pass over the file, so `gguf::import`'s two-way coverage check
/// still sees all 476 source tensors - splitting it per stage would check each
/// half against a subset and stop noticing an unclassified tensor.
pub fn encoder_weights(mg: &MmapGguf) -> Result<HashMap<String, Vec<f32>>, String> {
    let full = gguf::deepseek_ocr_vision::config_from_gguf(mg)?;
    let raw = gguf::import::to_map(mg, &full.param_list(), &|n| gguf::deepseek_ocr_vision::classify(n, &full), "deepseekocr-encoder")?;
    let mut init = HashMap::with_capacity(raw.len());
    for (name, data) in raw {
        let brain = match name.as_str() {
            "vision.projector.fc.weight" => PROJECTOR_W.to_string(),
            "vision.projector.fc.bias" => PROJECTOR_B.to_string(),
            n => match n.strip_prefix("vision.clip.") {
                Some(leaf) => leaf.to_string(),
                None => n.to_string(),
            },
        };
        if init.insert(brain.clone(), data).is_some() {
            return Err(format!("deepseekocr-encoder import: duplicate init name {brain}"));
        }
    }
    Ok(init)
}

/// [`encoder_weights`] from a path, opening and dropping the mmap itself.
pub fn encoder_weights_from(mmproj: &Path) -> Result<HashMap<String, Vec<f32>>, String> {
    let mg = MmapGguf::open(utf8(mmproj)?)?;
    encoder_weights(&mg)
}

/// The LM's fp32 expansion, converting it on first use.
///
/// Returns the path to use as a [`WeightReader`] source. The conversion streams
/// one tensor at a time (`deepseekv2::import::import_file`), so it costs disk,
/// not 11.7 GB of RAM - but it costs minutes, which is why the result is cached
/// beside the checkpoint and why this prints what it is doing.
pub fn expand_lm(lm: &Path, expanded: &Path) -> Result<String, String> {
    let out = utf8(expanded)?.to_string();
    if expanded.exists() {
        return Ok(out);
    }
    eprintln!("brain: expanding {} -> {} (once, ~12 GB on disk)", lm.display(), expanded.display());
    let stats = deepseekv2::import::import_file(utf8(lm)?, &out, None)?;
    eprintln!("brain: deepseek-ocr decoder import: {stats}");
    Ok(out)
}

/// A streaming source for the decoder's weights, expanding the LM if needed.
pub fn decoder_reader(files: &Files) -> Result<WeightReader, String> {
    let path = expand_lm(&files.lm, &files.expanded)?;
    WeightReader::open(&path).map_err(|e| format!("{path}: {e}"))
}

/// The composite's config, **derived** from the two files and then checked
/// against the documented preset.
///
/// `block_size` is `DeepseekV2Config`'s run-parameter sequence length; it is
/// inert for inference (`DeepseekOcr::new_with_prompt` takes the real `seq`
/// separately) and only has to be the same value on both sides of the
/// comparison below.
pub fn config(files: &Files, block_size: u32) -> Result<DeepseekOcrConfig, String> {
    let mg = MmapGguf::open(utf8(&files.mmproj)?)?;
    let vision = gguf::deepseek_ocr_vision::config_from_gguf(&mg)?;
    let cfg = DeepseekOcrConfig {
        sam: sam1::import::config_from_gguf(&mg)?,
        clip: ClipVisionConfig::from_gguf(&vision),
        decoder: deepseekv2::import::config_from_file(utf8(&files.lm)?, block_size)?,
        // No real-scale analogue; `check_real_scale_shaped` refuses it.
        patch_bypass: false,
    };
    drop(mg);
    let want = DeepseekOcrConfig::deepseek_ocr(block_size);
    if cfg != want {
        return Err(format!(
            "{}: the shipped checkpoint's shape is not DeepSeek-OCR's documented preset \
             (derived token grid {:?}, clip width {}, decoder {} layers x d_model {}; \
             want {:?}, {}, {} x {})",
            files.dir.display(),
            cfg.token_grid(),
            cfg.clip_width(),
            cfg.decoder.n_layers(),
            cfg.decoder.d_model(),
            want.token_grid(),
            want.clip_width(),
            want.decoder.n_layers(),
            want.decoder.d_model(),
        ));
    }
    Ok(cfg)
}

/// The LM's own tokenizer, out of the LM GGUF's `tokenizer.ggml.*` KV.
pub fn tokenizer(files: &Files) -> Result<QwenBpe, String> {
    crate::prompt::tokenizer_from_gguf(utf8(&files.lm)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `locate` names the file that is missing, rather than failing later inside
    /// an mmap with an errno.
    #[test]
    fn locate_reports_the_missing_half_by_name() {
        let e = Files::locate("/definitely/not/a/deepseek/dir").unwrap_err();
        assert!(e.contains("not a directory"), "{e}");

        let tmp = std::env::temp_dir().join(format!("brain-deepseekocr-import-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).expect("tmp dir");
        let e = Files::locate(&tmp).unwrap_err();
        assert!(e.contains(MMPROJ), "the missing file must be named: {e}");
        // ... and with only the mmproj present it is the LM that is reported.
        std::fs::write(tmp.join(MMPROJ), b"").expect("touch mmproj");
        let e = Files::locate(&tmp).unwrap_err();
        assert!(e.contains(LM) && !e.contains("not a directory"), "{e}");
        // Both present: the expansion is derived, so its absence is not an error.
        std::fs::write(tmp.join(LM), b"").expect("touch lm");
        let f = Files::locate(&tmp).expect("both shipped files present");
        assert_eq!(f.expanded, tmp.join(EXPANDED));
        assert!(!f.expanded.exists(), "the expansion is built on demand, not required");
        std::fs::remove_dir_all(&tmp).ok();
    }
}
