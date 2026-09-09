// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Real-checkpoint import** - a published DeepSeek-OCR checkpoint turned
//! into the two [`checkpoint::TensorSource`]s
//! [`crate::DeepseekOcr::new_with_prompt`] takes, plus the config and the
//! tokenizer that come off the same files.
//!
//! ## Two published shapes, one load path
//!
//! [`Files::locate`] recognizes BOTH releases of this model and [`Layout`]
//! records which one it found; everything below routes on that, so
//! `caps::Session::load` does not know or care which it got:
//!
//! * the pre-converted `ggml-org/DeepSeek-OCR-GGUF` pair, described below;
//! * the upstream `deepseek-ai/DeepSeek-OCR` `transformers` release, read in
//!   place through [`crate::hf`]'s rename - no conversion step, and no fp32
//!   expansion on disk at all, because its BF16 tensors stream straight
//!   through a [`checkpoint::remap::RemapSource`].
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
//!   `deepseek2::import::import_map` would materialise the whole 11.7 GB as a
//!   host `HashMap` *in addition to* the same bytes in the parameter store.
//!   On a 30 GiB box that is the difference between building and being killed.
//!
//! ## The config is derived from the files, then checked against the preset
//!
//! For the GGUF pair [`config`] reads the SAM/CLIP shapes off the mmproj's own
//! KV + tensor shapes and the decoder's off the LM header; for the HF layout
//! `crate::hf::config_from_shapes` reads them off the safetensors' own shapes.
//! Either way it then refuses anything that is not
//! [`DeepseekOcrConfig::deepseek_ocr`]. Deriving *and* comparing is the point:
//! deriving alone would silently serve a re-quantized checkpoint of a different
//! shape, and hardcoding alone would not notice one at all.
//!
//! Every entry point returns `Result` - a missing or wrong checkpoint is an
//! error message, never a panic.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use checkpoint::gguf::MmapGguf;
use checkpoint::remap::{Fetch, RemapSource};
use checkpoint::weightio::WeightReader;
use clip::config::ClipVisionConfig;
use data::qwen_tokenizer::QwenBpe;

use crate::config::{DeepseekOcrConfig, PROJECTOR_B, PROJECTOR_W};

/// The model-store repo the two GGUFs ship in.
pub const STORE: &str = "ggml-org/DeepSeek-OCR-GGUF";
/// The upstream `transformers` release the GGUF pair is a conversion of.
pub const HF_STORE: &str = "deepseek-ai/DeepSeek-OCR";
/// The vision half: SAM tower + CLIP tower + projector + the two learned rows.
pub const MMPROJ: &str = "mmproj-DeepSeek-OCR-Q8_0.gguf";
/// The language half: the 2.9 B-parameter decoder, and the tokenizer KV.
pub const LM: &str = "DeepSeek-OCR-Q8_0.gguf";
/// The LM's fp32 expansion, cached beside it. Derived, never shipped.
pub const EXPANDED: &str = "DeepSeek-OCR-brain-fp32.safetensors";

/// Which of the two published checkpoint shapes a directory holds.
///
/// Both are real releases of the same model, so both are first-class here
/// rather than one being converted into the other: the GGUF pair is what
/// llama.cpp ships and what this crate loaded first, and the `transformers`
/// release is the upstream original.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    /// The pre-converted [`STORE`] pair, plus the fp32 expansion
    /// [`expand_lm`] derives beside it.
    GgufPair {
        mmproj: PathBuf,
        lm: PathBuf,
        /// Where [`expand_lm`] caches the fp32 expansion. May not exist yet.
        expanded: PathBuf,
    },
    /// The upstream `deepseek-ai/DeepSeek-OCR` `transformers` release, read
    /// **in place**: `config.json` for the numbers no tensor shape carries,
    /// `tokenizer.json` for the vocabulary, and the safetensors themselves
    /// (single file or shard set) streamed through [`crate::hf`]'s rename.
    /// No conversion step and no fp32 expansion on disk.
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
    /// The GGUF pair is checked first because it is the more specific shape
    /// (two exactly-named files); an HF checkpoint is recognized by the
    /// `config.json` + safetensors pair every `transformers` release has.
    /// Only SHIPPED files are required to exist - the GGUF path's expansion
    /// is derived on demand, so its absence is not an error here.
    pub fn locate(dir: impl AsRef<Path>) -> Result<Files, String> {
        let dir = dir.as_ref().to_path_buf();
        if !dir.is_dir() {
            return Err(format!("{}: not a directory (expected the one holding {MMPROJ} and {LM}, or an HF {HF_STORE} checkpoint)", dir.display()));
        }
        let (mmproj, lm) = (dir.join(MMPROJ), dir.join(LM));
        if mmproj.exists() || lm.exists() {
            for p in [&mmproj, &lm] {
                if !p.exists() {
                    return Err(format!("{}: missing (the {STORE} checkpoint is two files: {MMPROJ} and {LM})", p.display()));
                }
            }
            let expanded = dir.join(EXPANDED);
            return Ok(Files { dir, layout: Layout::GgufPair { mmproj, lm, expanded } });
        }
        if dir.join("config.json").is_file() && checkpoint::safetensors::has_model_weights(&dir) {
            return Ok(Files { dir, layout: Layout::Hf });
        }
        Err(format!(
            "{}: holds neither the {STORE} pair ({MMPROJ} + {LM}) nor an HF {HF_STORE} checkpoint (config.json + model safetensors)",
            dir.display()
        ))
    }

    /// The mmproj path, for the GGUF layout only.
    pub fn mmproj(&self) -> Option<&Path> {
        match &self.layout {
            Layout::GgufPair { mmproj, .. } => Some(mmproj),
            Layout::Hf => None,
        }
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
/// one tensor at a time (`deepseek2::import::import_file`), so it costs disk,
/// not 11.7 GB of RAM - but it costs minutes, which is why the result is cached
/// beside the checkpoint and why this prints what it is doing.
pub fn expand_lm(lm: &Path, expanded: &Path, want: &[(String, usize)]) -> Result<String, String> {
    let out = utf8(expanded)?.to_string();
    if expanded.exists() {
        // A cached expansion is only usable if it carries the manifest THIS
        // build asks for. It is derived, not shipped, so a stale one (written
        // before a tensor-layout change) is re-derived rather than reported as
        // a missing weight several stages later, where the message names a
        // parameter and not the file that actually needs rebuilding.
        match WeightReader::open(&out) {
            Ok(r) if want.iter().all(|(n, _)| r.shape(n).is_some()) => return Ok(out),
            Ok(_) => eprintln!("brain: {out}: cached expansion does not match this build's tensor layout - rebuilding"),
            Err(e) => eprintln!("brain: {out}: cached expansion unreadable ({e}) - rebuilding"),
        }
        std::fs::remove_file(expanded).map_err(|e| format!("{}: removing the stale expansion: {e}", expanded.display()))?;
    }
    eprintln!("brain: expanding {} -> {} (once, ~12 GB on disk)", lm.display(), expanded.display());
    let stats = deepseek2::import::import_file(utf8(lm)?, &out, None)?;
    eprintln!("brain: deepseek-ocr decoder import: {stats}");
    Ok(out)
}

/// A streaming reader over whichever file the decoder's weights live in,
/// expanding the GGUF LM on first use.
///
/// The reader is returned rather than consumed here because the HF layout
/// needs it to outlive a [`checkpoint::remap::RemapSource`] borrowed from it
/// - see [`decoder_source`].
pub fn decoder_reader(files: &Files, cfg: &DeepseekOcrConfig) -> Result<WeightReader, String> {
    match &files.layout {
        Layout::GgufPair { lm, expanded, .. } => {
            let path = expand_lm(lm, expanded, &cfg.decoder.param_list())?;
            WeightReader::open(&path).map_err(|e| format!("{path}: {e}"))
        }
        // No expansion: the upstream shards are read where they landed.
        Layout::Hf => WeightReader::open_hf_dir(&files.dir).map_err(|e| format!("{}: {e}", files.dir.display())),
    }
}

/// The decoder's weights under brain's own names, streaming from `reader`.
///
/// Both layouts go through [`checkpoint::remap::RemapSource`] - the GGUF
/// expansion already carries brain's names, so its plan is the identity, and
/// running it through the same seam buys that layout the same
/// every-parameter-accounted-for check the rename needs.
pub fn decoder_source<'a>(files: &Files, reader: &'a WeightReader, cfg: &DeepseekOcrConfig) -> Result<RemapSource<'a>, String> {
    match files.layout {
        Layout::GgufPair { .. } => {
            let plan = cfg.decoder.param_list().into_iter().map(|(n, _)| (n.clone(), Fetch::Whole(n))).collect();
            let src = RemapSource::new(reader, plan);
            src.validate(&cfg.decoder.param_list())?;
            Ok(src)
        }
        Layout::Hf => crate::hf::decoder_source(reader, cfg),
    }
}

/// The encoder's whole init map, for whichever layout `files` names.
pub fn encoder_weights_for(files: &Files, cfg: &DeepseekOcrConfig) -> Result<HashMap<String, Vec<f32>>, String> {
    match &files.layout {
        Layout::GgufPair { mmproj, .. } => encoder_weights_from(mmproj),
        Layout::Hf => {
            let reader = WeightReader::open_hf_dir(&files.dir).map_err(|e| format!("{}: {e}", files.dir.display()))?;
            crate::hf::encoder_weights(&reader, cfg)
        }
    }
}

/// The composite's config, **derived** from the two files and then checked
/// against the documented preset.
///
/// `block_size` is `DeepseekV2Config`'s run-parameter sequence length; it is
/// inert for inference (`DeepseekOcr::new_with_prompt` takes the real `seq`
/// separately) and only has to be the same value on both sides of the
/// comparison below.
pub fn config(files: &Files, block_size: u32) -> Result<DeepseekOcrConfig, String> {
    let (mmproj, lm) = match &files.layout {
        Layout::GgufPair { mmproj, lm, .. } => (mmproj, lm),
        // The HF layout derives and compares in `crate::hf`, against the same
        // preset and with the same refusal.
        Layout::Hf => {
            let reader = WeightReader::open_hf_dir(&files.dir).map_err(|e| format!("{}: {e}", files.dir.display()))?;
            let path = files.dir.join("config.json");
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let json = serde_json::from_slice(&bytes).map_err(|e| format!("{}: unparseable config.json: {e}", path.display()))?;
            return crate::hf::config_from_shapes(&crate::hf::shapes(&reader), &json, block_size);
        }
    };
    let mg = MmapGguf::open(utf8(mmproj)?)?;
    let vision = gguf::deepseek_ocr_vision::config_from_gguf(&mg)?;
    let cfg = DeepseekOcrConfig {
        sam: sam1::import::config_from_gguf(&mg)?,
        clip: ClipVisionConfig::from_gguf(&vision),
        decoder: deepseek2::import::config_from_file(utf8(lm)?, block_size)?,
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

/// The checkpoint's own tokenizer.
///
/// The HF release ships the real `tokenizer.json`, so that layout reads the
/// merges and specials directly; the GGUF layout has only the
/// `tokenizer.ggml.*` KV, which carries the pre-tokenizer's NAME rather than
/// its regex (see `data::qwen_tokenizer::QwenBpe::from_gguf`).
pub fn tokenizer(files: &Files) -> Result<QwenBpe, String> {
    match &files.layout {
        Layout::GgufPair { lm, .. } => crate::prompt::tokenizer_from_gguf(utf8(lm)?),
        Layout::Hf => QwenBpe::from_dir(utf8(&files.dir)?),
    }
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
        let Layout::GgufPair { expanded, .. } = &f.layout else { panic!("expected the GGUF pair, got {:?}", f.layout) };
        assert_eq!(expanded, &tmp.join(EXPANDED));
        assert!(!expanded.exists(), "the expansion is built on demand, not required");
        std::fs::remove_dir_all(&tmp).ok();
    }

    /// The upstream `transformers` release is recognized on its own shape, and
    /// a directory that is NEITHER release names both rather than reporting a
    /// missing GGUF for what is obviously not a GGUF checkpoint.
    #[test]
    fn locate_recognizes_the_upstream_transformers_layout() {
        let tmp = std::env::temp_dir().join(format!("brain-deepseekocr-hf-{}", std::process::id()));
        std::fs::remove_dir_all(&tmp).ok();
        std::fs::create_dir_all(&tmp).expect("tmp dir");

        // Empty: neither shape, and the message must say so.
        let e = Files::locate(&tmp).unwrap_err();
        assert!(e.contains(STORE) && e.contains(HF_STORE), "an unrecognized directory names both releases: {e}");

        // `config.json` alone is not a checkpoint - the weights must be there
        // too, or `caps::Session::load` would fail much later and deeper.
        std::fs::write(tmp.join("config.json"), b"{}").expect("touch config");
        assert!(Files::locate(&tmp).is_err(), "config.json without weights is not a checkpoint");

        std::fs::write(tmp.join("model.safetensors"), b"").expect("touch weights");
        let f = Files::locate(&tmp).expect("an HF-shaped checkpoint");
        assert_eq!(f.layout, Layout::Hf);
        assert_eq!(f.dir, tmp);
        assert_eq!(f.mmproj(), None, "the HF layout has no mmproj");

        // A sharded release is the same shape, reached through the index -
        // which is also what makes a half-downloaded shard set a refusal
        // rather than a load that fails much later.
        std::fs::remove_file(tmp.join("model.safetensors")).expect("rm");
        let index = br#"{"weight_map":{"a":"model-00001-of-000001.safetensors"}}"#;
        std::fs::write(tmp.join("model.safetensors.index.json"), index).expect("touch index");
        assert!(Files::locate(&tmp).is_err(), "an index whose shard is absent is not a checkpoint");
        std::fs::write(tmp.join("model-00001-of-000001.safetensors"), b"").expect("touch shard");
        assert_eq!(Files::locate(&tmp).expect("a sharded HF checkpoint").layout, Layout::Hf);

        std::fs::remove_dir_all(&tmp).ok();
    }
}
