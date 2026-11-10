// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! From a record on disk to a prepared training sample: the model's tokenizer and reader charset, the decoded
//! image, and the decoder target.

use std::path::Path;

use super::data::{load_picture, Record, RecordError};
use super::objective::{text_target, Sample};
use super::trainer::Prepared;
use crate::cact::CactFile;
use crate::cactmap::CactModel;
use crate::config::{MirageConfig, LANGUAGE_TAGS};
use crate::import::Checkpoint;
use crate::tokenizer::Tokenizer;

/// What preparing a sample needs from the model beyond its tensors.
pub struct Resources {
    /// The SentencePiece tokenizer.
    pub tokenizer: Tokenizer,
    /// The reader's charset; label `i` (non-zero) is `charset[i - 1]`.
    pub charset: Vec<char>,
}

impl Resources {
    /// Read the tokenizer and charset from the `mirage.cact` of a model directory.
    pub fn open(dir: &Path, ck: &Checkpoint) -> Result<Resources, String> {
        let cact = CactModel::new(CactFile::open(&dir.join("mirage.cact")).map_err(|e| e.to_string())?, ck.config.clone()).map_err(|e| e.to_string())?;
        Ok(Resources { tokenizer: Tokenizer::from_blob(cact.tokenizer_blob().map_err(|e| e.to_string())?).map_err(|e| e.to_string())?, charset: cact.charset().map_err(|e| e.to_string())? })
    }
}

/// Decode `rec`'s image under `root` and build its sample and decoder target.
pub fn prepare(rec: &Record, root: &Path, cfg: &MirageConfig, res: &Resources) -> Result<Prepared, RecordError> {
    let pic = load_picture(root, rec)?;
    let sample = Sample::from_picture(rec, &pic, &res.charset)?;
    let lang = res.tokenizer.language_tag(&sample.language).ok_or_else(|| RecordError(format!("language `{}` is not one the tokenizer knows", sample.language)))?;
    let vocab = (cfg.text + LANGUAGE_TAGS) as u32;
    let encode = |s: &str| res.tokenizer.encode(s).unwrap_or_default();
    let target = text_target(cfg, &sample, &encode, lang, vocab);
    Ok(Prepared { sample, target })
}

/// Make a training run directory servable as a model: copy the deployment file (tokenizer and reader charset) from
/// `model_dir` and write a `config.json` that agrees with the run's weights, so `Mirage::open` and every surface
/// above it load the directory as it is.
pub fn make_servable(model_dir: &Path, run_dir: &Path) -> Result<(), String> {
    let weights = crate::import::load(&run_dir.join("mirage.safetensors"), &crate::import::ImportOptions::default()).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(model_dir.join("config.json")).map_err(|e| format!("{}: {e}", model_dir.display()))?;
    let mut sidecar = crate::config::CheckpointConfig::from_json(&text).map_err(|e| e.to_string())?;
    sidecar.training_step = weights.metadata.step;
    sidecar.source_checkpoint_sha256 = weights.metadata.source_sha256.clone();
    std::fs::write(run_dir.join("config.json"), serde_json::to_string_pretty(&sidecar).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    std::fs::copy(model_dir.join("mirage.cact"), run_dir.join("mirage.cact")).map(|_| ()).map_err(|e| format!("cannot copy mirage.cact into {}: {e}", run_dir.display()))
}
