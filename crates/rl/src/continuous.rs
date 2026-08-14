// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The continuous-training loop cycle: [`atif::ingest_dir`] +
//! [`crate::fit_weighted`] + a versioned LoRA adapter + residency hot-swap,
//! tied together - self-improve roadmap P5's "continuous-loop driver."
//!
//! Deliberately qwen3-specific glue (unlike [`crate::fit_weighted`] and
//! [`crate::atif`], which are generic/domain-general respectively): turning
//! a trained model into a SERVED adapter update is inherently about one
//! concrete resident (`resident_llm::QwenResident`), not something to
//! generalize further in this pass. A caller wires [`run_cycle`] to a
//! `QwenResident`/`Executor` pair it already owns - this module does not
//! reach into `crates/residency`/`crates/cli` itself, keeping the
//! dependency direction the same way it already runs (CLI/residency depend
//! on model crates, never the reverse).

use std::path::{Path, PathBuf};

use crate::atif::ingest_dir;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use model::FitOpts;
use qwen3::config::QwenConfig;
use qwen3::model::Qwen;

/// One continuous-training cycle over the trajectories currently sitting in
/// `trajectories_dir`:
///
/// 1. [`atif::ingest_dir`] into a scratch weighted-dataset directory.
/// 2. If nothing was ingested, returns `Ok(None)` - a quiet, non-error
///    "nothing new since last cycle."
/// 3. [`crate::fit_weighted`] resumes `training_checkpoint` (the FULL,
///    resumable base+adapter state - distinct from the small adapter-only
///    files this produces for serving) if it exists, else starts fresh from
///    `base_checkpoint` with `lora` freshly initialized, and trains `opts.
///    steps` steps on the ingested data.
/// 4. Reloads the just-saved `training_checkpoint` (a fresh `Qwen`
///    instance - `fit_weighted` itself only returns losses, matching
///    `model::train::fit`'s exact contract, so extracting the adapter needs
///    its own small reload rather than widening that signature) and writes
///    a NEW, versioned, adapter-only file via `qwen3::lora::save_adapter`
///    into `adapter_out_dir` - a distinct filename per cycle (`adapter-
///    <unix-ish counter>.safetensors`, from the number of files already
///    there), so a resident never gets pointed at a file that's still being
///    written.
///
/// Returns the new adapter file's path on success. Hot-swapping a resident
/// onto it (`QwenResident::set_adapter` + `Executor::evict`) is the
/// caller's job - this function only produces the artifact.
#[allow(clippy::too_many_arguments)]
pub fn run_cycle(
    trajectories_dir: &Path,
    base_checkpoint: &Path,
    training_checkpoint: &Path,
    adapter_out_dir: &Path,
    tok: &QwenBpe,
    tmpl: &ChatTemplate,
    lora_rank: u32,
    lora_alpha: f32,
    opts: &FitOpts,
) -> std::io::Result<Option<PathBuf>> {
    let base = checkpoint::load(base_checkpoint.to_str().expect("utf-8 path"));
    let vocab = QwenConfig::from_json(&base.header["config"]).vocab;

    let dataset_dir = training_checkpoint
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{}.dataset", training_checkpoint.file_stem().and_then(|s| s.to_str()).unwrap_or("rl")));
    let count = ingest_dir(trajectories_dir, tok, tmpl, vocab as usize, &dataset_dir)?;
    if count == 0 {
        return Ok(None);
    }

    let cfg = if training_checkpoint.exists() {
        // fit_weighted's own resume path reads the checkpoint's config, but
        // we need it HERE too (base_id/card wiring below) -- reading it
        // again is cheap (a small JSON header, not the tensors) and keeps
        // fit_weighted's generic signature untouched.
        let c = checkpoint::load(training_checkpoint.to_str().expect("utf-8 path"));
        QwenConfig::from_json(&c.header["config"])
    } else {
        let mut c = QwenConfig::from_json(&base.header["config"]);
        c.lora = Some(qwen3::config::LoraCfg::attn(lora_rank, lora_alpha));
        c
    };

    crate::fit_weighted::<Qwen>(&dataset_dir, cfg, opts, Some(training_checkpoint))?;

    // Extract just the adapter tensors for serving -- see this function's
    // doc comment on why a fresh reload, not a value fit_weighted returns.
    let trained = checkpoint::load(training_checkpoint.to_str().expect("utf-8 path"));
    let trained_cfg = QwenConfig::from_json(&trained.header["config"]);
    let init = trained.by_role("");
    let block = trained_cfg.block_size;
    let model = Qwen::new(trained_cfg, 1, block, &init);

    std::fs::create_dir_all(adapter_out_dir)?;
    let version = std::fs::read_dir(adapter_out_dir)?.count();
    let adapter_path = adapter_out_dir.join(format!("adapter-{version:06}.safetensors"));
    let base_id = base_checkpoint.file_stem().and_then(|s| s.to_str()).unwrap_or("base");
    qwen3::lora::save_adapter(adapter_path.to_str().expect("utf-8 path"), &model, &format!("adapter-{version:06}"), base_id, None)?;

    Ok(Some(adapter_path))
}
