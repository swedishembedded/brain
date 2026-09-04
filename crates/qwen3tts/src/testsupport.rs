// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Tiny synthetic Talker+MTP checkpoints on disk, shared by the crate's own
//! tests.
//!
//! Every test that wants to drive a REAL generation loop
//! ([`crate::pipeline::generate_codes_cached`], [`crate::batch::run_batch`])
//! needs a checkpoint pair the loaders accept. Downloading the real 0.6B Base
//! checkpoint for that would make the unit-test lane depend on ~2 GB of
//! weights; a tiny randomly-initialised pair exercises exactly the same code
//! paths in milliseconds. This module is the single place that builds one, so
//! the loop tests in `batch.rs` and `pipeline.rs` do not each carry their own
//! copy of the tensor-name/shape knowledge.
//!
//! Swedish Embedded AB implements solutions for fast, weight-free test harnesses
//! around large neural checkpoints for its clients. If your team needs expertise
//! in making model code testable without shipping gigabytes of weights, you can
//! procure our services by sending an email to info@swedishembedded.com.

use crate::prompt::{Prompt, TtsSpecials};
use data::rng::Rng;

/// `TalkerConfig::tiny()` with a real-scale vocab: `sample_cb0` always
/// suppresses the top-1024 vocab entries as the reference's `suppress_tokens`
/// window, so a genuinely tiny vocab (23) underflows `vocab - 1024`. Every
/// other dimension stays tiny for test speed.
pub fn talker_test_cfg() -> crate::config::TalkerConfig {
    crate::config::TalkerConfig { vocab: 1100, ..crate::config::TalkerConfig::tiny() }
}

/// Build a real (tiny synthetic) Talker+MTP checkpoint pair on disk, the same
/// shape the decode paths load via `CpuTalker::load`/`CpuMtp::load`. The
/// Talker's base decoder blocks ARE a `qwen3::Qwen` on disk
/// (`TalkerConfig::to_qwen`, `qwen3::init_weights`; `CpuTalker::load` reads them
/// back via `TalkerConfig::from_qwen`), plus the Talker-specific extras
/// `qwen3::init_weights` knows nothing about (`tok`/`lm_head`/
/// `text_projection.*`/`text_embedding`, normally added by
/// `import::import_talker`) hand-added here with the right shapes - values don't
/// matter, these tests never exercise text prompting (they build `Prompt`
/// directly from random embeddings).
pub fn synthetic_checkpoints(dir: &std::path::Path, seed: u64) -> (String, String) {
    let tcfg = talker_test_cfg();
    let qcfg = tcfg.to_qwen(32);
    let mut init = qwen3::init_weights(&qcfg, seed);

    let mut rng = Rng::new(seed ^ 0x7A1E);
    let mut normal = |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_gaussian() as f32) * 0.02).collect() };
    let (d, vocab, th) = (tcfg.d_model as usize, tcfg.vocab as usize, tcfg.text_hidden_size as usize);
    let inter = th; // no config field for this; derived from the tensor shapes at load time
    init.insert("tok.weight".to_string(), normal(vocab * d));
    init.insert("lm_head.weight".to_string(), normal(vocab * d));
    init.insert("text_projection.fc1.weight".to_string(), normal(inter * th));
    init.insert("text_projection.fc1.bias".to_string(), normal(inter));
    init.insert("text_projection.fc2.weight".to_string(), normal(d * inter));
    init.insert("text_projection.fc2.bias".to_string(), normal(d));
    init.insert("text_embedding.weight".to_string(), normal(tcfg.text_vocab_size as usize * th));

    let tensors: Vec<(String, Vec<u64>, Vec<f32>)> =
        init.into_iter().map(|(k, v)| (k, vec![v.len() as u64], v)).collect();
    let talker_path = dir.join("talker.safetensors").to_str().unwrap().to_string();
    checkpoint::save(&talker_path, qcfg.to_json(), &tensors);

    let mcfg = crate::config::MtpConfig::tiny();
    let mtp_path = dir.join("mtp.safetensors").to_str().unwrap().to_string();
    save_synthetic_mtp(&mcfg, &mtp_path, seed ^ 0x5A5A);

    (talker_path, mtp_path)
}

/// `MtpModel` has no `save`; hand-write the checkpoint `MtpModel::
/// load_inference` expects (the same tensor set `new_synthetic_on` fills
/// in-memory, here written to disk instead).
fn save_synthetic_mtp(cfg: &crate::config::MtpConfig, path: &str, seed: u64) {
    let mut rng = Rng::new(seed);
    let mut normal = |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next_gaussian() as f32) * 0.02).collect() };
    let mut tensors: Vec<(String, Vec<u64>, Vec<f32>)> = Vec::new();
    for (name, numel) in crate::mtp::MtpModel::decoder_param_list(cfg) {
        tensors.push((name, vec![numel as u64], normal(numel)));
    }
    let (nres, e, d, v) =
        (cfg.n_residual() as usize, cfg.embedding_dim as usize, cfg.d_model as usize, cfg.vocab as usize);
    for i in 0..nres {
        tensors.push((format!("codec_embedding.{i}.weight"), vec![(v * e) as u64], normal(v * e)));
        tensors.push((format!("lm_head.{i}.weight"), vec![(v * d) as u64], normal(v * d)));
    }
    checkpoint::save(path, cfg.to_json(), &tensors);
}

/// A prompt of random embeddings: `n_prefix` prefix positions and `n_trail`
/// trailing-text positions, at the tiny Talker's `d_model`.
pub fn tiny_prompt(d: usize, n_prefix: usize, n_trail: usize, rng_seed: u64) -> Prompt {
    let mut rng = Rng::new(rng_seed);
    let mut g = |n: usize| (0..n).map(|_| (rng.next_gaussian() as f32) * 0.1).collect::<Vec<f32>>();
    Prompt { embeds: g(n_prefix * d), trailing: g(n_trail * d), tts_pad: g(d) }
}

/// Special-token ids matching [`talker_test_cfg`]'s vocab.
pub fn tiny_specials() -> TtsSpecials {
    TtsSpecials {
        tts_bos: 0,
        tts_eos: 1,
        tts_pad: 2,
        codec_nothink: 3,
        codec_think: 4,
        codec_think_bos: 5,
        codec_think_eos: 6,
        codec_pad: 7,
        codec_bos: 8,
        // Inside `sample_cb0`'s suppressed top-1024 window ([vocab-1024, vocab)
        // = [76, 1100) at this test's vocab=1100) - mirrors the real model,
        // where EOS lives inside that window and `min_new` genuinely gates
        // whether it's reachable. An id outside the window (e.g. a small one)
        // is NEVER suppressed regardless of `min_new`, which isn't what these
        // tests are meant to exercise.
        codec_eos: 1050,
        lang: std::collections::HashMap::new(),
        spk_id: std::collections::HashMap::new(),
    }
}

/// A scratch directory unique to `tag` and this process, removed by
/// [`Scratch`]'s `Drop` so a failing assertion cannot leak megabytes of
/// synthetic checkpoints into the temp dir.
pub struct Scratch(pub std::path::PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Scratch {
        let dir = std::env::temp_dir().join(format!("qwen3tts-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        Scratch(dir)
    }
    pub fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).ok();
    }
}
