// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! LFM2.5-Encoder reference parity: brain's staged forward vs the transformers
//! `modeling_lfm2_bidirectional` reference, per stage (post-embedding residual,
//! every layer output, final hidden, MLM-logit probe rows, fill-mask top-1).
//!
//! Golden fixtures (`testdata/golden/lfm/lfm25_encoder_{230m,350m}.safetensors`,
//! fetched via `make fetch/testdata`, never committed) are baked by `tools/goldens/lfm2_dump_reference.py` from the released fp32
//! checkpoints with FIXED token ids — tokenizer parity is tested separately in
//! `crates/data`. The ~1–1.4 GB weights are NOT committed: set
//! `BRAIN_LFM25_230M` / `BRAIN_LFM25_350M` to the HF checkpoint dirs; the tests
//! skip (never fail) when unset. `BRAIN_DEVICE=cpu` recommended (deterministic).

use std::path::Path;

use lfm2::config::LfmConfig;
use lfm2::model::Lfm;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (&x, &y) in got.iter().zip(want) {
        num += (x as f64 - y as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den).sqrt()
}

/// Resolve a golden under the fetched `testdata/` tree (`make fetch/testdata`;
/// override the root with `BRAIN_TESTDATA`).
fn fixture(name: &str) -> String {
    let root = std::env::var("BRAIN_TESTDATA")
        .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata").to_string());
    format!("{root}/golden/lfm/{name}.safetensors")
}

struct Golden {
    tensors: Vec<checkpoint::safetensors::StTensor>,
}

impl Golden {
    fn get(&self, name: &str) -> &Vec<f32> {
        &self.tensors.iter().find(|t| t.name == name).unwrap_or_else(|| panic!("golden tensor {name}")).data
    }
    fn ids(&self, name: &str) -> Vec<u32> {
        self.get(name).iter().map(|&x| x as u32).collect()
    }
}

/// Stage gate: fp32 vs fp32 on the same math should be tight.
fn check_stage(stage: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{stage}: len {} != golden {}", got.len(), want.len());
    let cos = cosine(got, want);
    let rl2 = rel_l2(got, want);
    eprintln!("  {stage}: cosine={cos:.6} rel_l2={rl2:.5}");
    assert!(cos >= 0.9999, "{stage}: cosine {cos:.6} < 0.9999");
    assert!(rl2 <= 0.02, "{stage}: rel_l2 {rl2:.5} > 0.02");
}

fn run_parity(env_var: &str, fixture_name: &str, cfg: LfmConfig) {
    let hf_dir = match std::env::var(env_var) {
        Ok(p) if !p.is_empty() => p,
        _ => {
            eprintln!("SKIP: set {env_var} to the HF checkpoint dir");
            return;
        }
    };
    if !Path::new(&hf_dir).exists() {
        eprintln!("SKIP: {env_var}={hf_dir} not found");
        return;
    }

    let fx_path = fixture(fixture_name);
    if !std::path::Path::new(&fx_path).exists() {
        eprintln!("SKIP: fixture {fx_path} absent — run `make fetch/testdata`");
        return;
    }
    let golden = Golden { tensors: checkpoint::safetensors::read(&fx_path).expect("read fixture") };
    let tokens = golden.ids("tokens");
    let logit_rows = golden.ids("logit_rows");

    // Import the released weights in memory (same path `brain lfm import` takes).
    let tensors = checkpoint::safetensors::read_model_dir(Path::new(&hf_dir)).expect("read weights");
    let init = lfm2::import::brain_init_from_hf(tensors, &cfg).expect("brain_init_from_hf");
    let model = Lfm::new(cfg.clone(), 1, tokens.len() as u32, &init);
    model.set_tokens(&tokens);
    model.forward();

    eprintln!("{fixture_name} staged parity:");
    // Golden res{l} = HF hidden_states[l]: embeddings then layer outputs —
    // EXCEPT the last entry, which transformers replaces with the post-norm
    // `last_hidden_state` (== golden "hidden"). Brain's pre-norm res[n_layers]
    // is gated transitively through the "hidden" stage below.
    for l in 0..cfg.n_layers() as usize {
        check_stage(&format!("res{l}"), &model.read_res(l), golden.get(&format!("res{l}")));
    }
    check_stage("hidden", &model.read_hidden(), golden.get("hidden"));

    // Logit probe rows.
    let v = cfg.vocab as usize;
    let logits = model.read_logits();
    let want_probe = golden.get("logits_probe");
    for (i, &row) in logit_rows.iter().enumerate() {
        let got = &logits[row as usize * v..(row as usize + 1) * v];
        check_stage(&format!("logits[row {row}]"), got, &want_probe[i * v..(i + 1) * v]);
    }

    // Fill-mask agreement: brain's argmax at the mask row == reference top-1.
    let mask_row = tokens.iter().position(|&t| t == 16).expect("<|mask|> in tokens");
    let row = &logits[mask_row * v..(mask_row + 1) * v];
    let argmax = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0 as u32;
    let want_top = golden.ids("mask_top5_ids");
    eprintln!("  fill-mask: brain argmax {argmax}, reference top5 {want_top:?}");
    assert_eq!(argmax, want_top[0], "fill-mask top-1 mismatch");
    drop(model); // one live device at a time

    // The chunked inference regime against the same goldens (hidden + the
    // probe-row logits through the gathered head).
    let tensors = checkpoint::safetensors::read_model_dir(Path::new(&hf_dir)).expect("read weights");
    let init = lfm2::import::brain_init_from_hf(tensors, &cfg).expect("brain_init_from_hf");
    let model = Lfm::new_chunked(cfg.clone(), 1, tokens.len() as u32, &init, 1 << 30, logit_rows.len() as u32);
    model.set_tokens(&tokens);
    model.set_probe_rows(&logit_rows);
    model.forward();
    eprintln!("{fixture_name} chunked regime:");
    check_stage("chunked hidden", &model.read_hidden(), golden.get("hidden"));
    let probe = model.read_probe_logits();
    for (i, &row) in logit_rows.iter().enumerate() {
        check_stage(&format!("chunked logits[row {row}]"), &probe[i * v..(i + 1) * v], &want_probe[i * v..(i + 1) * v]);
    }
}

#[test]
fn lfm25_encoder_230m_matches_reference() {
    run_parity("BRAIN_LFM25_230M", "lfm25_encoder_230m", LfmConfig::lfm25_encoder_230m());
}

#[test]
fn lfm25_encoder_350m_matches_reference() {
    run_parity("BRAIN_LFM25_350M", "lfm25_encoder_350m", LfmConfig::lfm25_encoder_350m());
}

/// Device-free layout gate: the committed configs' parameter lists match the
/// real checkpoints name-for-name and count-for-count (asserted live when the
/// weights are present; the always-on shape/uniqueness checks live in
/// `crates/lfm/src/config.rs` tests).
#[test]
fn t0_param_layout_matches_checkpoint() {
    for (env_var, cfg) in [
        ("BRAIN_LFM25_230M", LfmConfig::lfm25_encoder_230m()),
        ("BRAIN_LFM25_350M", LfmConfig::lfm25_encoder_350m()),
    ] {
        let Ok(hf_dir) = std::env::var(env_var) else {
            eprintln!("SKIP: {env_var} unset");
            continue;
        };
        let cfg_json = std::fs::read_to_string(format!("{hf_dir}/config.json")).expect("config.json");
        let parsed = lfm2::import::config_from_hf(&cfg_json).expect("config_from_hf");
        assert_eq!(parsed.param_list(), cfg.param_list(), "{env_var}: param layout drift");
    }
}
