// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the real `CausalMaskedDiffWithXvec.inference()`
//! reference, dumped by `tools/goldens/cosyvoice_dump_reference.py`
//! (`flow_real_*`, `mel_real_*`, `campplus_real_*`, `s3tokenizer_real_*`,
//! `llm_real_ar_tokens.i32` - the same reference clip/text every other
//! real-weight test in this port reads, per `manifest.json`'s `run_params`).
//!
//! Parity ladder actually reached:
//!
//! Rung 1 (mapping units): `flow_import::tests` (name/shape coverage) plus
//! this file's own `import_flow_pt` call, which fails loudly on any
//! mismapping before any forward runs.
//!
//! Rungs 2-3 (stage + single-forward parity):
//! `real_conds_mu_embedding_match_the_reference` runs the token embedding,
//! `UpsampleConformerEncoder`, `encoder_proj`, and condition assembly on the
//! REAL fixture inputs (not hand-assembled - `campplus_real_out.f32` is the
//! actual x-vector, `s3tokenizer_real_tokens.i32` the actual prompt tokens,
//! `llm_real_ar_tokens.i32` the actual generated tokens, `mel_real_out.f32`
//! the actual prompt mel) and compares `conds`/`mu`/`embedding` against the
//! golden's own forward-hook captures.
//!
//! Rung 4 (composed-loop parity): `euler_loop_replay_matches_the_reference_steps`
//! feeds the GOLDEN's own captured `mu`/`conds`/`embedding` (bypassing this
//! port's encoder, isolating the CFM loop) through THIS port's `solve_euler`,
//! using the bit-exact-reproduced fixed noise buffer (see `crate::flow`'s
//! module doc), and compares all 10 Euler steps against
//! `flow_real_euler_steps.f32`.
//!
//! Rungs 3+5 (full independent forward): `full_forward_matches_the_reference_mel`
//! composes both of the above (this port's OWN encoder output feeding this
//! port's OWN Euler loop) end to end and compares the final mel against
//! `flow_real_mel_out.f32` - stronger than a "replay from a hooked midpoint"
//! since every real fixture input is independent, so this is a genuine
//! from-scratch reforward, not a resume.
//!
//! Skips cleanly when the golden or the checkpoint is absent.

use brain_testutil::{golden::Source, parity::Table, read_f32, read_i32, testdata_path};
use cosyvoice::flow;
use cosyvoice::flow_config::FlowConfig;
use cosyvoice::flow_import::{import_flow_pt, FlowWeights};

const DUMPER: &str = "tools/goldens/cosyvoice_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;
const REL_CEIL: f64 = 1e-3;

/// `BRAIN_COSYVOICE_FLOW`, else the repo-relative `resources/cosyvoice/weights`.
/// Same `weights_dir` convention every other real-weight test in this port
/// uses (`crates/cosyvoice/tests/llm_parity.rs`, `crates/campplus/tests/parity.rs`).
fn weights_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_COSYVOICE_FLOW") {
        let p = std::path::PathBuf::from(p);
        return p.join("flow.pt").is_file().then_some(p);
    }
    let p = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights"));
    p.join("flow.pt").is_file().then_some(p)
}

/// `[c, t]` channel-major -> `[t, c]` time-major (`mel_real_out.f32` is
/// dumped straight from `feat_extractor(wav)`'s own `(1, num_mels, T)`
/// layout - see the module doc - but `prompt_speech_feat` as `flow.inference()`
/// actually receives it is time-major, `_extract_speech_feat`'s own
/// `.transpose(0, 1)`).
fn transpose_ct_to_tc(x: &[f32], c: usize, t: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; c * t];
    for ci in 0..c {
        for ti in 0..t {
            y[ti * c + ci] = x[ci * t + ti];
        }
    }
    y
}

struct Fixtures {
    weights: FlowWeights,
    cfg: FlowConfig,
    prompt_tokens: Vec<u32>,
    gen_tokens: Vec<u32>,
    xvec: Vec<f32>,
    prompt_feat_tc: Vec<f32>,
    mel_len1: usize,
    dir: std::path::PathBuf,
}

fn load() -> Option<Fixtures> {
    let dir = testdata_path("golden/cosyvoice");
    let meta = dir.join("flow_real_meta.json");
    let src = Source::open_manifest(&meta, DUMPER)?;
    let cfg = FlowConfig::cosyvoice2();
    if !src.require(&[
        ("input_size", cfg.input_size as i64),
        ("output_size", cfg.output_size as i64),
        ("vocab_size", cfg.vocab_size as i64),
        ("token_mel_ratio", cfg.token_mel_ratio as i64),
        ("pre_lookahead_len", cfg.encoder.pre_lookahead_len as i64),
        ("n_timesteps", cfg.n_timesteps as i64),
    ]) {
        return None;
    }
    let wdir = weights_dir().or_else(|| {
        brain_testutil::skip("set BRAIN_COSYVOICE_FLOW to a directory containing flow.pt");
        None
    })?;

    let prompt_tokens = read_i32(dir.join("s3tokenizer_real_tokens.i32"))?;
    let gen_tokens = read_i32(dir.join("llm_real_ar_tokens.i32"))?;
    let xvec = read_f32(dir.join("campplus_real_out.f32"))?;
    let mel_ct = read_f32(dir.join("mel_real_out.f32"))?;
    let mel_len1 = mel_ct.len() / cfg.output_size as usize;
    let prompt_feat_tc = transpose_ct_to_tc(&mel_ct, cfg.output_size as usize, mel_len1);

    let flow_pt = wdir.join("flow.pt");
    let weights = import_flow_pt(flow_pt.to_str().unwrap(), &cfg).unwrap_or_else(|e| panic!("import_flow_pt: {e}"));

    Some(Fixtures { weights, cfg, prompt_tokens, gen_tokens, xvec, prompt_feat_tc, mel_len1, dir })
}

#[test]
fn real_conds_mu_embedding_match_the_reference() {
    let Some(f) = load() else { return };

    let want_conds = read_f32(f.dir.join("flow_real_conds.f32")).expect("flow_real_conds.f32");
    let want_mu = read_f32(f.dir.join("flow_real_mu.f32")).expect("flow_real_mu.f32");
    let want_embedding = read_f32(f.dir.join("flow_real_embedding.f32")).expect("flow_real_embedding.f32");

    let got_embedding = flow::speaker_embedding(&f.weights, &f.cfg, &f.xvec);
    let (got_mu, got_conds, mel_len1, mel_len2) =
        flow::assemble_conditions(&f.weights, &f.cfg, &f.prompt_tokens, &f.gen_tokens, &f.prompt_feat_tc, f.mel_len1);

    assert_eq!(mel_len1, f.mel_len1);
    println!("flow conds/mu/embedding: mel_len1={mel_len1} mel_len2={mel_len2} T={}", mel_len1 + mel_len2);
    assert_eq!(got_conds.len(), want_conds.len(), "conds length");
    assert_eq!(got_mu.len(), want_mu.len(), "mu length");
    assert_eq!(got_embedding.len(), want_embedding.len(), "embedding length");

    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    table.check("flow_real_conds", &got_conds, &want_conds);
    table.check("flow_real_mu", &got_mu, &want_mu);
    table.check("flow_real_embedding", &got_embedding, &want_embedding);
    table.print();
    table.assert_clean();
}

#[test]
fn euler_loop_replay_matches_the_reference_steps() {
    let Some(f) = load() else { return };

    let mu = read_f32(f.dir.join("flow_real_mu.f32")).expect("flow_real_mu.f32");
    let conds = read_f32(f.dir.join("flow_real_conds.f32")).expect("flow_real_conds.f32");
    let embedding = read_f32(f.dir.join("flow_real_embedding.f32")).expect("flow_real_embedding.f32");
    let want_steps = read_f32(f.dir.join("flow_real_euler_steps.f32")).expect("flow_real_euler_steps.f32");

    let mel = f.cfg.output_size as usize;
    let t = mu.len() / mel;
    assert_eq!(want_steps.len(), f.cfg.n_timesteps as usize * mel * t, "euler_steps golden length");

    let noise = flow::rand_noise();
    let noise_len = noise.len() / mel;
    let mut x0 = vec![0.0f32; mel * t];
    for c in 0..mel {
        x0[c * t..(c + 1) * t].copy_from_slice(&noise[c * noise_len..c * noise_len + t]);
    }

    let got_steps = flow::solve_euler(&f.cfg, &f.weights.estimator, &x0, &mu, &embedding, &conds, t, f.cfg.n_timesteps as usize);
    let got_flat: Vec<f32> = got_steps.into_iter().flatten().collect();

    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    table.check("flow_real_euler_steps (all 10)", &got_flat, &want_steps);
    // The LAST step is the one that actually reaches the final mel - break it
    // out too, since an early-step average could hide a last-step-only bug.
    let last_want = &want_steps[(f.cfg.n_timesteps as usize - 1) * mel * t..];
    let last_got = &got_flat[(f.cfg.n_timesteps as usize - 1) * mel * t..];
    table.check("flow_real_euler_steps (last step only)", last_got, last_want);
    table.print();
    table.assert_clean();
}

#[test]
fn full_forward_matches_the_reference_mel() {
    let Some(f) = load() else { return };

    let want_mel = read_f32(f.dir.join("flow_real_mel_out.f32")).expect("flow_real_mel_out.f32");
    let noise = flow::rand_noise();

    let out = flow::forward(
        &f.weights,
        &f.cfg,
        &f.prompt_tokens,
        &f.gen_tokens,
        &f.xvec,
        &f.prompt_feat_tc,
        f.mel_len1,
        &noise,
        f.cfg.n_timesteps as usize,
    );

    assert_eq!(out.mel.len(), want_mel.len(), "mel_out length");
    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    table.check("flow_real_mel_out", &out.mel, &want_mel);
    table.print();
    table.assert_clean();
}
