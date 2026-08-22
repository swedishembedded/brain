// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Forward parity vs the real `CausalMaskedDiffWithDiT.inference()`
//! reference, dumped by `tools/goldens/cosyvoice3_dump_reference.py`
//! (`flow_real_*`, `mel_real_*`, `campplus_real_*`, `s3tokenizer_real_*`,
//! `llm_real_ar_tokens.i32` - the same reference clip/text every other
//! CosyVoice 3 real-weight test in this port reads, per `manifest.json`'s
//! `run_params`).
//!
//! Parity ladder actually reached, climbing the same rungs
//! `crates/cosyvoice/tests/flow_parity.rs` (CosyVoice 2) already proved,
//! PLUS the two DiT-internal sub-stage taps CosyVoice 2's UNet estimator has
//! no equivalent of:
//!
//! Rung 1 (mapping units): `cv3_flow_import::tests` (name/shape coverage)
//! plus this file's own `import_cv3_flow_pt` call.
//!
//! Rung 2a (DiT sub-stages, real weights, real captured inputs):
//! `dit_input_embed_and_time_embed_match_the_reference` recomputes
//! `InputEmbedding`/`TimestepEmbedding` from the golden's own captured
//! four-tensor input (`x`/`cond`/`text_embed`/`spks`, both CFG batch rows -
//! see the module doc's note on why row 0 is the "real" branch and row 1 is
//! the all-zero "unconditional" branch) and compares against the
//! hook-captured output - gates the two new (non-CV2) building blocks in
//! isolation before anything downstream depends on them.
//!
//! Rungs 2b-3 (condition assembly + single-forward parity):
//! `real_conds_mu_embedding_match_the_reference` runs the token embedding,
//! `PreLookaheadLayer`, `repeat_interleave`, and condition assembly on the
//! REAL fixture inputs and compares `conds`/`mu`/`embedding` against the
//! golden's own forward-hook captures.
//!
//! Rung 4 (composed-loop parity, SLOW - real DiT forward x2 x10 steps):
//! `euler_loop_replay_matches_the_reference_steps` feeds the GOLDEN's own
//! captured `mu`/`conds`/`embedding` through THIS port's `cv3_flow::solve_euler`
//! and compares all 10 Euler steps against `flow_real_euler_steps.f32`. Run
//! with `--release` - a debug-build DiT forward is impractically slow (see
//! `crate::flow`'s own recorded CosyVoice 2 UNet performance gap; a 22-block,
//! 1024-dim DiT is comparably or more expensive).
//!
//! Rungs 3+5 (full independent forward, SLOW): `full_forward_matches_the_reference_mel`
//! composes this port's OWN condition assembly with this port's OWN Euler
//! loop end to end and compares the final mel against `flow_real_mel_out.f32`.
//!
//! Skips cleanly when the golden or the checkpoint is absent.

use brain_testutil::{golden::Source, parity::Table, read_f32, read_i32, testdata_path};
use cosyvoice::cv3_flow;
use cosyvoice::cv3_flow_config::Cv3FlowConfig;
use cosyvoice::cv3_flow_import::{import_cv3_flow_pt, Cv3FlowWeights};

const DUMPER: &str = "tools/goldens/cosyvoice3_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;
const REL_CEIL: f64 = 1e-3;

fn weights_dir() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("BRAIN_COSYVOICE3_FLOW") {
        let p = std::path::PathBuf::from(p);
        return p.join("flow.pt").is_file().then_some(p);
    }
    let p = std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resources/cosyvoice/weights3"));
    p.join("flow.pt").is_file().then_some(p)
}

/// `[c, t]` channel-major -> `[t, c]` time-major - same convention
/// `flow_parity.rs` uses for `mel_real_out.f32`.
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
    weights: Cv3FlowWeights,
    cfg: Cv3FlowConfig,
    prompt_tokens: Vec<u32>,
    gen_tokens: Vec<u32>,
    xvec: Vec<f32>,
    prompt_feat_tc: Vec<f32>,
    mel_len1: usize,
    dir: std::path::PathBuf,
}

fn load() -> Option<Fixtures> {
    let dir = testdata_path("golden/cosyvoice3");
    let meta = dir.join("flow_real_meta.json");
    let src = Source::open_manifest(&meta, DUMPER)?;
    let cfg = Cv3FlowConfig::cosyvoice3();
    if !src.require(&[
        ("input_size", cfg.input_size as i64),
        ("output_size", cfg.output_size as i64),
        ("vocab_size", cfg.vocab_size as i64),
        ("token_mel_ratio", cfg.token_mel_ratio as i64),
        ("pre_lookahead_len", cfg.pre_lookahead_len as i64),
        ("n_timesteps", cfg.n_timesteps as i64),
        ("dit_dim", cfg.dit.dim as i64),
        ("dit_depth", cfg.dit.depth as i64),
        ("dit_heads", cfg.dit.heads as i64),
        ("dit_dim_head", cfg.dit.dim_head as i64),
    ]) {
        return None;
    }
    let wdir = weights_dir().or_else(|| {
        brain_testutil::skip("set BRAIN_COSYVOICE3_FLOW to a directory containing flow.pt");
        None
    })?;

    let prompt_tokens = read_i32(dir.join("s3tokenizer_real_tokens.i32"))?;
    let gen_tokens = read_i32(dir.join("llm_real_ar_tokens.i32"))?;
    let xvec = read_f32(dir.join("campplus_real_out.f32"))?;
    let mel_ct = read_f32(dir.join("mel_real_out.f32"))?;
    let mel_len1 = mel_ct.len() / cfg.output_size as usize;
    let prompt_feat_tc = transpose_ct_to_tc(&mel_ct, cfg.output_size as usize, mel_len1);

    let flow_pt = wdir.join("flow.pt");
    let weights = import_cv3_flow_pt(flow_pt.to_str().unwrap(), &cfg).unwrap_or_else(|e| panic!("import_cv3_flow_pt: {e}"));

    Some(Fixtures { weights, cfg, prompt_tokens, gen_tokens, xvec, prompt_feat_tc, mel_len1, dir })
}

#[test]
fn dit_input_embed_and_time_embed_match_the_reference() {
    let Some(f) = load() else { return };

    let want_ie_out = read_f32(f.dir.join("flow_real_dit_input_embed_out.f32")).expect("flow_real_dit_input_embed_out.f32");
    let in_x = read_f32(f.dir.join("flow_real_dit_input_embed_in_x.f32")).expect("flow_real_dit_input_embed_in_x.f32");
    let in_cond = read_f32(f.dir.join("flow_real_dit_input_embed_in_cond.f32")).expect("flow_real_dit_input_embed_in_cond.f32");
    let in_text_embed =
        read_f32(f.dir.join("flow_real_dit_input_embed_in_text_embed.f32")).expect("flow_real_dit_input_embed_in_text_embed.f32");
    let in_spks = read_f32(f.dir.join("flow_real_dit_input_embed_in_spks.f32")).expect("flow_real_dit_input_embed_in_spks.f32");

    let mel = f.cfg.dit.mel_dim as usize;
    let dim = f.cfg.dit.dim as usize;
    let spk = f.cfg.dit.spk_dim as usize;
    // Captured shape is [2, T, mel] (batch=2 CFG trick - see the module doc).
    let t = in_x.len() / (2 * mel);
    assert_eq!(want_ie_out.len(), 2 * t * dim, "dit_input_embed_out golden length");
    assert_eq!(in_spks.len(), 2 * spk);

    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    for b in 0..2usize {
        let x_b = &in_x[b * t * mel..(b + 1) * t * mel];
        let cond_b = &in_cond[b * t * mel..(b + 1) * t * mel];
        let text_embed_b = &in_text_embed[b * t * mel..(b + 1) * t * mel];
        let spks_b = &in_spks[b * spk..(b + 1) * spk];
        let want_b = &want_ie_out[b * t * dim..(b + 1) * t * dim];

        let got = cv3_flow::input_embed(x_b, cond_b, text_embed_b, spks_b, &f.cfg.dit, &f.weights.dit.input_embed, t);
        table.check(&format!("dit_input_embed_out[batch {b}]"), &got, want_b);
    }

    let te_in = read_f32(f.dir.join("flow_real_dit_time_embed_in.f32")).expect("flow_real_dit_time_embed_in.f32");
    let te_out = read_f32(f.dir.join("flow_real_dit_time_embed_out.f32")).expect("flow_real_dit_time_embed_out.f32");
    assert_eq!(te_in.len(), 2, "dit_time_embed_in golden is [2] (both CFG batch rows share the same t)");
    assert_eq!(te_out.len(), 2 * dim);
    for b in 0..2usize {
        let got = cv3_flow::time_embed(te_in[b], &f.cfg.dit, &f.weights.dit.time_embed);
        table.check(&format!("dit_time_embed_out[batch {b}]"), &got, &te_out[b * dim..(b + 1) * dim]);
    }
    table.print();
    table.assert_clean();
}

#[test]
fn real_conds_mu_embedding_match_the_reference() {
    let Some(f) = load() else { return };

    let want_conds = read_f32(f.dir.join("flow_real_conds.f32")).expect("flow_real_conds.f32");
    let want_mu = read_f32(f.dir.join("flow_real_mu.f32")).expect("flow_real_mu.f32");
    let want_embedding = read_f32(f.dir.join("flow_real_embedding.f32")).expect("flow_real_embedding.f32");

    let got_embedding = cv3_flow::speaker_embedding(&f.weights, &f.cfg, &f.xvec);
    let (got_mu, got_conds, mel_len1, mel_len2) =
        cv3_flow::assemble_conditions(&f.weights, &f.cfg, &f.prompt_tokens, &f.gen_tokens, &f.prompt_feat_tc, f.mel_len1);

    assert_eq!(mel_len1, f.mel_len1);
    println!("cv3 flow conds/mu/embedding: mel_len1={mel_len1} mel_len2={mel_len2} T={}", mel_len1 + mel_len2);
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

    let noise = cv3_flow::rand_noise();
    let noise_len = noise.len() / mel;
    let mut x0 = vec![0.0f32; mel * t];
    for c in 0..mel {
        x0[c * t..(c + 1) * t].copy_from_slice(&noise[c * noise_len..c * noise_len + t]);
    }

    let got_steps = cv3_flow::solve_euler(&f.cfg, &f.weights.dit, &x0, &mu, &embedding, &conds, t, f.cfg.n_timesteps as usize);
    let got_flat: Vec<f32> = got_steps.into_iter().flatten().collect();

    let mut table = Table::new(COS_FLOOR, REL_CEIL);
    table.check("flow_real_euler_steps (all 10)", &got_flat, &want_steps);
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
    let noise = cv3_flow::rand_noise();

    let out = cv3_flow::forward(
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
