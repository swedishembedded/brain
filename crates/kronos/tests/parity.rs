// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity vs the reference Kronos on the deterministic rungs:
//! - **T1** tokenizer encode: brain's `(s1, s2)` tokens must be **integer-exact**.
//! - **T2** tokenizer decode: reconstruction cosine/pearson > 0.99.
//! - **T4** decoder `decode_s1`: s1 logits cosine/pearson > 0.99.
//!
//! Both sides read the same normalized context (`t_context.f32`). Env-gated on
//! the imported weights + the golden dump; skips otherwise so CI stays green.
//! Regenerate goldens with `tools/goldens/kronos_dump_reference.py`.

use kronos::{import, KronosTokenizerConfig};
use std::path::Path;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_u32(p: &Path) -> Vec<u32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    d / (na * nb + 1e-12)
}
fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let (ma, mb) = (a.iter().sum::<f32>() / n, b.iter().sum::<f32>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        c += (a[i] - ma) * (b[i] - mb);
        va += (a[i] - ma).powi(2);
        vb += (b[i] - mb).powi(2);
    }
    c / (va.sqrt() * vb.sqrt() + 1e-12)
}

#[test]
fn tokenizer_and_decoder_match_the_reference() {
    let (Ok(tok_dir), Ok(dec_dir)) =
        (std::env::var("BRAIN_KRONOS_TOKENIZER"), std::env::var("BRAIN_KRONOS_DECODER"))
    else {
        return brain_testutil::skip("BRAIN_KRONOS_TOKENIZER / BRAIN_KRONOS_DECODER unset; no Kronos parity");
    };
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS is set");
    }
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");
    let ctx_p = golden.join("t_context.f32");
    if !ctx_p.exists() {
        return brain_testutil::skip("golden dump missing; run tools/goldens/kronos_dump_reference.py");
    }
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(golden.join("t_meta.json")).unwrap()).unwrap();
    let feat = meta["feat"].as_u64().unwrap() as usize;
    let t = meta["context_len"].as_u64().unwrap() as usize;

    let context = read_f32(&ctx_p);
    let ref_s1 = read_u32(&golden.join("t1_s1.u32"));
    let ref_s2 = read_u32(&golden.join("t1_s2.u32"));
    let ref_recon = read_f32(&golden.join("t2_recon.f32"));
    let ref_logits = read_f32(&golden.join("t4_s1_logits.f32"));

    // Load the decoder from ITS OWN config.json rather than
    // `KronosConfig::default()`. The hardcoded default is the Kronos-small
    // tier, so pointing this test at any other release used to fail deep in
    // the importer with "embedding.emb_s1.weight has 851968 elems, expected
    // 524288" -- a tensor-shape error where the real problem is "these
    // goldens are not for this checkpoint". Now the tier is checked against
    // the golden's own record of what produced it, and a mismatch is a
    // MISSING FIXTURE (skip, or a hard failure under BRAIN_REQUIRE_FIXTURES),
    // not a parity violation.
    let (dec_cfg, dec_w) = import::load_decoder(&dec_dir).unwrap();
    let golden_d_model = meta["d_model"].as_u64().map(|v| v as usize);
    if golden_d_model != Some(dec_cfg.d_model) {
        return brain_testutil::skip(&format!(
            "golden dump is from a d_model={} Kronos decoder but BRAIN_KRONOS_DECODER is d_model={}; re-dump with tools/goldens/kronos_dump_reference.py against this checkpoint, or point at the matching tier",
            golden_d_model.map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
            dec_cfg.d_model
        ));
    }
    let tok = kronos::KronosTokenizer::from_weights(
        KronosTokenizerConfig::default(),
        &import::load_hf(&KronosTokenizerConfig::default().param_list(), &tok_dir).unwrap(),
    )
    .unwrap();
    let dec = kronos::KronosDecoder::from_weights(dec_cfg, &dec_w).unwrap();

    // T1 — encode: integer-exact tokens
    let (s1, s2) = tok.encode(&context, t);
    let s1_hits = s1.iter().zip(&ref_s1).filter(|(a, b)| a == b).count();
    let s2_hits = s2.iter().zip(&ref_s2).filter(|(a, b)| a == b).count();
    eprintln!("T1 tokens: s1 {}/{} s2 {}/{} exact", s1_hits, t, s2_hits, t);
    assert_eq!(s1, ref_s1, "T1: s1 tokens must be integer-exact");
    assert_eq!(s2, ref_s2, "T1: s2 tokens must be integer-exact");

    // T2 — decode reconstruction (use the reference tokens so T2 is independent of T1)
    let recon = tok.decode(&ref_s1, &ref_s2);
    let cos2 = cosine(&recon, &ref_recon);
    let pear2 = pearson(&recon, &ref_recon);
    eprintln!("T2 recon: cosine={cos2:.6} pearson={pear2:.6}");
    assert!(cos2 > 0.99 && pear2 > 0.99, "T2: reconstruction diverges (cos {cos2:.4})");
    let _ = feat;

    // T4 — decode_s1 logits (feed reference tokens; empty stamp = no temporal
    // embedding, matching the reference dump's `stamp=None`).
    let (logits, _ctx) = dec.decode_s1(&ref_s1, &ref_s2, &[]);
    let cos4 = cosine(&logits, &ref_logits);
    let pear4 = pearson(&logits, &ref_logits);
    eprintln!("T4 s1_logits: cosine={cos4:.6} pearson={pear4:.6}");
    assert!(cos4 > 0.99 && pear4 > 0.99, "T4: s1 logits diverge (cos {cos4:.4})");
}
