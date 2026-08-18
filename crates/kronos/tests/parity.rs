// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Parity vs the reference Kronos, rung by rung, on the deterministic ladder:
//!
//! - **N**  per-feature normalization of the raw bars.
//! - **T1** tokenizer encode: brain's `(s1, s2)` tokens must be **integer-exact**.
//! - **T2** tokenizer decode: the reconstruction.
//! - **T4** decoder `decode_s1` with no calendar: argmax ids **integer-exact** at
//!   every position, plus the trailing logit rows.
//! - **T4b** the same over the 512-bar window WITH the real calendar - the
//!   temporal embedding is otherwise untested.
//! - **T5** `decode_s2`: the dependency layer, called the way the reference calls
//!   it (one sampled `s1` as the query).
//! - **T6** the composed argmax rollout with a context LONGER than the 512-bar
//!   window, so it slides: the generated `(s1, s2)` streams **integer-exact**,
//!   and `forecast`'s denormalized bars.
//! - **T7** the same on a context short enough that the window never slides,
//!   where `forecast_cached` also has a right answer - see the note on T7 below.
//!
//! This is the only check in the tree with an answer that does not come from
//! brain. Every other forecasting gate compares one brain path against another,
//! which a defect present on both paths passes; T5 and T6 exist because two such
//! defects (a dependency layer that rotated its cross-attention, and a
//! detokenization that dropped the context window) did exactly that.
//!
//! **Why the cached rollout is only gated in T7.** The reference re-runs its
//! whole 512-bar window every step, from a window origin one bar later each
//! time. No K/V cache reproduces that: a cached key was computed under the
//! previous origin. It is not a rounding difference either - two *correct*
//! un-cached runs whose window origin differs by one bar disagree by ~1.2e-1
//! relative in the final s1 logits on this checkpoint, which is enough to move
//! the argmax. So `forecast_cached` is exact against the reference exactly while
//! `context + horizon <= max_context` (T7's regime) and an approximation beyond
//! it; pretending otherwise here would be gating a fiction.
//!
//! Env-gated on the imported weights + the golden dump; skips otherwise so CI
//! stays green. Regenerate goldens with `tools/goldens/kronos_dump_reference.py`.

use kronos::{import, GenOpts};
use std::path::Path;

use brain_testutil::testdata_path;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn read_u32(p: &Path) -> Vec<u32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    d / (na * nb + 1e-30)
}
/// Relative L2 error `‖a−b‖ / ‖b‖`. The scale-aware measure: cosine alone stays
/// ~1.0 through a gain error, and a plain max-abs says nothing without knowing
/// the field's magnitude.
fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>().sqrt();
    let den: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    num / (den + 1e-30)
}
fn argmax(x: &[f32]) -> u32 {
    x.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a }).0 as u32
}

/// Every float rung must land inside this relative-L2 band. fp32 reassociation
/// across two independent implementations costs ~1e-5 here; the two defects this
/// ladder was written for sat at 3.5e-2 and 3.1e-1.
const TOL: f64 = 1e-3;

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
    let golden = testdata_path("golden/kronos");
    let ctx_p = golden.join("t_context.f32");
    if !ctx_p.exists() {
        return brain_testutil::skip("golden dump missing; run tools/goldens/kronos_dump_reference.py");
    }
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(golden.join("t_meta.json")).unwrap()).unwrap();
    let feat = meta["feat"].as_u64().unwrap() as usize;
    let t = meta["context_len"].as_u64().unwrap() as usize;
    let pred_len = meta["pred_len"].as_u64().unwrap() as usize;
    let max_ctx = meta["max_context"].as_u64().unwrap() as usize;
    let clip = meta["clip"].as_f64().unwrap() as f32;
    let tail = meta["logit_tail"].as_u64().unwrap() as usize;

    // Load the decoder from ITS OWN config.json rather than
    // `KronosConfig::default()`. The hardcoded default is the Kronos-small
    // tier, so pointing this test at any other release used to fail deep in
    // the importer with "embedding.emb_s1.weight has 851968 elems, expected
    // 524288" -- a tensor-shape error where the real problem is "these
    // goldens are not for this checkpoint". Now the tier is checked against
    // the golden's own record of what produced it, and a mismatch is a
    // MISSING FIXTURE (skip, or a hard failure under BRAIN_REQUIRE_FIXTURES),
    // not a parity violation.
    let (dec_cfg, _) = import::load_decoder(&dec_dir).unwrap();
    const DUMPER: &str = "tools/goldens/kronos_dump_reference.py";
    let Some(src) = brain_testutil::golden::Source::open_manifest(&golden.join("t_meta.json"), DUMPER)
    else {
        return;
    };
    if !src.require(&[
        ("d_model", dec_cfg.d_model as i64),
        ("n_layers", dec_cfg.n_layers as i64),
        ("max_context", dec_cfg.max_context as i64),
    ]) {
        return;
    }
    assert_eq!(max_ctx, dec_cfg.max_context, "golden window != checkpoint window");

    let raw = read_f32(&golden.join("t_raw.f32"));
    let context = read_f32(&ctx_p);
    let stamp = read_u32(&golden.join("t_stamp.u32"));
    let ref_s1 = read_u32(&golden.join("t1_s1.u32"));
    let ref_s2 = read_u32(&golden.join("t1_s2.u32"));
    let ref_recon = read_f32(&golden.join("t2_recon.f32"));

    let model = import::load_model(&tok_dir, &dec_dir).unwrap();
    let (tok, dec) = (model.tokenizer(), model.decoder());
    let (vs1, vs2) = (dec_cfg.s1_vocab(), dec_cfg.s2_vocab());

    // N - normalization. The input contract: without it every rung below is
    // being fed a different series than the reference saw.
    let (_, z) = kronos::normalize(&raw, t, feat, clip);
    let n_rel = rel_l2(&z, &context);
    eprintln!("N  normalize:   rel_l2={n_rel:.3e}");
    assert!(n_rel < TOL, "N: normalized context diverges (rel {n_rel:.3e})");

    // T1 - encode: integer-exact tokens. Fed the REFERENCE normalized context so
    // this rung is independent of N.
    let (s1, s2) = tok.encode(&context, t);
    let s1_hits = s1.iter().zip(&ref_s1).filter(|(a, b)| a == b).count();
    let s2_hits = s2.iter().zip(&ref_s2).filter(|(a, b)| a == b).count();
    eprintln!("T1 tokens:      s1 {s1_hits}/{t} s2 {s2_hits}/{t} exact");
    assert_eq!(s1, ref_s1, "T1: s1 tokens must be integer-exact");
    assert_eq!(s2, ref_s2, "T1: s2 tokens must be integer-exact");

    // T2 - decode reconstruction (reference tokens in, so T2 is independent of T1)
    let recon = tok.decode(&ref_s1, &ref_s2);
    let (c2, r2) = (cosine(&recon, &ref_recon), rel_l2(&recon, &ref_recon));
    eprintln!("T2 recon:       cosine={c2:.9} rel_l2={r2:.3e}");
    assert!(r2 < TOL, "T2: reconstruction diverges (rel {r2:.3e})");

    // T4 - decode_s1 with no calendar (empty stamp = the reference's stamp=None).
    let (logits, _) = dec.decode_s1(&ref_s1, &ref_s2, &[]);
    check_s1(&logits, t, vs1, tail, &golden.join("t4_argmax.u32"), &golden.join("t4_logits_tail.f32"), "T4 ");

    // T4b - the same over the last `max_ctx` bars WITH the real calendar: the
    // temporal embedding tables and their summation, which T4 cannot see.
    let w0 = t.saturating_sub(max_ctx);
    let win = t - w0;
    let stw = &stamp[w0 * 5..t * 5];
    let (logits_b, ctxbuf) = dec.decode_s1(&ref_s1[w0..], &ref_s2[w0..], stw);
    check_s1(&logits_b, win, vs1, tail, &golden.join("t4b_argmax.u32"), &golden.join("t4b_logits_tail.f32"), "T4b");

    // T5 - decode_s2: the dependency layer. The reference conditions it on the
    // single sampled s1 at the last position; brain must reproduce that logit
    // row, and must pick the same s2 token from it.
    let ref_samp = read_u32(&golden.join("t5_samp_s1.u32"))[0];
    assert_eq!(argmax(&logits_b[(win - 1) * vs1..win * vs1]), ref_samp, "T5: argmax s1 at the last position");
    let mut s1_cond = ref_s1[w0..].to_vec();
    *s1_cond.last_mut().unwrap() = ref_samp;
    let s2l = dec.decode_s2(&ctxbuf, &s1_cond);
    let row = &s2l[(win - 1) * vs2..win * vs2];
    let ref_s2l = read_f32(&golden.join("t5_s2_logits_last.f32"));
    let (c5, r5) = (cosine(row, &ref_s2l), rel_l2(row, &ref_s2l));
    eprintln!("T5 s2_logits:   cosine={c5:.9} rel_l2={r5:.3e} argmax brain={} ref={}", argmax(row), argmax(&ref_s2l));
    assert_eq!(argmax(row), argmax(&ref_s2l), "T5: the sampled s2 token must match");
    assert!(r5 < TOL, "T5: dependency-layer s2 logits diverge (rel {r5:.3e})");

    // T6 - the composed argmax rollout. First the token streams, integer-exact
    // over every step: this is what catches a rollout that drifts.
    let ref_g1 = read_u32(&golden.join("t6_gen_s1.u32"));
    let ref_g2 = read_u32(&golden.join("t6_gen_s2.u32"));
    let (mut b1, mut b2) = (ref_s1.clone(), ref_s2.clone());
    for _ in 0..pred_len {
        let len = b1.len();
        let lo = len.saturating_sub(max_ctx);
        let ww = len - lo;
        let (lg, cx) = dec.decode_s1(&b1[lo..], &b2[lo..], &stamp[lo * 5..len * 5]);
        let a = argmax(&lg[(ww - 1) * vs1..ww * vs1]);
        let mut cond = b1[lo..].to_vec();
        *cond.last_mut().unwrap() = a;
        let s2lg = dec.decode_s2(&cx, &cond);
        let b = argmax(&s2lg[(ww - 1) * vs2..ww * vs2]);
        b1.push(a);
        b2.push(b);
    }
    eprintln!("T6 rollout s1:  brain={:?}", &b1[t..]);
    eprintln!("T6 rollout s1:  ref  ={ref_g1:?}");
    eprintln!("T6 rollout s2:  brain={:?}", &b2[t..]);
    eprintln!("T6 rollout s2:  ref  ={ref_g2:?}");
    assert_eq!(&b1[t..], &ref_g1[..], "T6: generated s1 stream must be integer-exact");
    assert_eq!(&b2[t..], &ref_g2[..], "T6: generated s2 stream must be integer-exact");

    // ...then the user-facing entry points end to end, raw bars in and
    // denormalized bars out. Both the plain and the KV-cached rollout: they are
    // separately gated against each other, but only this pins either to the
    // reference.
    let ref_pred = read_f32(&golden.join("t6_pred.f32"));
    let opts = GenOpts { temperature: 1.0, top_k: 0, top_p: 1.0, argmax: true, seed: 0 };
    let out = model.forecast(&raw, &stamp[..t * 5], &stamp[t * 5..], pred_len, &opts);
    assert_eq!(out.len(), pred_len * feat, "T6 forecast: shape");
    let r6 = rel_l2(&out, &ref_pred);
    eprintln!("T6 forecast:    rel_l2={r6:.3e} cosine={:.9}", cosine(&out, &ref_pred));
    assert!(r6 < TOL, "T6 forecast: predicted bars diverge from the reference (rel {r6:.3e})");

    // T7 - the non-sliding regime, where the KV-cached rollout is exact too.
    // This is the rung that pins the path the forecaster actually runs.
    let t7 = meta["t7_context_len"].as_u64().unwrap() as usize;
    assert!(t7 + pred_len <= max_ctx, "T7 golden must fit the window");
    let raw7 = read_f32(&golden.join("t7_raw.f32"));
    let stamp7 = read_u32(&golden.join("t7_stamp.u32"));
    let ref_pred7 = read_f32(&golden.join("t7_pred.f32"));
    let (cs, fs) = (&stamp7[..t7 * 5], &stamp7[t7 * 5..]);
    for (name, got) in [
        ("forecast", model.forecast(&raw7, cs, fs, pred_len, &opts)),
        ("forecast_cached", model.forecast_cached(&raw7, cs, fs, pred_len, &opts)),
        ("forecast_cached_samples", model.forecast_cached_samples(&raw7, cs, fs, pred_len, 1, &opts).remove(0)),
        ("forecast_cached_batch", model.forecast_cached_batch(std::slice::from_ref(&raw7), &[cs.to_vec()], &[fs.to_vec()], pred_len, &opts).remove(0)),
    ] {
        assert_eq!(got.len(), pred_len * feat, "T7 {name}: shape");
        let r7 = rel_l2(&got, &ref_pred7);
        eprintln!("T7 {name:<24} rel_l2={r7:.3e} cosine={:.9}", cosine(&got, &ref_pred7));
        assert!(r7 < TOL, "T7 {name}: predicted bars diverge from the reference (rel {r7:.3e})");
    }
}

/// One `decode_s1` rung: the argmax id at EVERY position must match (a causal
/// mask, a calendar table or a window offset that is off shows up there), and
/// the trailing `tail` logit rows must match numerically.
fn check_s1(logits: &[f32], t: usize, vocab: usize, tail: usize, ids_p: &Path, tail_p: &Path, label: &str) {
    let ref_ids = read_u32(ids_p);
    let ids: Vec<u32> = (0..t).map(|i| argmax(&logits[i * vocab..(i + 1) * vocab])).collect();
    let hits = ids.iter().zip(&ref_ids).filter(|(a, b)| a == b).count();
    let ref_tail = read_f32(tail_p);
    let got_tail = &logits[(t - tail) * vocab..];
    let (c, r) = (cosine(got_tail, &ref_tail), rel_l2(got_tail, &ref_tail));
    eprintln!("{label} s1_logits: argmax {hits}/{t} exact  tail cosine={c:.9} rel_l2={r:.3e}");
    assert_eq!(ids, ref_ids, "{label}: per-position argmax s1 must be integer-exact");
    assert!(r < TOL, "{label}: s1 logits diverge (rel {r:.3e})");
}
