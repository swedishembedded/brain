// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Does brain's INT8 tier still track fp32 on the REAL Qwen3.8-27B weights,
//! through a real stack of real layers?
//!
//! Nothing in this crate answered that before. `tests/model_i8_smoke.rs`
//! compares int8 against fp32 at `Qwen35Config::tiny_i8`'s toy dims;
//! `tests/int8_real_weight_sanity.rs` uses REAL weight values but only one
//! leaf at a time (cosine > 0.99 per linear); `tests/decode_step.rs` compares
//! int8 decode against int8 prefill, which is blind to anything the two tapes
//! get wrong together. None of them can see the failure mode that matters at
//! this scale: brain's int8 tier is **W8A8** (`model::ops::Ops::act`
//! dynamically quantizes ACTIVATIONS per row, it is not weight-only), and a
//! large decoder's residual stream is the textbook home of per-channel
//! outliers that a per-row activation scale flattens. A per-leaf cosine of
//! 0.99 says nothing about 64 of them in series.
//!
//! So: build the SAME layers twice from the SAME GGUF bytes - once fp32
//! (`Qwen35::new_fp32_shard_src`) and once int8 (`Qwen35::new_i8_shard`) -
//! and compare what they compute, token by token, through the identical
//! decode driver. The depth is TRUNCATED (`--layers`, default 8) so the fp32
//! side fits one card: at 27B dims an fp32 layer is ~1.5 GB against int8's
//! ~0.4 GB.
//!
//! Truncating the layer stack is legitimate here rather than a compromise:
//! the question is whether the quantized tape tracks the unquantized one on
//! this weight distribution, and that is a per-layer property whose answer
//! compounds with depth. If eight real layers already diverge, sixty-four
//! cannot converge.
//!
//! ```text
//! BRAIN_QWEN35_GGUF=/path/to/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test -p brain-qwen35 --release --test gguf_i8_vs_fp32_real -- --nocapture
//! ```
//!
//! Self-skips loudly without the file (`brain_testutil::skip`) or without a
//! discrete GPU (`brain_testutil::skip_unavailable`).

use checkpoint::gguf::MmapGguf;
use model::shard::Shard;
use qwen35::int8_gguf_resident::{resident_config, shard_source};
use qwen35::model::Qwen35;

/// How many real decoder layers to build on each side. 8 keeps the fp32 side
/// (~12 GB at 27B dims) inside one 24 GiB card while still crossing both
/// mixer types twice (GDN at 0,1,2,4,5,6; GQA at 3,7).
const LAYERS: u32 = 8;

/// [`LAYERS`], overridable so the divergence-vs-depth TREND can be swept on a
/// box with the file (`BRAIN_QWEN35_I8_LAYERS=4,8,12 ...`). The committed
/// default is what the assertion below is calibrated against; a sweep is a
/// diagnostic, not a different gate.
fn layers() -> u32 {
    std::env::var("BRAIN_QWEN35_I8_LAYERS").ok().and_then(|s| s.parse().ok()).unwrap_or(LAYERS).max(1)
}

/// How many real tokens to push through. Divergence compounds along the
/// sequence as well as along depth (the GDN recurrent state carries it), so a
/// single position would be the easiest possible case.
const STEPS: u32 = 8;

/// Regression floor on the worst per-position cosine - see the assertion's
/// own comment for why it is a floor and not a quality claim.
const FLOOR: f32 = 0.98;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)) as f32
}

fn rms(a: &[f32]) -> f32 {
    (a.iter().map(|x| x * x).sum::<f32>() / a.len() as f32).sqrt()
}

#[test]
fn int8_tracks_fp32_on_real_weights_through_real_layers() {
    let Ok(path) = std::env::var("BRAIN_QWEN35_GGUF") else {
        brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
        return;
    };
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("no discrete GPU - the fp32 side of this comparison needs ~12 GiB of VRAM");
        return;
    }

    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut cfg = resident_config(&mg, 128).expect("resident_config on the real checkpoint");
    // Truncate: `classify` drops every block past `cfg.n_layers` on its own
    // (that is how it already excludes the MTP block), so the fetch plan for
    // the truncated config names exactly layers 0..LAYERS.
    let layers = layers();
    cfg.n_layers = layers;
    let d = cfg.d_model as usize;

    // Both sides hold the same layers and neither holds an endpoint - the
    // resident's own stage shape (see `qwen35::int8_gguf_resident`).
    let shard = Shard { start: 0, end: layers as usize, embed: false, head: false, gpu_index: Shard::ANY_GPU };
    let src = shard_source(&mg, &cfg, &shard).expect("fetch plan for the truncated stack");

    println!("building {layers} real layers twice from {path}");
    let t0 = std::time::Instant::now();
    let i8 = Qwen35::new_i8_shard(cfg.clone(), 1, STEPS, &src, shard.clone());
    println!("  int8 stage built in {:.1} s", t0.elapsed().as_secs_f64());
    let t1 = std::time::Instant::now();
    let fp32 = Qwen35::new_fp32_shard_src(cfg.clone(), 1, STEPS, &src, shard.clone());
    println!("  fp32 stage built in {:.1} s", t1.elapsed().as_secs_f64());

    // Both instances are built at `t = STEPS`, so their OWN per-sequence
    // decode state has room for the whole comparison and neither needs an
    // external `DecodeCaches` (`Qwen35::step_with_input` is the
    // single-sequence wrapper).
    //
    // A real token sequence, embedded exactly the way the resident does it
    // (one row at a time out of the mapping).
    let gtok = mg.tokenizer().expect("embedded tokenizer");
    let tok = data::qwen_tokenizer::QwenBpe::from_gguf(&gtok).expect("QwenBpe::from_gguf");
    let ids = {
        use data::tokenizer::Tokenizer;
        tok.encode("The capital city of France is Paris, and the capital city of Germany is")
    };
    assert!(ids.len() as u32 >= STEPS, "prompt must be at least {STEPS} tokens, got {}", ids.len());

    i8.reset_decode_cache();
    fp32.reset_decode_cache();
    let mut worst = 1.0f32;
    for (i, &id) in ids.iter().take(STEPS as usize).enumerate() {
        let row = mg.tensor_range("token_embd.weight", id as usize * d, d).expect("embedding row").expect("dequantize");
        let a = i8.step_with_input(id, Some(&row));
        let b = fp32.step_with_input(id, Some(&row));
        let c = cosine(&a, &b);
        worst = worst.min(c);
        println!("  pos {i} (tok {id}): cosine(int8, fp32) = {c:.6}   rms int8 {:.4} / fp32 {:.4}", rms(&a), rms(&b));
        assert!(a.iter().all(|x| x.is_finite()) && b.iter().all(|x| x.is_finite()), "pos {i}: non-finite residual");
    }
    println!("worst cosine over {STEPS} positions and {layers} real layers: {worst:.6}");

    // A REGRESSION floor, not a quality claim. Measured on 2x Tesla P40 at
    // `LAYERS = 8`: worst cosine 0.9862. The floor sits below that with
    // margin so a driver or kernel change that makes the tier materially
    // worse fails here, while the tier's own real (and substantial) per-layer
    // loss does not.
    assert!(worst > FLOOR, "int8 has decorrelated from fp32 on real weights after {layers} layers: worst cosine {worst}");
}
