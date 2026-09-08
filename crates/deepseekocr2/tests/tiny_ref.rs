// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **The resampler's gate**: stage-by-stage forward parity against the
//! checkpoint-free DeepSeek-OCR-2 vision-tower reference dump.
//!
//! ```text
//! python3 tools/goldens/deepseekocr2_dump_reference.py --out testdata/deepseekocr2
//! ```
//!
//! writes `testdata/deepseekocr2/tiny/{ckpt/model.safetensors,golden.safetensors}`
//! and `manifest-tiny.json`. Fixtures resolve from `$BRAIN_TESTDATA`; this test
//! SKIPS itself when absent, the same convention `crates/deepseek2ocr`'s own
//! `tiny_ref.rs` uses.
//!
//! ## Why `scores_post_mask` is not a plain cosine check
//!
//! The reference script and brain's `attn_prefix_mask.wgsl` kernel both
//! implement `allow(i,j) = (i<P && j<P) || (j<=i)`, but with DIFFERENT
//! sentinel magnitudes for a disallowed pair (the reference adds `-1e9`, the
//! kernel subtracts `1e30`). Both underflow to an exact `0.0` probability
//! after softmax in fp32 - that is why [`GATE`] still applies to `probs` and
//! every downstream tap - but the raw post-mask SCORE at a masked position is
//! not the same number in the two implementations, and a cosine check over a
//! vector dominated by two wildly different huge outliers is not a
//! trustworthy signal either way. So this test checks the thing that
//! actually matters: which positions got masked, computed independently from
//! `i`/`j`/`prefix` rather than from either implementation's arithmetic, and
//! this same allow/disallow computation is what the mutation test below
//! inverts to prove the check can fail.

use std::collections::HashMap;
use std::path::PathBuf;

use brain_testutil::parity::{load, Report};
use checkpoint::safetensors::StTensor;
use deepseekocr2::config::{DeepseekOcr2VisionConfig, Qwen2EncoderConfig};
use deepseekocr2::encoder::{self, gather_rows, Resampler};

/// fp32 end to end, no quantization anywhere: anything below this is a bug.
const GATE: f64 = 0.999_99;

fn testdata(rel: &str) -> PathBuf {
    brain_testutil::testdata_path(&format!("deepseekocr2/{rel}"))
}

/// The tiny dims the golden dumper chose (`TINY` in
/// `tools/goldens/deepseekocr2_dump_reference.py`), read back from the
/// checkpoint's own tensor shapes rather than hard-coded twice.
struct TinyDims {
    hidden: u32,
    heads: u32,
    kv_heads: u32,
    ff: u32,
    layers: u32,
    n_query_local: u32,
    n_query_global: u32,
    decoder_hidden: u32,
}

fn dims_from_ckpt(ck: &HashMap<String, StTensor>) -> TinyDims {
    let shape = |name: &str| &ck.get(name).unwrap_or_else(|| panic!("checkpoint tensor {name} missing")).shape;
    let hidden = shape("encoder.layer0.ln1.weight")[0] as u32;
    let heads_hd = shape("encoder.layer0.attn_q.weight")[0] as u32;
    let kv_hd = shape("encoder.layer0.attn_k.weight")[0] as u32;
    let head_dim = 4; // fixed by the dumper's TINY config; not derivable from a square [heads*hd, hidden] weight alone.
    TinyDims {
        hidden,
        heads: heads_hd / head_dim,
        kv_heads: kv_hd / head_dim,
        ff: shape("encoder.layer0.ffn_gate.weight")[0] as u32,
        layers: 2,
        n_query_local: shape("query_bank.local")[0] as u32,
        n_query_global: shape("query_bank.global")[0] as u32,
        decoder_hidden: shape("projector.bias")[0] as u32,
    }
}

/// Translate the dump's flat attribute-path names into this crate's own
/// `vision.encoder.*` / `vision.projector.*` names - the same translation a
/// real `deepseekocr2::import` will do against the real mmproj's names
/// (`crates/gguf/src/deepseekocr2_vision.rs`), with two-way coverage: every
/// source tensor is consumed, every declared parameter is produced.
fn build_init(cfg: &DeepseekOcr2VisionConfig, ck: &HashMap<String, StTensor>) -> HashMap<String, Vec<f32>> {
    let mut out = HashMap::new();
    let take = |name: &str| ck.get(name).unwrap_or_else(|| panic!("checkpoint tensor {name} missing")).data.clone();
    for l in 0..cfg.encoder.n_layers {
        let src = |leaf: &str| format!("encoder.layer{l}.{leaf}");
        let dst = |leaf: &str| format!("vision.encoder.blocks.{l}.{leaf}");
        for (golden_leaf, mine_leaf) in [
            ("ln1.weight", "norm1.weight"),
            ("ln2.weight", "norm2.weight"),
            ("attn_out.weight", "attn.out.weight"),
            ("ffn_gate.weight", "mlp.gate.weight"),
            ("ffn_up.weight", "mlp.up.weight"),
            ("ffn_down.weight", "mlp.down.weight"),
        ] {
            out.insert(dst(mine_leaf), take(&src(golden_leaf)));
        }
        for (golden_letter, mine_leaf) in [("q", "attn.q"), ("k", "attn.k"), ("v", "attn.v")] {
            out.insert(dst(&format!("{mine_leaf}.weight")), take(&src(&format!("attn_{golden_letter}.weight"))));
            out.insert(dst(&format!("{mine_leaf}.bias")), take(&src(&format!("attn_{golden_letter}.bias"))));
        }
    }
    out.insert("vision.encoder.norm.weight".to_string(), take("encoder.norm.weight"));
    out.insert("vision.query_local.weight".to_string(), take("query_bank.local"));
    out.insert("vision.query_global.weight".to_string(), take("query_bank.global"));
    out.insert("vision.projector.fc.weight".to_string(), take("projector.weight"));
    out.insert("vision.projector.fc.bias".to_string(), take("projector.bias"));
    out.insert("vision.view_separator".to_string(), take("view_separator"));

    // Two-way coverage: every param this crate declares must have landed.
    let mut want: Vec<String> = cfg.encoder.param_list().into_iter().map(|(n, _)| n).collect();
    want.extend(cfg.projector_param_list().into_iter().map(|(n, _)| n));
    for name in &want {
        assert!(out.contains_key(name), "build_init produced nothing for {name}");
    }
    assert_eq!(out.len(), want.len(), "build_init produced an extra tensor nothing declared");
    out
}

/// `allow(i,j) = (i<P && j<P) || (j<=i)`, computed independently of either
/// implementation's masking arithmetic - the ground truth the post-mask-score
/// pattern check and the mutation test both hold both sides to.
fn allowed(i: u32, j: u32, prefix: u32) -> bool {
    (i < prefix && j < prefix) || (j <= i)
}

/// Assert the post-mask scores match THEIR OWN pre-mask scores at every
/// allowed `(i,j)` and are driven far more negative at every disallowed one -
/// checked separately for "got" (against its own pre-mask tap) and "want"
/// (against its own), never cross-implementation, since the two pre-mask
/// tensors already differ by an ordinary fp32 reduction-order epsilon (see
/// `scores_pre_mask`'s own `Report::check` a few lines up) that an exact
/// equality would wrongly flag as a masking bug. `invert`, when true, checks
/// the OPPOSITE pattern - the mutation test's hook.
#[allow(clippy::too_many_arguments)]
fn check_mask_pattern(name: &str, pre_got: &[f32], post_got: &[f32], pre_want: &[f32], post_want: &[f32], heads: u32, t: u32, prefix: u32, invert: bool) {
    for h in 0..heads {
        for i in 0..t {
            for j in 0..t {
                let idx = ((h * t + i) * t + j) as usize;
                let is_allowed = allowed(i, j, prefix) != invert;
                if is_allowed {
                    assert_eq!(post_got[idx], pre_got[idx], "{name}: h{h} ({i},{j}) should be UNMASKED (unchanged from pre-mask) but was not");
                    assert_eq!(post_want[idx], pre_want[idx], "{name}: h{h} ({i},{j}) golden disagrees with its own pre-mask score");
                } else {
                    assert!(post_got[idx] < pre_got[idx] - 1.0e6, "{name}: h{h} ({i},{j}) should be MASKED (driven very negative) but reads {}", post_got[idx]);
                    assert!(post_want[idx] < pre_want[idx] - 1.0e6, "{name}: h{h} ({i},{j}) golden's own mask did not fire");
                }
            }
        }
    }
}

fn run_view(rs: &Resampler, sam_tokens: &[f32], local: bool, n_query: u32, heads: u32, prefix: u32) -> (encoder::ViewTrace, Vec<u32>) {
    let trace = rs.resample_view(sam_tokens, local);
    let t = 2 * n_query;
    (trace, vec![heads, t, prefix])
}

#[test]
fn the_resampler_matches_the_checkpoint_free_golden_stage_by_stage() {
    let Some(ck) = maybe_load(&testdata("tiny/ckpt/model.safetensors")) else {
        return;
    };
    let golden = load(&testdata("tiny/golden.safetensors"));
    let dims = dims_from_ckpt(&ck);

    // This test does not run `sam1::SamEncoder` at all (SAM has its own gate;
    // `sam.local.tile*`/`sam.global` below are the golden's seeded stand-ins
    // for whatever it would have produced) - only `compress_out` is load-
    // bearing here, since `DeepseekOcr2VisionConfig::check` asserts it
    // matches the encoder's own width.
    let sam = sam1::SamViTConfig { compress_out: dims.hidden, ..sam1::SamViTConfig::tiny() };
    let cfg = DeepseekOcr2VisionConfig {
        sam,
        encoder: Qwen2EncoderConfig {
            d_model: dims.hidden,
            n_layers: dims.layers,
            n_heads: dims.heads,
            n_kv_heads: dims.kv_heads,
            ffn_hidden: dims.ff,
            rms_eps: 1e-6,
            rope_theta: 1_000_000.0,
            n_query_local: dims.n_query_local,
            n_query_global: dims.n_query_global,
        },
        decoder_hidden: dims.decoder_hidden,
    };
    let init = build_init(&cfg, &ck);

    let gpu = gpu_core::testgpu::dev(encoder::PIPELINES);
    let rs = Resampler::new_on(gpu, cfg, &init, false);

    let mut report = Report::wide(GATE, 26);
    let mut local_outs = Vec::new();
    let n_tiles = 6; // 3 wide x 2 tall, per the dumper's TINY config.
    for tile in 0..n_tiles {
        let sam_tokens = golden[&format!("sam.local.tile{tile}")].data.clone();
        let (trace, _) = run_view(&rs, &sam_tokens, true, dims.n_query_local, dims.heads, dims.n_query_local);
        check_view(&mut report, &format!("local.tile{tile}"), &golden, &trace, dims.heads, 2 * dims.n_query_local, dims.n_query_local);
        local_outs.push(trace.projected.clone());
    }
    let sam_global = golden["sam.global"].data.clone();
    let (global_trace, _) = run_view(&rs, &sam_global, false, dims.n_query_global, dims.heads, dims.n_query_global);
    check_view(&mut report, "global", &golden, &global_trace, dims.heads, 2 * dims.n_query_global, dims.n_query_global);

    let separator = rs.read_weight("vision.view_separator");
    let gathered = gather_rows(&local_outs, &global_trace.projected, &separator);
    report.check("gathered_rows", &gathered, &golden["gathered_rows"].data);

    // ---- the mutation check: invert the mask's two arms and confirm this
    // gate would actually notice ----
    let (t, prefix) = (2 * dims.n_query_local, dims.n_query_local);
    let pre = &golden["local.tile0.layer0.scores_pre_mask"].data;
    let want_post = &golden["local.tile0.layer0.scores_post_mask"].data;
    let inverted: Vec<f32> = (0..dims.heads)
        .flat_map(|h| {
            (0..t).flat_map(move |i| {
                (0..t).map(move |j| {
                    let idx = ((h * t + i) * t + j) as usize;
                    if allowed(i, j, prefix) {
                        pre[idx] - 1.0e30 // an inverted mask MASKS what should be allowed...
                    } else {
                        pre[idx] // ...and leaves what should be masked untouched.
                    }
                })
            })
        })
        .collect();
    let result = std::panic::catch_unwind(|| {
        check_mask_pattern("mutation", pre, &inverted, pre, want_post, dims.heads, t, prefix, false);
    });
    assert!(result.is_err(), "an inverted mask must fail the pattern check - the gate would otherwise pass hollow");
}

#[allow(clippy::too_many_arguments)]
fn check_view(report: &mut Report, name: &str, golden: &HashMap<String, StTensor>, trace: &encoder::ViewTrace, heads: u32, t: u32, prefix: u32) {
    report.check(&format!("{name}.concat_in"), &trace.concat_in, &golden[&format!("{name}.concat_in")].data);
    for (l, layer) in trace.layers.iter().enumerate() {
        let want_pre = &golden[&format!("{name}.layer{l}.scores_pre_mask")].data;
        let want_post = &golden[&format!("{name}.layer{l}.scores_post_mask")].data;
        report.check(&format!("{name}.layer{l}.scores_pre_mask"), &layer.scores_pre_mask, want_pre);
        check_mask_pattern(&format!("{name}.layer{l}.scores_post_mask"), &layer.scores_pre_mask, &layer.scores_post_mask, want_pre, want_post, heads, t, prefix, false);
        report.check(&format!("{name}.layer{l}.probs"), &layer.probs, &golden[&format!("{name}.layer{l}.probs")].data);
        report.check(&format!("{name}.layer{l}.out"), &layer.out, &golden[&format!("{name}.layer{l}.out")].data);
    }
    report.check(&format!("{name}.query_slice"), &trace.query_slice, &golden[&format!("{name}.query_slice")].data);
    report.check(&format!("{name}.projected"), &trace.projected, &golden[&format!("{name}.projected")].data);
}

/// `load`, but `None` (rather than a panic) when the fixture is simply
/// absent - the crate-local skip this test needs, since `brain_testutil::load`
/// itself always panics on a missing file (a caller that already knows it may
/// be absent is expected to check first).
fn maybe_load(path: &std::path::Path) -> Option<HashMap<String, StTensor>> {
    if !path.exists() {
        brain_testutil::skip("run `python3 tools/goldens/deepseekocr2_dump_reference.py --out testdata/deepseekocr2` first (or set BRAIN_TESTDATA)");
        return None;
    }
    Some(load(path))
}
