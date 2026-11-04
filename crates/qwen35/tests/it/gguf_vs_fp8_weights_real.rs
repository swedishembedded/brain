// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! M23.1: the direct per-tensor weight diff between what the GGUF resident
//! loads and what the FP8 safetensors route loads, for the SAME real
//! checkpoint - the decisive experiment that could not run earlier for lack
//! of a complete FP8 download. The FP8 route is the known-good side: the
//! SAME engine at the SAME INT8 tier over 64 real layers produced `" Paris."`
//! from FP8, while the GGUF route put `" Paris"` at rank 80. So a leaf-by-leaf
//! diff against FP8 localizes the defect directly instead of continuing to
//! probe it behaviourally.
//!
//! Two readers, both already the PRODUCTION load path for their side (not a
//! new parser written for this test):
//!   - FP8: `qwen35::import::import_layer`, the same per-layer streaming
//!     reader `real_weight_streaming.rs` gates - dequantizes FP8 pairs,
//!     classifies to brain-canonical names, folds plain RMSNorm weights.
//!   - GGUF: `qwen35::int8_gguf_resident::shard_source`, the same
//!     `TensorSource` `Qwen35::new_shard_dt` consumes when this resident
//!     loads for real - dequantizes Q8_0, applies the `ssm_a` LnNeg fix
//!     (lesson #70).
//!
//! For every leaf name `import_layer` returns, this fetches the SAME
//! canonical name from `shard_source` and reports cosine, relative L2, max
//! abs difference, AND a permutation-invariant digest (count, sum, sum of
//! squares, min, max - all invariant to any REORDERING of the same
//! multiset of values). The digest is what turns "cosine is low" into a
//! diagnosis:
//!
//! - cosine ~= 1, digest matches: agrees (Q8_0 rounding only)
//! - cosine low, digest ALSO matches: a PERMUTATION/grouping convention
//!   (repeat_interleave vs tile, a head-interleave, a split order)
//! - cosine low, digest does NOT match: a VALUE transform (the `ssm_a`
//!   LnNeg class)
//! - name missing on one side: a name-map gap
//!
//! This is a DIAGNOSTIC, not a pass/fail correctness gate - M23.3 is where a
//! found defect gets fixed and `gguf_resident_real.rs`'s RED gate turns
//! green. Printing the full per-tensor table is the point; the only hard
//! assertions here are process sanity (both readers must produce SOME
//! output, values must be finite).
//!
//! ```text
//! BRAIN_QWEN35_DIR=/path/to/Qwen3.8-27B-FP8 \
//! BRAIN_QWEN35_GGUF=/path/to/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test -p brain-qwen35 --release --test gguf_vs_fp8_weights_real -- --nocapture --test-threads=1
//! ```
//!
//! Layers swept: 0 and 1 (GDN, early), 3 (GQA, early - `full_attention_
//! interval=4`), 31 and 32 (mid-depth, both mixer types), 62 and 63 (GQA,
//! late - `n_layers=64`). Spans both mixer types across the full depth
//! range without dequantizing the whole 64-layer, ~108 GB model.

use std::path::PathBuf;

use checkpoint::TensorSource;
use checkpoint::gguf::MmapGguf;
use checkpoint::mmap::MmapSafetensors;
use model::Shard;
use qwen35::config::Qwen35Config;
use qwen35::import::import_layer;
use qwen35::int8_gguf_resident::shard_source;

/// Which real per-layer shard file `layer_0..2` etc. actually live in, per
/// `real_weight_streaming.rs`'s own `golden_dir`/`checkpoint_dir` helpers -
/// duplicated here rather than made `pub` there, since exposing test-only
/// path plumbing across a crate boundary is more coupling than a five-line
/// duplicate.
fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("BRAIN_QWEN35_DIR").map(PathBuf::from)
}

fn gguf_path() -> Option<String> {
    match std::env::var("BRAIN_QWEN35_GGUF") {
        Ok(p) if !p.is_empty() => Some(p),
        _ => None,
    }
}

/// A permutation-invariant summary: any reordering of the same multiset of
/// f32 values produces the identical digest. `sum`/`sum_sq` are computed in
/// f64 so a real 27B-scale row (thousands of elements) does not lose the
/// comparison to its own accumulation error before comparing to the other
/// side's.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Digest {
    n: usize,
    sum: f64,
    sum_sq: f64,
    min: f32,
    max: f32,
}

impl Digest {
    fn of(v: &[f32]) -> Digest {
        let mut sum = 0f64;
        let mut sum_sq = 0f64;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for &x in v {
            sum += x as f64;
            sum_sq += (x as f64) * (x as f64);
            min = min.min(x);
            max = max.max(x);
        }
        Digest { n: v.len(), sum, sum_sq, min, max }
    }

    /// Digests "match" within a tolerance scaled by the magnitude involved -
    /// the same backward-stable-bound reasoning `i8_gemv_reg_upgrade.rs`
    /// uses, not an exact float comparison two independently-summed f64
    /// accumulations (different term order) could never satisfy for a
    /// genuinely-equal multiset.
    fn approx_eq(&self, other: &Digest) -> bool {
        if self.n != other.n {
            return false;
        }
        let scale = self.sum_sq.abs().max(other.sum_sq.abs()).max(1e-12);
        let tol = scale * 1e-4;
        (self.sum - other.sum).abs() <= tol.max((self.sum.abs() + other.sum.abs()) * 1e-5 + 1e-6)
            && (self.sum_sq - other.sum_sq).abs() <= tol
            && (self.min - other.min).abs() <= 1e-3 * (self.min.abs().max(other.min.abs()).max(1.0))
            && (self.max - other.max).abs() <= 1e-3 * (self.max.abs().max(other.max.abs()).max(1.0))
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

fn rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in got.iter().zip(want) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    (num / den.max(1e-12)).sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[derive(Default)]
struct Summary {
    agree: u32,
    permutation: u32,
    value_differs: u32,
    missing: u32,
}

/// Diff every leaf `import_layer` returns for layer `l` against the same
/// canonical names from the GGUF `shard_source`, printing one row per leaf
/// and folding the outcome into `summary`.
fn diff_layer(l: usize, fp8: &MmapSafetensors, gguf: &MmapGguf, cfg: &Qwen35Config, block: usize, summary: &mut Summary) {
    let fp8_layer = match import_layer(fp8, cfg, l, block) {
        Ok(m) => m,
        Err(e) => {
            println!("  layer {l}: import_layer failed: {e} (layer not present in this FP8 shard set - skipping)");
            return;
        }
    };
    let shard = Shard { start: l, end: l + 1, embed: false, head: false, gpu_index: Shard::ANY_GPU };
    let Ok(src) = shard_source(gguf, cfg, &shard) else {
        println!("  layer {l}: shard_source failed to build - skipping");
        return;
    };

    let mut names: Vec<&String> = fp8_layer.keys().collect();
    names.sort();

    println!();
    println!("=== layer {l} ({} leaves) ===", names.len());
    println!(
        "  {:<45} {:>12} {:>7} {:>12} {:>12} {:>10}",
        "leaf", "n", "cosine", "rel_l2", "max_abs", "digest"
    );
    for name in names {
        let fp8_v = &fp8_layer[name];
        assert!(fp8_v.iter().all(|x| x.is_finite()), "layer {l}: FP8 side's {name} has a non-finite value");

        let mut gguf_v: Option<Vec<f32>> = None;
        src.with_tensor(name, &mut |raw| gguf_v = Some(raw.to_vec()));
        let Some(gguf_v) = gguf_v else {
            println!("  {name:<45} {:>12}  MISSING on the GGUF side", fp8_v.len());
            summary.missing += 1;
            continue;
        };
        assert!(gguf_v.iter().all(|x| x.is_finite()), "layer {l}: GGUF side's {name} has a non-finite value");

        if fp8_v.len() != gguf_v.len() {
            println!("  {name:<45} {:>12}  SHAPE MISMATCH: fp8={} gguf={}", fp8_v.len(), fp8_v.len(), gguf_v.len());
            summary.missing += 1;
            continue;
        }

        let cos = cosine(&gguf_v, fp8_v);
        let rel = rel_l2(&gguf_v, fp8_v);
        let abs = max_abs(&gguf_v, fp8_v);
        let d_fp8 = Digest::of(fp8_v);
        let d_gguf = Digest::of(&gguf_v);
        let digest_matches = d_fp8.approx_eq(&d_gguf);

        let verdict = if cos > 0.9999 {
            summary.agree += 1;
            "agree"
        } else if digest_matches {
            summary.permutation += 1;
            "PERMUTATION?"
        } else {
            summary.value_differs += 1;
            "VALUE DIFFERS"
        };
        println!("  {name:<45} {:>12} {cos:>7.4} {rel:>12.4e} {abs:>12.4e} {:>10}", fp8_v.len(), if digest_matches { "match" } else { "differ" });
        if verdict != "agree" {
            println!(
                "      -> {verdict}  (fp8: n={} sum={:.4e} sumsq={:.4e} min={:.4e} max={:.4e} | gguf: n={} sum={:.4e} sumsq={:.4e} min={:.4e} max={:.4e})",
                d_fp8.n, d_fp8.sum, d_fp8.sum_sq, d_fp8.min, d_fp8.max, d_gguf.n, d_gguf.sum, d_gguf.sum_sq, d_gguf.min, d_gguf.max
            );
        }
    }
}

#[test]
fn per_tensor_diff_gguf_vs_fp8_across_depth() {
    let Some(dir) = checkpoint_dir() else {
        brain_testutil::skip("BRAIN_QWEN35_DIR unset (set it to a downloaded Qwen/Qwen3.8-27B-FP8 dir to run this)");
        return;
    };
    let Some(gguf_p) = gguf_path() else {
        brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
        return;
    };

    let cfg = Qwen35Config::qwen38_27b();
    let gguf = MmapGguf::open(&gguf_p).unwrap_or_else(|e| panic!("open {gguf_p}: {e}"));

    let mut summary = Summary::default();
    // 0/1 (GDN, early), 3 (GQA, early), 31/32 (mid-depth, both mixer types),
    // 62/63 (GQA, late) - spans the full depth range and both mixer types
    // without dequantizing all 64 layers.
    for &l in &[0usize, 1, 3, 31, 32, 62, 63] {
        let shard_path = dir.join(format!("layers-{l}.safetensors"));
        let Ok(fp8) = MmapSafetensors::open(&shard_path) else {
            println!("layer {l}: {} not present under BRAIN_QWEN35_DIR - skipping", shard_path.display());
            continue;
        };
        diff_layer(l, &fp8, &gguf, &cfg, 128, &mut summary);
    }

    println!();
    println!("=== summary ===");
    println!("  agree:         {}", summary.agree);
    println!("  permutation?:  {}", summary.permutation);
    println!("  value differs: {}", summary.value_differs);
    println!("  missing:       {}", summary.missing);
    if summary.permutation > 0 {
        println!(
            "\n  {} leaf/leaves flagged PERMUTATION - their multiset of values matches but cosine does not, \
             which is the repeat_interleave-vs-tile / head-interleave / split-order signature M21 names. \
             Search the small candidate space for the reordering that restores cosine ~= 1 (M23.3).",
            summary.permutation
        );
    }
    if summary.value_differs > 0 {
        println!(
            "\n  {} leaf/leaves flagged VALUE DIFFERS - a transform, not a reordering (the ssm_a LnNeg class, \
             lesson #70). Check whether the destination name is a known Mapped::Transformed leaf in \
             gguf_import.rs and whether its transform is correctly applied.",
            summary.value_differs
        );
    }
    assert!(summary.agree + summary.permutation + summary.value_differs + summary.missing > 0, "no leaves were compared - checkpoint paths are likely wrong");
}
