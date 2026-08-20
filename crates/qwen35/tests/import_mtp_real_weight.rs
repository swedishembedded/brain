// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Real-weight gate for `qwen35::import::import_mtp` - the first-ever
//! real-weight MTP import in this crate (`classify()` had, until now,
//! always deliberately dropped every `mtp.*` tensor, so no real-weight MTP
//! import had ever been built or validated - see `crate::import::import_mtp`'s
//! own doc for the full context and the acknowledged lack of an external
//! numerical oracle for this head).
//!
//! Confirms, against the REAL `Qwen/Qwen3.8-27B-FP8` `mtp.safetensors`: the
//! exact expected name set (`Qwen35Config { mtp: true, .. }.param_list()`'s
//! own `mtp.*` subset), correct shapes, and finite values. Not a numerical
//! parity check - none is possible here (see the doc above).
//!
//! Self-skips loudly (never silently) without `BRAIN_QWEN35_DIR` or
//! `mtp.safetensors`. Run with:
//!
//! ```text
//! BRAIN_QWEN35_DIR=[path/to/qwen3.8] \
//!     cargo test -p brain-qwen35 --test import_mtp_real_weight -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use checkpoint::mmap::MmapSafetensors;
use qwen35::config::Qwen35Config;
use qwen35::import::import_mtp;

fn checkpoint_dir() -> Option<PathBuf> {
    std::env::var_os("BRAIN_QWEN35_DIR").map(PathBuf::from)
}

#[test]
#[ignore]
fn import_mtp_on_the_real_checkpoint_produces_the_expected_name_set_shapes_and_finite_values() {
    let Some(dir) = checkpoint_dir() else {
        brain_testutil::skip("BRAIN_QWEN35_DIR unset (set it to a downloaded Qwen/Qwen3.8-27B-FP8 dir to run this)");
        return;
    };
    let shard = dir.join("mtp.safetensors");
    if !shard.exists() {
        brain_testutil::skip_unavailable(&format!("{} missing under BRAIN_QWEN35_DIR", shard.display()));
        return;
    }

    let cfg = Qwen35Config::qwen38_27b();
    let reader = MmapSafetensors::open(&shard).unwrap_or_else(|e| panic!("open {}: {e}", shard.display()));
    let out = import_mtp(&reader, &cfg, 128).unwrap_or_else(|e| panic!("import_mtp: {e}"));

    let expected: Vec<(String, usize)> =
        Qwen35Config { mtp: true, ..cfg.clone() }.param_list().into_iter().filter(|(n, _)| n.starts_with("mtp.")).collect();

    println!("=== import_mtp (real checkpoint) ===");
    println!("expected mtp.* tensors: {}", expected.len());
    println!("produced mtp.* tensors: {}", out.len());

    assert_eq!(out.len(), expected.len(), "produced tensor count must match param_list()'s mtp.* subset exactly");
    for (name, numel) in &expected {
        let v = out.get(name).unwrap_or_else(|| panic!("missing real tensor {name} after import_mtp"));
        assert_eq!(v.len(), *numel, "{name}: real shape mismatch");
        assert!(v.iter().all(|x| x.is_finite()), "{name}: non-finite value(s) in real-weight import");
        println!("  {name}: {} values, finite, e.g. first={:.6}", v.len(), v[0]);
    }

    // The fc_e/fc_h split in particular: both halves must be genuinely
    // different sub-tensors of the real fc.weight (not, say, an accidental
    // aliasing bug that copied the same half twice) - a real-data sanity
    // check the synthetic unit test's hand-seeded distinct constants already
    // covers by construction, re-confirmed here against real values.
    assert_ne!(out["mtp.fc_e.weight"], out["mtp.fc_h.weight"], "fc_e/fc_h must not be identical on real weights");
}
