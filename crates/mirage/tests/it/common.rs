// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Shared fixtures: the released configuration and the model-store lookup
//! for the real checkpoint files.

use std::path::PathBuf;

/// The model-store reference of the released checkpoint.
pub const REFERENCE: &str = "Cactus-Compute/mirage";

/// The `architecture` object of the released checkpoint's `config.json`.
pub const RELEASED_ARCHITECTURE_JSON: &str = r#"{
  "text": 8192, "dim": 512, "kv_heads": 2, "qk_dim": 48, "v_dim": 64,
  "dec_layers": 8, "taps": 3, "lanes": 4, "sites": [3, 7], "orders": [2, 3],
  "slots": 18432, "hashes": 2, "rope": 100000.0, "canvas": 512,
  "stem": [32, 64, 128, 256, 512], "body_layers": 16, "hconv_kernel": 65,
  "mix_kernel": 3, "attn_layers": 12, "attn_heads": 4, "attn_kv_heads": 1,
  "attn_qk_dim": 32, "attn_v_dim": 64, "enc_rope": 10000.0, "registers": 4,
  "pooled_slots": true, "grn": true, "neck": true, "level_memory": true,
  "conv_norm": true, "max_tokens": 1536, "buckets": [512, 1024, 1536],
  "form_share": 0.1, "trace_share": 0.2, "name_loss": 0.5, "seg_loss": 0.0,
  "detections": 64, "regions": 32, "reads": 8, "heat_alpha": 0.54,
  "head_grad": 0.1, "stride": 8, "dense_dim": 128, "reader_dim": 384,
  "reader_blocks": 4, "read_width": 320, "pad": 0, "eos": 1, "bos": 2,
  "dtype": "float32"
}"#;

/// Host threads the test process may use. The vision and decoder host paths fan
/// out over every core by default, which on a large shared node burns far more
/// CPU time than these shapes need.
pub const HOST_TEST_THREADS: usize = 8;

/// Size the process-wide CPU pool once, before the first parallel loop.
pub fn cap_threads() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = backend_cpu::par::set_global_threads(HOST_TEST_THREADS);
    });
}

/// The released checkpoint directory in the model store, when it is there.
pub fn released_dir() -> Option<PathBuf> {
    cap_threads();
    let dir = PathBuf::from(brain_testutil::model_dir(REFERENCE)?);
    (dir.join("mirage.safetensors").is_file() && dir.join("mirage.cact").is_file() && dir.join("config.json").is_file()).then_some(dir)
}

/// The released checkpoint directory, or a loud skip naming what is missing.
pub fn released_or_skip(test: &str) -> Option<PathBuf> {
    let dir = released_dir();
    if dir.is_none() {
        brain_testutil::skip(&format!("{test}: {REFERENCE} (mirage.safetensors, mirage.cact, config.json) is not in the model store"));
    }
    dir
}

/// A complete checkpoint of the tiny configuration filled with deterministic noise.
pub fn tiny_checkpoint() -> mirage::import::Checkpoint {
    use data::rng::Lcg;
    use mirage::import::{Checkpoint, Metadata, Tensor};
    let config = mirage::config::MirageConfig::tiny();
    let mut rng = Lcg::new(0x5eed); // seed: fixed filler, the tests assert agreement between two paths over whatever it holds
    let tensors = mirage::spec::Manifest::new(&config)
        .iter()
        .map(|t| (t.name.clone(), Tensor { shape: t.shape.clone(), data: (0..t.numel()).map(|_| rng.signed() * 0.5).collect() }))
        .collect();
    Checkpoint { config, metadata: Metadata { source_sha256: "ab".repeat(32), step: 1234, format: "mirage".to_string() }, tensors, packed: Default::default() }
}
