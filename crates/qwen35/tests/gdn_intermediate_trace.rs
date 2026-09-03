// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Diagnostic (not a standing gate, same status as
//! `gguf_vs_fp8_permutation_search.rs`): dumps GDN layer 0's own 11-step
//! math, intermediate by intermediate, for the FIRST token of
//! `gguf_reference_parity_real.rs`'s pinned prompt (position 0, where the
//! recurrent state is zero on both sides, so no cross-position accumulation
//! muddies the comparison). Exists to localize
//! `tools/goldens/qwen35_gguf_reference_forward.py`'s remaining divergence
//! from this crate's GGUF route to ONE specific step, per that test's own
//! doc comment - printed here as plain, greppable `NAME: v0 v1 v2 v3 ...`
//! lines for a companion Python run
//! (`tools/goldens/qwen35_gguf_reference_forward.py --trace`) to diff
//! against.
//!
//! Deliberately `t = 1` (a single-token whole-sequence forward): with
//! `b == 1` too, every layout this crate's GDN mixer touches - NCL
//! (channel-major conv), chunk-major (`value_cm`/`beta_cm`) - degenerates to
//! plain per-channel/per-head indexing (`n_chunks == chunk == t == 1`), so
//! every dumped vector needs zero layout math to compare against the
//! reference script's own token-major/per-head arrays.
//!
//! ```text
//! BRAIN_QWEN35_GGUF=$HOME/models/unsloth/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test -p brain-qwen35 --release --test gdn_intermediate_trace -- --nocapture
//! ```

use checkpoint::gguf::MmapGguf;
use model::shard::Shard;
use qwen35::int8_gguf_resident::{resident_config, shard_source};
use qwen35::model::Qwen35;

/// Same prompt/tokenization `gguf_reference_parity_real.rs` pins; only the
/// first token is needed here (position 0).
const FIRST_TOKEN: u32 = 760;

#[test]
fn dump_gdn_layer0_position0_intermediates() {
    let Ok(path) = std::env::var("BRAIN_QWEN35_GGUF") else {
        brain_testutil::skip("BRAIN_QWEN35_GGUF unset (set it to a downloaded Qwen3.8-27B*.gguf to run this)");
        return;
    };
    if gpu_core::devices::gpus().is_empty() {
        brain_testutil::skip_unavailable("no discrete GPU - the fp32 side of this comparison needs VRAM");
        return;
    }

    let mg = MmapGguf::open(&path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mut cfg = resident_config(&mg, 64).expect("resident_config on the real checkpoint");
    cfg.n_layers = 1; // layer 0 only - a GDN layer per gguf_reference_parity_real.rs's own LAYERS doc.
    let d = cfg.d_model as usize;
    let shard = Shard { start: 0, end: 1, embed: false, head: false, gpu_index: Shard::ANY_GPU };
    let src = shard_source(&mg, &cfg, &shard).expect("fetch plan for the truncated stack");

    // t=1: a fresh single-token whole-sequence forward. `write_in_res` feeds
    // the SAME per-row real embedding the decode-based gate reads (never a
    // materialized 5.09 GB table), exactly as `gguf_reference_parity_real
    // .rs` does via `step_with_input`'s own row fetch.
    let m = Qwen35::new_fp32_shard_src_train(cfg.clone(), 1, 1, &src, shard);
    let row = mg.tensor_range("token_embd.weight", FIRST_TOKEN as usize * d, d).expect("embedding row").expect("dequantize");
    m.write_in_res(&row);

    let trace = m.debug_gdn_trace(0);
    println!("=== brain GDN layer 0, position 0 (tok {FIRST_TOKEN}) ===");
    for (name, v) in &trace {
        let head: Vec<String> = v.iter().take(8).map(|x| format!("{x:.7}")).collect();
        println!("{name}: len={} sum={:.7} {}", v.len(), v.iter().sum::<f32>(), head.join(" "));
    }
}
