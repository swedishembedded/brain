// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Diagnostic (not a standing gate): per-LAYER residual digest at position 0
//! of the pinned prompt, for the same truncated 4-layer stack
//! `gguf_reference_parity_real.rs` gates - the per-layer twin of
//! `tools/goldens/qwen35_gguf_reference_forward.py --layers 4 --tokens 760
//! --digest`'s own "pos 0 layer N rms ..." lines, so the two can be
//! compared layer boundary by layer boundary instead of only after the
//! whole stack, localizing a divergence to ONE layer before
//! `gdn_intermediate_trace.rs` localizes it further, to one step inside
//! that layer.
//!
//! ```text
//! BRAIN_QWEN35_GGUF=$HOME/models/unsloth/Qwen3.8-27B-Q8_0.gguf \
//!   cargo test -p brain-qwen35 --release --test gdn_layer_boundary_trace -- --nocapture
//! ```

use checkpoint::gguf::MmapGguf;
use model::shard::Shard;
use qwen35::int8_gguf_resident::{resident_config, shard_source};
use qwen35::model::Qwen35;

const FIRST_TOKEN: u32 = 760;
const LAYERS: u32 = 4;

fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len() as f32).sqrt()
}

#[test]
fn dump_per_layer_residual_digest_position0() {
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
    cfg.n_layers = LAYERS;
    let d = cfg.d_model as usize;
    let shard = Shard { start: 0, end: LAYERS as usize, embed: false, head: false, gpu_index: Shard::ANY_GPU };
    let src = shard_source(&mg, &cfg, &shard).expect("fetch plan for the truncated stack");

    let m = Qwen35::new_fp32_shard_src_train(cfg.clone(), 1, 1, &src, shard);
    let row = mg.tensor_range("token_embd.weight", FIRST_TOKEN as usize * d, d).expect("embedding row").expect("dequantize");
    m.write_in_res(&row);
    m.debug_gdn_trace(0); // runs run_forward() once; layer 0 must be GDN for this call to type-check.

    println!("=== brain per-layer residual digest, position 0 (tok {FIRST_TOKEN}) ===");
    for l in 0..LAYERS as usize {
        let r = m.debug_res(l + 1);
        let s: f32 = r.iter().sum();
        println!("  pos 0 layer {l:2} rms {:.6} sum {:.5} first4 {:?}", rms(&r), s, &r[..4]);
    }
}
