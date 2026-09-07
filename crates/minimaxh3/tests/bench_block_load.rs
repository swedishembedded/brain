// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where a streaming DiT forward's wall-clock actually goes, measured on the
//! real checkpoint and **with no GPU involved at all**.
//!
//! `H3Transformer::forward_streaming_with_taps` loads one
//! `transformer_blocks.{i}` per iteration and drops it, 50 times per forward.
//! A P40 run of that loop sits at idle GPU utilization for most of its
//! wall-clock while one host core stays pinned, which says the cost is on
//! the host side of the loader - so this file times the host side ALONE,
//! without opening a device, and separates the two things `block::load_dev`
//! does in one breath:
//!
//! 1. **Materializing** the tensor: `TensorSource::with_tensor` hands back a
//!    whole `Vec<f32>`, so a BF16 checkpoint pays a fresh allocation plus a
//!    scalar BF16 -> f32 decode of the entire tensor before a single byte
//!    reaches the card.
//! 2. **Streaming** it: `TensorSource::with_tensor_chunks` decodes
//!    `UPLOAD_CHUNK_WORDS` at a time into a scratch the source reuses.
//!
//! Both produce identical bytes; only the first allocates per tensor. One
//! real block is 2.58 GB of f32 (`adaln_proj.linear.weight` alone is 260 M
//! parameters = 1.04 GB), so the difference is 129 GB of allocation across a
//! forward if it is real. This measures whether it is.
//!
//! ```text
//! cargo test --release -p brain-minimaxh3 --test bench_block_load \
//!   -- --ignored --nocapture
//! ```

use std::time::Instant;

use checkpoint::TensorSource;

/// Block 0's tensor set with the names the real checkpoint actually uses
/// (`ff.net.0.proj` is the FUSED SwiGLU projection `block::load_fc1` splits).
fn block_tensor_names(i: usize) -> Vec<String> {
    let p = format!("transformer_blocks.{i}");
    [
        format!("{p}.attn.to_q.weight"),
        format!("{p}.attn.to_k.weight"),
        format!("{p}.attn.to_v.weight"),
        format!("{p}.attn.to_out.0.weight"),
        format!("{p}.attn.norm_q.weight"),
        format!("{p}.attn.norm_k.weight"),
        format!("{p}.norm1.weight"),
        format!("{p}.norm2.weight"),
        format!("{p}.ff.net.0.proj.weight"),
        format!("{p}.ff.net.2.weight"),
        format!("{p}.adaln_proj.linear.weight"),
        format!("{p}.adaln_proj.linear.bias"),
    ]
    .into()
}

fn open() -> Option<checkpoint::weightio::WeightReader> {
    let root = std::env::var("BRAIN_MINIMAXH3_DIR").ok().unwrap_or_else(|| {
        format!("{}/.local/share/brain/models/MiniMaxAI/MiniMax-H3", std::env::var("HOME").unwrap_or_default())
    });
    let dit = format!("{root}/transformer");
    if !std::path::Path::new(&dit).join("config.json").exists() {
        eprintln!("skip: no real DiT checkpoint at {dit}");
        return None;
    }
    Some(minimaxh3::caps::open_dit_reader(&dit).expect("open dit reader"))
}

/// The measurement: for each of block 0's tensors, time the whole-tensor
/// materialization `load_dev`/`load_host` use today against the bounded
/// chunked decode `paramstore::upload::Uploader::tensor` uses, and confirm
/// they agree bit-for-bit so the comparison is between two spellings of the
/// same result rather than between two different results.
#[test]
#[ignore]
fn whole_tensor_materialization_versus_bounded_chunked_decode() {
    let Some(r) = open() else { return };
    let chunk = 1usize << 20; // paramstore::UPLOAD_CHUNK_WORDS

    // Two blocks: the first pays cold page-cache, the second is the steady
    // state a 50-block loop actually runs in.
    for block in [0usize, 1] {
        println!("\n== transformer_blocks.{block} ==");
        println!("  {:>34}  {:>12}  {:>10}  {:>10}  {:>8}", "tensor", "MB (f32)", "whole ms", "chunked ms", "ratio");
        let (mut tot_whole, mut tot_chunk, mut tot_mb) = (0f64, 0f64, 0f64);
        for name in block_tensor_names(block) {
            // Whole-tensor: what `block::load_dev` did. `with_tensor` already
            // allocates the whole `Vec<f32>` itself and lends it, so the
            // consumer must NOT copy it again - `ctx.upload(data)` reads the
            // borrowed slice in place. An earlier revision of this probe
            // wrote `d.to_vec()` here and thereby charged this arm a second
            // allocation plus a full extra copy that the real loader never
            // pays, which inflated the gap below by roughly a factor of two.
            // Checksum instead: enough to keep the decode from being
            // optimized away, cheap enough not to be the thing measured.
            let t = Instant::now();
            let mut whole_len = 0usize;
            let mut whole_sum = 0f64;
            let found = r.with_tensor(&name, &mut |d| {
                whole_len = d.len();
                whole_sum = d.iter().map(|&x| x as f64).sum();
            });
            let el_whole = t.elapsed().as_secs_f64() * 1e3;
            assert!(found, "missing {name}");

            // Bounded chunked: what `Uploader::tensor` does - each chunk is
            // consumed (there, written to the device) and nothing is kept.
            let t = Instant::now();
            let mut streamed_len = 0usize;
            let mut streamed_sum = 0f64;
            let found = r.with_tensor_chunks(&name, chunk, &mut |_off, c| {
                streamed_len += c.len();
                streamed_sum += c.iter().map(|&x| x as f64).sum::<f64>();
            });
            let el_chunk = t.elapsed().as_secs_f64() * 1e3;
            assert!(found, "missing {name} (chunked)");
            assert_eq!(streamed_len, whole_len, "{name}: chunked decode must cover the whole tensor");
            assert_eq!(streamed_sum, whole_sum, "{name}: chunked decode must be numerically identical");

            let mb = 4.0 * whole_len as f64 / 1e6;
            let short = name.rsplit_once("transformer_blocks.").map(|(_, s)| s.to_string()).unwrap_or(name.clone());
            println!("  {:>34}  {:>12.1}  {:>10.1}  {:>10.1}  {:>8.2}x", short, mb, el_whole, el_chunk, el_whole / el_chunk.max(1e-9));
            tot_whole += el_whole;
            tot_chunk += el_chunk;
            tot_mb += mb;
        }
        println!("  {:>34}  {:>12.1}  {:>10.1}  {:>10.1}  {:>8.2}x", "TOTAL", tot_mb, tot_whole, tot_chunk, tot_whole / tot_chunk.max(1e-9));
        println!("  extrapolated over 50 blocks: whole {:.1}s vs chunked {:.1}s", tot_whole * 50.0 / 1e3, tot_chunk * 50.0 / 1e3);
    }
}

/// `adaln_proj.linear.weight` on its own: 260 M parameters, 40% of a block's
/// bytes, and the one tensor `block::load_block` keeps on the HOST
/// (`load_host`) rather than uploading - so no device change can touch it.
/// Timed separately because if it dominates, the fix is not an upload fix.
#[test]
#[ignore]
fn adaln_proj_host_materialization_cost() {
    let Some(r) = open() else { return };
    println!("\n== adaln_proj.linear.weight, per block ==");
    let mut total = 0f64;
    for block in 0..4usize {
        let name = format!("transformer_blocks.{block}.adaln_proj.linear.weight");
        let t = Instant::now();
        let mut n = 0usize;
        assert!(r.with_tensor(&name, &mut |d| n = d.len()), "missing {name}");
        let el = t.elapsed().as_secs_f64() * 1e3;
        total += el;
        println!("  block {block}: {:>10.1} ms  ({n} elems, {:.2} GB f32)", el, 4.0 * n as f64 / 1e9);
    }
    println!("  mean {:.1} ms/block => {:.1}s over 50 blocks, host-side, GPU idle", total / 4.0, total / 4.0 * 50.0 / 1e3);
}
