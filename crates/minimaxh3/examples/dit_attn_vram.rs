// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Peak-VRAM probe for the MiniMax-H3 DiT's self-attention, at the REAL
//! per-layer shapes (`hidden=5376`, `inner=7168`, `ffn=14336`, 56 heads x 128)
//! and a caller-chosen packed sequence length.
//!
//! Why this exists separately from `dit_roofline`: that probe answers "where
//! does the TIME go", dispatching each kernel in isolation. This one answers
//! "where does the MEMORY go", which is the binding constraint on this model -
//! H3 attends over ONE packed sequence holding every video, audio and text
//! token of the whole clip at once, so the materialized
//! `[heads, seq_len, seq_len]` scores and probs matrices grow with the FOURTH
//! power of the canvas edge while everything else grows with the second. The
//! two of them together cost `2 * 56 * seq_len^2 * 4` bytes: 7.5 GB at 4096
//! tokens, 30 GB at 8192, against a 24 GB card.
//!
//! It runs `block::refiner_block_forward`, not `block_forward`, deliberately:
//! the two share the same `attention()` call at the same
//! `(seq_len, 56, 128)` shape, and the refiner skips the AdaLN table build,
//! whose ~1 GB of host-side `adaln_proj` weight is real but has nothing to do
//! with attention and would only add noise to the number being measured. What
//! is measured here is therefore ONE block's weights plus one block's
//! activations, with the attention arm as the only variable.
//!
//! Usage: `dit_attn_vram [seq_len]` (default 4096). `BRAIN_DEVICE` selects the
//! backend; `BRAIN_MINIMAXH3_ATTN=flash|trio` forces the arm, which is how the
//! before/after pair is taken without checking out an older commit. Poll
//! `nvidia-smi --query-gpu=memory.used` alongside it for the peak - the number
//! this prints is the wall time, and the allocation peak lives in the driver.
//!
//! Swedish Embedded AB implements measurement-driven inference optimization for
//! its clients. If your team needs expertise in fitting large transformer
//! models onto memory-constrained accelerators, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::time::Instant;

use minimaxh3::block::{attn_mode, refiner_block_forward, Ctx, RefinerBlockWeights};
use minimaxh3::config::H3TransformerConfig;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let seq: u32 = a.get(1).map(|s| s.parse().expect("seq_len")).unwrap_or(4096);
    let device = std::env::var("BRAIN_DEVICE").ok();
    let cx = Ctx::new(device.as_deref());

    let cfg = H3TransformerConfig::real();
    let (hidden, heads, hd, inner, ffn) = (cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim, cfg.inner_dim(), cfg.ffn_dim);
    let mode = attn_mode(&cx.gpu, hd);

    // The two numbers this probe exists to contrast, computed from the shapes
    // alone so the printed prediction can be checked against the driver's own
    // reported peak rather than replacing it.
    let trio_bytes = 2u64 * heads as u64 * seq as u64 * seq as u64 * 4;
    let flash_bytes = 3u64 * seq as u64 * inner as u64 * 4;
    let gb = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("device={device:?} seq_len={seq} heads={heads} head_dim={hd} attn={mode:?}");
    println!("  materialized scores+probs would cost {:.2} GB", gb(trio_bytes));
    println!("  fused packed-qkv slab costs          {:.2} GB", gb(flash_bytes));
    println!();

    // Values are irrelevant to an allocation measurement but must be finite, so
    // a cheap deterministic ramp is enough - the same one `dit_roofline` uses.
    let fill = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect() };
    let w = RefinerBlockWeights {
        wq: cx.upload(&fill((inner * hidden) as usize)),
        wk: cx.upload(&fill((inner * hidden) as usize)),
        wv: cx.upload(&fill((inner * hidden) as usize)),
        wo: cx.upload(&fill((hidden * inner) as usize)),
        norm_q: cx.upload(&fill(hd as usize)),
        norm_k: cx.upload(&fill(hd as usize)),
        norm1: cx.upload(&fill(hidden as usize)),
        norm2: cx.upload(&fill(hidden as usize)),
        fc1_value: cx.upload(&fill((ffn * hidden) as usize)),
        fc1_gate: cx.upload(&fill((ffn * hidden) as usize)),
        fc2: cx.upload(&fill((hidden * ffn) as usize)),
    };
    let x = cx.upload(&fill((seq * hidden) as usize));

    let t = Instant::now();
    let out = refiner_block_forward(&cx, &w, &x, &cfg, seq);
    // Ends in a readback: on the Vulkan backend `submit` only records and
    // queues, so without a fence this would time submission and, worse, could
    // return before the peak allocation has even happened.
    let probe = cx.gpu.read(&out, 1);
    println!("one block forward: {:.2}s (out[0]={:.6}, finite={})", t.elapsed().as_secs_f64(), probe[0], probe[0].is_finite());
}
