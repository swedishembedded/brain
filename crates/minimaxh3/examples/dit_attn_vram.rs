// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Peak-VRAM probe for the MiniMax-H3 DiT's self-attention, at the REAL
//! per-layer shapes (`hidden=5376`, `inner=7168`, `ffn=14336`, 56 heads x 128)
//! and a caller-chosen packed sequence length.
//!
//! It answers both "where does the TIME go" and "where does the MEMORY go"
//! for one block, and memory is the binding constraint on this model: H3
//! attends over ONE packed sequence holding every video, audio and text
//! token of the whole clip at once, so materialized
//! `[heads, seq_len, seq_len]` scores and probs matrices would grow with the FOURTH
//! power of the canvas edge while everything else grows with the second. The
//! two of them together cost `2 * 56 * seq_len^2 * 4` bytes: 7.5 GB at 4096
//! tokens, 30 GB at 8192, against a 24 GB card.
//!
//! It runs `block::block_forward` itself, AdaLN modulation and gated
//! residuals included, over a one-timestep modulation table (the table's
//! values change no cost; its handful of rows is all a real step uploads).
//! What is measured is therefore ONE DiT block's weights plus everything its
//! forward allocates, which is what a card's share of the resident model has
//! to leave free beside its weights.
//!
//! Usage: `dit_attn_vram [seq_len] [f32|i8] [reps] [device] [flash|trio]`
//! (defaults 4096, f32, 3, the ambient device, the device's own pick): the
//! linears at the given tier, exactly as the DiT holds them, and the fastest
//! of `reps` timed forwards after a warm-up one. The last argument pins the
//! attention arm.
//! Poll `nvidia-smi --query-gpu=memory.used` alongside it for the peak - the
//! number this prints is the wall time, and the allocation peak lives in the
//! driver.
//!
//! Swedish Embedded AB implements measurement-driven inference optimization for
//! its clients. If your team needs expertise in fitting large transformer
//! models onto memory-constrained accelerators, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::time::Instant;

use gpu_core::select::Dtype;
use minimaxh3::block::{attn_mode, block_forward, ActLevels, AttnMode, BlockInputs, BlockWeights, Ctx, Linear};
use minimaxh3::config::{H3TransformerConfig, MODALITY_NUM};
use model::ops::Weight;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let seq: u32 = a.get(1).map(|s| s.parse().expect("seq_len")).unwrap_or(4096);
    let tier = match a.get(2).map(String::as_str).unwrap_or("f32") {
        "f32" => Dtype::F32,
        "i8" => Dtype::I8,
        other => panic!("tier {other:?}: f32 or i8"),
    };
    let reps: usize = a.get(3).map(|s| s.parse().expect("reps")).unwrap_or(3);
    let device = a.get(4).cloned();
    let mut cx = Ctx::new(device.as_deref());
    cx.attn = match a.get(5).map(String::as_str) {
        None => None,
        Some("flash") => Some(AttnMode::Flash),
        Some("trio") => Some(AttnMode::Materialized),
        Some(other) => panic!("attention {other:?}: flash or trio"),
    };

    let cfg = H3TransformerConfig::real();
    let (hidden, heads, hd, inner, ffn) = (cfg.hidden_size, cfg.num_attention_heads, cfg.attention_head_dim, cfg.inner_dim(), cfg.ffn_dim);
    let mode = cx.attn.unwrap_or_else(|| attn_mode(&cx.gpu, hd));

    // The two numbers this probe exists to contrast, computed from the shapes
    // alone so the printed prediction can be checked against the driver's own
    // reported peak rather than replacing it.
    let trio_bytes = 2u64 * heads as u64 * seq as u64 * seq as u64 * 4;
    let flash_bytes = 3u64 * seq as u64 * inner as u64 * 4;
    let gb = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    println!("device={device:?} seq_len={seq} heads={heads} head_dim={hd} attn={mode:?} tier={tier:?}");
    println!("  materialized scores+probs would cost {:.2} GB", gb(trio_bytes));
    println!("  fused packed-qkv slab costs          {:.2} GB", gb(flash_bytes));
    println!();

    // Values are irrelevant to an allocation measurement but must be finite, so
    // a cheap deterministic ramp is enough.
    let fill = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect() };
    let lin = |n: u32, k: u32| Linear::plain(Weight::upload(&cx.ops, &fill((n * k) as usize), n as usize, k as usize, tier));
    let w = BlockWeights {
        wq: lin(inner, hidden),
        wk: lin(inner, hidden),
        wv: lin(inner, hidden),
        wo: lin(hidden, inner),
        norm_q: cx.upload(&fill(hd as usize)),
        norm_k: cx.upload(&fill(hd as usize)),
        norm1: cx.upload(&fill(hidden as usize)),
        norm2: cx.upload(&fill(hidden as usize)),
        fc1_value: lin(ffn, hidden),
        fc1_gate: lin(ffn, hidden),
        fc2: lin(hidden, ffn),
    };
    let x = cx.upload(&fill((seq * hidden) as usize));
    // One distinct timestep, every row at its first modality's row: the
    // modulation's values change no cost, only its table's row count does.
    let modulation = fill(MODALITY_NUM as usize * 6 * hidden as usize);
    let adaln_idx = cx.upload_u32(&vec![0u32; seq as usize]);
    let half = 3 * cfg.rope_freq_dim;
    let (cos, sin) = (cx.upload(&vec![1.0; (seq * half) as usize]), cx.upload(&vec![0.0; (seq * half) as usize]));

    // Each forward ends in a readback: on the Vulkan backend `submit` only
    // records and queues, so without a fence this would time submission and
    // could return before the peak allocation has even happened.
    let run = || {
        let t = Instant::now();
        // The arena the model's own forward replays its blocks in.
        let out = {
            let arena = cx.gpu.scratch_scope_in("minimaxh3 block");
            block_forward(&cx, &w, &BlockInputs { x: &x, adaln_indices: &adaln_idx, cos: &cos, sin: &sin, modulation: &[&modulation], seq_len: seq, levels: ActLevels::One, scope: Some(&arena) }, &cfg).0
        };
        let probe = cx.gpu.read(&out, 1);
        assert!(probe[0].is_finite(), "non-finite block output");
        t.elapsed().as_secs_f64()
    };
    let warm = run();
    let best = (0..reps).map(|_| run()).fold(f64::INFINITY, f64::min);
    let flops = 2.0 * seq as f64 * (4.0 * inner as f64 * hidden as f64 + 3.0 * ffn as f64 * hidden as f64) + 4.0 * seq as f64 * seq as f64 * inner as f64;
    println!("one block forward: warm-up {warm:.3}s, best of {reps} {best:.3}s ({:.2} TFLOP/s on the linears + attention)", flops / best / 1e12);

    // Where one forward's DEVICE time goes, per kernel (timestamp queries,
    // not host wall clock): the device prints its `--kernel-profile` table
    // when it is dropped.
    if cx.gpu.set_kernel_timing(true) {
        cx.gpu.reset_kernel_times();
        run();
    }
}
