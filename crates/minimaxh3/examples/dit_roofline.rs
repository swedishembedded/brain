// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-kernel roofline probe for the MiniMax-H3 DiT at its REAL per-layer
//! shapes (`hidden=5376`, `inner=7168`, `ffn=14336`, 56 heads x 128) but a
//! caller-chosen sequence length and ONE block's worth of weights.
//!
//! Why this and not the end-to-end `generate_t2va` example: a real forward is
//! 50 blocks over a ~33B-param fp32 checkpoint and takes minutes, which is
//! useless as a profiling inner loop. Every op the DiT block runs is dispatched
//! here in isolation at the same shape it runs at inside
//! `crate::block::block_forward`, so the measured per-op cost multiplies out to
//! a predicted per-forward cost - and the prediction is checkable against a
//! real run, which is the whole point: measure where the time goes, never
//! assume it.
//!
//! Usage: `dit_roofline [seq_len] [reps]` (default 2560, 3). `BRAIN_DEVICE`
//! selects the backend, exactly as the real pipeline does.
//!
//! Swedish Embedded AB implements measurement-driven inference optimization for
//! its clients. If your team needs expertise in profiling and accelerating
//! large transformer inference on CPU and GPU, you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::time::Instant;

use minimaxh3::block::Ctx;

const K_MATMUL: usize = 0;
const K_RMSNORM_EPS: usize = 2;
const K_ROPE2D_PARTIAL: usize = 3;
const K_ATTN_SCORES_QK: usize = 4;
const K_ATTN_SOFTMAX_BIDIR: usize = 5;
const K_ATTN_APPLY_FULL: usize = 6;
const K_SILU_MUL: usize = 7;
const K_EMBED: usize = 8;
const K_GATE_ROW: usize = 9;
const K_MUL: usize = 10;
const K_ADD2: usize = 11;

/// Real `H3TransformerConfig::minimax_h3` proportions, restated here so the
/// probe cannot silently drift onto the tiny config's shapes.
const HIDDEN: u32 = 5376;
const HEADS: u32 = 56;
const HEAD_DIM: u32 = 128;
const INNER: u32 = HEADS * HEAD_DIM; // 7168
const FFN: u32 = 14336;
const TIME_EMBED_DIM: u32 = 2688;
const NUM_LAYERS: u32 = 50;
const MODALITY_NUM: u32 = 3;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let seq: u32 = a.get(1).map(|s| s.parse().expect("seq_len")).unwrap_or(2560);
    let reps: u32 = a.get(2).map(|s| s.parse().expect("reps")).unwrap_or(3);
    let device = std::env::var("BRAIN_DEVICE").ok();
    let cx = Ctx::new(device.as_deref());
    println!("device={:?} seq_len={seq} reps={reps} threads={}", device, std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
    println!();

    // One buffer per distinct shape the block needs. Values are irrelevant to
    // timing but must be finite, so a cheap deterministic ramp is enough.
    let fill = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect() };
    let x_hidden = cx.upload(&fill((seq * HIDDEN) as usize));
    let x_inner = cx.upload(&fill((seq * INNER) as usize));
    let x_ffn = cx.upload(&fill((seq * FFN) as usize));
    let w_qkv = cx.upload(&fill((INNER * HIDDEN) as usize));
    let w_o = cx.upload(&fill((HIDDEN * INNER) as usize));
    let w_fc1 = cx.upload(&fill((FFN * HIDDEN) as usize));
    let w_fc2 = cx.upload(&fill((HIDDEN * FFN) as usize));
    let w_norm = cx.upload(&fill(HIDDEN as usize));
    let w_hnorm = cx.upload(&fill(HEAD_DIM as usize));
    let half = 3 * 16u32;
    let cos = cx.upload(&fill((seq * half) as usize));
    let sin = cx.upload(&fill((seq * half) as usize));
    let idx = cx.upload_u32(&(0..seq).map(|i| i % (MODALITY_NUM * 2)).collect::<Vec<u32>>());
    let table = cx.upload(&fill((MODALITY_NUM * 2 * HIDDEN) as usize));

    let mut rows: Vec<(String, f64, f64, u32)> = Vec::new();
    // (label, seconds per call, FLOPs per call, calls per block)
    // Every timed region ENDS IN A READBACK, and that is not decoration. On
    // the Vulkan backend `submit` only records and queues work; without a
    // fence the loop below measures command submission, not execution. Timed
    // without this sync, the P40 reported 57 TFLOP/s for a 5376x7168 GEMM -
    // roughly 5x the card's entire fp32 peak, i.e. an obviously impossible
    // number that a less suspicious harness would have reported as a win.
    // `read` waits for the queue, so timing `reps` submissions followed by one
    // readback and dividing measures real work on both backends (on CPU the
    // dispatch is synchronous anyway and the readback is a no-op cost).
    //
    // MINIMUM over the trials, never the mean: this host may be running other
    // work, and a mean folds every unrelated stall into the number while the
    // minimum reports the closest thing to an uncontended run that was
    // actually observed.
    const TRIALS: u32 = 3;
    let sync = |b: &gpu_core::DeviceBuffer| {
        std::hint::black_box(cx.gpu.read(b, 1));
    };
    let mut bench = |label: &str, flops: f64, per_block: u32, mut f: Box<dyn FnMut() -> gpu_core::DeviceBuffer + '_>| {
        sync(&f()); // warm
        let mut best = f64::INFINITY;
        for _ in 0..TRIALS {
            let t = Instant::now();
            let mut last = None;
            for _ in 0..reps {
                last = Some(f());
            }
            sync(last.as_ref().expect("reps must be >= 1"));
            best = best.min(t.elapsed().as_secs_f64() / reps as f64);
        }
        rows.push((label.to_string(), best, flops, per_block));
    };

    let mm = |m: u32, k: u32, n: u32, x: &gpu_core::DeviceBuffer, w: &gpu_core::DeviceBuffer| -> gpu_core::DeviceBuffer {
        let y = cx.gpu.storage((m * n) as u64);
        cx.gpu.submit(&[], &[cx.gpu.step(K_MATMUL, &[x, w, &y], &[m, k, n], m * n)]);
        y
    };

    let f2 = |m: u32, k: u32, n: u32| 2.0 * m as f64 * k as f64 * n as f64;

    bench("matmul q/k/v  [S,5376]x[7168,5376]T", f2(seq, HIDDEN, INNER), 3, Box::new(|| mm(seq, HIDDEN, INNER, &x_hidden, &w_qkv)));
    bench("matmul to_out [S,7168]x[5376,7168]T", f2(seq, INNER, HIDDEN), 1, Box::new(|| mm(seq, INNER, HIDDEN, &x_inner, &w_o)));
    bench("matmul fc1    [S,5376]x[14336,5376]T", f2(seq, HIDDEN, FFN), 2, Box::new(|| mm(seq, HIDDEN, FFN, &x_hidden, &w_fc1)));
    bench("matmul fc2    [S,14336]x[5376,14336]T", f2(seq, FFN, HIDDEN), 1, Box::new(|| mm(seq, FFN, HIDDEN, &x_ffn, &w_fc2)));

    let scores_n = (HEADS * seq * seq) as u64;
    bench(
        "attn_scores_qk",
        2.0 * HEADS as f64 * seq as f64 * seq as f64 * HEAD_DIM as f64,
        1,
        Box::new(|| {
            let s = cx.gpu.storage(scores_n);
            cx.gpu.submit(&[], &[cx.gpu.step(K_ATTN_SCORES_QK, &[&x_inner, &x_inner, &s], &[1, HEADS, seq, HEAD_DIM, INNER, 0, gpu_core::f(0.088388)], (HEADS * seq * seq) as u32)]);
            s
        }),
    );
    let scores = cx.gpu.storage(scores_n);
    bench(
        "attn_softmax_bidir",
        5.0 * HEADS as f64 * seq as f64 * seq as f64,
        1,
        Box::new(|| {
            let p = cx.gpu.storage(scores_n);
            cx.gpu.submit(&[], &[cx.gpu.step(K_ATTN_SOFTMAX_BIDIR, &[&scores, &p], &[1, HEADS, seq], HEADS * seq)]);
            p
        }),
    );
    bench(
        "attn_apply_full",
        2.0 * HEADS as f64 * seq as f64 * seq as f64 * HEAD_DIM as f64,
        1,
        Box::new(|| {
            let o = cx.gpu.storage((seq * INNER) as u64);
            cx.gpu.submit(&[], &[cx.gpu.step(K_ATTN_APPLY_FULL, &[&scores, &x_inner, &o], &[1, HEADS, seq, HEAD_DIM, INNER, INNER], HEADS * seq * HEAD_DIM)]);
            o
        }),
    );

    bench(
        "rmsnorm [S,5376]",
        4.0 * seq as f64 * HIDDEN as f64,
        2,
        Box::new(|| {
            let y = cx.gpu.storage((seq * HIDDEN) as u64);
            cx.gpu.submit(&[], &[cx.gpu.step(K_RMSNORM_EPS, &[&x_hidden, &w_norm, &y], &[HIDDEN, seq, gpu_core::f(1e-6)], seq)]);
            y
        }),
    );
    bench(
        "rmsnorm qk [S*56,128]",
        4.0 * seq as f64 * INNER as f64,
        2,
        Box::new(|| {
            let y = cx.gpu.storage((seq * INNER) as u64);
            cx.gpu.submit(&[], &[cx.gpu.step(K_RMSNORM_EPS, &[&x_inner, &w_hnorm, &y], &[HEAD_DIM, seq * HEADS, gpu_core::f(1e-6)], seq * HEADS)]);
            y
        }),
    );
    bench(
        "rope2d_partial",
        6.0 * seq as f64 * HEADS as f64 * half as f64,
        2,
        Box::new(|| {
            cx.gpu.submit(&[], &[cx.gpu.step(K_ROPE2D_PARTIAL, &[&x_inner, &cos, &sin], &[seq, HEADS, half, INNER, 0, seq, gpu_core::f(1.0), HEAD_DIM], seq * HEADS * half)]);
            x_inner.clone()
        }),
    );
    bench(
        "silu_mul [S,14336]",
        5.0 * seq as f64 * FFN as f64,
        1,
        Box::new(|| {
            let y = cx.gpu.storage((seq * FFN) as u64);
            cx.gpu.submit(&[], &[cx.gpu.step(K_SILU_MUL, &[&x_ffn, &x_ffn, &y], &[seq * FFN], seq * FFN)]);
            y
        }),
    );
    bench(
        "embed/gather [S,5376]",
        0.0,
        6,
        Box::new(|| {
            let y = cx.gpu.storage((seq * HIDDEN) as u64);
            cx.gpu.submit(&[], &[cx.gpu.step(K_EMBED, &[&idx, &table, &y], &[HIDDEN, seq], seq * HIDDEN)]);
            y
        }),
    );
    bench(
        "mul [S,5376]",
        seq as f64 * HIDDEN as f64,
        2,
        Box::new(|| {
            let y = cx.gpu.storage((seq * HIDDEN) as u64);
            cx.gpu.submit(&[], &[cx.gpu.step(K_MUL, &[&x_hidden, &x_hidden, &y], &[seq * HIDDEN], seq * HIDDEN)]);
            y
        }),
    );
    bench(
        "add2 [S,5376]",
        seq as f64 * HIDDEN as f64,
        4,
        Box::new(|| {
            let y = cx.gpu.storage((seq * HIDDEN) as u64);
            cx.gpu.submit(&[], &[cx.gpu.step(K_ADD2, &[&x_hidden, &x_hidden, &y], &[seq * HIDDEN], seq * HIDDEN)]);
            y
        }),
    );
    bench(
        "gate_row [S,5376]",
        2.0 * seq as f64 * HIDDEN as f64,
        2,
        Box::new(|| {
            let y = cx.gpu.storage((seq * HIDDEN) as u64);
            cx.gpu.submit(&[], &[cx.gpu.step(K_GATE_ROW, &[&x_hidden, &x_hidden, &x_hidden, &y], &[seq, HIDDEN, 1], seq * HIDDEN)]);
            y
        }),
    );

    // The AdaLN projection, exactly as `block::adaln_tables` runs it: 18 host
    // `linear_rows` calls per block over a `[6*hidden*3, time_embed_dim]`
    // weight. `num_timesteps` is 2 in a real t2va forward (video + audio each
    // carry their own sigma).
    let num_timesteps = 2usize;
    let adaln_w = fill((6 * HIDDEN * MODALITY_NUM * TIME_EMBED_DIM) as usize);
    let temb_silu = fill(num_timesteps * TIME_EMBED_DIM as usize);
    bench(
        "adaln_proj (host, 18x linear_rows)",
        2.0 * 18.0 * num_timesteps as f64 * HIDDEN as f64 * TIME_EMBED_DIM as f64,
        1,
        Box::new(|| {
            for modality in 0..MODALITY_NUM as usize {
                for param in 0..6usize {
                    let feat_row0 = (modality * 6 + param) * HIDDEN as usize;
                    let w_slice = &adaln_w[feat_row0 * TIME_EMBED_DIM as usize..(feat_row0 + HIDDEN as usize) * TIME_EMBED_DIM as usize];
                    let out = model::hostmath::linear_rows(&temb_silu, w_slice, num_timesteps, TIME_EMBED_DIM as usize, HIDDEN as usize);
                    std::hint::black_box(&out);
                }
            }
            w_norm.clone()
        }),
    );

    println!("{:<40} {:>11} {:>12} {:>6} {:>13}", "op", "ms/call", "GFLOP/s", "x/blk", "s/50 blocks");
    println!("{}", "-".repeat(88));
    let mut total = 0.0f64;
    for (label, s, flops, per_block) in &rows {
        let gfs = if *flops > 0.0 { flops / s / 1e9 } else { 0.0 };
        let block_total = s * *per_block as f64 * NUM_LAYERS as f64;
        total += block_total;
        println!("{:<40} {:>11.3} {:>12.1} {:>6} {:>13.1}", label, s * 1e3, gfs, per_block, block_total);
    }
    println!("{}", "-".repeat(88));
    println!("{:<40} {:>45.1} s   ({:.2} min)", "PREDICTED per-forward (50 blocks)", total, total / 60.0);
}
