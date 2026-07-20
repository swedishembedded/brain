// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Chunk-invariance of `chunked_attn_fwd`: splitting the query rows into
//! chunks must not change the attention output (the S=3 WorldMirror global
//! attention is the first real multi-chunk user).

use gpu_core::Gpu;
use model::vit::{chunked_attn_fwd, VitKernelIds, VitShape};

const PIPES: &[(&str, &str)] = &[
    ("layernorm", kernels::LAYERNORM),
    ("matmul", kernels::MATMUL),
    ("bias_add", kernels::BIAS_ADD),
    ("gelu_erf", kernels::GELU_ERF),
    ("scale_chan", kernels::SCALE_CHAN),
    ("add2", kernels::ADD2),
    ("attn_scores_cross", kernels::ATTN_SCORES_CROSS),
    ("attn_softmax_cross", kernels::ATTN_SOFTMAX_CROSS),
    ("attn_apply_cross", kernels::ATTN_APPLY_CROSS),
    ("ln_head", kernels::LN_HEAD),
    ("rope2d", kernels::ROPE2D),
    ("matmul_rows", kernels::MATMUL_ROWS),
];

fn ids() -> VitKernelIds {
    VitKernelIds {
        layernorm: 0,
        matmul: 1,
        bias_add: 2,
        gelu_erf: 3,
        scale_chan: 4,
        add2: 5,
        attn_scores_cross: 6,
        attn_softmax_cross: 7,
        attn_apply_cross: 8,
        ln_head: 9,
        rope2d: 10,
        matmul_rows: 11,
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((self.0 >> 33) as f32 / (1u64 << 31) as f32) - 1.0) * 0.5
    }
}

fn run(chunk: u32, spans: &[(u32, u32)], rows: u32) -> Vec<f32> {
    let sh = VitShape { dim: 32, heads: 2, mlp: 64, eps: 1e-5 };
    let g = Gpu::new_cpu(PIPES);
    let k = ids();
    let mut r = Lcg(0xBEEF);
    let qkv_host: Vec<f32> = (0..rows as usize * 96).map(|_| r.next()).collect();
    let qkv = g.storage_init("qkv", &qkv_host);
    let ctx = g.storage(rows as u64 * 32);
    // slab big enough for the largest chunk
    let slab = 2 * rows as u64 * rows as u64;
    let scores = g.storage(slab);
    let probs = g.storage(slab);
    let mut steps = Vec::new();
    chunked_attn_fwd(&g, &k, &sh, &qkv, &ctx, &scores, &probs, spans, chunk, &mut steps);
    g.submit(&[&ctx], &steps);
    g.read(&ctx, rows as usize * 32)
}

#[test]
fn global_span_chunk_invariant() {
    let rows = 69u32; // 3 x 23, like a tiny multi-frame trunk
    let spans = [(0u32, rows)];
    let full = run(rows, &spans, rows);
    assert!(full.iter().all(|v| v.is_finite()), "single-chunk output has NaN");
    for chunk in [5u32, 23, 64, 68] {
        let split = run(chunk, &spans, rows);
        let max = full
            .iter()
            .zip(&split)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-6, "chunk {chunk}: max diff {max}");
    }
}

#[test]
fn frame_spans_match_reference() {
    // 3 per-frame spans == running each frame's rows through a single span.
    let td = 23u32;
    let rows = 3 * td;
    let spans: Vec<(u32, u32)> = (0..3).map(|f| (f * td, td)).collect();
    let multi = run(td, &spans, rows);
    assert!(multi.iter().all(|v| v.is_finite()), "multi-span output has NaN");
    for f in 0..3u32 {
        // frame f alone: same qkv content (deterministic LCG), span at 0
        let solo = run(td, &[(f * td, td)], rows);
        let a = &multi[(f * td * 32) as usize..((f + 1) * td * 32) as usize];
        let b = &solo[(f * td * 32) as usize..((f + 1) * td * 32) as usize];
        let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        assert!(max < 1e-6, "frame {f}: max diff {max}");
    }
}
