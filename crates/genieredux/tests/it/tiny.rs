// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for verifying neural-network ports
// without their checkpoints for its clients. If your team needs expertise in
// parity-gated model porting then you can procure our services by sending an
// email to info@swedishembedded.com.

//! A tiny random GenieRedux (tokenizer + dynamics) shared by the tests that
//! need real weights' structure but not real weights: 8x8 frames, 4x4 patches
//! (a 2x2 token grid), 16 codes. The weight streams are fixed per call site so a
//! failure reproduces; they are test inputs, not defaults of the library.

use genieredux::bias::CpbLayer;
use genieredux::import::{DynamicsConfig, TokenizerConfig};
use genieredux::{
    ff_inner, AttnWeights, DynamicsWeights, FfWeights, PatchEmbedWeights, PegWeights,
    StBlockWeights, StTransformerWeights, ToPixelsWeights, TokenizerWeights, VqWeights,
};

pub const GRID: (u32, u32) = (2, 2);
pub const HW: usize = 4;
pub const FRAME_LEN: usize = 3 * 8 * 8;

pub fn rand(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (((z ^ (z >> 31)) >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}
fn ones_plus(seed: u64, n: usize) -> Vec<f32> {
    rand(seed, n).iter().map(|v| v + 1.0).collect()
}
pub fn attn(dim: usize, inner: usize, hd: usize, s: u64) -> AttnWeights {
    AttnWeights {
        norm_gamma: ones_plus(s, dim),
        to_q: rand(s + 1, inner * dim),
        to_k: rand(s + 2, inner * dim),
        to_v: rand(s + 3, inner * dim),
        to_out: rand(s + 4, dim * inner),
        q_scale: ones_plus(s + 5, hd),
        k_scale: ones_plus(s + 6, hd),
    }
}
pub fn ff(dim: usize, inner: usize, s: u64) -> FfWeights {
    FfWeights {
        norm_gamma: ones_plus(s, dim),
        norm_beta: rand(s + 7, dim),
        w_x: rand(s + 1, inner * dim),
        w_gate: rand(s + 2, inner * dim),
        w_out: rand(s + 3, dim * inner),
    }
}
pub fn peg(dim: usize, s: u64) -> PegWeights {
    PegWeights { dsconv: rand(s, dim * 27).iter().map(|v| v * 0.3).collect(), bias: rand(s + 1, dim) }
}
pub fn block(dim: usize, heads: usize, hd: usize, s: u64) -> StBlockWeights {
    let (inner, ffi) = (heads * hd, ff_inner(dim as u32) as usize);
    StBlockWeights {
        spatial_peg: peg(dim, s),
        spatial_attn: attn(dim, inner, hd, s + 10),
        spatial_ff: ff(dim, ffi, s + 20),
        temporal_peg: peg(dim, s + 30),
        temporal_attn: attn(dim, inner, hd, s + 40),
        temporal_ff: ff(dim, ffi, s + 50),
    }
}
pub fn stack(dim: usize, heads: usize, hd: usize, layers: usize, s: u64) -> StTransformerWeights {
    StTransformerWeights {
        layers: (0..layers).map(|i| block(dim, heads, hd, s + i as u64 * 100)).collect(),
        norm_out_gamma: ones_plus(s + 9000, dim),
    }
}
fn patch(pf: usize, dim: usize, s: u64) -> PatchEmbedWeights {
    PatchEmbedWeights {
        ln1_g: ones_plus(s, pf),
        ln1_b: rand(s + 1, pf),
        lin_w: rand(s + 2, dim * pf),
        lin_b: rand(s + 3, dim),
        ln2_g: ones_plus(s + 4, dim),
        ln2_b: rand(s + 5, dim),
    }
}
fn cpb(heads: usize, hidden: usize, s: u64) -> Vec<CpbLayer> {
    vec![
        CpbLayer { w: rand(s, hidden * 2), b: rand(s + 1, hidden), in_dim: 2, out_dim: hidden },
        CpbLayer { w: rand(s + 2, hidden * hidden), b: rand(s + 3, hidden), in_dim: hidden, out_dim: hidden },
        CpbLayer { w: rand(s + 4, heads * hidden), b: rand(s + 5, heads), in_dim: hidden, out_dim: heads },
    ]
}

pub fn tokenizer_config() -> TokenizerConfig {
    TokenizerConfig {
        dim: 16, heads: 2, head_dim: 8, ff_inner: ff_inner(16), code_dim: 8, n_codes: 16,
        patch: 4, channels: 3, enc_layers: 2, dec_layers: 2, cpb_hidden: 16,
    }
}

pub fn tokenizer() -> TokenizerWeights {
    let c = tokenizer_config();
    let (dim, heads, hd, cd, k) = (c.dim as usize, c.heads as usize, c.head_dim as usize, c.code_dim as usize, c.n_codes as usize);
    let pf = (c.channels * c.patch * c.patch) as usize;
    TokenizerWeights {
        patch_first: patch(pf, dim, 10),
        patch_rest: patch(pf, dim, 20),
        encoder: stack(dim, heads, hd, c.enc_layers, 1000),
        vq: VqWeights {
            project_in_w: rand(30, cd * dim), project_in_b: rand(31, cd),
            codebook: rand(32, k * cd), project_out_w: rand(33, dim * cd), project_out_b: rand(34, dim),
        },
        decoder: stack(dim, heads, hd, c.dec_layers, 5000),
        to_pixels_first: ToPixelsWeights { lin_w: rand(40, pf * dim), lin_b: rand(41, pf) },
        to_pixels_rest: ToPixelsWeights { lin_w: rand(42, pf * dim), lin_b: rand(43, pf) },
        cpb_net: cpb(heads, c.cpb_hidden as usize, 50),
    }
}

pub fn dynamics_config() -> DynamicsConfig {
    let dim2 = 16 + 3;
    DynamicsConfig {
        dim: 16, dim2, heads: 2, head_dim: 8, ff_inner: ff_inner(dim2), n_codes: 16, code_dim: 8,
        num_actions: 3, layers: 2, cpb_hidden: 16, max_seq_len: 64,
    }
}

/// Dynamics weights that share the tokenizer's codebook and `project_out`, as
/// the released checkpoint does.
pub fn dynamics(tok: &TokenizerWeights) -> DynamicsWeights {
    let c = dynamics_config();
    let (dim, dim2, heads, hd) = (c.dim as usize, c.dim2 as usize, c.heads as usize, c.head_dim as usize);
    DynamicsWeights {
        token_emb: rand(60, (c.n_codes as usize + 1) * dim),
        pos_emb: rand(61, c.max_seq_len as usize * dim),
        codebook: tok.vq.codebook.clone(),
        project_out_w: tok.vq.project_out_w.clone(),
        project_out_b: tok.vq.project_out_b.clone(),
        transformer: stack(dim2, heads, hd, c.layers, 7000),
        cpb_net: cpb(heads, c.cpb_hidden as usize, 80),
        to_logits_w: rand(90, c.n_codes as usize * dim2),
        to_logits_b: rand(91, c.n_codes as usize),
    }
}

/// `n` frames of pixels in `[0, 1]`, frame-major (`[n][3][8][8]`), the layout
/// `WorldModel::reset` takes.
pub fn frames(seed: u64, n: usize) -> Vec<f32> {
    rand(seed, n * FRAME_LEN).iter().map(|v| v * 0.5 + 0.5).collect()
}

/// Frame-major `[n][3][8][8]` to the tokenizer's channels-first video
/// `[3][n][8][8]`.
pub fn to_video(frames: &[f32], n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; frames.len()];
    for f in 0..n {
        for c in 0..3 {
            let src = (f * 3 + c) * 64;
            let dst = (c * n + f) * 64;
            v[dst..dst + 64].copy_from_slice(&frames[src..src + 64]);
        }
    }
    v
}

/// One-hot actions `[frames, num_actions]`.
pub fn one_hot(actions: &[u32], num_actions: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; actions.len() * num_actions];
    for (i, &a) in actions.iter().enumerate() {
        v[i * num_actions + a as usize] = 1.0;
    }
    v
}
