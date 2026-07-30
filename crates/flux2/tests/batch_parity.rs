// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The batching gate: a batch-of-N `Flux2Model::forward_batch` must equal N
//! single `forward` calls **exactly**.
//!
//! Bit-identity (not "close enough") is the assertion, and it is a claim about
//! reduction order, not about luck:
//!
//! * the register-tiled GEMMs (`matmul_reg3`, `matmul_i8_dyn`) accumulate over
//!   K inside a fixed 128×128 output tile — `M` only adds tiles, it never
//!   changes an output element's summation order;
//! * `flash_attn_bidir_split` gives each `(sample, head, query-tile)` its own
//!   workgroup and takes `bsz` as a Param, so raising the batch adds
//!   workgroups and nothing else;
//! * LayerNorm / RMSNorm / SwiGLU / `gate_row` / the int8 quantizers are
//!   row-local, and per-sample modulation arrives through `gate_row`'s
//!   `rows_per_cond` condition groups and per-sample gamma/beta binding
//!   slices — the arithmetic per element is unchanged.
//!
//! If any of that stops holding, this test fails with a non-zero max_abs
//! rather than silently drifting a served image away from the latency path.
//!
//! Weight-free, toy dims (mirrors `tests/model_smoke.rs`), on the pooled test
//! device. Dims are chosen so every per-sample binding offset is a multiple of
//! 64 floats (the 256-byte `min_storage_buffer_offset_alignment`) — see the
//! guard in `Flux2Model::forward_batch`.

use flux2::{position_ids, Flux2Config, Flux2Model, Precision, Sample};

/// hidden / mlp_hidden / in_channels / txt_len*context_in_dim all multiples of
/// 64 floats, so the batched binding slices are 256-byte aligned on a GPU.
fn tiny_cfg() -> Flux2Config {
    Flux2Config {
        in_channels: 64,
        context_in_dim: 64,
        hidden: 64,
        n_heads: 2,
        depth_double: 2,
        depth_single: 2,
        mlp_ratio: 3.0, // mlp_hidden = 192
        axes_dim: [8, 8, 8, 8],
        txt_len: 64,
        ..Flux2Config::klein_4b()
    }
}

fn tiny_tensors(cfg: &Flux2Config) -> flux2::Tensors {
    let mut ts = flux2::Tensors::new();
    for (name, shape) in cfg.tensor_manifest() {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
        ts.insert(name, (shape, data));
    }
    ts
}

/// Deterministic per-sample latents / conditioning / timestep — different in
/// every field, so a batch that leaked one sample's state into another (a
/// mis-strided slab, a shared modulation vector, attention crossing samples)
/// cannot pass by coincidence.
fn sample_data(cfg: &Flux2Config, n_img: usize, b: usize) -> (Vec<f32>, Vec<f32>, f32) {
    let k = b as f32 + 1.0;
    let img: Vec<f32> = (0..n_img * cfg.in_channels).map(|i| (i as f32 * 0.7 * k).sin()).collect();
    let ctx: Vec<f32> = (0..cfg.txt_len * cfg.context_in_dim).map(|i| (i as f32 * 0.3 + k).cos()).collect();
    // distinct timesteps: the whole point of per-sample modulation groups
    let t = 0.15 + 0.2 * b as f32;
    (img, ctx, t)
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Batch of 3 (differing latents, prompts AND timesteps) == 3 single forwards,
/// bit for bit.
#[test]
fn batched_forward_is_bit_identical_to_single_forwards() {
    let cfg = tiny_cfg();
    let ts = tiny_tensors(&cfg);
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    let bsz = 3usize;
    let n_img = 4 * 4 + 2 * 2; // gen 4x4 + one 2x2 ref
    let n_pred = 16;
    let n_max = (cfg.txt_len + n_img) as u32;
    let ids = position_ids(cfg.txt_len, 4, 4, &[(2, 2)]);
    let data: Vec<(Vec<f32>, Vec<f32>, f32)> = (0..bsz).map(|b| sample_data(&cfg, n_img, b)).collect();

    // reference: one sample at a time, on a B=1 model (the latency path)
    let single = Flux2Model::new(&cfg, &ts, gpu.share(), n_max);
    let want: Vec<Vec<f32>> = data.iter().map(|(img, ctx, t)| single.forward(img, ctx, *t, &ids, n_pred)).collect();
    drop(single);

    let batched = Flux2Model::new_batched(&cfg, &ts, gpu, n_max, bsz as u32, Precision::F32);
    assert_eq!(batched.max_batch(), bsz as u32);
    let samples: Vec<Sample<'_>> = data.iter().map(|(img, ctx, t)| Sample { img_tokens: img, ctx, t: *t }).collect();
    let got = batched.forward_batch(&samples, &ids, n_pred);

    assert_eq!(got.len(), bsz);
    for (b, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.len(), n_pred * cfg.in_channels);
        assert!(g.iter().all(|v| v.is_finite()), "sample {b}: non-finite batched output");
        let m = max_abs(g, w);
        eprintln!("sample {b}: max_abs={m:e} cosine={:.9}", cosine(g, w));
        assert_eq!(m, 0.0, "sample {b}: batched forward differs from the single forward (max_abs {m:e})");
    }
    // guard against a degenerate pass: the samples must actually differ
    assert!(max_abs(&got[0], &got[1]) > 1e-6, "samples are identical — the fixtures are not exercising per-sample state");
}

/// A batch of ONE must still be exactly the unbatched forward — the latency
/// path shares this code and must not move.
#[test]
fn batch_of_one_equals_the_unbatched_forward() {
    let cfg = tiny_cfg();
    let ts = tiny_tensors(&cfg);
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    let n_img = 4 * 4;
    let n_max = (cfg.txt_len + n_img) as u32;
    let ids = position_ids(cfg.txt_len, 4, 4, &[]);
    let (img, ctx, t) = sample_data(&cfg, n_img, 0);

    let m1 = Flux2Model::new(&cfg, &ts, gpu.share(), n_max);
    let want = m1.forward(&img, &ctx, t, &ids, n_img);
    drop(m1);
    // a model sized for a batch, driven with one sample
    let m4 = Flux2Model::new_batched(&cfg, &ts, gpu, n_max, 4, Precision::F32);
    let got = m4.forward_batch(&[Sample { img_tokens: &img, ctx: &ctx, t }], &ids, n_img);
    assert_eq!(max_abs(&got[0], &want), 0.0, "B=1 on a batch-sized model diverged from the unbatched forward");
}

/// Same-timestep batching (the common continuous-batching case, where the
/// modulation is deduplicated on the host) is bit-identical too — the dedup
/// must not change which vector a sample reads.
#[test]
fn shared_timestep_batch_is_bit_identical() {
    let cfg = tiny_cfg();
    let ts = tiny_tensors(&cfg);
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    let bsz = 2usize;
    let n_img = 4 * 4;
    let n_max = (cfg.txt_len + n_img) as u32;
    let ids = position_ids(cfg.txt_len, 4, 4, &[]);
    let t = 0.42f32;
    let data: Vec<(Vec<f32>, Vec<f32>)> = (0..bsz)
        .map(|b| {
            let (i, c, _) = sample_data(&cfg, n_img, b);
            (i, c)
        })
        .collect();

    let single = Flux2Model::new(&cfg, &ts, gpu.share(), n_max);
    let want: Vec<Vec<f32>> = data.iter().map(|(img, ctx)| single.forward(img, ctx, t, &ids, n_img)).collect();
    drop(single);
    let batched = Flux2Model::new_batched(&cfg, &ts, gpu, n_max, bsz as u32, Precision::F32);
    let samples: Vec<Sample<'_>> = data.iter().map(|(img, ctx)| Sample { img_tokens: img, ctx, t }).collect();
    let got = batched.forward_batch(&samples, &ids, n_img);
    for (b, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(max_abs(g, w), 0.0, "sample {b}: shared-timestep batch diverged");
    }
}

/// The int8 (DP4A) tier batches too: the per-token activation scales and the
/// packed-activation scratch are row-indexed, so a batched forward quantizes
/// exactly the rows a single forward would. GPU only.
#[test]
fn int8_batched_forward_is_bit_identical() {
    let gpu = gpu_core::testgpu::dev(flux2::KERNELS);
    if !gpu.caps().workgroup_reductions {
        eprintln!("SKIP: int8 needs a GPU backend, current is {}", gpu.kind());
        return;
    }
    // int8 additionally needs the JOINT token count to be a multiple of 64
    // (the per-token scale buffer is sliced by row): 64 txt + 64 img.
    let cfg = tiny_cfg();
    let bsz = 2usize;
    let n_img = 8 * 8;
    let n_pred = n_img;
    let n_max = (cfg.txt_len + n_img) as u32;
    assert_eq!(n_max % 64, 0, "int8 batch slicing needs joint tokens % 64 == 0");
    let ts = tiny_tensors(&cfg);
    let ids = position_ids(cfg.txt_len, 8, 8, &[]);
    let data: Vec<(Vec<f32>, Vec<f32>, f32)> = (0..bsz).map(|b| sample_data(&cfg, n_img, b)).collect();

    let single = Flux2Model::new_with(&cfg, &ts, gpu.share(), n_max, Precision::Int8);
    let want: Vec<Vec<f32>> = data.iter().map(|(img, ctx, t)| single.forward(img, ctx, *t, &ids, n_pred)).collect();
    drop(single);
    let batched = Flux2Model::new_batched(&cfg, &ts, gpu, n_max, bsz as u32, Precision::Int8);
    let samples: Vec<Sample<'_>> = data.iter().map(|(img, ctx, t)| Sample { img_tokens: img, ctx, t: *t }).collect();
    let got = batched.forward_batch(&samples, &ids, n_pred);
    for (b, (g, w)) in got.iter().zip(&want).enumerate() {
        let m = max_abs(g, w);
        eprintln!("int8 sample {b}: max_abs={m:e}");
        assert_eq!(m, 0.0, "int8 sample {b}: batched forward differs (max_abs {m:e})");
    }
}
