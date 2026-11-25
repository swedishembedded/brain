// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements speech codec encoders that turn recorded
// corpora into training codes on GPUs for its clients. If your team needs
// expertise in batched neural audio coding then you can procure our services
// by sending an email to info@swedishembedded.com.

//! `Codec::encode_batch`: clips of different lengths encoded together in one
//! device pass each encode to exactly the codes they encode to alone. A tiny
//! random encoder (every tensor the encode path reads) small enough that the
//! padding, the per-clip masks before every strided conv and the per-clip
//! replicate edge of the downsample are all exercised on clips a few frames
//! long, including lengths one sample either side of a frame boundary.

use std::collections::HashMap;

use data::rng::{random_seed, Lcg};
use mimi::{Codec, CodecConfig, EncoderConfig};

/// Two SEANet stages (strides 3 and 2), one transformer layer, 1 semantic + 3
/// acoustic codebooks of 16 rows: 12 input samples per code frame.
fn tiny_cfg() -> CodecConfig {
    let enc = EncoderConfig {
        num_filters: 4,
        hidden_size: 8,
        kernel_size: 7,
        last_kernel_size: 3,
        residual_kernel_size: 3,
        compress: 2,
        dilation_growth_rate: 2,
        num_residual_layers: 1,
        upsampling_ratios: vec![2, 3],
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 4,
        intermediate_size: 16,
        rope_theta: 10000.0,
        norm_eps: 1e-5,
        codebook_dim: 4,
        codebook_size: 16,
        num_semantic_quantizers: 1,
        downsample_stride: 2,
        downsample_kernel: 4,
    };
    CodecConfig { num_quantizers: 4, num_semantic_quantizers: 1, enc, ..Default::default() }
}

/// Random tensors by name.
struct Weights<'a> {
    w: HashMap<String, Vec<f32>>,
    rng: &'a mut Lcg,
}

impl Weights<'_> {
    fn put(&mut self, name: String, n: u32, scale: f32) {
        let v = self.rng.vec_scaled(n as usize, scale);
        self.w.insert(name, v);
    }
    /// A conv's `[cout, cin, k]` weight and its bias.
    fn conv(&mut self, prefix: String, cout: u32, cin: u32, k: u32) {
        self.put(format!("{prefix}.conv.weight"), cout * cin * k, 1.0 / ((cin * k) as f32).sqrt());
        self.put(format!("{prefix}.conv.bias"), cout, 0.2);
    }
}

/// Every tensor `Codec::encode` reads, random.
fn tiny_weights(cfg: &CodecConfig, rng: &mut Lcg) -> HashMap<String, Vec<f32>> {
    let e = &cfg.enc;
    let mut w = Weights { w: HashMap::new(), rng };
    w.conv("encoder.encoder.layers.0".into(), e.num_filters, 1, e.kernel_size);
    let mut dim = e.num_filters;
    let mut li = 1;
    for &r in e.upsampling_ratios.iter().rev() {
        let hidden = dim / e.compress;
        w.conv(format!("encoder.encoder.layers.{li}.block.1"), hidden, dim, e.residual_kernel_size);
        w.conv(format!("encoder.encoder.layers.{li}.block.3"), dim, hidden, 1);
        li += 2;
        w.conv(format!("encoder.encoder.layers.{li}"), dim * 2, dim, 2 * r);
        li += 1;
        dim *= 2;
    }
    w.conv(format!("encoder.encoder.layers.{}", li + 1), e.hidden_size, dim, e.last_kernel_size);
    let (d, hq, ff) = (e.hidden_size, e.num_attention_heads * e.head_dim, e.intermediate_size);
    for layer in 0..e.num_hidden_layers {
        let p = |leaf: &str| format!("encoder.encoder_transformer.layers.{layer}.{leaf}");
        for (leaf, n, scale) in [
            ("self_attn.q_proj.weight", hq * d, 0.5),
            ("self_attn.k_proj.weight", hq * d, 0.5),
            ("self_attn.v_proj.weight", hq * d, 0.5),
            ("self_attn.o_proj.weight", d * hq, 0.5),
            ("mlp.fc1.weight", ff * d, 0.5),
            ("mlp.fc2.weight", d * ff, 0.3),
            ("input_layernorm.bias", d, 0.1),
            ("post_attention_layernorm.bias", d, 0.1),
            ("self_attn_layer_scale.scale", d, 0.5),
            ("mlp_layer_scale.scale", d, 0.5),
        ] {
            w.put(p(leaf), n, scale);
        }
        for leaf in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            w.w.insert(p(leaf), vec![1.0; d as usize]);
        }
    }
    let k = e.downsample_kernel;
    w.put("encoder.downsample.conv.weight".into(), d * d * k, 1.0 / ((d * k) as f32).sqrt());
    for (group, layers) in [("semantic", e.num_semantic_quantizers), ("acoustic", cfg.num_quantizers - e.num_semantic_quantizers)] {
        let prefix = format!("encoder.quantizer.{group}_residual_vector_quantizer");
        w.put(format!("{prefix}.input_proj.weight"), e.codebook_dim * d, 0.5);
        for q in 0..layers {
            w.put(format!("{prefix}.layers.{q}.table"), e.codebook_size * e.codebook_dim, 1.0);
        }
    }
    w.w
}

/// Frames a clip of `samples` encodes to: the ceil chain of every stride.
fn frames(cfg: &CodecConfig, samples: usize) -> usize {
    let strides = cfg.enc.upsampling_ratios.iter().chain(std::iter::once(&cfg.enc.downsample_stride));
    strides.fold(samples, |l, &s| l.div_ceil(s as usize))
}

#[test]
fn a_batch_encodes_each_clip_to_the_codes_it_encodes_to_alone() {
    let cfg = tiny_cfg();
    let mut rng = Lcg::new(random_seed());
    let codec = Codec::from_weights_on(gpu_core::testgpu::dev(mimi::PIPELINES), cfg.clone(), tiny_weights(&cfg, &mut rng));
    // 12 samples per frame: lengths on, beside and between frame boundaries,
    // the longest last and in the middle so the sort reorders them.
    let lengths = [1usize, 11, 12, 13, 47, 300, 121, 120, 23];
    let clips: Vec<Vec<f32>> = lengths.iter().map(|&n| rng.vec_scaled(n, 0.8)).collect();
    let refs: Vec<&[f32]> = clips.iter().map(Vec::as_slice).collect();

    let batch = codec.encode_batch(&refs);
    assert_eq!(batch.len(), clips.len());
    let nq = cfg.num_quantizers as usize;
    for (clip, got) in clips.iter().zip(&batch) {
        assert_eq!(got.len(), frames(&cfg, clip.len()) * nq, "{} samples: frame count", clip.len());
        assert!(got.iter().all(|&c| c < cfg.enc.codebook_size), "{} samples: a code out of range", clip.len());
        assert_eq!(got, &codec.encode(clip), "{} samples: the batched codes differ from the clip's own", clip.len());
    }
    let distinct: std::collections::BTreeSet<u32> = batch.iter().flatten().copied().collect();
    assert!(distinct.len() > 4, "the fixture is degenerate: {} distinct codes", distinct.len());
}
