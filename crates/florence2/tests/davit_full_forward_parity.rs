// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The FULL DaViT tower (all 4 stages' patch embeds + all 12 `(spatial,
//! channel)` block pairs) against the real checkpoint: `pixel_values` in,
//! compared to `unpooled` (== `stage3`, `davit.forward_features_unpool`'s
//! real output) - the strongest single check for M2, since every stage and
//! every block type has already been checked in isolation but never
//! chained end to end before this test.

use std::collections::HashMap;

use gpu_core::Gpu;
use paramstore::{ParamStore, Role};

use florence2::vision::{pipelines::PIPELINES, Davit, DavitConfig, DavitKernelIds};

const DAVIT_LN_EPS: f32 = 1e-5;
const MLP_RATIO: u32 = 4;

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Every tensor name the full DaViT tower needs, generated from the config
/// rather than hand-listed - 4 patch embeds x 4 tensors + 12 pairs x 2
/// block-types x 16 tensors = 400 names.
fn all_tensor_names(cfg: &DavitConfig) -> Vec<String> {
    let mut names = Vec::new();
    for (i, stage) in cfg.stages.iter().enumerate() {
        let p = format!("vision_tower.convs.{i}");
        for suffix in ["proj.weight", "proj.bias", "norm.weight", "norm.bias"] {
            names.push(format!("{p}.{suffix}"));
        }
        for pair in 0..stage.depth {
            for block in ["spatial_block", "channel_block"] {
                let bp = format!("vision_tower.blocks.{i}.{pair}.{block}");
                let attn = if block == "spatial_block" { "window_attn" } else { "channel_attn" };
                for suffix in [
                    "conv1.fn.dw.weight",
                    "conv1.fn.dw.bias",
                    "conv2.fn.dw.weight",
                    "conv2.fn.dw.bias",
                    "ffn.norm.weight",
                    "ffn.norm.bias",
                    "ffn.fn.net.fc1.weight",
                    "ffn.fn.net.fc1.bias",
                    "ffn.fn.net.fc2.weight",
                    "ffn.fn.net.fc2.bias",
                ] {
                    names.push(format!("{bp}.{suffix}"));
                }
                for suffix in ["norm.weight", "norm.bias", "fn.qkv.weight", "fn.qkv.bias", "fn.proj.weight", "fn.proj.bias"] {
                    names.push(format!("{bp}.{attn}.{suffix}"));
                }
            }
        }
    }
    names
}

#[test]
fn full_davit_forward_matches_real_checkpoint() {
    let Some(ckpt_dir) = std::env::var("FLORENCE2_DIR").ok() else {
        brain_testutil::skip("FLORENCE2_DIR unset");
        return;
    };
    let golden_path = brain_testutil::testdata("florence2/davit/stages.safetensors");
    if !std::path::Path::new(&golden_path).exists() {
        brain_testutil::skip("florence2 davit golden fixture absent - regenerate with tools/goldens/florence2_dump_reference.py");
        return;
    }

    let ckpt = checkpoint::safetensors::read(&format!("{ckpt_dir}/model.safetensors")).expect("read checkpoint");
    let mut source: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    for t in ckpt {
        source.insert(t.name, (t.shape, t.data));
    }

    let cfg = DavitConfig::florence2_base();

    // Pre-fold every channel-attn block's 1/sqrt(N) scale into its qkv
    // weight/bias Q rows - see ChannelAttn::new's doc.
    for (i, stage) in cfg.stages.iter().enumerate() {
        let n = stage.out_hw.0 * stage.out_hw.1;
        let scale = 1.0 / (n as f32).sqrt();
        for pair in 0..stage.depth {
            let prefix = format!("vision_tower.blocks.{i}.{pair}.channel_block.channel_attn.fn");
            let w_name = format!("{prefix}.qkv.weight");
            let (shape, data) = source.get_mut(&w_name).unwrap_or_else(|| panic!("missing {w_name}"));
            let in_dim = shape[1];
            for row in data[..(stage.dim_out as usize) * in_dim].iter_mut() {
                *row *= scale;
            }
            let b_name = format!("{prefix}.qkv.bias");
            let (_, bdata) = source.get_mut(&b_name).unwrap_or_else(|| panic!("missing {b_name}"));
            for v in bdata[..stage.dim_out as usize].iter_mut() {
                *v *= scale;
            }
        }
    }

    let roles: Vec<_> = all_tensor_names(&cfg)
        .into_iter()
        .map(|name| {
            let numel = source.get(&name).unwrap_or_else(|| panic!("missing checkpoint tensor {name}")).1.len();
            (name, numel, Role::Frozen)
        })
        .collect();

    let gpu = Gpu::new_cpu(PIPELINES);
    let ps = ParamStore::new_with_roles_src(&gpu, roles, &source);
    let k = DavitKernelIds::resolve(PIPELINES);
    let davit = Davit::new(&gpu, &k, "vision_tower", &cfg, MLP_RATIO, DAVIT_LN_EPS, false);

    let golden = checkpoint::safetensors::read(&golden_path).expect("read golden");
    let gmap: HashMap<String, checkpoint::safetensors::StTensor> = golden.into_iter().map(|t| (t.name.clone(), t)).collect();

    let px_host = &gmap["pixel_values"].data;
    let px_buf = gpu.storage(px_host.len() as u64);
    gpu.write_f32(&px_buf, px_host);

    let out_ref = davit.forward_features_unpool(&gpu, &k, &ps, &px_buf);
    gpu.poll_wait();
    let want = &gmap["unpooled"].data;
    let out = gpu.read(out_ref, want.len());

    let c = cosine(&out, want);
    assert!(c >= 0.999, "cosine {c:.6} (want >= 0.999); got[0..4]={:?} want[0..4]={:?}", &out[..4], &want[..4]);
}
