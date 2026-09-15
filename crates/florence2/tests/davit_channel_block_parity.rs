// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Stage 0's `ChannelBlock` (dwconv residual, channel attention, dwconv
//! residual, MLP) against the real checkpoint: input is
//! `stage0_spatial_block`'s golden (the SpatialBlock's real output,
//! already parity-checked in `davit_spatial_block_parity.rs`), expected
//! output is `stage0_channel_block`'s golden - captured via a forward hook
//! directly on `davit.blocks[0][0].channel_block`, i.e. the pair's real
//! final output, which is also stage0's golden.

use std::collections::HashMap;

use gpu_core::Gpu;
use paramstore::{ParamStore, Role};

use florence2::vision::{pipelines::PIPELINES, ChannelBlock, ChannelBlockKernelIds};

const DAVIT_LN_EPS: f32 = 1e-5;
const STAGE0_N: u32 = 192 * 192; // spatial token count feeding the N^-0.5 channel-attn scale.

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

#[test]
fn stage0_channel_block_matches_real_checkpoint() {
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
    let golden = checkpoint::safetensors::read(&golden_path).expect("read golden");
    let gmap: HashMap<String, checkpoint::safetensors::StTensor> = golden.into_iter().map(|t| (t.name.clone(), t)).collect();

    let prefix = "vision_tower.blocks.0.0.channel_block";
    let dim = 128u32;
    let scale = 1.0 / (STAGE0_N as f32).sqrt();

    // Pre-fold channel attention's 1/sqrt(N) scale into the qkv weight's Q
    // rows/bias entries - see ChannelAttn::new's doc for why this is the
    // caller's job, not the module's.
    {
        let w_name = format!("{prefix}.channel_attn.fn.qkv.weight");
        let (shape, data) = source.get_mut(&w_name).unwrap_or_else(|| panic!("missing {w_name}"));
        assert_eq!(shape[0] as u32, 3 * dim, "qkv weight rows");
        let in_dim = shape[1];
        for row in data[..(dim as usize) * in_dim].iter_mut() {
            *row *= scale;
        }
        let b_name = format!("{prefix}.channel_attn.fn.qkv.bias");
        let (_, bdata) = source.get_mut(&b_name).unwrap_or_else(|| panic!("missing {b_name}"));
        for v in bdata[..dim as usize].iter_mut() {
            *v *= scale;
        }
    }

    let names = [
        "conv1.fn.dw.weight",
        "conv1.fn.dw.bias",
        "conv2.fn.dw.weight",
        "conv2.fn.dw.bias",
        "channel_attn.norm.weight",
        "channel_attn.norm.bias",
        "channel_attn.fn.qkv.weight",
        "channel_attn.fn.qkv.bias",
        "channel_attn.fn.proj.weight",
        "channel_attn.fn.proj.bias",
        "ffn.norm.weight",
        "ffn.norm.bias",
        "ffn.fn.net.fc1.weight",
        "ffn.fn.net.fc1.bias",
        "ffn.fn.net.fc2.weight",
        "ffn.fn.net.fc2.bias",
    ];
    let mut roles = Vec::new();
    for n in names {
        let full = format!("{prefix}.{n}");
        let numel = source.get(&full).unwrap_or_else(|| panic!("missing checkpoint tensor {full}")).1.len();
        roles.push((full, numel, Role::Frozen));
    }

    let gpu = Gpu::new_cpu(PIPELINES);
    let ps = ParamStore::new_with_roles_src(&gpu, roles, &source);
    let k = ChannelBlockKernelIds::resolve(PIPELINES);

    // Stage 0: dim=128, groups=4, grid 192x192, mlp_ratio=4.
    let block = ChannelBlock::new(&gpu, &k, prefix, dim, 4, 192, 192, 4, DAVIT_LN_EPS, false);

    let x_host = &gmap["stage0_spatial_block"].data;
    let x_buf = gpu.storage(x_host.len() as u64);
    gpu.write_f32(&x_buf, x_host);

    let out_ref = block.forward(&gpu, &k, &ps, &x_buf);
    gpu.poll_wait();
    let out = gpu.read(out_ref, x_host.len());

    let want = &gmap["stage0_channel_block"].data;
    let c = cosine(&out, want);
    assert!(c >= 0.999, "cosine {c:.6} (want >= 0.999); got[0..4]={:?} want[0..4]={:?}", &out[..4], &want[..4]);
}
