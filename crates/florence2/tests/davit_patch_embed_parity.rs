// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Per-stage parity: `PatchEmbed::forward_{from_pixels,from_sequence}`
//! against the real `microsoft/Florence-2-base` checkpoint, compared to
//! goldens captured directly from `davit.convs[i]`'s forward hook (see
//! `tools/goldens/florence2_dump_reference.py`) - not the whole block
//! stack, so this isolates the conv+LayerNorm piece from the (not yet
//! implemented) attention blocks.
//!
//! Gated on `FLORENCE2_DIR` (real checkpoint) and the golden fixture under
//! `testdata/florence2/` (gitignored, regenerate with the dump script) -
//! skips cleanly when either is absent.

use std::collections::HashMap;

use gpu_core::Gpu;
use paramstore::{ParamStore, Role};
use vision::ids::ConvKernelIds;
use vision::net::Shape;

use florence2::vision::{pipelines::PIPELINES, DavitConfig, PatchEmbed, PatchEmbedKernelIds};

const DAVIT_LN_EPS: f32 = 1e-5; // torch.nn.LayerNorm's default - DaViT's own config never overrides it.

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
fn davit_patch_embed_matches_real_checkpoint_per_stage() {
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

    let mut roles = Vec::new();
    for i in 0..4 {
        for suffix in ["proj.weight", "proj.bias", "norm.weight", "norm.bias"] {
            let name = format!("vision_tower.convs.{i}.{suffix}");
            let numel = source.get(&name).unwrap_or_else(|| panic!("missing checkpoint tensor {name}")).1.len();
            roles.push((name, numel, Role::Frozen));
        }
    }

    let gpu = Gpu::new_cpu(PIPELINES);
    let ps = ParamStore::new_with_roles_src(&gpu, roles, &source);
    let conv_ids = ConvKernelIds::resolve(PIPELINES);
    let k = PatchEmbedKernelIds::resolve(PIPELINES);
    let cfg = DavitConfig::florence2_base();

    // Each stage's conv, in the real model, consumes the PREVIOUS stage's
    // BLOCK output (`stageN` in the golden) - not the previous stage's
    // conv-only output. Stage 0 is the exception (raw pixels). Feeding the
    // real per-stage golden as input isolates the patch-embed piece from
    // the attention blocks (not yet implemented), instead of compounding
    // "conv on the wrong input" with any future block bug.
    let mut failures = Vec::new();

    for (i, stage) in cfg.stages.iter().enumerate() {
        let (in_shape, in_host): (Shape, &[f32]) = if i == 0 {
            (Shape::new(1, 3, 768, 768), &gmap["pixel_values"].data)
        } else {
            let prev = &cfg.stages[i - 1];
            (Shape::new(1, prev.dim_out, prev.out_hw.0, prev.out_hw.1), &gmap[&format!("stage{}", i - 1)].data)
        };

        let pe = PatchEmbed::new(&gpu, &conv_ids, &format!("vision_tower.convs.{i}"), in_shape, &stage.patch, stage.dim_out, DAVIT_LN_EPS, false);

        let in_buf = gpu.storage(in_host.len() as u64);
        gpu.write_f32(&in_buf, in_host);

        let out: Vec<f32> = if i == 0 {
            let out_ref = pe.forward_from_pixels(&gpu, &k, &ps, &in_buf);
            gpu.poll_wait();
            gpu.read(out_ref, (pe.tokens() * stage.dim_out) as usize)
        } else {
            let out_ref = pe.forward_from_sequence(&gpu, &k, &ps, &in_buf, in_shape);
            gpu.poll_wait();
            gpu.read(out_ref, (pe.tokens() * stage.dim_out) as usize)
        };

        let want = &gmap[&format!("conv{i}")].data;
        let c = cosine(&out, want);
        if c.is_nan() || c < 0.999 {
            failures.push(format!("conv{i}: cosine {c:.6} (want >= 0.999), got[0..4]={:?} want[0..4]={:?}", &out[..4.min(out.len())], &want[..4.min(want.len())]));
        }
    }

    assert!(failures.is_empty(), "patch-embed parity failures:\n{}", failures.join("\n"));
}
