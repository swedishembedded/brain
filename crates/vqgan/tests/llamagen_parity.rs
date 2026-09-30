// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reference-parity image tokenizers for its
// clients. If your team needs a VQ image tokenizer ported and proven against
// its reference, you can procure our services by sending an email to
// info@swedishembedded.com.

//! LlamaGen VQ-16 parity on Janus-Pro-7B's real `gen_vision_model` weights.
//!
//! Goldens come from `tools/goldens/janus_vq16_dump_reference.py` (the
//! reference's own `vq_model.py`, CPU fp32) under `$BRAIN_TESTDATA/janus/vq16/`;
//! the weights are the store's `deepseek-ai/Janus-Pro-7B` download, read in
//! place. Either absent: the tests skip.
//!
//! Gates: decode of a fixed 2×(24×24) code grid at cosine ≥ 0.99999, and the
//! encode indices of a fixed 384² image with ZERO disagreements. The golden
//! records each query's margin to its runner-up code; the dump's smallest is
//! 8.4e-5 in squared distance, three orders above fp32 rounding of a unit-norm
//! distance, so any flipped index is a defect rather than a near tie.

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use brain_testutil::parity::WorstTable;
use brain_testutil::testdata;
use vae::blocks::Tensors;
use vqgan::{Vqgan, VqganConfig};

const DUMPER: &str = "tools/goldens/janus_vq16_dump_reference.py";
const PREFIX: &str = "gen_vision_model.";
const IMG: u32 = 384;

type Golden = HashMap<String, (Vec<usize>, Vec<f32>)>;

fn golden(name: &str) -> Option<Golden> {
    let dir = testdata("janus/vq16");
    let src = brain_testutil::golden::Source::open(Path::new(&dir), DUMPER)?;
    let cfg = VqganConfig::llamagen_vq16();
    if !src.require(&[
        ("codebook_size", cfg.codebook_size as i64),
        ("codebook_embed_dim", cfg.emb_dim as i64),
        ("z_channels", cfg.z_channels.unwrap_or(cfg.emb_dim) as i64),
        ("levels", cfg.ch_mult.len() as i64),
    ]) {
        return None;
    }
    let path = format!("{dir}/{name}");
    if !Path::new(&path).exists() {
        brain_testutil::skip(&format!("{path} absent (run {DUMPER})"));
        return None;
    }
    Some(
        checkpoint::safetensors::read(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"))
            .into_iter()
            .map(|t| (t.name, (t.shape, t.data)))
            .collect(),
    )
}

/// The imported VQ weights, read once per test binary (344 tensors out of a
/// 10 GB shard).
fn weights() -> Option<&'static Tensors> {
    static W: OnceLock<Option<Tensors>> = OnceLock::new();
    W.get_or_init(|| {
        let Some(dir) = brain_testutil::model_dir("deepseek-ai/Janus-Pro-7B") else {
            brain_testutil::skip("no model store root");
            return None;
        };
        if !Path::new(&dir).join("pytorch_model.bin.index.json").exists() {
            brain_testutil::skip(&format!("{dir}: Janus-Pro-7B not downloaded"));
            return None;
        }
        let cfg = VqganConfig::llamagen_vq16();
        let im = vqgan::import::load_hf_dir(Path::new(&dir), PREFIX, &cfg)
            .unwrap_or_else(|e| panic!("import {dir}: {e}"));
        assert_eq!(im.tensors.len(), cfg.tensor_manifest().len());
        assert_eq!(im.skipped, vec!["quantize.codebook_used"], "only the usage buffer is left over");
        Some(im.tensors)
    })
    .as_ref()
}

/// One GPU-heavy case at a time: both build a 384² graph with every
/// activation pinned for the stage taps.
fn heavy() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// The golden taps are named by LlamaGen module path; the model's by flat
/// block prefix.
fn tap_name(cfg: &VqganConfig, path: &str) -> Option<String> {
    cfg.llamagen_block_paths().into_iter().find(|(_, p)| p == path).map(|(flat, _)| flat)
}

/// Compare every golden stage tap the model also recorded (image 0 of the
/// batch; taps are `[n, ...]`).
fn add_taps(rep: &mut WorstTable, m: &Vqgan, cfg: &VqganConfig, g: &Golden, paths: &[&str]) {
    for path in paths {
        let Some((_, want)) = g.get(*path) else { continue };
        let flat = tap_name(cfg, path).unwrap_or_else(|| panic!("{path}: no block at that path"));
        let got = m.read_tap(&flat).unwrap_or_else(|| panic!("{flat} ({path}) was not tapped"));
        rep.add(&format!("{path} ({flat})"), &got[..want.len()], want);
    }
}

#[test]
fn janus_vq16_decode_matches_the_reference() {
    let Some(g) = golden("decode.safetensors") else { return };
    let Some(t) = weights() else { return };
    let _heavy = heavy();
    let cfg = VqganConfig::llamagen_vq16();
    let (cshape, codes) = &g["codes"];
    let n = cshape[0] as u32;
    let codes: Vec<u32> = codes.iter().map(|&v| v as u32).collect();
    let m = Vqgan::new_batched(cfg.clone(), t, IMG, IMG, gpu_core::testgpu::dev(&vqgan::KERNELS), true, n);
    let pixels = m.decode(&codes);

    let mut rep = WorstTable::new(56);
    add_taps(
        &mut rep,
        &m,
        &cfg,
        &g,
        &[
            "post_quant_conv",
            "decoder.conv_in",
            "decoder.mid.0",
            "decoder.mid.1",
            "decoder.mid.2",
            "decoder.conv_blocks.0.attn.2",
            "decoder.conv_blocks.0.upsample",
            "decoder.conv_blocks.1.upsample",
            "decoder.conv_blocks.2.upsample",
            "decoder.conv_out",
        ],
    );
    let want = &g["pixels"].1;
    let per = want.len() / n as usize;
    for i in 0..n as usize {
        rep.add(&format!("pixels[{i}]"), &pixels[i * per..(i + 1) * per], &want[i * per..(i + 1) * per]);
    }
    rep.finish("Janus-Pro VQ-16 decode (fp32)", 0.99999, 1e-4);
}

#[test]
fn janus_vq16_encode_indices_match_the_reference() {
    let Some(g) = golden("encode.safetensors") else { return };
    let Some(t) = weights() else { return };
    let _heavy = heavy();
    let cfg = VqganConfig::llamagen_vq16();
    let m = Vqgan::new(cfg.clone(), t, IMG, IMG, gpu_core::testgpu::dev(&vqgan::KERNELS), true);
    let (idx, _) = m.encode(&g["image"].1);

    let mut rep = WorstTable::new(56);
    add_taps(
        &mut rep,
        &m,
        &cfg,
        &g,
        &[
            "encoder.conv_in",
            "encoder.conv_blocks.0.downsample",
            "encoder.conv_blocks.1.downsample",
            "encoder.conv_blocks.2.downsample",
            "encoder.conv_blocks.3.downsample",
            "encoder.conv_blocks.4.attn.1",
            "encoder.mid.0",
            "encoder.mid.1",
            "encoder.mid.2",
            "encoder.conv_out",
            "quant_conv",
        ],
    );
    let z_norm = m.read_tap("z_norm").expect("the L2-normalised query rows are tapped");
    rep.add("z_norm", &z_norm, &g["z_norm"].1);

    let want: Vec<u32> = g["indices"].1.iter().map(|&v| v as u32).collect();
    let margin = &g["margin"].1;
    let flips: Vec<(usize, u32, u32, f32)> = idx
        .iter()
        .zip(&want)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| (i, *a, *b, margin[i]))
        .collect();
    println!("encode: {} queries, {} index disagreements {:?}", want.len(), flips.len(), &flips[..flips.len().min(8)]);
    rep.finish("Janus-Pro VQ-16 encode stages (fp32)", 0.99999, 1e-4);
    assert!(flips.is_empty(), "{} of {} encode indices disagree with the reference", flips.len(), want.len());
}
