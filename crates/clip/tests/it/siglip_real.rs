// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements reference-parity vision encoder ports for its
// clients. If your team needs expertise in porting vision-language models to
// custom inference engines then you can procure our services by sending an
// email to info@swedishembedded.com.

//! SigLIP-L/16@384 on the REAL checkpoints, through the production importer.
//!
//! * **DeepSeek-VL-7B-chat's low-resolution tower**
//!   (`vision_model.vision_tower_low.vision_tower.*`, fp16 safetensors) against
//!   the reference features `tools/goldens/siglip_dump_reference.py` dumps from
//!   the same weights: three block outputs and the post-norm features, each at
//!   cosine >= 0.9999 in fp32.
//! * **Janus-Pro-7B's understanding tower** (`vision_model.vision_tower.*`,
//!   bf16 inside `pytorch_model-*.bin`): the same builder and config, so the
//!   gate is the importer's two-way coverage over the real file plus a finite,
//!   correctly shaped forward - there is no Janus golden to compare against.
//!
//! Both skip (through `brain_testutil::skip`) when the checkpoint or the golden
//! is absent.

use std::path::Path;

use checkpoint::weightio::WeightReader;
use clip::config::ClipVisionConfig;
use clip::import::siglip::{import_timm, DEEPSEEK_VL_LOW_PREFIX, JANUS_PREFIX, UNUSED_ATTN_POOL};
use clip::model::{ClipVision, PatchSource, CLIP_VISION_PIPELINES};

const DUMPER: &str = "tools/goldens/siglip_dump_reference.py";
const COS_FLOOR: f64 = 0.9999;

/// The tower's identity as the checkpoint itself states it: width off the
/// position table, depth off the block indices present under `prefix`.
fn identity(rd: &WeightReader, prefix: &str) -> Option<(i64, i64)> {
    let width = *rd.shape(&format!("{prefix}pos_embed"))?.last()? as i64;
    let layers = rd
        .names()
        .filter_map(|n| n.strip_prefix(prefix)?.strip_prefix("blocks.")?.split_once('.')?.0.parse::<i64>().ok())
        .max()?
        + 1;
    Some((width, layers))
}

#[test]
fn deepseek_vl_low_res_tower_matches_the_reference() {
    let Some(dir) = brain_testutil::model_dir("deepseek-ai/deepseek-vl-7b-chat").filter(|d| Path::new(d).is_dir()) else {
        return brain_testutil::skip("deepseek-ai/deepseek-vl-7b-chat is not in the model store");
    };
    let golden = brain_testutil::testdata_path("siglip/deepseek_vl_low");
    let Some(src) = brain_testutil::golden::Source::open(&golden, DUMPER) else { return };
    let rd = WeightReader::open_hf_dir(Path::new(&dir)).expect("open the DeepSeek-VL checkpoint");
    let (width, layers) = identity(&rd, DEEPSEEK_VL_LOW_PREFIX).expect("the low-res tower is in the checkpoint");
    if !src.require(&[("width", width), ("layers", layers)]) {
        return;
    }
    let read = |f: &str| brain_testutil::read_f32(golden.join(f)).unwrap_or_else(|| panic!("golden {f} missing - re-run {DUMPER}"));

    let cfg = ClipVisionConfig::siglip_large_patch16_384();
    let (w, rep) = import_timm(&rd, DEEPSEEK_VL_LOW_PREFIX, &cfg).expect("import the low-res tower");
    assert_eq!(rep.skipped.len(), UNUSED_ATTN_POOL.len(), "exactly the MAP head is skipped: {:?}", rep.skipped);
    assert_eq!(rep.mapped, cfg.tensor_manifest().len());
    drop(rd);

    let m = ClipVision::new_on(gpu_core::testgpu::dev(CLIP_VISION_PIPELINES), cfg, 1, PatchSource::Pixels, &w);
    drop(w);
    m.set_pixels(&read("pixels.bin"));
    m.forward();

    let mut worst = 1.0f64;
    let mut check = |name: &str, got: Vec<f32>, want: Vec<f32>| {
        let (cos, max_abs) = brain_testutil::parity::compare(&got, &want);
        let rel = brain_testutil::parity::rel_l2(&got, &want);
        eprintln!("  siglip {name:<10} cosine {cos:.7}  max_abs {max_abs:.3e}  rel_l2 {rel:.3e}");
        worst = worst.min(cos);
        assert!(cos >= COS_FLOOR, "{name}: cosine {cos:.7} < {COS_FLOOR} (max_abs {max_abs:.3e})");
    };
    for l in [0usize, 11, 23] {
        check(&format!("block_{l:02}"), m.read_block_out(l), read(&format!("block_{l:02}.bin")));
    }
    check("features", m.read_output(), read("features.bin"));
    eprintln!("DeepSeek-VL low-res SigLIP parity: worst cosine {worst:.7}");
}

#[test]
fn janus_pro_understanding_tower_imports_with_two_way_coverage() {
    let Some(dir) = brain_testutil::model_dir("deepseek-ai/Janus-Pro-7B").filter(|d| Path::new(d).is_dir()) else {
        return brain_testutil::skip("deepseek-ai/Janus-Pro-7B is not in the model store");
    };
    let rd = WeightReader::open_hf_dir(Path::new(&dir)).expect("open the Janus-Pro checkpoint");
    let cfg = ClipVisionConfig::siglip_large_patch16_384();
    assert_eq!(identity(&rd, JANUS_PREFIX), Some((cfg.d_model() as i64, cfg.layers() as i64)));
    let (w, rep) = import_timm(&rd, JANUS_PREFIX, &cfg).expect("import the understanding tower");
    assert_eq!(rep.skipped.len(), UNUSED_ATTN_POOL.len(), "exactly the MAP head is skipped: {:?}", rep.skipped);
    assert_eq!(rep.mapped, cfg.tensor_manifest().len());
    drop(rd);

    let px = clip::init::fixed_pixels(&cfg, 1, 7);
    let m = ClipVision::new_on(gpu_core::testgpu::dev(CLIP_VISION_PIPELINES), cfg.clone(), 1, PatchSource::Pixels, &w);
    m.set_pixels(&px);
    m.forward();
    let out = m.read_output();
    assert_eq!(out.len(), (cfg.native_patches() * cfg.d_model()) as usize);
    assert!(out.iter().all(|v| v.is_finite()), "the Janus tower produced a non-finite feature");
}
