// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Real-weight parity for DeepSeek-VL's high-resolution SAM tower**
//! (`sam_b_downsample`: the SAM ViT-B tower plus the 96x96 neck resize and the
//! HD branch), against the pinned PyTorch reference module.
//!
//! The weights are the `vision_model.vision_tower_high.vision_tower.*` tensors
//! of `deepseek-ai/deepseek-vl-7b-chat`, loaded in place through
//! [`sam1::hf::source`]. The golden is
//! `tools/goldens/deepseek_vl_sam_dump_reference.py`'s fp32 CPU run of the
//! reference module on the same weights over a fixed synthetic 1024x1024
//! input, which the golden carries so this side replays it byte for byte.
//!
//! Both sides compute in fp32 from the same F16 weights, so the floor is fp32
//! round-off through twelve blocks, not a quantization gap: cosine >= 0.9999 on
//! every tap. The taps split the two paths so a failure names the one that
//! broke: the main neck and its resize, the HD neck and its resize, each path's
//! compressor output on its own (`main_out`, and `hd_out` before the scale),
//! and the combined output.
//!
//! Skips (via `brain_testutil::skip`, fatal under `BRAIN_REQUIRE_FIXTURES`)
//! when the checkpoint is not in the model store or the golden is absent from
//! `$BRAIN_TESTDATA/deepseek-vl/sam/`.

use brain_testutil::golden::Source;
use brain_testutil::parity::{load, Report};
use checkpoint::weightio::WeightReader;
use sam1::config::HD_ALPHA;
use sam1::hf::Spelling;
use sam1::{SamEncoder, SamViTConfig};

const REPO: &str = "deepseek-ai/deepseek-vl-7b-chat";
const PREFIX: &str = "vision_model.vision_tower_high.vision_tower.";
const DUMPER: &str = "tools/goldens/deepseek_vl_sam_dump_reference.py";
const FLOOR: f64 = 0.9999;

#[test]
fn deepseek_vl_sam_tower_matches_the_reference() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return brain_testutil::skip_unavailable("MOE_SKIP_GPU_TESTS set");
    }
    let Some(dir) = brain_testutil::model_dir(REPO).filter(|d| std::path::Path::new(d).join("model.safetensors.index.json").exists()) else {
        return brain_testutil::skip(&format!("{REPO} not in the model store (brain fetch {REPO})"));
    };
    let golden_dir = brain_testutil::testdata_path("deepseek-vl/sam");
    let golden_path = golden_dir.join("golden.safetensors");
    if !golden_path.exists() {
        return brain_testutil::skip(&format!("{} missing (run {DUMPER})", golden_path.display()));
    }
    let cfg = SamViTConfig::deepseek_vl();
    let Some(src) = Source::open(&golden_dir, DUMPER) else { return };
    let (gh, _) = cfg.compress_in_grid();
    if !src.require(&[
        ("d_model", cfg.d_model as i64),
        ("n_layers", cfg.n_layers as i64),
        ("neck_channels", cfg.neck_channels as i64),
        ("compress_out", cfg.compress_out as i64),
        ("resize", gh as i64),
    ]) {
        return;
    }
    let golden = load(&golden_path);

    let reader = WeightReader::open_hf_dir(std::path::Path::new(&dir)).expect("open the DeepSeek-VL checkpoint");
    let weights = sam1::hf::source(&reader, reader.names(), PREFIX, &cfg, Spelling::DeepseekVl).expect("two-way coverage of the SAM tower");
    let gpu = gpu_core::testgpu::dev(sam1::PIPELINES);
    let enc = SamEncoder::new_inference(gpu, cfg.clone(), &weights, 0);

    // The checkpoint's own scale, exactly, and what the reference ran with:
    // F16 0x1DB0 = 2^-8 * 1.421875 = 0.00555419921875, f32 bits 0x3BB60000.
    let alpha = enc.read_weight(HD_ALPHA);
    assert_eq!(alpha, golden["hd_alpha"].data, "hd_alpha must be the imported checkpoint scalar");
    assert_eq!(alpha, vec![f32::from_bits(0x3BB6_0000)], "the published checkpoint's hd_alpha_downsamples");

    enc.write_image(&golden["input"].data);
    enc.run();

    let rd = |b: &gpu_core::DeviceBuffer, n: usize| enc.gpu.read(b, n);
    let rows = (cfg.rows() * cfg.d_model) as usize;
    let mut r = Report::new(FLOOR);
    r.check("block02_out", &rd(enc.block_out(2), rows), &golden["block02_out"].data);
    r.check("block11_out", &rd(enc.block_out(11), rows), &golden["block11_out"].data);
    let tap = |stages: &[(&'static str, &gpu_core::DeviceBuffer, usize)], name: &str| -> Vec<f32> {
        let (_, b, n) = stages.iter().find(|(s, _, _)| *s == name).unwrap_or_else(|| panic!("no stage {name}"));
        rd(b, *n)
    };
    let main = enc.neck_stages();
    let hd = enc.hd_stages();
    r.check("neck_out", &tap(&main, "neck_norm2"), &golden["neck_out"].data);
    r.check("neck_resized", &tap(&main, "neck_resize"), &golden["neck_resized"].data);
    r.check("main_out", &tap(&main, "compress2"), &golden["main_out"].data);
    r.check("hd_neck_out", &tap(&hd, "neck_norm2"), &golden["hd_neck_out"].data);
    r.check("hd_resized", &tap(&hd, "neck_resize"), &golden["hd_resized"].data);
    r.check("hd_out", &tap(&hd, "compress2"), &golden["hd_out"].data);
    let out = rd(enc.output(), enc.out_len());
    r.check("output", &out, &golden["output"].data);
    // Cosine is blind to a uniform scale; the HD share is ~8% of the output
    // norm, so a mis-scaled branch also shows up as relative L2.
    let rel = brain_testutil::parity::rel_l2(&out, &golden["output"].data);
    assert!(rel < 1e-2, "output rel_l2 {rel:.3e}");
    r.finish("sam1 DeepSeek-VL real-weight");
}
