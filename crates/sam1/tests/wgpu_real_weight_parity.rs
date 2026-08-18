// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! **Real-weight, full-depth, production-shape cross-backend parity.**
//!
//! `tests/parity.rs` gates the forward against a real llama.cpp golden but
//! pins the CPU backend to do it, because the wgpu backend used to corrupt
//! this tower's per-block buffers once the graph held three or more blocks.
//! `tests/wgpu_block_count_corruption.rs::wgpu_backend_block0_is_unaffected_by_a_third_block`
//! gates ONE fixed defect (a repeated 2D-tiled in-place dispatch in
//! `attn_relpos_add.wgsl`) with a checkpoint-free two-block-prefix invariant.
//! This file is the real-weight, full-12-layer confirmation.
//!
//! It used to NOT be green: an earlier pass recorded `patch_tokens` alone
//! (computed before any block runs) disagreeing between backends at cosine
//! ~0.11 at full depth, suspected at the time to be a SECOND, distinct wgpu
//! defect. That was BEFORE `backend-wgpu::WgpuBackend::flush_serialized`
//! (commit `04675800`, mirroring `backend-vulkan`'s confirmed Intel ANV
//! sliced-binding fix) landed. With that fix in place, this test was run
//! FIVE times in a row against the real mmproj checkpoint: worst cosine
//! `1.0000000000` on every tap (`patch_tokens` included) every single run -
//! the "second defect" was the same sliced-binding race, now fixed, not an
//! independent one. See `tests/wgpu_block_count_corruption.rs`'s doc comment
//! for the fuller confirmation record (including a from-scratch reproduction
//! attempt under heavy induced machine contention matching the original
//! bisection's conditions, also clean).
//!
//! Needs no llama.cpp golden fixture (only the mmproj checkpoint, which
//! `tests/import.rs`'s coverage test already requires), so it runs wherever
//! that checkpoint is present rather than self-skipping on every machine that
//! lacks the golden dump. Skips itself when the mmproj checkpoint is absent -
//! never panics for a missing input - so, like `tests/parity.rs`'s own
//! real-weight tests, it does not need `#[ignore]`: no checkpoint means no
//! work, not a hard failure.

use sam1::{SamEncoder, SamViTConfig};

fn mmproj_path() -> Option<std::path::PathBuf> {
    let dir = brain_testutil::model_dir("ggml-org/DeepSeek-OCR-GGUF")?;
    let p = std::path::Path::new(&dir).join("mmproj-DeepSeek-OCR-Q8_0.gguf");
    p.exists().then_some(p)
}

#[test]
fn real_mmproj_sam_tower_agrees_cpu_vs_wgpu_at_full_depth() {
    if std::env::var("MOE_SKIP_GPU_TESTS").is_ok() {
        return;
    }
    let Some(mmproj) = mmproj_path() else {
        brain_testutil::skip("DeepSeek-OCR-GGUF mmproj not in the model store");
        return;
    };
    let mg = checkpoint::gguf::MmapGguf::open(mmproj.to_str().expect("utf-8 path")).expect("open mmproj");
    let (cfg, weights) = sam1::import::weights_from_gguf(&mg).expect("import weights");
    drop(mg);
    assert_eq!(cfg, SamViTConfig::deepseek_ocr(), "the shipped file disagrees with the preset");
    cfg.check_bindable();

    // The reference capture's own constant-gray input (see tests/parity.rs) -
    // this file compares backends against EACH OTHER, not against that
    // golden, so any finite, non-degenerate image works; the constant one is
    // reused because it is already proven not to hide an axis-order bug.
    let px = vec![0.5f32; 3 * cfg.image_h() as usize * cfg.image_w() as usize];

    let cpu = gpu_core::Gpu::new_cpu(sam1::PIPELINES);
    let enc_cpu = SamEncoder::new_inference(cpu, cfg.clone(), &weights, 0);
    enc_cpu.write_image(&px);
    let obj_cpu = enc_cpu.forward();
    assert!(obj_cpu.is_finite(), "CPU forward produced a non-finite objective");

    let wgpu = gpu_core::Gpu::new_wgpu(sam1::PIPELINES);
    let enc_wgpu = SamEncoder::new_inference(wgpu, cfg.clone(), &weights, 0);
    enc_wgpu.write_image(&px);
    let obj_wgpu = enc_wgpu.forward();
    assert!(obj_wgpu.is_finite(), "wgpu forward produced a non-finite objective");

    let rows = cfg.rows() as usize * cfg.d_model as usize;
    let worst = |a: &[f32], b: &[f32]| -> f64 {
        let (cos, max_abs) = brain_testutil::parity::compare(a, b);
        println!("    cos {cos:.10}  max_abs {max_abs:.3e}");
        cos
    };
    let mut floor = 1.0f64;
    print!("  patch_tokens (pre-block):");
    floor = floor.min(worst(&enc_cpu.gpu.read(enc_cpu.patch_tokens(), rows), &enc_wgpu.gpu.read(enc_wgpu.patch_tokens(), rows)));
    print!("  embedded_tokens (block0 input):");
    floor = floor.min(worst(&enc_cpu.gpu.read(enc_cpu.embedded_tokens(), rows), &enc_wgpu.gpu.read(enc_wgpu.embedded_tokens(), rows)));
    print!("  block 00 norm1:");
    let n1rows = (cfg.rows() as usize + 1) * cfg.d_model as usize;
    floor = floor.min(worst(&enc_cpu.gpu.read(enc_cpu.block_norm1(0), n1rows), &enc_wgpu.gpu.read(enc_wgpu.block_norm1(0), n1rows)));
    print!("  block 00 attn_res:");
    floor = floor.min(worst(&enc_cpu.gpu.read(enc_cpu.block_attn_res(0), rows), &enc_wgpu.gpu.read(enc_wgpu.block_attn_res(0), rows)));
    for l in 0..cfg.n_layers as usize {
        print!("  block {l:02}:");
        let a = enc_cpu.gpu.read(enc_cpu.block_out(l), rows);
        let b = enc_wgpu.gpu.read(enc_wgpu.block_out(l), rows);
        floor = floor.min(worst(&a, &b));
    }
    print!("  compressor output:");
    let a = enc_cpu.gpu.read(enc_cpu.output(), enc_cpu.out_len());
    let b = enc_wgpu.gpu.read(enc_wgpu.output(), enc_wgpu.out_len());
    floor = floor.min(worst(&a, &b));

    println!("sam1 cpu-vs-wgpu real-weight parity: worst cosine {floor:.10}");
    // fp32 vs fp32, both dequantizing the SAME Q8_0 blocks the same way -
    // this is two runs of the identical arithmetic on two backends, so the
    // floor is round-off, not quantization noise (contrast tests/parity.rs's
    // 0.999, which compares against ggml's int8-activation reference).
    assert!(floor > 0.999999, "cpu vs wgpu diverged: worst cosine {floor:.10}");
}
