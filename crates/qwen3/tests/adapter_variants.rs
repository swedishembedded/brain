// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for serving many fine-tuned
// variants of one model from a single resident copy of its weights for its
// clients. If your team needs expertise in multi-tenant inference engines
// then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Gates A2 and A3 (first architecture, dense GQA) for the adapter variants:
//! a model trained with rsLoRA + LoRA+ + LoRA-FA is saved, and then
//!
//! * fused: folding the saved file into the base gives the trained model's
//!   (unfused, live-adapter) logits,
//! * overlay: attaching the saved file to a resident base, with no folding,
//!   gives the same logits,
//! * the file carries the trained scale (`alpha / rank`), and LoRA-FA's frozen
//!   `A` is exactly its init in the file.

use qwen3::{LoraCfg, Qwen, QwenConfig};

const PROMPT: [u32; 8] = [3, 17, 5, 9, 22, 1, 14, 8];

fn variant() -> LoraCfg {
    LoraCfg {
        rank_stabilized: true,
        lr_ratio: 4.0,
        freeze_a: true,
        ..LoraCfg::new(3, 6.0, ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect())
    }
}

fn per_target() -> LoraCfg {
    variant_base().with_override("wq", Some(2), Some(2.0)).with_override("down", Some(4), Some(1.0))
}

fn variant_base() -> LoraCfg {
    LoraCfg::new(3, 6.0, ["wq", "wk", "wv", "wo", "gate", "up", "down"].iter().map(|s| s.to_string()).collect())
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn a_variant_adapter_folds_and_overlays_to_the_trained_models_logits() {
    folds_and_overlays(variant());
}

/// Targets with their own rank and alpha are written as one uniform adapter
/// (padded to the widest rank, each scale folded into B), so fold, overlay and
/// every other reader agree with the live per-target forward.
#[test]
fn per_target_rank_and_alpha_fold_and_overlay_to_the_trained_models_logits() {
    folds_and_overlays(per_target());
}

fn folds_and_overlays(lora: LoraCfg) {
    let tag = lora.rank_stabilized as u8 + 2 * lora.overrides.len() as u8;
    let cfg = QwenConfig { lora: Some(lora), ..QwenConfig::tiny() };
    let init = qwen3::init_weights(&cfg, 21);
    let x: Vec<u32> = (0..12).map(|i| (i * 7 + 1) % 23).collect();
    let y: Vec<u32> = (0..12).map(|i| (i * 7 + 2) % 23).collect();
    let trained = Qwen::new(cfg.clone(), 1, 12, &init);
    trained.set_batch(&x, &y);
    for step in 1..=8 {
        trained.zero_grads();
        trained.forward();
        trained.backward();
        trained.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
        trained.poll_wait();
    }
    let want = trained.logits_all(&PROMPT);

    let path = std::env::temp_dir().join(format!("brain-qwen3-variants-{}-{}.safetensors", std::process::id(), tag));
    let path_str = path.to_str().unwrap();
    qwen3::lora::save_adapter(path_str, &trained, "test/adapter", "test/base", None).unwrap();

    // The file carries the trained scale, and the frozen factor is its init.
    let file = model::lora::device_adapter::read_adapter(path_str).unwrap();
    let lora = cfg.lora.as_ref().unwrap();
    if lora.is_uniform() {
        assert!((file.scale() - lora.scale()).abs() < 1e-6, "file scale {} vs trained {}", file.scale(), lora.scale());
    } else {
        assert_eq!((file.rank, file.alpha), (lora.max_rank(), lora.max_rank() as f32), "a per-target adapter is written at the widest rank, scale 1");
    }
    if lora.freeze_a {
        for site in &file.sites {
            let a_name = model::adapter::device::param_name(&site.base, model::adapter::kind::LORA_A);
            assert_eq!(site.a, init[&a_name], "{a_name}: LoRA-FA must leave A at its init");
        }
    }

    let base: std::collections::HashMap<String, Vec<f32>> =
        trained.ps.params.iter().map(|(n, _)| n).filter(|n| !model::adapter::device::is_adapter_param(n)).map(|n| (n.clone(), trained.read_weight(n))).collect();
    let tiny = QwenConfig::tiny();
    let shard = || qwen3::Shard::whole(tiny.n_layers as usize);

    let mut folded_tensors = base.clone();
    qwen3::lora::fold_adapter_into(&mut folded_tensors, path_str).unwrap();
    let folded = Qwen::new_shard(tiny.clone(), 1, 12, &folded_tensors, false, shard());
    let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    let fused = max_diff(&folded.logits_all(&PROMPT), &want);
    assert!(fused <= 1e-4 * scale, "fused (folded) vs live adapter: {fused:e} over logit scale {scale:e}");

    let mut overlay = Qwen::new_shard(tiny.clone(), 1, 12, &base, false, shard());
    let plain = overlay.logits_all(&PROMPT);
    assert!(max_diff(&plain, &want) > 1e-2, "the trained adapter must move the logits for parity to mean anything");
    overlay.attach_adapter(path_str).unwrap();
    let over = max_diff(&overlay.logits_all(&PROMPT), &want);
    assert!(over <= 1e-4 * scale, "overlay vs live adapter: {over:e} over logit scale {scale:e}");
    std::fs::remove_file(&path).ok();
}

/// IA3, LoKr and DoRA have no low-rank form: a saved file folds into the base
/// (`AdapterFile::fold_into`), attaches beside the unmodified base through the
/// shared adapter runtime, and reloads into a fresh adapter build - each giving
/// the trained model's logits.
#[test]
fn ia3_lokr_and_dora_fold_overlay_and_reload_to_the_trained_models_logits() {
    for kind in [model::adapter::DeviceKind::Ia3, model::adapter::DeviceKind::Lokr, model::adapter::DeviceKind::Dora] {
        let lora = variant_base().with_kind(kind);
        let cfg = QwenConfig { lora: Some(lora), ..QwenConfig::tiny() };
        let init = qwen3::init_weights(&cfg, 21);
        let x: Vec<u32> = (0..12).map(|i| (i * 7 + 1) % 23).collect();
        let y: Vec<u32> = (0..12).map(|i| (i * 7 + 2) % 23).collect();
        let trained = Qwen::new(cfg.clone(), 1, 12, &init);
        trained.set_batch(&x, &y);
        for step in 1..=8 {
            trained.zero_grads();
            trained.forward();
            trained.backward();
            trained.adamw_step(step, 5e-2, 0.0, Default::default(), Some(1.0), 1.0);
            trained.poll_wait();
        }
        let want = trained.logits_all(&PROMPT);
        let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));

        let path = std::env::temp_dir().join(format!("brain-qwen3-kind-{}-{}.safetensors", std::process::id(), kind.name()));
        let path_str = path.to_str().unwrap();
        qwen3::lora::save_adapter(path_str, &trained, "test/adapter", "test/base", None).unwrap();
        let file = model::adapter::file::AdapterFile::read(path_str).unwrap();
        assert_eq!(file.kind, kind);
        assert!(!file.dims.is_empty(), "the file records the linears' dimensions");

        let base: std::collections::HashMap<String, Vec<f32>> =
            trained.ps.params.iter().map(|(n, _)| n).filter(|n| !model::adapter::device::is_adapter_param(n)).map(|n| (n.clone(), trained.read_weight(n))).collect();
        let tiny = QwenConfig::tiny();
        let shard = || qwen3::Shard::whole(tiny.n_layers as usize);

        // Fused: fold into a plain base.
        let mut folded_tensors = base.clone();
        file.fold_into(&mut folded_tensors).unwrap();
        let folded = Qwen::new_shard(tiny.clone(), 1, 12, &folded_tensors, false, shard());
        let fused = max_diff(&folded.logits_all(&PROMPT), &want);
        assert!(fused <= 1e-4 * scale, "{}: fused vs live adapter: {fused:e} over {scale:e}", kind.name());

        // Overlay: the unmodified base plus the attached adapter.
        let mut overlay = Qwen::new_shard(tiny.clone(), 1, 12, &base, false, shard());
        let plain = overlay.logits_all(&PROMPT);
        assert!(max_diff(&plain, &want) > 1e-2, "{}: the adapter must move the logits", kind.name());
        overlay.attach_adapter(path_str).unwrap();
        assert_eq!(overlay.attached_adapter().map(|(r, _)| r), Some(file.rank));
        let over = max_diff(&overlay.logits_all(&PROMPT), &want);
        assert!(over <= 1e-4 * scale, "{}: overlay vs live adapter: {over:e} over {scale:e}", kind.name());
        assert!(overlay.detach_adapter());
        assert_eq!(overlay.logits_all(&PROMPT), plain, "{}: detach restores the base", kind.name());

        // Reload: a fresh adapter build given the file's tensors.
        let fresh = Qwen::new(cfg.clone(), 1, 12, &init);
        for (name, data) in &file.tensors {
            fresh.write_weight(name, data);
        }
        let reloaded = max_diff(&fresh.logits_all(&PROMPT), &want);
        assert!(reloaded <= 1e-6 * scale.max(1.0), "{}: reloaded vs trained: {reloaded:e}", kind.name());
        std::fs::remove_file(&path).ok();
    }
}
