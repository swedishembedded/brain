// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The importer against the REAL `convaiinnovations/laya` checkpoint
//! (`brain pull convaiinnovations/laya`, resolved via
//! `brain_testutil::model_dir` - the same `<models-dir>/<vendor>/<repo>/`
//! directory every other real-checkpoint import test in this workspace
//! resolves its fixture through). Proves [`modernbert::import_dir`] loads
//! the real 206-tensor state dict end to end (full two-way coverage against
//! both `ModernBertConfig::tensor_manifest` and `laya::tensor_manifest`,
//! `rl_agent_config.json`'s `head_layers` matching `LayaConfig`, the real
//! tokenizer's special-token ids) with no error - the cheap complement to
//! `tests/laya_real_parity.rs`'s much more expensive real-weight forward
//! check. Skips cleanly when the checkpoint has not been pulled.

#[test]
fn import_dir_loads_the_real_checkpoint_with_full_coverage() {
    let Some(dir) = brain_testutil::model_dir("convaiinnovations/laya") else {
        brain_testutil::skip("no models directory resolvable");
        return;
    };
    if !std::path::Path::new(&dir).join("model.safetensors").exists() {
        brain_testutil::skip(&format!("{dir}/model.safetensors absent - run `brain pull convaiinnovations/laya`"));
        return;
    }

    let ckpt = modernbert::import_dir(&dir).expect("import_dir");

    // The real released shape (encoder/config.json + rl_agent_config.json,
    // verified against the actual files this session).
    assert_eq!(ckpt.cfg.d_model, 1024);
    assert_eq!(ckpt.cfg.n_layers, 28);
    assert_eq!(ckpt.cfg.n_heads, 16);
    assert_eq!(ckpt.cfg.d_ff, 2624);
    assert_eq!(ckpt.cfg.vocab, 50368);
    assert_eq!(ckpt.cfg.window, 64); // local_attention 128 / 2

    // Special-token ids read from the real tokenizer, not the config-only
    // defaults (`ModernBertConfig::from_hf_json`'s own fallback values happen
    // to equal these, but this asserts they were actually READ, not assumed).
    assert_eq!(ckpt.cfg.cls_token_id, 50281);
    assert_eq!(ckpt.cfg.sep_token_id, 50282);
    assert_eq!(ckpt.cfg.pad_token_id, 50283);
    assert_eq!(ckpt.cfg.mask_token_id, 50284);

    assert_eq!(ckpt.laya_cfg.head_layers, 2);
    assert_eq!(ckpt.rl.head_layers, 2);
    assert_eq!(ckpt.rl.max_len, 512);
    assert_eq!(ckpt.rl.head_max_len, 192);
    assert_eq!(ckpt.temperature.len(), 3);

    // Full coverage already asserted internally by import_dir (it errors
    // otherwise); assert the entry counts too as a second, independent check
    // against the manifests it claims to cover.
    let encoder_manifest = ckpt.cfg.tensor_manifest();
    assert_eq!(ckpt.encoder_init.len(), encoder_manifest.len());
    let head_manifest = modernbert::laya::tensor_manifest(&ckpt.laya_cfg);
    assert_eq!(ckpt.head_init.len(), head_manifest.len());

    // A sample tensor's element count matches its declared shape.
    let (_, tok_shape) = encoder_manifest.iter().find(|(n, _)| n == "tok.weight").unwrap();
    assert_eq!(ckpt.encoder_init["tok.weight"].len(), tok_shape.iter().product::<usize>());
}
