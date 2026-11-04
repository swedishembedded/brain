// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The real Qwen3-TTS checkpoint, read as downloaded, is the checkpoint
//! `brain qwen3tts import` writes from it: for the Talker, the MTP, the codec
//! and the speaker encoder, the same tensor names, values and config.
//! Compares against an imported `brain_tts/` directory beside the download
//! (`qwen3tts import --out-dir <ckpt>/brain_tts`), one tensor at a time.

use checkpoint::weightio::WeightReader;

fn assert_same(what: &str, view: &WeightReader, file: &WeightReader) {
    let mut a: Vec<&str> = view.names().collect();
    let mut b: Vec<&str> = file.names().collect();
    a.sort();
    b.sort();
    assert_eq!(a, b, "{what}: tensor names");
    // An import written before the config gained keys (`qk_norm`,
    // `attention_bias`) lacks them; everything it does record must agree.
    let (vc, fc) = (view.config(), file.config());
    for (k, v) in fc.as_object().unwrap() {
        assert_eq!(vc.get(k), Some(v), "{what}: config {k}");
    }
    for name in a {
        assert!(view.tensor(name) == file.tensor(name), "{what}: {name} differs");
    }
}

#[test]
fn the_downloaded_checkpoint_reads_as_its_import() {
    let Some(dir) = brain_testutil::model_dir("Qwen/Qwen3-TTS-12Hz-0.6B-Base") else {
        brain_testutil::skip("Qwen/Qwen3-TTS-12Hz-0.6B-Base not downloaded");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let imported = dir.join("brain_tts");
    if !imported.join("talker.safetensors").exists() {
        brain_testutil::skip("no imported brain_tts/ beside the checkpoint");
        return;
    }
    let file = |f: &str| WeightReader::open(imported.join(f).to_str().unwrap()).unwrap();
    let ckpt = dir.to_str().unwrap();
    assert_same("talker", &qwen3tts::import::open_talker(ckpt).unwrap(), &file("talker.safetensors"));
    assert_same("mtp", &qwen3tts::import::open_mtp(ckpt).unwrap(), &file("mtp.safetensors"));
    assert_same("codec", &mimi::import::open(dir.join("speech_tokenizer").to_str().unwrap()).unwrap(), &file("codec.safetensors"));
    assert_same("speaker", &ecapatdnn::import::open(ckpt).unwrap(), &file("speaker.safetensors"));

    let paths = qwen3tts::TtsPaths::new(&dir, ckpt);
    assert_eq!(paths.talker, ckpt, "a checkpoint dir is read as downloaded");
    paths.require(true).unwrap();
}
