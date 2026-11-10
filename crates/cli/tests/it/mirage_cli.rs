// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for structured visual perception
// behind command lines and SDKs for its clients. If your team needs expertise
// in turning images into structured data then you can procure our services by
// sending an email to info@swedishembedded.com.

//! `brain mirage describe|extract` on the cats photograph of the tracked
//! goldens, against the SDK: both run the same session, so the JSON the
//! command prints is the JSON `PerceptionPipeline` returns. `brain caps` lists
//! both actions with no weights.

use std::process::Command;

use serde_json::{json, Value};

const MODEL: &str = "Cactus-Compute/mirage";
const FORM: &str = r#"{"animal": {"enum": ["cat", "dog", "bird"]}, "count": {"type": "integer", "minimum": 0, "maximum": 6}}"#;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_brain")
}

/// The cats photograph as a PNG in a scratch directory, plus the scratch directory.
fn cats_png() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let golden = brain_testutil::testdata("golden/mirage/goldens/cats.safetensors");
    if !std::path::Path::new(&golden).is_file() {
        brain_testutil::skip("mirage cli: testdata/golden/mirage/goldens/cats.safetensors is absent");
        return None;
    }
    let dir = brain_testutil::model_dir(MODEL)?;
    if !std::path::Path::new(&dir).join("mirage.cact").is_file() {
        brain_testutil::skip("mirage cli: Cactus-Compute/mirage is not in the model store");
        return None;
    }
    let tensors = checkpoint::st::load_safetensors(&golden).expect("golden loads").tensors;
    let rgb: Vec<u8> = tensors["input_rgb"].iter().map(|&v| v as u8).collect();
    let scratch = std::env::temp_dir().join(format!("brain-mirage-cli-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    let png = scratch.join("cats.png");
    imaging::save(&png, &imaging::Rgb8::new(640, 480, rgb).unwrap()).unwrap();
    Some((png, scratch))
}

/// The JSON a `--json` run printed, without the accounting.
fn printed(out: &std::process::Output) -> Value {
    assert!(out.status.success(), "brain failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    let mut v: Value = serde_json::from_str(text.lines().last().expect("a JSON line")).expect("the last line is JSON");
    v.as_object_mut().unwrap().remove("usage");
    v
}

#[test]
fn caps_lists_both_actions_with_no_weights() {
    let out = Command::new(bin()).args(["caps", MODEL]).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains(MODEL) && text.contains("describe") && text.contains("extract"), "{text}");
    assert!(text.contains("--form") && text.contains("--mask_format"), "{text}");
    assert!(!text.contains("weights"), "no param names a checkpoint: {text}");
}

#[test]
fn the_command_prints_the_json_the_sdk_returns() {
    let Some((png, scratch)) = cats_png() else { return };
    let describe = Command::new(bin()).args(["mirage", "describe", "--image"]).arg(&png).args(["--mask_format", "bitmap", "--max_tokens", "600", "--json"]).output().unwrap();
    let extract = Command::new(bin()).args(["mirage", "extract", "--image"]).arg(&png).args(["--form", FORM, "--json"]).output().unwrap();

    let pipeline = brain::PerceptionPipeline::from_pretrained(MODEL).expect("the pipeline loads");
    let photo = brain::Image::open(&png).unwrap();
    let described = pipeline.describe_with(&photo, &brain::PerceptionOptions::new().mask_format(brain::MaskFormat::Bitmap).max_tokens(600)).unwrap();
    assert_eq!(&printed(&describe), described.json(), "describe: CLI JSON equals SDK JSON");
    assert!(described.usage().decoder_tokens > 20);
    let answered = pipeline.extract(&photo, &serde_json::from_str(FORM).unwrap()).unwrap();
    assert_eq!(&printed(&extract), answered.json(), "extract: CLI JSON equals SDK JSON");
    assert_eq!(answered.json()["form"].as_object().unwrap().len(), 2);
    assert!(answered.value("animal").is_some());
    let _ = json!(null);
    std::fs::remove_dir_all(scratch).ok();
}

#[test]
fn a_bad_option_is_refused_with_its_name_before_any_weights_load() {
    let Some((png, scratch)) = cats_png() else { return };
    let out = Command::new(bin()).args(["mirage", "describe", "--image"]).arg(&png).args(["--sections", "objects,bogus"]).output().unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("sections") || err.contains("bogus"), "{err}");
    std::fs::remove_dir_all(scratch).ok();
}
