// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements world-model simulators that train and run on
// its clients' own hardware. If your team needs expertise in diffusion world
// models for control and simulation then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The served actions end to end on a random-weight reference checkpoint of
//! the real Atari configuration: import it, export its UNet, roll it out and
//! record the session, replay the recording against the same seed, then fine-tune on it.

use capability::{Invocation, Registry};
use checkpoint::torchpt_write::TensorOut;
use diamond::caps::{DiamondProvider, MODEL};
use diamond::DiamondConfig;
use serde_json::json;

const NUM_ACTIONS: u32 = 4;
const SEED: i64 = 11;

/// Deterministic fan-in-scaled pseudo-random weights, so a deep stack neither
/// blows up nor vanishes.
fn random_reference_checkpoint(path: &str) {
    let cfg = DiamondConfig::atari(NUM_ACTIONS);
    let tensors: Vec<TensorOut> = cfg
        .param_list()
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let n: usize = shape.iter().product();
            let fan_in = if shape.len() == 4 { (shape[1] * shape[2] * shape[3]) as f32 } else { 1.0 };
            let scale = if shape.len() == 4 { 0.5 / fan_in.sqrt() } else { 0.25 };
            let mut s = 0x9E37_79B9_7F4A_7C15u64.wrapping_mul(i as u64 + 1);
            let data = (0..n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * scale
                })
                .collect();
            TensorOut { name: format!("denoiser.inner_model.{name}"), shape, data }
        })
        .collect();
    checkpoint::torchpt_write::write(path, &tensors).unwrap();
}

fn run(weights: Option<&str>, action: &str, inv: Invocation) -> capability::ActionResult {
    let mut reg = Registry::new();
    reg.register(std::sync::Arc::new(DiamondProvider::new(weights.map(str::to_string))));
    reg.run(MODEL, action, inv, &mut |_| {})
}

#[test]
fn import_rollout_replay_and_finetune_agree() {
    let dir = std::env::temp_dir().join(format!("diamond-served-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = |name: &str| dir.join(name).to_str().unwrap().to_string();
    let (pt, base, tuned, episode) = (path("agent.pt"), path("base.safetensors"), path("tuned.safetensors"), path("episode"));
    random_reference_checkpoint(&pt);

    let imported = run(None, "import", Invocation::new().set("source", json!(pt)).set("dest", json!(base)).set("actions_count", json!(NUM_ACTIONS))).unwrap();
    assert_eq!(imported.outputs["num_actions"], json!(NUM_ACTIONS));

    let onnx = path("unet.onnx");
    let exported = run(Some(&base), "export", Invocation::new().set("dest", json!(onnx))).unwrap();
    assert_eq!(exported.outputs["context_frames"], json!(4));
    assert!(std::fs::metadata(&onnx).unwrap().len() > 1_000_000, "the ONNX carries the UNet's weights");

    let rollout = |seed: i64, record: Option<&str>| {
        let mut inv = Invocation::new().set("seed", json!(seed)).set("action_seq", json!("1,2,3,0,1,2")).set("denoise_steps", json!(1));
        if let Some(r) = record {
            inv = inv.set("record", json!(r));
        }
        run(Some(&base), "rollout", inv).unwrap()
    };
    let recorded = rollout(SEED, Some(&episode));
    assert_eq!(recorded.outputs["steps"], json!(6));
    assert_eq!(recorded.outputs["recorded_frames"], json!(6));
    assert_eq!(recorded.outputs["rollout_hash"], rollout(SEED, None).outputs["rollout_hash"], "same seed, same rollout");
    assert_ne!(recorded.outputs["rollout_hash"], rollout(SEED + 1, None).outputs["rollout_hash"], "the seed drives the noise");

    let replay = run(Some(&base), "replay", Invocation::new().set("episode", json!(episode)).set("verify", json!(true)).set("seed", json!(SEED)).set("denoise_steps", json!(1)).set("tolerance", json!(0.0)));
    assert_eq!(replay.unwrap().outputs["verified_frames"], json!(6));

    let tuned_out = run(Some(&base), "finetune", Invocation::new().set("data", json!(episode)).set("save", json!(tuned)).set("steps", json!(2)).set("seed", json!(SEED))).unwrap();
    assert!(tuned_out.outputs["last_loss"].as_f64().unwrap().is_finite());
    // The fine-tuned file loads like an imported one.
    let (cfg, _) = diamond::import::load(&tuned).unwrap();
    assert_eq!(cfg.num_actions, NUM_ACTIONS);
    let _ = std::fs::remove_dir_all(&dir);
}
