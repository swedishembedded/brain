// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Rollouts on the serving engine sample the policy the trainer holds: after
//! a sync with a LoRA model's adapters, the engine (base weights with the
//! adapters folded in) decodes what the LoRA model itself does, and a sync
//! with zeroed adapters decodes the base model again.

use std::collections::HashMap;

use data::rng::Rng;
use model::rollout::{ModelRollout, Rollout, RolloutParams, SyncRollout};
use model::serve::SampleParams;
use qwen3::rollout::{EngineGeometry, EngineRollout};
use qwen3::{Dtype, LoraCfg, Qwen, QwenConfig};

fn gpu_disabled() -> bool {
    std::env::var("MOE_SKIP_GPU_TESTS").is_ok()
}

fn greedy() -> RolloutParams {
    RolloutParams { max_new: 6, sample: SampleParams::greedy(), eos: None }
}

fn trained(model: &Qwen) -> HashMap<String, Vec<f32>> {
    model::Model::optimized_params(model).unwrap().into_iter().map(|n| {
        let w = model::Model::read_weight(model, &n);
        (n, w)
    }).collect()
}

#[test]
fn the_engine_samples_what_the_synced_model_does() {
    if gpu_disabled() {
        return;
    }
    let lora = LoraCfg::attn(2, 4.0);
    let base_cfg = QwenConfig { block_size: 16, ..QwenConfig::tiny() };
    let cfg = QwenConfig { lora: Some(lora.clone()), ..base_cfg.clone() };
    let base = qwen3::init_weights(&base_cfg, 11);
    let mut init = qwen3::init_weights(&cfg, 11);
    init.extend(base.iter().map(|(k, v)| (k.clone(), v.clone())));
    // Non-zero adapters, large enough to change the greedy path.
    let mut rng = Rng::new(3);
    for (name, v) in init.iter_mut().filter(|(n, _)| n.contains(".lora_")) {
        let s = if name.ends_with(".lora_b") { 0.5 } else { 0.3 };
        v.iter_mut().for_each(|x| *x = (rng.next_f32() * 2.0 - 1.0) * s);
    }
    let model = Qwen::new(cfg, 1, 16, &init);
    let geometry = EngineGeometry { block_size: 4, num_blocks: 64, max_batch: 2, max_blocks_per_seq: 4, max_prefill: 8, tier: Dtype::F32, gpu: None };
    let mut engine = EngineRollout::new(base_cfg.clone(), base.clone(), Some(lora), geometry);
    let prompt = [1u32, 5, 9, 2];

    engine.sync(&trained(&model)).unwrap();
    let want = ModelRollout::new(&model).sample_n(&prompt, 2, &greedy(), &mut Rng::new(1));
    let got = engine.sample_n(&prompt, 2, &greedy(), &mut Rng::new(1));
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(g.tokens, w.tokens, "the folded adapters decode as the LoRA model does");
    }

    let zeroed: HashMap<String, Vec<f32>> = trained(&model).into_iter().map(|(k, v)| (k.clone(), if k.ends_with(".lora_b") { vec![0.0; v.len()] } else { v })).collect();
    engine.sync(&zeroed).unwrap();
    let plain = Qwen::new(base_cfg, 1, 16, &base);
    let want = ModelRollout::new(&plain).sample_n(&prompt, 1, &greedy(), &mut Rng::new(1));
    assert_eq!(engine.sample_n(&prompt, 1, &greedy(), &mut Rng::new(1))[0].tokens, want[0].tokens, "zeroed adapters are the base model");
    assert_ne!(want[0].tokens, got[0].tokens, "the adapters changed the greedy path, so the first check was not vacuous");
}
