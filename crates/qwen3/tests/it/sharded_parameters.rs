// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements full-parameter fine-tuning of language models larger than one accelerator, sharded
// over the cards of a cluster, for its clients. If your team needs expertise in ZeRO-3-style parameter sharding, you
// can procure our services by sending an email to info@swedishembedded.com.

//! A tiny Qwen3 trained with its parameters sharded (ZeRO-3) over in-process ranks through the generic fit loop, to
//! the end: the run writes its trained model (no rank holding more than a layer), and the file opens as a qwen3
//! checkpoint holding the single-rank run's weights within bf16 rounding.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use data::binio::{self, Meta};
use model::memplan::Capacity;
use model::resident_ft::{ClipPass, LayerWindow, ResidentFtConfig, ResidentValue};
use model::zero::{ShardGroup, Sharding};
use model::{Collective, GradSync, HostCollective, Model};
use optim::rule::Rule;
use paramstore::precision::{GradDtype, MasterPolicy, StateFormat};
use qwen3::{Qwen, QwenConfig, ResidentPlan};

const VOCAB: u32 = 24;
const T: u32 = 16;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("brain-qwen3-zero3-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// One token, repeated: every row drawn anywhere in the stream is the same, so every rank's batch is the single
/// rank's and the sharded run's gradients are the single run's.
fn write_dataset(dir: &Path) {
    let tokens = vec![7u32; 800];
    let mask = vec![true; 800];
    binio::write_u32_bin(&dir.join("train.u32.bin"), &tokens).unwrap();
    binio::write_mask_bin(&dir.join("train.mask.bin"), &mask).unwrap();
    binio::write_u32_bin(&dir.join("val.u32.bin"), &[]).unwrap();
    binio::write_mask_bin(&dir.join("val.mask.bin"), &[]).unwrap();
    std::fs::write(dir.join("meta.json"), Meta::vocab_only(VOCAB as usize)).unwrap();
}

fn cfg() -> QwenConfig {
    QwenConfig { vocab: VOCAB, block_size: T, n_layers: 3, d_model: 32, n_heads: 4, n_kv_heads: 2, head_dim: 8, d_ff: 64, max_position_embeddings: T, ..QwenConfig::tiny() }
}

fn opts(sharding: Sharding, seed: u64) -> model::FitOpts {
    model::FitOpts { steps: 4, batch_size: 1, block_size: T, lr: 1e-2, min_lr: 1e-3, warmup: 2, decay_iters: 4, weight_decay: 0.0, grad_clip: 0.5, eval_interval: 0, eval_batches: 0, seed, sharding, ..Default::default() }
}

/// The bf16 layer-streamed resident build a parameter-sharded run needs (over `shard`, or on one rank).
fn build(shard: Option<ShardGroup>) -> Qwen {
    let resident = ResidentFtConfig {
        value: ResidentValue::Bf16,
        master: MasterPolicy::F32,
        state: StateFormat::F32,
        rule: Rule::AdamW { adam: model::Adam::default(), wd: 0.0 },
        grad: GradDtype::Bf16,
        stochastic_state: false,
        seed: None,
        tile_elems: 256,
        state_dir: None,
        defer_publication: false,
        window: Some(LayerWindow { layers: 1 }),
        clip_pass: ClipPass::TwoPass,
        shard,
    };
    let plan = ResidentPlan { cfg: resident, capacity: Capacity { device_bytes: 1 << 30, host_bytes: 1 << 30 }, others: Vec::new() };
    let c = cfg();
    let init = qwen3::init_weights(&c, 7);
    Qwen::new_resident(c, 1, T, &init, &plan, gpu_core::provider::library::GemmPrecision::Fp32Reference).unwrap()
}

fn run(dir: &Path, opts: &model::FitOpts, m: Qwen, sync: Option<GradSync<'_>>, out: &Path) {
    let (train, val, bcfg, _, itos) = model::load_dataset_with_itos(dir, opts).unwrap();
    let control = model::FitControl { grad_sync: sync, ..Default::default() };
    model::fit_controlled(m, model::causal_lm::<Qwen>(train, val, bcfg, itos), opts, Some(out), control).unwrap();
}

/// Spec: a parameter-sharded run of qwen3 ends by writing its trained model, never failing at the save; the file
/// opens as a qwen3 checkpoint whose every weight is the single-rank run's within bf16 rounding and the sharded
/// update's rounding.
#[test]
fn a_parameter_sharded_run_writes_a_checkpoint_of_the_trained_model() {
    let dir = tmp("save");
    write_dataset(&dir);
    let seed = data::rng::random_seed();
    let (sharded_out, single_out) = (dir.join("sharded.safetensors"), dir.join("single.safetensors"));
    let world = 2;
    let coll = HostCollective::new(world);
    std::thread::scope(|s| {
        for rank in 0..world {
            let (coll, dir, out) = (Arc::clone(&coll), &dir, &sharded_out);
            s.spawn(move || {
                let shared: Arc<dyn Collective> = coll.clone();
                let m = build(Some(ShardGroup { collective: shared, rank }));
                run(dir, &opts(Sharding::Parameters, seed), m, Some(GradSync { collective: &*coll, rank, checkpoint: None, observer: None }), out);
            });
        }
    });
    run(&dir, &opts(Sharding::Replicated, seed), build(None), None, &single_out);
    let (got, want) = (Qwen::load(sharded_out.to_str().unwrap(), 1, T), Qwen::load(single_out.to_str().unwrap(), 1, T));
    let o = opts(Sharding::Parameters, seed);
    let names = want.param_names();
    assert_eq!(got.param_names(), names, "the written model holds another set of tensors");
    for name in names {
        let (a, b) = (got.read_weight(&name), want.read_weight(&name));
        let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let worst = a.iter().zip(&b).fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        // bf16 rounding of the written values, and AdamW's normalised step turning a rounding difference in a
        // near-zero gradient into up to a fraction of the learning rate.
        let bound = 4e-3 * scale + 0.1 * o.lr;
        assert!(worst <= bound, "{name}: {worst:e} from the single-rank run (bound {bound:e})");
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
