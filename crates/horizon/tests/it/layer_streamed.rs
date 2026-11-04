// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in training every weight through a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning through the shared contract (`model::streamed::check_layer_streaming`): the set
//! encoder's layers stream through a one-layer device window with their f32 values and optimizer state on the host,
//! and train exactly what the build holding every layer in that state trains, which is what the plain f32 build trains
//! on the device. Horizon trains on subject timelines, not a token corpus, so the exact-resume contract of
//! `model::trainprec::check_exact_resume` (a causal-LM corpus) is run here on its own objective: a streamed run
//! stopped after three of six steps and resumed through the fit loop ends with the loss and every weight of the
//! uninterrupted run, bit for bit. One test in its own binary, the contracts in turn, so nothing else allocates while
//! the device bytes are compared.

use data::rng::Rng;
use horizon::batch::assemble;
use horizon::encode::{encode, Encoded};
use horizon::synthetic::drifting::{self, Gaps};
use horizon::train::TimelineObjective;
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};
use model::Model;

const BATCH: u32 = 16;

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(horizon::PIPELINES))
}

#[test]
fn horizon_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    let gaps = Gaps { last: (0.0, 2.0), between: (0.5, 3.0), visits: (1, 5) };
    let (subjects, _) = drifting::population(400, 5, &gaps, 6.0);
    let codes = vec![drifting::CODE.to_string()];
    let vocab = Vocab::fit(&subjects, &codes, &codes, &FitOptions { knots: 9, min_count: 1 }).unwrap();
    let mut cfg = HorizonConfig::tiny(vocab.len(), 1);
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.max_tokens = 8;
    cfg.n_layers = 3;
    let encoded: Vec<Encoded> = subjects.iter().take(60).map(|s| encode(s, &vocab, &cfg)).collect();
    let init = horizon::init_weights(&cfg, 3);
    let build = || Horizon::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), BATCH, &init);

    let refs: Vec<&Encoded> = encoded.iter().take(BATCH as usize).collect();
    let batch = assemble(&cfg, &refs, BATCH as usize, 0.4, &mut Rng::new(data::rng::random_seed()));
    let step = |m: &Horizon, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&batch);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("horizon", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-horizon-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumed = resume_exactly(&build, &encoded, &scratch);
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}

/// A run of six steps, stopped after three and resumed from its state file in a fresh model, ends with the loss and
/// every weight of the uninterrupted run, bit for bit, the layers streaming through a one-layer window.
fn resume_exactly(build: &dyn Fn() -> Horizon, encoded: &[Encoded], scratch: &std::path::Path) -> Result<(), String> {
    let err = |e: std::io::Error| format!("horizon: {e}");
    let opts = model::FitOpts {
        steps: 6,
        batch_size: BATCH,
        lr: 1e-2,
        min_lr: 1e-3,
        warmup: 2,
        decay_iters: 6,
        weight_decay: 0.0,
        grad_clip: 1.0,
        eval_interval: 0,
        eval_batches: 0,
        checkpoint_secs: 0,
        // One drawn seed for the three runs: the comparison is between them, not against a fixed trajectory.
        seed: data::rng::random_seed(),
        precision: model::TrainPrecision::parse("weights=f32,window=1")?,
        ..Default::default()
    };
    let objective = || TimelineObjective::new(encoded, None, 0.4);
    let weights = |m: &Horizon| -> Vec<(String, Vec<u32>)> {
        let mut names = m.param_names();
        names.sort();
        names.into_iter().map(|n| (n.clone(), Model::read_weight(m, &n).into_iter().map(f32::to_bits).collect())).collect()
    };
    let (straight, straight_model) = model::fit_controlled(build(), objective(), &opts, None, model::FitControl::default()).map_err(err)?;
    let state = scratch.join("run.state");
    let identity = serde_json::json!({"test": "horizon"});
    let mut stop_at_3 = |r: &model::StepReport| r.step < 3;
    let control = model::FitControl { on_step: Some(&mut stop_at_3), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
    let (first, _) = model::fit_controlled(build(), objective(), &opts, None, control).map_err(err)?;
    if !first.interrupted || !state.with_extension("offdevice").is_dir() {
        return Err("horizon: the stopped run left no resume state with its off-device generation".into());
    }
    let control = model::FitControl { state: Some(&state), identity, ..Default::default() };
    let (second, resumed) = model::fit_controlled(build(), objective(), &opts, None, control).map_err(err)?;
    if second.resumed_at != Some(3) {
        return Err(format!("horizon: resumed at {:?}, not 3", second.resumed_at));
    }
    let (a, b) = (straight.final_loss.unwrap_or(f32::NAN), second.final_loss.unwrap_or(f32::NAN));
    if a.to_bits() != b.to_bits() {
        return Err(format!("horizon: the resumed run ends at loss {b}, the uninterrupted one at {a}"));
    }
    if let Some(((n, _), _)) = weights(&straight_model).into_iter().zip(weights(&resumed)).find(|((_, x), (_, y))| x != y) {
        return Err(format!("horizon: {n} differs between the resumed and the uninterrupted run"));
    }
    Ok(())
}
