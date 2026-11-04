// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in training every weight through a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Layer-streamed full fine-tuning through the shared contract (`model::streamed::check_layer_streaming`): the
//! encoder's and the decoder's layers (the decoder's cross-attention included) stream through a one-layer device
//! window with their f32 values and optimizer state on the host, and train exactly what the build holding every layer
//! in that state trains, which is what the plain f32 build trains on the device. The model trains on source/target
//! pairs, not a token corpus, so the exact-resume contract of `model::trainprec::check_exact_resume` is run here on
//! its own objective: a streamed run stopped after three of six steps and resumed through the fit loop ends with the
//! loss and every weight of the uninterrupted run, bit for bit. One test in its own binary, the contracts in turn,
//! so nothing else allocates while the device bytes are compared.

use data::rng::Rng;
use model::{Model, Objective};
use toyseq2seq::{Seq2Seq, Seq2SeqConfig};

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(toyseq2seq::model::PIPELINES))
}

const BATCH: u32 = 2;

/// A source/target task over a fixed set of pairs: each micro-step draws `BATCH` of them at random.
struct Pairs {
    /// `(source, target, labels)` per pair, sized for the model's `src_block_size` and `block_size`.
    pairs: Vec<(Vec<u32>, Vec<u32>, Vec<u32>)>,
}

impl Pairs {
    fn draw(&self, model: &Seq2Seq, rng: &mut Rng) {
        let (mut src, mut tgt, mut labels) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..BATCH {
            let (s, t, l) = &self.pairs[(rng.next_u64() % self.pairs.len() as u64) as usize];
            src.extend_from_slice(s);
            tgt.extend_from_slice(t);
            labels.extend_from_slice(l);
        }
        model.set_batch(&src, &tgt, &labels);
    }
}

impl Objective<Seq2Seq> for Pairs {
    fn regime(&self) -> &'static str {
        "pairs"
    }

    fn micro_step(&mut self, model: &Seq2Seq, rng: &mut Rng) -> f32 {
        self.draw(model, rng);
        let loss = model.forward();
        model.backward();
        loss
    }

    /// The initial loss estimate needs no gradient, and a layer-streamed build runs its backward inside the update.
    fn loss_probe(&mut self, model: &Seq2Seq, rng: &mut Rng) -> f32 {
        self.draw(model, rng);
        model.forward()
    }
}

#[test]
fn toyseq2seq_streams_its_layers_through_a_one_layer_window_and_resumes_exactly() {
    // Two encoder and two decoder layers: both stacks stream, each layer under its own index.
    let cfg = Seq2SeqConfig { n_enc: 2, n_dec: 2, ..Seq2SeqConfig::tiny() };
    let init = toyseq2seq::init_weights(&cfg, 11);
    let (t, s) = (cfg.block_size, cfg.src_block_size);
    let src: Vec<u32> = (0..BATCH * s).map(|i| (i * 5 + 1) % cfg.vocab).collect();
    let tgt: Vec<u32> = (0..BATCH * t).map(|i| (i * 7 + 2) % cfg.vocab).collect();
    let labels: Vec<u32> = (0..BATCH * t).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let build = || Seq2Seq::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), BATCH, t, &init);
    let step = |m: &Seq2Seq, step: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&src, &tgt, &labels);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, step, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_layer_streaming("toyseq2seq", build, step).unwrap();

    let scratch = std::env::temp_dir().join(format!("brain-toyseq2seq-streamed-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&scratch);
    std::fs::create_dir_all(&scratch).unwrap();
    let resumed = resume_exactly(&build, &cfg, &scratch);
    let _ = std::fs::remove_dir_all(&scratch);
    resumed.unwrap();
}

/// A run of six steps, stopped after three and resumed from its state file in a fresh model, ends with the loss and
/// every weight of the uninterrupted run, bit for bit, the layers streaming through a one-layer window.
fn resume_exactly(build: &dyn Fn() -> Seq2Seq, cfg: &Seq2SeqConfig, scratch: &std::path::Path) -> Result<(), String> {
    let err = |e: std::io::Error| format!("toyseq2seq: {e}");
    let pairs: Vec<_> = (0..24u32)
        .map(|k| {
            let src: Vec<u32> = (0..cfg.src_block_size).map(|i| (i * 3 + k) % cfg.vocab).collect();
            let tgt: Vec<u32> = (0..cfg.block_size).map(|i| (i * 5 + k) % cfg.vocab).collect();
            let labels: Vec<u32> = (0..cfg.block_size).map(|i| (i * 5 + k + 1) % cfg.vocab).collect();
            (src, tgt, labels)
        })
        .collect();
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
    let objective = || Pairs { pairs: pairs.clone() };
    let weights = |m: &Seq2Seq| -> Vec<(String, Vec<u32>)> {
        let mut names = m.param_names();
        names.sort();
        names.into_iter().map(|n| (n.clone(), Model::read_weight(m, &n).into_iter().map(f32::to_bits).collect())).collect()
    };
    let (straight, straight_model) = model::fit_controlled(build(), objective(), &opts, None, model::FitControl::default()).map_err(err)?;
    let state = scratch.join("run.state");
    let identity = serde_json::json!({"test": "toyseq2seq"});
    let mut stop_at_3 = |r: &model::StepReport| r.step < 3;
    let control = model::FitControl { on_step: Some(&mut stop_at_3), state: Some(&state), state_every: 0, identity: identity.clone(), snapshots: None, grad_sync: None };
    let (first, _) = model::fit_controlled(build(), objective(), &opts, None, control).map_err(err)?;
    if !first.interrupted || !state.with_extension("offdevice").is_dir() {
        return Err("toyseq2seq: the stopped run left no resume state with its off-device generation".into());
    }
    let control = model::FitControl { state: Some(&state), identity, ..Default::default() };
    let (second, resumed) = model::fit_controlled(build(), objective(), &opts, None, control).map_err(err)?;
    if second.resumed_at != Some(3) {
        return Err(format!("toyseq2seq: resumed at {:?}, not 3", second.resumed_at));
    }
    let (a, b) = (straight.final_loss.unwrap_or(f32::NAN), second.final_loss.unwrap_or(f32::NAN));
    if a.to_bits() != b.to_bits() {
        return Err(format!("toyseq2seq: the resumed run ends at loss {b}, the uninterrupted one at {a}"));
    }
    if let Some(((n, _), _)) = weights(&straight_model).into_iter().zip(weights(&resumed)).find(|((_, x), (_, y))| x != y) {
        return Err(format!("toyseq2seq: {n} differs between the resumed and the uninterrupted run"));
    }
    Ok(())
}
