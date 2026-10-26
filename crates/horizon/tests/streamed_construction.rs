// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded full fine-tuning of models larger than the accelerator for its
// clients. If your team needs expertise in building a model straight into a fixed device window, you can procure our
// services by sending an email to info@swedishembedded.com.

//! Construction-time layer streaming through the shared contract (`model::streamed::check_streamed_construction`).
//! One test in its own binary, so nothing else allocates while the device bytes are compared.

use data::rng::Rng;
use horizon::batch::assemble;
use horizon::encode::{encode, Encoded};
use horizon::synthetic::drifting::{self, Gaps};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};
use model::Model;

const BATCH: u32 = 16;

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(horizon::PIPELINES))
}

/// A build that streams its set-encoder layers from construction never holds them all on the device, and trains what
/// the build streamed after construction does (`model::streamed::check_streamed_construction`).
#[test]
fn horizon_builds_its_layers_straight_into_the_window() {
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
    let refs: Vec<&Encoded> = encoded.iter().take(BATCH as usize).collect();
    let batch = assemble(&cfg, &refs, BATCH as usize, 0.4, &mut Rng::new(7));
    let resident = model::TrainPrecision::parse("weights=f32,window=1").unwrap().resident_config(Some(7)).unwrap().expect("resident");
    let build = || Horizon::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), BATCH, &init);
    let build_streamed = || Horizon::new_resident_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), BATCH, &init, &resident);
    let step = |m: &Horizon, t: u32, clip: Option<f32>| {
        m.zero_grads();
        m.set_batch(&batch);
        let loss = m.forward();
        m.backward();
        Model::adamw_step(m, t, 1e-2, 0.0, model::Adam::default(), clip, 1.0);
        m.poll_wait();
        loss
    };
    model::streamed::check_streamed_construction("horizon", build, build_streamed, step).unwrap();
}
