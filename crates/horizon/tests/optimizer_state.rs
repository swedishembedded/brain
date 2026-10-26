// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of clinical time-to-event models for its clients.
// If your team needs expertise in training risk models with narrow optimizer state, you can procure our services by
// sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches Horizon through the generic path every `ParamStore` trainer shares
//! (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on the first step,
//! and a host-only format is refused by name.

use data::rng::Rng;
use horizon::batch::assemble;
use horizon::encode::{encode, Encoded};
use horizon::synthetic::drifting::{self, Gaps};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};
use model::Model as _;

fn pipelines() -> &'static [(&'static str, &'static str)] {
    static LIST: std::sync::OnceLock<Vec<(&'static str, &'static str)>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| optim::with_moment_kernels(horizon::PIPELINES))
}

#[test]
fn horizon_trains_with_narrow_optimizer_state() {
    let gaps = Gaps { last: (0.0, 2.0), between: (0.5, 3.0), visits: (1, 5) };
    let (subjects, _) = drifting::population(400, 5, &gaps, 6.0);
    let codes = vec![drifting::CODE.to_string()];
    let vocab = Vocab::fit(&subjects, &codes, &codes, &FitOptions { knots: 9, min_count: 1 }).unwrap();
    let mut cfg = HorizonConfig::tiny(vocab.len(), 1);
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.max_tokens = 8;
    let encoded: Vec<Encoded> = subjects.iter().take(30).map(|s| encode(s, &vocab, &cfg)).collect();
    let refs: Vec<&Encoded> = encoded.iter().collect();
    let batch = assemble(&cfg, &refs, 32, 0.4, &mut Rng::new(7));
    let init = horizon::init_weights(&cfg, 3);
    let lr = 1e-2;
    let build = || Horizon::new_on(gpu_core::testgpu::dev(pipelines()), cfg.clone(), 32, &init);
    let step = |m: &Horizon| {
        m.zero_grads();
        m.set_batch(&batch);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("horizon", build, step, lr).unwrap();
}
