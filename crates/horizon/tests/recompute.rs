// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of clinical time-to-event models for its clients.
// If your team needs expertise in training risk models within a fixed device budget, you can procure our services by
// sending an email to info@swedishembedded.com.

//! Activation recompute through the shared contract (`model::saved::check_recompute`): a build that keeps only the
//! residual between encoder layers and replays each layer before its backward computes the loss and gradients of the
//! build that keeps every layer, bit for bit, on fewer device bytes. One test in its own binary, so nothing else
//! allocates while the device bytes are compared.

use data::rng::Rng;
use horizon::batch::assemble;
use horizon::encode::{encode, Encoded};
use horizon::synthetic::drifting::{self, Gaps};
use horizon::vocab::{FitOptions, Vocab};
use horizon::{Horizon, HorizonConfig};
use model::Model as _;

#[test]
fn horizon_recomputes_every_layer_exactly() {
    let gaps = Gaps { last: (0.0, 2.0), between: (0.5, 3.0), visits: (1, 5) };
    let (subjects, _) = drifting::population(400, 5, &gaps, 6.0);
    let codes = vec![drifting::CODE.to_string()];
    let vocab = Vocab::fit(&subjects, &codes, &codes, &FitOptions { knots: 9, min_count: 1 }).unwrap();
    // Three encoder layers: every layer but the first shares its activation buffers.
    let mut cfg = HorizonConfig::tiny(vocab.len(), 1);
    cfg.d_model = 16;
    cfg.n_heads = 2;
    cfg.max_tokens = 8;
    cfg.n_layers = 3;
    let encoded: Vec<Encoded> = subjects.iter().take(30).map(|s| encode(s, &vocab, &cfg)).collect();
    let refs: Vec<&Encoded> = encoded.iter().collect();
    let batch = assemble(&cfg, &refs, 32, 0.4, &mut Rng::new(7));
    let init = horizon::init_weights(&cfg, 3);
    let build = || Horizon::new_on(gpu_core::testgpu::dev(horizon::PIPELINES), cfg.clone(), 32, &init);
    let step = |m: &Horizon| {
        m.zero_grads();
        m.set_batch(&batch);
        let loss = m.forward();
        m.backward();
        m.poll_wait();
        loss
    };
    model::saved::check_recompute("horizon", build, step).unwrap();
}
