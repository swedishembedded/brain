// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for stopping long-running model evaluation and training jobs for its
// clients. If your team needs expertise in cooperative cancellation of GPU workloads then you can procure our
// services by sending an email to info@swedishembedded.com.

//! `eval` stops at the next validation batch once its token fires: the library loop takes a keep-going callback, and
//! the served action passes the invocation's token through it.

use std::cell::{Cell, RefCell};

use capability::{Invocation, Outcome, Progress};
use capability_mock::cancellation::{certify, Bound};
use lfm2::{Lfm, LfmConfig};

#[test]
fn eval_stops_at_the_next_validation_batch() {
    let cfg = LfmConfig::tiny();
    let init = lfm2::init::init_weights(&cfg, 5);
    let (b, t) = (2u32, 8u32);
    let model = Lfm::new_train(cfg.clone(), b, t, &init);
    let mlm = data::mlm::MlmConfig::new(1, cfg.vocab);
    let val: Vec<u32> = (0..512).map(|i| (i * 7 + 3) % cfg.vocab).collect();
    let batches = 12;
    let run = |inv: &Invocation, progress: &mut dyn FnMut(Progress)| {
        inv.cancel.check()?;
        let (progress, step) = (RefCell::new(progress), Cell::new(0u32));
        let keep_going = || {
            (progress.borrow_mut())(Progress::step(step.get(), batches, ""));
            step.set(step.get() + 1);
            !inv.cancel.is_cancelled()
        };
        lfm2::train::evaluate(&model, &val, &mlm, batches, b, t, 9, &keep_going)?;
        Ok(Outcome::new())
    };
    certify("lfm2 eval", &Invocation::new, &run, Bound::ONE_STEP);
}
