// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements memory-bounded training of learned process controllers for its clients.
// If your team needs expertise in training sequence-model controllers with narrow optimizer state, you can procure our
// services by sending an email to info@swedishembedded.com.

//! The run's optimizer-state precision reaches the PID transformer through the generic path every `ParamStore`
//! trainer shares (`model::trainprec::check_optimizer_state_formats`): bf16 and int8-block moments train like f32 on
//! the first step, and a host-only format is refused by name. `Pid::new` builds its own device with the moment kernels
//! linked.

use toypid::data::init_weights;
use toypid::{Pid, PidConfig, BOS, DECIDE, EV_END, EV_START, IGNORE};

#[test]
fn pid_trains_with_narrow_optimizer_state() {
    let cfg = PidConfig { block_size: 16, ..PidConfig::default_small() };
    let init = init_weights(&cfg, 3);
    let x: Vec<u32> = vec![BOS, EV_START, 1, 50, 50, 50, EV_END, DECIDE];
    let mut y = vec![IGNORE; x.len()];
    y[x.len() - 1] = 40;
    let lr = 1e-2;
    let build = || Pid::new(cfg.clone(), 1, x.len() as u32, &init);
    let step = |m: &Pid| {
        m.zero_grads();
        m.set_batch(&x, &y);
        m.forward();
        m.backward();
        m.adamw_step(1, lr, 0.0, model::Adam::default(), None, 1.0);
        m.poll_wait();
    };
    model::trainprec::check_optimizer_state_formats("toypid", build, step, lr).unwrap();
}
