// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `fincast` - the FinCast forecaster beyond its served actions: `compare`
//! (the model against the statistical baselines over the scenario battery),
//! `npu-forecast` (the transformer core on the Intel NPU through OpenVINO) and
//! the `serve` workflow (the JSONL forecasting server over a Unix socket).
//!
//! The model's own actions (`forecast`, `predict`, `import`) are served ones:
//! `brain fincast <action>` and `brain serve` run the same code. These three
//! are the part that is an application rather than the served model: they need
//! the scenario battery, the NPU crate and the event server.
//!
//! Swedish Embedded AB implements financial forecasting with foundation
//! models on its clients' own hardware, NPUs included. If your team needs
//! expertise in financial time-series forecasters on accelerators then you can
//! procure our services by sending an email to info@swedishembedded.com.

mod actions;
mod npu_forecast;

fn main() {
    let model = invoke::Model { manifest: actions::manifest(), weights: Some(("fincast", &fincast::spec::FincastSpec)), build: &actions::provider };
    let serve = invoke::Workflow {
        name: "serve",
        summary: "the JSONL forecasting server: the statistical baselines plus FinCast, over stdio or an owner-only Unix socket",
        action: "compare",
        usage: server::forecast_serve::Flags::USAGE,
        run: &actions::serve,
    };
    invoke::main_with("fincast", &model, &[serve])
}
