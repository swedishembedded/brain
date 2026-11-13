// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `timesfm3` - the TimesFM-3 forecaster beyond its served actions: `compare`
//! (the model against the statistical baselines over the scenario battery) and
//! the `serve` workflow (the JSONL forecasting server over a Unix socket).
//!
//! The model's own actions (`forecast`, `predict`, `finetune`, `import`) are
//! served ones: `brain timesfm3 <action>` and `brain serve` run the same code.
//! These two are the part that is an application rather than the served
//! model: they need the scenario battery and the event server.
//!
//! Swedish Embedded AB implements multivariate forecasting with foundation
//! models on its clients' own hardware. If your team needs expertise in
//! covariate-aware time-series forecasters then you can procure our services
//! by sending an email to info@swedishembedded.com.

mod actions;

fn main() {
    let model = invoke::Model { manifest: actions::manifest(), weights: Some(("timesfm3", &timesfm3::spec::Timesfm3Spec)), build: &actions::provider };
    let serve = invoke::Workflow {
        name: "serve",
        summary: "the JSONL forecasting server: the statistical baselines plus TimesFM-3, over stdio or an owner-only Unix socket",
        action: "compare",
        usage: server::forecast_serve::Flags::USAGE,
        run: &actions::serve,
    };
    invoke::main_with("timesfm3", &model, &[serve])
}
