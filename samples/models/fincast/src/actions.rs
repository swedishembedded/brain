// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The program's manifest and provider: `compare` and `npu-forecast`, both over
//! the checkpoint the model store resolved (or `--weights` names), and the
//! `serve` workflow over the same one.
//!
//! Swedish Embedded AB implements forecast benchmarks and low-latency
//! forecasting servers for its clients. If your team needs expertise in
//! evaluating and serving time-series models then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Assembly, Invocation, Manifest, Progress, Provider};
use forecast::ForecastModel;

use crate::npu_forecast;

const MODEL: &str = "brain/fincast";

pub fn compare_spec() -> ActionSpec {
    forecastops::compare_spec("the statistical baselines over the scenario battery (a model x scenario x metric report), with FinCast when its checkpoint is found or named; fails if a model falsely beats naive on the random walk. Slow on a CPU: keep windows small")
}

pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "FinCast: the scenario battery and the NPU core", vec![compare_spec(), npu_forecast::spec()])
}

pub fn provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(Program(assembly.role_path("weights").ok())))
}

fn forecaster(weights: &str) -> Result<fincast::FincastForecaster, String> {
    fincast::FincastForecaster::load(weights).map_err(|e| format!("fincast: load {weights}: {e}"))
}

/// The checkpoint a run is over; `None` when no model was resolved (`compare`
/// and `serve` then run on the baselines alone).
struct Program(Option<String>);

impl Provider for Program {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "compare" => Some(Arc::new(Compare(self.0.clone()))),
            "npu-forecast" => self.0.clone().map(|w| Arc::new(npu_forecast::NpuForecast::new(w)) as Arc<dyn Action>),
            _ => None,
        }
    }
}

struct Compare(Option<String>);

impl Action for Compare {
    fn spec(&self) -> ActionSpec {
        compare_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let models: Vec<Box<dyn ForecastModel>> = match &self.0 {
            Some(weights) => vec![Box::new(forecaster(weights)?)],
            None => Vec::new(),
        };
        forecastops::run_compare(models, inv)
    }
}

/// The `serve` workflow: the baselines and FinCast over the JSONL protocol.
pub fn serve(call: &invoke::WorkflowCall) -> Result<(), String> {
    let flags = server::forecast_serve::Flags::parse(call.args)?;
    let weights = call.assembly.role_path("weights").ok();
    let load = move || match &weights {
        Some(w) => Ok(vec![Box::new(forecaster(w)?) as Box<dyn ForecastModel>]),
        None => Ok(Vec::new()),
    };
    match &flags.socket {
        Some(path) => eprintln!("fincast serve: serving on unix socket {}", path.display()),
        None => eprintln!("fincast serve: serving on stdio"),
    }
    server::forecast_serve::serve_forecasters(Arc::new(load), flags.socket.as_deref(), flags.opts)
}
