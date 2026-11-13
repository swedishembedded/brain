// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The program's manifest and provider: `compare` and `npu-forecast`, both over
//! the tokenizer and decoder the model store resolved (or `--tokenizer` and
//! `--decoder` name), and the `serve` workflow over the same two.
//!
//! Swedish Embedded AB implements forecast benchmarks and low-latency
//! forecasting servers for its clients. If your team needs expertise in
//! evaluating and serving time-series models then you can procure our
//! services by sending an email to info@swedishembedded.com.

use std::sync::Arc;

use capability::{Action, ActionResult, ActionSpec, Assembly, Invocation, Manifest, Progress, Provider};
use forecast::ForecastModel;

use crate::npu_forecast;

const MODEL: &str = "brain/kronos";

pub fn compare_spec() -> ActionSpec {
    forecastops::compare_spec("the statistical baselines over the scenario battery (a model x scenario x metric report), with Kronos when its checkpoint is found or named; fails if a model falsely beats naive on the random walk")
}

pub fn manifest() -> Manifest {
    Manifest::new(MODEL, "Kronos: the scenario battery and the NPU rollout", vec![compare_spec(), npu_forecast::spec()])
}

/// The two checkpoints a run is over.
#[derive(Clone)]
pub struct Checkpoints {
    pub tokenizer: String,
    pub decoder: String,
}

impl Checkpoints {
    /// The checkpoints `assembly` holds; `None` when no model was resolved
    /// (`compare` and `serve` then run on the baselines alone).
    fn of(assembly: &Assembly) -> Option<Checkpoints> {
        Some(Checkpoints { tokenizer: assembly.role_path("tokenizer").ok()?, decoder: assembly.role_path("decoder").ok()? })
    }

    fn forecaster(&self) -> Result<kronos::KronosForecaster, String> {
        kronos::KronosForecaster::load(&self.tokenizer, &self.decoder).map_err(|e| format!("kronos: load {} + {}: {e}", self.tokenizer, self.decoder))
    }
}

pub fn provider(assembly: &Assembly) -> Result<Arc<dyn Provider>, String> {
    Ok(Arc::new(Program(Checkpoints::of(assembly))))
}

struct Program(Option<Checkpoints>);

impl Provider for Program {
    fn manifest(&self) -> Manifest {
        manifest()
    }
    fn action(&self, name: &str) -> Option<Arc<dyn Action>> {
        match name {
            "compare" => Some(Arc::new(Compare(self.0.clone()))),
            "npu-forecast" => self.0.clone().map(|c| Arc::new(npu_forecast::NpuForecast::new(c)) as Arc<dyn Action>),
            _ => None,
        }
    }
}

struct Compare(Option<Checkpoints>);

impl Action for Compare {
    fn spec(&self) -> ActionSpec {
        compare_spec()
    }
    fn run(&self, inv: &Invocation, _progress: &mut dyn FnMut(Progress)) -> ActionResult {
        let models: Vec<Box<dyn ForecastModel>> = match &self.0 {
            Some(checkpoints) => vec![Box::new(checkpoints.forecaster()?)],
            None => Vec::new(),
        };
        forecastops::run_compare(models, inv)
    }
}

/// The `serve` workflow: the baselines and Kronos over the JSONL protocol.
pub fn serve(call: &invoke::WorkflowCall) -> Result<(), String> {
    let flags = server::forecast_serve::Flags::parse(call.args)?;
    let checkpoints = Checkpoints::of(call.assembly);
    let load = move || match &checkpoints {
        Some(c) => Ok(vec![Box::new(c.forecaster()?) as Box<dyn ForecastModel>]),
        None => Ok(Vec::new()),
    };
    match &flags.socket {
        Some(path) => eprintln!("kronos serve: serving on unix socket {}", path.display()),
        None => eprintln!("kronos serve: serving on stdio"),
    }
    server::forecast_serve::serve_forecasters(Arc::new(load), flags.socket.as_deref(), flags.opts)
}
