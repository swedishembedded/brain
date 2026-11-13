// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The forecasting event server: the statistical baselines and whichever
//! foundation forecasters a program loads, behind the one JSONL protocol on
//! stdio or an owner-only Unix socket, so a local client (`brain-py`'s
//! `BrainStdio.connect`) can drive forecasts and backtests warm across many
//! requests - the right shape for a batch job that forecasts a whole universe.
//!
//! Every connection builds its own models, so per-instance state never crosses
//! threads; sharing one replica across workers is the deferred throughput
//! optimization.
//!
//! Swedish Embedded AB implements low-latency forecasting services that keep
//! models warm across a batch job for its clients. If your team needs
//! expertise in serving time-series models then you can procure our services
//! by sending an email to info@swedishembedded.com.

use std::path::Path;
use std::sync::Arc;

use forecast::ForecastModel;

use crate::transport::SessionFactory;
use crate::{serve_stdio, serve_unix, ControllerSession, ServeOpts};

/// The flags of a program's `serve` workflow, after its weight flags are taken:
/// `--socket PATH` (else stdio) and `--max-connections N`.
#[derive(Clone, Debug)]
pub struct Flags {
    pub socket: Option<std::path::PathBuf>,
    pub opts: ServeOpts,
}

impl Flags {
    /// What the workflow's `usage` line names.
    pub const USAGE: &'static str = "[--socket PATH] [--max-connections N]";

    pub fn parse(args: &[String]) -> Result<Flags, String> {
        let mut flags = Flags { socket: None, opts: ServeOpts::default() };
        let mut it = args.iter();
        while let Some(flag) = it.next() {
            let mut value = |what: &str| it.next().ok_or_else(|| format!("{what} needs a value"));
            match flag.as_str() {
                "--socket" => flags.socket = Some(value("--socket")?.into()),
                "--max-connections" => {
                    let n = value("--max-connections")?;
                    flags.opts.max_connections = n.parse().ok().filter(|&n| n > 0).ok_or_else(|| format!("--max-connections {n}: expected a positive count"))?;
                }
                other => return Err(format!("unknown flag {other}; it takes {}", Flags::USAGE)),
            }
        }
        Ok(flags)
    }
}

/// Builds the foundation forecasters one connection serves beside the baselines.
pub type LoadModels = Arc<dyn Fn() -> Result<Vec<Box<dyn ForecastModel>>, String> + Send + Sync>;

/// One connection's controller: the baselines, then `models`.
fn controller(models: Vec<Box<dyn ForecastModel>>) -> runtime::Controller {
    let mut registry = runtime::Registry::new();
    for model in fcbench::baselines::default_set().into_iter().chain(models) {
        registry.register_forecast(Arc::from(model));
    }
    runtime::Controller::new(registry)
}

/// Serve on `socket` (an owner-only Unix socket) when given, else on stdio.
///
/// The models are loaded once up front, and the instance dropped, so a missing
/// or unreadable checkpoint fails the start rather than the first request; a
/// socket connection then loads its own.
pub fn serve_forecasters(load: LoadModels, socket: Option<&Path>, opts: ServeOpts) -> Result<(), String> {
    let first = load()?;
    match socket {
        Some(path) => {
            drop(first);
            let make: SessionFactory = Arc::new(move || {
                let models = load().unwrap_or_else(|e| {
                    eprintln!("forecast server: a connection could not load its models, serving the baselines only: {e}");
                    Vec::new()
                });
                Box::new(ControllerSession::new(controller(models)))
            });
            serve_unix(path, make, opts).map_err(|e| format!("serve on {}: {e}", path.display()))
        }
        None => {
            let mut session = ControllerSession::new(controller(first));
            serve_stdio(&mut session).map_err(|e| format!("serve on stdio: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{pump_connection, Session};
    use std::io::Cursor;

    #[test]
    fn the_flags_name_a_socket_and_a_connection_cap_and_refuse_anything_else() {
        let args = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let none = Flags::parse(&[]).unwrap();
        assert_eq!((none.socket, none.opts.max_connections), (None, ServeOpts::default().max_connections));
        let f = Flags::parse(&args(&["--socket", "/s", "--max-connections", "3"])).unwrap();
        assert_eq!((f.socket.as_deref(), f.opts.max_connections), (Some(Path::new("/s")), 3));
        assert!(Flags::parse(&args(&["--max-connections", "0"])).is_err());
        assert!(Flags::parse(&args(&["--socket"])).unwrap_err().contains("needs a value"));
        assert!(Flags::parse(&args(&["--horizon", "1"])).unwrap_err().contains("unknown flag"));
    }

    /// A session over just the baselines answers a capabilities request with
    /// them listed, and one given an extra model lists that too.
    #[test]
    fn a_connection_serves_the_baselines_and_any_model_it_is_given() {
        let ask = |models: Vec<Box<dyn ForecastModel>>| {
            let mut session: Box<dyn Session> = Box::new(ControllerSession::new(controller(models)));
            let line = events::encode_envelope(&events::Envelope::with_id(Some("c".into()), events::Event::CapabilitiesRequest));
            let mut out = Vec::new();
            pump_connection(Cursor::new(format!("{line}\n").into_bytes()), &mut out, session.as_mut()).unwrap();
            String::from_utf8(out).unwrap()
        };
        let bare = ask(Vec::new());
        assert!(bare.contains("naive"), "the baselines are always served: {bare}");
        assert!(!bare.contains("seasonal_naive"), "not a default baseline: {bare}");
        let with = ask(vec![Box::new(fcbench::SeasonalNaive { season: 7 })]);
        assert!(with.contains("seasonal_naive") && with.contains("naive"), "the given model is served beside the baselines: {with}");
    }
}
