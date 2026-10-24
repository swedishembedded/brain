// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A single-machine **distributed-training profile**: one tiny GPT trained as
//! `world` data-parallel replicas over each transport, measured end to end.
//!
//! Swedish Embedded AB implements data-parallel training drivers whose cost
//! and correctness can be measured before a cluster exists, for teams that
//! need to know whether a distributed run scales before paying for one. If
//! your team needs expertise in distributed-training profiling and replica
//! consistency, you can procure our services by sending an email to
//! info@swedishembedded.com.
//!
//! The driver sweeps `world` sizes over three [`Transport`]s:
//!
//! * `host` - threads in one process exchanging through
//!   [`HostCollective`](model::HostCollective)'s host staging area;
//! * `network` - the same threads over [`NetworkCollective`](model::NetworkCollective)'s
//!   coordinator-star TCP transport on loopback;
//! * `shaped` - [`HostCollective`](model::HostCollective) behind a
//!   [`SimulatedLink`] (propagation delay, a bandwidth cap and deterministic
//!   jitter), which is how a link that is not the one under the desk is
//!   simulated without a second machine.
//!
//! Every run trains the config in [`ProfileConfig`] through the public
//! [`model::fit_controlled_sharded`] seam, wrapping its collective in a
//! [`TimedCollective`] so the reported communication cost is the ops' own
//! measured wall time. The result is one JSON document: per-run step times,
//! tokens/s, communication bytes/time/share, throughput speedup against the
//! same transport's one-rank run, parallel efficiency, and whether every
//! replica ended bit-identical.
//!
//! **What the numbers do and do not include.** Speedup is the tokens/s ratio
//! against world 1, so parallel efficiency (speedup / world) is the same as
//! the step-time ratio for a constant amount of work per rank. The recorded
//! collective bytes are the payload each rank staged, not bytes on a NIC, and
//! the host transport moves them through host RAM rather than a device link.
//! There is no scheduler and no real cluster: the loopback transport's latency
//! is the kernel's TCP stack on one box, and the shaped transport's delay is
//! the configured model of a link, not a measurement of one.

use std::net::TcpListener;
use std::path::Path;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use gpt2::GptConfig;
use model::collective::{SimulatedLink, TimedCollective};
use model::train::{build_or_resume, causal_lm, fit_controlled_sharded, load_dataset_with_itos, FitOpts, FitControl, StepReport};
use model::{Collective, GradSync, HostCollective, Model, NetworkCollective, ShapedCollective, Snapshot};

use serde_json::{json, Value};

/// Which transport a run exchanges gradients through.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Threads sharing one process's host-staged collective.
    Host,
    /// The coordinator-star TCP collective on loopback.
    Network,
    /// The host collective behind a [`SimulatedLink`].
    Shaped,
}

impl Transport {
    pub fn name(self) -> &'static str {
        match self {
            Transport::Host => "host",
            Transport::Network => "network-loopback",
            Transport::Shaped => "shaped",
        }
    }

    /// Parse the spelling used on the command line and in the JSON.
    pub fn parse(name: &str) -> Option<Transport> {
        match name {
            "host" => Some(Transport::Host),
            "network" | "network-loopback" => Some(Transport::Network),
            "shaped" => Some(Transport::Shaped),
            _ => None,
        }
    }
}

/// What to profile. The model dimensions match the tiny GPT the distributed
/// training tests use, so a run is small enough to compare transports in a
/// test and still large enough for the reduction to matter.
#[derive(Clone, Debug)]
pub struct ProfileConfig {
    /// Steady-state optimizer steps reported per run. The run executes one
    /// more so the callback stream yields exactly this many step intervals.
    pub steps: u32,
    pub batch_size: u32,
    pub block_size: u32,
    pub n_layers: u32,
    pub d_model: u32,
    pub n_heads: u32,
    pub d_ff: u32,
    pub worlds: Vec<usize>,
    pub transports: Vec<Transport>,
    /// The link the `shaped` transport charges for.
    pub link: SimulatedLink,
    pub seed: u64,
    /// Examples in the generated calculator corpus.
    pub dataset_examples: usize,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        ProfileConfig {
            steps: 3,
            batch_size: 2,
            block_size: 16,
            n_layers: 1,
            d_model: 32,
            n_heads: 2,
            d_ff: 64,
            worlds: vec![1, 2, 4],
            transports: vec![Transport::Host, Transport::Network, Transport::Shaped],
            // A 100 Gb/s link with 100 us of one-way propagation.
            link: SimulatedLink {
                one_way_latency: Duration::from_micros(100),
                bandwidth_bytes_per_sec: 12.5e9,
                jitter: Duration::ZERO,
                seed: 0xD1_57_AB_1E,
                fail_after_ops: None,
            },
            seed: 7,
            dataset_examples: 2000,
        }
    }
}

impl ProfileConfig {
    fn fit_opts(&self) -> FitOpts {
        FitOpts {
            steps: self.steps + 1,
            batch_size: self.batch_size,
            block_size: self.block_size,
            lr: 1e-2,
            warmup: 0,
            decay_iters: self.steps + 1,
            eval_interval: 0,
            checkpoint_secs: 0,
            mask_before: Some('='),
            mask_per_line: true,
            seed: self.seed,
            ..Default::default()
        }
    }

    fn model_config(&self) -> GptConfig {
        GptConfig {
            vocab: 0,
            block_size: self.block_size,
            n_layers: self.n_layers,
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_ff: self.d_ff,
        }
    }
}

/// One rank's result: its step intervals and the collective accounting over
/// its own timed wrapper.
type RankResult = Result<(RankOutcome, Snapshot), String>;

/// One rank's trained result. The weights are kept as raw bits, not `f32`, so
/// "bit-identical" is decided by the same relation the model uses to store
/// them rather than by a float comparison.
struct RankOutcome {
    step_times: Vec<Duration>,
    weights: Vec<(String, Vec<u32>)>,
}

impl RankOutcome {
    fn same_weights(&self, other: &RankOutcome) -> bool {
        self.weights.len() == other.weights.len()
            && self
                .weights
                .iter()
                .zip(&other.weights)
                .all(|((an, av), (bn, bv))| an == bn && av == bv)
    }
}

/// One completed run, before the cross-run ratios are attached.
struct RunResult {
    transport: Transport,
    world: usize,
    step_times: Vec<Duration>,
    /// Token positions processed per step across every rank.
    tokens_per_step: u64,
    /// Rank 0's collective accounting.
    snapshot: Snapshot,
    replicas_bit_identical: bool,
}

/// Train one rank as part of `world`, returning its step intervals, weight
/// bits and the collective accounting taken over its own wrapper.
fn train_rank(timed: &TimedCollective<'_>, rank: usize, world: usize, cfg: &ProfileConfig, data: &Path) -> Result<(RankOutcome, Snapshot), String> {
    let opts = cfg.fit_opts();
    let (train, val, batch_cfg, vocab, itos) =
        load_dataset_with_itos(data, &opts).map_err(|e| format!("rank {rank}: load the profile dataset: {e}"))?;
    let model = build_or_resume::<gpt2::Gpt>(cfg.model_config(), &opts, None, vocab);
    let objective = causal_lm::<gpt2::Gpt>(train, val, batch_cfg, itos);

    let mut stamps: Vec<Instant> = Vec::with_capacity(cfg.steps as usize + 2);
    let mut hook = |_report: &StepReport| {
        stamps.push(Instant::now());
        true
    };
    let control = FitControl { on_step: Some(&mut hook), ..Default::default() };
    let sync = GradSync { collective: timed as &dyn Collective, rank };

    let (report, model) = fit_controlled_sharded(model, objective, &opts, None, control, Some(sync), Some((rank, world)))
        .map_err(|e| format!("rank {rank}: the distributed run failed: {e}"))?;
    let snapshot = timed.snapshot();

    if stamps.len() < 2 {
        return Err(format!(
            "rank {rank}: the run completed {} step(s), too few to measure a step interval",
            report.steps_completed
        ));
    }
    let step_times = stamps.windows(2).map(|pair| pair[1].duration_since(pair[0])).collect();

    let weights = model
        .param_names()
        .into_iter()
        .map(|name| {
            let values = model.read_weight(&name);
            (name, values.iter().map(|value| value.to_bits()).collect())
        })
        .collect();
    Ok((RankOutcome { step_times, weights }, snapshot))
}

/// Point the process at the CPU backend so the profile never depends on a
/// device being present, and every transport measures the same arithmetic.
/// A caller that already selected a backend keeps its choice.
fn select_cpu_if_unset() {
    if !gpu_core::backend_selected() {
        gpu_core::set_default_backend(gpu_core::Backend::Cpu);
    }
}

/// Run every configured transport at every configured world size over one
/// generated dataset, returning the JSON report described in the module doc.
///
/// A run that cannot start fails with the reason naming the rank and the
/// transport; nothing is skipped.
pub fn run_profile(cfg: &ProfileConfig) -> Result<Value, String> {
    select_cpu_if_unset();
    if cfg.worlds.is_empty() {
        return Err("the profile has no world sizes to run".to_string());
    }
    if cfg.transports.is_empty() {
        return Err("the profile has no transports to run".to_string());
    }

    let data = std::env::temp_dir().join(format!("brain-distributed-profile-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    data::prepare::prepare(data::prepare::Dataset::Calculator, &data, cfg.dataset_examples, cfg.seed)
        .map_err(|e| format!("prepare the profile dataset in {}: {e}", data.display()))?;

    let mut runs = Vec::new();
    for &transport in &cfg.transports {
        for &world in &cfg.worlds {
            runs.push(run_once(cfg, &data, transport, world));
        }
    }
    let _ = std::fs::remove_dir_all(&data);
    let runs: Vec<RunResult> = runs.into_iter().collect::<Result<_, _>>()?;

    // Throughput of one rank, per transport, for the speedup denominator.
    let baseline = |transport: Transport| {
        runs.iter()
            .find(|run| run.world == 1 && run.transport == transport)
            .map(throughput)
    };

    let run_json: Vec<Value> = runs
        .iter()
        .map(|run| {
            let tokens_per_s = throughput(run);
            let speedup = baseline(run.transport).map(|base| if base > 0.0 { tokens_per_s / base } else { 0.0 });
            let parallel_efficiency = speedup.map(|speedup| speedup / run.world as f64);
            json!({
                "transport": run.transport.name(),
                "world": run.world,
                "step_times_ms": run.step_times.iter().map(duration_ms).collect::<Vec<f64>>(),
                "mean_step_ms": duration_ms(&mean(&run.step_times)),
                "tokens_per_s": tokens_per_s,
                "comm": run.snapshot.to_json(),
                "comm_share": comm_share(run),
                "speedup": speedup,
                "parallel_efficiency": parallel_efficiency,
                "replicas_bit_identical": run.replicas_bit_identical,
            })
        })
        .collect();

    // An unbounded link has no Gb/s figure; JSON has no infinity, so it is null.
    let bandwidth_gbps = if cfg.link.bandwidth_bytes_per_sec.is_finite() {
        json!(cfg.link.bandwidth_bytes_per_sec * 8.0 / 1e9)
    } else {
        Value::Null
    };

    Ok(json!({
        "config": {
            "backend": gpu_core::backend_name(),
            "steps": cfg.steps,
            "batch_size": cfg.batch_size,
            "block_size": cfg.block_size,
            "seed": cfg.seed,
            "model": {
                "n_layers": cfg.n_layers,
                "d_model": cfg.d_model,
                "n_heads": cfg.n_heads,
                "d_ff": cfg.d_ff,
            },
            "link": {
                "one_way_latency_us": cfg.link.one_way_latency.as_secs_f64() * 1e6,
                "bandwidth_gbps": bandwidth_gbps,
                "jitter_us": cfg.link.jitter.as_secs_f64() * 1e6,
            },
        },
        "runs": run_json,
    }))
}

/// One transport at one world size: spawn every rank as a scope thread, give
/// each its own timed wrapper, and collect the rank results. A failed rank is
/// returned with its reason rather than silently dropped.
fn run_once(cfg: &ProfileConfig, data: &Path, transport: Transport, world: usize) -> Result<RunResult, String> {
    if world == 0 {
        return Err("a world size of zero has no ranks".to_string());
    }
    let host = matches!(transport, Transport::Host | Transport::Shaped).then(|| HostCollective::new(world));
    let listener = if transport == Transport::Network {
        Some(TcpListener::bind("127.0.0.1:0").map_err(|e| format!("{} world {world}: bind the loopback listener: {e}", transport.name()))?)
    } else {
        None
    };
    let address = listener.as_ref().map(|listener| listener.local_addr().map(|a| a.to_string())).transpose().map_err(|e| {
        format!("{} world {world}: read the loopback listener address: {e}", transport.name())
    })?;

    let results: Mutex<Vec<Option<RankResult>>> = Mutex::new((0..world).map(|_| None).collect());
    thread::scope(|scope| {
        for rank in 0..world {
            let results = &results;
            let host = host.as_ref();
            let listener = listener.as_ref();
            let address = address.as_deref();
            scope.spawn(move || {
                let outcome = match transport {
                    Transport::Host => {
                        let timed = TimedCollective::new(&**host.expect("the host transport has a collective"));
                        train_rank(&timed, rank, world, cfg, data)
                    }
                    Transport::Shaped => {
                        let shaped = ShapedCollective::new(&**host.expect("the shaped transport has a host collective"), cfg.link);
                        let timed = TimedCollective::new(&shaped);
                        train_rank(&timed, rank, world, cfg, data)
                    }
                    Transport::Network => {
                        let joined = if rank == 0 {
                            NetworkCollective::coordinator(listener.expect("the network transport has a listener"), world)
                        } else {
                            NetworkCollective::worker(rank, world, address.expect("the network transport has an address"))
                        };
                        match joined {
                            Ok(net) => {
                                let timed = TimedCollective::new(&net);
                                train_rank(&timed, rank, world, cfg, data)
                            }
                            Err(e) => Err(format!("rank {rank}: join the loopback group: {e}")),
                        }
                    }
                };
                let mut slots = results.lock().unwrap();
                slots[rank] = Some(outcome);
            });
        }
    });

    let mut outcomes: Vec<(RankOutcome, Snapshot)> = Vec::with_capacity(world);
    for (rank, slot) in results.into_inner().unwrap().into_iter().enumerate() {
        match slot {
            Some(Ok(outcome)) => outcomes.push(outcome),
            Some(Err(e)) => return Err(e),
            None => return Err(format!("rank {rank} returned no result")),
        }
    }

    let replicas_bit_identical = outcomes.iter().skip(1).all(|(outcome, _)| outcome.same_weights(&outcomes[0].0));
    let (first, snapshot) = outcomes.swap_remove(0);
    // Each rank trains one batch of `block` positions per step; the run's work
    // is that batch across every rank.
    let tokens_per_step = world as u64 * cfg.batch_size as u64 * cfg.block_size as u64;
    Ok(RunResult { transport, world, step_times: first.step_times, tokens_per_step, snapshot, replicas_bit_identical })
}

fn throughput(run: &RunResult) -> f64 {
    let seconds = run.step_times.iter().sum::<Duration>().as_secs_f64();
    if seconds <= 0.0 {
        0.0
    } else {
        run.tokens_per_step as f64 / seconds
    }
}

fn mean(samples: &[Duration]) -> Duration {
    if samples.is_empty() {
        return Duration::ZERO;
    }
    let total: Duration = samples.iter().sum();
    total / samples.len() as u32
}

fn duration_ms(d: &Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn comm_share(run: &RunResult) -> f64 {
    let step_seconds = run.step_times.iter().sum::<Duration>().as_secs_f64();
    if step_seconds <= 0.0 {
        0.0
    } else {
        run.snapshot.total.as_secs_f64() / step_seconds
    }
}
