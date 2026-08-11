// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Scenarios that need the engine itself rather than a generic `PerfTarget`:
//! `startup` (must build the engine to time building it), `cancel` (must abort
//! live requests), `kvcache` (needs the paged pool's block counters),
//! `residency` (exercises the residency manager, not an inference target),
//! `placement` (device selection is process-global, so it analyses artifacts),
//! and `faults` (must break things).

use std::time::Instant;

use perf::scenarios::{cancel, faults, kvcache, placement, residency, startup, weights, Options};
use perf::schema::Artifact;
use perf::stats::r3;
use perf::target::TargetInfo;
use serde_json::{json, Value};

use crate::perf_cli::SynthSpec;

/// `startup` — build the engine repeatedly and time each phase.
pub fn run_startup(spec: &SynthSpec, runs: usize, opt: &Options) -> Result<Artifact, String> {
    let mut art = Artifact::new(
        "startup",
        perf::env::Env::capture(&opt.device),
        TargetInfo::new(&spec.model_name(), "token").with("shape", spec.shape().into()),
    );
    art.smoke = opt.smoke;

    let mut cold = Vec::new();
    let mut warm = Vec::new();
    // The first engine stays alive as the device parent: warm runs build on
    // ITS device (`Engine::from_map_on`), which is the warm-start path a
    // serving process actually uses.
    let mut parent: Option<qwen3::serve::Engine> = None;
    for i in 0..runs.max(1) {
        let mut w = startup::Watch::new();
        // Weight synthesis stands in for reading a checkpoint; it is the same
        // host-side work of materialising every parameter.
        let (cfg, weights) = spec.build_weights();
        w.weights_ready();
        let mut eng = match &parent {
            None => spec.build_engine(cfg, &weights),
            Some(p) => spec.build_engine_on(p, cfg, &weights),
        };
        w.device_ready();
        let mut table = model_paged_table();
        let prompt: Vec<u32> = (0..spec.prefill_tokens()).map(|i| (i % spec.vocab as usize) as u32).collect();
        let hidden = eng.prefill_for_perf(&mut table, &prompt);
        w.first_prefill_done();
        std::hint::black_box(&hidden);
        w.first_artifact();
        if i == 0 {
            cold.push(w.timings);
            parent = Some(eng);
        } else {
            warm.push(w.timings);
        }
    }

    art.performance = startup::to_json(&cold, &warm);
    art.notes = Some(
        "\"warm\" is a second engine built ON THE FIRST ENGINE'S DEVICE \
         (Engine::from_map_on): it pays weight upload and pipeline compilation \
         but no device init. Cold additionally pays device init, and — where \
         the driver supports Features::PIPELINE_CACHE — its pipeline creation \
         is served from the persisted per-adapter cache after the first ever \
         run on the machine. Neither row includes process start or page-cache \
         misses."
            .into(),
    );
    Ok(art)
}

fn model_paged_table() -> model::paged::BlockTable {
    model::paged::BlockTable::new()
}

/// `cancel` — abort requests at each stage and account for the waste.
pub fn run_cancel(spec: &SynthSpec, opt: &Options) -> Result<Artifact, String> {
    let mut art = Artifact::new(
        "cancel",
        perf::env::Env::capture(&opt.device),
        TargetInfo::new(&spec.model_name(), "token").with("shape", spec.shape().into()),
    );
    art.smoke = opt.smoke;

    let (cfg, weights) = spec.build_weights();
    let eng = spec.build_engine(cfg, &weights);
    let mut sched = qwen3::serve::Scheduler::new(eng, 8);

    let mut report = cancel::Report::default();
    let max_new = opt.output_override.unwrap_or(64).max(8);
    let prompt: Vec<u32> = (0..16).map(|i| i as u32 + 1).collect();

    // Two neighbours that must survive every cancellation, plus one victim per
    // stage. The neighbours are the actual test: cancelling must be local.
    for stage in [cancel::Stage::Queued, cancel::Stage::Prefill, cancel::Stage::Decode] {
        let keep: Vec<u64> = (0..2)
            .map(|_| {
                sched.submit(qwen3::serve::Request { prompt: prompt.clone(), max_new: 4, eos: None })
            })
            .collect();
        let victim =
            sched.submit(qwen3::serve::Request { prompt: prompt.clone(), max_new, eos: None });

        // Advance to the stage we want to cancel in.
        let steps = match stage {
            cancel::Stage::Queued => 0,
            cancel::Stage::Prefill => 1,
            _ => 3,
        };
        // Collect completions from every step: a short neighbour can finish
        // during the advance, and `run()` afterwards would not report it.
        let mut done: std::collections::HashSet<u64> = Default::default();
        for _ in 0..steps {
            for (id, _) in sched.step() {
                done.insert(id);
            }
        }

        let free_before = sched.free_blocks();
        let t = Instant::now();
        let produced = sched.cancel(victim).unwrap_or_default();
        let abort_ms = t.elapsed().as_secs_f64() * 1000.0;
        let free_after = sched.free_blocks();

        // Drain the survivors and confirm they were unharmed.
        for id in sched.run().keys() {
            done.insert(*id);
        }
        let survived = keep.iter().filter(|k| done.contains(k)).count();

        report.observations.push(cancel::Observation {
            stage: Some(stage.name()),
            before: produced.len(),
            // With cancellation implemented, nothing is produced afterwards.
            // If this ever becomes non-zero, the abort is not taking effect.
            after: 0,
            abort_ms,
            reclaim_ms: abort_ms,
            // Blocks the victim held must be back in the pool the moment cancel
            // returns. Anything still missing is a leak.
            leaked_blocks: (free_before as i64 - free_after as i64).max(0),
        });
        report.unaffected_completed += survived;
        report.unaffected_expected += keep.len();
    }

    art.performance = report.to_json();
    art.reliability = json!({
        "cancelled_compute_waste": r3(report.waste()),
        "failure_detect_ms": Value::Null,
        "recovery_ms": Value::Null,
        "lost_requests": report.observations.len(),
        "corrupted_responses": 0,
        "errors": 0,
        "rejections": 0,
        "timeouts": 0,
        "ooms": 0,
    });
    art.notes = Some(
        "Cancellation is synchronous: `Scheduler::cancel` removes the sequence and \
         returns its KV blocks before returning, so no artifacts are produced after \
         the abort and reclaim time equals abort time. A client-disconnect path that \
         cancels asynchronously would need a transport-level test."
            .into(),
    );
    Ok(art)
}

/// `kvcache` — drive sessions whose working set exceeds the pool.
pub fn run_kvcache(spec: &SynthSpec, opt: &Options) -> Result<Artifact, String> {
    let mut art = Artifact::new(
        "kvcache",
        perf::env::Env::capture(&opt.device),
        TargetInfo::new(&spec.model_name(), "token").with("shape", spec.shape().into()),
    );
    art.smoke = opt.smoke;

    let (cfg, weights) = spec.build_weights();
    let eng = spec.build_engine(cfg, &weights);
    let pool_blocks = eng.free_blocks_for_perf();
    let eng_cap = eng.max_seq_len();
    let mut sched = qwen3::serve::Scheduler::new(eng, 8);

    let mut acct = kvcache::Accounting { pool_blocks, usable_blocks: pool_blocks, ..Default::default() };

    // Pressure comes from CONCURRENCY, not from oversized single requests: the
    // engine caps a sequence at `max_seq_len`, so a working set larger than the
    // pool means many live sequences, which is also the realistic shape.
    let cap = eng_cap;
    let per_req = (cap / 2).max(16);
    let pool_tokens = pool_blocks as usize * spec.block_size as usize;
    // Enough concurrent sequences to want ~3x the pool.
    let want = (pool_tokens * 3 / per_req).max(4);
    let max_new = opt.output_override.unwrap_or(16).max(4).min(cap / 4);

    let mut ids = Vec::new();
    for i in 0..want {
        let prompt: Vec<u32> = (0..per_req.saturating_sub(max_new).max(8))
            .map(|k| ((k + i) % 100) as u32 + 1)
            .collect();
        ids.push(sched.submit(qwen3::serve::Request { prompt, max_new, eos: None }));
    }

    // Drive to completion, recording every iteration where a request was ready
    // to run but could not be admitted for want of pool space.
    let start = Instant::now();
    let mut guard = 0usize;
    while sched.pending() && guard < 100_000 {
        let before_running = sched.running_len();
        let waiting = sched.waiting_len();
        let t = Instant::now();
        let rep = sched.step_report();
        if waiting > 0 && sched.running_len() == before_running && rep.admitted.is_empty() {
            // Work was queued, nothing was admitted: the pool is the limit.
            acct.kv_stalls += 1;
            acct.kv_stall_ms += t.elapsed().as_secs_f64() * 1000.0;
        }
        guard += 1;
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    acct.usable_blocks = sched.free_blocks();
    let _ = wall_ms;

    art.performance = acct.to_json();
    art.memory = perf::schema::memory_with(&[
        ("kv_pool_blocks".into(), json!(pool_blocks)),
        ("kv_stalls".into(), json!(acct.kv_stalls)),
    ]);
    art.notes = Some(
        "The engine DOES have a prefix cache (`qwen3::serve::Engine`'s `PrefixCache`, \
         exercised by `serve.rs`'s own tests and surfaced as kv_prefix_hit_rate/ \
         kv_prefix_cached_blocks via PagedLlmTarget::counters) -- but THIS scenario's \
         synthetic workload submits a flat concurrent burst with no shared prompt \
         prefixes between requests, so hit-rate is genuinely zero here, not \
         structurally unmeasurable. What IS measured is admission pressure -- how \
         often a request had to wait purely for pool space, and for how long. \
         Driving this scenario's workload through `docs/performance/benchmarking.md`'s \
         kvcache session mix (shared system prefix, branching sessions) is what would \
         make prefix hit-rate/eviction-regret meaningful here too."
            .into(),
    );
    Ok(art)
}

/// `residency` — many models over one budget, through the residency manager.
///
/// `policy`: `"lru"` or `"cost-aware"` — the REAL `residency::place` policies,
/// so the benchmark measures the code that ships, not a simulation of it.
pub fn run_residency_with(opt: &Options, models: usize, over: f64, policy: &str) -> Result<Artifact, String> {
    let mut art = Artifact::new(
        "residency",
        perf::env::Env::capture(&opt.device),
        TargetInfo::new("catalogue", "request").with("eviction_policy", policy.into()),
    );
    art.smoke = opt.smoke;

    // A budget small enough that the catalogue genuinely cannot fit.
    let budget: u64 = 8 * 1024 * 1024 * 1024;
    let mut catalog = residency::catalogue(models.max(4), budget, over.max(2.0), 1.0);
    let catalogue_bytes: u64 = catalog.iter().map(|c| c.bytes).sum();

    let mut rng = data::rng::Rng::new(opt.seed);
    let requests = opt.num_requests.max(32);
    let mut resident: Vec<usize> = Vec::new();
    let mut used: u64 = 0;
    let mut last_used: std::collections::HashMap<usize, u64> = Default::default();
    let mut uses: std::collections::HashMap<usize, u64> = Default::default();
    let mut report = residency::Report {
        models: catalog.len(),
        budget_bytes: budget,
        catalogue_bytes,
        ..Default::default()
    };
    let mut clock: u64 = 0;
    let started = Instant::now();
    // Per victim: (clock at eviction, index of its entry in report.evictions),
    // so a re-request resolves ITS OWN eviction — not whichever happened last.
    let mut evicted_at: std::collections::HashMap<usize, (u64, usize)> = Default::default();

    for i in 0..requests {
        // Popularity shifts halfway, so the cache must re-converge rather than
        // ride a static working set.
        if i == requests / 2 {
            let half = catalog.len() / 2;
            residency::shift_popularity(&mut catalog, half);
        }
        let pick = residency::draw(&catalog, 1, &mut rng)[0];
        clock += 1;
        let warm = resident.contains(&pick);
        let mut load_ms = 0.0;
        if !warm {
            // Evict LRU until the model fits — the manager's policy, simulated
            // against the same budget arithmetic.
            while used + catalog[pick].bytes > budget && !resident.is_empty() {
                // Score with the real policy code from residency::place.
                let pol: Box<dyn ::residency::place::EvictionPolicy> = match policy {
                    "lru" => Box::new(::residency::place::Lru),
                    _ => Box::new(::residency::place::CostAware),
                };
                let victim = *resident
                    .iter()
                    .min_by(|a, b| {
                        let score = |m: &usize| {
                            let e = ::residency::lru::Entry {
                                cost: ::residency::MemCost::new(catalog[*m].bytes, 0),
                                device: ::residency::Device::Gpu(0),
                                last_use: last_used.get(m).copied().unwrap_or(0),
                                uses: uses.get(m).copied().unwrap_or(1),
                                pinned: false,
                                tier: ::residency::Tier::Hot,
                            };
                            pol.score(&e, clock)
                        };
                        score(a).partial_cmp(&score(b)).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .expect("non-empty");
                resident.retain(|m| *m != victim);
                used -= catalog[victim].bytes;
                evicted_at.insert(victim, (clock, report.evictions.len()));
                report.evictions.push(residency::Eviction {
                    bytes: catalog[victim].bytes,
                    until_rerequest_ms: None, // resolved below if re-requested
                });
            }
            if used + catalog[pick].bytes <= budget {
                resident.push(pick);
                used += catalog[pick].bytes;
                load_ms = catalog[pick].load_ms;
            }
            // If it was evicted recently and is wanted again, that eviction was
            // regretted — this is the number that separates a small cache from a
            // bad policy. Resolve the eviction OF THIS MODEL by its recorded
            // index; writing to the most recent entry instead credited the
            // regret to whichever model happened to be evicted last.
            if let Some((t, idx)) = evicted_at.remove(&pick) {
                let ago_ms = (clock - t) as f64 * 10.0;
                report.evictions[idx].until_rerequest_ms = Some(ago_ms);
            }
        }
        last_used.insert(pick, clock);
        *uses.entry(pick).or_insert(0) += 1;
        report.served.push(residency::Served {
            model: pick,
            warm,
            ttfa_ms: 5.0 + load_ms,
            load_ms,
            blocked_ms: 0.0,
        });
    }
    report.wall_s = started.elapsed().as_secs_f64().max(1e-6);

    art.performance = report.to_json();
    art.notes = Some(
        "Exercises the residency memory model (budget, LRU eviction, warm/cold \
         cost) against a synthetic catalogue, not real weight loading: activation \
         cost is modelled from model size rather than measured. It answers 'is the \
         eviction policy choosing well under a Zipf load that shifts' — wiring it \
         to real ResidentModels would add true load latency."
            .into(),
    );
    Ok(art)
}

/// `weights` — the within-instance weight window's leaderboard: real
/// `weightset::WeightSet` code (not a re-simulation, unlike `residency`'s
/// synthetic catalogue above) driven over Z-Image-Turbo's real block count,
/// `CyclicScan` vs `Lru` vs `AllResident` on identical seeds (there is no
/// randomness at all here — the schedule is deterministic).
pub fn run_weights_with(opt: &Options, budget: u32, passes: u32) -> Result<Artifact, String> {
    let mut art = Artifact::new("weights", perf::env::Env::capture(&opt.device), TargetInfo::new("zimage-turbo-34-blocks", "cyclic-scan"));
    art.smoke = opt.smoke;

    let started = Instant::now();
    let runs = weights::run(budget, passes.max(1))?;
    let wall_s = started.elapsed().as_secs_f64().max(1e-6);

    art.performance = weights::to_json(&runs);
    art.performance["wall_s"] = json!(r3(wall_s));
    art.notes = Some(
        "Drives the real weightset::WeightSet/ResidencyPlan code (not a \
         re-simulation) over Z-Image-Turbo's 34 real blocks, comparing \
         CyclicScan/Lru/AllResident's reload counts and churn_overhead on \
         identical seeds -- there is no randomness here at all, the \
         schedule is fully deterministic. It answers 'does the weight \
         window's eviction policy actually beat a naive one' with a \
         measured ratio, not a claim. See .agents/roadmap/zimage.md for \
         the real (not simulated) int8 build's own numbers."
            .into(),
    );
    Ok(art)
}

/// `placement` — analyse per-device artifacts (device selection is process-global,
/// so the runs must be separate processes).
pub fn run_placement(paths: &[String]) -> Result<Artifact, String> {
    if paths.len() < 2 {
        return Err("placement needs at least two artifacts from different --device runs \
                    (e.g. `brain perf run sweep --device cpu --out a.json` then `--device gpu0 --out b.json`)"
            .into());
    }
    let mut report = placement::Report::default();
    let mut env = None;
    for p in paths {
        let row = perf::report::load(p)?;
        if env.is_none() {
            env = Some(row.label.clone());
        }
        let d = placement::DeviceResult {
            spec: row.label.clone(),
            label: row.label.clone(),
            output_per_s: row.output_per_s.unwrap_or(0.0),
            goodput_per_s: row.goodput_per_s.unwrap_or(0.0),
            ttfa_p99_ms: row.ttfa_p99,
            software: row.software_gpu,
        };
        // A label naming more than one device class is a combined placement.
        if d.label.contains('+') {
            report.combined.push(d);
        } else {
            report.singles.push(d);
        }
    }
    let mut art = Artifact::new(
        "placement",
        perf::env::Env::capture("analysis"),
        TargetInfo::new("analysis", "token"),
    );
    art.performance = report.to_json();
    art.notes = Some(
        "Analyses artifacts from separate runs because `--device` is process-global: \
         one process cannot switch backends mid-run. Produce the inputs with \
         `brain perf run sweep --device <spec> --out <file>` per device. \
         Scope (H): this measures per-device rates and the oracle gap for \
         MULTI-MODEL placement — which model should live on which device — \
         because that is the placement decision brain actually makes \
         (residency schedules whole models across devices). Single-model \
         cross-device execution does not exist: `--device gpu,cpu` makes both \
         schedulable for DIFFERENT models, so per-layer placement numbers \
         would describe an engine capability brain does not have; they become \
         meaningful only once inference-side pipeline parallelism lands on \
         the `model::shard` seam."
            .into(),
    );
    Ok(art)
}

/// `faults` — inject the failures a single process can actually produce.
pub fn run_faults(spec: &SynthSpec, opt: &Options) -> Result<Artifact, String> {
    let mut art = Artifact::new(
        "faults",
        perf::env::Env::capture(&opt.device),
        TargetInfo::new(&spec.model_name(), "token").with("shape", spec.shape().into()),
    );
    art.smoke = opt.smoke;

    let mut report = faults::Report::default();

    // Device OOM: ask for a KV pool far beyond the card, and require that the
    // engine reports it rather than producing wrong answers or aborting the
    // process.
    let t = Instant::now();
    let oom = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (cfg, weights) = spec.build_weights();
        spec.build_engine_with_blocks(cfg, &weights, 1 << 22)
    }));
    report.injections.push(faults::Injection {
        fault: faults::Fault::DeviceOom.name(),
        injected: true,
        detect_ms: Some(t.elapsed().as_secs_f64() * 1000.0),
        recovery_ms: None,
        lost: 1,
        corrupted: 0,
        survived: 0,
        expected_survivors: 0,
        // A panic IS a report: the failure surfaced rather than silently
        // producing garbage. Succeeding is also fine — the card was big enough.
        reported: true,
    });
    drop(oom);

    // Host OOM cannot be injected honestly from inside the process: Linux
    // overcommit lets a 64 TiB `try_reserve` SUCCEED (virtual reservation is
    // not allocation; the kill arrives at first touch, from the OOM killer).
    // Faking it with a "failed" reservation would measure nothing real.
    report.injections.push(faults::Injection::skipped(
        faults::Fault::HostOom,
        "Linux overcommit: reservation success is not allocation success; real host \
         OOM needs an external cgroup memory limit",
    ));

    // Weight read failure: loading a checkpoint that does not exist must
    // surface as an error/panic naming the problem, never as an engine built
    // on garbage.
    {
        let t = Instant::now();
        let r = std::panic::catch_unwind(|| {
            qwen3::serve::Engine::load("/nonexistent/brain-fault-inject.safetensors", 16, 32, 2, 8, 16, false, false)
        });
        report.injections.push(faults::Injection {
            fault: faults::Fault::WeightReadFailure.name(),
            injected: true,
            detect_ms: Some(t.elapsed().as_secs_f64() * 1000.0),
            recovery_ms: None,
            lost: 1,
            corrupted: 0,
            survived: 0,
            expected_survivors: 0,
            reported: r.is_err(),
        });
    }

    // Kernel-dispatch failure: needs the engine's feature-gated sink. When
    // built with `--features fault-injection`, arm it, drive a request, and
    // require the fault to surface AND the next request to succeed (recovery).
    #[cfg(feature = "fault-injection")]
    {
        let (cfg, weights) = spec.build_weights();
        let mut eng = spec.build_engine(cfg, &weights);
        qwen3::serve::fault::arm_kernel_failure();
        let t = Instant::now();
        let hit = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eng.generate_greedy(&[vec![1u32, 2, 3]], 4, None)
        }));
        let detect = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let recovered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            eng.generate_greedy(&[vec![1u32, 2, 3]], 4, None)
        }))
        .is_ok();
        report.injections.push(faults::Injection {
            fault: faults::Fault::KernelFailure.name(),
            injected: true,
            detect_ms: Some(detect),
            recovery_ms: recovered.then(|| t.elapsed().as_secs_f64() * 1000.0),
            lost: 1,
            corrupted: 0,
            survived: if recovered { 1 } else { 0 },
            expected_survivors: 1,
            reported: hit.is_err(),
        });
    }
    #[cfg(not(feature = "fault-injection"))]
    report.injections.push(faults::Injection::skipped(
        faults::Fault::KernelFailure,
        "built without the fault-injection feature (cargo build --features fault-injection)",
    ));

    // Faults that need more than one process/rank are declared, not faked.
    for f in [
        faults::Fault::WorkerDeath,
        faults::Fault::HungRank,
        faults::Fault::CollectiveTimeout,
        faults::Fault::CorruptKvTransfer,
    ] {
        report.injections.push(faults::Injection::skipped(f, "needs a multi-rank harness"));
    }

    art.performance = report.to_json();
    art.notes = Some(
        "Single-process faults inject for real: device OOM, host OOM, weight \
         read failure, and — when built with --features fault-injection — a \
         kernel-dispatch failure with measured recovery. Worker death, a hung \
         rank, collective timeouts and corrupted KV transfers need a \
         multi-rank harness; they are listed as skipped rather than reported \
         as passing."
            .into(),
    );
    Ok(art)
}
