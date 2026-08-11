// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain run` / `brain serve` — the event-driven stdio controller loop.
//!
//! Reads JSONL [`events::Event`] lines from stdin (a blocking read is the idle
//! wait), feeds each to a [`runtime::Controller`], and writes every emitted event
//! back as a JSONL line to stdout (flushed per line). Diagnostics go to stderr.
//!
//! Flags:
//!   * `--gpt <path>` (or env `BRAIN_GPT`) — load a GPT checkpoint as the text
//!     model. With none, a fake echo model is used so the loop is testable
//!     without a trained model.
//!   * `--yolo <path>` (or env `BRAIN_YOLO`) — load a YOLO checkpoint as the
//!     object detector. With none, a `FakeDetectModel` returns a fixed box so the
//!     loop runs without a trained detector.
//!   * `--conf <f32>` (or env `BRAIN_CONF`) — detection confidence threshold for
//!     the YOLO detector (default 0.25). Lower it so a lightly-trained tiny model's
//!     low-confidence boxes still surface. No effect on the fake detector.
//!   * `--max-new N`, `--temp X`, `--top-k K`, `--seed S` — generation config.
//!   * `--models-dir <path>` (or env `BRAIN_MODELS_DIR`) — the global model
//!     directory `brain serve --dbus` scans at startup to build the served-model
//!     catalog (one entry per carded file, keyed by model-card id). Defaults to
//!     `$XDG_DATA_HOME/brain/models` else `$HOME/.local/share/brain/models`.

use std::io::{BufRead, Write};
use std::sync::Arc;

use events::Envelope;
use memauth::PoolProbe;
use runtime::{
    Controller, DetectModel, Emit, FakeDetectModel, FakeInferModel, GenConfig, GptInfer, Registry,
    YoloDetect,
};

/// A live [`Emit`] sink over a stdout writer: encodes each envelope to a JSONL
/// line and flushes it immediately, so `brain run` streams token-by-token as the
/// controller produces them (not one batch at the end of the turn). `ok` latches
/// false once the pipe closes so the loop can stop.
struct StdoutSink<'a, W: Write> {
    w: &'a mut W,
    ok: bool,
}

impl<W: Write> Emit for StdoutSink<'_, W> {
    fn emit(&mut self, env: Envelope) {
        if !self.ok {
            return;
        }
        if writeln!(self.w, "{}", events::encode_envelope(&env)).is_err() {
            self.ok = false;
            return;
        }
        let _ = self.w.flush();
    }
}

const HELP: &str = "\
brain serve — serve brain's models over HTTP, D-Bus, or stdio  (alias: brain run)

USAGE:
  brain serve [SURFACE ...] [options]     # one or more serving surfaces
  brain serve                             # NO surface flag: the stdio JSONL loop

HTTP INFERENCE APIS  (each on its own localhost port, each behind its own key)
  --openai [PORT]        OpenAI-compatible dialect       (default port 8788)
  --openrouter [PORT]    OpenRouter-compatible dialect   (default port 8789)
  --anthropic [PORT]     Anthropic Messages dialect      (default port 8787)

  BASE URL — for OpenAI and OpenRouter, BOTH of these work, because every route
  is registered with and without the /v1 prefix:
      http://127.0.0.1:PORT        POST /chat/completions,    GET /models
      http://127.0.0.1:PORT/v1     POST /v1/chat/completions, GET /v1/models
  Point OPENAI_BASE_URL / a client's base_url at either one. The Anthropic
  dialect is /v1-ONLY: base URL http://127.0.0.1:PORT, routes POST /v1/messages
  and POST /v1/messages/count_tokens. GET /models and GET /v1/models are served
  on every surface. Also available on openai/openrouter: /embeddings and
  /images/generations, each with and without /v1.

  Auth:  Authorization: Bearer <key>   (openai, openrouter)
         x-api-key: <key>              (anthropic)
  A fresh key per surface per launch, printed on stdout as
  `APIKEY <provider> <key>`; --api-keys-out writes the same keys as JSON, 0600.

  Surfaces ALWAYS bind 127.0.0.1. There is no --listen / --host / --bind flag.

D-BUS CONTROL SURFACE
  --dbus                 serve com.swedishembedded.Brain1 on the session bus
  --dbus-system          use the system bus instead. Needs a system.d policy:
                         the deb installs a vetted one (calls restricted to
                         root + the 'brain' group); from a checkout, install
                         scripts/build/com.swedishembedded.Brain1.conf to
                         /usr/share/dbus-1/system.d/ yourself.
  --dbus-name NAME       request NAME instead of com.swedishembedded.Brain1

SERVING OPTIONS
  --models-dir DIR       directory scanned at startup for the served catalog
                         (else $BRAIN_MODELS_DIR, else $XDG_DATA_HOME/brain/models)
  --api-keys-out FILE    write {\"openai\":\"sk-brain-…\", …} as JSON, mode 0600
  --reserve-gb N         GB of VRAM kept free per GPU for activations (default 2)
  --ready-file PATH      create PATH (empty) once EVERY surface requested above
                         has bound its listener. Because the APIKEY lines and
                         --api-keys-out are both written BEFORE any bind, PATH
                         appearing means: keys are on disk AND every listener is
                         accepting. Wait on this one file instead of polling a
                         port or grepping the log.
                         It is NEVER created if any requested surface fails to
                         come up -- so a waiter must also bound its wait and
                         check the process is still alive.
                         \"Ready\" means listening, not warm: models load lazily.
                         A stale PATH from a previous run is removed at startup.
                         The file is empty and not a secret: it holds no key, no
                         pid and no address.

STDIO CONTROLLER  (the default, with no surface flag)
  --gpt PATH             GPT checkpoint (else $BRAIN_GPT; else a fake echo model)
  --yolo PATH             YOLO checkpoint (else $BRAIN_YOLO; else a fake detector)
  --conf X                detection confidence threshold (else $BRAIN_CONF, 0.25)
  --max-new N  --temp X  --top-k K  --seed S      generation config
  Reads JSONL events on stdin, writes JSONL events on stdout, one per line.
  Example: printf '{\"event\":\"user_text\",\"text\":\"hi\"}\\n' | brain run

MODEL CONFIGURATION (env-only — there is no config file)
  Which models this server actually serves is chosen ENTIRELY by BRAIN_* env
  vars (BRAIN_QWEN_WEIGHTS, BRAIN_LFM, BRAIN_NEMOTRON, ...): a model whose
  weights var is unset is simply not served. The full reference table for
  every serving variable is docs/using/configuration.md, section 'Configuration'.

GLOBAL
  --device cpu|gpu|npu|gpu0|cpu0-7|gpu,cpu   consumed before this subcommand
                                             (see brain --help); $BRAIN_DEVICE
  -v, --verbose [0-3]    diagnostic detail on stderr (else $BRAIN_VERBOSE):
                           0  errors only (default) -- unchanged from today
                           1  + warnings (e.g. a model family not configured)
                           2  + info (model registered/activating/resident,
                              evicted/demoted/promoted -- the residency
                              lifecycle: what's loaded, right now, and why)
                           3  + debug (finer scheduling detail)
                         Repeatable short form bumps by one level each time
                         (-v -v = 2, as separate args -- not bundled -vv);
                         bare --verbose (no number) also means 1.
                         Never gates the protocol output every surface always
                         prints regardless of this flag (the compute/model
                         summary, APIKEY lines, --ready-file).
  -h, --help                                 this text

EXAMPLES
  brain serve --openai                       # OpenAI API on http://127.0.0.1:8788
  brain serve --openai 9000 --api-keys-out /run/brain/keys.json \\
              --ready-file /run/brain/ready
  brain serve --dbus --anthropic --openrouter
";

/// The value that must follow a value-taking flag. A flag with no value is a
/// typo, not a request for the default: a trailing `--gpt` used to silently
/// mean \"no checkpoint\", erasing a `BRAIN_GPT` already read from the
/// environment, and a trailing `--models-dir` used to silently scan the XDG
/// default instead of the directory the caller meant to name.
fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    match args.get(*i) {
        Some(v) => v.clone(),
        None => {
            eprintln!("brain serve: {flag} needs a value\n");
            eprint!("{HELP}");
            std::process::exit(2);
        }
    }
}

/// The parsed value that must follow a value-taking numeric flag. Same policy
/// as [`val`], extended to the parse: an UNPARSEABLE value is a typo, not a
/// request for the default — `--reserve-gb 2G` used to silently serve with
/// the default 2 (a coincidence), and `--temp 0,8` silently sampled at the
/// default temperature. Exit 2 names the flag and the rejected value.
fn parsed<T: std::str::FromStr>(args: &[String], i: &mut usize, flag: &str) -> T {
    let v = val(args, i, flag);
    v.parse().unwrap_or_else(|_| {
        eprintln!("brain serve: {flag} {v:?} is not a valid value\n");
        eprint!("{HELP}");
        std::process::exit(2);
    })
}

pub fn run_serve(args: &[String]) {
    let mut gpt_path = std::env::var("BRAIN_GPT").ok();
    let mut yolo_path = std::env::var("BRAIN_YOLO").ok();
    let mut cfg = GenConfig { max_new: 256, temperature: 0.0, top_k: 0, eos: None, seed: 0 };
    // Optional detection confidence threshold for the YOLO detector. A tiny model
    // trained for only a few hundred steps emits low-confidence boxes that the
    // default 0.25 filter would drop, so the demo can lower it (also `BRAIN_CONF`).
    let mut conf: Option<f32> =
        std::env::var("BRAIN_CONF").ok().and_then(|s| s.parse().ok());
    // D-Bus control surface (`--dbus [--dbus-system] [--dbus-name NAME]`).
    let (mut dbus, mut dbus_system, mut dbus_name) = (false, false, None::<String>);
    let mut dbus_reserve_gb: u64 = 2; // GB kept free per GPU (headroom for activations)
    // Global model directory scanned at startup for the served-model catalog
    // (`--models-dir`, else BRAIN_MODELS_DIR / XDG default; see model_dir::resolve).
    let mut models_dir: Option<String> = None;
    // HTTP inference APIs (`--anthropic|--openai|--openrouter [PORT]`), each on its own
    // localhost port with a per-provider key generated at startup. All share the one
    // executor (with D-Bus, if also selected). `--api-keys-out FILE` writes the keys as
    // JSON for scripted clients / the e2e test.
    let (mut anthropic, mut openai, mut openrouter): (Option<u16>, Option<u16>, Option<u16>) =
        (None, None, None);
    let mut api_keys_out: Option<String> = None;
    let mut ready_file: Option<String> = None;
    // Diagnostic verbosity (`-v`/`--verbose [0-3]`, else $BRAIN_VERBOSE) -- see
    // HELP above for what each tier gates. Clamped by `residency::log::set_verbosity`,
    // not here, so an out-of-range env var doesn't need its own validation.
    let mut verbose: u8 = std::env::var("BRAIN_VERBOSE").ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    // Optional PORT immediately following an API flag (else the provider default).
    let take_port = |args: &[String], i: &mut usize, default: u16| -> u16 {
        if let Some(p) = args.get(*i + 1).and_then(|s| s.parse::<u16>().ok()) {
            *i += 1;
            p
        } else {
            default
        }
    };

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gpt" => gpt_path = Some(val(args, &mut i, "--gpt")),
            "--yolo" => yolo_path = Some(val(args, &mut i, "--yolo")),
            "--max-new" => cfg.max_new = parsed(args, &mut i, "--max-new"),
            "--temp" | "--temperature" => {
                let flag = args[i].clone();
                cfg.temperature = parsed(args, &mut i, &flag);
            }
            "--top-k" => cfg.top_k = parsed(args, &mut i, "--top-k"),
            "--seed" => cfg.seed = parsed(args, &mut i, "--seed"),
            "--conf" => conf = Some(parsed(args, &mut i, "--conf")),
            "--dbus" => dbus = true,
            "--dbus-system" => {
                dbus = true;
                dbus_system = true;
            }
            "--dbus-name" => dbus_name = Some(val(args, &mut i, "--dbus-name")),
            "--reserve-gb" => dbus_reserve_gb = parsed(args, &mut i, "--reserve-gb"),
            "--models-dir" => models_dir = Some(val(args, &mut i, "--models-dir")),
            "--anthropic" => anthropic = Some(take_port(args, &mut i, 8787)),
            "--openai" => openai = Some(take_port(args, &mut i, 8788)),
            "--openrouter" => openrouter = Some(take_port(args, &mut i, 8789)),
            "--api-keys-out" => api_keys_out = Some(val(args, &mut i, "--api-keys-out")),
            "--ready-file" => ready_file = Some(val(args, &mut i, "--ready-file")),
            "-v" => verbose = verbose.saturating_add(1),
            "--verbose" => {
                verbose = match args.get(i + 1).and_then(|s| s.parse::<u8>().ok()) {
                    Some(n) => {
                        i += 1;
                        n
                    }
                    None => 1, // bare --verbose (no numeric arg) means level 1
                };
            }
            "--help" | "-h" => {
                print!("{HELP}");
                return;
            }
            other => {
                // stderr, not stdout: brain serve's stdout is a protocol stream
                // (JSONL envelopes; `APIKEY <provider> <key>` lines a harness
                // greps), so dumping usage onto it on the ERROR path would
                // corrupt that stream. `--help` above -- a successful request
                // for the text -- still goes to stdout.
                eprintln!("brain serve: unknown flag {other:?}\n");
                eprint!("{HELP}");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    // Set BEFORE anything that could log (the executor's own construction
    // already registers models) -- both the stdio loop and run_apis below
    // read this same process-global level.
    residency::log::set_verbosity(verbose);

    let surfaces_requested =
        dbus as usize + anthropic.is_some() as usize + openai.is_some() as usize + openrouter.is_some() as usize;
    // The stdio loop counts as one "surface" too, so --ready-file means the same
    // thing in both modes: it fires at the same point the loop already emits
    // `events::Event::Ready`.
    let ready = match &ready_file {
        Some(p) => match brain_shutdown::ready::Gate::touching(p, surfaces_requested.max(1)) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("brain serve: --ready-file {p}: {e}");
                std::process::exit(2);
            }
        },
        None => brain_shutdown::ready::Gate::disabled(),
    };

    // The D-Bus control surface replaces the stdio loop when requested: it serves
    // every registered model over `com.swedishembedded.Brain1` until Ctrl-C.
    if dbus || anthropic.is_some() || openai.is_some() || openrouter.is_some() {
        return run_apis(RunApis {
            dbus,
            dbus_system,
            dbus_name,
            reserve_gb: dbus_reserve_gb,
            models_dir,
            anthropic,
            openai,
            openrouter,
            api_keys_out,
            ready,
        });
    }

    // Build the registry: a real GPT if a checkpoint was given, else a fake echo
    // model so the loop runs end-to-end without a trained model.
    let infer: Box<dyn runtime::InferModel> = match &gpt_path {
        Some(path) => {
            eprintln!("brain run: loading GPT checkpoint {path}");
            // Char models embed itos; the pump uses it for the EOS-less stop at
            // max_new. We leave eos as configured (None unless the user sets one).
            Box::new(GptInfer::load(path))
        }
        None => {
            eprintln!("brain run: no --gpt checkpoint; using fake echo model");
            // The fake echoes a fixed greeting and terminates at its EOS sentinel.
            cfg.eos = Some(256);
            Box::new(FakeInferModel::echoing("hello from brain"))
        }
    };
    // A real YOLO if a checkpoint was given, else the fixed-box fake detector.
    let detect: Box<dyn DetectModel> = match &yolo_path {
        Some(path) => {
            eprintln!("brain run: loading YOLO checkpoint {path}");
            let mut det = YoloDetect::load(path);
            if let Some(c) = conf {
                eprintln!("brain run: detection confidence threshold {c}");
                // Keep the default IoU (0.45); only override the confidence gate.
                det = det.with_thresholds(c, 0.45);
            }
            Box::new(det)
        }
        None => {
            eprintln!("brain run: no --yolo checkpoint; using fake detector");
            Box::new(FakeDetectModel::default())
        }
    };

    let mut ctrl = Controller::with_config(Registry::with_models(infer, detect), cfg);

    // Expose the generic capability providers over the event API (manifest_request
    // / action_request) — the same actions `brain do` runs, now network-reachable.
    ctrl.register_provider(std::sync::Arc::new(zimage::caps::ZImageProvider::load().expect("z-image provider")));
    ctrl.register_provider(std::sync::Arc::new(lfm::caps::LfmProvider::new()));

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    // Announce readiness.
    let _ = writeln!(out, "{}", events::encode_line(&events::Event::Ready));
    let _ = out.flush();
    ready.bound("stdio");

    // Blocking line read = idle wait. EOF (None) ends the loop.
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("brain run: stdin error: {e}");
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        // Stream each emitted envelope to stdout as it is produced (flushed per
        // line), so a long chat response appears token-by-token rather than all at
        // once when the turn completes. The req_id (if any) is echoed on every line
        // for client-side demuxing. No control source on stdin's blocking read: a
        // `cancel` is honored as the next line (recoverable), between turns.
        let mut sink = StdoutSink { w: &mut out, ok: true };
        ctrl.feed_line_streaming(&line, &mut sink, &mut ());
        if !sink.ok {
            return; // stdout closed
        }
    }
}

/// Discover schedulable compute (GPUs/NPUs/CPU RAM, narrowed by `--device`), resolve
/// the model directory, and build the one shared residency executor that every serving
/// surface (D-Bus + the HTTP APIs) drives.
fn build_serving_executor(reserve_gb: u64, models_dir: Option<String>) -> residency::Executor {
    // Discover the GPUs' capacity so the scheduler can budget/evict against real VRAM,
    // then narrow to what `--device` made schedulable. With no `--device` the set is
    // every device, which is exactly the "use all the hardware wisely" default.
    let mut all_gpus = query_gpu_mem();
    // No NVIDIA GPU, but the wgpu backend can drive an integrated GPU (e.g. Intel
    // Arc on Meteor Lake): budget it as a schedulable `Gpu` lane. Integrated GPUs
    // have no dedicated VRAM — they share system RAM — so size the budget like the
    // NPU (a modest fraction of RAM). This is what makes `--device gpu` (and the
    // all-devices default) actually schedule onto the iGPU on such boxes.
    // Devices this fallback creates ALWAYS share physical RAM with the CPU
    // (that is the case it exists for) — tracked so they can be declared into
    // the same memauth pool as Device::Cpu below, instead of budgeted as an
    // independent-but-physically-identical pool of bytes.
    let mut fallback_unified_gpus: Vec<u32> = Vec::new();
    if all_gpus.is_empty() {
        // Not `discrete_gpu_count` (that's 0 by definition on an integrated-only
        // box): `visible_gpu_count` counts the iGPU too, which is exactly the
        // case this fallback exists for.
        let n = gpu_core::visible_gpu_count();
        if n > 0 {
            // The real ceiling is the shared RAM pool declared below, not a
            // fraction reserved here — this device budget only needs to be AT
            // LEAST the pool's total so the pool (not a smaller guessed
            // fraction) is always the binding constraint.
            let ram = host_ram_available();
            all_gpus = (0..n as u32).map(|i| (i, ram)).collect();
            fallback_unified_gpus = (0..n as u32).collect();
            eprintln!("brain serve: no NVIDIA GPU; budgeting {n} integrated GPU(s), sharing the {} GB RAM pool (schedulable)", ram >> 30);
        }
    }
    let set = crate::compute_set();
    let gpus: Vec<(u32, u64)> = match set {
        Some(s) => all_gpus.iter().copied().filter(|(i, _)| s.gpus.contains(i)).collect(),
        None => all_gpus.clone(),
    };
    let cpu_schedulable = set.map(|s| s.cpu_enabled()).unwrap_or(true);
    // Devices whose bytes physically ARE the CPU's RAM: this fallback's
    // synthesized GPUs, plus any real GPU the device registry classifies as
    // integrated (an Arc/Xe iGPU reporting real VRAM via query_gpu_mem — the
    // common case on this box — never goes through the fallback above, so it
    // needs its own check here). A discrete GPU with dedicated VRAM is not
    // included. See `memauth`'s module doc for why declaring this matters:
    // without it, a GPU-side allocation and a CPU-side one are budgeted as
    // if they came from two separate pools of memory, when they are the same
    // physical bytes.
    let unified_gpus: Vec<u32> = gpus
        .iter()
        .map(|&(i, _)| i)
        .filter(|i| {
            fallback_unified_gpus.contains(i)
                || gpu_core::devices::gpus().iter().any(|d| d.index == *i && d.identity.class == backend_api::DeviceClass::IntegratedGpu)
        })
        .collect();

    if gpus.is_empty() && !all_gpus.is_empty() {
        eprintln!("brain serve: --device excluded every GPU; scheduling on CPU only");
    } else if all_gpus.is_empty() {
        eprintln!("brain serve: no GPUs detected (nvidia-smi); serving with CPU-only budget");
    }
    let ram = host_ram_available();
    let reserved = reserve_gb << 30;
    // Host RAM stays a cache/spill tier even when the CPU is not schedulable for
    // compute — `--device gpu` bounds where work runs, not where bytes may rest.
    let cpu_compute_ram = if cpu_schedulable { ram } else { 0 };
    // Schedulable NPUs: `--device` narrows to `set.npus`; with no `--device`, any NPU
    // present is scheduled. The Meteor-Lake-class NPU shares system RAM, so it gets a
    // modest per-device budget. A model with an NPU path (MemCost.npu > 0) is then
    // auto-placed on the NPU in preference to CPU/GPU (see place::pick_device).
    let npu_indices: Vec<u32> = match set {
        Some(s) => s.npus.clone(),
        None if npu::openvino::npu_present() => vec![0],
        None => vec![],
    };
    // NPUs always share system RAM (see the comment above); their device
    // budget only needs to be at least the pool's total, same reasoning as
    // the iGPU fallback — `resident::build_executor` declares them into the
    // shared pool alongside `unified_gpus` and Device::Cpu.
    let npus: Vec<(u32, u64)> = npu_indices.iter().map(|&i| (i, ram)).collect();
    eprintln!(
        "brain serve: compute {} | {} GPU(s), {} NPU(s) schedulable, {} GB reserved/card, {} GB RAM budget",
        set.map(|s| s.to_string()).unwrap_or_else(|| "all".into()),
        gpus.len(),
        npus.len(),
        reserve_gb,
        ram >> 30
    );
    // Resolve the global model directory (flag > BRAIN_MODELS_DIR > XDG default);
    // its scan appends every carded file as its own catalog entry.
    let dir = crate::model_dir::resolve(models_dir.as_deref());
    if let Some(d) = &dir {
        // First run on a fresh install: the dir doesn't exist yet. Create it so
        // the scan is clean (an empty catalog, not an ENOENT warning) — models
        // dropped in later are picked up on the next `brain serve` with no env
        // vars needed.
        if let Err(e) = std::fs::create_dir_all(d) {
            eprintln!("brain serve: could not create model dir {} ({e}); scan may be empty", d.display());
        }
        eprintln!("brain serve: scanning model dir {}", d.display());
    }
    let executor = crate::resident::build_executor(&gpus, &npus, &unified_gpus, reserved, cpu_compute_ram, ram, dir.as_deref(), residency::Policy::from_env());
    executor
}

/// Live host RAM this process could actually get right now: `MemAvailable`
/// intersected with any cgroup v2 limit — see `memauth::HostProbe`, whose
/// doc carries the `MemAvailable`-over-`MemTotal` rationale this used to
/// duplicate locally. `query_ram_bytes` (the old name, kept public within
/// the crate since `perf_cli.rs` calls it by that name) is now a thin alias.
pub(crate) fn host_ram_available() -> u64 {
    memauth::HostProbe::new(memauth::HOST_POOL).available(memauth::HOST_POOL)
}

/// Which serving surfaces to bring up and their config (see `run_apis`).
struct RunApis {
    dbus: bool,
    dbus_system: bool,
    dbus_name: Option<String>,
    reserve_gb: u64,
    models_dir: Option<String>,
    anthropic: Option<u16>,
    openai: Option<u16>,
    openrouter: Option<u16>,
    api_keys_out: Option<String>,
    /// Notified once per bound surface (HTTP + D-Bus); disabled unless
    /// `--ready-file` was given. See `brain_shutdown::ready::Gate`.
    ready: brain_shutdown::ready::Gate,
}

/// Build the one shared executor and bring up the requested surfaces: D-Bus
/// (`com.swedishembedded.Brain1`) and/or the HTTP inference APIs (Anthropic / OpenAI /
/// OpenRouter), each on its own localhost port with a key generated at startup. When
/// D-Bus and an HTTP surface both run, D-Bus gets its own thread (it owns a tokio
/// runtime) and the HTTP servers own the main thread; a single surface blocks directly.
/// A `brain_dbus::serve` failure is almost always "no bus at this address" (no
/// desktop session, no `dbus-run-session`, no system bus policy for this user) —
/// give a message that says what to try instead of the raw connect errno.
/// `http_up` reports whether an HTTP API surface is still serving, so the
/// operator knows this failure did not take the whole process down.
fn dbus_connect_hint(err: &dyn std::fmt::Display, system_bus: bool, http_up: bool) -> String {
    let status = if http_up {
        "HTTP API surface(s) remain up."
    } else {
        "no other surface was requested; exiting."
    };
    let advice = if system_bus {
        "Check the system bus is running and this user has a policy file for \
         the requested bus name, or drop --dbus-system for the per-user session bus."
    } else {
        "Run under `dbus-run-session -- <cmd>`, start a desktop session, or pass \
         --dbus-system if a system bus policy is installed for this service."
    };
    let kind = if system_bus { "system" } else { "session" };
    format!("brain serve --dbus: could not connect to the D-Bus {kind} bus ({err}). {advice} {status}")
}

/// How long [`run_apis`] waits for a backgrounded D-Bus surface to finish its own
/// graceful shutdown after the HTTP surface has drained, before giving up and
/// letting the process exit anyway. Bounded so a wedged D-Bus shutdown cannot
/// hang `brain serve` forever — [`brain_shutdown::install_signals`]'s own
/// second-signal escape hatch is the backstop if this window is not enough.
const DBUS_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Build the transparent auto-fetch supplier for a live `brain serve`, unless
/// disabled via `BRAIN_AUTO_FETCH=0` (also accepts `false`/`off`,
/// case-insensitive; anything else -- including unset -- is enabled, since
/// transparent auto-fetch is the point of `brain serve`; see
/// `docs/using/models-and-weights.md` and `.agents/rules/api-security.md`). `None` when
/// disabled OR when no models directory can be resolved at all (no `$HOME`) --
/// either way, an unresolved model then just 404s/errors with zero I/O, same as
/// before this existed.
///
/// Every HTTP/D-Bus surface in this process shares ONE supplier instance (not
/// one per surface): `StoreSupplier`'s in-flight map is what makes concurrent
/// requests for the same cold model share a single fetch rather than each
/// surface racing its own download.
fn build_auto_fetch_supplier(models_dir: Option<&str>) -> Option<Arc<dyn residency::ModelSupplier>> {
    let disabled = std::env::var("BRAIN_AUTO_FETCH")
        .ok()
        .is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off"));
    if disabled {
        eprintln!("brain serve: BRAIN_AUTO_FETCH disabled -- an unresolved model 404s/errors instead of auto-fetching");
        return None;
    }
    // The SAME models directory `build_serving_executor`'s startup scan
    // resolved, so a freshly auto-fetched model lands exactly where a restart's
    // scan would find it again.
    let dir = crate::model_dir::resolve(models_dir)?;
    let store = brain_modelstore::Store::new(dir);
    let hub: Box<dyn brain_modelstore::Hub> = Box::new(brain_modelstore::HfHub::new());
    Some(Arc::new(crate::supply::StoreSupplier::new(store, hub)))
}

fn run_apis(a: RunApis) {
    let supplier = build_auto_fetch_supplier(a.models_dir.as_deref());
    let executor = build_serving_executor(a.reserve_gb, a.models_dir);
    let manifests = executor.manifests();
    let served: Vec<&str> = manifests.iter().map(|m| m.model.as_str()).collect();
    eprintln!("brain serve: models: {}", served.join(", "));

    let http = a.anthropic.is_some() || a.openai.is_some() || a.openrouter.is_some();

    // One shutdown source for every surface this process serves. SIGINT/SIGTERM
    // disposition is process-wide: if D-Bus and HTTP each installed their own
    // `tokio::signal::ctrl_c()` handler, only one registration would ever
    // actually see the signal — see `brain_shutdown` for the failure this
    // caused. Installed once, here, before either surface's runtime exists, so
    // Ctrl-C/SIGTERM reaches whichever surfaces are actually running.
    let (trigger, shutdown) = brain_shutdown::channel();
    brain_shutdown::install_signals(trigger);

    let dbus_handle = if a.dbus {
        let opts = brain_dbus::DbusOpts {
            bus: if a.dbus_system { brain_dbus::BusKind::System } else { brain_dbus::BusKind::Session },
            name: a.dbus_name.unwrap_or_else(|| "com.swedishembedded.Brain1".to_string()),
        };
        let e = executor.clone();
        let sup = supplier.clone();
        if http {
            let sd = shutdown.clone();
            let dbus_system = a.dbus_system;
            let ready = a.ready.clone();
            let ready_for_diag = a.ready.clone();
            Some(std::thread::spawn(move || {
                let serve_opts = brain_dbus::ServeOpts::new().with_shutdown(sd).with_supplier(sup).with_ready(ready);
                if let Err(err) = brain_dbus::serve(e, opts, serve_opts) {
                    eprintln!("{}", dbus_connect_hint(&err, dbus_system, true));
                    if let Some(p) = ready_for_diag.path() {
                        eprintln!("brain serve: --ready-file {} will NEVER be created -- the D-Bus surface was requested but did not start", p.display());
                    }
                }
            }))
        } else {
            let serve_opts = brain_dbus::ServeOpts::new().with_shutdown(shutdown).with_supplier(sup).with_ready(a.ready.clone());
            if let Err(err) = brain_dbus::serve(e, opts, serve_opts) {
                eprintln!("{}", dbus_connect_hint(&err, a.dbus_system, false));
                std::process::exit(1);
            }
            return;
        }
    } else {
        None
    };

    if http {
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};
        let local = |port: u16| SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
        let mut surfaces = Vec::new();
        if let Some(p) = a.anthropic {
            surfaces.push(apiserve::Surface::generate(apiserve::Provider::Anthropic, local(p)));
        }
        if let Some(p) = a.openai {
            surfaces.push(apiserve::Surface::generate(apiserve::Provider::OpenAI, local(p)));
        }
        if let Some(p) = a.openrouter {
            surfaces.push(apiserve::Surface::generate(apiserve::Provider::OpenRouter, local(p)));
        }
        // ORDER IS THE CONTRACT: announce() and --api-keys-out both run BEFORE
        // any listener binds, and --ready-file is touched only from inside the
        // per-surface bind (apiserve::serve_all / brain_dbus::serve). That is
        // what lets a script wait on the ready file ALONE and then read the
        // keys with no retry. Do not move write_keys below serve_all.
        // Gate: tests/e2e/ready.bats.
        for s in &surfaces {
            s.announce();
        }
        if let Some(path) = &a.api_keys_out {
            if let Err(e) = apiserve::write_keys(&surfaces, std::path::Path::new(path)) {
                eprintln!("brain serve: --api-keys-out {path}: {e}");
            }
        }
        let opts = apiserve::ServeOpts::new().with_shutdown(shutdown).with_supplier(supplier).with_ready(a.ready);
        if let Err(e) = apiserve::serve_all(executor, surfaces, opts) {
            eprintln!("brain serve: {e}");
            std::process::exit(1);
        }
    }

    // HTTP has drained and returned. Give a backgrounded D-Bus surface a bounded
    // window to finish its own graceful shutdown (both saw the same `shutdown`
    // fire, so it should already be on its way out) rather than let `main`
    // return out from under a live thread.
    if let Some(h) = dbus_handle {
        let deadline = std::time::Instant::now() + DBUS_JOIN_TIMEOUT;
        while !h.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }
}

/// Per-GPU `(canonical index, total_bytes)`.
///
/// Capacities come from `nvidia-smi` (NVML), but NVML enumeration order is not
/// the placement order — budgets are keyed by **PCI bus id** through the device
/// registry, so `Device::Gpu(i)` budgets provably describe the same physical
/// card `gpu<i>` placement binds. Cards nvidia-smi does not report (or a
/// missing nvidia-smi) fall back to the registry's own VRAM size; with no
/// registry entries either (no GPU) the list is empty.
pub(crate) fn query_gpu_mem() -> Vec<(u32, u64)> {
    let mut mem: Vec<(u32, u64)> =
        gpu_core::devices::gpus().iter().map(|d| (d.index, d.identity.vram_bytes)).collect();
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=pci.bus_id,memory.total", "--format=csv,noheader,nounits"])
        .output();
    if let Ok(o) = out {
        if o.status.success() {
            for l in String::from_utf8_lossy(&o.stdout).lines() {
                let mut it = l.split(',').map(str::trim);
                let (Some(pci), Some(mib)) = (it.next(), it.next().and_then(|m| m.parse::<u64>().ok()))
                else {
                    continue;
                };
                if let Some(d) = gpu_core::devices::device_by_pci(pci) {
                    if let Some(slot) = mem.iter_mut().find(|(i, _)| *i == d.index) {
                        slot.1 = mib << 20;
                    }
                }
            }
        }
    }
    mem.retain(|&(_, bytes)| bytes > 0);
    mem
}

/// Old name for [`host_ram_available`], kept as a thin alias — `perf_cli.rs`
/// still calls it by this name and there is no reason to touch those four
/// call sites in the same change that fixes the unified-memory double-count.
/// The `/proc/meminfo`-only parsing this used to do locally now lives in
/// `memauth::HostProbe`, which additionally intersects a cgroup v2 limit —
/// tighter and more correct, never looser, than the old behaviour.
pub(crate) fn query_ram_bytes() -> u64 {
    host_ram_available()
}

#[cfg(test)]
mod tests {
    use super::HELP;

    /// Every flag the hand-rolled loop in `run_serve` actually parses must be
    /// documented in `HELP` -- this is the content-side gate for the bench
    /// incident (`--listen` was undocumented AND unparsed; a flag that is
    /// parsed but undocumented is the same bug in the other direction).
    #[test]
    fn help_documents_every_flag_the_parser_accepts() {
        for f in [
            "--gpt", "--yolo", "--max-new", "--temp", "--top-k", "--seed", "--conf", "--dbus", "--dbus-system", "--dbus-name",
            "--reserve-gb", "--models-dir", "--anthropic", "--openai", "--openrouter", "--api-keys-out", "--ready-file", "-v", "--verbose",
        ] {
            assert!(HELP.contains(f), "{f} is parsed by run_serve but not documented in HELP");
        }
    }

    #[test]
    fn help_states_the_default_ports() {
        for p in ["8788", "8787", "8789"] {
            assert!(HELP.contains(p), "default port {p} is not documented in HELP");
        }
    }

    #[test]
    fn help_states_both_openai_base_urls() {
        assert!(HELP.contains("http://127.0.0.1:PORT/v1"));
        assert!(HELP.contains("http://127.0.0.1:PORT "), "the bare (non-/v1) base URL must be documented too");
        assert!(HELP.contains("/chat/completions"));
        assert!(HELP.contains("/v1/chat/completions"));
    }

    #[test]
    fn help_states_the_anthropic_dialect_is_v1_only() {
        assert!(HELP.contains("/v1/messages"));
        assert!(HELP.contains("/v1-ONLY"));
    }

    /// Regression gate for the bench incident: `--listen HOST:PORT` was
    /// silently ignored because it never existed. State plainly that it does
    /// not exist, so a reader does not have to grep the parser to find out.
    #[test]
    fn help_states_there_is_no_listen_flag() {
        assert!(HELP.contains("no --listen"));
    }
}
