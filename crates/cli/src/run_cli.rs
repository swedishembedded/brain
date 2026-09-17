// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain serve --stdio` - the event-driven stdio controller loop.
//!
//! Reads JSONL [`events::Event`] lines from stdin (a blocking read is the idle
//! wait), feeds each to a [`runtime::Controller`], and writes every emitted event
//! back as a JSONL line to stdout (flushed per line). Diagnostics go to stderr.
//!
//! Flags:
//!   * `--gpt <path>` (or env `BRAIN_GPT2`) - load a GPT checkpoint as the text
//!     model. With none, a fake echo model is used so the loop is testable
//!     without a trained model.
//!   * `--yolo <path>` (or env `BRAIN_YOLOV8`) - load a YOLO checkpoint as the
//!     object detector. With none, a `FakeDetectModel` returns a fixed box so the
//!     loop runs without a trained detector.
//!   * `--conf <f32>` (or env `BRAIN_CONF`) - detection confidence threshold for
//!     the YOLO detector (default 0.25). Lower it so a lightly-trained tiny model's
//!     low-confidence boxes still surface. No effect on the fake detector.
//!   * `--max-new N`, `--temp X`, `--top-k K`, `--seed S` - generation config.
//!   * Text-to-speech has no flag: it is env-only (`BRAIN_QWEN3TTS_WEIGHTS` /
//!     `BRAIN_QWEN3TTS_CKPT` / `BRAIN_QWEN3TTS_LANG`, the same variables the
//!     D-Bus resident reads), and only in a build carrying the `qwen3tts-synth`
//!     feature - the TTS stack is too heavy for the default binary. Without it,
//!     a `user_synth_request` still answers, with the terminal empty chunk.
//!   * `--models-dir <path>` (or env `BRAIN_MODELS_DIR`) - the global model
//!     directory `brain serve --dbus` scans at startup to build the served-model
//!     catalog (one entry per carded file, keyed by model-card id). Defaults to
//!     `$XDG_DATA_HOME/brain/models` else `$HOME/.local/share/brain/models`.

use std::io::{BufRead, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use events::Envelope;
use runtime::{
    Controller, DetectModel, Emit, FakeDetectModel, FakeInferModel, GenConfig, GptInfer, Registry,
    YoloDetect,
};

/// A live [`Emit`] sink over a stdout writer: encodes each envelope to a JSONL
/// line and flushes it immediately, so `brain serve --stdio` streams token-by-token as the
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
brain serve - serve brain's models over HTTP, D-Bus, or stdio

USAGE:
  brain serve [SURFACE ...] [options]     # one or more serving surfaces
  brain serve --stdio                     # the event-driven stdio JSONL loop
  brain serve                             # NO surface flag: same as --stdio

HTTP INFERENCE APIS  (each behind its own key)
  --openai [ADDR]        OpenAI-compatible dialect       (default port 8788)
  --openrouter [ADDR]    OpenRouter-compatible dialect   (default port 8789)
  --anthropic [ADDR]     Anthropic Messages dialect      (default port 8787)

  ADDR is optional; when given it is a bare PORT (bind stays 127.0.0.1, same
  as always), a HOST:PORT (bind that one interface instead), a bare HOST (that
  interface, on the dialect's default port), or a comma-separated list of any
  of those to listen on SEVERAL interfaces at once for the SAME dialect - one
  key is generated per dialect regardless of how many addresses it binds, not
  one per address:
      brain serve --openai 0.0.0.0:8788                    # every interface
      brain serve --openai 192.168.1.5:8788,127.0.0.1:8788  # two, explicitly
      brain serve --openai 9000                             # old bare-port form

  BASE URL - for OpenAI and OpenRouter, BOTH of these work, because every route
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
  A fresh key per DIALECT per launch by default, printed on stdout as
  `APIKEY <provider> <key>`; --api-keys-out writes the same keys as JSON, 0600.
  $BRAIN_API_KEY pins a fixed key instead (e.g. via --config) - every dialect
  then shares that SAME key rather than each getting its own random one.

  With no ADDR given, a surface binds 127.0.0.1 (unchanged default). There is
  no --listen / --host / --bind flag -- the address goes directly after the
  dialect flag it names, as ADDR above.

D-BUS CONTROL SURFACE
  --dbus                 serve com.swedishembedded.Brain1 on the session bus
  --dbus-system          use the system bus instead. Needs a system.d policy:
                         the deb installs a vetted one (calls restricted to
                         root + the 'brain' group); from a checkout, install
                         scripts/build/com.swedishembedded.Brain1.conf to
                         /usr/share/dbus-1/system.d/ yourself.
  --dbus-name NAME       request NAME instead of com.swedishembedded.Brain1
  --dbus-address ADDR    serve on an explicit bus address (unix:path=/run/brain/bus)
                         instead of the one discovered from the environment.
                         A DETACHED server inherits no session bus, since the
                         shell that launched it is gone, so this is how -d and
                         --dbus are combined.

PROCESS LIFECYCLE
  -d, --detach           run in the background and return once every requested
                         surface is listening (not merely once the process has
                         started). Prints the pid; output goes to the log below.
  --reload               stop whatever server is already running, then take its
                         place. Imperative: nothing watches anything, so a
                         restart only ever happens when asked for.
  --stop                 stop the running server.
  --status               report whether one is running, and its pid. Exits 0
                         when one is, 3 when none is - so a script can branch
                         on it without parsing the message.

  At most one server runs per user at a time, enforced by an exclusive lock on
  the pidfile rather than by matching process names, so a crash leaves nothing
  to clean up and one user's server never blocks another's. State lives under
  $BRAIN_RUNTIME_DIR, else $XDG_RUNTIME_DIR/brain, else <tmpdir>/brain-<uid>:
  serve.pid, serve.log and (when -d is used without --ready-file) serve.ready.

SERVING OPTIONS
  --models-dir DIR       directory scanned at startup for the served catalog
                         (else $BRAIN_MODELS_DIR, else $XDG_DATA_HOME/brain/models)
  --api-keys-out FILE    write {\"openai\":\"sk-brain-…\", …} as JSON, mode 0600
  --reserve-gb N         GB of VRAM kept free per GPU for activations (default 2)
  --watch-adapters DIR   watch DIR for promoted LoRA adapters and hot-swap the
                         served Qwen3 onto the newest one, with no restart. OFF
                         unless given. The newest is the highest-versioned
                         adapter-NNNNNN.safetensors, never the newest mtime.
                         A request already running is never interrupted: the
                         swap applies to the next one.
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

QWEN3 SERVING TUNABLES  (which checkpoint to serve stays BRAIN_QWEN_WEIGHTS/
                         BRAIN_QWEN_TOKENIZER; everything about HOW is a flag)
  --qwen-ctx N           built context length. Default: auto-sized to the
                         target device's real free VRAM (minus --reserve-gb),
                         capped at the checkpoint's own trained
                         max_position_embeddings. Give this to pin an exact
                         value instead.
  --qwen-max-batch N     concurrent decode slots (default 16).
  --qwen-kv-fp32         opt OUT of int8 KV (on by default).
  --qwen-kv-calib        opt IN to a kv_calib.json beside the checkpoint
                         (off by default).
  --qwen-kv-offload-gb N host RAM for preempted sessions' KV, fractional
                         allowed (default 0, off).
  --qwen-weights-int8    quantize the 7 per-layer linears to int8 (off by
                         default) - for a checkpoint whose fp32 weights
                         alone do not fit any card's budget.
  --qwen-max-prefill N   cap the chunked-prefill row count below its 512
                         default (clamped 1..=512) - shrinks the paged-
                         attention scratch buffer linearly, the single
                         largest per-token-scaling cost in the engine.

STDIO CONTROLLER  (the default, with no surface flag)
  --gpt PATH             GPT checkpoint (else $BRAIN_GPT2; else a fake echo model)
  --yolo PATH             YOLO checkpoint (else $BRAIN_YOLOV8; else a fake detector)
  --conf X                detection confidence threshold (else $BRAIN_CONF, 0.25)
  --max-new N  --temp X  --top-k K  --seed S      generation config
  Reads JSONL events on stdin, writes JSONL events on stdout, one per line.
  Example: printf '{\"event\":\"user_text\",\"text\":\"hi\"}\\n' | brain serve --stdio

MODEL CONFIGURATION (WHICH models: env vars; HOW to serve one: flags)
  Which models this server actually serves is chosen ENTIRELY by BRAIN_* env
  vars (BRAIN_QWEN_WEIGHTS, BRAIN_LFM2, BRAIN_NEMOTRONASR, ...): a model whose
  weights var is unset is simply not served. Tuning an already-selected
  model (context length, batching, precision, ...) is a flag where one
  exists - see QWEN3 SERVING TUNABLES above for Qwen3's. Run `brain serve
  --help` for the full reference table of every serving variable.

  FETCHING: a request for a model that is not pulled errors with zero network
  I/O unless --autofetch (or $BRAIN_AUTO_FETCH=1) was passed; `brain pull`
  fetches without it.

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
  brain serve --openai 0.0.0.0:8788          # OpenAI API on every interface
  brain serve --dbus --anthropic --openrouter
";

/// The value that must follow a value-taking flag. A flag with no value is a
/// typo, not a request for the default: a trailing `--gpt` used to silently
/// mean \"no checkpoint\", erasing a `BRAIN_GPT2` already read from the
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
/// request for the default - `--reserve-gb 2G` used to silently serve with
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

/// Parse the optional ADDR value that may follow `--anthropic` / `--openai`
/// / `--openrouter` (see HELP's ADDR paragraph): a comma-separated list where
/// each item is a `HOST:PORT`, a bare `HOST` (bound on `default_port`), or a
/// bare `PORT` (bound on loopback) - so the pre-existing bare-port spelling
/// keeps meaning exactly what it always did. `None` if `spec` does not parse
/// as such a list AT ALL (not even one item), which is what lets
/// [`take_addrs`] tell "a real address list was given" apart from "the next
/// token is unrelated" (typically the next flag) without erroring on the
/// latter.
fn parse_listen_addrs(spec: &str, default_port: u16) -> Option<Vec<SocketAddr>> {
    let mut out = Vec::new();
    for tok in spec.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            return None;
        }
        let addr = if let Ok(a) = tok.parse::<SocketAddr>() {
            a
        } else if let Ok(port) = tok.parse::<u16>() {
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
        } else if let Ok(ip) = tok.parse::<IpAddr>() {
            SocketAddr::new(ip, default_port)
        } else {
            return None;
        };
        out.push(addr);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// The address list that must follow a dialect flag, when one was actually
/// given: peeks at the next token and consumes it only if
/// [`parse_listen_addrs`] accepts it whole, otherwise leaves it untouched (it
/// is the next flag, e.g. `--openai --dbus`) and returns the single loopback
/// default. Mirrors the old bare-port `take_port` closure's peek-without-
/// consuming contract exactly, widened from one `u16` to a `Vec<SocketAddr>`.
fn take_addrs(args: &[String], i: &mut usize, default_port: u16) -> Vec<SocketAddr> {
    match args.get(*i + 1).and_then(|s| parse_listen_addrs(s, default_port)) {
        Some(addrs) => {
            *i += 1;
            addrs
        }
        None => vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), default_port)],
    }
}

/// Fill the controller's text-to-speech seam, when this build has one.
///
/// Configuration is env-only and reuses the SAME variables the D-Bus resident
/// (`crate::resident_tts::TtsResident::from_env`) already reads - one spelling
/// across both serving surfaces, and no new flag on `brain serve`. With
/// `BRAIN_QWEN3TTS_WEIGHTS` unset there is no synth model, which is exactly the
/// behaviour of every build before this: a `user_synth_request` answers with the
/// terminal empty `audio_chunk`.
#[cfg(feature = "qwen3tts-synth")]
fn register_synth(registry: &mut Registry) {
    match runtime::Qwen3TtsSynthModel::from_env() {
        None => {}
        Some(Ok(m)) => {
            eprintln!("brain serve --stdio: Qwen3-TTS synth model registered");
            registry.synth = Some(Box::new(m));
        }
        // The weights var was set but the checkpoint is unusable: say so rather
        // than silently serving no audio from a path the operator meant to use.
        Some(Err(e)) => eprintln!("brain serve --stdio: {e}; synthesis disabled"),
    }
}

/// No-op twin for the default build, which does not link the TTS stack.
#[cfg(not(feature = "qwen3tts-synth"))]
fn register_synth(_registry: &mut Registry) {}

pub fn run_serve(args: &[String]) {
    let mut gpt_path = std::env::var("BRAIN_GPT2").ok();
    let mut yolo_path = std::env::var("BRAIN_YOLOV8").ok();
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
    // Opt-in continuous-learning hot swap (`--watch-adapters DIR`): watch DIR
    // for a promoted LoRA adapter and point the served Qwen3 at it without a
    // restart. `None` (the default) spawns no watcher at all -- a server that
    // silently reloads its weights because a file appeared on disk is not
    // something an operator should get without asking for it.
    let mut watch_adapters: Option<String> = None;
    // Every `--qwen-*` flag, built up as its own arm below is matched -
    // `crate::resident_llm::QwenServeConfig`'s own `Default` is every
    // historical env-var default, byte for byte.
    let mut qwen_cfg = crate::resident_llm::QwenServeConfig::default();
    // HTTP inference APIs (`--anthropic|--openai|--openrouter [ADDR]`), each
    // bound on one or more addresses (127.0.0.1 by default) with ONE
    // per-dialect key generated at startup regardless of how many addresses
    // it binds. All share the one executor (with D-Bus, if also selected).
    // `--api-keys-out FILE` writes the keys as JSON for scripted clients /
    // the e2e test.
    let mut anthropic: Option<Vec<SocketAddr>> = None;
    let mut openai: Option<Vec<SocketAddr>> = None;
    let mut openrouter: Option<Vec<SocketAddr>> = None;
    let mut api_keys_out: Option<String> = None;
    let mut ready_file: Option<String> = None;
    // Detached serving (`-d`), imperative replacement (`--reload`) and the two
    // lifecycle queries -- see `crate::serve_daemon`.
    let mut detach = false;
    let mut reload = false;
    let mut do_stop = false;
    let mut do_status = false;
    // An explicit bus address, for a bus that outlives the launching shell.
    let mut dbus_addr: Option<String> = None;
    // Diagnostic verbosity (`-v`/`--verbose [0-3]`, else $BRAIN_VERBOSE) -- see
    // HELP above for what each tier gates. Parsed and installed globally now
    // (`main::install_verbosity`, run before any subcommand including this
    // one), so by the time this loop runs, `-v`/`--verbose` are already
    // stripped from `args` -- no local state or match arms needed here.

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
            // The stdio JSONL controller is what runs when no surface flag
            // is given below; `--stdio` names that path explicitly rather
            // than leaving it as a silent default a reader has to infer.
            "--stdio" => {}
            "--dbus" => dbus = true,
            "--dbus-system" => {
                dbus = true;
                dbus_system = true;
            }
            "--dbus-name" => dbus_name = Some(val(args, &mut i, "--dbus-name")),
            "--dbus-address" => {
                dbus = true;
                dbus_addr = Some(val(args, &mut i, "--dbus-address"));
            }
            "-d" | "--detach" => detach = true,
            "--reload" => reload = true,
            "--stop" => do_stop = true,
            "--status" => do_status = true,
            "--reserve-gb" => dbus_reserve_gb = parsed(args, &mut i, "--reserve-gb"),
            "--models-dir" => models_dir = Some(val(args, &mut i, "--models-dir")),
            "--watch-adapters" => watch_adapters = Some(val(args, &mut i, "--watch-adapters")),
            "--qwen-ctx" => qwen_cfg.ctx = Some(parsed(args, &mut i, "--qwen-ctx")),
            "--qwen-max-batch" => qwen_cfg.max_batch = parsed(args, &mut i, "--qwen-max-batch"),
            "--qwen-kv-fp32" => qwen_cfg.kv_int8 = false,
            "--qwen-kv-calib" => qwen_cfg.kv_calib_opt_in = true,
            "--qwen-kv-offload-gb" => qwen_cfg.kv_offload_gb = parsed(args, &mut i, "--qwen-kv-offload-gb"),
            "--qwen-weights-int8" => qwen_cfg.weights_int8 = true,
            "--qwen-max-prefill" => qwen_cfg.max_prefill_cap = parsed(args, &mut i, "--qwen-max-prefill"),
            "--anthropic" => anthropic = Some(take_addrs(args, &mut i, 8787)),
            "--openai" => openai = Some(take_addrs(args, &mut i, 8788)),
            "--openrouter" => openrouter = Some(take_addrs(args, &mut i, 8789)),
            "--api-keys-out" => api_keys_out = Some(val(args, &mut i, "--api-keys-out")),
            "--ready-file" => ready_file = Some(val(args, &mut i, "--ready-file")),
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
    // Lifecycle flags (-d / --reload / --stop / --status) are handled in
    // `main`, before anything probes a device -- see `serve_daemon::lifecycle`
    // for why that ordering is load-bearing. They are still accepted by the
    // parser above so an unknown-flag error is never raised for them; by the
    // time this runs, their work is done.
    let _ = (detach, reload, do_stop, do_status);

    if !args.iter().any(|a| a == "--seed") {
        cfg.seed = data::rng::random_seed();
        eprintln!("brain serve: no --seed given, using random seed {} (pass --seed {} to reproduce)", cfg.seed, cfg.seed);
    }

    // Each bound ADDRESS reports its own `Gate::bound` call (`apiserve::serve_all`
    // spawns one listener per address), not each dialect flag - so a dialect
    // bound on N interfaces counts N times here, or `--ready-file` would wait
    // past the point every requested listener is actually up.
    let addr_count = |a: &Option<Vec<SocketAddr>>| a.as_ref().map_or(0, Vec::len);
    let surfaces_requested = dbus as usize + addr_count(&anthropic) + addr_count(&openai) + addr_count(&openrouter);
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
            dbus_addr,
            dbus_name,
            reserve_gb: dbus_reserve_gb,
            models_dir,
            anthropic,
            openai,
            openrouter,
            api_keys_out,
            ready,
            watch_adapters,
            qwen_cfg,
        });
    }

    // Build the registry: a real GPT if a checkpoint was given, else a fake echo
    // model so the loop runs end-to-end without a trained model.
    let infer: Box<dyn runtime::InferModel> = match &gpt_path {
        Some(path) => {
            eprintln!("brain serve --stdio: loading GPT checkpoint {path}");
            // Char models embed itos; the pump uses it for the EOS-less stop at
            // max_new. We leave eos as configured (None unless the user sets one).
            Box::new(GptInfer::load(path))
        }
        None => {
            eprintln!("brain serve --stdio: no --gpt checkpoint; using fake echo model");
            // The fake echoes a fixed greeting and terminates at its EOS sentinel.
            cfg.eos = Some(256);
            Box::new(FakeInferModel::echoing("hello from brain"))
        }
    };
    // A real YOLO if a checkpoint was given, else the fixed-box fake detector.
    let detect: Box<dyn DetectModel> = match &yolo_path {
        Some(path) => {
            eprintln!("brain serve --stdio: loading YOLO checkpoint {path}");
            let mut det = YoloDetect::load(path);
            if let Some(c) = conf {
                eprintln!("brain serve --stdio: detection confidence threshold {c}");
                // Keep the default IoU (0.45); only override the confidence gate.
                det = det.with_thresholds(c, 0.45);
            }
            Box::new(det)
        }
        None => {
            eprintln!("brain serve --stdio: no --yolo checkpoint; using fake detector");
            Box::new(FakeDetectModel::default())
        }
    };

    let mut registry = Registry::with_models(infer, detect);
    register_synth(&mut registry);
    let mut ctrl = Controller::with_config(registry, cfg);

    // Expose the generic capability providers over the event API (manifest_request
    // / action_request) - the same actions `brain do` runs, now network-reachable.
    ctrl.register_provider(std::sync::Arc::new(s3dit::caps::ZImageProvider::load().expect("z-image provider")));
    ctrl.register_provider(std::sync::Arc::new(lfm2::caps::LfmProvider::new()));

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
                eprintln!("brain serve --stdio: stdin error: {e}");
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
///
/// Returns `crate::resident::Serving`, not a bare `Executor`: the
/// continuous-learning hot swap needs the CONCRETE `QwenResident` handle
/// alongside the type-erased one the executor holds - see that type's doc.
fn build_serving_executor(reserve_gb: u64, models_dir: Option<String>, qwen_cfg: crate::resident_llm::QwenServeConfig) -> crate::resident::Serving {
    // Discover the GPUs' capacity so the scheduler can budget/evict against real VRAM,
    // then narrow to what `--device` made schedulable. With no `--device` the set is
    // every device, which is exactly the "use all the hardware wisely" default.
    // FREE bytes, not total. `--reserve-gb` is then carved out of what is
    // actually available, so a card a neighbouring process is already holding
    // 18 GiB of is budgeted at 6 GiB rather than 24. Budgeting from the card's
    // SIZE is what let the daemon plan a placement the driver then refused -
    // the scheduler's own accounting said the card was empty. Same probe the
    // one-shot placer uses (`gpu_core::capacity`), so the two halves of this
    // process can no longer disagree about the same card at the same instant.
    let mut all_gpus = gpu_core::capacity::available_gpus();
    // No NVIDIA GPU, but the wgpu backend can drive an integrated GPU (e.g. Intel
    // Arc on Meteor Lake): budget it as a schedulable `Gpu` lane. Integrated GPUs
    // have no dedicated VRAM - they share system RAM - so size the budget like the
    // NPU (a modest fraction of RAM). This is what makes `--device gpu` (and the
    // all-devices default) actually schedule onto the iGPU on such boxes.
    // Devices this fallback creates ALWAYS share physical RAM with the CPU
    // (that is the case it exists for) - tracked so they can be declared into
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
            // fraction reserved here - this device budget only needs to be AT
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
    // integrated (an Arc/Xe iGPU reporting real VRAM via query_gpu_mem - the
    // common case on this box - never goes through the fallback above, so it
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

    // Schedulable NPUs: `--device` narrows to `set.npus`; with no `--device`, any NPU
    // present is scheduled. The Meteor-Lake-class NPU shares system RAM, so it gets a
    // modest per-device budget. A model with an NPU path (MemCost.npu > 0) is then
    // auto-placed on the NPU in preference to CPU/GPU (see place::pick_device).
    let npu_indices: Vec<u32> = match set {
        Some(s) => s.npus.clone(),
        None if npu::openvino::npu_present() => vec![0],
        None => vec![],
    };
    let ram = host_ram_available();
    // NPUs always share system RAM (see the comment above); their device
    // budget only needs to be at least the pool's total, same reasoning as
    // the iGPU fallback - `resident::build_executor` declares them into the
    // shared pool alongside `unified_gpus` and Device::Cpu.
    let npus: Vec<(u32, u64)> = npu_indices.iter().map(|&i| (i, ram)).collect();

    // What is actually schedulable is `gpus`/`npus`/`cpu_schedulable`, not just
    // `gpus` - a prior version of this message said "scheduling on CPU only"
    // purely from `gpus.is_empty()`, which was wrong on two counts whenever an
    // NPU was involved: `--device npu` schedules on the NPU (never CPU - CPU
    // compute is excluded, see `cpu_compute_ram` below), and `--device npu,cpu`
    // schedules on both, not "CPU only".
    if gpus.is_empty() && npus.is_empty() {
        if all_gpus.is_empty() {
            eprintln!("brain serve: no GPUs or NPUs detected; serving with CPU-only budget");
        } else {
            eprintln!("brain serve: --device excluded every GPU; scheduling on CPU only");
        }
    } else if gpus.is_empty() && !npus.is_empty() {
        if cpu_schedulable {
            eprintln!("brain serve: --device excluded every GPU; scheduling on NPU + CPU");
        } else {
            eprintln!("brain serve: --device restricted to NPU; scheduling on NPU only (CPU and GPU excluded)");
        }
    }
    let reserved = reserve_gb << 30;
    // Host RAM stays a cache/spill tier even when the CPU is not schedulable for
    // compute - `--device gpu` bounds where work runs, not where bytes may rest.
    let cpu_compute_ram = if cpu_schedulable { ram } else { 0 };
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
    let dir = loader::model_dir::resolve(models_dir.as_deref());
    if let Some(d) = &dir {
        // First run on a fresh install: the dir doesn't exist yet. Create it so
        // the scan is clean (an empty catalog, not an ENOENT warning) - models
        // dropped in later are picked up on the next `brain serve` with no env
        // vars needed.
        if let Err(e) = std::fs::create_dir_all(d) {
            eprintln!("brain serve: could not create model dir {} ({e}); scan may be empty", d.display());
        }
        eprintln!("brain serve: scanning model dir {}", d.display());
    }
    crate::resident::build_executor(&gpus, &npus, &unified_gpus, reserved, cpu_compute_ram, ram, dir.as_deref(), residency::Policy::from_env(), qwen_cfg)
}

/// Live host RAM this process could actually get right now: `MemAvailable`
/// intersected with any cgroup v2 limit, then with `--limit-ram-total` if one
/// was published. A thin alias over `loader::placement::
/// host_ram_available` - the same probe the production placer's own re-probe
/// path uses, so there is exactly one implementation of "how much host RAM
/// may this process use", not a second copy living here. `query_ram_bytes`
/// (the old name, kept public within the crate since `perf_cli.rs` calls it
/// by that name) is a further thin alias over this.
pub(crate) fn host_ram_available() -> u64 {
    loader::placement::host_ram_available()
}

/// Which serving surfaces to bring up and their config (see `run_apis`).
struct RunApis {
    dbus: bool,
    dbus_system: bool,
    /// `--dbus-address ADDR`: serve on an explicit bus rather than the one
    /// discovered from the environment. A detached server inherits no session
    /// bus (the shell that launched it is gone), so this is how it reaches a
    /// long-lived one.
    dbus_addr: Option<String>,
    dbus_name: Option<String>,
    reserve_gb: u64,
    models_dir: Option<String>,
    anthropic: Option<Vec<SocketAddr>>,
    openai: Option<Vec<SocketAddr>>,
    openrouter: Option<Vec<SocketAddr>>,
    api_keys_out: Option<String>,
    /// Notified once per bound surface (HTTP + D-Bus); disabled unless
    /// `--ready-file` was given. See `brain_shutdown::ready::Gate`.
    ready: brain_shutdown::ready::Gate,
    /// `--watch-adapters DIR`: the continuous-learning hot-swap watcher's
    /// directory, `None` (no watcher) unless the flag was given.
    watch_adapters: Option<String>,
    /// Every `--qwen-*` flag, parsed once here - see
    /// `resident_llm::QwenServeConfig`'s own doc for each field.
    qwen_cfg: crate::resident_llm::QwenServeConfig,
}

/// Build the one shared executor and bring up the requested surfaces: D-Bus
/// (`com.swedishembedded.Brain1`) and/or the HTTP inference APIs (Anthropic / OpenAI /
/// OpenRouter), each on its own localhost port with a key generated at startup. When
/// D-Bus and an HTTP surface both run, D-Bus gets its own thread (it owns a tokio
/// runtime) and the HTTP servers own the main thread; a single surface blocks directly.
/// A `brain_dbus::serve` failure is almost always "no bus at this address" (no
/// desktop session, no `dbus-run-session`, no system bus policy for this user) -
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
/// hang `brain serve` forever - [`brain_shutdown::install_signals`]'s own
/// second-signal escape hatch is the backstop if this window is not enough.
const DBUS_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// Build the transparent auto-fetch supplier for a live `brain serve`, but
/// only when fetching is enabled: `--autofetch` (which `main` publishes by
/// setting `BRAIN_AUTO_FETCH=1`) or the same variable the caller exported.
/// Default is OFF -- a request for a model that is not pulled returns an
/// error to that caller, with zero network I/O; nothing downloads from a
/// node the operator did not opt in. `None` also when no models directory
/// can be resolved at all (no `$HOME`), same as before this existed.
///
/// Every HTTP/D-Bus surface in this process shares ONE supplier instance (not
/// one per surface): `StoreSupplier`'s in-flight map is what makes concurrent
/// requests for the same cold model share a single fetch rather than each
/// surface racing its own download.
fn build_auto_fetch_supplier(models_dir: Option<&str>) -> Option<Arc<dyn residency::ModelSupplier>> {
    if !crate::supply::auto_fetch_enabled() {
        eprintln!("brain serve: auto-fetch off -- a request for a model that is not pulled errors (pass --autofetch or set BRAIN_AUTO_FETCH=1 to enable on-demand fetching)");
        return None;
    }
    // The SAME models directory `build_serving_executor`'s startup scan
    // resolved, so a freshly auto-fetched model lands exactly where a restart's
    // scan would find it again.
    let dir = loader::model_dir::resolve(models_dir)?;
    let store = brain_modelstore::Store::new(dir);
    let hub: Box<dyn brain_modelstore::Hub> = Box::new(brain_modelstore::HfHub::new());
    Some(Arc::new(crate::supply::StoreSupplier::new(store, hub)))
}

fn run_apis(a: RunApis) {
    // Captured before `build_serving_executor` below moves `a.models_dir`.
    let models_dir_for_heal = a.models_dir.clone();
    let supplier = build_auto_fetch_supplier(a.models_dir.as_deref());
    let crate::resident::Serving { executor, qwen } = build_serving_executor(a.reserve_gb, a.models_dir, a.qwen_cfg);
    let manifests = executor.manifests();
    let served: Vec<&str> = manifests.iter().map(|m| m.model.as_str()).collect();
    eprintln!("brain serve: models: {}", served.join(", "));

    // `--autofetch` also proactively heals what the scan above already found
    // broken (an interrupted download, a GGUF still needing its one-time
    // conversion) in the background, instead of only fixing it the next time
    // a client happens to request that exact model. `supplier` is already
    // `None` unless auto-fetch is enabled, so no separate check is needed here.
    if let Some(sup) = &supplier {
        if let Some(dir) = loader::model_dir::resolve(models_dir_for_heal.as_deref()) {
            crate::supply::heal_missing_models_in_background(dir, sup.clone(), executor.clone());
        }
    }

    let http = a.anthropic.is_some() || a.openai.is_some() || a.openrouter.is_some();

    // One shutdown source for every surface this process serves. SIGINT/SIGTERM
    // disposition is process-wide: if D-Bus and HTTP each installed their own
    // `tokio::signal::ctrl_c()` handler, only one registration would ever
    // actually see the signal - see `brain_shutdown` for the failure this
    // caused. Installed once, here, before either surface's runtime exists, so
    // Ctrl-C/SIGTERM reaches whichever surfaces are actually running.
    let (trigger, shutdown) = brain_shutdown::channel();
    brain_shutdown::install_signals(trigger);

    // The continuous-learning hot swap, opt-in (`--watch-adapters DIR`). Held
    // for the whole serving lifetime: the handle stops and joins its thread on
    // drop, so the watcher cannot outlive the surfaces it was swapping models
    // under. `qwen` is the CONCRETE resident handle -- `set_adapter` is
    // inherent, so the erased one the executor holds could not do this.
    let _adapter_watcher = crate::continuous_train::spawn_adapter_watcher(a.watch_adapters.as_ref().map(std::path::Path::new), qwen, &executor);

    let dbus_handle = if a.dbus {
        let opts = brain_dbus::DbusOpts {
            // An explicit address wins over both defaults: it is the only one
            // the caller had to name deliberately.
            bus: match (&a.dbus_addr, a.dbus_system) {
                (Some(addr), _) => brain_dbus::BusKind::Address(addr.clone()),
                (None, true) => brain_dbus::BusKind::System,
                (None, false) => brain_dbus::BusKind::Session,
            },
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
        // ONE key per DIALECT, shared across every address it binds - a
        // client using the OpenAI surface must keep working with the same
        // key regardless of which of its bound interfaces it connects
        // through, and `--api-keys-out`'s `{provider: key}` map (keyed by
        // dialect, not by address) has nowhere to put a second key anyway.
        let mut surfaces = Vec::new();
        for (addrs, provider) in [
            (a.anthropic, apiserve::Provider::Anthropic),
            (a.openai, apiserve::Provider::OpenAI),
            (a.openrouter, apiserve::Provider::OpenRouter),
        ] {
            if let Some(addrs) = addrs {
                let key = apiserve::surface::resolved_key();
                for addr in addrs {
                    surfaces.push(apiserve::Surface::new(provider, addr, key.clone()));
                }
            }
        }
        // ORDER IS THE CONTRACT: announce() and --api-keys-out both run BEFORE
        // any listener binds, and --ready-file is touched only from inside the
        // per-surface bind (apiserve::serve_all / brain_dbus::serve). That is
        // what lets a script wait on the ready file ALONE and then read the
        // keys with no retry. Do not move write_keys below serve_all.
        // Gate: tests/e2e/ready.bats.
        // One APIKEY line per DIALECT (never per address): every surface in
        // a dialect's group carries the identical key built above, so a
        // second, third, ... address would only reprint the same line.
        let mut announced: Vec<apiserve::Provider> = Vec::new();
        for s in &surfaces {
            if !announced.contains(&s.provider) {
                s.announce();
                announced.push(s.provider);
            }
        }
        if let Some(path) = &a.api_keys_out {
            if let Err(e) = apiserve::write_keys(&surfaces, std::path::Path::new(path)) {
                eprintln!("brain serve: --api-keys-out {path}: {e}");
            }
        }
        // `serve_all` takes the executor BY VALUE and would drop it - and with
        // it every resident model and every GPU device those models hold - on
        // its way out. One handle is deliberately leaked so that drop is not
        // the last one: the residents (and their devices) then stay alive
        // until the process itself ends, instead of being torn down during
        // shutdown.
        //
        // This is a real, measured hazard rather than caution. A server that
        // had placed a model across TWO cards reproducibly faulted on SIGTERM
        // - intermittently, inside an NVIDIA driver thread (a jump to an
        // unmapped address, so no Rust frame is even involved), and never with
        // the same run restricted to ONE card. Nothing about that teardown is
        // needed: the process is about to exit, and the driver releases every
        // allocation when it does.
        //
        // `Executor` is a cheap Arc-backed handle, so the leak itself is a
        // pointer, and it is bounded by the process lifetime by construction.
        std::mem::forget(executor.clone());
        // Flipped as soon as shutdown FIRES rather than after `serve_all`
        // returns, because a model evicted during the drain would otherwise
        // still take the destroy path - see `gpu_core::set_process_exiting`.
        {
            let sd = shutdown.clone();
            let _ = std::thread::Builder::new().name("brain-gpu-exit".to_string()).spawn(move || {
                while !sd.is_shutdown() {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
                gpu_core::set_process_exiting();
            });
        }
        let opts = apiserve::ServeOpts::new().with_shutdown(shutdown).with_supplier(supplier).with_ready(a.ready);
        if let Err(e) = apiserve::serve_all(executor, surfaces, opts) {
            eprintln!("brain serve: {e}");
            std::process::exit(1);
        }
    }

    // Every surface has drained: from here on the process is only unwinding, so
    // GPU devices are leaked instead of destroyed. Destroying them buys nothing
    // (the driver releases everything when the process dies) and on this NVIDIA
    // driver a multi-device teardown reproducibly faulted -- see
    // `gpu_core::set_process_exiting`. Set BEFORE the executor's residents (and
    // their devices) are dropped below.
    gpu_core::set_process_exiting();

    // Give a backgrounded D-Bus surface a bounded window to finish its own
    // graceful shutdown (both saw the same `shutdown` fire, so it should
    // already be on its way out) rather than let `main` return out from under a
    // live thread.
    if let Some(h) = dbus_handle {
        let deadline = std::time::Instant::now() + DBUS_JOIN_TIMEOUT;
        while !h.is_finished() && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
    }

    // Every surface has drained and every reply has been written: end here
    // rather than returning into libc's `exit`, which would unmap the graphics
    // driver out from under worker threads this process does not own. See
    // `brain_shutdown::exit_now` for the fault that motivates it.
    brain_shutdown::exit_now(0);
}

/// Per-GPU `(canonical index, total_bytes)` - the card's SIZE.
///
/// Reporting only (`brain devices`, the perf suite's environment block).
/// Budgets are NOT built from this: a card's size says nothing about how much
/// of it a neighbouring process is already holding, and budgeting from it is
/// what let `brain serve` place a 16 GiB model onto a card with 6 GiB
/// physically free. See [`gpu_core::capacity`], which is the one probe both this
/// and the free-bytes figure come from.
pub(crate) fn query_gpu_mem() -> Vec<(u32, u64)> {
    gpu_core::capacity::gpu_totals()
}

/// Old name for [`host_ram_available`], kept as a thin alias - `perf_cli.rs`
/// still calls it by this name and there is no reason to touch those four
/// call sites in the same change that fixes the unified-memory double-count.
/// The `/proc/meminfo`-only parsing this used to do locally now lives in
/// `memauth::HostProbe`, which additionally intersects a cgroup v2 limit -
/// tighter and more correct, never looser, than the old behaviour.
pub(crate) fn query_ram_bytes() -> u64 {
    host_ram_available()
}

#[cfg(test)]
mod tests {
    use super::HELP;
    use brain_testutil::env_lock;

    /// The node's fetch gateway: with the default (no opt-in) there is NO
    /// auto-fetch supplier, so every surface answers a request for an
    /// unpulled model with an error and zero network I/O; `--autofetch`
    /// (published as `BRAIN_AUTO_FETCH=1`) builds the supplier.
    #[test]
    fn the_auto_fetch_supplier_is_built_only_when_fetching_is_enabled() {
        let _serial = env_lock();
        let dir = std::env::temp_dir().join(format!("run-cli-autofetch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::remove_var("BRAIN_AUTO_FETCH");
        assert!(
            super::build_auto_fetch_supplier(Some(dir.to_str().unwrap())).is_none(),
            "unset must build no supplier: fetching is opt-in"
        );
        std::env::set_var("BRAIN_AUTO_FETCH", "0");
        assert!(
            super::build_auto_fetch_supplier(Some(dir.to_str().unwrap())).is_none(),
            "=0 must keep fetching off"
        );
        std::env::set_var("BRAIN_AUTO_FETCH", "1");
        assert!(
            super::build_auto_fetch_supplier(Some(dir.to_str().unwrap())).is_some(),
            "=1 (what --autofetch publishes) must build the supplier"
        );
        std::env::remove_var("BRAIN_AUTO_FETCH");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Every flag the hand-rolled loop in `run_serve` actually parses must be
    /// documented in `HELP` -- this is the content-side gate for the bench
    /// incident (`--listen` was undocumented AND unparsed; a flag that is
    /// parsed but undocumented is the same bug in the other direction).
    /// `-v`/`--verbose` are documented here too (still valid on `brain serve
    /// ...`) but are no longer in this list: `main::install_verbosity` strips
    /// them from argv globally before `run_serve`'s own loop ever runs.
    #[test]
    fn help_documents_every_flag_the_parser_accepts() {
        for f in [
            "--gpt", "--yolo", "--max-new", "--temp", "--top-k", "--seed", "--conf", "--dbus", "--dbus-system", "--dbus-name",
            "--reserve-gb", "--models-dir", "--anthropic", "--openai", "--openrouter", "--api-keys-out", "--ready-file",
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

    use super::{parse_listen_addrs, take_addrs};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};

    fn local(port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
    }

    /// The pre-existing bare-port spelling (`--openai 9000`) must keep
    /// meaning exactly what it always did: loopback, that port.
    #[test]
    fn parse_listen_addrs_accepts_a_bare_port_as_loopback() {
        assert_eq!(parse_listen_addrs("9000", 8788), Some(vec![local(9000)]));
    }

    /// The new `HOST:PORT` spelling binds the named interface, not loopback.
    #[test]
    fn parse_listen_addrs_accepts_host_colon_port() {
        assert_eq!(parse_listen_addrs("0.0.0.0:8788", 8788), Some(vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 8788)]));
        assert_eq!(parse_listen_addrs("192.168.1.5:9000", 8788), Some(vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 5)), 9000)]));
    }

    /// A bare host with no port takes the dialect's own default port.
    #[test]
    fn parse_listen_addrs_accepts_a_bare_host_using_the_default_port() {
        assert_eq!(parse_listen_addrs("0.0.0.0", 8788), Some(vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 8788)]));
    }

    /// A comma-separated list binds every interface named, in order - the
    /// "OR more than one interface" case.
    #[test]
    fn parse_listen_addrs_accepts_a_comma_separated_list() {
        let got = parse_listen_addrs("0.0.0.0:8788,127.0.0.1:9000", 8788).unwrap();
        assert_eq!(got, vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 8788), local(9000)]);
    }

    /// Garbage, or an empty item in the list, must not parse at all - the
    /// caller (`take_addrs`) is what decides what "not a valid list" means
    /// (leave the token alone, fall back to the default), not this function
    /// guessing a partial answer.
    #[test]
    fn parse_listen_addrs_rejects_garbage_and_empty_items() {
        assert_eq!(parse_listen_addrs("not-an-address", 8788), None);
        assert_eq!(parse_listen_addrs("0.0.0.0:8788,", 8788), None);
        assert_eq!(parse_listen_addrs("", 8788), None);
    }

    /// `take_addrs` consumes the next token and uses it verbatim when it
    /// parses as an address list.
    #[test]
    fn take_addrs_consumes_a_valid_address_list() {
        let args = vec!["--openai".to_string(), "0.0.0.0:9000,127.0.0.1:9001".to_string()];
        let mut i = 0;
        let got = take_addrs(&args, &mut i, 8788);
        assert_eq!(got, vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)), 9000), local(9001)]);
        assert_eq!(i, 1, "the address token must be consumed");
    }

    /// With no value (the next token is unrelated, e.g. the next flag),
    /// `take_addrs` must NOT consume it and must fall back to the single
    /// loopback default on the dialect's own port - same contract the old
    /// bare-port `take_port` closure had.
    #[test]
    fn take_addrs_falls_back_to_loopback_default_without_consuming_an_unrelated_next_token() {
        let args = vec!["--openai".to_string(), "--dbus".to_string()];
        let mut i = 0;
        let got = take_addrs(&args, &mut i, 8788);
        assert_eq!(got, vec![local(8788)]);
        assert_eq!(i, 0, "an unrelated next token must not be consumed");
    }

    /// End of args (the flag is the last token) must behave exactly like an
    /// unrelated next token: the default, nothing consumed.
    #[test]
    fn take_addrs_falls_back_to_loopback_default_at_end_of_args() {
        let args = vec!["--openai".to_string()];
        let mut i = 0;
        let got = take_addrs(&args, &mut i, 8788);
        assert_eq!(got, vec![local(8788)]);
        assert_eq!(i, 0);
    }
}
