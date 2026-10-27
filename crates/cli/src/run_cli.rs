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
use std::path::{Path, PathBuf};
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
  brain serve                             # the OpenAI API on 127.0.0.1:8788
  brain serve [SURFACE ...] [options]     # one or more serving surfaces
  brain serve --stdio                     # the event-driven stdio JSONL loop

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
  `APIKEY <provider> <key>`; --api-keys-out writes the keys as JSON, 0600, and the
  lines then name that file instead of repeating the keys.
  --api-key-file FILE pins a fixed key instead (the file's one word) - every
  dialect then shares that SAME key rather than each getting its own random one.

  LOCAL CLIENTS need none of this: once every surface is bound, the server
  writes <runtime dir>/serve-<pid>.json (mode 0600) and removes it on exit:
      {\"pid\": N, \"surfaces\": [{\"provider\": \"openai\",
                               \"url\": \"http://127.0.0.1:8788\", \"key\": \"sk-brain-...\"}]}
  one entry per bound address, `url` the port actually bound (never a `:0`).

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
                         started). Prints the pid; output goes to the log below,
                         and is echoed here while the wait lasts - a cold start
                         scans the model directory and activates a checkpoint
                         before anything binds, which is minutes, not seconds.
                         A requested surface that cannot bind ends the process
                         rather than serving the rest, since readiness is the
                         AND of all of them and would otherwise never arrive.
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
  the runtime dir: serve.pid, serve.log, (when -d is used without --ready-file)
  serve.ready, and every server's connection record.
  --runtime-dir DIR      the runtime dir (default $XDG_RUNTIME_DIR/brain, else
                         <tmpdir>/brain-<uid>). A server run under its own DIR
                         is invisible to clients looking in the default one.

SERVING OPTIONS
  --models-dir DIR       the model store every served model is resolved from
                         (else $BRAIN_MODELS_DIR, else $XDG_DATA_HOME/brain/models)
  --api-keys-out FILE    write {\"openai\":\"sk-brain-…\", …} as JSON, mode 0600
  --reserve-gb N         GB of VRAM kept free per GPU for activations (default 2)
  --admit-deadline-ms N  how long a request may wait to START (be claimed onto a
                         lane) before it is shed with HTTP 429 (default 900000)
  --cold-build-admit-deadline-ms N
                         the longer wait that applies instead while the request's
                         own model is still cold-building (default 1800000)
  --watch-adapters DIR   watch DIR for promoted LoRA adapters and offer the
                         newest as a model of its own, with no restart. OFF
                         unless given. The newest is the highest-versioned
                         adapter-NNNNNN.safetensors, never the newest mtime.
                         The previous release is withdrawn once the new one is
                         offered; a request already running is never
                         interrupted, it finishes on the adapter it started on.
  --adapter FILE         offer this LoRA adapter release as a model of its own,
                         never replaced. It must verify against the served base
                         (the base digest its card records, else its base id)
                         or the server does not start. Prints
                         \"brain serve: <model> adapter=<id> digest=sha256:<hex>\".
  --adapter-manifest FILE
                         follow a release manifest,
                         {\"adapter\": \"<path>\", \"digest\": \"sha256:<hex>\"}
                         (a relative path is relative to FILE), re-read when it
                         changes: a new release is offered only once the file
                         hashes to that digest and verifies against the base,
                         then the previous one is withdrawn;
                         a bad manifest keeps the previous adapter and says why.
                         Replace FILE atomically (write aside, rename over it)
                         and never rewrite a published adapter file.
                         --adapter, --adapter-manifest and --watch-adapters are
                         mutually exclusive.

ADAPTERS  (one resident copy of the Qwen3 weights serves the base and every
           adapter at once, in the same batches. An adapter is a model: it is
           selected by its card id, <vendor>/<repo>:<owner>:<name>:<tag>, which
           GET /v1/models lists with its {id, digest}. The base keeps its own
           id and answers with no adapter. A model directory holding adapters
           for a served Qwen3 offers them the same way. Adapters are loaded
           by the operator only: no request can load, name a file for, or
           fetch an adapter. An id that is not offered is an unknown model.)
  --adapter-slots N      adapters resident on the device at once (default 4).
                         More adapters than slots are offered by swapping the
                         least recently used idle one out. Giving this, or
                         either flag below, or any adapter flag above, builds
                         the adapter bank; without them none is built and the
                         bank costs nothing.
  --adapter-rank-cap R   highest LoRA rank a slot holds (default 16). An
                         adapter of a higher rank is refused.
  --adapter-targets LIST linears the bank corrects: attention (wq,wk,wv,wo
                         on Qwen3, the default), all (every adapter site of
                         the model), or a comma-separated list of the model's
                         site names (Qwen3: wq, wk, wv, wo, gate, up, down).
                         An adapter that corrects any other linear is refused. The bank takes
                         slots x rank-cap x (k + n) x 4 bytes per targeted
                         linear per layer, counted in the model's footprint.
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

SERVING POLICY  (one model-agnostic policy for every model that serves it.
                 Each model declares what it supports; a choice it cannot
                 honour is refused naming the model and the supported set -
                 never a silent fallback. Everything not given here is derived
                 from the model's own metadata, the device and the memory plan,
                 and the values chosen are reported at startup.)
  --checkpoint PATH      serve this LLM checkpoint: the file or the repo
                         directory holding it (a tokenizer.json beside a
                         directory is picked up). The target of --adapter,
                         --adapter-manifest and --watch-adapters.
  --tokenizer PATH       the tokenizer.json of a --checkpoint that has none
                         beside it.
  --weight-format SPEC   weight storage format: f32|bf16|f16|int8|q4|nf4, or per
                         leaf `int8,blk.0.attn_q=f32` where the model supports
                         rules. Default: the model's own (a checkpoint that does
                         not fit its card in fp32 is served int8).
  --kv-cache TIER        KV-cache storage: f32|bf16|int8|fp8. Default: the
                         model's own (int8 for the Qwen3 engine; for the
                         released Qwen3.8 GGUFs the fastest that fits the
                         cards, int8 first).
  --gemm auto|fp32       `fp32` pins exact fp32 arithmetic for the linears (the
                         reference tier); `auto` (default) uses tensor cores
                         where the device has them.
  --kv-calibration       opt in to a kv_calib.json beside the checkpoint.
  --ctx N                cap the context window below the model's declared one
                         (default: the declared window, shrunk to what the
                         device holds).
  --max-batch N          concurrent sequences the server must carry (default:
                         the model's own; a configuration whose slots x ctx do
                         not fit the GPUs is refused at startup).
  --kv-offload-gb N      host RAM granted for preempted sessions' KV, fractional
                         allowed (default 0, off).
  --speculative MODE     speculative decoding for the Qwen3.8 GGUF models:
                         auto|off|mtp|dflash|ngram (default auto = the best
                         drafter the checkpoint offers, off when it has none;
                         off where the cards cannot run the CUDA verify
                         kernels or its memory would squeeze the context).
                         The scheduler verifies each sequence's proposals in
                         one batched step and keeps the longest prefix the
                         model agrees with, so greedy output is unchanged; it
                         falls back to plain decoding when the batch is large.
  --exact-decode         make a decode row's result independent of which other
                         requests share its batch: the int8 projections and the
                         head run in groups small enough to stay on the
                         bit-identical kernel, and every row's attention is
                         sized from --ctx, so an identical request returns
                         identical tokens and per-token log-probabilities alone
                         or beside any batch of neighbours. Costs one extra
                         pass over the weights per group (two at --max-batch 8
                         on int8 tensor cores) and turns speculation off.
                         Default off: today's batched numerics unchanged.
  Not flags any more (derived, reported at startup): the KV pool size, the
  prefix-cache slots, the prefill slice, batch and row cap, the norm fusion.
  Deprecated spellings (--qwen-*, --qwen35-*) still work, with a warning, until
  their callers have moved.

DECISION MODELS  (decide + Laya; every servable one in the model store is
                 served under its store id, e.g. convaiinnovations/laya)
  --decision-model REF|DIR  also serve this model, whatever state its files
                         are in (a missing one answers `not-installed`
                         instead of vanishing). Repeatable.
  --decision-default ID  the model `auto` resolves to, validated or not.
                         Without it `auto` serves a validated Laya model, else
                         a validated decide model, else fails with the reason.

STDIO CONTROLLER  (--stdio)
  --gpt PATH             GPT checkpoint (else $BRAIN_GPT2; else a fake echo model)
  --yolo PATH             YOLO checkpoint (else $BRAIN_YOLOV8; else a fake detector)
  --conf X                detection confidence threshold (else $BRAIN_CONF, 0.25)
  --max-new N  --temp X  --top-k K  --seed S      generation config
  Reads JSONL events on stdin, writes JSONL events on stdout, one per line.
  Example: printf '{\"event\":\"user_text\",\"text\":\"hi\"}\\n' | brain serve --stdio

MODEL CONFIGURATION
  Which models this server serves: every model in the model store, each
  under its store id, the --checkpoint, plus those some families still take
  from BRAIN_* env vars (BRAIN_LFM2, BRAIN_NEMOTRONASR, ...). A family with
  nothing on this machine is counted in one startup line; -v lists each. Tuning an already-selected
  model (context length, batching, precision, ...) is a flag where one
  exists - see QWEN3 / QWEN3.8 SERVING above. Run
  `brain serve --help` for the full reference table of every serving
  variable.

  FETCHING: a request for a model that is not pulled errors with zero network
  I/O unless --autofetch was passed; `brain pull`
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
  brain serve                                # OpenAI API on http://127.0.0.1:8788
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
/// A policy value or flag that does not parse: say so with the usage and stop.
fn policy_error(e: &str) -> ! {
    eprintln!("brain serve: {e}\n");
    eprint!("{HELP}");
    std::process::exit(2);
}

fn policy_flag(policy: &mut model::policy::ServePolicy, flag: &str, value: &str) {
    if let Err(e) = policy.set(flag, value) {
        policy_error(&e);
    }
}

/// Hand the model-agnostic policy to every serving family. A family that cannot
/// honour a precision choice refuses at build time, naming itself; the server
/// refuses to start only when NO family can honour it.
fn apply_policy(policy: &model::policy::ServePolicy, qwen: &mut catalog::resident_llm::QwenServeConfig, qwen35: &mut catalog::resident_qwen35::Qwen35ServeConfig, ocr: &mut ocrserve::OcrServeConfig) {
    if *policy == model::policy::ServePolicy::default() {
        return;
    }
    eprintln!("brain serve: serving policy: {}", policy.describe());
    match (qwen.apply_policy(policy), qwen35.apply_policy(policy), ocr.apply_policy(policy)) {
        (Err(a), Err(b), Err(c)) => policy_error(&format!("no served model family can honour this policy: {a}; {b}; {c}")),
        (a, b, c) => {
            for r in [a, b, c] {
                match r {
                    Ok(notes) => notes.iter().for_each(|n| eprintln!("brain serve: note: {n}")),
                    Err(why) => eprintln!("brain serve: note: {why} (that family will not serve)"),
                }
            }
        }
    }
}

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

/// The value that must follow a milliseconds flag, as a [`Duration`]. Zero is
/// rejected along with the unparseable: a zero wait would shed every request
/// that is not admitted instantly, and "no limit" is not offered.
fn positive_ms(args: &[String], i: &mut usize, flag: &str) -> std::time::Duration {
    let v = val(args, i, flag);
    match v.parse::<u64>() {
        Ok(ms) if ms > 0 => std::time::Duration::from_millis(ms),
        _ => {
            eprintln!("brain serve: {flag} {v:?} is not a positive number of milliseconds\n");
            eprint!("{HELP}");
            std::process::exit(2);
        }
    }
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
/// (`catalog::resident_tts::TtsResident::from_env`) already reads - one spelling
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

/// The OpenAI dialect's port, the surface `brain serve` binds when it is
/// asked for none.
const DEFAULT_OPENAI_PORT: u16 = 8788;

/// What a `brain serve` command line serves.
#[derive(Debug, PartialEq, Eq)]
enum SurfaceChoice {
    /// `--stdio`: the event-driven controller on stdin/stdout.
    Stdio,
    /// The HTTP/D-Bus surfaces the flags name.
    Named,
    /// None named: the OpenAI dialect on 127.0.0.1, the one a local client
    /// expects with no configuration.
    DefaultOpenAi,
}

/// Decide [`SurfaceChoice`]. `--stdio` and a surface flag are two different
/// servers, so naming both is refused, as is a stdio-only flag (`--gpt`,
/// `--yolo`, `--conf`) without `--stdio`, which would otherwise be ignored.
fn surface_choice(stdio: bool, any_surface: bool, stdio_only: Option<&str>) -> Result<SurfaceChoice, String> {
    match (stdio, any_surface, stdio_only) {
        (true, true, _) => Err("--stdio serves stdin/stdout; it cannot be combined with --openai, --anthropic, --openrouter or --dbus".to_string()),
        (true, false, _) => Ok(SurfaceChoice::Stdio),
        (false, _, Some(flag)) => Err(format!("{flag} applies to the stdio controller; add --stdio")),
        (false, true, None) => Ok(SurfaceChoice::Named),
        (false, false, None) => Ok(SurfaceChoice::DefaultOpenAi),
    }
}

/// The connection record this process publishes in its runtime directory.
fn serve_daemon_record(runtime_dir: Option<&Path>) -> PathBuf {
    crate::serve_daemon::record_path(&crate::serve_daemon::state_dir(runtime_dir))
}

pub fn run_serve(args: &[String]) {
    let (seed, rest) = crate::args::split_seed(args, "brain serve");
    let args = &rest[..];
    let mut gpt_path = std::env::var("BRAIN_GPT2").ok();
    // `--stdio` selects the event-driven controller; every other run serves
    // HTTP and/or D-Bus. The first stdio-only flag given is kept so a run that
    // names one without `--stdio` is refused instead of ignoring it.
    let mut stdio = false;
    let mut stdio_only: Option<&'static str> = None;
    let mut yolo_path = std::env::var("BRAIN_YOLOV8").ok();
    let mut cfg = GenConfig { max_new: 256, temperature: 0.0, top_k: 0, eos: None, seed };
    // Optional detection confidence threshold for the YOLO detector. A tiny model
    // trained for only a few hundred steps emits low-confidence boxes that the
    // default 0.25 filter would drop, so the demo can lower it (also `BRAIN_CONF`).
    let mut conf: Option<f32> =
        std::env::var("BRAIN_CONF").ok().and_then(|s| s.parse().ok());
    // D-Bus control surface (`--dbus [--dbus-system] [--dbus-name NAME]`).
    let (mut dbus, mut dbus_system, mut dbus_name) = (false, false, None::<String>);
    let mut dbus_reserve_gb: u64 = 2; // GB kept free per GPU (headroom for activations)
    // The one model store everything served reads: the startup catalog scan
    // and every architecture resolver (`--models-dir`, else BRAIN_MODELS_DIR /
    // XDG default; see model_dir::resolve).
    let mut models_dir: Option<String> = None;
    // Opt-in continuous-learning hot swap (`--watch-adapters DIR`): watch DIR
    // for a promoted LoRA adapter and point the served Qwen3 at it without a
    // restart. `None` (the default) spawns no watcher at all -- a server that
    // silently reloads its weights because a file appeared on disk is not
    // something an operator should get without asking for it.
    let mut watch_adapters: Option<String> = None;
    // `--adapter FILE` / `--adapter-manifest FILE`: an explicit release, pinned
    // or followed through a manifest (see `continuous_train::AdapterMode`).
    let mut adapter: Option<String> = None;
    let mut adapter_manifest: Option<String> = None;
    // `--adapter-slots` / `--adapter-rank-cap` / `--adapter-targets`: the size
    // of the adapter bank. Built only when one of them, or any adapter source
    // above, asks for adapters.
    let mut adapter_bank = catalog::adapter_registry::AdapterBankSettings::default();
    let mut adapter_bank_asked = false;
    // Every serving-policy flag, built up as its own arm below is matched -
    // `catalog::resident_llm::QwenServeConfig`'s own `Default` is every
    // historical env-var default, byte for byte.
    let mut qwen_cfg = catalog::resident_llm::QwenServeConfig::default();
    // Every Qwen3.8 serving-policy flag, same story: `Qwen35ServeConfig`'s `Default`
    // is every historical env-var default. `fp32` sizes the brain-checkpoint
    // resident, `gguf` holds the defaults of every released-GGUF multi-card
    // resident - two different models, so the flags are name-spaced apart.
    let mut qwen35_cfg = catalog::resident_qwen35::Qwen35ServeConfig::default();
    let mut policy = model::policy::ServePolicy::default();
    let mut ocr_cfg = ocrserve::OcrServeConfig::default();
    // `--decision-model` / `--decision-default`: the decision models served
    // beyond those the model store holds, and the one `auto` resolves to.
    let mut decision_cfg = catalog::resident_decision::DecisionServeConfig::default();
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
    // `--api-key-file FILE`: one fixed key every dialect shares, instead of a
    // fresh one per dialect per launch.
    let mut fixed_key: Option<String> = None;
    // `--runtime-dir DIR`: where the lifecycle files and the connection
    // record live (see `serve_daemon::state_dir`); `main` reads it too, for
    // `-d`/`--status`/`--stop`.
    let mut runtime_dir: Option<PathBuf> = None;
    let mut ready_file: Option<String> = None;
    // `--admit-deadline-ms` / `--cold-build-admit-deadline-ms`: how long a
    // request may wait to start before it is shed with a 429.
    let mut deadlines = residency::admission::Deadlines::default();
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
            "--gpt" => {
                gpt_path = Some(val(args, &mut i, "--gpt"));
                stdio_only.get_or_insert("--gpt");
            }
            "--yolo" => {
                yolo_path = Some(val(args, &mut i, "--yolo"));
                stdio_only.get_or_insert("--yolo");
            }
            "--max-new" => cfg.max_new = parsed(args, &mut i, "--max-new"),
            "--temp" | "--temperature" => {
                let flag = args[i].clone();
                cfg.temperature = parsed(args, &mut i, &flag);
            }
            "--top-k" => cfg.top_k = parsed(args, &mut i, "--top-k"),
            "--conf" => {
                conf = Some(parsed(args, &mut i, "--conf"));
                stdio_only.get_or_insert("--conf");
            }
            "--stdio" => stdio = true,
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
            "--adapter" => adapter = Some(val(args, &mut i, "--adapter")),
            "--adapter-manifest" => adapter_manifest = Some(val(args, &mut i, "--adapter-manifest")),
            "--adapter-slots" => {
                adapter_bank.slots = parsed(args, &mut i, "--adapter-slots");
                adapter_bank_asked = true;
            }
            "--adapter-rank-cap" => {
                adapter_bank.rank_cap = parsed(args, &mut i, "--adapter-rank-cap");
                adapter_bank_asked = true;
            }
            "--adapter-targets" => {
                let list = val(args, &mut i, "--adapter-targets");
                adapter_bank.targets = catalog::adapter_registry::AdapterTargets::parse(&list).unwrap_or_else(|e| {
                    eprintln!("brain serve: --adapter-targets {list:?}: {e}\n");
                    eprint!("{HELP}");
                    std::process::exit(2);
                });
                adapter_bank_asked = true;
                qwen_cfg.adapter_targets_named = true;
            }
            "--checkpoint" => qwen_cfg.weights = Some(val(args, &mut i, "--checkpoint")),
            f if model::policy::VALUE_FLAGS.contains(&f) => {
                let value = val(args, &mut i, f);
                policy_flag(&mut policy, f, &value);
            }
            f if model::policy::SWITCH_FLAGS.contains(&f) => {
                if let Err(e) = policy.switch(f) {
                    policy_error(&e);
                }
            }
            f if model::policy::rewrite_legacy(f).is_some() => {
                let r = model::policy::rewrite_legacy(f).expect("matched above");
                eprintln!("brain serve: {}", model::policy::deprecation_warning(f, r));
                use model::policy::Replacement as R;
                match r {
                    R::Renamed(to) => {
                        let value = val(args, &mut i, f);
                        if to == "--checkpoint" {
                            qwen_cfg.weights = Some(value);
                        } else {
                            policy_flag(&mut policy, to, &value);
                        }
                    }
                    R::Fixed(to, value) => policy_flag(&mut policy, to, value),
                    R::Switch(to) | R::DerivedSwitch(to) => {
                        if let Err(e) = policy.switch(to) {
                            policy_error(&e);
                        }
                    }
                    R::Derived(what) => {
                        let value = val(args, &mut i, f);
                        policy_flag(&mut policy, what, &value);
                    }
                }
            }
            "--decision-model" => decision_cfg.models.push(val(args, &mut i, "--decision-model")),
            "--decision-default" => decision_cfg.default = Some(val(args, &mut i, "--decision-default")),
            "--speculative" => {
                let mode = val(args, &mut i, "--speculative");
                qwen35_cfg.gguf.speculative = Some(qwen35::int8_gguf_resident::SpeculativeMode::parse(&mode).unwrap_or_else(|e| {
                    eprintln!("brain serve: --speculative {mode:?}: {e}\n");
                    eprint!("{HELP}");
                    std::process::exit(2);
                }));
            }
            "--anthropic" => anthropic = Some(take_addrs(args, &mut i, 8787)),
            "--openai" => openai = Some(take_addrs(args, &mut i, 8788)),
            "--openrouter" => openrouter = Some(take_addrs(args, &mut i, 8789)),
            "--api-keys-out" => api_keys_out = Some(val(args, &mut i, "--api-keys-out")),
            "--api-key-file" => {
                let file = val(args, &mut i, "--api-key-file");
                fixed_key = Some(apiserve::surface::read_key_file(Path::new(&file)).unwrap_or_else(|e| {
                    eprintln!("brain serve: --api-key-file {e}");
                    std::process::exit(2);
                }));
            }
            "--runtime-dir" => runtime_dir = Some(PathBuf::from(val(args, &mut i, "--runtime-dir"))),
            "--tokenizer" => qwen_cfg.tokenizer = Some(val(args, &mut i, "--tokenizer")),
            "--ready-file" => ready_file = Some(val(args, &mut i, "--ready-file")),
            "--admit-deadline-ms" => deadlines.admit = positive_ms(args, &mut i, "--admit-deadline-ms"),
            "--cold-build-admit-deadline-ms" => deadlines.cold_build = positive_ms(args, &mut i, "--cold-build-admit-deadline-ms"),
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

    let any_surface = dbus || anthropic.is_some() || openai.is_some() || openrouter.is_some();
    match surface_choice(stdio, any_surface, stdio_only) {
        Ok(SurfaceChoice::Stdio | SurfaceChoice::Named) => {}
        Ok(SurfaceChoice::DefaultOpenAi) => openai = Some(vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), DEFAULT_OPENAI_PORT)]),
        Err(e) => {
            eprintln!("brain serve: {e}\n");
            eprint!("{HELP}");
            std::process::exit(2);
        }
    }

    let adapter_mode = match crate::continuous_train::AdapterMode::from_flags(adapter.map(PathBuf::from), adapter_manifest.map(PathBuf::from), watch_adapters.map(PathBuf::from)) {
        Ok(mode) => mode,
        Err(e) => {
            eprintln!("brain serve: {e}");
            std::process::exit(2);
        }
    };

    // The adapter bank exists when adapters were asked for, by a source or by
    // a size, and not otherwise.
    if adapter_bank_asked || adapter_mode != crate::continuous_train::AdapterMode::Base {
        if let Err(e) = adapter_bank.validate() {
            eprintln!("brain serve: {e}");
            std::process::exit(2);
        }
        qwen_cfg.adapter_bank = Some(adapter_bank);
    }
    apply_policy(&policy, &mut qwen_cfg, &mut qwen35_cfg, &mut ocr_cfg);

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

    // Any surface replaces the stdio loop: it serves every registered model
    // over HTTP and/or `com.swedishembedded.Brain1` until Ctrl-C.
    if !stdio {
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
            fixed_key,
            record: serve_daemon_record(runtime_dir.as_deref()),
            ready,
            deadlines,
            adapter_mode,
            knobs: serving::ServeKnobs { qwen: qwen_cfg, qwen35: qwen35_cfg, decision: decision_cfg, ocr: ocr_cfg },
        });
    }

    // The stdio controller serves no Qwen3, so an adapter flag here would be
    // ignored while the operator believes it is in effect.
    if adapter_mode != crate::continuous_train::AdapterMode::Base {
        eprintln!("brain serve: --adapter, --adapter-manifest and --watch-adapters apply to the Qwen3 the HTTP and D-Bus surfaces serve; the stdio controller serves none. Add --openai, --anthropic, --openrouter or --dbus.");
        std::process::exit(2);
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
    // A capability whose weights are absent is left out, not a failed start:
    // the controller serves whatever this machine holds.
    match s3dit::caps::ZImageProvider::load() {
        Ok(provider) => ctrl.register_provider(std::sync::Arc::new(provider)),
        Err(e) => residency::log::not_served(&format!("brain: z-image not served ({e})")),
    }
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

/// Discover schedulable compute (GPUs/NPUs/CPU RAM, narrowed by `--device`) and build
/// the one shared residency executor that every serving surface (D-Bus + the HTTP APIs)
/// drives, with every model resolved from `models_dir` (already resolved by `run_apis`).
///
/// Returns `serving::Serving`, not a bare `Executor`: the
/// continuous-learning hot swap needs the CONCRETE `QwenResident` handle
/// alongside the type-erased one the executor holds - see that type's doc.
fn build_serving_executor(reserve_gb: u64, models_dir: Option<&Path>, knobs: serving::ServeKnobs) -> serving::Serving {
    let machine = serving::Machine::probe(crate::compute_set());
    let reserved = reserve_gb << 30;
    eprintln!(
        "brain serve: compute {} | {} GPU(s), {} NPU(s) schedulable, {} GB reserved/card, {} GB RAM budget",
        machine.compute,
        machine.gpus.len(),
        machine.npus.len(),
        reserve_gb,
        machine.pool_ram >> 30
    );
    // `models_dir` is the ONE store every resolver below reads (see
    // `run_apis`); the catalog scan appends every carded file in it as its
    // own catalog entry.
    if let Some(d) = models_dir {
        // First run on a fresh install: the dir doesn't exist yet. Create it so
        // the scan is clean (an empty catalog, not an ENOENT warning) - models
        // dropped in later are picked up on the next `brain serve` with no env
        // vars needed.
        if let Err(e) = std::fs::create_dir_all(d) {
            eprintln!("brain serve: could not create model dir {} ({e}); scan may be empty", d.display());
        }
        eprintln!("brain serve: scanning model dir {}", d.display());
    }
    serving::build_executor(&machine, reserved, models_dir, residency::Policy::from_env(), knobs)
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
    /// `--api-key-file`'s key, shared by every dialect; `None` draws one per
    /// dialect.
    fixed_key: Option<String>,
    /// Where the connection record is published once every HTTP surface has
    /// bound (see `apiserve::surface::record_json`).
    record: PathBuf,
    /// Notified once per bound surface (HTTP + D-Bus); disabled unless
    /// `--ready-file` was given. See `brain_shutdown::ready::Gate`.
    ready: brain_shutdown::ready::Gate,
    /// `--admit-deadline-ms` and `--cold-build-admit-deadline-ms`, handed to
    /// every transport.
    deadlines: residency::admission::Deadlines,
    /// Where the served Qwen3's adapter comes from: `--adapter`,
    /// `--adapter-manifest`, `--watch-adapters`, or none of them.
    adapter_mode: crate::continuous_train::AdapterMode,
    /// Every per-model serving flag (the deprecated aliases,
    /// `--decision-*`), parsed once here - see `serving::ServeKnobs`.
    knobs: serving::ServeKnobs,
}

/// Build the one shared executor and bring up the requested surfaces: D-Bus
/// (`com.swedishembedded.Brain1`) and/or the HTTP inference APIs (Anthropic / OpenAI /
/// OpenRouter), each on its own localhost port with a key generated at startup. When
/// D-Bus and an HTTP surface both run, D-Bus gets its own thread (it owns a tokio
/// runtime) and the HTTP servers own the main thread; a single surface blocks directly.
/// A `brain_dbus::serve` failure is almost always "no bus at this address" (no
/// desktop session, no `dbus-run-session`, no system bus policy for this user) -
/// give a message that says what to try instead of the raw connect errno.
fn dbus_connect_hint(err: &dyn std::fmt::Display, system_bus: bool) -> String {
    let advice = if system_bus {
        "Check the system bus is running and this user has a policy file for \
         the requested bus name, or drop --dbus-system for the per-user session bus."
    } else {
        "Run under `dbus-run-session -- <cmd>`, start a desktop session, or pass \
         --dbus-system if a system bus policy is installed for this service."
    };
    let kind = if system_bus { "system" } else { "session" };
    // What this means for the process is NOT said here: it depends on whether
    // the surface had already bound, which only the caller knows. Each call
    // site prints its own consequence, so the two can never contradict.
    format!("brain serve --dbus: could not connect to the D-Bus {kind} bus ({err}). {advice}")
}

/// How long [`run_apis`] waits for a backgrounded D-Bus surface to finish its own
/// graceful shutdown after the HTTP surface has drained, before giving up and
/// letting the process exit anyway. Bounded so a wedged D-Bus shutdown cannot
/// hang `brain serve` forever - [`brain_shutdown::install_signals`]'s own
/// second-signal escape hatch is the backstop if this window is not enough.
const DBUS_JOIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

/// How long the surfaces get to drain after the first SIGINT/SIGTERM before the
/// process ends anyway: a signalled server must never stay up holding its GPU
/// reservation because one connection or request will not finish.
const SHUTDOWN_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

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
fn build_auto_fetch_supplier(models_dir: Option<&Path>) -> Option<Arc<dyn residency::ModelSupplier>> {
    if !crate::supply::auto_fetch_enabled() {
        eprintln!("brain serve: auto-fetch off -- a request for a model that is not pulled errors (pass --autofetch to enable on-demand fetching)");
        return None;
    }
    // The SAME models directory `build_serving_executor`'s startup scan
    // reads, so a freshly auto-fetched model lands exactly where a restart's
    // scan would find it again.
    let store = brain_modelstore::Store::new(models_dir?.to_path_buf());
    let hub: Box<dyn brain_modelstore::Hub> = Box::new(brain_modelstore::HfHub::new());
    Some(Arc::new(crate::supply::StoreSupplier::new(store, hub)))
}

/// The startup line naming what is served. An empty list is said outright
/// and pointed at `brain doctor`: a server listening with nothing behind it
/// answers every request with an error, and the empty `models:` it used to
/// print looked like any other startup.
fn served_models_line(served: &[&str], models_dir: Option<&Path>) -> String {
    if !served.is_empty() {
        return format!("brain serve: models: {}", served.join(", "));
    }
    let store = models_dir.map_or_else(|| "no model store is configured".to_string(), |d| format!("the model store {}", d.display()));
    format!(
        "brain serve: no model is being served ({store} yielded none), so /v1/models is empty and every request will fail; run `brain doctor` to see why"
    )
}

/// The one startup line that stands in for every model family left out of
/// the catalog: a machine holds few of the families brain can serve, so at
/// the default verbosity they are counted, not listed. With `-v` each was
/// already printed with its reason, and this line would only repeat them.
fn unserved_line(count: usize, verbosity: u8) -> Option<String> {
    (count > 0 && verbosity == 0).then(|| format!("brain serve: {count} model families have nothing to serve on this machine (-v lists each and why)"))
}

fn run_apis(a: RunApis) {
    // Resolved ONCE (`--models-dir`, else `BRAIN_MODELS_DIR`/XDG/HOME - see
    // `loader::model_dir::resolve`) and handed to everything that reads the
    // store - the startup scan, every architecture resolver, the auto-fetch
    // supplier and the healer - so the flag decides the store everywhere.
    let models_dir = loader::model_dir::resolve(a.models_dir.as_deref());
    let supplier = build_auto_fetch_supplier(models_dir.as_deref());
    let serving::Serving { executor, qwen, qwen35 } = build_serving_executor(a.reserve_gb, models_dir.as_deref(), a.knobs);
    let manifests = executor.manifests();
    let served: Vec<&str> = manifests.iter().map(|m| m.model.as_str()).collect();
    eprintln!("{}", served_models_line(&served, models_dir.as_deref()));
    if let Some(line) = unserved_line(residency::log::not_served_count(), residency::log::verbosity()) {
        eprintln!("{line}");
    }

    // `--autofetch` also proactively heals what the scan above already found
    // broken (an interrupted download, a GGUF still needing its one-time
    // conversion) in the background, instead of only fixing it the next time
    // a client happens to request that exact model. `supplier` is already
    // `None` unless auto-fetch is enabled, so no separate check is needed here.
    if let (Some(sup), Some(dir)) = (&supplier, &models_dir) {
        crate::supply::heal_missing_models_in_background(dir.clone(), sup.clone(), executor.clone());
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

    // The served adapter, put in effect before any surface binds: a release
    // that does not verify ends the process here rather than serving a model
    // nobody asked for. Then the continuous-learning hot swap, opt-in
    // (`--watch-adapters DIR` / `--adapter-manifest FILE`). Held for the whole
    // serving lifetime: the handle stops and joins its thread on drop, so the
    // watcher cannot outlive the surfaces it was swapping models under. `qwen`
    // is the CONCRETE resident handle -- `adapter_view` is inherent, so the
    // erased one the executor holds could not do this.
    let host = crate::continuous_train::Host::pick(qwen, &qwen35);
    let follow = match host.start(a.adapter_mode, &executor) {
        Ok((follow, serving_line)) => {
            if let Some(line) = serving_line {
                eprintln!("{line}");
            }
            follow
        }
        Err(e) => {
            eprintln!("brain serve: {e}");
            std::process::exit(1);
        }
    };
    let _adapter_watcher = host.watch(follow, &executor);

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
            let ready_state = a.ready.clone();
            Some(std::thread::spawn(move || {
                let serve_opts = brain_dbus::ServeOpts::new().with_shutdown(sd).with_supplier(sup).with_ready(ready).with_deadlines(a.deadlines);
                let Err(err) = brain_dbus::serve(e, opts, serve_opts) else { return };
                eprintln!("{}", dbus_connect_hint(&err, dbus_system));
                // Before ready, this is a FAILED start and has to end the
                // process. Readiness is the AND of every requested surface
                // (see `brain_shutdown::ready::Gate`), so a process missing one
                // can never signal ready; staying up to serve the rest leaves
                // anything waiting on it -- `serve -d`, a test harness, a unit
                // file -- to sit out its whole timeout against a server that
                // will never arrive. Exiting makes the failure the answer.
                //
                // After ready it is a surface LOST by a server that is
                // otherwise working, and taking the rest down with it would be
                // the more destructive of the two mistakes.
                if ready_state.is_ready() {
                    eprintln!("brain serve: the D-Bus surface stopped; the remaining surfaces keep serving");
                    return;
                }
                eprintln!("brain serve: the D-Bus surface was requested but did not start -- exiting");
                std::process::exit(1);
            }))
        } else {
            let serve_opts = brain_dbus::ServeOpts::new().with_shutdown(shutdown).with_supplier(sup).with_ready(a.ready.clone()).with_deadlines(a.deadlines);
            if let Err(err) = brain_dbus::serve(e, opts, serve_opts) {
                eprintln!("{}", dbus_connect_hint(&err, a.dbus_system));
                eprintln!("brain serve: the D-Bus surface was the only one requested -- exiting");
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
                let key = apiserve::surface::resolved_key(a.fixed_key.as_deref());
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
        // The keys file first: the APIKEY lines then name it instead of
        // repeating the keys (stdout lands in logs other users of a shared
        // node can read). A file that could not be written leaves the lines
        // carrying the keys, the only place they remain.
        let keys_file = a.api_keys_out.as_deref().map(std::path::Path::new).filter(|path| match apiserve::write_keys(&surfaces, path) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("brain serve: --api-keys-out {}: {e}", path.display());
                false
            }
        });
        // One APIKEY line per DIALECT (never per address): every surface in
        // a dialect's group carries the identical key built above, so a
        // second, third, ... address would only reprint the same line.
        let mut announced: Vec<apiserve::Provider> = Vec::new();
        for s in &surfaces {
            if !announced.contains(&s.provider) {
                s.announce(keys_file);
                announced.push(s.provider);
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
        brain_shutdown::bound_drain(shutdown.clone(), SHUTDOWN_DRAIN_TIMEOUT, || {
            eprintln!("brain serve: the surfaces did not drain within {}s of the shutdown signal; exiting", SHUTDOWN_DRAIN_TIMEOUT.as_secs());
            brain_shutdown::exit_now(0);
        });
        let opts = apiserve::ServeOpts::new().with_shutdown(shutdown).with_supplier(supplier).with_ready(a.ready).with_deadlines(a.deadlines).with_record(Some(a.record));
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
    use super::{surface_choice, unserved_line, SurfaceChoice};

    /// With nothing named, `brain serve` is the OpenAI API a local client
    /// expects; `--stdio` is the controller; mixing the two, or a stdio-only
    /// flag without `--stdio`, is refused rather than half-honoured.
    #[test]
    fn serve_with_no_surface_named_is_the_openai_api() {
        assert_eq!(surface_choice(false, false, None), Ok(SurfaceChoice::DefaultOpenAi));
        assert_eq!(surface_choice(false, true, None), Ok(SurfaceChoice::Named));
        assert_eq!(surface_choice(true, false, None), Ok(SurfaceChoice::Stdio));
        assert_eq!(surface_choice(true, false, Some("--gpt")), Ok(SurfaceChoice::Stdio));
        assert!(surface_choice(true, true, None).unwrap_err().contains("--stdio"));
        let e = surface_choice(false, false, Some("--yolo")).unwrap_err();
        assert!(e.contains("--yolo") && e.contains("--stdio"), "{e}");
    }

    /// Families with nothing on disk are one counted line at the default
    /// level, and no line at all once `-v` lists them.
    #[test]
    fn unserved_families_are_counted_once_at_the_default_level() {
        assert_eq!(unserved_line(0, 0), None);
        assert!(unserved_line(12, 0).is_some_and(|l| l.contains("12") && l.contains("-v")));
        assert_eq!(unserved_line(12, 1), None);
    }

    #[test]
    fn a_server_with_models_lists_them_and_one_without_says_so_and_names_the_check() {
        use super::served_models_line;
        use std::path::Path;
        assert_eq!(served_models_line(&["qwen3", "flux2"], None), "brain serve: models: qwen3, flux2");
        let none = served_models_line(&[], Some(Path::new("/models")));
        assert!(none.contains("no model is being served") && none.contains("/models"), "{none}");
        assert!(none.contains("brain doctor"), "{none}");
        assert!(served_models_line(&[], None).contains("no model store is configured"));
    }

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
            super::build_auto_fetch_supplier(Some(&dir)).is_none(),
            "unset must build no supplier: fetching is opt-in"
        );
        std::env::set_var("BRAIN_AUTO_FETCH", "0");
        assert!(
            super::build_auto_fetch_supplier(Some(&dir)).is_none(),
            "=0 must keep fetching off"
        );
        std::env::set_var("BRAIN_AUTO_FETCH", "1");
        assert!(
            super::build_auto_fetch_supplier(Some(&dir)).is_some(),
            "=1 (what --autofetch publishes) must build the supplier"
        );
        std::env::remove_var("BRAIN_AUTO_FETCH");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Every flag the ad-hoc loop in `run_serve` actually parses must be
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
            "--admit-deadline-ms", "--cold-build-admit-deadline-ms",
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
