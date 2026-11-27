// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for putting one inference endpoint
// in front of every card of a cluster for its clients. If your team needs
// expertise in multi-node LLM serving then you can procure our services by
// sending an email to info@swedishembedded.com.

//! One OpenAI-compatible surface in front of several `brain serve` endpoints (`brain serve --upstream URL ...`).
//!
//! ```no_run
//! # fn main() -> Result<(), String> {
//! use apiserve::route::{Fleet, FleetConfig, UpstreamUrl};
//! let upstreams = ["http://127.0.0.1:8790", "http://127.0.0.1:8791"].iter().map(|u| UpstreamUrl::parse(u)).collect::<Result<_, _>>()?;
//! let fleet = Fleet::new(FleetConfig::new(upstreams))?;
//! # let _ = fleet; Ok(()) }
//! ```
//!
//! A client sees one URL and one key. Each upstream is a [`Member`] of the [`Fleet`]; a request goes to one of them,
//! chosen by the policy the in-process dispatcher uses for replicas of a model (`residency::affinity`): the upstream
//! whose prefix cache holds the longest useful prefix of the prompt, among those with a free slot, unless it carries
//! more than a few requests (a quarter of the lightest load, at least 8) beyond another, else the one with the lowest
//! load relative to its slots. The bound is in requests, not slots: an upstream's `decode_slots` is what it admits,
//! often far more than its card decodes at speed, so a cached prompt never holds a batch on one card while another
//! idles. Requests and responses pass through unbuffered (a streamed completion reaches the
//! client token by token), and a client that hangs up closes its upstream connection, which cancels the generation.
//!
//! What the router guarantees:
//!
//! - **Only routable upstreams take requests.** An upstream is in rotation while its `/health` and `/models` answer
//!   (probed every [`FleetConfig::health_interval`]), it is not draining, it is not stalled, and its health does not
//!   say `serving_held` (every card runs a workload, such as a fine-tune, that has not lent it). A drained upstream
//!   ([`Fleet::drain`], `POST /v1/upstreams/{n}/drain`) takes nothing new and finishes what it holds.
//! - **No request waits on an upstream that makes no progress.** An upstream holding work (requests the router sent
//!   it, or jobs it reports waiting) that neither answers, streams nor finishes anything for
//!   [`FleetConfig::stall_timeout`] is stalled: what the router sent it is failed (504, or an error ending the stream)
//!   and it leaves rotation, though its `/health` still answers (a hung device can leave the process answering), until
//!   it restarts, finishes something, or an operator resumes it. A restarted upstream's sessions and prefix record are
//!   dropped with its cache (what was sent to the old process failed with its connections).
//! - **Work is not queued behind work.** While some upstream reports no jobs waiting, one that reports them takes no
//!   new request, whatever prefix or session it holds: a queue that is not moving strands everything put into it.
//! - **A request is sent at most once.** It moves to another upstream only when the connection to the chosen one could
//!   not be opened, so no byte of it was sent. Once sent, a failure is the client's answer (502), because the
//!   generation may have run.
//! - **The fleet's edge is the sum of its decode slots.** An upstream is sent at most as many requests as it decodes at
//!   once (`decode_slots` in its `/health`), and at most as many requests for one model as it decodes of that model
//!   (`decode_slots` on the model's card, when the card names it), counted per slot group: an adapter's card names its
//!   base (`slot_group`), and the base and all its adapters share that one bound. Past that on every upstream serving the model a
//!   request waits in the router, in arrival order per model, for at most [`FleetConfig::queue_deadline`] (then 429);
//!   a request whose client goes away leaves the queue. Past [`FleetConfig::queue_bound`] waiting requests it is
//!   refused at once with 503. Both carry `Retry-After`.
//! - **A model goes where it is loaded, and spills only onto idle cards.** An upstream whose card says the model is not
//!   `resident` there would load it, evicting what that card holds, so it takes the model's requests only (a) while
//!   every upstream where it is has no free slot for it (an adapter is resident where its base is), and then an idle
//!   upstream first, else the one known to hold the fewest requests, never simply the lowest-index one, or
//!   (b) as a spill: every upstream where the model is resident
//!   already carries at least [`FleetConfig::spill_load`] requests for it (however many slots it advertises), and
//!   this one is idle, holding no request of the router's and reporting no job running or waiting, or holds only
//!   requests for this model (a spill already under way). An upstream busy with another model is never a spill target.
//!   Spilling is thrash-free: a card changes model only while nothing runs on it and the model it gains has a real
//!   batch elsewhere, and once loaded there the model is resident like any other. Without it a wide model (a 4B with
//!   hundreds of decode slots per card) never fills its one resident card and every card holding an idle model waits.
//! - **Conversations stay together while the fleet is balanced.** Requests carrying the same `prompt_cache_key`
//!   (OpenAI's routing hint) go to the same upstream, weighed as holding their whole prompt, so they leave it only
//!   when it carries more than the balance bound beyond another upstream with room, and then stay where they went;
//!   without a key, the prefix decides.
//! - **Only what is known is forwarded.** The JSON body is forwarded byte for byte after its top level is checked; the
//!   upstream sees only `Host`, `Content-Type`, `Content-Length`, `Accept`, the router's `X-Request-Id` for the request
//!   and the router's own upstream key, never the client's key or any other header. The client sees only the
//!   upstream's status, `Content-Type`, `Cache-Control`, `Retry-After` and body, and the same `X-Request-Id`.

mod key;
mod upstream;

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{DefaultBodyLimit, Extension, Path, Request, State};
use axum::http::{header, HeaderMap, Method, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use key::{Keyed, Parts, RequestKey};
use residency::affinity::{choose_replica, PrefixChain, PrefixDirectory, ReplicaView};
use serde::de::{Deserializer, MapAccess, Visitor};
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{json, Value};

pub use key::{PromptTokenizer, TokenizerSource};
pub use upstream::{check_upstream_exposure, UpstreamTls, UpstreamUrl};

use crate::auth::{Authenticator, Caller, Principal, Role};
use crate::error::ApiError;
use crate::listener::{self, ConnectionLimits};
use crate::surface::Provider;
use crate::tls::Tls;
use crate::MAX_BODY_BYTES;

/// The dialect the router speaks.
const PROVIDER: Provider = Provider::OpenAI;

/// The most upstreams one router fronts.
pub const MAX_UPSTREAMS: usize = 256;

/// The longest model id a request may name.
const MAX_MODEL_ID_BYTES: usize = 256;

/// The most models one upstream may list, and the most bytes its listing and its health answer may hold.
const MAX_LISTED_MODELS: usize = 4096;
const MAX_LISTING_BYTES: usize = 4 << 20;
const MAX_HEALTH_BYTES: usize = 64 << 10;

/// The most decode slots, and requests at once, an upstream may claim.
const MAX_ADVERTISED_SLOTS: u64 = 1 << 20;

/// The most idle keep-alive connections kept per upstream.
const MAX_IDLE_PER_UPSTREAM: usize = 64;

/// The longest drain deadline a request may ask for.
const MAX_DRAIN_DEADLINE: Duration = Duration::from_secs(24 * 3600);

/// How long one health probe may take.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Everything a [`Fleet`] is built from.
#[derive(Clone)]
pub struct FleetConfig {
    /// The upstream `brain serve` endpoints, in the order their indices name them.
    pub upstreams: Vec<UpstreamUrl>,
    /// The key the router presents to every upstream (`Authorization: Bearer`); `None` sends none.
    pub upstream_key: Option<String>,
    /// The authorities an `https://` upstream's certificate must chain to.
    pub upstream_tls: Option<UpstreamTls>,
    /// How long opening a connection (TCP, TLS and HTTP handshakes) may take before the upstream is passed over.
    pub connect_timeout: Duration,
    /// How long an upstream may take to answer a sent request with its status line. Its body then streams for as long
    /// as the upstream sends it.
    pub response_timeout: Duration,
    /// How often every upstream's health and model listing are read.
    pub health_interval: Duration,
    /// Prompts remembered per upstream for prefix affinity: the bound of the router's record of each upstream's
    /// prefix cache.
    pub cached_prompts: usize,
    /// Requests that may wait in the router for a free decode slot; one more is refused with 503.
    pub queue_bound: usize,
    /// How long a request may wait in the router for a slot before it is refused with 429.
    pub queue_deadline: Duration,
    /// How long a drained upstream may finish what it runs before the rest is cut off; `None` waits for it all.
    pub drain_deadline: Option<Duration>,
    /// How long an upstream holding work may go without answering, streaming or finishing anything before it is
    /// stalled. Bounds one request too: a request the upstream runs alone, unstreamed, for longer is cut as stalled.
    pub stall_timeout: Duration,
    /// Where the served models' tokenizers are found, to key prompts by tokens; `None` keys them by text.
    pub tokenizers: Option<Arc<dyn TokenizerSource>>,
    /// The requests for a model each upstream where the model is resident must carry before the model spills onto an
    /// idle upstream where it is not (see the module doc). `usize::MAX` never spills.
    pub spill_load: usize,
}

impl FleetConfig {
    /// `upstreams` with the defaults: 5 s to connect, 31 min for an answer to start (a long prefill on a busy
    /// upstream), 10 min without progress before an upstream is stalled, health every 2 s, 256 prompts remembered per
    /// upstream, 1024 requests waiting at most, each for at most the serving surfaces' admit deadline, and a model
    /// spilling onto idle upstreams once its resident upstreams each carry [`DEFAULT_SPILL_LOAD`] requests of it.
    pub fn new(upstreams: Vec<UpstreamUrl>) -> FleetConfig {
        FleetConfig {
            upstreams,
            upstream_key: None,
            upstream_tls: None,
            connect_timeout: Duration::from_secs(5),
            response_timeout: Duration::from_secs(31 * 60),
            health_interval: Duration::from_secs(2),
            cached_prompts: 256,
            queue_bound: 1024,
            queue_deadline: crate::DEFAULT_ADMIT_DEADLINE,
            drain_deadline: Some(Duration::from_secs(300)),
            stall_timeout: DEFAULT_STALL_TIMEOUT,
            tokenizers: None,
            spill_load: DEFAULT_SPILL_LOAD,
        }
    }
    pub fn with_spill_load(mut self, load: usize) -> FleetConfig {
        self.spill_load = load;
        self
    }
    pub fn with_tokenizers(mut self, source: Option<Arc<dyn TokenizerSource>>) -> FleetConfig {
        self.tokenizers = source;
        self
    }
    pub fn with_drain_deadline(mut self, deadline: Option<Duration>) -> FleetConfig {
        self.drain_deadline = deadline;
        self
    }
    pub fn with_queue(mut self, bound: usize, deadline: Duration) -> FleetConfig {
        self.queue_bound = bound;
        self.queue_deadline = deadline;
        self
    }
    pub fn with_upstream_key(mut self, key: Option<String>) -> FleetConfig {
        self.upstream_key = key;
        self
    }
    pub fn with_upstream_tls(mut self, tls: Option<UpstreamTls>) -> FleetConfig {
        self.upstream_tls = tls;
        self
    }
    pub fn with_connect_timeout(mut self, timeout: Duration) -> FleetConfig {
        self.connect_timeout = timeout;
        self
    }
    pub fn with_response_timeout(mut self, timeout: Duration) -> FleetConfig {
        self.response_timeout = timeout;
        self
    }
    pub fn with_health_interval(mut self, interval: Duration) -> FleetConfig {
        self.health_interval = interval;
        self
    }
    pub fn with_stall_timeout(mut self, timeout: Duration) -> FleetConfig {
        self.stall_timeout = timeout;
        self
    }
}

/// [`FleetConfig::stall_timeout`]'s default: longer than one long unstreamed generation on an otherwise idle upstream,
/// so that alone does not read as a hang.
pub const DEFAULT_STALL_TIMEOUT: Duration = Duration::from_secs(600);

/// [`FleetConfig::spill_load`]'s default: the small batch past which a cached prompt stops holding work on a busy
/// replica against an idle one (`residency::affinity::MIN_BALANCE_SLACK`, knowledge #607), so a model spills onto an
/// idle card where it is not loaded at the same load at which it would spread onto an idle card where it is. Under it
/// the model's demand is a handful of requests, one batch on its card, too few to pay for loading it elsewhere and
/// evicting what that card holds; past it an idle card is worth more. It is a count of requests, not a share of the
/// card's `decode_slots`: those are what a server admits, often far more than its card decodes at speed.
pub const DEFAULT_SPILL_LOAD: usize = residency::affinity::MIN_BALANCE_SLACK;

/// What the last good probe of an upstream said.
#[derive(Default)]
struct Advertised {
    /// Requests it decodes at once: the most the router sends it at once.
    slots: usize,
    /// Jobs waiting for a lane on the upstream itself.
    queue_depth: usize,
    /// Of its decode slots, those free (as of its last dispatcher pass).
    free_slots: usize,
    /// Every card runs a workload that has not lent it (`serving_held`): nothing sent there runs until it ends.
    held: bool,
    /// Free tokens of its KV pool, when it reports them.
    kv_free_tokens: Option<u64>,
    /// The process's identity: a change means it restarted, its caches (and maybe its weights) gone.
    instance: String,
    /// The model cards it lists, and their ids.
    cards: Vec<Value>,
    ids: HashSet<String>,
    /// For each model served with an adapter, the adapter's digest: part of the affinity namespace.
    adapter_digests: HashMap<String, String>,
    /// For each model whose card says so, the sequences of it the upstream decodes at once (`decode_slots`): the most
    /// of that model's requests it is sent at once. `slots` sums every model it could load.
    model_slots: HashMap<String, usize>,
    /// For each model whose card says so, whether it is built on the upstream's device now (`resident`). One that is
    /// not would have to be loaded there, evicting what the card holds.
    resident: HashMap<String, bool>,
    /// For each model whose card names one, the model whose instance serves it (`slot_group`): an adapter's base. The
    /// models of one group share one batch on the upstream, so their requests count against one bound.
    slot_group: HashMap<String, String>,
}

/// One upstream endpoint and what the router knows of it.
pub struct Member {
    url: UpstreamUrl,
    healthy: AtomicBool,
    /// Probed at least once: the first failure is reported, later ones only when the upstream was in rotation.
    probed: AtomicBool,
    draining: AtomicBool,
    in_flight: AtomicUsize,
    /// Of `in_flight`, the requests for each slot group (a model, or an adapter's base with all its adapters).
    model_in_flight: Mutex<HashMap<String, usize>>,
    advertised: RwLock<Advertised>,
    /// The upstream's cumulative served counters at the last two good probes, for its recent rates.
    readings: Mutex<Readings>,
    /// Its idle keep-alive connections.
    pool: Arc<upstream::Pool>,
    /// Set when what runs there is cut off (a drain's deadline passed, or it stalled): every request still running there
    /// ends with the reason.
    cut: tokio::sync::watch::Sender<Cut>,
    /// It held work and made no progress for the stall timeout: out of rotation until it restarts, finishes something
    /// or is resumed.
    stalled: AtomicBool,
    /// When it last made progress (answered, streamed or finished something, or held no work), in milliseconds since
    /// `epoch`.
    progress_ms: AtomicU64,
    epoch: std::time::Instant,
    /// Bumped by every drain and resume, so a deadline cuts only the drain that set it.
    drains: AtomicUsize,
}

/// Two readings of an upstream's served counters, the rates between them.
#[derive(Default)]
struct Readings {
    last: Option<(std::time::Instant, Served)>,
    tokens_per_s: f64,
    prefix_hit_rate: Option<f64>,
}

/// An upstream's cumulative served counters (`/v1/health`'s `served`).
#[derive(Clone, Copy, Default)]
struct Served {
    requests: u64,
    prompt_tokens: u64,
    completion_tokens: u64,
    cached_prompt_tokens: u64,
}

impl Member {
    fn new(url: UpstreamUrl) -> Member {
        Member {
            url,
            healthy: AtomicBool::new(false),
            probed: AtomicBool::new(false),
            draining: AtomicBool::new(false),
            in_flight: AtomicUsize::new(0),
            model_in_flight: Mutex::default(),
            advertised: RwLock::default(),
            readings: Mutex::default(),
            pool: Arc::new(upstream::Pool::new(MAX_IDLE_PER_UPSTREAM)),
            // The value before the first cut is never read: a receiver reads it only after a change.
            cut: tokio::sync::watch::Sender::new(Cut::Drained),
            drains: AtomicUsize::new(0),
            stalled: AtomicBool::new(false),
            progress_ms: AtomicU64::new(0),
            epoch: std::time::Instant::now(),
        }
    }

    fn advertised(&self) -> std::sync::RwLockReadGuard<'_, Advertised> {
        self.advertised.read().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The requests for slot group `group` it holds.
    fn model_load(&self, group: &str) -> usize {
        self.model_in_flight.lock().unwrap_or_else(std::sync::PoisonError::into_inner).get(group).copied().unwrap_or(0)
    }

    /// Whether it takes new requests: it answers its probes, is not draining, has not stalled, and its cards are not
    /// all held by a workload (knowledge #611).
    fn routable(&self) -> bool {
        self.healthy.load(Ordering::SeqCst) && !self.draining.load(Ordering::SeqCst) && !self.stalled.load(Ordering::SeqCst) && !self.advertised().held
    }

    /// It made progress now.
    fn progressed(&self) {
        self.progress_ms.store(self.epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
    }

    /// How long since it last made progress.
    fn since_progress(&self) -> Duration {
        self.epoch.elapsed().saturating_sub(Duration::from_millis(self.progress_ms.load(Ordering::Relaxed)))
    }
}

/// Why the requests running on an upstream were cut off.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cut {
    Drained,
    Stalled,
}

impl Cut {
    /// The client's answer for a request cut before the upstream answered.
    fn refusal(self, stall: Duration) -> ApiError {
        match self {
            Cut::Drained => ApiError::bad_gateway(PROVIDER, "the upstream was drained before it answered; the request may have run"),
            Cut::Stalled => ApiError::failure(PROVIDER, capability::FailureClass::Timeout, format!("the upstream made no progress for {}s and is out of rotation; the request may have run", stall.as_secs_f64())),
        }
    }

    /// Why a stream that had started was ended.
    fn ended(self) -> &'static str {
        match self {
            Cut::Drained => "the upstream was drained before the response ended",
            Cut::Stalled => "the upstream made no progress within the stall timeout before the response ended",
        }
    }
}

/// One request for a slot group held by an upstream: its load, given back when the response ends or is dropped, which
/// wakes the requests waiting for a slot.
struct InFlight(Arc<Member>, Arc<tokio::sync::Notify>, String);

impl InFlight {
    fn take(member: Arc<Member>, changed: Arc<tokio::sync::Notify>, model: &str) -> InFlight {
        *member.model_in_flight.lock().unwrap_or_else(std::sync::PoisonError::into_inner).entry(model.to_string()).or_default() += 1;
        InFlight(member, changed, model.to_string())
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        {
            let mut held = self.0.model_in_flight.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(n) = held.get_mut(&self.2) {
                *n -= 1;
                if *n == 0 {
                    held.remove(&self.2);
                }
            }
        }
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.1.notify_waiters();
    }
}

/// An upstream with room for a request, as [`Fleet::pick`] weighs it.
struct Open {
    index: usize,
    view: ReplicaView,
    /// The request's prefix namespace there (its model and adapter digest).
    namespace: String,
    /// The slot group the request counts against there.
    group: String,
    /// The requests it is known to hold or queue, the router's and those its health reports.
    busy: usize,
    /// It holds nothing of the router's and reports no job running or waiting.
    idle: bool,
}

/// The outcome of choosing an upstream.
enum Pick {
    Chosen(usize, InFlight),
    /// No upstream that answers lists the model.
    UnknownModel,
    /// Upstreams list the model, but none is in rotation now.
    NoneAvailable,
    /// Every upstream that could take it holds as many requests as it decodes at once.
    Saturated,
}

/// Why a request did not get an upstream.
enum Refusal {
    UnknownModel,
    NoneAvailable,
    /// The router's queue holds [`FleetConfig::queue_bound`] requests already.
    QueueFull,
    /// It waited [`FleetConfig::queue_deadline`] without a slot.
    Deadline,
}

/// The requests waiting for a slot, in arrival order, each with the model it asks for.
#[derive(Default)]
struct Waiting {
    next: u64,
    queue: std::collections::BTreeMap<u64, String>,
}

/// The longest session key (`prompt_cache_key`) a request may carry, and how many sessions are remembered.
const MAX_SESSION_KEY_BYTES: usize = 256;
const MAX_SESSIONS: usize = 65536;

/// Which upstream each recent session went to, oldest forgotten first.
#[derive(Default)]
struct Sessions {
    upstream: HashMap<String, (usize, u64)>,
    order: std::collections::BTreeMap<u64, String>,
    clock: u64,
}

impl Sessions {
    fn get(&self, key: &str) -> Option<usize> {
        self.upstream.get(key).map(|&(i, _)| i)
    }
    fn record(&mut self, key: &str, index: usize) {
        self.clock += 1;
        if let Some((_, stamp)) = self.upstream.insert(key.to_string(), (index, self.clock)) {
            self.order.remove(&stamp);
        }
        self.order.insert(self.clock, key.to_string());
        while self.upstream.len() > MAX_SESSIONS {
            let Some((_, oldest)) = self.order.pop_first() else { break };
            self.upstream.remove(&oldest);
        }
    }
    /// Forget every session that went to upstream `index`: what made it worth returning to is gone.
    fn forget(&mut self, index: usize) {
        let upstream = &mut self.upstream;
        self.order.retain(|_, key| match upstream.get(key) {
            Some(&(i, _)) if i == index => {
                upstream.remove(key);
                false
            }
            _ => true,
        });
    }
}

/// A request's place in the router's queue, given up when it gets a slot, gives up, or its client goes away.
struct Ticket<'a> {
    fleet: &'a Fleet,
    id: u64,
}

impl Drop for Ticket<'_> {
    fn drop(&mut self) {
        self.fleet.waiting.lock().unwrap_or_else(std::sync::PoisonError::into_inner).queue.remove(&self.id);
        // The next request for the same models may now be first.
        self.fleet.changed.notify_waiters();
    }
}

/// The upstreams one router fronts, their state, and the record of which prompts went where.
pub struct Fleet {
    members: Vec<Arc<Member>>,
    /// Per upstream, the prompts routed to it: what its prefix cache holds, as far as this router knows.
    directories: Mutex<Vec<PrefixDirectory>>,
    connector: upstream::Connector,
    key: Option<String>,
    response_timeout: Duration,
    health_interval: Duration,
    /// Requests waiting for a slot, and the signal that one may have freed (a response ended, a probe answered).
    waiting: Mutex<Waiting>,
    changed: Arc<tokio::sync::Notify>,
    queue_bound: usize,
    queue_deadline: Duration,
    drain_deadline: Option<Duration>,
    stall_timeout: Duration,
    sessions: Mutex<Sessions>,
    tokens: Arc<key::Tokens>,
    spill_load: usize,
}

impl std::fmt::Debug for Fleet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fleet").field("upstreams", &self.members.iter().map(|m| m.url.to_string()).collect::<Vec<_>>()).finish_non_exhaustive()
    }
}

impl Fleet {
    /// A fleet over `config.upstreams`. Every upstream starts out of rotation until its first probe ([`Fleet::probe`])
    /// answers.
    ///
    /// # Errors
    /// No upstream, more than [`MAX_UPSTREAMS`], one named twice, an empty key, or a zero timeout.
    pub fn new(config: FleetConfig) -> Result<Arc<Fleet>, String> {
        if config.upstreams.is_empty() {
            return Err("no upstream is given".into());
        }
        if config.upstreams.len() > MAX_UPSTREAMS {
            return Err(format!("at most {MAX_UPSTREAMS} upstreams"));
        }
        let mut seen = HashSet::new();
        if let Some(twice) = config.upstreams.iter().find(|u| !seen.insert(u.to_string())) {
            return Err(format!("upstream {twice} is given twice"));
        }
        if config.upstream_key.as_deref().is_some_and(|k| k.is_empty() || k.chars().any(|c| c.is_control() || c.is_whitespace())) {
            return Err("the upstream key is empty or holds whitespace".into());
        }
        if config.connect_timeout.is_zero() || config.response_timeout.is_zero() || config.health_interval.is_zero() || config.queue_deadline.is_zero() || config.stall_timeout.is_zero() {
            return Err("timeouts and the health interval are positive".into());
        }
        if config.spill_load == 0 {
            return Err("the spill load is at least one request".into());
        }
        let members: Vec<Arc<Member>> = config.upstreams.into_iter().map(|url| Arc::new(Member::new(url))).collect();
        let directories = Mutex::new(members.iter().map(|_| PrefixDirectory::new(config.cached_prompts)).collect());
        Ok(Arc::new(Fleet {
            members,
            directories,
            connector: upstream::Connector::new(config.upstream_tls, config.connect_timeout),
            key: config.upstream_key,
            response_timeout: config.response_timeout,
            health_interval: config.health_interval,
            waiting: Mutex::default(),
            changed: Arc::new(tokio::sync::Notify::new()),
            queue_bound: config.queue_bound,
            queue_deadline: config.queue_deadline,
            drain_deadline: config.drain_deadline,
            stall_timeout: config.stall_timeout,
            sessions: Mutex::default(),
            tokens: Arc::new(key::Tokens::new(config.tokenizers)),
            spill_load: config.spill_load,
        }))
    }

    /// Stop sending new requests to upstream `index`; what it holds finishes. `false` when there is no such upstream.
    ///
    /// With `deadline`, whatever it still runs when the deadline passes (and the upstream has not been resumed since)
    /// is cut off: the client's stream ends in an error and the upstream connection closes, which cancels the
    /// generation. Needs a Tokio runtime when a deadline is given.
    pub fn drain(&self, index: usize, deadline: Option<Duration>) -> bool {
        let Some(member) = self.members.get(index) else { return false };
        member.draining.store(true, Ordering::SeqCst);
        let drain = member.drains.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(deadline) = deadline {
            let member = member.clone();
            tokio::spawn(async move {
                tokio::time::sleep(deadline).await;
                if member.draining.load(Ordering::SeqCst) && member.drains.load(Ordering::SeqCst) == drain {
                    member.cut.send_replace(Cut::Drained);
                }
            });
        }
        true
    }

    /// Put a drained or stalled upstream `index` back in rotation. `false` when there is no such upstream.
    pub fn resume(&self, index: usize) -> bool {
        let found = self
            .members
            .get(index)
            .map(|m| {
                m.drains.fetch_add(1, Ordering::SeqCst);
                m.progressed();
                m.stalled.store(false, Ordering::SeqCst);
                m.draining.store(false, Ordering::SeqCst)
            })
            .is_some();
        self.changed.notify_waiters();
        found
    }

    /// The client connections the router must accept to run every decode slot of its upstreams (as last probed), hold
    /// a full queue, and still answer the overflow with 503: twice the slots and the queue bound.
    fn connections_needed(&self) -> usize {
        let slots: usize = self.members.iter().map(|m| m.advertised().slots).sum();
        slots.saturating_add(self.queue_bound).saturating_mul(2)
    }

    /// Whether some upstream, in rotation or not, lists `model`.
    fn lists(&self, model: &str) -> bool {
        self.members.iter().any(|m| m.advertised().ids.contains(model))
    }

    /// Whether some upstream is in rotation.
    pub fn available(&self) -> bool {
        self.members.iter().any(|m| m.routable())
    }

    /// The fleet as `GET /health` reports it, for a harness sizing its concurrency: per upstream (by index; addresses
    /// stay with the operator) whether it is healthy, draining or stalled, the requests it holds (`in_flight`), the most it is
    /// sent at once (`capacity`, its decode slots), the jobs waiting on it (`queue_depth`), and its recent completion
    /// token rate and prefix-cache hit rate between the last two probes; then the fleet's totals and the requests
    /// waiting in the router (`queued`).
    pub fn status(&self) -> Value {
        let (mut capacity, mut in_flight, mut upstream_queued, mut rate) = (0usize, 0usize, 0usize, 0f64);
        let upstreams: Vec<Value> = self
            .members
            .iter()
            .enumerate()
            .map(|(index, m)| {
                let adv = m.advertised();
                let readings = m.readings.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let held = m.in_flight.load(Ordering::SeqCst);
                in_flight += held;
                if m.routable() {
                    capacity += adv.slots;
                    upstream_queued += adv.queue_depth;
                    rate += readings.tokens_per_s;
                }
                json!({
                    "index": index,
                    "healthy": m.healthy.load(Ordering::SeqCst),
                    "serving_held": adv.held,
                    "draining": m.draining.load(Ordering::SeqCst),
                    "stalled": m.stalled.load(Ordering::SeqCst),
                    "in_flight": held,
                    "capacity": adv.slots,
                    "queue_depth": adv.queue_depth,
                    "tokens_per_s": readings.tokens_per_s,
                    "prefix_hit_rate": readings.prefix_hit_rate,
                    "idle_connections": m.pool.idle(),
                    "free_slots": adv.free_slots,
                    "kv_free_tokens": adv.kv_free_tokens,
                    "models": adv.ids.len(),
                })
            })
            .collect();
        let queued = self.waiting.lock().unwrap_or_else(std::sync::PoisonError::into_inner).queue.len();
        let status = if self.available() { "ok" } else { "unavailable" };
        json!({
            "status": status,
            "decode_slots": capacity,
            "capacity": capacity,
            "in_flight": in_flight,
            "queued": queued,
            "queue_bound": self.queue_bound,
            "upstream_queue_depth": upstream_queued,
            "tokens_per_s": rate,
            "keying": {
                "token_keyed_requests": self.tokens.token_keyed.load(Ordering::Relaxed),
                "text_keyed_requests": self.tokens.text_keyed.load(Ordering::Relaxed),
                "encoded_tokens": self.tokens.encoded_tokens.load(Ordering::Relaxed),
                "memo_tokens": self.tokens.memo_tokens.load(Ordering::Relaxed),
                "encode_seconds": self.tokens.encode_nanos.load(Ordering::Relaxed) as f64 / 1e9,
            },
            "upstreams": upstreams,
        })
    }

    /// Read every upstream's health and model listing once, concurrently, and put each in or out of rotation by the
    /// answer.
    pub async fn probe(&self) {
        futures::future::join_all((0..self.members.len()).map(|i| self.probe_one(i))).await;
    }

    async fn probe_one(&self, index: usize) {
        let member = &self.members[index];
        let key = self.key.as_deref();
        let read = async {
            let health = upstream::get_json(&self.connector, &member.url, "/v1/health", key, MAX_HEALTH_BYTES, PROBE_TIMEOUT).await?;
            let listing = upstream::get_json(&self.connector, &member.url, "/v1/models", key, MAX_LISTING_BYTES, PROBE_TIMEOUT).await?;
            parse_advertised(&health, &listing)
        };
        let queued = match read.await {
            Ok((advertised, served)) => {
                let (restarted, queued) = {
                    let mut current = member.advertised.write().unwrap_or_else(std::sync::PoisonError::into_inner);
                    let restarted = !current.instance.is_empty() && current.instance != advertised.instance;
                    *current = advertised;
                    (restarted, current.queue_depth)
                };
                if restarted {
                    // Another process now: what the router recorded of its cache is gone with the old one. What was sent
                    // to the old one failed with its connections; what was sent since went to the new one and runs on.
                    self.forget(index);
                    member.progressed();
                    if member.stalled.swap(false, Ordering::SeqCst) {
                        eprintln!("brain route: upstream {index} ({}) restarted; it is no longer stalled", member.url);
                    }
                    eprintln!("brain route: upstream {index} ({}) restarted; its sessions and prefix record are dropped", member.url);
                }
                let finished = member.readings.lock().unwrap_or_else(std::sync::PoisonError::into_inner).read(served);
                if finished {
                    member.progressed();
                    if member.stalled.swap(false, Ordering::SeqCst) {
                        eprintln!("brain route: upstream {index} ({}) finished work again; it is no longer stalled", member.url);
                    }
                }
                self.changed.notify_waiters();
                member.probed.store(true, Ordering::SeqCst);
                if !member.healthy.swap(true, Ordering::SeqCst) {
                    eprintln!("brain route: upstream {index} ({}) is in rotation", member.url);
                }
                queued
            }
            Err(why) => {
                self.mark_down(index, &why);
                0
            }
        };
        self.check_progress(index, queued);
    }

    /// Stall upstream `index` if it holds work (`queued` jobs waiting on it as it reports them, or requests the router
    /// sent it) and has made no progress for the stall timeout; an upstream holding none is not stuck.
    fn check_progress(&self, index: usize, queued: usize) {
        let member = &self.members[index];
        let held = member.in_flight.load(Ordering::SeqCst);
        if held == 0 && queued == 0 {
            member.progressed();
            return;
        }
        let idle = member.since_progress();
        if idle < self.stall_timeout || member.stalled.swap(true, Ordering::SeqCst) {
            return;
        }
        eprintln!(
            "brain route: upstream {index} ({}) made no progress for {:.1}s while holding {held} requests and reporting {queued} queued; \
             it is stalled: what it holds is failed and it is out of rotation until it restarts, finishes something or is resumed",
            member.url,
            idle.as_secs_f64()
        );
        self.forget(index);
        member.cut.send_replace(Cut::Stalled);
        self.changed.notify_waiters();
    }

    /// Drop what the router remembers of upstream `index`'s cache: its prefix record and the sessions that went there.
    fn forget(&self, index: usize) {
        self.directories.lock().unwrap_or_else(std::sync::PoisonError::into_inner)[index].clear();
        self.sessions.lock().unwrap_or_else(std::sync::PoisonError::into_inner).forget(index);
    }

    /// Take upstream `index` out of rotation until a probe answers again. Its prefix cache may not survive what
    /// happened to it, so the record of it, and the sessions that went there, are dropped.
    fn mark_down(&self, index: usize, why: &str) {
        let member = &self.members[index];
        let was_healthy = member.healthy.swap(false, Ordering::SeqCst);
        if was_healthy || !member.probed.swap(true, Ordering::SeqCst) {
            eprintln!("brain route: upstream {index} ({}) is out of rotation: {why}", member.url);
        }
        self.forget(index);
    }

    /// Probe every upstream every [`FleetConfig::health_interval`] until `shutdown` fires.
    async fn keep_probing(self: Arc<Fleet>, shutdown: brain_shutdown::Shutdown) {
        let stop = shutdown.wait();
        tokio::pin!(stop);
        loop {
            self.probe().await;
            tokio::select! {
                _ = tokio::time::sleep(self.health_interval) => {}
                _ = &mut stop => return,
            }
        }
    }

    /// The upstream that runs a request for `model` keyed by `key` (see the module doc), with one of its slots taken.
    /// `chains` holds the request's chain per namespace, built once. `excluded` upstreams failed to connect for it.
    fn pick(&self, model: &str, key: &RequestKey, chains: &mut HashMap<String, PrefixChain>, session: Option<&str>, excluded: &[bool]) -> Pick {
        let mut directories = self.directories.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let (mut known, mut open, mut candidates) = (false, Vec::<Open>::new(), 0usize);
        // The candidates where the model is resident, those of them carrying less than their spill load of it, and the
        // open candidates where it is not that may take a spill (see the module doc).
        let (mut residents, mut unpressed, mut spill) = (0usize, 0usize, Vec::new());
        for (i, m) in self.members.iter().enumerate() {
            let adv = m.advertised();
            let serves = adv.ids.contains(model);
            known |= serves;
            if !serves || excluded[i] || !m.routable() {
                continue;
            }
            candidates += 1;
            // The model's load is its slot group's: an adapter shares its base's batch with every other adapter of it.
            let group = adv.slot_group.get(model).map_or(model, String::as_str);
            // Its slots for this model when its card names them (they bound the group's own requests there), else
            // its slots for every model; either way never more than its slots in all.
            let total = m.in_flight.load(Ordering::SeqCst);
            let mine = m.model_load(group);
            let (load, slots) = match adv.model_slots.get(model) {
                Some(&slots) => (mine, slots),
                None => (total, adv.slots),
            };
            let resident = adv.resident.get(model).copied().unwrap_or(true);
            if resident {
                residents += 1;
                unpressed += usize::from(mine < self.spill_load);
            }
            if load < slots && total < adv.slots {
                let namespace = format!("{model}\u{1f}{}", adv.adapter_digests.get(model).map_or("", String::as_str));
                let chain = chains.entry(namespace.clone()).or_insert_with(|| key.chain(&namespace));
                let kv_free = if key.in_tokens() { adv.kv_free_tokens } else { None };
                // An upstream that says the model is not loaded there is chosen only when none where it is has room,
                // unless it may take a spill (marked below, once every resident upstream's load is known).
                let view = ReplicaView { load, slots, queued: adv.queue_depth, kv_free, active: resident, can_run: true, matched: directories[i].matched(chain) };
                // Idle: nothing of the router's and no job running or waiting by its last health. Or already spilled
                // onto: everything the router sent it is this model's. Anything else is another model's running work.
                let idle = total == 0 && adv.queue_depth == 0 && adv.free_slots >= adv.slots;
                if !resident && total == mine && (idle || total > 0) {
                    spill.push(open.len());
                }
                // All it is known to hold: the router's requests, or what it runs by its last health when that is more
                // (another router's), and what waits on it.
                let busy = total.max(adv.slots.saturating_sub(adv.free_slots)) + adv.queue_depth;
                open.push(Open { index: i, view, namespace, group: group.to_string(), busy, idle });
            }
        }
        // With no candidate where the model is resident there is nothing to spill from: the first rule places it.
        if residents > 0 && unpressed == 0 {
            for &o in &spill {
                open[o].view.active = true;
            }
        }
        if candidates == 0 {
            return if known || !self.available() { Pick::NoneAvailable } else { Pick::UnknownModel };
        }
        if open.is_empty() {
            return Pick::Saturated;
        }
        // Work is not queued behind work: while an upstream with room reports nothing waiting, one that reports jobs
        // waiting is passed over, whatever prefix or session it holds. A queue that does not move (a restarted upstream
        // still loading, a hung lane) would otherwise take every request its cache or a session points at.
        if open.iter().any(|o| o.view.queued == 0) {
            open.retain(|o| o.view.queued == 0);
        }
        // Nowhere with room is the model loaded (or may it spill): it is loaded where that costs least. An idle
        // upstream evicts nothing in use; failing one, the upstream known to hold the fewest requests. Never
        // simply the first open one, which put every such request on the lowest-index upstream whatever it ran
        // (knowledge #609).
        if !open.iter().any(|o| o.view.active) {
            if open.iter().any(|o| o.idle) {
                open.retain(|o| o.idle);
                for o in &mut open {
                    o.view.active = true;
                }
            } else {
                let lightest = (0..open.len()).min_by_key(|&o| (open[o].busy, open[o].index)).expect("open is not empty");
                open.swap(0, lightest);
                open.truncate(1);
                open[0].view.active = true;
            }
        }
        let mut sessions = self.sessions.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        // A session's upstream holds its whole history: it is weighed as holding the whole prompt, so the session goes
        // back there unless that upstream carries more than the balance bound beyond another (`balance_slack`), and
        // then follows the one the policy chooses. A rule of "while it has a free slot" pinned sessions to a card
        // whose advertised slots never fill.
        let size = key.size(&chains[&open[0].namespace]);
        let mut views: Vec<ReplicaView> = open.iter().map(|o| o.view).collect();
        if let Some(home) = session.and_then(|key| sessions.get(key)).and_then(|i| open.iter().position(|o| o.index == i)) {
            views[home].matched = views[home].matched.max(size.prompt);
        }
        let chosen = &open[choose_replica(&views, size)];
        if let Some(key) = session {
            sessions.record(key, chosen.index);
        }
        directories[chosen.index].record(&chains[&chosen.namespace]);
        let member = self.members[chosen.index].clone();
        if member.in_flight.fetch_add(1, Ordering::SeqCst) == 0 && member.advertised().queue_depth == 0 {
            // Work starts on an idle upstream: its stall clock starts now, not at the last probe.
            member.progressed();
        }
        Pick::Chosen(chosen.index, InFlight::take(member, self.changed.clone(), &chosen.group))
    }

    /// An upstream for a request, waiting in the router's queue while every upstream that serves its model holds as
    /// many requests as it decodes at once. Requests for one model get slots in arrival order; a request whose client
    /// goes away leaves the queue with it (this future is dropped).
    async fn admit(&self, model: &str, key: &RequestKey, session: Option<&str>, excluded: &[bool]) -> Result<(usize, InFlight), Refusal> {
        let mut chains = HashMap::new();
        let deadline = tokio::time::Instant::now() + self.queue_deadline;
        let mut ticket: Option<Ticket<'_>> = None;
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let first = {
                let waiting = self.waiting.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                let mine = ticket.as_ref().map_or(u64::MAX, |t| t.id);
                !waiting.queue.iter().any(|(&id, m)| id < mine && m == model)
            };
            if first {
                match self.pick(model, key, &mut chains, session, excluded) {
                    Pick::Chosen(index, held) => return Ok((index, held)),
                    Pick::UnknownModel => return Err(Refusal::UnknownModel),
                    Pick::NoneAvailable => return Err(Refusal::NoneAvailable),
                    Pick::Saturated => {}
                }
            }
            if ticket.is_none() {
                let mut waiting = self.waiting.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if waiting.queue.len() >= self.queue_bound {
                    return Err(Refusal::QueueFull);
                }
                waiting.next += 1;
                let id = waiting.next;
                waiting.queue.insert(id, model.to_string());
                ticket = Some(Ticket { fleet: self, id });
                continue;
            }
            tokio::select! {
                _ = &mut changed => {}
                _ = tokio::time::sleep_until(deadline) => return Err(Refusal::Deadline),
            }
        }
    }
}

/// The upstream's answers to `/v1/health` and `/v1/models`, checked: they come from another process and bound what
/// this one allocates.
fn parse_advertised(health: &Value, listing: &Value) -> Result<(Advertised, Served), String> {
    let slots = health.get("decode_slots").and_then(Value::as_u64).filter(|&n| (1..=MAX_ADVERTISED_SLOTS).contains(&n)).ok_or("its /v1/health names no decode_slots in range")? as usize;
    let queue_depth = health.get("queue_depth").and_then(Value::as_u64).unwrap_or(0).min(MAX_ADVERTISED_SLOTS) as usize;
    let counter = |name: &str| health.get("served").and_then(|s| s.get(name)).and_then(Value::as_u64).unwrap_or(0);
    let served = Served { requests: counter("requests"), prompt_tokens: counter("prompt_tokens"), completion_tokens: counter("completion_tokens"), cached_prompt_tokens: counter("cached_prompt_tokens") };
    let data = listing.get("data").and_then(Value::as_array).ok_or("its /v1/models has no data list")?;
    if data.len() > MAX_LISTED_MODELS {
        return Err(format!("it lists more than {MAX_LISTED_MODELS} models"));
    }
    let free_slots = health.get("free_slots").and_then(Value::as_u64).map_or(slots, |n| (n as usize).min(slots));
    let kv_free_tokens = health.get("kv_free_tokens").and_then(Value::as_u64);
    let instance = health.get("instance").and_then(Value::as_str).filter(|s| s.len() <= 64 && s.bytes().all(|b| b.is_ascii_alphanumeric())).unwrap_or("").to_string();
    let held = health.get("serving_held").and_then(Value::as_bool).unwrap_or(false);
    let mut adv = Advertised { slots, queue_depth, free_slots, held, kv_free_tokens, instance, ..Advertised::default() };
    for card in data {
        let Some(id) = card.get("id").and_then(Value::as_str).filter(|id| valid_model_id(id)) else {
            return Err("it lists a model whose id is not a plain string".into());
        };
        if adv.ids.insert(id.to_string()) {
            if let Some(slots) = card.get("decode_slots").and_then(Value::as_u64).filter(|&n| (1..=MAX_ADVERTISED_SLOTS).contains(&n)) {
                adv.model_slots.insert(id.to_string(), slots as usize);
            }
            if let Some(resident) = card.get("resident").and_then(Value::as_bool) {
                adv.resident.insert(id.to_string(), resident);
            }
            if let Some(group) = card.get("slot_group").and_then(Value::as_str).filter(|g| valid_model_id(g)) {
                adv.slot_group.insert(id.to_string(), group.to_string());
            }
            adv.cards.push(card.clone());
            if let Some(digest) = card.get("adapter").and_then(|a| a.get("digest")).and_then(Value::as_str).filter(|d| d.len() <= 128) {
                adv.adapter_digests.insert(id.to_string(), digest.to_string());
            }
        }
    }
    Ok((adv, served))
}

impl Readings {
    /// Take a new reading: the rates are over the time since the last one. A counter that went back (the upstream
    /// restarted) starts the rates over. `true` when the upstream finished something since the last reading.
    fn read(&mut self, now: Served) -> bool {
        let at = std::time::Instant::now();
        let finished = self.last.is_some_and(|(_, before)| now.requests > before.requests || now.completion_tokens > before.completion_tokens);
        if let Some((then, before)) = self.last {
            let secs = at.duration_since(then).as_secs_f64();
            let completion = now.completion_tokens.checked_sub(before.completion_tokens);
            let prompt = now.prompt_tokens.checked_sub(before.prompt_tokens);
            let cached = now.cached_prompt_tokens.checked_sub(before.cached_prompt_tokens);
            if let (Some(completion), Some(prompt), Some(cached)) = (completion, prompt, cached) {
                if secs > 0.0 {
                    self.tokens_per_s = completion as f64 / secs;
                }
                if prompt > 0 {
                    self.prefix_hit_rate = Some(cached as f64 / prompt as f64);
                }
            }
        }
        self.last = Some((at, now));
        finished
    }
}

fn valid_model_id(id: &str) -> bool {
    !id.is_empty() && id.len() <= MAX_MODEL_ID_BYTES && !id.chars().any(char::is_control)
}

/// The top level of a request body: its fields in order, each value as the client wrote it. A field named twice is
/// refused, so the router and the upstream cannot read two different requests out of one body.
struct Head(Vec<(String, Box<RawValue>)>);

impl<'de> Deserialize<'de> for Head {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Head, D::Error> {
        struct Fields;
        impl<'de> Visitor<'de> for Fields {
            type Value = Head;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Head, A::Error> {
                let mut fields: Vec<(String, Box<RawValue>)> = Vec::new();
                let mut names = HashSet::new();
                while let Some((name, value)) = map.next_entry::<String, Box<RawValue>>()? {
                    if !names.insert(name.clone()) {
                        return Err(serde::de::Error::custom(format!("the field {name:?} is given twice")));
                    }
                    fields.push((name, value));
                }
                Ok(Head(fields))
            }
        }
        deserializer.deserialize_map(Fields)
    }
}

impl Head {
    fn get(&self, name: &str) -> Option<&RawValue> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, v)| v.as_ref())
    }
}

/// The upstream routes a fleet client calls, each with how it is keyed. A path the router does not list here is a 404:
/// background jobs and video jobs live on the upstream that started them, which a stateless router cannot find again.
const FORWARDED: [(&str, Keyed); 10] = [
    ("/v1/chat/completions", Keyed::Chat),
    ("/chat/completions", Keyed::Chat),
    ("/v1/completions", Keyed::Completion),
    ("/completions", Keyed::Completion),
    ("/v1/embeddings", Keyed::Load),
    ("/embeddings", Keyed::Load),
    ("/v1/images/generations", Keyed::Load),
    ("/images/generations", Keyed::Load),
    ("/v1/audio/speech", Keyed::Load),
    ("/audio/speech", Keyed::Load),
];

#[derive(Clone)]
struct RouteState {
    fleet: Arc<Fleet>,
    authenticator: Arc<dyn Authenticator>,
}

/// The router's axum [`Router`]: every forwarded route, the merged `/models`, `/health`, and the drain routes, all
/// behind `authenticator` (fallback included) and the body limit. Exposed so a test can drive it without a socket.
pub fn router(fleet: Arc<Fleet>, authenticator: Arc<dyn Authenticator>) -> Router {
    let state = RouteState { fleet, authenticator };
    let mut app = Router::new()
        .route("/models", get(list_models))
        .route("/v1/models", get(list_models))
        .route("/models/{id}", get(one_model))
        .route("/v1/models/{id}", get(one_model))
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/metrics", get(health))
        .route("/v1/metrics", get(health))
        .route("/v1/upstreams/{index}/drain", post(drain))
        .route("/v1/upstreams/{index}/resume", post(resume));
    for (path, keyed) in FORWARDED {
        app = app.route(path, post(move |state: State<RouteState>, principal: Extension<Principal>, headers: HeaderMap, body: Bytes| forward(state, principal, headers, body, path, keyed)));
    }
    app.fallback(|| async { ApiError::not_found(PROVIDER, "no such route") })
        .layer(axum::middleware::from_fn_with_state(state.clone(), authenticate))
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state)
}

async fn authenticate(State(state): State<RouteState>, mut req: Request, next: Next) -> Result<Response, ApiError> {
    let principal = state.authenticator.authenticate(PROVIDER, req.headers())?;
    req.extensions_mut().insert(principal);
    Ok(next.run(req).await)
}

async fn list_models(State(state): State<RouteState>) -> Json<Value> {
    Json(crate::models::envelope(PROVIDER, merged_cards(&state.fleet).into_values().collect()))
}

async fn one_model(State(state): State<RouteState>, Path(id): Path<String>) -> Result<Json<Value>, ApiError> {
    merged_cards(&state.fleet).remove(&id).map(Json).ok_or_else(|| ApiError::model_not_found(PROVIDER, &id))
}

/// Every model an upstream in rotation lists, once, in id order; the first upstream's card for an id wins, without what
/// it says of serving that model on its own card (`decode_slots`, `resident`): the router's listing is the fleet's.
fn merged_cards(fleet: &Fleet) -> std::collections::BTreeMap<String, Value> {
    let mut cards = std::collections::BTreeMap::new();
    for m in fleet.members.iter().filter(|m| m.routable()) {
        let adv = m.advertised();
        for card in &adv.cards {
            if let Some(id) = card.get("id").and_then(Value::as_str) {
                cards.entry(id.to_string()).or_insert_with(|| {
                    let mut card = card.clone();
                    if let Some(fields) = card.as_object_mut() {
                        fields.remove("decode_slots");
                        fields.remove("resident");
                    }
                    card
                });
            }
        }
    }
    cards
}

async fn health(State(state): State<RouteState>) -> Response {
    let code = if state.fleet.available() { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE };
    (code, Json(state.fleet.status())).into_response()
}

/// Draining moves load between machines, so it is the operator's: with named callers only an admin may.
fn operator_only(principal: &Principal) -> Result<(), ApiError> {
    match Caller::of(Some(principal)) {
        Some(caller) if caller.role < Role::Admin => Err(ApiError::forbidden(PROVIDER, "only an admin caller may drain or resume an upstream")),
        _ => Ok(()),
    }
}

/// `POST /v1/upstreams/{n}/drain`, optionally with `{"deadline_ms": N}` (else [`FleetConfig::drain_deadline`]).
async fn drain(State(state): State<RouteState>, Extension(principal): Extension<Principal>, Path(index): Path<String>, body: Bytes) -> Result<Json<Value>, ApiError> {
    operator_only(&principal)?;
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct DrainBody {
        deadline_ms: Option<u64>,
    }
    let asked = if body.is_empty() { None } else { serde_json::from_slice::<DrainBody>(&body).map_err(|e| ApiError::invalid_request(PROVIDER, format!("drain body: {e}")))?.deadline_ms };
    let deadline = match asked {
        Some(ms) if ms == 0 || Duration::from_millis(ms) > MAX_DRAIN_DEADLINE => return Err(ApiError::invalid_request(PROVIDER, "deadline_ms is between 1 and 86400000")),
        Some(ms) => Some(Duration::from_millis(ms)),
        None => state.fleet.drain_deadline,
    };
    let index = index.parse::<usize>().ok().filter(|&i| state.fleet.drain(i, deadline)).ok_or_else(|| ApiError::not_found(PROVIDER, "no such upstream"))?;
    Ok(Json(json!({"index": index, "draining": true, "deadline_ms": deadline.map(|d| d.as_millis() as u64)})))
}

async fn resume(State(state): State<RouteState>, Extension(principal): Extension<Principal>, Path(index): Path<String>) -> Result<Json<Value>, ApiError> {
    operator_only(&principal)?;
    let index = index.parse::<usize>().ok().filter(|&i| state.fleet.resume(i)).ok_or_else(|| ApiError::not_found(PROVIDER, "no such upstream"))?;
    Ok(Json(json!({"index": index, "draining": false})))
}

/// The model a request names, checked.
fn requested_model(head: &Head) -> Result<String, ApiError> {
    let raw = head.get("model").ok_or_else(|| ApiError::invalid_request(PROVIDER, "the request names no model"))?;
    serde_json::from_str::<String>(raw.get()).ok().filter(|m| valid_model_id(m)).ok_or_else(|| ApiError::invalid_request(PROVIDER, format!("model must be a string of 1 to {MAX_MODEL_ID_BYTES} bytes")))
}

async fn forward(State(state): State<RouteState>, Extension(principal): Extension<Principal>, headers: HeaderMap, body: Bytes, path: &'static str, keyed: Keyed) -> Response {
    let head: Head = match serde_json::from_slice(&body) {
        Ok(head) => head,
        Err(e) => return ApiError::invalid_request(PROVIDER, format!("the request body is not a JSON object: {e}")).into_response(),
    };
    let model = match requested_model(&head) {
        Ok(model) => model,
        Err(e) => return e.into_response(),
    };
    let parts = Parts::of(&head, model.clone(), keyed);
    // OpenAI's routing hint for a conversation: requests that carry the same key go to the same upstream.
    let session = match head.get("prompt_cache_key").map(|raw| serde_json::from_str::<String>(raw.get())) {
        None => None,
        Some(Ok(key)) if !key.is_empty() && key.len() <= MAX_SESSION_KEY_BYTES => Some(format!("{model}\u{1f}{key}")),
        Some(_) => return ApiError::invalid_request(PROVIDER, format!("prompt_cache_key must be a string of 1 to {MAX_SESSION_KEY_BYTES} bytes")).into_response(),
    };
    drop(head);
    // A named caller holds at most its own share in flight, as on a serving surface.
    let caller_slot = match Caller::of(Some(&principal)) {
        Some(caller) => match Arc::downcast::<Caller>(principal.clone()).ok().and_then(|c| c.begin()) {
            Some(slot) => Some(slot),
            None => return ApiError::rate_limited(PROVIDER, format!("caller {} has its whole share of the server in flight; retry when a call ends", caller.name)).into_response(),
        },
        None => None,
    };
    let accept = headers.get(header::ACCEPT).filter(|v| v.to_str().is_ok_and(|s| s.len() <= 256)).cloned();
    // One id per request, of the router's making: the upstream sees it (its logs, a later dedup) and so does the client.
    let request_id = format!("route-{}", uuid::Uuid::new_v4().simple());
    let fleet = &state.fleet;
    // Keyed only for a model some upstream lists, so the tokenizer looked up is one the fleet serves.
    if !fleet.lists(&model) {
        return if fleet.available() { ApiError::model_not_found(PROVIDER, &model).into_response() } else { ApiError::saturated(PROVIDER, "no upstream serving this model is available").into_response() };
    }
    let tokens = fleet.tokens.clone();
    let key = match tokio::task::spawn_blocking(move || RequestKey::build(&tokens, parts)).await {
        Ok(key) => key,
        Err(_) => return ApiError::internal(PROVIDER, "the request could not be keyed").into_response(),
    };
    let mut excluded = vec![false; fleet.members.len()];
    let mut resends = 0usize;
    loop {
        let (index, in_flight) = match fleet.admit(&model, &key, session.as_deref(), &excluded).await {
            Ok(chosen) => chosen,
            Err(Refusal::UnknownModel) => return ApiError::model_not_found(PROVIDER, &model).into_response(),
            Err(Refusal::NoneAvailable) => return ApiError::saturated(PROVIDER, "no upstream serving this model is available").into_response(),
            Err(Refusal::QueueFull) => return ApiError::saturated(PROVIDER, "every upstream serving this model is at capacity and the router's queue is full").into_response(),
            Err(Refusal::Deadline) => return ApiError::overloaded(PROVIDER, "no upstream slot freed within the admit deadline").into_response(),
        };
        let member = &fleet.members[index];
        let conn = match fleet.connector.checkout(&member.url, &member.pool).await {
            Ok(conn) => conn,
            Err(not_sent) => {
                // Nothing was sent: the request may go to another upstream, and this one waits for its next probe.
                fleet.mark_down(index, &not_sent.0);
                excluded[index] = true;
                continue;
            }
        };
        let mut cut = member.cut.subscribe();
        let send = tokio::time::timeout(fleet.response_timeout, conn.send(Method::POST, path, fleet.key.as_deref(), accept.as_ref(), Some(&request_id), body.clone()));
        let sent = tokio::select! {
            sent = send => sent,
            _ = cut.changed() => return cut.borrow().refusal(fleet.stall_timeout).into_response(),
        };
        return match sent {
            Ok(Ok((response, conn))) => {
                member.progressed();
                let mut out = relay(response, conn, member.clone(), cut, (in_flight, caller_slot));
                if let Ok(id) = axum::http::HeaderValue::from_str(&request_id) {
                    out.headers_mut().insert("x-request-id", id);
                }
                out
            }
            Ok(Err(upstream::SendError::NotSent(why))) => {
                // A kept-alive connection the upstream had closed: the request was handed back unsent. Try again with
                // a fresh connection, a bounded number of times.
                resends += 1;
                if resends > MAX_UNSENT_RETRIES {
                    eprintln!("brain route: upstream {index} ({}) took no request on {resends} connections: {why}", member.url);
                    return ApiError::saturated(PROVIDER, "no upstream connection took the request").into_response();
                }
                continue;
            }
            Ok(Err(upstream::SendError::Sent(why))) => {
                eprintln!("brain route: upstream {index} ({}) failed a sent request: {why}", member.url);
                ApiError::bad_gateway(PROVIDER, "the upstream failed after the request was sent; it may have run").into_response()
            }
            Err(_) => ApiError::failure(PROVIDER, capability::FailureClass::Timeout, "the upstream did not answer in time; the request may have run").into_response(),
        };
    }
}

/// How many times one request is offered to fresh connections after kept-alive ones handed it back unsent.
const MAX_UNSENT_RETRIES: usize = 4;

/// The upstream's answer as the client's: its status, the headers a client of the API reads, and its body streamed
/// through, every chunk counted as the upstream's progress. `held` (the upstream slot, the caller's slot) lives until
/// the body ends, the client hangs up or a cut (a drain's deadline, a stall) ends it; a body read to its end
/// gives `conn` back to the upstream's pool, any other end closes it.
fn relay<H: Send + 'static>(response: hyper::Response<hyper::body::Incoming>, conn: upstream::Connection, member: Arc<Member>, mut cut: tokio::sync::watch::Receiver<Cut>, held: H) -> Response {
    const PASSED: [header::HeaderName; 3] = [header::CONTENT_TYPE, header::CACHE_CONTROL, header::RETRY_AFTER];
    let (parts, incoming) = response.into_parts();
    let mut upstream_body = Body::new(incoming).into_data_stream();
    let body = async_stream::stream! {
        let _held = held;
        let mut conn = Some(conn);
        loop {
            tokio::select! {
                chunk = upstream_body.next() => match chunk {
                    Some(Ok(chunk)) => {
                        member.progressed();
                        yield Ok(chunk);
                    }
                    Some(Err(e)) => {
                        yield Err(e);
                        break;
                    }
                    None => {
                        if let Some(conn) = conn.take() {
                            member.pool.give_back(conn);
                        }
                        break;
                    }
                },
                _ = cut.changed() => {
                    let why = cut.borrow().ended();
                    yield Err(axum::Error::new(std::io::Error::other(why)));
                    break;
                }
            }
        }
    };
    let mut out = Response::new(Body::from_stream(body));
    *out.status_mut() = parts.status;
    for name in PASSED {
        if let Some(value) = parts.headers.get(&name) {
            out.headers_mut().insert(name, value.clone());
        }
    }
    out
}

/// What [`serve`] needs beyond the fleet and the addresses.
#[derive(Default)]
pub struct RouteOpts {
    /// Stops serving; `None` installs one from this process's signals.
    pub shutdown: Option<brain_shutdown::Shutdown>,
    /// Notified once per address, after it binds.
    pub ready: brain_shutdown::ready::Gate,
    /// What one peer may hold of the surface.
    pub limits: ConnectionLimits,
    /// Serve over TLS with this certificate; plaintext only on loopback ([`crate::tls::check_exposure`]).
    pub tls: Option<Tls>,
    /// Where to publish the connection record once every address is bound, and the key it names (empty for none).
    pub record: Option<(PathBuf, String)>,
}

/// Serve `fleet` on `listener` under `authenticator` until `shutdown` fires, probing the upstreams meanwhile.
pub async fn run(fleet: Arc<Fleet>, listener: tokio::net::TcpListener, authenticator: Arc<dyn Authenticator>, limits: ConnectionLimits, tls: Option<Tls>, shutdown: brain_shutdown::Shutdown) {
    let probing = tokio::spawn(fleet.clone().keep_probing(shutdown.clone()));
    listener::serve(listener, router(fleet, authenticator), limits, tls, shutdown).await;
    probing.abort();
}

/// Serve `fleet` on every address in `addrs` until the shutdown fires. Blocks the calling thread. The upstreams are
/// probed once before any address binds, so a client that finds the server ready finds the upstreams that answered
/// already in rotation.
///
/// # Errors
/// An address that cannot be bound, or a runtime that cannot start.
pub fn serve(fleet: Arc<Fleet>, addrs: Vec<SocketAddr>, authenticator: Arc<dyn Authenticator>, opts: RouteOpts) -> anyhow::Result<()> {
    let shutdown = opts.shutdown.unwrap_or_else(brain_shutdown::Shutdown::from_signals);
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    let result = rt.block_on(async {
        fleet.probe().await;
        let up = fleet.members.iter().filter(|m| m.healthy.load(Ordering::SeqCst)).count();
        eprintln!("brain route: {up} of {} upstreams in rotation", fleet.members.len());
        // A connection over the listener's cap is closed unanswered: the cap must hold every request the fleet runs
        // and the router queues, so that overload is answered with 503 and Retry-After, not a reset.
        let mut limits = opts.limits;
        limits.max_connections = limits.max_connections.max(fleet.connections_needed());
        eprintln!("brain route: up to {} client connections (the upstreams' decode slots and --route-queue, with room to answer the rest)", limits.max_connections);
        let mut bound = Vec::new();
        let mut set = tokio::task::JoinSet::new();
        let scheme = if opts.tls.is_some() { "https" } else { "http" };
        for addr in addrs {
            let listener = tokio::net::TcpListener::bind(addr).await.map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;
            let addr = listener.local_addr().unwrap_or(addr);
            eprintln!("brain apiserve: {PROVIDER} on {scheme}://{addr} (routing to {} upstreams)", fleet.members.len());
            opts.ready.bound(PROVIDER.as_str());
            bound.push(addr);
            set.spawn(run(fleet.clone(), listener, authenticator.clone(), limits, opts.tls.clone(), shutdown.clone()));
        }
        if let Some((path, key)) = &opts.record {
            let entries: Vec<(Provider, SocketAddr, String)> = bound.iter().map(|&a| (PROVIDER, a, key.clone())).collect();
            let record = crate::surface::record_json(std::process::id(), scheme, &entries).to_string();
            if let Err(e) = path.parent().map_or(Ok(()), crate::surface::private_dir).and_then(|()| crate::surface::write_private(path, record.as_bytes())) {
                eprintln!("brain serve: the connection record {} could not be written ({e}); local clients will not find this server", path.display());
            }
        }
        let drain_expired = async {
            shutdown.wait().await;
            tokio::time::sleep(crate::SHUTDOWN_DRAIN).await;
        };
        tokio::pin!(drain_expired);
        loop {
            tokio::select! {
                joined = set.join_next() => if joined.is_none() { break },
                _ = &mut drain_expired => {
                    set.abort_all();
                    break;
                }
            }
        }
        Ok(())
    });
    rt.shutdown_background();
    if let Some((path, _)) = &opts.record {
        std::fs::remove_file(path).ok();
    }
    result
}

/// The upstream URLs in `text`: one per line; blank lines and lines starting with `#` are skipped.
///
/// # Errors
/// The first line that is not an upstream URL, with its line number.
pub fn parse_upstreams_file(text: &str) -> Result<Vec<UpstreamUrl>, String> {
    text.lines()
        .enumerate()
        .map(|(n, line)| (n + 1, line.trim()))
        .filter(|(_, line)| !line.is_empty() && !line.starts_with('#'))
        .map(|(n, line)| UpstreamUrl::parse(line).map_err(|e| format!("line {n}: {e}")))
        .collect()
}
