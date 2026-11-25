// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for training a served policy online, on the rollout groups an orchestrator
// produces while the same server serves it, for its clients. If your team needs expertise in asynchronous
// policy-gradient training with bounded staleness and backpressure then you can procure our services by sending an
// email to info@swedishembedded.com.

//! [`RolloutStream`]: rollout groups handed to a running policy-gradient fine-tune.
//!
//! A [`ChatFineTune`] that trains on a stream is opened with [`ChatFineTune::open_rollout_stream`], which checks the
//! run's base, template and policy-gradient settings once, and run with [`ChatFineTune::rollout_stream`] set. While it
//! runs, another thread hands it records ([`RolloutStream::ingest`]): training-record contract v2 rollout records, a
//! request's records forming complete groups. Each is read, rendered and checked exactly as the batch path checks a
//! dataset (the base's tokenizer digest included), each group's advantages are computed on arrival, and the groups join
//! the run's queue ([`rl::objective::stream_pg`]). [`RolloutStream::finish`] ends the run once what is queued is
//! trained. See [`rl::objective::stream_pg`] for staleness, backpressure and receipts.

use std::sync::Arc;

use data::chat::ChatSample;
use data::chat_template::ChatTemplate;
use data::qwen_tokenizer::QwenBpe;
use rl::objective::stream_pg::{groups_of_split, Admission, FeedRefusal, RolloutFeed, StreamSettings};

use crate::finetune::Asked;
use crate::{ChatFineTune, Error, PolicyGradientConfig, Result};

/// The step count of a stream with no end in view: its learning rate stays at its peak after the warmup.
pub const OPEN_ENDED_STREAM_STEPS: u32 = 1_000_000;
/// The longest warmup a stream takes, in steps: its length is not known in advance, so the warmup is not a share of it.
pub const STREAM_MAX_WARMUP: u32 = 10;

/// The most records one [`RolloutStream::ingest`] takes.
pub const MAX_STREAM_RECORDS: usize = 4096;

/// Why [`RolloutStream::ingest`] took nothing.
#[derive(Clone, Debug, PartialEq)]
pub enum StreamRefusal {
    /// A record breaks the contract or is one the stream does not train; the message names it.
    Invalid(String),
    /// The run's queue is full ([`FeedRefusal::Full`]).
    Full { queued_tokens: usize, allowance: usize, offered_tokens: usize },
    /// The run was told to finish and takes no more records.
    Closed,
    /// The run is not ready for records yet.
    Starting,
}

/// The run's end of a stream of rollout groups. See the module doc.
pub struct RolloutStream {
    feed: Arc<RolloutFeed>,
    config: PolicyGradientConfig,
    tok: QwenBpe,
    tmpl: ChatTemplate,
    no_think: Option<String>,
    render: data::chat::RenderOpts,
    pack: bool,
}

impl std::fmt::Debug for RolloutStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RolloutStream").field("settings", self.feed.settings()).finish_non_exhaustive()
    }
}

/// The run a [`ChatFineTune::rollout_stream`] trains from: the feed alone, so the builder stays a plain value.
#[derive(Clone)]
pub(crate) struct StreamRef(pub(crate) Arc<RolloutFeed>);

impl std::fmt::Debug for StreamRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("StreamRef").field(self.0.settings()).finish()
    }
}

impl RolloutStream {
    /// Read the JSONL records of `jsonl`, which form complete groups, and queue the groups this run has not taken.
    ///
    /// # Errors
    /// [`StreamRefusal::Invalid`] for a record the batch path would refuse, or one a stream does not train (see
    /// [`rl::objective::stream_pg::groups_of_split`]); nothing of the request is queued then. [`StreamRefusal::Full`]
    /// and [`StreamRefusal::Closed`] from the queue.
    pub fn ingest(&self, jsonl: &[u8]) -> std::result::Result<Admission, StreamRefusal> {
        let invalid = StreamRefusal::Invalid;
        let text = std::str::from_utf8(jsonl).map_err(|e| invalid(format!("the records are not UTF-8: {e}")))?;
        let lines: Vec<(usize, &str)> = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return Err(invalid("no records".to_string()));
        }
        if lines.len() > MAX_STREAM_RECORDS {
            return Err(invalid(format!("{} records; at most {MAX_STREAM_RECORDS} in one request", lines.len())));
        }
        let images = data::chat::TextOnly::TRAINER;
        let records = lines
            .iter()
            .map(|(at, line)| ChatSample::from_json_line_with(line, &images).map_err(|e| invalid(format!("line {}: {e}", at + 1))))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if let Some(i) = records.iter().position(|r| r.rollout.is_none()) {
            return Err(invalid(format!("line {}: carries no \"reward\" and \"group\"; a streamed policy gradient trains rollout records only", lines[i].0 + 1)));
        }
        let asked = Asked { tok: &self.tok, tmpl: &self.tmpl, no_think: self.no_think.as_deref(), render: self.render, pack: self.pack, images: &images };
        let samples = asked.samples(&records, "the records").map_err(|e| invalid(e.to_string()))?;
        crate::finetune::check_rollouts_together(&samples, &[], &[], self.config.truncated, self.tok.digest()).map_err(|e| invalid(e.to_string()))?;
        let split = data::chat::encode_sample_split_while(&samples, &self.tok, &self.tmpl, self.render, &mut || true).map_err(|e| invalid(e.to_string()))?.expect("encoding is not stopped");
        let groups = groups_of_split(&split, &self.config, self.feed.settings()).map_err(invalid)?;
        self.feed.offer(groups).map_err(|refusal| match refusal {
            FeedRefusal::Full { queued_tokens, allowance, offered_tokens } => StreamRefusal::Full { queued_tokens, allowance, offered_tokens },
            FeedRefusal::Closed => StreamRefusal::Closed,
            FeedRefusal::Starting => StreamRefusal::Starting,
        })
    }

    /// No more records: the run trains what is queued and ends.
    pub fn finish(&self) -> Admission {
        self.feed.finish()
    }

    /// The policy version the job's adapter was last published as: what staleness is measured against.
    pub fn set_version(&self, version: u64) {
        self.feed.set_version(version);
    }

    /// Wait at most `timeout` for the run to be ready for records; true once it is.
    pub fn wait_ready(&self, timeout: std::time::Duration) -> bool {
        self.feed.wait_open(timeout)
    }

    /// The policy version staleness is measured against now.
    pub fn version(&self) -> u64 {
        self.feed.version()
    }

    /// The `applied`, `committed` and `dropped` reports of the run since the last call, oldest first.
    pub fn drain_events(&self) -> Vec<serde_json::Value> {
        self.feed.drain_events()
    }

    pub fn settings(&self) -> StreamSettings {
        *self.feed.settings()
    }

    /// `sha256:<hex>` of the base's tokenizer file, which every record's `tokenizer_digest` must name; `None` when
    /// the base's tokenizer has no file to digest.
    pub fn tokenizer_digest(&self) -> Option<&str> {
        self.tok.digest()
    }
}

impl ChatFineTune {
    /// The stream this fine-tune would train from: its base's tokenizer and template and its policy-gradient settings,
    /// checked now. `step_tokens` policy tokens make one optimiser step, the queue holds `queue_tokens`, a record more
    /// than `max_staleness` published versions old is dropped, `top_k` is the sampler top-K every record carries
    /// for the score-centred estimator, and `support` the widest sampler support a record may carry per token (the
    /// trainer renormalises its log-probabilities over it); the training row is [`ChatFineTune::max_block`], which a stream must name. The
    /// learner waiting for records stops once `keep_going` says no. Between steps it lends its device through
    /// `lease` (the job's [`capability::Invocation::lease`]; [`Default`] when it runs on no shared server), so the
    /// server can serve the policy the records are drawn from on the same device.
    ///
    /// # Errors
    /// The base is not found, its architecture does not declare the policy-gradient objective, the settings are ones a
    /// stream cannot train with ([`rl::objective::stream_pg::check_streamable`]), or no `max_block` is set.
    #[allow(clippy::too_many_arguments)]
    pub fn open_rollout_stream(&self, step_tokens: usize, queue_tokens: usize, max_staleness: Option<u64>, top_k: Option<usize>, support: Option<usize>, keep_going: impl Fn() -> bool + Send + Sync + 'static, lease: capability::lease::DeviceLease) -> Result<RolloutStream> {
        let (located, config) = self.stream_setup(top_k)?;
        let block = self.max_block.ok_or_else(|| Error::InvalidRequest("ChatFineTune: a rollout stream trains rows of max_block tokens; set max_block".to_string()))?;
        let settings = StreamSettings { step_tokens, queue_tokens, max_staleness, block: block as usize, top_k, support };
        let feed = RolloutFeed::new(settings, keep_going, lease).map_err(|e| Error::InvalidRequest(format!("ChatFineTune: rollout stream: {e}")))?;
        // A run with no saved state has no consumed keys to restore: it takes records at once. A resumed one takes them
        // once its learner has restored the state.
        if !self.saved_state().is_some_and(|state| state.exists()) {
            feed.open();
        }
        let (tok, tmpl) = crate::finetune::tokenizer_and_template(&located.model_dir)?;
        let no_think = if self.thinking { None } else { tmpl.no_think_block() };
        let render = data::chat::RenderOpts { keep_reasoning: no_think.is_some() };
        Ok(RolloutStream { feed, config, tok, tmpl, no_think, render, pack: self.pack_answers })
    }

    /// Train on `stream` (from [`Self::open_rollout_stream`]) instead of a dataset: the run takes steps as the stream
    /// fills and ends when it is finished, or at [`Self::steps`].
    pub fn rollout_stream(mut self, stream: &RolloutStream) -> Self {
        self.stream = Some(StreamRef(Arc::clone(&stream.feed)));
        self
    }
}
