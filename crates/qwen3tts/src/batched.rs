// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Qwen3-TTS decoded many utterances at a time on the shared paged decoder.
//!
//! The Talker is a Qwen3 decoder whose inputs are embedding rows (the summed codebook
//! embeddings of the last frame plus a text row) and whose head reads codebook 0; the code
//! predictor is a five-layer Qwen3 decoder run for sixteen positions per frame, each
//! predicting the next codebook from its own head. Both run on [`qwen3::serve::Engine`]:
//! the paged KV pool, its batched decode and its kernel selection are the text models',
//! so every stream in flight advances together - one Talker step and sixteen code-predictor
//! steps per frame for all of them, not per stream.
//!
//! A prompt's prefix is one chunked prefill ([`qwen3::serve::Engine::prefill_mixed`]), not one
//! decode step per position. Codebook 0 is drawn on the host from the device's logits with
//! the reference's suppression, repetition penalty and filters ([`crate::sampling::sample_cb0`]);
//! the residual codebooks are drawn where their logits are, and the code predictor's fifteen steps run
//! as one chained window ([`qwen3::serve::Engine::forward_batched_embed_chain`]): each step reads its
//! own head, and each drawn code, offset into the stacked input table, is the next step's input on the
//! device, so a frame's residual codes cross to the host once.
//!
//! Streams join and leave between frames ([`BatchedDecoder::run`]): an engine serving a queue
//! admits a new request as soon as one finishes, and each finished stream's codes are handed
//! out at once.
//!
//! Swedish Embedded AB implements high-throughput speech synthesis services for its clients.
//! If your team needs expertise in serving neural speech on GPUs at scale then you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::collections::HashMap;
use std::sync::Arc;

use capability::CancelToken;
use gpu_core::Gpu;
use model::paged::BlockTable;
use model::serve::{draw_key, RowDraw};
use qwen3::serve::{ChainStep, Engine};
use qwen3::{Dtype, QwenConfig};

use crate::config::MtpConfig;
use crate::genconfig::GenerationPlan;
use crate::npu_gen::TalkerTables;
use crate::prompt::{MtpHost, Prompt, TalkerHost, TtsSpecials};
use crate::sampling::{sample_cb0, SamplerCfg};
use crate::{GenOpts, TtsPaths};

/// Codebooks per codec frame.
const CODEBOOKS: usize = 16;
/// Positions per KV block of both decoders; the code predictor's sixteen positions are one block.
const BLOCK: u32 = 16;
/// The longest Talker context (prompt plus frames) a stream may hold, in positions.
pub const MAX_CONTEXT: u32 = 2048;
/// Positions one prefill chunk carries.
const PREFILL_CHUNK: u32 = 512;
/// The precision the decoders' weights are held in on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightTier {
    /// As the checkpoint stores each tensor (bf16 for the published Qwen3-TTS checkpoints): nothing that was
    /// published is rounded, and the products take the half-precision tensor-core GEMMs.
    Stored,
    /// fp32 throughout: the precision the per-stream reference decoders compute in.
    F32,
}

/// What admitting a job came to.
enum Admission {
    Started(Stream),
    /// The pool cannot hold its whole utterance beside the live streams: it waits for one to finish.
    Wait(Job),
}

/// The code predictor's residual-codebook input tables, `[vocab, embedding_dim]` each, kept on
/// the host for the prompt builders and the feedback sum.
pub struct ResidualTables {
    tables: Vec<Vec<f32>>,
    e: usize,
}

impl MtpHost for ResidualTables {
    fn codec_embed(&self, residual_idx: usize, code: u32) -> &[f32] {
        let s = code as usize * self.e;
        &self.tables[residual_idx][s..s + self.e]
    }
}

/// One utterance to decode: its prompt and how to sample it.
pub struct Job {
    pub prompt: Prompt,
    pub opts: GenOpts,
    pub cancel: CancelToken,
}

/// A stream in flight.
struct Stream {
    index: usize,
    /// The Talker blocks reserved for this stream's whole utterance.
    blocks: u32,
    table: BlockTable,
    prompt: Prompt,
    plan: GenerationPlan,
    max_frames: usize,
    min_new: usize,
    seed: u64,
    rng: data::rng::Rng,
    cancel: CancelToken,
    history: Vec<u32>,
    frames: Vec<u32>,
    hidden: Vec<f32>,
    cb0: u32,
}

/// The Talker and the code predictor on the shared paged decoder, with the host tables the
/// prompt builders read.
pub struct BatchedDecoder {
    talker: Engine,
    mtp: Engine,
    tables: Arc<TalkerTables>,
    residual: Arc<ResidualTables>,
    /// `small_to_mtp_projection` (`[d_mtp, e]`, bias `[d_mtp]`) when the code predictor is
    /// narrower than the Talker (the 1.7B family).
    projection: Option<(Vec<f32>, Vec<f32>)>,
    d_mtp: usize,
    /// The code predictor's head of each chain step (step `s` predicts codebook `s + 1`).
    heads: Vec<String>,
    /// What each step adds to the code it draws to make the next step's input id ([`qwen3::serve::ChainStep`]).
    feed_offsets: Vec<u32>,
    max_streams: usize,
    /// Talker KV blocks the live streams may still grow into, so a stream is admitted only
    /// when its whole utterance fits beside them and the pool never runs dry mid-utterance.
    reserved: u32,
    /// The Talker pool's blocks.
    pool: u32,
}

/// A `[n, d]` table of `c` as a flat vector, or the error naming it.
fn take(c: &checkpoint::Container, name: &str) -> Result<Vec<f32>, String> {
    c.find(name, "").cloned().ok_or_else(|| format!("tts: the checkpoint has no {name}"))
}

/// The Talker KV blocks a decoder for `max_streams` streams holds: each stream at the longest
/// utterance it may ask for. A stream reserves its whole utterance when it is admitted, so a smaller
/// pool admits fewer streams than the decoder batches (measured: 12 of 16 at the served frame cap).
fn pool_blocks(max_streams: usize) -> u32 {
    max_streams as u32 * (MAX_CONTEXT / BLOCK)
}

/// The device bytes the Talker's KV pool takes for `max_streams` streams, beside the weights;
/// `None` when the Talker's config cannot be read.
pub fn kv_pool_bytes(paths: &TtsPaths, max_streams: usize) -> Option<u64> {
    let cfg = QwenConfig::from_json_checked(&crate::import::open_talker(&paths.talker).ok()?.config()).ok()?;
    Some(qwen3::serve::kv_pool_bytes(&cfg, BLOCK, pool_blocks(max_streams.max(1)), false))
}

/// `rows` (`[n, e]`) through `small_to_mtp_projection` (`[d_mtp, e]`, bias `[d_mtp]`) when there is one.
fn project(rows: Vec<f32>, projection: Option<&(Vec<f32>, Vec<f32>)>, e: usize, d_mtp: usize) -> Vec<f32> {
    let Some((w, b)) = projection else { return rows };
    let n = rows.len() / e;
    let mut out = model::hostmath::linear_rows(&rows, w, n, e, d_mtp);
    for row in out.chunks_mut(d_mtp) {
        for (x, bias) in row.iter_mut().zip(b) {
            *x += bias;
        }
    }
    out
}

impl BatchedDecoder {
    /// Load both decoders on `gpu`'s device for up to `max_streams` streams at once, their weights at `tier`.
    pub fn load_on(gpu: &Gpu, paths: &TtsPaths, max_streams: usize, tier: WeightTier) -> Result<BatchedDecoder, String> {
        let max_streams = max_streams.max(1);
        // The Talker streams from its reader tensor by tensor; nothing is held as fp32 beside the upload.
        let talker_reader = crate::import::open_talker(&paths.talker)?;
        let mut tcfg = QwenConfig::from_json_checked(&talker_reader.config())?;
        // Read through the checkpoint's packed view, so a bf16 tensor reaches its tier as stored.
        let talker_src = checkpoint::packed::stored_source(talker_reader);
        tcfg.block_size = MAX_CONTEXT;
        let per_seq = MAX_CONTEXT / BLOCK;
        let pool = pool_blocks(max_streams);
        let talker_tiers = match tier {
            WeightTier::Stored => model::ops::stored_tier_policy(&*talker_src, tcfg.param_list().iter().map(|(name, _)| name.as_str())),
            WeightTier::F32 => model::ops::TierPolicy::uniform(Dtype::F32),
        };
        let mut talker = Engine::from_source_tiers_on(gpu, tcfg, &*talker_src, BLOCK, pool, max_streams as u32, per_seq, PREFILL_CHUNK, false, &talker_tiers);
        // Every row count the streams pass through replays a recorded step; recording one while streams decode would
        // stall them, and recording a wider one after a narrower drops the narrower ones.
        talker.record_embed_decode_tapes(max_streams as u32);
        drop(talker_src);

        let tables_host = TalkerTables::load(&paths.talker);
        let mtp_ckpt = crate::import::load_mtp(&paths.mtp);
        let mc = MtpConfig::from_brain_json(&mtp_ckpt.header["config"]);
        let nres = mc.n_residual() as usize;
        let (d_mtp, e) = (mc.d_model as usize, mc.embedding_dim as usize);
        let tables = (0..nres).map(|i| take(&mtp_ckpt, &format!("codec_embedding.{i}.weight"))).collect::<Result<Vec<_>, _>>()?;
        let projection = match mtp_ckpt.find("small_to_mtp_projection.weight", "") {
            Some(w) => Some((w.clone(), take(&mtp_ckpt, "small_to_mtp_projection.bias")?)),
            None => None,
        };
        // Positions 1..=15 of a frame read a row of one of these tables (codebook 0's, then the residual tables of
        // codebooks 1..14), so they are stacked into the code predictor's input table, already projected to its
        // width: a drawn code plus its table's offset is the next position's input id, and the whole chain runs on
        // the device.
        let cb0_rows = tables_host.vocab();
        let mut stacked: Vec<f32> = Vec::with_capacity((cb0_rows + (nres - 1) * mc.vocab as usize) * e);
        for id in 0..cb0_rows as u32 {
            stacked.extend_from_slice(tables_host.codec_embed(id));
        }
        for table in &tables[..nres - 1] {
            stacked.extend_from_slice(table);
        }
        let stacked = project(stacked, projection.as_ref(), e, d_mtp);
        let feed_offsets: Vec<u32> = (0..nres).map(|s| if s + 1 < nres { (cb0_rows + s * mc.vocab as usize) as u32 } else { 0 }).collect();
        let mcfg = QwenConfig {
            vocab: (stacked.len() / d_mtp) as u32,
            block_size: CODEBOOKS as u32,
            n_layers: mc.n_layers,
            d_model: mc.d_model,
            n_heads: mc.n_heads,
            n_kv_heads: mc.n_kv_heads,
            head_dim: mc.head_dim,
            d_ff: mc.d_ff,
            rope_theta: mc.rope_theta,
            rms_eps: mc.rms_norm_eps,
            max_position_embeddings: CODEBOOKS as u32,
            tie_embeddings: false,
            qk_norm: true,
            attn_bias: false,
            rope_scaling: None,
            lora: None,
        };
        let mut mtp_map: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, _) in crate::mtp::MtpModel::decoder_param_list(&mc) {
            mtp_map.insert(name.clone(), take(&mtp_ckpt, &name)?);
        }
        mtp_map.insert("tok.weight".into(), stacked);
        // Every head is a named one, read at its own position; the engine's own head is never read.
        let heads: Vec<String> = (0..nres).map(|k| format!("lm_head.{k}.weight")).collect();
        for name in &heads {
            mtp_map.insert(name.clone(), take(&mtp_ckpt, name)?);
        }
        // The code predictor's tensors are held as its checkpoint stores its heads.
        let mtp_tier = match tier {
            WeightTier::Stored => {
                let src = checkpoint::packed::stored_source(crate::import::open_mtp(&paths.mtp)?);
                model::ops::stored_tier(&*src, "lm_head.0.weight").unwrap_or(Dtype::F32)
            }
            WeightTier::F32 => Dtype::F32,
        };
        let mut mtp = Engine::from_map_tier_on(gpu, mcfg, &mtp_map, BLOCK, max_streams as u32, max_streams as u32, 1, CODEBOOKS as u32, false, mtp_tier);
        let names: Vec<&str> = heads.iter().map(String::as_str).collect();
        mtp.add_heads(&mtp_map, &names, mc.vocab as usize)?;
        mtp.set_window_capacity(nres);
        mtp.record_embed_decode_tapes(max_streams as u32);
        drop(mtp_map);

        Ok(BatchedDecoder {
            talker,
            mtp,
            tables: Arc::new(tables_host),
            residual: Arc::new(ResidualTables { tables, e }),
            projection,
            d_mtp,
            heads,
            feed_offsets,
            max_streams,
            reserved: 0,
            pool,
        })
    }

    /// The Talker's host tables, for the prompt builders.
    pub fn talker_tables(&self) -> &TalkerTables {
        &self.tables
    }

    /// The code predictor's residual tables, for the prompt builders.
    pub fn residual_tables(&self) -> &ResidualTables {
        &self.residual
    }

    /// The codebook-0 logits after `prompt` (`[vocab]`): what the first frame is drawn from. The prompt's
    /// positions are released again; nothing a stream holds is touched.
    pub fn prompt_logits(&mut self, prompt: &Prompt) -> Vec<f32> {
        let d = self.tables.d();
        let mut table = BlockTable::new();
        let rows: Vec<qwen3::model::PrefillInput> = prompt.embeds.chunks(d).map(qwen3::model::PrefillInput::Embed).collect();
        let hidden = self.talker.prefill_mixed(&mut table, &rows);
        self.talker.release_table(&mut table);
        self.tables.codec_head_logits(&hidden)
    }

    /// The tiers the Talker's and the code predictor's weights are held in.
    pub fn weights_tiers(&self) -> (Dtype, Dtype) {
        (self.talker.weights_tier(), self.mtp.weights_tier())
    }

    /// Both host tables, shared: what a caller building prompts while [`Self::run`] holds the
    /// decoder keeps.
    pub fn hosts(&self) -> (Arc<TalkerTables>, Arc<ResidualTables>) {
        (self.tables.clone(), self.residual.clone())
    }

    /// The most streams decoded at once.
    pub fn max_streams(&self) -> usize {
        self.max_streams
    }

    /// The Talker positions a prompt with `prefix` rows and `max_frames` frames needs.
    pub fn fits(&self, prefix: usize, max_frames: usize) -> bool {
        prefix + max_frames < self.talker.max_seq_len()
    }

    /// `rows` (`[n, e]`) as the code predictor's inputs (`[n, d_mtp]`).
    fn to_mtp(&self, rows: Vec<f32>) -> Vec<f32> {
        project(rows, self.projection.as_ref(), self.residual.e, self.d_mtp)
    }

    /// Start `job` as stream `index`: prefill its prompt and draw its first codebook-0 token.
    fn admit(&mut self, index: usize, job: Job, sp: &TtsSpecials) -> Result<Admission, String> {
        let _range = gpu_core::markers::range(c"tts.admit");
        let d = self.tables.d();
        let prefix = job.prompt.embeds.len() / d;
        if !self.fits(prefix, job.opts.max_frames) {
            return Err(capability::failure::tag(
                capability::failure::FailureClass::InvalidRequest,
                format!("a prompt of {prefix} positions with up to {} frames does not fit the {} a stream may hold", job.opts.max_frames, self.talker.max_seq_len()),
            ));
        }
        let blocks = self.talker.blocks_for((prefix + job.opts.max_frames + 1) as u32);
        if blocks > self.pool {
            return Err(capability::failure::tag(capability::failure::FailureClass::InvalidRequest, format!("the utterance needs {blocks} KV blocks; the pool holds {}", self.pool)));
        }
        if self.reserved + blocks > self.pool {
            return Ok(Admission::Wait(job));
        }
        self.reserved += blocks;
        let plan = job.opts.plan();
        let mut table = BlockTable::new();
        let rows: Vec<qwen3::model::PrefillInput> = job.prompt.embeds.chunks(d).map(qwen3::model::PrefillInput::Embed).collect();
        let hidden = self.talker.prefill_mixed(&mut table, &rows);
        let mut rng = data::rng::Rng::new(job.opts.seed);
        let logits = self.tables.codec_head_logits(&hidden);
        let draw = sample_cb0(logits, sp.codec_eos, job.opts.min_new == 0, &plan.cb0, &[], &mut rng);
        Ok(Admission::Started(Stream {
            index,
            blocks,
            table,
            plan,
            max_frames: job.opts.max_frames,
            min_new: job.opts.min_new,
            seed: job.opts.seed,
            rng,
            cancel: job.cancel,
            history: Vec::new(),
            frames: Vec::new(),
            hidden,
            cb0: draw.token,
            prompt: job.prompt,
        }))
    }

    /// The draw of codebook `k` at `frame` of a stream sampled by `cfg` under `seed`: a keyed draw where the
    /// logits are, or, unsampled, the one that keeps the best code (the reference's argmax, ties to the lower id).
    fn residual_draw(cfg: &SamplerCfg, seed: u64, frame: usize, k: usize) -> RowDraw {
        if !cfg.do_sample || cfg.temperature <= 0.0 {
            return RowDraw { temp: 1.0, min_p: 0.0, top_p: 1.0, top_k: 1, top: 0, key: 0 };
        }
        RowDraw { temp: cfg.temperature, min_p: 0.0, top_p: cfg.top_p, top_k: cfg.top_k, top: 0, key: draw_key(seed, frame as u64, k as u32) }
    }

    /// One frame of every stream in `live`: the code predictor's chain for all of them, the
    /// feedback rows, one Talker step and the next codebook-0 draw. Returns the streams that
    /// finished this frame.
    fn step(&mut self, live: &mut Vec<Stream>, sp: &TtsSpecials) -> Vec<(Stream, Result<(), String>)> {
        let _frame = gpu_core::markers::range(c"tts.frame");
        let d = self.tables.d();
        // Frames end here before any work: a cancelled stream, or one whose codebook 0 says it is done.
        let mut done = Vec::new();
        let mut i = 0;
        while i < live.len() {
            let s = &live[i];
            let at = s.frames.len() / CODEBOOKS;
            let finished = (s.cb0 == sp.codec_eos && at >= s.min_new) || at >= s.max_frames;
            if s.cancel.is_cancelled() || finished {
                let mut s = live.swap_remove(i);
                self.talker.release_table(&mut s.table);
                self.reserved -= s.blocks;
                let result = if s.cancel.is_cancelled() { Err("cancelled".to_string()) } else { Ok(()) };
                done.push((s, result));
            } else {
                i += 1;
            }
        }
        if live.is_empty() {
            return done;
        }
        let b = live.len();

        // The code predictor: position 0 is the Talker's hidden state, position k (1..=15) the
        // embedding of codebook k-1, read by head k-1 to predict codebook k; the whole chain is one
        // submission, each drawn code feeding the next position on the device.
        let chain_range = gpu_core::markers::range(c"tts.mtp_chain");
        let mut mtp_tables: Vec<BlockTable> = (0..b).map(|_| BlockTable::new()).collect();
        let hidden: Vec<f32> = live.iter().flat_map(|s| s.hidden.iter().copied()).collect();
        let first: Vec<u32> = live.iter().map(|s| s.cb0).collect();
        let draws: Vec<Vec<RowDraw>> = live.iter().map(|s| (1..CODEBOOKS).map(|k| Self::residual_draw(&s.plan.subtalker, s.seed, s.frames.len() / CODEBOOKS, k)).collect()).collect();
        let chain: Vec<ChainStep<'_>> = self.heads.iter().zip(&self.feed_offsets).map(|(head, &feed_offset)| ChainStep { head, feed_offset }).collect();
        let codes_drawn = {
            let input = self.to_mtp(hidden);
            let mut refs: Vec<&mut BlockTable> = mtp_tables.iter_mut().collect();
            self.mtp.forward_batched_embed_chain(&mut refs, &input, &first, &chain, &draws)
        };
        for t in mtp_tables.iter_mut() {
            self.mtp.release_table(t);
        }
        drop(chain_range);
        let feed_range = gpu_core::markers::range(c"tts.feedback");
        let mut feed: Vec<f32> = live.iter().flat_map(|s| self.tables.codec_embed(s.cb0).iter().copied()).collect();
        let mut codes = vec![vec![0u32; CODEBOOKS - 1]; b];
        for (r, row) in codes_drawn.iter().enumerate() {
            for (k, &code) in row.iter().enumerate() {
                codes[r][k] = code;
                for (f, x) in feed[r * d..(r + 1) * d].iter_mut().zip(self.residual.codec_embed(k, code)) {
                    *f += x;
                }
            }
        }

        // Commit the frame, then the feedback row: the frame's codebook sum plus the next text row.
        for (r, s) in live.iter_mut().enumerate() {
            let at = s.frames.len() / CODEBOOKS;
            s.history.push(s.cb0);
            s.frames.push(s.cb0);
            s.frames.extend_from_slice(&codes[r]);
            let n_trailing = s.prompt.trailing.len() / d;
            let text = if at < n_trailing { &s.prompt.trailing[at * d..(at + 1) * d] } else { &s.prompt.tts_pad[..] };
            for (f, x) in feed[r * d..(r + 1) * d].iter_mut().zip(text) {
                *f += x;
            }
        }
        drop(feed_range);
        let talker_range = gpu_core::markers::range(c"tts.talker");
        let (hidden, logits) = {
            let mut refs: Vec<&mut BlockTable> = live.iter_mut().map(|s| &mut s.table).collect();
            self.talker.forward_batched_embed_hidden_logits(&mut refs, &feed, None)
        };
        drop(talker_range);
        let _sample = gpu_core::markers::range(c"tts.sample");
        let v = logits.len() / b;
        for (r, s) in live.iter_mut().enumerate() {
            s.hidden = hidden[r * d..(r + 1) * d].to_vec();
            let at = s.frames.len() / CODEBOOKS;
            let draw = sample_cb0(logits[r * v..(r + 1) * v].to_vec(), sp.codec_eos, at >= s.min_new, &s.plan.cb0, &s.history, &mut s.rng);
            s.cb0 = draw.token;
        }
        done
    }

    /// Decode a queue of jobs: `next(free)` hands over up to `free` new jobs between frames
    /// (each with the index its result is answered under), `on_frames(index, codes)` sees a
    /// stream's codes as they grow, and `finish(index, result)` its whole codes when it ends.
    /// Returns once `next` has nothing more and every stream has finished.
    pub fn run(
        &mut self,
        sp: &TtsSpecials,
        next: &mut dyn FnMut(usize) -> Vec<(usize, Job)>,
        on_frames: &mut dyn FnMut(usize, &[u32]),
        finish: &mut dyn FnMut(usize, Result<Vec<u32>, String>),
    ) {
        let mut live: Vec<Stream> = Vec::new();
        let mut waiting: std::collections::VecDeque<(usize, Job)> = std::collections::VecDeque::new();
        loop {
            let free = self.max_streams - live.len();
            if waiting.len() < free {
                waiting.extend(next(free - waiting.len()));
            }
            while live.len() < self.max_streams {
                let Some((index, job)) = waiting.pop_front() else { break };
                match self.admit(index, job, sp) {
                    Ok(Admission::Started(s)) => live.push(s),
                    Ok(Admission::Wait(job)) => {
                        waiting.push_front((index, job));
                        break;
                    }
                    Err(e) => finish(index, Err(e)),
                }
            }
            if live.is_empty() {
                if waiting.is_empty() {
                    return;
                }
                continue;
            }
            for (s, result) in self.step(&mut live, sp) {
                finish(s.index, result.map(|()| s.frames));
            }
            for s in &live {
                on_frames(s.index, &s.frames);
            }
        }
    }
}
