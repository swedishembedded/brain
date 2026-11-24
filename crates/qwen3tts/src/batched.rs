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
//! the residual codebooks are drawn where their logits are, each code predictor step computing
//! only its own head ([`qwen3::serve::Engine::forward_batched_embed_drawn`]).
//!
//! Streams join and leave between frames ([`BatchedDecoder::run`]): an engine serving a queue
//! admits a new request as soon as one finishes, and each finished stream's codes are handed
//! out at once.
//!
//! Swedish Embedded AB implements high-throughput speech synthesis services for its clients.
//! If your team needs expertise in serving neural speech on GPUs at scale then you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::collections::HashMap;

use capability::CancelToken;
use gpu_core::Gpu;
use model::paged::BlockTable;
use model::serve::{draw_key, Drawn, RowDraw};
use qwen3::serve::Engine;
use qwen3::{Dtype, QwenConfig};

use crate::config::MtpConfig;
use crate::genconfig::GenerationPlan;
use crate::npu_gen::TalkerTables;
use crate::prompt::{MtpHost, Prompt, TalkerHost, TtsSpecials};
use crate::sampling::{sample_cb0, sample_residual, SamplerCfg};
use crate::{GenOpts, TtsPaths};

/// Codebooks per codec frame.
const CODEBOOKS: usize = 16;
/// Positions per KV block of both decoders; the code predictor's sixteen positions are one block.
const BLOCK: u32 = 16;
/// The longest Talker context (prompt plus frames) a stream may hold, in positions.
pub const MAX_CONTEXT: u32 = 2048;
/// Positions one prefill chunk carries.
const PREFILL_CHUNK: u32 = 512;
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
    tables: TalkerTables,
    residual: ResidualTables,
    /// `small_to_mtp_projection` (`[d_mtp, e]`, bias `[d_mtp]`) when the code predictor is
    /// narrower than the Talker (the 1.7B family).
    projection: Option<(Vec<f32>, Vec<f32>)>,
    d_mtp: usize,
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

impl BatchedDecoder {
    /// Load both decoders on `gpu`'s device for up to `max_streams` streams at once.
    pub fn load_on(gpu: &Gpu, paths: &TtsPaths, max_streams: usize) -> Result<BatchedDecoder, String> {
        let max_streams = max_streams.max(1);
        // The Talker streams from its reader tensor by tensor; nothing is held as fp32 beside the upload.
        let talker_src = crate::import::open_talker(&paths.talker)?;
        let mut tcfg = QwenConfig::from_json_checked(&talker_src.config())?;
        tcfg.block_size = MAX_CONTEXT;
        let per_seq = MAX_CONTEXT / BLOCK;
        // The pool holds every stream at an average utterance, not each at the worst case: a
        // stream that would not fit waits for one to finish (`admit` checks the free blocks).
        let pool = (max_streams as u32 * per_seq / 2).max(per_seq);
        let talker = Engine::from_source_tier_on(gpu, tcfg, &talker_src, BLOCK, pool, max_streams as u32, per_seq, PREFILL_CHUNK, false, Dtype::F32);
        drop(talker_src);

        let mtp_ckpt = crate::import::load_mtp(&paths.mtp);
        let mc = MtpConfig::from_brain_json(&mtp_ckpt.header["config"]);
        let nres = mc.n_residual() as usize;
        let (d_mtp, e) = (mc.d_model as usize, mc.embedding_dim as usize);
        let mcfg = QwenConfig {
            vocab: mc.vocab,
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
        // The decoders read embedding rows, never token ids: the input table is only a placeholder.
        mtp_map.insert("tok.weight".into(), vec![0.0; mc.vocab as usize * d_mtp]);
        mtp_map.insert("lm_head.weight".into(), take(&mtp_ckpt, "lm_head.0.weight")?);
        let extra: Vec<String> = (1..nres).map(|k| format!("lm_head.{k}.weight")).collect();
        for name in &extra {
            mtp_map.insert(name.clone(), take(&mtp_ckpt, name)?);
        }
        let mut mtp = Engine::from_map_tier_on(gpu, mcfg, &mtp_map, BLOCK, max_streams as u32, max_streams as u32, 1, CODEBOOKS as u32, false, Dtype::F32);
        let names: Vec<&str> = extra.iter().map(String::as_str).collect();
        mtp.add_heads(&mtp_map, &names, mc.vocab as usize)?;
        drop(mtp_map);

        let tables = (0..nres).map(|i| take(&mtp_ckpt, &format!("codec_embedding.{i}.weight"))).collect::<Result<Vec<_>, _>>()?;
        let projection = match mtp_ckpt.find("small_to_mtp_projection.weight", "") {
            Some(w) => Some((w.clone(), take(&mtp_ckpt, "small_to_mtp_projection.bias")?)),
            None => None,
        };
        Ok(BatchedDecoder {
            talker,
            mtp,
            tables: TalkerTables::load(&paths.talker),
            residual: ResidualTables { tables, e },
            projection,
            d_mtp,
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
        match &self.projection {
            None => rows,
            Some((w, b)) => {
                let e = self.residual.e;
                let n = rows.len() / e;
                let mut out = model::hostmath::linear_rows(&rows, w, n, e, self.d_mtp);
                for row in out.chunks_mut(self.d_mtp) {
                    for (x, bias) in row.iter_mut().zip(b) {
                        *x += bias;
                    }
                }
                out
            }
        }
    }

    /// Start `job` as stream `index`: prefill its prompt and draw its first codebook-0 token.
    fn admit(&mut self, index: usize, job: Job, sp: &TtsSpecials) -> Result<Admission, String> {
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

    /// The residual draw of `cfg` for the row of `stream` at `frame`, codebook `k`.
    fn residual_draw(cfg: &SamplerCfg, seed: u64, frame: usize, k: usize) -> Option<RowDraw> {
        if !cfg.do_sample || cfg.temperature <= 0.0 {
            return None;
        }
        Some(RowDraw { temp: cfg.temperature, min_p: 0.0, top_p: cfg.top_p, top_k: cfg.top_k, key: draw_key(seed, frame as u64, k as u32) })
    }

    /// One frame of every stream in `live`: the code predictor's chain for all of them, the
    /// feedback rows, one Talker step and the next codebook-0 draw. Returns the streams that
    /// finished this frame.
    fn step(&mut self, live: &mut Vec<Stream>, sp: &TtsSpecials) -> Vec<(Stream, Result<(), String>)> {
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
        // embedding of codebook k-1, read by head k-1 to predict codebook k.
        let mut mtp_tables: Vec<BlockTable> = (0..b).map(|_| BlockTable::new()).collect();
        let hidden: Vec<f32> = live.iter().flat_map(|s| s.hidden.iter().copied()).collect();
        {
            let mut refs: Vec<&mut BlockTable> = mtp_tables.iter_mut().collect();
            self.mtp.forward_batched_embed(&mut refs, &self.to_mtp(hidden));
        }
        let mut input: Vec<f32> = live.iter().flat_map(|s| self.tables.codec_embed(s.cb0).iter().copied()).collect();
        let mut feed = input.clone();
        let mut codes = vec![vec![0u32; CODEBOOKS - 1]; b];
        for k in 1..CODEBOOKS {
            let draws: Vec<Option<RowDraw>> = live.iter().map(|s| Self::residual_draw(&s.plan.subtalker, s.seed, s.frames.len() / CODEBOOKS, k)).collect();
            let head = (k > 1).then(|| format!("lm_head.{}.weight", k - 1));
            let drawn = {
                let mut refs: Vec<&mut BlockTable> = mtp_tables.iter_mut().collect();
                self.mtp.forward_batched_embed_drawn(&mut refs, &self.to_mtp(input), head.as_deref(), &draws)
            };
            input = Vec::with_capacity(b * d);
            for (r, dr) in drawn.into_iter().enumerate() {
                // A draw the device kernels do not cover (a top-k inside the vocabulary) comes back
                // as the row, drawn here by the same rule the reference uses.
                let token = match dr {
                    Drawn::Token(t, _, _) => t,
                    Drawn::Row(row) => {
                        let s = &mut live[r];
                        sample_residual(&row, &s.plan.subtalker, &mut s.rng).token
                    }
                };
                codes[r][k - 1] = token;
                let row = self.residual.codec_embed(k - 1, token);
                input.extend_from_slice(row);
                for (f, x) in feed[r * d..(r + 1) * d].iter_mut().zip(row) {
                    *f += x;
                }
            }
        }
        for t in mtp_tables.iter_mut() {
            self.mtp.release_table(t);
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
        let (hidden, logits) = {
            let mut refs: Vec<&mut BlockTable> = live.iter_mut().map(|s| &mut s.table).collect();
            self.talker.forward_batched_embed_hidden_logits(&mut refs, &feed, None)
        };
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
