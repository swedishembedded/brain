// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements batched document-OCR serving for clients who
// convert large archives to text. If your team needs expertise in turning
// concurrent page requests into batched decoder rows then you can procure our
// services by sending an email to info@swedishembedded.com.

//! The scheduler-facing half of an OCR model: how a page request becomes the
//! prompt's token ids with the vision encoder's rows standing at the image
//! positions ([`model::serve::PromptEmbeds`]), and the decoder engine the
//! scheduler batches those requests onto.
//!
//! The encoder runs once per request; the decode then shares the decoder's
//! weight pass with every other request in flight ([`deepseek2::serve::Engine`]).
//! Admission, chunked prefill, batching, cancellation and streaming are
//! `model::serve` and `llmserve`, not this crate.
//!
//! The anti-repetition filter upstream recommends for long documents (vLLM's
//! no-repeat n-gram, 30 over a 90-token window, table-cell tags exempt) is the
//! scheduler's own [`model::serve::NoRepeatNgram`], set per request.

use std::sync::{mpsc, Arc};

use capability::{last_user_text, Invocation};
use data::qwen_chat::TemplateFlavor;
use data::qwen_tokenizer::QwenBpe;
use deepseek2::serve::{pipelines, Engine, EngineConfig, Precision};
use deepseek2::DeepseekV2Config;
use gpu_core::Gpu;
use llmserve::{Encoding, Frontend, ParsedRequest, ScheduledInstance};
use model::kv_tier::KvTier;
use model::serve::{NoRepeatNgram, PromptEmbeds, SampleParams, Scheduler};
use qwen3::chat::ToolChoice;

use crate::{DecoderDevice, DEFAULT_INSTRUCTION, DEFAULT_MAX_NEW, MAX_CONTEXT};

/// Pages decoded together: the batch the decoder's step buffers are sized for. A decode step reads the weights once for every row, so
/// the more rows share it the cheaper each token (4.8 ms a step at 32 rows, 6.9 ms at 64, `engine_bench`), and a recorded step is kept
/// per row count up to this many.
pub const MAX_BATCH: u32 = 64;
/// Tokens of K/V the pool holds across all pages in flight: 2048 per row at a full batch, in bf16 (the 14 GiB the decoder is budgeted).
pub const POOL_TOKENS: u32 = 128 * 1024;
/// Prefill rows per forward.
pub const PREFILL_CHUNK: u32 = 512;
/// vLLM's recipe for this model family: no 30-gram may repeat within a 90-token window.
pub const NGRAM_SIZE: u32 = 30;
pub const NGRAM_WINDOW: u32 = 90;

/// The decoder engine's shape at the served defaults.
pub fn engine_config() -> EngineConfig {
    EngineConfig { max_seq: MAX_CONTEXT, pool_tokens: POOL_TOKENS, max_batch: MAX_BATCH, chunk: PREFILL_CHUNK, block_size: 64, kv: KvTier::BF16, topk: 64, precision: Precision::Auto }
}

/// How a request asks its page to be read.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadOptions {
    /// The most local tiles the page is cut into: `Some(0)` reads it as one global view, `None` is the
    /// model's own default.
    pub max_tiles: Option<u32>,
}

/// One page to encode: the instruction, the decoded pixels (`h x w` RGB in `[0, 1]`) and how to read them.
pub struct PageRequest {
    pub instruction: String,
    pub hwc: Vec<f32>,
    pub w: u32,
    pub h: u32,
    pub read: ReadOptions,
}

/// A request's page before its pixels are decoded: the wire format is interleaved f32 (tens of megabytes for a
/// page), and unpacking it is work for the encoder's thread, not for the one that is decoding other requests.
pub struct PendingPage {
    instruction: String,
    image: Invocation,
    read: ReadOptions,
}

impl PendingPage {
    /// Unpack the pixels.
    fn decode(self) -> Result<PageRequest, String> {
        let (hwc, w, h) = capability::blob::decode_image(&self.image, "image")?;
        Ok(PageRequest { instruction: self.instruction, hwc, w, h, read: self.read })
    }
}

/// A page's prompt ids and the rows that stand at its image positions.
pub type Encoded = Result<(Vec<u32>, PromptEmbeds), String>;

/// The vision half of a request: a page's pixels to the prompt and the rows
/// that stand at its image positions.
pub trait PageEncoder {
    /// The tokenizer prompts are built with and generated ids decode through.
    fn tok(&self) -> &Arc<QwenBpe>;
    /// Preprocess the page, run the vision encoder and lay its rows out as the decoder's image
    /// block: the prompt's ids for the instruction and the rows that stand at its image positions.
    fn encode(&self, page: &PageRequest) -> Encoded;
    /// Run the encoder once over a small page and a large one, so the first real request does not pay for compiling its kernels.
    fn warm_up(&self) {
        for (w, h) in [(300u32, 200u32), (1300, 1000)] {
            let page = PageRequest { instruction: DEFAULT_INSTRUCTION.to_string(), hwc: vec![0.5; (w * h * 3) as usize], w, h, read: ReadOptions::default() };
            let _ = self.encode(&page);
        }
    }
    /// Encode every page waiting, reporting each through `done(index, result)` the moment it is ready, so a page's
    /// decode can start while the rest are still encoding. A model whose views batch through its encoder overrides
    /// this to pool them (and reports a pooled group together); the default encodes them one by one.
    fn encode_many(&self, pages: &[PageRequest], done: &mut dyn FnMut(usize, Encoded)) {
        for (i, page) in pages.iter().enumerate() {
            done(i, self.encode(page));
        }
    }
}

/// What a request carries to the encoder stage: the page, and how its decode is sampled.
struct EncodeJob {
    page: PendingPage,
    max_new: usize,
    reply: mpsc::Sender<Result<(ParsedRequest, Option<PromptEmbeds>), String>>,
}

/// The vision encoder on a thread of its own: pages queue to it and it encodes what waits ([`PageEncoder::encode_many`]) while the scheduler decodes other requests. A request's rows
/// arrive on a channel the scheduler polls, so an encode overlaps decode instead of stalling it.
struct EncodeStage {
    jobs: Option<mpsc::Sender<EncodeJob>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl EncodeStage {
    fn spawn<E: PageEncoder + Send + 'static>(enc: E, ngram: NoRepeatNgram) -> EncodeStage {
        let (jobs, queue) = mpsc::channel::<EncodeJob>();
        let worker = std::thread::Builder::new()
            .name("ocr-encode".into())
            .spawn(move || {
                // Everything that queued while the last batch ran is encoded together.
                while let Ok(first) = queue.recv() {
                    let mut batch = vec![first];
                    batch.extend(queue.try_iter());
                    let mut pages = Vec::with_capacity(batch.len());
                    let mut replies = Vec::with_capacity(batch.len());
                    for job in batch {
                        match job.page.decode() {
                            Ok(page) => {
                                pages.push(page);
                                replies.push((job.max_new, job.reply));
                            }
                            // A caller that gave up (cancelled, dropped its receiver) is not an error.
                            Err(e) => {
                                let _ = job.reply.send(Err(e));
                            }
                        }
                    }
                    enc.encode_many(&pages, &mut |i, result| {
                        let (max_new, reply) = &replies[i];
                        let _ = reply.send(result.map(|(ids, embeds)| (build_request(ids, *max_new, ngram), Some(embeds))));
                    });
                }
            })
            .expect("the encoder thread starts");
        EncodeStage { jobs: Some(jobs), worker: Some(worker) }
    }

    fn submit(&self, page: PendingPage, max_new: usize) -> Encoding {
        let (reply, rx) = mpsc::channel();
        let job = EncodeJob { page, max_new, reply };
        if let Some(jobs) = &self.jobs {
            // A stopped worker drops the reply sender, which the receiver reports as a stopped encoder.
            let _ = jobs.send(job);
        }
        rx
    }
}

impl Drop for EncodeStage {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// The scheduler request for an encoded page: greedy, raw (no chat template), the n-gram filter on.
fn build_request(ids: Vec<u32>, max_new: usize, ngram: NoRepeatNgram) -> ParsedRequest {
    ParsedRequest {
        ids,
        max_new,
        temp: 0.0,
        top_k: 0,
        top_p: 1.0,
        sampler: SampleParams { no_repeat: ngram, ..SampleParams::greedy() },
        seed: 0,
        stops: Vec::new(),
        tool_choice: ToolChoice::Auto,
        flavor: TemplateFlavor::default(),
        thinking_open: false,
        raw: true,
    }
}

/// An OCR request front end over a [`PageEncoder`] running on its own thread.
pub struct OcrFrontend {
    tok: Arc<QwenBpe>,
    label: &'static str,
    stop: Vec<u32>,
    stage: EncodeStage,
}

impl OcrFrontend {
    /// `eos` ends a generation; `allow` are the ids exempt from the n-gram filter (table-cell tags).
    ///
    /// Requests are built in the background, so the encoder runs while the decoder decodes - on another device, or on the same
    /// one: a tower's handle holds its freed blocks ([`serving_gpu`]), so it cannot stall the decoder's stream, and the two
    /// interleave their kernels (best of four alternating server runs on a shared card: 30 -> 36 pages/min median).
    pub fn new<E: PageEncoder + Send + 'static>(enc: E, label: &'static str, eos: u32, allow: &[u32]) -> OcrFrontend {
        let ngram = NoRepeatNgram::new(NGRAM_SIZE, NGRAM_WINDOW, allow);
        let t0 = std::time::Instant::now();
        enc.warm_up();
        crate::stage_time("load: encoder warm-up (kernels)", t0);
        OcrFrontend { tok: enc.tok().clone(), label, stop: vec![eos], stage: EncodeStage::spawn(enc, ngram) }
    }

    /// The page a request asks to read (pixels still packed), and its decode budget.
    fn page_of(&self, inv: &Invocation) -> Result<(PendingPage, usize), String> {
        let instruction = {
            let t = last_user_text(inv);
            if t.trim().is_empty() {
                DEFAULT_INSTRUCTION.to_string()
            } else {
                t
            }
        };
        let max_new = inv.get_i64("max_new").unwrap_or(DEFAULT_MAX_NEW).clamp(1, MAX_CONTEXT as i64) as usize;
        let blob = inv.get_blob("image").ok_or("missing required input 'image'")?.clone();
        let read = ReadOptions { max_tiles: inv.get_i64("max_tiles").map(|n| n.clamp(0, 64) as u32) };
        Ok((PendingPage { instruction, image: Invocation::new().blob("image", blob), read }, max_new))
    }
}

impl Frontend for OcrFrontend {
    fn tok(&self) -> &QwenBpe {
        &self.tok
    }
    fn stop(&self) -> &[u32] {
        &self.stop
    }
    fn label(&self) -> &str {
        self.label
    }
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String> {
        self.parse_embedded(inv).map(|(req, _)| req)
    }
    fn parse_embedded(&self, inv: &Invocation) -> Result<(ParsedRequest, Option<PromptEmbeds>), String> {
        let (page, max_new) = self.page_of(inv)?;
        self.stage.submit(page, max_new).recv().map_err(|_| format!("{}: the encoder stopped", self.label))?
    }
    /// The encoder runs on its own thread: the request is built there while the scheduler decodes.
    fn parse_in_background(&self, inv: &Invocation) -> Option<Encoding> {
        match self.page_of(inv) {
            Ok((page, max_new)) => Some(self.stage.submit(page, max_new)),
            // A bad request is answered at once by the inline path.
            Err(_) => None,
        }
    }
    /// Building a request runs the vision encoder: it is not run ahead of the request's slot.
    fn encodes_inputs(&self) -> bool {
        true
    }
}

/// A served OCR model: the front end and the scheduler that batches requests onto the decoder engine.
pub type OcrInstance = ScheduledInstance<Engine, OcrFrontend>;

/// A device handle for a serving tower on the ambient card. A step allocates activation temporaries and frees them again, and on CUDA a
/// driver free waits for the whole device - every other stream of the process included, the decoder's among them. Held freed blocks
/// make a free a bookkeeping entry, so a vision tower running beside the decoder cannot stall it.
pub fn serving_gpu(kernels: &[(&'static str, &'static str)]) -> Gpu {
    let gpu = Gpu::new(kernels);
    gpu.hold_freed_blocks(true);
    gpu
}

/// The decoder engine on `placement`'s device, from a decoder weight source under brain's names.
pub fn build_engine(cfg: DeepseekV2Config, decoder: &dyn checkpoint::TensorSource, placement: DecoderDevice) -> Result<Engine, String> {
    let kernels = pipelines();
    let gpu = match placement {
        DecoderDevice::Cpu => Gpu::new_cpu(&kernels),
        DecoderDevice::SameCard => Gpu::new(&kernels),
        DecoderDevice::Card(i) => gpu_core::devices::with_gpu(i, || Gpu::new(&kernels)).map_err(|e| format!("decoder on gpu{i}: {e}"))?,
    };
    let mut engine = Engine::new(gpu, cfg, decoder, engine_config())?;
    if !matches!(placement, DecoderDevice::Cpu) {
        // The first request would otherwise pay for kernel compilation and for the recording of every row count the batch passes through.
        let t0 = std::time::Instant::now();
        engine.warm_up();
        crate::stage_time("load: decoder warm-up (kernels, one decode step per row count)", t0);
    }
    Ok(engine)
}

/// Wrap `front` and `engine` in the continuously-batching instance.
pub fn instance(front: OcrFrontend, engine: Engine) -> OcrInstance {
    ScheduledInstance::new(front, Scheduler::new(engine, MAX_BATCH as usize), MAX_BATCH as usize)
}
