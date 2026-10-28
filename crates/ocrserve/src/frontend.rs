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
use deepseek2::serve::{pipelines, Engine, EngineConfig};
use deepseek2::DeepseekV2Config;
use gpu_core::Gpu;
use llmserve::{Encoding, Frontend, ParsedRequest, ScheduledInstance};
use model::serve::{NoRepeatNgram, PromptEmbeds, SampleParams, Scheduler};
use qwen3::chat::ToolChoice;

use crate::{DecoderDevice, DEFAULT_INSTRUCTION, DEFAULT_MAX_NEW, MAX_CONTEXT};

/// Pages decoded together: the batch the decoder's step buffers are sized for. A decode step reads the weights once for every row, so
/// the more rows share it the cheaper each token (4.8 ms a step at 32 rows, 6.9 ms at 64, `engine_bench`), and a recorded step is kept
/// per row count up to this many.
pub const MAX_BATCH: u32 = 64;
/// Tokens of K/V the pool holds across all pages in flight: 2048 per row at a full batch, in bf16 (the 14 GiB the decoder is budgeted).
pub const POOL_TOKENS: u32 = 128 * 1024;
/// Pages the vision encoders take at once: each is a full set of towers on the card (~4.5 GiB for DeepSeek-OCR).
pub const ENCODE_WORKERS: usize = 2;
/// Prefill rows per forward.
pub const PREFILL_CHUNK: u32 = 512;
/// Prompt rows prefilled between two decode steps: one chunk. The scheduler's default of two chunks makes every step that admits a
/// page ~80 ms longer for every page already decoding; with one, 8 full pages and 8 clients of small crops (best of two alternating
/// runs per setting, a neighbour on the card): the crops' median latency fell 3.5 -> 1.9 s (1.1 alone) and the pages' throughput
/// rose 36 -> 51 per minute. 256 and 128 rows were no better than 512.
pub const PREFILL_BUDGET: u32 = 512;
/// Prompt rows prefilled between two decode steps: one chunk. The scheduler's default of two chunks makes every step that admits a
/// page ~80 ms longer for every page already decoding; with one, 8 full pages and 8 clients of small crops (best of two alternating
/// runs per setting, a neighbour on the card): the crops' median latency fell 3.5 -> 1.9 s (1.1 alone) and the pages' throughput
/// rose 36 -> 51 per minute. 256 and 128 rows were no better than 512.
/// vLLM's recipe for this model family: no 30-gram may repeat within a 90-token window.
pub const NGRAM_SIZE: u32 = 30;
pub const NGRAM_WINDOW: u32 = 90;

/// The decoder engine's shape under the serving configuration ([`crate::policy`]).
pub fn engine_config() -> EngineConfig {
    crate::policy::config().engine_config()
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
    cancel: capability::CancelToken,
}

impl PendingPage {
    /// Pixels in the page, read from its metadata without unpacking it: what its encode costs, to the first order.
    fn pixels(&self) -> u64 {
        let dim = |k: &str| self.image.get_blob("image").and_then(|b| b.meta.get(k)).and_then(|v| v.as_u64()).unwrap_or(0);
        dim("w") * dim("h")
    }

    /// Whether its request was cancelled while it waited.
    fn cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

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

/// The pages waiting for an encoder, and whether the stage is closing.
#[derive(Default)]
struct EncodeQueue {
    waiting: Vec<(std::time::Instant, EncodeJob)>,
    closed: bool,
}

/// The vision encoders, one thread each: pages queue to them and each takes the next ([`next_job`]) and encodes it
/// ([`PageEncoder::encode_many`]) while the scheduler decodes other requests. A request's rows arrive on a channel the scheduler
/// polls, so an encode overlaps decode instead of stalling it. Each encoder owns its towers (their scratch is not shared), so two
/// pages encode at the same time on the same card from two streams: four tile encodes a second on one stream, six on two, seven
/// on three (`encode_concurrency`, best of three runs, a quiet card).
struct EncodeStage {
    queue: std::sync::Arc<(std::sync::Mutex<EncodeQueue>, std::sync::Condvar)>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

/// Which queued page the encoder takes next: the cheapest, discounted by how long it has waited, so a small crop is not held behind a
/// full page's encode (a crop is a tenth of the work) and a full page cannot be passed over forever.
fn next_job(waiting: &[(std::time::Instant, EncodeJob)]) -> usize {
    let cost = |(since, job): &(std::time::Instant, EncodeJob)| job.page.pixels() as f64 / (1.0 + since.elapsed().as_secs_f64());
    (0..waiting.len()).min_by(|&a, &b| cost(&waiting[a]).total_cmp(&cost(&waiting[b]))).expect("a non-empty queue")
}

impl EncodeStage {
    fn spawn<E: PageEncoder + Send + 'static>(encoders: Vec<E>, ngram: NoRepeatNgram) -> EncodeStage {
        let queue = std::sync::Arc::new((std::sync::Mutex::new(EncodeQueue::default()), std::sync::Condvar::new()));
        let workers = encoders
            .into_iter()
            .enumerate()
            .map(|(i, enc)| {
                let queue = queue.clone();
                std::thread::Builder::new()
                    .name(format!("ocr-encode-{i}"))
                    .spawn(move || {
                        let (lock, ready) = &*queue;
                        loop {
                            let job = {
                                let mut q = lock.lock().unwrap_or_else(|e| e.into_inner());
                                loop {
                                    if !q.waiting.is_empty() {
                                        let at = next_job(&q.waiting);
                                        break q.waiting.swap_remove(at).1;
                                    }
                                    if q.closed {
                                        return;
                                    }
                                    q = ready.wait(q).unwrap_or_else(|e| e.into_inner());
                                }
                            };
                            // A request cancelled while it waited costs nothing.
                            if job.page.cancelled() {
                                let _ = job.reply.send(Err("cancelled".to_string()));
                                continue;
                            }
                            let page = match job.page.decode() {
                                Ok(page) => page,
                                Err(e) => {
                                    let _ = job.reply.send(Err(e));
                                    continue;
                                }
                            };
                            // A page is answered the moment it is encoded; a caller that gave up is not an error.
                            enc.encode_many(std::slice::from_ref(&page), &mut |_, result| {
                                let _ = job.reply.send(result.map(|(ids, embeds)| (build_request(ids, job.max_new, ngram), Some(embeds))));
                            });
                        }
                    })
                    .expect("an encoder thread starts")
            })
            .collect();
        EncodeStage { queue, workers }
    }

    fn submit(&self, page: PendingPage, max_new: usize) -> Encoding {
        let (reply, rx) = mpsc::channel();
        let (lock, ready) = &*self.queue;
        lock.lock().unwrap_or_else(|e| e.into_inner()).waiting.push((std::time::Instant::now(), EncodeJob { page, max_new, reply }));
        ready.notify_one();
        rx
    }
}

impl Drop for EncodeStage {
    fn drop(&mut self) {
        let (lock, ready) = &*self.queue;
        lock.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
        ready.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// The scheduler request for an encoded page: greedy, raw (no chat template), the n-gram filter on.
fn build_request(ids: Vec<u32>, max_new: usize, ngram: NoRepeatNgram) -> ParsedRequest {
    ParsedRequest {
        ids,
        max_new: Some(max_new),
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
    pub fn new<E: PageEncoder + Send + 'static>(encoders: Vec<E>, label: &'static str, eos: u32, allow: &[u32]) -> OcrFrontend {
        let ngram = NoRepeatNgram::new(NGRAM_SIZE, NGRAM_WINDOW, allow);
        let t0 = std::time::Instant::now();
        for enc in &encoders {
            enc.warm_up();
        }
        crate::stage_time("load: encoder warm-up (kernels)", t0);
        OcrFrontend { tok: encoders.first().expect("an OCR front end needs an encoder").tok().clone(), label, stop: vec![eos], stage: EncodeStage::spawn(encoders, ngram) }
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
        Ok((PendingPage { instruction, image: Invocation::new().blob("image", blob), read, cancel: inv.cancel.clone() }, max_new))
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
    let rows = crate::policy::config().batch() as usize;
    let mut scheduler = Scheduler::new(engine, rows);
    scheduler.set_prefill_budget(PREFILL_BUDGET);
    ScheduledInstance::new(front, scheduler, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use capability::{Blob, Media};
    use std::time::{Duration, Instant};

    fn job(w: u32, h: u32, waited: Duration) -> (Instant, EncodeJob) {
        let blob = Blob::new(Media::Image, Vec::new()).with_meta(serde_json::json!({"w": w, "h": h, "c": 3}));
        let page = PendingPage { instruction: String::new(), image: Invocation::new().blob("image", blob), read: ReadOptions::default(), cancel: capability::CancelToken::default() };
        (Instant::now().checked_sub(waited).expect("a clock that runs long enough"), EncodeJob { page, max_new: 1, reply: mpsc::channel().0 })
    }

    /// The encoder serves the cheap page first - a crop is a tenth of a page's work - but a page that has waited long enough goes
    /// ahead of fresh crops, so a stream of crops cannot starve it.
    #[test]
    fn the_encoder_takes_the_cheapest_page_and_does_not_starve_a_large_one() {
        let fresh = Duration::ZERO;
        let queue = vec![job(2000, 3000, fresh), job(600, 400, fresh), job(1200, 1700, fresh)];
        assert_eq!(next_job(&queue), 1, "the crop first");
        let queue = vec![job(2000, 3000, Duration::from_secs(120)), job(600, 400, fresh)];
        assert_eq!(next_job(&queue), 0, "a page that waited two minutes goes before a fresh crop");
    }

    struct Sleepy {
        tok: Arc<QwenBpe>,
        busy: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl PageEncoder for Sleepy {
        fn tok(&self) -> &Arc<QwenBpe> {
            &self.tok
        }
        fn encode(&self, _page: &PageRequest) -> Encoded {
            use std::sync::atomic::Ordering::SeqCst;
            let now = self.busy.fetch_add(1, SeqCst) + 1;
            self.peak.fetch_max(now, SeqCst);
            std::thread::sleep(Duration::from_millis(120));
            self.busy.fetch_sub(1, SeqCst);
            Ok((vec![1], PromptEmbeds::run(4, 0, vec![0.0; 4]).expect("rows")))
        }
    }

    /// Two encoders take two pages at once, and every page is answered.
    #[test]
    fn pages_encode_on_as_many_threads_as_there_are_encoders() {
        use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
        let gt = checkpoint::gguf::GgufTokenizer { model: "gpt2".into(), pre: Some("deepseek-v3".into()), tokens: vec!["a".into()], merges: Vec::new(), token_types: vec![1], bos: Some(0), eos: Some(0), unk: None, pad: None, ..Default::default() };
        let tok = Arc::new(QwenBpe::from_gguf(&gt).expect("tokenizer"));
        let (busy, peak) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
        let encoders: Vec<Sleepy> = (0..2).map(|_| Sleepy { tok: tok.clone(), busy: busy.clone(), peak: peak.clone() }).collect();
        let stage = EncodeStage::spawn(encoders, NoRepeatNgram::new(NGRAM_SIZE, NGRAM_WINDOW, &[]));
        let page = || {
            let blob = capability::blob::image_blob(&[0.5; 12], 2, 2, 3);
            PendingPage { instruction: String::new(), image: Invocation::new().blob("image", blob), read: ReadOptions::default(), cancel: capability::CancelToken::default() }
        };
        let t0 = Instant::now();
        let replies: Vec<Encoding> = (0..4).map(|_| stage.submit(page(), 1)).collect();
        for r in replies {
            r.recv_timeout(Duration::from_secs(5)).expect("a reply").expect("an encoded page");
        }
        assert_eq!(peak.load(SeqCst), 2, "two pages were in the encoders at once");
        assert!(t0.elapsed() < Duration::from_millis(400), "four pages of 120 ms on two encoders take two rounds, not four: {:?}", t0.elapsed());
    }
}
