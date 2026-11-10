// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! [`MirageModel`]: the whole model with resident weights on a caller-chosen
//! device, answering batches of images.
//!
//! Per batch: letterbox and the letterboxed canvases go to the device in one
//! upload; the stem, body, top, dense head, memory construction, the batched
//! decoder loop and the reader run there. The host keeps what is index- or
//! text-shaped: letterbox resize, peak picking and ordering, box codes and
//! pooling ranges of the proposal slots, mask rendering, CTC collapse and JSON
//! assembly. Form extraction scores candidate branches with state cloning, so
//! it takes the device memories to the host decoder.
//!
//! Swedish Embedded AB implements solutions for serving perception models in
//! batches on shared accelerators for its clients. If your team needs
//! expertise in high-throughput vision inference then you can procure our
//! services by sending an email to info@swedishembedded.com.

use gpu_core::Gpu;
use serde_json::Value;

use super::decoder::DeviceDecoder;
use super::reader::{DeviceMemory, DeviceReader, SlotInputs, SlotItem};
use super::vision::{DeviceVision, Masks};
use super::{Device, WeightForm};
use crate::decoder::generate::{absent_references, GenerateParams};
use crate::describe::{assemble, Describe, Detection, Field, Inputs, Options, Section, DEFAULT_CONFIDENCE};
use crate::form::{self, Extract};
use crate::model::{Mirage, ModelError, DEFAULT_MAX_TOKENS};
use crate::vision::dense::{find_proposals, Kind, Proposal, MAX_OBJECTS, MAX_TEXT};
use crate::vision::masks::{coefficient_cell, object_field, round_box, Frame, COEF_SIDE};
use crate::vision::ops::valid_cells;
use crate::vision::preprocess::{canvas, letterbox, Resized};
use crate::vision::reader::{crop_rect, decode_windows, strip_from_crop, windows};

/// A crop rectangle `(x0, y0, x1, y1)` in source pixels.
type CropRect = (usize, usize, usize, usize);

/// An RGB input image.
#[derive(Debug, Clone, Copy)]
pub struct Image<'a> {
    /// `[height, width, 3]` bytes.
    pub rgb: &'a [u8],
    /// Height.
    pub height: usize,
    /// Width.
    pub width: usize,
}

/// What the decoder reads of one image: the live rows (cells and proposal slots) of its memory, `[rows, dim]`.
///
/// It depends on the pixels and on the proposal options (`conf`, `text_conf`, `max_objects`, `max_text_lines`) and on
/// nothing else, so a describe and an extract of the same image under the same proposal options share it.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderMemory {
    /// Row-major `[rows, dim]` values.
    pub rows: Vec<f32>,
    /// Width of a row.
    pub dim: usize,
}

impl DecoderMemory {
    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.rows.len() * std::mem::size_of::<f32>()
    }
}

/// One described image with its accounting.
#[derive(Debug, Clone, PartialEq)]
pub struct Described {
    /// The structured result.
    pub describe: Describe,
    /// Tokens the decoder generated for the image.
    pub decoder_tokens: usize,
    /// The decoder memory, when the batch was asked to keep it ([`BatchControl::keep_memory`]).
    pub memory: Option<std::sync::Arc<DecoderMemory>>,
}

/// One extracted form with its accounting.
#[derive(Debug, Clone, PartialEq)]
pub struct Extracted {
    /// The answers.
    pub extract: Extract,
    /// The decoder memory the answers were scored against (cached or computed).
    pub memory: std::sync::Arc<DecoderMemory>,
    /// Whether the memory came from the caller's cache instead of the vision half.
    pub reused: bool,
}

/// What a caller may ask of a batch beyond its images.
pub struct BatchControl<'a> {
    /// Return each image's decoder memory (one device read per batch).
    pub keep_memory: bool,
    /// Whether the request at this index of the batch has been cancelled. Asked before the batch starts and between
    /// decoder windows, so it must be cheap.
    pub cancelled: &'a (dyn Fn(usize) -> bool + Sync),
}

impl Default for BatchControl<'_> {
    fn default() -> Self {
        BatchControl { keep_memory: false, cancelled: &|_| false }
    }
}

/// Wall-clock split of the last `describe_batch`, in seconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    /// Letterbox, canvas build and upload, encoder, dense head, device synchronised.
    pub vision: f64,
    /// Proposal peaks, slot inputs, memory construction on the device.
    pub memory: f64,
    /// Batched decoder (prefill and steps) including the host loop guard.
    pub decode: f64,
    /// Mask maps, reader and JSON assembly.
    pub post: f64,
    /// Of `post`, the part spent on the device (mask maps and the batched reader).
    pub reader: f64,
}

/// The model with resident weights.
pub struct MirageModel {
    host: Mirage,
    dev: Device,
    vision: DeviceVision,
    reader: DeviceReader,
    memory: DeviceMemory,
    decoder: DeviceDecoder,
    timings: std::sync::Mutex<Timings>,
    resident: Option<u64>,
}

/// The vision half of a batch: proposals and the decoder memories.
struct Batch {
    images: Vec<Resized>,
    objects: Vec<Vec<Proposal>>,
    text: Vec<Vec<Proposal>>,
    memory: super::reader::MemoryBatch,
    masks: Masks,
    enc: super::vision::EncoderBatch,
    dense: super::vision::DenseBatch,
}

impl MirageModel {
    /// Put `host`'s weights on a device. `gpu` must have been created with
    /// [`Device::kernels`] (for example `Gpu::new(&Device::kernels())`).
    pub fn new(host: Mirage, gpu: Gpu) -> Result<MirageModel, ModelError> {
        MirageModel::new_with(host, gpu, WeightForm::default())
    }

    /// [`MirageModel::new`] with the residency form of the structured mixers chosen.
    pub fn new_with(host: Mirage, gpu: Gpu, form: WeightForm) -> Result<MirageModel, ModelError> {
        let dev = Device::new_with(gpu, form).map_err(ModelError::Request)?;
        let before = dev.gpu().memory_usage();
        let mut model = MirageModel {
            vision: DeviceVision::upload(&dev, &host.vision),
            reader: DeviceReader::upload(&dev, &host.vision.reader),
            memory: DeviceMemory::upload(&dev, &host.cfg, &host.vision.memory),
            decoder: DeviceDecoder::upload(&dev, &host.decoder),
            dev,
            host,
            timings: Default::default(),
            resident: None,
        };
        model.resident = before.zip(model.dev.gpu().memory_usage()).map(|(b, a)| a.live.saturating_sub(b.live));
        Ok(model)
    }

    /// Device bytes the model's weights and constants hold, by the backend's own allocation accounting (`None` on a
    /// backend that does not account). Measured around the upload, so it includes every table and kernel tensor.
    pub fn resident_bytes(&self) -> Option<u64> {
        self.resident
    }

    /// Where the last `describe_batch` spent its time.
    pub fn last_timings(&self) -> Timings {
        self.read_timings()
    }

    fn read_timings(&self) -> Timings {
        *self.timings.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn set_timings(&self, t: Timings) {
        *self.timings.lock().unwrap_or_else(|e| e.into_inner()) = t;
    }

    /// The device operators run on.
    pub fn device(&self) -> &Device {
        &self.dev
    }

    /// The host model (tokenizer, oracle operators).
    pub fn host(&self) -> &Mirage {
        &self.host
    }

    fn analyze(&self, images: &[Resized], opts: &Options) -> Batch {
        let cfg = &self.host.cfg;
        let dev = &self.dev;
        let (b, side) = (images.len(), cfg.canvas / cfg.stride);
        let sizes: Vec<(usize, usize)> = images.iter().map(|i| (i.h, i.w)).collect();
        let t0 = std::time::Instant::now();
        let flat: Vec<f32> = images.iter().flat_map(canvas).collect();
        let masks = Masks::new(dev, cfg, &sizes);
        let enc = self.vision.encode(dev, &dev.buf(&flat), b, &masks);
        let dense = self.vision.dense(dev, &enc, &masks);
        dev.gpu().poll_wait();
        let t1 = std::time::Instant::now();

        let head = dev.read_at(&dense.head, 0, b * side * side * 10);
        let head_in = dev.read_at(&dense.head_in, 0, b * side * side * cfg.dense_dim);
        let encoder = dev.read_at(&enc.encoder, 0, b * cfg.grid() * cfg.grid() * cfg.dim);
        let (hl, el, cl) = (side * side * 10, cfg.grid() * cfg.grid() * cfg.dim, side * side * cfg.dense_dim);
        let max_objects = opts.max_objects.unwrap_or(cfg.detections).min(MAX_OBJECTS);
        let max_text = opts.max_text_lines.unwrap_or(cfg.regions).min(MAX_TEXT);
        let (mut objects, mut text) = (Vec::new(), Vec::new());
        let valids: Vec<Vec<bool>> = sizes.iter().map(|&(h, w)| valid_cells(h, w, cfg.grid())).collect();
        for (i, &(h, w)) in sizes.iter().enumerate() {
            let valid = valid_cells(h, w, side);
            let head = &head[i * hl..(i + 1) * hl];
            objects.push(find_proposals(head, side, Kind::Object, opts.conf.unwrap_or(DEFAULT_CONFIDENCE), max_objects, &valid));
            text.push(find_proposals(head, side, Kind::Text, opts.text_conf.unwrap_or(DEFAULT_CONFIDENCE), max_text, &valid));
        }
        let items: Vec<SlotItem> = (0..b)
            .map(|i| SlotItem { encoder: &encoder[i * el..(i + 1) * el], head_in: &head_in[i * cl..(i + 1) * cl], valid: &valids[i], slots: self.host.vision.slots(&objects[i], &text[i]) })
            .collect();
        let slots = SlotInputs::collect(&self.host.vision.memory, &items, cfg.detections + cfg.regions);
        let memory = self.memory.build(dev, &enc.encoder, &valids, &slots);
        let mut t = self.read_timings();
        t.vision = (t1 - t0).as_secs_f64();
        t.memory = t1.elapsed().as_secs_f64();
        self.set_timings(t);
        Batch { images: images.to_vec(), objects, text, memory, masks, enc, dense }
    }

    fn letterboxed(images: &[Image]) -> Vec<Result<Resized, ModelError>> {
        images.iter().map(|i| letterbox(i.rgb, i.height, i.width).map_err(ModelError::from)).collect()
    }

    /// Analyse a batch of images and decode a description of each.
    pub fn describe_batch(&self, images: &[Image], opts: &Options) -> Vec<Result<Describe, ModelError>> {
        self.describe_batch_with(images, opts, &BatchControl::default()).into_iter().map(|r| r.map(|d| d.describe)).collect()
    }

    /// [`Self::describe_batch`] with accounting and control: the decoder tokens of each image, optionally the
    /// decoder memory for a later [`Self::extract_batch_with`], and cancellation. An image whose `cancelled` answers
    /// `true` before the batch starts takes no part in it; one cancelled while the batch decodes stops costing
    /// decoder steps once every live image has stopped, and both come back as [`ModelError::Cancelled`].
    pub fn describe_batch_with(&self, images: &[Image], opts: &Options, control: &BatchControl) -> Vec<Result<Described, ModelError>> {
        let refuse = |e: ModelError| -> Vec<Result<Described, ModelError>> { images.iter().map(|_| Err(ModelError::Request(e.to_string()))).collect() };
        if opts.form.is_some() {
            return refuse(ModelError::Request("`form` requests extraction, use `extract_batch`".to_string()));
        }
        let prompt = match self.host.prompt(opts.language.as_deref()) {
            Ok(p) => p,
            Err(e) => return refuse(e),
        };
        let boxed = Self::letterboxed(images);
        let mut results: Vec<Result<Described, ModelError>> = Vec::with_capacity(images.len());
        let (mut ok, mut resized): (Vec<usize>, Vec<Resized>) = (Vec::new(), Vec::new());
        for (i, r) in boxed.into_iter().enumerate() {
            results.push(match r {
                Err(e) => Err(e),
                Ok(_) if (control.cancelled)(i) => Err(ModelError::Cancelled),
                Ok(r) => {
                    ok.push(i);
                    resized.push(r);
                    Err(ModelError::Cancelled)
                }
            });
        }
        if resized.is_empty() {
            return results;
        }
        let cfg = &self.host.cfg;
        let batch = self.analyze(&resized, opts);
        let banned: Vec<Vec<u32>> = (0..resized.len()).map(|i| absent_references(cfg, batch.objects[i].len(), batch.text[i].len())).collect();
        let params: Vec<GenerateParams> = banned
            .iter()
            .map(|banned| GenerateParams { prompt: &prompt, banned, max_tokens: opts.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS), max_rank: opts.max_rank(), object_slots: cfg.detections, guard: true })
            .collect();
        let t2 = std::time::Instant::now();
        let generations = self.decoder.generate_batch_until(&self.dev, &batch.memory, &self.host.tokenizer, &params, &|k| (control.cancelled)(ok[k]));
        let live: Vec<bool> = ok.iter().map(|&i| !(control.cancelled)(i)).collect();

        let t3 = std::time::Instant::now();
        let side = cfg.canvas / cfg.stride;
        // Device work first, for the whole batch: the mask maps (when masks are wanted) and every text proposal
        // through the reader in as few batches as possible. What remains is host assembly, image by image.
        let n = resized.len();
        let maps: Option<(Vec<f32>, Vec<f32>)> = (opts.wants(Field::Mask) && batch.objects.iter().any(|o| !o.is_empty())).then(|| {
            let (p, c) = self.vision.mask_maps(&self.dev, &batch.enc, &batch.dense, &batch.masks);
            (self.dev.read_at(&p, 0, n * 4 * side * side * 32), self.dev.read_at(&c, 0, n * side * side * 32))
        });
        let frames: Vec<Frame> = ok.iter().enumerate().map(|(k, &i)| Frame { resized_w: batch.images[k].w, resized_h: batch.images[k].h, width: images[i].width, height: images[i].height }).collect();
        let reads: Vec<Vec<Option<String>>> = if opts.section_set().contains(&Section::Text) {
            let jobs: Vec<(usize, usize)> = (0..n).filter(|&k| live[k]).flat_map(|k| (0..batch.text[k].len()).map(move |j| (k, j))).collect();
            let strings = self.read_lines(&jobs.iter().map(|&(k, j)| (images[ok[k]], crop_rect(crate::describe::public_box(batch.text[k][j].bbox, frames[k]).map(|v| v as f32), images[ok[k]].width, images[ok[k]].height))).collect::<Vec<_>>());
            let mut per: Vec<Vec<Option<String>>> = (0..n).map(|k| vec![None; batch.text[k].len()]).collect();
            for (&(k, j), text) in jobs.iter().zip(strings) {
                per[k][j] = text;
            }
            per
        } else {
            (0..n).map(|_| Vec::new()).collect()
        };
        let t_dev = t3.elapsed().as_secs_f64();

        let assemble_one = |k: usize| -> Option<Described> {
            if !live[k] {
                return None;
            }
            let frame = frames[k];
            let objects: Vec<Detection> = batch.objects[k].iter().map(Detection::from).collect();
            let text: Vec<Detection> = batch.text[k].iter().map(Detection::from).collect();
            let field = |b: [f32; 4]| {
                let (protos, coefs) = maps.as_ref().expect("masks were requested");
                let (pl, cl) = (4 * side * side * 32, side * side * 32);
                let (cy, cx) = coefficient_cell(round_box(b));
                object_field(&protos[k * pl..(k + 1) * pl], &coefs[k * cl + (cy * COEF_SIDE + cx) * 32..][..32], b)
            };
            let read = |j: usize| -> Option<String> { reads[k].get(j).cloned().flatten() };
            let describe = assemble(&generations[k].text, &Inputs { objects: &objects, text: &text, frame, read: &read, field: &field }, opts);
            Some(Described { describe, decoder_tokens: generations[k].tokens.len(), memory: None })
        };
        // Assembly is pure host work per image (mask fields, rendering, JSON): fan it out.
        let workers = std::thread::available_parallelism().map_or(1, |p| p.get()).min(8).min(n);
        let described: Vec<Option<Described>> = std::thread::scope(|sc| {
            let handles: Vec<_> = (0..workers).map(|w| sc.spawn({ let assemble_one = &assemble_one; move || (w..n).step_by(workers).map(|k| (k, assemble_one(k))).collect::<Vec<_>>() })).collect();
            let mut all: Vec<(usize, Option<Described>)> = handles.into_iter().flat_map(|h| h.join().expect("assembly does not panic")).collect();
            all.sort_by_key(|(k, _)| *k);
            all.into_iter().map(|(_, d)| d).collect()
        });
        let memories = control.keep_memory.then(|| self.decoder_memories(&batch.memory, n));
        for (k, d) in described.into_iter().enumerate() {
            results[ok[k]] = match d {
                Some(mut d) => {
                    d.memory = memories.as_ref().map(|m| m[k].clone());
                    Ok(d)
                }
                None => Err(ModelError::Cancelled),
            };
        }
        let mut t = self.read_timings();
        t.decode = (t3 - t2).as_secs_f64();
        t.post = t3.elapsed().as_secs_f64();
        t.reader = t_dev;
        self.set_timings(t);
        results
    }

    /// Read many text lines at once: crops become strips, all windows of all strips run through the reader in
    /// batches, and the CTC path of each strip is collapsed on the host. `None` where a crop is unusable.
    fn read_lines(&self, jobs: &[(Image, CropRect)]) -> Vec<Option<String>> {
        const WINDOWS_PER_CALL: usize = 256;
        let strips: Vec<Option<crate::vision::preprocess::Resized>> = jobs
            .iter()
            .map(|&(img, (x0, y0, x1, y1))| {
                if x1 <= x0 || y1 <= y0 || x1 > img.width || y1 > img.height {
                    return None;
                }
                let mut crop = Vec::with_capacity((x1 - x0) * (y1 - y0) * 3);
                for y in y0..y1 {
                    crop.extend_from_slice(&img.rgb[(y * img.width + x0) * 3..(y * img.width + x1) * 3]);
                }
                strip_from_crop(&crop, y1 - y0, x1 - x0).ok()
            })
            .collect();
        let wins: Vec<Vec<Vec<u8>>> = strips.iter().map(|s| s.as_ref().map(windows).unwrap_or_default()).collect();
        let flat: Vec<&Vec<u8>> = wins.iter().flatten().collect();
        let per = crate::vision::reader::FRAMES * crate::config::READER_CLASSES;
        let mut logits: Vec<Vec<f32>> = Vec::with_capacity(flat.len());
        for chunk in flat.chunks(WINDOWS_PER_CALL) {
            let bytes: Vec<u8> = chunk.iter().flat_map(|w| w.iter().copied()).collect();
            let out = self.reader.forward(&self.dev, &bytes, chunk.len());
            logits.extend(out.chunks_exact(per).map(<[f32]>::to_vec));
        }
        let mut next = 0;
        strips
            .iter()
            .zip(&wins)
            .map(|(strip, w)| {
                let lg = &logits[next..next + w.len()];
                next += w.len();
                strip.as_ref().map(|s| decode_windows(lg, s.w, &self.host.charset))
            })
            .collect()
    }

    /// Analyse a batch of images and extract the fields of the matching form in `forms`.
    pub fn extract_batch(&self, images: &[Image], forms: &[Value], opts: &Options) -> Vec<Result<Extract, ModelError>> {
        let cached = vec![None; images.len()];
        self.extract_batch_with(images, forms, opts, &cached, &BatchControl::default()).into_iter().map(|r| r.map(|e| e.extract)).collect()
    }

    /// [`Self::extract_batch`] where `cached[i]` is the decoder memory of image `i` from an earlier call under the
    /// same proposal options: those images skip the vision half, the others are analysed in one batch. A request
    /// cancelled before its form is scored comes back as [`ModelError::Cancelled`].
    pub fn extract_batch_with(&self, images: &[Image], forms: &[Value], opts: &Options, cached: &[Option<std::sync::Arc<DecoderMemory>>], control: &BatchControl) -> Vec<Result<Extracted, ModelError>> {
        assert_eq!(images.len(), forms.len(), "one form per image");
        assert_eq!(images.len(), cached.len(), "one cache slot per image");
        let cfg = &self.host.cfg;
        let prompt = self.host.prompt(opts.language.as_deref());
        // The images that need the vision half: not cached, not cancelled, a usable picture.
        let boxed: Vec<Option<Result<Resized, ModelError>>> =
            (0..images.len()).map(|i| (cached[i].is_none() && !(control.cancelled)(i)).then(|| letterbox(images[i].rgb, images[i].height, images[i].width).map_err(ModelError::from))).collect();
        let analysed: Vec<usize> = (0..images.len()).filter(|&i| matches!(boxed[i], Some(Ok(_)))).collect();
        let resized: Vec<Resized> = analysed.iter().filter_map(|&i| boxed[i].as_ref().and_then(|r| r.as_ref().ok()).cloned()).collect();
        let fresh = (!resized.is_empty()).then(|| {
            let batch = self.analyze(&resized, opts);
            self.decoder_memories(&batch.memory, resized.len())
        });
        (0..images.len())
            .map(|i| -> Result<Extracted, ModelError> {
                if (control.cancelled)(i) {
                    return Err(ModelError::Cancelled);
                }
                let (memory, reused) = match (&cached[i], &boxed[i]) {
                    (Some(m), _) => (m.clone(), true),
                    (None, Some(Err(e))) => return Err(ModelError::Request(e.to_string())),
                    (None, _) => {
                        let k = analysed.iter().position(|&v| v == i).expect("an analysed image");
                        (fresh.as_ref().expect("a batch was analysed")[k].clone(), false)
                    }
                };
                let fields = form::parse_form(&forms[i])?;
                let cross = self.host.decoder.cross_memory(&memory.rows, memory.rows.len() / cfg.dim);
                let prompt = prompt.as_ref().map_err(|e| ModelError::Request(e.to_string()))?;
                let extract = form::extract(&self.host.decoder, &cross, &self.host.tokenizer, prompt, &fields, opts.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS))?;
                Ok(Extracted { extract, memory, reused })
            })
            .collect()
    }

    /// The live rows of each of the first `n` decoder memories of a batch, read back in one go.
    fn decoder_memories(&self, mem: &super::reader::MemoryBatch, n: usize) -> Vec<std::sync::Arc<DecoderMemory>> {
        let (d, stride) = (self.host.cfg.dim, mem.stride);
        let rows = self.dev.read_at(&mem.rows, 0, n * stride * d);
        let mask = self.dev.read_at(&mem.mask, 0, n * stride);
        (0..n)
            .map(|k| {
                let live: Vec<f32> = rows[k * stride * d..(k + 1) * stride * d].chunks_exact(d).zip(&mask[k * stride..(k + 1) * stride]).filter(|(_, m)| **m > 0.0).flat_map(|(r, _)| r.iter().copied()).collect();
                std::sync::Arc::new(DecoderMemory { rows: live, dim: d })
            })
            .collect()
    }
}
