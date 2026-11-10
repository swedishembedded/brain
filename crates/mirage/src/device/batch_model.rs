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

use std::cell::OnceCell;

use gpu_core::Gpu;
use serde_json::Value;

use super::decoder::DeviceDecoder;
use super::reader::{DeviceMemory, DeviceReader, SlotInputs, SlotItem};
use super::vision::{DeviceVision, Masks};
use super::Device;
use crate::decoder::generate::{absent_references, GenerateParams};
use crate::describe::{assemble, Describe, Detection, Inputs, Options, DEFAULT_CONFIDENCE};
use crate::form::{self, Extract};
use crate::model::{Mirage, ModelError, DEFAULT_MAX_TOKENS};
use crate::vision::dense::{find_proposals, Kind, Proposal, MAX_OBJECTS, MAX_TEXT};
use crate::vision::masks::{coefficient_cell, object_field, round_box, Frame, COEF_SIDE};
use crate::vision::ops::valid_cells;
use crate::vision::preprocess::{canvas, letterbox, Resized};
use crate::vision::reader::{crop_rect, decode_windows, strip_from_crop, windows};

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

/// Wall-clock split of the last `describe_batch`, in seconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timings {
    /// Letterbox, canvas build and upload, encoder, dense head, device synchronised.
    pub vision: f64,
    /// Proposal peaks, slot inputs, memory construction on the device.
    pub memory: f64,
    /// Batched decoder (prefill and steps) including the host loop guard.
    pub decode: f64,
    /// Mask fields, reader and JSON assembly.
    pub post: f64,
}

/// The model with resident weights.
pub struct MirageModel {
    host: Mirage,
    dev: Device,
    vision: DeviceVision,
    reader: DeviceReader,
    memory: DeviceMemory,
    decoder: DeviceDecoder,
    timings: std::cell::Cell<Timings>,
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
        let dev = Device::new(gpu).map_err(ModelError::Request)?;
        Ok(MirageModel {
            vision: DeviceVision::upload(&dev, &host.vision),
            reader: DeviceReader::upload(&dev, &host.vision.reader),
            memory: DeviceMemory::upload(&dev, &host.cfg, &host.vision.memory),
            decoder: DeviceDecoder::upload(&dev, &host.decoder),
            dev,
            host,
            timings: Default::default(),
        })
    }

    /// Where the last `describe_batch` spent its time.
    pub fn last_timings(&self) -> Timings {
        self.timings.get()
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
        let mut t = self.timings.get();
        t.vision = (t1 - t0).as_secs_f64();
        t.memory = t1.elapsed().as_secs_f64();
        self.timings.set(t);
        Batch { images: images.to_vec(), objects, text, memory, masks, enc, dense }
    }

    fn letterboxed(images: &[Image]) -> Vec<Result<Resized, ModelError>> {
        images.iter().map(|i| letterbox(i.rgb, i.height, i.width).map_err(ModelError::from)).collect()
    }

    /// Analyse a batch of images and decode a description of each.
    pub fn describe_batch(&self, images: &[Image], opts: &Options) -> Vec<Result<Describe, ModelError>> {
        if opts.form.is_some() {
            return images.iter().map(|_| Err(ModelError::Request("`form` requests extraction, use `extract_batch`".to_string()))).collect();
        }
        let prompt = match self.host.prompt(opts.language.as_deref()) {
            Ok(p) => p,
            Err(e) => return images.iter().map(|_| Err(ModelError::Request(e.to_string()))).collect(),
        };
        let (ok, resized): (Vec<usize>, Vec<Resized>) = Self::letterboxed(images).into_iter().enumerate().filter_map(|(i, r)| r.ok().map(|r| (i, r))).unzip();
        let mut results: Vec<Result<Describe, ModelError>> = Self::letterboxed(images).into_iter().map(|r| r.map(|_| Describe::default())).collect();
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
        let generations = self.decoder.generate_batch(&self.dev, &batch.memory, &self.host.tokenizer, &params);

        let t3 = std::time::Instant::now();
        let maps: OnceCell<(Vec<f32>, Vec<f32>)> = OnceCell::new();
        let side = cfg.canvas / cfg.stride;
        for (k, &i) in ok.iter().enumerate() {
            let img = images[i];
            let frame = Frame { resized_w: batch.images[k].w, resized_h: batch.images[k].h, width: img.width, height: img.height };
            let objects: Vec<Detection> = batch.objects[k].iter().map(Detection::from).collect();
            let text: Vec<Detection> = batch.text[k].iter().map(Detection::from).collect();
            let field = |b: [f32; 4]| {
                let (protos, coefs) = maps.get_or_init(|| {
                    let (p, c) = self.vision.mask_maps(&self.dev, &batch.enc, &batch.dense, &batch.masks);
                    let n = resized.len();
                    (self.dev.read_at(&p, 0, n * 4 * side * side * 32), self.dev.read_at(&c, 0, n * side * side * 32))
                });
                let (pl, cl) = (4 * side * side * 32, side * side * 32);
                let (cy, cx) = coefficient_cell(round_box(b));
                object_field(&protos[k * pl..(k + 1) * pl], &coefs[k * cl + (cy * COEF_SIDE + cx) * 32..][..32], b)
            };
            let read = |j: usize| -> Option<String> {
                let det = text.get(j)?;
                let public = crate::describe::public_box(det.bbox, frame).map(|v| v as f32);
                self.read_line(img, crop_rect(public, img.width, img.height)).ok()
            };
            results[i] = Ok(assemble(&generations[k].text, &Inputs { objects: &objects, text: &text, frame, read: &read, field: &field }, opts));
        }
        let mut t = self.timings.get();
        t.decode = (t3 - t2).as_secs_f64();
        t.post = t3.elapsed().as_secs_f64();
        self.timings.set(t);
        results
    }

    /// Read the text inside `rect` of `img` on the device.
    fn read_line(&self, img: Image, rect: (usize, usize, usize, usize)) -> Result<String, crate::vision::preprocess::ImageError> {
        let (x0, y0, x1, y1) = rect;
        if x1 <= x0 || y1 <= y0 || x1 > img.width || y1 > img.height {
            return Err(crate::vision::preprocess::ImageError(format!("crop {rect:?} is empty or outside the {}x{} image", img.width, img.height)));
        }
        let mut crop = Vec::with_capacity((x1 - x0) * (y1 - y0) * 3);
        for y in y0..y1 {
            crop.extend_from_slice(&img.rgb[(y * img.width + x0) * 3..(y * img.width + x1) * 3]);
        }
        let strip = strip_from_crop(&crop, y1 - y0, x1 - x0)?;
        let wins = windows(&strip);
        let logits = self.reader.forward(&self.dev, &wins.concat(), wins.len());
        let per = crate::vision::reader::FRAMES * crate::config::READER_CLASSES;
        let logits: Vec<Vec<f32>> = logits.chunks_exact(per).map(<[f32]>::to_vec).collect();
        Ok(decode_windows(&logits, strip.w, &self.host.charset))
    }

    /// Analyse a batch of images and extract the fields of the matching form in `forms`.
    pub fn extract_batch(&self, images: &[Image], forms: &[Value], opts: &Options) -> Vec<Result<Extract, ModelError>> {
        assert_eq!(images.len(), forms.len(), "one form per image");
        let mut results: Vec<Result<Extract, ModelError>> = Vec::new();
        let boxed: Vec<Result<Resized, ModelError>> = Self::letterboxed(images);
        let valid: Vec<usize> = (0..images.len()).filter(|&i| boxed[i].is_ok()).collect();
        let resized: Vec<Resized> = boxed.iter().filter_map(|r| r.as_ref().ok().cloned()).collect();
        let batch = (!resized.is_empty()).then(|| self.analyze(&resized, opts));
        let cfg = &self.host.cfg;
        let prompt = self.host.prompt(opts.language.as_deref());
        for (i, r) in boxed.into_iter().enumerate() {
            results.push((|| {
                r?;
                let batch = batch.as_ref().expect("a valid image has a batch");
                let k = valid.iter().position(|&v| v == i).expect("valid image");
                let fields = form::parse_form(&forms[i])?;
                let (d, stride) = (cfg.dim, batch.memory.stride);
                let rows = self.dev.read_at(&batch.memory.rows, k * stride * d, stride * d);
                let mask = self.dev.read_at(&batch.memory.mask, k * stride, stride);
                let live: Vec<f32> = rows.chunks_exact(d).zip(&mask).filter(|(_, m)| **m > 0.0).flat_map(|(r, _)| r.iter().copied()).collect();
                let cross = self.host.decoder.cross_memory(&live, live.len() / d);
                Ok(form::extract(&self.host.decoder, &cross, &self.host.tokenizer, &prompt.as_ref().map_err(|e| ModelError::Request(e.to_string()))?.clone(), &fields, opts.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS))?)
            })());
        }
        results
    }
}
