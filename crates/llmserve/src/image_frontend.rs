// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements image input for the LLMs its clients serve on their own GPUs. If your team needs
// expertise in turning chat requests with pictures into batched decoder rows then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The part of a vision-language front end that does not depend on the model: which requests carry images, the images
//! a request carries, a worker thread that builds image requests (running the vision tower) while the scheduler keeps
//! decoding, and a cache of the rows of pictures seen lately.
//!
//! A family supplies [`ImageRequests::build`]: how one request with images becomes a scheduler request and the rows
//! that stand at its placeholder positions (its template, its placeholder layout, its tower). [`ImageFrontend`] does the
//! rest: a request without images takes the family's ordinary [`TextFrontend`] and its intake threads, one with images
//! is built on the image worker in arrival order.

use std::sync::{mpsc, Arc, Mutex};

use capability::{failure::tag, FailureClass, Invocation};
use data::qwen_tokenizer::QwenBpe;
use imaging::pixels::Rgb8;
use model::serve::PromptEmbeds;

use crate::{Encoding, Frontend, ParsedRequest, TextFrontend};

/// The images of `inv`, in blob order (`image`, `image1`, ... up to [`capability::blob::MAX_IMAGES`]), each refused as
/// an invalid request when its blob is not an RGB image of its declared size.
pub fn request_images(inv: &Invocation) -> Result<Vec<Rgb8>, String> {
    let mut out = Vec::new();
    for i in 0..capability::blob::MAX_IMAGES {
        let key = capability::blob::image_key(i);
        if inv.get_blob(&key).is_none() {
            break;
        }
        let invalid = |e: String| tag(FailureClass::InvalidRequest, e);
        let (hwc, w, h) = capability::blob::decode_image(inv, &key).map_err(invalid)?;
        out.push(Rgb8::from_hwc_unit(&hwc, w, h).map_err(invalid)?);
    }
    Ok(out)
}

/// Whether `inv` carries an image.
pub fn has_images(inv: &Invocation) -> bool {
    inv.get_blob("image").is_some()
}

/// The rows a tower made for the images it saw lately, found again by the decoded image itself.
///
/// An agent resends its whole history every turn, the same attached picture in it, and its rows are the same every
/// time: a tower is deterministic in the pixels. Serving them from here keeps the tower off a picture it has already
/// read, and gives the decoder rows equal to the ones its prefix cache was filed under, so the picture's span of the
/// prompt is reused with the text around it. The image is compared byte for byte (a hash of client bytes could be
/// chosen to collide), so one client's picture can never be served another's rows.
pub struct EncodedImages<R> {
    capacity: usize,
    /// Least recently used first.
    recent: Mutex<Vec<(Rgb8, Arc<R>)>>,
    encodes: std::sync::atomic::AtomicU64,
}

impl<R> EncodedImages<R> {
    /// A cache of the rows of the `capacity` most recently used images (0 keeps none).
    pub fn new(capacity: usize) -> EncodedImages<R> {
        EncodedImages { capacity, recent: Mutex::new(Vec::new()), encodes: std::sync::atomic::AtomicU64::new(0) }
    }

    /// The rows of `image`: remembered ones, else `encode`'s, which are remembered. The lock is not held while
    /// encoding, so a picture two requests bring at once may be encoded twice, never wrongly.
    pub fn rows_of(&self, image: &Rgb8, encode: impl FnOnce(&Rgb8) -> R) -> Arc<R> {
        self.try_rows_of(image, |i| Ok::<R, std::convert::Infallible>(encode(i))).unwrap_or_else(|never| match never {})
    }

    /// [`Self::rows_of`] for an encoder that can fail: a failure is returned and nothing is remembered.
    pub fn try_rows_of<E>(&self, image: &Rgb8, encode: impl FnOnce(&Rgb8) -> Result<R, E>) -> Result<Arc<R>, E> {
        let mut recent = self.recent.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(i) = recent.iter().position(|(seen, _)| seen == image) {
            let hit = recent.remove(i);
            let rows = hit.1.clone();
            recent.push(hit);
            return Ok(rows);
        }
        drop(recent);
        let rows = Arc::new(encode(image)?);
        self.encodes.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if self.capacity > 0 {
            let mut recent = self.recent.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if recent.len() >= self.capacity {
                recent.remove(0);
            }
            recent.push((image.clone(), rows.clone()));
        }
        Ok(rows)
    }

    /// How many images went through the tower because they were not remembered.
    pub fn encodes(&self) -> u64 {
        self.encodes.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// How a family builds the scheduler request for a chat request that carries at least one image.
pub trait ImageRequests: Send + Sync + 'static {
    /// The request and the rows standing at its placeholder positions. Runs the vision tower; polls `inv.cancel`.
    /// Everything a client controls is checked here and refused as an invalid request.
    fn build(&self, inv: &Invocation) -> Result<(ParsedRequest, Option<PromptEmbeds>), String>;
}

/// What a request for the image worker carries and where its result goes.
type Job = (Invocation, mpsc::Sender<Result<(ParsedRequest, Option<PromptEmbeds>), String>>);

/// The front end of a model that reads images: requests without images go through `text`, requests with images are
/// built by `B` on a worker thread of their own.
pub struct ImageFrontend<B: ImageRequests> {
    text: TextFrontend,
    build: Arc<B>,
    queue: Option<mpsc::Sender<Job>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl<B: ImageRequests> ImageFrontend<B> {
    /// `text` serves requests without images, `build` those with, on a worker thread named `worker_name`.
    pub fn new(text: TextFrontend, build: B, worker_name: &str) -> Result<ImageFrontend<B>, String> {
        let build = Arc::new(build);
        let (queue, jobs) = mpsc::channel::<Job>();
        let worker_build = build.clone();
        let worker = std::thread::Builder::new()
            .name(worker_name.to_string())
            .spawn(move || {
                // Requests are built in arrival order; the scheduler decodes other requests meanwhile.
                for (inv, reply) in jobs {
                    // The caller may have gone (the request was cancelled): nothing to deliver.
                    let _ = reply.send(worker_build.build(&inv));
                }
            })
            .map_err(|e| format!("the image worker did not start: {e}"))?;
        Ok(ImageFrontend { text, build, queue: Some(queue), worker: Some(worker) })
    }

    /// The family's builder, for callers that measure its tower.
    pub fn builder(&self) -> &B {
        &self.build
    }
}

impl<B: ImageRequests> Drop for ImageFrontend<B> {
    fn drop(&mut self) {
        drop(self.queue.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl<B: ImageRequests> Frontend for ImageFrontend<B> {
    fn tok(&self) -> &QwenBpe {
        self.text.tok()
    }
    fn stop(&self) -> &[u32] {
        self.text.stop()
    }
    fn label(&self) -> &str {
        self.text.label()
    }
    fn parse(&self, inv: &Invocation) -> Result<ParsedRequest, String> {
        if has_images(inv) {
            return self.parse_embedded(inv).map(|(req, _)| req);
        }
        self.text.parse(inv)
    }
    fn parse_embedded(&self, inv: &Invocation) -> Result<(ParsedRequest, Option<PromptEmbeds>), String> {
        if has_images(inv) {
            return self.build.build(inv);
        }
        self.text.parse_embedded(inv)
    }
    fn parse_in_background(&self, inv: &Invocation) -> Option<Encoding> {
        if !has_images(inv) {
            return self.text.parse_in_background(inv);
        }
        let (reply, rx) = mpsc::channel();
        self.queue.as_ref()?.send((inv.clone(), reply)).ok()?;
        Some(rx)
    }
    /// Building an image request runs the vision tower: it is not run ahead of the request's slot.
    fn encodes_inputs(&self) -> bool {
        true
    }
}
