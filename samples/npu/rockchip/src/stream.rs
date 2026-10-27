// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The live video pipeline: camera -> (frame tap) -> box overlay -> H.264 ->
//! RTSP, on one thread.
//!
//! One thread, because every hand-off between threads is a wake-up that adds
//! latency, and the stages after the camera are hardware that this thread
//! only waits on. Per frame it
//!
//! 1. waits for the next camera frame,
//! 2. if the detector asked for one, has the 2D accelerator make its small RGB
//!    copy (before anything is drawn on the frame),
//! 3. draws the newest detection result onto the frame,
//! 4. hands the frame to the hardware encoder and reads the access unit back,
//! 5. writes the unit to every RTSP client and records the latency.
//!
//! Detection never sits on this path: it receives frames through
//! [`Stream::frames`] and posts results to [`Stream::board`].
//!
//! Swedish Embedded AB implements low-latency camera and video pipelines for
//! embedded Linux boards for its clients. If your team needs expertise in
//! real-time video with on-device AI then you can procure our services by
//! sending an email to info@swedishembedded.com.

use crate::camera::{Camera, CameraConfig, Frame};
use crate::config::StreamConfig;
use crate::detect::{DetectionBoard, Detections};
use crate::encoder::{Encoder, EncoderConfig};
use crate::error::{err, Context, Result};
use crate::latency::{LatencyStats, Sample};
use crate::overlay::{self, Nv12Layout};
use crate::rtsp::{RtspConfig, RtspServer};
use crate::tap::{FrameReceiver, FrameTap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// How long a detection result stays on screen without a newer one.
const MAX_RESULT_AGE: Duration = Duration::from_millis(600);
/// How long to wait for a camera frame before counting a stall.
const FRAME_TIMEOUT_MS: i32 = 200;
/// Consecutive stalls (about five seconds) after which the stream gives up.
const MAX_STALLS: u32 = 25;
const ENCODE_TIMEOUT_MS: i32 = 200;

/// A running stream and the two ends of the detection seam.
pub struct Stream {
    /// Post detection results here; the video thread draws the newest one.
    pub board: Arc<DetectionBoard>,
    /// Receive downscaled frames here (see [`crate::tap`]).
    pub frames: FrameReceiver,
    stop: Arc<AtomicBool>,
    thread: JoinHandle<Result<()>>,
}

impl Stream {
    /// Starts the camera, encoder and RTSP server and returns once they are
    /// running, or with the error that stopped them.
    pub fn start(cfg: StreamConfig) -> Result<Stream> {
        let board = Arc::new(DetectionBoard::new(MAX_RESULT_AGE));
        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread = {
            let (board, stop) = (Arc::clone(&board), Arc::clone(&stop));
            std::thread::Builder::new()
                .name("video".into())
                .spawn(move || video_thread(cfg, board, stop, ready_tx))
                .context("starting the video thread")?
        };
        match ready_rx.recv() {
            Ok(Ok(frames)) => Ok(Stream { board, frames, stop, thread }),
            Ok(Err(e)) => {
                let _ = thread.join();
                Err(e)
            }
            // The thread always reports before it returns, so a closed channel
            // means it panicked during start-up.
            Err(_) => Err(err!("the video thread panicked during start-up")),
        }
    }

    /// False once the video thread has ended (stopped or failed).
    pub fn is_running(&self) -> bool {
        !self.thread.is_finished()
    }

    /// Stops the pipeline and waits for it to release the camera, the ISP and
    /// the encoder. Returns the error that ended the stream early, if any.
    pub fn stop(self) -> Result<()> {
        self.stop.store(true, Ordering::Release);
        self.thread.join().map_err(|_| err!("the video thread panicked"))?
    }
}

fn video_thread(
    cfg: StreamConfig,
    board: Arc<DetectionBoard>,
    stop: Arc<AtomicBool>,
    ready: mpsc::Sender<Result<FrameReceiver>>,
) -> Result<()> {
    // Declaration order is teardown order in reverse: the RTSP server goes
    // first, the camera last, after the encoder and tap that borrow its media.
    let setup = || -> Result<_> {
        let camera = Camera::start(CameraConfig {
            width: cfg.width,
            height: cfg.height,
            iq_dir: cfg.iq_dir.clone(),
            buf_count: 2,
        })
        .context("starting the camera")?;
        eprintln!("after camera:  {}", crate::memory::report());
        let encoder = Encoder::start(
            camera.media(),
            &EncoderConfig {
                width: cfg.width,
                height: cfg.height,
                fps: cfg.fps,
                bitrate_kbps: cfg.bitrate_kbps,
                gop: cfg.gop,
                intra_refresh_rows: cfg.intra_refresh_rows,
            },
        )
        .context("starting the encoder")?;
        eprintln!("after encoder: {}", crate::memory::report());
        let (tap, receiver) = FrameTap::new(camera.media(), cfg.width, cfg.height).context("starting the frame tap")?;
        eprintln!("after tap:     {}", crate::memory::report());
        let rtsp = RtspServer::start(RtspConfig { port: cfg.rtsp_port, path: cfg.rtsp_path.clone(), fps: cfg.fps })
            .context("starting the RTSP server")?;
        Ok((camera, encoder, tap, rtsp, receiver))
    };
    let (camera, encoder, tap, rtsp, receiver) = match setup() {
        Ok(parts) => parts,
        Err(e) => {
            let _ = ready.send(Err(err!("{e}")));
            return Err(e);
        }
    };
    eprintln!("streaming {}x{} @ {} fps on {}", cfg.width, cfg.height, cfg.fps, rtsp.url_hint());
    let _ = ready.send(Ok(receiver));

    let layout_of = |f: &Frame<'_>| {
        let i = f.info();
        Nv12Layout {
            width: i.width as usize,
            height: i.height as usize,
            stride: i.vir_width as usize,
            vir_height: i.vir_height as usize,
        }
    };
    let mut stats = LatencyStats::new(cfg.report_every);
    let mut stalls = 0u32;
    let mut seq = 0u64;
    while !stop.load(Ordering::Acquire) {
        let Some(frame) = camera.next_frame(FRAME_TIMEOUT_MS) else {
            stalls += 1;
            if stalls >= MAX_STALLS {
                return Err(err!("the camera delivered no frame for {} consecutive waits", stalls));
            }
            continue;
        };
        stalls = 0;
        let dequeued_us = now_us();
        seq += 1;

        if tap.wants_frame() {
            tap.capture(&frame, seq);
        }
        let tapped_us = now_us();
        annotate(&frame, layout_of(&frame), board.latest().as_deref());
        let drawn_us = now_us();

        if rtsp.take_idr_request() {
            encoder.request_idr();
        }
        encoder.send(frame.info()).context("sending a frame to the encoder")?;
        let encoded = encoder.receive(ENCODE_TIMEOUT_MS).context("reading the encoded frame")?;
        let encoded_us = now_us();
        for pack in encoded.packs()? {
            rtsp.publish(pack, frame.info().pts_us);
        }
        let sent_us = now_us();

        let sample = Sample {
            capture_to_send: sent_us.saturating_sub(frame.info().pts_us),
            in_process: sent_us - dequeued_us,
            tap: tapped_us - dequeued_us,
            overlay: drawn_us - tapped_us,
            encode: encoded_us - drawn_us,
            send: sent_us - encoded_us,
        };
        if let Some(report) = stats.record(sample) {
            eprintln!("latency: {report}");
        }
    }
    Ok(())
}

/// Draws the current detections onto `frame` when there are any.
fn annotate(frame: &Frame<'_>, layout: Nv12Layout, detections: Option<&Detections>) {
    let Some(detections) = detections else { return };
    let Some((luma, chroma)) = overlay::dirty_ranges(&layout, &detections.boxes) else { return };
    frame.edit(layout.len(), &[luma, chroma], |pixels| overlay::draw(pixels, &layout, &detections.boxes));
}

/// Microseconds on `CLOCK_MONOTONIC`, the clock camera frames are stamped on.
pub fn now_us() -> u64 {
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    // SAFETY: `ts` is a valid timespec to fill in.
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as u64 * 1_000_000 + ts.tv_nsec as u64 / 1000
}
