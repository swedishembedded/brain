// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The frame tap: a small RGB copy of the live video for the detector.
//!
//! The detector asks for a frame ([`FrameReceiver::recv`]); the next camera
//! frame is then scaled and colour-converted by the board's 2D accelerator
//! (RGA) into a [`TAP_SIZE`] x [`TAP_SIZE`] RGB888 image, letterboxed so the
//! aspect ratio is preserved (grey padding, mapping back to the stream in
//! [`TapFrame::letterbox`]). Nothing is converted while the detector is busy:
//! the video thread checks [`FrameTap::wants_frame`] and skips the tap, so the
//! detector always receives the freshest frame and a slow detector costs the
//! video path nothing.
//!
//! There is exactly one tap buffer (300 KB of the board's scarce CMA). While
//! a [`TapFrame`] is held, no new frame is produced.
//!
//! Swedish Embedded AB implements real-time edge AI pipelines for embedded
//! Linux boards for its clients. If your team needs expertise in feeding
//! accelerators from a live camera then you can procure our services by
//! sending an email to info@swedishembedded.com.

use crate::camera::Frame;
use crate::detect::Letterbox;
use crate::error::{err, Context, Result};
use crate::ffi::rga::{Rga, RgaInfo, RgaRect, RK_FORMAT_RGB_888, RK_FORMAT_YCBCR_420_SP};
use crate::ffi::rockit::{MbBlk, MMZ_ALLOC_CMA_CACHEABLE, MMZ_SYNC_READONLY, MMZ_SYNC_RW};
use crate::ffi::check;
use crate::media::Media;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::Duration;

/// Side of the square tap image, in pixels.
pub const TAP_SIZE: u32 = 320;
const TAP_BYTES: usize = (TAP_SIZE * TAP_SIZE * 3) as usize;
/// The letterbox padding value (the usual 114 grey of YOLO preprocessing).
const PAD_GREY: u8 = 114;

/// Which camera frame a tap image came from.
#[derive(Clone, Copy, Debug)]
pub struct TapMeta {
    /// The pipeline's frame counter.
    pub seq: u64,
    /// Sensor timestamp in microseconds on `CLOCK_MONOTONIC`.
    pub pts_us: u64,
    pub stream_width: u32,
    pub stream_height: u32,
    pub letterbox: Letterbox,
}

#[derive(Clone, Copy)]
enum Slot {
    /// The buffer holds nothing of interest.
    Idle,
    /// The video thread is converting a frame into it.
    Writing,
    /// A converted frame waits for the detector.
    Ready(TapMeta),
    /// The detector holds the buffer.
    Leased,
}

struct State {
    slot: Slot,
    /// The detector is blocked in `recv` and wants the next frame.
    wanted: bool,
}

/// The tap buffer and its hand-off state, shared by producer and consumer.
struct Shared {
    media: Arc<Media>,
    blk: MbBlk,
    ptr: *mut u8,
    state: Mutex<State>,
    ready: Condvar,
}

// SAFETY: `ptr` addresses a CMA buffer owned by `Shared`. Access to its
// contents is serialised by `state`: only the producer writes (in `Writing`)
// and only the lease holder reads (in `Leased`).
unsafe impl Send for Shared {}
unsafe impl Sync for Shared {}

impl Shared {
    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn sync(&self, which: unsafe extern "C" fn(MbBlk, u32, u32, u32) -> i32, flags: u32) {
        // SAFETY: the block is live and the range is within it.
        let ret = unsafe { which(self.blk, 0, TAP_BYTES as u32, flags) };
        if ret != 0 {
            eprintln!("warning: tap buffer cache maintenance returned {ret:#010x}");
        }
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: balances the allocation in `FrameTap::new`.
        unsafe { (self.media.api.RK_MPI_MMZ_Free)(self.blk) };
    }
}

/// The video thread's end of the tap.
pub struct FrameTap {
    shared: Arc<Shared>,
    rga: Rga,
    fd: i32,
    letterbox: Letterbox,
    stream: (u32, u32),
}

/// The detector's end of the tap.
#[derive(Clone)]
pub struct FrameReceiver {
    shared: Arc<Shared>,
}

/// A tap image on loan to the detector. The tap produces no new image until
/// it is dropped, so a detector that keeps it for the length of an inference
/// can feed the pixels to the NPU without copying them.
pub struct TapFrame {
    shared: Arc<Shared>,
    meta: TapMeta,
}

impl FrameTap {
    /// Allocates the tap buffer and opens the 2D accelerator.
    /// `stream_width` x `stream_height` is the camera frame size.
    pub fn new(media: Arc<Media>, stream_width: u32, stream_height: u32) -> Result<(FrameTap, FrameReceiver)> {
        let rga = Rga::load().context("loading the RGA library")?;
        // SAFETY: no arguments.
        check(unsafe { (rga.c_RkRgaInit)() }, "c_RkRgaInit").context("opening the 2D accelerator")?;

        let mut blk: MbBlk = std::ptr::null_mut();
        // SAFETY: `blk` receives the allocated handle.
        check(
            unsafe { (media.api.RK_MPI_MMZ_Alloc)(&mut blk, TAP_BYTES as u32, MMZ_ALLOC_CMA_CACHEABLE) },
            "RK_MPI_MMZ_Alloc",
        )
        .context("allocating the tap buffer")?;
        // SAFETY: `blk` is the live handle just allocated.
        let (ptr, fd) = unsafe {
            ((media.api.RK_MPI_MMZ_Handle2VirAddr)(blk) as *mut u8, (media.api.RK_MPI_MMZ_Handle2Fd)(blk))
        };
        if ptr.is_null() || fd < 0 {
            // SAFETY: frees the block allocated above.
            unsafe { (media.api.RK_MPI_MMZ_Free)(blk) };
            return Err(err!("the tap buffer has no CPU mapping or dma-buf"));
        }
        let shared = Arc::new(Shared {
            media,
            blk,
            ptr,
            state: Mutex::new(State { slot: Slot::Idle, wanted: false }),
            ready: Condvar::new(),
        });
        // The accelerator only writes the scaled window, so the padding is
        // filled once and stays.
        // SAFETY: `ptr` addresses TAP_BYTES of writable memory.
        unsafe { std::ptr::write_bytes(ptr, PAD_GREY, TAP_BYTES) };
        shared.sync(shared.media.api.RK_MPI_MMZ_FlushCacheEnd, MMZ_SYNC_RW);

        let tap = FrameTap {
            shared: Arc::clone(&shared),
            rga,
            fd,
            letterbox: Letterbox::fit(stream_width, stream_height, TAP_SIZE),
            stream: (stream_width, stream_height),
        };
        Ok((tap, FrameReceiver { shared }))
    }

    /// True if the detector is waiting for a frame. Cheap; call it per frame.
    pub fn wants_frame(&self) -> bool {
        let st = self.shared.state();
        st.wanted && matches!(st.slot, Slot::Idle | Slot::Ready(_))
    }

    /// Converts `frame` for the detector. Call only when [`wants_frame`]
    /// said so, before drawing on the frame. A failure is reported and the
    /// detector simply keeps waiting for the next frame.
    ///
    /// [`wants_frame`]: FrameTap::wants_frame
    pub fn capture(&self, frame: &Frame<'_>, seq: u64) {
        {
            let mut st = self.shared.state();
            if matches!(st.slot, Slot::Leased | Slot::Writing) {
                return;
            }
            st.slot = Slot::Writing;
        }
        let ok = self.convert(frame);
        let mut st = self.shared.state();
        if ok {
            let meta = TapMeta {
                seq,
                pts_us: frame.info().pts_us,
                stream_width: self.stream.0,
                stream_height: self.stream.1,
                letterbox: self.letterbox,
            };
            st.slot = Slot::Ready(meta);
            self.shared.ready.notify_one();
        } else {
            st.slot = Slot::Idle;
        }
    }

    fn convert(&self, frame: &Frame<'_>) -> bool {
        let info = frame.info();
        let lb = &self.letterbox;
        let mut src = RgaInfo::from_fd(
            frame.dma_fd(),
            RgaRect {
                width: info.width as i32,
                height: info.height as i32,
                wstride: info.vir_width as i32,
                hstride: info.vir_height as i32,
                format: RK_FORMAT_YCBCR_420_SP,
                ..Default::default()
            },
        );
        let mut dst = RgaInfo::from_fd(
            self.fd,
            RgaRect {
                xoffset: lb.pad_x as i32,
                yoffset: lb.pad_y as i32,
                width: lb.scaled_width as i32,
                height: lb.scaled_height as i32,
                wstride: TAP_SIZE as i32,
                hstride: TAP_SIZE as i32,
                format: RK_FORMAT_RGB_888,
                ..Default::default()
            },
        );
        // SAFETY: both descriptors are live and describe dma-bufs that stay
        // valid for the duration of the (synchronous) blit.
        let ret = unsafe { (self.rga.c_RkRgaBlit)(&mut src, &mut dst, std::ptr::null_mut()) };
        if ret != 0 {
            eprintln!("warning: RGA blit for the detector tap failed ({ret})");
            return false;
        }
        // The accelerator wrote behind the CPU's cache: drop stale lines
        // before the detector reads.
        self.shared.sync(self.shared.media.api.RK_MPI_MMZ_FlushCacheStart, MMZ_SYNC_READONLY);
        true
    }
}

impl Drop for FrameTap {
    fn drop(&mut self) {
        // Wake a detector blocked in `recv` so it can see the tap is gone.
        self.shared.state().wanted = false;
        self.shared.ready.notify_all();
        // SAFETY: balances the successful c_RkRgaInit in `new`.
        unsafe { (self.rga.c_RkRgaDeInit)() };
    }
}

impl FrameReceiver {
    /// Waits up to `timeout` for the next camera frame, scaled for detection.
    /// Returns `None` on timeout. While this waits, the video thread converts
    /// the next frame it sees; nothing is converted otherwise.
    pub fn recv(&self, timeout: Duration) -> Option<TapFrame> {
        let mut st = self.shared.state();
        st.wanted = true;
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Slot::Ready(meta) = st.slot {
                st.slot = Slot::Leased;
                st.wanted = false;
                return Some(TapFrame { shared: Arc::clone(&self.shared), meta });
            }
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                st.wanted = false;
                return None;
            }
            st = self.shared.ready.wait_timeout(st, left).unwrap_or_else(|e| e.into_inner()).0;
        }
    }
}

impl TapFrame {
    /// Where this image came from and how it maps back to the stream.
    pub fn meta(&self) -> &TapMeta {
        &self.meta
    }

    /// The image: `TAP_SIZE * TAP_SIZE * 3` bytes of RGB, row by row.
    pub fn rgb(&self) -> &[u8] {
        // SAFETY: while Leased the producer does not touch the buffer, and
        // the allocation lives as long as `shared`.
        unsafe { std::slice::from_raw_parts(self.shared.ptr, TAP_BYTES) }
    }
}

impl Drop for TapFrame {
    fn drop(&mut self) {
        self.shared.sync(self.shared.media.api.RK_MPI_MMZ_FlushCacheEnd, MMZ_SYNC_READONLY);
        self.shared.state().slot = Slot::Idle;
    }
}
