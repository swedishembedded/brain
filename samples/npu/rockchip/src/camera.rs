// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The camera: the sensor's ISP 3A loop (`rkaiq`) plus one video-input
//! channel delivering NV12 frames.
//!
//! Swedish Embedded AB implements low-latency camera and video pipelines for
//! embedded Linux boards for its clients. If your team needs expertise in
//! ISP tuning and sensor bring-up then you can procure our services by
//! sending an email to info@swedishembedded.com.

use crate::error::{err, Context, Result};
use crate::ffi::rkaiq::{Rkaiq, AIQ_WORKING_MODE_NORMAL, STATIC_INFO_SIZE};
use crate::ffi::rockit::{
    ViChnAttr, ViDevAttr, ViDevBindPipe, VideoFrame, MMZ_SYNC_READONLY, MMZ_SYNC_WRITEONLY, RK_FMT_YUV420SP,
    VI_V4L2_MEMORY_TYPE_DMABUF,
};
use crate::ffi::check;
use crate::media::Media;
use std::ffi::{c_void, CStr, CString};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// The board has one sensor: device 0, pipe 0, channel 0.
const VI_PIPE: i32 = 0;
const VI_CHN: i32 = 0;
const VI_DEV: i32 = 0;

#[derive(Clone, Debug)]
pub struct CameraConfig {
    pub width: u32,
    pub height: u32,
    /// Directory holding the sensor tuning (IQ) files.
    pub iq_dir: String,
    /// VI buffers. Two is the minimum that lets the ISP write one frame while
    /// the pipeline holds another.
    pub buf_count: u32,
}

/// A running camera. Dropping it stops capture and the ISP, in the order the
/// vendor requires.
pub struct Camera {
    media: Arc<Media>,
    aiq: Rkaiq,
    aiq_ctx: *mut c_void,
    cfg: CameraConfig,
}

/// A captured frame, returned to the VI driver when dropped.
pub struct Frame<'a> {
    camera: &'a Camera,
    info: VideoFrame,
}

impl Camera {
    pub fn start(cfg: CameraConfig) -> Result<Camera> {
        let aiq = Rkaiq::load().context("loading the rkaiq ISP library")?;

        let mut static_info = [0u8; STATIC_INFO_SIZE];
        // SAFETY: the buffer is the measured size of rk_aiq_static_info_t.
        check(
            unsafe { (aiq.rk_aiq_uapi2_sysctl_enumStaticMetas)(0, static_info.as_mut_ptr()) },
            "rk_aiq_uapi2_sysctl_enumStaticMetas",
        )
        .context("looking for the camera sensor")?;
        let sensor = CStr::from_bytes_until_nul(&static_info)
            .map_err(|_| err!("the sensor name is not NUL-terminated"))?;
        let iq_dir = CString::new(cfg.iq_dir.as_str()).map_err(|_| err!("iq dir contains a NUL"))?;

        // SAFETY: both strings are NUL-terminated and outlive the call; the
        // two callbacks are optional and passed as null.
        let aiq_ctx = unsafe {
            (aiq.rk_aiq_uapi2_sysctl_init)(sensor.as_ptr(), iq_dir.as_ptr(), std::ptr::null(), std::ptr::null())
        };
        if aiq_ctx.is_null() {
            return Err(err!(
                "starting the ISP for sensor {sensor:?} with IQ files in {}: rk_aiq_uapi2_sysctl_init failed",
                cfg.iq_dir
            ));
        }
        // From here on, `camera` owns the ISP context and Drop releases it.
        let media = match Media::init() {
            Ok(m) => Arc::new(m),
            Err(e) => {
                // SAFETY: aiq_ctx came from a successful init above.
                unsafe { (aiq.rk_aiq_uapi2_sysctl_deinit)(aiq_ctx) };
                return Err(e);
            }
        };
        let mut camera = Camera { media, aiq, aiq_ctx, cfg };
        camera.run_isp().context("starting the ISP 3A loop")?;
        camera.start_vi().context("starting video input")?;
        Ok(camera)
    }

    pub fn media(&self) -> Arc<Media> {
        Arc::clone(&self.media)
    }

    fn run_isp(&mut self) -> Result<()> {
        // SAFETY: aiq_ctx is a live context from rk_aiq_uapi2_sysctl_init.
        unsafe {
            check(
                (self.aiq.rk_aiq_uapi2_sysctl_prepare)(self.aiq_ctx, 0, 0, AIQ_WORKING_MODE_NORMAL),
                "rk_aiq_uapi2_sysctl_prepare",
            )?;
            check((self.aiq.rk_aiq_uapi2_sysctl_start)(self.aiq_ctx), "rk_aiq_uapi2_sysctl_start")
        }
    }

    fn start_vi(&mut self) -> Result<()> {
        let api = &self.media.api;
        let mut dev = ViDevAttr::default();
        // SAFETY: all pointers refer to live locals of the declared types.
        unsafe {
            // An unconfigured device reports an error from GetDevAttr; that is
            // the signal to set its (default) attributes once.
            if (api.RK_MPI_VI_GetDevAttr)(VI_DEV, &mut dev) != 0 {
                check((api.RK_MPI_VI_SetDevAttr)(VI_DEV, &dev), "RK_MPI_VI_SetDevAttr")?;
            }
            if (api.RK_MPI_VI_GetDevIsEnable)(VI_DEV) != 0 {
                check((api.RK_MPI_VI_EnableDev)(VI_DEV), "RK_MPI_VI_EnableDev")?;
                let mut bind = ViDevBindPipe { num: 1, ..Default::default() };
                bind.pipe_id[0] = VI_PIPE;
                check((api.RK_MPI_VI_SetDevBindPipe)(VI_DEV, &bind), "RK_MPI_VI_SetDevBindPipe")?;
            }

            let mut chn = ViChnAttr::zeroed();
            chn.isp_opt.buf_count = self.cfg.buf_count;
            chn.isp_opt.memory_type = VI_V4L2_MEMORY_TYPE_DMABUF;
            chn.size.width = self.cfg.width;
            chn.size.height = self.cfg.height;
            chn.pixel_format = RK_FMT_YUV420SP;
            // The depth is how many frames the application may hold at once.
            chn.depth = 2;
            check((api.RK_MPI_VI_SetChnAttr)(VI_PIPE, VI_CHN, &chn), "RK_MPI_VI_SetChnAttr")?;
            check((api.RK_MPI_VI_EnableChn)(VI_PIPE, VI_CHN), "RK_MPI_VI_EnableChn")
        }
    }

    /// Waits up to `timeout_ms` for the next frame. Any non-zero status is
    /// reported as `None` (normally a timeout), so a stalled sensor shows up
    /// as a run of timeouts the caller can count and give up on.
    pub fn next_frame(&self, timeout_ms: i32) -> Option<Frame<'_>> {
        let mut info = VideoFrame::zeroed();
        // SAFETY: `info` is a live VIDEO_FRAME_INFO_S.
        let ret = unsafe { (self.media.api.RK_MPI_VI_GetChnFrame)(VI_PIPE, VI_CHN, &mut info, timeout_ms) };
        (ret == 0).then_some(Frame { camera: self, info })
    }
}

impl Frame<'_> {
    pub fn info(&self) -> &VideoFrame {
        &self.info
    }

    /// The NV12 pixels: the luma plane then the interleaved chroma plane.
    pub fn base(&self) -> *mut u8 {
        // SAFETY: the block handle is valid while the frame is held.
        unsafe { (self.camera.media.api.RK_MPI_MB_Handle2VirAddr)(self.info.mb_blk) as *mut u8 }
    }

    /// Lets the CPU modify the frame's bytes. The listed byte `ranges` are
    /// made coherent with the hardware before `edit` runs (stale cache lines
    /// dropped) and written back after it, so the encoder sees the edit and
    /// the neighbouring pixels survive. `edit` receives `len` bytes from the
    /// start of the frame and must only touch bytes inside `ranges`.
    pub fn edit<R>(&self, len: usize, ranges: &[Range<usize>], edit: impl FnOnce(&mut [u8]) -> R) -> R {
        let api = &self.camera.media.api;
        let blk = self.info.mb_blk;
        // The CPU maps these frames cached although the block's own
        // IsCacheable answer says otherwise (measured: without maintenance
        // part of every drawn row is lost). Invalidate before the edit so no
        // stale line is merged back, clean after it so the encoder sees it.
        for r in ranges {
            // SAFETY: the block is held and the range lies within the frame.
            let ret = unsafe { (api.RK_MPI_MMZ_FlushCacheStart)(blk, r.start as u32, r.len() as u32, MMZ_SYNC_READONLY) };
            warn_once(ret, "RK_MPI_MMZ_FlushCacheStart");
        }
        // SAFETY: the mapping covers the whole frame and nothing else writes
        // it while this frame is held.
        let result = edit(unsafe { std::slice::from_raw_parts_mut(self.base(), len) });
        for r in ranges {
            // SAFETY: as above.
            let ret = unsafe { (api.RK_MPI_MMZ_FlushCacheEnd)(blk, r.start as u32, r.len() as u32, MMZ_SYNC_WRITEONLY) };
            warn_once(ret, "RK_MPI_MMZ_FlushCacheEnd");
        }
        result
    }

    /// The dma-buf behind the frame, for hardware that reads it directly.
    pub fn dma_fd(&self) -> i32 {
        // SAFETY: the block handle is valid while the frame is held.
        unsafe { (self.camera.media.api.RK_MPI_MB_Handle2Fd)(self.info.mb_blk) }
    }
}

static CACHE_WARNED: AtomicBool = AtomicBool::new(false);

/// Reports a non-zero cache-maintenance status the first time it happens.
fn warn_once(ret: i32, call: &str) {
    if ret != 0 && !CACHE_WARNED.swap(true, Ordering::Relaxed) {
        eprintln!("warning: {call} returned {ret:#010x} (further occurrences not reported)");
    }
}

impl Drop for Frame<'_> {
    fn drop(&mut self) {
        // SAFETY: `info` came from a successful GetChnFrame on this channel.
        let ret = unsafe { (self.camera.media.api.RK_MPI_VI_ReleaseChnFrame)(VI_PIPE, VI_CHN, &self.info) };
        if ret != 0 {
            eprintln!("warning: RK_MPI_VI_ReleaseChnFrame returned {ret:#010x}");
        }
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        let api = &self.media.api;
        // SAFETY: balances start(): VI first, then the ISP, as the vendor's
        // own sample does. Failures here cannot be acted on beyond reporting.
        unsafe {
            (api.RK_MPI_VI_DisableChn)(VI_PIPE, VI_CHN);
            (api.RK_MPI_VI_DisableDev)(VI_DEV);
            (self.aiq.rk_aiq_uapi2_sysctl_stop)(self.aiq_ctx, false);
            (self.aiq.rk_aiq_uapi2_sysctl_deinit)(self.aiq_ctx);
        }
    }
}
