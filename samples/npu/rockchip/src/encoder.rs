// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The hardware H.264 encoder, configured for low latency.
//!
//! No B-frames (the GOP mode is plain P), constant bit rate with a one-second
//! statistics window so the rate controller reacts within a second, a
//! baseline profile every decoder accepts, and periodic intra refresh so
//! the stream never has to carry a large IDR frame after the first one.
//! Frames are handed to the hardware as they arrive and the encoded access
//! unit is read straight back; there is no queue between the two.
//!
//! Swedish Embedded AB implements low-latency video encoding for embedded
//! Linux boards for its clients. If your team needs expertise in hardware
//! encoders and rate control then you can procure our services by sending an
//! email to info@swedishembedded.com.

use crate::error::{err, Context, Result};
use crate::ffi::check;
use crate::ffi::rockit::{
    VencChnAttr, VencH264Cbr, VencIntraRefresh, VencPack, VencRcAttr, VencRecvPicParam, VencStream, VideoFrame,
    H264E_PROFILE_BASELINE, INTRA_REFRESH_ROW, RK_FMT_YUV420SP, RK_VIDEO_ID_AVC, VENC_GOPMODE_NORMALP,
    VENC_RC_MODE_H264CBR,
};
use crate::media::Media;
use std::sync::Arc;

const VENC_CHN: i32 = 0;
const START_ATTEMPTS: u32 = 3;
const START_RETRY_DELAY: std::time::Duration = std::time::Duration::from_millis(500);

#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub bitrate_kbps: u32,
    /// Frames between IDR frames. Large, because intra refresh keeps the
    /// picture recoverable and a viewer joining late asks for an IDR.
    pub gop: u32,
    /// Macroblock rows refreshed per frame; 0 turns intra refresh off.
    pub intra_refresh_rows: u32,
}

pub struct Encoder {
    media: Arc<Media>,
}

/// One encoded access unit (Annex B, start codes included), returned to the
/// encoder when dropped.
pub struct EncodedFrame<'a> {
    encoder: &'a Encoder,
    stream: VencStream,
    // Boxed so the pointer stored in `stream` stays valid when this moves.
    _pack: Box<VencPack>,
}

impl Encoder {
    pub fn start(media: Arc<Media>, cfg: &EncoderConfig) -> Result<Encoder> {
        let mut attr = VencChnAttr::default();
        attr.venc.codec = RK_VIDEO_ID_AVC;
        attr.venc.pixel_format = RK_FMT_YUV420SP;
        attr.venc.profile = H264E_PROFILE_BASELINE;
        attr.venc.by_frame = 1;
        attr.venc.max_pic_width = cfg.width;
        attr.venc.max_pic_height = cfg.height;
        attr.venc.pic_width = cfg.width;
        attr.venc.pic_height = cfg.height;
        attr.venc.vir_width = cfg.width;
        attr.venc.vir_height = cfg.height;
        // One output buffer is enough because each frame is read back before
        // the next is sent; the buffer holds one worst-case (IDR) frame.
        attr.venc.stream_buf_cnt = 1;
        attr.venc.buf_size = cfg.width * cfg.height / 2;
        attr.rc = VencRcAttr {
            rc_mode: VENC_RC_MODE_H264CBR,
            h264_cbr: VencH264Cbr {
                gop: cfg.gop,
                src_frame_rate_num: cfg.fps,
                src_frame_rate_den: 1,
                dst_frame_rate_num: cfg.fps,
                dst_frame_rate_den: 1,
                bit_rate_kbps: cfg.bitrate_kbps,
                stat_time_s: 1,
            },
            ..Default::default()
        };
        attr.gop.gop_mode = VENC_GOPMODE_NORMALP;

        // Right after a previous run released its buffers the allocator can
        // briefly report out-of-memory; that is transient, so retry a few
        // times before giving up.
        let mut last = err!("no attempt made");
        for attempt in 1..=START_ATTEMPTS {
            match Self::try_start(Arc::clone(&media), &attr, cfg) {
                Ok(encoder) => return Ok(encoder),
                Err(e) => {
                    eprintln!("encoder start attempt {attempt}/{START_ATTEMPTS} failed: {e}");
                    last = e;
                    std::thread::sleep(START_RETRY_DELAY);
                }
            }
        }
        Err(last)
    }

    fn try_start(media: Arc<Media>, attr: &VencChnAttr, cfg: &EncoderConfig) -> Result<Encoder> {
        let api = &media.api;
        // SAFETY: `attr` is a live VENC_CHN_ATTR_S.
        check(unsafe { (api.RK_MPI_VENC_CreateChn)(VENC_CHN, attr) }, "RK_MPI_VENC_CreateChn")
            .context("creating the H.264 encoder")?;
        // From here the Encoder owns the channel and Drop destroys it, also
        // when configuring fails below.
        let encoder = Encoder { media };
        encoder.configure_and_start(cfg)?;
        Ok(encoder)
    }

    fn configure_and_start(&self, cfg: &EncoderConfig) -> Result<()> {
        let api = &self.media.api;
        if cfg.intra_refresh_rows > 0 {
            let refresh = VencIntraRefresh {
                enable: 1,
                mode: INTRA_REFRESH_ROW,
                refresh_num: cfg.intra_refresh_rows,
                req_i_qp: 0,
            };
            // Intra refresh is an optimisation, not a requirement: report and
            // carry on with periodic IDR frames if the encoder refuses it.
            // SAFETY: `refresh` is a live VENC_INTRA_REFRESH_S.
            let ret = unsafe { (api.RK_MPI_VENC_SetIntraRefresh)(VENC_CHN, &refresh) };
            if ret != 0 {
                eprintln!("warning: intra refresh unavailable (status {ret:#010x}); relying on IDR frames");
            }
        }
        let recv = VencRecvPicParam { recv_pic_num: -1 };
        // SAFETY: `recv` is a live VENC_RECV_PIC_PARAM_S.
        check(unsafe { (api.RK_MPI_VENC_StartRecvFrame)(VENC_CHN, &recv) }, "RK_MPI_VENC_StartRecvFrame")
            .context("starting the H.264 encoder")
    }

    /// Hands a captured frame to the hardware. Returns as soon as the frame is
    /// queued, so the caller can do other work while it is encoded.
    pub fn send(&self, frame: &VideoFrame) -> Result<()> {
        // SAFETY: `frame` is a live VIDEO_FRAME_INFO_S and its block is held.
        check(unsafe { (self.media.api.RK_MPI_VENC_SendFrame)(VENC_CHN, frame, 0) }, "RK_MPI_VENC_SendFrame")
    }

    /// Waits up to `timeout_ms` for the next encoded access unit.
    pub fn receive(&self, timeout_ms: i32) -> Result<EncodedFrame<'_>> {
        let mut pack = Box::new(
            // SAFETY: VENC_PACK_S is integers and raw pointers; zero is valid.
            unsafe { std::mem::zeroed::<VencPack>() },
        );
        let mut stream = VencStream::new(&mut *pack);
        // SAFETY: `stream` points at a live pack the encoder fills in.
        check(
            unsafe { (self.media.api.RK_MPI_VENC_GetStream)(VENC_CHN, &mut stream, timeout_ms) },
            "RK_MPI_VENC_GetStream",
        )?;
        Ok(EncodedFrame { encoder: self, stream, _pack: pack })
    }

    /// Asks for an IDR frame so a newly joined viewer can start decoding.
    pub fn request_idr(&self) {
        // SAFETY: plain call on the live channel.
        let ret = unsafe { (self.media.api.RK_MPI_VENC_RequestIDR)(VENC_CHN, 0) };
        if ret != 0 {
            eprintln!("warning: RK_MPI_VENC_RequestIDR returned {ret:#010x}");
        }
    }
}

impl Drop for Encoder {
    fn drop(&mut self) {
        let api = &self.media.api;
        // SAFETY: balances start(); nothing more can be done about a failure.
        unsafe {
            (api.RK_MPI_VENC_StopRecvFrame)(VENC_CHN);
            (api.RK_MPI_VENC_DestroyChn)(VENC_CHN);
        }
    }
}

impl EncodedFrame<'_> {
    /// The encoded bytes of every pack of this access unit, in order.
    pub fn packs(&self) -> Result<Vec<&[u8]>> {
        let count = self.stream.pack_count as usize;
        if count == 0 || self.stream.pack.is_null() {
            return Err(err!("the encoder returned a stream with no packs"));
        }
        // SAFETY: the encoder fills `count` consecutive packs behind the
        // pointer; we only supplied room for one, which by-frame mode uses.
        let packs = unsafe { std::slice::from_raw_parts(self.stream.pack, count.min(1)) };
        Ok(packs
            .iter()
            .map(|p| {
                // SAFETY: the pack's block is valid until the stream is
                // released, and its address is the start of `len` bytes of
                // encoded data. `offset` is where the frame sits inside the
                // encoder's output ring (measured: it grows frame by frame),
                // not an offset from the block address, so it is not added.
                unsafe {
                    let start = (self.encoder.media.api.RK_MPI_MB_Handle2VirAddr)(p.mb_blk) as *const u8;
                    std::slice::from_raw_parts(start, p.len as usize)
                }
            })
            .collect())
    }
}

impl Drop for EncodedFrame<'_> {
    fn drop(&mut self) {
        // SAFETY: `stream` came from a successful GetStream on this channel.
        let ret = unsafe { (self.encoder.media.api.RK_MPI_VENC_ReleaseStream)(VENC_CHN, &mut self.stream) };
        if ret != 0 {
            eprintln!("warning: RK_MPI_VENC_ReleaseStream returned {ret:#010x}");
        }
    }
}
