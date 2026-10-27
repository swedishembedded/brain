// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Declarations for the parts of Rockchip's rockit media library
//! (`librockit.so`: VI, VENC, MMZ) that the camera and encoder use.
//!
//! Provenance: the types and signatures are derived from Rockchip's rockit
//! headers (`rk_comm_vi.h`, `rk_comm_venc.h`, `rk_comm_video.h`,
//! `rk_comm_rc.h`, `rk_mpi_*.h`), which are licensed
//! `GPL-2.0 WITH Linux-syscall-note OR Apache-2.0`. The structs are mirrored
//! field for field. Every layout this sample relies on was measured by
//! compiling the vendor headers with the board's own uClibc gcc, and the
//! 32-bit assertions at the bottom keep the Rust side honest.
//!
//! Swedish Embedded AB implements low-latency camera and video pipelines for
//! embedded Linux boards for its clients. If your team needs expertise in
//! vendor media stacks then you can procure our services by sending an email
//! to info@swedishembedded.com.

use super::dl::dl_api;
use std::ffi::c_void;

pub type MbBlk = *mut c_void;

/// `RK_FMT_YUV420SP`: NV12, the format VI delivers and VENC takes.
pub const RK_FMT_YUV420SP: u32 = 0;
/// `RK_VIDEO_ID_AVC`.
pub const RK_VIDEO_ID_AVC: u32 = 8;
/// `H264E_PROFILE_HIGH` is avoided: baseline needs no CABAC and decodes
/// everywhere, which matters more than the few percent it costs in bitrate.
pub const H264E_PROFILE_BASELINE: u32 = 66;
pub const VENC_RC_MODE_H264CBR: u32 = 1;
pub const VENC_GOPMODE_NORMALP: u32 = 1;
pub const VI_V4L2_MEMORY_TYPE_DMABUF: u32 = 4;
pub const INTRA_REFRESH_ROW: u32 = 0;

/// `RK_MMZ_ALLOC_TYPE_CMA | RK_MMZ_ALLOC_CACHEABLE`.
pub const MMZ_ALLOC_CMA_CACHEABLE: u32 = 0x1;
/// `RK_MMZ_SYNC_RW`.
pub const MMZ_SYNC_RW: u32 = 2;
/// `RK_MMZ_SYNC_READONLY`: before the CPU reads, drop stale cache lines.
pub const MMZ_SYNC_READONLY: u32 = 0;
/// `RK_MMZ_SYNC_WRITEONLY`: after the CPU wrote, write the lines back.
pub const MMZ_SYNC_WRITEONLY: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

/// `VI_DEV_ATTR_S`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ViDevAttr {
    pub intf_mode: u32,
    pub work_mode: u32,
    pub data_seq: u32,
    pub input_data_type: u32,
    pub max_size: Size,
    pub data_rate: u32,
    pub pix_fmt: u32,
    pub mem_mode: u32,
    pub buf_type: u32,
    pub buf_count: u32,
    pub hdr_mode: u32,
    pub crop_rect: Rect,
}

/// `VI_DEV_BIND_PIPE_S`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ViDevBindPipe {
    pub num: u32,
    pub pipe_id: [i32; 16],
    pub data_offline: u32,
    pub user_start_pipe: [u32; 16],
}

/// `VI_ISP_OPT_S`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ViIspOpt {
    pub buf_count: u32,
    pub buf_size: u32,
    pub capture_type: u32,
    pub memory_type: u32,
    pub entity_name: [u8; 32],
    pub no_use_libv4l2: u32,
    pub max_size: Size,
    pub window: Rect,
}

/// `VI_CHN_ATTR_S`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ViChnAttr {
    pub size: Size,
    pub pixel_format: u32,
    pub dynamic_range: u32,
    pub video_format: u32,
    pub compress_mode: u32,
    pub mirror: u32,
    pub flip: u32,
    pub depth: u32,
    pub src_frame_rate: i32,
    pub dst_frame_rate: i32,
    pub alloc_buf_type: u32,
    pub isp_opt: ViIspOpt,
    pub share_buf_chn: [i32; 3],
}

impl ViChnAttr {
    pub fn zeroed() -> Self {
        // SAFETY: plain integers and arrays; all-zero is the C default.
        unsafe { std::mem::zeroed() }
    }
}

/// `VIDEO_FRAME_INFO_S` (a `VIDEO_FRAME_S` and nothing else).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VideoFrame {
    pub mb_blk: MbBlk,
    pub width: u32,
    pub height: u32,
    pub vir_width: u32,
    pub vir_height: u32,
    pub field: u32,
    pub pixel_format: u32,
    pub video_format: u32,
    pub compress_mode: u32,
    pub dynamic_range: u32,
    pub color_gamut: u32,
    pub vir_addr: [*mut c_void; 2],
    pub time_ref: u32,
    /// Capture time in microseconds on `CLOCK_MONOTONIC`.
    pub pts_us: u64,
    pub private_data: u64,
    pub frame_flag: u32,
}

impl VideoFrame {
    pub fn zeroed() -> Self {
        // SAFETY: integers and raw pointers; all-zero is the C default.
        unsafe { std::mem::zeroed() }
    }
}

/// `VENC_H264_CBR_S`, the one rate-control arm this sample uses.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VencH264Cbr {
    pub gop: u32,
    pub src_frame_rate_num: u32,
    pub src_frame_rate_den: u32,
    pub dst_frame_rate_num: u32,
    pub dst_frame_rate_den: u32,
    pub bit_rate_kbps: u32,
    pub stat_time_s: u32,
}

/// `VENC_ATTR_S`, with the codec-specific union kept as opaque padding.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VencAttr {
    pub codec: u32,
    pub max_pic_width: u32,
    pub max_pic_height: u32,
    pub pixel_format: u32,
    pub mirror: u32,
    pub buf_size: u32,
    pub profile: u32,
    pub by_frame: u32,
    pub pic_width: u32,
    pub pic_height: u32,
    pub vir_width: u32,
    pub vir_height: u32,
    pub stream_buf_cnt: u32,
    pub codec_union: [u32; 7],
}

/// `VENC_RC_ATTR_S` holding an H.264 CBR arm (the union is 36 bytes wide).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VencRcAttr {
    pub rc_mode: u32,
    pub h264_cbr: VencH264Cbr,
    pub union_tail: [u32; 2],
}

/// `VENC_GOP_ATTR_S`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VencGopAttr {
    pub gop_mode: u32,
    pub vir_idr_len: i32,
    pub max_ltr_count: u32,
    pub tsvc_preload: u32,
}

/// `VENC_CHN_ATTR_S`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct VencChnAttr {
    pub venc: VencAttr,
    pub rc: VencRcAttr,
    pub gop: VencGopAttr,
}

/// `VENC_RECV_PIC_PARAM_S`.
#[repr(C)]
pub struct VencRecvPicParam {
    pub recv_pic_num: i32,
}

/// `VENC_INTRA_REFRESH_S`.
#[repr(C)]
pub struct VencIntraRefresh {
    pub enable: u32,
    pub mode: u32,
    pub refresh_num: u32,
    pub req_i_qp: u32,
}

/// `VENC_PACK_S`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VencPack {
    pub mb_blk: MbBlk,
    pub len: u32,
    pub pts_us: u64,
    pub frame_end: u32,
    pub stream_end: u32,
    pub data_type: u32,
    pub offset: u32,
    pub data_num: u32,
    pub pack_info: [[u32; 3]; 8],
}

/// `VENC_STREAM_S`: the leading fields are mirrored, the codec statistics
/// that follow are opaque padding up to the measured 400-byte size.
#[repr(C, align(8))]
pub struct VencStream {
    pub pack: *mut VencPack,
    _pad0: [u8; 12],
    pub pack_count: u32,
    pub seq: u32,
    _stats: [u8; 376],
}

impl VencStream {
    pub fn new(pack: *mut VencPack) -> Self {
        VencStream { pack, _pad0: [0; 12], pack_count: 0, seq: 0, _stats: [0; 376] }
    }
}

dl_api! {
    pub struct Rockit from "/oem/usr/lib/librockit.so" {
        fn RK_MPI_SYS_Init() -> i32;
        fn RK_MPI_SYS_Exit() -> i32;

        fn RK_MPI_VI_GetDevAttr(dev: i32, attr: *mut ViDevAttr) -> i32;
        fn RK_MPI_VI_SetDevAttr(dev: i32, attr: *const ViDevAttr) -> i32;
        fn RK_MPI_VI_GetDevIsEnable(dev: i32) -> i32;
        fn RK_MPI_VI_EnableDev(dev: i32) -> i32;
        fn RK_MPI_VI_DisableDev(dev: i32) -> i32;
        fn RK_MPI_VI_SetDevBindPipe(dev: i32, bind: *const ViDevBindPipe) -> i32;
        fn RK_MPI_VI_SetChnAttr(pipe: i32, chn: i32, attr: *const ViChnAttr) -> i32;
        fn RK_MPI_VI_EnableChn(pipe: i32, chn: i32) -> i32;
        fn RK_MPI_VI_DisableChn(pipe: i32, chn: i32) -> i32;
        fn RK_MPI_VI_GetChnFrame(pipe: i32, chn: i32, frame: *mut VideoFrame, timeout_ms: i32) -> i32;
        fn RK_MPI_VI_ReleaseChnFrame(pipe: i32, chn: i32, frame: *const VideoFrame) -> i32;

        fn RK_MPI_VENC_CreateChn(chn: i32, attr: *const VencChnAttr) -> i32;
        fn RK_MPI_VENC_DestroyChn(chn: i32) -> i32;
        fn RK_MPI_VENC_StartRecvFrame(chn: i32, param: *const VencRecvPicParam) -> i32;
        fn RK_MPI_VENC_StopRecvFrame(chn: i32) -> i32;
        fn RK_MPI_VENC_SendFrame(chn: i32, frame: *const VideoFrame, timeout_ms: i32) -> i32;
        fn RK_MPI_VENC_GetStream(chn: i32, stream: *mut VencStream, timeout_ms: i32) -> i32;
        fn RK_MPI_VENC_ReleaseStream(chn: i32, stream: *mut VencStream) -> i32;
        fn RK_MPI_VENC_RequestIDR(chn: i32, instant: u32) -> i32;
        fn RK_MPI_VENC_SetIntraRefresh(chn: i32, refresh: *const VencIntraRefresh) -> i32;

        fn RK_MPI_MB_Handle2VirAddr(blk: MbBlk) -> *mut c_void;
        fn RK_MPI_MB_Handle2Fd(blk: MbBlk) -> i32;
        fn RK_MPI_MB_GetSize(blk: MbBlk) -> u64;

        fn RK_MPI_MMZ_Alloc(blk: *mut MbBlk, len: u32, flags: u32) -> i32;
        fn RK_MPI_MMZ_Free(blk: MbBlk) -> i32;
        fn RK_MPI_MMZ_Handle2VirAddr(blk: MbBlk) -> *mut c_void;
        fn RK_MPI_MMZ_Handle2Fd(blk: MbBlk) -> i32;
        fn RK_MPI_MMZ_IsCacheable(blk: MbBlk) -> i32;
        fn RK_MPI_MMZ_FlushCacheStart(blk: MbBlk, offset: u32, len: u32, flags: u32) -> i32;
        fn RK_MPI_MMZ_FlushCacheEnd(blk: MbBlk, offset: u32, len: u32, flags: u32) -> i32;
    }
}

// The layouts below were measured with the vendor headers and the board's
// uClibc gcc (sizeof/offsetof). They only hold on the 32-bit board target.
#[cfg(target_pointer_width = "32")]
const _: () = {
    use std::mem::{offset_of, size_of};
    assert!(size_of::<ViDevAttr>() == 64);
    assert!(size_of::<ViDevBindPipe>() == 136);
    assert!(size_of::<ViChnAttr>() == 136);
    assert!(offset_of!(ViChnAttr, depth) == 32);
    assert!(offset_of!(ViChnAttr, src_frame_rate) == 36);
    assert!(offset_of!(ViChnAttr, alloc_buf_type) == 44);
    assert!(offset_of!(ViChnAttr, isp_opt) == 48);
    assert!(offset_of!(ViIspOpt, memory_type) == 12);
    assert!(size_of::<ViIspOpt>() == 76);
    assert!(size_of::<VideoFrame>() == 80);
    assert!(offset_of!(VideoFrame, field) == 20);
    assert!(offset_of!(VideoFrame, vir_addr) == 44);
    assert!(offset_of!(VideoFrame, time_ref) == 52);
    assert!(offset_of!(VideoFrame, pts_us) == 56);
    assert!(offset_of!(VideoFrame, frame_flag) == 72);
    assert!(size_of::<VencAttr>() == 80);
    assert!(offset_of!(VencAttr, stream_buf_cnt) == 48);
    assert!(offset_of!(VencAttr, profile) == 24);
    assert!(size_of::<VencRcAttr>() == 40);
    assert!(size_of::<VencH264Cbr>() == 28);
    assert!(size_of::<VencGopAttr>() == 16);
    assert!(size_of::<VencChnAttr>() == 136);
    assert!(offset_of!(VencChnAttr, rc) == 80);
    assert!(offset_of!(VencChnAttr, gop) == 120);
    assert!(size_of::<VencPack>() == 136);
    assert!(offset_of!(VencPack, pts_us) == 8);
    assert!(offset_of!(VencPack, frame_end) == 16);
    assert!(offset_of!(VencPack, offset) == 28);
    assert!(offset_of!(VencPack, pack_info) == 36);
    assert!(size_of::<VencStream>() == 400);
    assert!(offset_of!(VencStream, pack_count) == 16);
    assert!(size_of::<VencIntraRefresh>() == 16);
};
