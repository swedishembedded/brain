// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Minimal V4L2 mmap-streaming capture over a hand-rolled ioctl FFI.
//!
//! No `v4l`/`nokhwa` crate: those pull `bindgen` -> libclang at build time, which
//! would break brain's self-contained `cargo build`. The V4L2 surface is 6 libc
//! symbols + 8 ioctls — smaller than the SDL surface already hand-rolled in
//! `wm-display`.
//!
//! ## ABI discipline
//!
//! Every ioctl request number, struct size and field offset here was PRINTED by a C
//! program against `<linux/videodev2.h>` (see `tests/abi.rs`, which re-derives them
//! when a compiler is present and fails on any drift) — never hand-computed. The
//! structs are opaque byte buffers of the exact size, written at the derived
//! offsets, exactly as `wm-display/src/sys.rs` treats `SDL_Event`. `_IOWR` encodes
//! `sizeof` into the request number, so a size mismatch fails the FIRST ioctl with
//! `ENOTTY` — loud, never silent corruption.
//!
//! Forces `V4L2_PIX_FMT_YUYV`: MJPEG would need a JPEG decoder brain does not have.

#![allow(non_upper_case_globals)]

use std::os::raw::{c_int, c_ulong, c_void};

// ---- ioctl request numbers (from tests/abi.rs / videodev2.h) ----
pub const VIDIOC_QUERYCAP: c_ulong = 0x8068_5600;
pub const VIDIOC_S_FMT: c_ulong = 0xc0d0_5605;
pub const VIDIOC_REQBUFS: c_ulong = 0xc014_5608;
pub const VIDIOC_QUERYBUF: c_ulong = 0xc058_5609;
pub const VIDIOC_QBUF: c_ulong = 0xc058_560f;
pub const VIDIOC_DQBUF: c_ulong = 0xc058_5611;
pub const VIDIOC_STREAMON: c_ulong = 0x4004_5612;
pub const VIDIOC_STREAMOFF: c_ulong = 0x4004_5613;

pub const V4L2_PIX_FMT_YUYV: u32 = 0x5659_5559;
pub const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
pub const V4L2_MEMORY_MMAP: u32 = 1;
pub const V4L2_FIELD_NONE: u32 = 1;
pub const V4L2_CAP_VIDEO_CAPTURE: u32 = 0x0000_0001;
pub const V4L2_CAP_STREAMING: u32 = 0x0400_0000;

// ---- struct sizes ----
pub const SZ_CAPABILITY: usize = 104;
pub const SZ_FORMAT: usize = 208;
pub const SZ_REQUESTBUFFERS: usize = 20;
pub const SZ_BUFFER: usize = 88;

// ---- field offsets ----
const OFF_CAP_DEVICE_CAPS: usize = 88;
const OFF_FMT_TYPE: usize = 0;
const OFF_FMT_PIX: usize = 8; // v4l2_pix_format begins here
const OFF_PIX_WIDTH: usize = 0;
const OFF_PIX_HEIGHT: usize = 4;
const OFF_PIX_PIXFMT: usize = 8;
const OFF_PIX_FIELD: usize = 12;
#[allow(dead_code)]
const OFF_PIX_SIZEIMAGE: usize = 20;
const OFF_REQ_COUNT: usize = 0;
const OFF_REQ_TYPE: usize = 4;
const OFF_REQ_MEMORY: usize = 8;
const OFF_BUF_INDEX: usize = 0;
const OFF_BUF_TYPE: usize = 4;
const OFF_BUF_BYTESUSED: usize = 8;
const OFF_BUF_MEMORY: usize = 60;
const OFF_BUF_M_OFFSET: usize = 64;
const OFF_BUF_LENGTH: usize = 72;

const O_RDWR: c_int = 2;
const PROT_READ: c_int = 1;
const PROT_WRITE: c_int = 2;
const MAP_SHARED: c_int = 1;
const MAP_FAILED: isize = -1;

extern "C" {
    fn open(path: *const u8, flags: c_int) -> c_int;
    fn close(fd: c_int) -> c_int;
    fn ioctl(fd: c_int, req: c_ulong, arg: *mut c_void) -> c_int;
    fn mmap(addr: *mut c_void, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> c_int;
    #[allow(dead_code)]
    fn __errno_location() -> *mut c_int;
}

fn errno() -> i32 {
    unsafe { *__errno_location() }
}

fn wr_u32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_ne_bytes());
}
fn rd_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_ne_bytes(buf[off..off + 4].try_into().unwrap())
}

/// One mmap'd capture buffer.
struct MappedBuf {
    ptr: *mut c_void,
    len: usize,
}

/// A streaming YUYV capture device.
pub struct Device {
    fd: c_int,
    pub width: u32,
    pub height: u32,
    bufs: Vec<MappedBuf>,
    /// Reused DQBUF/QBUF struct (88 bytes) to avoid per-frame allocation.
    buf: [u8; SZ_BUFFER],
}

// SAFETY: Device exclusively OWNS its fd and its mmap'd buffers, and is moved to
// (never shared across) the capture thread. The raw pointers are only dereferenced
// by the owning thread, so transferring ownership between threads is sound. It is
// deliberately NOT Sync — two threads must not touch it at once.
unsafe impl Send for Device {}

impl Device {
    /// Open `path` (e.g. `/dev/video0`), negotiate YUYV at `width x height`, map
    /// `nbuf` buffers, and start streaming. The driver may return a nearby
    /// resolution; the accepted one is in [`Device::width`]/`height`.
    pub fn open(path: &str, width: u32, height: u32, nbuf: u32) -> Result<Device, String> {
        let mut cpath: Vec<u8> = path.bytes().collect();
        cpath.push(0);
        let fd = unsafe { open(cpath.as_ptr(), O_RDWR) };
        if fd < 0 {
            return Err(format!("open {path}: errno {}", errno()));
        }
        let mut dev = Device { fd, width, height, bufs: Vec::new(), buf: [0u8; SZ_BUFFER] };
        dev.init(width, height, nbuf)?;
        Ok(dev)
    }

    fn xioctl(&self, req: c_ulong, arg: *mut c_void) -> Result<(), String> {
        // V4L2 ioctls can return EINTR; retry a bounded number of times.
        for _ in 0..8 {
            let rc = unsafe { ioctl(self.fd, req, arg) };
            if rc == 0 {
                return Ok(());
            }
            let e = errno();
            if e != 4 {
                return Err(format!("ioctl 0x{req:08x}: errno {e}"));
            }
        }
        Err(format!("ioctl 0x{req:08x}: EINTR storm"))
    }

    fn init(&mut self, width: u32, height: u32, nbuf: u32) -> Result<(), String> {
        // 1. QUERYCAP — must be a streaming capture device.
        let mut cap = [0u8; SZ_CAPABILITY];
        self.xioctl(VIDIOC_QUERYCAP, cap.as_mut_ptr() as *mut c_void)?;
        let caps = rd_u32(&cap, OFF_CAP_DEVICE_CAPS);
        if caps & V4L2_CAP_VIDEO_CAPTURE == 0 {
            return Err("device is not a video-capture device".into());
        }
        if caps & V4L2_CAP_STREAMING == 0 {
            return Err("device does not support streaming I/O".into());
        }

        // 2. S_FMT — force YUYV.
        let mut fmt = [0u8; SZ_FORMAT];
        wr_u32(&mut fmt, OFF_FMT_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE);
        wr_u32(&mut fmt, OFF_FMT_PIX + OFF_PIX_WIDTH, width);
        wr_u32(&mut fmt, OFF_FMT_PIX + OFF_PIX_HEIGHT, height);
        wr_u32(&mut fmt, OFF_FMT_PIX + OFF_PIX_PIXFMT, V4L2_PIX_FMT_YUYV);
        wr_u32(&mut fmt, OFF_FMT_PIX + OFF_PIX_FIELD, V4L2_FIELD_NONE);
        self.xioctl(VIDIOC_S_FMT, fmt.as_mut_ptr() as *mut c_void)?;
        // The driver may have adjusted the geometry and format.
        if rd_u32(&fmt, OFF_FMT_PIX + OFF_PIX_PIXFMT) != V4L2_PIX_FMT_YUYV {
            return Err("device would not accept YUYV (likely MJPEG-only; brain has no JPEG decoder)".into());
        }
        self.width = rd_u32(&fmt, OFF_FMT_PIX + OFF_PIX_WIDTH);
        self.height = rd_u32(&fmt, OFF_FMT_PIX + OFF_PIX_HEIGHT);

        // 3. REQBUFS — mmap buffers.
        let mut req = [0u8; SZ_REQUESTBUFFERS];
        wr_u32(&mut req, OFF_REQ_COUNT, nbuf);
        wr_u32(&mut req, OFF_REQ_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE);
        wr_u32(&mut req, OFF_REQ_MEMORY, V4L2_MEMORY_MMAP);
        self.xioctl(VIDIOC_REQBUFS, req.as_mut_ptr() as *mut c_void)?;
        let got = rd_u32(&req, OFF_REQ_COUNT);
        if got < 2 {
            return Err(format!("driver granted only {got} buffers (need >= 2)"));
        }

        // 4. QUERYBUF + mmap each, then QBUF it.
        for i in 0..got {
            let mut b = [0u8; SZ_BUFFER];
            wr_u32(&mut b, OFF_BUF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE);
            wr_u32(&mut b, OFF_BUF_MEMORY, V4L2_MEMORY_MMAP);
            wr_u32(&mut b, OFF_BUF_INDEX, i);
            self.xioctl(VIDIOC_QUERYBUF, b.as_mut_ptr() as *mut c_void)?;
            let len = rd_u32(&b, OFF_BUF_LENGTH) as usize;
            let off = rd_u32(&b, OFF_BUF_M_OFFSET) as i64;
            let ptr = unsafe { mmap(std::ptr::null_mut(), len, PROT_READ | PROT_WRITE, MAP_SHARED, self.fd, off) };
            if ptr as isize == MAP_FAILED {
                return Err(format!("mmap buffer {i}: errno {}", errno()));
            }
            self.bufs.push(MappedBuf { ptr, len });
            self.queue(i)?;
        }

        // 5. STREAMON.
        let mut ty = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        self.xioctl(VIDIOC_STREAMON, &mut ty as *mut u32 as *mut c_void)?;
        Ok(())
    }

    fn queue(&self, index: u32) -> Result<(), String> {
        let mut b = [0u8; SZ_BUFFER];
        wr_u32(&mut b, OFF_BUF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE);
        wr_u32(&mut b, OFF_BUF_MEMORY, V4L2_MEMORY_MMAP);
        wr_u32(&mut b, OFF_BUF_INDEX, index);
        self.xioctl(VIDIOC_QBUF, b.as_mut_ptr() as *mut c_void)
    }

    /// Block until the next frame, run `f` over its YUYV bytes, then requeue the
    /// buffer. `f`'s result is returned. Copies nothing itself — the closure decides
    /// (the demo converts straight into an RGB frame).
    pub fn next_frame<T>(&mut self, f: impl FnOnce(&[u8], u32, u32) -> T) -> Result<T, String> {
        self.buf = [0u8; SZ_BUFFER];
        wr_u32(&mut self.buf, OFF_BUF_TYPE, V4L2_BUF_TYPE_VIDEO_CAPTURE);
        wr_u32(&mut self.buf, OFF_BUF_MEMORY, V4L2_MEMORY_MMAP);
        let buf_ptr = self.buf.as_mut_ptr() as *mut c_void;
        self.xioctl(VIDIOC_DQBUF, buf_ptr)?;
        let index = rd_u32(&self.buf, OFF_BUF_INDEX);
        let bytesused = rd_u32(&self.buf, OFF_BUF_BYTESUSED) as usize;
        let mb = &self.bufs[index as usize];
        let n = bytesused.min(mb.len);
        let slice = unsafe { std::slice::from_raw_parts(mb.ptr as *const u8, n) };
        let out = f(slice, self.width, self.height);
        self.queue(index)?;
        Ok(out)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let mut ty = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        unsafe {
            ioctl(self.fd, VIDIOC_STREAMOFF, &mut ty as *mut u32 as *mut c_void);
            for b in &self.bufs {
                munmap(b.ptr, b.len);
            }
            close(self.fd);
        }
    }
}
