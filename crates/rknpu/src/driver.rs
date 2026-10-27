// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for running neural networks on the
// NPUs of low-cost embedded Linux SoCs for its clients. If your team needs
// expertise in embedded NPU bring-up then you can procure our services by
// sending an email to info@swedishembedded.com.

//! The `/dev/rknpu` character device of Rockchip's vendor NPU driver: buffer
//! allocation, mapping, cache maintenance and job submission.
//!
//! The ioctl ABI (structure layouts, request numbers, job and memory flags) is
//! the one documented by open-rknpu's libc runtime (MIT, see `/NOTICE.md`),
//! which reproduces it from the driver's public interface. Every structure is
//! `repr(C)` with its size pinned by a compile-time assertion, so a layout slip
//! on a 32-bit target fails the build instead of corrupting a job.

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Arc;

use crate::{Error, Result};

const DEVICE_PATH: &str = "/dev/rknpu";

#[repr(C)]
#[derive(Default)]
struct MemCreate {
    handle: u32,
    flags: u32,
    size: u64,
    obj_addr: u64,
    dma_addr: u64,
    sram_size: u64,
}

#[repr(C)]
struct MemDestroy {
    handle: u32,
    reserved: u32,
    obj_addr: u64,
}

#[repr(C)]
struct MemSync {
    flags: u32,
    reserved: u32,
    obj_addr: u64,
    offset: u64,
    size: u64,
}

/// One entry of a job's task table, read by the driver from a kernel-mapped
/// buffer: which engines the task enables, which interrupt ends it, and where
/// its register command program lives.
#[repr(C, packed)]
#[derive(Clone, Copy, Default)]
pub struct TaskEntry {
    pub flags: u32,
    pub op_idx: u32,
    pub enable_mask: u32,
    pub int_mask: u32,
    pub int_clear: u32,
    pub int_status: u32,
    pub regcfg_amount: u32,
    pub regcfg_offset: u32,
    pub regcmd_addr: u64,
}

#[repr(C)]
struct Submit {
    flags: u32,
    timeout: u32,
    task_start: u32,
    task_number: u32,
    task_counter: u32,
    priority: i32,
    task_obj_addr: u64,
    regcfg_obj_addr: u64,
    task_base_addr: u64,
    user_data: u64,
    core_mask: u32,
    fence_fd: i32,
    subcore_task: [u32; 10],
}

const _: () = assert!(std::mem::size_of::<MemCreate>() == 40);
const _: () = assert!(std::mem::size_of::<MemDestroy>() == 16);
const _: () = assert!(std::mem::size_of::<MemSync>() == 32);
const _: () = assert!(std::mem::size_of::<TaskEntry>() == 40);
const _: () = assert!(std::mem::size_of::<Submit>() == 104);

/// `_IOWR('r', nr, T)` in the generic Linux ioctl encoding (ARM uses it).
const fn iowr(nr: u32, size: usize) -> u32 {
    (3 << 30) | ((size as u32) << 16) | ((b'r' as u32) << 8) | nr
}

const IOCTL_SUBMIT: u32 = iowr(1, std::mem::size_of::<Submit>());
const IOCTL_MEM_CREATE: u32 = iowr(2, std::mem::size_of::<MemCreate>());
const IOCTL_MEM_DESTROY: u32 = iowr(4, std::mem::size_of::<MemDestroy>());
const IOCTL_MEM_SYNC: u32 = iowr(5, std::mem::size_of::<MemSync>());

/// Memory flags: a cacheable buffer (needs [`Buffer::sync_to_device`] and
/// [`Buffer::sync_from_device`] around device access), optionally mapped into
/// the kernel too (the task table must be, since the driver reads it).
const MEM_CACHEABLE: u32 = 1 << 1;
const MEM_KERNEL_MAPPING: u32 = 1 << 3;

const SYNC_TO_DEVICE: u32 = 1;
const SYNC_FROM_DEVICE: u32 = 2;

/// Job flags: program-counter mode, and ping-pong fetch of the next task's
/// commands. Ping-pong is only valid while every task's program fits the
/// fetch window, which [`Device::submit`] checks.
const JOB_PC: u32 = 1;
const JOB_PINGPONG: u32 = 1 << 2;
const PINGPONG_MAX_AMOUNT: u32 = 256;

/// How long the driver waits for one job before resetting the NPU.
const JOB_TIMEOUT_MS: u32 = 1000;

fn os_error(op: &'static str) -> Error {
    Error::Driver { op, source: io::Error::last_os_error() }
}

/// An open handle on the NPU.
pub struct Device {
    file: Arc<File>,
}

impl Device {
    /// Open `/dev/rknpu`. Fails with the OS error when the driver is not loaded
    /// or the caller may not use it.
    pub fn open() -> Result<Device> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_CLOEXEC)
            .open(DEVICE_PATH)
            .map_err(|source| Error::Driver { op: "open /dev/rknpu", source })?;
        Ok(Device { file: Arc::new(file) })
    }

    /// Allocate `size` bytes of physically contiguous, NPU-addressable memory
    /// and map it into this process.
    pub fn alloc(&self, size: usize, kernel_mapped: bool) -> Result<Buffer> {
        let size = size.max(1).next_multiple_of(page_size());
        let flags = MEM_CACHEABLE | if kernel_mapped { MEM_KERNEL_MAPPING } else { 0 };
        let mut req = MemCreate { flags, size: size as u64, ..Default::default() };
        // SAFETY: `req` is a live, correctly sized repr(C) struct for this request.
        if unsafe { libc::ioctl(self.fd(), IOCTL_MEM_CREATE as _, &mut req) } < 0 {
            return Err(os_error("RKNPU_MEM_CREATE"));
        }
        // The driver returns the buffer as a dma-buf file descriptor: own it
        // first so every later failure path closes it.
        // SAFETY: on success `handle` is a fresh descriptor this process owns.
        let dmabuf = unsafe { OwnedFd::from_raw_fd(req.handle as RawFd) };
        let destroy = |dev: &Device| {
            let mut d = MemDestroy { handle: req.handle, reserved: 0, obj_addr: req.obj_addr };
            // SAFETY: as above; releasing the object we just created.
            unsafe { libc::ioctl(dev.fd(), IOCTL_MEM_DESTROY as _, &mut d) };
        };
        if req.dma_addr > u64::from(u32::MAX) {
            destroy(self);
            return Err(Error::Unsupported(format!("buffer at dma {:#x} is outside the NPU's 32-bit address space", req.dma_addr)));
        }
        // SAFETY: mapping `size` bytes of the dma-buf we own, shared, read/write.
        let ptr = unsafe { libc::mmap(std::ptr::null_mut(), size, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_SHARED, dmabuf.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            let err = os_error("mmap NPU buffer");
            destroy(self);
            return Err(err);
        }
        Ok(Buffer {
            device: Arc::clone(&self.file),
            dmabuf,
            handle: req.handle,
            obj_addr: req.obj_addr,
            dma_addr: req.dma_addr as u32,
            ptr: ptr.cast(),
            size,
        })
    }

    /// Run `count` tasks starting at `start` from `tasks` (a kernel-mapped
    /// buffer of [`TaskEntry`]) as one job, blocking until it completes.
    /// `base` is the DMA address every address in the tasks' register programs
    /// is relative to; `max_amount` is the largest `regcfg_amount` among them.
    pub fn submit(&self, tasks: &Buffer, start: u32, count: u32, base: u32, max_amount: u32) -> Result<()> {
        let flags = if max_amount > PINGPONG_MAX_AMOUNT { JOB_PC } else { JOB_PC | JOB_PINGPONG };
        let mut req = Submit {
            flags,
            timeout: JOB_TIMEOUT_MS,
            task_start: start,
            task_number: count,
            task_counter: 0,
            priority: 0,
            task_obj_addr: tasks.obj_addr,
            regcfg_obj_addr: 0,
            task_base_addr: u64::from(base),
            user_data: 0,
            core_mask: 0,
            fence_fd: 0,
            subcore_task: [0; 10],
        };
        // SAFETY: `req` is a live, correctly sized repr(C) struct for this request.
        if unsafe { libc::ioctl(self.fd(), IOCTL_SUBMIT as _, &mut req) } < 0 {
            return Err(os_error("RKNPU_SUBMIT"));
        }
        Ok(())
    }

    fn fd(&self) -> RawFd {
        self.file.as_raw_fd()
    }
}

/// A contiguous NPU buffer mapped into this process, unmapped and released
/// on drop. It keeps the device open for as long as it lives.
pub struct Buffer {
    device: Arc<File>,
    dmabuf: OwnedFd,
    handle: u32,
    obj_addr: u64,
    dma_addr: u32,
    ptr: *mut u8,
    size: usize,
}

// SAFETY: the mapping is plain memory owned by this value; access goes through
// `&self`/`&mut self` like any owned byte buffer.
unsafe impl Send for Buffer {}

impl Buffer {
    /// The address the NPU sees.
    pub fn dma_addr(&self) -> u32 {
        self.dma_addr
    }

    /// The buffer as a dma-buf, for zero-copy producers (RGA, V4L2).
    pub fn dmabuf_fd(&self) -> RawFd {
        self.dmabuf.as_raw_fd()
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` maps `size` bytes for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Write back CPU caches for `range` before the NPU reads it.
    pub fn sync_to_device(&self, range: std::ops::Range<usize>) -> Result<()> {
        self.sync(SYNC_TO_DEVICE, range)
    }

    /// Invalidate CPU caches for `range` after the NPU wrote it.
    pub fn sync_from_device(&self, range: std::ops::Range<usize>) -> Result<()> {
        self.sync(SYNC_FROM_DEVICE, range)
    }

    fn sync(&self, flags: u32, range: std::ops::Range<usize>) -> Result<()> {
        if range.end > self.size || range.start > range.end {
            return Err(Error::Unsupported(format!("sync range {range:?} outside a {}-byte buffer", self.size)));
        }
        let mut req = MemSync { flags, reserved: 0, obj_addr: self.obj_addr, offset: range.start as u64, size: (range.end - range.start) as u64 };
        // SAFETY: `req` is a live, correctly sized repr(C) struct for this request.
        if unsafe { libc::ioctl(self.device.as_raw_fd(), IOCTL_MEM_SYNC as _, &mut req) } < 0 {
            return Err(os_error("RKNPU_MEM_SYNC"));
        }
        Ok(())
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: unmapping our own mapping and releasing our own object; the
        // dma-buf descriptor closes when `dmabuf` drops after this.
        unsafe {
            libc::munmap(self.ptr.cast(), self.size);
            let mut d = MemDestroy { handle: self.handle, reserved: 0, obj_addr: self.obj_addr };
            libc::ioctl(self.device.as_raw_fd(), IOCTL_MEM_DESTROY as _, &mut d);
        }
    }
}

fn page_size() -> usize {
    // SAFETY: sysconf has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page > 0 {
        page as usize
    } else {
        4096
    }
}
