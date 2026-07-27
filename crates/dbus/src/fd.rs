// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! File-descriptor transport: move bulk data (images, results) by fd instead of
//! marshalling bytes through D-Bus.
//!
//! - Results/inputs use a **sealed memfd** (portable, zero-copy host-side). A
//!   client mmaps it read-only; the seals let the receiver trust the size and that
//!   the contents will not change under it.
//! - Reads are **mmap-based and offset-independent**, so it does not matter where a
//!   sender left the fd's file offset (a real gotcha: an fd passed via `SCM_RIGHTS`
//!   shares its open-file offset with the sender). Works for memfd, files, and
//!   dmabuf (which is mmap-able); a dmabuf export path lives in
//!   [`crate::dmabuf`] and falls back here.

use std::ffi::CString;
use std::num::NonZeroUsize;
use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};

use nix::fcntl::{fcntl, FcntlArg, SealFlag};
use nix::sys::memfd::{memfd_create, MemFdCreateFlag};
use nix::sys::mman::{mmap, munmap, MapFlags, ProtFlags};
use nix::sys::stat::fstat;

/// Create a sealed, read-only-once memfd holding `bytes`; the returned [`OwnedFd`]
/// owns the memory (dropped ⇒ freed). Sealed `SHRINK|GROW|WRITE` so a receiver can
/// trust `st_size` and immutability.
pub fn memfd_seal(name: &str, bytes: &[u8]) -> anyhow::Result<OwnedFd> {
    let cname = CString::new(name).unwrap_or_else(|_| CString::new("brain").unwrap());
    let fd = memfd_create(&cname, MemFdCreateFlag::MFD_CLOEXEC | MemFdCreateFlag::MFD_ALLOW_SEALING)?;
    let mut off = 0usize;
    while off < bytes.len() {
        let n = nix::unistd::write(&fd, &bytes[off..])?;
        if n == 0 {
            anyhow::bail!("short write to memfd ({off}/{} bytes)", bytes.len());
        }
        off += n;
    }
    fcntl(fd.as_raw_fd(), FcntlArg::F_ADD_SEALS(SealFlag::F_SEAL_SHRINK | SealFlag::F_SEAL_GROW | SealFlag::F_SEAL_WRITE))?;
    Ok(fd)
}

/// Read the whole contents of `fd` into a `Vec` by mmap (offset-independent). Empty
/// fds return an empty vec. The caller keeps ownership of `fd`.
pub fn read_fd_to_vec(fd: BorrowedFd) -> anyhow::Result<Vec<u8>> {
    let st = fstat(fd.as_raw_fd())?;
    let len = st.st_size as usize;
    if len == 0 {
        return Ok(Vec::new());
    }
    let nz = NonZeroUsize::new(len).unwrap();
    // SAFETY: fresh read-only shared mapping of a valid fd; unmapped before return.
    let ptr = unsafe { mmap(None, nz, ProtFlags::PROT_READ, MapFlags::MAP_SHARED, fd, 0)? };
    let out = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u8, len) }.to_vec();
    // SAFETY: `ptr`/`len` are exactly what mmap returned above.
    unsafe { munmap(ptr, len)? };
    Ok(out)
}

/// Convenience for `AsFd` sources (OwnedFd, BorrowedFd, sockets…).
pub fn read_owned_to_vec(fd: &impl AsFd) -> anyhow::Result<Vec<u8>> {
    read_fd_to_vec(fd.as_fd())
}

/// Package `bytes` as an fd for return over D-Bus. `want_dmabuf` first tries a real
/// dmabuf from the kernel DMA-heap (zero-copy-importable by GPU consumers); if no
/// heap is available (common in containers/older kernels) it transparently falls
/// back to a sealed memfd. Returns `(fd, transport)` where `transport` is
/// `"dmabuf"` or `"memfd"` so the caller can report what was actually provided.
pub fn bytes_to_fd(name: &str, bytes: &[u8], want_dmabuf: bool) -> anyhow::Result<(OwnedFd, &'static str)> {
    if want_dmabuf {
        if let Ok(fd) = dmabuf_from_heap(bytes) {
            return Ok((fd, "dmabuf"));
        }
    }
    Ok((memfd_seal(name, bytes)?, "memfd"))
}

// ---- dmabuf via /dev/dma_heap (best-effort) ----

#[repr(C)]
struct DmaHeapAllocationData {
    len: u64,
    fd: u32,
    fd_flags: u32,
    heap_flags: u64,
}
// DMA_HEAP_IOCTL_ALLOC = _IOWR('H', 0x0, struct dma_heap_allocation_data)
nix::ioctl_readwrite!(dma_heap_alloc, b'H', 0x0, DmaHeapAllocationData);

/// Allocate a dmabuf from the system DMA-heap, fill it with `bytes`, return the fd.
/// Errors (propagated so the caller falls back to memfd) if the heap node is absent
/// or inaccessible.
fn dmabuf_from_heap(bytes: &[u8]) -> anyhow::Result<OwnedFd> {
    use nix::fcntl::{open, OFlag};
    use nix::sys::stat::Mode;
    let heap = open("/dev/dma_heap/system", OFlag::O_RDWR | OFlag::O_CLOEXEC, Mode::empty())?;
    let heap = unsafe { OwnedFd::from_raw_fd(heap) };
    let mut req = DmaHeapAllocationData {
        len: bytes.len().max(1) as u64,
        fd: 0,
        fd_flags: (OFlag::O_RDWR | OFlag::O_CLOEXEC).bits() as u32,
        heap_flags: 0,
    };
    // SAFETY: valid heap fd + a correctly-shaped request struct for this ioctl.
    unsafe { dma_heap_alloc(heap.as_raw_fd(), &mut req)? };
    let buf = unsafe { OwnedFd::from_raw_fd(req.fd as i32) };
    if !bytes.is_empty() {
        // mmap writable, copy the payload in, unmap.
        let nz = NonZeroUsize::new(bytes.len()).unwrap();
        let ptr = unsafe {
            mmap(None, nz, ProtFlags::PROT_READ | ProtFlags::PROT_WRITE, MapFlags::MAP_SHARED, buf.as_fd(), 0)?
        };
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr() as *mut u8, bytes.len()) };
        unsafe { munmap(ptr, bytes.len())? };
    }
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memfd_roundtrips_regardless_of_offset() {
        let data = b"brain-dbus fd transport \x00\x01\x02 payload".to_vec();
        let fd = memfd_seal("brain-test", &data).expect("memfd");
        // Read back via mmap — must match even though the write left the offset at EOF.
        let got = read_fd_to_vec(fd.as_fd()).expect("read");
        assert_eq!(got, data);
        // A second read is identical (mmap is offset-independent).
        assert_eq!(read_fd_to_vec(fd.as_fd()).expect("read2"), data);
    }

    #[test]
    fn empty_payload_ok() {
        let fd = memfd_seal("empty", &[]).expect("memfd");
        assert!(read_fd_to_vec(fd.as_fd()).expect("read").is_empty());
    }
}
