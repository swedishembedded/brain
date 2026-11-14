// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for moving bulk data between a
// supervisor and the worker processes it isolates device code in, for its
// clients. If your team needs expertise in Unix descriptor passing and
// process isolation then you can procure our services by sending an email to
// info@swedishembedded.com.

//! Frames and descriptors on a supervisor's socket, for both of its ends.
//!
//! A frame is one line. A frame may carry open file descriptors beside its
//! bytes (`SCM_RIGHTS`): bulk data such as an image or an audio clip goes in a
//! sealed memfd ([`seal`]), never on the line, so a frame stays small however
//! large the payload. A frame that carries descriptors says how many in its
//! own text (the worker protocol writes `ok+<n>` and `progress+<n>`), and the
//! reader hands them out in the order they arrived ([`Reader::take_fds`]):
//! the kernel delivers a message's descriptors with its first byte, so the
//! descriptors of a frame have always arrived by the time its line is
//! complete, and taking them by count keeps each frame's own.
//!
//! Every received descriptor is close-on-exec, and a message whose
//! descriptors did not fit the control buffer is an error rather than a
//! silent loss ([`MAX_FDS`]).

use std::collections::VecDeque;
use std::io::{ErrorKind, Write};
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::os::unix::net::UnixStream;

/// The most descriptors one received message may carry.
pub const MAX_FDS: usize = 32;

/// The largest blob [`read_sealed`] accepts by default: a few minutes of
/// decoded video frames, far past any single request's image or clip.
pub const MAX_BLOB_BYTES: u64 = 4 << 30;

/// A control buffer for [`MAX_FDS`] descriptors, aligned for `cmsghdr`.
#[repr(C, align(8))]
struct Control([u8; 512]);

const _: () = assert!(std::mem::size_of::<libc::cmsghdr>() + MAX_FDS * std::mem::size_of::<libc::c_int>() <= 512);

#[cfg(target_os = "linux")]
const RECV_FLAGS: libc::c_int = libc::MSG_CMSG_CLOEXEC;
#[cfg(not(target_os = "linux"))]
const RECV_FLAGS: libc::c_int = 0;

#[cfg(target_os = "linux")]
const SEND_FLAGS: libc::c_int = libc::MSG_NOSIGNAL;
#[cfg(not(target_os = "linux"))]
const SEND_FLAGS: libc::c_int = 0;

/// Write `frame` and pass `fds` with its first byte. Honors the socket's
/// write timeout (a full socket past it is [`ErrorKind::WouldBlock`]).
pub fn send(sock: &UnixStream, frame: &[u8], fds: &[BorrowedFd<'_>]) -> std::io::Result<()> {
    if fds.is_empty() {
        return (&mut &*sock).write_all(frame);
    }
    if fds.len() > MAX_FDS {
        return Err(std::io::Error::new(ErrorKind::InvalidInput, format!("a frame carries at most {MAX_FDS} descriptors, not {}", fds.len())));
    }
    if frame.is_empty() {
        return Err(std::io::Error::new(ErrorKind::InvalidInput, "descriptors need at least one byte to travel with"));
    }
    let mut control = Control([0; 512]);
    let data_len = std::mem::size_of_val(fds) as u32;
    let mut iov = libc::iovec { iov_base: frame.as_ptr() as *mut libc::c_void, iov_len: frame.len() };
    // SAFETY: an all-zero msghdr is its documented empty value.
    let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
    msg.msg_iov = &mut iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control.0.as_mut_ptr().cast();
    // SAFETY: CMSG_SPACE is arithmetic on its argument.
    msg.msg_controllen = unsafe { libc::CMSG_SPACE(data_len) } as _;
    // SAFETY: the control buffer is large enough for one header and
    // `fds.len()` ints (checked above), aligned, and owned by this frame.
    unsafe {
        let c = libc::CMSG_FIRSTHDR(&msg);
        (*c).cmsg_level = libc::SOL_SOCKET;
        (*c).cmsg_type = libc::SCM_RIGHTS;
        (*c).cmsg_len = libc::CMSG_LEN(data_len) as _;
        let data = libc::CMSG_DATA(c).cast::<libc::c_int>();
        for (i, fd) in fds.iter().enumerate() {
            std::ptr::write_unaligned(data.add(i), fd.as_raw_fd());
        }
    }
    let sent = loop {
        // SAFETY: `msg` points at live buffers for the whole call.
        let n = unsafe { libc::sendmsg(sock.as_raw_fd(), &msg, SEND_FLAGS) };
        if n >= 0 {
            break n as usize;
        }
        let e = std::io::Error::last_os_error();
        if e.kind() != ErrorKind::Interrupted {
            return Err(e);
        }
    };
    (&mut &*sock).write_all(&frame[sent..])
}

/// The reading end: bytes and descriptors as they arrive, cut into lines.
pub struct Reader {
    buf: Vec<u8>,
    fds: VecDeque<OwnedFd>,
    max_line: usize,
}

impl Reader {
    /// A reader that refuses any line longer than `max_line` bytes.
    pub fn new(max_line: usize) -> Reader {
        Reader { buf: Vec::new(), fds: VecDeque::new(), max_line }
    }

    /// One `recvmsg`: `Ok(0)` at end of stream. Honors the socket's read
    /// timeout ([`ErrorKind::WouldBlock`] or [`ErrorKind::TimedOut`] when it
    /// passes); an interrupted call is retried.
    pub fn fill(&mut self, sock: &UnixStream) -> std::io::Result<usize> {
        let mut chunk = [0u8; 64 * 1024];
        let mut control = Control([0; 512]);
        let mut iov = libc::iovec { iov_base: chunk.as_mut_ptr().cast(), iov_len: chunk.len() };
        // SAFETY: an all-zero msghdr is its documented empty value.
        let mut msg: libc::msghdr = unsafe { std::mem::zeroed() };
        msg.msg_iov = &mut iov;
        msg.msg_iovlen = 1;
        msg.msg_control = control.0.as_mut_ptr().cast();
        msg.msg_controllen = control.0.len() as _;
        let n = loop {
            // SAFETY: `msg` points at live buffers for the whole call.
            let n = unsafe { libc::recvmsg(sock.as_raw_fd(), &mut msg, RECV_FLAGS) };
            if n >= 0 {
                break n as usize;
            }
            let e = std::io::Error::last_os_error();
            if e.kind() != ErrorKind::Interrupted {
                return Err(e);
            }
        };
        // Take ownership of every descriptor first, so even a refused message
        // closes what it brought.
        // SAFETY: the kernel filled `control` up to `msg_controllen`; the
        // CMSG_* walk stays within it.
        unsafe {
            let mut c = libc::CMSG_FIRSTHDR(&msg);
            while !c.is_null() {
                if (*c).cmsg_level == libc::SOL_SOCKET && (*c).cmsg_type == libc::SCM_RIGHTS {
                    let data = libc::CMSG_DATA(c).cast::<libc::c_int>();
                    let bytes = (*c).cmsg_len as usize - (data as usize - c as usize);
                    for i in 0..bytes / std::mem::size_of::<libc::c_int>() {
                        let fd = std::ptr::read_unaligned(data.add(i));
                        self.fds.push_back(OwnedFd::from_raw_fd(fd));
                        #[cfg(not(target_os = "linux"))]
                        libc::fcntl(fd, libc::F_SETFD, libc::FD_CLOEXEC);
                    }
                }
                c = libc::CMSG_NXTHDR(&msg, c);
            }
        }
        if msg.msg_flags & libc::MSG_CTRUNC != 0 {
            return Err(std::io::Error::new(ErrorKind::InvalidData, format!("a message carried more than {MAX_FDS} descriptors")));
        }
        self.buf.extend_from_slice(&chunk[..n]);
        Ok(n)
    }

    /// The next complete line without its newline, `Ok(None)` when there is
    /// none yet. A line that is not UTF-8 or grows past the bound is an error.
    pub fn line(&mut self) -> Result<Option<String>, String> {
        match self.buf.iter().position(|&b| b == b'\n') {
            Some(end) if end > self.max_line => Err(format!("a line longer than {} bytes", self.max_line)),
            Some(end) => {
                let line: Vec<u8> = self.buf.drain(..=end).collect();
                String::from_utf8(line[..end].to_vec()).map(Some).map_err(|_| "a line that is not UTF-8".to_string())
            }
            None if self.buf.len() > self.max_line => Err(format!("a line longer than {} bytes", self.max_line)),
            None => Ok(None),
        }
    }

    /// The next `n` descriptors received, in arrival order: the ones the
    /// frame just read declared. Fewer than `n` is an error.
    pub fn take_fds(&mut self, n: usize) -> Result<Vec<OwnedFd>, String> {
        if self.fds.len() < n {
            return Err(format!("a frame declared {n} descriptor(s) and {} arrived", self.fds.len()));
        }
        Ok(self.fds.drain(..n).collect())
    }

    /// Descriptors received that no frame has claimed yet.
    pub fn unclaimed_fds(&self) -> usize {
        self.fds.len()
    }
}

/// `bytes` in a new memfd sealed against writing, growing and shrinking, so
/// the receiver can trust its size and that it will not change. Linux only.
#[cfg(target_os = "linux")]
pub fn seal(name: &str, bytes: &[u8]) -> std::io::Result<OwnedFd> {
    let cname = std::ffi::CString::new(name.replace('\0', "")).unwrap_or_default();
    // SAFETY: memfd_create takes a NUL-terminated name and flags.
    let raw = unsafe { libc::memfd_create(cname.as_ptr(), libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING) };
    if raw < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: memfd_create returned a fresh descriptor this function owns.
    let fd = unsafe { OwnedFd::from_raw_fd(raw) };
    let mut file = std::fs::File::from(fd);
    file.write_all(bytes)?;
    let fd = OwnedFd::from(file);
    // SAFETY: fcntl on a descriptor this function owns.
    if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_ADD_SEALS, libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE | libc::F_SEAL_SEAL) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(fd)
}

#[cfg(not(target_os = "linux"))]
pub fn seal(_name: &str, _bytes: &[u8]) -> std::io::Result<OwnedFd> {
    Err(std::io::Error::new(ErrorKind::Unsupported, "sealed memfds need Linux"))
}

/// The whole contents of a sealed memfd from the other end, at most `max`
/// bytes. Refuses a descriptor that is not sealed against writing and
/// resizing (its sender could change it while it is read) and one larger than
/// `max`, checked before anything is allocated. Reads with an explicit
/// offset, so where the sender left the shared file position does not matter.
#[cfg(target_os = "linux")]
pub fn read_sealed(fd: &OwnedFd, max: u64) -> std::io::Result<Vec<u8>> {
    let want = libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_WRITE;
    // SAFETY: fcntl on a descriptor the caller owns.
    let seals = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GET_SEALS) };
    if seals < 0 || seals & want != want {
        return Err(std::io::Error::new(ErrorKind::InvalidData, "a blob descriptor that is not a sealed memfd"));
    }
    let file = std::fs::File::from(fd.try_clone()?);
    let len = file.metadata()?.len();
    if len > max {
        return Err(std::io::Error::new(ErrorKind::InvalidData, format!("a blob of {len} bytes is over the {max} byte bound")));
    }
    let mut out = vec![0u8; len as usize];
    std::os::unix::fs::FileExt::read_exact_at(&file, &mut out, 0)?;
    Ok(out)
}

#[cfg(not(target_os = "linux"))]
pub fn read_sealed(_fd: &OwnedFd, _max: u64) -> std::io::Result<Vec<u8>> {
    Err(std::io::Error::new(ErrorKind::Unsupported, "sealed memfds need Linux"))
}
