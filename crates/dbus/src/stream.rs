// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Streaming transport for `Subscribe`: a `SOCK_SEQPACKET` pair whose client end
//! is handed back over D-Bus. The worker sends one **framed datagram per event**;
//! `SEQPACKET` preserves message boundaries, so each `sendmsg` is exactly one
//! `recvmsg` on the client — no length-prefix framing needed.
//!
//! Frames are a small JSON header; a `blob` frame additionally carries a memfd as
//! an out-of-band `SCM_RIGHTS` ancillary fd. Sends are **non-blocking**: a slow or
//! stalled subscriber never stalls the inference path — progress frames are dropped
//! and a disconnected peer ends the stream (per the "never await a slow subscriber"
//! rule).

use std::os::fd::{AsRawFd, BorrowedFd, OwnedFd};

use nix::sys::socket::{
    sendmsg, socketpair, AddressFamily, ControlMessage, MsgFlags, SockFlag, SockType,
};
use serde_json::json;

/// The service-side (writer) end of a subscription stream.
pub struct StreamTx {
    sock: OwnedFd,
    /// Events dropped because the client's socket buffer was full (back-pressure).
    pub dropped: u64,
    /// Set once the peer has gone away; further sends are no-ops.
    pub disconnected: bool,
}

/// Create a `SEQPACKET` pair; returns `(service_writer, client_fd)`. The client fd
/// is transferred to the subscriber in the D-Bus reply.
pub fn pair() -> anyhow::Result<(StreamTx, OwnedFd)> {
    let (service, client) = socketpair(
        AddressFamily::Unix,
        SockType::SeqPacket,
        None,
        SockFlag::SOCK_CLOEXEC | SockFlag::SOCK_NONBLOCK,
    )?;
    Ok((StreamTx { sock: service, dropped: 0, disconnected: false }, client))
}

impl StreamTx {
    /// Send one frame (JSON header + optional out-of-band fd). Non-blocking: a full
    /// client buffer drops the frame (`dropped += 1`); a dead peer sets
    /// `disconnected`. Never returns an error for those expected conditions.
    fn send(&mut self, header: &serde_json::Value, fd: Option<BorrowedFd>) {
        if self.disconnected {
            return;
        }
        let bytes = serde_json::to_vec(header).expect("frame header serializes");
        let iov = [std::io::IoSlice::new(&bytes)];
        let raw_fds;
        let cmsg;
        let cmsgs: &[ControlMessage] = if let Some(f) = fd {
            raw_fds = [f.as_raw_fd()];
            cmsg = [ControlMessage::ScmRights(&raw_fds)];
            &cmsg
        } else {
            &[]
        };
        match sendmsg::<()>(self.sock.as_raw_fd(), &iov, cmsgs, MsgFlags::MSG_DONTWAIT, None) {
            Ok(_) => {}
            Err(nix::errno::Errno::EAGAIN) => self.dropped += 1,
            Err(nix::errno::Errno::EPIPE | nix::errno::Errno::ECONNRESET | nix::errno::Errno::ENOTCONN) => {
                self.disconnected = true;
            }
            Err(_) => self.dropped += 1,
        }
    }

    /// A progress update.
    pub fn progress(&mut self, step: u32, total: u32, message: &str) {
        self.send(&json!({"type": "progress", "step": step, "total": total, "message": message}), None);
    }

    /// A streaming-transcription segment: the `text` decoded for window `index` —
    /// for a live `transcribe_stream` session that is the *newly emitted* delta of
    /// one growing transcription (concatenate segments verbatim); for the offline
    /// per-window fallback it is that window's independent transcription.
    /// `is_final` marks the last segment (the input stream reached EOF). Non-blocking
    /// like every frame — a slow subscriber drops segments rather than stalling the
    /// inference path.
    pub fn segment(&mut self, index: u32, text: &str, is_final: bool) {
        self.send(&json!({"type": "segment", "index": index, "text": text, "final": is_final}), None);
    }

    /// An output blob: JSON header + the payload as an out-of-band memfd.
    pub fn blob(&mut self, name: &str, media: &str, meta: &serde_json::Value, fd: BorrowedFd) {
        self.send(&json!({"type": "blob", "name": name, "media": media, "meta": meta}), Some(fd));
    }

    /// Terminal success frame carrying the scalar result JSON.
    pub fn done(&mut self, result: &serde_json::Value) {
        self.send(&json!({"type": "done", "result": result}), None);
    }

    /// Terminal error frame.
    pub fn error(&mut self, message: &str) {
        self.send(&json!({"type": "error", "message": message}), None);
    }
}
