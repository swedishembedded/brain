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

    /// A progress update. `phase` is `None` for a running job's normal
    /// progress (today's wire shape, unchanged); `Some("fetching")` for a
    /// cold auto-fetch tick ahead of the real job (`step`/`total` are then
    /// byte counts, not steps) -- an additive JSON key, so an older client
    /// that doesn't look for `phase` sees exactly the frame it always has.
    /// `delta` is a per-token text fragment (`capability::Progress::delta`) and
    /// `event` a structured out-of-band payload (reasoning/tool-call chunks,
    /// `capability::Progress::event`) — both `None` for a plain step tick. A
    /// `Subscribe`r that drops these gets a `done` frame with the final blob but
    /// NO token stream in between; forwarding them is what makes `Subscribe`
    /// usable for chat the same way the HTTP SSE surface already is.
    pub fn progress(&mut self, step: u32, total: u32, message: &str, phase: Option<&str>, delta: Option<&str>, event: Option<&serde_json::Value>) {
        self.send(
            &json!({"type": "progress", "step": step, "total": total, "message": message, "phase": phase, "delta": delta, "event": event}),
            None,
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Read exactly one SEQPACKET datagram off the client fd and parse it as a
    /// frame header (no ancillary fd expected).
    fn recv_frame(client: &OwnedFd) -> serde_json::Value {
        let mut buf = [0u8; 4096];
        let n = nix::sys::socket::recv(client.as_raw_fd(), &mut buf, MsgFlags::empty()).expect("recv one frame");
        serde_json::from_slice(&buf[..n]).expect("frame is valid JSON")
    }

    /// REGRESSION: a `progress` frame must carry `delta`/`event` when the
    /// caller supplies them — before this fix, `Subscribe` dropped BOTH
    /// unconditionally (only step/total/message ever reached the wire), so a
    /// bus subscriber to a streaming chat action received no token text at
    /// all, only the terminal blob.
    #[test]
    fn progress_frame_carries_delta_and_event() {
        let (mut tx, client) = pair().unwrap();
        tx.progress(1, 10, "generating", None, Some("hel"), Some(&json!({"kind": "reasoning", "text": "thinking"})));
        let frame = recv_frame(&client);
        assert_eq!(frame["type"], "progress");
        assert_eq!(frame["step"], 1);
        assert_eq!(frame["delta"], "hel");
        assert_eq!(frame["event"]["kind"], "reasoning");
        assert_eq!(frame["event"]["text"], "thinking");
    }

    /// A plain step tick (no delta/event) must still round-trip as explicit
    /// JSON `null`s, not an absent key — an older/simpler client can ignore
    /// them either way, but the shape must be stable.
    #[test]
    fn progress_frame_without_delta_or_event_sends_explicit_nulls() {
        let (mut tx, client) = pair().unwrap();
        tx.progress(2, 10, "working", None, None, None);
        let frame = recv_frame(&client);
        assert!(frame["delta"].is_null());
        assert!(frame["event"].is_null());
    }
}
