// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Streaming transport for `Subscribe`: a `SOCK_SEQPACKET` pair whose client end
//! is handed back over D-Bus. The worker sends one **framed datagram per event**;
//! `SEQPACKET` preserves message boundaries, so each `sendmsg` is exactly one
//! `recvmsg` on the client — no length-prefix framing needed.
//!
//! Frames are a small JSON header; a `blob` frame additionally carries a memfd as
//! an out-of-band `SCM_RIGHTS` ancillary fd. Sends distinguish two frame classes:
//!
//! * **Droppable** (`progress`/`segment`): non-blocking — a slow or stalled
//!   subscriber never stalls the inference path; a full buffer drops the frame
//!   (per the "never await a slow subscriber" rule).
//! * **Terminal** (`blob`/`done`/`error`): these carry the job's OUTCOME, and a
//!   subscriber that misses one hangs forever waiting for a completion that will
//!   never come (draining stale progress and then blocking on `recvmsg`). So a
//!   full buffer here gets a **bounded** wait for writability
//!   ([`TERMINAL_SEND_TIMEOUT`]) instead of a silent drop; only a peer that
//!   stays unwritable past the deadline is declared dead (`disconnected`),
//!   which visibly ends the stream rather than leaving it "open" minus its
//!   result.

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};

use nix::poll::{poll, PollFd, PollFlags, PollTimeout};
use nix::sys::socket::{
    sendmsg, socketpair, AddressFamily, ControlMessage, MsgFlags, SockFlag, SockType,
};
use serde_json::json;

/// Total bounded wait for a terminal (`blob`/`done`/`error`) frame when the
/// subscriber's buffer is full: long enough to ride out a briefly-stalled
/// reader (GC pause, page fault, a burst of frames it hasn't drained yet),
/// short enough that an abandoned subscriber cannot pin a worker thread
/// indefinitely. Droppable frames never wait at all.
const TERMINAL_SEND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

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
    /// Send one droppable frame (JSON header, no fd). Non-blocking: a full
    /// client buffer drops the frame (`dropped += 1`); a dead peer sets
    /// `disconnected`. Never returns an error for those expected conditions.
    fn send(&mut self, header: &serde_json::Value, fd: Option<BorrowedFd>) {
        if self.disconnected {
            return;
        }
        if self.try_send(header, fd).is_err() {
            self.dropped += 1;
        }
    }

    /// Send one terminal frame (`blob`/`done`/`error`): a full buffer is NOT a
    /// drop — wait (bounded, [`TERMINAL_SEND_TIMEOUT`]) for the peer to drain
    /// and retry, and only a peer that never becomes writable is declared dead.
    /// A dropped terminal frame is a protocol violation (the module docs'
    /// contract is "progress frames are dropped" — completion is not progress):
    /// the subscriber would drain its stale progress backlog and then block
    /// forever on a `done` that was silently discarded.
    fn send_terminal(&mut self, header: &serde_json::Value, fd: Option<BorrowedFd>) {
        if self.disconnected {
            return;
        }
        let deadline = std::time::Instant::now() + TERMINAL_SEND_TIMEOUT;
        loop {
            match self.try_send(header, fd) {
                Err(nix::errno::Errno::EAGAIN) => {}
                _ => return, // sent, or the peer died (disconnected latched)
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                // Peer stayed unwritable for the whole window: declare it dead
                // (visibly ends the stream) rather than silently dropping its
                // completion while keeping the stream "open".
                self.dropped += 1;
                self.disconnected = true;
                return;
            }
            let mut fds = [PollFd::new(self.sock.as_fd(), PollFlags::POLLOUT)];
            let timeout = PollTimeout::try_from(remaining.as_millis().min(u16::MAX as u128) as u16)
                .unwrap_or(PollTimeout::MAX);
            match poll(&mut fds, timeout) {
                Ok(_) => {} // writable, hung up, or timed out — the retry decides
                Err(nix::errno::Errno::EINTR) => {}
                Err(_) => {
                    self.disconnected = true;
                    return;
                }
            }
        }
    }

    /// One `sendmsg` attempt (non-blocking). `EPIPE`-class errors latch
    /// `disconnected`; every other error is returned to the caller so the two
    /// send policies above can differ on `EAGAIN`.
    fn try_send(&mut self, header: &serde_json::Value, fd: Option<BorrowedFd>) -> Result<(), nix::errno::Errno> {
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
            Ok(_) => Ok(()),
            Err(nix::errno::Errno::EPIPE | nix::errno::Errno::ECONNRESET | nix::errno::Errno::ENOTCONN) => {
                self.disconnected = true;
                Ok(())
            }
            Err(e) => Err(e),
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
    /// Terminal-class: the result fd would be lost with the frame, so it gets
    /// the bounded-wait delivery, never the silent drop.
    pub fn blob(&mut self, name: &str, media: &str, meta: &serde_json::Value, fd: BorrowedFd) {
        self.send_terminal(&json!({"type": "blob", "name": name, "media": media, "meta": meta}), Some(fd));
    }

    /// Terminal success frame carrying the scalar result JSON.
    pub fn done(&mut self, result: &serde_json::Value) {
        self.send_terminal(&json!({"type": "done", "result": result}), None);
    }

    /// Terminal error frame.
    pub fn error(&mut self, message: &str) {
        self.send_terminal(&json!({"type": "error", "message": message}), None);
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

    /// Fill the client's socket buffer with progress frames until one is
    /// dropped (back-pressure engaged), returning how many were actually sent.
    fn stall_subscriber(tx: &mut StreamTx) -> u64 {
        let mut sent = 0u64;
        for step in 0..1_000_000u32 {
            tx.progress(step, 1_000_000, "flood", None, None, None);
            if tx.dropped > 0 {
                return sent;
            }
            sent += 1;
        }
        panic!("could not fill the socket buffer");
    }

    /// SPEC (the F3 contract): a subscriber whose buffer is full at the moment
    /// the job completes — easy: many progress frames + a briefly stalled
    /// reader — must STILL receive the terminal `done` frame once it resumes
    /// reading. Before this fix, `done` was dropped on EAGAIN exactly like a
    /// progress frame, and the resumed reader drained stale progress and then
    /// blocked forever on a completion that no longer existed.
    #[test]
    fn a_briefly_stalled_subscriber_still_receives_the_terminal_done_frame() {
        let (mut tx, client) = pair().unwrap();
        stall_subscriber(&mut tx);

        // The reader resumes shortly after the terminal send starts waiting.
        let drainer = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            let mut buf = [0u8; 65536];
            loop {
                let n = nix::sys::socket::recv(client.as_raw_fd(), &mut buf, MsgFlags::empty()).expect("recv");
                assert!(n > 0, "stream ended before a done frame arrived");
                let frame: serde_json::Value = serde_json::from_slice(&buf[..n]).expect("frame JSON");
                if frame["type"] == "done" {
                    assert_eq!(frame["result"]["text"], "finished");
                    return;
                }
                assert_eq!(frame["type"], "progress", "unexpected frame before done");
            }
        });

        // Called while the buffer is full: must wait for the drain, not drop.
        tx.done(&json!({"text": "finished"}));
        assert!(!tx.disconnected, "a peer that resumed within the deadline must not be declared dead");
        drainer.join().expect("subscriber must observe the done frame");
    }

    /// The complement: a peer that NEVER drains cannot pin the worker forever —
    /// after the bounded wait it is declared dead and the stream visibly ends.
    /// (Uses the full TERMINAL_SEND_TIMEOUT, so this is a slow-ish test by
    /// design — the deadline IS the spec.)
    #[test]
    #[ignore = "slow by design: waits out the full TERMINAL_SEND_TIMEOUT deadline"]
    fn a_permanently_stalled_subscriber_is_declared_dead_after_the_deadline() {
        let (mut tx, _client) = pair().unwrap();
        stall_subscriber(&mut tx);
        let t0 = std::time::Instant::now();
        tx.done(&json!({"text": "finished"}));
        assert!(tx.disconnected, "an unwritable peer must be declared dead, not silently skipped");
        assert!(t0.elapsed() >= TERMINAL_SEND_TIMEOUT, "the wait must be the full bounded deadline");
    }

    /// Droppable frames keep the old contract: a full buffer drops them
    /// immediately (no waiting), and the peer stays connected.
    #[test]
    fn progress_frames_still_drop_immediately_under_backpressure() {
        let (mut tx, _client) = pair().unwrap();
        stall_subscriber(&mut tx);
        let before = tx.dropped;
        let t0 = std::time::Instant::now();
        tx.progress(1, 2, "tick", None, None, None);
        assert!(t0.elapsed() < std::time::Duration::from_millis(100), "a droppable frame must never wait");
        assert_eq!(tx.dropped, before + 1);
        assert!(!tx.disconnected);
    }
}
