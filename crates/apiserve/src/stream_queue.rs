// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for inference servers whose memory
// stays bounded however slowly their clients read for its clients. If your
// team needs expertise in streaming backpressure and cancellation then you can
// procure our services by sending an email to info@swedishembedded.com.

//! The bounded hand-off from a streamed generation to its response body.
//!
//! The producer is the job's progress callback on a lane thread; the consumer
//! is the response body, polled only as fast as the client reads. The queue
//! holds at most [`STREAM_QUEUE_MESSAGES`] messages, so a client that reads
//! slowly holds the generation back instead of growing memory:
//!
//! - [`StreamSender::send`] waits for room. While it waits it watches the
//!   request's cancel token and the receiver, so a disconnect or a shutdown
//!   frees it within [`CANCEL_POLL`].
//! - When the queue is full and both the waiting message and the newest
//!   queued one are text deltas, the text is appended to the queued delta, up
//!   to [`STREAM_COALESCE_BYTES`]. The concatenated text and the order of
//!   events are unchanged; only the chunking is coarser.
//! - **Stall policy.** A sender that has waited for room for the stall
//!   deadline with nothing taken from the queue cancels the generation and
//!   ends the stream with a terminal timeout error, which the reader gets
//!   after what is already queued if it ever reads again.
//! - The terminal message ([`StreamSender::finish`]) never waits: it has its
//!   own slot beside the queue, so the reply callback cannot block and a
//!   stream always ends with one terminal message, never two.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use capability::{CancelToken, FailureClass};
use futures::task::AtomicWaker;

use crate::bridge::StreamMsg;
use crate::error::ApiError;
use crate::surface::Provider;

/// The most messages one stream holds before its generation waits.
pub const STREAM_QUEUE_MESSAGES: usize = 64;

/// The size up to which a waiting text delta is appended to the newest queued
/// one instead of waiting. A queued delta is at most this plus one piece.
pub const STREAM_COALESCE_BYTES: usize = 1024;

/// The default stall deadline ([`crate::AppState::stream_stall`]): long enough
/// for any client that is reading, short enough that an abandoned connection
/// does not hold a lane.
pub const DEFAULT_STREAM_STALL: Duration = Duration::from_secs(30);

/// How often a waiting sender looks at the cancel token, which has no way to
/// wake it.
const CANCEL_POLL: Duration = Duration::from_millis(50);

struct State {
    items: VecDeque<StreamMsg>,
    /// The terminal message, handed out after every queued one.
    terminal: Option<StreamMsg>,
    /// A terminal message has been accepted; nothing more is queued.
    ended: bool,
    receiver_gone: bool,
    /// Live [`StreamSender`]s. The stream ends when the last one goes without a
    /// terminal message (a job dropped before it ran).
    senders: usize,
}

struct Shared {
    state: Mutex<State>,
    /// Signalled when room appears or the receiver goes.
    space: Condvar,
    receiver: AtomicWaker,
    capacity: usize,
    stall: Duration,
    cancel: CancelToken,
    provider: Provider,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Opens a stream queue for one request. `cancel` is the request's token:
/// the sender stops when it fires, and fires it when the stall policy ends the
/// stream.
pub(crate) fn channel(provider: Provider, cancel: CancelToken, stall: Duration) -> (StreamSender, StreamReceiver) {
    with_capacity(provider, cancel, stall, STREAM_QUEUE_MESSAGES)
}

fn with_capacity(provider: Provider, cancel: CancelToken, stall: Duration, capacity: usize) -> (StreamSender, StreamReceiver) {
    let shared = Arc::new(Shared {
        state: Mutex::new(State { items: VecDeque::with_capacity(capacity), terminal: None, ended: false, receiver_gone: false, senders: 1 }),
        space: Condvar::new(),
        receiver: AtomicWaker::new(),
        capacity,
        stall,
        cancel,
        provider,
    });
    (StreamSender { shared: Arc::clone(&shared) }, StreamReceiver { shared })
}

/// The producing end, held by the job's callbacks. Blocking: use it from a
/// worker thread, never from an async task.
pub(crate) struct StreamSender {
    shared: Arc<Shared>,
}

impl Clone for StreamSender {
    fn clone(&self) -> StreamSender {
        self.shared.lock().senders += 1;
        StreamSender { shared: Arc::clone(&self.shared) }
    }
}

impl Drop for StreamSender {
    fn drop(&mut self) {
        let last = {
            let mut state = self.shared.lock();
            state.senders -= 1;
            state.senders == 0
        };
        if last {
            self.shared.receiver.wake();
        }
    }
}

impl StreamSender {
    /// Queues `msg`, waiting for room. `false` when the stream takes nothing
    /// more (the reader went, the request was cancelled, the stream ended or
    /// stalled), and `msg` is dropped; the generation should stop.
    pub(crate) fn send(&self, msg: StreamMsg) -> bool {
        let shared = &*self.shared;
        let mut state = shared.lock();
        let mut waiting_since: Option<Instant> = None;
        loop {
            if state.ended || state.receiver_gone || shared.cancel.is_cancelled() {
                return false;
            }
            if state.items.len() < shared.capacity {
                state.items.push_back(msg);
                drop(state);
                shared.receiver.wake();
                return true;
            }
            if let (StreamMsg::Delta(piece), Some(StreamMsg::Delta(newest))) = (&msg, state.items.back_mut()) {
                if newest.len() < STREAM_COALESCE_BYTES {
                    newest.push_str(piece);
                    return true;
                }
            }
            let waited = waiting_since.get_or_insert_with(Instant::now).elapsed();
            if waited >= shared.stall {
                state.ended = true;
                state.terminal = Some(StreamMsg::Err(stalled(shared.provider, shared.stall)));
                drop(state);
                shared.cancel.cancel();
                shared.receiver.wake();
                residency::log::info(&format!("{:?} stream: the client read nothing for {:?}; generation cancelled", shared.provider, shared.stall));
                return false;
            }
            state = shared.space.wait_timeout(state, CANCEL_POLL.min(shared.stall - waited)).unwrap_or_else(PoisonError::into_inner).0;
        }
    }

    /// Queues `msg` only if there is room right now, for a message the next one
    /// supersedes (a fetch progress tick). `false` when it was dropped.
    pub(crate) fn offer(&self, msg: StreamMsg) -> bool {
        let mut state = self.shared.lock();
        if state.ended || state.receiver_gone || state.items.len() >= self.shared.capacity {
            return false;
        }
        state.items.push_back(msg);
        drop(state);
        self.shared.receiver.wake();
        true
    }

    /// Ends the stream with `msg`, after everything queued. Never waits. A
    /// stream that already ended (the stall policy wrote its own terminal
    /// error) keeps its first terminal message and drops this one.
    pub(crate) fn finish(&self, msg: StreamMsg) {
        let mut state = self.shared.lock();
        if state.ended {
            return;
        }
        state.ended = true;
        state.terminal = Some(msg);
        drop(state);
        self.shared.receiver.wake();
    }
}

/// The terminal error of a stream whose reader stalled.
fn stalled(provider: Provider, stall: Duration) -> ApiError {
    ApiError::failure(provider, FailureClass::Timeout, format!("the client read none of the stream for {stall:?}; the generation was cancelled"))
}

/// The consuming end, inside the response body.
pub(crate) struct StreamReceiver {
    shared: Arc<Shared>,
}

impl StreamReceiver {
    /// The next message: queued ones first, then the terminal one, then `None`.
    pub(crate) fn poll_recv(&mut self, cx: &mut Context<'_>) -> Poll<Option<StreamMsg>> {
        // Registered before looking, so a message queued in between still wakes this task.
        self.shared.receiver.register(cx.waker());
        let mut state = self.shared.lock();
        if let Some(msg) = state.items.pop_front() {
            drop(state);
            self.shared.space.notify_all();
            return Poll::Ready(Some(msg));
        }
        if let Some(msg) = state.terminal.take() {
            return Poll::Ready(Some(msg));
        }
        if state.ended || state.senders == 0 {
            return Poll::Ready(None);
        }
        Poll::Pending
    }

    pub(crate) async fn recv(&mut self) -> Option<StreamMsg> {
        std::future::poll_fn(|cx| self.poll_recv(cx)).await
    }

    #[cfg(test)]
    fn queued(&self) -> usize {
        self.shared.lock().items.len()
    }
}

impl Drop for StreamReceiver {
    fn drop(&mut self) {
        self.shared.lock().receiver_gone = true;
        self.shared.space.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A producer that outruns its reader fills the queue to its capacity,
    /// coalesces at most up to the limit, then waits; the reader still gets the
    /// whole text in order, then the terminal message.
    #[tokio::test]
    async fn a_full_queue_holds_the_producer_and_loses_nothing() {
        let (tx, mut rx) = with_capacity(Provider::OpenAI, CancelToken::armed(), Duration::from_secs(30), 4);
        let sent = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&sent);
        let producer = std::thread::spawn(move || {
            for i in 0..2000 {
                assert!(tx.send(StreamMsg::Delta(format!("{i:04} "))));
                counter.fetch_add(1, Ordering::SeqCst);
            }
            tx.finish(StreamMsg::Progress(1, 1));
        });
        tokio::time::sleep(Duration::from_millis(200)).await;
        let held = sent.load(Ordering::SeqCst);
        assert!(held < 2000, "the producer waits for the reader");
        assert_eq!(rx.queued(), 4, "the queue is full and no fuller");
        // Four single deltas, the newest then coalesced up to the limit.
        assert!(held <= 4 + STREAM_COALESCE_BYTES / 5 + 1, "{held} deltas taken by a 4-message queue");

        let mut got = String::new();
        let terminal = loop {
            match rx.recv().await.expect("the stream ends with its terminal message") {
                StreamMsg::Delta(t) => {
                    assert!(t.len() <= STREAM_COALESCE_BYTES + 5);
                    got.push_str(&t);
                }
                other => break other,
            }
        };
        assert!(matches!(terminal, StreamMsg::Progress(1, 1)));
        assert!(rx.recv().await.is_none());
        producer.join().unwrap();
        assert_eq!(got, (0..2000).map(|i| format!("{i:04} ")).collect::<String>());
    }

    /// A producer whose reader takes nothing for the stall deadline cancels the
    /// request and ends the stream with a timeout error; the reply that follows
    /// is not a second terminal message.
    #[tokio::test]
    async fn a_stalled_reader_cancels_the_request_and_gets_one_timeout() {
        let token = CancelToken::armed();
        let (tx, mut rx) = with_capacity(Provider::OpenAI, token.clone(), Duration::from_millis(100), 2);
        let producer = std::thread::spawn(move || {
            let mut sent = 0;
            while tx.send(StreamMsg::Event(serde_json::json!({"n": sent}))) {
                sent += 1;
            }
            tx.finish(StreamMsg::Progress(9, 9));
            sent
        });
        let sent = producer.join().unwrap();
        assert_eq!(sent, 2, "events do not coalesce: the queue took two");
        assert!(token.is_cancelled(), "the stall cancels the generation");
        assert!(matches!(rx.recv().await, Some(StreamMsg::Event(_))));
        assert!(matches!(rx.recv().await, Some(StreamMsg::Event(_))));
        match rx.recv().await {
            Some(StreamMsg::Err(e)) => assert_eq!(e.body()["error"]["code"], "timeout"),
            _ => panic!("expected the stall's timeout error"),
        }
        assert!(rx.recv().await.is_none(), "the late reply is dropped");
    }

    /// A waiting producer is released at once when the reader goes or the
    /// request is cancelled, long before the stall deadline.
    #[test]
    fn a_waiting_producer_is_released_by_a_disconnect_or_a_cancel() {
        for disconnect in [true, false] {
            let token = CancelToken::armed();
            let (tx, rx) = with_capacity(Provider::OpenAI, token.clone(), Duration::from_secs(60), 1);
            assert!(tx.send(StreamMsg::Progress(0, 1)));
            let started = Instant::now();
            let producer = std::thread::spawn(move || tx.send(StreamMsg::Progress(1, 1)));
            std::thread::sleep(Duration::from_millis(100));
            if disconnect {
                drop(rx);
            } else {
                token.cancel();
            }
            assert!(!producer.join().unwrap(), "the waiting message is refused");
            assert!(started.elapsed() < Duration::from_secs(5));
        }
    }

    /// Senders that all go without a terminal message (a job dropped before it
    /// ran) end the stream instead of leaving the reader waiting.
    #[tokio::test]
    async fn the_stream_ends_when_every_sender_is_gone() {
        let (tx, mut rx) = with_capacity(Provider::OpenAI, CancelToken::armed(), Duration::from_secs(30), 4);
        let second = tx.clone();
        assert!(second.send(StreamMsg::Progress(0, 1)));
        drop(tx);
        drop(second);
        assert!(matches!(rx.recv().await, Some(StreamMsg::Progress(0, 1))));
        assert!(rx.recv().await.is_none());
    }

    /// A tick offered to a full queue is dropped rather than waited for.
    #[test]
    fn an_offered_tick_never_waits() {
        let (tx, _rx) = with_capacity(Provider::OpenAI, CancelToken::armed(), Duration::from_secs(30), 1);
        assert!(tx.offer(StreamMsg::Progress(0, 2)));
        assert!(!tx.offer(StreamMsg::Progress(1, 2)));
    }
}
