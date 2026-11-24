// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for long-running served jobs that keep taking input from their client while
// they run, for its clients. If your team needs expertise in streaming work into a running training or inference job
// then you can procure our services by sending an email to info@swedishembedded.com.

//! Input to a running job: what a client sends a job after it started (a streaming training job's rollout records,
//! its `finish`), each message answered by the job.
//!
//! The surface that started the job keeps the [`InboxSender`]; the job takes the receiving end from its
//! [`Invocation`](crate::Invocation)'s [`Inbox`] once and answers each [`JobInput`] on its reply channel. Like the
//! [`CancelToken`](crate::CancelToken) it rides beside, an inbox is unarmed by default: an invocation that crossed a
//! process boundary, or came from a surface that delivers no input, has none, and an action that needs one refuses
//! the call by name. A sender reports at once when the job does not read its inbox (it never took it) or has ended
//! (it dropped it), so a client is never left waiting on a job that cannot answer.
//!
//! Every message is bounded: the surface caps the body before it is sent ([`MAX_INPUT_BYTES`]), the channel holds a
//! few messages ([`QUEUED_INPUTS`]), and the job decides what it accepts.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;

/// The largest body one input message carries.
pub const MAX_INPUT_BYTES: usize = 64 * 1024 * 1024;
/// Messages waiting for the job to read them before a sender is told the job is busy.
pub const QUEUED_INPUTS: usize = 4;
/// How long a surface waits for a job to answer one input.
pub const INPUT_WAIT: Duration = Duration::from_secs(120);

/// Why a job did not take an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputRefusalKind {
    /// The message is malformed or breaks the job's contract; resending it unchanged cannot succeed.
    Invalid,
    /// The job has no room for it now (a full queue); the refusal names how much it takes.
    Busy,
    /// The message disagrees with the job's state (another run epoch, input after `finish`).
    Conflict,
    /// The job does not read input, or has ended.
    Gone,
    /// The job is not ready to answer yet (it is starting) or did not answer in time: retry.
    Unavailable,
}

/// A refused input: its kind, a message the client may read, and the job's own detail (an allowance, an epoch).
#[derive(Clone, Debug, PartialEq)]
pub struct InputRefusal {
    pub kind: InputRefusalKind,
    pub message: String,
    pub detail: Value,
}

impl InputRefusal {
    pub fn new(kind: InputRefusalKind, message: impl Into<String>) -> InputRefusal {
        InputRefusal { kind, message: message.into(), detail: Value::Null }
    }

    pub fn with_detail(mut self, detail: Value) -> InputRefusal {
        self.detail = detail;
        self
    }
}

/// The job's answer to one input.
pub type InputReply = Result<Value, InputRefusal>;

/// One message to a running job.
pub struct JobInput {
    /// What the message is (`records`, `finish`): the job's own vocabulary.
    pub kind: String,
    /// Its query, as the surface parsed it (`epoch`), else `null`.
    pub query: Value,
    pub body: Vec<u8>,
    reply: mpsc::SyncSender<InputReply>,
}

impl JobInput {
    /// Answer the message; a sender that stopped waiting is not an error of the job's.
    pub fn answer(self, reply: InputReply) {
        let _ = self.reply.send(reply);
    }
}

struct Shared {
    receiver: Mutex<Option<mpsc::Receiver<JobInput>>>,
    /// The job took the receiving end: it reads its input.
    taken: AtomicBool,
    /// The job said, before it started, that it will take its input ([`Inbox::will_read`]).
    expected: AtomicBool,
}

/// The receiving end of a job's input, as it rides on its invocation. See the module doc.
#[derive(Clone, Default)]
pub struct Inbox(Option<Arc<Shared>>);

impl std::fmt::Debug for Inbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "Inbox(armed)" } else { "Inbox(unarmed)" })
    }
}

/// The surface's end of a job's input.
#[derive(Clone)]
pub struct InboxSender {
    sender: mpsc::SyncSender<JobInput>,
    shared: Arc<Shared>,
}

/// A new inbox: the sender the surface keeps and the inbox the job's invocation carries.
pub fn inbox() -> (InboxSender, Inbox) {
    let (sender, receiver) = mpsc::sync_channel(QUEUED_INPUTS);
    let shared = Arc::new(Shared { receiver: Mutex::new(Some(receiver)), taken: AtomicBool::new(false), expected: AtomicBool::new(false) });
    (InboxSender { sender, shared: Arc::clone(&shared) }, Inbox(Some(shared)))
}

impl Inbox {
    /// Whether a surface delivers input to this invocation.
    pub fn is_armed(&self) -> bool {
        self.0.is_some()
    }

    /// Declare, at the job's door (before it is queued), that the job will take its input: a sender then waits for the
    /// job to start instead of answering at once that it reads none.
    pub fn will_read(&self) {
        if let Some(shared) = &self.0 {
            shared.expected.store(true, Ordering::SeqCst);
        }
    }

    /// The receiving end, once: the job that calls this reads its input. `None` when unarmed or already taken.
    pub fn take(&self) -> Option<mpsc::Receiver<JobInput>> {
        let shared = self.0.as_ref()?;
        let receiver = shared.receiver.lock().unwrap_or_else(|e| e.into_inner()).take()?;
        shared.taken.store(true, Ordering::SeqCst);
        Some(receiver)
    }
}

impl InboxSender {
    /// Send one message and wait (at most `wait`) for the job's answer. A job that neither took its inbox nor said it
    /// will ([`Inbox::will_read`]), or has ended, is refused here as [`InputRefusalKind::Gone`]; one that does not answer
    /// within `wait` (still starting, or busy) as [`InputRefusalKind::Unavailable`].
    /// Blocking: call it off an async runtime's workers.
    pub fn send(&self, kind: &str, query: Value, body: Vec<u8>, wait: Duration) -> InputReply {
        if body.len() > MAX_INPUT_BYTES {
            return Err(InputRefusal::new(InputRefusalKind::Invalid, format!("the input is {} bytes; at most {MAX_INPUT_BYTES} are accepted", body.len())));
        }
        if !self.shared.taken.load(Ordering::SeqCst) && !self.shared.expected.load(Ordering::SeqCst) {
            return Err(InputRefusal::new(InputRefusalKind::Gone, "this job reads no input"));
        }
        let (reply, answer) = mpsc::sync_channel(1);
        match self.sender.try_send(JobInput { kind: kind.to_string(), query, body, reply }) {
            Ok(()) => {}
            Err(mpsc::TrySendError::Full(_)) => return Err(InputRefusal::new(InputRefusalKind::Unavailable, "the job has not read the inputs already sent; retry")),
            Err(mpsc::TrySendError::Disconnected(_)) => return Err(InputRefusal::new(InputRefusalKind::Gone, "the job has ended")),
        }
        match answer.recv_timeout(wait) {
            Ok(reply) => reply,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(InputRefusal::new(InputRefusalKind::Unavailable, format!("the job did not answer within {} s (it may still be starting); retry", wait.as_secs()))),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(InputRefusal::new(InputRefusalKind::Gone, "the job ended before it answered")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const WAIT: Duration = Duration::from_secs(5);

    #[test]
    fn a_job_answers_what_its_surface_sends_and_a_job_that_reads_nothing_or_ended_says_so() {
        let (sender, inbox) = inbox();
        assert_eq!(sender.send("records", Value::Null, b"x".to_vec(), WAIT).unwrap_err().kind, InputRefusalKind::Gone, "a job that reads none");
        // A job that will read its input but has not started yet: the message waits for it.
        inbox.will_read();
        let early = { let sender = sender.clone(); std::thread::spawn(move || sender.send("records", Value::Null, b"early".to_vec(), WAIT)) };
        let receiver = inbox.take().expect("armed");
        let first = receiver.recv().unwrap();
        assert_eq!(first.body, b"early");
        first.answer(Ok(json!("first")));
        assert_eq!(early.join().unwrap().unwrap(), json!("first"));
        assert!(inbox.clone().take().is_none(), "taken once");
        let job = std::thread::spawn(move || {
            for input in receiver.iter().take(1) {
                let (kind, n) = (input.kind.clone(), input.body.len());
                input.answer(Ok(json!({"kind": kind, "bytes": n})));
            }
        });
        assert_eq!(sender.send("records", Value::Null, b"abc".to_vec(), WAIT).unwrap(), json!({"kind": "records", "bytes": 3}));
        job.join().unwrap();
        assert_eq!(sender.send("finish", Value::Null, Vec::new(), WAIT).unwrap_err().kind, InputRefusalKind::Gone, "the job dropped its end");
        assert!(!Inbox::default().is_armed() && Inbox::default().take().is_none());
    }
}
