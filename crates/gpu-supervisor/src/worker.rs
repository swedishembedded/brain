// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for recovering accelerator
// services from wedged and crashing drivers for its clients. If your team
// needs expertise in process supervision and driver isolation then you can
// procure our services by sending an email to info@swedishembedded.com.

//! One supervised child process and the line protocol it speaks.
//!
//! The parent and the worker share one `AF_UNIX` stream socket pair; the
//! worker's end is its stdin and stdout (stderr stays the parent's, so driver
//! messages still reach the operator). A request is one line,
//! `<id> <request>\n`; the answer is one line, `<id> ok <payload>` or
//! `<id> err <message>`. One request is in flight at a time and each has a
//! fresh id, so a reply that carries an older id (an answer to a request the
//! parent already gave up on) is discarded instead of being taken for the
//! current one. Each direction has exactly one writer, so a frame can never
//! interleave with another.
//!
//! The deadline is a socket read timeout: no helper thread waits on the
//! worker, so nothing is left running when the worker is abandoned. On a
//! missed deadline, an end of stream or a malformed reply the worker's whole
//! process group is sent `SIGKILL` and the worker is reaped within a bound.
//! `SIGKILL` is what makes the kernel reclaim every file descriptor, mapping
//! and driver context the worker held; a process inside an uninterruptible
//! driver call dies only when that call returns, which is why reaping is
//! bounded and a worker that outlives the bound is handed back
//! ([`Worker::take_unreaped`]) for the caller to keep polling rather than
//! waited on forever.
//!
//! On Linux the worker is started with `PR_SET_PDEATHSIG = SIGKILL`, so it
//! cannot outlive a parent that crashes without running its destructors.
//! Linux delivers that signal when the parent THREAD that spawned the worker
//! exits, so a worker must be spawned on a thread that lives at least as long
//! as the worker is used (a device lane, not a short-lived helper thread).

use std::ffi::OsString;
use std::io::{ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::unix::net::UnixStream;

/// The largest reply line accepted. A worker answers with small JSON (bulk
/// tensors are not carried on this socket), so anything larger is a broken or
/// hostile worker and is treated as a protocol fault, never buffered.
pub const MAX_REPLY_BYTES: usize = 1 << 20;

/// The largest request line sent.
const MAX_REQUEST_BYTES: usize = 1 << 20;

/// How long a worker whose stream ended is given to finish exiting on its own
/// before it is killed, so its real exit status (the signal that crashed it)
/// is what gets reported rather than the `SIGKILL` sent to clean up.
const EXIT_GRACE: Duration = Duration::from_millis(200);

/// How to start a worker: a program and its arguments, run with `execve`.
/// Never a fork of the caller and never a re-exec of the current executable
/// (a test binary re-executing itself would run its test suite again).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerCommand {
    program: PathBuf,
    args: Vec<OsString>,
}

impl WorkerCommand {
    pub fn new(program: impl Into<PathBuf>) -> WorkerCommand {
        WorkerCommand { program: program.into(), args: Vec::new() }
    }

    pub fn arg(mut self, arg: impl Into<OsString>) -> WorkerCommand {
        self.args.push(arg.into());
        self
    }

    pub fn program(&self) -> &Path {
        &self.program
    }
}

/// Why a worker could not answer. Each one leaves the worker killed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FaultKind {
    /// The request was not answered within its deadline.
    Wedged { deadline: Duration },
    /// The worker's stream ended or a write to it failed: it died.
    Crashed { status: String },
    /// The worker answered something that is not a reply to this request.
    Protocol { detail: String },
    /// The worker process could not be started.
    Spawn { detail: String },
}

impl std::fmt::Display for FaultKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FaultKind::Wedged { deadline } => write!(f, "exceeded its {deadline:?} deadline"),
            FaultKind::Crashed { status } => write!(f, "lost its worker, which {status}"),
            FaultKind::Protocol { detail } => write!(f, "got a reply that breaks the worker protocol ({detail})"),
            FaultKind::Spawn { detail } => write!(f, "could not start its worker ({detail})"),
        }
    }
}

/// What [`Worker::request`] returns instead of an answer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkerFailure {
    /// The request was refused before anything was sent (it would break the
    /// framing, or this worker was already terminated). Says nothing about
    /// the device.
    Invalid(String),
    /// The worker failed; it has been killed and, within the bound, reaped.
    Fault(FaultKind),
}

/// A supervised child process. Dropping it kills it; the reaper thread reaps it.
pub struct Worker {
    /// `None` once reaped, or once an unreaped child was handed over.
    child: Option<Child>,
    #[cfg(unix)]
    sock: UnixStream,
    pid: u32,
    next_id: u64,
    /// Bytes read past the last complete reply.
    buf: Vec<u8>,
    reap_within: Duration,
    /// Set once the worker was killed: it never serves another request.
    terminated: bool,
    /// The exit status, once reaped.
    status: Option<String>,
}

impl Worker {
    /// Start `cmd` as a worker, reaped within 2 s of being killed.
    pub fn spawn(cmd: &WorkerCommand) -> std::io::Result<Worker> {
        Worker::spawn_with(cmd, Duration::from_secs(2))
    }

    /// Start `cmd` as a worker, reaped within `reap_within` of being killed.
    #[cfg(unix)]
    pub fn spawn_with(cmd: &WorkerCommand, reap_within: Duration) -> std::io::Result<Worker> {
        use std::os::fd::OwnedFd;
        use std::os::unix::process::CommandExt;
        let (sock, theirs) = UnixStream::pair()?;
        let stdin: OwnedFd = theirs.try_clone()?.into();
        let stdout: OwnedFd = theirs.into();
        let mut command = Command::new(&cmd.program);
        command.args(&cmd.args).stdin(Stdio::from(stdin)).stdout(Stdio::from(stdout)).stderr(Stdio::inherit());
        // Its own process group, so one kill reaches anything it started.
        command.process_group(0);
        set_parent_death_signal(&mut command);
        let child = command.spawn()?;
        // The parent's copies of the worker's end close here, so the parent
        // sees end of stream the moment the worker is gone.
        drop(command);
        let pid = child.id();
        Ok(Worker { child: Some(child), sock, pid, next_id: 1, buf: Vec::new(), reap_within, terminated: false, status: None })
    }

    #[cfg(not(unix))]
    pub fn spawn_with(cmd: &WorkerCommand, _reap_within: Duration) -> std::io::Result<Worker> {
        Err(std::io::Error::new(ErrorKind::Unsupported, format!("supervised device workers need a Unix host ({})", cmd.program.display())))
    }

    pub fn pid(&self) -> u32 {
        self.pid
    }

    /// The exit status, once the worker has been reaped.
    pub fn status(&self) -> Option<&str> {
        self.status.as_deref()
    }

    /// A killed worker the kernel has not released within the reap bound, for
    /// the caller to keep polling. `None` once reaped (or never killed).
    pub fn take_unreaped(&mut self) -> Option<Child> {
        if self.terminated {
            self.child.take()
        } else {
            None
        }
    }

    /// Send one request and wait at most `deadline` for its answer:
    /// `Ok(Ok(payload))` or `Ok(Err(message))` when the worker answered, a
    /// [`WorkerFailure`] otherwise. Every [`WorkerFailure::Fault`] leaves the
    /// worker killed and, within the reap bound, reaped; it serves no further
    /// request.
    #[cfg(unix)]
    pub fn request(&mut self, request: &str, deadline: Duration) -> Result<Result<String, String>, WorkerFailure> {
        if self.terminated {
            return Err(WorkerFailure::Invalid(format!("worker {} was terminated by an earlier fault; start a new one", self.pid)));
        }
        if request.is_empty() || request.len() > MAX_REQUEST_BYTES || request.contains(['\n', '\r']) {
            return Err(WorkerFailure::Invalid(format!("a request must be one non-empty line of at most {MAX_REQUEST_BYTES} bytes")));
        }
        let until = Instant::now() + deadline;
        let id = self.next_id;
        self.next_id += 1;
        let frame = format!("{id} {request}\n");
        let _ = self.sock.set_write_timeout(Some(deadline.max(Duration::from_millis(1))));
        if let Err(e) = self.sock.write_all(frame.as_bytes()) {
            let kind = match e.kind() {
                ErrorKind::WouldBlock | ErrorKind::TimedOut => FaultKind::Wedged { deadline },
                _ => FaultKind::Crashed { status: String::new() },
            };
            return Err(WorkerFailure::Fault(self.fail(kind)));
        }
        loop {
            match self.take_line() {
                Ok(Some(line)) => match parse_reply(&line, id) {
                    Reply::Stale => continue,
                    Reply::Answer(answer) => return Ok(answer),
                    Reply::Broken(detail) => return Err(WorkerFailure::Fault(self.fail(FaultKind::Protocol { detail }))),
                },
                Ok(None) => {}
                Err(detail) => return Err(WorkerFailure::Fault(self.fail(FaultKind::Protocol { detail }))),
            }
            let left = until.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return Err(WorkerFailure::Fault(self.fail(FaultKind::Wedged { deadline })));
            }
            let _ = self.sock.set_read_timeout(Some(left));
            let mut chunk = [0u8; 64 * 1024];
            match self.sock.read(&mut chunk) {
                Ok(0) => return Err(WorkerFailure::Fault(self.fail(FaultKind::Crashed { status: String::new() }))),
                Ok(n) => self.buf.extend_from_slice(&chunk[..n]),
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
                Err(_) => return Err(WorkerFailure::Fault(self.fail(FaultKind::Crashed { status: String::new() }))),
            }
        }
    }

    #[cfg(not(unix))]
    pub fn request(&mut self, _request: &str, _deadline: Duration) -> Result<Result<String, String>, WorkerFailure> {
        Err(WorkerFailure::Invalid("supervised device workers need a Unix host".into()))
    }

    /// The next complete line in the buffer, `Ok(None)` if there is none yet.
    fn take_line(&mut self) -> Result<Option<String>, String> {
        match self.buf.iter().position(|&b| b == b'\n') {
            Some(end) => {
                let line: Vec<u8> = self.buf.drain(..=end).collect();
                String::from_utf8(line[..end].to_vec()).map(Some).map_err(|_| "a reply that is not UTF-8".to_string())
            }
            None if self.buf.len() > MAX_REPLY_BYTES => Err(format!("a reply longer than {MAX_REPLY_BYTES} bytes")),
            None => Ok(None),
        }
    }

    /// Kill and reap after a failure; fills a crash's exit status in.
    fn fail(&mut self, kind: FaultKind) -> FaultKind {
        // A worker whose stream ended is usually already exiting: let it, so
        // the status names what killed it rather than our cleanup signal.
        let grace = if matches!(kind, FaultKind::Crashed { .. }) { EXIT_GRACE } else { Duration::ZERO };
        self.terminate(grace);
        match kind {
            FaultKind::Crashed { .. } => FaultKind::Crashed { status: self.status.clone().unwrap_or_else(|| "was killed but is not reaped yet".into()) },
            other => other,
        }
    }

    /// Kill the worker's process group and reap the worker, waiting at most
    /// `grace` for it to exit unprompted first and then at most the reap
    /// bound. Idempotent.
    fn terminate(&mut self, grace: Duration) {
        self.terminated = true;
        let Some(child) = self.child.as_mut() else { return };
        let mut status = poll_exit(child, grace);
        kill_group(self.pid);
        if status.is_none() {
            let _ = child.kill();
            status = poll_exit(child, self.reap_within);
        }
        if let Some(s) = status {
            self.status = Some(describe(s));
            self.child = None;
        }
    }
}

/// `SIGKILL` to the process group a worker leads. Always sent, even when the
/// leader already exited: what it started may still run. The group id stays
/// reserved while any member lives, so this cannot reach an unrelated process.
fn kill_group(leader: u32) {
    #[cfg(unix)]
    // SAFETY: kill(2) with a negative pid signals that process group; no memory is touched.
    unsafe {
        libc::kill(-(leader as libc::pid_t), libc::SIGKILL);
    }
    #[cfg(not(unix))]
    let _ = leader;
}

impl Drop for Worker {
    /// Kills the worker and hands it to the reaper without waiting: a worker
    /// is dropped wherever its owner is, which for an evicted instance is the
    /// scheduler's one thread, and a healthy worker still takes a moment to
    /// exit (one inside a driver call, until that call returns).
    fn drop(&mut self) {
        self.terminated = true;
        let Some(mut child) = self.child.take() else { return };
        kill_group(self.pid);
        let _ = child.kill();
        reap_later(child);
    }
}

/// Kill the worker when the parent thread that spawned it exits.
#[cfg(target_os = "linux")]
fn set_parent_death_signal(command: &mut Command) {
    use std::os::unix::process::CommandExt;
    let parent = std::process::id() as libc::pid_t;
    // SAFETY: the closure runs in the forked child before exec and calls
    // only async-signal-safe functions (prctl, getppid, _exit).
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL as libc::c_ulong, 0, 0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // The parent died between fork and prctl: nobody would ever kill this worker.
            if libc::getppid() != parent {
                libc::_exit(1);
            }
            Ok(())
        });
    }
}

/// Other Unix hosts have no parent-death hook: a worker whose parent died
/// without running its destructors exits when it reads end of stream on its
/// stdin, unless it is stuck in the driver, which only killing it ends.
#[cfg(all(unix, not(target_os = "linux")))]
fn set_parent_death_signal(_command: &mut Command) {}

enum Reply {
    /// An answer to an earlier request this worker gave up on.
    Stale,
    Answer(Result<String, String>),
    Broken(String),
}

fn parse_reply(line: &str, want: u64) -> Reply {
    let mut parts = line.splitn(3, ' ');
    let Some(id) = parts.next().and_then(|s| s.parse::<u64>().ok()) else {
        return Reply::Broken(format!("no request id in {:?}", truncate(line)));
    };
    if id < want {
        return Reply::Stale;
    }
    if id != want {
        return Reply::Broken(format!("an answer to request {id} while {want} is in flight"));
    }
    let rest = parts.next().unwrap_or("");
    let payload = parts.next().unwrap_or("").to_string();
    match rest {
        "ok" => Reply::Answer(Ok(payload)),
        "err" => Reply::Answer(Err(payload)),
        other => Reply::Broken(format!("status {:?} is neither ok nor err", truncate(other))),
    }
}

fn truncate(s: &str) -> &str {
    match s.char_indices().nth(80) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `try_wait` until the child exits or `within` passes.
fn poll_exit(child: &mut Child, within: Duration) -> Option<ExitStatus> {
    let until = Instant::now() + within;
    loop {
        match child.try_wait() {
            Ok(Some(s)) => return Some(s),
            Ok(None) if Instant::now() >= until => return None,
            Ok(None) => std::thread::sleep(Duration::from_millis(2)),
            // ECHILD: already reaped elsewhere; there is nothing left to wait for.
            Err(_) => return None,
        }
    }
}

fn describe(s: ExitStatus) -> String {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(sig) = s.signal() {
            let name = match sig {
                libc::SIGABRT => " (SIGABRT)",
                libc::SIGKILL => " (SIGKILL)",
                libc::SIGSEGV => " (SIGSEGV)",
                libc::SIGBUS => " (SIGBUS)",
                _ => "",
            };
            return format!("was terminated by signal {sig}{name}");
        }
    }
    match s.code() {
        Some(code) => format!("exited with code {code}"),
        None => format!("ended ({s})"),
    }
}

/// How often the reaper polls the workers it holds.
const REAP_POLL: Duration = Duration::from_millis(5);

/// Hand a killed worker to the process's reaper thread, which reaps it as
/// soon as the kernel releases it (at once for a healthy worker, when its
/// driver call returns for one stuck inside the driver). Never blocks: if the
/// reaper cannot be started the child is reaped here when it already exited,
/// and otherwise left to the kernel's reparenting at exit.
fn reap_later(mut child: Child) {
    if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
        return;
    }
    static REAPER: std::sync::OnceLock<Option<std::sync::mpsc::Sender<Child>>> = std::sync::OnceLock::new();
    let reaper = REAPER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Child>();
        std::thread::Builder::new().name("brain-gpu-reaper".into()).spawn(move || reaper_loop(rx)).ok().map(|_| tx)
    });
    if let Some(tx) = reaper {
        let _ = tx.send(child);
    }
}

/// The reaper thread: waits for workers while it holds none, polls the ones
/// it holds until each has exited. Runs for the life of the process (its
/// sender is never dropped).
fn reaper_loop(rx: std::sync::mpsc::Receiver<Child>) {
    let mut held: Vec<Child> = Vec::new();
    loop {
        if held.is_empty() {
            match rx.recv() {
                Ok(child) => held.push(child),
                Err(_) => return,
            }
        } else if let Ok(child) = rx.recv_timeout(REAP_POLL) {
            held.push(child);
        }
        held.extend(rx.try_iter());
        held.retain_mut(|c| matches!(c.try_wait(), Ok(None)));
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    /// A process that is still alive when it is handed over (as a killed
    /// worker inside a driver call is) does not hold up the caller, and is
    /// reaped, not left a zombie, once it exits.
    #[test]
    fn a_worker_the_kernel_has_not_released_is_reaped_later_without_blocking() {
        let child = Command::new("/bin/sh").arg("-c").arg("sleep 0.3").spawn().expect("spawn sh");
        let pid = child.id();
        let started = Instant::now();
        reap_later(child);
        assert!(started.elapsed() < Duration::from_millis(100), "handing over waited {:?}", started.elapsed());
        assert!(std::path::Path::new(&format!("/proc/{pid}")).exists(), "the process was still running when handed over");
        let until = Instant::now() + Duration::from_secs(10);
        while std::path::Path::new(&format!("/proc/{pid}")).exists() {
            assert!(Instant::now() < until, "process {pid} was never reaped");
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}
