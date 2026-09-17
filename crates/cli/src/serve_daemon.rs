// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Detached serving for `brain serve`: `-d` to background the server, and
//! `--reload` to replace whatever is already running.
//!
//! Swedish Embedded AB implements process lifecycle for inference servers --
//! single-instance guarantees, readiness handshakes, supervised restarts --
//! for clients running models on their own hardware. If your team needs
//! expertise in daemonizing and supervising GPU services, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! # Why this is a flag and not a supervisor
//!
//! `docker compose up -d` looks like this but is not comparable: there, `-d`
//! hands work to a daemon (`dockerd`) that is already running and owns
//! lifecycle. brain has no such daemon, so `-d` means brain detaches ITSELF.
//! Three things follow, and all three are the actual work here:
//!
//! * **Double fork + `setsid`.** One fork alone leaves the server in the
//!   launching shell's process group, where a Ctrl-C or a closing terminal
//!   still reaches it. `setsid` between the two forks makes the daemon a
//!   session leader with no controlling terminal, and the second fork ensures
//!   it is not a session leader itself, so it can never acquire one later.
//! * **An flock'd pidfile.** "Only one server" has to be atomic. A pid file
//!   that is merely written and read races two simultaneous starts, and a
//!   `pgrep`-style pattern match cannot tell one user's server from another's.
//!   A lock held on an open file description is released by the kernel when
//!   the process dies, so a crashed server leaves no stale lock to clean up.
//! * **A readiness handshake.** `-d` returning before the server is listening
//!   would be a lie a caller cannot defend against; that is the exact failure
//!   `--ready-file` already exists to prevent, so the parent waits on it
//!   rather than inventing a second readiness mechanism.
//!
//! Deliberately NOT here: any notion of watching the binary and restarting
//! when it changes. `--reload` is imperative -- the caller says when -- so it
//! never has to decide what a mid-flight job, a resident weight set or a GPU
//! allocation should do at a moment nobody asked for.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// How long `--reload` waits for the previous server to exit on `SIGTERM`
/// before escalating to `SIGKILL`. Generous: a server shutting down may be
/// finishing an in-flight job and freeing device memory.
pub const STOP_TIMEOUT: Duration = Duration::from_secs(30);

/// How long `-d` waits for the detached server to report itself listening.
/// Model scanning and a cold activation happen before the first surface
/// binds, so this is minutes, not seconds.
pub const READY_TIMEOUT: Duration = Duration::from_secs(600);

/// Where the pidfile, log and default ready file live.
///
/// `$XDG_RUNTIME_DIR` when the platform provides one (per-user, cleaned on
/// logout, which is exactly what this state is), else a per-user directory
/// inside the environment's temp dir. The uid suffix matters: the path must
/// never be shared, or two users' servers would collide on one lock.
pub fn state_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("BRAIN_RUNTIME_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(dir).join("brain");
    }
    std::env::temp_dir().join(format!("brain-{}", unsafe { libc::getuid() }))
}

pub fn pid_path() -> PathBuf {
    state_dir().join("serve.pid")
}
pub fn log_path() -> PathBuf {
    state_dir().join("serve.log")
}
pub fn default_ready_path() -> PathBuf {
    state_dir().join("serve.ready")
}

/// An exclusive claim on "the brain server on this machine", held for as long
/// as the file stays open. Dropping it (or the process dying, however
/// abruptly) releases the lock.
pub struct PidGuard {
    file: File,
}

impl PidGuard {
    /// Keep the claim for the rest of the process's life.
    ///
    /// The lock belongs to the open file description, so the claim lasts
    /// exactly as long as the fd does. Leaking it deliberately is the whole
    /// mechanism: there is no correct moment for a serving process to release
    /// it early.
    pub fn hold_forever(self) {
        std::mem::forget(self.file);
    }
}

/// The outcome of trying to claim the pidfile.
pub enum Claim {
    /// The claim is ours.
    Ours(PidGuard),
    /// Someone else holds it; their pid if the file was readable.
    HeldBy(Option<i32>),
}

/// Try to claim the single-server lock, writing our pid into it on success.
pub fn claim(path: &Path) -> io::Result<Claim> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path)?;
    // LOCK_NB: report the conflict rather than blocking behind a healthy
    // server, so `brain serve -d` fails fast the way `docker run` does on a
    // taken port.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            return Ok(Claim::HeldBy(read_pid(path)));
        }
        return Err(err);
    }
    file.set_len(0)?;
    file.rewind()?;
    write!(file, "{}", std::process::id())?;
    file.flush()?;
    Ok(Claim::Ours(PidGuard { file }))
}

/// The pid recorded in `path`, if it parses.
pub fn read_pid(path: &Path) -> Option<i32> {
    let mut s = String::new();
    File::open(path).ok()?.read_to_string(&mut s).ok()?;
    s.trim().parse().ok()
}

/// Whether `pid` is a live process we may signal.
///
/// `kill(pid, 0)` also succeeds for a ZOMBIE -- a process that has exited but
/// whose parent has not reaped it. That never affects the uses here: `stop`
/// and `wait_ready` both run in a process that is not the server's parent (a
/// detached server is reparented to init, which reaps it immediately), so a
/// pid this sees as alive is genuinely running. A caller signalling its OWN
/// child would need `waitpid` instead.
pub fn alive(pid: i32) -> bool {
    pid > 0 && unsafe { libc::kill(pid, 0) } == 0
}

/// The pid of a server currently holding `path`, if any is actually running.
pub fn running(path: &Path) -> Option<i32> {
    read_pid(path).filter(|p| alive(*p))
}

/// Stop the server recorded in `path`: `SIGTERM`, then `SIGKILL` if it has not
/// exited within `timeout`. `Ok(None)` when nothing was running.
pub fn stop(path: &Path, timeout: Duration) -> io::Result<Option<i32>> {
    let Some(pid) = running(path) else {
        let _ = std::fs::remove_file(path);
        return Ok(None);
    };
    unsafe { libc::kill(pid, libc::SIGTERM) };
    let deadline = Instant::now() + timeout;
    while alive(pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }
    if alive(pid) {
        unsafe { libc::kill(pid, libc::SIGKILL) };
        let hard = Instant::now() + Duration::from_secs(5);
        while alive(pid) && Instant::now() < hard {
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    let _ = std::fs::remove_file(path);
    Ok(Some(pid))
}


/// Handle `brain serve`'s lifecycle flags **before anything probes a device**,
/// returning the argv the rest of `main` should use.
///
/// Placement is the whole point. `main` resolves `--device`/`--backend` by
/// enumerating adapters, and that enumeration builds a Vulkan instance cached
/// in a process-wide `OnceLock`. Vulkan is explicitly not fork-safe: the cached
/// registry is plain memory and survives a fork intact, but the instance behind
/// it does not. Detaching after that point produced a daemon that still
/// believed it had two Tesla P40s, failed the next enumeration with "wgpu
/// enumerated 0 adapters", and silently fell back to a software rasteriser --
/// a server that ran every model on the CPU while reporting success.
///
/// So the fork happens first, before any of it. `--status` and `--stop` are
/// handled here too: neither forks, but neither has any reason to pay for an
/// adapter probe.
///
/// A detached run always gets a `--ready-file`, appended here when the caller
/// did not name one, because that file is what the parent waits on.
pub fn lifecycle(mut argv: Vec<String>) -> Vec<String> {
    if argv.get(1).map(String::as_str) != Some("serve") {
        return argv;
    }
    let has = |flag: &str| argv.iter().any(|a| a == flag);
    let (detach, reload, stop_flag, status) =
        (has("-d") || has("--detach"), has("--reload"), has("--stop"), has("--status"));
    if !(detach || reload || stop_flag || status) {
        return argv;
    }

    let pid_file = pid_path();
    if status {
        match running(&pid_file) {
            Some(pid) => println!("brain serve: running, pid {pid} (log {})", log_path().display()),
            None => println!("brain serve: not running"),
        }
        std::process::exit(0);
    }
    if stop_flag {
        match stop(&pid_file, STOP_TIMEOUT) {
            Ok(Some(pid)) => println!("brain serve: stopped pid {pid}"),
            Ok(None) => println!("brain serve: not running"),
            Err(e) => {
                eprintln!("brain serve: --stop: {e}");
                std::process::exit(1);
            }
        }
        std::process::exit(0);
    }
    if reload {
        match stop(&pid_file, STOP_TIMEOUT) {
            Ok(Some(pid)) => eprintln!("brain serve: --reload: replaced pid {pid}"),
            Ok(None) => {}
            Err(e) => {
                eprintln!("brain serve: --reload: {e}");
                std::process::exit(1);
            }
        }
    } else if let Some(pid) = running(&pid_file) {
        // Advisory only; the authoritative claim is the lock taken below. It
        // turns the common case into one clear sentence rather than a server
        // that starts and then loses the race for the bus name.
        eprintln!("brain serve: already running (pid {pid}); use --reload to replace it, or --stop");
        std::process::exit(1);
    }

    if detach {
        let ready = match argv.iter().position(|a| a == "--ready-file") {
            Some(i) => argv.get(i + 1).map(PathBuf::from).unwrap_or_else(default_ready_path),
            None => {
                let p = default_ready_path();
                argv.push("--ready-file".to_string());
                argv.push(p.to_string_lossy().into_owned());
                p
            }
        };
        let log = log_path();
        if let Err(e) = detach_process(&ready, &pid_file, &log, READY_TIMEOUT) {
            eprintln!("brain serve: -d: {e}");
            std::process::exit(1);
        }
        // Only the daemon reaches here.
    }

    // Taken by whichever process actually serves, and held for its whole life.
    match claim(&pid_file) {
        Ok(Claim::Ours(guard)) => guard.hold_forever(),
        Ok(Claim::HeldBy(pid)) => {
            eprintln!("brain serve: another server holds {} (pid {pid:?})", pid_file.display());
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("brain serve: {}: {e}", pid_file.display());
            std::process::exit(1);
        }
    }
    argv
}

/// Detach into the background, returning **only in the daemon**.
///
/// The caller continues exactly as it would have in the foreground; the
/// original process never returns from here, exiting 0 once the daemon
/// reports ready or non-zero if it died first.
///
/// `ready` must be the same path handed to `--ready-file`, since that is the
/// signal being waited on.
pub fn detach_process(ready: &Path, pid_file: &Path, log: &Path, timeout: Duration) -> io::Result<()> {
    if let Some(parent) = log.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // A ready file left by a previous run would make the wait below return
    // instantly against a server that is not up yet.
    let _ = std::fs::remove_file(ready);

    // Everything below forks, so it must happen before any thread or async
    // runtime exists: only the forking thread survives into the child, and a
    // runtime whose worker threads vanished is unusable.
    let first = unsafe { libc::fork() };
    if first < 0 {
        return Err(io::Error::last_os_error());
    }
    if first > 0 {
        // Original process: reap the intermediate (it exits immediately), then
        // wait for the real daemon to come up before reporting success.
        let mut status = 0;
        unsafe { libc::waitpid(first, &mut status, 0) };
        match wait_ready(ready, pid_file, timeout) {
            Ok(pid) => {
                println!("brain serve: detached, pid {pid}");
                println!("brain serve: log {}", log.display());
                std::process::exit(0);
            }
            Err(e) => {
                eprintln!("brain serve: {e}");
                eprintln!("brain serve: see {}", log.display());
                std::process::exit(1);
            }
        }
    }

    // Intermediate: leave the launching shell's session entirely, then fork
    // again so the daemon is not a session leader and can never acquire a
    // controlling terminal.
    unsafe { libc::setsid() };
    let second = unsafe { libc::fork() };
    if second < 0 {
        unsafe { libc::_exit(1) };
    }
    if second > 0 {
        // `_exit`, not `exit`: this process shares the parent's stdio buffers,
        // and running atexit handlers here would flush them a second time.
        unsafe { libc::_exit(0) };
    }

    redirect_stdio(log)?;
    Ok(())
}

/// Point stdin at `/dev/null` and stdout/stderr at the log.
///
/// Deliberately no `chdir("/")`: the traditional daemon step would break every
/// relative path the caller already resolved (a `--models-dir`, a
/// `--ready-file`, a checkpoint path on the command line).
fn redirect_stdio(log: &Path) -> io::Result<()> {
    let out = OpenOptions::new().create(true).append(true).open(log)?;
    let null = OpenOptions::new().read(true).open("/dev/null")?;
    unsafe {
        libc::dup2(null.as_raw_fd(), libc::STDIN_FILENO);
        libc::dup2(out.as_raw_fd(), libc::STDOUT_FILENO);
        libc::dup2(out.as_raw_fd(), libc::STDERR_FILENO);
    }
    Ok(())
}

/// Wait for the detached server to record its pid and report itself listening.
fn wait_ready(ready: &Path, pid_file: &Path, timeout: Duration) -> Result<i32, String> {
    let deadline = Instant::now() + timeout;
    let mut seen_pid = None;
    while Instant::now() < deadline {
        if seen_pid.is_none() {
            seen_pid = running(pid_file);
        }
        if let Some(pid) = seen_pid {
            // Checked every iteration, not once: `--ready-file` is never
            // created if a surface fails to bind, so without this the wait
            // would run to the full timeout against a process that has
            // already exited.
            if !alive(pid) {
                return Err("the server exited during startup".to_string());
            }
            if ready.exists() {
                return Ok(pid);
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    match seen_pid {
        Some(_) => Err(format!("the server did not become ready within {}s", timeout.as_secs())),
        None => Err(format!("the server did not start within {}s", timeout.as_secs())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(tag: &str) -> PathBuf {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("brain-serve-daemon-{tag}-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The single-instance guarantee. A second claim must be refused while the
    /// first is held, and must report who holds it -- this is what makes
    /// `brain serve -d` fail fast instead of starting a second server that
    /// then loses the race for the bus name.
    #[test]
    fn a_second_claim_is_refused_while_the_first_is_held() {
        let path = tmp("claim").join("serve.pid");
        let first = match claim(&path).unwrap() {
            Claim::Ours(g) => g,
            Claim::HeldBy(_) => panic!("the first claim must succeed"),
        };
        match claim(&path).unwrap() {
            Claim::HeldBy(pid) => assert_eq!(pid, Some(std::process::id() as i32)),
            Claim::Ours(_) => panic!("a second claim must not succeed while the first is held"),
        }
        drop(first);
        // Released with the fd: the next start reclaims it with no cleanup.
        assert!(matches!(claim(&path).unwrap(), Claim::Ours(_)));
    }

    /// A crashed server leaves a pidfile behind. It must not block the next
    /// start, because nothing is going to come along and tidy it up.
    #[test]
    fn a_stale_pidfile_from_a_dead_process_does_not_block_a_claim() {
        let path = tmp("stale").join("serve.pid");
        // pid 1 is alive but is not ours; a pid that cannot exist is the
        // honest "this record is stale" case.
        std::fs::write(&path, "2147483646").unwrap();
        assert!(running(&path).is_none(), "a dead pid must not read as running");
        assert!(matches!(claim(&path).unwrap(), Claim::Ours(_)));
    }

    #[test]
    fn stopping_nothing_is_not_an_error() {
        let path = tmp("stop-none").join("serve.pid");
        assert_eq!(stop(&path, Duration::from_secs(1)).unwrap(), None);
        std::fs::write(&path, "2147483646").unwrap();
        assert_eq!(stop(&path, Duration::from_secs(1)).unwrap(), None, "a stale record is nothing to stop");
        assert!(!path.exists(), "and it is cleaned up");
    }

    /// `stop` must actually end the process, and must not report success for
    /// one it never reached.
    #[test]
    fn stop_terminates_a_real_process() {
        let dir = tmp("stop-real");
        let path = dir.join("serve.pid");
        let mut child = std::process::Command::new("sleep").arg("120").spawn().unwrap();
        let pid = child.id() as i32;
        std::fs::write(&path, pid.to_string()).unwrap();
        assert!(alive(pid));

        assert_eq!(stop(&path, Duration::from_secs(10)).unwrap(), Some(pid));
        // The target here is this test's OWN child, so it lingers as a zombie
        // until reaped and `kill(pid, 0)` keeps succeeding -- see `alive`'s
        // doc. Reaping is what makes the liveness check meaningful again. A
        // real server is never the stopping process's child, so this step has
        // no production counterpart.
        let status = child.wait().unwrap();
        assert!(!alive(pid), "the process must be gone once stop returns");
        assert!(status.code().is_none(), "it was signalled, not a clean exit: {status:?}");
    }
}
