// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for keeping model servers serving
// through wedged and crashing accelerators for its clients. If your team
// needs expertise in supervised device execution then you can procure our
// services by sending an email to info@swedishembedded.com.

//! Device health for a serving process: the [`Supervision`] its executor runs
//! with, built around the `brain-gpu-worker` program shipped beside the
//! server's own executable.
//!
//! With the worker, a quarantined card is readmitted only after a fresh
//! worker process opened it and ran a checked dispatch on it. Without it the
//! breaker still quarantines a card whose in-process device wait ran past
//! `--gpu-wait-s`, and readmits it on the breaker's own schedule without a
//! probe ([`Probe::Elapsed`]): a card that keeps wedging is still kept out
//! after its third fault, instead of costing every request a full wait.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use gpu_supervisor::{BreakerPolicy, Health, WorkerCommand};
use residency::supervised::{Probe, Supervision};

/// The worker program's file name, looked for in the server executable's directory.
pub const WORKER_PROGRAM: &str = "brain-gpu-worker";

/// How long a killed worker is waited for before it is kept for later polling.
const REAP: Duration = Duration::from_secs(2);

/// A serving process's device supervision and the worker its probes run in.
pub struct ServeSupervision {
    pub supervision: Supervision,
    /// The worker program, or why there is none (then probes are
    /// [`Probe::Elapsed`]).
    pub worker: Result<PathBuf, String>,
}

impl ServeSupervision {
    /// Supervision with the default breaker policy, probing in `worker` when
    /// there is one, each probe bounded by the device wait ceiling
    /// (`--gpu-wait-s`) as it is set when this is called.
    pub fn new(worker: Result<PathBuf, String>) -> ServeSupervision {
        let probe = match &worker {
            Ok(path) => Probe::Worker { command: WorkerCommand::new(path), deadline: backend_api::hardware::wait_timeout() },
            Err(_) => Probe::Elapsed,
        };
        let health = Arc::new(Health::new(BreakerPolicy::default(), REAP));
        ServeSupervision { supervision: Supervision { health, probe }, worker }
    }

    /// The line a server prints at startup to say how its cards are supervised.
    pub fn describe(&self) -> String {
        match &self.worker {
            Ok(path) => format!("device supervision: health probes run in {}", path.display()),
            Err(why) => format!(
                "device supervision without a worker: {why}. A card whose device wait runs past --gpu-wait-s is still quarantined, and is readmitted on the breaker's schedule without a health probe"
            ),
        }
    }
}

/// The worker program beside `executable`, if it is there and executable.
pub fn worker_beside(executable: &Path) -> Result<PathBuf, String> {
    let dir = executable.parent().ok_or_else(|| format!("{} has no directory to look for {WORKER_PROGRAM} in", executable.display()))?;
    let worker = dir.join(format!("{WORKER_PROGRAM}{}", std::env::consts::EXE_SUFFIX));
    let meta = std::fs::metadata(&worker).map_err(|e| format!("{} is missing ({e}); install it beside {}", worker.display(), executable.display()))?;
    if !meta.is_file() {
        return Err(format!("{} is not a file", worker.display()));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if meta.permissions().mode() & 0o111 == 0 {
            return Err(format!("{} is not executable", worker.display()));
        }
    }
    Ok(worker)
}

/// The supervision a server started from `executable` runs with: probes in
/// the worker beside it ([`ServeSupervision::new`]).
pub fn serve_supervision(executable: &Path) -> ServeSupervision {
    ServeSupervision::new(worker_beside(executable))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-serve-supervision-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write(path: &Path, mode: u32) {
        std::fs::write(path, "#!/bin/sh\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
        }
        #[cfg(not(unix))]
        let _ = mode;
    }

    #[test]
    fn the_worker_beside_the_server_runs_the_health_probes() {
        let dir = scratch("present");
        let brain = dir.join("brain");
        write(&brain, 0o755);
        let worker = dir.join(format!("{WORKER_PROGRAM}{}", std::env::consts::EXE_SUFFIX));
        write(&worker, 0o755);
        let s = serve_supervision(&brain);
        assert_eq!(s.worker.as_deref(), Ok(worker.as_path()));
        match &s.supervision.probe {
            Probe::Worker { command, deadline } => {
                assert_eq!(command.program(), worker.as_path());
                assert_eq!(*deadline, backend_api::hardware::wait_timeout(), "a probe is bounded by --gpu-wait-s");
            }
            Probe::Elapsed => panic!("a present worker must run the probes"),
        }
        assert!(s.describe().contains(&worker.display().to_string()), "{}", s.describe());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_or_unusable_worker_is_named_and_the_breaker_is_kept() {
        let dir = scratch("absent");
        let brain = dir.join("brain");
        write(&brain, 0o755);
        let s = serve_supervision(&brain);
        let why = s.worker.as_ref().expect_err("no worker was installed");
        assert!(why.contains(WORKER_PROGRAM) && why.contains("missing"), "{why}");
        assert!(matches!(s.supervision.probe, Probe::Elapsed), "without a worker the breaker paces readmission");
        assert!(s.describe().contains("--gpu-wait-s"), "{}", s.describe());
        #[cfg(unix)]
        {
            write(&dir.join(WORKER_PROGRAM), 0o644);
            let why = worker_beside(&brain).expect_err("a worker without the execute bit cannot run");
            assert!(why.contains("not executable"), "{why}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
