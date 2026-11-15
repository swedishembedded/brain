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
//!
//! Model code runs in a worker for the families an operator names
//! (`--worker-hosted`, [`worker_hosting`]): the worker is the server's own
//! executable re-executed in its internal worker mode ([`WORKER_MODE`]), so
//! one binary holds every model, and health probes stay in the small
//! `brain-gpu-worker`, which opens a card with nothing but one backend and a
//! probe kernel and so tests the card, not a model build.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use gpu_supervisor::{BreakerPolicy, Health, WorkerCommand};
use residency::supervised::{Deadlines, Probe, Supervision, WorkerHosting};
use residency::Device;

/// The worker program's file name, looked for in the server executable's directory.
pub const WORKER_PROGRAM: &str = "brain-gpu-worker";

/// How long a killed worker is waited for before it is kept for later polling.
const REAP: Duration = Duration::from_secs(2);

/// The argument that starts `brain` in its internal worker mode. Reached only
/// by a supervisor's spawn ([`worker_hosting`]); never listed in help.
pub const WORKER_MODE: &str = "__worker";

/// `--worker-deadline-s` unset: how long a worker may take to load its model,
/// and how long a job may go without reporting progress, before it counts as
/// wedged.
pub const DEFAULT_WORKER_DEADLINE: Duration = Duration::from_secs(900);

/// Worker hosting for the model `families` an operator named, each worker
/// being `executable` re-executed as `<executable> <device_flags> __worker`:
/// the server's own device flags (`--backend`, `--gpu-wait-s`, ...), so a
/// worker selects and bounds its card as the server does. `deadline` bounds
/// a worker's load and the silence of each of its jobs. A family no worker can host is
/// refused, naming the ones it can.
pub fn worker_hosting(executable: &Path, device_flags: &[String], families: &[String], deadline: Duration, health: Arc<Health<Device>>) -> Result<WorkerHosting, String> {
    let known: Vec<&str> = catalog::hosted_families().iter().map(|f| f.name).collect();
    let mut chosen: Vec<String> = Vec::new();
    for family in families {
        if !known.contains(&family.as_str()) {
            return Err(format!("--worker-hosted: no model family {family:?} can be hosted in a worker yet (the ones that can: {})", known.join(", ")));
        }
        if !chosen.contains(family) {
            chosen.push(family.clone());
        }
    }
    let command = device_flags.iter().fold(WorkerCommand::new(executable), |c, flag| c.arg(flag)).arg(WORKER_MODE);
    Ok(WorkerHosting::new(command, health, Deadlines { open: deadline, run: deadline, reap: REAP }, chosen))
}

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
    fn worker_hosting_re_executes_the_server_with_its_device_flags_for_the_families_named() {
        let health = Arc::new(Health::new(BreakerPolicy::default(), REAP));
        let flags = ["--backend".to_string(), "cuda".to_string(), "--gpu-wait-s".to_string(), "12".to_string()];
        let families = ["decision".to_string(), "decision".to_string()];
        let hosting = worker_hosting(Path::new("/proc/self/exe"), &flags, &families, Duration::from_secs(7), health.clone()).expect("decision models can be hosted");
        assert!(hosting.hosts("decision"));
        assert!(!hosting.hosts("qwen3"), "only the families named");
        let command = hosting.command();
        assert_eq!(command.program(), Path::new("/proc/self/exe"));
        let args: Vec<&str> = command.args().iter().map(|a| a.to_str().unwrap()).collect();
        assert_eq!(args, ["--backend", "cuda", "--gpu-wait-s", "12", WORKER_MODE]);
        let err = worker_hosting(Path::new("/proc/self/exe"), &[], &["qwen3".to_string()], Duration::from_secs(7), health).err().expect("a family no worker hosts is refused");
        assert!(err.contains("qwen3") && err.contains("decision"), "{err}");
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
