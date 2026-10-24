// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements distributed training runs that account for
// their own communication cost for its clients. If your team needs expertise
// in distributed-training profiling or data-parallel run accounting, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `--profile-collectives FILE`: a distributed run reports what its own
//! collective cost, one JSON document per rank.
//!
//! The flag is a value on the command line, never an environment variable, so
//! a run is reproducible from what its launcher typed. It requires a job plan:
//! a one-process run has no collective to measure, and the single-machine
//! `distributed_profile` binary already covers world 1.
//!
//! The trainer's caller owns the collective (`FitControl`/`GradSync` are not
//! told about timing), so the verbs wrap the joined transport in
//! [`model::TimedCollective`] themselves and keep the wrapper alive until the
//! run ends. Without the flag nothing is wrapped and no file is written.

use std::path::{Path, PathBuf};
use std::time::Duration;

use model::collective::{CollectiveProfile, Snapshot};

use crate::args::Args;

/// The flag these verbs take.
pub(crate) const FLAG: &str = "--profile-collectives";

/// Take [`FLAG`] off `args`. A request that has no job plan is refused by
/// name instead of producing an empty profile of a one-process run.
pub(crate) fn take_flag(args: &mut Args, distributed: bool) -> Result<Option<PathBuf>, String> {
    match (args.take_str(FLAG), distributed) {
        (None, _) => Ok(None),
        (Some(path), true) => Ok(Some(PathBuf::from(path))),
        (Some(_), false) => Err(format!(
            "{FLAG} FILE needs a distributed run: give it with --job-plan FILE and --job-rank N, \
             which is the collective it would measure (a one-process run has none; the \
             distributed_profile bench covers world 1)"
        )),
    }
}

/// Write `snapshot` as this rank's document beside `base` and say where it
/// went. A requested profile that could not be written is an error, not a
/// zero-exit run with a missing file.
pub(crate) fn write(base: &Path, rank: usize, world: usize, plan: &str, run: Duration, snapshot: &Snapshot) {
    let share = if run.is_zero() { 0.0 } else { snapshot.total.as_secs_f64() / run.as_secs_f64() };
    let document = CollectiveProfile::new(rank, world, plan, run, snapshot.clone());
    match document.write(base) {
        Ok(path) => println!(
            "collective profile: {} ({} ops, {} bytes, {:.3} ms of {:.3} ms, {:.1}%)",
            path.display(),
            snapshot.count,
            snapshot.bytes,
            snapshot.total.as_secs_f64() * 1e3,
            run.as_secs_f64() * 1e3,
            share * 1e2
        ),
        Err(e) => {
            eprintln!("error: writing the collective profile: {e}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(tokens: &[&str]) -> Args {
        Args::new(&tokens.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn a_profile_flag_needs_a_distributed_run_and_is_still_consumed_when_refused() {
        assert_eq!(take_flag(&mut args(&[]), true).unwrap(), None);
        assert_eq!(take_flag(&mut args(&["--profile-collectives", "p.json"]), true).unwrap(), Some(PathBuf::from("p.json")));

        let mut one_process = args(&["--profile-collectives", "p.json"]);
        let err = take_flag(&mut one_process, false).expect_err("a one-process run has nothing to profile");
        assert!(err.contains(FLAG), "the refusal must name the flag: {err}");
        assert!(err.contains("--job-plan"), "and the missing companion: {err}");
        assert!(one_process.leftovers().is_empty(), "the flag and its value must still be consumed: {:?}", one_process.leftovers());
    }
}
