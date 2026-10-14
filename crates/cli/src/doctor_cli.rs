// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements model-serving setups that explain what is
// wrong before a model is loaded, for its clients. If your team needs
// expertise in making local inference deployments diagnosable, you can
// procure our services by sending an email to info@swedishembedded.com.

//! `brain doctor` -- can a model be served here? Takes the readings (the
//! model store, the free disk, each GPU's free memory) and hands them to the
//! checks in [`loader::doctor`]; loads no weights, opens no network and
//! allocates nothing on a card.

use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use loader::doctor::{diagnose, Report, Verdict};

use crate::args::Args;

const USAGE: &str = "usage: brain doctor [--model <vendor>/<repo>] [--models-dir DIR] [--json]";

/// Bytes a non-privileged writer can still create on the filesystem holding
/// `path`, or its nearest existing ancestor; `None` when the system will not
/// say.
fn free_bytes(path: &Path) -> Option<u64> {
    let anchor = path.ancestors().find(|a| a.exists())?;
    let c_path = CString::new(anchor.as_os_str().as_bytes()).ok()?;
    let mut stat = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: `c_path` is a valid NUL-terminated string and `stat` is a
    // writable `statvfs` that the call fully initialises when it returns 0.
    if unsafe { libc::statvfs(c_path.as_ptr(), stat.as_mut_ptr()) } != 0 {
        return None;
    }
    // SAFETY: statvfs returned 0, so the structure is initialised.
    let stat = unsafe { stat.assume_init() };
    // The fields are 32 or 64 bits wide by platform; widen so one expression
    // is right for both.
    let bytes = u128::from(stat.f_bavail).saturating_mul(u128::from(stat.f_frsize));
    Some(u64::try_from(bytes).unwrap_or(u64::MAX))
}

fn render(report: &Report) -> String {
    let mut out = String::new();
    for check in &report.checks {
        let mark = match check.verdict {
            Verdict::Ok => "ok  ",
            Verdict::Warn => "warn",
            Verdict::Fail => "FAIL",
        };
        out.push_str(&format!("{mark} {}: {}\n", check.name, check.detail));
        if let Some(fix) = &check.fix {
            out.push_str(&format!("     fix: {fix}\n"));
        }
    }
    out.push_str(if report.ready() { "ready: brain can serve here\n" } else { "not ready: fix the failed checks above\n" });
    out
}

/// Runs `brain doctor`; the exit status is 1 when a check fails and 2 for a
/// usage error.
pub fn run_doctor(args: &[String]) -> i32 {
    let mut a = Args::new(args);
    let json = a.take_flag("--json");
    let model = a.take_str("--model");
    let models_dir = a.take_str("--models-dir");
    if a.positional().is_some() {
        eprintln!("{USAGE}");
        return 2;
    }
    a.finish();

    let root = loader::doctor::store_root(models_dir.as_deref());
    let free = root.as_deref().and_then(free_bytes);
    let cards = gpu_core::capacity::probe_gpus();
    let report = diagnose(root.as_deref(), model.as_deref(), free, &cards);
    if json {
        println!("{}", serde_json::to_string_pretty(&report.to_json()).unwrap_or_default());
    } else {
        print!("{}", render(&report));
    }
    i32::from(!report.ready())
}

#[cfg(test)]
mod tests {
    use super::*;
    use loader::doctor::{store_check, Report};

    #[test]
    fn free_space_is_read_for_a_path_that_does_not_exist_yet_from_its_nearest_parent() {
        let dir = std::env::temp_dir();
        assert!(free_bytes(&dir.join("brain-doctor-not-made-yet/deeper")).is_some());
    }

    #[test]
    fn a_report_prints_each_fix_under_its_check_and_ends_with_the_verdict() {
        let report = Report { checks: vec![store_check(None)] };
        let text = render(&report);
        assert!(text.contains("FAIL store:") && text.contains("fix: pass --models-dir"), "{text}");
        assert!(text.ends_with("not ready: fix the failed checks above\n"));
    }
}
