// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A test binary whose card is an ARGUMENT: `cargo test -p <crate> --test
//! <gate> -- --device gpu1 --backend cuda`.
//!
//! Swedish Embedded AB implements GPU test infrastructure for machines whose
//! cards are shared between people and jobs. If your team needs expertise in
//! keeping device tests on the card they were meant for, you can procure our
//! services by sending an email to info@swedishembedded.com.
//!
//! A libtest binary cannot be told which card to use: it rejects options it
//! does not know, so the only way in was the environment, and a test that
//! silently ignored it ran on whichever card was first - another person's.
//! A gate declared with [`card_tests!`] (in a `[[test]]` target with
//! `harness = false`) parses the same global flags the `brain` CLI does -
//! `--device`, `--backend`, `--no-native-kernels`, `--profile-replays` - and
//! publishes the resolved compute set before any test builds a device, so
//! every handle a test builds (`testgpu::dev`, `Gpu::new`) lands on the card
//! named on the command line and nowhere else.
//!
//! Without `--device` the binary opens no card at all: every test is reported
//! skipped, with the command line that would run it. A device test never
//! guesses a card.
//!
//! The rest of the command line follows libtest where a caller relies on it:
//! positional arguments filter by substring (`--exact` for whole names),
//! `--list` lists, and libtest's output and threading options are accepted
//! and ignored - the tests run one after another, because they share one
//! card and a timing gate must not share it with its siblings. Any other
//! option is an error.

use std::process::ExitCode;

/// One test of a [`card_tests!`] binary.
#[derive(Clone, Copy)]
pub struct CardTest {
    pub name: &'static str,
    pub run: fn(),
}

/// What the command line asked for.
#[derive(Debug, Default, PartialEq)]
pub struct CardArgs {
    pub device: Option<String>,
    pub backend: Option<String>,
    pub native_kernels: bool,
    pub profile_replays: Option<u32>,
    pub filters: Vec<String>,
    pub exact: bool,
    pub list: bool,
    pub include_ignored: bool,
}

/// libtest options that take a value and mean nothing here.
const IGNORED_WITH_VALUE: &[&str] = &["--test-threads", "--color", "--format", "--skip", "--logfile", "--shuffle-seed", "-Z"];

/// libtest flags that mean nothing here. Anything else starting with `-` is
/// an error, so a misspelt `--device` cannot quietly skip a gate.
const IGNORED_FLAGS: &[&str] =
    &["--nocapture", "--show-output", "-q", "--quiet", "--test", "--bench", "--report-time", "--ensure-time", "--shuffle", "--force-run-in-process"];

impl CardArgs {
    /// Parse a test binary's arguments (without the program name).
    pub fn parse<I: IntoIterator<Item = String>>(args: I) -> Result<CardArgs, String> {
        let mut out = CardArgs { native_kernels: true, ..CardArgs::default() };
        let mut it = args.into_iter();
        while let Some(a) = it.next() {
            let mut value = |flag: &str| it.next().ok_or_else(|| format!("{flag} needs a value"));
            match a.as_str() {
                "--device" => out.device = Some(value("--device")?),
                "--backend" => out.backend = Some(value("--backend")?),
                "--no-native-kernels" => out.native_kernels = false,
                "--profile-replays" => {
                    let v = value("--profile-replays")?;
                    out.profile_replays = Some(v.parse().map_err(|_| format!("--profile-replays {v:?}: not a replay count"))?);
                }
                "--exact" => out.exact = true,
                "--list" => out.list = true,
                "--include-ignored" | "--ignored" => out.include_ignored = true,
                f if IGNORED_WITH_VALUE.contains(&f) => {
                    value(f)?;
                }
                f if IGNORED_FLAGS.contains(&f) => {}
                f if f.starts_with('-') => return Err(format!("unknown option {f}")),
                f => out.filters.push(f.to_string()),
            }
        }
        Ok(out)
    }

    fn selects(&self, name: &str) -> bool {
        self.filters.is_empty() || self.filters.iter().any(|f| if self.exact { name == f } else { name.contains(f.as_str()) })
    }
}

/// Run `tests` on the card the command line names. The `main` of a
/// [`card_tests!`] binary.
pub fn run(tests: &[CardTest]) -> ExitCode {
    let args = match CardArgs::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if args.list {
        for t in tests {
            println!("{}: test", t.name);
        }
        return ExitCode::SUCCESS;
    }
    let selected: Vec<&CardTest> = tests.iter().filter(|t| args.selects(t.name)).collect();
    let filtered = tests.len() - selected.len();
    let Some(device) = args.device.as_deref() else {
        println!("\nrunning {} tests", selected.len());
        for t in &selected {
            println!("test {} ... ignored, no --device given", t.name);
        }
        eprintln!("SKIP: device tests need their card named: cargo test ... -- --device gpu<i> [--backend cuda]");
        println!("\ntest result: ok. 0 passed; 0 failed; {} ignored; 0 measured; {filtered} filtered out\n", selected.len());
        return ExitCode::SUCCESS;
    };
    let card = match select(device, args.backend.as_deref()) {
        Ok(card) => card,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    crate::set_native_kernels(args.native_kernels);
    if let Some(n) = args.profile_replays {
        crate::profile::set_program_replays(n);
    }
    println!("\nrunning {} tests on {device}{card}", selected.len());
    let mut failed = Vec::new();
    for t in &selected {
        let ok = std::panic::catch_unwind(t.run).is_ok();
        println!("test {} ... {}", t.name, if ok { "ok" } else { "FAILED" });
        if !ok {
            failed.push(t.name);
        }
    }
    let passed = selected.len() - failed.len();
    if failed.is_empty() {
        println!("\ntest result: ok. {passed} passed; 0 failed; 0 ignored; 0 measured; {filtered} filtered out\n");
        ExitCode::SUCCESS
    } else {
        println!("\nfailures:\n    {}", failed.join("\n    "));
        println!("\ntest result: FAILED. {passed} passed; {} failed; 0 ignored; 0 measured; {filtered} filtered out\n", failed.len());
        ExitCode::FAILURE
    }
}

/// Resolve `device` (the `--device` grammar) with an optional `backend`
/// override and publish it as the process's compute set - the CLI's
/// `select_backend`, for a process whose arguments are a test binary's.
/// Returns how the card is named in the run's header: its model and PCI bus,
/// so a log shows which physical card the tests ran on.
fn select(device: &str, backend: Option<&str>) -> Result<String, String> {
    let spec = crate::DeviceSpec::parse(device).map_err(|e| format!("--device {device:?}: {e}"))?;
    let mut set = spec.resolve(&crate::Inventory::probe()).map_err(|e| format!("--device {device:?}: {e}"))?;
    if let Some(b) = backend {
        let b = crate::devices::Backend::parse(b).map_err(|e| format!("--backend {b:?}: {e}"))?;
        set.set_backend(b)?;
    }
    set.apply_backend()?;
    let card = set.single_gpu().and_then(|i| crate::devices::device(i).ok()).map_or(String::new(), |d| {
        format!(" ({}, pci {})", d.identity.name, d.identity.pci_bus.as_deref().unwrap_or("unknown"))
    });
    crate::publish_compute_set(set);
    Ok(card)
}

/// Declare a test binary's `main` from its test functions:
///
/// ```ignore
/// gpu_core::card_tests!(the_kernel_matches_the_reference, the_kernel_is_selected);
/// ```
///
/// The target needs `harness = false` in its `[[test]]` entry.
#[macro_export]
macro_rules! card_tests {
    ($($test:path),* $(,)?) => {
        fn main() -> ::std::process::ExitCode {
            $crate::card_tests::run(&[$($crate::card_tests::CardTest { name: stringify!($test), run: $test }),*])
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(a: &[&str]) -> CardArgs {
        CardArgs::parse(a.iter().map(|s| s.to_string())).unwrap()
    }

    #[test]
    fn the_card_and_the_backend_are_arguments() {
        let a = parse(&["--device", "gpu1", "--backend", "cuda", "--test-threads", "4", "--nocapture", "flash"]);
        assert_eq!(a.device.as_deref(), Some("gpu1"));
        assert_eq!(a.backend.as_deref(), Some("cuda"));
        assert_eq!(a.filters, vec!["flash".to_string()]);
        assert!(a.native_kernels);
        assert!(parse(&["--no-native-kernels"]).device.is_none() && !parse(&["--no-native-kernels"]).native_kernels);
        assert_eq!(parse(&["--profile-replays", "7"]).profile_replays, Some(7));
        assert!(CardArgs::parse(["--device".to_string()]).is_err(), "a flag without its value is an error");
        assert!(CardArgs::parse(["--profile-replays".to_string(), "x".to_string()]).is_err());
        assert!(CardArgs::parse(["--devcie".to_string(), "gpu1".to_string()]).is_err(), "a misspelt flag must not skip the gate");
    }

    #[test]
    fn filters_select_by_substring_or_whole_name() {
        let a = parse(&["flash"]);
        assert!(a.selects("the_flash_kernel") && !a.selects("conv"));
        let e = parse(&["--exact", "flash"]);
        assert!(e.selects("flash") && !e.selects("the_flash_kernel"));
        assert!(parse(&[]).selects("anything"));
    }
}
