// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `--device` and `--backend`: which compute a run is allowed to schedule on.
//!
//! **Nothing here reads the environment.** A selection brain acts on comes
//! from what the caller typed, so a run is reproducible from its own command
//! line and a shell that was exported into three weeks ago cannot change what
//! a benchmark measures.

use gpu_core::devices::{Backend, ComputeSet, DeviceSpec, Inventory};

use crate::{Args, Options};

/// A parsed hardware selection: what to schedule on, and which backend to
/// reach it through.
#[derive(Clone, Debug, Default)]
pub struct Hardware {
    /// Empty (the default) means every device the machine has.
    pub device: DeviceSpec,
    /// Unset means the backend the resolved device implies.
    pub backend: Option<Backend>,
}

impl Options for Hardware {
    /// Take `--device` and `--backend` out of `args`.
    ///
    /// A bad value is an error rather than a fallback to "everything": the
    /// caller asked for specific hardware, and running the whole job somewhere
    /// else because of a typo is worse than not running it.
    fn take(args: &mut Args) -> Result<Hardware, String> {
        let device = match args.take_str("--device") {
            Some(s) => DeviceSpec::parse(&s).map_err(|e| format!("--device: {e}"))?,
            None => DeviceSpec::default(),
        };
        let backend = match args.take_str("--backend") {
            Some(s) => Some(Backend::parse(&s).map_err(|e| format!("--backend: {e}"))?),
            None => None,
        };
        Ok(Hardware { device, backend })
    }

    fn help() -> &'static str {
        "  --device SPEC       cpu | gpu | npu | gpu0 | cpu0-7 | gpu,cpu   [all present]
  --backend NAME      wgpu | vulkan | cuda | cpu                  [device implies]"
    }
}

impl Hardware {
    /// Resolve against this machine and make it this process's ambient
    /// selection, so every model built afterwards lands where it was asked to.
    ///
    /// The backend override is applied to the resolved set BEFORE it is
    /// published, because a published set and a flag that arrives after it are
    /// two different answers to the same question.
    pub fn apply(&self) -> Result<ComputeSet, String> {
        let probe = Inventory::probe();
        let mut set = self.device.resolve(&probe)?;
        if let Some(b) = self.backend {
            set.backend = b;
        }
        set.apply()?;
        gpu_core::publish_compute_set(set.clone());
        Ok(set)
    }

    /// One line naming what was selected, for a run's own log. A benchmark
    /// that does not record which hardware produced it is a number without a
    /// unit.
    pub fn describe(&self) -> String {
        let dev = if self.device.is_all() { "every device" } else { &self.device.source };
        match self.backend {
            Some(b) => format!("{dev} via {b:?}"),
            None => dev.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_are_taken_out_of_the_line_and_the_rest_survives() {
        let argv: Vec<String> =
            ["--steps", "7", "--device", "cpu", "--backend", "vulkan", "--window"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        let mut args = Args::new(&argv);
        let hw = Hardware::take(&mut args).expect("parses");
        assert_eq!(hw.device.source, "cpu");
        assert_eq!(hw.backend, Some(Backend::Vulkan));
        // The application's own flags are untouched, and `finish` has nothing
        // to complain about - which is the whole reason this consumes rather
        // than scans.
        assert_eq!(args.usize_or("--steps", 0), 7);
        assert!(args.take_flag("--window"));
        args.finish();
    }

    #[test]
    fn a_flag_nobody_claimed_is_not_silently_dropped() {
        // The same failure as a typo'd --device and just as quiet: the run
        // completes, prints numbers, and the numbers are for a configuration
        // nobody asked for.
        let argv: Vec<String> =
            ["--device", "cpu", "--warmup-episodes", "16"].iter().map(|s| s.to_string()).collect();
        let mut args = Args::new(&argv);
        Hardware::take(&mut args).expect("parses");
        assert_eq!(args.leftovers(), vec!["--warmup-episodes", "16"]);
    }

    #[test]
    fn a_typo_is_an_error_not_a_silent_fallback() {
        // Running the whole job on every device because `--device gpu0` was
        // typed `--device gpu:0` is worse than not running it: the numbers
        // come out plausible and describe the wrong hardware.
        let argv: Vec<String> = ["--device", "gpu:0"].iter().map(|s| s.to_string()).collect();
        let mut args = Args::new(&argv);
        let e = Hardware::take(&mut args).expect_err("must not accept");
        assert!(e.starts_with("--device:"), "{e}");
    }

    #[test]
    fn no_selection_means_every_device() {
        let argv: Vec<String> = Vec::new();
        let mut args = Args::new(&argv);
        let hw = Hardware::take(&mut args).expect("parses");
        assert!(hw.device.is_all());
        assert_eq!(hw.describe(), "every device");
    }
}
