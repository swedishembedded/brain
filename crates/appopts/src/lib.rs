// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Command-line options shared by every application built on brain.
//!
//! ## The problem this exists for
//!
//! Every sample, every test harness and the `brain` binary itself need the
//! same options. Which device to run on. Where the weights are. How many
//! iterations to train for. Whether to open a window. Written once per
//! application, they drift: `--flag` at the end of a line means one thing in
//! one binary and another in the next, an unknown flag is reported here and
//! silently ignored there, and `--device gpu:0` is an error in one and "run on
//! everything" in another.
//!
//! ## The shape
//!
//! Options come in **groups**, and a group is reusable on its own. [`Args`] is
//! a consuming parser: a group [`Options::take`]s its own flags out of the
//! line and what remains belongs to somebody else. An application composes the
//! groups it needs, in any order, and calls [`Args::finish`] to have anything
//! left over reported rather than ignored.
//!
//! ```no_run
//! use appopts::{Args, Hardware, Options};
//! let argv: Vec<String> = std::env::args().skip(1).collect();
//! let mut args = Args::new(&argv);
//! let hw = Hardware::take(&mut args)?;         // core: --device, --backend
//! let steps = args.usize_or("--steps", 100);   // this application's own
//! args.finish();
//! hw.apply()?;
//! # Ok::<(), String>(())
//! ```
//!
//! **Core groups live here** - the ones that mean the same thing whatever the
//! application is, which today is [`Hardware`]. **Surface groups live with
//! their surface**, in `brain::options`: the training knobs next to the
//! decision pipeline they configure, the window flags next to the viewport.
//! That split is deliberate. A group that needs a surface's types belongs
//! behind that surface's feature, or this crate becomes a second place where
//! every part of brain is described.
//!
//! ## Nothing here reads the environment
//!
//! A selection brain acts on comes from what the caller typed. A run is then
//! reproducible from its own command line, and a shell that was exported into
//! three weeks ago cannot quietly change what a benchmark measures.
//!
//! Swedish Embedded AB builds the tooling layer around products - one
//! implementation of the things every binary in a system needs, rather than
//! one per binary. If your team needs that, you can procure our services by
//! sending an email to info@swedishembedded.com.

pub mod args;
pub mod hardware;

pub use args::Args;
pub use hardware::Hardware;

/// A reusable group of command-line options.
///
/// Implemented by anything that owns a set of flags several applications want:
/// the hardware selection here, a training spec in `brain::options`, a
/// sample's own window flags. The two halves are deliberately paired -
/// whatever [`Options::take`] consumes, [`Options::help`] documents, so a flag
/// cannot be added without appearing in `--help`.
pub trait Options: Sized {
    /// Remove this group's flags from `args` and build the group.
    ///
    /// Returns an error rather than a default on a bad value: a caller who
    /// typed a flag meant it, and silently running somewhere else - or for a
    /// different number of steps - produces a result that looks fine and
    /// answers a different question.
    fn take(args: &mut Args) -> Result<Self, String>;

    /// The `--help` lines for this group, one per flag, so every application
    /// spells them the same way.
    fn help() -> &'static str;
}

/// Join the help of several groups, in the order given.
///
/// ```
/// # use appopts::{Hardware, Options};
/// let text = appopts::help_of(&[Hardware::help()]);
/// assert!(text.contains("--device"));
/// ```
pub fn help_of(groups: &[&str]) -> String {
    groups.join("\n")
}
