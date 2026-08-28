// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain pull <model>` - fetch a model's official weights into the model
//! store, out loud.
//!
//! The verb is a front door, not a second fetcher: it parses what the user
//! typed ([`brain_modelstore::refurl`]), resolves the store directory through
//! the one resolver every other surface uses ([`crate::model_dir::resolve`]),
//! and then drives exactly the plan/execute/finish sequence the auto-fetch
//! path already runs ([`crate::supply::execute_plan`]). What `brain pull`
//! adds is the reporting, and one deliberate choice about where it goes.
//!
//! # Two progress modes, chosen by the stream they are written to
//!
//! Progress is written to **stdout**, and the mode is chosen by whether
//! stdout is a terminal. Deciding on one stream and writing to another is how
//! you end up redrawing an ANSI bar into a log file, so the decision and the
//! destination are the same stream by construction. `brain pull` produces no
//! other machine-readable stdout, and a user who pipes it is capturing the
//! progress log itself - that is what the sparse mode is FOR. Diagnostics and
//! errors still go to stderr. (This differs on purpose from `wan`/`ltxv`,
//! whose progress goes to stderr because their stdout carries a report.)
//!
//! * **Terminal**: an apt-style bar redrawn in place with `\r` - fraction
//!   complete, bytes, throughput and ETA, never scrolling.
//! * **Pipe**: ten plain lines for the WHOLE pull. The budget is spent over
//!   the total bytes of every file in the plan, so a six-shard model costs
//!   ten lines, not sixty. Each line is one greppable fact with no carriage
//!   returns and no escapes. A header line and a completion line bracket
//!   those ten.
//!
//! Swedish Embedded AB implements model distribution and weight-management
//! tooling for its clients. If your team needs expertise in shipping large
//! model artifacts to edge fleets then you can procure our services by
//! sending an email to info@swedishembedded.com.

use std::io::{IsTerminal, Write};

use std::time::{Duration, Instant};

use brain_modelstore::refurl::parse_pull_arg;
use brain_modelstore::{HfHub, Remaining, Step, Store};

/// How many progress lines a whole piped pull is allowed to spend.
const PIPE_LINES: u32 = 10;
/// Width of the drawn bar, in characters.
const BAR_WIDTH: usize = 32;
/// Shortest gap between two terminal redraws. The transfer reports every
/// chunk, which is far more often than a human eye or a serial console wants.
const REDRAW_EVERY: Duration = Duration::from_millis(100);
/// Throughput is averaged over the most recent samples inside this window, so
/// the number shown is the speed NOW rather than the speed since the start -
/// which is what a user watching a stalled transfer needs to see move.
const RATE_WINDOW: Duration = Duration::from_secs(3);

/// Spends a fixed number of report lines over a known total number of bytes.
///
/// Pure and cumulative: [`tick`](LineBudget::tick) takes the bytes transferred
/// across the WHOLE pull, not within one file, and answers with the new
/// percentage threshold that call just crossed. That is the entire difference
/// between "ten lines for this download" and "ten lines per file", so it is a
/// separate testable thing rather than an `if` inside the reporter.
///
/// A crossing that jumps several thresholds at once (one chunk bigger than a
/// tenth of the pull) reports only the highest, so the budget is a ceiling:
/// never more than `lines` lines, sometimes fewer for a very small model.
pub(crate) struct LineBudget {
    total: u64,
    step: u32,
    last: Option<u32>,
}

impl LineBudget {
    pub(crate) fn new(total: u64, lines: u32) -> LineBudget {
        LineBudget { total, step: 100 / lines.max(1), last: None }
    }

    /// `Some(pct)` when `cumulative` just crossed a new threshold. `None`
    /// when it did not, and always `None` when the total is unknown (`0`) -
    /// a percentage of nothing is not a number, and must never be a divide.
    pub(crate) fn tick(&mut self, cumulative: u64) -> Option<u32> {
        if self.total == 0 || self.step == 0 {
            return None;
        }
        let pct = (cumulative.min(self.total) * 100 / self.total) as u32;
        let bucket = (pct / self.step) * self.step;
        if bucket == 0 || self.last.is_some_and(|l| bucket <= l) {
            return None;
        }
        self.last = Some(bucket);
        Some(bucket)
    }
}

/// Which of the two reporting shapes to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Terminal,
    Pipe,
}

impl Mode {
    /// The mode is a function of one thing: whether the stream being written
    /// to is a terminal.
    pub(crate) fn of(is_tty: bool) -> Mode {
        if is_tty {
            Mode::Terminal
        } else {
            Mode::Pipe
        }
    }
}

/// Renders a pull's progress onto one stream.
///
/// Takes the stream as a `&mut dyn Write` rather than reaching for stdout, so
/// both modes are testable byte-for-byte without a pty and without capturing
/// process output.
pub(crate) struct Reporter<'a> {
    mode: Mode,
    out: &'a mut dyn Write,
    label: String,
    remaining: Remaining,
    budget: LineBudget,
    /// Bytes belonging to files already finished in this pull.
    done_before: u64,
    /// The file in flight, and how much of it has arrived.
    current: (String, u64),
    started: Instant,
    /// `(when, cumulative)` samples inside [`RATE_WINDOW`], oldest first.
    samples: std::collections::VecDeque<(Instant, u64)>,
    last_draw: Option<Instant>,
    /// Whether a bar has been drawn, so [`finish`](Reporter::finish) knows if
    /// it owes the terminal a newline.
    drew_bar: bool,
}

impl<'a> Reporter<'a> {
    pub(crate) fn new(mode: Mode, out: &'a mut dyn Write, label: impl Into<String>, remaining: Remaining) -> Reporter<'a> {
        let now = Instant::now();
        Reporter {
            mode,
            out,
            label: label.into(),
            remaining,
            budget: LineBudget::new(remaining.bytes, PIPE_LINES),
            done_before: 0,
            current: (String::new(), 0),
            started: now,
            samples: std::collections::VecDeque::from([(now, 0)]),
            last_draw: None,
            drew_bar: false,
        }
    }

    /// What is about to happen, before a single byte moves - the same courtesy
    /// `apt` extends with "Need to get ...". One line in both modes.
    pub(crate) fn header(&mut self) {
        let size = if self.remaining.sizes_known {
            human_bytes(self.remaining.bytes)
        } else {
            format!("at least {}", human_bytes(self.remaining.bytes))
        };
        let plural = if self.remaining.files == 1 { "file" } else { "files" };
        let _ = writeln!(self.out, "brain pull {}: need {} in {} {plural}", self.label, size, self.remaining.files);
        let _ = self.out.flush();
    }

    /// One transfer report: `file` has `got` of its `total` bytes.
    pub(crate) fn on_bytes(&mut self, file: &str, got: u64, _total: Option<u64>) {
        if file != self.current.0 {
            self.done_before += self.current.1;
            self.current = (file.to_string(), 0);
        }
        self.current.1 = got;
        let cumulative = self.done_before + got;
        match self.mode {
            Mode::Pipe => {
                if let Some(pct) = self.budget.tick(cumulative) {
                    let line = self.status(cumulative, pct);
                    let _ = writeln!(self.out, "{line}");
                    let _ = self.out.flush();
                }
            }
            Mode::Terminal => {
                let now = Instant::now();
                let complete = self.remaining.bytes > 0 && cumulative >= self.remaining.bytes;
                let due = self.last_draw.is_none_or(|t| now.duration_since(t) >= REDRAW_EVERY);
                if due || complete {
                    self.last_draw = Some(now);
                    let bar = self.bar(cumulative);
                    // Pad so a shorter redraw fully erases a longer one; the
                    // house style for in-place CLI progress in this repo.
                    let _ = write!(self.out, "\r{bar}          ");
                    let _ = self.out.flush();
                    self.drew_bar = true;
                }
            }
        }
    }

    /// Close the report -- ending the redrawn line if there is one -- and hand
    /// back what moved and how long it took, so the caller states the outcome
    /// ONCE, with the destination it only learns after the finish step ran.
    /// Two closing lines (one from here, one from the caller) would say the
    /// same thing twice and spend the budget the piped mode is trying to keep.
    pub(crate) fn finish(&mut self) -> (u64, f64) {
        if self.drew_bar {
            let _ = writeln!(self.out);
            let _ = self.out.flush();
        }
        (self.done_before + self.current.1, self.started.elapsed().as_secs_f64())
    }

    /// Bytes per second over the recent window, `None` until there is enough
    /// history to divide by.
    fn rate(&mut self, cumulative: u64) -> Option<f64> {
        let now = Instant::now();
        self.samples.push_back((now, cumulative));
        while self.samples.len() > 2 && now.duration_since(self.samples[1].0) > RATE_WINDOW {
            self.samples.pop_front();
        }
        let (t0, b0) = *self.samples.front()?;
        let dt = now.duration_since(t0).as_secs_f64();
        (dt > 0.05).then(|| (cumulative.saturating_sub(b0)) as f64 / dt)
    }

    /// The one-line pipe status for a crossed threshold.
    fn status(&mut self, cumulative: u64, pct: u32) -> String {
        let rate = self.rate(cumulative);
        format!(
            "brain pull {}: {pct}% {}/{} {} eta {}",
            self.label,
            human_bytes(cumulative),
            human_bytes(self.remaining.bytes),
            human_rate(rate),
            human_eta(self.remaining.bytes.saturating_sub(cumulative), rate)
        )
    }

    /// The one-line terminal bar.
    fn bar(&mut self, cumulative: u64) -> String {
        let rate = self.rate(cumulative);
        let left = self.remaining.bytes.saturating_sub(cumulative);
        render_bar(&self.label, cumulative, self.remaining.bytes, rate, human_eta(left, rate), BAR_WIDTH)
    }
}

/// The apt-style bar, as a pure function of what it shows. Separated from the
/// reporter so the layout is testable without a transfer or a clock.
///
/// With no known total there is no fraction, no bar and no ETA to draw - only
/// what has arrived and how fast - and that is said rather than faked with a
/// bar stuck at zero.
pub(crate) fn render_bar(label: &str, done: u64, total: u64, rate: Option<f64>, eta: String, width: usize) -> String {
    if total == 0 {
        return format!("{label}  {}  {}", human_bytes(done), human_rate(rate));
    }
    let done = done.min(total);
    let filled = ((done as u128 * width as u128) / total.max(1) as u128) as usize;
    let mut track = String::with_capacity(width);
    for i in 0..width {
        track.push(if i < filled.saturating_sub(1) {
            '='
        } else if i < filled {
            '>'
        } else {
            ' '
        });
    }
    let pct = done * 100 / total.max(1);
    format!("{label}  [{track}] {pct:>3}%  {}/{}  {}  eta {eta}", human_bytes(done), human_bytes(total), human_rate(rate))
}

/// Binary units, matching how brain states every other size (`--limit-vram-total`).
pub(crate) fn human_bytes(n: u64) -> String {
    const UNITS: [(&str, u64); 4] = [("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10), ("B", 1)];
    for (name, scale) in UNITS {
        if n >= scale {
            return if scale == 1 { format!("{n} B") } else { format!("{:.1} {name}", n as f64 / scale as f64) };
        }
    }
    "0 B".to_string()
}

/// A transfer rate, or `?` when there is not yet enough history to know one.
pub(crate) fn human_rate(rate: Option<f64>) -> String {
    match rate {
        Some(r) if r >= 1.0 => format!("{}/s", human_bytes(r as u64)),
        _ => "?/s".to_string(),
    }
}

/// How much longer, from what is left and how fast it is arriving.
pub(crate) fn human_eta(bytes_left: u64, rate: Option<f64>) -> String {
    match rate {
        Some(r) if r >= 1.0 => human_secs(bytes_left as f64 / r),
        _ => "?".to_string(),
    }
}

/// A duration a human reads at a glance: `41s`, `7m12s`, `3h04m`.
pub(crate) fn human_secs(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    match s {
        0..=59 => format!("{s}s"),
        60..=3599 => format!("{}m{:02}s", s / 60, s % 60),
        _ => format!("{}h{:02}m", s / 3600, (s % 3600) / 60),
    }
}

const USAGE: &str = "\
usage: brain pull <model> [--brain-data-dir DIR]

Fetch a model's official weights into brain's model store and make them
servable. <model> is the canonical reference or a HuggingFace URL - the repo
page, a branch view, or one file's page:

  brain pull Qwen/Qwen3-0.6B
  brain pull https://huggingface.co/Qwen/Qwen3-0.6B
  brain pull https://huggingface.co/Qwen/Qwen3-0.6B/tree/main

A file URL pulls exactly that ONE file, whatever its extension, from whatever
revision the URL names - nothing is inferred, because the file is named:

  brain pull https://huggingface.co/unsloth/FLUX.2-klein-9B-GGUF/blob/main/flux-2-klein-9b-Q8_0.gguf

A GGUF repo publishes many quantizations of one model, and only ONE is ever
fetched. Name it with the reference grammar's own quantization suffix, or
name none and let brain pick the highest-fidelity one the repo offers (which
it prints):

  brain pull unsloth/FLUX.2-klein-9B-GGUF-Q4_K_M
  brain pull unsloth/FLUX.2-klein-9B-GGUF

Progress goes to stdout: an in-place bar with throughput and ETA on a
terminal, ten plain lines for the whole pull when piped.

Re-running a pull is cheap for what already landed: a file already complete in
the store is not fetched again. Resume is per FILE, not per byte - a transfer
interrupted part-way through a file restarts that file from the beginning. For
a single-file GGUF that is the whole transfer.

--brain-data-dir DIR   brain's data root; models land in <DIR>/models.
                       Default ~/.local/share/brain. This is a GLOBAL option
                       (valid on any subcommand) and outranks BRAIN_MODELS_DIR.
";

/// `brain pull <model>`. Returns the process exit code.
pub fn run_pull(args: &[String]) -> i32 {
    let mut model: Option<&str> = None;
    for a in args {
        match a.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return 0;
            }
            flag if flag.starts_with('-') => {
                eprintln!("brain pull: unknown flag {flag:?}\n{USAGE}");
                return 2;
            }
            positional if model.is_none() => model = Some(positional),
            extra => {
                eprintln!("brain pull: unexpected extra argument {extra:?} -- pull takes one model\n{USAGE}");
                return 2;
            }
        }
    }
    let Some(model) = model else {
        eprint!("{USAGE}");
        return 2;
    };

    let target = match parse_pull_arg(model) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("brain pull: {e}");
            return 2;
        }
    };
    let reference = target.reference.clone();
    let Some(root) = crate::model_dir::resolve(None) else {
        eprintln!("brain pull: no models directory (pass --brain-data-dir, or set BRAIN_MODELS_DIR or HOME)");
        return 2;
    };
    let store = Store::new(root);
    let hub = HfHub::new();

    // One argument, two plans: a URL that named a file asks for exactly that
    // artifact, anything else asks for the repo. Both honour the revision the
    // argument named.
    let built = match target.artifact.as_deref() {
        Some(file) => brain_modelstore::plan_file(&reference, file, target.revision.as_deref(), &hub),
        None => brain_modelstore::plan_at(&reference, target.revision.as_deref(), &store, &hub),
    };
    let plan = match built {
        Ok(p) => p,
        Err(e) => {
            eprintln!("brain pull: {e}");
            return 1;
        }
    };
    let dir = store.repo_dir(&reference.base());
    if plan.steps == [Step::Serve] {
        println!("brain pull {reference}: already complete in {}", dir.display());
        return 0;
    }
    // A choice made for the user is a choice said out loud. The planner puts
    // what it resolved on the plan's reference, so this fires exactly when
    // the repo offered several interchangeable artifacts and the argument
    // named none of them.
    if reference.quant().is_none() && target.artifact.is_none() {
        if let Some(q) = plan.reference.quant() {
            println!("brain pull {reference}: no quantization named, selected {q} (the highest-fidelity one this repo offers)");
            println!("brain pull {reference}: pull another as {reference}-<QUANT>, or paste a file's URL to name it exactly");
        }
    }
    let remaining = match brain_modelstore::remaining_download(&store, &hub, &plan) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("brain pull: {reference}: {e}");
            return 1;
        }
    };

    let mode = Mode::of(std::io::stdout().is_terminal());
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let label = plan.reference.to_string();
    let mut reporter = Reporter::new(mode, &mut lock, label.clone(), remaining);
    if remaining.files > 0 {
        reporter.header();
    }
    let outcome = crate::supply::execute_plan_opt(&store, &hub, &plan, &label, &mut |name, got, total| reporter.on_bytes(name, got, total));
    let (moved, secs) = reporter.finish();
    drop(reporter);

    match outcome {
        Ok(local) => {
            // Where it landed. A pull that produced exactly ONE file still
            // there afterwards reports that FILE: it is the path a
            // `--dit`/`--text-encoder` flag gets pointed at, and naming the
            // directory instead would send the user looking. A pull whose
            // artifact was rewritten or deleted by its finish step (the yolo
            // `.pt`), or that produced several files, reports the servable
            // model's directory as before.
            let where_ = match (landed_file(&plan, &dir), local) {
                (Some(f), _) => f,
                (None, Some(l)) => l.dir.display().to_string(),
                (None, None) => dir.display().to_string(),
            };
            println!("brain pull {label}: fetched {} in {} -> {where_}", human_bytes(moved), human_secs(secs));
            0
        }
        Err(e) => {
            eprintln!("brain pull: {e}");
            1
        }
    }
}

/// The path of the single file a plan produced, when it produced exactly one
/// AND that file is still there -- the artifact a file URL named, or the one
/// quantization a GGUF repo resolved to. `None` for a multi-file pull, and
/// for a finish step that consumed its download (`convert_yolo` deletes the
/// `.pt` it rewrote), so this never names a path that is not there.
fn landed_file(plan: &brain_modelstore::Plan, dir: &std::path::Path) -> Option<String> {
    let mut downloads = plan.steps.iter().filter_map(|s| match s {
        Step::Download { dest_name, .. } => Some(dest_name),
        _ => None,
    });
    let dest = match (downloads.next(), downloads.next()) {
        (Some(dest), None) => dir.join(dest),
        _ => return None,
    };
    dest.is_file().then(|| dest.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_modelstore::{FakeHub, Store};
    use brain_modelref::ModelRef;

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("brain-pull-test-{name}"));
        std::fs::remove_dir_all(&dir).ok();
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// The budget arithmetic, directly: ten thresholds over the WHOLE total,
    /// each reported once, in order, and never more than ten.
    #[test]
    fn the_line_budget_spends_ten_thresholds_over_the_total_once_each() {
        let total = 1000u64;
        let mut budget = LineBudget::new(total, 10);
        let mut fired = Vec::new();
        // One byte at a time: the finest possible reporting, which must still
        // cost exactly ten lines.
        for got in 1..=total {
            if let Some(pct) = budget.tick(got) {
                fired.push(pct);
            }
        }
        assert_eq!(fired, (1..=10).map(|i| i * 10).collect::<Vec<u32>>());
    }

    /// A single report bigger than one threshold collapses to ONE line, so
    /// the budget is a ceiling rather than a quota that has to be spent. A
    /// transfer chunk larger than a tenth of a small model must not print the
    /// thresholds it skipped.
    #[test]
    fn a_jump_across_several_thresholds_costs_one_line() {
        let mut budget = LineBudget::new(1000, 10);
        assert_eq!(budget.tick(550), Some(50));
        assert_eq!(budget.tick(1000), Some(100));
        // Nothing after completion, however many more reports arrive.
        assert_eq!(budget.tick(1000), None);
    }

    /// Progress that does not advance past its threshold is silent -- this is
    /// what keeps a piped pull at ten lines rather than one per chunk.
    #[test]
    fn the_line_budget_is_silent_between_thresholds() {
        let mut budget = LineBudget::new(1000, 10);
        assert_eq!(budget.tick(100), Some(10));
        for got in 101..200 {
            assert_eq!(budget.tick(got), None, "{got} is still inside the 10% bucket");
        }
        assert_eq!(budget.tick(200), Some(20));
    }

    /// An unknown total has no percentage. Never a divide, never a fabricated
    /// zero -- the caller falls back to reporting bytes.
    #[test]
    fn an_unknown_total_yields_no_percentage_and_no_divide() {
        let mut budget = LineBudget::new(0, 10);
        for got in [0, 1, 1 << 20, u64::MAX] {
            assert_eq!(budget.tick(got), None);
        }
    }

    /// The bar's layout, as a pure function: the track fills in proportion,
    /// and every field a user watching a slow transfer needs is present.
    #[test]
    fn the_bar_fills_in_proportion_and_shows_every_field() {
        let bar = render_bar("V/R", 250, 1000, Some(1_048_576.0), "12s".to_string(), 20);
        let track: String = bar.chars().skip_while(|c| *c != '[').skip(1).take_while(|c| *c != ']').collect();
        assert_eq!(track.chars().count(), 20, "the track is a fixed width: {bar:?}");
        assert_eq!(track.chars().filter(|c| *c == '=' || *c == '>').count(), 5, "a quarter of twenty is five: {bar:?}");
        assert!(bar.contains(" 25%"), "{bar}");
        assert!(bar.contains("1.0 MiB/s"), "{bar}");
        assert!(bar.contains("eta 12s"), "{bar}");

        // Empty and full are drawn as such, not off by one.
        let empty = render_bar("V/R", 0, 1000, None, "?".to_string(), 20);
        assert!(!empty.contains('='), "an untouched bar has no fill: {empty:?}");
        let full = render_bar("V/R", 1000, 1000, None, "?".to_string(), 20);
        let track: String = full.chars().skip_while(|c| *c != '[').skip(1).take_while(|c| *c != ']').collect();
        assert!(!track.contains(' '), "a finished bar is full: {full:?}");
        assert!(full.contains("100%"));

        // With no known total there is nothing honest to draw, so nothing is
        // drawn -- rather than a bar permanently stuck at zero.
        let unknown = render_bar("V/R", 4096, 0, Some(2048.0), "?".to_string(), 20);
        assert!(!unknown.contains('['), "an unknown total must not fake a bar: {unknown:?}");
        assert!(unknown.contains("4.0 KiB"), "{unknown}");
    }

    /// The human-readable units, including the cases that read wrong if the
    /// thresholds are off: exactly one unit boundary, and just below it.
    #[test]
    fn sizes_rates_and_durations_read_the_way_a_human_expects() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "1.0 KiB");
        assert_eq!(human_bytes(1 << 20), "1.0 MiB");
        assert_eq!(human_bytes((3 << 30) + (512 << 20)), "3.5 GiB");

        assert_eq!(human_rate(None), "?/s");
        assert_eq!(human_rate(Some(0.0)), "?/s");
        assert_eq!(human_rate(Some(2.0 * 1048576.0)), "2.0 MiB/s");

        assert_eq!(human_secs(0.0), "0s");
        assert_eq!(human_secs(59.4), "59s");
        assert_eq!(human_secs(60.0), "1m00s");
        assert_eq!(human_secs(432.0), "7m12s");
        assert_eq!(human_secs(11040.0), "3h04m");

        // An unknown rate cannot produce an ETA out of thin air.
        assert_eq!(human_eta(1 << 30, None), "?");
        assert_eq!(human_eta(1 << 20, Some(1048576.0)), "1s");
    }

    /// A `transformers`-shaped repo with SIX downloadable files: three small
    /// metadata files and three weight shards big enough that a tenth of the
    /// whole pull is many transfer chunks wide.
    fn six_file_hub() -> FakeHub {
        let mut hub = FakeHub::new();
        let shard = vec![7u8; 2 << 20];
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "config.json", br#"{"architectures":["Qwen3ForCausalLM"]}"#.to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "tokenizer.json", b"{}".to_vec());
        hub.add_file("Qwen", "Qwen3-0.6B", "main", "model.safetensors.index.json", b"{}".to_vec());
        for i in 1..=3 {
            hub.add_file("Qwen", "Qwen3-0.6B", "main", &format!("model-0000{i}-of-00003.safetensors"), shard.clone());
        }
        hub
    }

    /// Drive the REAL plan/size/execute path with a reporter writing into a
    /// buffer, and return (captured output, number of files downloaded).
    fn run_capture(dir: &std::path::Path, hub: &FakeHub, mode: Mode) -> (String, usize) {
        let store = Store::new(dir.to_path_buf());
        let reference = ModelRef::parse("Qwen/Qwen3-0.6B").unwrap();
        let plan = brain_modelstore::plan(&reference, &store, hub).unwrap();
        let remaining = brain_modelstore::remaining_download(&store, hub, &plan).unwrap();
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut rep = Reporter::new(mode, &mut buf, "Qwen/Qwen3-0.6B", remaining);
            rep.header();
            brain_modelstore::execute(&store, hub, &plan, &mut |name, got, total| rep.on_bytes(name, got, total)).unwrap();
            let _ = rep.finish();
        }
        (String::from_utf8(buf).unwrap(), remaining.files)
    }

    /// THE gate this whole mode exists for. The budget is spent over the
    /// TOTAL bytes of the pull, so a six-file model still costs ten progress
    /// lines -- not ten per file. A per-file counter would print sixty here
    /// and is the mistake most easily made.
    #[test]
    fn pipe_mode_spends_ten_lines_on_the_whole_pull_not_ten_per_file() {
        let dir = scratch("pipe-budget");
        let hub = six_file_hub();
        let (out, files) = run_capture(&dir, &hub, Mode::Pipe);
        assert_eq!(files, 6, "the fixture must exercise a MULTI-file pull");
        let progress: Vec<&str> = out.lines().filter(|l| l.contains('%')).collect();
        assert_eq!(progress.len(), 10, "expected ten progress lines for the whole pull, got {}:\n{out}", progress.len());
        assert_ne!(progress.len(), 10 * files, "a ten-lines-PER-FILE budget is not what was asked for");
        // Monotone 10..=100, each printed once: proof the percentages are of
        // the whole pull, not of whichever file is in flight.
        let pcts: Vec<u32> = progress.iter().filter_map(|l| l.split('%').next()?.rsplit(|c: char| !c.is_ascii_digit()).next()?.parse().ok()).collect();
        assert_eq!(pcts, (1..=10).map(|i| i * 10).collect::<Vec<u32>>(), "percentages of the whole pull, in order:\n{out}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Pipe output must be greppable: one fact per line, no carriage returns,
    /// no ANSI escapes, and the model named on every line.
    #[test]
    fn pipe_mode_output_is_plain_and_greppable() {
        let dir = scratch("pipe-plain");
        let hub = six_file_hub();
        let (out, _) = run_capture(&dir, &hub, Mode::Pipe);
        assert!(!out.contains('\r'), "pipe mode must not use carriage returns");
        assert!(!out.contains('\x1b'), "pipe mode must not emit ANSI escapes");
        assert!(out.ends_with('\n'));
        for line in out.lines() {
            assert!(line.contains("Qwen/Qwen3-0.6B"), "every line names the model: {line:?}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Terminal mode redraws ONE line in place and shows all three things a
    /// user watching a slow download needs: how far along, how fast, and how
    /// much longer. Asserted structurally: EVERY redraw must land inside a
    /// single physical line, so a bar that emitted a newline per update --
    /// scrolling the terminal, which is exactly what the `\r` is for -- fails
    /// here even though each individual frame would look right.
    #[test]
    fn terminal_mode_redraws_in_place_with_rate_fraction_and_eta() {
        let dir = scratch("term-bar");
        let hub = six_file_hub();
        let (out, _) = run_capture(&dir, &hub, Mode::Terminal);
        let with_cr: Vec<&str> = out.split('\n').filter(|l| l.contains('\r')).collect();
        assert_eq!(with_cr.len(), 1, "every redraw must share ONE physical line, found {} lines carrying one:\n{out:?}", with_cr.len());
        let bar_line = with_cr[0];
        assert!(bar_line.matches('\r').count() >= 2, "the bar must actually redraw more than once:\n{out:?}");

        let last = bar_line.rsplit('\r').next().unwrap_or("");
        assert!(last.contains('%'), "fraction complete missing from {last:?}");
        assert!(last.contains("/s"), "throughput missing from {last:?}");
        assert!(last.contains("eta"), "ETA missing from {last:?}");
        assert!(last.contains('[') && last.contains(']'), "no progress bar in {last:?}");
        assert!(last.contains("Qwen/Qwen3-0.6B"), "the bar does not name the model: {last:?}");
        // The final frame is a completed bar, not one frozen mid-transfer.
        assert!(last.contains("100%"), "the last redraw should show a finished pull: {last:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Re-running an interrupted pull skips the whole files already on disk,
    /// and -- the part that matters for progress -- excludes them from the
    /// byte budget, so the bar measures what is LEFT rather than jumping
    /// partway along instantly. Whole files: a file that was only partly
    /// transferred is not on disk at all (it is still a `.part`) and is
    /// re-fetched from the start, so it counts in full here.
    #[test]
    fn a_resumed_pull_sizes_only_what_is_still_missing() {
        let dir = scratch("resume");
        let hub = six_file_hub();
        let store = Store::new(dir.clone());
        let reference = ModelRef::parse("Qwen/Qwen3-0.6B").unwrap();
        let plan = brain_modelstore::plan(&reference, &store, &hub).unwrap();
        let full = brain_modelstore::remaining_download(&store, &hub, &plan).unwrap();
        assert_eq!(full.files, 6);
        assert!(full.sizes_known);

        // Simulate an interrupted pull: one shard landed, plus the `.part`
        // file a killed transfer leaves behind for the next one.
        let repo = store.repo_dir(&reference.base());
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(repo.join("model-00001-of-00003.safetensors"), vec![7u8; 2 << 20]).unwrap();
        std::fs::write(repo.join("model-00002-of-00003.safetensors.part"), vec![7u8; 1 << 20]).unwrap();

        let left = brain_modelstore::remaining_download(&store, &hub, &plan).unwrap();
        assert_eq!(left.files, 5, "the completed shard must not be re-counted");
        assert_eq!(left.bytes, full.bytes - (2 << 20), "the completed shard's bytes must leave the budget");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A model already in the store costs no network calls at all: the plan
    /// is a single `Serve` step, so there is nothing to size and nothing to
    /// download.
    #[test]
    fn an_already_complete_model_plans_no_downloads() {
        let dir = scratch("complete");
        let store = Store::new(dir.clone());
        let reference = ModelRef::parse("Qwen/Qwen3-0.6B").unwrap();
        let repo = store.repo_dir(&reference.base());
        std::fs::create_dir_all(&repo).unwrap();
        let card = checkpoint::st::ModelCard::new("Qwen/Qwen3-0.6B", "qwen");
        checkpoint::st::save_safetensors(repo.join("model.brain.safetensors").to_str().unwrap(), &[("w".to_string(), vec![2], vec![1.0, 2.0])], &serde_json::json!({}), Some(&card))
            .unwrap();

        // An empty FakeHub errors on every call, so reaching a Serve plan and
        // a zero-byte Remaining proves neither touched the network.
        let hub = FakeHub::new();
        let plan = brain_modelstore::plan(&reference, &store, &hub).unwrap();
        assert_eq!(plan.steps, vec![brain_modelstore::Step::Serve]);
        let left = brain_modelstore::remaining_download(&store, &hub, &plan).unwrap();
        assert_eq!(left, brain_modelstore::Remaining { files: 0, bytes: 0, sizes_known: true });
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The mode is a function of one thing only: whether the stream progress
    /// is written to is a terminal.
    #[test]
    fn the_mode_follows_the_stream_it_writes_to() {
        assert_eq!(Mode::of(true), Mode::Terminal);
        assert_eq!(Mode::of(false), Mode::Pipe);
    }
}
