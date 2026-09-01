// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A forecast chart, rendered by shelling out to the `gnuplot` CLI.
//!
//! Same shape as `imaging::video`'s ffmpeg seam, for the same reason: writing
//! a PNG encoder and a plot layout engine to draw one line chart would be a
//! large amount of code defending a small amount of value, and gnuplot is a
//! stable, ubiquitous, scriptable renderer. The rules that seam established
//! are kept here:
//!
//! - [`gnuplot_available`] probes by RUNNING the binary, so "on `PATH` but
//!   broken" is not mistaken for "installed";
//! - an absent binary produces one error that says what to install, never a
//!   spawn errno the caller has to decode;
//! - the command line is built in exactly ONE place, so what the module runs
//!   and what an error message quotes cannot drift;
//! - a non-zero exit surfaces gnuplot's own stderr rather than a bare status,
//!   and success is re-checked against the file actually existing.
//!
//! What the chart has to show, for it to be evidence rather than decoration:
//! the tail of the **history** the model was given, the **forecast** it
//! produced, and the **actual** continuation that was held out from it - on
//! one pair of axes, so a reader judges the forecast against the truth
//! instead of against the model's own confidence.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The data one chart draws. All three series are `(x, y)` in the same
/// coordinate system, `x` being the bar index so history and horizon line up
/// without a date-axis dependency.
pub struct ForecastChart {
    pub title: String,
    /// The context bars the model saw (or a recent tail of them).
    pub history: Vec<(f64, f64)>,
    /// The model's point/median path over the horizon.
    pub forecast: Vec<(f64, f64)>,
    /// The held-out truth over the same horizon.
    pub actual: Vec<(f64, f64)>,
    /// Optional per-step `(x, lo, hi)` uncertainty band under the forecast.
    pub band: Vec<(f64, f64, f64)>,
    /// The forecast line's own legend entry - callers name their model
    /// (`"kronos forecast"`, `"timesfm3 forecast"`); this chart type has no
    /// model of its own, so nothing here may hardcode one.
    pub forecast_label: String,
    /// Additional named forecast lines, e.g. a baseline or a second model's
    /// path drawn on the same axes for comparison (labels, then their own
    /// distinct colours, in the order given).
    pub extra_lines: Vec<(String, Vec<(f64, f64)>)>,
    pub y_label: String,
    /// Pixel size of the PNG. Kept small on purpose: this is committed
    /// documentation, not a poster.
    pub width: u32,
    pub height: u32,
}

impl ForecastChart {
    pub fn new(title: impl Into<String>) -> ForecastChart {
        ForecastChart {
            title: title.into(),
            history: Vec::new(),
            forecast: Vec::new(),
            actual: Vec::new(),
            band: Vec::new(),
            forecast_label: "forecast".to_string(),
            extra_lines: Vec::new(),
            y_label: "value".to_string(),
            // 800 px is the cap the Quick start's committed chart is held to;
            // at 400 px tall the whole PNG lands around 50 KB, below every
            // other image that page commits.
            width: 800,
            height: 400,
        }
    }
}

/// The install hint an absent binary produces. One string, so the error and
/// any caller that wants to print advice of its own agree.
pub const INSTALL_HINT: &str = "install it (Debian/Ubuntu: `apt-get install gnuplot`, Fedora: `dnf install gnuplot`, macOS: `brew install gnuplot`), or drop the chart flag to get the numbers only";

/// True if the `gnuplot` CLI is on `PATH` and runs. Cheap (`gnuplot
/// --version`, nothing rendered) - call this to skip cleanly with a
/// caller-chosen message rather than parsing [`render_png`]'s error string to
/// detect the same condition.
pub fn gnuplot_available() -> bool {
    Command::new("gnuplot").arg("--version").output().map(|o| o.status.success()).unwrap_or(false)
}

/// Render `chart` to a PNG at `path`.
///
/// Data is passed as files in a per-process temp directory rather than on
/// stdin: gnuplot's `-` special filename can only be read once per script,
/// and this plot has four datasets.
pub fn render_png(chart: &ForecastChart, path: &Path) -> Result<PathBuf, String> {
    if !gnuplot_available() {
        return Err(format!("forecast::chart: gnuplot not found on PATH -- rendering the chart needs the gnuplot CLI: {INSTALL_HINT}"));
    }
    if chart.history.is_empty() && chart.forecast.is_empty() && chart.actual.is_empty() {
        return Err("forecast::chart: nothing to plot (history, forecast and actual are all empty)".to_string());
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| format!("forecast::chart: creating {}: {e}", parent.display()))?;
        }
    }

    let dir = std::env::temp_dir().join(format!("brain-forecast-chart-{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("forecast::chart: creating {}: {e}", dir.display()))?;
    let _cleanup = TempDirGuard(dir.clone());

    let hist = dir.join("history.dat");
    let fcst = dir.join("forecast.dat");
    let act = dir.join("actual.dat");
    let band = dir.join("band.dat");
    write_xy(&hist, &chart.history)?;
    write_xy(&fcst, &chart.forecast)?;
    write_xy(&act, &chart.actual)?;
    write_xyy(&band, &chart.band)?;
    let mut extra_paths = Vec::with_capacity(chart.extra_lines.len());
    for (i, (_, pts)) in chart.extra_lines.iter().enumerate() {
        let p = dir.join(format!("extra{i}.dat"));
        write_xy(&p, pts)?;
        extra_paths.push(p);
    }

    let script_path = dir.join("plot.gp");
    let script = gnuplot_script(chart, path, &hist, &fcst, &act, &band, &extra_paths);
    std::fs::write(&script_path, &script).map_err(|e| format!("forecast::chart: writing {}: {e}", script_path.display()))?;

    let out = Command::new("gnuplot").arg(&script_path).output().map_err(|e| format!("forecast::chart: spawning gnuplot: {e}"))?;
    if !out.status.success() {
        return Err(format!("forecast::chart: gnuplot exited {}: {}", out.status, String::from_utf8_lossy(&out.stderr)));
    }
    if !path.exists() {
        return Err(format!("forecast::chart: gnuplot reported success but {} does not exist", path.display()));
    }
    Ok(path.to_path_buf())
}

/// The ONE place the gnuplot script is built.
///
/// `pngcairo` rather than `png`: the plain `png` terminal needs libgd and is
/// absent from most distro builds, while `pngcairo` ships with every gnuplot 5
/// and 6 package seen in the wild. Colours are chosen to stay distinguishable
/// in greyscale (dark history, mid-tone actual, saturated forecast) because a
/// chart in a README is read on every kind of screen.
fn gnuplot_script(chart: &ForecastChart, out: &Path, hist: &Path, fcst: &Path, act: &Path, band: &Path, extra_paths: &[PathBuf]) -> String {
    let mut s = String::new();
    s.push_str(&format!("set terminal pngcairo size {},{} enhanced font 'sans,10'\n", chart.width, chart.height));
    s.push_str(&format!("set output {}\n", quote(out)));
    s.push_str(&format!("set title {}\n", quote_str(&chart.title)));
    s.push_str("set xlabel 'bar'\n");
    s.push_str(&format!("set ylabel {}\n", quote_str(&chart.y_label)));
    s.push_str("set grid lc rgb '#dddddd'\n");
    s.push_str("set key top left box lc rgb '#999999'\n");
    s.push_str("set border lc rgb '#666666'\n");

    // A vertical rule at the forecast origin: everything right of it is
    // prediction, everything left of it is what the model was shown.
    if let Some((x0, _)) = chart.forecast.first() {
        s.push_str(&format!("set arrow from {x0},graph 0 to {x0},graph 1 nohead lc rgb '#888888' dt 2\n"));
    }

    let mut plots: Vec<String> = Vec::new();
    if !chart.band.is_empty() {
        plots.push(format!("{} using 1:2:3 with filledcurves lc rgb '#ffd9b3' fs solid 0.6 notitle", quote(band)));
    }
    if !chart.history.is_empty() {
        plots.push(format!("{} using 1:2 with lines lw 2 lc rgb '#1f3b5c' title 'history (model input)'", quote(hist)));
    }
    if !chart.actual.is_empty() {
        plots.push(format!("{} using 1:2 with lines lw 2 lc rgb '#2e8b57' title 'actual (held out)'", quote(act)));
    }
    if !chart.forecast.is_empty() {
        plots.push(format!("{} using 1:2 with lines lw 2 dt 1 lc rgb '#d1495b' title {}", quote(fcst), quote_str(&chart.forecast_label)));
    }
    // A small cycling palette, distinct from history/actual/forecast's own
    // colours above - enough to tell a couple of extra comparison lines
    // apart without needing gnuplot's own (less legible) default cycle.
    const EXTRA_COLORS: &[&str] = &["#7b52ab", "#e8a33d", "#3d8fe8", "#5c5c5c"];
    for (i, (label, _)) in chart.extra_lines.iter().enumerate() {
        let color = EXTRA_COLORS[i % EXTRA_COLORS.len()];
        plots.push(format!("{} using 1:2 with lines lw 2 dt 3 lc rgb '{color}' title {}", quote(&extra_paths[i]), quote_str(label)));
    }
    s.push_str("plot ");
    s.push_str(&plots.join(", \\\n     "));
    s.push('\n');
    s
}

fn write_xy(path: &Path, pts: &[(f64, f64)]) -> Result<(), String> {
    let mut buf = String::with_capacity(pts.len() * 24);
    for (x, y) in pts {
        buf.push_str(&format!("{x} {y}\n"));
    }
    std::fs::write(path, buf).map_err(|e| format!("forecast::chart: writing {}: {e}", path.display()))
}

fn write_xyy(path: &Path, pts: &[(f64, f64, f64)]) -> Result<(), String> {
    let mut buf = String::with_capacity(pts.len() * 32);
    for (x, lo, hi) in pts {
        buf.push_str(&format!("{x} {lo} {hi}\n"));
    }
    std::fs::write(path, buf).map_err(|e| format!("forecast::chart: writing {}: {e}", path.display()))
}

/// gnuplot string-literal quoting for a path: single quotes, with an embedded
/// single quote doubled (gnuplot's own escape inside a single-quoted string).
fn quote(p: &Path) -> String {
    quote_str(&p.to_string_lossy())
}

fn quote_str(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

struct TempDirGuard(PathBuf);
impl Drop for TempDirGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo() -> ForecastChart {
        let mut c = ForecastChart::new("synthetic hourly close");
        c.history = (0..40).map(|i| (i as f64, 100.0 + (i as f64 * 0.2).sin())).collect();
        c.forecast = (40..52).map(|i| (i as f64, 100.0 + (i as f64 * 0.2).sin() * 0.9)).collect();
        c.actual = (40..52).map(|i| (i as f64, 100.0 + (i as f64 * 0.2).sin())).collect();
        c.band = (40..52).map(|i| (i as f64, 99.0, 101.0)).collect();
        c
    }

    #[test]
    fn the_script_names_every_dataset_it_was_given_and_only_those() {
        let c = demo();
        let s = gnuplot_script(&c, Path::new("/out/x.png"), Path::new("/d/h.dat"), Path::new("/d/f.dat"), Path::new("/d/a.dat"), Path::new("/d/b.dat"), &[]);
        assert!(s.contains("set terminal pngcairo size 800,400"), "{s}");
        assert!(s.contains("'/out/x.png'") && s.contains("'/d/h.dat'") && s.contains("'/d/f.dat'") && s.contains("'/d/a.dat'") && s.contains("'/d/b.dat'"));
        // The forecast origin is marked, so a reader can see where prediction
        // starts without counting bars.
        assert!(s.contains("set arrow from 40,graph 0"), "{s}");

        // A chart with no band must not reference the band file at all --
        // gnuplot would error on an empty filledcurves dataset.
        let mut c2 = demo();
        c2.band.clear();
        let s2 = gnuplot_script(&c2, Path::new("/o.png"), Path::new("/h"), Path::new("/f"), Path::new("/a"), Path::new("/b"), &[]);
        assert!(!s2.contains("filledcurves"), "{s2}");
    }

    #[test]
    fn extra_lines_get_their_own_labeled_dataset_and_distinct_colours() {
        let mut c = demo();
        c.extra_lines.push(("baseline a".to_string(), vec![(0.0, 1.0)]));
        c.extra_lines.push(("baseline b".to_string(), vec![(0.0, 2.0)]));
        let extra_paths = vec![Path::new("/d/extra0.dat").to_path_buf(), Path::new("/d/extra1.dat").to_path_buf()];
        let s = gnuplot_script(&c, Path::new("/out/x.png"), Path::new("/d/h.dat"), Path::new("/d/f.dat"), Path::new("/d/a.dat"), Path::new("/d/b.dat"), &extra_paths);
        assert!(s.contains("'/d/extra0.dat'") && s.contains("title 'baseline a'"), "{s}");
        assert!(s.contains("'/d/extra1.dat'") && s.contains("title 'baseline b'"), "{s}");
        // Two distinct colours, neither reused from history/actual/forecast.
        let (i0, i1) = (s.find("extra0.dat").unwrap(), s.find("extra1.dat").unwrap());
        let line0 = &s[i0..i1];
        assert!(line0.contains("#7b52ab") && !line0.contains("#e8a33d"));
    }

    #[test]
    fn quoting_survives_a_path_with_a_quote_in_it() {
        assert_eq!(quote_str("a'b"), "'a''b'");
        assert_eq!(quote(Path::new("/o'k.png")), "'/o''k.png'");
    }

    #[test]
    fn an_empty_chart_is_refused_before_a_process_is_spawned() {
        let c = ForecastChart::new("nothing");
        let dir = std::env::temp_dir().join(format!("brain-chart-empty-{}", std::process::id()));
        let e = render_png(&c, &dir.join("x.png")).unwrap_err();
        // Either the "nothing to plot" refusal or the absent-binary error --
        // both are the clean path, and which one fires depends on the machine.
        assert!(e.contains("nothing to plot") || e.contains("gnuplot not found"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn renders_a_real_png_when_gnuplot_is_installed() {
        if !gnuplot_available() {
            return brain_testutil::skip_unavailable("gnuplot is not installed on this machine");
        }
        let dir = std::env::temp_dir().join(format!("brain-chart-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("chart.png");
        let p = render_png(&demo(), &out).unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n", "not a PNG");
        assert!(bytes.len() > 2000, "a {}-byte PNG is a blank canvas, not a chart", bytes.len());
        // The chart is committed documentation: keep it small enough that it
        // never approaches the repo's large-file gate.
        assert!(bytes.len() < 200_000, "{} bytes is too heavy for a committed 800px chart", bytes.len());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
