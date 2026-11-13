// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A universe of instruments to fine-tune a forecaster on: a directory of
//! `<TICKER>.csv` files, every one validated before anything is loaded.
//!
//! Swedish Embedded AB implements forecasting pipelines that refuse bad data
//! loudly instead of training on a fraction of it for its clients. If your
//! team needs expertise in time-series data validation then you can procure
//! our services by sending an email to info@swedishembedded.com.

/// The index the universe is drawn FROM, rather than one of its constituents:
/// a file for it is skipped by name (and said so), never trained on.
pub const INDEX_ETF: &str = "QQQ";

/// One CSV the loader refused, and why. `reason` carries the message
/// [`forecast::csv::parse_ohlcv`] produced, so it already names the offending
/// 1-based file line - a universe is hundreds of vendor files, and "something
/// in your data is wrong" is not something a caller can act on.
#[derive(Debug)]
pub struct Rejected {
    pub file: String,
    pub reason: String,
}

/// What a universe directory turned out to hold. Every count the caller needs
/// to state what it trained on is here; nothing is dropped on the floor.
#[derive(Debug, Default)]
pub struct Universe {
    /// The files that passed validation, in file-name order.
    pub series: Vec<forecast::train_data::Series>,
    /// The files that did not, in file-name order.
    pub rejected: Vec<Rejected>,
    /// Files deliberately not treated as instruments (see [`INDEX_ETF`]).
    pub excluded: Vec<String>,
}

/// Load a directory of `<TICKER>.csv` (`Date,open,high,low,close,volume`) into
/// leak-safe-windowing [`forecast::train_data::Series`].
///
/// Every file goes through [`forecast::csv::parse_ohlcv`] - the same structural
/// and semantic validation a held-out `predict` uses - so a ragged row, an
/// unparseable number, a repeated or backwards date, a non-finite value or an
/// impossible bar is REPORTED with its file and line instead of skipped. A
/// fine-tune that quietly trains on a fraction of the data it was pointed at
/// still prints a promotion verdict, and that verdict is then about an
/// experiment nobody chose.
///
/// `min_rows` is the request's own length floor (`context + horizon`): a file
/// too short to yield a single training window contributes nothing to the run,
/// so it is a rejection rather than a silent no-op.
///
/// `Err` is a directory-level failure (unreadable, or holding no CSV at all).
/// Per-file failures come back in [`Universe::rejected`] for [`accept`] to
/// apply the run's policy to.
pub fn load(dir: &str, min_rows: usize) -> Result<Universe, String> {
    let rd = std::fs::read_dir(dir).map_err(|e| format!("read {dir}: {e}"))?;
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for ent in rd {
        let ent = ent.map_err(|e| format!("read {dir}: {e}"))?;
        let path = ent.path();
        if path.extension().is_some_and(|x| x.eq_ignore_ascii_case("csv")) && path.is_file() {
            files.push(path);
        }
    }
    if files.is_empty() {
        return Err(format!("{dir}: no `<TICKER>.csv` files (expected one OHLCV file per instrument)"));
    }
    // `read_dir` yields whatever order the filesystem feels like, which would
    // make the universe - and therefore the window order the fine-tune walks -
    // differ run to run on the same directory. Sort, so a rerun is a rerun.
    files.sort();

    let mut u = Universe::default();
    for path in files {
        let file = path.file_name().unwrap_or(path.as_os_str()).to_string_lossy().to_string();
        let ticker = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        if ticker.eq_ignore_ascii_case(INDEX_ETF) {
            u.excluded.push(file);
            continue;
        }
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) => {
                u.rejected.push(Rejected { file, reason: format!("read failed: {e}") });
                continue;
            }
        };
        let parsed = match forecast::csv::parse_ohlcv(&text) {
            Ok(s) => s,
            Err(e) => {
                u.rejected.push(Rejected { file, reason: e });
                continue;
            }
        };
        if parsed.len() < min_rows {
            u.rejected.push(Rejected {
                file,
                reason: format!("csv: {} data rows is too few for one training window ({min_rows} needed) - shorten context/horizon or supply a longer file", parsed.len()),
            });
            continue;
        }
        u.series.push(forecast::train_data::Series {
            ticker,
            dates: parsed.bars.iter().map(|b| (b.stamp.year, b.stamp.month, b.stamp.day)).collect(),
            ohlcv: parsed.bars.iter().map(|b| b.ohlcv).collect(),
        });
    }
    Ok(u)
}

/// The series of an accepted universe, and what was said about the rest:
/// one line per rejected file, per excluded file and a loaded/rejected total,
/// for the caller to show however it reports.
#[derive(Debug)]
pub struct Accepted {
    pub series: Vec<forecast::train_data::Series>,
    pub notes: Vec<String>,
}

/// Apply the run's strictness policy to a loaded [`Universe`].
///
/// **Strict is the default, and a rejected file fails the whole run.** A
/// fine-tune is an hours-long job whose entire output is one promote/keep
/// verdict about one universe; a universe that quietly lost a third of its
/// names yields a verdict about a different experiment, and nothing in the
/// output distinguishes the two afterwards. Validation costs milliseconds and
/// happens before any weights load, so the refusal is cheap and immediate,
/// while the wasted run it prevents is not.
///
/// `skip_invalid` is the escape hatch for the case where refusing everything
/// is the wrong answer - a several-hundred-name vendor dump with one bad row in
/// one file. It still lists every rejection with its file and line, and both
/// paths report the loaded/rejected split, so the run is never quieter about
/// its data than the data deserves.
pub fn accept(dir: &str, label: &str, u: Universe, skip_invalid: bool) -> Result<Accepted, String> {
    let total = u.series.len() + u.rejected.len();
    let mut notes: Vec<String> = u.rejected.iter().map(|r| format!("{label}: {}: {}", r.file, r.reason)).collect();
    if !u.excluded.is_empty() {
        notes.push(format!("{label}: excluded {} ({INDEX_ETF} is the benchmark index, not one of its constituents)", u.excluded.join(", ")));
    }
    notes.push(format!("{label}: {dir}: {} of {total} series loaded, {} rejected", u.series.len(), u.rejected.len()));
    if !u.rejected.is_empty() && !skip_invalid {
        return Err(format!(
            "{}\n{label}: refusing to run: {} of {total} files in {dir} failed validation (listed above) - fix them, or set skip_invalid to train on the {} that parsed",
            notes.join("\n"),
            u.rejected.len(),
            u.series.len()
        ));
    }
    if u.series.is_empty() {
        return Err(format!("{}\n{label}: {dir}: no usable series", notes.join("\n")));
    }
    Ok(Accepted { series: u.series, notes })
}

/// [`load`] then [`accept`], the one call a fine-tune makes per directory.
pub fn load_accepted(dir: &str, label: &str, min_rows: usize, skip_invalid: bool) -> Result<Accepted, String> {
    let u = load(dir, min_rows).map_err(|e| format!("{label}: {e}"))?;
    accept(dir, label, u, skip_invalid)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n` valid daily bars from 2024-01-01: strictly increasing dates (28 days
    /// a month keeps every one a real calendar date) and OHLC invariants that
    /// hold. Every negative case below is this with exactly one thing broken -
    /// the same shape `forecast::csv`'s own tests use.
    fn good_csv(n: usize) -> String {
        let mut s = String::from("Date,open,high,low,close,volume\n");
        for i in 0..n {
            let (month, day) = (1 + i / 28, 1 + i % 28);
            let o = 100.0 + i as f32;
            s.push_str(&format!("2024-{month:02}-{day:02},{o:.2},{:.2},{:.2},{:.2},1000\n", o + 2.0, o - 2.0, o + 1.0));
        }
        s
    }

    /// A scratch directory holding `files`, removed when the test drops it.
    struct Dir(std::path::PathBuf);

    impl Dir {
        fn new(tag: &str, files: &[(&str, String)]) -> Dir {
            let d = std::env::temp_dir().join(format!("brain-forecastops-universe-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).expect("create scratch dir");
            for (name, text) in files {
                std::fs::write(d.join(name), text).expect("write csv");
            }
            Dir(d)
        }
        fn path(&self) -> String {
            self.0.to_string_lossy().into_owned()
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Load a one-file universe and return the rejection message, asserting the
    /// file was refused rather than silently dropped.
    fn reason(tag: &str, file: &str, text: String, min_rows: usize) -> String {
        let d = Dir::new(tag, &[(file, text)]);
        let u = load(&d.path(), min_rows).expect("the directory itself is readable");
        assert!(u.series.is_empty(), "{file} should not have loaded as a series");
        assert_eq!(u.rejected.len(), 1, "{file} should be rejected exactly once");
        assert_eq!(u.rejected[0].file, file, "the rejection must name the file");
        u.rejected[0].reason.clone()
    }

    #[test]
    fn a_well_formed_universe_loads_every_bar() {
        let d = Dir::new("good", &[("AAPL.csv", good_csv(30)), ("MSFT.csv", good_csv(40))]);
        let u = load(&d.path(), 10).unwrap();
        assert!(u.rejected.is_empty(), "{:?}", u.rejected.iter().map(|r| &r.reason).collect::<Vec<_>>());
        // Sorted by file name, so a rerun walks the universe in the same order.
        assert_eq!(u.series.iter().map(|s| s.ticker.as_str()).collect::<Vec<_>>(), ["AAPL", "MSFT"]);
        assert_eq!(u.series[0].len(), 30);
        assert_eq!(u.series[1].len(), 40);
        assert_eq!(u.series[0].dates[0], (2024, 1, 1));
        assert_eq!(u.series[0].ohlcv[0], [100.0, 102.0, 98.0, 101.0, 1000.0]);
    }

    /// One test per class of defect that used to be `continue`d past. The
    /// assertion is on the message: a rejection that does not say WHICH file and
    /// WHICH line is barely better than the silent skip it replaces.
    #[test]
    fn a_ragged_row_is_refused_with_its_file_and_line() {
        let mut text = good_csv(20);
        text.push_str("2024-01-21,120.00,122.00,118.00\n"); // 4 fields, not 6
        let r = reason("ragged", "AAPL.csv", text, 10);
        assert!(r.contains("line 22") && r.contains("4 fields"), "{r}");
    }

    #[test]
    fn an_unparseable_number_is_refused_with_its_file_and_line() {
        let mut text = good_csv(20);
        text.push_str("2024-01-21,120.00,122.00,118.00,n/a,1000\n");
        let r = reason("unparseable", "MSFT.csv", text, 10);
        assert!(r.contains("line 22") && r.contains("is not a number"), "{r}");
    }

    #[test]
    fn a_non_monotonic_date_is_refused_with_its_file_and_line() {
        let mut text = good_csv(20);
        text.push_str("2024-01-19,120.00,122.00,118.00,121.00,1000\n"); // backwards
        let r = reason("backwards", "NVDA.csv", text, 10);
        assert!(r.contains("line 22") && r.contains("not after the previous"), "{r}");
    }

    #[test]
    fn an_impossible_bar_is_refused_with_its_file_and_line() {
        let mut text = good_csv(20);
        // high below the close: a shape-only parser waves this straight through.
        text.push_str("2024-01-21,120.00,122.00,118.00,130.00,1000\n");
        let r = reason("impossible", "TSLA.csv", text, 10);
        assert!(r.contains("line 22") && r.contains("is below open"), "{r}");
    }

    #[test]
    fn a_file_too_short_for_one_window_is_refused_with_its_count() {
        // 12 rows cannot serve a 10-bar context plus a 5-bar horizon.
        let r = reason("short", "AMD.csv", good_csv(12), 15);
        assert!(r.contains("12 data rows") && r.contains("15 needed"), "{r}");
    }

    #[test]
    fn a_header_only_file_is_refused_rather_than_counted_as_a_series() {
        let r = reason("empty", "INTC.csv", "Date,open,high,low,close,volume\n".into(), 1);
        assert!(r.contains("no data rows"), "{r}");
    }

    #[test]
    fn an_unreadable_or_empty_directory_is_an_error_not_an_empty_universe() {
        let missing = std::env::temp_dir().join(format!("brain-forecastops-universe-absent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&missing);
        let e = load(&missing.to_string_lossy(), 1).unwrap_err();
        assert!(e.contains("read") && e.contains("absent"), "{e}");
        // A directory that exists but holds no CSV is the same mistake with a
        // different spelling, and used to return an empty Vec just as quietly.
        let d = Dir::new("nocsv", &[("README.md", "not data".into())]);
        let e = load(&d.path(), 1).unwrap_err();
        assert!(e.contains("no `<TICKER>.csv` files"), "{e}");
    }

    /// The directory-case decision, gated: a mixed directory FAILS the whole run
    /// by default and only trains on the survivors when the caller explicitly
    /// asks. Either way the counts are reported, and the count is the point.
    #[test]
    fn a_mixed_directory_fails_by_default_and_skips_only_when_asked() {
        let mut bad = good_csv(20);
        bad.push_str("2024-01-21,120.00,122.00,118.00\n");
        let files = [("AAPL.csv", good_csv(30)), ("BAD.csv", bad), ("MSFT.csv", good_csv(30)), ("QQQ.csv", good_csv(30))];
        let d = Dir::new("mixed", &files);

        let u = load(&d.path(), 10).unwrap();
        assert_eq!(u.series.len(), 2, "two good names");
        assert_eq!(u.rejected.len(), 1, "one bad name");
        assert_eq!(u.excluded, ["QQQ.csv"], "the index is excluded by name, and said so");

        // Default: the run is refused, and the message says how many failed, how
        // many survived, and how to proceed anyway.
        let e = accept(&d.path(), "finetune", u, false).unwrap_err();
        assert!(e.contains("1 of 3 files") && e.contains("skip_invalid"), "{e}");

        // Opt-in: the survivors train, and only the survivors.
        let u = load(&d.path(), 10).unwrap();
        let a = accept(&d.path(), "finetune", u, true).unwrap();
        assert_eq!(a.series.iter().map(|s| s.ticker.as_str()).collect::<Vec<_>>(), ["AAPL", "MSFT"]);
        assert!(a.notes.iter().any(|n| n.contains("BAD.csv")), "the rejection is still listed: {:?}", a.notes);

        // ... but skip_invalid never degrades into "trained on nothing".
        let d = Dir::new("allbad", &[("BAD.csv", "Date,open,high,low,close,volume\nnonsense\n".into())]);
        let u = load(&d.path(), 10).unwrap();
        let e = accept(&d.path(), "finetune", u, true).unwrap_err();
        assert!(e.contains("no usable series"), "{e}");
    }
}
