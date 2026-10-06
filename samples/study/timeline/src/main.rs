// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Train a timeline model on irregular records with competing outcomes, check
//! it against the risk that generated the data, turn its risk into calibrated
//! intervals, and save it for serving - through the public brain SDK.
//!
//! ```text
//! sample-study-timeline [--out DIR] [--subjects N] [--steps N] [--seed N]
//! ```
//!
//! The data are brain's synthetic population: age-dependent competing deaths
//! driven by two measurements (one detection-limited) and a diagnosis, a
//! non-absorbing onset, and an irrelevant covariate - with the true
//! cumulative incidence of every subject known. The model sees only the
//! records. On subjects it never saw, its ten-year risk of the first cause of
//! death is compared with the truth and with a covariate-blind estimate; the
//! program exits non-zero unless it is closer to the truth and has the lower
//! Brier score.
//!
//! Swedish Embedded AB implements time-to-event prediction from irregular
//! records for its clients. If your team needs expertise in survival
//! modelling, competing risks or calibrated risk intervals, you can procure
//! our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use brain::survival::brier::brier;
use brain::survival::calibration::at_horizon;
use brain::survival::concordance::uno;
use brain::survival::estimate::{aalen_johansen, censoring};
use brain::survival::venn_abers::{merged, VennAbers};
use brain::timeline::synthetic::drifting::{self, Gaps};
use brain::timeline::synthetic::{population, Truth, CODES};
use brain::timeline::{observed, Subject};
use brain::{TimelineModel, TimelineSpec};

const USAGE: &str = "\
usage: sample-study-timeline [--out DIR] [--subjects N] [--steps N] [--seed N]

  --out DIR        where the saved model and a file of test subjects go
                   (default: <tmp>/sample-study-timeline)
  --subjects N     training subjects (default 20000; a fifth as many are held
                   out for early stopping and calibration, a quarter tested)
  --steps N        optimizer steps at most (default 3000)
  --seed N         seed of the weights and batches (default 1)
";

/// The cause the sample reports on, and the horizon, in years.
const CAUSE: &str = "death:a";
const HORIZON: f64 = 10.0;
/// Test subjects written beside the model for the serving sample.
const SERVED_SUBJECTS: usize = 20;
/// The visit model's horizon, in years.
const VISIT_HORIZON: f64 = 10.0;

struct Args(Vec<String>);

impl Args {
    fn take(&mut self, flag: &str) -> Option<String> {
        let i = self.0.iter().position(|a| a == flag)?;
        if i + 1 >= self.0.len() {
            return None;
        }
        self.0.remove(i);
        Some(self.0.remove(i))
    }

    fn parse<T: std::str::FromStr>(&mut self, flag: &str, default: T) -> Result<T, String> {
        self.take(flag)
            .map(|v| v.parse().map_err(|_| format!("{flag} {v:?}: not a valid value")))
            .unwrap_or(Ok(default))
    }
}

fn main() -> ExitCode {
    let mut a = Args(std::env::args().skip(1).collect());
    if a.0.iter().any(|x| x == "--help" || x == "-h") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let parsed = (|| -> Result<(PathBuf, usize, u32, u64), String> {
        let out = a.take("--out").map(PathBuf::from).unwrap_or_else(|| std::env::temp_dir().join("sample-study-timeline"));
        let config = (out, a.parse("--subjects", 20_000)?, a.parse("--steps", 3000)?, a.parse("--seed", 1)?);
        match a.0.first() {
            Some(extra) => Err(format!("unexpected argument {extra:?}")),
            None => Ok(config),
        }
    })();
    let (out, n, steps, seed) = match parsed {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&out, n, steps, seed) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The risk of [`CAUSE`] by [`HORIZON`] per subject, as the model predicts it.
fn risk(model: &TimelineModel, subjects: &[Subject]) -> brain::Result<Vec<f64>> {
    Ok(model.predict(subjects)?.iter().map(|p| p.cif(CAUSE, HORIZON).unwrap_or(f64::NAN)).collect())
}

fn run(out: &Path, n: usize, steps: u32, seed: u64) -> brain::Result<bool> {
    let (train, _) = population(n, 1);
    let (held, _) = population(n / 5, 2);
    let (test, truth): (Vec<Subject>, Vec<Truth>) = population(n / 4, 3);
    let deaths = ["death:a", "death:b"];
    let spec = TimelineSpec::new(CODES, deaths)
        .knots(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0])
        .max_tokens(8)
        .steps(steps)
        .seed(seed);
    println!("training on {} subjects, early-stopping on {}", train.len(), held.len());
    let (model, report) = TimelineModel::train(&train, &held, &spec)?;
    println!("stopped after {} steps; held-out event NLL {:.4}", report.steps, report.held_out_event_nll);

    // The cause of interest first, the other death competing.
    let obs = observed(&test, &deaths);
    let g = censoring(&obs);
    let predicted = risk(&model, &test)?;
    let true_risk: Vec<f64> = truth.iter().map(|t| t.cif(0, HORIZON)).collect();
    let blind_value = aalen_johansen(&observed(&train, &deaths), 0).at(HORIZON);
    let blind = vec![blind_value; test.len()];
    let error = |r: &[f64]| r.iter().zip(&true_risk).map(|(a, b)| (a - b).abs()).sum::<f64>() / r.len() as f64;
    let score = |r: &[f64]| brier(r, &obs, 0, HORIZON, &g);
    let (model_error, blind_error) = (error(&predicted), error(&blind));
    println!("\n{CAUSE} by {HORIZON} years on {} unseen subjects:", test.len());
    println!("  mean |risk - true risk|  model {model_error:.4}  covariate-blind {blind_error:.4}");
    let briers = (score(&predicted), score(&true_risk), score(&blind));
    if let (Some(m), Some(t), Some(b)) = briers {
        println!("  IPCW Brier score         model {m:.4}  true risk {t:.4}  covariate-blind {b:.4}");
    }
    if let Some(c) = uno(&predicted, &obs, 0, HORIZON, &g) {
        println!("  Uno concordance          model {c:.3}");
    }

    // Intervals calibrated on the early-stopping subjects, never trained on.
    let held_obs = observed(&held, &deaths);
    let va = VennAbers::at_horizon(&risk(&model, &held)?, &held_obs, 0, HORIZON, &censoring(&held_obs));
    let intervals: Vec<(f64, f64)> = predicted.iter().map(|&r| va.interval(r, 1.0)).collect();
    let merged_risk: Vec<f64> = intervals.iter().copied().map(merged).collect();
    let mut widths: Vec<f64> = intervals.iter().map(|(p0, p1)| p1 - p0).collect();
    widths.sort_by(f64::total_cmp);
    let (raw, cal) = (at_horizon(&predicted, &obs, 0, HORIZON, &g, 10), at_horizon(&merged_risk, &obs, 0, HORIZON, &g, 10));
    println!("  calibration slope        raw {:.3}  Venn-Abers {:.3}", raw.slope, cal.slope);
    println!("  interval width           median {:.4}  90th percentile {:.4}", widths[widths.len() / 2], widths[widths.len() * 9 / 10]);
    println!("  first subject            risk {:.4}  interval [{:.4}, {:.4}]  truth {:.4}", predicted[0], intervals[0].0, intervals[0].1, true_risk[0]);

    let dir = out.join("model");
    model.save(&dir)?;
    let lines: String = test.iter().take(SERVED_SUBJECTS).map(|s| serde_json::to_string(s).map(|l| l + "\n")).collect::<Result<_, _>>().map_err(|e| brain::Error::Backend(e.to_string()))?;
    std::fs::write(out.join("subjects.jsonl"), lines)?;
    println!("\nsaved the model to {} and {SERVED_SUBJECTS} test subjects to {}", dir.display(), out.join("subjects.jsonl").display());

    let better = model_error < blind_error && matches!(briers, (Some(m), _, Some(b)) if m < b);
    if !better {
        eprintln!("the model did not beat the covariate-blind estimate");
    }
    write_datasets(out)?;
    let visits_better = visits_model(out, n, steps, seed)?;
    Ok(better && visits_better)
}

/// Four disjoint `timeline-v1` files of the synthetic population in
/// `<out>/data`: training, held-out (early stopping), validation (calibration)
/// and test subjects, what the `brain horizon` train, eval and calibrate
/// actions read (`samples/shell/timeline/lifecycle`).
fn write_datasets(out: &Path) -> brain::Result<()> {
    let dir = out.join("data");
    std::fs::create_dir_all(&dir)?;
    for (name, n, seed) in [("train", 4000, 31), ("held-out", 1000, 32), ("validation", 2000, 33), ("test", 2000, 34)] {
        let lines: String = population(n, seed).0.iter().map(|s| serde_json::to_string(s).map(|l| l + "\n")).collect::<Result<_, _>>().map_err(|e| brain::Error::Backend(e.to_string()))?;
        std::fs::write(dir.join(format!("{name}.jsonl")), lines)?;
    }
    println!("\nwrote train, held-out, validation and test subjects to {}", dir.display());
    Ok(())
}

/// A second model for histories that GROW: trained on records whose risk
/// factor is measured at several visits and drifts between them
/// (`synthetic::drifting`, whose best possible prediction is known exactly),
/// reading the most recent four visits one by one. A patient history with a
/// checkup appended is a record like these; the first model, trained on one
/// visit per subject, is outside its support there and says so. Saved to
/// `<out>/model-visits` with twenty test subjects in `<out>/subjects-visits.jsonl`.
fn visits_model(out: &Path, n: usize, steps: u32, seed: u64) -> brain::Result<bool> {
    let gaps = Gaps { last: (0.0, 2.0), between: (0.5, 2.0), visits: (1, 4) };
    let (train, _) = drifting::population(n / 2, 4, &gaps, 10.0);
    let (held, _) = drifting::population(n / 10, 5, &gaps, 10.0);
    let (test, best) = drifting::population(n / 5, 6, &gaps, 10.0);
    let spec = TimelineSpec::new([drifting::CODE], [drifting::CODE])
        .knots(vec![0.0, 2.0, 5.0, 10.0])
        .max_tokens(8)
        .visits(4)
        .steps(steps)
        .seed(seed);
    println!("\ntraining the visit model on {} subjects with up to four visits each", train.len());
    let (model, report) = TimelineModel::train(&train, &held, &spec)?;
    let predicted: Vec<f64> = model.predict(&test)?.iter().map(|p| p.cif(drifting::CODE, VISIT_HORIZON).unwrap_or(f64::NAN)).collect();
    let true_risk: Vec<f64> = best.iter().map(|b| b.cif(VISIT_HORIZON)).collect();
    let blind_value = aalen_johansen(&observed(&train, &[drifting::CODE]), 0).at(VISIT_HORIZON);
    let error = |r: &dyn Fn(usize) -> f64| (0..test.len()).map(|i| (r(i) - true_risk[i]).abs()).sum::<f64>() / test.len() as f64;
    let (model_error, blind_error) = (error(&|i| predicted[i]), error(&|_| blind_value));
    println!("visit model stopped after {} steps; {} by {VISIT_HORIZON} years on {} unseen subjects:", report.steps, drifting::CODE, test.len());
    println!("  mean |risk - best possible risk|  model {model_error:.4}  covariate-blind {blind_error:.4}");

    let dir = out.join("model-visits");
    model.save(&dir)?;
    let lines: String = test.iter().take(SERVED_SUBJECTS).map(|s| serde_json::to_string(s).map(|l| l + "\n")).collect::<Result<_, _>>().map_err(|e| brain::Error::Backend(e.to_string()))?;
    std::fs::write(out.join("subjects-visits.jsonl"), lines)?;
    println!("saved the visit model to {} and {SERVED_SUBJECTS} test subjects to {}", dir.display(), out.join("subjects-visits.jsonl").display());
    let better = model_error < blind_error;
    if !better {
        eprintln!("the visit model did not beat the covariate-blind estimate");
    }
    Ok(better)
}
