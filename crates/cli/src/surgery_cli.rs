// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain speech-surgery` - a Qwen3-TTS checkpoint made physically smaller by
//! a surgery plan, over [`brain::SpeechSurgery`].
//!
//! Swedish Embedded AB implements solutions for shrinking trained speech
//! synthesizers into smaller models that still load in their upstream
//! tooling, for its clients. If your team needs expertise in compressing
//! text-to-speech models then you can procure our services by sending an
//! email to info@swedishembedded.com.

const USAGE: &str = "\
brain speech-surgery BASE --calibration SHARD --plan FILE --out DIR [options]

  Calibrate the Qwen3-TTS release directory BASE on the speech shard SHARD,
  apply the surgery plan FILE (a JSON array of edits: trim_vocabulary,
  fold_embedding, select_channels, rotate_stream, remove_layers), measure each
  edit's loss and KL divergence from the unedited predictions, and write the
  narrower checkpoint in BASE's upstream layout to DIR/checkpoint, with
  DIR/surgery.json (the manifest and measurements) and
  DIR/calibration.safetensors.

  --held-out SHARD       measure on this shard (default: the calibration shard).
  --profile              measure every edit alone on BASE; write no checkpoint.
  --corpus FILE          text, one utterance per line, whose tokens a
                         vocabulary trim keeps beside the shards' own.
  --probe FILE           text whose encodings the trim's report compares.
  --records N            calibration records (default 64; 0: all).
  --held-out-records N   held-out records measured on (default 64; 0: all).
  --batch N              records per calibration batch (default 8).
  --sample-rows N        rows per stream point and batch the statistics take
                         (default 512).
  --exact                exact fp32 products instead of tensor cores.
  --id ID                the model id the checkpoint's manifest registers.
";

#[derive(Debug, Default, PartialEq)]
struct Args {
    base: String,
    calibration: Option<String>,
    plan: Option<String>,
    out: Option<String>,
    held_out: Option<String>,
    profile: bool,
    corpus: Option<String>,
    probe: Option<String>,
    records: Option<usize>,
    held_out_records: Option<usize>,
    batch: Option<usize>,
    sample_rows: Option<usize>,
    exact: bool,
    id: Option<String>,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut i = 0;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let mut value = || -> Result<String, String> {
            i += 1;
            argv.get(i).cloned().ok_or_else(|| format!("brain speech-surgery: {flag} needs a value"))
        };
        let number = |v: String| v.parse::<usize>().map_err(|_| format!("brain speech-surgery: {flag} wants a number, got '{v}'"));
        match flag {
            "-h" | "--help" => return Err(String::new()),
            "--calibration" => a.calibration = Some(value()?),
            "--plan" => a.plan = Some(value()?),
            "--out" => a.out = Some(value()?),
            "--held-out" => a.held_out = Some(value()?),
            "--profile" => a.profile = true,
            "--corpus" => a.corpus = Some(value()?),
            "--probe" => a.probe = Some(value()?),
            "--records" => a.records = Some(number(value()?)?),
            "--held-out-records" => a.held_out_records = Some(number(value()?)?),
            "--batch" => a.batch = Some(number(value()?)?),
            "--sample-rows" => a.sample_rows = Some(number(value()?)?),
            "--exact" => a.exact = true,
            "--id" => a.id = Some(value()?),
            other if other.starts_with('-') => return Err(format!("brain speech-surgery: unknown option '{other}'")),
            other if a.base.is_empty() => a.base = other.to_string(),
            other => return Err(format!("brain speech-surgery: unexpected argument '{other}'")),
        }
        i += 1;
    }
    if a.base.is_empty() {
        return Err("brain speech-surgery: no base given".into());
    }
    for (what, v) in [("--calibration", &a.calibration), ("--plan", &a.plan), ("--out", &a.out)] {
        if v.is_none() {
            return Err(format!("brain speech-surgery: {what} is required"));
        }
    }
    Ok(a)
}

/// The lines of a text file that hold something.
fn lines(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("brain speech-surgery: {path}: {e}"))?;
    Ok(text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect())
}

fn surgery(a: &Args) -> Result<brain::SpeechSurgery, String> {
    let plan = std::fs::read_to_string(a.plan.as_deref().unwrap_or_default()).map_err(|e| format!("brain speech-surgery: --plan: {e}"))?;
    let mut s = brain::SpeechSurgery::new(&a.base, a.calibration.clone().unwrap_or_default()).plan_json(&plan).map_err(|e| e.to_string())?.profile(a.profile).exact(a.exact);
    s = s.out_dir(a.out.clone().unwrap_or_default());
    if let Some(h) = &a.held_out {
        s = s.held_out(h);
    }
    if let Some(c) = &a.corpus {
        s = s.corpus(lines(c)?);
    }
    if let Some(p) = &a.probe {
        s = s.probe(lines(p)?);
    }
    for (set, v) in [(brain::SpeechSurgery::records as fn(_, _) -> _, a.records), (brain::SpeechSurgery::held_out_records, a.held_out_records), (brain::SpeechSurgery::batch, a.batch), (brain::SpeechSurgery::sample_rows, a.sample_rows)] {
        if let Some(n) = v {
            s = set(s, n);
        }
    }
    if let Some(id) = &a.id {
        s = s.id(id.clone());
    }
    Ok(s)
}

pub fn run_speech_surgery(argv: &[String]) -> ! {
    let s = match parse(argv).and_then(|a| surgery(&a)) {
        Ok(s) => s,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("{msg}\n");
            }
            print!("{USAGE}");
            std::process::exit(if msg.is_empty() { 0 } else { 2 });
        }
    };
    let outcome = match s.run_with(|stage| eprintln!("{stage}")) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("brain speech-surgery: {e}");
            std::process::exit(1);
        }
    };
    let b = &outcome.baseline;
    println!("baseline: loss {:.5} (talker {:.5}, code predictor {:.5})", b.loss, b.talker_ce, b.code_predictor_ce);
    for e in &outcome.edits {
        println!("{}: params {} -> {}, loss {:.5}, kl {:.5} (talker {:.5}, code predictor {:.5})", serde_json::to_string(&e.edit).unwrap_or_default(), e.params_before, e.params_after, e.loss, e.kl, e.talker_kl, e.code_predictor_kl);
    }
    if let Some(v) = &outcome.vocabulary {
        println!("vocabulary: {} of {} ids kept, {} of {} merges; corpus texts changed {} of {}, probe texts changed {} of {}", v.kept, v.of, v.merges_kept, v.merges_of, v.corpus_changed, v.corpus_texts, v.probe_changed, v.probe_texts);
    }
    if let Some((n, differ)) = outcome.shard_texts {
        println!("calibration shard: {differ} of {n} utterance texts hold other ids than the tokenizer files encode them to");
    }
    println!("parameters {} -> {}", outcome.params_before, outcome.params_after);
    if let Some(c) = &outcome.checkpoint {
        println!("checkpoint: {}", c.display());
    }
    println!("report: {}", outcome.manifest.display());
    std::process::exit(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn the_flags_parse_and_a_missing_required_one_is_refused() {
        let a = parse(&argv("base --calibration c --plan p --out o --profile --records 8 --exact")).unwrap();
        assert_eq!((a.base.as_str(), a.profile, a.records, a.exact), ("base", true, Some(8), true));
        assert!(parse(&argv("base --plan p --out o")).unwrap_err().contains("--calibration"));
        assert!(parse(&argv("base --calibration c --plan p --out o --records x")).unwrap_err().contains("number"));
        assert!(parse(&argv("base --calibration c --plan p --out o --what")).unwrap_err().contains("unknown option"));
    }
}
