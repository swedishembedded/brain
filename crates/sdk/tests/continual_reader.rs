// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain::ContinualReader`'s record surfaces, against a real run directory.
//!
//! What a reader learned needs a model. What a reader DID is in the run
//! directory it wrote, and the clauses of the acceptance block that are
//! questions about its episodes are answerable from that alone - which is
//! why they are worth having separately, and why these tests need no
//! weights, no device and no network.
//!
//! The property under test is the one the block rests on: a surface that
//! reports what the record says, and declines to invent the rest.
//!
//! Swedish Embedded AB builds the reporting surfaces that let a training run
//! be audited after the fact by someone who was not there when it ran. If
//! your team needs a system whose own record is enough to check it, you can
//! procure our services by sending an email to info@swedishembedded.com.

use std::path::PathBuf;

use brain::{ContinualReader, LedgerFacts};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("brain-continual-reader-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// A run directory with the rows a reader would have written.
fn run_with(dir: &std::path::Path, rows: &[(&str, &str, bool, Option<&str>, usize)]) {
    let cfg = audit::reader::ReaderConfig::default();
    let manifest = audit::run::Manifest { schema: audit::run::SCHEMA, model: "fixture".to_string(), cfg };
    let run = audit::run::Run::create(dir, &manifest).expect("run");
    for (i, (source, stage, promoted, cause, decodes)) in rows.iter().enumerate() {
        run.append(&audit::run::LedgerRow {
            episode: i as u64,
            id: format!("id{i}"),
            source: source.to_string(),
            stage: stage.to_string(),
            promoted: *promoted,
            // The real arm carries exactly what it promoted; these fixtures
            // have no null-gate arm to make the two differ.
            carried: *promoted,
            train_loss: None,
            cause: cause.map(str::to_string),
            audited: 2,
            audit_decodes: *decodes,
            diagnosis: None,
            action: None,
        })
        .expect("append");
    }
}

/// The surface reports what the run's own record says, and a reader that
/// was not there when it ran can check it.
#[test]
fn the_record_surfaces_answer_from_the_ledger_alone() {
    let dir = tmp("facts");
    run_with(
        &dir,
        &[
            ("learn/a.txt", "gate", true, None, 32),
            ("noise/p.txt", "screen", false, Some("no_structure"), 0),
            ("repeat/a.txt", "reach", false, Some("already_known"), 0),
            ("contradict/a.txt", "gate", false, Some("block_regressed"), 32),
        ],
    );

    let reader = ContinualReader::from_pretrained("fixture").run_dir(&dir);
    let facts = reader.ledger_facts().expect("facts");
    assert_eq!(facts, LedgerFacts { episodes: 4, promoted: 1, rejections: 3, rejections_with_cause: 3, eval_decodes: 64 });
    assert!(reader.unexplained_refusals().expect("refusals").is_empty(), "every refusal named its cause");

    let _ = std::fs::remove_dir_all(&dir);
}

/// Clause 1 has to be ACTIONABLE, so an unexplained refusal is reported as
/// the document it refused rather than as a count.
#[test]
fn an_unexplained_refusal_is_reported_as_the_document_it_refused() {
    let dir = tmp("unexplained");
    run_with(&dir, &[("learn/a.txt", "gate", true, None, 0), ("mystery/b.txt", "gate", false, None, 0)]);

    let reader = ContinualReader::from_pretrained("fixture").run_dir(&dir);
    assert_eq!(reader.unexplained_refusals().expect("refusals"), vec!["mystery/b.txt".to_string()]);
    assert_eq!(reader.ledger_facts().expect("facts").rejections_with_cause, 0);

    let _ = std::fs::remove_dir_all(&dir);
}

/// A directory nobody has read into is not a run of zero episodes: saying
/// so would report a clean record for a run that never happened.
#[test]
fn a_directory_that_is_not_a_run_is_refused_rather_than_reported_as_empty() {
    let dir = tmp("absent");
    std::fs::create_dir_all(&dir).expect("mkdir");
    let reader = ContinualReader::from_pretrained("fixture").run_dir(&dir);
    assert!(reader.ledger_facts().is_err(), "an absent run must not read as an empty one");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A run that was stopped and reopened reports the whole run, not the part
/// after the restart - the record is the run, which is the property the
/// whole directory design rests on.
#[test]
fn a_reopened_run_reports_every_episode_it_ever_read() {
    let dir = tmp("resumed");
    run_with(&dir, &[("learn/a.txt", "gate", true, None, 16), ("learn/b.txt", "gate", true, None, 16)]);

    // A second process appending to the same directory.
    let run = audit::run::Run::open(&dir).expect("reopen");
    run.append(&audit::run::LedgerRow {
        episode: 2,
        id: "id2".into(),
        source: "learn/c.txt".into(),
        stage: "gate".into(),
        promoted: false,
        carried: false,
        train_loss: None,
        cause: Some("effect_too_small".into()),
        audited: 2,
        audit_decodes: 16,
        diagnosis: None,
        action: None,
    })
    .expect("append");

    let facts = ContinualReader::from_pretrained("fixture").run_dir(&dir).ledger_facts().expect("facts");
    assert_eq!(facts.episodes, 3, "a resumed run's record is the whole run");
    assert_eq!(facts.promoted, 2);
    assert_eq!(facts.eval_decodes, 48);

    let _ = std::fs::remove_dir_all(&dir);
}
