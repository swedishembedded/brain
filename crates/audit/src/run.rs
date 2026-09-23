// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The directory IS the reader.
//!
//! A reader is meant to be left running for days and stopped whenever it is
//! convenient, so everything it has learned and everything it knows about
//! what it learned lives in one directory that can be reopened. Nothing is
//! held only in memory, and nothing is held anywhere else.
//!
//! ```text
//! run/
//!   manifest.json   the schema, the model it reads with, and its config
//!   state.json      the probe bank, the audit rotation, the reservoir,
//!                   the promote history, and where the stream got to
//!   ledger.jsonl    one line per episode, append-only
//!   pool/           the adapters, their archive and their index
//! ```
//!
//! **Every write is a temporary file and a rename.** A process killed
//! mid-save leaves the PREVIOUS state intact rather than a half-written one:
//! a reader that could be made unopenable by a power cut is not a reader you
//! can leave running.
//!
//! **The ledger is append-only and its last line may be torn.** One line per
//! episode, written as it happens, so a crash costs at most the episode in
//! progress. A reader of the ledger skips a trailing partial line rather
//! than refusing the whole file, because the alternative is losing a run's
//! entire history to one interrupted write.
//!
//! **There is no best-versus-latest distinction here, and that is not an
//! omission.** A design where training can silently degrade needs to keep
//! the best state it ever reached and roll back to it. Nothing ungated ever
//! lands in this directory: an adapter is written only after the gate
//! promoted it against frozen probes and the earlier episodes' own blocks.
//! The latest state IS the best one the reader has evidence for, and adding
//! a rollback mechanism would be adding a remedy for a failure this design
//! does not have.
//!
//! **The ledger's shape is deliberate, not a dump of internal types.** It is
//! the artefact a person reads and a script greps months later, so it is
//! flat, stable and versioned, and it survives the internals being
//! refactored underneath it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::bank::ProbeSet;
use crate::growth::{Action, Diagnosis, Growth};
use crate::reader::{Outcome, ReaderConfig, Row};
use crate::reservoir::Reservoir;
use crate::schedule::Schedule;
use crate::stream::{Cursor, EpisodeId};

/// Bumped whenever `state.json`'s shape changes. A directory written by a
/// different one is refused by name rather than half-understood.
pub const SCHEMA: u32 = 1;

const MANIFEST: &str = "manifest.json";
const STATE: &str = "state.json";
const LEDGER: &str = "ledger.jsonl";
const POOL: &str = "pool";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub schema: u32,
    /// The model reference the reader reads with, resolved by the engine's
    /// own model handler. Recorded because a run's numbers mean nothing
    /// without it.
    pub model: String,
    pub cfg: ReaderConfig,
}

/// Everything the reader must restore to carry on as if it had not stopped.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReaderState {
    pub schema: u32,
    pub bank: BTreeMap<EpisodeId, ProbeSet>,
    pub schedule: Schedule,
    pub reservoir: Reservoir,
    pub growth: Growth,
    pub episode: u64,
    /// Where the stream had got to, so reading resumes rather than restarts.
    pub cursor: Option<Cursor>,
}

/// One line of the ledger.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LedgerRow {
    pub episode: u64,
    pub id: String,
    pub source: String,
    /// Which filter ended this episode.
    pub stage: String,
    pub promoted: bool,
    /// Why, when there is a why. A rejection with no cause would make the
    /// ledger say less than the run knew.
    pub cause: Option<String>,
    pub audited: usize,
    pub audit_decodes: usize,
    pub diagnosis: Option<String>,
    pub action: Option<String>,
}

impl LedgerRow {
    pub fn of(episode: u64, row: &Row) -> LedgerRow {
        let (cause, diagnosis, action) = match &row.outcome {
            Outcome::Leaked { .. } => (Some("probe_leak".to_string()), None, None),
            Outcome::Decided(v) => (
                v.cause().map(str::to_string),
                row.diagnosis.map(name_diagnosis).map(str::to_string),
                row.action.map(name_action).map(str::to_string),
            ),
        };
        LedgerRow {
            episode,
            id: row.episode.as_str().to_string(),
            source: row.source.to_string_lossy().into_owned(),
            stage: row.outcome.stage().to_string(),
            promoted: row.outcome.promoted(),
            cause,
            audited: row.audited,
            audit_decodes: row.audit_decodes,
            diagnosis,
            action,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RunError {
    #[error("run io at {0}: {1}")]
    Io(PathBuf, String),
    #[error("{0} is not a run directory: no {MANIFEST}")]
    NotARun(PathBuf),
    #[error("{path} was written by schema {found}, this build speaks schema {expected} - the state would be half-understood rather than read")]
    Schema { path: PathBuf, found: u32, expected: u32 },
}

type Result<T> = std::result::Result<T, RunError>;

/// A run directory.
#[derive(Debug, Clone)]
pub struct Run {
    root: PathBuf,
}

impl Run {
    /// Start a run, writing its manifest.
    pub fn create(root: &Path, manifest: &Manifest) -> Result<Run> {
        std::fs::create_dir_all(root.join(POOL)).map_err(|e| RunError::Io(root.to_path_buf(), e.to_string()))?;
        let run = Run { root: root.to_path_buf() };
        run.write_atomic(MANIFEST, &serde_json::to_vec_pretty(manifest).map_err(|e| RunError::Io(root.join(MANIFEST), e.to_string()))?)?;
        Ok(run)
    }

    /// Reopen one. Refuses a directory that is not a run, and one written by
    /// a schema this build does not speak.
    pub fn open(root: &Path) -> Result<Run> {
        let path = root.join(MANIFEST);
        if !path.exists() {
            return Err(RunError::NotARun(root.to_path_buf()));
        }
        let run = Run { root: root.to_path_buf() };
        let m = run.manifest()?;
        if m.schema != SCHEMA {
            return Err(RunError::Schema { path, found: m.schema, expected: SCHEMA });
        }
        Ok(run)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where the adapter pool lives.
    pub fn pool_root(&self) -> PathBuf {
        self.root.join(POOL)
    }

    pub fn manifest(&self) -> Result<Manifest> {
        let path = self.root.join(MANIFEST);
        let raw = std::fs::read(&path).map_err(|e| RunError::Io(path.clone(), e.to_string()))?;
        serde_json::from_slice(&raw).map_err(|e| RunError::Io(path, e.to_string()))
    }

    /// Persist the reader's state, atomically.
    pub fn save_state(&self, state: &ReaderState) -> Result<()> {
        let bytes = serde_json::to_vec(state).map_err(|e| RunError::Io(self.root.join(STATE), e.to_string()))?;
        self.write_atomic(STATE, &bytes)
    }

    /// The state, or `None` for a run that has not saved one yet.
    pub fn load_state(&self) -> Result<Option<ReaderState>> {
        let path = self.root.join(STATE);
        if !path.exists() {
            return Ok(None);
        }
        let raw = std::fs::read(&path).map_err(|e| RunError::Io(path.clone(), e.to_string()))?;
        let state: ReaderState = serde_json::from_slice(&raw).map_err(|e| RunError::Io(path.clone(), e.to_string()))?;
        if state.schema != SCHEMA {
            return Err(RunError::Schema { path, found: state.schema, expected: SCHEMA });
        }
        Ok(Some(state))
    }

    /// Append one episode's row.
    pub fn append(&self, row: &LedgerRow) -> Result<()> {
        use std::io::Write;
        let path = self.root.join(LEDGER);
        let mut line = serde_json::to_vec(row).map_err(|e| RunError::Io(path.clone(), e.to_string()))?;
        line.push(b'\n');
        // A torn previous line would otherwise have this one appended to it,
        // turning one lost row into two. A newline first costs a byte and
        // makes a tear self-healing.
        let torn = std::fs::read(&path).map(|b| !b.is_empty() && !b.ends_with(b"\n")).unwrap_or(false);
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path).map_err(|e| RunError::Io(path.clone(), e.to_string()))?;
        if torn {
            f.write_all(b"\n").map_err(|e| RunError::Io(path.clone(), e.to_string()))?;
        }
        f.write_all(&line).map_err(|e| RunError::Io(path, e.to_string()))
    }

    /// Every complete row. A torn trailing line is skipped rather than
    /// failing the read: see this module's doc.
    pub fn ledger(&self) -> Result<Vec<LedgerRow>> {
        let path = self.root.join(LEDGER);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let raw = std::fs::read_to_string(&path).map_err(|e| RunError::Io(path, e.to_string()))?;
        // A line that does not parse is a line an interrupted write left
        // behind. Skipping it keeps the run's history; refusing the file
        // would throw away every complete row to punish one incomplete one.
        Ok(raw.lines().filter(|l| !l.trim().is_empty()).filter_map(|l| serde_json::from_str(l).ok()).collect())
    }

    /// Write `name` by way of a temporary file and a rename, so an
    /// interrupted write cannot replace a good file with a partial one.
    fn write_atomic(&self, name: &str, bytes: &[u8]) -> Result<()> {
        let tmp = self.root.join(format!("{name}.tmp"));
        std::fs::write(&tmp, bytes).map_err(|e| RunError::Io(tmp.clone(), e.to_string()))?;
        std::fs::rename(&tmp, self.root.join(name)).map_err(|e| RunError::Io(self.root.join(name), e.to_string()))
    }
}

fn name_diagnosis(d: Diagnosis) -> &'static str {
    match d {
        Diagnosis::Healthy => "healthy",
        Diagnosis::Interference => "interference",
        Diagnosis::Saturation => "saturation",
        Diagnosis::Inconclusive => "inconclusive",
    }
}

fn name_action(a: Action) -> &'static str {
    match a {
        Action::Nothing => "nothing",
        Action::RaiseRehearsal { .. } => "raise_rehearsal",
        Action::Grow => "grow",
        Action::Hold { .. } => "hold",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::reservoir::ReservoirConfig;
    use crate::schedule::AuditConfig;
    use crate::triage::{Unstructured, Verdict};

    static N: AtomicUsize = AtomicUsize::new(0);

    struct Dir(PathBuf);

    impl Dir {
        fn new() -> Dir {
            let d = std::env::temp_dir().join(format!("brain-run-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&d);
            Dir(d)
        }
    }

    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn manifest() -> Manifest {
        Manifest { schema: SCHEMA, model: "qwen3:0.6b".to_string(), cfg: ReaderConfig::default() }
    }

    fn state(episode: u64) -> ReaderState {
        let mut schedule = Schedule::new(AuditConfig::default());
        for i in 0..5 {
            schedule.admit(EpisodeId::of(&format!("e{i}")));
        }
        ReaderState {
            schema: SCHEMA,
            bank: BTreeMap::new(),
            schedule,
            reservoir: Reservoir::new(ReservoirConfig::default()),
            growth: Growth::new(Default::default()),
            episode,
            cursor: None,
        }
    }

    fn row(i: u64, stage: &str) -> LedgerRow {
        LedgerRow {
            episode: i,
            id: format!("id{i}"),
            source: "m.txt".to_string(),
            stage: stage.to_string(),
            promoted: stage == "gate",
            cause: None,
            audited: 2,
            audit_decodes: 32,
            diagnosis: None,
            action: None,
        }
    }

    /// Stopping a reader and reopening it has to restore what it knew, not
    /// an approximation of it.
    #[test]
    fn a_run_directory_round_trips_its_state() {
        let d = Dir::new();
        let r = Run::create(&d.0, &manifest()).expect("create");
        assert_eq!(r.manifest().expect("manifest"), manifest());
        assert!(r.load_state().expect("load").is_none(), "a fresh run has no state yet");

        r.save_state(&state(42)).expect("save");
        let back = r.load_state().expect("load").expect("saved");
        assert_eq!(back.episode, 42);
        assert_eq!(back.schedule.len(), 5);

        let reopened = Run::open(&d.0).expect("open");
        assert_eq!(reopened.load_state().expect("load").expect("saved").episode, 42);
    }

    /// The property that makes a reader safe to leave running. A save killed
    /// part way must cost the save, never the run.
    #[test]
    fn an_interrupted_save_leaves_the_previous_state_loadable() {
        let d = Dir::new();
        let r = Run::create(&d.0, &manifest()).expect("create");
        r.save_state(&state(7)).expect("save");

        // What a process killed mid-save leaves behind: a temporary file,
        // never renamed over the real one.
        std::fs::write(d.0.join(format!("{STATE}.tmp")), b"{\"schema\":1,\"bank\":").expect("partial write");

        let back = r.load_state().expect("the previous state must still load").expect("saved");
        assert_eq!(back.episode, 7, "the completed save must be what survives");
    }

    /// A run's whole history must not be lost to one interrupted append.
    #[test]
    fn the_ledger_skips_a_torn_last_line_rather_than_refusing_the_file() {
        let d = Dir::new();
        let r = Run::create(&d.0, &manifest()).expect("create");
        for i in 0..4 {
            r.append(&row(i, "gate")).expect("append");
        }
        assert_eq!(r.ledger().expect("ledger").len(), 4);

        // A crash between writing a line and finishing it.
        let mut raw = std::fs::read_to_string(d.0.join(LEDGER)).expect("read");
        raw.push_str("{\"episode\":4,\"id\":\"id4\",\"sou");
        std::fs::write(d.0.join(LEDGER), raw).expect("write");

        let rows = r.ledger().expect("a torn last line must not fail the read");
        assert_eq!(rows.len(), 4, "the four complete rows must survive");
        assert_eq!(rows[3].episode, 3);

        // And appending after a tear must not compound it.
        r.append(&row(5, "screen")).expect("append");
        let rows = r.ledger().expect("ledger");
        assert_eq!(rows.last().expect("rows").episode, 5);
    }

    /// State written by a different shape must be refused by name, not read
    /// as though the fields it is missing were simply absent.
    #[test]
    fn a_directory_written_by_another_schema_is_refused_and_says_so() {
        let d = Dir::new();
        Run::create(&d.0, &Manifest { schema: SCHEMA + 1, ..manifest() }).expect("create");
        match Run::open(&d.0) {
            Err(RunError::Schema { found, expected, .. }) => {
                assert_eq!(found, SCHEMA + 1);
                assert_eq!(expected, SCHEMA);
            }
            other => panic!("expected a schema refusal, got {}", other.map(|_| "an open run").unwrap_or("another error")),
        }
    }

    /// And a directory that is not a run at all must say that rather than
    /// behaving as an empty one.
    #[test]
    fn a_directory_that_is_not_a_run_is_refused() {
        let d = Dir::new();
        std::fs::create_dir_all(&d.0).expect("mkdir");
        assert!(matches!(Run::open(&d.0), Err(RunError::NotARun(_))));
    }

    /// A ledger row is the artefact someone reads months later, so a
    /// rejection has to carry its reason rather than only its stage.
    #[test]
    fn a_ledger_row_records_why_an_episode_was_refused() {
        let outcome = Outcome::Decided(Verdict::Unstructured(Unstructured::NoStructure { structure: 0.01, floor: 0.15 }));
        let r = Row {
            episode: EpisodeId::of("x"),
            source: PathBuf::from("noise.txt"),
            outcome,
            audited: 0,
            audit_decodes: 0,
            diagnosis: None,
            action: None,
        };
        let line = LedgerRow::of(9, &r);
        assert_eq!(line.stage, "screen");
        assert!(!line.promoted);
        assert_eq!(line.cause.as_deref(), Some("no_structure"));
        assert_eq!(line.source, "noise.txt");

        let leaked = Row { outcome: Outcome::Leaked { line: 12 }, ..r };
        let line = LedgerRow::of(10, &leaked);
        assert_eq!(line.stage, "ingest");
        assert_eq!(line.cause.as_deref(), Some("probe_leak"));
    }
}
