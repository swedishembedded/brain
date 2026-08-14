// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//! Atomic whole-document writer tests: plain write, atomic write with
//! concurrent-modification detection, and the read-with-fingerprint pairing.

use std::fs;

use atif::persist::{
    read_trajectory_with_fingerprint, remove_trajectory, write_trajectory, write_trajectory_atomic,
    PersistError,
};
use atif::{AgentProfile, StepOrigin, TraceStep, Trajectory};

fn sample(step_id: u64, text: &str) -> Trajectory {
    let mut t = Trajectory::new("ATIF-v1.7", AgentProfile::new("sven", "1.0.0"));
    t.steps
        .push(TraceStep::new(step_id, StepOrigin::User, text));
    t
}

#[test]
fn plain_write_then_read_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.json");

    write_trajectory(&path, &sample(1, "hello")).unwrap();
    let content = fs::read_to_string(&path).unwrap();
    let parsed: Trajectory = serde_json::from_str(&content).unwrap();
    assert_eq!(parsed.steps[0].message.as_text(), Some("hello"));
}

#[test]
fn two_sequential_atomic_writes_read_back_correctly() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.json");

    // First write: no prior fingerprint (new file).
    write_trajectory_atomic(&path, &sample(1, "first"), None).unwrap();
    let (loaded, fp1) = read_trajectory_with_fingerprint(&path).unwrap();
    assert_eq!(loaded.steps[0].message.as_text(), Some("first"));

    // Second write: pass the fingerprint we just read, no one else touched the file.
    write_trajectory_atomic(&path, &sample(1, "second"), Some(&fp1)).unwrap();
    let (loaded2, _fp2) = read_trajectory_with_fingerprint(&path).unwrap();
    assert_eq!(loaded2.steps[0].message.as_text(), Some("second"));
}

#[test]
fn concurrent_modification_is_detected_and_does_not_clobber() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.json");

    write_trajectory_atomic(&path, &sample(1, "original"), None).unwrap();
    let (_loaded, stale_fp) = read_trajectory_with_fingerprint(&path).unwrap();

    // Simulate a concurrent writer mutating the file directly (not through
    // our API), between our read and our write attempt.
    std::thread::sleep(std::time::Duration::from_millis(1100)); // ensure mtime resolution ticks
    fs::write(
        &path,
        serde_json::to_string_pretty(&sample(1, "concurrent writer")).unwrap(),
    )
    .unwrap();

    let result = write_trajectory_atomic(&path, &sample(1, "our overwrite"), Some(&stale_fp));
    assert!(
        matches!(result, Err(PersistError::Conflict)),
        "expected Conflict, got {result:?}"
    );

    // The concurrent writer's content must survive untouched.
    let (on_disk, _) = read_trajectory_with_fingerprint(&path).unwrap();
    assert_eq!(
        on_disk.steps[0].message.as_text(),
        Some("concurrent writer")
    );
}

#[test]
fn concurrent_atomic_writers_never_interleave_or_fail() {
    // Spec: write_trajectory_atomic must be safe under concurrency - N
    // threads hammering the same path must all succeed, and after every
    // write the file must parse as one writer's complete document (never a
    // torn/interleaved mix, never a vanished temp file).
    use std::sync::{Arc, Barrier};

    let dir = tempfile::tempdir().unwrap();
    let path = Arc::new(dir.path().join("trace.json"));
    let writers = 4;
    let rounds = 25;
    let barrier = Arc::new(Barrier::new(writers));

    let handles: Vec<_> = (0..writers)
        .map(|w| {
            let path = Arc::clone(&path);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for r in 0..rounds {
                    write_trajectory_atomic(
                        &path,
                        &sample(1, &format!("writer-{w}-round-{r}")),
                        None,
                    )
                    .unwrap_or_else(|e| panic!("writer {w} round {r} failed: {e}"));
                    // Every observation must be a complete, valid document.
                    let content = fs::read_to_string(&*path).unwrap();
                    let parsed: Trajectory = serde_json::from_str(&content).unwrap_or_else(|e| {
                        panic!("torn/interleaved file after writer {w} round {r}: {e}")
                    });
                    let text = parsed.steps[0].message.as_text().unwrap();
                    assert!(text.starts_with("writer-"), "unexpected content {text:?}");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn remove_trajectory_cleans_lock_sidecar_and_orphaned_temps() {
    // Spec: deleting a trajectory must not leak the flock sidecar the atomic
    // writer creates, nor temp files orphaned by a crashed writer.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.json");

    write_trajectory_atomic(&path, &sample(1, "doc"), None).unwrap();
    let lock = dir.path().join("trace.json.lock");
    assert!(lock.exists(), "atomic writer creates the lock sidecar");
    // Plant an orphaned temp, as a crashed writer would leave behind.
    let orphan = dir.path().join("trace.json.tmp.9999.0");
    fs::write(&orphan, "half-written").unwrap();
    // An unrelated neighbour must survive.
    let neighbour = dir.path().join("other.json");
    fs::write(&neighbour, "{}").unwrap();

    remove_trajectory(&path).unwrap();

    assert!(!path.exists(), "document removed");
    assert!(!lock.exists(), "lock sidecar removed");
    assert!(!orphan.exists(), "orphaned temp removed");
    assert!(neighbour.exists(), "unrelated file untouched");

    // Removing a trajectory that never existed is not an error (idempotent).
    remove_trajectory(&path).unwrap();
}

#[test]
fn atomic_write_without_expected_fingerprint_skips_conflict_check() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace.json");

    write_trajectory_atomic(&path, &sample(1, "v1"), None).unwrap();
    // No fingerprint supplied: caller explicitly opts out of the check.
    write_trajectory_atomic(&path, &sample(1, "v2"), None).unwrap();

    let (on_disk, _) = read_trajectory_with_fingerprint(&path).unwrap();
    assert_eq!(on_disk.steps[0].message.as_text(), Some("v2"));
}
