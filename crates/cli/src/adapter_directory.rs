// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for refitting a serving model from
// a learning loop without restarting it, for its clients. If your team needs
// expertise in serving many fine-tuned variants of one model while a trainer
// keeps producing new ones, then you can procure our services by sending an
// email to info@swedishembedded.com.

//! `--serve-adapters DIR`: the adapters in a directory are the adapters a
//! running server offers.
//!
//! Every `*.safetensors` file under the directory is a LoRA adapter release.
//! The server keeps its offered set equal to the files that verify against its
//! base: a file that appears is offered as a model of its own under its card
//! id, a file whose bytes change (an atomic rename over it) replaces the
//! adapter under that id with the next policy version, and a file that
//! disappears withdraws its adapter once the requests already accepted for it
//! are done. Nothing restarts and no other adapter or the base is disturbed;
//! each adapter is a model id of its own and they are served in the same
//! batches. This is the directory form of [`LiveAdapters`]; `--watch-adapters`
//! follows a promotion sequence (one adapter, the highest version) instead.
//!
//! The directory is the operator's (whoever can write to it can put an adapter
//! on the server: verification binds an adapter to its base, it does not
//! authenticate its author), but its files are hostile input:
//!
//! * symbolic links, hidden files, other file types and anything not named
//!   `*.safetensors` are ignored, so a link cannot lead the server out of the
//!   directory;
//! * the walk is bounded in depth and in files ([`MAX_DEPTH`], [`MAX_FILES`]);
//! * a file is read once, size-capped, digested and parsed from the same
//!   bytes, and bound to the served base its card names, then verified against
//!   it exactly as `--adapter` is ([`verify_adapter_among`]): a server that
//!   serves several bases offers each adapter on its own base, and an adapter
//!   whose base is not served is refused naming that base;
//! * a file that fails to verify is reported once, per version of the file,
//!   and the adapter it replaces keeps being served;
//! * a card id that another file already serves is refused, naming both files,
//!   so two files can never fight over one model id.
//!
//! A trainer publishes by writing the file aside (a name not ending in
//! `.safetensors`) and renaming it into place: a half-written file that is
//! seen anyway fails to parse and is read again when it changes.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use std::sync::Arc;

use llmserve::adapter_registry::{AdapterHost, LiveAdapters};
use llmserve::adapter_release::{verify_adapter_among, ServedBase};
use residency::Executor;

/// How many directory levels below the root are searched: room for the model
/// store's `<owner>/<name>/<tag>/adapter.brain.safetensors` layout.
pub const MAX_DEPTH: usize = 4;

/// The most adapter files one pass looks at; the rest are ignored, and said so.
pub const MAX_FILES: usize = 1024;

/// What identifies a file's content between passes: size, modification time,
/// and the inode and change time, so a release renamed over a file of the same
/// size within the filesystem's timestamp granularity is still seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stamp {
    len: u64,
    modified: Option<SystemTime>,
    inode: u64,
    changed: (i64, i64),
}

impl Stamp {
    fn of(meta: &std::fs::Metadata) -> Stamp {
        use std::os::unix::fs::MetadataExt;
        Stamp { len: meta.len(), modified: meta.modified().ok(), inode: meta.ino(), changed: (meta.ctime(), meta.ctime_nsec()) }
    }
}

/// What a pass over the directory did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Pass {
    /// Adapters offered or replaced.
    pub offered: usize,
    /// Adapters withdrawn because their file is gone or now names another id.
    pub retired: usize,
    /// Files refused (new or changed since last said).
    pub refused: usize,
}

/// A served base the directory's adapters may ride on, and where its adapters are offered.
pub struct BankHost {
    base: ServedBase,
    live: LiveAdapters,
}

impl BankHost {
    /// The adapters of `resident`'s base, offered on `executor` (the executor `resident` is registered with).
    pub fn new<R: AdapterHost>(resident: Arc<R>, executor: Executor) -> BankHost {
        BankHost { base: resident.served_base(), live: LiveAdapters::new(resident, executor) }
    }
}

/// What the watcher knows about one file.
enum Known {
    /// Offered under `id` on the host at index `host`, read when the file looked like `stamp`.
    Served { id: String, host: usize, stamp: Stamp },
    /// Refused when it looked like `stamp`; read again only when it changes.
    Refused { stamp: Stamp },
}

impl Known {
    fn stamp(&self) -> Stamp {
        match self {
            Known::Served { stamp, .. } | Known::Refused { stamp } => *stamp,
        }
    }
}

/// The directory an adapter bank follows, and what it has offered from it.
pub struct BankDirectory {
    dir: PathBuf,
    hosts: Vec<BankHost>,
    /// The bases of `hosts`, in order, as [`verify_adapter_among`] takes them.
    bases: Vec<ServedBase>,
    known: BTreeMap<PathBuf, Known>,
}

impl BankDirectory {
    /// Follow `dir`, binding every adapter to the base among `hosts` its card names.
    pub fn new(dir: PathBuf, hosts: Vec<BankHost>) -> BankDirectory {
        let bases = hosts.iter().map(|h| h.base.clone()).collect();
        BankDirectory { dir, hosts, bases, known: BTreeMap::new() }
    }


    /// The directory followed.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The card ids offered from this directory now.
    #[cfg(test)]
    fn served_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.known.values().filter_map(|k| if let Known::Served { id, .. } = k { Some(id.clone()) } else { None }).collect();
        ids.sort();
        ids
    }

    /// Bring what the hosts offer in line with the directory as it is now.
    pub fn poll(&mut self) -> Pass {
        let mut pass = Pass::default();
        let files = match scan(&self.dir) {
            Ok(files) => files,
            // A directory that has not been created yet is the state before
            // the first publish, not a fault.
            Err(e) => {
                residency::log::debug(&format!("adapter bank: {}: {e}", self.dir.display()));
                BTreeMap::new()
            }
        };

        for (path, stamp) in &files {
            if self.known.get(path).is_some_and(|k| k.stamp() == *stamp) {
                continue;
            }
            match verify_adapter_among(path, &self.bases) {
                Err(why) => {
                    eprintln!("brain serve: not offering the adapter {}: {why}", path.display());
                    pass.refused += 1;
                    // The release this file used to be keeps being served.
                    match self.known.get_mut(path) {
                        Some(Known::Served { stamp: served_at, .. }) => *served_at = *stamp,
                        _ => {
                            self.known.insert(path.clone(), Known::Refused { stamp: *stamp });
                        }
                    }
                }
                Ok((host, loaded)) => {
                    let release = loaded.release.clone();
                    if let Some(owner) = self.owner_of(&release.id).filter(|owner| *owner != path) {
                        eprintln!("brain serve: not offering the adapter {}: its card id {} is already served from {}; give one of them another tag", path.display(), release.id, owner.display());
                        pass.refused += 1;
                        self.known.entry(path.clone()).and_modify(|k| *k = Known::Refused { stamp: *stamp }).or_insert(Known::Refused { stamp: *stamp });
                        continue;
                    }
                    // The file now rides on another base: its adapter leaves the old one first, since a registry
                    // replaces only its own adapters.
                    let moved = match self.known.get(path) {
                        Some(Known::Served { id: old, host: old_host, .. }) if *old_host != host => Some((old.clone(), *old_host)),
                        _ => None,
                    };
                    if let Some((old, old_host)) = moved {
                        self.hosts[old_host].live.retire(&old);
                        pass.retired += 1;
                        self.known.remove(path);
                    }
                    match self.hosts[host].live.serve(loaded) {
                        Ok(()) => {
                            eprintln!("{}", release.serving_line(&release.id));
                            pass.offered += 1;
                            let before = self.known.insert(path.clone(), Known::Served { id: release.id.clone(), host, stamp: *stamp });
                            // The file now names another id: the old one goes.
                            if let Some(Known::Served { id: old, host: old_host, .. }) = before {
                                if old != release.id {
                                    self.hosts[old_host].live.retire(&old);
                                    pass.retired += 1;
                                }
                            }
                        }
                        Err(why) => {
                            eprintln!("brain serve: not offering the adapter {}: {why}", path.display());
                            pass.refused += 1;
                            self.known.entry(path.clone()).and_modify(|k| *k = Known::Refused { stamp: *stamp }).or_insert(Known::Refused { stamp: *stamp });
                        }
                    }
                }
            }
        }

        // Files that are gone: their adapters are withdrawn.
        let gone: Vec<PathBuf> = self.known.keys().filter(|p| !files.contains_key(*p)).cloned().collect();
        for path in gone {
            if let Some(Known::Served { id, host, .. }) = self.known.remove(&path) {
                eprintln!("brain serve: adapter {id}: {} is gone, withdrawing it", path.display());
                self.hosts[host].live.retire(&id);
                pass.retired += 1;
            }
        }
        pass
    }

    /// The file that serves card id `id`, if one does.
    fn owner_of(&self, id: &str) -> Option<&PathBuf> {
        self.known.iter().find_map(|(path, k)| matches!(k, Known::Served { id: served, .. } if served == id).then_some(path))
    }
}

/// The adapter files under `dir`, with their stamps. Symbolic links, hidden
/// entries and everything that is not a regular `*.safetensors` file are not
/// listed; the walk stops at [`MAX_DEPTH`] levels and [`MAX_FILES`] files.
fn scan(dir: &Path) -> std::io::Result<BTreeMap<PathBuf, Stamp>> {
    let mut files = BTreeMap::new();
    let mut pending: Vec<(PathBuf, usize)> = vec![(dir.to_path_buf(), 0)];
    let mut warned = false;
    while let Some((here, depth)) = pending.pop() {
        let entries = match std::fs::read_dir(&here) {
            Ok(entries) => entries,
            // The root failing is the caller's to report; a subdirectory that
            // vanished or cannot be read is skipped.
            Err(e) if depth == 0 => return Err(e),
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') {
                continue;
            }
            // `file_type` does not follow a link, so a link is neither a file nor a directory here.
            let Ok(kind) = entry.file_type() else { continue };
            let path = entry.path();
            if kind.is_dir() && depth < MAX_DEPTH {
                pending.push((path, depth + 1));
            } else if kind.is_file() && name.ends_with(".safetensors") {
                if files.len() >= MAX_FILES {
                    if !std::mem::replace(&mut warned, true) {
                        eprintln!("brain serve: {} holds more than {MAX_FILES} adapter files; the rest are ignored", dir.display());
                    }
                    continue;
                }
                let Ok(meta) = entry.metadata() else { continue };
                files.insert(path, Stamp::of(&meta));
            }
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use brain_testutil::adapters::{file_digest as digest, scratch_dir as tmp, write_adapter, write_base};
    use llmserve::adapter_registry::AdapterBankSettings;
    use catalog::resident_llm::{QwenResident, QwenServeConfig};
    use checkpoint::st::ModelCard;
    use residency::{budget::Budgets, Device, Executor, Policy};
    use std::sync::Arc;

    const BASE_ID: &str = "local/base";
    const A: &str = "local/base:lab:alpha:v1";
    const B: &str = "local/base:lab:beta:v1";

    struct Fixture {
        /// Holds the scratch directory: the bases and the bank live in it, and it is removed when the fixture drops.
        _scratch: brain_testutil::Scratch,
        dir: PathBuf,
        bank: BankDirectory,
        executor: Executor,
        /// The served bases, in the order `ids` named them: their files and residents.
        bases: Vec<(PathBuf, Arc<QwenResident>)>,
        base: PathBuf,
    }

    /// A server of one toy Qwen3 base, `BASE_ID`, following a bank directory.
    fn fixture(name: &str) -> Fixture {
        fixture_of(name, &[BASE_ID])
    }

    /// A server of a toy Qwen3 base under each of `ids` (distinct files, so distinct digests), following a bank
    /// directory over all of them.
    fn fixture_of(name: &str, ids: &[&str]) -> Fixture {
        let dir = tmp(name);
        let config = QwenServeConfig { adapter_bank: Some(AdapterBankSettings::default()), ..QwenServeConfig::default() };
        let bases: Vec<(PathBuf, Arc<QwenResident>)> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| {
                let file = dir.join(format!("base-{i}.safetensors"));
                write_base(&file, 1.0 + i as f32);
                let card = ModelCard::new(*id, "qwen");
                (file.clone(), Arc::new(QwenResident::from_card_configured(file.to_str().unwrap(), &card, None, config.clone())))
            })
            .collect();
        let mut budgets = Budgets::new();
        budgets.set(Device::Cpu, 8 << 30, 0);
        let residents: Vec<Arc<dyn residency::ResidentModel>> = bases.iter().map(|(_, r)| r.clone() as Arc<dyn residency::ResidentModel>).collect();
        let executor = Executor::start(residents, budgets, Policy::default());
        let hosts = bases.iter().map(|(_, r)| BankHost::new(r.clone(), executor.clone())).collect();
        let published = dir.join("bank");
        std::fs::create_dir_all(&published).unwrap();
        let base = bases[0].0.clone();
        Fixture { bank: BankDirectory::new(published.clone(), hosts), dir: published, executor, bases, base, _scratch: dir }
    }

    fn listed(f: &Fixture, id: &str) -> bool {
        f.executor.manifests().iter().any(|m| m.model == id)
    }

    /// Publish an adapter the way a trainer should: aside, then renamed into place.
    fn publish(f: &Fixture, file: &str, id: &str, fill: f32) -> PathBuf {
        let at = f.dir.join(file);
        std::fs::create_dir_all(at.parent().unwrap()).unwrap();
        let aside = at.with_file_name(format!(".{}.partial", at.file_name().unwrap().to_string_lossy()));
        write_adapter(&aside, id, BASE_ID, Some(digest(&f.base)), fill);
        std::fs::rename(&aside, &at).unwrap();
        at
    }

    /// The spec: the offered set follows the directory. Several adapters are
    /// offered at once under their own ids, a replaced file takes the next
    /// policy version, a removed file withdraws only its adapter.
    #[test]
    fn the_offered_adapters_follow_the_files_in_the_directory() {
        let mut f = fixture("bank-follows");
        assert_eq!(f.bank.poll(), Pass::default(), "an empty directory offers nothing");

        let a = publish(&f, "alpha.safetensors", A, 0.5);
        publish(&f, "nested/beta.safetensors", B, 0.25);
        let pass = f.bank.poll();
        assert_eq!(pass, Pass { offered: 2, retired: 0, refused: 0 }, "both files are offered, one a level down");
        assert!(listed(&f, A) && listed(&f, B) && listed(&f, BASE_ID), "each adapter is a model of its own beside the base");
        assert_eq!(f.executor.served_adapter(A).unwrap().version, 1);
        assert_eq!(f.bank.poll(), Pass::default(), "nothing changed, nothing done");

        // A new release of the same adapter, renamed over the file: the same id, the next version.
        let before = digest(&a);
        publish(&f, "alpha.safetensors", A, 0.125);
        assert_ne!(digest(&a), before, "the file's bytes changed");
        assert_eq!(f.bank.poll(), Pass { offered: 1, retired: 0, refused: 0 });
        let served = f.executor.served_adapter(A).unwrap();
        assert_eq!((served.version, served.digest), (2, digest(&a)), "the replaced adapter is version 2 of the same id");
        assert_eq!(f.executor.served_adapter(B).unwrap().version, 1, "an adapter nobody touched keeps its version");

        // A removed file withdraws its own adapter and no other.
        std::fs::remove_file(&a).unwrap();
        assert_eq!(f.bank.poll(), Pass { offered: 0, retired: 1, refused: 0 });
        assert!(!listed(&f, A) && listed(&f, B) && listed(&f, BASE_ID));
        assert_eq!(f.bank.served_ids(), vec![B.to_string()]);

        // Offered again: the id continues its count, so a version never repeats.
        publish(&f, "alpha.safetensors", A, 0.0625);
        f.bank.poll();
        assert_eq!(f.executor.served_adapter(A).unwrap().version, 3);
        f.executor.shutdown();
    }

    /// Hostile or broken files: nothing is offered for them, the adapter a
    /// broken replacement would have replaced keeps being served, each file is
    /// said once per version, and a link or a hidden file is never read.
    #[test]
    fn files_that_do_not_verify_are_refused_and_never_replace_a_served_adapter() {
        let mut f = fixture("bank-refuses");
        let a = publish(&f, "alpha.safetensors", A, 0.5);
        f.bank.poll();
        let good = digest(&a);

        // A truncated replacement.
        let bytes = std::fs::read(&a).unwrap();
        std::fs::write(&a, &bytes[..bytes.len() / 2]).unwrap();
        assert_eq!(f.bank.poll(), Pass { offered: 0, retired: 0, refused: 1 });
        assert_eq!(f.executor.served_adapter(A).unwrap().digest, good, "the release already served keeps serving");
        assert_eq!(f.bank.poll(), Pass::default(), "a refused file is said once, not every tick");

        // An adapter trained on another base.
        let other = f.dir.join("other.safetensors");
        write_adapter(&other, B, BASE_ID, Some("sha256:".to_string() + &"0".repeat(64)), 0.5);
        assert_eq!(f.bank.poll(), Pass { offered: 0, retired: 0, refused: 1 });
        assert!(!listed(&f, B), "an adapter of another base is never offered");

        // The same card id from a second file.
        let twin = f.dir.join("twin.safetensors");
        write_adapter(&twin, A, BASE_ID, Some(digest(&f.base)), 0.75);
        assert_eq!(f.bank.poll(), Pass { offered: 0, retired: 0, refused: 1 });
        assert_eq!(f.executor.served_adapter(A).unwrap().digest, good, "two files cannot fight over one model id");

        // Links, hidden files and other names are not adapters.
        std::os::unix::fs::symlink(&twin, f.dir.join("link.safetensors")).unwrap();
        write_adapter(&f.dir.join(".hidden.safetensors"), B, BASE_ID, Some(digest(&f.base)), 0.5);
        write_adapter(&f.dir.join("notes.txt"), B, BASE_ID, Some(digest(&f.base)), 0.5);
        assert_eq!(f.bank.poll(), Pass::default());
        assert!(!listed(&f, B));
        assert_eq!(f.bank.served_ids(), vec![A.to_string()], "only the first file's adapter is offered");
        f.executor.shutdown();
    }

    /// The ids of the adapters `resident` holds.
    fn held(resident: &QwenResident) -> Vec<String> {
        let mut ids = llmserve::adapter_registry::AdapterHost::adapters(resident).ids();
        ids.sort();
        ids
    }

    /// A store with two bases served at once: each adapter is offered on the base its card names (verified by that
    /// base's digest), adapters of both are served together, and one whose base is not served is refused naming it.
    #[test]
    fn each_adapter_rides_on_the_base_its_card_names() {
        const SMALL: &str = "local/small";
        const LARGE: &str = "local/large";
        let mut f = fixture_of("bank-two-bases", &[SMALL, LARGE]);
        let (small, large) = (f.bases[0].clone(), f.bases[1].clone());
        let on_small = format!("{SMALL}:lab:alpha:v1");
        let on_large = format!("{LARGE}:lab:alpha:v1");
        write_adapter(&f.dir.join("small.safetensors"), &on_small, SMALL, Some(digest(&small.0)), 0.5);
        write_adapter(&f.dir.join("large.safetensors"), &on_large, LARGE, Some(digest(&large.0)), 0.25);
        assert_eq!(f.bank.poll(), Pass { offered: 2, retired: 0, refused: 0 }, "an adapter of each base is offered");
        assert_eq!((held(&small.1), held(&large.1)), (vec![on_small.clone()], vec![on_large.clone()]), "each adapter on its own base");
        assert!(listed(&f, &on_small) && listed(&f, &on_large) && listed(&f, SMALL) && listed(&f, LARGE));

        // Its base is not served: refused, naming the base, and nothing is offered on another one.
        let orphan = f.dir.join("orphan.safetensors");
        write_adapter(&orphan, "local/absent:lab:alpha:v1", "local/absent", Some("sha256:".to_string() + &"1".repeat(64)), 0.5);
        match verify_adapter_among(&orphan, &f.bank.bases) {
            Err(llmserve::adapter_release::BindError::BaseNotServed { base, served, .. }) => {
                assert_eq!(base, "\"local/absent\"");
                assert_eq!(served, vec![SMALL.to_string(), LARGE.to_string()]);
            }
            other => panic!("an adapter of a base nobody serves must be refused naming it, got {:?}", other.map(|(i, l)| (i, l.release.id))),
        }
        assert_eq!(f.bank.poll(), Pass { offered: 0, retired: 0, refused: 1 });
        assert!(!listed(&f, "local/absent:lab:alpha:v1"));

        // A card naming one base but trained on the other is refused by the digest, as for a single base.
        let crossed = f.dir.join("crossed.safetensors");
        write_adapter(&crossed, &format!("{SMALL}:lab:beta:v1"), SMALL, Some(digest(&large.0)), 0.5);
        assert_eq!(f.bank.poll(), Pass { offered: 0, retired: 0, refused: 1 });
        assert_eq!(held(&small.1), vec![on_small.clone()], "nothing trained on another base is put on this one");

        // A card whose base id no served base carries (the base is served under another name) binds by the digest it
        // was trained against, as a single base binds it.
        let by_digest = f.dir.join("by-digest.safetensors");
        write_adapter(&by_digest, &format!("{LARGE}:lab:gamma:v1"), "local/renamed", Some(digest(&large.0)), 0.5);
        assert!(matches!(verify_adapter_among(&by_digest, &f.bank.bases), Ok((1, _))), "bound to the base whose digest it records");
        f.executor.shutdown();
    }

    /// A file whose card now names another id moves its adapter: the new id is
    /// offered and the old one withdrawn.
    #[test]
    fn a_file_that_changes_its_card_id_moves_its_adapter() {
        let mut f = fixture("bank-renames");
        publish(&f, "alpha.safetensors", A, 0.5);
        f.bank.poll();
        publish(&f, "alpha.safetensors", B, 0.25);
        assert_eq!(f.bank.poll(), Pass { offered: 1, retired: 1, refused: 0 });
        assert!(listed(&f, B) && !listed(&f, A));
        f.executor.shutdown();
    }

    /// A served training run adapts all seven projections by default, and the bank a server builds with no
    /// `--adapter-targets` (the fixture's: `--serve-adapters` alone) must serve what it publishes without a restart.
    #[test]
    fn a_default_bank_serves_an_adapter_trained_on_every_projection() {
        let mut f = fixture("bank-all-targets");
        let file = f.dir.join("trained.safetensors");
        let sites = ["attn.wq", "attn.wk", "attn.wv", "attn.wo", "mlp.gate", "mlp.up", "mlp.down"];
        brain_testutil::adapters::write_adapter_over(&file, A, BASE_ID, Some(digest(&f.base)), 0.5, &sites);
        assert_eq!(f.bank.poll(), Pass { offered: 1, retired: 0, refused: 0 }, "an adapter of every projection is served by the default bank");
        assert!(listed(&f, A));
        f.executor.shutdown();
    }
}
