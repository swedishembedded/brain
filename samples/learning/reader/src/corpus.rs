// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A command-line tool that does not exist, so the model cannot already know
//! it.
//!
//! The hard part of demonstrating continual learning is not the learning, it
//! is having a claim anyone can check. "It got better at text" is not
//! checkable. So this sample invents a tool from a seed, writes that tool's
//! manual pages as the corpus the model reads, and ships the tool's own
//! GRAMMAR as the verifier. Before reading, the model writes invocations the
//! tool rejects; after reading, it writes invocations the tool accepts. The
//! judge is a parser, not another model, and the baseline is provably zero
//! because the tool did not exist until the seed was drawn.
//!
//! Two tools are generated and read in sequence, which is what makes
//! catastrophic forgetting legible: the question is whether learning the
//! second destroys the first.
//!
//! ## The lanes, and why each one is here
//!
//! A corpus of only learnable material would show that the reader can learn
//! and nothing about whether it can REFUSE. Every lane carries a label, and
//! the `selftest` verb checks each episode's verdict against it.

use std::collections::BTreeMap;
use std::path::Path;

/// Deterministic generator. A sample is self-contained and may not reach for
/// an engine crate, and ten lines of SplitMix64 is cheaper than a dependency
/// for the amount of randomness a corpus needs.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[(self.next() % items.len() as u64) as usize]
    }
}

/// One flag of one subcommand.
#[derive(Clone, Debug, PartialEq)]
pub struct Flag {
    /// Including the leading dashes.
    pub name: String,
    /// `Some(placeholder)` when the flag takes a value.
    pub value: Option<String>,
    pub required: bool,
    pub summary: String,
}

/// One subcommand.
#[derive(Clone, Debug, PartialEq)]
pub struct Command {
    pub name: String,
    pub summary: String,
    pub flags: Vec<Flag>,
    /// Pairs of flag indices that may not appear together.
    pub exclusive: Vec<(usize, usize)>,
}

/// A tool nobody has ever seen.
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    pub name: String,
    pub commands: Vec<Command>,
}

/// Why an invocation is not valid. Each variant is a distinct way a model
/// that has not read the manual gets it wrong, and naming them is what lets
/// a report say HOW the answers improved rather than only that they did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Invalid {
    NotThisTool,
    UnknownCommand(String),
    UnknownFlag(String),
    MissingRequired(String),
    MissingValue(String),
    UnexpectedValue(String),
    Exclusive(String, String),
}

const STEMS: &[&str] = &["vex", "quil", "zorn", "plim", "drax", "fenn", "gorp", "hask"];
const VERBS: &[&str] = &["sweep", "graft", "purge", "tally", "hoist", "splice", "render", "seal"];
const NOUNS: &[&str] = &["depth", "window", "budget", "origin", "stride", "cutoff", "weight", "anchor"];
const PLACEHOLDERS: &[&str] = &["PATH", "COUNT", "TIME", "SIZE", "NAME"];

impl Tool {
    /// Invent a tool. Deterministic in `seed`, so a run reproduces and two
    /// seeds give two genuinely different tools.
    pub fn generate(seed: u64) -> Tool {
        Tool::generate_disjoint(seed, &[])
    }

    /// Invent a tool sharing no command and no flag name with `others`.
    ///
    /// Two tools read in sequence are how this sample makes forgetting
    /// legible, and that question is only sharp if they genuinely do not
    /// overlap. A shared `--depth` between them would mean learning the
    /// second REINFORCES part of the first, and "did B destroy A" would be
    /// measuring something other than retention. The word lists are small
    /// enough that collisions are likely rather than rare, so disjointness
    /// is constructed rather than hoped for.
    pub fn generate_disjoint(seed: u64, others: &[&Tool]) -> Tool {
        let taken_commands: Vec<String> = others.iter().flat_map(|t| t.commands.iter().map(|c| c.name.clone())).collect();
        let taken_flags: Vec<String> = others.iter().flat_map(|t| t.commands.iter().flat_map(|c| c.flags.iter().map(|f| f.name.clone()))).collect();
        let mut rng = Rng::new(seed);
        let name = format!("{}{}", rng.pick(STEMS), rng.next() % 10);
        let n_commands = 4 + (rng.next() % 3) as usize;
        let mut used_commands: Vec<String> = taken_commands;
        let mut commands = Vec::new();
        for _ in 0..n_commands {
            let mut cmd = rng.pick(VERBS).to_string();
            while used_commands.contains(&cmd) {
                cmd = format!("{}{}", rng.pick(VERBS), rng.next() % 100);
            }
            used_commands.push(cmd.clone());

            let n_flags = 3 + (rng.next() % 3) as usize;
            // Seeded with every flag name any other tool uses, so the two
            // tools' vocabularies cannot overlap at all.
            let mut used_flags: Vec<String> = taken_flags.iter().map(|f| f.trim_start_matches("--").to_string()).collect();
            let mut flags = Vec::new();
            for f in 0..n_flags {
                let mut stem = rng.pick(NOUNS).to_string();
                while used_flags.contains(&stem) {
                    stem = format!("{}{}", rng.pick(NOUNS), rng.next() % 100);
                }
                used_flags.push(stem.clone());
                let takes_value = !rng.next().is_multiple_of(3);
                let value = takes_value.then(|| rng.pick(PLACEHOLDERS).to_string());
                let required = f == 0;
                let summary = match &value {
                    Some(v) => format!("set the {stem} to {v}"),
                    None => format!("turn on {stem}"),
                };
                flags.push(Flag { name: format!("--{stem}"), value, required, summary });
            }
            // One exclusive pair per command where there is room for it, so
            // the grammar has a rule that cannot be guessed from the flag
            // names alone.
            let exclusive = if flags.len() >= 3 { vec![(1, 2)] } else { Vec::new() };
            commands.push(Command { name: cmd.clone(), summary: format!("{cmd} the {} store", rng.pick(NOUNS)), flags, exclusive });
        }
        Tool { name, commands }
    }

    pub fn command(&self, name: &str) -> Option<&Command> {
        self.commands.iter().find(|c| c.name == name)
    }

    /// The manual page for one subcommand: what the model reads.
    pub fn man_page(&self, cmd: &Command) -> String {
        let mut out = String::new();
        out.push_str(&format!("{} {} - {}\n", self.name, cmd.name, cmd.summary));
        out.push_str(&format!("usage: {} {} [OPTIONS]\n", self.name, cmd.name));
        out.push_str("options:\n");
        for f in &cmd.flags {
            let spec = match &f.value {
                Some(v) => format!("{} {v}", f.name),
                None => f.name.clone(),
            };
            let req = if f.required { " (required)" } else { "" };
            out.push_str(&format!("  {spec}   {}{req}\n", f.summary));
        }
        for &(a, b) in &cmd.exclusive {
            out.push_str(&format!("  {} may not be given together with {}\n", cmd.flags[a].name, cmd.flags[b].name));
        }
        out
    }

    /// Every manual page, in command order.
    pub fn manual(&self) -> String {
        self.commands.iter().map(|c| self.man_page(c)).collect::<Vec<String>>().join("\n")
    }

    /// The canonical invocation for a command: its required flags, with a
    /// value where one is wanted.
    pub fn canonical(&self, cmd: &Command) -> String {
        let mut out = format!("{} {}", self.name, cmd.name);
        for f in cmd.flags.iter().filter(|f| f.required) {
            out.push(' ');
            out.push_str(&f.name);
            if let Some(v) = &f.value {
                out.push(' ');
                out.push_str(&value_for(v));
            }
        }
        out
    }

    /// Does the tool accept this? The verifier, and the whole reason the
    /// claim is checkable: a parser, never a second model's opinion.
    pub fn parse(&self, invocation: &str) -> Result<(), Invalid> {
        let mut words = invocation.split_whitespace();
        match words.next() {
            Some(w) if w == self.name => {}
            _ => return Err(Invalid::NotThisTool),
        }
        let cmd_name = words.next().unwrap_or_default().to_string();
        let cmd = self.command(&cmd_name).ok_or(Invalid::UnknownCommand(cmd_name))?;

        let mut seen: Vec<usize> = Vec::new();
        let rest: Vec<&str> = words.collect();
        let mut i = 0;
        while i < rest.len() {
            let word = rest[i];
            if !word.starts_with("--") {
                return Err(Invalid::UnexpectedValue(word.to_string()));
            }
            let idx = cmd.flags.iter().position(|f| f.name == word).ok_or_else(|| Invalid::UnknownFlag(word.to_string()))?;
            seen.push(idx);
            match cmd.flags[idx].value {
                Some(_) => {
                    let next = rest.get(i + 1);
                    match next {
                        Some(v) if !v.starts_with("--") => i += 2,
                        _ => return Err(Invalid::MissingValue(word.to_string())),
                    }
                }
                None => {
                    if rest.get(i + 1).is_some_and(|v| !v.starts_with("--")) {
                        return Err(Invalid::UnexpectedValue(word.to_string()));
                    }
                    i += 1;
                }
            }
        }

        for (n, f) in cmd.flags.iter().enumerate() {
            if f.required && !seen.contains(&n) {
                return Err(Invalid::MissingRequired(f.name.clone()));
            }
        }
        for &(a, b) in &cmd.exclusive {
            if seen.contains(&a) && seen.contains(&b) {
                return Err(Invalid::Exclusive(cmd.flags[a].name.clone(), cmd.flags[b].name.clone()));
            }
        }
        Ok(())
    }

    /// The capability battery: one task per subcommand, asked in words and
    /// answered with an invocation.
    ///
    /// These are frozen BEFORE any reading, and the model is never trained
    /// on them. They are what the user gains, and the only number in this
    /// sample that is a result rather than plumbing.
    pub fn battery(&self) -> Vec<(String, String)> {
        self.commands
            .iter()
            .map(|c| (format!("How do I {} with {}? Answer with the command only.\n", c.summary, self.name), self.canonical(c)))
            .collect()
    }
}

/// A plausible value for a placeholder, so a canonical invocation is a thing
/// someone would actually type.
fn value_for(placeholder: &str) -> String {
    match placeholder {
        "PATH" => "/var/data".to_string(),
        "COUNT" => "16".to_string(),
        "TIME" => "7d".to_string(),
        "SIZE" => "64M".to_string(),
        _ => "main".to_string(),
    }
}

/// What a lane's episodes are expected to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expect {
    /// Should be learned.
    Promote,
    /// Should be refused, at any stage before the gate promotes it.
    Refuse,
    /// May go either way; the lane proves something other than the verdict.
    Either,
}

/// Write the corpus, and the labels the selftest checks verdicts against.
///
/// Every lane exists because a corpus of only learnable material would show
/// that the reader can learn and nothing about whether it can refuse.
pub fn write(dir: &Path, a: &Tool, b: &Tool) -> std::io::Result<BTreeMap<String, Expect>> {
    let mut labels = BTreeMap::new();
    let put = |rel: &str, body: &str, expect: Expect, labels: &mut BTreeMap<String, Expect>| -> std::io::Result<()> {
        let path = dir.join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&path, body)?;
        labels.insert(rel.to_string(), expect);
        Ok(())
    };

    // learn: the real manuals for both tools, in sequence. Whether learning
    // B destroys A is catastrophic forgetting in its most legible form.
    put("learn/a-manual.txt", &a.manual(), Expect::Promote, &mut labels)?;
    put("learn/b-manual.txt", &b.manual(), Expect::Promote, &mut labels)?;

    // repeat: already known by the time it is read, and must cost one
    // forward pass rather than a training run.
    put("repeat/a-again.txt", &a.manual(), Expect::Refuse, &mut labels)?;

    // noise: valid UTF-8 with no structure in it. The stream's binary screen
    // passes this; the triage screen is what must not.
    let mut rng = Rng::new(0xDEAD_BEEF);
    const B64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let noise: String = (0..4000).map(|_| B64[(rng.next() % 64) as usize] as char).collect();
    put("noise/payload.txt", &noise, Expect::Refuse, &mut labels)?;

    // contradict: says a flag takes a value when it does not. Must be
    // refused or flagged, and must never silently flip what was learned.
    let cmd = &a.commands[0];
    let flag = &cmd.flags[0];
    let contradiction = format!(
        "{} {} - corrected notes\n  {} {}   the {} is now given as an argument\n  {} no longer takes a value\n",
        a.name, cmd.name, flag.name, "ALWAYS", flag.name, cmd.flags[1].name
    );
    put("contradict/a-notes.txt", &contradiction, Expect::Refuse, &mut labels)?;

    // degenerate: structured but uninformative. NOT a screening concern -
    // it scores high on structure - and is caught by the reach filter
    // instead, where a model predicts it trivially.
    let degenerate = format!("{} {} --{}\n", a.name, cmd.name, "again").repeat(500);
    put("degenerate/loop.txt", &degenerate, Expect::Refuse, &mut labels)?;

    // shortcut: contains a battery answer verbatim. The reader must refuse
    // it at ingest rather than train on the answers it is scored against.
    let (_, answer) = &a.battery()[0];
    let leak = format!("{}\nexample session:\n{}\n{}\n", a.man_page(cmd), answer, a.man_page(&a.commands[1]));
    put("shortcut/a-examples.txt", &leak, Expect::Either, &mut labels)?;

    // format: the shape of a manual page describing nothing real. May
    // promote; what must not move is the paraphrase probes.
    let hollow: String = (0..30).map(|i| format!("  --option{i:03} VALUE   set the option {i:03} to VALUE\n")).collect();
    put("format/hollow.txt", &format!("generic tool - options\nusage: generic [OPTIONS]\noptions:\n{hollow}"), Expect::Either, &mut labels)?;

    // counterfact: near-identical lines with opposite meanings, which is the
    // surface-memorisation trap stated as data.
    let cf = format!(
        "{} {} - windows\n  --window-before TIME   act on entries older than TIME\n  --window-behind TIME   act on entries newer than TIME\n",
        a.name, a.commands[1].name
    );
    put("counterfact/a-windows.txt", &cf, Expect::Either, &mut labels)?;

    // rare: one subcommand documented here and never mentioned again. It
    // must still be invokable at the end of the run.
    let rare = a.commands.last().expect("a tool has commands");
    put("rare/a-seldom.txt", &a.man_page(rare), Expect::Promote, &mut labels)?;

    let json: String = serde_json_labels(&labels);
    // Dot-prefixed so the reader's own stream does not read the answer
    // key as a document - see `audit::stream`'s rule 5. A corpus that
    // contained its own labels would be a corpus with the answers in it.
    std::fs::write(dir.join(LABELS), json)?;
    Ok(labels)
}

/// The labels a corpus was written with, read back from the corpus itself.
///
/// The selftest checks verdicts against these, and reads them rather than
/// regenerating the corpus from a seed: a regeneration would overwrite the
/// very documents a finished run was about, and under the wrong seed it
/// would replace them with another tool's while still reporting a verdict.
/// The labels on disk belong to the corpus that was actually read.
/// The answer key's file name.
pub const LABELS: &str = ".labels.json";

pub fn labels(dir: &Path) -> std::io::Result<BTreeMap<String, Expect>> {
    let path = dir.join(LABELS);
    let raw = std::fs::read_to_string(&path)?;
    let mut out = BTreeMap::new();
    for line in raw.lines() {
        let line = line.trim().trim_end_matches(',');
        let Some((k, v)) = line.split_once(':') else { continue };
        let (k, v) = (k.trim().trim_matches('"'), v.trim().trim_matches('"'));
        let expect = match v {
            "promote" => Expect::Promote,
            "refuse" => Expect::Refuse,
            "either" => Expect::Either,
            _ => continue,
        };
        out.insert(k.to_string(), expect);
    }
    if out.is_empty() {
        return Err(std::io::Error::other(format!("{}: no labels, so there is nothing to check verdicts against", path.display())));
    }
    Ok(out)
}

/// The labels file, written without a serialisation dependency: it is a flat
/// map of strings to one of three words.
fn serde_json_labels(labels: &BTreeMap<String, Expect>) -> String {
    let rows: Vec<String> = labels
        .iter()
        .map(|(k, v)| {
            let word = match v {
                Expect::Promote => "promote",
                Expect::Refuse => "refuse",
                Expect::Either => "either",
            };
            format!("  {:?}: {:?}", k, word)
        })
        .collect();
    format!("{{\n{}\n}}\n", rows.join(",\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> Tool {
        Tool::generate(42)
    }

    /// The selftest's labels come from the corpus that was read, so they
    /// must round-trip through the file and reading them must leave that
    /// corpus untouched.
    #[test]
    fn labels_are_read_back_from_the_corpus_without_rewriting_it() {
        let dir = std::env::temp_dir().join(format!("sample-reader-labels-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let a = Tool::generate(7);
        let b = Tool::generate_disjoint(8, &[&a]);
        let written = write(&dir, &a, &b).expect("corpus");

        let one = std::fs::read(dir.join("learn/a-manual.txt")).expect("an episode");
        let read_back = labels(&dir).expect("labels");
        assert_eq!(read_back, written, "the labels on disk are the labels the run was written with");
        assert_eq!(one, std::fs::read(dir.join("learn/a-manual.txt")).expect("an episode"), "reading labels must not rewrite the corpus");

        let empty = dir.join("empty");
        std::fs::create_dir_all(&empty).expect("mkdir");
        std::fs::write(empty.join(LABELS), "{}\n").expect("write");
        assert!(labels(&empty).is_err(), "a corpus with no labels must say so rather than check nothing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The claim this whole sample rests on: the tool did not exist until
    /// the seed was drawn, so a model cannot already know it, and two seeds
    /// give two genuinely different tools rather than two spellings of one.
    #[test]
    fn a_tool_is_deterministic_in_its_seed_and_different_seeds_differ() {
        assert_eq!(Tool::generate(7), Tool::generate(7));
        let (a, b) = (Tool::generate(7), Tool::generate(8));
        assert_ne!(a.name, b.name);
        assert_ne!(a.commands, b.commands);
    }

    /// A generated tool must be internally coherent, or the verifier would
    /// be scoring against a grammar the manual never described.
    #[test]
    fn every_generated_command_has_distinct_flags_and_exactly_one_required() {
        for seed in 0..12u64 {
            let t = Tool::generate(seed);
            assert!(t.commands.len() >= 4, "seed {seed}: too few commands");
            let names: Vec<&str> = t.commands.iter().map(|c| c.name.as_str()).collect();
            let mut sorted = names.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), names.len(), "seed {seed}: duplicate command names");
            for c in &t.commands {
                let mut fl: Vec<&str> = c.flags.iter().map(|f| f.name.as_str()).collect();
                let n = fl.len();
                fl.sort_unstable();
                fl.dedup();
                assert_eq!(fl.len(), n, "seed {seed}: {} has duplicate flags", c.name);
                assert_eq!(c.flags.iter().filter(|f| f.required).count(), 1, "seed {seed}: {} must have one required flag", c.name);
            }
        }
    }

    /// The success case. Every battery answer must be something the tool
    /// actually accepts, or the target the model is being trained towards is
    /// itself wrong.
    #[test]
    fn every_battery_answer_is_an_invocation_the_tool_accepts() {
        for seed in 0..12u64 {
            let t = Tool::generate(seed);
            for (task, answer) in t.battery() {
                assert!(t.parse(&answer).is_ok(), "seed {seed}: canonical answer {answer:?} is rejected by its own grammar (task {task:?})");
            }
            assert_eq!(t.battery().len(), t.commands.len(), "one battery task per subcommand");
        }
    }

    /// And the failure cases, one per way of getting it wrong. A verifier
    /// that only ever accepted would make the before/after difference
    /// meaningless, so each rejection is named rather than merely counted.
    #[test]
    fn the_verifier_rejects_each_distinct_way_of_getting_it_wrong() {
        let t = tool();
        let cmd = &t.commands[0];
        let required = cmd.flags.iter().find(|f| f.required).expect("one required flag");

        assert_eq!(t.parse("someothertool sweep --depth 1"), Err(Invalid::NotThisTool));
        assert!(matches!(t.parse(&format!("{} notacommand", t.name)), Err(Invalid::UnknownCommand(_))));
        assert!(matches!(t.parse(&format!("{} {} --invented 1", t.name, cmd.name)), Err(Invalid::UnknownFlag(_))));

        // Everything but the required flag: the mistake a model makes when
        // it has seen the flag names but not which one is mandatory.
        let optional: Vec<&Flag> = cmd.flags.iter().filter(|f| !f.required).collect();
        let mut without = format!("{} {}", t.name, cmd.name);
        if let Some(f) = optional.first() {
            without.push_str(&format!(" {}", f.name));
            if let Some(v) = &f.value {
                without.push_str(&format!(" {}", value_for(v)));
            }
        }
        assert_eq!(t.parse(&without), Err(Invalid::MissingRequired(required.name.clone())));

        // A flag that wants a value, given none.
        if let Some(f) = cmd.flags.iter().find(|f| f.value.is_some()) {
            let bad = format!("{} {} {}", t.name, cmd.name, f.name);
            assert!(matches!(t.parse(&bad), Err(Invalid::MissingValue(_)) | Err(Invalid::MissingRequired(_))));
        }
    }

    /// The exclusive rule is the part of the grammar that cannot be guessed
    /// from the flag names, so it is the part reading the manual actually
    /// buys. Both halves: together is refused, separately is fine.
    #[test]
    fn mutually_exclusive_flags_are_refused_together_and_accepted_apart() {
        let t = tool();
        let cmd = t.commands.iter().find(|c| !c.exclusive.is_empty()).expect("a generated tool has an exclusive pair");
        let (a, b) = cmd.exclusive[0];
        let req = cmd.flags.iter().position(|f| f.required).expect("one required");

        let with = |idx: &[usize]| {
            let mut s = format!("{} {}", t.name, cmd.name);
            for &i in idx {
                s.push_str(&format!(" {}", cmd.flags[i].name));
                if let Some(v) = &cmd.flags[i].value {
                    s.push_str(&format!(" {}", value_for(v)));
                }
            }
            s
        };

        match t.parse(&with(&[req, a, b])) {
            Err(Invalid::Exclusive(_, _)) => {}
            other => panic!("expected an exclusivity refusal, got {other:?}"),
        }
        assert!(t.parse(&with(&[req, a])).is_ok(), "either one alone must be accepted");
        assert!(t.parse(&with(&[req, b])).is_ok(), "either one alone must be accepted");
    }

    /// A manual page has to actually describe the grammar, or reading it
    /// could not teach anything the verifier checks.
    #[test]
    fn a_manual_page_names_every_flag_and_the_exclusive_rule() {
        let t = tool();
        for c in &t.commands {
            let page = t.man_page(c);
            for f in &c.flags {
                assert!(page.contains(&f.name), "{} does not mention {}", c.name, f.name);
            }
            for &(a, b) in &c.exclusive {
                assert!(
                    page.contains(&format!("{} may not be given together with {}", c.flags[a].name, c.flags[b].name)),
                    "{} does not state its exclusive rule",
                    c.name
                );
            }
        }
    }

    /// The lanes are the failure-mode half of this sample. Each must be
    /// written, labelled, and actually be what it claims: the noise lane
    /// genuinely unstructured, the shortcut lane genuinely containing an
    /// answer, the contradiction genuinely contradicting.
    #[test]
    fn every_lane_is_written_labelled_and_is_what_it_claims_to_be() {
        let dir = std::env::temp_dir().join(format!("sample-reader-corpus-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (a, b) = (Tool::generate(1), Tool::generate(2));
        let labels = write(&dir, &a, &b).expect("corpus writes");

        for lane in ["learn/", "repeat/", "noise/", "contradict/", "degenerate/", "shortcut/", "format/", "counterfact/", "rare/"] {
            assert!(labels.keys().any(|k| k.starts_with(lane)), "no episode written for lane {lane}");
        }
        for rel in labels.keys() {
            assert!(dir.join(rel).exists(), "{rel} is labelled but was not written");
        }
        assert!(dir.join(LABELS).exists());

        let noise = std::fs::read_to_string(dir.join("noise/payload.txt")).expect("noise");
        assert!(noise.chars().all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/'), "the noise lane must be valid text, or the STREAM refuses it and triage is never exercised");

        let leak = std::fs::read_to_string(dir.join("shortcut/a-examples.txt")).expect("shortcut");
        assert!(leak.contains(&a.battery()[0].1), "the shortcut lane must actually contain a battery answer");

        let contra = std::fs::read_to_string(dir.join("contradict/a-notes.txt")).expect("contradict");
        assert!(contra.contains(&a.commands[0].flags[0].name), "the contradiction must be about a flag the manual described");

        let degen = std::fs::read_to_string(dir.join("degenerate/loop.txt")).expect("degenerate");
        let lines: Vec<&str> = degen.lines().collect();
        assert!(lines.len() > 400 && lines.iter().all(|l| *l == lines[0]), "the degenerate lane must be one line repeated");

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Two tools must not overlap, or "did learning B destroy A" is not a
    /// question about retention at all.
    #[test]
    fn the_two_tools_share_no_command_and_no_flag_name() {
        let a = Tool::generate(1);
        let b = Tool::generate_disjoint(2, &[&a]);
        assert_ne!(a.name, b.name);

        let names = |t: &Tool| t.commands.iter().map(|c| c.name.clone()).collect::<Vec<String>>();
        let flags = |t: &Tool| t.commands.iter().flat_map(|c| c.flags.iter().map(|f| f.name.clone())).collect::<Vec<String>>();
        for n in names(&a) {
            assert!(!names(&b).contains(&n), "the two tools share the command {n}");
        }
        for f in flags(&a) {
            assert!(!flags(&b).contains(&f), "the two tools share the flag {f}, so learning one reinforces the other");
        }
        // Whatever else overlaps, an invocation of one must not parse as the
        // other: the tool name alone guarantees it.
        let (_, answer) = &a.battery()[0];
        assert_eq!(b.parse(answer), Err(Invalid::NotThisTool));
    }
}
