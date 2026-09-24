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

use std::collections::{BTreeMap, BTreeSet};
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
    /// The flag takes a value of a particular shape and got something else.
    /// Distinct from [`Invalid::MissingValue`]: the model DID supply one,
    /// and a tool that accepted `--depth banana` because it accepted any
    /// word would call a wrong answer right.
    BadValue { flag: String, wants: String, got: String },
    /// The same flag twice. A real tool either refuses this or silently
    /// keeps one of them; refusing is the only one of those that cannot
    /// quietly accept an answer nobody meant.
    RepeatedFlag(String),
}

/// Whether `value` is the shape `placeholder` names.
///
/// The strictness the whole verification rests on. A parser that accepts any
/// word as any value reports a model that wrote `--depth banana` as having
/// answered correctly, and every number built on that is inflated.
fn value_is(placeholder: &str, value: &str) -> bool {
    let digits_then = |suffixes: &str| {
        let split = value.find(|c: char| !c.is_ascii_digit()).unwrap_or(value.len());
        let (n, unit) = value.split_at(split);
        !n.is_empty() && unit.len() == 1 && suffixes.contains(unit)
    };
    match placeholder {
        "PATH" => value.starts_with('/') && value.len() > 1,
        "COUNT" => !value.is_empty() && value.chars().all(|c| c.is_ascii_digit()),
        "TIME" => digits_then("smhd"),
        "SIZE" => digits_then("KMG"),
        "NAME" => !value.is_empty() && value.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
        // An unknown placeholder is a defect in the generator, not in the
        // answer: accept rather than fail every invocation that uses it.
        _ => true,
    }
}

/// What running a command did, as the tool reports it.
///
/// **Nothing here touches anything outside this struct.** The tool has no
/// filesystem, no network and no state that outlives the call, which is what
/// makes it safe to execute a language model's guess: the worst a wrong
/// answer can do is produce a different transcript.
///
/// Deterministic in the parsed invocation and nothing else, so two
/// invocations that MEAN the same thing compare equal - flag order does not
/// matter - and two that differ in anything the tool acts on do not.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    pub lines: Vec<String>,
}

const STEMS: &[&str] = &["vex", "quil", "zorn", "plim", "drax", "fenn", "gorp", "hask", "mubi", "tarn", "wold", "brim"];
/// What a made-up tool's name ends in, so it reads like something someone
/// shipped rather than like a placeholder with a counter after it.
const TOOL_SUFFIXES: &[&str] = &["ctl", "kit", "adm", "svc", "fs", "d"];
const VERBS: &[&str] =
    &["sweep", "graft", "purge", "tally", "hoist", "splice", "render", "seal", "prune", "stage", "verify", "rebuild"];
const NOUNS: &[&str] = &[
    "depth", "window", "budget", "origin", "stride", "cutoff", "weight", "anchor", "margin", "offset", "retry", "shard",
    "region", "bucket", "label", "quota",
];
/// The other half of a compound flag name. A real tool disambiguates
/// `--depth` into `--max-depth` and `--keep-depth`, not into `--depth55`.
const MODIFIERS: &[&str] =
    &["max", "min", "keep", "skip", "force", "dry", "auto", "soft", "hard", "strict", "fast", "deep"];
const PLACEHOLDERS: &[&str] = &["PATH", "COUNT", "TIME", "SIZE", "NAME"];

/// A name not already in `used`, drawn from `simple` first and then from the
/// compound space `modifier-simple`.
///
/// Bounded random draws, then a deterministic sweep of the whole space, so
/// this terminates and stays reproducible. The compound space is what makes
/// a numeric suffix unnecessary: 12 modifiers over 16 nouns is 208 plausible
/// flag names, against the roughly 50 a pair of tools needs.
fn fresh_name(rng: &mut Rng, used: &[String], simple: &[&str], modifiers: &[&str], forbidden: &BTreeSet<String>) -> String {
    // A candidate is out if any WORD of it is spoken for. `--skip-budget`
    // beside `budget-rebuild` collides on `budget`, which is enough to make
    // "set the budget" and "budget rebuild" ambiguous even though the two
    // names differ.
    let clashes = |c: &str| c.split('-').any(|w| forbidden.contains(w));
    for _ in 0..48 {
        let candidate = if rng.next().is_multiple_of(3) {
            rng.pick(simple).to_string()
        } else {
            format!("{}-{}", rng.pick(modifiers), rng.pick(simple))
        };
        if !used.contains(&candidate) && !clashes(&candidate) {
            return candidate;
        }
    }
    for m in modifiers {
        for w in simple {
            let candidate = format!("{m}-{w}");
            if !used.contains(&candidate) && !clashes(&candidate) {
                return candidate;
            }
        }
    }
    unreachable!("the compound space is far larger than any tool this generates")
}

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
        let name = format!("{}{}", rng.pick(STEMS), rng.pick(TOOL_SUFFIXES));
        let n_commands = 4 + (rng.next() % 3) as usize;
        let mut used_commands: Vec<String> = taken_commands;
        let mut commands = Vec::new();
        for _ in 0..n_commands {
            let cmd = fresh_name(&mut rng, &used_commands, VERBS, NOUNS, &BTreeSet::new());
            used_commands.push(cmd.clone());

            // Enough flags that a SINGLE command's page is a document the
            // reader can judge. Its structure screen cannot tell text from
            // noise under `brain::MIN_EPISODE_CHARS`, and several lanes are
            // built from one page, so a thinner command makes those lanes
            // refusable for their length rather than for what they contain.
            // Pinned by
            // `every_generated_document_clears_the_readers_own_length_floor`
            // across seeds, because the count is drawn per seed.
            let n_flags = 6 + (rng.next() % 4) as usize;
            // Seeded with every flag name any other tool uses, so the two
            // tools' vocabularies cannot overlap at all.
            // Seeded with every flag name any other tool uses, so the two
            // tools' vocabularies cannot overlap at all - AND with this
            // command's own name, so no flag can share a word with the
            // command it belongs to. `quota-hoist` beside `--quota` is
            // ambiguous in a way no reader can resolve: "set the quota" and
            // "quota hoist" name different things with the same word.
            let mut used_flags: Vec<String> = taken_flags.iter().map(|f| f.trim_start_matches("--").to_string()).collect();
            let own_words: BTreeSet<String> = cmd.split('-').map(str::to_string).collect();
            let mut flags = Vec::new();
            for f in 0..n_flags {
                let stem = fresh_name(&mut rng, &used_flags, NOUNS, MODIFIERS, &own_words);
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
            let action = cmd.replace('-', " ");
            commands.push(Command { name: cmd.clone(), summary: format!("{action} the {} store", rng.pick(NOUNS)), flags, exclusive });
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

        // What a placeholder actually accepts. Without this the manual
        // documents that `--depth` takes a TIME and never says what a TIME
        // looks like, so a model reading it writes `--depth 5` and the tool
        // refuses it - a task that cannot be learned from the material it is
        // supposed to be learned from.
        let used: BTreeSet<&str> = cmd.flags.iter().filter_map(|f| f.value.as_deref()).collect();
        if !used.is_empty() {
            out.push_str("values:\n");
            for placeholder in &used {
                out.push_str(&format!("  {placeholder} is {}\n", value_spec(placeholder)));
            }
        }

        // Worked invocations, so the SHAPE of a command line is demonstrated
        // and not only its grammar. Deliberately not the canonical one - see
        // `Tool::examples`.
        let examples = self.examples(cmd);
        if !examples.is_empty() {
            out.push_str("examples:\n");
            for (invocation, what) in examples {
                out.push_str(&format!("  {invocation}\n      {what}\n"));
            }
        }
        out
    }

    /// Worked invocations for a command, for its manual page.
    ///
    /// Every one of them uses at least one OPTIONAL flag, which keeps them
    /// clear of the capability battery: the battery asks for the canonical
    /// invocation, which is the required flags and nothing else, and a
    /// manual that printed that answer would be teaching the test rather
    /// than the tool. Demonstrating the form on other combinations is what
    /// makes the canonical one derivable instead of memorable.
    pub fn examples(&self, cmd: &Command) -> Vec<(String, String)> {
        let required: Vec<&Flag> = cmd.flags.iter().filter(|f| f.required).collect();
        let base: Vec<String> = required.iter().map(|f| spec_of(f)).collect();
        let excluded: BTreeSet<&str> =
            cmd.exclusive.iter().flat_map(|&(a, b)| [cmd.flags[a].name.as_str(), cmd.flags[b].name.as_str()]).collect();
        cmd.flags
            .iter()
            .filter(|f| !f.required && !excluded.contains(f.name.as_str()))
            .take(2)
            .map(|f| {
                let invocation = format!("{} {} {} {}", self.name, cmd.name, base.join(" "), spec_of(f));
                (invocation.split_whitespace().collect::<Vec<&str>>().join(" "), format!("{}, and {}", cmd.summary, f.summary))
            })
            .collect()
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
        self.parse_into(invocation).map(|_| ())
    }

    /// Run `invocation` and report what it did.
    ///
    /// The oracle the whole pipeline rests on: a generated answer is correct
    /// when running it produces the same observable result as running the
    /// reference, which is a stronger claim than "it looks similar" and a
    /// different one from "it parses". `hask0 graft --cutoff --depth 7d`
    /// parses and is not the same command as `hask0 graft --cutoff`.
    ///
    /// Safe to call on anything a model produced: see [`Outcome`].
    pub fn run(&self, invocation: &str) -> Result<Outcome, Invalid> {
        let (cmd, given) = self.parse_into(invocation)?;
        let mut lines = vec![format!("{} {}: begin", self.name, cmd.name)];
        // Sorted by flag name, so the transcript is a function of WHAT was
        // asked for and not of the order it was written in.
        for (name, value) in &given {
            lines.push(match value {
                Some(v) => format!("  set {name} = {v}"),
                None => format!("  enable {name}"),
            });
        }
        lines.push(format!("{} {}: {} option(s) applied to the {} store", self.name, cmd.name, given.len(), noun_of(&cmd.summary)));
        Ok(Outcome { lines })
    }

    /// Parse, returning the command and the flags actually given, in name
    /// order. The single place an invocation is understood; [`Tool::parse`]
    /// and [`Tool::run`] are both this.
    fn parse_into(&self, invocation: &str) -> Result<(&Command, BTreeMap<String, Option<String>>), Invalid> {
        let mut words = invocation.split_whitespace();
        match words.next() {
            Some(w) if w == self.name => {}
            _ => return Err(Invalid::NotThisTool),
        }
        let cmd_name = words.next().unwrap_or_default().to_string();
        let cmd = self.command(&cmd_name).ok_or(Invalid::UnknownCommand(cmd_name))?;

        let mut seen: Vec<usize> = Vec::new();
        let mut given: BTreeMap<String, Option<String>> = BTreeMap::new();
        let rest: Vec<&str> = words.collect();
        let mut i = 0;
        while i < rest.len() {
            let word = rest[i];
            if !word.starts_with("--") {
                return Err(Invalid::UnexpectedValue(word.to_string()));
            }
            let idx = cmd.flags.iter().position(|f| f.name == word).ok_or_else(|| Invalid::UnknownFlag(word.to_string()))?;
            if seen.contains(&idx) {
                return Err(Invalid::RepeatedFlag(word.to_string()));
            }
            seen.push(idx);
            match &cmd.flags[idx].value {
                Some(placeholder) => {
                    let next = rest.get(i + 1);
                    match next {
                        Some(v) if !v.starts_with("--") => {
                            if !value_is(placeholder, v) {
                                return Err(Invalid::BadValue {
                                    flag: word.to_string(),
                                    wants: placeholder.clone(),
                                    got: (*v).to_string(),
                                });
                            }
                            given.insert(word.to_string(), Some((*v).to_string()));
                            i += 2;
                        }
                        _ => return Err(Invalid::MissingValue(word.to_string())),
                    }
                }
                None => {
                    if rest.get(i + 1).is_some_and(|v| !v.starts_with("--")) {
                        return Err(Invalid::UnexpectedValue(word.to_string()));
                    }
                    given.insert(word.to_string(), None);
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
        Ok((cmd, given))
    }

    /// Why `question` is not a fair question for `answer`, or `None` when it
    /// is.
    ///
    /// The check the answer being correct does NOT make unnecessary. A model
    /// asked for several questions about one command drifts onto the other
    /// flags it can see in the material, and the answer is attached
    /// regardless - so "what command turns on weight44?" gets paired with a
    /// command that sets origin55. Fluent, correctly formatted, and teaching
    /// the model that the two are the same thing.
    ///
    /// Mechanical, from this tool's own vocabulary: every flag name the
    /// question mentions must be one the answer actually sets. The
    /// command's own store noun is exempt - "graft the anchor store" names
    /// the store, not `--anchor`.
    pub fn mismatch(&self, question: &str, answer: &str) -> Option<String> {
        // Hyphens are part of a flag name, not punctuation: splitting on
        // them turns `--force-label` into two words that match nothing, and
        // every compound-named flag stops being checked.
        let words: BTreeSet<String> = question
            .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
            .map(|w| w.trim_matches('-'))
            .filter(|w| !w.is_empty())
            .map(str::to_lowercase)
            .collect();

        // A raw placeholder in a question is never something a user would
        // say: it has been copied out of the manual's syntax.
        for p in PLACEHOLDERS {
            if question.split_whitespace().any(|w| w.trim_matches(|c: char| !c.is_ascii_alphanumeric()) == *p) {
                return Some(format!("the question quotes the placeholder {p:?} instead of a value"));
            }
        }

        // A command is an instruction to DO something, so the question has
        // to ASK for one. Stated as what a question must contain rather
        // than as a list of what it must not: the ways of asking what
        // something IS are endless - "what is the purpose of", "how does it
        // work", "what action does it perform", "what is the default" - and
        // a list of them is a list that is always one phrasing out of date.
        let lower = question.to_lowercase();
        const ASKS_FOR_A_COMMAND: [&str; 7] =
            ["command", "how do i", "how can i", "how should i", "what flags", "which flags", "what options should"];
        if !ASKS_FOR_A_COMMAND.iter().any(|m| lower.contains(m)) {
            return Some("does not ask for a command, so a command does not answer it".to_string());
        }

        let Ok((cmd, given)) = self.parse_into(answer) else {
            return Some("the answer does not parse".to_string());
        };
        let set: BTreeSet<String> = given.keys().map(|k| k.trim_start_matches('-').to_lowercase()).collect();
        let every_flag: BTreeSet<String> =
            self.commands.iter().flat_map(|c| c.flags.iter()).map(|f| f.name.trim_start_matches('-').to_lowercase()).collect();

        // The store noun is exempt only where it NAMES the store - "graft
        // the anchor store". "set the anchor path" is about the flag, and
        // exempting it there let a genuinely mismatched row through.
        let store = noun_of(&cmd.summary);
        let names_the_store = lower.contains(&format!("{store} store"));
        // A subcommand's own name is not a flag mention. `quota-hoist`
        // shares a word with `--quota`, so a question naming the command
        // read as one asking for the flag - and an answer that set it was
        // then accepted for a question that never asked.
        // The command's own words, and only where the question actually
        // NAMES the command - the same rule the store noun gets. Exempting
        // them everywhere fixed `quota-hoist` being read as `--quota` and
        // immediately broke the other direction: "how do I set the quota to
        // 7d?" became a question that had not asked for anything.
        let spoken = cmd.name.replace('-', " ");
        let names_the_command = lower.contains(&spoken) || lower.contains(&cmd.name);
        let command_words: BTreeSet<String> =
            if names_the_command { cmd.name.split('-').map(str::to_lowercase).collect() } else { BTreeSet::new() };
        // Subtracted once, so BOTH directions agree about what was asked
        // for. Filtering only the forward check left the reverse one - "the
        // answer sets something the question never asked about" - still
        // reading the command name as a request.
        let asked: BTreeSet<&String> = words.iter().filter(|w| !command_words.contains(*w)).collect();

        for w in &asked {
            // A hyphenated token is this tool's flag shape. One that is not
            // a flag at all is a name the model invented - `--hard-cutoff`
            // for a tool that has no such thing - and a question built on
            // invented terminology teaches the model that it exists. The
            // earlier check could not see these: it only compared against
            // REAL flag names, so a hallucinated one passed untouched.
            if w.contains('-') && !every_flag.contains(*w) {
                return Some(format!("the question names {w:?}, which is not a flag this tool has"));
            }
            if !every_flag.contains(*w) || set.contains(*w) {
                continue;
            }
            if **w == store && names_the_store {
                continue;
            }
            return Some(format!("the question asks about {w:?}, which the answer does not set"));
        }

        // And the other direction. An answer that sets an OPTIONAL flag the
        // question never asked for is teaching the model to volunteer it.
        // Required flags are exempt: they are part of running the command at
        // all, and no user asks for them by name.
        for (flag, _) in given.iter() {
            let bare = flag.trim_start_matches('-').to_lowercase();
            let required = cmd.flags.iter().any(|f| f.name == *flag && f.required);
            if !required && !asked.contains(&bare) {
                return Some(format!("the answer sets {bare:?}, which the question never asked for"));
            }
        }
        None
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

/// One flag as it is written on a command line, with a real value where the
/// flag wants one.
fn spec_of(f: &Flag) -> String {
    match &f.value {
        Some(v) => format!("{} {}", f.name, value_for(v)),
        None => f.name.clone(),
    }
}

/// What a placeholder accepts, in the words the manual uses to teach it.
///
/// The one statement of each format. `value_is` enforces exactly these and
/// `value_for` produces one of each, so what the manual promises, what the
/// parser accepts and what an example shows cannot drift apart.
fn value_spec(placeholder: &str) -> String {
    match placeholder {
        "PATH" => "an absolute path, beginning with a slash, like /var/data".to_string(),
        "COUNT" => "a whole number, like 16".to_string(),
        "TIME" => "a whole number and a unit, one of s m h d, like 7d or 30s".to_string(),
        "SIZE" => "a whole number and a unit, one of K M G, like 64M".to_string(),
        "NAME" => "a word of letters, digits, dashes or underscores, like main".to_string(),
        other => other.to_string(),
    }
}

/// The noun a command's summary names ("graft the anchor store" -> "anchor"),
/// so a transcript says which store it acted on.
fn noun_of(summary: &str) -> String {
    let words: Vec<&str> = summary.split_whitespace().collect();
    words.iter().position(|w| *w == "store").and_then(|i| i.checked_sub(1)).map(|i| words[i].to_string()).unwrap_or_else(|| "default".to_string())
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
    // Stated as a correction to the page it contradicts, which is both how
    // a real one arrives and what gives the lane enough text for the reader
    // to judge it on its content rather than refuse it for its length.
    let cmd = &a.commands[0];
    let flag = &cmd.flags[0];
    let contradiction = format!(
        "{} {} - corrected notes\nthese notes supersede the pages below.\n\n{}\n{}\ncorrections:\n  {} {}   the {} is now given as an argument\n  {} no longer takes a value\n",
        a.name,
        cmd.name,
        a.man_page(cmd),
        a.man_page(&a.commands[1 % a.commands.len()]),
        flag.name,
        "ALWAYS",
        flag.name,
        cmd.flags[1].name
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
    // Several pairs rather than one: a single pair is a trap the model can
    // pass by chance, and one pair is also far too little text for the
    // reader to judge at all.
    const OPPOSITES: [(&str, &str, &str); 5] = [
        ("window", "before", "behind"),
        ("bound", "under", "over"),
        ("edge", "leading", "trailing"),
        ("side", "inner", "outer"),
        ("end", "head", "tail"),
    ];
    let mut cf = format!("{} {} - windows and bounds\nusage: {} {} [OPTIONS]\noptions:\n", a.name, a.commands[1].name, a.name, a.commands[1].name);
    for (noun, lo, hi) in OPPOSITES {
        cf.push_str(&format!("  --{noun}-{lo} TIME   act on entries older than TIME\n"));
        cf.push_str(&format!("  --{noun}-{hi} TIME   act on entries newer than TIME\n"));
    }
    cf.push_str("each pair differs in one word and means the opposite thing; a model that reads the shape and not the word will get half of them backwards.\n");
    put("counterfact/a-windows.txt", &cf, Expect::Either, &mut labels)?;

    // rare: one subcommand documented here and never mentioned again. It
    // must still be invokable at the end of the run.
    let rare = a.commands.last().expect("a tool has commands");
    let mut seldom = format!("{} - a command line tool\nusage: {} <command> [OPTIONS]\ncommands: {}\n\n", a.name, a.name, a.commands.iter().map(|c| c.name.clone()).collect::<Vec<String>>().join(", "));
    seldom.push_str(&a.man_page(rare));
    seldom.push_str("\nexamples:\n");
    for (i, f) in rare.flags.iter().enumerate() {
        let spec = match &f.value {
            Some(v) => format!("{} {}", f.name, value_for(v)),
            None => f.name.clone(),
        };
        seldom.push_str(&format!("  {} {} {spec}\n      {} - case {i}\n", a.name, rare.name, f.summary));
    }
    seldom.push_str(&format!(
        "\nthis subcommand appears once in the whole manual. at the end of the run {} {} must still be invokable.\n",
        a.name, rare.name
    ));
    put("rare/a-seldom.txt", &seldom, Expect::Promote, &mut labels)?;

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

    /// A corpus whose documents are shorter than the reader's own floor
    /// tests the floor rather than the reader. The structure screen cannot
    /// judge text below `brain::MIN_EPISODE_CHARS` at all, so a lane written
    /// under it is refused as too short whatever it holds - and a lane
    /// LABELLED promote that is too short to judge makes the selftest fail
    /// for a reason that has nothing to do with the model.
    ///
    /// Every lane, not only the learnable ones: a refusal lane must be
    /// refused for the reason it is about, not for its length.
    #[test]
    fn every_generated_document_clears_the_readers_own_length_floor() {
        let dir = std::env::temp_dir().join(format!("sample-reader-lengths-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        // Across several seeds: the tools are drawn per seed, so a lane can
        // clear the floor at the seed it was written at and fall under it at
        // the next one.
        let mut short = Vec::new();
        for seed in 1..6u64 {
            let a = Tool::generate(seed);
            let b = Tool::generate_disjoint(seed + 1, &[&a]);
            let labels = write(&dir, &a, &b).expect("corpus");
            for rel in labels.keys() {
                let text = std::fs::read_to_string(dir.join(rel)).expect("episode");
                let chars = text.chars().count();
                if chars < brain::MIN_EPISODE_CHARS {
                    short.push(format!("seed {seed}: {rel} ({chars} chars)"));
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
        assert!(short.is_empty(), "these lanes are under the reader's {} char floor: {short:?}", brain::MIN_EPISODE_CHARS);
    }

    /// The strictness the whole verification rests on. A tool that took any
    /// word as any value would report `--depth banana` as a correct answer,
    /// and every number built on it would be inflated.
    #[test]
    fn a_value_of_the_wrong_shape_is_refused_and_the_right_shape_is_not() {
        let t = tool();
        // One command with a valued flag of each placeholder kind the
        // generator can draw, exercised through the real parser.
        for cmd in &t.commands {
            for f in cmd.flags.iter().filter(|f| f.value.is_some()) {
                let want = f.value.clone().expect("valued");
                let required: Vec<String> = cmd
                    .flags
                    .iter()
                    .filter(|r| r.required && r.name != f.name)
                    .map(|r| match &r.value {
                        Some(v) => format!("{} {}", r.name, value_for(v)),
                        None => r.name.clone(),
                    })
                    .collect();
                let base = format!("{} {} {}", t.name, cmd.name, required.join(" "));
                let good = format!("{base} {} {}", f.name, value_for(&want));
                let bad = format!("{base} {} banana!!", f.name);
                if t.parse(&good).is_err() {
                    continue; // an exclusivity rule collided; not this test's subject
                }
                match t.parse(&bad) {
                    Err(Invalid::BadValue { wants, .. }) => assert_eq!(wants, want),
                    // NAME genuinely accepts most words; it is the one
                    // placeholder a nonsense token can legitimately satisfy.
                    other => assert_eq!(want, "NAME", "{} {} took {bad:?}: {other:?}", cmd.name, f.name),
                }
            }
        }
    }

    /// The same flag twice is refused rather than silently reduced to one.
    #[test]
    fn a_repeated_flag_is_refused() {
        let t = tool();
        let cmd = &t.commands[0];
        let req = cmd.flags.iter().find(|f| f.required).expect("one required flag");
        let spec = match &req.value {
            Some(v) => format!("{} {}", req.name, value_for(v)),
            None => req.name.clone(),
        };
        let twice = format!("{} {} {spec} {spec}", t.name, cmd.name);
        assert!(matches!(t.parse(&twice), Err(Invalid::RepeatedFlag(_))), "got {:?}", t.parse(&twice));
    }

    /// The oracle: running is what decides, and it decides on MEANING. Flag
    /// order is not meaning; an extra flag is.
    #[test]
    fn running_the_same_command_two_ways_gives_one_result_and_a_different_command_does_not() {
        let t = tool();
        let cmd = t.commands.iter().find(|c| c.flags.iter().filter(|f| !f.required).count() >= 2).expect("a command with spare flags");
        let req = cmd.flags.iter().find(|f| f.required).expect("required");
        let req_spec = match &req.value {
            Some(v) => format!("{} {}", req.name, value_for(v)),
            None => req.name.clone(),
        };
        let spare: Vec<&Flag> = cmd.flags.iter().filter(|f| !f.required && f.value.is_none()).take(2).collect();
        if spare.len() < 2 {
            return;
        }
        let a = format!("{} {} {req_spec} {} {}", t.name, cmd.name, spare[0].name, spare[1].name);
        let b = format!("{} {} {} {} {req_spec}", t.name, cmd.name, spare[1].name, spare[0].name);
        let less = format!("{} {} {req_spec} {}", t.name, cmd.name, spare[0].name);
        if t.parse(&a).is_err() || t.parse(&less).is_err() {
            return; // exclusivity collision; the pair above is not this test's subject
        }
        assert_eq!(t.run(&a).expect("runs"), t.run(&b).expect("runs"), "flag order is not meaning");
        assert_ne!(t.run(&a).expect("runs"), t.run(&less).expect("runs"), "an extra flag IS meaning");
    }

    /// Running is safe on anything: the tool has no state outside its own
    /// return value, which is what lets a model's guess be executed at all.
    #[test]
    fn running_an_invalid_invocation_reports_why_and_does_nothing_else() {
        let t = tool();
        assert!(t.run("rm -rf /").is_err());
        assert!(t.run("").is_err());
        let ok = t.run(&t.canonical(&t.commands[0])).expect("the canonical invocation runs");
        assert!(ok.lines.first().is_some_and(|l| l.starts_with(&t.name)), "{:?}", ok.lines);
    }

    /// The corpus has to TEACH the task, or a refusal measures the corpus
    /// rather than the model. Two halves.
    ///
    /// Every placeholder a command uses is explained in words, with a worked
    /// value - `--depth TIME` alone never says what a TIME is, and a model
    /// reading it writes `--depth 5`, which the tool then refuses.
    ///
    /// And what the manual promises, the parser accepts: the example value
    /// for each placeholder is run through the real `value_is`, so the
    /// documentation and the grammar cannot drift apart.
    #[test]
    fn the_manual_explains_every_value_format_it_uses_and_the_parser_agrees() {
        let t = tool();
        for cmd in &t.commands {
            let page = t.man_page(cmd);
            for f in cmd.flags.iter().filter_map(|f| f.value.as_deref()) {
                assert!(page.contains(&format!("{f} is ")), "{} does not say what a {f} is:\n{page}", cmd.name);
                assert!(
                    value_is(f, &value_for(f)),
                    "the manual's own example value for {f} ({:?}) is one the parser refuses",
                    value_for(f)
                );
            }
        }
    }

    /// Worked examples demonstrate the SHAPE of a command line, which is the
    /// other half of what the manual has to teach. None of them may be the
    /// canonical invocation itself: that is the capability battery's answer,
    /// and a manual printing it would be teaching the test.
    #[test]
    fn every_example_runs_and_none_of_them_is_the_batterys_own_answer() {
        let t = tool();
        for cmd in &t.commands {
            let canonical = t.canonical(cmd);
            let examples = t.examples(cmd);
            assert!(!examples.is_empty(), "{} has no worked example", cmd.name);
            for (invocation, _) in &examples {
                assert!(t.run(invocation).is_ok(), "the manual shows {invocation:?}, which does not run: {:?}", t.run(invocation));
                assert_ne!(invocation, &canonical, "an example must not BE the answer the battery asks for");
            }
        }
    }

    /// The four ways a pairing goes wrong that the answer being correct does
    /// nothing about.
    ///
    /// Derived from the tool, so the test keeps testing when the generator's
    /// vocabulary changes.
    #[test]
    fn a_pairing_is_refused_when_the_question_and_the_answer_are_about_different_things() {
        let t = Tool::generate(1);
        let cmd = t.commands.iter().find(|c| c.flags.iter().filter(|f| !f.required).count() >= 2).expect("spare flags");
        let canonical = t.canonical(cmd);
        let store = noun_of(&cmd.summary);
        let action = cmd.name.replace('-', " ");
        let spare: Vec<&Flag> = cmd.flags.iter().filter(|f| !f.required).take(2).collect();
        let unmentioned = spare[0].name.trim_start_matches('-');

        // The fair baseline: asks for the command, names only the store.
        assert_eq!(t.mismatch(&format!("What is the command to {action} the {store} store?"), &canonical), None);

        // Asks what something IS. The whitelist refuses it for what it does
        // NOT ask, which is what makes this robust to the phrasing.
        for asking in [
            "What is the purpose of sealing?",
            "How does the flag work?",
            "What action does it perform?",
            "What is the default margin?",
        ] {
            assert!(t.mismatch(asking, &canonical).is_some(), "{asking:?} does not ask for a command");
        }

        // Names a flag the answer does not set.
        let other = t
            .commands
            .iter()
            .flat_map(|c| c.flags.iter())
            .find(|f| !cmd.flags.iter().any(|g| g.name == f.name))
            .expect("another command has other flags");
        let q = format!("What is the command to {action} the {store} store and turn on {}?", other.name.trim_start_matches('-'));
        assert!(t.mismatch(&q, &canonical).is_some(), "{q:?} names a flag {canonical:?} does not set");

        // The answer volunteers an optional flag nobody asked about.
        let volunteered = format!("{canonical} {}", spare[0].name);
        if t.parse(&volunteered).is_ok() && spare[0].value.is_none() {
            let q = format!("What is the command to {action} the {store} store?");
            assert!(t.mismatch(&q, &volunteered).is_some(), "the answer sets {unmentioned:?}, which {q:?} never asked for");
        }

        // A placeholder copied out of the manual is never a real question.
        let valued = cmd.flags.iter().find(|f| f.value.is_some());
        if let Some(f) = valued {
            let p = f.value.clone().expect("valued");
            let q = format!("What is the command to set the {} to {p}?", f.name.trim_start_matches('-'));
            assert!(t.mismatch(&q, &canonical).is_some(), "{q:?} quotes the placeholder");
        }
    }

    /// Two ways a question goes wrong that naming real flags correctly does
    /// not rule out, both from rows a model actually produced.
    #[test]
    fn an_invented_flag_name_and_a_question_about_a_value_are_both_refused() {
        let t = Tool::generate(1);
        let cmd = &t.commands[0];
        let canonical = t.canonical(cmd);

        // A flag this tool does not have. The check that compares against
        // REAL flag names cannot see this one, because it is not one.
        let invented = "What flags are required to set the hard-cutoff?";
        assert!(t.mismatch(invented, &canonical).is_some(), "{invented:?} names a flag that does not exist");

        // Asking for a VALUE, answered with a command.
        let value_q = "What is the default margin for the pruning operation?";
        assert!(t.mismatch(value_q, &canonical).is_some(), "{value_q:?} asks for a default, which a command does not give");
        // And the shapes that DO ask for one are kept.
        let store = noun_of(&cmd.summary);
        let action = cmd.name.replace('-', " ");
        for good in [
            format!("What is the command to {action} the {store} store?"),
            format!("How do I {action} the {store} store?"),
            format!("What flags are required to {action} the {store} store?"),
        ] {
            assert_eq!(t.mismatch(&good, &canonical), None, "{good:?} asks for a command");
        }

        // And a real, well-formed compound flag the answer does set is fine.
        let real = cmd.flags.iter().find(|f| f.name.contains('-')).map(|f| f.name.trim_start_matches('-').to_string());
        if let Some(flag) = real {
            let spec = cmd.flags.iter().find(|f| f.name.trim_start_matches('-') == flag).expect("found above");
            let answer = if spec.required { canonical.clone() } else { format!("{canonical} {}", spec_of(spec)) };
            if t.parse(&answer).is_ok() {
                let q = format!("What is the command to turn on {flag}?");
                assert_eq!(t.mismatch(&q, &answer), None, "{q:?} names a real flag the answer sets");
            }
        }
    }

    /// A subcommand whose name shares a word with a flag: naming the command
    /// is not asking for the flag.
    ///
    /// Found by reading the rows - `quota-hoist` and `--quota` - and it let
    /// an answer through that set a flag its question never asked about.
    #[test]
    fn a_command_name_that_contains_a_flag_word_is_not_read_as_asking_for_the_flag() {
        let t = Tool::generate(1);
        let Some(cmd) = t
            .commands
            .iter()
            .find(|c| c.name.split('-').any(|w| c.flags.iter().any(|f| f.name.trim_start_matches('-') == w)))
        else {
            return; // this seed drew no such collision
        };
        let shared = cmd.name.split('-').find(|w| cmd.flags.iter().any(|f| f.name.trim_start_matches('-') == *w)).expect("found above");
        let flag = cmd.flags.iter().find(|f| f.name.trim_start_matches('-') == shared).expect("found above");
        if flag.required {
            return; // a required flag is exempt anyway; not this test's case
        }
        let action = cmd.name.replace('-', " ");
        let store = noun_of(&cmd.summary);
        let question = format!("What is the command to {action} the {store} store?");
        let over = format!("{} {}", t.canonical(cmd), spec_of(flag));
        if t.parse(&over).is_ok() {
            assert!(
                t.mismatch(&question, &over).is_some(),
                "{question:?} names the command, not the flag, so {over:?} sets {shared:?} unasked"
            );
        }
    }

    /// No flag may share a word with the command it belongs to.
    ///
    /// `quota-hoist` beside `--quota` cannot be read unambiguously: "set the
    /// quota" and "quota hoist the weight store" name different things with
    /// the same word, and a checker resolving it one way gets the other
    /// wrong. Removed at the source rather than parsed around.
    #[test]
    fn no_flag_shares_a_word_with_its_own_command() {
        for seed in 1..8u64 {
            let a = Tool::generate(seed);
            let b = Tool::generate_disjoint(seed + 1, &[&a]);
            for t in [&a, &b] {
                for cmd in &t.commands {
                    let words: BTreeSet<&str> = cmd.name.split('-').collect();
                    for f in &cmd.flags {
                        for part in f.name.trim_start_matches('-').split('-') {
                            assert!(
                                !words.contains(part),
                                "seed {seed}: {} {} has {} - the word {part:?} means two things",
                                t.name,
                                cmd.name,
                                f.name
                            );
                        }
                    }
                }
            }
        }
    }

    /// A made-up tool has to read like a real one. Names disambiguated with
    /// a counter - `--origin55`, `--weight44` - are not flags anybody has
    /// ever typed, and a corpus full of them is teaching the model a shape
    /// it will never see again.
    #[test]
    fn no_generated_name_is_disambiguated_with_a_number() {
        for seed in 1..8u64 {
            let a = Tool::generate(seed);
            let b = Tool::generate_disjoint(seed + 1, &[&a]);
            for t in [&a, &b] {
                for cmd in &t.commands {
                    assert!(
                        !cmd.name.chars().any(|c| c.is_ascii_digit()),
                        "seed {seed}: subcommand {:?} is a counter, not a name",
                        cmd.name
                    );
                    for f in &cmd.flags {
                        assert!(
                            !f.name.chars().any(|c| c.is_ascii_digit()),
                            "seed {seed}: flag {:?} is a counter, not a name",
                            f.name
                        );
                    }
                }
            }
        }
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

#[cfg(test)]
mod oracle {
    use super::*;

    /// The oracle, and the two cases that separate EXECUTION from parsing.
    ///
    /// Built from the tool rather than from remembered strings: the cases
    /// that matter are the CATEGORIES of wrong answer, and a test pinned to
    /// one generated vocabulary stops testing anything the moment the
    /// generator improves.
    #[test]
    fn running_a_candidate_separates_correct_from_merely_valid() {
        let t = Tool::generate(1);
        let cmd = t
            .commands
            .iter()
            .find(|c| c.flags.iter().any(|f| !f.required && f.value.is_some()))
            .expect("a command with an optional valued flag");
        let canonical = t.canonical(cmd);
        let spare = cmd.flags.iter().find(|f| !f.required && f.value.is_some()).expect("checked above");

        let verdict = |candidate: &str| -> String {
            match (t.run(&canonical), t.run(candidate)) {
                (Ok(want), Ok(got)) if want == got => "correct".to_string(),
                (Ok(_), Ok(_)) => "valid but different".to_string(),
                (Ok(_), Err(e)) => format!("{e:?}"),
                (Err(e), _) => panic!("the reference itself is invalid: {e:?}"),
            }
        };

        assert_eq!(verdict(&canonical), "correct");

        // Valid, runs, and is not the command that was asked for. Only
        // running it says so - it parses exactly as well as the canonical.
        let extra = format!("{canonical} {} {}", spare.name, value_for(spare.value.as_deref().expect("valued")));
        assert!(t.parse(&extra).is_ok(), "the premise: {extra:?} is a perfectly valid invocation");
        assert_eq!(verdict(&extra), "valid but different");

        // A value of the wrong shape, which a tool that took any word would
        // have called correct.
        let bad = format!("{canonical} {} banana!!", spare.name);
        assert!(matches!(t.run(&bad), Err(Invalid::BadValue { .. })), "{:?}", t.run(&bad));

        // Not this tool at all.
        assert_eq!(verdict(&format!("not{canonical}")), "NotThisTool");

        // The required flags missing.
        let bare = format!("{} {}", t.name, cmd.name);
        assert!(matches!(t.run(&bare), Err(Invalid::MissingRequired(_))), "{:?}", t.run(&bare));
    }
}
