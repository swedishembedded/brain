// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A model that reads a directory continually and can say what it gained.
//!
//! ```text
//! sample-learning-reader generate --corpus corpus/          # invent a tool
//! sample-learning-reader battery  --model M --run-dir run/  # 2/5, frozen
//! sample-learning-reader read     --corpus corpus/ --run-dir run/
//! sample-learning-reader battery  --model M --run-dir run/  # 5/5
//! sample-learning-reader report   --run-dir run/
//! sample-learning-reader selftest --corpus corpus/ --run-dir run/
//! ```
//!
//! The corpus is a command-line tool that does not exist until `generate`
//! draws it from a seed, so the model's starting score is provably zero
//! rather than argued. The judge is that tool's own grammar parser, never a
//! second model's opinion.
//!
//! `battery` is the point. `read`'s promote rate is plumbing; the change in
//! the battery between two calls is the ability the user gained.
//!
//! Swedish Embedded AB builds continual-learning systems whose claims are
//! checkable by something other than the system making them. If your team
//! needs a model that keeps learning from your own material and can show
//! what that bought and what it cost, you can procure our services by
//! sending an email to info@swedishembedded.com.

mod corpus;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::ExitCode;

use brain::artifact;
use brain::qa::{self, Candidate, Distil, Known, Passage, Phrase, Verdict};
use brain::qa::Split;
use brain::{BatteryTask, ContinualReader, LedgerFacts, TextGenerationPipeline};
use corpus::{Expect, Tool};

const USAGE: &str = "\
usage: sample-learning-reader <verb> [options]

  generate --corpus DIR [--seed N]
        invent two tools and write the corpus, its adversarial lanes and
        their labels. Nothing else needs preparing.

  battery  --model REF --run-dir DIR [--seed N]
        score the held-out tasks against what is currently served. Run it
        before reading and again after: the difference is the result.

  read     --corpus DIR --run-dir DIR --model REF [--until N] [--seed N]
           [--null-gate N | --shuffled-labels N]
        read the corpus, deciding per episode what is worth learning.
        Resumes if the run directory has been read before.

        --null-gate N        control arm: a coin with seed N decides what
                             carries forward. The real gate still runs and
                             the ledger still records it, so this arm's
                             promote rate is comparable with the real one's.
        --shuffled-labels N  control arm: each episode is trained on its own
                             rows and gated against ANOTHER episode's frozen
                             probes. It should promote at chance.

        Run a control into its OWN run directory: it is a different arm, not
        a continuation of the real one.

  distil   --corpus DIR --out DIR --model REF [--seed N] [--per-passage N]
        turn the corpus into question/answer pairs the model can be TRAINED
        to answer, and keep only the ones whose answer actually runs.

        Writes one artefact per stage into --out, so every step can be read
        rather than trusted: passages.jsonl, replies.jsonl (what the model
        said about each passage, including the ones that yielded nothing),
        candidates.jsonl (the pairs read out of those), checked.jsonl (what
        the tool said about each) and accepted.jsonl (what survived).

  phrase   --corpus DIR --out DIR --model REF [--seed N] [--per-answer N]
           [--passes N] [--temperature T]
        the distil stage done the other way round, and the one that yields
        a training set: the TOOL supplies the answers, which makes them
        correct by construction, and the model only phrases the questions.

  sft      --in DIR --out DIR [--held-out N] [--seed N] [--control]
        turn the verified pairs into a training set and the held-out
        questions that will judge it. Splits by PHRASING, so the probe asks
        for something the training half teaches, in words it never used.

        --control  write the SHUFFLED-ANSWER arm instead: the same questions,
                   the same answers, the same row count, every pairing wrong.
                   A probe that scores as well from this is not measuring the
                   mapping.

  score    --probes DIR --model REF [--adapter PATH] [--max-new N]
        ask the held-out questions and RUN each answer. A reply counts only
        when running it produces the same observable result as running the
        reference - the measurement the whole pipeline exists to make.

  audit    --sft DIR [--seed N]
        prove the splits are what they claim before anything is trained on
        them: no question shared between halves, no answer the training half
        never teaches, every answer executable, and the probe genuinely
        unseen. Exits non-zero on any of it.

  report   --run-dir DIR
        every episode the run recorded, and where each one stopped.

  selftest --corpus DIR --run-dir DIR
        check each episode's verdict against its lane's label, and exit
        non-zero if the reader learned something it should have refused.

The model is named by --model and resolved by brain's own model handler, so
a reference that is not on disk is fetched. Without one, `generate`,
`report` and `selftest` still work.
";

/// A tiny flag reader. A sample parses its own arguments rather than sharing
/// the engine's parser: it is a standalone application whose only brain
/// dependency is the SDK.
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
    fn path(&mut self, flag: &str, default: &str) -> PathBuf {
        PathBuf::from(self.take(flag).unwrap_or_else(|| default.to_string()))
    }
    fn number(&mut self, flag: &str, default: u64) -> Result<u64, String> {
        match self.take(flag) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|e| format!("{flag}: {e}")),
        }
    }
    /// Refuse anything the verb did not understand. Every flag this sample
    /// has selects which run, which seed or which corpus a number is about,
    /// so one that fell through to a default would report a believable
    /// number about a different run than the one asked for.
    fn finish(&self) -> Result<(), String> {
        if self.0.is_empty() {
            return Ok(());
        }
        Err(format!("not understood: {}\n\n{USAGE}", self.0.join(" ")))
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() || argv.iter().any(|a| a == "-h" || a == "--help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match run(argv) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("sample-learning-reader: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(argv: Vec<String>) -> Result<ExitCode, String> {
    let mut a = Args(argv);
    let verb = a.0.remove(0);
    match verb.as_str() {
        "generate" => generate(&mut a),
        "battery" => battery(&mut a),
        "read" => read(&mut a),
        "distil" => distil(&mut a),
        "phrase" => phrase(&mut a),
        "sft" => sft(&mut a),
        "score" => score(&mut a),
        "audit" => audit(&mut a),
        "report" => report(&mut a),
        "selftest" => selftest(&mut a),
        other => Err(format!("unknown verb {other:?}\n\n{USAGE}")),
    }
}

/// The invocation inside a chat model's answer.
///
/// An instruction-tuned model answers a question with a sentence and then
/// the command ("The command to splice the budget store is:\n\nhask0 splice
/// --anchor 16"), so taking the first line scores the preamble and reports
/// the model got it wrong. The tool's own parser decides which line is the
/// answer: the first one it accepts.
///
/// Falling back to the first non-empty line when none parses is what keeps
/// a wrong answer visible as what the model actually wrote, rather than as
/// an empty string.
fn command_in(answer: &str, tool: &Tool) -> String {
    // A chat model asked for a command very often fences it. The fence is
    // presentation, not part of the answer, and leaving it in makes every
    // such answer unparseable for a reason that is not about the command.
    let lines: Vec<&str> = answer
        .lines()
        .map(|l| l.trim().trim_start_matches("```bash").trim_start_matches("```sh").trim_start_matches("```").trim_end_matches("```").trim())
        .filter(|l| !l.is_empty())
        .collect();
    lines
        .iter()
        .find(|l| tool.parse(l).is_ok())
        .or_else(|| lines.first())
        .map(|l| l.to_string())
        .unwrap_or_default()
}

/// The two tools a run uses. Derived from one seed so `generate`, `battery`
/// and `selftest` agree about which tools they are talking about without
/// having to pass them between processes.
fn tools(seed: u64) -> (Tool, Tool) {
    let a = Tool::generate(seed);
    let b = Tool::generate_disjoint(seed.wrapping_add(1), &[&a]);
    (a, b)
}

fn generate(a: &mut Args) -> Result<ExitCode, String> {
    let dir = a.path("--corpus", "corpus");
    let seed = a.number("--seed", 1)?;
    a.finish()?;
    let (ta, tb) = tools(seed);
    let labels = corpus::write(&dir, &ta, &tb).map_err(|e| format!("{}: {e}", dir.display()))?;
    println!("invented {} and {}, {} subcommands between them", ta.name, tb.name, ta.commands.len() + tb.commands.len());
    println!("wrote {} episodes across {} lanes to {}", labels.len(), 9, dir.display());
    println!("battery: {} held-out tasks, none of which appear in the corpus", ta.battery().len());
    Ok(ExitCode::SUCCESS)
}

fn battery_tasks(seed: u64) -> Vec<BatteryTask> {
    let (ta, _) = tools(seed);
    ta.battery().into_iter().map(|(prompt, expected)| BatteryTask { prompt, expected }).collect()
}

fn battery(a: &mut Args) -> Result<ExitCode, String> {
    let model = a.take("--model").ok_or("battery needs --model")?;
    let run_dir = a.path("--run-dir", "run");
    let seed = a.number("--seed", 1)?;
    a.finish()?;
    let tasks = battery_tasks(seed);

    let reader = ContinualReader::from_pretrained(&model).run_dir(&run_dir).seed(seed);
    let score = reader.battery(&tasks).map_err(|e| e.to_string())?;
    let (ta, _) = tools(seed);

    // Two numbers, because they answer different questions. Exact match is
    // "did it write the canonical answer"; acceptance is "would the tool
    // have run it", which is the ability a user actually gains and is
    // strictly the looser of the two. Reporting only the first would call a
    // correct invocation with the flags in another order a failure.
    let accepted = score.answers.iter().filter(|a| ta.parse(&command_in(a, &ta)).is_ok()).count();
    println!("capability {}/{} exact, {}/{} accepted by {}", score.passed, score.total, accepted, score.total, ta.name);
    for ((t, ok), answer) in tasks.iter().zip(&score.per_task).zip(&score.answers) {
        let got = command_in(answer, &ta);
        let verdict = match (ok, ta.parse(&got)) {
            (true, _) => "pass".to_string(),
            (false, Ok(())) => "valid, not canonical".to_string(),
            (false, Err(why)) => format!("{why:?}"),
        };
        println!("  {verdict:<28} want {:<40} got {got:?}", t.expected);
        // The whole answer when no line of it was a command. A run that only
        // ever shows the line the parser rejected cannot tell "the model
        // wrote prose" from "the model wrote a command this tool does not
        // have", and those are different failures.
        if ta.parse(&got).is_err() {
            println!("       full answer: {:?}", answer.trim());
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn read(a: &mut Args) -> Result<ExitCode, String> {
    let model = a.take("--model").ok_or("read needs --model")?;
    let corpus_dir = a.path("--corpus", "corpus");
    let run_dir = a.path("--run-dir", "run");
    let seed = a.number("--seed", 1)?;
    let until = a.take("--until").map(|v| v.parse::<usize>().map_err(|e| format!("--until: {e}"))).transpose()?;
    let null_gate = a.take("--null-gate").map(|v| v.parse::<u64>().map_err(|e| format!("--null-gate: {e}"))).transpose()?;
    let shuffled = a.take("--shuffled-labels").map(|v| v.parse::<u64>().map_err(|e| format!("--shuffled-labels: {e}"))).transpose()?;
    a.finish()?;
    // One arm at a time. Two controls at once answers neither question, and
    // silently honouring the first would report the run as the other.
    if null_gate.is_some() && shuffled.is_some() {
        return Err("--null-gate and --shuffled-labels are different control arms; run one at a time, into its own run directory".to_string());
    }

    let mut reader = ContinualReader::from_pretrained(&model).run_dir(&run_dir).corpus(&corpus_dir).seed(seed);
    if let Some(n) = until {
        reader = reader.until(n);
    }
    if let Some(s) = null_gate {
        reader = reader.null_gate(s);
    }
    if let Some(s) = shuffled {
        reader = reader.shuffled_labels(s);
    }
    let arm = match (null_gate, shuffled) {
        (Some(s), _) => format!("null-gate arm (coin seed {s})"),
        (_, Some(s)) => format!("shuffled-labels arm (seed {s})"),
        _ => "real arm".to_string(),
    };
    println!("{arm}");
    let out = reader.read().map_err(|e| e.to_string())?;

    println!("read {} episodes, promoted {}", out.episodes, out.promoted);
    println!("bank {} earlier episodes, detection latency {} episodes", out.bank, out.detection_latency);
    println!("what that means: any regression larger than the per-block bar is found within {} episodes.", out.detection_latency);

    let (learned, revisited) = out.retention_coverage;
    match out.bwt {
        Some(bwt) => {
            let worst = out.worst_block_drop.unwrap_or(0.0);
            println!("bwt {bwt:+.4} over {revisited} of {learned} learned episodes revisited, worst single drop {worst:.4}");
        }
        // Said rather than left out. A run whose schedule never came back
        // round has no backward transfer, and printing nothing there is how
        // an unanswered clause gets read as a passed one.
        None => println!("bwt UNMEASURED: {learned} episodes learned, none revisited yet - not a backward transfer of zero"),
    }
    Ok(ExitCode::SUCCESS)
}

/// Split a document into the units a question can be asked about.
///
/// The corpus's manual pages are one subcommand each, separated by a blank
/// line, so that is the unit. Reading a particular corpus's shape is the
/// sample's job precisely because it is particular: the SDK cannot guess it,
/// and guessing wrong makes passages no question can be answered from.
fn passages_of(text: &str, source: &str, tool: &Tool) -> Vec<Passage> {
    text.split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .filter_map(|p| {
            // The subcommand a page documents is its identity, and the thing
            // an answer drawn from it has to be about.
            let head = p.lines().next().unwrap_or_default();
            let name = head.split_whitespace().nth(1)?;
            tool.command(name)?;
            Some(Passage { id: name.to_string(), source: source.to_string(), text: p.to_string() })
        })
        .collect()
}

/// The oracle: run the answer, and require it to be about the passage it came
/// from.
///
/// Two ways a generated pair fails, and they are different failures. An
/// answer the tool refuses is a model that wrote something that is not a
/// command. An answer that runs but invokes a DIFFERENT subcommand is a model
/// that answered a question about material it was not shown - fluent, valid,
/// and not grounded in the passage.
fn verify(tool: &Tool, c: &Candidate) -> Verdict {
    let command = command_in(&c.answer, tool);
    match tool.run(&command) {
        Err(why) => Verdict::Reject { reason: format!("{why:?}") },
        Ok(outcome) => {
            let invoked = command.split_whitespace().nth(1).unwrap_or_default();
            if invoked != c.passage {
                return Verdict::Reject {
                    reason: format!("runs, but invokes {invoked:?} and the passage is about {:?}", c.passage),
                };
            }
            Verdict::Accept { evidence: outcome.lines.last().cloned().unwrap_or_default() }
        }
    }
}

fn distil(a: &mut Args) -> Result<ExitCode, String> {
    let corpus_dir = a.path("--corpus", "corpus");
    let out_dir = a.path("--out", "distil");
    let model = a.take("--model").ok_or("distil needs --model")?;
    let seed = a.number("--seed", 1)?;
    let per_passage = a.number("--per-passage", 4)? as usize;
    a.finish()?;

    let (tool, _) = tools(seed);

    // Stage 1: the source material, cut into units a question can be asked
    // about. Only the lanes that document the tool - the adversarial lanes
    // exist to be refused by the reader, not to be asked about.
    let mut passages = Vec::new();
    for rel in ["learn/a-manual.txt", "learn/b-manual.txt", "rare/a-seldom.txt"] {
        let path = corpus_dir.join(rel);
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        passages.extend(passages_of(&text, rel, &tool));
    }
    if passages.is_empty() {
        return Err(format!("{}: no manual pages to ask about", corpus_dir.display()));
    }
    artifact::write_jsonl(out_dir.join("passages.jsonl"), &passages).map_err(|e| e.to_string())?;
    println!("passages  {:>4}  from {}", passages.len(), corpus_dir.display());

    // Stage 2: what the model proposes. Nothing here is believed yet.
    let pipeline = TextGenerationPipeline::from_pretrained(&model).map_err(|e| e.to_string())?;
    let distil = Distil::with(pipeline)
        .seed(seed)
        .per_passage(per_passage)
        .instruction(format!(
            "The reference material above is one subcommand's manual page for a command-line tool called {}. \
             Write questions a user might ask about how to USE that subcommand, and for each the exact \
             command line that answers it, taken only from the material. Every answer must start with {} \
             and must only use flags the material lists. Reply with nothing but pairs in this form:\nQ: <question>\nA: <command>",
            tool.name, tool.name
        ));
    let replies = distil.ask(&passages).map_err(|e| e.to_string())?;
    artifact::write_jsonl(out_dir.join("replies.jsonl"), &replies).map_err(|e| e.to_string())?;
    let candidates: Vec<Candidate> = replies.iter().flat_map(|r| qa::parse_pairs(&r.passage, &r.raw)).collect();
    artifact::write_jsonl(out_dir.join("candidates.jsonl"), &candidates).map_err(|e| e.to_string())?;
    let silent = replies.iter().filter(|r| r.pairs == 0).count();
    println!("proposed  {:>4}  question/answer pairs", candidates.len());
    if silent > 0 {
        // Named, because a passage that yielded nothing is the one worth
        // reading and the easiest one to not notice.
        println!("  {silent} of {} passages yielded none: {}", replies.len(), replies.iter().filter(|r| r.pairs == 0).map(|r| r.passage.as_str()).collect::<Vec<&str>>().join(", "));
    }

    // Stage 3: what the tool says about each. Both outcomes are written: the
    // refusals are the evidence that the checker did anything.
    let checked = qa::check_all(candidates, &|c: &Candidate| verify(&tool, c));
    artifact::write_jsonl(out_dir.join("checked.jsonl"), &checked).map_err(|e| e.to_string())?;
    let (kept, tally) = qa::accepted(&checked);
    artifact::write_jsonl(out_dir.join("accepted.jsonl"), &kept).map_err(|e| e.to_string())?;
    println!("verified  {:>4}  of {} ran and were about their own passage ({:.0}%)", tally.accepted, tally.offered, tally.rate() * 100.0);

    // Why the rest were refused, in the caller's own words, most common
    // first - the number that says whether to fix the generator or the
    // prompt.
    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for c in checked.iter() {
        if let Verdict::Reject { reason } = &c.verdict {
            *reasons.entry(reason.split(" {").next().unwrap_or(reason).to_string()).or_default() += 1;
        }
    }
    let mut ranked: Vec<(String, usize)> = reasons.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    for (reason, n) in ranked.iter().take(8) {
        println!("  refused {n:>3}  {reason}");
    }
    println!("\nwrote {}", out_dir.display());
    if kept.is_empty() {
        println!("nothing survived verification: there is no training set here, and that is the result");
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
}

/// Every command the tool can be asked to run, with the invocation that does
/// it - built by the tool, so correct by construction.
///
/// The canonical invocation of each subcommand, plus one variant per optional
/// flag. Each is RUN here before it is offered, so a defect in the generator
/// is caught before it becomes a training row rather than after.
fn answerable(tool: &Tool) -> Vec<Known> {
    let mut out = Vec::new();
    for cmd in &tool.commands {
        let page = tool.man_page(cmd);
        let mut wanted: Vec<(String, String)> = vec![(tool.canonical(cmd), cmd.summary.clone())];
        for (invocation, what) in tool.examples(cmd) {
            wanted.push((invocation, what));
        }
        for (n, (invocation, intent)) in wanted.into_iter().enumerate() {
            // The tool built it; the tool has to accept it. A goal that does
            // not run is a bug here, not a wrong answer, and it must never
            // reach a training set.
            if tool.run(&invocation).is_err() {
                continue;
            }
            // One id per ANSWER, not per command: a command contributes its
            // canonical invocation and one per optional flag, and an id they
            // shared would make several answers indistinguishable to
            // anything downstream that groups by it.
            out.push(Known { id: format!("{}#{n}", cmd.name), context: page.clone(), answer: invocation, intent });
        }
    }
    out
}

fn phrase(a: &mut Args) -> Result<ExitCode, String> {
    let corpus_dir = a.path("--corpus", "corpus");
    let out_dir = a.path("--out", "phrased");
    let model = a.take("--model").ok_or("phrase needs --model")?;
    let seed = a.number("--seed", 1)?;
    let per_answer = a.number("--per-answer", 3)? as usize;
    // Greedy decoding returns one reply per prompt, so distinct phrasings
    // need sampling and more than one pass. Seeded per pass, so the whole
    // set still reproduces.
    let passes = a.number("--passes", 1)? as usize;
    let temperature = a.take("--temperature").map(|v| v.parse::<f32>().map_err(|e| format!("--temperature: {e}"))).transpose()?;
    a.finish()?;
    let _ = &corpus_dir;

    let (tool, _) = tools(seed);

    // Stage 1: the answers, from the tool rather than from a model.
    let known = answerable(&tool);
    artifact::write_jsonl(out_dir.join("known.jsonl"), &known).map_err(|e| e.to_string())?;
    println!("answers   {:>4}  built by the tool and verified to run", known.len());

    // Stage 2: the questions. This is all the model is trusted with, and it
    // is the half it cannot get wrong in a way that reaches the weights.
    let pipeline = TextGenerationPipeline::from_pretrained(&model).map_err(|e| e.to_string())?;
    let mut phrasing = Phrase::with(pipeline).seed(seed).per_known(per_answer).passes(passes);
    if let Some(t) = temperature {
        phrasing = phrasing.temperature(t);
    }
    let phrasing = phrasing.instruction(format!(
        "Write questions a user of the {} tool would ask, whose answer is exactly the command above. \
         Every question must ASK FOR A COMMAND - begin each one with \"What is the command to\", \
         \"How do I\" or \"What flags are required to\". Never ask what something is, what it does, \
         or what its default is: the answer is a command, so the question must be a request for one. \
         Name only what the command actually sets. Never quote the command itself and never mention {}. \
         Reply with nothing but questions, one per line, each beginning with `Q: `.",
        tool.name, tool.name
    ));
    let replies = phrasing.ask(&known).map_err(|e| e.to_string())?;
    artifact::write_jsonl(out_dir.join("replies.jsonl"), &replies).map_err(|e| e.to_string())?;
    let candidates = phrasing.generate(&known).map_err(|e| e.to_string())?;
    artifact::write_jsonl(out_dir.join("candidates.jsonl"), &candidates).map_err(|e| e.to_string())?;
    println!("questions {:>4}  phrased for them", candidates.len());

    // Stage 3: what is left to check. The answer is correct already, so this
    // is about the QUESTION: it must not give the answer away, and it must
    // actually be a question.
    let checked = qa::check_all(candidates, &|c: &Candidate| {
        if qa::leaks_answer(&c.question, &c.answer) {
            return Verdict::Reject { reason: "the question quotes its own answer".to_string() };
        }
        if !c.question.ends_with('?') {
            return Verdict::Reject { reason: "not a question".to_string() };
        }
        // The answer being correct does not make the PAIRING correct. A
        // model asked for several questions about one command drifts onto
        // the other flags in front of it, and the answer is attached
        // regardless.
        if let Some(why) = tool.mismatch(&c.question, &c.answer) {
            return Verdict::Reject { reason: why };
        }
        match tool.run(&c.answer) {
            Ok(outcome) => Verdict::Accept { evidence: outcome.lines.last().cloned().unwrap_or_default() },
            Err(why) => Verdict::Reject { reason: format!("the ANSWER does not run, which is a defect here: {why:?}") },
        }
    });
    artifact::write_jsonl(out_dir.join("checked.jsonl"), &checked).map_err(|e| e.to_string())?;
    let (verified, tally) = qa::accepted(&checked);
    // Asking for many phrasings gets fewer distinct ones, and a repeat is
    // not new data: the same question can otherwise land in two halves of a
    // split, which turns a held-out score into a recall score.
    let (kept, dup) = qa::dedupe(&verified);
    artifact::write_jsonl(out_dir.join("accepted.jsonl"), &kept).map_err(|e| e.to_string())?;
    println!("training  {:>4}  of {} rows survived ({:.0}%)", tally.accepted, tally.offered, tally.rate() * 100.0);
    if dup.repeated > 0 || !dup.contradictory.is_empty() {
        println!(
            "  deduped {:>3}  repeated question(s){}",
            dup.repeated,
            if dup.contradictory.is_empty() {
                String::new()
            } else {
                format!(", and dropped {} question(s) that answered to two different commands", dup.contradictory.len())
            }
        );
        println!("distinct  {:>4}  rows written", kept.len());
    }

    let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
    for c in checked.iter() {
        if let Verdict::Reject { reason } = &c.verdict {
            *reasons.entry(reason.clone()).or_default() += 1;
        }
    }
    let mut ranked: Vec<(String, usize)> = reasons.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    for (reason, n) in ranked.iter().take(6) {
        println!("  refused {n:>3}  {reason}");
    }
    println!("\nwrote {}", out_dir.display());
    if kept.is_empty() {
        println!("nothing survived: there is no training set here, and that is the result");
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
}

fn sft(a: &mut Args) -> Result<ExitCode, String> {
    let in_dir = a.path("--in", "phrased");
    let out_dir = a.path("--out", "sft");
    let held_out = a.number("--held-out", 1)? as usize;
    let seed = a.number("--seed", 1)?;
    let control = a.0.iter().position(|x| x == "--control").map(|i| a.0.remove(i)).is_some();
    a.finish()?;

    let accepted: Vec<Candidate> =
        artifact::read_jsonl(in_dir.join("accepted.jsonl")).map_err(|e| format!("{}: {e}", in_dir.display()))?;
    if accepted.is_empty() {
        return Err(format!("{}: no verified pairs to build a training set from", in_dir.display()));
    }

    // Deduped again here rather than trusted from upstream: this stage is
    // what a split's honesty rests on, and a repeated question is what makes
    // the same row land in two halves of it.
    let (accepted, dup) = qa::dedupe(&accepted);
    if dup.repeated > 0 || !dup.contradictory.is_empty() {
        println!("deduped   {:>4}  repeated, {} contradictory", dup.repeated, dup.contradictory.len());
    }
    let split: Split = qa::split_by_phrasing(&accepted, held_out, seed);
    // Refused rather than reported: a split with either defect produces a
    // number that looks like learning and is not.
    if let Some(defect) = split.defect() {
        return Err(format!("this split cannot be trusted: {defect}"));
    }

    // The control arm mispairs only the TRAINING half. Validation and probe
    // stay exactly as the real arm has them, or the two runs would be scored
    // against different questions and the comparison would be meaningless.
    let train_rows = if control {
        println!("CONTROL ARM: every training question is given another row's answer");
        qa::shuffle_answers(&split.train, seed)
    } else {
        split.train.clone()
    };
    qa::write_chat_jsonl(out_dir.join("train.jsonl"), &train_rows).map_err(|e| e.to_string())?;
    // `validation.jsonl` is the name the trainer looks for. It is SPENT on
    // deciding when to stop, which is why it is not the probe.
    qa::write_chat_jsonl(out_dir.join("validation.jsonl"), &split.validation).map_err(|e| e.to_string())?;
    qa::write_chat_jsonl(out_dir.join("probe.jsonl"), &split.probe).map_err(|e| e.to_string())?;
    artifact::write_jsonl(out_dir.join("split.jsonl"), &[&split]).map_err(|e| e.to_string())?;

    println!("train     {:>4}  rows, answer supervised and question masked", train_rows.len());
    println!("validate  {:>4}  phrasings spent on deciding when to stop", split.validation.len());
    println!("probe     {:>4}  held-out phrasings, never used for any decision", split.probe.len());
    let answers: BTreeMap<&str, usize> = split.train.iter().fold(BTreeMap::new(), |mut m, c| {
        *m.entry(c.answer.as_str()).or_default() += 1;
        m
    });
    println!("covering  {:>4}  distinct commands", answers.len());
    println!("\nwrote {}", out_dir.display());
    Ok(ExitCode::SUCCESS)
}

fn score(a: &mut Args) -> Result<ExitCode, String> {
    let probes_dir = a.path("--probes", "sft");
    let model = a.take("--model").ok_or("score needs --model")?;
    let adapter = a.take("--adapter");
    let seed = a.number("--seed", 1)?;
    let max_new = a.number("--max-new", 96)? as u32;
    a.finish()?;

    let (tool, _) = tools(seed);
    let rows: Vec<Candidate> =
        qa::read_chat_jsonl(probes_dir.join("probe.jsonl")).map_err(|e| format!("{}: {e}", probes_dir.display()))?;
    if rows.is_empty() {
        return Err(format!("{}: no held-out probes to score", probes_dir.display()));
    }

    let mut builder = TextGenerationPipeline::builder(&model);
    if let Some(path) = &adapter {
        builder = builder.adapter(path);
        println!("serving {model} with {path}");
    }
    // Load and answer are timed separately: one is paid once per process,
    // the other once per question, and a single total hides which.
    let loading = std::time::Instant::now();
    let pipeline = builder.load().map_err(|e| e.to_string())?;
    let load_secs = loading.elapsed().as_secs_f64();
    let mut passed = 0usize;
    let mut answer_ms: Vec<f64> = Vec::with_capacity(rows.len());
    let mut completion_tokens = 0u32;
    for row in &rows {
        let (question, expected) = (&row.question, &row.answer);
        let opts = brain::TextGenerationOptions::new().max_new_tokens(max_new).temperature(0.0).thinking(false).seed(seed);
        let asked = std::time::Instant::now();
        let generated = pipeline.generate_with(question, opts).map_err(|e| e.to_string())?;
        answer_ms.push(asked.elapsed().as_secs_f64() * 1000.0);
        completion_tokens += generated.completion_tokens;
        let reply = generated.text;
        let got = command_in(&reply, &tool);

        // Running is what decides. An answer that parses and does something
        // else is wrong, and only executing both says so.
        let verdict = match (tool.run(expected), tool.run(&got)) {
            (Ok(want), Ok(have)) if want == have => {
                passed += 1;
                "pass".to_string()
            }
            (Ok(_), Ok(_)) => "valid, different command".to_string(),
            (Ok(_), Err(why)) => format!("{why:?}"),
            (Err(why), _) => return Err(format!("the expected answer {expected:?} does not run: {why:?}")),
        };
        println!("  {verdict:<28} want {expected:<48} got {got:?}");
    }
    println!("\nscore {passed}/{} on held-out questions, every answer executed", rows.len());
    if !answer_ms.is_empty() {
        let mut sorted = answer_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mean = answer_ms.iter().sum::<f64>() / answer_ms.len() as f64;
        let median = sorted[sorted.len() / 2];
        let total: f64 = answer_ms.iter().sum::<f64>() / 1000.0;
        println!(
            "load {load_secs:.1} s, then {:.0} ms per answer on average ({:.0} ms median, {:.0} ms slowest), \
             {total:.1} s for all {} questions",
            mean,
            median,
            sorted.last().copied().unwrap_or(0.0),
            answer_ms.len()
        );
        if total > 0.0 {
            println!("{completion_tokens} completion tokens, {:.1} tokens/s", completion_tokens as f64 / total);
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// Everything that has to be true about the splits before a number taken
/// from them means anything.
///
/// Run as its own step rather than trusted from the stage that wrote them: a
/// split is easy to build correctly and easy to break later, and the failure
/// is silent - it shows up as a better result.
fn audit(a: &mut Args) -> Result<ExitCode, String> {
    let dir = a.path("--sft", "sft");
    let seed = a.number("--seed", 1)?;
    a.finish()?;
    let (tool, _) = tools(seed);

    let read = |name: &str| -> Result<Vec<Candidate>, String> {
        qa::read_chat_jsonl(dir.join(name)).map_err(|e| format!("{}/{name}: {e}", dir.display()))
    };
    let train = read("train.jsonl")?;
    let validation = read("validation.jsonl")?;
    let probe = read("probe.jsonl")?;

    let mut failed = Vec::new();
    let mut check = |name: &str, ok: bool, detail: String| {
        println!("  {:<52} {}", name, if ok { "PASS".to_string() } else { format!("FAIL  {detail}") });
        if !ok {
            failed.push(name.to_string());
        }
    };

    // 1. The three halves share no question. This is the one that turns a
    //    held-out score into a recall score if it ever stops holding.
    let qset = |rows: &[Candidate]| -> BTreeSet<String> { rows.iter().map(|c| c.question.to_lowercase()).collect() };
    let (qt, qv, qp) = (qset(&train), qset(&validation), qset(&probe));
    let overlap = |a: &BTreeSet<String>, b: &BTreeSet<String>| -> Vec<String> { a.intersection(b).cloned().collect() };
    for (name, shared) in [
        ("no question is both trained on and probed", overlap(&qt, &qp)),
        ("no question is both trained on and validated", overlap(&qt, &qv)),
        ("no question both decides the stop and reports the result", overlap(&qv, &qp)),
    ] {
        check(name, shared.is_empty(), format!("{} shared: {:?}", shared.len(), shared.iter().take(2).collect::<Vec<&String>>()));
    }

    // 2. Every question is distinct WITHIN a half too: a probe asked twice
    //    is one question counted twice.
    for (name, rows) in [("train", &train), ("validation", &validation), ("probe", &probe)] {
        let unique = qset(rows).len();
        check(&format!("every {name} question is distinct"), unique == rows.len(), format!("{} rows, {unique} distinct", rows.len()));
    }

    // 3. Every answer the model is asked for must be one the training half
    //    teaches. A probe about a command never trained on measures the
    //    split, not the model.
    let taught: BTreeSet<&str> = train.iter().map(|c| c.answer.as_str()).collect();
    for (name, rows) in [("validation", &validation), ("probe", &probe)] {
        let orphans: Vec<&str> = rows.iter().map(|c| c.answer.as_str()).filter(|a| !taught.contains(a)).collect();
        check(
            &format!("every {name} answer is taught by the training half"),
            orphans.is_empty(),
            format!("{} untaught: {:?}", orphans.len(), orphans.iter().take(2).collect::<Vec<&&str>>()),
        );
    }

    // 4. Every answer, in every half, actually runs. A reference that does
    //    not execute turns a correct reply into a failure.
    for (name, rows) in [("train", &train), ("validation", &validation), ("probe", &probe)] {
        let broken: Vec<&str> = rows.iter().map(|c| c.answer.as_str()).filter(|a| tool.run(a).is_err()).collect();
        check(
            &format!("every {name} answer executes"),
            broken.is_empty(),
            format!("{} do not run: {:?}", broken.len(), broken.iter().take(2).collect::<Vec<&&str>>()),
        );
    }

    // 5. No question gives away its own answer, in any half.
    for (name, rows) in [("train", &train), ("validation", &validation), ("probe", &probe)] {
        let leaks: Vec<&str> =
            rows.iter().filter(|c| qa::leaks_answer(&c.question, &c.answer)).map(|c| c.question.as_str()).collect();
        check(
            &format!("no {name} question quotes its own answer"),
            leaks.is_empty(),
            format!("{} leak: {:?}", leaks.len(), leaks.iter().take(2).collect::<Vec<&&str>>()),
        );
    }

    // 6. And the pairing is still sound in every half - the same check the
    //    data was accepted under, re-run against what was actually written.
    for (name, rows) in [("train", &train), ("validation", &validation), ("probe", &probe)] {
        let bad: Vec<String> = rows
            .iter()
            .filter_map(|c| tool.mismatch(&c.question, &c.answer).map(|why| format!("{:?}: {why}", c.question)))
            .collect();
        check(&format!("every {name} pairing is about one command"), bad.is_empty(), format!("{} bad: {:?}", bad.len(), bad.first()));
    }

    println!("\ntrain {} / validation {} / probe {}", train.len(), validation.len(), probe.len());
    if failed.is_empty() {
        println!("every check passed: these splits are testing what they claim to");
        return Ok(ExitCode::SUCCESS);
    }
    println!("{} check(s) failed: {failed:?}", failed.len());
    Ok(ExitCode::FAILURE)
}

fn report(a: &mut Args) -> Result<ExitCode, String> {
    let run_dir = a.path("--run-dir", "run");
    a.finish()?;
    let rows = ContinualReader::from_pretrained("unused").run_dir(&run_dir).ledger().map_err(|e| e.to_string())?;
    if rows.is_empty() {
        println!("nothing read yet");
        return Ok(ExitCode::SUCCESS);
    }
    for r in &rows {
        let cause = r.cause.clone().unwrap_or_default();
        // The training loss beside the verdict, because "the gate found no
        // significant gain" and "training did not move the model" are the
        // same verdict and a different problem.
        let trained = match r.train_loss {
            Some((a, b)) => format!("loss {a:.3} -> {b:.3}"),
            None => "not trained".to_string(),
        };
        println!("{:>5}  {:<28} {:<8} {:<18} {:<22} audited {}", r.episode, r.source, r.stage, cause, trained, r.audited);
    }
    let promoted = rows.iter().filter(|r| r.promoted).count();
    println!("\n{} episodes, {promoted} promoted", rows.len());
    for stage in ["screen", "reach", "ingest", "evidence", "gate"] {
        let n = rows.iter().filter(|r| r.stage == stage).count();
        if n > 0 {
            println!("  stopped at {stage}: {n}");
        }
    }
    Ok(ExitCode::SUCCESS)
}

/// The verb that makes the failure-mode half checkable: every episode's
/// verdict against the label of the lane it came from.
fn selftest(a: &mut Args) -> Result<ExitCode, String> {
    let corpus_dir = a.path("--corpus", "corpus");
    let run_dir = a.path("--run-dir", "run");
    a.finish()?;
    // The labels of the corpus that was READ, taken from the corpus itself.
    // Regenerating them from a seed would overwrite the documents the run is
    // about, and under a seed other than the one it was generated with it
    // would replace them with a different tool's while still printing a
    // verdict about this run.
    let labels = corpus::labels(&corpus_dir).map_err(|e| format!("{}: {e}", corpus_dir.display()))?;

    let rows = ContinualReader::from_pretrained("unused").run_dir(&run_dir).ledger().map_err(|e| e.to_string())?;
    if rows.is_empty() {
        return Err("nothing has been read yet, so there are no verdicts to check".to_string());
    }

    let mut failures = 0usize;
    for row in &rows {
        let Some((_, expect)) = labels.iter().find(|(rel, _)| row.source == **rel) else {
            continue;
        };
        let ok = match expect {
            Expect::Promote => row.promoted,
            Expect::Refuse => !row.promoted,
            Expect::Either => true,
        };
        if !ok {
            failures += 1;
            let want = if *expect == Expect::Promote { "should have been learned" } else { "should have been refused" };
            println!("FAIL {:<28} {want}, stopped at {} ({})", row.source, row.stage, row.cause.clone().unwrap_or_default());
        }
    }

    let checked = rows.iter().filter(|r| labels.contains_key(&r.source)).count();
    if failures == 0 {
        println!("selftest: {checked} labelled episodes, every verdict as expected");
    } else {
        println!("selftest: {failures} of {checked} labelled episodes went the wrong way");
    }

    // The acceptance block, as far as this run can answer it. Printing the
    // unanswerable clauses is the point rather than an omission: a reader
    // that has not been run against a real model cannot have a battery
    // delta or a null-gate arm, and a report that quietly left those out
    // would look like a run that passed them.
    let facts = LedgerFacts::of(&rows);
    println!(
        "\nrun record: {} episodes, {} promoted, {} refused ({} explained), {} audit decodes",
        facts.episodes, facts.promoted, facts.rejections, facts.rejections_with_cause, facts.eval_decodes
    );
    let unexplained = LedgerFacts::unexplained(&rows);
    if unexplained.is_empty() {
        println!("  clause 1 (every refusal names its cause): PASS");
    } else {
        println!("  clause 1 (every refusal names its cause): FAIL, unexplained: {}", unexplained.join(", "));
    }
    println!("  clause 6 (control arms)        : needs a null-gate arm and a second seed");
    println!("  clause 4 (independent battery) : needs `battery` before and after a read");
    println!("  clause 3 (bwt, per-block bar)  : needs the retention matrix of a real run");
    println!("\nthose three are UNANSWERED, not passed: this run has not produced the numbers they are about.");

    if failures == 0 && unexplained.is_empty() {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::FAILURE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every number this sample reports is about a particular seed and a
    /// particular run directory. A misspelled flag that fell through to a
    /// default would produce a plausible number about the wrong run, which
    /// is worse than no number at all.
    #[test]
    fn a_misspelled_flag_is_refused_rather_than_silently_defaulted() {
        let dir = std::env::temp_dir().join(format!("sample-reader-unknown-flag-{}", std::process::id()));
        let argv = vec!["generate".to_string(), "--corpus".to_string(), dir.to_string_lossy().into_owned(), "--seeed".to_string(), "2".to_string()];
        let err = run(argv).expect_err("an unknown flag must not be ignored");
        assert!(err.contains("--seeed"), "the message must name what was not understood: {err}");
        assert!(!dir.exists(), "a refused invocation must not have written anything");
    }

    /// The stay-silent half: the flags a verb documents are accepted.
    #[test]
    fn the_documented_flags_of_a_verb_are_accepted() {
        let dir = std::env::temp_dir().join(format!("sample-reader-args-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let argv = vec!["generate".to_string(), "--corpus".to_string(), dir.to_string_lossy().into_owned(), "--seed".to_string(), "2".to_string()];
        run(argv).expect("the documented flags are accepted");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
