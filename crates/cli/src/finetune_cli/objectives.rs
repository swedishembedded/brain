// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The objectives of `brain finetune` that have a pipeline of their own in the SDK: preference optimization
//! ([`brain::PreferenceFineTune`]: DPO, IPO, SimPO, ORPO), KTO on judged completions ([`brain::FeedbackFineTune`]) and
//! top-K distillation ([`brain::DistillFineTune`]). The CLI builds the same pipeline types the training job of the HTTP
//! API builds; this module only maps flags onto their setters and prints what their outcomes carry.
//!
//! Swedish Embedded AB implements solutions for aligning open-weight language models with a team's own judgements on
//! the memory it actually has, for its clients. If your team needs expertise in preference optimization, KTO or
//! distillation of language models then you can procure our services by sending an email to
//! info@swedishembedded.com.

use super::Args;

/// What a run trains toward (`--objective`); the supervised loss is the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Objective {
    #[default]
    Sft,
    PolicyGradient,
    Preference,
    Feedback,
    Distill,
}

impl Objective {
    const NAMES: &'static str = "sft, policy-gradient, preference, feedback or distill";

    pub(super) fn parse(name: &str) -> Result<Objective, String> {
        match name {
            "sft" => Ok(Objective::Sft),
            "policy-gradient" | "policy_gradient" => Ok(Objective::PolicyGradient),
            "preference" => Ok(Objective::Preference),
            "feedback" => Ok(Objective::Feedback),
            "distill" => Ok(Objective::Distill),
            other => Err(format!("brain finetune: --objective {other:?} is none of {}", Objective::NAMES)),
        }
    }

    pub(super) fn name(self) -> &'static str {
        match self {
            Objective::Sft => "sft",
            Objective::PolicyGradient => "policy-gradient",
            Objective::Preference => "preference",
            Objective::Feedback => "feedback",
            Objective::Distill => "distill",
        }
    }

    /// Whether the objective has its own pipeline here, trained as an adapter only.
    pub(super) fn is_paired(self) -> bool {
        matches!(self, Objective::Preference | Objective::Feedback | Objective::Distill)
    }
}

/// The teacher of a distillation (`--teacher`): the policy of the same base that scores every record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum Teacher {
    Base,
    Start,
    /// The adapter file of another policy on this base.
    Adapter(String),
}

impl Teacher {
    pub(super) fn parse(spec: &str) -> Result<Teacher, String> {
        match spec {
            "base" => Ok(Teacher::Base),
            "start" => Ok(Teacher::Start),
            other => match other.strip_prefix("adapter:") {
                Some(path) if !path.is_empty() => Ok(Teacher::Adapter(path.to_string())),
                _ => Err(format!("brain finetune: --teacher {other:?} is none of base, start, adapter:FILE")),
            },
        }
    }

    fn sdk(&self) -> brain::DistillTeacher {
        match self {
            Teacher::Base => brain::DistillTeacher::Base,
            Teacher::Start => brain::DistillTeacher::Start,
            Teacher::Adapter(path) => brain::DistillTeacher::Adapter(path.into()),
        }
    }
}

/// The pair loss `--loss` names, SimPO's margin from `--gamma`.
pub(super) fn pair_loss(a: &Args) -> Result<brain::PairLoss, String> {
    let name = a.loss.as_deref().unwrap_or("dpo");
    if a.gamma.is_some() && name != "simpo" {
        return Err(format!("brain finetune: --gamma is simpo's target margin; the loss is {name}"));
    }
    brain::PairLoss::parse(name, a.gamma.unwrap_or(brain::DEFAULT_SIMPO_GAMMA)).map_err(|e| format!("brain finetune: --loss: {e}"))
}

/// Refuses, by name, every flag the run's objective does not read, and every objective flag given to another one: a
/// setting that is accepted and not applied is a silent degradation.
pub(super) fn validate(a: &Args) -> Result<(), String> {
    let objective = a.objective;
    let flags_of = |own: Objective, flags: &[(&str, bool)]| -> Result<(), String> {
        for (flag, given) in flags {
            if *given && objective != own {
                return Err(format!("brain finetune: {flag} belongs to --objective {}", own.name()));
            }
        }
        Ok(())
    };
    flags_of(Objective::Preference, &[("--loss", a.loss.is_some()), ("--gamma", a.gamma.is_some()), ("--nll-weight", a.nll_weight.is_some())])?;
    flags_of(Objective::Feedback, &[("--desirable-weight", a.desirable_weight.is_some()), ("--undesirable-weight", a.undesirable_weight.is_some())])?;
    flags_of(Objective::Distill, &[("--teacher", a.teacher.is_some()), ("--top-k", a.top_k.is_some())])?;
    if a.beta.is_some() && !matches!(objective, Objective::Preference | Objective::Feedback) {
        return Err("brain finetune: --beta belongs to --objective preference or feedback".to_string());
    }
    if a.keep_reasoning && !matches!(objective, Objective::Preference | Objective::Feedback) {
        return Err("brain finetune: --keep-reasoning belongs to --objective preference or feedback (the supervised objectives take --no-thinking)".to_string());
    }
    if !objective.is_paired() {
        return Ok(());
    }
    if a.lora.is_none() {
        return Err(format!("brain finetune: --objective {} trains an adapter; give --lora RANK", objective.name()));
    }
    let mut unread = vec![("--weight-decay", a.weight_decay.is_some()), ("--modules-to-save", !a.modules_to_save.is_empty())];
    if objective != Objective::Preference {
        unread.extend([("--held-out", a.held_out.is_some()), ("--stages", a.stages.is_some()), ("--native-sparse", a.native_sparse), ("--activations", a.activations.is_some())]);
    }
    if objective == Objective::Distill {
        // Distillation reads the records' supervised turns as `sft` does.
    } else {
        unread.push(("--no-thinking", a.no_thinking));
    }
    for (flag, given) in unread {
        if given {
            return Err(format!("brain finetune: --objective {} does not read {flag}", objective.name()));
        }
    }
    if objective == Objective::Preference {
        pair_loss(a)?;
    }
    if objective == Objective::Distill && a.teacher == Some(Teacher::Start) && a.continue_from.is_none() {
        return Err("brain finetune: --teacher start is the adapter the run starts from; give --continue FILE".to_string());
    }
    Ok(())
}

fn show(v: Option<f32>) -> String {
    v.map_or("-".to_string(), |v| format!("{v:.4}"))
}

/// Runs the paired objective `a` names; the process exit code.
pub(super) fn run(a: &Args) -> i32 {
    let cancel = brain::CancelToken::default();
    let progress = |p: &brain::FineTuneProgress| eprintln!("step {}: loss {:.4}", p.step, p.loss);
    let lines = match a.objective {
        Objective::Preference => preference(a).run_with(&cancel, progress).map(|o| {
            let score = |name: &str, s: &Option<brain::PreferenceScore>| s.map(|s| format!("{name}: {} pairs, accuracy {}, mean margin {}", s.pairs, show(s.accuracy), show(s.mean_margin)));
            let mut lines = vec![format!("status: {:?}, {} steps, loss {} -> {} ({})", o.status, o.steps_completed, show(o.initial_loss), show(o.final_loss), o.loss.name())];
            lines.extend(score("train", &o.train_score).into_iter().chain(score("held-out", &o.held_out_score)));
            lines.extend(o.adapter.map(|p| format!("adapter: {}", p.display())));
            lines.extend(o.resume_state.map(|p| format!("resume state: {}", p.display())));
            lines
        }),
        Objective::Feedback => feedback(a).run_with(&cancel, progress).map(|o| {
            let mut lines = vec![format!("status: {:?}, {} steps, loss {} -> {}", o.status, o.steps_completed, show(o.initial_loss), show(o.final_loss))];
            lines.push(format!("log-ratio to the reference: desirable {} ({} examples), undesirable {} ({} examples)", show(o.desirable_logratio), o.desirable_examples, show(o.undesirable_logratio), o.undesirable_examples));
            lines.extend(o.adapter.map(|p| format!("adapter: {}", p.display())));
            lines.extend(o.resume_state.map(|p| format!("resume state: {}", p.display())));
            lines
        }),
        Objective::Distill => distill(a).run_with(&cancel, progress).map(|o| {
            let mut lines = vec![format!("status: {:?}, {} steps, loss {} -> {}", o.status, o.steps_completed, show(o.initial_loss), show(o.final_loss))];
            lines.push(format!("teacher: {} (top {} of {} records)", o.teacher, o.top_k, o.examples));
            lines.extend(o.adapter.map(|p| format!("adapter: {}", p.display())));
            lines.extend(o.resume_state.map(|p| format!("resume state: {}", p.display())));
            lines
        }),
        Objective::Sft | Objective::PolicyGradient => unreachable!("the supervised and policy-gradient objectives run through the chat trainers"),
    };
    match lines {
        Ok(lines) => {
            lines.iter().for_each(|l| println!("{l}"));
            0
        }
        Err(e) => {
            eprintln!("brain finetune: {e}");
            1
        }
    }
}

fn preference(a: &Args) -> brain::PreferenceFineTune {
    let mut t = brain::PreferenceFineTune::from_pretrained(a.base.clone()).rank(a.lora.expect("validate checked --lora")).keep_reasoning(a.keep_reasoning).strict_updates(a.strict_updates).loss(pair_loss(a).expect("validate checked --loss"));
    if let Some(v) = &a.dataset {
        t = t.dataset(v);
    }
    if let Some(v) = &a.held_out {
        t = t.held_out(v);
    }
    if let Some(v) = &a.out {
        t = t.out_dir(v);
    }
    if let Some(v) = a.alpha {
        t = t.alpha(v);
    }
    if let Some(v) = &a.continue_from {
        t = t.continue_from(v);
    }
    if let Some(v) = &a.adapter_id {
        t = t.adapter_id(v.clone());
    }
    if let Some(v) = &a.arch {
        t = t.arch(v.clone());
    }
    if let Some(v) = &a.models_dir {
        t = t.models_dir(v.clone());
    }
    if let Some(v) = a.beta {
        t = t.beta(v);
    }
    if let Some(v) = a.nll_weight {
        t = t.nll_weight(v);
    }
    if let Some(v) = a.steps {
        t = t.steps(v);
    }
    if let Some(v) = a.lr {
        t = t.lr(v);
    }
    if let Some(v) = a.grad_accum {
        t = t.grad_accum(v);
    }
    if let Some(v) = a.stages {
        t = t.stages(v);
    }
    if let Some((cfg, long_range)) = super::sparse(a) {
        t = t.native_sparse_attention(cfg, long_range);
    }
    if let Some(v) = a.seed {
        t = t.seed(v);
    }
    if let Some(v) = a.max_block {
        t = t.max_block(v);
    }
    if let Some(v) = &a.base_format {
        t = t.base_format(v.clone());
    }
    if let Some(v) = a.activations {
        t = t.activations(v);
    }
    t
}

fn feedback(a: &Args) -> brain::FeedbackFineTune {
    let mut t = brain::FeedbackFineTune::from_pretrained(a.base.clone()).rank(a.lora.expect("validate checked --lora")).keep_reasoning(a.keep_reasoning).strict_updates(a.strict_updates);
    if let Some(v) = &a.dataset {
        t = t.dataset(v);
    }
    if let Some(v) = &a.out {
        t = t.out_dir(v);
    }
    if let Some(v) = a.alpha {
        t = t.alpha(v);
    }
    if let Some(v) = &a.continue_from {
        t = t.continue_from(v);
    }
    if let Some(v) = &a.adapter_id {
        t = t.adapter_id(v.clone());
    }
    if let Some(v) = &a.arch {
        t = t.arch(v.clone());
    }
    if let Some(v) = &a.models_dir {
        t = t.models_dir(v.clone());
    }
    if let Some(v) = a.beta {
        t = t.beta(v);
    }
    if let Some(v) = a.desirable_weight {
        t = t.desirable_weight(v);
    }
    if let Some(v) = a.undesirable_weight {
        t = t.undesirable_weight(v);
    }
    if let Some(v) = a.steps {
        t = t.steps(v);
    }
    if let Some(v) = a.lr {
        t = t.lr(v);
    }
    if let Some(v) = a.grad_accum {
        t = t.grad_accum(v);
    }
    if let Some(v) = a.seed {
        t = t.seed(v);
    }
    if let Some(v) = a.max_block {
        t = t.max_block(v);
    }
    if let Some(v) = &a.base_format {
        t = t.base_format(v.clone());
    }
    t
}

fn distill(a: &Args) -> brain::DistillFineTune {
    let mut t = brain::DistillFineTune::from_pretrained(a.base.clone()).rank(a.lora.expect("validate checked --lora")).thinking(!a.no_thinking).strict_updates(a.strict_updates);
    if let Some(v) = &a.dataset {
        t = t.dataset(v);
    }
    if let Some(v) = &a.out {
        t = t.out_dir(v);
    }
    if let Some(v) = a.alpha {
        t = t.alpha(v);
    }
    if let Some(v) = &a.teacher {
        t = t.teacher(v.sdk());
    }
    if let Some(v) = a.top_k {
        t = t.top_k(v);
    }
    if let Some(v) = &a.continue_from {
        t = t.continue_from(v);
    }
    if let Some(v) = &a.adapter_id {
        t = t.adapter_id(v.clone());
    }
    if let Some(v) = &a.arch {
        t = t.arch(v.clone());
    }
    if let Some(v) = &a.models_dir {
        t = t.models_dir(v.clone());
    }
    if let Some(v) = a.steps {
        t = t.steps(v);
    }
    if let Some(v) = a.lr {
        t = t.lr(v);
    }
    if let Some(v) = a.grad_accum {
        t = t.grad_accum(v);
    }
    if let Some(v) = a.seed {
        t = t.seed(v);
    }
    if let Some(v) = a.max_block {
        t = t.max_block(v);
    }
    if let Some(v) = &a.base_format {
        t = t.base_format(v.clone());
    }
    t
}
