// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain finetune` - supervised fine-tuning of a chat model through the SDK's generic trainers
//! ([`brain::ChatFineTune`], [`brain::ChatFullFineTune`]); the architecture is a flag, not a token.
//!
//! Swedish Embedded AB implements solutions for adapting open-weight language models to a team's own data on the
//! memory it actually has, for its clients. If your team needs expertise in fine-tuning large language models of
//! any architecture then you can procure our services by sending an email to info@swedishembedded.com.

mod objectives;

use objectives::Objective;

const USAGE: &str = "\
brain finetune BASE --dataset FILE --out DIR [options]

  Fine-tune a chat model on a chat JSONL dataset: with --lora RANK a LoRA adapter over the frozen base
  (written to DIR/adapter.safetensors, which `brain serve` loads beside the base), without it every
  weight (written as a checkpoint directory under DIR).

  BASE               a checkpoint file or directory, or a vendor/repo in the model store.
  --dataset FILE     chat JSONL: messages with `train` marking the supervised turns.
  --out DIR          where the prepared dataset, the result and its training record go.
  --lora RANK        train an adapter of this rank (default: a full fine-tune).
  --alpha A          the adapter's scale (default: the rank).
  --continue FILE    continue the adapter in FILE, at its own rank.
  --adapter-id ID    the id written on the adapter's card.
  --arch ID          the base's architecture (default: what its config names).
  --models-dir DIR   where a vendor/repo base resolves.
  --steps N          optimizer steps (default 100).
  --lr X             the peak learning rate.
  --weight-decay X   AdamW weight decay.
  --grad-accum N     micro-batches per step.
  --strict-updates   abort at the first optimiser step whose loss or gradient is not finite (default: skip it and
                     abort on the third in a row); the weights are never updated with a non-finite value.
  --seed N           the seed of the initial adapter and the batch order (default random, reported).
  --max-block N      the longest record, in tokens.
  --base-format F    the frozen base's format for an adapter: f32|bf16|int8|nf4|fp8, as far as the
                     architecture's trainer holds it (default: chosen from the device and what fits).
  --stages N         cut the model into exactly N pipeline stages (default: one stage when a device holds it, else
                     the fewest that fit the selected cards), for architectures whose trainer takes a count.
  --native-sparse    train the global-attention layers with native sparse attention (N-LSA): each query reads only
                     its window, sinks and clustered long-range keys. Qwen3 and Ripple take it; it needs a device
                     that runs workgroup-barrier kernels (a GPU, or --device software for a small check).
  --lsa-window N --lsa-sink N --lsa-top-k N --lsa-top-p N --lsa-cluster-block N
                     the selection's budget (defaults: the paper's D-LSA configuration).
  --lsa-window-only  with --native-sparse: window and sinks only, no long-range keys (the baseline).
  --modules-to-save LIST
                     with --lora: also train these base tensors in full and carry them in the adapter file (PEFT's
                     modules_to_save), a comma list of tensor names, leaves or the architecture's groups (Ripple:
                     norms, gates); a served adapter's rows read them per row.
  --held-out FILE    chat JSONL scored on the model in the training process before the first step and after the last
                     (the loss is printed); for architectures whose trainer scores it (Ripple, qwen35).
  --objective O      what the adapter is trained toward: sft (default), policy-gradient, preference, feedback or distill
                     (the last three train an adapter, so they need --lora RANK).
  --loss L --gamma X --beta X --nll-weight X
                     with --objective preference: the pair loss dpo (default), ipo, simpo or orpo over `chosen`/`rejected`
                     pairs (--dataset and --held-out hold pairs); --gamma is simpo's target margin, --beta the loss's
                     temperature (simpo's beta, orpo's lambda; also KTO's), --nll-weight the anchor on the chosen answer.
  --desirable-weight X --undesirable-weight X
                     with --objective feedback: KTO's weights of a desirable and an undesirable completion's term (default 1
                     each); --beta is its temperature.
  --teacher T --top-k N
                     with --objective distill: the policy of the same base that teaches, base (default), start (the adapter of
                     --continue) or adapter:FILE, and how many of its most likely tokens per position are kept (default 16).
  --keep-reasoning   with --objective preference or feedback: train on each candidate's reasoning block (default: dropped).
  --policy-gradient  train the clipped policy gradient on rollout records (`reward` and `group` on each record,
                     training-record contract v2) instead of the supervised loss, for an adapter; --ratio-clip X and
                     --kl-beta X set its clip band and KL weight.
  --algorithm A      with --policy-gradient: a named algorithm whose settings are this run's defaults (any flag below
                     overrides it): grpo (z-score, seq-mean, symmetric clip, the KL term), dr-grpo (mean-centred,
                     constant-length over --loss-agg-length or --max-block, no KL), dapo (z-score, clip-higher 0.28,
                     token-sum, truncated masked, no KL) or rloo (leave-one-out, token-sum).
  --advantage-norm N with --policy-gradient: how each group's rewards become advantages: mean-centred (default),
                     z-score or leave-one-out.
  --loss-agg A       with --policy-gradient: how the token terms are normalised: seq-mean (default; each sequence by its own
                     token count), token-sum (every token of a step by the step's count) or constant-length (each sequence's
                     sum by --loss-agg-length, else --max-block, averaged over the sequences: Dr. GRPO).
  --loss-agg-length N
                     with --loss-agg constant-length: the constant, in tokens.
  --advantage-floor F
                     with --policy-gradient: none (default) or zero: clamp each advantage at zero after it is computed
                     against its whole group (failures set the baseline, never train).
  --truncated T      with --policy-gradient: what a rollout marked truncated does: mask (default; never trained, its reward
                     takes no part in its group) or keep (trained, its reward counted).
  --ratio-clip-high X
                     with --policy-gradient: widen the clip band's upper edge to 1+X (clip-higher; at least --ratio-clip).
  --dual-clip C      with --policy-gradient: dual-clip PPO's bound on a negative advantage's term, max(min(r*A, clip(r)*A),
                     c*A), above 1 (default 3); each step reports the share it bounded.
  --importance C     with --policy-gradient: correct for the sampling engine's gap to the trainer: none (default),
                     truncate:CAP (weight each token's policy term by min(pi_trainer / pi_rollout, CAP)) or mask:CAP
                     (drop tokens whose weight leaves [1/CAP, CAP]); needs recorded behaviour log-probabilities.
  --max-step-kl X --max-step-grad-norm X
                     with --policy-gradient: skip, and log, an optimiser step whose KL to the reference, or whose gradient
                     norm, is over the bound (a run that skips 8 steps in a row stops).
  --kv-roundtrip TIER
                     with --policy-gradient: read the trainer's attention keys and values through the encoding of the
                     KV cache the rollouts were sampled with (int8 or fp8, as the server's --kv-cache; the trace's
                     brain.kv_cache names it), so the teacher-forced forward has the server's numerics.
  --numerics-gate    with --policy-gradient: take no optimiser step; measure, over the records the run draws (raise --steps
                     until it covers them), how far the trainer's own log-probabilities of the recorded tokens are from the
                     recorded behaviour ones (mean and largest gap, mismatch KL, the step-0 ratio, effective sample size) and
                     the gradient norms of the batches, print the report as JSON and exit 3 when a bound below is broken.
  --gate-max-mean-gap X --gate-max-gap X --gate-max-mismatch-kl X --gate-min-sequence-ess X --gate-max-grad-norm X
                     the gate's bounds (each optional: an unset one is reported and not enforced).
  --behaviour B      with --policy-gradient: where the behaviour log-probabilities of the ratio come from.
                     recorded (default): the sampler's own, on every supervised token of every record; start_of_run_proxy:
                     where none was recorded, the start-of-run policy's (exact only if the rollouts were sampled from it).
  --activations P    what the backward keeps of the forward: keep|recompute.
  --train-precision SPEC
                     a full fine-tune's storage: `state=f32|bf16|int8[:B]|log8[:B]` for the optimizer state,
                     `weights=bf16|fp8[:B]` to hold the linears and the tables on the device in that encoding with
                     their f32 master copy and the optimizer state off it, `window=N` to keep only N layers on the
                     device (the rest streamed from the host), `clip=two-pass|running`. An architecture that does not
                     run a choice refuses it by name.
  --no-thinking      render records without the reasoning block.
  --job-plan FILE --job-rank N --job-secret FILE
                     train as rank N of the distributed job the plan names (the same three on every rank, a different
                     N): each rank trains on its own shard of the records and every step's gradients are reduced over
                     the job, on the cards when every rank can. The plan's `sharding` (optimizer|gradients) shards the
                     optimizer state, and the reduced gradients, across the ranks instead of replicating them. Without
                     --seed every rank uses the plan's digest; rank 0 writes the result.

Run on a device with `--device` (default: every device brain finds).
";

#[derive(Debug, Default)]
struct Args {
    /// `--job-plan`, `--job-rank`, `--job-secret`: this run as one rank of a distributed job.
    job_plan: Option<String>,
    job_rank: Option<usize>,
    job_secret: Option<String>,
    base: String,
    dataset: Option<String>,
    out: Option<String>,
    lora: Option<u32>,
    alpha: Option<f32>,
    continue_from: Option<String>,
    adapter_id: Option<String>,
    arch: Option<String>,
    models_dir: Option<String>,
    steps: Option<u32>,
    lr: Option<f32>,
    weight_decay: Option<f32>,
    grad_accum: Option<u32>,
    strict_updates: bool,
    stages: Option<u32>,
    native_sparse: bool,
    lsa_window: Option<usize>,
    lsa_sink: Option<usize>,
    lsa_top_k: Option<usize>,
    lsa_top_p: Option<usize>,
    lsa_cluster_block: Option<usize>,
    lsa_window_only: bool,
    seed: Option<u64>,
    max_block: Option<u32>,
    base_format: Option<String>,
    activations: Option<model::saved::SavedPolicy>,
    no_thinking: bool,
    train_precision: Option<model::TrainPrecision>,
    held_out: Option<String>,
    modules_to_save: Vec<String>,
    policy_gradient: bool,
    /// `--objective`, as given; [`Args::objective`] is the one the run trains.
    objective_flag: Option<Objective>,
    objective: Objective,
    loss: Option<String>,
    gamma: Option<f32>,
    beta: Option<f32>,
    nll_weight: Option<f32>,
    keep_reasoning: bool,
    desirable_weight: Option<f32>,
    undesirable_weight: Option<f32>,
    teacher: Option<objectives::Teacher>,
    top_k: Option<usize>,
    clip_eps: Option<f32>,
    kl_beta: Option<f32>,
    behaviour: Option<brain::BehaviourSource>,
    advantage_norm: Option<brain::AdvantageNorm>,
    loss_agg: Option<String>,
    algorithm: Option<brain::PolicyGradientAlgorithm>,
    truncated: Option<brain::TruncatedRollouts>,
    advantage_floor: Option<brain::AdvantageFloor>,
    loss_agg_length: Option<u32>,
    clip_eps_high: Option<f32>,
    dual_clip: Option<f32>,
    importance: Option<brain::ImportanceCorrection>,
    max_step_kl: Option<f32>,
    max_step_grad_norm: Option<f32>,
    numerics_gate: bool,
    gate: brain::NumericsBounds,
    kv_roundtrip: Option<brain::KvElement>,
}

/// The exit status of a numerics gate whose bounds were broken (1 is a failed run, 2 a usage error).
const NUMERICS_GATE_FAILED: i32 = 3;

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut i = 0;
    let need = |i: &mut usize, flag: &str| -> Result<String, String> {
        *i += 1;
        argv.get(*i).cloned().ok_or_else(|| format!("brain finetune: {flag} needs a value"))
    };
    fn number<T: std::str::FromStr>(v: &str, flag: &str) -> Result<T, String> {
        v.parse().map_err(|_| format!("brain finetune: {flag} wants a number, got '{v}'"))
    }
    while i < argv.len() {
        let flag = argv[i].clone();
        match flag.as_str() {
            "--dataset" => a.dataset = Some(need(&mut i, &flag)?),
            "--out" => a.out = Some(need(&mut i, &flag)?),
            "--lora" => a.lora = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--alpha" => a.alpha = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--continue" => a.continue_from = Some(need(&mut i, &flag)?),
            "--adapter-id" => a.adapter_id = Some(need(&mut i, &flag)?),
            "--arch" => a.arch = Some(need(&mut i, &flag)?),
            "--models-dir" => a.models_dir = Some(need(&mut i, &flag)?),
            "--steps" => a.steps = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--lr" => a.lr = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--weight-decay" => a.weight_decay = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--grad-accum" => a.grad_accum = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--strict-updates" => a.strict_updates = true,
            "--native-sparse" => a.native_sparse = true,
            "--lsa-window-only" => a.lsa_window_only = true,
            "--lsa-window" => a.lsa_window = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--lsa-sink" => a.lsa_sink = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--lsa-top-k" => a.lsa_top_k = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--lsa-top-p" => a.lsa_top_p = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--lsa-cluster-block" => a.lsa_cluster_block = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--stages" => a.stages = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--seed" => a.seed = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--max-block" => a.max_block = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--base-format" => a.base_format = Some(need(&mut i, &flag)?),
            "--activations" => a.activations = Some(crate::args::parse_activations(&need(&mut i, &flag)?).map_err(|e| format!("brain finetune: {e}"))?),
            "--no-thinking" => a.no_thinking = true,
            "--job-plan" => a.job_plan = Some(need(&mut i, &flag)?),
            "--job-rank" => a.job_rank = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--job-secret" => a.job_secret = Some(need(&mut i, &flag)?),
            "--train-precision" => {
                let spec = need(&mut i, &flag)?;
                a.train_precision = Some(model::TrainPrecision::parse(&spec).map_err(|e| format!("brain finetune: --train-precision {spec:?}: {e}"))?);
            }
            "--held-out" => a.held_out = Some(need(&mut i, &flag)?),
            "--modules-to-save" => a.modules_to_save = need(&mut i, &flag)?.split(',').map(str::trim).filter(|p| !p.is_empty()).map(String::from).collect(),
            "--policy-gradient" => a.policy_gradient = true,
            "--objective" => a.objective_flag = Some(Objective::parse(&need(&mut i, &flag)?)?),
            "--loss" => a.loss = Some(need(&mut i, &flag)?),
            "--gamma" => a.gamma = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--beta" => a.beta = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--nll-weight" => a.nll_weight = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--keep-reasoning" => a.keep_reasoning = true,
            "--desirable-weight" => a.desirable_weight = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--undesirable-weight" => a.undesirable_weight = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--teacher" => a.teacher = Some(objectives::Teacher::parse(&need(&mut i, &flag)?)?),
            "--top-k" => a.top_k = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--ratio-clip" => a.clip_eps = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--kl-beta" => a.kl_beta = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--behaviour" => a.behaviour = Some(need(&mut i, &flag)?.parse().map_err(|e| format!("brain finetune: {flag}: {e}"))?),
            "--advantage-norm" => a.advantage_norm = Some(brain::AdvantageNorm::parse(&need(&mut i, &flag)?).map_err(|e| format!("brain finetune: {flag}: {e}"))?),
            "--loss-agg" => a.loss_agg = Some(need(&mut i, &flag)?),
            "--algorithm" => a.algorithm = Some(brain::PolicyGradientAlgorithm::parse(&need(&mut i, &flag)?).map_err(|e| format!("brain finetune: {flag}: {e}"))?),
            "--advantage-floor" => a.advantage_floor = Some(brain::AdvantageFloor::parse(&need(&mut i, &flag)?).map_err(|e| format!("brain finetune: {flag}: {e}"))?),
            "--truncated" => a.truncated = Some(brain::TruncatedRollouts::parse(&need(&mut i, &flag)?).map_err(|e| format!("brain finetune: {flag}: {e}"))?),
            "--loss-agg-length" => a.loss_agg_length = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--ratio-clip-high" => a.clip_eps_high = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--dual-clip" => a.dual_clip = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--importance" => a.importance = Some(need(&mut i, &flag)?.parse().map_err(|e| format!("brain finetune: {flag}: {e}"))?),
            "--max-step-kl" => a.max_step_kl = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--max-step-grad-norm" => a.max_step_grad_norm = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--numerics-gate" => a.numerics_gate = true,
            "--kv-roundtrip" => a.kv_roundtrip = Some(brain::KvTier::parse(&need(&mut i, &flag)?).map_err(|e| format!("brain finetune: {flag}: {e}"))?.element()),
            "--gate-max-mean-gap" => a.gate.max_mean_abs_gap = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--gate-max-gap" => a.gate.max_abs_gap = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--gate-max-mismatch-kl" => a.gate.max_mismatch_kl = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--gate-min-sequence-ess" => a.gate.min_sequence_ess_fraction = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--gate-max-grad-norm" => a.gate.max_grad_norm = Some(number(&need(&mut i, &flag)?, &flag)?),
            "-h" | "--help" => return Err(String::new()),
            other if other.starts_with('-') => return Err(format!("brain finetune: unknown option '{other}'")),
            other if a.base.is_empty() => a.base = other.to_string(),
            other => return Err(format!("brain finetune: unexpected argument '{other}'")),
        }
        i += 1;
    }
    if a.base.is_empty() {
        return Err("brain finetune: no base given".to_string());
    }
    a.objective = match (a.objective_flag, a.policy_gradient) {
        (None, false) => Objective::Sft,
        (None, true) | (Some(Objective::PolicyGradient), _) => Objective::PolicyGradient,
        (Some(Objective::Sft), true) => return Err("brain finetune: --policy-gradient trains the policy-gradient objective; --objective sft contradicts it".to_string()),
        (Some(o), true) => return Err(format!("brain finetune: --policy-gradient trains the policy-gradient objective; --objective {} contradicts it", o.name())),
        (Some(o), false) => o,
    };
    a.policy_gradient = a.objective == Objective::PolicyGradient;
    objectives::validate(&a)?;
    // The budget flags shape the selection; without --native-sparse they would be ignored.
    if !a.native_sparse {
        for (flag, given) in [("--lsa-window", a.lsa_window.is_some()), ("--lsa-sink", a.lsa_sink.is_some()), ("--lsa-top-k", a.lsa_top_k.is_some()), ("--lsa-top-p", a.lsa_top_p.is_some()), ("--lsa-cluster-block", a.lsa_cluster_block.is_some()), ("--lsa-window-only", a.lsa_window_only)] {
            if given {
                return Err(format!("brain finetune: {flag} shapes native sparse attention; give --native-sparse"));
            }
        }
    }
    if !a.policy_gradient {
        for (flag, given) in [("--ratio-clip", a.clip_eps.is_some()), ("--kl-beta", a.kl_beta.is_some()), ("--behaviour", a.behaviour.is_some()), ("--advantage-norm", a.advantage_norm.is_some()), ("--algorithm", a.algorithm.is_some()), ("--loss-agg", a.loss_agg.is_some()), ("--loss-agg-length", a.loss_agg_length.is_some()), ("--truncated", a.truncated.is_some()), ("--advantage-floor", a.advantage_floor.is_some()), ("--ratio-clip-high", a.clip_eps_high.is_some()), ("--dual-clip", a.dual_clip.is_some()), ("--importance", a.importance.is_some()), ("--max-step-kl", a.max_step_kl.is_some()), ("--max-step-grad-norm", a.max_step_grad_norm.is_some()), ("--kv-roundtrip", a.kv_roundtrip.is_some())] {
            if given {
                return Err(format!("brain finetune: {flag} shapes the policy gradient; give --policy-gradient"));
            }
        }
    }
    if a.policy_gradient {
        algorithm_config(&a)?;
        loss_agg(&a)?;
    }
    if !a.numerics_gate && a.gate != brain::NumericsBounds::default() {
        return Err("brain finetune: the --gate-* bounds belong to the numerics gate; give --numerics-gate".to_string());
    }
    if a.numerics_gate {
        if !a.policy_gradient {
            return Err("brain finetune: --numerics-gate measures a policy gradient's step 0; give --policy-gradient".to_string());
        }
        a.gate.validate().map_err(|e| format!("brain finetune: {e}"))?;
    }
    if a.lora.is_some() && a.train_precision.is_some() {
        return Err("brain finetune: --train-precision stores a full fine-tune's weights and optimizer state; an adapter run takes --base-format".to_string());
    }
    if let (Some(precision), Some(policy)) = (a.train_precision.as_mut(), a.activations) {
        crate::args::merge_activations(Some(policy), precision).map_err(|e| format!("brain finetune: {e}"))?;
    }
    if a.lora.is_none() {
        // Each of these shapes an adapter; a full fine-tune that was given one would silently ignore it.
        for (flag, given) in [("--alpha", a.alpha.is_some()), ("--continue", a.continue_from.is_some()), ("--adapter-id", a.adapter_id.is_some()), ("--base-format", a.base_format.is_some()), ("--held-out", a.held_out.is_some()), ("--policy-gradient", a.policy_gradient), ("--modules-to-save", !a.modules_to_save.is_empty())] {
            if given {
                return Err(format!("brain finetune: {flag} applies to an adapter; give --lora RANK"));
            }
        }
    }
    if job(&a)?.is_some() && a.objective.is_paired() {
        return Err("brain finetune: a distributed run (--job-plan) trains the sft and policy-gradient objectives; the paired objectives run in one process".to_string());
    }
    Ok(a)
}

/// The aggregation `--loss-agg` names (`None`: the configuration's own), its constant length from `--loss-agg-length`
/// or else `--max-block`.
fn loss_agg(a: &Args) -> Result<Option<brain::LossAgg>, String> {
    match (a.loss_agg.as_deref(), a.loss_agg_length) {
        (Some("constant-length"), length) => match length.or(a.max_block) {
            Some(n) => Ok(Some(brain::LossAgg::ConstantLength(n))),
            None => Err("brain finetune: --loss-agg constant-length divides by a constant; give --loss-agg-length N or --max-block N".to_string()),
        },
        // Dr. GRPO's own aggregation is a constant length: its constant alone is enough.
        (None, Some(_)) if a.algorithm == Some(brain::PolicyGradientAlgorithm::DrGrpo) => Ok(None),
        (_, Some(_)) => Err("brain finetune: --loss-agg-length is the constant of --loss-agg constant-length".to_string()),
        (Some(v), None) => v.parse().map(Some).map_err(|e| format!("brain finetune: --loss-agg: {e}")),
        (None, None) => Ok(None),
    }
}

/// The configuration `--algorithm` names (the default without one), before the flags are laid over it.
fn algorithm_config(a: &Args) -> Result<brain::PolicyGradientConfig, String> {
    let length = a.loss_agg_length.or(a.max_block);
    match (a.algorithm, length) {
        (None, _) => Ok(brain::PolicyGradientConfig::default()),
        (Some(brain::PolicyGradientAlgorithm::DrGrpo), None) => Err("brain finetune: --algorithm dr-grpo divides each sequence by a constant length; give --loss-agg-length N or --max-block N".to_string()),
        (Some(algorithm), Some(n)) => Ok(algorithm.config(n)),
        // Only Dr. GRPO reads the constant.
        (Some(algorithm), None) => Ok(algorithm.config(0)),
    }
}

fn lora(a: &Args, rank: u32) -> brain::ChatFineTune {
    let mut t = brain::ChatFineTune::from_pretrained(a.base.clone()).rank(rank).thinking(!a.no_thinking);
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
    if let Some(v) = a.steps {
        t = t.steps(v);
    }
    if let Some(v) = a.lr {
        t = t.lr(v);
    }
    if let Some(v) = a.weight_decay {
        t = t.weight_decay(v);
    }
    if let Some(v) = a.grad_accum {
        t = t.grad_accum(v);
    }
    t = t.strict_updates(a.strict_updates);
    if let Some(v) = a.stages {
        t = t.stages(v);
    }
    if let Some((cfg, long_range)) = sparse(a) {
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
    if let Some(v) = &a.held_out {
        t = t.held_out(v);
    }
    if !a.modules_to_save.is_empty() {
        t = t.modules_to_save(a.modules_to_save.clone());
    }
    if a.policy_gradient {
        let d = algorithm_config(a).expect("parse checked --algorithm");
        t = t.policy_gradient(brain::PolicyGradientConfig { clip_eps: a.clip_eps.unwrap_or(d.clip_eps), kl_beta: a.kl_beta.unwrap_or(d.kl_beta), behaviour: a.behaviour.unwrap_or(d.behaviour), numerics: a.numerics_gate.then_some(a.gate), kv_roundtrip: a.kv_roundtrip.or(d.kv_roundtrip), clip_eps_high: a.clip_eps_high.or(d.clip_eps_high), dual_clip: a.dual_clip.or(d.dual_clip), importance: a.importance.unwrap_or(d.importance), max_step_kl: a.max_step_kl.or(d.max_step_kl), max_step_grad_norm: a.max_step_grad_norm.or(d.max_step_grad_norm), advantage_norm: a.advantage_norm.unwrap_or(d.advantage_norm), loss_agg: loss_agg(a).expect("parse checked --loss-agg").unwrap_or(d.loss_agg), truncated: a.truncated.unwrap_or(d.truncated), advantage_floor: a.advantage_floor.unwrap_or(d.advantage_floor), ..d });
    }
    t
}

fn full(a: &Args) -> brain::ChatFullFineTune {
    let mut t = brain::ChatFullFineTune::from_pretrained(a.base.clone()).thinking(!a.no_thinking);
    if let Some(v) = &a.dataset {
        t = t.dataset(v);
    }
    if let Some(v) = &a.out {
        t = t.out_dir(v);
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
    if let Some(v) = a.weight_decay {
        t = t.weight_decay(v);
    }
    if let Some(v) = a.grad_accum {
        t = t.grad_accum(v);
    }
    t = t.strict_updates(a.strict_updates);
    if let Some(v) = a.stages {
        t = t.stages(v);
    }
    if let Some((cfg, long_range)) = sparse(a) {
        t = t.native_sparse_attention(cfg, long_range);
    }
    if let Some(v) = a.seed {
        t = t.seed(v);
    }
    if let Some(v) = a.max_block {
        t = t.max_block(v);
    }
    if let Some(precision) = &a.train_precision {
        t = t.train_precision(precision.clone());
    } else if let Some(v) = a.activations {
        t = t.train_precision(model::TrainPrecision { activations: Some(v), ..Default::default() });
    }
    t
}

/// The native sparse attention the flags ask for: the paper's configuration with the budget flags laid over it.
fn sparse(a: &Args) -> Option<(brain::LsaConfig, bool)> {
    a.native_sparse.then(|| {
        let d = brain::LsaConfig::default();
        (
            brain::LsaConfig {
                window: a.lsa_window.unwrap_or(d.window),
                sink: a.lsa_sink.unwrap_or(d.sink),
                top_k: a.lsa_top_k.unwrap_or(d.top_k),
                top_p: a.lsa_top_p.unwrap_or(d.top_p),
                cluster_block: a.lsa_cluster_block.unwrap_or(d.cluster_block),
                ..d
            },
            !a.lsa_window_only,
        )
    })
}

fn losses(initial: Option<f32>, last: Option<f32>) -> String {
    let show = |v: Option<f32>| v.map_or("-".to_string(), |v| format!("{v:.4}"));
    format!("loss {} -> {}", show(initial), show(last))
}

pub fn run_finetune(argv: &[String]) -> ! {
    let args = match parse(argv) {
        Ok(a) => a,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("{msg}\n");
            }
            print!("{USAGE}");
            std::process::exit(if msg.is_empty() { 0 } else { 2 });
        }
    };
    let code = match args.lora {
        Some(_) if args.objective.is_paired() => objectives::run(&args),
        Some(rank) => match lora(&args, rank).and_then_job(&args).run_with(&brain::CancelToken::default(), |p| eprintln!("step {}: loss {:.4}", p.step, p.loss)) {
            Ok(o) => {
                println!("status: {:?}, {} steps, {}", o.status, o.steps_completed, losses(o.initial_loss, o.final_loss));
                // The run's report, the document the served train job answers with; one writer per job.
                if let (Some(out), true, brain::FineTuneStatus::Completed) = (&args.out, args.job_rank.unwrap_or(0) == 0, o.status) {
                    let path = std::path::Path::new(out).join("report.json");
                    if let Err(e) = std::fs::write(&path, serde_json::to_vec_pretty(&o.report()).unwrap_or_default()) {
                        eprintln!("brain finetune: writing {}: {e}", path.display());
                    }
                }
                if let (Some(base), Some(tuned)) = (&o.base_score, &o.tuned_score) {
                    let show = |v: Option<f32>| v.map_or("-".to_string(), |v| format!("{v:.4}"));
                    println!("held-out: {} records, {} positions, loss {} -> {}", base.records, base.positions, show(base.loss), show(tuned.loss));
                }
                let mut gate_failed = false;
                if let Some(report) = o.policy_gradient.as_ref().and_then(|pg| pg.stats.numerics.as_ref()) {
                    println!("numerics: {}", serde_json::to_string_pretty(report).unwrap_or_default());
                    gate_failed = !report.passed();
                    for why in &report.violations {
                        eprintln!("brain finetune: numerics gate: {why}");
                    }
                }
                if let Some(pg) = &o.policy_gradient {
                    println!("policy gradient: {} steps ({} skipped by the guard), mean loss {:?}, mean clip fraction {:?}, mean kl {:?}", pg.stats.steps.len(), pg.stats.steps_skipped_guard, pg.stats.mean_loss(), pg.stats.mean_clip_fraction(), pg.stats.mean_kl());
                }
                if let Some(p) = &o.adapter {
                    println!("adapter: {}", p.display());
                }
                if let Some(p) = &o.resume_state {
                    println!("resume state: {}", p.display());
                }
                if gate_failed {
                    NUMERICS_GATE_FAILED
                } else {
                    0
                }
            }
            Err(e) => {
                eprintln!("brain finetune: {e}");
                1
            }
        },
        None => match full(&args).and_then_job(&args).run_with(&brain::CancelToken::default(), |p| eprintln!("step {}: loss {:.4}", p.step, p.loss)) {
            Ok(o) => {
                println!("status: {:?}, {} steps, {}", o.status, o.steps_completed, losses(o.initial_loss, o.final_loss));
                if let Some(p) = &o.checkpoint {
                    println!("checkpoint: {}", p.display());
                }
                0
            }
            Err(e) => {
                eprintln!("brain finetune: {e}");
                1
            }
        },
    };
    crate::drain_before_exit(code)
}

/// The job a `--job-plan`/`--job-rank`/`--job-secret` run joins, or `None`; part of the three is a usage error.
fn job(a: &Args) -> Result<Option<model::job_params::JobRequest>, String> {
    match (&a.job_plan, a.job_rank, &a.job_secret) {
        (None, None, None) => Ok(None),
        (Some(plan), Some(rank), Some(secret)) => Ok(Some(model::job_params::JobRequest { plan: plan.into(), rank, secret: secret.into(), profile: None })),
        _ => Err("brain finetune: --job-plan, --job-rank and --job-secret go together: the plan every rank shares, this rank's place in it, and the secret that proves it belongs".into()),
    }
}

/// A fine-tune with the rank of the job `a` names joined, if it names one.
trait JoinJob: Sized {
    fn and_then_job(self, a: &Args) -> Self;
}

impl JoinJob for brain::ChatFineTune {
    fn and_then_job(self, a: &Args) -> Self {
        let fail = |e: String| -> ! {
            eprintln!("{e}");
            std::process::exit(2)
        };
        // `parse` refused part of the three already.
        let Some(job) = job(a).unwrap_or_else(|e| fail(e)) else { return self };
        let seeded = match a.seed {
            Some(_) => self,
            None => self.seed(job.plan_seed().unwrap_or_else(|e| fail(format!("brain finetune: {e}")))),
        };
        let rank = brain::DataParallel::join_job(&job.plan, job.rank, &job.secret).unwrap_or_else(|e| fail(format!("brain finetune: {e}")));
        seeded.data_parallel(rank)
    }
}

impl JoinJob for brain::ChatFullFineTune {
    fn and_then_job(self, a: &Args) -> Self {
        let fail = |e: String| -> ! {
            eprintln!("{e}");
            std::process::exit(2)
        };
        let Some(job) = job(a).unwrap_or_else(|e| fail(e)) else { return self };
        let seeded = match a.seed {
            Some(_) => self,
            None => self.seed(job.plan_seed().unwrap_or_else(|e| fail(format!("brain finetune: {e}")))),
        };
        let rank = brain::DataParallel::join_job(&job.plan, job.rank, &job.secret).unwrap_or_else(|e| fail(format!("brain finetune: {e}")));
        seeded.data_parallel(rank)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    /// A distributed run names its plan, rank and secret together, and trains an adapter.
    #[test]
    fn a_job_is_all_three_flags_for_an_adapter_or_a_full_fine_tune() {
        let err = parse(&argv("base --dataset d --out o --lora 4 --job-plan p.json")).unwrap_err();
        assert!(err.contains("go together"), "{err}");
        let full = parse(&argv("base --dataset d --out o --job-plan p.json --job-rank 1 --job-secret s")).unwrap();
        assert_eq!(job(&full).unwrap().map(|j| j.rank), Some(1), "a full fine-tune runs as a rank of a job too");
        let a = parse(&argv("base --dataset d --out o --lora 4 --job-plan p.json --job-rank 1 --job-secret s")).unwrap();
        assert_eq!(job(&a).unwrap().map(|j| j.rank), Some(1));
        let err = parse(&argv("base --dataset d --out o --lora 4 --objective feedback --job-plan p.json --job-rank 1 --job-secret s")).unwrap_err();
        assert!(err.contains("one process"), "{err}");
    }

    #[test]
    fn a_full_fine_tune_refuses_the_flags_that_shape_an_adapter() {
        for flag in ["--alpha 4", "--continue a.safetensors", "--adapter-id x", "--base-format nf4"] {
            let err = parse(&argv(&format!("base --dataset d --out o {flag}"))).unwrap_err();
            assert!(err.contains("--lora"), "{flag}: {err}");
        }
        assert!(parse(&argv("base --dataset d --out o --lora 4 --alpha 8 --base-format nf4")).is_ok());
    }

    #[test]
    fn the_policy_gradient_settings_need_the_policy_gradient_and_an_adapter() {
        let err = parse(&argv("base --lora 4 --ratio-clip 0.1")).unwrap_err();
        assert!(err.contains("--policy-gradient"), "{err}");
        let err = parse(&argv("base --policy-gradient")).unwrap_err();
        assert!(err.contains("--lora"), "{err}");
        let err = parse(&argv("base --held-out h.jsonl")).unwrap_err();
        assert!(err.contains("--lora"), "{err}");
        let a = parse(&argv("base --lora 4 --policy-gradient --kl-beta 0.1 --held-out h.jsonl")).unwrap();
        assert!(a.policy_gradient && a.kl_beta == Some(0.1) && a.held_out.as_deref() == Some("h.jsonl"));
    }

    /// The help lists exactly the values `--activations` accepts, from the one list the parser and every capability
    /// read, and each of them parses.
    #[test]
    fn the_activations_help_names_what_the_parser_accepts() {
        let choices = model::saved::SavedPolicy::ACTIVATIONS_CHOICES;
        let line = USAGE.lines().find(|l| l.trim_start().starts_with("--activations")).expect("--activations is in the help");
        assert!(line.trim_end().ends_with(&format!(": {}.", choices.join("|"))), "the help lists {choices:?}: {line}");
        for value in choices {
            assert!(parse(&argv(&format!("base --activations {value}"))).is_ok(), "{value} parses");
        }
        assert!(parse(&argv("base --activations fp8")).unwrap_err().contains(&choices.join(", ")));
    }

    #[test]
    fn the_train_precision_is_a_full_fine_tunes_and_folds_in_the_activations() {
        let a = parse(&argv("base --train-precision weights=bf16,window=2 --activations recompute")).unwrap();
        let precision = a.train_precision.unwrap();
        assert!(precision.resident.is_some_and(|r| r.window == Some(2)) && precision.activations.is_some());
        let staged = parse(&argv("base --stages 2 --train-precision weights=bf16")).unwrap();
        assert_eq!((staged.stages, staged.train_precision.is_some_and(|p| p.resident.is_some())), (Some(2), true), "a pipeline of resident stages is one command");
        assert!(parse(&argv("base --lora 4 --train-precision weights=bf16")).unwrap_err().contains("--base-format"));
        assert!(parse(&argv("base --train-precision weights=quantum")).unwrap_err().contains("--train-precision"));
        assert!(parse(&argv("base --train-precision activations=keep --activations recompute")).unwrap_err().contains("conflicts"));
    }

    #[test]
    fn the_stability_settings_need_the_policy_gradient_and_parse_by_name() {
        let err = parse(&argv("base --lora 4 --max-step-kl 0.1")).unwrap_err();
        assert!(err.contains("--max-step-kl") && err.contains("--policy-gradient"), "{err}");
        let a = parse(&argv("base --lora 4 --policy-gradient --ratio-clip-high 0.28 --importance truncate:2 --max-step-kl 0.1 --max-step-grad-norm 5 --advantage-norm leave-one-out")).unwrap();
        assert_eq!(a.advantage_norm, Some(brain::AdvantageNorm::LeaveOneOut));
        assert!(parse(&argv("base --lora 4 --advantage-norm z-score")).unwrap_err().contains("--policy-gradient"));
        assert!(parse(&argv("base --lora 4 --policy-gradient --advantage-norm standardised")).unwrap_err().contains("leave-one-out"));
        assert_eq!((a.clip_eps_high, a.max_step_kl, a.max_step_grad_norm), (Some(0.28), Some(0.1), Some(5.0)));
        assert_eq!(a.importance, Some(brain::ImportanceCorrection::Truncate { cap: 2.0 }));
        assert_eq!(parse(&argv("base --lora 4 --policy-gradient --dual-clip 5")).unwrap().dual_clip, Some(5.0));
        assert!(parse(&argv("base --lora 4 --dual-clip 5")).unwrap_err().contains("--policy-gradient"));
        let err = parse(&argv("base --lora 4 --policy-gradient --importance sometimes")).unwrap_err();
        assert!(err.contains("--importance") && err.contains("truncate:CAP"), "{err}");
    }

    #[test]
    fn the_constant_length_aggregation_takes_its_length_or_the_max_block() {
        let agg = |s: &str| parse(&argv(s)).and_then(|a| loss_agg(&a));
        assert_eq!(agg("base --lora 4 --policy-gradient --loss-agg constant-length --loss-agg-length 512"), Ok(Some(brain::LossAgg::ConstantLength(512))));
        assert_eq!(agg("base --lora 4 --policy-gradient --loss-agg constant-length --max-block 1024"), Ok(Some(brain::LossAgg::ConstantLength(1024))));
        assert_eq!(agg("base --lora 4 --policy-gradient --loss-agg token-sum"), Ok(Some(brain::LossAgg::TokenSum)));
        assert!(agg("base --lora 4 --policy-gradient --loss-agg constant-length").unwrap_err().contains("--loss-agg-length"));
        assert!(agg("base --lora 4 --policy-gradient --loss-agg token-sum --loss-agg-length 8").unwrap_err().contains("constant-length"));
        assert!(agg("base --lora 4 --loss-agg token-sum").unwrap_err().contains("--policy-gradient"));
        assert_eq!(parse(&argv("base --lora 4 --policy-gradient --truncated keep")).unwrap().truncated, Some(brain::TruncatedRollouts::Keep));
        assert!(parse(&argv("base --lora 4 --truncated keep")).unwrap_err().contains("--policy-gradient"));
        assert!(parse(&argv("base --lora 4 --policy-gradient --truncated drop")).unwrap_err().contains("mask"));
        assert_eq!(parse(&argv("base --lora 4 --policy-gradient --advantage-floor zero")).unwrap().advantage_floor, Some(brain::AdvantageFloor::Zero));
        assert!(parse(&argv("base --lora 4 --advantage-floor zero")).unwrap_err().contains("--policy-gradient"));
    }

    #[test]
    fn a_named_algorithm_is_the_base_the_flags_override() {
        let config = |s: &str| parse(&argv(s)).and_then(|a| algorithm_config(&a));
        assert_eq!(config("base --lora 4 --policy-gradient --algorithm dapo"), Ok(brain::PolicyGradientAlgorithm::Dapo.config(0)));
        assert_eq!(config("base --lora 4 --policy-gradient --algorithm dr-grpo --max-block 256").map(|c| c.loss_agg), Ok(brain::LossAgg::ConstantLength(256)));
        assert_eq!(config("base --lora 4 --policy-gradient --algorithm dr-grpo --loss-agg-length 64").map(|c| c.loss_agg), Ok(brain::LossAgg::ConstantLength(64)));
        assert!(config("base --lora 4 --policy-gradient --algorithm dr-grpo").unwrap_err().contains("--loss-agg-length"));
        assert!(parse(&argv("base --lora 4 --algorithm rloo")).unwrap_err().contains("--policy-gradient"));
        assert!(parse(&argv("base --lora 4 --policy-gradient --algorithm ppo")).unwrap_err().contains("dr-grpo"));
        let a = parse(&argv("base --lora 4 --policy-gradient --algorithm dapo --ratio-clip-high 0.4 --kl-beta 0.01")).unwrap();
        assert_eq!((a.clip_eps_high, a.kl_beta), (Some(0.4), Some(0.01)));
    }

    #[test]
    fn the_numerics_gate_needs_the_policy_gradient_and_takes_its_bounds_by_name() {
        assert!(parse(&argv("base --lora 4 --numerics-gate")).unwrap_err().contains("--policy-gradient"));
        assert!(parse(&argv("base --lora 4 --policy-gradient --gate-max-gap 0.5")).unwrap_err().contains("--numerics-gate"));
        let a = parse(&argv("base --lora 4 --policy-gradient --numerics-gate --gate-max-mean-gap 0.02 --gate-max-gap 0.5 --gate-max-mismatch-kl 0.001 --gate-min-sequence-ess 0.5 --gate-max-grad-norm 5")).unwrap();
        assert!(a.numerics_gate);
        assert_eq!((a.gate.max_mean_abs_gap, a.gate.max_abs_gap, a.gate.max_mismatch_kl, a.gate.min_sequence_ess_fraction, a.gate.max_grad_norm), (Some(0.02), Some(0.5), Some(0.001), Some(0.5), Some(5.0)));
        assert!(parse(&argv("base --lora 4 --policy-gradient --numerics-gate --gate-min-sequence-ess 2")).unwrap_err().contains("min_sequence_ess_fraction"));
    }

    #[test]
    fn each_paired_objective_trains_an_adapter_and_takes_only_its_own_settings() {
        let err = |s: &str| parse(&argv(s)).unwrap_err();
        for objective in ["preference", "feedback", "distill"] {
            assert!(err(&format!("base --objective {objective}")).contains("--lora"), "{objective}");
        }
        assert!(err("base --objective pairwise").contains("preference"));
        assert!(err("base --lora 4 --policy-gradient --objective preference").contains("contradicts"));
        assert_eq!(parse(&argv("base --lora 4 --objective policy-gradient")).unwrap().objective, Objective::PolicyGradient);
        assert_eq!(parse(&argv("base --lora 4 --policy-gradient")).unwrap().objective, Objective::PolicyGradient);
        assert_eq!(parse(&argv("base --lora 4")).unwrap().objective, Objective::Sft);
        // a setting of one objective is refused beside another, by name
        assert!(err("base --lora 4 --loss ipo").contains("--objective preference"));
        assert!(err("base --lora 4 --objective feedback --loss ipo").contains("--objective preference"));
        assert!(err("base --lora 4 --objective preference --desirable-weight 2").contains("--objective feedback"));
        assert!(err("base --lora 4 --objective preference --top-k 4").contains("--objective distill"));
        assert!(err("base --lora 4 --objective distill --beta 0.2").contains("preference or feedback"));
        assert!(err("base --lora 4 --keep-reasoning").contains("preference or feedback"));
        // and so is a setting the objective's trainer does not read
        assert!(err("base --lora 4 --objective feedback --weight-decay 0.1").contains("--weight-decay"));
        assert!(err("base --lora 4 --objective distill --held-out h.jsonl").contains("--held-out"));
        assert!(err("base --lora 4 --objective preference --no-thinking").contains("--no-thinking"));
        assert!(parse(&argv("base --lora 4 --objective preference --held-out h.jsonl --stages 2 --activations recompute")).is_ok());
    }

    #[test]
    fn a_preference_loss_is_named_and_simpo_alone_takes_a_margin() {
        let loss = |s: &str| parse(&argv(&format!("base --lora 4 --objective preference {s}"))).and_then(|a| objectives::pair_loss(&a));
        assert_eq!(loss(""), Ok(brain::PairLoss::Dpo));
        assert_eq!(loss("--loss ipo"), Ok(brain::PairLoss::Ipo));
        assert_eq!(loss("--loss orpo --beta 0.2"), Ok(brain::PairLoss::Orpo));
        assert_eq!(loss("--loss simpo"), Ok(brain::PairLoss::SimPo { gamma: brain::DEFAULT_SIMPO_GAMMA }));
        assert_eq!(loss("--loss simpo --gamma 1.2"), Ok(brain::PairLoss::SimPo { gamma: 1.2 }));
        assert!(loss("--loss dpo --gamma 1").unwrap_err().contains("simpo"));
        let err = loss("--loss kto").unwrap_err();
        assert!(err.contains("--loss") && err.contains("simpo"), "{err}");
    }

    #[test]
    fn a_distillation_teacher_is_the_base_the_start_adapter_or_an_adapter_file() {
        let teacher = |s: &str| parse(&argv(&format!("base --lora 4 --objective distill {s}"))).map(|a| a.teacher);
        assert_eq!(teacher(""), Ok(None));
        assert_eq!(teacher("--teacher base"), Ok(Some(objectives::Teacher::Base)));
        assert_eq!(teacher("--teacher start --continue a.safetensors"), Ok(Some(objectives::Teacher::Start)));
        assert_eq!(teacher("--teacher adapter:t.safetensors --top-k 4"), Ok(Some(objectives::Teacher::Adapter("t.safetensors".into()))));
        assert!(teacher("--teacher start").unwrap_err().contains("--continue"));
        assert!(teacher("--teacher adapter:").unwrap_err().contains("adapter:FILE"));
        assert!(teacher("--teacher tutor").unwrap_err().contains("base, start"));
    }

    #[test]
    fn strict_updates_is_a_flag_and_off_by_default() {
        assert!(!parse(&argv("base --steps 3")).unwrap().strict_updates);
        assert!(parse(&argv("base --strict-updates")).unwrap().strict_updates);
    }

    #[test]
    fn a_number_that_does_not_parse_is_an_error_naming_the_flag() {
        let err = parse(&argv("base --steps many")).unwrap_err();
        assert!(err.contains("--steps") && err.contains("many"), "{err}");
    }

    #[test]
    fn the_usage_names_every_flag_the_parser_takes() {
        for flag in ["--dataset", "--out", "--lora", "--alpha", "--continue", "--adapter-id", "--arch", "--models-dir", "--steps", "--lr", "--weight-decay", "--grad-accum", "--strict-updates", "--stages", "--native-sparse", "--lsa-window", "--lsa-sink", "--lsa-top-k", "--lsa-top-p", "--lsa-cluster-block", "--lsa-window-only", "--seed", "--max-block", "--base-format", "--activations", "--no-thinking", "--train-precision", "--held-out", "--modules-to-save", "--policy-gradient", "--algorithm", "--loss-agg", "--loss-agg-length", "--truncated", "--advantage-floor", "--ratio-clip", "--kl-beta", "--behaviour", "--ratio-clip-high", "--dual-clip", "--importance", "--max-step-kl", "--max-step-grad-norm", "--numerics-gate", "--gate-max-mean-gap", "--gate-max-gap", "--gate-max-mismatch-kl", "--gate-min-sequence-ess", "--gate-max-grad-norm", "--kv-roundtrip", "--objective", "--loss", "--gamma", "--beta", "--nll-weight", "--keep-reasoning", "--desirable-weight", "--undesirable-weight", "--teacher", "--top-k"] {
            assert!(USAGE.contains(flag), "{flag} is parsed but absent from the usage");
        }
    }
}
