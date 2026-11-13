// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain finetune` - supervised fine-tuning of a chat model through the SDK's generic trainers
//! ([`brain::ChatFineTune`], [`brain::ChatFullFineTune`]); the architecture is a flag, not a token.
//!
//! Swedish Embedded AB implements solutions for adapting open-weight language models to a team's own data on the
//! memory it actually has, for its clients. If your team needs expertise in fine-tuning large language models of
//! any architecture then you can procure our services by sending an email to info@swedishembedded.com.

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
  --policy-gradient  train the clipped policy gradient on rollout records (`advantage` on each record) instead of the
                     supervised loss, for an adapter; --ratio-clip X and --kl-beta X set its clip band and KL weight.
  --ratio-clip-high X
                     with --policy-gradient: widen the clip band's upper edge to 1+X (clip-higher; at least --ratio-clip).
  --importance C     with --policy-gradient: correct for the sampling engine's gap to the trainer: none (default),
                     truncate:CAP (weight each token's policy term by min(pi_trainer / pi_rollout, CAP)) or mask:CAP
                     (drop tokens whose weight leaves [1/CAP, CAP]); needs recorded behaviour log-probabilities.
  --max-step-kl X --max-step-grad-norm X
                     with --policy-gradient: skip, and log, an optimiser step whose KL to the reference, or whose gradient
                     norm, is over the bound (a run that skips 8 steps in a row stops).
  --behaviour B      with --policy-gradient: where the behaviour log-probabilities of the ratio come from.
                     recorded (default): the sampler's own, on every supervised token of every record; start_of_run_proxy:
                     where none was recorded, the start-of-run policy's (exact only if the rollouts were sampled from it).
  --activations P    what the backward keeps of the forward: keep|recompute|bf16|fp8.
  --train-precision SPEC
                     a full fine-tune's storage: `state=f32|bf16|int8[:B]|log8[:B]` for the optimizer state,
                     `weights=bf16|fp8[:B]` to hold the linears and the tables on the device in that encoding with
                     their f32 master copy and the optimizer state off it, `window=N` to keep only N layers on the
                     device (the rest streamed from the host), `clip=two-pass|running`. An architecture that does not
                     run a choice refuses it by name.
  --no-thinking      render records without the reasoning block.

Run on a device with `--device` (default: every device brain finds).
";

#[derive(Debug, Default)]
struct Args {
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
    clip_eps: Option<f32>,
    kl_beta: Option<f32>,
    behaviour: Option<brain::BehaviourSource>,
    clip_eps_high: Option<f32>,
    importance: Option<brain::ImportanceCorrection>,
    max_step_kl: Option<f32>,
    max_step_grad_norm: Option<f32>,
}

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
            "--train-precision" => {
                let spec = need(&mut i, &flag)?;
                a.train_precision = Some(model::TrainPrecision::parse(&spec).map_err(|e| format!("brain finetune: --train-precision {spec:?}: {e}"))?);
            }
            "--held-out" => a.held_out = Some(need(&mut i, &flag)?),
            "--modules-to-save" => a.modules_to_save = need(&mut i, &flag)?.split(',').map(str::trim).filter(|p| !p.is_empty()).map(String::from).collect(),
            "--policy-gradient" => a.policy_gradient = true,
            "--ratio-clip" => a.clip_eps = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--kl-beta" => a.kl_beta = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--behaviour" => a.behaviour = Some(need(&mut i, &flag)?.parse().map_err(|e| format!("brain finetune: {flag}: {e}"))?),
            "--ratio-clip-high" => a.clip_eps_high = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--importance" => a.importance = Some(need(&mut i, &flag)?.parse().map_err(|e| format!("brain finetune: {flag}: {e}"))?),
            "--max-step-kl" => a.max_step_kl = Some(number(&need(&mut i, &flag)?, &flag)?),
            "--max-step-grad-norm" => a.max_step_grad_norm = Some(number(&need(&mut i, &flag)?, &flag)?),
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
    // The budget flags shape the selection; without --native-sparse they would be ignored.
    if !a.native_sparse {
        for (flag, given) in [("--lsa-window", a.lsa_window.is_some()), ("--lsa-sink", a.lsa_sink.is_some()), ("--lsa-top-k", a.lsa_top_k.is_some()), ("--lsa-top-p", a.lsa_top_p.is_some()), ("--lsa-cluster-block", a.lsa_cluster_block.is_some()), ("--lsa-window-only", a.lsa_window_only)] {
            if given {
                return Err(format!("brain finetune: {flag} shapes native sparse attention; give --native-sparse"));
            }
        }
    }
    if !a.policy_gradient {
        for (flag, given) in [("--ratio-clip", a.clip_eps.is_some()), ("--kl-beta", a.kl_beta.is_some()), ("--behaviour", a.behaviour.is_some()), ("--ratio-clip-high", a.clip_eps_high.is_some()), ("--importance", a.importance.is_some()), ("--max-step-kl", a.max_step_kl.is_some()), ("--max-step-grad-norm", a.max_step_grad_norm.is_some())] {
            if given {
                return Err(format!("brain finetune: {flag} shapes the policy gradient; give --policy-gradient"));
            }
        }
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
    Ok(a)
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
        let d = brain::PolicyGradientConfig::default();
        t = t.policy_gradient(brain::PolicyGradientConfig { clip_eps: a.clip_eps.unwrap_or(d.clip_eps), kl_beta: a.kl_beta.unwrap_or(d.kl_beta), behaviour: a.behaviour.unwrap_or(d.behaviour), clip_eps_high: a.clip_eps_high, importance: a.importance.unwrap_or(d.importance), max_step_kl: a.max_step_kl, max_step_grad_norm: a.max_step_grad_norm, ..d });
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
        Some(rank) => match lora(&args, rank).run_with(&brain::CancelToken::default(), |p| eprintln!("step {}: loss {:.4}", p.step, p.loss)) {
            Ok(o) => {
                println!("status: {:?}, {} steps, {}", o.status, o.steps_completed, losses(o.initial_loss, o.final_loss));
                if let (Some(base), Some(tuned)) = (&o.base_score, &o.tuned_score) {
                    let show = |v: Option<f32>| v.map_or("-".to_string(), |v| format!("{v:.4}"));
                    println!("held-out: {} records, {} positions, loss {} -> {}", base.records, base.positions, show(base.loss), show(tuned.loss));
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
                0
            }
            Err(e) => {
                eprintln!("brain finetune: {e}");
                1
            }
        },
        None => match full(&args).run_with(&brain::CancelToken::default(), |p| eprintln!("step {}: loss {:.4}", p.step, p.loss)) {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn argv(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
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
        let a = parse(&argv("base --lora 4 --policy-gradient --ratio-clip-high 0.28 --importance truncate:2 --max-step-kl 0.1 --max-step-grad-norm 5")).unwrap();
        assert_eq!((a.clip_eps_high, a.max_step_kl, a.max_step_grad_norm), (Some(0.28), Some(0.1), Some(5.0)));
        assert_eq!(a.importance, Some(brain::ImportanceCorrection::Truncate { cap: 2.0 }));
        let err = parse(&argv("base --lora 4 --policy-gradient --importance sometimes")).unwrap_err();
        assert!(err.contains("--importance") && err.contains("truncate:CAP"), "{err}");
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
        for flag in ["--dataset", "--out", "--lora", "--alpha", "--continue", "--adapter-id", "--arch", "--models-dir", "--steps", "--lr", "--weight-decay", "--grad-accum", "--strict-updates", "--stages", "--native-sparse", "--lsa-window", "--lsa-sink", "--lsa-top-k", "--lsa-top-p", "--lsa-cluster-block", "--lsa-window-only", "--seed", "--max-block", "--base-format", "--activations", "--no-thinking", "--train-precision", "--held-out", "--modules-to-save", "--policy-gradient", "--ratio-clip", "--kl-beta", "--behaviour", "--ratio-clip-high", "--importance", "--max-step-kl", "--max-step-grad-norm"] {
            assert!(USAGE.contains(flag), "{flag} is parsed but absent from the usage");
        }
    }
}
