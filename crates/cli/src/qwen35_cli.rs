// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain qwen35 ...` - run the Qwen3.8-27B dense hybrid decoder.
//!
//!   brain qwen35 infer  --weights F [--tokenizer tokenizer.json] --prompt "..."
//!                     [--adapter adapter.safetensors] [--max-new N --temp X --top-k K --chat]
//!   brain qwen35 finetune <data_dir> --base F --out F [--mode lora|full]
//!                     [--rank R --alpha A] [--steps N --lr X ...]
//!
//! GGUF import lives in the GENERIC `brain import-gguf` command
//! ([`crate::gguf_import`]), which dispatches on the file's own
//! `general.architecture`; `brain qwen35 import` remains as a deprecated
//! forward to it - mirrors `qwen35moe_cli`'s own `import` exactly. This
//! architecture now HAS a GGUF importer registered there
//! (`qwen35::gguf_import`, `general.architecture = "qwen35"`), a second
//! source format alongside the HF-safetensors FP8 route the upstream
//! checkpoint ships as; what is unique to it is that it imports the MTP
//! head, which the sibling `qwen35moe` GGUF importer drops from its own
//! checkpoint.
//!
//! No `export` subcommand: unlike `qwen35moe`, this crate has no NPU/ONNX
//! export path (`npu::qwen35_export` does not exist) - out of scope for
//! this port, matching the recorded NPU gap.
//!
//! `qwen35::model::Qwen35`'s public constructors (`new_on`) take a
//! `&HashMap<String, Vec<f32>>`, matching every other model crate's
//! simplest load path (`checkpoint::load(path).by_role("")`).

use std::path::Path;

use crate::args::Args;
use data::rng::Rng;
use data::tokenizer::Tokenizer;
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

pub fn run_qwen35(args: &[String]) {
    match args.first().map(|s| crate::args::canon_verb(s)) {
        Some("import") => import(&args[1..]),
        Some("infer") => infer(&args[1..]),
        Some("finetune") => finetune(&args[1..]),
        other => eprintln!("usage: brain qwen35 <import|infer|finetune> ...  (got {other:?})"),
    }
}

fn val(args: &[String], i: &mut usize, flag: &str) -> String {
    *i += 1;
    args.get(*i).cloned().unwrap_or_else(|| {
        eprintln!("{flag} requires a value");
        std::process::exit(2);
    })
}

/// `brain qwen35 import --gguf FILE --out qwen35.safetensors` - **deprecated**
/// alias for the generic `brain import-gguf FILE [--out PATH] [--id NAME]`.
/// Mirrors `qwen35moe_cli::import` exactly.
fn import(args: &[String]) {
    eprintln!("brain qwen35 import is deprecated -- use `brain import-gguf FILE [--out PATH] [--id NAME]`");
    crate::gguf_import::run_import_gguf(args);
}

/// `brain qwen35 infer --weights F [--tokenizer T | --gguf G] --prompt "..."
/// [--adapter FILE]`: single-sequence greedy/sampled generation via
/// `Qwen35::step`, through `qwen35::sample::generate_kv` (its own decode
/// path). Not the paged `PagedDecoder`/`Scheduler` serving path
/// (`qwen35::serve`) - this is the same "simple, direct, one request" tier
/// `qwen3::sample::generate_kv` occupies alongside `qwen3::serve::Engine`.
///
/// `--adapter FILE` names a LoRA adapter saved by `qwen35::lora::
/// save_adapter` (an adapter-only `.safetensors`, NOT a full checkpoint):
/// its `.lora_a`/`.lora_b` deltas are folded into the loaded base tensors
/// once, before `Qwen35::new_on` ever runs (`qwen35::lora::
/// fold_adapter_into`, gated exact against the live unfolded forward by
/// `crates/qwen35/tests/lora_adapter_file.rs`) - so every downstream stage
/// (KV cache, sampling) sees a plain adapted model at zero extra per-token
/// cost, with no other code path in this function aware an adapter was even
/// involved. A plain local file path (unlike `qwen_cli::finetune_lora`'s
/// `OWNER/NAME[:TAG]` model-store refs) - this crate has no `ModelStore`/
/// `ModelRef` integration yet, so wiring that here would be new scope well
/// beyond making the fold reachable.
fn infer(args: &[String]) {
    let mut weights = String::new();
    let mut tokenizer = String::new();
    let mut gguf_for_tok = String::new();
    let mut prompt = String::new();
    let mut adapter = String::new();
    let mut max_new = 32usize;
    let mut temp = 0.0f32;
    let mut top_k = 0usize;
    let mut top_p = 1.0f32;
    let mut seed = 0u64;
    let mut chat = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = val(args, &mut i, "--weights"),
            "--tokenizer" => tokenizer = val(args, &mut i, "--tokenizer"),
            "--gguf" => gguf_for_tok = val(args, &mut i, "--gguf"),
            "--prompt" => prompt = val(args, &mut i, "--prompt"),
            "--adapter" => adapter = val(args, &mut i, "--adapter"),
            "--max-new" => max_new = val(args, &mut i, "--max-new").parse().unwrap_or(max_new),
            "--temp" => temp = val(args, &mut i, "--temp").parse().unwrap_or(temp),
            "--top-k" => top_k = val(args, &mut i, "--top-k").parse().unwrap_or(top_k),
            "--top-p" => top_p = val(args, &mut i, "--top-p").parse().unwrap_or(top_p),
            "--seed" => seed = val(args, &mut i, "--seed").parse().unwrap_or(seed),
            "--chat" => chat = true,
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }
    if weights.is_empty() || (tokenizer.is_empty() && gguf_for_tok.is_empty()) {
        eprintln!(
            "usage: brain qwen35 infer --weights F (--tokenizer tokenizer.json | --gguf original.gguf) --prompt \"...\" \
             [--adapter adapter.safetensors] [--max-new N --temp X --top-k K --top-p P --seed S --chat]"
        );
        return;
    }

    let tok = if !tokenizer.is_empty() {
        data::qwen_tokenizer::QwenBpe::from_file(&tokenizer)
    } else {
        checkpoint::gguf::MmapGguf::open(&gguf_for_tok)
            .map_err(|e| format!("open {gguf_for_tok}: {e}"))
            .and_then(|mg| mg.tokenizer().ok_or_else(|| format!("{gguf_for_tok}: no embedded tokenizer")))
            .and_then(|t| data::qwen_tokenizer::QwenBpe::from_gguf(&t))
    };
    let tok = match tok {
        Ok(t) => t,
        Err(e) => {
            eprintln!("tokenizer load failed: {e}");
            return;
        }
    };

    let text = if chat { tok.apply_chat_template(&[("user", &prompt)], true) } else { prompt.clone() };
    let ids = tok.encode(&text);
    if ids.is_empty() {
        eprintln!("empty prompt");
        return;
    }

    let container = checkpoint::load(&weights);
    let cfg = Qwen35Config::from_json(&container.header["config"]);
    let mut init = container.by_role("");
    if !adapter.is_empty() {
        if let Err(e) = qwen35::lora::fold_adapter_into(&mut init, &adapter) {
            eprintln!("--adapter {adapter:?}: {e}");
            return;
        }
    }
    let cap = (ids.len() + max_new) as u32;

    let t_load = std::time::Instant::now();
    let model = Qwen35::new_on(gpu_core::Gpu::new(pipelines()), cfg, 1, cap, &init);
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;

    let eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.encode(s).first().copied()).collect();
    let mut rng = Rng::new(seed);
    let t_gen = std::time::Instant::now();
    let gen = qwen35::sample::generate_kv(&model, &ids, max_new, temp, top_k, top_p, &eos_ids, &mut rng);
    let gen_ms = t_gen.elapsed().as_secs_f64() * 1e3;

    eprintln!("qwen35-timing load_ms={load_ms:.1} gen_ms={gen_ms:.1} tokens={}", gen.len());
    print!("{prompt}");
    print!("{}", tok.decode(&gen));
    println!();
}

/// LoRA rank used by `--mode lora` when `--rank` is omitted: 8, the rank this
/// model family's LoRA work is documented against (see AGENTS.md's
/// `qwen35moe` entry). `--alpha` then defaults to `2 * rank`, the convention
/// every other adapter-training verb in this crate already uses
/// (`qwen_cli::finetune_lora`, `forecast_cli::train`).
const DEFAULT_LORA_RANK: u32 = 8;

const FINETUNE_USAGE: &str = "usage: brain qwen35 finetune <data_dir> --base F --out F [--mode lora|full] [--rank R --alpha A] \
     [--steps N --lr X --batch B --block T --grad-accum G --warmup W --weight-decay X --grad-clip X --save-secs S --seed S --mask = --align]";

/// Everything [`qwen35::finetune::finetune`] needs, parsed off the argv.
/// Separated from [`finetune`] so the flag grammar - which defaults are
/// derived, which are the caller's - is testable without a GPU, a checkpoint
/// or a corpus.
#[derive(Debug)]
struct FinetuneArgs {
    base: String,
    data_dir: String,
    out: String,
    opts: model::FitOpts,
    mode: qwen35::finetune::Mode,
}

fn parse_finetune(args: &[String]) -> Result<FinetuneArgs, String> {
    let mut a = Args::new(args);
    let base = a.str_or("--base", "");
    let out = a.str_or("--out", "");
    let mode_name = a.str_or("--mode", "full");
    let rank = a.u32_or("--rank", DEFAULT_LORA_RANK);
    let alpha = a.f32_or("--alpha", rank as f32 * 2.0);

    let d = model::FitOpts::default();
    let steps = a.u32_or("--steps", d.steps);
    let lr = a.f32_or("--lr", d.lr);
    let mask = a.char_opt("--mask");
    let opts = model::FitOpts {
        steps,
        batch_size: a.u32_or("--batch", d.batch_size),
        block_size: a.u32_or("--block", d.block_size),
        lr,
        // Derived from THIS run's length and lr, exactly as `brain qwen3
        // train` / `brain glm train` derive them - not left at the struct
        // default. `FitOpts::default()`'s `decay_iters: 2000` against a
        // `--steps 50` finetune would spend the whole run in the first 2.5%
        // of the cosine schedule, so the LR would never anneal at all.
        decay_iters: steps,
        min_lr: lr * 0.1,
        warmup: a.u32_or("--warmup", d.warmup),
        weight_decay: a.f32_or("--weight-decay", d.weight_decay),
        grad_clip: a.f32_or("--grad-clip", d.grad_clip),
        grad_accum: a.u32_or("--grad-accum", d.grad_accum),
        checkpoint_secs: a.u64_or("--save-secs", d.checkpoint_secs),
        seed: a.u64_or("--seed", d.seed),
        mask_before: mask,
        mask_per_line: mask.is_some(),
        align_to_lines: a.take_flag("--align"),
        ..d
    };
    // The corpus is positional, matching `brain glm finetune <data_dir> ...`
    // and `brain flux2 finetune <data_dir> ...`. Taken LAST, after every
    // value flag has claimed its own value: `Args::positional` returns the
    // first token not yet marked used, so asking for it first would happily
    // return `--base`'s argument out of `finetune --base b.safetensors corpus`.
    let data_dir = a.positional().unwrap_or_default();
    a.finish();

    // `full` is the default mode because that is what the sibling verbs mean
    // by a plain finetune: `brain qwen3 finetune` and `brain glm finetune`
    // are both full-parameter unless a LoRA rank is asked for explicitly.
    let mode = match mode_name.as_str() {
        "full" => qwen35::finetune::Mode::FullOffload,
        "lora" => qwen35::finetune::Mode::Lora { rank, alpha },
        other => return Err(format!("--mode {other:?}: expected \"lora\" or \"full\"")),
    };
    if data_dir.is_empty() {
        return Err("the finetune corpus directory is required (a positional argument)".into());
    }
    if base.is_empty() {
        return Err("--base is required (a brain-native checkpoint, e.g. from `brain import-gguf`)".into());
    }
    if out.is_empty() {
        return Err("--out is required (where to write the finetuned checkpoint)".into());
    }
    Ok(FinetuneArgs { base, data_dir, out, opts, mode })
}

/// `brain qwen35 finetune <data_dir> --base F --out F [--mode lora|full] ...`:
/// full-parameter (AdamW moments offloaded to system RAM) or LoRA fine-tuning
/// of a brain-native qwen35 checkpoint, via [`qwen35::finetune::finetune`].
///
/// `<data_dir>` is a `model::load_dataset` corpus - `train.u32.bin` /
/// `val.u32.bin` / `meta.json`, optionally with the token-level
/// `train.mask.bin` a chat/tool-call dataset carries (auto-detected; `--mask`
/// is only for the char-boundary kind).
fn finetune(args: &[String]) {
    let ft = match parse_finetune(args) {
        Ok(ft) => ft,
        Err(e) => {
            eprintln!("brain qwen35 finetune: {e}");
            eprintln!("{FINETUNE_USAGE}");
            return;
        }
    };
    let mode_desc = match &ft.mode {
        qwen35::finetune::Mode::FullOffload => "full (offloaded AdamW moments)".to_string(),
        qwen35::finetune::Mode::Lora { rank, alpha } => format!("lora rank={rank} alpha={alpha}"),
    };
    eprintln!(
        "qwen35 finetune: {} on {} [{mode_desc}] steps={} batch={} block={} lr={} grad-accum={} seed={} -> {}",
        ft.base, ft.data_dir, ft.opts.steps, ft.opts.batch_size, ft.opts.block_size, ft.opts.lr, ft.opts.grad_accum, ft.opts.seed, ft.out
    );
    let t0 = std::time::Instant::now();
    match qwen35::finetune::finetune(&ft.base, Path::new(&ft.data_dir), &ft.opts, &ft.mode, &ft.out) {
        Ok((initial_loss, final_loss)) => eprintln!(
            "qwen35 finetune: loss {initial_loss:.4} -> {final_loss:.4} in {:.1}s; saved {}",
            t0.elapsed().as_secs_f64(),
            ft.out
        ),
        Err(e) => eprintln!("qwen35 finetune failed: {e}"),
    }
}

#[cfg(test)]
mod finetune_cli_tests {
    use super::*;

    fn argv(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    /// The three required arguments, and nothing else: a plain finetune is a
    /// FULL one (same as `brain qwen3 finetune` / `brain glm finetune` with no
    /// LoRA rank), and every `FitOpts` field the caller did not name keeps the
    /// shared `FitOpts::default()` value rather than a qwen35-private one.
    #[test]
    fn a_bare_finetune_is_a_full_finetune_on_the_shared_fitopts_defaults() {
        let ft = parse_finetune(&argv(&["corpus", "--base", "base.safetensors", "--out", "tuned.safetensors"])).unwrap();
        assert_eq!(ft.data_dir, "corpus");
        assert_eq!(ft.base, "base.safetensors");
        assert_eq!(ft.out, "tuned.safetensors");
        assert!(matches!(ft.mode, qwen35::finetune::Mode::FullOffload), "{:?}", ft.mode);
        let d = model::FitOpts::default();
        assert_eq!(ft.opts.steps, d.steps);
        assert_eq!(ft.opts.batch_size, d.batch_size);
        assert_eq!(ft.opts.block_size, d.block_size);
        assert_eq!(ft.opts.lr, d.lr);
        assert_eq!(ft.opts.grad_accum, d.grad_accum);
        assert_eq!(ft.opts.warmup, d.warmup);
        assert_eq!(ft.opts.weight_decay, d.weight_decay);
        assert_eq!(ft.opts.grad_clip, d.grad_clip);
        assert_eq!(ft.opts.checkpoint_secs, d.checkpoint_secs);
        assert_eq!(ft.opts.seed, d.seed);
        assert_eq!(ft.opts.mask_before, None);
    }

    /// The cosine schedule has to be defined against the run the caller asked
    /// for. A short finetune left at `FitOpts::default()`'s `decay_iters` would
    /// never leave the flat top of the schedule - the LR flag would silently
    /// mean something other than it says.
    #[test]
    fn the_lr_schedule_follows_the_requested_run_length_not_the_struct_default() {
        let ft = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--steps", "50", "--lr", "1e-3"])).unwrap();
        assert_eq!(ft.opts.steps, 50);
        assert_eq!(ft.opts.decay_iters, 50);
        assert_eq!(ft.opts.lr, 1e-3);
        assert!((ft.opts.min_lr - 1e-4).abs() < 1e-9, "min_lr {}", ft.opts.min_lr);
    }

    /// `--mode lora` must be usable on its own: a caller who has not yet
    /// chosen a rank gets the family's documented one, and `alpha = 2 * rank`.
    #[test]
    fn lora_mode_defaults_its_rank_and_alpha() {
        let ft = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--mode", "lora"])).unwrap();
        let qwen35::finetune::Mode::Lora { rank, alpha } = ft.mode else { panic!("--mode lora must select a LoRA mode") };
        assert_eq!(rank, DEFAULT_LORA_RANK);
        assert_eq!(alpha, DEFAULT_LORA_RANK as f32 * 2.0);

        let ft = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--mode", "lora", "--rank", "4"])).unwrap();
        let qwen35::finetune::Mode::Lora { rank, alpha } = ft.mode else { panic!("expected LoRA") };
        assert_eq!((rank, alpha), (4, 8.0), "alpha must follow an explicit --rank, not stay at the default rank's");

        let ft = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--mode", "lora", "--rank", "4", "--alpha", "1.5"])).unwrap();
        let qwen35::finetune::Mode::Lora { rank, alpha } = ft.mode else { panic!("expected LoRA") };
        assert_eq!((rank, alpha), (4, 1.5));
    }

    /// Every flag the caller passes has to reach `FitOpts` - a silently
    /// ignored `--grad-accum` or `--weight-decay` is a run that trained
    /// something other than what was asked for.
    #[test]
    fn every_training_flag_reaches_fitopts() {
        let ft = parse_finetune(&argv(&[
            "corpus", "--base", "b", "--out", "o", "--batch", "2", "--block", "16", "--grad-accum", "4", "--warmup", "7",
            "--weight-decay", "0.05", "--grad-clip", "0.5", "--save-secs", "0", "--seed", "99", "--mask", "=", "--align",
        ]))
        .unwrap();
        assert_eq!(ft.opts.batch_size, 2);
        assert_eq!(ft.opts.block_size, 16);
        assert_eq!(ft.opts.grad_accum, 4);
        assert_eq!(ft.opts.warmup, 7);
        assert_eq!(ft.opts.weight_decay, 0.05);
        assert_eq!(ft.opts.grad_clip, 0.5);
        assert_eq!(ft.opts.checkpoint_secs, 0);
        assert_eq!(ft.opts.seed, 99);
        assert_eq!(ft.opts.mask_before, Some('='));
        assert!(ft.opts.mask_per_line);
        assert!(ft.opts.align_to_lines);
    }

    /// The corpus is a positional, so it must be recognised wherever it sits -
    /// including AFTER a value flag, where a naive "first non-`--` token" scan
    /// would pick up that flag's own value instead of the directory.
    #[test]
    fn the_corpus_positional_is_not_confused_with_a_flags_value() {
        let ft = parse_finetune(&argv(&["--base", "base.safetensors", "--out", "o", "corpus"])).unwrap();
        assert_eq!(ft.data_dir, "corpus");
        assert_eq!(ft.base, "base.safetensors");
    }

    /// A missing required argument is a usage message, not a panic and not a
    /// run that writes a checkpoint somewhere unintended.
    #[test]
    fn the_required_arguments_are_reported_by_name() {
        assert!(parse_finetune(&argv(&["--base", "b", "--out", "o"])).unwrap_err().contains("corpus directory"));
        assert!(parse_finetune(&argv(&["corpus", "--out", "o"])).unwrap_err().contains("--base"));
        assert!(parse_finetune(&argv(&["corpus", "--base", "b"])).unwrap_err().contains("--out"));
        let err = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--mode", "qlora"])).unwrap_err();
        assert!(err.contains("lora") && err.contains("full"), "{err}");
    }
}
