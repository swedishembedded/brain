// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! `brain qwen35 ...` - run the Qwen3.8-27B dense hybrid decoder.
//!
//!   brain qwen35 infer  [--weights F] [--tokenizer tokenizer.json | --gguf G] --prompt "..."
//!                     [--adapter adapter.safetensors] [--max-new N --temp X --top-k K --chat]
//!   brain qwen35 finetune <data_dir> --base F --out F [--mode lora|full]
//!                     [--lora-rank R --lora-alpha A] [--base-format f32|bf16|nf4|fp8]
//!                     [--seq-len T] [--recompute on|off] [--steps N --lr X ...]
//!   brain qwen35 sample-adapter --base DIR --prompt TEXT [--prompt TEXT ...] [--adapter FILE]
//!                     [--base-format nf4|bf16|fp8|f32] [--max-new N] [--no-adapter | --compare]
//!
//! `infer`'s `--weights`/`--tokenizer` are optional: naming both explicitly
//! (or `--weights` plus `--gguf`) uses exactly those files, unchanged; naming
//! less than that resolves the rest through the model-store scan
//! (`crate::resolver_cli`, `qwen35::spec::Qwen35Spec`) instead of erroring -
//! printing every real candidate and exiting on an ambiguous or missing
//! outcome.
//!
//! GGUF import lives in the GENERIC `brain import-gguf` command
//! ([`serving::gguf_import`]), which dispatches on the file's own
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
use gpu_core::select::Dtype;
use qwen35::base_source::{is_frozen_only, TRAIN_BASES};
use qwen35::config::Qwen35Config;
use qwen35::model::{pipelines, Qwen35};

pub fn run_qwen35(args: &[String]) {
    match args.first().map(|s| crate::args::canon_verb(s)) {
        Some("import") => import(&args[1..]),
        Some("infer") => infer(&args[1..]),
        Some("finetune") => finetune(&args[1..]),
        Some("sample-adapter") => sample(&args[1..]),
        other => eprintln!("usage: brain qwen35 <import|infer|finetune|sample-adapter> ...  (got {other:?})"),
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
    crate::import_gguf_cli::run_import_gguf(args);
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
    // `--weights`/`--tokenizer` are the resolver's own role-override flags
    // (`crate::resolver_cli::extract_role_overrides`) - pulled out FIRST so
    // the loop below never sees them, matching `flux2_cli.rs`'s own shape.
    let spec = qwen35::spec::Qwen35Spec;
    let (mut overrides, args) = crate::resolver_cli::extract_role_overrides(&spec, args);
    let (seed, args) = crate::args::split_seed(&args, "qwen35 infer");
    let mut gguf_for_tok = String::new();
    let mut prompt = String::new();
    let mut adapter = String::new();
    let mut max_new = 32usize;
    let mut temp = 0.0f32;
    let mut top_k = 0usize;
    let mut top_p = 1.0f32;
    let mut chat = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--gguf" => gguf_for_tok = val(&args, &mut i, "--gguf"),
            "--prompt" => prompt = val(&args, &mut i, "--prompt"),
            "--adapter" => adapter = val(&args, &mut i, "--adapter"),
            "--max-new" => max_new = val(&args, &mut i, "--max-new").parse().unwrap_or(max_new),
            "--temp" => temp = val(&args, &mut i, "--temp").parse().unwrap_or(temp),
            "--top-k" => top_k = val(&args, &mut i, "--top-k").parse().unwrap_or(top_k),
            "--top-p" => top_p = val(&args, &mut i, "--top-p").parse().unwrap_or(top_p),
            "--chat" => chat = true,
            other => eprintln!("ignoring unknown flag {other:?}"),
        }
        i += 1;
    }

    // `--weights`/(`--tokenizer` or `--gguf`) named explicitly is the
    // existing, unchanged manual path (a caller who already knows exactly
    // which files to use - the resolver never second-guesses that). Anything
    // less than that resolves through the model-store scan instead of
    // hard-erroring: `resolve_or_exit` prints every real candidate and exits
    // on an `Ambiguous`/`Missing` outcome, the same as `flux2_cli.rs`'s own
    // `resolve_flux2`.
    let weights_named = overrides.contains_key("weights");
    let tokenizer_named = overrides.contains_key("tokenizer") || !gguf_for_tok.is_empty();
    let (weights, tokenizer) = if weights_named && tokenizer_named {
        (overrides.remove("weights").unwrap_or_default(), overrides.remove("tokenizer").unwrap_or_default())
    } else {
        let assembly = crate::resolver_cli::resolve_or_exit("qwen35", &spec, &overrides);
        let get = |role: &str| assembly.roles.get(role).map(|p| p.to_string_lossy().into_owned()).unwrap_or_default();
        (get("weights"), get("tokenizer"))
    };

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

    // `qwen35::spec::Qwen35Spec`'s `weights` role accepts a Q8_0 GGUF as well
    // as a brain-format checkpoint, but this command only ever calls
    // `checkpoint::load`, which reads safetensors alone and panics on
    // anything else - refuse cleanly rather than reach that call.
    if weights.ends_with(".gguf") {
        eprintln!("qwen35 infer: '{weights}' is a GGUF checkpoint; this command only loads brain-format .safetensors (name one explicitly with --weights)");
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

const SAMPLE_USAGE: &str = "usage: brain qwen35 sample-adapter --base F|DIR --prompt TEXT [--prompt TEXT ...] [--adapter FILE] \
     [--base-format nf4|bf16|fp8|f32] [--max-new N] [--tokenizer FILE] [--chat] [--no-adapter | --compare]
  --base        a HuggingFace checkpoint directory (FP8 release included) or a brain-native file; streamed onto
                the device at --base-format with the adapter overlaid, never folded into a host copy
  --adapter     a LoRA adapter written by `brain qwen35 finetune` (omit for the base alone)
  --no-adapter  decode with the adapter switched off (the base model); --compare prints both answers per prompt
  --tokenizer   defaults to tokenizer.json inside the --base directory
  Decoding is greedy, by repeated full forwards at a fixed length (no KV cache): slow, but exactly the forward
  the adapter was trained through.";

/// Everything `brain qwen35 sample-adapter` needs, parsed off the argv.
#[derive(Debug)]
struct SampleArgs {
    base: String,
    adapter: Option<String>,
    prompts: Vec<String>,
    base_dtype: Dtype,
    max_new: usize,
    tokenizer: Option<String>,
    chat: bool,
    /// Which decodes to print for each prompt.
    which: SampleWhich,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum SampleWhich {
    AdapterOnly,
    BaseOnly,
    Both,
}

fn parse_sample(args: &[String]) -> Result<SampleArgs, String> {
    let mut a = Args::new(args);
    let base = a.str_or("--base", "");
    let adapter = a.take_str("--adapter");
    let mut prompts = Vec::new();
    while let Some(p) = a.take_str("--prompt") {
        prompts.push(p);
    }
    let base_dtype = TRAIN_BASES.parse(&aliased(&mut a, "--base-format", "--base-dtype")?.unwrap_or_else(|| "nf4".into()))?;
    let max_new = parse_num("--max-new", a.take_str("--max-new"), 32usize)?;
    let tokenizer = a.take_str("--tokenizer");
    let chat = a.take_flag("--chat");
    let (no_adapter, compare) = (a.take_flag("--no-adapter"), a.take_flag("--compare"));
    if let Some(extra) = a.leftovers().first() {
        return Err(format!("unrecognised argument {extra:?}"));
    }
    if base.is_empty() {
        return Err("--base is required".into());
    }
    if prompts.is_empty() {
        return Err("at least one --prompt is required".into());
    }
    if max_new == 0 {
        return Err("--max-new must be at least 1".into());
    }
    let which = match (no_adapter, compare) {
        (true, true) => return Err("--no-adapter and --compare are exclusive".into()),
        (true, false) => SampleWhich::BaseOnly,
        (false, true) => SampleWhich::Both,
        (false, false) => SampleWhich::AdapterOnly,
    };
    if adapter.is_none() && which != SampleWhich::BaseOnly {
        return Err("--adapter is required unless --no-adapter names the base alone".into());
    }
    Ok(SampleArgs { base, adapter, prompts, base_dtype, max_new, tokenizer, chat, which })
}

/// `brain qwen35 sample-adapter`: greedy answers of a (quantised, streamed) base with a
/// trained LoRA adapter overlaid, via [`qwen35::adapter_sample::AdapterSampler`].
fn sample(args: &[String]) {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{SAMPLE_USAGE}");
        return;
    }
    let sa = match parse_sample(args) {
        Ok(sa) => sa,
        Err(e) => {
            eprintln!("brain qwen35 sample-adapter: {e}");
            eprintln!("{SAMPLE_USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run_sample(&sa) {
        eprintln!("brain qwen35 sample-adapter failed: {e}");
        std::process::exit(1);
    }
}

fn run_sample(sa: &SampleArgs) -> Result<(), String> {
    let base = Path::new(&sa.base);
    let tok_path = sa.tokenizer.clone().unwrap_or_else(|| base.join("tokenizer.json").to_string_lossy().into_owned());
    let tok = data::qwen_tokenizer::QwenBpe::from_file(&tok_path).map_err(|e| format!("tokenizer {tok_path}: {e}"))?;
    let encoded: Vec<Vec<u32>> = sa
        .prompts
        .iter()
        .map(|p| tok.encode(&if sa.chat { tok.apply_chat_template(&[("user", p)], true) } else { p.clone() }))
        .collect();
    let longest = encoded.iter().map(Vec::len).max().unwrap_or(0);
    if longest == 0 {
        return Err("a prompt encodes to no tokens".into());
    }
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|s| tok.encode(s).first().copied()).collect();
    let t0 = std::time::Instant::now();
    let adapter = sa.adapter.as_deref().map(Path::new);
    let mut sampler = qwen35::adapter_sample::AdapterSampler::open(base, adapter, sa.base_dtype, (longest + sa.max_new) as u32)?;
    eprintln!("qwen35 sample-adapter: base {} ({}) loaded in {:.1}s, forward length {}", sa.base, model::policy::format_name(sa.base_dtype), t0.elapsed().as_secs_f64(), longest + sa.max_new);
    let modes: &[(&str, bool)] = match sa.which {
        SampleWhich::AdapterOnly => &[("adapter", true)],
        SampleWhich::BaseOnly => &[("base", false)],
        SampleWhich::Both => &[("adapter", true), ("base", false)],
    };
    for (prompt, ids) in sa.prompts.iter().zip(&encoded) {
        for (label, on) in modes {
            sampler.set_adapter(*on);
            let t = std::time::Instant::now();
            let out = sampler.greedy(ids, sa.max_new, &eos)?;
            println!("[{label}] {prompt}{}", tok.decode(&out));
            eprintln!("qwen35 sample-adapter: {} new tokens in {:.1}s", out.len(), t.elapsed().as_secs_f64());
        }
    }
    Ok(())
}

/// LoRA rank used by `--mode lora` when `--rank` is omitted: 8, the rank this
/// model family's LoRA work is documented against (see AGENTS.md's
/// `qwen35moe` entry). `--alpha` then defaults to `2 * rank`, the convention
/// every other adapter-training verb in this crate already uses
/// (`qwen_cli::finetune_lora`, `forecast_cli::train`).
const DEFAULT_LORA_RANK: u32 = 8;

/// ` rs lr-ratio=X freeze-a` for the hyper-parameters a run turned on.
fn tuning_desc(t: &model::adapter::device::AdapterTuning) -> String {
    let mut out = String::new();
    if t.rank_stabilized {
        out.push_str(" rs");
    }
    if t.lr_ratio != 1.0 {
        out.push_str(&format!(" lr-ratio={}", t.lr_ratio));
    }
    if t.freeze_a {
        out.push_str(" freeze-a");
    }
    out
}

const FINETUNE_USAGE: &str = "usage: brain qwen35 finetune <data_dir> --base F|DIR --out F [--mode lora|full] [--lora-rank R --lora-alpha A --lora-variant V --lora-plus-ratio X --adapter-kind K] \
     [--base-format f32|bf16|nf4|fp8] [--seq-len T] [--recompute on|off] [--gdn-segment N] [--bf16-residuals on|off] [--mlp-rows N] \
     [--eval-interval N --eval-batches N] [--steps N --lr X --batch B --grad-accum G --warmup W --weight-decay X --grad-clip X --save-secs S --seed S --mask = --align] \
     [--train-precision SPEC] [--activations keep|recompute]
  --train-precision  how a --mode full run stores what it trains, one model-agnostic value: state=f32|bf16|int8[:B] (device AdamW
                moments), master=none|bf16|f32, grad=f32|bf16, sr=on|off, narrow=on|off (let the planner narrow the state), and for
                the linears resident in a low-precision encoding weights=bf16|fp8[:B]|int8|nf4 with window=N (layers kept on the
                device, the rest streamed) and clip=two-pass|running; a choice the model does not run is refused by name
  --base-format how the frozen base is held on the device: f32 (default, whole model resident; full or LoRA),
                bf16 (half the bytes; LoRA only) or nf4 (4-bit QLoRA; LoRA only). bf16/nf4 stream the base from
                a HuggingFace checkpoint directory (FP8 release included) tensor by tensor.
  --seq-len     training sequence length (alias of --block); --rank/--alpha are aliases of --lora-rank/--lora-alpha
  --adapter-kind  the adapter method: lora (default), ia3 (learned output-channel scales) or lokr (Kronecker-product delta)
  --lora-variant, --lora-plus-ratio  rsLoRA and LoRA-FA (--lora-variant rs,fa: scale alpha/sqrt(rank); A stays at its
                init) and LoRA+ (B trains at X times the learning rate); a continued adapter must be continued with the same
  --recompute   recompute layer activations in the backward pass instead of keeping them (default on for bf16/nf4)
  --activations what the backward keeps of the forward: keep every layer's activations or recompute them from each layer's input
                (same gradients, lower memory). With --mode lora it is --recompute off|on by another name (naming both with
                different values is refused); with --mode full it is the activations= key of --train-precision (naming both
                with different values is refused). Absent leaves each trainer's own default
  --gdn-segment, --bf16-residuals, --mlp-rows  the long-sequence settings (segmented Gated-DeltaNet layers, bf16 residual
                stream, MLP in token chunks) of a bf16/nf4 base. Rows of 2048 tokens or more at batch 1 get 2048 / on / 2048
                by themselves; naming one replaces that choice (0 turns --gdn-segment and --mlp-rows off)
  --adapter-id  the card id of the adapter a bf16/nf4 run writes, an adapter reference <vendor>/<repo>[-QUANT]:<owner>:<name>:<tag>
                (served under this id by `brain serve`). Default: <base release>:local:lora:latest, with the release id
                (<vendor>/<repo>) taken from the base directory's last two components; --name NAME sets the name part
  --eval-interval  evaluate on the held-out split every N steps (default 250, 0: never); --eval-batches batches per evaluation
  Every step prints its loss to stderr. --backend cuda|cpu|vulkan is the global backend flag and applies as for every other command";

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
    base_dtype: Dtype,
    /// Recompute activations in backward. Only a frozen-base tier has a
    /// recomputing trainer; the f32 resident one keeps every activation.
    recompute: bool,
    /// Explicit long-sequence settings of a streamed (bf16/nf4) base.
    long: qwen35::finetune::LongSeqOverrides,
    /// What the written adapter calls itself.
    naming: qwen35::finetune::AdapterNaming,
}

/// A flag that has a short alias: the value of whichever is given, an error
/// when both are (two values for one setting is a mistake, not a precedence rule).
fn aliased(a: &mut Args, flag: &str, alias: &str) -> Result<Option<String>, String> {
    match (a.take_str(flag), a.take_str(alias)) {
        (Some(_), Some(_)) => Err(format!("{flag} and {alias} name the same setting; give one")),
        (v, w) => Ok(v.or(w)),
    }
}

fn parse_num<T: std::str::FromStr>(flag: &str, v: Option<String>, default: T) -> Result<T, String> {
    v.map_or(Ok(default), |s| s.parse().map_err(|_| format!("{flag} {s:?}: not a valid number")))
}

fn parse_finetune(args: &[String]) -> Result<FinetuneArgs, String> {
    let mut a = Args::new(args);
    let seed = a.seed("qwen35 finetune");
    let base = a.str_or("--base", "");
    let out = a.str_or("--out", "");
    let mode_name = a.str_or("--mode", "full");
    let rank = parse_num("--lora-rank", aliased(&mut a, "--lora-rank", "--rank")?, DEFAULT_LORA_RANK)?;
    let alpha = parse_num("--lora-alpha", aliased(&mut a, "--lora-alpha", "--alpha")?, rank as f32 * 2.0)?;
    let tuning = model::adapter::device::AdapterTuning::parse(a.take_str("--lora-variant").as_deref(), a.take_str("--lora-plus-ratio").as_deref(), a.take_str("--adapter-kind").as_deref())?;
    let base_dtype = TRAIN_BASES.parse(&aliased(&mut a, "--base-format", "--base-dtype")?.unwrap_or_else(|| "f32".into()))?;
    let recompute_flag = match a.take_str("--recompute").as_deref() {
        None => None,
        Some("on") => Some(true),
        Some("off") => Some(false),
        Some(other) => return Err(format!("--recompute {other:?}: expected \"on\" or \"off\"")),
    };
    let long = qwen35::finetune::LongSeqOverrides {
        gdn_segment: a.take_str("--gdn-segment").map(|v| parse_num("--gdn-segment", Some(v), 0u32)).transpose()?,
        bf16_residuals: match a.take_str("--bf16-residuals").as_deref() {
            None => None,
            Some("on") => Some(true),
            Some("off") => Some(false),
            Some(other) => return Err(format!("--bf16-residuals {other:?}: expected \"on\" or \"off\"")),
        },
        mlp_rows: a.take_str("--mlp-rows").map(|v| parse_num("--mlp-rows", Some(v), 0u32)).transpose()?,
    };
    let naming = qwen35::finetune::AdapterNaming { id: a.take_str("--adapter-id"), name: a.take_str("--name") };
    let block_size = parse_num("--seq-len", aliased(&mut a, "--seq-len", "--block")?, model::FitOpts::default().block_size)?;

    let activations = a.take_str(crate::args::ACTIVATIONS_FLAG).map(|value| crate::args::parse_activations(&value)).transpose()?;
    let precision = match a.take_str(crate::args::TRAIN_PRECISION_FLAG) {
        Some(spec) => model::TrainPrecision::parse(&spec).map_err(|e| format!("{} {spec:?}: {e}", crate::args::TRAIN_PRECISION_FLAG))?,
        None => model::TrainPrecision::default(),
    };
    let d = model::FitOpts::default();
    let steps = a.u32_or("--steps", d.steps);
    let lr = a.f32_or("--lr", d.lr);
    let mask = a.char_opt("--mask");
    let mut opts = model::FitOpts {
        precision,
        steps,
        batch_size: a.u32_or("--batch", d.batch_size),
        block_size,
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
        eval_interval: a.u32_or("--eval-interval", d.eval_interval),
        eval_batches: a.u32_or("--eval-batches", d.eval_batches),
        checkpoint_secs: a.u64_or("--save-secs", d.checkpoint_secs),
        seed,
        mask_before: mask,
        mask_per_line: mask.is_some(),
        align_to_lines: a.take_flag("--align"),
        patience: 0,
        ..d
    };
    // The corpus is positional, matching `brain glm finetune <data_dir> ...`
    // and `brain flux2 finetune <data_dir> ...`. Taken LAST, after every
    // value flag has claimed its own value: `Args::positional` returns the
    // first token not yet marked used, so asking for it first would happily
    // return `--base`'s argument out of `finetune --base b.safetensors corpus`.
    let data_dir = a.positional().unwrap_or_default();
    a.finish();
    if naming.id.is_some() && naming.name.is_some() {
        return Err("--adapter-id and --name name the same thing; give one".into());
    }

    // `full` is the default mode because that is what the sibling verbs mean
    // by a plain finetune: `brain qwen3 finetune` and `brain glm finetune`
    // are both full-parameter unless a LoRA rank is asked for explicitly.
    if tuning.kind == model::adapter::DeviceKind::Dora && is_frozen_only(base_dtype) {
        return Err(format!("--adapter-kind dora reads the base weight, so it needs --base-format f32 (a {} base is held narrow)", model::policy::format_name(base_dtype)));
    }
    let mode = match mode_name.as_str() {
        "full" => qwen35::finetune::Mode::FullOffload,
        "lora" => qwen35::finetune::Mode::Lora { rank, alpha, tuning },
        other => return Err(format!("--mode {other:?}: expected \"lora\" or \"full\"")),
    };
    if is_frozen_only(base_dtype) && !matches!(mode, qwen35::finetune::Mode::Lora { .. }) {
        return Err(format!(
            "--base-format {} holds the base frozen, so only --mode lora can train on it; full-parameter fine-tuning needs --base-format f32",
            model::policy::format_name(base_dtype)
        ));
    }
    // `--activations` is the full run's `activations=` precision key, and the LoRA run's `--recompute`.
    let mut recompute_flag = recompute_flag;
    if let Some(policy) = activations {
        match mode {
            qwen35::finetune::Mode::FullOffload => crate::args::merge_activations(Some(policy), &mut opts.precision)?,
            qwen35::finetune::Mode::Lora { .. } => {
                let wanted = policy == model::saved::SavedPolicy::Recompute;
                if let Some(named) = recompute_flag.filter(|named| *named != wanted) {
                    let word = |on: bool| if on { "on" } else { "off" };
                    return Err(format!("{} {policy:?} conflicts with --recompute {}; name the choice once", crate::args::ACTIVATIONS_FLAG, word(named)));
                }
                recompute_flag = Some(wanted);
            }
        }
    }
    if !opts.precision.is_default() && !matches!(mode, qwen35::finetune::Mode::FullOffload) {
        return Err(format!("{} describes how a full fine-tune stores what it trains: use it with --mode full", crate::args::TRAIN_PRECISION_FLAG));
    }
    if recompute_flag == Some(true) && !is_frozen_only(base_dtype) {
        let flag = if activations.is_some() { "--activations recompute" } else { "--recompute on" };
        return Err(format!("{flag}: the f32 resident trainer keeps every layer's activations; recompute is available with a narrow --base-format (bf16, nf4 or fp8)"));
    }
    let recompute = recompute_flag.unwrap_or(is_frozen_only(base_dtype));
    if long != qwen35::finetune::LongSeqOverrides::default() && !is_frozen_only(base_dtype) {
        return Err("--gdn-segment, --bf16-residuals and --mlp-rows apply to the streamed trainer: use a narrow --base-format (bf16, nf4 or fp8)".into());
    }
    if data_dir.is_empty() {
        return Err("the finetune corpus directory is required (a positional argument)".into());
    }
    if base.is_empty() {
        return Err("--base is required (a brain-native checkpoint, e.g. from `brain import-gguf`)".into());
    }
    if out.is_empty() {
        return Err("--out is required (where to write the finetuned checkpoint)".into());
    }
    Ok(FinetuneArgs { base, data_dir, out, opts, mode, base_dtype, recompute, long, naming })
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
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{FINETUNE_USAGE}");
        return;
    }
    let ft = match parse_finetune(args) {
        Ok(ft) => ft,
        Err(e) => {
            eprintln!("brain qwen35 finetune: {e}");
            eprintln!("{FINETUNE_USAGE}");
            std::process::exit(2);
        }
    };
    let mode_desc = match &ft.mode {
        qwen35::finetune::Mode::FullOffload => "full (offloaded AdamW moments)".to_string(),
        qwen35::finetune::Mode::Lora { rank, alpha, tuning } => format!("lora rank={rank} alpha={alpha}{}", tuning_desc(tuning)),
    };
    eprintln!(
        "qwen35 finetune: {} on {} [{mode_desc}, base {}{}] steps={} batch={} block={} lr={} grad-accum={} seed={} -> {}",
        ft.base,
        ft.data_dir,
        model::policy::format_name(ft.base_dtype),
        if ft.recompute { ", recompute" } else { "" },
        ft.opts.steps,
        ft.opts.batch_size,
        ft.opts.block_size,
        ft.opts.lr,
        ft.opts.grad_accum,
        ft.opts.seed,
        ft.out
    );
    let t0 = std::time::Instant::now();
    match run_finetune(&ft) {
        Ok((initial_loss, final_loss)) => eprintln!(
            "qwen35 finetune: loss {initial_loss:.4} -> {final_loss:.4} in {:.1}s; saved {}",
            t0.elapsed().as_secs_f64(),
            ft.out
        ),
        Err(e) => {
            eprintln!("qwen35 finetune failed: {e}");
            std::process::exit(1);
        }
    }
}

/// Train as `ft` says: the classic resident trainer for an f32 base file, the
/// streamed-base trainer for a bf16/nf4 base or a HuggingFace directory.
/// Returns `(initial_loss, final_loss)`.
fn run_finetune(ft: &FinetuneArgs) -> std::io::Result<(f32, f32)> {
    let (base, data) = (Path::new(&ft.base), Path::new(&ft.data_dir));
    if ft.base_dtype == Dtype::F32 && !base.is_dir() {
        qwen35::finetune::finetune(&ft.base, data, &ft.opts, &ft.mode, &ft.out)
    } else {
        qwen35::finetune::finetune_streamed_named(base, data, &ft.opts, &ft.mode, &ft.out, false, &qwen35::finetune::StreamedRun { dtype: ft.base_dtype, recompute: ft.recompute, long: ft.long }, &ft.naming)
    }
}

#[cfg(test)]
mod finetune_cli_tests {
    use super::*;

    fn argv(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    /// Without `--seed` every finetune draws its own seed: two parses of the
    /// same argv are two independent runs, not one repeated default.
    #[test]
    fn an_unseeded_finetune_draws_a_fresh_seed_each_time() {
        let argv = argv(&["corpus", "--base", "b.safetensors", "--out", "t.safetensors"]);
        let first = parse_finetune(&argv).unwrap().opts.seed;
        let second = parse_finetune(&argv).unwrap().opts.seed;
        assert_ne!(first, second);
    }

    #[test]
    fn the_adapter_is_named_by_an_id_or_by_a_name_but_not_both() {
        let ft = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--adapter-id", "Qwen/Qwen3.8-27B-FP8:lab:facts:v1"])).unwrap();
        assert_eq!(ft.naming.id.as_deref(), Some("Qwen/Qwen3.8-27B-FP8:lab:facts:v1"));
        let ft = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--name", "facts"])).unwrap();
        assert_eq!(ft.naming.name.as_deref(), Some("facts"));
        let err = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--name", "facts", "--adapter-id", "a/b:o:n:t"])).unwrap_err();
        assert!(err.contains("--adapter-id") && err.contains("--name"), "{err}");
    }

    /// The three required arguments, and nothing else: a plain finetune is a
    /// FULL one (same as `brain qwen3 finetune` / `brain glm finetune` with no
    /// LoRA rank), and every `FitOpts` field the caller did not name keeps the
    /// shared `FitOpts::default()` value rather than a qwen35-private one.
    #[test]
    fn a_bare_finetune_is_a_full_finetune_on_the_shared_fitopts_defaults() {
        // `--seed` is pinned explicitly to the struct default's OWN value:
        // everything else here checks that an omitted flag falls back to
        // `FitOpts::default()`, but `--seed` is the one flag whose omission is
        // deliberately NOT the struct default any more (it randomizes) - see
        // `data::rng::random_seed`. Pinning it keeps this test's premise
        // (every field ends up at `d`'s value) true without hardcoding `d.seed`
        // twice.
        let d = model::FitOpts::default();
        let seed_str = d.seed.to_string();
        let ft = parse_finetune(&argv(&["corpus", "--base", "base.safetensors", "--out", "tuned.safetensors", "--seed", &seed_str])).unwrap();
        assert_eq!(ft.data_dir, "corpus");
        assert_eq!(ft.base, "base.safetensors");
        assert_eq!(ft.out, "tuned.safetensors");
        assert!(matches!(ft.mode, qwen35::finetune::Mode::FullOffload), "{:?}", ft.mode);
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

    /// Spec: `--train-precision` is the one model-agnostic flag for how a full run stores what it trains; it parses into
    /// the typed value the generic fit loop reads, names the key it cannot read, and is a full-fine-tune option.
    #[test]
    fn a_train_precision_is_one_typed_value_for_a_full_finetune_only() {
        let base = ["corpus", "--base", "b", "--out", "o"];
        let with = |extra: &[&str]| parse_finetune(&argv(&[&base[..], extra].concat()));
        let ft = with(&["--train-precision", "state=int8,weights=fp8,window=2,clip=running"]).unwrap();
        let r = ft.opts.precision.resident.expect("resident request");
        assert_eq!((r.window, r.clip_pass), (Some(2), model::resident_ft::ClipPass::Running));
        assert!(with(&[]).unwrap().opts.precision.is_default());
        let e = with(&["--train-precision", "state=fp4"]).unwrap_err();
        assert!(e.contains("--train-precision") && e.contains("state"), "{e}");
        let e = with(&["--mode", "lora", "--train-precision", "state=bf16"]).unwrap_err();
        assert!(e.contains("--mode full"), "{e}");
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
        let qwen35::finetune::Mode::Lora { rank, alpha, .. } = ft.mode else { panic!("--mode lora must select a LoRA mode") };
        assert_eq!(rank, DEFAULT_LORA_RANK);
        assert_eq!(alpha, DEFAULT_LORA_RANK as f32 * 2.0);

        let ft = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--mode", "lora", "--rank", "4"])).unwrap();
        let qwen35::finetune::Mode::Lora { rank, alpha, .. } = ft.mode else { panic!("expected LoRA") };
        assert_eq!((rank, alpha), (4, 8.0), "alpha must follow an explicit --rank, not stay at the default rank's");

        let ft = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--mode", "lora", "--rank", "4", "--alpha", "1.5"])).unwrap();
        let qwen35::finetune::Mode::Lora { rank, alpha, .. } = ft.mode else { panic!("expected LoRA") };
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

    /// rsLoRA, LoRA+ and LoRA-FA reach the trainer through the LoRA mode by the
    /// model-agnostic options, and a value that cannot train is refused.
    #[test]
    fn the_adapter_variant_options_are_flags_of_the_lora_mode() {
        let base = ["corpus", "--base", "b", "--out", "o", "--mode", "lora"];
        let with = |extra: &[&str]| parse_finetune(&argv(&[&base[..], extra].concat()));
        let plain = with(&[]).unwrap();
        assert!(matches!(plain.mode, qwen35::finetune::Mode::Lora { tuning, .. } if tuning == model::adapter::device::AdapterTuning::default()));
        let tuned = with(&["--lora-variant", "rs,fa", "--lora-plus-ratio", "16"]).unwrap();
        let want = model::adapter::device::AdapterTuning { rank_stabilized: true, lr_ratio: 16.0, freeze_a: true, ..Default::default() };
        assert!(matches!(tuned.mode, qwen35::finetune::Mode::Lora { tuning, .. } if tuning == want), "{:?}", tuned.mode);
        assert!(with(&["--lora-plus-ratio", "0"]).unwrap_err().contains("LoRA+"));
        assert!(with(&["--lora-variant", "dora"]).unwrap_err().contains("--lora-variant"));
        let lokr = with(&["--adapter-kind", "lokr"]).unwrap();
        assert!(matches!(lokr.mode, qwen35::finetune::Mode::Lora { tuning, .. } if tuning.kind == model::adapter::DeviceKind::Lokr));
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

    #[test]
    fn the_base_precision_flags_parse_with_their_aliases() {
        let ft = parse_finetune(&argv(&[
            "corpus", "--base", "b", "--out", "o", "--mode", "lora", "--base-dtype", "nf4", "--lora-rank", "16", "--lora-alpha", "8", "--seq-len", "512",
        ]))
        .unwrap();
        assert_eq!(ft.base_dtype, Dtype::NF4);
        assert!(matches!(ft.mode, qwen35::finetune::Mode::Lora { rank: 16, alpha, .. } if alpha == 8.0), "{:?}", ft.mode);
        assert_eq!(ft.opts.block_size, 512, "--seq-len is the training sequence length");
        assert!(ft.recompute, "a frozen-base tier recomputes by default");

        let old = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--mode", "lora", "--rank", "4", "--block", "64", "--base-dtype", "bf16", "--recompute", "off"])).unwrap();
        assert_eq!((old.base_dtype, old.opts.block_size, old.recompute), (Dtype::BF16, 64, false));

        let plain = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o"])).unwrap();
        assert_eq!((plain.base_dtype, plain.recompute), (Dtype::F32, false), "the default keeps the classic f32 trainer");
    }

    /// Spec: `--activations` is the full run's `activations=` precision key and the LoRA run's `--recompute`; naming
    /// both spellings with different values is refused by name, and a bogus value is refused naming the flag.
    #[test]
    fn activations_route_to_the_precision_key_or_the_recompute_switch() {
        let full = ["corpus", "--base", "b", "--out", "o"];
        let with = |base: &[&str], extra: &[&str]| parse_finetune(&argv(&[base, extra].concat()));
        assert_eq!(with(&full, &["--activations", "recompute"]).unwrap().opts.precision.activations, Some(model::saved::SavedPolicy::Recompute));
        assert_eq!(with(&full, &[]).unwrap().opts.precision.activations, None, "absent leaves the trainer's own default");
        let e = with(&full, &["--activations", "keep", "--train-precision", "activations=recompute"]).unwrap_err();
        assert!(e.contains("--activations") && e.contains("--train-precision"), "{e}");
        assert!(with(&full, &["--activations", "bogus"]).unwrap_err().contains("--activations"));

        let lora = ["corpus", "--base", "b", "--out", "o", "--mode", "lora", "--base-dtype", "bf16"];
        assert!(with(&lora, &["--activations", "recompute"]).unwrap().recompute);
        assert!(!with(&lora, &["--activations", "keep"]).unwrap().recompute);
        let e = with(&lora, &["--activations", "keep", "--recompute", "on"]).unwrap_err();
        assert!(e.contains("--activations") && e.contains("--recompute"), "{e}");
        let f32_lora = ["corpus", "--base", "b", "--out", "o", "--mode", "lora"];
        let e = with(&f32_lora, &["--activations", "recompute"]).unwrap_err();
        assert!(e.contains("--activations recompute") && e.contains("f32"), "{e}");
    }

    #[test]
    fn unsupported_tier_combinations_are_refused_with_the_reason() {
        let base = ["corpus", "--base", "b", "--out", "o"];
        let with = |extra: &[&str]| parse_finetune(&argv(&[&base[..], extra].concat())).unwrap_err();
        let err = with(&["--base-dtype", "nf4"]);
        assert!(err.contains("frozen") && err.contains("--mode lora") && err.contains("f32"), "{err}");
        assert!(with(&["--base-dtype", "bf16", "--mode", "full"]).contains("frozen"));
        assert!(with(&["--base-dtype", "int8", "--mode", "lora"]).contains("f32, bf16, nf4, fp8"));
        assert!(with(&["--recompute", "on"]).contains("narrow --base-format"));
        assert!(with(&["--recompute", "maybe"]).contains("\"on\" or \"off\""));
        assert!(with(&["--mode", "lora", "--rank", "4", "--lora-rank", "4"]).contains("same setting"));
        assert!(with(&["--seq-len", "64", "--block", "64"]).contains("same setting"));
        assert!(with(&["--seq-len", "long"]).contains("not a valid number"));
    }

    #[test]
    fn the_long_sequence_and_eval_flags_reach_the_trainer() {
        let base = ["corpus", "--base", "b", "--out", "o", "--mode", "lora", "--base-dtype", "nf4"];
        let with = |extra: &[&str]| parse_finetune(&argv(&[&base[..], extra].concat()));
        let plain = with(&[]).unwrap();
        assert_eq!((plain.long, plain.opts.eval_interval), (qwen35::finetune::LongSeqOverrides::default(), model::FitOpts::default().eval_interval));
        let ft = with(&["--gdn-segment", "256", "--bf16-residuals", "off", "--mlp-rows", "0", "--eval-interval", "5", "--eval-batches", "3"]).unwrap();
        assert_eq!(ft.long, qwen35::finetune::LongSeqOverrides { gdn_segment: Some(256), bf16_residuals: Some(false), mlp_rows: Some(0) });
        assert_eq!((ft.opts.eval_interval, ft.opts.eval_batches), (5, 3));
        assert!(with(&["--bf16-residuals", "maybe"]).unwrap_err().contains("\"on\" or \"off\""));
        assert!(with(&["--mlp-rows", "many"]).unwrap_err().contains("not a valid number"));
        let resident = parse_finetune(&argv(&["corpus", "--base", "b", "--out", "o", "--gdn-segment", "128"])).unwrap_err();
        assert!(resident.contains("narrow --base-format"), "{resident}");
    }

    #[test]
    fn the_sample_flags_parse_and_refuse_contradictions() {
        let ok = parse_sample(&argv(&["--base", "dir", "--adapter", "a.safetensors", "--prompt", "one", "--prompt", "two", "--max-new", "9", "--compare"])).unwrap();
        assert_eq!((ok.prompts.len(), ok.max_new, ok.which, ok.base_dtype), (2, 9, SampleWhich::Both, Dtype::NF4));
        let base_only = parse_sample(&argv(&["--base", "dir", "--prompt", "q", "--no-adapter"])).unwrap();
        assert_eq!((base_only.which, base_only.adapter), (SampleWhich::BaseOnly, None));
        let with = |extra: &[&str]| parse_sample(&argv(&[&["--base", "dir", "--prompt", "q"][..], extra].concat())).unwrap_err();
        assert!(with(&[]).contains("--adapter"), "an adapter-on run needs an adapter");
        assert!(with(&["--adapter", "a", "--no-adapter", "--compare"]).contains("exclusive"));
        assert!(with(&["--adapter", "a", "--max-new", "0"]).contains("at least 1"));
        assert!(with(&["--adapter", "a", "--typo"]).contains("--typo"));
        assert!(parse_sample(&argv(&["--prompt", "q", "--no-adapter"])).unwrap_err().contains("--base"));
        assert!(parse_sample(&argv(&["--base", "dir", "--no-adapter"])).unwrap_err().contains("--prompt"));
    }

    /// End to end through the CLI's own entry (flags -> trainer), tiny
    /// synthetic checkpoint and corpus: the f32 LoRA path trains, the loss is
    /// finite and the adapter file is written.
    #[test]
    fn a_tiny_f32_lora_finetune_runs_through_the_cli_path() {
        use data::binio::{self, Meta};
        let dir = std::env::temp_dir().join(format!("brain-qwen35-cli-smoke-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = Qwen35Config::tiny();
        // seed: any weights train; this checks the plumbing, not a loss value
        let init = qwen35::init::init_weights(&cfg, 7);
        let base = dir.join("base.safetensors");
        Qwen35::new_on(gpu_core::Gpu::new(pipelines()), cfg.clone(), 1, cfg.block_size, &init).save(base.to_str().unwrap());
        let data = dir.join("data");
        std::fs::create_dir_all(&data).unwrap();
        let tokens: Vec<u32> = (0..400).map(|i| [2, 5, 8, 15][i % 4]).collect();
        binio::write_u32_bin(&data.join("train.u32.bin"), &tokens).unwrap();
        binio::write_u32_bin(&data.join("val.u32.bin"), &[]).unwrap();
        std::fs::write(data.join("meta.json"), Meta::vocab_only(cfg.vocab as usize)).unwrap();
        let out = dir.join("adapter.safetensors");

        // Every adapter method trains through the same entry and writes a file
        // whose card names it.
        for kind in ["lora", "ia3", "lokr", "dora"] {
            let _ = std::fs::remove_file(&out);
            let ft = parse_finetune(&argv(&[
                data.to_str().unwrap(), "--base", base.to_str().unwrap(), "--out", out.to_str().unwrap(), "--mode", "lora", "--lora-rank", "2", "--adapter-kind", kind,
                "--steps", "3", "--batch", "1", "--seq-len", &cfg.block_size.to_string(), "--save-secs", "0", "--seed", "5",
            ]))
            .unwrap();
            let (initial, last) = run_finetune(&ft).expect("tiny finetune");
            assert!(initial.is_finite() && last.is_finite(), "{kind}: {initial} -> {last}");
            assert!(out.exists(), "the trained checkpoint must be written to --out");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The same run on a bf16 base, through the same entry.
    #[test]
    fn a_tiny_bf16_lora_finetune_runs_through_the_cli_path() {
        let dir = std::env::temp_dir().join(format!("brain-qwen35-cli-smoke-bf16-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = Qwen35Config::tiny();
        // seed: any weights train; this checks the plumbing, not a loss value
        let init = qwen35::init::init_weights(&cfg, 7);
        let base = dir.join("base.safetensors");
        Qwen35::new_on(gpu_core::Gpu::new(pipelines()), cfg.clone(), 1, cfg.block_size, &init).save(base.to_str().unwrap());
        let data = dir.join("data");
        std::fs::create_dir_all(&data).unwrap();
        let tokens: Vec<u32> = (0..400).map(|i| [2, 5, 8, 15][i % 4]).collect();
        data::binio::write_u32_bin(&data.join("train.u32.bin"), &tokens).unwrap();
        data::binio::write_u32_bin(&data.join("val.u32.bin"), &[]).unwrap();
        std::fs::write(data.join("meta.json"), data::binio::Meta::vocab_only(cfg.vocab as usize)).unwrap();
        let out = dir.join("adapter.safetensors");
        let ft = parse_finetune(&argv(&[
            data.to_str().unwrap(), "--base", base.to_str().unwrap(), "--out", out.to_str().unwrap(), "--mode", "lora", "--base-dtype", "bf16",
            "--lora-rank", "2", "--steps", "3", "--seq-len", &cfg.block_size.to_string(), "--seed", "5",
        ]))
        .unwrap();
        let (initial, last) = run_finetune(&ft).expect("tiny bf16 finetune");
        assert!(initial.is_finite() && last.is_finite());
        assert!(out.exists());
        std::fs::remove_dir_all(&dir).ok();
    }
}
