// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! A tiny Ouro looped decoder trained from random weights through the SDK, end to end:
//!
//! 1. [`brain::ModelInit`] learns a byte-level BPE vocabulary on a text, declares the chat and tool-call tokens and a
//!    chat template, and writes a randomly initialised checkpoint of the chosen configuration;
//! 2. [`brain::Pretrain`] trains every weight on the text (windows of the document stream, every position
//!    supervised), scoring a held-out tenth of it as it goes, and the sample reports the training rate in tokens a
//!    second;
//! 3. the trained checkpoint continues a prompt greedily through the model's own decode step.
//!
//! ```text
//! sample-models-ouro --out runs/ouro [--text FILE] [--size tiny|small|default] [--steps N] [--seed S] [--device cpu] [--eval-every N]
//! ```
//!
//! Swedish Embedded AB implements small language models trained from scratch on a client's own text. If your team
//! needs a model of its own, end to end, you can procure our services by sending an email to info@swedishembedded.com.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use brain::{Device, ModelInit, Pretrain};

const USAGE: &str = "\
usage: sample-models-ouro --out DIR [--text FILE] [--size tiny|small|default] [--steps N]
                          [--seed S] [--device SPEC] [--tokens N]

  --out DIR       where the model, the run and the corpus go (one directory per run)
  --text FILE     the training text (plain UTF-8; default: a short built-in text)
  --size NAME     tiny (about 0.15M parameters, seconds on a CPU), small (about 1.3M:
                   hidden 128, 4 layers, 4 passes, vocab 2048, rows of 512) or default
                   (5.3M: hidden 256, 4 layers, 4 heads, 4 passes, vocab 4096, rows of 1024)
  --steps N       pretraining steps (default 200 tiny, 100 default)
  --seed S        the initialization and window seed (drawn and printed when not given)
  --device SPEC   cpu, gpu0, ... (default cpu)
  --tokens N      tokens to continue the prompt with (default 48)
  --eval-every N  score the held-out text every N steps (default a quarter of the run; 0: never)
  --block N       tokens per training row (default: the size's)
  --batch N       rows per step (default: the size's)
";

/// The chat template the model is made with: ChatML-like turns, tool schemas in the system turn, calls as
/// `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
const TEMPLATE: &str = r#"{%- if tools %}<|im_start|>system
{% if messages[0].role == 'system' %}{{ messages[0].content }}
{% endif %}# Tools
{% for t in tools %}{{ t | tojson }}
{% endfor %}<|im_end|>
{% elif messages[0].role == 'system' %}<|im_start|>system
{{ messages[0].content }}<|im_end|>
{% endif %}
{%- for m in messages %}
{%- if m.role == 'system' %}
{%- elif m.role == 'assistant' %}<|im_start|>assistant
{{ m.content }}{% for c in m.tool_calls %}<tool_call>{"name": "{{ c.function.name }}", "arguments": {{ c.function.arguments | tojson }}}</tool_call>{% endfor %}<|im_end|>
{% else %}<|im_start|>{{ m.role }}
{{ m.content }}<|im_end|>
{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#;

const SPECIALS: [&str; 5] = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<tool_call>", "</tool_call>"];

/// The text trained on when none is given: plain sentences, enough distinct words for a few hundred merges.
const BUILT_IN: &str = "The river runs past the old mill and under the stone bridge. \
In the morning the baker opens the shop and the street smells of bread. \
A small boat waits at the pier while the fisherman mends his net. \
Children walk to school along the canal, counting the ducks on the water. \
At noon the market fills with voices, apples, cheese and bright flowers. \
In the evening the lamps are lit one by one along the quiet harbour. \
The library keeps maps of every street, and the clock in the square strikes nine. \
Seven boxes stand in the storeroom; the third holds letters, the fifth holds keys.";

/// One model size.
struct Size {
    config: serde_json::Value,
    vocab: usize,
    block: u32,
    batch: u32,
    steps: u32,
}

fn size(name: &str) -> Option<Size> {
    let config = |hidden: u32, layers: u32, heads: u32, ff: u32, passes: u32, positions: u32| {
        serde_json::json!({"architectures": ["OuroForCausalLM"], "model_type": "ouro", "hidden_size": hidden, "num_hidden_layers": layers,
            "num_attention_heads": heads, "num_key_value_heads": heads, "head_dim": hidden / heads, "intermediate_size": ff,
            "max_position_embeddings": positions, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "total_ut_steps": passes, "tie_word_embeddings": false})
    };
    match name {
        "tiny" => Some(Size { config: config(64, 2, 2, 176, 2, 256), vocab: 384, block: 64, batch: 8, steps: 200 }),
        "small" => Some(Size { config: config(128, 4, 4, 352, 4, 1024), vocab: 2048, block: 512, batch: 8, steps: 100 }),
        "default" => Some(Size { config: config(256, 4, 4, 704, 4, 2048), vocab: 4096, block: 1024, batch: 4, steps: 100 }),
        _ => None,
    }
}

struct Args(Vec<String>);

impl Args {
    fn take(&mut self, flag: &str) -> Option<String> {
        let i = self.0.iter().position(|a| a == flag)?;
        let value = self.0.get(i + 1).cloned();
        self.0.drain(i..(i + 2).min(self.0.len()));
        value
    }
}

/// `text` as a JSONL corpus of one document per non-blank paragraph line, the last tenth held out.
fn write_corpora(text: &str, dir: &Path) -> std::io::Result<(PathBuf, PathBuf)> {
    let mut docs: Vec<String> = text.split(['\n', '.']).map(str::trim).filter(|d| !d.is_empty()).map(|d| format!("{d}.")).collect();
    if docs.len() < 20 {
        // A short text is repeated in rotation so the stream has room for windows and a held-out part.
        let base = docs.clone();
        while docs.len() < 400 {
            docs.extend(base.iter().cloned());
        }
    }
    let held = docs.split_off(docs.len() - docs.len() / 10);
    let write = |name: &str, docs: &[String]| -> std::io::Result<PathBuf> {
        let path = dir.join(name);
        std::fs::write(&path, docs.iter().map(|d| serde_json::json!({"text": d}).to_string()).collect::<Vec<_>>().join("\n"))?;
        Ok(path)
    };
    Ok((write("train.jsonl", &docs)?, write("held_out.jsonl", &held)?))
}

/// Greedy continuation of `prompt` by the checkpoint in `dir`.
fn continue_prompt(dir: &Path, prompt: &str, tokens: usize) -> Result<String, String> {
    use data::tokenizer::Tokenizer;
    let tok = data::qwen_tokenizer::QwenBpe::from_file(dir.join("tokenizer.json").to_str().ok_or("not UTF-8")?)?;
    let (cfg, src) = ouro::finetune::open(&dir.join("model.brain.safetensors"))?;
    let ids = tok.encode(prompt);
    let cap = (ids.len() + tokens).min(cfg.block_size as usize) as u32;
    let base = ouro::model::Ouro::new_inference(cfg, 1, 1, &*src);
    let decode = ouro::decode::Decode::new(&base, cap);
    let (mut out, mut logits) = (Vec::new(), Vec::new());
    for (pos, id) in ids.iter().enumerate() {
        logits = decode.step(*id, pos as u32);
    }
    for pos in ids.len()..cap as usize {
        let next = logits.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as u32).unwrap_or(0);
        out.push(next);
        if pos + 1 < cap as usize {
            logits = decode.step(next, pos as u32);
        }
    }
    Ok(tok.decode(&out))
}

fn run(mut args: Args) -> Result<(), String> {
    let out = PathBuf::from(args.take("--out").ok_or("--out is required")?);
    let size_name = args.take("--size").unwrap_or_else(|| "tiny".to_string());
    let mut size = size(&size_name).ok_or_else(|| format!("--size {size_name}: tiny, small or default"))?;
    if let Some(v) = args.take("--block") {
        size.block = v.parse().map_err(|_| format!("--block {v}"))?;
    }
    if let Some(v) = args.take("--batch") {
        size.batch = v.parse().map_err(|_| format!("--batch {v}"))?;
    }
    let steps: u32 = args.take("--steps").map_or(Ok(size.steps), |s| s.parse().map_err(|_| format!("--steps {s}")))?;
    let eval_every: u32 = args.take("--eval-every").map_or(Ok((steps / 4).max(1)), |s| s.parse().map_err(|_| format!("--eval-every {s}")))?;
    let tokens: usize = args.take("--tokens").map_or(Ok(48), |s| s.parse().map_err(|_| format!("--tokens {s}")))?;
    let seed: u64 = match args.take("--seed") {
        Some(s) => s.parse().map_err(|_| format!("--seed {s}"))?,
        None => brain::random_seed(),
    };
    let device = Device::parse(&args.take("--device").unwrap_or_else(|| "cpu".to_string())).map_err(|e| e.to_string())?;
    let text = match args.take("--text") {
        Some(path) => std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?,
        None => BUILT_IN.to_string(),
    };
    if let Some(extra) = args.0.first() {
        return Err(format!("unknown argument {extra}"));
    }
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let (train, held) = write_corpora(&text, &out).map_err(|e| e.to_string())?;

    let clock = Instant::now();
    let made = ModelInit::new("ouro")
        .config(size.config)
        .corpus(&train)
        .vocab_size(size.vocab)
        .special_tokens(SPECIALS)
        .eos_token("<|im_end|>")
        .chat_template(TEMPLATE)
        .seed(seed)
        .out_dir(out.join("model"))
        .run()
        .map_err(|e| e.to_string())?;
    println!("created {} ({} parameters, vocabulary {}, {} merges, seed {}, tool calls {}) in {:.1} s", made.checkpoint.display(), made.params, made.vocab_size, made.merges, made.seed, made.tool_calls, clock.elapsed().as_secs_f64());

    let clock = Instant::now();
    let step_tokens = f64::from(size.batch) * f64::from(size.block);
    let mut last = Instant::now();
    let trained = Pretrain::from_pretrained(made.checkpoint.to_str().ok_or("not UTF-8")?)
        .corpus(&train)
        .held_out(&held)
        .steps(steps)
        .batch(size.batch)
        .block(size.block)
        .eval_every(eval_every)
        .seed(seed)
        .device(device)
        .out_dir(out.join("pretrain"))
        .run_with(&brain::CancelToken::default(), |p| {
            // The rate of each step on its own, the evaluation a step ran included.
            let rate = step_tokens / last.elapsed().as_secs_f64();
            last = Instant::now();
            match p.eval_loss {
                Some(eval) => println!("step {:>5}  loss {:.4}  held-out {eval:.4}  {rate:.0} tokens/s", p.step, p.loss),
                None if p.step % 10 == 0 || p.step <= 3 => println!("step {:>5}  loss {:.4}  {rate:.0} tokens/s", p.step, p.loss),
                None => {}
            }
        })
        .map_err(|e| e.to_string())?;
    let seconds = clock.elapsed().as_secs_f64();
    let trained_tokens = u64::from(trained.steps_completed) * u64::from(size.batch) * u64::from(size.block);
    println!(
        "pretrained {} steps of {} x {} tokens: loss {:.4} -> {:.4}, {:.0} tokens/s over the whole run (load, steps, evaluations, export)",
        trained.steps_completed,
        size.batch,
        size.block,
        trained.initial_loss,
        trained.final_loss.unwrap_or(f32::NAN),
        trained_tokens as f64 / seconds
    );
    let checkpoint = trained.checkpoint.ok_or("the run wrote no checkpoint")?;
    let prompt = "In the morning the";
    println!("{prompt}{}", continue_prompt(&checkpoint, prompt, tokens)?);
    Ok(())
}

fn main() -> ExitCode {
    let args = Args(std::env::args().skip(1).collect());
    if args.0.iter().any(|a| a == "--help" || a == "-h") || args.0.is_empty() {
        eprint!("{USAGE}");
        return if args.0.is_empty() { ExitCode::FAILURE } else { ExitCode::SUCCESS };
    }
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("sample-models-ouro: {e}");
            ExitCode::FAILURE
        }
    }
}
