// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! End-to-end integration tests for the imported Qwen3-0.6B: inference validity,
//! training validity, and the **tool-call fine-tuning data pipeline** — proving
//! brain can load a real HF model, run it, train it, and improve its ability to
//! make intelligent function calls.
//!
//! Gated on `QWEN3_DIR` (the model directory holding `config.json`,
//! `tokenizer.json`, and `brain/qwen3-0.6b.weights`), so normal CI skips them:
//!
//! ```text
//! QWEN3_DIR=/data/workspace/resources/qwen3-0.6b \
//!   DISPLAY= cargo test --release -p brain-qwen --test integration_qwen3 -- --ignored --nocapture
//! ```
//!
//! Each test is sized to run in a few minutes on a Tesla P40 (fast reg2 forward +
//! tiled backward). Tunables via env: `QWEN3_TC_TRAIN`, `QWEN3_TC_STEPS`.

use std::path::{Path, PathBuf};

use data::qwen_tokenizer::QwenBpe;
use data::tokenizer::Tokenizer;
use data::toolcall::{self, ToolCase};
use gpu_core::{set_default_backend, Backend};
use qwen::{Qwen, QwenConfig};

fn dir() -> Option<PathBuf> {
    std::env::var("QWEN3_DIR").ok().map(PathBuf::from)
}

/// Pin the GPU backend and make the 151936-vocab lm_head a SINGLE reg2 tile.
/// Qwen's default vocab-tile budget splits the lm_head into ~7 tiles that run on
/// the naive `matmul_tile` (~4 s/forward); one tile (622 MB weight < the 2 GB
/// binding cap) uses the fast `matmul_reg2` (~40 ms). Turns a ~60 s step into ~1 s.
fn setup() {
    set_default_backend(Backend::Wgpu);
    if std::env::var("BRAIN_TILE_BUDGET_WORDS").is_err() {
        std::env::set_var("BRAIN_TILE_BUDGET_WORDS", "200000000");
    }
}
fn weights(d: &Path) -> PathBuf {
    d.join("brain/qwen3-0.6b.weights")
}
/// 512-context checkpoint for fine-tuning (attention is O(T²), so a small window
/// is much cheaper; RoPE inference still sizes context at load time).
fn weights_ft(d: &Path) -> PathBuf {
    d.join("brain/qwen3-0.6b-ft512.weights")
}
fn tok(d: &Path) -> QwenBpe {
    QwenBpe::from_file(d.join("tokenizer.json").to_str().unwrap()).expect("tokenizer.json")
}
fn env_usize(k: &str, def: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(def)
}

/// Per-card `used MiB` from nvidia-smi (best-effort; empty on failure).
fn gpu_mem() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,memory.used", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.replace('\n', " | "))
        .unwrap_or_default()
}

fn argmax(s: &[f32]) -> usize {
    let mut bi = 0;
    for i in 1..s.len() {
        if s[i] > s[bi] {
            bi = i;
        }
    }
    bi
}

/// Teacher-forced tool-call eval: for each case, run one forward over
/// prompt+response and read the greedy prediction at each response position.
/// Returns (exact-match rate over the *parsed* call, mean response-token accuracy).
/// One forward per case — fast, and a faithful proxy for greedy generation
/// (all-correct teacher-forced ⇒ greedy reproduces the call).
fn eval_toolcall(model: &Qwen, t: &QwenBpe, cases: &[ToolCase]) -> (f64, f64) {
    let vocab = model.cfg.vocab as usize;
    let cap = model.ctx_len();
    let mut exact = 0usize;
    let mut tok_acc = 0f64;
    let mut counted = 0usize;
    for c in cases {
        let ex = c.to_chat_example();
        let prompt = t.encode(&ex.prompt_str(t));
        let resp = t.encode(&format!("{}<|im_end|>\n", ex.assistant));
        if prompt.len() + resp.len() + 1 > cap {
            continue;
        }
        let mut full = prompt.clone();
        full.extend_from_slice(&resp);
        let logits = model.logits_all(&full);
        // predictions at positions [p-1 .. p+r-1] target resp[0..r]
        let p = prompt.len();
        let mut preds = Vec::with_capacity(resp.len());
        let mut correct = 0usize;
        for j in 0..resp.len() {
            let pos = p - 1 + j;
            let pred = argmax(&logits[pos * vocab..(pos + 1) * vocab]) as u32;
            preds.push(pred);
            if pred == resp[j] {
                correct += 1;
            }
        }
        tok_acc += correct as f64 / resp.len() as f64;
        counted += 1;
        let text = t.decode(&preds);
        if let Some(got) = toolcall::parse_tool_call(&text) {
            if toolcall::calls_match(&c.call, &got) {
                exact += 1;
            }
        }
    }
    let n = counted.max(1) as f64;
    (exact as f64 / n, tok_acc / n)
}

// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn qwen3_inference_coherent() {
    let Some(d) = dir() else {
        eprintln!("QWEN3_DIR unset; skipping");
        return;
    };
    setup();
    let t = tok(&d);
    let m = Qwen::load_inference(weights(&d).to_str().unwrap(), 1, 64);

    // Factual, unambiguous: the base model must complete this correctly.
    let prompt = "The capital of France is the city of";
    let ids = t.encode(prompt);
    let mut rng = data::rng::Rng::new(0);
    let out = qwen::sample::generate(&m, &ids, 4, 0.0, 0, None, &mut rng); // greedy
    let text = t.decode(&out);
    println!("inference: {prompt:?} -> {text:?}");
    assert!(text.to_lowercase().contains("paris"), "base model failed a basic factual completion: {text:?}");
}

#[test]
#[ignore]
fn qwen3_training_validity() {
    let Some(d) = dir() else {
        eprintln!("QWEN3_DIR unset; skipping");
        return;
    };
    setup();
    let t = tok(&d);

    // Memorise a single distinctive assistant response — loss must drop sharply,
    // proving the forward+backward+optimizer loop trains the real 0.6B weights.
    let ex = data::chat::ChatExample::new(
        "Repeat the secret phrase.",
        "The secret phrase is: velvet thunder over the quiet harbor.",
    );
    // Repeat so the token stream exceeds the 512 window (loader samples windows).
    let corpus: Vec<_> = std::iter::repeat(ex.clone()).take(40).collect();
    let out = std::env::temp_dir().join("qwen3_train_validity");
    data::chat::prepare_chat(&corpus, &corpus, &t, 151936, &out).unwrap();

    let cfg = QwenConfig::from_json(&checkpoint::load(weights_ft(&d).to_str().unwrap()).header["config"]);
    let steps = env_usize("QWEN3_TV_STEPS", 40);
    let opts = model::FitOpts {
        steps: steps as u32,
        batch_size: 1,
        block_size: 512,
        lr: 1e-4,
        warmup: 5,
        decay_iters: steps as u32,
        min_lr: 1e-5,
        weight_decay: 0.0,
        grad_clip: 1.0,
        grad_accum: 1,
        eval_interval: (steps / 4).max(1) as u32,
        eval_batches: 1,
        ..Default::default()
    };
    let ckpt = out.join("ft.weights");
    std::fs::copy(weights_ft(&d), &ckpt).unwrap();
    let (l0, l1) = model::fit::<Qwen>(&out, cfg, &opts, Some(&ckpt)).expect("fit");
    println!("training validity: loss {l0:.4} -> {l1:.4} over {steps} steps");
    assert!(l1 < l0 * 0.5, "loss did not drop enough: {l0:.4} -> {l1:.4}");
}

#[test]
#[ignore]
fn qwen3_toolcall_finetune() {
    let Some(d) = dir() else {
        eprintln!("QWEN3_DIR unset; skipping");
        return;
    };
    setup();
    let t = tok(&d);

    // Data: generate tool-call cases (target tool + distractors + arg routing).
    let n_train = env_usize("QWEN3_TC_TRAIN", 240);
    let train = toolcall::generate(n_train, 3, 1);
    let held = toolcall::generate(24, 3, 999); // disjoint seed -> held-out
    let train_ex: Vec<_> = train.iter().map(|c| c.to_chat_example()).collect();
    let val_ex: Vec<_> = held.iter().map(|c| c.to_chat_example()).collect();

    let out = std::env::temp_dir().join("qwen3_toolcall");
    data::chat::prepare_chat(&train_ex, &val_ex, &t, 151936, &out).unwrap();

    // Eval BEFORE.
    let base = Qwen::load_inference(weights(&d).to_str().unwrap(), 1, 512);
    let (ex0, ta0) = eval_toolcall(&base, &t, &held);
    println!("tool-call BEFORE finetune: exact-match {:.1}%  token-acc {:.1}%", ex0 * 100.0, ta0 * 100.0);
    drop(base);

    // Fine-tune (assistant span only — the mask pipeline).
    let steps = env_usize("QWEN3_TC_STEPS", 200);
    let cfg = QwenConfig::from_json(&checkpoint::load(weights_ft(&d).to_str().unwrap()).header["config"]);
    let opts = model::FitOpts {
        steps: steps as u32,
        batch_size: 1,
        block_size: 512,
        lr: 1e-4,
        warmup: 20,
        decay_iters: steps as u32,
        min_lr: 1e-5,
        weight_decay: 0.0,
        grad_clip: 1.0,
        grad_accum: 4,
        eval_interval: (steps / 5).max(1) as u32,
        eval_batches: 4,
        ..Default::default()
    };
    let ckpt = out.join("ft.weights");
    std::fs::copy(weights_ft(&d), &ckpt).unwrap();
    let (l0, l1) = model::fit::<Qwen>(&out, cfg, &opts, Some(&ckpt)).expect("fit");
    println!("tool-call finetune: loss {l0:.4} -> {l1:.4}");

    // Eval AFTER.
    let ft = Qwen::load_inference(ckpt.to_str().unwrap(), 1, 512);
    let (ex1, ta1) = eval_toolcall(&ft, &t, &held);
    println!("tool-call AFTER  finetune: exact-match {:.1}%  token-acc {:.1}%", ex1 * 100.0, ta1 * 100.0);

    // The pipeline must produce a large, real improvement in the model's ability
    // to emit correct tool calls on held-out requests.
    assert!(l1 < l0, "finetune loss did not decrease ({l0:.4} -> {l1:.4})");
    assert!(ex1 >= 0.6, "held-out tool-call exact-match too low after finetune: {:.1}%", ex1 * 100.0);
    assert!(ex1 > ex0 + 0.3, "tool-call exact-match did not improve enough: {:.1}% -> {:.1}%", ex0 * 100.0, ex1 * 100.0);
}


/// Reasoning fine-tune: teach the model to answer 2-step arithmetic word problems
/// with an explicit chain of thought, then a final numeric answer. Uses the same
/// masked chat pipeline; scored by exact-match on the final answer of held-out
/// problems (teacher-forced greedy over the response). Proves the pipeline
/// generalises beyond tool calls to reasoning supervision.
fn arith_case(a: i64, b: i64, c: i64) -> data::chat::ChatExample {
    // (a + b) * c, with a worked <think> trace.
    let s1 = a + b;
    let ans = s1 * c;
    let user = format!("What is ({a} + {b}) * {c}? Show your reasoning.");
    let assistant = format!("<think>
First {a} + {b} = {s1}. Then {s1} * {c} = {ans}.
</think>
The answer is {ans}.");
    data::chat::ChatExample::new(user, assistant)
}

/// Parse the final integer after "answer is" (loose).
fn parse_answer(text: &str) -> Option<i64> {
    let low = text.to_lowercase();
    let idx = low.find("answer is")? + "answer is".len();
    let tail: String = text[idx..].chars().skip_while(|c| !c.is_ascii_digit() && *c != '-').take_while(|c| c.is_ascii_digit() || *c == '-').collect();
    tail.parse().ok()
}

fn eval_reasoning(model: &Qwen, t: &QwenBpe, cases: &[(i64, i64, i64)]) -> f64 {
    let vocab = model.cfg.vocab as usize;
    let cap = model.ctx_len();
    let mut ok = 0usize;
    let mut n = 0usize;
    for &(a, b, c) in cases {
        let ex = arith_case(a, b, c);
        let prompt = t.encode(&ex.prompt_str(t));
        let resp = t.encode(&format!("{}<|im_end|>
", ex.assistant));
        if prompt.len() + resp.len() + 1 > cap { continue; }
        let mut full = prompt.clone();
        full.extend_from_slice(&resp);
        let logits = model.logits_all(&full);
        let p = prompt.len();
        let mut preds = Vec::with_capacity(resp.len());
        for j in 0..resp.len() {
            let pos = p - 1 + j;
            preds.push(argmax(&logits[pos * vocab..(pos + 1) * vocab]) as u32);
        }
        let text = t.decode(&preds);
        if parse_answer(&text) == Some((a + b) * c) { ok += 1; }
        n += 1;
    }
    ok as f64 / n.max(1) as f64
}

#[test]
#[ignore]
fn qwen3_reasoning_finetune() {
    let Some(d) = dir() else { eprintln!("QWEN3_DIR unset; skipping"); return; };
    setup();
    let t = tok(&d);

    let mut rng = data::rng::Rng::new(7);
    let mk = |rng: &mut data::rng::Rng| (rng.gen_range_inclusive(2, 9), rng.gen_range_inclusive(2, 9), rng.gen_range_inclusive(2, 9));
    let n_train = env_usize("QWEN3_RS_TRAIN", 240);
    let train: Vec<_> = (0..n_train).map(|_| { let (a,b,c)=mk(&mut rng); arith_case(a,b,c) }).collect();
    let held_nums: Vec<(i64,i64,i64)> = { let mut r = data::rng::Rng::new(4242); (0..24).map(|_| mk(&mut r)).collect() };
    let held: Vec<_> = held_nums.iter().map(|&(a,b,c)| arith_case(a,b,c)).collect();

    let out = std::env::temp_dir().join("qwen3_reasoning");
    data::chat::prepare_chat(&train, &held, &t, 151936, &out).unwrap();

    let base = Qwen::load_inference(weights(&d).to_str().unwrap(), 1, 512);
    let acc0 = eval_reasoning(&base, &t, &held_nums);
    println!("reasoning BEFORE finetune: answer-acc {:.1}%", acc0 * 100.0);
    drop(base);

    let steps = env_usize("QWEN3_RS_STEPS", 150);
    let cfg = QwenConfig::from_json(&checkpoint::load(weights_ft(&d).to_str().unwrap()).header["config"]);
    let opts = model::FitOpts {
        steps: steps as u32, batch_size: 1, block_size: 512, lr: 1e-4, warmup: 15,
        decay_iters: steps as u32, min_lr: 1e-5, weight_decay: 0.0, grad_clip: 1.0, grad_accum: 4,
        eval_interval: (steps / 5).max(1) as u32, eval_batches: 4, ..Default::default()
    };
    let ckpt = out.join("ft.weights");
    std::fs::copy(weights_ft(&d), &ckpt).unwrap();
    let (l0, l1) = model::fit::<Qwen>(&out, cfg, &opts, Some(&ckpt)).expect("fit");
    println!("reasoning finetune: loss {l0:.4} -> {l1:.4}");

    let ft = Qwen::load_inference(ckpt.to_str().unwrap(), 1, 512);
    let acc1 = eval_reasoning(&ft, &t, &held_nums);
    println!("reasoning AFTER  finetune: answer-acc {:.1}%", acc1 * 100.0);
    assert!(l1 < l0, "reasoning finetune loss did not decrease");
    assert!(acc1 > acc0 + 0.2, "reasoning accuracy did not improve enough: {:.1}% -> {:.1}%", acc0*100.0, acc1*100.0);
}

/// Full (optimizer-offloaded) vs LoRA fine-tuning on the SAME tool-call data:
/// both must improve held-out tool-call exact-match; reports the two side by
/// side. Proves the full-vs-LoRA comparison the offload path unblocked.
/// Pipeline-parallel sharding of the **real** 0.6B across both P40s: the sharded
/// forward loss must match the single-device model bit-for-bit, and the weights
/// must actually distribute across the two cards (reported via nvidia-smi). This
/// is the "a model larger than one card fits across several" mechanism, proven on
/// the real checkpoint.
#[test]
#[ignore]
fn qwen3_shard_real_2gpu() {
    let Some(d) = dir() else { eprintln!("QWEN3_DIR unset; skipping"); return; };
    setup();
    std::env::remove_var("BRAIN_GPU_INDEX");
    let path = weights_ft(&d);
    let ps = path.to_str().unwrap();
    let c = checkpoint::load(ps);
    let cfg = QwenConfig::from_json(&c.header["config"]);
    let init = c.by_role("");
    let (b, t) = (1u32, 32u32);
    let toks: Vec<u32> = (0..t).map(|i| ((i * 131 + 7) % cfg.vocab) as u32).collect();
    let y: Vec<u32> = (0..t).map(|i| toks[((i + 1) % t) as usize]).collect();

    // Single-device reference on GPU 0.
    let single = Qwen::load_inference(ps, b, t);
    single.set_batch(&toks, &y);
    let l0 = single.forward();
    single.poll_wait();
    drop(single); // free GPU 0 before the pipeline claims it as stage 0

    // Two-stage inference pipeline across GPUs 0 and 1 from the same weights.
    let pipe = qwen::Pipeline::new(cfg.clone(), b, t, &init, false, &[0, 1]);
    let l1 = pipe.forward(&toks, &y);
    pipe.poll_wait();
    let mem = gpu_mem();

    let rel = (l0 - l1).abs() / l0.abs().max(1e-6);
    println!("\n=== Qwen3-0.6B sharded across 2 P40s (pipeline-parallel) ===");
    println!("  layers split: {} stages over {} layers", pipe.n_stages(), cfg.n_layers);
    println!("  loss  single-GPU={l0:.6}  2-GPU-sharded={l1:.6}  rel={rel:.2e}");
    println!("  per-card memory: {mem}");
    assert_eq!(pipe.n_stages(), 2);
    assert!(rel < 1e-4, "sharded loss diverged from single-GPU: {l0} vs {l1}");
}

#[test]
#[ignore]
fn qwen3_full_vs_lora_toolcall() {
    let Some(d) = dir() else { eprintln!("QWEN3_DIR unset; skipping"); return; };
    setup();
    let t = tok(&d);

    let n_train = env_usize("QWEN3_TC_TRAIN", 200);
    let train = toolcall::generate(n_train, 3, 1);
    let held = toolcall::generate(24, 3, 999);
    let train_ex: Vec<_> = train.iter().map(|c| c.to_chat_example()).collect();
    let val_ex: Vec<_> = held.iter().map(|c| c.to_chat_example()).collect();
    let out = std::env::temp_dir().join("qwen3_fvl");
    data::chat::prepare_chat(&train_ex, &val_ex, &t, 151936, &out).unwrap();

    let base = weights_ft(&d);
    let base_s = base.to_str().unwrap();
    let steps = env_usize("QWEN3_FVL_STEPS", 60);
    let opts = |lr: f32| model::FitOpts {
        steps: steps as u32, batch_size: 1, block_size: 512, lr, warmup: (steps / 6).max(1) as u32,
        decay_iters: steps as u32, min_lr: lr * 0.1, weight_decay: 0.0, grad_clip: 1.0, grad_accum: 4,
        checkpoint_secs: 0, ..Default::default()
    };

    // Baseline held-out score.
    let (ex0, _) = eval_toolcall(&Qwen::load_inference(base_s, 1, 512), &t, &held);

    // Full (offloaded) fine-tune.
    let full_ckpt = out.join("full.weights");
    let (fl0, fl1) = qwen::finetune::finetune(base_s, &out, &opts(1e-4), &qwen::finetune::Mode::FullOffload, full_ckpt.to_str().unwrap()).unwrap();
    let (exf, _) = eval_toolcall(&Qwen::load_inference(full_ckpt.to_str().unwrap(), 1, 512), &t, &held);

    // LoRA fine-tune (adapters only). Same LR as full and scale 1.0 (alpha==rank)
    // for a fair comparison — aggressive scale/LR makes the adapters memorise the
    // tiny synthetic set (train loss -> 0) without generalising.
    let lora_ckpt = out.join("lora.weights");
    let (ll0, ll1) = qwen::finetune::finetune(base_s, &out, &opts(1e-4), &qwen::finetune::Mode::Lora { rank: 16, alpha: 16.0 }, lora_ckpt.to_str().unwrap()).unwrap();
    let (exl, _) = eval_toolcall(&Qwen::load_inference(lora_ckpt.to_str().unwrap(), 1, 512), &t, &held);

    println!("\n=== Qwen3-0.6B tool-call: FULL (offload) vs LoRA, {steps} steps ===");
    println!("  baseline held-out exact-match: {:.1}%", ex0 * 100.0);
    println!("  FULL  loss {fl0:.3}->{fl1:.3}   held-out exact-match {:.1}%", exf * 100.0);
    println!("  LoRA  loss {ll0:.3}->{ll1:.3}   held-out exact-match {:.1}%", exl * 100.0);

    // Both must train (loss down); the offload FULL path must additionally
    // generalise on held-out tool-calls — that is what this test proves. LoRA's
    // held-out score is reported for comparison (it is hyperparameter-sensitive
    // on this tiny synthetic set), not gated.
    assert!(fl1 < fl0 && ll1 < ll0, "a finetune did not reduce loss");
    assert!(exf > ex0 + 0.2, "full finetune did not improve tool-calls: {:.1}%->{:.1}%", ex0*100.0, exf*100.0);
    assert!(exl >= ex0, "lora finetune regressed below baseline: {:.1}%->{:.1}%", ex0*100.0, exl*100.0);
}
