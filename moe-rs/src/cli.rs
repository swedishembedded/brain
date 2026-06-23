// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! CLI for the PID event/effect transformer: validate / validate-stream /
//! train / rollout. Dispatched from `main` via `moe pid <action> ...`.

use std::collections::HashMap;

use crate::checkpoint;
use crate::pid::{Pid, PidConfig, BOS, DECIDE, IGNORE};
use crate::pid_data::{self, Rng};

fn map_label(v: f32) -> u32 {
    if v < -0.5 {
        IGNORE
    } else {
        v.round() as u32
    }
}

fn max_err(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut mae = 0.0f32;
    let mut mre = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let ae = (x - y).abs();
        mae = mae.max(ae);
        if y.abs() > 1e-3 {
            mre = mre.max(ae / y.abs());
        }
    }
    (mae, mre)
}

pub fn run_pid(args: &[String]) {
    match args.first().map(|s| s.as_str()) {
        Some("validate") => validate(args.get(1).map(|s| s.as_str()).unwrap_or("../pid_ref.bin")),
        Some("stream") => validate_stream(args.get(1).map(|s| s.as_str()).unwrap_or("../pid_stream.bin")),
        Some("train") => train(&args[1..]),
        Some("rollout") => rollout(&args[1..]),
        Some("profile") => profile(&args[1..]),
        #[cfg(feature = "vulkan-coopmat")]
        Some("vk-info") => crate::vulkan::print_vk_info(),
        #[cfg(feature = "vulkan-coopmat")]
        Some("vk-matmul") => crate::vulkan::cooperative_matmul_demo(),
        other => {
            eprintln!("usage: moe pid <validate|stream|train|rollout|profile> ...  (got {other:?})");
        }
    }
}

/// Single-step parity gate vs the PyTorch reference.
fn validate(path: &str) {
    let c = checkpoint::load(path);
    let cfg = PidConfig::from_json(&c.header["config"]);
    let b = c.header["B"].as_u64().unwrap() as u32;
    let t = c.header["T"].as_u64().unwrap() as u32;
    let init = c.by_role("init");
    let grad = c.by_role("grad");
    let updated = c.by_role("updated");
    let opt = &c.header["opt"];
    let lr = opt["lr"].as_f64().unwrap() as f32;
    let wd = opt["weight_decay"].as_f64().unwrap() as f32;

    let batch_x = c.find("batch_x", "data").expect("batch_x");
    let batch_y = c.find("batch_y", "data").expect("batch_y");
    let xs: Vec<u32> = batch_x.iter().map(|v| v.round() as u32).collect();
    let ys: Vec<u32> = batch_y.iter().map(|&v| map_label(v)).collect();

    let model = Pid::new(cfg.clone(), b, t, &init);
    model.set_batch(&xs, &ys);

    let ce = model.forward();
    println!("loss: rust_ce={:.6}  py_ce={:.6}", ce, c.header["losses"]["ce"].as_f64().unwrap());

    if std::env::var("PID_DUMP").is_ok() {
        let r0 = model.read_res0();
        println!("rust res0[0..6] = {:?}", &r0[0..6]);
        let pw = model.read_weight("pos.weight");
        println!("rust pos.weight[0..6] = {:?}", &pw[0..6]);
        let tw = model.read_weight("tok.weight");
        let bos = 257usize * 32;
        println!("rust tok.weight[257][0..6] = {:?}", &tw[bos..bos + 6]);
    }
    if let Some(ref_logits) = c.find("forward_logits", "logits") {
        let r = model.read_logits();
        let (mae, mre) = max_err(&r, ref_logits);
        println!("\n== forward logits ==\nmax abs logit error = {:.3e}  max rel = {:.3e}", mae, mre);
        if std::env::var("PID_DUMP").is_ok() {
            println!("rust logits[0..6] = {:?}", &r[0..6]);
            println!("ref  logits[0..6] = {:?}", &ref_logits[0..6]);
        }
    }

    model.backward();
    println!("\n== gradient check (Rust vs PyTorch autograd) ==");
    let (mut g_mae, mut g_mre, mut worst) = (0.0f32, 0.0f32, String::new());
    for (name, _) in model.ps.params.iter() {
        let r = model.read_grad(name);
        let (mae, mre) = max_err(&r, &grad[name]);
        if mae > g_mae {
            g_mae = mae;
            worst = name.clone();
        }
        g_mre = g_mre.max(mre);
    }
    println!("max abs grad error = {:.3e} (worst: {})", g_mae, worst);
    println!("max rel grad error = {:.3e}", g_mre);

    model.adamw_step(1, lr, wd, Some(1.0), 1.0);
    println!("\n== weight check after one AdamW(+clip) step ==");
    let (mut w_mae, mut w_mre) = (0.0f32, 0.0f32);
    for (name, _) in model.ps.params.iter() {
        let r = model.read_weight(name);
        let (mae, mre) = max_err(&r, &updated[name]);
        w_mae = w_mae.max(mae);
        w_mre = w_mre.max(mre);
    }
    println!("max abs weight error = {:.3e}", w_mae);
    println!("max rel weight error = {:.3e}", w_mre);

    let ok = g_mae < 2e-3 && w_mae < 2e-4;
    println!("\n{}", if ok { "VALIDATION PASSED" } else { "VALIDATION FAILED" });
}

/// Fixed-data multi-step check: replay the exact PyTorch batch stream and
/// compare the loss curve + final weights (no RNG divergence).
fn validate_stream(path: &str) {
    let c = checkpoint::load(path);
    let cfg = PidConfig::from_json(&c.header["config"]);
    let b = c.header["B"].as_u64().unwrap() as u32;
    let t = c.header["T"].as_u64().unwrap() as u32;
    let steps = c.header["steps"].as_u64().unwrap() as usize;
    let opt = &c.header["opt"];
    let lr = opt["lr"].as_f64().unwrap() as f32;
    let wd = opt["weight_decay"].as_f64().unwrap() as f32;
    let init = c.by_role("init");
    let final_w = c.by_role("final");
    let py_losses: Vec<f32> = c.header["losses"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();

    let xs_all = c.find("batch_x", "data").expect("batch_x");
    let ys_all = c.find("batch_y", "data").expect("batch_y");
    let span = (b * t) as usize;

    let model = Pid::new(cfg.clone(), b, t, &init);
    println!("replaying {steps} steps (B={b}, T={t})");
    let mut max_loss_err = 0.0f32;
    for step in 0..steps {
        let xs: Vec<u32> = xs_all[step * span..(step + 1) * span].iter().map(|v| v.round() as u32).collect();
        let ys: Vec<u32> = ys_all[step * span..(step + 1) * span].iter().map(|&v| map_label(v)).collect();
        model.set_batch(&xs, &ys);
        model.zero_grads();
        let loss = model.forward();
        model.backward();
        model.adamw_step((step + 1) as u32, lr, wd, Some(1.0), 1.0);
        let e = (loss - py_losses[step]).abs();
        max_loss_err = max_loss_err.max(e);
        if step < 3 || step % 50 == 0 || step + 1 == steps {
            println!("step {:4} | rust {:.5} | py {:.5} | |d| {:.2e}", step + 1, loss, py_losses[step], e);
        }
    }
    println!("\nmax per-step loss error = {:.3e}", max_loss_err);
    let (mut w_mae, mut worst) = (0.0f32, String::new());
    for (name, _) in model.ps.params.iter() {
        let r = model.read_weight(name);
        let (mae, _) = max_err(&r, &final_w[name]);
        if mae > w_mae {
            w_mae = mae;
            worst = name.clone();
        }
    }
    println!("max abs final-weight error = {:.3e} (worst: {})", w_mae, worst);
    let ok = max_loss_err < 5e-3 && w_mae < 1e-2;
    println!("\n{}", if ok { "STREAM VALIDATION PASSED" } else { "STREAM VALIDATION FAILED" });
}

struct TrainCfg {
    steps: u32,
    eff_batch: u32,
    seq_steps: u32,
    t_pad: u32,
    d_model: u32,
    layers: u32,
    heads: u32,
    lr: f32,
    wd: f32,
    seed: u64,
    n_traj: usize,
    traj_len: usize,
    mem_budget: u64,
    out: String,
    ckpt_every: u32,
}

/// Memory-budgeted microbatch + accumulation planner.
fn plan_batch(cfg: &PidConfig, t: u32, eff_batch: u32, budget: u64) -> (u32, u32) {
    // batch-independent: weights + grad + 2 adam moments (4x params).
    let total_params: u64 = cfg.param_list().iter().map(|(_, n)| *n as u64).sum();
    let fixed = 4 * total_params * 4;
    // per-sample activation + backward scratch bytes (rough closed form).
    let d = cfg.d_model as u64;
    let ff = cfg.d_ff as u64;
    let u = cfg.u_bins as u64;
    let h = cfg.n_heads as u64;
    let tt = t as u64;
    let per_sample = 4 * (
        (cfg.n_layers as u64 + 1) * d * 2
        + cfg.n_layers as u64 * (2 * d + 3 * d + d + d + d + 3 * ff + 3 * ff)
        + h * tt * 3
        + d * 6 + u * 2 + ff * 3
    );
    let avail = budget.saturating_sub(fixed).max(per_sample);
    let mut b_micro = ((avail / per_sample) as u32).clamp(1, eff_batch);
    while eff_batch % b_micro != 0 {
        b_micro -= 1;
    }
    (b_micro, eff_batch / b_micro)
}

fn parse_train(args: &[String]) -> TrainCfg {
    let mut c = TrainCfg {
        steps: 2000,
        eff_batch: 32,
        seq_steps: 12,
        t_pad: 256,
        d_model: 64,
        layers: 2,
        heads: 4,
        lr: 1e-3,
        wd: 0.01,
        seed: 7,
        n_traj: 180,
        traj_len: 80,
        mem_budget: 4u64 << 30,
        out: "moe_pid.weights".to_string(),
        ckpt_every: 100,
    };
    let mut i = 0;
    while i < args.len() {
        let next = || args.get(i + 1).cloned().expect("missing value");
        match args[i].as_str() {
            "--steps" => c.steps = next().parse().unwrap(),
            "--effective-batch" | "--batch-size" => c.eff_batch = next().parse().unwrap(),
            "--seq-steps" => c.seq_steps = next().parse().unwrap(),
            "--seq-len" | "--block-size" => c.t_pad = next().parse().unwrap(),
            "--d-model" => c.d_model = next().parse().unwrap(),
            "--layers" => c.layers = next().parse().unwrap(),
            "--heads" => c.heads = next().parse().unwrap(),
            "--lr" => c.lr = next().parse().unwrap(),
            "--weight-decay" => c.wd = next().parse().unwrap(),
            "--seed" => c.seed = next().parse().unwrap(),
            "--n-traj" => c.n_traj = next().parse().unwrap(),
            "--traj-len" => c.traj_len = next().parse().unwrap(),
            "--mem-budget" => c.mem_budget = next().parse().unwrap(),
            "--out" => c.out = next(),
            "--checkpoint-every" => c.ckpt_every = next().parse().unwrap(),
            other => panic!("unknown pid train flag: {other}"),
        }
        i += 2;
    }
    c
}

fn train(args: &[String]) {
    let tc = parse_train(args);
    // Each event/effect step is at most ~17 tokens (variable-length CBOR), plus the
    // leading BOS. Raise --seq-len if it is too small to hold a full window.
    let min_t = 1 + 18 * tc.seq_steps;
    let t_pad = tc.t_pad.max(min_t);
    if t_pad != tc.t_pad {
        eprintln!("note: raising --seq-len {} -> {} to fit {} steps/window", tc.t_pad, t_pad, tc.seq_steps);
    }
    let cfg = PidConfig {
        block_size: t_pad,
        n_layers: tc.layers,
        d_model: tc.d_model,
        n_heads: tc.heads,
        d_ff: 4 * tc.d_model,
        ..PidConfig::default_small()
    };
    let (b_micro, n_accum) = plan_batch(&cfg, t_pad, tc.eff_batch, tc.mem_budget);
    println!(
        "pid train: eff_batch={} -> microbatch={} x accum={} (mem budget {} MiB)",
        tc.eff_batch, b_micro, n_accum, tc.mem_budget >> 20
    );

    let plants = pid_data::training_plants();
    let ds = pid_data::PidDataset::new(&plants, tc.n_traj, tc.traj_len, tc.seq_steps as usize, t_pad as usize, tc.seed);
    let init = pid_data::init_weights(&cfg, tc.seed);
    let model = Pid::new(cfg.clone(), b_micro, t_pad, &init);

    let mut rng = Rng::new(tc.seed ^ 0xABCD_1234);
    for step in 1..=tc.steps {
        let logging = step == 1 || step % 100 == 0 || step == tc.steps;
        model.zero_grads();
        for _ in 0..n_accum {
            let (xs, ys) = ds.batch(b_micro as usize, &mut rng);
            model.set_batch(&xs, &ys);
            // forward/backward stay fully on-device -- no per-microbatch readback.
            model.forward_submit();
            model.backward();
        }
        model.adamw_step(step, tc.lr, tc.wd, Some(1.0), 1.0 / n_accum as f32);
        // Wait for this step's GPU work and reclaim its transient submit memory.
        // The loop is otherwise submit-only between log intervals, so without
        // this the per-submit staging/command buffers accumulate until the GPU
        // aperture is exhausted and a later allocation fails mid-run.
        model.poll_wait();
        if logging {
            // read the loss only when logging (one D2H every log interval, not per step)
            println!("step {:5} | loss {:.4}", step, model.loss());
        }
        // Periodic checkpoint so a long run survives a crash/device loss. The
        // write is atomic (temp + rename), so the on-disk checkpoint is always a
        // complete, loadable file. Resume/inspect with `moe pid rollout --weights`.
        if tc.ckpt_every > 0 && step % tc.ckpt_every == 0 && step != tc.steps {
            model.save(&tc.out);
            println!("  checkpoint saved at step {step} -> {}", tc.out);
        }
    }
    model.save(&tc.out);
    println!("saved {}", tc.out);

    // closed-loop generalization report (train + interpolated validation plants)
    let dec = Pid::new(cfg.clone(), 1, t_pad, &model_weights(&model));
    report("TRAIN plants", &dec, &cfg, &pid_data::training_plants());
    report("VALIDATION plants (interpolated)", &dec, &cfg, &pid_data::validation_plants());
}

fn model_weights(model: &Pid) -> HashMap<String, Vec<f32>> {
    model.ps.params.iter().map(|(n, _)| (n.clone(), model.read_weight(n))).collect()
}

fn report(title: &str, dec: &Pid, cfg: &PidConfig, plants: &[pid_data::PlantSpec]) {
    println!("\n== {title}: closed loop (model vs per-plant oracle) ==");
    let mut gap = 0.0;
    for spec in plants {
        let (mse, ss) = pid_data::rollout_on_plant(dec, cfg, spec, 180);
        let oracle = pid_data::run_oracle_closed_loop(spec, 180);
        let omse = oracle.iter().map(|&(_, sp, y)| (sp - y).powi(2)).sum::<f32>() / oracle.len() as f32;
        gap += mse - omse;
        println!("  tau={:.3} gain={:.3} | model MSE {:.4} oracle {:.4} | model ss|e| {:.4}", spec.tau, spec.gain, mse, omse, ss);
    }
    println!("  mean MSE gap = {:+.4}", gap / plants.len() as f32);
}

/// Load a checkpoint and run the closed-loop generalization report.
fn rollout(args: &[String]) {
    let mut weights = "moe_pid.weights".to_string();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--weights" => weights = args.get(i + 1).cloned().expect("missing"),
            other => panic!("unknown rollout flag: {other}"),
        }
        i += 2;
    }
    let c = checkpoint::load(&weights);
    let cfg = PidConfig::from_json(&c.header["config"]);
    let init = c.by_role("");
    let dec = Pid::new(cfg.clone(), 1, cfg.block_size, &init);
    report("TRAIN plants", &dec, &cfg, &pid_data::training_plants());
    report("VALIDATION plants (interpolated)", &dec, &cfg, &pid_data::validation_plants());
}

/// Time a single inference path (one forward to the DECIDE position) averaged
/// over `--cycles`, against previously trained weights.
fn profile(args: &[String]) {
    let mut weights = "moe_pid.weights".to_string();
    let mut cycles = 100usize;
    let mut seq_len = 64usize;
    let mut warmup = 10usize;
    let mut i = 0;
    while i < args.len() {
        let next = || args.get(i + 1).cloned().expect("missing value");
        match args[i].as_str() {
            "--weights" => weights = next(),
            "--cycles" => cycles = next().parse().unwrap(),
            "--seq-len" => seq_len = next().parse().unwrap(),
            "--warmup" => warmup = next().parse().unwrap(),
            other => panic!("unknown profile flag: {other}"),
        }
        i += 2;
    }

    let c = checkpoint::load(&weights);
    let cfg = PidConfig::from_json(&c.header["config"]);
    let init = c.by_role("");
    let dec = Pid::new(cfg.clone(), 1, cfg.block_size, &init);
    let seq_len = seq_len.min(cfg.block_size as usize).max(1);

    // Build a representative context of exactly `seq_len` tokens (BOS + repeated
    // event/DECIDE/effect records). Forward cost depends on the token count, not
    // the values, so any valid stream of this length is representative.
    let mut ctx = vec![BOS];
    while ctx.len() < seq_len {
        ctx.extend(crate::pid_data::encode_event(0.3, 0.1, 0.2));
        ctx.push(DECIDE);
        ctx.extend(crate::pid_data::encode_effect_bin(40));
    }
    ctx.truncate(seq_len);

    let (mean, min, max) = dec.profile_inference(&ctx, cycles, warmup);
    println!("inference profile  (weights={weights})");
    println!("  config: d_model={} layers={} heads={} block_size={}", cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.block_size);
    println!("  context length: {seq_len} tokens   cycles: {cycles}   warmup: {warmup}");
    println!("  mean: {:.6} ms  ({:.1} us)", mean * 1e3, mean * 1e6);
    println!("  min:  {:.6} ms   max: {:.6} ms", min * 1e3, max * 1e3);
    println!("  throughput: {:.1} inferences/sec", 1.0 / mean);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_mapping() {
        assert_eq!(map_label(-100.0), IGNORE);
        assert_eq!(map_label(-1.0), IGNORE);
        assert_eq!(map_label(0.0), 0);
        assert_eq!(map_label(40.0), 40);
        assert_eq!(map_label(80.0), 80);
    }

    #[test]
    fn max_err_abs_and_rel() {
        let (mae, mre) = max_err(&[1.0, 2.0, 3.0], &[1.0, 2.1, 3.0]);
        assert!((mae - 0.1).abs() < 1e-6);
        assert!((mre - 0.1 / 2.1).abs() < 1e-6);
        // small reference magnitudes are excluded from the relative metric
        let (mae2, mre2) = max_err(&[0.0005], &[0.0001]);
        assert!((mae2 - 0.0004).abs() < 1e-9);
        assert_eq!(mre2, 0.0);
    }

    #[test]
    fn batch_planner_budget_and_divisibility() {
        let cfg = PidConfig::default_small();
        // generous budget -> single microbatch, no accumulation
        let (bm, acc) = plan_batch(&cfg, 64, 16, 8u64 << 30);
        assert_eq!(bm, 16);
        assert_eq!(acc, 1);
        // tiny budget -> microbatch shrinks but still divides the effective batch
        let (bm2, acc2) = plan_batch(&cfg, 256, 12, 1 << 20);
        assert!(bm2 >= 1 && bm2 <= 12);
        assert_eq!(12 % bm2, 0);
        assert_eq!(bm2 * acc2, 12);
    }
}
