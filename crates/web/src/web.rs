// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Browser (wasm32 + WebGPU) entry point for PID inference.
//!
//! Compiled only for `wasm32` under the `webgpu` feature. Exposes a single
//! `#[wasm_bindgen]` async function that takes a `.weights` byte blob (fetched
//! by JS) and a token sequence, builds the PID decoder, and returns the U-bin
//! logits at the final (DECIDE) position.
//!
//! Everything here mirrors the native `cli::rollout` setup, except:
//!   * weights are parsed from an in-memory slice (no `std::fs`), and
//!   * device init + buffer readback are awaited rather than blocked on.

use wasm_bindgen::prelude::*;

use pid::{Pid, PidConfig, BOS, DECIDE};
use pid::data::{
    dequantize_u, encode_effect_bin, encode_event, eval_step_schedule, pole_place_pi,
    velocity_pi_bin, Plant, PlantSpec,
};

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0 as u32
}

/// Install the panic hook so Rust panics surface as readable JS console errors.
/// Safe to call more than once.
#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Run one PID inference in the browser.
///
/// * `weights` — the raw bytes of a `.weights` container (same format the native
///   CLI loads from disk).
/// * `tokens`  — the input token sequence (B=1); the model must have been
///   trained with `block_size >= tokens.len()`.
///
/// Returns the `U_BINS` logits at the last position. The caller can argmax these
/// in JS to recover the predicted control bin.
#[wasm_bindgen]
pub async fn run_inference(weights: &[u8], tokens: &[u32]) -> Vec<f32> {
    console_error_panic_hook::set_once();

    let container = checkpoint::parse(weights);
    let cfg = PidConfig::from_json(&container.header["config"]);
    // Saved inference checkpoints store weights with role "" (see `Pid::save`).
    let init = container.by_role("");

    let decoder = Pid::new_async(cfg.clone(), 1, cfg.block_size, &init).await;
    decoder.logits_last_async(tokens).await
}

/// Convenience wrapper that returns the argmax U-bin instead of the full logit
/// vector, for callers that only want the decision.
#[wasm_bindgen]
pub async fn run_inference_argmax(weights: &[u8], tokens: &[u32]) -> u32 {
    let logits = run_inference(weights, tokens).await;
    argmax(&logits)
}

/// Run BOTH closed loops on the same plant + setpoint schedule and return the
/// time series as a JSON string for the React demo:
///   {
///     "t":[..], "setpoint":[..],
///     "model_y":[..],  "model_u":[..],  "model_mse": f,
///     "oracle_y":[..], "oracle_u":[..], "oracle_mse": f,
///     "tau": f, "gain": f, "steps": n
///   }
///
/// The model loop drives the plant entirely from the transformer's decisions
/// (event -> DECIDE -> effect fed back); the oracle loop uses that plant's
/// pole-placed velocity-PI. Both reuse the exact validated control code, so the
/// comparison is faithful (no re-implemented physics in JS).
#[wasm_bindgen]
pub async fn rollout_compare(weights: &[u8], tau: f32, gain: f32, steps: u32) -> String {
    console_error_panic_hook::set_once();
    let container = checkpoint::parse(weights);
    let cfg = PidConfig::from_json(&container.header["config"]);
    let init = container.by_role("");
    let dec = Pid::new_async(cfg.clone(), 1, cfg.block_size, &init).await;

    let spec = PlantSpec { tau, gain, disturbance: 0.0 };
    let block = cfg.block_size as usize;
    let n = steps as usize;

    // --- model closed loop (transformer controls the plant) ---
    let mut plant = Plant::new(spec);
    let mut ctx: Vec<u32> = vec![BOS];
    let (mut ts, mut sp_s) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut my, mut mu) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let mut m_sse = 0.0f32;
    for t in 0..n {
        let sp = eval_step_schedule(t);
        let measured = plant.y;
        let error = sp - measured;
        ctx.extend(encode_event(sp, measured, error));
        ctx.push(DECIDE);
        let start = ctx.len().saturating_sub(block);
        let u_bin = argmax(&dec.logits_last_async(&ctx[start..]).await);
        ctx.extend(encode_effect_bin(u_bin));
        let u = dequantize_u(u_bin);
        plant.step(u);
        ts.push(t as f32);
        sp_s.push(sp);
        my.push(measured);
        mu.push(u);
        m_sse += (sp - measured) * (sp - measured);
        if ctx.len() > block {
            let mut nc = vec![BOS];
            nc.extend_from_slice(&ctx[ctx.len() - (block - 1)..]);
            ctx = nc;
        }
    }

    // --- oracle closed loop (this plant's perfect velocity-PI) ---
    let (kp, ki) = pole_place_pi(&spec);
    let mut op = Plant::new(spec);
    let (mut oy, mut ou) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut u_prev, mut prev_error) = (0.0f32, 0.0f32);
    let mut o_sse = 0.0f32;
    for t in 0..n {
        let sp = eval_step_schedule(t);
        let measured = op.y;
        let error = sp - measured;
        let u = dequantize_u(velocity_pi_bin(kp, ki, u_prev, error, prev_error));
        op.step(u);
        oy.push(measured);
        ou.push(u);
        o_sse += (sp - measured) * (sp - measured);
        prev_error = error;
        u_prev = u;
    }

    let denom = n.max(1) as f32;
    serde_json::json!({
        "t": ts, "setpoint": sp_s,
        "model_y": my, "model_u": mu, "model_mse": m_sse / denom,
        "oracle_y": oy, "oracle_u": ou, "oracle_mse": o_sse / denom,
        "tau": tau, "gain": gain, "steps": n
    })
    .to_string()
}
