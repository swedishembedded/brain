// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
//
// Swedish Embedded AB implements solutions for recovering accelerator-backed
// services from wedged and crashing device drivers for its clients. If your
// team needs expertise in supervised GPU execution and driver isolation then
// you can procure our services by sending an email to
// info@swedishembedded.com.

//! `brain-gpu-worker`: the child process a supervisor runs device work in.
//!
//! It owns at most one card's device for its whole life and answers the
//! supervisor's line protocol on stdin/stdout (`gpu_supervisor`'s worker
//! side): one request line `<id> <request>`, one answer line
//! `<id> ok <payload>` or `<id> err <message>`. Every driver call happens in
//! this process, so a call that wedges is ended by the supervisor killing
//! this process, and the kernel reclaims the device context with it.
//!
//! | request | effect | answer payload |
//! |---|---|---|
//! | `open gpu<i>` | create the device on card `i` and compile the probe kernel | `{"device": name}` |
//! | `probe gpu<i>` | `open`, then one probe dispatch | `{"device": name, "sum": [...]}` |
//! | `run probe <params>` | one probe dispatch on the open device | `{"sum": [...]}` |
//!
//! The probe dispatch adds two four-element vectors on the device and checks
//! the result, so a pass means the card created a device, compiled a kernel,
//! ran it and returned correct numbers. An answer starting `no-device` means
//! the host has no such card; every other `err` is the device failing.

use std::io::{BufRead, Write};

use backend_wgpu::WgpuBackend;

/// `dst += src` on these inputs.
const PROBE_SRC: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
const PROBE_DST: [f32; 4] = [10.0, 20.0, 30.0, 40.0];

struct Device {
    backend: WgpuBackend,
    name: String,
}

fn main() {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    let mut device: Option<Device> = None;
    for line in stdin.lock().lines() {
        // The supervisor closing its end is the order to exit.
        let Ok(line) = line else { return };
        let (id, request) = line.split_once(' ').unwrap_or((line.as_str(), ""));
        let answer = match handle(request, &mut device) {
            Ok(payload) => format!("{id} ok {payload}"),
            Err(message) => format!("{id} err {}", message.replace(['\n', '\r'], " ")),
        };
        if writeln!(stdout, "{answer}").and_then(|()| stdout.flush()).is_err() {
            return;
        }
    }
}

fn handle(request: &str, device: &mut Option<Device>) -> Result<String, String> {
    let mut words = request.splitn(3, ' ');
    match (words.next(), words.next(), words.next()) {
        (Some("open"), Some(label), None) => {
            let d = open(label)?;
            let payload = serde_json::json!({ "device": d.name }).to_string();
            *device = Some(d);
            Ok(payload)
        }
        (Some("probe"), Some(label), None) => {
            let d = open(label)?;
            let sum = dispatch(&d)?;
            Ok(serde_json::json!({ "device": d.name, "sum": sum }).to_string())
        }
        (Some("run"), Some("probe"), _) => {
            let d = device.as_ref().ok_or("no device is open; send `open gpu<i>` first")?;
            Ok(serde_json::json!({ "sum": dispatch(d)? }).to_string())
        }
        (Some("run"), Some(action), _) => Err(format!("this worker has no action {action:?}")),
        _ => Err(format!("unknown request {request:?}")),
    }
}

/// The card `gpu<i>` names, in the same enumeration order the device registry uses.
fn open(label: &str) -> Result<Device, String> {
    let index: usize = label.strip_prefix("gpu").and_then(|i| i.parse().ok()).ok_or_else(|| format!("{label:?} is not a GPU (gpu<i>)"))?;
    let gpus = backend_wgpu::enumerate_gpus();
    let identity = gpus.get(index).ok_or_else(|| format!("no-device: {label} is not among the {} GPU(s) this host enumerates", gpus.len()))?;
    let backend = WgpuBackend::new_on(&[("add", kernels::ADD)], identity);
    Ok(Device { backend, name: identity.name.clone() })
}

fn dispatch(d: &Device) -> Result<Vec<f32>, String> {
    let b = &d.backend;
    let src = b.storage_init("probe_src", &PROBE_SRC);
    let dst = b.storage_init("probe_dst", &PROBE_DST);
    let step = b.step(0, &[&*src, &*dst], &[PROBE_SRC.len() as u32], PROBE_SRC.len() as u32);
    b.submit(&[], &[step]);
    let got = b.read(&dst, PROBE_DST.len());
    let want: Vec<f32> = PROBE_SRC.iter().zip(PROBE_DST).map(|(s, d)| s + d).collect();
    if got != want {
        return Err(format!("{}: the probe dispatch returned {got:?}, expected {want:?}", d.name));
    }
    Ok(got)
}
