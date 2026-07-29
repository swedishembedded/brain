// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Backward (transpose) of rel_shift.wgsl. The forward map output→input is
// injective (a reindex of a padded grid), so each output scatters its grad to a
// unique input slot; inputs never referenced keep 0. Caller must zero `dx` first.
//   dx[r,ip,kp-1] = dy[r,i,j]   for kp != 0
// Same closed-form index as the forward. No atomics needed (no two outputs hit
// the same dx slot); the gradcheck test verifies this.

struct Params {
    rows: u32,
    q: u32,
    p: u32,
};

@group(0) @binding(0) var<uniform> pm: Params;
@group(0) @binding(1) var<storage, read>       dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.y * (nwg.x * 64u) + gid.x;
    let qp = pm.q * pm.p;
    if (idx >= pm.rows * qp) { return; }
    let r = idx / qp;
    let rem = idx % qp;
    let i = rem / pm.p;
    let j = rem % pm.p;
    let f = i * pm.p + j;
    let s = (f / pm.q + 1u) * pm.q + (f % pm.q);
    let ip = s / (pm.p + 1u);
    let kp = s % (pm.p + 1u);
    if (kp != 0u) {
        dx[r * qp + ip * pm.p + (kp - 1u)] = dy[idx];
    }
}
