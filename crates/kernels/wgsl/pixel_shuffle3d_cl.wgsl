// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  3D pixel shuffle (depth-to-space), CHANNELS-LAST
// @how   one thread per output element
// @opt   1
// @cpu   yes
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// `LinearPixelShuffleUpsample`'s rearrange (LTX diffusion video VAE decoder,
// `ltx_core...video_vae.transformer.layers`): `'b t h w (c p1 p2 p3) -> b (t
// p1) (h p2) (w p3) c'` - channel-OUTER group order `(c, p1=T, p2=H, p3=W)`,
// height-offset BEFORE width-offset. This is the SAME sub-order
// `vae::blocks3d`'s `space_to_depth3d`/`depth_to_space3d` use (confirmed by
// reading the einops pattern directly: both name their sub-axes `p1/p2/p3`
// tied to `(t p1)(h p2)(w p3)` in that order) - it is NOT the video VAE's
// OUTER pixel `patchify`/`unpatchify` convention (`ops.py`'s `(c p r q)`,
// width before height - see `crate::patchify`'s module doc for that
// documented trap). The two conventions must not be conflated; this kernel
// implements the FORMER, verified against `layers.py`'s own einops string,
// not assumed from precedent.
//
// The one genuine difference from `depth_to_space3d.wgsl`: that kernel is
// CHANNEL-FIRST (`[C,T,H,W]`, the VAE conv decoder's own layout); this
// decoder's NA transformer stack is CHANNELS-LAST throughout (`[T,H,W,C]`),
// so the buffer layout - not just the sub-order - genuinely differs, which
// is why this is a new kernel rather than a call-site reuse of the existing
// one (checked first, per this repo's own kernel-reuse checklist: layout
// mismatch, not a duplicate).
//
//   x : [T, H, W, Cin]              Cin = Cout * p1 * p2 * p3
//   y : [T*p1, H*p2, W*p3, Cout]    one invocation per OUTPUT element
//
//   y[t*p1+it, h*p2+ih, w*p3+iw, c] = x[t, h, w, ((c*p1+it)*p2+ih)*p3+iw]

struct Params {
    T: u32,
    H: u32,
    W: u32,
    Cout: u32,
    p1: u32,
    p2: u32,
    p3: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let gidx = gid.y * (nwg.x * 64u) + gid.x;
    let idx = gidx;
    let To = p.T * p.p1;
    let Ho = p.H * p.p2;
    let Wo = p.W * p.p3;
    let Cin = p.Cout * p.p1 * p.p2 * p.p3;
    let total = To * Ho * Wo * p.Cout;
    if (idx >= total) { return; }

    // Decode output coordinate (to, ho, wo, c) - c fastest.
    let c = idx % p.Cout;
    let t1 = idx / p.Cout;
    let wo = t1 % Wo;
    let t2 = t1 / Wo;
    let ho = t2 % Ho;
    let to = t2 / Ho;

    let iw = wo % p.p3;
    let w = wo / p.p3;
    let ih = ho % p.p2;
    let h = ho / p.p2;
    let it = to % p.p1;
    let t = to / p.p1;

    let cin = ((c * p.p1 + it) * p.p2 + ih) * p.p3 + iw;
    let x_idx = ((t * p.H + h) * p.W + w) * Cin + cin;
    y[idx] = x[x_idx];
}
