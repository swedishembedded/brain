// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! RRDBNet forward, recorded on [`vae::blocks::Builder`].
//!
//! The whole net is conv + LeakyReLU + channel concat + nearest-2x upsample, so
//! it needs **no new kernel**: `conv`/`upsample`/`add` come from the shared
//! block builder, and only `leaky_relu` and `concat2` are appended.
//!
//! Why the activation is dispatched rather than folded into `ConvSpec::act`:
//! `vision::blocks::Act` carries no slope, and its `fused_code` selector is
//! documented as correctness-critical (a code the fused WGSL does not branch on
//! silently becomes the identity). RRDBNet needs slope **0.2** on every
//! activation, so it dispatches `leaky_relu` explicitly instead of widening a
//! shared enum that yolo and depth depend on.
//!
//! The two residual scalings are the architecture, not a detail: each dense
//! block and each RRDB returns `x + 0.2 * f(x)`. Dropping either leaves a net
//! that still runs and produces a plausible-looking, wrong image — the same
//! failure shape as `docs/lessons.md` #16 — so `RESIDUAL_SCALE` is named once
//! and the parity test taps both block outputs.

use gpu_core::{f, DeviceBuffer, Gpu, Step};
use vae::blocks::{BlockNames, Builder, Tensors};

use crate::config::RrdbConfig;

/// `LeakyReLU(negative_slope=0.2)` — every activation in the reference
/// (`RRDBNet.lrelu`), including the ones between the upsample stages.
const LRELU_SLOPE: f32 = 0.2;

/// The residual weight on BOTH the dense block and the RRDB
/// (`return out * 0.2 + x` in the reference, twice).
const RESIDUAL_SCALE: f32 = 0.2;

const N_BLOCKS: usize = vae::blocks::NEXT_SLOT;
const K_LEAKY_RELU: usize = N_BLOCKS;
const K_CONCAT2: usize = N_BLOCKS + 1;
const K_SCALE_ADD: usize = N_BLOCKS + 2;

/// This model's kernel set: the shared block kernels verbatim (copied by
/// [`vae::blocks::kernels_with`], never restated) plus the three this net adds.
pub const KERNELS: [(&str, &str); N_BLOCKS + 3] = kernel_set();

const fn kernel_set() -> [(&'static str, &'static str); N_BLOCKS + 3] {
    let mut k = vae::blocks::kernels_with::<{ N_BLOCKS + 3 }>();
    k[K_LEAKY_RELU] = ("leaky_relu", kernels::LEAKY_RELU);
    k[K_CONCAT2] = ("concat2", kernels::CONCAT2);
    k[K_SCALE_ADD] = ("scale_add", kernels::SCALE_ADD);
    k
}

/// A built RRDBNet at one input size.
pub struct Rrdb {
    gpu: Gpu,
    cfg: RrdbConfig,
    steps: Vec<Step>,
    x_in: DeviceBuffer,
    y_out: DeviceBuffer,
    taps: Vec<(String, DeviceBuffer, usize)>,
    hw: (u32, u32),
    /// The one-element `[RESIDUAL_SCALE]` buffer every residual's `scale_add`
    /// binds. Held because the recorded steps reference it: dropping it would
    /// free device memory the graph still dispatches against.
    _scale: DeviceBuffer,
}

impl Rrdb {
    /// Record the graph for a `[1, in_channels, h, w]` input.
    pub fn new(gpu: Gpu, cfg: RrdbConfig, w: &Tensors, h: u32, wd: u32, taps_on: bool) -> Rrdb {
        // eps/groups are unused here (no GroupNorm in this net); the builder
        // takes them for the AutoencoderKL shape it also serves.
        // The residual weight, uploaded once and bound by every `scale_add`.
        let scale_buf = gpu.storage(1);
        gpu.write_f32(&scale_buf, &[RESIDUAL_SCALE]);

        let mut b = Builder::new(&gpu, w, 1e-6, 32, BlockNames::diffusers(), taps_on);

        let (f_, g) = (cfg.num_feat, cfg.num_grow_ch);
        let hw = (h as u64) * (wd as u64);
        let x_in = b.act((cfg.in_channels as u64) * hw);

        // fea = conv_first(x)
        let fea = b.conv("conv_first", cfg.in_channels, f_, 3, 1, h, wd, &x_in);
        b.tap("conv_first".into(), &fea, (f_ as u64 * hw) as u32);

        // trunk: num_block x RRDB, each three dense blocks with a scaled residual.
        let mut t = fea.clone();
        for i in 0..cfg.num_block {
            t = rrdb_block(&mut b, &format!("body.{i}"), f_, g, h, wd, &t, &scale_buf);
            if i == 0 {
                b.tap("body.0".into(), &t, (f_ as u64 * hw) as u32);
            }
        }
        // fea = fea + conv_body(trunk)
        let body = b.conv("conv_body", f_, f_, 3, 1, h, wd, &t);
        let mut cur = b.add((f_ as u64 * hw) as u32, &fea, &body);
        b.tap("body_out".into(), &cur, (f_ as u64 * hw) as u32);

        // Upsample: nearest-2x -> conv -> lrelu, once per doubling.
        let (mut ch, mut cw) = (h, wd);
        for i in 1..=cfg.scale.trailing_zeros() {
            let up = b.upsample(f_, ch, cw, &cur);
            ch *= 2;
            cw *= 2;
            let c = b.conv(&format!("conv_up{i}"), f_, f_, 3, 1, ch, cw, &up);
            cur = lrelu(&mut b, (f_ as u64) * (ch as u64) * (cw as u64), &c);
            b.tap(format!("up{i}"), &cur, (f_ as u64 * ch as u64 * cw as u64) as u32);
        }

        let hr = b.conv("conv_hr", f_, f_, 3, 1, ch, cw, &cur);
        let hr = lrelu(&mut b, (f_ as u64) * (ch as u64) * (cw as u64), &hr);
        let y_out = b.conv("conv_last", f_, cfg.out_channels, 3, 1, ch, cw, &hr);
        // The output is tapped too: `run` clamps to [0,1], so a ladder that read
        // only the returned pixels could not tell a correct value from a clipped
        // one, and the reference's own output is unclamped.
        b.tap("out".into(), &y_out, (cfg.out_channels as u64 * ch as u64 * cw as u64) as u32);

        let (steps, taps) = b.finish();
        Rrdb { gpu, cfg, steps, x_in, y_out, taps, hw: (ch, cw), _scale: scale_buf }
    }

    /// Output size `(h, w)` — the input scaled by [`RrdbConfig::scale`].
    pub fn out_hw(&self) -> (u32, u32) {
        self.hw
    }

    pub fn steps(&self) -> &[Step] {
        &self.steps
    }

    /// Run one image, CHW in `[0,1]`, and return CHW `[0,1]` at `scale`x.
    ///
    /// The reference clamps its output to `[0,1]` before writing pixels; a GAN
    /// upscaler overshoots at edges, so skipping the clamp is a visible ringing
    /// artefact rather than a rounding difference.
    pub fn run(&self, chw: &[f32]) -> Vec<f32> {
        let want = (self.cfg.in_channels as usize) * self.in_numel();
        assert_eq!(chw.len(), want, "rrdb: input is {} floats, expected {want}", chw.len());
        self.gpu.write_f32(&self.x_in, chw);
        self.gpu.submit(&[], &self.steps);
        self.gpu.poll_wait();
        let n = (self.cfg.out_channels as usize) * (self.hw.0 as usize) * (self.hw.1 as usize);
        self.gpu.read(&self.y_out, n).into_iter().map(|v| v.clamp(0.0, 1.0)).collect()
    }

    fn in_numel(&self) -> usize {
        let s = self.cfg.scale as usize;
        (self.hw.0 as usize / s) * (self.hw.1 as usize / s)
    }

    /// A recorded intermediate, by name — the parity ladder's rungs.
    pub fn read_tap(&self, name: &str) -> Vec<f32> {
        let (_, buf, len) =
            self.taps.iter().find(|(n, _, _)| n == name).unwrap_or_else(|| panic!("no tap `{name}`"));
        self.gpu.submit(&[], &self.steps);
        self.gpu.poll_wait();
        self.gpu.read(buf, *len)
    }

    pub fn tap_names(&self) -> Vec<String> {
        self.taps.iter().map(|(n, _, _)| n.clone()).collect()
    }
}

/// `y = leaky_relu(x, 0.2)`, out of place.
fn lrelu(b: &mut Builder, n: u64, x: &DeviceBuffer) -> DeviceBuffer {
    let y = b.act(n);
    let g = b.gpu();
    b.push_step(g.step(K_LEAKY_RELU, &[x, &y], &[n as u32, f(LRELU_SLOPE)], n as u32));
    y
}

/// `out = x + RESIDUAL_SCALE * fx`, out of place — the `out * 0.2 + x` that
/// closes both a dense block and an RRDB.
///
/// Two dispatches over existing kernels rather than a new one. `scale_add` is
/// the MoE gated accumulate (`acc = gate[0] * src` at `n_experts = 1`,
/// `e_idx = 0`, `accumulate = 0`), which is exactly a scalar multiply when the
/// scalar lives in a one-element buffer — the same use `crates/restore` makes
/// of it. `axpy` would be one dispatch, but it accumulates IN PLACE into a
/// buffer that must already hold `x`, and `Builder::act` hands back pooled
/// buffers with arbitrary contents, so it would need a copy first.
fn residual(b: &mut Builder, n: u64, scale: &DeviceBuffer, x: &DeviceBuffer, fx: &DeviceBuffer) -> DeviceBuffer {
    let t = b.act(n);
    let g = b.gpu();
    // `scale_add` Params: [seq_len, d_model, n_experts, e_idx, accumulate].
    b.push_step(g.step(K_SCALE_ADD, &[scale, fx, &t], &[1, n as u32, 1, 0, 0], n as u32));
    let y = b.add(n as u32, x, &t);
    b.free(n, t);
    y
}

/// Concatenate `[1, ca, h, w]` and `[1, cb, h, w]` along channels.
fn cat(b: &mut Builder, ca: u32, cb: u32, h: u32, w: u32, a: &DeviceBuffer, bb: &DeviceBuffer) -> DeviceBuffer {
    let hw = (h as u64) * (w as u64);
    let y = b.act(((ca + cb) as u64) * hw);
    let g = b.gpu();
    // concat2 params: (rows, a_cols, b_cols) over row-major [rows, cols]; with
    // NCHW and one image, a "row" is a channel of `hw` elements, so the concat
    // is over the channel axis exactly.
    // `concat2` Params: [N, Ca, Cb, H, W] — it decomposes the output index as
    // NCHW itself, so the concat is over the CHANNEL axis by construction.
    b.push_step(g.step(K_CONCAT2, &[a, bb, &y], &[1, ca, cb, h, w], ((ca + cb) as u64 * hw) as u32));
    y
}

/// `ResidualDenseBlock_5C`: five convs whose inputs accumulate every earlier
/// output, then `x + 0.2 * conv5(...)`.
#[allow(clippy::too_many_arguments)]
fn dense_block(
    b: &mut Builder,
    prefix: &str,
    f_: u32,
    g: u32,
    h: u32,
    w: u32,
    x: &DeviceBuffer,
    scale: &DeviceBuffer,
) -> DeviceBuffer {
    let hw = (h as u64) * (w as u64);
    // `acc` is the running concat: x, then x‖x1, then x‖x1‖x2, ...
    let mut acc = x.clone();
    let mut acc_c = f_;
    let mut last = x.clone();
    for c in 1..=5u32 {
        let cin = acc_c;
        let cout = if c == 5 { f_ } else { g };
        let y = b.conv(&format!("{prefix}.conv{c}"), cin, cout, 3, 1, h, w, &acc);
        // conv5 has NO activation in the reference — it feeds the residual.
        last = if c == 5 { y } else { lrelu(b, (cout as u64) * hw, &y) };
        if c < 5 {
            acc = cat(b, acc_c, cout, h, w, &acc, &last);
            acc_c += cout;
        }
    }
    residual(b, (f_ as u64) * hw, scale, x, &last)
}

/// `RRDB`: three dense blocks, then `x + 0.2 * out`.
#[allow(clippy::too_many_arguments)]
fn rrdb_block(
    b: &mut Builder,
    prefix: &str,
    f_: u32,
    g: u32,
    h: u32,
    w: u32,
    x: &DeviceBuffer,
    scale: &DeviceBuffer,
) -> DeviceBuffer {
    let mut cur = x.clone();
    for r in 1..=3u32 {
        cur = dense_block(b, &format!("{prefix}.rdb{r}"), f_, g, h, w, &cur, scale);
    }
    residual(b, (f_ as u64) * (h as u64) * (w as u64), scale, x, &cur)
}

#[cfg(test)]
mod tests {
    /// The slot consts index [`KERNELS`] positionally, so inserting one
    /// re-points every const after it — a mismatched kernel is wrong output,
    /// not a crash (`docs/kernel-checklist.md` §B).
    #[test]
    fn slot_constants_name_the_kernel_they_index() {
        for (slot, want) in [
            (super::K_LEAKY_RELU, "leaky_relu"),
            (super::K_CONCAT2, "concat2"),
            (super::K_SCALE_ADD, "scale_add"),
        ] {
            assert_eq!(super::KERNELS[slot].0, want, "slot {slot} is not '{want}'");
        }
        // And the shared block set still occupies the front, unshifted.
        assert_eq!(super::KERNELS[0].0, vae::blocks::KERNELS[0].0);
        assert_eq!(super::KERNELS[super::N_BLOCKS - 1].0, vae::blocks::KERNELS[super::N_BLOCKS - 1].0);
    }
}
