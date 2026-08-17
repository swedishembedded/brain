// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Host-side conditioning math for the DIAMOND denoiser. All of this is
//! tiny (256-dim vectors, ~44 small linears per denoise step), so it runs on
//! the host and the results are written into device buffers:
//!
//! - EDM conditioners (with sigma-offset-noise folded in, exactly as
//!   `denoiser.py::compute_conditioners`):
//!   s' = sqrt(s^2 + so^2); c_in = 1/sqrt(s'^2 + sd^2);
//!   c_skip = sd^2/(s'^2 + sd^2); c_out = s'*sqrt(c_skip);
//!   c_noise = ln(s')/4
//! - Fourier noise embedding: f = 2*pi*c_noise*W (W = noise_emb.weight[1,C/2])
//!   -> [cos f, sin f]  (C floats)
//! - Action embedding: Embedding(num_actions, C/nsc) per context action,
//!   flattened (C floats), ADDED to the Fourier embedding
//! - cond MLP: Linear -> SiLU -> Linear (cond_proj.0 / cond_proj.2)
//! - Per AdaGroupNorm site: gb = linear(cond) giving [scale||shift]; the
//!   device `gn_apply` consumes gamma||beta, so gamma = 1 + scale, beta =
//!   shift are what gets written.

/// EDM conditioners for one sigma.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conditioners {
    pub c_in: f32,
    pub c_out: f32,
    pub c_skip: f32,
    pub c_noise: f32,
}

pub fn conditioners(sigma: f32, sigma_data: f32, sigma_offset_noise: f32) -> Conditioners {
    let s2 = sigma * sigma + sigma_offset_noise * sigma_offset_noise;
    let s = s2.sqrt();
    let sd2 = sigma_data * sigma_data;
    let c_in = 1.0 / (s2 + sd2).sqrt();
    let c_skip = sd2 / (s2 + sd2);
    let c_out = s * c_skip.sqrt();
    let c_noise = s.ln() / 4.0;
    Conditioners { c_in, c_out, c_skip, c_noise }
}

/// Karras sigma schedule + trailing 0 (`build_sigmas` in the reference).
pub fn build_sigmas(num_steps: u32, sigma_min: f32, sigma_max: f32, rho: f32) -> Vec<f32> {
    let n = num_steps.max(1);
    let min_inv = sigma_min.powf(1.0 / rho);
    let max_inv = sigma_max.powf(1.0 / rho);
    let mut out = Vec::with_capacity(n as usize + 1);
    for i in 0..n {
        let l = if n == 1 { 0.0 } else { i as f32 / (n - 1) as f32 };
        out.push((max_inv + l * (min_inv - max_inv)).powf(rho));
    }
    out.push(0.0);
    out
}


/// Row-major Linear: y[o] = b[o] + sum_i w[o*ni + i] * x[i].
pub fn linear(w: &[f32], b: &[f32], x: &[f32], no: usize, ni: usize) -> Vec<f32> {
    assert_eq!(w.len(), no * ni);
    assert_eq!(b.len(), no);
    assert_eq!(x.len(), ni);
    let mut y = vec![0.0f32; no];
    for o in 0..no {
        let mut acc = b[o];
        let row = &w[o * ni..(o + 1) * ni];
        for i in 0..ni {
            acc += row[i] * x[i];
        }
        y[o] = acc;
    }
    y
}

/// Host weights of the conditioning path (kept off-device).
pub struct CondNet {
    pub cond_channels: usize,
    pub num_steps_conditioning: usize,
    /// noise_emb.weight, [cond_channels/2].
    pub fourier_w: Vec<f32>,
    /// act_emb.0.weight, [num_actions, cond_channels/nsc].
    pub act_emb: Vec<f32>,
    pub num_actions: usize,
    pub mlp0_w: Vec<f32>,
    pub mlp0_b: Vec<f32>,
    pub mlp2_w: Vec<f32>,
    pub mlp2_b: Vec<f32>,
}

impl CondNet {
    /// cond = mlp2(model::hostmath::silu(mlp0(fourier(c_noise) + act_flat))), exactly
    /// `InnerModel.forward`'s cond path.
    pub fn cond(&self, c_noise: f32, actions: &[u32]) -> Vec<f32> {
        let cc = self.cond_channels;
        assert_eq!(actions.len(), self.num_steps_conditioning);
        let half = cc / 2;
        let mut e = vec![0.0f32; cc];
        for i in 0..half {
            let f = 2.0 * std::f32::consts::PI * c_noise * self.fourier_w[i];
            e[i] = f.cos();
            e[half + i] = f.sin();
        }
        let per = cc / self.num_steps_conditioning;
        for (t, &a) in actions.iter().enumerate() {
            assert!((a as usize) < self.num_actions, "action {a} out of range");
            let row = &self.act_emb[a as usize * per..(a as usize + 1) * per];
            for i in 0..per {
                e[t * per + i] += row[i];
            }
        }
        let h: Vec<f32> =
            linear(&self.mlp0_w, &self.mlp0_b, &e, cc, cc).into_iter().map(model::hostmath::silu).collect();
        linear(&self.mlp2_w, &self.mlp2_b, &h, cc, cc)
    }
}

/// One AdaGroupNorm site: host linear producing the device gamma/beta buffer.
pub struct AdaGnSite {
    /// linear.weight [2C, cond_channels] row-major, linear.bias [2C].
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub c: usize,
}

impl AdaGnSite {
    /// gamma||beta for `gn_apply`: gamma = 1 + scale, beta = shift
    /// (reference chunks linear(cond) into scale then shift).
    pub fn gb(&self, cond: &[f32]) -> Vec<f32> {
        let sb = linear(&self.w, &self.b, cond, 2 * self.c, cond.len());
        let mut gb = sb;
        for g in gb[..self.c].iter_mut() {
            *g += 1.0;
        }
        gb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cond_edm_conditioners_match_hand_calc() {
        // sigma=1, sd=0.5, so=0.3: s'^2 = 1.09, s' = 1.0440307,
        // c_in = 1/sqrt(1.34) = 0.8638684, c_skip = 0.25/1.34 = 0.1865672,
        // c_out = 1.0440307*sqrt(0.1865672) = 0.4509526,
        // c_noise = ln(1.0440307)/4 = 0.0107722.
        let c = conditioners(1.0, 0.5, 0.3);
        assert!((c.c_in - 0.8638684).abs() < 1e-6, "{}", c.c_in);
        assert!((c.c_skip - 0.1865672).abs() < 1e-6, "{}", c.c_skip);
        assert!((c.c_out - 0.4509526).abs() < 1e-6, "{}", c.c_out);
        assert!((c.c_noise - 0.0107722).abs() < 1e-6, "{}", c.c_noise);
    }

    #[test]
    fn cond_sigmas_karras_endpoints_and_monotonic() {
        let s = build_sigmas(3, 2e-3, 5.0, 7.0);
        assert_eq!(s.len(), 4);
        assert!((s[0] - 5.0).abs() < 1e-5);
        assert!((s[2] - 2e-3).abs() < 1e-6);
        assert_eq!(s[3], 0.0);
        assert!(s[0] > s[1] && s[1] > s[2]);
    }

    #[test]
    fn cond_linear_and_adagn_gb_hand_calc() {
        // 2x2 linear: w=[[1,2],[3,4]], b=[0.5,-0.5], x=[1,-1] -> [-0.5,-1.5].
        let y = linear(&[1.0, 2.0, 3.0, 4.0], &[0.5, -0.5], &[1.0, -1.0], 2, 2);
        assert_eq!(y, vec![-0.5, -1.5]);
        // AdaGN C=1: sb = [scale, shift] = [-0.5,-1.5] -> gb = [0.5, -1.5].
        let site = AdaGnSite { w: vec![1.0, 2.0, 3.0, 4.0], b: vec![0.5, -0.5], c: 1 };
        assert_eq!(site.gb(&[1.0, -1.0]), vec![0.5, -1.5]);
    }

    #[test]
    fn cond_fourier_plus_act_path_shapes_and_determinism() {
        let cc = 8;
        let net = CondNet {
            cond_channels: cc,
            num_steps_conditioning: 4,
            fourier_w: vec![0.1, 0.2, 0.3, 0.4],
            act_emb: (0..6).map(|i| i as f32 * 0.1).collect(),
            num_actions: 3,
            mlp0_w: (0..cc * cc).map(|i| (i as f32 * 0.01).sin() * 0.1).collect(),
            mlp0_b: vec![0.0; cc],
            mlp2_w: (0..cc * cc).map(|i| (i as f32 * 0.02).cos() * 0.1).collect(),
            mlp2_b: vec![0.0; cc],
        };
        let a = net.cond(0.5, &[0, 1, 2, 0]);
        let b = net.cond(0.5, &[0, 1, 2, 0]);
        assert_eq!(a, b);
        assert_eq!(a.len(), cc);
        // Different actions change the embedding path.
        let c = net.cond(0.5, &[1, 1, 2, 0]);
        assert_ne!(a, c);
    }
}
