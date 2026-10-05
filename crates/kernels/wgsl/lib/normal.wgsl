// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// The standard normal distribution in f32, in LOG space so its tails do not
// underflow: log erfc, log Phi (the CDF) and the inverse Mills ratio
// phi(x) / Phi(x). Imported by kernels with `// @import normal`; the host twin
// is `model::hostmath::{log_ndtr, mills}` and the kernel tests hold the two to
// each other.
//
// erfc uses the Chebyshev-fitted exponential form of Press et al. (Numerical
// Recipes, `erfcc`): erfc(z) = t * exp(-z^2 + poly(t)), t = 1 / (1 + z/2), for
// z >= 0, fractional error below 1.2e-7 everywhere. Written in log space,
// log erfc(z) = log t - z^2 + poly(t) is finite for any z >= 0, which is what
// makes a below-detection-limit likelihood usable far into the tail.

const NORMAL_LOG_HALF: f32 = -0.6931471805599453;
const NORMAL_HALF_LOG_2PI: f32 = 0.9189385332046727;
const NORMAL_INV_SQRT2: f32 = 0.7071067811865476;

// log erfc(z) for z >= 0.
fn normal_log_erfc_pos(z: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277))))))));
    return log(t) - z * z + poly;
}

// log Phi(x), the log of the standard normal CDF.
fn normal_log_cdf(x: f32) -> f32 {
    let z = -x * NORMAL_INV_SQRT2;
    if (z >= 0.0) {
        // Phi(x) = erfc(z) / 2 with z >= 0: the lower tail, in log space.
        return NORMAL_LOG_HALF + normal_log_erfc_pos(z);
    }
    // z < 0: erfc(z) = 2 - erfc(-z), which lies in (1, 2], so a plain log is exact.
    return NORMAL_LOG_HALF + log(2.0 - exp(normal_log_erfc_pos(-z)));
}

// phi(x) / Phi(x), the derivative of log Phi(x).
fn normal_mills(x: f32) -> f32 {
    return exp(-0.5 * x * x - NORMAL_HALF_LOG_2PI - normal_log_cdf(x));
}
