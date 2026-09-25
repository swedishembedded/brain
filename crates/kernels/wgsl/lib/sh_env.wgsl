// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Real spherical harmonics of any degree up to 8, for the environment a
// splat scene is seen against (`splat::env`, whose host evaluation is the
// same formula in f64). Index l*l + l + m; Y_l^0 = K_l^0 Q_l^0(z),
// Y_l^{+m} = sqrt(2) K_l^m Q_l^m(z) Re (x + i y)^m, Y_l^{-m} with Im, where
// Q_l^m = P_l^m / (1 - z^2)^(m/2) is carried by the recurrences
//   Q_m^m = (-1)^m (2m - 1)!!,   Q_{m+1}^m = (2m + 1) z Q_m^m,
//   Q_l^m = ((2l - 1) z Q_{l-1}^m - (l + m - 1) Q_{l-2}^m) / (l - m),
// and K_l^m = sqrt((2l + 1) / 4pi (l - m)! / (l + m)!). The basis is
// orthonormal over the sphere.

const SH_ENV_MAX_COEFFS: u32 = 81u;

fn sh_env_basis(d: vec3<f32>, degree: u32) -> array<f32, 81> {
    var y: array<f32, 81>;
    for (var k = 0u; k < SH_ENV_MAX_COEFFS; k = k + 1u) { y[k] = 0.0; }
    let lmax = min(degree, 8u);
    var cm = 1.0;
    var sm = 0.0;
    var qmm = 1.0;
    for (var m = 0u; m <= lmax; m = m + 1u) {
        if (m > 0u) {
            let c2 = d.x * cm - d.y * sm;
            sm = d.x * sm + d.y * cm;
            cm = c2;
            qmm = -qmm * f32(2u * m - 1u);
        }
        var q2 = 0.0; // Q_{l-2}^m
        var q1 = 0.0; // Q_{l-1}^m
        for (var l = m; l <= lmax; l = l + 1u) {
            var q = qmm;
            if (l == m + 1u) {
                q = f32(2u * m + 1u) * d.z * qmm;
            } else if (l > m + 1u) {
                q = (f32(2u * l - 1u) * d.z * q1 - f32(l + m - 1u) * q2) / f32(l - m);
            }
            q2 = q1;
            q1 = q;
            var ratio = 1.0;
            for (var j = l - m + 1u; j <= l + m; j = j + 1u) { ratio = ratio / f32(j); }
            let k = sqrt(f32(2u * l + 1u) / 12.566370614359172 * ratio);
            let at = l * l + l;
            if (m == 0u) {
                y[at] = k * q;
            } else {
                y[at + m] = 1.4142135623730951 * k * q * cm;
                y[at - m] = 1.4142135623730951 * k * q * sm;
            }
        }
    }
    return y;
}
