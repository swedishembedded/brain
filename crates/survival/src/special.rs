// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! The special functions the tests of fit need: log-gamma and the regularized
//! upper incomplete gamma function, for chi-square tail probabilities.

/// `ln Gamma(x)` for `x > 0` (Lanczos, g = 7, nine coefficients; relative
/// error below 1e-14 over the range a chi-square test reaches).
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let s = G
        .iter()
        .enumerate()
        .skip(1)
        .fold(G[0], |acc, (i, &c)| acc + c / (x + i as f64));
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + s.ln()
}

/// `Q(a, x) = Gamma(a, x) / Gamma(a)`: a series below `x < a + 1`, Lentz's
/// continued fraction above (Numerical Recipes `gammq`).
pub fn gamma_q(a: f64, x: f64) -> f64 {
    assert!(a > 0.0 && x >= 0.0, "gamma_q({a}, {x})");
    if x == 0.0 {
        return 1.0;
    }
    let front = -x + a * x.ln() - ln_gamma(a);
    if x < a + 1.0 {
        let (mut ap, mut del, mut sum) = (a, 1.0 / a, 1.0 / a);
        for _ in 0..1000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        1.0 - sum * front.exp()
    } else {
        let tiny = 1e-300;
        let (mut b, mut c, mut d) = (x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a));
        let mut h = d;
        for i in 1..1000 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < tiny {
                d = tiny;
            }
            c = b + an / c;
            if c.abs() < tiny {
                c = tiny;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        front.exp() * h
    }
}

/// `P(X > x)` for `X ~ chi-square(df)`.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    gamma_q(0.5 * df, 0.5 * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against scipy.stats.chi2.sf.
    #[test]
    fn chi_square_tails() {
        let cases = [
            (3.0, 9.0, 0.964_294_972_685_089_1),
            (16.92, 9.0, 0.049_983_606_387_505_65),
            (40.0, 9.0, 7.598_525_229_464_264e-6),
            (0.5, 1.0, 0.479_500_122_186_953_37),
            (25.0, 4.0, 5.030_981_782_306_207_5e-5),
        ];
        for (x, df, want) in cases {
            let got = chi2_sf(x, df);
            assert!(
                (got - want).abs() <= 1e-10 + 1e-9 * want,
                "chi2_sf({x}, {df}) = {got}, want {want}"
            );
        }
    }
}
