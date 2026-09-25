// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// Camera models on the device: the WGSL twin of `crates/camera` (pinhole,
// OpenCV Brown-Conrady with rational radial, tangential and thin prism terms,
// Kannala-Brandt fisheye, equirectangular), in f32. Imported by kernels with
// `// @import camera`; `crates/camera/tests` and the splat kernel tests hold
// the two implementations to each other.
//
// Conventions: camera frame +X right, +Y down, +Z forward; continuous pixel
// coordinates with pixel (i, j)'s centre at (i + 0.5, j + 0.5); unprojection
// returns UNIT rays. `bound` is the model's validity radius (normalized r for
// Brown, theta for the fisheye; negative = unbounded), computed on the host
// by `camera::Intrinsics::valid_radius`: past it a polynomial lens folds
// geometry back into the frame, so nothing past it projects.

struct Lens {
    code: u32,   // 0 pinhole, 1 Brown, 2 fisheye, 3 equirect
    bound: f32,
    pad0: u32,
    pad1: u32,
    f: vec4<f32>,  // fx, fy, cx, cy
    ka: vec4<f32>, // Brown k1..k4 | fisheye k1..k4
    kb: vec4<f32>, // Brown k5, k6, p1, p2
    kc: vec4<f32>, // Brown s1..s4
};

// A projection: the pixel, whether the lens images the direction at all, and
// the Jacobian d(pixel)/d(direction) as two rows.
struct LensProj {
    uv: vec2<f32>,
    ok: f32,
    j0: vec3<f32>,
    j1: vec3<f32>,
};

// Brown distortion of normalized (x, y): the distorted point and its 2x2
// Jacobian, columns-by-row as (d xd/dx, d xd/dy, d yd/dx, d yd/dy).
struct Brown {
    xd: vec2<f32>,
    j: vec4<f32>,
};

fn lens_brown(l: Lens, x: f32, y: f32) -> Brown {
    let r2 = x * x + y * y;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let num = 1.0 + l.ka.x * r2 + l.ka.y * r4 + l.ka.z * r6;
    let den = 1.0 + l.ka.w * r2 + l.kb.x * r4 + l.kb.y * r6;
    let rad = num / den;
    let dnum = l.ka.x + 2.0 * l.ka.y * r2 + 3.0 * l.ka.z * r4;
    let dden = l.ka.w + 2.0 * l.kb.x * r2 + 3.0 * l.kb.y * r4;
    let drad = (dnum * den - num * dden) / (den * den);
    let p1 = l.kb.z;
    let p2 = l.kb.w;
    let s = l.kc;
    var b: Brown;
    b.xd = vec2<f32>(
        x * rad + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + s.x * r2 + s.y * r4,
        y * rad + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + s.z * r2 + s.w * r4);
    b.j = vec4<f32>(
        rad + 2.0 * x * x * drad + 2.0 * p1 * y + 6.0 * p2 * x + 2.0 * s.x * x + 4.0 * s.y * r2 * x,
        2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y + 2.0 * s.x * y + 4.0 * s.y * r2 * y,
        2.0 * x * y * drad + 2.0 * p1 * x + 2.0 * p2 * y + 2.0 * s.z * x + 4.0 * s.w * r2 * x,
        rad + 2.0 * y * y * drad + 6.0 * p1 * y + 2.0 * p2 * x + 2.0 * s.z * y + 4.0 * s.w * r2 * y);
    return b;
}

fn lens_fisheye_poly(l: Lens, t: f32) -> vec2<f32> {
    // (theta_d, d theta_d / d theta)
    let t2 = t * t;
    let k = l.ka;
    let td = t * (1.0 + t2 * (k.x + t2 * (k.y + t2 * (k.z + t2 * k.w))));
    let dtd = 1.0 + t2 * (3.0 * k.x + t2 * (5.0 * k.y + t2 * (7.0 * k.z + t2 * 9.0 * k.w)));
    return vec2<f32>(td, dtd);
}

// Project camera-frame point or direction `d`; ok = 0 where the lens does not
// image it.
fn lens_project(l: Lens, d: vec3<f32>) -> LensProj {
    var o: LensProj;
    o.ok = 0.0;
    o.uv = vec2<f32>(0.0, 0.0);
    o.j0 = vec3<f32>(0.0, 0.0, 0.0);
    o.j1 = vec3<f32>(0.0, 0.0, 0.0);
    let fx = l.f.x;
    let fy = l.f.y;
    if (l.code <= 1u) {
        if (d.z <= 1e-12) { return o; }
        let iz = 1.0 / d.z;
        let x = d.x * iz;
        let y = d.y * iz;
        var xd = vec2<f32>(x, y);
        var j = vec4<f32>(1.0, 0.0, 0.0, 1.0);
        if (l.code == 1u) {
            if (l.bound >= 0.0 && x * x + y * y > l.bound * l.bound) { return o; }
            let b = lens_brown(l, x, y);
            xd = b.xd;
            j = b.j;
        }
        // d(x, y)/d(X, Y, Z) = [[iz, 0, -x iz], [0, iz, -y iz]]
        o.uv = vec2<f32>(fx * xd.x + l.f.z, fy * xd.y + l.f.w);
        o.j0 = fx * vec3<f32>(j.x * iz, j.y * iz, -(j.x * x + j.y * y) * iz);
        o.j1 = fy * vec3<f32>(j.z * iz, j.w * iz, -(j.z * x + j.w * y) * iz);
        o.ok = 1.0;
        return o;
    }
    if (l.code == 2u) {
        let a2 = d.x * d.x + d.y * d.y;
        let a = sqrt(a2);
        let rho2 = a2 + d.z * d.z;
        if (rho2 <= 1e-24) { return o; }
        let theta = atan2(a, d.z);
        if (l.bound >= 0.0 && theta > l.bound) { return o; }
        let tp = lens_fisheye_poly(l, theta);
        var s = 0.0;
        var c1 = 0.0;
        var dsdz = 0.0;
        if (a > 1e-6 * abs(d.z)) {
            s = tp.x / a;
            c1 = (tp.y * d.z / rho2 - tp.x / a) / a2;
            dsdz = -tp.y / rho2;
        } else {
            if (d.z < 0.0) { return o; }
            let iz = 1.0 / d.z;
            s = iz * (1.0 + (l.ka.x - 1.0 / 3.0) * a2 * iz * iz);
            c1 = 2.0 * (l.ka.x - 1.0 / 3.0) * iz * iz * iz;
            dsdz = -iz * iz;
        }
        o.uv = vec2<f32>(fx * s * d.x + l.f.z, fy * s * d.y + l.f.w);
        o.j0 = fx * vec3<f32>(s + d.x * d.x * c1, d.x * d.y * c1, d.x * dsdz);
        o.j1 = fy * vec3<f32>(d.x * d.y * c1, s + d.y * d.y * c1, d.y * dsdz);
        o.ok = 1.0;
        return o;
    }
    // equirect
    let h2 = d.x * d.x + d.z * d.z;
    if (h2 <= 1e-24) { return o; }
    let h = sqrt(h2);
    let rho2 = h2 + d.y * d.y;
    let lon = atan2(d.x, d.z);
    let lat = atan2(d.y, h);
    o.uv = vec2<f32>(l.f.z + fx * lon, l.f.w + fy * lat);
    o.j0 = vec3<f32>(fx * d.z / h2, 0.0, -fx * d.x / h2);
    o.j1 = vec3<f32>(-fy * d.y * d.x / (h * rho2), fy * h / rho2, -fy * d.y * d.z / (h * rho2));
    o.ok = 1.0;
    return o;
}

// The unit ray through continuous pixel coordinate `px`; w = 1 where the lens
// has a ray there, 0 where it does not.
fn lens_unproject(l: Lens, px: vec2<f32>) -> vec4<f32> {
    let xd = (px.x - l.f.z) / l.f.x;
    let yd = (px.y - l.f.w) / l.f.y;
    if (l.code == 0u) {
        return vec4<f32>(normalize(vec3<f32>(xd, yd, 1.0)), 1.0);
    }
    if (l.code == 1u) {
        var x = xd;
        var y = yd;
        for (var it = 0u; it < 20u; it = it + 1u) {
            let b = lens_brown(l, x, y);
            let ex = b.xd.x - xd;
            let ey = b.xd.y - yd;
            let det = b.j.x * b.j.w - b.j.y * b.j.z;
            if (abs(det) < 1e-12) { return vec4<f32>(0.0, 0.0, 1.0, 0.0); }
            var dx = (b.j.w * ex - b.j.y * ey) / det;
            var dy = (-b.j.z * ex + b.j.x * ey) / det;
            // damped so a step never leaves the valid disc
            var step = 1.0;
            for (var h = 0u; h < 12u; h = h + 1u) {
                let nx = x - step * dx;
                let ny = y - step * dy;
                if (l.bound < 0.0 || nx * nx + ny * ny <= l.bound * l.bound) { break; }
                step = step * 0.5;
            }
            x = x - step * dx;
            y = y - step * dy;
        }
        let b = lens_brown(l, x, y);
        let err = abs(b.xd.x - xd) + abs(b.xd.y - yd);
        var ok = 1.0;
        if (err > 1e-4 * max(1.0, abs(xd) + abs(yd))) { ok = 0.0; }
        if (l.bound >= 0.0 && x * x + y * y > l.bound * l.bound * 1.0001) { ok = 0.0; }
        return vec4<f32>(normalize(vec3<f32>(x, y, 1.0)), ok);
    }
    if (l.code == 2u) {
        let rd = sqrt(xd * xd + yd * yd);
        var tmax = 3.14159265;
        if (l.bound >= 0.0) { tmax = l.bound; }
        var t = min(rd, tmax);
        for (var it = 0u; it < 20u; it = it + 1u) {
            let tp = lens_fisheye_poly(l, t);
            if (tp.y <= 0.0) { return vec4<f32>(0.0, 0.0, 1.0, 0.0); }
            t = clamp(t - (tp.x - rd) / tp.y, 0.0, tmax);
        }
        let back = lens_fisheye_poly(l, t).x;
        var ok = 1.0;
        if (abs(back - rd) > 1e-4 * max(1.0, rd)) { ok = 0.0; }
        if (rd < 1e-12) { return vec4<f32>(0.0, 0.0, 1.0, ok); }
        let st = sin(t);
        return vec4<f32>(st * xd / rd, st * yd / rd, cos(t), ok);
    }
    // equirect: xd is the longitude, yd the latitude
    var ok = 1.0;
    if (abs(yd) > 1.57079633 || abs(xd) > 3.14159265) { ok = 0.0; }
    let cl = cos(yd);
    return vec4<f32>(cl * sin(xd), sin(yd), cl * cos(xd), ok);
}

// Pixels per radian where direction `d` images: the local sampling rate, the
// geometric mean of the projection Jacobian's singular values across the ray.
fn lens_pixels_per_radian(l: Lens, d: vec3<f32>) -> f32 {
    let p = lens_project(l, d);
    if (p.ok == 0.0) { return 0.0; }
    let r = length(d);
    let n = d / r;
    var a = vec3<f32>(1.0, 0.0, 0.0);
    if (abs(n.x) >= 0.9) { a = vec3<f32>(0.0, 1.0, 0.0); }
    let e1 = normalize(cross(n, a));
    let e2 = cross(n, e1);
    let c1 = vec2<f32>(dot(p.j0, e1), dot(p.j1, e1)) * r;
    let c2 = vec2<f32>(dot(p.j0, e2), dot(p.j1, e2)) * r;
    return sqrt(abs(c1.x * c2.y - c1.y * c2.x));
}

// d(pixel)/d(calibration) at camera-frame direction `d`, one column per
// parameter in `camera::Intrinsics::params` order: fx, fy, cx, cy, then the
// lens's coefficients (Brown k1..k6 p1 p2 s1..s4; fisheye k1..k4). Unused
// columns are zero. Assumes `lens_project(l, d).ok`.
fn lens_param_jac(l: Lens, d: vec3<f32>) -> array<vec2<f32>, 16> {
    var jp: array<vec2<f32>, 16>;
    for (var k = 0u; k < 16u; k = k + 1u) { jp[k] = vec2<f32>(0.0, 0.0); }
    let fx = l.f.x;
    let fy = l.f.y;
    jp[2] = vec2<f32>(1.0, 0.0);
    jp[3] = vec2<f32>(0.0, 1.0);
    if (l.code <= 1u) {
        let x = d.x / d.z;
        let y = d.y / d.z;
        if (l.code == 0u) {
            jp[0] = vec2<f32>(x, 0.0);
            jp[1] = vec2<f32>(0.0, y);
            return jp;
        }
        let b = lens_brown(l, x, y);
        jp[0] = vec2<f32>(b.xd.x, 0.0);
        jp[1] = vec2<f32>(0.0, b.xd.y);
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let num = 1.0 + l.ka.x * r2 + l.ka.y * r4 + l.ka.z * r6;
        let den = 1.0 + l.ka.w * r2 + l.kb.x * r4 + l.kb.y * r6;
        let dd = num / (den * den);
        jp[4] = vec2<f32>(fx * x * r2 / den, fy * y * r2 / den);
        jp[5] = vec2<f32>(fx * x * r4 / den, fy * y * r4 / den);
        jp[6] = vec2<f32>(fx * x * r6 / den, fy * y * r6 / den);
        jp[7] = vec2<f32>(-fx * x * r2 * dd, -fy * y * r2 * dd);
        jp[8] = vec2<f32>(-fx * x * r4 * dd, -fy * y * r4 * dd);
        jp[9] = vec2<f32>(-fx * x * r6 * dd, -fy * y * r6 * dd);
        jp[10] = vec2<f32>(fx * 2.0 * x * y, fy * (r2 + 2.0 * y * y));
        jp[11] = vec2<f32>(fx * (r2 + 2.0 * x * x), fy * 2.0 * x * y);
        jp[12] = vec2<f32>(fx * r2, 0.0);
        jp[13] = vec2<f32>(fx * r4, 0.0);
        jp[14] = vec2<f32>(0.0, fy * r2);
        jp[15] = vec2<f32>(0.0, fy * r4);
        return jp;
    }
    if (l.code == 2u) {
        let a = sqrt(d.x * d.x + d.y * d.y);
        let theta = atan2(a, d.z);
        let tp = lens_fisheye_poly(l, theta);
        var ux = 0.0;
        var uy = 0.0;
        if (a > 1e-12) {
            ux = d.x / a;
            uy = d.y / a;
        }
        jp[0] = vec2<f32>(tp.x * ux, 0.0);
        jp[1] = vec2<f32>(0.0, tp.x * uy);
        let t2 = theta * theta;
        var tpow = theta * t2;
        for (var k = 0u; k < 4u; k = k + 1u) {
            jp[4u + k] = vec2<f32>(fx * tpow * ux, fy * tpow * uy);
            tpow = tpow * t2;
        }
        return jp;
    }
    let h = sqrt(d.x * d.x + d.z * d.z);
    jp[0] = vec2<f32>(atan2(d.x, d.z), 0.0);
    jp[1] = vec2<f32>(0.0, atan2(d.y, h));
    return jp;
}

// How the unit ray through a fixed pixel moves with parameter column `g`
// (= d(pixel)/d(param) at that ray): the implicit derivative of
// project(dir(theta); theta) = pixel under |dir| = 1, i.e. the solution of
// [J; d^T] x = [-g; 0].
fn lens_ray_param_col(p: LensProj, d: vec3<f32>, g: vec2<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(vec3<f32>(p.j0.x, p.j1.x, d.x), vec3<f32>(p.j0.y, p.j1.y, d.y), vec3<f32>(p.j0.z, p.j1.z, d.z));
    let det = determinant(m);
    if (abs(det) < 1e-20) { return vec3<f32>(0.0, 0.0, 0.0); }
    // Cramer's rule on the columns of m
    let b = vec3<f32>(-g.x, -g.y, 0.0);
    let c0 = m[0];
    let c1 = m[1];
    let c2 = m[2];
    return vec3<f32>(
        determinant(mat3x3<f32>(b, c1, c2)),
        determinant(mat3x3<f32>(c0, b, c2)),
        determinant(mat3x3<f32>(c0, c1, b))) / det;
}
