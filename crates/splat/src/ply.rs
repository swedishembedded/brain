// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Inria 3DGS `.ply` reader/writer (binary little-endian).
//!
//! On-disk fields per vertex (all float32): `x y z  nx ny nz  f_dc_0..2
//! [f_rest_*]  opacity  scale_0..2  rot_0..3`, where opacity is a pre-sigmoid
//! logit, scales are log-space, quats wxyz un-normalized, and colors are SH
//! coefficients (`RGB = 0.282095*dc + 0.5`). The reader applies the
//! activations so [`Splats`] is always render-ready; the writer inverts them.

use std::io::{BufWriter, Write};

use crate::types::Splats;

pub const SH_C0: f32 = 0.282_094_8;

/// Parse a binary little-endian Inria 3DGS PLY.
pub fn read(path: &str) -> Result<Splats, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    parse(&bytes)
}

pub fn parse(bytes: &[u8]) -> Result<Splats, String> {
    // ---- header ----
    let hdr_end = find_header_end(bytes).ok_or("no end_header found")?;
    let header = std::str::from_utf8(&bytes[..hdr_end]).map_err(|_| "header not utf8")?;
    let mut n: usize = 0;
    let mut props: Vec<String> = Vec::new();
    let mut in_vertex = false;
    for line in header.lines() {
        let toks: Vec<&str> = line.split_whitespace().collect();
        match toks.as_slice() {
            ["format", f, ..] if *f != "binary_little_endian" => {
                return Err(format!("unsupported PLY format `{f}` (need binary_little_endian)"))
            }
            ["element", "vertex", count] => {
                in_vertex = true;
                n = count.parse().map_err(|_| "bad vertex count")?;
            }
            ["element", ..] => in_vertex = false,
            ["property", ty, name] if in_vertex => {
                if *ty != "float" && *ty != "float32" {
                    return Err(format!("property `{name}` has type `{ty}` (need float)"));
                }
                props.push(name.to_string());
            }
            _ => {}
        }
    }
    if n == 0 {
        return Err("no vertex element".into());
    }
    let stride = props.len();
    let data = &bytes[hdr_end..];
    if data.len() < n * stride * 4 {
        return Err(format!(
            "file truncated: need {} bytes of vertex data, have {}",
            n * stride * 4,
            data.len()
        ));
    }
    let idx = |name: &str| props.iter().position(|p| p == name);
    let need = |name: &str| idx(name).ok_or_else(|| format!("missing property `{name}`"));
    let (ix, iy, iz) = (need("x")?, need("y")?, need("z")?);
    let (idc0, idc1, idc2) = (need("f_dc_0")?, need("f_dc_1")?, need("f_dc_2")?);
    let iop = need("opacity")?;
    let (is0, is1, is2) = (need("scale_0")?, need("scale_1")?, need("scale_2")?);
    let (ir0, ir1, ir2, ir3) = (need("rot_0")?, need("rot_1")?, need("rot_2")?, need("rot_3")?);
    // f_rest_* count → SH degree (channel-planar: 3*(K-1) coeffs).
    let n_rest = props.iter().filter(|p| p.starts_with("f_rest_")).count();
    let rest0 = idx("f_rest_0");
    let sh_deg = match n_rest {
        0 => 0,
        9 => 1,
        24 => 2,
        45 => 3,
        other => return Err(format!("unsupported f_rest count {other}")),
    };

    let at = |g: usize, p: usize| -> f32 {
        let off = (g * stride + p) * 4;
        f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
    };

    let mut s = Splats::default();
    s.means.reserve(n * 3);
    s.quats.reserve(n * 4);
    s.scales.reserve(n * 3);
    s.opacities.reserve(n);
    s.colors.reserve(n * 3);
    let mut rest = if sh_deg > 0 { Vec::with_capacity(n * n_rest) } else { Vec::new() };
    for g in 0..n {
        s.means.extend_from_slice(&[at(g, ix), at(g, iy), at(g, iz)]);
        s.quats.extend_from_slice(&[at(g, ir0), at(g, ir1), at(g, ir2), at(g, ir3)]);
        s.scales.extend_from_slice(&[at(g, is0).exp(), at(g, is1).exp(), at(g, is2).exp()]);
        s.opacities.push(sigmoid(at(g, iop)));
        s.colors.extend_from_slice(&[
            SH_C0 * at(g, idc0) + 0.5,
            SH_C0 * at(g, idc1) + 0.5,
            SH_C0 * at(g, idc2) + 0.5,
        ]);
        if let Some(r0) = rest0 {
            for k in 0..n_rest {
                rest.push(at(g, r0 + k));
            }
        }
    }
    if sh_deg > 0 {
        s.sh_rest = Some((sh_deg, rest));
    }
    Ok(s)
}

/// Write a scene as an Inria-layout PLY (degree-0 unless `sh_rest` is set).
pub fn write(path: &str, s: &Splats) -> Result<(), String> {
    let n = s.len();
    let (sh_deg, rest) = match &s.sh_rest {
        Some((d, r)) => (*d, r.as_slice()),
        None => (0, &[][..]),
    };
    let n_rest = match sh_deg {
        0 => 0,
        1 => 9,
        2 => 24,
        3 => 45,
        _ => return Err("unsupported SH degree".into()),
    };
    let f = std::fs::File::create(path).map_err(|e| format!("cannot create {path}: {e}"))?;
    let mut w = BufWriter::new(f);
    let mut hdr = String::from("ply\nformat binary_little_endian 1.0\n");
    hdr += &format!("element vertex {n}\n");
    for p in ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] {
        hdr += &format!("property float {p}\n");
    }
    for k in 0..n_rest {
        hdr += &format!("property float f_rest_{k}\n");
    }
    for p in ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] {
        hdr += &format!("property float {p}\n");
    }
    hdr += "end_header\n";
    w.write_all(hdr.as_bytes()).map_err(|e| e.to_string())?;

    let mut rec: Vec<f32> = Vec::with_capacity(17 + n_rest);
    for g in 0..n {
        rec.clear();
        rec.extend_from_slice(&s.means[g * 3..g * 3 + 3]);
        rec.extend_from_slice(&[0.0, 0.0, 0.0]); // nx ny nz
        for k in 0..3 {
            rec.push((s.colors[g * 3 + k] - 0.5) / SH_C0);
        }
        rec.extend_from_slice(&rest[g * n_rest..g * n_rest + n_rest]);
        rec.push(logit(s.opacities[g].clamp(1e-6, 1.0 - 1e-6)));
        for k in 0..3 {
            rec.push(s.scales[g * 3 + k].max(1e-12).ln());
        }
        rec.extend_from_slice(&s.quats[g * 4..g * 4 + 4]);
        let bytes: Vec<u8> = rec.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_all(&bytes).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    let needle = b"end_header\n";
    bytes.windows(needle.len()).position(|w| w == needle).map(|p| p + needle.len())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn logit(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}
