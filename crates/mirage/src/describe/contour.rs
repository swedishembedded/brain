// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Polygon extraction from a probability field: marching squares at a level
//! with linear interpolation, outer boundaries only, simplified with the
//! Douglas-Peucker rule, and a bridge merge that joins several polygons into
//! one ring.

use std::collections::HashMap;

/// A point in sample coordinates (`x` = column, `y` = row).
pub type Point = [f32; 2];

/// Edge of the padded sample grid a contour crosses: `(vertical?, row, col)`.
type Edge = (bool, usize, usize);

/// Outer boundaries of the region where `field` (`side x side`) exceeds
/// `level`, as closed rings in sample coordinates, largest area first.
pub fn outer_contours(field: &[f32], side: usize, level: f32) -> Vec<Vec<Point>> {
    let n = side + 2;
    let at = |r: usize, c: usize| -> f32 { if (1..=side).contains(&r) && (1..=side).contains(&c) { field[(r - 1) * side + c - 1] } else { 0.0 } };
    let point = |e: Edge| -> Point {
        let (vertical, r, c) = e;
        let (a, b) = if vertical { (at(r, c), at(r + 1, c)) } else { (at(r, c), at(r, c + 1)) };
        let t = if (b - a).abs() > f32::EPSILON { ((level - a) / (b - a)).clamp(0.0, 1.0) } else { 0.5 };
        if vertical { [c as f32 - 1.0, r as f32 - 1.0 + t] } else { [c as f32 - 1.0 + t, r as f32 - 1.0] }
    };
    // Directed segments, interior on the right-hand side of travel.
    let mut next: HashMap<Edge, Edge> = HashMap::new();
    for r in 0..n - 1 {
        for c in 0..n - 1 {
            // Corners clockwise from the top-left, and the edge after each.
            let corners = [(r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c)];
            let inside = corners.map(|(rr, cc)| at(rr, cc) > level);
            let edges = [(false, r, c), (true, r, c + 1), (false, r + 1, c), (true, r, c)];
            // Edge i joins corner i and corner i+1 (clockwise).
            let mut enter = Vec::new();
            let mut leave = Vec::new();
            for i in 0..4 {
                match (inside[i], inside[(i + 1) % 4]) {
                    (false, true) => enter.push(i),
                    (true, false) => leave.push(i),
                    _ => {}
                }
            }
            let centre_inside = corners.iter().map(|&(rr, cc)| at(rr, cc)).sum::<f32>() / 4.0 > level;
            for &e in &enter {
                // Next leave edge clockwise, or the previous one when the saddle centre is inside.
                let target = if centre_inside && leave.len() == 2 {
                    *leave.iter().rev().find(|&&l| l < e).or(leave.last()).expect("a leave edge")
                } else {
                    *leave.iter().find(|&&l| l > e).or(leave.first()).expect("a leave edge")
                };
                next.insert(edges[e], edges[target]);
            }
        }
    }
    let mut visited: std::collections::HashSet<Edge> = std::collections::HashSet::new();
    let mut rings = Vec::new();
    let mut starts: Vec<Edge> = next.keys().copied().collect();
    starts.sort_unstable();
    for s in starts {
        if visited.contains(&s) {
            continue;
        }
        let mut ring = Vec::new();
        let mut e = s;
        while visited.insert(e) {
            ring.push(point(e));
            match next.get(&e) {
                Some(&n) => e = n,
                None => break,
            }
        }
        // Image coordinates (y down) with interior on the right give a positive shoelace area for outer rings.
        if ring.len() >= 3 && area(&ring) < 0.0 {
            rings.push(ring);
        }
    }
    rings.sort_by(|a, b| area(b).abs().total_cmp(&area(a).abs()));
    rings
}

/// Signed shoelace area.
pub fn area(ring: &[Point]) -> f32 {
    (0..ring.len()).map(|i| {
        let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
        a[0] * b[1] - b[0] * a[1]
    }).sum::<f32>() / 2.0
}

fn distance_to_chord(p: Point, a: Point, b: Point) -> f32 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len = dx.hypot(dy);
    if len == 0.0 {
        return (p[0] - a[0]).hypot(p[1] - a[1]);
    }
    ((p[0] - a[0]) * dy - (p[1] - a[1]) * dx).abs() / len
}

fn simplify_open(points: &[Point], eps: f32) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }
    let (a, b) = (points[0], points[points.len() - 1]);
    let (idx, dist) = (1..points.len() - 1).map(|i| (i, distance_to_chord(points[i], a, b))).fold((0, -1.0), |m, x| if x.1 > m.1 { x } else { m });
    if dist > eps {
        let mut left = simplify_open(&points[..=idx], eps);
        left.pop();
        left.extend(simplify_open(&points[idx..], eps));
        left
    } else {
        vec![a, b]
    }
}

/// Douglas-Peucker on a closed ring, split at the point farthest from the first.
pub fn simplify_ring(ring: &[Point], eps: f32) -> Vec<Point> {
    if ring.len() < 4 {
        return ring.to_vec();
    }
    let far = (0..ring.len()).max_by(|&i, &j| {
        let d = |k: usize| (ring[k][0] - ring[0][0]).hypot(ring[k][1] - ring[0][1]);
        d(i).total_cmp(&d(j))
    }).expect("non-empty ring");
    let mut first = simplify_open(&ring[..=far], eps);
    let mut second_in: Vec<Point> = ring[far..].to_vec();
    second_in.push(ring[0]);
    let mut second = simplify_open(&second_in, eps);
    first.pop();
    second.pop();
    first.extend(second);
    first
}

/// Join `parts` into one ring: each later polygon is spliced into the first
/// at its nearest pair of vertices, repeating the two splice vertices so the
/// bridge has no width.
pub fn merge_rings(parts: &[Vec<Point>]) -> Vec<Point> {
    let mut merged = parts[0].clone();
    for part in &parts[1..] {
        let (mut best, mut bi, mut bj) = (f32::INFINITY, 0, 0);
        for (i, a) in merged.iter().enumerate() {
            for (j, b) in part.iter().enumerate() {
                let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
                if d < best {
                    (best, bi, bj) = (d, i, j);
                }
            }
        }
        let mut out: Vec<Point> = merged[..=bi].to_vec();
        out.extend_from_slice(&part[bj..]);
        out.extend_from_slice(&part[..=bj]);
        out.extend_from_slice(&merged[bi..]);
        merged = out;
    }
    merged
}
