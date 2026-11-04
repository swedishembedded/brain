// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Gates for recovering the eye's lattice from connectivity.
//!
//! The synthetic tests come first and matter most. A spectral embedding always
//! returns numbers and numbers always plot, so the only way to know the method
//! works is to run it on a sheet whose answer is already known and check that
//! it comes back. A hexagonal lattice is that sheet: it is what an insect eye
//! is, and its true coordinates are the ones used to build it.
//!
//! The dataset test then states what BANC supports, and it is deliberately
//! phrased as a claim about THIS method on THIS release rather than about the
//! animal.

use connectome::retinotopy::{agreement, Policy, Sheet};

/// Build a hexagonal sheet directly as a `Sheet`, with known coordinates.
fn hex(rows: usize, cols: usize) -> (Sheet, Vec<[f32; 2]>) {
    let mut truth = Vec::new();
    let mut at = std::collections::HashMap::new();
    for r in 0..rows {
        for q in 0..cols {
            at.insert((r as i32, q as i32), truth.len() as u32);
            // Offset every other row: this is a hexagonal packing, not a grid.
            truth.push([q as f32 + if r % 2 == 1 { 0.5 } else { 0.0 }, r as f32 * 0.866]);
        }
    }
    let n = truth.len();
    let mut neighbours = vec![Vec::new(); n];
    for r in 0..rows as i32 {
        for q in 0..cols as i32 {
            let me = at[&(r, q)];
            let odd = r.rem_euclid(2) == 1;
            let shift = if odd { 1 } else { -1 };
            for (dr, dq) in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, shift), (-1, shift)] {
                if let Some(&other) = at.get(&(r + dr, q + dq)) {
                    neighbours[me as usize].push((other, 1.0));
                }
            }
        }
    }
    let (component, component_size) = relabel(&neighbours);
    let sheet = Sheet { columns: (0..n as u32).collect(), neighbours, component, component_size };
    assert_eq!(sheet.component_size.len(), 1, "a hexagonal sheet is connected");
    (sheet, truth)
}

#[test]
fn a_known_hexagonal_sheet_comes_back_out_of_its_own_connectivity() {
    let (sheet, truth) = hex(20, 20);
    assert_eq!(sheet.len(), 400);
    let e = sheet.embed(3000);
    assert_eq!(e.len(), 400, "the whole lattice is one component");

    // Neighbour agreement against the graph: the embedding has to put graph
    // neighbours near each other.
    let a = agreement(&sheet, &e, 6);
    eprintln!("hexagonal lattice: neighbour agreement {:.0}%", 100.0 * a);
    assert!(a > 0.6, "a clean lattice should embed well, got {:.0}%", 100.0 * a);

    // And against the TRUTH, which the graph never saw: distances in the
    // embedding must correlate with distances in the real sheet.
    let mut pairs: Vec<(f32, f32)> = Vec::new();
    for i in 0..80 {
        for j in 0..80 {
            if i == j {
                continue;
            }
            let (a, b) = (i * 5, j * 5);
            let d_true = ((truth[a][0] - truth[b][0]).powi(2) + (truth[a][1] - truth[b][1]).powi(2)).sqrt();
            let (pa, pb) = (e[a].1, e[b].1);
            let d_emb = ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2)).sqrt();
            pairs.push((d_true, d_emb));
        }
    }
    let r = pearson(&pairs);
    eprintln!("hexagonal lattice: distance correlation with the truth r = {r:.3}");
    assert!(r > 0.8, "recovered distances should track real ones, got r = {r:.3}");
}

/// The failure this module exists to prevent.
#[test]
fn isolated_columns_do_not_get_to_decide_the_embedding() {
    let (mut sheet, _) = hex(16, 16);
    let n = sheet.len();
    // Add ten columns connected to nothing, exactly as an under-reconstructed
    // hemisphere produces.
    for _ in 0..10 {
        sheet.columns.push(u32::MAX);
        sheet.neighbours.push(Vec::new());
    }
    let (label, sizes) = relabel(&sheet.neighbours);
    sheet.component = label;
    sheet.component_size = sizes;
    assert_eq!(sheet.component_size.len(), 11, "one sheet and ten islands");

    let giant = sheet.giant();
    assert_eq!(giant.len(), n, "the giant component is the sheet, and the islands are not in it");
    let e = sheet.embed(3000);
    assert_eq!(e.len(), n, "the embedding covers the giant component only");
    let a = agreement(&sheet, &e, 6);
    eprintln!("lattice plus ten isolated columns: neighbour agreement {:.0}%", 100.0 * a);
    assert!(a > 0.6, "isolated columns must not disturb the sheet's embedding, got {:.0}%", 100.0 * a);
}

fn relabel(neighbours: &[Vec<(u32, f32)>]) -> (Vec<u32>, Vec<usize>) {
    let n = neighbours.len();
    let mut label = vec![u32::MAX; n];
    let mut sizes = Vec::new();
    for s in 0..n {
        if label[s] != u32::MAX {
            continue;
        }
        let c = sizes.len() as u32;
        let mut stack = vec![s];
        label[s] = c;
        let mut size = 0;
        while let Some(u) = stack.pop() {
            size += 1;
            for &(v, _) in &neighbours[u] {
                if label[v as usize] == u32::MAX {
                    label[v as usize] = c;
                    stack.push(v as usize);
                }
            }
        }
        sizes.push(size);
    }
    (label, sizes)
}

fn pearson(p: &[(f32, f32)]) -> f64 {
    let n = p.len() as f64;
    let (mx, my) = (
        p.iter().map(|x| x.0 as f64).sum::<f64>() / n,
        p.iter().map(|x| x.1 as f64).sum::<f64>() / n,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in p {
        let (dx, dy) = (*x as f64 - mx, *y as f64 - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        0.0
    } else {
        sxy / (sxx * syy).sqrt()
    }
}

/// What BANC v888 supports, stated as a claim about this method on this
/// release.
///
/// Both optic lobes recover a lattice. The left is sparser, and the raw
/// difference is documented rather than anatomical: BANC's annotation effort
/// concentrated on the right hemisphere, and the lamina is absent from the
/// dataset entirely, so L1 is reconstructed from its medulla arbor alone on
/// both sides and is correspondingly sensitive to proofreading.
///
/// The gate is that BOTH sides produce a giant component holding most of their
/// columns and an embedding that agrees with the graph well above chance. An
/// earlier version of this analysis embedded the whole adjacency including
/// isolated columns, concluded the left lobe was shattered, and would have
/// made the animal permanently one-eyed on the strength of a bug.
#[test]
fn both_optic_lobes_recover_a_lattice_from_connectivity() {
    let Ok(root) = std::env::var("BRAIN_CONNECTOME_DIR") else {
        eprintln!("skipping: set BRAIN_CONNECTOME_DIR");
        return;
    };
    let Ok((neurons, edges)) = connectome::find(&root, "banc") else {
        eprintln!("skipping: no banc dataset under {root}");
        return;
    };
    let c = connectome::load("banc", &neurons, &edges).expect("BANC loads");
    let policy = Policy::default();

    for side in ["right", "left"] {
        let sheet = Sheet::build(&c, side, &policy);
        let giant = sheet.giant();
        let share = giant.len() as f64 / sheet.len() as f64;
        let e = sheet.embed(4000);
        let a = agreement(&sheet, &e, 6);
        eprintln!(
            "{side}: {} columns, {} components, giant {} ({:.0}%), neighbour agreement {:.0}%",
            sheet.len(),
            sheet.component_size.len(),
            giant.len(),
            100.0 * share,
            100.0 * a
        );
        assert!(sheet.len() > 700, "{side}: expected about 800 L1 columns, found {}", sheet.len());
        assert!(share > 0.9, "{side}: the giant component holds only {:.0}% of columns", 100.0 * share);
        assert!(a > 0.35, "{side}: neighbour agreement {:.0}% is too close to chance", 100.0 * a);
    }
}
