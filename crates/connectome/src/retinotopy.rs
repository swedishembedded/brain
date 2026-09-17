// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Where each column of the eye points, recovered from connectivity alone.
//!
//! An eye needs a map. To turn a rendered image into current arriving at
//! photoreceptor cells, something has to say which cell looks at which
//! direction, and BANC's export carries no coordinates at all: no soma
//! position, no column identity, nothing spatial. The obvious conclusion is
//! that a visual front end cannot be built on it.
//!
//! It can, because retinotopy is not only a fact about space. Neighbouring
//! columns of the medulla share downstream partners - the cells that span
//! several columns connect to a compact neighbourhood and not to a scattered
//! set - so "which columns are near each other" is written in the wiring. Two
//! columns that share many partners are adjacent; the whole sheet is then a
//! graph whose shape is the retina's, and recovering coordinates is a spectral
//! embedding of it.
//!
//! What this module does NOT do is claim the result is a retinal position. It
//! recovers the LATTICE: a set of 2D coordinates in which graph neighbours are
//! near each other. Turning that into a direction in the world needs an
//! anchor - which edge is dorsal, which is anterior - that connectivity cannot
//! supply and anatomy has to.
//!
//! A warning that this module exists to make loud. The embedding is of ONE
//! connected component. A graph with isolated nodes has an exact zero mode per
//! isolated node, so the leading eigenvectors of the whole matrix are
//! indicator functions of single cells, and the embedding they produce is
//! noise. Analysing BANC's left optic lobe that way suggested it was
//! shattered; restricted to its giant component, which holds 93% of its
//! columns, it behaves like the right one and is merely sparser.
//!
//! Swedish Embedded AB implements structure recovery from connectivity and
//! other large graphs for its clients, including the spectral methods that
//! turn an unlabelled adjacency into usable coordinates. If your team needs
//! this, you can procure our services by sending an email to
//! info@swedishembedded.com.

use std::collections::HashMap;

use crate::Connectome;

/// How a column graph is built. Every one of these is a judgement call, so
/// every one is a named knob rather than a literal in a loop.
#[derive(Clone, Debug, PartialEq)]
pub struct Policy {
    /// The cell type that marks a column. `L1` by default: one per
    /// ommatidium, and the fly's principal achromatic channel.
    pub column_type: String,
    /// A downstream partner counts only above this many synapses.
    pub min_synapses: f32,
    /// A partner shared by more than this many columns says nothing local.
    ///
    /// Wide-field cells span a whole optic lobe and connect columns that are
    /// not neighbours at all. Left in, they make the graph a near-clique and
    /// the embedding collapses; the number is a statement about how many
    /// columns a genuinely columnar cell touches.
    pub max_span: usize,
}

impl Default for Policy {
    fn default() -> Self {
        Policy { column_type: "L1".into(), min_synapses: 3.0, max_span: 12 }
    }
}

/// A sheet of columns and how they sit next to each other.
#[derive(Clone, Debug, Default)]
pub struct Sheet {
    /// Graph index of each column, in this sheet's own order.
    pub columns: Vec<u32>,
    /// Neighbours of each column, with the number of shared partners.
    pub neighbours: Vec<Vec<(u32, f32)>>,
    /// Which connected component each column is in.
    pub component: Vec<u32>,
    /// Size of each component.
    pub component_size: Vec<usize>,
}

impl Sheet {
    /// Build the shared-partner graph over one side's columns.
    pub fn build(c: &Connectome, side: &str, policy: &Policy) -> Sheet {
        let columns = c.population(|n| n.cell_type == policy.column_type && n.soma_side == side);
        let of_column: HashMap<u32, u32> = columns.iter().enumerate().map(|(i, &g)| (g, i as u32)).collect();

        // Which columns reach each downstream cell. A postsynaptic cell is
        // found by walking the CSC column of every neuron once, which is the
        // only direction this layout indexes.
        let mut owners: HashMap<u32, Vec<u32>> = HashMap::new();
        for post in 0..c.csc.n as usize {
            let (lo, hi) = (c.csc.indptr[post] as usize, c.csc.indptr[post + 1] as usize);
            for k in lo..hi {
                if c.csc.w[k] < policy.min_synapses {
                    continue;
                }
                if let Some(&col) = of_column.get(&c.csc.pre[k]) {
                    owners.entry(post as u32).or_default().push(col);
                }
            }
        }

        let mut shared: HashMap<(u32, u32), f32> = HashMap::new();
        for (_, cols) in owners.iter() {
            if cols.len() < 2 || cols.len() > policy.max_span {
                continue;
            }
            for i in 0..cols.len() {
                for j in i + 1..cols.len() {
                    let (a, b) = (cols[i].min(cols[j]), cols[i].max(cols[j]));
                    if a != b {
                        *shared.entry((a, b)).or_insert(0.0) += 1.0;
                    }
                }
            }
        }

        let mut neighbours = vec![Vec::new(); columns.len()];
        for ((a, b), w) in shared {
            neighbours[a as usize].push((b, w));
            neighbours[b as usize].push((a, w));
        }
        let (component, component_size) = label_components(&neighbours);
        Sheet { columns, neighbours, component, component_size }
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The largest connected component, as indices into this sheet.
    ///
    /// The only set worth embedding. See this module's header: isolated nodes
    /// contribute exact zero modes that displace the modes carrying the
    /// sheet's shape, so embedding the whole graph reports noise with complete
    /// confidence.
    pub fn giant(&self) -> Vec<u32> {
        let Some(big) = (0..self.component_size.len()).max_by_key(|&i| self.component_size[i]) else {
            return Vec::new();
        };
        (0..self.len() as u32).filter(|&i| self.component[i as usize] == big as u32).collect()
    }

    /// Recover 2D coordinates for the giant component.
    ///
    /// Returns `(index into this sheet, position)`. The positions are
    /// arbitrary up to rotation, reflection and scale: a spectral embedding
    /// knows the shape of the sheet and nothing about how it is oriented in
    /// the head.
    pub fn embed(&self, iterations: usize) -> Vec<(u32, [f32; 2])> {
        let g = self.giant();
        if g.len() < 8 {
            return Vec::new();
        }
        let local: HashMap<u32, usize> = g.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let n = g.len();
        let adj: Vec<Vec<(usize, f32)>> = g
            .iter()
            .map(|&c| {
                self.neighbours[c as usize]
                    .iter()
                    .filter_map(|&(v, w)| local.get(&v).map(|&j| (j, w)))
                    .collect()
            })
            .collect();
        let deg: Vec<f32> = adj.iter().map(|row| row.iter().map(|&(_, w)| w).sum()).collect();
        let dm: Vec<f32> = deg.iter().map(|d| if *d > 0.0 { 1.0 / d.sqrt() } else { 0.0 }).collect();

        // Subspace iteration on M = (I + D^-1/2 A D^-1/2) / 2.
        //
        // The normalised Laplacian is L = I - D^-1/2 A D^-1/2, so M's LARGEST
        // eigenvectors are L's smallest, which are the ones that describe the
        // sheet. The shift into [0, 1] is not cosmetic: without it the most
        // negative eigenvalue can be larger in magnitude than the ones being
        // sought, and the iteration converges confidently to the wrong
        // subspace.
        let trivial: Vec<f32> = {
            let mut v: Vec<f32> = deg.iter().map(|d| d.sqrt()).collect();
            normalise(&mut v);
            v
        };
        let mut basis: Vec<Vec<f32>> = (0..2)
            .map(|k| {
                // Deterministic start: a search that needs a lucky seed is not
                // a measurement.
                let mut v: Vec<f32> = (0..n).map(|i| ((i * (k + 3) * 2654435761) % 1000) as f32 / 500.0 - 1.0).collect();
                project_out(&mut v, &trivial);
                normalise(&mut v);
                v
            })
            .collect();

        for _ in 0..iterations {
            for v in basis.iter_mut() {
                let mut out = vec![0.0f32; n];
                for i in 0..n {
                    let mut acc = 0.0;
                    for &(j, w) in &adj[i] {
                        acc += w * dm[i] * dm[j] * v[j];
                    }
                    out[i] = 0.5 * (v[i] + acc);
                }
                *v = out;
                project_out(v, &trivial);
            }
            // Orthonormalise, so the two vectors do not both slide onto the
            // single slowest mode.
            for k in 0..basis.len() {
                let (before, rest) = basis.split_at_mut(k);
                for prev in before.iter() {
                    project_out(&mut rest[0], prev);
                }
                normalise(&mut rest[0]);
            }
        }

        g.iter()
            .enumerate()
            .map(|(i, &c)| (c, [basis[0][i] * dm[i], basis[1][i] * dm[i]]))
            .collect()
    }
}

fn normalise(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn project_out(v: &mut [f32], u: &[f32]) {
    let d: f32 = v.iter().zip(u).map(|(a, b)| a * b).sum();
    for (a, b) in v.iter_mut().zip(u) {
        *a -= d * b;
    }
}

fn label_components(neighbours: &[Vec<(u32, f32)>]) -> (Vec<u32>, Vec<usize>) {
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

/// How well an embedding agrees with the graph it came from.
///
/// The fraction of each column's `k` nearest neighbours in the embedding that
/// are also among its strongest graph neighbours. This is the measurement that
/// says whether coordinates were recovered or merely produced: a spectral
/// embedding always returns numbers, and numbers always plot.
pub fn agreement(sheet: &Sheet, embedding: &[(u32, [f32; 2])], k: usize) -> f64 {
    if embedding.len() < k + 1 {
        return 0.0;
    }
    let mut hit = 0usize;
    let mut total = 0usize;
    for (i, &(col, p)) in embedding.iter().enumerate() {
        let mut by_distance: Vec<(f32, usize)> = embedding
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(j, &(_, q))| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2), j))
            .collect();
        by_distance.sort_by(|a, b| a.0.total_cmp(&b.0));
        let near: std::collections::HashSet<u32> =
            by_distance.iter().take(k).map(|&(_, j)| embedding[j].0).collect();

        let mut graph: Vec<(u32, f32)> = sheet.neighbours[col as usize].clone();
        graph.sort_by(|a, b| b.1.total_cmp(&a.1));
        let graph: Vec<u32> = graph.into_iter().take(k).map(|(v, _)| v).collect();
        if graph.is_empty() {
            continue;
        }
        total += graph.len();
        hit += graph.iter().filter(|v| near.contains(v)).count();
    }
    if total == 0 {
        0.0
    } else {
        hit as f64 / total as f64
    }
}
