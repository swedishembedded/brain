// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

//! Which image pairs are worth matching: a global descriptor per image ranks
//! every other image by how much of the same scene it plausibly shows, and
//! only the best-ranked pairs (plus neighbours in capture order) go on to
//! descriptor matching and geometric verification.
//!
//! Matching every pair costs `n(n-1)/2` quadratic descriptor searches, the
//! bulk of a capture's run time well before a hundred photographs. Most
//! pairs of a large capture share nothing, and a global descriptor says so
//! for the price of one dot product.
//!
//! The descriptor is VLAD (Jégou et al., "Aggregating local descriptors into
//! a compact image representation", CVPR 2010) over the RootSIFT descriptors
//! already computed, with intra-normalization (Arandjelović & Zisserman,
//! "All about VLAD", CVPR 2013) and signed square-root power normalization.
//! Its vocabulary is learned by k-means on the capture's own descriptors, so
//! nothing is downloaded and no trained weights are involved: VLAD's
//! residuals carry the fine distinctions a coarse vocabulary leaves out,
//! which is why a few dozen words learned on the spot are enough to rank
//! the images of one capture against each other.
//!
//! Swedish Embedded AB implements image retrieval and matching for
//! photogrammetry for its clients. If your team needs large photo
//! collections turned into reconstructions, you can procure our services by
//! sending an email to info@swedishembedded.com.

use crate::sift::DESC;
use data::rng::Lcg;

/// How the pairs to match are chosen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairSelection {
    /// Match every pair of a capture of at most this many photographs.
    pub exhaustive_up_to: usize,
    /// Per image, match the images most similar to it by global descriptor.
    pub top_k: usize,
    /// Per image, also match the images this many places before and after
    /// it in input order (a video or a walk is captured in order).
    pub sequential: usize,
    /// Visual words of the VLAD vocabulary.
    pub words: usize,
}

impl Default for PairSelection {
    fn default() -> Self {
        PairSelection { exhaustive_up_to: 24, top_k: 12, sequential: 2, words: 64 }
    }
}

/// Descriptors sampled to learn the vocabulary: k-means cost is linear in
/// them, and a few hundred per word pin a centre down.
const TRAIN_SAMPLE: usize = 40_000;
const KMEANS_ITERS: usize = 12;

/// A VLAD vocabulary: `words` centres of `DESC` floats.
pub struct Vocabulary {
    centres: Vec<f32>,
    words: usize,
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

impl Vocabulary {
    /// k-means (k-means++ seeding, Lloyd iterations) over a deterministic
    /// sample of every image's descriptors.
    pub fn learn(images: &[&[f32]], words: usize, seed: u64) -> Vocabulary {
        let total: usize = images.iter().map(|d| d.len() / DESC).sum();
        let mut rng = Lcg::new(seed);
        let stride = (total / TRAIN_SAMPLE).max(1);
        let mut sample: Vec<f32> = Vec::with_capacity(total.min(TRAIN_SAMPLE) * DESC);
        let mut k = 0usize;
        for d in images {
            for f in d.chunks_exact(DESC) {
                if k.is_multiple_of(stride) {
                    sample.extend_from_slice(f);
                }
                k += 1;
            }
        }
        let m = sample.len() / DESC;
        let words = words.clamp(1, m.max(1));
        if m == 0 {
            return Vocabulary { centres: vec![0.0; words * DESC], words };
        }
        let row = |i: usize| &sample[i * DESC..(i + 1) * DESC];
        // k-means++: each next centre drawn in proportion to its squared
        // distance from the nearest centre so far
        let mut centres: Vec<f32> = row(rng.next_u32() as usize % m).to_vec();
        let mut near: Vec<f32> = (0..m).map(|i| dist2(row(i), &centres[..DESC])).collect();
        while centres.len() / DESC < words {
            let sum: f64 = near.iter().map(|&v| v as f64).sum();
            let mut pick = rng.unit() as f64 * sum;
            let mut chosen = m - 1;
            for (i, &v) in near.iter().enumerate() {
                pick -= v as f64;
                if pick <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            let c = row(chosen).to_vec();
            for (i, n) in near.iter_mut().enumerate() {
                *n = n.min(dist2(row(i), &c));
            }
            centres.extend(c);
        }
        let mut voc = Vocabulary { centres, words };
        for _ in 0..KMEANS_ITERS {
            let assign = backend_cpu::par::map(m, |i| voc.nearest(row(i)));
            let mut sum = vec![0.0f64; words * DESC];
            let mut count = vec![0usize; words];
            for (i, &w) in assign.iter().enumerate() {
                count[w] += 1;
                for (s, v) in sum[w * DESC..(w + 1) * DESC].iter_mut().zip(row(i)) {
                    *s += *v as f64;
                }
            }
            for w in 0..words {
                // an emptied word keeps its centre
                if count[w] > 0 {
                    for j in 0..DESC {
                        voc.centres[w * DESC + j] = (sum[w * DESC + j] / count[w] as f64) as f32;
                    }
                }
            }
        }
        voc
    }

    fn centre(&self, w: usize) -> &[f32] {
        &self.centres[w * DESC..(w + 1) * DESC]
    }

    fn nearest(&self, d: &[f32]) -> usize {
        (0..self.words).map(|w| (w, dist2(d, self.centre(w)))).min_by(|a, b| a.1.total_cmp(&b.1)).map_or(0, |b| b.0)
    }

    /// The VLAD vector of one image's descriptors: per word, the sum of the
    /// residuals of the descriptors assigned to it, each word's block
    /// normalized on its own (so a burst of one repeated texture cannot
    /// dominate), signed-square-rooted, and the whole normalized to unit
    /// length. An image with no descriptors gets the zero vector.
    pub fn vlad(&self, desc: &[f32]) -> Vec<f32> {
        let mut v = vec![0.0f32; self.words * DESC];
        for d in desc.chunks_exact(DESC) {
            let w = self.nearest(d);
            for (acc, (x, c)) in v[w * DESC..(w + 1) * DESC].iter_mut().zip(d.iter().zip(self.centre(w))) {
                *acc += x - c;
            }
        }
        for block in v.chunks_exact_mut(DESC) {
            let n = block.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 {
                block.iter_mut().for_each(|x| *x /= n);
            }
        }
        v.iter_mut().for_each(|x| *x = x.signum() * x.abs().sqrt());
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if n > 0.0 {
            v.iter_mut().for_each(|x| *x /= n);
        }
        v
    }
}

/// The pairs `(a, b)`, `a < b`, sorted, of the images whose descriptors are
/// `images` (RootSIFT, `DESC` floats each) that should be matched: every pair
/// of a small capture, otherwise each image's `top_k` nearest by VLAD
/// similarity and its `sequential` neighbours in input order.
pub fn select_pairs(images: &[&[f32]], cfg: &PairSelection, seed: u64) -> Vec<(usize, usize)> {
    let n = images.len();
    let all = || (0..n).flat_map(|a| (a + 1..n).map(move |b| (a, b))).collect();
    if n <= cfg.exhaustive_up_to || cfg.top_k + 1 >= n {
        return all();
    }
    let voc = Vocabulary::learn(images, cfg.words, seed);
    let g: Vec<Vec<f32>> = backend_cpu::par::map(n, |i| voc.vlad(images[i]));
    let mut pairs = std::collections::BTreeSet::new();
    for a in 0..n {
        let mut sim: Vec<(usize, f32)> = (0..n).filter(|&b| b != a).map(|b| (b, g[a].iter().zip(&g[b]).map(|(x, y)| x * y).sum())).collect();
        sim.sort_by(|x, y| y.1.total_cmp(&x.1).then(x.0.cmp(&y.0)));
        for &(b, _) in sim.iter().take(cfg.top_k) {
            pairs.insert((a.min(b), a.max(b)));
        }
        for b in a + 1..(a + 1 + cfg.sequential).min(n) {
            pairs.insert((a, b));
        }
    }
    pairs.into_iter().collect()
}
