// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

// @what  Latent sparse attention: Ward-merge one closed cluster block's index keys into buckets and write its index page
// @how   64-thread workgroup per block: pool sums, a tiled Gram matrix updated in place per merge, mutual-cheapest merges, centroids and member lists written into the page
// @opt   4
// @cpu   no
// @gpu   yes
// @npu   no
// @quant none
// @dtype f32
//
// Swedish Embedded AB implements long-context attention whose key index is
// built where the keys live, for clients that cannot afford a host round trip
// per indexed block. If your team needs expertise in GPU clustering and
// indexing kernels, you can procure our services by sending an email to
// info@swedishembedded.com.
//
// The device twin of `model::lsa::cluster_block` (Ward as `model::lsa::ward`
// documents it): the block's `cluster_block` unit index keys are summed into
// pools of `pool_size`; each round every live bucket finds its cheapest
// partner the size cap allows (cost `a(b) + a(b') - a(b u b')`, `a` the norm
// of a bucket's key sum, a tie to the lower index), every mutual pair merges
// into its lower member, and the round that would overshoot the target count
// merges only its cheapest pairs (a tie to the lower member). Buckets are
// numbered by their first pool; a centroid is the normalised sum of its keys;
// members are absolute positions `base + key` in ascending order, the rest
// of the row `0xFFFFFFFF`. The page is written at row `page *
// buckets_per_block` of the pool.
//
// The Gram matrix of the pool sums is built once (tiles staged through
// workgroup memory) and kept current by linearity: merging `g` into `k` adds
// row `g` to row `k`, then column `g` to column `k` - the two passes in that
// order also merge pairs of one round correctly - O(pools) a merge where the
// host reference recomputes it from the sums every round. The passes sum
// entries (i, j) and (j, i) in different orders, so a pair's cost reads one
// triangle: a cost exactly the same in both directions makes the cheapest
// pair (a tie to the lower index) mutual, and a round can never come up
// without a merge while buckets remain to merge. Arithmetic is f32
// against the host's f64, so a near-tie may merge differently; well-separated
// keys merge identically (the gate).
//
// One dispatch clusters many blocks at once: workgroup `w` reads its keys at
// `keys[w * cluster_block ..]`, owns scratch slab `w` (`pools * index_dim +
// pools * pools` floats) and writes page `page` of slot `w` (bucket rows from
// `(w * pages + page) * buckets_per_block`). Serving gives each sparse layer
// a slot and clusters one block of a sequence for every layer (`seq_blocks`
// 0: every slot's block starts at `base`); training gives every block of
// `b` sequences of `seq_blocks` blocks a slot of one page, block `w %
// seq_blocks` of its sequence starting at `base + (w % seq_blocks) *
// cluster_block`. Limits: pools (cluster_block / pool_size) <= 256,
// buckets_per_block <= 128, index_dim <= 1024 (64-channel chunks).

const WG: u32 = 64u;
const MAX_POOLS: u32 = 256u;
const NO_KEY: u32 = 0xFFFFFFFFu;
const INF: f32 = 3.4e38;

struct Params {
    index_dim: u32,
    cluster_block: u32,
    pool_size: u32,
    buckets_per_block: u32,
    bucket_cap: u32,
    page: u32,
    base: u32,       // position of the block's first key
    pages: u32,      // pages per layer slot
    seq_blocks: u32, // blocks per sequence when every slot is a block (0: none)
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       keys:      array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch:   array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(4) var<storage, read_write> members:   array<u32>;

// The scratch is read back by other threads after it changes (the Gram
// matrix is merged in place every round), so every access is an atomic load
// or store: a plain load may return a value the thread saw before another
// thread's store.
fn ld(i: u32) -> f32 { return bitcast<f32>(atomicLoad(&scratch[i])); }
fn st(i: u32, v: f32) { atomicStore(&scratch[i], bitcast<u32>(v)); }

var<workgroup> alive:   array<u32, MAX_POOLS>;
var<workgroup> size:    array<u32, MAX_POOLS>;
var<workgroup> parent:  array<u32, MAX_POOLS>;
var<workgroup> best:    array<u32, MAX_POOLS>;
var<workgroup> bestc:   array<f32, MAX_POOLS>;
var<workgroup> label:   array<u32, MAX_POOLS>;
var<workgroup> pick_i:  array<u32, 128>;      // mutual pairs, cheapest first
var<workgroup> pick_c:  array<f32, 128>;
var<workgroup> n_pick:  u32;
var<workgroup> n_live:  u32;
var<workgroup> ta:      array<f32, 1024>;     // a 16-pool tile of sums, 64 channels
var<workgroup> tb:      array<f32, 1024>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) nwg: vec3<u32>) {
    let w = wgid.y * nwg.x + wgid.x;
    let lt = lid.x;
    let d = p.index_dim;
    let np = p.cluster_block / p.pool_size;
    let slab = w * (np * d + np * np);
    let sums = slab;              // [np, d]
    let gram = slab + np * d;     // [np, np]
    let kbase = w * p.cluster_block * d;

    // Pool sums, every pool alive with its keys.
    for (var e = lt; e < np * d; e = e + WG) {
        let pool = e / d;
        let c = e % d;
        var acc = 0.0;
        for (var k = 0u; k < p.pool_size; k = k + 1u) {
            acc = acc + keys[kbase + (pool * p.pool_size + k) * d + c];
        }
        st(sums + e, acc);
    }
    for (var i = lt; i < np; i = i + WG) {
        alive[i] = 1u;
        size[i] = p.pool_size;
        parent[i] = i;
    }
    if (lt == 0u) { n_live = np; }
    storageBarrier();
    workgroupBarrier();

    // The Gram matrix of the pool sums, 16 x 16 pool tiles at a time, each
    // accumulated over 64-channel chunks (a thread owns 4 of a tile's 256
    // entries).
    let tiles = (np + 15u) / 16u;
    for (var ti = 0u; ti < tiles; ti = ti + 1u) {
        for (var tj = ti; tj < tiles; tj = tj + 1u) {
            var g: array<f32, 4>;
            for (var k = 0u; k < 4u; k = k + 1u) { g[k] = 0.0; }
            for (var c0 = 0u; c0 < d; c0 = c0 + 64u) {
                let cw = min(64u, d - c0);
                for (var e = lt; e < 16u * 64u; e = e + WG) {
                    let r = e / 64u;
                    let c = e % 64u;
                    let pa = ti * 16u + r;
                    let pb = tj * 16u + r;
                    ta[e] = select(0.0, ld(sums + pa * d + c0 + c), pa < np && c < cw);
                    tb[e] = select(0.0, ld(sums + pb * d + c0 + c), pb < np && c < cw);
                }
                workgroupBarrier();
                for (var k = 0u; k < 4u; k = k + 1u) {
                    let e = lt + k * WG;
                    var acc = g[k];
                    for (var c = 0u; c < cw; c = c + 1u) {
                        acc = acc + ta[(e / 16u) * 64u + c] * tb[(e % 16u) * 64u + c];
                    }
                    g[k] = acc;
                }
                workgroupBarrier();
            }
            for (var k = 0u; k < 4u; k = k + 1u) {
                let e = lt + k * WG;
                let i = ti * 16u + e / 16u;
                let j = tj * 16u + e % 16u;
                if (i < np && j < np) {
                    st(gram + i * np + j, g[k]);
                    st(gram + j * np + i, g[k]);
                }
            }
        }
    }
    storageBarrier();
    workgroupBarrier();

    for (var round = 0u; round < 256u; round = round + 1u) {
        let live = workgroupUniformLoad(&n_live);
        if (live <= p.buckets_per_block) { break; }

        // Each live bucket's cheapest allowed partner, a tie to the lower index.
        for (var i = lt; i < np; i = i + WG) {
            var bj = NO_KEY;
            var bc = INF;
            if (alive[i] == 1u) {
                let gii = ld(gram + i * np + i);
                let ai = sqrt(max(gii, 0.0));
                for (var j = 0u; j < np; j = j + 1u) {
                    if (j != i && alive[j] == 1u && size[i] + size[j] <= p.bucket_cap) {
                        let gjj = ld(gram + j * np + j);
                        // One triangle: the cost is the same both ways.
                        let gij = ld(gram + min(i, j) * np + max(i, j));
                        let merged = sqrt(max(gii + gjj + 2.0 * gij, 0.0));
                        let cost = ai + sqrt(max(gjj, 0.0)) - merged;
                        if (cost < bc) {
                            bc = cost;
                            bj = j;
                        }
                    }
                }
            }
            best[i] = bj;
            bestc[i] = bc;
        }
        workgroupBarrier();

        // Mutual pairs listed by their lower member, cheapest first (a tie to
        // the lower member), cut at the merges the round still needs.
        if (lt == 0u) {
            var n = 0u;
            for (var i = 0u; i < np; i = i + 1u) {
                let j = best[i];
                if (j != NO_KEY && i < j && best[j] == i) {
                    var at = n;
                    while (at > 0u && (pick_c[at - 1u] > bestc[i] || (pick_c[at - 1u] == bestc[i] && pick_i[at - 1u] > i))) {
                        pick_c[at] = pick_c[at - 1u];
                        pick_i[at] = pick_i[at - 1u];
                        at = at - 1u;
                    }
                    pick_c[at] = bestc[i];
                    pick_i[at] = i;
                    n = n + 1u;
                }
            }
            n_pick = min(n, live - p.buckets_per_block);
        }
        workgroupBarrier();
        let picks = workgroupUniformLoad(&n_pick);
        if (picks == 0u) { break; }

        // Merge each picked pair into its lower member: row `gone` into row
        // `kept`, then column `gone` into column `kept`.
        for (var e = lt; e < picks * np; e = e + WG) {
            let kept = pick_i[e / np];
            let gone = best[kept];
            let j = e % np;
            st(gram + kept * np + j, ld(gram + kept * np + j) + ld(gram + gone * np + j));
        }
        storageBarrier();
        workgroupBarrier();
        for (var e = lt; e < picks * np; e = e + WG) {
            let kept = pick_i[e / np];
            let gone = best[kept];
            let i = e % np;
            st(gram + i * np + kept, ld(gram + i * np + kept) + ld(gram + i * np + gone));
        }
        storageBarrier();
        workgroupBarrier();
        if (lt == 0u) {
            for (var k = 0u; k < picks; k = k + 1u) {
                let kept = pick_i[k];
                let gone = best[kept];
                size[kept] = size[kept] + size[gone];
                alive[gone] = 0u;
                parent[gone] = kept;
            }
            n_live = live - picks;
        }
        workgroupBarrier();
    }

    // Number the surviving buckets in index order and label every pool.
    if (lt == 0u) {
        var next = 0u;
        for (var i = 0u; i < np; i = i + 1u) {
            if (alive[i] == 1u) {
                label[i] = next;
                next = next + 1u;
            }
        }
        for (var i = 0u; i < np; i = i + 1u) {
            var r = i;
            while (parent[r] != r) { r = parent[r]; }
            best[i] = label[r];
        }
    }
    workgroupBarrier();

    // Centroids: the normalised sum of each bucket's keys, summed 64
    // channels at a time into the page, then scaled in place.
    let page_row = (w * p.pages + p.page) * p.buckets_per_block;
    var base = p.base;
    if (p.seq_blocks > 0u) { base = base + (w % p.seq_blocks) * p.cluster_block; }
    for (var bkt = lt; bkt < p.buckets_per_block; bkt = bkt + WG) {
        let crow = (page_row + bkt) * d;
        var norm2 = 0.0;
        for (var c0 = 0u; c0 < d; c0 = c0 + 64u) {
            let cw = min(64u, d - c0);
            var acc: array<f32, 64>;
            for (var c = 0u; c < cw; c = c + 1u) { acc[c] = 0.0; }
            for (var key = 0u; key < p.cluster_block; key = key + 1u) {
                if (best[key / p.pool_size] == bkt) {
                    for (var c = 0u; c < cw; c = c + 1u) { acc[c] = acc[c] + keys[kbase + key * d + c0 + c]; }
                }
            }
            for (var c = 0u; c < cw; c = c + 1u) {
                centroids[crow + c0 + c] = acc[c];
                norm2 = norm2 + acc[c] * acc[c];
            }
        }
        let inv = 1.0 / max(sqrt(norm2), 1e-30);
        for (var c = 0u; c < d; c = c + 1u) {
            centroids[crow + c] = centroids[crow + c] * inv;
        }
        // Members in ascending position, padded.
        var n = 0u;
        for (var key = 0u; key < p.cluster_block; key = key + 1u) {
            if (best[key / p.pool_size] == bkt && n < p.bucket_cap) {
                members[(page_row + bkt) * p.bucket_cap + n] = base + key;
                n = n + 1u;
            }
        }
        for (var m = n; m < p.bucket_cap; m = m + 1u) {
            members[(page_row + bkt) * p.bucket_cap + m] = NO_KEY;
        }
    }
}
