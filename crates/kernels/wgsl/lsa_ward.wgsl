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
// One dispatch builds one block's page for every sparse layer: workgroup `w`
// is layer slot `w`, reads its keys at `keys[w * cluster_block ..]`, owns
// scratch slab `w` (`pools * index_dim + pools * pools` floats) and writes
// page `page` of its slot (bucket rows from `(w * pages + page) *
// buckets_per_block`), so the layers cluster concurrently. Limits: pools
// (cluster_block / pool_size) <= 256, buckets_per_block <= 128.

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
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read>       keys:      array<f32>;
@group(0) @binding(2) var<storage, read_write> scratch:   array<f32>;
@group(0) @binding(3) var<storage, read_write> centroids: array<f32>;
@group(0) @binding(4) var<storage, read_write> members:   array<u32>;

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
var<workgroup> ta:      array<f32, 1024>;     // a 16-pool tile of sums (index_dim <= 64)
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
        scratch[sums + e] = acc;
    }
    for (var i = lt; i < np; i = i + WG) {
        alive[i] = 1u;
        size[i] = p.pool_size;
        parent[i] = i;
    }
    if (lt == 0u) { n_live = np; }
    storageBarrier();
    workgroupBarrier();

    // The Gram matrix of the pool sums, 16 x 16 pool tiles at a time.
    let tiles = (np + 15u) / 16u;
    for (var ti = 0u; ti < tiles; ti = ti + 1u) {
        for (var tj = ti; tj < tiles; tj = tj + 1u) {
            for (var e = lt; e < 16u * d; e = e + WG) {
                let r = e / d;
                let c = e % d;
                let pa = ti * 16u + r;
                let pb = tj * 16u + r;
                ta[r * 64u + c] = select(0.0, scratch[sums + pa * d + c], pa < np);
                tb[r * 64u + c] = select(0.0, scratch[sums + pb * d + c], pb < np);
            }
            workgroupBarrier();
            for (var e = lt; e < 256u; e = e + WG) {
                let i = ti * 16u + e / 16u;
                let j = tj * 16u + e % 16u;
                var g = 0.0;
                for (var c = 0u; c < d; c = c + 1u) {
                    g = g + ta[(e / 16u) * 64u + c] * tb[(e % 16u) * 64u + c];
                }
                if (i < np && j < np) {
                    scratch[gram + i * np + j] = g;
                    scratch[gram + j * np + i] = g;
                }
            }
            workgroupBarrier();
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
                let gii = scratch[gram + i * np + i];
                let ai = sqrt(max(gii, 0.0));
                for (var j = 0u; j < np; j = j + 1u) {
                    if (j != i && alive[j] == 1u && size[i] + size[j] <= p.bucket_cap) {
                        let gjj = scratch[gram + j * np + j];
                        // One triangle: the cost is the same both ways.
                        let gij = scratch[gram + min(i, j) * np + max(i, j)];
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
            scratch[gram + kept * np + j] = scratch[gram + kept * np + j] + scratch[gram + gone * np + j];
        }
        storageBarrier();
        workgroupBarrier();
        for (var e = lt; e < picks * np; e = e + WG) {
            let kept = pick_i[e / np];
            let gone = best[kept];
            let i = e % np;
            scratch[gram + i * np + kept] = scratch[gram + i * np + kept] + scratch[gram + i * np + gone];
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

    // Centroids: the normalised sum of each bucket's keys.
    let page_row = (w * p.pages + p.page) * p.buckets_per_block;
    for (var bkt = lt; bkt < p.buckets_per_block; bkt = bkt + WG) {
        var acc: array<f32, 64>;
        for (var c = 0u; c < d; c = c + 1u) { acc[c] = 0.0; }
        for (var key = 0u; key < p.cluster_block; key = key + 1u) {
            if (best[key / p.pool_size] == bkt) {
                for (var c = 0u; c < d; c = c + 1u) { acc[c] = acc[c] + keys[kbase + key * d + c]; }
            }
        }
        var norm2 = 0.0;
        for (var c = 0u; c < d; c = c + 1u) { norm2 = norm2 + acc[c] * acc[c]; }
        let inv = 1.0 / max(sqrt(norm2), 1e-30);
        for (var c = 0u; c < d; c = c + 1u) {
            centroids[(page_row + bkt) * d + c] = acc[c] * inv;
        }
        // Members in ascending position, padded.
        var n = 0u;
        for (var key = 0u; key < p.cluster_block; key = key + 1u) {
            if (best[key / p.pool_size] == bkt && n < p.bucket_cap) {
                members[(page_row + bkt) * p.bucket_cap + n] = p.base + key;
                n = n + 1u;
            }
        }
        for (var m = n; m < p.bucket_cap; m = m + 1u) {
            members[(page_row + bkt) * p.bucket_cap + m] = NO_KEY;
        }
    }
}
