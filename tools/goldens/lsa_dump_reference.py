#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Dump latent sparse attention reference goldens for brain's `model::lsa`.

Swedish Embedded AB implements long-context attention that is proven against
the method's own reference before it is optimised, for its clients. If your
team needs expertise in sparse attention, you can procure our services by
sending an email to info@swedishembedded.com.

The reference is the authors' own torch implementation
(github.com/Prefrontal-AI/latent-sparse-attention, Apache-2.0,
`src/lsa/core/reference.py`), imported from a checkout given by `--lsa-src`
and run unmodified: `build_buckets` clusters the closed blocks of fixed index
keys, and `read` selects each query's keys. Nothing here re-implements it.

The keys are drawn around a handful of directions plus noise, so Ward has real
clusters to find rather than only near-ties between random vectors.

Written under `--out` (default `testdata/lsa`):

  flat_reference.safetensors
    config            [8]  pool_size, bucket_size, bucket_cap, cluster_block,
                           top_p, top_k, sink, window
    keys              [T, d] unit index keys
    queries           [Q, d] unit index queries
    positions         [Q]
    bucket_centroids  [rows, d]
    bucket_members    [rows, bucket_cap]  -1 padded
    selected          [Q, top_k + window + sink]  -1 padded
    lengths           [Q]

Everything is f32 (brain's safetensors reader is F32/F16/BF16-only); every
integer here is below 2^24 and exact. Fixed seed, CPU only.

Usage:
  python3 tools/goldens/lsa_dump_reference.py --lsa-src <checkout>/src
"""

import argparse
import json
import pathlib
import struct
import sys


def save_safetensors(path: pathlib.Path, tensors: dict) -> None:
    """Write f32 tensors in the safetensors layout: header length, JSON header, raw data."""
    header, blobs, offset = {}, [], 0
    for name, tensor in tensors.items():
        data = tensor.contiguous().float().numpy().tobytes() if hasattr(tensor, "numpy") else bytes(tensor)
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for blob in blobs:
            f.write(blob)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lsa-src", required=True, help="the reference checkout's src/ directory")
    parser.add_argument("--out", default="testdata/lsa")
    args = parser.parse_args()
    sys.path.insert(0, args.lsa_src)

    import torch
    from lsa.core.reference import build_buckets, read

    torch.manual_seed(20261030)
    index_dim, keys_count, query_count = 64, 4 * 512 + 100, 64
    pool_size, bucket_size, bucket_cap, cluster_block = 4, 8, 16, 512
    top_p, top_k, sink, window = 24, 96, 4, 512

    centers = torch.nn.functional.normalize(torch.randn(12, index_dim), dim=1)
    pick = torch.randint(0, centers.shape[0], (keys_count,))
    keys = torch.nn.functional.normalize(centers[pick] + 0.35 * torch.randn(keys_count, index_dim), dim=1)
    queries = torch.nn.functional.normalize(torch.randn(query_count, index_dim), dim=1)
    # Every block boundary's neighbourhood, plus positions spread over the rest.
    edges = [0, 1, 511, 512, 513, 1023, 1024, 1535, 1536, 2047, 2048, keys_count - 1]
    spread = torch.randint(0, keys_count, (query_count - len(edges),)).tolist()
    positions = torch.tensor(edges + spread, dtype=torch.int64)

    buckets = build_buckets(keys, pool_size, bucket_size, bucket_cap, cluster_block)
    selected, lengths = read(queries, positions, buckets, top_p, top_k, sink, window)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = torch.tensor([pool_size, bucket_size, bucket_cap, cluster_block, top_p, top_k, sink, window])
    save_safetensors(
        out / "flat_reference.safetensors",
        {
            "config": config,
            "keys": keys,
            "queries": queries,
            "positions": positions,
            "bucket_centroids": buckets.bucket_centroids,
            "bucket_members": buckets.bucket_members,
            "selected": selected,
            "lengths": lengths,
        },
    )
    print(f"wrote {out / 'flat_reference.safetensors'} (torch {torch.__version__})")


if __name__ == "__main__":
    main()
