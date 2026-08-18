#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The one writer of a golden manifest's `source` block.

A golden dump is a set of tensors plus a claim - "this is what the reference
produced" - and that claim only means something together with WHICH checkpoint
produced it. That half was never recorded, so a dump from one tier could be
paired with another tier's weights and the suite would either die deep in the
importer with a tensor-shape error, or compare against the wrong reference and
certify a number that meant nothing, or print to stderr and return, which cargo
reports as a pass. All three have happened here.

So every `*_dump_reference.py` writes this block into its `manifest.json`:

    "source": {
      "checkpoint": "NeoQuasar/Kronos-small",
      "files": {"model.safetensors": "sha256:9f86d0..."},
      "identity": {"d_model": 512, "n_layers": 12}
    }

`identity` is the enforced field, and it is deliberately not a path or a name.
A path resolves on one machine; a name is a label a dumper can get wrong.
`identity` is the architectural config that FIXES EVERY TENSOR SHAPE in the
dump - width, depth, head count, vocab - which is precisely what the reading
test can recompute from the checkpoint it is about to compare against, and
precisely what two tiers of one architecture cannot agree on.

`files` and `checkpoint` are forensics: they name the artifact to go look at
once a mismatch is reported. They are not compared, because the same weights
legitimately arrive under several names and paths.

The Rust side that enforces it is `brain_testutil::golden::Source`.

Usage inside a dumper:

    from golden_source import source_block
    manifest["source"] = source_block(
        checkpoint="NeoQuasar/Kronos-small",
        files=[os.path.join(args.decoder, "model.safetensors")],
        identity={"d_model": cfg.d_model, "n_layers": cfg.n_layers},
    )
"""

import hashlib
import os

__all__ = ["source_block", "sha256_of"]


def sha256_of(path, chunk=1 << 22):
    """`sha256:<hex>` of a file, streamed - a checkpoint does not fit in RAM
    twice, and a dumper that OOMs recording provenance helps nobody."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return "sha256:" + h.hexdigest()


def source_block(identity, checkpoint=None, files=(), hash_files=True):
    """Build the `source` block.

    identity     dict of config fields that determine the dumped tensors'
                 shapes. REQUIRED and must be non-empty: a block without one
                 records nothing a test can check, which is the state this
                 whole convention exists to leave.
    checkpoint   the upstream reference the dumper was pointed at
                 ("<vendor>/<repo>"), informational.
    files        paths to the weight files actually read, recorded by BASENAME
                 (an absolute path is machine-specific and would leak one
                 machine's layout into a committed manifest).
    hash_files   set False to record the file names without reading them, for
                 a dumper over weights so large that hashing dominates its
                 runtime. The identity still carries the enforced half.
    """
    if not identity:
        raise ValueError(
            "source_block(identity=...) must name at least one shape-determining "
            "config field; an empty identity is what this block exists to replace"
        )
    ident = {}
    for k, v in identity.items():
        if isinstance(v, bool) or not isinstance(v, (int,)):
            # Only integers are enforced on the Rust side, because only they
            # compare exactly. A float tolerance would put the tier check back
            # into judgement, which is where it went wrong.
            raise ValueError(f"source_block identity[{k!r}] must be an int, got {type(v).__name__}")
        ident[k] = int(v)

    block = {"identity": ident}
    if checkpoint:
        block["checkpoint"] = str(checkpoint)
    if files:
        block["files"] = {
            os.path.basename(p): (sha256_of(p) if hash_files else None) for p in files
        }
    return block
