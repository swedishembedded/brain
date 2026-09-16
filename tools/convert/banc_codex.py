#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Convert the BANC v888 Dataverse release into the Codex CSV schema brain reads.

BANC - brain AND nerve cord in one volume, 188,508 neurons - is published two
ways. Codex serves it behind a Google sign-in and a click-through licence, so
it is not scriptable and `tools/buzzfly/collect.sh` cannot fetch it. Harvard
Dataverse serves the same v888 snapshot under CC BY 4.0 with NO login
(doi:10.7910/DVN/7WTH1N), which is the route this uses - but it publishes
Arrow, and `crates/connectome` reads Codex CSV precisely so that the engine
needs no Arrow in its load path. So the conversion happens here, once, by hand,
exactly as `flybody_walking_reference.py` does for HDF5.

WHAT IT WRITES, AND WHY THE THIRD FILE IS THE POINT

Two files are the Codex schema and load through the existing reader with no
new code:

    neurons.csv.gz                one row per neuron, annotation included
    connections_princeton.csv.gz  one row per (pre, post) pair

The third is the reason this exists at all:

    bridge_manc.csv.gz            BANC neuron <-> MANC neuron, per neuron

BANC's own metadata carries a `manc_match` column: for 1,268 of its 1,316
descending neurons and most of its ascending ones, the PUBLISHED identity of
the same cell in MANC, as a MANC root id in the same id space as the MANC
export's own `Root ID`. That is what makes "BANC's brain drives MANC's cord" a
wiring rather than a guess - the alternative, matching on cell-type NAME, is a
heuristic that silently fails on the types whose names differ between
datasets. The bridge is written as its own file because `manc_match` is not a
Codex column and inventing one would make this output no longer be the Codex
schema.

WHY BOTH DATASETS AND NOT JUST BANC

BANC contains a nerve cord too, and it would be simpler to use it alone.
Measured, it is four to five times more sparsely reconstructed than MANC's on
its own territory (in-degree onto motor neurons: 110 against MANC's 224), so
the cord that drives the legs stays MANC and the brain that decides where to
go is BANC. The seam between them is this file.

    pip install pyarrow pandas
    tools/convert/banc_codex.py RAWDIR OUTDIR

RAWDIR holds the two Dataverse files, downloadable without an account:

    curl -L -o banc_888_meta.feather \\
      'https://dataverse.harvard.edu/api/access/datafile/14033740'
    curl -L -o banc_888_edgelist_simple_v3.feather \\
      'https://dataverse.harvard.edu/api/access/datafile/13918810'
"""

import argparse
import csv
import gzip
import pathlib
import sys

import pandas as pd

# The Codex neuron schema, verbatim and in order. Taken from the MANC export
# this workspace already reads rather than from documentation, because the
# reader matches on these exact strings.
NEURON_HEADER = [
    "Root ID",
    "Top in/out region",
    "Community labels",
    "Predicted NT type",
    "Predicted NT confidence",
    "Verified NT type",
    "Verified Neuropeptide",
    "Body Part",
    "Function",
    "Flow",
    "Super Class",
    "Class",
    "Sub Class",
    "Hemilineage",
    "Nerve",
    "Soma side",
    "Primary Cell Type",
    "Alternative Cell Type(s)",
    "Cable length (nm)",
    "Surface area (nm^2)",
    "Volume (nm^3)",
]

EDGE_HEADER = ["pre_root_id", "post_root_id", "neuropil", "syn_count", "nt_type"]

BRIDGE_HEADER = [
    "banc_root_id",
    "manc_root_id",
    "flow",
    "banc_cell_type",
    "manc_cell_type",
    # BANC's independent morphology-based match. Carried because `manc_match`
    # alone is MANY-TO-ONE on 391 MANC cells - a curator recording one
    # exemplar for a type with several members - and where the two columns
    # agree, that is the member the curated match was about.
    "manc_nblast_match",
]

# Codex spells neurotransmitters in upper case; BANC's metadata spells them out.
NT = {
    "acetylcholine": "ACH",
    "gaba": "GABA",
    "glutamate": "GLUT",
    "dopamine": "DA",
    "serotonin": "SER",
    "octopamine": "OCT",
    "histamine": "HA",
    "tyramine": "TYR",
    "unknown": "",
}


def text(v):
    """A cell as Codex writes it: empty rather than the string 'nan'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    s = str(v).strip()
    return "" if s in ("nan", "None", "<NA>") else s


def nt_name(v):
    s = text(v).lower()
    return NT.get(s, s.upper())


def write_neurons(meta, out):
    """One Codex row per BANC neuron."""
    kept = 0
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(NEURON_HEADER)
        for r in meta.itertuples(index=False):
            rid = text(r.root_888)
            if not rid:
                continue
            # Volume is what BANC populates and what `Connectome::size` reads;
            # the surface-area column is written empty, which is exactly the
            # state MANC's own export is in and which the import gate asserts
            # per dataset in both directions.
            w.writerow(
                [
                    rid,
                    text(r.region),
                    "",
                    nt_name(r.neurotransmitter_predicted),
                    text(r.neurotransmitter_score),
                    nt_name(r.neurotransmitter_verified),
                    text(r.neuropeptide_verified),
                    text(r.body_part_effector) or text(r.body_part_sensory),
                    text(r.cell_function),
                    text(r.flow),
                    text(r.super_class),
                    text(r.cell_class),
                    text(r.cell_sub_class),
                    text(r.hemilineage),
                    text(r.nerve),
                    text(r.side),
                    text(r.cell_type),
                    "",
                    "",
                    "",
                    text(r.volume_nm3),
                ]
            )
            kept += 1
    return kept


def write_edges(edges, known, out):
    """One row per (pre, post) pair, dropping endpoints with no annotation.

    Codex's own file is per (pre, post, NEUROPIL) and the reader aggregates
    across neuropils; this release is already aggregated, so every pair is
    written once with an empty neuropil. That is a real difference from the
    MANC file and it is visible rather than hidden: the reader's aggregation
    step becomes a no-op, and the edge and synapse totals it reports are the
    same either way.
    """
    pre = edges["pre"].astype(str).to_numpy()
    post = edges["post"].astype(str).to_numpy()
    cnt = edges["count"].to_numpy()
    kept = dropped = 0
    syn = 0
    with gzip.open(out, "wt", newline="") as fh:
        fh.write(",".join(EDGE_HEADER) + "\n")
        buf = []
        for p, q, c in zip(pre, post, cnt):
            if p not in known or q not in known:
                dropped += 1
                continue
            buf.append(f"{p},{q},,{c},\n")
            kept += 1
            syn += int(c)
            if len(buf) >= 65536:
                fh.write("".join(buf))
                buf.clear()
        fh.write("".join(buf))
    return kept, dropped, syn


def write_bridge(meta, out):
    """The published BANC <-> MANC identity, for the neurons that cross."""
    cross = meta[meta.super_class.isin(["descending", "ascending", "sensory_ascending", "sensory_descending"])]
    rows = 0
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(BRIDGE_HEADER)
        for r in cross.itertuples(index=False):
            manc = text(r.manc_match)
            if not manc or not manc.isdigit():
                continue
            w.writerow(
                [
                    text(r.root_888),
                    manc,
                    text(r.super_class),
                    text(r.cell_type),
                    text(r.manc_cell_type),
                    text(r.manc_nblast_match),
                ]
            )
            rows += 1
    return rows, len(cross)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=pathlib.Path, help="directory holding the two Dataverse .feather files")
    ap.add_argument("out", type=pathlib.Path, help="directory to write the Codex-shaped export into")
    ap.add_argument("--min-synapses", type=int, default=1, help="drop pairs below this synapse count (1 keeps all)")
    a = ap.parse_args()

    meta_path = a.raw / "banc_888_meta.feather"
    edge_path = a.raw / "banc_888_edgelist_simple_v3.feather"
    for p in (meta_path, edge_path):
        if not p.is_file():
            sys.exit(f"{p} is missing; see this script's header for the two curl commands")

    a.out.mkdir(parents=True, exist_ok=True)
    print(f"reading {meta_path.name}")
    meta = pd.read_feather(meta_path)
    # A neuron with no id cannot be an endpoint of anything, and `not_a_neuron`
    # rows are glia and trachea the reconstruction carries deliberately.
    meta = meta[meta.root_888.notna() & (meta.super_class != "not_a_neuron")]
    print(f"  {len(meta)} annotated neurons")

    n = write_neurons(meta, a.out / "neurons.csv.gz")
    print(f"  wrote neurons.csv.gz: {n} rows")

    print(f"reading {edge_path.name}")
    edges = pd.read_feather(edge_path)
    if a.min_synapses > 1:
        edges = edges[edges["count"] >= a.min_synapses]
    # Ids are STRINGS in both files and are kept that way: they are 18-digit
    # segment ids, they are only ever compared and written, and a round trip
    # through int64 is a conversion this has no reason to perform.
    known = set(meta.root_888.astype(str).tolist())
    kept, dropped, syn = write_edges(edges, known, a.out / "connections_princeton.csv.gz")
    print(f"  wrote connections_princeton.csv.gz: {kept} edges, {syn} synapses ({dropped} rows had an unannotated endpoint)")

    rows, cross = write_bridge(meta, a.out / "bridge_manc.csv.gz")
    print(f"  wrote bridge_manc.csv.gz: {rows} of {cross} crossing neurons carry a published MANC match")

    (a.out / "SOURCE.txt").write_text(
        "BANC v888, converted from the Harvard Dataverse release by\n"
        "tools/convert/banc_codex.py.\n"
        "\n"
        "Source:  doi:10.7910/DVN/7WTH1N, CC BY 4.0, no account required.\n"
        "  banc_888_meta.feather              datafile 14033740\n"
        "  banc_888_edgelist_simple_v3.feather datafile 13918810\n"
        "\n"
        "The Codex copy of the same dataset is behind a Google sign-in and is\n"
        "not scriptable; this route is, and carries the same licence.\n"
        "\n"
        "bridge_manc.csv.gz is NOT a Codex file. It carries BANC's own\n"
        "`manc_match` column - the published identity of each crossing neuron\n"
        "in MANC - which is what lets BANC's brain drive MANC's cord through\n"
        "the neurons that actually cross, rather than through a name match.\n"
        "`manc_nblast_match` comes along because `manc_match` is many-to-one\n"
        "on 391 MANC cells and the morphology match breaks most of those ties.\n"
    )
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
