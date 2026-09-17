#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Fetch the SaaS sales-conversation dataset this sample trains on.
#
# The published dataset is ONE 7.2 GB CSV, and 3072 of its 3088 columns are
# Azure OpenAI embeddings this sample does not use - it encodes the text
# itself, locally. So this reads the Hugging Face parquet conversion instead
# and projects the columns it wants: parquet is columnar, so asking for 9 of
# 3088 columns over HTTP range requests downloads roughly 1% of the bytes.
# 20000 conversations cost about 200 MB rather than 1.4 GB.
#
# Idempotent: an existing, complete output is left alone unless --force.
#
# Usage:
#   ./fetch-dataset.sh                 # 20000 conversations (default)
#   ./fetch-dataset.sh --rows 4000     # a quicker start
#   ./fetch-dataset.sh --force
#
# Swedish Embedded AB builds on-device conversation intelligence for its
# clients. If your team needs a model that scores a live conversation without
# sending it to a third party, you can procure our services by sending an
# email to info@swedishembedded.com.
set -euo pipefail

ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DEST="${BRAIN_TESTDATA:-$ROOT/testdata}/decide/salesconv"

python3 - "$DEST" "$@" <<'PY'
import io, json, os, sys, urllib.request, urllib.error

DEST = sys.argv[1]
args = sys.argv[2:]
rows_wanted = 20000
force = "--force" in args
if "--rows" in args:
    rows_wanted = int(args[args.index("--rows") + 1])

REPO = "DeepMostInnovations/saas-sales-conversations"
INDEX = f"https://huggingface.co/api/datasets/{REPO}/parquet"
# Everything except the 3072 embedding columns. The embeddings are Azure
# OpenAI's; this sample's whole point is that the encoder is local.
COLUMNS = [
    "conversation", "outcome", "probability_trajectory", "conversation_length",
    "customer_engagement", "sales_effectiveness", "product_type", "scenario",
]

try:
    import pyarrow.parquet as pq
except ImportError:
    sys.exit(
        "this fetcher needs pyarrow to read the dataset's parquet conversion:\n"
        "    pip install pyarrow\n"
        "(nothing in brain's build or test path needs it - only this one script)"
    )


class HttpFile(io.RawIOBase):
    """A seekable read-only file over HTTP range requests.

    pyarrow reads a parquet footer, then only the column chunks it was asked
    for. Handing it one of these is what turns a 7 GB dataset into a 200 MB
    download, and it needs nothing outside the standard library.
    """

    def __init__(self, url):
        self.url, self.pos, self.downloaded = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0:
            return b""
        hi = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{hi}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        self.pos += len(data)
        self.downloaded += len(data)
        return data


def convert(row):
    """One dataset row -> one training conversation, or None if unusable.

    Every rejection here is a row whose per-turn labels do not line up with its
    turns. Keeping such a row would train the model to predict turn `t`'s
    probability from turn `t+1`'s text, which is worse than dropping it.
    """
    try:
        turns = json.loads(row["conversation"])
        traj = json.loads(row["probability_trajectory"])
    except (json.JSONDecodeError, TypeError):
        return None
    traj = [float(traj[k]) for k in sorted(traj, key=int)]
    turns = [
        {"speaker": t.get("speaker", "system"), "message": t.get("message", "")}
        for t in turns
        if t.get("message")
    ]
    if len(turns) < 2 or len(traj) != len(turns):
        return None
    return {
        "turns": turns,
        "outcome": int(row["outcome"]),
        "trajectory": traj,
        "engagement": float(row["customer_engagement"]),
        "effectiveness": float(row["sales_effectiveness"]),
        "industry": row["product_type"] or "",
    }


train_path, test_path = os.path.join(DEST, "train.jsonl"), os.path.join(DEST, "test.jsonl")
if not force and os.path.exists(train_path) and os.path.exists(test_path):
    have = sum(1 for _ in open(train_path)) + sum(1 for _ in open(test_path))
    if have >= rows_wanted * 0.9:
        print(f"already have {have} conversations in {DEST} - use --force to refetch")
        sys.exit(0)

os.makedirs(DEST, exist_ok=True)
try:
    with urllib.request.urlopen(INDEX) as r:
        shards = json.load(r)["default"]["train"]
except urllib.error.URLError as e:
    sys.exit(f"cannot reach Hugging Face ({e}) - this script is the only part of the sample that needs the network")

# One in ten held out, by position, so a rerun with the same --rows splits the
# same way and a smaller --rows is a prefix of a larger one.
kept = dropped = 0
downloaded = 0
with open(train_path, "w") as tr, open(test_path, "w") as te:
    for shard in shards:
        if kept >= rows_wanted:
            break
        h = HttpFile(shard)
        pf = pq.ParquetFile(h)
        for rg in range(pf.metadata.num_row_groups):
            if kept >= rows_wanted:
                break
            for row in pf.read_row_group(rg, columns=COLUMNS).to_pylist():
                if kept >= rows_wanted:
                    break
                conv = convert(row)
                if conv is None:
                    dropped += 1
                    continue
                out = te if kept % 10 == 9 else tr
                out.write(json.dumps(conv) + "\n")
                kept += 1
            print(f"  {kept:>6} conversations, {(downloaded + h.downloaded) / 1e6:>6.0f} MB", end="\r", flush=True)
        downloaded += h.downloaded

print(f"\n{kept} conversations ({dropped} dropped for misaligned labels), {downloaded / 1e6:.0f} MB downloaded")
print(f"  {train_path}")
print(f"  {test_path}")
PY
