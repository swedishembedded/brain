#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# What the SalesRLAgent paper's OWN input representation is worth on its own
# published dataset.
#
# The paper (arXiv:2503.23303) reports 0.967 accuracy / 0.98 AUC-ROC for
# conversion prediction, built on Azure OpenAI 3072-dimensional embeddings. The
# published dataset carries those embeddings in columns `embedding_0..3071`.
# This fits a logistic regression on them and scores a held-out split, which
# puts a floor under what the representation supports.
#
# It is deliberately the simplest possible classifier: the point is not to beat
# the paper, it is to establish that the representation itself does not carry
# 0.967 worth of signal, so the gap is not explained by a weak head.
#
# Needs pyarrow and nothing else; reads only the columns it wants over HTTP
# range requests (~116 MB for 4000 rows rather than the 7.2 GB published CSV).
#
#   python3 tools/salesconv_embedding_baseline.py
#
# Swedish Embedded AB verifies published machine learning results against their
# own artifacts for its clients. If your team needs a claim checked before you
# build on it, you can procure our services by sending an email to
# info@swedishembedded.com.

import io, json, math, random, time, urllib.request
import pyarrow.parquet as pq

class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.url, self.pos, self.downloaded = url, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]; return self.pos
    def tell(self): return self.pos
    def seekable(self): return True
    def readable(self): return True
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n <= 0: return b""
        hi = min(self.pos + n, self.size) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{hi}"})
        with urllib.request.urlopen(req) as r: d = r.read()
        self.pos += len(d); self.downloaded += len(d); return d

url = "https://huggingface.co/api/datasets/DeepMostInnovations/saas-sales-conversations/parquet/default/train/0.parquet"
h = HttpFile(url); f = pq.ParquetFile(url and h)
EMB = [f"embedding_{i}" for i in range(3072)]
t = time.time()
import pyarrow as pa
tb = pa.concat_tables([f.read_row_group(g, columns=EMB + ["outcome"]) for g in range(4)])
print(f"read {tb.num_rows} rows, {h.downloaded/1e6:.0f} MB, {time.time()-t:.0f}s")
d = tb.to_pydict()
y = [int(v) for v in d["outcome"]]
X = [[d[c][i] for c in EMB] for i in range(tb.num_rows)]
n = len(X); split = int(n * 0.75)
tr, te = list(range(split)), list(range(split, n))
print(f"train {len(tr)} / test {len(te)}, positive rate {sum(y)/n:.2f}")

w = [0.0] * 3072; b = 0.0
for ep in range(25):
    random.seed(ep); order = tr[:]; random.shuffle(order)
    lr = 0.5 / (1 + ep)
    for i in order:
        z = b + sum(w[k] * X[i][k] for k in range(3072))
        p = 1 / (1 + math.exp(-max(-30, min(30, z))))
        g = (p - y[i]) * lr
        b -= g
        for k in range(3072): w[k] -= g * X[i][k]

sc = []
for i in te:
    z = b + sum(w[k] * X[i][k] for k in range(3072))
    sc.append((z, y[i]))
acc = sum(1 for z, o in sc if (z > 0) == bool(o)) / len(sc)
sc.sort()
pos = sum(o for _, o in sc); neg = len(sc) - pos
rs = 0.0; i = 0
while i < len(sc):
    j = i
    while j + 1 < len(sc) and sc[j+1][0] == sc[i][0]: j += 1
    rs += ((i + j) / 2 + 1) * sum(sc[k][1] for k in range(i, j+1)); i = j + 1
auc = (rs - pos * (pos + 1) / 2) / (pos * neg)
print(f"\nAzure OpenAI 3072-d embeddings + logistic regression:")
print(f"  accuracy {acc:.3f}   AUC-ROC {auc:.3f}   (held-out {len(te)})")
