#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Build the brain documentation bundle: one self-contained Markdown file and one
professional PDF, from the docs listed in docs/manifest.txt.

Outputs (git-ignored, regenerate on demand):
    build/docs/brain-docs.md
    build/docs/brain-docs.pdf

No HTML is produced. Run from the repo root:  python3 docs/pandoc/build-docs.py
Requires pandoc + xelatex.
"""
import os, re, sys, subprocess, posixpath

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(ROOT, "build", "docs")
os.makedirs(OUT, exist_ok=True)

DATE = "July 2026"
META = f"""---
title: "brain --- Engineering Documentation"
subtitle: "Pure-Rust GPU machine learning: engine, models, and multi-GPU scaling"
date: "{DATE}"
toc-title: "Contents"
---

"""

def manifest():
    for line in open(os.path.join(DOCS, "manifest.txt")):
        line = line.strip()
        if line and not line.startswith("#"):
            yield line

# strip cross-document .md links to their text (no dead links in the bundle),
# keep external links, anchors, and images.
LINK = re.compile(r"(?<!\!)\[([^\]]+)\]\(([^)]+?\.md(?:#[^)]*)?)\)")
IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

def transform(rel_path, text):
    doc_dir = posixpath.dirname(rel_path)  # e.g. models/yolo
    def img_repl(m):
        alt, src = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "/")):
            return m.group(0)
        # repo-root-relative so pandoc (run from ROOT) resolves it
        new = posixpath.normpath(posixpath.join("docs", doc_dir, src))
        return f"![{alt}]({new})"
    text = IMG.sub(img_repl, text)
    text = LINK.sub(lambda m: m.group(1), text)  # flatten cross-doc links to text
    return text

def build_markdown():
    parts = [META]
    for rel in manifest():
        p = os.path.join(DOCS, rel)
        if not os.path.isfile(p):
            print(f"WARN missing: {rel}", file=sys.stderr)
            continue
        parts.append(transform(rel, open(p).read()).rstrip() + "\n\n")
    md = os.path.join(OUT, "brain-docs.md")
    open(md, "w").write("".join(parts))
    print(f"wrote {md} ({os.path.getsize(md)//1024} KiB)")
    return md

def build_pdf(md):
    pdf = os.path.join(OUT, "brain-docs.pdf")
    log = os.path.join(OUT, "pandoc.log")
    cmd = [
        "pandoc", md, "-o", pdf,
        "--pdf-engine=xelatex",
        "--toc", "--toc-depth=2", "--number-sections",
        "--top-level-division=chapter",
        "--highlight-style=tango",
        "-H", os.path.join(DOCS, "pandoc", "header.tex"),
        "--resource-path", f"{ROOT}:{DOCS}",
        "-V", "documentclass=report",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=10pt",
        "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue", "-V", "toccolor=black",
        "--pdf-engine-opt=-interaction=nonstopmode",
    ]
    env = dict(os.environ, TEXMFVAR=os.path.join(OUT, "texmf"))
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    open(log, "w").write(r.stdout + "\n---STDERR---\n" + r.stderr)
    if r.returncode != 0:
        print("pandoc FAILED; tail of output:", file=sys.stderr)
        print((r.stdout + r.stderr)[-2000:], file=sys.stderr)
        sys.exit(1)
    print(f"wrote {pdf} ({os.path.getsize(pdf)//1024} KiB)")
    # Report overflow: xelatex logs 'Overfull \hbox (N.NNpt too wide)'.
    overfull = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", r.stdout + r.stderr)
    bad = [float(x) for x in overfull if float(x) > 2.0]  # ignore < 2pt hairline
    print(f"overfull hboxes >2pt: {len(bad)}" + (f"  (max {max(bad):.1f}pt)" if bad else ""))
    return pdf, bad

if __name__ == "__main__":
    md = build_markdown()
    pdf, bad = build_pdf(md)
    if bad:
        print("WARNING: some content overflows the page; see build/docs/pandoc.log", file=sys.stderr)
