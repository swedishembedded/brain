#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""Build the brain documentation manual: one self-contained Markdown file and
one PDF, ordered by the navigation in docs/manifest.yaml (the single source of
page order, shared with the published documentation).

Outputs (git-ignored, regenerate on demand):
    build/pdf/brain-docs.md
    build/pdf/brain-docs.pdf

build/docs/ is reserved for the publication built by the docs publisher.
No HTML is produced. Run from the repo root:  python3 docs/pandoc/build-docs.py
Requires pandoc + xelatex and PyYAML.
"""
import hashlib, os, re, sys, subprocess, posixpath
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS = os.path.join(ROOT, "docs")
OUT = os.path.join(ROOT, "build", "pdf")
os.makedirs(OUT, exist_ok=True)

# Used only when the sources are not in a git checkout, so the output stays
# deterministic instead of depending on the wall clock.
FALLBACK_DATE = "Unreleased"

# The navigation part that holds the book (see manifest()).
BOOK_PART = "Book"


def source_date():
    """Month and year of the last commit that touched docs/ (stable per commit)."""
    r = subprocess.run(
        ["git", "log", "-1", "--format=%cd", "--date=format:%B %Y", "--", "docs"],
        cwd=ROOT, capture_output=True, text=True,
    )
    out = r.stdout.strip()
    return out if r.returncode == 0 and out else FALLBACK_DATE


# `author` reaches the PDF title page; `header.tex` puts the same name in the
# footer of every page. brain is Swedish Embedded AB's work and the document
# that gets handed to a reader should say so on its face, the way any other
# published engineering manual does.
AUTHOR = "Swedish Embedded AB"
META = f"""---
title: "brain --- Engineering Documentation"
subtitle: "Pure-Rust GPU machine learning: engine, models, and multi-GPU scaling"
author: "{AUTHOR} --- info@swedishembedded.com"
date: "{source_date()}"
toc-title: "Contents"
---

"""


def manifest():
    """Yields ('part', title) for each top-level navigation group and
    ('doc', rel_path) for every page in it, depth first, in manifest order.

    A doc path is repo-root-relative (e.g. `docs/reference/cli.md` or
    `samples/python/dbus/brain-dbus/README.md`), not docs/-relative - so the
    manual can pull in a sample's own README alongside the docs/ pages."""
    with open(os.path.join(DOCS, "manifest.yaml"), encoding="utf-8") as fh:
        nav = yaml.safe_load(fh)["navigation"]

    def pages(items):
        for item in items:
            if isinstance(item, str):
                yield ("doc", item)
            elif "pages" in item:
                yield from pages(item["pages"])
            else:
                yield ("doc", item["source"])

    for group in nav:
        # The book is published with the docs but typeset as its own volume
        # (scripts/book/pdf.py); the engineering manual leaves it out.
        if group["title"] == BOOK_PART:
            continue
        yield ("part", group["title"])
        yield from pages(group["pages"])

# strip cross-document .md links to their text (no dead links in the bundle),
# keep external links, anchors, and images.
LINK = re.compile(r"(?<!\!)\[([^\]]+)\]\(([^)]+?\.md(?:#[^)]*)?)\)")
IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

def transform(rel_path, text):
    doc_dir = posixpath.dirname(rel_path)  # e.g. docs/models/yolo or samples/python/dbus/brain-dbus
    def img_repl(m):
        alt, src = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "/")):
            return m.group(0)
        # repo-root-relative so pandoc (run from ROOT) resolves it
        new = posixpath.normpath(posixpath.join(doc_dir, src))
        return f"![{alt}]({new})"
    text = IMG.sub(img_repl, text)
    text = LINK.sub(lambda m: m.group(1), text)  # flatten cross-doc links to text
    return text

def build_markdown():
    parts = [META]
    for kind, val in manifest():
        if kind == "part":
            latex_part = val.replace("{", "\\{").replace("}", "\\}")
            parts.append(f"`\\part{{{latex_part}}}`{{=latex}}\n\n")
            continue
        rel = val
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            sys.exit(f"docs/manifest.yaml: missing file: {rel}")
        parts.append(transform(rel, open(p).read()).rstrip() + "\n\n")
    md = os.path.join(OUT, "brain-docs.md")
    open(md, "w").write("".join(parts))
    print(f"wrote {md} ({os.path.getsize(md)//1024} KiB)")
    return md

SVG_LINK = re.compile(r"\]\(([^)\s]+\.svg)\)")
# xelatex passes before giving up on references that keep moving; a build
# whose aux files carry over from the last one usually needs one or two.
MAX_PASSES = 4
RERUN = re.compile(r"Rerun to get|Label\(s\) may have changed|Table widths have changed|There were undefined references")
STATE = (".aux", ".toc", ".out")


def cached_pdf(rel, cache):
    """Where the PDF of the SVG `rel` is kept: named by the hash of its content."""
    src = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
    with open(src, "rb") as f:
        return os.path.join(cache, hashlib.sha256(f.read()).hexdigest()[:20] + ".pdf")


def figures_as_pdf(md, cache):
    """`md` with every SVG figure replaced by a PDF converted once and kept in
    `cache` under the hash of its content, so an unchanged figure is never
    converted again. The conversion is the one pandoc itself runs."""
    os.makedirs(cache, exist_ok=True)
    text = open(md, encoding="utf-8").read()
    converted = {}
    new = sum(1 for rel in set(SVG_LINK.findall(text)) if not os.path.exists(cached_pdf(rel, cache)))
    if new:
        print(f"converting {new} new figure(s) to PDF (kept in {os.path.relpath(cache, ROOT)})", flush=True)

    def to_pdf(m):
        rel = m.group(1)
        if rel not in converted:
            src = rel if os.path.isabs(rel) else os.path.join(ROOT, rel)
            out = cached_pdf(rel, cache)
            if not os.path.exists(out):
                tmp = out + f".{os.getpid()}.tmp"
                subprocess.run(["rsvg-convert", "-f", "pdf", "-a", "--dpi-x", "96", "--dpi-y", "96", "-o", tmp, src],
                               check=True)
                os.replace(tmp, out)
            converted[rel] = out
        return f"]({converted[rel]})"

    text = SVG_LINK.sub(to_pdf, text)
    out_md = os.path.splitext(md)[0] + ".pdf-figures.md"
    open(out_md, "w", encoding="utf-8").write(text)
    return out_md, len(converted)


def state_of(stem):
    return {ext: open(stem + ext, "rb").read() for ext in STATE if os.path.exists(stem + ext)}


def xelatex_pass(work, n, env, label):
    """One xelatex pass in `work`, printing progress every 500 pages; returns
    (exit code, output)."""
    proc = subprocess.Popen(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "input.tex"],
                            cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    out, shown = [], 0
    for line in proc.stdout:
        out.append(line)
        for page in re.findall(r"\[(\d+)", line):
            if int(page) >= shown + 500:
                shown = int(page) // 500 * 500
                print(f"{label}: pass {n}, page {shown}", flush=True)
    return proc.wait(), "".join(out)


def build_pdf(md, pdf=None, header=None, extra=()):
    """Typeset `md` with pandoc + xelatex. Shared with the book's PDF build,
    which passes its own output, header and document-class options.

    The stages run here rather than inside pandoc so that each one reports
    and none repeats work: figures are converted once (`figures_as_pdf`),
    pandoc writes LaTeX, and xelatex runs in a work directory kept between
    builds, whose aux, toc and outline files let a rebuild settle in one or
    two passes instead of starting from nothing every time."""
    pdf = pdf or os.path.join(OUT, "brain-docs.pdf")
    out_dir = os.path.dirname(pdf)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(pdf)[0]
    label = os.path.basename(stem)
    log = stem + ".pandoc.log"
    work = stem + ".tex.d"
    os.makedirs(work, exist_ok=True)
    md, figures = figures_as_pdf(md, os.path.join(os.path.dirname(OUT), "figures-pdf"))
    cmd = [
        "pandoc", md, "-o", os.path.join(work, "input.tex"), "--standalone",
        "--toc", "--number-sections",
        "--highlight-style=tango",
        "-H", header or os.path.join(DOCS, "pandoc", "header.tex"),
        "--resource-path", f"{ROOT}:{DOCS}",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=10pt",
        "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue", "-V", "toccolor=black",
        *(extra or ("--toc-depth=2", "--top-level-division=chapter", "-V", "documentclass=report")),
    ]
    stamp = stem + ".inputs.sha256"
    key = inputs_digest(md, cmd)
    if os.path.exists(pdf) and os.path.exists(log) and os.path.exists(stamp) and open(stamp).read() == key:
        print(f"{label}: up to date", flush=True)
        return pdf, overfull_widths(open(log, encoding="utf-8", errors="replace").read())
    if os.path.exists(stamp):
        os.remove(stamp)
    print(f"{label}: {figures} figures ready, writing LaTeX", flush=True)
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        open(log, "w").write(r.stdout + "\n---STDERR---\n" + r.stderr)
        print("pandoc FAILED; tail of output:", file=sys.stderr)
        print((r.stdout + r.stderr)[-2000:], file=sys.stderr)
        sys.exit(1)
    env = dict(os.environ, TEXMFVAR=os.path.join(out_dir, "texmf"))
    # LaTeX's log of every pass, marked as pandoc marks them, so a reader of
    # the log (the book's overflow report) sees the same layout as before.
    runs, fresh = [], False
    n = 0
    while n < MAX_PASSES:
        n += 1
        before = state_of(os.path.join(work, "input"))
        code, out = xelatex_pass(work, n, env, label)
        runs.append(f"[makePDF] LaTeX run number {n}\n{out}")
        if code != 0 and not fresh:
            # A build killed mid-pass leaves a truncated aux file that stops
            # the next one at \begin{document}: start this build clean once.
            for ext in STATE:
                if os.path.exists(os.path.join(work, "input" + ext)):
                    os.remove(os.path.join(work, "input" + ext))
            fresh, n = True, 0
            print(f"{label}: pass failed on carried-over state, starting clean", flush=True)
            continue
        if code != 0:
            break
        pages = re.search(r"Output written on input\.pdf \((\d+) pages", out)
        settled = state_of(os.path.join(work, "input")) == before and not RERUN.search(out)
        print(f"{label}: pass {n} done, {pages.group(1) if pages else '?'} pages"
              + ("" if settled else ", references moved, another pass"), flush=True)
        if settled:
            break
    else:
        print(f"{label}: references still moving after {MAX_PASSES} passes; see {log}", file=sys.stderr)
    text = "".join(runs)
    open(log, "w").write(text + "\n---STDERR---\n" + r.stderr)
    if code != 0:
        print("xelatex FAILED; tail of output:", file=sys.stderr)
        print(text[-2000:], file=sys.stderr)
        sys.exit(1)
    os.replace(os.path.join(work, "input.pdf"), pdf)
    print(f"wrote {pdf} ({os.path.getsize(pdf)//1024} KiB)")
    bad = overfull_widths(text)
    print(f"overfull hboxes >2pt: {len(bad)}" + (f"  (max {max(bad):.1f}pt)" if bad else ""))
    # Written last, so a build that stops early is never taken as up to date.
    open(stamp, "w").write(key)
    return pdf, bad


def overfull_widths(log):
    """Widths of the lines wider than the text (xelatex logs 'Overfull \\hbox
    (N.NNpt too wide)'), each place once although every pass repeats it,
    ignoring hairlines under 2pt."""
    overfull = set(re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\) (.*)", log))
    return [float(w) for w, _ in overfull if float(w) > 2.0]


IMAGE_PATH = re.compile(r"(/[^\s{}()\[\]`'\"]+\.(?:jpe?g|png|pdf))")


def inputs_digest(md, cmd):
    """A hash of everything a PDF is made from: the markdown (whose figure
    links name their converted PDFs by content), every file the pandoc command
    reads (headers, filters), the images those name by path, this program,
    and the pandoc and xelatex versions. Equal digests mean an equal PDF."""
    h = hashlib.sha256()
    h.update("\0".join(cmd[2:]).encode())
    h.update(open(__file__, "rb").read())
    for tool in ("pandoc", "xelatex"):
        h.update(subprocess.run([tool, "--version"], capture_output=True, text=True).stdout.split("\n")[0].encode())
    files = [md] + [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a in ("-H", "--lua-filter")]
    for f in files:
        text = open(f, "rb").read()
        h.update(f.encode() + b"\0" + text)
        for img in sorted(set(IMAGE_PATH.findall(text.decode("utf-8", "replace")))):
            if os.path.isfile(img):
                h.update(img.encode() + b"\0" + open(img, "rb").read())
    return h.hexdigest()

if __name__ == "__main__":
    md = build_markdown()
    pdf, bad = build_pdf(md)
    if bad:
        print("WARNING: some content overflows the page; see build/pdf/brain-docs.pandoc.log", file=sys.stderr)
