#!/usr/bin/env bats
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for publishing technical books in
# print and electronic formats from one verified source. If your team needs
# expertise in e-book pipelines for engineering content then you can procure
# our services by sending an email to info@swedishembedded.com.

# Spec for the book's EPUB build (scripts/book/epub.py with
# docs/pandoc/book-epub.lua and book-math-lines.lua): figures are numbered
# per chapter, a chapter's own "**Figure N.**" paragraph becomes the caption
# instead of a second label, tables scroll, raw HTML is reported, the filter
# reports how many formulas, tables and images it read, and the structural
# verifier fails a formula left as TeX, a count that does not match, and a
# link to an anchor that does not exist. Long display formulas break where a
# typesetter would break them (arrows, relations, side-by-side equations,
# never inside plain parentheses or braces) and keep their equation numbers.
# Each case builds a small EPUB from markdown under the test's own
# directory, never the book.

setup() {
  REPO="$(cd "$(dirname "$BATS_TEST_FILENAME")/../.." && pwd)"
  WORK="$BATS_TEST_TMPDIR"
  printf '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>\n' \
    >"$WORK/dot.svg"
}

# Build $WORK/book.md into $WORK/book.epub with the book's filters, in the
# order scripts/book/epub.py runs them; their report goes to $WORK/stderr.
build() {
  (cd "$WORK" && pandoc book.md -f markdown -t epub3 --mathml --split-level=2 \
    --metadata title=Test --lua-filter "$REPO/docs/pandoc/book-math-lines.lua" \
    --lua-filter "$REPO/docs/pandoc/book-figures.lua" --lua-filter "$REPO/docs/pandoc/book-epub.lua" \
    -o book.epub 2>stderr)
}

# The structural verifier (everything but epubcheck), with the source counts
# the filter reported, so a mismatch shows as a problem.
structure() {
  local want="${WANT:-}"
  [ -n "$want" ] || want='{}'
  run python3 - "$REPO/scripts/book" "$WORK/book.epub" "$WORK/stderr" "$want" <<'PY'
import json, re, sys
sys.path.insert(0, sys.argv[1])
import epub
from pathlib import Path
m = re.search(r"book-epub: counts math=(\d+) table=(\d+) image=(\d+)", Path(sys.argv[3]).read_text())
want = dict(zip(("Math", "Table", "Image"), map(int, m.groups())))
want.update(json.loads(sys.argv[4]))
sys.argv[1:] = sys.argv[2:]
for p in epub.structure(Path(sys.argv[1]), want):
    print(p)
PY
}

page() {
  unzip -p "$WORK/book.epub" 'EPUB/text/ch*.xhtml'
}

@test "a clean book passes and the filter reports what it read" {
  cat >"$WORK/book.md" <<'MD'
# A chapter {.chapter}

Inline $x^2$ and display $$\frac{a}{b}$$

| a | b |
|---|---|
| 1 | 2 |

![a dot](dot.svg)
MD
  build
  grep -q 'book-epub: counts math=2 table=1 image=1' "$WORK/stderr"
  structure
  [ "$status" -eq 0 ]
  [ -z "$output" ]
  # A narrow table is left as it is, and nothing is put in a scrolling box.
  ! page | grep -q 'table-fit\|table-scroll'
}

@test "a table too wide for a phone line is set smaller, never below 60 percent" {
  {
    printf '# One {.chapter}\n\n'
    printf '| %s |\n' "$(printf 'aaaaaaaaaaaa | %.0s' 1 2 3 4 5 6 7 8)" | sed 's/ | |$/ |/'
    printf '|%s\n' "$(printf -- '---|%.0s' 1 2 3 4 5 6 7 8)"
    printf '| %s |\n' "$(printf '1 | %.0s' 1 2 3 4 5 6 7 8)" | sed 's/ | |$/ |/'
  } >"$WORK/book.md"
  build
  html="$(page)"
  [[ "$html" == *'<div class="table-fit" style="font-size: '*'%">'* ]]
}

@test "figures are numbered per chapter and a chapter's own caption replaces the alt text" {
  cat >"$WORK/book.md" <<'MD'
# One {.chapter}

![first, described](dot.svg)

![second, described](dot.svg)

**Figure 2.** The chapter's own caption.

# Two {.chapter}

![third, described](dot.svg)
MD
  build
  html="$(page)"
  [[ "$html" == *"<strong>Figure 1.</strong> first, described"* ]]
  [[ "$html" == *"<strong>Figure 2.</strong> The chapter’s own caption."* ]]
  [[ "$html" != *"<strong>Figure 2.</strong> second"* ]]
  # The count starts again in the next chapter.
  [[ "$html" == *"<strong>Figure 1.</strong> third, described"* ]]
  [ "$(grep -o 'Figure 2\.' <<<"$html" | wc -l)" -eq 1 ]
}

@test "a caption numbered out of order is reported" {
  cat >"$WORK/book.md" <<'MD'
# One {.chapter}

![only, described](dot.svg)

**Figure 3.** Misnumbered.
MD
  build
  grep -q 'book-epub: figure captioned 3 is figure 1 of its chapter' "$WORK/stderr"
}

@test "a formula pandoc cannot convert to MathML fails" {
  printf '# One {.chapter}\n\nRow $p \\bmod$ the size.\n' >"$WORK/book.md"
  build
  structure
  [[ "$output" == *"formula left as TeX source"* ]]
}

@test "an element missing from the EPUB fails the count" {
  printf '# One {.chapter}\n\n$x$\n' >"$WORK/book.md"
  build
  WANT='{"Math": 2}' structure
  [[ "$output" == *"2 math element(s) in the source, 1 in the EPUB"* ]]
}

@test "a link to an anchor that does not exist fails" {
  printf '# One {.chapter}\n\nSee [nowhere](#no-such-section).\n' >"$WORK/book.md"
  build
  structure
  [[ "$output" == *"names an anchor that does not exist"* ]]
}

@test "an angle-bracketed word that markdown takes for HTML is reported" {
  printf '# One {.chapter}\n\nThe generator run<i> of <method>.\n' >"$WORK/book.md"
  build
  grep -q 'book-epub: raw HTML <i>' "$WORK/stderr"
  grep -q 'book-epub: raw HTML <method>' "$WORK/stderr"
}

# The line breaker of docs/pandoc/book-math-lines.lua on one formula.
broken() {
  run pandoc lua - "$REPO/docs/pandoc/book-math-lines.lua" "$1" <<'LUA'
dofile(arg[1])
io.write(BookMathLines.break_display(arg[2]))
LUA
}

@test "a short formula is left as written" {
  broken 'a = b + c'
  [ "$output" = 'a = b + c' ]
}

@test "a long chain breaks before its implication arrows, aligned on them" {
  broken '0 \le \lVert\mathbf{r}\rVert^2 = \lVert\mathbf{v}\rVert^2 - \frac{(\mathbf{v}\cdot\mathbf{u})^2}{\lVert\mathbf{u}\rVert^2} \;\Rightarrow\; (\mathbf{v}\cdot\mathbf{u})^2 \le \lVert\mathbf{v}\rVert^2\lVert\mathbf{u}\rVert^2 \;\Rightarrow\; |\mathbf{v}\cdot\mathbf{u}| \le \lVert\mathbf{v}\rVert\,\lVert\mathbf{u}\rVert'
  [[ "$output" == '\begin{aligned}0 &\le '* ]]
  [[ "$output" == *' \\ &\Rightarrow\; (\mathbf{v}'* ]]
  [[ "$output" == *'\end{aligned}' ]]
}

@test "equations set side by side with \\qquad go on rows of their own" {
  broken '\mu_j = \frac1B\sum_{i} X_{ij}, \qquad \sigma_j^2 = \frac1B\sum_i (X_{ij} - \mu_j)^2, \qquad \hat X_{ij} = \frac{X_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}'
  [[ "$output" == '\begin{gathered}'* ]]
  [[ "$output" == *' \\ \hat X_{ij} = '* ]]
}

@test "a line never breaks inside plain parentheses or braces" {
  broken '\alpha_t(s) = \bigl(\alpha_{t-1}(s) + \alpha_{t-1}(s-1) + [\,y_s \ne \mathrm{blank}\,]\;\alpha_{t-1}(s-2)\bigr)\,P[t, y_s] + \frac{a + b + c + d}{e + f + g + h}'
  [[ "$output" == *'(s-2)'* ]]
  [[ "$output" == *'(s-1)'* ]]
  [[ "$output" == *'P[t, y_s]'* ]]
  [[ "$output" == *'\frac{a + b + c + d}{e + f + g + h}'* ]]
}

@test "an equation number survives as (N) at the end of the last row" {
  broken 'a = b \tag{7}'
  [ "$output" = 'a = b \qquad\text{(7)}' ]
  broken 'c_x = (2 o_x - 0.5 + g_x)\, s, \quad c_y = (2 o_y - 0.5 + g_y)\, s, \quad w = (2 o_w)^2 a_w, \quad h = (2 o_h)^2 a_h, \tag{2}'
  [[ "$output" == *'\qquad\text{(2)}\end{gathered}' ]]
}

@test "a long row of an aligned environment is split at its side-by-side equations" {
  broken '\begin{aligned} \Delta &= \frac{\mathrm{hi} - \mathrm{lo}}{255}, \qquad q_0 = \mathrm{clamp}\bigl(\mathrm{round}(-\mathrm{lo}/\Delta), 0, 255\bigr), \\ q &= x. \end{aligned}'
  [[ "$output" == *'\\ q_0 &= \mathrm{clamp}'* ]]
}

@test "a long product breaks between two factors, the next row starting with an explicit times" {
  broken 'M_{ij} = \sum_{k=1}^{3} w_k \,\sin\!\Big(\frac{k\pi(i + \tfrac12)}{8}\Big) \cos\!\Big(\frac{k\pi(j + \tfrac12)}{8}\Big) \cos\!\Big(\frac{k\pi(j + \tfrac12)}{8}\Big)'
  [[ "$output" == *' \\ &\quad \times \cos\!\Big('* ]]
  # A function applied to its argument is never split from it.
  [[ "$output" != *'\sin\! \\'* ]]
}

# scripts/book/epub.py's linking of "Volume II, Chapter 1" references, for
# the book's own pages, in the whole book or one volume.
references() {
  run python3 - "$REPO/scripts/book" "$1" "$2" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
import edition, epub
from bookspec import load
book = load()
ed = edition.whole(book) if sys.argv[2] == "whole" else edition.volume(book, book.volumes[11])
print(epub.references("v12-reinforcement-learning/01-mdps.md", sys.argv[3], set(ed.pages()), book))
PY
}

@test "a chapter reference is a link in the whole book, and in a volume only when the chapter is in it" {
  references whole 'See Volume II, Chapter 1 and Chapter 2; `Chapter 3` stays code.'
  [[ "$output" == *'[Volume II, Chapter 1](#page-v02-learning-01-learning-problem)'* ]]
  [[ "$output" == *'[Chapter 2](#page-v12-reinforcement-learning-02-value-learning)'* ]]
  [[ "$output" == *'`Chapter 3` stays code'* ]]
  references volume 'See Volume II, Chapter 1 and Chapter 2.'
  [[ "$output" == *'See Volume II, Chapter 1 and [Chapter 2]'* ]]
}

@test "a formula with no place to break is set smaller to fit, never below 60 percent" {
  cd "$WORK"
  printf '%s\n' '$$\boxed{\;X[k] = E[k] + \omega^{k}O[k], \qquad X[k + N/2] = E[k] - \omega^{k}O[k], \qquad k = 0, \dots, N/2 - 1\;}$$' '' '$$a = b$$' >f.md
  run pandoc f.md -t html5 --mathml --lua-filter "$REPO/docs/pandoc/book-math-lines.lua"
  [ "$(grep -c 'class="math-scaled"' <<<"$output")" -eq 1 ]
  [[ "$output" == *'font-size: 60%'* ]]
}
