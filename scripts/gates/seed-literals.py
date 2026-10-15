#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for reproducible, auditable
# randomness in ML pipelines for its clients. If your team needs expertise in
# seeding discipline for training and generation then you can procure our
# services by sending an email to info@swedishembedded.com.

"""List every literal seed in the tree as `path<TAB>symbol<TAB>kind<TAB>context<TAB>reasoned|bare`.

Rule: every default seed is random. A seed that is a literal in the source is
therefore either a salt, a world constant, a frozen-artifact seed read back, a
RNG whose output is never consulted, a test double or a benchmark input - or a
bug. This module is the single detector behind check-seed-literals.sh, which
ratchets the production hits against scripts/gates/seed-literals.allow.d/.

Detected, in tracked and untracked-but-not-ignored files (the walk is used for
trees without a .git, i.e. the gate fixtures):

  Rust (*.rs)   kind `default`: `seed: 1`, `seed = 1`, `seed: u64 = 1`,
                `seed: Some(1)`, `seed: SOME_CONST`, `("--seed", 1)` (any
                `_or`/`number`/`parse` flag reader), `seed ... .unwrap_or(1)`,
                `map_or(Ok(1)` on a seed line, `default_value_t = 1` on a
                `seed` field, `.default(json!(1))` of a `"seed"` param,
                `.seed(1)` / `.with_seed(1)`;
                kind `const`: `const FOO_SEED: u64 = 1`;
                kind `rng`: `Rng::new(1)`, `Lcg::new(1)`, `seed_from_u64(1)`.
  Python (*.py) kind `python-default`: `add_argument("--...seed...", default=1)`.
  Makefile      kind `make-default`: `SEED ?= 1`; `make-flag`: a variable
                assigned a value containing `--seed 1`.

Context is `prod` or `test`. Test context is: a `tests/` or `benches/`
directory, a `#[cfg(test)]` module (including the file pulled in by
`#[cfg(test)] mod x;`), a `#[test]` function, a Python `test_*.py`.

A symbol is `<enclosing item>:<field>=<literal>`, so entries survive line
moves and a changed literal is a new entry. Hits that repeat the same symbol
in one item collapse to one row.

Output: path, symbol, kind, context (prod|test), and `reasoned`/`bare`: whether
every occurrence carries a `// seed:` (or `# seed:`) comment on its line or the
line above.

usage: seed-literals.py [--root DIR] [--lines]
"""
import argparse
import bisect
import os
import re
import subprocess
import sys
from pathlib import Path

INT = r"-?(?:0[xX][0-9a-fA-F_]+|\d[\d_]*)(?:_?[iu](?:8|16|32|64|128|size))?"
SKIP_DIRS = {"target", ".git", "node_modules", "out", "build", "results"}
SKIP_PREFIXES = ("scripts/gates/testdata/",)
REASON = re.compile(r"(?://|#)\s*seed:")

FIELD_INIT = re.compile(r"\b(\w*seed\w*)\s*:\s*(" + INT + r")(?![\w.])", re.I)
FIELD_SOME = re.compile(r"\b(\w*seed\w*)\s*:\s*Some\(\s*(" + INT + r")\s*\)", re.I)
FIELD_CONST = re.compile(r"\b((?i:\w*seed\w*))\s*:\s*([A-Z][A-Z0-9_]{2,})\b(?!\s*[:(])")
ASSIGN = re.compile(
    r"\b(\w*seed\w*)\s*(?::\s*[\w:<>]+\s*)?(?<![=!<>])=(?!=)\s*(" + INT + r")(?![\w.])", re.I
)
FLAG_READ = re.compile(r'\(\s*"(?:--)?([\w-]*seed[\w-]*)"\s*,\s*(' + INT + r")\s*\)", re.I)
UNWRAP_OR = re.compile(r"\bunwrap_or\(\s*(" + INT + r")\s*\)")
MAP_OR_OK = re.compile(r"\bmap_or\(\s*Ok\(\s*(" + INT + r")\s*\)")
DEFAULT_VALUE_T = re.compile(r"\bdefault_value_t\s*=\s*(" + INT + r")")
PARAM_NEW = re.compile(r'ParamSpec::new\(\s*"([^"]*)"')
DEFAULT_JSON = re.compile(r"\.default\(\s*json!\(\s*(" + INT + r")\s*\)\s*\)")
SEED_CALL = re.compile(r"\.(\w*seed\w*)\(\s*(" + INT + r")\s*\)", re.I)
CONST_SEED = re.compile(
    r"\b(?:const|static)\s+(\w*SEED\w*)\s*:[^=]*=\s*[\[(]?\s*(" + INT + r")", re.I
)
RNG_CTOR = re.compile(
    r"\b(\w*(?:Rng|Lcg|SplitMix\d*|Xorshift\w*|Pcg\w*))::(new|seed_from_u64|with_seed|seeded)"
    r"\(\s*(" + INT + r")"
)
SEED_WORD = re.compile(r"\bseed", re.I)
SEED_NAME_ON_LINE = re.compile(r"(\w*seed\w*)\s*:", re.I)

BLOCK_CHARS = re.compile(r"[{};]")
CANDIDATE_NEEDLES = ("seed", "rng::", "lcg::", "splitmix", "xorshift", "pcg", "default_value_t", ".default(json!")
CFG_TEST_ATTR = re.compile(
    r"#\[\s*(?:cfg\s*\(\s*(?:test|all\s*\(\s*test\b)|test\b|bench\b|[\w:]*::test\b|rstest\b)"
)
CFG_TEST_FILE = re.compile(r"#!\[\s*cfg\s*\(\s*test\s*\)\s*\]")
CFG_TEST_MOD_DECL = re.compile(
    r"#\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*;"
)


SPECIAL = re.compile(r"""//[^\n]*|/\*|"|'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]+\}|.)|[^\\'])'""")
RAW_PREFIX = re.compile(r"(?<![\w])b?r(#*)$")
BLOCK_COMMENT_EDGE = re.compile(r"/\*|\*/")
STRING_END = re.compile(r'(?:\\.|[^"\\])*"', re.S)


class _Blanker(dict):
    """str.translate table: every character becomes a space except those kept."""

    def __init__(self, keep: str):
        super().__init__({ord(c): ord(c) for c in keep})

    def __missing__(self, key):
        return 32


BLANK = _Blanker("\n")
BLANK_STRING = _Blanker('\n"')


def blank(text: str) -> str:
    return text.translate(BLANK)


def strip_rust(text: str):
    """Per line: (code view with strings blanked, view with strings intact).

    Comments are blanked in both, so braces and semicolons that remain in the
    code view are never inside a string, a comment or a char literal.
    """
    code, intact = [], []
    pos, n = 0, len(text)
    while pos < n:
        m = SPECIAL.search(text, pos)
        if not m:
            code.append(text[pos:])
            intact.append(text[pos:])
            break
        code.append(text[pos : m.start()])
        intact.append(text[pos : m.start()])
        tok = m.group(0)
        if tok.startswith("//"):
            end = m.end()
            code.append(blank(tok))
            intact.append(blank(tok))
        elif tok == "/*":
            depth, end = 1, m.end()
            while depth and end < n:
                nxt = BLOCK_COMMENT_EDGE.search(text, end)
                if not nxt:
                    end = n
                    break
                depth += 1 if nxt.group(0) == "/*" else -1
                end = nxt.end()
            code.append(blank(text[m.start() : end]))
            intact.append(blank(text[m.start() : end]))
        elif tok == '"':
            raw = RAW_PREFIX.search(text, max(0, m.start() - 40), m.start())
            if raw:
                close = '"' + raw.group(1)
                stop = text.find(close, m.end())
                end = n if stop < 0 else stop + len(close)
            else:
                sm = STRING_END.match(text, m.end())
                end = sm.end() if sm else n
            body = text[m.start() : end]
            code.append(body.translate(BLANK_STRING))
            intact.append(body)
        else:  # char literal
            end = m.end()
            code.append("'" + " " * (len(tok) - 2) + "'")
            intact.append(tok)
        pos = end
    return "".join(code).split("\n"), "".join(intact).split("\n")


def header_item(header: str):
    """Classify the statement text that precedes a `{`: (kind, name) or None."""
    m = re.search(r"\bfn\s+(\w+)", header)
    if m:
        return ("fn", m.group(1))
    m = re.search(r"\b(?:const|static)\s+(\w+)", header)
    if m and m.group(1) != "fn":
        return ("const", m.group(1))
    m = re.search(r"\bimpl\b(.*)", header, re.S)
    if m:
        rest = m.group(1).strip()
        if rest.startswith("<"):
            depth = 0
            for idx, c in enumerate(rest):
                depth += c == "<"
                depth -= c == ">"
                if depth == 0:
                    rest = rest[idx + 1 :].strip()
                    break
        if re.search(r"\bfor\b", rest):
            rest = re.split(r"\bfor\b", rest, maxsplit=1)[1]
        names = re.findall(r"[A-Za-z_]\w*", rest.split("where")[0])
        names = [x for x in names if x not in ("dyn", "mut", "const")]
        if names:
            return ("impl", names[0] if "::" not in rest else names[-1])
    m = re.search(r"\b(trait|struct|enum|union)\s+(\w+)", header)
    if m:
        return ("impl", m.group(2))
    m = re.search(r"\bmod\s+(\w+)", header)
    if m:
        return ("mod", m.group(1))
    return None


class RustFile:
    """Walks one Rust file, yielding (line_no, symbol, kind, in_test) hits."""

    def __init__(self, rel: str, text: str, file_is_test: bool):
        self.rel = rel
        self.raw = text.splitlines()
        self.code, self.intact = strip_rust(text)
        self.file_is_test = file_is_test or bool(CFG_TEST_FILE.search("\n".join(self.code)))

    def hits(self):
        candidates = self.candidate_lines()
        stack = []  # (kind, name, is_test)
        header = ""
        pending_test = False
        for idx, code in enumerate(self.code):
            line_no = idx + 1
            attr_here = "#[" in code and bool(CFG_TEST_ATTR.search(code))
            in_test = (
                self.file_is_test
                or any(entry[2] for entry in stack)
                or pending_test
                or attr_here
            )
            symbol_stack = list(stack)
            for found in self.line_hits(idx) if idx in candidates else ():
                yield (line_no, self.enclosing(symbol_stack, header), found, in_test)
            # advance the block structure past this line
            if attr_here:
                pending_test = True
            last = 0
            for m in BLOCK_CHARS.finditer(code):
                header += code[last : m.start()]
                last = m.end()
                ch = m.group(0)
                if ch == "{":
                    item = header_item(header)
                    stack.append((item[0] if item else None, item[1] if item else None, pending_test))
                    pending_test = False
                elif ch == "}":
                    if stack:
                        stack.pop()
                else:
                    pending_test = False
                header = ""
            header += code[last:]
            header += "\n"

    def candidate_lines(self):
        """Indices of the lines any detector could match (substring scans, no regex)."""
        joined = "\n".join(self.intact)
        lowered = joined.lower()
        starts = [0]
        pos = joined.find("\n")
        while pos >= 0:
            starts.append(pos + 1)
            pos = joined.find("\n", pos + 1)
        found = set()
        for needle in CANDIDATE_NEEDLES:
            pos = lowered.find(needle)
            while pos >= 0:
                found.add(bisect.bisect_right(starts, pos) - 1)
                pos = lowered.find(needle, pos + 1)
        return found

    @staticmethod
    def enclosing(stack, header):
        owner = next((n for k, n, _ in stack if k == "impl"), None)
        inner = next((n for k, n, _ in reversed(stack) if k in ("fn", "const")), None)
        if inner is None:
            item = header_item(header)
            if item and item[0] == "const":
                inner = item[1]
        parts = [p for p in (owner, inner) if p]
        return "::".join(dict.fromkeys(parts)) if parts else "-"

    def line_hits(self, idx):
        """(field=literal, kind) for every pattern matching line `idx`."""
        code = self.code[idx]
        intact = self.intact[idx]
        out = []
        decl = CONST_SEED.search(code)
        if decl:
            return [(f"{decl.group(1)}={decl.group(2)}", "const")]
        for m in FIELD_INIT.finditer(code):
            out.append((f"{m.group(1)}={m.group(2)}", "default"))
        for m in FIELD_SOME.finditer(code):
            out.append((f"{m.group(1)}=Some({m.group(2)})", "default"))
        for m in FIELD_CONST.finditer(code):
            out.append((f"{m.group(1)}={m.group(2)}", "default"))
        for m in ASSIGN.finditer(code):
            out.append((f"{m.group(1)}={m.group(2)}", "default"))
        for m in FLAG_READ.finditer(intact):
            out.append((f"--{m.group(1)}={m.group(2)}", "default"))
        if SEED_WORD.search(intact):
            for m in UNWRAP_OR.finditer(code):
                out.append((f"unwrap_or={m.group(1)}", "default"))
            for m in MAP_OR_OK.finditer(code):
                out.append((f"map_or={m.group(1)}", "default"))
        for m in DEFAULT_VALUE_T.finditer(code):
            window = " ".join(self.code[idx : idx + 4])
            field = SEED_NAME_ON_LINE.search(window.split("default_value_t", 1)[1]) or SEED_NAME_ON_LINE.search(code)
            if field and re.search("seed", field.group(1), re.I):
                out.append((f"{field.group(1)}={m.group(1)}", "default"))
        for m in DEFAULT_JSON.finditer(intact):
            before = "\n".join(self.intact[max(0, idx - 6) : idx] + [intact[: m.start()]])
            owners = PARAM_NEW.findall(before)
            if owners and re.search("seed", owners[-1], re.I):
                out.append((f"default={m.group(1)}", "default"))
        for m in SEED_CALL.finditer(code):
            out.append((f".{m.group(1)}({m.group(2)})", "default"))
        for m in RNG_CTOR.finditer(code):
            out.append((f"{m.group(1)}::{m.group(2)}({m.group(3)})", "rng"))
        return list(dict.fromkeys(out))


def justified(raw_lines, line_no):
    here = raw_lines[line_no - 1] if line_no - 1 < len(raw_lines) else ""
    above = raw_lines[line_no - 2] if line_no >= 2 else ""
    return bool(REASON.search(here) or REASON.search(above))


def python_hits(text: str):
    """(line_no, flag, literal) for `add_argument("--*seed*", ..., default=<int>)`."""
    for m in re.finditer(r"add_argument\(", text):
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += text[j] == "("
            depth -= text[j] == ")"
            j += 1
        call = text[m.end() : j]
        flag = re.search(r"""["'](--[\w-]*seed[\w-]*)["']""", call, re.I)
        default = re.search(r"\bdefault\s*=\s*(" + INT + r")(?![\w.])", call)
        if flag and default:
            yield text.count("\n", 0, m.start()) + 1, flag.group(1), default.group(1)


MAKE_VAR = re.compile(r"^\s*(?:export\s+)?(\w*SEED\w*)\s*(?:\?=|:=|=)\s*(" + INT + r")\s*(?:#.*)?$", re.I)
MAKE_FLAG = re.compile(r"^\s*(\w+)\s*(?:\?=|:=|=)\s*.*?(--[\w-]*seed[\w-]*)[ =](" + INT + r")(?![\w.])", re.I)


def makefile_hits(text: str):
    for no, line in enumerate(text.splitlines(), 1):
        if line.startswith("\t"):
            continue
        m = MAKE_VAR.match(line)
        if m:
            yield no, f"{m.group(1)}={m.group(2)}", "make-default"
            continue
        m = MAKE_FLAG.match(line)
        if m:
            yield no, f"{m.group(1)}:{m.group(2)}={m.group(3)}", "make-flag"


def tracked_files(root: Path):
    if (root / ".git").exists():
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True,
            check=True,
        )
        names = [n for n in proc.stdout.decode().split("\0") if n]
    else:
        names = []
        for dirpath, dirnames, filenames in os.walk(root):
            top = Path(dirpath) == root
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not (top and d == "testdata")]
            names += [Path(dirpath, f).relative_to(root).as_posix() for f in filenames]
    return sorted(n for n in names if (root / n).is_file() and not n.startswith(SKIP_PREFIXES))


def path_is_test(rel: str) -> bool:
    parts = rel.split("/")
    return "tests" in parts[:-1] or "benches" in parts[:-1] or bool(re.match(r"test_.*\.py$|.*_test\.py$", parts[-1]))


def scan(root: Path):
    """All hits as (rel, symbol, kind, context, line_no, justified)."""
    files = tracked_files(root)
    rust = {n: (root / n).read_text(encoding="utf-8", errors="replace") for n in files if n.endswith(".rs")}
    test_files = set()
    for rel, text in rust.items():
        base = Path(rel)
        for mod in CFG_TEST_MOD_DECL.findall(text) if "cfg(test)" in text else ():
            dirs = [base.parent] if base.name in ("lib.rs", "main.rs", "mod.rs") else [base.parent / base.stem]
            for d in dirs:
                for cand in (d / f"{mod}.rs", d / mod / "mod.rs"):
                    test_files.add(cand.as_posix())
    rows = []
    for rel, text in rust.items():
        lowered = text.lower()
        if not any(needle in lowered for needle in CANDIDATE_NEEDLES):
            continue
        file_test = path_is_test(rel) or rel in test_files
        f = RustFile(rel, text, file_test)
        for line_no, enclosing, (field, kind), in_test in f.hits():
            rows.append((rel, f"{enclosing}:{field}", kind, "test" if in_test else "prod", line_no, justified(f.raw, line_no)))
    for rel in files:
        text = None
        if rel.endswith(".py"):
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            raw = text.splitlines()
            for no, flag, lit in python_hits(text):
                rows.append((rel, f"-:{flag}={lit}", "python-default", "test" if path_is_test(rel) else "prod", no, justified(raw, no)))
        elif Path(rel).name in ("Makefile", "makefile") or rel.endswith(".mk"):
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            raw = text.splitlines()
            for no, symbol, kind in makefile_hits(text):
                rows.append((rel, f"-:{symbol}", kind, "prod", no, justified(raw, no)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--lines", action="store_true", help="append line numbers (diagnostics)")
    args = ap.parse_args()
    seen = {}
    for rel, symbol, kind, context, line_no, ok in scan(Path(args.root)):
        entry = seen.setdefault((rel, symbol, kind, context), [True, []])
        entry[0] &= ok
        entry[1].append(line_no)
    for (rel, symbol, kind, context), (ok, lines) in sorted(seen.items()):
        extra = "\t" + ",".join(map(str, lines)) if args.lines else ""
        sys.stdout.write(f"{rel}\t{symbol}\t{kind}\t{context}\t{'reasoned' if ok else 'bare'}{extra}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
