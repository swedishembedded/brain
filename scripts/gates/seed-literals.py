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
                kind `rng`: `Rng::new(1)`, `Lcg::new(1)`, `seed_from_u64(1)`,
                `Mt19937::new(1)`, `ChaCha8Rng::..`, `Philox..::new(1)`;
                kind `default` also for `fn name() -> Seed { Seed::new(1) }`
                (a zero-argument function that returns a literal and mentions
                seed in its name or return type, or is the target of a
                `#[serde(default = "..")]` on a seed field), for
                `for seed in 1..n` and `(1..=n).map(|seed| ..)`;
                kind `positional`: a literal, or a constant that is not a seed
                constant itself, passed at the index of a parameter named
                `*seed*` (type u64, i64, u128 or Seed). The callee is resolved
                within its crate; a name defined twice with different
                parameter lists is skipped, and so is a call whose argument
                count differs from the definition. Test code is reported only
                with --positional-tests.
  Python (*.py) kind `python-default`: `add_argument("--...seed...", default=1)`,
                `seed[: int] = 1` as a parameter, field, keyword or local,
                `"seed": 1` dict entries, `x if x is not None else 1` and
                `x or 1` on a seed name, and `environ.get("..SEED..", "1")`;
                kind `python-const`: a module-level `SEED = 1` / `NOISE_SEED = 1`;
                kind `python-rng`: `torch.manual_seed(1)`,
                `torch.Generator().manual_seed(1)`, `np.random.seed(1)`,
                `random.seed(1)`, `default_rng(1)`, `RandomState(1)`,
                `random.Random(1)`.
  Makefile      kind `make-default`: `SEED ?= 1`; `make-flag`: a variable
                assigned a value containing `--seed 1`, or a recipe line with
                `--seed 1` (symbol `<target>:--seed=1`).
  Shell (*.sh, *.bats)
                kind `shell-seed`: `--seed 1`, `--noise-seed=1`,
                `${SEED:-1}`, `SEED=1`, and `seq 1 3` feeding a seed.
  Markdown (*.md)
                kind `doc-seed`: `--seed 1` inside a fenced block unless the
                word "reproduce" is on the line, a neighbouring line or the
                prose line above the fence.

Context is `prod` or `test`. Test context is: a `tests/` or `benches/`
directory, a `#[cfg(test)]` module (including the file pulled in by
`#[cfg(test)] mod x;`), a `#[test]` function, a Python `test_*.py`.

A symbol is `<enclosing item>:<field>=<literal>`, so entries survive line
moves and a changed literal is a new entry. Hits that repeat the same symbol
in one item collapse to one row.

Output: path, symbol, kind, context (prod|test), and `reasoned`/`bare`: whether
every occurrence carries a `// seed:` (or `# seed:`) comment on its line or the
line above.

A `// seed:` reason that starts with "any" does not count: it says the value
does not matter, which is a reason to draw it.

usage: seed-literals.py [--root DIR] [--lines] [--positional-tests]
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
SKIP_PREFIXES = (
    "scripts/gates/testdata/",
    "scripts/gates/seed-literals.py",
    "tests/gates/seed_literals.bats",
)
# A reason must say why this stream is needed; "any value" says the opposite.
REASON = re.compile(r"(?://|#)\s*seed:[ \t]*(?![ \t]|any\b)", re.I)

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
    r"\b(\w*(?:Rng|Lcg|SplitMix\d*|Xorshift\w*|Pcg\w*)|Mt19937\w*|ChaCha\w*|Philox\w*)"
    r"::(new|seed_from_u64|with_seed|seeded)"
    r"\(\s*(" + INT + r")"
)
FOR_SEED = re.compile(r"\bfor\s+(\w*seed\w*)\s+in\s+\(?\s*(" + INT + r")\s*\.\.", re.I)
RANGE_MAP_SEED = re.compile(
    r"\(\s*(" + INT + r")\s*\.\.=?[^()]*\)\s*\.\s*(?:map|for_each|flat_map|filter_map)\s*\(\s*(?:move\s*)?\|\s*(\w*seed\w*)\s*\|",
    re.I,
)
FN_ZERO_ARG = re.compile(
    r"\bfn\s+(\w+)\s*\(\s*\)\s*->\s*([^{;]+?)\s*\{\s*(?:[\w:]*\(\s*(" + INT + r")\s*\)|(" + INT + r"))\s*\}"
)
SERDE_SEED_DEFAULT = re.compile(
    r'#\[\s*serde\s*\([^\]]*?\bdefault\s*=\s*"(\w+)"[^\]]*\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?\w*seed\w*\s*:',
    re.I,
)
FN_DEF = re.compile(r"\bfn\s+(\w+)\s*(?:<[^()]*?>)?\s*\(")
SEED_PARAM_TYPE = re.compile(r"^(?:u64|i64|u128|Seed)$")
POSITIONAL_CONST = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")
SEED_WORD = re.compile(r"\bseed", re.I)
SEED_NAME_ON_LINE = re.compile(r"(\w*seed\w*)\s*:", re.I)

BLOCK_CHARS = re.compile(r"[{};]")
CANDIDATE_NEEDLES = (
    "seed", "rng::", "lcg::", "splitmix", "xorshift", "pcg", "mt19937", "chacha", "philox",
    "default_value_t", ".default(json!",
)
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


NOT_NEWLINE = re.compile(r"[^\n]")
NOT_NEWLINE_OR_QUOTE = re.compile(r'[^\n"]')


def blank(text: str) -> str:
    """Every character but a newline becomes a space, so offsets and lines survive."""
    return NOT_NEWLINE.sub(" ", text) if "\n" in text else " " * len(text)


def blank_string(text: str) -> str:
    """Like `blank`, but double quotes survive."""
    if len(text) >= 2 and text[0] == text[-1] == '"' and '"' not in text[1:-1] and "\n" not in text:
        return '"' + " " * (len(text) - 2) + '"'
    return NOT_NEWLINE_OR_QUOTE.sub(" ", text)


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
            code.append(blank_string(body))
            intact.append(body)
        else:  # char literal
            end = m.end()
            code.append("'" + " " * (len(tok) - 2) + "'")
            intact.append(tok)
        pos = end
    return "".join(code).split("\n"), "".join(intact).split("\n")


RNG_TYPE_BEFORE = re.compile(r"(?:Rng|Lcg|SplitMix\d*|Xorshift\w*|Pcg\w*|Mt19937\w*|ChaCha\w*|Philox\w*)$")
FN_KEYWORD_BEFORE = re.compile(r"\bfn\s+$")
SELF_PARAM = re.compile(r"^\s*(?:&\s*(?:'\w+\s+)?)?(?:mut\s+)?self\b")


_CALL_PATTERNS = {}


def seeded_call_pattern(seeded: dict):
    """One alternation over the callee names of a crate, so a file is scanned once."""
    key = id(seeded)
    if key not in _CALL_PATTERNS:
        names = "|".join(sorted(map(re.escape, seeded), key=len, reverse=True))
        _CALL_PATTERNS[key] = re.compile(r"\b(" + names + r")\s*\(")
    return _CALL_PATTERNS[key]


def split_args(text: str, start: int, generics: bool = False):
    """The top-level comma-separated pieces between the `(` before `start` and
    its `)`, or None when the parenthesis never closes. Angle brackets nest
    only in a parameter list (`generics`), where they are types; in a call they
    are comparisons. `->` never nests."""
    depth, piece, out = 1, start, []
    pos = start
    while pos < len(text):
        ch = text[pos]
        if ch == "-" and text[pos : pos + 2] == "->":
            pos += 2
            continue
        if ch in "([{" or (generics and ch == "<"):
            depth += 1
        elif ch in ")]}" or (generics and ch == ">"):
            depth -= 1
            if depth == 0:
                tail = text[piece:pos]
                if tail.strip():
                    out.append(tail)
                return out
        elif ch == "," and depth == 1:
            out.append(text[piece:pos])
            piece = pos + 1
        pos += 1
    return None


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
        self.joined = "\n".join(self.code)
        self.file_is_test = file_is_test or bool(CFG_TEST_FILE.search(self.joined))
        self.extra = {}  # line index -> [(field, kind)] found by whole-file scans
        self._starts = [0]
        for m in re.finditer("\n", self.joined):
            self._starts.append(m.end())

    def line_of(self, offset: int) -> int:
        return bisect.bisect_right(self._starts, offset) - 1

    def add_extra(self, offset: int, field: str, kind: str):
        self.extra.setdefault(self.line_of(offset), []).append((field, kind))

    def hidden_defaults(self):
        """Zero-argument functions that return a literal seed: the default is
        hidden behind a call, so no `seed: 1` is ever written."""
        serde = set(SERDE_SEED_DEFAULT.findall("\n".join(self.intact)))
        for m in FN_ZERO_ARG.finditer(self.joined):
            name, ret = m.group(1), m.group(2)
            if name in serde or SEED_WORD.search(name) or SEED_WORD.search(ret):
                self.add_extra(m.start(), f"fn {name}={m.group(3) or m.group(4)}", "default")

    def signatures(self):
        """(name, seed parameter index or None, parameter count, has self) per `fn`;
        the index and count exclude a `self` receiver."""
        for m in FN_DEF.finditer(self.joined):
            params = split_args(self.joined, m.end(), generics=True)
            if params is None:
                continue
            has_self = bool(params) and bool(SELF_PARAM.match(params[0]))
            plain = params[1:] if has_self else params
            seed_idx = None
            for i, param in enumerate(plain):
                name, _, ty = param.partition(":")
                if SEED_WORD.search(name) and SEED_PARAM_TYPE.match(ty.strip()):
                    seed_idx = i
                    break
            yield m.group(1), seed_idx, len(plain), has_self

    def positional_calls(self, seeded):
        """Flag calls that pass a literal (or a constant that is not a seed
        constant of its own) where the callee takes a seed. `seeded` maps a
        callee name to its (seed index, parameter count, has self); a name
        whose definitions disagree is not in it."""
        if not seeded:
            return
        for m in seeded_call_pattern(seeded).finditer(self.joined):
            before = self.joined[max(0, m.start() - 8) : m.start()]
            if FN_KEYWORD_BEFORE.search(before):
                continue
            seed_idx, count, has_self = seeded[m.group(1)]
            qualifier = "." if before.endswith(".") else "::" if before.endswith("::") else ""
            args = split_args(self.joined, m.end())
            if args is None:
                continue
            if qualifier == "." and not has_self:
                continue
            if qualifier == "::" and RNG_TYPE_BEFORE.search(self.joined[max(0, m.start() - 40) : m.start() - 2]):
                continue  # a generator constructor: reported as kind `rng`
            shift = 1 if (has_self and qualifier == "::") else 0
            if len(args) != count + shift:
                continue
            arg = args[seed_idx + shift].strip()
            if re.fullmatch(INT, arg) or (POSITIONAL_CONST.match(arg) and "SEED" not in arg):
                self.add_extra(m.start(1), f"{m.group(1)}#{seed_idx + shift}={arg}", "positional")

    def hits(self):
        candidates = self.candidate_lines() | set(self.extra)
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
        for m in FOR_SEED.finditer(code):
            out.append((f"for {m.group(1)} in {m.group(2)}..", "default"))
        for m in RANGE_MAP_SEED.finditer(code):
            out.append((f"|{m.group(2)}| in ({m.group(1)}..)", "default"))
        out.extend(self.extra.get(idx, ()))
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


PY_SEED_CONST = re.compile(r"^(\w*SEED\w*)\s*(?::\s*[\w.\[\]]+\s*)?=\s*(" + INT + r")\s*(?:#.*)?$")
PY_SEED_CALL = re.compile(
    r"\b(manual_seed|np\.random\.seed|numpy\.random\.seed|random\.seed|default_rng|RandomState|random\.Random)"
    r"\(\s*(" + INT + r")\s*\)"
)
PY_SEED_ASSIGN = re.compile(
    r"(?<![\w.])(\w*seed\w*)\s*(?::\s*[\w.\[\], ]+?)?\s*(?<![=!<>])=(?!=)\s*(" + INT + r")(?![\w.])", re.I
)
PY_SEED_KEY = re.compile(r"""["'](\w*seed\w*)["']\s*:\s*(""" + INT + r")(?![\w.])", re.I)
PY_SEED_FALLBACK = re.compile(
    r"(?<![\w.])([\w.]*seed\w*)\s+(?:if\b[^\n]*?\bis\s+not\s+None\s+else|or)\s+(" + INT + r")(?![\w.])", re.I
)
PY_SEED_LOOKUP = re.compile(
    r"""(?:environ\.get|getenv|\.get)\(\s*["'](\w*seed\w*)["']\s*,\s*["']?(""" + INT + r""")["']?\s*\)""", re.I
)
PY_DEF = re.compile(r"^(\s*)(?:async\s+def|def|class)\s+(\w+)")


def python_views(text: str):
    """Per line: (code view with comments and strings blanked, view with only
    comments blanked). Triple-quoted strings span lines; escapes are skipped."""
    code, keep = [], []
    pos, n = 0, len(text)
    out_code, out_keep = [], []
    while pos < n:
        ch = text[pos]
        if ch == "#":
            end = text.find("\n", pos)
            end = n if end < 0 else end
            out_code.append(" " * (end - pos))
            out_keep.append(" " * (end - pos))
            pos = end
        elif ch in "'\"":
            quote = text[pos : pos + 3] if text[pos : pos + 3] in ("'\'\'", '"""') else ch
            end = pos + len(quote)
            while end < n and not text.startswith(quote, end):
                end += 2 if text[end] == "\\" else 1
                if len(quote) == 1 and end < n and text[end] == "\n":
                    break
            end = min(n, end + len(quote))
            body = text[pos:end]
            out_code.append(blank(body))
            out_keep.append(body)
            pos = end
        else:
            out_code.append(ch)
            out_keep.append(ch)
            pos += 1
    return "".join(out_code).split("\n"), "".join(out_keep).split("\n")


def python_seed_hits(text: str):
    """(line_no, symbol, kind) for module-level seed constants, seeding calls,
    seed parameters, fields, keywords and dict entries with an integer literal,
    and fallbacks that substitute a literal for a missing seed; the symbol is
    `<enclosing def or class>:<call>=<literal>`."""
    code_lines, keep_lines = python_views(text)
    scopes = []  # (indent, name) of the open `def`s and `class`es
    for no, (line, kept) in enumerate(zip(code_lines, keep_lines), 1):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        while scopes and scopes[-1][0] >= indent:
            scopes.pop()
        d = PY_DEF.match(line)
        if d:
            scopes.append((len(d.group(1)), d.group(2)))
        owner = scopes[-1][1] if scopes else "-"
        const = PY_SEED_CONST.match(line)
        if const:
            yield no, f"-:{const.group(1)}={const.group(2)}", "python-const"
        else:
            for m in PY_SEED_ASSIGN.finditer(line):
                yield no, f"{owner}:{m.group(1)}={m.group(2)}", "python-default"
        for m in PY_SEED_CALL.finditer(line):
            yield no, f"{owner}:{m.group(1)}={m.group(2)}", "python-rng"
        for m in PY_SEED_KEY.finditer(kept):
            yield no, f'{owner}:"{m.group(1)}"={m.group(2)}', "python-default"
        for m in PY_SEED_FALLBACK.finditer(line):
            yield no, f"{owner}:seed-fallback={m.group(2)}", "python-default"
        for m in PY_SEED_LOOKUP.finditer(kept):
            source = "environ" if "environ" in kept or "getenv" in kept else "get"
            yield no, f"{owner}:{source}.{m.group(1)}={m.group(2)}", "python-default"


MAKE_VAR = re.compile(r"^\s*(?:export\s+)?(\w*SEED\w*)\s*(?:\?=|:=|=)\s*(" + INT + r")\s*(?:#.*)?$", re.I)
MAKE_FLAG = re.compile(r"^\s*(\w+)\s*(?:\?=|:=|=)\s*.*?(--[\w-]*seed[\w-]*)[ =](" + INT + r")(?![\w.])", re.I)


SHELL_FLAG = re.compile(r"(?<![\w-])(--[\w-]*seed[\w-]*)[ =]\s*(" + INT + r")(?![\w.])", re.I)
SHELL_DEFAULT = re.compile(r"\$\{(\w*SEED\w*):-\s*(" + INT + r")\s*\}")
SHELL_ASSIGN = re.compile(r"(?<![\w-])(\w*SEED\w*)=(" + INT + r")(?![\w.])")
SHELL_SEQ = re.compile(r"\bseq\s+(" + INT + r"(?:\s+" + INT + r")*)")
SHELL_FUNC = re.compile(r"^\s*(?:function\s+)?(\w+)\s*\(\)\s*\{")
MAKE_TARGET = re.compile(r"^([\w./$()%-]+)\s*:(?!=)")
FENCE = re.compile(r"^\s*(```|~~~)")


def command_line_hits(line: str):
    """(field, kind-free) for each way one shell command line pins a seed."""
    for m in SHELL_FLAG.finditer(line):
        yield f"{m.group(1)}={m.group(2)}"
    for m in SHELL_DEFAULT.finditer(line):
        yield f"{m.group(1)}:-{m.group(2)}"
    for m in SHELL_ASSIGN.finditer(line):
        yield f"{m.group(1)}={m.group(2)}"


def strip_shell_comment(line: str) -> str:
    return "" if line.lstrip().startswith("#") else re.split(r"\s#\s", line, maxsplit=1)[0]


def shell_hits(text: str):
    """(line_no, symbol, kind) for flags, defaults and assignments that pin a
    seed in a shell script, and a `seq` that enumerates seeds."""
    lines = text.splitlines()
    func = "-"
    for no, raw in enumerate(lines, 1):
        fn = SHELL_FUNC.match(raw)
        if fn:
            func = fn.group(1)
        elif raw.startswith("}"):
            func = "-"
        line = strip_shell_comment(raw)
        for field in command_line_hits(line):
            yield no, f"{func}:{field}", "shell-seed"
        seq = SHELL_SEQ.search(line)
        if seq and any(re.search("seed", near, re.I) for near in lines[no - 1 : no + 3]):
            yield no, f"{func}:seq {seq.group(1)}", "shell-seed"


def makefile_hits(text: str):
    target = "-"
    for no, line in enumerate(text.splitlines(), 1):
        if line.startswith("\t"):
            for field in command_line_hits(strip_shell_comment(line.lstrip("\t"))):
                yield no, f"{target}:{field}", "make-flag"
            continue
        tm = MAKE_TARGET.match(line)
        if tm:
            target = tm.group(1)
        m = MAKE_VAR.match(line)
        if m:
            yield no, f"{m.group(1)}={m.group(2)}", "make-default"
            continue
        m = MAKE_FLAG.match(line)
        if m:
            yield no, f"{m.group(1)}:{m.group(2)}={m.group(3)}", "make-flag"


def markdown_hits(text: str):
    """(line_no, symbol, kind) for a `--seed N` inside a fenced block that is
    not explained as the way to reproduce something: the word `reproduce` on
    the line, a neighbouring line or the prose line above the fence."""
    lines = text.splitlines()
    fence_prose = ""
    in_fence = False
    last_prose = ""
    for no, raw in enumerate(lines, 1):
        if FENCE.match(raw):
            in_fence = not in_fence
            fence_prose = last_prose if in_fence else ""
            continue
        if not in_fence:
            if raw.strip():
                last_prose = raw
            continue
        near = " ".join([fence_prose] + lines[max(0, no - 2) : no + 1])
        if re.search("reproduc", near, re.I):
            continue
        for m in SHELL_FLAG.finditer(strip_shell_comment(raw)):
            yield no, f"-:{m.group(1)}={m.group(2)}", "doc-seed"


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


def crate_of(rel: str) -> str:
    """The unit whose function names are resolved together: a crate, a sample
    directory (`samples/<area>/<name>`) or a top-level directory."""
    parts = rel.split("/")
    if parts[0] == "crates" and len(parts) > 2:
        return "/".join(parts[:2])
    if parts[0] == "samples" and len(parts) > 3:
        return "/".join(parts[:3])
    return parts[0]


def seeded_callees(files: dict) -> dict:
    """crate -> {callee name: (seed index, parameter count, has self)} for every
    function that takes a seed and whose definitions in the crate agree."""
    by_name = {}
    for rel, f in files.items():
        for name, seed_idx, count, has_self in f.signatures():
            by_name.setdefault((crate_of(rel), name), set()).add((seed_idx, count, has_self))
    seeded = {}
    for (crate, name), sigs in by_name.items():
        if len(sigs) == 1:
            (sig,) = sigs
            if sig[0] is not None:
                seeded.setdefault(crate, {})[name] = sig
    return seeded


def scan(root: Path, positional_tests: bool = False):
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
    parsed = {rel: RustFile(rel, text, path_is_test(rel) or rel in test_files) for rel, text in rust.items()}
    seeded = seeded_callees(parsed)
    rows = []
    for rel, f in parsed.items():
        f.hidden_defaults()
        f.positional_calls(seeded.get(crate_of(rel), {}))
        lowered = rust[rel].lower()
        if not f.extra and not any(needle in lowered for needle in CANDIDATE_NEEDLES):
            continue
        for line_no, enclosing, (field, kind), in_test in f.hits():
            if kind == "positional" and in_test and not positional_tests:
                continue
            rows.append((rel, f"{enclosing}:{field}", kind, "test" if in_test else "prod", line_no, justified(f.raw, line_no)))
    for rel in files:
        name = Path(rel).name
        context = "test" if path_is_test(rel) else "prod"
        if rel.endswith(".py"):
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            raw = text.splitlines()
            for no, flag, lit in python_hits(text):
                rows.append((rel, f"-:{flag}={lit}", "python-default", context, no, justified(raw, no)))
            for no, symbol, kind in python_seed_hits(text):
                rows.append((rel, symbol, kind, context, no, justified(raw, no)))
        elif name in ("Makefile", "makefile") or rel.endswith(".mk"):
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            raw = text.splitlines()
            for no, symbol, kind in makefile_hits(text):
                rows.append((rel, f"-:{symbol}", kind, "prod", no, justified(raw, no)))
        elif rel.endswith((".sh", ".bats")):
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            raw = text.splitlines()
            for no, symbol, kind in shell_hits(text):
                rows.append((rel, symbol, kind, "test" if rel.endswith(".bats") else context, no, justified(raw, no)))
        elif rel.endswith(".md"):
            text = (root / rel).read_text(encoding="utf-8", errors="replace")
            raw = text.splitlines()
            for no, symbol, kind in markdown_hits(text):
                rows.append((rel, symbol, kind, "prod", no, justified(raw, no)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--lines", action="store_true", help="append line numbers (diagnostics)")
    ap.add_argument(
        "--positional-tests",
        action="store_true",
        help="also list literals passed positionally as a seed in test code (kind `positional`, context test)",
    )
    args = ap.parse_args()
    seen = {}
    for rel, symbol, kind, context, line_no, ok in scan(Path(args.root), args.positional_tests):
        entry = seen.setdefault((rel, symbol, kind, context), [True, []])
        entry[0] &= ok
        entry[1].append(line_no)
    for (rel, symbol, kind, context), (ok, lines) in sorted(seen.items()):
        extra = "\t" + ",".join(map(str, lines)) if args.lines else ""
        sys.stdout.write(f"{rel}\t{symbol}\t{kind}\t{context}\t{'reasoned' if ok else 'bare'}{extra}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
