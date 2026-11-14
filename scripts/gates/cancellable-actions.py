# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements solutions for long-running inference and
# training jobs that must stop when their caller goes away. If your team needs
# expertise in cooperative cancellation across model serving stacks then you
# can procure our services by sending an email to info@swedishembedded.com.
"""Every action spec built in production code declares `.cancellable()`.

An action is built as one `ActionSpec::new(..)` builder expression. This
finds each one outside test code and requires `.cancellable()` among the
calls chained onto it (up to the end of the expression: the `;`, `,` or
closing bracket at the depth it started), or on a builder it is handed to in
the same file. A spec built in a helper that its callers extend is found by
following the helper's name (`fn x() -> ActionSpec`): a call of `x()`
chained with `.cancellable()` counts for it, and so does `.cancellable()`
inside `x` itself.

The behavioural half of the rule is `crates/catalog/tests/it/cancellation.rs`:
every listed action declares the bit, and every one that can be built
without weights refuses a token fired before the call.

Usage: cancellable-actions.py ROOT
"""
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'

# Crates whose sources construct specs that are not served actions: the
# capability crate defines ActionSpec and documents it, capability-mock's
# declarative mock takes the caller's own spec.
EXEMPT = ('crates/capability/src/', 'crates/capability-mock/src/')


def blank(segment):
    """`segment` emptied but for its newlines, so line numbers survive."""
    return '\n' * segment.count('\n')


def strip_tests(text):
    """The file without its `#[cfg(test)]` items (a module or a function)."""
    out = []
    i = 0
    while True:
        j = text.find('#[cfg(test)]', i)
        if j < 0:
            out.append(text[i:])
            return ''.join(out)
        out.append(text[i:j])
        k = text.find('{', j)
        semi = text.find(';', j)
        if k < 0 or (0 <= semi < k):
            i = semi + 1 if semi >= 0 else len(text)
            out.append(blank(text[j:i]))
            continue
        depth = 0
        m = k
        while m < len(text):
            c = text[m]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            m += 1
        out.append(blank(text[j:m + 1]))
        i = m + 1


def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', lambda m: blank(m.group(0)), text, flags=re.S)
    return re.sub(r'//[^\n]*', '', text)


def strip_strings(text):
    # Raw strings first, then ordinary ones; contents can hold brackets.
    text = re.sub(r'r(#+)".*?"\1', lambda m: '""' + blank(m.group(0)), text, flags=re.S)
    return re.sub(r'"(?:\\.|[^"\\])*"', lambda m: '""' + blank(m.group(0)), text, flags=re.S)


def expression_end(text, start):
    """Index just past the builder expression that starts at `start`."""
    depth = 0
    i = start
    while i < len(text):
        c = text[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            if depth == 0:
                return i
            depth -= 1
        elif c in ';,' and depth == 0:
            return i
        i += 1
    return i


def line_of(text, idx):
    return text.count('\n', 0, idx) + 1


def main():
    bad = []
    for crate in sorted(os.listdir(os.path.join(ROOT, 'crates'))):
        src = os.path.join(ROOT, 'crates', crate, 'src')
        for dirpath, _, files in os.walk(src):
            for f in files:
                if not f.endswith('.rs'):
                    continue
                path = os.path.join(dirpath, f)
                rel = os.path.relpath(path, ROOT)
                if rel.startswith(EXEMPT) or '/tests/' in rel or f == 'tests.rs':
                    continue
                raw = open(path, encoding='utf-8').read()
                text = strip_strings(strip_comments(strip_tests(raw)))
                # helper fns returning a spec: their callers may declare for them
                helpers = {}
                for m in re.finditer(r'fn\s+(\w+)\s*\([^)]*\)\s*->\s*ActionSpec\s*\{', text):
                    body_end = expression_end(text, m.end())
                    helpers[m.group(1)] = (m.end(), body_end)
                for m in re.finditer(r'ActionSpec::new\s*\(', text):
                    end = expression_end(text, m.start())
                    if '.cancellable()' in text[m.start():end]:
                        continue
                    owner = next((name for name, (b, e) in helpers.items() if b <= m.start() < e), None)
                    if owner is not None:
                        b, e = helpers[owner]
                        if '.cancellable()' in text[b:e]:
                            continue
                        calls = [c for c in re.finditer(r'\b' + owner + r'\s*\(', text) if not (b <= c.start() < e)]
                        if calls and all('.cancellable()' in text[c.start():expression_end(text, c.start())] for c in calls):
                            continue
                    bad.append(f'{rel}:{line_of(text, m.start())}')
    if bad:
        print('check-cancellable-actions: these action specs do not declare `.cancellable()`:')
        for b in bad:
            print(f'  {b}')
        print('  An action checks its invocation\'s CancelToken on entry, before any weight is read or lock')
        print('  taken, and at least once per bounded unit of work, then declares it with .cancellable()')
        print('  (crates/capability/src/lib.rs, ActionSpec::cancellable; .agents/roadmap/cancellation.md).')
        return 1
    print('check-cancellable-actions: OK (every production action spec declares .cancellable())')
    return 0


sys.exit(main())
