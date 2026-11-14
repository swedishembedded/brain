#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements release pipelines whose green result means the
# tests really ran, for its clients. If your team needs expertise in CI gates
# for machine-learning and embedded software, you can procure our services by
# sending an email to info@swedishembedded.com.

"""Turns the output of a test run into a verdict that cannot be green by skipping.

`cargo test` reports a skipped comparison as a pass and `bats` exits 0 when
every test skips, so an exit code alone cannot tell "every check matched" from
"no check ran". This script reads the run's own output and fails a required
suite that reported no tests, a failure, or a skip.

    suite-report.py check  --name NAME --format cargo|tap --input FILE|- --out DIR
                           [--min-tests N] [--max-skips N] [--fail-on-ignored]
    suite-report.py matrix --results DIR --require NAME[,NAME...] [--summary FILE]

`check` writes DIR/NAME.json and exits 1 unless the suite passed. `matrix`
reads every DIR/*.json, prints a pass/fail/skip/not-run table as Markdown (and
appends it to --summary, the job summary of a workflow), and exits 1 unless
every required suite passed: a required suite with no result is `not-run`.

What counts as a skip:
  tap    a test line carrying `# skip` (what `skip` in a bats test prints)
  cargo  a `SKIP:` line (what brain_testutil::skip prints for an absent
         fixture or device); `ignored` tests are counted apart, because an
         `#[ignore = "..."]` is declared in the source and names its reason,
         and fail the suite only under --fail-on-ignored
"""

import argparse
import json
import re
import sys
from pathlib import Path

PASS, FAIL, SKIP, EMPTY, NOT_RUN = "pass", "fail", "skip", "empty", "not-run"

CARGO_RESULT = re.compile(r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;")
CARGO_SKIP = re.compile(r"^\s*SKIP:")
TAP_PLAN = re.compile(r"^1\.\.(\d+)")
TAP_TEST = re.compile(r"^(not ok|ok)\s+\d+\b(.*)$")


def parse_cargo(lines):
    passed = failed = ignored = skipped = 0
    for line in lines:
        m = CARGO_RESULT.match(line)
        if m:
            passed += int(m.group(1))
            failed += int(m.group(2))
            ignored += int(m.group(3))
        elif CARGO_SKIP.match(line):
            skipped += 1
    return {"passed": passed, "failed": failed, "skipped": skipped, "ignored": ignored, "problems": []}


def parse_tap(lines):
    passed = failed = skipped = 0
    planned = None
    for line in lines:
        plan = TAP_PLAN.match(line)
        if plan:
            # A run of several files prints one plan each: the plans add up.
            planned = (planned or 0) + int(plan.group(1))
            continue
        m = TAP_TEST.match(line)
        if not m:
            continue
        if m.group(1) == "not ok":
            failed += 1
        elif re.search(r"#\s*skip", m.group(2), re.IGNORECASE):
            skipped += 1
        else:
            passed += 1
    problems = []
    seen = passed + failed + skipped
    if planned is not None and planned != seen:
        problems.append(f"the plan announced {planned} tests and {seen} reported")
    return {"passed": passed, "failed": failed, "skipped": skipped, "ignored": 0, "problems": problems}


def verdict(counts, min_tests, max_skips, fail_on_ignored):
    """The status of a suite and the reasons it is not a pass."""
    reasons = list(counts["problems"])
    ran = counts["passed"] + counts["failed"]
    reported = ran + counts["skipped"]
    if counts["failed"]:
        reasons.append(f"{counts['failed']} failed")
    if reported < min_tests:
        reasons.append(f"{reported} tests reported, at least {min_tests} required")
    if counts["skipped"] > max_skips:
        reasons.append(f"{counts['skipped']} skipped, at most {max_skips} allowed")
    if fail_on_ignored and counts["ignored"]:
        reasons.append(f"{counts['ignored']} ignored")
    if counts["failed"] or counts["problems"]:
        return FAIL, reasons
    if reported < min_tests:
        return EMPTY, reasons
    if counts["skipped"] > max_skips:
        return SKIP, reasons
    return (FAIL if reasons else PASS), reasons


def check(args):
    stream = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8", errors="replace")
    with stream:
        lines = stream.read().splitlines()
    counts = (parse_cargo if args.format == "cargo" else parse_tap)(lines)
    status, reasons = verdict(counts, args.min_tests, args.max_skips, args.fail_on_ignored)
    result = {"name": args.name, "status": status, **{k: counts[k] for k in ("passed", "failed", "skipped", "ignored")}, "reasons": reasons}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{args.name}: {status} ({counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} skipped, {counts['ignored']} ignored)" + "".join(f"\n  - {r}" for r in reasons))
    return 0 if status == PASS else 1


def matrix(args):
    required = [n for n in args.require.split(",") if n]
    found = {}
    for path in sorted(Path(args.results).glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        found[result["name"]] = result
    rows = ["| suite | status | pass | fail | skip | ignored | note |", "|---|---|---|---|---|---|---|"]
    ok = True
    for name in required + sorted(set(found) - set(required)):
        r = found.get(name)
        if r is None:
            rows.append(f"| {name} | {NOT_RUN} | - | - | - | - | required, no result |")
            ok = False
            continue
        if name in required and r["status"] != PASS:
            ok = False
        rows.append(f"| {name} | {r['status']} | {r['passed']} | {r['failed']} | {r['skipped']} | {r['ignored']} | {'; '.join(r['reasons'])} |")
    table = "\n".join(rows)
    print(table)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as summary:
            summary.write(f"### {args.title}\n\n{table}\n\n")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="verdict of one suite's output")
    c.add_argument("--name", required=True)
    c.add_argument("--format", choices=("cargo", "tap"), required=True)
    c.add_argument("--input", required=True, help="the run's output, or - for stdin")
    c.add_argument("--out", required=True, help="directory the NAME.json result goes to")
    c.add_argument("--min-tests", type=int, default=1)
    c.add_argument("--max-skips", type=int, default=0)
    c.add_argument("--fail-on-ignored", action="store_true")
    c.set_defaults(run=check)
    m = sub.add_parser("matrix", help="the pass/fail/skip/not-run table of every suite")
    m.add_argument("--results", required=True, help="directory of NAME.json results")
    m.add_argument("--require", required=True, help="comma-separated suites that must have passed")
    m.add_argument("--summary", help="file the table is appended to (GITHUB_STEP_SUMMARY)")
    m.add_argument("--title", default="Test matrix")
    m.set_defaults(run=matrix)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
