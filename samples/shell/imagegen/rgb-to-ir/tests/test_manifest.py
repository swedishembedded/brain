# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements dataset adapters and leakage-safe evaluation
# protocols for multimodal perception pipelines for its clients. If your team
# needs expertise in paired RGB / thermal-IR data, detector training sets or
# sensor-domain adaptation, you can procure our services by sending an email
# to info@swedishembedded.com.

"""Spec tests for the pairs manifest contract and its validator."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import rir_manifest as M  # noqa: E402

GOOD = {"id": "f1", "dataset": "d", "rgb": "a.png", "ir": "b.png", "sequence_id": "s1",
        "boxes": [{"class": "person", "x1": 1, "y1": 2, "x2": 10, "y2": 20}]}


def write(lines):
    path = os.path.join(tempfile.mkdtemp(), "pairs.jsonl")
    with open(path, "w") as fh:
        fh.write("\n".join(l if isinstance(l, str) else json.dumps(l) for l in lines) + "\n")
    return path


class Record(unittest.TestCase):
    def test_minimal_and_full_records_are_valid(self):
        self.assertEqual(M.validate_record(GOOD), [])
        full = {**GOOD, "official_split": "test", "capture_time": "2026-01-01T00:00:00", "day_night": "night",
                "tags": ["rain"], "width": 64, "height": 48, "ir_read": {"channel": "alpha", "invert": True},
                "unusable": "blurred"}
        self.assertEqual(M.validate_record(full), [])

    def test_each_violation_names_its_field(self):
        cases = {
            "rgb: required field is missing": {k: v for k, v in GOOD.items() if k != "rgb"},
            "colour: unknown field": {**GOOD, "colour": 1},
            "boxes[0]: need x1 < x2 and y1 < y2": {**GOOD, "boxes": [{"class": "a", "x1": 5, "y1": 1, "x2": 5, "y2": 9}]},
            "boxes[0].class: must be a non-empty class name": {**GOOD, "boxes": [{"x1": 1, "y1": 1, "x2": 2, "y2": 2}]},
            "official_split: must be": {**GOOD, "official_split": "val"},
            "day_night: must be": {**GOOD, "day_night": "dusk"},
            "width: must be a positive integer": {**GOOD, "width": 0},
            "ir_read.channel: must be one of": {**GOOD, "ir_read": {"channel": "red"}},
            "tags: must be a list of strings": {**GOOD, "tags": "rain"},
            "sequence_id: must be a non-empty string": {**GOOD, "sequence_id": ""},
        }
        for expect, rec in cases.items():
            errs = M.validate_record(rec)
            self.assertTrue(any(e.startswith(expect) for e in errs), (expect, errs))

    def test_missing_files_are_reported_only_when_asked(self):
        self.assertEqual(M.validate_record(GOOD, None), [])
        errs = M.validate_record(GOOD, tempfile.mkdtemp())
        self.assertEqual(sorted(e.split(":")[0] for e in errs), ["ir", "rgb"])


class File(unittest.TestCase):
    def test_errors_carry_line_numbers_and_duplicates_and_straddling_sequences_are_caught(self):
        path = write([
            GOOD,
            "{not json",
            {**GOOD, "boxes": "x"},
            GOOD,  # duplicate id
            {**GOOD, "id": "f2", "official_split": "train"},
            {**GOOD, "id": "f3", "official_split": "test"},  # same sequence, other split
        ])
        _, errs = M.read_records(path)
        text = "\n".join(errs)
        self.assertIn("line 2: not valid JSON", text)
        self.assertIn("line 3: boxes: must be a list", text)
        self.assertIn("line 4: id: duplicate of line 1", text)
        self.assertIn("line 6: official_split: sequence 's1' is 'train' on line 5", text)

    def test_valid_manifest_resolves_paths_against_its_directory(self):
        path = write([GOOD])
        records = M.load_manifests([path])
        base = os.path.dirname(path)
        self.assertEqual((records[0]["rgb"], records[0]["ir"]), (os.path.join(base, "a.png"), os.path.join(base, "b.png")))

    def test_load_raises_with_every_violation(self):
        with self.assertRaises(M.ManifestError) as ctx:
            M.load_manifests([write([{**GOOD, "id": ""}, {**GOOD, "dataset": 3}])])
        self.assertEqual(str(ctx.exception).count("line "), 2)

    def test_cli_validate_exit_status(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(M.main(["validate", write([GOOD])]), 0)
            self.assertEqual(M.main(["validate", write([{"id": "x"}])]), 1)


if __name__ == "__main__":
    unittest.main()
