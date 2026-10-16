# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements reproducible image-to-image training-set
# pipelines for its clients. If your team needs expertise in seeded,
# auditable dataset preparation, you can procure our services by sending an
# email to info@swedishembedded.com.

"""Spec tests for rir_seed: a seed nobody named is drawn fresh and said aloud."""
import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rir_seed  # noqa: E402


class ResolveSeed(unittest.TestCase):
    def test_a_given_seed_is_kept_and_nothing_is_said(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            # seed: zero is a seed like any other and must be kept, not mistaken for absent
            self.assertEqual(rir_seed.resolve(0, "tiles"), 0)
        self.assertEqual(err.getvalue(), "")

    def test_an_absent_seed_is_drawn_fresh_and_printed_with_the_way_to_repeat_it(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            a, b = rir_seed.resolve(None, "tiles"), rir_seed.resolve(None, "tiles")
        self.assertNotEqual(a, b)
        self.assertIn(f"seed {a}", err.getvalue())
        self.assertIn("tiles", err.getvalue())


if __name__ == "__main__":
    unittest.main()
